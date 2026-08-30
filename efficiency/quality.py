"""Generate held-out outputs and compute table-facing PSNR/SSIM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .benchmark import build_adapter, parser as benchmark_parser, system_metadata
from .base import load_dataset_records
from .datasets import load_task_manifest, select_indices
from .metrics import release_inference_result, seed_everything


RESAMPLING = getattr(Image, "Resampling", Image)


def psnr_ssim(reference: Image.Image, prediction: Image.Image) -> tuple[float, float]:
    """Match skimage's default RGB PSNR and channel-averaged 7x7 SSIM."""
    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float64)
    prediction_array = np.asarray(prediction.convert("RGB"), dtype=np.float64)
    if reference_array.shape != prediction_array.shape:
        raise ValueError(
            f"Metric images have different shapes: {reference_array.shape} and "
            f"{prediction_array.shape}"
        )

    error = reference_array - prediction_array
    mse = float(np.mean(error * error, dtype=np.float64))
    psnr = math.inf if mse == 0 else 10.0 * math.log10((255.0**2) / mse)

    # skimage structural_similarity defaults: win_size=7, gaussian_weights=False,
    # use_sample_covariance=True, K1=.01, K2=.03, channel_axis=-1.
    reference_tensor = torch.from_numpy(reference_array).permute(2, 0, 1).unsqueeze(0)
    prediction_tensor = torch.from_numpy(prediction_array).permute(2, 0, 1).unsqueeze(0)
    if min(reference_tensor.shape[-2:]) < 7:
        raise ValueError("SSIM requires image dimensions of at least 7 pixels")
    mean_reference = F.avg_pool2d(reference_tensor, kernel_size=7, stride=1)
    mean_prediction = F.avg_pool2d(prediction_tensor, kernel_size=7, stride=1)
    covariance_normalization = 49.0 / 48.0
    variance_reference = covariance_normalization * (
        F.avg_pool2d(reference_tensor * reference_tensor, 7, stride=1)
        - mean_reference * mean_reference
    )
    variance_prediction = covariance_normalization * (
        F.avg_pool2d(prediction_tensor * prediction_tensor, 7, stride=1)
        - mean_prediction * mean_prediction
    )
    covariance = covariance_normalization * (
        F.avg_pool2d(reference_tensor * prediction_tensor, 7, stride=1)
        - mean_reference * mean_prediction
    )
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2 * mean_reference * mean_prediction + c1) * (
        2 * covariance + c2
    )
    denominator = (
        mean_reference * mean_reference + mean_prediction * mean_prediction + c1
    ) * (variance_reference + variance_prediction + c2)
    ssim = float((numerator / denominator).mean().item())
    return psnr, ssim


def _controlled_image(image: Image.Image, resolution: int) -> Image.Image:
    rgb = image.convert("RGB").copy()
    if rgb.size != (resolution, resolution):
        resized = rgb.resize((resolution, resolution), RESAMPLING.BICUBIC)
        rgb.close()
        rgb = resized
    return rgb


def _selected_indices(count: int, limit: int, seed: int) -> list[int]:
    if limit == -1:
        return list(range(count))
    if limit < 1:
        raise ValueError("--max-samples must be -1 or a positive integer")
    return select_indices(count, limit, seed)


def _task_seed(base: int, task_name: str) -> int:
    return base + sum((index + 1) * ord(char) for index, char in enumerate(task_name))


def _heldout_record_indices(
    records: list[dict[str, Any]], demo_input: str, demo_output: str
) -> tuple[list[int], list[int]]:
    heldout = []
    excluded = []
    for index, record in enumerate(records):
        if (
            record.get("image_path") == demo_input
            or record.get("target_path") == demo_output
        ):
            excluded.append(index)
        else:
            heldout.append(index)
    if not heldout:
        raise ValueError("The demonstration pair leaves no held-out query records")
    return heldout, excluded


def _fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_scalar(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return _json_scalar(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _load_completed(path: Path, fingerprint: str, resume: bool) -> dict[int, dict]:
    if not path.exists() or not resume:
        return {}
    completed = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_fingerprint") != fingerprint:
                raise ValueError(
                    f"Cannot resume {path}: line {line_number} belongs to a different "
                    "model/task/metric configuration"
                )
            completed[int(row["sample_index"])] = row
    return completed


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty quality result")
    psnr_values = [float(row["psnr"]) for row in rows]
    ssim_values = [float(row["ssim"]) for row in rows]
    return {
        "count": len(rows),
        "psnr_mean": statistics.fmean(psnr_values),
        "psnr_std": statistics.pstdev(psnr_values),
        "ssim_mean": statistics.fmean(ssim_values),
        "ssim_std": statistics.pstdev(ssim_values),
    }


def run_quality(args) -> dict[str, Any]:
    if args.resolution is not None and args.resolution < 7:
        raise ValueError("--resolution must be at least 7 for SSIM")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        args.task_manifest or Path(__file__).with_name("tasks.json")
    ).expanduser().resolve()
    manifest, tasks = load_task_manifest(manifest_path)
    if args.tasks:
        requested = set(args.tasks)
        tasks = [task for task in tasks if task.name in requested]
        missing = requested - {task.name for task in tasks}
        if missing:
            raise ValueError(f"Unknown tasks: {sorted(missing)}")
    if not tasks:
        raise ValueError("No tasks selected")
    if any(task.is_cross_task for task in tasks):
        raise ValueError(
            "Table-facing Prompt Diffusion/InstructDiffusion quality requires the "
            "same-task paired-image manifest"
        )

    data_root = args.data_root or project_root / manifest["data_root"]
    args.data_root = data_root.expanduser().resolve()
    args.resolution = args.resolution or int(
        manifest["controlled_protocol"]["resolution"]
    )
    args.shape_policy = "fixed-square"
    args.sample_index = 0

    first = tasks[0]
    args.dataset_json = args.data_root / first.eval_json
    args.demo_input = first.demo_input
    args.demo_output = first.demo_output
    args.text_prompt = first.text_prompt

    output_root = args.output_dir.expanduser()
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()
    records_root = output_root / "records"
    images_root = output_root / "images"
    records_root.mkdir(parents=True, exist_ok=True)
    if not args.no_save_outputs:
        images_root.mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(args, project_root)
    print(f"Loading {adapter.name} for quality evaluation...")
    adapter.setup()
    conditions = list(args.conditions or adapter.conditions)
    invalid = set(conditions) - set(adapter.conditions)
    if invalid:
        raise ValueError(f"Unsupported conditions: {sorted(invalid)}")

    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": "image_quality_suite",
        "adapter": adapter.name,
        "protocol": adapter.protocol,
        "task_manifest": str(manifest_path),
        "benchmark_family": manifest.get("benchmark_family"),
        "system": system_metadata(project_root),
        "metric_protocol": {
            "resolution": args.resolution,
            "color_space": "RGB",
            "range": "uint8 [0,255]",
            "resize": "bicubic prediction and ground truth to the controlled square",
            "psnr": "10*log10(255^2/MSE), RGB pixels",
            "ssim": (
                "channel-averaged skimage-compatible SSIM: 7x7 uniform window, "
                "sample covariance, K1=0.01, K2=0.03"
            ),
            "aggregation": "arithmetic mean over distinct held-out query records",
            "timing_separation": (
                "This is an untimed generation pass; saving and metrics are never included "
                "in efficiency latency."
            ),
        },
        "max_samples": args.max_samples,
        "sampling_seed": args.sampling_seed,
        "conditions": [],
    }
    source_cache: dict[Path, list[dict[str, Any]]] = {}

    try:
        for condition in conditions:
            adapter.prepare_condition(condition)
            condition_result = {"condition": condition, "tasks": []}
            try:
                for task in tasks:
                    json_path = (args.data_root / task.eval_json).resolve()
                    source_records = source_cache.get(json_path)
                    if source_records is None:
                        source_records = load_dataset_records(json_path)
                        source_cache[json_path] = source_records
                    assert task.demo_input is not None
                    assert task.demo_output is not None
                    heldout_source_indices, excluded_demo_indices = (
                        _heldout_record_indices(
                            source_records, task.demo_input, task.demo_output
                        )
                    )
                    adapter.configure_samples(
                        json_path,
                        demo_input=task.demo_input,
                        demo_output=task.demo_output,
                        record_indices=heldout_source_indices,
                    )
                    if hasattr(adapter, "text_prompt"):
                        adapter.text_prompt = task.text_prompt
                    selected = _selected_indices(
                        adapter.sample_count(),
                        args.max_samples,
                        _task_seed(args.sampling_seed, task.name),
                    )
                    selected_source_indices = [
                        heldout_source_indices[index] for index in selected
                    ]
                    configuration = {
                        "adapter": adapter.name,
                        "condition": condition,
                        "task": task.name,
                        "dataset_json": str(json_path),
                        "selected_indices": selected,
                        "selected_source_indices": selected_source_indices,
                        "excluded_demo_record_indices": excluded_demo_indices,
                        "model_id": getattr(adapter, "model_id", None),
                        "checkpoint": getattr(adapter, "checkpoint", None),
                        "steps": getattr(adapter, "steps", None),
                        "seed": getattr(adapter, "seed", None),
                        "resolution": args.resolution,
                        "text_prompt": task.text_prompt,
                        "dtype": args.dtype,
                        "config": args.config,
                        "cfg_text": args.cfg_text,
                        "cfg_image": args.cfg_image,
                        "shape_policy": args.shape_policy,
                        "painter_task": args.painter_task,
                        "painter_include_script_loss": (
                            args.painter_include_script_loss
                        ),
                        "demo_input": task.demo_input,
                        "demo_output": task.demo_output,
                    }
                    fingerprint = _fingerprint(configuration)
                    record_path = records_root / (
                        f"{adapter.name}__{condition}__{task.name}.jsonl"
                    )
                    completed = _load_completed(record_path, fingerprint, args.resume)
                    mode = "a" if args.resume else "w"
                    task_image_root = images_root / condition / task.name
                    if not args.no_save_outputs:
                        task_image_root.mkdir(parents=True, exist_ok=True)
                        for index, row in list(completed.items()):
                            saved = row.get("output_path")
                            if not saved or not Path(saved).is_file():
                                del completed[index]
                    print(
                        f"Quality: condition={condition}, task={task.name}, "
                        f"queries={len(selected)}, resumed={len(completed)}"
                    )
                    with record_path.open(mode, encoding="utf-8", buffering=1) as handle:
                        for sample_index in selected:
                            if sample_index in completed:
                                continue
                            adapter.select_sample(sample_index)
                            sample = adapter.sample
                            if sample is None or not sample.task_b_output:
                                raise ValueError(
                                    f"Task {task.name} sample {sample_index} has no target"
                                )
                            seed_everything(getattr(adapter, "seed", args.seed))
                            result = adapter.run(condition)
                            prediction = None
                            target = None
                            try:
                                prediction = _controlled_image(
                                    result.output, args.resolution
                                )
                                target_path = args.data_root / sample.task_b_output
                                with Image.open(target_path) as target_source:
                                    target = _controlled_image(
                                        target_source, args.resolution
                                    )
                                psnr, ssim = psnr_ssim(target, prediction)
                                output_path = None
                                if not args.no_save_outputs:
                                    output_path = task_image_root / f"{sample_index:06d}.png"
                                    prediction.save(output_path, format="PNG")
                                row = {
                                    "run_fingerprint": fingerprint,
                                    "sample_index": sample_index,
                                    "source_record_index": heldout_source_indices[
                                        sample_index
                                    ],
                                    "sample": sample.as_dict(),
                                    "output_path": (
                                        str(output_path) if output_path else None
                                    ),
                                    "psnr": _json_scalar(psnr),
                                    "ssim": _json_scalar(ssim),
                                }
                                handle.write(
                                    json.dumps(row, allow_nan=False) + "\n"
                                )
                                handle.flush()
                                os.fsync(handle.fileno())
                                completed[sample_index] = row
                            finally:
                                if target is not None:
                                    target.close()
                                if prediction is not None:
                                    prediction.close()
                                release_inference_result(result)
                    rows = [completed[index] for index in selected]
                    condition_result["tasks"].append(
                        {
                            "task": task.name,
                            "dataset_json": str(json_path),
                            "source_records": len(source_records),
                            "heldout_records": len(heldout_source_indices),
                            "excluded_demo_record_indices": excluded_demo_indices,
                            "selected_indices": selected,
                            "selected_source_indices": selected_source_indices,
                            "demo_input": task.demo_input,
                            "demo_output": task.demo_output,
                            "text_prompt": task.text_prompt,
                            "records_jsonl": str(record_path),
                            **_summary(rows),
                        }
                    )
                document["conditions"].append(condition_result)
            finally:
                adapter.release_condition(condition)
    finally:
        adapter.close()
    return document


def write_quality(document: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    adapter = document["adapter"]
    stamped = output_dir / f"{adapter}_quality_{stamp}.json"
    latest = output_dir / f"{adapter}_quality_latest.json"
    payload = json.dumps(_json_safe(document), indent=2, allow_nan=False)
    stamped.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")
    rows = []
    for condition in document["conditions"]:
        for task in condition["tasks"]:
            rows.append(
                {
                    "adapter": adapter,
                    "condition": condition["condition"],
                    "task": task["task"],
                    "count": task["count"],
                    "psnr_mean": task["psnr_mean"],
                    "psnr_std": task["psnr_std"],
                    "ssim_mean": task["ssim_mean"],
                    "ssim_std": task["ssim_std"],
                }
            )
    with (output_dir / f"{adapter}_quality.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stamped


def parser() -> argparse.ArgumentParser:
    result = benchmark_parser()
    result.description = __doc__
    for action in result._actions:
        if action.dest == "adapter":
            action.choices = [
                "painter",
                "prompt-diffusion",
                "instruct-diffusion",
            ]
        elif action.dest in {
            "dataset_json",
            "demo_input",
            "demo_output",
            "warmup",
            "repeats",
            "sample_index",
            "profile_flops",
            "flops_top_k",
            "parameters_only",
            "shape_policy",
        }:
            action.help = argparse.SUPPRESS
    result.add_argument("--task-manifest", type=Path)
    result.add_argument("--tasks", nargs="+")
    result.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Held-out queries per task; -1 uses the entire declared split",
    )
    result.add_argument("--sampling-seed", type=int, default=2026)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--no-save-outputs", action="store_true")
    result.set_defaults(
        data_root=None,
        resolution=448,
        shape_policy="fixed-square",
        output_dir=Path("efficiency/quality"),
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    document = run_quality(args)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    path = write_quality(document, output_dir.resolve())
    print(f"Wrote {path}")
    print(f"Wrote {output_dir / (document['adapter'] + '_quality.csv')}")


if __name__ == "__main__":
    main()
