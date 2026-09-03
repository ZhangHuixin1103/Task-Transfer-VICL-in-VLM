from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


FIELDS = [
    "adapter",
    "condition",
    "task",
    "benchmark_family",
    "protocol",
    "resolution",
    "measured_queries",
    "sampling_steps",
    "total_parameters",
    "stored_parameter_elements",
    "trained_parameters",
    "parameter_storage_gib",
    "quantized_parameter_tensors",
    "meta_parameter_tensors",
    "parameter_storage_complete",
    "formula_flops",
    "opaque_module_flops",
    "estimated_flops",
    "estimated_tflops",
    "estimated_flops_min",
    "estimated_flops_max",
    "flops_profile_samples",
    "flop_confidence",
    "unsupported_nontrivial_operators",
    "incremental_parameters_vs_fixed",
    "incremental_flops_vs_fixed",
    "latency_mean_s",
    "latency_median_s",
    "latency_std_s",
    "latency_overhead_vs_fixed_s",
    "latency_ratio_vs_fixed",
    "throughput_images_s",
    "prompt_generation_mean_s",
    "image_generation_mean_s",
    "peak_cuda_memory_gib",
    "peak_cuda_memory_sum_gib",
    "peak_cuda_reserved_sum_gib",
    "resident_cuda_memory_sum_gib",
    "device",
    "status",
]


def _stage_mean(condition: Dict[str, Any], names: Sequence[str]) -> float | None:
    stages = (condition.get("latency") or {}).get("stages", {})
    values = [stages[name]["mean_s"] for name in names if name in stages]
    return sum(values) if values else None


def _flatten_single_result(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for condition in document.get("conditions", []):
        parameters = condition.get("parameters", {}).get("unique_total", {})
        latency = (condition.get("latency") or {}).get("end_to_end", {})
        flops = condition.get("flops") or {}
        latency_report = condition.get("latency") or {}
        peaks = latency_report.get(
            "peak_cuda_memory_allocated_bytes",
            latency_report.get("peak_cuda_memory_bytes", {}),
        )
        reserved = latency_report.get("peak_cuda_memory_reserved_bytes", {})
        resident = latency_report.get("resident_cuda_memory_bytes", {})
        total_parameters = parameters.get("total")
        storage = parameters.get("storage_bytes")
        estimated_flops = flops.get(
            "estimated_total_flops", flops.get("observed_flops")
        )
        unsupported = flops.get("unsupported_nontrivial_operators") or []
        row = {
            "adapter": document.get("adapter"),
            "condition": condition.get("condition"),
            "task": condition.get("task"),
            "benchmark_family": document.get("benchmark_family"),
            "protocol": document.get("protocol"),
            "resolution": condition.get("resolution"),
            "measured_queries": condition.get("measured_queries"),
            "sampling_steps": condition.get("sampling_steps"),
            "total_parameters": total_parameters,
            "stored_parameter_elements": parameters.get("stored_elements"),
            "trained_parameters": condition.get("trained_parameters"),
            "parameter_storage_gib": storage / 2**30 if storage is not None else None,
            "quantized_parameter_tensors": parameters.get("quantized_tensors"),
            "meta_parameter_tensors": parameters.get("meta_tensors"),
            "parameter_storage_complete": condition.get("parameters", {}).get(
                "storage_complete"
            ),
            "formula_flops": flops.get("formula_flops"),
            "opaque_module_flops": flops.get("opaque_module_flops"),
            "estimated_flops": estimated_flops,
            "estimated_tflops": (
                estimated_flops / 1e12 if estimated_flops is not None else None
            ),
            "estimated_flops_min": flops.get("estimated_total_flops_min"),
            "estimated_flops_max": flops.get("estimated_total_flops_max"),
            "flops_profile_samples": flops.get("profile_sample_count"),
            "flop_confidence": flops.get("confidence"),
            "unsupported_nontrivial_operators": len(unsupported),
            "incremental_parameters_vs_fixed": None,
            "incremental_flops_vs_fixed": None,
            "latency_mean_s": latency.get("mean_s"),
            "latency_median_s": latency.get("median_s"),
            "latency_std_s": latency.get("std_s"),
            "latency_overhead_vs_fixed_s": None,
            "latency_ratio_vs_fixed": None,
            "throughput_images_s": (
                1.0 / latency["mean_s"] if latency.get("mean_s", 0) > 0 else None
            ),
            "prompt_generation_mean_s": _stage_mean(condition, ["prompt_generation"]),
            "image_generation_mean_s": _stage_mean(
                condition, ["image_generation", "diffusion_generation", "model_forward"]
            ),
            "peak_cuda_memory_gib": max(peaks.values()) / 2**30 if peaks else None,
            "peak_cuda_memory_sum_gib": sum(peaks.values()) / 2**30 if peaks else None,
            "peak_cuda_reserved_sum_gib": (
                sum(reserved.values()) / 2**30 if reserved else None
            ),
            "resident_cuda_memory_sum_gib": (
                sum(resident.values()) / 2**30 if resident else None
            ),
            "device": ", ".join(document.get("system", {}).get("cuda_devices", []))
            or document.get("system", {}).get("platform", "unknown"),
            "status": (
                "parameters_only"
                if condition.get("latency") is None
                else flops.get("status", "latency_only")
            ),
        }
        rows.append(row)
    fixed = next((row for row in rows if row["condition"] == "fixed"), None)
    if fixed is not None:
        for row in rows:
            if (
                row["total_parameters"] is not None
                and fixed["total_parameters"] is not None
            ):
                row["incremental_parameters_vs_fixed"] = (
                    row["total_parameters"] - fixed["total_parameters"]
                )
            if (
                row["estimated_flops"] is not None
                and fixed["estimated_flops"] is not None
            ):
                row["incremental_flops_vs_fixed"] = (
                    row["estimated_flops"] - fixed["estimated_flops"]
                )
            if (
                row["latency_mean_s"] is not None
                and fixed["latency_mean_s"] is not None
            ):
                row["latency_overhead_vs_fixed_s"] = (
                    row["latency_mean_s"] - fixed["latency_mean_s"]
                )
                if fixed["latency_mean_s"] > 0:
                    row["latency_ratio_vs_fixed"] = (
                        row["latency_mean_s"] / fixed["latency_mean_s"]
                    )
    return rows


def _suite_condition_for_task(
    document: Dict[str, Any], condition: Dict[str, Any], task: Dict[str, Any]
) -> Dict[str, Any]:
    sampling = (task.get("metadata") or {}).get("sampling") or {}
    latency = task.get("latency") or {}
    return {
        "condition": condition.get("condition"),
        "task": task.get("task"),
        "resolution": document.get("controlled_protocol", {}).get("resolution"),
        "measured_queries": (latency.get("end_to_end") or {}).get("count"),
        "sampling_steps": sampling.get("steps"),
        "parameters": condition.get("parameters"),
        "trained_parameters": condition.get("trained_parameters"),
        "latency": latency,
        "flops": task.get("flops"),
    }


def _flatten_suite_result(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    conditions = document.get("conditions", [])
    task_names = []
    for condition in conditions:
        for task in condition.get("tasks", []):
            if task.get("task") not in task_names:
                task_names.append(task.get("task"))

    rows: List[Dict[str, Any]] = []
    for task_name in task_names:
        pseudo_conditions = []
        for condition in conditions:
            task = next(
                (
                    item
                    for item in condition.get("tasks", [])
                    if item.get("task") == task_name
                ),
                None,
            )
            if task is not None:
                pseudo_conditions.append(
                    _suite_condition_for_task(document, condition, task)
                )
        rows.extend(
            _flatten_single_result(
                {
                    "adapter": document.get("adapter"),
                    "benchmark_family": document.get("benchmark_family"),
                    "protocol": document.get("protocol"),
                    "system": document.get("system", {}),
                    "conditions": pseudo_conditions,
                }
            )
        )

    macro_conditions = []
    for condition in conditions:
        aggregate = condition.get("aggregate")
        if not aggregate:
            continue
        macro_flops = aggregate.get("macro_task_flops")
        macro_conditions.append(
            {
                "condition": condition.get("condition"),
                "task": "__macro__",
                "resolution": document.get("controlled_protocol", {}).get(
                    "resolution"
                ),
                "measured_queries": aggregate.get("queries"),
                "sampling_steps": None,
                "parameters": condition.get("parameters"),
                "trained_parameters": condition.get("trained_parameters"),
                "latency": {
                    "end_to_end": {
                        "count": aggregate.get("queries"),
                        "mean_s": aggregate.get("macro_task_mean_latency_s"),
                        "median_s": aggregate.get("macro_task_median_latency_s"),
                        "std_s": None,
                    }
                },
                "flops": {
                    "estimated_total_flops": macro_flops,
                    "status": (
                        "ok"
                        if aggregate.get("flops_complete_for_all_tasks")
                        else "partial"
                    ),
                    "confidence": "macro over exact per-task profiles",
                },
            }
        )
    if macro_conditions:
        rows.extend(
            _flatten_single_result(
                {
                    "adapter": document.get("adapter"),
                    "benchmark_family": document.get("benchmark_family"),
                    "protocol": document.get("protocol"),
                    "system": document.get("system", {}),
                    "conditions": macro_conditions,
                }
            )
        )
    return rows


def flatten_result(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    if document.get("kind") == "multi_task_comparison_suite":
        return _flatten_suite_result(document)
    return _flatten_single_result(document)


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_reports(rows: Iterable[Dict[str, Any]], output_dir: Path) -> None:
    rows = list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Method",
        "Condition",
        "Task",
        "Family",
        "Resolution",
        "Queries",
        "Steps",
        "Params (B)",
        "Delta Params (B)",
        "Trained (B)",
        "Runtime TFLOPs",
        "FLOP confidence",
        "Latency mean (s)",
        "Latency delta (s)",
        "Latency ratio",
        "Latency std (s)",
        "Peak GPU max/sum (GiB)",
        "Protocol",
    ]
    lines = [
        "# Resource Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        total = row["total_parameters"]
        trained = row["trained_parameters"]
        values = [
            row["adapter"],
            row["condition"],
            str(row.get("task") or ""),
            str(row.get("benchmark_family") or ""),
            str(row.get("resolution") or ""),
            str(row.get("measured_queries") or ""),
            str(row.get("sampling_steps") or ""),
            _format(total / 1e9 if total is not None else None),
            _format(
                row["incremental_parameters_vs_fixed"] / 1e9
                if row["incremental_parameters_vs_fixed"] is not None
                else None
            ),
            _format(trained / 1e9 if trained is not None else None),
            _format(row["estimated_tflops"]),
            str(row["flop_confidence"] or "N/A"),
            _format(row["latency_mean_s"]),
            _format(row["latency_overhead_vs_fixed_s"]),
            _format(row["latency_ratio_vs_fixed"]),
            _format(row["latency_std_s"]),
            f"{_format(row['peak_cuda_memory_gib'])}/{_format(row['peak_cuda_memory_sum_gib'])}",
            str(row["protocol"]).replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Runtime TFLOPs use dispatch-level formulas over the complete dynamic inference. ",
            "Rows with unsupported nontrivial operators are marked partial; the JSON lists them. ",
            "Status=ok means all observed nontrivial operators have a registered formula ",
            "under the stated convention; it is not a hardware instruction count. ",
            "Latency is synchronized wall-clock time and is comparable only for rows measured ",
            "on the same hardware, software stack, precision, resolution, and sampling steps.",
            "Rows from different benchmark families use different task manifests and must not ",
            "be interpreted as task-aligned accuracy comparisons.",
            "",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def load_documents(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    documents = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("schema_version") not in (1, 2):
            raise ValueError(f"Unsupported result schema in {path}")
        documents.append(document)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate comparison benchmark JSON files"
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for document in load_documents(args.results):
        rows.extend(flatten_result(document))
    write_reports(rows, args.output_dir)
    print(f"Wrote {args.output_dir / 'comparison.csv'}")
    print(f"Wrote {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
