#!/usr/bin/env python3
"""Prepare a deterministic, uploadable subset of data/others."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path, PurePosixPath


def task_seed(base: int, task_name: str) -> int:
    return base + sum((index + 1) * ord(char) for index, char in enumerate(task_name))


def select_indices(count: int, limit: int, seed: int) -> list[int]:
    if limit >= count:
        return list(range(count))
    return sorted(random.Random(seed).sample(range(count), limit))


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--manifest", type=Path, default=Path("efficiency/tasks.json"))
    args = parser.parse_args()
    if args.samples_per_task < 1:
        raise SystemExit("--samples-per-task must be positive")

    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        args.manifest.resolve()
        if args.manifest.is_absolute()
        else (project_root / args.manifest).resolve()
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = (project_root / manifest["data_root"]).resolve()
    subset_dir = data_root / "_efficiency_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)

    upload_paths: set[str] = set()
    subset_tasks = []
    selection = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "samples_per_task": args.samples_per_task,
        "sampling_seed": args.seed,
        "tasks": [],
    }

    for task in manifest["tasks"]:
        if task.get("task_a") is not None or task.get("task_b") is not None:
            raise ValueError("Sparse data/others preparation accepts same-task entries only")
        demo_input = safe_relative_path(task["demo_input"])
        demo_output = safe_relative_path(task["demo_output"])
        source_json = data_root / safe_relative_path(task["eval_json"])
        records = json.loads(source_json.read_text(encoding="utf-8"))
        heldout_source_indices = []
        excluded_demo_indices = []
        for index, record in enumerate(records):
            image_path = safe_relative_path(record["image_path"])
            target_path = safe_relative_path(record["target_path"])
            if image_path == demo_input or target_path == demo_output:
                excluded_demo_indices.append(index)
            else:
                heldout_source_indices.append(index)
        if not heldout_source_indices:
            raise ValueError(f"Task {task['name']} has no held-out records")

        local_indices = select_indices(
            len(heldout_source_indices),
            args.samples_per_task,
            task_seed(args.seed, task["name"]),
        )
        source_indices = [heldout_source_indices[index] for index in local_indices]
        selected_records = [records[index] for index in source_indices]
        subset_relative = f"_efficiency_subset/{task['name']}.json"
        subset_path = data_root / subset_relative
        subset_path.write_text(json.dumps(selected_records, indent=2), encoding="utf-8")

        subset_task = dict(task)
        subset_task["eval_json"] = subset_relative
        subset_tasks.append(subset_task)
        upload_paths.add(str(subset_path.relative_to(project_root)))
        upload_paths.add(f"data/others/{demo_input}")
        upload_paths.add(f"data/others/{demo_output}")
        for record in selected_records:
            upload_paths.add(
                f"data/others/{safe_relative_path(record['image_path'])}"
            )
            upload_paths.add(
                f"data/others/{safe_relative_path(record['target_path'])}"
            )
        selection["tasks"].append(
            {
                "task": task["name"],
                "source_json": str(source_json.relative_to(project_root)),
                "source_records": len(records),
                "heldout_records": len(heldout_source_indices),
                "selected_records": len(selected_records),
                "selected_source_indices": source_indices,
                "excluded_demo_source_indices": excluded_demo_indices,
            }
        )

    sparse_manifest = dict(manifest)
    sparse_manifest["benchmark_family"] = "same-task-third-party-sparse-subset"
    sparse_manifest["tasks"] = subset_tasks
    sparse_manifest["controlled_protocol"] = {
        **manifest["controlled_protocol"],
        "measured_queries_per_task": args.samples_per_task,
        "note": (
            "Deterministic sparse upload subset. Every listed record is measured; "
            "warm-up reuses measured records when the full subset is used."
        ),
    }
    sparse_manifest_path = project_root / "efficiency/tasks_sparse.json"
    sparse_manifest_path.write_text(
        json.dumps(sparse_manifest, indent=2), encoding="utf-8"
    )
    selection_path = project_root / "efficiency/upload_subset_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    upload_paths.update(
        {
            str(sparse_manifest_path.relative_to(project_root)),
            str(selection_path.relative_to(project_root)),
        }
    )

    upload_list = project_root / "efficiency/upload_subset_files.txt"
    upload_paths.add(str(upload_list.relative_to(project_root)))
    upload_list.write_text("\n".join(sorted(upload_paths)) + "\n", encoding="utf-8")
    missing = [path for path in sorted(upload_paths) if not (project_root / path).is_file()]
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Selected upload files are missing:\n{preview}")
    total_bytes = sum((project_root / path).stat().st_size for path in upload_paths)
    print(f"Wrote {sparse_manifest_path}")
    print(f"Wrote {selection_path}")
    print(f"Wrote {upload_list}")
    print(f"Files: {len(upload_paths):,}; uncompressed bytes: {total_bytes:,}")
    for task in selection["tasks"]:
        print(f"  {task['task']}: {task['selected_records']} pairs")


if __name__ == "__main__":
    main()
