from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Sequence

from PIL import Image

from .base import load_dataset_records


@dataclass(frozen=True)
class TaskSpec:
    name: str
    eval_json: str
    demo_input: str | None = None
    demo_output: str | None = None
    text_prompt: str = ""
    task_a: str | None = None
    task_b: str | None = None

    def __post_init__(self) -> None:
        if (self.task_a is None) != (self.task_b is None):
            raise ValueError(
                f"Task {self.name!r} must define task_a and task_b together"
            )
        if self.task_a is None and (not self.demo_input or not self.demo_output):
            raise ValueError(
                f"Same-task entry {self.name!r} requires demo_input and demo_output"
            )

    @property
    def is_cross_task(self) -> bool:
        return self.task_a is not None


def _path_task_name(path: Any) -> str:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if not parts or parts[0] in {".", "..", "/"}:
        raise ValueError(f"Cannot derive task name from relative image path {path!r}")
    return parts[0]


def matching_record_indices(
    records: Sequence[dict[str, Any]], task: TaskSpec
) -> list[int] | None:
    """Return source indices for one directional T2T pair, or None for paired JSON."""
    if not task.is_cross_task:
        return None
    assert task.task_a is not None and task.task_b is not None
    required = ("taskA_input", "taskA_output", "taskB_input", "taskB_output")
    selected: list[int] = []
    for index, record in enumerate(records):
        missing = [field for field in required if field not in record]
        if missing:
            raise KeyError(
                f"Cross-task dataset record {index} is missing fields {missing}"
            )
        record_task_a = _path_task_name(record["taskA_input"])
        record_task_b = _path_task_name(record["taskB_input"])
        if record_task_a != task.task_a or record_task_b != task.task_b:
            continue
        output_task_a = _path_task_name(record["taskA_output"])
        output_task_b = _path_task_name(record["taskB_output"])
        if output_task_a != task.task_a or output_task_b != task.task_b:
            raise ValueError(
                f"Cross-task record {index} has inconsistent input/output task roots: "
                f"A={record_task_a}/{output_task_a}, B={record_task_b}/{output_task_b}"
            )
        selected.append(index)
    if not selected:
        raise ValueError(
            f"No records match directional pair {task.task_a} -> {task.task_b} "
            f"for manifest entry {task.name!r}"
        )
    return selected


def load_task_manifest(path: Path) -> tuple[dict[str, Any], list[TaskSpec]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported task manifest schema in {path}")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"No tasks found in {path}")
    tasks = [TaskSpec(**item) for item in raw_tasks]
    names = [task.name for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError(f"Task names must be unique in {path}")
    return document, tasks


def select_indices(count: int, limit: int, seed: int) -> list[int]:
    if count < 1:
        raise ValueError("count must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit >= count:
        return list(range(count))
    generator = random.Random(seed)
    return sorted(generator.sample(range(count), limit))


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def validate_task(
    task: TaskSpec,
    data_root: Path,
    check_images: int,
    max_examples: int,
) -> dict[str, Any]:
    json_path = (data_root / task.eval_json).resolve()
    result: dict[str, Any] = {
        "task": task.name,
        "spec": asdict(task),
        "json_path": str(json_path),
        "json_exists": json_path.is_file(),
        "records": 0,
        "missing_paths": [],
        "unreadable_paths": [],
        "pair_size_mismatches": [],
        "input_sizes": {},
        "target_sizes": {},
        "demo": {},
    }
    if not json_path.is_file():
        result["status"] = "missing_json"
        return result

    source_records = load_dataset_records(json_path)
    record_indices = matching_record_indices(source_records, task)
    records = (
        source_records
        if record_indices is None
        else [source_records[index] for index in record_indices]
    )
    result["source_records"] = len(source_records)
    result["records"] = len(records)
    result["source_record_indices"] = record_indices
    if task.is_cross_task:
        path_fields = (
            "taskA_input",
            "taskA_output",
            "taskB_input",
            "taskB_output",
        )
        paired_fields = (
            ("taskA_input", "taskA_output", "A"),
            ("taskB_input", "taskB_output", "B"),
        )
    else:
        path_fields = ("image_path", "target_path")
        paired_fields = (("image_path", "target_path", "query"),)

    missing_count = 0
    for index, record in enumerate(records):
        for key in path_fields:
            if key not in record:
                raise KeyError(f"{json_path} record {index} is missing {key}")
            candidate = data_root / record[key]
            if not candidate.is_file():
                missing_count += 1
                if len(result["missing_paths"]) < max_examples:
                    result["missing_paths"].append(
                        {"index": index, "field": key, "path": record[key]}
                    )
    result["missing_path_count"] = missing_count

    demo_missing = 0
    if not task.is_cross_task:
        for field in ("demo_input", "demo_output"):
            relative = getattr(task, field)
            assert relative is not None
            path = data_root / relative
            exists = path.is_file()
            demo_missing += not exists
            result["demo"][field] = {"path": relative, "exists": exists}
    result["demo_missing_count"] = int(demo_missing)

    available = [
        (index, record)
        for index, record in enumerate(records)
        if all((data_root / record[field]).is_file() for field in path_fields)
    ]
    if check_images != 0:
        if check_images < 0:
            inspected = available
        else:
            inspected = [
                available[index]
                for index in select_indices(len(available), check_images, seed=2026)
            ] if available else []
        input_sizes: Counter[str] = Counter()
        target_sizes: Counter[str] = Counter()
        unreadable_count = 0
        mismatch_count = 0
        for index, record in inspected:
            for input_field, target_field, pair_name in paired_fields:
                try:
                    input_size = _image_size(data_root / record[input_field])
                    target_size = _image_size(data_root / record[target_field])
                except Exception as error:
                    unreadable_count += 1
                    if len(result["unreadable_paths"]) < max_examples:
                        result["unreadable_paths"].append(
                            {
                                "index": index,
                                "pair": pair_name,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                    continue
                input_sizes[f"{input_size[0]}x{input_size[1]}"] += 1
                target_sizes[f"{target_size[0]}x{target_size[1]}"] += 1
                if input_size != target_size:
                    mismatch_count += 1
                    if len(result["pair_size_mismatches"]) < max_examples:
                        result["pair_size_mismatches"].append(
                            {
                                "index": index,
                                "pair": pair_name,
                                "input": record[input_field],
                                "target": record[target_field],
                                "input_size": input_size,
                                "target_size": target_size,
                            }
                        )
        result["checked_image_pairs"] = len(inspected)
        result["unreadable_path_count"] = unreadable_count
        result["pair_size_mismatch_count"] = mismatch_count
        result["input_sizes"] = dict(input_sizes.most_common())
        result["target_sizes"] = dict(target_sizes.most_common())

    failures = (
        result["missing_path_count"]
        + result["demo_missing_count"]
        + result.get("unreadable_path_count", 0)
    )
    result["native_pair_size_policy"] = (
        "Recorded as a warning, not a failure: controlled efficiency adapters "
        "resize every consumed image and requested output to the manifest resolution."
    )
    result["status"] = "ok" if failures == 0 else "incomplete"
    return result


def validate_json_file(
    json_path: Path,
    data_root: Path,
    check_images: int,
    max_examples: int,
) -> dict[str, Any]:
    """Validate every record in one paired-image JSON, including non-eval splits."""
    result: dict[str, Any] = {
        "json_path": str(json_path),
        "relative_json_path": str(json_path.relative_to(data_root)),
        "records": 0,
        "missing_paths": [],
        "unreadable_paths": [],
        "pair_size_mismatches": [],
        "input_sizes": {},
        "target_sizes": {},
    }
    try:
        records = load_dataset_records(json_path)
    except Exception as error:
        result.update(
            status="invalid_json",
            error=f"{type(error).__name__}: {error}",
        )
        return result

    result["records"] = len(records)
    missing_count = 0
    available: list[tuple[int, dict[str, Any]]] = []
    malformed_count = 0
    for index, record in enumerate(records):
        missing_fields = [
            field for field in ("image_path", "target_path") if field not in record
        ]
        if missing_fields:
            malformed_count += 1
            if len(result["missing_paths"]) < max_examples:
                result["missing_paths"].append(
                    {"index": index, "missing_fields": missing_fields}
                )
            continue

        paths_exist = True
        for field in ("image_path", "target_path"):
            candidate = data_root / record[field]
            if not candidate.is_file():
                paths_exist = False
                missing_count += 1
                if len(result["missing_paths"]) < max_examples:
                    result["missing_paths"].append(
                        {"index": index, "field": field, "path": record[field]}
                    )
        if paths_exist:
            available.append((index, record))

    result["malformed_record_count"] = malformed_count
    result["missing_path_count"] = missing_count

    if check_images != 0 and available:
        if check_images < 0:
            inspected = available
        else:
            inspected = [
                available[index]
                for index in select_indices(len(available), check_images, seed=2026)
            ]
        input_sizes: Counter[str] = Counter()
        target_sizes: Counter[str] = Counter()
        unreadable_count = 0
        mismatch_count = 0
        for index, record in inspected:
            try:
                input_size = _image_size(data_root / record["image_path"])
                target_size = _image_size(data_root / record["target_path"])
            except Exception as error:
                unreadable_count += 1
                if len(result["unreadable_paths"]) < max_examples:
                    result["unreadable_paths"].append(
                        {"index": index, "error": f"{type(error).__name__}: {error}"}
                    )
                continue
            input_sizes[f"{input_size[0]}x{input_size[1]}"] += 1
            target_sizes[f"{target_size[0]}x{target_size[1]}"] += 1
            if input_size != target_size:
                mismatch_count += 1
                if len(result["pair_size_mismatches"]) < max_examples:
                    result["pair_size_mismatches"].append(
                        {
                            "index": index,
                            "input": record["image_path"],
                            "target": record["target_path"],
                            "input_size": input_size,
                            "target_size": target_size,
                        }
                    )
        result["checked_image_pairs"] = len(inspected)
        result["unreadable_path_count"] = unreadable_count
        result["pair_size_mismatch_count"] = mismatch_count
        result["input_sizes"] = dict(input_sizes.most_common())
        result["target_sizes"] = dict(target_sizes.most_common())

    failures = (
        malformed_count
        + missing_count
        + result.get("unreadable_path_count", 0)
    )
    result["native_pair_size_policy"] = (
        "Recorded as a warning, not a failure under the fixed-size efficiency protocol."
    )
    result["status"] = "ok" if failures == 0 else "incomplete"
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate all task JSON records, demo paths, image readability, and pair sizes"
    )
    result.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("tasks.json")
    )
    result.add_argument("--data-root", type=Path)
    result.add_argument("--tasks", nargs="+")
    result.add_argument(
        "--check-images",
        type=int,
        default=100,
        help="Pairs to decode per task; 0 skips decoding and -1 checks every available pair",
    )
    result.add_argument("--max-examples", type=int, default=20)
    result.add_argument(
        "--all-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also validate every paired-image JSON below data-root (default: enabled)",
    )
    result.add_argument("--output", type=Path)
    result.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the summary while still writing the complete JSON report",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    document, tasks = load_task_manifest(manifest_path)
    if args.tasks:
        requested = set(args.tasks)
        tasks = [task for task in tasks if task.name in requested]
        missing = requested - {task.name for task in tasks}
        if missing:
            raise ValueError(f"Unknown tasks: {sorted(missing)}")
    default_root = manifest_path.parents[1] / document["data_root"]
    data_root = (args.data_root or default_root).resolve()
    reports = [
        validate_task(task, data_root, args.check_images, args.max_examples)
        for task in tasks
    ]
    json_reports = (
        [
            validate_json_file(path, data_root, args.check_images, args.max_examples)
            for path in sorted(data_root.rglob("*.json"))
        ]
        if args.all_json
        else []
    )
    result = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "tasks": reports,
        "json_files": json_reports,
        "summary": {
            "tasks": len(reports),
            "ok": sum(report["status"] == "ok" for report in reports),
            "missing_paths": sum(report.get("missing_path_count", 0) for report in reports),
            "missing_demos": sum(report.get("demo_missing_count", 0) for report in reports),
            "unreadable_paths": sum(
                report.get("unreadable_path_count", 0) for report in reports
            ),
            "pair_size_mismatches": sum(
                report.get("pair_size_mismatch_count", 0) for report in reports
            ),
            "all_json_files": len(json_reports),
            "all_json_ok": sum(
                report.get("status") == "ok" for report in json_reports
            ),
            "all_json_missing_paths": sum(
                report.get("missing_path_count", 0) for report in json_reports
            ),
            "all_json_malformed_records": sum(
                report.get("malformed_record_count", 0) for report in json_reports
            ),
            "all_json_unreadable_paths": sum(
                report.get("unreadable_path_count", 0) for report in json_reports
            ),
            "all_json_pair_size_mismatches": sum(
                report.get("pair_size_mismatch_count", 0) for report in json_reports
            ),
        },
    }
    payload = json.dumps(result, indent=2)
    print(
        json.dumps(result["summary"], indent=2)
        if args.summary_only
        else payload
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
