"""Build target-task-balanced T2T and same-task competitor evaluation splits.

The original ``eval_dataset.json`` is organized as 26 directional Task-A to
Task-B groups. A target task may therefore occur several times and the same
Task-B query may be duplicated under different source tasks. Paper-facing
per-task comparisons must not count those duplicate queries more than once.

This module keeps an auditable view of the available T2T records and builds an
exactly balanced competitor split. Missing Task-B queries are filled first from
the existing paired evaluation pool and then, only if needed, from paired files
already present below ``data/tasks``. No inpainting target is fabricated.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


FIELDS = ("taskA_input", "taskA_output", "taskB_input", "taskB_output")


def _task_name(path: Any) -> str:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if not parts:
        raise ValueError(f"Cannot derive a task name from {path!r}")
    return parts[0]


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list in {path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record {index} in {path} is not an object")
        missing = [field for field in FIELDS if field not in record]
        if missing:
            raise KeyError(f"Record {index} in {path} is missing {missing}")
    return records


def _validated_task_pair(record: dict[str, Any], index: int) -> tuple[str, str]:
    task_a = _task_name(record["taskA_input"])
    task_b = _task_name(record["taskB_input"])
    if _task_name(record["taskA_output"]) != task_a:
        raise ValueError(f"Record {index} has inconsistent Task-A roots")
    if _task_name(record["taskB_output"]) != task_b:
        raise ValueError(f"Record {index} has inconsistent Task-B roots")
    return task_a, task_b


def build_target_task_records(
    records: Sequence[dict[str, Any]],
    max_per_task: int = 100,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Select at most ``max_per_task`` globally unique Task-B pairs per task."""
    if max_per_task < 1:
        raise ValueError("max_per_task must be positive")
    candidates: dict[
        str, dict[tuple[str, str], list[tuple[int, dict[str, Any]]]]
    ] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        _, task_b = _validated_task_pair(record, index)
        query_key = (str(record["taskB_input"]), str(record["taskB_output"]))
        candidates[task_b][query_key].append((index, record))

    selected: list[dict[str, Any]] = []
    for task_b in sorted(candidates):
        task_rng = random.Random(f"{seed}:{task_b}")
        query_keys = sorted(candidates[task_b])
        if len(query_keys) > max_per_task:
            query_keys = sorted(task_rng.sample(query_keys, max_per_task))
        for query_key in query_keys:
            occurrences = candidates[task_b][query_key]
            source_index, source = task_rng.choice(occurrences)
            task_a, _ = _validated_task_pair(source, source_index)
            selected.append(
                {
                    **source,
                    "query_source": "data/dataset/eval_dataset.json",
                    "benchmark_target_task": task_b,
                    "source_task_a": task_a,
                    "source_task_b": task_b,
                    "source_record_index": source_index,
                    "duplicate_source_record_indices": [
                        index for index, _ in occurrences
                    ],
                }
            )
    return selected


def _load_paired_records(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load the local ``task/input/output`` paired-evaluation schema."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {path}")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record {index} in {path} is not an object")
        missing = [field for field in ("task", "input", "output") if field not in record]
        if missing:
            raise KeyError(f"Record {index} in {path} is missing {missing}")
        task = str(record["task"])
        input_path = str(record["input"])
        output_path = str(record["output"])
        if _task_name(input_path) != task or _task_name(output_path) != task:
            raise ValueError(f"Record {index} in {path} has inconsistent task roots")
        grouped[task].append((input_path, output_path))
    return grouped


def _filesystem_pairs(data_root: Path, task: str) -> list[tuple[str, str]]:
    input_dir = data_root / task / "input"
    output_dir = data_root / task / "output"
    if not input_dir.is_dir() or not output_dir.is_dir():
        raise FileNotFoundError(f"Missing paired task directories below {data_root / task}")
    input_names = {path.name for path in input_dir.iterdir() if path.is_file()}
    output_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    return [
        (f"{task}/input/{name}", f"{task}/output/{name}")
        for name in sorted(input_names & output_names)
    ]


def supplement_target_task_records(
    selected_records: Sequence[dict[str, Any]],
    data_root: Path,
    max_per_task: int = 100,
    seed: int = 2026,
    paired_source: Path | None = None,
) -> list[dict[str, Any]]:
    """Fill every observed target task to exactly ``max_per_task`` unique pairs."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(selected_records):
        _, task_b = _validated_task_pair(record, index)
        grouped[task_b].append(dict(record))

    paired = _load_paired_records(paired_source) if paired_source else {}
    output: list[dict[str, Any]] = []
    for task_b in sorted(grouped):
        task_records = grouped[task_b]
        if len(task_records) > max_per_task:
            raise ValueError(f"Target task {task_b} exceeds {max_per_task} records")
        used = {
            (str(record["taskB_input"]), str(record["taskB_output"]))
            for record in task_records
        }
        needed = max_per_task - len(task_records)
        task_rng = random.Random(f"{seed}:{task_b}:supplement")

        candidate_groups = [
            (
                f"data/dataset/{paired_source.name}" if paired_source else "",
                sorted(set(paired.get(task_b, [])) - used),
            ),
            ("data/tasks paired pool", _filesystem_pairs(data_root, task_b)),
        ]
        for origin, candidates in candidate_groups:
            if needed == 0:
                break
            available = sorted(set(candidates) - used)
            take = min(needed, len(available))
            chosen = task_rng.sample(available, take)
            for input_path, output_path in sorted(chosen):
                task_records.append(
                    {
                        "query_source": origin,
                        "benchmark_target_task": task_b,
                        "source_task_a": None,
                        "source_task_b": task_b,
                        "source_record_index": None,
                        "duplicate_source_record_indices": [],
                        # These placeholders are replaced by build_same_task_records.
                        "taskA_input": input_path,
                        "taskA_output": output_path,
                        "taskB_input": input_path,
                        "taskB_output": output_path,
                    }
                )
                used.add((input_path, output_path))
            needed -= take
        if needed:
            raise ValueError(
                f"Target task {task_b!r} is short by {needed} paired examples"
            )
        output.extend(task_records)
    return output


def build_same_task_records(
    selected_records: Sequence[dict[str, Any]],
    demo_source_records: Sequence[dict[str, Any]] | None = None,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Replace each selected cross-task demonstration with a distinct Task-B pair."""
    demo_source_records = demo_source_records or selected_records
    demo_pools: dict[str, dict[tuple[str, str], tuple[int, dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for index, record in enumerate(demo_source_records):
        _, task_b = _validated_task_pair(record, index)
        key = (str(record["taskB_input"]), str(record["taskB_output"]))
        demo_pools[task_b].setdefault(key, (index, record))

    output: list[dict[str, Any]] = []
    for selected_index, query in enumerate(selected_records):
        _, task_b = _validated_task_pair(query, selected_index)
        query_key = (str(query["taskB_input"]), str(query["taskB_output"]))
        demo_candidates = [
            item for key, item in sorted(demo_pools[task_b].items()) if key != query_key
        ]
        if not demo_candidates:
            raise ValueError(f"Target task {task_b!r} has no distinct demo pair")
        demo_rng = random.Random(
            f"{seed}:{task_b}:{query['taskB_input']}:{query['taskB_output']}"
        )
        demo_source_index, demo = demo_rng.choice(demo_candidates)
        source_task_a = str(
            query.get("source_task_a") or _task_name(query["taskA_input"])
        )
        source_record_index = query.get("source_record_index", selected_index)
        output.append(
            {
                "query_source": query.get("query_source"),
                "benchmark_target_task": task_b,
                "source_task_a": source_task_a,
                "source_task_b": task_b,
                "source_record_index": source_record_index,
                "duplicate_source_record_indices": list(
                    query.get("duplicate_source_record_indices", [source_record_index])
                ),
                "same_task_demo_source_record_index": demo_source_index,
                "taskA_input": demo["taskB_input"],
                "taskA_output": demo["taskB_output"],
                "taskB_input": query["taskB_input"],
                "taskB_output": query["taskB_output"],
            }
        )
    return output


def validate_derivation(
    t2t_records: Sequence[dict[str, Any]],
    same_task_records: Sequence[dict[str, Any]],
    max_per_task: int = 100,
    require_exact_competitor_count: bool = False,
) -> None:
    def validate_unique(records: Sequence[dict[str, Any]], label: str) -> Counter[str]:
        seen_queries: set[tuple[str, str]] = set()
        seen_query_inputs: set[str] = set()
        seen_query_targets: set[str] = set()
        counts: Counter[str] = Counter()
        for index, record in enumerate(records):
            task_b = _task_name(record["taskB_input"])
            query_key = (str(record["taskB_input"]), str(record["taskB_output"]))
            if query_key in seen_queries:
                raise ValueError(f"Duplicate Task-B query/target in {label} at record {index}")
            if query_key[0] in seen_query_inputs:
                raise ValueError(f"Duplicate Task-B query input in {label} at record {index}")
            if query_key[1] in seen_query_targets:
                raise ValueError(f"Duplicate Task-B target output in {label} at record {index}")
            seen_queries.add(query_key)
            seen_query_inputs.add(query_key[0])
            seen_query_targets.add(query_key[1])
            counts[task_b] += 1
            if counts[task_b] > max_per_task:
                raise ValueError(f"Target task {task_b} exceeds {max_per_task} records")
        return counts

    t2t_counts = validate_unique(t2t_records, "T2T split")
    counts = validate_unique(same_task_records, "competitor split")
    if set(counts) != set(t2t_counts):
        raise ValueError("T2T and competitor splits declare different target tasks")
    if require_exact_competitor_count:
        short = {task: count for task, count in counts.items() if count != max_per_task}
        if short:
            raise ValueError(f"Competitor split is not exactly balanced: {short}")

    t2t_queries = {
        (str(row["taskB_input"]), str(row["taskB_output"])) for row in t2t_records
    }
    competitor_queries = {
        (str(row["taskB_input"]), str(row["taskB_output"]))
        for row in same_task_records
    }
    if not t2t_queries.issubset(competitor_queries):
        raise ValueError("Competitor split dropped Task-B queries retained for T2T")

    for index, same_task in enumerate(same_task_records):
        task_b = _task_name(same_task["taskB_input"])
        if _task_name(same_task["taskA_input"]) != task_b:
            raise ValueError(f"Same-task demo has the wrong task at record {index}")
        if _task_name(same_task["taskA_output"]) != task_b:
            raise ValueError(f"Same-task demo target has the wrong task at record {index}")
        if same_task["taskA_input"] == same_task["taskB_input"]:
            raise ValueError(f"Demo/query input overlap at record {index}")
        if same_task["taskA_output"] == same_task["taskB_output"]:
            raise ValueError(f"Demo/query target overlap at record {index}")


def task_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(_task_name(row["taskB_input"]) for row in records).items())
    )


def parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source",
        type=Path,
        default=project_root / "data/dataset/eval_dataset.json",
    )
    result.add_argument(
        "--t2t-output",
        type=Path,
        default=project_root / "data/dataset/eval_dataset_target_task.json",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/dataset/eval_dataset_same_task.json",
    )
    result.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data/tasks",
    )
    result.add_argument(
        "--paired-source",
        type=Path,
        default=project_root / "data/dataset/eval_dataset_1.json",
        help="Existing paired evaluation pool used before falling back to data/tasks",
    )
    result.add_argument("--max-per-task", type=int, default=100)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument(
        "--check",
        action="store_true",
        help="Verify that both existing outputs exactly match the deterministic build",
    )
    return result


def _write(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    source_path = args.source.expanduser().resolve()
    t2t_output_path = args.t2t_output.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    paired_source = args.paired_source.expanduser().resolve()
    source = _load_records(source_path)
    expected_t2t = build_target_task_records(
        source, max_per_task=args.max_per_task, seed=args.seed
    )
    competitor_queries = supplement_target_task_records(
        expected_t2t,
        data_root=data_root,
        max_per_task=args.max_per_task,
        seed=args.seed,
        paired_source=paired_source,
    )
    expected_same_task = build_same_task_records(
        competitor_queries, demo_source_records=competitor_queries, seed=args.seed
    )
    validate_derivation(
        expected_t2t,
        expected_same_task,
        max_per_task=args.max_per_task,
        require_exact_competitor_count=True,
    )
    if args.check:
        actual_t2t = _load_records(t2t_output_path)
        actual_same_task = _load_records(output_path)
        validate_derivation(
            actual_t2t,
            actual_same_task,
            max_per_task=args.max_per_task,
            require_exact_competitor_count=True,
        )
        if actual_t2t != expected_t2t:
            raise ValueError(f"{t2t_output_path} is stale or was edited")
        if actual_same_task != expected_same_task:
            raise ValueError(f"{output_path} is stale or was edited")
        print(
            f"Verified {len(actual_same_task)} unique competitor Task-B records: "
            f"{task_counts(actual_same_task)}"
        )
        return
    _write(t2t_output_path, expected_t2t)
    _write(output_path, expected_same_task)
    print(
        f"Wrote {len(expected_t2t)} available T2T records to {t2t_output_path}; "
        f"wrote {len(expected_same_task)} balanced competitor records to {output_path}: "
        f"{task_counts(expected_same_task)}"
    )


if __name__ == "__main__":
    main()
