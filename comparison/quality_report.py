"""Combine quality runs into paper-facing per-task PSNR/SSIM tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def _finite(value: Any, field: str, source: Path) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field} in {source}")
    return result


def load_quality(path: Path, require_count: int | None = 100) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != "image_quality_suite":
        raise ValueError(f"Not an image_quality_suite result: {path}")
    rows: list[dict[str, Any]] = []
    conditions = document.get("conditions") or []
    for condition in conditions:
        condition_name = str(condition.get("condition"))
        method = str(document.get("adapter"))
        if condition_name != "official":
            method = f"{method}:{condition_name}"
        for task in condition.get("tasks") or []:
            count = int(task["count"])
            if require_count is not None and count != require_count:
                raise ValueError(
                    f"{path}: {method}/{task['task']} has {count} records; "
                    f"expected {require_count}"
                )
            rows.append(
                {
                    "method": method,
                    "task": str(task["task"]),
                    "count": count,
                    "psnr": _finite(task["psnr_mean"], "PSNR", path),
                    "ssim": _finite(task["ssim_mean"], "SSIM", path),
                    "source_type": "comparison_quality",
                    "source": str(path),
                }
            )
    if not rows:
        raise ValueError(f"No quality rows in {path}")
    return rows


def load_legacy_pair_summary(specification: str) -> list[dict[str, Any]]:
    try:
        method, raw_path = specification.split("=", 1)
    except ValueError as error:
        raise ValueError("--legacy-result must have the form METHOD=PATH") from error
    path = Path(raw_path).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a task-pair result object in {path}")
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for pair, result in document.items():
        if not isinstance(result, dict) or "__" not in pair:
            continue
        task_b = pair.rsplit("__", 1)[1]
        grouped[task_b].append(
            (
                int(result["num_samples"]),
                _finite(result["avg_psnr"], "PSNR", path),
                _finite(result["avg_ssim"], "SSIM", path),
            )
        )
    rows = []
    for task, values in sorted(grouped.items()):
        count = sum(item[0] for item in values)
        rows.append(
            {
                "method": method,
                "task": task,
                "count": count,
                "psnr": sum(n * psnr for n, psnr, _ in values) / count,
                "ssim": sum(n * ssim for n, _, ssim in values) / count,
                "source_type": "legacy_direction_weighted",
                "source": str(path),
            }
        )
    if not rows:
        raise ValueError(f"No task-pair summaries found in {path}")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    seen: set[tuple[str, str]] = set()
    methods: list[str] = []
    tasks: set[str] = set()
    for row in rows:
        key = (row["method"], row["task"])
        if key in seen:
            raise ValueError(f"Duplicate method/task row: {key}")
        seen.add(key)
        if row["method"] not in methods:
            methods.append(row["method"])
        tasks.add(row["task"])
    expected = set(tasks)
    for method in methods:
        present = {row["task"] for row in rows if row["method"] == method}
        if present != expected:
            raise ValueError(
                f"{method} has a different task set; missing={sorted(expected - present)}, "
                f"extra={sorted(present - expected)}"
            )
    return methods, sorted(tasks)


def add_macro_rows(rows: list[dict[str, Any]], methods: list[str]) -> None:
    for method in methods:
        values = [row for row in rows if row["method"] == method]
        rows.append(
            {
                "method": method,
                "task": "__macro__",
                "count": sum(row["count"] for row in values),
                "psnr": statistics.fmean(row["psnr"] for row in values),
                "ssim": statistics.fmean(row["ssim"] for row in values),
                "source_type": values[0]["source_type"],
                "source": values[0]["source"],
            }
        )


def write_reports(
    rows: list[dict[str, Any]], methods: list[str], tasks: list[str], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "quality_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["method"], row["task"]): row for row in rows}
    display_tasks = [*tasks, "__macro__"]
    markdown = [
        "| Task | " + " | ".join(methods) + " |",
        "| --- | " + " | ".join("---:" for _ in methods) + " |",
    ]
    latex = []
    for task in display_tasks:
        cells = []
        for method in methods:
            row = lookup[(method, task)]
            cells.append(f"{row['psnr']:.2f} / {row['ssim']:.3f}")
        label = "Macro average" if task == "__macro__" else task.replace("_", " ")
        markdown.append(f"| {label} | " + " | ".join(cells) + " |")
        latex_label = label.replace("_", r"\_")
        latex.append(f"{latex_label} & " + " & ".join(cells) + r" \\")
    (output_dir / "quality_comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "quality_comparison_rows.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("results", nargs="*", type=Path)
    result.add_argument(
        "--legacy-result",
        action="append",
        default=[],
        help="Existing directional summary as METHOD=PATH",
    )
    result.add_argument("--require-count", type=int, default=100)
    result.add_argument("--allow-incomplete", action="store_true")
    result.add_argument(
        "--output-dir", type=Path, default=Path("comparison/outputs/paper_tables")
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not args.results and not args.legacy_result:
        raise ValueError("Supply at least one quality result")
    require_count = None if args.allow_incomplete else args.require_count
    rows: list[dict[str, Any]] = []
    for path in args.results:
        rows.extend(load_quality(path.expanduser().resolve(), require_count))
    for specification in args.legacy_result:
        rows.extend(load_legacy_pair_summary(specification))
    if args.legacy_result:
        print(
            "WARNING: legacy summaries may use ground-truth-selected outputs; "
            "do not present them as single-shot results."
        )
    methods, tasks = validate_rows(rows)
    add_macro_rows(rows, methods)
    write_reports(rows, methods, tasks, args.output_dir.expanduser().resolve())
    print(f"Wrote quality tables to {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
