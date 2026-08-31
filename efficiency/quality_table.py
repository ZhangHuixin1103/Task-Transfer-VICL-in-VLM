"""Fill Prompt Diffusion/InstructDiffusion PSNR and SSIM cells in a TeX table."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence


ADAPTER_COLUMNS = {
    "prompt-diffusion": 2,
    "instruct-diffusion": 3,
}
TABLE_TASKS = {
    "Deraining": "deraining",
    "Denoising": "denoising",
    "Light Enhancement": "low_light_enhancement",
    "Dehazing": "dehazing",
    "Deblurring": "deblurring",
    "Shadow Removal": "shadow_removal",
    "Inpainting": "inpainting",
    "Relighting": "relighting",
    "Demoiring": "demoireing",
    "Reflection Removal": "reflection_removal",
}
TASK_HEADING = re.compile(r"\\multirow\{2\}\{\*\}\{([^}]+)\}")
METRIC_ROW = re.compile(r"^(\s*)&\s*(PSNR|SSIM)(\$\\uparrow\$\s*)&(.*?)(\\\\\s*)$")


def load_quality_result(path: Path) -> tuple[str, dict[str, dict[str, float]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != "image_quality_suite":
        raise ValueError(f"Not an image_quality_suite result: {path}")
    resolution = document.get("metric_protocol", {}).get("resolution")
    if resolution != 448:
        raise ValueError(f"{path} uses resolution {resolution!r}, expected 448")
    adapter = document.get("adapter")
    if adapter not in ADAPTER_COLUMNS:
        raise ValueError(
            f"{path} uses adapter {adapter!r}; expected one of "
            f"{sorted(ADAPTER_COLUMNS)}"
        )
    conditions = document.get("conditions", [])
    official = [item for item in conditions if item.get("condition") == "official"]
    if len(official) != 1:
        raise ValueError(f"{path} must contain exactly one official condition")
    tasks: dict[str, dict[str, float]] = {}
    for task in official[0].get("tasks", []):
        name = task["task"]
        if name in tasks:
            raise ValueError(f"Duplicate task {name!r} in {path}")
        count = int(task["count"])
        selected = task.get("selected_indices", [])
        heldout = int(task.get("heldout_records", -1))
        if (
            count < 1
            or count != heldout
            or count != len(selected)
            or sorted(selected) != list(range(heldout))
        ):
            raise ValueError(
                f"Task {name!r} is not a complete declared held-out split in {path}"
            )
        psnr = float(task["psnr_mean"])
        ssim = float(task["ssim_mean"])
        if not math.isfinite(psnr) or not math.isfinite(ssim):
            raise ValueError(f"Non-finite aggregate metric for task {name!r}")
        tasks[name] = {"psnr": psnr, "ssim": ssim}
    return adapter, tasks


def _replace_column(row: str, column: int, value: str) -> str:
    match = METRIC_ROW.match(row)
    if not match:
        raise ValueError(f"Unrecognized metric row: {row!r}")
    cells = [cell.strip() for cell in match.group(4).split("&")]
    # The captured payload contains the four method columns. The TeX table's
    # absolute columns 2 and 3 therefore map to payload positions 0 and 1.
    payload_index = column - 2
    if len(cells) != 4 or payload_index not in range(len(cells)):
        raise ValueError(f"Expected four method cells in row: {row!r}")
    cells[payload_index] = value
    return (
        f"{match.group(1)}& {match.group(2)}{match.group(3)}& "
        f"{' & '.join(cells)} {match.group(5)}"
    )


def fill_table(
    template: str,
    results: dict[str, dict[str, dict[str, float]]],
    allow_partial: bool = False,
) -> str:
    expected = set(TABLE_TASKS.values())
    for adapter, tasks in results.items():
        missing = expected - set(tasks)
        if missing and not allow_partial:
            raise ValueError(f"{adapter} result is missing table tasks: {sorted(missing)}")

    current_task: str | None = None
    seen: dict[str, set[tuple[str, str]]] = {adapter: set() for adapter in results}
    output: list[str] = []
    for line in template.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        heading = TASK_HEADING.search(content)
        if heading:
            current_task = TABLE_TASKS.get(heading.group(1))
        metric = METRIC_ROW.match(content)
        if metric and current_task:
            metric_name = metric.group(2).lower()
            for adapter, tasks in results.items():
                values = tasks.get(current_task)
                if values is None:
                    continue
                digits = 2 if metric_name == "psnr" else 3
                content = _replace_column(
                    content,
                    ADAPTER_COLUMNS[adapter],
                    f"{values[metric_name]:.{digits}f}",
                )
                seen[adapter].add((current_task, metric_name))
        output.append(content + newline)

    for adapter, tasks in results.items():
        expected_seen = {
            (task, metric)
            for task in expected & set(tasks)
            for metric in ("psnr", "ssim")
        }
        missing_rows = expected_seen - seen[adapter]
        if missing_rows:
            raise ValueError(
                f"Template has no metric rows for {adapter}: {sorted(missing_rows)}"
            )
    return "".join(output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--template", type=Path, required=True)
    result.add_argument("--result", type=Path, action="append", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--allow-partial", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    loaded: dict[str, dict[str, dict[str, float]]] = {}
    for path in args.result:
        adapter, tasks = load_quality_result(path.expanduser().resolve())
        if adapter in loaded:
            raise ValueError(f"More than one result was supplied for {adapter}")
        loaded[adapter] = tasks
    rendered = fill_table(
        args.template.expanduser().read_text(encoding="utf-8"),
        loaded,
        allow_partial=args.allow_partial,
    )
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
