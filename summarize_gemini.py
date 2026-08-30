import argparse
import json
import os
from typing import Dict, List, Optional


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def fmt_coverage(value: Optional[float], count: Optional[int], total: int) -> str:
    if value is None:
        return f"-- ({count or 0}/{total})"
    return f"{value:.3f} ({count or 0}/{total})"


def is_number(value) -> bool:
    return isinstance(value, (int, float))


def summarize_masked_subset(
    qwen_by_id: Dict[str, dict], masked_rows: List[dict], changed_only: bool
) -> dict:
    selected = [
        row
        for row in masked_rows
        if row["combo_id"] in qwen_by_id
        and (not changed_only or row.get("prompt_changed") is True)
    ]
    qwen_rows = [qwen_by_id[row["combo_id"]] for row in selected]
    result = {
        "num_pairs": len(selected),
        "num_prompt_changed": sum(row.get("prompt_changed") is True for row in selected),
        "qwen": {},
        "qwen_masked": {},
    }
    for metric in ["first_psnr", "first_ssim", "first_viescore"]:
        paired = [
            (qwen_row, masked_row)
            for qwen_row, masked_row in zip(qwen_rows, selected)
            if is_number(qwen_row.get(metric)) and is_number(masked_row.get(metric))
        ]
        result["qwen"][metric] = (
            sum(float(qwen_row[metric]) for qwen_row, _ in paired) / len(paired)
            if paired
            else None
        )
        result["qwen_masked"][metric] = (
            sum(float(masked_row[metric]) for _, masked_row in paired) / len(paired)
            if paired
            else None
        )
        result["qwen"][f"{metric}_count"] = len(paired)
        result["qwen_masked"][f"{metric}_count"] = len(paired)
    return result


def write_qwen_masked_comparison(root: str) -> None:
    qwen_path = os.path.join(root, "qwen", "combo_summary.json")
    masked_path = os.path.join(root, "qwen_masked", "combo_summary.json")
    if not os.path.exists(qwen_path) or not os.path.exists(masked_path):
        return

    qwen_rows = read_json(qwen_path)
    masked_rows = read_json(masked_path)
    qwen_by_id = {row["combo_id"]: row for row in qwen_rows}
    result = {
        "all_shared": summarize_masked_subset(qwen_by_id, masked_rows, False),
        "explicit_task_term_hit_only": summarize_masked_subset(
            qwen_by_id, masked_rows, True
        ),
    }
    json_path = os.path.join(root, "qwen_masked_comparison.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    md_path = os.path.join(root, "qwen_masked_comparison.md")
    with open(md_path, "w") as f:
        f.write(
            "| Subset | Method | N | First PSNR | First SSIM | "
            "First VIE (coverage) |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|\n")
        for subset, stats in result.items():
            for method in ["qwen", "qwen_masked"]:
                values = stats[method]
                f.write(
                    "| {subset} | {method} | {n} | {psnr} | {ssim} | {vie} |\n".format(
                        subset=subset,
                        method=method,
                        n=stats["num_pairs"],
                        psnr=fmt(values["first_psnr"]),
                        ssim=fmt(values["first_ssim"]),
                        vie=fmt_coverage(
                            values["first_viescore"],
                            values["first_viescore_count"],
                            stats["num_pairs"],
                        ),
                    )
                )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Collect summary_global.json files from Gemini condition runs."
    )
    parser.add_argument("--root", default="data/output/supplementary/gemini")
    parser.add_argument(
        "--modes",
        default="fixed,qwen,qwen_masked,task_name,target_desc",
        help="Comma-separated prompt modes to include.",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    for mode in modes:
        path = os.path.join(args.root, mode, "summary_global.json")
        if not os.path.exists(path):
            continue
        data = read_json(path)
        rows.append((mode, data))

    out_path = os.path.join(args.root, "summary_across_modes.md")
    os.makedirs(args.root, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("| Mode | N | First PSNR | Mean PSNR | Std PSNR | Best PSNR | First VIE (coverage) | Mean VIE (coverage) | Best VIE (coverage) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for mode, data in rows:
            f.write(
                "| {mode} | {n} | {first_psnr} | {mean_psnr} | {std_psnr} | {best_psnr} | {first_vie} | {mean_vie} | {best_vie} |\n".format(
                    mode=mode,
                    n=data.get("num_combos", 0),
                    first_psnr=fmt(data.get("avg_first_psnr")),
                    mean_psnr=fmt(data.get("avg_mean_psnr")),
                    std_psnr=fmt(data.get("avg_std_psnr")),
                    best_psnr=fmt(data.get("avg_best_psnr")),
                    first_vie=fmt_coverage(
                        data.get("avg_first_viescore"),
                        data.get("count_first_viescore"),
                        data.get("num_combos", 0),
                    ),
                    mean_vie=fmt_coverage(
                        data.get("avg_mean_viescore"),
                        data.get("count_mean_viescore"),
                        data.get("num_combos", 0),
                    ),
                    best_vie=fmt_coverage(
                        data.get("avg_best_viescore"),
                        data.get("count_best_viescore"),
                        data.get("num_combos", 0),
                    ),
                )
            )
    print(f"Wrote {out_path}")
    write_qwen_masked_comparison(args.root)


if __name__ == "__main__":
    main()
