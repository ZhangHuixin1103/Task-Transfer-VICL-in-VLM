import argparse
import json
import os
import random
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


DATA_TASKS_DIR = "data/tasks"


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: str, rows: List[dict]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_json_atomic(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_combo_summary(root: str, mode: str) -> Dict[str, dict]:
    path = os.path.join(root, mode, "combo_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = read_json(path)
    return {item["combo_id"]: item for item in data}


def load_first_attempts(root: str, mode: str) -> Dict[str, dict]:
    """Load strict attempt_00 outputs without depending on a scoped summary file."""
    mode_dir = os.path.join(root, mode)
    records = {}
    for current_root, _, files in os.walk(mode_dir):
        if "attempt_00.json" not in files:
            continue
        sidecar_path = os.path.join(current_root, "attempt_00.json")
        try:
            record = read_json(sidecar_path)
        except Exception:
            continue
        if record.get("status") != "ok" or record.get("attempt_index") != 0:
            continue
        image_path = record.get("image")
        canonical_image = os.path.join(current_root, "attempt_00.png")
        if not image_path or not os.path.exists(image_path):
            image_path = canonical_image
        if not os.path.exists(image_path):
            continue
        record = dict(record)
        record["first_image"] = image_path
        records[record["combo_id"]] = record
    return records


def load_mode_records(root: str, mode: str, selector: str) -> Dict[str, dict]:
    if selector == "first":
        return load_first_attempts(root, mode)
    return load_combo_summary(root, mode)


def resolve_image(record: dict, selector: str) -> Optional[str]:
    if selector == "first":
        return record.get("first_image")
    if selector == "best_psnr":
        return record.get("best_image")
    raise ValueError(f"Unknown selector: {selector}")


def load_img(path: str, size=(256, 256)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def label(draw: ImageDraw.ImageDraw, xy, text: str) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([xy[0], xy[1], xy[0] + 220, xy[1] + 18], fill=(255, 255, 255))
    draw.text((xy[0] + 4, xy[1] + 3), text, fill=(0, 0, 0), font=font)


def make_panel(
    task_a_input: str,
    task_a_output: str,
    task_b_input: str,
    left_output: str,
    right_output: str,
    out_path: str,
    include_gt: bool = False,
    task_b_output: Optional[str] = None,
) -> None:
    cell = (256, 256)
    gap = 10
    top = [
        load_img(os.path.join(DATA_TASKS_DIR, task_a_input), cell),
        load_img(os.path.join(DATA_TASKS_DIR, task_a_output), cell),
        load_img(os.path.join(DATA_TASKS_DIR, task_b_input), cell),
    ]
    bottom = [load_img(left_output, cell), load_img(right_output, cell)]
    if include_gt and task_b_output:
        bottom.append(load_img(os.path.join(DATA_TASKS_DIR, task_b_output), cell))

    width = max(len(top), len(bottom)) * cell[0] + (max(len(top), len(bottom)) - 1) * gap
    height = cell[1] * 2 + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(top):
        x = i * (cell[0] + gap)
        canvas.paste(img, (x, 0))
    top_labels = ["Task A input", "Task A output", "Task B query"]
    for i, text in enumerate(top_labels):
        label(draw, (i * (cell[0] + gap), 0), text)

    y = cell[1] + gap
    for i, img in enumerate(bottom):
        x = i * (cell[0] + gap)
        canvas.paste(img, (x, y))
    bottom_labels = ["Output A", "Output B"] + (["Reference output"] if include_gt else [])
    for i, text in enumerate(bottom_labels):
        label(draw, (i * (cell[0] + gap), y), text)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Create blind A/B human-evaluation panels."
    )
    parser.add_argument("--root", default="data/output/supplementary/gemini")
    parser.add_argument("--left_mode", default="fixed")
    parser.add_argument("--right_mode", default="qwen")
    parser.add_argument("--selector", choices=["first", "best_psnr"], default="first")
    parser.add_argument("--max_items", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--include_gt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the reference output. Existing manifests keep their original setting.",
    )
    parser.add_argument("--out_dir", default="data/output/supplementary/human_eval")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard the existing manifest/answer key and create a new randomized study.",
    )
    args = parser.parse_args()

    left = load_mode_records(args.root, args.left_mode, args.selector)
    right = load_mode_records(args.root, args.right_mode, args.selector)
    shared = sorted(set(left) & set(right))
    rng = random.Random(args.seed)
    rng.shuffle(shared)

    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    answer_key_path = os.path.join(args.out_dir, "answer_key.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_rows: List[dict] = []
    answer_rows: List[dict] = []
    if not args.reset and (os.path.exists(manifest_path) or os.path.exists(answer_key_path)):
        if not os.path.exists(manifest_path) or not os.path.exists(answer_key_path):
            raise RuntimeError(
                "Found only one of manifest.jsonl and answer_key.jsonl. "
                "Restore the missing file or pass --reset explicitly."
            )
        manifest_rows = read_jsonl(manifest_path)
        answer_rows = read_jsonl(answer_key_path)
        manifest_ids = [row.get("item_id") for row in manifest_rows]
        answer_ids = [row.get("item_id") for row in answer_rows]
        if manifest_ids != answer_ids or len(set(manifest_ids)) != len(manifest_ids):
            raise RuntimeError("Existing manifest and answer key are inconsistent.")
        expected_ids = [f"item_{index:04d}" for index in range(len(manifest_ids))]
        if manifest_ids != expected_ids:
            raise RuntimeError(
                "Existing item IDs are not contiguous; refusing to append and risk collisions."
            )
        if any(row.get("selector") != args.selector for row in manifest_rows):
            raise RuntimeError(
                f"Existing study does not use selector={args.selector!r}; use matching options or --reset."
            )
        if any(
            {row.get("A_method"), row.get("B_method")}
            != {args.left_mode, args.right_mode}
            for row in answer_rows
        ):
            raise RuntimeError(
                "Existing study compares different methods; use matching options or --reset."
            )

    if manifest_rows:
        reference_flags = {bool(row.get("include_reference")) for row in manifest_rows}
        if len(reference_flags) != 1:
            raise RuntimeError("Existing manifest mixes reference and no-reference panels.")
        existing_include_gt = reference_flags.pop()
        if args.include_gt is None:
            include_gt = existing_include_gt
        elif args.include_gt != existing_include_gt:
            raise RuntimeError(
                "Existing study uses a different reference-output setting; use matching options or --reset."
            )
        else:
            include_gt = args.include_gt
    else:
        include_gt = bool(args.include_gt)

    existing_combo_ids = {row["combo_id"] for row in answer_rows}
    if len(existing_combo_ids) != len(answer_rows):
        raise RuntimeError("Existing answer key contains duplicate combo IDs.")
    used_combo_ids = set(existing_combo_ids)
    strict_shared_ids = set(shared)
    existing_without_strict_attempt_00 = sorted(existing_combo_ids - strict_shared_ids)

    written = len(manifest_rows)
    added = 0
    skipped = 0
    new_combo_ids = []
    for combo_id in shared:
        if written >= args.max_items:
            break
        if combo_id in used_combo_ids:
            continue
        lrec = left[combo_id]
        rrec = right[combo_id]
        if args.selector == "first" and (
            lrec.get("attempt_index") != 0 or rrec.get("attempt_index") != 0
        ):
            raise RuntimeError(
                f"Strict first-attempt selection received a nonzero attempt for {combo_id}."
            )
        limg = resolve_image(lrec, args.selector)
        rimg = resolve_image(rrec, args.selector)
        if not limg or not rimg or not os.path.exists(limg) or not os.path.exists(rimg):
            skipped += 1
            continue

        swap = rng.random() < 0.5
        left_img, right_img = (rimg, limg) if swap else (limg, rimg)
        left_label = args.right_mode if swap else args.left_mode
        right_label = args.left_mode if swap else args.right_mode

        item_id = f"item_{written:04d}"
        panel_path = os.path.join(args.out_dir, "panels", f"{item_id}.png")
        make_panel(
            lrec["taskA_input"],
            lrec["taskA_output"],
            lrec["taskB_input"],
            left_img,
            right_img,
            panel_path,
            include_gt=include_gt,
            task_b_output=lrec.get("taskB_output"),
        )

        question = (
            "Using the reference output as guidance, which generated output better matches "
            "the target transformation for the query image while preserving relevant image content? "
            "Choose A, B, or Tie."
            if include_gt
            else (
                "Which output better performs the target transformation for the query image "
                "while preserving relevant image content? Choose A, B, or Tie."
            )
        )
        manifest_rows.append(
            {
                "item_id": item_id,
                "panel": panel_path,
                "question": question,
                "choices": ["A", "B", "Tie"],
                "pair_key": lrec["pair_key"],
                "selector": args.selector,
                "source_attempt_index": 0 if args.selector == "first" else None,
                "include_reference": include_gt,
            }
        )
        answer_rows.append(
            {
                "item_id": item_id,
                "combo_id": combo_id,
                "A_method": left_label,
                "B_method": right_label,
                "pair_key": lrec["pair_key"],
                "A_attempt_index": 0 if args.selector == "first" else None,
                "B_attempt_index": 0 if args.selector == "first" else None,
            }
        )
        used_combo_ids.add(combo_id)
        new_combo_ids.append(combo_id)
        written += 1
        added += 1

    overlap = sorted(existing_combo_ids & set(new_combo_ids))
    if overlap:
        raise RuntimeError(f"New human-evaluation items overlap existing items: {overlap}")

    write_jsonl_atomic(manifest_path, manifest_rows)
    write_jsonl_atomic(answer_key_path, answer_rows)
    audit_path = os.path.join(args.out_dir, "selection_audit.json")
    write_json_atomic(
        audit_path,
        {
            "selector": args.selector,
            "strict_first_attempt": args.selector == "first",
            "shared_candidates": len(shared),
            "existing_items": len(existing_combo_ids),
            "existing_items_without_current_strict_attempt_00": len(
                existing_without_strict_attempt_00
            ),
            "existing_combo_ids_without_current_strict_attempt_00": (
                existing_without_strict_attempt_00
            ),
            "appended_items": added,
            "appended_combo_ids": new_combo_ids,
            "existing_appended_overlap": overlap,
            "total_items": written,
        },
    )

    print(f"Wrote {manifest_path}")
    print(f"Wrote {answer_key_path}")
    print(f"Wrote {audit_path}")
    print(
        f"Study now has {written} items ({added} appended); "
        f"skipped {skipped} new shared combos with missing images."
    )
    print(
        "Selection audit: "
        f"existing/new overlap={len(overlap)}; "
        f"existing items without current strict attempt_00="
        f"{len(existing_without_strict_attempt_00)}."
    )


if __name__ == "__main__":
    main()
