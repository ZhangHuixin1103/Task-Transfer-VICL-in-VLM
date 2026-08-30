import argparse
import json
import re
from collections import Counter


TASK_NAME_PATTERNS = {
    "colorization": [r"\bcolori[sz](?:e|ation|ing|ed)?\b"],
    "deblurring": [r"\bdeblur(?:ring|red|s)?\b"],
    "dehazing": [r"\bdehaz(?:e|ing|ed|es)?\b"],
    "demoireing": [r"\bdemoir[eé](?:ing|ed|s)?\b", r"\bmoire removal\b"],
    "denoising": [r"\bdenois(?:e|ing|ed|es)?\b"],
    "deraining": [r"\bderain(?:ing|ed|s)?\b"],
    "harmonization": [r"\bharmoni[sz](?:e|ation|ing|ed)?\b"],
    "inpainting": [r"\binpaint(?:ing|ed|s)?\b"],
    "light_enhancement": [r"\blow[- ]light enhancement\b", r"\blight enhancement\b"],
    "reflection_removal": [r"\breflection removal\b"],
    "shadow_removal": [r"\bshadow removal\b"],
    "style_transfer": [r"\bstyle transfer\b", r"\bstyli[sz](?:e|ation|ing|ed)?\b"],
}


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Audit direct task-name leakage in teacher/generated prompt text."
    )
    parser.add_argument("--dataset", default="data/dataset/train_dataset.json")
    parser.add_argument("--text_field", default="description")
    parser.add_argument(
        "--out", default="data/output/supplementary/prompt_leakage.json"
    )
    args = parser.parse_args()

    data = read_json(args.dataset)
    compiled = {
        task: [re.compile(p, re.IGNORECASE) for p in patterns]
        for task, patterns in TASK_NAME_PATTERNS.items()
    }

    counts = Counter()
    by_pair = Counter()
    hit_terms = Counter()

    for entry in data:
        text = entry.get(args.text_field, "") or ""
        task_a = entry["taskA_input"].split("/")[0]
        task_b = entry["taskB_input"].split("/")[0]
        hits = set()
        for task, patterns in compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    hits.add(task)
                    hit_terms[match.group(0).lower()] += 1

        counts["total"] += 1
        if hits:
            counts["any_task_name_hit"] += 1
        if task_a in hits:
            counts["source_task_name_hit"] += 1
        if task_b in hits:
            counts["target_task_name_hit"] += 1
        if task_a in hits and task_b in hits:
            counts["source_and_target_task_name_hit"] += 1
        by_pair[f"{task_a}__{task_b}"] += int(bool(hits))

    total = counts["total"]
    result = {
        "dataset": args.dataset,
        "text_field": args.text_field,
        "total": total,
        "counts": dict(counts),
        "rates": {
            key: (value / total if total else 0.0)
            for key, value in counts.items()
            if key != "total"
        },
        "top_hit_terms": hit_terms.most_common(50),
        "pair_hit_counts": dict(by_pair),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["rates"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
