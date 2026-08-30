import argparse
import json
from collections import Counter


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Summarize blind A/B human preference responses."
    )
    parser.add_argument(
        "--answer_key",
        default="data/output/supplementary/human_eval/answer_key.jsonl",
    )
    parser.add_argument(
        "--responses",
        required=True,
        nargs="+",
        help="One or more disjoint JSONL files with item_id and choice (A, B, or Tie).",
    )
    args = parser.parse_args()

    key = {x["item_id"]: x for x in read_jsonl(args.answer_key)}
    counts = Counter()
    by_pair = {}

    seen = set()
    accepted = set()
    invalid = []
    for response_path in args.responses:
        for resp in read_jsonl(response_path):
            item_id = resp["item_id"]
            choice = str(resp["choice"]).strip()
            if item_id in seen:
                raise ValueError(
                    f"Duplicate response for {item_id}; response files must be disjoint."
                )
            seen.add(item_id)
            if item_id not in key or choice not in {"A", "B", "Tie"}:
                invalid.append(item_id)
                continue
            accepted.add(item_id)
            meta = key[item_id]
            pair = meta["pair_key"]
            by_pair.setdefault(pair, Counter())
            if choice == "Tie":
                winner = "Tie"
            else:
                winner = meta[f"{choice}_method"]
            counts[winner] += 1
            by_pair[pair][winner] += 1

    total = sum(counts.values())
    missing = sorted(set(key) - accepted)
    print(
        f"Accepted responses: {len(accepted)}/{len(key)} answer-key items; "
        f"missing={len(missing)}, invalid={len(invalid)}"
    )
    print("Global human preference")
    for label, n in counts.most_common():
        pct = 100 * n / total if total else 0
        print(f"{label}: {n}/{total} ({pct:.1f}%)")
    non_tie_total = total - counts.get("Tie", 0)
    if non_tie_total:
        print("\nPreference excluding ties")
        for label, n in counts.most_common():
            if label != "Tie":
                print(f"{label}: {n}/{non_tie_total} ({100*n/non_tie_total:.1f}%)")
    print("\nBy pair")
    for pair, counter in sorted(by_pair.items()):
        subtotal = sum(counter.values())
        parts = [f"{k}={v} ({100*v/subtotal:.1f}%)" for k, v in counter.most_common()]
        print(f"{pair}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
