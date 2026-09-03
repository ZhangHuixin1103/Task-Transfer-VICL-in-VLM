from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from comparison.datasets import TaskSpec, matching_record_indices
from comparison.prepare_same_task_eval import (
    build_same_task_records,
    build_target_task_records,
    supplement_target_task_records,
    task_counts,
    validate_derivation,
)


def record(task_a: str, task_b: str, index: int) -> dict[str, str]:
    return {
        "taskA_input": f"{task_a}/input/{index}.png",
        "taskA_output": f"{task_a}/output/{index}.png",
        "taskB_input": f"{task_b}/input/{index}.png",
        "taskB_output": f"{task_b}/output/{index}.png",
    }


class SameTaskDerivationTest(unittest.TestCase):
    def test_target_sampling_deduplicates_task_b_globally_across_source_tasks(self):
        source = [
            record("rain", "haze", 0),
            record("rain", "haze", 1),
            record("blur", "haze", 0),
            record("blur", "haze", 3),
        ]
        selected = build_target_task_records(source, max_per_task=100, seed=2026)
        derived = build_same_task_records(selected, source, seed=2026)
        validate_derivation(selected, derived)

        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["taskB_input"] for row in selected}), 3)
        self.assertEqual(len({row["taskB_output"] for row in selected}), 3)
        duplicate = next(row for row in selected if row["taskB_input"].endswith("0.png"))
        self.assertEqual(duplicate["duplicate_source_record_indices"], [0, 2])
        for original, same_task in zip(selected, derived):
            self.assertEqual(same_task["taskB_input"], original["taskB_input"])
            self.assertEqual(same_task["taskB_output"], original["taskB_output"])
            self.assertNotEqual(same_task["taskA_input"], same_task["taskB_input"])
            self.assertNotEqual(same_task["taskA_output"], same_task["taskB_output"])

    def test_wildcard_source_filter_aggregates_one_target_task(self):
        source = [
            record("rain", "haze", 0),
            record("rain", "haze", 1),
            record("blur", "haze", 2),
            record("blur", "haze", 3),
        ]
        selected = build_target_task_records(source)
        task = TaskSpec(
            name="haze",
            eval_json="target.json",
            task_a="*",
            task_b="haze",
        )

        self.assertFalse(task.is_same_task_vicl)
        self.assertTrue(task.is_cross_task)
        self.assertEqual(matching_record_indices(selected, task), [0, 1, 2, 3])

    def test_validation_rejects_repeated_task_b_input_even_if_target_differs(self):
        original = [record("rain", "haze", 0), record("blur", "haze", 1)]
        original[1]["taskB_input"] = original[0]["taskB_input"]
        derived = [
            {
                **row,
                "taskA_input": f"haze/input/demo_{index}.png",
                "taskA_output": f"haze/output/demo_{index}.png",
            }
            for index, row in enumerate(original)
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate Task-B query input"):
            validate_derivation(original, derived)

    def test_balanced_competitor_split_uses_paired_pool_then_task_files(self):
        source = [record("rain", "haze", index) for index in range(2)]
        selected = build_target_task_records(source, max_per_task=4, seed=2026)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                for side in ("input", "output"):
                    path = root / "haze" / side / f"{index}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            paired_source = root / "paired.json"
            paired_source.write_text(
                json.dumps(
                    [
                        {
                            "task": "haze",
                            "input": "haze/input/2.png",
                            "output": "haze/output/2.png",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            augmented = supplement_target_task_records(
                selected,
                data_root=root,
                max_per_task=4,
                seed=2026,
                paired_source=paired_source,
            )
            derived = build_same_task_records(augmented, augmented, seed=2026)
            validate_derivation(
                selected,
                derived,
                max_per_task=4,
                require_exact_competitor_count=True,
            )

        self.assertEqual(task_counts(derived), {"haze": 4})
        self.assertEqual(len({row["taskB_input"] for row in derived}), 4)
        self.assertEqual(
            {row["query_source"] for row in derived},
            {
                "data/dataset/eval_dataset.json",
                "data/dataset/paired.json",
                "data/tasks paired pool",
            },
        )
        for row in derived:
            self.assertNotEqual(row["taskA_input"], row["taskB_input"])
            self.assertNotEqual(row["taskA_output"], row["taskB_output"])


if __name__ == "__main__":
    unittest.main()
