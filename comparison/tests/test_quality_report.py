from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from comparison.quality_report import (
    add_macro_rows,
    load_legacy_pair_summary,
    load_quality,
    validate_rows,
    write_reports,
)


class QualityReportTest(unittest.TestCase):
    def test_combines_standard_and_legacy_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = root / "standard.json"
            standard.write_text(
                json.dumps(
                    {
                        "kind": "image_quality_suite",
                        "adapter": "painter",
                        "conditions": [
                            {
                                "condition": "official",
                                "tasks": [
                                    {
                                        "task": "dehazing",
                                        "count": 100,
                                        "psnr_mean": 20,
                                        "ssim_mean": 0.8,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            legacy = root / "legacy.json"
            legacy.write_text(
                json.dumps(
                    {
                        "a__dehazing": {
                            "num_samples": 40,
                            "avg_psnr": 10,
                            "avg_ssim": 0.4,
                        },
                        "b__dehazing": {
                            "num_samples": 60,
                            "avg_psnr": 20,
                            "avg_ssim": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )
            rows = load_quality(standard)
            rows.extend(load_legacy_pair_summary(f"T2T-VICL={legacy}"))
            methods, tasks = validate_rows(rows)
            add_macro_rows(rows, methods)
            write_reports(rows, methods, tasks, root / "tables")

            legacy_row = next(
                row
                for row in rows
                if row["method"] == "T2T-VICL" and row["task"] == "dehazing"
            )
            self.assertEqual(legacy_row["count"], 100)
            self.assertAlmostEqual(legacy_row["psnr"], 16)
            self.assertAlmostEqual(legacy_row["ssim"], 0.52)
            self.assertTrue((root / "tables/quality_comparison.csv").is_file())
            self.assertTrue((root / "tables/quality_comparison.md").is_file())
            self.assertTrue((root / "tables/quality_comparison_rows.tex").is_file())

    def test_rejects_incomplete_standard_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text(
                json.dumps(
                    {
                        "kind": "image_quality_suite",
                        "adapter": "painter",
                        "conditions": [
                            {
                                "condition": "official",
                                "tasks": [
                                    {
                                        "task": "dehazing",
                                        "count": 99,
                                        "psnr_mean": 20,
                                        "ssim_mean": 0.8,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 100"):
                load_quality(path)


if __name__ == "__main__":
    unittest.main()
