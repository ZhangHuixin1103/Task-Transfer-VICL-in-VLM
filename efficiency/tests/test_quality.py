from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from efficiency.base import InferenceResult, load_dataset_records, vicl_sample_from_records
from efficiency.quality import (
    _heldout_record_indices,
    psnr_ssim,
    run_quality,
    write_quality,
)
from efficiency.quality_table import fill_table


class QualityMetricTest(unittest.TestCase):
    def test_demonstration_pair_is_excluded_from_heldout_queries(self):
        records = [
            {"image_path": "demo.png", "target_path": "answer.png"},
            {"image_path": "query.png", "target_path": "target.png"},
        ]
        heldout, excluded = _heldout_record_indices(
            records, "demo.png", "answer.png"
        )
        self.assertEqual(heldout, [1])
        self.assertEqual(excluded, [0])

    def test_identical_images_have_unit_ssim(self):
        pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        image = Image.fromarray(pixels)
        psnr, ssim = psnr_ssim(image, image)
        self.assertTrue(math.isinf(psnr))
        self.assertAlmostEqual(ssim, 1.0, places=12)

    def test_metrics_match_skimage_defaults(self):
        try:
            from skimage.metrics import peak_signal_noise_ratio
            from skimage.metrics import structural_similarity
        except ImportError:
            self.skipTest("scikit-image is unavailable in this test environment")
        generator = np.random.default_rng(7)
        reference = generator.integers(0, 256, (31, 29, 3), dtype=np.uint8)
        prediction = generator.integers(0, 256, (31, 29, 3), dtype=np.uint8)
        psnr, ssim = psnr_ssim(
            Image.fromarray(reference), Image.fromarray(prediction)
        )
        self.assertAlmostEqual(
            psnr,
            peak_signal_noise_ratio(reference, prediction, data_range=255),
            places=12,
        )
        self.assertAlmostEqual(
            ssim,
            structural_similarity(
                reference, prediction, data_range=255, channel_axis=-1
            ),
            places=12,
        )


class QualityTableTest(unittest.TestCase):
    def test_fill_table_changes_only_supplied_baseline_column(self):
        template = (
            "    \\multirow{2}{*}{Deraining}\n"
            "    & PSNR$\\uparrow$ & 10.23 & 15.39 & -- & 25.82 \\\\\n"
            "    & SSIM$\\uparrow$ & 0.096 & 0.433 & -- & 0.837 \\\\\n"
        )
        tasks = {"deraining": {"psnr": 20.126, "ssim": 0.7656}}
        rendered = fill_table(
            template,
            {"prompt-diffusion": tasks},
            allow_partial=True,
        )
        self.assertIn("& PSNR$\\uparrow$ & 20.13 & 15.39 & -- & 25.82", rendered)
        self.assertIn("& SSIM$\\uparrow$ & 0.766 & 0.433 & -- & 0.837", rendered)


class QualityPipelineTest(unittest.TestCase):
    def test_quality_pass_saves_outputs_and_records_query_target_direction(self):
        class FakeAdapter:
            name = "prompt-diffusion"
            protocol = "test"
            conditions = ("official",)

            def setup(self):
                self.closed = False

            def prepare_condition(self, condition):
                self.condition = condition

            def release_condition(self, condition):
                self.condition = None

            def configure_samples(
                self,
                dataset_json,
                demo_input=None,
                demo_output=None,
                record_indices=None,
            ):
                self.records = load_dataset_records(dataset_json)
                self.demo_input = demo_input
                self.demo_output = demo_output
                self.select_sample(0)

            def sample_count(self):
                return len(self.records)

            def select_sample(self, sample_index):
                self.sample = vicl_sample_from_records(
                    self.records,
                    sample_index,
                    demo_input=self.demo_input,
                    demo_output=self.demo_output,
                )

            def run(self, condition):
                with Image.open(root / self.sample.task_b_input) as source:
                    return InferenceResult(output=source.convert("RGB").copy())

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (
                ("demo_in.png", 1),
                ("demo_out.png", 2),
                ("query.png", 10),
                ("target.png", 20),
            ):
                Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(
                    root / name
                )
            (root / "pairs.json").write_text(
                json.dumps(
                    [{"image_path": "query.png", "target_path": "target.png"}]
                ),
                encoding="utf-8",
            )
            manifest = root / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "data_root": ".",
                        "benchmark_family": "test",
                        "controlled_protocol": {"resolution": 8},
                        "tasks": [
                            {
                                "name": "deblurring",
                                "eval_json": "pairs.json",
                                "demo_input": "demo_in.png",
                                "demo_output": "demo_out.png",
                                "text_prompt": "remove blur",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "quality"
            args = SimpleNamespace(
                task_manifest=manifest,
                tasks=None,
                data_root=root,
                resolution=8,
                steps=1,
                output_dir=output,
                no_save_outputs=False,
                conditions=["official"],
                max_samples=-1,
                sampling_seed=2026,
                resume=False,
                seed=3,
                dtype="fp32",
                config="config.yaml",
                cfg_text=3.5,
                cfg_image=1.25,
                shape_policy="fixed-square",
                painter_task="restoration",
                painter_include_script_loss=False,
            )
            adapter = FakeAdapter()
            with mock.patch("efficiency.quality.build_adapter", return_value=adapter):
                document = run_quality(args)
            result_path = write_quality(document, output)

            task = document["conditions"][0]["tasks"][0]
            self.assertEqual(task["count"], 1)
            self.assertEqual(adapter.sample.task_b_input, "query.png")
            self.assertEqual(adapter.sample.task_b_output, "target.png")
            self.assertTrue((output / "images/official/deblurring/000000.png").is_file())
            self.assertEqual(json.loads(result_path.read_text())["kind"], "image_quality_suite")
            self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
