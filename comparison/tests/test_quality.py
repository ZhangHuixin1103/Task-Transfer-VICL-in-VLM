from __future__ import annotations

import math
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from comparison.base import InferenceResult, load_dataset_records, vicl_sample_from_records
from comparison.quality import (
    _heldout_record_indices,
    _metric_images,
    psnr_ssim,
    run_quality,
    write_quality,
)


class QualityMetricTest(unittest.TestCase):
    def test_native_metric_protocol_resizes_only_prediction_to_target(self):
        reference = Image.new("RGB", (19, 11), "white")
        prediction = Image.new("RGB", (7, 7), "black")
        target, generated = _metric_images(reference, prediction, None)
        self.assertEqual(target.size, (19, 11))
        self.assertEqual(generated.size, (19, 11))
        reference.close()
        prediction.close()
        target.close()
        generated.close()

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


class QualityPipelineTest(unittest.TestCase):
    def test_embedded_same_task_manifest_uses_its_per_query_demonstration(self):
        class FakeAdapter:
            name = "painter"
            protocol = "test"
            conditions = ("official",)

            def setup(self):
                pass

            def prepare_condition(self, condition):
                pass

            def release_condition(self, condition):
                pass

            def configure_samples(
                self,
                dataset_json,
                demo_input=None,
                demo_output=None,
                record_indices=None,
            ):
                source = load_dataset_records(dataset_json)
                self.records = [source[index] for index in record_indices]
                self.select_sample(0)

            def sample_count(self):
                return len(self.records)

            def select_sample(self, sample_index):
                self.sample = vicl_sample_from_records(self.records, sample_index)

            def run(self, condition):
                with Image.open(root / self.sample.task_b_input) as source:
                    return InferenceResult(output=source.convert("RGB").copy())

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for source_task, value in (("rain", 10), ("blur", 30)):
                for index in range(2):
                    prefix = f"haze/{source_task}_{index}"
                    for suffix, delta in (
                        ("demo_in", 0),
                        ("demo_out", 1),
                        ("query", 2),
                        ("target", 3),
                    ):
                        path = f"{prefix}_{suffix}.png"
                        (root / path).parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(
                            np.full((8, 8, 3), value + index + delta, dtype=np.uint8)
                        ).save(root / path)
                    records.append(
                        {
                            "source_task_a": source_task,
                            "source_task_b": "haze",
                            "taskA_input": f"{prefix}_demo_in.png",
                            "taskA_output": f"{prefix}_demo_out.png",
                            "taskB_input": f"{prefix}_query.png",
                            "taskB_output": f"{prefix}_target.png",
                        }
                    )
            (root / "same.json").write_text(json.dumps(records), encoding="utf-8")
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
                                "name": "rain__haze",
                                "eval_json": "same.json",
                                "task_a": "haze",
                                "task_b": "haze",
                                "source_task_a": "rain",
                                "source_task_b": "haze",
                                "text_prompt": "remove haze",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                task_manifest=manifest,
                tasks=None,
                data_root=root,
                resolution=8,
                steps=1,
                output_dir=root / "quality",
                no_save_outputs=True,
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
            with mock.patch("comparison.quality.build_adapter", return_value=adapter):
                document = run_quality(args)

            task = document["conditions"][0]["tasks"][0]
            self.assertEqual(task["heldout_records"], 2)
            self.assertEqual(task["selected_source_indices"], [0, 1])
            self.assertEqual(adapter.records, records[:2])
            self.assertEqual(adapter.sample.task_a_input, records[1]["taskA_input"])
            self.assertEqual(adapter.sample.task_b_input, records[1]["taskB_input"])

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
            with mock.patch("comparison.quality.build_adapter", return_value=adapter):
                document = run_quality(args)
            result_path = write_quality(document, output)

            task = document["conditions"][0]["tasks"][0]
            self.assertEqual(task["count"], 1)
            self.assertEqual(adapter.sample.task_b_input, "query.png")
            self.assertEqual(adapter.sample.task_b_output, "target.png")
            self.assertTrue((output / "images/official/deblurring/000000.png").is_file())
            self.assertEqual(json.loads(result_path.read_text())["kind"], "image_quality_suite")
            self.assertTrue(adapter.closed)

    def test_resume_expands_sample_limit_without_regenerating_completed_outputs(self):
        class FakeAdapter:
            name = "prompt-diffusion"
            protocol = "test"
            conditions = ("official",)

            def __init__(self):
                self.run_indices = []

            def setup(self):
                pass

            def prepare_condition(self, condition):
                pass

            def release_condition(self, condition):
                pass

            def configure_samples(
                self,
                dataset_json,
                demo_input=None,
                demo_output=None,
                record_indices=None,
            ):
                source = load_dataset_records(dataset_json)
                self.records = [source[index] for index in record_indices]
                self.demo_input = demo_input
                self.demo_output = demo_output
                self.select_sample(0)

            def sample_count(self):
                return len(self.records)

            def select_sample(self, sample_index):
                self.sample_index = sample_index
                self.sample = vicl_sample_from_records(
                    self.records,
                    sample_index,
                    demo_input=self.demo_input,
                    demo_output=self.demo_output,
                )

            def run(self, condition):
                self.run_indices.append(self.sample_index)
                with Image.open(root / self.sample.task_b_input) as source:
                    return InferenceResult(output=source.convert("RGB").copy())

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("demo_in.png", 1), ("demo_out.png", 2)):
                Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(
                    root / name
                )
            pairs = []
            for index in range(3):
                query = f"query_{index}.png"
                target = f"target_{index}.png"
                Image.fromarray(
                    np.full((8, 8, 3), 10 + index, dtype=np.uint8)
                ).save(root / query)
                Image.fromarray(
                    np.full((8, 8, 3), 20 + index, dtype=np.uint8)
                ).save(root / target)
                pairs.append({"image_path": query, "target_path": target})
            (root / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
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

            def arguments(max_samples, resume, reverse_order):
                return SimpleNamespace(
                    task_manifest=manifest,
                    tasks=None,
                    data_root=root,
                    resolution=8,
                    steps=1,
                    output_dir=output,
                    no_save_outputs=False,
                    conditions=["official"],
                    max_samples=max_samples,
                    sampling_seed=2026,
                    resume=resume,
                    reverse_order=reverse_order,
                    seed=3,
                    dtype="fp32",
                    config="config.yaml",
                    cfg_text=3.5,
                    cfg_image=1.25,
                    shape_policy="fixed-square",
                    painter_task="restoration",
                    painter_include_script_loss=False,
                )

            first = FakeAdapter()
            with mock.patch("comparison.quality.build_adapter", return_value=first):
                first_document = run_quality(arguments(1, False, False))
            self.assertEqual(first_document["conditions"][0]["tasks"][0]["count"], 1)
            self.assertEqual(len(first.run_indices), 1)

            second = FakeAdapter()
            with mock.patch("comparison.quality.build_adapter", return_value=second):
                second_document = run_quality(arguments(3, True, True))
            task = second_document["conditions"][0]["tasks"][0]
            self.assertEqual(task["count"], 3)
            self.assertEqual(task["processing_order"], "reverse")
            self.assertEqual(len(second.run_indices), 2)
            self.assertEqual(
                len(list((output / "images/official/deblurring").glob("*.png"))), 3
            )


if __name__ == "__main__":
    unittest.main()
