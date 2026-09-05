from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from comparison.adapters.common import require_checkpoint_file, require_model_file
from comparison.base import (
    ComparisonAdapter,
    InferenceResult,
    load_dataset_records,
    load_vicl_sample,
)
from comparison.benchmark import (
    _external_repository,
    _repository_revision,
    parser,
    run_benchmark,
)
from comparison.prompting import RELATION_REQUEST
from comparison.suite import parser as suite_parser, run_suite


TASK_TRANSFER = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT_VALUE = os.environ.get("VICL_EXTERNAL_ROOT")
EXTERNAL_ROOT = (
    Path(EXTERNAL_ROOT_VALUE).expanduser().resolve()
    if EXTERNAL_ROOT_VALUE
    else TASK_TRANSFER / "third_party"
)


def definitions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


class InterfaceContractTest(unittest.TestCase):
    def test_checkpoint_fallback_recovers_old_flattened_readme_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "weights" / "Painter" / "painter_vit_large.pth"
            nested.parent.mkdir(parents=True)
            nested.touch()
            resolved = require_checkpoint_file(
                str(root / "weights" / "painter_vit_large.pth"),
                "Painter",
                nested,
            )
        self.assertEqual(Path(resolved), nested.resolve())

    def test_checkpoint_error_lists_the_requested_path(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.ckpt"
            with self.assertRaisesRegex(FileNotFoundError, str(missing)):
                require_checkpoint_file(str(missing), "InstructDiffusion")

    def test_required_model_file_accepts_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "model.yaml"
            expected.touch()
            resolved = require_model_file(
                None, "MAE-VQGAN VQGAN config", expected
            )
        self.assertEqual(Path(resolved), expected.resolve())

    def test_t2t_import_path_does_not_import_external_adapters(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import comparison.adapters; "
                    "assert 'comparison.adapters.external' not in sys.modules"
                ),
            ],
            cwd=TASK_TRANSFER,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_paired_image_json_maps_to_vicl_sample_with_explicit_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.json"
            path.write_text(
                json.dumps(
                    [{"image_path": "query.png", "target_path": "answer.png"}]
                ),
                encoding="utf-8",
            )
            sample = load_vicl_sample(
                path,
                0,
                demo_input="demo_input.png",
                demo_output="demo_output.png",
            )
        self.assertEqual(sample.task_a_input, "demo_input.png")
        self.assertEqual(sample.task_a_output, "demo_output.png")
        self.assertEqual(sample.task_b_input, "query.png")
        self.assertEqual(sample.task_b_output, "answer.png")

    def test_t2t_adapter_caches_json_and_selects_task_subset_in_memory(self):
        from comparison.adapters.t2t import T2TVICLAdapter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "taskA_input": "a/input/0.png",
                            "taskA_output": "a/output/0.png",
                            "taskB_input": "b/input/0.png",
                            "taskB_output": "b/output/0.png",
                        },
                        {
                            "taskA_input": "c/input/1.png",
                            "taskA_output": "c/output/1.png",
                            "taskB_input": "d/input/1.png",
                            "taskB_output": "d/output/1.png",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            adapter = T2TVICLAdapter(
                project_root=TASK_TRANSFER,
                backend="qwen",
                requested_conditions=("fixed",),
                dataset_json=path,
                data_root=Path(directory),
            )
            with mock.patch(
                "comparison.adapters.t2t.load_dataset_records",
                wraps=load_dataset_records,
            ) as loader:
                adapter.configure_samples(path, record_indices=[1])
                adapter.select_sample(0)
                self.assertEqual(adapter.sample.task_a_input, "c/input/1.png")
                adapter.configure_samples(path, record_indices=[0])
                adapter.select_sample(0)
                self.assertEqual(adapter.sample.task_b_input, "b/input/0.png")
                self.assertEqual(loader.call_count, 1)

    def assert_function_prefix(self, path: Path, name: str, expected):
        node = definitions(path).get(name)
        self.assertIsInstance(node, ast.FunctionDef, f"{name} missing in {path}")
        actual = [argument.arg for argument in node.args.args]
        self.assertEqual(actual[: len(expected)], list(expected))

    def assert_function_defaults(self, path: Path, name: str, expected):
        node = definitions(path).get(name)
        self.assertIsInstance(node, ast.FunctionDef, f"{name} missing in {path}")
        names = [argument.arg for argument in node.args.args]
        defaults = [ast.literal_eval(value) for value in node.args.defaults]
        actual = dict(zip(names[-len(defaults) :], defaults))
        for argument, value in expected.items():
            self.assertIn(argument, actual)
            self.assertEqual(actual[argument], value)

    def test_current_t2t_backend_contracts(self):
        self.assert_function_prefix(
            TASK_TRANSFER / "eval_qwen.py",
            "generate_image_qwen",
            ("pipeline", "img_paths", "text_prompt"),
        )
        self.assert_function_prefix(
            TASK_TRANSFER / "eval_flux.py",
            "generate_image_flux",
            ("pipe", "taskA_in", "taskA_out", "taskB_in", "text_prompt"),
        )
        self.assert_function_prefix(
            TASK_TRANSFER / "eval_omnigen.py",
            "generate_image_omnigen",
            ("pipe", "taskA_in", "taskA_out", "taskB_in", "text_prompt"),
        )
        self.assert_function_prefix(
            TASK_TRANSFER / "eval_firered.py",
            "generate_image_firered",
            ("pipe", "taskA_in", "taskA_out", "taskB_in", "text_prompt"),
        )
        self.assert_function_prefix(
            TASK_TRANSFER / "eval.py",
            "generate_text_prompt",
            (
                "taskA_input",
                "taskA_output",
                "taskB_input",
                "model",
                "processor",
            ),
        )

    def test_t2t_comparison_options_preserve_original_defaults(self):
        self.assert_function_defaults(
            TASK_TRANSFER / "eval_qwen.py",
            "generate_image_qwen",
            {
                "seed": 42,
                "generator_device": None,
                "num_inference_steps": 40,
                "input_resolution": None,
                "height": None,
                "width": None,
            },
        )
        self.assert_function_defaults(
            TASK_TRANSFER / "eval_flux.py",
            "generate_image_flux",
            {
                "seed": 42,
                "num_inference_steps": 30,
                "input_resolution": None,
                "height": None,
                "width": None,
            },
        )
        self.assert_function_defaults(
            TASK_TRANSFER / "eval_omnigen.py",
            "generate_image_omnigen",
            {
                # The legacy evaluation path did not pass a generator. The
                # comparison adapter supplies its seed explicitly.
                "seed": None,
                "num_inference_steps": 50,
                "input_resolution": None,
                "height": 1024,
                "width": 1024,
            },
        )
        self.assert_function_defaults(
            TASK_TRANSFER / "eval_firered.py",
            "generate_image_firered",
            {
                "seed": 42,
                "true_cfg_scale": 4.0,
                "num_inference_steps": 40,
                "input_resolution": None,
                "height": None,
                "width": None,
            },
        )

    def test_lightweight_prompt_protocol_matches_original_eval(self):
        tree = ast.parse(
            (TASK_TRANSFER / "eval.py").read_text(encoding="utf-8"),
            filename="eval.py",
        )
        prompts = [
            value.value
            for value in ast.walk(tree)
            if isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and "Below are two vision tasks, A and B" in value.value
        ]
        self.assertIn(RELATION_REQUEST, prompts)

    def test_external_repository_defaults_to_vendored_project_sources(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _external_repository("Painter", "Painter"),
                (TASK_TRANSFER / "third_party/Painter/Painter").resolve(),
            )

        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "Painter" / "Painter"
            expected.mkdir(parents=True)
            with mock.patch.dict(
                os.environ, {"VICL_EXTERNAL_ROOT": directory}, clear=False
            ):
                self.assertEqual(
                    _external_repository("Painter", "Painter"), expected.resolve()
                )

    def test_vendored_repository_revision_uses_sources_manifest(self):
        self.assertEqual(
            _repository_revision(TASK_TRANSFER / "third_party/Painter/Painter"),
            "cfec84587da2a2cc4e0b9acebe6fba1666e66477",
        )

    def test_new_official_baseline_entry_points_exist(self):
        mae_utils = definitions(
            EXTERNAL_ROOT / "MAE-VQGAN" / "evaluate" / "mae_utils.py"
        )
        self.assertIn("prepare_model", mae_utils)
        self.assertIn("generate_mask_for_evaluation", mae_utils)
        self.assertIn("generate_image", mae_utils)

        prompt_gip = definitions(
            EXTERNAL_ROOT / "PromptGIP" / "models_mae_PromptGIP_CNN_Head.py"
        )
        self.assertIn("mae_vit_large_patch16_dec512d8b_input256", prompt_gip)
        prompt_gip_source = (
            EXTERNAL_ROOT / "PromptGIP" / "models_mae_PromptGIP_CNN_Head.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "mae_vit_large_patch16_input256 = "
            "mae_vit_large_patch16_dec512d8b_input256",
            prompt_gip_source,
        )

        visualcloze = definitions(EXTERNAL_ROOT / "VisualCloze" / "visualcloze.py")
        model = visualcloze.get("VisualClozeModel")
        self.assertIsInstance(model, ast.ClassDef)
        methods = {
            node.name for node in model.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("set_grid_size", methods)
        self.assertIn("process_images", methods)

    def test_new_baselines_are_exposed_by_benchmark_and_quality_parsers(self):
        benchmark_choices = next(
            action.choices
            for action in parser()._actions
            if action.dest == "adapter"
        )
        for name in ("mae-vqgan", "prompt-gip", "visualcloze"):
            self.assertIn(name, benchmark_choices)

        from comparison.quality import parser as quality_parser

        quality_choices = next(
            action.choices
            for action in quality_parser()._actions
            if action.dest == "adapter"
        )
        for name in (
            "t2t-qwen",
            "mae-vqgan",
            "prompt-gip",
            "visualcloze",
        ):
            self.assertIn(name, quality_choices)

    def test_external_adapter_defaults_match_released_inference(self):
        from comparison.adapters.external import (
            InstructDiffusionAdapter,
            PromptDiffusionAdapter,
            VisualClozeAdapter,
        )

        common = {
            "dataset_json": TASK_TRANSFER / "data/dataset/eval_dataset_same_task.json",
            "data_root": TASK_TRANSFER / "data/tasks",
        }
        prompt_diffusion = PromptDiffusionAdapter(
            repository=EXTERNAL_ROOT / "Prompt-Diffusion",
            checkpoint="network-step=04999.ckpt",
            **common,
        )
        visualcloze = VisualClozeAdapter(
            repository=EXTERNAL_ROOT / "VisualCloze",
            checkpoint="visualcloze-384-lora.pth",
            **common,
        )
        instruct = InstructDiffusionAdapter(
            repository=EXTERNAL_ROOT / "InstructDiffusion",
            checkpoint="v1-5-pruned-emaonly-adaption-task.ckpt",
            **common,
        )
        self.assertEqual(prompt_diffusion.steps, 100)
        self.assertEqual(prompt_diffusion.seed, 1)
        self.assertEqual(prompt_diffusion.resolution, 512)
        self.assertEqual(prompt_diffusion.guidance_scale, 9.0)
        self.assertEqual(prompt_diffusion.strength, 1.0)
        self.assertEqual(prompt_diffusion.eta, 0.0)
        self.assertEqual(prompt_diffusion.dtype, torch.float32)
        self.assertEqual(visualcloze.steps, 30)
        self.assertEqual(visualcloze.seed, 0)
        self.assertEqual(instruct.steps, 100)
        self.assertEqual(instruct.cfg_text, 5.0)
        self.assertEqual(instruct.cfg_image, 1.25)

    def test_mae_vqgan_normalizes_after_building_the_official_canvas(self):
        from comparison.adapters.external import MAEVQGANAdapter

        class FakeUtils:
            canvas = None

            @staticmethod
            def generate_mask_for_evaluation():
                return torch.arange(4), 3

            @classmethod
            def generate_image(cls, canvas, *args, **kwargs):
                cls.canvas = canvas.detach().cpu()
                return None, np.zeros((224, 224, 3), dtype=np.uint8), None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                "a/input/0.png",
                "a/output/0.png",
                "b/input/1.png",
                "b/output/1.png",
            ]
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (20, 20), "white").save(path)
            dataset = root / "eval.json"
            dataset.write_text(
                json.dumps(
                    [
                        dict(
                            zip(
                                (
                                    "taskA_input",
                                    "taskA_output",
                                    "taskB_input",
                                    "taskB_output",
                                ),
                                paths,
                            )
                        )
                    ]
                ),
                encoding="utf-8",
            )
            adapter = MAEVQGANAdapter(
                repository=root,
                dataset_json=dataset,
                data_root=root,
                checkpoint="checkpoint.pth",
                device="cpu",
            )
            adapter.configure_samples(dataset)
            adapter.model = object()
            adapter.mae_utils = FakeUtils
            result = adapter.run("official")

        self.assertEqual(result.output.size, (111, 111))
        expected_gap = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )
        expected_white = (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
            [0.229, 0.224, 0.225]
        )
        torch.testing.assert_close(FakeUtils.canvas[0, :, 111, 111], expected_gap)
        torch.testing.assert_close(FakeUtils.canvas[0, :, 0, 0], expected_white)
        result.output.close()

    def test_mae_vqgan_loads_vqgan_assets_beside_main_checkpoint(self):
        from comparison.adapters.external import MAEVQGANAdapter

        missing = object()
        original_np_float = np.__dict__.get("float", missing)

        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = torch.nn.Identity()
                self.num_heads = 1
                self.scale = 1.0
                self.proj = torch.nn.Identity()
                self.proj_drop = torch.nn.Identity()
                self.q_norm = torch.nn.Identity()
                self.k_norm = torch.nn.Identity()

        class ModernTimmBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = torch.nn.Identity()
                self.attn = FakeAttention()
                self.drop_path1 = torch.nn.Identity()
                self.drop_path2 = torch.nn.Identity()
                self.ls1 = torch.nn.Identity()
                self.ls2 = torch.nn.Identity()
                self.norm2 = torch.nn.Identity()
                self.mlp = torch.nn.Identity()

        class FakeModel:
            def __init__(self):
                self.patch_embed = SimpleNamespace(patch_size=(16, 16))
                self.pos_embed = object()
                self.cls_token = object()
                self.blocks = []
                self.norm = object()
                self.decoder_embed = object()
                self.mask_token = object()
                self.decoder_pos_embed = object()
                self.decoder_blocks = [ModernTimmBlock()]
                self.decoder_norm = object()
                self.decoder_pred = object()
                self.vae = SimpleNamespace(
                    quantize=SimpleNamespace(get_codebook_entry=lambda *args: None),
                    decode=lambda *args: None,
                )
                self.unpatchify = lambda *args: None

            def eval(self):
                return self

            def to(self, *args, **kwargs):
                return self

        class FakeModelsMAE:
            call = None

            @classmethod
            def get_vq_model(cls, config_path=None, ckpt_path=None):
                cls.call = (config_path, ckpt_path)
                return object()

        class FakeUtils:
            models_mae = FakeModelsMAE

            @staticmethod
            def prepare_model(*args, **kwargs):
                assert "float" in np.__dict__
                FakeModelsMAE.get_vq_model()
                return FakeModel()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            weights = root / "weights/MAE-VQGAN"
            weights.mkdir(parents=True)
            checkpoint = weights / "checkpoint-3400.pth"
            config = weights / "model.yaml"
            vqgan_checkpoint = weights / "last.ckpt"
            for path in (checkpoint, config, vqgan_checkpoint):
                path.touch()
            dataset = root / "eval.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "taskA_input": "demo-in.png",
                            "taskA_output": "demo-out.png",
                            "taskB_input": "query.png",
                            "taskB_output": "target.png",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            adapter = MAEVQGANAdapter(
                repository=root / "source",
                dataset_json=dataset,
                data_root=root,
                checkpoint=str(checkpoint),
                device="cpu",
            )
            with mock.patch(
                "comparison.adapters.external.import_from_root",
                return_value=FakeUtils,
            ):
                adapter.setup()

        self.assertIs(np.__dict__.get("float", missing), original_np_float)
        self.assertEqual(
            FakeModelsMAE.call,
            (str(config.resolve()), str(vqgan_checkpoint.resolve())),
        )
        self.assertEqual(adapter.runtime_compatibility["legacy_drop_path_aliases"], [0])

    def test_mae_vqgan_runtime_rejects_unexpected_new_timm_semantics(self):
        from comparison.adapters.external import prepare_mae_vqgan_runtime

        class NonIdentityDropPath(torch.nn.Module):
            drop_prob = 0.1

            def forward(self, value):
                return value

        block = SimpleNamespace(
            norm1=object(),
            attn=SimpleNamespace(
                qkv=object(),
                num_heads=1,
                scale=1.0,
                proj=object(),
                proj_drop=object(),
                q_norm=torch.nn.Identity(),
                k_norm=torch.nn.Identity(),
            ),
            drop_path1=NonIdentityDropPath(),
            drop_path2=NonIdentityDropPath(),
            ls1=torch.nn.Identity(),
            ls2=torch.nn.Identity(),
            norm2=object(),
            mlp=object(),
        )
        model = SimpleNamespace(
            patch_embed=SimpleNamespace(patch_size=(16, 16)),
            pos_embed=object(),
            cls_token=object(),
            blocks=[],
            norm=object(),
            decoder_embed=object(),
            mask_token=object(),
            decoder_pos_embed=object(),
            decoder_blocks=[block],
            decoder_norm=object(),
            decoder_pred=object(),
            vae=SimpleNamespace(
                quantize=SimpleNamespace(get_codebook_entry=lambda *args: None),
                decode=lambda *args: None,
            ),
            unpatchify=lambda *args: None,
        )
        with self.assertRaisesRegex(RuntimeError, "non-default"):
            prepare_mae_vqgan_runtime(model)

    def test_visualcloze_reports_incompatible_diffusers_environment(self):
        from comparison.adapters.external import VisualClozeAdapter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "visualcloze-384-lora.pth"
            checkpoint.touch()
            dataset = root / "eval.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "taskA_input": "demo-in.png",
                            "taskA_output": "demo-out.png",
                            "taskB_input": "query.png",
                            "taskB_output": "target.png",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            adapter = VisualClozeAdapter(
                repository=root,
                dataset_json=dataset,
                data_root=root,
                checkpoint=str(checkpoint),
            )
            with mock.patch(
                "comparison.adapters.external.import_from_root",
                side_effect=RuntimeError("attention_dispatch infer_schema(func)"),
            ), mock.patch(
                "comparison.preflight.inspect_visualcloze_environment",
                return_value={"status": "pass", "errors": []},
            ):
                with self.assertRaisesRegex(RuntimeError, "dedicated VisualCloze"):
                    adapter.setup()

    def test_visualcloze_preflight_rejects_bad_versions_before_import(self):
        from comparison.preflight import inspect_visualcloze_environment

        versions = {
            "numpy": "2.2.6",
            "torch": "2.1.0",
            "torchvision": "0.16.0",
            "diffusers": "0.32.1",
            "transformers": "4.47.1",
            "accelerate": "1.2.1",
            "flash-attn": "2.7.2.post1",
            "opencv-python": "4.12.0.88",
        }
        with mock.patch(
            "comparison.preflight._version", side_effect=versions.get
        ):
            report = inspect_visualcloze_environment()

        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("NumPy must be 1.x" in error for error in report["errors"]))
        self.assertTrue(any("Diffusers must be 0.31.0" in error for error in report["errors"]))

    def test_prompt_diffusion_uses_original_cldm_conditioning_and_aligns_demo(self):
        from comparison.adapters.external import PromptDiffusionAdapter

        class FakeCV2:
            INTER_LINEAR = 1

            @staticmethod
            def resize(array, size, interpolation):
                return np.asarray(Image.fromarray(array).resize(size))

        class FakeEinops:
            @staticmethod
            def rearrange(tensor, pattern):
                if pattern == "b h w c -> b c h w":
                    return tensor.permute(0, 3, 1, 2)
                if pattern == "b c h w -> b h w c":
                    return tensor.permute(0, 2, 3, 1)
                raise AssertionError(pattern)

        class FakeModel:
            control_scales = None

            @staticmethod
            def get_learned_conditioning(prompts):
                return tuple(prompts)

            @staticmethod
            def decode_first_stage(samples):
                return torch.zeros(
                    (1, 3, samples.shape[-2] * 8, samples.shape[-1] * 8)
                )

        class FakeSampler:
            call = None

            @classmethod
            def sample(cls, *args, **kwargs):
                cls.call = (args, kwargs)
                shape = args[2]
                return torch.zeros((1, *shape)), {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "demo_in.png": (19, 17),
                "demo_out.png": (17, 17),
                "query.png": (31, 17),
                "target.png": (31, 17),
            }
            for name, size in paths.items():
                Image.new("RGB", size, "white").save(root / name)
            dataset = root / "eval.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "taskA_input": "demo_in.png",
                            "taskA_output": "demo_out.png",
                            "taskB_input": "query.png",
                            "taskB_output": "target.png",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            adapter = PromptDiffusionAdapter(
                repository=root,
                dataset_json=dataset,
                data_root=root,
                checkpoint="network-step=04999.ckpt",
            )
            adapter.device = "cpu"
            adapter.model = FakeModel()
            adapter.sampler = FakeSampler()
            adapter.prompt_diffusion_config = SimpleNamespace(save_memory=False)
            adapter.cv2 = FakeCV2
            adapter.einops = FakeEinops
            adapter.hwc3 = lambda array: array

            def resize_image(array, resolution):
                width = 896 if array.shape[1] > array.shape[0] else 512
                return np.zeros((512, width, 3), dtype=np.uint8)

            adapter.resize_image = resize_image
            adapter.configure_samples(dataset)
            result = adapter.run("official")

        args, kwargs = FakeSampler.call
        self.assertEqual(args[:3], (100, 1, (4, 64, 112)))
        self.assertEqual(kwargs["eta"], 0.0)
        self.assertEqual(kwargs["unconditional_guidance_scale"], 9.0)
        self.assertEqual(kwargs["unconditional_conditioning"]["example_pair"][0].shape[1], 6)
        self.assertEqual(args[3]["query"][0].shape, (1, 3, 512, 896))
        self.assertEqual(result.output.size, (896, 512))
        self.assertTrue(result.metadata["demo_target_shape_adjusted"])
        result.output.close()

    def test_prompt_diffusion_only_ignores_legacy_clip_position_ids(self):
        from comparison.adapters.external import (
            PROMPT_DIFFUSION_LEGACY_POSITION_IDS,
            load_prompt_diffusion_checkpoint,
        )

        model = torch.nn.Linear(2, 2)
        state_dict = dict(model.state_dict())
        state_dict[PROMPT_DIFFUSION_LEGACY_POSITION_IDS] = torch.arange(77)[None]
        ignored = load_prompt_diffusion_checkpoint(model, state_dict)
        self.assertEqual(ignored, [PROMPT_DIFFUSION_LEGACY_POSITION_IDS])

        state_dict["unexpected.real_weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected.real_weight"):
            load_prompt_diffusion_checkpoint(model, state_dict)

    def test_painter_restores_the_original_query_size(self):
        from comparison.adapters.external import PainterAdapter

        class FakePainter:
            patch_embed = SimpleNamespace(num_patches=4)
            blocks = []

            @staticmethod
            def forward_encoder(*args):
                return []

            @staticmethod
            def forward_decoder(latent):
                return torch.zeros((1, 3, 896, 448))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = [(20, 20), (20, 20), (31, 17), (31, 17)]
            paths = [
                "a/input/0.png",
                "a/output/0.png",
                "b/input/1.png",
                "b/output/1.png",
            ]
            for relative, size in zip(paths, sizes):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", size, "white").save(path)
            dataset = root / "eval.json"
            dataset.write_text(
                json.dumps(
                    [
                        dict(
                            zip(
                                (
                                    "taskA_input",
                                    "taskA_output",
                                    "taskB_input",
                                    "taskB_output",
                                ),
                                paths,
                            )
                        )
                    ]
                ),
                encoding="utf-8",
            )
            adapter = PainterAdapter(
                repository=root,
                dataset_json=dataset,
                data_root=root,
                device="cpu",
            )
            adapter.configure_samples(dataset)
            adapter.model = FakePainter()
            result = adapter.run("official")

        self.assertEqual(result.output.size, (31, 17))
        result.output.close()

    def test_parallel_launcher_assigns_one_visible_gpu(self):
        from types import SimpleNamespace

        from comparison.launch_t2t import Job, _child_environment

        args = SimpleNamespace(sampling_seed=2026, cpu_threads_per_job=4)
        job = Job("t2t-qwen", "3", sys.executable, "model")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            environment = _child_environment(args, job)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(environment["OMP_NUM_THREADS"], "4")
        self.assertEqual(environment["MKL_NUM_THREADS"], "4")

    def test_parallel_launcher_runs_jobs_with_isolated_gpu_visibility(self):
        from comparison.launch_t2t import main as launch_main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fake_python"
            executable.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$CUDA_VISIBLE_DEVICES\"\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            output = root / "output"
            launch_main(
                [
                    "--job",
                    "t2t-qwen",
                    "1",
                    str(executable),
                    "qwen-model",
                    "--job",
                    "t2t-flux2",
                    "3",
                    str(executable),
                    "flux-model",
                    "--conditions",
                    "fixed",
                    "--output-root",
                    str(output),
                ]
            )
            manifest = json.loads((output / "launch_manifest.json").read_text())

            self.assertEqual(
                (output / "logs" / "t2t-qwen.log").read_text().strip(), "1"
            )
            self.assertEqual(
                (output / "logs" / "t2t-flux2.log").read_text().strip(), "3"
            )
            self.assertEqual(
                [job["return_code"] for job in manifest["parallel_jobs"]], [0, 0]
            )

    def test_external_entry_points_exist(self):
        painter = definitions(
            EXTERNAL_ROOT / "Painter" / "Painter" / "models_painter.py"
        )
        self.assertIn(
            "painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1", painter
        )
        prompt_model = definitions(
            EXTERNAL_ROOT / "Prompt-Diffusion" / "cldm" / "model.py"
        )
        prompt_sampler = definitions(
            EXTERNAL_ROOT / "Prompt-Diffusion" / "cldm" / "ddim_hacked.py"
        )
        self.assertIn("create_model", prompt_model)
        self.assertIn("load_state_dict", prompt_model)
        self.assertIn("DDIMSampler", prompt_sampler)
        instruct = definitions(EXTERNAL_ROOT / "InstructDiffusion" / "edit_cli.py")
        self.assertIn("CFGDenoiser", instruct)
        self.assertIn("load_model_from_config", instruct)
    def test_condition_resources_are_isolated(self):
        class LifecycleAdapter(ComparisonAdapter):
            name = "lifecycle"
            protocol = "test"

            def __init__(self):
                self.editor = torch.nn.Linear(2, 2)
                self.prompt = None
                self.events = []

            @property
            def conditions(self):
                return ("fixed", "ours")

            def setup(self):
                self.events.append("setup")

            def prepare_condition(self, condition):
                self.events.append(f"prepare:{condition}")
                if condition == "ours":
                    self.prompt = torch.nn.Linear(2, 2)
                else:
                    self.assert_prompt_absent()

            def assert_prompt_absent(self):
                if self.prompt is not None:
                    raise AssertionError("prompt resource leaked into Fixed")

            def release_condition(self, condition):
                self.events.append(f"release:{condition}")
                if condition == "ours":
                    self.prompt = None

            def run(self, condition):
                if condition == "fixed":
                    self.assert_prompt_absent()
                return InferenceResult(output=torch.ones(1))

            def parameter_components(self, condition):
                components = {"editor": self.editor}
                if condition == "ours":
                    components["prompt"] = self.prompt
                return components

            def close(self):
                self.events.append("close")

        adapter = LifecycleAdapter()
        args = parser().parse_args(
            [
                "--adapter",
                "toy",
                "--conditions",
                "fixed",
                "ours",
                "--parameters-only",
            ]
        )
        with mock.patch("comparison.benchmark.build_adapter", return_value=adapter):
            document = run_benchmark(args)
        self.assertEqual(
            adapter.events,
            [
                "setup",
                "prepare:fixed",
                "release:fixed",
                "prepare:ours",
                "release:ours",
                "close",
            ],
        )
        fixed, ours = document["conditions"]
        self.assertLess(
            fixed["parameters"]["unique_total"]["total"],
            ours["parameters"]["unique_total"]["total"],
        )

    def test_multi_task_suite_uses_distinct_queries_and_builds_macro_row(self):
        class DatasetAdapter(ComparisonAdapter):
            name = "dataset-test"
            protocol = "test"

            def __init__(self):
                self.model = torch.nn.Linear(2, 2)
                self.records = []
                self.index = None
                self.visited = []

            @property
            def conditions(self):
                return ("official",)

            def setup(self):
                pass

            def configure_samples(self, dataset_json, **kwargs):
                self.records = json.loads(Path(dataset_json).read_text())

            def sample_count(self):
                return len(self.records)

            def select_sample(self, sample_index):
                self.index = sample_index

            def run(self, condition):
                self.visited.append(self.index)
                return InferenceResult(output=torch.ones(1))

            def parameter_components(self, condition):
                return {"model": self.model}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "pairs.json").write_text(
                json.dumps(
                    [
                        {"image_path": f"input_{index}.png", "target_path": f"target_{index}.png"}
                        for index in range(4)
                    ]
                ),
                encoding="utf-8",
            )
            manifest = root / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "data_root": "data",
                        "controlled_protocol": {"resolution": 448},
                        "tasks": [
                            {
                                "name": "task",
                                "eval_json": "pairs.json",
                                "demo_input": "demo_in.png",
                                "demo_output": "demo_out.png",
                                "text_prompt": "test",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = suite_parser().parse_args(
                [
                    "--adapter",
                    "prompt-diffusion",
                    "--conditions",
                    "official",
                    "--task-manifest",
                    str(manifest),
                    "--data-root",
                    str(data_root),
                    "--max-samples",
                    "3",
                    "--warmup",
                    "1",
                ]
            )
            adapter = DatasetAdapter()
            with mock.patch("comparison.suite.build_adapter", return_value=adapter):
                document = run_suite(args)

        task_result = document["conditions"][0]["tasks"][0]
        measured = task_result["latency"]["measured_indices"]
        warmup = task_result["latency"]["warmup_indices"]
        self.assertEqual(len(measured), 3)
        self.assertEqual(len(set(measured)), 3)
        self.assertEqual(len(warmup), 1)
        self.assertTrue(set(warmup).isdisjoint(measured))
        self.assertEqual(task_result["warmup_policy"], "disjoint_from_measured")
        self.assertEqual(document["conditions"][0]["aggregate"]["tasks"], 1)

    def test_suite_reuses_warmup_only_when_every_query_is_measured(self):
        from comparison.suite import _select_warmup_indices

        warmup, policy = _select_warmup_indices(
            count=3, measured_indices=[0, 1, 2], warmup=2, seed=2026
        )
        self.assertEqual(warmup, [0, 1])
        self.assertEqual(
            policy, "reuses_measured_indices_because_the_full_split_is_measured"
        )

    def test_suite_resume_expands_limit_and_measures_missing_indices_in_reverse(self):
        class DatasetAdapter(ComparisonAdapter):
            name = "dataset-resume-test"
            protocol = "test"

            def __init__(self):
                self.model = torch.nn.Linear(2, 2)
                self.records = []
                self.index = 0
                self.visited = []

            @property
            def conditions(self):
                return ("official",)

            def setup(self):
                pass

            def configure_samples(self, dataset_json, **kwargs):
                self.records = json.loads(Path(dataset_json).read_text())

            def sample_count(self):
                return len(self.records)

            def select_sample(self, sample_index):
                self.index = sample_index

            def run(self, condition):
                self.visited.append(self.index)
                return InferenceResult(output=torch.ones(1))

            def parameter_components(self, condition):
                return {"model": self.model}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            records = [
                {"image_path": f"input_{index}.png", "target_path": f"target_{index}.png"}
                for index in range(4)
            ]
            (data_root / "pairs.json").write_text(
                json.dumps(records), encoding="utf-8"
            )
            manifest = root / "tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "data_root": "data",
                        "controlled_protocol": {"resolution": 448},
                        "tasks": [
                            {
                                "name": "task",
                                "eval_json": "pairs.json",
                                "demo_input": "demo_in.png",
                                "demo_output": "demo_out.png",
                                "text_prompt": "test",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            common = [
                "--adapter",
                "prompt-diffusion",
                "--conditions",
                "official",
                "--task-manifest",
                str(manifest),
                "--data-root",
                str(data_root),
                "--warmup",
                "0",
            ]
            first_args = suite_parser().parse_args([*common, "--max-samples", "1"])
            first_adapter = DatasetAdapter()
            with mock.patch(
                "comparison.suite.build_adapter", return_value=first_adapter
            ):
                first_document = run_suite(first_args)
            resume_path = root / "resume.json"
            resume_path.write_text(json.dumps(first_document), encoding="utf-8")

            second_args = suite_parser().parse_args(
                [
                    *common,
                    "--max-samples",
                    "4",
                    "--resume-from",
                    str(resume_path),
                    "--reverse-order",
                ]
            )
            second_adapter = DatasetAdapter()
            with mock.patch(
                "comparison.suite.build_adapter", return_value=second_adapter
            ):
                second_document = run_suite(second_args)

        first_latency = first_document["conditions"][0]["tasks"][0]["latency"]
        latency = second_document["conditions"][0]["tasks"][0]["latency"]
        reused = first_latency["measured_indices"]
        expected_new = list(reversed([index for index in range(4) if index not in reused]))
        self.assertEqual(second_adapter.visited, expected_new)
        self.assertEqual(latency["end_to_end"]["count"], 4)
        self.assertEqual(latency["measured_indices"], [0, 1, 2, 3])
        self.assertEqual(latency["resume"]["reused_indices"], reused)
        self.assertEqual(latency["resume"]["newly_measured_indices"], expected_new)
        self.assertTrue(latency["resume"]["cross_process_reuse"])

    def test_cross_task_suite_filters_directional_records_before_sampling(self):
        class FilterAdapter(ComparisonAdapter):
            name = "t2t-filter-test"
            protocol = "test"

            def __init__(self):
                self.model = torch.nn.Linear(2, 2)
                self.records = []
                self.selected_source_indices = None
                self.index = 0

            @property
            def conditions(self):
                return ("official",)

            def setup(self):
                pass

            def configure_samples(self, dataset_json, record_indices=None, **kwargs):
                source = json.loads(Path(dataset_json).read_text())
                self.selected_source_indices = record_indices
                self.records = (
                    source
                    if record_indices is None
                    else [source[index] for index in record_indices]
                )

            def sample_count(self):
                return len(self.records)

            def select_sample(self, sample_index):
                self.index = sample_index

            def run(self, condition):
                return InferenceResult(
                    output=torch.ones(1),
                    metadata={"taskB_input": self.records[self.index]["taskB_input"]},
                )

            def parameter_components(self, condition):
                return {"model": self.model}

        def record(task_a, task_b, index):
            return {
                "taskA_input": f"{task_a}/input/{index}.png",
                "taskA_output": f"{task_a}/output/{index}.png",
                "taskB_input": f"{task_b}/input/{index}.png",
                "taskB_output": f"{task_b}/output/{index}.png",
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "eval.json").write_text(
                json.dumps(
                    [record("a", "b", 0), record("b", "a", 1), record("a", "b", 2)]
                ),
                encoding="utf-8",
            )
            manifest = root / "t2t_tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "data_root": "data",
                        "controlled_protocol": {"resolution": 448},
                        "tasks": [
                            {
                                "name": "a__b",
                                "eval_json": "eval.json",
                                "task_a": "a",
                                "task_b": "b",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = suite_parser().parse_args(
                [
                    "--adapter",
                    "t2t-qwen",
                    "--conditions",
                    "official",
                    "--task-manifest",
                    str(manifest),
                    "--data-root",
                    str(data_root),
                    "--max-samples",
                    "2",
                    "--warmup",
                    "0",
                ]
            )
            adapter = FilterAdapter()
            with mock.patch("comparison.suite.build_adapter", return_value=adapter):
                document = run_suite(args)

        result = document["conditions"][0]["tasks"][0]
        self.assertEqual(result["source_dataset_records"], 3)
        self.assertEqual(result["dataset_records"], 2)
        self.assertEqual(result["record_filter"]["source_record_indices"], [0, 2])
        self.assertEqual(result["record_filter"]["task_a"], "a")
        self.assertEqual(result["record_filter"]["task_b"], "b")
        self.assertEqual(
            [sample["metadata"]["taskB_input"] for sample in result["latency"]["samples"]],
            ["b/input/0.png", "b/input/2.png"],
        )


if __name__ == "__main__":
    unittest.main()
