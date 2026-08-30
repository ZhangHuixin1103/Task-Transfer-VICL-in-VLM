from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from efficiency.adapters.external import (
    HIDDEN_SHOT_TASK_LABELS,
    HiddenShotAdapter,
    _infer_hidden_pgn_model,
)


class _FakeClip:
    def __init__(self):
        self.prompts: list[str] = []

    def tokenize(self, prompt: str) -> torch.Tensor:
        self.prompts.append(prompt)
        return torch.zeros((1, 77), dtype=torch.long)


class _PatchEmbed:
    num_patches = (896 // 16) * (448 // 16)


class _FakeHiddenShot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.prompt_module = torch.nn.Linear(2, 2)
        self.encoder_inputs = None
        self.decoder_inputs = None

    def forward_encoder(self, images, targets, mask):
        self.encoder_inputs = (images.detach().clone(), targets.detach().clone(), mask)
        return [torch.zeros((1, 56, 28, 1024)) for _ in range(4)]

    def forward_decoder(self, latent, images, text_tokens):
        self.decoder_inputs = (latent, images.detach().clone(), text_tokens.detach())
        return torch.zeros((1, 3, 896, 448))


def _write_rgb(path: Path, value: tuple[int, int, int]) -> None:
    Image.new("RGB", (9, 7), value).save(path)


class HiddenShotAdapterTest(unittest.TestCase):
    def test_checkpoint_pgn_backbone_detection(self):
        self.assertEqual(
            _infer_hidden_pgn_model(
                {
                    "prompt_module.pgn_module.model.input_net.0.weight": torch.empty(1),
                }
            ),
            "resnet10",
        )
        self.assertEqual(
            _infer_hidden_pgn_model(
                {
                    "prompt_module.pgn_module.model.layer1.0.conv1.weight": torch.empty(1),
                }
            ),
            "resnet18",
        )
        self.assertEqual(
            _infer_hidden_pgn_model(
                {
                    "prompt_module.pgn_module.model.features.denseblock1.x": torch.empty(1),
                    "prompt_module.pgn_module.model.classifier.weight": torch.empty(
                        4096, 96
                    ),
                }
            ),
            "densenet18",
        )
        with self.assertRaisesRegex(ValueError, "--hidden-pgn-model"):
            _infer_hidden_pgn_model({"blocks.0.weight": torch.empty(1)})

    def test_run_preserves_three_image_direction_and_returns_query_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_rgb(root / "demo_input.png", (255, 0, 0))
            _write_rgb(root / "demo_output.png", (0, 255, 0))
            _write_rgb(root / "query.png", (0, 0, 255))
            _write_rgb(root / "target.png", (255, 255, 255))
            records = root / "pairs.json"
            records.write_text(
                json.dumps(
                    [{"image_path": "query.png", "target_path": "target.png"}]
                ),
                encoding="utf-8",
            )

            adapter = HiddenShotAdapter(
                repository=root,
                dataset_json=records,
                data_root=root,
                checkpoint="unused.pth",
                device="cpu",
                dtype="fp32",
                demo_input="demo_input.png",
                demo_output="demo_output.png",
            )
            adapter.configure_samples(
                records,
                demo_input="demo_input.png",
                demo_output="demo_output.png",
            )
            adapter.model = _FakeHiddenShot()
            adapter.clip = _FakeClip()
            adapter.set_task_prompt("low_light_enhancement", "enhance the image")
            result = adapter.run("official")

        self.assertEqual(result.output.size, (448, 448))
        self.assertEqual(result.metadata["task_label"], "enhancement")
        self.assertEqual(
            adapter.clip.prompts, ["This is a photo of a enhancement task"]
        )
        images, targets, mask = adapter.model.encoder_inputs
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        expected_demo_input = (torch.tensor([1.0, 0.0, 0.0]) - mean) / std
        expected_query = (torch.tensor([0.0, 0.0, 1.0]) - mean) / std
        expected_demo_output = (torch.tensor([0.0, 1.0, 0.0]) - mean) / std
        self.assertTrue(torch.allclose(images[0, :, 0, 0], expected_demo_input))
        self.assertTrue(torch.allclose(images[0, :, -1, 0], expected_query))
        self.assertTrue(torch.allclose(targets[0, :, 0, 0], expected_demo_output))
        self.assertTrue(torch.allclose(targets[0, :, -1, 0], expected_demo_output))
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[0, -1]), 1)
        self.assertEqual(adapter.model.decoder_inputs[1].shape, (1, 3, 896, 448))
        result.output.close()

    def test_current_sparse_tasks_have_explicit_hidden_shot_labels(self):
        expected = {
            "deblurring",
            "dehazing",
            "demoireing",
            "denoising",
            "deraining",
            "low_light_enhancement",
            "shadow_removal",
            "reflection_removal",
            "relighting",
            "inpainting",
        }
        self.assertEqual(set(HIDDEN_SHOT_TASK_LABELS), expected)


if __name__ == "__main__":
    unittest.main()
