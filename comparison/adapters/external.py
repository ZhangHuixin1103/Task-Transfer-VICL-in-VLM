from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from ..base import (
    ComparisonAdapter,
    InferenceResult,
    VICLSample,
    load_dataset_records,
    select_dataset_records,
    vicl_sample_from_records,
)
from ..metrics import StageTimer, seed_everything
from .common import (
    import_from_root,
    require_checkpoint_file,
    require_model_file,
    resolve_model_reference,
    torch_dtype,
    working_directory,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
RESAMPLING = getattr(Image, "Resampling", Image)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIFFUSION_LEGACY_POSITION_IDS = (
    "cond_stage_model.transformer.text_model.embeddings.position_ids"
)


def _nested_attribute(value: Any, path: str) -> Any:
    for name in path.split("."):
        value = getattr(value, name)
    return value


def prepare_mae_vqgan_runtime(model: Any) -> Dict[str, Any]:
    """Validate the official hand-written decoder against the installed timm."""
    required_model_attributes = (
        "patch_embed",
        "patch_embed.patch_size",
        "pos_embed",
        "cls_token",
        "blocks",
        "norm",
        "decoder_embed",
        "mask_token",
        "decoder_pos_embed",
        "decoder_blocks",
        "decoder_norm",
        "decoder_pred",
        "vae.quantize.get_codebook_entry",
        "vae.decode",
        "unpatchify",
    )
    missing: list[str] = []
    for path in required_model_attributes:
        try:
            _nested_attribute(model, path)
        except AttributeError:
            missing.append(f"model.{path}")

    patched_blocks: list[int] = []
    incompatible_blocks: list[str] = []
    for index, block in enumerate(getattr(model, "decoder_blocks", ())):
        for path in (
            "norm1",
            "attn.qkv",
            "attn.num_heads",
            "attn.scale",
            "attn.proj",
            "attn.proj_drop",
            "norm2",
            "mlp",
        ):
            try:
                _nested_attribute(block, path)
            except AttributeError:
                missing.append(f"model.decoder_blocks[{index}].{path}")

        if hasattr(block, "drop_path"):
            continue
        drop_path1 = getattr(block, "drop_path1", None)
        drop_path2 = getattr(block, "drop_path2", None)
        if drop_path1 is None or drop_path2 is None:
            missing.append(f"model.decoder_blocks[{index}].drop_path")
            continue

        # timm split the old drop_path into one module per residual branch.
        # MAE constructs these blocks with zero stochastic depth, so both new
        # modules must be no-ops to preserve the released decoder exactly.
        no_op_drop_paths = all(
            isinstance(module, torch.nn.Identity)
            or getattr(module, "drop_prob", None) in (0, 0.0)
            for module in (drop_path1, drop_path2)
        )
        no_op_layer_scales = all(
            isinstance(getattr(block, name, torch.nn.Identity()), torch.nn.Identity)
            for name in ("ls1", "ls2")
        )
        no_op_qk_norms = all(
            isinstance(getattr(block.attn, name, torch.nn.Identity()), torch.nn.Identity)
            for name in ("q_norm", "k_norm")
        )
        if not (no_op_drop_paths and no_op_layer_scales and no_op_qk_norms):
            incompatible_blocks.append(str(index))
            continue
        block.drop_path = drop_path1
        patched_blocks.append(index)

    if missing or incompatible_blocks:
        details = []
        if missing:
            details.append("missing attributes: " + ", ".join(missing))
        if incompatible_blocks:
            details.append(
                "non-default layer-scale/drop-path/qk-norm in decoder blocks: "
                + ", ".join(incompatible_blocks)
            )
        raise RuntimeError(
            "Installed timm is incompatible with the released MAE-VQGAN decoder; "
            + "; ".join(details)
        )
    return {
        "validated_decoder_blocks": len(getattr(model, "decoder_blocks", ())),
        "legacy_drop_path_aliases": patched_blocks,
    }


def load_prompt_diffusion_checkpoint(model, state_dict: Mapping[str, Any]) -> list[str]:
    """Strictly load 04999 while bridging one Transformers buffer change."""
    ignored: list[str] = []
    model_keys = set(model.state_dict())
    if (
        PROMPT_DIFFUSION_LEGACY_POSITION_IDS in state_dict
        and PROMPT_DIFFUSION_LEGACY_POSITION_IDS not in model_keys
    ):
        state_dict = dict(state_dict)
        state_dict.pop(PROMPT_DIFFUSION_LEGACY_POSITION_IDS)
        ignored.append(PROMPT_DIFFUSION_LEGACY_POSITION_IDS)
    model.load_state_dict(state_dict, strict=True)
    return ignored


class PainterAdapter(ComparisonAdapter):
    name = "painter"
    protocol = (
        "[demo input, demo output, query] -> query output (masked image completion)"
    )

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        sample_index: int = 0,
        checkpoint: str | None = None,
        device: str = "cuda",
        dtype: str = "fp32",
        resolution: int = 448,
        task_protocol: str = "restoration",
        include_script_loss: bool = False,
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        valid_protocols = {"restoration", "depth", "semantic", "discrete", "generic"}
        if task_protocol not in valid_protocols:
            raise ValueError(
                f"Unsupported Painter task protocol {task_protocol!r}; "
                f"choose from {sorted(valid_protocols)}"
            )
        if resolution != 448:
            raise ValueError(
                "The released Painter ViT-Large constructor has a fixed 896x448 patch "
                "layout; --resolution must be 448 for comparable inference."
            )
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.sample_index = sample_index
        self.checkpoint = (
            resolve_model_reference(checkpoint, self.repository) if checkpoint else None
        )
        self.device = device
        self.dtype = torch_dtype(dtype)
        self.resolution = resolution
        self.task_protocol = task_protocol
        self.include_script_loss = include_script_loss
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.model = None
        self.windowed_blocks: list[int] = []
        self.checkpoint_missing_keys: list[str] = []
        self.checkpoint_unexpected_keys: list[str] = []

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        requested_name = (
            Path(self.checkpoint).name if self.checkpoint else "painter_vit_large.pth"
        )
        self.checkpoint = require_checkpoint_file(
            self.checkpoint,
            "Painter",
            PROJECT_ROOT / "weights" / "Painter" / requested_name,
        )
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)
        module = import_from_root("models_painter", self.repository)
        self.model = module.painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1()
        if self.checkpoint:
            checkpoint = torch.load(self.checkpoint, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            self.checkpoint_missing_keys = list(incompatible.missing_keys)
            self.checkpoint_unexpected_keys = list(incompatible.unexpected_keys)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.windowed_blocks = [
            index
            for index, block in enumerate(self.model.blocks)
            if int(getattr(block, "window_size", 0)) > 0
        ]

    def _open_resized(
        self, relative_path: str, nearest: bool = False
    ) -> tuple[np.ndarray, tuple[int, int]]:
        with Image.open(self.data_root / relative_path) as source:
            image = source.convert("RGB")
            original_size = image.size
            method = RESAMPLING.NEAREST if nearest else RESAMPLING.BICUBIC
            image = image.resize((self.resolution, self.resolution), method)
            return np.asarray(image, dtype=np.float32) / 255.0, original_size

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.model is None:
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            demo_input, _ = self._open_resized(self.sample.task_a_input)
            demo_output, _ = self._open_resized(
                self.sample.task_a_output,
                nearest=self.task_protocol == "generic",
            )
            query, query_size = self._open_resized(self.sample.task_b_input)
            images = np.concatenate((demo_input, query), axis=0)
            targets = np.concatenate((demo_output, demo_output), axis=0)
            images = (images - IMAGENET_MEAN) / IMAGENET_STD
            targets = (targets - IMAGENET_MEAN) / IMAGENET_STD
            images_tensor = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0)
            targets_tensor = torch.from_numpy(targets).permute(2, 0, 1).unsqueeze(0)
            images_tensor = images_tensor.to(device=self.device, dtype=self.dtype)
            targets_tensor = targets_tensor.to(device=self.device, dtype=self.dtype)
            mask = torch.zeros(self.model.patch_embed.num_patches, device=self.device)
            mask[self.model.patch_embed.num_patches // 2 :] = 1
            mask = mask.unsqueeze(0)
            valid = (
                torch.ones_like(targets_tensor) if self.include_script_loss else None
            )
            torch.manual_seed(2)

        with StageTimer(stages, "model_forward"):
            with torch.inference_mode():
                if self.include_script_loss:
                    _, prediction, _ = self.model(
                        images_tensor, targets_tensor, mask, valid
                    )
                    prediction = self.model.unpatchify(prediction)
                else:
                    latent = self.model.forward_encoder(
                        images_tensor, targets_tensor, mask
                    )
                    prediction = self.model.forward_decoder(latent)

        with StageTimer(stages, "postprocess"):
            prediction = prediction.permute(0, 2, 3, 1).detach().cpu()
            output = prediction[0, prediction.shape[1] // 2 :, :, :]
            output = (
                output * torch.as_tensor(IMAGENET_STD)
                + torch.as_tensor(IMAGENET_MEAN)
            )
            interpolation = {
                "depth": "bilinear",
                "semantic": "bilinear",
                "discrete": "nearest",
                "generic": "nearest",
                "restoration": "bicubic",
            }[self.task_protocol]
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2),
                size=[query_size[1], query_size[0]],
                mode=interpolation,
            ).permute(0, 2, 3, 1)[0]
            scale = 10000 if self.task_protocol == "depth" else 255
            output = torch.clip(output * scale, 0, scale)
            if self.task_protocol == "depth":
                image = Image.fromarray(output.mean(-1).to(torch.int32).numpy())
            else:
                image = Image.fromarray(output.to(torch.uint8).numpy())
        if image.size != query_size:
            raise RuntimeError(
                f"Painter output-size mismatch: expected {query_size}, got {image.size}"
            )
        return InferenceResult(
            output=image,
            stage_seconds=stages,
            metadata={
                "output_size": list(image.size),
                "input_direction": "[demo input, demo output, query input] -> query output",
                "sample": self.sample.as_dict(),
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {"painter": self.model}

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_json = dataset_json.resolve()
        self.demo_input = demo_input
        self.demo_output = demo_output
        source_records = self._record_cache.get(self.dataset_json)
        if source_records is None:
            source_records = load_dataset_records(self.dataset_json)
            self._record_cache[self.dataset_json] = source_records
        self.records = select_dataset_records(source_records, record_indices)
        self.select_sample(0)

    def sample_count(self) -> int:
        return len(self.records)

    def select_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index
        self.sample = vicl_sample_from_records(
            self.records,
            sample_index,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
            source=str(self.dataset_json),
        )

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {
            "condition": condition,
            "architecture": "painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1",
            "resolution": [self.resolution * 2, self.resolution],
            "model_output_resolution": [self.resolution, self.resolution],
            "device": self.device,
            "dtype": str(self.dtype),
            "task_protocol": self.task_protocol,
            "include_unused_inference_loss": self.include_script_loss,
            "seed": 2,
            "windowed_block_indexes": self.windowed_blocks,
            "global_attention_blocks": 24 - len(self.windowed_blocks),
            "released_architecture_warning": (
                "The released 8glb constructor creates no windowed blocks because "
                "window_block_indexes is a tuple of lists; this run reports released-code behavior."
                if not self.windowed_blocks
                else None
            ),
            "checkpoint": self.checkpoint,
            "weights_initialized": self.checkpoint is not None,
            "checkpoint_missing_keys": self.checkpoint_missing_keys,
            "checkpoint_unexpected_keys": self.checkpoint_unexpected_keys,
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The official Painter prompt/query layout is preserved. All restoration images "
                "are resized to the controlled 448x448 protocol before the fixed 896x448 stitch; "
                "the predicted query half is resized back to the original query dimensions. "
                "The default prediction-only path skips the "
                "unused SmoothL1 loss computed by the released evaluation scripts."
            ),
        }


class MAEVQGANAdapter(ComparisonAdapter):
    """Official MAE-VQGAN 2x2 visual-prompt canvas inference."""

    name = "mae-vqgan"
    protocol = "2x2 [demo input, demo output; query, masked target] -> query output"

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        checkpoint: str,
        sample_index: int = 0,
        device: str = "cuda",
        dtype: str = "fp32",
        architecture: str = "mae_vit_large_patch16",
        vqgan_config: str | None = None,
        vqgan_checkpoint: str | None = None,
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        if torch_dtype(dtype) != torch.float32:
            raise ValueError("The official MAE-VQGAN evaluation path uses FP32")
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = resolve_model_reference(checkpoint, self.repository)
        self.sample_index = sample_index
        self.device = device
        self.dtype = torch.float32
        self.architecture = architecture
        self.vqgan_config_reference = vqgan_config
        self.vqgan_checkpoint_reference = vqgan_checkpoint
        self.vqgan_config: str | None = None
        self.vqgan_checkpoint: str | None = None
        self.resolution = 224
        self.canvas_resolution = 224
        self.quadrant_resolution = 111
        self.padding = 1
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.model = None
        self.mae_utils = None
        self.runtime_compatibility: Dict[str, Any] | None = None

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        self.checkpoint = require_checkpoint_file(
            self.checkpoint,
            "MAE-VQGAN MAE",
            PROJECT_ROOT / "weights/MAE-VQGAN/checkpoint-3400.pth",
        )
        checkpoint_dir = Path(self.checkpoint).parent
        asset_dirs = (
            checkpoint_dir,
            PROJECT_ROOT / "weights/MAE-VQGAN",
            self.repository,
        )
        self.vqgan_config = require_model_file(
            self.vqgan_config_reference,
            "MAE-VQGAN VQGAN config (model.yaml)",
            *(directory / "model.yaml" for directory in asset_dirs),
        )
        self.vqgan_checkpoint = require_model_file(
            self.vqgan_checkpoint_reference,
            "MAE-VQGAN VQGAN checkpoint (last.ckpt)",
            *(directory / "last.ckpt" for directory in asset_dirs),
        )
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)
        self.mae_utils = import_from_root("evaluate.mae_utils", self.repository)
        models_mae = self.mae_utils.models_mae
        official_get_vq_model = models_mae.get_vq_model

        def get_vq_model_from_weights():
            return official_get_vq_model(
                config_path=self.vqgan_config,
                ckpt_path=self.vqgan_checkpoint,
            )

        models_mae.get_vq_model = get_vq_model_from_weights
        had_legacy_np_float = "float" in np.__dict__
        legacy_np_float = np.__dict__.get("float")
        if not had_legacy_np_float:
            # The released 2022 MAE utility still requests np.float, which NumPy
            # removed in 1.24. Keep the compatibility local to model creation.
            setattr(np, "float", float)
        try:
            with working_directory(self.repository):
                self.model = self.mae_utils.prepare_model(
                    self.checkpoint, arch=self.architecture, device="cpu"
                )
        finally:
            models_mae.get_vq_model = official_get_vq_model
            if had_legacy_np_float:
                setattr(np, "float", legacy_np_float)
            else:
                delattr(np, "float")
        self.runtime_compatibility = prepare_mae_vqgan_runtime(self.model)
        self.model.eval().to(self.device)

    def _image_tensor(self, relative_path: str) -> torch.Tensor:
        with Image.open(self.data_root / relative_path) as source:
            image = source.convert("RGB").resize(
                (self.quadrant_resolution, self.quadrant_resolution),
                RESAMPLING.BILINEAR,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.model is None:
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            demo_input = self._image_tensor(self.sample.task_a_input)
            demo_output = self._image_tensor(self.sample.task_a_output)
            query = self._image_tensor(self.sample.task_b_input)
            canvas = torch.zeros((3, 224, 224), dtype=torch.float32)
            canvas[:, :111, :111] = demo_input
            canvas[:, :111, -111:] = demo_output
            canvas[:, -111:, :111] = query
            canvas[:, -111:, -111:] = query
            mean = torch.as_tensor(IMAGENET_MEAN, dtype=canvas.dtype)[:, None, None]
            std = torch.as_tensor(IMAGENET_STD, dtype=canvas.dtype)[:, None, None]
            canvas = (canvas - mean) / std
            ids_shuffle, len_keep = self.mae_utils.generate_mask_for_evaluation()
        with StageTimer(stages, "model_forward"):
            with torch.inference_mode():
                _, completed, _ = self.mae_utils.generate_image(
                    canvas.unsqueeze(0).to(self.device),
                    self.model,
                    ids_shuffle.to(self.device),
                    len_keep,
                    device=self.device,
                )
        with StageTimer(stages, "postprocess"):
            array = np.asarray(completed, dtype=np.uint8)
            output = Image.fromarray(array[113:, 113:, :].copy())
        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "output_size": list(output.size),
                "input_direction": self.protocol,
                "sample": self.sample.as_dict(),
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {"mae_vqgan": self.model}

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_json = dataset_json.resolve()
        self.demo_input = demo_input
        self.demo_output = demo_output
        source_records = self._record_cache.get(self.dataset_json)
        if source_records is None:
            source_records = load_dataset_records(self.dataset_json)
            self._record_cache[self.dataset_json] = source_records
        self.records = select_dataset_records(source_records, record_indices)
        self.select_sample(0)

    def sample_count(self) -> int:
        return len(self.records)

    def select_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index
        self.sample = vicl_sample_from_records(
            self.records,
            sample_index,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
            source=str(self.dataset_json),
        )

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {
            "condition": condition,
            "checkpoint": self.checkpoint,
            "vqgan_config": self.vqgan_config,
            "vqgan_checkpoint": self.vqgan_checkpoint,
            "runtime_compatibility": self.runtime_compatibility,
            "architecture": self.architecture,
            "device": self.device,
            "dtype": str(self.dtype),
            "canvas_resolution": [224, 224],
            "quadrant_resolution": [111, 111],
            "mask": "official bottom-right 7x7 patch quadrant",
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The official demo notebook canvas, ImageNet normalization, fixed "
                "evaluation mask, VQGAN decode, and bottom-right crop are preserved."
            ),
        }


class PromptGIPAdapter(ComparisonAdapter):
    """Official PromptGIP four-image masked-target inference."""

    name = "prompt-gip"
    protocol = "[demo input, demo output, query, masked target] -> query output"

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        checkpoint: str,
        sample_index: int = 0,
        device: str = "cuda",
        dtype: str = "fp32",
        architecture: str = "mae_vit_large_patch16_input256",
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        if not device.startswith("cuda"):
            raise ValueError("The released PromptGIP constructor requires CUDA")
        if torch_dtype(dtype) != torch.float32:
            raise ValueError("The official PromptGIP evaluation path uses FP32")
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = resolve_model_reference(checkpoint, self.repository)
        self.sample_index = sample_index
        self.device = device
        self.dtype = torch.float32
        self.architecture = architecture
        self.resolution = 256
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.model = None

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(
                f"PromptGIP requires its official released checkpoint: {self.checkpoint}"
            )
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)
        module = import_from_root(
            "models_mae_PromptGIP_CNN_Head", self.repository
        )
        with working_directory(self.repository):
            self.model = getattr(module, self.architecture)()
            checkpoint = torch.load(self.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval().to(self.device)

    def _image_tensor(self, relative_path: str) -> torch.Tensor:
        with Image.open(self.data_root / relative_path) as source:
            image = source.convert("RGB").resize(
                (self.resolution, self.resolution), RESAMPLING.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.model is None:
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            demo_input = self._image_tensor(self.sample.task_a_input)
            demo_output = self._image_tensor(self.sample.task_a_output)
            query = self._image_tensor(self.sample.task_b_input)
            inputs = [demo_input, demo_output, query, query]
        with StageTimer(stages, "model_forward"):
            with torch.inference_mode():
                _, _, _, _, prediction = self.model(
                    imgs=inputs,
                    mask_ratio=1,
                    input_is_list=True,
                    train_mode=False,
                )
        with StageTimer(stages, "postprocess"):
            pixels = (
                prediction[0]
                .detach()
                .float()
                .clamp(0, 1)
                .permute(1, 2, 0)
                .mul(255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            output = Image.fromarray(pixels)
        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "output_size": list(output.size),
                "input_direction": self.protocol,
                "sample": self.sample.as_dict(),
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {"prompt_gip": self.model}

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_json = dataset_json.resolve()
        self.demo_input = demo_input
        self.demo_output = demo_output
        source_records = self._record_cache.get(self.dataset_json)
        if source_records is None:
            source_records = load_dataset_records(self.dataset_json)
            self._record_cache[self.dataset_json] = source_records
        self.records = select_dataset_records(source_records, record_indices)
        self.select_sample(0)

    def sample_count(self) -> int:
        return len(self.records)

    def select_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index
        self.sample = vicl_sample_from_records(
            self.records,
            sample_index,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
            source=str(self.dataset_json),
        )

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {
            "condition": condition,
            "checkpoint": self.checkpoint,
            "architecture": self.architecture,
            "device": self.device,
            "dtype": str(self.dtype),
            "native_resolution": [256, 256],
            "mask_ratio": 1,
            "train_mode": False,
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The official four-image order, fully masked fourth image, CNN head, "
                "released checkpoint, and 256x256 evaluation resolution are preserved."
            ),
        }


class VisualClozeAdapter(ComparisonAdapter):
    """Official VisualCloze one-example, two-column visual grid inference."""

    name = "visualcloze"
    protocol = "2x2 [demo input, demo output; query, masked target] + text -> output"

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        checkpoint: str,
        sample_index: int = 0,
        device: str = "cuda",
        dtype: str = "bf16",
        resolution: int = 384,
        steps: int = 30,
        seed: int = 0,
        text_prompt: str = "perform the demonstrated visual task",
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        if not device.startswith("cuda"):
            raise ValueError("The released VisualCloze model requires CUDA")
        if resolution not in {384, 512}:
            raise ValueError("VisualCloze released checkpoints use resolution 384 or 512")
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = resolve_model_reference(checkpoint, self.repository)
        self.sample_index = sample_index
        self.device = device
        self.dtype_name = dtype
        self.dtype = torch_dtype(dtype)
        self.resolution = resolution
        self.steps = steps
        self.seed = seed
        self.text_prompt = text_prompt
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.model = None

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(
                f"VisualCloze requires its official LoRA checkpoint: {self.checkpoint}"
            )
        from ..preflight import inspect_visualcloze_environment

        environment = inspect_visualcloze_environment()
        if environment["status"] != "pass":
            raise RuntimeError(
                "VisualCloze environment preflight failed: "
                + "; ".join(environment["errors"])
                + ". Reinstall comparison/requirements/visualcloze.txt."
            )
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)
        try:
            module = import_from_root("visualcloze", self.repository)
        except (AttributeError, ImportError, RuntimeError) as error:
            message = str(error)
            if "infer_schema(func)" in message or "attention_dispatch" in message:
                raise RuntimeError(
                    "VisualCloze cannot import with this Torch/Diffusers combination. "
                    "Run it in the dedicated VisualCloze environment from "
                    "comparison/README.md (torch==2.1.0, diffusers==0.32.1); do not "
                    "run this adapter in the Qwen environment."
                ) from error
            raise
        with working_directory(self.repository):
            self.model = module.VisualClozeModel(
                model_path=self.checkpoint,
                resolution=self.resolution,
                lora_rank=256,
                precision=self.dtype_name,
            )
        self.model.set_grid_size(2, 2)

    @staticmethod
    def _layout_prompt() -> str:
        return "4 images are organized into a grid of 2 rows and 2 columns, evenly spaced."

    def _task_prompt(self) -> str:
        return (
            "Each row shows the same image transformation: [IMAGE1] is the input "
            f"and [IMAGE2] is the output. The transformation should {self.text_prompt}."
        )

    def _open_rgb(self, relative_path: str) -> Image.Image:
        with Image.open(self.data_root / relative_path) as source:
            return source.convert("RGB").copy()

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.model is None:
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            demo_input = self._open_rgb(self.sample.task_a_input)
            demo_output = self._open_rgb(self.sample.task_a_output)
            query = self._open_rgb(self.sample.task_b_input)
            images = [[demo_input, demo_output], [query, None]]
            prompts = [self._layout_prompt(), self._task_prompt(), ""]
        try:
            with StageTimer(stages, "diffusion_generation"):
                with torch.inference_mode():
                    output = self.model.process_images(
                        images,
                        prompts,
                        seed=self.seed,
                        cfg=30,
                        steps=self.steps,
                        upsampling_steps=10,
                        upsampling_noise=0.4,
                        is_upsampling=True,
                    )[0]
        finally:
            demo_input.close()
            demo_output.close()
            query.close()
        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "output_size": list(output.size),
                "input_direction": self.protocol,
                "sample": self.sample.as_dict(),
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {
            "visualcloze_flow": getattr(self.model, "model", None),
            "visualcloze_vae": getattr(self.model, "ae", None),
            "visualcloze_t5": getattr(self.model, "t5", None),
            "visualcloze_clip": getattr(self.model, "clip", None),
        }

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_json = dataset_json.resolve()
        self.demo_input = demo_input
        self.demo_output = demo_output
        source_records = self._record_cache.get(self.dataset_json)
        if source_records is None:
            source_records = load_dataset_records(self.dataset_json)
            self._record_cache[self.dataset_json] = source_records
        self.records = select_dataset_records(source_records, record_indices)
        self.select_sample(0)

    def sample_count(self) -> int:
        return len(self.records)

    def select_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index
        self.sample = vicl_sample_from_records(
            self.records,
            sample_index,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
            source=str(self.dataset_json),
        )

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {
            "condition": condition,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "dtype": str(self.dtype),
            "native_grid_resolution": self.resolution,
            "steps": self.steps,
            "seed": self.seed,
            "guidance_scale": 30,
            "upsampling_steps": 10,
            "upsampling_noise": 0.4,
            "lora_rank": 256,
            "layout_prompt": self._layout_prompt(),
            "task_prompt": self._task_prompt(),
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The released VisualClozeModel one-example grid API and its 384-grid, "
                "30-step, CFG-30, and SDEdit-upsample defaults are preserved."
            ),
        }


class PromptDiffusionAdapter(ComparisonAdapter):
    name = "prompt-diffusion"
    protocol = (
        "same-task [demo input, demo output, query] + official quality prompt "
        "-> query output (diffusion)"
    )

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        checkpoint: str,
        sample_index: int = 0,
        config: str = "models/cldm_v15.yaml",
        device: str = "cuda",
        dtype: str = "fp32",
        steps: int = 100,
        seed: int = 1,
        content_prompt: str = "",
        resolution: int = 512,
        positive_prompt: str = "best quality, extremely detailed",
        negative_prompt: str = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        ),
        strength: float = 1.0,
        guidance_scale: float = 9.0,
        eta: float = 0.0,
        guess_mode: bool = False,
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        if not device.startswith("cuda"):
            raise ValueError("Prompt-Diffusion's released DDIM sampler requires CUDA")
        if torch_dtype(dtype) != torch.float32:
            raise ValueError("The official network-step=04999.ckpt inference uses FP32")
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = resolve_model_reference(checkpoint, self.repository)
        self.config_path = Path(resolve_model_reference(config, self.repository))
        self.sample_index = sample_index
        self.device = device
        self.dtype = torch.float32
        self.steps = steps
        self.seed = seed
        self.content_prompt = content_prompt.strip()
        self.resolution = resolution
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.guess_mode = guess_mode
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.model = None
        self.sampler = None
        self.prompt_diffusion_config = None
        self.cv2 = None
        self.einops = None
        self.hwc3 = None
        self.resize_image = None
        self.checkpoint_compatibility_ignored_keys: list[str] = []
        self.manifest_task_name: str | None = None
        self.manifest_task_instruction: str | None = None

    @property
    def positive_conditioning(self) -> str:
        """Build the official positive CLIP string without a leading comma."""
        return ", ".join(
            value
            for value in (self.content_prompt, self.positive_prompt.strip())
            if value
        )

    def set_task_prompt(self, task_name: str, prompt: str) -> None:
        """Record the manifest instruction, but do not feed it to Prompt-Diffusion."""
        self.manifest_task_name = task_name
        self.manifest_task_instruction = prompt

    def text_conditioning_metadata(self) -> Dict[str, Any]:
        return {
            "policy_version": "empty-content-v1",
            "content_prompt": self.content_prompt,
            "positive_prompt": self.positive_prompt,
            "effective_positive_conditioning": self.positive_conditioning,
            "negative_prompt": self.negative_prompt,
            "manifest_task_name": self.manifest_task_name,
            "manifest_task_instruction": self.manifest_task_instruction,
            "manifest_task_instruction_used": False,
            "task_source": "same-task visual demonstration only",
        }

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        requested_name = Path(self.checkpoint).name
        self.checkpoint = require_checkpoint_file(
            self.checkpoint,
            "Prompt-Diffusion",
            self.repository / "ckpts" / requested_name,
            PROJECT_ROOT / "weights" / "Prompt-Diffusion" / requested_name,
        )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Prompt-Diffusion model config was not found: {self.config_path}"
            )
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))
        with working_directory(self.repository):
            import config as prompt_diffusion_config
            import cv2
            import einops
            from annotator.util import HWC3, resize_image
            from cldm.ddim_hacked import DDIMSampler
            from cldm.model import create_model, load_state_dict

            self.model = create_model(str(self.config_path)).cpu()
            state_dict = load_state_dict(self.checkpoint, location="cpu")
            self.checkpoint_compatibility_ignored_keys = (
                load_prompt_diffusion_checkpoint(self.model, state_dict)
            )
        self.model = self.model.to(self.device).eval()
        self.sampler = DDIMSampler(self.model)
        self.prompt_diffusion_config = prompt_diffusion_config
        self.cv2 = cv2
        self.einops = einops
        self.hwc3 = HWC3
        self.resize_image = resize_image

    def run(self, condition: str) -> InferenceResult:
        if (
            condition != "official"
            or self.sample is None
            or self.model is None
            or self.sampler is None
        ):
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            query_image = self._load_uint8(self.sample.task_b_input)
            demo_input = self._load_uint8(self.sample.task_a_input)
            demo_output = self._load_uint8(self.sample.task_a_output)
            query_shape = self.resize_image(
                self.hwc3(query_image), self.resolution
            ).shape
            height, width, _ = query_shape
            query_map = self.cv2.resize(
                self.hwc3(query_image),
                (width, height),
                interpolation=self.cv2.INTER_LINEAR,
            )
            demo_source = self.cv2.resize(
                self.hwc3(demo_input),
                (width, height),
                interpolation=self.cv2.INTER_LINEAR,
            )
            demo_target = self.resize_image(
                self.hwc3(demo_output), self.resolution
            )
            demo_target_shape_adjusted = demo_target.shape != query_shape
            if demo_target_shape_adjusted:
                # The official notebook assumes matching prompt/query aspect ratios.
                # Our held-out pairs need this interface-only spatial alignment.
                demo_target = self.cv2.resize(
                    demo_target,
                    (width, height),
                    interpolation=self.cv2.INTER_LINEAR,
                )
            example = np.concatenate([demo_source, demo_target], axis=2)
            query = 2 * torch.from_numpy(query_map.copy()).float().to(self.device) / 255 - 1
            example = 2 * torch.from_numpy(example.copy()).float().to(self.device) / 255 - 1
            query = self.einops.rearrange(query[None], "b h w c -> b c h w").clone()
            example = self.einops.rearrange(
                example[None], "b h w c -> b c h w"
            ).clone()
            seed_everything(self.seed)

        with StageTimer(stages, "diffusion_generation"):
            with torch.inference_mode():
                if self.prompt_diffusion_config.save_memory:
                    self.model.low_vram_shift(is_diffusing=False)
                cond = {
                    "c_crossattn": [
                        self.model.get_learned_conditioning(
                            [self.positive_conditioning]
                        )
                    ],
                    "example_pair": [example],
                    "query": [query],
                }
                uncond = {
                    "c_crossattn": [
                        self.model.get_learned_conditioning([self.negative_prompt])
                    ],
                    "example_pair": [example],
                    "query": [query],
                }
                if self.prompt_diffusion_config.save_memory:
                    self.model.low_vram_shift(is_diffusing=True)
                self.model.control_scales = (
                    [
                        self.strength * (0.825 ** float(12 - index))
                        for index in range(13)
                    ]
                    if self.guess_mode
                    else [self.strength] * 13
                )
                samples, _ = self.sampler.sample(
                    self.steps,
                    1,
                    (4, height // 8, width // 8),
                    cond,
                    verbose=False,
                    eta=self.eta,
                    unconditional_guidance_scale=self.guidance_scale,
                    unconditional_conditioning=uncond,
                )
                if self.prompt_diffusion_config.save_memory:
                    self.model.low_vram_shift(is_diffusing=False)
                decoded = self.model.decode_first_stage(samples)
                decoded = (
                    self.einops.rearrange(decoded, "b c h w -> b h w c")
                    * 127.5
                    + 127.5
                ).cpu().numpy().clip(0, 255).astype(np.uint8)
                output = Image.fromarray(decoded[0])

        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "output_size": list(output.size),
                "demo_target_shape_adjusted": demo_target_shape_adjusted,
                "input_direction": "[demo input, demo output, query input] -> query output",
                "text_conditioning": self.text_conditioning_metadata(),
                "sample": self.sample.as_dict(),
            },
        )

    def _load_uint8(self, relative_path: str) -> np.ndarray:
        with Image.open(self.data_root / relative_path) as source:
            return np.asarray(source.convert("RGB"), dtype=np.uint8)

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {"prompt_diffusion": self.model}

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_json = dataset_json.resolve()
        self.demo_input = demo_input
        self.demo_output = demo_output
        source_records = self._record_cache.get(self.dataset_json)
        if source_records is None:
            source_records = load_dataset_records(self.dataset_json)
            self._record_cache[self.dataset_json] = source_records
        self.records = select_dataset_records(source_records, record_indices)
        self.select_sample(0)

    def sample_count(self) -> int:
        return len(self.records)

    def select_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index
        self.sample = vicl_sample_from_records(
            self.records,
            sample_index,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
            source=str(self.dataset_json),
        )

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {
            "condition": condition,
            "checkpoint": self.checkpoint,
            "checkpoint_compatibility_ignored_keys": (
                self.checkpoint_compatibility_ignored_keys
            ),
            "config": str(self.config_path),
            "sampler": "official cldm.ddim_hacked.DDIMSampler",
            "steps": self.steps,
            "device": self.device,
            "dtype": str(self.dtype),
            "seed": self.seed,
            "image_resolution_min_side": self.resolution,
            "content_prompt": self.content_prompt,
            "positive_prompt": self.positive_prompt,
            "effective_positive_conditioning": self.positive_conditioning,
            "negative_prompt": self.negative_prompt,
            "text_conditioning": self.text_conditioning_metadata(),
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "eta": self.eta,
            "guess_mode": self.guess_mode,
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The official network-step=04999.ckpt, cldm_v15 config, conditioning "
                "dictionary, and DDIMSampler path from run_prompt_diffusion.ipynb are used. "
                "Because this dataset has no target-image content captions, the content "
                "prompt is empty; only the released positive/negative quality prompts are "
                "used, and the task is inferred from the same-task visual example. "
                "When a held-out demonstration and query have different aspect ratios, the "
                "demonstration target is spatially aligned to the official query canvas before "
                "forming the six-channel example pair."
            ),
        }


class InstructDiffusionAdapter(ComparisonAdapter):
    name = "instruct-diffusion"
    protocol = "source image + text instruction -> edited image (diffusion)"

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        checkpoint: str,
        sample_index: int = 0,
        config: str = "configs/instruct_diffusion.yaml",
        device: str = "cuda",
        dtype: str = "fp16",
        resolution: int = 512,
        steps: int = 100,
        seed: int = 42,
        text_prompt: str = "Restore the image according to the instruction.",
        cfg_text: float = 5.0,
        cfg_image: float = 1.25,
        fixed_square: bool = False,
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = resolve_model_reference(checkpoint, self.repository)
        self.sample_index = sample_index
        self.config_path = Path(resolve_model_reference(config, self.repository))
        self.device = device
        self.dtype = torch_dtype(dtype)
        self.resolution = resolution
        self.steps = steps
        self.seed = seed
        self.text_prompt = text_prompt
        self.cfg_text = cfg_text
        self.cfg_image = cfg_image
        self.fixed_square = fixed_square
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.module = None
        self.model = None
        self.model_wrap = None
        self.model_wrap_cfg = None
        self.null_token = None

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        requested_name = Path(self.checkpoint).name
        self.checkpoint = require_checkpoint_file(
            self.checkpoint,
            "InstructDiffusion",
            self.repository / "checkpoints" / requested_name,
            PROJECT_ROOT / "weights" / "InstructDiffusion" / requested_name,
        )
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)
        self.module = import_from_root("edit_cli", self.repository)
        config = self.module.OmegaConf.load(str(self.config_path))
        with working_directory(self.repository):
            self.model = self.module.load_model_from_config(config, self.checkpoint)
        self.model.eval().to(self.device)
        self.model_wrap = self.module.K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = self.module.CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.model is None:
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            with Image.open(self.data_root / self.sample.task_b_input) as source_file:
                source = source_file.convert("RGB").copy()
            original_size = source.size
            if self.fixed_square:
                width = height = self.resolution
                source = source.resize(
                    (width, height), RESAMPLING.BICUBIC
                )
            else:
                factor = self.resolution / max(source.size)
                factor = math.ceil(min(source.size) * factor / 64) * 64 / min(source.size)
                width = int((source.width * factor) // 64) * 64
                height = int((source.height * factor) // 64) * 64
                source = ImageOps.fit(
                    source, (width, height), method=RESAMPLING.LANCZOS
                )
            source_tensor = 2 * torch.tensor(np.array(source)).float() / 255 - 1
            source_tensor = source_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with StageTimer(stages, "diffusion_generation"):
            with torch.no_grad(), autocast_context:
                cond = {
                    "c_crossattn": [
                        self.model.get_learned_conditioning([self.text_prompt])
                    ],
                    "c_concat": [self.model.encode_first_stage(source_tensor).mode()],
                }
                uncond = {
                    "c_crossattn": [self.null_token],
                    "c_concat": [torch.zeros_like(cond["c_concat"][0])],
                }
                sigmas = self.model_wrap.get_sigmas(self.steps)
                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": self.cfg_text,
                    "image_cfg_scale": self.cfg_image,
                }
                torch.manual_seed(self.seed)
                latents = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                latents = self.module.K.sampling.sample_euler_ancestral(
                    self.model_wrap_cfg, latents, sigmas, extra_args=extra_args
                )
                decoded = self.model.decode_first_stage(latents)

        with StageTimer(stages, "postprocess"):
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
            decoded = (255.0 * decoded[0].permute(1, 2, 0)).to(torch.uint8)
            output = Image.fromarray(decoded.cpu().numpy())
            if not self.fixed_square:
                output = ImageOps.fit(
                    output, original_size, method=RESAMPLING.LANCZOS
                )
        if self.fixed_square and output.size != (self.resolution, self.resolution):
            raise RuntimeError(
                "InstructDiffusion ignored the controlled output size: "
                f"expected {(self.resolution, self.resolution)}, got {output.size}"
            )
        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "output_size": list(output.size),
                "input_direction": "B_input + instruction -> B_output",
                "sample": self.sample.as_dict(),
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {"instruct_diffusion": self.model}

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_json = dataset_json.resolve()
        self.demo_input = demo_input
        self.demo_output = demo_output
        source_records = self._record_cache.get(self.dataset_json)
        if source_records is None:
            source_records = load_dataset_records(self.dataset_json)
            self._record_cache[self.dataset_json] = source_records
        self.records = select_dataset_records(source_records, record_indices)
        self.select_sample(0)

    def sample_count(self) -> int:
        return len(self.records)

    def select_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index
        self.sample = vicl_sample_from_records(
            self.records,
            sample_index,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
            source=str(self.dataset_json),
        )

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {
            "condition": condition,
            "checkpoint": self.checkpoint,
            "config": str(self.config_path),
            "resolution_long_side": self.resolution,
            "shape_policy": "fixed_square" if self.fixed_square else "native_aspect",
            "device": self.device,
            "autocast_dtype": str(self.dtype),
            "steps": self.steps,
            "sampler": "Euler ancestral",
            "cfg_text": self.cfg_text,
            "cfg_image": self.cfg_image,
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "InstructDiffusion consumes only B_in plus an instruction. It is included as an "
                "instruction-conditioned generalist reference, not as an input-equivalent VICL method."
            ),
        }
