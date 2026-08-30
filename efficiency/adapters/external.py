from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from ..base import (
    EfficiencyAdapter,
    InferenceResult,
    VICLSample,
    load_dataset_records,
    select_dataset_records,
    vicl_sample_from_records,
)
from ..metrics import StageTimer
from .common import (
    import_from_root,
    resolve_model_reference,
    torch_dtype,
    working_directory,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
RESAMPLING = getattr(Image, "Resampling", Image)


HIDDEN_SHOT_TASK_LABELS = {
    "deblurring": "deblurring",
    "dehazing": "dehazing",
    "demoireing": "demoire",
    "denoising": "denoising",
    "deraining": "deraining",
    "low_light_enhancement": "enhancement",
    "shadow_removal": "shadow removal",
    "reflection_removal": "reflection removal",
    "relighting": "relighting",
    "inpainting": "inpainting",
}


def _infer_hidden_pgn_model(state_dict: Mapping[str, Any]) -> str:
    """Infer the PGN CNN required to instantiate a released Hidden-Shot checkpoint."""
    keys = tuple(str(key) for key in state_dict)
    marker = "prompt_module.pgn_module.model."
    pgn_keys = [key[key.index(marker) + len(marker) :] for key in keys if marker in key]
    if any(key.startswith("input_net.") for key in pgn_keys):
        return "resnet10"
    if any(key.startswith("layer1.") for key in pgn_keys):
        return "resnet18"
    if any(key.startswith("features.denseblock") for key in pgn_keys):
        classifier = next(
            (
                value
                for key, value in state_dict.items()
                if str(key).endswith(
                    "prompt_module.pgn_module.model.classifier.weight"
                )
            ),
            None,
        )
        in_features = (
            int(classifier.shape[1])
            if isinstance(classifier, torch.Tensor) and classifier.ndim == 2
            else None
        )
        by_width = {
            96: "densenet18",
            512: "densenet121",
            832: "densenet169",
            960: "densenet201",
            2208: "densenet161",
        }
        if in_features in by_width:
            return by_width[in_features]
    raise ValueError(
        "Could not infer Hidden-Shot's PGN backbone from the checkpoint. "
        "Pass --hidden-pgn-model explicitly."
    )


class PainterAdapter(EfficiencyAdapter):
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

    def _open_resized(self, relative_path: str, nearest: bool = False) -> np.ndarray:
        with Image.open(self.data_root / relative_path) as source:
            image = source.convert("RGB")
            method = RESAMPLING.NEAREST if nearest else RESAMPLING.BICUBIC
            image = image.resize((self.resolution, self.resolution), method)
            return np.asarray(image, dtype=np.float32) / 255.0

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.model is None:
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            demo_input = self._open_resized(self.sample.task_a_input)
            demo_output = self._open_resized(
                self.sample.task_a_output,
                nearest=self.task_protocol == "generic",
            )
            query = self._open_resized(self.sample.task_b_input)
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
            scale = 10000 if self.task_protocol == "depth" else 255
            output = torch.clip(
                (
                    output * torch.as_tensor(IMAGENET_STD)
                    + torch.as_tensor(IMAGENET_MEAN)
                )
                * scale,
                0,
                scale,
            )
            if self.task_protocol == "depth":
                image = Image.fromarray(output.mean(-1).to(torch.int32).numpy())
            else:
                image = Image.fromarray(output.to(torch.uint8).numpy())
        if image.size != (self.resolution, self.resolution):
            raise RuntimeError(
                f"Painter output-size mismatch: expected "
                f"{(self.resolution, self.resolution)}, got {image.size}"
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
            "output_resolution": [self.resolution, self.resolution],
            "device": self.device,
            "dtype": str(self.dtype),
            "task_protocol": self.task_protocol,
            "include_unused_inference_loss": self.include_script_loss,
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
                "the predicted query half is already 448x448 and is not resized afterward. "
                "The default prediction-only path skips the "
                "unused SmoothL1 loss computed by the released evaluation scripts. A cross-task "
                "VICL sample is used only to keep input image count and resolution comparable."
            ),
        }


class HiddenShotAdapter(EfficiencyAdapter):
    name = "hidden-shot"
    protocol = (
        "[demo input, demo output, query] + task name -> query output "
        "(learned prompt + masked image completion)"
    )

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        checkpoint: str,
        sample_index: int = 0,
        device: str = "cuda",
        dtype: str = "fp32",
        resolution: int = 448,
        clip_architecture: str = "ViT-B/32",
        pgn_model_type: str = "auto",
        task_name: str = "restoration",
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        if resolution != 448:
            raise ValueError(
                "Hidden-Shot inherits Painter's fixed 896x448 patch layout; "
                "--resolution must be 448."
            )
        if torch_dtype(dtype) != torch.float32:
            raise ValueError(
                "The released Hidden-Shot inference path is FP32 and its PGN mixes "
                "plain FP32 tensors with model parameters; use --dtype fp32."
            )
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = resolve_model_reference(checkpoint, self.repository)
        self.sample_index = sample_index
        self.device = device
        self.dtype = torch.float32
        self.resolution = resolution
        self.clip_architecture = clip_architecture
        self.requested_pgn_model_type = pgn_model_type
        self.pgn_model_type: str | None = None
        self.task_name = task_name
        self.task_label = HIDDEN_SHOT_TASK_LABELS.get(
            task_name, task_name.replace("_", " ")
        )
        self.manifest_text_prompt: str | None = None
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.model = None
        self.clip = None
        self.checkpoint_missing_keys: list[str] = []
        self.checkpoint_unexpected_keys: list[str] = []

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        checkpoint_path = Path(self.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Hidden-Shot requires one released trained .pth checkpoint; "
                f"not found: {checkpoint_path}"
            )
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, Mapping)
            else checkpoint
        )
        if not isinstance(state_dict, Mapping):
            raise TypeError("Hidden-Shot checkpoint must contain a model state dict")
        self.pgn_model_type = (
            _infer_hidden_pgn_model(state_dict)
            if self.requested_pgn_model_type == "auto"
            else self.requested_pgn_model_type
        )
        pgn_settings = {
            "prompt_mode": "pgn",
            "pgn_resolution": 224,
            "nr_output_vectors": 16,
            "vector_dim": 768,
            "mixture_size": 256,
            "pretrained_pgn": False,
            "model_type": self.pgn_model_type,
            "proj_type": "linear",
            "pgn_act_fn": "softmax",
            "nr_groups": 4,
            "blocks_per_group": 1,
            "initial_channels": 16,
            "init_max_pool": True,
        }
        module = import_from_root("models_painter2", self.repository)
        self.model = module.painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1(
            clip_architecture=self.clip_architecture,
            pgn_settings=pgn_settings,
        )
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        self.checkpoint_missing_keys = list(incompatible.missing_keys)
        self.checkpoint_unexpected_keys = list(incompatible.unexpected_keys)
        critical = [
            key
            for key in self.checkpoint_missing_keys + self.checkpoint_unexpected_keys
            if "prompt_module" in key
        ]
        if critical:
            raise RuntimeError(
                "Hidden-Shot checkpoint does not match the inferred/configured prompt "
                f"architecture ({self.pgn_model_type}); first incompatible keys: "
                f"{critical[:10]}"
            )
        del checkpoint, state_dict
        self.model.eval().to(device=self.device, dtype=self.dtype)
        with working_directory(self.repository):
            from clip import clip

        self.clip = clip

    def set_task_prompt(self, task_name: str, manifest_prompt: str) -> None:
        self.task_name = task_name
        self.task_label = HIDDEN_SHOT_TASK_LABELS.get(
            task_name, task_name.replace("_", " ")
        )
        self.manifest_text_prompt = manifest_prompt

    def _open_resized(self, relative_path: str) -> np.ndarray:
        with Image.open(self.data_root / relative_path) as source:
            image = source.convert("RGB")
            image = image.resize(
                (self.resolution, self.resolution), RESAMPLING.BICUBIC
            )
            return np.asarray(image, dtype=np.float32) / 255.0

    def run(self, condition: str) -> InferenceResult:
        if (
            condition != "official"
            or self.sample is None
            or self.model is None
            or self.clip is None
        ):
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        text = f"This is a photo of a {self.task_label} task"
        with StageTimer(stages, "preprocess"):
            demo_input = self._open_resized(self.sample.task_a_input)
            demo_output = self._open_resized(self.sample.task_a_output)
            query = self._open_resized(self.sample.task_b_input)
            images = np.concatenate((demo_input, query), axis=0)
            targets = np.concatenate((demo_output, demo_output), axis=0)
            images = (images - IMAGENET_MEAN) / IMAGENET_STD
            targets = (targets - IMAGENET_MEAN) / IMAGENET_STD
            images_tensor = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0)
            targets_tensor = torch.from_numpy(targets).permute(2, 0, 1).unsqueeze(0)
            images_tensor = images_tensor.to(device=self.device, dtype=self.dtype)
            targets_tensor = targets_tensor.to(device=self.device, dtype=self.dtype)
            text_tokens = self.clip.tokenize(text).to(self.device)
            mask = torch.zeros(self.model.patch_embed.num_patches, device=self.device)
            mask[self.model.patch_embed.num_patches // 2 :] = 1
            mask = mask.unsqueeze(0)

        with StageTimer(stages, "model_forward"):
            with torch.inference_mode():
                latent = self.model.forward_encoder(
                    images_tensor, targets_tensor, mask
                )
                prediction = self.model.forward_decoder(
                    latent, images_tensor, text_tokens
                )

        with StageTimer(stages, "postprocess"):
            prediction = prediction.permute(0, 2, 3, 1).detach().cpu()
            output = prediction[0, prediction.shape[1] // 2 :, :, :]
            output = torch.clip(
                (
                    output * torch.as_tensor(IMAGENET_STD)
                    + torch.as_tensor(IMAGENET_MEAN)
                )
                * 255,
                0,
                255,
            )
            image = Image.fromarray(output.to(torch.uint8).numpy())
        if image.size != (self.resolution, self.resolution):
            raise RuntimeError(
                "Hidden-Shot output-size mismatch: expected "
                f"{(self.resolution, self.resolution)}, got {image.size}"
            )
        return InferenceResult(
            output=image,
            stage_seconds=stages,
            metadata={
                "output_size": list(image.size),
                "input_direction": (
                    "[demo input, demo output, query input] + task name -> query output"
                ),
                "task_name": self.task_name,
                "task_label": self.task_label,
                "clip_text": text,
                "sample": self.sample.as_dict(),
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        if condition != "official" or self.model is None:
            raise ValueError(condition)
        return {
            "hidden_shot_complete": self.model,
            "learned_prompt_module": self.model.prompt_module,
        }

    def trained_parameter_count(self, condition: str) -> int | None:
        if condition != "official" or self.model is None:
            raise ValueError(condition)
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

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
            "architecture": "models_painter2.Painter + PGNCLIP",
            "checkpoint": self.checkpoint,
            "checkpoint_missing_keys": self.checkpoint_missing_keys,
            "checkpoint_unexpected_keys": self.checkpoint_unexpected_keys,
            "clip_architecture": self.clip_architecture,
            "pgn_model_type": self.pgn_model_type,
            "pgn_model_type_source": (
                "checkpoint_auto_detection"
                if self.requested_pgn_model_type == "auto"
                else "command_line"
            ),
            "resolution": [self.resolution * 2, self.resolution],
            "output_resolution": [self.resolution, self.resolution],
            "device": self.device,
            "dtype": str(self.dtype),
            "task_name": self.task_name,
            "task_label": self.task_label,
            "clip_text": f"This is a photo of a {self.task_label} task",
            "manifest_text_prompt": self.manifest_text_prompt,
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The released models_painter2 inference path is preserved: the Painter "
                "canvas contains demo input/query as image and demo output/placeholder as "
                "target; PGNCLIP receives the image canvas and task-name tokens. The unused "
                "SmoothL1 evaluation loss is skipped. Parameters, FLOPs, and latency include "
                "the learned PGN, frozen CLIP encoders, and Painter image generator."
            ),
        }


class PromptDiffusionAdapter(EfficiencyAdapter):
    name = "prompt-diffusion"
    protocol = "[demo input, demo output, query] + text -> query output (diffusion)"

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        sample_index: int = 0,
        model_id: str = "zhendongw/prompt-diffusion-diffusers",
        device: str = "cuda",
        dtype: str = "fp16",
        steps: int = 50,
        seed: int = 2023,
        text_prompt: str = "perform the demonstrated visual task",
        resolution: int | None = None,
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.sample_index = sample_index
        self.model_id = resolve_model_reference(model_id, self.repository)
        self.device = device
        self.dtype = torch_dtype(dtype)
        self.steps = steps
        self.seed = seed
        self.text_prompt = text_prompt
        self.resolution = resolution
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.pipeline = None

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
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
            from diffusers import DDIMScheduler
            from pipeline_prompt_diffusion import PromptDiffusionPipeline
            from promptdiffusioncontrolnet import PromptDiffusionControlNetModel

            controlnet = PromptDiffusionControlNetModel.from_pretrained(
                self.model_id, subfolder="controlnet", torch_dtype=self.dtype
            )
            self.pipeline = PromptDiffusionPipeline.from_pretrained(
                self.model_id, controlnet=controlnet, torch_dtype=self.dtype
            )
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.pipeline is None:
            raise ValueError(condition)
        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            demo_input = self._load_rgb(self.sample.task_a_input)
            demo_output = self._load_rgb(self.sample.task_a_output)
            query = self._load_rgb(self.sample.task_b_input)
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
        with StageTimer(stages, "diffusion_generation"):
            try:
                with torch.inference_mode():
                    output = self.pipeline(
                        self.text_prompt,
                        num_inference_steps=self.steps,
                        generator=generator,
                        image_pair=[demo_input, demo_output],
                        image=query,
                        height=self.resolution,
                        width=self.resolution,
                    ).images[0]
            finally:
                demo_input.close()
                demo_output.close()
                query.close()
        output_size = getattr(output, "size", None)
        if self.resolution is not None and output_size != (
            self.resolution,
            self.resolution,
        ):
            raise RuntimeError(
                "Prompt-Diffusion ignored the controlled output size: "
                f"expected {(self.resolution, self.resolution)}, got {output_size}"
            )
        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "output_size": list(output_size) if output_size is not None else None,
                "input_direction": "[demo input, demo output, query input] -> query output",
                "sample": self.sample.as_dict(),
            },
        )

    def _load_rgb(self, relative_path: str) -> Image.Image:
        with Image.open(self.data_root / relative_path) as source:
            image = source.convert("RGB").copy()
        if self.resolution is not None:
            image = image.resize(
                (self.resolution, self.resolution), RESAMPLING.BICUBIC
            )
        return image

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {"prompt_diffusion_pipeline": self.pipeline}

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
            "model_id": self.model_id,
            "scheduler": "DDIMScheduler",
            "steps": self.steps,
            "device": self.device,
            "dtype": str(self.dtype),
            "seed": self.seed,
            "resolution": (
                [self.resolution, self.resolution] if self.resolution else None
            ),
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The official image_pair=[demo input, demo output], image=query call is used. "
                "Prompt-Diffusion was designed primarily for same-task prompting; cross-task input "
                "here standardizes efficiency measurement, not task accuracy."
            ),
        }


class InstructDiffusionAdapter(EfficiencyAdapter):
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
        steps: int = 50,
        seed: int = 42,
        text_prompt: str = "Restore the image according to the instruction.",
        cfg_text: float = 3.5,
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
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(
                "InstructDiffusion requires its official checkpoint; pass --checkpoint"
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
