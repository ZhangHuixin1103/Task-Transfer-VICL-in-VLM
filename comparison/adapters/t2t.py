from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from ..base import (
    ComparisonAdapter,
    InferenceResult,
    VICLSample,
    load_dataset_records,
    select_dataset_records,
    vicl_sample_from_records,
)
from ..metrics import StageTimer, logical_parameter_numel
from ..prompting import INSTRUCT_TEXT, generate_text_prompt
from .common import import_from_root, resolve_model_reference, torch_dtype


FIXED_PROMPT = INSTRUCT_TEXT

DEFAULT_MODEL_IDS = {
    "qwen": "Qwen/Qwen-Image-Edit-2511",
    "flux2": "diffusers/FLUX.2-dev-bnb-4bit",
    "omnigen2": "OmniGen2/OmniGen2",
    "firered": "FireRedTeam/FireRed-Image-Edit-1.1",
}

BACKEND_MODULES = {
    "qwen": "eval_qwen",
    "flux2": "eval_flux",
    "omnigen2": "eval_omnigen",
    "firered": "eval_firered",
}


class T2TVICLAdapter(ComparisonAdapter):
    protocol = "[A_in, A_out, B_in] + text prompt -> B_out"

    def __init__(
        self,
        project_root: Path,
        backend: str,
        requested_conditions: Sequence[str],
        dataset_json: Path,
        data_root: Path,
        sample_index: int = 0,
        model_id: str | None = None,
        prompt_checkpoint: str = "Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875",
        prompt_base_model: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        dtype: str = "bf16",
        optimized: bool = False,
        seed: int = 42,
        steps: int | None = None,
        resolution: int | None = None,
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        if backend not in BACKEND_MODULES:
            raise ValueError(f"Unsupported T2T-VICL backend: {backend}")
        normalized_dtype = dtype.lower()
        if backend in {"flux2", "omnigen2", "firered"} and normalized_dtype not in {
            "bf16",
            "bfloat16",
        }:
            raise ValueError(
                f"{backend} uses the repository's fixed BF16 inference path; "
                f"--dtype {dtype} would be misleading"
            )
        if backend in {"flux2", "omnigen2", "firered"} and device not in {
            "cuda",
            "cuda:0",
        }:
            raise ValueError(
                f"{backend} uses CUDA device 0 in the repository's official path"
            )
        invalid = set(requested_conditions) - {"fixed", "ours"}
        if invalid:
            raise ValueError(f"Unsupported T2T-VICL conditions: {sorted(invalid)}")
        self.project_root = project_root.resolve()
        self.backend = backend
        self.name = f"t2t-{backend}"
        self._conditions = tuple(requested_conditions)
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.sample_index = sample_index
        self.model_id = model_id or DEFAULT_MODEL_IDS[backend]
        self.prompt_checkpoint = resolve_model_reference(
            prompt_checkpoint, self.project_root
        )
        self.prompt_base_model = prompt_base_model
        self.device = device
        self.dtype = torch_dtype(dtype)
        self.optimized = optimized
        self.seed = seed
        self.steps = steps
        self.resolution = resolution
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.backend_module = None
        self.editor = None
        self.prompt_model = None
        self.prompt_processor = None
        self._prompt_trained_parameters: int | None = None
        self._prompt_training_policy: str | None = None

    @property
    def conditions(self) -> Iterable[str]:
        return self._conditions

    def setup(self) -> None:
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)
        self.backend_module = import_from_root(
            BACKEND_MODULES[self.backend], self.project_root
        )
        self.backend_module.DATA_TASKS_DIR = str(self.data_root)
        self.editor = self._load_editor()

    def prepare_condition(self, condition: str) -> None:
        if condition not in self._conditions:
            raise ValueError(condition)
        if condition == "ours" and self.prompt_model is None:
            self._load_prompt_generator()

    def release_condition(self, condition: str) -> None:
        if condition != "ours":
            return
        self.prompt_model = None
        self.prompt_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_editor(self):
        module = self.backend_module
        assert module is not None
        if self.backend == "qwen":
            pipeline = module.QwenImageEditPlusPipeline.from_pretrained(
                self.model_id, torch_dtype=self.dtype
            )
            pipeline.to(self.device)
            pipeline.set_progress_bar_config(disable=True)
            return pipeline
        if self.backend == "flux2":
            module.REPO_ID = self.model_id
            return module.load_flux_pipeline()
        if self.backend == "omnigen2":
            module.OMNIGEN_MODEL_PATH = self.model_id
            return module.load_omnigen_pipeline()
        if self.backend == "firered":
            return module.load_firered_pipeline(self.model_id, optimized=self.optimized)
        raise AssertionError(self.backend)

    def _load_prompt_generator(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        checkpoint = self.prompt_checkpoint
        if os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
            from peft import PeftModel, get_peft_model_state_dict

            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.prompt_base_model, torch_dtype="auto", device_map="auto"
            )
            prompt_model = PeftModel.from_pretrained(base_model, checkpoint)
            adapter_state = get_peft_model_state_dict(prompt_model)
            seen_tensors: set[tuple[Any, ...]] = set()
            adapter_parameters = 0
            for tensor in adapter_state.values():
                if not isinstance(tensor, torch.Tensor):
                    continue
                try:
                    key = (
                        tensor.device.type,
                        tensor.device.index,
                        tensor.untyped_storage().data_ptr(),
                        tensor.storage_offset(),
                        tuple(tensor.shape),
                    )
                except (AttributeError, RuntimeError, NotImplementedError):
                    key = ("tensor", id(tensor))
                if key not in seen_tensors:
                    seen_tensors.add(key)
                    adapter_parameters += tensor.numel()
            if not adapter_parameters:
                raise RuntimeError(
                    f"PEFT checkpoint {checkpoint} exposed no adapter parameters"
                )
            self._prompt_trained_parameters = adapter_parameters
            self._prompt_training_policy = (
                "PEFT adapter parameters from get_peft_model_state_dict; base weights frozen"
            )
            try:
                prompt_model = prompt_model.merge_and_unload()
            except (AttributeError, RuntimeError):
                pass
        else:
            prompt_model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint, torch_dtype="auto", device_map="auto"
            )
            self._prompt_training_policy = (
                "visual tower frozen; visual merger, language model, and lm_head trained"
            )
        prompt_model.eval()
        self.prompt_model = prompt_model
        self.prompt_processor = AutoProcessor.from_pretrained(self.prompt_base_model)

    def _generate_image(self, prompt: str):
        assert self.sample is not None and self.backend_module is not None
        sample = self.sample
        if self.backend == "qwen":
            paths = [sample.task_a_input, sample.task_a_output, sample.task_b_input]
            return self.backend_module.generate_image_qwen(
                self.editor,
                paths,
                prompt,
                seed=self.seed,
                generator_device=self.device,
                num_inference_steps=self.steps if self.steps is not None else 40,
                input_resolution=self.resolution,
                height=self.resolution,
                width=self.resolution,
            )
        if self.backend == "flux2":
            return self.backend_module.generate_image_flux(
                self.editor,
                sample.task_a_input,
                sample.task_a_output,
                sample.task_b_input,
                prompt,
                seed=self.seed,
                num_inference_steps=self.steps if self.steps is not None else 30,
                input_resolution=self.resolution,
                height=self.resolution,
                width=self.resolution,
            )
        if self.backend == "omnigen2":
            return self.backend_module.generate_image_omnigen(
                self.editor,
                sample.task_a_input,
                sample.task_a_output,
                sample.task_b_input,
                prompt,
                seed=self.seed,
                num_inference_steps=self.steps if self.steps is not None else 50,
                input_resolution=self.resolution,
                height=self.resolution or 1024,
                width=self.resolution or 1024,
            )
        if self.backend == "firered":
            kwargs = {"seed": self.seed}
            if self.steps is not None:
                kwargs["num_inference_steps"] = self.steps
            kwargs.update(
                input_resolution=self.resolution,
                height=self.resolution,
                width=self.resolution,
            )
            return self.backend_module.generate_image_firered(
                self.editor,
                sample.task_a_input,
                sample.task_a_output,
                sample.task_b_input,
                prompt,
                **kwargs,
            )
        raise AssertionError(self.backend)

    def run(self, condition: str) -> InferenceResult:
        if condition not in self._conditions:
            raise ValueError(f"Condition {condition!r} was not configured")
        assert self.sample is not None
        stages: Dict[str, Any] = {}
        prompt = FIXED_PROMPT
        if condition == "ours":
            if self.prompt_model is None or self.prompt_processor is None:
                raise RuntimeError("Prompt generator was not loaded")
            with StageTimer(stages, "prompt_generation"):
                prompt = generate_text_prompt(
                    self.sample.task_a_input,
                    self.sample.task_a_output,
                    self.sample.task_b_input,
                    model=self.prompt_model,
                    processor=self.prompt_processor,
                    data_tasks_dir=self.data_root,
                    input_resolution=self.resolution,
                )
        with StageTimer(stages, "image_generation"):
            output = self._generate_image(prompt)
        if output is None:
            raise RuntimeError(f"{self.name} returned no image")
        output_size = getattr(output, "size", None)
        if self.resolution is not None and output_size != (
            self.resolution,
            self.resolution,
        ):
            raise RuntimeError(
                f"{self.name} ignored the controlled output size: "
                f"expected {(self.resolution, self.resolution)}, got {output_size}"
            )
        prompt_tokens = None
        if self.prompt_processor is not None:
            tokenizer = getattr(self.prompt_processor, "tokenizer", None)
            if tokenizer is not None:
                try:
                    encoded = tokenizer(prompt, add_special_tokens=False)
                    input_ids = encoded["input_ids"]
                    prompt_tokens = len(
                        input_ids[0]
                        if input_ids and isinstance(input_ids[0], list)
                        else input_ids
                    )
                except (KeyError, TypeError, ValueError):
                    prompt_tokens = None
        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "prompt_characters": len(prompt),
                "editor_prompt_tokens": prompt_tokens,
                "output_size": list(output_size) if output_size is not None else None,
                "input_direction": "[A_input, A_output, B_input] -> B_output",
                "sample": self.sample.as_dict(),
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        components: Dict[str, Any] = {"image_editor": self.editor}
        if condition == "ours":
            components["prompt_generator"] = self.prompt_model
        return components

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

    def trained_parameter_count(self, condition: str) -> int | None:
        if condition != "ours" or self.prompt_model is None:
            return 0
        if self._prompt_trained_parameters is not None:
            return self._prompt_trained_parameters
        seen: set[int] = set()
        total = 0
        for name, parameter in self.prompt_model.named_parameters():
            prefix = "base_model.model."
            normalized = name[len(prefix) :] if name.startswith(prefix) else name
            selected = (
                "language_model" in normalized
                or "lm_head" in normalized
                or "visual.merger" in normalized
            )
            if selected and id(parameter) not in seen:
                seen.add(id(parameter))
                total += logical_parameter_numel(parameter)[0]
        return total

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        defaults = {
            "qwen": {
                "steps": self.steps if self.steps is not None else 40,
                "height": self.resolution,
                "width": self.resolution,
                "true_cfg_scale": 4.0,
                "guidance_scale": 1.0,
                "seed": self.seed,
            },
            "flux2": {
                "steps": self.steps if self.steps is not None else 30,
                "height": self.resolution,
                "width": self.resolution,
                "guidance_scale": 4.0,
                "cpu_offload": True,
                "seed": self.seed,
            },
            "omnigen2": {
                "steps": self.steps if self.steps is not None else 50,
                "height": self.resolution or 1024,
                "width": self.resolution or 1024,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 3.0,
                "seed": self.seed,
            },
            "firered": {
                "steps": self.steps if self.steps is not None else 40,
                "height": self.resolution,
                "width": self.resolution,
                "true_cfg_scale": 4.0,
                "optimized": self.optimized,
                "seed": self.seed,
            },
        }
        return {
            "condition": condition,
            "backend": self.backend,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "prompt_checkpoint": (
                self.prompt_checkpoint if condition == "ours" else None
            ),
            "prompt_training_policy": (
                self._prompt_training_policy
                if condition == "ours"
                else None
            ),
            "sampling": defaults[self.backend],
            "controlled_resolution": (
                [self.resolution, self.resolution] if self.resolution else None
            ),
            "prompt_generator_visual_resolution": (
                [self.resolution, self.resolution]
                if condition == "ours" and self.resolution
                else None
            ),
            "sample": self.sample.as_dict() if self.sample else None,
            "condition_resource_policy": (
                "prompt generator is loaded only for Ours and absent during Fixed measurement"
                if condition == "ours"
                else "image editor only; prompt generator is not resident"
            ),
        }

    def close(self) -> None:
        self.release_condition("ours")
        self.editor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
