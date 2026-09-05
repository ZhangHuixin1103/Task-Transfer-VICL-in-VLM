"""Check a model environment before comparison runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from typing import Sequence


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_visualcloze_environment() -> dict[str, object]:
    versions = {
        package: _version(package)
        for package in (
            "numpy",
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "flash-attn",
            "opencv-python",
        )
    }
    errors = []
    numpy_version = versions["numpy"]
    if numpy_version is None or int(numpy_version.split(".")[0]) >= 2:
        errors.append(f"NumPy must be 1.x, found {numpy_version}")
    if versions["diffusers"] != "0.31.0":
        errors.append(
            f"Diffusers must be 0.31.0 with the CUDA PyTorch 2.1 build, found "
            f"{versions['diffusers']}"
        )
    torch_version = versions["torch"]
    if torch_version is None or not torch_version.startswith("2.1.0"):
        errors.append(f"PyTorch must be 2.1.0, found {torch_version}")

    import_error = None
    if not errors:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            import torch  # noqa: F401
            from diffusers.models import AutoencoderKL  # noqa: F401
            from einops import rearrange  # noqa: F401
            from flash_attn import flash_attn_varlen_func  # noqa: F401
            from flash_attn.bert_padding import (  # noqa: F401
                index_first_axis,
                pad_input,
                unpad_input,
            )
            from imwatermark import WatermarkEncoder  # noqa: F401
            from safetensors.torch import load_file  # noqa: F401
            from torchdiffeq import odeint  # noqa: F401
            from torchvision import transforms  # noqa: F401
            from torchvision.transforms.functional import to_pil_image  # noqa: F401
            from transformers import (  # noqa: F401
                CLIPTextModel,
                CLIPTokenizer,
                T5EncoderModel,
                T5Tokenizer,
            )
        except Exception as error:  # dependency imports expose several exception types
            import_error = f"{type(error).__name__}: {error}"
            errors.append(import_error)
    return {
        "status": "pass" if not errors else "fail",
        "versions": versions,
        "import_error": import_error,
        "errors": errors,
    }


def inspect_environment() -> dict[str, object]:
    import torch

    try:
        from torch.utils.flop_counter import FlopCounterMode  # noqa: F401

        dispatch_flops = True
        dispatch_error = None
    except (ImportError, AttributeError) as error:
        dispatch_flops = False
        dispatch_error = f"{type(error).__name__}: {error}"

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "dispatch_flop_counter_available": dispatch_flops,
        "dispatch_flop_counter_error": dispatch_error,
        "paper_flops_preflight": (
            "pass"
            if dispatch_flops
            else "fail: only the partial module-hook fallback is possible"
        ),
        "note": (
            "A pass confirms the required counter API, not model-specific operator "
            "coverage. The benchmark JSON must still report flops.status=ok."
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--require-dispatch-flops",
        action="store_true",
        help="Exit nonzero when PyTorch dispatch FLOP counting is unavailable",
    )
    result.add_argument(
        "--require-visualcloze",
        action="store_true",
        help="Exit nonzero unless the pinned VisualCloze runtime imports cleanly",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    visualcloze = (
        inspect_visualcloze_environment() if args.require_visualcloze else None
    )
    if visualcloze is not None and visualcloze["status"] != "pass":
        print(
            json.dumps(
                {"python": platform.python_version(), "visualcloze": visualcloze},
                indent=2,
            )
        )
        raise SystemExit(2)
    report = inspect_environment()
    if visualcloze is not None:
        report["visualcloze"] = visualcloze
    print(json.dumps(report, indent=2))
    if args.require_dispatch_flops and not report["dispatch_flop_counter_available"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
