"""Check whether one model environment can produce auditable runtime FLOPs."""

from __future__ import annotations

import argparse
import json
import platform
from typing import Sequence

import torch


def inspect_environment() -> dict[str, object]:
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
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    report = inspect_environment()
    print(json.dumps(report, indent=2))
    if args.require_dispatch_flops and not report["dispatch_flop_counter_available"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
