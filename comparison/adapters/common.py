from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def import_from_root(module_name: str, root: Path):
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    with working_directory(root):
        return importlib.import_module(module_name)


def torch_dtype(name: str) -> torch.dtype:
    choices = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return choices[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def resolve_model_reference(reference: str, project_root: Path) -> str:
    candidate = Path(reference).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return str(candidate.resolve()) if candidate.exists() else reference


def require_model_file(
    reference: str | None,
    description: str,
    *fallbacks: Path,
) -> str:
    """Resolve one required model file and report every location that was checked."""
    candidates: list[Path] = []
    if reference:
        requested = Path(reference).expanduser()
        candidates.append(requested)
    # Model adapters supply the old README's intended nested locations here,
    # keeping already-issued flattened commands recoverable.
    candidates.extend(fallbacks)

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return str(resolved)

    locations = "\n  - ".join(str(path) for path in checked) or "(none)"
    raise FileNotFoundError(
        f"{description} was not found. Checked:\n  - {locations}"
    )


def require_checkpoint_file(
    reference: str | None,
    model_name: str,
    *fallbacks: Path,
) -> str:
    """Resolve one checkpoint while preserving conventional model subdirectories."""
    return require_model_file(reference, f"{model_name} checkpoint", *fallbacks)
