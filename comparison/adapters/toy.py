from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import torch

from ..base import ComparisonAdapter, InferenceResult


class ToyAdapter(ComparisonAdapter):
    name = "toy"
    protocol = "synthetic tensor -> synthetic tensor"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.inputs = None

    @property
    def conditions(self) -> Iterable[str]:
        return ("toy",)

    def setup(self) -> None:
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(8, 3, 3, padding=1),
        ).to(self.device)
        self.model.eval()
        self.inputs = torch.randn(1, 3, 32, 32, device=self.device)

    def run(self, condition: str) -> InferenceResult:
        if condition != "toy" or self.model is None or self.inputs is None:
            raise ValueError(condition)
        with torch.inference_mode():
            output = self.model(self.inputs)
        return InferenceResult(output=output)

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        return {"toy_model": self.model}

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {"condition": condition, "input_shape": [1, 3, 32, 32]}
