from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class VICLSample:
    task_a_input: str
    task_a_output: str
    task_b_input: str
    task_b_output: str | None = None

    def as_dict(self) -> Dict[str, str | None]:
        return {
            "task_a_input": self.task_a_input,
            "task_a_output": self.task_a_output,
            "task_b_input": self.task_b_input,
            "task_b_output": self.task_b_output,
        }


@dataclass
class InferenceResult:
    output: Any
    stage_seconds: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_dataset_records(dataset_json: Path) -> list[dict[str, Any]]:
    import json

    with dataset_json.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {dataset_json}")
    if not records:
        raise ValueError(f"No records found in {dataset_json}")
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"Every record in {dataset_json} must be a JSON object")
    return records


def load_vicl_sample(
    dataset_json: Path,
    sample_index: int,
    demo_input: str | None = None,
    demo_output: str | None = None,
) -> VICLSample:
    records = load_dataset_records(dataset_json)
    return vicl_sample_from_records(
        records,
        sample_index,
        demo_input=demo_input,
        demo_output=demo_output,
        source=str(dataset_json),
    )


def select_dataset_records(
    records: Sequence[dict[str, Any]],
    record_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Select a stable task subset once, outside the timed inference loop."""
    if record_indices is None:
        return list(records)
    selected: list[dict[str, Any]] = []
    for index in record_indices:
        if index < 0 or index >= len(records):
            raise IndexError(
                f"record index {index} is outside [0, {len(records) - 1}]"
            )
        selected.append(records[index])
    if not selected:
        raise ValueError("record_indices selected no records")
    return selected


def vicl_sample_from_records(
    records: Sequence[dict[str, Any]],
    sample_index: int,
    demo_input: str | None = None,
    demo_output: str | None = None,
    source: str = "cached records",
) -> VICLSample:
    """Map an already-loaded record to the common three-image VICL interface."""
    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(
            f"sample_index={sample_index} is outside [0, {len(records) - 1}]"
        )
    item = records[sample_index]
    if all(key in item for key in ("taskA_input", "taskA_output", "taskB_input")):
        return VICLSample(
            task_a_input=item["taskA_input"],
            task_a_output=item["taskA_output"],
            task_b_input=item["taskB_input"],
            task_b_output=item.get("taskB_output"),
        )

    missing = [key for key in ("image_path", "target_path") if key not in item]
    if missing:
        raise KeyError(
            f"Record {sample_index} in {source} is neither T2T-VICL nor paired-image "
            f"schema; missing {missing}"
        )
    if not demo_input or not demo_output:
        raise ValueError(
            "Paired-image JSON requires explicit demo_input and demo_output paths"
        )
    return VICLSample(
        task_a_input=demo_input,
        task_a_output=demo_output,
        task_b_input=item["image_path"],
        task_b_output=item["target_path"],
    )


class ComparisonAdapter(ABC):
    """One process-local adapter around an official model inference path."""

    name: str
    protocol: str

    @property
    @abstractmethod
    def conditions(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, condition: str) -> InferenceResult:
        raise NotImplementedError

    @abstractmethod
    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        raise NotImplementedError

    def prepare_condition(self, condition: str) -> None:
        """Load resources used only by one benchmark condition."""

    def release_condition(self, condition: str) -> None:
        """Release resources that must not contaminate another condition."""

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        return {}

    def trained_parameter_count(self, condition: str) -> int | None:
        """Return method-trained parameters when they differ from requires_grad."""
        return None

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        """Switch the query dataset without reloading model weights."""
        raise NotImplementedError(f"{self.name} does not support dataset switching")

    def sample_count(self) -> int:
        raise NotImplementedError(f"{self.name} does not expose a query dataset")

    def select_sample(self, sample_index: int) -> None:
        raise NotImplementedError(f"{self.name} does not support sample selection")

    def close(self) -> None:
        pass
