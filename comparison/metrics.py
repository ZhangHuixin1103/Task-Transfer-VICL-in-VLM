from __future__ import annotations

import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

import torch

from .base import InferenceResult
from .flops import dispatch_flop_count, module_hook_flop_count


@dataclass(frozen=True)
class ParameterStats:
    total: int
    trainable: int
    storage_bytes: int
    stored_elements: int
    tensors: int
    quantized_tensors: int
    meta_tensors: int
    logical_shape_sources: Dict[str, int]


def iter_modules(value: Any) -> Iterable[torch.nn.Module]:
    """Find modules inside nn.Module, Diffusers pipelines, and containers."""
    seen_objects: set[int] = set()

    def visit(obj: Any) -> Iterable[torch.nn.Module]:
        if obj is None or id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            yield obj
            return
        if isinstance(obj, Mapping):
            for child in obj.values():
                yield from visit(child)
            return
        if isinstance(obj, (list, tuple, set)):
            for child in obj:
                yield from visit(child)
            return
        components = getattr(obj, "components", None)
        if isinstance(components, Mapping):
            yield from visit(components)

    yield from visit(value)


def _owned_parameter_objects(
    value: Any,
) -> Iterable[tuple[torch.nn.Parameter, torch.nn.Module]]:
    seen_parameters: set[int] = set()
    seen_modules: set[int] = set()
    for root in iter_modules(value):
        for module in root.modules():
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            for parameter in module.parameters(recurse=False):
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                yield parameter, module


def logical_parameter_numel(
    parameter: torch.nn.Parameter, owner: torch.nn.Module | None = None
) -> tuple[int, str]:
    quant_state = getattr(parameter, "quant_state", None)
    quant_shape = getattr(quant_state, "shape", None)
    if quant_shape is not None:
        try:
            return math.prod(int(value) for value in quant_shape), "quant_state.shape"
        except (TypeError, ValueError):
            pass

    parameter_owner = owner if owner is not None else getattr(parameter, "module", None)
    owner_module = (
        parameter_owner.__class__.__module__.lower()
        if parameter_owner is not None
        else ""
    )
    owner_class = (
        parameter_owner.__class__.__name__.lower()
        if parameter_owner is not None
        else ""
    )
    if "bitsandbytes" in owner_module and "linear" in owner_class:
        in_features = getattr(parameter_owner, "in_features", None)
        out_features = getattr(parameter_owner, "out_features", None)
        if in_features is not None and out_features is not None:
            return int(in_features) * int(out_features), "bitsandbytes.linear_shape"

    ds_numel = getattr(parameter, "ds_numel", None)
    if ds_numel is not None:
        try:
            return int(ds_numel), "deepspeed.ds_numel"
        except (TypeError, ValueError):
            pass

    return parameter.numel(), "tensor.shape"


def _physical_storage_bytes(parameter: torch.nn.Parameter) -> tuple[int, Any]:
    if parameter.device.type == "meta":
        return 0, ("meta", id(parameter))
    try:
        storage = parameter.untyped_storage()
        key = (
            parameter.device.type,
            parameter.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        )
        return storage.nbytes(), key
    except (AttributeError, RuntimeError, NotImplementedError):
        size = parameter.numel() * parameter.element_size()
        return size, ("tensor", id(parameter))


def count_parameters(value: Any) -> ParameterStats:
    total = 0
    trainable = 0
    storage_bytes = 0
    stored_elements = 0
    tensors = 0
    quantized_tensors = 0
    meta_tensors = 0
    logical_shape_sources: MutableMapping[str, int] = defaultdict(int)
    seen_storages: set[Any] = set()
    for parameter, owner in _owned_parameter_objects(value):
        count, source = logical_parameter_numel(parameter, owner=owner)
        total += count
        trainable += count if parameter.requires_grad else 0
        stored_elements += parameter.numel()
        physical_bytes, storage_key = _physical_storage_bytes(parameter)
        if storage_key not in seen_storages:
            seen_storages.add(storage_key)
            storage_bytes += physical_bytes
        tensors += 1
        logical_shape_sources[source] += 1
        if source != "tensor.shape":
            quantized_tensors += 1
        if parameter.device.type == "meta":
            meta_tensors += 1
    return ParameterStats(
        total=total,
        trainable=trainable,
        storage_bytes=storage_bytes,
        stored_elements=stored_elements,
        tensors=tensors,
        quantized_tensors=quantized_tensors,
        meta_tensors=meta_tensors,
        logical_shape_sources=dict(logical_shape_sources),
    )


def parameter_report(components: Mapping[str, Any]) -> Dict[str, Any]:
    per_component = {
        name: asdict(count_parameters(value)) for name, value in components.items()
    }
    unique = asdict(count_parameters(components))
    return {
        "unique_total": unique,
        "storage_complete": unique["meta_tensors"] == 0,
        "components": per_component,
        "logical_parameter_definition": (
            "Unique logical scalar weights. Packed bitsandbytes tensors use their original "
            "quant_state/module shape rather than packed storage numel."
        ),
        "storage_definition": (
            "Deduplicated physical storage of parameter payload tensors at runtime; optimizer "
            "state, activations, non-parameter buffers, and quantization auxiliary state are excluded."
        ),
    }


def _active_cuda_device_indices() -> list[int]:
    if not torch.cuda.is_available():
        return []
    active = []
    for index in range(torch.cuda.device_count()):
        try:
            if torch.cuda.memory_allocated(index) > 0:
                active.append(index)
        except (RuntimeError, AssertionError):
            continue
    if active:
        return active
    try:
        return [torch.cuda.current_device()]
    except (RuntimeError, AssertionError):
        return []


def synchronize_accelerators() -> None:
    for index in _active_cuda_device_indices():
        try:
            torch.cuda.synchronize(index)
        except (RuntimeError, AssertionError):
            continue


def _reset_peak_memory() -> None:
    for index in _active_cuda_device_indices():
        try:
            torch.cuda.reset_peak_memory_stats(index)
        except (RuntimeError, AssertionError):
            continue


def _empty_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except (RuntimeError, AssertionError):
        pass


def _memory_by_device(kind: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for index in _active_cuda_device_indices():
        try:
            if kind == "allocated":
                value = torch.cuda.max_memory_allocated(index)
            elif kind == "reserved":
                value = torch.cuda.max_memory_reserved(index)
            elif kind == "current_allocated":
                value = torch.cuda.memory_allocated(index)
            else:
                raise ValueError(kind)
            values[f"cuda:{index}"] = value
        except (RuntimeError, AssertionError):
            continue
    return values


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_seconds(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"samples_s": [], "count": 0}
    return {
        "samples_s": list(values),
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "std_s": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_s": min(values),
        "max_s": max(values),
        "p90_s": _percentile(values, 0.90),
        "p95_s": _percentile(values, 0.95),
    }


def release_inference_result(result: InferenceResult) -> None:
    """Release file-backed/generated outputs between long benchmark iterations."""
    pending = [result.output]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set)):
            pending.extend(value)
        else:
            close = getattr(value, "close", None)
            if callable(close):
                close()


def benchmark_callable(
    fn: Callable[[], InferenceResult],
    warmup: int,
    repeats: int,
    seed: int | None = None,
) -> Dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    _empty_cuda_cache()
    for _ in range(warmup):
        seed_everything(seed)
        warmup_result = fn()
        if warmup_result.output is None:
            raise RuntimeError("Inference returned None during warm-up")
        release_inference_result(warmup_result)
        del warmup_result
    synchronize_accelerators()
    _reset_peak_memory()
    resident_memory = _memory_by_device("current_allocated")

    totals: list[float] = []
    stages: MutableMapping[str, list[float]] = defaultdict(list)
    output_metadata: list[Dict[str, Any]] = []
    for _ in range(repeats):
        seed_everything(seed)
        synchronize_accelerators()
        start = time.perf_counter_ns()
        result = fn()
        synchronize_accelerators()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        if result.output is None:
            raise RuntimeError("Inference returned None during measurement")
        totals.append(elapsed)
        for name, measurement in result.stage_seconds.items():
            if hasattr(measurement, "resolve"):
                seconds = measurement.resolve()
            else:
                seconds = float(measurement)
            stages[name].append(seconds)
        output_metadata.append(result.metadata)
        release_inference_result(result)
        del result

    return {
        "end_to_end": summarize_seconds(totals),
        "stages": {name: summarize_seconds(values) for name, values in stages.items()},
        "resident_cuda_memory_bytes": resident_memory,
        "peak_cuda_memory_allocated_bytes": _memory_by_device("allocated"),
        "peak_cuda_memory_reserved_bytes": _memory_by_device("reserved"),
        # Backward-compatible alias.
        "peak_cuda_memory_bytes": _memory_by_device("allocated"),
        "run_metadata": output_metadata,
        "timing_definition": (
            "Synchronized wall-clock time around the complete adapter call with barrier-free "
            "stage event instrumentation; model loading and warm-up are excluded, "
            "preprocessing and postprocessing are included."
        ),
        "stage_timing_definition": (
            "Diagnostic, non-additive stage spans: max(host scope, CUDA current-stream event "
            "elapsed time) resolved after the end-to-end synchronization. No stage-boundary "
            "barriers are inserted, so end-to-end latency is the paper-grade timing."
        ),
        "allocator_policy": (
            "Empty unused CUDA allocator cache before warm-up, then reset peak statistics after "
            "warm-up; report allocated and reserved peaks separately."
        ),
        "seed_policy": (
            f"Reset Python, NumPy, CPU, and CUDA RNGs to seed {seed} before every repeat"
            if seed is not None
            else "RNG state was not reset between repeats"
        ),
    }


def benchmark_dataset_callable(
    fn: Callable[[int], InferenceResult],
    sample_indices: Sequence[int],
    warmup_indices: Sequence[int],
    seed: int | None = None,
    prepare_sample: Callable[[int], None] | None = None,
) -> Dict[str, Any]:
    """Measure one warm end-to-end inference for each distinct query sample."""
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_indices must be unique")

    _empty_cuda_cache()
    for sample_index in warmup_indices:
        seed_everything(seed)
        if prepare_sample is not None:
            prepare_sample(sample_index)
        warmup_result = fn(sample_index)
        if warmup_result.output is None:
            raise RuntimeError("Inference returned None during warm-up")
        release_inference_result(warmup_result)
        del warmup_result
    synchronize_accelerators()
    _reset_peak_memory()
    resident_memory = _memory_by_device("current_allocated")

    totals: list[float] = []
    stages: MutableMapping[str, list[float]] = defaultdict(list)
    rows: list[Dict[str, Any]] = []
    for sample_index in sample_indices:
        seed_everything(seed)
        if prepare_sample is not None:
            prepare_sample(sample_index)
        synchronize_accelerators()
        start = time.perf_counter_ns()
        result = fn(sample_index)
        synchronize_accelerators()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        if result.output is None:
            raise RuntimeError(
                f"Inference returned None for sample_index={sample_index}"
            )
        totals.append(elapsed)
        resolved_stages: Dict[str, float] = {}
        for name, measurement in result.stage_seconds.items():
            seconds = (
                measurement.resolve()
                if hasattr(measurement, "resolve")
                else float(measurement)
            )
            stages[name].append(seconds)
            resolved_stages[name] = seconds
        rows.append(
            {
                "sample_index": sample_index,
                "end_to_end_s": elapsed,
                "stages_s": resolved_stages,
                "metadata": result.metadata,
            }
        )
        release_inference_result(result)
        del result

    return {
        "end_to_end": summarize_seconds(totals),
        "stages": {name: summarize_seconds(values) for name, values in stages.items()},
        "samples": rows,
        "warmup_indices": list(warmup_indices),
        "measured_indices": list(sample_indices),
        "resident_cuda_memory_bytes": resident_memory,
        "peak_cuda_memory_allocated_bytes": _memory_by_device("allocated"),
        "peak_cuda_memory_reserved_bytes": _memory_by_device("reserved"),
        "peak_cuda_memory_bytes": _memory_by_device("allocated"),
        "timing_definition": (
            "Synchronized wall-clock end-to-end latency for one pass over each distinct query. "
            "Model loading and warm-up are excluded; adapter preprocessing and postprocessing "
            "are included. Cached record selection and optional input-file cache priming occur "
            "before the timer, while image open/decode/resize remains inside adapter "
            "preprocessing. Batch size and concurrency are one."
        ),
        "sample_policy": (
            "Each measured dataset index appears exactly once; aggregate statistics therefore "
            "capture query-to-query latency variation instead of rerunning one image."
        ),
        "seed_policy": (
            f"Reset Python, NumPy, CPU, and CUDA RNGs to seed {seed} before every query"
            if seed is not None
            else "RNG state was not reset between queries"
        ),
    }


def _finalize_dispatch_report(dispatch: Dict[str, Any]) -> Dict[str, Any]:
    dispatch["flop_convention"] = dispatch["convention"]
    if dispatch.get("unsupported_nontrivial_operators"):
        dispatch["status"] = "partial"
    return dispatch


def profile_flops(
    fn: Callable[[], InferenceResult],
    components: Mapping[str, Any] | None = None,
    top_k: int = 20,
    seed: int | None = None,
) -> Dict[str, Any]:
    modules = list(iter_modules(components or {}))
    seed_everything(seed)
    synchronize_accelerators()
    try:
        dispatch = dispatch_flop_count(fn, modules=modules, top_k=top_k)
    except Exception as error:
        oom_type = getattr(torch, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(error, oom_type):
            raise
        seed_everything(seed)
        synchronize_accelerators()
        dispatch = module_hook_flop_count(
            fn,
            modules=modules,
            top_k=top_k,
            reason=f"Dispatch-level counting failed: {type(error).__name__}: {error}",
        )
    synchronize_accelerators()
    return _finalize_dispatch_report(dispatch)


@dataclass
class DeferredStageMeasurement:
    host_seconds: float
    cuda_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]]

    def resolve(self) -> float:
        device_seconds = []
        for _, start, end in self.cuda_events:
            try:
                device_seconds.append(start.elapsed_time(end) / 1000.0)
            except (RuntimeError, AssertionError):
                continue
        return max([self.host_seconds, *device_seconds])


class StageTimer:
    """Barrier-free stage timer resolved after the outer inference synchronization."""

    def __init__(self, destination: MutableMapping[str, Any], name: str):
        self.destination = destination
        self.name = name
        self.started_ns = 0
        self.cuda_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []

    def __enter__(self) -> "StageTimer":
        if torch.cuda.is_available():
            for index in _active_cuda_device_indices():
                try:
                    with torch.cuda.device(index):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(torch.cuda.current_stream(index))
                    self.cuda_events.append((index, start, end))
                except (RuntimeError, AssertionError):
                    continue
        self.started_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        host_seconds = (time.perf_counter_ns() - self.started_ns) / 1e9
        for index, _, end in self.cuda_events:
            try:
                with torch.cuda.device(index):
                    end.record(torch.cuda.current_stream(index))
            except (RuntimeError, AssertionError):
                continue
        self.destination[self.name] = DeferredStageMeasurement(
            host_seconds=host_seconds,
            cuda_events=self.cuda_events,
        )
