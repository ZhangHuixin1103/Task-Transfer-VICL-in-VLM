from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping

import torch


def _shape_numel(shape: Any) -> int:
    if shape is None:
        return 0
    if isinstance(shape, torch.Size):
        return math.prod(shape)
    sym_int = getattr(torch, "SymInt", int)
    if isinstance(shape, (tuple, list)) and all(
        isinstance(value, (int, sym_int)) for value in shape
    ):
        return math.prod(shape)
    return 0


def _first_tensor_numel(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, (tuple, list)):
        for child in value:
            count = _first_tensor_numel(child)
            if count:
                return count
    if isinstance(value, Mapping):
        for child in value.values():
            count = _first_tensor_numel(child)
            if count:
                return count
    return 0


def _close_inference_output(result: Any) -> None:
    """Release generated images retained only for one profiling inference."""

    def close(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                close(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                close(child)
        else:
            close_method = getattr(value, "close", None)
            if callable(close_method):
                close_method()

    output = getattr(result, "output", None)
    if output is None:
        raise RuntimeError("Inference returned None during FLOPs profiling")
    close(output)


def _output_numel(out_shape: Any) -> int:
    direct = _shape_numel(out_shape)
    if direct:
        return direct
    if isinstance(out_shape, (tuple, list)):
        return _shape_numel(out_shape[0]) if out_shape else 0
    return 0


def _input_numel(args: tuple[Any, ...]) -> int:
    return _shape_numel(args[0]) if args else 0


def _pointwise_formula(cost: int):
    def formula(*args, out_shape=None, **kwargs) -> int:
        return cost * _output_numel(out_shape)

    return formula


def _reduction_formula(cost: int = 1, output_cost: int = 0):
    def formula(*args, out_shape=None, **kwargs) -> int:
        return cost * _input_numel(args) + output_cost * _output_numel(out_shape)

    return formula


def _indexed_reduction_formula(*args, out_shape=None, **kwargs) -> int:
    source_shape = args[3] if len(args) > 3 else None
    return _shape_numel(source_shape)


def _normalization_formula(*args, out_shape=None, **kwargs) -> int:
    # Mean, variance/RMS, normalization, and optional affine transform.
    return 7 * _output_numel(out_shape)


def _softmax_formula(*args, out_shape=None, **kwargs) -> int:
    # max/subtract, exp, reduction, and division per element.
    return 5 * _output_numel(out_shape)


def _log_softmax_formula(*args, out_shape=None, **kwargs) -> int:
    return 6 * _output_numel(out_shape)


def _upsample_formula(taps: int):
    def formula(*args, out_shape=None, **kwargs) -> int:
        # Each tap contributes a multiply and an accumulation.
        return 2 * taps * _output_numel(out_shape)

    return formula


def _addmm_formula(self_shape, a_shape, b_shape, *args, out_shape=None, **kwargs):
    m, k = a_shape
    _, n = b_shape
    return 2 * m * n * k + _output_numel(out_shape)


def _baddbmm_formula(self_shape, a_shape, b_shape, *args, out_shape=None, **kwargs):
    batch, m, k = a_shape
    _, _, n = b_shape
    return 2 * batch * m * n * k + _output_numel(out_shape)


def _scaled_mm_formula(a_shape, b_shape, *args, out_shape=None, **kwargs):
    m, k = a_shape
    _, n = b_shape
    # Matrix multiplication plus input/output scale application.
    return 2 * m * n * k + 2 * _output_numel(out_shape)


def _convolution_formula(
    input_shape,
    weight_shape,
    bias_shape=None,
    stride=None,
    padding=None,
    dilation=None,
    transposed=False,
    output_padding=None,
    groups=1,
    *,
    out_shape=None,
    **kwargs,
):
    output_elements = _output_numel(out_shape)
    if not output_elements or len(weight_shape) < 3:
        return 0
    kernel_elements = math.prod(weight_shape[2:])
    if transposed:
        # ConvTranspose applies an output-channel kernel to each input value.
        multiply_adds = _shape_numel(input_shape) * weight_shape[1] * kernel_elements
    else:
        # weight is [out_channels, in_channels / groups, *kernel].
        multiply_adds = output_elements * weight_shape[1] * kernel_elements
    bias_flops = output_elements if bias_shape is not None else 0
    return 2 * multiply_adds + bias_flops


def _attention_score_elements(
    batch: int,
    heads: int,
    query_length: int,
    key_length: int,
    is_causal: bool,
) -> int:
    if not is_causal:
        visible_per_head = query_length * key_length
    else:
        # PyTorch SDPA uses an upper-left aligned lower-triangular causal bias.
        visible_per_head = sum(
            min(query_index + 1, key_length)
            for query_index in range(query_length)
        )
    return batch * heads * visible_per_head


def _sdpa_formula(
    query_shape,
    key_shape,
    value_shape,
    *args,
    out_shape=None,
    is_causal=False,
    has_attention_bias=False,
    **kwargs,
):
    if not (len(query_shape) == 4 and len(key_shape) == 4 and len(value_shape) == 4):
        raise ValueError(
            "The enhanced SDPA formula requires [batch, heads, sequence, dim] tensors"
        )
    batch, query_heads, query_length, query_dim = query_shape
    _, kv_heads, key_length, key_dim = key_shape
    _, value_heads, value_length, value_dim = value_shape
    if key_dim != query_dim or value_length != key_length or value_heads != kv_heads:
        raise ValueError("Incompatible query/key/value shapes for SDPA FLOP counting")
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by key/value heads")
    score_elements = _attention_score_elements(
        batch,
        query_heads,
        query_length,
        key_length,
        bool(is_causal),
    )
    qk = 2 * score_elements * query_dim
    av = 2 * score_elements * value_dim
    bias = score_elements if has_attention_bias else 0
    # Scale plus a five-operation softmax convention. Mask creation is not a FLOP.
    return qk + av + bias + 6 * score_elements


def _flash_sdpa_formula(
    query_shape,
    key_shape,
    value_shape,
    dropout_p=0.0,
    is_causal=False,
    *args,
    out_shape=None,
    attn_mask=None,
    **kwargs,
):
    return _sdpa_formula(
        query_shape,
        key_shape,
        value_shape,
        out_shape=out_shape,
        is_causal=is_causal,
        has_attention_bias=attn_mask is not None,
    )


def _biased_sdpa_formula(
    query_shape,
    key_shape,
    value_shape,
    attention_bias=None,
    compute_log_sumexp=False,
    dropout_p=0.0,
    is_causal=False,
    *args,
    out_shape=None,
    **kwargs,
):
    return _sdpa_formula(
        query_shape,
        key_shape,
        value_shape,
        out_shape=out_shape,
        is_causal=is_causal,
        has_attention_bias=attention_bias is not None,
    )


def _packet(name: str):
    return getattr(torch.ops.aten, name, None)


def enhanced_flop_mapping() -> Dict[Any, Any]:
    mapping: Dict[Any, Any] = {}

    def register(names: Iterable[str], formula) -> None:
        for name in names:
            packet = _packet(name)
            if packet is not None:
                mapping[packet] = formula

    register(("addmm",), _addmm_formula)
    register(("baddbmm",), _baddbmm_formula)
    register(("_scaled_mm",), _scaled_mm_formula)
    register(("convolution", "_convolution"), _convolution_formula)

    register(
        (
            "_scaled_dot_product_flash_attention",
            "_scaled_dot_product_flash_attention_for_cpu",
        ),
        _flash_sdpa_formula,
    )
    register(
        (
            "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_cudnn_attention",
        ),
        _biased_sdpa_formula,
    )

    register(
        (
            "add",
            "sub",
            "rsub",
            "mul",
            "div",
            "true_divide",
            "floor_divide",
            "remainder",
            "fmod",
            "maximum",
            "minimum",
            "clamp",
            "clamp_min",
            "clamp_max",
            "relu",
            "leaky_relu",
            "threshold",
            "abs",
            "neg",
            "sign",
            "reciprocal",
            "square",
            "round",
            "where",
        ),
        _pointwise_formula(1),
    )
    register(("pow", "lerp"), _pointwise_formula(2))
    register(
        (
            "exp",
            "exp2",
            "expm1",
            "log",
            "log2",
            "log10",
            "log1p",
            "sqrt",
            "rsqrt",
            "sin",
            "cos",
            "tan",
            "erf",
        ),
        _pointwise_formula(1),
    )
    register(("sigmoid", "tanh", "silu"), _pointwise_formula(4))
    register(("gelu",), _pointwise_formula(8))

    register(("sum", "prod", "amax", "amin", "max", "min"), _reduction_formula())
    register(("cumsum", "cumprod"), _reduction_formula())
    register(("index_add", "scatter_add"), _indexed_reduction_formula)
    register(("mean",), _reduction_formula(output_cost=1))
    register(("var", "std"), _reduction_formula(cost=3, output_cost=1))
    register(("var_mean", "std_mean"), _reduction_formula(cost=3, output_cost=2))

    register(("_softmax", "softmax"), _softmax_formula)
    register(("_log_softmax", "log_softmax"), _log_softmax_formula)
    register(
        (
            "native_layer_norm",
            "native_group_norm",
            "native_batch_norm",
            "_native_batch_norm_legit",
            "rms_norm",
        ),
        _normalization_formula,
    )

    register(("upsample_nearest2d", "_upsample_nearest_exact2d"), _pointwise_formula(0))
    register(("upsample_bilinear2d",), _upsample_formula(4))
    register(("upsample_bicubic2d",), _upsample_formula(16))
    register(("grid_sampler_2d",), _upsample_formula(4))

    return mapping


ZERO_FLOP_OPERATORS = {
    "alias",
    "arange",
    "as_strided",
    "bitwise",
    "cat",
    "chunk",
    "clone",
    "contiguous",
    "copy",
    "detach",
    "empty",
    "embedding",
    "eq",
    "expand",
    "fill",
    "full",
    "gather",
    "ge",
    "gt",
    "index",
    "index_select",
    "isfinite",
    "isinf",
    "isnan",
    "le",
    "lift",
    "logical",
    "lt",
    "masked_select",
    "ne",
    "new_empty",
    "nonzero",
    "ones",
    "permute",
    "pixel_shuffle",
    "pixel_unshuffle",
    "repeat",
    "repeat_interleave",
    "roll",
    "rand",
    "reshape",
    "scalar_tensor",
    "scatter",
    "select",
    "slice",
    "split",
    "squeeze",
    "stack",
    "sort",
    "t",
    "_to_copy",
    "transpose",
    "topk",
    "tril",
    "triu",
    "unbind",
    "unfold",
    "unsqueeze",
    "view",
    "argmax",
    "argmin",
    "argsort",
    "zeros",
}

ZERO_FLOP_PREFIXES = (
    "alias",
    "as_strided",
    "bitwise_",
    "copy",
    "empty",
    "logical_",
    "new_empty",
    "rand",
    "slice",
    "split",
    "view",
    "zeros",
)


def _is_structural_or_nonfloating_operator(name: str) -> bool:
    lowered = name.lower()
    operator = lowered.split(".", 1)[-1].split("::")[-1]
    return operator in ZERO_FLOP_OPERATORS or operator.startswith(ZERO_FLOP_PREFIXES)


@dataclass
class OperatorObservation:
    calls: int = 0
    output_elements: int = 0


def _operator_name(operator: Any) -> str:
    return str(operator).replace("<built-in method ", "").replace(">", "")


class _UnavailableFlopCounter:
    pass


try:
    from torch.utils.flop_counter import FlopCounterMode
except (ImportError, AttributeError):
    FlopCounterMode = _UnavailableFlopCounter  # type: ignore[assignment,misc]


if FlopCounterMode is not _UnavailableFlopCounter:

    class AuditedFlopCounterMode(FlopCounterMode):
        def __init__(self, custom_mapping: Dict[Any, Any]):
            super().__init__(display=False, custom_mapping=custom_mapping)
            self.observations: Dict[str, OperatorObservation] = defaultdict(
                OperatorObservation
            )
            self.unsupported: Dict[str, OperatorObservation] = defaultdict(
                OperatorObservation
            )

        def _count_flops(self, func_packet, out, args, kwargs):
            name = _operator_name(func_packet)
            output_elements = _first_tensor_numel(out)
            observation = self.observations[name]
            observation.calls += 1
            observation.output_elements += output_elements
            if func_packet not in self.flop_registry:
                unsupported = self.unsupported[name]
                unsupported.calls += 1
                unsupported.output_elements += output_elements
            return super()._count_flops(func_packet, out, args, kwargs)

else:
    AuditedFlopCounterMode = None  # type: ignore[assignment,misc]


def _is_bnb_4bit_linear(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    module_name = module.__class__.__module__.lower()
    return (
        "bitsandbytes" in module_name
        and "linear" in class_name
        and ("4bit" in class_name or "fp4" in class_name or "nf4" in class_name)
    )


def _external_attention_kind(module: torch.nn.Module) -> str | None:
    class_name = module.__class__.__name__.lower()
    module_name = module.__class__.__module__.lower()
    processor = getattr(module, "processor", None)
    processor_name = (
        processor.__class__.__name__.lower() if processor is not None else ""
    )
    config = getattr(module, "config", None)
    attention_implementation = str(
        getattr(module, "_attn_implementation", "")
        or getattr(config, "_attn_implementation", "")
    ).lower()
    if (
        "flash2varlen" in processor_name
        or "flashattention" in processor_name
        or "flash_attn" in processor_name
        or "flash_attn" in module_name
        or attention_implementation == "flash_attention_2"
    ):
        return "flash_attention"
    if "memoryefficient" in class_name or "xformers" in processor_name:
        return "xformers_attention"
    return None


def _external_norm_kind(module: torch.nn.Module) -> str | None:
    class_name = module.__class__.__name__.lower()
    module_name = module.__class__.__module__.lower()
    if "triton" in module_name and "norm" in class_name:
        return "triton_norm"
    return None


def _external_activation_kind(module: torch.nn.Module) -> str | None:
    swiglu = getattr(module, "swiglu", None)
    swiglu_module = getattr(swiglu, "__module__", "").lower()
    if "flash_attn" in swiglu_module and hasattr(module, "linear_1"):
        return "flash_swiglu"
    return None


def _attention_flops(module, inputs, kwargs, output) -> int:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return 0
    hidden = inputs[0]
    heads = int(
        getattr(module, "heads", 0)
        or getattr(module, "num_heads", 0)
        or getattr(module, "num_attention_heads", 0)
    )
    if not heads:
        return 0
    head_dim = int(
        getattr(module, "dim_head", 0)
        or getattr(module, "head_dim", 0)
        or getattr(module, "embed_dim", 0) // heads
    )
    if hidden.ndim == 3:
        if isinstance(module, torch.nn.MultiheadAttention) and not module.batch_first:
            query_length, batch, width = hidden.shape
        else:
            batch, query_length, width = hidden.shape
    elif hidden.ndim == 4:
        batch = hidden.shape[0]
        expected_width = heads * head_dim
        if expected_width and hidden.shape[-1] == expected_width:
            height, image_width, width = hidden.shape[1:]
        else:
            width, height, image_width = hidden.shape[1:]
        query_length = height * image_width
    else:
        return 0

    context = kwargs.get("encoder_hidden_states")
    if context is None:
        context = kwargs.get("context")
    if context is None and len(inputs) > 1 and isinstance(inputs[1], torch.Tensor):
        context = inputs[1]
    key_length = query_length
    if isinstance(context, torch.Tensor):
        key_length = context.shape[1] if context.ndim == 3 else query_length

    if not head_dim:
        head_dim = width // heads
    attention_mask = kwargs.get("attention_mask")
    if (
        attention_mask is None
        and len(inputs) > 2
        and isinstance(inputs[2], torch.Tensor)
    ):
        attention_mask = inputs[2]
    is_causal = bool(
        kwargs.get("is_causal", False) or getattr(module, "is_causal", False)
    )
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        valid_key_lengths = [
            sum(bool(value) for value in row)
            for row in attention_mask.detach().cpu().tolist()
        ]
        if query_length == key_length and attention_mask.shape[1] == query_length:
            if is_causal:
                score_elements = heads * sum(
                    length * (length + 1) // 2 for length in valid_key_lengths
                )
            else:
                score_elements = heads * sum(length**2 for length in valid_key_lengths)
        else:
            score_elements = heads * query_length * sum(valid_key_lengths)
    else:
        score_elements = _attention_score_elements(
            batch, heads, query_length, key_length, is_causal
        )
    return 4 * score_elements * head_dim + 6 * score_elements


def _is_generic_attention_module(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    has_heads = any(
        int(getattr(module, name, 0) or 0) > 0
        for name in ("heads", "num_heads", "num_attention_heads")
    )
    return has_heads and (class_name == "attention" or class_name.endswith("attention"))


@contextmanager
def module_formula_flop_hooks(
    modules: Iterable[torch.nn.Module],
) -> Iterator[Dict[str, int]]:
    counts: Dict[str, int] = defaultdict(int)
    handles = []
    seen: set[int] = set()

    normalization_types = tuple(
        layer_type
        for layer_type in (
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            getattr(torch.nn, "RMSNorm", None),
        )
        if layer_type is not None
    )
    activation_costs = {
        torch.nn.ReLU: 1,
        torch.nn.LeakyReLU: 1,
        torch.nn.Sigmoid: 4,
        torch.nn.Tanh: 4,
        torch.nn.SiLU: 4,
        torch.nn.GELU: 8,
    }

    def hook(module, inputs, output):
        output_elements = _first_tensor_numel(output)
        if not output_elements:
            return
        name = module.__class__.__name__
        if isinstance(module, torch.nn.Linear):
            value = 2 * output_elements * int(module.in_features)
            if module.bias is not None:
                value += output_elements
        elif isinstance(
            module,
            (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Conv3d,
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            kernel = math.prod(module.kernel_size)
            if isinstance(
                module,
                (
                    torch.nn.ConvTranspose1d,
                    torch.nn.ConvTranspose2d,
                    torch.nn.ConvTranspose3d,
                ),
            ):
                input_elements = _first_tensor_numel(inputs)
                value = (
                    2
                    * input_elements
                    * (int(module.out_channels) // int(module.groups))
                    * kernel
                )
            else:
                value = (
                    2
                    * output_elements
                    * (int(module.in_channels) // int(module.groups))
                    * kernel
                )
            if module.bias is not None:
                value += output_elements
        elif isinstance(module, normalization_types):
            value = 7 * output_elements
        elif type(module) in activation_costs:
            value = activation_costs[type(module)] * output_elements
        elif isinstance(module, torch.nn.Softmax):
            value = 5 * output_elements
        elif _is_generic_attention_module(module):
            value = _attention_flops(module, inputs, {}, output)
        else:
            return
        counts[name] += int(value)

    for root in modules:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if (
                _is_bnb_4bit_linear(module)
                or _external_attention_kind(module)
                or _external_norm_kind(module)
                or _external_activation_kind(module)
            ):
                continue
            if (
                isinstance(
                    module,
                    (
                        torch.nn.Linear,
                        torch.nn.Conv1d,
                        torch.nn.Conv2d,
                        torch.nn.Conv3d,
                        torch.nn.ConvTranspose1d,
                        torch.nn.ConvTranspose2d,
                        torch.nn.ConvTranspose3d,
                        *normalization_types,
                        torch.nn.Softmax,
                    ),
                )
                or type(module) in activation_costs
                or _is_generic_attention_module(module)
            ):
                handles.append(module.register_forward_hook(hook))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def opaque_module_flop_hooks(
    modules: Iterable[torch.nn.Module],
) -> Iterator[Dict[str, int]]:
    counts: Dict[str, int] = defaultdict(int)
    handles = []
    seen: set[int] = set()

    def linear_hook(module, inputs, output):
        output_elements = _first_tensor_numel(output)
        in_features = int(getattr(module, "in_features", 0))
        if output_elements and in_features:
            flops = 2 * output_elements * in_features
            if getattr(module, "bias", None) is not None:
                flops += output_elements
            counts[f"bnb4bit::{module.__class__.__name__}"] += flops

    def attention_hook(module, inputs, kwargs, output):
        kind = _external_attention_kind(module)
        flops = _attention_flops(module, inputs, kwargs, output)
        if kind and flops:
            counts[f"{kind}::{module.__class__.__name__}"] += flops

    def norm_hook(module, inputs, output):
        input_elements = _first_tensor_numel(inputs)
        if input_elements:
            counts[f"triton_norm::{module.__class__.__name__}"] += 7 * input_elements

    def activation_hook(module, inputs, output):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        hidden = inputs[0]
        input_width = hidden.shape[-1]
        inner_width = int(getattr(module.linear_1, "out_features", 0))
        if input_width and inner_width:
            activation_elements = hidden.numel() // input_width * inner_width
            counts[f"flash_swiglu::{module.__class__.__name__}"] += (
                5 * activation_elements
            )

    for root in modules:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if _is_bnb_4bit_linear(module):
                handles.append(module.register_forward_hook(linear_hook))
            elif _external_attention_kind(module):
                try:
                    handles.append(
                        module.register_forward_hook(attention_hook, with_kwargs=True)
                    )
                except TypeError:
                    handles.append(
                        module.register_forward_hook(
                            lambda mod, args, out: attention_hook(mod, args, {}, out)
                        )
                    )
            elif _external_norm_kind(module):
                handles.append(module.register_forward_hook(norm_hook))
            elif _external_activation_kind(module):
                handles.append(module.register_forward_hook(activation_hook))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def _serialize_observations(
    observations: Mapping[str, OperatorObservation], top_k: int | None = None
) -> list[Dict[str, Any]]:
    rows = [
        {
            "operator": name,
            "calls": observation.calls,
            "output_elements": observation.output_elements,
        }
        for name, observation in observations.items()
    ]
    rows.sort(key=lambda row: (row["output_elements"], row["calls"]), reverse=True)
    return rows if top_k is None else rows[:top_k]


def module_hook_flop_count(
    fn,
    modules: Iterable[torch.nn.Module],
    top_k: int = 20,
    reason: str = "torch.utils.flop_counter.FlopCounterMode is unavailable",
) -> Dict[str, Any]:
    modules = list(modules)
    with module_formula_flop_hooks(modules) as module_counts, opaque_module_flop_hooks(
        modules
    ) as opaque_counts:
        result = fn()
        _close_inference_output(result)
    estimated = int(sum(module_counts.values()) + sum(opaque_counts.values()))
    combined_counts = {**module_counts, **opaque_counts}
    opaque_total = int(sum(opaque_counts.values()))
    quantized_total = int(
        sum(
            value
            for name, value in opaque_counts.items()
            if name.startswith("bnb4bit::")
        )
    )
    return {
        "status": "partial",
        "error": reason,
        "formula_flops": estimated,
        "opaque_module_flops": opaque_total,
        "opaque_quantized_linear_flops": quantized_total,
        "estimated_total_flops": estimated,
        "confidence": "partial_module_hook_fallback",
        "top_operators": [
            {"operator": name, "flops": value}
            for name, value in sorted(
                combined_counts.items(), key=lambda item: item[1], reverse=True
            )[:top_k]
        ],
        "opaque_module_breakdown": dict(opaque_counts),
        "included_opaque_module_breakdown": dict(opaque_counts),
        "supplemented_operators": sorted(opaque_counts),
        "operators_observed": None,
        "unsupported_operators": [],
        "unsupported_nontrivial_operators": [
            {
                "operator": "functional_and_custom_ops_not_visible_to_module_hooks",
                "calls": None,
                "output_elements": None,
            }
        ],
        "definition": (
            "Dynamic inference counted with module hooks because dispatch-level "
            "counting was unavailable or incompatible. Linear, convolution, normalization, "
            "activation, softmax, recognizable attention, and known opaque fused modules are "
            "covered; functional/custom ops remain a reported limitation."
        ),
        "scope": (
            "PyTorch model/tensor arithmetic during the complete adapter call. File I/O, "
            "PIL/NumPy image decoding or resizing, Python control flow, and data movement "
            "are outside the conventional model-FLOPs count."
        ),
        "convention": "One multiply-add is two FLOPs",
    }


def dispatch_flop_count(
    fn,
    modules: Iterable[torch.nn.Module],
    top_k: int = 20,
) -> Dict[str, Any]:
    modules = list(modules)
    if AuditedFlopCounterMode is None:
        return module_hook_flop_count(fn, modules, top_k=top_k)

    counter = AuditedFlopCounterMode(enhanced_flop_mapping())
    with opaque_module_flop_hooks(modules) as opaque_counts:
        with counter:
            result = fn()
            _close_inference_output(result)

    formula_flops = int(counter.get_total_flops())
    unsupported_names = {name.lower() for name in counter.unsupported}
    counted_operator_names = {
        _operator_name(operator).lower()
        for operator, flops in counter.get_flop_counts().get("Global", {}).items()
        if flops
    }
    has_bnb_custom_op = any("bitsandbytes" in name for name in unsupported_names)
    has_external_attention_op = any(
        token in name
        for name in unsupported_names
        for token in (
            "flash_attn",
            "flashattention",
            "flash_attention",
            "xformers",
            "memory_efficient",
        )
    )
    counted_external_attention = any(
        token in name
        for name in counted_operator_names
        for token in (
            "flash_attn::",
            "xformers::",
            "memory_efficient_attention",
        )
    )
    counted_external_norm = any(
        token in name
        for name in counted_operator_names
        for token in ("triton::rms_norm", "triton::layer_norm")
    )
    counted_external_swiglu = any(
        "flash_attn::swiglu" in name for name in counted_operator_names
    )
    counted_standard_attention = any(
        "scaled_dot_product" in name for name in counted_operator_names
    )
    counted_standard_norm = any(
        token in name
        for name in counted_operator_names
        for token in (
            "native_layer_norm",
            "native_group_norm",
            "native_batch_norm",
            "rms_norm",
        )
    )
    counted_standard_swiglu = (
        any("silu" in name for name in counted_operator_names)
        and any("mul" in name for name in counted_operator_names)
    )
    included_opaque_counts = {
        name: value
        for name, value in opaque_counts.items()
        if (name.startswith("bnb4bit::") and has_bnb_custom_op)
        or (
            name.startswith(("flash_attention::", "xformers_attention::"))
            and not counted_external_attention
            and not counted_standard_attention
        )
        or (
            name.startswith("triton_norm::")
            and not counted_external_norm
            and not counted_standard_norm
        )
        or (
            name.startswith("flash_swiglu::")
            and not counted_external_swiglu
            and not counted_standard_swiglu
        )
    }
    omitted_opaque_counts = {
        name: value
        for name, value in opaque_counts.items()
        if name not in included_opaque_counts
    }
    unverified_opaque = {
        name: value
        for name, value in omitted_opaque_counts.items()
        if name.startswith("bnb4bit::") and not has_bnb_custom_op
    }
    opaque_flops = int(sum(included_opaque_counts.values()))
    quantized_linear_flops = int(
        sum(
            value
            for name, value in included_opaque_counts.items()
            if name.startswith("bnb4bit::")
        )
    )
    supplemented_operator_names = {
        name
        for name in counter.unsupported
        if (has_bnb_custom_op and "bitsandbytes" in name.lower())
        or (
            has_external_attention_op
            and any(
                token in name.lower()
                for token in (
                    "flash_attn",
                    "flashattention",
                    "flash_attention",
                    "xformers",
                    "memory_efficient",
                )
            )
        )
        or (
            any(key.startswith("triton_norm::") for key in included_opaque_counts)
            and any(token in name.lower() for token in ("rms_norm", "layer_norm"))
        )
        or (
            any(key.startswith("flash_swiglu::") for key in included_opaque_counts)
            and "swiglu" in name.lower()
        )
    }
    unsupported_nontrivial = {
        name: observation
        for name, observation in counter.unsupported.items()
        if not _is_structural_or_nonfloating_operator(name)
        and name not in supplemented_operator_names
    }
    unsupported_nontrivial.update(
        {
            f"unverified_opaque_hook::{name}": OperatorObservation(
                calls=1, output_elements=0
            )
            for name in unverified_opaque
        }
    )
    flop_rows = []
    for operator, flops in counter.get_flop_counts().get("Global", {}).items():
        flop_rows.append({"operator": _operator_name(operator), "flops": int(flops)})
    flop_rows.sort(key=lambda row: row["flops"], reverse=True)

    if unsupported_nontrivial:
        confidence = "partial_unmodeled_compute_ops"
    elif opaque_flops:
        confidence = "registered_formulas_plus_opaque_module_hooks"
    else:
        confidence = "registered_formulas_complete_for_observed_ops"

    return {
        "status": "ok",
        "formula_flops": formula_flops,
        "opaque_module_flops": opaque_flops,
        "opaque_quantized_linear_flops": quantized_linear_flops,
        "estimated_total_flops": formula_flops + opaque_flops,
        "confidence": confidence,
        "top_operators": flop_rows[:top_k],
        "opaque_module_breakdown": dict(opaque_counts),
        "included_opaque_module_breakdown": included_opaque_counts,
        "omitted_opaque_module_breakdown": omitted_opaque_counts,
        "unverified_opaque_module_breakdown": unverified_opaque,
        "opaque_overlap_policy": (
            "Opaque hook estimates are omitted when an equivalent SDPA, normalization, "
            "or SwiGLU dispatch formula was observed. A 4-bit linear hook is included "
            "only when a bitsandbytes custom operator was observed; otherwise the row "
            "is marked partial rather than risking double counting."
        ),
        "supplemented_operators": sorted(
            supplemented_operator_names | set(included_opaque_counts)
        ),
        "operators_observed": len(counter.observations),
        "unsupported_operators": _serialize_observations(counter.unsupported),
        "unsupported_nontrivial_operators": _serialize_observations(
            unsupported_nontrivial
        ),
        "definition": (
            "Formula-based runtime FLOPs over one complete inference call using PyTorch "
            "dispatch observations. "
            "The count includes dynamic autoregressive and denoising iterations, enhanced "
            "attention/convolution/normalization/activation/reduction/interpolation formulas, "
            "and explicit hooks for bitsandbytes 4-bit linear, external fused-attention, Triton "
            "normalization, and FlashAttention SwiGLU modules. An opaque hook is omitted when an "
            "equivalent fused or standard operator was already counted. Unsupported nontrivial operators are "
            "reported instead of silently treated as zero. Status=ok means formula-complete "
            "for observed operators under the stated convention, not a hardware instruction count."
        ),
        "scope": (
            "PyTorch model/tensor arithmetic during the complete adapter call. File I/O, "
            "PIL/NumPy image decoding or resizing, Python control flow, and data movement "
            "are outside the conventional model-FLOPs count."
        ),
        "convention": (
            "One multiply-add is two FLOPs. Scalar elementary functions count as one operation; "
            "softmax uses five and normalization seven operations per output element."
        ),
    }
