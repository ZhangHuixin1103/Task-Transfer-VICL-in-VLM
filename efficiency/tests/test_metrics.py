from __future__ import annotations

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from efficiency.adapters.toy import ToyAdapter
from efficiency.metrics import (
    benchmark_callable,
    benchmark_dataset_callable,
    count_parameters,
    parameter_report,
    profile_flops,
)
from efficiency.flops import (
    _attention_score_elements,
    _is_structural_or_nonfloating_operator,
    dispatch_flop_count,
    module_hook_flop_count,
)
from efficiency.report import flatten_result, write_reports
from efficiency.suite import _aggregate_flop_profiles, _select_flop_indices


class SharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Linear(4, 4)
        self.first = layer
        self.second = layer


class PackedQuantizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.nn.Parameter(
            torch.zeros(16, dtype=torch.uint8), requires_grad=False
        )
        weight.quant_state = SimpleNamespace(shape=(4, 8))
        self.register_parameter("weight", weight)


class Linear4bit(torch.nn.Module):
    __module__ = "bitsandbytes.nn.modules"

    def __init__(self):
        super().__init__()
        self.in_features = 8
        self.out_features = 4
        self.weight = torch.nn.Parameter(
            torch.zeros(16, dtype=torch.uint8), requires_grad=False
        )


class MetaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(3, 5, device="meta"))


class AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = torch.nn.LayerNorm(16)

    def forward(self, inputs):
        query = inputs.reshape(1, 2, 8, 16)
        attended = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        return self.normalization(attended)


class FakeFlash2Varlen:
    pass


class OpaqueFlashAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.dim_head = 4
        self.processor = FakeFlash2Varlen()

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        return hidden_states.clone()


class FlashMarkedSDPAAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.dim_head = 4
        self.processor = FakeFlash2Varlen()

    def forward(self, hidden_states):
        batch, sequence, width = hidden_states.shape
        query = hidden_states.reshape(batch, sequence, self.heads, self.dim_head)
        query = query.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        return output.transpose(1, 2).reshape(batch, sequence, width)


class MetricsTest(unittest.TestCase):
    def test_flop_profile_sampling_and_coverage_aggregation(self):
        indices = _select_flop_indices(list(range(10)), 5)
        self.assertEqual(indices, [0, 2, 4, 7, 9])
        profiles = [
            {
                "status": "ok",
                "confidence": "registered_formulas",
                "estimated_total_flops": value,
            }
            for value in (10, 20, 30, 40, 50)
        ]
        aggregate = _aggregate_flop_profiles(indices, profiles)
        self.assertEqual(aggregate["status"], "ok")
        self.assertEqual(aggregate["estimated_total_flops"], 30)
        self.assertEqual(aggregate["estimated_total_flops_min"], 10)
        self.assertEqual(aggregate["estimated_total_flops_max"], 50)
        profiles[-1]["status"] = "partial"
        self.assertEqual(
            _aggregate_flop_profiles(indices, profiles)["status"], "partial"
        )

    def test_flop_profilers_close_generated_outputs(self):
        model = torch.nn.Linear(4, 4)
        closed = []

        class Output:
            def close(self):
                closed.append(True)

        def inference():
            model(torch.ones(1, 4))
            return type("Result", (), {"output": Output()})()

        dispatch_flop_count(inference, [model])
        module_hook_flop_count(inference, [model])
        self.assertEqual(len(closed), 2)

    def test_parameter_count_deduplicates_shared_parameters(self):
        model = SharedModel()
        stats = count_parameters({"a": model, "b": model})
        self.assertEqual(stats.total, 20)
        report = parameter_report({"first": model, "second": model})
        self.assertEqual(report["unique_total"]["total"], 20)
        self.assertEqual(report["components"]["first"]["total"], 20)

    def test_parameter_count_restores_packed_quantized_logical_shape(self):
        stats = count_parameters(PackedQuantizedModel())
        self.assertEqual(stats.total, 32)
        self.assertEqual(stats.stored_elements, 16)
        self.assertEqual(stats.storage_bytes, 16)
        self.assertEqual(stats.quantized_tensors, 1)
        self.assertEqual(stats.logical_shape_sources, {"quant_state.shape": 1})

    def test_parameter_count_uses_owning_4bit_linear_shape_as_fallback(self):
        stats = count_parameters(Linear4bit())
        self.assertEqual(stats.total, 32)
        self.assertEqual(stats.stored_elements, 16)
        self.assertEqual(stats.logical_shape_sources, {"bitsandbytes.linear_shape": 1})

    def test_meta_parameters_keep_logical_count_and_mark_storage_incomplete(self):
        stats = count_parameters(MetaModel())
        self.assertEqual(stats.total, 15)
        self.assertEqual(stats.storage_bytes, 0)
        self.assertEqual(stats.meta_tensors, 1)
        report = parameter_report({"model": MetaModel()})
        self.assertFalse(report["storage_complete"])

    def test_toy_latency_and_flops(self):
        adapter = ToyAdapter(device="cpu")
        adapter.setup()
        latency = benchmark_callable(lambda: adapter.run("toy"), warmup=1, repeats=2)
        self.assertEqual(latency["end_to_end"]["count"], 2)
        self.assertGreater(latency["end_to_end"]["mean_s"], 0)
        flops = profile_flops(lambda: adapter.run("toy"), top_k=5)
        self.assertEqual(flops["estimated_total_flops"], 961536)
        self.assertEqual(flops["unsupported_nontrivial_operators"], [])

    def test_latency_does_not_retain_previous_inference_outputs(self):
        class Output:
            pass

        live_outputs = weakref.WeakSet()

        def inference():
            gc.collect()
            self.assertEqual(len(live_outputs), 0)
            output = Output()
            live_outputs.add(output)
            return type(
                "Result",
                (),
                {"output": output, "stage_seconds": {}, "metadata": {}},
            )()

        benchmark_callable(inference, warmup=1, repeats=3)

    def test_dataset_latency_uses_distinct_indices_and_closes_outputs(self):
        calls = []
        closed = []

        class Output:
            def __init__(self, index):
                self.index = index

            def close(self):
                closed.append(self.index)

        def inference(index):
            calls.append(index)
            return type(
                "Result",
                (),
                {
                    "output": Output(index),
                    "stage_seconds": {},
                    "metadata": {"index": index},
                },
            )()

        result = benchmark_dataset_callable(
            inference,
            sample_indices=[2, 4, 8],
            warmup_indices=[2],
            seed=7,
        )
        self.assertEqual(calls, [2, 2, 4, 8])
        self.assertEqual(closed, [2, 2, 4, 8])
        self.assertEqual(result["end_to_end"]["count"], 3)
        self.assertEqual(result["measured_indices"], [2, 4, 8])

    def test_dispatch_flops_include_attention_and_normalization(self):
        model = AttentionModel()
        inputs = torch.randn(1, 16, 16)
        flops = profile_flops(
            lambda: type("Result", (), {"output": model(inputs)})(),
            components={"model": model},
            top_k=5,
        )
        self.assertEqual(flops["estimated_total_flops"], 10752)
        self.assertEqual(flops["unsupported_nontrivial_operators"], [])

    def test_extended_runtime_formulas_cover_released_model_operators(self):
        pool_input = torch.ones(1, 1, 4, 4)
        polar_abs = torch.ones(4)
        polar_angle = torch.zeros(4)

        def inference():
            pooled = torch.nn.functional.avg_pool2d(pool_input, 2, stride=2)
            activated = pooled.relu_()
            probabilities = torch.ops.aten._safe_softmax.default(activated, -1)
            complex_values = torch.polar(polar_abs, polar_angle)
            norm = torch.linalg.vector_norm(polar_abs)
            return type(
                "Result",
                (),
                {"output": (probabilities, complex_values, norm)},
            )()

        flops = profile_flops(inference, top_k=10)
        # Pool 16, ReLU 4, softmax 20, polar 16, and vector norm 13.
        self.assertEqual(flops["estimated_total_flops"], 69)
        self.assertEqual(flops["status"], "ok")
        self.assertEqual(flops["unsupported_nontrivial_operators"], [])

    def test_released_model_structural_and_sampling_ops_are_zero_flop(self):
        for operator in (
            "aten._unsafe_view",
            "aten._unsafe_index",
            "aten.constant_pad_nd",
            "aten.index_put_",
            "aten.masked_fill_",
            "aten.multinomial",
            "aten._local_scalar_dense",
        ):
            self.assertTrue(_is_structural_or_nonfloating_operator(operator))

    def test_causal_attention_counts_only_visible_scores(self):
        model = AttentionModel()
        inputs = torch.randn(1, 16, 16)

        def inference():
            query = inputs.reshape(1, 2, 8, 16)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query, query, query, is_causal=True
            )
            return type("Result", (), {"output": model.normalization(attended)})()

        flops = profile_flops(
            inference,
            components={"model": model},
            top_k=5,
        )
        self.assertEqual(flops["estimated_total_flops"], 6832)
        self.assertEqual(flops["unsupported_nontrivial_operators"], [])
        self.assertEqual(_attention_score_elements(1, 2, 3, 5, True), 12)

    def test_linear_and_transposed_convolution_include_bias(self):
        linear = torch.nn.Linear(4, 3, bias=True)
        transpose = torch.nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, bias=True)
        linear_input = torch.randn(2, 4)
        transpose_input = torch.randn(1, 2, 4, 4)
        flops = profile_flops(
            lambda: type(
                "Result",
                (),
                {"output": (linear(linear_input), transpose(transpose_input))},
            )(),
            components={"linear": linear, "transpose": transpose},
            top_k=5,
        )
        # Linear: 2*2*3*4 + 6 bias. ConvTranspose: 2*32*3*9 + 147 bias.
        self.assertEqual(flops["estimated_total_flops"], 1929)
        self.assertEqual(flops["unsupported_nontrivial_operators"], [])

        fallback = module_hook_flop_count(
            lambda: type(
                "Result",
                (),
                {"output": (linear(linear_input), transpose(transpose_input))},
            )(),
            [linear, transpose],
        )
        self.assertEqual(fallback["estimated_total_flops"], 1929)

    def test_opaque_flash_attention_hook_counts_unpadded_sequences(self):
        model = OpaqueFlashAttention()
        inputs = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        flops = profile_flops(
            lambda: type(
                "Result",
                (),
                {"output": model(inputs, inputs, attention_mask=mask)},
            )(),
            components={"model": model},
            top_k=5,
        )
        # 2 heads * (4^2 + 2^2) scores, QK + AV + scale/softmax.
        self.assertEqual(flops["estimated_total_flops"], 880)
        self.assertEqual(
            flops["confidence"], "registered_formulas_plus_opaque_module_hooks"
        )

    def test_opaque_attention_hook_does_not_double_count_observed_sdpa(self):
        model = FlashMarkedSDPAAttention()
        inputs = torch.randn(1, 4, 8)
        flops = profile_flops(
            lambda: type("Result", (), {"output": model(inputs)})(),
            components={"model": model},
            top_k=5,
        )
        # 2 heads * 4x4 scores: QK + AV plus scale/softmax.
        self.assertEqual(flops["estimated_total_flops"], 704)
        self.assertEqual(flops["opaque_module_flops"], 0)
        self.assertIn(
            "flash_attention::FlashMarkedSDPAAttention",
            flops["omitted_opaque_module_breakdown"],
        )

    def test_unmodeled_compute_is_reported_as_partial(self):
        inputs = torch.randn(16)
        flops = profile_flops(
            lambda: type("Result", (), {"output": torch.fft.fft(inputs)})(),
            top_k=5,
        )
        self.assertEqual(flops["status"], "partial")
        self.assertEqual(flops["confidence"], "partial_unmodeled_compute_ops")
        self.assertIn(
            "aten._fft_r2c",
            [row["operator"] for row in flops["unsupported_nontrivial_operators"]],
        )

    def test_module_hook_fallback_counts_dynamic_model_execution(self):
        adapter = ToyAdapter(device="cpu")
        adapter.setup()
        with mock.patch("efficiency.flops.AuditedFlopCounterMode", None):
            flops = profile_flops(
                lambda: adapter.run("toy"),
                components={"toy_model": adapter.model},
                top_k=5,
            )
        self.assertEqual(flops["estimated_total_flops"], 961536)
        self.assertEqual(flops["status"], "partial")
        self.assertEqual(flops["confidence"], "partial_module_hook_fallback")

    def test_dispatch_failure_uses_partial_module_hook_fallback(self):
        adapter = ToyAdapter(device="cpu")
        adapter.setup()
        with mock.patch(
            "efficiency.metrics.dispatch_flop_count",
            side_effect=RuntimeError("incompatible dispatch mode"),
        ):
            flops = profile_flops(
                lambda: adapter.run("toy"),
                components={"toy_model": adapter.model},
                top_k=5,
            )
        self.assertEqual(flops["estimated_total_flops"], 961536)
        self.assertEqual(flops["status"], "partial")
        self.assertIn("incompatible dispatch mode", flops["error"])

    def test_report_generation(self):
        document = {
            "schema_version": 2,
            "adapter": "toy",
            "protocol": "tensor -> tensor",
            "system": {"platform": "test", "cuda_devices": []},
            "conditions": [
                {
                    "condition": "toy",
                    "parameters": {
                        "unique_total": {
                            "total": 100,
                            "trainable": 100,
                            "storage_bytes": 400,
                            "tensors": 2,
                        }
                    },
                    "trained_parameters": 100,
                    "latency": {
                        "end_to_end": {"mean_s": 1.0, "median_s": 1.0, "std_s": 0.0},
                        "stages": {},
                        "peak_cuda_memory_bytes": {},
                    },
                    "flops": {"observed_flops": 200},
                }
            ],
        }
        rows = flatten_result(document)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_reports(rows, output)
            self.assertTrue((output / "comparison.csv").exists())
            self.assertTrue((output / "comparison.md").exists())

    def test_report_flattens_multi_task_suite_and_aligns_fixed_ours(self):
        def condition(name, parameter_count, latency, flops):
            task = {
                "task": "denoising",
                "metadata": {"sampling": {"steps": 40}},
                "latency": {
                    "end_to_end": {
                        "count": 100,
                        "mean_s": latency,
                        "median_s": latency,
                        "std_s": 0.1,
                    },
                    "stages": {},
                },
                "flops": {
                    "status": "ok",
                    "confidence": "registered_formulas",
                    "estimated_total_flops": flops,
                    "estimated_total_flops_min": flops - 10,
                    "estimated_total_flops_max": flops + 10,
                    "profile_sample_count": 5 if name == "ours" else 1,
                },
            }
            return {
                "condition": name,
                "parameters": {
                    "storage_complete": True,
                    "unique_total": {
                        "total": parameter_count,
                        "stored_elements": parameter_count,
                        "storage_bytes": parameter_count * 2,
                    },
                },
                "trained_parameters": 10 if name == "ours" else 0,
                "tasks": [task],
                "aggregate": {
                    "queries": 100,
                    "macro_task_mean_latency_s": latency,
                    "macro_task_median_latency_s": latency,
                    "macro_task_flops": flops,
                    "flops_complete_for_all_tasks": True,
                },
            }

        rows = flatten_result(
            {
                "kind": "multi_task_efficiency_suite",
                "adapter": "t2t-qwen",
                "benchmark_family": "original-t2t-vicl-cross-task",
                "protocol": "test",
                "controlled_protocol": {"resolution": 448},
                "system": {"platform": "test", "cuda_devices": []},
                "conditions": [
                    condition("fixed", 100, 2.0, 1000),
                    condition("ours", 130, 3.0, 1400),
                ],
            }
        )
        task_rows = [row for row in rows if row["task"] == "denoising"]
        self.assertEqual(len(task_rows), 2)
        self.assertEqual(task_rows[1]["incremental_parameters_vs_fixed"], 30)
        self.assertEqual(task_rows[1]["incremental_flops_vs_fixed"], 400)
        self.assertEqual(task_rows[1]["latency_overhead_vs_fixed_s"], 1.0)
        self.assertEqual(task_rows[1]["resolution"], 448)
        self.assertEqual(task_rows[1]["measured_queries"], 100)
        self.assertEqual(
            task_rows[1]["benchmark_family"], "original-t2t-vicl-cross-task"
        )
        self.assertEqual(len([row for row in rows if row["task"] == "__macro__"]), 2)

    def test_fixed_to_ours_deltas(self):
        def condition(name, parameters, flops, latency):
            return {
                "condition": name,
                "parameters": {
                    "unique_total": {
                        "total": parameters,
                        "trainable": 0,
                        "storage_bytes": parameters * 2,
                        "tensors": 1,
                    }
                },
                "trained_parameters": 10 if name == "ours" else 0,
                "latency": {
                    "end_to_end": {
                        "mean_s": latency,
                        "median_s": latency,
                        "std_s": 0.0,
                    },
                    "stages": {},
                    "peak_cuda_memory_bytes": {},
                },
                "flops": {"observed_flops": flops, "status": "ok"},
            }

        rows = flatten_result(
            {
                "schema_version": 1,
                "adapter": "t2t-test",
                "protocol": "test",
                "system": {"platform": "test", "cuda_devices": []},
                "conditions": [
                    condition("fixed", 100, 1000, 2.0),
                    condition("ours", 130, 1400, 3.0),
                ],
            }
        )
        ours = rows[1]
        self.assertEqual(ours["incremental_parameters_vs_fixed"], 30)
        self.assertEqual(ours["incremental_flops_vs_fixed"], 400)
        self.assertEqual(ours["latency_overhead_vs_fixed_s"], 1.0)
        self.assertEqual(ours["latency_ratio_vs_fixed"], 1.5)


if __name__ == "__main__":
    unittest.main()
