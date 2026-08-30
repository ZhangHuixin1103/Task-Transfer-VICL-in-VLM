# Efficiency Benchmark

This package measures logical parameters, runtime formula FLOPs, and actual inference
latency for the original T2T-VICL pipeline and three locally vendored VICL/generalist
vision baselines. It reuses the inference functions in `eval_qwen.py`, `eval_flux.py`,
`eval_omnigen.py`, and `eval_firered.py` instead of maintaining a second set of model
calls.

## What Is Measured

- **Logical parameters**: unique parameters are deduplicated by object identity across
  all pipeline components. Packed bitsandbytes 4-bit weights use the original shape in
  `quant_state.shape` (or the owning `Linear4bit` input/output dimensions), rather than
  packed tensor `numel()`. This makes FLUX.2's default CPU-offloaded 4-bit execution
  comparable to dense models at the logical architecture level. `stored_elements` and
  physical payload bytes are reported separately; offload placement does not change the
  logical count. If Accelerate leaves any parameter on the `meta` device, the JSON/CSV
  marks physical storage incomplete instead of presenting it as a full payload size.
- **Method-trained parameters**: for the current T2T-VICL checkpoint, this follows
  `finetune.sh`: the Qwen vision tower is frozen, while `visual.merger`,
  `language_model`, and `lm_head` are trained. This is distinct from both total loaded
  parameters and the runtime `requires_grad` state.
- **Latency**: synchronized wall-clock time. Model loading and warm-up are excluded;
  cached JSON-record selection is excluded. Input files are read immediately before
  timing to remove physical-disk cold starts; image open/decode/resize, preprocessing,
  prompt generation, image generation, and postprocessing remain included. Stage timing
  uses deferred CUDA events and adds no synchronization barriers inside the pipeline.
  RNGs are reset before every repeat. Condition-only resources are loaded and released
  around that condition, so the prompt generator is not resident during `Fixed`. Unused
  allocator cache is emptied before warm-up and allocated/reserved memory are reported
  separately.
- **Runtime model FLOPs**: dispatch-level formulas are evaluated over one complete, real
  inference call, so the actual number of autoregressive and denoising iterations is
  included. This is PyTorch tensor arithmetic: file I/O, PIL/NumPy decoding/resizing,
  Python control flow, and data movement are not conventional model FLOPs. The registry
  covers matrix multiplication, convolution including bias,
  SDPA/FlashAttention, normalization, softmax, activation, reduction, indexed reduction,
  and interpolation. Runtime hooks add logical MAC-equivalent FLOPs for opaque
  bitsandbytes 4-bit linear layers, external fused attention, Triton normalization, and
  FlashAttention SwiGLU modules.
- **Coverage audit**: undecomposable operators without a formula are listed with call
  counts and output sizes. A row is marked `partial` whenever nontrivial compute remains
  unmodeled. `Status=ok` means formula-complete for the operators observed in that run
  under the stated convention; it is not a hardware instruction count.
- **Peak GPU memory**: maximum allocated PyTorch memory observed during measured
  repeats. It does not include memory allocated outside PyTorch.

No tool can infer the mathematical work of an arbitrary opaque CUDA kernel merely from
its runtime name. Therefore use `Runtime FLOPs` only for rows whose `confidence` is not
`partial`; otherwise resolve the listed custom operators or report the value with its
coverage status. The convention is one multiply-add = two FLOPs, one operation for a
scalar elementary function, five per softmax element, and seven per normalization
element. Quantized linear layers report dense logical MAC-equivalent FLOPs; kernel
dequantization overhead and data movement are not folded into that conventional model
FLOP count. Comparison, indexing, reshaping, and memory-copy operators are zero FLOPs
under this arithmetic convention, even though they can affect latency. In an old
environment without `torch.utils.flop_counter.FlopCounterMode`, the code uses dynamic
module hooks and marks the result `partial_module_hook_fallback`.

## Controlled Multi-Task Protocol

The paper-facing protocol is executed by `efficiency.suite`, not by repeatedly timing
one hand-picked image. It intentionally has two manifests because the original and
third-party code use different datasets:

- `efficiency/t2t_tasks.json` is exclusively for the four original T2T-VICL backends. It
  uses `data/dataset/eval_dataset.json` plus images in `data/tasks`, and selects all 26
  directional `Task A -> Task B` groups, exactly 100 original records per group.
- `efficiency/tasks.json` is for Painter, Prompt-Diffusion, and InstructDiffusion.
  Each restoration task supplies one same-task
  degraded/target demonstration and distinct query pairs from `data/others`.

The suite chooses these defaults from `--adapter` and rejects a manifest from the wrong
family. Paper commands still pass `--task-manifest` explicitly so the protocol remains
visible in logs and job scripts.

Both protocols standardize the processed visual workload for systems measurement, but
their per-task rows are not the same experiment. Do not compare a same-task accuracy row
against a cross-task T2T row or present the former as a replacement for the paper's
cross-task evaluation.

1. Load one model once, then evaluate every selected task in the same process.
2. Use batch size 1, concurrency 1, the model's quality-evaluation sampling settings,
   and one recorded precision/execution mode.
3. Directly bicubic-resize every demonstration input, demonstration output, query, and
   requested output to `448 x 448`. The T2T prompt generator receives the same resized
   visual inputs. Painter keeps its released `896 x 448` stitched canvas and emits a
   fixed `448 x 448` query output.
4. Run five unreported warm-up queries, then measure up to 100 deterministic, distinct
   records per task or task pair exactly once. Warm-up indices are disjoint from measured
   indices whenever the split has spare records. Every original T2T pair has exactly 100
   records, so its five warm-ups necessarily reuse records that are later measured; the
   JSON records this policy. Small official splits are not duplicated: LOL, for example,
   contributes its 15 distinct validation pairs.
5. Report raw samples plus mean, median, population standard deviation, p90, and p95 for
   each task. The `__macro__` row gives every task equal weight; pooled-query latency is
   retained in JSON as a secondary statistic.
6. Count parameters once per condition because model weights do not depend on task.
   `Fixed` and official baselines profile one deterministic middle query per task because
   their prompt, spatial shapes, and step count are fixed. `Ours` profiles five evenly
   spaced measured queries by default and reports mean/min/max because autoregressive
   relation-prompt length is input dependent. An aggregate is `ok` only when every
   per-query FLOPs audit is complete.

There is no field-wide rule that a latency table must contain 1000 images. Published
vision/generation systems commonly use 100-image averages, while standardized systems
benchmarks can require thousands. Here 100 distinct queries is a practical main-study
target for expensive image generators and is substantially stronger than repeating one
image ten times. If a test set has fewer than 100 pairs, report the exact count; do not
resample it to create pseudo-replication. A 30-query pilot can estimate server time, but
it should not replace the 100-query main run when at least 100 official queries exist.

The fixed-size table answers architectural/runtime efficiency under equal visual work.
A separate native-resolution supplement is useful for deployment realism, but it must
record the processed dimensions for every query and must not be averaged into the
controlled table. Original files do not need identical dimensions inside a task because
the adapter applies one explicit resize policy before model input. Tasks do need the same
processed dimensions in the primary cross-task/model comparison.

The full original-data audit currently records 24 native input/output size warnings in
style-transfer records (for example, 640x360 paired with 854x480). Their paths and
directions are valid, and all consumed tensors are still 448x448 under this protocol;
the validator therefore reports these warnings without failing the run.

Related measurement references include the 100-image average used by
[StreamDiffusion](https://openaccess.thecvf.com/content/CVPR2024/html/Kodaira_StreamDiffusion_A_Pipeline-Level_Solution_for_Real-Time_Interactive_Generation_CVPR_2024_paper.html),
the NTIRE 2022 efficient super-resolution challenge's average runtime over 100
validation and 100 test images with FLOPs evaluated at a fixed 256x256 input
([challenge report](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Li_NTIRE_2022_Challenge_on_Efficient_Super-Resolution_Methods_and_Results_CVPRW_2022_paper.pdf)),
PyTorch/Torch-TensorRT's recommendation to warm up CUDA work before synchronized timing
([performance tuning guide](https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html)),
and MLPerf's much larger minimum sample count for a standardized text-to-image systems
benchmark ([MLPerf Inference](https://docs.mlcommons.org/inference/benchmarks/text_to_image/)).
These are precedents, not interchangeable protocols; the JSON records this study's exact
sample count, warm-up count, hardware, software, and raw timings.

Painter itself trains and evaluates with a default 448x448 image size before composing
the 896x448 prompt/query canvas. This gives 448 a method-grounded role here in addition
to being a cross-model control. It does not imply that a 448 result can be compared
numerically with a paper's 256, 512, or 1024 result.

## PSNR/SSIM Quality Pass

`efficiency.suite` intentionally discards generated images after each timed query. It
cannot fill an accuracy table. Run quality separately so PNG saving, target loading,
PSNR, and SSIM never contaminate latency:

```bash
python -m efficiency.quality \
  --adapter prompt-diffusion --conditions official \
  --task-manifest efficiency/tasks_sparse.json \
  --model-id /weights/prompt-diffusion-diffusers \
  --steps 50 --resolution 448 --max-samples -1 --resume \
  --output-dir efficiency/quality/prompt_diffusion

python -m efficiency.quality \
  --adapter instruct-diffusion --conditions official \
  --task-manifest efficiency/tasks_sparse.json \
  --checkpoint /external/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt \
  --steps 50 --resolution 448 --max-samples -1 --resume \
  --output-dir efficiency/quality/instruct_diffusion
```

The quality default is every held-out record in the supplied manifest. With
`tasks_sparse.json`, this is the same deterministic subset used for latency. A
demonstration that also appears in the JSON is excluded and reported by source index.
Metrics use RGB uint8 pixels at 448x448,
per-image PSNR, and the default channel-averaged skimage SSIM convention. The runner
writes every sample path and score, saves predictions, and supports
configuration-checked resume. `quality_table` reads the two summary JSON files and
updates only their columns in a supplied TeX table.

## Supported Adapters

The original T2T adapters need no external source-tree setting. For Painter,
Prompt-Diffusion, or InstructDiffusion, point one environment variable at the directory
that contains those three repositories:

```bash
export VICL_EXTERNAL_ROOT=/absolute/path/to/external-models
```

| Adapter | Official input protocol | Local source |
| --- | --- | --- |
| `t2t-qwen` | `[A_in, A_out, B_in] + prompt -> B_out` | `eval_qwen.py` |
| `t2t-flux2` | `[A_in, A_out, B_in] + prompt -> B_out` | `eval_flux.py` |
| `t2t-omnigen2` | `[A_in, A_out, B_in] + prompt -> B_out` | `eval_omnigen.py` |
| `t2t-firered` | `[A_in, A_out, B_in] + prompt -> B_out` | `eval_firered.py` |
| `painter` | visual demo pair + query -> output | `$VICL_EXTERNAL_ROOT/Painter/Painter` |
| `prompt-diffusion` | visual demo pair + query + text -> output | `$VICL_EXTERNAL_ROOT/Prompt-Diffusion` |
| `instruct-diffusion` | source image + instruction -> output | `$VICL_EXTERNAL_ROOT/InstructDiffusion` |

The three third-party methods are not interchangeable:

- **Painter** converts task outputs into images and performs one masked-completion
  forward pass over a vertically stitched demo/query canvas. Its official ViT-Large
  architecture uses a `896 x 448` input.
- **Prompt-Diffusion** adds both the demo pair and query condition to a
  ControlNet-style Stable Diffusion pipeline. Every denoising step runs the custom
  Prompt-Diffusion ControlNet and the SD UNet.
- **InstructDiffusion** is a unified instruction-conditioned latent diffusion model
  for perception, restoration, and editing. Its official classifier-free guidance
  stacks three branches per denoising step. It does not consume a visual demo pair,
  so it must be labeled as an instruction-conditioned reference rather than an
  input-equivalent VICL baseline.

## Sanity Check

Run the dependency-light toy adapter before using a large checkpoint:

```bash
python -m efficiency.benchmark \
  --adapter toy \
  --device cpu \
  --warmup 1 \
  --repeats 3 \
  --profile-flops \
  --output-dir efficiency/results/toy
```

## Original T2T-VICL

This path is self-contained within root directory: it uses `efficiency/`, the four
`eval_*.py` backends, `model/` where a backend has local source, `data/`, and the prompt
checkpoint supplied through `--prompt-checkpoint`. It neither reads an external source
root nor imports the external baseline adapters. Model weights may be local paths or
Hugging Face identifiers.

Check each model-specific Python environment before loading a checkpoint:

```bash
python -m efficiency.preflight --require-dispatch-flops
```

For the controlled multi-task experiment, use the suite entry point:

```bash
python -m efficiency.suite \
  --adapter t2t-qwen \
  --conditions fixed ours \
  --task-manifest efficiency/t2t_tasks.json \
  --tasks deblurring__deraining dehazing__denoising \
  --max-samples 100 \
  --warmup 5 \
  --resolution 448 \
  --profile-flops \
  --output-dir efficiency/results/qwen_suite
```

The older single-sample command below remains useful for debugging and direct
Fixed/Ours paired checks; it is not the recommended paper latency protocol.

Run `Fixed` and `Ours` in the same process so the editor, sample, device, and software
stack are identical:

```bash
python -m efficiency.benchmark \
  --adapter t2t-qwen \
  --conditions fixed ours \
  --model-id Qwen/Qwen-Image-Edit-2511 \
  --prompt-checkpoint Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875 \
  --sample-index 0 \
  --warmup 1 \
  --repeats 10 \
  --profile-flops \
  --output-dir efficiency/results/qwen
```

Replace the adapter with `t2t-firered`, `t2t-omnigen2`, or `t2t-flux2` to reproduce
the calls in the corresponding evaluation file. The defaults intentionally preserve
the paper code:

- Qwen-Image-Edit: 40 steps, `true_cfg_scale=4.0`, `guidance_scale=1.0`.
- FireRed: 40 steps and `true_cfg_scale=4.0`; pass `--optimized` only when reporting
  the official fast-pipeline variant as a separate row.
- OmniGen2: `1024 x 1024`, 50 steps, text guidance 5.0, image guidance 3.0.
- FLUX.2: 30 steps, guidance 4.0, and model CPU offload, exactly as in
  `eval_flux.py`. CPU-offloaded latency is not directly comparable to fully
  GPU-resident latency unless the table states this execution mode.

`--steps` and `--seed` are passed through to all four local T2T backends. Input images
are closed after each generation, so warm-up and repeated measurements do not accumulate
open file handles. The repository paths for FLUX.2, OmniGen2, and FireRed are fixed to
CUDA device 0 and BF16; the benchmark rejects conflicting `--device`/`--dtype` values
instead of recording a configuration those helpers did not execute. Qwen follows the
requested device and dtype.

### One Model Per GPU

The launcher accepts one repeated four-field `--job` entry per model: adapter, physical
GPU id, Python executable, and model path/id. Each child receives a one-entry
`CUDA_VISIBLE_DEVICES`, so backend references to `cuda` or `cuda:0` resolve to that
job's assigned physical card. Different jobs may use incompatible Python environments.

```bash
python -m efficiency.launch_t2t \
  --job t2t-qwen 1 /envs/qwen/bin/python /weights/Qwen-Image-Edit-2511 \
  --job t2t-flux2 2 /envs/flux2/bin/python /weights/FLUX.2-dev-bnb-4bit \
  --job t2t-omnigen2 3 /envs/omnigen2/bin/python /weights/OmniGen2 \
  --conditions fixed ours \
  --prompt-checkpoint /weights/checkpoint-4875 \
  --task-manifest efficiency/t2t_tasks.json \
  --max-samples 100 --warmup 5 --resolution 448 \
  --profile-flops \
  --output-root efficiency/results/parallel
```

Use `--dry-run` to inspect commands and `--fail-fast` to stop the other jobs after one
fails. Per-model logs and `launch_manifest.json` are written below the output root.
GPU visibility is isolated, but CPU, RAM, storage, PCIe, and host power are shared.
Parameter and FLOP counts are unaffected by concurrent execution; paper-facing latency
must be checked against exclusive one-model runs, especially for CPU-offloaded FLUX.2.

Use `--parameters-only` to audit a model before paying the cost of generation. The
T2T-VICL prompt generator uses the same messages, sampling parameters, and
`max_new_tokens=8192` bound as `generate_text_prompt`; the controlled suite additionally
fixes all three visual inputs to the declared resolution. A contract-tested lightweight
mirror lives in
`efficiency/prompting.py`, so importing a backend for benchmarking no longer requires
Gemini, GrAInS, VIEScore, or other evaluation-only packages; the test fails if its
relation-request text drifts from `eval.py`. Console printing from the evaluation loop is
not part of timed inference.

To expand a completed smoke run without repeating its measured queries, pass its suite
JSON to `--resume-from`, increase `--max-samples`, and optionally add
`--reverse-order`. The merged JSON recomputes statistics from all raw rows and records
reused/new indices plus a cross-process latency warning. Model, manifest, condition,
seed, resolution, and checkpoint signatures must match exactly.

## Third-Party Models

Painter can report architecture parameters without a checkpoint, but paper latency
should be measured with the official checkpoint and recorded precision:

```bash
python -m efficiency.benchmark \
  --adapter painter \
  --checkpoint /path/to/painter_vit_large.pth \
  --painter-task restoration \
  --dtype fp32 \
  --warmup 3 \
  --repeats 10 \
  --profile-flops \
  --output-dir efficiency/results/painter
```

`--painter-task` selects the official task-family output conversion (including the
discrete-label scaling rules). In the controlled suite, every converted query output
remains `448 x 448`; it is not resized back to each source file's native dimensions.
The default prediction-only path skips the SmoothL1 loss that the released evaluation
scripts compute but never use. Pass `--painter-include-script-loss` only to reproduce
the old script literally. The JSON also records the instantiated window/global block
layout, exposing the released `8glb` constructor's tuple-of-lists anomaly.
The released architecture has a fixed `896 x 448` patch layout, so the adapter rejects
other `--resolution` values instead of failing later with a misleading mask-shape error.

Prompt-Diffusion uses its official Diffusers model and DDIM call:

```bash
python -m efficiency.benchmark \
  --adapter prompt-diffusion \
  --model-id zhendongw/prompt-diffusion-diffusers \
  --steps 50 \
  --dtype fp16 \
  --warmup 1 \
  --repeats 10 \
  --profile-flops \
  --output-dir efficiency/results/prompt_diffusion
```

InstructDiffusion requires the official checkpoint and should normally run in its
official environment:

```bash
python -m efficiency.benchmark \
  --adapter instruct-diffusion \
  --checkpoint /path/to/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt \
  --resolution 512 \
  --steps 50 \
  --dtype fp16 \
  --warmup 1 \
  --repeats 10 \
  --profile-flops \
  --output-dir efficiency/results/instruct_diffusion
```

Painter, Prompt-Diffusion, and InstructDiffusion pin old and mutually different
dependencies. Run each in its official environment and aggregate their JSON outputs
afterward; do not force all three into the current T2T-VICL environment merely to
produce one command.

## Aggregate Separate Runs

```bash
python -m efficiency.report \
  efficiency/results/qwen/t2t-qwen_suite_latest.json \
  efficiency/results/painter/painter_suite_latest.json \
  efficiency/results/prompt_diffusion/prompt-diffusion_suite_latest.json \
  efficiency/results/instruct_diffusion/instruct-diffusion_suite_latest.json \
  --output-dir efficiency/results/paper_table
```

The aggregator recognizes both single-sample and multi-task schemas. Suite rows are
aligned by task before computing `Fixed`/`Ours` deltas and retain a separate
`__macro__` row.

Each benchmark writes a timestamped JSON, a `*_latest.json`, `comparison.csv`, and
`comparison.md`. Keep the JSON files as the reproducibility record; they contain raw
latency samples, exact sampling configuration, sample paths, software versions, GPU
names/driver/power limit snapshot, parameter breakdowns, and top FLOP operators.

End-to-end latency is the paper-grade number. Stage spans are barrier-free diagnostics
defined as the maximum of their host scope and CUDA current-stream events; they are
intentionally non-additive and do not perturb the pipeline with stage-boundary
synchronizations. FLOPs are profiled in a separate inference after latency measurement.

## Paper Reporting Checklist

Use the same GPU model, batch size 1, concurrency 1, processed resolution, warm-up count,
precision, and software environment for every comparable row. Use each model's exact
quality-evaluation sampling steps in the primary table and report those steps and any CPU
offload/quantization/optimization mode in the caption. A separate 30-step table may
isolate sampler-step cost but cannot replace the quality-matched table. Measure T2T-VICL
`Fixed` and `Ours` in one invocation. At minimum, include:

1. unique logical loaded parameters, physical packed parameter payload, and parameter
   coverage/completeness;
2. method-trained parameters, separately from total loaded parameters;
3. full-pipeline runtime FLOPs, mean/min/max and sample count for input-dependent
   `Ours`, the one-MAC-equals-two-FLOPs convention, and the exact coverage/status field;
4. prompt-generation, editor/model, and synchronized end-to-end latency;
5. per-task query count, mean, median, standard deviation, p90, and p95;
6. GPU, precision, processed resolution, sampling steps, batch/concurrency, software
   versions, and execution/offload mode.

Do not publish `estimated_total_flops` as complete when status is `partial`. Resolve the
listed unsupported custom operators first, or label the number with its coverage status.
The suite suppresses macro FLOPs unless every task has status `ok`.

For multi-GPU runs, the report stores both maximum per-device and summed PyTorch peak
allocation. Peak allocation includes resident model tensors but excludes CPU-offloaded
memory and allocations made outside PyTorch. Keep `Fixed` and `Ours` in one invocation;
schema version 2 isolates their condition-specific model residency automatically.

Closed API models such as Gemini and Seedream can provide client-observed latency, but
their parameters and FLOPs are undisclosed. Do not infer those values. API latency
also includes network and service queueing, so place it in a separate table or mark it
as non-comparable to local GPU latency.
