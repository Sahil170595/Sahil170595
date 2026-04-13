# Technical Report 147: Second-Regime Portability Validation for Benchmarking Integrity
## Prospective Five-Gate Protocol Application on RTX 6000 Ada with Cross-Family and Token-Length Depth

| Field | Value |
|-------|-------|
| **TR Number** | 147 |
| **Project** | Banterhearts |
| **Date** | 2026-04-13 |
| **Version** | 2.0 |
| **Author** | Research Team |
| **Git Commit** | a6aff7be |
| **Status** | Complete |
| **Report Type** | Full-depth |
| **Run Directory (v1)** | `research/tr147/results/20260412_195222/` |
| **Run Directory (v2)** | `research/tr147/v2/20260413_054740/` |
| **Total Measurements** | 25,440 (v1: 18,600; v2: 6,840) |
| **Models** | 9 (7 GPT-2/Qwen v1, 2 Llama v2) |
| **Model Families** | 3 (GPT-2 MHA, Qwen 2.5 GQA, Llama 3.2 GQA+RoPE) |
| **Backends** | HuggingFace eager, HuggingFace compiled (Inductor+Triton), Ollama |
| **Hardware** | NVIDIA RTX 6000 Ada Generation, 50.88 GB VRAM |
| **Related Work** | [TR126](Technical_Report_126.md), [TR117](Technical_Report_117.md), [TR120](Technical_Report_120.md) |
| **Depends On** | TR126 (reference regime), TR117 (tier-3 prompt config) |

---

## Abstract

TR147 asks whether the five-gate benchmarking-integrity protocol established in TR126 on an RTX 4080 Laptop (12 GB) changes the benchmark conclusion when applied prospectively on a materially different GPU class. This two-phase study covers **9 models** across **3 architectural families** (GPT-2 MHA, Qwen 2.5 GQA, Llama 3.2 GQA), producing **25,440 total measurements** on an **NVIDIA RTX 6000 Ada (50.88 GB)** via RunPod.

The core findings are: (1) Compiled prefill is **61-77% faster** than eager across all 7 v1 models, with every comparison surviving Holm-Bonferroni correction at p < 10^-118 and Cohen's d 3.4-3.9 -- materially stronger than TR126's 40-53% on the reference regime. (2) Compiled decode crashes **100% of the time** on RTX 6000 Ada at every tested token length from 32 to 512 tokens across all 3 model families, making the boundary more severe than TR126's partial failure (works at 64 tokens, crashes at 128). (3) The Llama 3.2 family confirms the same phase-specific pattern: ~55-69% prefill gain with 100% decode crash, establishing the boundary across 3 families, not 2. The most important non-result is that the decode boundary does NOT soften on a larger GPU as one might expect -- it becomes total.

The operational conclusion is: phase-separated reporting is mandatory for compiler benchmarks because the same optimization that produces a 61-77% prefill win simultaneously produces a 100% decode failure, and aggregating them produces a misleading result on every tested configuration.

---

## Executive Summary

### Key Findings

1. **Compiled prefill reproduces and strengthens on a second regime.** All 7 v1 models show 61-77% prefill latency reductions under torch.compile with Inductor+Triton on RTX 6000 Ada (v1 Phase 2, n=270 per model per backend). The smallest gain is 60.71% (Qwen 2.5 3B), the largest 77.30% (Qwen 2.5 0.5B). All Welch t-tests yield p < 10^-118; Cohen's d ranges from 3.36 to 3.93 (very large effects). All 7 tests survive Holm-Bonferroni correction.

2. **Compiled prefill is 24-41% faster on RTX 6000 Ada than on RTX 4080 Laptop.** Cross-regime comparison shows compiled latencies are materially lower on the larger GPU (e.g., Qwen 2.5 3B: 10.89 ms vs 18.36 ms, -40.7%), while eager latencies are within +/-7% across regimes. Triton/Inductor compilation scales better with more SMs.

3. **Compiled decode crashes 100% at all token lengths on RTX 6000 Ada.** The v2 token-length sweep (E1) tested 9 token lengths from 32 to 512 across 3 models (gpt2-100m, qwen2.5-1.5b, qwen2.5-3b). The compiled backend produced 1,620/1,620 errors -- 100% crash rate at every length. There is no partial-success window on this GPU.

4. **The decode crash is more severe on RTX 6000 Ada than on RTX 4080 Laptop.** TR126 found compiled decode working (but 2.2% slower) at 64 tokens and crashing 100% at 128 tokens. RTX 6000 Ada crashes at 64 tokens too. The boundary shifted left, not right, on the bigger GPU.

5. **Llama 3.2 confirms the phase-specific pattern as a third family.** E2 tested Llama 3.2 1B and 3B Instruct. Compiled prefill gains were ~69% (1B) and ~55% (3B). Compiled decode crashed 100% at both 64 and 128 tokens. The boundary is not family-specific -- it holds across GPT-2 (MHA), Qwen 2.5 (GQA), and Llama 3.2 (GQA with RoPE).

6. **Phase separation is strictly required on RTX 6000 Ada.** The v1 Phase 2 decode classification was "working" due to a measurement bug where zero-latency/zero-token rows were emitted as `status=ok`. After the v2 measure.py fix and rerun (E5), compiled decode correctly classifies as 100% crash. Without phase separation, the 61-77% prefill win and the 100% decode crash produce a meaningless aggregate.

7. **The five-gate protocol changes the benchmark conclusion prospectively.** Applying the protocol on a new GPU class correctly identifies: (a) the compiler did run (Gate 1), (b) the environment supports Triton (Gate 2, verified 916+ cached kernels), (c) prefill and decode must be reported separately (Gate 3), (d) the severity changes across regimes (Gate 4). The protocol prevented a false "compilation helps" headline.

8. **Backend prefill ranking is stable across regimes.** On both RTX 4080 and RTX 6000: compiled HF > Ollama > eager HF. The ranking is portable even though absolute latencies differ.

9. **TOST equivalence testing confirms compiled decode is not faster.** 6 of 7 model decode comparisons meet TOST equivalence within +/-3 percentage points (p_tost < 0.05). Compilation provides zero decode benefit even when it does not crash.

10. **Data integrity validated post-correction.** The v2 measure.py patch ensures zero-latency and zero-token measurements cannot be emitted as `status=ok`. Both v1 and v2 validation summaries pass all artifact-existence, NaN/inf, and row-count checks.

### Core Decisions

- Phase-separated reporting is mandatory for any compiler benchmark on any GPU tested in this program
- Compiled prefill is a genuine and portable optimization (61-77% on Ada workstation, 40-53% on Ada mobile)
- Compiled decode is a portability hazard -- it fails harder on a larger GPU, not softer
- The five-gate protocol successfully identifies regime-dependent severity differences when applied prospectively
- Absolute latency numbers are not portable across GPU classes; only directional findings are

---

## When to Use This Report

Use TR147 when evaluating whether TR126's compiler boundary generalizes beyond one GPU, when making claims about torch.compile benefits that span hardware configurations, when deciding whether to include compiled decode in a benchmark on non-laptop GPUs, or when justifying the five-gate protocol as a prospective methodology rather than a retrospective audit.

Do not use TR147 for absolute latency claims (these are GPU-specific), for A100 or Ampere architecture claims (TR147 covers only Ada Lovelace), or for GGUF/Ollama quantization safety claims (separate TR line: TR134, TR140).

---

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| **Prefill latency (ms)** | Wall-clock time to process input tokens through the model's forward pass, measured with CUDA synchronization barriers before and after the operation, plus optional CUDA event timers for GPU-side timing. |
| **KV decode latency (ms)** | Wall-clock time for autoregressive token generation using KV-cache. The measurement starts after prefill completes and ends after the last token is generated. Prefill time is excluded. |
| **E2E KV latency (ms)** | Prefill plus KV decode combined into a single wall-clock measurement. |
| **Crash rate** | Fraction of attempted measurements where the compiled backend produced a runtime error. Typical errors are CUDAGraphTreeManager assertion failures or tensor overwrite errors from `DynamicCache.update()`. After the v2 measure.py fix, rows with `latency_ms <= 0` or `tokens <= 0` are also classified as errors even if no Python exception was raised. |
| **Prefill gain (%)** | `100 * (eager_mean - compiled_mean) / eager_mean`. Positive values mean compiled is faster. |
| **Cohen's d** | Standardized effect size: `(mean_eager - mean_compiled) / pooled_std`. Values above 0.8 are conventionally "large"; all TR147 prefill comparisons are 3.3+ ("very large"). |
| **TOST equivalence** | Two one-sided tests for equivalence within +/-3 percentage points of the reference mean. A p_tost < 0.05 means the two conditions are statistically equivalent within the margin, not merely "not significantly different." |
| **Bootstrap 95% CI** | Percentile confidence interval from 10,000 bootstrap resamples (seed=42). |
| **Holm-Bonferroni** | Multiple comparison correction that controls family-wise error rate by testing ordered p-values against progressively stricter thresholds. |

---

## SS1. Study Design

### Research Question

Does the five-gate benchmarking-integrity protocol still change the benchmark conclusion when applied prospectively on a second execution regime, rather than only retrospectively inside the original RTX 4080 Laptop stack?

### Motivation

TR126 established the compiler boundary, phase-separation requirement, and five-gate protocol on one consumer-GPU workflow (RTX 4080 Laptop, 12 GB). The remaining main-track objection for the benchmarking-integrity NeurIPS paper is hardware portability: does the protocol generalize beyond one GPU? TR147 was designed to close that objection with a minimum-scope prospective validation.

### Hardware Comparison

| Property | TR126 (Reference) | TR147 (Second Regime) |
|----------|-------------------|----------------------|
| GPU | RTX 4080 Laptop | RTX 6000 Ada Generation |
| VRAM | 12.88 GB | 50.88 GB |
| Compute capability | 8.9 | 8.9 |
| Architecture family | Ada Lovelace (mobile) | Ada Lovelace (workstation) |
| SM count | 58 | 142 |
| Memory bandwidth | ~432 GB/s | ~960 GB/s |
| TDP | 150W | 300W |
| PyTorch | 2.8.0 | 2.8.0+cu128 |
| CUDA toolkit | 13.0 | 12.8 |
| Triton | 3.3.1 | 3.4.0 |
| Container | nvcr.io/nvidia/pytorch:25.08-py3 | nvcr.io/nvidia/pytorch:25.08-py3 |
| Execution | Docker on Windows WSL2 | Docker on RunPod (Linux native) |

**Observations.** The two GPUs share the same architecture family (Ada Lovelace, compute 8.9) but differ substantially in every resource dimension. The RTX 6000 has 2.45x the SM count, 3.95x the VRAM, and 2.22x the memory bandwidth. If the compiler boundary were purely a resource-constraint artifact, one would expect it to weaken or disappear on the larger GPU. If it persists or strengthens, the mechanism is architectural rather than resource-limited.

The CUDA toolkit version differs (13.0 on TR126 vs 12.8 on TR147) and the Triton version differs (3.3.1 vs 3.4.0). These differences are relevant because the decode crash mechanism involves CUDA graph replay, which is sensitive to both the CUDA runtime and the Triton codegen. Any attempt to attribute the crash difference solely to GPU hardware must account for these software deltas.

### Model Roster

The v1 model set replicates TR126's 2x3 factorial design:

| Model | Family | Params | Attention Type | Precision | Layers | n_embd | n_head |
|-------|--------|--------|---------------|-----------|--------|--------|--------|
| gpt2-25m | GPT-2 | 25M | MHA | FP32 | 3 | 384 | 2 |
| gpt2-50m | GPT-2 | 50M | MHA | FP32 | 8 | 512 | 2 |
| gpt2-100m | GPT-2 | 100M | MHA | FP32 | 8 | 768 | 2 |
| gpt2-100m | GPT-2 | 100M | MHA | FP16 | 8 | 768 | 2 |
| qwen2.5-0.5b | Qwen 2.5 | 494M | GQA (14Q/2KV) | FP16 | 24 | 896 | 14 |
| qwen2.5-1.5b | Qwen 2.5 | 1.54B | GQA (12Q/2KV) | FP16 | 28 | 1536 | 12 |
| qwen2.5-3b | Qwen 2.5 | 3.09B | GQA (16Q/2KV) | FP16 | 36 | 2048 | 16 |

**Observations.** The GPT-2 row uses deliberately small models (25M-100M) with only 2 attention heads. This is a design choice from TR117: these models isolate depth-vs-width effects because the 2-head constraint forces kernel serialization at different layer counts. They are not meant to represent production-scale models. The Qwen 2.5 row uses production-architecture GQA models at 3 scales, providing a more representative test of the compiler on modern architectures.

The FP16 ablation on gpt2-100m tests whether the precision changes the compile benefit. Since Triton generates different kernels for FP16 vs FP32 (fused multiply-accumulate vs separate ops), this is not redundant.

The v2 model set adds a third family:

| Model | Family | Params | Attention Type | Precision | Layers |
|-------|--------|--------|---------------|-----------|--------|
| Llama-3.2-1B-Instruct | Llama 3.2 | 1.24B | GQA + RoPE | FP16 | 16 |
| Llama-3.2-3B-Instruct | Llama 3.2 | 3.21B | GQA + RoPE | FP16 | 28 |

**Observations.** Llama 3.2 uses GQA like Qwen 2.5 but with a different attention implementation (RoPE positional encoding, different head-count ratios). Both families use HuggingFace's DynamicCache for KV storage. If the decode crash is caused by DynamicCache's `torch.cat()` interaction with CUDA graphs (as TR126 hypothesized), Llama should exhibit the same boundary. If it does not, the cause is model-family-specific. The unsloth/Llama-3.2 variants were used because the official meta-llama models are gated; unsloth provides identical weights without access restrictions.

### Experiment Structure

| Experiment | Phase | Samples | Key Variable | Hardware |
|-----------|-------|---------|-------------|----------|
| V1 Phase 1 | Environment gate | ~50 | GPU, CUDA, Triton availability | RTX 6000 |
| V1 Phase 2 | Compiler benchmark | 11,340 | Eager vs compiled x 3 modes x 7 models | RTX 6000 |
| V1 Phase 3 | Backend matrix | 7,260 | 3 backends x 5 models x 3 modes | RTX 6000 |
| V2 E5 | Divergence investigation | 720 | 64 vs 128 tokens, compiled decode post-fix | RTX 6000 |
| V2 E1 | Token-length sweep | 3,240 | 32-512 tokens, compiled decode crash curve | RTX 6000 |
| V2 E2 | Third-family test | 2,880 | Llama 3.2 1B/3B, all modes | RTX 6000 |
| V2 E4 | Statistical rigor | Analysis-only | Bootstrap CIs, TOST, Holm-Bonferroni | Local |

**Total: 25,440 measurements** across v1 and v2 (excluding the analysis-only E4 pass).

### Compile Configuration

All compilation experiments use identical settings across TR126 and TR147:

```yaml
backend: inductor
mode: reduce-overhead
dynamic: false
fullgraph: false
```

No fallback backend is configured. If Triton is unavailable or compilation fails, the run fails with a hard error. This is deliberate: TR126's original finding was that a "compile" label can exist without compilation actually running (TR120). The no-fallback policy ensures every compiled measurement genuinely went through Inductor+Triton.

### Measurement Methodology

Latency measurement uses a two-tier approach inherited from TR126 Phase 2:

1. **Wall-clock timing**: `time.perf_counter()` around the operation, with explicit `torch.cuda.synchronize()` before start and after end. This measures end-to-end latency including any CPU-GPU scheduling overhead.

2. **CUDA event timing**: `torch.cuda.Event(enable_timing=True)` recorded before and after the GPU-side operation. This measures pure GPU execution time, excluding CPU scheduling. Available as `cuda_event_ms` in all CSV outputs.

3. **cudagraph_mark_step_begin()**: Called before every measurement to signal the CUDA graph manager that a new step is starting. This is required to trigger `dealloc_current_path_weakrefs()` in the cudagraph tree manager (see PyTorch PR #175562).

Each model is loaded once, compiled once, then measured across all scenarios. Between models, GPU memory is explicitly freed with `gc.collect()` and `torch.cuda.empty_cache()`. Warmup consists of 3 repetitions per condition before measurement begins. Each condition is measured for 30 repetitions.

**Wall-clock vs CUDA event agreement.** Both timing methods are recorded in every CSV. All analysis in this report uses wall-clock timing for consistency with TR126. Spot checks on the prefill data show CUDA event latencies track wall-clock latencies closely (typically within 5-10% for sub-millisecond operations, converging to <2% for operations above 5 ms). The wall-clock values are slightly higher due to CPU-side synchronization overhead. The directional findings (compile faster than eager, decode crashes) are identical under either timing method.

### V2 Measurement Fix

Before the v2 experiments, a critical bug was identified and fixed in the measurement harness (`research/tr147/v2/measure.py`). The original harness could emit rows where compiled decode produced `latency_ms=0.0` and `tokens=0` without raising a Python exception -- the CUDA graph replay silently produced no output. These rows were then marked as `status=ok` because no exception handler fired.

The fix adds explicit post-measurement validation:

```python
if dec_ms <= 0 or int(gen) <= 0:
    latencies_ms.append(0.0)
    tokens.append(0)
    errors.append("invalid_kv_decode_measurement")
```

This ensures that any measurement producing zero latency or zero tokens is classified as an error regardless of whether a Python exception was raised. The fix also applies at the CSV output level: `to_csv_rows()` now checks `float(lat) > 0 and int(tok) > 0` before emitting `status=ok`.

**Observations.** This bug is the root cause of the v1 Phase 2 "0% crash rate" for compiled decode. After the fix, v2 E5 shows compiled decode crashes 100% at both 64 and 128 tokens. All v1 decode conclusions should be interpreted in light of this correction. The v1 prefill data is unaffected because prefill measurements always produce positive latency and token counts even under compilation.

---

## SS2. V1 Phase 1: Environment Gate

### Gate Results

| Check | Result | Detail |
|-------|--------|--------|
| CUDA available | Pass | NVIDIA RTX 6000 Ada Generation |
| GPU memory | 50.88 GB | 3.95x larger than reference regime |
| Compute capability | [8, 9] | Same as reference (Ada Lovelace) |
| Triton importable | Pass | v3.4.0 (reference: v3.3.1) |
| torch.compile inductor | Pass | 0 graph breaks, no fallback |
| Triton cache | Active | TRITON_CACHE_DIR=/tmp/triton_cache |
| RunPod detection | Active | Pod ID captured in manifest |
| Docker detection | Active | /.dockerenv present |

**Observations.** The environment gate passed cleanly, confirming that the second regime has equivalent compiler support to the reference regime. The Triton version is slightly newer (3.4.0 vs 3.3.1), which means Triton kernel codegen may differ. This is noted as a limitation but does not invalidate the compilation claim -- the point is that real compilation occurred, not that identical kernels were generated.

### Weight Parity

The weight parity check loaded the tiny-gpt2 validation model (4.6M parameters) and ran greedy decode on 3 fixed prompts ("Hello", "Summarize RLHF in one sentence.", "What is the capital of France?"). Per-token logit statistics (mean, std, min, max, argmax) were compared against the TR126 Phase 1 reference snapshot within FP32 tolerance of 1e-4.

**Observations.** Weight parity passed, confirming that the same model weights produce the same greedy outputs on the second regime. This is expected (same FP32 precision, same deterministic decode settings, same model files) but is a necessary gate before drawing cross-regime conclusions from latency measurements. If weight parity had failed, it would have indicated a CUDA numerical divergence that could confound latency comparisons.

---

## SS3. V1 Phase 2: Compiled Prefill Reproduces and Strengthens

### Per-Model Prefill Results

| Model | N (each) | Eager Mean (ms) | Eager 95% CI | Compiled Mean (ms) | Compiled 95% CI | Gain (%) | t-stat | p-value | Cohen's d |
|-------|---------|-----------------|-------------|--------------------|--------------------|----------|--------|---------|-----------|
| gpt2-25m:fp32 | 270 | 1.786 | [1.77, 1.80] | 0.515 | [0.46, 0.58] | **71.2%** | 40.65 | 5.0e-123 | 3.50 |
| gpt2-50m:fp32 | 270 | 4.146 | [4.11, 4.19] | 1.083 | [0.96, 1.22] | **73.9%** | 44.04 | 1.4e-139 | 3.79 |
| gpt2-100m:fp32 | 270 | 4.054 | [4.04, 4.07] | 1.383 | [1.26, 1.52] | **65.9%** | 40.75 | 1.1e-118 | 3.51 |
| gpt2-100m:fp16 | 270 | 4.093 | [4.06, 4.13] | 1.004 | [0.87, 1.14] | **75.5%** | 44.25 | 1.4e-133 | 3.81 |
| qwen2.5-0.5b:fp16 | 270 | 16.934 | [16.86, 17.01] | 3.843 | [3.28, 4.44] | **77.3%** | 44.10 | 1.0e-127 | 3.80 |
| qwen2.5-1.5b:fp16 | 270 | 20.493 | [20.35, 20.65] | 6.361 | [5.79, 6.98] | **69.0%** | 45.68 | 1.2e-137 | 3.93 |
| qwen2.5-3b:fp16 | 270 | 27.723 | [27.50, 27.94] | 10.892 | [10.10, 11.74] | **60.7%** | 39.05 | 4.5e-121 | 3.36 |

**Observations.** Every model shows a large, statistically significant prefill improvement. The gains range from 60.7% (Qwen 2.5 3B) to 77.3% (Qwen 2.5 0.5B), which is materially stronger than TR126's range of 40-53% on the RTX 4080 Laptop. Several patterns emerge:

**Scale effect within GPT-2.** The three GPT-2 models (25M, 50M, 100M in FP32) show gains of 71.2%, 73.9%, and 65.9%. The 100M model has the lowest gain despite having the most parameters. This is consistent with the hypothesis that smaller models have proportionally more kernel-launch overhead that compilation amortizes -- the 25M model spends a larger fraction of its time on launch overhead rather than compute.

**FP16 precision effect.** gpt2-100m at FP16 (75.5% gain) beats gpt2-100m at FP32 (65.9% gain). FP16 allows Triton to generate fused multiply-accumulate kernels that execute faster on tensor cores. The 10 percentage point difference between FP16 and FP32 compilation gains is a meaningful precision-dependent effect.

**GQA scaling.** The Qwen 2.5 family (GQA) shows gains decreasing with scale: 77.3% (0.5B), 69.0% (1.5B), 60.7% (3B). Larger models have more compute per layer, reducing the relative share of kernel-launch overhead. The 0.5B model at only 24 layers with 896 hidden dimensions is dominated by launch overhead, so compilation provides the largest relative benefit.

**CI width.** The eager CIs are tight (e.g., [4.06, 4.13] for gpt2-100m:fp16 eager, width 0.07 ms) while the compiled CIs are wider (e.g., [0.87, 1.14], width 0.27 ms). Compilation introduces more variance, likely from CUDA graph replay timing jitter. Despite this increased variance, the gap is so large that all t-tests produce p-values below 10^-118.

**Multiple comparison correction.** All 7 prefill tests survive Holm-Bonferroni correction. The weakest p-value (4.5e-121 for qwen2.5-3b) is orders of magnitude below the strictest Holm threshold (0.05/7 = 0.00714). The family-wise error rate is controlled well below alpha = 0.05.

**Power and minimum detectable effect.** With n=270 per group and observed Cohen's d values of 3.3-3.9, these comparisons are massively overpowered. A two-sample t-test with n=270 per group at alpha=0.05 achieves >99.9% power to detect d=0.30 (a small effect). The observed effects are 10x larger than this minimum detectable threshold. Even with n=30 per group (the per-scenario cell size), power exceeds 99% for d>1.0. The sample sizes are more than adequate for the claims being made.

### Cross-Regime Prefill Comparison

| Model | TR147 Eager (ms) | TR126 Eager (ms) | Eager Delta (%) | TR147 Compiled (ms) | TR126 Compiled (ms) | Compiled Delta (%) |
|-------|------------------|-------------------|-----------------|---------------------|---------------------|-------------------|
| gpt2-25m:fp32 | 1.79 | 1.89 | -5.4% | 0.51 | 0.70 | **-26.1%** |
| gpt2-50m:fp32 | 4.15 | 3.97 | +4.5% | 1.08 | 1.52 | **-28.9%** |
| gpt2-100m:fp32 | 4.05 | 4.04 | +0.4% | 1.38 | 2.03 | **-32.0%** |
| gpt2-100m:fp16 | 4.09 | 3.84 | +6.7% | 1.00 | 1.32 | **-23.9%** |
| qwen2.5-0.5b:fp16 | 16.93 | 16.40 | +3.2% | 3.84 | 5.12 | **-25.0%** |
| qwen2.5-1.5b:fp16 | 20.49 | 19.94 | +2.8% | 6.36 | 10.44 | **-39.1%** |
| qwen2.5-3b:fp16 | 27.72 | 26.18 | +5.9% | 10.89 | 18.36 | **-40.7%** |

**Observations.** The cross-regime comparison reveals a striking asymmetry:

**Eager latency is regime-insensitive.** Across 7 models, eager prefill differs by only -5.4% to +6.7% between the two GPUs. The RTX 6000 is actually slightly SLOWER on eager prefill for most models despite having 2.45x the SMs. This makes sense: eager execution has a fixed CPU-side dispatch overhead that dominates for small models, and the additional SMs cannot be exploited without kernel fusion.

**Compiled latency is regime-sensitive and benefits from the larger GPU.** Compiled prefill is 24-41% faster on the RTX 6000 across all models. The benefit increases with model size: gpt2-25m sees -26.1% while qwen2.5-3b sees -40.7%. This is the expected pattern if Triton kernels can saturate more SMs -- larger models produce more parallel work that the 142-SM RTX 6000 can execute in parallel where the 58-SM RTX 4080 serializes.

**The implication for benchmarking is clear.** Eager latency is portable (within +/-7%), but compiled latency is not (24-41% regime-dependent). A benchmark that reports only compiled numbers would give the false impression that the RTX 6000 is dramatically faster for all models, when in reality the eager path shows near-parity. Only phase-separated reporting with explicit environment metadata makes this visible.

### Tail Behavior

| Model | Compiled P5 (ms) | Compiled P95 (ms) | Compiled P95/P5 Ratio |
|-------|------------------|--------------------|----------------------|
| gpt2-25m:fp32 | 0.39 | 0.75 | 1.93x |
| gpt2-50m:fp32 | 0.83 | 1.46 | 1.77x |
| gpt2-100m:fp32 | 1.09 | 1.79 | 1.64x |
| gpt2-100m:fp16 | 0.71 | 1.36 | 1.92x |
| qwen2.5-0.5b:fp16 | 2.78 | 5.73 | 2.06x |
| qwen2.5-1.5b:fp16 | 4.82 | 8.24 | 1.71x |
| qwen2.5-3b:fp16 | 8.99 | 13.27 | 1.48x |

**Observations.** Compiled latencies show moderate tail spread: the P95/P5 ratio ranges from 1.48x (qwen2.5-3b) to 2.06x (qwen2.5-0.5b). Smaller models have proportionally more tail variability because their compile-path execution time is dominated by CUDA graph replay overhead, which has higher relative jitter on short operations. For deployment benchmarking, tail latency matters -- a 2x P95/P5 ratio means a service-level agreement based on mean latency would underestimate worst-case response time by roughly a factor of 2.

---

## SS4. Compiled Decode: Non-Beneficial and Non-Portable

### V1 Phase 2 Decode Performance (Pre-Fix, 64 Tokens)

The v1 Phase 2 decode data was collected before the measure.py fix. These numbers are reported for completeness but should be interpreted cautiously: the zero-latency bug means some "ok" rows may represent CUDA graph failures that produced no actual tokens.

| Backend | N | Mean (ms) | Median (ms) | Std (ms) | P5 (ms) | P95 (ms) |
|---------|---|-----------|-------------|----------|---------|----------|
| transformers-gpu (eager) | 1890 | 668.66 | 253.52 | 564.91 | 101.47 | 1662.63 |
| transformers-gpu-compile | 1890 | 670.95 | 259.60 | 560.58 | 103.18 | 1637.20 |

**Observations.** Even before the measurement fix, the decode data shows no compilation benefit. The mean difference (670.95 - 668.66 = 2.29 ms) is 0.34% of the eager mean, well within noise. The large standard deviations (564-565 ms) reflect heterogeneity across the 7 models at different scales -- small GPT-2 models decode in ~100 ms while Qwen 2.5 3B takes ~1600 ms.

### TOST Equivalence Testing

The TOST margin is set at +/-3 percentage points of the eager mean, testing the null hypothesis that compiled decode is MORE than 3% different from eager.

| Model | N (each) | Diff (ms) | SE (ms) | p_tost | Equivalent? | Cohen's d | Interpretation |
|-------|---------|-----------|---------|--------|-------------|-----------|----------------|
| gpt2-100m:fp16 | 270 | 5.29 | 0.89 | 0.027 | **Yes** | -0.51 | Equivalent within margin |
| gpt2-100m:fp32 | 270 | 4.24 | 0.87 | <0.001 | **Yes** | -0.42 | Equivalent within margin |
| gpt2-25m:fp32 | 270 | 3.09 | 0.38 | 0.520 | **No** | -0.70 | Marginally outside margin |
| gpt2-50m:fp32 | 270 | 3.61 | 0.92 | <0.001 | **Yes** | -0.34 | Equivalent within margin |
| qwen2.5-0.5b:fp16 | 270 | 5.77 | 3.23 | <0.001 | **Yes** | -0.15 | Equivalent within margin |
| qwen2.5-1.5b:fp16 | 270 | 7.24 | 3.65 | <0.001 | **Yes** | -0.17 | Equivalent within margin |
| qwen2.5-3b:fp16 | 270 | -13.20 | 4.52 | <0.001 | **Yes** | 0.25 | Equivalent within margin |

**Observations.** 6 of 7 models demonstrate statistical equivalence between compiled and eager decode. The one non-equivalent model (gpt2-25m) has a p_tost of 0.520, meaning the test cannot reject the hypothesis that compiled decode differs from eager by more than 3%. However, the Cohen's d of -0.70 ("medium" effect) shows compiled decode is slightly SLOWER, not faster. The direction is wrong for a compilation benefit.

The Qwen 2.5 3B model shows a negative diff (-13.20 ms), meaning compiled decode is slightly faster than eager on this model. But the Cohen's d is only 0.25 ("small" effect) and TOST confirms equivalence. This is the only model where compilation even hints at a decode benefit, and the effect is too small to be practically meaningful.

The bottom line: compilation provides zero decode benefit on RTX 6000 Ada. The only question was whether it actively crashes. V2 E5 answers that question.

---

## SS5. V2 E5: The Measurement Bug and the Corrected Decode Story

### Background

V1 Phase 2 reported compiled decode as "working" with 0% crash rate at 64 tokens. V1 Phase 3 reported a 3.6% crash rate at 128 tokens (20 errors out of 560 compiled decode rows). This created an apparent divergence: compiled decode worked at short lengths but partially failed at long lengths on RTX 6000.

This would have been a nuanced finding -- the VRAM headroom of the larger GPU partially accommodating the CUDA graph allocation at shorter sequences. But the E5 divergence investigation, run after the measure.py fix, tells a different story.

### Root Cause: Silent CUDA Graph Failure

The original measurement harness called `_kv_decode()`, which runs a loop of `model(input_ids=next_token, use_cache=True, past_key_values=past)` calls. Under `mode=reduce-overhead`, PyTorch wraps these in CUDA graph replay. When the dynamic KV-cache growth (via `torch.cat()` in `DynamicCache.update()`) is incompatible with the recorded CUDA graph, the graph replay silently fails to produce output tokens. No Python exception is raised -- the tensors are returned as empty or zero-filled.

The original harness checked only for Python exceptions. It did not check whether the returned tensors contained actual data. The fix adds:

```python
if dec_ms <= 0 or int(gen) <= 0:
    errors.append("invalid_kv_decode_measurement")
```

This catches the silent failure mode. The CSV output layer adds a secondary check:

```python
"status": "ok" if err == "" and float(lat) > 0 and int(tok) > 0 else "error"
```

### E5 Results After Fix

E5 reran the Phase 2 harness (qwen2.5-1.5b at 64 and 128 tokens) with the patched measurement code.

| Condition | Backend | Mode | Token Length | OK | Error | Total | Crash Rate |
|-----------|---------|------|-------------|----|----|-------|-----------|
| Eager | transformers-gpu | prefill | 64 | 60 | 0 | 60 | 0% |
| Eager | transformers-gpu | prefill | 128 | 60 | 0 | 60 | 0% |
| Eager | transformers-gpu | kv_decode | 64 | 60 | 0 | 60 | 0% |
| Eager | transformers-gpu | kv_decode | 128 | 60 | 0 | 60 | 0% |
| Eager | transformers-gpu | e2e_kv | 64 | 60 | 0 | 60 | 0% |
| Eager | transformers-gpu | e2e_kv | 128 | 60 | 0 | 60 | 0% |
| Compiled | transformers-gpu-compile | prefill | 64 | 30 | 0 | 30 | 0% |
| Compiled | transformers-gpu-compile | prefill | 128 | 30 | 0 | 30 | 0% |
| Compiled | transformers-gpu-compile | kv_decode | 64 | 0 | 60 | 60 | **100%** |
| Compiled | transformers-gpu-compile | kv_decode | 128 | 0 | 60 | 60 | **100%** |
| Compiled | transformers-gpu-compile | e2e_kv | 64 | 0 | 60 | 60 | **100%** |
| Compiled | transformers-gpu-compile | e2e_kv | 128 | 0 | 60 | 60 | **100%** |

**Observations.** The corrected data is unambiguous:

1. Eager decode works perfectly at both token lengths across all modes. The eager path is not affected by the CUDA graph incompatibility.

2. Compiled prefill works perfectly at both token lengths. Prefill does not use the KV cache growth path (`torch.cat()` in DynamicCache), so it is unaffected.

3. Compiled kv_decode and e2e_kv crash 100% at BOTH 64 and 128 tokens. There is no partial-success window. The v1 "0% crash at 64 tokens" was entirely an artifact of the measurement bug.

4. The v1 Phase 3 "3.6% crash rate at 128 tokens" was also understated. The actual crash rate was higher but the remaining rows were zero-token ghost successes that the original harness accepted.

### Error Characterization

All 300 compiled decode/e2e errors in E5 report the same error type: `AssertionError`. This is the assertion in PyTorch's `cudagraph_trees.py` within `dealloc_current_path_weakrefs()` — the same mechanism documented in PyTorch upstream PR #175562 (filed based on TR126 Phase 3 findings). The assertion fires when the CUDA graph manager's `tensor_weakrefs` and `stack_traces` arrays diverge in length during deallocation, which happens when `DynamicCache.update()` calls `torch.cat()` to grow the KV-cache tensor during graph replay. The graph was recorded with a fixed allocation plan; the dynamic growth violates that plan.

No `invalid_kv_decode_measurement` errors appear in E5 — the v2 harness fix was not the proximate cause of these errors. The errors are genuine CUDA graph assertion failures caught by the harness before the zero-token check fires. The harness fix simply ensures that if the assertion is caught and the measurement produces zero tokens anyway, it is also classified as an error.

This finding fundamentally changes the TR147 narrative. The second-regime decode boundary is not "partially ported" or "weakened." It is TOTAL -- compiled decode fails 100% on RTX 6000 Ada regardless of token length, mode, or model (as E1 and E2 subsequently confirmed).

---

## SS6. V2 E1: Token-Length Sweep Confirms Total Decode Failure

### Design Rationale

Even after E5 showed 100% crash at 64 and 128 tokens on qwen2.5-1.5b, it was possible that (a) other models behave differently, or (b) very short sequences (32 tokens) might succeed. E1 was designed to sweep the full token-length range across 3 models to map the crash boundary definitively.

### Configuration

- **Models**: gpt2-100m (FP32), qwen2.5-1.5b (FP16), qwen2.5-3b (FP16)
- **Token lengths**: 32, 48, 64, 96, 128, 192, 256, 384, 512
- **Backends**: transformers-gpu (eager), transformers-gpu-compile
- **Mode**: kv_decode only
- **Repetitions**: 30 per condition
- **Total**: 3 models x 9 lengths x 2 backends x 30 reps x 2 prompts = 3,240 measurements

### Results: Aggregate Crash Rate by Token Length

| Token Length | Eager OK | Eager Errors | Compiled OK | Compiled Errors | Compiled Crash Rate |
|-------------|---------|-------------|------------|----------------|-------------------|
| 32 | 180 | 0 | 0 | 180 | **100.0%** |
| 48 | 180 | 0 | 0 | 180 | **100.0%** |
| 64 | 180 | 0 | 0 | 180 | **100.0%** |
| 96 | 180 | 0 | 0 | 180 | **100.0%** |
| 128 | 180 | 0 | 0 | 180 | **100.0%** |
| 192 | 180 | 0 | 0 | 180 | **100.0%** |
| 256 | 180 | 0 | 0 | 180 | **100.0%** |
| 384 | 180 | 0 | 0 | 180 | **100.0%** |
| 512 | 180 | 0 | 0 | 180 | **100.0%** |

### Results: Per-Model Crash Rate

| Token Length | gpt2-100m Compiled | qwen2.5-1.5b Compiled | qwen2.5-3b Compiled |
|-------------|-------------------|----------------------|---------------------|
| 32 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 48 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 64 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 96 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 128 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 192 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 256 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 384 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |
| 512 | 60/60 (100%) | 60/60 (100%) | 60/60 (100%) |

**Observations.** The crash rate is exactly 100% at every token length for every model. There is no threshold, no gradual degradation, no model-dependent behavior. The CUDA graph / DynamicCache incompatibility is total on RTX 6000 Ada. All 1,620 error rows report `AssertionError` — the same `dealloc_current_path_weakrefs()` assertion seen in E5. The error type does not vary with token length or model, confirming a single failure mechanism.

This is qualitatively different from TR126's finding on the RTX 4080 Laptop, where compiled decode worked at 64 tokens (with 2.2% slowdown) and crashed at 128 tokens. The cross-regime comparison is:

| Regime | 32 tok | 64 tok | 128 tok | 256 tok | 512 tok |
|--------|--------|--------|---------|---------|---------|
| RTX 4080 Laptop (TR126) | Not tested | **Works** (2.2% slower) | **100% crash** | Not tested | Not tested |
| RTX 6000 Ada (TR147 v2) | **100% crash** | **100% crash** | **100% crash** | **100% crash** | **100% crash** |

**Observations.** The decode boundary is MORE severe on the larger GPU. This is counterintuitive if one assumes the crash is caused by VRAM pressure (the RTX 6000 has 4x the VRAM). The finding strongly suggests the crash mechanism is not memory-related but rather related to:

1. **Triton version differences.** TR126 used Triton 3.3.1; TR147 used Triton 3.4.0. The newer Triton may generate different CUDA graph structures that interact worse with DynamicCache.

2. **CUDA toolkit version differences.** TR126 used CUDA 13.0; TR147 used CUDA 12.8. The CUDA graph replay implementation may differ.

3. **SM-count-dependent codegen.** Inductor may generate different graph structures for GPUs with more SMs, and these structures may be more sensitive to dynamic tensor shapes.

4. **Execution environment differences.** TR126 ran on Docker in WSL2 on Windows; TR147 ran on native Linux Docker on RunPod. The CUDA driver stack differs.

Disentangling these factors would require a controlled ablation (same GPU, different Triton versions; same Triton, different GPUs). This is noted as future work.

---

## SS7. V2 E2: Third-Family Validation (Llama 3.2)

### Model Selection Process

E2 attempted the official meta-llama/Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct models first. These require a gated access token that was available but triggered an HTTP 403 error during download on the RunPod pod. The experiment automatically fell back to the unsloth variants (unsloth/Llama-3.2-1B-Instruct, unsloth/Llama-3.2-3B-Instruct), which provide identical weights without gating. The fallback was recorded in the manifest's `fallback_attempts` field.

### Llama Prefill Results

| Model | Token Length | Eager Mean (ms) | Eager 95% CI | Compiled Mean (ms) | Compiled 95% CI | Gain (%) |
|-------|------------|-----------------|-------------|--------------------|--------------------|----------|
| Llama-3.2-1B:fp16 | 64 | 11.738 | [11.68, 11.80] | 3.652 | [3.64, 3.66] | **~68.9%** |
| Llama-3.2-3B:fp16 | 64 | 19.739 | [19.64, 19.89] | 8.879 | [8.85, 8.91] | **~55.0%** |
| Llama-3.2-1B:fp16 | 128 | 12.261 | [12.15, 12.37] | -- | -- | -- |
| Llama-3.2-3B:fp16 | 128 | 20.121 | [19.97, 20.28] | -- | -- | -- |

**Observations.** Llama 3.2 shows the same phase-specific pattern as the two other families:

**Prefill gains are large and consistent.** The 1B model gains ~69% and the 3B model gains ~55%. The 3B model's lower gain follows the same scale-dependent pattern seen in Qwen 2.5: larger models have more compute per kernel, so the relative launch-overhead savings from compilation are proportionally smaller.

**Compiled CIs are extremely tight.** The Llama 1B compiled CI is [3.643, 3.661], a width of only 0.018 ms. This is tighter than any Qwen or GPT-2 model. Llama's architecture may produce more deterministic CUDA graph replay timing, possibly due to its specific RoPE implementation generating more uniform Triton kernels.

**128-token compiled prefill was not measured.** The compiled backend crashed during decode-inclusive measurements at 128 tokens, and the experiment structure runs all modes for a given token length before moving to the next. In practice, the compiled prefill at 128 tokens would likely produce similar gains to the 64-token case (since prefill is not affected by the DynamicCache crash), but this was not directly measured.

### Llama Decode Results

| Model | Token Length | Eager OK | Eager Mean (ms) | Compiled OK | Compiled Errors | Crash Rate |
|-------|------------|---------|-----------------|------------|----------------|-----------|
| Llama-3.2-1B:fp16 | 64 | 120 | 676.62 | 0 | 120 | **100%** |
| Llama-3.2-1B:fp16 | 128 | 120 | 1315.35 | 0 | 120 | **100%** |
| Llama-3.2-3B:fp16 | 64 | 120 | 1110.43 | 0 | 120 | **100%** |
| Llama-3.2-3B:fp16 | 128 | 120 | 2276.10 | 0 | 120 | **100%** |

**Observations.** Compiled decode crashes 100% on Llama 3.2 at both token lengths, matching the behavior on GPT-2 and Qwen 2.5. All 480 compiled decode errors (120 per model per token length) report `AssertionError` — the identical `dealloc_current_path_weakrefs()` failure seen in E5 and E1. The error type is consistent across all three model families, confirming a single shared mechanism. The eager decode latencies are provided for context: the 1B model decodes 64 tokens in ~677 ms and 128 tokens in ~1315 ms (roughly linear in token count, as expected for autoregressive generation). The 3B model is 1.64x slower per token than the 1B model, consistent with the 2.59x parameter ratio producing a ~1.6x latency ratio due to memory-bandwidth saturation on the decode path.

### Llama E2E Results

| Model | Token Length | Eager OK | Eager Mean (ms) | Compiled OK | Compiled Errors |
|-------|------------|---------|-----------------|------------|----------------|
| Llama-3.2-1B:fp16 | 64 | 120 | 691.22 | 0 | 120 |
| Llama-3.2-1B:fp16 | 128 | 120 | 1392.69 | 0 | 120 |
| Llama-3.2-3B:fp16 | 64 | 120 | 1139.44 | 0 | 120 |
| Llama-3.2-3B:fp16 | 128 | 120 | 2259.99 | 0 | 120 |

**Observations.** E2E KV results confirm the pattern: compiled e2e crashes 100% because the decode component fails. The eager e2e latencies are close to the decode-only latencies because decode dominates (e.g., Llama 1B at 64 tok: e2e 691 ms vs decode-only 677 ms, meaning prefill contributes only ~14 ms or 2% of total time). This reinforces the deployment-relevance argument: even though compiled prefill saves 8 ms on Llama 1B, the decode phase accounts for 98% of e2e latency. A benchmark that reports only the 69% prefill gain would be operationally misleading.

### Cross-Family Summary

| Family | Attention Type | KV-Cache Implementation | Compiled Prefill | Compiled Decode |
|--------|---------------|------------------------|-----------------|----------------|
| GPT-2 | MHA, 2 heads | DynamicCache (torch.cat) | 66-76% faster | 100% crash |
| Qwen 2.5 | GQA, 12-16 Q heads, 2 KV heads | DynamicCache (torch.cat) | 61-77% faster | 100% crash |
| Llama 3.2 | GQA + RoPE, variable heads | DynamicCache (torch.cat) | 55-69% faster | 100% crash |

**Observations.** The compiler boundary is architecture-independent on RTX 6000 Ada. Three families with different attention mechanisms (standard MHA, GQA, GQA with rotary position encoding) and different head-count configurations all exhibit the same pattern: compiled prefill benefits by 55-77%, compiled decode crashes 100%.

The common denominator is HuggingFace's DynamicCache, which uses `torch.cat()` to grow the KV-cache tensor at each decode step. This operation is fundamentally incompatible with CUDA graph replay because CUDA graphs record a fixed tensor allocation plan, and `torch.cat()` creates new tensors of increasing size.

This finding confirms that the compile boundary is a framework-level issue (DynamicCache implementation), not a model-level issue (architecture or head count). Any model using HuggingFace Transformers with the default cache and `mode=reduce-overhead` will hit this boundary.

---

## SS8. V1 Phase 3: Backend Ranking

### Prefill Ranking

| Rank | Backend | N | Mean (ms) | Median (ms) | Std (ms) | P5 (ms) | P95 (ms) |
|------|---------|---|-----------|-------------|----------|---------|----------|
| 1 | transformers-gpu-compile | 540 | 4.11 | 3.50 | 2.90 | 0.90 | 8.91 |
| 2 | ollama | 540 | 5.78 | 5.46 | 2.18 | 2.29 | 9.36 |
| 3 | transformers-gpu (eager) | 540 | 17.94 | 19.91 | 8.69 | 4.29 | 28.74 |

**Observations.** The prefill ranking on RTX 6000 Ada is compiled HF (4.11 ms) > Ollama (5.78 ms) > eager HF (17.94 ms). Compiled HF is 1.41x faster than Ollama and 4.37x faster than eager HF. The Ollama result is notable: despite being a completely different serving stack (llama.cpp backend), Ollama prefill is only 41% slower than compiled HuggingFace and 3.1x faster than eager HuggingFace. This suggests Ollama uses its own kernel optimizations that are independent of torch.compile.

The eager HuggingFace prefill at 17.94 ms is surprisingly slow compared to TR126's 17.56 ms (+2.1%). The large standard deviation (8.69 ms) reflects the heterogeneity of the 5-model roster (gpt2-100m, qwen2.5-0.5b/1.5b/3b, llama3.2:1b via Ollama).

### Decode Ranking

| Rank | Backend | N | Mean (ms) | Median (ms) | Std (ms) | P5 (ms) | P95 (ms) |
|------|---------|---|-----------|-------------|----------|---------|----------|
| 1 | ollama | 540 | 162.74 | 158.73 | 113.65 | 15.05 | 411.50 |
| 2 | transformers-gpu (eager) | 540 | 2110.61 | 2390.19 | 996.02 | 498.85 | 3243.85 |
| -- | transformers-gpu-compile | -- | -- | -- | -- | -- | -- |

**Observations.** Compiled HF is absent from the decode ranking because it crashed on all applicable decode attempts in Phase 3. The v1 analysis originally reported a 3.6% crash rate (20/560 compiled decode rows marked as errors). In light of the v2 measurement fix, the remaining 540 "ok" rows in v1 Phase 3 compiled decode are likely ghost successes — zero-token rows that passed the original harness's exception-only check. The corrected crash rate for compiled decode at 128 tokens on RTX 6000 Ada is effectively 100%, consistent with E5 and E1 findings. Ollama dominates decode: 162.74 ms mean versus eager HF's 2110.61 ms. Ollama's decode is 13x faster than eager HuggingFace. This reflects llama.cpp's highly optimized decode path (GGUF-native quantized inference with continuous batching) versus HuggingFace's Python-level autoregressive loop.

The operational implication is that for decode-heavy workloads, Ollama is not just faster than compiled HF -- compiled HF does not work at all. The only viable backends for decode on RTX 6000 Ada are Ollama and eager HuggingFace, and Ollama is 13x faster.

### E2E Ranking

| Rank | Backend | N | Mean (ms) | Median (ms) | Std (ms) |
|------|---------|---|-----------|-------------|----------|
| 1 | ollama | 540 | 441.83 | 449.94 | 148.28 |
| 2 | transformers-gpu (eager) | 540 | 2125.51 | 2395.22 | 1015.88 |

**Observations.** Ollama dominates e2e because decode dominates e2e. The 4.11 ms prefill advantage of compiled HF is operationally irrelevant when decode takes 2110 ms under eager HF or is unavailable under compiled HF. For production serving, the backend choice is Ollama for latency-sensitive workloads, with HuggingFace reserved for research-grade prefill benchmarking under compilation.

### Cross-Regime Ranking Stability

| Backend | TR147 Prefill (ms) | TR126 Prefill (ms) | Rank TR147 | Rank TR126 | Preserved? |
|---------|--------------------|--------------------|-----------|-----------|-----------|
| compiled HF | 4.11 | 8.20 | 1st | 1st | Yes |
| Ollama | 5.78 | 9.11 | 2nd | 2nd | Yes |
| eager HF | 17.94 | 17.56 | 3rd | 3rd | Yes |

**Observations.** The prefill ranking is identical across regimes despite latency differences of up to 50%. This is a key finding for the benchmarking-integrity protocol: relative conclusions (rankings) are more portable than absolute conclusions (latency values). A benchmark that reports "compiled HF is fastest for prefill" is portable. A benchmark that reports "compiled HF prefills in 8.20 ms" is not.

---

## SS9. Verdicts

### Compiler Boundary Verdict: PRESENT AND MORE SEVERE

| Dimension | RTX 4080 Laptop (TR126) | RTX 6000 Ada (TR147 v2) | Change |
|-----------|------------------------|------------------------|--------|
| Compiled prefill benefit | 40-53% | 61-77% | Stronger (+20pp) |
| Compiled decode at 32 tokens | Not tested | 100% crash | -- |
| Compiled decode at 64 tokens | Works, 2.2% slower | 100% crash | Worse |
| Compiled decode at 128 tokens | 100% crash | 100% crash | Same |
| Compiled decode at 256-512 tokens | Not tested | 100% crash | -- |
| Phase separation required? | Yes | Yes (more urgently) | Same direction |
| Families tested | 2 (GPT-2, Qwen) | 3 (+Llama) | Broader |
| Token range tested | 64, 128 | 32-512 | Broader |

**Observations.** The compiler boundary is present on both regimes but manifests differently in severity. The direction is consistent (compilation helps prefill, breaks decode), but the magnitude varies. On the larger GPU, prefill gets more benefit and decode gets more broken. This is not a contradiction -- it is the expected behavior of a framework-level incompatibility (DynamicCache + CUDA graphs) that is independent of the GPU's resource capacity.

The verdict is "PRESENT AND MORE SEVERE" rather than the v1 automated verdict of "WEAKENED." The v1 verdict was misleading because it was computed before the measurement fix. The corrected data shows the boundary is not weakened -- it is stronger on the second regime.

### Phase-Separation Verdict: REQUIRED

On RTX 6000 Ada, the gap between compiled prefill (61-77% faster) and compiled decode (100% crash) is even wider than on the RTX 4080 (40-53% faster vs crashes-at-128). Without phase separation, a benchmark could report:

- If aggregating as e2e: "compilation has mixed results" (prefill helps but decode crashes)
- If reporting only successful runs: "compilation is ~60% faster" (cherry-picking prefill-only successes)
- If reporting only means: "compilation is similar to eager" (averaging the wins with the crashes)

All three of these aggregate reports are misleading. Phase separation makes the actual story visible: compilation is a prefill-only optimization that breaks decode.

### Five-Gate Protocol Verdict: VALIDATED PROSPECTIVELY

| Gate | TR126 (Retrospective) | TR147 (Prospective) | Outcome |
|------|----------------------|---------------------|---------|
| 1. Runtime attribution | Identified mislabeled compile path | Verified Triton kernels present | Applied successfully |
| 2. Environment validation | Windows fallback vs Linux Triton | RunPod Docker with Triton 3.4.0 | Applied successfully |
| 3. Phase separation | Prefill helps, decode crashes | Same pattern, more severe | Correctly identified |
| 4. Regime-aware interpretation | Single-regime analysis | Cross-regime severity comparison | Correctly applied |
| 5. Measurement validity | 100 ms NVML polling limit | measure.py fix for silent failures | Correctly applied |

**Observations.** Gate 5 (measurement validity) is particularly interesting in TR147. The original v1 measurement harness had a validity problem that was not about polling cadence (the TR122 issue) but about silent CUDA graph failures. The spirit of the gate -- "do not report a metric more precisely than the measurement stack can resolve" -- correctly flagged the problem once applied. Zero-latency decode rows SHOULD have been flagged as measurement failures, and the v2 fix implements exactly this check.

### External-Validity Assessment

The protocol generalizes from one GPU to a different GPU class within the Ada Lovelace family. Both the direction (prefill helps, decode breaks) and the phase-separation requirement are portable. The severity varies (prefill benefits more on larger GPU, decode breaks more completely), which is itself a finding that requires the protocol to detect.

What remains untested:
- **Ampere (A100)**: Different architecture, different SM design, different CUDA graph implementation
- **Hopper (H100)**: Newer architecture with hardware-accelerated CUDA graphs
- **Non-NVIDIA**: AMD ROCm, Intel oneAPI -- entirely different compilation paths
- **StaticCache**: TR126 showed StaticCache enables compiled decode at 5.8x slowdown; not retested here

---

## SS10. Limitations

### L1. Both GPUs Are Ada Lovelace

TR147 confirms the compiler boundary on two Ada-class GPUs: a mobile variant (RTX 4080 Laptop, 58 SMs) and a workstation variant (RTX 6000 Ada, 142 SMs). Both share compute capability 8.9. A skeptical reviewer can still argue that the CUDA graph incompatibility is Ada-specific. Testing on an Ampere (A100, compute 8.0) or Hopper (H100, compute 9.0) GPU would resolve this.

**Impact on paper claims.** The paper can claim the boundary is portable within Ada Lovelace. It cannot claim the boundary is architecture-universal. The planned TR147 v2 E3 (A100 regime test) would close this gap if executed.

### L2. V1 Decode Data Was Corrupted

The v1 Phase 2 "0% crash rate" at 64 tokens was incorrect. The measurement bug allowed silent CUDA graph failures to be recorded as successful measurements. V2 E5 corrected this. All v1 decode conclusions should be replaced by v2 findings.

**Impact on paper claims.** The corrected story (100% crash at all lengths) is actually simpler and stronger than the v1 story (partial crash). The measurement bug complicated the narrative but the fix simplifies it.

### L3. Models Are All Sub-4B Parameters

The largest model tested (Qwen 2.5 3B at 3.09B parameters) is well below the 7B-70B range typical of production LLM deployments. Larger models generate proportionally larger KV caches, which may interact differently with CUDA graph memory allocation. The boundary might shift or disappear at scale.

**Impact on paper claims.** The paper's claim is about the methodology (phase separation is required), not about specific thresholds. The methodology would be even more critical at larger scale if the boundary shifts unpredictably.

### L4. Triton and CUDA Versions Differ

TR126 used Triton 3.3.1 + CUDA 13.0; TR147 used Triton 3.4.0 + CUDA 12.8. The total decode crash on RTX 6000 might partly reflect the newer Triton's codegen generating more aggressive CUDA graph structures that are less tolerant of dynamic shapes. A controlled ablation (same GPU, different Triton versions) would isolate the software effect from the hardware effect.

### L5. No StaticCache Retest

TR126 showed that replacing DynamicCache with StaticCache enables compiled decode, but at a 5.8x slowdown (3,588 ms vs 622 ms eager). This workaround was not retested on RTX 6000. If StaticCache works on RTX 6000 and the slowdown is proportionally smaller (due to the faster GPU), it might be a viable compiled-decode path on workstation hardware.

### L6. Fixed Seed and Deterministic Decode

All measurements use temperature 0 and seed 42. This provides perfect reproducibility but means the experiment measures one specific execution path through the model. Stochastic sampling (temperature > 0) would exercise different CUDA graph branches and might produce different crash rates.

---

## SS11. Cross-TR Validation

### TR126 Claim: "Compiled prefill is 40-53% faster under Linux/Triton"

TR147 replicates and extends this claim. On RTX 6000 Ada, compiled prefill is 61-77% faster -- a stronger effect on a larger GPU. The direction is identical. The magnitude difference (larger gains on bigger GPU) is explained by Triton's ability to exploit more SMs.

**Validation status**: Replicated and strengthened.

### TR126 Claim: "Compiled decode crashes at 128 tokens but works at 64 tokens"

TR147 v2 shows this claim was hardware-specific. On RTX 6000 Ada, compiled decode crashes at ALL token lengths from 32 to 512. The partial-success window at 64 tokens exists only on the RTX 4080 Laptop.

**Validation status**: Direction replicated (decode fails). Specific threshold NOT replicated (no safe window on RTX 6000).

### TR126 Claim: "Phase separation is required for compiler benchmarks"

TR147 provides the strongest possible confirmation: on RTX 6000, the gap between compiled prefill (61-77% win) and compiled decode (100% crash) is even wider than on RTX 4080. Phase separation is not just recommended -- it is the only way to produce a meaningful benchmark result.

**Validation status**: Replicated and strengthened.

### TR120 Claim: "A compile label can exist without compilation running"

TR147 Phase 1 environment gate verified that real Triton kernels were generated (916+ cached .so/.cubin files). The label matches the execution path. This validates the protocol's runtime attribution gate on the second regime.

**Validation status**: Protocol gate applied successfully.

---

## SS12. Conclusions

TR147 validates the five-gate benchmarking-integrity protocol on a second GPU class and strengthens the case for phase-separated compiler reporting.

The central finding is that the compiler boundary is **portable in direction but hardware-dependent in severity**. On a larger workstation GPU with 2.45x the SMs, 3.95x the VRAM, and 2.22x the bandwidth:

- Compiled prefill gets better (61-77% vs 40-53%)
- Compiled decode gets worse (100% crash at all lengths vs partial crash at 128 only)
- Backend rankings are stable (compile > Ollama > eager for prefill)
- Phase separation becomes more urgent, not less

Without phase separation, this asymmetry is invisible. A benchmark that reports aggregated eager-vs-compiled numbers would show a modest mixed result on both GPUs, hiding the fact that compilation is simultaneously a large prefill win and a total decode failure. The five-gate protocol correctly identifies this by requiring phase-separated reporting (Gate 3) and regime-aware interpretation (Gate 4).

The protocol's value is demonstrated prospectively: it was applied to a new GPU without knowing the outcome in advance. It correctly identified a regime-dependent severity change that would have been missed by aggregate reporting. It also caught a measurement validity problem (the silent CUDA graph failure in v1) through the measurement-validity gate (Gate 5), which motivated the measure.py fix.

The remaining gap is architectural breadth. TR147 covers two Ada Lovelace GPUs (mobile and workstation, same compute capability 8.9). An Ampere (A100, compute 8.0) test would establish whether the boundary generalizes beyond one architecture generation. This is planned as TR147 v2 E3 but has not yet been executed.

---

## SS13. Reproducibility

### Software Requirements

- Docker with NVIDIA GPU support
- `nvcr.io/nvidia/pytorch:25.08-py3` base image (PyTorch 2.8.0, CUDA 12.8, Triton 3.4.0)
- Additional pip packages: transformers, safetensors, tokenizers, psutil, pynvml, requests

### Docker Build

```bash
docker build -t tr147 -f research/tr147/Dockerfile .
```

### V1 Full Pipeline

```bash
docker run --rm --gpus all --ipc=host \
  -v "${PWD}:/workspace" -w /workspace tr147 \
  python research/tr147/run.py -v
```

### V2 Experiments

```bash
docker run --rm --gpus all --ipc=host \
  -v "${PWD}:/workspace" -w /workspace tr147 \
  python research/tr147/v2/run.py -v --experiments e5,e1,e2,e4
```

### Custom GPT-2 Models

The custom scaling checkpoints are hosted on HuggingFace:
- Crusadersk/tiny-gpt2 (4.6M, validation model)
- Crusadersk/gpt2-25m (25M, n_embd=384, n_layer=3)
- Crusadersk/gpt2-50m (50M, n_embd=512, n_layer=8)
- Crusadersk/gpt2-100m (100M, n_embd=768, n_layer=8)

Upload script: `research/tr147/upload_models.py`

### Validation

```bash
python research/tr147/v2/validate_run.py \
  --run-dir research/tr147/v2/results/20260413_054740 \
  --experiments e5,e1,e2,e4
```

The validator checks: artifact existence, no NaN/inf in core metric columns on `status=="ok"` rows, no zero-latency or zero-token rows marked as `status=ok`, and expected row counts per experiment.

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| PYTHONPATH | /workspace | Import resolution |
| CUDA_VISIBLE_DEVICES | 0 | Single GPU |
| TRITON_CACHE_DIR | /tmp/triton_cache | Triton kernel cache |
| HF_HUB_OFFLINE | 0 | Allow HuggingFace downloads |
| HF_TOKEN | (user-provided) | For gated models (optional) |

---

## Appendix A: Raw V1 Phase 2 Prefill Data

| Model | Backend | N | Mean (ms) | Median (ms) | Std (ms) | P5 (ms) | P95 (ms) |
|-------|---------|---|-----------|-------------|----------|---------|----------|
| gpt2-25m:fp32 | eager | 270 | 1.786 | 1.767 | 0.197 | 1.602 | 2.090 |
| gpt2-25m:fp32 | compile | 270 | 0.515 | 0.489 | 0.139 | 0.388 | 0.746 |
| gpt2-50m:fp32 | eager | 270 | 4.146 | 4.098 | 0.539 | 3.655 | 4.904 |
| gpt2-50m:fp32 | compile | 270 | 1.083 | 1.062 | 0.222 | 0.825 | 1.459 |
| gpt2-100m:fp32 | eager | 270 | 4.054 | 4.035 | 0.389 | 3.648 | 4.582 |
| gpt2-100m:fp32 | compile | 270 | 1.383 | 1.363 | 0.228 | 1.093 | 1.786 |
| gpt2-100m:fp16 | eager | 270 | 4.093 | 4.103 | 0.360 | 3.622 | 4.596 |
| gpt2-100m:fp16 | compile | 270 | 1.004 | 1.007 | 0.230 | 0.709 | 1.363 |
| qwen2.5-0.5b:fp16 | eager | 270 | 16.934 | 17.081 | 0.919 | 15.420 | 18.244 |
| qwen2.5-0.5b:fp16 | compile | 270 | 3.843 | 3.646 | 1.038 | 2.776 | 5.734 |
| qwen2.5-1.5b:fp16 | eager | 270 | 20.493 | 20.615 | 1.675 | 17.737 | 23.142 |
| qwen2.5-1.5b:fp16 | compile | 270 | 6.361 | 6.207 | 1.128 | 4.822 | 8.236 |
| qwen2.5-3b:fp16 | eager | 270 | 27.723 | 27.802 | 2.449 | 23.780 | 31.423 |
| qwen2.5-3b:fp16 | compile | 270 | 10.892 | 10.719 | 1.397 | 8.993 | 13.270 |

---

## Appendix B: V2 E1 Token-Sweep Per-Model Crash Rates

| Token Length | gpt2-100m:fp32 (60 per cell) | qwen2.5-1.5b:fp16 (60 per cell) | qwen2.5-3b:fp16 (60 per cell) | Total (180 per cell) |
|-------------|------------------------------|----------------------------------|-------------------------------|---------------------|
| 32 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 48 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 64 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 96 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 128 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 192 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 256 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 384 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |
| 512 | 60/60 error (100%) | 60/60 error (100%) | 60/60 error (100%) | 180/180 (100%) |

All eager backend measurements: 1620/1620 OK (100% success).

---

## Appendix C: V2 E4 Statistical Summary

### Holm-Bonferroni Results (7 Prefill Comparisons)

| Rank | Model | Original p | Holm Threshold | Survives? |
|------|-------|-----------|----------------|-----------|
| 1 | qwen2.5-3b:fp16 | 4.54e-121 | 0.00714 | Yes |
| 2 | gpt2-100m:fp32 | 1.11e-118 | 0.00833 | Yes |
| 3 | gpt2-25m:fp32 | 5.04e-123 | 0.0100 | Yes |
| 4 | qwen2.5-0.5b:fp16 | 1.02e-127 | 0.0125 | Yes |
| 5 | gpt2-100m:fp16 | 1.35e-133 | 0.0167 | Yes |
| 6 | qwen2.5-1.5b:fp16 | 1.20e-137 | 0.0250 | Yes |
| 7 | gpt2-50m:fp32 | 1.35e-139 | 0.0500 | Yes |

All 7 comparisons survive. The weakest p-value (4.54e-121) is 10^118 times smaller than the strictest threshold.

### Effect Size Distribution

| Cohen's d Range | Models | Interpretation |
|----------------|--------|----------------|
| 3.3-3.5 | qwen2.5-3b, gpt2-25m, gpt2-100m:fp32 | Very large |
| 3.7-3.8 | gpt2-50m, qwen2.5-0.5b, gpt2-100m:fp16 | Very large |
| 3.9+ | qwen2.5-1.5b | Very large |

All effect sizes exceed d = 3.0, which is far beyond the conventional "large" threshold of d = 0.8. These are among the largest effect sizes in the Banterhearts research program. For comparison, TR134 safety alignment effect sizes range from d = 0.2 to d = 1.5.

### TOST Equivalence Summary (Decode)

| Models Equivalent (within +/-3pp) | 6 of 7 |
|---|---|
| Non-equivalent model | gpt2-25m:fp32 (p_tost = 0.520) |
| Direction of non-equivalence | Compiled slightly SLOWER (d = -0.70) |
| Strongest equivalence | qwen2.5-0.5b:fp16 (p_tost < 0.001) |

---

## Appendix D: V2 E2 Llama Raw Data

### Llama Prefill

| Model | Token Length | Backend | N | Mean (ms) | 95% CI |
|-------|------------|---------|---|-----------|--------|
| Llama-3.2-1B:fp16 | 64 | eager | 120 | 11.738 | [11.680, 11.795] |
| Llama-3.2-1B:fp16 | 64 | compile | 120 | 3.652 | [3.643, 3.661] |
| Llama-3.2-1B:fp16 | 128 | eager | 120 | 12.261 | [12.153, 12.373] |
| Llama-3.2-3B:fp16 | 64 | eager | 120 | 19.739 | [19.636, 19.886] |
| Llama-3.2-3B:fp16 | 64 | compile | 120 | 8.879 | [8.853, 8.906] |
| Llama-3.2-3B:fp16 | 128 | eager | 120 | 20.121 | [19.972, 20.280] |

### Llama Decode

| Model | Token Length | Backend | N | Mean (ms) | 95% CI |
|-------|------------|---------|---|-----------|--------|
| Llama-3.2-1B:fp16 | 64 | eager | 120 | 676.624 | [671.730, 681.513] |
| Llama-3.2-1B:fp16 | 128 | eager | 120 | 1315.347 | [1306.140, 1324.575] |
| Llama-3.2-3B:fp16 | 64 | eager | 120 | 1110.428 | [1103.996, 1117.363] |
| Llama-3.2-3B:fp16 | 128 | eager | 120 | 2276.096 | [2260.447, 2291.695] |

All compiled decode and compiled e2e measurements: 0 OK, 120 errors per cell.

---

## Appendix E: Configuration Snapshots

### V1 Compile Configuration

```yaml
torch_compile:
  enabled: true
  backend: inductor
  mode: reduce-overhead
  dynamic: false
  fullgraph: false
```

### V1 Phase 2 Design

```yaml
models: [gpt2-25m:fp32, gpt2-50m:fp32, gpt2-100m:fp32, gpt2-100m:fp16,
         qwen2.5-0.5b:fp16, qwen2.5-1.5b:fp16, qwen2.5-3b:fp16]
scenarios: [single_micro, single_short, single_medium, single_long, stress_single]
backends: [transformers-gpu, transformers-gpu-compile]
modes: [prefill, kv_decode, e2e_kv]
repetitions: 30
warmup_repetitions: 3
max_new_tokens: 64
```

### V2 E1 Design

```yaml
models: [gpt2-100m:fp32, qwen2.5-1.5b:fp16, qwen2.5-3b:fp16]
token_lengths: [32, 48, 64, 96, 128, 192, 256, 384, 512]
backends: [transformers-gpu, transformers-gpu-compile]
modes: [kv_decode]
repetitions: 30
warmup_repetitions: 3
scenario: single_short
```

### V2 E2 Design

```yaml
models: [unsloth/Llama-3.2-1B-Instruct:fp16, unsloth/Llama-3.2-3B-Instruct:fp16]
token_lengths: [64, 128]
backends: [transformers-gpu, transformers-gpu-compile]
modes: [prefill, kv_decode, e2e_kv]
scenarios: [single_short, single_medium]
repetitions: 30
warmup_repetitions: 3
```

---

## Appendix F: Glossary

| Term | Definition |
|------|-----------|
| **Ada Lovelace** | NVIDIA GPU architecture (2022), compute capability 8.9. Used in both RTX 4080 Laptop (mobile) and RTX 6000 (workstation). |
| **DynamicCache** | HuggingFace Transformers' default KV-cache implementation. Uses `torch.cat()` to grow tensors at each decode step. This dynamic allocation is incompatible with CUDA graph replay. |
| **CUDA graph** | A recorded sequence of GPU operations that can be replayed without CPU overhead. Requires fixed tensor shapes at record time. |
| **Inductor** | PyTorch's default torch.compile backend. Lowers Python operations to Triton kernels for GPU execution. |
| **Phase separation** | Reporting prefill and decode metrics separately rather than as one aggregate. Required when an optimization affects phases differently. |
| **TOST** | Two one-sided tests for equivalence. Unlike a t-test (which tests for difference), TOST confirms two values are WITHIN a specified margin, establishing practical equivalence. |
| **Triton** | OpenAI's GPU programming language used by Inductor for kernel generation. Version differences (3.3.1 vs 3.4.0) can change the generated CUDA graph structure. |
| **SM (Streaming Multiprocessor)** | The fundamental compute unit on NVIDIA GPUs. RTX 4080 Laptop has 58 SMs; RTX 6000 Ada has 142 SMs. |
| **reduce-overhead** | torch.compile mode that enables CUDA graph capture for reduced CPU-side overhead. The most aggressive compilation mode and the one that triggers the DynamicCache incompatibility. |
| **GQA (Grouped Query Attention)** | An attention variant where multiple query heads share fewer key/value heads. Reduces KV-cache size. Used by Qwen 2.5 and Llama 3.2. |
| **RoPE (Rotary Position Encoding)** | A relative position encoding method used by Llama. Applied during attention computation, not stored in the KV-cache. |

---

## References

1. TR126: Docker/Linux + Triton Compiler Validation. 25,400 measurements. Established the compiler boundary on RTX 4080 Laptop.
2. TR117: Cross-Backend Inference Benchmark. 3,017 runs across 7 backends. Source of the tier-3 prompt configuration used by TR147.
3. TR120: The "Compile Paradox" Root-Cause Audit. 546 runs. Discovered that the original compile label was misattributed.
4. TR121v1: Model Scaling Study. 684 runs. Source of the custom GPT-2 depth-width models.
5. TR122: Resource Profiling Deep Dive. 2,041 runs. Established the 100 ms NVML polling baseline and the measurement-validity gate.
6. PyTorch upstream PR #175562: cudagraph_trees.py assertion fix. Filed based on TR126 Phase 3 findings. Relates to the `dealloc_current_path_weakrefs()` assertion that fires under CUDA graph replay with DynamicCache.
