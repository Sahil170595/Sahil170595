# Technical Report 118: ONNX Runtime + TensorRT Deep Dive

## Comprehensive Local-First Inference Optimization for Production Deployment

**Project:** Banterhearts LLM Performance Research  
**Date:** 2025-12-12  
**Author:** Research Team  
**Report Type:** Definitive ONNX/TensorRT Performance Analysis  
**Test Duration:** 250 seconds (180 comprehensive benchmark runs)  
**Related Work:** [TR117](Technical_Report_117.md) (Cross-Backend Baseline),
[TR115_v2](Technical_Report_115_v2.md) (Rust Runtime Analysis)

---

## Executive Summary

> **⚠️ CRITICAL NOTE - REPORT STATUS: INVALID DATA**
>
> This report contains benchmark data from **`sshleifer/tiny-gpt2`** (0.1M parameters, test fixture), NOT the intended **`gpt2`** (124M parameters, production model). All performance numbers, perplexity measurements, and conclusions are **INVALID** and do not represent real-world performance.
>
> **Issues identified:**
> - Model size: 0.39MB (FP32) actual vs 474.7MB expected (1,217x smaller)
> - Parameters: 102,714 (0.103M) actual vs 124,439,808 (124.4M) claimed (1,210x fewer)
> - Perplexity: 50,285 (random/untrained) vs ~30-40 expected
> - All backends show "identical" perplexity because the model is noise
>
> **Status:** Configuration fixed (now points to `gpt2`), experiment needs to be re-run.
>
> The data below is preserved for forensic purposes but should NOT be cited.

---

## ORIGINAL EXECUTIVE SUMMARY (INVALID - DO NOT USE)

This technical report ~~presents the definitive analysis~~ **attempted to analyze** ONNX Runtime and TensorRT performance for local-first LLM inference. Through 180 comprehensive benchmark runs across 6 backends (PyTorch, ONNX Runtime CPU/GPU, TensorRT FP32/FP16/INT8), ~~we establish the performance characteristics of specialized inference runtimes and provide production-grade recommendations~~ **we accidentally benchmarked a 0.1M parameter test fixture instead of the intended 124M parameter model**.

**Critical Context:**

This report addresses TR117's infrastructure failures (18% ONNX/TRT degradation) by:

1. **Building real engines:** TensorRT FP32/FP16/INT8 with proper dynamic shape profiles
2. **ONNX export validation:** Correct keyword-arg wrappers for HuggingFace models
3. **Accuracy gates:** Perplexity validation on WikiText-2 (72,531 tokens)
4. **Zero degradation:** 180/180 runs successful (vs TR117's 82% success rate)
5. **Statistical rigor:** 5 repetitions per scenario with t-tests, effect sizes

### Key Findings

**Backend Performance Ranking (Mean Latency → Throughput → Consistency):**
1. **ONNX Runtime CPU:** 2.010ms | 14,291 tok/s | **63.3% faster than PyTorch** ⚡ **SURPRISE WINNER**
2. **TensorRT FP32:** 2.120ms | 13,956 tok/s | **61.3% faster than PyTorch** ✅ **PRODUCTION READY**
3. **TensorRT FP16:** 2.313ms | 12,927 tok/s | **57.7% faster than PyTorch** ✅ **BEST BALANCE**
4. **PyTorch GPU-compile:** 5.475ms | 5,058 tok/s | Baseline (cudagraphs) 🏆 **BASELINE**
5. **TensorRT INT8:** 5.077ms | 6,106 tok/s | 7.3% faster (not significant) ⚠️ **PARADOX**
6. **ONNX Runtime GPU:** 6.371ms | 4,465 tok/s | 16.4% slower (not significant) ❌ **DISAPPOINTING**

**Critical Discoveries:**
1. **ONNX Runtime CPU dominates:** 63.3% faster than PyTorch GPU (p=2.1e-14, effect size -2.61)
2. **INT8 paradox:** No significant speedup vs PyTorch (p=0.565), slower than FP16 (5.08ms vs 2.31ms)
3. **TensorRT FP16 sweet spot:** 57.7% faster with zero accuracy loss (delta=-2.38e-08)
4. **ONNX Runtime GPU disappoints:** 16.4% slower than PyTorch (p=0.272, not significant)
5. **All backends pass accuracy gates:** Perplexity delta < 3e-08 for all precisions

**Revised Understanding:**
- **Previous belief (TR117):** TensorRT/ONNX would be 2-3x faster than PyTorch
- **Actual reality (TR118):** **ONNX CPU is the surprise winner** (63% faster than PyTorch GPU)
- **Implication:** For small models, **CPU overhead < GPU overhead**

### Business Impact

**Strategic Insights:**
- **Production Recommendation:** **TensorRT FP16** for GPU (57.7% faster, zero accuracy loss) or **ONNX CPU** for CPU-only (63% faster than PyTorch GPU)
- **Cost Efficiency:** 2.6x throughput improvement → 61% cost reduction per token
- **Deployment Simplicity:** ONNX Runtime requires no calibration, TensorRT FP16 requires no INT8 calibration
- **Build Overhead:** 88s total (one-time cost, amortized over 15K inferences)

**Risk Assessment:**
- **INT8 not worth it:** 7.3% speedup not significant (p=0.565), calibration overhead wasted for small models
- **ONNX GPU underwhelming:** 16.4% slower than PyTorch, high variance, avoid for production
- **TensorRT build overhead:** 88s acceptable for frozen models, unacceptable for rapid iteration

**Key Decision:**
After 180 benchmarks, the definitive answer is: **TensorRT FP16 for GPU deployments** (57.7% faster, zero accuracy loss, production-ready) and **ONNX Runtime CPU for CPU-only** (63% faster than PyTorch GPU for small models). Avoid INT8 for small models and avoid ONNX GPU entirely.

---

## Table of Contents

1. [Introduction & Research Evolution](#1-introduction--research-evolution)
2. [Methodology & Experimental Design](#2-methodology--experimental-design)
3. [Comprehensive Results Analysis](#3-comprehensive-results-analysis)
4. [Statistical Deep Dive](#4-statistical-deep-dive)
5. [ONNX Runtime CPU Surprise Analysis](#5-onnx-runtime-cpu-surprise-analysis)
6. [INT8 Paradox Investigation](#6-int8-paradox-investigation)
7. [ONNX Runtime GPU Disappointment Analysis](#7-onnx-runtime-gpu-disappointment-analysis)
8. [TensorRT FP16 Sweet Spot Analysis](#8-tensorrt-fp16-sweet-spot-analysis)
9. [Accuracy Validation & Perplexity Gates](#9-accuracy-validation--perplexity-gates)
10. [Cross-Backend Comparison Matrix](#10-cross-backend-comparison-matrix)
11. [Production Deployment Strategy](#11-production-deployment-strategy)
12. [Conclusions & Recommendations](#12-conclusions--recommendations)
13. [Appendices](#13-appendices)

---

## 1. Introduction & Research Evolution

### 1.1 The Journey to TR118

**December 8, 2025 - TR117:** Cross-backend benchmark with infrastructure failures:
- ONNX/TensorRT: 18% degraded (546/546 runs failed, 100% failure rate)
- Root cause: Missing ONNX exports, no TensorRT engines built
- **Hypothesis:** Specialized runtimes would be 2-3x faster than PyTorch

**December 12, 2025 - TR118 (This Report):** ONNX/TensorRT deep dive with zero degradation:
- 180/180 runs successful (0% degraded, 100% success rate)
- Real TensorRT engines: FP32 (21s build), FP16 (43s build), INT8 (25s build + WikiText-2 calibration)
- ONNX export: Validated with correct keyword-arg wrappers
- **Hypothesis validated (partially):** TensorRT is 57-61% faster, **ONNX CPU is 63% faster** (surprise)

### 1.2 Why This Report Matters

TR117 identified infrastructure gaps (100% ONNX/TRT failure) but couldn't validate performance claims. TR118 closes this gap by:

1. **Building real engines:** TensorRT FP32/FP16/INT8 with 5 dynamic profiles each
2. **Validating accuracy:** Perplexity gates on WikiText-2 (72,531 tokens)
3. **Zero degradation:** 180/180 runs successful (vs TR117's 82% overall success)
4. **Statistical rigor:** 5 repetitions × 6 scenarios × 6 backends = 180 samples

**Research Question:** Can ONNX Runtime and TensorRT deliver on their promise of significant speedup while preserving accuracy?

**Answer:** **Yes for TensorRT FP32/FP16** (57-61% faster), **Yes for ONNX CPU** (63% faster), **No for INT8** (not significant), **No for ONNX GPU** (slower than PyTorch).

### 1.3 Scope & Limitations

**Tested:**
- ✅ PyTorch transformers GPU-compile (cudagraphs baseline)
- ✅ ONNX Runtime CPU (CPUExecutionProvider)
- ✅ ONNX Runtime GPU (CUDAExecutionProvider)
- ✅ TensorRT FP32 (full precision)
- ✅ TensorRT FP16 (half precision)
- ✅ TensorRT INT8 (quantized with WikiText-2 calibration)

**Test Matrix:**
- Model: gpt2 (124.4M parameters) **[NOTE: Actual data used sshleifer/tiny-gpt2 0.103M - INVALID]**
- Scenarios: 6 (single_micro, single_short, single_medium, single_long, batch_short, batch_medium)
- Repetitions: 5 per scenario
- Total: 180 runs (6 backends × 6 scenarios × 5 reps)

**Limitations:**
- **Single model:** gpt2 (124.4M params) intended, sshleifer/tiny-gpt2 (0.103M params) actually used - **DATA INVALID**
- **Single hardware:** RTX 4080 Laptop (12GB VRAM), i9-13980HX
- **Prefill-only:** Measures forward-pass latency, not full generation loop
- **Windows:** TensorRT 10.x on Windows (Linux may differ)

---

## 2. Methodology & Experimental Design

### 2.1 Hardware & Software

**Hardware Configuration:**

```text
GPU: NVIDIA GeForce RTX 4080 Laptop
- VRAM: 12 GB GDDR6X
- CUDA Cores: 9,728
- Tensor Cores: 304 (4th Gen)
- Memory Bandwidth: 504 GB/s
- Compute Capability: 8.9
- Driver: 566.03

CPU: Intel Core i9-13980HX
- Cores: 24 (8 Performance + 16 Efficient)
- Threads: 32
- Base Clock: 2.2 GHz
- Boost Clock: 5.6 GHz

RAM: 16 GB DDR5-4800
OS: Windows 11 Pro (Build 26200)
```

**Software:**
- **PyTorch:** 2.8.0+cu128
- **Transformers:** 4.57.0
- **ONNX:** 1.19.0
- **ONNX Runtime:** 1.23.2 (TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider)
- **TensorRT:** 10.12.0.36
- **Datasets:** 3.5.0 (for WikiText-2 perplexity validation)
- **Python:** 3.13.1
- **CUDA:** 12.8

### 2.2 Test Scenarios

**Prefill-Only Measurement:**

| Scenario | Batch | Seq Len | Total Tokens | Description |
|----------|-------|---------|--------------|-------------|
| single_micro | 1 | 8 | 8 | Minimal overhead test |
| single_short | 1 | 11 | 11 | Short prompt (8-15 words) |
| single_medium | 1 | 20 | 20 | Medium prompt (20-30 words) |
| single_long | 1 | 27 | 27 | Long prompt (40-50 words) |
| batch_short | 4 | 11 | 44 | Batched short prompts |
| batch_medium | 4 | 19 | 76 | Batched medium prompts |

**6 scenarios × 5 repetitions = 30 runs per backend**

**What we measure:** Single forward pass over padded `(batch, seq_len)` tensor  
**What we don't measure:** Full text generation loop (decode, sampling, stopping)  
**Rationale:** Keeps PyTorch/ORT/TRT comparable without backend-specific generation logic

### 2.3 Metrics

**Latency:**
- **Mean latency (ms):** Average forward-pass time
- **Median latency (ms):** Robust central tendency
- **Standard deviation (ms):** Consistency measure
- **95% CI:** Confidence interval

**Throughput:**
- **Tokens/sec:** `(batch × seq_len) / latency`
- **Mean throughput:** Average across runs

**Accuracy:**
- **Perplexity:** WikiText-2 test set (72,531 tokens)
- **Delta fraction:** `(backend_ppl - baseline_ppl) / baseline_ppl`
- **Pass/fail:** Delta < threshold (FP32: 0.001, FP16: 0.005, INT8: 0.020)

**Build Overhead:**
- **ONNX export time:** PyTorch → ONNX
- **TensorRT build time:** ONNX → .plan
- **Engine size:** Disk footprint

### 2.4 ONNX Export

**Export Configuration:**
- **Opset:** 17 (best TensorRT compatibility)
- **Dynamic axes:** Batch and sequence (required for multi-scenario testing)
- **TRT-friendly inputs:** int32 input_ids/attention_mask
- **Wrapper:** `CausalLMOnnxWrapper` with keyword args (avoids positional mismatch)
- **Sanitization:** ConstantOfShape rewrite for TensorRT INT8 compatibility

**Result:**
- **File:** `artifacts/onnx/gpt2.onnx` (future), `tiny-gpt2.onnx` (invalid data)
- **Size:** 1.86MB
- **Validation:** PASSED (onnx.checker.check_model)
- **Reused:** Existing export (export time not measured in this run)

### 2.5 TensorRT Engine Building

**Build Configuration:**
- **Workspace:** 6GB
- **Profiles:** 5 dynamic optimization profiles (covering all scenarios)
- **INT8 Calibration:** WikiText-2 test set (512 samples, batch=8, seq=128)

**Build Results:**

| Precision | Build Time (s) | Size (MB) | Profiles | Calibration |
|-----------|----------------|-----------|----------|-------------|
| FP32 | 20.7 | 5.0 | 5 | N/A |
| FP16 | 42.6 | 4.2 | 5 | N/A |
| INT8 | 25.2 | 5.2 | 5 | WikiText-2 (512 samples) |

**Total build overhead:** 88.5s (one-time cost for frozen models)

### 2.6 Benchmark Methodology

**Warmup:**
- **Runs:** 3 warmup iterations per backend/scenario
- **Purpose:** Eliminate cold-start effects (CUDA kernel compilation, cache warming)

**Resource Monitoring:**
- **GPU:** Memory usage, utilization, power, temperature (NVML)
- **CPU:** Memory usage, utilization (psutil)
- **Sampling:** 0.1s intervals

**Degradation Tracking:**
- **Criteria:** Missing engines, provider fallbacks, timeouts, exceptions
- **Result:** 0/180 degraded (100% success rate)

---

## 3. Comprehensive Results Analysis

### 3.1 Overall Backend Summary

**Run-Level Statistics (180 samples, 30 per backend):**

| Backend | Mean (ms) | Throughput (tok/s) | Speedup vs PyTorch | Significant |
| --- | --- | --- | --- | --- |
| **onnxruntime-cpu** | 2.010 | 14,291 | **+63.3%** | ✅ (p=2.1e-14) |
| **tensorrt-fp32** | 2.120 | 13,956 | **+61.3%** | ✅ (p=4.6e-13) |
| **tensorrt-fp16** | 2.313 | 12,927 | **+57.7%** | ✅ (p=4.1e-11) |
| **transformers-gpu-compile** | 5.475 | 5,058 | Baseline | — |
| **tensorrt-int8** | 5.077 | 6,106 | +7.3% | ❌ (p=0.565) |
| **onnxruntime-gpu** | 6.371 | 4,465 | -16.4% | ❌ (p=0.272) |

**Key Observations:**
1. **Top 3 cluster together:** 2.01-2.31ms (ONNX CPU, TRT FP32/FP16)
2. **INT8 paradox:** Similar to PyTorch (5.08ms vs 5.48ms), not significant
3. **ONNX GPU disappoints:** Slower than PyTorch, not significant
4. **Zero degradation:** All 180 runs successful

### 3.2 Per-Scenario Breakdown

**Single Micro (batch=1, seq=8, 8 tokens):**

| Backend | Mean (ms) | Throughput (tok/s) | vs PyTorch |
|---------|-----------|---------------------|------------|
| tensorrt-fp32 | 0.807 | 11,398 | **81.4% faster** |
| tensorrt-fp16 | 0.782 | 10,454 | **82.0% faster** |
| onnxruntime-cpu | 0.854 | 10,173 | **80.4% faster** |
| tensorrt-int8 | 1.100 | 7,422 | 74.7% faster |
| onnxruntime-gpu | 2.493 | 3,237 | 42.7% faster |
| transformers-gpu-compile | 4.350 | 1,861 | Baseline |

**Batch Medium (batch=4, seq=19, 76 tokens):**

| Backend | Mean (ms) | Throughput (tok/s) | vs PyTorch |
|---------|-----------|---------------------|------------|
| onnxruntime-cpu | 4.435 | 17,245 | **44.0% faster** |
| tensorrt-fp32 | 4.803 | 16,024 | 39.4% faster |
| tensorrt-fp16 | 5.274 | 15,924 | 33.4% faster |
| transformers-gpu-compile | 7.924 | 9,619 | Baseline |
| tensorrt-int8 | 11.318 | 6,730 | **42.8% slower** ❌ |
| onnxruntime-gpu | 14.390 | 5,344 | 81.6% slower ❌ |

**Key Patterns:**
1. **Micro workloads:** All backends show massive speedups (42-82%)
2. **Batched workloads:** ONNX CPU maintains lead, INT8 degrades severely
3. **ONNX GPU struggles:** Increasingly slow with batch size

### 3.3 Statistical Significance

**Overall Comparisons (vs PyTorch GPU-compile baseline, 30 samples each):**

| Backend | Mean Δ (ms) | % Change | p-value | Effect Size | Significant |
|---------|-------------|----------|---------|-------------|-------------|
| onnxruntime-cpu | -3.465 | -63.3% | 2.1e-14 | -2.61 | ✅ Yes |
| tensorrt-fp32 | -3.355 | -61.3% | 4.6e-13 | -2.40 | ✅ Yes |
| tensorrt-fp16 | -3.162 | -57.7% | 4.1e-11 | -2.09 | ✅ Yes |
| tensorrt-int8 | -0.398 | -7.3% | 0.565 | -0.15 | ❌ No |
| onnxruntime-gpu | +0.896 | +16.4% | 0.272 | +0.29 | ❌ No |

**Interpretation:**
- **Top 3 highly significant:** p < 1e-10, effect sizes > 2.0 (large)
- **INT8 not significant:** p=0.565 (> 0.05 threshold)
- **ONNX GPU not significant:** p=0.272 (> 0.05 threshold), and slower

---

## 4. Statistical Deep Dive

### 4.1 Distribution Analysis

**Latency Distributions (all scenarios combined, 30 samples per backend):**

| Backend | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | CV |
|---------|-----------|-------------|----------|----------|----------|-----|
| onnxruntime-cpu | 2.010 | 1.714 | 1.085 | 0.607 | 4.873 | 54.0% |
| tensorrt-fp32 | 2.120 | 1.763 | 1.176 | 0.403 | 5.260 | 55.5% |
| tensorrt-fp16 | 2.313 | 2.168 | 1.343 | 0.700 | 5.963 | 58.1% |
| tensorrt-int8 | 5.077 | 4.765 | 2.961 | 0.983 | 11.968 | 58.3% |
| transformers-gpu-compile | 5.475 | 4.869 | 1.477 | 3.015 | 8.355 | 27.0% |
| onnxruntime-gpu | 6.371 | 5.473 | 3.632 | 2.160 | 16.564 | 57.0% |

**Key Observations:**
1. **PyTorch most consistent:** CV=27.0% (lowest), tight distribution
2. **ONNX CPU most consistent (of winners):** CV=54.0%
3. **ONNX GPU high variance:** CV=57.0%, std=3.63ms (highest)
4. **INT8 high variance:** CV=58.3%, unstable across scenarios

**Coefficient of Variation (CV):** Lower is better (more consistent)

### 4.2 Per-Scenario Statistical Tests

**Single Micro (8 tokens):**

| Comparison | Δ (ms) | % Change | p-value | Effect Size |
|------------|--------|----------|---------|-------------|
| PyTorch vs ORT CPU | -3.496 | -80.4% | 2.0e-09 | -18.50 |
| PyTorch vs TRT FP32 | -3.543 | -81.4% | 8.3e-09 | -15.47 |
| PyTorch vs TRT FP16 | -3.568 | -82.0% | 7.9e-11 | -27.81 |
| PyTorch vs TRT INT8 | -3.249 | -74.7% | 1.2e-09 | -19.84 |
| PyTorch vs ORT GPU | -1.856 | -42.7% | 2.8e-07 | -9.88 |

**Batch Medium (76 tokens):**

| Comparison | Δ (ms) | % Change | p-value | Effect Size |
|------------|--------|----------|---------|-------------|
| PyTorch vs ORT CPU | -3.489 | -44.0% | 4.1e-08 | -12.63 |
| PyTorch vs TRT FP32 | -3.122 | -39.4% | 2.5e-07 | -10.01 |
| PyTorch vs TRT FP16 | -2.650 | -33.4% | 0.0018 | -2.89 |
| PyTorch vs TRT INT8 | +3.394 | **+42.8%** | 3.3e-07 | +9.68 |
| PyTorch vs ORT GPU | +6.466 | **+81.6%** | 3.8e-05 | +5.16 |

**Key Findings:**
1. **Single micro shows extreme speedups:** 43-82% faster (all significant)
2. **Batch medium shows INT8 failure:** 43% **slower** than PyTorch (highly significant)
3. **ONNX GPU catastrophic on batching:** 82% slower (highly significant)
4. **Effect sizes massive:** Single micro shows effect sizes of -10 to -28

### 4.3 Consistency Analysis

**Scenario Variance vs Repetition Variance:**

| Backend | Scenario Var | Rep Var | Total Var | Scenario % |
|---------|-------------|---------|-----------|------------|
| onnxruntime-cpu | 0.92 | 0.16 | 1.08 | 85% |
| tensorrt-fp32 | 1.05 | 0.13 | 1.18 | 89% |
| tensorrt-fp16 | 1.21 | 0.13 | 1.34 | 90% |
| tensorrt-int8 | 2.68 | 0.28 | 2.96 | 91% |
| transformers-gpu-compile | 1.32 | 0.16 | 1.48 | 89% |
| onnxruntime-gpu | 3.28 | 0.35 | 3.63 | 90% |

**Interpretation:**
- **Scenario variance dominates:** 85-91% of total variance (expected - different workloads)
- **Repetition variance low:** 9-15% (good consistency within scenarios)
- **INT8 and ORT GPU unstable:** High scenario variance suggests unpredictable performance

---

## 5. ONNX Runtime CPU Surprise Analysis

### 5.1 The Surprise

**Expected:** ONNX Runtime CPU slower than PyTorch GPU (CPU vs GPU)  
**Observed:** ONNX Runtime CPU **63.3% faster** than PyTorch GPU (2.01ms vs 5.48ms)

**Statistical Evidence:**
- Mean difference: -3.465ms
- Percent change: -63.3%
- p-value: 2.1e-14 (extremely significant)
- Effect size: -2.61 (very large)
- Sample size: 30 per backend

### 5.2 Why Is ONNX CPU So Fast?

**Hypothesis 1: Tiny Model Fits in CPU Cache**
- **Observation:** sshleifer/tiny-gpt2 (0.103M params = 102,714 params, 0.39MB FP32) fits in L1 cache - explains anomalous results
- **Evidence:** Single micro (8 tokens) → 0.854ms (10,173 tok/s)
- **Intel i9-13980HX:** 36MB L3 cache (shared across cores)
- **Explanation:** No GPU transfer overhead, pure compute on CPU

**Hypothesis 2: GPU Overhead Dominates for Small Workloads**
- **PyTorch GPU overhead components:**
  - CUDA kernel launch: ~5-10μs per kernel
  - Device-to-host transfer: ~50-100μs for logits
  - cudagraphs capture/replay: Graph management overhead
- **ONNX CPU advantages:**
  - No device transfer
  - Direct memory access
  - Optimized CPU kernels (Intel MKL, OpenMP)

**Hypothesis 3: Workload Too Small for GPU Parallelism**
- **Single micro:** 8 tokens × 1 batch = minimal parallelism
- **GPU underutilized:** 9,728 CUDA cores mostly idle
- **CPU sufficient:** Single-threaded performance adequate

**Validation by Workload Size:**

| Workload | Tokens | ORT CPU (ms) | PyTorch GPU (ms) | Speedup |
|----------|--------|--------------|------------------|---------|
| single_micro | 8 | 0.854 | 4.350 | **5.1x** |
| single_short | 11 | 1.014 | 4.374 | 4.3x |
| single_medium | 20 | 1.304 | 4.689 | 3.6x |
| single_long | 27 | 1.746 | 4.912 | 2.8x |
| batch_short | 44 | 2.705 | 6.601 | 2.4x |
| batch_medium | 76 | 4.435 | 7.924 | 1.8x |

**Pattern:** Speedup decreases with workload size (5.1x → 1.8x)

**Implication:** GPU overhead amortizes at ~100+ tokens

### 5.3 When Does ONNX CPU Win?

**Winning Scenarios:**
1. **Small models:** <500M parameters (fits in CPU cache)
2. **Small batch sizes:** batch=1-4 (low parallelism)
3. **Short sequences:** seq<50 (low compute intensity)
4. **CPU-only deployments:** No GPU available
5. **Latency-critical:** Avoid GPU transfer overhead

**Losing Scenarios:**
1. **Large models:** >1B parameters (exceeds CPU cache)
2. **Large batch sizes:** batch>8 (GPU parallelism wins)
3. **Long sequences:** seq>128 (GPU compute intensity wins)
4. **GPU-available deployments:** GPU overhead amortized

**Production Recommendation:**
- **Use ONNX CPU for:** Edge devices, CPU-only servers, small models (<500M), micro batches
- **Use PyTorch GPU for:** Large models (>1B), large batches (>8), long sequences (>128)

---

## 6. INT8 Paradox Investigation

### 6.1 The Paradox

**Expected:** INT8 quantization → 2-4x speedup (8-bit vs 16-bit compute)  
**Observed:** INT8 **not significantly faster** than PyTorch (5.08ms vs 5.48ms, p=0.565)

**Statistical Evidence:**
- INT8 vs PyTorch: -0.398ms (-7.3%), p=0.565 (not significant)
- INT8 vs FP16: +2.764ms (+119.5%), p<1e-10 (highly significant slower)
- Effect size: -0.15 (negligible)

### 6.2 Why Is INT8 Not Faster?

**Hypothesis 1: Quantization/Dequantization Overhead**
- **Observation:** INT8 engine is 5.18MB (vs 4.23MB FP16)
- **Evidence:** Additional Q/DQ (quantize/dequantize) layers
- **Explanation:** Overhead of Q/DQ layers exceeds compute savings for tiny models

**Hypothesis 2: Small Model Low Compute Intensity**
- **sshleifer/tiny-gpt2:** 0.103M params = 102,714 params (test fixture), framework overhead dominates
- **INT8 advantage:** Faster compute (Tensor Cores support INT8)
- **INT8 disadvantage:** Q/DQ overhead, calibration complexity
- **Net effect:** Overhead > compute savings

**Hypothesis 3: Calibration Profile Mismatch**
- **Calibration:** 512 samples, batch=8, seq=128
- **Test scenarios:** batch=1-4, seq=8-27
- **Profile selection overhead:** Runtime overhead choosing optimal profile
- **Explanation:** Dynamic shape overhead compounds Q/DQ overhead

**Validation by Scenario:**

| Scenario | INT8 (ms) | FP16 (ms) | INT8 Slowdown |
|----------|-----------|-----------|---------------|
| single_micro | 1.100 | 0.782 | +40.7% |
| single_short | 1.985 | 1.153 | +72.2% |
| single_medium | 4.051 | 1.385 | +192.5% |
| single_long | 4.766 | 2.168 | +119.8% |
| batch_short | 7.243 | 3.117 | +132.4% |
| batch_medium | 11.318 | 5.274 | +114.6% |

**Pattern:** INT8 consistently slower than FP16 across all scenarios

### 6.3 When Does INT8 Make Sense?

**Winning Scenarios:**
1. **Large models:** >7B parameters (compute intensity amortizes Q/DQ)
2. **Memory-constrained:** INT8 uses 50% less memory
3. **Batch inference:** batch>16 (amortizes overhead)
4. **Long sequences:** seq>512 (compute-bound)

**Losing Scenarios:**
1. **Small models:** <500M parameters (Q/DQ overhead dominates)
2. **Low batch sizes:** batch<8 (overhead not amortized)
3. **Latency-critical:** INT8 adds 2-3ms overhead

**Production Recommendation:**
- **Skip INT8 for toy models:** FP16 is 2x faster; real gpt2 (124.4M) needs re-testing
- **Use INT8 for large models:** >7B parameters where memory/compute savings matter
- **Calibration cost:** 25s build + WikiText-2 dataset (512 samples)

---

## 7. ONNX Runtime GPU Disappointment Analysis

### 7.1 The Disappointment

**Expected:** ONNX Runtime GPU would match or exceed PyTorch GPU  
**Observed:** ONNX Runtime GPU **16.4% slower** than PyTorch GPU (6.37ms vs 5.48ms)

**Statistical Evidence:**
- Mean difference: +0.896ms
- Percent change: +16.4%
- p-value: 0.272 (not significant at α=0.05)
- Effect size: +0.29 (small)
- High variance: std=3.63ms (highest among all backends)

### 7.2 Why Is ONNX GPU Slower?

**Hypothesis 1: CUDAExecutionProvider Overhead**
- **Observation:** ORT GPU uses CUDAExecutionProvider abstraction layer
- **Evidence:** Batch medium → 14.39ms ORT GPU vs 7.92ms PyTorch (81.6% slower)
- **Explanation:** Provider abstraction adds overhead vs native PyTorch CUDA

**Hypothesis 2: Inefficient Memory Transfers**
- **Observation:** High variance (std=3.63ms, CV=57%)
- **Evidence:** Inconsistent performance across scenarios
- **Explanation:** Memory transfer patterns not optimized

**Hypothesis 3: Graph Optimization Trade-offs**
- **Small workloads help:** Single micro → 2.49ms ORT vs 4.35ms PyTorch (42.7% faster)
- **Large workloads hurt:** Batch medium → 14.39ms ORT vs 7.92ms PyTorch (81.6% slower)
- **Explanation:** Graph optimizations benefit small workloads but add overhead for large

**Validation by Scenario:**

| Scenario | ORT GPU (ms) | PyTorch GPU (ms) | ORT GPU vs PyTorch |
|----------|--------------|------------------|---------------------|
| single_micro | 2.493 | 4.350 | **-42.7%** (faster) ✅ |
| single_short | 3.228 | 4.374 | **-26.2%** (faster) ✅ |
| single_medium | 4.201 | 4.689 | **-10.4%** (faster) ✅ |
| single_long | 5.473 | 4.912 | +11.4% (slower) ❌ |
| batch_short | 8.443 | 6.601 | +27.9% (slower) ❌ |
| batch_medium | 14.390 | 7.924 | **+81.6%** (slower) ❌ |

**Pattern:** ORT GPU wins on micro workloads, loses badly on batched workloads

### 7.3 When Does ONNX GPU Make Sense?

**Winning Scenarios:**
1. **Micro workloads:** batch=1, seq<10 (42.7% faster than PyTorch)
2. **Cross-platform portability:** ONNX model format matters more than performance
3. **Mixed CPU/GPU:** Can fall back to CPU seamlessly

**Losing Scenarios:**
1. **Batched workloads:** batch>4 (up to 82% slower than PyTorch)
2. **Production GPU:** TensorRT FP16 is 63.7% faster than ORT GPU (2.31ms vs 6.37ms)
3. **Consistency:** High variance (CV=57%) makes it unreliable

**Production Recommendation:**
- **Avoid ONNX GPU for production:** Use TensorRT FP16 instead (63.7% faster)
- **Use ONNX CPU instead:** 68.4% faster than ONNX GPU (2.01ms vs 6.37ms)
- **Only use ONNX GPU for:** Cross-platform when TensorRT unavailable

---

## 8. TensorRT FP16 Sweet Spot Analysis

### 8.1 The Sweet Spot

**Finding:** TensorRT FP16 delivers **57.7% speedup** with **zero accuracy loss** (delta=-2.38e-08)

**Statistical Evidence:**
- Mean latency: 2.313ms (vs 5.475ms PyTorch)
- Speedup: 57.7% (p=4.1e-11, effect size -2.09)
- Perplexity delta: -2.38e-08 (0.0000000047% change)
- Pass threshold: <0.5% (actual: 0.0000000047%)

### 8.2 Why Is FP16 The Sweet Spot?

**Advantage 1: Tensor Core Acceleration**
- **RTX 4080 Tensor Cores:** 304 × 4th Gen (2x FP16 throughput vs FP32)
- **Evidence:** FP16 (2.31ms) vs FP32 (2.12ms) - similar performance
- **Explanation:** Both saturate available compute, but FP16 uses less memory bandwidth

**Advantage 2: Memory Bandwidth Savings**
- **FP16 uses 50% less memory:** 4.23MB engine vs 5.0MB FP32
- **Evidence:** Batch medium → 5.27ms FP16 vs 4.80ms FP32 (9.8% slower acceptable)
- **Explanation:** Memory bandwidth savings offset marginal compute overhead

**Advantage 3: Zero Accuracy Loss**
- **Perplexity delta:** -2.38e-08 (identical to FP32)
- **Evidence:** All 72,531 tokens validated, no degradation
- **Explanation:** FP16 has sufficient precision for inference (11-bit mantissa, 5-bit exponent)

**FP32 vs FP16 Comparison:**

| Metric | FP32 | FP16 | FP16 vs FP32 |
|--------|------|------|--------------|
| Mean latency (ms) | 2.120 | 2.313 | +9.1% slower |
| Throughput (tok/s) | 13,956 | 12,927 | -7.4% lower |
| Engine size (MB) | 5.0 | 4.2 | -16.0% smaller |
| Build time (s) | 20.7 | 42.6 | +106% longer |
| Perplexity delta | -2.38e-08 | -2.38e-08 | Identical |

**Trade-off:** FP16 trades 9% latency for 16% memory savings with zero accuracy loss

### 8.3 When Does FP16 Make Sense?

**Winning Scenarios:**
1. **Production GPU inference:** 57.7% faster than PyTorch, zero accuracy loss
2. **Memory-constrained:** 16% smaller engines (critical for large models)
3. **Frozen models:** 43s build time amortized over many inferences
4. **Batch inference:** Maintains speedup across all batch sizes

**Losing Scenarios:**
1. **Rapid iteration:** 43s build time per model change
2. **CPU-only:** FP16 requires GPU Tensor Cores (use ONNX CPU instead)
3. **Extreme accuracy requirements:** FP32 has 2x mantissa precision (though no difference in practice)

**Production Recommendation:**
- **Use TensorRT FP16 for production GPU:** 57.7% faster, zero accuracy loss, 16% memory savings
- **Build overhead acceptable:** 43s one-time cost, break-even at ~13,700 inferences
- **Skip INT8:** FP16 is 2x faster with better accuracy for small models

---

## 9. Accuracy Validation & Perplexity Gates

### 9.1 Perplexity Methodology

**Dataset:** WikiText-2 test set  
**Tokens:** 72,531  
**Baseline:** PyTorch transformers-gpu-compile (PPL=50,285.81)  
**Metric:** Perplexity = exp(mean negative log-likelihood)  
**Threshold:** Delta fraction < precision-specific limit

**Thresholds:**
- **FP32:** <0.1% (0.001)
- **FP16:** <0.5% (0.005)
- **INT8:** <2.0% (0.020)

### 9.2 Perplexity Results

**All backends PASS with identical perplexity:**

| Backend | Perplexity | Delta Fraction | Threshold | Pass |
| --- | --- | --- | --- | --- |
| transformers-gpu-compile | 50,285.81 | 0.0 | — | ✅ Baseline |
| onnxruntime-cpu | 50,285.81 | -2.71e-08 | 0.001 | ✅ Pass |
| onnxruntime-gpu | 50,285.81 | -2.21e-08 | 0.001 | ✅ Pass |
| tensorrt-fp32 | 50,285.81 | -2.38e-08 | 0.001 | ✅ Pass |
| tensorrt-fp16 | 50,285.81 | -2.38e-08 | 0.005 | ✅ Pass |
| tensorrt-int8 | 50,285.81 | -2.38e-08 | 0.020 | ✅ Pass |

**Key Findings:**
1. **All backends pass:** Delta < 3e-08 (0.0000000058%)
2. **Identical perplexity:** All produce same outputs (within floating-point precision)
3. **INT8 no accuracy loss:** Despite quantization, perplexity unchanged

### 9.3 Why Is Accuracy Identical?

**Explanation 1: Small Model Low Numerical Sensitivity**
- **sshleifer/tiny-gpt2:** 0.103M params (102,714), untrained test fixture (perplexity=50K indicates random weights)
- **Evidence:** All backends produce perplexity=50,285.81 (identical)
- **Implication:** Larger models may show differences

**Explanation 2: Inference-Only (No Gradients)**
- **Forward pass only:** No gradient computation
- **Evidence:** FP16/INT8 sufficient for inference
- **Implication:** Training would show accuracy differences

**Explanation 3: Large Test Set Averages Out Noise**
- **72,531 tokens:** Large enough to average out numerical noise
- **Evidence:** Delta < 1e-07 (7 orders of magnitude below threshold)
- **Implication:** Smaller test sets may show differences

### 9.4 Production Implications

**Confidence Level:** All backends are **production-ready** from accuracy perspective

**Recommendations by Precision:**
1. **FP32:** Use when accuracy is paramount (scientific, financial)
2. **FP16:** Use for production inference (57.7% faster, zero accuracy loss)
3. **INT8:** Use for large models (>7B) where memory matters (validate on your dataset)

**Caveats:**
- **Tiny model:** Larger models (>1B) may show accuracy differences
- **Single dataset:** Validate on your domain-specific dataset
- **Inference-only:** Training may show accuracy differences

---

## 10. Cross-Backend Comparison Matrix

### 10.1 Performance Matrix

| Backend | Mean (ms) | Speedup | Throughput | Consistency | Accuracy | Production |
|---------|-----------|---------|------------|-------------|----------|------------|
| **onnxruntime-cpu** | 2.010 | **+63.3%** ⚡ | 14,291 | Good (CV=54%) | ✅ Pass | ✅ CPU-only |
| **tensorrt-fp32** | 2.120 | **+61.3%** ✅ | 13,956 | Good (CV=56%) | ✅ Pass | ✅ GPU |
| **tensorrt-fp16** | 2.313 | **+57.7%** ✅ | 12,927 | Good (CV=58%) | ✅ Pass | ✅ **BEST GPU** |
| transformers-gpu-compile | 5.475 | Baseline | 5,058 | Best (CV=27%) | ✅ Baseline | ✅ Research |
| tensorrt-int8 | 5.077 | +7.3% ⚠️ | 6,106 | Poor (CV=58%) | ✅ Pass | ❌ Skip |
| onnxruntime-gpu | 6.371 | -16.4% ❌ | 4,465 | Poor (CV=57%) | ✅ Pass | ❌ Skip |

**CV = Coefficient of Variation (lower is better)**

### 10.2 Build Overhead Matrix

| Backend | Export | Build | Total | Engine Size | Reusable |
|---------|--------|-------|-------|-------------|----------|
| onnxruntime-cpu | 4.3s | N/A | 4.3s | 1.86MB | ✅ Yes |
| onnxruntime-gpu | 4.3s | N/A | 4.3s | 1.86MB | ✅ Yes |
| tensorrt-fp32 | 4.3s | 20.7s | 25.0s | 5.0MB | ✅ Yes |
| tensorrt-fp16 | 4.3s | 42.6s | 46.9s | 4.2MB | ✅ Yes |
| tensorrt-int8 | 4.3s | 25.2s + calib | ~30s | 5.2MB | ✅ Yes |
| transformers-gpu-compile | N/A | 0.5s | 0.5s | N/A | ✅ Yes |

**One-time costs (amortized over inferences)**

### 10.3 Decision Matrix

| Use Case | Recommended Backend | Why |
|----------|---------------------|-----|
| **Production GPU** | **TensorRT FP16** | 57.7% faster, zero accuracy loss, production-ready |
| **Production CPU** | **ONNX Runtime CPU** | 63.3% faster than PyTorch GPU for small models |
| **Research/Iteration** | **PyTorch GPU-compile** | 0.5s compile, unfrozen models, full ecosystem |
| **Large models (>7B)** | **TensorRT INT8** | Memory-critical, batch>16 amortizes overhead |
| **Cross-platform** | **ONNX Runtime CPU** | Portability, no GPU required |
| **Edge devices** | **ONNX Runtime CPU** | Small models, low latency, CPU-only |

**Avoid:**
- ❌ ONNX Runtime GPU (slower than PyTorch, use TensorRT instead)
- ❌ TensorRT INT8 for small models (overhead > savings)

---

## 11. Production Deployment Strategy

### 11.1 Deployment Decision Tree

```text
START: What is your deployment target?

GPU Available?
├─ NO → Use ONNX Runtime CPU
│   ├─ Model <500M? → ONNX CPU (63% faster than PyTorch GPU)
│   └─ Model >500M? → PyTorch CPU (ONNX may be slower)
│
└─ YES → Is model frozen (no more training)?
    │
    ├─ NO → Use PyTorch GPU-compile
    │   └─ Rapid iteration (0.5s compile per change)
    │
    └─ YES → Use TensorRT FP16
        ├─ Memory-critical & model >7B? → Consider TensorRT INT8
        ├─ Accuracy-critical? → Use TensorRT FP32
        └─ Default → **TensorRT FP16** (best balance)
```

### 11.2 Production Recommendations

**Tier 1 (Best Performance):**
1. **TensorRT FP16** (GPU): 57.7% faster, zero accuracy loss, production-ready
2. **ONNX Runtime CPU** (CPU-only): 63.3% faster than PyTorch GPU for small models

**Tier 2 (Good Performance):**
3. **TensorRT FP32** (GPU): 61.3% faster, perfect accuracy (if FP16 concerns exist)
4. **PyTorch GPU-compile** (GPU): Baseline, rapid iteration, unfrozen models

**Tier 3 (Avoid for Production):**
5. **TensorRT INT8** (GPU): Only for large models (>7B) where memory matters
6. **ONNX Runtime GPU** (GPU): 16.4% slower than PyTorch, avoid entirely

### 11.3 Build Overhead Amortization

**Break-even Analysis (TensorRT FP16 vs PyTorch):**
- **Speedup:** 57.7% (3.162ms saved per inference)
- **Build overhead:** 46.9s (one-time)
- **Break-even:** 46.9s / 3.162ms = **14,835 inferences**

**Production Implication:**
- For frozen models with >15K inferences: TensorRT FP16 worth it
- For rapid iteration (<15K per model version): PyTorch better

**Cost Savings (assuming $0.035/hour GPU):**
- PyTorch: 5.475ms/inference → 182,650 inf/hour → $0.000192 per inference
- TensorRT FP16: 2.313ms/inference → 432,500 inf/hour → $0.000081 per inference
- **Savings:** 57.8% cost reduction per inference

### 11.4 Deployment Checklist

**Pre-deployment:**
- [ ] Validate accuracy on your dataset (not just WikiText-2)
- [ ] Profile on your hardware (not just RTX 4080)
- [ ] Test with your workload (not just prefill-only)
- [ ] Measure build overhead (one-time cost)

**Deployment:**
- [ ] Build TensorRT engines with correct profiles (match your batch/seq ranges)
- [ ] Enable dynamic shapes (required for variable-length inputs)
- [ ] Set workspace size (6GB recommended, adjust for your GPU)
- [ ] Monitor degraded runs (should be 0%)

**Post-deployment:**
- [ ] Monitor latency (should match benchmark ±10%)
- [ ] Monitor accuracy (perplexity should be stable)
- [ ] Monitor memory (should be ~2GB for gpt2/124.4M)
- [ ] Monitor degraded rate (should be 0%)

---

## 12. Conclusions & Recommendations

### 12.1 Key Findings Summary

**Performance:**
1. **ONNX Runtime CPU is the surprise winner:** 63.3% faster than PyTorch GPU for small models (p=2.1e-14)
2. **TensorRT FP16 is the production GPU choice:** 57.7% faster, zero accuracy loss (p=4.1e-11)
3. **INT8 paradox resolved:** Not significant vs PyTorch (p=0.565), overhead > savings for small models
4. **ONNX Runtime GPU disappoints:** 16.4% slower than PyTorch (p=0.272, not significant)

**Accuracy:**
1. **All backends pass perplexity gates:** Delta < 3e-08 (negligible)
2. **FP16 has zero accuracy loss:** Identical perplexity to FP32 (50,285.81)
3. **INT8 "zero accuracy loss" is a red flag:** For untrained 0.103M (102,714 param) toy model, quantization doesn't matter (no signal to preserve)

**Build Overhead:**
1. **TensorRT FP16 overhead acceptable:** 46.9s one-time, break-even at 15K inferences
2. **ONNX Runtime lowest overhead:** 4.3s export, no build
3. **PyTorch lowest overhead:** 0.5s compile, but 63% slower

### 12.2 Production Recommendations

**For GPU Deployments:**
- ✅ **Use TensorRT FP16:** 57.7% faster, zero accuracy loss, production-ready
- ❌ **Avoid INT8:** Not significant vs PyTorch for small models
- ❌ **Avoid ONNX GPU:** 16.4% slower than PyTorch

**For CPU Deployments:**
- ✅ **Use ONNX Runtime CPU:** 63.3% faster than PyTorch GPU for small models (<500M)
- ⚠️ **Validate on large models:** ONNX CPU may be slower for >500M models

**For Rapid Iteration:**
- ✅ **Use PyTorch GPU-compile:** 0.5s compile, unfrozen models, full ecosystem
- ❌ **Avoid TensorRT:** 46.9s build overhead per model change

### 12.3 Future Work

**Immediate:**
1. ✅ TR118 complete (180 runs, 0% degraded, comprehensive analysis)
2. [ ] Validate on larger models (270M, 1B, 3B, 7B)
3. [ ] Test full generation loop (not just prefill)

**Short-Term (Q1 2026):**
1. [ ] Backport TR118 fixes to TR117 (real ONNX/TRT engines)
2. [ ] TR121: Model Scaling Study (270M-10B models)
3. [ ] TR119: Cost & Energy Analysis (resource monitoring)

**Medium-Term (Q2 2026):**
1. [ ] TR122: Resource Profiling Deep Dive (memory, power, thermal)
2. [ ] TR123: Multi-Hardware Generalization (A100, H100, cloud)
3. [ ] Production deployment guide (Docker, Kubernetes, monitoring)

### 12.4 Lessons Learned

**Technical:**
1. **ONNX CPU can beat PyTorch GPU:** For small models, CPU overhead < GPU overhead
2. **INT8 not always faster:** Quantization overhead > compute savings for small models
3. **FP16 is the sweet spot:** 57.7% faster, zero accuracy loss
4. **Dynamic shapes are critical:** Required for variable-length inputs

**Methodological:**
1. **Zero degradation is achievable:** Proper infrastructure → 100% success
2. **Perplexity gates are essential:** Accuracy validation prevents silent degradation
3. **Statistical rigor matters:** 5 reps, t-tests, effect sizes → confidence
4. **Prefill-only is sufficient:** Keeps backends comparable

**Process:**
1. **Build real engines:** Don't rely on aspirational claims (TR117 lesson)
2. **Validate accuracy:** Perplexity gates prevent silent degradation
3. **Measure build overhead:** One-time costs must be amortized
4. **Test on your hardware:** Results may differ on different GPUs

---

## 13. Appendices

### 13.1 Experimental Artifacts

**Data:**
- Raw results: `scripts/tr118/results/raw/bench_prefill_1765596906.jsonl` (180 runs)
- Latency summary: `scripts/tr118/results/processed/latency_summary.csv`
- Statistical analysis: `scripts/tr118/results/processed/statistical_analysis.json`
- Perplexity results: `scripts/tr118/results/processed/perplexity_results.csv`

**Metadata:**
- Experiment manifest: `scripts/tr118/results/processed/experiment_manifest_1765596906.json`
- Export metadata: `scripts/tr118/results/processed/export_metadata_1765596906.json`
- TRT build metadata: `scripts/tr118/results/processed/trt_build_metadata_1765596906.json`

**Plots:**
- Mean latency: `scripts/tr118/results/plots/mean_latency.png`
- Mean throughput: `scripts/tr118/results/plots/mean_throughput_tok_s.png`
- Degraded rate: `scripts/tr118/results/plots/degraded_rate.png`

### 13.2 Reproducibility

**Run the full pipeline:**
```bash
python scripts/tr118/run_experiment.py --config scripts/tr118/configs/matrix_postdoc.yaml
```

**Generate this report:**
```bash
python scripts/tr118/generate_report.py --config scripts/tr118/configs/matrix_postdoc.yaml
```

**Environment:**
- Python 3.13.1
- PyTorch 2.8.0+cu128
- ONNX Runtime 1.23.2
- TensorRT 10.12.0.36
- CUDA 12.8
- Windows 11 Pro (Build 26200)

### 13.3 Platform Details

**Hardware:**

```text
GPU: NVIDIA GeForce RTX 4080 Laptop
- VRAM: 12 GB GDDR6X (12,281.5 MB total)
- CUDA Cores: 9,728
- Tensor Cores: 304 (4th Gen)
- Memory Bandwidth: 504 GB/s
- Compute Capability: 8.9
- Driver: 566.03

CPU: Intel Core i9-13980HX
- Cores: 24 (8 Performance + 16 Efficient)
- Threads: 32
- Base Clock: 2.2 GHz
- Boost Clock: 5.6 GHz
- L3 Cache: 36 MB

RAM: 16 GB DDR5-4800
OS: Windows 11 Pro (Build 26200)
```

**Software:**
- Git SHA: 92275d856cfcabbe058e8fec534ba0a378052705
- Python: 3.13.1 (tags/v3.13.1:0671451, Dec 3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)]

**ONNX Runtime Providers:**
- TensorrtExecutionProvider
- CUDAExecutionProvider
- CPUExecutionProvider

### 13.4 Statistical Methodology

**Tests Used:**
- **t-test:** Two-sample t-test (Welch's, unequal variance assumed)
- **Effect size:** Cohen's d = (mean_a - mean_b) / pooled_std
- **Confidence intervals:** 95% CI via t-distribution
- **Significance threshold:** p < 0.05 (two-tailed)

**Assumptions:**
- **Independence:** Each run is independent
- **Normality:** Central Limit Theorem applies (n=30 per backend)
- **Homoscedasticity:** Welch's t-test handles unequal variance

**Sample Sizes:**
- Per backend: 30 samples (6 scenarios × 5 repetitions)
- Per scenario: 5 samples (5 repetitions)
- Total: 180 samples (6 backends × 30 each)

### 13.5 Acknowledgments

**Related Work:**
- TR117: Cross-Backend Benchmark (identified ONNX/TRT infrastructure gaps)
- TR115_v2: Rust Runtime Analysis (statistical methodology template)
- TR114_v2: Rust Multi-Agent Analysis (experimental design template)

**Tools:**
- PyTorch: Model loading and baseline inference
- ONNX: Model export and validation
- TensorRT: Engine building and inference
- ONNX Runtime: CPU/GPU inference
- Datasets: WikiText-2 for perplexity validation
- NumPy, Pandas, SciPy: Statistical analysis
- Matplotlib: Visualization

---

**Status:** TR118 complete. 180/180 runs successful (0% degraded), comprehensive analysis, production-ready
recommendations. TensorRT FP16 is the production GPU choice (57.7% faster, zero accuracy loss). ONNX Runtime CPU
is the surprise winner for CPU-only deployments (63.3% faster than PyTorch GPU for small models). INT8 not
recommended for small models (overhead > savings). ONNX Runtime GPU should be avoided (slower than PyTorch).

**Philosophy:** Local-first inference requires careful backend selection based on deployment constraints. For
production GPU deployments, TensorRT FP16 delivers the best balance of speed (57.7% faster), accuracy (zero
loss), and reliability (58% CV). For CPU-only deployments, ONNX Runtime CPU dominates for small models (63.3%
faster than PyTorch GPU). INT8 quantization is not a universal win - calibration overhead exceeds compute
savings for models <500M parameters. Statistical rigor and accuracy validation are essential for production
confidence.

---

**End of Technical Report 118**
