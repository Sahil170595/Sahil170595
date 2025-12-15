# Technical Report 118: ONNX Runtime + TensorRT Deep Dive


**Version:** 1.0
**Date:** 2025-12-14
**Status:** Draft (auto-generated from artifacts)
**Git SHA:** `2b683994e8336492ac0268344935fc895e5a074f`

## Abstract

TR118 deep-dives ONNX Runtime and TensorRT for local-first LLM inference, closing the TR117 gap where ONNX/TRT runs were fully degraded.
We report performance, degraded-rate, and accuracy (perplexity) gates, using artifact-driven reproducibility (JSONL + CSV + manifests).

## Executive Summary

### Key Findings

- Reliability: 360 run-level records across prefill, generate; degraded-rate = 25.0% (90/360)
- Accuracy: perplexity gate passed (see `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\perplexity_results.csv`)
- prefill: best mean latency = tensorrt-fp32 (6.1 ms), vs baseline transformers-gpu-compile (20 ms, -69.44%)
- generate: best mean latency = onnxruntime-gpu (35.4 ms), vs baseline transformers-gpu-compile (87.6 ms, -59.61%)
- Note: PyTorch `transformers-gpu-compile` uses `torch.compile(..., backend="cudagraphs", dynamic=False)` on Windows (no Triton).

### Honest Limitations

- `generate` mode is an uncached greedy loop (`use_cache=False`) and is not representative of KV-cached decoding throughput.
- `models/tiny-gpt2` in this repo is a toy/untrained model; perplexity is expected to be near-uniform (~vocab) and accuracy deltas mainly reflect numerical consistency.
- Single model (gpt2/124M) and single machine; results may not generalize to larger models (see TR121).
- Latency excludes end-to-end serving overhead (tokenization, networking, batching policies).

## Introduction

- TR117 established a cross-backend baseline and identified ONNX/TRT infrastructure failures.
- TR118 focuses on making ONNX export + TRT engine builds real and measurable, with explicit degraded reasons and accuracy gates.

## Methodology

### Metrics
- Latency (ms), throughput (tok/s), degraded rate.
- Generation mode (if enabled) uses an uncached greedy loop (repeated full forward passes).

### Accuracy Gate
- Perplexity on WikiText-2 vs PyTorch baseline with per-precision thresholds.

### Statistical Analysis
- 95% confidence intervals + t-tests + Cohen's d via TR117 helpers.

## Experimental Design

- Config: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\config_gpt2-100m.yaml`
- Prompt config: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr117\configs\matrix_tier3.yaml`
- Modes: prefill, generate
- Backends: transformers-gpu-compile, onnxruntime-cpu, onnxruntime-gpu, tensorrt-fp32, tensorrt-fp16, tensorrt-int8
- Scenarios: single_micro, single_short, single_medium, single_long, batch_short, batch_medium
- Repetitions: 5

**Artifacts root:** `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m`

### Run Manifest

- Manifest: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\experiment_manifest_1765753713.json`
- Duration (s): 635.4266421794891

## Environment

- OS: Windows-11-10.0.26200-SP0
- Python: 3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)]
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU (12282 MB, CC 8.9)
- ONNXRuntime providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
- Key packages: torch=2.8.0+cu128, transformers=4.57.0, onnxruntime=1.23.2, tensorrt=10.12.0.36

## Sanity Checks

### Model / ONNX Artifacts

| Field | Value |
| --- | --- |
| model | C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\models\gpt2-100m |
| onnx_path | C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\artifacts\tr118v2_crossover\gpt2-100m\onnx\gpt2-100m.onnx |
| onnx_sha256 | N/A |
| onnx_file_size_mb | 514.1 |
| external_data | False |
| external_total_mb | 0 |
| total_artifact_mb | 514.1 |
| initializer_numel_est | 134685696 |
| initializer_bytes_est_mb | 513.8 |
| weight_files_total_mb | 366.6 |

- Weight files: model.safetensors: 366.55745697021484

### TensorRT Engine Inspection

| Precision | Plan_MB | Layers | InspectorType | INT8_in_JSON | INT8_tensors | OutputDTypes | CalibSource | CalibCacheHit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | 520.5 | 609 | dict | False | False | Float=528, Int64=168 | N/A | N/A |
| fp16 | 850.6 | 594 | dict | False | False | Float=322, Half=190, Int64=168 | N/A | N/A |
| int8 | 522.6 | 694 | dict | False | False | Float=631, Int64=169 | dataset | True |

### Perplexity Correctness

| Field | Value |
| --- | --- |
| baseline_ppl | 358.2 |
| baseline_mean_nll | 5.881 |
| baseline_token_count | 72531 |
| ln_vocab | 10.82 |
| expected_uniform_ppl | 5.026e+04 |
| ref_loss_mean_nll | 5.658 |
| ref_loss_ppl | 286.5 |

### Logit Diffs vs PyTorch (Last Token)

| Backend | mean_abs | max_abs | providers_used | error |
| --- | --- | --- | --- | --- |
| onnxruntime-cpu | 1.219e-06 | 9.06e-06 | ['CPUExecutionProvider'] | N/A |
| onnxruntime-gpu | 0.0002704 | 0.001471 | ['CUDAExecutionProvider', 'CPUExecutionProvider'] | N/A |
| tensorrt-fp32 | 0.0002588 | 0.001593 | N/A | N/A |
| tensorrt-fp16 | 0.003046 | 0.02361 | N/A | N/A |
| tensorrt-int8 | 0.0002761 | 0.001754 | N/A | N/A |

### Sanity Warnings

- TensorRT INT8 engine inspector does not report INT8 coverage; treat INT8 claims as unverified (likely FP16/FP32 fallback).

## Results

### Mode: `prefill`

#### Overall Backend Summary (Run-Level)

| Backend | n_ok | n_total | degraded_rate | lat_mean_ms | lat_ci95 | thr_mean_tok_s | thr_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tensorrt-fp32 | 30 | 30 | 0 | 6.104 | [4.863, 7.345] | 4577 | [4014, 5140] |
| tensorrt-int8 | 30 | 30 | 0 | 6.13 | [4.765, 7.495] | 4636 | [4112, 5160] |
| onnxruntime-gpu | 30 | 30 | 0 | 6.233 | [4.128, 8.337] | 5217 | [4306, 6129] |
| tensorrt-fp16 | 30 | 30 | 0 | 6.243 | [4.928, 7.559] | 4537 | [3952, 5122] |
| onnxruntime-cpu | 30 | 30 | 0 | 17.6 | [15.64, 19.56] | 1547 | [1263, 1831] |
| transformers-gpu-compile | 30 | 30 | 0 | 19.97 | [14.19, 25.76] | 1804 | [1393, 2216] |

#### Resource Summary (Run-Level)

| Backend | n_ok | gpu_power_mean_w | gpu_mem_peak_mb | gpu_temp_peak_c | cpu_mem_peak_mb |
| --- | --- | --- | --- | --- | --- |
| onnxruntime-cpu | 30 | 23.29 | 4623 | 46.6 | 4972 |
| onnxruntime-gpu | 30 | 22.99 | 4633 | 47.2 | 5116 |
| tensorrt-fp16 | 30 | 17.42 | 6618 | 46 | 5189 |
| tensorrt-fp32 | 30 | 17.27 | 5586 | 46 | 5164 |
| tensorrt-int8 | 30 | 17.4 | 7534 | 45.17 | 5194 |
| transformers-gpu-compile | 30 | 14.49 | 3630 | 46.5 | 4976 |

- Summary CSV: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\latency_summary_prefill.csv`

#### Baseline Comparisons (Overall)

| baseline | candidate | metric | mean_a | mean_b | pct_change | p_value | cohens_d | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformers-gpu-compile | onnxruntime-cpu | latency_ms | 19.97 | 17.6 | -11.88 | 0.43 | -0.2052 | False |
| transformers-gpu-compile | onnxruntime-gpu | latency_ms | 19.97 | 6.233 | -68.8 | 2.624e-05 | -1.179 | True |
| transformers-gpu-compile | tensorrt-fp16 | latency_ms | 19.97 | 6.243 | -68.74 | 1.448e-05 | -1.223 | True |
| transformers-gpu-compile | tensorrt-fp32 | latency_ms | 19.97 | 6.104 | -69.44 | 1.165e-05 | -1.239 | True |
| transformers-gpu-compile | tensorrt-int8 | latency_ms | 19.97 | 6.13 | -69.31 | 1.301e-05 | -1.231 | True |

#### Figures

![mean_latency_prefill](../../scripts/tr118/results/tr118v2_crossover/gpt2-100m/plots/mean_latency_prefill.png)

![mean_throughput_tok_s_prefill](../../scripts/tr118/results/tr118v2_crossover/gpt2-100m/plots/mean_throughput_tok_s_prefill.png)

![degraded_rate_prefill](../../scripts/tr118/results/tr118v2_crossover/gpt2-100m/plots/degraded_rate_prefill.png)


### Mode: `generate`

#### Overall Backend Summary (Run-Level)

| Backend | n_ok | n_total | degraded_rate | lat_mean_ms | lat_ci95 | thr_mean_tok_s | thr_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| onnxruntime-gpu | 30 | 30 | 0 | 35.37 | [28.37, 42.36] | 365.9 | [290.4, 441.3] |
| transformers-gpu-compile | 30 | 30 | 0 | 87.57 | [74.14, 101] | 151 | [121.8, 180.3] |
| onnxruntime-cpu | 30 | 30 | 0 | 147.7 | [125, 170.4] | 79.81 | [67.75, 91.88] |
| tensorrt-fp16 | 0 | 30 | 1 | N/A | N/A | N/A | N/A |
| tensorrt-fp32 | 0 | 30 | 1 | N/A | N/A | N/A | N/A |
| tensorrt-int8 | 0 | 30 | 1 | N/A | N/A | N/A | N/A |

#### TTFT Summary (Run-Level)

| Backend | n_ok | ttft_mean_ms | ttft_ci95 | ttft_median_ms |
| --- | --- | --- | --- | --- |
| onnxruntime-gpu | 30 | 5.978 | [3.871, 8.085] | 4.669 |
| transformers-gpu-compile | 30 | 12.26 | [10.82, 13.69] | 12.38 |
| onnxruntime-cpu | 30 | 22.72 | [19.51, 25.92] | 18.41 |

#### Resource Summary (Run-Level)

| Backend | n_ok | gpu_power_mean_w | gpu_mem_peak_mb | gpu_temp_peak_c | cpu_mem_peak_mb |
| --- | --- | --- | --- | --- | --- |
| onnxruntime-cpu | 30 | 9.796 | 4894 | 45.27 | 2071 |
| onnxruntime-gpu | 30 | 36.58 | 4904 | 48.07 | 2292 |
| transformers-gpu-compile | 30 | 27.33 | 3901 | 47.3 | 2033 |

- Summary CSV: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\latency_summary_generate.csv`

#### Baseline Comparisons (Overall)

| baseline | candidate | metric | mean_a | mean_b | pct_change | p_value | cohens_d | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformers-gpu-compile | onnxruntime-cpu | latency_ms | 87.57 | 147.7 | 68.64 | 1.864e-05 | 1.204 | True |
| transformers-gpu-compile | onnxruntime-gpu | latency_ms | 87.57 | 35.37 | -59.61 | 2.417e-09 | -1.82 | True |

#### Figures

![mean_latency_generate](../../scripts/tr118/results/tr118v2_crossover/gpt2-100m/plots/mean_latency_generate.png)

![mean_throughput_tok_s_generate](../../scripts/tr118/results/tr118v2_crossover/gpt2-100m/plots/mean_throughput_tok_s_generate.png)

![degraded_rate_generate](../../scripts/tr118/results/tr118v2_crossover/gpt2-100m/plots/degraded_rate_generate.png)

### Accuracy (Perplexity Gate)

- Results CSV: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\perplexity_results.csv`
- Diagnostics JSON: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\perplexity_results.json`

| Backend | PPL | delta_frac | Threshold | Pass | Error |
| --- | --- | --- | --- | --- | --- |
| transformers-gpu-compile | 358.2 | 0 | nan | True | N/A |
| onnxruntime-cpu | 358.2 | -2.664e-06 | 0.001 | True | N/A |
| onnxruntime-gpu | 358.2 | 4.695e-07 | 0.001 | True | N/A |
| tensorrt-fp32 | 358.2 | 9.913e-07 | 0.001 | True | N/A |
| tensorrt-fp16 | 358.2 | 5.827e-05 | 0.005 | True | N/A |
| tensorrt-int8 | 358.2 | 9.913e-07 | 0.02 | True | N/A |

### Export Overhead (ONNX)

| Field | Value |
| --- | --- |
| onnx_path | C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\artifacts\tr118v2_crossover\gpt2-100m\onnx\gpt2-100m.onnx |
| export_time_s | 8.141 |
| file_size_mb | 514.1 |
| opset_version | 17 |
| dynamic_axes | True |
| trt_friendly_inputs | True |
| reused | False |
| valid | True |
| onnx_sha256 | N/A |

### TensorRT Build Overhead

- Build metadata: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\trt_build_metadata_1765753713.json`

| Precision | Plan | Built | Reused | Build s | Size MB | Dynamic | Profiles | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | gpt2-100m_fp32.plan | True | False | 74.82 | 520.5 | True | 5 | N/A |
| fp16 | gpt2-100m_fp16.plan | True | False | 162.7 | 850.6 | True | 5 | N/A |
| int8 | gpt2-100m_int8.plan | True | False | 93.64 | 522.6 | True | 5 | N/A |

## Discussion

### Interpretation
- This run demonstrates that ONNX export + TensorRT engine builds can be made reliable on a single Windows + CUDA workstation.
- For this tiny model and short prompts, ORT-CPU can win on latency due to reduced GPU launch/dispatch overhead; larger models should re-test (TR121).
- TensorRT build cost is non-trivial; treat it as an offline step that must be amortized for production value.

### Limitations / Threats to Validity
- See `Executive Summary: Honest Limitations` for the primary caveats.

## Conclusions

TR118 provides an artifact-driven pipeline for measuring ONNX Runtime and TensorRT locally, including degraded-rate accounting, build/export metadata, and perplexity gates.

## Recommendations

- If you need portability/simplicity: start with ONNX Runtime (CPU or CUDA EP).
- If you can prebuild engines and need maximum GPU throughput: TensorRT (FP16/INT8 as permitted by accuracy gates).
- Keep PyTorch as the reference baseline; on Windows prefer `torch.compile(..., backend="cudagraphs", dynamic=False)` for stability.

## Reproducibility

Run the full pipeline:

```bash
python scripts/tr118/run_experiment.py --config C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\config_gpt2-100m.yaml --device cuda
```

Generate this report from artifacts:

```bash
python scripts/tr118/generate_report.py --config C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\config_gpt2-100m.yaml --manifest C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed\experiment_manifest_1765753713.json
```

## Appendix

- Artifacts root: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m`
- Processed dir: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr118\results\tr118v2_crossover\gpt2-100m\processed`
