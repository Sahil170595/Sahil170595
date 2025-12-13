# Technical Report 118: ONNX Runtime vs TensorRT (Local-First Prefill Benchmark)


**Version:** 1.0
**Date:** 2025-12-12
**Status:** Publish-ready template (auto-filled when artifacts exist)
**Git SHA:** `f30e3dc9c803f472b75911081c298c6b8a5078c8`

## Executive Summary

This report evaluates ONNX Runtime and TensorRT against a PyTorch baseline for **fully local** inference.
To keep ORT/TRT comparable, TR118 measures **prefill forward-pass** latency (no text generation loop).

**Primary question:** Can ORT/TRT match or exceed `transformers-gpu-compile` while preserving accuracy?

## Methodology

### What We Measure (Important)
- **Prefill forward-pass latency:** one model forward call over a padded `(batch, seq_len)` tensor.
- **Throughput (tok/s):** `(batch * seq_len) / latency` for the same forward pass.
- **Degraded runs:** missing engines/sessions, provider fallbacks, timeouts, or runtime exceptions.

### Accuracy Gate
- Perplexity on WikiText-2 (configurable), compared to PyTorch baseline.
- Pass/fail thresholds are defined in `accuracy.perplexity_thresholds`.

## Experimental Design

- Config: `scripts\tr118\configs\matrix_postdoc.yaml`
- Prompt config: `scripts/tr117/configs/matrix_tier3.yaml`
- Backends: transformers-gpu-compile, onnxruntime-cpu, onnxruntime-gpu, tensorrt-fp32, tensorrt-fp16, tensorrt-int8
- Scenarios: single_micro, single_short, single_medium, single_long, batch_short, batch_medium
- Repetitions: 5

**Artifacts root:** `scripts\tr118\results`

### Run Manifest

- Manifest: `scripts\tr118\results\processed\experiment_manifest_1765576421.json`
- Raw results: `scripts\tr118\results\raw\bench_1765576430.jsonl`
- Duration: 159.37971258163452

## Results

### Overall Backend Summary (Run-Level)

| Backend | n_ok | n_total | degraded_rate | lat_mean_ms | lat_ci95 | thr_mean_tok_s | thr_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| onnxruntime-cpu | 30 | 30 | 0 | 2.01 | [1.534, 2.485] | 1.429e+04 | [1.318e+04, 1.54e+04] |
| tensorrt-fp32 | 30 | 30 | 0 | 2.12 | [1.59, 2.65] | 1.396e+04 | [1.284e+04, 1.507e+04] |
| tensorrt-fp16 | 30 | 30 | 0 | 2.313 | [1.704, 2.922] | 1.293e+04 | [1.164e+04, 1.422e+04] |
| tensorrt-int8 | 30 | 30 | 0 | 5.077 | [3.771, 6.383] | 6106 | [5732, 6480] |
| transformers-gpu-compile | 30 | 30 | 0 | 5.475 | [4.959, 5.991] | 5058 | [4055, 6060] |
| onnxruntime-gpu | 30 | 30 | 0 | 6.371 | [4.802, 7.941] | 4465 | [4120, 4810] |

### Per-Scenario Summary (From `latency_summary.csv`)

- Summary CSV: `scripts\tr118\results\processed\latency_summary.csv`

### Accuracy (Perplexity Gate)

- Results CSV: `scripts\tr118\results\processed\perplexity_results.csv`

| Backend | PPL | Δ frac | Threshold | Pass | Error |
| --- | --- | --- | --- | --- | --- |
| transformers-gpu-compile | 5.029e+04 | 0 | nan | True | — |
| onnxruntime-cpu | 5.029e+04 | -4.923e-08 | 0.001 | True | — |
| onnxruntime-gpu | 5.029e+04 | -4.418e-08 | 0.001 | True | — |
| tensorrt-fp32 | 5.029e+04 | -5.07e-08 | 0.001 | True | — |
| tensorrt-fp16 | 5.029e+04 | -5.07e-08 | 0.005 | True | — |
| tensorrt-int8 | 5.029e+04 | -5.07e-08 | 0.02 | True | — |

### Export Overhead (ONNX)

| Field | Value |
| --- | --- |
| onnx_path | artifacts\onnx\tiny-gpt2.onnx |
| export_time_s | 4.254 |
| file_size_mb | 1.859 |
| opset_version | 17 |
| dynamic_axes | True |
| trt_friendly_inputs | True |
| valid | True |

### TensorRT Build Overhead

- Build metadata: `scripts\tr118\results\processed\trt_build_metadata.json`

| Precision | Build s | Size MB | Dynamic | Profiles | Error |
| --- | --- | --- | --- | --- | --- |
| int8 | 41.04 | 3.333 | True | 5 | — |

## Discussion

This report is designed to support a production decision for **fully local** systems:
- Prefer **TensorRT FP16** when the model is frozen and engine build/calibration overhead is acceptable.
- Prefer **ONNX Runtime** when portability matters and TensorRT is unavailable.
- Prefer **PyTorch** when iteration speed and feature velocity outweigh deployment complexity.

_Decision claims should only be made once both performance and perplexity gates pass._

## Reproducibility

Run the full pipeline:

```bash
python scripts/tr118/run_experiment.py --config scripts\tr118\configs\matrix_postdoc.yaml
```

Generate this report from artifacts:

```bash
python scripts/tr118/generate_report.py --config scripts\tr118\configs\matrix_postdoc.yaml
```
