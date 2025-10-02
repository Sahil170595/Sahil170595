# Performance Deep Dive (Quantization + Kernels)

Generated on 2025-10-02 14:02:01 using the local GPU.

## Quantization Summary

| Mode | Accuracy | Loss | Model Size (KB) |
| --- | --- | --- | --- |
| BASELINE | 0.750 | 0.608 | 2.7 |
| QAT | 0.375 | 1.047 | 3.7 |
| INT8 | 0.750 | 0.608 | 3.0 |
| FP8 | 0.750 | 0.608 | 2.7 |

![Quant Accuracy](artifacts/quantization/accuracy_by_mode.png)
![Quant Loss](artifacts/quantization/loss_by_mode.png)
![Quant Size](artifacts/quantization/model_size.png)

- INT8 uses bitsandbytes on CUDA 12.8 (serialized footprint ~3.7 MB).
- QAT evaluation currently relies on TorchAO dynamic quantization; tune training or migrate fully to TorchAO for production workloads.
- FP8 snapshots remain float32 forward passes; ready for native FP8 kernels when available.


## Kernel Optimization Summary

### Attention Kernels

| Variant | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
| --- | --- | --- | --- | --- |
| torch | 24.62 | 73.16 | 0.17 | 244.09 |

![Attention](artifacts/kernel_optimization/attention_latency.png)

### Matmul (Tensor Core)

| Variant | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
| --- | --- | --- | --- | --- |
| 512x512x512 | 16.70 | 49.97 | 0.03 | 166.62 |

![Matmul](artifacts/kernel_optimization/matmul_latency.png)

### Kernel Fusion

| Variant | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
| --- | --- | --- | --- | --- |
| baseline_linear_gelu | 0.76 | 2.05 | 0.04 | 6.92 |
| fused_linear_gelu | 0.05 | 0.01 | 0.04 | 0.07 |

![Fusion](artifacts/kernel_optimization/fusion_latency.png)

- Benchmarks executed on CUDA (PyTorch 2.8.0+cu128).
- Flash Attention results appear when the package is active on CUDA.
- Matmul timings compare baseline matmul vs. Tensor Core path; fusion compares sequential vs. fused helper.


## Reproduction
```bash
# Quantization
set BANTERHEARTS_ENABLE_BITSANDBYTES=1
python scripts/quantization/generate_report.py

# Kernel suite
python -c "import torch; from banterhearts.optimization.kernels.attention.run import benchmark_attention_kernels; from banterhearts.optimization.kernels.matmul.run import benchmark_tensor_core_performance; from banterhearts.optimization.kernels.fusion.run import benchmark_fusion_patterns; device='cuda' if torch.cuda.is_available() else 'cpu'; print(benchmark_attention_kernels(device=device)); print(benchmark_tensor_core_performance(device=device)); print(benchmark_fusion_patterns(device=device))"
```
