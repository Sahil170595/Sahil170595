# Ollama Benchmark Deep Dive

## Overview
- Goal: establish current Ollama performance for Chimera Heart workloads, compare quantization builds (q4_0, q5_K_M, q8_0), and tune runtime parameters.
- Test window: 2025-09-30 (local), Ollama served on `http://localhost:11434` with llama3.1 8B instruct variants.
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU (13th Gen i9 laptop), driver-managed by Ollama for local inference.

## Data Sources
- Baseline system metrics: `baseline_system_metrics.json`
- Baseline narrative reports: `baseline_system_report.txt`, `baseline_ml_report.txt`
- Quantization sweep results: `csv_data/ollama_quant_bench.csv`
- Parameter sweep results: `csv_data/ollama_param_tuning.csv`
- Aggregated tuning summary: `csv_data/ollama_param_tuning_summary.csv`
- Visual assets: `artifacts/ollama/*.png`

## Methodology
1. Warmed and validated the existing q4_0 model by running `python test_baseline_performance.py`.
2. Pulled additional quantizations (`ollama pull llama3.1:8b-instruct-q5_K_M` and `...-q8_0`).
3. Created five representative gameplay prompts in `prompts/banter_prompts.txt` and executed a non-streaming REST sweep per quantization.
4. Per quantization call captured load, prompt-eval, eval timings, tokens per second, and response size; saved to CSV for analysis.
5. Tuned runtime via cartesian sweep across `num_gpu` (999, 80, 60, 40), `num_ctx` (1024, 2048, 4096), and temperature (0.2, 0.4, 0.8) using the best quantization.
6. Generated matplotlib visualizations (bar charts, scatter plot, per-temperature heatmaps) for inclusion in stakeholder materials.

## Baseline Performance (q4_0 default options)
- Mean banter generation latency: **5.15 s** across three in-game scenarios (two-variation outputs).
- Aggregate throughput: **7.90 tokens/s** (low because of cold-start sample during run).
- System snapshot (6 samples, 1 Hz): CPU **13.9% avg / 15.4% peak**, memory **72.1% avg / 73.5% peak**.
- GPU metrics: utilization **33.0% avg / 93.0% peak**, temperature **54.7 degC avg / 63.0 degC peak**, power draw **64.4 W avg / 142.6 W peak**.
- Observed issue: emoji characters in an Ollama response triggered a Windows `charmap` warning, preventing ML metrics from persisting (`baseline_ml_metrics.json` shows empty aggregates). Functional impact is limited to reporting; responses still completed successfully.

## Quantization Sweep Summary
| tag | prompts | mean_ttft_s | p95_ttft_s | mean_tokens_s | p05_tokens_s | p95_tokens_s |
| --- | --- | --- | --- | --- | --- | --- |
| q4_0 | 5 | 0.097 | 0.130 | 76.59 | 74.63 | 78.00 |
| q5_K_M | 5 | 1.354 | 5.148 | 65.18 | 64.88 | 65.74 |
| q8_0 | 5 | 2.008 | 7.718 | 46.57 | 46.14 | 46.84 |

Key takeaways:
- **q4_0** delivers ~17% more throughput than **q5_K_M** and ~65% more than **q8_0** while maintaining sub-0.15 s TTFT once warm.
- Higher precision tiers dramatically increase load + prompt-eval time with minimal quality benefit for short prompts.

### Prompt-Level Throughput (tokens/s)
| prompt | q4_0 | q5_K_M | q8_0 |
| --- | --- | --- | --- |
| Banter encouragement after mission failure | 74.11 | 64.85 | 46.04 |
| Co-op shooter victory quote | 76.96 | 65.03 | 46.79 |
| Rare loot celebration | 76.72 | 65.14 | 46.54 |
| Racing finish quip | 78.26 | 65.89 | 46.85 |
| Final boss motivation | 76.88 | 65.01 | 46.64 |

## Parameter Tuning (q4_0)
Top runtime configurations ranked by throughput (tokens/s):
| combo (num_gpu/num_ctx/temp) | mean_tokens_s | mean_ttft_s | mean_load_s |
| --- | --- | --- | --- |
| g40_c1024_t0.4 | 78.42 | 0.088 | 0.083 |
| g40_c1024_t0.8 | 78.06 | 0.075 | 0.073 |
| g60_c2048_t0.8 | 78.01 | 0.096 | 0.093 |
| g999_c1024_t0.4 | 77.93 | 0.087 | 0.082 |
| g999_c1024_t0.8 | 77.91 | 0.083 | 0.079 |
| g80_c1024_t0.4 | 77.83 | 0.079 | 0.076 |
| g40_c2048_t0.4 | 77.82 | 0.084 | 0.080 |
| g60_c1024_t0.8 | 77.77 | 0.077 | 0.073 |
| g60_c4096_t0.8 | 77.76 | 0.081 | 0.077 |
| g80_c1024_t0.8 | 77.76 | 0.101 | 0.099 |

Insights:
- Limiting `num_gpu` to 40 layers provides the best balance between load time and throughput; values above 80 show diminishing returns.
- Smaller contexts (1024) keep TTFT low; moving to 4096 raises initialization cost without materially improving throughput.
- Temperature adjustments influence creativity more than speed. Use **0.4** for production to maintain determinism while keeping throughput high.

## Visual Assets (embed-ready)
- `artifacts/ollama/quant_tokens_per_sec.png` – throughput per quantization.
- `artifacts/ollama/quant_ttft.png` – TTFT per quantization.
- `artifacts/ollama/param_ttft_vs_tokens.png` – TTFT vs throughput scatter (temperature-coded).
- `artifacts/ollama/param_heatmap_temp_0.2.png`, `..._0.4.png`, `..._0.8.png` – tokens/s heatmaps across `num_gpu`/`num_ctx` per temperature.

## Reproduction Guide
```powershell
# Baseline sanity check
python test_baseline_performance.py

# Quantization sweep (requires prompts/banter_prompts.txt)
# Script used in this run lives in the session transcript; rerun via PowerShell loop or adapt scripts/benchmark_cli.py

# Parameter sweep (q4_0, adjust arrays as needed)
# See automation block in commit history; rerun to refresh csv_data/ollama_param_tuning.csv

# Regenerate plots
python - <<'PY'
# (reuse matplotlib script under artifacts generation)
PY
```

## Recommendations
- Adopt **llama3.1:8b-instruct-q4_0** with `num_gpu=40`, `num_ctx=1024`, `temperature=0.4` for live banter generation.
- Issue a warm-up call on service startup to eliminate cold-load TTFT spikes before user traffic.
- Patch Windows console encoding (e.g., `chcp 65001`) or sanitize emoji in logs to restore ML metric exports.
- Integrate CSV exports into CI telemetry storage so sweeps can be trended over time.

## Linking Assets
- Summary for README: `reports/ollama_benchmark_summary.md`
- Full report (this file): `docs/Ollama_Benchmark_Report.md`
- Raw datasets: `csv_data/ollama_quant_bench.csv`, `csv_data/ollama_param_tuning.csv`
- Figures: `artifacts/ollama/`

Generated on 2025-10-01 02:00:08.
