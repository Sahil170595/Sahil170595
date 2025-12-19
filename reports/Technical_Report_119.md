# Technical Report 119: Cost & Energy Analysis


**Version:** 0.1
**Date:** 2025-12-18
**Status:** Draft (auto-generated from artifacts)
**Git SHA:** `4f09b5c5381989515a29b321bcb24d1804e370e2`

## Abstract

TR119 estimates the total cost of ownership (TCO) for multiple inference backends,
combining latency/throughput measurements with power telemetry to approximate dollars
and energy per 1M tokens. This draft is artifact-driven and makes conservative assumptions
about pricing and energy costs.

## Executive Summary

- Best-cost backend (mean across scenarios, on-demand): **transformers-gpu** at ~**$0.244 per 1M tokens**.
- Worst-cost backend: **transformers-gpu-compile** at ~**$0.2687 per 1M tokens**.
- Best spot pricing: **transformers-gpu** at ~**$0.08394 per 1M tokens** (65.6% savings vs on-demand).
- Lowest carbon footprint: **transformers-gpu-compile** at ~**35.2 gCO2e per 1M tokens**.
- Best-latency backend (mean across scenarios): **transformers-gpu** at ~**9.06 ms**.

### Honest Limitations

- Cloud pricing model includes on-demand, spot, and reserved tiers; actual savings depend on availability and commitment terms.
- Energy model assumes constant mean GPU power; no idle/warmup distinctions or CPU-only loads.
- Carbon intensity uses configurable regional averages; actual footprint depends on grid mix.
- Single hardware; multi-cloud/multi-GPU generalization is left to TR123.

## Introduction

- TR117 provided a baseline cross-backend benchmark with a simplistic cost model.
- TR119 extends this by incorporating resource telemetry (GPU power, utilization) and
  simple cloud/energy pricing assumptions to estimate cost and energy per token.

## Methodology

### Metrics
- Latency (ms), throughput (tokens/sec).
- GPU power (W), temperature (°C), memory (MB); CPU power when available.

### Cost & Energy Model (Current Draft)
- On-demand price: derived from `cloud_pricing.on_demand_usd_per_hour` in the TR119 config.
- Energy price: `energy.usd_per_kwh` in the TR119 config.
- Approximate GPU-hours per 1M tokens from throughput and mean GPU power.
- Compute:
  - `infra_cost_usd_per_1m_tokens` from GPU-hours × on-demand price.
  - `energy_cost_usd_per_1m_tokens` from kWh × energy price.
  - `total_cost_usd_per_1m_tokens` as their sum.

## Experimental Design

- Config: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr119\configs\smoke.yaml`
- Results root: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr119\results\tr119_smoke`

### Environment

- OS: Windows-10-10.0.26200-SP0
- Python: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU (12282 MB, CC 8.9)

## Results

### Latency, Throughput, and Telemetry (Per Backend/Scenario)

| Backend | Scenario | lat_mean_ms | lat_ci_lower | lat_ci_upper | throughput_mean_tok_s | gpu_power_mean_w | gpu_temp_mean_c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformers-gpu | single_short | 9.056 | nan | nan | 1215 | 30.48 | 47 |
| transformers-gpu-compile | single_short | 10.1 | nan | nan | 1091 | 25.06 | 46.67 |

### Cost & Energy per 1M Tokens (On-Demand Pricing)

| Backend | Scenario | throughput_mean_tok_s | gpu_power_mean_w | gpu_hours_per_1M_tok | infra_cost_usd_per_1M_tok | energy_cost_usd_per_1M_tok | total_cost_usd_per_1M_tok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformers-gpu | single_short | 1215 | 30.48 | 0.2287 | 0.2287 | 0.01533 | 0.244 |
| transformers-gpu-compile | single_short | 1091 | 25.06 | 0.2547 | 0.2547 | 0.01407 | 0.2687 |

### Multi-Tier Pricing Comparison (per 1M Tokens)

| Backend | Scenario | on_demand_usd | spot_usd | reserved_usd |
| --- | --- | --- | --- | --- |
| transformers-gpu | single_short | 0.244 | 0.08394 | 0.1754 |
| transformers-gpu-compile | single_short | 0.2687 | 0.09047 | 0.1923 |

### Carbon Footprint per 1M Tokens

| Backend | Scenario | energy_kwh_per_1M_tok | carbon_gco2e_per_1M_tok |
| --- | --- | --- | --- |
| transformers-gpu | single_short | 0.07667 | 38.33 |
| transformers-gpu-compile | single_short | 0.07034 | 35.17 |

### Figures

![mean_latency_tr119](../../scripts/tr119/results/tr119_smoke/plots/mean_latency_tr119.png)

![total_cost_per_1m_tokens_tr119](../../scripts/tr119/results/tr119_smoke/plots/total_cost_per_1m_tokens_tr119.png)

![cost_tiers_tr119](../../scripts/tr119/results/tr119_smoke/plots/cost_tiers_tr119.png)

![energy_efficiency_tr119](../../scripts/tr119/results/tr119_smoke/plots/energy_efficiency_tr119.png)

![carbon_footprint_tr119](../../scripts/tr119/results/tr119_smoke/plots/carbon_footprint_tr119.png)

![cost_vs_throughput_tr119](../../scripts/tr119/results/tr119_smoke/plots/cost_vs_throughput_tr119.png)

## Discussion

- In this draft, the lowest-cost backend is primarily driven by a combination of high throughput
  and moderate GPU power draw. Backends with low throughput and high power consumption naturally
  occupy the expensive end of the spectrum.
- Statistical analysis (when available) provides confidence intervals and significance testing
  to distinguish real performance differences from measurement noise.
- As we refine TR119, we will:
  - Expand the pricing model to multiple instance types and regions.
  - Incorporate CPU-only scenarios and heterogeneous workloads.
  - Calibrate against real cloud bills to target ±10% accuracy.

## Conclusions

TR119 lays the groundwork for a cost- and energy-aware backend comparison, turning raw
latency and power metrics into approximate dollars and Joules per token.

## Recommendations

- Use TR119 outputs as **relative** guidance between backends rather than absolute dollar values.
- For production, plug in your actual cloud/energy prices into the TR119 config and re-run.
- Combine TR119 with TR117/118 to select backends that jointly optimize latency, cost, and reliability.

## Reproducibility

Run the full TR119 pipeline:

```bash
python scripts/tr119/run_experiment.py --config C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\scripts\tr119\configs\smoke.yaml --device cuda
```
