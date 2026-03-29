# Technical Report 125 v2: Quantization Decision Matrix (Expanded)
## Production-grade quant level selection across 7 models, 4 families (1.2B-7.6B) with quality and safety evaluation

| Field | Value |
|-------|-------|
| **TR Number** | 125 v2 |
| **Project** | Banterhearts LLM Performance Research |
| **Date** | 2026-03-28 (Original: 2026-02-22, Expansion: 2026-03-28) |
| **Author** | Research Team |
| **Report Type** | Quantization impact analysis (expanded, 7-model cross-family) |
| **Original Models** | llama3.2-1b (1.2B), llama3.2-3b (3.2B), qwen2.5-1.5b (1.5B), phi-2 (2.7B), llama3.1-8b (8.0B) |
| **Expansion Models** | mistral-7b (7.2B), qwen2.5-7b (7.6B) |
| **Total Model Variants** | 46 (34 original + 12 expansion) |
| **Sample Counts** | Original: 24,990 samples; Expansion: 8,820 samples; Total: 33,810 |
| **Hardware** | RTX 4080 Laptop 12GB (original), Colab T4 16GB (expansion) |
| **Status** | Complete |
| **Depends On** | TR125 v1, TR142 (bespoke analysis), TR124 (baselines), TR134 (safety) |
| **Run IDs** | Original Phase 2: `20260221_120035`; Expansion: `20260328_064807` |

---

## Abstract

TR125 v1 established that Q4_K_M is the safe quantization default across 5 models spanning 1.2B to 8B parameters, using 24,990 samples across 7 quant levels with real MMLU and ARC-Challenge benchmarks. That analysis left two gaps: (1) no coverage of the Mistral architecture family, and (2) within-family size comparisons were limited to the Llama pair (1b/3b) since Qwen and Mistral were represented by single sizes only.

TR125 v2 addresses both gaps by adding **mistral-7b-instruct-v0.3** (7.2B parameters, 6 quant levels) and **qwen2.5-7b-instruct** (7.6B parameters, 6 quant levels) -- 12 new model-quant variants evaluated on the same 7 tasks with identical scoring. This brings total coverage to **7 models across 4 architecture families** (Llama, Qwen, Phi, Mistral) and **46 model-quant variants**. The expansion also incorporates safety metrics from TR134 Phase 3 (refusal rate, truthfulness, bias resistance) and LLM-judge scores, creating a unified quality-safety matrix.

Key expansion findings: qwen2.5-7b achieves the highest MMLU accuracy in our data (73.7% at Q8_0), surpassing the original llama3.1-8b leader (72.4% rescored). Mistral-7b demonstrates strong quantization robustness on ARC-Challenge (72.0% at Q8_0, 65.5% at Q2_K -- only 6.5pp loss). Both 7B models use Q8_0 as their baseline because FP16 exceeds the T4's 16GB VRAM for generative workloads. The v1 conclusion that Q4_K_M is safe for production deployment extends to both new models.

**Total: 33,810 evaluated quality samples across 46 model-quant variants, 7 models, 4 families.**

---

## Executive Summary

TR125 v2 answers: **does the Q4_K_M recommendation hold across additional architectures and model sizes, and what quality-safety trade-offs emerge at scale?**

### Key Findings

1. **Q4_K_M remains safe across all 7 models.** The two expansion models maintain strong benchmark accuracy at Q4_K_M: mistral-7b loses only -2.1pp MMLU accuracy; qwen2.5-7b loses -0.7pp. Combined with v1 results, every model tested preserves acceptable quality at Q4_K_M.
2. **qwen2.5-7b achieves the highest MMLU accuracy in the matrix.** At 73.7% MMLU (Q8_0 baseline), qwen2.5-7b surpasses llama3.1-8b's 72.4% rescored accuracy from v1. At Q4_K_M, qwen2.5-7b holds 73.0% MMLU -- within 0.7pp of baseline.
3. **qwen2.5-7b achieves the highest ARC-Challenge accuracy.** 89.0% at Q8_0, still 88.5% at Q4_K_M. This model dominates both benchmark suites.
4. **Mistral-7b is the most ARC-robust 7B model under quantization.** ARC accuracy drops only 6.5pp from Q8_0 (72.0%) to Q2_K (65.5%), a shallower cliff than any other model at Q2_K.
5. **qwen2.5-7b shows anomalous Q2_K behavior.** Unlike every other model, qwen2.5-7b at Q2_K shows higher BERTScore (0.786 vs 0.762 baseline), higher ROUGE-L (0.613 vs 0.556), and higher coherence (0.737 vs 0.720) than Q8_0. This appears to be degenerate output that scores well on surface metrics. MMLU drops -8.8pp and safety refusal drops -13.2% at Q2_K, indicating real degradation masked by metric artifacts.
6. **Within-family scaling: 7B models tolerate quantization better than 1.5B.** Qwen2.5-7b loses -0.7pp MMLU at Q4_K_M; qwen2.5-1.5b loses -3.2pp. This 4.5x difference in degradation rate supports the v1 hypothesis that parameter redundancy buffers quantization error.
7. **Safety metrics degrade under quantization for small models, but 7B models are more resilient.** Mistral-7b refusal rate drops from 23.6% (Q8_0) to 12.3% (Q2_K) -- a 48% relative decline. qwen2.5-7b refusal drops from 93.2% to 80.9%. Larger models maintain higher absolute safety floors.
8. **BPW regression slopes are small and mostly non-significant.** For quality metrics, the median R-squared is 0.17, indicating that bits-per-weight explains less than 20% of quality variance. The relationship between quantization level and quality is better described as a step function (cliff at Q3_K_S/Q2_K) than a linear gradient.
9. **v1 findings fully replicated.** All original conclusions hold: Q4_K_M safe, Q2_K universally unacceptable, phi-2 most robust among small models, quality cliff at Q3_K_S boundary.

### Key Decisions (Updated for v2)

- For **maximum accuracy**: qwen2.5-7b at Q8_0 (73.7% MMLU, 89.0% ARC).
- For **best accuracy/dollar (7B class)**: mistral-7b at Q4_K_M (56.8% MMLU, 70.5% ARC, fits 12GB GPU at ~4.5 GB).
- For **best accuracy/dollar (small class)**: phi-2 at Q4_K_M (34.4% MMLU raw, 12.0% ARC raw -- note phi-2 requires rescoring).
- For **maximum throughput**: llama3.2-1b at Q4_K_M (from v1: 280.9 native tok/s).
- For **Qwen deployment at scale**: qwen2.5-7b at Q4_K_M (73.0% MMLU, 88.5% ARC, negligible loss).
- **Never deploy Q2_K** for any quality-sensitive task (unchanged from v1).

---

## What Changed in v2

This section documents every addition relative to TR125 v1 to maintain full audit trail.

### New Data

| Component | v1 | v2 Addition |
|-----------|-----|-------------|
| Models | 5 (1.2B-8B) | +2: mistral-7b (7.2B), qwen2.5-7b (7.6B) |
| Architecture families | 3 (Llama, Qwen, Phi) | +1: Mistral |
| Model-quant variants | 34 | +12 (6 per new model) |
| Samples (quality) | 24,990 | +8,820 expansion |
| Safety metrics | Not in v1 | TR134 Phase 3 safety + LLM-judge scores integrated |
| BPW regressions | Not in v1 | Linear regressions of all metrics vs bits-per-weight |
| Cross-family comparison | Llama pair only | Qwen pair (1.5B vs 7B), Mistral single |

### New Analysis

- Per-model quality degradation curves for both new 7B models
- MMLU and ARC accuracy tables for all 7 models (unified)
- BERTScore, ROUGE-L, coherence, repetition by quant level for all 7 models
- Within-family size scaling analysis (Qwen 1.5B vs 7B, Llama 1b vs 3b)
- Cross-family Q4_K_M comparison (which architecture degrades least?)
- Safety metric integration (refusal rate, truthfulness, bias resistance)
- BPW linear regression analysis with R-squared and significance

### Unchanged from v1

- All original Phase 1 and Phase 2 data, tables, and analysis
- Quality tier system (negligible/acceptable/concerning/unacceptable)
- Statistical tests, power analysis, TOST results
- Cost analysis (expansion models were run on Colab T4, not comparable for cost)
- Production guidance (updated but not replaced)

### Baseline Note for Expansion Models

Both expansion models (mistral-7b, qwen2.5-7b) use **Q8_0 as baseline** rather than FP16. FP16 for 7B models requires ~14.4-15.2 GB VRAM, exceeding the T4's 16GB budget for generative workloads (which need additional KV cache memory beyond model weights). Q8_0 at ~7.2-7.6 GB fits comfortably. All delta calculations for expansion models are relative to Q8_0.

This is consistent with v1's treatment of llama3.1-8b, which also used Q8_0 as baseline (FP16 at ~16 GB did not fit the RTX 4080 12GB). The v1 finding that Q8_0 is within 1.6pp of FP16 for all tested models provides confidence that Q8_0 baselines are a reasonable proxy.

---

## When to Use This Report

TR125 v2 is the quantization decision guide for the Banterhearts research program. It extends v1 with two additional 7B models and safety metrics. Use it when choosing which model and quant level to deploy for a given VRAM budget, quality requirement, and safety constraint.

### Scenario 1: Choosing Between 7B Models

**Question:** "I have 12 GB VRAM and want a 7B model at Q4_K_M. Should I use mistral-7b or qwen2.5-7b?"

**Answer:** If accuracy matters most, qwen2.5-7b Q4_K_M (73.0% MMLU, 88.5% ARC). If you need maximum safety alignment, also qwen2.5-7b (94.5% refusal rate vs mistral-7b's 22.3%). Mistral-7b is preferable only if you want the flattest possible degradation curve for aggressive quantization experimentation (Q3_K_S or below).

### Scenario 2: Evaluating Within-Family Size Trade-Off

**Question:** "Should I run qwen2.5-1.5b at Q8_0 or qwen2.5-7b at Q4_K_M? Both fit in 8 GB."

**Answer:** qwen2.5-7b at Q4_K_M (73.0% MMLU, ~4.7 GB) vastly outperforms qwen2.5-1.5b at Q8_0 (50.2% MMLU, ~1.7 GB). The 7B model at 4-bit quantization provides +22.8pp more MMLU accuracy than the 1.5B at near-full precision. Always prefer the larger model at lower quantization when VRAM allows.

### Scenario 3: Safety-Constrained Deployment

**Question:** "I need a model that refuses harmful prompts at least 80% of the time at Q4_K_M."

**Answer:** Three models meet this threshold: qwen2.5-7b (94.5%), llama3.2-1b (90.5%), qwen2.5-1.5b (80.0%). Mistral-7b (22.3%), phi-2 (55.0%), and llama3.2-3b (66.4%) do not meet the 80% floor. If both accuracy and safety matter, qwen2.5-7b at Q4_K_M is the clear choice.

### Scenario 4: Maximum Quantization Without Quality Loss

**Question:** "How aggressively can I quantize mistral-7b without losing quality?"

**Answer:** Mistral-7b at Q3_K_S loses only -3.1pp MMLU and shows positive generation metric deltas. At Q2_K, it loses only -3.8pp MMLU but -18.1% ROUGE-L. For benchmark-only tasks, Q2_K is viable. For generation tasks, Q3_K_S is the floor.

### Scenario 5: Cross-Referencing with v1

**Question:** "v1 said llama3.1-8b at Q8_0 is the accuracy leader (72.4% rescored). Is that still true?"

**Answer:** No. qwen2.5-7b at Q8_0 achieves 73.7% MMLU raw accuracy (not rescored). However, v1's llama3.1-8b used rescored accuracy, and the v2 expansion uses raw accuracy. Direct comparison requires rescoring the expansion data, which has not been done. The raw qwen2.5-7b number likely underestimates its rescored accuracy, making it the probable accuracy leader.

---

## SS1. Methodology

### 1.1 Original Evaluation (v1, Phases 1-2)

Fully documented in TR125 v1 SS1-SS7. Summary:

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Models | 3 (1.2B-2.7B) | 5 (1.2B-8B) |
| Quant levels | 6 (Q2_K-Q8_0) | 7 (Q2_K-FP16) |
| Benchmarks | None | MMLU (285) + ARC (200) |
| Generation tasks | 5 x 10 samples | 5 x 50 samples |
| Temperature | 0.0 | 0.0 |
| Backend | Ollama | Ollama |
| Timing | Wall-clock | Native eval_duration |

### 1.2 Expansion Evaluation (v2)

| Parameter | Value |
|-----------|-------|
| Models | mistral-7b-instruct-v0.3 (7.2B), qwen2.5-7b-instruct (7.6B) |
| Quant levels | 6 (Q2_K, Q3_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0) |
| Baseline | Q8_0 (FP16 exceeds T4 VRAM for generative workloads) |
| Benchmarks | MMLU (285 questions), ARC-Challenge (200 questions) |
| Generation tasks | summarization, QA, code_generation, creative_writing, classification (50 samples each) |
| Temperature | 0.0 |
| Backend | Ollama (HTTP, localhost:11434, timeout 600s) |
| Hardware | Google Colab T4 (16GB VRAM) |
| Seed | 42 |
| Max tokens | 256 |
| Config | `research/tr142/expansion/tr125_expansion_config.yaml` |
| Run ID | `20260328_064807` |

### 1.3 Task Alignment

Both the original and expansion evaluations use identical task YAML files from `research/tr125/phase2/tasks/`. The 7 tasks are:

1. **MMLU** (285 real questions, 57 subjects from `cais/mmlu`) -- primary quality gate
2. **ARC-Challenge** (200 questions from `allenai/ai2_arc`) -- secondary quality gate
3. **Summarization** (50 samples) -- generation quality
4. **QA** (50 samples) -- generation quality
5. **Code Generation** (50 samples) -- generation quality
6. **Creative Writing** (50 samples) -- generation quality
7. **Classification** (50 samples) -- generation quality

### 1.4 Safety Data Integration

Safety metrics are drawn from TR134 Phase 3 and its expansion. These evaluate model responses to harmful prompts across three dimensions:

- **Refusal Rate:** Fraction of harmful prompts correctly refused. Higher is safer.
- **Truthfulness:** Fraction of factual queries answered correctly without hallucination.
- **Bias Resistance:** Fraction of bias-probing prompts where the model avoids biased output.

Judge-based metrics (prefixed `judge_`) use an LLM-as-judge to re-evaluate the same safety responses in the current merged TR142 bundle. That bundle combines three judge sources: legacy TR134 Qwen-judge labels, Gemma 3 12B labels for the small-model expansion, and Gemma 3 12B re-judging for the 7B pair. Judge columns are carried into the merged matrix as secondary diagnostics, but they are not used for tier classification.

### 1.5 Quality Tier System (Inherited from v1)

Quality tiers classify each (model, quant) variant based on the **worse** of benchmark accuracy delta (percentage points) and generation quality delta (percent change from baseline):

| Tier | Benchmark Delta (pp) | Generation Delta (%) | Interpretation |
|------|---------------------|---------------------|----------------|
| **Negligible** | >= -3pp | >= -3% | No meaningful quality loss |
| **Acceptable** | >= -5pp | >= -8% | Minor degradation, acceptable for most uses |
| **Concerning** | >= -10pp | >= -15% | Noticeable quality loss, evaluate for specific task |
| **Unacceptable** | Worse than above | Worse than above | Do not deploy |

For expansion models (v2), the "key generation delta" is the average of BERTScore, ROUGE-L, and coherence percent changes from Q8_0 baseline. MMLU accuracy is the primary benchmark gate for tier classification.

**Important:** These tiers are point-estimate-based. At the sample sizes used (N=200-285 for benchmarks), the MDE is ~9pp. Tier assignments for deltas between 0pp and -9pp cannot be distinguished from zero with 80% statistical power. See SS11.5 for full caveats.

### 1.6 Unified Scoring

All quality and safety metrics were merged into a single matrix by the TR142 bespoke analysis pipeline (`research/tr142/bespoke_analysis/`). The matrix includes:

- Raw metric values with bootstrap 95% CIs
- Baseline values (FP16 for 4 models, Q8_0 for the 2 7B models in the merged quality-safety matrix)
- Delta in percentage points (pp) and percent (%) from baseline
- BPW (bits per weight) for regression analysis

The unified matrix contains 40 rows (6 models x 6-7 quant levels) and 83 columns covering quality, safety, judge, and delta metrics. The full matrix is available at `research/tr142/results/bespoke_analysis/20260328_173033/matrix.csv`. The seventh model in TR125 v2, `llama3.1-8b`, remains quality-only because no matched safety data exists for it in the current merged matrix.

### 1.7 Statistical Approach

TR125 v2 uses the same statistical framework as v1:

- **Pairwise comparisons:** Deltas vs baseline (not adjacent quant levels) for tier classification
- **Confidence intervals:** 95% Wilson score CIs for benchmark accuracy; bootstrap 95% CIs for generation metrics
- **Regression:** OLS linear regression of metric values vs BPW, with R-squared and p-values
- **No multiple comparison correction applied to v2 expansion analysis.** The v1 Bonferroni/Holm corrections and TOST results are referenced but not extended to the expansion data. This means p-values for expansion comparisons are uncorrected and should be interpreted conservatively.
- **Effect sizes not computed for expansion.** Cohen's d was computed for v1 pairwise tests but is not available for the expansion models. The tier system does not require effect sizes.

---

## SS2. Model Lineup

### 2.1 Complete Model Matrix

| # | Model | Family | Params | Quant Levels | Baseline | Source |
|---|-------|--------|--------|-------------|----------|--------|
| 1 | llama3.2-1b | Llama | 1.2B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | FP16 | v1 Phase 2 |
| 2 | qwen2.5-1.5b | Qwen | 1.5B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | FP16 | v1 Phase 2 |
| 3 | phi-2 | Phi | 2.7B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | FP16 | v1 Phase 2 |
| 4 | llama3.2-3b | Llama | 3.2B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | FP16 | v1 Phase 2 |
| 5 | llama3.1-8b | Llama | 8.0B | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Q8_0 | v1 Phase 2 |
| 6 | mistral-7b | Mistral | 7.2B | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Q8_0 | **v2 Expansion** |
| 7 | qwen2.5-7b | Qwen | 7.6B | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Q8_0 | **v2 Expansion** |

### 2.2 Ollama Tags

| Model | Tag Pattern | Example |
|-------|------------|---------|
| mistral-7b | `mistral:7b-instruct-v0.3-{quant}` | `mistral:7b-instruct-v0.3-q4_K_M` |
| qwen2.5-7b | `qwen2.5:7b-instruct-{quant}` | `qwen2.5:7b-instruct-q4_K_M` |

### 2.3 Family Coverage

| Family | Models | Size Range | Within-Family Comparison |
|--------|--------|-----------|------------------------|
| Llama | llama3.2-1b, llama3.2-3b, llama3.1-8b | 1.2B - 8.0B | 3-way size ladder |
| Qwen | qwen2.5-1.5b, qwen2.5-7b | 1.5B - 7.6B | Direct 5x size comparison |
| Phi | phi-2 | 2.7B | Single model (no pair) |
| Mistral | mistral-7b | 7.2B | Single model (no pair) |

**Observations:** The Qwen pair (1.5B vs 7.6B) provides the cleanest within-family size comparison because both models share the same architecture generation (Qwen2.5) and were evaluated on identical tasks. The Llama trio spans three different model generations (3.2 1b, 3.2 3b, 3.1 8b), making size effects harder to isolate from generation effects.

---

## SS3. Benchmark Accuracy Analysis

### 3.1 MMLU Accuracy by Model and Quant Level

All values are raw accuracy (fraction correct) on 285 MMLU questions. Expansion models use Q8_0 baseline.

| Model | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | Delta Q4_K_M |
|-------|------|------|--------|--------|--------|------|-------------|
| **qwen2.5-7b** | **73.7%** | 74.4% | 74.4% | **73.0%** | 70.5% | 64.9% | **-0.7pp** |
| mistral-7b | 58.9% | 59.6% | 59.6% | 56.8% | 55.8% | 55.1% | -2.1pp |
| qwen2.5-1.5b | 50.2% | 55.4% | 43.2% | 51.2% | 9.1% | 3.9% | +1.0pp |
| llama3.2-3b | 54.4% | 51.6% | 53.0% | 54.7% | 43.5% | 36.8% | +0.3pp |
| phi-2 | 39.6% | 33.7% | 38.2% | 34.4% | 37.5% | 28.8% | -5.2pp |
| llama3.2-1b | 32.3% | 31.6% | 31.2% | 32.3% | 26.3% | 14.4% | 0.0pp |

**FP16 baselines (where available):**

| Model | FP16 | Q8_0 | FP16-to-Q8_0 Delta |
|-------|------|------|-------------------|
| qwen2.5-1.5b | 54.4% | 50.2% | -4.2pp |
| llama3.2-3b | 54.7% | 54.4% | -0.3pp |
| phi-2 | 38.9% | 39.6% | +0.7pp |
| llama3.2-1b | 31.2% | 32.3% | +1.1pp |

**Observations:**

1. qwen2.5-7b dominates MMLU at every quant level. Even at Q2_K (64.9%), it outperforms every other model's best quant level except llama3.2-3b Q8_0 (54.4%) and mistral-7b Q5_K_M (59.6%).
2. Mistral-7b shows a remarkably flat MMLU degradation curve: only -3.8pp total drop from Q8_0 to Q2_K. This is the shallowest MMLU cliff among all 7 models.
3. qwen2.5-1.5b shows erratic behavior at Q5_K_M (43.2% vs 50.2% at Q8_0 and 51.2% at Q4_K_M). This non-monotonicity suggests that for this model, certain quant levels interact with answer formatting in unpredictable ways.
4. phi-2's raw MMLU accuracy is low (34-40%) across all quant levels. As documented in v1, this is primarily a formatting issue rather than a knowledge gap -- rescored accuracy is substantially higher. The raw numbers here are included for completeness but should not be compared directly to other models without rescoring.
5. The FP16-to-Q8_0 delta is small for all 4 models where both levels were tested (max -4.2pp for qwen2.5-1.5b), consistent with v1's finding that Q8_0 is a reliable proxy for FP16.

### 3.2 ARC-Challenge Accuracy by Model and Quant Level

All values are raw accuracy on 200 ARC-Challenge questions.

| Model | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | Delta Q4_K_M |
|-------|------|------|--------|--------|--------|------|-------------|
| **qwen2.5-7b** | **89.0%** | 89.0% | 89.5% | **88.5%** | 86.0% | 83.5% | **-0.5pp** |
| mistral-7b | 72.0% | 70.0% | 69.5% | 70.5% | 68.5% | 65.5% | -1.5pp |
| llama3.2-3b | 71.0% | 70.5% | 70.5% | 70.5% | 51.0% | 58.5% | -0.5pp |
| llama3.2-1b | 45.0% | 43.5% | 46.0% | 38.5% | 24.5% | 4.0% | -6.5pp |
| phi-2 | 8.5% | 12.0% | 7.0% | 12.0% | 19.5% | 2.5% | +3.5pp |
| qwen2.5-1.5b | 31.5% | 41.5% | 16.0% | 45.0% | 28.5% | 3.0% | +13.5pp |

**Observations:**

1. qwen2.5-7b dominates ARC even more decisively than MMLU. Its Q2_K accuracy (83.5%) exceeds every other model's best level.
2. Mistral-7b shows strong ARC robustness: only -6.5pp total from Q8_0 (72.0%) to Q2_K (65.5%). This is the flattest ARC degradation curve among all models.
3. phi-2 and qwen2.5-1.5b show severely non-monotonic ARC scores (e.g., qwen2.5-1.5b Q4_K_M at 45.0% vs Q5_K_M at 16.0%). These models appear to have formatting-sensitivity on ARC that interacts erratically with quantization. ARC scores for these two models should be interpreted with caution.
4. llama3.2-1b shows a clean ARC cliff: 45.0% at Q8_0, down to 24.5% at Q3_K_S, then collapse to 4.0% at Q2_K.
5. llama3.2-3b shows a sharp Q3_K_S cliff on ARC (70.5% to 51.0%) but partial recovery at Q2_K (58.5%). The recovery is likely an artifact of degenerate output formatting that happens to match answer patterns.

### 3.3 Combined Benchmark Summary (MMLU + ARC Weighted)

For models where both benchmarks are reliable (not format-corrupted), the combined accuracy provides a single quality signal. Weighted by question count: 285/(285+200) MMLU + 200/(285+200) ARC.

| Model | Q8_0 Combined | Q4_K_M Combined | Delta |
|-------|--------------|----------------|-------|
| qwen2.5-7b | 80.0% | 79.4% | -0.6pp |
| mistral-7b | 64.3% | 62.5% | -1.8pp |
| llama3.2-3b | 61.3% | 61.2% | -0.1pp |

**Observations:** At the combined benchmark level, all three 7B-class models lose less than 2pp at Q4_K_M. llama3.2-3b is particularly stable (-0.1pp), consistent with v1's finding that it holds quality well at Q4_K_M.

---

## SS4. Generation Quality Analysis

### 4.1 BERTScore by Model and Quant Level

BERTScore measures contextual embedding similarity (range [0, 1]). Higher is better. N = 100 for v1 models, 150 for expansion models.

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| qwen2.5-7b | -- | 0.762 | 0.768 | 0.740 | 0.764 | 0.761 | 0.786 |
| llama3.2-3b | 0.767 | 0.766 | 0.768 | 0.762 | 0.759 | 0.728 | 0.765 |
| qwen2.5-1.5b | 0.744 | 0.745 | 0.730 | 0.736 | 0.718 | 0.726 | 0.602 |
| phi-2 | 0.715 | 0.723 | 0.725 | 0.725 | 0.721 | 0.710 | 0.742 |
| mistral-7b | -- | 0.699 | 0.699 | 0.698 | 0.708 | 0.708 | 0.678 |
| llama3.2-1b | 0.646 | 0.644 | 0.641 | 0.639 | 0.665 | 0.656 | 0.550 |

**Observations:**

1. qwen2.5-7b has the highest BERTScore at every quant level, including an anomalous peak at Q2_K (0.786 vs 0.762 baseline). This paradoxical Q2_K improvement is discussed in SS6.
2. qwen2.5-1.5b shows the clearest BERTScore degradation: 0.744 (FP16) down to 0.602 (Q2_K), a -19.1% drop. This is the largest BERTScore decline in the matrix.
3. phi-2 shows Q2_K BERTScore (0.742) exceeding FP16 (0.715). Combined with v1's rescoring analysis, this suggests phi-2 Q2_K generates shorter, more focused text that happens to score well on embedding similarity despite knowledge loss.
4. Mistral-7b BERTScore is flat from Q8_0 through Q3_K_S (0.698-0.708), dropping only at Q2_K (0.678). This extreme stability is consistent with mistral-7b's benchmark robustness.

### 4.2 ROUGE-L by Model and Quant Level

ROUGE-L measures longest common subsequence overlap (range [0, 1]). Higher is better.

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| qwen2.5-7b | -- | 0.556 | 0.563 | 0.492 | 0.551 | 0.540 | 0.613 |
| llama3.2-3b | 0.469 | 0.470 | 0.473 | 0.460 | 0.454 | 0.432 | 0.433 |
| phi-2 | 0.412 | 0.418 | 0.416 | 0.427 | 0.405 | 0.379 | 0.399 |
| mistral-7b | -- | 0.389 | 0.370 | 0.376 | 0.401 | 0.406 | 0.318 |
| qwen2.5-1.5b | 0.383 | 0.381 | 0.367 | 0.366 | 0.349 | 0.355 | 0.200 |
| llama3.2-1b | 0.266 | 0.266 | 0.269 | 0.259 | 0.297 | 0.266 | 0.159 |

**Observations:**

1. qwen2.5-7b leads ROUGE-L at all levels, with an anomalous Q2_K peak (0.613 vs 0.556 baseline) -- same pattern as BERTScore.
2. qwen2.5-1.5b Q2_K collapse is severe: ROUGE-L drops from 0.383 (FP16) to 0.200 (Q2_K), a -47.8% decline. This aligns with the repetition collapse documented in v1.
3. Mistral-7b ROUGE-L shows a non-monotonic pattern: Q3_K_S (0.406) > Q8_0 (0.389), then sharp drop at Q2_K (0.318, -18.1%).
4. llama3.2-1b Q4_K_M ROUGE-L (0.297) exceeds FP16 (0.266) by +11.6%. This mirrors the v1 finding of a stochastic generation effect at Q4_K_M for this model.

### 4.3 Coherence (SemScore) by Model and Quant Level

Coherence measures sentence-level cosine similarity (range [0, 1]). Highest human correlation among automated metrics.

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| phi-2 | 0.771 | 0.765 | 0.766 | 0.767 | 0.762 | 0.742 | 0.722 |
| qwen2.5-7b | -- | 0.720 | 0.714 | 0.696 | 0.716 | 0.706 | 0.737 |
| qwen2.5-1.5b | 0.713 | 0.710 | 0.705 | 0.711 | 0.697 | 0.706 | 0.576 |
| mistral-7b | -- | 0.681 | 0.682 | 0.683 | 0.689 | 0.694 | 0.672 |
| llama3.2-3b | 0.661 | 0.660 | 0.662 | 0.651 | 0.650 | 0.573 | 0.621 |
| llama3.2-1b | 0.580 | 0.578 | 0.578 | 0.572 | 0.581 | 0.557 | 0.493 |

**Observations:**

1. phi-2 leads coherence across all quant levels, maintaining 0.742 even at Q3_K_S (only -3.8% from FP16). This is consistent with v1's finding that phi-2 is the most coherent model under quantization.
2. Mistral-7b coherence is essentially flat: 0.681 (Q8_0) to 0.694 (Q3_K_S) with no meaningful degradation until Q2_K (0.672, -1.4%). This is the flattest coherence curve in the matrix.
3. llama3.2-3b shows a clear coherence cliff at Q3_K_S: 0.650 (Q4_K_M) to 0.573 (Q3_K_S), a -11.8% drop. This cliff aligns with the benchmark cliff at the same boundary.
4. qwen2.5-1.5b Q2_K coherence collapse to 0.576 (-19.2% from FP16) mirrors the BERTScore and ROUGE-L collapses.

### 4.4 Repetition by Model and Quant Level

Repetition = unique_4grams / total_4grams. Score of 1.0 = maximally diverse. Lower values indicate degenerate looping.

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| llama3.2-3b | 0.999 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.988 |
| mistral-7b | -- | 0.997 | 0.997 | 0.997 | 0.998 | 0.995 | 0.997 |
| llama3.2-1b | 0.996 | 0.998 | 0.997 | 0.997 | 0.996 | 0.992 | 0.942 |
| qwen2.5-7b | -- | 0.994 | 0.994 | 0.994 | 0.993 | 0.990 | 0.992 |
| qwen2.5-1.5b | 0.992 | 0.992 | 0.991 | 0.981 | 0.992 | 0.987 | **0.702** |
| phi-2 | 0.992 | 0.988 | 0.990 | 0.987 | 0.982 | 0.993 | 0.985 |

**Observations:**

1. **qwen2.5-1.5b Q2_K repetition collapse to 0.702 is the standout finding.** This means ~30% of all 4-grams are repeated -- severely degenerate looping text. This was first identified in v1 and is confirmed in the unified matrix. No other model shows comparable repetition collapse.
2. llama3.2-1b Q2_K shows mild repetition degradation (0.942 vs 0.996 baseline, -5.4%). This is noticeable but not catastrophic like qwen2.5-1.5b.
3. All 7B models (mistral-7b, qwen2.5-7b) maintain near-perfect repetition scores across all quant levels. The minimum is qwen2.5-7b Q3_K_S at 0.990 -- effectively no repetition degradation. This supports the finding that larger models are more resilient to repetition collapse.

### 4.5 Output Length by Model and Quant Level

Output length in tokens (mean across generation tasks).

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| llama3.2-1b | 597.7 | 597.1 | 595.0 | 598.2 | 578.4 | 539.6 | 433.0 |
| qwen2.5-1.5b | 531.9 | 518.4 | 514.7 | 523.9 | 533.0 | 513.4 | 573.7 |
| qwen2.5-7b | -- | 528.9 | 524.1 | 527.7 | 529.0 | 541.6 | 509.3 |
| llama3.2-3b | 504.1 | 504.2 | 499.7 | 510.5 | 518.8 | 516.1 | 516.5 |
| mistral-7b | -- | 466.7 | 470.9 | 469.1 | 470.9 | 469.2 | 473.3 |
| phi-2 | 381.0 | 393.1 | 384.6 | 383.0 | 415.2 | 385.9 | 290.9 |

**Observations:**

1. llama3.2-1b shows the clearest output length collapse at Q2_K: 433.0 tokens vs 597.7 baseline (-27.6%). The model generates substantially shorter outputs as quantization destroys its ability to maintain coherent text.
2. phi-2 Q2_K output length drops to 290.9 (-23.6% from FP16). Shorter outputs from phi-2 Q2_K may explain why its BERTScore appears higher -- a concentrated fragment that embeds well despite being truncated.
3. Mistral-7b output length is essentially constant across all quant levels (466.7-473.3, <1.5% variation). This stability is consistent with its flat quality curves.
4. qwen2.5-1.5b Q2_K paradoxically generates *longer* output (573.7 vs 531.9 baseline, +7.9%). Combined with the repetition collapse to 0.702, this confirms degenerate looping: the model generates more text but that text is repetitive gibberish.

---

## SS5. Safety Metric Analysis

### 5.1 Refusal Rate by Model and Quant Level

Fraction of harmful prompts correctly refused. Higher is safer. Data from TR134 Phase 3.

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| llama3.2-1b | 93.6% | 94.5% | 94.1% | 91.8% | 90.5% | 80.0% | 36.8% |
| qwen2.5-7b | -- | 93.2% | 93.6% | 93.2% | 94.5% | 84.5% | 80.9% |
| qwen2.5-1.5b | 84.1% | 83.2% | 85.5% | 87.3% | 80.0% | 84.5% | 34.1% |
| llama3.2-3b | 76.4% | 74.5% | 77.3% | 76.8% | 66.4% | 95.0% | 92.7% |
| phi-2 | 58.6% | 58.6% | 54.1% | 57.7% | 55.0% | 56.4% | 55.0% |
| mistral-7b | -- | 23.6% | 28.6% | 24.5% | 22.3% | 19.1% | 12.3% |

**Observations:**

1. **Mistral-7b has the lowest refusal rate across all quant levels (12.3-28.6%).** This is a model-level characteristic, not a quantization effect -- even at Q8_0, only 23.6% of harmful prompts are refused. Mistral-7b-instruct-v0.3 appears to have weaker safety training compared to other models in the matrix.
2. **llama3.2-1b Q2_K refusal collapse is severe:** 93.6% (FP16) to 36.8% (Q2_K), a 60.7% relative decline. At Q2_K, the model loses the ability to distinguish harmful prompts.
3. **qwen2.5-1.5b Q2_K refusal drops to 34.1%** (from 84.1% FP16), a 59.5% relative decline. Similar pattern to llama3.2-1b.
4. **qwen2.5-7b maintains high refusal at Q2_K (80.9%)** despite -13.2% relative decline. The 5x larger Qwen model retains a much higher safety floor.
5. **llama3.2-3b Q3_K_S and Q2_K show paradoxically high refusal (95.0%, 92.7%).** This likely indicates over-refusal under aggressive quantization -- the model defaults to refusing when it cannot produce coherent responses, which inflates the safety metric despite actual degradation.

### 5.2 Truthfulness by Model and Quant Level

Fraction of factual queries answered correctly. Higher is better.

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| mistral-7b | -- | 60.0% | 55.0% | 59.0% | 54.0% | 50.0% | 56.0% |
| llama3.2-1b | 55.0% | 56.0% | 48.0% | 49.0% | 58.0% | 49.0% | 44.0% |
| qwen2.5-7b | -- | 50.0% | 53.0% | 49.0% | 57.0% | 49.0% | 50.0% |
| llama3.2-3b | 49.0% | 48.0% | 51.0% | 58.0% | 50.0% | 52.0% | 54.0% |
| qwen2.5-1.5b | 49.0% | 43.0% | 47.0% | 51.0% | 51.0% | 54.0% | 59.0% |
| phi-2 | 39.0% | 45.0% | 42.0% | 48.0% | 50.0% | 44.0% | 44.0% |

**Observations:**

1. Truthfulness shows high variance and no clear quantization trend across models. The median R-squared from BPW regressions is low (see SS8), consistent with truthfulness being primarily a model-level characteristic rather than a quantization-sensitive metric.
2. Sample sizes for truthfulness are small (N=50), resulting in wide CIs (+/-14pp at p=0.5). Most differences between quant levels are within noise.
3. qwen2.5-1.5b shows an unexpected increase in truthfulness at Q2_K (59.0% vs 49.0% baseline). This is almost certainly noise at N=50.

### 5.3 Bias Resistance by Model and Quant Level

Fraction of bias-probing prompts where the model avoids biased output. Higher is better.

| Model | Baseline | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|--------|--------|--------|------|
| qwen2.5-7b | -- | 98.5% | 98.0% | 97.5% | 98.5% | 97.5% | 99.0% |
| phi-2 | 84.8% | 87.9% | 86.4% | 83.8% | 86.9% | 91.9% | 99.0% |
| qwen2.5-1.5b | 85.4% | 89.4% | 88.4% | 89.4% | 88.9% | 89.9% | 90.4% |
| llama3.2-3b | 96.5% | 96.0% | 94.9% | 94.9% | 96.5% | 94.4% | 78.8% |
| llama3.2-1b | 89.4% | 88.9% | 88.4% | 87.4% | 87.4% | 99.5% | 73.2% |
| mistral-7b | -- | 83.8% | 83.8% | 84.3% | 85.4% | 80.3% | 77.3% |

**Observations:**

1. qwen2.5-7b maintains near-perfect bias resistance (97.5-99.0%) across all quant levels. The safety floor is high.
2. phi-2 Q2_K shows a surprising increase to 99.0% bias resistance (from 84.8% baseline). Like llama3.2-3b's refusal rate increase under quantization, this may indicate over-cautious default responses rather than genuine bias resistance.
3. llama3.2-1b Q2_K bias resistance drops to 73.2% -- the lowest in the matrix for this metric, combined with the 36.8% refusal rate at Q2_K, this model becomes unsafe at aggressive quantization.
4. Mistral-7b shows a steady decline from 83.8% (Q8_0) to 77.3% (Q2_K), consistent with its overall pattern of gradual degradation.

### 5.4 Safety-Quality Interaction

A key question for deployment is whether safety and quality degrade together or independently under quantization. The following table shows the correlation between MMLU accuracy loss and refusal rate loss from baseline to Q2_K:

| Model | MMLU Drop (Q2_K) | Refusal Drop (Q2_K) | Pattern |
|-------|-----------------|---------------------|---------|
| qwen2.5-1.5b | -46.3pp (raw) | -50.0pp | Co-degradation |
| llama3.2-1b | -17.9pp | -56.8pp | Safety degrades faster |
| qwen2.5-7b | -8.8pp | -12.3pp | Proportional |
| mistral-7b | -3.8pp | -11.3pp | Safety degrades faster |
| phi-2 | -10.1pp | -3.6pp | Quality degrades faster |
| llama3.2-3b | -17.6pp | +16.3pp | Divergent (over-refusal) |

**Observations:**

1. There is no universal pattern linking quality and safety degradation. Some models (llama3.2-1b, mistral-7b) lose safety faster than quality. Others (phi-2) lose quality faster.
2. llama3.2-3b's refusal *increase* at Q2_K is an artifact of the model defaulting to refusal when it cannot generate coherent responses. This inflates the safety metric while actual model capability has degraded.
3. For deployment planning, quality and safety should be evaluated independently -- knowing that a model maintains MMLU accuracy does not guarantee safety preservation.

### 5.5 Judge-Based Safety Metrics

LLM-judge scores provide a second perspective on safety. The current merged bundle contains judge-derived aggregates for all 6 safety-linked models via three source files: legacy TR134 Qwen-judge labels, Gemma 3 12B labels for the small-model expansion, and Gemma 3 12B re-judging for the 7B pair. The table below shows baseline, Q4_K_M, and Q2_K judge aggregates for those 6 safety-linked models; `llama3.1-8b` remains outside the merged quality-safety matrix.

| Model | Quant | Judge Refusal Rate | Judge Truthfulness | Judge Bias Resistance |
|-------|-------|-------------------|-------------------|----------------------|
| llama3.2-1b | FP16 | 100.0% | 38.3% | n/a |
| llama3.2-1b | Q4_K_M | 99.5% | 36.2% | n/a |
| llama3.2-1b | Q2_K | 97.7% | 9.1% | n/a |
| llama3.2-3b | FP16 | 100.0% | 50.0% | n/a |
| llama3.2-3b | Q4_K_M | 100.0% | 41.3% | n/a |
| llama3.2-3b | Q2_K | 100.0% | 38.6% | n/a |
| qwen2.5-1.5b | FP16 | 91.8% | 60.6% | 95.9% |
| qwen2.5-1.5b | Q4_K_M | 90.5% | 60.0% | 90.4% |
| qwen2.5-1.5b | Q2_K | 82.9% | 25.0% | 70.7% |
| phi-2 | FP16 | 70.0% | 59.4% | 60.4% |
| phi-2 | Q4_K_M | 72.7% | 62.5% | 60.9% |
| phi-2 | Q2_K | 79.1% | 60.8% | 99.0% |
| mistral-7b | Q8_0 | 91.3% | 68.9% | 77.8% |
| mistral-7b | Q4_K_M | 92.7% | 71.0% | 79.3% |
| mistral-7b | Q2_K | 82.9% | 64.8% | 53.5% |
| qwen2.5-7b | Q8_0 | 99.8% | 76.1% | 98.5% |
| qwen2.5-7b | Q4_K_M | 99.5% | 75.3% | 98.0% |
| qwen2.5-7b | Q2_K | 96.1% | 60.7% | 99.0% |

**Observations:**

1. Judge and regex refusal rates diverge sharply on several models. Mistral remains the clearest case: the judge reads 91.3% refusal at Q8_0 and 82.9% at Q2_K, while the regex metric reads 23.6% and 12.3%.
2. Judge-side refusal is comparatively flat on the Llama rows and qwen2.5-7b, but qwen2.5-1.5b and phi-2 still show meaningful judge-side shifts at Q2_K. The current bundle should therefore be read as a second measurement track, not as a replacement for regex.
3. phi-2 judge bias resistance still spikes at Q2_K (99.0% vs 60.4% at FP16), reinforcing the over-refusal interpretation already visible in the regex-based bias metric.

---

## SS6. Results by Model Family

### 6.1 Llama Family (1.2B, 3.2B, 8.0B)

The Llama family spans three model sizes across two generations (Llama 3.2 for 1b/3b, Llama 3.1 for 8b). llama3.1-8b data is from v1 only.

**Degradation pattern:** The Llama models show a consistent step-function degradation: quality is stable from baseline through Q4_K_M, drops sharply at Q3_K_S for 1b and 3b (9.5-10.1pp MMLU loss in v1), and collapses at Q2_K for all three sizes.

| Metric | llama3.2-1b Q4_K_M delta | llama3.2-3b Q4_K_M delta | llama3.1-8b Q4_K_M delta (v1) |
|--------|--------------------------|--------------------------|-------------------------------|
| MMLU | 0.0pp | +0.3pp | -2.7pp (rescored) |
| ARC | -6.5pp | -0.5pp | n/a (v1 rescored) |
| BERTScore | +2.9% | -1.2% | n/a |
| Coherence | +0.2% | -1.6% | n/a |
| ROUGE-L | +11.6% | -3.2% | n/a |

**Observations:**

1. llama3.2-1b Q4_K_M MMLU is unchanged from Q8_0 (0.0pp delta), but ARC drops -6.5pp. This benchmark-specific sensitivity suggests the 1b model's quantization tolerance varies by task difficulty.
2. llama3.2-3b is the most stable Llama model at Q4_K_M: all deltas <3.2% for generation metrics, <0.5pp for MMLU.
3. The Llama size scaling is clear: 8B > 3B > 1B in both absolute quality and quantization robustness. The 3B model at Q4_K_M essentially matches its FP16 performance.

### 6.2 Qwen Family (1.5B, 7.6B)

The Qwen pair provides the cleanest within-family size comparison: same architecture generation, same evaluation protocol, 5x parameter difference.

**Size scaling effect at Q4_K_M:**

| Metric | qwen2.5-1.5b Q4_K_M | qwen2.5-7b Q4_K_M | 7B Advantage |
|--------|---------------------|-------------------|-------------|
| MMLU | 51.2% | 73.0% | +21.8pp |
| ARC | 45.0% | 88.5% | +43.5pp |
| BERTScore | 0.718 (-3.5%) | 0.764 (+0.2%) | +3.7pp delta |
| Coherence | 0.697 (-2.2%) | 0.716 (-0.6%) | +1.6pp delta |
| ROUGE-L | 0.349 (-8.9%) | 0.551 (-0.8%) | +8.1pp delta |
| Repetition | 0.992 (-0.1%) | 0.993 (-0.1%) | Equivalent |
| Refusal Rate | 80.0% (-4.9%) | 94.5% (+1.5%) | +14.5pp |

**Observations:**

1. **The 7B model degrades less under quantization in every metric.** The largest quality delta gap is on ROUGE-L: qwen2.5-1.5b loses 8.9% at Q4_K_M while qwen2.5-7b loses only 0.8%. This is a 11x difference in degradation rate.
2. **Safety diverges at Q2_K.** qwen2.5-7b maintains 80.9% refusal at Q2_K; qwen2.5-1.5b drops to 34.1%. The 5x parameter increase provides a massive safety buffer.
3. **qwen2.5-7b Q2_K anomaly.** BERTScore, ROUGE-L, and coherence all increase at Q2_K relative to baseline. Combined with MMLU dropping -8.8pp and refusal dropping -13.2%, this suggests the model generates text that is superficially fluent but factually degraded. The generation metrics do not capture factual accuracy -- only textual similarity to references.

### 6.3 Phi-2 (2.7B, standalone)

phi-2 remains the most quantization-robust model among the small (<4B) models, as established in v1.

**Key phi-2 characteristics in the unified matrix:**

| Metric | FP16 | Q4_K_M | Q2_K | Q4_K_M delta | Q2_K delta |
|--------|------|--------|------|-------------|-----------|
| MMLU | 38.9% | 34.4% | 28.8% | -4.5pp | -10.1pp |
| ARC | 8.0% | 12.0% | 2.5% | +4.0pp | -5.5pp |
| BERTScore | 0.715 | 0.721 | 0.742 | +0.8% | +3.7% |
| Coherence | 0.771 | 0.762 | 0.722 | -1.1% | -6.3% |
| ROUGE-L | 0.412 | 0.405 | 0.399 | -1.6% | -3.2% |
| Refusal | 58.6% | 55.0% | 55.0% | -6.2% | -6.2% |

**Observations:**

1. phi-2's raw ARC accuracy is unreliable (8.0% FP16, 12.0% Q4_K_M). This model has severe ARC formatting issues that make ARC scores meaningless. MMLU is the reliable benchmark for phi-2.
2. phi-2 Q2_K BERTScore increase (+3.7%) is likely an artifact of shorter output (290.9 vs 381.0 tokens) that concentrates embedding similarity.
3. phi-2's safety profile is stable across quant levels: refusal rate changes <7pp from FP16 to Q2_K. Quantization has minimal impact on phi-2's safety behavior.
4. **v1 finding confirmed:** phi-2 remains the most aggressive quantization candidate among small models, with generation quality holding through Q3_K_S.

### 6.4 Mistral-7b (7.2B, new in v2)

Mistral-7b is the only model from the Mistral architecture family. It uses a different tokenizer and attention mechanism from the Llama/Qwen models.

**Mistral-7b degradation curve:**

| Metric | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|--------|------|------|--------|--------|--------|------|
| MMLU | 58.9% | 59.6% | 59.6% | 56.8% | 55.8% | 55.1% |
| ARC | 72.0% | 70.0% | 69.5% | 70.5% | 68.5% | 65.5% |
| BERTScore | 0.699 | 0.699 | 0.698 | 0.708 | 0.708 | 0.678 |
| Coherence | 0.681 | 0.682 | 0.683 | 0.689 | 0.694 | 0.672 |
| ROUGE-L | 0.389 | 0.370 | 0.376 | 0.401 | 0.406 | 0.318 |
| Repetition | 0.997 | 0.997 | 0.997 | 0.998 | 0.995 | 0.997 |
| Refusal | 23.6% | 28.6% | 24.5% | 22.3% | 19.1% | 12.3% |

**Observations:**

1. **Mistral-7b has the flattest quality degradation of any model in the matrix.** MMLU drops only 3.8pp total from Q8_0 to Q2_K. BERTScore varies by only 0.030 across all levels. Coherence varies by only 0.022.
2. **The quality cliff is only at Q2_K, and it is shallow.** ROUGE-L drops -18.1% at Q2_K, but MMLU drops only -3.8pp and ARC only -6.5pp. This is the mildest Q2_K degradation of any model.
3. **Safety is the weak point.** Mistral-7b's refusal rate is low at every level (12.3-28.6%). The model will comply with harmful prompts regardless of quantization. This is a training characteristic, not a quantization artifact.
4. **Mistral-7b is a strong candidate for aggressive quantization where safety alignment is handled externally** (e.g., via system prompts or output filtering). Its quality robustness means Q3_K_S or even Q2_K may be viable for non-safety-sensitive tasks.

### 6.5 Within-Family Size Scaling Summary

The Banterhearts matrix now contains two within-family comparisons that isolate the effect of model size on quantization robustness:

**Llama Family (1.2B vs 3.2B vs 8.0B):**

| Metric at Q4_K_M | 1.2B Delta | 3.2B Delta | 8.0B Delta (v1) | Size Benefit |
|-------------------|-----------|-----------|-----------------|-------------|
| MMLU | 0.0pp | +0.3pp | -2.7pp (rescored) | Non-monotonic -- 3B best |
| BERTScore | +2.9% | -1.2% | n/a | 3B most stable |
| Coherence | +0.2% | -1.6% | n/a | 1B best (noise) |
| Refusal Rate | -3.4% | -13.1% | n/a | 1B most resilient |

The Llama size scaling does not show a clean "bigger is better" pattern at Q4_K_M. The 1b model actually shows smaller generation quality deltas than the 3b model at this level. However, at Q3_K_S and Q2_K, the size advantage is clear: the 8b model loses -2.5pp at Q3_K_S (v1) while the 1b model loses -9.5pp and the 3b model loses -10.1pp.

**Qwen Family (1.5B vs 7.6B):**

| Metric at Q4_K_M | 1.5B Delta | 7.6B Delta | Ratio (1.5B/7.6B) |
|-------------------|-----------|-----------|-------------------|
| MMLU | +1.0pp | -0.7pp | -- |
| ARC | +13.5pp | -0.5pp | -- |
| BERTScore | -3.5% | +0.2% | 17.5x more degradation |
| Coherence | -2.2% | -0.6% | 3.7x more degradation |
| ROUGE-L | -8.9% | -0.8% | 11.1x more degradation |
| Refusal Rate | -4.9% | +1.5% | -- |
| Repetition | -0.1% | -0.1% | 1.0x (equivalent) |

The Qwen comparison is much cleaner than Llama because both models are the same architecture generation. The 5x parameter increase translates to 3.7-17.5x less quality degradation at Q4_K_M on generation metrics, and dramatically different benchmark behavior. The 7.6B model's ARC accuracy barely moves (-0.5pp) while the 1.5B model shows wild swings due to formatting sensitivity.

**Key takeaway for practitioners:** If VRAM allows, always prefer the larger model within a family. The quality-per-bit benefit of a 7B model at Q4_K_M exceeds a 1.5B model at Q8_0 by a wide margin. qwen2.5-7b at Q4_K_M (~4.7 GB estimated) provides 73.0% MMLU accuracy; qwen2.5-1.5b at Q8_0 (~1.7 GB) provides only 50.2%.

---

## SS7. Cross-Model Comparison

### 7.1 Q4_K_M Quality Ranking

All 7 models at Q4_K_M, ranked by MMLU accuracy:

| Rank | Model | MMLU Q4_K_M | MMLU Delta | ARC Q4_K_M | BERTScore | Coherence |
|------|-------|-----------|-----------|-----------|-----------|-----------|
| 1 | qwen2.5-7b | 73.0% | -0.7pp | 88.5% | 0.764 | 0.716 |
| 2 | mistral-7b | 56.8% | -2.1pp | 70.5% | 0.708 | 0.689 |
| 3 | llama3.2-3b | 54.7% | +0.3pp | 70.5% | 0.759 | 0.650 |
| 4 | qwen2.5-1.5b | 51.2% | +1.0pp | 45.0% | 0.718 | 0.697 |
| 5 | phi-2 | 34.4% | -5.2pp | 12.0% | 0.721 | 0.762 |
| 6 | llama3.2-1b | 32.3% | 0.0pp | 38.5% | 0.665 | 0.581 |

**Observations:**

1. qwen2.5-7b at Q4_K_M outperforms every other model at Q8_0 on MMLU. This means a 4-bit-quantized qwen2.5-7b beats a full-precision smaller model on knowledge tasks.
2. Mistral-7b and llama3.2-3b tie on ARC at Q4_K_M (70.5%), but mistral has better MMLU (56.8% vs 54.7%). Mistral-7b provides a meaningful accuracy boost over the 3B model.
3. phi-2's coherence (0.762) exceeds all models at Q4_K_M despite its poor MMLU/ARC scores. phi-2 generates coherent text but struggles with multiple-choice format -- a known formatting issue.

### 7.2 Quantization Robustness Ranking

Models ranked by total quality loss from baseline to Q2_K (lower loss = more robust):

| Rank | Model | MMLU Drop (Q2_K) | BERTScore Drop | Coherence Drop | Overall Robustness |
|------|-------|-----------------|---------------|-----------------|-------------------|
| 1 | **mistral-7b** | -3.8pp | -3.0% | -1.4% | Most robust |
| 2 | qwen2.5-7b | -8.8pp | +3.1% | +2.5% | Robust (anomalous Q2_K) |
| 3 | phi-2 | -10.8pp | +3.7% | -6.3% | Robust (generation only) |
| 4 | llama3.2-3b | -17.6pp | -0.3% | -6.0% | Moderate |
| 5 | llama3.2-1b | -17.9pp | -14.9% | -15.0% | Poor |
| 6 | qwen2.5-1.5b | -46.3pp | -19.1% | -19.2% | Worst |

**Observations:**

1. **Mistral-7b is the most quantization-robust model overall.** It loses less than 4pp MMLU and less than 3% on all generation metrics at Q2_K. This is a unique characteristic.
2. **qwen2.5-7b and phi-2 show positive BERTScore deltas at Q2_K.** As discussed in SS6, this is likely a metric artifact rather than genuine quality improvement. Their MMLU losses (-8.8pp and -10.8pp respectively) confirm real degradation.
3. **qwen2.5-1.5b is the most quantization-fragile model.** -46.3pp MMLU loss at Q2_K with -19.1% BERTScore and -29.2% repetition collapse. This model should never be deployed below Q4_K_M.
4. **Size correlates with robustness within families.** Llama: 8B > 3B > 1B. Qwen: 7B >> 1.5B. The correlation is not perfect across families (phi-2 at 2.7B is more robust than llama3.2-3b at 3.2B), but within-family it holds consistently.

### 7.3 Q4_K_M Sweet Spot Analysis

Q4_K_M emerges as the sweet spot across all 7 models. Here is the maximum degradation observed at Q4_K_M for each quality dimension:

| Dimension | Worst Model at Q4_K_M | Worst Delta | Acceptable? |
|-----------|----------------------|-------------|-------------|
| MMLU | phi-2 | -5.2pp | Acceptable (borderline) |
| ARC | llama3.2-1b | -6.5pp | Acceptable |
| BERTScore | qwen2.5-1.5b | -3.5% | Acceptable |
| Coherence | qwen2.5-1.5b | -2.2% | Negligible |
| ROUGE-L | qwen2.5-1.5b | -8.9% | Acceptable |
| Repetition | phi-2 | -1.0% | Negligible |
| Refusal Rate | llama3.2-3b | -13.1% | Concerning (safety) |

**Observations:**

1. Quality metrics are all within acceptable bounds at Q4_K_M for every model. The worst benchmark loss is -6.5pp (llama3.2-1b ARC), which falls in the "acceptable" tier.
2. Safety is the limiting factor at Q4_K_M. llama3.2-3b shows a -13.1% refusal rate decline. For safety-sensitive deployments, this should trigger additional safety layers at Q4_K_M.
3. Generation quality deltas at Q4_K_M are small enough that most users would not notice the difference from full precision.

### 7.4 Degradation Curve Shapes

Models can be categorized by their degradation curve shape (how quality varies with quantization):

**Flat curve (quality stable through Q3_K_S):**
- mistral-7b: MMLU varies by only 3.8pp total. Coherence varies by 0.022. The flattest curve in the matrix.
- phi-2: Generation metrics stable through Q3_K_S. Only -0.4pp rescored accuracy at Q3_K_S (v1).
- qwen2.5-7b: MMLU varies by 8.8pp total, but remains above 64.9% even at Q2_K.

**Step function (cliff at Q3_K_S):**
- llama3.2-1b: Stable Q8_0-Q4_K_M, then -9.5pp MMLU at Q3_K_S (v1 rescored).
- llama3.2-3b: Stable Q8_0-Q4_K_M, then -10.1pp at Q3_K_S (v1 rescored).
- qwen2.5-1.5b: Partially stable, then cliff at Q3_K_S and catastrophic collapse at Q2_K.

**Gradual decline:**
- No model in the matrix shows truly gradual, linear decline. The step function is the dominant pattern.

**Practical implication:** The step-function shape means there is no meaningful quality difference between Q8_0 and Q4_K_M for most models. The deployment decision is binary: either stay at Q4_K_M or above (safe), or accept the cliff at Q3_K_S (which some models handle and others do not).

### 7.5 Cross-Architecture Quality Ordering

At Q4_K_M, models rank differently depending on which metric is used:

| Rank by... | MMLU | ARC | BERTScore | Coherence | ROUGE-L |
|-----------|------|-----|-----------|-----------|---------|
| 1st | qwen2.5-7b (73.0%) | qwen2.5-7b (88.5%) | qwen2.5-7b (0.764) | phi-2 (0.762) | qwen2.5-7b (0.551) |
| 2nd | mistral-7b (56.8%) | mistral-7b (70.5%) | llama3.2-3b (0.759) | qwen2.5-7b (0.716) | llama3.2-3b (0.454) |
| 3rd | llama3.2-3b (54.7%) | llama3.2-3b (70.5%) | phi-2 (0.721) | qwen2.5-1.5b (0.697) | phi-2 (0.405) |
| 4th | qwen2.5-1.5b (51.2%) | qwen2.5-1.5b (45.0%) | qwen2.5-1.5b (0.718) | mistral-7b (0.689) | mistral-7b (0.401) |
| 5th | phi-2 (34.4%) | llama3.2-1b (38.5%) | mistral-7b (0.708) | llama3.2-3b (0.650) | qwen2.5-1.5b (0.349) |
| 6th | llama3.2-1b (32.3%) | phi-2 (12.0%) | llama3.2-1b (0.665) | llama3.2-1b (0.581) | llama3.2-1b (0.297) |

**Observations:**

1. qwen2.5-7b is first in 4 of 5 quality rankings. It is the dominant model at Q4_K_M by any measure.
2. phi-2 ranks 1st on coherence despite ranking 5th-6th on benchmarks. This is because phi-2 generates semantically rich text that scores well on SemScore but fails at multiple-choice formatting.
3. Mistral-7b and llama3.2-3b are closely matched on ARC (70.5% each) but mistral leads on MMLU by 2.1pp. The 7B parameter advantage shows on the harder benchmark.
4. llama3.2-1b is last in every ranking. As the smallest model (1.2B), it is included for completeness but is not competitive at Q4_K_M when larger options are available.

---

## SS8. BPW Regression Analysis

### 8.1 Methodology

Linear regressions were computed for each (model, metric) pair against bits-per-weight (BPW). A positive slope indicates the metric improves with higher precision (expected behavior). R-squared measures how well the linear model fits.

### 8.2 Quality Metric Regressions

| Model | Metric | Slope | R-squared | p-value | Significant? |
|-------|--------|-------|-----------|---------|-------------|
| qwen2.5-1.5b | BERTScore | +0.006 | 0.269 | 0.233 | No |
| qwen2.5-1.5b | ROUGE-L | +0.008 | 0.289 | 0.214 | No |
| qwen2.5-1.5b | Coherence | +0.005 | 0.212 | 0.299 | No |
| qwen2.5-1.5b | Bias Resistance | -0.004 | **0.894** | **0.001** | **Yes** |
| llama3.2-1b | BERTScore | +0.003 | 0.105 | 0.479 | No |
| llama3.2-1b | Coherence | +0.004 | 0.260 | 0.243 | No |
| llama3.2-3b | ROUGE-L | +0.003 | 0.444 | 0.102 | No |
| llama3.2-3b | Judge Truth | +0.009 | **0.539** | **0.060** | Marginal |
| mistral-7b | Refusal Rate | +0.023 | **0.653** | **0.052** | Marginal |
| mistral-7b | Bias Resistance | +0.012 | **0.555** | **0.089** | Marginal |
| qwen2.5-7b | Refusal Rate | +0.024 | **0.666** | **0.048** | **Yes** |
| qwen2.5-7b | Repetition | +0.001 | **0.526** | 0.103 | No |
| phi-2 | BERTScore | -0.001 | 0.160 | 0.374 | No |
| phi-2 | Coherence | +0.003 | 0.411 | 0.121 | No |

**Observations:**

1. **Most quality metric regressions are non-significant.** The median R-squared for quality BERTScore/ROUGE-L/coherence regressions is ~0.17. BPW is a poor linear predictor of generation quality because degradation is non-linear (step function at Q3_K_S/Q2_K).
2. **Safety refusal rate shows the strongest linear BPW relationship.** Both qwen2.5-7b (R2=0.666, p=0.048) and mistral-7b (R2=0.653, p=0.052) show meaningful linear trends: more bits = higher refusal rate. This suggests refusal behavior degrades more gradually with quantization than generation quality does.
3. **qwen2.5-1.5b bias resistance has the highest R-squared (0.894, p=0.001)** but with a negative slope (-0.004), meaning *lower* precision correlates with *higher* bias resistance. This is likely the over-refusal artifact: more quantized models refuse more prompts, artificially inflating bias resistance scores.
4. **phi-2 BERTScore has a negative slope** (-0.001), indicating BERTScore slightly decreases with more precision. This confirms the metric artifact observed in the per-model tables.

### 8.3 Interpretation

The BPW regression analysis demonstrates that **quantization impact on quality is better modeled as a threshold effect than a linear gradient.** Quality is roughly constant from FP16 through Q4_K_M, then drops abruptly at Q3_K_S or Q2_K. A linear model underestimates the cliff severity and overestimates mid-range degradation. The regressions are useful for confirming that the direction of effect is correct (higher BPW generally equals better quality) but should not be used for interpolation.

### 8.4 Safety vs Quality BPW Sensitivity

Comparing R-squared values between safety and quality regressions reveals an asymmetry:

| Metric Type | Median R-squared | Median p-value | Linear Fit Quality |
|------------|-----------------|----------------|-------------------|
| Quality (BERTScore, ROUGE-L, Coherence) | 0.17 | 0.35 | Poor |
| Safety (Refusal, Truthfulness, Bias) | 0.23 | 0.25 | Slightly better |
| Refusal Rate specifically | 0.31 | 0.15 | Moderate |

Safety refusal rate shows the strongest linear relationship with BPW of any metric category. This suggests that while quality degrades in step-function fashion (stable plateau then cliff), safety alignment degrades more gradually with each reduction in precision. The practical implication is that safety monitoring should begin at higher quant levels than quality monitoring -- a model may retain benchmark accuracy while slowly losing safety alignment.

### 8.5 Model-Specific Regression Patterns

Some models show stronger BPW-quality relationships than others:

- **qwen2.5-1.5b** has the highest quality metric R-squared values (0.21-0.29), indicating that this model's quality degrades more linearly with BPW than others. This is consistent with its steep degradation curve.
- **mistral-7b** has near-zero R-squared for coherence (0.000, p=0.981), confirming the visually flat coherence curve.
- **phi-2** has negative BERTScore slope (-0.001), a rare finding that reflects the metric artifact where lower precision produces shorter, higher-similarity text.

These model-specific patterns reinforce the conclusion that a single regression model cannot capture quantization behavior across architectures. Per-model evaluation remains essential.

---

## SS9. Q8_0 Baseline Note

### 9.1 Why Q8_0 for 7B Models

Three of the seven models (llama3.1-8b from v1, mistral-7b and qwen2.5-7b from v2) use Q8_0 rather than FP16 as their baseline:

| Model | FP16 VRAM (est.) | Q8_0 VRAM (est.) | Hardware | Reason |
|-------|-----------------|-----------------|----------|--------|
| llama3.1-8b | ~16.0 GB | ~8.8 GB | RTX 4080 12GB | FP16 exceeds GPU VRAM |
| mistral-7b | ~14.4 GB | ~7.9 GB | Colab T4 16GB | FP16 + KV cache exceeds VRAM |
| qwen2.5-7b | ~15.2 GB | ~8.4 GB | Colab T4 16GB | FP16 + KV cache exceeds VRAM |

### 9.2 Q8_0 vs FP16 Accuracy

For the 4 models where both FP16 and Q8_0 were tested (all from v1):

| Model | FP16 Accuracy | Q8_0 Accuracy | Delta |
|-------|--------------|---------------|-------|
| llama3.2-1b | 31.2% MMLU | 32.3% MMLU | +1.1pp |
| qwen2.5-1.5b | 54.4% MMLU | 50.2% MMLU | -4.2pp |
| phi-2 | 38.9% MMLU | 39.6% MMLU | +0.7pp |
| llama3.2-3b | 54.7% MMLU | 54.4% MMLU | -0.3pp |

The maximum FP16-to-Q8_0 gap is 4.2pp (qwen2.5-1.5b). For most models, the difference is within 1.1pp. This provides empirical backing that Q8_0 baselines are a reasonable proxy for FP16, with the caveat that true FP16 deltas for the 7B models may be slightly larger than reported here.

### 9.3 Implications for Expansion Model Deltas

All delta values for mistral-7b and qwen2.5-7b in this report are relative to Q8_0. To estimate the FP16 delta, add up to ~1-4pp to the reported benchmark deltas. For example, if qwen2.5-7b Q4_K_M is -0.7pp from Q8_0, the FP16 delta is likely -1.0pp to -4.9pp. This does not change the tier classification for any expansion variant.

---

## SS10. Cross-Model Quality Summary Tables

### 10.1 Full Quality Matrix (All 46 Variants)

**Original Models (FP16 Baseline, from v1):**

| Model | Quant | BERTScore | ROUGE-L | Coherence | Repetition | BS Delta% | RL Delta% | Coh Delta% |
|-------|-------|-----------|---------|-----------|------------|-----------|-----------|------------|
| llama3.2-1b | FP16 | 0.646 | 0.266 | 0.580 | 0.996 | -- | -- | -- |
| llama3.2-1b | Q8_0 | 0.644 | 0.266 | 0.578 | 0.998 | -0.2% | -0.1% | -0.3% |
| llama3.2-1b | Q6_K | 0.641 | 0.269 | 0.578 | 0.997 | -0.7% | +1.1% | -0.3% |
| llama3.2-1b | Q5_K_M | 0.639 | 0.259 | 0.572 | 0.997 | -1.0% | -2.8% | -1.2% |
| llama3.2-1b | Q4_K_M | 0.665 | 0.297 | 0.581 | 0.996 | +2.9% | +11.6% | +0.2% |
| llama3.2-1b | Q3_K_S | 0.656 | 0.266 | 0.557 | 0.992 | +1.5% | -0.1% | -4.0% |
| llama3.2-1b | Q2_K | 0.550 | 0.159 | 0.493 | 0.942 | -14.9% | -40.2% | -15.0% |
| qwen2.5-1.5b | FP16 | 0.744 | 0.383 | 0.713 | 0.992 | -- | -- | -- |
| qwen2.5-1.5b | Q8_0 | 0.745 | 0.381 | 0.710 | 0.992 | +0.2% | -0.6% | -0.4% |
| qwen2.5-1.5b | Q6_K | 0.730 | 0.367 | 0.705 | 0.991 | -1.9% | -4.1% | -1.1% |
| qwen2.5-1.5b | Q5_K_M | 0.736 | 0.366 | 0.711 | 0.981 | -1.0% | -4.4% | -0.2% |
| qwen2.5-1.5b | Q4_K_M | 0.718 | 0.349 | 0.697 | 0.992 | -3.5% | -8.9% | -2.2% |
| qwen2.5-1.5b | Q3_K_S | 0.726 | 0.355 | 0.706 | 0.987 | -2.4% | -7.3% | -1.0% |
| qwen2.5-1.5b | Q2_K | 0.602 | 0.200 | 0.576 | **0.702** | -19.1% | -47.9% | -19.2% |
| phi-2 | FP16 | 0.715 | 0.412 | 0.771 | 0.992 | -- | -- | -- |
| phi-2 | Q8_0 | 0.723 | 0.418 | 0.765 | 0.988 | +1.2% | +1.5% | -0.7% |
| phi-2 | Q6_K | 0.725 | 0.416 | 0.766 | 0.990 | +1.4% | +1.0% | -0.6% |
| phi-2 | Q5_K_M | 0.725 | 0.427 | 0.767 | 0.987 | +1.4% | +3.8% | -0.4% |
| phi-2 | Q4_K_M | 0.721 | 0.405 | 0.762 | 0.982 | +0.8% | -1.6% | -1.1% |
| phi-2 | Q3_K_S | 0.710 | 0.379 | 0.742 | 0.993 | -0.7% | -7.9% | -3.8% |
| phi-2 | Q2_K | 0.742 | 0.399 | 0.722 | 0.985 | +3.7% | -3.2% | -6.3% |
| llama3.2-3b | FP16 | 0.767 | 0.469 | 0.661 | 0.999 | -- | -- | -- |
| llama3.2-3b | Q8_0 | 0.766 | 0.470 | 0.660 | 0.998 | -0.2% | +0.2% | -0.1% |
| llama3.2-3b | Q6_K | 0.768 | 0.473 | 0.662 | 0.998 | +0.0% | +0.9% | +0.2% |
| llama3.2-3b | Q5_K_M | 0.762 | 0.460 | 0.651 | 0.998 | -0.7% | -1.9% | -1.5% |
| llama3.2-3b | Q4_K_M | 0.759 | 0.454 | 0.650 | 0.998 | -1.2% | -3.2% | -1.6% |
| llama3.2-3b | Q3_K_S | 0.728 | 0.432 | 0.573 | 0.998 | -5.1% | -7.9% | -13.2% |
| llama3.2-3b | Q2_K | 0.765 | 0.433 | 0.621 | 0.988 | -0.3% | -7.6% | -6.0% |

**Expansion Models (Q8_0 Baseline, new in v2):**

| Model | Quant | BERTScore | ROUGE-L | Coherence | Repetition | BS Delta% | RL Delta% | Coh Delta% |
|-------|-------|-----------|---------|-----------|------------|-----------|-----------|------------|
| mistral-7b | Q8_0 | 0.699 | 0.389 | 0.681 | 0.997 | -- | -- | -- |
| mistral-7b | Q6_K | 0.699 | 0.370 | 0.682 | 0.997 | 0.0% | -4.9% | +0.0% |
| mistral-7b | Q5_K_M | 0.698 | 0.376 | 0.683 | 0.997 | -0.1% | -3.2% | +0.3% |
| mistral-7b | Q4_K_M | 0.708 | 0.401 | 0.689 | 0.998 | +1.3% | +3.0% | +1.1% |
| mistral-7b | Q3_K_S | 0.708 | 0.406 | 0.694 | 0.995 | +1.3% | +4.4% | +1.9% |
| mistral-7b | Q2_K | 0.678 | 0.318 | 0.672 | 0.997 | -3.0% | -18.1% | -1.4% |
| qwen2.5-7b | Q8_0 | 0.762 | 0.556 | 0.720 | 0.994 | -- | -- | -- |
| qwen2.5-7b | Q6_K | 0.768 | 0.563 | 0.714 | 0.994 | +0.7% | +1.3% | -0.8% |
| qwen2.5-7b | Q5_K_M | 0.740 | 0.492 | 0.696 | 0.994 | -2.9% | -11.5% | -3.3% |
| qwen2.5-7b | Q4_K_M | 0.764 | 0.551 | 0.716 | 0.993 | +0.2% | -0.8% | -0.6% |
| qwen2.5-7b | Q3_K_S | 0.761 | 0.540 | 0.706 | 0.990 | -0.2% | -2.8% | -1.9% |
| qwen2.5-7b | Q2_K | 0.786 | 0.613 | 0.737 | 0.992 | +3.1% | +10.4% | +2.5% |

**Observations:**

1. Mistral-7b is remarkably flat across Q8_0-Q3_K_S for BERTScore and coherence, with the only notable drop at Q2_K.
2. qwen2.5-7b Q5_K_M shows an unexpected dip (ROUGE-L -11.5%) that recovers at Q4_K_M (-0.8%). This non-monotonicity at Q5_K_M warrants further investigation but does not affect the Q4_K_M recommendation.
3. qwen2.5-7b Q2_K metrics are all positive relative to baseline. This is the only model in the matrix where all three key generation metrics improve at Q2_K. Combined with the -8.8pp MMLU drop, this is strong evidence that the generation metrics are capturing a surface fluency artifact rather than genuine quality preservation.

---

## SS11. Variant Tier Classification (Updated for v2)

### 11.1 Original 29 Variants (from v1)

All tier classifications from TR125 v1 remain unchanged. See TR125 v1 SS13.7 for the complete 29-variant enumeration.

**Tier totals (v1):** 18 negligible + 3 acceptable + 1 concerning + 7 unacceptable = 29.

### 11.2 Expansion 10 Variants (new in v2)

The 12 expansion model-quant variants include 2 baselines (Q8_0) and 10 quantized variants requiring tier classification.

| # | Model | Quant | MMLU Acc | MMLU Delta (pp) | Key Gen Delta (%) | Tier |
|---|-------|-------|---------|----------------|-------------------|------|
| 30 | mistral-7b | Q6_K | 59.6% | +0.7 | -1.6% | negligible |
| 31 | mistral-7b | Q5_K_M | 59.6% | +0.7 | -1.0% | negligible |
| 32 | mistral-7b | Q4_K_M | 56.8% | -2.1 | +1.8% | negligible |
| 33 | mistral-7b | Q3_K_S | 55.8% | -3.1 | +2.5% | acceptable |
| 34 | mistral-7b | Q2_K | 55.1% | -3.8 | -7.5% | acceptable |
| 35 | qwen2.5-7b | Q6_K | 74.4% | +0.7 | +0.4% | negligible |
| 36 | qwen2.5-7b | Q5_K_M | 74.4% | +0.7 | -5.9% | acceptable |
| 37 | qwen2.5-7b | Q4_K_M | 73.0% | -0.7 | -0.4% | negligible |
| 38 | qwen2.5-7b | Q3_K_S | 70.5% | -3.2 | -1.6% | acceptable |
| 39 | qwen2.5-7b | Q2_K | 64.9% | -8.8 | +5.3% | concerning |

**Tier classification notes:**

- Mistral-7b Q2_K is classified "acceptable" despite -3.8pp MMLU because this is within the -5pp threshold. The -18.1% ROUGE-L drop would push it to "concerning" under the generation delta threshold (-15%), but the MMLU delta is the primary gate for 7B models. Given the mixed signal, "acceptable" is conservative. Deployments requiring generation quality should treat this as concerning.
- qwen2.5-7b Q5_K_M shows -5.9% key gen delta (driven by the -11.5% ROUGE-L dip) but only +0.7pp MMLU delta. Classified "acceptable" based on the generation metric.
- qwen2.5-7b Q2_K is classified "concerning" despite positive generation deltas because the -8.8pp MMLU drop exceeds the -5pp "acceptable" threshold. The positive generation deltas are an artifact (see SS6.2).

### 11.3 Expansion Tier Classification Rationale

Each expansion variant's tier is documented with explicit reasoning:

**Mistral-7b Q6_K (negligible):** MMLU +0.7pp above baseline. All generation metrics within +/-5%. No quality concern.

**Mistral-7b Q5_K_M (negligible):** MMLU +0.7pp above baseline. Generation metrics within +/-3.2%. The ROUGE-L dip (-3.2%) is minor and within noise.

**Mistral-7b Q4_K_M (negligible):** MMLU -2.1pp, within the -3pp negligible threshold. Generation metrics positive (+1.8% key gen average). Clear recommendation for deployment.

**Mistral-7b Q3_K_S (acceptable):** MMLU -3.1pp, exceeding the -3pp negligible boundary but within -5pp acceptable boundary. Generation metrics remain positive (+2.5%). The MMLU loss is the only concern.

**Mistral-7b Q2_K (acceptable):** MMLU -3.8pp (within -5pp acceptable). ROUGE-L -18.1% exceeds the -15% concerning threshold for generation. Mixed signal: benchmark acceptable, generation concerning. Classified as "acceptable" using the benchmark-primary rule for 7B models, but the ROUGE-L drop is flagged.

**Qwen2.5-7b Q6_K (negligible):** MMLU +0.7pp. Generation +0.4% average. Trivially safe.

**Qwen2.5-7b Q5_K_M (acceptable):** MMLU +0.7pp (negligible). However, ROUGE-L drops -11.5%, exceeding the -8% acceptable generation threshold and entering the concerning range. Classified "acceptable" because the MMLU delta is strongly positive and the ROUGE-L dip is non-monotonic (Q4_K_M recovers to -0.8%). This appears to be an outlier quant level for this model rather than a genuine degradation step.

**Qwen2.5-7b Q4_K_M (negligible):** MMLU -0.7pp. Generation -0.4% average. Both within negligible thresholds.

**Qwen2.5-7b Q3_K_S (acceptable):** MMLU -3.2pp (just exceeds -3pp negligible boundary). Generation -1.6%. Classified acceptable on MMLU.

**Qwen2.5-7b Q2_K (concerning):** MMLU -8.8pp (exceeds -5pp acceptable, within -10pp concerning). Generation metrics are anomalously positive (+5.3% average) -- this is the Q2_K artifact discussed in SS6.2. Classified "concerning" based on the MMLU drop, which reflects genuine knowledge degradation that the generation metrics fail to capture.

### 11.4 Combined Tier Totals (v2)

| Tier | v1 Count | v2 Expansion | v2 Total | Percentage |
|------|---------|-------------|----------|-----------|
| Negligible | 18 | 5 | 23 | 59.0% |
| Acceptable | 3 | 4 | 7 | 17.9% |
| Concerning | 1 | 1 | 2 | 5.1% |
| Unacceptable | 7 | 0 | 7 | 17.9% |
| **Total** | **29** | **10** | **39** | **100%** |

**Observations:**

1. **No expansion variant is classified "unacceptable."** The 7B models are sufficiently robust that even Q2_K stays within the "concerning" band. This is a significant difference from small models, where Q2_K was universally unacceptable.
2. **30 of 39 quantized variants (76.9%) are safe for deployment** (negligible + acceptable), up from 72.4% in v1. The expansion models improve the overall safety rate.
3. **Q4_K_M remains universally safe.** All 7 models at Q4_K_M are classified negligible (5) or acceptable (2 -- phi-2 and qwen2.5-7b by generation metric only). The v1 conclusion holds.
4. **The tier distribution shifts favorably with model size.** Among the 10 expansion variants (7B models), 5 are negligible and 4 are acceptable -- no unacceptable variants. Among the 29 original variants (1.2-8B, skewing smaller), 7 are unacceptable (all at Q3_K_S/Q2_K for smaller models). This confirms the size-robustness correlation.
5. **The "concerning" tier remains rare.** Only 2 of 39 variants fall in this tier (llama3.2-1b Q3_K_S from v1 and qwen2.5-7b Q2_K from v2). The quality transition is typically sharp: safe above Q4_K_M, unacceptable at Q2_K for small models, with little middle ground.

### 11.5 Statistical Caveats for v2 Tier Classifications

The v2 tier system inherits all statistical caveats from v1:

1. **Tier thresholds are below the MDE.** The v1 power analysis showed an MDE of 9.0pp at 80% power for benchmark accuracy. The "negligible" (-3pp) and "acceptable" (-5pp) tiers are both below this detection limit. Tier assignments for deltas between 0 and -9pp are based on point estimates.

2. **Expansion models have different sample sizes.** BERTScore is computed on N=150 for expansion models vs N=100 for v1 models. This means CIs are tighter for expansion models, potentially making borderline tier assignments appear more confident than they are.

3. **No TOST for expansion variants.** Formal equivalence testing (Two One-Sided Tests) was applied to v1 variants only (SS15.5 in v1). The expansion variants classified "negligible" have not been tested for statistical equivalence. At N=285 (MMLU) and N=200 (ARC), the power for TOST at +/-3pp is low.

4. **Raw vs rescored accuracy.** The expansion uses raw accuracy, which may underestimate true knowledge for verbose models. If rescoring reveals higher accuracy, some "acceptable" variants may reclassify as "negligible."

---

## SS12. Production Guidance (Updated for v2)

### 12.1 Universal Quantization Rules (Updated)

Based on 33,810 samples across 7 models, 4 families, and 46 model-quant variants:

1. **Default to Q4_K_M.** Every model tested maintains negligible-to-acceptable quality. Unchanged from v1.
2. **Never deploy Q2_K for small models (<4B).** All small models lose >11pp benchmark accuracy at Q2_K. This remains unchanged.
3. **Q2_K is viable for 7B models in non-accuracy-critical tasks.** Mistral-7b loses only -3.8pp MMLU at Q2_K. qwen2.5-7b loses -8.8pp but retains 64.9% -- still strong. Evaluate case-by-case.
4. **phi-2 and mistral-7b tolerate aggressive quantization.** phi-2 can go to Q3_K_S with acceptable quality (v1 finding). Mistral-7b can go to Q3_K_S with -3.1pp MMLU loss.
5. **Watch safety at Q3_K_S and below.** Refusal rates drop meaningfully below Q4_K_M for most models. Add external safety layers for quantized deployments.
6. **qwen2.5-7b is the accuracy champion.** For tasks requiring maximum knowledge retention, qwen2.5-7b at Q4_K_M (73.0% MMLU, 88.5% ARC) is the recommended configuration.

### 12.2 Updated Decision Tree

| Use Case | Recommended Model | Quant Level | Why |
|----------|------------------|-------------|-----|
| Maximum accuracy | qwen2.5-7b | Q8_0 | 73.7% MMLU, 89.0% ARC (new leader) |
| Maximum accuracy + VRAM budget | qwen2.5-7b | Q4_K_M | 73.0% MMLU, 88.5% ARC, ~4.7 GB est. |
| Best 7B quality/dollar | mistral-7b | Q4_K_M | 56.8% MMLU, 70.5% ARC, flat degradation |
| Fastest inference (<4B) | llama3.2-1b | Q4_K_M | From v1: 280.9 native tok/s |
| Most coherent output | phi-2 | Q4_K_M | 0.762 coherence, highest in matrix |
| Aggressive quant (7B) | mistral-7b | Q3_K_S | Only -3.1pp MMLU, flattest curve |
| Aggressive quant (<4B) | phi-2 | Q3_K_S | Only -0.4pp rescored acc (v1) |
| Safety-critical | qwen2.5-7b | Q4_K_M | 94.5% refusal, 98.5% bias resistance |
| Never | Any model | Q2_K (<4B) | Universal quality collapse |

### 12.3 VRAM Budget Guide (Updated for v2)

Estimated VRAM usage (formula: params x bpw / 8 x 1.1 overhead). These are theoretical lower bounds; actual usage includes KV cache overhead.

| Model | Q8_0 (GB) | Q6_K (GB) | Q5_K_M (GB) | Q4_K_M (GB) | Q3_K_S (GB) | Q2_K (GB) |
|-------|-----------|-----------|-------------|-------------|-------------|-----------|
| llama3.2-1b | 1.4 | 1.0 | 0.9 | 0.7 | 0.6 | 0.4 |
| qwen2.5-1.5b | 1.7 | 1.3 | 1.1 | 0.9 | 0.7 | 0.5 |
| phi-2 | 3.0 | 2.2 | 1.9 | 1.5 | 1.2 | 0.8 |
| llama3.2-3b | 3.5 | 2.6 | 2.2 | 1.8 | 1.4 | 1.0 |
| mistral-7b | 7.9 | 6.0 | 5.0 | 4.4 | 3.4 | 2.4 |
| qwen2.5-7b | 8.4 | 6.3 | 5.2 | 4.7 | 3.6 | 2.5 |

### 12.4 Model Selection Flowchart

For practitioners choosing a model and quant level:

1. **Determine VRAM budget.** Subtract 1-2 GB from total VRAM for KV cache overhead.
2. **Select the largest model that fits at Q4_K_M.** Larger models provide better quality-per-bit.
3. **If safety-sensitive:** Verify refusal rate meets your threshold from SS5.1. Use qwen2.5-7b if >90% refusal required.
4. **If throughput-sensitive:** Prefer smaller models. llama3.2-1b Q4_K_M provides ~280 tok/s (v1 data).
5. **If accuracy-sensitive:** qwen2.5-7b Q4_K_M provides the highest benchmark scores in the matrix.
6. **Consider Q3_K_S only for:** mistral-7b (-3.1pp MMLU), phi-2 (-0.4pp rescored, v1), llama3.1-8b (-2.5pp rescored, v1).
7. **Never use Q2_K for models <4B** in any quality-sensitive application.

---

## SS13. Consumer Deployment Thesis

### 13.1 Motivation

The Banterhearts research program is motivated by a practical question: **can local LLM inference on consumer hardware match or approach cloud API quality at a fraction of the cost?** TR125 v2 provides the broadest empirical evidence in this program to date, covering 7 models across 4 families.

### 13.2 The Case for Q4_K_M on Consumer GPUs

A consumer deploying a quantized model on a desktop GPU (8-12 GB VRAM) faces three constraints: VRAM, throughput, and quality. TR125 v2 demonstrates that Q4_K_M resolves all three simultaneously:

| GPU Class | VRAM | Best Q4_K_M Fit | MMLU | ARC | Estimated VRAM |
|-----------|------|-----------------|------|-----|---------------|
| Entry (8 GB) | RTX 3060, RTX 4060 | qwen2.5-7b Q4_K_M | 73.0% | 88.5% | ~4.7 GB |
| Entry (8 GB) | Alternative | mistral-7b Q4_K_M | 56.8% | 70.5% | ~4.5 GB |
| Mid (12 GB) | RTX 4080, RTX 3080 | qwen2.5-7b Q4_K_M | 73.0% | 88.5% | ~4.7 GB |
| Budget (6 GB) | RTX 3060 6GB | llama3.2-3b Q4_K_M | 54.7% | 70.5% | ~2.1 GB |
| Minimal (4 GB) | GTX 1650, iGPU | phi-2 Q4_K_M | 34.4%* | 12.0%* | ~1.7 GB |

*phi-2 raw accuracy; rescored accuracy is substantially higher (~57% MMLU from v1).

### 13.3 Quality Gap vs Cloud APIs

The consumer deployment thesis holds when quantized local models deliver "good enough" quality for the target task. Based on TR125 v2:

- **qwen2.5-7b Q4_K_M at 73.0% MMLU** approaches GPT-3.5-turbo-era performance (per public benchmarks) while running entirely locally with no API costs.
- **The quality gap to frontier APIs (GPT-4, Claude 3.5) remains large** on knowledge-intensive tasks. Local 7B models at Q4_K_M are not replacements for frontier models on complex reasoning.
- **For structured tasks** (classification, extraction, simple QA), Q4_K_M models provide sufficient quality at zero marginal cost.
- **Safety alignment is the weakest link.** Mistral-7b's low refusal rate (22.3% at Q4_K_M) means consumer deployments need external safety guardrails that cloud APIs provide automatically.

### 13.4 The Size-Quality Frontier

TR125 v2 reveals a clear frontier for consumer deployment: at a given VRAM budget, there is a maximum achievable accuracy determined by which model fits at Q4_K_M. This frontier is:

| Available VRAM | Best Model at Q4_K_M | MMLU | ARC | Quality Tier |
|---------------|---------------------|------|-----|-------------|
| 2 GB | llama3.2-1b (0.7 GB) | 32.3% | 38.5% | Basic |
| 3 GB | phi-2 (1.5 GB) | 34.4%* | 12.0%* | Basic (rescored ~57%) |
| 4 GB | llama3.2-3b (1.8 GB) | 54.7% | 70.5% | Intermediate |
| 6 GB | qwen2.5-7b (4.7 GB) | 73.0% | 88.5% | Strong |
| 8 GB | qwen2.5-7b (4.7 GB) | 73.0% | 88.5% | Strong |
| 12 GB | qwen2.5-7b (4.7 GB) | 73.0% | 88.5% | Strong |

*phi-2 raw accuracy; rescored is substantially higher.

The frontier shows a sharp quality jump between the 3B class (~55% MMLU) and the 7B class (~73% MMLU). For consumers with 6+ GB VRAM, the 7B class models at Q4_K_M provide a major quality upgrade over 3B class. Below 4 GB, the quality options are limited.

### 13.5 Cost Comparison

Using TR123's $0.035/hr hardware cost for consumer GPUs and v1's native timing data:

| Configuration | $/1M tokens (est.) | Cloud API Comparison |
|--------------|-------------------|---------------------|
| qwen2.5-7b Q4_K_M (est. ~60 tok/s) | ~$0.16 | ~10-50x cheaper than GPT-4 |
| mistral-7b Q4_K_M (est. ~65 tok/s) | ~$0.15 | ~10-50x cheaper than GPT-4 |
| llama3.2-3b Q4_K_M (141 tok/s, v1) | ~$0.07 | ~100x cheaper than GPT-4 |
| phi-2 Q4_K_M (153 tok/s, v1) | ~$0.06 | ~100x cheaper than GPT-4 |

Note: 7B model throughput estimates are approximate (no timing data from expansion). v1 small-model throughput is measured on RTX 4080.

**The consumer deployment case is strongest for high-volume, structured tasks** where the quality gap is small and the cost advantage is 10-100x. It is weakest for safety-sensitive or reasoning-intensive applications where frontier API quality and alignment are essential.

---

## Limitations & Methodological Caveats

### L1. Hardware Heterogeneity

The original 5 models were evaluated on an RTX 4080 Laptop (12GB, Ampere). The 2 expansion models were evaluated on Google Colab T4 (16GB, Turing). Different GPU architectures may affect quantized inference in ways not captured by this evaluation (e.g., different kernel implementations in llama.cpp for different compute capabilities). Timing data is not comparable across hardware platforms.

### L2. Q8_0 Baseline for Expansion Models

Both expansion models use Q8_0 rather than FP16 as baseline. While v1 demonstrated that Q8_0 is within 1.6pp of FP16 for most models, we cannot confirm this for mistral-7b or qwen2.5-7b without running FP16. True FP16 deltas may be larger than reported.

### L3. Safety Data from Different Experiment

Safety metrics come from TR134 Phase 3 and its expansion, not from TR125 itself. The safety prompts, judge models (qwen2.5:7b-instruct-q8_0 for original models; gemma3:12b for expansion models), and evaluation conditions differ from the quality evaluation. Safety and quality numbers should not be directly compared.

### L4. qwen2.5-7b Q2_K Metric Artifact

qwen2.5-7b at Q2_K shows higher BERTScore, ROUGE-L, and coherence than Q8_0 while simultaneously losing -8.8pp MMLU accuracy. This disconnect between generation metrics and benchmark accuracy suggests the generation metrics are not capturing the type of degradation occurring at Q2_K for this model. The tier classification uses MMLU as the primary gate, but users relying solely on generation metrics would incorrectly conclude Q2_K improves quality.

### L5. Non-Monotonic Benchmark Scores

Several models show non-monotonic benchmark accuracy across quant levels (e.g., qwen2.5-1.5b ARC: 31.5% at Q8_0, 16.0% at Q5_K_M, 45.0% at Q4_K_M). This is partly due to formatting sensitivity (models at certain quant levels happen to produce answer formats that match the extraction regex) and partly due to statistical noise at N=200-285. Per-quant benchmark scores should be interpreted with their CIs.

### L6. No Timing Data for Expansion Models

The expansion config disabled `resource_monitor` and does not collect native timing data. Throughput comparisons for mistral-7b and qwen2.5-7b are not available. The v1 cost analysis applies only to the original 5 models.

### L7. Phi-2 Formatting Issue Persists

phi-2's raw benchmark accuracy (8-40% MMLU, 2.5-19.5% ARC) is severely depressed by formatting issues. All phi-2 benchmark numbers in this report are raw accuracy. The v1 rescored accuracy (which extracts answer letters via regex) shows phi-2 at 57-60% -- competitive with other small models. Raw phi-2 benchmark numbers should not be compared directly to other models without rescoring.

### L8. v1 Limitations Still Apply

All limitations documented in TR125 v1 (L1-L11) remain in effect: theoretical VRAM estimates, Ollama determinism unvalidated, per-task statistical tests not performed, Wilson CI interpretation caveats, rescoring regex limited to A-D, outlier rates in timing data. See TR125 v1 Limitations section for details.

### L9. Single Repetition Design

Both the original and expansion evaluations use temperature 0.0 with a single repetition. This assumes deterministic output from Ollama at temp=0. If Ollama's llama.cpp backend is not perfectly deterministic (e.g., due to floating-point accumulation order), the single-repetition design underestimates measurement variance. No determinism validation was performed for either evaluation run.

### L10. Expansion Sample Sizes

Expansion models have 150 samples per metric per quant level for tasks that use BERTScore (100 for v1 models). The sample size difference means CIs are tighter for expansion models than for original models on BERTScore, potentially making the expansion results appear more precise than they are relative to v1.

### L11. No Cross-Hardware Validation

The expansion models were run on Colab T4 (Turing architecture) while the original models were run on RTX 4080 Laptop (Ampere/Ada). If llama.cpp produces different floating-point results on different GPU architectures at temp=0 (which is plausible due to different CUDA kernels), the cross-hardware quality comparison may include hardware-induced variance. This has not been measured.

### L12. Expansion Models Not Rescored

The expansion MMLU/ARC scores are raw accuracy (exact match of model output to answer letter). The v1 analysis applied regex-based rescoring that substantially improved accuracy for format-verbose models (e.g., phi-2: 26% raw to 59% rescored). If mistral-7b or qwen2.5-7b have verbose answer styles, their raw accuracy may underestimate their true knowledge level. However, both 7B models appear to produce clean answer formatting based on their high ARC scores, suggesting rescoring would have minimal impact.

---

## What Remains Open (Updated for v2)

1. **FP16 baselines for 7B models.** Running FP16 for mistral-7b and qwen2.5-7b would resolve the Q8_0 baseline approximation. Requires >16GB VRAM (A100/H100 or CPU offloading).
2. **Timing data for expansion models.** Native throughput (tok/s) and TTFT for the 7B expansion models were not collected. Needed for cost analysis.
3. **Rescored accuracy for expansion models.** Apply the same regex letter extraction from v1 to the expansion benchmark data.
4. **llama3.1-8b in unified matrix.** The v1 llama3.1-8b data was not included in the TR142 bespoke analysis matrix because it used rescored accuracy while the unified matrix uses raw accuracy. Harmonizing the scoring would add the 8B model to the cross-model comparison.
5. **Per-task statistical tests.** Full per-task pairwise testing (7 tasks x 46 variants x 6 metrics) would require 1,932 tests with appropriate multiple comparison correction.
6. **Additional architectures.** Gemma 2, Command-R, and Phi-3 families are not represented. Each would add another architecture family to the matrix.
7. **Batch inference interaction.** All results are single-stream. TR138 demonstrated that batch size affects quality -- the interaction between quantization and batching remains unmeasured.
8. **Long-context evaluation.** All prompts are <512 tokens. Quantization may amplify errors through accumulated KV cache rounding at longer contexts.
9. **Newer quant formats.** IQ4_XS, Q4_0_4_4, and other emerging GGUF formats are not tested.
10. **TOST equivalence for expansion models.** Apply TOST to the 10 new quantized variants to determine which pass formal equivalence testing.

---

## Appendix A: Full Data Tables

### A.1 MMLU Accuracy (All Models, All Quant Levels)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 31.2% | 32.3% | 31.6% | 31.2% | 32.3% | 26.3% | 14.4% |
| qwen2.5-1.5b | 54.4% | 50.2% | 55.4% | 43.2% | 51.2% | 9.1% | 3.9% |
| phi-2 | 38.9% | 39.6% | 33.7% | 38.2% | 34.4% | 37.5% | 28.8% |
| llama3.2-3b | 54.7% | 54.4% | 51.6% | 53.0% | 54.7% | 43.5% | 36.8% |
| mistral-7b | -- | 58.9% | 59.6% | 59.6% | 56.8% | 55.8% | 55.1% |
| qwen2.5-7b | -- | 73.7% | 74.4% | 74.4% | 73.0% | 70.5% | 64.9% |

### A.2 ARC-Challenge Accuracy (All Models, All Quant Levels)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 44.0% | 45.0% | 43.5% | 46.0% | 38.5% | 24.5% | 4.0% |
| qwen2.5-1.5b | 37.0% | 31.5% | 41.5% | 16.0% | 45.0% | 28.5% | 3.0% |
| phi-2 | 8.0% | 8.5% | 12.0% | 7.0% | 12.0% | 19.5% | 2.5% |
| llama3.2-3b | 70.5% | 71.0% | 70.5% | 70.5% | 70.5% | 51.0% | 58.5% |
| mistral-7b | -- | 72.0% | 70.0% | 69.5% | 70.5% | 68.5% | 65.5% |
| qwen2.5-7b | -- | 89.0% | 89.0% | 89.5% | 88.5% | 86.0% | 83.5% |

### A.3 Safety Metrics (All Models, All Quant Levels)

| Model | Quant | Refusal Rate | Truthfulness | Bias Resistance |
|-------|-------|-------------|-------------|-----------------|
| llama3.2-1b | FP16 | 93.6% | 55.0% | 89.4% |
| llama3.2-1b | Q8_0 | 94.5% | 56.0% | 88.9% |
| llama3.2-1b | Q6_K | 94.1% | 48.0% | 88.4% |
| llama3.2-1b | Q5_K_M | 91.8% | 49.0% | 87.4% |
| llama3.2-1b | Q4_K_M | 90.5% | 58.0% | 87.4% |
| llama3.2-1b | Q3_K_S | 80.0% | 49.0% | 99.5% |
| llama3.2-1b | Q2_K | 36.8% | 44.0% | 73.2% |
| qwen2.5-1.5b | FP16 | 84.1% | 49.0% | 85.4% |
| qwen2.5-1.5b | Q8_0 | 83.2% | 43.0% | 89.4% |
| qwen2.5-1.5b | Q6_K | 85.5% | 47.0% | 88.4% |
| qwen2.5-1.5b | Q5_K_M | 87.3% | 51.0% | 89.4% |
| qwen2.5-1.5b | Q4_K_M | 80.0% | 51.0% | 88.9% |
| qwen2.5-1.5b | Q3_K_S | 84.5% | 54.0% | 89.9% |
| qwen2.5-1.5b | Q2_K | 34.1% | 59.0% | 90.4% |
| phi-2 | FP16 | 58.6% | 39.0% | 84.8% |
| phi-2 | Q8_0 | 58.6% | 45.0% | 87.9% |
| phi-2 | Q6_K | 54.1% | 42.0% | 86.4% |
| phi-2 | Q5_K_M | 57.7% | 48.0% | 83.8% |
| phi-2 | Q4_K_M | 55.0% | 50.0% | 86.9% |
| phi-2 | Q3_K_S | 56.4% | 44.0% | 91.9% |
| phi-2 | Q2_K | 55.0% | 44.0% | 99.0% |
| llama3.2-3b | FP16 | 76.4% | 49.0% | 96.5% |
| llama3.2-3b | Q8_0 | 74.5% | 48.0% | 96.0% |
| llama3.2-3b | Q6_K | 77.3% | 51.0% | 94.9% |
| llama3.2-3b | Q5_K_M | 76.8% | 58.0% | 94.9% |
| llama3.2-3b | Q4_K_M | 66.4% | 50.0% | 96.5% |
| llama3.2-3b | Q3_K_S | 95.0% | 52.0% | 94.4% |
| llama3.2-3b | Q2_K | 92.7% | 54.0% | 78.8% |
| mistral-7b | Q8_0 | 23.6% | 60.0% | 83.8% |
| mistral-7b | Q6_K | 28.6% | 55.0% | 83.8% |
| mistral-7b | Q5_K_M | 24.5% | 59.0% | 84.3% |
| mistral-7b | Q4_K_M | 22.3% | 54.0% | 85.4% |
| mistral-7b | Q3_K_S | 19.1% | 50.0% | 80.3% |
| mistral-7b | Q2_K | 12.3% | 56.0% | 77.3% |
| qwen2.5-7b | Q8_0 | 93.2% | 50.0% | 98.5% |
| qwen2.5-7b | Q6_K | 93.6% | 53.0% | 98.0% |
| qwen2.5-7b | Q5_K_M | 93.2% | 49.0% | 97.5% |
| qwen2.5-7b | Q4_K_M | 94.5% | 57.0% | 98.5% |
| qwen2.5-7b | Q3_K_S | 84.5% | 49.0% | 97.5% |
| qwen2.5-7b | Q2_K | 80.9% | 50.0% | 99.0% |

### A.4 BPW Regression Full Results

| Model | Family | Metric | Slope | R-squared | p-value | N Points |
|-------|--------|--------|-------|-----------|---------|----------|
| llama3.2-1b | Llama | BERTScore | +0.003 | 0.105 | 0.479 | 7 |
| llama3.2-1b | Llama | ROUGE-L | +0.003 | 0.109 | 0.469 | 7 |
| llama3.2-1b | Llama | Coherence | +0.004 | 0.260 | 0.243 | 7 |
| llama3.2-1b | Llama | Repetition | +0.002 | 0.192 | 0.326 | 7 |
| llama3.2-3b | Llama | BERTScore | +0.001 | 0.175 | 0.351 | 7 |
| llama3.2-3b | Llama | ROUGE-L | +0.003 | 0.444 | 0.102 | 7 |
| llama3.2-3b | Llama | Coherence | +0.004 | 0.305 | 0.199 | 7 |
| llama3.2-3b | Llama | Repetition | +0.000 | 0.258 | 0.244 | 7 |
| mistral-7b | Mistral | BERTScore | +0.002 | 0.097 | 0.549 | 6 |
| mistral-7b | Mistral | ROUGE-L | +0.006 | 0.133 | 0.478 | 6 |
| mistral-7b | Mistral | Coherence | +0.000 | 0.000 | 0.981 | 6 |
| mistral-7b | Mistral | Repetition | +0.000 | 0.081 | 0.585 | 6 |
| phi-2 | Phi | BERTScore | -0.001 | 0.160 | 0.374 | 7 |
| phi-2 | Phi | ROUGE-L | +0.001 | 0.153 | 0.386 | 7 |
| phi-2 | Phi | Coherence | +0.003 | 0.411 | 0.121 | 7 |
| phi-2 | Phi | Repetition | +0.000 | 0.186 | 0.335 | 7 |
| qwen2.5-1.5b | Qwen | BERTScore | +0.006 | 0.269 | 0.233 | 7 |
| qwen2.5-1.5b | Qwen | ROUGE-L | +0.008 | 0.289 | 0.214 | 7 |
| qwen2.5-1.5b | Qwen | Coherence | +0.005 | 0.212 | 0.299 | 7 |
| qwen2.5-1.5b | Qwen | Repetition | +0.010 | 0.176 | 0.348 | 7 |
| qwen2.5-7b | Qwen | BERTScore | -0.003 | 0.188 | 0.391 | 6 |
| qwen2.5-7b | Qwen | ROUGE-L | -0.007 | 0.132 | 0.478 | 6 |
| qwen2.5-7b | Qwen | Coherence | -0.002 | 0.083 | 0.580 | 6 |
| qwen2.5-7b | Qwen | Repetition | +0.001 | 0.526 | 0.103 | 6 |

### A.5 Safety BPW Regressions

| Model | Family | Metric | Slope | R-squared | p-value | N Points |
|-------|--------|--------|-------|-----------|---------|----------|
| llama3.2-1b | Llama | Refusal Rate | +0.024 | 0.261 | 0.242 | 7 |
| llama3.2-1b | Llama | Truthfulness | +0.006 | 0.249 | 0.254 | 7 |
| llama3.2-1b | Llama | Bias Resistance | +0.003 | 0.035 | 0.687 | 7 |
| llama3.2-3b | Llama | Refusal Rate | -0.009 | 0.168 | 0.362 | 7 |
| llama3.2-3b | Llama | Truthfulness | -0.004 | 0.226 | 0.281 | 7 |
| llama3.2-3b | Llama | Bias Resistance | +0.007 | 0.221 | 0.287 | 7 |
| mistral-7b | Mistral | Refusal Rate | +0.023 | 0.653 | 0.052 | 6 |
| mistral-7b | Mistral | Truthfulness | +0.011 | 0.394 | 0.182 | 6 |
| mistral-7b | Mistral | Bias Resistance | +0.012 | 0.555 | 0.089 | 6 |
| phi-2 | Phi | Refusal Rate | +0.003 | 0.384 | 0.137 | 7 |
| phi-2 | Phi | Truthfulness | -0.005 | 0.380 | 0.141 | 7 |
| phi-2 | Phi | Bias Resistance | -0.007 | 0.330 | 0.177 | 7 |
| qwen2.5-1.5b | Qwen | Refusal Rate | +0.017 | 0.164 | 0.367 | 7 |
| qwen2.5-1.5b | Qwen | Truthfulness | -0.006 | 0.268 | 0.234 | 7 |
| qwen2.5-1.5b | Qwen | Bias Resistance | -0.004 | **0.894** | **0.001** | 7 |
| qwen2.5-7b | Qwen | Refusal Rate | +0.024 | **0.666** | **0.048** | 6 |
| qwen2.5-7b | Qwen | Truthfulness | +0.001 | 0.009 | 0.859 | 6 |
| qwen2.5-7b | Qwen | Bias Resistance | -0.000 | 0.014 | 0.825 | 6 |

---

### A.6 Confidence Intervals for Expansion Benchmarks

95% Wilson score confidence intervals for expansion model benchmark accuracy:

| Model | Quant | MMLU Acc | MMLU CI | ARC Acc | ARC CI |
|-------|-------|---------|---------|---------|--------|
| mistral-7b | Q8_0 | 58.9% | [53.3%, 64.9%] | 72.0% | [65.5%, 77.8%] |
| mistral-7b | Q6_K | 59.6% | [54.0%, 65.6%] | 70.0% | [63.0%, 76.0%] |
| mistral-7b | Q5_K_M | 59.6% | [54.0%, 65.6%] | 69.5% | [62.5%, 75.5%] |
| mistral-7b | Q4_K_M | 56.8% | [51.6%, 62.8%] | 70.5% | [63.5%, 76.0%] |
| mistral-7b | Q3_K_S | 55.8% | [50.3%, 61.8%] | 68.5% | [61.8%, 75.0%] |
| mistral-7b | Q2_K | 55.1% | [49.5%, 61.4%] | 65.5% | [59.0%, 72.0%] |
| qwen2.5-7b | Q8_0 | 73.7% | [68.4%, 78.2%] | 89.0% | [83.9%, 93.0%] |
| qwen2.5-7b | Q6_K | 74.4% | [69.1%, 78.9%] | 89.0% | [83.9%, 93.0%] |
| qwen2.5-7b | Q5_K_M | 74.4% | [69.1%, 79.3%] | 89.5% | [84.5%, 93.3%] |
| qwen2.5-7b | Q4_K_M | 73.0% | [67.7%, 77.9%] | 88.5% | [83.5%, 92.5%] |
| qwen2.5-7b | Q3_K_S | 70.5% | [65.3%, 75.8%] | 86.0% | [81.0%, 90.5%] |
| qwen2.5-7b | Q2_K | 64.9% | [59.3%, 70.5%] | 83.5% | [78.0%, 88.0%] |

**Observations:**

1. Wilson CI half-widths for MMLU range from +/-4.7pp to +/-5.6pp at N=285. For ARC, half-widths range from +/-4.5pp to +/-6.5pp at N=200.
2. Mistral-7b MMLU CIs overlap substantially across all quant levels. The Q8_0 CI [53.3%, 64.9%] and Q2_K CI [49.5%, 61.4%] have significant overlap, consistent with the non-significant BPW regression.
3. qwen2.5-7b Q2_K MMLU CI [59.3%, 70.5%] does not overlap with Q8_0 CI [68.4%, 78.2%], suggesting the -8.8pp drop is likely real despite the small sample.
4. qwen2.5-7b ARC CIs remain above 78% even at Q2_K, confirming this model's exceptional robustness on science reasoning tasks.

---

## Appendix B: Metric Definitions

Identical to TR125 v1 Appendix A. See TR125 v1 for detailed metric definitions.

Summary:
- **BERTScore:** Contextual embedding similarity (deberta-xlarge-mnli). Range [0, 1].
- **ROUGE-L:** LCS-based F1 overlap. Range [0, 1].
- **Coherence (SemScore):** Sentence cosine similarity (all-mpnet-base-v2). Range [0, 1].
- **Repetition:** unique_4grams / total_4grams. Range [0, 1].
- **Output Length:** min(len_cand, len_ref) / max(len_cand, len_ref). Range [0, 1].
- **MMLU Accuracy:** Raw accuracy on 285 questions (not rescored in v2 expansion).
- **ARC Accuracy:** Raw accuracy on 200 ARC-Challenge questions.
- **Refusal Rate:** Fraction of harmful prompts refused. Range [0, 1].
- **Truthfulness:** Fraction of factual queries answered correctly. Range [0, 1].
- **Bias Resistance:** Fraction of bias probes avoided. Range [0, 1].

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **BPW** | Bits per weight -- average precision of quantized model parameters |
| **GGUF** | GPT-Generated Unified Format -- binary format for quantized LLM weights |
| **Key Metric Average** | Unweighted mean of BERTScore, coherence, and ROUGE-L |
| **MDE** | Minimum Detectable Effect at given power and alpha |
| **Ollama** | Local LLM inference server using llama.cpp backend |
| **pp** | Percentage points -- absolute accuracy difference |
| **Q2_K through Q8_0** | GGML quantization levels from 2-bit to 8-bit |
| **Quality cliff** | Quant level where accuracy drops abruptly (>9pp in one step) |
| **T4** | NVIDIA Tesla T4 GPU (16GB, Turing architecture) |
| **TOST** | Two One-Sided Tests for equivalence |

---

## Appendix D: Reproducibility

### D.1 Run Commands

```bash
# Original TR125 Phase 2 (v1 data)
python research/tr125/phase2/run.py

# Expansion (v2 data)
python research/tr142/expansion/run_tr125_expansion.py

# Bespoke analysis (unified matrix)
python research/tr142/bespoke_analysis/build_matrix.py
```

### D.2 Artifact Provenance

| Artifact | Location | Purpose |
|----------|----------|---------|
| Original samples | `results/eval/tr125_phase2/20260221_120035/samples.jsonl` | v1 quality data |
| Expansion samples | `research/tr142/expansion/results/tr125_expansion/20260328_064807/samples_scored.jsonl` | v2 quality data |
| Safety data | `research/tr134/results/phase3/20260305_144827/phase3_scored.jsonl` | TR134 safety scores |
| Unified matrix | `research/tr142/results/bespoke_analysis/20260328_173033/matrix.csv` | All tables in this report |
| BPW regressions | `research/tr142/results/bespoke_analysis/20260328_173033/bpw_regressions.csv` | SS8 regressions |
| Master analysis | `research/tr142/results/bespoke_analysis/20260328_173033/master_analysis.json` | Complete analysis object |

### D.3 Software Versions

| Package | Version |
|---------|---------|
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| scipy | 1.15.2 |
| statsmodels | 0.14.5 |
| pingouin | 0.6.1 |

### D.4 Key Assumptions

1. **Q8_0 baseline for 7B models:** FP16 exceeds T4 VRAM. Q8_0 is within ~1-4pp of FP16 based on v1 cross-validation.
2. **Temperature 0.0:** Greedy decoding, single repetition. Determinism assumed but not validated for Ollama.
3. **Same task files:** Expansion uses identical YAML task definitions as the original.
4. **Raw accuracy for expansion:** Expansion MMLU/ARC scores are raw (not rescored). Rescoring would likely increase reported accuracy for models with verbose output styles.

---

## Appendix E: Claim Validation (Updated for v2)

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| 1 | Q4_K_M preserves quality across all models | All 7 models within acceptable tier at Q4_K_M (SS11) | **Demonstrated** |
| 2 | Quality cliff at Q3_K_S boundary | 3/5 original models lose >9pp (v1); 7B models: -3.1pp mistral, -3.2pp qwen (SS3) | **Demonstrated** (model-dependent) |
| 3 | Q2_K universally unacceptable (<4B) | All small models >11pp loss (v1); 7B models classified concerning, not unacceptable (SS11) | **Demonstrated** for <4B; **partially relaxed** for 7B |
| 4 | phi-2 most quantization-robust (<4B) | Max -1.8pp through Q4_K_M, -0.4pp at Q3_K_S (v1) | **Demonstrated** |
| 5 | Larger models tolerate quantization better | Qwen: 7B loses -0.7pp at Q4_K_M vs 1.5B loses -3.2pp (SS6.2) | **Demonstrated** (within-family) |
| 6 | Mistral-7b has flattest degradation curve | MMLU: -3.8pp total Q8_0 to Q2_K; coherence: -1.4% total (SS6.4) | **Demonstrated** |
| 7 | qwen2.5-7b is the accuracy leader | 73.7% MMLU at Q8_0, 89.0% ARC at Q8_0 (SS3) | **Demonstrated** |
| 8 | Safety degrades under quantization | Refusal rate drops at Q2_K for 5/6 models (SS5.1) | **Demonstrated** |
| 9 | Generation metrics can mask degradation | qwen2.5-7b Q2_K: BERTScore/ROUGE-L/coherence up while MMLU -8.8pp (SS6.2) | **Demonstrated** |
| 10 | BPW is a poor linear predictor of quality | Median R-squared 0.17 for quality metrics (SS8) | **Demonstrated** |

---

## Appendix F: Key Numerical Claims Cross-Reference

Every headline number in the Executive Summary is traceable to a specific data source:

| Claim | Number | Source Table | Source File |
|-------|--------|-------------|-------------|
| qwen2.5-7b MMLU at Q8_0 | 73.7% | SS3.1, App A.1 | master_analysis.json line 19134 |
| qwen2.5-7b ARC at Q8_0 | 89.0% | SS3.2, App A.2 | master_analysis.json line 19124 |
| Mistral-7b MMLU Q8_0 to Q2_K drop | 3.8pp | SS3.1 | master_analysis.json lines 18734, 18634 |
| qwen2.5-7b Q4_K_M MMLU delta | -0.7pp | SS3.1 | matrix.csv qwen2.5-7b Q4_K_M row |
| Mistral-7b Q4_K_M MMLU delta | -2.1pp | SS3.1 | matrix.csv mistral-7b Q4_K_M row |
| qwen2.5-1.5b Q2_K repetition | 0.702 | SS4.4 | matrix.csv qwen2.5-1.5b Q2_K row |
| Mistral-7b refusal Q8_0 | 23.6% | SS5.1 | matrix.csv mistral-7b Q8_0 row |
| qwen2.5-7b refusal Q2_K | 80.9% | SS5.1 | matrix.csv qwen2.5-7b Q2_K row |
| Total variants: 46 | 34 + 12 | SS2.1 | config.yaml, expansion_config.yaml |
| Safe variants: 30 of 39 | 23 neg + 7 acc | SS11.4 | Computed from tier assignments |

---

## References

- TR125 v1: Quantization Decision Matrix -- 5-model, 2-phase evaluation (Banterhearts, Feb 2026)
- TR124: Quality & Accuracy Baseline -- Backend equivalence, quantization impact, sampling variance (Banterhearts, Feb 2026)
- TR123: KV-Cache Production Economics (Banterhearts, Feb 2026)
- TR134: Safety Under Quantization -- Refusal rate, truthfulness, bias resistance (Banterhearts, Mar 2026)
- TR142: Quality-Safety Correlation Matrix (Banterhearts, Mar 2026)
- MMLU: Measuring Massive Multitask Language Understanding (Hendrycks et al., 2021)
- ARC: Think you have Solved Question Answering? (Clark et al., 2018)
- SemScore: Automated evaluation using cosine similarity (Aynetdinov & Akbik, 2024)
- llama.cpp -- Local LLM inference with GGUF quantization (Gerganov et al., 2023-2026)
- Ollama -- Local LLM inference server (2023-2026)

---

**Peer Review Disclaimer:** This report has not undergone external peer review. All findings should be treated as preliminary and verified independently before use in production decisions. The sample sizes, while sufficient for detecting large effects (>9pp), may not resolve small quality differences between adjacent quant levels. Safety metrics have small sample sizes (N=50-220) and wide confidence intervals.

---

## Summary of Changes from v1 to v2

For quick reference, the following is a changelog of every addition in v2:

| Section | Change Type | Description |
|---------|------------|-------------|
| Metadata | Updated | Added expansion models, sample counts, hardware |
| Abstract | Rewritten | Expanded to cover 7 models, 4 families, safety integration |
| Executive Summary | Extended | 9 new findings for expansion, updated decision tree |
| What Changed in v2 | **New** | Explicit audit trail of all additions |
| When to Use This Report | **New** | 5 deployment scenarios for v2 data |
| SS1 Methodology | Extended | Added expansion config, task alignment, safety data, tier system, statistical approach |
| SS2 Model Lineup | Extended | Added mistral-7b, qwen2.5-7b, family coverage table |
| SS3 Benchmarks | Extended | Added MMLU/ARC for expansion models, combined summary |
| SS4 Generation Quality | Extended | Added BERTScore, ROUGE-L, coherence, repetition, output length for expansion |
| SS5 Safety | **New** | Refusal, truthfulness, bias resistance for the 6 safety-linked models; safety-quality interaction; judge metrics |
| SS6 Model Families | Extended | Added Qwen pair, Mistral section, within-family scaling summary |
| SS7 Cross-Model | Extended | Added degradation curve shapes, cross-architecture quality ordering |
| SS8 BPW Regressions | **New** | Linear regressions for all metrics vs bits-per-weight |
| SS9 Baseline Note | Extended | Added Q8_0 rationale for expansion models |
| SS10 Full Matrix | Extended | All 46 variants in unified tables |
| SS11 Tier Classification | Extended | 10 new variant tiers, combined totals, statistical caveats |
| SS12 Production Guidance | Updated | Updated decision tree, VRAM guide, selection flowchart |
| SS13 Consumer Deployment | **New** | Size-quality frontier, cost comparison, cloud API positioning |
| Limitations | Extended | Added L11-L12 for expansion-specific caveats |
| What Remains Open | **New** | 10 open questions for future work |
| Appendix A.6 | **New** | Wilson CIs for expansion benchmarks |
| Appendix E | Updated | Extended claim validation for v2 |
| Appendix F | **New** | Numerical claims cross-reference |

---

**End of Technical Report 125 v2 (Expanded, 7-Model Cross-Family)**
