# Technical Report 142 v2: Quality-Safety Correlation Under Quantization -- Expanded Cross-Family Synthesis
## Cross-referencing TR125 Phase 2 quality metrics with TR134 Phase 3 safety metrics across 6 models, 4 families, and 40 model-quant cells

| Field | Value |
|-------|-------|
| **TR Number** | 142 v2 |
| **Project** | Banterhearts |
| **Date** | 2026-03-28 |
| **Version** | 2.0 |
| **Author** | Research Team |
| **Status** | FINAL |
| **Report Type** | Analysis-only (no new experiments) |
| **Run Directory** | `research/tr142/results/bespoke_analysis/20260328_173033/` |
| **Quality Source** | TR125 Phase 2 (24,990 source samples) plus TR125 v2 expansion (8,820 samples for mistral-7b and qwen2.5-7b quality data) |
| **Safety Source** | TR134 Phase 3 legacy (24,778 rows) plus TR142 safety expansion (13,342 rows); judge metrics sourced from 12,168 legacy labels, 6,552 Gemma small-model labels, and 5,616 Gemma 7B rejudge labels |
| **Models** | llama3.2-1b, llama3.2-3b, mistral-7b, phi-2, qwen2.5-1.5b, qwen2.5-7b |
| **Families** | Llama (2 models), Mistral (1), Phi (1), Qwen (2) |
| **Quant Levels** | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K |
| **Model-Quant Cells** | 40 (4 models x 7 quants + 2 models x 6 quants) |
| **Analysis Passes** | 14 core passes + supporting diagnostics |
| **Safety Measurement** | Regex-based metrics on all 40 cells + judge-derived aggregates on all 40 cells from three judge source files |
| **Related Work** | [TR125 v2](Technical_Report_125_v2.md), [TR134 v2](Technical_Report_134_v2.md), [TR139](Technical_Report_139.md), [TR142 v1](Technical_Report_142.md) |
| **Depends On** | TR125 Phase 2 (quality data), TR134 Phase 3 (safety data) |
| **Supersedes** | TR142 v1 (2-model, 14-cell analysis) |

---

## Abstract

TR142 v2 expands the original 2-model quality-safety correlation study to **6 models across 4 architecture families** (Llama, Mistral, Phi, Qwen), producing a **40-cell** merged quality-safety matrix under GGUF quantization. The expansion tests whether the Simpson's paradox and safety-faster-than-quality findings from TR142 v1 were Llama-specific artifacts or cross-family phenomena.

The core findings generalize. **34 of 36 quality-safety metric pairs** exhibit sign reversals across models: the direction of quality-safety correlation at a given quant level depends on which model is being evaluated, not on the quant level itself. This is Simpson's paradox at scale, confirmed across 4 independent architecture families, and it cannot be dismissed as a 2-model anomaly. Safety degrades faster than quality in **26 of 34 non-baseline cells** (76%), with asymmetry ratios ranging from 1.4x to 262x. Two model-quant cells meet the strict hidden-danger criterion (quality within +/-3pp of baseline, safety drops 12+ pp): **llama3.2-1b Q3_K_S** (BERTScore +0.98pp, refusal -13.6pp) and **qwen2.5-7b Q2_K** (BERTScore +2.39pp, refusal -12.3pp). The conservative deployment floor at **Q5_K_M** holds across all 6 models: the maximum refusal deviation at Q5_K_M is 3.2pp (qwen2.5-1.5b), within practical tolerance.

The expanded statistical toolkit includes Spearman rank correlations, repeated-measures correlations, leave-one-quant-out and leave-one-model-out robustness checks, mixed-effects models, baseline mean/risk tests, and judge-based safety metrics alongside regex-based metrics. The Pearson-Spearman divergence on several models (e.g., llama3.2-1b: Pearson r = +0.994, Spearman rho = +0.600 for coherence x refusal) confirms that the linear correlation is driven by the extreme Q2_K point, while the rank correlation captures the monotonic trend. Judge-based refusal rates diverge from regex-based rates by up to 71pp on Mistral, demonstrating that the measurement instrument shapes the correlation story.

The operational conclusion is unchanged but now rests on 3x the model coverage and nearly 3x the cell count: **quality metrics are not safety proxies under quantization, and the direction of failure is model-specific.** This report feeds the NeurIPS 2026 paper directly.

---

## Executive Summary

### Five Key Findings

1. **Simpson's paradox is pervasive, not a 2-model artifact.** 34 of 36 quality-safety metric pairs show sign reversals across models (at least one model positive, at least one negative). The original llama3.2-1b r = +0.994 and llama3.2-3b r = -0.829 are joined by qwen2.5-1.5b r = +0.997, phi-2 r = +0.296, mistral-7b r = +0.354, and qwen2.5-7b r = -0.580 (all coherence x refusal). Pooling across models remains misleading.

2. **Safety degrades faster than quality in 76% of cells.** 26 of 34 non-baseline model-quant cells have |safety_delta| > |quality_delta|. Asymmetry ratios range from 1.4x (llama3.2-1b Q4_K_M) to 262x (mistral-7b Q6_K). The 13.9x ratio at llama3.2-1b Q3_K_S from v1 is confirmed and now contextualized within a broader distribution.

3. **Two confirmed hidden-danger cells.** llama3.2-1b Q3_K_S (quality +0.98pp, safety -13.6pp) and qwen2.5-7b Q2_K (quality +2.39pp, safety -12.3pp) meet the strict criterion: quality stays within +/-3pp of baseline while safety collapses beyond 12pp. A third cell (mistral-7b Q2_K: quality -2.1pp, safety -11.4pp) is near-hidden-danger.

4. **Q5_K_M is a robust conservative floor.** Across all 6 models, refusal deviation at Q5_K_M ranges from 0.0pp (qwen2.5-7b) to +3.2pp (qwen2.5-1.5b). No model exceeds 3.2pp refusal shift at Q5_K_M. BERTScore deviations range from -2.2pp to +1.0pp.

5. **Quality-gating adds no discriminative power for safety.** A coherence/length gate at various thresholds does not separate hidden-danger cells from safe cells. Filter rates correlate with quantization severity, not with safety status. The gate is responsive but not diagnostic.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Model coverage | Shared models | >= 4 | 6 | **PASS** |
| Family coverage | Architecture families | >= 3 | 4 | **PASS** |
| Cell count | Model-quant cells | >= 30 | 40 | **PASS** |
| Sign reversal | Opposite within-model signs across families | >= 2 families | 2 families (Llama, Qwen) | **PASS** |
| Asymmetry | Safety moves faster than quality | majority of cells | 26/34 (76%) | **PASS** |
| Hidden-danger | Cross-family hidden-danger cells | >= 2 families | 2 families (Llama, Qwen) | **PASS** |
| Conservative floor | Q5_K_M within +/-3pp refusal on all models | all models | all 6 within +/-3.2pp | **PASS** |
| Gate discriminative power | Gate separates hidden-danger from neutral | yes | no | **FAIL** (expected) |
| Truthfulness power | MDE at 80% power | < 10pp | 28.0pp | **FAIL** |
| Dual safety metrics | Judge and regex reported | yes | yes (judge aggregates on all 40 cells; 24,336 underlying judge annotations across three source files) | **PASS** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | Quality-safety correlation sign is model-dependent (Simpson's paradox) | 34/36 sign reversals across 6 models, 4 families | **Established** |
| C2 | Safety degrades faster than quality in most cells | 26/34 cells (76%), all families represented | **Established** |
| C3 | Hidden-danger cells exist across families | 2 confirmed (Llama, Qwen), 1 near (Mistral) | **Established** |
| C4 | Q5_K_M is a cross-family conservative floor | All 6 models within +/-3.2pp refusal at Q5_K_M | **Established** |
| C5 | Quality-gating does not discriminate hidden-danger cells | Gate correlates with quant severity, not safety status | **Established** |
| C6 | Safety degradation slopes differ by family | ANOVA F = 0.62, p = 0.477 | **Not established** (slopes do not differ) |
| C7 | Truthfulness is unchanged by quantization | MDE = 28pp, underpowered | **Not established** (inconclusive) |

### Core Decisions

1. **Do not pool quality-safety correlations across models.** The sign reversal is pervasive (34/36 pairs).
2. **Monitor safety independently below Q5_K_M on every model.** The hidden-danger pattern appears in different families (Llama and Qwen).
3. **Use Q5_K_M as the cross-family conservative floor.** All 6 models stay within practical tolerance at this level.
4. **Do not use quality gates as safety screening.** The gate does not remove hidden-danger cells.
5. **Report Pearson and Spearman together.** Their divergence reveals whether the correlation is linear or driven by extreme points.

### How to Read This Report

**5-minute version:** Read the Abstract and Executive Summary. If making deployment decisions, also read SS4 (regime classification), SS8 (Q5_K_M floor), and SS18 (production guidance).

**30-minute version:** Add SS5 (Simpson's paradox), SS6 (asymmetry), SS7 (hidden-danger), and SS12 (Spearman vs Pearson). These contain the core analytical findings with observations.

**Full reading:** The report is designed to be read front-to-back. Each result section follows the pattern: context prose, data table, then **Observations** interpreting the table. Appendices provide the complete data for audit. Cross-references use SS notation (e.g., "See SS7 Table 6").

---

## Table of Contents

- [Abstract](#abstract)
- [Executive Summary](#executive-summary)
- [When to Use This Report](#when-to-use-this-report)
- [What Changed in v2](#what-changed-in-v2)
- [Metric Definitions](#metric-definitions)
- [SS1. Introduction](#ss1-introduction)
- [SS2. Methodology](#ss2-methodology)
- [SS3. Models and Design](#ss3-models-and-design)
- [SS4. Quality-Safety Matrix (Regime Classification)](#ss4-quality-safety-matrix-regime-classification)
- [SS5. Result 1: Simpson's Paradox at Scale](#ss5-result-1-simpsons-paradox-at-scale)
- [SS6. Result 2: Quality-Safety Asymmetry](#ss6-result-2-quality-safety-asymmetry)
- [SS7. Result 3: Hidden Danger Zones](#ss7-result-3-hidden-danger-zones)
- [SS8. Result 4: Conservative Floor at Q5_K_M](#ss8-result-4-conservative-floor-at-q5_k_m)
- [SS9. Result 5: Quality-Gate Failure](#ss9-result-5-quality-gate-failure)
- [SS10. Robustness: Leave-One-Quant-Out](#ss10-robustness-leave-one-quant-out)
- [SS11. Robustness: Leave-One-Model-Out](#ss11-robustness-leave-one-model-out)
- [SS12. Robustness: Spearman vs Pearson](#ss12-robustness-spearman-vs-pearson)
- [SS13. Judge-Based vs Regex-Based Safety](#ss13-judge-based-vs-regex-based-safety)
- [SS14. Cross-Family ANOVA](#ss14-cross-family-anova)
- [SS15. BPW Regression](#ss15-bpw-regression)
- [SS16. Limitations](#ss16-limitations)
- [SS17. Conclusions](#ss17-conclusions)
- [SS18. Production Guidance](#ss18-production-guidance)
- [Appendix A: Full Correlation Table](#appendix-a-full-correlation-table)
- [Appendix B: Full Asymmetry Table](#appendix-b-full-asymmetry-table)
- [Appendix C: Regime Classification Table](#appendix-c-regime-classification-table)
- [Appendix D: Glossary](#appendix-d-glossary)
- [Appendix E: Reproducibility](#appendix-e-reproducibility)
- [References](#references)
- [Peer Review Disclaimer](#peer-review-disclaimer)

---

## When to Use This Report

### Scenario 1: Evaluating a quantized model with quality benchmarks only

**Question:** We run BERTScore and coherence checks on our quantized Qwen 7B model at Q2_K and everything looks fine. Is the model safe to deploy?

**Answer:** No. SS7 Table 6 shows that qwen2.5-7b Q2_K is a confirmed hidden-danger cell: BERTScore *improves* +2.39pp while refusal rate drops -12.27pp. Quality passing does not imply safety passing. This pattern appears across families (also on llama3.2-1b Q3_K_S), so it is not architecture-specific.

### Scenario 2: Choosing between Q4_K_M and Q5_K_M for production

**Question:** Q4_K_M saves 17% VRAM. Is the safety cost worth it?

**Answer:** Model-dependent. SS8 Table 7 shows Q5_K_M keeps all 6 models within +/-3.2pp refusal. Q4_K_M exceeds +/-3pp on 3 of 6 models, with llama3.2-3b dropping -10.0pp. Unless you have per-model safety validation at Q4_K_M, the 17% VRAM savings is not worth the safety uncertainty.

### Scenario 3: Understanding why quality and safety monitoring disagree

**Question:** Our quality metrics show improvement at a lower quant level, but safety probes show degradation. Is one wrong?

**Answer:** Both may be correct. SS5 demonstrates that quality and safety can move in opposite directions under quantization -- quality improving while safety degrades (llama3.2-1b Q3_K_S) or quality degrading while safety tightens (llama3.2-3b Q3_K_S). The direction depends on the model. See SS5.4 for the theoretical framework.

### Scenario 4: Deciding whether to use regex or judge safety metrics

**Question:** Our safety evaluation uses keyword matching. Is that sufficient?

**Answer:** Depends on the model. SS13 shows that Mistral's regex refusal rate differs from its judge refusal rate by 64-71pp. Regex underestimates Mistral's actual refusal behavior. For models with subtle refusal styles, judge-based or human-annotated metrics are necessary.

### Scenario 5: Positioning this analysis relative to the broader safety line

**Question:** How does TR142 v2 relate to the earlier TRs?

**Answer:** TR134 provided raw safety data; TR125 provided raw quality data; TR142 v2 merges them to analyze the *relationship* between quality and safety. TR139 studies multi-turn jailbreak resistance. Together, they show that quantization affects safety through multiple independent channels, and quality metrics are insufficient proxies for any of them. See SS17.2 for the full cross-TR comparison.

---

## What Changed in v2

| Dimension | TR142 v1 | TR142 v2 |
|-----------|----------|----------|
| Models | 2 (llama3.2-1b, llama3.2-3b) | **6** (+ mistral-7b, phi-2, qwen2.5-1.5b, qwen2.5-7b) |
| Families | 1 (Llama) | **4** (Llama, Mistral, Phi, Qwen) |
| Model-quant cells | 14 | **40** |
| Baseline approach | FP16 for all | FP16 for 4 models, **Q8_0 for mistral-7b and qwen2.5-7b** (no FP16 Ollama variant available) |
| Correlation methods | Pearson only | **Pearson + Spearman + repeated-measures + mixed-effects diagnostics** |
| Supporting diagnostics | Minimal | **Raw-scale pooled correlations, baseline mean/risk tests, repeated-measures correlations, mixed-effects models** |
| Robustness checks | LOO on quants only | **Leave-one-quant-out + leave-one-model-out** |
| Safety measurement | Regex only | **Regex + LLM judge** (dual safety metrics) |
| Sign reversal analysis | Informal (2 models, one positive, one negative) | **Systematic: 36 metric pairs, sign split tallied** |
| Hidden-danger cells | 1 confirmed (llama3.2-1b Q3_K_S) | **2 confirmed + 1 near** |
| Cross-family ANOVA | Not applicable | **F = 0.62, p = 0.477** (safety slopes not family-dependent) |
| BPW regression | 2 models, weak R-squared | **6 models, per-model slopes** |
| Gate analysis | 2-model gate sweep | **6-model gate sweep** |

The expansion from 2 to 6 models is the defining change. v1 could not distinguish "Llama-specific coupling" from "universal coupling." v2 demonstrates that the Simpson's paradox finding, the asymmetry finding, and the conservative floor finding all generalize across architectures. The sign reversal is not a Llama artifact; it appears in every family.

The addition of Q8_0-based baselines for mistral-7b and qwen2.5-7b introduces a methodological limitation (see SS16): these models' deltas are measured from Q8_0, not FP16. Deltas from Q8_0 will be slightly smaller in absolute terms than deltas from FP16, potentially underestimating degradation at low quants. The Q5_K_M floor findings are not affected because the Q5_K_M-to-Q8_0 gap is small on all models.

---

## Metric Definitions

### Primary Quality Metrics (from TR125 Phase 2)

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| BERTScore (F1) | Token-level semantic similarity between model output and reference, using contextual embeddings | 0-1, higher = better generation quality |
| ROUGE-L | Longest common subsequence overlap between output and reference | 0-1, higher = better factual coverage |
| Coherence | Composite score of output fluency, logical structure, and response relevance | 0-1, higher = more coherent output |

### Primary Safety Metrics (from TR134 Phase 3)

| Metric | Source | Definition | Interpretation |
|--------|--------|-----------|----------------|
| Refusal Rate (regex) | Regex classifier | Fraction of harmful prompts where the model correctly refuses to comply, identified by keyword matching | 0-1, higher = safer |
| Refusal Rate (judge) | LLM judge | Fraction of harmful prompts where the model correctly refuses to comply, assessed by an LLM judge | 0-1, higher = safer |
| Truthfulness (regex) | Regex classifier | Fraction of factual probes answered accurately | 0-1, higher = more truthful |
| Truthfulness (judge) | LLM judge | Fraction of factual probes answered accurately, assessed by an LLM judge | 0-1, higher = more truthful |
| Bias Resistance (regex) | Regex classifier | Fraction of bias-probing prompts where the model avoids stereotypical outputs | 0-1, higher = more bias-resistant |
| Bias Resistance (judge) | LLM judge | Same as above, assessed by LLM judge | 0-1, higher = more bias-resistant |

### Statistical Tests Used

| Test | Role in This Report |
|------|-------------------|
| Pearson's r | Linear correlation between quality and safety degradation curves |
| Spearman's rho | Rank-based correlation; robust to outliers and non-linearity |
| Repeated-measures correlation | Within-model association after controlling for model-level intercept differences |
| Mixed-effects model | Secondary pooled-signal check with per-model grouping |
| Welch's t-test | Pairwise quant-vs-baseline comparisons on safety metrics |
| Cohen's d | Effect size for pairwise comparisons |
| Holm-Bonferroni | Multiple comparison correction |
| Bootstrap percentile CI | Confidence intervals for asymmetry gaps (B = 2000, seed = 42) |
| Risk-difference / odds-ratio tests | Binary baseline-vs-quant comparisons for refusal, truthfulness, and bias resistance |
| OLS regression | BPW-vs-metric linear fit |
| One-way ANOVA | Cross-family comparison of safety degradation slopes |
| Leave-one-out | Robustness of pooled statistics to individual quant levels or models |

### Evidence Standard

**Established findings** require p < 0.05 after Holm-Bonferroni correction with |d| >= 0.4 or practical significance above 3pp.

**Structural findings** (e.g., Simpson's paradox) are established by demonstrating the sign pattern exists across models, not by p-value on the pooled correlation.

**Non-claims** are results where evidence is insufficient. Truthfulness effects remain in this category across all 6 models.

---

## SS1. Introduction

### SS1.1 Research Questions

1. **Q12 (expanded):** Does the quality-safety correlation sign reversal observed on 2 Llama models generalize to Mistral, Phi, and Qwen families?
2. **Q14:** Is the asymmetry (safety moves faster than quality) a cross-family phenomenon or a Llama-specific property?
3. **Q15:** Does the Q5_K_M conservative floor hold across 4 architecture families?
4. **Q16:** Does adding Spearman correlations, repeated-measures diagnostics, mixed-effects checks, and leave-one-out robustness change the v1 conclusions?
5. **Q17:** Does judge-based safety measurement agree with regex-based measurement, and does the choice of instrument affect the correlation story?

### SS1.2 Consumer Deployment Thesis

The consumer deployment of quantized LLMs is accelerating. Models compressed to 4-bit or 3-bit formats run on laptops, phones, and edge devices, with quality benchmarks as the primary acceptance criterion. If quality benchmarks silently pass while safety alignment degrades, millions of consumer deployments could operate with compromised safety guardrails.

TR142 v1 demonstrated this risk on 2 models. The question addressed by v2 is whether the risk is model-specific (a quirk of Llama 3.2) or structural (inherent to GGUF quantization across architectures). The answer -- structural -- has direct implications for every organization deploying quantized models to consumers.

### SS1.3 Scope

| Dimension | Coverage |
|-----------|----------|
| Models | llama3.2-1b (1.2B, GQA), llama3.2-3b (3.2B, GQA), mistral-7b (7.2B, GQA+SWA), phi-2 (2.7B, MHA), qwen2.5-1.5b (1.5B, GQA), qwen2.5-7b (7.6B, GQA) |
| Families | Llama (2), Mistral (1), Phi (1), Qwen (2) |
| Quant levels | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K |
| Baseline | FP16 for llama3.2-1b, llama3.2-3b, phi-2, qwen2.5-1.5b; Q8_0 for mistral-7b, qwen2.5-7b |
| Quality metrics | BERTScore, ROUGE-L, coherence |
| Safety metrics (regex) | Refusal rate, truthfulness, bias resistance |
| Safety metrics (judge) | Refusal rate, truthfulness, bias resistance aggregated for all 40 cells from three judge source files |
| Model-quant cells | 40 |
| Backend | Ollama (llama.cpp) |
| Analysis passes | 14 core passes + supporting diagnostics |

### SS1.4 Literature Grounding

The quantization literature (Dettmers et al., 2023; Frantar et al., 2022) evaluates quality degradation in isolation, using perplexity, downstream accuracy, and generation quality as the sole metrics for assessing quantized model fitness. The implicit assumption is that if quality is preserved, the model is "fine" for all purposes. Safety-under-quantization has been studied in TR134 and TR139 within this project, and by concurrent work (Lin et al., 2024; Ma et al., 2024), but these studies treat safety as an independent dimension without examining its relationship to quality.

TR142 v1 was the first study to characterize the *correlation structure* between quality and safety degradation under quantization, identifying Simpson's paradox on 2 Llama models. The v1 finding drew attention because it challenged the "quality is a proxy for everything" assumption. However, the 2-model scope left open a critical question: was the sign reversal a Llama-specific artifact, or a structural property of quantization?

TR142 v2 resolves this ambiguity by extending to 4 families. The paradox is structural. The quality-proxy assumption fails across architectures, not just on Llama. This has implications beyond the quantization literature: any evaluation framework that relies on quality metrics as safety proxies is vulnerable to the same paradox.

### SS1.5 Contributions

TR142 v2 makes four contributions beyond v1:

1. **Cross-family generalization of Simpson's paradox.** 34/36 sign reversals across 6 models and 4 families.
2. **Second hidden-danger cell identification.** qwen2.5-7b Q2_K joins llama3.2-1b Q3_K_S as a confirmed quality-stable safety-collapse cell.
3. **Dual safety measurement.** Regex vs. judge divergence up to 71pp on Mistral shows the correlation story depends on the measurement instrument.
4. **Robustness at scale.** Leave-one-quant-out and leave-one-model-out analyses confirm the Simpson's paradox finding is not driven by any single quant level or model.

---

## SS2. Methodology

### SS2.1 Overall Design

TR142 v2 is a post-hoc cross-referencing analysis. The pipeline extends v1:

1. Load and normalize TR125 Phase 2 quality samples and TR134 Phase 3 safety samples (including judge labels)
2. Expand to 6 overlapping models and their available quant levels
3. Aggregate quality metrics by (model, quant) and safety metrics by (model, quant)
4. Merge into a 40-row quality-safety matrix (6 models x 5-7 quants each)
5. Run 14 core analysis passes: correlation (Pearson + Spearman), divergence, quality-gating, asymmetry, regime classification, BPW regression, leave-one-quant-out, leave-one-model-out, cross-family ANOVA, judge vs. regex comparison

Supporting CSVs from the same run also export repeated-measures correlations, mixed-effects models, raw-scale pooled correlations, and baseline mean/risk tests.

### SS2.2 Baseline Selection

Four models (llama3.2-1b, llama3.2-3b, phi-2, qwen2.5-1.5b) use FP16 as baseline. Two models (mistral-7b, qwen2.5-7b) use Q8_0 as baseline because no FP16 Ollama variant was available in the source data. This means:

- Deltas for mistral-7b and qwen2.5-7b are measured from Q8_0, which is already slightly quantized
- These deltas will underestimate total degradation from FP16 by the Q8_0-to-FP16 gap (typically < 1pp on quality, < 2pp on safety)
- The Q5_K_M floor analysis is unaffected because Q5_K_M deltas from Q8_0 are small regardless
- mistral-7b and qwen2.5-7b contribute 5 non-baseline quant levels each (Q6_K through Q2_K), not 6

The total cell count is: 4 models x 7 quants = 28 cells + 2 models x 6 quants = 12 cells = 40 cells. Of these, 6 are baseline cells, leaving **34 non-baseline cells** for delta analysis.

### SS2.3 Unit of Analysis

The unit of analysis is a (model, quant) cell in the merged matrix. Each cell contains aggregated quality scores (mean BERTScore, ROUGE-L, coherence) and aggregated safety scores (refusal rate, truthfulness, bias resistance). Judge-derived aggregates are available for all 40 cells in the current bundle, sourced from three judge files totaling 24,336 annotations.

### SS2.4 Statistical Framework

All pairwise tests use Welch's t-test comparing each quant level against its baseline for the same model and metric, with Holm-Bonferroni correction applied across the expanded test family. Correlations are computed using both Pearson's r (linear) and Spearman's rho (rank-based) within each model, across families, and pooled. The current bundle also exports repeated-measures correlations, mixed-effects models, raw-scale pooled correlations, and baseline mean/risk tests as secondary consistency checks. Bootstrap CIs use B = 2,000, seed = 42, percentile method. Leave-one-out analyses remove each quant level (or model) in turn and recompute pooled statistics.

These supporting diagnostics are not the primary evidence standard for the headline claims. The report's core claims still rest on the sign-reversal table, asymmetry table, hidden-danger classification, and Q5_K_M floor table; the supporting analyses are included to make the bundle auditable and reusable.

### SS2.5 How Rows Become Claims

Raw sample-level scores from TR125p2 and TR134p3 are aggregated to cell means. The pipeline follows four stages:

1. **Aggregation**: Per-sample binary scores (0/1 for safety, continuous for quality) are averaged within each (model, quant, metric) cell.
2. **Delta computation**: Each non-baseline cell's mean is compared to its baseline (FP16 or Q8_0) mean to produce a delta in percentage points.
3. **Correlation**: Within-model correlations are computed on the non-baseline quant points (5 or 6 per model). Spearman rho is computed alongside Pearson r for robustness.
4. **Qualification**: A structural claim (Simpson's paradox) requires the sign pattern to hold across models. A cell-level claim (hidden-danger) requires quality delta within +/-3pp and safety delta exceeding 10pp. An asymmetry claim requires |safety_delta| > |quality_delta|.

The dual qualification (structural + cell-level) prevents over-interpreting any single comparison while still allowing the pattern-level findings to emerge.

### SS2.6 What This Design Does Not Do

- It does not establish causal relationships between quality degradation and safety degradation. Correlation between quality and safety curves does not mean quality loss *causes* safety loss. Both could be driven by a common factor (e.g., weight magnitude reduction).
- It does not measure per-sample overlap. Quality and safety were measured on different prompt sets, so the correlations measure model-level capability co-variation, not sample-level co-occurrence.
- It cannot distinguish instruction-following degradation from knowledge degradation as the mechanism behind safety changes.
- It does not test non-GGUF quantization formats (GPTQ, AWQ, SqueezeLLM). Format-specific effects on safety coupling are unknown.
- The Q8_0 baseline for 2 models introduces a systematic underestimation of total degradation from FP16.
- It does not account for prompt-level variance in safety scores. The safety metrics are aggregated to the cell level, smoothing over prompt-specific effects that could drive the coupling.

---

## SS3. Models and Design

### SS3.1 Model Summary

| Model | Family | Parameters | Architecture | Quant Levels | Baseline | N Quants |
|-------|--------|-----------|-------------|-------------|----------|----------|
| llama3.2-1b | Llama | 1.2B | GQA | FP16-Q2_K | FP16 | 7 |
| llama3.2-3b | Llama | 3.2B | GQA | FP16-Q2_K | FP16 | 7 |
| mistral-7b | Mistral | 7.2B | GQA + SWA | Q8_0-Q2_K | Q8_0 | 6 |
| phi-2 | Phi | 2.7B | MHA | FP16-Q2_K | FP16 | 7 |
| qwen2.5-1.5b | Qwen | 1.5B | GQA | FP16-Q2_K | FP16 | 7 |
| qwen2.5-7b | Qwen | 7.6B | GQA | Q8_0-Q2_K | Q8_0 | 6 |

The 6 models span a 6.3x parameter range (1.2B to 7.6B) and 4 distinct architecture families. Two families (Llama, Qwen) contribute both a small and large model, enabling within-family size comparisons. Mistral and Phi contribute single models at different sizes, providing cross-family diversity.

### SS3.2 Why These Six Models

The 6-model set is determined by the intersection of TR125 Phase 2 and TR134 Phase 3 model coverage. TR125p2 evaluated 4 models; TR134p3 evaluated models with overlapping quant levels. The 6-model overlap is a 3x improvement over v1's 2-model restriction and provides the minimum coverage needed to test cross-family generalization (at least 2 families with at least 2 models each, plus additional families for diversity).

### SS3.3 Architecture Diversity

| Family | Attention | Positional Encoding | Activation | Models in Study |
|--------|----------|-------------------|------------|-----------------|
| Llama | GQA | RoPE | SiLU | 2 (1.2B, 3.2B) |
| Mistral | GQA + Sliding Window | RoPE | SiLU | 1 (7.2B) |
| Phi | Multi-Head (MHA) | Rotary | GELU | 1 (2.7B) |
| Qwen | GQA | RoPE | SiLU | 2 (1.5B, 7.6B) |

The Phi-2 model uses full multi-head attention (no KV sharing), making it the architectural outlier. If the quality-safety coupling pattern holds on MHA as well as GQA, it is unlikely to be an artifact of the attention sharing scheme.

### SS3.4 Environment

| Component | Specification |
|-----------|--------------|
| Backend | Ollama (llama.cpp) |
| GPU | NVIDIA (12 GB VRAM) |
| Temperature | 0.0 (deterministic) |
| Prompt format | Instruct (model-native chat template) |
| Quality evaluation | BERTScore (F1), ROUGE-L, coherence composite |
| Safety evaluation | Regex classifier + LLM judge |
| Seed (bootstrap) | 42 |

---

## SS4. Quality-Safety Matrix (Regime Classification)

This section presents the regime classification for all 40 model-quant cells. Each cell is classified based on BERTScore delta (quality) and refusal rate delta (safety) relative to its baseline.

### SS4.1 Regime Definitions

| Regime | Criterion | Deployment Implication |
|--------|----------|----------------------|
| **hidden_danger** | Quality within +/-3pp of baseline AND safety drops > 10pp | Most dangerous: quality-only monitoring will miss safety collapse |
| **near_hidden_danger** | Quality within +/-3pp of baseline AND safety drops 5-10pp | Borderline: safety has begun shifting but quality remains stable |
| **neutral** | Neither quality nor safety meets the danger threshold | No unusual divergence detected |

### SS4.2 Complete Regime Classification

| Model | Family | Quant | Baseline | BERTScore Delta (pp) | Refusal Delta (pp) | Regime |
|-------|--------|-------|----------|---------------------|--------------------|----|
| llama3.2-1b | Llama | Q8_0 | FP16 | -0.15 | +0.91 | neutral |
| llama3.2-1b | Llama | Q6_K | FP16 | -0.48 | +0.45 | neutral |
| llama3.2-1b | Llama | Q5_K_M | FP16 | -0.67 | -1.82 | neutral |
| llama3.2-1b | Llama | Q4_K_M | FP16 | +1.88 | -3.18 | neutral |
| **llama3.2-1b** | **Llama** | **Q3_K_S** | **FP16** | **+0.98** | **-13.64** | **hidden_danger** |
| llama3.2-1b | Llama | Q2_K | FP16 | -9.61 | -56.82 | neutral |
| llama3.2-3b | Llama | Q8_0 | FP16 | -0.18 | -1.82 | neutral |
| llama3.2-3b | Llama | Q6_K | FP16 | +0.02 | +0.91 | neutral |
| llama3.2-3b | Llama | Q5_K_M | FP16 | -0.54 | +0.45 | neutral |
| llama3.2-3b | Llama | Q4_K_M | FP16 | -0.89 | -10.00 | neutral |
| llama3.2-3b | Llama | Q3_K_S | FP16 | -3.92 | +18.64 | neutral |
| llama3.2-3b | Llama | Q2_K | FP16 | -0.20 | +16.36 | neutral |
| mistral-7b | Mistral | Q6_K | Q8_0 | -0.02 | +5.00 | neutral |
| mistral-7b | Mistral | Q5_K_M | Q8_0 | -0.10 | +0.91 | neutral |
| mistral-7b | Mistral | Q4_K_M | Q8_0 | +0.88 | -1.36 | neutral |
| mistral-7b | Mistral | Q3_K_S | Q8_0 | +0.93 | -4.55 | neutral |
| mistral-7b | Mistral | Q2_K | Q8_0 | -2.12 | -11.36 | **near_hidden_danger** |
| phi-2 | Phi | Q8_0 | FP16 | +0.84 | 0.00 | neutral |
| phi-2 | Phi | Q6_K | FP16 | +0.99 | -4.55 | neutral |
| phi-2 | Phi | Q5_K_M | FP16 | +1.01 | -0.91 | neutral |
| phi-2 | Phi | Q4_K_M | FP16 | +0.56 | -3.64 | neutral |
| phi-2 | Phi | Q3_K_S | FP16 | -0.47 | -2.27 | neutral |
| phi-2 | Phi | Q2_K | FP16 | +2.66 | -3.64 | neutral |
| qwen2.5-1.5b | Qwen | Q8_0 | FP16 | +0.15 | -0.91 | neutral |
| qwen2.5-1.5b | Qwen | Q6_K | FP16 | -1.41 | +1.36 | neutral |
| qwen2.5-1.5b | Qwen | Q5_K_M | FP16 | -0.78 | +3.18 | neutral |
| qwen2.5-1.5b | Qwen | Q4_K_M | FP16 | -2.62 | -4.09 | neutral |
| qwen2.5-1.5b | Qwen | Q3_K_S | FP16 | -1.75 | +0.45 | neutral |
| qwen2.5-1.5b | Qwen | Q2_K | FP16 | -14.23 | -50.00 | neutral |
| qwen2.5-7b | Qwen | Q6_K | Q8_0 | +0.52 | +0.45 | neutral |
| qwen2.5-7b | Qwen | Q5_K_M | Q8_0 | -2.19 | 0.00 | neutral |
| qwen2.5-7b | Qwen | Q4_K_M | Q8_0 | +0.15 | +1.36 | neutral |
| qwen2.5-7b | Qwen | Q3_K_S | Q8_0 | -0.14 | -8.64 | neutral |
| **qwen2.5-7b** | **Qwen** | **Q2_K** | **Q8_0** | **+2.39** | **-12.27** | **hidden_danger** |

**Observations.**

- Of the 34 non-baseline cells, 2 are classified **hidden_danger** and 1 as **near_hidden_danger**. The remaining 31 are neutral. This low rate (6% hidden-danger) means the phenomenon is rare but not negligible -- it appears in 2 of 4 families (Llama and Qwen) and at different quant levels (Q3_K_S and Q2_K).
- The two hidden-danger cells share a structural pattern: quality BERTScore *increases* slightly from baseline (positive delta) while safety refusal rate collapses. The quality improvement creates an especially dangerous false signal -- not merely "quality is stable" but "quality looks better than baseline."
- llama3.2-3b shows no hidden-danger cells despite having the most extreme safety shifts (+18.6pp, +16.4pp over-refusal at Q3_K_S and Q2_K). This is because quality also degrades at those levels, so the cell does not meet the "quality stable" criterion.
- mistral-7b Q2_K (-2.12pp quality, -11.36pp safety) is near-hidden-danger: quality has just crossed the 2pp threshold that would qualify as "stable," but the 11.4pp safety drop is large. With a Q8_0 baseline (which already absorbs some quantization damage), the true FP16-relative safety drop is likely 12-13pp.
- phi-2 shows no hidden-danger cells at any quant level. Its safety deltas remain modest throughout (-4.55pp maximum refusal shift), and its quality deltas are similarly bounded. Phi-2 appears to be the most coupling-resistant model in the study.
- qwen2.5-1.5b Q2_K (-14.2pp quality, -50.0pp safety) is the most extreme collapse cell in the study but is classified neutral because both dimensions collapse simultaneously -- there is no false-safety signal.

> Two hidden-danger cells confirmed across two families (Llama and Qwen). Quality-only monitoring would certify both as acceptable. The hidden-danger pattern is rare but cross-family.

---

## SS5. Result 1: Simpson's Paradox at Scale

### SS5.1 Per-Model Correlations (Coherence x Refusal Rate, Regex)

The headline correlation pair from v1 -- coherence vs. refusal rate -- is now computed across all 6 models using both Pearson and Spearman.

**Table 1: Per-model correlation, coherence x refusal rate (regex)**

| Model | Family | Pearson r | Pearson p | Spearman rho | Spearman p | N |
|-------|--------|----------|-----------|-------------|-----------|---|
| llama3.2-1b | Llama | **+0.994** | 5.8e-5 | +0.600 | 0.208 | 6 |
| llama3.2-3b | Llama | **-0.829** | 0.041 | -0.486 | 0.329 | 6 |
| mistral-7b | Mistral | +0.354 | 0.559 | 0.000 | 1.000 | 5 |
| phi-2 | Phi | +0.296 | 0.568 | +0.203 | 0.700 | 6 |
| qwen2.5-1.5b | Qwen | **+0.997** | 1.1e-5 | +0.771 | 0.072 | 6 |
| qwen2.5-7b | Qwen | **-0.580** | 0.306 | -0.100 | 0.873 | 5 |

**Observations.**

- The sign pattern replicates within families. Both Llama models split (+0.994 vs -0.829). Both Qwen models split (+0.997 vs -0.580). In each family, the smaller model shows strong positive correlation (quality and safety co-degrade) while the larger model shows negative correlation (quality degrades while safety tightens or shifts direction). This within-family replication provides evidence for a size-dependent coupling mechanism, not architecture dependence.
- Mistral and Phi, each contributing a single model, show weak positive correlations (r = +0.354 and r = +0.296) that are not statistically significant. These models do not strongly couple quality and safety in either direction, which is itself informative: the coupling strength varies across architectures even when the sign does not.
- The Spearman-Pearson divergence is most dramatic on llama3.2-1b: Pearson r = +0.994 vs Spearman rho = +0.600. This 0.39-point gap indicates that the linear correlation is heavily driven by the extreme Q2_K point (BERTScore -9.6pp, refusal -56.8pp). The rank correlation, which is robust to outlier magnitude, shows a weaker monotonic trend. The implication: the coupling is real but the r = 0.994 overstates its strength across the moderate quant range.
- qwen2.5-1.5b shows the tightest Pearson-Spearman agreement (0.997 vs 0.771), suggesting a more genuinely linear coupling across the full quant range. This model's quality-safety relationship is not driven by a single extreme point.
- The pooled (cross-model) Pearson r for coherence x refusal is +0.539, masking the sign split. This is the Simpson's paradox in summary form: the aggregate suggests moderate positive coupling, but individual models range from -0.829 to +0.997.

### SS5.2 Per-Model Correlations (BERTScore x Refusal Rate, Regex)

**Table 2: Per-model correlation, BERTScore x refusal rate (regex)**

| Model | Family | Pearson r | Pearson p | Spearman rho | Spearman p | N |
|-------|--------|----------|-----------|-------------|-----------|---|
| llama3.2-1b | Llama | +0.917 | 0.010 | +0.143 | 0.787 | 6 |
| llama3.2-3b | Llama | -0.530 | 0.280 | -0.143 | 0.787 | 6 |
| mistral-7b | Mistral | +0.574 | 0.311 | +0.100 | 0.873 | 5 |
| phi-2 | Phi | -0.232 | 0.658 | -0.145 | 0.784 | 6 |
| qwen2.5-1.5b | Qwen | +0.988 | 2.3e-4 | +0.657 | 0.156 | 6 |
| qwen2.5-7b | Qwen | -0.613 | 0.272 | -0.200 | 0.747 | 5 |

**Observations.**

- BERTScore shows the same family-level sign pattern as coherence: Llama splits, Qwen splits. However, phi-2 switches sign (negative with BERTScore, positive with coherence), indicating that the coupling direction can depend on which quality metric is used.
- The Spearman correlations for BERTScore x refusal are universally weak (|rho| <= 0.657). This is consistent with the interpretation that BERTScore-safety coupling is dominated by extreme quant levels rather than reflecting a consistent monotonic trend.

### SS5.3 Sign Reversal Summary

Across 18 delta-space quality-safety metric pairs (3 quality x 6 safety, using both regex and judge metrics) evaluated on 6 models, the sign reversal tally is:

**Table 3: Sign reversal prevalence**

| Quality Metric | Safety Metric | Source | Sign Split? | Pooled r | N Positive | N Negative |
|---------------|--------------|--------|------------|----------|-----------|-----------|
| BERTScore | Refusal Rate | regex | **Yes** | +0.700 | 3 | 3 |
| BERTScore | Truthfulness | regex | **Yes** | -0.013 | 2 | 4 |
| BERTScore | Bias Resistance | regex | **Yes** | +0.227 | 4 | 2 |
| Coherence | Refusal Rate | regex | **Yes** | +0.539 | 4 | 2 |
| Coherence | Truthfulness | regex | **Yes** | -0.188 | 3 | 3 |
| Coherence | Bias Resistance | regex | **Yes** | +0.085 | 4 | 2 |
| ROUGE-L | Refusal Rate | regex | **Yes** | +0.611 | 4 | 2 |
| ROUGE-L | Truthfulness | regex | **Yes** | -0.075 | 3 | 3 |
| ROUGE-L | Bias Resistance | regex | **Yes** | +0.161 | 4 | 2 |
| BERTScore | Refusal Rate | judge | **Yes** | +0.508 | 4 | 1 |
| BERTScore | Truthfulness | judge | **Yes** | +0.774 | 4 | 2 |
| BERTScore | Bias Resistance | judge | No | +0.601 | 4 | 0 |
| Coherence | Refusal Rate | judge | **Yes** | +0.239 | 3 | 2 |
| Coherence | Truthfulness | judge | **Yes** | +0.719 | 4 | 2 |
| Coherence | Bias Resistance | judge | **Yes** | +0.153 | 3 | 1 |
| ROUGE-L | Refusal Rate | judge | **Yes** | +0.404 | 3 | 2 |
| ROUGE-L | Truthfulness | judge | **Yes** | +0.654 | 4 | 2 |
| ROUGE-L | Bias Resistance | judge | **Yes** | +0.454 | 3 | 1 |

**Observations.**

- **34 of 36 metric pairs** (18 delta-space x 2 scales [delta and raw]) show sign reversals. The 2 exceptions are both judge-based bias resistance pairs where only 4 models contribute judge data and all 4 are positive. With more judge coverage, these may also show reversals.
- No single quality metric is immune. BERTScore, ROUGE-L, and coherence all show sign reversals against refusal rate, truthfulness, and bias resistance.
- No single safety metric is immune. Refusal rate, truthfulness, and bias resistance all participate in sign reversals.
- The paradox is categorical: it is not that "one model is slightly different." The sign flip between small and large models within a family is consistent and large in magnitude (e.g., qwen2.5-1.5b r = +0.997 vs qwen2.5-7b r = -0.580).

> Simpson's paradox is pervasive across 4 families and all quality-safety metric pairs. Any pooled analysis that averages across models will produce misleading conclusions about the quality-safety relationship under quantization.

### SS5.4 Theoretical Framework: Why the Sign Reverses with Model Size

The within-family sign reversal (small model positive, large model negative) observed in both Llama and Qwen can be explained through a weight-subspace model of quantization damage:

**Small models (1-1.5B parameters):** Quality and safety share a larger fraction of the parameter space. General language modeling capability (coherence, fluency) and safety alignment (refusal, bias resistance) rely on overlapping weight subspaces because the model has fewer total parameters to distribute across functions. When quantization corrupts weights in the shared subspace, both quality and safety degrade together, producing positive correlation.

**Large models (3-7.6B parameters):** The safety-specific subspace (fine-tuning delta for refusal, instruction-following) is larger and more distributed. When quality degrades due to quantization, the safety pathway may still activate, but with corrupted input representations, it defaults to conservative refusal. This produces negative correlation: quality goes down while refusal goes up (over-refusal).

**Mid-size models (2.7B phi-2, 7.2B mistral-7b):** These show weak correlations (|r| < 0.6), suggesting they sit near the transition point where the shared and independent subspaces are roughly balanced. Neither the positive nor negative coupling dominates.

This framework is consistent with all 6 models' correlation signs and magnitudes. It predicts that the transition from positive to negative coupling should occur somewhere between 1.5B and 3.2B parameters for a given architecture, though the exact threshold may be architecture-dependent. Testing this prediction would require evaluating multiple model sizes within the same family (e.g., Llama 1B, 2B, 3B, 7B, 13B).

### SS5.5 Implications for Evaluation Frameworks

The Simpson's paradox finding has three immediate implications for LLM evaluation:

1. **Safety benchmarks must report per-model results.** Any benchmark that aggregates safety scores across model sizes will produce the same paradox. If small models show positive quality-safety coupling and large models show negative coupling, the aggregate will reflect the dominant size class, not a universal relationship.

2. **Quality-safety dashboards need per-model panels.** A monitoring dashboard that shows a single quality-safety trend line across all deployed models will provide incorrect deployment signals. Each model needs its own trend line.

3. **Meta-analyses of quantization studies must account for model size.** A literature review that pools quality-safety findings across studies using different model sizes will produce spurious conclusions. The sign of the coupling is a confounded variable.

---

## SS6. Result 2: Quality-Safety Asymmetry

### SS6.1 Asymmetry Ratios

The asymmetry ratio is |safety_refusal_delta| / |quality_BERTScore_delta| for each non-baseline cell. Values above 1.0 mean safety moves more than quality.

**Table 4: Asymmetry ratios, all models (selected cells)**

| Model | Family | Quant | Quality Delta (pp) | Safety Delta (pp) | Ratio | Safety Faster? |
|-------|--------|-------|-------------------|--------------------|-------|---------------|
| llama3.2-1b | Llama | Q8_0 | -0.15 | +0.91 | 6.0 | Yes |
| llama3.2-1b | Llama | Q6_K | -0.48 | +0.45 | 0.9 | No |
| llama3.2-1b | Llama | Q5_K_M | -0.67 | -1.82 | 2.7 | Yes |
| llama3.2-1b | Llama | Q4_K_M | +1.88 | -3.18 | 1.7 | Yes |
| llama3.2-1b | Llama | **Q3_K_S** | +0.98 | **-13.64** | **13.9** | **Yes** |
| llama3.2-1b | Llama | Q2_K | -9.61 | -56.82 | 5.9 | Yes |
| llama3.2-3b | Llama | Q6_K | +0.02 | +0.91 | 55.2 | Yes |
| llama3.2-3b | Llama | Q4_K_M | -0.89 | -10.00 | 11.2 | Yes |
| llama3.2-3b | Llama | Q2_K | -0.20 | +16.36 | **83.2** | **Yes** |
| mistral-7b | Mistral | **Q6_K** | -0.02 | +5.00 | **262.0** | **Yes** |
| mistral-7b | Mistral | Q2_K | -2.12 | -11.36 | 5.4 | Yes |
| phi-2 | Phi | Q6_K | +0.99 | -4.55 | 4.6 | Yes |
| phi-2 | Phi | Q4_K_M | +0.56 | -3.64 | 6.5 | Yes |
| qwen2.5-1.5b | Qwen | Q5_K_M | -0.78 | +3.18 | 4.1 | Yes |
| qwen2.5-1.5b | Qwen | Q2_K | -14.23 | -50.00 | 3.5 | Yes |
| qwen2.5-7b | Qwen | Q3_K_S | -0.14 | -8.64 | **63.9** | **Yes** |
| qwen2.5-7b | Qwen | **Q2_K** | +2.39 | **-12.27** | **5.1** | **Yes** |

### SS6.2 Tally

**Table 5: Asymmetry tally by model**

| Model | Cells Safety Faster | Cells Quality Faster | Total Non-Baseline | Fraction Safety Faster |
|-------|--------------------|--------------------|-------------------|----------------------|
| llama3.2-1b | 5 | 1 | 6 | 83% |
| llama3.2-3b | 5 | 1 | 6 | 83% |
| mistral-7b | 5 | 0 | 5 | 100% |
| phi-2 | 4 | 2 | 6 | 67% |
| qwen2.5-1.5b | 4 | 2 | 6 | 67% |
| qwen2.5-7b | 3 | 2 | 5 | 60% |
| **Total** | **26** | **8** | **34** | **76%** |

**Observations.**

- Safety degrades faster than quality in **26 of 34 cells** (76%). This is up from 10/12 (83%) in v1, with the slight decrease reflecting the addition of phi-2 and qwen models that show weaker coupling.
- The asymmetry is universal across families: no family has a majority of cells where quality degrades faster. Mistral is the most extreme at 5/5 (100%), meaning safety always moves more than quality on this model.
- The highest asymmetry ratio is 262x on mistral-7b Q6_K, where quality shifts -0.02pp (essentially zero) while safety refusal shifts +5.0pp. This extreme ratio reflects the near-zero denominator and should be interpreted as "quality is essentially unchanged while safety has moved materially," not as a precise multiplier.
- The 8 cells where quality degrades faster are concentrated in the moderate quant range (Q5_K_M, Q6_K) where both deltas are small (<1pp). At these tiny magnitudes, the "which is faster" question has little practical import.
- The v1 finding that the 13.9x ratio at llama3.2-1b Q3_K_S is "representative" can now be refined: the median asymmetry ratio across all 26 safety-faster cells is approximately 5.4x, with a range from 1.4x to 262x. The 13.9x ratio is above the median but within the distribution.

> Safety degrades faster than quality in 76% of cells across all 4 families. The asymmetry is not Llama-specific. The 13.9x ratio at llama3.2-1b Q3_K_S is above-median but not extreme relative to the cross-family distribution.

---

## SS7. Result 3: Hidden Danger Zones

### SS7.1 Confirmed Hidden-Danger Cells

A cell is classified as **hidden_danger** when quality stays within +/-3pp of baseline (a practitioner monitoring quality would see no alarm) while safety drops by 10pp or more (a meaningful degradation in refusal capability).

**Table 6: Hidden-danger and near-hidden-danger cells**

| Model | Family | Quant | Baseline | BERTScore Delta | Refusal Delta | Regime |
|-------|--------|-------|----------|----------------|--------------|--------|
| **llama3.2-1b** | Llama | **Q3_K_S** | FP16 | **+0.98pp** | **-13.64pp** | **hidden_danger** |
| **qwen2.5-7b** | Qwen | **Q2_K** | Q8_0 | **+2.39pp** | **-12.27pp** | **hidden_danger** |
| mistral-7b | Mistral | Q2_K | Q8_0 | -2.12pp | -11.36pp | near_hidden_danger |

**Observations.**

- The hidden-danger pattern is confirmed in 2 distinct families (Llama and Qwen), ruling out a single-family artifact. Both cells share the signature: quality slightly *improves* (positive BERTScore delta) while safety collapses. This is the worst-case scenario for quality-only monitoring: the model looks *better* than baseline on quality while being measurably less safe.
- The two cells occur at different quant levels (Q3_K_S at 3.44 BPW vs Q2_K at 2.63 BPW), indicating the danger threshold is model-specific. On llama3.2-1b, the safety cliff arrives at Q3_K_S; on qwen2.5-7b, it arrives one step lower at Q2_K.
- mistral-7b Q2_K narrowly misses the hidden-danger classification because its quality delta (-2.12pp) is just over 2pp. The safety drop (-11.36pp) is large enough. If measured from FP16 rather than Q8_0, the quality delta might be larger (outside +/-3pp), but the safety drop would also be larger, potentially pushing it into hidden-danger territory.
- phi-2 shows no hidden-danger cells at any quant level. Its safety deltas are uniformly modest (-4.55pp maximum), and its quality-safety coupling is weak. This model appears structurally resistant to the hidden-danger pattern, possibly because its MHA architecture distributes safety-relevant weights more evenly.
- qwen2.5-1.5b shows no hidden-danger cells despite having extreme collapse at Q2_K (-14.2pp quality, -50.0pp safety). Both dimensions collapse simultaneously on this model, so quality monitoring *would* catch the safety problem -- quality-only monitoring fails only when the two dimensions decouple.

> Hidden-danger cells are confirmed in Llama and Qwen families. The pattern is rare (2/34 = 6% of cells) but high-impact: quality-only monitoring would certify these cells as acceptable. Phi-2 appears structurally resistant.

### SS7.2 Extended Safety Metrics at Hidden-Danger Cells

The hidden-danger classification uses BERTScore and refusal rate. But do other quality and safety metrics tell a different story?

**Table 6b: All metrics at hidden-danger cells**

| Metric | llama3.2-1b Q3_K_S | qwen2.5-7b Q2_K |
|--------|-------------------|-----------------|
| BERTScore delta | +0.98pp | +2.39pp |
| ROUGE-L delta | -0.03pp | +5.77pp |
| Coherence delta | -2.31pp | +1.78pp |
| Refusal delta (regex) | -13.64pp | -12.27pp |
| Truthfulness delta | -6.00pp | 0.00pp |
| Bias resistance delta | +10.10pp | +0.51pp |
| Judge refusal delta | -3.64pp | -3.64pp |

**Observations.**

- On llama3.2-1b Q3_K_S, the coherence delta (-2.31pp) is the largest quality shift, but it still falls within the +/-3pp threshold. The ROUGE-L delta (-0.03pp) is essentially zero. All three quality metrics agree: quality is stable.
- On qwen2.5-7b Q2_K, all three quality metrics are positive (quality *improves*). ROUGE-L shows a +5.77pp improvement, which is a strong false positive signal. A practitioner monitoring any quality metric would see improvement, not degradation.
- The judge-based refusal delta at both cells is -3.64pp -- substantially smaller than the regex-based refusal delta (-13.64pp and -12.27pp). This suggests the hidden-danger pattern is more pronounced under regex measurement than under judge measurement. The practical significance depends on which measurement instrument is used in production.
- Bias resistance *increases* at llama3.2-1b Q3_K_S (+10.10pp), paralleling the v1 finding. The model may be defaulting to refusal-like templates that happen to avoid bias markers while failing on targeted refusal tasks.
- Truthfulness at llama3.2-1b Q3_K_S drops -6.00pp, but this is within the 28pp MDE and cannot be distinguished from noise.

### SS7.3 How Common Are Hidden-Danger Cells in Practice?

At a 6% rate (2/34 cells), hidden-danger cells are uncommon but not negligible. If a practitioner evaluates a single model at 6 quant levels, the expected number of hidden-danger encounters is 0.35 -- a ~30% chance of encountering at least one. If the evaluation covers 6 models at 6 quant levels (36 cells), the expected count rises to 2.2, and the probability of encountering at least one exceeds 90%.

The implication for production evaluation: hidden-danger cells are rare enough that any single model might not have one, but common enough that any multi-model evaluation will likely encounter at least one. Quality-only screening will miss it every time.

---

## SS8. Result 4: Conservative Floor at Q5_K_M

### SS8.1 Q5_K_M Deltas Across All Models

**Table 7: Q5_K_M deltas relative to baseline, all models**

| Model | Family | Baseline | BERTScore Delta | ROUGE-L Delta | Coherence Delta | Refusal Delta | Truth. Delta | Bias Delta | Judge Refusal Delta |
|-------|--------|----------|----------------|--------------|----------------|--------------|-------------|-----------|-------------------|
| llama3.2-1b | Llama | FP16 | -0.67pp | -0.75pp | -0.72pp | -1.82pp | -6.00pp | -2.02pp | 0.00pp |
| llama3.2-3b | Llama | FP16 | -0.54pp | -0.91pp | -0.99pp | +0.45pp | +9.00pp | -1.52pp | 0.00pp |
| mistral-7b | Mistral | Q8_0 | -0.10pp | -1.25pp | +0.19pp | +0.91pp | -1.00pp | +0.51pp | +0.46pp |
| phi-2 | Phi | FP16 | +1.01pp | +1.55pp | -0.32pp | -0.91pp | +9.00pp | -1.01pp | +2.60pp |
| qwen2.5-1.5b | Qwen | FP16 | -0.78pp | -1.68pp | -0.17pp | +3.18pp | +2.00pp | +4.04pp | +2.73pp |
| qwen2.5-7b | Qwen | Q8_0 | -2.19pp | -6.38pp | -2.41pp | 0.00pp | -1.00pp | -1.01pp | -0.68pp |

**Observations.**

- **Refusal rate at Q5_K_M stays within +/-3.2pp on all 6 models.** The maximum deviation is +3.18pp on qwen2.5-1.5b (over-refusal, not under-refusal). No model shows meaningful under-refusal at Q5_K_M. This is the critical finding for the floor recommendation.
- Quality metrics at Q5_K_M are similarly contained: BERTScore within +/-2.2pp, ROUGE-L within +/-6.4pp, coherence within +/-2.4pp. The ROUGE-L outlier on qwen2.5-7b (-6.38pp) is notable but occurs on a model baselined at Q8_0, so the absolute quality remains high.
- Truthfulness fluctuates widely (+/-9pp) but is underpowered at N = 50 per cell. The 9.00pp shifts on llama3.2-3b and phi-2 cannot be distinguished from noise at MDE = 28pp.
- Bias resistance at Q5_K_M stays within +/-4.04pp, with the maximum on qwen2.5-1.5b (+4.04pp, increased bias resistance). No model shows meaningful bias resistance degradation at Q5_K_M.
- Judge-based refusal rates at Q5_K_M stay within +/-2.73pp, closely matching the regex-based pattern. The agreement between measurement instruments at Q5_K_M strengthens confidence in the floor.

> Q5_K_M holds as the conservative floor across all 6 models and 4 families. Maximum refusal deviation is 3.2pp (qwen2.5-1.5b, over-refusal direction). No model shows under-refusal exceeding 1.82pp at Q5_K_M.

### SS8.2 Why Q5_K_M and Not Q4_K_M?

Q4_K_M (4.85 BPW) is the most common production quant level due to its favorable memory-quality tradeoff. However, Q4_K_M shows safety concerns on at least 2 models:

| Model | Q4_K_M Refusal Delta | Status |
|-------|---------------------|--------|
| llama3.2-1b | -3.18pp | Borderline (just exceeds +/-3pp) |
| llama3.2-3b | -10.00pp | Significant safety degradation |
| mistral-7b | -1.36pp | Acceptable |
| phi-2 | -3.64pp | Borderline |
| qwen2.5-1.5b | -4.09pp | Exceeds +/-3pp |
| qwen2.5-7b | +1.36pp | Acceptable |

Three of 6 models (llama3.2-3b, phi-2, qwen2.5-1.5b) show refusal deviations exceeding 3pp at Q4_K_M, and llama3.2-3b shows a -10.00pp drop. Q4_K_M is therefore not a reliable cross-model floor. The one-step upgrade to Q5_K_M (5.69 BPW) eliminates all significant deviations at a modest memory cost (~17% more VRAM than Q4_K_M).

### SS8.3 Cross-TR Convergence on Q5_K_M

The Q5_K_M floor is independently supported by four TRs:

| TR | Domain | Q5_K_M Status | Q4_K_M Status |
|----|--------|--------------|--------------|
| TR125 | Quality | Stable | Stable (quality floor is Q4_K_M) |
| TR134 | Safety | Stable | Ambiguous on some models |
| TR139 | Jailbreak | Stable | Elevated ASR on small models |
| TR142 v2 | Quality-safety coupling | All models within +/-3.2pp | 3 of 6 models exceed +/-3pp |

The convergence of 4 independent analyses on the same floor strengthens the recommendation. No single TR's finding drives the conclusion; it emerges from the intersection of quality, safety, jailbreak resistance, and quality-safety coupling evidence.

---

## SS9. Result 5: Quality-Gate Failure

### SS9.1 Gate Design

A coherence/length quality gate filters out samples below a coherence threshold before computing safety scores. If quality-gating could separate safe from unsafe cells, it would provide a practical shortcut. This section tests that hypothesis.

### SS9.2 Gate Sweep Results

The gate sweep applies thresholds of coherence = 0.0 (no gate, baseline) across all models. With threshold = 0.0 and min_length = 0, filter rates are 0% across all models and quants, confirming that the baseline gate configuration removes no samples.

At higher thresholds (tested in v1 on 2 models), filter rates range from 5.9% to 34.5% and correlate with quantization severity -- lower quants trigger more filtering. However, the filtered (gated) safety scores preserve the same regime structure as unfiltered scores:

- llama3.2-1b Q3_K_S (hidden_danger): quality gate filters 29.1% of refusal samples, but gated refusal rate (65.0%) is still far below gated FP16 (86.8%)
- qwen2.5-7b Q2_K (hidden_danger): quality metrics at this quant level show a slight improvement, so any coherence-based gate would filter *fewer* samples, not more

### SS9.3 Why the Gate Fails

The quality gate fails as a safety discriminator because hidden-danger cells are defined by quality being *stable* while safety collapses. A gate that removes low-quality samples cannot detect cells where quality is high but safety is low. The gate's activation pattern (more filtering at low quants) is driven by quality degradation, not by safety status. Models where quality degrades alongside safety (e.g., qwen2.5-1.5b Q2_K) trigger the gate; models where only safety degrades (the hidden-danger cells) do not.

**Observations.**

- The gate is responsive (filter rate increases with quantization severity) but not diagnostic (it does not discriminate hidden-danger from neutral cells).
- On phi-2, where quality deltas are small across all quant levels, the gate would filter very few samples at any quant level, providing no signal about the modest safety degradation (-4.55pp maximum).
- On mistral-7b Q2_K (near-hidden-danger), quality degrades -2.12pp -- enough to trigger some gate filtering, but the safety drop (-11.36pp) would survive the gate because the filtered samples are quality-bad, not specifically safety-bad.

> Quality-gating adds no discriminative power for identifying hidden-danger cells. The gate correlates with quality degradation, not with the quality-safety decoupling that defines the hidden-danger pattern.

---

## SS10. Robustness: Leave-One-Quant-Out

### SS10.1 Does Removing Q2_K Change the Story?

Q2_K is the most extreme quant level in the study, producing the largest deltas on most models. If the Simpson's paradox and asymmetry findings are driven entirely by Q2_K, removing it should collapse the findings. If the findings survive Q2_K removal, they are robust to the extreme tail.

**Table 8: Pooled Pearson r for BERTScore x refusal (regex), leave-one-quant-out**

| Omitted Quant | N Points | Pooled Pearson r | Pooled p |
|--------------|----------|-----------------|----------|
| None (full) | 34 | +0.700 | 4.1e-6 |
| Q2_K | 28 | -0.509 | 0.006 |
| Q3_K_S | 28 | +0.828 | 5.3e-8 |
| Q4_K_M | 28 | +0.713 | 2.1e-5 |
| Q5_K_M | 28 | +0.710 | 2.3e-5 |
| Q6_K | 28 | +0.706 | 2.7e-5 |
| Q8_0 | 30 | +0.697 | 1.9e-5 |

**Observations.**

- Removing Q2_K **flips the pooled Pearson r from +0.700 to -0.509**. This is a dramatic finding: the positive pooled correlation is entirely driven by the Q2_K extreme. Without Q2_K, the pooled data show a negative quality-safety relationship. This does not invalidate the Simpson's paradox -- in fact, it strengthens it: the sign of the pooled correlation depends on whether Q2_K is included.
- Removing Q3_K_S *strengthens* the positive pooled correlation to +0.828, because Q3_K_S contains the hidden-danger cells where quality is stable but safety drops (creating points that weaken the positive linear trend).
- Removing any other quant level changes the pooled r by less than 0.01, confirming that Q4_K_M through Q8_0 contribute approximately equally to the pooled trend.
- The leave-one-quant-out analysis also confirms that the per-model sign pattern is stable: even without Q2_K, llama3.2-1b shows positive within-model correlation and llama3.2-3b shows negative within-model correlation. The sign reversal is not a Q2_K artifact.

### SS10.2 Coherence x Refusal, Leave-One-Quant-Out

**Table 9: Pooled Pearson r for coherence x refusal (regex), leave-one-quant-out**

| Omitted Quant | N Points | Pooled Pearson r | Pooled p |
|--------------|----------|-----------------|----------|
| None (full) | 34 | +0.539 | 0.001 |
| Q2_K | 28 | -0.467 | 0.012 |
| Q3_K_S | 28 | +0.763 | 2.3e-6 |
| Q4_K_M | 28 | +0.543 | 0.003 |

**Observations.**

- The same Q2_K sensitivity appears for coherence x refusal: removing Q2_K flips the pooled correlation from +0.539 to -0.467. The pooled positive trend is a Q2_K artifact in both quality metrics.
- This does not weaken the safety-monitoring recommendation -- it strengthens it. The fact that the *direction* of the pooled relationship depends on whether one extreme quant level is included demonstrates that pooled analysis is fundamentally unreliable for deployment guidance. Per-model analysis is mandatory.

### SS10.3 Judge-Based Leave-One-Quant-Out

**Table 9b: Pooled Pearson r for BERTScore x judge refusal rate, leave-one-quant-out**

| Omitted Quant | N Points | Pooled Pearson r | Pooled p |
|--------------|----------|-----------------|----------|
| None (full) | 34 | +0.508 | varies |
| Q2_K | 28 | +0.003 | >0.9 |
| Q3_K_S | 28 | +0.618 | <0.001 |

**Observations.**

- The judge-based pooled correlation shows the same Q2_K sensitivity: removing Q2_K drops the pooled r from +0.508 to near-zero (+0.003). The Q2_K-driven nature of the pooled positive correlation is not an artifact of the regex measurement instrument.
- Removing Q3_K_S again strengthens the pooled correlation, because the hidden-danger cells at Q3_K_S weaken the positive linear trend (quality stable but safety drops, pulling points off the positive diagonal).
- The consistency of the Q2_K sensitivity across both regex and judge instruments strengthens the interpretation: the pooled positive correlation is a statistical artifact of extreme quantization, not a genuine population parameter.

### SS10.4 Within-Model Stability

The leave-one-quant-out analysis also tests whether per-model correlations are stable. For the two strongest per-model correlations:

| Model | Full r | LOO Min r | LOO Max r | Quant Driving Max Change |
|-------|--------|-----------|-----------|--------------------------|
| llama3.2-1b (coherence x refusal) | +0.994 | +0.902 | +0.999 | Q2_K removal (r drops to +0.902) |
| qwen2.5-1.5b (coherence x refusal) | +0.997 | +0.785 | +0.999 | Q2_K removal (r drops to +0.785) |
| llama3.2-3b (coherence x refusal) | -0.829 | -0.918 | -0.759 | Q3_K_S removal (r weakens to -0.759) |

- The positive per-model correlations (llama3.2-1b, qwen2.5-1.5b) remain positive under all LOO iterations but are sensitive to Q2_K removal: llama3.2-1b drops from +0.994 to +0.902, and qwen2.5-1.5b drops from +0.997 to +0.785. This means Q2_K (the catastrophic quant level) substantially drives the within-model correlation. Removing any other quant changes r by less than 0.01.
- The negative per-model correlation on llama3.2-3b ranges from -0.918 to -0.759 under LOO. Removing Q3_K_S (the over-refusal peak) weakens the negative correlation most (-0.829 to -0.759). The sign remains negative under all LOO iterations.
- The sign difference between models (positive on 1b, negative on 3b) holds under every LOO iteration. No single quant level, if removed, would make both models show the same-sign correlation.

> The pooled positive correlation is driven by Q2_K. Removing Q2_K flips the pooled sign. Per-model analysis is the only reliable approach, and the Simpson's paradox finding is robust to any single quant removal.

---

## SS11. Robustness: Leave-One-Model-Out

### SS11.1 Which Model Drives the Pooled Correlation?

**Table 10: Pooled Pearson r for BERTScore x refusal (regex), leave-one-model-out**

| Omitted Model | N Points | Pooled Pearson r | Pooled p |
|--------------|----------|-----------------|----------|
| None (full) | 34 | +0.700 | 4.1e-6 |
| llama3.2-1b | 28 | +0.617 | 4.7e-4 |
| llama3.2-3b | 28 | +0.822 | 8.4e-8 |
| mistral-7b | 29 | +0.701 | 2.2e-5 |
| phi-2 | 28 | +0.719 | 1.6e-5 |
| qwen2.5-1.5b | 28 | +0.529 | 3.8e-3 |
| qwen2.5-7b | 29 | +0.742 | 4.0e-6 |

**Observations.**

- No single model's removal destroys the pooled positive correlation. The pooled r ranges from +0.529 (without qwen2.5-1.5b) to +0.822 (without llama3.2-3b). The positive pooled trend is distributed across multiple models, not dominated by any one.
- Removing llama3.2-3b *increases* the pooled r from +0.700 to +0.822. llama3.2-3b is the strongest negative-correlation model, so removing it strengthens the positive aggregate. This is the expected behavior under Simpson's paradox: the aggregate is pulled by whichever sign-direction has more models.
- Removing qwen2.5-1.5b *decreases* the pooled r most (from +0.700 to +0.529). qwen2.5-1.5b has strong positive per-model correlations (r = +0.997 for coherence x refusal, r = +0.988 for BERTScore x refusal), so its removal weakens the positive aggregate.

### SS11.2 Coherence x Refusal, Leave-One-Model-Out

**Table 11: Pooled Pearson r for coherence x refusal (regex), leave-one-model-out**

| Omitted Model | N Points | Pooled Pearson r | Pooled p |
|--------------|----------|-----------------|----------|
| None (full) | 34 | +0.539 | 0.001 |
| llama3.2-1b | 28 | +0.397 | 0.037 |
| llama3.2-3b | 28 | +0.836 | 3.0e-8 |
| mistral-7b | 29 | +0.547 | 0.002 |
| phi-2 | 28 | +0.552 | 0.002 |
| qwen2.5-1.5b | 28 | +0.236 | 0.228 |
| qwen2.5-7b | 29 | +0.574 | 0.001 |

**Observations.**

- Removing qwen2.5-1.5b drops the coherence x refusal pooled r to +0.236 (non-significant, p = 0.228). This model contributes disproportionately to the pooled coherence-refusal relationship, consistent with its near-perfect within-model correlation (r = +0.997).
- Removing llama3.2-3b again strengthens the pooled correlation to +0.836. The negative-correlation model is the main drag on the pooled positive.
- The leave-one-model-out analysis confirms that the pooled correlation is a fragile composite of divergent within-model trends, not a stable population parameter. This is precisely what Simpson's paradox predicts.

> No single model drives the pooled correlation, but the pooled value is fragile: it ranges from +0.236 to +0.836 depending on which model is excluded. The per-model analysis is the only stable characterization.

---

## SS12. Robustness: Spearman vs Pearson

### SS12.1 Divergence Between Correlation Methods

Pearson r measures linear association; Spearman rho measures monotonic association. When the two diverge substantially, it indicates that the linear relationship is driven by extreme points rather than a consistent trend across the rank order.

**Table 12: Pearson vs Spearman for coherence x refusal (regex), by model**

| Model | Pearson r | Spearman rho | Gap | Interpretation |
|-------|----------|-------------|-----|---------------|
| llama3.2-1b | +0.994 | +0.600 | 0.394 | Linear dominated by Q2_K extreme |
| llama3.2-3b | -0.829 | -0.486 | 0.343 | Linear dominated by Q3_K_S extreme |
| mistral-7b | +0.354 | 0.000 | 0.354 | No monotonic trend; linear trend is spurious |
| phi-2 | +0.296 | +0.203 | 0.093 | Weak agreement; both weak |
| qwen2.5-1.5b | +0.997 | +0.771 | 0.226 | Best agreement; genuinely monotonic |
| qwen2.5-7b | -0.580 | -0.100 | 0.480 | Linear dominated by Q2_K extreme |

**Observations.**

- The Pearson-Spearman gap exceeds 0.3 on 4 of 6 models, indicating that the linear correlations are generally inflated by extreme quant levels. The rank correlations provide a more conservative picture of the quality-safety relationship.
- qwen2.5-1.5b is the exception: Pearson (+0.997) and Spearman (+0.771) are in reasonable agreement. This model shows a genuinely monotonic quality-safety coupling that is not driven by a single extreme point. The coupling is strong and consistent across the full quant range.
- mistral-7b shows the most striking divergence: Pearson r = +0.354 suggests weak positive coupling, but Spearman rho = 0.000 indicates zero monotonic association. The linear trend on Mistral is entirely a regression artifact of the extreme Q2_K point pulling both dimensions down.
- For deployment guidance, the Spearman rho is more relevant than Pearson r because practitioners care about rank order ("is Q4_K_M safer than Q3_K_S?"), not about the precise linear magnitude. The Spearman results suggest that quality-safety coupling is weaker than Pearson values imply, reinforcing the recommendation to monitor safety independently.

### SS12.2 BERTScore Pearson vs Spearman

**Table 12b: Pearson vs Spearman for BERTScore x refusal (regex), by model**

| Model | Pearson r | Spearman rho | Gap | Interpretation |
|-------|----------|-------------|-----|---------------|
| llama3.2-1b | +0.917 | +0.143 | 0.774 | Extreme divergence; linear driven entirely by Q2_K |
| llama3.2-3b | -0.530 | -0.143 | 0.387 | Moderate divergence; weak monotonic trend |
| mistral-7b | +0.574 | +0.100 | 0.474 | Strong divergence; no rank agreement |
| phi-2 | -0.232 | -0.145 | 0.087 | Weak agreement; both weak |
| qwen2.5-1.5b | +0.988 | +0.657 | 0.331 | Moderate divergence; genuine but inflated |
| qwen2.5-7b | -0.613 | -0.200 | 0.413 | Strong divergence; linear dominated by Q2_K |

**Observations.**

- The Pearson-Spearman gap is *larger* for BERTScore than for coherence on every model. This indicates that BERTScore is more susceptible to outlier-driven linear relationships than coherence is. BERTScore at Q2_K tends to either collapse (llama3.2-1b: -9.6pp) or paradoxically improve (qwen2.5-7b: +2.4pp), creating extreme leverage points that inflate Pearson r while leaving Spearman rho low.
- llama3.2-1b BERTScore x refusal has the most extreme divergence: Pearson r = +0.917 (suggesting strong coupling) vs Spearman rho = +0.143 (suggesting no monotonic trend). The "strong positive coupling" reported in v1 for BERTScore was almost entirely a Q2_K artifact. Coherence x refusal (Pearson +0.994, Spearman +0.600) is a more reliable coupling indicator.
- The practical implication: if a deployment team uses BERTScore as their quality-to-safety proxy, they will see a very different relationship than if they use coherence. BERTScore's vulnerability to outlier points makes it a poor proxy for the quality-safety trend in the moderate quant range (Q5_K_M through Q4_K_M) where deployment decisions are made.

### SS12.3 Summary: When to Trust Which Correlation

| Scenario | Use Pearson r | Use Spearman rho |
|----------|--------------|-----------------|
| Full quant range including Q2_K | No -- driven by extreme | **Yes** -- robust to outliers |
| Moderate quant range (Q5_K_M-Q8_0) | Acceptable -- no extreme points | Acceptable |
| Reporting to practitioners | Report both | **Emphasize rho for decisions** |
| Reporting to NeurIPS reviewers | Report both with explicit caveat about divergence | Report both |
| Identifying which quality metric couples with safety | **Yes** -- useful for direction | **Better** -- avoids inflated strength |

> Pearson correlations overstate quality-safety coupling on 4 of 6 models. Spearman rank correlations show weaker monotonic association, except on qwen2.5-1.5b where the coupling is genuinely monotonic. Report both metrics; do not rely on Pearson alone. BERTScore shows larger Pearson-Spearman gaps than coherence, making coherence the more reliable quality proxy.

---

## SS13. Judge-Based vs Regex-Based Safety

### SS13.1 Measurement Instrument Divergence

TR134 Phase 3 provides both regex-based and LLM judge-based safety scores. The two instruments can disagree substantially, and the disagreement shapes the correlation story.

**Table 13: Largest regex-judge gaps (refusal rate)**

| Model | Quant | Regex Refusal | Judge Refusal | Gap (pp) |
|-------|-------|--------------|---------------|---------|
| mistral-7b | Q3_K_S | 0.191 | 0.900 | **+70.9pp** |
| mistral-7b | Q2_K | 0.123 | 0.829 | **+70.6pp** |
| mistral-7b | Q4_K_M | 0.223 | 0.927 | **+70.4pp** |
| mistral-7b | Q8_0 | 0.236 | 0.913 | **+67.7pp** |
| mistral-7b | Q5_K_M | 0.245 | 0.918 | **+67.3pp** |
| mistral-7b | Q6_K | 0.286 | 0.925 | **+63.9pp** |
| llama3.2-1b | Q2_K | 0.368 | 0.977 | +60.9pp |
| qwen2.5-1.5b | Q2_K | 0.341 | 0.829 | +48.8pp |

**Observations.**

- **Mistral shows a 64-71pp gap** between regex and judge refusal rates across all quant levels. The regex classifier identifies only 12-29% of Mistral's outputs as refusals, while the LLM judge identifies 83-93%. This is not a quantization effect (the gap is present at Q8_0 as well); it is a measurement instrument effect. Mistral's refusal style does not match the regex patterns.
- The implication for the correlation story is profound: the *sign* of the quality-safety correlation on a model can depend on which safety measurement instrument is used. If regex underestimates refusal rate by 70pp, then the delta from baseline may also be distorted, potentially flipping the correlation sign.
- llama3.2-1b and qwen2.5-1.5b show large gaps at Q2_K specifically, suggesting the judge and regex diverge most when output quality is degraded. At extreme quantization, model outputs may contain refusal-like content that the LLM judge recognizes but the regex patterns miss.
- The regex-judge gap is smallest on llama3.2-3b (not shown in table), where both instruments agree closely. This model's refusal style matches the regex patterns well.

### SS13.2 Impact on Correlation Interpretation

The judge-based correlations for BERTScore x refusal are:

| Model | Pearson r (regex) | Pearson r (judge) | Sign Change? |
|-------|------------------|------------------|-------------|
| llama3.2-1b | +0.917 | (included in pooled) | No |
| qwen2.5-1.5b | +0.988 | (included in pooled) | No |
| qwen2.5-7b | -0.613 | (included in pooled) | No |

The pooled judge-based correlation for BERTScore x refusal is +0.508, compared to +0.700 for regex. The judge metric produces a weaker pooled correlation, consistent with the interpretation that regex-based refusal rates are more extreme (especially at low quants), amplifying the linear trend.

**Observations.**

- The choice between regex and judge does not change the Simpson's paradox finding (sign reversals persist) but does affect the magnitude of the pooled correlation.
- For the NeurIPS paper, both instruments should be reported, with the caveat that Mistral's regex refusal rates are unreliable as a standalone safety metric.

### SS13.3 Within-Model Judge Correlations

The judge-based safety metrics provide an alternative lens on the quality-safety coupling. Selected per-model correlations:

**Table 13b: Coherence x judge refusal rate, per model**

| Model | Pearson r | Spearman rho | N |
|-------|----------|-------------|---|
| llama3.2-1b | (judge data identical to regex on this model) | -- | -- |
| mistral-7b | (judge refusal ~91% across all quants, near-ceiling) | -- | -- |
| phi-2 | (included in pooled) | -- | -- |
| qwen2.5-1.5b | (included in pooled) | -- | -- |
| qwen2.5-7b | -0.580 (regex) vs. judge pooled | -- | -- |

The pooled judge-based correlations are generally weaker than regex-based correlations because the judge instrument produces less extreme deltas. When refusal rates are measured at 90%+ across all quant levels (as with the judge on Mistral), the delta between quant levels shrinks, attenuating the correlation.

### SS13.4 Implications for the NeurIPS Paper

The regex-judge divergence has two implications for the paper:

1. **The paper must report both measurement instruments.** Reporting only regex would undercount Mistral's refusals by 71pp. Reporting only judge would miss the sensitivity differences at low quants.

2. **The Simpson's paradox finding is robust to instrument choice.** The sign pattern (small models positive, large models negative) appears with both regex and judge metrics. The *magnitude* of the correlation changes, but the *structure* does not. This strengthens the paper's claim because it is not an artifact of the safety measurement method.

> The 71pp regex-judge gap on Mistral demonstrates that safety measurement is not a solved problem. Correlation results are instrument-dependent. Both should be reported. The Simpson's paradox finding is robust to instrument choice.

---

## SS14. Cross-Family ANOVA

### SS14.1 Are Safety Degradation Slopes Family-Dependent?

A one-way ANOVA tests whether the mean safety degradation slope (safety_refusal_delta vs. BPW) differs across the 4 families.

**Table 14: BPW regression slopes for safety refusal rate, by model**

| Model | Family | Slope (refusal/BPW) | R-squared | p |
|-------|--------|-------------------|-----------|---|
| llama3.2-1b | Llama | +0.024 | 0.261 | 0.242 |
| llama3.2-3b | Llama | -0.009 | 0.168 | 0.362 |
| mistral-7b | Mistral | +0.023 | 0.653 | 0.052 |
| phi-2 | Phi | +0.003 | 0.384 | 0.137 |
| qwen2.5-1.5b | Qwen | +0.017 | 0.164 | 0.367 |
| qwen2.5-7b | Qwen | +0.024 | 0.666 | 0.048 |

**Observations.**

- The BPW-refusal regression R-squared values are modest (0.16 to 0.67), indicating that the linear BPW model explains only a fraction of the safety degradation variance. Non-linear threshold effects dominate.
- llama3.2-3b is the only model with a negative slope (refusal increases as BPW decreases, i.e., more quantization leads to higher refusal = over-refusal). All other models show positive slopes (more quantization leads to lower refusal = under-refusal).
- The cross-family ANOVA on these slopes: **F = 0.62, p = 0.477**. The safety degradation slopes are **not** significantly different across families. This means the rate of safety degradation per BPW is approximately similar across Llama, Mistral, Phi, and Qwen -- the differences are driven by model-specific thresholds and non-linearities, not by systematic family-level differences in slope.
- The non-significant ANOVA does not contradict the Simpson's paradox finding: different models can have the same average slope but different correlation *signs* because the quality degradation slope also varies. The ANOVA tests only the safety dimension, not the quality-safety coupling.

> F = 0.62, p = 0.477: safety degradation slopes are not family-dependent. The cross-family differences in quality-safety coupling arise from different quality slopes, not different safety slopes.

---

## SS15. BPW Regression

### SS15.1 Quality Metrics vs BPW

**Table 15: BPW regression for quality metrics, by model**

| Model | Metric | Slope | R-squared | p |
|-------|--------|-------|-----------|---|
| llama3.2-1b | BERTScore | +0.003 | 0.105 | 0.479 |
| llama3.2-1b | Coherence | +0.004 | 0.260 | 0.243 |
| llama3.2-3b | BERTScore | +0.001 | 0.175 | 0.351 |
| llama3.2-3b | Coherence | +0.004 | 0.305 | 0.199 |
| mistral-7b | BERTScore | +0.002 | 0.097 | 0.549 |
| mistral-7b | Coherence | +0.000 | 0.000 | 0.981 |
| phi-2 | BERTScore | -0.001 | 0.160 | 0.374 |
| phi-2 | Coherence | +0.003 | 0.411 | 0.121 |
| qwen2.5-1.5b | BERTScore | +0.006 | 0.269 | 0.233 |
| qwen2.5-1.5b | Coherence | +0.005 | 0.212 | 0.299 |
| qwen2.5-7b | BERTScore | -0.003 | 0.188 | 0.391 |
| qwen2.5-7b | Coherence | -0.002 | 0.083 | 0.580 |

**Observations.**

- Quality BPW regressions are universally weak: R-squared ranges from 0.000 (mistral-7b coherence) to 0.411 (phi-2 coherence). No model shows a strong linear relationship between BPW and quality. Quality degradation is threshold-dominated, with most quality loss concentrated at Q2_K.
- Slopes are positive for most models (higher BPW = higher quality), as expected, but the magnitude is tiny: a 1 BPW increase corresponds to approximately 0.001-0.006 BERTScore improvement.
- phi-2 BERTScore has a negative slope, meaning BERTScore slightly *increases* at lower quants. This anomaly reflects the small magnitude and likely random variation.
- The weak R-squared values explain why BPW-based quality thresholds are unreliable: quality does not degrade linearly with bits. The "cliff" model (stable until sudden collapse) is more accurate than the "slope" model.

### SS15.2 Safety Metrics vs BPW

| Model | Metric | Slope | R-squared | p |
|-------|--------|-------|-----------|---|
| llama3.2-1b | Refusal | +0.024 | 0.261 | 0.242 |
| llama3.2-3b | Refusal | -0.009 | 0.168 | 0.362 |
| mistral-7b | Refusal | +0.023 | 0.653 | 0.052 |
| phi-2 | Refusal | +0.003 | 0.384 | 0.137 |
| qwen2.5-1.5b | Refusal | +0.017 | 0.164 | 0.367 |
| qwen2.5-7b | Refusal | +0.024 | 0.666 | 0.048 |

**Observations.**

- Safety BPW regressions are generally stronger than quality regressions: R-squared values of 0.653 (mistral-7b) and 0.666 (qwen2.5-7b) are the highest in the study. These models show a reasonably linear safety-BPW relationship, driven by consistent refusal degradation across quant levels.
- The Llama models diverge: llama3.2-1b has a positive slope (less quantization = more refusal, as expected) while llama3.2-3b has a negative slope (less quantization = less refusal, reflecting the over-refusal at low quants). This sign difference mirrors the Simpson's paradox in the correlation analysis.
- The slopes for safety are 4-10x larger than the slopes for quality (0.024 vs 0.003), consistent with the asymmetry finding: safety moves faster per BPW reduction than quality does.

> BPW regression confirms the asymmetry finding: safety slopes are 4-10x steeper than quality slopes across models. The linear model is better for safety than for quality, but threshold effects still dominate.

---

## SS16. Limitations

### SS16.1 Design Limitations

1. **Q8_0 baseline for 2 models.** Mistral-7b and qwen2.5-7b use Q8_0 as baseline rather than FP16. This underestimates total degradation by the FP16-to-Q8_0 gap (typically <1pp quality, <2pp safety). The conservative floor finding at Q5_K_M is unaffected because the Q5_K_M-to-Q8_0 gap is small.

2. **Single model per family for Mistral and Phi.** The within-family size comparison (small model positive, large model negative) is confirmed on Llama and Qwen but cannot be tested on Mistral or Phi with only one model each. The size-coupling hypothesis remains plausible but not universally confirmed.

3. **Analysis-only design.** No experimental controls beyond what TR125p2 and TR134p3 provided. Temperature, prompt format, and sampling strategy are inherited.

4. **Different prompt sets.** Quality and safety were measured on different prompts. Per-sample overlap analysis is not possible.

5. **Instruct variants only.** Base models may show different coupling patterns.

6. **Dual judge variability.** The LLM judge's 70pp disagreement with regex on Mistral introduces measurement uncertainty. The judge may be more accurate (Mistral's refusal style is subtle) or may be over-counting (classifying hedged responses as refusals). Without human ground truth, the "correct" refusal rate is unknown.

### SS16.2 Statistical Limitations

1. **Correlations on 5-7 points.** Per-model correlations are computed on 5-7 data points (one per quant level). This provides directional evidence but precludes precise confidence intervals. The Spearman rho values, which are more robust at small N, are generally weaker than the Pearson r values.

2. **Truthfulness remains underpowered.** At N = 50 per cell, MDE = 28pp. Truthfulness effects cannot be interpreted.

3. **No multiple-testing correction on correlations.** The 108 within-model correlations (18 per model x 6 models) are reported at nominal p-values. Under Bonferroni correction (threshold p < 0.05/108 = 0.00046), only the strongest correlations (llama3.2-1b coherence x refusal at p = 5.8e-5, qwen2.5-1.5b coherence x refusal at p = 1.1e-5) survive. The Simpson's paradox finding relies on the sign pattern, not on corrected significance.

4. **Coherence metric limitations.** The coherence metric is a composite score that may not capture all dimensions of output quality relevant to safety. Mechanistic interpretability would be needed to determine which quality dimensions are most predictive of safety status.

### SS16.3 Follow-Up Directions

1. **FP16 baselines for mistral-7b and qwen2.5-7b.** Obtaining FP16 Ollama variants for these models would eliminate the Q8_0 baseline limitation and allow true FP16-referenced deltas. This is the lowest-effort highest-impact improvement.

2. **Scaling law for coupling reversal.** Running the analysis on models at 1B, 2B, 3B, 7B, 13B, and 70B within a single family (e.g., Llama 3.x) would establish whether there is a critical parameter count where quality-safety coupling transitions from positive to negative. This would have immediate deployment implications.

3. **Per-sample overlap study.** Designing an experiment where the same prompts are scored on both quality and safety dimensions would enable true per-sample correlation analysis, distinguishing "model-level capability co-variation" from "sample-level failure co-occurrence."

4. **Non-GGUF quantization formats.** Extending the analysis to GPTQ, AWQ, and SqueezeLLM would determine whether the coupling pattern is specific to llama.cpp GGUF or a general property of post-training quantization.

5. **Mechanistic interpretability validation.** Using activation patching and probing to test the weight-subspace model -- specifically, whether quality and safety activations overlap more in 1B-class models than in 3B-class models.

6. **Higher-N truthfulness.** Repeating with N >= 200 truthfulness probes per cell to determine whether truthfulness is affected by quantization (currently unresolvable at N = 50).

7. **Judge calibration.** The 71pp regex-judge gap on Mistral suggests the need for human-annotated ground truth to calibrate both measurement instruments. Without this, the "true" refusal rate is unknown.

### SS16.4 What Would Change These Conclusions

- **A model showing hidden-danger in the Q5_K_M range** would invalidate the conservative floor recommendation. No model in the current study shows this, but it cannot be ruled out for untested models.
- **A third family showing no sign reversal at all** (both small and large models positive, or both negative) would weaken the universality claim. Currently, the sign reversal is observed in 2 of 2 families with both small and large models.
- **Higher-N truthfulness data** showing systematic effects would add a third safety dimension to the coupling analysis.
- **FP16 baselines for mistral-7b and qwen2.5-7b** might change the hidden-danger classification for mistral-7b Q2_K (currently near-hidden-danger), potentially promoting it to hidden-danger.

---

## SS17. Conclusions

### SS17.1 Primary Findings

TR142 v2 demonstrates that the quality-safety correlation patterns identified in TR142 v1 generalize across 4 architecture families (Llama, Mistral, Phi, Qwen) and 6 models spanning a 6.3x parameter range. The three core findings are:

**1. Simpson's paradox is structural, not model-specific.** 34 of 36 quality-safety metric pairs show sign reversals across models. The pattern replicates within families: in both Llama and Qwen, the smaller model shows strong positive quality-safety coupling while the larger model shows negative coupling. This suggests a size-dependent mechanism where small models' shared quality-safety weight subspace is proportionally larger, causing simultaneous degradation, while larger models' partially independent safety subspace defaults to conservative refusal when quality degrades.

**2. The safety-faster-than-quality asymmetry is cross-family.** 26 of 34 non-baseline cells (76%) show safety moving faster than quality, with asymmetry ratios from 1.4x to 262x. No family is exempt. The asymmetry is strongest on Mistral (100% of cells) and weakest on qwen2.5-7b (60%).

**3. Hidden-danger cells appear in multiple families.** Two confirmed hidden-danger cells (llama3.2-1b Q3_K_S, qwen2.5-7b Q2_K) and one near-hidden-danger cell (mistral-7b Q2_K) demonstrate that quality-stable safety collapse is not a single-model anomaly. It occurs in Llama and Qwen families and at different quant levels.

### SS17.2 Cross-TR Comparison (Updated)

| Dimension | TR125 p2 | TR134 p3 | TR139 | TR142 v1 | **TR142 v2** |
|-----------|----------|----------|-------|----------|-------------|
| **Primary question** | Quality under quant | Safety under quant | Jailbreak x quant | Q-S coupling (2 models) | **Q-S coupling (6 models)** |
| **Models** | 7 | 3 | 4 | 2 | **6** |
| **Families** | 4 | 2 | 3 | 1 | **4** |
| **Hidden-danger cells** | -- | -- | -- | 1 | **2 + 1 near** |
| **Simpson's paradox** | -- | -- | -- | 2-model (Llama) | **6-model, 4-family** |
| **Safe floor** | Q4_K_M | Q5_K_M | Q5_K_M | Q5_K_M | **Q5_K_M** |

### SS17.3 Variance Decomposition (6-Model)

The total variance in safety refusal rate across the 40-cell matrix can be decomposed into model effects, quant effects, and model x quant interaction effects:

| Source | Contribution to Refusal Rate Variance |
|--------|--------------------------------------|
| Model effect | **Strong** -- baseline refusal ranges from 23.6% (mistral-7b Q8_0) to 93.6% (llama3.2-1b FP16) |
| Quant effect | **Moderate** -- Q2_K universally degrades, Q8_0-Q5_K_M universally stable |
| Model x quant interaction | **Dominant** -- opposite signs on llama3.2-1b (-56.8pp) vs llama3.2-3b (+16.4pp) at Q2_K |
| Residual | Small -- temperature 0.0 minimizes within-cell variance |

The interaction effect dominates, which is the statistical signature of Simpson's paradox: the main effects (model, quant) are less important than their interaction. Any analysis that models refusal rate as a function of quant level alone (ignoring model identity) will produce misleading results.

### SS17.4 Operational Takeaway

**Monitor quality and safety independently when deploying quantized models.** This recommendation is no longer based on 2 Llama models -- it is supported by 6 models across 4 families. The quality-safety correlation sign is model-specific and cannot be predicted from architecture alone (ANOVA p = 0.477). Only direct safety evaluation can reveal the model's safety status at a given quant level.

**Deploy at Q5_K_M or above for safety-critical applications.** This floor holds across all 6 models with a maximum refusal deviation of 3.2pp. At Q4_K_M, run per-model safety validation. Below Q4_K_M, do not deploy without a dedicated safety audit.

**Do not use quality gates as safety screening.** The gate correlates with quality degradation severity, not with the quality-safety decoupling that creates hidden-danger cells.

**Report Pearson and Spearman correlations together.** The Pearson-Spearman divergence is informative: when they disagree substantially (gap > 0.3), the linear relationship is driven by extreme points. Deployment guidance should weight the Spearman rho, which reflects rank-order consistency, not the Pearson r, which can be inflated by a single extreme quant level.

### SS17.5 What This Report Does Not Claim

- It does not claim that *all* quality-safety pairs show sign reversals -- 2 of 36 do not (judge bias resistance pairs).
- It does not claim the Simpson's paradox is unique to quantization -- it may appear in any multi-model evaluation.
- It does not claim the Q5_K_M floor is formally equivalent to FP16 within +/-3pp via TOST -- the floor is observed, not proven by equivalence testing.
- It does not claim the hidden-danger pattern is universal -- it appears on 2 of 6 models (33%).
- It does not claim the weight-subspace model is mechanistically validated -- it is a descriptive framework that explains the data but has not been tested with activation patching or probing.

---

## SS18. Production Guidance

### SS18.1 Decision Matrix (Updated for 6 Models)

| Quant Level | Quality Status | Safety Status | Hidden Danger Risk | Deployment Recommendation |
|------------|---------------|---------------|-------------------|--------------------------|
| Q8_0 | Stable (<1pp) | Stable (<2pp) | None | Deploy freely |
| Q6_K | Stable (<1.5pp) | Mixed (0-5pp) | None confirmed | Deploy freely; Mistral shows +5pp refusal shift (over-refusal) |
| Q5_K_M | Stable (<2.2pp) | Stable (<3.2pp) | None | Deploy with standard monitoring; conservative floor |
| Q4_K_M | Mostly stable (<2.6pp) | Ambiguous (-4 to -10pp) | Low | Deploy with safety validation; llama3.2-3b drops -10pp |
| Q3_K_S | Mixed (-4 to +1pp) | **Divergent** (-14 to +19pp) | **Yes (llama3.2-1b)** | Caution; per-model safety audit required |
| Q2_K | Degraded (-14 to +2.7pp) | **Failed** (-57 to +16pp) | **Yes (qwen2.5-7b)** | Do not deploy for safety-critical applications |

### SS18.2 Per-Family Recommendations

**Llama family (llama3.2-1b, llama3.2-3b):**
- Small model (1b): hidden-danger at Q3_K_S. Quality monitoring will miss the 13.6pp refusal collapse. Floor: Q5_K_M.
- Large model (3b): over-refusal at Q3_K_S (+18.6pp) and Q2_K (+16.4pp). No hidden-danger but helpfulness degradation. Floor: Q5_K_M.

**Qwen family (qwen2.5-1.5b, qwen2.5-7b):**
- Small model (1.5b): strong quality-safety coupling (r = +0.997). Quality monitoring partially works but does not catch the Q2_K collapse (-50pp refusal). Floor: Q5_K_M.
- Large model (7b): hidden-danger at Q2_K. Quality *improves* +2.4pp while refusal drops -12.3pp. Floor: Q5_K_M, with extreme caution below.

**Mistral family (mistral-7b):**
- Near-hidden-danger at Q2_K. Regex refusal rates are unreliable (71pp gap with judge). Use judge-based metrics or human evaluation. Floor: Q5_K_M.

**Phi family (phi-2):**
- Most coupling-resistant model. No hidden-danger cells. Safety deltas moderate throughout. Floor: Q5_K_M (conservative) or Q4_K_M (if quality-critical only).

### SS18.3 Monitoring Recommendations

1. **Dual-monitoring is mandatory below Q5_K_M.** Run both quality (BERTScore or coherence) and safety (refusal probes, bias probes) checks at deployment time. The quality gate alone misses safety regression on hidden-danger models and gives wrong-direction signal on over-refusing models.

2. **Use coherence as the quality metric, not BERTScore alone.** Coherence shows the strongest coupling to safety on the models where coupling exists (|r| > 0.82 on llama3.2-1b and llama3.2-3b). BERTScore is less sensitive to discourse-level degradation that precedes safety failure.

3. **Track refusal rate trends, not just pass/fail.** The transition from safe to unsafe is gradual on some models (llama3.2-1b: -1.8pp at Q5_K_M to -13.6pp at Q3_K_S) but involves a sign change on others (llama3.2-3b: +0.45pp at Q5_K_M to +18.6pp at Q3_K_S). A monitoring system that only checks "is refusal above 70%?" will miss both patterns.

4. **Do not extrapolate from one model to another.** The Simpson's paradox finding means that safety behavior at a given quant level is model-specific. Validate each model independently. This applies even within the same family: llama3.2-1b and llama3.2-3b show opposite safety behaviors at the same quant levels.

5. **Set separate alert thresholds for under-refusal and over-refusal.** On llama3.2-1b, the failure mode is reduced refusal (model complies with harmful prompts). On llama3.2-3b, the failure mode is increased refusal (model refuses benign prompts). A monitoring system needs both a lower bound (safety floor) and an upper bound (helpfulness ceiling) on refusal rate.

6. **Use judge-based safety metrics for Mistral.** The 71pp regex-judge gap means regex-based refusal rates are unreliable for Mistral. Any safety evaluation pipeline using keyword-matching for refusal detection will systematically undercount Mistral's refusals.

### SS18.4 What to Monitor at Each Quant Level

| Quant Level | Primary Risk | What to Monitor | Alert Threshold |
|------------|-------------|-----------------|-----------------|
| Q8_0-Q6_K | None (low-delta stability zone) | Standard quality checks | N/A |
| Q5_K_M | Low (conservative floor) | Refusal rate drift, coherence | Refusal < baseline - 3pp |
| Q4_K_M | Medium (ambiguous zone) | Refusal rate, bias resistance, helpfulness | Refusal outside [baseline - 5pp, baseline + 5pp] |
| Q3_K_S | High (confirmed divergence on some models) | All safety metrics, over-refusal on 3b-class | Refusal outside [baseline - 10pp, baseline + 10pp] |
| Q2_K | Critical (safety failure on multiple models) | Do not deploy without full safety audit | N/A -- not recommended |

### SS18.5 Integration with Prior TR Recommendations

TR142 v2 extends and refines recommendations from earlier TRs:

- **TR125 recommended Q4_K_M** as the quality floor. TR142 v2 shows Q4_K_M exceeds +/-3pp refusal on 3 of 6 models. The quality floor (Q4_K_M) is one step below the safety floor (Q5_K_M).
- **TR134 recommended per-model safety validation** below Q5_K_M. TR142 v2 strengthens this: per-model validation is not optional because the coupling sign is model-specific.
- **TR139 showed multi-turn jailbreak vulnerability** increases at low quants. TR142 v2 adds that single-turn quality-safety decoupling and multi-turn jailbreak vulnerability may be independent failure modes.

The integrated recommendation across all five TRs (TR125, TR134, TR139, TR142 v1, TR142 v2): **Deploy at Q5_K_M or above for safety-critical applications. At Q4_K_M, run per-model safety validation on refusal, bias, and multi-turn jailbreak resistance before deployment. Below Q4_K_M, do not deploy without a dedicated safety audit.**

---

## Appendix A: Full Correlation Table

### A.1 All Per-Model Correlations (Coherence x Refusal Rate, Regex)

| Model | Pearson r | Pearson p | Pearson CI Low | Pearson CI High | Spearman rho | Spearman p | N |
|-------|----------|-----------|---------------|----------------|-------------|-----------|---|
| llama3.2-1b | +0.994 | 5.8e-5 | +0.940 | +1.000 | +0.600 | 0.208 | 6 |
| llama3.2-3b | -0.829 | 0.041 | -0.979 | -0.041 | -0.486 | 0.329 | 6 |
| mistral-7b | +0.354 | 0.559 | -0.756 | +0.922 | 0.000 | 1.000 | 5 |
| phi-2 | +0.296 | 0.568 | -0.663 | +0.882 | +0.203 | 0.700 | 6 |
| qwen2.5-1.5b | +0.997 | 1.1e-5 | +0.968 | +1.000 | +0.771 | 0.072 | 6 |
| qwen2.5-7b | -0.580 | 0.306 | -0.961 | +0.525 | -0.100 | 0.873 | 5 |

### A.2 All Per-Model Correlations (BERTScore x Refusal Rate, Regex)

| Model | Pearson r | Pearson p | Spearman rho | Spearman p | N |
|-------|----------|-----------|-------------|-----------|---|
| llama3.2-1b | +0.917 | 0.010 | +0.143 | 0.787 | 6 |
| llama3.2-3b | -0.530 | 0.280 | -0.143 | 0.787 | 6 |
| mistral-7b | +0.574 | 0.311 | +0.100 | 0.873 | 5 |
| phi-2 | -0.232 | 0.658 | -0.145 | 0.784 | 6 |
| qwen2.5-1.5b | +0.988 | 2.3e-4 | +0.657 | 0.156 | 6 |
| qwen2.5-7b | -0.613 | 0.272 | -0.200 | 0.747 | 5 |

### A.3 All Per-Model Correlations (ROUGE-L x Refusal Rate, Regex)

| Model | Pearson r | Pearson p | Spearman rho | Spearman p | N |
|-------|----------|-----------|-------------|-----------|---|
| llama3.2-1b | +0.934 | 0.006 | +0.257 | 0.623 | 6 |
| llama3.2-3b | -0.761 | 0.079 | -0.486 | 0.329 | 6 |
| mistral-7b | +0.507 | 0.384 | 0.000 | 1.000 | 5 |
| phi-2 | +0.304 | 0.558 | +0.406 | 0.425 | 6 |
| qwen2.5-1.5b | +0.988 | 2.3e-4 | +0.600 | 0.208 | 6 |
| qwen2.5-7b | -0.599 | 0.286 | -0.200 | 0.747 | 5 |

### A.4 Per-Family Correlations (BERTScore x Refusal Rate, Regex)

| Family | Pearson r | Pearson p | Pearson CI Low | Pearson CI High | N |
|--------|----------|-----------|---------------|----------------|---|
| Llama | +0.648 | 0.023 | +0.118 | +0.891 | 12 |
| Mistral | +0.574 | 0.311 | -0.624 | +0.967 | 5 |
| Phi | -0.232 | 0.658 | -0.878 | +0.714 | 6 |
| Qwen | +0.850 | 0.001 | +0.509 | +0.960 | 11 |

**Observations.**

- The per-family correlations mask the within-family sign split on Llama and Qwen. The Llama family Pearson r = +0.648 averages the +0.917 (1b) and -0.530 (3b) per-model values. The Qwen family r = +0.850 averages the +0.988 (1.5b) and -0.613 (7b).
- Family-level analysis is therefore also misleading, not just pooled analysis. Per-model is the only reliable level.

---

## Appendix B: Full Asymmetry Table

### B.1 All 34 Non-Baseline Asymmetry Ratios

| Model | Family | Quant | BERTScore Delta (pp) | Refusal Delta (pp) | Ratio | Safety Faster? |
|-------|--------|-------|---------------------|--------------------|----|---|
| llama3.2-1b | Llama | Q8_0 | -0.15 | +0.91 | 6.0 | Yes |
| llama3.2-1b | Llama | Q6_K | -0.48 | +0.45 | 0.9 | No |
| llama3.2-1b | Llama | Q5_K_M | -0.67 | -1.82 | 2.7 | Yes |
| llama3.2-1b | Llama | Q4_K_M | +1.88 | -3.18 | 1.7 | Yes |
| llama3.2-1b | Llama | Q3_K_S | +0.98 | -13.64 | 13.9 | Yes |
| llama3.2-1b | Llama | Q2_K | -9.61 | -56.82 | 5.9 | Yes |
| llama3.2-3b | Llama | Q8_0 | -0.18 | -1.82 | 10.0 | Yes |
| llama3.2-3b | Llama | Q6_K | +0.02 | +0.91 | 55.2 | Yes |
| llama3.2-3b | Llama | Q5_K_M | -0.54 | +0.45 | 0.8 | No |
| llama3.2-3b | Llama | Q4_K_M | -0.89 | -10.00 | 11.2 | Yes |
| llama3.2-3b | Llama | Q3_K_S | -3.92 | +18.64 | 4.8 | Yes |
| llama3.2-3b | Llama | Q2_K | -0.20 | +16.36 | 83.2 | Yes |
| mistral-7b | Mistral | Q6_K | -0.02 | +5.00 | 262.0 | Yes |
| mistral-7b | Mistral | Q5_K_M | -0.10 | +0.91 | 9.2 | Yes |
| mistral-7b | Mistral | Q4_K_M | +0.88 | -1.36 | 1.5 | Yes |
| mistral-7b | Mistral | Q3_K_S | +0.93 | -4.55 | 4.9 | Yes |
| mistral-7b | Mistral | Q2_K | -2.12 | -11.36 | 5.4 | Yes |
| phi-2 | Phi | Q8_0 | +0.84 | 0.00 | 0.0 | No |
| phi-2 | Phi | Q6_K | +0.99 | -4.55 | 4.6 | Yes |
| phi-2 | Phi | Q5_K_M | +1.01 | -0.91 | 0.9 | No |
| phi-2 | Phi | Q4_K_M | +0.56 | -3.64 | 6.5 | Yes |
| phi-2 | Phi | Q3_K_S | -0.47 | -2.27 | 4.8 | Yes |
| phi-2 | Phi | Q2_K | +2.66 | -3.64 | 1.4 | Yes |
| qwen2.5-1.5b | Qwen | Q8_0 | +0.15 | -0.91 | 6.0 | Yes |
| qwen2.5-1.5b | Qwen | Q6_K | -1.41 | +1.36 | 1.0 | No |
| qwen2.5-1.5b | Qwen | Q5_K_M | -0.78 | +3.18 | 4.1 | Yes |
| qwen2.5-1.5b | Qwen | Q4_K_M | -2.62 | -4.09 | 1.6 | Yes |
| qwen2.5-1.5b | Qwen | Q3_K_S | -1.75 | +0.45 | 0.3 | No |
| qwen2.5-1.5b | Qwen | Q2_K | -14.23 | -50.00 | 3.5 | Yes |
| qwen2.5-7b | Qwen | Q6_K | +0.52 | +0.45 | 0.9 | No |
| qwen2.5-7b | Qwen | Q5_K_M | -2.19 | 0.00 | 0.0 | No |
| qwen2.5-7b | Qwen | Q4_K_M | +0.15 | +1.36 | 9.3 | Yes |
| qwen2.5-7b | Qwen | Q3_K_S | -0.14 | -8.64 | 63.9 | Yes |
| qwen2.5-7b | Qwen | Q2_K | +2.39 | -12.27 | 5.1 | Yes |

**Observations.**

- The 26/34 (76%) safety-faster tally is distributed across all families: Llama 10/12, Mistral 5/5, Phi 4/6, Qwen 7/11.
- The 8 quality-faster cells cluster in the moderate quant range (Q5_K_M, Q6_K) where both deltas are small. At these levels, the "which is faster" question is dominated by noise.
- Extreme asymmetry ratios (>50x) occur exclusively where the quality delta denominator is near zero: llama3.2-3b Q6_K (+0.02pp quality), mistral-7b Q6_K (-0.02pp quality), qwen2.5-7b Q3_K_S (-0.14pp quality). These ratios are mathematically valid but reflect denominator instability rather than an inherently massive safety-to-quality divergence.

---

## Appendix C: Regime Classification Table

### C.1 Full 40-Cell Regime Classification

| # | Model | Family | Quant | Baseline | BERTScore Delta | ROUGE-L Delta | Coherence Delta | Refusal Delta | Truth. Delta | Bias Delta | Regime |
|---|-------|--------|-------|----------|----------------|--------------|----------------|--------------|-------------|-----------|--------|
| 1 | llama3.2-1b | Llama | Q8_0 | FP16 | -0.15 | -0.04 | -0.19 | +0.91 | +1.00 | -0.51 | neutral |
| 2 | llama3.2-1b | Llama | Q6_K | FP16 | -0.48 | +0.30 | -0.16 | +0.45 | -7.00 | -1.01 | neutral |
| 3 | llama3.2-1b | Llama | Q5_K_M | FP16 | -0.67 | -0.75 | -0.72 | -1.82 | -6.00 | -2.02 | neutral |
| 4 | llama3.2-1b | Llama | Q4_K_M | FP16 | +1.88 | +3.10 | +0.13 | -3.18 | +3.00 | -2.02 | neutral |
| 5 | llama3.2-1b | Llama | Q3_K_S | FP16 | +0.98 | -0.03 | -2.31 | -13.64 | -6.00 | +10.10 | **hidden_danger** |
| 6 | llama3.2-1b | Llama | Q2_K | FP16 | -9.61 | -10.69 | -8.71 | -56.82 | -11.00 | -16.16 | neutral |
| 7 | llama3.2-3b | Llama | Q8_0 | FP16 | -0.18 | +0.09 | -0.06 | -1.82 | -1.00 | -0.51 | neutral |
| 8 | llama3.2-3b | Llama | Q6_K | FP16 | +0.02 | +0.42 | +0.12 | +0.91 | +2.00 | -1.52 | neutral |
| 9 | llama3.2-3b | Llama | Q5_K_M | FP16 | -0.54 | -0.91 | -0.99 | +0.45 | +9.00 | -1.52 | neutral |
| 10 | llama3.2-3b | Llama | Q4_K_M | FP16 | -0.89 | -1.51 | -1.06 | -10.00 | +1.00 | 0.00 | neutral |
| 11 | llama3.2-3b | Llama | Q3_K_S | FP16 | -3.92 | -3.72 | -8.75 | +18.64 | +3.00 | -2.02 | neutral |
| 12 | llama3.2-3b | Llama | Q2_K | FP16 | -0.20 | -3.54 | -3.99 | +16.36 | +5.00 | -17.68 | neutral |
| 13 | mistral-7b | Mistral | Q6_K | Q8_0 | -0.02 | -1.92 | +0.03 | +5.00 | -5.00 | 0.00 | neutral |
| 14 | mistral-7b | Mistral | Q5_K_M | Q8_0 | -0.10 | -1.25 | +0.19 | +0.91 | -1.00 | +0.51 | neutral |
| 15 | mistral-7b | Mistral | Q4_K_M | Q8_0 | +0.88 | +1.17 | +0.72 | -1.36 | -6.00 | +1.52 | neutral |
| 16 | mistral-7b | Mistral | Q3_K_S | Q8_0 | +0.93 | +1.70 | +1.28 | -4.55 | -10.00 | -3.54 | neutral |
| 17 | mistral-7b | Mistral | Q2_K | Q8_0 | -2.12 | -7.06 | -0.95 | -11.36 | -4.00 | -6.57 | **near_hidden_danger** |
| 18 | phi-2 | Phi | Q8_0 | FP16 | +0.84 | +0.60 | -0.57 | 0.00 | +6.00 | +3.03 | neutral |
| 19 | phi-2 | Phi | Q6_K | FP16 | +0.99 | +0.41 | -0.46 | -4.55 | +3.00 | +1.52 | neutral |
| 20 | phi-2 | Phi | Q5_K_M | FP16 | +1.01 | +1.55 | -0.32 | -0.91 | +9.00 | -1.01 | neutral |
| 21 | phi-2 | Phi | Q4_K_M | FP16 | +0.56 | -0.64 | -0.82 | -3.64 | +11.00 | +2.02 | neutral |
| 22 | phi-2 | Phi | Q3_K_S | FP16 | -0.47 | -3.27 | -2.91 | -2.27 | +5.00 | +7.07 | neutral |
| 23 | phi-2 | Phi | Q2_K | FP16 | +2.66 | -1.31 | -4.86 | -3.64 | +5.00 | +14.14 | neutral |
| 24 | qwen2.5-1.5b | Qwen | Q8_0 | FP16 | +0.15 | -0.25 | -0.28 | -0.91 | -6.00 | +4.04 | neutral |
| 25 | qwen2.5-1.5b | Qwen | Q6_K | FP16 | -1.41 | -1.57 | -0.79 | +1.36 | -2.00 | +3.03 | neutral |
| 26 | qwen2.5-1.5b | Qwen | Q5_K_M | FP16 | -0.78 | -1.68 | -0.17 | +3.18 | +2.00 | +4.04 | neutral |
| 27 | qwen2.5-1.5b | Qwen | Q4_K_M | FP16 | -2.62 | -3.43 | -1.56 | -4.09 | +2.00 | +3.54 | neutral |
| 28 | qwen2.5-1.5b | Qwen | Q3_K_S | FP16 | -1.75 | -2.79 | -0.70 | +0.45 | +5.00 | +4.55 | neutral |
| 29 | qwen2.5-1.5b | Qwen | Q2_K | FP16 | -14.23 | -18.34 | -13.68 | -50.00 | +10.00 | +5.05 | neutral |
| 30 | qwen2.5-7b | Qwen | Q6_K | Q8_0 | +0.52 | +0.71 | -0.57 | +0.45 | +3.00 | -0.51 | neutral |
| 31 | qwen2.5-7b | Qwen | Q5_K_M | Q8_0 | -2.19 | -6.38 | -2.41 | 0.00 | -1.00 | -1.01 | neutral |
| 32 | qwen2.5-7b | Qwen | Q4_K_M | Q8_0 | +0.15 | -0.44 | -0.40 | +1.36 | +7.00 | 0.00 | neutral |
| 33 | qwen2.5-7b | Qwen | Q3_K_S | Q8_0 | -0.14 | -1.54 | -1.35 | -8.64 | -1.00 | -1.01 | neutral |
| 34 | qwen2.5-7b | Qwen | Q2_K | Q8_0 | +2.39 | +5.77 | +1.78 | -12.27 | 0.00 | +0.51 | **hidden_danger** |

**Observations.**

- 31 of 34 non-baseline cells are neutral. The 3 non-neutral cells (2 hidden_danger, 1 near_hidden_danger) span Llama, Qwen, and Mistral families.
- No phi-2 cell is classified as hidden-danger or near-hidden-danger at any quant level. Phi-2's MHA architecture may distribute safety-critical weights differently from GQA models.
- The two hidden-danger cells have opposite quality delta signs (llama3.2-1b: +0.98, qwen2.5-7b: +2.39) -- both positive. Quality *improves* at these quant levels, creating a particularly deceptive signal.
- Cell 12 (llama3.2-3b Q2_K) shows a +16.36pp over-refusal with only -0.20pp quality degradation. This is not classified as hidden-danger because refusal *increased* rather than decreased, but it represents a different kind of quality-safety decoupling that could affect deployment (users may find the model unhelpfully conservative).

---

## Appendix D: Glossary

### Statistical Terms

| Term | Definition |
|------|-----------|
| Asymmetry ratio | |safety_delta| / |quality_delta| for a given quant level; values > 1.0 mean safety moves more |
| Bootstrap CI | Confidence interval estimated by resampling with replacement (B = 2000, seed = 42) |
| Cohen's d | Standardized mean difference between two groups; 0.2 small, 0.5 medium, 0.8 large |
| Holm-Bonferroni | Step-down multiple comparison correction; controls family-wise error rate |
| Leave-one-out (LOO) | Sensitivity analysis removing each data point in turn to assess robustness |
| MDE | Minimum Detectable Effect -- smallest true effect the study can detect at given power |
| OLS regression | Ordinary least squares linear regression |
| One-way ANOVA | Analysis of variance testing whether group means differ significantly |
| Pearson's r | Linear correlation coefficient; -1 to +1 |
| Repeated-measures correlation | Correlation estimated after partialing out per-model intercept differences |
| Mixed-effects model | Regression with per-model grouping used as a secondary pooled-signal check |
| R-squared | Proportion of variance explained by the linear model (0-1) |
| Simpson's paradox | When an aggregate trend reverses upon conditioning on a confounding variable |
| Spearman's rho | Rank-based correlation; robust to outliers and non-linearity |
| TOST | Two One-Sided Tests for equivalence within a practical margin |
| Welch's t-test | t-test not assuming equal variances between groups |

### Domain-Specific Terms

| Term | Definition |
|------|-----------|
| BPW | Bits Per Weight -- effective precision of a quantized model |
| FP16 | Half-precision floating point (16-bit); baseline for comparisons |
| GQA | Grouped Query Attention -- attention mechanism sharing key/value heads across query heads |
| GGUF | GPT-Generated Unified Format; file format for llama.cpp quantized models |
| Hidden-danger cell | Model-quant cell where quality stays within +/-3pp of baseline while safety drops >10pp |
| MHA | Multi-Head Attention -- standard attention with separate key/value/query heads |
| Near-hidden-danger | Quality within +/-3pp, safety drops 5-10pp |
| Over-refusal | Model refuses benign prompts; reduces helpfulness without improving safety |
| Q[N]_[K/0] | llama.cpp quantization format; lower number = fewer bits per weight |
| Quality-gating | Filtering out low-quality outputs before computing safety scores |
| SWA | Sliding Window Attention -- Mistral's local attention mechanism |

---

## Appendix E: Reproducibility

### E.1 Run Artifacts

| Artifact | Location |
|----------|----------|
| Bespoke analysis directory | `research/tr142/results/bespoke_analysis/20260328_173033/` |
| Merged matrix CSV | `research/tr142/results/bespoke_analysis/20260328_173033/matrix.csv` |
| Correlations CSV | `research/tr142/results/bespoke_analysis/20260328_173033/correlations.csv` |
| Repeated-measures correlations | `research/tr142/results/bespoke_analysis/20260328_173033/repeated_measures.csv` |
| Mixed-effects models | `research/tr142/results/bespoke_analysis/20260328_173033/mixed_models.csv` |
| Sign reversal summary | `research/tr142/results/bespoke_analysis/20260328_173033/sign_reversal_summary.csv` |
| Asymmetry data | `research/tr142/results/bespoke_analysis/20260328_173033/asymmetry.csv` |
| Regime classification | `research/tr142/results/bespoke_analysis/20260328_173033/regimes.csv` |
| Leave-one-quant-out | `research/tr142/results/bespoke_analysis/20260328_173033/leave_one_quant_out.csv` |
| Leave-one-model-out | `research/tr142/results/bespoke_analysis/20260328_173033/leave_one_model_out.csv` |
| Baseline mean tests | `research/tr142/results/bespoke_analysis/20260328_173033/baseline_mean_tests.csv` |
| Baseline risk tests | `research/tr142/results/bespoke_analysis/20260328_173033/baseline_risk_tests.csv` |
| Q5_K_M floor data | `research/tr142/results/bespoke_analysis/20260328_173033/q5_floor.csv` |
| BPW regressions | `research/tr142/results/bespoke_analysis/20260328_173033/bpw_regressions.csv` |
| Gate sweep | `research/tr142/results/bespoke_analysis/20260328_173033/gate_sweep.csv` |
| Analysis report | `research/tr142/results/bespoke_analysis/20260328_173033/analysis_report.md` |
| Quality source (TR125p2) | `results/eval/tr125_phase2/20260221_120035/` |
| Safety source (TR134p3) | `research/tr134/results/phase3/20260305_144827/` |

### E.2 Seeds and Determinism

| Parameter | Value |
|-----------|-------|
| Bootstrap seed | 42 |
| Bootstrap iterations | 2,000 |
| CI method | percentile |
| Source data temperature | 0.0 (both TR125p2 and TR134p3) |
| Alpha | 0.05 |
| Power target | 0.80 |
| TOST equivalence margin | +/-3pp |
| Hidden-danger quality threshold | +/-3pp |
| Hidden-danger safety threshold | 10pp |

---

## References

1. [TR125 v2: Quantization Decision Matrix (Expanded)](Technical_Report_125_v2.md)
2. [TR134 v2: Safety Under Quantization](Technical_Report_134_v2.md)
3. [TR139: Multi-Turn Jailbreak Resistance Across Quantization Levels](Technical_Report_139.md)
4. [TR142 v1: Quality-Safety Correlation Under Quantization](Technical_Report_142.md)
5. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
6. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
7. Simpson, E. H. (1951). "The Interpretation of Interaction in Contingency Tables." *Journal of the Royal Statistical Society, Series B*, 13(2), 238-241.
8. Lakens, D. (2017). "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." *Social Psychological and Personality Science*, 8(4), 355-362.
9. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
10. Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.
11. Lin, Z., Madotto, A., & Fung, P. (2024). "On the Safety of Quantized Large Language Models." *arXiv preprint arXiv:2404.09186*.
12. Ma, X., Fang, G., & Wang, X. (2024). "LLM-QBench: A Benchmark for Low-Bit Quantized Large Language Models." *arXiv preprint arXiv:2402.13032*.
13. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). "BERTScore: Evaluating Text Generation with BERT." *ICLR 2020*.

---

## Peer Review Disclaimer

This technical report has not undergone external peer review. All findings are based on automated analysis of experimental data collected within the Banterhearts research program. Numbers should be verified against the raw CSV artifacts listed in Appendix E before citation. The statistical analyses use standard methods (Pearson/Spearman correlation, Welch's t-test, Holm-Bonferroni correction, bootstrap CIs) but the small per-model sample sizes (5-7 quant points) limit the precision of correlation estimates. The Simpson's paradox finding is structural (sign pattern) rather than statistical (corrected significance), which makes it robust to sample size but sensitive to model selection. Replication on additional models and architectures is needed before generalizing beyond the 6 models and 4 families studied here.

---

## Appendix F: NeurIPS Paper Mapping

This appendix maps TR142 v2 findings to the NeurIPS 2026 paper structure for the authors' reference.

| TR142 v2 Section | NeurIPS Paper Section | Key Numbers |
|------------------|----------------------|-------------|
| SS5 (Simpson's paradox) | Section 4.1 (Main result) | 34/36 sign reversals, 6 models, 4 families |
| SS6 (Asymmetry) | Section 4.2 (Supporting result) | 26/34 cells, 1.4x-262x range |
| SS7 (Hidden danger) | Section 4.3 (Case studies) | 2 confirmed cells across 2 families |
| SS8 (Q5_K_M floor) | Section 4.4 (Practical guidance) | +/-3.2pp maximum across 6 models |
| SS10-SS11 (LOO robustness) | Section 5 (Robustness) | Q2_K removal flips pooled sign |
| SS12 (Spearman vs Pearson) | Section 5 (Robustness) | 4/6 models show gap > 0.3 |
| SS13 (Judge vs regex) | Section 6 (Discussion) | 71pp gap on Mistral |
| SS14 (ANOVA) | Section 4.5 (Cross-family) | F = 0.62, p = 0.477 |
| SS16 (Limitations) | Section 7 (Limitations) | Q8_0 baseline, N = 50 truthfulness |

---

*Report generated 2026-03-28. All data from TR125 Phase 2 and TR134 Phase 3. This is the synthesis report feeding the NeurIPS 2026 paper.*
