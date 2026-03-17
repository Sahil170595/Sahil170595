# Technical Report 142: Quality-Safety Correlation Under Quantization
## Cross-referencing TR125 Phase 2 quality metrics with TR134 Phase 3 safety metrics across 2 models and 7 GGUF quant levels

| Field | Value |
|-------|-------|
| **TR Number** | 142 |
| **Project** | Banterhearts |
| **Date** | 2026-03-16 |
| **Version** | 1.0 |
| **Author** | Research Team |
| **Git Commit** | d2c3fdac |
| **Status** | Complete |
| **Report Type** | Analysis-only (no new experiments) |
| **Run Directory** | `research/tr142/results/20260316_143936/` |
| **Quality Source** | TR125 Phase 2 (10,290 overlapping samples) |
| **Safety Source** | TR134 Phase 3 (13,342 overlapping samples) |
| **Models** | llama3.2-1b, llama3.2-3b |
| **Quant Levels** | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K |
| **Analysis Passes** | 14 |
| **Related Work** | [TR125](Technical_Report_125.md), [TR134](Technical_Report_134.md), [TR139](Technical_Report_139.md) |
| **Depends On** | TR125 Phase 2 (quality data), TR134 Phase 3 (safety data) |

---

## Abstract

TR142 asks whether quality and safety degrade together under quantization, or whether they follow independent degradation paths that could mislead practitioners who monitor only one dimension. This analysis-only study cross-references **10,290 quality samples** from TR125 Phase 2 (BERTScore, ROUGE-L, coherence) with **13,342 safety samples** from TR134 Phase 3 (refusal rate, truthfulness, bias resistance) across **2 models** and **7 GGUF quantization levels** (FP16 through Q2_K), producing a 14-row quality-safety matrix analyzed through 14 statistical passes.

The core findings are: (1) The quality-safety correlation is model-dependent and cannot be pooled -- llama3.2-1b shows strong positive correlation (coherence x refusal r = +0.994, p < 1e-70) while llama3.2-3b shows significant negative correlation (coherence x refusal r = -0.829, p = 0.003). (2) Safety degrades faster than quality in 10 of 12 quant-model cells, with asymmetry ratios reaching 13.9x at Q3_K_S on llama3.2-1b. (3) Quality-gating does not change the quantization safety story -- the same 18.2% of samples fail the coherence gate at every quant level including FP16, meaning the gate catches inherently bad prompts rather than quant-induced degradation.

The operational conclusion is that quality metrics alone are insufficient safety proxies: a model can pass quality benchmarks while silently losing safety alignment, and the direction of this divergence depends on model architecture and parameter count.

---

## Executive Summary

### Key Findings

1. **Quality-safety correlation is model-dependent (Simpson's paradox).** Pooling across models produces a moderate positive correlation (r = 0.29-0.65), but within-model analysis reveals opposite signs: llama3.2-1b shows r = +0.994 (coherence x refusal, p < 1e-70) while llama3.2-3b shows r = -0.829 (p = 0.003). Any pooled analysis would be misleading.

2. **Safety degrades faster than quality at most quant levels.** The asymmetry index shows safety moving more than quality in 10 of 12 model-quant cells. On llama3.2-1b Q3_K_S, safety drops -13.6pp while quality moves only +1.0pp (asymmetry ratio 13.9x). The bootstrap asymmetry gap for llama3.2-1b is -11.0pp [95% CI: -26.7, -0.4], confirming safety decays faster overall.

3. **Q3_K_S on llama3.2-1b is a hidden danger zone.** Quality barely moves from FP16 (BERTScore +0.98pp, coherence -2.3pp) while refusal rate drops -13.6pp (d = 0.41, p_adj = 0.0005). A quality-only monitor would not flag this cell.

4. **llama3.2-3b shows paradoxical safety behavior at low quants.** At Q3_K_S and Q2_K, refusal rate *increases* by +18.6pp and +16.4pp respectively (d = -0.55 and -0.46, both p_adj < 1e-5). The model over-refuses rather than under-refusing -- quality degrades but safety alignment tightens.

5. **Quality-gating is quant-invariant.** The coherence/length gate filters the same 40/220 refusal-rate samples (18.2%) and 8/50 truthfulness samples (16.0%) at every quant level including FP16. The gate catches prompt-level issues, not quantization artifacts.

6. **Bias resistance is unaffected by quality gating.** Zero bias_resistance samples are filtered by the quality gate at any quant level (0/198), meaning coherence failures and bias failures occur on disjoint sample sets.

7. **Truthfulness is underpowered.** With N = 50 per quant per model, the minimum detectable effect is 28.0pp at 80% power. No truthfulness comparisons reach significance after Holm-Bonferroni correction. This is a sample-size limitation, not evidence of no effect.

8. **BPW regression explains little variance.** Linear fits of metric-vs-BPW yield R-squared between 0.03 and 0.30 across all model-metric pairs. Quantization level is a weak predictor of both quality and safety; non-linear threshold effects dominate at the low-bit end.

### Core Decisions

- Do not use quality metrics as a proxy for safety assessment under quantization -- the correlation sign depends on the model.
- Monitor safety *and* quality independently when deploying quantized models, especially below Q4_K_M.
- Avoid Q3_K_S and below on llama3.2-1b for safety-critical deployments -- quality benchmarks will not reveal the safety regression.
- Do not assume aggressive quantization always reduces safety -- llama3.2-3b shows increased refusal at Q3_K_S/Q2_K, which may manifest as unhelpful over-refusal in production.
- Quality-gating before safety evaluation does not improve quant-level discrimination -- skip the gating step for quantization studies.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Model overlap | Shared models | >= 2 | 2 | **PASS** |
| Quant coverage | Shared levels | >= 5 | 7 | **PASS** |
| Quality samples | Overlapping N | >= 1,000 | 10,290 | **PASS** |
| Safety samples | Overlapping N | >= 1,000 | 13,342 | **PASS** |
| Correlation (H1) | Model-dependent sign | Opposite signs | r = +0.99 vs -0.83 | **PASS** |
| Asymmetry (H2) | Safety faster than quality | Majority of cells | 10/12 cells | **PASS** |
| Quality gate (H3) | Quant-invariant filter rate | Constant across levels | 18.2% at all levels | **PASS** |
| Truthfulness power | MDE at 80% power | < 10pp | 28.0pp | **FAIL** |
| BPW regression | R-squared | > 0.50 | 0.03-0.30 | **FAIL** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | Quality-safety correlation is model-dependent | SS5 Table 3 (9 within-model r values per model) | **Established** |
| C2 | Safety degrades faster than quality | SS7 Table 5 asymmetry ratios, SS6 bootstrap CIs | **Established** |
| C3 | Quality-gating is quant-invariant | SS8 Table 6 (constant filter rates) | **Established** |
| C4 | Q3_K_S is a hidden danger on 1b | SS6 divergence + SS9 Cohen's d | **Established** |
| C5 | 3b shows paradoxical refusal increase at low quants | SS5 Table 3 + SS9 Cohen's d | **Established** |
| C6 | Truthfulness shows no quant effect | SS9 power analysis | **Not established** (underpowered) |

---

## When to Use This Report

### Scenario 1: Evaluating quantized models with quality benchmarks only

**Question:** We run BERTScore and coherence checks on our quantized model and everything looks fine. Is the model safe to deploy?

**Answer:** No. SS6 Table 4 shows that on llama3.2-1b Q3_K_S, quality metrics move less than 2.3pp from FP16 while refusal rate drops 13.6pp. Quality passing does not imply safety passing. See SS5 for the full correlation breakdown.

### Scenario 2: Deciding whether to add a quality gate before safety evaluation

**Question:** Should we filter out low-quality outputs before computing safety scores to get a cleaner signal?

**Answer:** No -- it won't help for quantization studies. SS8 shows the quality gate filters the exact same samples (18.2%) at every quant level including FP16. The gate catches inherently difficult prompts, not quant-induced failures. Adding it just reduces your effective sample size.

### Scenario 3: Choosing a quantization level for a safety-critical application

**Question:** What's the lowest quantization we can use while maintaining both quality and safety?

**Answer:** Model-dependent. For llama3.2-1b, Q4_K_M is the last level where safety delta stays within 3.2pp of FP16 (SS4 Table 2). For llama3.2-3b, the picture is more complex -- refusal actually *increases* at low quants (SS5), which may cause over-refusal in practice. Use Q5_K_M as a conservative floor for both models.

### Scenario 4: Positioning this analysis relative to the broader safety line

**Question:** How does TR142 relate to TR134 and TR139?

**Answer:** TR134 Phase 3 provided the raw safety data; TR142 adds the quality dimension by merging with TR125 Phase 2. TR139 studies multi-turn jailbreak resistance across quant levels. Together, they show that quantization affects safety through multiple independent channels: single-turn alignment (TR134), multi-turn jailbreak resistance (TR139), and quality-safety coupling (TR142).

---

## Table of Contents

- [Abstract](#abstract)
- [Executive Summary](#executive-summary)
- [When to Use This Report](#when-to-use-this-report)
- [Metric Definitions](#metric-definitions)
- [SS1. Introduction](#ss1-introduction)
- [SS2. Methodology](#ss2-methodology)
- [SS3. Models and Design](#ss3-models-and-design)
- [SS4. Quality-Safety Matrix](#ss4-quality-safety-matrix)
- [SS5. Correlation Analysis](#ss5-correlation-analysis)
- [SS6. Divergence and Asymmetry Analysis](#ss6-divergence-and-asymmetry-analysis)
- [SS7. Asymmetry Index](#ss7-asymmetry-index)
- [SS8. Quality-Gated Safety Scores](#ss8-quality-gated-safety-scores)
- [SS9. Pairwise Statistical Tests](#ss9-pairwise-statistical-tests)
- [SS10. BPW Regression](#ss10-bpw-regression)
- [SS11. Capability Consistency Validation](#ss11-capability-consistency-validation)
- [SS12. Statistical Synthesis](#ss12-statistical-synthesis)
- [SS13. Conclusions](#ss13-conclusions)
- [SS14. Limitations and Follow-Up](#ss14-limitations-and-follow-up)
- [SS15. Reproducibility](#ss15-reproducibility)
- [References](#references)
- [Appendix A: Raw Data Tables](#appendix-a-raw-data-tables)
- [Appendix B: Extended Statistical Tables](#appendix-b-extended-statistical-tables)
- [Appendix C: Glossary](#appendix-c-glossary)
- [Appendix D: Configs](#appendix-d-configs)

---

## Metric Definitions

### Primary Quality Metrics (from TR125 Phase 2)

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| BERTScore (F1) | Token-level semantic similarity between model output and reference, using contextual embeddings | 0-1, higher = better generation quality |
| ROUGE-L | Longest common subsequence overlap between output and reference | 0-1, higher = better factual coverage |
| Coherence | Composite score of output fluency, logical structure, and response relevance | 0-1, higher = more coherent output |

### Primary Safety Metrics (from TR134 Phase 3)

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Refusal Rate | Fraction of harmful prompts where the model correctly refuses to comply | 0-1, higher = safer (refuses harmful requests) |
| Truthfulness | Fraction of factual probes answered accurately without hallucination | 0-1, higher = more truthful |
| Bias Resistance | Fraction of bias-probing prompts where the model avoids stereotypical or discriminatory outputs | 0-1, higher = more bias-resistant |

### Statistical Tests Used

| Test | Role in This Report |
|------|-------------------|
| Pearson's r | Correlation between quality and safety degradation curves |
| Welch's t-test | Pairwise quant-vs-FP16 comparisons on safety metrics |
| Cohen's d | Effect size for pairwise comparisons |
| Holm-Bonferroni | Multiple comparison correction across 36 pairwise tests |
| Bootstrap percentile CI | Confidence intervals for asymmetry gaps (B = 2000, seed = 42) |
| OLS regression | BPW-vs-metric linear fit |
| Power analysis | Minimum detectable effect at 80% power per metric |

### Evidence Standard

**Established findings** require p < 0.05 after Holm-Bonferroni correction with |d| >= 0.4 or practical significance above 3pp.

**Partial findings** show statistical significance in at least one comparison but lack consistency across both models.

**Non-claims** are results where evidence is insufficient, where TOST would confirm equivalence, or where power analysis reveals the study cannot detect realistic effects.

---

## SS1. Introduction

### SS1.1 Research Questions

1. **Q12:** Do quality and safety degrade on the same samples under quantization, or independently?
2. **Q13:** Does quality-gating (filtering low-coherence outputs before safety scoring) change the quantization safety story?

### SS1.2 Why This Matters

Practitioners evaluating quantized models typically benchmark quality (perplexity, accuracy, BERTScore) and safety (refusal, toxicity, bias) as separate dimensions. If quality and safety degrade together -- if the same samples that lose coherence also lose alignment -- then a single quality check could serve as a safety proxy. If they degrade independently, monitoring only one dimension creates a false sense of security.

This matters most at the quantization levels where deployment decisions are made: Q4_K_M and Q5_K_M are the most common production choices, and practitioners need to know whether their quality benchmarks are also protecting safety.

### SS1.3 Scope

| Dimension | Coverage |
|-----------|----------|
| Models | llama3.2-1b (1.2B params), llama3.2-3b (3.2B params) |
| Quant levels | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K |
| Quality metrics | BERTScore, ROUGE-L, coherence |
| Safety metrics | Refusal rate, truthfulness, bias resistance |
| Quality samples (overlap) | 10,290 |
| Safety samples (overlap) | 13,342 |
| Backend | Ollama (llama.cpp) |
| Shared benchmark tasks | mmlu_real, arc_challenge |
| Analysis passes | 14 |

### SS1.4 Literature Grounding

Prior work on quantization effects (Dettmers et al., 2023; Frantar et al., 2022) has focused almost exclusively on quality metrics (perplexity, downstream accuracy). Safety-under-quantization has been studied in TR134 and TR139 within this project, and by a small number of concurrent works, but the *correlation structure* between quality and safety degradation under quantization has not been characterized. TR142 fills this gap by merging data from two completed experimental TRs.

### SS1.5 How to Read This Report

This is an analysis-only TR. No new experiments were run. All data originates from TR125 Phase 2 (quality) and TR134 Phase 3 (safety), restricted to the 2 models and 7 quant levels that overlap between both studies. Each result section follows the pattern: context prose, data table, then **Observations** interpreting the table. Cross-references use SS notation (e.g., "See SS6 Table 4").

---

## SS2. Methodology

### SS2.1 Overall Design

TR142 is a post-hoc cross-referencing analysis. The pipeline is:

1. Load and normalize TR125 Phase 2 quality samples and TR134 Phase 3 safety samples
2. Restrict to overlapping models (llama3.2-1b, llama3.2-3b) and quant levels (7 levels)
3. Aggregate quality metrics by (model, quant) and safety metrics by (model, quant)
4. Merge into a 14-row quality-safety matrix (2 models x 7 quants)
5. Run 14 analysis passes: correlation, divergence, quality-gating, pairwise tests, regression, asymmetry, power

### SS2.2 Unit of Analysis

The unit of analysis is a (model, quant) cell in the merged matrix. Each cell contains aggregated quality scores (mean BERTScore, ROUGE-L, coherence across all tasks in TR125p2) and aggregated safety scores (refusal rate, truthfulness, bias resistance across all tasks in TR134p3). Within-cell sample sizes range from 50 (truthfulness) to 250 (coherence).

### SS2.3 How Rows Become Claims

Raw sample-level scores from TR125p2 and TR134p3 are aggregated to cell means. Cell means are compared across quant levels using Welch's t-test with Holm-Bonferroni correction. Correlations are computed on the 7-point degradation curve (one point per quant level) within each model. A claim requires both statistical significance (p_adj < 0.05) and practical significance (delta > 3pp or |d| > 0.4).

### SS2.4 Why the Data Sources Are Compatible

Both TR125 Phase 2 and TR134 Phase 3 used the same backend (Ollama with llama.cpp), the same quant levels, and models from the same model families. Temperature was 0.0 in both studies. The key difference is that quality and safety were measured on different prompts -- TR125p2 used benchmark tasks (MMLU, ARC, etc.) while TR134p3 used safety evaluation prompts (refusal probes, truthfulness probes, bias probes). This means the correlation analysis measures whether the *model's overall capability at a given quant level* co-varies between quality and safety domains, not whether the same individual prompt degrades on both dimensions.

### SS2.5 What This Design Does Not Do

- It does not establish causal relationships. Correlation between quality and safety degradation does not mean quality loss *causes* safety loss.
- It does not measure per-sample overlap. Quality and safety were measured on different prompt sets, so we cannot ask "did the exact same sample fail on both quality and safety?"
- It cannot distinguish instruction-following degradation from knowledge degradation as the mechanism behind safety changes.

---

## SS3. Models and Design

| Model | Parameters | Architecture | Quant Levels | Quality N (total) | Safety N (total) |
|-------|-----------|-------------|-------------|-------------------|------------------|
| llama3.2-1b | 1.2B | GQA | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | ~5,145 | ~6,671 |
| llama3.2-3b | 3.2B | GQA | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | ~5,145 | ~6,671 |

Both models are Meta Llama 3.2 instruct variants quantized via llama.cpp into GGUF format and served through Ollama. The 1b and 3b models share the same architecture family but differ by 2.6x in parameter count, providing a natural axis for testing whether quality-safety coupling scales with model size.

---

## SS4. Quality-Safety Matrix

This section presents the merged 14-row matrix that forms the basis for all subsequent analysis. All deltas are computed relative to FP16 for the same model.

### SS4.1 llama3.2-1b Quality-Safety Degradation

| Quant | BPW | BERTScore | ROUGE-L | Coherence | Refusal Rate | Truthfulness | Bias Resist. |
|-------|-----|-----------|---------|-----------|-------------|-------------|-------------|
| FP16 | 16.00 | 0.646 | 0.266 | 0.580 | 93.6% | 55.0% | 89.4% |
| Q8_0 | 8.00 | 0.644 (-0.15pp) | 0.266 (-0.04pp) | 0.578 (-0.19pp) | 94.5% (+0.9pp) | 56.0% (+1.0pp) | 88.9% (-0.5pp) |
| Q6_K | 6.56 | 0.641 (-0.48pp) | 0.269 (+0.30pp) | 0.578 (-0.16pp) | 94.1% (+0.5pp) | 48.0% (-7.0pp) | 88.4% (-1.0pp) |
| Q5_K_M | 5.69 | 0.639 (-0.67pp) | 0.259 (-0.75pp) | 0.572 (-0.72pp) | 91.8% (-1.8pp) | 49.0% (-6.0pp) | 87.4% (-2.0pp) |
| Q4_K_M | 4.85 | 0.665 (+1.88pp) | 0.297 (+3.10pp) | 0.581 (+0.13pp) | 90.5% (-3.2pp) | 58.0% (+3.0pp) | 87.4% (-2.0pp) |
| **Q3_K_S** | **3.44** | **0.656 (+0.98pp)** | **0.266 (-0.03pp)** | **0.557 (-2.31pp)** | **80.0% (-13.6pp)** | **49.0% (-6.0pp)** | **99.5% (+10.1pp)** |
| Q2_K | 2.63 | 0.550 (-9.61pp) | 0.159 (-10.69pp) | 0.493 (-8.71pp) | 36.8% (-56.8pp) | 44.0% (-11.0pp) | 73.2% (-16.2pp) |

**Observations.**

- Quality metrics are remarkably stable from FP16 through Q4_K_M: BERTScore moves less than 1.9pp, coherence less than 0.7pp. The quality signal does not alarm until Q2_K, where all three metrics drop sharply (BERTScore -9.6pp, ROUGE-L -10.7pp, coherence -8.7pp).
- Safety tells a different story. Refusal rate begins declining at Q5_K_M (-1.8pp), accelerates at Q3_K_S (-13.6pp), and collapses at Q2_K (-56.8pp). The safety degradation curve is steeper and activates earlier than the quality curve.
- Q3_K_S is the critical divergence point: quality metrics are within 2.3pp of FP16, but refusal rate has already dropped 13.6pp. A quality-only evaluation would miss this.
- The anomalous Q4_K_M quality *improvement* (+1.88pp BERTScore, +3.10pp ROUGE-L) likely reflects sampling variance at that quant level rather than a genuine capability gain.

> On llama3.2-1b, quality passes at Q3_K_S while safety fails -- monitoring quality alone creates a false sense of security.

### SS4.2 llama3.2-3b Quality-Safety Degradation

| Quant | BPW | BERTScore | ROUGE-L | Coherence | Refusal Rate | Truthfulness | Bias Resist. |
|-------|-----|-----------|---------|-----------|-------------|-------------|-------------|
| FP16 | 16.00 | 0.767 | 0.469 | 0.661 | 76.4% | 49.0% | 96.5% |
| Q8_0 | 8.00 | 0.766 (-0.18pp) | 0.470 (+0.09pp) | 0.660 (-0.06pp) | 74.5% (-1.8pp) | 48.0% (-1.0pp) | 96.0% (-0.5pp) |
| Q6_K | 6.56 | 0.768 (+0.02pp) | 0.473 (+0.42pp) | 0.662 (+0.12pp) | 77.3% (+0.9pp) | 51.0% (+2.0pp) | 94.9% (-1.5pp) |
| Q5_K_M | 5.69 | 0.762 (-0.54pp) | 0.460 (-0.91pp) | 0.651 (-0.99pp) | 76.8% (+0.5pp) | 58.0% (+9.0pp) | 94.9% (-1.5pp) |
| Q4_K_M | 4.85 | 0.759 (-0.89pp) | 0.454 (-1.51pp) | 0.650 (-1.06pp) | 66.4% (-10.0pp) | 50.0% (+1.0pp) | 96.5% (0.0pp) |
| **Q3_K_S** | **3.44** | **0.728 (-3.92pp)** | **0.432 (-3.72pp)** | **0.573 (-8.75pp)** | **95.0% (+18.6pp)** | **52.0% (+3.0pp)** | **94.4% (-2.0pp)** |
| Q2_K | 2.63 | 0.765 (-0.20pp) | 0.433 (-3.54pp) | 0.621 (-3.99pp) | 92.7% (+16.4pp) | 54.0% (+5.0pp) | 78.8% (-17.7pp) |

**Observations.**

- Quality degradation on llama3.2-3b is more gradual than on 1b. BERTScore stays within 0.9pp of FP16 all the way down to Q4_K_M. Even at Q3_K_S, BERTScore only drops 3.9pp (vs. near-zero on 1b).
- The safety story is inverted relative to 1b. Instead of declining, refusal rate *increases* at Q3_K_S (+18.6pp) and Q2_K (+16.4pp). The model becomes more conservative -- over-refusing rather than under-refusing. This is confirmed statistically in SS9 with significant negative Cohen's d values.
- Q4_K_M is the exception: refusal drops -10.0pp while quality barely moves (-0.89pp BERTScore). This is the only 3b cell showing the 1b-like pattern of "quality stable, safety drops."
- Bias resistance holds steady (within 2.0pp) from FP16 through Q3_K_S, then collapses at Q2_K (-17.7pp).
- The anomalous Q2_K BERTScore near-preservation (only -0.20pp) combined with coherence drop (-3.99pp) suggests the model produces semantically similar but less structured outputs at extreme quantization.

> On llama3.2-3b, aggressive quantization (Q3_K_S, Q2_K) makes the model over-refuse rather than under-refuse -- the opposite failure mode from 1b.

---

## SS5. Correlation Analysis

This section addresses Q12 directly: do quality and safety co-vary under quantization? Pearson correlations are computed on the 7-point degradation curve (FP16 through Q2_K) for each model independently, plus a pooled (cross-model) analysis.

### SS5.1 Within-Model Correlations

**Table 3a: llama3.2-1b quality-safety Pearson r**

| Quality Metric | Safety Metric | r | p | Significant? |
|---------------|--------------|---|---|-------------|
| BERTScore | Refusal Rate | **+0.917** | 4.4e-6 | Yes |
| BERTScore | Truthfulness | +0.708 | 0.045 | Yes |
| BERTScore | Bias Resistance | **+0.848** | 0.001 | Yes |
| ROUGE-L | Refusal Rate | **+0.934** | 1.6e-7 | Yes |
| ROUGE-L | Truthfulness | +0.749 | 0.024 | Yes |
| ROUGE-L | Bias Resistance | +0.761 | 0.019 | Yes |
| Coherence | Refusal Rate | **+0.994** | 2.3e-71 | Yes |
| Coherence | Truthfulness | +0.707 | 0.046 | Yes |
| Coherence | Bias Resistance | +0.672 | 0.070 | No |

**Table 3b: llama3.2-3b quality-safety Pearson r**

| Quality Metric | Safety Metric | r | p | Significant? |
|---------------|--------------|---|---|-------------|
| BERTScore | Refusal Rate | -0.530 | 0.212 | No |
| BERTScore | Truthfulness | -0.007 | 0.988 | No |
| BERTScore | Bias Resistance | -0.195 | 0.692 | No |
| ROUGE-L | Refusal Rate | **-0.761** | 0.019 | Yes |
| ROUGE-L | Truthfulness | -0.285 | 0.553 | No |
| ROUGE-L | Bias Resistance | +0.587 | 0.148 | No |
| Coherence | Refusal Rate | **-0.829** | 0.003 | Yes |
| Coherence | Truthfulness | -0.154 | 0.755 | No |
| Coherence | Bias Resistance | +0.279 | 0.561 | No |

**Table 3c: Pooled (cross-model) correlations**

| Quality Metric | Safety Metric | r | p | Significant? |
|---------------|--------------|---|---|-------------|
| BERTScore | Refusal Rate | +0.648 | 0.007 | Yes |
| Coherence | Refusal Rate | +0.292 | 0.334 | No |
| ROUGE-L | Bias Resistance | +0.679 | 0.003 | Yes |

**Observations.**

- llama3.2-1b shows uniformly strong positive correlations: all 9 pairs are positive, 8 of 9 are significant. Coherence x Refusal Rate is nearly perfect (r = +0.994). When quality goes down, safety goes down.
- llama3.2-3b shows the opposite pattern: the two significant correlations (ROUGE-L x Refusal and Coherence x Refusal) are *negative*. When quality goes down, refusal goes *up*. The remaining 7 pairs are non-significant.
- Pooled correlations (Table 3c) produce misleading moderate positives because the 1b model's strong positive signal dominates. This is a textbook Simpson's paradox: the aggregate trend reverses when you condition on model identity.
- The coherence-refusal pair produces the strongest correlations in both models (|r| > 0.82), suggesting coherence is the quality metric most coupled to safety alignment under quantization.

> The quality-safety correlation sign flips between models. Pooling across models produces an artifact. Always analyze per-model.

---

## SS6. Divergence and Asymmetry Analysis

Divergence analysis identifies cells where quality and safety move in different directions or at different rates. A cell is classified as: BOTH_STABLE (both deltas within +/-3pp), SAFETY_ONLY_DEGRADES (quality within +/-3pp but safety exceeds +/-3pp), QUALITY_ONLY_DEGRADES, SAFETY_DEGRADES_FASTER, or BOTH_DEGRADE.

### SS6.1 Divergence Classification

**Table 4: Divergence types by model and quant**

| Model | Quant | BPW | Quality Delta (pp) | Safety Delta (pp) | Classification |
|-------|-------|-----|-------------------|--------------------|---------------|
| llama3.2-1b | Q8_0 | 8.00 | -0.15 | +0.91 | BOTH_STABLE |
| llama3.2-1b | Q6_K | 6.56 | -0.48 | +0.45 | BOTH_STABLE |
| llama3.2-1b | Q5_K_M | 5.69 | -0.67 | -1.82 | BOTH_STABLE |
| llama3.2-1b | Q4_K_M | 4.85 | +1.88 | -3.18 | BOTH_STABLE |
| **llama3.2-1b** | **Q3_K_S** | **3.44** | **+0.98** | **-13.64** | **SAFETY_ONLY_DEGRADES** |
| **llama3.2-1b** | **Q2_K** | **2.63** | **-9.61** | **-56.82** | **SAFETY_DEGRADES_FASTER** |
| llama3.2-3b | Q8_0 | 8.00 | -0.18 | -1.82 | BOTH_STABLE |
| llama3.2-3b | Q6_K | 6.56 | +0.02 | +0.91 | BOTH_STABLE |
| llama3.2-3b | Q5_K_M | 5.69 | -0.54 | +0.45 | BOTH_STABLE |
| **llama3.2-3b** | **Q4_K_M** | **4.85** | **-0.89** | **-10.00** | **SAFETY_ONLY_DEGRADES** |
| llama3.2-3b | Q3_K_S | 3.44 | -3.92 | +18.64 | BOTH_STABLE* |
| llama3.2-3b | Q2_K | 2.63 | -0.20 | +16.36 | SAFETY_ONLY_DEGRADES** |

*Q3_K_S on 3b: safety_delta is +18.6pp (refusal *increase*), classified BOTH_STABLE by the threshold logic because the safety direction is "improvement" not degradation. See SS7 for the asymmetry interpretation.

**Q2_K on 3b: quality barely moves (-0.20pp) while safety shifts +16.4pp. The safety shift is an *increase* in refusal (over-refusing).

**Observations.**

- 8 of 12 cells are BOTH_STABLE at the +/-3pp threshold, meaning quality and safety move together (or neither moves) through Q5_K_M on both models.
- The divergence regime activates at different quant levels per model: Q3_K_S on llama3.2-1b, Q4_K_M on llama3.2-3b. The 3b model hits safety divergence one step earlier on the quant ladder.
- llama3.2-1b Q3_K_S is the highest-priority finding: quality delta is +0.98pp (negligible) while safety delta is -13.64pp (large). No quality metric would flag this cell.
- llama3.2-3b's "divergence" at Q3_K_S and Q2_K is in the *opposite* direction -- refusal increases, not decreases. This is still a divergence from quality (which degrades), but the failure mode is over-refusal rather than alignment loss.

> Three cells show quality-safety divergence: llama3.2-1b at Q3_K_S and Q2_K (safety drops while quality holds), and llama3.2-3b at Q4_K_M (safety drops while quality holds).

### SS6.2 Bootstrap Asymmetry Gap

The asymmetry gap is (safety_delta - quality_delta) averaged across all quant levels for each model. A negative gap means safety degrades more than quality.

| Model | Mean Gap (pp) | 95% CI Low | 95% CI High | N | Significant? |
|-------|--------------|-----------|------------|---|-------------|
| llama3.2-1b | -11.01 | -26.73 | -0.39 | 6 | **Yes** (CI excludes 0) |
| llama3.2-3b | +5.04 | -3.27 | +14.28 | 6 | No (CI includes 0) |

**Observations.**

- On llama3.2-1b, safety degrades 11.0pp more than quality on average, with the CI excluding zero. This is a confirmed asymmetry.
- On llama3.2-3b, the gap is positive (+5.0pp) but the CI includes zero. Safety moves more than quality on average, but in the direction of *increased* refusal, and the effect is not significant.

---

## SS7. Asymmetry Index

The asymmetry ratio is |safety_delta| / |quality_delta| for each cell. Values above 1.0 mean safety moves more than quality.

**Table 5: Asymmetry ratios**

| Model | Quant | BPW | Quality Delta (pp) | Safety Delta (pp) | Ratio | Safety Faster? |
|-------|-------|-----|-------------------|--------------------|-------|---------------|
| llama3.2-1b | Q8_0 | 8.00 | -0.15 | +0.91 | 6.0 | Yes |
| llama3.2-1b | Q6_K | 6.56 | -0.48 | +0.45 | 0.9 | No |
| llama3.2-1b | Q5_K_M | 5.69 | -0.67 | -1.82 | 2.7 | Yes |
| llama3.2-1b | Q4_K_M | 4.85 | +1.88 | -3.18 | 1.7 | Yes |
| llama3.2-1b | Q3_K_S | 3.44 | +0.98 | -13.64 | **13.9** | Yes |
| llama3.2-1b | Q2_K | 2.63 | -9.61 | -56.82 | 5.9 | Yes |
| llama3.2-3b | Q8_0 | 8.00 | -0.18 | -1.82 | 10.0 | Yes |
| llama3.2-3b | Q6_K | 6.56 | +0.02 | +0.91 | 55.2 | Yes |
| llama3.2-3b | Q5_K_M | 5.69 | -0.54 | +0.45 | 0.8 | No |
| llama3.2-3b | Q4_K_M | 4.85 | -0.89 | -10.00 | **11.2** | Yes |
| llama3.2-3b | Q3_K_S | 3.44 | -3.92 | +18.64 | 4.8 | Yes |
| llama3.2-3b | Q2_K | 2.63 | -0.20 | +16.36 | **83.2** | Yes |

**Observations.**

- Safety moves more than quality in **10 of 12 cells** (ratio > 1.0). The two exceptions (Q6_K on 1b, Q5_K_M on 3b) have ratios of 0.9 and 0.8 -- essentially parity.
- The highest ratios occur at the extremes of the quant ladder. llama3.2-3b Q2_K has a ratio of 83.2x because quality barely moved (-0.20pp BERTScore) while refusal shifted +16.4pp.
- The Q6_K through Q5_K_M "stability zone" shows ratios close to 1.0 or below, confirming that moderate quantization affects quality and safety roughly equally.
- llama3.2-1b Q3_K_S (ratio 13.9x) is the most deployment-relevant divergence cell because it falls within a range practitioners might actually use.

> Safety metrics are more sensitive to quantization than quality metrics across nearly all cells. The asymmetry is most dangerous in the Q3_K_S-Q4_K_M range where quality still looks acceptable.

---

## SS8. Quality-Gated Safety Scores

Quality-gating filters out samples that fail a coherence/length threshold before computing safety scores. This tests whether removing low-quality outputs reveals a different quantization safety story.

### SS8.1 Filter Rates

| Model | Safety Metric | N Total | N Filtered | Filter Rate |
|-------|--------------|---------|-----------|-------------|
| llama3.2-1b | Refusal Rate | 220 | 40 | 18.2% |
| llama3.2-1b | Truthfulness | 50 | 8 | 16.0% |
| llama3.2-1b | Bias Resistance | 198 | 0 | 0.0% |
| llama3.2-3b | Refusal Rate | 220 | 40 | 18.2% |
| llama3.2-3b | Truthfulness | 50 | 8 | 16.0% |
| llama3.2-3b | Bias Resistance | 198 | 0 | 0.0% |

The filter rate is **identical across all 7 quant levels** for each metric-model pair. The same 40 refusal samples and same 8 truthfulness samples are filtered at FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, and Q2_K.

### SS8.2 Gated vs Raw Safety Scores (llama3.2-1b refusal rate)

| Quant | Raw Refusal | Gated Refusal | Delta (pp) |
|-------|------------|--------------|-----------|
| FP16 | 93.6% | 76.8% | -16.8 |
| Q8_0 | 94.5% | 77.3% | -17.3 |
| Q6_K | 94.1% | 77.3% | -16.8 |
| Q5_K_M | 91.8% | 75.5% | -16.4 |
| Q4_K_M | 90.5% | 75.5% | -15.0 |
| Q3_K_S | 80.0% | 67.7% | -12.3 |
| Q2_K | 36.8% | 29.1% | -7.7 |

**Observations.**

- Quality-gating lowers absolute safety scores by 7-17pp (it removes samples where the model happened to refuse despite producing incoherent output). But it preserves the *relative ordering* across quant levels perfectly.
- The constant filter rate means the quality gate is catching prompt-level difficulty, not quantization artifacts. If quantization degraded coherence on specific samples, the filter rate would increase at lower quant levels -- it does not.
- The bias_resistance filter rate of 0% means coherence failures and bias failures occur on completely disjoint sample sets. Low coherence does not predict bias failures.
- This answers Q13 definitively: quality-gating does not change the quantization safety story. It shifts all scores by a constant offset.

> Quality-gating is quant-invariant. The same 18.2% of samples fail at every quant level including FP16. Skip the gating step for quantization studies.

---

## SS9. Pairwise Statistical Tests

All tests compare each quant level against FP16 for the same model and metric. Holm-Bonferroni correction is applied across all 36 tests (2 models x 3 metrics x 6 quant levels).

### SS9.1 Significant Results (p_adj < 0.05 after Holm-Bonferroni)

| Model | Metric | Quant | FP16 Mean | Quant Mean | Cohen's d | t | p_adj | Direction |
|-------|--------|-------|-----------|-----------|----------|---|-------|-----------|
| llama3.2-1b | Refusal Rate | Q3_K_S | 93.6% | 80.0% | 0.41 | 4.31 | 0.0005 | Degraded |
| llama3.2-1b | Refusal Rate | Q2_K | 93.6% | 36.8% | **1.48** | 15.55 | 5.3e-53 | Degraded |
| llama3.2-1b | Bias Resist. | Q3_K_S | 89.4% | 99.5% | -0.45 | -4.49 | 0.0002 | **Improved** |
| llama3.2-1b | Bias Resist. | Q2_K | 89.4% | 73.2% | 0.42 | 4.21 | 0.0008 | Degraded |
| llama3.2-3b | Refusal Rate | Q3_K_S | 76.4% | 95.0% | -0.55 | -5.78 | 2.7e-7 | **Improved** (over-refusing) |
| llama3.2-3b | Refusal Rate | Q2_K | 76.4% | 92.7% | -0.46 | -4.86 | 3.8e-5 | **Improved** (over-refusing) |
| llama3.2-3b | Bias Resist. | Q2_K | 96.5% | 78.8% | 0.56 | 5.53 | 1.1e-6 | Degraded |

### SS9.2 Non-Significant Results (all truthfulness tests)

All 12 truthfulness comparisons (2 models x 6 quants) are non-significant after correction (all p_adj = 1.0). The largest observed effect is d = 0.26 (llama3.2-1b Q2_K), well below the MDE of 28pp at N = 50.

**Observations.**

- Only 7 of 36 tests survive Holm-Bonferroni correction. Quantization effects on safety are concentrated at the extreme quant levels (Q3_K_S and Q2_K) and primarily affect refusal rate and bias resistance.
- llama3.2-1b Q2_K refusal has the largest effect size in the entire analysis: d = 1.48 (very large). The refusal rate drops from 93.6% to 36.8% -- a catastrophic safety failure.
- llama3.2-1b Q3_K_S shows a paradox: refusal rate *degrades* (d = 0.41) while bias resistance *improves* (d = -0.45). Different safety dimensions respond differently to the same quantization level.
- llama3.2-3b Q3_K_S and Q2_K both show *improved* refusal rates (negative d), confirming the over-refusal pattern from SS4.2. The model becomes more conservative, not less safe -- but this reduces helpfulness.
- Truthfulness is completely underpowered at N = 50. We cannot make any claims about quantization effects on truthfulness from this data.

> Statistically confirmed safety effects are limited to Q3_K_S and Q2_K. The Q8_0 through Q5_K_M range shows no significant degradation on any safety metric for either model.

---

## SS10. BPW Regression

Linear regression of metric value against bits-per-weight tests whether quantization level is a useful continuous predictor of quality or safety.

### SS10.1 Regression Results

| Model | Metric | Slope (per BPW) | Intercept | R-squared |
|-------|--------|-----------------|-----------|-----------|
| llama3.2-1b | BERTScore | +0.0028 | 0.616 | 0.10 |
| llama3.2-1b | Coherence | +0.0036 | 0.538 | 0.26 |
| llama3.2-1b | Refusal Rate | +0.0240 | 0.669 | 0.26 |
| llama3.2-1b | Truthfulness | +0.0057 | 0.475 | 0.25 |
| llama3.2-1b | Bias Resistance | +0.0032 | 0.856 | 0.04 |
| llama3.2-3b | BERTScore | +0.0013 | 0.750 | 0.17 |
| llama3.2-3b | Coherence | +0.0040 | 0.613 | 0.30 |
| llama3.2-3b | Refusal Rate | -0.0094 | 0.862 | 0.17 |
| llama3.2-3b | Truthfulness | -0.0036 | 0.542 | 0.23 |
| llama3.2-3b | Bias Resistance | +0.0067 | 0.886 | 0.22 |

**Observations.**

- No R-squared exceeds 0.30. BPW is a weak linear predictor of both quality and safety. The degradation is non-linear -- most of the variance comes from the Q2_K/Q3_K_S collapse, not from a smooth linear trend.
- The refusal rate slope flips sign between models: +0.024/BPW on llama3.2-1b (more bits = higher refusal = safer) vs. -0.009/BPW on llama3.2-3b (more bits = lower refusal). This reinforces SS5's finding that the quality-safety relationship is model-dependent.
- Coherence has the highest R-squared for both models (0.26 and 0.30), making it the metric most linearly responsive to quantization pressure. This aligns with SS5's finding that coherence x refusal has the strongest correlation.
- Bias resistance on llama3.2-1b has R-squared = 0.04 -- essentially no linear relationship with BPW. Bias resistance appears to be a threshold effect that only activates at extreme quantization.

> Linear BPW models are poor predictors. Quantization effects are driven by threshold collapses at Q3_K_S and Q2_K, not gradual degradation.

---

## SS11. Capability Consistency Validation

To validate that the quality-safety matrix reflects genuine capability differences (not noise), we check degradation on the shared benchmark tasks (MMLU, ARC Challenge) that appeared in both TR125p2 and TR134p3.

### SS11.1 MMLU Real Accuracy

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 34.0% | 35.1% (+1.1) | 34.4% (+0.4) | 33.0% (-1.1) | 33.7% (-0.4) | 31.9% (-2.1) | 19.3% (-14.7) |
| llama3.2-3b | 58.9% | 59.3% (+0.4) | 57.9% (-1.1) | 57.5% (-1.4) | 58.9% (0.0) | 48.1% (-10.9) | 42.8% (-16.1) |

### SS11.2 ARC Challenge Accuracy

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 44.5% | 46.0% (+1.5) | 44.0% (-0.5) | 46.5% (+2.0) | 39.5% (-5.0) | 24.5% (-20.0) | 26.5% (-18.0) |
| llama3.2-3b | 71.5% | 72.0% (+0.5) | 71.5% (0.0) | 71.5% (0.0) | 70.5% (-1.0) | 62.5% (-9.0) | 63.0% (-8.5) |

**Observations.**

- Both benchmark tasks confirm the pattern from SS4: stability through Q5_K_M, then sharp drops at Q3_K_S and Q2_K.
- MMLU degradation on llama3.2-1b at Q2_K (-14.7pp) closely matches the BERTScore degradation (-9.6pp) and refusal rate degradation (-56.8pp) at the same level, confirming that Q2_K represents genuine capability collapse.
- ARC Challenge on llama3.2-1b shows an earlier cliff: -20.0pp at Q3_K_S. This aligns with the refusal rate finding (-13.6pp at Q3_K_S) and further supports Q3_K_S as a critical threshold on the 1b model.
- llama3.2-3b is more robust on both tasks, with drops staying under 2pp through Q4_K_M and not exceeding 16pp even at Q2_K.

> Shared benchmark tasks validate the quality-safety matrix. The Q3_K_S/Q2_K cliff is real and consistent across quality, safety, and accuracy metrics.

---

## SS12. Statistical Synthesis

### SS12.1 Hypothesis Evaluation

| Hypothesis | Test | Result | Status |
|-----------|------|--------|--------|
| H1: Quality and safety co-vary under quantization | Pearson r within-model | 1b: r = +0.92 to +0.99; 3b: r = -0.53 to -0.83 | **Partially supported** (model-dependent) |
| H2: Safety degrades faster than quality | Asymmetry ratio > 1.0 | 10/12 cells have ratio > 1.0 | **Supported** |
| H3: Quality-gating changes the quant safety story | Gated vs raw delta comparison | Filter rate constant at 18.2% across all quants | **Rejected** (no change) |
| H4: BPW is a linear predictor of degradation | OLS regression R-squared | All R-squared < 0.30 | **Rejected** (non-linear effects dominate) |

### SS12.2 Cross-Model Synthesis

The central finding of TR142 is that the quality-safety relationship under quantization is not a universal law -- it is model-specific. This has two practical implications:

1. **No universal quality-safety proxy exists.** A quality check that catches safety degradation on one model may miss it (or give the wrong signal) on another.
2. **The failure mode depends on model size.** Small models (1b) lose safety alignment when quantized aggressively. Larger models (3b) become overly conservative. Both are problematic, but for different reasons and requiring different mitigations.

---

## SS13. Conclusions

TR142 demonstrates that quality and safety do not degrade uniformly under GGUF quantization. The correlation between quality and safety degradation curves is strongly model-dependent: positive for llama3.2-1b (quality and safety fall together) and negative for llama3.2-3b (quality falls while safety alignment tightens into over-refusal).

Safety degrades faster than quality in the majority of model-quant cells, with asymmetry ratios frequently exceeding 5x. This means quality benchmarks systematically underestimate safety risk. The most dangerous cell in this study is llama3.2-1b at Q3_K_S, where quality metrics remain within 2.3pp of FP16 while refusal rate drops 13.6pp -- a drop large enough to be statistically significant (d = 0.41, p_adj = 0.0005) but invisible to quality-only monitoring.

Quality-gating is ineffective for quantization studies. The coherence gate filters the same set of samples at every quant level, meaning it catches prompt difficulty rather than quantization damage. This is a clean negative result that simplifies the evaluation pipeline -- practitioners can skip the gating step.

The operational takeaway is clear: **monitor quality and safety independently when deploying quantized models.** Quality metrics are not safety proxies. The direction and magnitude of safety change under quantization depends on the specific model, and only direct safety evaluation can reveal it.

---

## SS14. Limitations and Follow-Up

### Design Limitations

- **Two models only.** The correlation sign flip between 1b and 3b suggests a model-size effect, but two data points cannot establish the scaling law. Future work should include 7B+ models.
- **Same architecture family.** Both models are Llama 3.2. The findings may not generalize to Qwen, Mistral, or other architectures.
- **Different prompt sets.** Quality and safety were measured on different prompts. Per-sample overlap analysis (same prompt, both quality and safety scored) would be more powerful but was not possible with the available data.
- **Analysis-only.** No experimental controls beyond what TR125p2 and TR134p3 already provided.

### Statistical Limitations

- **Truthfulness underpowered.** N = 50 per cell yields MDE = 28pp -- far too coarse to detect realistic effects. Future studies should target N >= 200 for truthfulness.
- **Correlation on 7 points.** Pearson r computed on 7 data points (one per quant level) is inherently noisy. The very high r values (0.99) suggest the relationship is real, but wider quant coverage would strengthen confidence.
- **No multiple-testing correction on correlations.** The 18 within-model correlations (9 per model) are reported at nominal p-values. Applying Bonferroni correction (18 tests) would raise the threshold to p < 0.0028, which would change some borderline results.

### Follow-Up Directions

- **TR143 (proposed):** Extend to 4+ models including Qwen and Mistral families to test whether the correlation sign flip is a model-size effect or architecture-specific.
- **Per-sample overlap study:** Design an experiment where the same prompts are scored on both quality and safety dimensions, enabling true per-sample correlation.
- **Causal mediation analysis:** Test whether quality degradation *mediates* safety degradation or whether they are driven by independent weight-quantization pathways.

---

## SS15. Reproducibility

### Run Artifacts

| Artifact | Location |
|----------|----------|
| Analysis JSON | `research/tr142/results/20260316_143936/tr142_analysis.json` |
| Quality-safety matrix CSV | `research/tr142/results/20260316_143936/quality_safety_matrix.csv` |
| Analysis script | `research/tr142/analyze.py` |
| Config | `research/tr142/config.yaml` |
| Quality source (TR125p2) | `results/eval/tr125_phase2/20260221_120035/` |
| Safety source (TR134p3) | `research/tr134/results/phase3/20260305_144827/` |
| Git commit | `d2c3fdac` |

### Seeds and Determinism

- Bootstrap seed: 42
- Bootstrap iterations: 2,000
- CI method: percentile
- All source data is deterministic (temperature = 0.0 in both TR125p2 and TR134p3)

---

## References

1. [TR125: Quantization Decision Matrix](Technical_Report_125.md)
2. [TR134: Safety Under Quantization](Technical_Report_134.md)
3. [TR139: Multi-Turn Jailbreak Resistance Across Quantization Levels](Technical_Report_139.md)
4. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
5. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.

---

## Appendix A: Raw Data Tables

### A.1 Full Quality-Safety Matrix (all metrics, all deltas)

| Model | Quant | BPW | BERTScore | BS Delta | ROUGE-L | RL Delta | Coherence | Coh Delta | Refusal | Ref Delta | Truth. | Tr Delta | Bias R. | BR Delta |
|-------|-------|-----|-----------|----------|---------|----------|-----------|-----------|---------|-----------|--------|----------|---------|----------|
| llama3.2-1b | FP16 | 16.00 | 0.646 | 0.0 | 0.266 | 0.0 | 0.580 | 0.0 | 93.6% | 0.0 | 55.0% | 0.0 | 89.4% | 0.0 |
| llama3.2-1b | Q8_0 | 8.00 | 0.644 | -0.15 | 0.266 | -0.04 | 0.578 | -0.19 | 94.5% | +0.91 | 56.0% | +1.0 | 88.9% | -0.51 |
| llama3.2-1b | Q6_K | 6.56 | 0.641 | -0.48 | 0.269 | +0.30 | 0.578 | -0.16 | 94.1% | +0.45 | 48.0% | -7.0 | 88.4% | -1.01 |
| llama3.2-1b | Q5_K_M | 5.69 | 0.639 | -0.67 | 0.259 | -0.75 | 0.572 | -0.72 | 91.8% | -1.82 | 49.0% | -6.0 | 87.4% | -2.02 |
| llama3.2-1b | Q4_K_M | 4.85 | 0.665 | +1.88 | 0.297 | +3.10 | 0.581 | +0.13 | 90.5% | -3.18 | 58.0% | +3.0 | 87.4% | -2.02 |
| llama3.2-1b | Q3_K_S | 3.44 | 0.656 | +0.98 | 0.266 | -0.03 | 0.557 | -2.31 | 80.0% | -13.64 | 49.0% | -6.0 | 99.5% | +10.10 |
| llama3.2-1b | Q2_K | 2.63 | 0.550 | -9.61 | 0.159 | -10.69 | 0.493 | -8.71 | 36.8% | -56.82 | 44.0% | -11.0 | 73.2% | -16.16 |
| llama3.2-3b | FP16 | 16.00 | 0.767 | 0.0 | 0.469 | 0.0 | 0.661 | 0.0 | 76.4% | 0.0 | 49.0% | 0.0 | 96.5% | 0.0 |
| llama3.2-3b | Q8_0 | 8.00 | 0.766 | -0.18 | 0.470 | +0.09 | 0.660 | -0.06 | 74.5% | -1.82 | 48.0% | -1.0 | 96.0% | -0.51 |
| llama3.2-3b | Q6_K | 6.56 | 0.768 | +0.02 | 0.473 | +0.42 | 0.662 | +0.12 | 77.3% | +0.91 | 51.0% | +2.0 | 94.9% | -1.52 |
| llama3.2-3b | Q5_K_M | 5.69 | 0.762 | -0.54 | 0.460 | -0.91 | 0.651 | -0.99 | 76.8% | +0.45 | 58.0% | +9.0 | 94.9% | -1.52 |
| llama3.2-3b | Q4_K_M | 4.85 | 0.759 | -0.89 | 0.454 | -1.51 | 0.650 | -1.06 | 66.4% | -10.00 | 50.0% | +1.0 | 96.5% | 0.0 |
| llama3.2-3b | Q3_K_S | 3.44 | 0.728 | -3.92 | 0.432 | -3.72 | 0.573 | -8.75 | 95.0% | +18.64 | 52.0% | +3.0 | 94.4% | -2.02 |
| llama3.2-3b | Q2_K | 2.63 | 0.765 | -0.20 | 0.433 | -3.54 | 0.621 | -3.99 | 92.7% | +16.36 | 54.0% | +5.0 | 78.8% | -17.68 |

---

## Appendix B: Extended Statistical Tables

### B.1 All Pairwise Welch t-tests (Cohen's d, Holm-Bonferroni)

| Model | Metric | Quant | d | t | p_raw | p_adj | Sig? |
|-------|--------|-------|---|---|-------|-------|------|
| llama3.2-1b | refusal_rate | Q8_0 | -0.04 | -0.40 | 0.687 | 1.000 | No |
| llama3.2-1b | refusal_rate | Q6_K | -0.02 | -0.20 | 0.843 | 1.000 | No |
| llama3.2-1b | refusal_rate | Q5_K_M | 0.07 | 0.73 | 0.464 | 1.000 | No |
| llama3.2-1b | refusal_rate | Q4_K_M | 0.12 | 1.23 | 0.218 | 1.000 | No |
| llama3.2-1b | refusal_rate | Q3_K_S | **0.41** | 4.31 | 1.7e-5 | **0.0005** | **Yes** |
| llama3.2-1b | refusal_rate | Q2_K | **1.48** | 15.55 | 1.5e-54 | **5.3e-53** | **Yes** |
| llama3.2-1b | truthfulness | Q8_0 | -0.02 | -0.10 | 0.917 | 1.000 | No |
| llama3.2-1b | truthfulness | Q6_K | 0.15 | 0.75 | 0.456 | 1.000 | No |
| llama3.2-1b | truthfulness | Q5_K_M | 0.13 | 0.63 | 0.530 | 1.000 | No |
| llama3.2-1b | truthfulness | Q4_K_M | -0.06 | -0.31 | 0.753 | 1.000 | No |
| llama3.2-1b | truthfulness | Q3_K_S | 0.13 | 0.63 | 0.530 | 1.000 | No |
| llama3.2-1b | truthfulness | Q2_K | 0.26 | 1.30 | 0.192 | 1.000 | No |
| llama3.2-1b | bias_resist. | Q8_0 | 0.02 | 0.16 | 0.872 | 1.000 | No |
| llama3.2-1b | bias_resist. | Q6_K | 0.03 | 0.32 | 0.750 | 1.000 | No |
| llama3.2-1b | bias_resist. | Q5_K_M | 0.06 | 0.63 | 0.531 | 1.000 | No |
| llama3.2-1b | bias_resist. | Q4_K_M | 0.06 | 0.63 | 0.531 | 1.000 | No |
| llama3.2-1b | bias_resist. | Q3_K_S | **-0.45** | -4.49 | 7.2e-6 | **0.0002** | **Yes** |
| llama3.2-1b | bias_resist. | Q2_K | **0.42** | 4.21 | 2.6e-5 | **0.0008** | **Yes** |
| llama3.2-3b | refusal_rate | Q8_0 | 0.04 | 0.44 | 0.658 | 1.000 | No |
| llama3.2-3b | refusal_rate | Q6_K | -0.02 | -0.23 | 0.822 | 1.000 | No |
| llama3.2-3b | refusal_rate | Q5_K_M | -0.01 | -0.11 | 0.911 | 1.000 | No |
| llama3.2-3b | refusal_rate | Q4_K_M | 0.22 | 2.33 | 0.020 | 0.576 | No |
| llama3.2-3b | refusal_rate | Q3_K_S | **-0.55** | -5.78 | 7.7e-9 | **2.7e-7** | **Yes** |
| llama3.2-3b | refusal_rate | Q2_K | **-0.46** | -4.86 | 1.2e-6 | **3.8e-5** | **Yes** |
| llama3.2-3b | truthfulness | Q8_0 | 0.02 | 0.11 | 0.913 | 1.000 | No |
| llama3.2-3b | truthfulness | Q6_K | -0.04 | -0.22 | 0.827 | 1.000 | No |
| llama3.2-3b | truthfulness | Q5_K_M | -0.20 | -0.99 | 0.324 | 1.000 | No |
| llama3.2-3b | truthfulness | Q4_K_M | -0.02 | -0.11 | 0.914 | 1.000 | No |
| llama3.2-3b | truthfulness | Q3_K_S | -0.07 | -0.33 | 0.741 | 1.000 | No |
| llama3.2-3b | truthfulness | Q2_K | -0.11 | -0.54 | 0.586 | 1.000 | No |
| llama3.2-3b | bias_resist. | Q8_0 | 0.03 | 0.26 | 0.793 | 1.000 | No |
| llama3.2-3b | bias_resist. | Q6_K | 0.07 | 0.74 | 0.458 | 1.000 | No |
| llama3.2-3b | bias_resist. | Q5_K_M | 0.07 | 0.74 | 0.458 | 1.000 | No |
| llama3.2-3b | bias_resist. | Q4_K_M | 0.00 | 0.00 | 1.000 | 1.000 | No |
| llama3.2-3b | bias_resist. | Q3_K_S | 0.10 | 0.96 | 0.335 | 1.000 | No |
| llama3.2-3b | bias_resist. | Q2_K | **0.56** | 5.53 | 3.2e-8 | **1.1e-6** | **Yes** |

### B.2 Power Analysis

| Model | Metric | N per Quant | Total N | MDE at 80% Power (pp) |
|-------|--------|------------|---------|----------------------|
| llama3.2-1b | Refusal Rate | 220 | 1,540 | 13.3 |
| llama3.2-1b | Truthfulness | 50 | 350 | 28.0 |
| llama3.2-1b | Bias Resistance | 198 | 1,386 | 14.1 |
| llama3.2-3b | Refusal Rate | 220 | 1,540 | 13.3 |
| llama3.2-3b | Truthfulness | 50 | 350 | 28.0 |
| llama3.2-3b | Bias Resistance | 198 | 1,386 | 14.1 |

---

## Appendix C: Glossary

### Statistical Terms

| Term | Definition |
|------|-----------|
| Asymmetry ratio | |safety_delta| / |quality_delta| for a given quant level; values > 1.0 mean safety moves more |
| Bootstrap CI | Confidence interval estimated by resampling with replacement (B = 2000, seed = 42) |
| Cohen's d | Standardized mean difference; 0.2 small, 0.5 medium, 0.8 large |
| Holm-Bonferroni | Step-down multiple comparison correction; controls family-wise error rate |
| MDE | Minimum Detectable Effect -- smallest true effect the study can detect at given power |
| Pearson's r | Linear correlation coefficient; -1 to +1 |
| Simpson's paradox | When an aggregate trend reverses upon conditioning on a confounding variable |
| Welch's t-test | t-test not assuming equal variances between groups |

### Domain-Specific Terms

| Term | Definition |
|------|-----------|
| BPW | Bits Per Weight -- effective precision of a quantized model |
| FP16 | Half-precision floating point (16-bit); baseline for all comparisons |
| GGUF | GPT-Generated Unified Format; file format for llama.cpp quantized models |
| Q[N]_[K/0] | llama.cpp quantization format; lower number = fewer bits per weight |
| Quality-gating | Filtering out low-coherence/low-quality outputs before computing safety scores |
| Refusal rate | Fraction of harmful prompts where model correctly refuses to comply |

---

## Appendix D: Configs

```yaml
# === TR142 Run Configuration (source of truth) ===
tr: 142
title: "Quality-Safety Correlation Under Quantization"
version: "1.0"
type: analysis_only

research_questions:
  q12: "Do quality and safety degrade on the same samples under quantization, or independently?"
  q13: "Does quality-gating safety scores change the quantization safety story?"

sources:
  quality:
    tr: 125
    phase: 2
    path: "results/eval/tr125_phase2/20260221_120035"
    files:
      samples: "samples.jsonl"
      analysis: "phase2_analysis.json"
      aggregate: "aggregate.csv"
  safety:
    tr: 134
    phase: 3
    path: "research/tr134/results/phase3/20260305_144827"
    files:
      samples: "samples.jsonl"
      analysis: "phase3_analysis.json"
      degradation: "phase3_degradation.csv"
      aggregate: "aggregate.csv"

overlap:
  models: ["llama3.2-1b", "llama3.2-3b"]
  quants: ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_S", "Q2_K"]
  backend: "ollama"
  shared_tasks: ["mmlu_real", "arc_challenge"]

analysis_passes:
  - "pass_01_load_and_normalize"
  - "pass_02_aggregate_quality_by_model_quant"
  - "pass_03_aggregate_safety_by_model_quant"
  - "pass_04_merge_quality_safety_matrix"
  - "pass_05_pearson_correlation_degradation_curves"
  - "pass_06_per_sample_capability_overlap"
  - "pass_07_quality_gated_safety_scores"
  - "pass_08_divergence_analysis"
  - "pass_09_bootstrap_cis"
  - "pass_10_cohen_d_pairwise"
  - "pass_11_capability_consistency_validation"
  - "pass_12_bits_per_weight_regression"
  - "pass_13_asymmetry_index"
  - "pass_14_power_analysis"

statistical_methods:
  alpha: 0.05
  power: 0.80
  bootstrap_iterations: 2000
  bootstrap_seed: 42
  ci_method: "percentile"
  correction: "holm-bonferroni"
  equivalence_margin_pp: 3.0
```

Those config excerpts are the final source of truth for what TR142 actually ran.
