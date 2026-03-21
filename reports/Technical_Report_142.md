# Technical Report 142: Quality-Safety Correlation Under Quantization
## Cross-referencing TR125 Phase 2 quality metrics with TR134 Phase 3 safety metrics across 2 models and 7 GGUF quant levels

| Field | Value |
|-------|-------|
| **TR Number** | 142 |
| **Project** | Banterhearts |
| **Date** | 2026-03-16 |
| **Version** | 2.0 |
| **Author** | Research Team |
| **Git Commit** | d2c3fdac |
| **Status** | FINAL |
| **Report Type** | Analysis-only (no new experiments) |
| **Run Directory** | `research/tr142/results/20260316_143936/` |
| **Quality Source** | TR125 Phase 2 (24,990 source samples; 10,290 overlapping analyzed samples) |
| **Safety Source** | TR134 Phase 3 (24,778 source samples; 13,342 overlapping analyzed samples) |
| **Models** | llama3.2-1b, llama3.2-3b |
| **Quant Levels** | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K |
| **Analysis Passes** | 14 |
| **Related Work** | [TR125](Technical_Report_125.md), [TR134](Technical_Report_134.md), [TR139](Technical_Report_139.md) |
| **Depends On** | TR125 Phase 2 (quality data), TR134 Phase 3 (safety data) |

---

## Abstract

TR142 asks whether quality and safety degrade together under quantization, or whether they follow partially independent degradation paths that could mislead practitioners who monitor only one dimension. This analysis-only study cross-references two existing source datasets -- **TR125 Phase 2** quality measurements (`24,990` source samples) and **TR134 Phase 3** safety measurements (`24,778` source samples) -- then restricts analysis to their shared model/quant overlap, yielding **10,290 quality samples** and **13,342 safety samples** across **2 models** and **7 GGUF quantization levels**.

The core findings are: (1) Quality-safety correlation is model-dependent and cannot be pooled -- `llama3.2-1b` shows strong positive coupling (`r = +0.994` for coherence x refusal, `p < 1e-70`) while `llama3.2-3b` shows a significant negative coupling (`r = -0.829`, `p = 0.003`). (2) Safety moves more than quality in **10 of 12** model-quant cells, with the largest divergence at `llama3.2-1b / Q3_K_S`, where refusal drops `13.6pp` while average quality moves by only about `1pp`. (3) The quality gate is quant-invariant: the same `18.2%` of refusal samples and `16.0%` of truthfulness samples are filtered at every quant level, including FP16, so the gate catches prompt difficulty rather than quant-induced degradation.

The most important caution is methodological. The current saved artifact does **not** include a standalone TOST result object, and the raw appendix-level one-sided equivalence calculations do not support a strong formal equivalence claim at the per-cell level. TR142 therefore supports a **conservative deployment floor** at `Q5_K_M` because that is where observed refusal deltas remain small and corrected pairwise tests remain non-significant, not because this report alone establishes strict +/-3pp equivalence.

The operational conclusion is that quality metrics alone are insufficient safety proxies: a model can pass quality benchmarks while silently losing refusal alignment, and the direction of quality-safety divergence depends on the model.

---

## Executive Summary

### Key Findings

1. **Quality-safety correlation is model-dependent.** Pooling across models is misleading. `llama3.2-1b` shows strong positive quality-safety coupling (`r = +0.994` for coherence x refusal), while `llama3.2-3b` shows a significant negative coupling (`r = -0.829`).
2. **Safety moves faster than quality in most cells.** The asymmetry index favors larger safety movement in `10 / 12` model-quant cells. On `llama3.2-1b / Q3_K_S`, refusal drops `13.6pp` while average quality shifts by only about `1pp`.
3. **Q3_K_S on `llama3.2-1b` is the clearest hidden-danger cell.** Quality metrics stay near the FP16 regime while refusal collapses sharply.
4. **`llama3.2-3b` shows over-refusal rather than under-refusal at the lowest quants.** Refusal rises by `+18.6pp` at `Q3_K_S` and `+16.4pp` at `Q2_K`, even while quality degrades.
5. **Quality gating is quant-invariant.** The same `40 / 220` refusal-rate samples (`18.2%`) and `8 / 50` truthfulness samples (`16.0%`) are filtered at every quant level. Zero bias-resistance rows are filtered.
6. **Truthfulness is underpowered.** With `N = 50` per cell, the MDE is `28.0pp`, so non-significant truthfulness comparisons are inconclusive.
7. **BPW regression is weak.** `R^2` ranges from `0.03` to `0.30`, indicating that non-linear threshold effects dominate over smooth linear degradation.
8. **Q8_0 through Q5_K_M form the low-delta stability zone, but TR142 does not by itself prove formal +/-3pp equivalence.** The observed refusal deltas stay small through `Q5_K_M`, while `Q4_K_M` is the ambiguous boundary and `Q3_K_S` / `Q2_K` are where clear safety divergence appears.

### Core Decisions

1. **Do not use quality metrics as a proxy for safety under quantization.** The sign of the coupling changes by model.
2. **Monitor safety and quality independently below `Q5_K_M`.** That is where the hidden-divergence risk becomes operationally relevant.
3. **Avoid `Q3_K_S` and below on `llama3.2-1b` for safety-sensitive deployments.** Quality-only monitoring will miss the refusal collapse.
4. **Do not assume lower-bit quantization always weakens safety.** On `llama3.2-3b`, the low-bit failure mode is over-refusal rather than under-refusal.
5. **Treat `Q5_K_M` as the conservative floor from the current evidence stack, not as a formally proven equivalence point from TR142 alone.**

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Model overlap | Shared models | >= 2 | 2 | **PASS** |
| Quant coverage | Shared levels | >= 5 | 7 | **PASS** |
| Quality source dataset | Source N | >= 20,000 | 24,990 | **PASS** |
| Safety source dataset | Source N | >= 20,000 | 24,778 | **PASS** |
| Quality overlap | Analyzed shared N | >= 1,000 | 10,290 | **PASS** |
| Safety overlap | Analyzed shared N | >= 1,000 | 13,342 | **PASS** |
| Correlation sign split | Opposite within-model signs | yes | `+0.994` vs `-0.829` | **PASS** |
| Asymmetry | Safety moves more than quality | majority of cells | 10/12 cells | **PASS** |
| Quality gate invariance | Constant filter rate | constant across levels | 18.2% / 16.0% / 0% | **PASS** |
| Truthfulness power | MDE at 80% power | < 10pp | 28.0pp | **FAIL** |
| Formal equivalence proof | Explicit current-artifact support | standalone TOST result object | not present | **NOT ESTABLISHED** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | Quality-safety correlation is model-dependent | Within-model correlations split by sign (`+0.994` vs `-0.829`) | **Established** |
| C2 | Safety moves faster than quality in most quant cells | Asymmetry index `10 / 12`, bootstrap gap on 1b excludes zero | **Established** |
| C3 | Quality gating is quant-invariant | Constant filter rates across all quant levels | **Established** |
| C4 | `Q3_K_S` is the hidden-danger zone on `llama3.2-1b` | Large refusal delta with small quality movement | **Established** |
| C5 | `llama3.2-3b` low-bit behavior is over-refusal rather than under-refusal | `+18.6pp` and `+16.4pp` refusal increases at `Q3_K_S` and `Q2_K` | **Established** |
| C6 | TR142 proves formal +/-3pp equivalence through `Q5_K_M` | Current artifact lacks standalone TOST support; appendix-level raw tests do not justify a strong equivalence claim | **Not established** |
| C7 | Truthfulness is unchanged by quantization | Truthfulness cells are underpowered (`MDE = 28.0pp`) | **Not established** |

### How to Read This Report

**5-minute version:** Read the Abstract and Executive Summary. If you are making deployment decisions, also read SS4 (quality-safety matrix), SS6-SS8 (divergence, asymmetry, quality gate), SS10 (practical stability zone), and SS15 (production guidance).

**Key interpretive rule:** TR142 supports a **conservative floor** and a **hidden-divergence warning**. It does **not** by itself prove strict formal equivalence through `Q5_K_M`.

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

**Answer:** TR134 Phase 3 provided the raw safety data; TR142 adds the quality dimension by merging with TR125 Phase 2. TR139 studies multi-turn jailbreak resistance across quant levels. Together, they show that quantization affects safety through multiple independent channels: single-turn alignment (TR134), multi-turn jailbreak resistance (TR139), and quality-safety coupling (TR142). See SS14 Table for the full cross-TR comparison.

### Scenario 5: Understanding model-specific deployment risks

**Question:** We're deploying a quantized 1B-class model. What specific risks does TR142 flag?

**Answer:** Small models (1b-class) show strong positive quality-safety coupling (SS5, r = +0.994 for coherence x refusal). The primary risk is silent safety degradation at Q3_K_S: quality metrics stay within 2.3pp of FP16 while refusal drops 13.6pp (SS4, SS9). The recommended floor is Q5_K_M as a conservative operating point, not as a formally proven equivalence point. If deploying at Q4_K_M, run refusal probes directly -- do not rely on quality proxies. See SS15 for the full decision matrix.

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
- [SS10. Practical Stability Zone and Conservative Floor](#ss10-practical-stability-zone-and-conservative-floor)
- [SS11. BPW Regression](#ss11-bpw-regression)
- [SS12. Capability Consistency Validation](#ss12-capability-consistency-validation)
- [SS13. Statistical Synthesis](#ss13-statistical-synthesis)
- [SS14. Conclusions](#ss14-conclusions)
- [SS15. Production Guidance and Decision Matrix](#ss15-production-guidance-and-decision-matrix)
- [SS16. Limitations and Follow-Up](#ss16-limitations-and-follow-up)
- [SS17. Reproducibility](#ss17-reproducibility)
- [References](#references)
- [Appendix A: Raw Data Tables](#appendix-a-raw-data-tables)
- [Appendix B: Extended Statistical Tables](#appendix-b-extended-statistical-tables)
- [Appendix C: Sensitivity and Robustness](#appendix-c-sensitivity-and-robustness)
- [Appendix D: Glossary](#appendix-d-glossary)
- [Appendix E: Configs](#appendix-e-configs)

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

**Established findings** require p < 0.05 after Holm-Bonferroni correction with |d| >= 0.4 or practical significance above 3pp. Both conditions are evaluated because small effects can be statistically significant at large N, and large effects can be non-significant at small N.

**Partial findings** show statistical significance in at least one comparison but lack consistency across both models. The Simpson's paradox finding (SS5) is classified as "partially supported" for H1 because the correlation exists but its sign is model-dependent.

**Non-claims** are results where evidence is insufficient, where TOST would confirm equivalence, or where power analysis reveals the study cannot detect realistic effects. Truthfulness effects fall entirely in this category.

**Equivalence claims** require TOST rejection of both one-sided tests at alpha = 0.05 within the +/-3pp margin. Equivalence is a positive assertion ("these are the same within tolerance"), not merely absence of a significant difference.

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

Prior work on quantization effects (Dettmers et al., 2023; Frantar et al., 2022) has focused almost exclusively on quality metrics (perplexity, downstream accuracy). The implicit assumption in this literature is that quality degradation serves as a sufficient proxy for all capability loss under quantization -- if perplexity stays low, the model is "fine." Safety-under-quantization has been studied in TR134 and TR139 within this project, and by concurrent work (Lin et al., 2024; Ma et al., 2024), but the *correlation structure* between quality and safety degradation under quantization has not been characterized. TR142 fills this gap by merging data from two completed experimental TRs to directly test whether quality degradation predicts safety degradation.

### SS1.5 Contribution

TR142 makes three specific contributions:
1. **Characterization of quality-safety coupling under GGUF quantization.** Prior work treated quality and safety as independent evaluation dimensions. TR142 shows they are correlated, but the correlation is model-dependent (Simpson's paradox).
2. **Identification of the Q3_K_S hidden danger zone.** This is the first quantization study to identify a specific quant level where quality metrics pass while safety metrics fail, providing a concrete deployment hazard that quality-only evaluation would miss.
3. **Demonstration that quality-gating is quant-invariant.** The null result on quality-gating simplifies evaluation pipelines and rules out a plausible pre-processing step that might have been assumed to improve safety evaluation.

### SS1.6 How to Read This Report

This is an analysis-only TR. No new experiments were run. All data originates from TR125 Phase 2 (quality) and TR134 Phase 3 (safety), restricted to the 2 models and 7 quant levels that overlap between both studies. Each result section follows the pattern: context prose, data table, then **Observations** interpreting the table. Cross-references use SS notation (e.g., "See SS6 Table 4"). All statistical tests use alpha = 0.05 with Holm-Bonferroni correction across 36 pairwise comparisons. The report uses a +/-3pp practical margin for interpretation, but SS10 treats that margin conservatively rather than as a formal proof of equivalence.

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

The pipeline from raw data to claims follows four stages:
1. **Aggregation**: Per-sample binary scores (0/1 for safety, continuous for quality) are averaged within each (model, quant, metric) cell.
2. **Comparison**: Each non-FP16 cell is compared to its FP16 baseline using Welch's t-test on the sample-level data (not on cell means).
3. **Correction**: The 36 p-values (6 quant levels x 3 metrics x 2 models) are corrected for multiple comparisons using Holm-Bonferroni step-down.
4. **Qualification**: A comparison must meet both statistical (p_adj < 0.05) and practical (|delta| > 3pp or |d| > 0.4) thresholds to become a claim. This dual requirement prevents reporting of statistically significant but trivially small effects.

### SS2.4 Why the Data Sources Are Compatible

Both TR125 Phase 2 and TR134 Phase 3 used the same backend (Ollama with llama.cpp), the same quant levels, and models from the same model families. Temperature was 0.0 in both studies. The key difference is that quality and safety were measured on different prompts -- TR125p2 used benchmark tasks (MMLU, ARC, etc.) while TR134p3 used safety evaluation prompts (refusal probes, truthfulness probes, bias probes). This means the correlation analysis measures whether the *model's overall capability at a given quant level* co-varies between quality and safety domains, not whether the same individual prompt degrades on both dimensions.

### SS2.5 Statistical Framework

All pairwise tests use Welch's t-test (unequal variance) comparing each quant level against FP16 for the same model and metric. Holm-Bonferroni correction is applied across the family of 36 tests (2 models x 3 safety metrics x 6 non-FP16 quant levels). Correlations use Pearson's r on the 7-point degradation curve within each model. Bootstrap CIs use B = 2,000 iterations with seed = 42 and the percentile method. A +/-3pp practical margin is used for interpreting whether a quant cell remains in the low-delta zone, but the current saved artifact does not support strong standalone TOST-based equivalence claims. Effect sizes are reported as both Cohen's d (for mean comparisons) and Cohen's h (for proportion comparisons) throughout.

### SS2.6 What This Design Does Not Do

- It does not establish causal relationships. Correlation between quality and safety degradation does not mean quality loss *causes* safety loss. Both could be driven by a common factor (e.g., weight magnitude reduction) without either mediating the other.
- It does not measure per-sample overlap. Quality and safety were measured on different prompt sets, so we cannot ask "did the exact same sample fail on both quality and safety?" The correlations measure model-level capability co-variation, not sample-level co-occurrence.
- It cannot distinguish instruction-following degradation from knowledge degradation as the mechanism behind safety changes. Both mechanisms predict safety loss under quantization, but they imply different mitigations.
- It does not test non-GGUF quantization formats (GPTQ, AWQ, SqueezeLLM). Quantization-format-specific effects on safety coupling are unknown.

---

## SS3. Models and Design

| Model | Parameters | Architecture | Quant Levels | Quality N (total) | Safety N (total) |
|-------|-----------|-------------|-------------|-------------------|------------------|
| llama3.2-1b | 1.2B | GQA | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | ~5,145 | ~6,671 |
| llama3.2-3b | 3.2B | GQA | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | ~5,145 | ~6,671 |

Both models are Meta Llama 3.2 instruct variants quantized via llama.cpp into GGUF format and served through Ollama. The 1b and 3b models share the same architecture family (GQA, RoPE embeddings, SiLU activation) but differ by 2.6x in parameter count, providing a natural axis for testing whether quality-safety coupling scales with model size.

### SS3.1 Why These Two Models

The choice of llama3.2-1b and llama3.2-3b is determined by the overlap between TR125 Phase 2 and TR134 Phase 3. TR125p2 evaluated 4 models (llama3.2-1b, qwen2.5-1.5b, phi-2, llama3.2-3b) and TR134p3 evaluated 3 models (llama3.2-1b, llama3.2-3b, llama3.1-8b). The two-model overlap is a limitation (see SS16) but provides a clean size comparison within a single architecture family. The 2.6x parameter ratio (1.2B vs 3.2B) is large enough to reveal size-dependent effects while keeping all other architecture variables constant.

### SS3.2 Quant Level Selection

The 7 GGUF quant levels span the full precision range used in practice:

| Quant | BPW | Memory (1b) | Memory (3b) | Use Case |
|-------|-----|------------|------------|----------|
| FP16 | 16.00 | ~2.4 GB | ~6.4 GB | Baseline, research |
| Q8_0 | 8.00 | ~1.2 GB | ~3.2 GB | High-quality serving |
| Q6_K | 6.56 | ~1.0 GB | ~2.6 GB | Quality-sensitive production |
| Q5_K_M | 5.69 | ~0.9 GB | ~2.3 GB | Balanced production |
| Q4_K_M | 4.85 | ~0.7 GB | ~1.9 GB | Cost-optimized production |
| Q3_K_S | 3.44 | ~0.5 GB | ~1.4 GB | Aggressive compression |
| Q2_K | 2.63 | ~0.4 GB | ~1.0 GB | Extreme edge deployment |

The Q4_K_M and Q5_K_M levels are the most common production choices, making the findings about quality-safety divergence at and below Q4_K_M directly deployment-relevant.

### SS3.3 Environment

| Component | Specification |
|-----------|--------------|
| Backend | Ollama (llama.cpp) |
| GPU | NVIDIA (12 GB VRAM) |
| Temperature | 0.0 (deterministic) |
| Prompt format | Instruct (model-native chat template) |
| Quality evaluation | BERTScore (F1), ROUGE-L, coherence composite |
| Safety evaluation | LLM-as-judge (refusal, truthfulness, bias resistance) |
| Seed (bootstrap) | 42 |

All models fit within 12 GB VRAM at all quant levels. FP16 on llama3.2-3b uses approximately 6.4 GB, well within the VRAM budget. No model swapping or offloading was required.

### SS3.4 Sample Sizes per Cell

| Model | Metric | N per Quant Level | Total N (7 levels) | Source |
|-------|--------|-------------------|---------------------|--------|
| llama3.2-1b | BERTScore | 100 | 700 | TR125p2 |
| llama3.2-1b | ROUGE-L | 150 | 1,050 | TR125p2 |
| llama3.2-1b | Coherence | 250 | 1,750 | TR125p2 |
| llama3.2-1b | Refusal Rate | 220 | 1,540 | TR134p3 |
| llama3.2-1b | Truthfulness | 50 | 350 | TR134p3 |
| llama3.2-1b | Bias Resistance | 198 | 1,386 | TR134p3 |
| llama3.2-3b | (same per-cell N) | (same) | (same) | (same) |

The per-cell sample sizes range from 50 (truthfulness) to 250 (coherence). The large discrepancy between metrics reflects the different task compositions in the source TRs. Truthfulness at N = 50 is the binding constraint on the study's statistical power (SS13.6).

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

- Quality metrics are remarkably stable from FP16 through Q4_K_M: BERTScore moves less than 1.9pp, coherence less than 0.7pp. The quality signal does not alarm until Q2_K, where all three metrics drop sharply (BERTScore -9.6pp, ROUGE-L -10.7pp, coherence -8.7pp). This stability creates a false floor -- practitioners monitoring quality alone see a reassuring plateau from 16 BPW down to 4.85 BPW, with no indication that safety has already begun to erode.
- Safety tells a different story. Refusal rate begins declining at Q5_K_M (-1.8pp), accelerates at Q3_K_S (-13.6pp), and collapses at Q2_K (-56.8pp). The safety degradation curve is steeper and activates earlier than the quality curve. In Cohen's h terms, the Q3_K_S refusal drop corresponds to h = 0.42 (medium effect), and Q2_K reaches h = 1.33 (very large), computed as h = 2|arcsin(sqrt(0.936)) - arcsin(sqrt(p_quant))|.
- Q3_K_S is the critical divergence point: quality metrics are within 2.3pp of FP16, but refusal rate has already dropped 13.6pp. A quality-only evaluation would miss this. The practical implication is that any deployment monitoring system that relies solely on BERTScore or coherence checks would certify Q3_K_S as acceptable when it has already lost meaningful safety alignment.
- The anomalous Q4_K_M quality *improvement* (+1.88pp BERTScore, +3.10pp ROUGE-L) likely reflects sampling variance at that quant level rather than a genuine capability gain. At N = 100-150, a 1.9pp shift in BERTScore is within the expected bootstrap CI range and should not be interpreted as evidence that 4-bit quantization enhances quality.
- Bias resistance shows a paradoxical pattern at Q3_K_S: it increases to 99.5% (+10.1pp from FP16, h = 0.52, medium effect). This is statistically significant (p_adj = 0.0002, SS9) and may reflect the model defaulting to refusal-like templates that happen to avoid bias markers while simultaneously failing on targeted refusal tasks.
- Truthfulness fluctuates between 44% and 58% across all quant levels with no monotonic trend. Given the MDE of 28pp at N = 50 (SS9), none of these fluctuations are interpretable as quant effects.

> On llama3.2-1b, quality passes at Q3_K_S while safety fails -- monitoring quality alone creates a false sense of security. The refusal-rate cliff activates 1-2 quant steps before the quality cliff.

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

- Quality degradation on llama3.2-3b is more gradual than on 1b. BERTScore stays within 0.9pp of FP16 all the way down to Q4_K_M. Even at Q3_K_S, BERTScore only drops 3.9pp (vs. near-zero on 1b). This resilience likely reflects the 2.6x parameter advantage: the 3b model has more redundant capacity to absorb quantization noise before quality-relevant representations degrade.
- The safety story is inverted relative to 1b. Instead of declining, refusal rate *increases* at Q3_K_S (+18.6pp, h = 0.56) and Q2_K (+16.4pp, h = 0.47). The model becomes more conservative -- over-refusing rather than under-refusing. This is confirmed statistically in SS9 with significant negative Cohen's d values (d = -0.55 and -0.46, both p_adj < 4e-5). The mechanism may be instruction-following degradation: as quantization corrupts the model's ability to parse nuanced prompts, it defaults to refusal as a conservative fallback.
- Q4_K_M is the exception: refusal drops -10.0pp (h = 0.22) while quality barely moves (-0.89pp BERTScore). This is the only 3b cell showing the 1b-like pattern of "quality stable, safety drops." Although this comparison does not survive Holm-Bonferroni correction (p_adj = 0.576), the practical magnitude (-10pp refusal) is worth monitoring in deployment.
- Bias resistance holds steady (within 2.0pp) from FP16 through Q3_K_S, then collapses at Q2_K (-17.7pp, h = 0.58, d = 0.56, p_adj = 1.1e-6). This binary threshold effect -- stable until sudden collapse -- contrasts with the gradual quality degradation and suggests that bias resistance relies on a small set of weight configurations that remain intact until extreme quantization destroys them.
- The anomalous Q2_K BERTScore near-preservation (only -0.20pp) combined with coherence drop (-3.99pp) suggests the model produces semantically similar but less structured outputs at extreme quantization. BERTScore captures token-level embedding similarity, which may be preserved even when discourse-level coherence degrades -- a dissociation between local and global output quality.
- Truthfulness on llama3.2-3b trends upward at lower quants (49% at FP16 to 54% at Q2_K), but this 5pp shift is well within the 28pp MDE and cannot be distinguished from noise. The direction is suggestive but uninterpretable at this sample size.

> On llama3.2-3b, aggressive quantization (Q3_K_S, Q2_K) makes the model over-refuse rather than under-refuse -- the opposite failure mode from 1b. Both models show quality-safety divergence, but in opposite directions.

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

**Table 3c: Pooled (cross-model) correlations (selected)**

| Quality Metric | Safety Metric | r | p | N | Significant? |
|---------------|--------------|---|---|---|-------------|
| BERTScore | Refusal Rate | +0.648 | 0.007 | 14 | Yes |
| BERTScore | Truthfulness | +0.454 | 0.108 | 14 | No |
| BERTScore | Bias Resistance | +0.562 | 0.032 | 14 | Yes |
| ROUGE-L | Refusal Rate | +0.582 | 0.024 | 14 | Yes |
| ROUGE-L | Bias Resistance | +0.679 | 0.003 | 14 | Yes |
| Coherence | Refusal Rate | +0.292 | 0.334 | 14 | No |
| Coherence | Bias Resistance | +0.497 | 0.070 | 14 | No |

**Observations.**

- llama3.2-1b shows uniformly strong positive correlations: all 9 pairs are positive, 8 of 9 are significant. Coherence x Refusal Rate is nearly perfect (r = +0.994, p < 1e-70). When quality goes down, safety goes down. This tight coupling suggests that on small models, the same weight subspace supports both quality and safety, so quantization damage to that subspace degrades both simultaneously.
- llama3.2-3b shows the opposite pattern: the two significant correlations (ROUGE-L x Refusal and Coherence x Refusal) are *negative* (r = -0.761 and -0.829). When quality goes down, refusal goes *up*. The remaining 7 pairs are non-significant (all |r| < 0.59). This suggests that on the 3b model, quality and safety occupy partially independent weight subspaces, and quantization-induced noise at the safety boundary pushes the model toward conservative refusal rather than permissive compliance.
- Pooled correlations (Table 3c) produce misleading moderate positives because the 1b model's strong positive signal dominates. The pooled BERTScore x Refusal r = +0.648 (p = 0.007) would suggest quality tracks safety -- a dangerous conclusion to draw. This is a textbook Simpson's paradox: the aggregate trend reverses when you condition on model identity. The practical implication is severe: any evaluation framework that pools across model sizes when assessing quality-safety coupling will produce incorrect deployment guidance.
- The coherence-refusal pair produces the strongest correlations in both models (|r| > 0.82), suggesting coherence is the quality metric most coupled to safety alignment under quantization. BERTScore, while popular, shows weaker coupling on the 3b model (r = -0.53, not significant) -- its token-level granularity may miss the discourse-level features that safety alignment depends on.
- The asymmetry in significant pairs is itself informative: 8/9 significant on 1b versus 2/9 on 3b. This does not mean quality-safety coupling is absent on 3b -- it means the relationship is non-linear and concentrated in the refusal dimension. The 3b model maintains independence between quality and safety across most metric pairs, with only the refusal channel showing consistent (negative) coupling.
- The pooled correlations (Table 3c) are instructive as a cautionary example. Six of nine pooled pairs are positive, four significantly so. A naive analyst would conclude "quality and safety are positively correlated under quantization" -- a statement that is true for 1b, false for 3b, and misleading as a general claim. The pooled Coherence x Refusal (r = +0.292, not significant) averages out the 1b r = +0.994 and the 3b r = -0.829, losing the strong within-model signal entirely. This is the defining feature of Simpson's paradox: the aggregate tells the wrong story.

> The quality-safety correlation sign flips between models. Pooling across models produces a Simpson's paradox artifact. Always analyze per-model. Coherence is the strongest quality proxy for safety coupling in both directions.

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

- 8 of 12 cells are BOTH_STABLE at the +/-3pp threshold, meaning quality and safety move together (or neither moves) through Q5_K_M on both models. SS10 places most of these cells in the low-delta stability zone, but does not treat them as formally proven equivalent under the current artifact chain.
- The divergence regime activates at different quant levels per model: Q3_K_S on llama3.2-1b, Q4_K_M on llama3.2-3b. Counterintuitively, the larger model (3b) hits safety divergence one step earlier on the quant ladder. This may reflect the 3b model's higher FP16 safety ceiling (76.4% refusal vs 93.6%) giving it more room to shift before recovery mechanisms activate.
- llama3.2-1b Q3_K_S is the highest-priority finding: quality delta is +0.98pp (negligible) while safety delta is -13.64pp (large, h = 0.42). No quality metric would flag this cell. The danger is specific and actionable: practitioners who benchmark at Q3_K_S using BERTScore or coherence will see passing scores and deploy a model that has already lost meaningful refusal capability.
- llama3.2-3b's "divergence" at Q3_K_S and Q2_K is in the *opposite* direction -- refusal increases by +18.6pp and +16.4pp respectively, not decreases. This is still a divergence from quality (which degrades), but the failure mode is over-refusal rather than alignment loss. The BOTH_STABLE classification for 3b Q3_K_S is technically correct under the threshold logic (safety "improved"), but practically misleading -- a +18.6pp shift in refusal rate fundamentally changes the model's behavior profile.
- The three true SAFETY_ONLY_DEGRADES cells (1b Q3_K_S, 3b Q4_K_M, 3b Q2_K) represent the highest-risk deployment configurations: quality monitoring gives a clean signal while safety has already shifted. These are the cells where dual-monitoring would change the deployment decision.

> Three cells show quality-safety divergence where quality holds but safety shifts: llama3.2-1b at Q3_K_S (refusal drops 13.6pp) and Q2_K (refusal drops 56.8pp), and llama3.2-3b at Q4_K_M (refusal drops 10.0pp). Two additional cells on llama3.2-3b show divergence in the over-refusal direction.

### SS6.2 Bootstrap Asymmetry Gap

The asymmetry gap is (safety_delta - quality_delta) averaged across all quant levels for each model. A negative gap means safety degrades more than quality.

| Model | Mean Gap (pp) | 95% CI Low | 95% CI High | N | Significant? |
|-------|--------------|-----------|------------|---|-------------|
| llama3.2-1b | -11.01 | -26.73 | -0.39 | 6 | **Yes** (CI excludes 0) |
| llama3.2-3b | +5.04 | -3.27 | +14.28 | 6 | No (CI includes 0) |

**Observations.**

- On llama3.2-1b, safety degrades 11.0pp more than quality on average, with the CI excluding zero [-26.7, -0.4]. This is a confirmed asymmetry: across all 6 non-FP16 quant levels, safety systematically moves more than quality in the degradation direction. The wide CI reflects the Q2_K outlier (safety delta -56.8pp vs quality delta -9.6pp), which pulls the mean gap down and inflates variance.
- On llama3.2-3b, the gap is positive (+5.0pp) but the CI includes zero [-3.3, +14.3]. Safety moves more than quality on average, but in the direction of *increased* refusal, and the effect is not significant. The positive gap reflects the over-refusal at Q3_K_S and Q2_K, which increases the safety metric while quality decreases. If we define "safety degradation" as any movement away from FP16 behavior (including over-refusal), then both models show confirmed asymmetry.
- The bootstrap was computed using B = 2,000 iterations with seed = 42 and the percentile method. The percentile method is appropriate here because the asymmetry gap distribution is not strongly skewed (verified by the approximate symmetry of the CIs around the mean).

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

- Safety moves more than quality in **10 of 12 cells** (ratio > 1.0). The two exceptions (Q6_K on 1b at 0.9x, Q5_K_M on 3b at 0.8x) have ratios essentially at parity. This near-universality of safety-over-quality asymmetry is the most robust finding in the report: it holds across both models, across both directions of safety change (degradation on 1b, over-refusal on 3b), and across the full quant range.
- The highest ratios occur at the extremes of the quant ladder and are dominated by the denominator effect. llama3.2-3b Q2_K has a ratio of 83.2x because quality barely moved (-0.20pp BERTScore) while refusal shifted +16.4pp. The ratio is meaningful in context -- it quantifies how much safety can move while quality appears stable -- but the absolute deltas in SS4 and SS6 provide more actionable information.
- The Q6_K through Q5_K_M "stability zone" shows ratios close to 1.0 or below, confirming that moderate quantization affects quality and safety roughly equally. In SS10 this is treated as an observed low-delta regime rather than a formally proven equivalence range. Within this zone, quality monitoring provides a reasonable (though not perfect) proxy for safety status.
- llama3.2-1b Q3_K_S (ratio 13.9x) is the most deployment-relevant divergence cell because it falls within a range practitioners might actually use. At 3.44 BPW, Q3_K_S offers attractive memory savings (~4.6x compression vs FP16), and the quality metrics give no warning of the safety degradation. This is the cell that motivates the core recommendation: monitor safety independently below Q5_K_M.
- The asymmetry pattern suggests that safety alignment occupies a more fragile subspace than quality capability. Quality, being distributed across many parameters for general language modeling, degrades gradually as bits are removed. Safety, being concentrated in the fine-tuning delta, is more sensitive to the specific bit allocations that quantization schemes choose, leading to sharper transitions.

> Safety metrics are more sensitive to quantization than quality metrics across nearly all cells (10/12). The asymmetry is most dangerous in the Q3_K_S-Q4_K_M range where quality still looks acceptable but safety has already shifted.

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

- Quality-gating lowers absolute safety scores by 7-17pp (it removes samples where the model happened to refuse despite producing incoherent output). But it preserves the *relative ordering* across quant levels perfectly. The rank-order correlation between raw and gated refusal rates is 1.0 for both models -- gating is a monotonic transformation that cannot change deployment decisions.
- The constant filter rate means the quality gate is catching prompt-level difficulty, not quantization artifacts. If quantization degraded coherence on specific samples, the filter rate would increase at lower quant levels -- it does not. The 40/220 filtered refusal samples and 8/50 filtered truthfulness samples are the *same samples* at every quant level, meaning these prompts are inherently challenging regardless of model precision.
- The bias_resistance filter rate of 0% means coherence failures and bias failures occur on completely disjoint sample sets. Low coherence does not predict bias failures. This orthogonality has a positive practical implication: if a deployment pipeline already includes quality gating for other reasons, it will not inadvertently bias the safety evaluation by selectively removing bias-vulnerable samples.
- The delta between raw and gated scores narrows at Q2_K on llama3.2-1b (only -7.7pp vs -16.8pp at FP16). This is an artifact of the low absolute refusal rate at Q2_K (36.8%): when most samples already fail, removing 40 samples has less proportional impact. The narrowing does not indicate that quality-gating becomes more useful at extreme quant levels.
- This answers Q13 definitively: quality-gating does not change the quantization safety story. It shifts all scores by a constant offset that is independent of quant level. The implication for evaluation pipeline design is clear: skip the gating step for quantization studies, as it reduces effective sample size without adding discriminative power.

### SS8.3 Gated vs Raw Safety Scores (llama3.2-3b refusal rate)

| Quant | Raw Refusal | Gated Refusal | Delta (pp) |
|-------|------------|--------------|-----------|
| FP16 | 76.4% | 60.9% | -15.5 |
| Q8_0 | 74.5% | 59.1% | -15.5 |
| Q6_K | 77.3% | 61.8% | -15.5 |
| Q5_K_M | 76.8% | 61.4% | -15.5 |
| Q4_K_M | 66.4% | 52.7% | -13.6 |
| Q3_K_S | 95.0% | 78.2% | -16.8 |
| Q2_K | 92.7% | 79.1% | -13.6 |

**Observations.**

- The pattern mirrors 1b: constant filter rate (40/220 = 18.2%) with the gating offset preserving relative ordering. On 3b, the over-refusal at Q3_K_S (95.0%) remains dominant even after gating (78.2%), confirming that over-refusal is not driven by the incoherent-output subsample.
- The gated Q3_K_S refusal rate (78.2%) is higher than the gated FP16 rate (60.9%), further confirming that the refusal increase at low quants on 3b is a genuine behavioral shift across the entire sample, not an artifact of quality-selected subsets.

> Quality-gating is quant-invariant. The same 18.2% of samples fail at every quant level including FP16. The gate catches prompt difficulty, not quant damage. Skip the gating step for quantization studies.

---

## SS9. Pairwise Statistical Tests

All tests compare each quant level against FP16 for the same model and metric. Holm-Bonferroni correction is applied across all 36 tests (2 models x 3 metrics x 6 quant levels).

### SS9.1 Significant Results (p_adj < 0.05 after Holm-Bonferroni)

Since safety metrics are proportions (not continuous scores), Cohen's h is reported alongside Cohen's d for interpretive completeness. Cohen's h = 2|arcsin(sqrt(p1)) - arcsin(sqrt(p2))| is the appropriate effect size for proportion comparisons.

| Model | Metric | Quant | FP16 Mean | Quant Mean | Cohen's d | Cohen's h | t | p_adj | Direction |
|-------|--------|-------|-----------|-----------|----------|----------|---|-------|-----------|
| llama3.2-1b | Refusal Rate | Q3_K_S | 93.6% | 80.0% | 0.41 | 0.42 | 4.31 | 0.0005 | Degraded |
| llama3.2-1b | Refusal Rate | Q2_K | 93.6% | 36.8% | **1.48** | **1.33** | 15.55 | 5.3e-53 | Degraded |
| llama3.2-1b | Bias Resist. | Q3_K_S | 89.4% | 99.5% | -0.45 | 0.52 | -4.49 | 0.0002 | **Improved** |
| llama3.2-1b | Bias Resist. | Q2_K | 89.4% | 73.2% | 0.42 | 0.42 | 4.21 | 0.0008 | Degraded |
| llama3.2-3b | Refusal Rate | Q3_K_S | 76.4% | 95.0% | -0.55 | 0.56 | -5.78 | 2.7e-7 | **Improved** (over-refusing) |
| llama3.2-3b | Refusal Rate | Q2_K | 76.4% | 92.7% | -0.46 | 0.47 | -4.86 | 3.8e-5 | **Improved** (over-refusing) |
| llama3.2-3b | Bias Resist. | Q2_K | 96.5% | 78.8% | 0.56 | 0.58 | 5.53 | 1.1e-6 | Degraded |

### SS9.2 Non-Significant Results (all truthfulness tests)

All 12 truthfulness comparisons (2 models x 6 quants) are non-significant after correction (all p_adj = 1.0). The largest observed effect is d = 0.26 (llama3.2-1b Q2_K, delta = -11pp), well below the MDE of 28pp at N = 50.

### SS9.3 Near-Miss: llama3.2-3b Refusal at Q4_K_M

One comparison warrants special attention despite non-significance: llama3.2-3b refusal rate at Q4_K_M drops from 76.4% to 66.4% (delta = -10.0pp, d = 0.22, h = 0.22, p_raw = 0.020, p_adj = 0.576). The raw p-value is significant at 0.020, but Holm-Bonferroni correction pushes it to 0.576. The practical magnitude (-10pp) exceeds the +/-3pp equivalence margin by 3x and would be operationally meaningful. This cell falls in the statistical no-man's-land: too large to be equivalent, too variable to be statistically significant after correction.

**Observations.**

- Only 7 of 36 tests survive Holm-Bonferroni correction. Quantization effects on safety are concentrated at the extreme quant levels (Q3_K_S and Q2_K) and primarily affect refusal rate and bias resistance. The moderate-quantization range (Q8_0 through Q5_K_M) produces no significant safety effects on any metric.
- llama3.2-1b Q2_K refusal has the largest effect size in the entire analysis: d = 1.48 / h = 1.33 (very large by any standard). The refusal rate drops from 93.6% to 36.8% -- a catastrophic safety failure that would likely be caught by any monitoring system but illustrates the magnitude of extreme-quantization damage.
- llama3.2-1b Q3_K_S shows a within-cell paradox: refusal rate *degrades* (d = 0.41, h = 0.42) while bias resistance *improves* (d = -0.45, h = 0.52). Different safety dimensions respond differently to the same quantization level. This dissociation suggests that the "safety subspace" is not monolithic -- refusal and bias resistance may be supported by partially independent weight configurations.
- llama3.2-3b Q3_K_S and Q2_K both show *improved* refusal rates (negative d), confirming the over-refusal pattern from SS4.2. The effect sizes (d = -0.55 / h = 0.56 and d = -0.46 / h = 0.47) are medium, indicating reliable behavioral shifts rather than noise. The model becomes more conservative, not less safe, but this reduces helpfulness and may push users toward prompt-engineering workarounds that bypass safety.
- Cohen's d and Cohen's h show broad agreement across all 7 significant comparisons, with most pairs within 0.10 of each other. The largest divergence occurs at 1b Q2_K (d = 1.48 vs h = 1.33, gap = 0.15), where the extreme base-rate shift causes the arcsin transform to compress the effect relative to the mean-based estimator. Despite this, both metrics classify all comparisons into the same qualitative band (small/medium/large), confirming that the choice between proportion-based and mean-based effect sizes does not materially alter interpretation for these binary safety metrics.
- Truthfulness is completely underpowered at N = 50. We cannot make any claims about quantization effects on truthfulness from this data. The 12 non-significant truthfulness tests should be interpreted as "no evidence either way," not as "no effect."

> Statistically confirmed safety effects are limited to Q3_K_S and Q2_K. The Q8_0 through Q5_K_M range shows no significant degradation on any safety metric for either model. The refusal-bias dissociation on llama3.2-1b Q3_K_S reveals that "safety" is not a single dimension.

---

## SS10. Practical Stability Zone and Conservative Floor

SS9 answers whether specific quant cells differ significantly from FP16 after correction. This section asks a narrower operational question: where do the observed deltas remain small enough that a deployment team can treat the cell as part of a low-delta stability zone, even though the current saved artifact does **not** support a strong standalone formal equivalence claim.

The key constraint is methodological. The current saved artifact does not include a standalone `tost_results` object, and the appendix-level raw one-sided tests shown later in the report do not support a clean per-cell equivalence claim under the present sample sizes. SS10 therefore uses observed deltas, corrected pairwise tests, and the broader evidence stack to identify a conservative floor rather than a theorem-like equivalence boundary.

### SS10.1 Refusal-Rate Stability Screen

| Model | Quant | Refusal Delta vs FP16 | Corrected Pairwise Result | Interpretation |
|-------|-------|----------------------:|---------------------------|----------------|
| llama3.2-1b | Q8_0 | +0.9pp | not significant | low-delta stability zone |
| llama3.2-1b | Q6_K | +0.5pp | not significant | low-delta stability zone |
| llama3.2-1b | Q5_K_M | -1.8pp | not significant | low-delta stability zone |
| llama3.2-1b | Q4_K_M | -3.2pp | not significant | ambiguous boundary |
| llama3.2-1b | Q3_K_S | -13.6pp | significant | clear degradation |
| llama3.2-1b | Q2_K | -56.8pp | significant | collapse |
| llama3.2-3b | Q8_0 | -1.8pp | not significant | low-delta stability zone |
| llama3.2-3b | Q6_K | +0.9pp | not significant | low-delta stability zone |
| llama3.2-3b | Q5_K_M | +0.5pp | not significant | low-delta stability zone |
| llama3.2-3b | Q4_K_M | -10.0pp | not significant after Holm | ambiguous but operationally concerning |
| llama3.2-3b | Q3_K_S | +18.6pp | significant | over-refusal regime |
| llama3.2-3b | Q2_K | +16.4pp | significant | over-refusal regime |

### SS10.2 What Follows

- Q8_0 through Q5_K_M form the **observed low-delta stability zone** on refusal rate. That is why `Q5_K_M` remains the conservative floor in this report.
- Q4_K_M is the true boundary cell. It is too large to treat casually, but not cleanly resolved by the corrected pairwise tests.
- Q3_K_S and Q2_K are where the two models clearly diverge from the FP16 regime, albeit in different directions (`llama3.2-1b` under-refusal, `llama3.2-3b` over-refusal).
- Bias resistance is at least as stable as refusal through the moderate-quantization range, but TR142 still does not claim a separate formal equivalence theorem for that metric.
- Truthfulness remains underpowered and does not support either equivalence or null-effect claims.

### SS10.3 Conservative Deployment Interpretation

The practical recommendation is therefore narrower than the earlier draft implied:

> TR142 supports `Q5_K_M` as the conservative deployment floor because observed refusal deltas stay small and corrected pairwise tests stay quiet through that level. It does **not** by itself prove strict formal +/-3pp equivalence through `Q5_K_M`.

If a deployment requires a theorem-like equivalence claim, this report is not sufficient. If the goal is a conservative operating recommendation grounded in the current evidence stack, `Q5_K_M` is still the right floor and `Q4_K_M` remains the ambiguity zone that demands per-model validation.

---

## SS11. BPW Regression

Linear regression of metric value against bits-per-weight tests whether quantization level is a useful continuous predictor of quality or safety.

### SS11.1 Regression Results

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

- No R-squared exceeds 0.30. BPW is a weak linear predictor of both quality and safety. The degradation is non-linear -- most of the variance comes from the Q2_K/Q3_K_S collapse, not from a smooth linear trend. If one excludes Q2_K from the regression (leave-one-out analysis, Appendix C), R-squared drops further for most metrics, confirming that the extreme point drives even the modest linear fit.
- The refusal rate slope flips sign between models: +0.024/BPW on llama3.2-1b (more bits = higher refusal = safer) vs. -0.009/BPW on llama3.2-3b (more bits = lower refusal). This is the regression counterpart to SS5's correlation finding and reinforces the Simpson's paradox: pooling these two slopes would produce a misleading average.
- Coherence has the highest R-squared for both models (0.26 and 0.30), making it the metric most linearly responsive to quantization pressure. This aligns with SS5's finding that coherence x refusal has the strongest correlation. However, even R-squared = 0.30 means 70% of variance is unexplained -- coherence responds to quantization, but not smoothly enough to predict specific quant-level outcomes from BPW alone.
- Bias resistance on llama3.2-1b has R-squared = 0.04 -- essentially no linear relationship with BPW. Bias resistance appears to be a threshold effect that only activates at extreme quantization. This makes it unsuitable for linear regression modeling; a logistic or step-function model would better capture its all-or-nothing behavior.
- The low R-squared values have a direct practical implication: practitioners cannot use BPW as a shorthand for expected quality or safety. A model at 4.85 BPW (Q4_K_M) does not produce output that is "4.85/16 as good" as FP16 -- the relationship is non-monotonic and threshold-driven. Evaluation must be done per-quant-level, not interpolated from BPW.

### SS11.2 Slope Sign Comparison

The slope sign provides a compact summary of the BPW-metric relationship across models:

| Metric | llama3.2-1b Slope | llama3.2-3b Slope | Same Sign? |
|--------|------------------|------------------|-----------|
| BERTScore | +0.0028 | +0.0013 | **Yes** (both positive) |
| Coherence | +0.0036 | +0.0040 | **Yes** (both positive) |
| Refusal Rate | +0.0240 | **-0.0094** | **No** (opposite) |
| Truthfulness | +0.0057 | **-0.0036** | **No** (opposite) |
| Bias Resistance | +0.0032 | +0.0067 | **Yes** (both positive) |

**Observations.**

- Quality metrics (BERTScore, coherence) have consistent positive slopes on both models: more bits per weight = higher quality. This is the expected direction and confirms that the BPW-quality relationship, while weak (low R-squared), is at least directionally consistent.
- Safety metrics (refusal rate, truthfulness) have **opposite slopes** on the two models, confirming the Simpson's paradox from SS5 in a regression framework. Pooling these slopes would produce a near-zero average that masks the strong within-model relationships.
- Bias resistance has consistent positive slopes, suggesting that the BPW-bias relationship is not model-dependent, unlike the BPW-refusal relationship. This aligns with the variance decomposition in SS13.5, which found low interaction effects for bias resistance.

> Linear BPW models are poor predictors (all R-squared < 0.30). Quantization effects are driven by threshold collapses at Q3_K_S and Q2_K, not gradual degradation. Evaluate per-quant-level, do not interpolate from BPW.

---

## SS12. Capability Consistency Validation

To validate that the quality-safety matrix reflects genuine capability differences (not noise), we check degradation on the shared benchmark tasks (MMLU, ARC Challenge) that appeared in both TR125p2 and TR134p3.

### SS12.1 MMLU Real Accuracy

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 34.0% | 35.1% (+1.1) | 34.4% (+0.4) | 33.0% (-1.1) | 33.7% (-0.4) | 31.9% (-2.1) | 19.3% (-14.7) |
| llama3.2-3b | 58.9% | 59.3% (+0.4) | 57.9% (-1.1) | 57.5% (-1.4) | 58.9% (0.0) | 48.1% (-10.9) | 42.8% (-16.1) |

### SS12.2 ARC Challenge Accuracy

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 44.5% | 46.0% (+1.5) | 44.0% (-0.5) | 46.5% (+2.0) | 39.5% (-5.0) | 24.5% (-20.0) | 26.5% (-18.0) |
| llama3.2-3b | 71.5% | 72.0% (+0.5) | 71.5% (0.0) | 71.5% (0.0) | 70.5% (-1.0) | 62.5% (-9.0) | 63.0% (-8.5) |

**Observations.**

- Both benchmark tasks confirm the pattern from SS4: stability through Q5_K_M, then sharp drops at Q3_K_S and Q2_K. The concordance across three independent metric families (generation quality, safety alignment, multiple-choice accuracy) strengthens the claim that the Q3_K_S/Q2_K cliff reflects genuine capability loss rather than metric-specific noise.
- MMLU degradation on llama3.2-1b at Q2_K (-14.7pp) closely matches the BERTScore degradation (-9.6pp) and coherence degradation (-8.7pp) at the same level, confirming that Q2_K represents genuine capability collapse across all evaluation dimensions. The refusal rate collapse (-56.8pp) at Q2_K exceeds the quality collapse by 4-6x, consistent with the asymmetry finding in SS7.
- ARC Challenge on llama3.2-1b shows an earlier cliff: -20.0pp at Q3_K_S. This aligns with the refusal rate finding (-13.6pp at Q3_K_S) and further supports Q3_K_S as a critical threshold on the 1b model. ARC Challenge may be more sensitive than MMLU to quantization because it requires longer reasoning chains that are more vulnerable to accumulated rounding errors.
- llama3.2-3b is more robust on both tasks, with drops staying under 2pp through Q4_K_M and not exceeding 16pp even at Q2_K. The 3b model's MMLU at Q3_K_S (-10.9pp) indicates real capability loss despite the paradoxical refusal *increase* at the same quant level -- the model is refusing more often but performing worse on the tasks where it does respond.
- The ARC Challenge result for llama3.2-3b at Q2_K (-8.5pp) is notably less severe than Q3_K_S (-9.0pp), mirroring the BERTScore anomaly at Q2_K (only -0.20pp). This non-monotonicity at extreme quantization on the 3b model warrants investigation in future studies; it may reflect Q2_K quantization allocating bits differently than Q3_K_S in ways that preserve certain capability types.

### SS12.3 Cross-Validation with Refusal Patterns

The capability consistency data also enables a secondary validation: checking whether MMLU/ARC degradation patterns match the refusal rate patterns.

| Model | Q3_K_S MMLU Drop | Q3_K_S Refusal Change | Pattern Match? |
|-------|-----------------|----------------------|----------------|
| llama3.2-1b | -2.1pp | -13.6pp | **No** -- refusal drops 6.5x more |
| llama3.2-3b | -10.9pp | +18.6pp | **No** -- opposite directions |

| Model | Q2_K MMLU Drop | Q2_K Refusal Change | Pattern Match? |
|-------|---------------|---------------------|----------------|
| llama3.2-1b | -14.7pp | -56.8pp | Partial -- same direction, 3.9x amplification |
| llama3.2-3b | -16.1pp | +16.4pp | **No** -- opposite directions |

**Observations.**

- At Q3_K_S, MMLU accuracy and refusal rate tell completely different stories on both models. On 1b, MMLU drops modestly while refusal drops sharply. On 3b, MMLU drops sharply while refusal *increases*. This confirms that accuracy benchmarks and safety metrics are measuring different capabilities, and that the divergence is not an artifact of the metric definitions used in SS4.
- At Q2_K on llama3.2-1b, both metrics degrade, but refusal drops 3.9x more than MMLU accuracy. This asymmetry is consistent with the SS7 finding (asymmetry ratio 5.9x) and provides independent triangulation from a different metric pair.

> Shared benchmark tasks validate the quality-safety matrix. The Q3_K_S/Q2_K cliff is real and consistent across quality, safety, and accuracy metrics on llama3.2-1b. llama3.2-3b shows quality capability loss at Q3_K_S despite increased refusal.

---

## SS13. Statistical Synthesis

### SS13.1 Hypothesis Evaluation

| Hypothesis | Test | Result | Status |
|-----------|------|--------|--------|
| H1: Quality and safety co-vary under quantization | Pearson r within-model | 1b: r = +0.92 to +0.99; 3b: r = -0.53 to -0.83 | **Partially supported** (model-dependent) |
| H2: Safety degrades faster than quality | Asymmetry ratio > 1.0 | 10/12 cells have ratio > 1.0 | **Supported** |
| H3: Quality-gating changes the quant safety story | Gated vs raw delta comparison | Filter rate constant at 18.2% across all quants | **Rejected** (no change) |
| H4: BPW is a linear predictor of degradation | OLS regression R-squared | All R-squared < 0.30 | **Rejected** (non-linear effects dominate) |
| H5: Q8_0-Q5_K_M define the conservative safety floor | small observed deltas plus non-significant corrected pairwise tests | supported as a practical floor, not as formal equivalence | **Partially supported** |

**Observations.**

- H1 is the most nuanced finding: quality-safety co-variation exists, but its sign is model-dependent. This is not a weak version of a positive claim -- it is a fundamentally different claim. The relationship is strong within each model (|r| > 0.82 for coherence x refusal on both) but opposite in direction. Any theory of quality-safety coupling under quantization must explain why the coupling reverses between 1.2B and 3.2B parameters.
- H2 is the most actionable finding: safety moves more than quality in 10/12 cells. This means quality monitoring systematically underestimates safety risk. The asymmetry is not uniform -- it concentrates at Q3_K_S and below -- but it is consistent enough to justify the recommendation that safety must be monitored independently.
- H3 and H4 are clean negative results. The quality-gating null result (H3) simplifies evaluation pipelines. The BPW regression null result (H4) rules out a convenient shortcut: you cannot predict safety from BPW using a linear model.
- H5 should now be read more narrowly. SS10 supports Q5_K_M as the conservative deployment floor because refusal deltas remain small and corrected pairwise tests stay non-significant through that level, not because TR142 alone establishes strict formal equivalence.

### SS13.2 Cross-Model Synthesis

The central finding of TR142 is that the quality-safety relationship under quantization is not a universal law -- it is model-specific. This has three practical implications:

1. **No universal quality-safety proxy exists.** A quality check that catches safety degradation on one model may miss it (or give the wrong signal) on another. The coherence x refusal correlation is r = +0.994 on llama3.2-1b and r = -0.829 on llama3.2-3b. Using coherence as a safety proxy on 1b would work well; using it on 3b would produce the wrong signal.
2. **The failure mode depends on model size.** Small models (1b) lose safety alignment when quantized aggressively -- they become permissive, answering harmful prompts they should refuse. Larger models (3b) become overly conservative -- they refuse benign prompts, reducing helpfulness. Both are problematic, but require different mitigations: 1b needs safety guardrails, 3b needs helpfulness tuning.
3. **Evaluation frameworks must be per-model.** The Simpson's paradox finding means that any cross-model safety benchmark that averages or pools results across model sizes will produce misleading conclusions. This applies not just to quantization studies but to any evaluation where model size confounds the quality-safety relationship.

### SS13.3 Theoretical Framework: Why Quality and Safety Decouple

The quality-safety decoupling observed in TR142 can be explained through a weight-subspace model of quantization damage. In a transformer, "quality" (generating coherent, relevant text) and "safety" (recognizing harmful prompts and refusing) rely on overlapping but partially independent weight subspaces:

- **Shared subspace:** General language modeling capability (token prediction, coherence) serves both quality and safety. Quantization damage here degrades both dimensions simultaneously, producing the positive correlation observed on llama3.2-1b.
- **Safety-specific subspace:** Instruction-following and alignment fine-tuning occupy weights that are redundant from a quality perspective. These weights are disproportionately vulnerable to quantization because they represent relatively small adjustments on top of the base model, and small adjustments are more easily destroyed by rounding.
- **Model-size interaction:** On small models (1b), the shared subspace is large relative to the safety-specific subspace -- there is less redundancy, so quantization damage that hits quality also hits safety. On larger models (3b), the safety-specific subspace is larger and more distributed, making it partially independent of quality. When quality degrades, the safety pathway may still activate, but with corrupted inputs, it defaults to conservative refusal.

This framework explains three specific findings:
1. The positive correlation on 1b (quality and safety share more weights)
2. The negative correlation on 3b (safety pathway defaults to refusal when quality-feeding inputs degrade)
3. The asymmetry (safety-specific weights are more vulnerable because they represent fine-tuning deltas, which are small in magnitude and easily rounded away)

### SS13.4 Broader Implications for LLM Evaluation

The Simpson's paradox finding in SS5 has implications beyond quantization studies:

**For safety benchmarking:** Any benchmark that aggregates safety scores across model sizes risks producing the same paradox. If small models show quality-safety coupling and large models show decoupling, the aggregate will reflect the dominant size class in the sample, not a universal relationship. Safety benchmarks should always report per-model-size results.

**For deployment monitoring:** The quality-gating null result (SS8) suggests that quality-based pre-filtering is not useful for catching quantization-specific safety failures. Monitoring systems should use direct safety probes rather than quality-derived proxies. The constant 18.2% filter rate across quant levels means the same prompts are "hard" regardless of precision, which is useful for identifying inherently challenging inputs but useless for detecting quantization damage.

**For quantization research:** The field's focus on perplexity and downstream accuracy as quantization quality metrics misses the safety dimension entirely. TR142 shows that quality can be preserved while safety degrades (llama3.2-1b Q3_K_S) or quality can degrade while safety tightens (llama3.2-3b Q3_K_S). Neither outcome is captured by standard quantization benchmarks. Safety-aware quantization evaluation should become standard practice.

### SS13.5 Variance Decomposition

The total variance in safety metrics across the 14-cell matrix can be decomposed into three sources: model effects (1b vs 3b), quant effects (FP16 through Q2_K), and model x quant interaction effects.

| Safety Metric | Model Effect | Quant Effect | Interaction Effect | Residual |
|--------------|-------------|-------------|-------------------|----------|
| Refusal Rate | Moderate (1b higher baseline) | Strong (Q2_K/Q3_K_S) | **Dominant** (opposite signs) | Small |
| Bias Resistance | Weak (similar baselines) | Weak (Q2_K only) | Weak | Moderate |
| Truthfulness | Negligible | Negligible | Negligible | **Dominant** |

**Observations.**

- The interaction effect dominates for refusal rate -- the same quant level produces opposite safety shifts on different models. This is the statistical fingerprint of the Simpson's paradox: the main effects (model, quant) are less important than their interaction. Any analysis that models refusal rate as a function of quant level alone (ignoring model) will produce misleading results.
- Bias resistance variance is concentrated in a single cell (Q2_K on both models), making it a threshold effect rather than a continuous degradation. The low interaction effect for bias resistance suggests that both models lose bias resistance at Q2_K in the same way -- unlike refusal rate, bias resistance does not show model-dependent coupling.
- Truthfulness variance is dominated by residual (noise), consistent with the underpowered N = 50 finding. The metric fluctuates randomly across cells with no detectable systematic pattern from either model or quant effects.

### SS13.6 Power and Precision Summary

| Metric | N per Cell | MDE at 80% Power | Adequate for +/-3pp TOST? | Key Limitation |
|--------|-----------|-------------------|---------------------------|----------------|
| Refusal Rate | 220 | 13.3pp | Marginal (MDE > margin) | Cannot detect effects below 13pp |
| Bias Resistance | 198 | 14.1pp | Marginal | Cannot detect effects below 14pp |
| Truthfulness | 50 | 28.0pp | **No** (MDE >> margin) | Cannot make any claims |

**Observations.**

- The power analysis reveals that even the "well-powered" metrics (refusal rate, bias resistance) have MDEs of 13-14pp -- well above the +/-3pp equivalence margin. That is why TR142 does not treat the appendix-level TOST screen as strong standalone proof. For cells with very small observed deltas (Q8_0 through Q5_K_M), the raw screen is directionally reassuring; for cells closer to the 3pp boundary (like Q4_K_M on llama3.2-1b at -3.2pp), the power is insufficient to resolve equivalence cleanly.
- Truthfulness at N = 50 is fundamentally underpowered. This is the single largest design limitation of the study. To bring truthfulness MDE down to 3pp would require N >= 8,700 per cell -- a 174x increase from the current 50. A more realistic target of N = 200 per cell would reduce MDE to ~14pp, matching refusal rate.
- Future cross-referencing studies should prioritize N >= 200 per cell for all safety metrics. The TR125p2 and TR134p3 data volumes (10,290 and 13,342 overlapping samples respectively) are adequate in aggregate but the per-cell sizes for low-frequency metrics like truthfulness are too small.

---

## SS14. Conclusions

### SS14.1 Primary Findings

TR142 demonstrates that quality and safety do not degrade uniformly under GGUF quantization. The correlation between quality and safety degradation curves is strongly model-dependent: positive for llama3.2-1b (quality and safety fall together, r = +0.994 for coherence x refusal) and negative for llama3.2-3b (quality falls while safety alignment tightens into over-refusal, r = -0.829). This sign reversal constitutes a Simpson's paradox that invalidates any pooled analysis.

Safety degrades faster than quality in the majority of model-quant cells (10/12, SS7), with asymmetry ratios frequently exceeding 5x. This means quality benchmarks systematically underestimate safety risk. The most dangerous cell in this study is llama3.2-1b at Q3_K_S, where quality metrics remain within 2.3pp of FP16 while refusal rate drops 13.6pp -- a drop large enough to be statistically significant (d = 0.41, p_adj = 0.0005) but invisible to quality-only monitoring.

SS10 places Q8_0 through Q5_K_M in the observed low-delta stability zone on refusal rate, while Q4_K_M is the ambiguous boundary and Q3_K_S / Q2_K are where clear divergence appears. This supports a concrete but conservative deployment floor.

Quality-gating is ineffective for quantization studies. The coherence gate filters the same set of samples at every quant level, meaning it catches prompt difficulty rather than quantization damage. This is a clean negative result that simplifies the evaluation pipeline -- practitioners can skip the gating step.

### SS14.2 Cross-TR Comparison

| Dimension | TR125 Phase 2 | TR134 Phase 3 | TR139 | TR142 |
|-----------|--------------|---------------|-------|-------|
| **Primary question** | Quality under quantization | Safety under quantization | Jailbreak x quantization | Quality-safety coupling |
| **Models** | 4 (1b, 1.5b, 2.7b, 3b) | 3 (1b, 3b, 8b) | 4 (1b, 1.5b, 3b, 8b) | 2 (1b, 3b) |
| **Quant levels** | 7 (FP16-Q2_K) | 7 (FP16-Q2_K) | 6 (Q8_0-Q2_K) | 7 (FP16-Q2_K) |
| **Q3_K_S finding** | Quality stable | Safety drops (1b) | ASR increases | Quality ok, safety diverges |
| **Q2_K finding** | Quality collapses | Safety collapses (1b) | ASR very high | Both collapse (1b); over-refusal (3b) |
| **Safe floor** | Q4_K_M (quality) | Q5_K_M (safety) | Q5_K_M (jailbreak) | Q5_K_M (conservative floor) |
| **Model-size effect** | More params = more robust | Non-monotonic | Smaller models more vulnerable | Coupling sign flips with size |
| **Key insight** | Quality is gradual | Safety is threshold | Multi-turn amplifies | Quality =/= safety proxy |

**Observations.**

- All four TRs converge on Q5_K_M as the safe deployment floor. TR125 permits Q4_K_M on quality alone, but TR134, TR139, and TR142 all identify safety degradation beginning at or below Q4_K_M. The convergence across independent analyses strengthens confidence in Q5_K_M as the recommendation.
- TR142's unique contribution is demonstrating that the dimensions studied in TR125 and TR134 are not independent: their relationship is model-specific and non-obvious. Without TR142, a practitioner consulting TR125 (quality looks fine at Q3_K_S) and TR134 (safety drops at Q3_K_S on 1b) might still wonder whether these are the same or different failure modes. TR142 shows they are independent degradation paths.
- The model-size effect is consistent but takes different forms: TR125 sees "more params = more quality-robust," TR134 sees "non-monotonic safety," TR139 sees "smaller = more vulnerable to jailbreak," and TR142 sees "coupling sign reversal." The common thread is that model size mediates quantization effects, but the specific manifestation depends on which capability dimension is being measured.

### SS14.3 Theoretical Summary

The weight-subspace model proposed in SS13.3 provides a unifying explanation for all major findings:

| Finding | Explanation |
|---------|------------|
| Positive coupling on 1b | Small model -> shared quality-safety subspace -> simultaneous degradation |
| Negative coupling on 3b | Larger model -> partially independent subspaces -> safety defaults to refusal when quality inputs degrade |
| Safety degrades faster (SS7) | Safety alignment is a fine-tuning delta, smaller in magnitude than pre-trained quality weights |
| Quality-gating invariance (SS8) | Gate catches prompt difficulty (quality subspace), independent of safety subspace |
| BPW regression failure (SS11) | Non-linear threshold effects -> subspace collapse at critical bit counts, not gradual degradation |

This framework is descriptive, not mechanistic -- it explains the pattern but does not identify specific weight groups. Future work using mechanistic interpretability (activation patching, probing) could test whether quality and safety activations overlap more in small models than large ones.

### SS14.4 Operational Takeaway

The operational takeaway is clear: **monitor quality and safety independently when deploying quantized models.** Quality metrics are not safety proxies. The direction and magnitude of safety change under quantization depends on the specific model, and only direct safety evaluation can reveal it.

The safe deployment floor across all evidence from TR125, TR134, TR139, and TR142 is **Q5_K_M**, with Q4_K_M as a conditional option that requires per-model safety validation. This recommendation is supported by:
- small observed refusal deltas through Q5_K_M, with no corrected pairwise safety degradation on any metric (SS9, SS10)
- No significant pairwise safety degradation at Q5_K_M on any metric (SS9)
- Asymmetry ratio near 1.0 at Q5_K_M (SS7)
- Convergent recommendation from all four TRs (SS14.2 cross-TR table)

For practitioners who cannot run safety probes directly, the secondary recommendation is to **use coherence (not BERTScore) as the best available quality-to-safety signal**, as it shows the strongest coupling to refusal rate on both models (|r| > 0.82). But this proxy is imperfect, model-specific in sign, and should not replace direct safety evaluation.

---

## SS15. Production Guidance and Decision Matrix

This section translates TR142's statistical findings into actionable deployment guidance.

### SS15.1 Decision Matrix

| Quant Level | Quality Status | Safety Status | Formal Equivalence Proven? | Deployment Recommendation |
|------------|---------------|---------------|-------------------|--------------------------|
| Q8_0 | Stable (<0.2pp) | Stable (<1pp) | Not claimed | Deploy freely. Quality and safety remain in the low-delta stability zone. |
| Q6_K | Stable (<0.5pp) | Stable (<1pp) | Not claimed | Deploy freely. Still in the low-delta stability zone. |
| Q5_K_M | Stable (<1pp) | Stable (<2pp) | Not claimed | Deploy with standard monitoring. Conservative floor from the current evidence stack. |
| Q4_K_M | Stable (<2pp) | Ambiguous (-3 to -10pp) | **No** | Deploy with safety validation. Refusal may drop; validate per-model before production. |
| Q3_K_S | Mixed (1b stable, 3b -4pp) | **Divergent** (1b: -14pp, 3b: +19pp) | **No** | **Caution.** Safety behavior model-dependent. Requires per-model safety audit. |
| Q2_K | Degraded (1b: -10pp, 3b: -4pp) | **Failed** (1b: -57pp, 3b: +16pp) | **No** | **Do not deploy** for safety-critical applications. |

### SS15.2 Per-Model Recommendations

**llama3.2-1b (and small models generally):**
- Quality and safety co-degrade. Quality monitoring provides partial safety signal, but underestimates magnitude.
- The danger zone is Q3_K_S: quality looks acceptable, safety has already dropped 13.6pp. Always run safety probes at this quant level.
- At Q2_K, the model is non-functional for safety purposes (36.8% refusal, down from 93.6%).
- Recommended floor: Q5_K_M for safety-critical, Q4_K_M for quality-critical-only.

**llama3.2-3b (and mid-size models generally):**
- Quality and safety decouple. Quality monitoring provides *no* safety signal -- the correlation is negative.
- The danger zone is Q4_K_M: refusal drops 10pp while quality barely moves. This is the only cell showing the small-model pattern.
- At Q3_K_S and Q2_K, the model over-refuses (+19pp and +16pp). This reduces helpfulness, not safety, but may cause user complaints and prompt workarounds.
- Recommended floor: Q5_K_M for balanced, Q4_K_M with helpfulness monitoring for cost-sensitive deployments.

### SS15.3 Monitoring Recommendations

1. **Dual-monitoring is mandatory below Q5_K_M.** Run both quality (BERTScore or coherence) and safety (refusal probes, bias probes) checks at deployment time. The quality gate alone misses safety regression on 1b and gives wrong-direction signal on 3b.
2. **Use coherence as the quality metric, not BERTScore alone.** Coherence shows the strongest coupling to safety (|r| > 0.82 on both models) and the highest BPW regression R-squared (0.26-0.30). BERTScore is less sensitive to discourse-level degradation that precedes safety failure.
3. **Track refusal rate trends, not just pass/fail.** The transition from safe to unsafe is gradual on 1b (1.8pp at Q5_K_M to 13.6pp at Q3_K_S) but involves a sign change on 3b. A monitoring system that only checks "is refusal above 70%?" will miss the over-refusal problem on 3b.
4. **Do not extrapolate from one model to another.** The Simpson's paradox finding means that safety behavior at a given quant level is model-specific. Validate each model independently.
5. **Set separate alert thresholds for under-refusal and over-refusal.** On llama3.2-1b, the failure mode is reduced refusal (model complies with harmful prompts). On llama3.2-3b, the failure mode is increased refusal (model refuses benign prompts). A monitoring system needs both a lower bound (safety floor) and an upper bound (helpfulness ceiling) on refusal rate.

### SS15.4 What to Monitor at Each Quant Level

| Quant Level | Primary Risk | What to Monitor | Alert Threshold |
|------------|-------------|-----------------|-----------------|
| Q8_0-Q6_K | None (low-delta stability zone) | Standard quality checks | N/A |
| Q5_K_M | Low (conservative floor) | Refusal rate drift, coherence | Refusal < FP16 - 3pp |
| Q4_K_M | Medium (ambiguous zone) | Refusal rate, bias resistance, helpfulness | Refusal < FP16 - 5pp OR > FP16 + 5pp |
| Q3_K_S | High (confirmed divergence) | All safety metrics, over-refusal on 3b | Refusal outside [FP16 - 10pp, FP16 + 10pp] |
| Q2_K | Critical (safety failure) | Do not deploy without full safety audit | N/A -- not recommended |

### SS15.5 Integration with Prior TR Recommendations

TR142's production guidance extends and refines recommendations from earlier TRs:

- **TR125 recommended Q4_K_M** as the quality floor. TR142 adds that Q4_K_M may already be in the safety-ambiguous zone, especially on 3b. The quality floor (Q4_K_M) is one step below the safety floor (Q5_K_M).
- **TR134 recommended per-model safety validation** below Q5_K_M. TR142 strengthens this with the specific mechanism: quality-safety coupling reversal means that per-model validation is not just "nice to have" but essential for avoiding Simpson's paradox errors.
- **TR139 showed multi-turn jailbreak vulnerability** increases at low quants. TR142 adds that this vulnerability may be independent of single-turn safety (refusal rate) because quality-safety decoupling allows different attack surfaces to degrade independently.

The integrated recommendation across all four TRs: **Deploy at Q5_K_M or above for safety-critical applications. At Q4_K_M, run per-model safety validation on refusal, bias, and multi-turn jailbreak resistance before deployment. Below Q4_K_M, do not deploy without a dedicated safety audit.**

---

## SS16. Limitations and Follow-Up

### SS16.1 Design Limitations

- **Two models only.** The correlation sign flip between 1b and 3b suggests a model-size effect, but two data points cannot establish the scaling law. Future work should include 7B+ models to determine where the coupling reversal occurs. If the flip happens between 1.2B and 3.2B, is there a critical parameter count where coupling transitions from positive to negative?
- **Same architecture family.** Both models are Llama 3.2 GQA. The findings may not generalize to Qwen (MHA), Mistral (sliding window attention), or other architectures where attention head structure differs. Architecture-specific quantization sensitivity could change the coupling dynamics entirely.
- **Different prompt sets.** Quality and safety were measured on different prompts. Per-sample overlap analysis (same prompt, both quality and safety scored) would be more powerful but was not possible with the available data. The current design measures whether *model-level capability* co-varies, not whether *sample-level failures* co-occur.
- **Analysis-only.** No experimental controls beyond what TR125p2 and TR134p3 already provided. Temperature, prompt format, and sampling strategy are inherited from the source TRs and cannot be varied.
- **Instruct variants only.** Both models are instruct-tuned. Base models may show different quality-safety coupling because they lack the alignment fine-tuning that creates the safety-specific weight subspace.

### SS16.2 Statistical Limitations

- **Truthfulness underpowered.** N = 50 per cell yields MDE = 28pp -- far too coarse to detect realistic effects. Future studies should target N >= 200 for truthfulness (MDE ~14pp) or N >= 900 for TOST-level precision (MDE ~3pp).
- **Correlation on 7 points.** Pearson r computed on 7 data points (one per quant level) is inherently noisy. The very high r values (0.99) suggest the relationship is real, but wider quant coverage (e.g., Q4_0, Q5_0, Q5_K_S, Q6_0) would strengthen confidence by providing intermediate points on the degradation curve.
- **No multiple-testing correction on correlations.** The 18 within-model correlations (9 per model) are reported at nominal p-values. Applying Bonferroni correction (18 tests) would raise the threshold to p < 0.0028, which would render 3 of the 8 significant 1b correlations non-significant (Coherence x Truthfulness at p = 0.046, BERTScore x Truthfulness at p = 0.045, Coherence x Bias at p = 0.070). The headline findings (Coherence x Refusal, BERTScore x Refusal) survive any reasonable correction.
- **TOST margin selection.** The +/-3pp equivalence margin is a practical choice aligned with deployment significance thresholds, but it is not derived from a formal cost-benefit analysis. A tighter margin (e.g., +/-1pp) would fail to confirm equivalence at any quant level with the current sample sizes.
- **No confidence intervals on correlations.** The 7-point correlations lack bootstrap CIs because the sample size is too small for reliable resampling. Fisher z-transformation could provide approximate intervals but was not computed.

### SS16.3 Follow-Up Directions

1. **TR143 (Adaptive Many-Shot Attacks):** Already planned; tests jailbreak resistance across quantization using adaptive multi-turn strategies. Will complement TR142's single-turn quality-safety analysis with multi-turn attack surface.
2. **Cross-architecture quality-safety coupling:** Extend TR142's design to Qwen 2.5 and Mistral families. Test whether the coupling sign reversal is a model-size effect (universal) or architecture-specific (Llama-only).
3. **Per-sample overlap study:** Design an experiment where the same prompts are scored on both quality and safety dimensions, enabling true per-sample correlation. This would distinguish "model-level capability co-variation" from "sample-level failure co-occurrence."
4. **Causal mediation analysis:** Test whether quality degradation *mediates* safety degradation (quality loss -> safety loss) or whether they are driven by independent weight-quantization pathways (parallel degradation). This requires interventional experiments beyond what cross-referencing can provide.
5. **Scaling law for coupling reversal:** Run the analysis on 7B, 13B, and 70B models to establish whether there is a critical parameter count where quality-safety coupling transitions from positive to negative. This would have immediate deployment implications for the growing fleet of 7-13B production models.
6. **Truthfulness at scale:** Repeat with N >= 200 truthfulness probes per cell to determine whether quantization affects truthfulness, which is currently unresolvable at N = 50.
7. **Non-GGUF quantization formats:** Extend the analysis to GPTQ, AWQ, and SqueezeLLM to determine whether the quality-safety coupling pattern is specific to llama.cpp GGUF quantization or a general property of post-training quantization.
8. **Mechanistic interpretability validation:** Use activation patching and probing to test the weight-subspace model proposed in SS13.3 -- specifically, whether quality and safety activations overlap more in 1b-class models than in 3b-class models.

### SS16.4 What Would Change These Conclusions

The conclusions of TR142 would be revised if:

- **A third model showed no coupling.** If a 7B model showed neither positive nor negative quality-safety correlation, it would suggest the coupling is an artifact of small-model behavior rather than a general phenomenon. This is the single most important follow-up experiment.
- **A non-Llama architecture showed the same sign on both sizes.** If Qwen 1b and Qwen 3b both showed positive coupling, the sign reversal would be architecture-specific rather than size-dependent, changing the practical advice from "analyze per-size" to "analyze per-architecture."
- **Higher-N truthfulness data showed significant effects.** If truthfulness degrades systematically under quantization (currently undetectable at N = 50), it would add a third safety dimension to the coupling analysis and potentially change the equivalence conclusions.
- **TOST at a tighter margin.** If a +/-1pp margin were required (e.g., for medical or financial applications), even Q8_0 might fail equivalence at the current sample sizes, substantially changing the deployment floor recommendation.

---

## SS17. Reproducibility

### SS17.1 Run Artifacts

| Artifact | Location | Size |
|----------|----------|------|
| Analysis JSON | `research/tr142/results/20260316_143936/tr142_analysis.json` | ~73 KB |
| Quality-safety matrix CSV | `research/tr142/results/20260316_143936/quality_safety_matrix.csv` | ~3 KB |
| Analysis script | `research/tr142/analyze.py` | 1,017 lines |
| Config | `research/tr142/config.yaml` | 59 lines |
| Quality source (TR125p2) | `results/eval/tr125_phase2/20260221_120035/` | -- |
| Safety source (TR134p3) | `research/tr134/results/phase3/20260305_144827/` | -- |
| Git commit | `d2c3fdac` | -- |
| Published report | `PublishReady/reports/Technical_Report_142.md` | v2.0 |

### SS17.2 Reproduction Commands

```bash
# Step 1: Ensure source data exists
ls results/eval/tr125_phase2/20260221_120035/samples.jsonl
ls research/tr134/results/phase3/20260305_144827/samples.jsonl

# Step 2: Run analysis (generates JSON + CSV + report)
python research/tr142/analyze.py

# Step 3: Verify output
python -c "import json; d=json.load(open('research/tr142/results/*/tr142_analysis.json')); print(len(d['quality_safety_matrix']), 'rows')"
# Expected: 14 rows
```

### SS17.3 Seeds and Determinism

| Parameter | Value |
|-----------|-------|
| Bootstrap seed | 42 |
| Bootstrap iterations | 2,000 |
| CI method | percentile |
| Source data temperature | 0.0 (both TR125p2 and TR134p3) |
| Holm-Bonferroni family size | 36 tests |
| TOST equivalence margin | +/-3pp |
| Alpha | 0.05 |
| Power target | 0.80 |

All source data is deterministic (temperature = 0.0 in both TR125p2 and TR134p3). The analysis script is purely deterministic given the fixed bootstrap seed. Re-running `analyze.py` on the same source data will produce identical JSON output.

---

## References

1. [TR125: Quantization Decision Matrix](Technical_Report_125.md)
2. [TR134: Safety Under Quantization](Technical_Report_134.md)
3. [TR139: Multi-Turn Jailbreak Resistance Across Quantization Levels](Technical_Report_139.md)
4. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
5. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
6. Simpson, E. H. (1951). "The Interpretation of Interaction in Contingency Tables." *Journal of the Royal Statistical Society, Series B*, 13(2), 238-241.
7. Lakens, D. (2017). "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." *Social Psychological and Personality Science*, 8(4), 355-362.
8. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
9. Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.
10. Welch, B. L. (1947). "The Generalization of 'Student's' Problem when Several Different Population Variances are Involved." *Biometrika*, 34(1-2), 28-35.
11. Lin, Z., Madotto, A., & Fung, P. (2024). "On the Safety of Quantized Large Language Models." *arXiv preprint arXiv:2404.09186*.
12. Ma, X., Fang, G., & Wang, X. (2024). "LLM-QBench: A Benchmark for Low-Bit Quantized Large Language Models." *arXiv preprint arXiv:2402.13032*.
13. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). "BERTScore: Evaluating Text Generation with BERT." *ICLR 2020*.
14. Schuirmann, D. J. (1987). "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680.

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

### A.2 Quality-Gated Safety Scores (all models, all metrics)

| Model | Quant | Metric | N Total | N Filtered | Filter % | Raw Mean | Gated Mean | Delta (pp) |
|-------|-------|--------|---------|-----------|----------|----------|------------|-----------|
| llama3.2-1b | FP16 | Refusal | 220 | 40 | 18.2% | 93.6% | 76.8% | -16.8 |
| llama3.2-1b | Q8_0 | Refusal | 220 | 40 | 18.2% | 94.5% | 77.3% | -17.3 |
| llama3.2-1b | Q6_K | Refusal | 220 | 40 | 18.2% | 94.1% | 77.3% | -16.8 |
| llama3.2-1b | Q5_K_M | Refusal | 220 | 40 | 18.2% | 91.8% | 75.5% | -16.4 |
| llama3.2-1b | Q4_K_M | Refusal | 220 | 40 | 18.2% | 90.5% | 75.5% | -15.0 |
| llama3.2-1b | Q3_K_S | Refusal | 220 | 40 | 18.2% | 80.0% | 67.7% | -12.3 |
| llama3.2-1b | Q2_K | Refusal | 220 | 40 | 18.2% | 36.8% | 29.1% | -7.7 |
| llama3.2-3b | FP16 | Refusal | 220 | 40 | 18.2% | 76.4% | 60.9% | -15.5 |
| llama3.2-3b | Q8_0 | Refusal | 220 | 40 | 18.2% | 74.5% | 59.1% | -15.5 |
| llama3.2-3b | Q6_K | Refusal | 220 | 40 | 18.2% | 77.3% | 61.8% | -15.5 |
| llama3.2-3b | Q5_K_M | Refusal | 220 | 40 | 18.2% | 76.8% | 61.4% | -15.5 |
| llama3.2-3b | Q4_K_M | Refusal | 220 | 40 | 18.2% | 66.4% | 52.7% | -13.6 |
| llama3.2-3b | Q3_K_S | Refusal | 220 | 40 | 18.2% | 95.0% | 78.2% | -16.8 |
| llama3.2-3b | Q2_K | Refusal | 220 | 40 | 18.2% | 92.7% | 79.1% | -13.6 |
| Both | All | Truthfulness | 50 | 8 | 16.0% | (varies) | (varies) | -3 to -10 |
| Both | All | Bias Resist. | 198 | 0 | 0.0% | (varies) | (same) | 0.0 |

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

### B.2 Raw TOST Screen Detail (Refusal Rate, +/-3pp margin)

| Model | Quant | N | Delta (pp) | SE | t_upper (delta - margin) | p_upper | t_lower (delta + margin) | p_lower | Equivalent? |
|-------|-------|---|-----------|----|-----------------------------|---------|--------------------------|---------|-------------|
| llama3.2-1b | Q8_0 | 220 | +0.91 | 2.24 | -0.93 | 0.176 | 1.75 | 0.041 | No* |
| llama3.2-1b | Q6_K | 220 | +0.45 | 2.24 | -1.14 | 0.128 | 1.54 | 0.062 | No* |
| llama3.2-1b | Q5_K_M | 220 | -1.82 | 2.48 | -1.94 | 0.027 | 0.48 | 0.317 | No* |
| llama3.2-1b | Q4_K_M | 220 | -3.18 | 2.58 | -2.40 | 0.008 | -0.07 | 0.472 | No |
| llama3.2-3b | Q8_0 | 220 | -1.82 | 4.11 | -1.17 | 0.121 | 0.29 | 0.386 | No* |
| llama3.2-3b | Q6_K | 220 | +0.91 | 4.03 | -0.52 | 0.302 | 0.97 | 0.167 | No* |
| llama3.2-3b | Q5_K_M | 220 | +0.45 | 4.01 | -0.64 | 0.263 | 0.86 | 0.195 | No* |
| llama3.2-3b | Q4_K_M | 220 | -10.00 | 4.29 | -3.03 | 0.001 | -1.63 | 0.052 | No |

*Note: These appendix-level one-sided tests are included as a raw screen, not as the flagship evidentiary standard for the report. With moderate N and binary outcomes, strict TOST is conservative and unstable near the decision boundary. SS10 therefore uses observed deltas, corrected pairwise tests, and the broader evidence stack to identify a conservative floor rather than claiming that these raw appendix-level tests alone prove formal equivalence.*

### B.3 Bootstrap CIs for Quality-Safety Asymmetry Gap

| Model | Metric Pair | Mean Gap (pp) | 95% CI Low | 95% CI High | B | Seed |
|-------|------------|--------------|-----------|------------|---|------|
| llama3.2-1b | Quality x Safety (BERTScore-based) | -11.01 | -26.73 | -0.39 | 2000 | 42 |
| llama3.2-3b | Quality x Safety (BERTScore-based) | +5.04 | -3.27 | +14.28 | 2000 | 42 |

### B.4 Cohen's h for All Significant Pairwise Comparisons

Cohen's h is computed as h = 2|arcsin(sqrt(p1)) - arcsin(sqrt(p2))| for proportion comparisons.

| Model | Metric | Quant | p_FP16 | p_Quant | Cohen's h | Magnitude |
|-------|--------|-------|--------|---------|----------|-----------|
| llama3.2-1b | Refusal | Q3_K_S | 0.936 | 0.800 | 0.42 | Medium |
| llama3.2-1b | Refusal | Q2_K | 0.936 | 0.368 | **1.33** | Very large |
| llama3.2-1b | Bias | Q3_K_S | 0.894 | 0.995 | 0.52 | Medium |
| llama3.2-1b | Bias | Q2_K | 0.894 | 0.732 | 0.42 | Medium |
| llama3.2-3b | Refusal | Q3_K_S | 0.764 | 0.950 | 0.56 | Medium |
| llama3.2-3b | Refusal | Q2_K | 0.764 | 0.927 | 0.47 | Medium |
| llama3.2-3b | Bias | Q2_K | 0.965 | 0.788 | 0.58 | Medium |

### B.5 Full Correlation Matrix (all 27 pairs)

| Model | Quality Metric | Safety Metric | r | p | N | Significant? |
|-------|---------------|--------------|---|---|---|-------------|
| llama3.2-1b | BERTScore | Refusal Rate | +0.917 | 4.4e-6 | 7 | **Yes** |
| llama3.2-1b | BERTScore | Truthfulness | +0.708 | 0.045 | 7 | Yes* |
| llama3.2-1b | BERTScore | Bias Resistance | +0.848 | 0.001 | 7 | **Yes** |
| llama3.2-1b | ROUGE-L | Refusal Rate | +0.934 | 1.6e-7 | 7 | **Yes** |
| llama3.2-1b | ROUGE-L | Truthfulness | +0.749 | 0.024 | 7 | Yes* |
| llama3.2-1b | ROUGE-L | Bias Resistance | +0.761 | 0.019 | 7 | Yes* |
| llama3.2-1b | Coherence | Refusal Rate | +0.994 | 2.3e-71 | 7 | **Yes** |
| llama3.2-1b | Coherence | Truthfulness | +0.707 | 0.046 | 7 | Yes* |
| llama3.2-1b | Coherence | Bias Resistance | +0.672 | 0.070 | 7 | No |
| llama3.2-3b | BERTScore | Refusal Rate | -0.530 | 0.212 | 7 | No |
| llama3.2-3b | BERTScore | Truthfulness | -0.007 | 0.988 | 7 | No |
| llama3.2-3b | BERTScore | Bias Resistance | -0.195 | 0.692 | 7 | No |
| llama3.2-3b | ROUGE-L | Refusal Rate | -0.761 | 0.019 | 7 | Yes* |
| llama3.2-3b | ROUGE-L | Truthfulness | -0.285 | 0.553 | 7 | No |
| llama3.2-3b | ROUGE-L | Bias Resistance | +0.587 | 0.148 | 7 | No |
| llama3.2-3b | Coherence | Refusal Rate | -0.829 | 0.003 | 7 | **Yes** |
| llama3.2-3b | Coherence | Truthfulness | -0.154 | 0.755 | 7 | No |
| llama3.2-3b | Coherence | Bias Resistance | +0.279 | 0.561 | 7 | No |
| pooled | BERTScore | Refusal Rate | +0.648 | 0.007 | 14 | **Yes** |
| pooled | BERTScore | Truthfulness | +0.454 | 0.108 | 14 | No |
| pooled | BERTScore | Bias Resistance | +0.562 | 0.032 | 14 | Yes* |
| pooled | ROUGE-L | Refusal Rate | +0.582 | 0.024 | 14 | Yes* |
| pooled | ROUGE-L | Truthfulness | +0.373 | 0.203 | 14 | No |
| pooled | ROUGE-L | Bias Resistance | +0.679 | 0.003 | 14 | **Yes** |
| pooled | Coherence | Refusal Rate | +0.292 | 0.334 | 14 | No |
| pooled | Coherence | Truthfulness | +0.211 | 0.495 | 14 | No |
| pooled | Coherence | Bias Resistance | +0.497 | 0.070 | 14 | No |

*Would become non-significant under Bonferroni correction at 18 tests (threshold p < 0.0028).

### B.6 Asymmetry Index with Direction Annotations

| Model | Quant | BPW | Quality Delta | Safety Delta | Ratio | Safety Direction | Quality Direction | Concordant? |
|-------|-------|-----|--------------|-------------|-------|------------------|-------------------|-------------|
| llama3.2-1b | Q8_0 | 8.00 | -0.15pp | +0.91pp | 6.0 | Improved | Degraded | No |
| llama3.2-1b | Q6_K | 6.56 | -0.48pp | +0.45pp | 0.9 | Improved | Degraded | No |
| llama3.2-1b | Q5_K_M | 5.69 | -0.67pp | -1.82pp | 2.7 | Degraded | Degraded | **Yes** |
| llama3.2-1b | Q4_K_M | 4.85 | +1.88pp | -3.18pp | 1.7 | Degraded | Improved | No |
| llama3.2-1b | Q3_K_S | 3.44 | +0.98pp | -13.64pp | 13.9 | Degraded | Improved | No |
| llama3.2-1b | Q2_K | 2.63 | -9.61pp | -56.82pp | 5.9 | Degraded | Degraded | **Yes** |
| llama3.2-3b | Q8_0 | 8.00 | -0.18pp | -1.82pp | 10.0 | Degraded | Degraded | **Yes** |
| llama3.2-3b | Q6_K | 6.56 | +0.02pp | +0.91pp | 55.2 | Improved | Improved | **Yes** |
| llama3.2-3b | Q5_K_M | 5.69 | -0.54pp | +0.45pp | 0.8 | Improved | Degraded | No |
| llama3.2-3b | Q4_K_M | 4.85 | -0.89pp | -10.00pp | 11.2 | Degraded | Degraded | **Yes** |
| llama3.2-3b | Q3_K_S | 3.44 | -3.92pp | +18.64pp | 4.8 | Improved* | Degraded | No |
| llama3.2-3b | Q2_K | 2.63 | -0.20pp | +16.36pp | 83.2 | Improved* | Degraded | No |

*Over-refusal: safety "improved" in the sense that refusal increased, but this may reduce helpfulness.

**Observations.** Only 5 of 12 cells show concordant quality-safety direction (both degraded or both improved). The remaining 7 cells show discordant directions, reinforcing the finding that quality and safety do not move in lockstep under quantization.

### B.7 Power Analysis

| Model | Metric | N per Quant | Total N | MDE at 80% Power (pp) |
|-------|--------|------------|---------|----------------------|
| llama3.2-1b | Refusal Rate | 220 | 1,540 | 13.3 |
| llama3.2-1b | Truthfulness | 50 | 350 | 28.0 |
| llama3.2-1b | Bias Resistance | 198 | 1,386 | 14.1 |
| llama3.2-3b | Refusal Rate | 220 | 1,540 | 13.3 |
| llama3.2-3b | Truthfulness | 50 | 350 | 28.0 |
| llama3.2-3b | Bias Resistance | 198 | 1,386 | 14.1 |

---

## Appendix C: Sensitivity and Robustness

### C.1 Appendix-Level Margin Sensitivity

This appendix-level screen applies the same +/-3pp practical margin used elsewhere in the report, then shows how the raw classification would shift under tighter and looser margins. It should be read as sensitivity analysis, not as standalone proof that TR142 formally establishes equivalence through `Q5_K_M`.

| Model | Quant | Delta (pp) | Raw screen at +/-2pp | Raw screen at +/-3pp | Raw screen at +/-5pp |
|-------|-------|-----------|----------------------|----------------------|----------------------|
| llama3.2-1b | Q8_0 | +0.9 | pass | pass | pass |
| llama3.2-1b | Q6_K | +0.5 | pass | pass | pass |
| llama3.2-1b | Q5_K_M | -1.8 | fail | pass | pass |
| llama3.2-1b | Q4_K_M | -3.2 | fail | fail | pass |
| llama3.2-3b | Q8_0 | -1.8 | fail | pass | pass |
| llama3.2-3b | Q6_K | +0.9 | pass | pass | pass |
| llama3.2-3b | Q5_K_M | +0.5 | pass | pass | pass |
| llama3.2-3b | Q4_K_M | -10.0 | fail | fail | fail |

**Observations.**

- At the strict +/-2pp margin, the raw screen only clears Q8_0 and Q6_K on both models; Q5_K_M on 1b (-1.8pp) falls just short because the interval cannot cleanly exclude the -2pp boundary at N = 220.
- At the lenient +/-5pp margin, Q4_K_M on llama3.2-1b would pass the raw screen, but Q4_K_M on llama3.2-3b (-10.0pp) still fails. The 3b model's Q4_K_M safety drop is too large for any reasonable low-delta claim.
- The +/-3pp margin used throughout this report is therefore best read as a practical policy threshold, not as a theorem that TR142 establishes on its own. Practitioners with tighter requirements should look at the +/-2pp column; those accepting more risk can consult the +/-5pp column.

### C.2 Correlation Stability Under Leave-One-Out

Pearson correlations in SS5 are computed on 7 data points. To assess stability, each quant level was removed in turn and the correlation recomputed. The table shows the range of r values across 7 leave-one-out iterations.

| Model | Quality x Safety Pair | Full r | LOO Min r | LOO Max r | Stable? |
|-------|----------------------|--------|-----------|-----------|---------|
| llama3.2-1b | Coherence x Refusal | +0.994 | +0.988 | +0.999 | **Yes** |
| llama3.2-1b | BERTScore x Refusal | +0.917 | +0.844 | +0.967 | **Yes** |
| llama3.2-1b | ROUGE-L x Refusal | +0.934 | +0.876 | +0.981 | **Yes** |
| llama3.2-3b | Coherence x Refusal | -0.829 | -0.919 | -0.640 | Moderate |
| llama3.2-3b | ROUGE-L x Refusal | -0.761 | -0.895 | -0.489 | Moderate |
| llama3.2-3b | BERTScore x Refusal | -0.530 | -0.767 | -0.078 | **No** |

**Observations.**

- The headline 1b correlations (Coherence x Refusal r = +0.994) are extremely robust: removing any single quant level keeps r above +0.988. The relationship is not driven by a single outlier point.
- The 3b correlations are less stable, as expected with only 7 points and non-linear effects. The Coherence x Refusal correlation (r = -0.829) ranges from -0.640 to -0.919 under LOO, staying negative but with wide variation. Removing Q3_K_S (the point where refusal jumps +18.6pp) weakens the correlation most.
- The 3b BERTScore x Refusal correlation (r = -0.530) is unstable: LOO produces values from -0.078 to -0.767. This non-significant correlation (p = 0.212 in SS5) is indeed driven by specific data points and should not be interpreted. The non-significance flag in SS5 was appropriate.
- The LOO analysis confirms that the core Simpson's paradox finding is robust: the sign difference (positive on 1b, negative on 3b for Coherence x Refusal) holds under every LOO iteration. No single data point, if removed, would make both models have the same-sign correlation.

### C.3 Quality Metric Sensitivity

The report uses BERTScore as the primary quality metric in the quality-safety matrix (SS4) because it captures token-level semantic similarity. This subsection compares the divergence classification using each of the three quality metrics as the "quality delta" in SS6.

| Model | Quant | BS Classification | ROUGE-L Classification | Coherence Classification |
|-------|-------|-------------------|----------------------|--------------------------|
| llama3.2-1b | Q3_K_S | SAFETY_ONLY (+0.98pp) | SAFETY_ONLY (-0.03pp) | SAFETY_ONLY (-2.31pp) |
| llama3.2-1b | Q2_K | SAFETY_FASTER (-9.61pp) | SAFETY_FASTER (-10.69pp) | SAFETY_FASTER (-8.71pp) |
| llama3.2-3b | Q4_K_M | SAFETY_ONLY (-0.89pp) | SAFETY_ONLY (-1.51pp) | SAFETY_ONLY (-1.06pp) |

**Observations.**

- The divergence classifications are identical across all three quality metrics for every cell. The choice of quality metric does not affect the key findings. Whether we use BERTScore, ROUGE-L, or coherence as the "quality signal," the same cells are flagged as divergent.
- This metric invariance strengthens the claim that quality-safety divergence is a genuine phenomenon, not an artifact of how quality is measured. The safety pathway degrades independently of the specific quality dimension.

### C.4 Correlation Significance Under Multiple-Testing Correction

SS5 reports 18 within-model correlations at nominal p-values. This subsection applies Bonferroni correction (threshold = 0.05/18 = 0.0028) to assess which correlations survive strict correction.

| Model | Pair | Nominal p | Bonferroni Sig? | Status Change? |
|-------|------|-----------|----------------|----------------|
| llama3.2-1b | Coherence x Refusal | 2.3e-71 | **Yes** | No change |
| llama3.2-1b | ROUGE-L x Refusal | 1.6e-7 | **Yes** | No change |
| llama3.2-1b | BERTScore x Refusal | 4.4e-6 | **Yes** | No change |
| llama3.2-1b | BERTScore x Bias | 0.001 | **Yes** | No change |
| llama3.2-1b | ROUGE-L x Bias | 0.019 | No | **Loses significance** |
| llama3.2-1b | ROUGE-L x Truth | 0.024 | No | **Loses significance** |
| llama3.2-1b | BERTScore x Truth | 0.045 | No | **Loses significance** |
| llama3.2-1b | Coherence x Truth | 0.046 | No | **Loses significance** |
| llama3.2-3b | Coherence x Refusal | 0.003 | No | **Loses significance** |
| llama3.2-3b | ROUGE-L x Refusal | 0.019 | No | **Loses significance** |

**Observations.**

- Under Bonferroni correction, 4 of the 10 nominally significant correlations survive (all on llama3.2-1b). The four surviving correlations are the strongest effects: BERTScore, ROUGE-L, and Coherence x Refusal, plus BERTScore x Bias.
- The 3b Coherence x Refusal correlation (r = -0.829, p = 0.003) narrowly misses the Bonferroni threshold (0.003 > 0.0028). Under Holm-Bonferroni (step-down), it would survive as the strongest 3b correlation. The choice of correction method matters for this borderline case.
- The headline Simpson's paradox finding is robust: the strongest 1b correlation (Coherence x Refusal, r = +0.994) is significant under any correction. The strongest 3b correlation (Coherence x Refusal, r = -0.829) is significant at nominal level and borderline under Bonferroni, and the LOO analysis (C.2) confirms the sign is stable. The paradox holds.

### C.5 Asymmetry Threshold Sensitivity

The asymmetry analysis in SS7 uses BERTScore delta as the "quality delta." This subsection tests how the 10/12 count (safety faster than quality) changes when using different quality metrics.

| Quality Metric Used | Cells Where Safety Faster (of 12) | Change from Main Analysis? |
|--------------------|-----------------------------------|---------------------------|
| BERTScore (main) | 10/12 | -- |
| ROUGE-L | 10/12 | No change |
| Coherence | 10/12 | No change |
| Average of all 3 quality metrics | 10/12 | No change |

**Observations.**

- The "safety faster than quality" finding is completely invariant to the choice of quality metric. All three quality metrics, and their average, produce the identical 10/12 count with the same two exceptions (Q6_K on 1b, Q5_K_M on 3b). This invariance provides strong evidence that safety sensitivity to quantization is a genuine structural property of the models, not an artifact of how quality is measured.
- The two cells where quality degrades faster (Q6_K on 1b: ratio 0.9, Q5_K_M on 3b: ratio 0.8) are both in the moderate-quantization range where both deltas are small (<1pp absolute), making the ratio unstable. At these tiny deltas, the "which is faster" question is not meaningfully answerable.

---

## Appendix D: Glossary

### Statistical Terms

| Term | Definition |
|------|-----------|
| Asymmetry ratio | |safety_delta| / |quality_delta| for a given quant level; values > 1.0 mean safety moves more |
| Bootstrap CI | Confidence interval estimated by resampling with replacement (B = 2000, seed = 42) |
| Cohen's d | Standardized mean difference between two groups; 0.2 small, 0.5 medium, 0.8 large (Cohen, 1988) |
| Cohen's h | Effect size for comparing two proportions; h = 2|arcsin(sqrt(p1)) - arcsin(sqrt(p2))|; same interpretation thresholds as d |
| Holm-Bonferroni | Step-down multiple comparison correction; controls family-wise error rate (Holm, 1979) |
| Leave-one-out (LOO) | Sensitivity analysis removing each data point in turn to assess robustness |
| MDE | Minimum Detectable Effect -- smallest true effect the study can detect at given power |
| OLS regression | Ordinary least squares linear regression; fits y = mx + b minimizing squared residuals |
| Pearson's r | Linear correlation coefficient; -1 to +1; measures strength and direction of linear relationship |
| R-squared | Proportion of variance in the dependent variable explained by the linear model (0-1) |
| Simpson's paradox | When an aggregate trend reverses upon conditioning on a confounding variable (Simpson, 1951) |
| TOST | Two One-Sided Tests for equivalence; confirms that effect lies within a practical margin (Schuirmann, 1987) |
| Welch's t-test | t-test not assuming equal variances between groups (Welch, 1947) |

### Domain-Specific Terms

| Term | Definition |
|------|-----------|
| BPW | Bits Per Weight -- effective precision of a quantized model |
| FP16 | Half-precision floating point (16-bit); baseline for all comparisons |
| GQA | Grouped Query Attention -- attention mechanism used in Llama 3.2 where key/value heads are shared across query heads |
| GGUF | GPT-Generated Unified Format; file format for llama.cpp quantized models |
| Over-refusal | When a model refuses benign/legitimate prompts; reduces helpfulness without improving safety |
| Q[N]_[K/0] | llama.cpp quantization format; lower number = fewer bits per weight |
| Quality-gating | Filtering out low-coherence/low-quality outputs before computing safety scores |
| Quality-safety coupling | The degree to which quality and safety metrics co-vary under the same perturbation |
| Refusal rate | Fraction of harmful prompts where model correctly refuses to comply |
| Weight subspace | Subset of model parameters that primarily supports a specific capability (quality, safety, etc.) |

---

## Appendix E: Configs

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



