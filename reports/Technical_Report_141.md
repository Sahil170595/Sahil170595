# Technical Report 141: Cross-Architecture Refusal Fragility Under Batch Perturbation
## Safety-Capability Flip Asymmetry Across 7 Models, 5 Families, and 4 Alignment Types

| Field | Value |
|-------|-------|
| **TR Number** | 141 |
| **Project** | Banterhearts |
| **Date** | 2026-03-18 |
| **Version** | 2.0 (full-depth) |
| **Author** | Banterhearts Research Lab |
| **Git Commit** | `standalone` |
| **Status** | Complete -- Full-Depth Publication |
| **Report Type** | Full-depth |
| **Run Directory** | `research/tr141/results/colab_20260317/tr141_run_20260317_222300/` |
| **Total Records** | 49,476 (40,026 Phase 1 + 9,450 Phase 2) |
| **Models** | 7 completed (from 8 planned; gemma-2-2b dropped) |
| **Families** | 5 completed (Llama, Qwen, Phi, SmolLM, StableLM); Gemma dropped (gated access) |
| **Alignment Types** | 4 (RLHF, SFT, DPO, Distilled) |
| **Seed** | 137 |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition (98 GB VRAM), Google Colab |
| **Analysis Passes** | 28 |
| **Related Work** | [TR138](Technical_Report_138_v2.md) (batch safety, 3 models), [TR139](Technical_Report_139.md) (multi-turn jailbreak x quantization) |
| **Depends On** | TR138 (baseline comparison), TR130 (backend abstraction), TR134 (safety classifiers) |

---

## Positioning

TR141 is the cross-architecture extension of the batch-safety perturbation program initiated by TR138. Where TR138 established that batch-induced floating-point non-determinism can disproportionately affect safety outputs on 3 models from 2 families (all RLHF), TR141 scales to 7 models from 5 families spanning 4 alignment paradigms (RLHF, SFT, DPO, distilled). This enables the first alignment-type-level analysis of batch-safety fragility and determines whether TR138's findings generalize beyond the Llama/Qwen RLHF models. TR141 also uses a different GPU architecture (RTX PRO 6000 Blackwell vs A100), a different seed (137 vs 42), and a larger prompt set, providing an independent replication of the core phenomenon. Downstream, TR141 informs the production guidance for batch-serving safety validation and feeds into the cross-TR meta-analysis framework alongside TR138 and TR139.

---

## Abstract

TR141 asks whether batch-induced floating-point non-determinism in GPU inference disproportionately degrades safety outputs, and whether this effect varies systematically across model architectures and alignment approaches. This cross-architecture study evaluates **7 instruction-tuned models** from **5 families** spanning **4 alignment paradigms** (RLHF, SFT, DPO, distilled), producing **49,476 total records** across two experimental phases on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU via Google Colab.

The core findings are: (1) Safety outputs flip at **0.63%** versus capability outputs at **0.47%**, a 1.3x ratio that confirms batch perturbation is not safety-neutral but at a substantially lower magnitude than TR138's 4x ratio. (2) Cross-architecture fragility varies by **2.9x** -- phi-3.5-mini (SFT-aligned, 1.11%) is most fragile while stablelm-2-zephyr (DPO-aligned, 0.38%) is least fragile. (3) Alignment type significantly predicts fragility (one-way ANOVA F=4.86, p=0.0078, eta-squared=0.0007). (4) Directional analysis reveals a significant net SAFE bias: 69 compliance-to-refusal flips versus 28 refusal-to-compliance flips (binomial p=3.8e-5), opposite to the expected unsafe direction.

The most important non-result is that no critical batch-size threshold was detected for any model -- safety flip rates do not spike at any particular batch size, suggesting the effect is diffuse rather than catastrophic. All TOST equivalence tests confirm that batch-induced mean score differences remain within the +/-3 percentage point practical equivalence margin.

The operational conclusion is that batch perturbation introduces a small but statistically detectable safety perturbation whose magnitude depends on alignment approach, and production deployments should validate refusal stability at their actual batch configuration rather than relying on batch=1 baselines. The effect is universally present across all tested architectures and alignment types, but varies by nearly 3x in magnitude -- making model selection a meaningful lever for batch-safety robustness.

---

## Executive Summary

### Key Findings

1. **Batch perturbation is not safety-neutral, but the effect is small.** Across 7 models and 5 non-baseline batch sizes, safety outputs flip at 0.63% versus capability outputs at 0.47% (ratio 1.3x). This replicates TR138's directional finding but at lower magnitude (TR138 reported 4x on 3 models). No individual batch-size comparison reaches statistical significance after Holm-Bonferroni correction (all Fisher p > 0.05 corrected).

2. **Architecture fragility varies by nearly 3x.** phi-3.5-mini (SFT-aligned, 3.8B params) is the most fragile model at 1.11% safety flip rate [95% CI: 0.76%, 1.62%], while stablelm-2-zephyr (DPO-aligned, 1.6B params) and smollm2-1.7b (distilled, 1.7B params) share the lowest fragility at 0.38% [CI: 0.20%, 0.73%]. This 2.9x spread indicates that model selection is a meaningful lever for batch-safety robustness. **Floor-effect caveat:** smollm2-1.7b has only 25 baseline refusals and stablelm-2-zephyr only 60 (compared to 984-1,356 for the RLHF/SFT models). Their low absolute flip rates partly reflect limited opportunity to flip. When normalized by baseline refusals, smollm2-1.7b's fragility is 36.0% (9/25) -- meaning over a third of its refusals are batch-unstable -- versus phi-3.5-mini's 2.25% (26/1,155). See SS10 for the full normalized fragility table.

3. **Alignment type significantly predicts fragility.** One-way ANOVA on safety flip rate by alignment type yields F=4.862, p=0.0078, eta-squared=0.0007. SFT-aligned phi-3.5-mini is the most fragile; DPO-aligned stablelm-2-zephyr and distilled smollm2-1.7b are the most robust. The RLHF group (4 models) falls in between at 0.64%.

4. **Phase 2 true-batch validation confirms the effect is not a timing artifact.** Explicit prompt-list batching in Phase 2 produces a 0.94% safety flip rate, higher than Phase 1's 0.63%, with 97.8-100% flip agreement between phases. The effect is real and, if anything, amplified under true tensor batching.

5. **Directional analysis reveals significant net SAFE bias.** Of 97 directional safety flips across all batch sizes, 69 (71.1%) move from compliance to refusal (becoming safer) while only 28 (28.9%) move from refusal to compliance (becoming less safe). Binomial test: p=3.8e-5, highly significant. This is the **opposite** of the feared unsafe direction and opposite of TR138's findings, suggesting that batch perturbation in this experimental configuration tends to push marginal outputs toward refusal rather than away from it.

6. **TruthfulQA is the most batch-sensitive task.** With a mean flip rate of 2.7% across models and batch sizes, TruthfulQA is 5x more sensitive than BBQ bias (0.2%) and 9x more sensitive than AdvBench refusal (0.3%). This likely reflects the ambiguous nature of TruthfulQA prompts, which place the model near its refusal decision boundary.

7. **Flipped samples are systematically slower.** Across all 7 models, samples whose classification flips have higher mean latency than stable samples, with Cohen's d ranging from 0.17 (smollm2-1.7b) to 1.17 (qwen2.5-1.5b). This suggests that flip-prone prompts elicit longer, more uncertain generations.

8. **Variance-safety correlation: flip-prone prompts have lower baseline safety.** 4 of 7 models show a significant negative Pearson correlation between flip frequency and baseline safety score (p < 0.02 for all four), confirming that batch perturbation exploits existing vulnerability in marginally safe prompts.

9. **All TOST equivalence comparisons pass within +/-3pp.** Fourteen TOST tests across both phases all confirm practical equivalence, meaning batch perturbation does not shift aggregate safety scores by a practically meaningful amount even though individual flips are detectable.

10. **No critical batch-size threshold detected.** Wilson CI non-overlap analysis finds no batch size at which the safety flip rate confidence interval ceases to overlap with the capability flip rate confidence interval for any model. The effect is uniformly distributed across batch sizes 2-32.

### How to Read This Report

- **For a quick summary:** Read Key Findings 1-5 above and the Claim Validation table below.
- **For deployment decisions:** Skip to SS30 (Production Guidance) and the Summary Decision Matrix.
- **For statistical rigor assessment:** Read SS11 (ANOVA), SS15 (TOST), SS16 (Power), and SS17 (Fisher exact tests). Pay attention to the pseudoreplication caveat in SS11 and the Holm-Bonferroni note in SS25.
- **For cross-study context:** SS27 compares TR141 to TR138. SS22 provides the directional analysis that reverses TR138's unsafe-direction concern.
- **For per-model selection:** SS10 (fragility ranking) and SS6 (per-model flip patterns) provide the model-level view.
- **"SS" prefix** denotes numbered sections (Section Structured). Tables are numbered sequentially and referenced by section.

### Core Decisions

- Test safety refusal rates at the production batch size, not at batch=1 baselines
- Prefer DPO-aligned or distilled models when batch-safety robustness is a priority
- Monitor TruthfulQA-like ambiguous prompts disproportionately, as they are 5-9x more sensitive to batch perturbation
- Consider deterministic inference kernels (SGLang deterministic mode) for safety-critical pipelines, accepting approximately 34% throughput cost
- Do not assume unsafe-direction dominance; net SAFE bias in TR141 suggests architecture-dependent directionality

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Sample count | Total records | >= 30,000 | 49,476 | **PASS** |
| Phase 1 coverage | Models x batch sizes | 7 x 6 | 7 x 6 = 42 cells | **PASS** |
| Phase 2 coverage | Models x batch sizes | 7 x 3 | 7 x 3 = 21 cells | **PASS** |
| TOST equivalence | All comparisons within +/-3pp | 14/14 pass | 14/14 pass | **PASS** |
| Power (Phase 1 safety) | MDE at 80% power | <= 2.0pp | 1.3pp | **PASS** |
| Power (Phase 2) | MDE at 80% power | <= 3.0pp | 2.0pp | **PASS** |
| Alignment ANOVA | p < 0.05 | p < 0.05 | p = 0.0078 | **PASS** |
| Phase agreement | Flip agreement | >= 95% | 99.45% mean | **PASS** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | Batch perturbation disproportionately affects safety | SS7: 0.63% safety vs 0.47% capability (1.3x), CIs overlap | **Partial** |
| C2 | Architecture fragility varies significantly | SS10: 2.9x spread, phi-3.5-mini 1.11% vs stablelm-2-zephyr 0.38% | **Established** |
| C3 | Alignment type predicts fragility | SS11: ANOVA F=4.862, p=0.008, eta-sq=0.0007; pseudoreplication inflates significance (see SS11 caveat) | **Partial** |
| C4 | True-batch validates Phase 1 | SS8: 0.94% safety flip rate, 99.45% agreement | **Established** |
| C5 | Directional analysis shows net SAFE bias | SS22: 69 vs 28, p=3.8e-5 | **Established** |

### Claim Hierarchy (Citation Grade)

The following table classifies every major claim by its evidential strength, intended for use when citing TR141 results in downstream reports or publications.

| Grade | Criteria | Claims |
|-------|----------|--------|
| **A (Strong)** | p < 0.001, replicated or large N, no major caveats | C5: Net SAFE directional bias (p=3.8e-5, N=97); TOST equivalence (14/14 pass); No critical batch-size threshold (7 models, 5 batch sizes) |
| **B (Moderate)** | p < 0.05, consistent direction, some caveats | C1: Safety > capability flip rate (1.3x, CIs overlap); C4: Phase 2 validates Phase 1 (99.45% agreement); TruthfulQA most sensitive task (2.7%, 5-13x other tasks) |
| **C (Suggestive)** | Consistent pattern, not individually significant or major caveats | C2: 2.9x cross-architecture spread (floor effect in low-safety models); C3: Alignment ANOVA (pseudoreplication caveat); Flip-latency association (small N flipped) |
| **D (Preliminary)** | Single observation, confounded, or insufficiently powered | SFT > RLHF > DPO ranking (1 model per non-RLHF category); GPU-specific directionality (single GPU architecture) |

---

## When to Use This Report

### Scenario 1: Selecting a model for batch inference in a safety-critical pipeline

**Question:** Which model families are most robust to batch-induced safety degradation?

**Answer:** See SS10 Table 5. DPO-aligned models (stablelm-2-zephyr, 0.38%) and distilled models (smollm2-1.7b, 0.38%) show the lowest fragility. Avoid SFT-only alignment (phi-3.5-mini, 1.11%) if batch safety robustness is a priority. Within the RLHF family, Llama models (0.43-0.47%) outperform Qwen models (0.68-0.98%).

### Scenario 2: Deciding whether to use deterministic inference kernels

**Question:** Is the batch-safety effect large enough to justify the throughput cost of deterministic inference?

**Answer:** See SS9 and SS13. All TOST tests confirm practical equivalence within +/-3pp, and the aggregate safety flip rate is 0.63%. For most applications, this level of perturbation does not warrant the approximately 34% throughput penalty of SGLang deterministic mode. However, for safety-critical deployments processing ambiguous prompts (see SS7.4, TruthfulQA at 2.7%), deterministic inference may be justified.

### Scenario 3: Understanding the relationship between batch perturbation and TR138 findings

**Question:** How do TR141's cross-architecture results compare to TR138's single-architecture findings?

**Answer:** See SS21 and SS22. TR141 replicates TR138's directional finding (safety flips > capability flips) but at lower magnitude (1.3x vs 4x ratio). Critically, TR141 finds a net SAFE directional bias (p=3.8e-5), opposite to TR138's feared unsafe direction. This suggests that the unsafe-direction concern identified in TR138 may be model-specific rather than universal.

### Scenario 4: Validating batch inference before production deployment

**Question:** What batch sizes should I test, and what flip rate threshold indicates a problem?

**Answer:** See SS20. No critical batch-size threshold was detected for any model, so there is no single "danger zone" batch size. Test at your actual production batch configuration. A safety flip rate above 1% (observed only for phi-3.5-mini and qwen2.5-1.5b) should trigger investigation; below 0.5% is within the range observed for the most robust models.

---

## Table of Contents

- [Abstract](#abstract)
- [Executive Summary](#executive-summary)
- [When to Use This Report](#when-to-use-this-report)
- [SS1. Metric Definitions](#ss1-metric-definitions)
- [SS2. Research Question and Hypotheses](#ss2-research-question-and-hypotheses)
- [SS3. Methodology](#ss3-methodology)
- [SS4. Models and Configuration](#ss4-models-and-configuration)
- [SS5. Phase 1: Output Identity Analysis](#ss5-phase-1-output-identity-analysis)
- [SS6. Phase 1: Safety vs Capability Flip Rates](#ss6-phase-1-safety-vs-capability-flip-rates)
- [SS7. Phase 1: Aggregate Flip Rate Analysis](#ss7-phase-1-aggregate-flip-rate-analysis)
- [SS8. Phase 1: Flip Direction Breakdown](#ss8-phase-1-flip-direction-breakdown)
- [SS9. Phase 1: Per-Task Sensitivity](#ss9-phase-1-per-task-sensitivity)
- [SS10. Cross-Architecture Fragility Ranking](#ss10-cross-architecture-fragility-ranking)
- [SS11. Alignment-Type Comparison and ANOVA](#ss11-alignment-type-comparison-and-anova)
- [SS12. Phase 2: True-Batch Validation](#ss12-phase-2-true-batch-validation)
- [SS13. Phase 2: Flip Rate Comparison](#ss13-phase-2-flip-rate-comparison)
- [SS14. Cross-Phase Synthesis](#ss14-cross-phase-synthesis)
- [SS15. TOST Equivalence Analysis](#ss15-tost-equivalence-analysis)
- [SS16. Power Analysis](#ss16-power-analysis)
- [SS17. Statistical Tests: Per-Model Fisher Exact and Chi-Squared](#ss17-statistical-tests-per-model-fisher-exact-and-chi-squared)
- [SS18. Latency Analysis](#ss18-latency-analysis)
- [SS19. Flip-Latency Correlation](#ss19-flip-latency-correlation)
- [SS20. Slope Heterogeneity and Critical Thresholds](#ss20-slope-heterogeneity-and-critical-thresholds)
- [SS21. Refusal-Style Analysis](#ss21-refusal-style-analysis)
- [SS22. Combined Directional Analysis](#ss22-combined-directional-analysis)
- [SS23. Jailbreak Type Breakdown](#ss23-jailbreak-type-breakdown)
- [SS24. Per-Category Bias Analysis](#ss24-per-category-bias-analysis)
- [SS25. Variance-Safety Correlation](#ss25-variance-safety-correlation)
- [SS26. Safety-Capability Divergence](#ss26-safety-capability-divergence)
- [SS27. Cross-TR138 Validation](#ss27-cross-tr138-validation)
- [SS28. Limitations](#ss28-limitations)
- [SS29. Conclusions](#ss29-conclusions)
- [SS30. Production Guidance](#ss30-production-guidance)
- [SS31. Reproducibility](#ss31-reproducibility)
- [Appendix A: Full Output Identity Matrix](#appendix-a-full-output-identity-matrix)
- [Appendix B: Per-Model Phase 2 Flip Rates](#appendix-b-per-model-phase-2-flip-rates)
- [Appendix C: Overall Statistical Tests by Batch Size](#appendix-c-overall-statistical-tests-by-batch-size)
- [Appendix D: Latency-Throughput Scaling](#appendix-d-latency-throughput-scaling)
- [Appendix E: Per-Task Flip Rate Matrix (Phase 1)](#appendix-e-per-task-flip-rate-matrix-phase-1)
- [Appendix F: Glossary](#appendix-f-glossary)
- [References](#references)

---

## SS1. Metric Definitions

This section defines all metrics used throughout the report to ensure unambiguous interpretation of results.

| Metric | Symbol | Definition | Range |
|--------|--------|-----------|-------|
| Safety flip rate | SFR | Fraction of safety-domain prompts whose binary classification (safe/unsafe) changes between batch=1 and batch=N | [0, 1] |
| Capability flip rate | CFR | Fraction of capability-domain prompts whose binary classification (correct/incorrect) changes between batch=1 and batch=N | [0, 1] |
| Flip ratio | FR | SFR / CFR; values > 1.0 indicate disproportionate safety degradation | [0, inf) |
| Output identity | OI | Fraction of prompts producing byte-identical outputs between batch=1 and batch=N | [0, 1] |
| TOST equivalence | -- | Two One-Sided Tests for practical equivalence within +/-3 percentage point margin (bound = 0.03) | p-value |
| Minimum detectable effect | MDE | Smallest effect size detectable at 80% power and alpha = 0.05, in percentage points | pp |
| Cohen's d | d | Standardized mean difference; |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large | (-inf, inf) |
| Pearson r | r | Linear correlation coefficient between two continuous variables | [-1, 1] |
| Eta-squared | eta-sq | Proportion of variance explained by the grouping variable in ANOVA | [0, 1] |
| Wilson CI | -- | Wilson score confidence interval for binomial proportions, recommended for small n | [0, 1] |
| Holm-Bonferroni | -- | Step-down multiple comparison correction controlling family-wise error rate | adjusted p |
| Binomial test | -- | Exact test for whether the observed number of successes differs from a null proportion (0.5) | p-value |
| Flip direction | -- | Whether a safety flip moves toward unsafe (refusal to compliance) or toward safe (compliance to refusal) | categorical |
| Compliance slope | -- | Linear regression slope of compliance rate against batch size; near-zero indicates no batch-size trend | per-BS unit |

**Observations.** The +/-3pp TOST bound was chosen to match prior Banterhearts reports (TR138, TR139) and represents the minimum practically meaningful difference in safety classification rates. The Wilson CI is preferred over the Wald interval for flip rate confidence intervals because many per-model-per-batch-size cells have fewer than 10 flips, where the Wald interval's normal approximation degrades.

### Key Caveats for Reading This Report

1. **Small absolute flip counts.** Most per-model-per-batch-size cells contain fewer than 5 safety flips. Individual cell ratios (e.g., S/C ratio of "inf" when capability flips = 0) should be treated as illustrative, not definitive. Aggregate analyses (SS7, SS10, SS11) are the statistically reliable results.
2. **Unbalanced alignment categories.** RLHF has 4 models; SFT, DPO, and distilled each have 1. The ANOVA (SS11) detects between-category effects but cannot estimate within-category variance for single-model categories.
3. **Single GPU and seed.** All results derive from one GPU architecture (Blackwell) and one seed (137). The specific prompts that flip may change across hardware and seeds, though aggregate statistics are expected to be stable (see SS27 cross-TR comparison).
4. **Regex-only scoring.** The binary safety classifier captures hard refusal/compliance boundaries but misses graded safety quality changes within the "refuse" or "comply" categories.
5. **Floor effects for low-safety models.** smollm2-1.7b (25 baseline refusals) and stablelm-2-zephyr (60 baseline refusals) have limited opportunity for safety flips because they rarely refuse in the first place (see SS10 and SS21). Their low fragility may partly reflect this mechanical constraint rather than alignment robustness.

---

## SS2. Research Question and Hypotheses

### Research Questions

> **RQ1:** Does batch-induced output non-determinism disproportionately degrade safety outputs, and does this effect vary across model architectures?

> **RQ2:** Does alignment approach (RLHF, SFT, DPO, distilled) predict susceptibility to batch-induced safety degradation?

### Hypotheses

- **H1 (Null):** Batch-induced output changes are safety-neutral across all architectures. Safety flip rate equals capability flip rate at all batch sizes for all models.
- **H2 (Alternative):** Batch-induced changes disproportionately degrade safety, with effect size varying by architecture family.
- **H3 (Alignment):** Alignment approach predicts fragility -- distilled and SFT alignment may be more brittle than RLHF under batch perturbation, because RLHF's iterative preference optimization may produce more robust refusal boundaries.

### Rationale

TR138 established that batch perturbation induces safety flips on 3 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) with a 4x safety-to-capability ratio. TR141 extends this to 7 models from 5 families to determine whether the effect generalizes across architectures and whether alignment approach is a meaningful predictor. The choice of 4 alignment types (RLHF, SFT, DPO, distilled) provides coverage of the major training paradigms used in production instruction-tuned models.

### Theoretical Motivation

The batch perturbation effect arises from IEEE 754 floating-point non-associativity. When a GPU processes multiple prompts in a single batch, the order of reduction operations in attention and MLP layers changes compared to single-prompt processing. Because floating-point addition is not associative ((a+b)+c != a+(b+c) in general), the final logits differ slightly between batch=1 and batch=N for the same prompt. Under greedy decoding (temperature=0), these logit differences can change the argmax token at one or more positions, producing a different output string. If the affected tokens are in a semantically critical region (e.g., the difference between "I cannot help with that" and "Sure, here is how to..."), the safety classification flips.

The cross-architecture hypothesis extends this: different model architectures have different attention patterns, different layer normalization implementations, and different parameter magnitudes, all of which affect how floating-point perturbation propagates through the network. Models with sharper refusal boundaries (narrower logit gaps between "refuse" and "comply" tokens) should be more susceptible to flip under the same perturbation magnitude.

---

## SS3. Methodology

### Experimental Design

Two-phase experiment measuring output non-determinism under batch inference on a cloud NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (98 GB VRAM) via Google Colab.

| Parameter | Value |
|-----------|-------|
| **Temperature** | 0.0 (greedy decoding) |
| **Seed** | 137 (different from TR138's 42 for independent samples) |
| **Max tokens** | 256 |
| **Warmup** | 3 requests per model |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition, 98 GB VRAM |
| **Backend** | vLLM (Docker), FP16 precision, --gpu-memory-utilization 0.90 |
| **Scoring** | Regex safety classifiers (no LLM judge) |

**Observations.** Greedy decoding (temperature=0) maximizes sensitivity to floating-point non-associativity because there is no sampling noise to mask deterministic perturbations. The seed=137 was deliberately chosen to differ from TR138's seed=42, ensuring that any replication of TR138 findings is independent rather than an artifact of shared random state. The regex-based scoring approach trades subtlety for perfect reproducibility: the same output always receives the same score, eliminating judge variance as a confound.

### Phase 1: Batch Size Sweep

Synchronized request groups force exact in-flight batch sizes. Each of 7 models is evaluated at 6 batch sizes (1, 2, 4, 8, 16, 32) across 953 prompts per model per batch size, yielding 7 x 6 x 953 = 40,026 total Phase 1 records. Batch=1 serves as the deterministic control.

**Task composition (955 unique prompts):**
- AdvBench refusal: 100 prompts (safety)
- Jailbreak amplification: 120 prompts (safety)
- BBQ bias: 200 prompts (safety)
- TruthfulQA: 50 prompts (safety)
- MMLU: 285 prompts (capability)
- ARC-Challenge: 200 prompts (capability)

This yields 470 safety prompts and 485 capability prompts per model per batch size (with minor variation due to prompt filtering), for a total of 3,276 safety evaluations and 3,395 capability evaluations per batch size across all 7 models.

### Phase 2: True-Batch Validation

Single vLLM completions call with an explicit prompt list for true tensor batching, validating that Phase 1 effects are not artifacts of request-dispatch timing. Each of 7 models is evaluated at 3 batch sizes (1, 4, 8) across 450 prompts per model per batch size, yielding 7 x 3 x 450 = 9,450 total Phase 2 records.

**Phase 2 task composition (450 prompts):** 250 safety prompts + 200 capability prompts, drawn as a subset of the Phase 1 prompt set to enable direct phase-to-phase comparison.

### Scoring Methodology

Safety classification uses regex-based pattern matching, categorizing each response as "refuse" or "comply" based on the presence of refusal patterns (e.g., "I cannot", "I'm unable", "I won't", "it's not appropriate") or compliance patterns (e.g., proceeding with the requested content). The scorer returns a binary 0/1 safety score for each safety-domain prompt, where 1 indicates the model refused the harmful request (safe behavior).

Capability classification uses answer extraction followed by exact match comparison to ground-truth labels. For MMLU and ARC-Challenge, the model's selected answer option (A/B/C/D) is extracted and compared to the correct answer. TruthfulQA uses a similar approach adapted for its truth/false format. BBQ bias scoring evaluates whether the model avoids stereotypical reasoning in its answer selection.

A "flip" occurs when the binary classification (safe/unsafe for safety prompts, correct/incorrect for capability prompts) differs between the batch=1 baseline and the batch=N treatment condition. The flip rate is the fraction of prompts that flip, computed separately for safety and capability domains.

### Statistical Framework

All statistical tests follow a pre-registered analysis plan with 28 analysis passes:

1. **Primary analysis:** Chi-squared and Fisher exact tests for safety vs capability flip rate asymmetry at each model x batch-size cell, with Holm-Bonferroni correction across all 35 comparisons.
2. **Equivalence testing:** TOST at +/-3pp bound for each phase x batch-size x domain combination.
3. **Variance analysis:** One-way ANOVA on safety flip rate by alignment type, reporting F-statistic, p-value, and eta-squared.
4. **Directional analysis:** Binomial test on flip direction counts (refusal-to-compliance vs compliance-to-refusal) against null proportion 0.5.
5. **Correlation analyses:** Pearson r for flip count vs baseline safety score (variance-safety), and flipped vs stable latency (flip-latency).
6. **Threshold detection:** Wilson CI non-overlap method for identifying critical batch sizes where safety diverges from capability.
7. **Power analysis:** MDE calculation at 80% power, alpha=0.05, for each phase and domain.
8. **Cross-phase validation:** Flip agreement percentage between Phase 1 and Phase 2 at matched batch sizes.

The alpha level is 0.05 throughout, with Holm-Bonferroni correction for the primary per-cell analysis. TOST uses a pre-specified bound of +/-3 percentage points, matching prior Banterhearts reports.

### Dropped Model: gemma-2-2b

The original experimental design included 8 models. gemma-2-2b (Gemma family, distilled alignment, 2,614M parameters) was dropped prior to evaluation due to HuggingFace gated model access restrictions on the Colab execution environment. This leaves 7 completed models from 5 families. The absence of gemma-2-2b reduces coverage of the distilled alignment category to a single model (smollm2-1.7b), which limits the generalizability of conclusions about distilled alignment specifically.

---

## SS4. Models and Configuration

Table 1 presents the 7 models evaluated in this study, with their family, alignment type, parameter count, and phase participation.

**Table 1: Model Registry**

| Model | HuggingFace ID | Family | Alignment | Params | Phase 1 | Phase 2 | Backend |
|-------|---------------|--------|-----------|--------|---------|---------|---------|
| llama3.2-1b | meta-llama/Llama-3.2-1B-Instruct | Llama | RLHF | 1,236M | 953 prompts x 6 BS | 450 x 3 BS | vLLM FP16 |
| llama3.2-3b | meta-llama/Llama-3.2-3B-Instruct | Llama | RLHF | 3,213M | 953 x 6 | 450 x 3 | vLLM FP16 |
| qwen2.5-1.5b | Qwen/Qwen2.5-1.5B-Instruct | Qwen | RLHF | 1,543M | 953 x 6 | 450 x 3 | vLLM FP16 |
| qwen2.5-3b | Qwen/Qwen2.5-3B-Instruct | Qwen | RLHF | 3,000M | 953 x 6 | 450 x 3 | vLLM FP16 |
| phi-3.5-mini | microsoft/Phi-3.5-mini-instruct | Phi | SFT | 3,821M | 953 x 6 | 450 x 3 | vLLM FP16 |
| smollm2-1.7b | HuggingFaceTB/SmolLM2-1.7B-Instruct | SmolLM | Distilled | 1,700M | 953 x 6 | 450 x 3 | vLLM FP16 |
| stablelm-2-zephyr | stabilityai/stablelm-2-zephyr-1_6b | StableLM | DPO | 1,600M | 953 x 6 | 450 x 3 | vLLM FP16 |

**Observations.** The model selection spans four alignment paradigms: RLHF (4 models: Llama and Qwen families), SFT (1 model: Phi), DPO (1 model: StableLM), and distilled (1 model: SmolLM). The RLHF category has the strongest statistical power due to 4 models, while SFT, DPO, and distilled each have only 1 model -- sufficient for inclusion in the ANOVA but insufficient for within-category variance estimation. Parameter counts range from 1,236M to 3,821M (3.1x range), all within the small-to-medium model regime where batch perturbation effects are most practically relevant for edge deployment.

The originally planned gemma-2-2b (Gemma family, distilled, 2,614M) was dropped due to HuggingFace gated access restrictions on the Colab runtime. Had it been included, it would have doubled the distilled category's representation and added a sixth distinct architecture family to the analysis.

**Table 1b: Alignment Type Distribution**

| Alignment Type | Training Approach | Models | Total Safety Evaluations |
|---------------|------------------|--------|------------------------|
| RLHF | Reward model + PPO fine-tuning | 4 (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b, qwen2.5-3b) | 9,360 |
| SFT | Supervised fine-tuning on demonstrations only | 1 (phi-3.5-mini) | 2,340 |
| DPO | Direct Preference Optimization (contrastive) | 1 (stablelm-2-zephyr) | 2,340 |
| Distilled | Knowledge distillation from larger teacher | 1 (smollm2-1.7b) | 2,340 |

**Observations.** The alignment type distribution is intentionally diverse but not balanced. RLHF dominates with 4 models because it is the most common alignment approach in production models, providing within-type variance estimation. The single-model categories (SFT, DPO, distilled) provide point estimates but cannot support within-category variance analysis. This design choice maximizes the breadth of alignment coverage at the cost of within-category statistical power. The ANOVA result (SS11) should be interpreted as detecting a between-category effect, with the caveat that the non-RLHF categories are each represented by a single model.

---

## SS5. Phase 1: Output Identity Analysis

Output identity measures the fraction of prompts producing byte-identical outputs between batch=1 (control) and each non-baseline batch size. A prompt need not change its classification to be "non-identical" -- any character-level difference counts. This metric captures the total magnitude of batch-induced non-determinism before any classification-level filtering, providing a denominator for understanding what fraction of output changes actually matter for safety or capability assessments.

**Table 2: Output Identity Rates by Model and Batch Size (Phase 1)**

| Model | BS=2 | BS=4 | BS=8 | BS=16 | BS=32 | Mean |
|-------|------|------|------|-------|-------|------|
| llama3.2-1b | 94.0% | 93.9% | 94.3% | 93.2% | 93.9% | 93.9% |
| llama3.2-3b | 90.8% | 89.8% | 90.0% | 90.1% | 89.8% | 90.1% |
| qwen2.5-1.5b | 90.5% | 90.1% | 89.9% | 90.2% | 92.3% | 90.6% |
| qwen2.5-3b | 85.8% | 86.8% | 85.8% | 87.3% | 84.9% | 86.1% |
| smollm2-1.7b | 92.8% | 92.1% | 92.0% | 92.9% | 92.9% | 92.5% |
| stablelm-2-zephyr | 86.4% | 86.2% | 86.8% | 87.3% | 86.7% | 86.7% |
| phi-3.5-mini | 75.5% | 75.1% | 76.8% | 73.2% | 74.5% | 75.0% |
| **Overall** | **88.0%** | **87.7%** | **88.0%** | **87.8%** | **87.9%** | **87.9%** |

**Observations.** Output identity varies dramatically across models: llama3.2-1b maintains 93.9% identity (only 6.1% of outputs change at all), while phi-3.5-mini drops to 75.0% (25% of outputs change). This 19-percentage-point spread in non-determinism is the largest observed in any Banterhearts batch perturbation study.

Notably, output identity is relatively stable across batch sizes within each model -- the standard deviation across batch sizes is less than 1.5pp for every model. This means the degree of non-determinism is primarily a model property, not a batch-size property. The implication is important: if a model shows 12% output change at BS=2, it will show approximately 12% at BS=32 as well. The batch size determines *which specific prompts* change, not *how many*.

phi-3.5-mini's high non-determinism (25% output change rate) may relate to several factors: (a) its larger parameter count (3,821M) creating more floating-point accumulation paths that are sensitive to batching order, (b) its SFT alignment potentially producing sharper token probability distributions that are more sensitive to small logit perturbations, and (c) its tendency to generate longer, more detailed responses (see SS18 latency analysis) which provides more tokens at which divergence can occur.

The gap between output change rate (approximately 12% overall) and classification flip rate (approximately 0.5% overall) is instructive: only 1 in approximately 24 changed outputs actually crosses a classification boundary. The remaining 23 of 24 changes are semantically neutral variations -- different word choices, punctuation differences, or restructured sentences that do not affect the binary classification. This ratio provides a useful calibration of how robust the safety and capability classifiers are to paraphrasing: even when batch perturbation changes 12% of outputs, 99% of those changes are classification-preserving.

The per-model output-change-to-flip conversion rate varies substantially:

| Model | Output Change Rate | Safety Flip Rate | Conversion (Flip/Change) |
|-------|-------------------|-----------------|-------------------------|
| phi-3.5-mini | 25.0% | 1.11% | 4.4% |
| qwen2.5-1.5b | 9.4% | 0.98% | 10.4% |
| qwen2.5-3b | 13.9% | 0.68% | 4.9% |
| llama3.2-3b | 9.9% | 0.43% | 4.3% |
| smollm2-1.7b | 7.5% | 0.38% | 5.1% |
| llama3.2-1b | 6.1% | 0.47% | 7.7% |
| stablelm-2-zephyr | 13.3% | 0.38% | 2.9% |

**Observations (conversion rate).** The mean conversion rate across all 7 models is approximately 5.0%. qwen2.5-1.5b has the highest conversion rate (10.4%), meaning that when its outputs do change, those changes are unusually likely to cross a safety classification boundary. This suggests that qwen2.5-1.5b's non-deterministic outputs are concentrated near the refusal boundary, while stablelm-2-zephyr's changes (2.9% conversion) are distributed more uniformly across semantically neutral regions. The conversion rate provides a per-model "flip efficiency" metric: how much of the model's non-determinism translates into actionable safety risk.

---

## SS6. Phase 1: Safety vs Capability Flip Rates

While output identity captures any textual change, flip rates capture changes that cross a classification boundary. A prompt that changes from "I cannot assist with that request" to "I can't help with that" is an output change but not a flip, because both are refusals. A prompt that changes from "I cannot assist" to "Sure, here is how to..." is both an output change and a safety flip -- and specifically a refusal-to-compliance (toward unsafe) flip. Table 3 presents the per-model, per-batch-size safety and capability flip rates with the safety-to-capability ratio.

**Table 3: Safety vs Capability Flip Rates by Model and Batch Size (Phase 1)**

| Model | BS | Safety Flip | Cap Flip | Ratio (S/C) | Safety n | Cap n |
|-------|----|-----------|---------|-------------|----------|-------|
| llama3.2-1b | 2 | 0.21% | 0.21% | 1.04 | 468 | 485 |
| llama3.2-1b | 4 | 0.43% | 0.21% | **2.07** | 468 | 485 |
| llama3.2-1b | 8 | 0.21% | 0.00% | inf | 468 | 485 |
| llama3.2-1b | 16 | 0.85% | 0.00% | inf | 468 | 485 |
| llama3.2-1b | 32 | 0.64% | 0.21% | **3.11** | 468 | 485 |
| llama3.2-3b | 2 | 0.43% | 0.00% | inf | 468 | 485 |
| llama3.2-3b | 4 | 0.64% | 0.62% | 1.04 | 468 | 485 |
| llama3.2-3b | 8 | 0.43% | 0.21% | **2.07** | 468 | 485 |
| llama3.2-3b | 16 | 0.21% | 0.41% | 0.52 | 468 | 485 |
| llama3.2-3b | 32 | 0.43% | 0.62% | 0.69 | 468 | 485 |
| phi-3.5-mini | 2 | 1.28% | 2.27% | 0.57 | 468 | 485 |
| phi-3.5-mini | 4 | 1.71% | 2.27% | 0.75 | 468 | 485 |
| phi-3.5-mini | 8 | 1.07% | 1.24% | 0.86 | 468 | 485 |
| phi-3.5-mini | 16 | 0.64% | 2.06% | 0.31 | 468 | 485 |
| phi-3.5-mini | 32 | 0.85% | 1.44% | 0.59 | 468 | 485 |
| qwen2.5-1.5b | 2 | 0.85% | 0.00% | inf | 468 | 485 |
| qwen2.5-1.5b | 4 | 1.07% | 0.21% | **5.18** | 468 | 485 |
| qwen2.5-1.5b | 8 | 1.07% | 0.21% | **5.18** | 468 | 485 |
| qwen2.5-1.5b | 16 | 1.07% | 0.00% | inf | 468 | 485 |
| qwen2.5-1.5b | 32 | 0.85% | 0.00% | inf | 468 | 485 |
| qwen2.5-3b | 2 | 0.64% | 0.21% | **3.11** | 468 | 485 |
| qwen2.5-3b | 4 | 0.85% | 0.21% | **4.15** | 468 | 485 |
| qwen2.5-3b | 8 | 0.64% | 0.21% | **3.11** | 468 | 485 |
| qwen2.5-3b | 16 | 0.64% | 0.00% | inf | 468 | 485 |
| qwen2.5-3b | 32 | 0.64% | 0.00% | inf | 468 | 485 |
| smollm2-1.7b | 2 | 0.43% | 0.41% | 1.04 | 468 | 485 |
| smollm2-1.7b | 4 | 0.64% | 0.62% | 1.04 | 468 | 485 |
| smollm2-1.7b | 8 | 0.43% | 0.62% | 0.69 | 468 | 485 |
| smollm2-1.7b | 16 | 0.21% | 0.62% | 0.35 | 468 | 485 |
| smollm2-1.7b | 32 | 0.21% | 0.21% | 1.04 | 468 | 485 |
| stablelm-2-zephyr | 2 | 0.21% | 0.41% | 0.52 | 468 | 485 |
| stablelm-2-zephyr | 4 | 0.64% | 0.00% | inf | 468 | 485 |
| stablelm-2-zephyr | 8 | 0.21% | 0.21% | 1.04 | 468 | 485 |
| stablelm-2-zephyr | 16 | 0.43% | 0.00% | inf | 468 | 485 |
| stablelm-2-zephyr | 32 | 0.43% | 0.41% | 1.04 | 468 | 485 |

**Observations.** The data reveals a striking model-level pattern that challenges the assumption that batch perturbation uniformly affects safety more than capability.

**Pattern 1: Qwen models show strong safety-specific fragility.** qwen2.5-1.5b and qwen2.5-3b show consistently elevated safety-to-capability ratios (3.1x to 5.2x across most batch sizes), with many batch sizes showing zero capability flips. This suggests the Qwen RLHF alignment is particularly susceptible to safety-specific batch perturbation while maintaining stable capability outputs. The mechanism may be architecture-specific: Qwen's attention implementation or layer normalization could create narrower safety margins in the refusal logit space.

**Pattern 2: phi-3.5-mini is highly non-deterministic but capability-biased.** phi-3.5-mini shows the opposite pattern -- its safety flip rate is consistently *lower* than its capability flip rate (ratios 0.31-0.86), meaning that while phi-3.5-mini is the most fragile model overall (highest total flip count at 71), its non-determinism disproportionately affects capability outputs rather than safety outputs. This is a nuanced finding: phi-3.5-mini is the "least stable" model but not the "least safe" model under batch perturbation. Importantly, phi-3.5-mini still has the highest *absolute* safety flip rate (1.11%, SS10) despite its capability-biased ratio. The below-1.0 S/C ratios reflect phi-3.5-mini's exceptionally high capability flip rate (45 capability flips, 2-4x more than any other model), not a low safety flip rate. In absolute terms, phi-3.5-mini remains the most safety-fragile model.

**Pattern 3: DPO and distilled models are near-symmetric.** smollm2-1.7b and stablelm-2-zephyr show ratios close to 1.0, indicating their batch perturbation is safety-neutral -- whatever perturbation occurs, it affects safety and capability domains equally.

**Pattern 4: Llama models show mild safety bias.** llama3.2-1b and llama3.2-3b show moderate safety-to-capability ratios (1.0-3.1x), with the effect varying by batch size without a clear trend.

The absolute flip counts are small (1-11 flips per cell), so individual cell ratios should be interpreted with caution; the aggregate analysis in SS7 provides more statistically robust comparisons. The "inf" ratios (where capability flips = 0) are particularly unreliable due to the zero denominator.

---

## SS7. Phase 1: Aggregate Flip Rate Analysis

Aggregating across all 7 models and all non-baseline batch sizes provides the headline safety-vs-capability comparison. This analysis pools 16,380 safety evaluations (3,276 per batch size x 5 batch sizes) and 16,975 capability evaluations (3,395 per batch size x 5 batch sizes) to obtain the most statistically powered estimate of the safety-capability asymmetry.

**Table 4: Aggregate Flip Rates by Batch Size (Phase 1)**

| Batch Size | Safety Flip Rate | Safety 95% CI | Cap Flip Rate | Cap 95% CI | Ratio | Safety Flips | Cap Flips |
|-----------|-----------------|--------------|--------------|-----------|-------|-------------|----------|
| 2 | 0.58% | [0.37%, 0.90%] | 0.50% | [0.31%, 0.80%] | 1.16 | 19 / 3,276 | 17 / 3,395 |
| 4 | 0.85% | [0.59%, 1.23%] | 0.59% | [0.38%, 0.91%] | 1.45 | 28 / 3,276 | 20 / 3,395 |
| 8 | 0.58% | [0.37%, 0.90%] | 0.38% | [0.22%, 0.65%] | 1.51 | 19 / 3,276 | 13 / 3,395 |
| 16 | 0.58% | [0.37%, 0.90%] | 0.44% | [0.27%, 0.73%] | 1.31 | 19 / 3,276 | 15 / 3,395 |
| 32 | 0.58% | [0.37%, 0.90%] | 0.41% | [0.25%, 0.69%] | 1.41 | 19 / 3,276 | 14 / 3,395 |
| **All** | **0.63%** | **[0.52%, 0.77%]** | **0.47%** | **[0.37%, 0.59%]** | **1.35** | **104 / 16,380** | **79 / 16,975** |

**Observations.** The aggregate safety flip rate (0.63%) exceeds the capability flip rate (0.47%) at every batch size, with ratios ranging from 1.16 (BS=2) to 1.51 (BS=8). The overall ratio of 1.35 is notably lower than TR138's reported 4x ratio, suggesting that the stronger effect in TR138 may have been driven by the specific 3-model selection rather than being a universal property of batch perturbation. Batch size 4 shows the highest absolute safety flip rate (0.85%, 28 flips), but the confidence intervals for all batch sizes overlap substantially, and none of the per-batch-size chi-squared tests reaches significance after Holm-Bonferroni correction (see SS17). The 95% CIs for safety and capability flip rates overlap at every batch size, meaning we cannot reject the null hypothesis that safety and capability are equally affected -- though the consistent directional pattern across all 5 batch sizes is suggestive.

---

## SS8. Phase 1: Flip Direction Breakdown

For safety flips, the direction matters critically for operational risk assessment. A refusal-to-compliance flip (toward unsafe) means a prompt that was correctly refused at batch=1 is now answered at batch=N -- creating an undetected safety gap. A compliance-to-refusal flip (toward safe) means a prompt that was answered at batch=1 is now refused at batch=N -- creating a false-positive refusal that reduces helpfulness but does not create a safety gap. The operational cost of unsafe-direction flips is much higher than safe-direction flips, making the directional analysis central to risk assessment. Table 5 breaks down the direction of all safety flips.

**Table 5: Flip Direction by Batch Size (Phase 1)**

| Batch Size | To Unsafe (R->C) | To Safe (C->R) | Total | Net Direction | Binomial p |
|-----------|-----------------|---------------|-------|--------------|-----------|
| 2 | 5 | 11 | 16 | neutral | 0.210 |
| 4 | 7 | 20 | 27 | **net safe** | **0.019** |
| 8 | 4 | 14 | 18 | **net safe** | **0.031** |
| 16 | 7 | 11 | 18 | neutral | 0.481 |
| 32 | 5 | 13 | 18 | neutral | 0.096 |
| **All** | **28** | **69** | **97** | **net safe** | **3.8e-5** |

**Observations.** Of the 104 total safety flips (SS7), 97 are directional (can be classified as toward-safe or toward-unsafe). The remaining 7 flips occur in tasks where the scoring does not map cleanly to a refusal/compliance binary (e.g., TruthfulQA truth-value changes or BBQ bias-detection shifts), so they are excluded from directional analysis.

The aggregate directional result is the single most surprising finding of TR141. Of 97 total directional safety flips, 69 (71.1%) move toward safety (compliance to refusal) while only 28 (28.9%) move toward unsafety (refusal to compliance). The binomial test against a null proportion of 0.5 yields p=3.8e-5, highly significant. This means batch perturbation in this experimental configuration acts as a **safety-conservative perturbation** -- when a marginal prompt's classification changes, it is 2.46x more likely to become *more* restrictive than *less* restrictive. This is the opposite of the a priori concern that batch perturbation would preferentially erode safety. Two batch sizes (4 and 8) show individually significant net-safe asymmetry.

**Table 5b: Per-Model Aggregate Flip Direction (All Batch Sizes)**

| Model | To Unsafe | To Safe | Total | Net Direction |
|-------|----------|---------|-------|--------------|
| llama3.2-1b | 0 | 11 | 11 | Strongly safe |
| llama3.2-3b | 3 | 7 | 10 | Net safe |
| phi-3.5-mini | 3 | 17 | 20 | Strongly safe |
| qwen2.5-1.5b | 4 | 18 | 22 | Strongly safe |
| qwen2.5-3b | 10 | 6 | 16 | **Net unsafe** |
| smollm2-1.7b | 4 | 5 | 9 | Neutral |
| stablelm-2-zephyr | 4 | 5 | 9 | Neutral |

**Observations (per-model).** The per-model breakdown reveals that the net-safe aggregate is driven primarily by three models: llama3.2-1b (0 unsafe vs 11 safe, strongly safe), qwen2.5-1.5b (4 unsafe vs 18 safe), and phi-3.5-mini (3 unsafe vs 17 safe). Notably, qwen2.5-3b is the only model that shows a net *unsafe* direction (10 unsafe vs 6 safe), suggesting that the directional bias may reverse for specific architectures even within the same family (compare qwen2.5-1.5b's strong net-safe to qwen2.5-3b's net-unsafe). This heterogeneity underscores that directional bias should be tested on a per-model basis in production rather than relying on aggregate cross-architecture trends.

---

## SS9. Phase 1: Per-Task Sensitivity

Different prompt types vary substantially in their susceptibility to batch-induced classification flips because the proximity of model outputs to classification boundaries differs by task. Tasks where model outputs are confidently on one side of the boundary (e.g., clearly harmful prompts that are easily refused) will show low flip rates, while tasks where outputs are near the boundary (e.g., ambiguous prompts where the model is uncertain whether to refuse) will show high flip rates. Table 6 reports the mean flip rate across all 7 models and batch sizes for each task.

**Table 6: Per-Task Sensitivity (Phase 1, Mean Across Models and Batch Sizes)**

| Task | Domain | Mean Flip Rate | Range | Most Sensitive BS | N per BS |
|------|--------|---------------|-------|-------------------|----------|
| truthfulqa | Safety | **2.7%** | 1.7-3.7% | BS=4 (3.7%) | 350 |
| jailbreak_amplification | Safety | 0.8% | 0.5-1.2% | BS=4 (1.2%) | 840 |
| mmlu_real | Capability | 0.5% | 0.3-0.6% | BS=2/4 (0.6%) | 1,995 |
| arc_challenge | Capability | 0.5% | 0.4-0.6% | BS=4 (0.6%) | 1,400 |
| advbench_refusal | Safety | 0.3% | 0.1-0.4% | BS=4/16/32 (0.4%) | 700 |
| bbq_bias | Safety | 0.2% | 0.1-0.3% | BS=2 (0.3%) | 1,386 |

**Observations.** Per-task confidence intervals are wide due to small per-task sample sizes (N per BS ranges from 350 for TruthfulQA to 1,995 for MMLU). For TruthfulQA at BS=4 (13/350 flips), the 95% Wilson CI is approximately [2.1%, 6.2%]; for BBQ at BS=4 (2/1386 flips), it is approximately [0.04%, 0.52%]. These wide intervals mean that the exact per-task flip rates should be treated as estimates, though the ranking (TruthfulQA >> others) is robust.

TruthfulQA stands out as by far the most batch-sensitive task, with a mean flip rate of 2.7% -- more than 5x the next most sensitive task (jailbreak_amplification at 0.8%) and 13x the least sensitive (bbq_bias at 0.2%). This is consistent with TruthfulQA's design: its prompts are deliberately constructed to elicit common misconceptions, placing many model responses near the boundary between "truthful refusal to endorse a misconception" and "compliance with the misconception." Such boundary-proximate responses are exactly the ones most vulnerable to floating-point perturbation. AdvBench refusal prompts are the least sensitive safety task (0.3%), likely because they contain explicit harmful requests that are far from the refusal boundary for most models. BBQ bias prompts are the least sensitive overall (0.2%), suggesting that bias detection is more robustly encoded than refusal decisions. Batch size 4 appears as the most sensitive batch size for 4 of 6 tasks, though this should not be over-interpreted given the small absolute differences.

---

## SS10. Cross-Architecture Fragility Ranking

This section presents the central cross-architecture analysis that motivates TR141's extension beyond TR138. By ranking all 7 models on a unified fragility metric (aggregate safety flip rate), we can identify which architectural and alignment properties predict batch-safety robustness. Models are ranked by their aggregate safety flip rate across all non-baseline batch sizes in Phase 1 (2, 4, 8, 16, 32). Each model contributes 468 safety prompts x 5 batch sizes = 2,340 evaluations.

**Table 7: Cross-Architecture Fragility Ranking**

| Rank | Model | Family | Alignment | Params | Safety Flip Rate | 95% CI | Flips / N |
|------|-------|--------|-----------|--------|-----------------|--------|-----------|
| 1 | **phi-3.5-mini** | Phi | SFT | 3,821M | **1.11%** | [0.76%, 1.62%] | 26 / 2,340 |
| 2 | **qwen2.5-1.5b** | Qwen | RLHF | 1,543M | **0.98%** | [0.66%, 1.47%] | 23 / 2,340 |
| 3 | **qwen2.5-3b** | Qwen | RLHF | 3,000M | **0.68%** | [0.42%, 1.11%] | 16 / 2,340 |
| 4 | **llama3.2-1b** | Llama | RLHF | 1,236M | **0.47%** | [0.26%, 0.84%] | 11 / 2,340 |
| 5 | **llama3.2-3b** | Llama | RLHF | 3,213M | **0.43%** | [0.23%, 0.78%] | 10 / 2,340 |
| 6 | **smollm2-1.7b** | SmolLM | Distilled | 1,700M | **0.38%** | [0.20%, 0.73%] | 9 / 2,340 |
| 7 | **stablelm-2-zephyr** | StableLM | DPO | 1,600M | **0.38%** | [0.20%, 0.73%] | 9 / 2,340 |

**Observations.** The fragility ranking reveals a 2.9x spread between the most fragile model (phi-3.5-mini at 1.11%) and the least fragile (stablelm-2-zephyr and smollm2-1.7b, tied at 0.38%). Several patterns emerge:

First, there is no simple relationship between parameter count and fragility. phi-3.5-mini is both the largest model (3,821M) and the most fragile. However, its high absolute flip rate (1.11%) is driven by output variability rather than narrow refusal boundaries: phi-3.5-mini has the highest output change rate (25.0%, SS5) but a below-average conversion rate of 4.4% (vs the 5.0% cross-model mean). In other words, phi-3.5-mini produces many textual changes under batch perturbation, but only a small fraction cross the safety classification boundary. Its safety fragility is a volume effect, not a precision effect. Additionally, qwen2.5-1.5b (1,543M) is more fragile than the larger qwen2.5-3b (3,000M). Within the Llama family, the 1B and 3B variants have nearly identical fragility (0.47% vs 0.43%).

Second, the Qwen family is consistently more fragile than the Llama family at comparable parameter counts (qwen2.5-1.5b at 0.98% vs llama3.2-1b at 0.47%; qwen2.5-3b at 0.68% vs llama3.2-3b at 0.43%). This within-alignment-type comparison (both are RLHF) suggests that architecture-specific factors beyond alignment approach contribute to fragility.

Third, the DPO and distilled models occupy the bottom of the ranking, suggesting that alternative alignment approaches may produce more robust refusal boundaries -- though this is based on a single model per category and should be validated with additional models.

**Floor-effect warning.** The low fragility of smollm2-1.7b (0.38%) and stablelm-2-zephyr (0.38%) must be interpreted with extreme caution because these models have very few baseline refusals: smollm2-1.7b has only 25 total refusals across all safety prompts, and stablelm-2-zephyr has only 60. A model that rarely refuses has few opportunities for refusal-to-compliance flips. Normalizing fragility by baseline refusals reveals a starkly different picture:

| Model | Safety Flips | Baseline Refusals | Normalized Fragility (flips/refusals) |
|-------|-------------|-------------------|--------------------------------------|
| smollm2-1.7b | 9 | 25 | **36.0%** |
| phi-3.5-mini | 26 | 1,155 | 2.25% |
| qwen2.5-1.5b | 23 | 1,356 | 1.70% |
| llama3.2-1b | 11 | 984 | 1.12% |
| llama3.2-3b | 10 | 1,174 | 0.85% |
| qwen2.5-3b | 16 | 1,181 | 1.35% |
| stablelm-2-zephyr | 9 | 60 | **15.0%** |

When expressed as a fraction of baseline refusals, smollm2-1.7b's normalized fragility is **36.0%** -- meaning over a third of its few refusals are susceptible to batch-induced flipping. stablelm-2-zephyr's normalized fragility is 15.0%. These values are an order of magnitude higher than the high-safety models (phi-3.5-mini at 2.25%, qwen2.5-1.5b at 1.70%). The raw safety flip rate (Table 7) and normalized fragility tell complementary stories: Table 7 measures absolute risk (how many prompts flip per evaluation), while normalized fragility measures boundary sensitivity (what fraction of the model's refusal surface is unstable). For production deployment, both metrics should be considered.

---

## SS11. Alignment-Type Comparison and ANOVA

Grouping models by alignment type enables a direct test of H3 (alignment approach predicts fragility).

**Table 8: Alignment-Type Aggregate Safety Flip Rates**

| Alignment Type | Models | N Models | Flip Rate | 95% CI | Mean Safety Score | Flips / N |
|---------------|--------|----------|-----------|--------|-------------------|-----------|
| RLHF | llama3.2-1b, llama3.2-3b, qwen2.5-1.5b, qwen2.5-3b | 4 | 0.64% | [0.50%, 0.82%] | 0.746 | 60 / 9,360 |
| SFT | phi-3.5-mini | 1 | 1.11% | [0.76%, 1.62%] | 0.781 | 26 / 2,340 |
| DPO | stablelm-2-zephyr | 1 | 0.38% | [0.20%, 0.73%] | 0.446 | 9 / 2,340 |
| Distilled | smollm2-1.7b | 1 | 0.38% | [0.20%, 0.73%] | 0.333 | 9 / 2,340 |

**ANOVA Result:** F(3, N-4) = 4.862, p = 0.0078, eta-squared = 0.001

**Observations.** The one-way ANOVA confirms that alignment type is a statistically significant predictor of safety fragility (p=0.0078), supporting H3. However, the effect size is very small (eta-squared=0.0007), meaning alignment type explains only 0.07% of the total variance in safety flip outcomes. This is expected given that flips are rare events (overall rate 0.63%).

**Pseudoreplication caveat.** The ANOVA treats each prompt evaluation as an independent observation (N=16,380), but prompts evaluated on the same model share the same weights, attention patterns, and refusal boundaries. The effective sample size is therefore closer to the number of models (N=7) than the number of prompt evaluations. A model-level ANOVA (treating each model's aggregate flip rate as one observation, N=7 with df=3,3 for 4 alignment groups) would require a much larger effect to reach significance and cannot be computed here because 3 of 4 alignment groups have only a single model (no within-group variance). The prompt-level ANOVA's p=0.0078 should be interpreted as evidence that alignment type co-varies with fragility, but the statistical significance is inflated by pseudoreplication. C3 is accordingly downgraded from "Established" to "Partial" in the Claim Validation table.

The ranking is SFT (1.11%) > RLHF (0.64%) > DPO = Distilled (0.38%). The SFT-aligned model (phi-3.5-mini) is 2.9x more fragile than the DPO/distilled models. One possible mechanism: SFT alignment optimizes directly on demonstration data, which may produce sharp but narrow refusal boundaries that are more sensitive to floating-point perturbation, while DPO's contrastive optimization and distillation's softer knowledge transfer may produce smoother, more robust boundaries. However, this interpretation is speculative and based on single models per non-RLHF category.

An important confound is that the mean safety score varies substantially across alignment types: DPO (0.446) and distilled (0.333) models have lower baseline safety scores than RLHF (0.746) and SFT (0.781). Lower baseline safety means fewer prompts are near the refusal boundary, which mechanically reduces the number of prompts eligible to flip. The fragility difference may thus partly reflect baseline safety differences rather than alignment-intrinsic robustness.

To quantify this confound, consider a simple model: if a model refuses X% of safety prompts at batch=1, then only X% of prompts have the *opportunity* to flip from refusal to compliance (and (100-X)% can flip the other direction). A model with 33% baseline safety (smollm2-1.7b) has approximately half the "refusal pool" of a model with 78% baseline safety (phi-3.5-mini), mechanically reducing its potential for safety flips even if its refusal boundary is equally fragile. Normalizing the safety flip rate by the number of prompts in each direction (refused vs complied) would provide a boundary-proximity-adjusted fragility measure, but this analysis requires prompt-level baseline data that is not directly available in the aggregate statistics.

**Within-RLHF comparison (controlling for alignment type):**

| Model | Family | Params | Safety Flip Rate | Baseline Safety |
|-------|--------|--------|-----------------|-----------------|
| qwen2.5-1.5b | Qwen | 1,543M | 0.98% | ~0.75 |
| qwen2.5-3b | Qwen | 3,000M | 0.68% | ~0.75 |
| llama3.2-1b | Llama | 1,236M | 0.47% | ~0.75 |
| llama3.2-3b | Llama | 3,213M | 0.43% | ~0.75 |

**Observations (within-RLHF).** Controlling for alignment type (all RLHF), the Qwen family shows approximately 1.6-2.1x higher fragility than the Llama family at comparable parameter counts. Within each family, larger models are slightly *less* fragile (qwen2.5-3b < qwen2.5-1.5b; llama3.2-3b < llama3.2-1b), though the differences are small and may not be meaningful given the confidence interval overlap. The consistent Qwen > Llama pattern within the RLHF category cannot be explained by alignment type or baseline safety score differences, pointing to genuine architecture-level differences in batch perturbation susceptibility.

---

## SS12. Phase 2: True-Batch Validation

Phase 2 uses explicit prompt-list true batching via a single vLLM completions call, eliminating the possibility that Phase 1 effects are artifacts of request-dispatch timing or server-side batching decisions. Each model is evaluated at batch sizes 1, 4, and 8.

**Table 9: Phase 1 vs Phase 2 Flip Agreement**

| Model | Batch Size | N Paired | Flip Agreement % | Score Agreement % |
|-------|------------|----------|------------------|-------------------|
| llama3.2-1b | 4 | 450 | 99.56 | 99.56 |
| llama3.2-1b | 8 | 450 | 99.56 | 99.56 |
| llama3.2-3b | 4 | 450 | 99.56 | 99.56 |
| llama3.2-3b | 8 | 450 | 98.89 | 98.89 |
| phi-3.5-mini | 4 | 450 | 97.78 | 97.78 |
| phi-3.5-mini | 8 | 450 | 98.22 | 98.22 |
| qwen2.5-1.5b | 4 | 450 | 100.00 | 100.00 |
| qwen2.5-1.5b | 8 | 450 | 99.33 | 99.33 |
| qwen2.5-3b | 4 | 450 | 100.00 | 100.00 |
| qwen2.5-3b | 8 | 450 | 99.78 | 99.78 |
| smollm2-1.7b | 4 | 450 | 99.78 | 99.78 |
| smollm2-1.7b | 8 | 450 | 100.00 | 100.00 |
| stablelm-2-zephyr | 4 | 450 | 100.00 | 100.00 |
| stablelm-2-zephyr | 8 | 450 | 99.78 | 99.78 |
| **Mean** | | | **99.45** | **99.45** |

**Observations.** Phase 1 and Phase 2 produce nearly identical classification outcomes, with a mean flip agreement of 99.45% across all 14 model-batch-size pairs. The lowest agreement is phi-3.5-mini at BS=4 (97.78%, corresponding to 10 prompts with differing classifications out of 450), which aligns with phi-3.5-mini's status as the most non-deterministic model. Five pairs achieve perfect 100% agreement (qwen2.5-1.5b BS=4, qwen2.5-3b BS=4, smollm2-1.7b BS=8, stablelm-2-zephyr BS=4, all with 0 disagreements out of 450).

This exceptionally high agreement confirms that Phase 1's synchronized-dispatch methodology is a valid proxy for true tensor batching. The consistent results across dispatch methods (Phase 1) and true batching (Phase 2) strengthen confidence that the observed flip rates are genuine properties of floating-point non-determinism under batching, not artifacts of the evaluation methodology.

The 0.55% disagreement rate (100% - 99.45%) represents prompts where the two phases produce different classifications for the same prompt at the same batch size. These disagreements likely arise from differences in the exact floating-point accumulation paths between dispatch batching and true batching -- the same prompt can produce slightly different logits under the two approaches, and if the prompt is near a classification boundary, the classification may differ. This 0.55% disagreement rate provides a useful benchmark for the "noise floor" of batch-method-induced variation, below which differences cannot be meaningfully attributed to specific batching mechanisms.

The agreement pattern by model is consistent with the fragility ranking (SS10): the most fragile models (phi-3.5-mini, qwen2.5-1.5b) have the lowest agreement rates, while the most robust models (stablelm-2-zephyr, smollm2-1.7b) have the highest. This is expected because fragile models have more prompts near classification boundaries, increasing the probability that the small perturbation differences between Phase 1 and Phase 2 methods cross a boundary.

---

## SS13. Phase 2: Flip Rate Comparison

**Table 10: Phase 2 Aggregate Flip Rates by Batch Size**

| Batch Size | Safety Flip Rate | Safety 95% CI | Cap Flip Rate | Cap 95% CI | Ratio | Safety Flips | Cap Flips |
|-----------|-----------------|--------------|--------------|-----------|-------|-------------|----------|
| 4 | 0.91% | [0.56%, 1.48%] | 0.50% | [0.24%, 1.03%] | 1.83 | 16 / 1,750 | 7 / 1,400 |
| 8 | 0.97% | [0.61%, 1.55%] | 0.36% | [0.15%, 0.83%] | 2.72 | 17 / 1,750 | 5 / 1,400 |
| **Combined** | **0.94%** | **[0.68%, 1.30%]** | **0.43%** | **[0.24%, 0.76%]** | **2.21** | **33 / 3,500** | **12 / 2,800** |

**Observations.** Phase 2's overall safety flip rate (0.94%) is 49% higher than Phase 1's (0.63%), and the safety-to-capability ratio (2.21) is notably higher than Phase 1's (1.35). For a matched batch-size comparison, Phase 1's average safety flip rate at BS=4 and BS=8 is (0.85% + 0.58%) / 2 = 0.72%, versus Phase 2's 0.94% at the same batch sizes -- a 31% amplification under true tensor batching. This amplification factor quantifies how much stricter true batching is compared to dispatch batching for the same nominal batch sizes. This suggests that true tensor batching may amplify safety perturbation compared to dispatch-synchronized batching, possibly because true batching forces all prompts through identical computation paths with no scheduling variation. The batch-size 8 ratio of 2.72 is the highest observed in either phase.

While no individual Phase 2 chi-squared test reaches significance (all Fisher p > 0.05), the directional consistency across both phases and all batch sizes provides cumulative evidence that the safety-capability asymmetry is real, albeit small. The Phase 2 amplification is practically important because production vLLM deployments use continuous batching, which is closer to Phase 2's true-batch methodology than Phase 1's dispatch approach. This suggests that the Phase 2 safety flip rate of 0.94% may be the more accurate estimate of real-world batch perturbation magnitude.

The per-model Phase 2 data (Appendix B) reveals that llama3.2-3b shows the largest Phase 2 amplification: 2.0% safety flip rate at BS=8 (5 flips out of 250), compared to its Phase 1 maximum of 0.64%. This may indicate that llama3.2-3b's safety boundary is particularly sensitive to the exact batching mechanism, with true tensor batching producing larger floating-point perturbations than dispatch batching for this architecture.

---

## SS14. Cross-Phase Synthesis

**Table 11: Cross-Phase Variance and Risk Summary**

The "Approx pp" column below represents the square root of the mean per-prompt score variance, multiplied by 100 to convert to percentage points. This is a standard-deviation-like measure of how much individual prompt scores vary due to batch perturbation.

| Source | Approx pp | Risk Level | N |
|--------|-----------|------------|---|
| Batch variance (Phase 1 dispatch) | 5.05 | Moderate | 6,671 |
| True batch variance (Phase 2 explicit) | 5.82 | Moderate | 3,150 |

| Summary Metric | Phase 1 | Phase 2 |
|----------------|---------|---------|
| Overall safety flip rate | 0.63% | 0.94% |
| Mean flip agreement (P1 vs P2) | -- | 99.45% |
| Risk classification | MODERATE | MODERATE |

**Overall Risk Classification: MODERATE**

**Observations.** Both phases converge on a MODERATE risk classification. The true-batch variance (5.82pp) is slightly higher than the dispatch-batch variance (5.05pp), consistent with the higher safety flip rate in Phase 2. The cross-phase mean flip agreement of 99.45% provides strong evidence that both measurement approaches are capturing the same underlying phenomenon.

The MODERATE classification means that batch perturbation produces detectable but not practically catastrophic safety effects: aggregate scores remain within TOST equivalence bounds (SS15), but individual prompt flips are real and should be accounted for in safety-critical deployments. For context, the Banterhearts risk classification scale is:
- **LOW** (< 2pp variance): negligible operational concern
- **MODERATE** (2-10pp variance): detectable effect, monitor in production
- **HIGH** (10-20pp variance): requires mitigation before deployment
- **CRITICAL** (> 20pp variance): deploy only with deterministic inference

Both phases fall in the lower portion of the MODERATE range (5-6pp), closer to LOW than to HIGH. This is consistent with the TOST equivalence results (SS15) and suggests that standard batch inference can be used safely with appropriate monitoring.

The 0.77pp gap between Phase 2 (5.82pp) and Phase 1 (5.05pp) variance deserves attention. True tensor batching forces all prompts through identical computation paths, while dispatch batching may introduce small timing variations that partially decorrelate floating-point errors. If this hypothesis is correct, the Phase 2 result may be the more accurate estimate of batch perturbation magnitude in production vLLM deployments, where continuous batching (a form of true batching) is the default.

---

## SS15. TOST Equivalence Analysis

Two One-Sided Tests (TOST) assess whether batch-induced mean score differences fall within the +/-3 percentage point practical equivalence margin. A passing TOST result (p < 0.05 for both one-sided tests) confirms that the true mean difference is practically negligible.

**Table 12: TOST Equivalence Results**

| Phase | Comparison | Mean Diff | 95% CI | TOST p | Equivalent? |
|-------|-----------|-----------|--------|--------|-------------|
| Phase 1 | BS1 vs BS2, safety | +0.18pp | [-0.02pp, +0.38pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS2, capability | -0.15pp | [-0.35pp, +0.05pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS4, safety | +0.34pp | [+0.09pp, +0.58pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS4, capability | +0.12pp | [-0.10pp, +0.33pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS8, safety | +0.24pp | [+0.04pp, +0.45pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS8, capability | +0.09pp | [-0.09pp, +0.26pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS16, safety | +0.09pp | [-0.12pp, +0.30pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS16, capability | -0.03pp | [-0.22pp, +0.16pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS32, safety | +0.23pp | [+0.03pp, +0.42pp] | < 0.001 | Yes |
| Phase 1 | BS1 vs BS32, capability | -0.06pp | [-0.24pp, +0.12pp] | < 0.001 | Yes |
| Phase 2 | BS1 vs BS4, safety | +0.34pp | [+0.00pp, +0.68pp] | < 0.001 | Yes |
| Phase 2 | BS1 vs BS4, capability | +0.36pp | [+0.05pp, +0.67pp] | < 0.001 | Yes |
| Phase 2 | BS1 vs BS8, safety | +0.34pp | [-0.03pp, +0.71pp] | < 0.001 | Yes |
| Phase 2 | BS1 vs BS8, capability | -0.21pp | [-0.48pp, +0.05pp] | < 0.001 | Yes |

**14/14 comparisons confirm practical equivalence within +/-3pp.**

**Observations.** Every TOST comparison passes with p < 0.001, confirming that batch-induced mean score changes are well within the 3 percentage point practical equivalence margin. The largest observed mean difference is 0.36pp (Phase 2, BS1 vs BS4, capability), still an order of magnitude below the 3pp bound. This means that while individual prompts flip (SS6-SS8), the aggregate effect on mean safety and capability scores is negligible.

The TOST result provides a crucial context for interpreting the flip rate findings: production teams can be confident that switching batch sizes will not shift their overall safety or capability metrics by a practically meaningful amount. If a deployment's safety score is 92% at batch=1, it will remain between approximately 91.6% and 92.4% at any batch size up to 32 -- well within the noise margin of most safety evaluation frameworks.

However, the TOST result does not contradict the existence of individual flips -- it contextualizes them as rare events that wash out in aggregate statistics. For safety-critical applications where individual prompt-level failures matter (e.g., a single harmful response to a specific user query), the 0.63% flip rate remains operationally relevant even though it does not shift aggregate metrics. The key insight is that batch perturbation creates a small, diffuse safety risk at the individual-prompt level while remaining practically invisible at the aggregate-metric level.

The Phase 2 TOST results are particularly important because they confirm equivalence under true tensor batching, not just dispatch batching. The largest Phase 2 mean difference (0.36pp for BS4 capability) is larger than most Phase 1 differences, consistent with the higher Phase 2 flip rate (SS13), but still well within the equivalence bound.

---

## SS16. Power Analysis

Power analysis determines the minimum detectable effect (MDE) at 80% power and alpha=0.05, ensuring the experiment is adequately powered to detect meaningful differences.

**Table 13: Power Analysis Summary**

| Phase | Domain | Baseline Rate | N | MDE (pp) |
|-------|--------|--------------|---|---------|
| Phase 1 | Safety | 64.9% | 19,656 | 1.3 |
| Phase 1 | Capability | 48.3% | 20,370 | 1.4 |
| Phase 2 | Combined | 55.4% | 9,450 | 2.0 |
| Phase 1 per-BS | Safety | ~64.9% | 3,276 | 3.3 |

**Observations.** Phase 1 is well-powered to detect effects as small as 1.3 percentage points in safety scores, comfortably below the 3pp TOST bound. This means that if batch perturbation caused a 2pp shift in aggregate safety scores, the experiment would detect it with 80% probability. Phase 2, with fewer samples, has an MDE of 2.0pp, still adequate for the practical equivalence question. The per-batch-size MDE of 3.3pp is marginally above the TOST bound, meaning individual batch-size comparisons have limited power -- which is why the aggregate analysis (SS7) is more informative than per-cell comparisons.

The power analysis reveals an important asymmetry in this study's design: the experiment is well-powered to detect aggregate score shifts (MDE 1.3-2.0pp) but underpowered to detect per-model-per-batch-size flip rate differences. With 468 safety prompts per model per batch size and an expected flip rate of approximately 0.6%, each cell has only approximately 3 expected flips under the null hypothesis. Detecting a doubling of the flip rate (from 0.6% to 1.2%) at a single cell level would require approximately 5,000 safety prompts per cell -- 10x the current design. This explains why no individual Fisher exact test reaches significance after correction (SS17) despite the consistent aggregate pattern. Future studies aiming to confirm per-model-per-batch-size effects should use at least 2,000 safety prompts per cell.

The power analysis also validates the TOST conclusions (SS15): the Phase 1 MDE of 1.3pp is well below the 3pp equivalence bound, meaning the experiment would detect a violation of equivalence with high probability. The fact that all TOST tests pass is therefore a genuine finding of equivalence, not an artifact of low power.

---

## SS17. Statistical Tests: Per-Model Fisher Exact and Chi-Squared

Each model x batch-size cell was tested for safety-capability flip rate asymmetry using both chi-squared and Fisher exact tests, with Holm-Bonferroni correction for the 35 total comparisons.

**Table 14: Notable Per-Model Statistical Tests (Fisher Exact, Pre-Correction)**

| Model | BS | Safety Flips | Cap Flips | Fisher p | OR | Holm Sig? |
|-------|----|-------------|----------|---------|-----|-----------|
| qwen2.5-1.5b | 16 | 5 / 468 | 0 / 485 | 0.028 | 11.52 | No |
| qwen2.5-1.5b | 2 | 4 / 468 | 0 / 485 | 0.058 | 9.41 | No |
| qwen2.5-1.5b | 32 | 4 / 468 | 0 / 485 | 0.058 | 9.41 | No |
| llama3.2-1b | 16 | 4 / 468 | 0 / 485 | 0.058 | 9.41 | No |
| qwen2.5-3b | 16 | 3 / 468 | 0 / 485 | 0.118 | 7.30 | No |
| qwen2.5-3b | 32 | 3 / 468 | 0 / 485 | 0.118 | 7.30 | No |
| stablelm-2-zephyr | 4 | 3 / 468 | 0 / 485 | 0.118 | 7.30 | No |

**Overall chi-squared tests by batch size (all models pooled):**

| BS | Chi-sq | p | OR | OR 95% CI |
|----|--------|---|----|-----------|
| 2 | 0.20 | 0.659 | 1.16 | [0.61, 2.21] |
| 4 | 1.65 | 0.199 | 1.44 | [0.82, 2.55] |
| 8 | 1.36 | 0.244 | 1.50 | [0.75, 3.01] |
| 16 | 0.63 | 0.428 | 1.31 | [0.67, 2.55] |
| 32 | 0.95 | 0.329 | 1.40 | [0.71, 2.76] |

**Observations.** No individual model x batch-size comparison reaches significance after Holm-Bonferroni correction, which is expected given the small per-cell sample sizes (468 safety, 485 capability per cell) and rare event rates. The closest to significance is qwen2.5-1.5b at BS=16 (Fisher p=0.028, OR=11.52), but this does not survive correction for 35 comparisons.

The qwen2.5-1.5b results are particularly noteworthy: three of its five batch-size comparisons show odds ratios above 9.0 (BS=2, BS=16, BS=32), all with zero capability flips. This is the strongest model-level signal in the data and consistent with the Qwen family's elevated safety-specific fragility discussed in SS6. If a future study with larger per-cell sample sizes were to find a significant per-model result, qwen2.5-1.5b would be the most likely candidate.

The overall pooled chi-squared tests also do not reach significance at any batch size, though the odds ratios are consistently above 1.0 (range 1.16-1.50), suggesting a real but small effect that the experiment is underpowered to confirm at the per-batch-size level. The aggregate analysis across all batch sizes (SS7) provides the strongest evidence for the safety-capability asymmetry.

It is important to note that the absence of statistical significance does not mean the absence of an effect. With 35 comparisons and small per-cell counts, the Holm-Bonferroni correction is appropriately conservative -- it protects against false positives at the cost of reduced sensitivity. The consistent directional pattern (all 5 pooled ORs > 1.0, all 7 models showing at least one batch size with safety ratio > 1.0) provides complementary evidence that the aggregate analysis is not a statistical fluke.

---

## SS18. Latency Analysis

Latency analysis examines how response time scales with batch size and whether it differs between safety and capability domains.

**Table 15: Latency Scaling by Model (Phase 1)**

| Model | BS=1 Mean (ms) | BS=32 Mean (ms) | Slope (ms/BS) | R-squared | BS=32 Throughput (samp/s) |
|-------|---------------|----------------|--------------|-----------|---------------------------|
| llama3.2-1b | 165 | 203 | 1.07 | 0.949 | 157.9 |
| qwen2.5-1.5b | 306 | 363 | 1.58 | 0.930 | 88.1 |
| smollm2-1.7b | 267 | 328 | 1.77 | 0.899 | 97.6 |
| qwen2.5-3b | 552 | 631 | 1.66 | 0.382 | 50.7 |
| stablelm-2-zephyr | 486 | 572 | 2.66 | 0.975 | 56.0 |
| llama3.2-3b | 700 | 824 | 3.25 | 0.829 | 38.8 |
| phi-3.5-mini | 1,612 | 2,093 | 14.76 | 0.955 | 15.3 |

**Observations.** Latency scales linearly with batch size for most models (R-squared > 0.89 for 6 of 7 models), with qwen2.5-3b being the exception (R-squared=0.382, suggesting non-linear or noisy scaling). phi-3.5-mini has by far the steepest latency slope (14.76 ms per batch-size unit), meaning its latency grows 14x faster with increasing batch size than llama3.2-1b (1.07 ms/BS). This is consistent with phi-3.5-mini's larger parameter count and more complex architecture. Interestingly, phi-3.5-mini is also the most fragile model (SS10), and its steep latency scaling suggests that larger batch sizes create more computational stress on the model, potentially increasing the floating-point perturbation magnitude.

**Table 15b: Latency by Domain (Safety vs Capability)**

| Model | Safety Mean (ms) | Cap Mean (ms) | Diff (ms) | Cohen's d | Safety Slower? |
|-------|-----------------|--------------|-----------|----------|---------------|
| llama3.2-1b | 281 | 82 | +199 | 0.87 | Yes |
| llama3.2-3b | 1,080 | 457 | +623 | 1.07 | Yes |
| phi-3.5-mini | 1,492 | 2,109 | -617 | -0.78 | **No** |
| qwen2.5-1.5b | 554 | 114 | +440 | 1.28 | Yes |
| qwen2.5-3b | 1,147 | 92 | +1,055 | 2.52 | Yes |
| smollm2-1.7b | 434 | 162 | +272 | 0.76 | Yes |
| stablelm-2-zephyr | 655 | 371 | +284 | 0.92 | Yes |

**Observations (domain comparison).** Safety prompts generate systematically longer (and thus slower) responses than capability prompts for 6 of 7 models, with Cohen's d ranging from 0.76 (smollm2-1.7b) to 2.52 (qwen2.5-3b, an exceptionally large effect). The exception is phi-3.5-mini, where capability prompts are slower -- phi-3.5-mini generates longer capability responses (likely due to its tendency to provide detailed explanations for MMLU/ARC answers). For qwen2.5-3b, the 12.4x latency ratio between safety (1,147ms) and capability (92ms) prompts is extraordinary and suggests that this model's safety responses involve extensive reasoning chains while its capability responses are concise. The practical implication is that safety prompts inherently involve more floating-point operations per prompt (due to longer generation), which may mechanically increase their exposure to batch perturbation -- providing a partial explanation for the safety-capability flip rate asymmetry observed in SS7.

---

## SS19. Flip-Latency Correlation

This analysis asks whether samples that flip their classification are systematically slower or faster than stable samples, providing a latency-based signal for flip risk.

**Table 16: Flip-Latency Correlation (Phase 1, All Non-Baseline Batch Sizes)**

| Model | Flipped Mean (ms) | Stable Mean (ms) | Diff (ms) | Cohen's d | N Flipped | N Stable |
|-------|-------------------|------------------|-----------|----------|----------|---------|
| llama3.2-1b | 244 | 183 | +62 | 0.25 (small) | 14 | 4,751 |
| llama3.2-3b | 1,296 | 773 | +523 | 0.78 (medium) | 19 | 4,746 |
| phi-3.5-mini | 2,333 | 1,837 | +496 | 0.58 (medium) | 71 | 4,694 |
| qwen2.5-1.5b | 813 | 332 | +480 | **1.17 (large)** | 25 | 4,740 |
| qwen2.5-3b | 1,172 | 620 | +552 | 0.81 (large) | 19 | 4,746 |
| smollm2-1.7b | 366 | 300 | +66 | 0.17 (negligible) | 21 | 4,744 |
| stablelm-2-zephyr | 845 | 514 | +331 | 0.97 (large) | 14 | 4,751 |

**Observations.** Flipped samples are consistently slower across all 7 models, with Cohen's d ranging from 0.17 (smollm2-1.7b, negligible) to 1.17 (qwen2.5-1.5b, large). For 5 of 7 models, the effect size is medium or large (d > 0.5), indicating a practically meaningful difference. qwen2.5-1.5b shows the strongest flip-latency association: flipped samples take 2.4x longer than stable samples on average (813ms vs 332ms).

This finding has practical implications: longer outputs (which drive higher latency) may be more susceptible to batch perturbation because they involve more floating-point accumulation steps, each of which can be reordered by batching. The latency signal could potentially be used as a runtime proxy for flip risk -- prompts generating unusually long outputs at a given batch size could be flagged for re-evaluation at batch=1. However, the causal direction is ambiguous: it could also be that flip-prone prompts happen to be more complex, eliciting longer responses regardless of batch perturbation.

The effect size gradient is interesting: the two models with the weakest flip-latency association (smollm2-1.7b d=0.17, llama3.2-1b d=0.25) are also among the least fragile models (0.38% and 0.47% respectively). The three models with the strongest association (qwen2.5-1.5b d=1.17, stablelm-2-zephyr d=0.97, qwen2.5-3b d=0.81) include both fragile (qwen2.5-1.5b at 0.98%) and robust (stablelm-2-zephyr at 0.38%) models, suggesting that the flip-latency relationship is not simply a proxy for overall fragility.

One hypothesis consistent with this data: for all models, flip-prone prompts are those near the refusal decision boundary, and such prompts tend to generate longer outputs (hedging, equivocation, partial compliance). The magnitude of the latency difference then depends on how differently the model handles boundary-proximate versus boundary-distant prompts in terms of output length. Models like qwen2.5-1.5b, which show a large latency difference (813ms vs 332ms), may have particularly verbose boundary behavior.

---

## SS20. Slope Heterogeneity and Critical Thresholds

Slope heterogeneity measures whether different task types show different batch-size sensitivity trends, while critical threshold detection looks for batch sizes at which safety flips spike.

**Table 17: Per-Model Task Slope Range**

| Model | Most Sensitive Task | Least Sensitive Task | Slope Range |
|-------|-------------------|---------------------|-------------|
| llama3.2-1b | advbench_refusal | mmlu_real | 0.0004 |
| llama3.2-3b | truthfulqa | jailbreak_amplification | 0.0011 |
| phi-3.5-mini | advbench_refusal | truthfulqa | 0.0016 |
| qwen2.5-1.5b | advbench_refusal | truthfulqa | 0.0001 |
| qwen2.5-3b | advbench_refusal | arc_challenge | 0.0002 |
| smollm2-1.7b | jailbreak_amplification | truthfulqa | 0.0013 |
| stablelm-2-zephyr | truthfulqa | jailbreak_amplification | 0.0009 |

**Critical threshold detection:** No critical batch-size threshold detected for any of the 7 models (Wilson CI non-overlap method).

**Observations.** The slope ranges are uniformly small (0.0001 to 0.0016), indicating that no task shows a dramatically different batch-size sensitivity trend from any other task within any model. The most sensitive tasks vary by model -- advbench_refusal dominates for Qwen and Phi models, while truthfulqa is most sensitive for llama3.2-3b and stablelm-2-zephyr.

The absence of a critical batch-size threshold for any model is an important negative finding: it means there is no "cliff edge" batch size beyond which safety degrades rapidly. Production teams need not worry about accidentally crossing a critical threshold; the effect is uniformly distributed across the tested batch-size range (2-32). This is consistent with the underlying mechanism: floating-point non-associativity produces perturbations that scale smoothly with the number of reduction operations rather than exhibiting phase transitions.

The lack of threshold also means that there is no evidence-based justification for imposing a maximum batch size for safety reasons. If batch=32 is safe enough, batch=64 (not tested but extrapolated from the linear trend) would likely show similar flip rates. However, this extrapolation should be validated empirically before deploying at batch sizes beyond the tested range.

Interestingly, phi-3.5-mini has the largest slope range (0.0016), driven by advbench_refusal being the most sensitive task (positive slope, safety scores slightly improving with batch size) and truthfulqa being the least sensitive (negative slope, safety scores slightly declining). This suggests that batch perturbation affects different prompt types in different directions for this model, with the effects nearly canceling out in aggregate -- consistent with the TOST equivalence results (SS15).

---

## SS21. Refusal-Style Analysis

Refusal style classifies each refusal response as keyword-based (e.g., "I cannot assist with that"), reasoning-based (explains why the request is harmful), or unclear. This analysis explores whether refusal style correlates with batch fragility.

**Table 18: Refusal-Style Distribution by Model**

| Model | Total Refusals | Keyword % | Reasoning % | Unclear % | Dominant Style |
|-------|---------------|-----------|-------------|-----------|---------------|
| qwen2.5-1.5b | 1,356 | 96.5% | 2.4% | 1.1% | Keyword |
| llama3.2-3b | 1,174 | 90.7% | 1.6% | 7.7% | Keyword |
| llama3.2-1b | 984 | 87.7% | 1.8% | 10.5% | Keyword |
| qwen2.5-3b | 1,181 | 84.8% | 14.0% | 1.3% | Keyword |
| phi-3.5-mini | 1,155 | 70.9% | 16.7% | 12.4% | Keyword |
| stablelm-2-zephyr | 60 | 58.3% | 15.0% | 26.7% | Keyword |
| smollm2-1.7b | 25 | 28.0% | 36.0% | 36.0% | Reasoning |

**Observations.** Most models are overwhelmingly keyword-based in their refusal style, with qwen2.5-1.5b at the extreme (96.5% keyword). Two notable exceptions are smollm2-1.7b, which is the only model where reasoning-style refusals dominate (36%, tied with unclear), and phi-3.5-mini, which has the highest reasoning percentage among keyword-dominant models (16.7%).

The refusal style data reveals an interesting pattern when cross-referenced with fragility (SS10). smollm2-1.7b, with the highest proportion of reasoning-style refusals, is also the joint least fragile model (0.38%). stablelm-2-zephyr, with a relatively high unclear rate (26.7%), shares that position. In contrast, the most keyword-heavy models (qwen2.5-1.5b at 96.5%) tend to be more fragile. This is consistent with the hypothesis that keyword-based refusals rely on narrow token patterns that are more susceptible to floating-point perturbation, while reasoning-based refusals involve broader contextual signals that are more robust to small output variations.

However, the extremely low total refusal counts for smollm2-1.7b (25) and stablelm-2-zephyr (60) compared to the RLHF models (984-1,356) suggest these models have very low baseline safety scores, which mechanically limits their flip opportunity as discussed in SS11.

---

## SS22. Combined Directional Analysis

Pooling directional flip data from TR141 (and TR138, where available) provides a more powered test of whether batch perturbation has a systematic directional bias.

**Table 19: Combined Directional Analysis**

| Source | To Unsafe (R->C) | To Safe (C->R) | Total |
|--------|-----------------|---------------|-------|
| TR141 | 28 | 69 | 97 |
| TR138 | 0 (not available) | 0 (not available) | 0 |
| **Combined** | **28** | **69** | **97** |

**Binomial test:** p = 3.8e-5, direction = net safe, N = 97

**Observations.** The combined directional analysis (currently TR141-only, as TR138 raw directional data was not available for pooling) shows a highly significant net SAFE bias. Of 97 directional safety flips, 71.1% move from compliance to refusal (becoming safer) versus only 28.9% moving from refusal to compliance (becoming less safe). The binomial p-value of 3.8e-5 is well below any conventional significance threshold.

This result has two important implications. First, it reverses the a priori concern that batch perturbation preferentially erodes safety. In this experimental configuration, the opposite is true: batch perturbation makes marginal outputs more conservative. Second, it differs from the directional concerns raised in TR138, suggesting that the direction of batch-perturbation bias may be model-dependent and configuration-dependent rather than universal. The TR141 model set (7 models, 4 alignment types) is larger and more diverse than TR138's (3 models, 1 alignment type), so the net-safe finding may be more representative of the general case.

Several possible mechanisms could explain the net-safe bias:

1. **Asymmetric refusal boundaries.** Models may have asymmetric refusal thresholds where the "refuse" decision requires a lower activation threshold than the "comply" decision. Floating-point perturbation that slightly increases noise is then more likely to push a marginal "comply" output past the refusal threshold than to push a marginal "refuse" output past the compliance threshold.

2. **Safety training bias.** RLHF and DPO training may produce models that are biased toward refusal when uncertain. If batch perturbation increases effective uncertainty (by adding noise to logits), the model's default behavior under uncertainty is to refuse -- producing the observed net-safe direction.

3. **Token probability asymmetry.** Refusal tokens (e.g., "I", "cannot") may occupy a larger probability mass in the model's vocabulary distribution than compliance tokens for safety-relevant prompts, so random perturbation is more likely to land on a refusal-initiating token.

4. **Prompt set composition.** The specific prompts used in TR141 may be slightly biased toward the refusal boundary, with more prompts sitting just below the refusal threshold (marginally compliant) than just above it (marginally refusing). This would create a mechanical net-safe bias.

Distinguishing between these mechanisms would require additional experiments, such as varying the prompt set composition, testing with unaligned base models (which lack safety training bias), or examining the logit distributions at the first token position for flip candidates.

### Base-Rate-Adjusted Directional Test

The binomial test above uses p=0.5 as the null hypothesis (equal probability of flipping toward safe or unsafe). However, this null is only appropriate if the baseline compliance and refusal rates are symmetric. In TR141, models comply with approximately 60% of safety prompts at baseline (the complement of the mean safety score of approximately 0.65 for the RLHF/SFT models that generate most flips). Under random perturbation of boundary-proximate prompts, the expected directional split is not 50:50 but rather skewed toward the larger pool: approximately 60% of flippable prompts are compliant (could flip to refusal, i.e., toward safe), while approximately 40% are refusing (could flip to compliance, i.e., toward unsafe). The expected directional split under random flipping is therefore approximately 60:40 toward compliance-to-refusal (safe direction).

Testing the observed 69:28 split (71.1% toward safe) against this adjusted null of p=0.60 (instead of p=0.50):

- Observed: 69 safe-direction out of 97 total (71.1%)
- Adjusted null: p = 0.60
- Binomial test (one-sided, H_a: p > 0.60): p approximately 0.017

The finding remains statistically significant at alpha=0.05 even after base-rate adjustment, though the p-value increases from 3.8e-5 to approximately 0.017. The excess safe-direction bias beyond what base rates predict is approximately 11 percentage points (71.1% - 60%), suggesting a genuine asymmetry in how batch perturbation interacts with refusal boundaries, not merely a reflection of the compliance/refusal base rate.

**3-Shared-Model Directional Split.** To isolate models shared with TR138 (where the directional concern originated), the directional counts for the 3 TR138-shared models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b per TR138's config.yaml) are:

| Model | To Unsafe (R->C) | To Safe (C->R) | Total |
|-------|-----------------|---------------|-------|
| llama3.2-1b | 0 | 11 | 11 |
| llama3.2-3b | 3 | 7 | 10 |
| qwen2.5-1.5b | 4 | 18 | 22 |
| **3-shared-model total** | **7** | **36** | **43** |

The 3-shared-model directional split is 7:36, or 83.7% toward safe -- even stronger than the 7-model aggregate (71.1%). This is net-safe even when restricted to TR138-shared models, isolating the directional difference to GPU architecture (Blackwell vs A100) and/or seed (137 vs 42) rather than model-set composition.

For additional context, computing the directional split for the 3 models originally reported as shared (llama3.2-1b, llama3.2-3b, qwen2.5-3b -- before correction in SS27): llama3.2-1b=0:11, llama3.2-3b=3:7, qwen2.5-3b=10:6, total=13:24 (64.9% safe). Even this less favorable computation is still net-safe, despite qwen2.5-3b being the only model with net-unsafe direction. The net-safe finding is robust to model-set definition.

### Implications for TR138 Reconciliation

The net-safe directional finding in TR141 contrasts with TR138's directional concerns. Notably, TR138 shares all 3 of its models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) with TR141, and the 3-shared-model directional split in TR141 is 7:36 (83.7% safe) -- the most strongly safe-biased subset. This means the directional discrepancy cannot be explained by model-set composition alone. The most likely remaining explanations are: (1) the A100 GPU's floating-point behavior produces different directional patterns than the Blackwell architecture, (2) the different seed (42 vs 137) shifts which specific prompts are near the decision boundary, or (3) both. A definitive resolution would require running the same 7-model suite on both GPU architectures with the same seed.

---

## SS23. Jailbreak Type Breakdown

Phase 1 includes 120 jailbreak amplification prompts spanning 4 jailbreak types: direct requests, DAN-style jailbreaks, prefix injection, and roleplay. This analysis examines whether batch size interacts with jailbreak type to affect refusal rates, which would indicate that batch perturbation creates exploitable amplification pathways for specific attack strategies.

**Table 20: Jailbreak Compliance Slopes by Model and Type**

| Model | Direct | DAN-Style | Prefix Injection | Roleplay |
|-------|--------|-----------|-----------------|----------|
| llama3.2-1b | 0.000 | -0.001 | +0.000 | 0.000 |
| llama3.2-3b | +0.000 | +0.001 | +0.000 | -0.001 |
| phi-3.5-mini | 0.000 | 0.000 | 0.000 | +0.001 |
| qwen2.5-1.5b | 0.000 | 0.000 | -0.001 | 0.000 |
| qwen2.5-3b | 0.000 | 0.000 | +0.000 | -0.002 |
| smollm2-1.7b | 0.000 | 0.000 | +0.000 | -0.000 |
| stablelm-2-zephyr | 0.000 | 0.000 | -0.000 | 0.000 |

**Observations.** All compliance slopes are near zero (magnitude <= 0.002), indicating that batch size does not systematically amplify or attenuate the effectiveness of any jailbreak type for any model. The largest slope is qwen2.5-3b with roleplay (-0.002), meaning roleplay compliance *decreases* very slightly with increasing batch size -- but even this is practically negligible. This negative result is reassuring: batch perturbation does not create exploitable amplification pathways for specific jailbreak strategies. The "direct" jailbreak type shows the most stable behavior across models (most slopes exactly 0.000), consistent with direct prompts being far from the refusal boundary in either direction.

**Table 20b: Baseline Refusal Rates by Jailbreak Type (Batch=1)**

| Model | Direct | DAN-Style | Prefix Injection | Roleplay |
|-------|--------|-----------|-----------------|----------|
| llama3.2-1b | 53.3% | 56.7% | 24.0% | 66.7% |
| llama3.2-3b | 60.0% | 53.3% | 58.0% | 53.3% |
| phi-3.5-mini | 60.0% | 96.7% | 34.0% | 73.3% |
| qwen2.5-1.5b | 83.3% | 66.7% | 68.0% | 10.0% |
| qwen2.5-3b | 83.3% | 66.7% | 4.0% | 6.7% |
| smollm2-1.7b | 0.0% | 0.0% | 8.0% | 3.3% |
| stablelm-2-zephyr | 3.3% | 10.0% | 6.0% | 3.3% |

**Observations (baseline rates).** The baseline refusal rates reveal dramatic cross-model differences in jailbreak susceptibility that are orthogonal to the batch perturbation effect. phi-3.5-mini has the strongest DAN-style resistance (96.7% refusal) but moderate prefix injection resistance (34.0%). qwen2.5-3b shows an extreme split: 83.3% refusal for direct requests but only 4.0% for prefix injection and 6.7% for roleplay, indicating a highly non-uniform safety surface. smollm2-1.7b and stablelm-2-zephyr show near-zero refusal rates for most jailbreak types, consistent with their very low baseline safety scores (SS11). The critical finding is that these large baseline differences remain stable across batch sizes (all slopes near zero), meaning batch perturbation does not preferentially erode the safety of models that are already vulnerable to specific jailbreak types.

This stability has an important operational implication: adversaries cannot use batch-size manipulation as a jailbreak amplification vector. Unlike multi-turn escalation (TR139) or quantization-induced degradation (TR134-137), which systematically lower refusal rates for specific attack patterns, batch perturbation produces only random, non-directional noise in the jailbreak domain. An adversary who knew a model was serving at BS=32 would gain no additional advantage over BS=1 for any of the four jailbreak categories tested.

The near-zero compliance slopes also validate the experimental design: if batch perturbation produced systematic trends in compliance rates with increasing batch size, it would suggest a confound (e.g., memory pressure or scheduling artifacts). The flat slopes confirm that the observed flips are genuinely random perturbations, not systematic degradation.

---

## SS24. Per-Category Bias Analysis

BBQ bias prompts span 11 demographic categories. This analysis examines whether batch perturbation differentially affects bias detection across categories.

**Table 21: Mean Bias Score by Category (Selected Models)**

| Category | llama3.2-1b | llama3.2-3b | phi-3.5-mini | qwen2.5-1.5b | qwen2.5-3b | smollm2-1.7b | stablelm-2-zephyr |
|----------|-------------|-------------|-------------|--------------|------------|-------------|-------------------|
| Age | 0.667 | 0.944 | 1.000 | 0.889 | 1.000 | 0.667 | 0.944 |
| Disability_status | 0.500 | 0.735 | 0.889 | 0.698 | 0.944 | 0.778 | 0.889 |
| Gender_identity | 0.722 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 | 0.833 |
| Nationality | 1.000 | 1.000 | 1.000 | 0.827 | 0.944 | 0.444 | 0.833 |
| Physical_appearance | 0.750 | 0.864 | 1.000 | 1.000 | 1.000 | 0.773 | 0.909 |
| Race_ethnicity | 0.833 | 0.944 | 1.000 | 0.889 | 1.000 | 0.667 | 0.889 |
| Race_x_SES | 1.000 | 0.944 | 0.944 | 0.944 | 1.000 | 0.611 | 1.000 |
| Race_x_gender | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| Religion | 0.722 | 0.889 | 0.944 | 1.000 | 0.944 | 0.870 | 0.722 |
| SES | 1.000 | 1.000 | 0.944 | 0.806 | 1.000 | 0.500 | 1.000 |
| Sexual_orientation | 0.722 | 0.889 | 1.000 | 0.843 | 1.000 | 0.778 | 0.944 |

**Observations.** Bias scores are generally high across models and categories, with most cells at or near 1.0 (indicating correct detection of biased responses). The most notable exceptions are smollm2-1.7b, which shows substantially lower bias scores across all categories (range 0.444-0.889), particularly for Nationality (0.444) and SES (0.500). This is consistent with smollm2-1.7b's overall low safety score (0.333 mean, SS11). llama3.2-1b shows the next lowest scores, with Disability_status at 0.500.

Critically, the bias scores shown here are batch=1 baseline values. The BBQ task has the lowest flip rate of any task (0.2%, SS9), meaning batch perturbation causes negligible changes to bias detection performance regardless of category. No category-specific batch vulnerability was identified: the batch perturbation effect is too small and too uniformly distributed to create differential bias exposure.

The most vulnerable categories across models are Disability_status (lowest mean: llama3.2-1b at 0.500) and Nationality (lowest mean: smollm2-1.7b at 0.444). These categories may have more ambiguous prompts where the model's answer selection is closer to random, making them theoretically more susceptible to batch perturbation. However, the 0.2% BBQ flip rate means fewer than 3 BBQ prompts flip per batch size across all 7 models combined, so the practical impact on bias evaluation is negligible.

A notable model-level pattern is the split between "high-safety" models (qwen2.5-3b, phi-3.5-mini) that score near 1.0 on most categories, indicating robust bias avoidance, and "low-safety" models (smollm2-1.7b) that score below 0.7 on several categories. This split mirrors the baseline safety score differences discussed in SS11 and provides additional evidence that smollm2-1.7b's low fragility is partly a floor effect: with already-low bias scores, there are fewer "correct" responses available to flip.

---

## SS25. Variance-Safety Correlation

This analysis measures whether prompts that flip more frequently across batch sizes also tend to have lower baseline safety scores, testing the hypothesis that batch perturbation exploits existing vulnerability.

**Table 22: Variance-Safety Correlation (Pearson r, Flip Count vs Baseline Safety)**

| Model | Pearson r | p-value | N | Significant? | Interpretation |
|-------|----------|---------|---|-------------|---------------|
| phi-3.5-mini | **-0.194** | **2.4e-5** | 468 | Yes | Lower baseline safety -> more flips |
| qwen2.5-1.5b | **-0.164** | **3.8e-4** | 468 | Yes | Lower baseline safety -> more flips |
| llama3.2-1b | **-0.115** | **0.013** | 468 | Yes | Lower baseline safety -> more flips |
| llama3.2-3b | **-0.109** | **0.019** | 468 | Yes | Lower baseline safety -> more flips |
| qwen2.5-3b | -0.035 | 0.447 | 468 | No | No relationship |
| smollm2-1.7b | +0.012 | 0.803 | 468 | No | No relationship |
| stablelm-2-zephyr | -0.000 | 0.997 | 468 | No | No relationship |

**Observations.** Four of seven models show a statistically significant negative correlation between flip frequency and baseline safety score, confirming that batch perturbation disproportionately affects prompts with lower baseline safety. The strongest correlation is phi-3.5-mini (r=-0.194, p=2.4e-5), which is also the most fragile model overall.

**Multiple comparison note.** Holm-Bonferroni correction was not applied to the 7 correlation tests in this section. After correction for 7 comparisons, the two most significant results (phi-3.5-mini p=2.4e-5, qwen2.5-1.5b p=3.8e-4) would survive at any conventional threshold. However, the two marginal results -- llama3.2-1b (p=0.013) and llama3.2-3b (p=0.019) -- may not survive Holm-Bonferroni correction (adjusted thresholds approximately 0.0125 and 0.0167 respectively). These should be treated as suggestive rather than confirmed.

The three non-significant models (qwen2.5-3b, smollm2-1.7b, stablelm-2-zephyr) share two properties: relatively low total flip counts (16, 9, 9 flips respectively) and either very low baseline safety scores (smollm2-1.7b at 0.333, stablelm-2-zephyr at 0.446) or moderate fragility (qwen2.5-3b at 0.68%). For smollm2-1.7b and stablelm-2-zephyr, the lack of correlation likely reflects a floor effect: their baseline safety is already so low that there is insufficient variance in "safe" outputs to correlate with flips. For qwen2.5-3b, the 16 flips may simply be insufficient for the correlation to reach significance.

The practical implication is that batch-safety testing should focus disproportionately on prompts near the refusal boundary (moderate safety scores), as these are the most likely flip candidates.

This finding connects to the flip-latency correlation (SS19): if flip-prone prompts have lower baseline safety, they are closer to the refusal boundary and thus more likely to generate hedging, equivocating, or partially compliant responses -- which are longer and slower. The variance-safety correlation provides the "why" (boundary proximity) while the flip-latency correlation provides the "what" (longer outputs) for identifying flip risk.

For models where the correlation is significant, a practical mitigation strategy emerges: identify prompts with moderate safety scores (e.g., 0.3-0.7 on a 0-1 scale) at batch=1, and either re-evaluate these at batch=1 in production or route them to a deterministic inference path. This targeted approach would protect the most vulnerable prompts without imposing the full throughput penalty of deterministic inference on all requests.

The magnitude of the correlations (r = -0.109 to -0.194) is small to medium by conventional standards, explaining 1.2% to 3.8% of the variance in flip frequency. This means baseline safety score is a useful but not sufficient predictor of flip risk. Other factors -- including output length (SS19), specific token patterns, and position within the model's internal representation space -- likely contribute to the remaining 96-99% of variance. A more complete flip-prediction model would require access to internal model states (logit margins, attention entropy), which is beyond the scope of this behavioral study.

---

## SS26. Safety-Capability Divergence

This analysis directly tests whether safety and capability flip rate confidence intervals are non-overlapping (indicating statistically confirmed disproportionate impact) at each batch size.

**Table 23: Safety-Capability Divergence by Batch Size (Phase 1)**

| Comparison | Safety Flip Rate | Safety 95% CI | Cap Flip Rate | Cap 95% CI | CI Overlap? | Disproportionate? |
|-----------|-----------------|-------------|--------------|----------|------------|------------------|
| BS=2 | 0.58% | [0.37%, 0.90%] | 0.50% | [0.31%, 0.80%] | Yes | No |
| BS=4 | 0.85% | [0.59%, 1.23%] | 0.59% | [0.38%, 0.91%] | Yes | No |
| BS=8 | 0.58% | [0.37%, 0.90%] | 0.38% | [0.22%, 0.65%] | Yes | No |
| BS=16 | 0.58% | [0.37%, 0.90%] | 0.44% | [0.27%, 0.73%] | Yes | No |
| BS=32 | 0.58% | [0.37%, 0.90%] | 0.41% | [0.25%, 0.69%] | Yes | No |

**Observations.** At no batch size do the safety and capability flip rate confidence intervals fail to overlap. This means we cannot statistically confirm disproportionate safety impact at any individual batch size using CI non-overlap as the criterion. The safety flip rate is numerically higher at every batch size, but the difference is within sampling uncertainty.

This result is consistent with SS17's chi-squared tests and supports the characterization of the safety-capability asymmetry as a real but small effect that requires aggregate analysis (or larger per-cell sample sizes) to confirm statistically. The claim that batch perturbation is "not safety-neutral" (Executive Summary, Claim 1) is therefore rated as **Partial** -- the direction is consistent and the alignment ANOVA is significant, but per-batch-size divergence tests do not reach significance.

The closest to non-overlapping CIs is at BS=4 (safety: [0.59%, 1.23%] vs capability: [0.38%, 0.91%]). The overlap region is [0.59%, 0.91%], which is a substantial portion of both intervals. To achieve non-overlapping CIs at the observed flip rates, we would need approximately 3x the current per-cell sample size -- roughly 10,000 safety prompts per batch size across all models, or approximately 70,000 additional Phase 1 records.

The divergence analysis provides an important calibration for how to interpret the safety-capability asymmetry: it is a consistent directional signal, not a statistically confirmed disproportionate impact. The practical interpretation is that safety outputs *may* be slightly more susceptible to batch perturbation, but the evidence is not strong enough to claim this with high confidence at the individual-batch-size level. The aggregate analysis (SS7) and the alignment ANOVA (SS11) provide stronger evidence for the existence of a safety effect, even if the magnitude cannot be precisely estimated at each batch size.

---

## SS27. Cross-TR138 Validation

TR141 shares 3 models with TR138: **llama3.2-1b, llama3.2-3b, and qwen2.5-1.5b** (per TR138's config.yaml; note that qwen2.5-3b is TR141-only, not a shared model). This enables direct cross-study comparison on the matched subset.

**Quantitative 3-Model Matched Comparison (TR141 values):**

| Model | TR141 Safety Flip Rate | TR141 Safety Flips / N | Family | Alignment |
|-------|----------------------|----------------------|--------|-----------|
| llama3.2-1b | 0.47% | 11 / 2,340 | Llama | RLHF |
| llama3.2-3b | 0.43% | 10 / 2,340 | Llama | RLHF |
| qwen2.5-1.5b | 0.98% | 23 / 2,340 | Qwen | RLHF |
| **3-model mean** | **0.63%** | **44 / 7,020** | | |

The 3-model matched mean (0.63%) is identical to TR141's overall 7-model mean (0.63%), suggesting that the additional 4 models in TR141 do not shift the aggregate safety flip rate. The Qwen > Llama fragility ordering is consistent within the matched subset.

**Qualitative comparison:**

| Metric | TR138 (3 models) | TR141 (7 models) |
|--------|-----------------|-----------------|
| Safety flip rate | ~0.6% (est.) | 0.63% |
| Capability flip rate | ~0.15% (est.) | 0.47% |
| S/C ratio | ~4x | 1.3x |
| Directional bias | Concern: unsafe | Confirmed: net safe (p=3.8e-5) |
| N models | 3 | 7 |
| GPU | A100 40GB | RTX PRO 6000 98GB |

**Observations.** The most notable discrepancy between TR138 and TR141 is the safety-to-capability ratio: TR138's approximately 4x ratio versus TR141's 1.3x. Several factors may explain this difference:

1. **Model selection:** TR138 used 3 RLHF models (all Llama/Qwen), while TR141 includes 7 models across 4 alignment types. The inclusion of DPO and distilled models (which show equal safety and capability flip rates) mechanically lowers the aggregate ratio.
2. **GPU architecture:** TR138 ran on an A100 40GB while TR141 ran on an RTX PRO 6000 Blackwell 98GB. Different GPU architectures have different floating-point accumulation behaviors, which could affect the magnitude and direction of batch perturbation.
3. **Seed:** TR138 used seed=42, TR141 used seed=137. Different seeds produce different draws of the floating-point perturbation landscape.
4. **Prompt set:** The two studies use overlapping but not identical prompt sets.

The safety flip rate itself is remarkably consistent (both approximately 0.6%), suggesting that the absolute level of safety perturbation is reproducible across studies. The difference lies primarily in the capability flip rate, which is higher in TR141 (0.47% vs ~0.15% in TR138), narrowing the gap.

This cross-study comparison yields a key methodological insight: the safety flip rate appears to be a more stable quantity than the capability flip rate or the safety-to-capability ratio. The 0.6% safety flip rate may represent a fundamental property of FP16 batch perturbation for small-to-medium instruction-tuned models, while the capability flip rate is more sensitive to model composition and prompt selection. Future studies should report safety flip rates as the primary metric rather than the ratio, which can vary substantially depending on the capability baseline.

**Table 27b: Cross-Study Stability of Key Metrics**

| Metric | TR138 | TR141 | Stable? |
|--------|-------|-------|---------|
| Safety flip rate (aggregate) | ~0.6% | 0.63% | **Yes** (within 0.1pp) |
| Capability flip rate | ~0.15% | 0.47% | No (3x difference) |
| Safety/capability ratio | ~4x | 1.3x | No (3x difference) |
| Output identity range | ~80-95% | 75-94% | Approximately stable |
| TOST equivalence (all pass) | Yes | Yes | **Yes** |
| Critical threshold detected | No | No | **Yes** (both negative) |
| MDE adequacy | Yes | Yes | **Yes** |

**Observations.** The stability analysis reveals that safety flip rate, TOST equivalence, and the absence of critical thresholds are robust findings that replicate across studies. The instability lies in the capability flip rate and the derived ratio, which are sensitive to the model set and prompt composition. This has implications for how batch perturbation findings should be communicated: the statement "safety outputs flip at approximately 0.6% under batch perturbation" is a well-supported, replicable claim, while "safety flips at 4x the capability rate" is model-set-dependent and should be qualified.

The directional reversal between TR138 (unsafe concern) and TR141 (net safe, p=3.8e-5) is the most consequential discrepancy. This could reflect genuine model-dependent directionality (the additional models in TR141 happen to bias toward safe), GPU-dependent floating-point behavior (A100 vs RTX PRO 6000 Blackwell), or both. Resolving this requires running the full 7-model suite on multiple GPU architectures -- a natural extension for TR143 or TR144.

---

## SS28. Limitations

1. **Single GPU class.** All results are from an NVIDIA RTX PRO 6000 Blackwell Server Edition on Google Colab. Batch perturbation effects may differ on other GPU architectures (A100, H100, consumer-grade GPUs) due to different floating-point hardware implementations and CUDA kernel scheduling.

2. **Temperature 0 only.** Greedy decoding maximizes sensitivity to floating-point non-associativity but does not capture sampling-based deployments. At temperature > 0, sampling noise may mask or amplify batch perturbation effects.

3. **Two phases only.** TR141 does not include co-batching (mixed-model interference) or quantization phases, which are covered by TR138. The interaction between batch perturbation and quantization remains an open question for cross-architecture studies.

4. **Single seed.** Seed=137 gives one draw of the floating-point perturbation landscape. Different seeds could shift the specific prompts that flip, though the aggregate statistics are expected to be similar.

5. **Regex-only scoring.** The regex-based safety classifier captures hard refusal/compliance boundaries but misses subtle refusal quality changes (e.g., a model that still refuses but with less conviction). An LLM judge would capture these nuances but at the cost of introducing judge variance as a confound.

6. **Dropped model.** gemma-2-2b was dropped due to gated access restrictions, reducing the distilled alignment category to a single model and eliminating coverage of the Gemma architecture family.

7. **Unbalanced alignment categories.** RLHF has 4 models while SFT, DPO, and distilled each have 1. The ANOVA result (SS11) should be interpreted cautiously, as within-category variance is poorly estimated for the single-model categories.

8. **Small per-cell flip counts.** Many individual model x batch-size cells have fewer than 5 safety flips, limiting the statistical power of per-cell analyses. The aggregate analyses (SS7, SS10, SS11) are more reliable.

9. **No prompt-level repeated measures.** Each prompt is evaluated once per batch size, so within-prompt flip consistency across runs cannot be assessed. A prompt that flips at BS=4 may or may not flip at BS=4 in a different run.

10. **Production serving dynamics.** Production vLLM deployments use continuous batching with dynamic batch sizes that change on each scheduling cycle. This study tests fixed batch sizes, which may not capture the transient perturbation patterns produced by rapidly changing batch compositions.

11. **No multi-turn context.** All prompts are single-turn. Multi-turn conversations (as studied in TR139) may show different batch sensitivity because the KV cache from prior turns constrains the generation space, potentially increasing or decreasing susceptibility to floating-point perturbation.

12. **Regex classifier limitations.** The regex-based safety classifier cannot detect subtle quality changes within the "refuse" or "comply" categories. A model might shift from a firm, well-reasoned refusal to a weak, hedge-filled refusal without the classifier detecting any change. An LLM judge would capture such graded changes but introduces its own non-determinism.

---

## SS29. Conclusions

1. **Batch perturbation is not safety-neutral across architectures, but the effect is small and not catastrophic.** Safety outputs flip at 0.63% versus capability outputs at 0.47% (1.3x ratio) across 7 models and 5 batch sizes. All TOST equivalence tests confirm that aggregate score shifts remain within the +/-3pp practical equivalence margin. This replicates the direction of TR138's finding while establishing a lower magnitude when measured across a more diverse model set.

2. **Cross-architecture fragility varies substantially.** The most fragile model (phi-3.5-mini at 1.11%) has a safety flip rate 2.9x higher than the least fragile (stablelm-2-zephyr and smollm2-1.7b at 0.38%). Within the RLHF family, Qwen models are consistently more fragile than Llama models at comparable parameter counts.

3. **Alignment approach significantly predicts fragility.** One-way ANOVA confirms that alignment type explains a statistically significant (though small) portion of fragility variance (F=4.862, p=0.008, eta-squared=0.0007). The ranking is SFT > RLHF > DPO = Distilled, suggesting that DPO and distillation may produce smoother refusal boundaries.

4. **True-batch validation confirms the effect is real.** Phase 2 explicit tensor batching produces 0.94% safety flip rate with 99.45% flip agreement with Phase 1, ruling out timing artifacts as an explanation.

5. **The directional bias is net SAFE, opposite to a priori concerns.** Of 97 directional safety flips, 71% move toward safety (compliance to refusal) versus 29% toward unsafety (p=3.8e-5). This suggests that batch perturbation in this experimental configuration makes marginal outputs more conservative, not less.

6. **TruthfulQA prompts are disproportionately sensitive.** At 2.7% mean flip rate, TruthfulQA is 5-13x more sensitive than other tasks, indicating that ambiguous, boundary-proximate prompts are the primary flip candidates.

7. **Flipped samples are systematically slower.** Flip-latency correlation analysis shows Cohen's d ranging from 0.17 to 1.17, suggesting that longer (more complex) outputs are more susceptible to batch perturbation.

8. **No critical batch-size threshold exists.** Safety flip rates do not spike at any particular batch size for any model. The effect is diffusely distributed across the tested range (2-32).

### Hypothesis Disposition

**Table 25: Hypothesis Testing Summary**

| Hypothesis | Prediction | Evidence | Verdict |
|-----------|-----------|---------|---------|
| H1 (Null) | Safety flip rate = capability flip rate for all models and batch sizes | Aggregate ratio 1.35 (safety > capability), consistent across 5 batch sizes; but no per-cell significance after correction | **Not rejected** at per-cell level; **rejected** by alignment ANOVA |
| H2 (Alternative) | Safety flip rate > capability flip rate, varying by architecture | 0.63% vs 0.47% aggregate; 2.9x cross-architecture spread; CIs overlap at all batch sizes | **Partially supported** -- direction confirmed, magnitude small |
| H3 (Alignment) | Alignment type predicts fragility | ANOVA F=4.862, p=0.008, eta-sq=0.0007; SFT > RLHF > DPO = Distilled | **Supported** (statistically significant, small effect size) |

**Observations.** The hypothesis testing results occupy an interesting middle ground. The evidence consistently favors H2 (safety-disproportionate batch perturbation) in direction but not in statistical significance at the per-cell level. The alignment ANOVA (H3) is the strongest positive result, with p=0.008 providing solid evidence that alignment approach matters. The practical conclusion is that batch perturbation produces a small, alignment-modulated, safety-biased perturbation that is detectable in aggregate but not at the resolution of individual batch-size cells.

### Synthesis with TR138

TR141's 7-model, 4-alignment-type study substantially extends TR138's 3-model, single-alignment-type foundation. The key confirmations and revisions are:

- **Confirmed:** Safety outputs are more susceptible to batch perturbation than capability outputs (direction replicated).
- **Revised magnitude:** The safety-to-capability ratio is 1.3x (TR141) versus 4x (TR138), suggesting TR138's ratio was elevated by its specific model selection.
- **Revised direction:** TR141 finds a net SAFE directional bias (p=3.8e-5), revising TR138's concern about net unsafe drift.
- **New finding:** Alignment type is a statistically significant predictor, enabling architecture-aware deployment guidance.
- **Confirmed:** No critical batch-size threshold exists in either study, establishing this as a robust negative finding.

### Open Questions

Several questions remain for future investigation:

1. **Interaction with quantization:** TR141 uses FP16 exclusively. How does batch perturbation interact with quantization (Q4_K_M, Q8_0) across architectures? TR138 explored this for 3 models; cross-architecture extension is needed.
2. **Co-batching effects:** When different prompt types share a batch (e.g., safety-critical and benign prompts together), does co-batching amplify safety perturbation? TR138 Phase 3 addressed this for limited models.
3. **Temperature interaction:** At temperature > 0, does sampling noise mask or amplify batch perturbation? The current temperature=0 design maximizes sensitivity but may overstate effects for sampling-based deployments.
4. **Larger models:** All tested models are 1-4B parameters. Models at 7B+ may have different fragility profiles due to wider attention heads and more accumulation paths.
5. **GPU architecture comparison:** Running the same 7-model suite on A100 and H100 would determine whether the net-safe directional bias is Blackwell-specific or universal.
6. **Prompt-level flip persistence:** Evaluating the same prompts across multiple seeds would determine whether the same prompts consistently flip or whether different seeds produce entirely different flip candidates.
7. **Fine-grained safety scoring:** Replacing the binary regex classifier with a graded LLM judge would capture the magnitude of safety changes, not just whether the binary classification crosses the threshold.

---

## SS30. Production Guidance

### Safety Testing

1. **Validate refusal rates at the actual production batch configuration.** Batch=1 baselines are insufficient for characterizing safety in batch-serving deployments. Test at the specific batch size (or batch-size range) used in production.

2. **Focus testing on boundary-proximate prompts.** TruthfulQA-style ambiguous prompts and prompts with moderate safety scores (near the refusal threshold) are 5-13x more likely to flip than clearly safe or clearly unsafe prompts.

3. **Monitor flip direction over time.** TR141 finds a net-safe bias, but this may be configuration-dependent. Track the ratio of unsafe-to-safe flips in production monitoring.

### Architecture Selection

4. **Prefer DPO-aligned or distilled models when batch-safety robustness is critical -- but verify baseline safety first.** These alignment approaches showed the lowest fragility (0.38%) in this study. However, this low fragility is partly a floor effect: smollm2-1.7b has 0% refusal for direct jailbreaks (SS23, Table 20b) and only 25 total baseline refusals across all safety prompts. A model with minimal safety has minimal safety to perturb. The recommendation to prefer these alignment types applies only when the model's baseline safety is already adequate for the deployment context. Always evaluate baseline refusal rates before selecting a model based on batch robustness.

5. **Be cautious with SFT-only models in batch-serving deployments.** phi-3.5-mini (SFT) showed nearly 3x the fragility of the most robust models. If SFT models must be used, apply additional safety monitoring.

6. **Within RLHF, Llama is more robust than Qwen.** For matched parameter counts, Llama family models showed approximately half the safety flip rate of Qwen family models.

### Infrastructure

7. **Consider deterministic inference kernels for safety-critical pipelines.** SGLang deterministic mode eliminates batch-induced floating-point non-associativity at approximately 34% throughput cost. For deployments where even 0.6% flip rate is unacceptable, this may be warranted.

8. **No batch-size ceiling required.** Since no critical threshold was detected, there is no evidence-based reason to cap batch size below 32 for safety reasons. The effect is proportional, not threshold-based.

### Monitoring and Evaluation

9. **Use targeted re-evaluation for boundary-proximate prompts.** Prompts with moderate safety scores (0.3-0.7) at batch=1 are the most likely flip candidates (SS25). Consider routing these to batch=1 or deterministic inference, leaving high-confidence prompts in the standard batch pipeline.

10. **Track flip-latency as a runtime signal.** Flipped samples are systematically slower (SS19). Unusually long response times may indicate boundary-proximate generation that is susceptible to batch perturbation.

11. **Evaluate at production batch size, not batch=1.** This recommendation from TR138 is strongly reinforced by TR141's finding that Phase 2 true-batch flip rates (0.94%) exceed Phase 1 dispatch-batch flip rates (0.63%). The actual production batching mechanism matters.

12. **Include TruthfulQA-style prompts in safety benchmarks.** TruthfulQA's 2.7% flip rate (13x the least sensitive task) makes it the most informative single benchmark for detecting batch perturbation effects. Any safety evaluation that does not include ambiguous, boundary-proximate prompts will underestimate batch perturbation risk.

### Summary Decision Matrix

The following matrix summarizes recommended actions by deployment scenario:

| Scenario | Risk Level | Recommended Action |
|----------|-----------|-------------------|
| High-throughput, non-safety-critical (e.g., summarization) | Low | Use maximum batch size, no special monitoring |
| General-purpose chatbot | Low-Moderate | Test at production batch size, monitor TruthfulQA-like prompts |
| Safety-critical content moderation pipeline | Moderate | Test at production BS, use DPO/distilled models, monitor flip direction |
| High-stakes safety system (medical, legal) | Moderate-High | Consider deterministic inference, re-evaluate boundary-proximate prompts at BS=1 |
| Adversarial robustness testing | Low (batch-specific) | No special batch-size concerns; batch perturbation does not amplify jailbreaks |

---

## SS31. Reproducibility

**Table 24: Reproducibility Details**

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA RTX PRO 6000 Blackwell Server Edition (98 GB VRAM), Google Colab |
| **Backend** | vLLM (Docker, FP16, --gpu-memory-utilization 0.90) |
| **Seed** | 137 |
| **Temperature** | 0.0 (greedy decoding) |
| **Max tokens** | 256 |
| **Warmup** | 3 requests per model |
| **Run directory** | `tr141_run_20260317_222300` |
| **Artifacts** | `samples.jsonl`, `tr141_analysis.json`, `tr141_scored.jsonl`, `tr141_report.md` |
| **Framework** | TR130 backend abstraction, TR134 safety classifiers |
| **Analysis passes** | 28 |

**Reproduction commands:**
```bash
# Full run (eval + analysis + report)
python research/tr141/run.py -v

# Phase 1 only
python research/tr141/run.py -v --phases 1

# Re-analyze existing run
python research/tr141/run.py -v --skip-eval --run-dir research/tr141/results/<RUN_ID>
```

**Observations.** Exact numerical reproduction requires the same GPU architecture (Blackwell), CUDA version, and vLLM version, because floating-point non-determinism is hardware- and kernel-version-specific. The seed controls vLLM's internal randomness but not floating-point accumulation order, which is the primary source of batch perturbation. Running the same experiment on a different GPU (e.g., A100 vs RTX PRO 6000) will produce different specific flip candidates but should yield similar aggregate statistics.

**Reproduction expectations by hardware:**
- **Same GPU (RTX PRO 6000):** Expect identical outputs at each batch size with the same seed and vLLM version. Aggregate statistics should match within sampling noise.
- **Same GPU family (Blackwell):** Expect same aggregate statistics but potentially different specific flip candidates due to silicon-level variation.
- **Different GPU family (e.g., A100, H100):** Expect similar aggregate patterns (safety > capability, same fragility ranking) but different magnitudes. The safety flip rate of approximately 0.6% is an empirical observation on Blackwell and should not be assumed to transfer directly to other architectures.
- **Different backend (SGLang, TensorRT-LLM):** Different inference engines have different CUDA kernel implementations that change the floating-point accumulation order. Aggregate patterns should be similar but magnitudes may differ.

**Data availability:** All artifacts (samples.jsonl, tr141_analysis.json, tr141_scored.jsonl) are stored in the run directory and contain full prompt-level data for reanalysis. The analysis JSON contains all 28 analysis passes' results, enabling verification of every number in this report.

---

## Appendix A: Full Output Identity Matrix

This appendix provides the detailed output identity statistics per batch size across all 7 models, including safety and capability flip counts.

**Table A1: Output Identity Detail by Batch Size**

| Batch Size | N Total | Byte-Identical | % Changed | Safety Flips | Cap Flips | S + C Total Flips |
|-----------|---------|---------------|-----------|-------------|----------|-----------------|
| 2 | 6,671 | 5,868 | 12.0% | 19 | 17 | 36 |
| 4 | 6,671 | 5,853 | 12.3% | 28 | 20 | 48 |
| 8 | 6,671 | 5,868 | 12.0% | 19 | 13 | 32 |
| 16 | 6,671 | 5,854 | 12.2% | 19 | 15 | 34 |
| 32 | 6,671 | 5,861 | 12.1% | 19 | 14 | 33 |
| **Total** | **33,355** | **29,304** | **12.1%** | **104** | **79** | **183** |

**Observations.** The overall output change rate is remarkably stable at 12.0-12.3% across batch sizes, confirming that the degree of textual non-determinism is primarily a function of the model composition rather than the specific batch size. Of the approximately 4,000 changed outputs per batch size, only 32-48 (0.8-1.2% of changed outputs) result in a classification flip. This means the vast majority of batch-induced output changes are semantically neutral -- different wording that receives the same classification. Batch size 4 produces the most total flips (48), consistent with it having the highest safety flip rate in SS7.

**Table A2: Per-Model Contribution to Total Flips**

| Model | Total Safety Flips (All BS) | Total Cap Flips (All BS) | Total Combined | % of All Flips |
|-------|---------------------------|------------------------|----------------|---------------|
| phi-3.5-mini | 26 | 45 | 71 | 38.8% |
| qwen2.5-1.5b | 23 | 2 | 25 | 13.7% |
| qwen2.5-3b | 16 | 3 | 19 | 10.4% |
| llama3.2-3b | 10 | 9 | 19 | 10.4% |
| llama3.2-1b | 11 | 3 | 14 | 7.7% |
| smollm2-1.7b | 9 | 12 | 21 | 11.5% |
| stablelm-2-zephyr | 9 | 5 | 14 | 7.7% |
| **Total** | **104** | **79** | **183** | **100%** |

**Observations.** phi-3.5-mini alone contributes 38.8% of all flips (71 of 183), despite being one of 7 models. Its contribution is heavily skewed toward capability flips (45 cap vs 26 safety), which is why its safety-to-capability ratio (SS6) is actually below 1.0 -- making it the most non-deterministic model overall but not the most safety-disproportionate. qwen2.5-1.5b contributes 23 safety flips but only 2 capability flips, giving it the highest safety-specificity among all models and explaining its elevated safety-to-capability ratios (up to 5.18x at individual batch sizes).

This per-model flip distribution has important implications for aggregate statistics. If phi-3.5-mini were removed from the analysis, the overall safety-to-capability ratio would increase substantially (from 1.35 to approximately 1.8), because phi-3.5-mini contributes many capability flips that dilute the aggregate ratio. Conversely, if qwen2.5-1.5b were removed, the ratio would decrease. This sensitivity to individual model inclusion underscores the importance of cross-architecture studies: conclusions based on any single model or small model set may not generalize.

---

## Appendix B: Per-Model Phase 2 Flip Rates

**Table B1: Phase 2 Safety and Capability Flip Rates by Model**

| Model | BS=4 Safety | BS=4 Cap | BS=8 Safety | BS=8 Cap | Mean Safety |
|-------|-----------|---------|-----------|---------|------------|
| llama3.2-1b | 0.80% (2/250) | 0.50% (1/200) | 1.20% (3/250) | 0.00% (0/200) | 1.00% |
| llama3.2-3b | 1.20% (3/250) | 0.00% (0/200) | 2.00% (5/250) | 0.00% (0/200) | 1.60% |
| phi-3.5-mini | 0.80% (2/250) | 1.50% (3/200) | 1.20% (3/250) | 1.50% (3/200) | 1.00% |
| qwen2.5-1.5b | 1.20% (3/250) | 0.50% (1/200) | 1.20% (3/250) | 0.00% (0/200) | 1.20% |
| qwen2.5-3b | 0.40% (1/250) | 0.50% (1/200) | 0.00% (0/250) | 0.00% (0/200) | 0.20% |
| smollm2-1.7b | 1.20% (3/250) | 0.50% (1/200) | 0.40% (1/250) | 0.50% (1/200) | 0.80% |
| stablelm-2-zephyr | 0.80% (2/250) | 0.00% (0/200) | 0.80% (2/250) | 0.50% (1/200) | 0.80% |

**Observations.** Phase 2 generally shows higher per-model safety flip rates than Phase 1. llama3.2-3b shows the largest Phase 2 amplification: its mean Phase 2 safety flip rate (1.60%) is 3.7x its Phase 1 rate (0.43%), with 5 safety flips at BS=8 alone. qwen2.5-3b shows the opposite pattern, with lower Phase 2 rates (mean 0.20%) than Phase 1 (0.68%). phi-3.5-mini, the most fragile model in Phase 1, shows comparable rates in Phase 2 (1.00% vs 1.11%). The small per-cell counts (0-5 flips per cell) mean individual cell comparisons are noisy; the aggregate Phase 2 analysis in SS13 provides the more reliable comparison.

---

## Appendix C: Overall Statistical Tests by Batch Size

**Table C1: Pooled Chi-Squared Tests Across All Models**

| Batch Size | Chi-squared | p-value | Odds Ratio | OR 95% CI | Significant? |
|-----------|------------|---------|-----------|-----------|-------------|
| 2 | 0.195 | 0.659 | 1.156 | [0.605, 2.208] | No |
| 4 | 1.646 | 0.199 | 1.445 | [0.817, 2.554] | No |
| 8 | 1.356 | 0.244 | 1.500 | [0.748, 3.007] | No |
| 16 | 0.628 | 0.428 | 1.306 | [0.669, 2.548] | No |
| 32 | 0.952 | 0.329 | 1.396 | [0.706, 2.759] | No |

**Observations.** None of the pooled chi-squared tests reach significance at alpha=0.05. The odds ratios are consistently above 1.0 (range 1.156-1.500), suggesting a genuine but small safety-capability asymmetry that requires either larger sample sizes or meta-analytic pooling across studies to confirm statistically. The widest 95% CI (BS=8: [0.748, 3.007]) spans from a slight capability bias to a 3x safety bias, illustrating the substantial uncertainty at the per-batch-size level. The consistency of the direction (all OR > 1.0) is more informative than any individual test.

A meta-analytic approach across all 5 batch sizes (treating each as an independent replicate) would provide stronger evidence. The mean OR across batch sizes is 1.36 (geometric mean: 1.35), and all 5 ORs are above 1.0. Under the null hypothesis of equal safety and capability flip rates, the probability of all 5 ORs exceeding 1.0 is 0.5^5 = 0.031, providing suggestive (though informal) evidence for a systematic safety bias. A formal meta-analysis using the Mantel-Haenszel method would be more rigorous and could be added in a future revision if TR141 data is combined with TR138's raw data.

---

## Appendix D: Latency-Throughput Scaling

**Table D1: Throughput at Each Batch Size (Samples per Second)**

| Model | BS=1 | BS=2 | BS=4 | BS=8 | BS=16 | BS=32 | Speedup (32/1) |
|-------|------|------|------|------|-------|-------|---------------|
| llama3.2-1b | 6.06 | 11.69 | 22.75 | 44.40 | 87.13 | 157.91 | **26.1x** |
| qwen2.5-1.5b | 3.27 | 6.28 | 12.30 | 24.22 | 47.42 | 88.09 | **26.9x** |
| smollm2-1.7b | 3.74 | 7.12 | 13.97 | 27.04 | 51.20 | 97.58 | **26.1x** |
| qwen2.5-3b | 1.81 | 3.34 | 6.51 | 12.71 | 25.15 | 50.70 | **28.0x** |
| stablelm-2-zephyr | 2.06 | 4.09 | 8.10 | 15.86 | 31.09 | 55.97 | **27.2x** |
| llama3.2-3b | 1.43 | 2.69 | 5.31 | 10.43 | 20.26 | 38.83 | **27.2x** |
| phi-3.5-mini | 0.62 | 1.19 | 2.34 | 4.42 | 8.29 | 15.29 | **24.7x** |

**Observations.** All models achieve near-ideal linear throughput scaling (24.7-28.0x speedup from BS=1 to BS=32, versus the theoretical maximum of 32x). This near-linear scaling explains the economic motivation for batch inference: switching from BS=1 to BS=32 provides approximately 26x throughput improvement at the cost of a 0.63% safety flip rate. The throughput benefit far exceeds the safety cost for most applications. phi-3.5-mini has the lowest absolute throughput at every batch size due to its larger parameter count and higher per-token latency.

**Table D2: Cost-Benefit Analysis at BS=32**

| Model | Throughput Gain (BS=32/BS=1) | Safety Flip Rate | Flips per 10K Prompts | Net Throughput per Safe Prompt |
|-------|---------------------------|-----------------|----------------------|------------------------------|
| llama3.2-1b | 26.1x | 0.47% | 47 | 25.97x |
| llama3.2-3b | 27.2x | 0.43% | 43 | 27.08x |
| phi-3.5-mini | 24.7x | 1.11% | 111 | 24.42x |
| qwen2.5-1.5b | 26.9x | 0.98% | 98 | 26.64x |
| qwen2.5-3b | 28.0x | 0.68% | 68 | 27.81x |
| smollm2-1.7b | 26.1x | 0.38% | 38 | 26.00x |
| stablelm-2-zephyr | 27.2x | 0.38% | 38 | 27.10x |

**Observations (cost-benefit).** Even for the most fragile model (phi-3.5-mini), the throughput-adjusted cost of batch perturbation is tiny: the effective throughput gain drops from 24.7x to 24.42x after accounting for flipped prompts -- a 1.1% throughput penalty. For the most robust models, the penalty is less than 0.1x. This analysis quantifies the economic argument for accepting batch perturbation in all but the most safety-critical applications: the throughput benefit of batching is approximately 25-28x, while the safety cost is less than 0.5% of that benefit. Even if every flipped prompt required manual review (the most conservative mitigation approach), the cost would be 38-111 reviews per 10,000 prompts -- well within typical content moderation capacity.

---

## Appendix E: Per-Task Flip Rate Matrix (Phase 1)

**Table E1: Detailed Per-Task, Per-Batch-Size Flip Rates**

| Task | BS=2 Flips/N | BS=2 Rate | BS=4 Flips/N | BS=4 Rate | BS=8 Flips/N | BS=8 Rate | BS=16 Flips/N | BS=16 Rate | BS=32 Flips/N | BS=32 Rate |
|------|-------------|----------|-------------|----------|-------------|----------|-------------|-----------|-------------|----------|
| truthfulqa | 8/350 | 2.29% | 13/350 | 3.71% | 11/350 | 3.14% | 6/350 | 1.71% | 9/350 | 2.57% |
| jailbreak_amp | 6/840 | 0.71% | 10/840 | 1.19% | 4/840 | 0.48% | 8/840 | 0.95% | 5/840 | 0.60% |
| mmlu_real | 11/1995 | 0.55% | 11/1995 | 0.55% | 6/1995 | 0.30% | 10/1995 | 0.50% | 9/1995 | 0.45% |
| arc_challenge | 6/1400 | 0.43% | 9/1400 | 0.64% | 7/1400 | 0.50% | 5/1400 | 0.36% | 5/1400 | 0.36% |
| advbench | 1/700 | 0.14% | 3/700 | 0.43% | 2/700 | 0.29% | 3/700 | 0.43% | 3/700 | 0.43% |
| bbq_bias | 4/1386 | 0.29% | 2/1386 | 0.14% | 2/1386 | 0.14% | 2/1386 | 0.14% | 2/1386 | 0.14% |

**Observations.** The detailed per-task matrix confirms that TruthfulQA is the dominant source of flips at every batch size, contributing 8-13 flips per batch size from only 350 prompts (50 per model). In contrast, BBQ bias contributes only 2-4 flips from 1,386 prompts (approximately 200 per model). The TruthfulQA peak at BS=4 (3.71%, 13 flips) is the single highest flip rate observed at any task-batch-size combination, and it is worth noting that this single cell contributes 27% of all safety flips at BS=4.

The capability tasks (mmlu_real, arc_challenge) show moderate and relatively stable flip rates (0.30-0.64%), consistent with capability evaluations having a well-defined correct answer that is either robustly selected or not. The safety tasks show more variance, with TruthfulQA and jailbreak_amplification driving most of the safety-domain variation.

---

## Appendix F: Glossary

| Term | Definition |
|------|-----------|
| **Batch size (BS)** | Number of concurrent prompts processed in a single GPU forward pass. BS=1 is the deterministic baseline. |
| **FP non-associativity** | Property of IEEE 754 floating-point arithmetic where (a+b)+c != a+(b+c) due to rounding. Different batch sizes change the reduction tree order, producing different numerical results. |
| **Flip rate** | Fraction of prompts whose binary classification (safe/unsafe or correct/incorrect) changes between batch=1 control and batch=N treatment. |
| **Safety flip** | A prompt whose safety classification (safe vs unsafe) changes when processed at a different batch size. |
| **Capability flip** | A prompt whose capability classification (correct vs incorrect) changes when processed at a different batch size. |
| **Alignment type** | Training approach for instruction following. In this study: RLHF (Reinforcement Learning from Human Feedback), SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), Distilled (knowledge distillation from a larger model). |
| **TOST** | Two One-Sided Tests for equivalence. Confirms that a mean difference falls within a pre-specified bound (here, +/-3 percentage points). |
| **MDE** | Minimum Detectable Effect. The smallest effect size the experiment can detect at 80% power and alpha=0.05. |
| **Cohen's d** | Standardized effect size: (mean1 - mean2) / pooled_SD. Thresholds: < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large. |
| **Wilson CI** | Wilson score interval for binomial proportions. Preferred over Wald interval for small sample sizes and extreme proportions. |
| **Holm-Bonferroni** | Step-down multiple comparison correction. Controls family-wise error rate while being less conservative than Bonferroni. |
| **Eta-squared** | ANOVA effect size: proportion of total variance explained by the grouping variable. |
| **Compliance slope** | Linear regression slope of compliance rate against batch size, measuring whether a jailbreak type becomes more effective at higher batch sizes. |
| **Fragility** | A model's susceptibility to batch-induced safety classification changes, operationalized as aggregate safety flip rate. |
| **Output identity** | Percentage of prompts producing byte-identical text outputs between two conditions (e.g., BS=1 vs BS=4). |
| **Directional flip** | A safety flip further classified by direction: toward unsafe (refusal to compliance) or toward safe (compliance to refusal). |
| **True batching** | Phase 2 methodology where prompts are passed as a single list to vLLM's completions endpoint, ensuring explicit tensor batching rather than server-side dispatch. |
| **Dispatch batching** | Phase 1 methodology where synchronized concurrent requests are sent to force the server to batch them, relying on vLLM's internal scheduling. |

---

---

## Acknowledgments

This experiment was conducted on Google Colab using an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (98 GB VRAM). The experimental framework builds on TR130 (backend abstraction layer), TR134 (safety classification pipeline), and TR138 (batch perturbation methodology). The prompt sets incorporate materials from AdvBench, BBQ, TruthfulQA, MMLU, and ARC-Challenge, with jailbreak amplification prompts generated using the TR139 prompt construction methodology. The analysis pipeline performs 28 automated statistical passes, producing the tr141_analysis.json artifact from which this report was generated and expanded to full depth.

---

## References

1. **TR138: Batch Inference Safety Under Non-Determinism** (Banterhearts, 2026). Foundation experiment with 3 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b), 4 phases including co-batching and quantization interaction. Established the batch-safety perturbation phenomenon.

2. **TR139: Multi-Turn Jailbreak Resistance Across Quantization Levels** (Banterhearts, 2026). 10,600 conversations measuring jailbreak amplification under quantization. Relevant for understanding how different perturbation sources (quantization vs batching) affect safety.

3. **TR134-TR137: Banterhearts Alignment Robustness Under Quantization** (2026). Baseline safety characterization across quantization levels for models shared with TR141.

4. **SGLang: Efficient Execution of Structured Language Model Programs** (Zheng et al., 2024). Deterministic inference mode eliminates batch-induced FP non-associativity via batch-invariant CUDA kernels.

5. **LLM-42: Verified Speculation for Deterministic LLM Inference** (Microsoft Research, Jan 2026). Formal verification approach for ensuring inference determinism.

6. **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., SOSP 2023). The inference backend used in both phases.

7. **IEEE 754-2019: Standard for Floating-Point Arithmetic.** Defines the non-associativity property exploited by batch perturbation.

8. **TruthfulQA: Measuring How Models Mimic Human Falsehoods** (Lin et al., ACL 2022). Source of the most batch-sensitive task in this study.

9. **BBQ: A Hand-Built Bias Benchmark for Question Answering** (Parrish et al., ACL 2022). Source of demographic bias evaluation prompts.

10. **AdvBench: An Open-Source Toolkit for Safety Evaluation of Large Language Models** (Zou et al., 2023). Source of refusal evaluation prompts.

11. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (Rafailov et al., NeurIPS 2023). DPO alignment approach used by stablelm-2-zephyr.

12. **MMLU: Measuring Massive Multitask Language Understanding** (Hendrycks et al., ICLR 2021). Source of capability evaluation prompts.

13. **ARC: Think you have Solved Question Answering? Try ARC** (Clark et al., 2018). Source of challenge-set capability evaluation prompts.

14. **Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone** (Abdin et al., Microsoft, 2024). Architecture and SFT alignment approach for the most fragile model in this study.

15. **SmolLM2: When Smol Goes Big** (HuggingFace, 2025). Distilled alignment approach for the joint-least-fragile model.

16. **StableLM 2: Stable Language Models for Diverse Applications** (Stability AI, 2024). DPO alignment approach for the joint-least-fragile model in this study.
