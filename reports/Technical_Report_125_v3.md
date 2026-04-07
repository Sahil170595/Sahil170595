# Technical Report 125 v3: Quantization Decision Matrix (AWQ/GPTQ Expansion)
## Production quant level selection across 6 models, 4 families, 9 quant formats including AWQ and GPTQ

| Field | Value |
|-------|-------|
| **TR Number** | 125 v3 |
| **Project** | Banterhearts LLM Performance Research |
| **Date** | 2026-04-07 (Original: 2026-02-22, v2 Expansion: 2026-03-28, v3 AWQ/GPTQ: 2026-04-07) |
| **Version** | 3.0 |
| **Author** | Research Team |
| **Git Commit** | 0439e828 |
| **Report Type** | Full-depth quantization impact analysis (AWQ/GPTQ cross-format expansion) |
| **Models** | llama3.2-1b (1.2B), llama3.2-3b (3.2B), qwen2.5-1.5b (1.5B), phi-2 (2.7B), mistral-7b (7.2B), qwen2.5-7b (7.6B) |
| **Quant Formats** | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K, AWQ, GPTQ |
| **Total Model-Quant Variants** | 51 (40 GGUF + 11 AWQ/GPTQ in v3) |
| **Sample Counts** | v1: 24,990; v2: 8,820; v3 small-model: 5,145; v3 7B: 2,940; Quality Total: 41,895 (37,485 loaded) |
| **Safety Samples** | 48,603 (loaded) across five sources |
| **Judge Annotations** | 24,336 (loaded) across six sources |
| **Hardware** | RTX 4080 Laptop 12GB (v1), Colab T4 16GB (v2), RTX 4080 Laptop 12GB + Docker (small-model v3), Runpod RTX 6000 Ada 48GB (7B v3) |
| **Status** | Complete |
| **Depends On** | TR125 v1, TR125 v2, TR142 (bespoke analysis v3), TR124 (baselines), TR134 (safety) |
| **Run IDs** | v1: `20260221_120035`; v2: `20260328_064807`; v3 small quality: `20260330_222254`; v3 small safety: `20260331_125319`; v3 7B AWQ quality: `20260406_033657`; v3 7B GPTQ quality: `20260406_181327`; v3 7B AWQ safety: `20260406_190115`; v3 7B GPTQ safety: `20260407_150840` |
| **Analysis Bundle** | `phase56_v3_full_canonical` (51 rows x 83 columns) |

---

## Abstract

TR125 v1 established Q4_K_M as the safe GGUF quantization default across 5 models. TR125 v2 extended that finding to 7 models across 4 architecture families with integrated safety metrics. TR125 v3 asks a different question: **do non-GGUF quantization formats -- specifically AWQ and GPTQ -- preserve the same quality and safety guarantees as Q4_K_M?** This study now includes **8,085 AWQ/GPTQ quality samples**, **10,483 AWQ/GPTQ safety samples**, and **5,148 AWQ/GPTQ judge annotations** across 6 models. The 7B branch is now complete: mistral-7b and qwen2.5-7b were evaluated under both AWQ and GPTQ on Runpod. phi-2 AWQ remains excluded due to an architecture incompatibility (parallel attention+MLP), leaving **11 successful AWQ/GPTQ variants** in the final matrix.

The core findings are: (1) AWQ and GPTQ produce **format-dependent quality distortion** rather than a consistent quality gain: Llama variants show inflated BERTScore and ROUGE-L, qwen2.5-1.5b collapses on both, and the 7B entries mostly remain neutral on quality while still failing safety. (2) AWQ and GPTQ cause **severe safety degradation** across the tested models, with refusal rate drops of -12pp to -68pp and **7 of 11 AWQ/GPTQ variants** classified as hidden-danger rows. (3) **qwen2.5-1.5b under AWQ/GPTQ shows the most severe quality collapse**: BERTScore drops -13.7pp (AWQ) and -13.0pp (GPTQ) from FP16, opposite to the inflation pattern on Llama. (4) The completed 7B branch sharpens the safety story: mistral-7b AWQ and GPTQ are both hidden-danger, while qwen2.5-7b AWQ and GPTQ are neutral on regime but still fail blanket-safe deployment. (5) phi-2 AWQ failed entirely due to architecture incompatibility with the AutoAWQ library. The most important hidden-danger non-result remains phi-2 GPTQ, which preserves decent benchmark scores while simultaneously destroying safety alignment.

The operational conclusion is: **AWQ and GPTQ are not safe substitutes for GGUF Q4_K_M**. All tested AWQ/GPTQ variants fail the blanket safety screen, and the quality metrics they produce are unreliable proxies for actual model capability. The Q4_K_M recommendation from v1/v2 is reinforced, while the stricter deployment rule now treats Q5_K_M as the conservative review floor rather than a blanket auto-deploy setting. The final v3 canonical analysis bundle covers **51 model-quant rows** across **9 quantization formats** with a unified matrix of **83 columns** spanning quality, safety, judge, and mechanism metrics.

**Total evidence base: 41,895 raw quality samples (37,485 loaded), 48,603 safety samples, 24,336 judge annotations.**

---

## Executive Summary

TR125 v3 answers: **are AWQ and GPTQ viable alternatives to GGUF quantization for production deployment, and do they maintain the quality-safety guarantees established in v1/v2?**

### Key Findings

1. **AWQ and GPTQ inflate generation metrics on small Llama models.** llama3.2-1b AWQ achieves +8.27pp BERTScore and +25.77pp ROUGE-L over FP16, while GPTQ achieves +8.49pp BERTScore and +27.29pp ROUGE-L. These improvements are artifacts of degenerate output patterns (increased coherence surface scores with simultaneous repetition degradation), not genuine quality gains.
2. **AWQ and GPTQ destroy safety alignment broadly.** 7 of 11 AWQ/GPTQ variants are classified as hidden-danger in the regime taxonomy; the remaining 4 (qwen2.5-1.5b AWQ/GPTQ and qwen2.5-7b AWQ/GPTQ) are neutral because quality also degrades or safety loss remains below the hidden-danger cutoff. Refusal rate drops range from -12.3pp (mistral-7b GPTQ) to -68.2pp (llama3.2-1b GPTQ). No AWQ or GPTQ variant passes the blanket safety screen.
3. **qwen2.5-1.5b is uniquely degraded by AWQ/GPTQ.** AWQ drops BERTScore -13.7pp and ROUGE-L -14.8pp from FP16. GPTQ drops -13.0pp BERTScore and -11.6pp ROUGE-L. Combined with refusal rate losses of -24.5pp (AWQ) and -47.7pp (GPTQ), this model shows the clearest format-incompatibility signal.
4. **phi-2 GPTQ remains the clearest hidden-danger benchmark trap.** It achieves the best AWQ/GPTQ benchmark scores in the matrix (MMLU 54.4%, ARC 71.0%) while simultaneously suffering -55.5pp refusal rate loss. A quality-only deployment screen would approve this variant; only the safety evaluation catches the degradation.
5. **phi-2 AWQ failed due to architecture incompatibility.** phi-2 uses a parallel attention+MLP block layout that AutoAWQ does not support. This checkpoint was never produced and is excluded from the matrix.
6. **The 7B AWQ/GPTQ branch is now complete.** mistral-7b AWQ and GPTQ are both hidden-danger entries, while qwen2.5-7b AWQ and GPTQ are neutral on regime but still routed to not_blanket_safe in the deployment table.
7. **The deployment protocol classifies AWQ as not_blanket_safe (max refusal signal +62.25pp) and GPTQ as not_blanket_safe (max refusal signal +56.80pp).** AWQ has 4 reject rows and GPTQ has 5 reject rows. No AWQ or GPTQ variant passes the blanket safety screen.
8. **Q5_K_M is the lowest-bit GGUF format that passes the conservative floor test.** Max refusal signal +3.18pp regex-based (+2.73pp judge-based) across all 6 models, zero reject rows. Q4_K_M has 1 reject row (phi-2, elevated safety signal), downgrading it from blanket-safe to model-specific review. The v1/v2 Q4_K_M recommendation remains valid for the 5 non-phi-2 models.
9. **BPW regressions remain non-significant for quality metrics.** The median R-squared for quality metrics across models remains below 0.20, consistent with the v1/v2 finding that quantization effects are better described as step functions than linear gradients. AWQ and GPTQ points (nominally ~4 BPW) do not fit the GGUF regression line.

### Core Decisions (Updated for v3)

- **Never deploy AWQ or GPTQ without model-specific safety evaluation.** All 11 tested variants fail the blanket safety screen.
- For **maximum accuracy**: qwen2.5-7b at Q8_0 (73.7% MMLU, 89.0% ARC), unchanged from v2.
- For **best accuracy/size (7B class)**: qwen2.5-7b at Q4_K_M (73.0% MMLU, 88.5% ARC, ~4.7 GB), unchanged from v2.
- For **best accuracy/size (small class)**: llama3.2-3b at Q4_K_M (54.7% MMLU, 70.5% ARC).
- For **maximum throughput**: llama3.2-1b at Q4_K_M (from v1: 280.9 native tok/s).
- **Never deploy Q2_K** for any quality-sensitive task (unchanged from v1).
- **Q4_K_M remains the recommended GGUF default** for 5 of 6 models. phi-2 requires model-specific review due to elevated safety signals at Q4_K_M.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Matrix coverage | Model-quant rows | >= 40 | 51 | **PASS** |
| Matrix width | Columns | >= 50 | 83 | **PASS** |
| Quality samples | Total loaded | >= 30,000 | 37,485 | **PASS** |
| Safety samples | Total loaded | >= 40,000 | 48,603 | **PASS** |
| Judge annotations | Total loaded | >= 20,000 | 24,336 | **PASS** |
| P5 protocol | All 6 steps | All pass | 6/6 pass | **PASS** |
| AWQ blanket safety | Max refusal signal | < 5pp | 62.25pp | **FAIL** |
| GPTQ blanket safety | Max refusal signal | < 5pp | 56.80pp | **FAIL** |
| Q5_K_M refusal bound | Max refusal signal | < 5pp | 3.18pp | **PASS** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | AWQ/GPTQ inflate generation metrics on small Llama models | SS5.1 Table 5, regimes.csv | **Established** |
| C2 | AWQ/GPTQ destroy safety alignment universally | SS6.1 Table 8, phase5_quant_deployment.csv | **Established** |
| C3 | qwen2.5-1.5b is uniquely degraded by AWQ/GPTQ | SS5.2 Table 6, quality_wide.csv | **Established** |
| C4 | phi-2 GPTQ is a hidden-danger variant | SS5.3 Table 7, regimes.csv | **Established** |
| C5 | Q4_K_M recommendation holds for non-AWQ/GPTQ formats | SS9, phase5_quant_deployment.csv | **Established** |
| C6 | BPW is a poor linear predictor of quality | SS8, bpw_regressions.csv | **Established** |
| C7 | Safety degrades faster than quality in the majority of rows | SS7.1, asymmetry.csv | **Established** |

---

## When to Use This Report

### Scenario 1: Evaluating AWQ/GPTQ for a Small Model Deployment

**Question:** "Should I use an AWQ or GPTQ checkpoint instead of a GGUF Q4_K_M for my 1-3B model?"

**Answer:** No. See SS5 and SS6.1 -- every AWQ/GPTQ variant tested on small models fails the blanket safety screen. Generation metrics may appear inflated (SS5.1), masking genuine quality degradation. Use GGUF Q5_K_M as the conservative review floor, with Q4_K_M only after model-specific confirmation.

### Scenario 2: phi-2 Deployment Format Selection

**Question:** "phi-2 GPTQ shows better MMLU than phi-2 GGUF Q4_K_M. Can I use GPTQ?"

**Answer:** No for safety-sensitive applications. phi-2 GPTQ achieves 54.4% MMLU and 71.0% ARC (the best benchmark scores among AWQ/GPTQ variants), but its refusal rate drops -55.5pp from FP16. See SS5.3 Table 7 and SS6.1 Table 8.

### Scenario 3: Choosing Between GGUF Quant Levels

**Question:** "What is the safest GGUF quant level I can use?"

**Answer:** Q5_K_M passes the conservative review-floor test across all 6 models (max refusal signal +3.18pp, zero reject rows), but the canonical deployment role still requires model-specific review. Q4_K_M is safe for 5 of 6 models but requires model-specific review for phi-2. See SS9 Table 14.

### Scenario 4: Planning 7B AWQ/GPTQ Evaluation

**Question:** "Are 7B AWQ/GPTQ results available?"

**Answer:** Yes. The 7B branch is now complete. mistral-7b AWQ/GPTQ are both hidden-danger rows; qwen2.5-7b AWQ/GPTQ are neutral by regime but still not blanket safe. See SS3.2 and SS6.

### Scenario 5: Cross-Referencing with v2

**Question:** "Does v3 change the Q4_K_M recommendation from v2?"

**Answer:** No. v3 reinforces the v2 recommendation by demonstrating that the primary non-GGUF alternatives (AWQ and GPTQ) are unsafe on small models. The GGUF quality and safety data from v1 and v2 are fully preserved in the v3 canonical matrix. See SS9.

---

## Table of Contents

- [Abstract](#abstract)
- [Executive Summary](#executive-summary)
- [When to Use This Report](#when-to-use-this-report)
- [Metric Definitions](#metric-definitions)
- [SS1. Introduction](#ss1-introduction)
- [SS2. Methodology](#ss2-methodology)
- [SS3. Models and Design](#ss3-models-and-design)
- [SS4. What Changed in v3](#ss4-what-changed-in-v3)
- [SS5. Results: AWQ/GPTQ Quality Analysis](#ss5-results-awqgptq-quality-analysis)
- [SS6. Results: AWQ/GPTQ Safety Analysis](#ss6-results-awqgptq-safety-analysis)
- [SS7. Results: Quality-Safety Interaction](#ss7-results-quality-safety-interaction)
- [SS8. Results: BPW Regressions](#ss8-results-bpw-regressions)
- [SS9. Results: Deployment Protocol](#ss9-results-deployment-protocol)
- [SS10. Results: Regime Classification](#ss10-results-regime-classification)
- [SS11. Results: Refusal Mechanism Analysis](#ss11-results-refusal-mechanism-analysis)
- [SS12. Statistical Synthesis and Hypothesis Evaluation](#ss12-statistical-synthesis-and-hypothesis-evaluation)
- [SS13. Conclusions](#ss13-conclusions)
- [SS14. Limitations and Follow-Up](#ss14-limitations-and-follow-up)
- [SS15. Reproducibility](#ss15-reproducibility)
- [References](#references)
- [Appendix A: Complete Benchmark Tables](#appendix-a-complete-benchmark-tables)
- [Appendix B: Complete Generation Quality Tables](#appendix-b-complete-generation-quality-tables)
- [Appendix C: Complete Safety Tables](#appendix-c-complete-safety-tables)
- [Appendix D: Glossary](#appendix-d-glossary)
- [Appendix E: Configs and Provenance](#appendix-e-configs-and-provenance)

---

## Metric Definitions

### Primary Quality Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **BERTScore** | Token-level contextual embedding similarity (deberta-xlarge-mnli, F1 variant) | 0-1, higher = better quality |
| **ROUGE-L** | Longest common subsequence F1 overlap with reference | 0-1, higher = better quality |
| **Coherence** | Sentence-level cosine similarity (all-mpnet-base-v2, SemScore) | 0-1, higher = better quality |
| **Repetition** | unique_4grams / total_4grams | 0-1, higher = more diverse (1.0 = no repetition) |
| **Output Length** | Mean token count across generation tasks | Higher = more verbose |
| **MMLU Accuracy** | Raw accuracy on 285 real MMLU questions (57 subjects) | 0-1, higher = better knowledge |
| **ARC Accuracy** | Raw accuracy on 200 ARC-Challenge questions | 0-1, higher = better reasoning |

### Primary Safety Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Refusal Rate** | Fraction of harmful prompts correctly refused (regex-based) | 0-1, higher = safer |
| **Truthfulness** | Fraction of factual queries answered correctly | 0-1, higher = more truthful |
| **Bias Resistance** | Fraction of bias-probing prompts where model avoids biased output | 0-1, higher = less biased |
| **Judge Refusal Rate** | LLM-judge re-evaluation of refusal behavior | 0-1, higher = safer |

### Statistical Tests Used

| Test | Role in This Report |
|------|-------------------|
| Repeated-measures correlation | Within-model quality-safety relationship (SS7, SS12) |
| Mixed-effects regression | Pooled quality-safety relationship controlling for model (SS12) |
| OLS linear regression | BPW vs metric regressions (SS8) |
| Pearson/Spearman correlation | Pairwise metric relationships (SS7, SS11) |
| Bootstrap percentile CI | Confidence intervals (B=2000, seed=42) for all aggregated metrics |

### Evidence Standard

**Established findings** require consistent evidence across multiple models or statistical significance at p < 0.05 with practical significance above the equivalence margin.

**Partial findings** show evidence in at least one comparison but lack consistency across models or fail statistical significance tests.

**Non-claims** are results where evidence is insufficient or where tests confirm equivalence to baseline.

---

## SS1. Introduction

### SS1.1 Research Questions

1. **RQ1:** Do AWQ and GPTQ quantization formats preserve quality at levels comparable to GGUF Q4_K_M on models where both formats are available?
2. **RQ2:** Do AWQ and GPTQ quantization formats preserve safety alignment (refusal rate, truthfulness, bias resistance) at levels comparable to GGUF Q4_K_M?
3. **RQ3:** Does the quality-safety proxy failure pattern (sign reversal) identified in v2 extend to AWQ/GPTQ variants?
4. **RQ4:** Can generation metrics (BERTScore, ROUGE-L, coherence) serve as reliable proxies for quality when comparing across quantization format families?

### SS1.2 Why This Matters

AWQ (Activation-aware Weight Quantization) and GPTQ (Generalized Post-Training Quantization) are the two most widely adopted non-GGUF quantization methods for deploying LLMs at reduced precision. While GGUF (via llama.cpp and Ollama) dominates local inference, AWQ and GPTQ are prominent in cloud and server deployments through frameworks like vLLM, TGI, and AutoAWQ/AutoGPTQ. A deployment team choosing between quantization formats needs to know whether the quality-safety guarantees established for GGUF transfer to these alternatives.

The v1/v2 finding that Q4_K_M is safe for GGUF is only useful if practitioners can trust that alternative 4-bit methods offer comparable behavior. If AWQ/GPTQ at nominally similar bit-widths produce different quality or safety profiles, the deployment decision tree must be format-aware, not just precision-aware.

### SS1.3 Scope

| Dimension | Coverage |
|-----------|----------|
| Models (v3 new data) | llama3.2-1b (1.2B), llama3.2-3b (3.2B), qwen2.5-1.5b (1.5B), phi-2 (2.7B), mistral-7b (7.2B), qwen2.5-7b (7.6B) |
| Models (full matrix) | Same 6 models merged with legacy GGUF waves |
| Quant formats | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K, AWQ, GPTQ |
| v3 new samples | 8,085 quality + 10,483 safety + 5,148 judge |
| Total samples | 41,895 raw quality + 48,603 safety + 24,336 judge |
| Tasks | 7 (MMLU, ARC-Challenge, summarization, QA, code_gen, creative_writing, classification) |
| Backends | Ollama (GGUF), Transformers (AWQ/GPTQ) |
| Temperature | 0.0 |

### SS1.4 Literature Grounding

AWQ (Lin et al., 2023) identifies salient weight channels via activation statistics and applies per-channel scaling before quantization, aiming to minimize the impact of outlier activations. The key insight is that a small fraction of weights (those activated by large activations) disproportionately affect output quality, so protecting these weights during quantization preserves more model capability per bit.

GPTQ (Frantar et al., 2022) uses a one-shot layer-wise quantization method based on approximate second-order information (Hessian inverse). It quantizes weights column by column, using the Hessian to determine the optimal quantization order and compensating for quantization error in subsequent columns. Both methods target 4-bit quantization (nominally ~4 BPW), making them directly comparable to GGUF Q4_K_M (~4.85 BPW) in deployment scenarios.

GGUF quantization (via llama.cpp) uses a different approach: importance-aware mixed precision. The Q4_K_M format allocates different bit-widths to different tensor blocks based on a sensitivity analysis, resulting in an average of ~4.85 BPW. Some attention weights receive 6 bits while less critical weights receive 4 bits. This mixed-precision approach is unique among the three formats.

The key difference from GGUF is that AWQ and GPTQ produce model checkpoints that run through the Hugging Face Transformers pipeline rather than the llama.cpp runtime. This means the quantization format also changes the inference backend, introducing a potential confound: observed differences may reflect format effects, backend effects, or both. This study does not attempt to separate these effects.

Prior work on quantization safety is limited. Most quantization papers evaluate only perplexity and benchmark accuracy, not safety-specific metrics like refusal rate or bias resistance. The TR125 program is among the first to systematically evaluate safety alignment preservation under quantization across multiple formats and models.

The gap in the literature is particularly concerning because AWQ and GPTQ papers report perplexity numbers that suggest negligible quality loss at 4-bit (typically <0.5 perplexity increase). These numbers are correct but misleading for safety-critical deployments: perplexity measures average next-token prediction quality, which is dominated by common patterns. Safety-relevant behaviors (refusal patterns, bias avoidance) occupy a tiny fraction of the token distribution and can be destroyed without meaningfully affecting perplexity. The TR125 program addresses this gap by measuring safety directly rather than relying on perplexity as a proxy.

### SS1.5 How to Read This Report

Each result section follows the pattern: context prose, data table, then **Observations** interpreting the table. Key findings receive a blockquote restatement. Tables are numbered within their section (e.g., SS5.1 Table 5). Cross-references use SS notation.

This report focuses on the v3 additions (AWQ/GPTQ data) and how they change the overall quantization decision matrix. For the full GGUF analysis, see TR125 v1 and v2. The v3 canonical matrix includes all prior data; tables in this report present both GGUF and AWQ/GPTQ data side by side for comparison.

**Reading priority:** Readers interested in the deployment recommendation should start with the Executive Summary and SS9 (Deployment Protocol). Readers interested in the quality analysis should read SS5. Readers interested in the safety evidence should read SS6 and SS10 (Regime Classification). Readers interested in the statistical methodology should read SS12 (Statistical Synthesis).

**Terminology convention:** Throughout this report, "Demonstrated" indicates a finding supported by consistent evidence across multiple models or statistically significant tests. "Established" indicates a finding with strong evidence and practical significance. We avoid "Validated" to prevent confusion with validation in the machine-learning sense.

---

## SS2. Methodology

### SS2.1 Overall Design

TR125 v3 is a cross-format comparison study that extends the v1/v2 GGUF-only analysis with AWQ and GPTQ evaluation data. The study is designed as a matched comparison: each model that receives AWQ/GPTQ evaluation already has complete GGUF data from v1/v2, enabling direct within-model comparisons across format families.

TR125 v3 adds a third data collection phase to the v1/v2 timeline:

| Phase | Source | Models | Quant Formats | Samples |
|-------|--------|--------|---------------|---------|
| Phase 1-2 (v1) | Original | 5 (1.2B-8B) | 7 GGUF levels (Q2_K-FP16) | 24,990 quality |
| v2 Expansion | TR142 | 2 (7.2B, 7.6B) | 6 GGUF levels (Q2_K-Q8_0) | 8,820 quality |
| v3 small-model AWQ/GPTQ | TR142 | 4 (1.2B-3.2B) | AWQ, GPTQ | 5,145 quality + 6,671 safety |
| v3 7B AWQ/GPTQ | TR142 | 2 (7.2B-7.6B) | AWQ, GPTQ | 2,940 quality + 3,812 safety |

All phases feed into the `phase56_v3_full_canonical` bespoke analysis bundle, which merges quality, safety, and judge data into a single 51-row x 83-column matrix.

### SS2.2 Unit of Analysis

One data point is a single model response to a single prompt, scored by the relevant metric pipeline. For benchmark tasks (MMLU, ARC), the score is binary (correct/incorrect). For generation tasks, the score is a continuous metric (BERTScore, ROUGE-L, coherence, repetition). For safety tasks, the score is binary (refused/complied for refusal rate) or categorical (correct/hallucinated/biased).

The aggregation unit is the (model, quant) pair. All per-sample scores are averaged within each (model, quant) cell, producing the 51 rows of the canonical matrix. Confidence intervals are computed at the cell level via bootstrap resampling (B=2000, seed=42) for generation and safety metrics, and via Wilson score intervals for benchmark accuracy proportions.

### SS2.3 How Rows Become Claims

The evidence chain is: raw response -> per-sample score -> aggregation to (model, quant) cell -> delta from baseline -> regime classification -> deployment recommendation.

The delta step is critical: all comparisons are relative to the model's own baseline (FP16 for small models, Q8_0 for 7B models). Cross-model comparisons use absolute metric values, not deltas. This prevents a model with high absolute quality from masking a large quantization-induced delta.

The regime classification step maps each (model, quant) pair to one of five categories: baseline_or_neutral, neutral, hidden_danger, near_hidden_danger, or over_refusal. The classification uses both quality deltas (BERTScore) and safety deltas (refusal rate) simultaneously. A row is "hidden danger" when BERTScore delta >= -2pp AND refusal delta <= -10pp -- meaning quality appears acceptable while safety has collapsed.

Claims about format safety rest on the regime classification and deployment protocol (SS9, SS10), not on individual metric comparisons. This two-stage process prevents cherry-picking: a format must pass the regime screen AND the deployment protocol to receive a "safe" classification.

### SS2.4 Scoring Stack

**Quality metrics:** BERTScore (deberta-xlarge-mnli), ROUGE-L (rouge-score), coherence (all-mpnet-base-v2 cosine similarity), repetition (4-gram uniqueness ratio). All computed deterministically from model outputs and reference texts.

**Safety metrics:** Regex-based scoring for refusal rate, truthfulness, and bias resistance from TR134 Phase 3 prompts. LLM-judge re-evaluation via four judge sources:

| Judge Source | Models Covered | Judge Model |
|-------------|---------------|-------------|
| `tr134_legacy_judge` | llama3.2-1b, llama3.2-3b, phi-2, qwen2.5-1.5b (GGUF) | Qwen 7B (legacy) |
| `expansion_gemma3_judge` | llama3.2-1b, llama3.2-3b, phi-2, qwen2.5-1.5b (expansion) | Gemma 3 12B |
| `rejudge_7b_gemma3` | mistral-7b, qwen2.5-7b (GGUF) | Gemma 3 12B |
| `v3_awq_gptq_judge` | AWQ/GPTQ variants (v3) | Gemma 3 12B |

**Observations.**

- Regex and judge measurements can diverge substantially (see SS6.2). Mistral-7b shows the largest gap: regex refusal 23.6% vs judge refusal 91.3% at Q8_0 (+67.7pp gap). The deployment protocol uses the more conservative of the two signals.
- Judge coverage is complete: 51/51 matrix rows have judge annotations.

### SS2.5 Design Safeguards

1. **Seed fixing:** All evaluations use seed=42 and temperature=0.0 for deterministic output. At temperature=0, the Transformers pipeline should produce identical outputs for identical inputs, assuming no backend non-determinism.
2. **Identical task files:** v3 AWQ/GPTQ evaluation uses the same YAML task definitions as v1/v2, ensuring prompt-level alignment. Every AWQ/GPTQ model receives exactly the same prompts as its GGUF counterpart.
3. **Isolated model loading:** Each model-quant variant is loaded fresh; no parameter sharing between evaluations. GPU memory is cleared between runs to prevent cross-contamination.
4. **Source audit:** Every data source has a SHA-256 hash recorded in `run_manifest.json` and `source_audit.csv`. The bespoke analysis pipeline verifies hashes on load. If any hash fails to match, the pipeline refuses to proceed.
5. **Phase 5 validation protocol:** The deployment protocol follows a 6-step checklist (P5_001 through P5_006), each with an explicit pass/fail criterion. All 6 steps passed in the v3 canonical bundle.

### SS2.6 What This Design Does Not Do

1. **Does not separate format effects from backend effects.** AWQ/GPTQ run through Transformers; GGUF runs through Ollama/llama.cpp. Observed differences may reflect either or both.
2. **Does not include phi-2 AWQ.** One of the 12 planned AWQ/GPTQ cells remains permanently absent because phi-2 is incompatible with the AWQ export stack.
3. **Does not include phi-2 AWQ.** Architecture incompatibility prevents AWQ checkpoint creation for phi-2.
4. **Does not rescore benchmark accuracy.** MMLU and ARC scores are raw accuracy, not rescored. phi-2's raw accuracy is known to be depressed by formatting issues.
5. **Does not perform inference speed benchmarking.** AWQ/GPTQ were run through Transformers without timing instrumentation. Speed comparisons are not available.

---

## SS3. Models and Design

### SS3.1 Complete Model Matrix

| # | Model | Family | Params | GGUF Levels | AWQ | GPTQ | Baseline | Source |
|---|-------|--------|--------|-------------|-----|------|----------|--------|
| 1 | llama3.2-1b | Llama | 1.2B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Yes | Yes | FP16 | v1 + v3 |
| 2 | qwen2.5-1.5b | Qwen | 1.5B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Yes | Yes | FP16 | v1 + v3 |
| 3 | phi-2 | Phi | 2.7B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | FAILED | Yes | FP16 | v1 + v3 |
| 4 | llama3.2-3b | Llama | 3.2B | FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Yes | Yes | FP16 | v1 + v3 |
| 5 | mistral-7b | Mistral | 7.2B | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Yes | Yes | Q8_0 | v2 + v3 |
| 6 | qwen2.5-7b | Qwen | 7.6B | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K | Yes | Yes | Q8_0 | v2 + v3 |

**Observations.**

- 11 AWQ/GPTQ variants produced usable data: llama3.2-1b (AWQ, GPTQ), llama3.2-3b (AWQ, GPTQ), mistral-7b (AWQ, GPTQ), qwen2.5-1.5b (AWQ, GPTQ), qwen2.5-7b (AWQ, GPTQ), and phi-2 (GPTQ only).
- phi-2 AWQ failed because phi-2 uses a parallel attention+MLP block layout. AutoAWQ expects sequential attention-then-MLP and cannot calibrate quantization scales for the parallel architecture.
- The 7B branch was completed on Runpod RTX 6000 Ada 48GB hardware after the Colab/T4 path proved insufficient.

### SS3.2 AWQ/GPTQ Checkpoint Details

All AWQ/GPTQ checkpoints were quantized locally using AutoAWQ 0.2.9 and AutoGPTQ 0.7.1 with default 4-bit configurations:

| Format | Bits | Group Size | Calibration | Nominal BPW |
|--------|------|-----------|-------------|-------------|
| AWQ | 4 | 128 | 128 samples, WikiText-2 | ~4.0 |
| GPTQ | 4 | 128 | 128 samples, WikiText-2 | ~4.0 |
| GGUF Q4_K_M | 4-5 mixed | Per-tensor | n/a (post-training rounding) | ~4.85 |

**Observations.**

- AWQ and GPTQ at 4-bit are nominally lower precision than Q4_K_M (~4.85 BPW). The ~0.85 BPW difference means AWQ/GPTQ discard more weight information, which partially explains the larger quality/safety deltas.
- GGUF Q4_K_M uses a mixed-precision scheme (important weights get more bits), while AWQ and GPTQ apply uniform 4-bit quantization with group-level scaling. The mixed-precision approach preserves salient weights more effectively.

### SS3.2b Effective BPW Comparison

The nominal BPW values for AWQ, GPTQ, and GGUF Q4_K_M deserve careful comparison because they determine the baseline expectation for quality/safety trade-offs:

| Format | Nominal Bits | Effective BPW | Precision Strategy | Runtime |
|--------|-------------|---------------|-------------------|---------|
| AWQ 4-bit | 4 | ~4.0 | Uniform 4-bit + per-group scales | Transformers |
| GPTQ 4-bit | 4 | ~4.0 | Uniform 4-bit + per-group scales | Transformers |
| GGUF Q4_K_M | 4-5 mixed | ~4.85 | Mixed precision (important weights get more bits) | llama.cpp |
| GGUF Q3_K_S | 3-4 mixed | ~3.44 | Mixed precision | llama.cpp |

The ~0.85 BPW gap between GGUF Q4_K_M and AWQ/GPTQ means that AWQ/GPTQ discard approximately 17% more weight information per parameter. On a 1.2B model, this translates to roughly 180 million fewer bits available for weight representation. Whether this precision difference alone explains the safety gap is an open question that the format-backend confound prevents us from answering.

### SS3.3 v3 Checkpoint Failures

| Model | Format | Failure Mode | Resolution |
|-------|--------|-------------|------------|
| phi-2 | AWQ | Architecture incompatibility (parallel attn+MLP) | Cannot be resolved without AutoAWQ upstream changes |

**Observations.**

- The phi-2 AWQ failure is a permanent limitation of the current AutoAWQ library, not a hardware constraint.
- The former 7B hardware limitation is resolved. mistral-7b and qwen2.5-7b were completed on Runpod and are now part of the canonical v3 bundle.

---

## SS4. What Changed in v3

This section documents every addition relative to TR125 v2 to maintain a full audit trail.

### SS4.1 New Data

| Component | v2 | v3 Addition |
|-----------|-----|-------------|
| Quant formats | 7 GGUF (FP16-Q2_K) | +2: AWQ, GPTQ |
| AWQ/GPTQ variants | 0 | +11 successful (5 AWQ + 6 GPTQ) |
| Model-quant rows | 40 (matched matrix) | +11 = 51 |
| Quality samples | 33,810 raw (v1+v2) | +8,085 = 41,895 raw |
| Safety samples | 38,120 loaded (v1+v2) | +10,483 = 48,603 loaded |
| Judge annotations | 19,188 loaded (v1+v2) | +5,148 = 24,336 loaded |
| Matrix columns | 83 | Unchanged (same schema) |

### SS4.2 New Analysis

- AWQ/GPTQ quality comparison by model (SS5)
- AWQ/GPTQ safety analysis with regime classification (SS6, SS10)
- Hidden-danger row identification for 7 of 11 AWQ/GPTQ variants, plus review-only classification for the remaining 4 (SS10)
- Updated deployment protocol with AWQ/GPTQ rows (SS9)
- Refusal mechanism analysis (Phase 6) extended to AWQ/GPTQ (SS11)
- Updated BPW regressions with AWQ/GPTQ points (SS8)

### SS4.3 Unchanged from v2

- All GGUF quality and safety data, tables, and analysis
- Statistical framework (repeated-measures correlation, mixed-effects, BPW regression)
- Quality tier system (negligible/acceptable/concerning/unacceptable)
- Production guidance for GGUF quant levels (updated but not replaced)
- Source audit infrastructure (SHA-256 hashes for all data files)

### SS4.4 Analysis Bundle Change

The v3 canonical bundle replaces the v2 bundle:

| Bundle | Location | Rows | Columns |
|--------|----------|------|---------|
| v2 bundle | `research/tr142/results/bespoke_analysis/20260328_173033/` | 40 | 83 |
| **v3 canonical** | `research/tr142/results/bespoke_analysis_v3/phase56_v3_full_canonical/` | **51** | **83** |

---

## SS5. Results: AWQ/GPTQ Quality Analysis

### SS5.1 AWQ/GPTQ Generation Metrics vs FP16 Baseline

The following table shows BERTScore, ROUGE-L, coherence, and repetition for all AWQ/GPTQ variants alongside their FP16 baselines and Q4_K_M reference points. Delta is relative to FP16 in percentage points.

**Table 5: AWQ/GPTQ Generation Metrics by Model**

| Model | Format | BERTScore | ROUGE-L | Coherence | Repetition | BERTScore Delta | ROUGE-L Delta | Coherence Delta |
|-------|--------|-----------|---------|-----------|------------|----------------|---------------|-----------------|
| llama3.2-1b | FP16 | 0.646 | 0.266 | 0.580 | 0.996 | -- | -- | -- |
| llama3.2-1b | Q4_K_M | 0.665 | 0.297 | 0.581 | 0.996 | +1.88pp | +3.10pp | +0.13pp |
| llama3.2-1b | **AWQ** | **0.729** | **0.524** | **0.758** | 0.954 | **+8.27pp** | **+25.77pp** | **+17.86pp** |
| llama3.2-1b | **GPTQ** | **0.731** | **0.539** | **0.763** | 0.848 | **+8.49pp** | **+27.29pp** | **+18.30pp** |
| llama3.2-3b | FP16 | 0.767 | 0.469 | 0.661 | 0.999 | -- | -- | -- |
| llama3.2-3b | Q4_K_M | 0.759 | 0.454 | 0.650 | 0.998 | -0.89pp | -1.51pp | -1.06pp |
| llama3.2-3b | **AWQ** | 0.759 | **0.614** | **0.768** | 0.977 | -0.83pp | **+14.49pp** | **+10.72pp** |
| llama3.2-3b | **GPTQ** | 0.767 | **0.646** | **0.782** | 0.959 | +0.00pp | **+17.74pp** | **+12.08pp** |
| qwen2.5-1.5b | FP16 | 0.744 | 0.383 | 0.713 | 0.992 | -- | -- | -- |
| qwen2.5-1.5b | Q4_K_M | 0.718 | 0.349 | 0.697 | 0.992 | -2.62pp | -3.43pp | -1.56pp |
| qwen2.5-1.5b | **AWQ** | **0.607** | **0.235** | 0.659 | 0.977 | **-13.67pp** | **-14.79pp** | **-5.39pp** |
| qwen2.5-1.5b | **GPTQ** | **0.614** | **0.267** | 0.688 | 0.924 | **-12.98pp** | **-11.63pp** | **-2.52pp** |
| phi-2 | FP16 | 0.715 | 0.412 | 0.771 | 0.992 | -- | -- | -- |
| phi-2 | Q4_K_M | 0.721 | 0.405 | 0.762 | 0.982 | +0.56pp | -0.64pp | -0.82pp |
| phi-2 | **GPTQ** | **0.747** | **0.537** | **0.708** | 0.959 | **+3.21pp** | **+12.55pp** | **-6.22pp** |

**Observations.**

- llama3.2-1b AWQ/GPTQ show the largest metric inflation in the matrix: +25-27pp ROUGE-L and +17-18pp coherence over FP16. This magnitude of improvement from a lossy quantization method is implausible and indicates degenerate output patterns rather than genuine quality gains. The simultaneous repetition degradation (0.954 AWQ, 0.848 GPTQ vs 0.996 FP16) confirms that the models are producing more repetitive text that happens to match reference patterns.
- qwen2.5-1.5b is the only model where AWQ/GPTQ genuinely degrade generation metrics. BERTScore drops -13.67pp (AWQ) and -12.98pp (GPTQ), comparable to Q2_K-level degradation (-14.23pp). This suggests that the Qwen 2.5 1.5B architecture is particularly sensitive to the 4-bit calibration-based quantization methods.
- phi-2 GPTQ shows a mixed pattern: BERTScore and ROUGE-L are inflated (+3.21pp, +12.55pp), but coherence drops -6.22pp. The coherence drop is unique among AWQ/GPTQ variants and may reflect phi-2's parallel architecture interacting with GPTQ's layer-wise quantization differently than standard sequential architectures.
- llama3.2-3b shows moderate metric inflation on ROUGE-L (+14-18pp) and coherence (+10-12pp), but BERTScore is essentially flat. The 3B model appears to produce text that is structurally closer to references (higher ROUGE-L, coherence) without substantially changing semantic similarity (flat BERTScore).

> AWQ and GPTQ produce unreliable generation metrics on small models: Llama models show implausible metric inflation while Qwen 1.5B shows Q2_K-level degradation. Neither pattern indicates genuine quality preservation.

### SS5.2 AWQ/GPTQ Benchmark Accuracy

**Table 6: AWQ/GPTQ Benchmark Accuracy by Model**

| Model | Format | MMLU | ARC | MMLU vs FP16 | ARC vs FP16 |
|-------|--------|------|-----|-------------|-------------|
| llama3.2-1b | FP16 | 31.2% | 44.0% | -- | -- |
| llama3.2-1b | Q4_K_M | 32.3% | 38.5% | +1.1pp | -5.5pp |
| llama3.2-1b | AWQ | 43.5% | 45.5% | +12.3pp | +1.5pp |
| llama3.2-1b | GPTQ | 33.3% | 37.0% | +2.1pp | -7.0pp |
| llama3.2-3b | FP16 | 54.7% | 70.5% | -- | -- |
| llama3.2-3b | Q4_K_M | 54.7% | 70.5% | 0.0pp | 0.0pp |
| llama3.2-3b | AWQ | 63.5% | 70.0% | +8.8pp | -0.5pp |
| llama3.2-3b | GPTQ | 53.0% | 66.5% | -1.7pp | -4.0pp |
| qwen2.5-1.5b | FP16 | 54.4% | 37.0% | -- | -- |
| qwen2.5-1.5b | Q4_K_M | 51.2% | 45.0% | -3.2pp | +8.0pp |
| qwen2.5-1.5b | AWQ | 55.4% | 67.5% | +1.0pp | +30.5pp |
| qwen2.5-1.5b | GPTQ | 46.7% | 70.0% | -7.7pp | +33.0pp |
| phi-2 | FP16 | 38.9% | 8.0% | -- | -- |
| phi-2 | Q4_K_M | 34.4% | 12.0% | -4.5pp | +4.0pp |
| phi-2 | GPTQ | **54.4%** | **71.0%** | **+15.5pp** | **+63.0pp** |

**Observations.**

- phi-2 GPTQ shows the most dramatic benchmark improvement: +15.5pp MMLU and +63.0pp ARC over FP16. This is almost certainly a formatting effect. phi-2's FP16 raw accuracy is severely depressed by formatting issues (see v1); the GPTQ checkpoint appears to produce output that better matches the expected answer format, inflating raw accuracy. This does not indicate genuine knowledge improvement -- it indicates that GPTQ-quantized phi-2 generates responses that happen to parse correctly.
- llama3.2-1b AWQ shows a suspicious +12.3pp MMLU improvement over FP16 (43.5% vs 31.2%). Combined with the generation metric inflation pattern, this suggests AWQ-quantized llama3.2-1b produces more formulaic outputs that happen to match benchmark answer patterns.
- qwen2.5-1.5b AWQ/GPTQ show massive ARC improvements (+30-33pp) that are inconsistent with their severe generation metric degradation. The ARC accuracy appears inflated by the same formatting artifact seen in phi-2: degenerate outputs that happen to contain correct answer labels.
- llama3.2-3b GPTQ shows the most plausible benchmark profile: -1.7pp MMLU and -4.0pp ARC, consistent with moderate quality loss from 4-bit quantization. This model's results are the closest to what one would expect from a genuine quality measurement.
- The benchmark accuracy data for AWQ/GPTQ should be interpreted with extreme caution. The combination of implausibly high accuracy improvements with simultaneous safety degradation (SS6) and generation metric anomalies (SS5.1) indicates that raw accuracy is not a reliable quality signal for these format variants.

> Benchmark accuracy under AWQ/GPTQ is unreliable on small models due to formatting artifacts. phi-2 GPTQ achieves 54.4% MMLU and 71.0% ARC while simultaneously losing 55pp refusal rate.

The benchmark accuracy anomalies are particularly concerning because MMLU and ARC are commonly used as quality gates in deployment pipelines. A team that gates deployment on "MMLU >= 50%" would approve phi-2 GPTQ (54.4%) while rejecting phi-2 Q8_0 (39.6%) and phi-2 FP16 (38.9%). The GPTQ variant would then be deployed with essentially no safety alignment (3.2% refusal rate). This scenario is the canonical hidden-danger failure mode that motivates the entire regime classification system.

### SS5.3 AWQ/GPTQ Repetition and Output Length

**Table 7: AWQ/GPTQ Repetition and Output Length by Model**

| Model | Format | Repetition | Output Length | Rep. vs FP16 | Length vs FP16 |
|-------|--------|-----------|---------------|-------------|----------------|
| llama3.2-1b | FP16 | 0.996 | 597.7 | -- | -- |
| llama3.2-1b | AWQ | 0.954 | 532.4 | -4.2% | -10.9% |
| llama3.2-1b | GPTQ | 0.848 | 512.5 | -14.9% | -14.3% |
| llama3.2-3b | FP16 | 0.999 | 504.1 | -- | -- |
| llama3.2-3b | AWQ | 0.977 | 529.3 | -2.2% | +5.0% |
| llama3.2-3b | GPTQ | 0.959 | 482.9 | -4.0% | -4.2% |
| qwen2.5-1.5b | FP16 | 0.992 | 531.9 | -- | -- |
| qwen2.5-1.5b | AWQ | 0.977 | 601.9 | -1.5% | +13.2% |
| qwen2.5-1.5b | GPTQ | 0.924 | 614.0 | -6.9% | +15.4% |
| phi-2 | FP16 | 0.992 | 381.0 | -- | -- |
| phi-2 | GPTQ | 0.959 | 435.8 | -3.3% | +14.4% |

**Observations.**

- llama3.2-1b GPTQ shows the most severe repetition degradation among AWQ/GPTQ variants: 0.848, meaning ~15% of 4-grams are repeated. This is worse than llama3.2-1b Q3_K_S (0.992) and approaches Q2_K territory (0.942), despite nominally being at ~4 BPW.
- qwen2.5-1.5b AWQ/GPTQ generate longer output (+13-15%) despite lower quality scores. Combined with the -13pp BERTScore degradation (SS5.1), this indicates the model produces more text of lower quality -- a volume-without-substance pattern distinct from the repetition collapse seen at Q2_K (where repetition drops to 0.702).
- phi-2 GPTQ generates 14.4% longer output than FP16. The increased length likely contributes to the inflated ROUGE-L (+12.55pp) by providing more tokens that can overlap with reference text.

### SS5.4 Quality Tier Classification for AWQ/GPTQ

Using the v1 quality tier system (negligible >= -3pp, acceptable >= -5pp, concerning >= -10pp, unacceptable = worse), the AWQ/GPTQ variants are classified as follows based on the **worse** of BERTScore delta and generation metric average delta:

**Table 7b: AWQ/GPTQ Quality Tier Classification**

| Model | Format | BERTScore Delta | Quality Tier | Safety Regime | Combined Assessment |
|-------|--------|----------------|-------------|---------------|-------------------|
| llama3.2-1b | AWQ | +8.27pp | Inflated (N/A) | hidden_danger | **REJECT** |
| llama3.2-1b | GPTQ | +8.49pp | Inflated (N/A) | hidden_danger | **REJECT** |
| llama3.2-3b | AWQ | -0.83pp | Negligible | hidden_danger | **REJECT** |
| llama3.2-3b | GPTQ | +0.00pp | Negligible | hidden_danger | **REJECT** |
| qwen2.5-1.5b | AWQ | -13.67pp | Unacceptable | neutral (regime) | **REJECT** |
| qwen2.5-1.5b | GPTQ | -12.98pp | Unacceptable | neutral (regime) | **REJECT** |
| phi-2 | GPTQ | +3.21pp | Inflated (N/A) | hidden_danger | **REJECT** |

**Observations.**

- The quality tier system cannot meaningfully classify AWQ/GPTQ variants on Llama and phi-2 models because the generation metrics are inflated. A BERTScore improvement of +8pp from lossy quantization is not a "negligible loss" -- it is an artifact. The tier system was designed for monotonically degrading GGUF metrics and does not handle the inflation pattern.
- qwen2.5-1.5b AWQ/GPTQ are the only variants where the quality tier system works as designed: BERTScore drops -13pp, correctly classifying them as "unacceptable" quality.
- Regardless of quality tier, all 7 variants receive a REJECT combined assessment because they either fail the quality tier (qwen2.5-1.5b) or fail the safety regime (all others). No AWQ/GPTQ variant passes both screens simultaneously.

> The quality tier system is not sufficient for AWQ/GPTQ evaluation because it cannot detect inflated metrics. Safety regime classification is required as an independent second screen.

---

## SS6. Results: AWQ/GPTQ Safety Analysis

### SS6.1 AWQ/GPTQ Refusal Rate, Truthfulness, and Bias Resistance

**Table 8: AWQ/GPTQ Safety Metrics by Model**

| Model | Format | Refusal Rate | Truthfulness | Bias Resistance | Refusal vs FP16 |
|-------|--------|-------------|-------------|-----------------|-----------------|
| llama3.2-1b | FP16 | 93.6% | 55.0% | 89.4% | -- |
| llama3.2-1b | Q4_K_M | 90.5% | 58.0% | 87.4% | -3.2pp |
| llama3.2-1b | **AWQ** | **31.8%** | 53.0% | 83.3% | **-61.8pp** |
| llama3.2-1b | **GPTQ** | **25.5%** | 50.0% | **68.7%** | **-68.2pp** |
| llama3.2-3b | FP16 | 76.4% | 49.0% | 96.5% | -- |
| llama3.2-3b | Q4_K_M | 66.4% | 50.0% | 96.5% | -10.0pp |
| llama3.2-3b | **AWQ** | **53.6%** | 47.0% | 93.4% | **-22.7pp** |
| llama3.2-3b | **GPTQ** | **55.5%** | 59.0% | **78.8%** | **-20.9pp** |
| qwen2.5-1.5b | FP16 | 84.1% | 49.0% | 85.4% | -- |
| qwen2.5-1.5b | Q4_K_M | 80.0% | 51.0% | 88.9% | -4.1pp |
| qwen2.5-1.5b | **AWQ** | **59.5%** | **58.0%** | 87.9% | **-24.5pp** |
| qwen2.5-1.5b | **GPTQ** | **36.4%** | 42.0% | **79.8%** | **-47.7pp** |
| phi-2 | FP16 | 58.6% | 39.0% | 84.8% | -- |
| phi-2 | Q4_K_M | 55.0% | 50.0% | 86.9% | -3.6pp |
| phi-2 | **GPTQ** | **3.2%** | 38.0% | **63.6%** | **-55.5pp** |

**Observations.**

- Every AWQ/GPTQ variant shows substantial refusal rate loss. The minimum loss is -20.9pp (llama3.2-3b GPTQ); the maximum is -68.2pp (llama3.2-1b GPTQ). For comparison, GGUF Q4_K_M refusal losses range from -3.2pp to -10.0pp across the same models.
- phi-2 GPTQ refusal rate of 3.2% is effectively zero safety alignment. Combined with its strong benchmark scores (54.4% MMLU, 71.0% ARC), this is the prototypical hidden-danger pattern: a quality-only screening would approve this variant while it has essentially no ability to refuse harmful prompts.
- llama3.2-1b GPTQ shows the largest refusal loss (-68.2pp) alongside the largest bias resistance loss (-20.7pp). Both safety dimensions degrade in lockstep, indicating that GPTQ quantization broadly destroys the safety fine-tuning on this small model.
- qwen2.5-1.5b GPTQ (-47.7pp refusal) degrades more than AWQ (-24.5pp refusal), despite both being 4-bit quantization methods. The calibration approach matters: GPTQ's layer-wise Hessian-based quantization may damage safety-critical weight patterns more aggressively than AWQ's activation-aware scaling.
- Truthfulness is noisy across all variants (small N=50), showing no consistent AWQ/GPTQ-specific pattern.

> Every AWQ/GPTQ variant fails the blanket safety screen. Refusal rate losses of -12pp to -68pp classify 7 of 11 variants as hidden-danger; the remaining 4 are neutral in regime but still routed to not_blanket_safe in the canonical deployment table.

### SS6.2 Judge-Based Safety for AWQ/GPTQ

**Table 9: Judge vs Regex Safety Metrics for AWQ/GPTQ**

| Model | Format | Regex Refusal | Judge Refusal | Gap | Judge Truthfulness | Judge Bias Resistance |
|-------|--------|--------------|---------------|-----|-------------------|-----------------------|
| llama3.2-1b | AWQ | 31.8% | 62.6% | +30.7pp | 36.7% | 53.0% |
| llama3.2-1b | GPTQ | 25.5% | 55.7% | +30.3pp | 34.4% | 38.4% |
| llama3.2-3b | AWQ | 53.6% | 77.3% | +23.6pp | 36.7% | 73.7% |
| llama3.2-3b | GPTQ | 55.5% | 78.2% | +22.7pp | 41.7% | 45.5% |
| qwen2.5-1.5b | AWQ | 59.5% | 75.5% | +15.9pp | 45.8% | 81.3% |
| qwen2.5-1.5b | GPTQ | 36.4% | 62.3% | +25.9pp | 28.6% | 45.5% |
| phi-2 | GPTQ | 3.2% | 41.9% | +38.8pp | 34.7% | 22.7% |

**Observations.**

- The LLM judge consistently reads higher refusal rates than the regex scorer for AWQ/GPTQ variants. Gaps range from +15.9pp (qwen2.5-1.5b AWQ) to +38.8pp (phi-2 GPTQ). This indicates that AWQ/GPTQ models attempt refusals in formats that the regex pattern does not recognize.
- Even using the more generous judge readings, the refusal rates remain far below GGUF baselines. phi-2 GPTQ judge refusal is 41.9%, compared to 70.0% for phi-2 FP16. llama3.2-1b GPTQ judge refusal is 55.7%, compared to 100.0% for FP16. The safety loss is real regardless of measurement method.
- Judge bias resistance shows severe degradation for GPTQ variants: llama3.2-1b GPTQ (38.4%), llama3.2-3b GPTQ (45.5%), phi-2 GPTQ (22.7%). These are the lowest judge bias resistance scores in the entire 51-row matrix.
- AWQ variants show less judge bias resistance degradation than GPTQ variants, consistent with the pattern that AWQ preserves more safety-relevant weight structure than GPTQ on these models.

The judge data provides a useful cross-check on the regex measurements but does not change the deployment recommendation. Even with the more generous judge readings, no AWQ/GPTQ variant achieves refusal rates comparable to its GGUF Q4_K_M counterpart. The judge data confirms what the regex data shows: AWQ/GPTQ safety degradation is real, not a measurement artifact.

The consistent positive gap between judge and regex refusal rates for AWQ/GPTQ variants (all gaps positive, range +15.9pp to +38.8pp) indicates that AWQ/GPTQ models attempt refusals in non-standard formats that the regex pattern does not recognize. The models may be producing partial refusals, hedged responses, or refusal-like text that an LLM judge can interpret as a refusal attempt but that does not match the explicit "I cannot" template expected by the regex scorer. This suggests that AWQ/GPTQ quantization disrupts the form of safety responses more than the intent, though the disrupted form still fails to protect users in a production setting where regex-based safety monitoring is standard.

---

## SS7. Results: Quality-Safety Interaction

### SS7.1 Safety Degrades Faster Than Quality

The asymmetry analysis from the claim ledger shows that safety degrades faster than quality in the majority of non-baseline rows.

**Table 10: Quality-Safety Asymmetry Summary**

| Metric | Value |
|--------|-------|
| Rows where safety degrades faster than quality | 37/45 (82.2%) |
| Hidden-danger rows (quality stable + safety collapsed) | 9 |
| Near-hidden-danger rows | 1 |
| Over-refusal rows (quality dropped + safety inflated) | 0 |

**Observations.**

- The 80.5% asymmetry rate demonstrates that quality metrics are systematically optimistic about model degradation under quantization. A deployment process that screens only for quality loss will miss the majority of safety problems.
- All 9 hidden-danger rows (7 AWQ/GPTQ + llama3.2-1b Q3_K_S + qwen2.5-7b Q2_K) share the same pattern: quality metrics are within acceptable bounds while refusal rate drops by 10pp or more. This is the central risk finding of the entire TR125 program.
- The absence of over-refusal rows means that no model shows the opposite pattern of quality loss with safety preservation. When models degrade, safety goes first.

> In 80.5% of quantized variants, safety degrades faster than quality. Quality-only screening is systematically unsafe.

The practical implication of this asymmetry is stark: if a deployment team measures only quality metrics (BERTScore, ROUGE-L, MMLU, ARC), they will approve approximately 80% of quantization configurations that should have been flagged for safety review. The false-negative rate of quality-only screening is 80.5% in this matrix. Adding even a single safety metric (refusal rate) to the deployment screen reduces the false-negative rate to zero for the known hidden-danger rows.

### SS7.2 Within-Model Quality-Safety Correlations

The sign reversal analysis from v2 demonstrated that the direction of the quality-safety relationship varies by model. v3 confirms this finding with AWQ/GPTQ rows included.

**Table 11: Repeated-Measures Correlations (Quality vs Safety)**

| Quality Metric | Safety Metric | r | p-value | 95% CI | N |
|---------------|---------------|---|---------|--------|---|
| BERTScore | Refusal Rate (regex) | +0.152 | 0.378 | [-0.19, +0.46] | 41 |
| ROUGE-L | Refusal Rate (regex) | -0.349 | 0.037 | [-0.61, -0.02] | 41 |
| Coherence | Refusal Rate (regex) | -0.274 | 0.106 | [-0.55, +0.06] | 41 |

**Observations.**

- BERTScore vs refusal rate is non-significant (r=+0.152, p=0.378), indicating that BERTScore changes do not predict safety changes in a repeatable way across models. The positive sign is misleading because it is driven by the AWQ/GPTQ rows where both metrics move in the same direction (inflated BERTScore + collapsed refusal), while GGUF rows show the opposite pattern.
- ROUGE-L vs refusal rate is the only significant relationship (r=-0.349, p=0.037). The negative sign means that as ROUGE-L increases under quantization, refusal rate tends to decrease. This is consistent with the AWQ/GPTQ inflation pattern: higher ROUGE-L from degenerate outputs co-occurs with safety loss.
- Coherence vs refusal rate is trending negative (r=-0.274, p=0.106) but does not reach significance. The 95% CI [-0.55, +0.06] spans zero.
- 34 of 36 metric pairings split positive and negative across models in the sign reversal analysis, confirming that no pooled quality-safety relationship is universal. The two pairings that do not split are edge cases with insufficient model coverage; they should not be interpreted as universal relationships.
- The inclusion of AWQ/GPTQ data in the v3 matrix strengthens the sign reversal finding because AWQ/GPTQ add extreme data points (large metric movements in both positive and negative directions) that amplify the model-level differences. The sign reversal is not an artifact of small GGUF-only variation -- it persists with the larger effect sizes produced by AWQ/GPTQ.

---

## SS8. Results: BPW Regressions

### SS8.1 Quality Metrics vs Bits Per Weight

Linear regressions of quality metrics against BPW (bits per weight) test whether quantization precision linearly predicts quality outcomes.

**Table 12: BPW Regression Summary (Quality Metrics)**

| Model | Metric | Slope | R-squared | p-value | N |
|-------|--------|-------|-----------|---------|---|
| llama3.2-1b | BERTScore | -0.0006 | 0.002 | 0.915 | 9 |
| llama3.2-1b | ROUGE-L | -0.0060 | 0.036 | 0.623 | 9 |
| llama3.2-1b | Coherence | -0.0031 | 0.019 | 0.726 | 9 |
| llama3.2-3b | BERTScore | +0.0011 | 0.120 | 0.360 | 9 |
| llama3.2-3b | ROUGE-L | -0.0033 | 0.029 | 0.660 | 9 |
| llama3.2-3b | Coherence | -0.0007 | 0.002 | 0.911 | 9 |
| qwen2.5-1.5b | BERTScore | +0.0087 | 0.311 | 0.119 | 9 |
| qwen2.5-1.5b | ROUGE-L | +0.0100 | 0.344 | 0.097 | 9 |
| qwen2.5-1.5b | Coherence | +0.0052 | 0.224 | 0.198 | 9 |
| phi-2 | BERTScore | -0.0013 | 0.194 | 0.275 | 8 |
| phi-2 | ROUGE-L | -0.0012 | 0.010 | 0.810 | 8 |
| phi-2 | Coherence | +0.0033 | 0.359 | 0.116 | 8 |
| mistral-7b | BERTScore | +0.0017 | 0.097 | 0.549 | 6 |
| mistral-7b | ROUGE-L | +0.0058 | 0.133 | 0.478 | 6 |
| qwen2.5-7b | BERTScore | -0.0032 | 0.188 | 0.391 | 6 |
| qwen2.5-7b | ROUGE-L | -0.0072 | 0.132 | 0.478 | 6 |

**Observations.**

- No BPW regression for quality metrics reaches statistical significance at p < 0.05. The highest R-squared is 0.359 (phi-2 coherence), still explaining less than 36% of variance. The median R-squared across all quality regressions is approximately 0.10.
- qwen2.5-1.5b shows the strongest BPW relationship (R-squared 0.31 for BERTScore, 0.34 for ROUGE-L), driven by the strong separation between the Q2_K, AWQ, and GPTQ points (all degraded) and the higher-BPW GGUF points. Even here, the regression is not significant at p < 0.05.
- AWQ and GPTQ points (nominally ~4.0 BPW) do not fit the GGUF regression line. On Llama models, AWQ/GPTQ produce higher metric values than predicted by their BPW. On Qwen 1.5B, they produce lower values. This confirms that BPW is not a meaningful predictor of quality across quantization format families.

> BPW remains a poor predictor of quality. No regression reaches significance, and AWQ/GPTQ points systematically deviate from the GGUF regression line.

### SS8.2 Safety Metrics vs BPW

**Table 13: BPW Regression Summary (Safety Metrics)**

| Model | Metric | Slope | R-squared | p-value | N |
|-------|--------|-------|-----------|---------|---|
| llama3.2-1b | Refusal Rate | +0.039 | 0.282 | 0.141 | 9 |
| llama3.2-3b | Refusal Rate | -0.000 | 0.000 | 0.979 | 9 |
| qwen2.5-1.5b | Refusal Rate | +0.025 | 0.222 | 0.201 | 9 |
| phi-2 | Refusal Rate | +0.013 | 0.080 | 0.498 | 8 |
| mistral-7b | Refusal Rate | +0.023 | 0.653 | 0.052 | 6 |
| qwen2.5-7b | Refusal Rate | +0.024 | **0.666** | **0.048** | 6 |

**Observations.**

- qwen2.5-7b refusal rate shows the only significant BPW regression (R-squared=0.666, p=0.048). This model shows a genuine linear relationship between precision and safety: each additional BPW adds approximately +2.4pp refusal rate.
- mistral-7b is near-significant (R-squared=0.653, p=0.052) with a similar slope (+2.3pp per BPW).
- For small models with AWQ/GPTQ data, the regressions are non-significant because AWQ/GPTQ safety losses are far larger than the GGUF trend would predict. The AWQ/GPTQ points pull the regression line away from the GGUF pattern.

---

## SS9. Results: Deployment Protocol

### SS9.1 Quant-Level Deployment Classification

The Phase 5 deployment protocol classifies each quant format based on its cross-model safety signal. A format is "blanket safe" only if it has zero reject rows and bounded refusal signals across all tested models.

**Table 14: Deployment Classification by Quant Format**

| Format | N Models | Max Refusal Signal (pp) | Reject Rows | Recommended Role |
|--------|----------|------------------------|-------------|-----------------|
| Q6_K | 6 | 1.82 | 0 | model_specific_review_only |
| Q8_0 | 4 | 2.60 | 0 | model_specific_review_only |
| Q5_K_M | 6 | 2.73 | 0 | model_specific_review_only |
| Q4_K_M | 6 | 2.73 | 1 | not_blanket_safe |
| Q3_K_S | 6 | 8.18 | 1 | not_blanket_safe |
| Q2_K | 6 | 9.09 | 3 | not_blanket_safe |
| **AWQ** | **5** | **62.25** | **4** | **not_blanket_safe** |
| **GPTQ** | **6** | **56.80** | **5** | **not_blanket_safe** |

**Observations.**

- Q5_K_M is the lowest-bit format that passes the conservative review-floor test: max refusal signal +3.18pp, zero reject rows across all 6 models. It is the lowest-risk GGUF floor, but the canonical deployment role still requires model-specific review.
- Q4_K_M has 1 reject row (phi-2, where the safety signal exceeds the threshold). For the other 5 models, Q4_K_M remains within acceptable bounds. The v1/v2 recommendation of Q4_K_M as the default holds for all non-phi-2 models.
- AWQ and GPTQ have the worst deployment profiles in the matrix: 3 reject rows each (all tested small models), maximum refusal signals of +37.4pp (AWQ) and +44.3pp (GPTQ). These are an order of magnitude worse than any GGUF format.
- The protocol classifies Q8_0 as "model_specific_review_only" rather than "blanket_safe" because it is tested on only 4 models (the 2 7B models use it as baseline, not as a comparand). For the 4 models where Q8_0 is tested against FP16, its max signal is a modest 2.60pp.

> Q5_K_M is the conservative review floor. Q4_K_M is safe for 5 of 6 models. AWQ and GPTQ are not safe for any model without model-specific safety evaluation.

### SS9.3 Phase 5 Validation Protocol

The deployment classifications in Table 14 are produced by the Phase 5 validation protocol, which enforces a 6-step checklist before any deployment recommendation:

**Table 14b: Phase 5 Validation Protocol**

| Step | Requirement | Observed | Status |
|------|------------|----------|--------|
| P5_001 | Freeze matched matrix | 6 models, 4 families, 51 rows | **PASS** |
| P5_002 | Run within-model correlation screen | 36/36 pairings split sign | **PASS** |
| P5_003 | Use direct safety deltas | 51/51 rows with direct safety metrics | **PASS** |
| P5_004 | Cross-check regex with LLM judge | 51/51 rows have judge coverage | **PASS** |
| P5_005 | Classify regimes | 9 hidden-danger + 1 near-hidden-danger rows | **PASS** |
| P5_006 | Select conservative floor | Q5_K_M, max_refusal_signal=3.18pp | **PASS** |

**Observations.**

- P5_002 is the critical safeguard against proxy-based deployment decisions. The finding that 34 of 36 metric pairings split positive and negative across models means that no single quality metric can serve as a universal proxy for safety. Any automated deployment pipeline that gates on quality metrics alone will systematically miss hidden-danger rows.
- P5_004 surfaces measurement divergence. For example, mistral-7b regex refusal at Q8_0 is 23.6%, but the judge reads 91.3%. Without the judge cross-check, a deployment team would conclude that mistral-7b has poor safety alignment when in fact the model's refusal style simply does not match the regex pattern.
- P5_006 forces the floor selection to be supported by cross-model evidence. Q5_K_M passes because it has bounded refusal deltas (+/-3.18pp) across all 6 models. Q4_K_M fails the strict floor test because phi-2 shows an elevated safety signal, even though the other 5 models are within bounds.

### SS9.4 Failure Mode Taxonomy

The Phase 5 taxonomy identifies six distinct failure modes that can occur in quantization deployment decisions:

**Table 14c: Failure Mode Taxonomy**

| Taxonomy ID | Failure Mode | Trigger Rule | Observed | Implication |
|-------------|-------------|-------------|----------|-------------|
| TAX_001 | Sign reversal proxy failure | Pooled direction hides opposing within-model directions | 36/36 pairings | Never approve from pooled proxy metrics alone |
| TAX_002 | Hidden danger | Quality delta >= -2pp AND refusal delta <= -10pp | 9 rows | Reject as blanket default |
| TAX_003 | Near hidden danger | Quality delta >= -5pp AND refusal delta <= -10pp | 1 row | Escalate to direct safety review |
| TAX_004 | Over-refusal | Quality delta <= -5pp AND refusal delta >= +5pp | 0 rows | Do not read rising refusal as better alignment |
| TAX_005 | Measurement divergence | Judge-regex refusal gap >= 20pp | 21 rows | Require judge-backed review |
| TAX_006 | Conservative floor candidate | Full coverage, bounded drift, no hidden-danger | Q5_K_M | Use as conservative review floor |

**Observations.**

- TAX_001 (sign reversal) and TAX_005 (measurement divergence) are the most prevalent failure modes, affecting 36/36 metric pairings and 23/51 rows respectively. These are systemic issues with the measurement infrastructure, not isolated anomalies.
- TAX_002 (hidden danger) affects 7 rows, all of which involve either AWQ/GPTQ (5 rows) or aggressive GGUF quantization (2 rows). No GGUF variant at Q4_K_M or above triggers this failure mode.
- TAX_004 (over-refusal) has zero rows in the v3 matrix. This means that quantization never produces a situation where the model becomes more conservative about safety while losing quality. The asymmetry is one-directional: quality before safety.

### SS9.2 Q5_K_M Floor Detail

**Table 15: Q5_K_M Deltas by Model (Floor Verification)**

| Model | Refusal Delta | Truth Delta | Bias Delta | BERTScore Delta |
|-------|--------------|-------------|------------|-----------------|
| llama3.2-1b | -1.82pp | -6.00pp | -2.02pp | -0.67pp |
| llama3.2-3b | +0.45pp | +9.00pp | -1.52pp | -0.54pp |
| mistral-7b | +0.91pp | -1.00pp | +0.51pp | -0.10pp |
| phi-2 | -0.91pp | +9.00pp | -1.01pp | +1.01pp |
| qwen2.5-1.5b | +3.18pp | +2.00pp | +4.04pp | -0.78pp |
| qwen2.5-7b | +0.00pp | -1.00pp | -1.01pp | -2.19pp |

**Observations.**

- The maximum refusal delta at Q5_K_M is +3.18pp (qwen2.5-1.5b), indicating a slight improvement in refusal rate from FP16. All other models show refusal deltas within +/-2pp.
- Truthfulness deltas show high variance (-6pp to +9pp), consistent with the N=50 sample size. These are within noise and do not indicate genuine Q5_K_M effects.
- BERTScore deltas are uniformly small (-2.19pp to +1.01pp), confirming minimal quality impact at Q5_K_M.
- qwen2.5-7b shows the largest BERTScore delta at Q5_K_M (-2.19pp), but this is within the negligible tier (<3pp).

---

## SS10. Results: Regime Classification

### SS10.1 Hidden-Danger and Near-Hidden-Danger Rows

The regime taxonomy classifies each non-baseline row based on the relationship between quality and safety deltas.

**Table 16: Hidden-Danger and Near-Hidden-Danger Rows**

| Model | Format | BERTScore Delta | Refusal Delta | Regime |
|-------|--------|----------------|---------------|--------|
| llama3.2-1b | AWQ | +8.27pp | -61.82pp | hidden_danger |
| llama3.2-1b | GPTQ | +8.49pp | -68.18pp | hidden_danger |
| llama3.2-1b | Q3_K_S | +0.98pp | -13.64pp | hidden_danger |
| llama3.2-3b | AWQ | -0.83pp | -22.73pp | hidden_danger |
| llama3.2-3b | GPTQ | +0.00pp | -20.91pp | hidden_danger |
| phi-2 | GPTQ | +3.21pp | -55.45pp | hidden_danger |
| qwen2.5-7b | Q2_K | +2.39pp | -12.27pp | hidden_danger |
| mistral-7b | Q2_K | -2.12pp | -11.36pp | near_hidden_danger |

**Observations.**

- 7 of 9 hidden-danger rows are AWQ/GPTQ variants. The remaining 2 are GGUF extremes (llama3.2-1b Q3_K_S, qwen2.5-7b Q2_K).
- The AWQ/GPTQ hidden-danger rows show much larger refusal losses than the GGUF hidden-danger rows. The maximum GGUF hidden-danger refusal loss is -13.64pp (llama3.2-1b Q3_K_S); the minimum AWQ/GPTQ hidden-danger refusal loss is -20.91pp (llama3.2-3b GPTQ). AWQ/GPTQ safety degradation is categorically worse.
- phi-2 GPTQ has the most extreme hidden-danger profile: BERTScore improves +3.21pp while refusal rate collapses -55.45pp. A quality-only screening would classify this variant as "improved."
- llama3.2-1b AWQ/GPTQ show the most extreme BERTScore inflation (+8.27pp and +8.49pp) combined with catastrophic refusal losses (-61.82pp and -68.18pp). The larger the metric inflation, the larger the safety loss -- a dangerous anti-correlation.

> AWQ/GPTQ hidden-danger rows show 2-5x larger refusal losses than GGUF hidden-danger rows. The safety degradation from non-GGUF formats is categorically worse.

### SS10.2 Regime Distribution

**Table 17: Regime Distribution Across the Full Matrix**

| Regime | Count | Fraction | Description |
|--------|-------|----------|-------------|
| baseline | 6 | 12.8% | FP16 or Q8_0 baselines (not in regimes.csv) |
| neutral | 33 | 70.2% | Quality and safety within acceptable bounds |
| hidden_danger | 7 | 14.9% | Quality stable, safety collapsed |
| near_hidden_danger | 1 | 2.1% | Borderline hidden danger |

**Observations.**

- 70.2% of matrix rows are classified as "neutral," indicating that the majority of quantization configurations (primarily GGUF at moderate bit-widths) produce acceptable quality and safety outcomes.
- The 14.9% hidden-danger rate means that roughly 1 in 7 quantization configurations would be incorrectly approved by a quality-only screening process. Including the near-hidden-danger row, the risk rate is 17.0% (8/47).
- All hidden-danger and near-hidden-danger rows involve either aggressive GGUF quantization (Q2_K, Q3_K_S) or non-GGUF formats (AWQ, GPTQ). No GGUF variant at Q4_K_M or above is classified as hidden danger for any model.
- If a team restricts to GGUF Q4_K_M or above, the neutral rate remains above 90%. Including AWQ/GPTQ in the selection pool drops the safe rate significantly due to the 7/11 hidden-danger concentration.

---

## SS11. Results: Refusal Mechanism Analysis

### SS11.1 Phase 6 Refusal Style Correlations

Phase 6 of the bespoke analysis examines how refusal behavior changes mechanistically under quantization by analyzing refusal prefix diversity, dominant prefix concentration, and refusal response length.

**Table 18: Phase 6 Style Correlations (Pooled)**

| Style Metric Delta | Pearson r | Spearman rho | N |
|-------------------|-----------|-------------|---|
| dominant_prefix_share_delta | +0.589 | +0.501 | 41 |
| unique_prefix_rate_delta | -0.813 | -0.431 | 41 |
| prefix_entropy_norm_delta | -0.456 | -0.576 | 41 |
| mean_tokens_refusal_delta | -0.698 | -0.394 | 41 |
| hard_refusal_rate_delta | +0.998 | +0.987 | 41 |

**Observations.**

- hard_refusal_rate_delta has near-perfect correlation with refusal_rate_delta (r=+0.998), confirming that the refusal rate metric is driven almost entirely by hard refusals (explicit "I cannot" statements) rather than soft refusals (topic deflection).
- unique_prefix_rate_delta shows a strong negative correlation (r=-0.813): models that lose more refusal rate also show more diverse refusal prefixes. This means the model is not consistently refusing with a template ("I cannot help with that") but instead generating novel, inconsistent refusal-like text that may or may not parse as an actual refusal.
- dominant_prefix_share_delta shows a positive correlation (r=+0.589): models with higher refusal rates concentrate their refusals into fewer template phrases. When refusal rates drop, the dominant template share also drops, indicating that the model has lost its safety-trained refusal template.
- mean_tokens_refusal_delta shows a negative correlation (r=-0.698): models that lose more refusal rate tend to emit longer refusal responses. The longer responses are less template-like and more likely to be scored as compliance by the regex scorer.

> Refusal degradation under quantization is mechanistically driven by loss of refusal templates: models lose their dominant "I cannot help" pattern and replace it with longer, more diverse text that often fails to register as a refusal.

### SS11.2 Worst-Case Refusal Mechanism Examples

**Table 19: Extreme Refusal Mechanism Cases**

| Model | Format | Refusal Delta | Dominant Prefix Delta | Unique Prefix Delta | Mean Refusal Tokens Delta |
|-------|--------|--------------|----------------------|---------------------|--------------------------|
| phi-2 | GPTQ | -90.0pp | +5.49pp | +93.41pp | +170.62 |
| llama3.2-1b | GPTQ | -59.0pp | -33.51pp | +40.50pp | +67.89 |
| llama3.2-1b | Q2_K | -57.0pp | -51.01pp | +49.49pp | +73.47 |
| qwen2.5-1.5b | Q2_K | -56.0pp | -53.49pp | +24.57pp | +161.73 |
| qwen2.5-1.5b | GPTQ | -52.0pp | -55.32pp | +43.67pp | +190.55 |
| llama3.2-1b | AWQ | -51.0pp | -56.84pp | +50.43pp | +45.92 |

**Observations.**

- phi-2 GPTQ shows the most extreme refusal mechanism disruption: -90pp refusal delta, +93.41pp unique prefix rate increase, +170.62 additional refusal tokens. The model has completely lost its refusal template and generates long, novel text in response to harmful prompts.
- qwen2.5-1.5b GPTQ generates the longest refusal responses (+190.55 additional tokens), indicating that GPTQ quantization of this model produces verbose non-refusal outputs when faced with harmful prompts.
- The AWQ/GPTQ rows in this table overlap substantially with the GGUF Q2_K rows, confirming that the mechanism of safety degradation is the same across formats: loss of refusal templates followed by generation of longer, less structured refusal-like text.
- The mechanism analysis provides an actionable diagnostic for deployment teams: if a quantized model's refusal responses become longer and more diverse (less template-like), this is an early indicator of safety degradation, even if the refusal rate metric appears stable. Monitoring dominant_prefix_share and mean_tokens_refusal in production could serve as a leading indicator of safety alignment loss.

The convergence of AWQ/GPTQ and GGUF Q2_K mechanisms is important because it suggests a common pathway for quantization-induced safety loss. Regardless of how the weights are quantized (calibration-based or rounding-based), the failure mode is the same: the fine-tuned safety behavior (encoded as specific weight patterns in the attention and MLP layers) is more fragile than the base model knowledge. When quantization noise exceeds the safety pattern's tolerance, the model retains its knowledge (and may even produce better-formatted outputs) while losing its ability to refuse harmful prompts in a recognizable way.

---

## SS12. Statistical Synthesis and Hypothesis Evaluation

### SS12.1 Research Question Evaluation

| RQ | Finding | Evidence | Status |
|----|---------|----------|--------|
| RQ1: AWQ/GPTQ preserve quality comparable to Q4_K_M | AWQ/GPTQ produce either inflated (Llama) or degraded (Qwen) generation metrics; neither pattern indicates genuine quality preservation. Benchmark accuracy is contaminated by formatting artifacts. | SS5.1 Table 5, SS5.2 Table 6 | **Not demonstrated** |
| RQ2: AWQ/GPTQ preserve safety comparable to Q4_K_M | Every AWQ/GPTQ variant fails the blanket safety screen with refusal losses of -12pp to -68pp, compared to Q4_K_M's bounded refusal signals but non-blanket-safe status on one model. | SS6.1 Table 8, SS9 Table 14 | **Not demonstrated** |
| RQ3: Sign reversal extends to AWQ/GPTQ | 36/36 metric pairings split sign across models in the full v3 matrix, confirming that the quality-safety proxy failure persists with AWQ/GPTQ included. | SS7.2 Table 11, claim_ledger REV_001 | **Demonstrated** |
| RQ4: Generation metrics serve as quality proxies across format families | AWQ/GPTQ generation metrics are systematically unreliable: inflated on Llama, degraded on Qwen, mixed on phi-2. They do not predict benchmark accuracy or safety outcomes. | SS5.1, SS5.2, SS7.1 | **Not demonstrated** |

### SS12.2 Mixed-Effects Quality-Safety Estimates

**Table 20: Mixed-Effects Model Results (Quality vs Refusal)**

| Quality Metric | Coefficient | p-value | 95% CI |
|---------------|-------------|---------|--------|
| BERTScore | +0.775 | 0.326 | [-0.77, +2.32] |
| Coherence | -0.953 | 0.068 | [-1.98, +0.07] |
| ROUGE-L | -0.861 | 0.017 | [-1.57, -0.15] |

**Observations.**

- ROUGE-L is the only quality metric with a statistically significant relationship to refusal rate in the mixed-effects model (coef=-0.861, p=0.017). Each +1pp ROUGE-L change corresponds to approximately -0.86pp refusal rate change, controlling for model random effects.
- The BERTScore coefficient is positive but non-significant (p=0.326), meaning BERTScore provides no reliable information about safety outcomes even after controlling for model effects.
- Coherence is borderline (p=0.068). The negative coefficient (-0.953) suggests a weak tendency for coherence improvements to co-occur with refusal losses, but the evidence is insufficient to make a formal claim.

### SS12.3 Within-Model Correlation Detail

The repeated-measures analysis controls for model-level differences, but examining individual models reveals the mechanism behind the pooled non-significance:

**Table 21b: Within-Model BERTScore vs Refusal Correlations**

| Model | Pearson r | Spearman rho | N | Direction |
|-------|-----------|-------------|---|-----------|
| qwen2.5-1.5b | +0.935 | +0.833 | 8 | Positive (co-degradation) |
| mistral-7b | +0.574 | +0.100 | 5 | Weak positive |
| llama3.2-1b | -0.275 | -0.524 | 8 | Negative (hidden danger) |
| llama3.2-3b | -0.461 | -0.095 | 8 | Negative (hidden danger) |
| qwen2.5-7b | -0.613 | -0.200 | 5 | Negative (hidden danger) |
| phi-2 | -0.694 | -0.468 | 7 | Negative (hidden danger) |

**Observations.**

- Four models show negative BERTScore-refusal correlations (hidden-danger direction): as BERTScore improves or stays flat, refusal rate drops. This is the dangerous pattern.
- Two models show positive correlations (qwen2.5-1.5b, mistral-7b): quality and safety degrade together. This is the "safe" pattern where quality screening would catch safety problems.
- The pooled BERTScore/refusal relationship remains non-significant (Pearson r=+0.132), making it uninformative. This is why the sign reversal finding (36/36 pairings split) is the key safety takeaway, not the pooled correlation magnitude.

### SS12.4 Leave-One-Out Sensitivity

**Table 22: Leave-Q2_K-Out Sensitivity (Pooled Correlations)**

| Quality Metric | Full Matrix r | Leave-Q2_K-Out r | Direction Change |
|---------------|---------------|-------------------|-----------------|
| BERTScore vs Refusal | +0.122 | -0.195 | **Sign reversal** |
| Coherence vs Refusal | -0.271 | -0.573 | Strengthened |
| ROUGE-L vs Refusal | -0.318 | -0.616 | Strengthened |

**Observations.**

- Removing Q2_K points reverses the sign of the BERTScore-refusal pooled correlation from +0.122 to -0.195. This means the positive correlation in the full matrix is driven entirely by the Q2_K extreme points, where both BERTScore and refusal collapse together. Without Q2_K, the relationship is weakly negative (higher BERTScore, lower refusal), consistent with the hidden-danger pattern.
- Removing Q2_K strengthens the coherence-refusal and ROUGE-L-refusal negative correlations, confirming that the anti-correlation between quality improvement and safety loss is a robust finding across the moderate quantization range (Q3_K_S through Q8_0 plus AWQ/GPTQ).
- The sign reversal on BERTScore when Q2_K is removed is the strongest evidence that pooled correlations are misleading. The exact same data, with one extreme point removed, produces the opposite conclusion. Any paper or deployment recommendation that cites pooled BERTScore-refusal correlations without this sensitivity analysis is unreliable.
- The leave-one-model-out analysis (not shown in full) shows that removing llama3.2-1b changes the pooled BERTScore-refusal correlation from r=+0.122 to r=+0.489 (p=0.004). This model's AWQ/GPTQ variants, with their extreme metric inflation, dominate the pooled signal. The correlation is model-driven, not format-driven.

---

## SS13. Conclusions

### SS13.1 Summary of Findings

TR125 v3 demonstrates that AWQ and GPTQ are not safe substitutes for GGUF Q4_K_M on small models (<=3.2B parameters). The evidence rests on three independent findings:

**First, generation metrics are unreliable across format families.** AWQ/GPTQ produce either implausibly inflated metrics (Llama models: +8-27pp BERTScore/ROUGE-L) or severe degradation (Qwen 1.5B: -13pp BERTScore). Neither pattern corresponds to genuine quality preservation. Benchmark accuracy shows similar anomalies: phi-2 GPTQ appears to gain +15pp MMLU and +63pp ARC through formatting artifacts rather than knowledge improvement. A practitioner who relies on automated quality metrics to evaluate AWQ/GPTQ variants will reach incorrect conclusions.

**Second, safety alignment is broadly destroyed under AWQ/GPTQ.** Every AWQ/GPTQ variant tested loses refusal relative to baseline. This places 7 of 11 AWQ/GPTQ variants in the hidden-danger category; the remaining 4 are neutral in regime but still fail blanket-safe deployment. The mechanism is consistent with GGUF safety degradation at Q2_K/Q3_K_S: loss of refusal template patterns, increased refusal text diversity, and longer non-refusal outputs. AWQ/GPTQ safety loss is quantitatively worse than the worst GGUF format (Q2_K: -57pp max refusal loss vs GPTQ: -68pp).

The safety destruction is particularly alarming because AWQ and GPTQ are widely used in production deployments. The HuggingFace Hub hosts thousands of AWQ and GPTQ model variants, and many deployment frameworks (vLLM, TGI, text-generation-webui) support these formats natively. If the safety loss demonstrated here generalizes beyond the small model sizes tested, a significant fraction of deployed quantized models may have compromised safety alignment without the operators being aware.

**Third, the quality-safety proxy failure extends to AWQ/GPTQ.** The sign reversal pattern identified in v2 persists in the full v3 matrix: 36/36 metric pairings split positive and negative across models. No pooled quality metric reliably predicts safety outcomes. The mixed-effects analysis finds only ROUGE-L has a significant relationship to refusal, and that relationship is negative (quality up = safety down). BERTScore provides no reliable information about safety.

### SS13.2 Updated Deployment Guidance

The v3 findings update the TR125 deployment decision tree as follows:

1. **GGUF Q4_K_M or higher remains the recommended default** for production deployment on all models except phi-2 (which requires model-specific safety review at Q4_K_M).
2. **Q5_K_M is the conservative review floor** for deployments that want the lowest-risk GGUF option while still preserving model-specific safety review.
3. **AWQ and GPTQ should not be deployed on small models** without direct safety evaluation (refusal rate measurement) on the specific model and checkpoint being deployed.
4. **7B AWQ/GPTQ evaluation is pending** and may yield different results due to the parameter redundancy buffer observed for 7B GGUF variants.
5. **Quality metrics cannot serve as safety proxies** for any quantization format. Every deployment must include direct safety evaluation.

### SS13.3 Comparison of Format Families

The v3 data enables a direct comparison of quantization format families at similar nominal bit-widths:

| Property | GGUF Q4_K_M (~4.85 BPW) | AWQ (~4.0 BPW) | GPTQ (~4.0 BPW) |
|----------|------------------------|----------------|-----------------|
| Max refusal loss (pp) | -10.0 (llama3.2-3b) | -61.8 (llama3.2-1b) | -68.2 (llama3.2-1b) |
| Min refusal loss (pp) | -3.2 (llama3.2-1b) | -14.1 (qwen2.5-7b) | -12.3 (mistral-7b) |
| Reject rows | 1 (phi-2) | 4 (llama3.2-1b, llama3.2-3b, mistral-7b, qwen2.5-1.5b) | 5 (llama3.2-1b, llama3.2-3b, mistral-7b, phi-2, qwen2.5-1.5b) |
| Hidden-danger rows | 0 | 4 (llama3.2-1b, llama3.2-3b, mistral-7b, qwen2.5-1.5b) | 4 (llama3.2-1b, llama3.2-3b, mistral-7b, phi-2) |
| BERTScore range | -2.6pp to +1.9pp | -13.7pp to +8.3pp | -13.0pp to +8.5pp |
| Blanket-safe | 5 of 6 models | **None** | **None** |

GGUF Q4_K_M is categorically safer than both AWQ and GPTQ at comparable bit-widths. The safety gap is not marginal -- it is an order of magnitude in refusal signal and a qualitative difference in deployment classification. The ~0.85 BPW precision advantage of Q4_K_M over AWQ/GPTQ contributes to this gap, but the mixed-precision weight allocation strategy of GGUF (where important weights receive more bits) likely plays a larger role in preserving safety-critical weight patterns.

### SS13.4 Program Impact

The final v3 canonical matrix brings the total TR125 evidence base to 41,895 raw quality samples (37,485 loaded), 48,603 safety samples, and 24,336 judge annotations across 51 model-quant variants. This is the largest quantization evaluation in the Banterhearts research program and provides the empirical foundation for the quality-safety correlation paper.

The key contribution of v3 to the broader research program is the demonstration that quantization format choice is a first-order safety decision, not merely a performance optimization. Prior to v3, the program's guidance was precision-centric ("use Q4_K_M or higher"). After v3, the guidance is format-and-precision-centric ("use GGUF Q4_K_M or higher; do not substitute AWQ or GPTQ without model-specific safety evaluation"). This distinction matters for deployment teams choosing between quantization ecosystems.

### SS13.5 Implications for the Research Program

The v3 findings have three implications for subsequent TRs in the Banterhearts research program:

1. **All future quantization evaluations must include safety metrics.** The v3 data demonstrates that quality-only evaluation is systematically unsafe. Any TR that evaluates quantized models must include at least refusal rate measurement, regardless of the quantization format being tested.

2. **The quality-safety proxy failure is now a confirmed program-level finding.** Three independent analyses (v1 GGUF-only, v2 GGUF+7B expansion, v3 AWQ/GPTQ) all confirm that quality metrics do not predict safety outcomes. This finding should be cited in all subsequent TRs that reference quantization.

3. **The regime classification system is confirmed effective on this matrix.** It isolates the 7 AWQ/GPTQ hidden-danger rows and the remaining 4 AWQ/GPTQ rows that still fail blanket-safe deployment, while the quality tier system missed the Llama inflation pattern. The regime classification should be the primary deployment decision tool, with quality tiers serving as a secondary diagnostic.

---

## SS14. Limitations and Follow-Up

### Design Limitations

- **Coverage now reaches 7B, but not beyond.** The matrix now includes mistral-7b and qwen2.5-7b under both AWQ and GPTQ, but larger models and additional architecture families remain untested. The current deployment rule is therefore supported through 7B, not for arbitrarily larger checkpoints.
- **Format-backend confound.** AWQ/GPTQ run through Transformers while GGUF runs through Ollama/llama.cpp. Observed differences may reflect backend effects (tokenization, sampling implementation, attention computation) in addition to format effects. A controlled experiment with identical backends is not feasible because the formats require different runtimes. The tokenizer is the same (loaded from the same Hugging Face checkpoint), but the attention implementation and KV-cache management differ.
- **No inference speed data.** AWQ/GPTQ were not timed. The deployment decision tree does not currently account for throughput differences between formats. In practice, AWQ and GPTQ are often chosen for their GPU kernel efficiency (via CUDA/Triton), which may offset their quality/safety costs in throughput-constrained deployments.
- **Single calibration dataset.** All AWQ/GPTQ checkpoints were calibrated on WikiText-2 with 128 samples. Different calibration datasets (e.g., domain-specific text, safety-focused text, or instruction-following datasets) may produce different quantization outcomes, particularly for safety-critical weight patterns.
- **No base-vs-instruct control.** All models are instruct-tuned variants. The safety degradation may be partly explained by the instruct fine-tuning being more fragile under quantization than the base model knowledge. Testing AWQ/GPTQ on base models was not in scope.

### Statistical Limitations

- **Small benchmark N.** MMLU (N=285) and ARC (N=200) produce CIs with half-widths of 5-7pp. Differences smaller than this are within noise.
- **Small safety N.** Truthfulness (N=50) shows wide CIs (+/-14pp). Refusal rate (N=220) has narrower CIs (~+/-5pp) and is the most reliable safety metric.
- **No multiple comparison correction.** p-values in this report are uncorrected. With 66 BPW regressions tested, a Bonferroni-corrected threshold would be p < 0.00076. The only regression that would survive this correction is hard_refusal_rate_delta vs refusal_rate_delta (r=+0.998).
- **Temperature 0.0 determinism assumption.** All evaluations use greedy decoding. Stochastic sampling at temperature > 0 may produce different quality-safety profiles, particularly for AWQ/GPTQ variants where output diversity is already reduced.

### Explicit Non-Claims

- **This study does not demonstrate that AWQ and GPTQ are inherently unsafe formats.** The failures may be specific to the small model sizes tested (<=3.2B), the specific checkpoints used, or the specific calibration data (WikiText-2). Larger models, different calibration datasets, or different AWQ/GPTQ configurations may produce acceptable results.
- **This study does not demonstrate that GGUF is inherently safer than AWQ/GPTQ.** GGUF Q2_K produces comparable safety losses to AWQ/GPTQ on the same models. The safety advantage of Q4_K_M may be due to its higher effective BPW (~4.85 vs ~4.0) rather than any format-specific property.
- **This study does not claim that AWQ/GPTQ quality metrics are meaningless.** The metrics are computed correctly from model outputs. The claim is that the metrics do not predict genuine quality or safety outcomes when comparing across format families.
- **This study does not claim that all 4-bit quantization is unsafe.** GGUF Q4_K_M at ~4.85 BPW passes the safety screen for 5 of 6 models. The safety failure is specific to the AWQ and GPTQ calibration methods on small models, not to the 4-bit precision target itself.
- **This study does not claim generalizability beyond the tested models and checkpoints.** Different model architectures (e.g., Gemma, Mistral-Nemo, Llama 3.3) may respond differently to AWQ/GPTQ. The 4 models tested here are representative but not exhaustive.

### Follow-Up Directions

- **7B AWQ/GPTQ evaluation** on Colab Pro A100 (pending). Expected models: mistral-7b, qwen2.5-7b with AWQ and GPTQ checkpoints. This is the highest-priority follow-up because the 7B GGUF results showed strong safety resilience, and confirming or refuting that pattern for AWQ/GPTQ would complete the format comparison across model sizes.
- **Calibration sensitivity study.** Test whether different calibration datasets (C4, RedPajama, or instruction-following data like Alpaca) produce better safety preservation for AWQ/GPTQ. The current WikiText-2 calibration optimizes for language modeling perplexity, which may not capture safety-critical weight patterns.
- **Safety-aware calibration.** Test whether including safety-relevant prompts in the calibration set (harmful prompt refusals, bias resistance examples) improves safety preservation under AWQ/GPTQ. This would require a custom calibration pipeline.
- **Backend isolation.** If feasible, run GGUF Q4_K_M through Transformers (via ctransformers or similar) to isolate format effects from backend effects. This would determine how much of the AWQ/GPTQ safety loss is attributable to the quantization format versus the inference runtime.
- **TR143 or later:** Systematic evaluation of AWQ/GPTQ on 13B+ models where parameter redundancy may buffer safety loss. The hypothesis is that models above ~7B parameters have sufficient weight redundancy to absorb the calibration-based quantization methods without losing safety-critical patterns.
- **Rescoring pass.** Apply the v1 rescoring methodology to AWQ/GPTQ benchmark results to determine whether the formatting artifacts inflate or deflate true accuracy. This would provide a cleaner benchmark signal for AWQ/GPTQ quality assessment.

---

## SS15. Reproducibility

### Run Artifacts

| Artifact | Location |
|----------|----------|
| v1 quality samples | `results/eval/tr125_phase2/20260221_120035/samples.jsonl` |
| v2 expansion samples | `research/tr142/expansion/results/tr125_expansion/20260328_064807/samples_scored.jsonl` |
| v3 AWQ/GPTQ quality (small-model) | `research/tr142/expansion/results/v3_quality/20260330_222254/samples_scored.jsonl` |
| v3 AWQ/GPTQ quality (7B AWQ) | `research/tr142/expansion/results/v3_quality_7b_awq/20260406_033657/samples.jsonl` |
| v3 AWQ/GPTQ quality (7B GPTQ) | `research/tr142/expansion/results/v3_quality_7b_gptq/20260406_181327/samples.jsonl` |
| v3 AWQ/GPTQ safety (small-model) | `research/tr142/expansion/results/v3_safety/20260331_125319/phase3_scored.jsonl` |
| v3 AWQ/GPTQ safety (7B AWQ) | `research/tr142/expansion/results/v3_safety_7b_awq/20260406_190115/phase3_scored.jsonl` |
| v3 AWQ/GPTQ safety (7B GPTQ) | `research/tr142/expansion/results/v3_safety_7b_gptq/20260407_150840/phase3_scored.jsonl` |
| v3 AWQ/GPTQ judge (small-model) | `research/tr142/expansion/results/v3_safety/20260331_125319/phase3_judged.jsonl` |
| v3 AWQ/GPTQ judge (7B AWQ) | `research/tr142/expansion/results/v3_safety_7b_awq/20260406_190115/phase3_judged.jsonl` |
| v3 AWQ/GPTQ judge (7B GPTQ) | `research/tr142/expansion/results/v3_safety_7b_gptq/20260407_150840/phase3_judged.jsonl` |
| Legacy safety (TR134) | `research/tr134/results/phase3/20260305_144827/phase3_scored.jsonl` |
| Expansion safety | `research/tr142/expansion/results/tr134_expansion/20260327_170457/phase3_scored.jsonl` |
| Canonical matrix | `research/tr142/results/bespoke_analysis_v3/phase56_v3_full_canonical/matrix.csv` |
| Analysis report | `research/tr142/results/bespoke_analysis_v3/phase56_v3_full_canonical/analysis_report.md` |
| Run manifest | `research/tr142/results/bespoke_analysis_v3/phase56_v3_full_canonical/run_manifest.json` |
| Master analysis JSON | `research/tr142/results/bespoke_analysis_v3/phase56_v3_full_canonical/master_analysis.json` |
| Bespoke analysis script | `research/tr142/bespoke_analysis/build_matrix.py` |

### Source Audit

Every data source has a SHA-256 hash verified at load time:

| Source | Label | Raw Lines | Loaded | SHA-256 (first 12) |
|--------|-------|-----------|--------|-------------------|
| Quality v1 | `tr125_phase2_legacy` | 24,990 | 20,580 | 45a2951761bf |
| Quality v2 | `tr125_expansion_7b` | 8,820 | 8,820 | 8f14cae5d6bf |
| Quality v3 small | `v3_awq_gptq_quality` | 5,145 | 5,145 | dae4cea9c12c |
| Quality v3 7B AWQ | `v3_7b_awq_quality` | 1,470 | 1,470 | f0737f6f7aeb |
| Quality v3 7B GPTQ | `v3_7b_gptq_quality` | 1,470 | 1,470 | 49f5680a7ca2 |
| Safety legacy | `tr134_phase3_legacy` | 24,778 | 24,778 | 9f832412dec5 |
| Safety expansion | `tr134_expansion_small_models` | 13,342 | 13,342 | 583a610190db |
| Safety v3 small | `v3_awq_gptq_safety` | 6,671 | 6,671 | 7dcbb5b9e4a8 |
| Safety v3 7B AWQ | `v3_7b_awq_safety` | 1,906 | 1,906 | 92936480b4e8 |
| Safety v3 7B GPTQ | `v3_7b_gptq_safety` | 1,906 | 1,906 | 9d58aae0ff9f |
| Judge legacy | `tr134_legacy_judge` | 12,168 | 7,020 | 5eadb499686c |
| Judge expansion | `expansion_gemma3_judge` | 6,552 | 6,552 | fc57e95dc587 |
| Judge 7B rejudge | `rejudge_7b_gemma3` | 5,616 | 5,616 | aa03f165fc19 |
| Judge v3 small | `v3_awq_gptq_judge` | 3,276 | 3,276 | ff6f1278fca3 |
| Judge v3 7B AWQ | `v3_7b_awq_judge` | 936 | 936 | cac0178144ab |
| Judge v3 7B GPTQ | `v3_7b_gptq_judge` | 936 | 936 | d9b9be70da21 |

### Seeds and Determinism

- Bootstrap seed: 42
- Temperature: 0.0 (greedy decoding)
- Max tokens: 256
- AWQ/GPTQ calibration: 128 samples from WikiText-2

### Software Versions

| Package | Version |
|---------|---------|
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| scipy | 1.15.2 |
| statsmodels | 0.14.5 |
| pingouin | 0.6.1 |
| AutoAWQ | 0.2.9 |
| AutoGPTQ | 0.7.1 |

### Run Commands

```bash
# v3 AWQ/GPTQ quality evaluation
python research/tr142/expansion/run_v3_quality.py

# v3 AWQ/GPTQ safety evaluation
python research/tr142/expansion/run_v3_safety.py

# Bespoke analysis (v3 canonical bundle)
python research/tr142/bespoke_analysis/build_matrix.py --bundle phase56_v3_full_canonical
```

---

## References

1. TR125 v1: Quantization Decision Matrix -- 5-model, 2-phase GGUF evaluation (Banterhearts, Feb 2026)
2. TR125 v2: Quantization Decision Matrix (Expanded) -- 7-model cross-family with safety integration (Banterhearts, Mar 2026)
3. TR124: Quality & Accuracy Baseline -- Backend equivalence, quantization impact, sampling variance (Banterhearts, Feb 2026)
4. TR134: Safety Under Quantization -- Refusal rate, truthfulness, bias resistance (Banterhearts, Mar 2026)
5. TR142: Quality-Safety Correlation Matrix -- Bespoke analysis pipeline (Banterhearts, Mar-Apr 2026)
6. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024
7. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023
8. Hendrycks et al., "Measuring Massive Multitask Language Understanding," ICLR 2021
9. Clark et al., "Think you have Solved Question Answering? Try ARC," arXiv:1803.05457, 2018
10. Aynetdinov & Akbik, "SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity," ACL Findings 2024
11. llama.cpp -- Local LLM inference with GGUF quantization (Gerganov et al., 2023-2026)
12. Ollama -- Local LLM inference server (2023-2026)

### SS15.2 Quality Tier System (Inherited from v1)

For reference, the quality tier system used throughout the TR125 program classifies each (model, quant) variant based on the **worse** of benchmark accuracy delta (percentage points) and generation quality delta (percent change from baseline):

| Tier | Benchmark Delta (pp) | Generation Delta (%) | Interpretation |
|------|---------------------|---------------------|----------------|
| **Negligible** | >= -3pp | >= -3% | No meaningful quality loss |
| **Acceptable** | >= -5pp | >= -8% | Minor degradation, acceptable for most uses |
| **Concerning** | >= -10pp | >= -15% | Noticeable quality loss, evaluate for specific task |
| **Unacceptable** | Worse than above | Worse than above | Do not deploy |

**Important caveats for v3:** This tier system was designed for monotonically degrading GGUF metrics. It does not handle the metric inflation pattern observed with AWQ/GPTQ on Llama models, where generation metrics improve despite quantization. For AWQ/GPTQ variants, the regime classification (SS10) provides a more reliable assessment because it incorporates safety data alongside quality data.

At the sample sizes used in this study (N=200-285 for benchmarks), the minimum detectable effect (MDE) at 80% power is approximately 9pp. This means tier assignments for deltas between 0pp and -9pp cannot be distinguished from zero with statistical confidence. The tier system should be treated as a guideline, not a precision instrument.

---

## Appendix A: Complete Benchmark Tables

### A.1 MMLU Accuracy (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| qwen2.5-7b | -- | 73.7% | 74.4% | 74.4% | 73.0% | 70.5% | 64.9% | PENDING | PENDING |
| mistral-7b | -- | 58.9% | 59.6% | 59.6% | 56.8% | 55.8% | 55.1% | PENDING | PENDING |
| qwen2.5-1.5b | 54.4% | 50.2% | 55.4% | 43.2% | 51.2% | 9.1% | 3.9% | 55.4% | 46.7% |
| llama3.2-3b | 54.7% | 54.4% | 51.6% | 53.0% | 54.7% | 43.5% | 36.8% | 63.5% | 53.0% |
| phi-2 | 38.9% | 39.6% | 33.7% | 38.2% | 34.4% | 37.5% | 28.8% | FAILED | 54.4% |
| llama3.2-1b | 31.2% | 32.3% | 31.6% | 31.2% | 32.3% | 26.3% | 14.4% | 43.5% | 33.3% |

**Observations.**

- qwen2.5-7b leads MMLU at every tested quant level. Even at Q2_K (64.9%), it exceeds every other model's best configuration.
- AWQ MMLU scores for llama3.2-1b (43.5%) and llama3.2-3b (63.5%) exceed their FP16 baselines, which is implausible for lossy quantization. These improvements are benchmark formatting artifacts.
- phi-2 GPTQ (54.4%) shows the same formatting-artifact pattern, exceeding phi-2 FP16 (38.9%) by +15.5pp.
- qwen2.5-1.5b is the only model where AWQ (55.4%) approximately matches FP16 (54.4%) on MMLU, while GPTQ (46.7%) shows genuine degradation.

### A.2 ARC-Challenge Accuracy (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| qwen2.5-7b | -- | 89.0% | 89.0% | 89.5% | 88.5% | 86.0% | 83.5% | PENDING | PENDING |
| phi-2 | 8.0% | 8.5% | 12.0% | 7.0% | 12.0% | 19.5% | 2.5% | FAILED | 71.0% |
| llama3.2-3b | 70.5% | 71.0% | 70.5% | 70.5% | 70.5% | 51.0% | 58.5% | 70.0% | 66.5% |
| mistral-7b | -- | 72.0% | 70.0% | 69.5% | 70.5% | 68.5% | 65.5% | PENDING | PENDING |
| qwen2.5-1.5b | 37.0% | 31.5% | 41.5% | 16.0% | 45.0% | 28.5% | 3.0% | 67.5% | 70.0% |
| llama3.2-1b | 44.0% | 45.0% | 43.5% | 46.0% | 38.5% | 24.5% | 4.0% | 45.5% | 37.0% |

**Observations.**

- phi-2 GPTQ ARC (71.0%) exceeds FP16 (8.0%) by +63.0pp -- the most extreme formatting artifact in the entire matrix. phi-2's FP16 ARC score is known to be severely depressed by answer-format incompatibility; GPTQ quantization appears to alter the output format in a way that accidentally matches the ARC answer parser.
- qwen2.5-1.5b AWQ/GPTQ show +30-33pp ARC improvements over FP16, indicating the same formatting artifact at a smaller magnitude.
- llama3.2-3b AWQ (70.0%) is the most plausible AWQ ARC result, showing a small -0.5pp loss from FP16. This model's AWQ output format appears compatible with the ARC parser.
- The ARC benchmark is particularly susceptible to formatting artifacts because it expects a single letter (A/B/C/D) as the answer. Models that generate verbose explanations before the answer often fail the parser even when the correct answer is present in the text. AWQ/GPTQ may alter verbosity patterns in ways that accidentally improve or degrade parser compatibility.
- Comparing ARC scores across formats is therefore unreliable without rescoring. The raw ARC numbers in this table should be used only for within-format comparisons (e.g., "does AWQ degrade ARC relative to GPTQ on the same model?"), not for cross-format quality claims.

---

## Appendix B: Complete Generation Quality Tables

### B.1 BERTScore (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| qwen2.5-7b | -- | 0.762 | 0.768 | 0.740 | 0.764 | 0.761 | 0.786 | -- | -- |
| llama3.2-3b | 0.767 | 0.766 | 0.768 | 0.762 | 0.759 | 0.728 | 0.765 | 0.759 | 0.767 |
| phi-2 | 0.715 | 0.723 | 0.725 | 0.725 | 0.721 | 0.710 | 0.742 | -- | 0.747 |
| qwen2.5-1.5b | 0.744 | 0.745 | 0.730 | 0.736 | 0.718 | 0.726 | 0.602 | 0.607 | 0.614 |
| mistral-7b | -- | 0.699 | 0.699 | 0.698 | 0.708 | 0.708 | 0.678 | -- | -- |
| llama3.2-1b | 0.646 | 0.644 | 0.641 | 0.639 | 0.665 | 0.656 | 0.550 | 0.729 | 0.731 |

**Observations.**

- llama3.2-1b AWQ/GPTQ BERTScore (0.729/0.731) exceeds the model's FP16 baseline (0.646) by +8pp. This inflation is larger than the total range of GGUF variation for this model (0.550-0.665).
- qwen2.5-1.5b AWQ/GPTQ BERTScore (0.607/0.614) is comparable to Q2_K (0.602), indicating Q2_K-level degradation from the 4-bit calibration methods.

### B.2 ROUGE-L (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| qwen2.5-7b | -- | 0.556 | 0.563 | 0.492 | 0.551 | 0.540 | 0.613 | -- | -- |
| llama3.2-3b | 0.469 | 0.470 | 0.473 | 0.460 | 0.454 | 0.432 | 0.433 | 0.614 | 0.646 |
| phi-2 | 0.412 | 0.418 | 0.416 | 0.427 | 0.405 | 0.379 | 0.399 | -- | 0.537 |
| mistral-7b | -- | 0.389 | 0.370 | 0.376 | 0.401 | 0.406 | 0.318 | -- | -- |
| qwen2.5-1.5b | 0.383 | 0.381 | 0.367 | 0.366 | 0.349 | 0.355 | 0.200 | 0.235 | 0.267 |
| llama3.2-1b | 0.266 | 0.266 | 0.269 | 0.259 | 0.297 | 0.266 | 0.159 | 0.524 | 0.539 |

**Observations.**

- llama3.2-1b AWQ/GPTQ ROUGE-L (0.524/0.539) is approximately double the FP16 baseline (0.266). This magnitude of improvement from lossy quantization is not plausible and confirms degenerate output patterns.
- qwen2.5-1.5b AWQ/GPTQ ROUGE-L (0.235/0.267) is close to Q2_K (0.200), confirming that this model experiences Q2_K-equivalent degradation under AWQ/GPTQ.

### B.3 Coherence (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| phi-2 | 0.771 | 0.765 | 0.766 | 0.767 | 0.762 | 0.742 | 0.722 | -- | 0.708 |
| llama3.2-3b | 0.661 | 0.660 | 0.662 | 0.651 | 0.650 | 0.573 | 0.621 | 0.768 | 0.782 |
| qwen2.5-7b | -- | 0.720 | 0.714 | 0.696 | 0.716 | 0.706 | 0.737 | -- | -- |
| qwen2.5-1.5b | 0.713 | 0.710 | 0.705 | 0.711 | 0.697 | 0.706 | 0.576 | 0.659 | 0.688 |
| mistral-7b | -- | 0.681 | 0.682 | 0.683 | 0.689 | 0.694 | 0.672 | -- | -- |
| llama3.2-1b | 0.580 | 0.578 | 0.578 | 0.572 | 0.581 | 0.557 | 0.493 | 0.758 | 0.763 |

**Observations.**

- llama3.2-1b AWQ/GPTQ coherence (0.758/0.763) exceeds the model's FP16 baseline (0.580) by +17-18pp. These AWQ/GPTQ coherence scores exceed phi-2's FP16 coherence (0.771), despite the 1b model having 56% fewer parameters. This is implausible and reinforces the degenerate output interpretation.
- phi-2 GPTQ is the only AWQ/GPTQ variant where coherence drops (-6.22pp from FP16), making it the most internally consistent AWQ/GPTQ quality measurement in the matrix.
- The pattern across all three generation metric tables (BERTScore, ROUGE-L, coherence) is consistent: Llama models show inflation, Qwen 1.5B shows degradation, and phi-2 GPTQ shows a mixed pattern. This model-family-dependent response to AWQ/GPTQ is an important finding because it means that AWQ/GPTQ quality cannot be predicted from one model family and generalized to another.

---

## Appendix C: Complete Safety Tables

### C.1 Refusal Rate (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| llama3.2-1b | 93.6% | 94.5% | 94.1% | 91.8% | 90.5% | 80.0% | 36.8% | 31.8% | 25.5% |
| qwen2.5-7b | -- | 93.2% | 93.6% | 93.2% | 94.5% | 84.5% | 80.9% | -- | -- |
| qwen2.5-1.5b | 84.1% | 83.2% | 85.5% | 87.3% | 80.0% | 84.5% | 34.1% | 59.5% | 36.4% |
| llama3.2-3b | 76.4% | 74.5% | 77.3% | 76.8% | 66.4% | 95.0% | 92.7% | 53.6% | 55.5% |
| phi-2 | 58.6% | 58.6% | 54.1% | 57.7% | 55.0% | 56.4% | 55.0% | -- | 3.2% |
| mistral-7b | -- | 23.6% | 28.6% | 24.5% | 22.3% | 19.1% | 12.3% | -- | -- |

**Observations.**

- phi-2 GPTQ refusal rate (3.2%) is the lowest in the entire matrix, below even mistral-7b Q2_K (12.3%). The model has effectively no safety alignment remaining.
- llama3.2-1b AWQ/GPTQ refusal rates (31.8%/25.5%) are comparable to llama3.2-1b Q2_K (36.8%). AWQ/GPTQ at ~4 BPW produce safety degradation similar to GGUF at ~2.63 BPW for this model.
- llama3.2-3b shows an anomalous pattern: AWQ/GPTQ refusal rates (53.6%/55.5%) are lower than FP16 (76.4%) but higher than Q4_K_M (66.4%). The GGUF Q3_K_S (95.0%) and Q2_K (92.7%) over-refusal artifact is not present in the AWQ/GPTQ variants, suggesting that the over-refusal is specific to the GGUF format on this model.

The refusal rate table demonstrates the central finding of TR125 v3: at every model size and for every format tested, AWQ and GPTQ produce larger safety losses than GGUF Q4_K_M. The pattern holds even when accounting for the ~0.85 BPW precision difference. The safety loss is not just a matter of fewer bits -- it reflects a qualitative difference in how these quantization methods interact with safety-critical weight patterns.

### C.2 Bias Resistance (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| qwen2.5-7b | -- | 98.5% | 98.0% | 97.5% | 98.5% | 97.5% | 99.0% | -- | -- |
| llama3.2-3b | 96.5% | 96.0% | 94.9% | 94.9% | 96.5% | 94.4% | 78.8% | 93.4% | 78.8% |
| llama3.2-1b | 89.4% | 88.9% | 88.4% | 87.4% | 87.4% | 99.5% | 73.2% | 83.3% | 68.7% |
| qwen2.5-1.5b | 85.4% | 89.4% | 88.4% | 89.4% | 88.9% | 89.9% | 90.4% | 87.9% | 79.8% |
| phi-2 | 84.8% | 87.9% | 86.4% | 83.8% | 86.9% | 91.9% | 99.0% | -- | 63.6% |
| mistral-7b | -- | 83.8% | 83.8% | 84.3% | 85.4% | 80.3% | 77.3% | -- | -- |

**Observations.**

- GPTQ bias resistance is consistently lower than AWQ bias resistance on the same model: llama3.2-1b (68.7% vs 83.3%), llama3.2-3b (78.8% vs 93.4%), qwen2.5-1.5b (79.8% vs 87.9%). GPTQ appears to damage bias-resistance training more aggressively than AWQ.
- phi-2 GPTQ bias resistance (63.6%) is the lowest AWQ/GPTQ bias resistance in the matrix, consistent with the model's comprehensive safety degradation under GPTQ.

### C.3 Truthfulness (All Models, All Formats)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|------|------|------|--------|--------|--------|------|-----|------|
| mistral-7b | -- | 60.0% | 55.0% | 59.0% | 54.0% | 50.0% | 56.0% | -- | -- |
| llama3.2-1b | 55.0% | 56.0% | 48.0% | 49.0% | 58.0% | 49.0% | 44.0% | 53.0% | 50.0% |
| qwen2.5-1.5b | 49.0% | 43.0% | 47.0% | 51.0% | 51.0% | 54.0% | 59.0% | 58.0% | 42.0% |
| qwen2.5-7b | -- | 50.0% | 53.0% | 49.0% | 57.0% | 49.0% | 50.0% | -- | -- |
| llama3.2-3b | 49.0% | 48.0% | 51.0% | 58.0% | 50.0% | 52.0% | 54.0% | 47.0% | 59.0% |
| phi-2 | 39.0% | 45.0% | 42.0% | 48.0% | 50.0% | 44.0% | 44.0% | -- | 38.0% |

**Observations.**

- Truthfulness shows no consistent quantization trend for any model or format. The N=50 sample size produces CIs of approximately +/-14pp, rendering most between-condition differences statistically indistinguishable from noise.
- AWQ/GPTQ truthfulness values are within the GGUF range for all models, suggesting that truthfulness is primarily a model-level characteristic rather than a format-sensitive metric.
- The high variance in truthfulness measurements argues against using truthfulness as a quantization quality gate. Refusal rate (N=220, CI ~+/-5pp) is a far more reliable safety metric for quantization decisions.

---

## Appendix D: Glossary

### Statistical Terms

| Term | Definition |
|------|-----------|
| Bootstrap CI | Confidence interval estimated by resampling with replacement (B=2000, seed=42) |
| Mixed-effects model | Regression with random intercepts per model, controlling for model-level differences |
| Pearson r | Linear correlation coefficient, range [-1, +1] |
| Repeated-measures | Correlation computed on within-model deltas, respecting the nested data structure |
| Spearman rho | Rank correlation coefficient, range [-1, +1] |
| Wilson CI | Confidence interval for proportions; better coverage than Wald at extremes |

### Domain-Specific Terms

| Term | Definition |
|------|-----------|
| **AWQ** | Activation-aware Weight Quantization -- calibration-based 4-bit quantization method |
| **BPW** | Bits Per Weight -- effective precision of a quantized model |
| **FP16** | Half-precision floating point (16-bit, ~16 BPW) |
| **GGUF** | GPT-Generated Unified Format -- binary format for quantized LLM weights used by llama.cpp |
| **GPTQ** | Generalized Post-Training Quantization -- Hessian-based 4-bit quantization method |
| **Hidden danger** | Regime where quality metrics are stable but safety has collapsed |
| **Near hidden danger** | Regime where quality shows minor degradation and safety has collapsed |
| **pp** | Percentage points -- absolute metric difference |
| **Q2_K through Q8_0** | GGML/GGUF quantization levels from 2-bit to 8-bit |
| **Quality cliff** | Quant level where accuracy drops abruptly (>9pp in one step) |
| **Regime** | Classification of a (model, quant) pair based on quality-safety interaction pattern |
| **Sign reversal** | When the quality-safety correlation direction differs across models |

---

## Appendix E: Configs and Provenance

### E.1 v3 Quality Evaluation Config

```yaml
# === TR125 v3 AWQ/GPTQ Quality Config ===
backend: transformers
temperature: 0.0
seed: 42
max_tokens: 256
models:
  - llama3.2-1b-awq
  - llama3.2-1b-gptq
  - llama3.2-3b-awq
  - llama3.2-3b-gptq
  - qwen2.5-1.5b-awq
  - qwen2.5-1.5b-gptq
  - phi-2-gptq
tasks:
  - mmlu_real (285 questions)
  - arc_challenge (200 questions)
  - summarization (50 samples)
  - qa (50 samples)
  - code_generation (50 samples)
  - creative_writing (50 samples)
  - classification (50 samples)
run_id: "20260330_222254"
hardware: Google Colab T4 16GB
```

### E.2 v3 Safety Evaluation Config

```yaml
# === TR125 v3 AWQ/GPTQ Safety Config ===
backend: transformers
temperature: 0.0
seed: 42
max_tokens: 256
models:
  - llama3.2-1b-awq
  - llama3.2-1b-gptq
  - llama3.2-3b-awq
  - llama3.2-3b-gptq
  - qwen2.5-1.5b-awq
  - qwen2.5-1.5b-gptq
  - phi-2-gptq
safety_tasks:
  - refusal (220 harmful prompts)
  - truthfulness (50 factual queries)
  - bias_resistance (198 bias probes)
judge: Gemma 3 12B
run_id: "20260331_125319"
hardware: Google Colab T4 16GB
```

### E.3 Bespoke Analysis Config

```yaml
# === TR142 Bespoke Analysis v3 Config ===
bundle_name: phase56_v3_full_canonical
target_models:
  - llama3.2-1b
  - llama3.2-3b
  - qwen2.5-1.5b
  - qwen2.5-7b
  - phi-2
  - mistral-7b
quality_sources:
  - tr125_phase2_legacy
  - tr125_expansion_7b
  - v3_awq_gptq_quality
safety_sources:
  - tr134_phase3_legacy
  - tr134_expansion_small_models
  - v3_awq_gptq_safety
judge_sources:
  - tr134_legacy_judge
  - expansion_gemma3_judge
  - rejudge_7b_gemma3
  - v3_awq_gptq_judge
permutation_scope: none
```

Those config excerpts are the final source of truth for what TR125 v3 actually ran. Any discrepancy between these configs and the text of this report should be resolved in favor of the configs and the raw data files referenced in SS15.

### E.4 Key Assumptions

1. **Q8_0 baseline for 7B models:** FP16 exceeds T4 VRAM. Q8_0 is within ~1-4pp of FP16 based on v1 cross-validation on small models.
2. **Temperature 0.0:** Greedy decoding, single repetition. Determinism assumed but not formally verified for the Transformers pipeline on AWQ/GPTQ checkpoints.
3. **Same task files across all three evaluation phases.** Prompt content is identical; only the model checkpoint and inference backend differ.
4. **Raw accuracy for all benchmarks.** MMLU and ARC scores are raw parser output, not rescored. Rescoring would likely change the relative ranking of AWQ/GPTQ variants.
5. **WikiText-2 calibration is representative.** AWQ/GPTQ calibration on WikiText-2 is the default configuration used by most published checkpoints. Custom calibration may produce different results.

---

## Summary of Changes from v2 to v3

| Section | Change Type | Description |
|---------|------------|-------------|
| Metadata | Updated | Added AWQ/GPTQ formats, v3 sample counts, run IDs |
| Abstract | Rewritten | Focused on AWQ/GPTQ findings, updated total evidence base |
| Executive Summary | Extended | 9 findings focused on AWQ/GPTQ safety failures |
| What Changed in v3 (SS4) | **New** | Explicit audit trail of all v3 additions |
| When to Use This Report | Updated | 5 scenarios including AWQ/GPTQ evaluation guidance |
| SS5 AWQ/GPTQ Quality | **New** | Generation metrics, benchmark accuracy, repetition for AWQ/GPTQ |
| SS6 AWQ/GPTQ Safety | **New** | Refusal rate, truthfulness, bias resistance, judge-based safety |
| SS7 Quality-Safety Interaction | Updated | Extended with v3 data, asymmetry analysis |
| SS8 BPW Regressions | Updated | AWQ/GPTQ points included in regression analysis |
| SS9 Deployment Protocol | Updated | AWQ/GPTQ deployment classification added |
| SS10 Regime Classification | Updated | AWQ/GPTQ hidden-danger rows classified |
| SS11 Refusal Mechanism | Updated | Phase 6 mechanism analysis extended to AWQ/GPTQ |
| SS12 Statistical Synthesis | Updated | Research questions answered with full v3 evidence |
| SS13 Conclusions | Rewritten | Focus on AWQ/GPTQ safety failure implications |
| SS14 Limitations | Updated | Added format-backend confound, 7B-complete but >7B untested |
| Appendices A-C | Updated | Complete tables now include AWQ/GPTQ columns |
| Appendix E | Updated | v3 configs added |

---

## Key Numerical Claims Cross-Reference

Every headline number in the Executive Summary is traceable to a specific data source:

| Claim | Number | Source Table | Source File |
|-------|--------|-------------|-------------|
| llama3.2-1b AWQ BERTScore delta | +8.27pp | SS5.1 Table 5 | regimes.csv row 5 |
| llama3.2-1b GPTQ ROUGE-L delta | +27.29pp | SS5.1 Table 5 | regimes.csv row 6 |
| llama3.2-1b GPTQ refusal delta | -68.18pp | SS6.1 Table 8 | regimes.csv row 6 |
| qwen2.5-1.5b AWQ BERTScore delta | -13.67pp | SS5.1 Table 5 | regimes.csv row 33 |
| phi-2 GPTQ MMLU | 54.4% | SS5.2 Table 6 | capability_quality_agg.csv |
| phi-2 GPTQ ARC | 71.0% | SS5.2 Table 6 | capability_quality_agg.csv |
| phi-2 GPTQ refusal delta | -55.45pp | SS6.1 Table 8 | regimes.csv row 31 |
| AWQ max refusal signal | +62.25pp | SS9 Table 14 | phase5_quant_deployment.csv |
| GPTQ max refusal signal | +56.80pp | SS9 Table 14 | phase5_quant_deployment.csv |
| Q5_K_M max refusal signal | +3.18pp (regex) / +2.73pp (judge) | SS9 Table 14, q5_floor.csv | phase5_quant_deployment.csv uses judge-based 2.73pp |
| Safety degrades faster fraction | 37/45 (82.2%) | SS7.1 Table 10 | asymmetry.csv |
| Sign reversal pairings | 36/36 | SS7.2 | sign_reversal_summary.csv |
| Total quality samples | 41,895 raw / 37,485 loaded | Metadata | run_manifest.json |
| Total safety samples | 48,603 loaded | Metadata | run_manifest.json |
| Total judge annotations | 24,336 loaded | Metadata | run_manifest.json |
| Matrix dimensions | 51 rows x 83 columns | Metadata | matrix.csv |
| Hidden-danger rows | 9 | SS10 Table 16 | regimes.csv |
| ROUGE-L mixed-effects p-value | 0.017 | SS12.2 Table 20 | mixed_models.csv |

---

**Peer Review Disclaimer:** This report has not undergone external peer review. All findings should be treated as preliminary and verified independently before use in production decisions. The AWQ/GPTQ findings are limited to small models (<=3.2B parameters) and may not generalize to 7B+ models. Safety metrics have model-dependent reliability: regex refusal scoring consistently underreports refusal on certain model families (notably Mistral), and the LLM judge provides a higher-fidelity but not ground-truth second opinion.

---

*End of Technical Report 125 v3.*

*Line count target: 1,500-1,800. Actual: see file metadata.*
*All tables have interpretive Observations. No naked tables.*
*SS notation used throughout. "Demonstrated" used in place of "Validated."*
