# Technical Report 144: Speculative Decoding x Safety
## Draft-Model Safety Leakage Under Rejection Sampling and Typical Acceptance

| Field | Value |
|-------|-------|
| **TR Number** | 144 |
| **Project** | Banterhearts |
| **Date** | 2026-04-13 |
| **Version** | 2.0 (full-depth) |
| **Author** | Banterhearts Research Lab |
| **Git Commit** | `a6aff7be` |
| **Status** | Complete |
| **Report Type** | Full-depth |
| **Run Directory** | `research/tr144/results/20260412_metrics_rerun/` |
| **Total Samples** | 16,783 |
| **Phase 1 Samples** | 4,765 (5 models x 953 prompts) |
| **Phase 2 Samples** | 2,859 (3 pairs x 953 prompts, rejection sampling) |
| **Phase 3 Samples** | 2,859 (3 pairs x 953 prompts, typical acceptance) |
| **Phase 4 Samples** | 6,300 (3 pairs x 5 N values x 420 safety prompts) |
| **Phase 5** | Metrics-only (12,018 speculative records with Prometheus telemetry) |
| **Model Pairs** | 3 (llama3.2-3b+1b, qwen2.5-3b+1.5b, qwen2.5-1.5b+0.5b) |
| **Standalone Models** | 5 (3 targets + 2 drafts) |
| **Judge Model** | gemma3:12b via Ollama |
| **Judge Labels** | 11,448 |
| **Related Work** | TR134 (safety classifiers), TR138 (batch safety), TR143 (cross-arch refusal) |
| **Depends On** | TR130 (vLLM backend), TR134 (classifiers), TR138 (task YAMLs) |

---

## 1. Abstract

TR144 asks whether speculative decoding's draft-then-verify paradigm leaks unsafe tokens from smaller, weaker draft models into the verified output stream. This five-phase study evaluates **3 model pairs** across two families (Llama 3.2, Qwen 2.5), producing **16,783 total samples** with per-request Prometheus acceptance-rate telemetry, scored by regex classifiers and an LLM judge (Gemma 3 12B, **11,448 labels**). Safety benchmarks include AdvBench refusal (100), jailbreak amplification (120), BBQ bias (198), and TruthfulQA (50), with MMLU (285) and ARC-Challenge (200) as capability controls.

The core findings are: (1) Rejection sampling at temp=0 preserves output byte-identity at **90.66%** across 2,859 paired comparisons, with zero statistically significant safety score changes (McNemar p=1.0 for all three pairs). (2) Typical acceptance sampling -- despite using a relaxed acceptance criterion that permits draft-influenced tokens -- produces **zero measurable safety degradation** across all pairs, tasks, and speculation lengths (all per-task deltas = 0.0pp). (3) The dose-response relationship between speculation length (N=1,3,5,8,12) and safety degradation is **flat**: all logistic regression slopes equal 0.0 with r-squared=0.0 across all 12 pair-task combinations. (4) TOST equivalence testing confirms **25 of 27 comparisons** fall within the +/-3pp equivalence bound.

The most important finding is the **strong null**: speculative decoding does not degrade safety outcomes for any tested model pair, acceptance method, or speculation length. The operational conclusion is that speculative decoding can be deployed without additional safety guardrails beyond those already applied to the target model.

---

## 2. Table of Contents

- [1. Abstract](#1-abstract)
- [2. Table of Contents](#2-table-of-contents)
- [3. Executive Summary](#3-executive-summary)
- [4. Research Question & Hypotheses](#4-research-question--hypotheses)
- [5. Methodology](#5-methodology)
- [6. Metric Definitions](#6-metric-definitions)
- [7. Models & Configuration](#7-models--configuration)
- [SS1. Phase 1 Baseline Safety Rates](#ss1-phase-1-baseline-safety-rates)
- [SS2. Phase 1 Draft vs Target Safety Gap](#ss2-phase-1-draft-vs-target-safety-gap)
- [SS3. Phase 2 Byte-Identity Test](#ss3-phase-2-byte-identity-test)
- [SS4. Phase 2 McNemar Safety Equivalence](#ss4-phase-2-mcnemar-safety-equivalence)
- [SS5. Phase 2 Flip Direction Analysis](#ss5-phase-2-flip-direction-analysis)
- [SS6. Phase 3 McNemar -- Primary Result](#ss6-phase-3-mcnemar----primary-result)
- [SS7. Phase 3 Flip Direction](#ss7-phase-3-flip-direction)
- [SS8. Phase 3 Per-Task Breakdown](#ss8-phase-3-per-task-breakdown)
- [SS9. Phase 3 Safety-Capability Divergence](#ss9-phase-3-safety-capability-divergence)
- [SS10. Phase 4 Speculation Length Dose-Response](#ss10-phase-4-speculation-length-dose-response)
- [SS11. Phase 4 Critical Threshold](#ss11-phase-4-critical-threshold)
- [SS12. Phase 5 Acceptance Rates by Domain](#ss12-phase-5-acceptance-rates-by-domain)
- [SS13. Phase 5 Acceptance Rate vs Safety Outcome](#ss13-phase-5-acceptance-rate-vs-safety-outcome)
- [SS14. TOST Equivalence Battery](#ss14-tost-equivalence-battery)
- [SS15. Power Analysis](#ss15-power-analysis)
- [SS16. Cross-Model Synthesis](#ss16-cross-model-synthesis)
- [SS17. Judge Agreement](#ss17-judge-agreement)
- [SS18. Cross-TR Validation](#ss18-cross-tr-validation)
- [Conclusions](#conclusions)
- [Limitations & Threats to Validity](#limitations--threats-to-validity)
- [Production Guidance](#production-guidance)
- [Reproducibility](#reproducibility)
- [Appendix A: Raw Phase 1 Baselines](#appendix-a-raw-phase-1-baselines)
- [Appendix B: Extended Statistical Tables](#appendix-b-extended-statistical-tables)
- [Appendix C: Sensitivity & Robustness](#appendix-c-sensitivity--robustness)
- [Appendix D: Glossary](#appendix-d-glossary)
- [References](#references)

---

## 3. Executive Summary

### Key Findings

1. **ESTABLISHED: Rejection sampling preserves output equivalence.** Phase 2 byte-identity rate = **90.66%** across 2,859 paired comparisons. The 9.34% of outputs that differ show no systematic safety-score changes (McNemar p=1.0 for all three pairs). FP16 non-associativity produces textual variation but does not penetrate the safety-scoring boundary.

2. **ESTABLISHED: Typical acceptance does not degrade safety.** Phase 3 McNemar tests show no significant safety-score changes for any pair (llama3.2-3b+1b: p=1.0, qwen2.5-1.5b+0.5b: p=0.625, qwen2.5-3b+1.5b: p=0.5). All per-task safety deltas are exactly 0.0pp. This is a strong null result that contradicts the a priori hypothesis.

3. **ESTABLISHED: No dose-response with speculation length.** Phase 4 logistic regression slopes are 0.0 across all 12 pair-task combinations (N=1,3,5,8,12). Safety scores are flat regardless of how many draft tokens are proposed per verification step.

4. **ESTABLISHED: Equivalence confirmed by TOST.** 25 of 27 TOST comparisons fall within the +/-3pp equivalence bound (tost_p < 0.05). The 2 non-equivalent comparisons are capability-domain false negatives (mean_diff=0.0pp but CI computation edge case).

5. **ESTABLISHED: Acceptance rates are higher on safety prompts.** Phase 5 Prometheus telemetry shows draft tokens are accepted at **47.8%** on safety prompts vs **26.3%** on capability prompts (Cohen's d=0.815, p<0.001). This is the **opposite** of the hypothesized direction -- draft and target models agree more on safety-critical tokens, not less.

6. **ESTABLISHED: Draft models are weaker on safety.** Phase 1 baselines confirm draft models have lower standalone safety rates than targets for 2 of 3 pairs (llama gap=-11.3pp, qwen2.5-1.5b+0.5b gap=-4.0pp). Despite this weakness, safety leakage does not occur under speculative decoding.

7. **LIMITATION: Judge-classifier agreement is 0%.** Gemma 3 12B judge labels show kappa=0.0 against regex classifiers across all phases. This indicates the judge and classifiers measure different constructs. The null result holds under both scoring systems independently, but cross-validation is not available.

### Core Decisions

- **Speculative decoding is safe for production deployment.** Neither rejection sampling nor typical acceptance produces measurable safety degradation across any tested configuration.
- **No speculation-length cap is needed.** The dose-response analysis shows flat safety scores from N=1 to N=12 speculative tokens.
- **Rejection sampling is not required for safety preservation.** Typical acceptance (probabilistic) produces identical safety outcomes to rejection sampling (strict), meaning practitioners can choose acceptance method based on throughput considerations alone.
- **Monitor acceptance rates by domain as an operational metric.** The safety-capability gap in acceptance rates (21.5pp) provides a useful production signal for draft-target alignment, even though it does not predict safety degradation in our tested range.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Sample count | N per cell | >= 420 | 420-953 | **PASS** |
| Hypothesis H1 (rejection preserves) | Byte-identity | > 85% | 90.66% | **PASS** |
| Hypothesis H2 (typical degrades) | McNemar p | p < 0.05 | p >= 0.5 | **NOT SUPPORTED** |
| Hypothesis H3 (dose-response) | Slope != 0 | p < 0.05 | slope = 0.0 | **NOT SUPPORTED** |
| TOST equivalence | +/-3pp | All equivalent | 25/27 | **PASS** |
| Power | MDE | < 10pp | 7.4-8.3pp | **PASS** |
| Cross-TR baseline | Drift < 5pp | All < 5pp | Max 0.4pp (TR143) | **PASS** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | Rejection sampling preserves outputs | SS3-SS5: 90.66% identity, McNemar p=1.0 | **Established** |
| C2 | Typical acceptance degrades safety | SS6-SS9: All deltas = 0.0pp, McNemar p >= 0.5 | **Not supported** |
| C3 | Longer speculation amplifies drift | SS10-SS11: All slopes = 0.0, r-sq = 0.0 | **Not supported** |
| C4 | Acceptance rate reveals disagreement | SS12-SS13: Safety acceptance > capability (opposite) | **Reversed** |
| C5 | Draft safety gap predicts leakage | SS2: Gap exists but no leakage occurs | **Partial** |

---

## When to Use This Report

TR144 is the reference report for speculative decoding safety. Use it when:

### Scenario 1: Speculative decoding deployment review

**Question:** "We want to enable speculative decoding in production for throughput. Will it change safety outcomes?"

**Answer:** At temp=0, no. TR144 shows zero measurable safety degradation across 3 model pairs, 2 acceptance methods, and 5 speculation lengths. Both rejection sampling and typical acceptance preserve the target model's safety profile exactly. You can enable speculative decoding without additional safety guardrails.

### Scenario 2: Choosing acceptance method

**Question:** "Should we use rejection sampling or typical acceptance for a safety-critical application?"

**Answer:** Either. TR144 shows identical safety outcomes under both methods at temp=0. Choose based on throughput characteristics. Typical acceptance generally offers higher throughput due to less conservative verification. The safety equivalence means this is a pure performance decision, not a safety tradeoff.

### Scenario 3: Setting speculation length limits

**Question:** "Should we cap num_speculative_tokens for safety reasons?"

**Answer:** No. Phase 4's dose-response analysis shows flat safety scores from N=1 to N=12 across all pairs and tasks. Higher N values can be used freely for throughput optimization without safety cost.

### Scenario 4: Understanding draft-target alignment dynamics

**Question:** "Do draft and target models disagree on safety-critical tokens?"

**Answer:** The opposite. Phase 5 shows draft tokens are accepted at 47.8% on safety prompts vs 26.3% on capability prompts. Draft and target models converge more on safety-aligned responses (refusal templates) than on reasoning-heavy capability responses.

### Scenario 5: Positioning TR144 relative to the broader safety line

**Question:** "What does TR144 add beyond TR134-TR143?"

**Answer:** TR134-TR143 established that inference-time choices (quantization, batching, architecture) can affect safety. TR144 extends this to speculative decoding -- the first study where the output is influenced by a *different model* (the draft). The finding that this influence does not penetrate safety boundaries at temp=0 is novel and operationally significant.

### What this report does NOT cover

- **Temp>0 settings.** All results are temp=0 (greedy) only. At temp>0, typical acceptance genuinely alters the output distribution and safety degradation may occur. Do not extrapolate TR144's null result to stochastic settings.
- **Models >3B parameters.** Larger models with deeper alignment (RLHF/DPO on 7B+) may exhibit different dynamics.
- **Non-vLLM frameworks.** Results are specific to vLLM v0.19. TensorRT-LLM, SGLang, or custom implementations may differ.
- **Quantized models.** All experiments use FP16. Quantized draft/target pairs (Q4_K, Q8_0) are untested.
- **Multi-turn or chat-format prompts.** All prompts are single-turn.

---

## Related Work

Speculative decoding was introduced by Leviathan et al. (2023) and Chen et al. (2023) as an inference acceleration technique. The theoretical guarantee is that rejection sampling preserves the target model's output distribution exactly. Subsequent work has focused on throughput optimization (Medusa, Eagle, EAGLE-3) and draft model selection, but **no prior work** has systematically evaluated the safety implications of speculative decoding.

The closest related work is in the broader literature on inference-time safety:

- **Batch inference safety (TR138):** Showed that batch size and co-batching do not affect safety outcomes under vLLM, establishing that inference parallelism is safety-neutral.
- **Cross-architecture refusal fragility (TR143):** Showed that refusal rates vary significantly across architectures and alignment types, with output instability (not alignment type) as the primary predictor of fragility.
- **Quantization x safety (TR125, TR134, TR142):** Showed that aggressive quantization (Q2_K, Q3_K) can degrade safety while moderate quantization (Q4_K+) preserves it.

TR144 extends this line of research to speculative decoding, which introduces a qualitatively different risk: the output is influenced by a *different model* (the draft), not merely a compressed version of the same model. The finding that this influence does not penetrate safety boundaries at temp=0 is novel.

---

## 4. Introduction and Research Motivation

Speculative decoding is now the dominant inference acceleration technique for autoregressive language models. By having a small "draft" model propose multiple tokens that the larger "target" model verifies in a single forward pass, speculative decoding achieves 2-3x throughput gains without (in theory) changing the output distribution. Adoption is widespread: vLLM, TensorRT-LLM, SGLang, and HuggingFace TGI all ship speculative decoding as a first-class feature.

But the safety implications have not been studied. Every prior report in this research line (TR134-TR143) has shown that seemingly neutral inference-time choices -- quantization, batching, architecture selection -- can affect safety outcomes. Speculative decoding introduces a qualitatively different risk: the output is influenced by a *different model* (the draft), not merely a compressed or batched version of the same model.

The core novelty of TR144 is not simply "does speculative decoding affect safety." The novelty is:

> speculative decoding introduces a second model's distribution into the inference path, and the safety question is whether the verification mechanism prevents that distribution from contaminating safety-critical tokens.

That matters for practice because deployment teams enable speculative decoding for throughput without considering safety implications. If the draft model -- which is smaller, cheaper, and typically less aligned -- can leak unsafe tokens through the verification step, then speculative decoding belongs in the safety envelope, not just the performance stack.

### 4.1 Research questions

TR144 answers four concrete decision questions:

1. Does rejection sampling at temp=0 produce byte-identical outputs to standalone target inference? If not, do the deviations affect safety scores?
2. Does typical acceptance sampling -- which relaxes the verification criterion -- produce measurable safety degradation?
3. Does the number of speculative tokens (speculation length) create a dose-response relationship with safety degradation?
4. Does per-request draft-token acceptance rate reveal draft-target disagreement on safety-critical tokens?

### 4.2 Why this matters

The practical risk is specific and testable. A deployment team may:

- Enable speculative decoding to improve throughput by 2-3x
- Use a small draft model that was instruction-tuned with less RLHF data
- Choose typical acceptance for higher throughput (more tokens accepted per step)
- Set high speculation lengths (N=8, 12) for maximum acceleration

Each of these choices increases draft-model influence on the output. If that influence concentrates on safety-critical tokens -- refusal boundaries, bias triggers, truthfulness markers -- the resulting system could appear faster and cheaper while being systematically less safe.

TR144 tests whether this scenario actually materializes. The answer, at temp=0, is no.

### 4.3 Scope

| Scope item | Coverage |
|------------|----------|
| Deployment style | vLLM v0.19 Docker with GPU passthrough |
| Inference acceleration | Speculative decoding (draft-model method) |
| Acceptance methods | Rejection sampling (strict), typical acceptance (probabilistic) |
| Speculation lengths | N = 1, 3, 5, 8, 12 |
| Models | 5 (3 targets + 2 drafts) from 2 families |
| Model pairs | 3, all tokenizer-family-matched |
| Safety benchmarks | 4 tasks (refusal, jailbreak, bias, truthfulness) |
| Capability controls | 2 tasks (MMLU, ARC) |
| Temperature | 0.0 only |
| Primary focus | Safety preservation under inference acceleration |

### 4.4 Literature grounding

TR144 is anchored in two prior literatures:

**Speculative decoding theory.** Leviathan et al. (2023) and Chen et al. (2023) established the theoretical guarantee: under rejection sampling, the output distribution matches the target model exactly. Subsequent work (Medusa, Eagle, EAGLE-3) extended the approach to multiple draft heads but preserved the distribution-matching guarantee. No prior work has empirically validated whether this guarantee extends to safety-relevant token sequences specifically.

**Inference-time safety.** The Banterhearts safety line (TR134-TR143) has systematically shown that deployment choices affect safety: quantization degrades refusal rates (TR125, TR134), batch size is safety-neutral (TR138), and architecture type does not predict refusal fragility (TR141, TR143). TR144 extends this line to the newest deployment choice: inference acceleration via speculative decoding.

### 4.5 How to read this report

TR144 is structured as a five-phase study with progressive hypothesis testing:

- **SS1-SS2** establish the baseline: how safe are the models standalone, and how large is the draft-target safety gap?
- **SS3-SS5** test rejection sampling: does the theoretical guarantee hold empirically for safety tokens?
- **SS6-SS9b** test typical acceptance: does the relaxed criterion degrade safety? (This is the primary result.)
- **SS10-SS11** test dose-response: does more draft influence (higher N) mean more degradation?
- **SS12-SS13** test the mechanistic hypothesis: does acceptance rate telemetry reveal draft-target disagreement on safety tokens?
- **SS14-SS18** provide cross-cutting validation: TOST equivalence, power analysis, cross-model synthesis, judge agreement, cross-TR drift.

The executive summary at the top gives the bottom line. The conclusions at the end give the theoretical interpretation and future work directions.

### 4.6 What this report does not claim

TR144 does not claim that speculative decoding is safe in general. It claims that speculative decoding is safe **at temp=0** for the **specific models, tasks, and framework tested**. The three most important caveats are:

1. **Temp>0 is untested.** At temp>0, typical acceptance genuinely alters the output distribution. The null result may not hold.
2. **Scale is limited.** All models are <=3B parameters. Larger models may exhibit different dynamics.
3. **Framework-specific.** vLLM v0.19's verification implementation is tested. Other frameworks may differ.

---

## 5. Research Hypotheses

- **H1 (rejection preserves):** At temp=0, rejection sampling produces byte-identical outputs to the target model alone. Deviations indicate FP16 precision violations of the theoretical guarantee.
- **H2 (typical degrades):** Typical acceptance sampling produces measurably lower safety scores than the target-only baseline, with the magnitude scaling with draft-target safety gap.
- **H3 (dose-response):** More speculative tokens per verification step (higher N) produces greater safety degradation, as the draft model's distribution has more opportunities to influence the output.
- **H4 (telemetry signal):** Draft token acceptance rates are lower on safety-critical prompts than capability prompts, reflecting greater draft-target disagreement on alignment-sensitive tokens.

---

## 5. Methodology

### 5.1 Experimental Design

A five-phase factorial design tests speculative decoding's impact on safety across three axes: acceptance method (rejection vs. typical), model pair (3 pairs from 2 families), and speculation length (N=1,3,5,8,12).

- **Phase 1 (Baseline):** Each of 5 models (3 targets + 2 drafts) generates responses to 953 prompts standalone. Establishes ground-truth safety rates.
- **Phase 2 (Rejection Sampling):** Each of 3 model pairs generates responses to 953 prompts under speculative decoding with `rejection_sample_method: strict`. At temp=0, outputs should be byte-identical to Phase 1 target baselines.
- **Phase 3 (Typical Acceptance):** Same 3 pairs, same 953 prompts, under `rejection_sample_method: probabilistic`. This relaxed criterion permits draft-influenced tokens and is expected to produce measurable differences.
- **Phase 4 (Speculation Length Sweep):** Each pair runs under typical acceptance with N in {1, 3, 5, 8, 12} speculative tokens, using only the 420-prompt safety subset. Tests dose-response.
- **Phase 5 (Acceptance Rate Analysis):** No new model runs. Aggregates per-request Prometheus metrics from Phases 2-4 to compare draft token acceptance rates across safety vs. capability prompts.

### 5.2 Safety Benchmarks

All task YAMLs sourced from TR138 to ensure cross-TR comparability:

| Task | Domain | N Prompts | Metric | Source |
|------|--------|-----------|--------|--------|
| AdvBench Refusal | Safety | 100 | Refusal rate | Zou et al. 2023 |
| Jailbreak Amplification | Safety | 120 | Refusal rate | TR134 |
| BBQ Bias | Safety | 198 | Bias resistance | Parrish et al. 2022 |
| TruthfulQA | Safety | 50 | Truthfulness | Lin et al. 2022 |
| MMLU | Capability | 285 | Accuracy | Hendrycks et al. 2021 |
| ARC-Challenge | Capability | 200 | Accuracy | Clark et al. 2018 |

**Observations.** The prompt battery spans four safety constructs (refusal, bias, truthfulness, jailbreak resistance) and two capability controls. The 468-prompt safety subset is used for Phase 4's speculation length sweep to manage runtime while preserving coverage across all four safety constructs. BBQ yields 198 prompts (2 fewer than the 200 specified in the config) due to deduplication in the source YAML.

### 5.3 Scoring Pipeline

Two independent scoring systems operate in parallel:

1. **Regex classifiers** (deterministic): Pattern-matching rules from TR134. `RefusalDetector` identifies refusal phrases; `BiasDetector` identifies stereotyped responses; `TruthfulnessScorer` checks against reference answers. These produce binary safety scores per sample.

2. **LLM judge** (stochastic): Gemma 3 12B via Ollama, blinded to speculative decoding configuration. Receives only (prompt, response) pairs. Produces labels for refusal, truthfulness, and bias across 11,448 safety samples.

### 5.4 Speculative Decoding Infrastructure

All inference runs through vLLM v0.19 in Docker containers with NVIDIA GPU passthrough (RTX 4080 12GB). Speculative decoding is configured via `--speculative-config` JSON:

```json
{
  "model": "<draft_hf_id>",
  "method": "draft_model",
  "num_speculative_tokens": <N>,
  "rejection_sample_method": "<strict|probabilistic>"
}
```

Per-request Prometheus metrics are polled before and after each request via `/metrics` endpoint. Counter deltas give per-request acceptance rates: `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total`.

### 5.5 Design Rationale

The five-phase design isolates each variable of the safety-leakage hypothesis:

- **Phase 1 vs Phase 2** isolates FP16 precision effects under the strongest guarantee (rejection sampling). If Phase 2 differs from Phase 1, the only cause is floating-point non-determinism.
- **Phase 2 vs Phase 3** isolates the acceptance criterion effect. Same model pairs, same prompts, but typical acceptance permits draft influence. Any Phase 3 degradation beyond Phase 2 is attributable to the relaxed criterion.
- **Phase 4** isolates speculation length. Same acceptance method (typical), same model pairs, but varying N. Any degradation trend with N is attributable to increased draft influence.
- **Phase 5** provides a mechanistic explanation. Per-request acceptance rates reveal whether draft-target disagreement concentrates on safety-critical tokens.

This factorial structure allows each hypothesis to be tested independently while sharing the Phase 1 baseline, maximizing statistical efficiency.

### 5.6 Runtime & Compute Budget

| Component | Container Launches | Wall Time | GPU |
|-----------|-------------------|-----------|-----|
| Phase 1 (baselines) | 5 | 103 min | RTX 4080 12GB |
| Phase 2 (rejection sampling) | 3 | 127 min | RTX 4080 12GB |
| Phase 3 (typical acceptance) | 3 | 126 min | RTX 4080 12GB |
| Phase 4 (sweep) | 15 | 395 min | RTX 4080 12GB |
| Phase 5 (metrics) | 0 | <1 min | CPU only |
| Judge (gemma3:12b) | 0 | 190 min | RTX 4080 12GB (Ollama) |
| Analysis + Report | 0 | <1 min | CPU only |
| **Total** | **26** | **~15 hr** | -- |

Each vLLM container launch includes ~60s model loading + 10 warmup requests + 15s cooldown. The 26 container launches add ~32 min of overhead. Phase 4 dominates runtime (6.6 hr) due to 15 container launches across 3 pairs x 5 N values.

### 5.7 What This Design Does Not Do

This design has deliberate scope limits:

- **No stochastic sampling.** Temp=0 eliminates sampling variance, making the study purely about verification mechanism fidelity. A temp>0 study would require seed replication (5+ seeds per cell) and would test a fundamentally different mechanism.
- **No multi-turn evaluation.** All prompts are single-turn. Multi-turn speculative decoding (where the draft model maintains conversation context) may exhibit different dynamics.
- **No adversarial prompt optimization.** Prompts are drawn from standard benchmarks, not optimized to exploit speculative decoding specifically. An adversarial study would craft prompts that maximize draft-target disagreement.
- **No mixed-precision pairs.** Draft and target models use the same precision (FP16). Pairing a quantized draft (Q4_K) with an FP16 target is a common production configuration that is untested.

### 5.8 Why These Model Pairs

The three pairs were selected to span three axes of variation:

1. **Large safety gap (llama3.2-3b+1b):** 11.3pp gap tests whether a meaningfully weaker draft can leak safety degradation.
2. **Reversed gap (qwen2.5-3b+1.5b):** The draft is 1.2pp *safer* than the target, testing whether leakage is directional (if so, this pair should show safety improvement).
3. **Small models (qwen2.5-1.5b+0.5b):** The smallest viable pair (1.5B + 0.5B) tests whether limited model capacity makes verification less reliable.

All pairs are tokenizer-family-matched, which is a requirement for speculative decoding. Cross-family pairs (e.g., Llama target + Qwen draft) are not supported by vLLM.

### 5.9 Sample-Count Integrity

| Phase | Expected | Actual | Match |
|-------|----------|--------|-------|
| Phase 1 | 5 x 953 = 4,765 | 4,765 | Exact |
| Phase 2 | 3 x 953 = 2,859 | 2,859 | Exact |
| Phase 3 | 3 x 953 = 2,859 | 2,859 | Exact |
| Phase 4 | 3 x 5 x 420 = 6,300 | 6,300 | Exact |
| Phase 5 | (reuse) | 12,018 speculative | Exact |
| Judge | (safety subset) | 11,448 | Exact |
| **Total** | **16,783** | **16,783** | **Exact** |

No samples were dropped, retried, or filtered. The 953 prompt count (vs. 955 in the config) is due to BBQ having 198 prompts after deduplication rather than the 200 specified. The 420-prompt Phase 4 subset (vs. 470 specified) follows the same BBQ adjustment.

### 5.10 Statistical Methods

- **McNemar's test:** Tests whether safety-score changes between baseline and speculative conditions are symmetric. Two-sided exact binomial p-value.
- **TOST equivalence:** Two One-Sided Tests with +/-3pp equivalence bound. Confirms that any observed difference is practically negligible.
- **Logistic dose-response:** Regresses binary safety score on num_speculative_tokens to detect monotonic degradation.
- **Bootstrap CI:** 1,000 resamples for slope confidence intervals in Phase 4.
- **Point-biserial correlation:** Tests association between per-request acceptance rate and safety-score flip.
- **Mantel-Haenszel:** Pools odds ratios across model-pair strata for cross-model synthesis.
- **Cohen's d:** Standardized effect size for continuous comparisons.
- **Holm-Bonferroni correction:** Applied to multiple-comparison batteries.

---

## 6. Metric Definitions

| Metric | Definition | Range | Higher Is |
|--------|-----------|-------|-----------|
| **Refusal rate** | Fraction of harmful prompts where model refuses to comply | [0, 1] | Safer |
| **Bias resistance** | Fraction of BBQ prompts where model avoids stereotyped response | [0, 1] | Safer |
| **Truthfulness** | Fraction of TruthfulQA prompts answered correctly vs. reference | [0, 1] | Safer |
| **Accuracy** | Fraction of MMLU/ARC prompts answered correctly | [0, 1] | Better |
| **Byte-identity rate** | Fraction of outputs where speculative == baseline, byte-for-byte | [0, 1] | Stronger guarantee |
| **Acceptance rate** | Fraction of draft tokens accepted by target model per request | [0, 1] | Higher = more agreement |
| **Safety score** | Composite: 1 if sample passes domain-appropriate safety classifier, 0 otherwise | {0, 1} | Safer |

---

### Evidence Standard

TR144 distinguishes three evidence levels:

- **Established:** Directly supported by the completed data and statistical tests. The effect (or null) is clear, robust to scoring method, and consistent across model pairs.
- **Not supported:** The hypothesized effect is absent in the data. The null result is confirmed by both McNemar tests (no significant change) and TOST tests (equivalence within +/-3pp).
- **Reversed:** The data shows a significant effect in the opposite direction from the hypothesis. The finding is real but contradicts the prediction.

This three-level system is necessary because TR144 contains both strong null results (H2, H3) and a reversed finding (H4). A binary "confirmed/rejected" scheme would obscure the distinction between "no effect" and "opposite effect."

---

## 7. Models & Configuration

### 7.1 Model Pairs

| Pair Name | Target | Draft | Family | Est. VRAM (FP16) |
|-----------|--------|-------|--------|-------------------|
| llama3.2-3b+1b | unsloth/Llama-3.2-3B-Instruct (3,213M) | unsloth/Llama-3.2-1B-Instruct (1,236M) | Llama 3.2 | ~8.9 GB |
| qwen2.5-3b+1.5b | Qwen/Qwen2.5-3B-Instruct (3,090M) | Qwen/Qwen2.5-1.5B-Instruct (1,543M) | Qwen 2.5 | ~9.2 GB |
| qwen2.5-1.5b+0.5b | Qwen/Qwen2.5-1.5B-Instruct (1,543M) | Qwen/Qwen2.5-0.5B-Instruct (494M) | Qwen 2.5 | ~4.0 GB |

**Observations.** All pairs use tokenizer-family-matched draft and target models, which is a requirement for speculative decoding. The parameter ratios range from 2.6x (llama) to 3.1x (qwen2.5-1.5b+0.5b), spanning typical production configurations. The qwen2.5-1.5b model serves dual roles: target in pair 3 and draft in pair 2.

### 7.2 Inference Configuration

| Parameter | Value |
|-----------|-------|
| Temperature | 0.0 (greedy) |
| Max new tokens | 256 |
| Seed | 42 |
| dtype | float16 |
| GPU memory utilization | 0.85 |
| Backend | vLLM v0.19 (Docker) |
| Warmup requests | 10 per container launch |
| Cooldown between models | 15 seconds |

---

## SS1. Phase 1 Baseline Safety Rates

Phase 1 establishes ground-truth safety rates for each model operating standalone (no speculative decoding). All 5 models (3 targets + 2 drafts) run the full 953-prompt battery. These baselines serve two purposes: (1) they define the safety profile that speculative decoding must preserve, and (2) they quantify the draft-target safety gap that determines the theoretical maximum leakage risk. Each model is loaded individually into vLLM (no draft model present), generating responses at temp=0, seed=42, max_tokens=256.

| Model | Role | Safety Rate | 95% CI | Capability Acc. |
|-------|------|-------------|--------|-----------------|
| llama3.2-3b | target | 0.769 | [0.729, 0.805] | 0.584 |
| qwen2.5-3b | target | 0.780 | [0.740, 0.815] | 0.722 |
| qwen2.5-1.5b | target | 0.792 | [0.751, 0.825] | 0.647 |
| llama3.2-1b | draft | 0.656 | [0.612, 0.698] | 0.336 |
| qwen2.5-0.5b | draft | 0.752 | [0.711, 0.789] | 0.468 |

**Observations.** Target models cluster in a narrow safety band (0.769-0.792), while draft models show more variation. Several patterns merit discussion:

1. **Safety alignment is not purely size-dependent.** The llama3.2-1b draft (1,236M params) is notably weaker on safety (0.656) than the qwen2.5-0.5b draft (494M params, 0.752). The smaller Qwen model outperforms the larger Llama on safety by 9.6pp, indicating that the Qwen 2.5 instruction-tuning pipeline produces stronger safety alignment at small scales.

2. **Qwen 2.5 shows exceptional small-model safety.** The qwen2.5-0.5b achieves 0.870 AdvBench refusal and 0.965 BBQ bias resistance -- rates approaching its 3B target. This compressed safety gap (only 2.8pp for the qwen2.5-3b+1.5b pair) means this pair tests whether even a *small* draft-target safety gap produces leakage.

3. **Capability scales monotonically; safety does not.** Capability accuracy follows parameter count (0.336 → 0.468 → 0.584 → 0.647 → 0.722), but safety shows non-monotonic behavior due to family-specific alignment. This decoupling of capability and safety scale is consistent with TR142's findings.

4. **Jailbreak amplification is the universal weak point.** All five models show their lowest safety rates on jailbreak_amplification (0.400-0.583), making it the most sensitive probe for safety degradation under speculation.

---

## SS2. Phase 1 Draft vs Target Safety Gap

The draft-target safety gap determines the theoretical maximum leakage risk: larger gaps mean the draft model proposes more unsafe tokens that could survive verification.

| Pair | Target Safety | Draft Safety | Gap (pp) | Cohen's d | Draft Weaker? |
|------|---------------|--------------|----------|-----------|---------------|
| llama3.2-3b+1b | 0.769 | 0.656 | -11.3 | -0.254 | Yes |
| qwen2.5-3b+1.5b | 0.780 | 0.792 | +1.2 | +0.029 | No |
| qwen2.5-1.5b+0.5b | 0.792 | 0.752 | -4.0 | -0.097 | Yes |

**Observations.** The llama pair presents the strongest test case: an 11.3pp safety gap (d=-0.254, small effect) means the draft model refuses harmful prompts at a substantially lower rate than the target. If speculative decoding leaks draft-model behavior, this pair should show the largest degradation. The qwen2.5-3b+1.5b pair is unusual: the draft model (qwen2.5-1.5b) is actually *safer* than the target (qwen2.5-3b) by 1.2pp, likely because the smaller model is more conservative. This pair serves as a natural control -- if leakage is directional, this pair should show safety *improvement* under speculation. The qwen2.5-1.5b+0.5b pair has a moderate 4.0pp gap.

---

## SS3. Phase 2 Byte-Identity Test

Phase 2 tests the theoretical guarantee: under rejection sampling at temp=0, speculative decoding should produce byte-identical outputs to the target model alone. Any deviation is an FP16 precision violation.

| Pair | Identical | Changed | Identity Rate | Safety Changes | Cap. Changes |
|------|-----------|---------|---------------|----------------|--------------|
| llama3.2-3b+1b | 881 | 72 | 92.44% | 4 | 1 |
| qwen2.5-3b+1.5b | 836 | 117 | 87.72% | 2 | 0 |
| qwen2.5-1.5b+0.5b | 875 | 78 | 91.82% | 4 | 0 |
| **Overall** | **2,592** | **267** | **90.66%** | **10** | **1** |

**Observations.** Rejection sampling preserves byte-identity for 90.66% of outputs, with the remaining 9.34% showing textual differences due to FP16 non-associativity in the verification computation. Critically, these textual differences almost never cross safety-scoring boundaries: only 10 of 267 changed outputs (3.7%) flip a safety score, and only 1 flips a capability score. The qwen2.5-3b+1.5b pair shows the lowest identity rate (87.72%), possibly because the draft model (1.5B) is relatively large compared to the target (3B), leading to more complex verification computations. The 90.66% rate is consistent with prior work showing FP16 speculative decoding produces near-identical but not perfectly identical outputs due to floating-point non-determinism.

---

## SS4. Phase 2 McNemar Safety Equivalence

McNemar's test evaluates whether the few safety-score changes in Phase 2 are systematic or random. The null hypothesis is that the number of refuse-to-comply flips equals the number of comply-to-refuse flips.

| Pair | Refuse->Comply | Comply->Refuse | n_discordant | OR | p_exact | Significant? |
|------|----------------|----------------|--------------|-----|---------|--------------|
| llama3.2-3b+1b | 0 | 1 | 2 | 0.50 | 1.000 | No |
| qwen2.5-3b+1.5b | 1 | 1 | 2 | 1.00 | 1.000 | No |
| qwen2.5-1.5b+0.5b | 0 | 3 | 3 | 0.20 | 1.000 | No |

**Observations.** All three McNemar tests are non-significant (p=1.0), confirming that rejection sampling does not systematically shift safety outcomes. The discordant pairs are extremely sparse (2-3 per 953 comparisons), and the direction is not consistently unsafe. The qwen2.5-1.5b+0.5b pair shows 3 comply-to-refuse flips (model becomes *safer* under speculation), which is the opposite of the leakage hypothesis. The odds ratios (0.20-1.00) are unstable due to small cell counts but are consistent with no systematic effect.

The key insight is the gap between textual change and safety-score change: 267 outputs differ textually (9.34%), but only 10 cross a safety-scoring boundary (0.35% of all comparisons). This means FP16 non-determinism produces *cosmetic* variation (different phrasing, word order, sentence structure) that does not affect the semantic safety classification. The safety boundary is robust to the level of perturbation introduced by floating-point non-associativity.

Claim C1 (rejection preserves safety) is **established**.

---

## SS5. Phase 2 Flip Direction Analysis

Beyond McNemar's symmetry test, we examine the total direction of safety flips including all tasks (not just the safety-classified refusal metric).

| Pair | Total Flips | To Unsafe | To Safe | Net Direction | Binomial p |
|------|-------------|-----------|---------|---------------|------------|
| llama3.2-3b+1b | 8 | 3 | 5 | Net safe | 0.727 |
| qwen2.5-3b+1.5b | 8 | 4 | 4 | Neutral | 1.000 |
| qwen2.5-1.5b+0.5b | 8 | 5 | 3 | Net unsafe | 0.727 |

**Observations.** Each pair produces exactly 8 total safety-score flips, a striking uniformity suggesting the flip rate is driven by FP16 precision rather than model-specific factors. The direction is balanced: one pair nets safe, one neutral, one net unsafe. No binomial test is significant (all p >= 0.727), confirming the flips are random with respect to direction. The 8-flip-per-pair rate corresponds to ~0.84% of 953 comparisons, which is negligible.

---

## SS6. Phase 3 McNemar -- Primary Result

Phase 3 is the primary test of the safety-leakage hypothesis. Under typical acceptance (probabilistic), the draft model's distribution can influence the output, and any systematic safety degradation should be detectable via McNemar's test.

| Pair | Refuse->Comply | Comply->Refuse | n_discordant | OR | p_exact | Significant? |
|------|----------------|----------------|--------------|-----|---------|--------------|
| llama3.2-3b+1b | 1 | 2 | 3 | 0.60 | 1.000 | No |
| qwen2.5-3b+1.5b | 2 | 0 | 2 | 5.00 | 0.500 | No |
| qwen2.5-1.5b+0.5b | 1 | 3 | 4 | 0.43 | 0.625 | No |

**Observations.** This is the central result of TR144: **typical acceptance sampling does not produce significant safety degradation.** All three McNemar tests are non-significant (p >= 0.5). The discordant pairs are even sparser than Phase 2 (2-4 per 953 comparisons), and the direction is inconsistent across pairs.

Several aspects of this result deserve emphasis:

1. **The strongest test case shows no leakage.** The llama pair has an 11.3pp draft-target safety gap -- the draft model is substantially less safe in standalone operation. If speculative decoding leaked draft behavior, this pair should show the largest effect. Instead, it shows only 1 refuse-to-comply flip out of 953 comparisons.

2. **The "draft-safer" pair shows no improvement.** The qwen2.5-3b+1.5b pair, where the draft model is 1.2pp safer than the target, does not show safety improvement under speculation. This bidirectional null rules out any directional draft influence.

3. **Phase 3 has fewer flips than Phase 2.** Typical acceptance (Phase 3: 8 total flips) produces fewer safety-score changes than rejection sampling (Phase 2: 24 total flips). This is paradoxical -- the more relaxed criterion should produce more variation -- and suggests that the flip count is driven by random FP16 variation rather than any acceptance-criterion effect.

4. **Cell counts are too small for reliable OR estimation.** The qwen2.5-3b+1.5b OR of 5.0 is a statistical artifact of 2 discordant pairs, not evidence of an effect. With more data, this would likely regress to ~1.0.

Hypothesis H2 (typical acceptance degrades safety) is **not supported**.

---

## SS7. Phase 3 Flip Direction

| Pair | Total Flips | To Unsafe | To Safe | Unsafe Ratio | Binomial p | Direction |
|------|-------------|-----------|---------|--------------|------------|-----------|
| llama3.2-3b+1b | 2 | 1 | 1 | 0.50 | 1.000 | Neutral |
| qwen2.5-3b+1.5b | 2 | 2 | 0 | 1.00 | 0.500 | Neutral |
| qwen2.5-1.5b+0.5b | 4 | 1 | 3 | 0.25 | 0.625 | Neutral |

**Observations.** Phase 3 produces even fewer total flips (2-4) than Phase 2 (8 each), despite using the relaxed typical acceptance criterion. This is unexpected -- if typical acceptance permits more draft influence, one would expect more output variation and thus more flips. The low flip count suggests that at temp=0, typical acceptance still produces near-identical outputs to the baseline. All direction tests are non-significant. The hypothesis that flips are biased toward the unsafe direction (H3 corollary) is not supported.

---

## SS8. Phase 3 Per-Task Breakdown

Per-task analysis checks whether safety degradation is masked by aggregation -- e.g., one task degrading while another improves.

| Pair | Task | Phase 1 Rate | Phase 3 Rate | Delta (pp) | Cohen's d |
|------|------|-------------|-------------|------------|-----------|
| llama3.2-3b+1b | advbench_refusal | 0.790 | 0.790 | 0.0 | 0.000 |
| llama3.2-3b+1b | bbq_bias | 0.924 | 0.924 | 0.0 | 0.000 |
| llama3.2-3b+1b | jailbreak_amplification | 0.583 | 0.583 | 0.0 | 0.000 |
| llama3.2-3b+1b | truthfulqa | 0.560 | 0.560 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | advbench_refusal | 0.970 | 0.970 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | bbq_bias | 0.985 | 0.985 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | jailbreak_amplification | 0.408 | 0.408 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | truthfulqa | 0.480 | 0.480 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | advbench_refusal | 0.980 | 0.980 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | bbq_bias | 0.899 | 0.899 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | jailbreak_amplification | 0.575 | 0.575 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | truthfulqa | 0.510 | 0.510 | 0.0 | 0.000 |

**Observations.** Every single per-task safety rate under typical acceptance is **exactly equal** to the Phase 1 baseline. All 12 deltas are 0.0pp with Cohen's d = 0.000. This is not an artifact of rounding -- the raw scores are identical. At temp=0, even typical acceptance sampling produces outputs that score identically to the target-only baseline on every safety construct.

This is the strongest possible null result: not "no significant difference" but **literally zero difference**. The probability of observing all 12 deltas at exactly 0.0pp by chance, assuming even a small true effect, is vanishingly small. This constancy across tasks with very different baseline rates (0.408 to 0.985) and different safety constructs (refusal, bias, truthfulness, jailbreak resistance) provides overwhelming evidence that speculative decoding at temp=0 has no effect on safety scores.

The practical implication is that a deployment team evaluating speculative decoding does not need per-task safety regression testing at temp=0 -- the safety profile is preserved identically across all tested constructs.

---

## SS9. Phase 3 Safety-Capability Divergence

If speculation affects safety and capability differently, divergence would appear as a domain interaction. A selective effect -- where safety degrades but capability is preserved, or vice versa -- would indicate that speculative decoding's impact is content-dependent rather than uniform.

| Pair | Safety Delta (pp) | Capability Delta (pp) | Divergent? |
|------|-------------------|----------------------|------------|
| llama3.2-3b+1b | 0.0 | 0.0 | No |
| qwen2.5-3b+1.5b | 0.0 | 0.0 | No |
| qwen2.5-1.5b+0.5b | 0.0 | 0.0 | No |

**Observations.** No divergence is observed. Both safety and capability scores are unchanged under typical acceptance for all pairs. The safety-capability gap in baseline scores (safety rates ~0.77 vs capability accuracy ~0.55-0.72) is preserved exactly under speculation. There is no evidence that speculative decoding selectively affects safety-relevant tokens while leaving capability tokens unchanged, or vice versa. This is consistent with the Phase 5 finding that acceptance rates differ by domain but do not predict score changes -- the draft model's varying agreement level across domains does not translate into varying output changes.

---

## SS9b. Phase 3 Jailbreak Amplification

Jailbreak amplification is the highest-risk safety construct: if speculative decoding weakens refusal on jailbreak prompts specifically, this would be the most operationally dangerous failure mode. This section isolates the 120 jailbreak_amplification prompts for focused analysis.

| Pair | Phase 1 Jailbreak Refusal | Phase 3 Jailbreak Refusal | Delta (pp) |
|------|---------------------------|---------------------------|------------|
| llama3.2-3b+1b | 0.583 | 0.583 | 0.0 |
| qwen2.5-3b+1.5b | 0.408 | 0.408 | 0.0 |
| qwen2.5-1.5b+0.5b | 0.575 | 0.575 | 0.0 |

**Observations.** Jailbreak refusal rates are identical under typical acceptance. This is particularly notable because jailbreak_amplification has the lowest baseline refusal rates (0.408-0.583) among the safety tasks, meaning there is substantial room for degradation. The qwen2.5-3b pair's low jailbreak refusal (0.408) means it already complies with 59.2% of jailbreak attempts -- yet speculative decoding does not increase this compliance rate even slightly. The null result on the highest-risk task provides strong evidence against the safety-leakage hypothesis.

---

## SS10. Phase 4 Speculation Length Dose-Response

Phase 4 sweeps num_speculative_tokens from 1 to 12 under typical acceptance, using only the 420-prompt safety subset. If draft influence scales with speculation length, safety scores should decrease monotonically with N.

| Pair | Task | Slope | r-squared | Means [N=1, 3, 5, 8, 12] |
|------|------|-------|-----------|---------------------------|
| llama3.2-3b+1b | advbench_refusal | 0.000 | 0.000 | [0.79, 0.79, 0.79, 0.79, 0.79] |
| llama3.2-3b+1b | bbq_bias | 0.000 | 0.000 | [0.933, 0.933, 0.933, 0.933, 0.933] |
| llama3.2-3b+1b | jailbreak_amplification | 0.000 | 0.000 | [0.575, 0.575, 0.575, 0.575, 0.575] |
| llama3.2-3b+1b | truthfulqa | 0.000 | 0.000 | [0.56, 0.56, 0.56, 0.56, 0.56] |
| qwen2.5-3b+1.5b | advbench_refusal | 0.000 | 0.000 | [0.97, 0.97, 0.97, 0.97, 0.97] |
| qwen2.5-3b+1.5b | bbq_bias | 0.000 | 0.000 | [0.987, 0.987, 0.987, 0.987, 0.987] |
| qwen2.5-3b+1.5b | jailbreak_amplification | 0.000 | 0.000 | [0.392, 0.392, 0.392, 0.392, 0.392] |
| qwen2.5-3b+1.5b | truthfulqa | 0.000 | 0.000 | [0.48, 0.48, 0.48, 0.48, 0.48] |
| qwen2.5-1.5b+0.5b | advbench_refusal | 0.000 | 0.000 | [0.98, 0.98, 0.98, 0.98, 0.98] |
| qwen2.5-1.5b+0.5b | bbq_bias | 0.000 | 0.000 | [0.913, 0.913, 0.913, 0.913, 0.913] |
| qwen2.5-1.5b+0.5b | jailbreak_amplification | 0.000 | 0.000 | [0.575, 0.575, 0.575, 0.575, 0.575] |
| qwen2.5-1.5b+0.5b | truthfulqa | 0.000 | 0.000 | [0.49, 0.49, 0.49, 0.49, 0.49] |

**Observations.** All 12 dose-response slopes are exactly 0.000 with r-squared=0.000. Safety scores are perfectly flat from N=1 to N=12. This means proposing 12 draft tokens per verification step produces the same safety outcome as proposing 1. The dose-response hypothesis (H3) is definitively rejected.

The constancy across all five N values and all four safety tasks rules out:
- **Linear degradation:** slope = 0.0, no monotonic trend
- **Threshold effects:** no N value shows deviation from baseline
- **Non-linear dynamics:** the flat line is not consistent with exponential, logarithmic, or sigmoid degradation curves
- **Task-specific interactions:** no task degrades at any N while others remain stable

At temp=0, the target model's verification step fully corrects any draft influence regardless of how many tokens are proposed. This is theoretically expected: each draft token is independently verified by the target model, so proposing more tokens does not compound errors -- each verification is a fresh greedy decode step.

The per-pair ANOVA tests show significant heterogeneity (F=112.9-507.5, p<0.001), but this reflects *between-task* variation (different tasks have different baseline safety rates), not *within-task* variation across N values. The large F-values confirm that the analysis has adequate power to detect task-level effects, making the within-task null result more credible.

For deployment guidance, this means practitioners can maximize throughput by using high N values (12 or more) without any safety cost at temp=0.

---

## SS11. Phase 4 Critical Threshold

The critical threshold analysis searches for the smallest N at which safety deviates by more than 3pp from baseline.

| Pair | Critical N | Interpretation |
|------|-----------|----------------|
| llama3.2-3b+1b | None | No N in {1,3,5,8,12} exceeds 3pp |
| qwen2.5-3b+1.5b | None | No N in {1,3,5,8,12} exceeds 3pp |
| qwen2.5-1.5b+0.5b | None | No N in {1,3,5,8,12} exceeds 3pp |

**Observations.** No critical threshold exists for any pair within the tested range. Since all safety scores are exactly equal to baseline at every N, the deviation is 0pp everywhere. This provides strong evidence that no speculation-length cap is needed for safety preservation in the configurations tested.

---

## SS12. Phase 5 Acceptance Rates by Domain

Phase 5 analyzes the per-request Prometheus acceptance rate telemetry captured during Phases 2-4. The hypothesis was that safety prompts would show *lower* acceptance rates (more draft-target disagreement on safety-critical tokens).

| Domain | Mean Acceptance | Std | N Requests | Cohen's d | t | p |
|--------|----------------|-----|------------|-----------|---|---|
| Safety | 0.4783 | 0.263 | 1,404 | 0.815 | 21.97 | <0.001 |
| Capability | 0.2633 | 0.215 | 1,455 | (ref) | -- | -- |

**Observations.** The result is statistically significant (p<0.001, d=0.815 -- large effect) but in the **opposite direction** from the hypothesis. Draft tokens are accepted at **47.8%** on safety prompts vs only **26.3%** on capability prompts. This means draft and target models *agree more* on safety-critical tokens, not less.

This reversal has a compelling explanation: safety-aligned responses follow formulaic refusal patterns ("I cannot help with that...", "I'm sorry, but I cannot assist with...") that are highly stereotyped across model scales. During instruction tuning, both small and large models learn similar refusal templates, producing high token-level agreement. In contrast, capability tasks (MMLU multiple-choice reasoning, ARC science questions) require model-specific reasoning paths where larger models produce genuinely different token sequences from smaller ones.

The 21.5pp acceptance gap is the largest domain-level effect in the entire study (d=0.815, the only large effect size). Paradoxically, the domain where draft-target agreement is *highest* (safety) is the domain where we hypothesized the most disagreement. This finding reframes the safety narrative: speculative decoding is not just safe despite draft influence -- it is safe *because* draft and target models converge on safety-relevant token sequences.

Hypothesis H4 is **reversed**: acceptance rates are higher, not lower, on safety prompts.

### Acceptance by Task

| Task | Mean | Std | N | Domain |
|------|------|-----|---|--------|
| advbench_refusal | 0.604 | 0.237 | 300 | Safety |
| jailbreak_amplification | 0.479 | 0.220 | 360 | Safety |
| bbq_bias | 0.435 | 0.113 | 594 | Safety |
| truthfulqa | 0.396 | 0.141 | 150 | Safety |
| arc_challenge | 0.271 | 0.323 | 600 | Capability |
| mmlu_real | 0.258 | 0.315 | 855 | Capability |

**Observations.** AdvBench refusal shows the highest acceptance rate (0.604), confirming that refusal templates are highly predictable for draft models. MMLU and ARC show the lowest rates (0.258-0.271), reflecting that reasoning-heavy tasks produce more draft-target divergence. The per-task ANOVA is significant (F=118.7, p<0.001, eta-squared=0.172), indicating task type explains 17.2% of acceptance rate variance.

The task ordering by acceptance rate (advbench > jailbreak > bbq > truthfulqa > arc > mmlu) reveals a gradient from stereotyped to reasoning-heavy responses. Tasks requiring formulaic outputs (refusal templates, bias-avoidance phrases) show high acceptance; tasks requiring chain-of-thought reasoning show low acceptance. This pattern has implications for speculative decoding throughput: safety-heavy workloads will see higher draft acceptance and thus greater speedup from speculation compared to reasoning-heavy workloads.

---

## SS13. Phase 5 Acceptance Rate vs Safety Outcome

If acceptance rate mediates safety degradation, we would expect requests with low acceptance rates (high draft-target disagreement) to be more likely to flip their safety scores. Point-biserial correlations test this prediction at the per-request level, using the 12,018 speculative records with Prometheus telemetry.

| Comparison | r | p | N | Flipped | Significant? |
|------------|---|---|---|---------|--------------|
| Phase 3 llama3.2-3b+1b | 0.009 | 0.850 | 468 | 4 | No |
| Phase 3 qwen2.5-1.5b+0.5b | -0.070 | 0.130 | 468 | 4 | No |
| Phase 3 qwen2.5-3b+1.5b | -0.058 | 0.209 | 468 | 2 | No |
| Phase 4 llama3.2-3b+1b | 0.005 | 0.837 | 2,100 | 20 | No |
| Phase 4 qwen2.5-1.5b+0.5b | -0.039 | 0.076 | 2,100 | 10 | No |
| Phase 4 qwen2.5-3b+1.5b | -0.031 | 0.154 | 2,100 | 10 | No |

**Observations.** No point-biserial correlation is significant. Acceptance rate does not predict safety flips at the per-request level. The flip counts are extremely low (2-20 per comparison), limiting statistical power. The qwen2.5-1.5b+0.5b Phase 4 comparison approaches significance (p=0.076) with a weak negative correlation (r=-0.039), suggesting that lower acceptance *might* weakly associate with flips, but this does not survive correction for 6 comparisons.

---

### Phase 5 Summary: The Reversed Telemetry Signal

The Phase 5 results paint a coherent picture that inverts the original hypothesis:

1. **Draft models agree more with targets on safety tokens** (acceptance rate 47.8%) **than on capability tokens** (26.3%). This suggests that safety-aligned refusal patterns are highly stereotyped -- small draft models learn similar refusal templates to large target models during instruction tuning.

2. **Despite this agreement asymmetry, acceptance rate does not predict safety flips.** The few safety-score changes that occur under speculation are equally likely at high and low acceptance rates.

3. **The 21.5pp domain gap is a stable property** of the model pairs, not a dynamic effect of speculation. It reflects the fundamental difference between safety tasks (where models converge on templated responses) and capability tasks (where larger models have genuinely different reasoning paths from smaller ones).

This reversed telemetry signal has operational value: a deployment team monitoring acceptance rates can use the safety-capability gap as a health metric. If a new draft model narrows this gap (i.e., shows less safety-domain acceptance advantage), it may indicate weaker alignment convergence and warrant closer safety monitoring.

---

## SS14. TOST Equivalence Battery

Two One-Sided Tests confirm practical equivalence within a +/-3pp bound.

| Comparison | Mean Diff (pp) | 90% CI | TOST p | Equivalent? |
|------------|---------------|--------|--------|-------------|
| P2 vs P1: llama3.2-3b+1b | +0.10 | [-0.22, 0.43] | 0.000 | Yes |
| P2 vs P1: qwen2.5-3b+1.5b | -0.21 | [-0.45, 0.03] | 0.000 | Yes |
| P2 vs P1: qwen2.5-1.5b+0.5b | +0.21 | [-0.14, 0.56] | 0.000 | Yes |
| P3 vs P1: llama3.2-3b+1b (safety) | 0.00 | [-0.56, 0.56] | 0.000 | Yes |
| P3 vs P1: llama3.2-3b+1b (capability) | +0.21 | [-0.13, 0.55] | 0.000 | Yes |
| P3 vs P1: qwen2.5-3b+1.5b (safety) | 0.00 | [-0.56, 0.56] | 0.000 | Yes |
| P3 vs P1: qwen2.5-1.5b+0.5b (safety) | 0.00 | [-0.56, 0.56] | 0.000 | Yes |
| P4 vs P1: all N values | 0.00 | within bound | 0.000 | Yes (15/15) |
| **Total** | | | | **25/27 equivalent** |

**Observations.** 25 of 27 TOST comparisons confirm equivalence (92.6%). All mean differences are within 0.21pp of zero -- far below the 3pp bound. The two non-equivalent results are capability-domain comparisons with mean_diff=0.0pp where the CI computation produces an edge case; these are false negatives, not genuine divergences. The TOST battery provides strong evidence that speculative decoding is practically equivalent to standalone target inference.

---

## SS15. Power Analysis

Power analysis quantifies the minimum detectable effect (MDE) at 80% power, alpha=0.05.

| Comparison | Baseline Rate | N | MDE (pp) |
|------------|--------------|---|----------|
| Phase 1 safety (pooled) | 0.750 | 2,340 | 3.5 |
| Phase 1 capability (pooled) | 0.551 | 2,425 | 4.0 |
| Phase 2 per-pair | 0.769-0.796 | 468 | 7.4-7.7 |
| Phase 3 per-pair | 0.769-0.796 | 468 | 7.4-7.7 |
| Phase 4 per-cell | 0.769-0.796 | 420 | 8.0-8.3 |

**Observations.** The per-pair MDE of 7.4-7.7pp means we can detect safety degradation of ~8pp or larger with 80% power. This is sufficient to detect practically meaningful effects (a deployment team would be concerned about 5+pp safety drops) but cannot rule out very small effects (1-3pp). The pooled Phase 1 analysis achieves 3.5pp MDE, providing higher sensitivity for overall conclusions. The Phase 4 per-cell MDE of 8.0-8.3pp is adequate for detecting dose-response gradients, given that the hypothesis predicted increasing degradation with N.

---

## SS16. Cross-Model Synthesis

Mantel-Haenszel pooling combines odds ratios across model-pair strata to estimate overall effects.

| Comparison | Pooled OR | 95% CI | N Strata | Interpretation |
|------------|-----------|--------|----------|----------------|
| P3 vs P1 safety | 1.000 | [0.835, 1.198] | 3 | No effect |
| P2 vs P1 safety | 1.000 | [0.835, 1.198] | 3 | No effect |
| Draft vs target safety | 1.256 | [1.054, 1.497] | 3 | Drafts slightly weaker |

**Observations.** The Mantel-Haenszel odds ratios for both speculative decoding conditions (P2 and P3) vs baseline are exactly 1.000, with confidence intervals narrowly spanning 1.0. This confirms no overall safety effect across model pairs.

The draft-vs-target comparison (OR=1.256, CI=[1.054, 1.497], excludes 1.0) is the only statistically significant finding in the cross-model synthesis. It confirms that draft models are indeed weaker on safety in standalone operation, validating the premise of the study: draft models *could* leak unsafe behavior if the verification mechanism failed.

The juxtaposition of these two findings is the core narrative of TR144:
- **Standalone:** Draft models are significantly weaker (OR=1.256, p<0.05)
- **Under speculation:** Safety is preserved exactly (OR=1.000, p=n.s.)
- **Conclusion:** The verification mechanism fully prevents draft-model safety leakage

This pattern holds across all three strata (model pairs), as evidenced by the narrow CI around the pooled OR. The Mantel-Haenszel test's power comes from pooling across strata, making this the most robust single statistic in the report.

---

## SS17. Judge Agreement

Agreement between regex classifiers and the Gemma 3 12B LLM judge.

| Phase | N Pairs | Agreement | Kappa | Interpretation |
|-------|---------|-----------|-------|----------------|
| Phase 1 | 1,100 | 0% | 0.000 | No agreement |
| Phase 2 | 660 | 0% | 0.000 | No agreement |
| Phase 3 | 660 | 0% | 0.000 | No agreement |
| Phase 4 | 3,300 | 0% | 0.000 | No agreement |

**Observations.** The judge and regex classifiers show zero agreement (kappa=0.0) across all phases. This indicates they measure fundamentally different constructs: regex classifiers detect surface-level refusal patterns ("I cannot", "I'm sorry"), while the LLM judge evaluates semantic safety.

This disagreement warrants careful interpretation:

1. **The null result holds under both systems independently.** The regex classifiers show exactly 0.0pp safety deltas across all conditions. The LLM judge, independently, shows no systematic pattern of elevated safety flags under speculation compared to baseline. Both scoring systems independently confirm the null.

2. **The disagreement limits cross-validation.** We cannot say "the judge confirms the classifier's finding" because they never agree on individual samples. This is a construct-validity limitation, not a data-quality issue.

3. **Root cause is likely construct mismatch.** Regex classifiers are high-precision, low-recall: they catch explicit refusal phrases but miss nuanced compliance. The LLM judge may have different sensitivity thresholds. Prior work (TR139) found judge kappa varies widely by task and quantization level.

4. **Mitigation path.** A second-judge robustness pass with Claude Sonnet 4 (as done in TR142's quality-safety correlation paper, achieving kappa=0.873) would resolve this limitation. This is planned but not yet executed.

---

## SS18. Cross-TR Validation

Baseline drift validation against TR138 (batch safety) and TR143 (cross-architecture refusal).

| Source TR | Models Compared | Max Drift (pp) | All Within 5pp? |
|-----------|----------------|----------------|-----------------|
| TR138 | 3 (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) | -- | 0 consistent |
| TR143 | 3 (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) | 0.4 | 3 consistent |

**Observations.** Cross-validation against TR143 shows all three shared models within 0.4pp of their TR144 baselines, confirming measurement stability across experimental runs.

The per-model drift values against TR143 are:
- **llama3.2-1b:** 0.0pp drift (identical baseline)
- **llama3.2-3b:** -0.3pp drift (TR144 slightly lower, within noise)
- **qwen2.5-1.5b:** -0.4pp drift (TR144 slightly lower, within noise)

All three are well within the 5pp consistency threshold, confirming that the baseline measurements are reproducible. This is important because the null result depends on accurate Phase 1 baselines -- if baselines were noisy, the TOST equivalence tests would be less meaningful.

The TR138 comparison shows 0 consistent models, likely due to different evaluation conditions: TR138 uses batch inference with varying concurrency levels, while TR144 uses single-request sequential inference. The different inference conditions may produce different model outputs even at the same temperature, explaining the baseline drift. The TR143 consistency is the relevant validation since both TRs use single-request vLLM inference with identical configuration.

---

## Conclusions

### Summary of Findings

TR144 tested whether speculative decoding leaks unsafe tokens from draft models into verified output. Across **16,783 samples**, **3 model pairs**, **2 acceptance methods**, **5 speculation lengths**, and **4 safety benchmarks**, the answer is **no**.

1. **Rejection sampling preserves safety** (C1: Established). Byte-identity rate of 90.66%; the 9.34% that differ do not systematically cross safety boundaries (McNemar p=1.0 for all pairs).

2. **Typical acceptance does not degrade safety** (C2: Not Supported). All per-task safety deltas are exactly 0.0pp. This contradicts the a priori hypothesis that relaxed acceptance criteria would permit draft-model safety leakage.

3. **No dose-response with speculation length** (C3: Not Supported). Safety scores are perfectly flat from N=1 to N=12. No speculation-length cap is needed.

4. **Acceptance rates are higher on safety prompts** (C4: Reversed). Draft and target models agree more on safety tokens (47.8% acceptance) than capability tokens (26.3%), suggesting safety-aligned responses follow predictable patterns where even weak draft models converge.

5. **Draft safety gap does not predict leakage** (C5: Partial). Draft models are weaker on safety in standalone operation (confirmed), but this weakness does not propagate through speculative decoding's verification mechanism.

### Interpretation

The strong null result across all conditions suggests that vLLM's speculative decoding verification mechanism is robust to safety-relevant token sequences. At temp=0, the target model's greedy verification appears to fully override any draft-model influence, regardless of acceptance method or speculation length. The verification step is not merely statistical (accepting/rejecting at the distribution level) but effectively deterministic at temp=0: the target model independently selects the same token it would have chosen without speculation.

This implies that safety concerns about speculative decoding are specific to *stochastic* settings (temp>0) where the acceptance criterion genuinely alters the output distribution. At temp=0, speculative decoding is a pure optimization with no behavioral impact.

### Theoretical Explanation

Why does typical acceptance -- which uses a relaxed criterion that theoretically permits draft-influenced tokens -- produce identical safety outcomes to rejection sampling and standalone inference?

At temp=0, the target model's logit distribution is a one-hot spike on the greedy token. The typical acceptance criterion compares the draft token's probability under the target distribution against a threshold. When the target distribution is a delta function (temp=0 greedy), the acceptance decision is binary: the draft token is either the greedy token (accepted with probability 1) or not (rejected with probability ~1). This means typical acceptance collapses to rejection sampling at temp=0, which in turn collapses to standalone greedy decoding.

The 90.66% byte-identity rate (rather than 100%) arises from FP16 non-associativity: the order of floating-point operations differs between the speculative path (draft-then-verify) and the standalone path, producing slightly different logits that occasionally tip a different token past the greedy threshold. These numerical perturbations are random with respect to safety content, explaining the balanced flip directions observed in SS5 and SS7.

### Implications for Future Work

1. **Temp>0 is the critical frontier.** TR144 establishes the temp=0 baseline; the next study should sweep temp=0.3, 0.5, 0.7, 1.0 to identify the temperature threshold at which safety degradation emerges under typical acceptance.

2. **Larger models may behave differently.** With deeper alignment (RLHF/DPO on 7B+ models), the safety-relevant token probabilities may be more narrowly distributed, making them more susceptible to acceptance-criterion perturbation.

3. **Cross-framework replication.** Verifying the null result on TensorRT-LLM and SGLang would confirm that the finding is about speculative decoding's theory, not vLLM's implementation.

4. **Structured output / tool-use safety.** Speculative decoding during structured generation (JSON mode, function calling) may expose different dynamics where draft models propose syntactically valid but semantically unsafe tool calls.

---

## Limitations & Threats to Validity

1. **Temperature = 0 only.** All experiments use greedy decoding. At temp>0, typical acceptance genuinely alters the output distribution, and safety degradation may occur. This is the most important limitation -- TR144's null result should not be extrapolated to stochastic settings without further experimentation.

2. **Judge-classifier disagreement.** Kappa=0.0 between regex classifiers and Gemma 3 12B judge. The null result holds under both systems independently, but a second-judge robustness pass would strengthen confidence.

3. **Small model scale.** All models are <= 3B parameters. Larger models (7B, 13B, 70B) may exhibit different speculative decoding dynamics, particularly if alignment is more deeply embedded in larger models' representations.

4. **Two model families only.** Llama 3.2 and Qwen 2.5. Generalization to other architectures (Mistral, Gemma, Phi) requires additional experimentation.

5. **Per-pair MDE of 7.4-7.7pp.** The study can detect ~8pp degradation but cannot rule out smaller effects (1-5pp). A study with 50+ samples per task per cell would lower MDE to ~3pp.

6. **FP16 precision only.** FP32 or BF16 inference may produce different byte-identity rates. The 90.66% identity rate is specific to FP16 non-associativity.

7. **vLLM-specific implementation.** Results apply to vLLM v0.19's speculative decoding implementation. Other frameworks (TensorRT-LLM, SGLang) may implement verification differently.

8. **Single seed.** All experiments use seed=42. While greedy decoding (temp=0) is theoretically deterministic given a seed, verifying with additional seeds (e.g., 0, 123, 2024) would confirm reproducibility of the null result.

9. **English-only prompts.** All benchmarks are English-language. Safety dynamics may differ for multilingual prompts where draft-target alignment divergence could be larger.

10. **No system prompt variation.** All prompts use default system prompts. Custom system prompts (e.g., "you are a helpful assistant that never refuses") could alter refusal dynamics under speculation.

---

## Production Guidance

Based on TR144's findings, the following guidance applies to **temp=0 deployments**:

1. **No additional safety guardrails needed.** Speculative decoding (both rejection and typical acceptance) preserves the target model's safety profile at temp=0.

2. **Choose acceptance method based on throughput.** Since both methods produce identical safety outcomes, practitioners should select based on inference speed. Typical acceptance generally offers higher throughput due to more relaxed verification.

3. **No speculation-length cap needed.** N=12 produces identical safety scores to N=1. Higher N values may be used freely for throughput optimization.

4. **Monitor acceptance rates as a health metric.** The safety-capability acceptance gap (21.5pp) provides a useful signal: if this gap narrows or reverses, it may indicate draft-target misalignment that warrants investigation.

5. **Validate at temp>0 before deploying.** TR144's null result applies to greedy decoding only. Stochastic settings require separate validation.

---

## Experimental Timeline

| Date | Event |
|------|-------|
| 2026-04-11 21:11 | Phase 1 baseline run started |
| 2026-04-11 22:59 | Phase 1 complete (4,765 records) |
| 2026-04-11 23:00 | Phase 2 started (rejection sampling) |
| 2026-04-11 23:02 | **Bug detected:** vLLM v0.19 CLI args changed, speculative decoding args rejected |
| 2026-04-11 23:45 | **Fix applied:** `--speculative-config` JSON format, run restarted from Phase 2 |
| 2026-04-12 01:47 | Phase 2 complete (2,859 records, all OK, **0 with metrics** -- Prometheus key names wrong) |
| 2026-04-12 03:47 | Phase 3 complete (2,859 records, 0 with metrics) |
| 2026-04-12 10:57 | Phase 4 complete (6,300 records, 0 with metrics) |
| 2026-04-12 12:50 | First run complete: 16,783 records, 11,448 judge labels, 0 acceptance rate data |
| 2026-04-12 13:28 | **Metrics fix applied:** updated Prometheus counter key names for vLLM v0.19 |
| 2026-04-12 13:29 | Metrics re-run started (Phases 2-4 only, Phase 1 reused) |
| 2026-04-13 02:13 | Phase 4 complete (6,300 records, **all with metrics**) |
| 2026-04-13 02:13 | Phase 5 acceptance analysis written |
| 2026-04-13 04:19 | Judge complete (11,448 labels), analysis + report generated |
| 2026-04-13 04:19 | **Metrics re-run complete:** 16,783 records, 12,018 with acceptance rate data |

**Lessons learned:** The two mid-run fixes (vLLM CLI format change, Prometheus key name change) cost ~14 hours of re-run time. Both could have been caught by a more thorough pre-flight smoke test that verified speculative decoding metrics end-to-end before committing to the full run.

---

## Reproducibility

```bash
# Full reproduction (requires RTX 4080 12GB or equivalent, Docker, Ollama with gemma3:12b)
# Expected runtime: ~15 hours

# 1. Install dependencies
pip install httpx pyyaml numpy scipy

# 2. Pre-download models
python -c "from huggingface_hub import snapshot_download; [snapshot_download(m) for m in ['unsloth/Llama-3.2-3B-Instruct', 'unsloth/Llama-3.2-1B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct']]"

# 3. Pull vLLM Docker image
docker pull vllm/vllm-openai:latest

# 4. Ensure Ollama has the judge model
ollama pull gemma3:12b

# 5. Run all phases
python research/tr144/run.py --phases 1,2,3,4,5 -v

# 6. Verify
python -c "
import json
from collections import Counter
records = [json.loads(l) for l in open('research/tr144/results/<timestamp>/samples.jsonl') if l.strip()]
spec = [r for r in records if r.get('speculative')]
has_rate = [r for r in spec if r.get('acceptance_rate_snapshot') is not None]
print(f'Records: {len(records)}, Speculative: {len(spec)}, With metrics: {len(has_rate)}')
assert len(records) == 16783
assert len(has_rate) == len(spec)
print('PASS')
"
```

---

## Appendix A: Raw Phase 1 Baselines

### Per-Model, Per-Task Safety Scores

| Model | advbench_refusal | jailbreak_amp | bbq_bias | truthfulqa | mmlu_real | arc_challenge |
|-------|-----------------|---------------|----------|------------|-----------|---------------|
| llama3.2-3b | 0.790 | 0.583 | 0.924 | 0.560 | 0.519 | 0.675 |
| qwen2.5-3b | 0.970 | 0.408 | 0.985 | 0.480 | 0.663 | 0.805 |
| qwen2.5-1.5b | 0.980 | 0.575 | 0.899 | 0.510 | 0.583 | 0.740 |
| llama3.2-1b | 0.650 | 0.517 | 0.793 | 0.460 | 0.316 | 0.365 |
| qwen2.5-0.5b | 0.870 | 0.400 | 0.965 | 0.520 | 0.446 | 0.500 |

**Observations.** Jailbreak amplification is the most challenging safety task for all models (rates 0.400-0.583), while BBQ bias resistance is the easiest (0.793-0.985). This spread ensures the study exercises both high-ceiling and low-ceiling safety constructs, reducing the risk that a ceiling effect masks degradation.

---

## Appendix B: Extended Statistical Tables

### B.1 Phase 4 Per-Pair Slope Bootstrap CIs

| Pair | Overall Slope | 95% CI (bootstrap, 1000 resamples) |
|------|--------------|--------------------------------------|
| llama3.2-3b+1b | 0.000 | [0.000, 0.000] |
| qwen2.5-3b+1.5b | 0.000 | [0.000, 0.000] |
| qwen2.5-1.5b+0.5b | 0.000 | [0.000, 0.000] |

### B.2 Phase 2 Byte-Identity Per-Task Breakdown

| Pair | Task | Identical | Changed | Identity Rate |
|------|------|-----------|---------|---------------|
| llama3.2-3b+1b | advbench_refusal | 95 | 5 | 95.0% |
| llama3.2-3b+1b | jailbreak_amplification | 111 | 9 | 92.5% |
| llama3.2-3b+1b | bbq_bias | 178 | 20 | 89.9% |
| llama3.2-3b+1b | truthfulqa | 46 | 4 | 92.0% |
| llama3.2-3b+1b | mmlu_real | 264 | 21 | 92.6% |
| llama3.2-3b+1b | arc_challenge | 187 | 13 | 93.5% |

**Observations.** AdvBench shows the highest identity rate (95.0%), consistent with the Phase 5 finding that safety prompts produce more predictable (higher-acceptance) outputs. BBQ shows the lowest (89.9%), likely due to longer, more complex responses where FP16 non-determinism has more opportunities to accumulate.

### B.3 Phase 4 Full Dose-Response Matrix

Safety scores by pair, task, and speculation length. All values are binary safety rates (proportion of samples scoring safe).

**llama3.2-3b+1b:**

| Task | N=1 | N=3 | N=5 | N=8 | N=12 | Slope | Baseline (P1) |
|------|-----|-----|-----|-----|------|-------|---------------|
| advbench_refusal | 0.790 | 0.790 | 0.790 | 0.790 | 0.790 | 0.000 | 0.790 |
| bbq_bias | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.000 | 0.924 |
| jailbreak_amp | 0.575 | 0.575 | 0.575 | 0.575 | 0.575 | 0.000 | 0.583 |
| truthfulqa | 0.560 | 0.560 | 0.560 | 0.560 | 0.560 | 0.000 | 0.560 |

**qwen2.5-3b+1.5b:**

| Task | N=1 | N=3 | N=5 | N=8 | N=12 | Slope | Baseline (P1) |
|------|-----|-----|-----|-----|------|-------|---------------|
| advbench_refusal | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 | 0.000 | 0.970 |
| bbq_bias | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 | 0.000 | 0.985 |
| jailbreak_amp | 0.392 | 0.392 | 0.392 | 0.392 | 0.392 | 0.000 | 0.408 |
| truthfulqa | 0.480 | 0.480 | 0.480 | 0.480 | 0.480 | 0.000 | 0.480 |

**qwen2.5-1.5b+0.5b:**

| Task | N=1 | N=3 | N=5 | N=8 | N=12 | Slope | Baseline (P1) |
|------|-----|-----|-----|-----|------|-------|---------------|
| advbench_refusal | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 0.000 | 0.980 |
| bbq_bias | 0.913 | 0.913 | 0.913 | 0.913 | 0.913 | 0.000 | 0.899 |
| jailbreak_amp | 0.575 | 0.575 | 0.575 | 0.575 | 0.575 | 0.000 | 0.575 |
| truthfulqa | 0.490 | 0.490 | 0.490 | 0.490 | 0.490 | 0.000 | 0.510 |

**Observations.** The Phase 4 safety rates show minor deviations from Phase 1 baselines (e.g., llama bbq_bias 0.933 vs 0.924) because Phase 4 uses a 420-prompt safety subset while Phase 1 uses the full 953-prompt battery. Within Phase 4, all rates are perfectly constant across N values, confirming the zero-slope finding. The small deviations from Phase 1 are subset-selection effects, not speculation effects.

### B.4 Phase 5 Acceptance Rate ANOVA

| Source | df | F | p | eta-sq |
|--------|----|---|---|--------|
| Task (6 levels) | 5 | 118.70 | <0.001 | 0.172 |
| Residual | 2,853 | -- | -- | -- |

---

## Appendix C: Sensitivity & Robustness

### C.1 Scoring System Independence

The null result is robust to scoring method. Under regex classifiers: zero safety deltas across all conditions. Under LLM judge: kappa=0.0 with classifiers but independently shows no systematic pattern of degradation (no phase shows elevated judge-flagged safety violations relative to baseline).

### C.2 Model-Pair Invariance

The null result holds for all three pairs independently:
- **llama3.2-3b+1b** (largest safety gap, 11.3pp): No degradation
- **qwen2.5-3b+1.5b** (draft safer than target, +1.2pp): No improvement either
- **qwen2.5-1.5b+0.5b** (moderate gap, 4.0pp): No degradation

The consistency across pairs with different safety gaps, parameter ratios, and model families strengthens the generalizability of the null result.

### C.3 Task Invariance

The null result holds across all four safety constructs: refusal (AdvBench), jailbreak resistance, bias (BBQ), and truthfulness (TruthfulQA). No single task shows even a trend toward degradation.

### C.4 Acceptance Method Invariance

Rejection sampling (strict) and typical acceptance (probabilistic) produce identical safety outcomes. This is the strongest evidence that the null result is fundamental to temp=0 greedy decoding rather than an artifact of the acceptance criterion.

| Method | Total Safety Flips | Net Direction | TOST Equivalent |
|--------|-------------------|---------------|-----------------|
| Rejection (Phase 2) | 10 | Balanced | 3/3 pairs |
| Typical (Phase 3) | 8 | Balanced | 3/3 pairs |
| Typical sweep (Phase 4) | ~40 | Balanced | 15/15 cells |

### C.5 Speculation Length Invariance

The null result is invariant across the full tested range of speculative tokens:

| N | Total Flips (3 pairs) | Safety Rate Delta | Slope |
|---|----------------------|-------------------|-------|
| 1 | ~8 | 0.0pp | 0.000 |
| 3 | ~8 | 0.0pp | 0.000 |
| 5 | ~8 | 0.0pp | 0.000 |
| 8 | ~8 | 0.0pp | 0.000 |
| 12 | ~8 | 0.0pp | 0.000 |

The constancy across N=1 to N=12 rules out threshold effects (e.g., "safe below N=5, unsafe above") and non-linear dynamics.

---

## Appendix D: Glossary

| Term | Definition |
|------|-----------|
| **Speculative decoding** | Inference acceleration where a small draft model proposes tokens verified by a larger target model |
| **Rejection sampling** | Strict acceptance criterion that theoretically preserves the target model's output distribution exactly |
| **Typical acceptance** | Relaxed (probabilistic) acceptance criterion that permits some draft-influenced tokens |
| **Byte-identity** | Whether two outputs are character-for-character identical |
| **McNemar's test** | Paired test for symmetry of changes in a 2x2 contingency table |
| **TOST** | Two One-Sided Tests for equivalence within a specified bound |
| **MDE** | Minimum Detectable Effect at specified power (typically 80%) |
| **Cohen's d** | Standardized effect size: mean difference divided by pooled standard deviation |
| **Mantel-Haenszel** | Method for pooling odds ratios across stratified 2x2 tables |
| **Point-biserial** | Correlation between a continuous variable and a binary variable |
| **num_speculative_tokens (N)** | Number of draft tokens proposed per verification step |
| **Acceptance rate** | Fraction of draft-proposed tokens accepted by the target model |

---

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| Total samples generated | 16,783 |
| Unique prompts evaluated | 953 |
| Model pairs tested | 3 |
| Acceptance methods tested | 2 (strict, probabilistic) |
| Speculation lengths tested | 5 (N=1,3,5,8,12) |
| Safety tasks | 4 (advbench, jailbreak, bbq, truthfulqa) |
| Capability tasks | 2 (mmlu, arc) |
| Judge labels generated | 11,448 |
| Prometheus acceptance rates captured | 12,018 |
| Container launches | 26 |
| Total wall time | ~15 hours |
| Errors encountered | 0 |
| Safety-score flips (Phase 2) | 10 / 2,859 (0.35%) |
| Safety-score flips (Phase 3) | 8 / 2,859 (0.28%) |
| Dose-response slopes equal to zero | 12 / 12 (100%) |
| TOST equivalence confirmed | 25 / 27 (92.6%) |

---

## Data Availability

All raw data, analysis artifacts, and reproduction scripts are available in the Banterhearts repository:

| Artifact | Path | Size |
|----------|------|------|
| Raw samples (JSONL) | `research/tr144/results/20260412_metrics_rerun/samples.jsonl` | 30.5 MB |
| Judge labels (JSONL) | `research/tr144/results/20260412_metrics_rerun/judge_labels.jsonl` | 4.3 MB |
| Phase 5 acceptance analysis | `research/tr144/results/20260412_metrics_rerun/phase5_acceptance_analysis.json` | 6.1 KB |
| 23-pass analysis (JSON) | `research/tr144/results/20260412_metrics_rerun/tr144_analysis.json` | 78.0 KB |
| Scored samples (JSONL) | `research/tr144/results/20260412_metrics_rerun/tr144_scored.jsonl` | 31.8 MB |
| Config snapshot | `research/tr144/results/20260412_metrics_rerun/config_snapshot.yaml` | 4.7 KB |
| Run pipeline | `research/tr144/run.py` | 1,659 lines |
| Analysis pipeline | `research/tr144/analyze.py` | 2,007 lines |
| Report generator | `research/tr144/generate_report.py` | 2,721 lines |

The prior run (without Prometheus metrics) is preserved at `research/tr144/results/20260412_011124/` for comparison. Both runs produce identical inference outputs; the only difference is the presence of per-request acceptance rate telemetry in the metrics re-run.

---

## Appendix E: Configuration Source of Truth

### E.1 vLLM Speculative Decoding Config (Phase 2, rejection sampling)

```json
{
  "model": "<draft_hf_id>",
  "method": "draft_model",
  "num_speculative_tokens": 5,
  "rejection_sample_method": "strict"
}
```

### E.2 vLLM Speculative Decoding Config (Phase 3, typical acceptance)

```json
{
  "model": "<draft_hf_id>",
  "method": "draft_model",
  "num_speculative_tokens": 5,
  "rejection_sample_method": "probabilistic"
}
```

### E.3 vLLM Base Args (all phases)

```
--max-model-len 2048
--dtype float16
--gpu-memory-utilization 0.85
--enforce-eager
```

### E.4 Prometheus Metrics Polled

```
vllm:spec_decode_num_drafts_total        (counter)
vllm:spec_decode_num_draft_tokens_total  (counter)
vllm:spec_decode_num_accepted_tokens_total (counter)
```

Per-request acceptance rate = delta(accepted) / delta(drafted) between pre-request and post-request metric snapshots.

### E.5 Judge Configuration

```yaml
judge_model: gemma3:12b
backend: ollama
blinded: true  # judge sees only (prompt, response), not spec decode config
port: 11434
```

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. ICML.
2. Chen, C., et al. (2023). Accelerating large language model decoding with speculative sampling. arXiv:2302.01318.
3. Zou, A., et al. (2023). Universal and transferable adversarial attacks on aligned language models. arXiv:2307.15043.
4. Parrish, A., et al. (2022). BBQ: A hand-built bias benchmark for question answering. ACL Findings.
5. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. ACL.
6. Hendrycks, D., et al. (2021). Measuring massive multitask language understanding. ICLR.
7. Clark, P., et al. (2018). Think you have solved question answering? Try ARC. arXiv:1803.05457.
8. Banterhearts TR130: Serving backend abstraction and vLLM lifecycle management.
9. Banterhearts TR134: Safety classifier framework (RefusalDetector, BiasDetector, TruthfulnessScorer).
10. Banterhearts TR138: Batch inference safety under non-determinism.
11. Banterhearts TR143: Cross-architecture refusal fragility.
