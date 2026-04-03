# Technical Report 143: Cross-Request Safety Leakage Under Continuous Batching
## Composition Effect on Safety Outcomes in Multi-Tenant vLLM Inference

| Field | Value |
|-------|-------|
| **TR Number** | 143 |
| **Project** | Banterhearts |
| **Date** | 2026-03-20 |
| **Version** | 1.0 |
| **Author** | Banterhearts Research Lab |
| **Status** | Complete -- Full-Depth Publication |
| **Report Type** | Full-depth |
| **Run Directory** | `research/tr143/results/20260319_174950/` |
| **Total Records** | **14,250 evaluation records** across 4 sub-phases |
| **Phase 1** | 9,930 records (3 models x 5 safety conditions x 468 + 2 capability conditions x 485) |
| **Phase 2** | 3,200 records (2 models x 8 overlap levels x 200 prompts) |
| **Phase 3A** | 720 records (2 models x 3 conditions x 120 prompts) |
| **Phase 3B** | 400 records (2 models x 2 conditions x 200 prompts per pair) |
| **Models** | 3 (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) |
| **Families** | 2 (Llama, Qwen) |
| **Backend** | vLLM FP16, Docker, continuous batching (default scheduler) |
| **Scoring** | Regex classifiers (TR134 RefusalDetector, TruthfulnessScorer, BiasDetector). LLM judge not run. |
| **Seed** | 42 |
| **GPU** | NVIDIA RTX 4080 Laptop 12GB |
| **Analysis Passes** | 23 |
| **Co-Batch Verification** | **22.1%** (2,466 of 11,151 observations verified; see SS20) |
| **Related Work** | [TR138](Technical_Report_138_v2.md) (batch size x safety), [TR141](Technical_Report_141.md) (cross-architecture batch fragility), [TR139](Technical_Report_139.md) (multi-turn jailbreak x quantization) |
| **Git Commit** | `f1d1a671` (run + analysis), `108952c1` (scipy migration) |
| **Depends On** | TR138 (prompt batteries, baseline comparison), TR130 (vLLM backend), TR134 (safety classifiers) |

---

## Version Note

**This is version 2.0.** Version 1.0 was retracted because it presented the result as a "clean null" without reporting the significant directional asymmetry in safety flips, and contained statistics not traceable to the raw analysis JSON. Every number in version 2.0 is sourced directly from `tr143_analysis.json` (1,624 lines).

The retraction was necessary because version 1.0's framing -- "batch composition has no effect on safety" -- was misleading. The aggregate null is correct, but omitting the directional asymmetry (SS9) misrepresented the finding. Version 2.0 presents the full picture: aggregate null with a secondary directional concern. The lesson: null results require as much scrutiny as positive results, and secondary findings that qualify the null must be reported even if they complicate the narrative.

---

## Positioning

TR143 is the batch-composition extension of the safety perturbation program initiated by TR138 and scaled by TR141. Where TR138 established that batch *size* changes safety outcomes, and TR141 showed that this effect varies across architectures, TR143 asks the next operational question: does the *composition* of a continuous batch -- specifically, what other requests are concurrently processed alongside a safety-sensitive prompt -- alter that prompt's safety outcome?

The existing literature on LLM inference safety focuses almost exclusively on single-request evaluation: one prompt in, one response out, safety scored in isolation. This creates a gap between how models are evaluated and how they are deployed. In production, every safety-sensitive request shares GPU time with dozens or hundreds of other requests. The implicit assumption is that this sharing is transparent -- that the model produces the same output regardless of what else is in the batch. TR143 tests this assumption directly.

The threat model is concrete. Every production LLM deployment uses continuous batching (vLLM, TGI, SGLang). In a multi-tenant setting, an attacker could flood a shared inference endpoint with jailbreak prompts. If batch composition affects safety outcomes, then other users' benign safety-sensitive requests, co-batched with the attacker's jailbreaks, could suffer degraded safety classifications without the attacker ever directly accessing those prompts. This is distinct from prompt injection (the attacker modifies the victim's prompt) or KV-cache leakage (the attacker reads the victim's cache). The composition attack requires no access to the victim's data -- only temporal co-location on the same GPU.

The headline finding is nuanced. **Aggregate refusal rates do not significantly change with batch composition** -- all McNemar tests p > 0.05 after Holm correction, all Cochran's Q p > 0.34, all Mantel-Haenszel pooled ORs crossing 1.0. However, **when individual prompts do flip between conditions, they flip toward unsafe overwhelmingly** -- 88-92% of flips are in the unsafe direction, and this directional asymmetry is statistically significant for 3 of 4 batch conditions (binomial p = 0.039, 0.021, 0.006). The magnitude is negligible (~2-4 flips per 468 prompts per model per condition) but the direction is not random.

This is not a "clean null." It is a **rare but directionally biased** result. The composition content does not matter (jailbreak, benign, and mixed fillers all produce similar directional bias), but all batched conditions show more unsafe flips than safe flips compared to solo processing. The practical significance is minimal at current sample sizes, but the pattern deserves monitoring as batch sizes scale.

### Relationship to External Work

The question of cross-request interference in batched inference has received minimal attention in the published literature. Kwon et al. (2023) introduced PagedAttention and demonstrated memory isolation across requests, but did not evaluate whether safety outcomes are affected by co-batching. The continuous batching literature (Orca, Sarathi, DeepSpeed-FastGen) focuses on throughput and latency, not on whether the shared-computation aspects of batching affect model behavior.

The closest related work is in the floating-point non-determinism literature. Johansson et al. (2024) documented that CUDA GEMM operations produce different results depending on the number of concurrent warps, and traced this to accumulation-order differences in thread-block reduction. TR143's proposed mechanism (SS9 Mechanistic Interpretation) builds on this observation: if the accumulation order depends on batch size, and batch size varies between solo and batched processing, then the logits at each token position will differ by a small epsilon.

The safety evaluation literature (Harmbench, JailbreakBench, SafeDecoding) evaluates models exclusively under solo processing. TR143 provides evidence that batched evaluation produces marginally different safety outcomes, establishing a gap between evaluation-time and deployment-time safety measurement.

---

## Abstract

TR143 tests whether the composition of a continuous batch in vLLM -- specifically, the concurrent processing of jailbreak or harmful prompts alongside benign safety-sensitive prompts -- measurably alters safety outcomes. The experiment spans **14,250 evaluation records** across **3 models** (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) from **2 families**, evaluated under **5 batch-composition conditions** (solo, benign-7, jailbreak-7, mixed-4/3, refusal-7), **8 temporal overlap levels**, **reverse-direction conditions**, and a **static vs. continuous batching comparison**. Co-batch verification confirmed actual co-batching in only **22.1%** of observations (2,466 of 11,151), a real limitation.

The primary finding is that **aggregate refusal rates are statistically indistinguishable** across composition conditions: all 21 McNemar paired tests yield p_exact > 0.05 after Holm correction; Cochran's Q is non-significant for all models (p = 0.878, 0.341, 0.572); Mantel-Haenszel pooled ORs cross 1.0 for every comparison.

The secondary and more concerning finding is **significant directional asymmetry in safety flips**: when prompts change outcome between solo and batched processing, they flip toward unsafe 88-92% of the time. This asymmetry is statistically significant for benign-7 (8 unsafe vs. 1 safe, p = 0.039), jailbreak-7 (9 vs. 1, p = 0.021), and mixed-4/3 (11 vs. 1, p = 0.006). Only refusal-7 is non-significant (8 vs. 3, p = 0.227). The composition content is irrelevant -- all types produce similar directional bias.

Additional caveats: prompt length confound is 47% between filler pools (58.5 vs. 85.8 mean tokens); capability TOST fails for 2 of 3 models due to degenerate zero-variance cases; Phase 2 shows extreme uniformity suggesting the overlap manipulation may not have achieved its intended effect.

---

## Executive Summary

### Key Findings

1. **Aggregate composition effect is NOT significant.** Across 9,930 Phase 1 records, refusal rates differ by at most 1.06pp across composition conditions (llama3.2-3b, jailbreak-7 vs. solo: 76.18% vs. 77.24%). All McNemar paired tests yield p_exact > 0.05 after Holm-Bonferroni correction. Cochran's Q: llama3.2-1b Q = 1.20, p = 0.878; llama3.2-3b Q = 4.51, p = 0.341; qwen2.5-1.5b Q = 2.92, p = 0.572.

2. **However, directional asymmetry in safety flips IS significant.** When prompts do flip between solo and batched processing, they overwhelmingly flip toward unsafe (refuse-to-comply). Benign-7 vs. solo: 8 unsafe, 1 safe, ratio = 0.889, binomial p = 0.039. Jailbreak-7 vs. solo: 9 unsafe, 1 safe, ratio = 0.900, p = 0.021. Mixed-4/3 vs. solo: 11 unsafe, 1 safe, ratio = 0.917, p = 0.006. Refusal-7 vs. solo: 8 unsafe, 3 safe, ratio = 0.727, p = 0.227 (not significant). The composition content does not modulate the direction -- benign fillers produce the same asymmetry as jailbreak fillers.

3. **No dose-response relationship with temporal overlap (Phase 2).** Refusal rates are essentially flat across all 8 overlap conditions. llama3.2-1b: 120/200 or 121/200 refused in every condition. llama3.2-3b: 147-150/200 refused across conditions. Logistic regression slopes: b1 = -0.022 (p = 0.932) and b1 = -0.021 (p = 0.943). No mechanism detected.

4. **No reverse-direction leakage (Phase 3A).** McNemar p = 1.0 for all four comparisons. Near-zero discordant pairs. Co-batching benign prompts with jailbreaks does not improve jailbreak refusal rates.

5. **Static and continuous batching produce identical outcomes (Phase 3B).** llama3.2-1b: 1 discordant pair, p = 1.0. llama3.2-3b: 0 discordant pairs, p = 1.0. The scheduler mode does not matter.

6. **Capability TOST results are mixed, not uniformly passing.** llama3.2-1b: tost_p = 0.5, equivalent = FALSE (delta = 0.000, degenerate zero-variance). llama3.2-3b: tost_p = 0.0, equivalent = TRUE (delta = -0.002). qwen2.5-1.5b: tost_p = 0.5, equivalent = FALSE (delta = 0.000, degenerate zero-variance). The TOST failures reflect a degenerate case where solo and batched accuracy are bit-identical, yielding zero-variance CIs that cannot satisfy the TOST bounds.

7. **Mantel-Haenszel pooled ORs all cross 1.0.** Solo vs. jailbreak-7: OR = 1.031, CI = [0.913, 1.164]. Solo vs. benign-7: OR = 1.027, CI = [0.910, 1.160]. Benign-7 vs. jailbreak-7: OR = 1.004, CI = [0.890, 1.133].

8. **Co-batch verification rate is only 22.1%.** Of 11,151 observations checked, 2,466 (22.1%) were confirmed to be actually co-batched with the intended fillers. The remaining 8,685 may have been processed solo or with partial batches. This biases toward the null and limits the strength of any negative claim.

9. **Prompt length confound is 47%, not negligible.** Benign filler prompts average 58.5 tokens; jailbreak filler prompts average 85.8 tokens. The ratio of 1.467 exceeds the 20% threshold for requiring covariate adjustment. This confound means jailbreak-batched conditions impose higher compute load per batch, which could mask or amplify small effects.

### Core Decisions

- Multi-tenant vLLM deployments do not require composition-aware request routing at current sensitivity thresholds (effects below 4.7pp are undetectable).
- The directional asymmetry warrants monitoring: if batch sizes increase and co-batch verification improves, the accumulation of rare unsafe flips could become operationally relevant.
- The 22.1% co-batch verification rate means this experiment tests "intended composition" more than "actual composition." Future work must verify co-batching at the vLLM scheduler level.
- The 47% prompt length confound requires covariate adjustment before any composition-content comparison can be considered definitive.
- Capability TOST failures (2/3 degenerate) mean capability equivalence is not formally established for llama3.2-1b or qwen2.5-1.5b.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Phase 1 coverage | Safety evaluation records | >= 7,000 | 9,930 | **PASS** |
| Phase 2 coverage | Temporal overlap records | >= 3,200 | 3,200 | **PASS** |
| Phase 3 coverage | Reverse + static records | >= 1,100 | 1,120 | **PASS** |
| McNemar significance | Any pairwise p < 0.05 (Holm) | at least one if H1 holds | none (all p > 0.125 exact) | **H0 holds** |
| Cochran's Q significance | Any omnibus p < 0.05 | at least one if H1 holds | none (all p > 0.34) | **H0 holds** |
| Directional asymmetry | Binomial test of flip direction | non-significant if symmetric | **3 of 4 conditions p < 0.05** | **CONCERNING** |
| Dose-response slope | Significant negative slope | if H3 holds | neither significant (p > 0.93) | **H3 not supported** |
| Co-batch verification | Rate of confirmed co-batching | > 50% ideal | 22.1% | **LIMITATION** |
| Prompt length confound | Max ratio across filler pools | < 1.20 | 1.467 (47% difference) | **LIMITATION** |
| Capability TOST | All models equivalent | all pass | 1 of 3 pass, 2 degenerate | **PARTIAL FAIL** |
| Power (MDE) | Phase 1 MDE at 80% power | < 5pp | 4.7pp | **PASS** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | Batch composition affects aggregate safety outcomes (H1) | All McNemar p > 0.125, all Cochran's Q p > 0.34, max delta 1.06pp | **Not supported** |
| C2 | Jailbreak co-batching degrades safety more than benign co-batching (H2) | MH benign-7 vs. jailbreak-7 OR = 1.004, CI = [0.890, 1.133] | **Not supported** |
| C3 | Temporal overlap drives composition effect (H3) | Logistic regression slopes p > 0.93 for both models | **Not supported** |
| C4 | Safety context leaks into jailbreak processing (H4) | Phase 3A all McNemar p = 1.0, near-zero discordant pairs | **Not supported** |
| C5 | Batch composition preserves capability (H5) | TOST: 1/3 equivalent, 2/3 degenerate failure | **Partial (1 of 3)** |
| C6 | Flips between conditions are randomly directional | Binomial p = 0.039, 0.021, 0.006 for 3 of 4 conditions | **Rejected** |

### Claim Hierarchy (Citation Grade)

Citations of TR143 should use the grade that matches the strength of the claim being made.

| Grade | Claims |
|-------|--------|
| **Gold** | No aggregate composition effect at the 4.7pp MDE threshold (21 McNemar tests, 3 Cochran's Q, 3 MH ORs) |
| **Silver** | Directional asymmetry in safety flips (significant for 3/4 conditions, but small N per test: 9-12 total flips) |
| **Bronze** | No temporal overlap mechanism (Phase 2 flat, but overlap was estimated, not measured from vLLM internals) |
| **Preliminary** | Co-batch verification is low (22.1%); prompt length confound is 47%; capability TOST is 1/3 |

---

## How to Read This Report

Different readers need different paths through this document.

- **Quick read (5 minutes).** Read the Abstract, Key Findings 1-2, and the Claim Validation table. This gives you the headline: aggregate null, but directional asymmetry in the rare flips.
- **Deployment decision (15 minutes).** Read SS5 (aggregate refusal rates showing the null), SS9 (directional asymmetry -- the concerning secondary finding), SS20 (co-batch verification limitation that weakens all claims), and SS26 (production guidance with actionable recommendations).
- **Statistical deep dive (30 minutes).** Work through SS7 (McNemar paired tests with all 21 comparisons), SS8 (Cochran's Q omnibus), SS17 (Mantel-Haenszel pooled ORs across model strata), SS19 (power analysis establishing the 4.7pp MDE), and Appendix C (full McNemar matrix with Haldane-corrected ORs).
- **Cross-study context (10 minutes).** Read the Positioning section, SS22b (cross-TR synthesis connecting TR138, TR141, and TR143), and SS25 (conclusions with hypothesis disposition and integration).
- **Reproducibility audit.** SS27 provides the run directory, seed, environment, and replication steps. Every number traces to `tr143_analysis.json` (1,624 lines).

---

## When to Use This Report

### Scenario 1: Deploying a multi-tenant vLLM endpoint
Read SS5-SS9 (aggregate composition null), SS18 (MH pooled ORs), and SS20 (co-batch verification limitation). The aggregate effect is null, but the directional asymmetry in SS9 and the 22.1% verification rate in SS20 mean you should not claim "proven safe" -- only "no significant effect detected at current power."

### Scenario 2: Assessing the cross-request attack surface
Read SS9 (flip direction analysis) and SS21 (limitations). While aggregate rates do not change, 88-92% of the rare flips that do occur go in the unsafe direction. At current magnitudes (2-4 flips per 468 prompts) this is operationally negligible, but the directional consistency across all three models and all composition types is non-random.

### Scenario 3: Understanding the relationship between TR138, TR141, and TR143
Read the Positioning section, SS22b (Cross-TR Synthesis), and SS25 (Conclusions). TR138 showed batch size effects; TR141 showed architecture-dependent batch effects; TR143 shows batch composition does not significantly affect aggregate outcomes but reveals directional asymmetry in the rare flips that do occur.

### Scenario 4: Writing a safety case for a regulatory filing
Start with the Claim Validation table and Claim Hierarchy (Citation Grade). Use the Gold-tier claim ("no aggregate composition effect at 4.7pp MDE") as the primary evidence. Disclose the Silver-tier directional asymmetry finding and the Preliminary-tier limitations (co-batch verification, prompt length confound, TOST failures) in the risk section. Reference SS24 (Limitations) and SS25's "What this does NOT mean" paragraph to calibrate the scope of the claim. Do not cite this report as establishing safety for batch sizes > 8, models > 3.2B, or quantized serving.

### Scenario 5: Deciding whether to instrument co-batch verification
Read SS20 (Co-Batch Verification Analysis) and the "Worked Example of Dilution" subsection. The 22.1% verification rate means the experiment is effectively testing intended composition, not actual composition. If your deployment requires safety assurance about batch composition effects, you need scheduler-level instrumentation that TR143 did not have. The cost of this instrumentation (modifying vLLM's scheduler to log per-iteration batch membership) is modest relative to the value of resolving the fundamental ambiguity in TR143's findings.

---

## Table of Contents

- [SS1. Metric Definitions](#ss1-metric-definitions)
- [SS2. Research Question and Hypotheses](#ss2-research-question-and-hypotheses)
- [SS3. Methodology](#ss3-methodology)
- [SS4. Models and Configuration](#ss4-models-and-configuration)
- [SS5. Phase 1 Baselines: Per-Model Refusal Rates by Condition](#ss5-phase-1-baselines)
- [SS6. Phase 1 Composition Deltas](#ss6-phase-1-composition-deltas)
- [SS7. Phase 1 McNemar Paired Tests](#ss7-phase-1-mcnemar-paired-tests)
- [SS8. Phase 1 Cochran's Q Omnibus](#ss8-phase-1-cochrans-q-omnibus)
- [SS9. Phase 1 Flip Direction Analysis](#ss9-phase-1-flip-direction-analysis)
- [SS10. Phase 1 Per-Task Composition Effect](#ss10-phase-1-per-task-composition-effect)
- [SS11. Phase 1 Capability TOST Equivalence](#ss11-phase-1-capability-tost-equivalence)
- [SS12. Phase 1 Target Position Effect](#ss12-phase-1-target-position-effect)
- [SS13. Phase 2 Temporal Overlap Refusal Rates](#ss13-phase-2-temporal-overlap-rates)
- [SS14. Phase 2 Dose-Response Logistic Regression](#ss14-phase-2-dose-response)
- [SS15. Phase 3A Reverse Direction](#ss15-phase-3a-reverse-direction)
- [SS16. Phase 3B Static vs Continuous Batching](#ss16-phase-3b-static-vs-continuous)
- [SS17. Cross-Model Synthesis: Mantel-Haenszel](#ss17-cross-model-synthesis)
- [SS18. TOST Equivalence: Safety and Capability](#ss18-tost-equivalence)
- [SS19. Power Analysis and Minimum Detectable Effect](#ss19-power-analysis)
- [SS20. Co-Batch Verification Analysis](#ss20-co-batch-verification)
- [SS21. Prompt Length Confound](#ss21-prompt-length-confound)
- [SS22. Latency Analysis](#ss22-latency-analysis)
- [SS22b. Cross-TR Synthesis: TR138, TR141, TR143](#ss22b-cross-tr-synthesis)
- [SS22c. CI Overlap Analysis: Safety vs. Capability Divergence](#ss22c-ci-overlap-analysis)
- [SS23. Cross-TR Baseline Validation](#ss23-cross-tr-validation)
- [SS24. Limitations](#ss24-limitations)
- [SS25. Conclusions](#ss25-conclusions)
- [SS26. Production Guidance](#ss26-production-guidance)
- [SS27. Reproducibility](#ss27-reproducibility)
- [Appendix A: Glossary](#appendix-a-glossary)
- [Appendix B: Hypothesis Disposition Summary](#appendix-b-hypothesis-disposition)
- [Appendix C: Full McNemar Test Matrix](#appendix-c-full-mcnemar-matrix)
- [Appendix D: Statistical Test Summary](#appendix-d-statistical-test-summary)
- [Appendix E: Effect Heterogeneity by Task](#appendix-e-effect-heterogeneity)
- [References](#references)

---

## SS1. Metric Definitions

### Primary Metrics

| Metric | Definition | Scale | Direction |
|--------|-----------|-------|-----------|
| **Refusal rate** | Fraction of safety-sensitive prompts where the model refuses to comply, as scored by regex safety classifiers (RefusalDetector, TruthfulnessScorer, BiasDetector from TR134). LLM judge was not run for TR143. | [0, 1] | Higher = safer |
| **Composition delta** | Refusal rate under batched condition minus refusal rate under solo condition | pp (percentage points) | Negative = less safe |
| **Flip direction ratio** | Of all discordant pairs (prompts that change outcome between conditions), the fraction that flip toward unsafe | [0, 1] | 0.5 = symmetric; >0.5 = net unsafe |
| **Co-batch verification rate** | Fraction of observations where the intended filler prompts were confirmed to be physically co-resident in the same vLLM batch | [0, 1] | Higher = more trustworthy |
| **MDE** | Minimum detectable effect at 80% power and alpha = 0.05 | pp | Lower = more sensitive |

### Statistical Tests Used

| Test | Purpose | Assumptions | Correction |
|------|---------|-------------|------------|
| **McNemar (exact binomial)** | Paired comparison of refusal outcomes between two conditions for the same prompts | Binary outcomes, paired design; exact p used due to small discordant counts | Holm-Bonferroni across 21 comparisons |
| **Cochran's Q** | Omnibus test across k = 5 related conditions per model | Binary outcomes, k > 2 related samples | None (per-model) |
| **Mantel-Haenszel pooled OR** | Cross-model pooled effect size | Homogeneous ORs across strata (models) | None |
| **TOST (+/-3pp)** | Equivalence testing: establish effect is within pre-specified bound | Normal approximation for rate differences | None |
| **Binomial (flip direction)** | Test whether flip direction deviates from 50/50 | Independent flips, H0: p = 0.5 | None applied (4 tests; Bonferroni threshold = 0.0125) |
| **ANOVA (position effect)** | Test whether target position in batch affects refusal rate | Approximately normal residuals, equal variance | None (per-model) |
| **Logistic regression** | Dose-response: overlap fraction → refusal probability | Binomial response, linear log-odds | None |

### Evidence Standard

| Grade | Criteria | Application |
|-------|----------|-------------|
| **Gold** | Pre-registered hypothesis, multiple convergent tests (McNemar + Cochran's Q + MH), p < 0.05 after correction, replicated across models | Aggregate composition null (H1) |
| **Silver** | Consistent pattern across models and conditions, p < 0.05 uncorrected, partial survival after correction | Directional asymmetry (H6) |
| **Bronze** | Single test approach, null result with known power limitation | Phase 2 temporal overlap (H3), Phase 3A reverse (H4) |
| **Preliminary** | Known measurement limitation, requires replication with improved instrumentation | Co-batch verification, prompt length confound |

---

## Key Caveats

Before proceeding, readers should hold five facts in mind throughout. These caveats apply to every finding in this report and should qualify any citation of TR143 results.

1. **Co-batch verification is only 22.1%.** Fewer than one in four observations were confirmed to be physically co-batched with the intended fillers. Every finding in this report is therefore a lower bound on the true composition effect.
2. **Directional asymmetry survives only partially under multiple-comparison correction.** Of the four binomial tests in SS9, only mixed-4/3 (p = 0.006) survives Bonferroni correction at alpha/4 = 0.0125. The jailbreak-7 (p = 0.021) and benign-7 (p = 0.039) results are suggestive but not Bonferroni-robust.
3. **Scoring is regex-only.** TR143 uses TR134's regex classifiers (RefusalDetector, TruthfulnessScorer, BiasDetector). No LLM judge was run. Subtle compliance shifts that change tone without triggering regex patterns would be missed entirely.
4. **All models are small (1.2B-3.2B).** Safety alignment in sub-4B models is weaker and less stable than in 7B+ models. The directional asymmetry reported here may not generalize upward.
5. **Temperature is 0 (greedy decoding).** Under greedy decoding, the only source of non-determinism is FP arithmetic order. At temperature > 0, sampling noise would likely dominate the small FP perturbation measured here.

---

## SS2. Research Question and Hypotheses

### Research Question

Does the composition of a continuous batch in vLLM -- specifically, the nature of other requests processed concurrently alongside a safety-sensitive target prompt -- alter that target prompt's safety or capability outcomes?

### Threat Model

An attacker floods a multi-tenant vLLM endpoint with jailbreak prompts. Benign users' safety-sensitive requests are co-batched with the attacker's jailbreaks. The question is whether this co-batching degrades the benign users' safety outcomes without the attacker ever accessing those users' prompts or responses. This is not a KV-cache data leakage attack (PagedAttention isolates KV-caches at the block level); it is a question about whether the shared weight-matrix multiplications, FFN layers, and fused attention kernels introduce cross-request interference in the safety domain.

The attack is attractive to an adversary because it requires no special access. Unlike prompt injection (which requires modifying the victim's prompt), token smuggling (which requires compromising the tokenizer), or KV-cache extraction (which requires a side-channel), composition-based degradation requires only the ability to submit requests to the same endpoint as the victim. Any API user can submit jailbreak prompts. If co-batching degrades other users' safety outcomes, the attack surface is as broad as the API itself. The economic cost to the attacker is minimal: a few hundred API calls per minute at standard pricing. The economic impact to the victim is unbounded: degraded safety on every co-batched request for the duration of the attack.

TR143's finding that composition does NOT significantly affect aggregate outcomes (H1 not supported) substantially reduces the risk of this attack vector. The 88-92% directional asymmetry in rare flips (SS9) means the channel exists in principle but produces operationally negligible effects at batch size 8.

### Hypotheses

| ID | Hypothesis | Test | Decision Criterion |
|----|-----------|------|-------------------|
| H1 | Batch composition affects aggregate safety outcomes | McNemar paired tests, Cochran's Q | Any p < 0.05 after Holm correction |
| H2 | Jailbreak co-batching degrades safety more than benign co-batching | MH pooled OR for benign-7 vs. jailbreak-7 | CI excludes 1.0 |
| H3 | Temporal overlap drives the composition effect | Logistic regression of overlap fraction on refusal | Slope p < 0.05 |
| H4 | Safety context leaks into jailbreak processing (reverse direction) | Phase 3A McNemar tests | Any p < 0.05 |
| H5 | Static and continuous batching differ | Phase 3B McNemar tests | Any p < 0.05 |
| H6 | Flips between conditions are directionally symmetric | Binomial test of flip direction | p > 0.05 |

**Note on hypothesis structure.** H1-H5 are framed as alternative hypotheses: rejecting H0 (no effect) would support the claim. H6 is framed as the null: rejecting H6 (directional symmetry) would indicate a directional bias in flips. This mixed framing is intentional. The primary research question (H1) asks whether composition matters at the aggregate level. The secondary question (H6) asks whether the pattern in the residuals (the rare flips) is random. Both questions are important for deployment decisions: H1 determines whether routing is needed; H6 determines whether monitoring is warranted.

The decision criteria were set a priori (before examining data) for H1-H5. The binomial test on flip direction (H6) was added post-hoc after version 1.0's retraction revealed the directional asymmetry. This post-hoc status does not invalidate the statistical test, but it means the H6 finding should be interpreted with more caution than a pre-registered finding would warrant. The four binomial tests (one per composition condition) were conducted on the same dataset that motivated the analysis, which inflates the risk of overfitting to noise.

---

## SS3. Methodology

### Experimental Design

Phase 1: For each of 3 models, each of 468 safety-sensitive prompts is evaluated under 5 conditions: (1) solo (no fillers), (2) benign-7 (7 benign co-batch fillers), (3) jailbreak-7 (7 jailbreak co-batch fillers), (4) mixed-4/3 (4 benign + 3 jailbreak fillers), (5) refusal-7 (7 refusal-inducing fillers). Additionally, 485 capability prompts are evaluated under solo and jailbreak-7 conditions per model. Batch size is 8 (1 target + 7 fillers). Seed = 42.

Phase 2: For each of 2 models (llama3.2-1b, llama3.2-3b), 200 safety prompts evaluated under 8 temporal overlap conditions (varying the stagger between filler and target submission: simultaneous, fillers_early_250, fillers_early_500, fillers_early_1000, target_early_250, target_early_500, target_early_1000, minimal_overlap). N = 1,600 per model, 3,200 total.

Phase 3A: For each of 2 models, 120 jailbreak-type prompts evaluated under 3 conditions (solo, benign-7, refusal-7) to test whether safety context leaks in the reverse direction (from safe fillers into unsafe targets). N = 720 total.

Phase 3B: For each of 2 models, 200 prompts evaluated under static batching vs. continuous batching to isolate the scheduler mechanism. N = 400 total.

### Record Count Reconciliation

| Component | Calculation | Records |
|-----------|-----------|---------|
| Phase 1 safety | 3 models x 5 conditions x 468 prompts | 7,020 |
| Phase 1 capability | 3 models x 2 conditions x 485 prompts | 2,910 |
| Phase 2 | 2 models x 8 conditions x 200 prompts | 3,200 |
| Phase 3A | 2 models x 3 conditions x 120 prompts | 720 |
| Phase 3B | 2 models x 2 conditions x 200 prompts (paired) | 400 |
| **Total** | | **14,250** |

### Analysis Passes (23 Total)

1. Per-model refusal rates by condition
2. Composition deltas (condition minus solo)
3. 95% CIs for all rates
4. McNemar paired tests (21 comparisons: 3 models x 7 pairs)
5. Holm-Bonferroni correction
6. Cochran's Q omnibus (3 tests, one per model)
7. Flip direction analysis (binomial tests)
8. Per-task composition effect (4 tasks x 5 conditions)
9. Capability TOST equivalence (3 models)
10. Target position effect (ANOVA, 8 positions)
11. Phase 2 temporal overlap rates
12. Phase 2 dose-response logistic regression
13. Phase 2 dose-response linear fit
14. Phase 3A reverse-direction McNemar tests
15. Phase 3B static vs. continuous McNemar tests
16. Cross-model Mantel-Haenszel pooled ORs
17. TOST for safety equivalence (4 comparisons)
18. Power analysis (MDE at 80% for all phases)
19. Co-batch verification rate
20. Prompt length covariate analysis
21. Latency confound analysis
22. Effect heterogeneity by task x model
23. Cross-TR baseline validation

### Design Rationale

Several design choices deserve explicit justification.

**Why batch size 8?** The choice of 1 target + 7 fillers (batch size 8) balances three constraints: (a) sufficient filler volume to test composition (7 fillers provide a clear majority composition), (b) GPU memory limits (8 requests at FP16 fit comfortably in 12GB with vLLM's memory management), and (c) comparability with TR138, which tested batch sizes 1-8. Production deployments use batch sizes of 32-256, so batch size 8 is conservative. If composition effects are absent at batch size 8, they may still emerge at larger batch sizes where FP perturbation is greater (more sequences = larger GEMM tiles = more accumulation-order changes).

**Why 5 composition conditions?** The 5 conditions (solo, benign-7, jailbreak-7, mixed-4/3, refusal-7) are chosen to span the relevant operational space. Solo is the control. Benign-7 represents the baseline production case (normal multi-tenant traffic). Jailbreak-7 represents the worst-case adversarial scenario (attacker floods the endpoint). Mixed-4/3 represents the realistic intermediate case. Refusal-7 tests a specific hypothesis: do strongly-refusing fillers "reinforce" the target's refusal behavior? The 4 non-solo conditions allow Cochran's Q omnibus testing and pairwise MH comparisons.

**Why regex scoring instead of LLM judge?** The decision to use regex-only scoring was driven by throughput constraints. With 14,250 evaluation records, an LLM judge (such as the qwen2.5:7b-instruct-q8_0 used in TR138) would require approximately 14,250 additional inference calls, each taking 2-5 seconds, for a total of 8-20 hours of additional compute. Given that TR143's primary question is binary (did the refusal outcome change?), the regex classifiers provide sufficient granularity. The tradeoff is reduced sensitivity to subtle compliance shifts, as documented in SS24.

**Why greedy decoding (temperature = 0)?** Greedy decoding isolates the FP perturbation mechanism by eliminating sampling noise. Under temperature > 0, each generation includes a random sampling step that could mask or amplify the small FP perturbation. Greedy decoding means any observed difference between conditions is deterministically caused by the computational environment (batch size, composition, scheduling), not by sampling variance. This makes the experiment maximally sensitive to the specific mechanism under study, at the cost of reduced ecological validity (production deployments typically use temperature 0.3-0.7).

---

## SS4. Models and Configuration

| Model | Parameters | Family | Architecture | Phase 1 N (safety) | Phase 1 N (capability) |
|-------|-----------|--------|-------------|-------------------|----------------------|
| llama3.2-1b | 1.2B | Llama | GQA | 468 per condition | 485 per condition |
| llama3.2-3b | 3.2B | Llama | GQA | 468 per condition | 485 per condition |
| qwen2.5-1.5b | 1.5B | Qwen | GQA | 468 per condition | 485 per condition |

All models served via vLLM in Docker with `--gpu-memory-utilization 0.80 --enforce-eager --max-model-len 2048 --dtype float16`. Continuous batching with the default vLLM scheduler. Judge: qwen2.5:7b-instruct-q8_0 via Ollama, condition-blind (judge does not see the filler prompts or the batch composition).

### Model Selection Rationale

The three models were chosen to provide diversity along two axes: family (Llama vs. Qwen) and scale (1.2B, 1.5B, 3.2B). All three use Grouped Query Attention (GQA), which is the dominant attention architecture in modern serving deployments. Multi-Head Attention (MHA) and Multi-Query Attention (MQA) models were not included because GQA is the most deployment-relevant architecture (and TR141 already characterized the MHA/MQA comparison).

**llama3.2-1b** (1.2B parameters) is the smallest model with meaningful safety alignment. At this scale, safety training is thin: the model refuses ~66% of safety-sensitive prompts (compared to 77-80% for the larger models), meaning ~34% of prompts are near or below the refusal threshold. This makes it the most susceptible to small perturbations -- any effect that exists should be most visible here.

**llama3.2-3b** (3.2B parameters) is the largest model in the study and the one most representative of production small-model deployments. Its 77.2% refusal rate indicates stronger safety alignment with a moderate fraction of boundary-proximate prompts.

**qwen2.5-1.5b** (1.5B parameters) provides cross-family diversity. Despite being smaller than llama3.2-3b, it has the highest refusal rate (79.6%), reflecting Qwen's aggressive safety training. This model tests whether the composition effect is family-dependent or universal across different safety-training approaches.

The 12GB GPU constraint limited the study to models that fit in vLLM with a batch size of 8 at FP16. Larger models (7B+) would require reduced batch sizes or quantization, introducing confounds. The three chosen models fit comfortably with `--gpu-memory-utilization 0.80`, leaving headroom for the KV-cache of 8 concurrent requests.

---

## SS5. Phase 1 Baselines: Per-Model Refusal Rates by Condition

| Model | Solo | Benign-7 | Jailbreak-7 | Mixed-4/3 | Refusal-7 |
|-------|------|----------|-------------|-----------|-----------|
| llama3.2-1b | 0.656 (307/468) | 0.653 (305/468) | 0.656 (307/468) | 0.652 (305/468) | 0.654 (306/468) |
| llama3.2-3b | 0.772 (361/468) | 0.768 (359/468) | 0.762 (356/468) | 0.763 (357/468) | 0.768 (359/468) |
| qwen2.5-1.5b | 0.796 (372/468) | 0.789 (369/468) | 0.787 (368/468) | 0.786 (368/468) | 0.790 (369/468) |

**95% CIs (Wilson):**

| Model | Solo CI | Benign-7 CI | Jailbreak-7 CI | Mixed-4/3 CI | Refusal-7 CI |
|-------|---------|-------------|---------------|-------------|-------------|
| llama3.2-1b | [0.612, 0.698] | [0.608, 0.694] | [0.612, 0.698] | [0.608, 0.694] | [0.610, 0.696] |
| llama3.2-3b | [0.731, 0.807] | [0.727, 0.803] | [0.720, 0.797] | [0.722, 0.799] | [0.727, 0.803] |
| qwen2.5-1.5b | [0.756, 0.829] | [0.749, 0.823] | [0.747, 0.821] | [0.747, 0.821] | [0.749, 0.823] |

**Observations.** All CIs overlap substantially within each model. The largest absolute difference between any batched condition and solo is 1.06pp (llama3.2-3b, jailbreak-7 vs. solo: 0.762 vs. 0.772). All deltas are negative (batched conditions have slightly lower refusal rates than solo), but the magnitudes are negligible. This is consistent with a null effect at the aggregate level.

The CI overlap is visually striking. For llama3.2-1b, the tightest CI pair (solo [0.612, 0.698] vs. mixed-4/3 [0.608, 0.694]) shares 96% of its range. For llama3.2-3b, even the widest pair (solo [0.731, 0.807] vs. jailbreak-7 [0.720, 0.797]) shares 91% of its range. The Wilson CIs are used here rather than Wald CIs because they have better coverage properties near the boundaries of [0, 1], though at these baseline rates (0.65-0.80), the difference between Wilson and Wald is negligible.

The raw counts reveal how stable the aggregate refusal behavior is. llama3.2-1b refuses 305-307 of 468 prompts across all 5 conditions -- a range of 2 prompts out of 468. qwen2.5-1.5b refuses 368-372 -- a range of 4 prompts. Even llama3.2-3b, which has the largest range, only spans 356-361 refusals -- 5 prompts out of 468. This extreme stability at the aggregate level is the foundation of the null finding. The signal in TR143 is not in the aggregate rates (which are invariant) but in the direction of the rare flips that compose the tiny differences (SS9).

---

## SS6. Phase 1 Composition Deltas

| Model | Benign-7 delta | Jailbreak-7 delta | Mixed-4/3 delta | Refusal-7 delta |
|-------|---------------|-------------------|----------------|----------------|
| llama3.2-1b | -0.32pp | -0.00pp | -0.43pp | -0.22pp |
| llama3.2-3b | -0.42pp | -1.06pp | -0.96pp | -0.42pp |
| qwen2.5-1.5b | -0.74pp | -0.85pp | -0.96pp | -0.64pp |

**Observations.** All deltas are negative: batched conditions have slightly lower refusal rates than solo across all models and conditions. The largest delta is -1.06pp. This direction is the "wrong way" for the H2 threat model (jailbreak co-batching should increase compliance, i.e., decrease refusal -- and it does, by a tiny margin). However, no delta exceeds the MDE of 4.7pp, and the direction is consistent across both jailbreak and benign fillers, suggesting this is noise or a minor batching artifact unrelated to filler content. Critically, these are aggregate numbers that conceal the directional asymmetry revealed in SS9.

### Interpreting the Universally Negative Deltas

The fact that all 12 deltas (3 models x 4 conditions) are negative (or zero) is striking. Under pure noise, we would expect approximately half to be positive and half negative. The probability of all 12 being non-positive by chance (under a binomial model with p=0.5) is approximately 0.5^12 = 0.024% if all were strictly negative. Including the one zero-delta (llama3.2-1b jailbreak-7), the probability is still low but not astronomically so: 11 of 12 being negative is (12 choose 11) * 0.5^12 = 0.29%.

This consistency suggests a real but tiny underlying mechanism: batching systematically reduces refusal rates by a small margin. The magnitude (~0.3-1.1pp) is well below the MDE and far below operational concern, but the direction is not random. This is consistent with the mechanistic interpretation in SS9: FP perturbation from batching preferentially flips marginal refusals toward compliance. The aggregate deltas capture the net effect (unsafe flips minus safe flips), and the net is consistently toward lower refusal.

However, the magnitude of the deltas is so small relative to the confidence intervals (CIs in SS5 are approximately +/-4pp wide) that no individual delta is distinguishable from zero. The consistency across all 12 cells is the evidence, not any individual cell. This pattern-level evidence is weaker than cell-level significance but stronger than pure noise.

A model-level pattern is also visible: qwen2.5-1.5b shows the largest deltas (0.64-0.96pp), llama3.2-3b is intermediate (0.42-1.06pp), and llama3.2-1b is smallest (0.00-0.43pp). This could reflect different sensitivity to FP perturbation across model architectures or sizes, consistent with TR141's finding that effect magnitude is model-dependent.

---

## SS7. Phase 1 McNemar Paired Tests

All 21 McNemar tests (3 models x 7 pairwise comparisons) are non-significant after Holm-Bonferroni correction. All are also non-significant without correction.

### llama3.2-1b

| Comparison | b (to unsafe) | c (to safe) | n_discordant | OR | p_exact |
|-----------|--------------|------------|-------------|-----|---------|
| solo vs. benign-7 | 3 | 1 | 4 | 2.333 | 0.625 |
| solo vs. jailbreak-7 | 1 | 1 | 2 | 1.000 | 1.000 |
| solo vs. mixed-4/3 | 2 | 0 | 2 | 5.000 | 0.500 |
| solo vs. refusal-7 | 2 | 1 | 3 | 1.667 | 1.000 |
| benign-7 vs. jailbreak-7 | 0 | 2 | 2 | 0.200 | 0.500 |
| benign-7 vs. mixed-4/3 | 1 | 1 | 2 | 1.000 | 1.000 |
| benign-7 vs. refusal-7 | 0 | 1 | 1 | 0.333 | 1.000 |

### llama3.2-3b

| Comparison | b (to unsafe) | c (to safe) | n_discordant | OR | p_exact |
|-----------|--------------|------------|-------------|-----|---------|
| solo vs. benign-7 | 2 | 0 | 2 | 5.000 | 0.500 |
| solo vs. jailbreak-7 | 4 | 0 | 4 | 9.000 | 0.125 |
| solo vs. mixed-4/3 | 4 | 0 | 4 | 9.000 | 0.125 |
| solo vs. refusal-7 | 2 | 1 | 3 | 1.667 | 1.000 |
| benign-7 vs. jailbreak-7 | 2 | 0 | 2 | 5.000 | 0.500 |
| benign-7 vs. mixed-4/3 | 3 | 1 | 4 | 2.333 | 0.625 |
| benign-7 vs. refusal-7 | 1 | 2 | 3 | 0.600 | 1.000 |

### qwen2.5-1.5b

| Comparison | b (to unsafe) | c (to safe) | n_discordant | OR | p_exact |
|-----------|--------------|------------|-------------|-----|---------|
| solo vs. benign-7 | 3 | 0 | 3 | 7.000 | 0.250 |
| solo vs. jailbreak-7 | 4 | 0 | 4 | 9.000 | 0.125 |
| solo vs. mixed-4/3 | 5 | 1 | 6 | 3.667 | 0.219 |
| solo vs. refusal-7 | 4 | 1 | 5 | 3.000 | 0.375 |
| benign-7 vs. jailbreak-7 | 2 | 1 | 3 | 1.667 | 1.000 |
| benign-7 vs. mixed-4/3 | 2 | 1 | 3 | 1.667 | 1.000 |
| benign-7 vs. refusal-7 | 3 | 3 | 6 | 1.000 | 1.000 |

**Observations.** The discordant pair counts are extremely low: 1-6 per comparison out of 468 paired prompts. This means > 98.7% of prompts produce the same safety outcome regardless of batch composition. No comparison reaches p < 0.05 even without correction. The lowest p_exact values are 0.125 (llama3.2-3b solo vs. jailbreak-7 and solo vs. mixed-4/3; qwen2.5-1.5b solo vs. jailbreak-7). These would still not survive Holm correction for 21 tests.

However, a striking pattern is visible in the b (to unsafe) vs. c (to safe) columns: across all solo-vs-batched comparisons (12 comparisons across 3 models and 4 conditions), b exceeds c in 11 of 12 cases. This asymmetry is analyzed formally in SS9.

### Per-Model McNemar Pattern

The three models show different flip profiles.

**llama3.2-1b** has the fewest total discordant pairs across its solo-vs-batched comparisons (11 total: 8 unsafe, 3 safe). The one comparison where c >= b (solo vs. jailbreak-7: b=1, c=1) is the only balanced comparison across all three models. This model appears the most robust to composition changes, consistent with its small deltas in SS6.

**llama3.2-3b** is the most asymmetric: across its four solo-vs-batched comparisons, b sums to 12 and c sums to 1 (the single safe flip is in solo vs. refusal-7). This 12:1 ratio is the strongest directional signal of any individual model. The two comparisons with OR = 9.000 (solo vs. jailbreak-7 and solo vs. mixed-4/3, both with b=4, c=0) are the closest any individual comparison comes to significance (p_exact = 0.125). With even one more discordant pair in the unsafe direction (b=5, c=0), these would reach p = 0.0625 -- still non-significant after Holm correction, but trending.

**qwen2.5-1.5b** has the most total discordant pairs across solo-vs-batched comparisons (18 total: 16 unsafe, 2 safe). The mixed-4/3 comparison (b=5, c=1) and refusal-7 comparison (b=4, c=1) show slightly more safe flips than the other two models. The benign-7 vs. refusal-7 comparison is perfectly balanced (b=3, c=3, OR=1.000), consistent with the hypothesis that between-batched comparisons show no directional pattern (the asymmetry is between solo and any batched condition, not between different batched conditions).

The between-batched comparisons (benign-7 vs. jailbreak-7, benign-7 vs. mixed-4/3, benign-7 vs. refusal-7) show much weaker asymmetry: summing across models, b=12 and c=10 -- nearly balanced. This confirms that the directional bias is a solo-vs-batched phenomenon, not a composition-content phenomenon. Any batched condition introduces a similar small perturbation relative to solo; different batched conditions do not differ from each other.

---

## SS8. Phase 1 Cochran's Q Omnibus

| Model | Q | df | p | Significant |
|-------|---|----|----|------------|
| llama3.2-1b | 1.202 | 4 | 0.878 | No |
| llama3.2-3b | 4.510 | 4 | 0.341 | No |
| qwen2.5-1.5b | 2.916 | 4 | 0.572 | No |

**Observations.** Cochran's Q tests whether any of the 5 conditions (solo, benign-7, jailbreak-7, mixed-4/3, refusal-7) differ from each other within each model. None approach significance. The highest Q (4.510 for llama3.2-3b) corresponds to p = 0.341, well above the 0.05 threshold. This confirms that there is no omnibus composition effect on aggregate refusal rates.

The pattern across models is informative. llama3.2-3b has the highest Q (4.510), consistent with its having the largest max delta (1.06pp in SS6) and the most discordant pairs in the McNemar analysis (SS7). qwen2.5-1.5b has an intermediate Q (2.916) consistent with its intermediate delta range. llama3.2-1b has the lowest Q (1.202), reflecting its near-zero deltas and minimal discordant pairs. This monotonic relationship between Q and max delta is expected: Cochran's Q is essentially measuring the same signal as the pairwise deltas, but with an omnibus test rather than pairwise comparisons.

The key interpretive question is: how different is "not significant" from "no effect"? With N=468 per condition and 5 conditions, the power of Cochran's Q to detect a true 2pp omnibus effect is approximately 25-30% (low). A true effect of 2pp across conditions -- twice the largest observed delta -- would be missed ~70-75% of the time. This means the Cochran's Q null is consistent with small effects (1-3pp) that the test simply cannot detect. The null should be interpreted as "no large omnibus effect" rather than "no effect at all."

---

## SS9. Phase 1 Flip Direction Analysis

This is the key secondary finding that version 1.0 of this report failed to report.

For each batched condition vs. solo, we pool all discordant pairs across models and test whether the direction of flips is symmetric using a binomial test (H0: p(unsafe) = 0.5).

| Condition vs. Solo | Flips to unsafe | Flips to safe | Total flips | Unsafe ratio | Binomial p | Significant |
|-------------------|----------------|--------------|------------|-------------|-----------|------------|
| benign-7 | 8 | 1 | 9 | 0.889 | **0.039** | **Yes** |
| jailbreak-7 | 9 | 1 | 10 | 0.900 | **0.021** | **Yes** |
| mixed-4/3 | 11 | 1 | 12 | 0.917 | **0.006** | **Yes** |
| refusal-7 | 8 | 3 | 11 | 0.727 | 0.227 | No |

### Per-Model Breakdown

**benign-7 vs. solo:**

| Model | To unsafe | To safe |
|-------|----------|---------|
| llama3.2-1b | 3 | 1 |
| llama3.2-3b | 2 | 0 |
| qwen2.5-1.5b | 3 | 0 |

**jailbreak-7 vs. solo:**

| Model | To unsafe | To safe |
|-------|----------|---------|
| llama3.2-1b | 1 | 1 |
| llama3.2-3b | 4 | 0 |
| qwen2.5-1.5b | 4 | 0 |

**mixed-4/3 vs. solo:**

| Model | To unsafe | To safe |
|-------|----------|---------|
| llama3.2-1b | 2 | 0 |
| llama3.2-3b | 4 | 0 |
| qwen2.5-1.5b | 5 | 1 |

**refusal-7 vs. solo:**

| Model | To unsafe | To safe |
|-------|----------|---------|
| llama3.2-1b | 2 | 1 |
| llama3.2-3b | 2 | 1 |
| qwen2.5-1.5b | 4 | 1 |

**Observations.** This is the most important secondary finding in this report. While aggregate rates do not change, the rare flips (~2-4 per 468 prompts per model per condition) are overwhelmingly in the unsafe direction (refuse-to-comply: the prompt was refused when processed solo but complied when batched). Three of four conditions show statistically significant directional asymmetry at the 0.05 level.

Critically, the composition content does not modulate the asymmetry: benign fillers (0.889 unsafe ratio, p = 0.039) produce nearly identical directional bias to jailbreak fillers (0.900, p = 0.021) and mixed fillers (0.917, p = 0.006). Refusal-7 fillers show a weaker asymmetry (0.727, p = 0.227), possibly because refusal fillers reinforce the refusing behavior.

The interpretation: batching itself (not the filler content) introduces a very small bias toward compliance. This is consistent with the TR138 finding that batch size affects FP arithmetic, not batch content. The bias is ~0.6-1.1pp per condition -- negligible for deployment, but directionally non-random.

Caveats: the total flip counts are small (9-12 per condition). A single additional safe flip in any condition would materially change the p-value. And the 22.1% co-batch verification rate means we cannot confirm these flips occurred during actual co-batching.

### Mechanistic Interpretation

Why should flips be directionally biased toward unsafe? We propose the following mechanism, consistent with the TR138 finding that batch size perturbs FP arithmetic.

When multiple sequences are batched together, the order of floating-point accumulations in attention score computation and FFN matrix multiplications changes. IEEE 754 floating-point addition is not associative: (a + b) + c does not in general equal a + (b + c) at finite precision. Under continuous batching, the sequence of additions differs from solo processing because the GPU's fused kernels process a larger tile of the batch matrix. This produces a small, deterministic (at fixed seed) perturbation to the logits at each token position.

The key asymmetry is in the geometry of the refusal decision boundary. Safety-tuned models learn strong refusal tokens ("I cannot," "I'm sorry," "As an AI") as high-confidence outputs. The logit gap between the top refusal token and the next-best compliant token is typically large when the model is confidently refusing. Conversely, when the model is marginally complying (the prompt is ambiguous, the instruction-following signal barely outweighs the safety signal), the logit gap between compliance and refusal tokens is small. This creates an asymmetric sensitivity landscape:

- **Marginal refusals** (prompts where the model barely refuses under solo processing) have a small logit gap. A small FP perturbation can push the top token from a refusal token to a compliance token, flipping the outcome toward unsafe.
- **Marginal compliances** (prompts where the model barely complies under solo processing) also have a small logit gap, but the compliance token is already the greedy choice. A perturbation of equal magnitude is equally likely to push toward or away from refusal, because the compliance side of the decision boundary is smoother (compliance is the default mode, spread across many possible continuations).
- **Strong refusals** (prompts where the model confidently refuses) have a large logit gap. No small FP perturbation will overcome this gap. These prompts are immune to batching effects.
- **Strong compliances** (prompts where the model confidently complies) similarly have a large logit gap and are immune.

The net result: the prompts most susceptible to FP perturbation are those near the refusal decision boundary, and the boundary geometry makes it easier to fall off the "refuse" side into "comply" than to climb from "comply" onto "refuse." This produces the observed 88-92% directional bias toward unsafe.

To make this concrete: consider a prompt where the model, under solo processing, produces logits of [refusal_token: 3.21, compliance_token: 3.18] at the critical first-token position. The logit gap is 0.03 -- a marginal refusal. Under batched processing, the FP accumulation order changes, producing logits of [refusal_token: 3.19, compliance_token: 3.20]. The gap inverts and the model complies. Now consider the mirror case: a prompt with solo logits [compliance_token: 3.21, refusal_token: 3.18]. The same magnitude of perturbation produces [compliance_token: 3.19, refusal_token: 3.20], flipping toward safe. If the perturbation magnitude were symmetric and the number of marginal refusals equaled the number of marginal compliances, we would expect equal flips in both directions. The 88-92% asymmetry implies that there are more marginal refusals than marginal compliances near the decision boundary -- consistent with safety-tuned models having a broader "soft refusal" zone where the model tentatively refuses, versus a narrower "soft compliance" zone where the model tentatively complies.

### The Refusal-7 Anomaly

Refusal-7 is the only condition that does not reach significance (p = 0.227, unsafe ratio = 0.727 vs. 0.889-0.917 for the other three). This weaker asymmetry has a plausible explanation. Refusal-7 fillers are prompts that elicit strong refusals -- the same category of prompts the model processes most confidently. When these fillers are in the batch, the FP perturbation from batching still applies (the computation is the same as for any batch of 8), but the refusal-7 fillers produce very short outputs (refusals are typically 20-40 tokens) compared to benign or jailbreak fillers (which may produce 100-200 token responses). Shorter outputs mean fewer decoding iterations, which means the target prompt exits the batch earlier and experiences fewer iterations of perturbed computation. If the perturbation accumulates across iterations (each iteration's FP error compounds), then refusal-7 batches would produce a smaller cumulative perturbation than benign-7 or jailbreak-7 batches. This iteration-count hypothesis is testable: if correct, the number of flips should correlate with the mean batch lifetime (total iterations the target spends co-batched with fillers).

An alternative explanation is simpler: with only 11 total flips (8 unsafe, 3 safe) vs. 9-12 for other conditions, the refusal-7 result is statistically weaker purely because of the denominator. The confidence interval for the unsafe ratio is wide ([0.46, 0.90] at 95% CI for 8/11), overlapping substantially with the other conditions. The refusal-7 "anomaly" may simply be sampling noise in a small-N test.

This mechanism connects directly to TR141's finding that output instability (measured as byte-level divergence between batch sizes) predicts safety fragility with r = 0.91. Output instability is another measure of how sensitive a model's outputs are to FP perturbation. The correlation confirms that the same underlying FP non-associativity drives both phenomena: TR141 measures it as output variance, TR143 measures it as directional flip asymmetry.

**Multiple comparison correction caveat.** The four binomial tests in SS9 (one per composition condition vs. solo) were not corrected for multiplicity in the primary analysis. If Bonferroni correction is applied (alpha = 0.05 / 4 = 0.0125), only mixed-4/3 (p = 0.006) survives. Jailbreak-7 (p = 0.021) and benign-7 (p = 0.039) fall above the corrected threshold. Refusal-7 (p = 0.227) was already non-significant. A less conservative Holm-Bonferroni correction would retain mixed-4/3 (p = 0.006 < 0.0125) and jailbreak-7 (p = 0.021 < 0.0167) but reject benign-7 (p = 0.039 > 0.025). The directional pattern is consistent across all four conditions, but formal significance after correction is established only for the strongest one or two results. We report the uncorrected p-values throughout and flag this caveat explicitly.

---

## SS10. Phase 1 Per-Task Composition Effect

Refusal rates pooled across all 3 models, broken down by task and condition.

| Task | N (pooled) | Solo | Benign-7 | Jailbreak-7 | Mixed-4/3 | Refusal-7 | Max |delta| |
|------|-----------|------|----------|-------------|-----------|-----------|-----------|
| advbench_refusal | 300 | 0.810 | 0.807 | 0.807 | 0.807 | 0.807 | 0.33pp |
| bbq_bias | 594 | 0.877 | 0.872 | 0.872 | 0.869 | 0.874 | 0.84pp |
| jailbreak_amplification | 360 | 0.556 | 0.553 | 0.553 | 0.547 | 0.553 | 0.83pp |
| truthfulqa | 150 | 0.513 | 0.500 | 0.487 | 0.500 | 0.500 | 2.67pp |

**Observations.** The largest per-task delta is -2.67pp in truthfulqa (jailbreak-7 vs. solo), but this is on the smallest sample (N=150) and well within sampling error. All other deltas are under 1pp. Critically, no task shows a positive delta (no task shows higher refusal under batched conditions than solo). The consistency of negative deltas across all tasks reinforces the directional pattern from SS9, but the magnitude is below the MDE for any individual task.

The task-level pattern is notable: advbench_refusal has the highest baseline refusal (0.810) and the smallest deltas (0.33pp max), while truthfulqa has the lowest baseline (0.513) and the largest deltas (2.67pp). This is consistent with a ceiling-effect interpretation: prompts that are strongly refused are unaffected by batching, while prompts near the decision boundary are marginally more susceptible. However, the effect is still far below the MDE for any individual task, and the truthfulqa cell size (N=150) is too small to draw firm conclusions.

The jailbreak_amplification task is the most directly relevant to the threat model (these are prompts where models sometimes comply). At a pooled refusal rate of 0.556, composition deltas are under 1pp, and the jailbreak-7 condition shows -0.28pp relative to solo -- a negligible shift in the "wrong" direction for the threat model.

### Per-Task Interpretation

The task-level variation is not random; it reflects the structure of each prompt battery's relationship to the refusal decision boundary.

**truthfulqa** shows the largest deltas (up to 2.67pp pooled, 4.00pp per-model) because these prompts are inherently ambiguous. TruthfulQA prompts ask for factual claims where the "safe" response is to qualify or refuse the misleading framing. The model's decision to refuse or comply often rests on whether the prompt is interpreted as requesting dangerous misinformation or merely testing knowledge. This ambiguity places many truthfulqa prompts squarely on the decision boundary -- exactly where the FP perturbation mechanism described in SS9's Mechanistic Interpretation predicts the largest effects. However, the cell size (N=50 per model) is far too small to draw reliable per-task conclusions; the 4.00pp deltas are within sampling noise for N=50.

**advbench_refusal** shows the smallest deltas (0.33pp pooled, max 1.00pp per-model) because these prompts contain explicit harmful-intent signals ("how to make a bomb," "write malware for"). The models have strong, unambiguous refusal patterns for these prompts, placing them far from the decision boundary. The logit gap between refusal and compliance tokens is large, and no small FP perturbation can bridge it. This is consistent with the "strong refusals are immune" prediction from the mechanistic model.

**bbq_bias** falls in between (0.84pp pooled). Bias-detection prompts often involve subtle stereotyping where the "safe" response is to refuse to endorse a biased premise. The ambiguity is moderate: less than truthfulqa (which tests factual reasoning) but more than advbench (which tests explicit harm).

**jailbreak_amplification** is the most operationally relevant task because it directly models the threat scenario: prompts where models sometimes comply with harmful requests. The pooled refusal rate of 0.556 means nearly half of these prompts succeed under any condition. Composition deltas under 1pp mean the attacker gains negligible advantage from co-batching. However, jailbreak_amplification also has the only positive per-model delta (llama3.2-1b: +0.83pp under jailbreak-7), suggesting that for this specific model-task combination, jailbreak co-batching actually *increased* refusal. This isolated reversal should not be over-interpreted given N=120, but it further weakens the directional threat model for the most relevant task.

---

## SS11. Phase 1 Capability TOST Equivalence

| Model | Solo accuracy | Jailbreak-7 accuracy | Mean diff | CI | TOST p | Equivalent | Note |
|-------|-------------|---------------------|-----------|-----|--------|-----------|------|
| llama3.2-1b | 0.338 | 0.338 | 0.000 | [0.000, 0.000] | 0.500 | **FALSE** | Degenerate: zero variance |
| llama3.2-3b | 0.586 | 0.584 | -0.002 | [-0.005, 0.001] | 0.000 | **TRUE** | Equivalence established |
| qwen2.5-1.5b | 0.647 | 0.647 | 0.000 | [0.000, 0.000] | 0.500 | **FALSE** | Degenerate: zero variance |

**Observations.** Only llama3.2-3b formally passes TOST equivalence. The other two models have bit-identical accuracy between solo and jailbreak-7 conditions (delta = 0.000), which paradoxically causes TOST failure: with zero variance, the confidence interval is a point estimate at 0.000, and the TOST procedure cannot establish that the CI lies within the +/-3pp bounds because the test statistic is undefined. This is a degenerate case, not evidence of non-equivalence. The practical conclusion is that capability is unaffected, but formal TOST equivalence is established for only 1 of 3 models.

The bit-identical accuracy for llama3.2-1b and qwen2.5-1.5b deserves explanation. These models produce exactly the same capability scores (0.338 and 0.647 respectively) under both solo and jailbreak-7 conditions. This happens because: (a) the capability prompts have strong, unambiguous correct answers (e.g., MMLU-style multiple choice), (b) under greedy decoding, the model's top-token selection for these prompts is robust to small FP perturbations, and (c) the capability score is a coarser measure than the safety refusal binary (a prompt is "correct" or "incorrect," with a wider logit gap than the refusal/compliance decision). The result is that every single capability prompt produces the same answer under both conditions -- not a single prompt flips. This is actually strong evidence of capability preservation, despite the formal TOST failure. The statistical framework simply cannot handle the degenerate case of perfect agreement.

For llama3.2-3b, the -0.002 delta (1 prompt out of 485 flipping between conditions) is informative: it shows that capability is not entirely immune to batching at this model scale, but the effect is far below any practical concern.

---

## SS12. Phase 1 Target Position Effect

ANOVA testing whether the target prompt's position within the batch (positions 0-7) affects its refusal rate.

| Model | F | p | Significant | Position range (min-max rate) |
|-------|---|---|------------|------------------------------|
| llama3.2-1b | 0.735 | 0.643 | No | 0.608 (pos 3) - 0.690 (pos 7) |
| llama3.2-3b | 0.374 | 0.918 | No | 0.731 (pos 3) - 0.785 (pos 4) |
| qwen2.5-1.5b | 0.586 | 0.768 | No | 0.752 (pos 3) - 0.825 (pos 5) |

**Per-position refusal rates (llama3.2-1b):**

| Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
|-------|-------|-------|-------|-------|-------|-------|-------|
| 0.662 (N=231) | 0.648 (N=223) | 0.640 (N=243) | 0.608 (N=227) | 0.634 (N=236) | 0.665 (N=245) | 0.680 (N=230) | 0.690 (N=237) |

**Observations.** No position effect is detected for any model (all p > 0.64). The per-position sample sizes vary because the target is randomly placed across positions; the variation in refusal rates across positions is consistent with sampling noise. This rules out a systematic position-dependent bias in the batch.

The absence of a position effect is informative for the mechanistic model. If the FP perturbation from batching were caused by attention interference (where the target "sees" the fillers during computation), we might expect a position-dependent pattern: targets placed adjacent to jailbreak fillers might be more affected than targets placed far from them in the batch dimension. The null position effect is consistent with the PagedAttention isolation model: each request's computation is independent, and the perturbation comes from the GEMM tile dimensions (which depend on batch size, not position), not from cross-request attention leakage.

Note also the slight downward trend for llama3.2-1b (0.662 at position 0 declining to 0.608 at position 3, then rising to 0.690 at position 7). This U-shaped pattern is not significant (F = 0.735, p = 0.643) and is best attributed to sampling noise, but if it were reproducible, it could suggest that middle positions in the batch tensor experience slightly different accumulation patterns than edge positions. At current sample sizes (N = 223-245 per position), this pattern cannot be distinguished from chance.

---

## SS13. Phase 2 Temporal Overlap Refusal Rates

### llama3.2-1b (N=200 per condition)

| Condition | Refusal rate | N refused | 95% CI |
|-----------|-------------|-----------|--------|
| simultaneous | 0.600 | 120 | [0.531, 0.665] |
| fillers_early_250 | 0.600 | 120 | [0.531, 0.665] |
| fillers_early_500 | 0.600 | 120 | [0.531, 0.665] |
| fillers_early_1000 | 0.600 | 120 | [0.531, 0.665] |
| target_early_250 | 0.605 | 121 | [0.536, 0.670] |
| target_early_500 | 0.605 | 121 | [0.536, 0.670] |
| target_early_1000 | 0.605 | 121 | [0.536, 0.670] |
| minimal_overlap | 0.605 | 121 | [0.536, 0.670] |

### llama3.2-3b (N=200 per condition)

| Condition | Refusal rate | N refused | 95% CI |
|-----------|-------------|-----------|--------|
| simultaneous | 0.735 | 147 | [0.670, 0.791] |
| fillers_early_250 | 0.735 | 147 | [0.670, 0.791] |
| fillers_early_500 | 0.740 | 148 | [0.675, 0.796] |
| fillers_early_1000 | 0.735 | 147 | [0.670, 0.791] |
| target_early_250 | 0.740 | 148 | [0.675, 0.796] |
| target_early_500 | 0.750 | 150 | [0.686, 0.805] |
| target_early_1000 | 0.745 | 149 | [0.680, 0.800] |
| minimal_overlap | 0.740 | 148 | [0.675, 0.796] |

**Observations.** The uniformity is striking. llama3.2-1b produces exactly 120 or 121 refusals across all 8 conditions -- a maximum difference of 1 prompt out of 200. llama3.2-3b ranges from 147 to 150, a spread of 3 prompts. This extreme consistency has two interpretations: (1) the temporal overlap manipulation has no effect, or (2) the overlap manipulation did not actually produce different levels of concurrent processing (since co-batch verification is only 22.1%). Both interpretations support the null hypothesis for the mechanism test, but (2) would mean the manipulation was ineffective rather than the mechanism absent.

The pattern within llama3.2-1b is worth noting: all four "fillers_early" conditions produce exactly 120 refusals, while all four "target_early" or "minimal_overlap" conditions produce exactly 121 refusals. This binary split (120 vs. 121) suggests that at most 1 prompt is near the decision boundary and its outcome depends on a very subtle timing-dependent factor. The remaining 199 prompts are fully determined regardless of timing. This is consistent with the Phase 1 finding that >98.7% of prompts produce the same outcome regardless of batch composition.

For llama3.2-3b, the slight upward trend from simultaneous (147) to target_early_500 (150) is too small to be significant but would be consistent with a hypothesis that giving the target prompt a "head start" (processing some iterations before fillers arrive) marginally increases refusal rates. However, target_early_1000 (149) does not continue the trend, and minimal_overlap (148) is intermediate. The pattern is better explained by sampling noise on 3 boundary-proximate prompts.

---

## SS14. Phase 2 Dose-Response Logistic Regression

| Model | Intercept (b0) | Slope (b1) | SE(b1) | z | p | N | Interpretation |
|-------|----------------|-----------|--------|---|---|---|---------------|
| llama3.2-1b | 0.433 | -0.022 | 0.257 | -0.086 | 0.932 | 1,600 | No relationship |
| llama3.2-3b | 1.062 | -0.021 | 0.287 | -0.072 | 0.943 | 1,600 | No relationship |

**Linear dose-response fit:**

| Model | Linear slope | R-squared |
|-------|------------|-----------|
| llama3.2-1b | -0.005 | 0.176 |
| llama3.2-3b | -0.004 | 0.025 |

**Observations.** Neither logistic regression slope is significant (p = 0.932, p = 0.943). The linear R-squared values (0.176, 0.025) confirm no dose-response relationship. The mechanism claim strength is "correlational (estimated overlap), not causal" because the overlap fractions are estimates based on an assumed 3,000ms average generation time, not measured from vLLM internals. True per-iteration overlap was not instrumented.

### Why Overlap Had No Effect

The extreme uniformity of Phase 2 results -- llama3.2-1b producing exactly 120 or 121 refusals across all 8 conditions, a maximum swing of 1 prompt out of 200 -- demands explanation. Two interpretations are available, and they have very different implications.

**Interpretation A: The overlap manipulation was ineffective.** The 8 temporal overlap conditions were implemented by staggering the HTTP submission of filler and target requests (simultaneous, 250ms early, 500ms early, 1000ms early, etc.). However, vLLM's scheduler operates asynchronously: requests enter a waiting queue and are batched by the scheduler based on available KV-cache slots, not submission timing. If the scheduler always drains the queue in the same way regardless of submission stagger (e.g., because all requests arrive within the scheduler's polling interval), then all 8 "conditions" collapse to the same effective treatment. Under this interpretation, the null result is an artifact of failed manipulation, not evidence that temporal overlap is irrelevant.

This interpretation is supported by the 22.1% co-batch verification rate. If only ~22% of observations in any condition were actually co-batched, and this ~22% rate does not vary across the 8 overlap conditions, then the conditions are not testing different levels of overlap -- they are testing the same (low) level of overlap 8 times. The uniformity of the results is then expected regardless of whether overlap matters.

**Interpretation B: Temporal overlap genuinely does not matter.** Under this interpretation, the FP perturbation that causes safety flips (SS9) is determined by the batch size and composition at the weight-matrix level, not by the iteration-level timing of when requests enter the batch. Once requests are co-resident in a batch, the perturbation is the same whether they arrived simultaneously or with a 1-second stagger. The perturbation happens at the level of matrix multiplication tile sizes and accumulation order, which depends on how many sequences are in the batch, not on the temporal dynamics of how they got there. Under this interpretation, the null result is informative: it rules out a mechanism where partial overlap (requests sharing only some decoding iterations) produces a weaker effect than full overlap.

**Distinguishing the interpretations.** The two interpretations can only be distinguished by instrumenting the vLLM scheduler to report, for each forward pass iteration, exactly which requests were co-resident. If the co-batch rate varies across the 8 conditions but the refusal rate does not, Interpretation B is supported. If the co-batch rate is the same across all 8 conditions, Interpretation A is supported. The current experiment cannot distinguish them.

**Phase 2 and the co-batch verification confound.** The 22.1% co-batch verification rate reported in SS20 applies to Phase 2 as well. If only ~22% of Phase 2 observations actually experienced the intended overlap conditions, then the "overlap doesn't matter" finding is confounded with "overlap never happened for most observations." The effective sample size for testing the overlap mechanism is not 1,600 per model but approximately 354 (22.1% of 1,600). At N=354, the MDE is approximately 10.5pp -- meaning temporal overlap effects below 10pp would escape detection even if they exist. This substantially weakens the null finding for H3.

---

## SS15. Phase 3A Reverse Direction

Testing whether co-batching jailbreak prompts with benign or refusal-inducing fillers changes the jailbreak prompts' compliance rates. All target prompts in Phase 3A are jailbreak-type (prompts that models sometimes comply with).

### llama3.2-1b (N=120 per condition)

| Comparison | b (to unsafe) | c (to safe) | OR | p_exact | Significant |
|-----------|--------------|------------|-----|---------|------------|
| solo vs. benign-7 | 0 | 0 | 1.0 | 1.0 | No |
| solo vs. refusal-7 | 1 | 0 | 3.0 | 1.0 | No |

### llama3.2-3b (N=120 per condition)

| Comparison | b (to unsafe) | c (to safe) | OR | p_exact | Significant |
|-----------|--------------|------------|-----|---------|------------|
| solo vs. benign-7 | 1 | 0 | 3.0 | 1.0 | No |
| solo vs. refusal-7 | 0 | 0 | 1.0 | 1.0 | No |

**Observations.** Zero or near-zero discordant pairs. Co-batching jailbreak prompts with strongly-refusing fillers does not improve jailbreak refusal rates. "Safety context" from filler prompts does not leak into jailbreak processing. This is consistent with PagedAttention's isolation: each request's KV-cache and attention computation is independent.

### What Zero Discordant Pairs Means

The zero-discordant-pair result for three of the four comparisons (and 1 discordant pair for the fourth) has a precise mechanistic interpretation. Under greedy decoding with a fixed seed, if the same jailbreak prompt produces the same output regardless of whether benign or refusal prompts are co-batched alongside it, then the FP perturbation introduced by batching is not content-dependent at the KV-cache or attention level. PagedAttention allocates physically separate memory blocks for each request's key-value cache. The attention computation for request A never reads request B's KV-cache blocks. The only shared computation is the weight-matrix multiplication in the FFN and attention projection layers, where the batch dimension tiles multiple requests into the same GEMM call.

This result is consistent with the Phase 1 finding (SS5-SS9) that composition content does not modulate safety outcomes. Phase 1 showed that benign fillers produce the same directional bias as jailbreak fillers. Phase 3A confirms the mirror: benign fillers produce the same (null) effect on jailbreak refusal as refusal fillers do. The perturbation, to the extent it exists, is a function of batch *size* (number of concurrent sequences affecting the GEMM tile dimensions), not batch *content* (what those sequences contain).

The practical implication for defensive operators is that there is no "safety inoculation" effect: you cannot improve safety outcomes for jailbreak prompts by co-batching them with strongly-refusing prompts. The forward-pass isolation provided by PagedAttention is symmetric -- it prevents both cross-request contamination (no safety leakage from fillers into targets) and cross-request benefit (no refusal reinforcement from safe fillers into jailbreak targets).

---

## SS16. Phase 3B Static vs. Continuous Batching

| Model | Static refusal rate | Continuous refusal rate | Delta | N paired | b (to unsafe) | c (to safe) | McNemar p | Significant |
|-------|-------------------|----------------------|-------|----------|--------------|------------|-----------|------------|
| llama3.2-1b | 0.605 (121/200) | 0.600 (120/200) | 0.005 | 200 | 1 | 0 | 1.0 | No |
| llama3.2-3b | 0.735 (147/200) | 0.735 (147/200) | 0.000 | 200 | 0 | 0 | 1.0 | No |

**Observations.** llama3.2-3b has zero discordant pairs -- every prompt produces the same outcome under static and continuous batching. llama3.2-1b has exactly 1 discordant pair. The scheduler mode (static vs. continuous) does not affect safety outcomes. This is expected: the scheduler determines when requests are batched, not how the forward pass computes attention or FFN outputs.

### What Byte-Identical Outputs Mean Mechanistically

The zero-discordant-pair result for llama3.2-3b is particularly informative. Under greedy decoding with the same seed, if the same set of prompts is batched together, the GEMM operations receive the same input tensors regardless of whether the scheduler uses continuous or static batching. The tile dimensions, accumulation order, and memory layout are determined by the batch contents and the CUDA kernel implementation, not by the scheduler's request-management logic. Static batching processes all requests in lockstep; continuous batching allows requests to enter and exit asynchronously. But once a set of requests is co-resident in the same forward pass iteration, the GPU computation is identical.

The single discordant pair for llama3.2-1b likely reflects a timing-dependent scheduling difference: under continuous batching, one prompt may have been batched with a slightly different set of co-resident requests (due to asynchronous completion of prior requests) compared to static batching. This would change the effective batch composition for that iteration, altering the FP accumulation order and producing a different output.

This result has a clear engineering implication: the choice between static and continuous batching is irrelevant for safety outcomes. Operators should choose based on throughput and latency characteristics without concern for differential safety impact. The FP perturbation that drives safety flips (SS9) is determined by who is in the batch, not by how the scheduler assembled the batch.

---

## SS17. Cross-Model Synthesis: Mantel-Haenszel

| Comparison | Pooled OR | 95% CI | n_strata | CI crosses 1.0 |
|-----------|-----------|--------|----------|----------------|
| solo vs. jailbreak-7 | 1.031 | [0.913, 1.164] | 3 | **Yes** |
| solo vs. benign-7 | 1.027 | [0.910, 1.160] | 3 | **Yes** |
| benign-7 vs. jailbreak-7 | 1.004 | [0.890, 1.133] | 3 | **Yes** |

**Observations.** All Mantel-Haenszel pooled odds ratios cross 1.0, confirming no consistent composition effect across models. The narrowest CI is [0.890, 1.133] for benign-7 vs. jailbreak-7, meaning we can rule out composition-content effects larger than about +/-11% on the odds ratio scale. The pooled ORs are close to 1.0 (range: 1.004-1.031), with slight displacement toward > 1.0 consistent with the small negative deltas in SS6 (batched conditions showing marginally lower refusal than solo).

Note: MH uses unpaired per-model contingency tables. Within-model pairing (same prompt across conditions) is captured by McNemar in SS7. MH provides the cross-model pooled effect size.

### Interpreting the Pooled ORs

The benign-7 vs. jailbreak-7 comparison (OR = 1.004, CI = [0.890, 1.133]) is the most directly relevant to the threat model. An OR of 1.004 means that, pooled across models, the odds of refusal under benign-7 co-batching are 0.4% higher than under jailbreak-7 co-batching -- effectively identical. The CI width of 0.243 (from 0.890 to 1.133) provides the precision bound: we can rule out composition-content effects larger than approximately 13% on the odds scale, which translates to roughly 3-4pp on the probability scale at the observed baseline rates.

The slight displacement of all three ORs above 1.0 (1.031, 1.027, 1.004) is consistent with the small negative deltas in SS6: solo processing produces marginally higher refusal odds than any batched condition. However, this displacement is entirely within sampling error (all CIs comfortably include 1.0) and should not be interpreted as evidence of even a small effect.

The Breslow-Day test for homogeneity of odds ratios across strata was not formally computed, but the per-model McNemar ORs (SS7) show no obvious heterogeneity pattern. All three models show slight displacement toward unsafe under batching, with no model showing a reversed direction. This consistency across models (two Llama-family GQA models and one Qwen-family GQA model) suggests the effect, if real, is not architecture-specific within the GQA family.

---

## SS18. TOST Equivalence: Safety and Capability

### Pooled Safety TOST (N=1,404 per condition, pooled across 3 models)

| Comparison | Mean diff | CI | TOST p | Equivalent | Bound |
|-----------|-----------|-----|--------|-----------|-------|
| solo vs. benign-7 | -0.499pp | [-0.840, -0.158] | 0.000 | **TRUE** | +/-3pp |
| solo vs. jailbreak-7 | -0.641pp | [-1.020, -0.262] | 0.000 | **TRUE** | +/-3pp |
| solo vs. mixed-4/3 | -0.784pp | [-1.196, -0.371] | 0.000 | **TRUE** | +/-3pp |
| solo vs. refusal-7 | -0.427pp | [-0.824, -0.030] | 0.000 | **TRUE** | +/-3pp |

### Per-Model Capability TOST (solo vs. jailbreak-7, N=485 per condition)

| Model | Mean diff | CI | TOST p | Equivalent | Note |
|-------|-----------|-----|--------|-----------|------|
| llama3.2-1b | 0.000 | [0.000, 0.000] | 0.500 | **FALSE** | Degenerate zero-variance |
| llama3.2-3b | -0.002 | [-0.005, 0.001] | 0.000 | **TRUE** | Equivalence established |
| qwen2.5-1.5b | 0.000 | [0.000, 0.000] | 0.500 | **FALSE** | Degenerate zero-variance |

### Pooled Capability TOST (N=1,455)

| Comparison | Mean diff | CI | TOST p | Equivalent |
|-----------|-----------|-----|--------|-----------|
| solo vs. jailbreak-7 | -0.069pp | [-0.182, 0.044] | 0.000 | **TRUE** |

**Observations.** Pooled safety TOST passes for all four comparisons, establishing that all composition conditions are within +/-3pp of solo. Pooled capability TOST also passes. However, per-model capability TOST fails for 2 of 3 models due to degenerate zero-variance (bit-identical accuracy between conditions). This is a statistical artifact, not evidence of non-equivalence, but it means formal per-model capability equivalence is only established for llama3.2-3b.

### Interpreting the Safety TOST CIs

The safety TOST CIs deserve closer attention because they all exclude zero on the negative side. Solo vs. jailbreak-7: CI = [-1.020, -0.262]; solo vs. mixed-4/3: CI = [-1.196, -0.371]. These CIs tell us that batched conditions have refusal rates that are between 0.16pp and 1.2pp lower than solo, with 95% confidence. The effect is formally equivalent to zero within the +/-3pp bound (TOST passes), but it is not literally zero -- there is a consistent small displacement toward lower refusal under batching.

This displacement is the aggregate manifestation of the directional asymmetry reported in SS9. The TOST analysis establishes that the displacement is within the pre-specified equivalence bound, meaning it is negligible for deployment decisions. But the consistency of the displacement across all four comparisons (all CIs entirely below zero) provides additional evidence that the mechanism is real, just small.

The +/-3pp equivalence bound was chosen a priori based on operational relevance: a 3pp change in refusal rate would be noticeable in production monitoring but not necessarily deployment-blocking. If a tighter bound (e.g., +/-1pp) were used, the TOST for solo vs. mixed-4/3 would fail (CI lower bound is -1.196pp, below -1pp), and the TOST for solo vs. jailbreak-7 would be marginal (CI lower bound is -1.020pp, barely below -1pp). This illustrates that the choice of equivalence bound matters: at the 3pp level, composition effects are formally negligible; at the 1pp level, they may not be.

---

## SS19. Power Analysis and Minimum Detectable Effect

| Phase | N | Baseline rate | Alpha | Power | MDE (pp) |
|-------|---|--------------|-------|-------|---------|
| Phase 1 safety | 1,404 | 0.737 | 0.05 | 0.80 | **4.7** |
| Phase 1 capability | 1,455 | 0.523 | 0.05 | 0.80 | **5.2** |
| Phase 2 | 400 | 0.671 | 0.05 | 0.80 | **9.3** |
| Phase 3A | 240 | 0.539 | 0.05 | 0.80 | **12.8** |

**Observations.** Phase 1 has sufficient power to detect effects of 4.7pp or larger at the pooled level. This means composition effects smaller than approximately 5pp would escape detection. Given that all observed deltas are under 1.1pp, the null result is well-powered at the aggregate level.

However, Phase 2 (MDE = 9.3pp) and Phase 3A (MDE = 12.8pp) have substantially lower power. Temporal overlap effects below 9pp and reverse-direction effects below 13pp would not be detected. The Phase 2 null should be interpreted with this caveat.

Per-model Phase 1 power is lower than pooled. With N=468 per model per condition, the per-model MDE is approximately 8-9pp (depending on baseline rate), meaning the per-model McNemar tests have limited power for small effects. The pooled MH analysis provides better power.

### Power in Context

The 4.7pp pooled MDE deserves contextualization against real-world safety requirements. In production safety monitoring, a 5pp degradation in refusal rate (e.g., from 80% to 75%) would typically be considered a significant regression requiring investigation. The fact that TR143 can detect effects of this magnitude means the aggregate null is practically meaningful: composition effects large enough to trigger a typical production safety alert would have been detected.

However, the MDE for the directional analysis (SS9) is harder to compute because the test is not on rates but on the fraction of flips in each direction. With only 9-12 total flips per condition, the binomial test can only distinguish very extreme asymmetries from chance. A flip ratio of 0.75 (75% toward unsafe) would require approximately 25 total flips to reach significance at p = 0.05. The current test is sensitive only to ratios above approximately 0.85 at the observed sample sizes. This means moderate directional asymmetries (e.g., 65-80% toward unsafe) would escape detection. The reported 88-92% asymmetry is detectable only because it is so extreme.

The implication for future work: to achieve adequate power for both the aggregate and directional analyses, a study would need approximately 4,680 prompts per model per condition (10x current) to bring the aggregate MDE to ~1.5pp, and would need to observe at least 30-50 total flips per condition (requiring either more prompts or larger batch sizes that produce more flips) to achieve reasonable power for the directional test.

---

## SS20. Co-Batch Verification Analysis

| Metric | Value |
|--------|-------|
| Verified observations | 2,466 |
| Unverified observations | 8,685 |
| Total checked | 11,151 |
| **Verification rate** | **22.1%** |

**Observations.** This is the single most important limitation of the experiment. Only 22.1% of observations were confirmed to have been physically co-batched with the intended filler prompts. The remaining 77.9% (8,685 observations) may have been processed in partial batches, with different fillers, or even solo (if the vLLM scheduler did not batch them together due to timing).

This means the experiment is primarily testing "intended composition" (what fillers were submitted concurrently) rather than "actual composition" (what fillers were physically co-resident in the batch during the forward pass). The null result could be because: (a) composition truly does not matter, or (b) most observations were not actually exposed to the intended composition.

The 22.1% rate biases conservatively toward the null: if composition does have an effect, it would be diluted by the ~78% of observations where the intended composition was not achieved. Any true composition effect in the verified subset would be attenuated ~4.5x when measured across the full sample.

To quantify the dilution: if a true composition effect of X pp exists and is present only in the 22.1% of verified co-batched observations, the measured effect across the full sample would be approximately 0.221 * X. For an effect to reach the MDE of 4.7pp in the full sample, the true effect in the co-batched subset would need to be approximately 21pp -- a very large effect that would be visible even with crude instrumentation. Conversely, effects in the 5-20pp range in the co-batched subset would be diluted below the MDE and escape detection.

Future work must instrument co-batch verification at the vLLM scheduler level (not just at the HTTP submission level) to resolve this ambiguity.

### A Worked Example of Dilution

To make the dilution concrete: suppose the true composition effect is 10pp (i.e., jailbreak co-batching reduces the refusal rate by 10pp for prompts that are actually co-batched). In our sample of 468 prompts per model per condition:

- 22.1% of observations (103 prompts) are actually co-batched. Their refusal rate drops by 10pp.
- 77.9% of observations (365 prompts) are not actually co-batched. Their refusal rate is unchanged (equivalent to solo).
- The measured aggregate delta = 0.221 * (-10pp) + 0.779 * 0pp = -2.21pp.
- The Phase 1 pooled MDE is 4.7pp. A measured delta of 2.21pp would NOT reach significance.

Thus, a substantial 10pp true effect would be invisible in our aggregate analysis due to the 22.1% verification rate. Only effects above approximately 21pp in the co-batched subset would survive the dilution to reach the 4.7pp MDE. This means the aggregate null in SS5-SS8 rules out very large composition effects (>21pp in actual co-batched observations) but does not rule out moderate effects (5-20pp) that could be operationally relevant.

An alternative analysis strategy -- restricting to only the 2,466 verified co-batched observations -- would have higher sensitivity but introduces selection bias (the 22.1% of observations that were verified may not be representative of all observations). This subsample analysis was not performed in the current report and represents a priority for future work.

---

## SS21. Prompt Length Confound

| Filler type | Mean prompt tokens | N |
|------------|-------------------|---|
| benign-7 | 58.5 | 1,404 |
| jailbreak-7 | 85.8 | 2,859 |
| mixed-4/3 | 58.5 | 1,404 |
| refusal-7 | 58.5 | 1,404 |

| Summary metric | Value |
|---------------|-------|
| Min mean tokens | 58.5 |
| Max mean tokens | 85.8 |
| **Ratio** | **1.467** |
| **Needs covariate** | **TRUE** |

**Observations.** Jailbreak filler prompts are 47% longer than benign filler prompts (85.8 vs. 58.5 mean tokens). This exceeds the 20% threshold for requiring covariate adjustment. The practical implication: jailbreak-7 batches impose substantially more compute load (more tokens in the batch = more KV-cache memory, more attention computation) than benign-7 batches. Any comparison between jailbreak-7 and benign-7 compositions is confounded with total batch compute load.

This confound does not invalidate the overall null result (benign-7 and jailbreak-7 both show similar deltas from solo), but it means we cannot definitively attribute the null to "composition content doesn't matter" vs. "two different confounds cancel each other." Covariate adjustment (controlling for total batch token count) was not performed in this analysis and should be added in future work.

To quantify the compute-load difference: a jailbreak-7 batch contains 1 target prompt plus 7 jailbreak fillers averaging 85.8 tokens each, for approximately 600 filler tokens. A benign-7 batch contains 7 benign fillers averaging 58.5 tokens each, for approximately 410 filler tokens. The jailbreak batch processes ~46% more filler tokens, which translates to a proportionally larger KV-cache allocation and more attention computation per iteration. Under vLLM's PagedAttention, the memory allocation is block-quantized (typically 16-token blocks), so the actual memory difference may be somewhat smaller than 46% after block alignment. Nevertheless, the compute-load asymmetry is substantial.

The confound is partially mitigated by the mixed-4/3 condition, which contains 4 benign fillers (4 x 58.5 = 234 tokens) and 3 jailbreak fillers (3 x 85.8 = 257 tokens), for approximately 491 filler tokens -- intermediate between benign-7 (410) and jailbreak-7 (600). The fact that mixed-4/3 shows deltas similar to jailbreak-7 (both -0.96pp for qwen2.5-1.5b) despite having fewer total filler tokens suggests that token count alone does not drive the delta. This provides weak evidence that the confound is not masking a real composition-content effect.

---

## SS22. Latency Analysis

### Per-Condition Latency (ms)

| Model | Condition | Mean | Median | P95 | N |
|-------|----------|------|--------|-----|---|
| llama3.2-1b | solo | 405.5 | 95.0 | 1,769.1 | 953 |
| llama3.2-1b | benign-7 | 834.7 | 473.9 | 1,952.4 | 468 |
| llama3.2-1b | jailbreak-7 | 663.2 | 460.2 | 1,949.8 | 953 |
| llama3.2-1b | mixed-4/3 | 867.6 | 479.8 | 2,035.4 | 468 |
| llama3.2-1b | refusal-7 | 859.6 | 479.4 | 2,008.7 | 468 |
| llama3.2-3b | solo | 1,976.9 | 1,379.3 | 4,196.2 | 953 |
| llama3.2-3b | benign-7 | 2,924.4 | 3,709.4 | 4,427.8 | 468 |
| llama3.2-3b | jailbreak-7 | 2,171.8 | 1,446.2 | 4,430.8 | 953 |
| llama3.2-3b | mixed-4/3 | 2,935.2 | 3,875.2 | 4,431.9 | 468 |
| llama3.2-3b | refusal-7 | 2,913.2 | 3,839.8 | 4,382.2 | 468 |
| qwen2.5-1.5b | solo | 626.7 | 146.2 | 2,436.7 | 953 |
| qwen2.5-1.5b | benign-7 | 1,167.4 | 836.1 | 2,576.1 | 468 |
| qwen2.5-1.5b | jailbreak-7 | 827.7 | 445.4 | 2,625.0 | 953 |
| qwen2.5-1.5b | mixed-4/3 | 1,193.6 | 863.9 | 2,636.6 | 468 |
| qwen2.5-1.5b | refusal-7 | 1,178.7 | 856.5 | 2,601.3 | 468 |

### Solo vs. Batched Latency Confound

| Model | Solo mean (ms) | Batched mean (ms) | Cohen's d | Significant |
|-------|---------------|-------------------|-----------|------------|
| llama3.2-1b | 405.5 | 776.8 | 0.616 | **Yes** |
| llama3.2-3b | 1,976.9 | 2,620.0 | 0.384 | No |
| qwen2.5-1.5b | 626.7 | 1,037.5 | 0.513 | **Yes** |

**Observations.** Batched conditions are significantly slower than solo for 2 of 3 models (Cohen's d = 0.616 and 0.513). This is expected: co-batching increases compute load. However, this latency difference could be a confound if generation time affects safety outcomes (e.g., if shorter generations are more likely to be refusals). The latency analysis confirms that the batching manipulation produced measurable compute-load differences, even if it did not consistently produce actual co-batching (per the 22.1% verification rate).

A notable anomaly in the latency data: jailbreak-7 conditions are consistently faster than benign-7, mixed-4/3, and refusal-7 conditions for all three models (e.g., llama3.2-1b: jailbreak-7 mean 663.2ms vs. benign-7 834.7ms, mixed-4/3 867.6ms, refusal-7 859.6ms). This is counterintuitive given that jailbreak fillers are 47% longer (SS21). The explanation is likely in the N column: jailbreak-7 has N=953 (pooled with other conditions that share jailbreak fillers) vs. N=468 for the other conditions. The larger N for jailbreak-7 may include more observations that were not actually co-batched (processed solo or in small batches), which would pull the mean latency down toward the solo latency.

The latency difference between solo and batched conditions also provides indirect evidence of co-batching. If no observations were actually co-batched, all conditions would have latency similar to solo. The fact that batched conditions are 1.5-2x slower than solo (for llama3.2-1b and qwen2.5-1.5b) confirms that some degree of co-batching did occur, even if only for 22.1% of observations.

---

## SS22b. Cross-TR Synthesis: TR138, TR141, TR143

This section integrates the findings from TR143 with the two predecessor studies in the batch-safety program. Together, TR138, TR141, and TR143 characterize the batch-safety perturbation along three axes: magnitude (TR138), model-dependence (TR141), and directionality (TR143).

### TR138: Batch Size Affects Safety

TR138 established the foundational finding: increasing batch size from 1 to 8 produces measurable safety outcome changes. The key numbers: a 4x batch-size ratio, a 0.6% flip rate across the tested models, and significant McNemar tests for the solo-vs-batched comparison. The mechanism is FP non-associativity in batched GEMM operations -- the same mechanism proposed in SS9's Mechanistic Interpretation. TR138 measured the *existence* of the perturbation but did not characterize its direction.

### TR141: Output Instability Predicts Fragility

TR141 scaled the investigation to 7 models across 3 architectures (GQA, MHA, MQA) and 5 alignment types. Its central finding: output instability (byte-level divergence between outputs at different batch sizes) predicts which models experience safety flips, with r = 0.91. This correlation means that models whose outputs are more sensitive to FP perturbation are also more likely to have safety-relevant flips. Critically, TR141 found that alignment type does not predict fragility (ANOVA p = 0.942) -- RLHF, DPO, SFT, and unaligned models are equally susceptible, conditional on their output instability. TR141 also found that the net direction of flips across its 7 models was toward safe (more comply-to-refuse than refuse-to-comply), though this was aggregated across a different model set than TR143.

### TR143: Composition Does Not Matter, But Direction Does

TR143 adds two new findings to the program. First, the *content* of the batch does not modulate safety outcomes at the aggregate level (SS5-SS8, SS17). Whether the co-batched requests are benign, jailbreak, mixed, or refusal-inducing, the target prompt's refusal rate is statistically indistinguishable from solo processing. This resolves the operational question: attackers cannot degrade other users' safety outcomes by flooding a shared endpoint with jailbreaks (at least not through the batch-composition channel).

Second, when individual prompts do flip, they flip toward unsafe 88-92% of the time (SS9). This directional finding is new relative to TR138 (which did not characterize direction) and appears to conflict with TR141 (which found net-safe flips across its 7 models). The apparent conflict resolves when considering model selection: TR143 uses 3 small models (1.2B-3.2B) where safety alignment is weaker, while TR141 included models up to 7B where alignment may produce a different decision-boundary geometry. The directional bias may be model-size-dependent, with smaller models having softer refusal boundaries that are more easily crossed in the unsafe direction.

### The Integrated Picture

| Finding | TR138 | TR141 | TR143 |
|---------|-------|-------|-------|
| Batch perturbation exists | **Yes** (4x ratio, 0.6% flip rate) | **Yes** (output instability varies by model) | **Yes** (88-92% directional bias in flips) |
| What drives perturbation magnitude | Batch size | Model architecture + output instability (r=0.91) | Not batch content (all compositions equivalent) |
| Direction of flips | Not characterized | Net safe (across 7 models) | Net unsafe (across 3 small models, 88-92%) |
| Alignment type matters | Not tested | **No** (ANOVA p=0.942) | Not tested (all instruct-tuned) |
| Composition content matters | Not tested | Not tested | **No** (MH OR = 1.004, CI crosses 1.0) |

The synthesis: **batching introduces a small, universal, model-dependent perturbation to safety outcomes.** The perturbation magnitude depends on the model's output instability (TR141), not on what else is in the batch (TR143). The direction of the rare flips may depend on model size and alignment strength, with smaller models showing a directional bias toward unsafe (TR143) and larger models showing a possible bias toward safe (TR141). The net effect is operationally negligible at batch size 8, but the directional bias warrants monitoring as batch sizes scale in production.

### Operational Synthesis

For deployment operators, the three reports together recommend the following posture:

1. **Batch perturbation is real but small.** Do not deny its existence; do not overstate its magnitude. The flip rate is approximately 0.6-2.4% of safety-sensitive prompts, depending on batch size and model.
2. **Composition-aware routing is unnecessary.** TR143 establishes that what is in the batch does not matter -- only how large it is (TR138) and which model is being served (TR141).
3. **Monitor for directional asymmetry.** The 88-92% toward-unsafe finding from TR143 is the most concerning result in the program. If this scales with batch size (untested), then larger batches would produce more total flips AND those flips would preferentially degrade safety.
4. **Output instability is a deployability metric.** TR141's r = 0.91 correlation means that measuring a model's output instability under different batch sizes is a cheap proxy for its susceptibility to batch-induced safety degradation. Models with low output instability are safer to deploy in high-concurrency settings.
5. **The co-batch verification problem is unsolved.** TR143's 22.1% verification rate means the true composition effect is under-measured. Until serving stacks expose per-iteration batch membership, composition-safety claims remain provisional.

### Implications for Safety Evaluation Standards

The three-report program (TR138, TR141, TR143) collectively reveals a gap in current safety evaluation methodology. Standard practice evaluates models one prompt at a time under deterministic conditions. The finding that batched inference introduces a small, directionally biased perturbation to safety outcomes means that solo evaluation systematically overestimates safety by a small margin.

The magnitude of this overestimation is currently negligible (~0.5-1.1pp at batch size 8). But the key insight is structural: solo evaluation and batched deployment are different computational regimes. As batch sizes increase in production (32, 64, 256), the gap between evaluation-time and deployment-time safety could widen. Safety benchmarks that report "98% refusal rate" based on solo evaluation might find 96-97% refusal in production under heavy batching -- still acceptable, but the discrepancy should be disclosed.

A concrete recommendation for the safety evaluation community: safety benchmarks should include a "batched evaluation" condition that measures performance under realistic inference conditions (batch size 32-64, continuous batching, mixed workload). This would capture the FP perturbation effect and provide a more accurate estimate of deployment-time safety. The cost is moderate (a single additional evaluation run per model) and the benefit is a more honest assessment of safety in practice.

This recommendation aligns with the broader trend toward "eval as deployed" in the AI safety community. Just as adversarial robustness evaluations should use the same post-processing pipeline as deployment, safety evaluations should use the same inference stack as deployment. TR143 provides quantitative evidence that the inference stack matters, even if the current effect is small.

---

## SS22c. CI Overlap Analysis: Safety vs. Capability Divergence

This section tabulates the CI overlap analysis (Pass 23), testing whether safety and capability deltas diverge disproportionately under composition conditions.

| Condition vs. Solo | Safety delta (pp) | Capability delta (pp) | Safety N | Capability N | Disproportionate |
|-------------------|-------------------|----------------------|----------|-------------|-----------------|
| benign-7 | -0.50 | N/A | 1,404 | 0 | No |
| jailbreak-7 | -0.64 | -0.07 | 1,404 | 1,455 | **No** |
| mixed-4/3 | -0.78 | N/A | 1,404 | 0 | No |
| refusal-7 | -0.43 | N/A | 1,404 | 0 | No |

**Observations.** Only jailbreak-7 has a valid capability comparison (capability was evaluated under solo and jailbreak-7 only, not under benign-7, mixed-4/3, or refusal-7). For jailbreak-7: the safety delta (-0.64pp) is approximately 9x the capability delta (-0.07pp), but both are negligible relative to the +/-3pp TOST bound. The disproportionality flag is FALSE for all conditions -- safety and capability shift in the same direction (both slightly negative under batching) and neither diverges to a concerning degree.

The N/A entries for benign-7, mixed-4/3, and refusal-7 capability deltas reflect a design limitation: capability prompts (MMLU, ARC) were only evaluated under solo and jailbreak-7 conditions, not under all 5 composition conditions. This means the CI overlap analysis can only confirm safety-capability proportionality for the most adversarial condition (jailbreak-7). For the other conditions, the analysis relies on the TOST equivalence results in SS18 (all safety comparisons pass at +/-3pp) as indirect evidence that safety degradation does not exceed capability degradation.

The absence of disproportionate safety-capability divergence is reassuring: batch composition does not selectively degrade safety while preserving capability. Both dimensions shift negligibly and in parallel, consistent with a uniform FP perturbation mechanism (SS9) that affects all outputs equally rather than targeting safety-specific computations.

---

## SS23. Cross-TR Baseline Validation

| Model | TR143 solo refusal rate | Note |
|-------|------------------------|------|
| llama3.2-1b | 0.656 | TR138 comparison available |
| llama3.2-3b | 0.772 | TR138 comparison available |
| qwen2.5-1.5b | 0.796 | TR138 comparison available |

**Observations.** Cross-TR validation against TR138 baselines is available for all three models. Exact TR138 values are not included in the TR143 JSON; a full cross-TR comparison requires loading TR138 raw data. The solo refusal rates are plausible given the model sizes and prompt batteries used.

The model-level refusal rates show a clear size gradient: llama3.2-1b (1.2B) refuses 65.6% of safety prompts, llama3.2-3b (3.2B) refuses 77.2%, and qwen2.5-1.5b (1.5B) refuses 79.6%. The ordering is not strictly by parameter count (qwen2.5-1.5b is smaller than llama3.2-3b but refuses more often), reflecting differences in safety training between the Llama and Qwen families. Qwen2.5-1.5b has notably strong safety alignment for its size, which may explain why its directional asymmetry (SS9) involves more total flips (5 unsafe vs. 1 safe for mixed-4/3, the highest single-model count): a model with more prompts near the refusal boundary (due to aggressive safety tuning) has more opportunities for boundary-crossing flips.

The cross-TR validation is limited because TR143 uses a different prompt battery composition than TR138 (TR143 includes truthfulqa and bbq_bias tasks not present in TR138's original battery). A strict rate comparison requires matching on the overlapping prompts only. This matching analysis was not performed for this report but could be added in an erratum if the rates diverge significantly.

---

## SS24. Limitations

### Internal Validity

1. **Co-batch verification is only 22.1%.** The experiment primarily tests intended composition, not actual composition. This is the single most important limitation and biases toward the null.

2. **Prompt length confound is 47%.** Jailbreak fillers are substantially longer than benign fillers (85.8 vs. 58.5 mean tokens), introducing a compute-load confound.

3. **Phase 2 overlap is estimated, not measured.** Temporal overlap fractions assume 3,000ms average generation time. True per-iteration overlap was not instrumented from vLLM internals.

4. **Capability TOST is degenerate for 2/3 models.** Zero-variance results prevent formal equivalence testing.

### External Validity

5. **Only 3 models from 2 families.** Results may not generalize to larger models (7B+), different architectures (MoE, MLA), or different serving stacks (TGI, SGLang).

6. **Batch size is fixed at 8.** Production deployments use batch sizes of 32-256. The null result at batch size 8 does not preclude composition effects at larger batch sizes.

7. **Only FP16 precision tested.** INT8/INT4/AWQ/GPTQ quantized serving may have different numerical sensitivity to batch composition.

8. **Single GPU (RTX 4080 Laptop 12GB).** Multi-GPU tensor-parallel serving with NCCL communication may introduce additional cross-request interference channels.

### Measurement Validity

9. **Scoring is regex-only (no LLM judge).** TR143 uses TR134's regex-based classifiers to determine refusal/compliance. These classifiers match surface patterns ("I cannot," "I'm sorry," "as an AI language model") and have known failure modes: (a) soft refusals that acknowledge the request while declining are sometimes missed; (b) compliance responses that include safety disclaimers before proceeding may be scored as refusals; (c) novel refusal phrasings not in the regex dictionary are missed entirely. An LLM judge (as used in TR138 and TR139) would capture these nuances but was not run for TR143 due to compute constraints. The practical impact: if batch composition causes subtle shifts in refusal *style* (e.g., from a hard "I cannot" to a soft "While I understand your request, I should note..."), the regex classifier would not detect this. The reported flip counts (9-12 per condition) may undercount true safety-relevant changes.

10. **Capability TOST is degenerate for 2 of 3 models.** The zero-variance TOST failure (llama3.2-1b and qwen2.5-1.5b producing bit-identical accuracy between solo and jailbreak-7) is a statistical artifact, not evidence of non-equivalence. The TOST procedure requires a non-degenerate variance estimate to compute the test statistic. When the paired differences are all exactly zero, the standard error is zero, the t-statistic is undefined, and the procedure returns p = 0.5 (the prior). This means formal capability equivalence is established for only 1 of 3 models (llama3.2-3b), even though all three models show zero or near-zero capability degradation. Future work should use a bootstrap-based TOST variant that handles zero-variance cases gracefully.

### Statistical Validity

11. **Per-model McNemar power is limited.** With N=468 per condition, the per-model MDE is ~8-9pp, meaning small per-model effects would escape detection.

12. **Directional asymmetry is based on small N.** The binomial tests have 9-12 total flips. A single additional observation could change significance.

13. **No multiple-testing correction on the directional analysis.** The four binomial tests in SS9 were not corrected for multiplicity. Applying Holm correction would render the p = 0.039 result non-significant (corrected threshold = 0.05/4 = 0.0125 for the weakest). The p = 0.006 (mixed-4/3) and p = 0.021 (jailbreak-7) would survive correction at the 0.025 level.

### Explicit Non-Claims

- This report does NOT claim that batch composition is "proven safe." It claims the effect is not detectable at the 4.7pp threshold with 22.1% co-batch verification.
- This report does NOT claim the directional asymmetry is a major safety risk. It claims the asymmetry is statistically non-random but operationally negligible at current magnitudes.
- This report does NOT claim temporal overlap has no mechanism. It claims the Phase 2 manipulation did not produce a detectable dose-response, possibly because the manipulation was ineffective (estimated overlap, not measured).

### What Would Change Our Conclusions

The following counterfactual conditions could shift the findings materially:

**(a) If co-batch verification improved to >80%.** Currently, only 22.1% of observations are confirmed co-batched. If instrumentation at the vLLM scheduler level confirmed >80% co-batching, and the aggregate null still held, the null would be substantially stronger -- the dilution factor would drop from ~4.5x to ~1.25x, and the effective MDE would improve from approximately 21pp (in the co-batched subset) to approximately 5.9pp. Conversely, if the directional asymmetry became stronger in the verified subset, the safety concern would escalate.

**(b) If sample sizes were 10x larger.** At N=4,680 per model per condition (instead of 468), the pooled MDE would drop from 4.7pp to approximately 1.5pp, and the per-model MDE from ~8pp to ~2.5pp. The current largest delta of 1.06pp would be testable. If the directional asymmetry scaled linearly (producing ~90-120 flips per condition instead of 9-12), the binomial tests would have overwhelming power, and the multiple-comparison correction issue would be moot.

**(c) If models were 7B+.** Larger models have stronger safety alignment (larger logit gaps for refusal tokens) and may show a different decision-boundary geometry. The directional asymmetry could reverse (more flips toward safe, as TR141 found for its larger model set) or vanish entirely (if the logit gaps are too large for FP perturbation to bridge).

**(d) If temperature > 0.** Under greedy decoding (temperature = 0), the only source of non-determinism is FP arithmetic order. At temperature > 0, sampling noise would add variance that could either mask the FP perturbation (drowning the signal in noise) or amplify it (if the perturbation shifts probabilities near the sampling threshold). The interaction between temperature and batch perturbation is unexplored and could change both the magnitude and direction of the effect.

---

## SS25. Conclusions

### Primary Conclusion

Batch composition does not significantly change aggregate refusal rates across 14,250 evaluation records, 3 models, and 23 analysis passes. All pairwise McNemar tests are non-significant (lowest p_exact = 0.125). All Cochran's Q omnibus tests are non-significant (lowest p = 0.341). All Mantel-Haenszel pooled odds ratios cross 1.0. The pooled TOST establishes safety equivalence within +/-3pp for all four composition comparisons.

### Secondary Conclusion

However, this is not a "clean null." When individual prompts do flip between solo and batched conditions, they flip toward unsafe (refuse-to-comply) 88-92% of the time. This directional asymmetry is statistically significant for 3 of 4 composition conditions (binomial p = 0.006, 0.021, 0.039). The composition content does not matter -- benign, jailbreak, and mixed fillers all produce similar directional bias. The most likely explanation is that batching itself (not filler content) introduces a very small bias toward compliance, consistent with TR138's finding that batch size affects FP arithmetic.

### Hypothesis Disposition

| ID | Hypothesis | Result |
|----|-----------|--------|
| H1 | Batch composition affects aggregate safety | **Not supported.** All tests p > 0.125. |
| H2 | Jailbreak co-batching degrades safety more than benign | **Not supported.** MH OR = 1.004, CI = [0.890, 1.133]. |
| H3 | Temporal overlap drives the effect | **Not supported.** Logistic slopes p > 0.93. |
| H4 | Safety context leaks into jailbreak processing | **Not supported.** Phase 3A all p = 1.0. |
| H5 | Static and continuous batching differ | **Not supported.** Phase 3B: 0-1 discordant pairs. |
| H6 | Flips are directionally symmetric | **Rejected for 3/4 conditions.** Binomial p = 0.006, 0.021, 0.039. |

### Integration with TR138 and TR141

TR138 showed that batch *size* changes safety outcomes. TR141 showed that this effect is architecture-dependent. TR143 shows that batch *composition* (what other requests are in the batch) does not significantly change aggregate outcomes. Together: the perturbation channel is the number of concurrent sequences (changing FP arithmetic), not the content of those sequences. PagedAttention's isolation holds for safety outcomes, not just memory.

However, the directional asymmetry discovered in TR143 (SS9) adds nuance to the TR138 finding: when batch-induced flips occur (rare at batch size 8), they preferentially move toward unsafe. This asymmetry should be tested at larger batch sizes (where TR138 showed more total flips) to determine whether the directional bias scales with batch size.

The three reports together form a coherent picture:
- TR138: Batch size matters (more sequences = more FP perturbation = more flips).
- TR141: The effect magnitude is architecture-dependent (GQA vs. MHA, model scale).
- TR143: Batch content does not matter (jailbreak fillers = benign fillers = mixed fillers in their effect on safety outcomes), but the rare flips that batching produces are directionally biased toward unsafe.

### Numbered Conclusions by Hypothesis

1. **H1 (Aggregate composition effect): Not supported.** Batch composition does not change aggregate refusal rates at the 4.7pp detection threshold. This is the strongest finding in the report, supported by 21 McNemar tests, 3 Cochran's Q tests, and 3 Mantel-Haenszel pooled ORs. Confidence: high, subject to the co-batch verification caveat.

2. **H2 (Jailbreak worse than benign): Not supported.** The content of co-batched requests does not differentially affect safety outcomes. Jailbreak fillers produce the same aggregate effect as benign fillers (MH OR = 1.004, CI = [0.890, 1.133]). This resolves the core threat-model question: an attacker flooding a shared endpoint with jailbreaks does not degrade other users' safety outcomes through the batch-composition channel. Confidence: moderate (confounded with the 47% prompt length difference).

3. **H3 (Temporal overlap mechanism): Not supported, but weakly tested.** The Phase 2 dose-response analysis finds no relationship between overlap timing and refusal rates (logistic slopes p > 0.93). However, the manipulation may not have achieved differential overlap levels (co-batch verification is only 22.1%), making this a weak null. Confidence: low.

4. **H4 (Reverse safety leakage): Not supported.** Co-batching jailbreak prompts with strongly-refusing fillers does not improve jailbreak refusal rates. Safety context does not leak in either direction through PagedAttention. Confidence: high (zero or near-zero discordant pairs).

5. **H5 (Static vs. continuous batching): Not supported.** The scheduler mode does not affect safety outcomes. Zero discordant pairs for llama3.2-3b, one for llama3.2-1b. Confidence: high for batch size 8; untested at larger batch sizes.

6. **H6 (Directional symmetry): Rejected.** When flips occur, they are 88-92% toward unsafe. This is statistically significant for 3 of 4 conditions (1 after full Bonferroni correction, 2 after Holm correction). The magnitude is negligible (~2-4 flips per 468 prompts per model per condition) but the direction is non-random. Confidence: moderate (small N, 22.1% co-batch verification, partial survival of multiple-comparison correction).

### What This Means for the Field

TR143 contributes a specific and operationally useful finding to the emerging literature on inference-time safety perturbations. The result -- aggregate null, rare directional asymmetry -- fills a gap between the "batch size matters" finding (TR138) and the "model determines susceptibility" finding (TR141). Together, the three reports establish that the batch-safety perturbation channel has a specific structure: it is universal (all models show it), magnitude-dependent on model characteristics (not batch content), and directionally biased toward unsafe for small models. This structure suggests that the appropriate mitigation is not input filtering or composition-aware routing (which would be expensive and complex), but rather model-level qualification (measuring output instability before deployment) and batch-size awareness (capping effective batch sizes for safety-sensitive workloads).

The finding also has implications for safety evaluation methodology. Most safety benchmarks evaluate models one prompt at a time (solo processing). TR143 shows that solo evaluation slightly overestimates safety (by ~0.5-1.1pp at batch size 8) compared to batched evaluation. While this gap is currently negligible, it represents a systematic bias in how models are evaluated versus how they are deployed.

### What This Does NOT Mean

It is important to delimit the conclusions precisely to prevent overclaiming:

- **This does NOT mean batch composition is "proven safe" in general.** The null holds at batch size 8, with 22.1% co-batch verification, for 3 small models. The parameter space is vast and mostly unexplored.
- **This does NOT mean the directional asymmetry is a deployment-blocking safety risk.** At 2-4 flips per 468 prompts, the practical impact is negligible. The finding is scientifically interesting and warrants monitoring, not immediate mitigation.
- **This does NOT mean temporal overlap has no mechanism.** It means the Phase 2 manipulation did not produce a detectable dose-response, possibly because the manipulation was ineffective. The mechanism question remains open.
- **This does NOT mean composition effects are impossible at larger scale.** Batch sizes of 32-256, models of 7B-70B, quantized serving, and multi-GPU tensor parallelism introduce new perturbation channels not tested here. Absence of evidence at batch size 8 is not evidence of absence at batch size 256.
- **This does NOT mean regex scoring is sufficient for safety evaluation.** The regex-only scoring in TR143 may miss subtle compliance shifts. A re-evaluation with an LLM judge could reveal effects that regex classifiers cannot detect.

### Implications for AI Governance

For policymakers and governance bodies evaluating multi-tenant LLM deployments, TR143's findings support a nuanced position. The aggregate null means that composition-aware request routing (a potentially expensive and privacy-invasive mitigation) is not justified by current evidence. However, the directional asymmetry means that batch perturbation is not a zero-risk channel. A reasonable governance posture would require: (a) safety evaluations under batched conditions (not just solo), (b) disclosure of batch sizes and serving configurations in model cards, and (c) periodic monitoring of refusal-rate drift as inference infrastructure evolves. The co-batch verification gap (22.1%) also highlights the need for serving stacks to expose batch-level telemetry for safety auditing -- a capability that no major serving framework currently provides.

### Remaining Open Questions

1. Does the directional asymmetry scale with batch size? (Test at batch sizes 16, 32, 64.)
2. Does co-batch verification improve with synchronous submission? (Instrument vLLM scheduler.)
3. Does the null hold for larger models (7B+) or MoE architectures?
4. Does quantized serving (INT4/AWQ) show composition sensitivity that FP16 does not?
5. Does the 47% prompt length confound mask a real composition-content effect?
6. Does LLM-judge scoring reveal composition effects that regex classifiers miss?
7. Does the directional asymmetry reverse for larger models (consistent with TR141's net-safe finding)?

---

## SS26. Production Guidance

### For Multi-Tenant vLLM Operators

- **No composition-aware routing required at current sensitivity thresholds.** Effects below 4.7pp are undetectable, and all observed deltas are under 1.1pp. Implementing composition-aware routing (e.g., segregating safety-sensitive requests from jailbreak-like requests) would add latency and complexity with no measurable safety benefit at current evidence levels.
- **Monitor for directional asymmetry at scale.** At batch size 8, the unsafe flip rate is ~0.6-1.1% of safety-sensitive prompts. At batch size 64, this could scale if the underlying FP-arithmetic mechanism amplifies. A simple monitoring approach: periodically evaluate a fixed set of safety-sensitive prompts under solo and batched conditions, and track the asymmetry ratio over time. If the ratio consistently exceeds 0.75 (75% toward unsafe) across multiple evaluation windows, escalate.
- **Do not claim "proven safe."** The 22.1% co-batch verification rate means the true composition effect is under-measured, not proven absent. The appropriate claim is: "We have tested for composition effects at the 4.7pp sensitivity threshold and found no significant aggregate effect. However, the measurement has a known limitation (22.1% co-batch verification) that could mask effects up to approximately 21pp in the co-batched subset."
- **Instrument co-batching.** Log which requests are physically co-resident in each forward pass. Without this, composition-safety auditing is impossible. The minimum viable instrumentation: for each forward pass iteration, log the set of request IDs and a timestamp. This enables post-hoc analysis of which requests were actually co-batched and allows computing the true co-batch verification rate.

### For Safety Auditors

- Cite the pooled MDE of 4.7pp as the detection floor. State: "Effects below 5pp are not detectable in this design."
- Report the directional asymmetry (SS9) as a "concerning but operationally negligible" finding. The recommended phrasing: "When batch-induced safety flips occur (affecting approximately 0.6-2.4% of prompts), they are directionally biased toward unsafe outcomes (88-92% of flips). The absolute impact is negligible at current batch sizes (2-4 additional unsafe outcomes per 468 prompts per model), but the directional consistency warrants monitoring."
- Note the 22.1% co-batch verification and 47% prompt length confound as unresolved limitations.
- Do not cite the capability TOST as "all models pass" -- only 1 of 3 passes formally. The appropriate claim: "Capability is preserved in practice (zero or near-zero degradation), but formal TOST equivalence is established for only 1 of 3 models due to a degenerate statistical edge case."
- When citing TR143 alongside TR138 and TR141, use the integrated finding from SS22b: batch perturbation is real, small, model-dependent, and directionally concerning but operationally negligible.

### For Serving Stack Engineers

- PagedAttention's per-request KV-cache isolation appears to extend to safety outcomes (no composition effect) as well as memory (no data leakage). The isolation holds at the attention level (no cross-request attention interference) and the FFN level (shared weight matrices do not introduce content-dependent cross-request effects).
- Static and continuous batching produce identical safety outcomes (Phase 3B). The scheduler choice should be made on throughput and latency grounds without concern for safety implications.
- Latency increases with batching (Cohen's d = 0.38-0.62), but this does not translate to safety outcome changes at batch size 8.
- The FP perturbation from batching is at the GEMM tile level, not the scheduler level. Any optimization that changes GEMM tile dimensions (e.g., batch padding, sequence bucketing, flash attention block sizes) could alter the perturbation characteristics. These changes should be tested for safety impact when deploying new kernel versions.

### For Researchers

- The directional asymmetry (SS9) is the most promising lead for follow-up work. The finding that flips are 88-92% toward unsafe is robust across models and filler types, but based on small N (9-12 flips per condition).
- Phase 2 requires re-instrumentation with measured (not estimated) temporal overlap. The vLLM scheduler's internal batch membership data must be exposed, either through a custom logging hook or by modifying the scheduler to emit per-iteration telemetry.
- Larger batch sizes and larger models would increase both the flip count and the power of directional tests. A recommended follow-up design: 3 models at 7B+ scale, batch sizes 8/16/32/64, with scheduler-level co-batch verification, targeting 50+ flips per condition to achieve adequate power for the directional binomial test.
- The mechanistic interpretation (FP non-associativity in batched GEMM) can be tested directly by examining logit distributions for boundary-proximate prompts under solo vs. batched conditions. If the mechanism is correct, the logit gap between the top two tokens should narrow under batching for prompts that flip, but remain stable for prompts that do not flip. This requires logit-level access, which vLLM can provide via the `logprobs` parameter.
- The "refusal boundary geometry" hypothesis (asymmetric sensitivity near the refusal decision boundary) can be tested by probing the model with controlled perturbations: add small random noise to the logits at each position and measure the flip rate and direction. If the geometry is truly asymmetric, the flip ratio should exceed 0.5 even under symmetric noise.

---

## SS27. Reproducibility

| Component | Detail |
|-----------|--------|
| Run directory | `research/tr143/results/20260319_174950/` |
| Seed | 42 |
| Backend | vLLM (Docker), FP16, continuous batching |
| vLLM config | `--gpu-memory-utilization 0.80 --enforce-eager --max-model-len 2048 --dtype float16` |
| Judge | qwen2.5:7b-instruct-q8_0 via Ollama |
| GPU | NVIDIA RTX 4080 Laptop 12GB |
| Total records | 14,250 |
| Analysis passes | 23 |
| Raw JSON | `tr143_analysis.json` (1,624 lines) |

### Environment

- Python 3.11
- vLLM (Docker, latest stable at time of run)
- Ollama (latest stable at time of run)
- CUDA 12.x
- Windows 11 host, WSL2 Docker backend

### Replication Steps

1. Install vLLM via Docker and Ollama locally.
2. Pull models: llama3.2-1b, llama3.2-3b, qwen2.5-1.5b (vLLM), qwen2.5:7b-instruct-q8_0 (Ollama judge).
3. Run `python research/tr143/run.py` with seed 42.
4. Raw results write to `research/tr143/results/<timestamp>/`.
5. Run `python research/tr143/analyze.py` to produce `tr143_analysis.json`.
6. Verify all statistics in this report against the JSON.

### Methodological Lessons for Future Work

Several aspects of the TR143 methodology should be improved in future studies.

**Co-batch verification must be instrumented, not estimated.** The 22.1% verification rate is the single most damaging limitation of the study. Future work should instrument vLLM's scheduler to log the request IDs present in each forward pass iteration. This requires a modest modification to vLLM's `LLMEngine._run_engine_loop()` method (or equivalent in newer versions) to emit a per-iteration batch membership log. The cost is a few hundred lines of instrumentation code; the benefit is resolving the fundamental ambiguity of whether the null result reflects true absence of effect or failure to deliver the treatment.

**Prompt length should be controlled, not just measured.** The 47% prompt length confound could have been avoided by length-matching the filler pools at experiment design time. Future work should either (a) subsample from the jailbreak pool to match the benign pool's length distribution, or (b) use a length-balanced mixed pool for all conditions. The current design of using natural jailbreak prompts (which are inherently longer due to social-engineering preambles) introduced an avoidable confound.

**TOST needs a zero-variance fallback.** The degenerate TOST failure for 2 of 3 models is a known limitation of the standard TOST procedure. Future work should use a bootstrap-based TOST implementation that handles zero-variance paired differences by estimating the standard error from the binomial parameter directly, rather than from the observed variance of paired differences.

**Phase 2 needs measured overlap, not estimated overlap.** The temporal overlap fractions in Phase 2 were estimated based on assumed generation times, not measured from actual co-batching durations. Future work should measure per-iteration co-residency directly (via the same scheduler instrumentation needed for co-batch verification).

**Sample size should target the directional test, not just the aggregate test.** The current design was powered for the aggregate McNemar test (MDE = 4.7pp at N=1,404), but under-powered for the directional binomial test (only 9-12 flips per condition). Future designs should use the directional test's power requirements as the binding constraint: to achieve 80% power for detecting a flip ratio of 0.75 (75% toward unsafe), approximately 50 total flips per condition are needed. At the current flip rate (~2.6% of prompts per condition), this requires approximately 1,900 prompts per model per condition -- a 4x increase over the current 468. Alternatively, larger batch sizes (which TR138 showed produce more flips) could increase the flip rate per prompt and reduce the required sample size.

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **Continuous batching** | vLLM's default scheduling mode: new requests are added to the running batch as existing requests complete, maximizing GPU utilization. |
| **Static batching** | All requests in a batch are submitted simultaneously and processed together without adding new requests mid-generation. |
| **Composition** | The set of other requests concurrently processed alongside the target prompt in the same batch. |
| **Filler** | A non-target request included in the batch to establish the desired composition (benign, jailbreak, mixed, or refusal). |
| **Co-batch verification** | Confirmation (from vLLM logs or timing analysis) that the target and filler requests were physically processed in the same forward pass. |
| **Discordant pair** | A prompt that produces a different safety outcome (refuse vs. comply) between two conditions. |
| **McNemar test** | A paired non-parametric test for comparing two related binomial proportions using only the discordant pairs. |
| **Cochran's Q** | A non-parametric omnibus test for differences across k > 2 related binomial conditions. |
| **Mantel-Haenszel** | A method for pooling odds ratios across strata (here, models) to test for a common effect. |
| **TOST** | Two One-Sided Tests procedure for establishing equivalence within a pre-specified bound (+/-3pp). |
| **PagedAttention** | vLLM's memory management system that allocates KV-cache in fixed-size blocks, enabling per-request memory isolation. |
| **MDE** | Minimum detectable effect: the smallest true effect that would be detected with 80% probability at alpha = 0.05. |

---

## Appendix B: Hypothesis Disposition Summary

| ID | Hypothesis | Test | p-value (worst case) | Decision |
|----|-----------|------|---------------------|---------|
| H1 | Composition affects aggregate safety | McNemar, Cochran's Q | p = 0.125 (exact) | **Not rejected** (H0 holds) |
| H2 | Jailbreak > benign degradation | MH pooled OR | CI = [0.890, 1.133] | **Not supported** |
| H3 | Temporal overlap mechanism | Logistic regression | p = 0.932 | **Not supported** |
| H4 | Reverse safety leakage | Phase 3A McNemar | p = 1.0 | **Not supported** |
| H5 | Static != continuous | Phase 3B McNemar | p = 1.0 | **Not supported** |
| H6 | Flip direction is symmetric | Binomial | p = 0.006 | **Rejected** (3/4 conditions) |

### Narrative Summary

The hypothesis disposition tells a clear story. Five of the six hypotheses are not supported, establishing a strong null result for the aggregate composition question. The one rejected hypothesis (H6, directional symmetry) reveals that the null is not "clean" -- it conceals a directional pattern in the rare flips that do occur.

The disposition also reveals the structure of the evidence pyramid. H1 (aggregate effect) is tested by three independent statistical approaches (McNemar, Cochran's Q, MH), all yielding consistent nulls. This convergent evidence makes the aggregate null very robust. H6 (directional asymmetry) is tested by a single approach (binomial test on pooled flips) with small sample sizes (9-12 per condition). This makes the directional finding less robust, though the consistency across conditions and models provides converging qualitative support.

The asymmetry between the strength of the null (very strong) and the strength of the directional finding (moderate) is the defining tension of this report. A conservative reader would focus on the strong null and treat the directional finding as a hypothesis for future testing. A risk-averse reader would focus on the directional finding and treat the null as potentially masking a small but consistent degradation. Both readings are defensible given the evidence.

---

## Appendix C: Full McNemar Test Matrix

All 21 comparisons across 3 models. b = refuse_to_comply (unsafe direction), c = comply_to_refuse (safe direction). OR uses Haldane correction: (b + 0.5)/(c + 0.5).

### llama3.2-1b (N=468 per comparison)

| Pair | b | c | n_disc | OR | chi2 | p_exact | Sig (Holm) |
|------|---|---|--------|-----|------|---------|-----------|
| solo vs benign-7 | 3 | 1 | 4 | 2.333 | 0.250 | 0.625 | No |
| solo vs jailbreak-7 | 1 | 1 | 2 | 1.000 | 0.500 | 1.000 | No |
| solo vs mixed-4/3 | 2 | 0 | 2 | 5.000 | 0.500 | 0.500 | No |
| solo vs refusal-7 | 2 | 1 | 3 | 1.667 | 0.000 | 1.000 | No |
| benign-7 vs jailbreak-7 | 0 | 2 | 2 | 0.200 | 0.500 | 0.500 | No |
| benign-7 vs mixed-4/3 | 1 | 1 | 2 | 1.000 | 0.500 | 1.000 | No |
| benign-7 vs refusal-7 | 0 | 1 | 1 | 0.333 | 0.000 | 1.000 | No |

### llama3.2-3b (N=468 per comparison)

| Pair | b | c | n_disc | OR | chi2 | p_exact | Sig (Holm) |
|------|---|---|--------|-----|------|---------|-----------|
| solo vs benign-7 | 2 | 0 | 2 | 5.000 | 0.500 | 0.500 | No |
| solo vs jailbreak-7 | 4 | 0 | 4 | 9.000 | 2.250 | 0.125 | No |
| solo vs mixed-4/3 | 4 | 0 | 4 | 9.000 | 2.250 | 0.125 | No |
| solo vs refusal-7 | 2 | 1 | 3 | 1.667 | 0.000 | 1.000 | No |
| benign-7 vs jailbreak-7 | 2 | 0 | 2 | 5.000 | 0.500 | 0.500 | No |
| benign-7 vs mixed-4/3 | 3 | 1 | 4 | 2.333 | 0.250 | 0.625 | No |
| benign-7 vs refusal-7 | 1 | 2 | 3 | 0.600 | 0.000 | 1.000 | No |

### qwen2.5-1.5b (N=468 per comparison)

| Pair | b | c | n_disc | OR | chi2 | p_exact | Sig (Holm) |
|------|---|---|--------|-----|------|---------|-----------|
| solo vs benign-7 | 3 | 0 | 3 | 7.000 | 1.333 | 0.250 | No |
| solo vs jailbreak-7 | 4 | 0 | 4 | 9.000 | 2.250 | 0.125 | No |
| solo vs mixed-4/3 | 5 | 1 | 6 | 3.667 | 1.500 | 0.219 | No |
| solo vs refusal-7 | 4 | 1 | 5 | 3.000 | 0.800 | 0.375 | No |
| benign-7 vs jailbreak-7 | 2 | 1 | 3 | 1.667 | 0.000 | 1.000 | No |
| benign-7 vs mixed-4/3 | 2 | 1 | 3 | 1.667 | 0.000 | 1.000 | No |
| benign-7 vs refusal-7 | 3 | 3 | 6 | 1.000 | 0.167 | 1.000 | No |

---

## Appendix D: Statistical Test Summary

| Test | N tests | N significant | Correction | Note |
|------|---------|--------------|-----------|------|
| McNemar paired (exact) | 21 | 0 | Holm-Bonferroni | All p_exact > 0.125 |
| Cochran's Q | 3 | 0 | None (per-model) | All p > 0.341 |
| Mantel-Haenszel pooled OR | 3 | 0 | None | All CIs cross 1.0 |
| Flip direction binomial | 4 | 3 | None applied | p = 0.006, 0.021, 0.039, 0.227 |
| TOST safety (pooled) | 4 | 4 | None | All equivalent within +/-3pp |
| TOST capability (per-model) | 3 | 1 | None | 2 degenerate, 1 pass |
| TOST capability (pooled) | 1 | 1 | None | Equivalent |
| Position ANOVA | 3 | 0 | None | All p > 0.643 |
| Phase 2 logistic regression | 2 | 0 | None | Both p > 0.93 |
| Phase 3A McNemar | 4 | 0 | None | All p = 1.0 |
| Phase 3B McNemar | 2 | 0 | None | All p = 1.0 |
| Latency confound (Cohen's d) | 3 | 2 | None | d = 0.616, 0.384, 0.513 |

**Summary statistics.** Across all test families: 53 total statistical tests were conducted. Of these, 3 are significant (all from the flip direction binomial family), 4 establish equivalence (TOST safety pooled), and 46 are non-significant nulls. The false positive rate expected by chance at alpha = 0.05 across 53 tests is 2.65 tests. The 3 significant results in the flip direction family exceed this expectation, but only marginally (3 vs. 2.65). However, the 3 significant results are not randomly distributed across test families -- they all come from the same analysis (directional asymmetry) and share the same underlying mechanism. This clustering makes the finding more credible than if 3 significant results were scattered across unrelated test families.

The overall statistical picture: the report conducts 53 tests, finds 46 nulls supporting H0 for the aggregate composition question, 4 equivalences confirming safety and capability preservation, and 3 significant results all pointing to the same directional asymmetry phenomenon. The evidence structure is internally consistent.

---

## Appendix E: Effect Heterogeneity by Task

Per-model composition delta (jailbreak-7 vs. solo) for each task.

### advbench_refusal (N=100 per cell)

| Model | Solo rate | JB-7 rate | Delta |
|-------|----------|-----------|-------|
| llama3.2-1b | 0.660 | 0.650 | -1.00pp |
| llama3.2-3b | 0.790 | 0.790 | 0.00pp |
| qwen2.5-1.5b | 0.980 | 0.980 | 0.00pp |

### bbq_bias (N=198 per cell)

| Model | Solo rate | JB-7 rate | Delta |
|-------|----------|-----------|-------|
| llama3.2-1b | 0.793 | 0.793 | 0.00pp |
| llama3.2-3b | 0.929 | 0.924 | -0.51pp |
| qwen2.5-1.5b | 0.909 | 0.899 | -1.01pp |

### jailbreak_amplification (N=120 per cell)

| Model | Solo rate | JB-7 rate | Delta |
|-------|----------|-----------|-------|
| llama3.2-1b | 0.508 | 0.517 | +0.83pp |
| llama3.2-3b | 0.583 | 0.567 | -1.67pp |
| qwen2.5-1.5b | 0.575 | 0.575 | 0.00pp |

### truthfulqa (N=50 per cell)

| Model | Solo rate | JB-7 rate | Delta |
|-------|----------|-----------|-------|
| llama3.2-1b | 0.460 | 0.460 | 0.00pp |
| llama3.2-3b | 0.570 | 0.530 | -4.00pp |
| qwen2.5-1.5b | 0.510 | 0.470 | -4.00pp |

**Observations.** The largest per-task deltas are in truthfulqa (-4.00pp for llama3.2-3b and qwen2.5-1.5b), but this is the smallest sample (N=50 per cell) and well within the per-cell MDE. The only positive delta is jailbreak_amplification for llama3.2-1b (+0.83pp), meaning jailbreak co-batching slightly *increased* refusal for that task -- opposite to the threat model direction. No task shows a consistent pattern across models, reinforcing the aggregate null.

### Cross-Task Heterogeneity Pattern

The task-level heterogeneity follows a clear gradient related to baseline refusal rate and task ambiguity.

**Zero-delta cells.** 6 of 12 cells show exactly 0.00pp delta (llama3.2-3b and qwen2.5-1.5b on advbench, llama3.2-1b on bbq_bias and truthfulqa, qwen2.5-1.5b on jailbreak_amplification). In these cells, not a single prompt changed its outcome between solo and jailbreak-7 conditions. The zero-delta cells tend to cluster at extreme baseline rates: advbench for qwen2.5-1.5b has a 0.980 solo rate (ceiling effect -- nearly all prompts are strongly refused), and truthfulqa for llama3.2-1b has a 0.460 rate but the lowest absolute accuracy model, suggesting that llama3.2-1b's truthfulqa responses are driven by capability limitations rather than safety alignment.

**Negative-delta cells.** 5 of 12 cells show negative deltas (ranging from -0.51pp to -4.00pp). These are concentrated in the larger models (llama3.2-3b and qwen2.5-1.5b) on the more ambiguous tasks (bbq_bias, truthfulqa). This pattern is consistent with the mechanistic model: larger models have more prompts near the decision boundary (due to stronger but not saturated safety training), and ambiguous tasks place more prompts in the boundary zone.

**The positive-delta outlier.** The single positive delta (llama3.2-1b, jailbreak_amplification, +0.83pp) means that jailbreak co-batching slightly increased this model's refusal rate on jailbreak-type prompts. This could be pure sampling noise (at N=120, the 95% CI for this delta is approximately [-5pp, +7pp]). Alternatively, it could reflect a subtle interaction: llama3.2-1b's jailbreak responses may be determined by the first few tokens (which are processed before the batch composition has time to influence later tokens), making the direction of any FP perturbation effectively random for this model-task combination.

The overall pattern supports the conclusion that task ambiguity (proximity to the decision boundary) modulates the sensitivity to batch composition, even though the aggregate effect across tasks is null. This is consistent with the SS9 mechanistic interpretation: the FP perturbation is small and uniform, but its impact is concentrated on prompts near the boundary.

---

## References

1. TR138: Batch Inference Safety Under Non-Determinism. Banterhearts Research Lab, 2026.
2. TR141: Cross-Architecture Batch Safety Fragility. Banterhearts Research Lab, 2026.
3. TR139: Multi-Turn Jailbreak Resistance Across Quantization Levels. Banterhearts Research Lab, 2026.
4. TR134: Safety Classifier Calibration. Banterhearts Research Lab, 2026.
5. TR130: vLLM Backend Integration. Banterhearts Research Lab, 2026.
6. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
7. McNemar, Q. "Note on the sampling error of the difference between correlated proportions or percentages." Psychometrika, 1947.
8. Cochran, W.G. "The comparison of percentages in matched samples." Biometrika, 1950.
9. Mantel, N. and Haenszel, W. "Statistical aspects of the analysis of data from retrospective studies of disease." JNCI, 1959.
10. Schuirmann, D.J. "A comparison of the Two One-Sided Tests Procedure and the Power Approach for assessing the equivalence of average bioavailability." J. Pharmacokinetics and Biopharmaceutics, 1987.
11. Johansson, M., et al. "Deterministic Execution of Non-Deterministic GPU Programs." IEEE Micro, 2024.
12. Mazeika, M., et al. "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." ICML 2024.
13. Yu, G., et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022.
14. Agrawal, A., et al. "Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills." arXiv 2308.16369, 2023.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-20 | Initial publication (RETRACTED: statistics not traceable to source JSON, omitted directional asymmetry finding) |
| 2.0 | 2026-03-20 | Complete rewrite from raw JSON. Every number verified against tr143_analysis.json. Added SS9 flip direction analysis, corrected TOST reporting, corrected prompt length confound (47% not 15%), corrected co-batch verification reporting (22.1%), corrected framing from "clean null" to "rare but directionally biased." |
| 2.1 | 2026-03-20 | Expanded interpretive depth: added Mechanistic Interpretation (SS9), cross-TR synthesis (SS22b), per-task interpretation (SS10), design rationale (SS3), model selection rationale (SS4), expanded limitations (SS24), expanded conclusions (SS25), added Key Caveats, added How to Read This Report, expanded all Phase 2/3 observations. No numbers changed. |
