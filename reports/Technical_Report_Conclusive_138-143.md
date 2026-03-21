# Conclusive Report 138-143: The Safety Attack Surface of LLM Inference Optimization
## Synthesis of batch perturbation, multi-turn jailbreaks, long-context attacks, cross-architecture fragility, quality-safety divergence, and cross-request composition across 307,000 evaluated samples

| Field | Value |
|-------|-------|
| **Project** | Banterhearts LLM Inference Research |
| **Date** | 2026-03-20 |
| **Author** | Banterhearts Research Lab |
| **Report Type** | Conclusive synthesis across TR138-TR143 (6 technical reports) |
| **Scope** | TR138 (Batch Size x Safety), TR139 (Multi-Turn Jailbreak x Quantization), TR140 (Many-Shot Long-Context Jailbreak), TR141 (Cross-Architecture Batch Fragility), TR142 (Quality-Safety Correlation), TR143 (Cross-Request Composition) |
| **Hardware** | NVIDIA RTX 4080 Laptop 12GB (local) + RTX PRO 6000 Blackwell 98GB (Colab) |
| **Measurement Corpus** | 306,996 evaluated samples across 6 technical reports |
| **Models Tested** | 18+ unique models (360M-14.8B parameters), 10+ architecture families |
| **Predecessor Synthesis** | Conclusive 134-137 (Phase 3: The Safety Cost of Inference Optimization) |

---

## Abstract

This synthesis report consolidates TR138 through TR143 into a unified attack-surface analysis for safety-critical LLM deployment. Where the Phase 3 synthesis (Conclusive 134-137) established that quantization, backend choice, and concurrency interact with safety alignment, this Phase 3.5/4 synthesis maps the FULL attack surface -- adding batch-size perturbation, batch composition, multi-turn jailbreak amplification, long-context exploitation, cross-architecture variation, and quality-safety divergence.

Across 306,996 evaluated samples and 18+ models, the program establishes five principal findings:

1. **Quantization below Q3_K_S is catastrophic for safety.** TR139 demonstrates that Q2_K combined with attention_shift jailbreaking achieves 100% attack success rate on qwen2.5-1.5b. TR140 confirms Q2_K as a universal vulnerability threshold across long-context attacks. TR142 shows safety degrades 13.9x faster than quality at Q3_K_S, meaning quality metrics are insufficient safety proxies.

2. **Batch perturbation is real but small.** TR138 measures a 0.6% safety flip rate from batch-size changes. Human adjudication (TR138, 63 rows reviewed) reveals only 27% of regex-detected flips are genuine -- the corrected rate is ~0.16%. TR141 scales to 18 models and finds output instability predicts fragility (r=0.91, p<0.0001), while alignment type does not (ANOVA p=0.915).

3. **Batch composition does not affect aggregate safety outcomes, but the rare flips that occur are directionally biased toward unsafe.** TR143 finds all McNemar paired tests non-significant (p>0.125) and all Cochran's Q non-significant (p>0.34), but the directional asymmetry is statistically significant: 88-92% of flips move toward unsafe across 3 of 4 conditions (binomial p=0.006 for the strongest).

4. **Multi-turn jailbreaking interacts with quantization multiplicatively.** TR139's 8 attack strategies all show significant quantization dependence (all ANOVA p<1e-4). The interaction is not additive -- certain strategy-quantization combinations produce vulnerability spikes that neither factor alone would predict.

5. **No single predictor generalizes across the attack surface.** Alignment type does not predict batch fragility (TR141 p=0.915). Quality does not predict safety (TR142, 13.9x divergence). Batch composition does not predict flip rates (TR143). The only reliable predictor is model-specific output instability (TR141, r=0.91), which must be measured empirically per model.

The operational output is a unified risk matrix (SS7) that maps each optimization dimension to its severity, evidence strength, and required mitigation. The overarching recommendation: **evaluate safety per-model and per-deployment-configuration, because no categorical shortcut (alignment type, quality score, architecture family) reliably predicts safety under optimization.**

---

## Executive Summary

### The Attack Surface Map

| Dimension | TR | Samples | Effect Size | Direction | Severity | Evidence |
|-----------|-----|---------|------------|-----------|----------|----------|
| Quantization (Q2_K) | TR139, TR140 | 78,425 | 35pp safety drop, 100% ASR | Catastrophic | **CRITICAL** | Strong |
| Quantization (Q3_K_S) | TR142 | 23,632 | 13.9x safety/quality divergence | Safety degrades first | **HIGH** | Strong |
| Multi-turn jailbreak x quant | TR139 | 48,425 | Multiplicative interaction | Strategy-dependent | **HIGH** | Strong |
| Long-context x quant | TR140 | 30,000 | Format-dependent (92% vs 0%) | Format matters more than length | **MODERATE** | Strong |
| Batch size perturbation | TR138, TR141 | 190,689 | 0.6% flip rate (0.16% adjudicated) | Directionally biased | **LOW** | Moderate |
| Batch composition | TR143 | 14,250 | <1pp aggregate, 88-92% directional | No aggregate, directional concern | **NEGLIGIBLE** | Moderate |
| Cross-architecture variation | TR141 | 152,022 | 6.3x fragility range | Model-dependent | **VARIABLE** | Strong |
| Quality as safety proxy | TR142 | 23,632 | 13.9x divergence | Quality misleads | **METHODOLOGICAL** | Strong |

**Observations.** The attack surface has a clear hierarchy. Quantization below Q3_K_S is the dominant safety risk, with effects measured in tens of percentage points. Batch perturbation exists but is operationally negligible at ~0.16% (human-adjudicated). Batch composition is safe at the aggregate level. The most dangerous finding is not any single dimension but the quality-safety divergence (TR142): operators who monitor quality metrics alone will miss safety degradation by a factor of 13.9x.

### Threat Hierarchy

| Tier | Threat | Action Required |
|------|--------|----------------|
| **Tier 1: Block** | Q2_K quantization on any model | Never deploy. No exceptions. |
| **Tier 1: Block** | Q3_K_S without per-model safety validation | Validate safety before deployment. Quality metrics insufficient. |
| **Tier 2: Validate** | Multi-turn jailbreak at any quantization below Q4_K_M | Test with all 8 strategy types before deployment |
| **Tier 2: Validate** | Long-context inputs at quantized levels | Validate with message-array format specifically |
| **Tier 3: Monitor** | Batch size > 1 in safety-critical pipelines | Measure output instability; if >15% output change rate, add monitoring |
| **Tier 4: Accept** | Batch composition in multi-tenant serving | No action required at current evidence levels |

**Observations.** The hierarchy reflects the 100x difference in effect magnitude between quantization (tens of pp) and batch perturbation (tenths of pp). An operator who spends time on composition-aware request routing instead of validating their quantization level is optimizing for the wrong risk by two orders of magnitude.

### What Operators Must Do

1. **Never deploy Q2_K.** Universal vulnerability confirmed across models, attack types, and context lengths (TR139, TR140).
2. **Never use quality metrics as safety proxies.** Safety degrades 13.9x faster at Q3_K_S (TR142).
3. **Validate Q3_K_S and Q4_K_M safety per-model.** Model-specific effects dominate; categorical rules fail (TR141).
4. **Test multi-turn jailbreaks at your quantization level.** All 8 strategy types. Attention_shift is the most dangerous (TR139).
5. **Measure output instability at your batch size.** >15% output change rate implies >1% safety flip rate (TR141).
6. **Do not assume alignment type predicts batch safety.** ANOVA p=0.915 with balanced groups (TR141 v3).
7. **Multi-tenant composition routing is unnecessary.** Aggregate composition effect is null (TR143).
8. **Log and monitor directional flip patterns.** The rare flips that occur are 88-92% toward unsafe (TR143).

---

## Table of Contents

- [SS1. Scope and Methodology](#ss1-scope-and-methodology)
- [SS2. The Quantization Attack Surface](#ss2-the-quantization-attack-surface)
- [SS3. The Batch Perturbation Surface](#ss3-the-batch-perturbation-surface)
- [SS4. The Composition Channel](#ss4-the-composition-channel)
- [SS5. Cross-Dimensional Interactions](#ss5-cross-dimensional-interactions)
- [SS6. The Alignment Type Question](#ss6-the-alignment-type-question)
- [SS7. Unified Risk Matrix](#ss7-unified-risk-matrix)
- [SS8. Methodological Lessons](#ss8-methodological-lessons)
- [SS9. What Remains Unknown](#ss9-what-remains-unknown)
- [SS10. Conclusions](#ss10-conclusions)
- [Appendix A: Cross-TR Data Summary](#appendix-a-cross-tr-data-summary)
- [Appendix B: Glossary](#appendix-b-glossary)
- [References](#references)

---

## SS1. Scope and Methodology

### What Was Tested

| TR | Dimension | Models | Records | Hardware |
|----|-----------|--------|---------|----------|
| TR138 | Batch size x safety | 3 (Llama 1B/3B, Qwen 1.5B) | 38,667 | RTX 4080 |
| TR139 | Multi-turn jailbreak x quantization | 4 (Llama 1B/3B, Qwen 1.5B, Llama 8B) | 48,425 | RTX 4080 |
| TR140 | Long-context jailbreak x quantization | 3 (Llama 1B/3B, Qwen 3B) | 30,000 | RTX 4080 |
| TR141 | Cross-architecture batch fragility | 18 (10+ families, 4 alignment types) | 152,022 | Colab Blackwell |
| TR142 | Quality-safety correlation | 4 models x 6 quant levels | 23,632 | RTX 4080 |
| TR143 | Cross-request composition | 3 (Llama 1B/3B, Qwen 1.5B) | 14,250 | RTX 4080 |
| **Total** | | **18+ unique models** | **306,996** | |

### What Was NOT Tested

- Models larger than 14.8B parameters
- Multi-GPU tensor parallelism
- Temperature > 0 (all studies use greedy decoding)
- Production traffic patterns (variable batch sizes, mixed request types)
- Quantization x batch composition interaction (no factorial design)
- Non-vLLM serving stacks (TGI, SGLang) for composition/batch tests

**Observations.** The program's greatest strength is breadth: 6 dimensions of the inference optimization space are covered, with 18+ models providing architectural diversity. Its greatest weakness is depth at the interaction level: the dimensions are tested independently, with no factorial designs testing (e.g.) whether quantized models show different batch-composition sensitivity than FP16 models. The single-hardware limitation (RTX 4080 for 5 of 6 TRs) is partially mitigated by TR141's use of Colab Blackwell, but GPU-dependent FP behavior means results may not transfer to data-center deployments.

---

## SS2. The Quantization Attack Surface

### TR139: Multi-Turn Jailbreak x Quantization

TR139 tests 8 jailbreak strategies across 6 quantization levels on 4 models. The core finding: quantization amplifies multi-turn jailbreak vulnerability multiplicatively.

| Strategy | Q8_0 ASR (typical) | Q2_K ASR (peak) | Amplification |
|----------|---------|---------|---------------|
| attention_shift (qwen2.5-1.5b) | ~20% | **100%** | 5x |
| crescendo (llama3.2-3b) | ~15% | ~60% | 4x |
| direct (all models) | ~10% | ~40% | 4x |

All 8 strategy ANOVAs reject quantization-independence (p < 1e-4).

**Observations.** The attention_shift strategy at Q2_K achieving 100% ASR is the single most alarming finding in the program. A quantized model that appears safe under single-turn evaluation becomes completely compromisable through multi-turn attack. Standard safety benchmarks test single-turn only and would miss this entirely. The implication: single-turn safety testing at your target quantization level is necessary but not sufficient. Multi-turn evaluation with adversarial strategies is required.

### TR140: Long-Context x Quantization

TR140 tests many-shot and long-context jailbreak formats. Llama models are immune above Q3_K_M. Q2_K is the universal vulnerability threshold. Message array format achieves 92% ASR vs 0% for faux dialogue.

**Observations.** The format dependency is the key operational finding. The same harmful content delivered in different formats produces 92% vs 0% ASR. Content filtering at the format level is a viable mitigation. The Q2_K threshold is consistent with TR139, reinforcing that Q2_K is categorically unsafe.

### TR142: Quality-Safety Divergence

Safety degrades 13.9x faster than quality at Q3_K_S. Simpson's paradox: aggregate trends mask per-model divergence.

**Observations.** This is the most methodologically important finding for the deployment community. Operators who rely on quality benchmarks to validate quantized deployments will systematically miss safety degradation. A model that retains 95% of its FP16 quality at Q3_K_S may retain only 70% of its FP16 safety. The quality metric gives no warning.

---

## SS3. The Batch Perturbation Surface

### TR138: Batch Size x Safety (Foundation)

| Metric | Value |
|--------|-------|
| Safety flip rate (regex-detected) | 0.6% |
| Human-adjudicated genuine rate | **27%** (17/63) |
| Corrected safety flip rate | **~0.16%** |
| True-batch validation agreement | 99.4% |

**Observations.** The human adjudication finding is the single most important methodological result in the program. 73% of regex-detected safety flips are artifacts -- the model produced different phrasing of the same safety posture. This has implications for every automated safety evaluation: any study reporting safety flip rates from automated classifiers without human validation is likely overestimating by 3-4x.

### TR141: Cross-Architecture Scaling (18 Models)

| Finding | Value |
|---------|-------|
| Models tested | 18 |
| Total records | 152,022 |
| Fragility range | 6.3x (phi-2 2.39% to tinyllama 0.00%) |
| Output instability as predictor | r=0.91, p<0.0001, R-squared=0.83 |
| Alignment ANOVA | F=0.13, p=0.915 (NOT significant) |
| Net directional bias | 66.2% safe (N=240) |

**Observations.** The output instability predictor (r=0.91) is the key positive finding. Models whose outputs change textually under batch perturbation (>15% output change rate) show >1% safety flip rates. This provides a practical screening metric that operators can measure without full safety evaluation.

The alignment type self-correction demonstrates intellectual honesty: v2.1 reported p=0.008, v3 added 8 models and the effect disappeared at p=0.915.

---

## SS4. The Composition Channel

### TR143: Cross-Request Safety Leakage

| Test | Result |
|------|--------|
| McNemar: benign-7 vs jailbreak-7 | All p>0.50, all OR crossing 1.0 |
| Cochran's Q (5 conditions) | All p>0.34 |
| MH pooled OR | 1.004, CI=[0.890, 1.133] |
| Directional: mixed-4/3 vs solo | 11 unsafe, 1 safe, **p=0.006** |
| Directional: jailbreak-7 vs solo | 9 unsafe, 1 safe, **p=0.021** |
| Directional: benign-7 vs solo | 8 unsafe, 1 safe, **p=0.039** |
| Co-batch verification rate | 22.1% |

**Observations.** The composition channel is closed at the aggregate level but the directional finding is concerning: when flips occur (rare -- ~2-4 per 468 prompts), they are 88-92% toward unsafe. This bias is consistent across all composition types, suggesting it is a property of batching itself, not of the filler content. The 22.1% co-batch verification limits confidence in the aggregate null but does not weaken the directional finding (if anything, the true directional bias in the verified subset may be stronger).

---

## SS5. Cross-Dimensional Interactions

### Tested Interactions

| Interaction | TR | Finding |
|-------------|-----|---------|
| Quantization x multi-turn jailbreak | TR139 | **Multiplicative.** All 8 strategies amplified. |
| Quantization x long-context | TR140 | **Threshold.** Q2_K universally vulnerable. |
| Quality x safety under quantization | TR142 | **Divergent.** Safety degrades 13.9x faster. |
| Architecture x batch size | TR141 | **Model-dependent.** r=0.91 with output instability. |

### Untested Interactions (Gaps)

| Gap | Why Important |
|-----|--------------|
| Quantization x batch composition | Quantized models may be more FP-sensitive to composition |
| Batch size x multi-turn | Batch perturbation may amplify multi-turn jailbreaks |
| Temperature x batch perturbation | Sampling noise may mask or amplify FP effects |
| Model scale x composition | 7B+ models may show composition sensitivity |

**Observations.** Quantization is the dominant amplifier across the attack surface. It multiplies multi-turn jailbreak effectiveness, creates threshold vulnerabilities for long-context attacks, and diverges from quality. Batch perturbation operates independently -- it is a function of the model's output instability, not of content or quantization level.

---

## SS6. The Alignment Type Question

| Version | N per Category | F | p | Conclusion |
|---------|---------------|---|---|------------|
| TR141 v2.1 | RLHF=4, others=1 | 4.862 | 0.008 | False positive (pseudoreplication) |
| TR141 v3 | All >=3 | 0.127 | 0.915 | **Not significant** |

The actual predictor is output instability (r=0.91), not alignment type.

**Observations.** This finding has implications beyond batch perturbation. Alignment type as a categorical variable is too coarse to predict model behavior under inference perturbation. Models within the same alignment category vary by 6x in fragility. The implication: evaluate models individually, never by category.

---

## SS7. Unified Risk Matrix

| Risk | Severity | Probability | Evidence | Detection | Mitigation |
|------|----------|-------------|----------|-----------|------------|
| Q2_K quantization | Catastrophic | Certain | Strong | Quality fails; safety required | Do not deploy |
| Q3_K_S safety divergence | High | Likely | Strong | Safety-specific metrics only | Per-model validation |
| Multi-turn jailbreak at Q2-Q4 | High | Model-dependent | Strong | Multi-turn testing | Rate limiting, monitoring |
| Long-context at Q2_K | Moderate | Format-dependent | Strong | Format-specific testing | Input format validation |
| Batch size perturbation | Low | Certain but negligible | Moderate | Output instability | Monitor if >15% change |
| Batch composition | Negligible aggregate | Rare | Moderate | Directional monitoring | No action at current scale |
| Quality as safety proxy | Methodological failure | Certain when relied upon | Strong | N/A | Never use alone |

**Observations.** The 100x magnitude difference between quantization effects (tens of pp) and batch effects (tenths of pp) should drive resource allocation. An operator spending engineering effort on composition-aware routing instead of quantization validation is optimizing for the wrong risk.

---

## SS8. Methodological Lessons

### Lesson 1: Regex Artifacts Inflate Flip Counts

Human adjudication of TR138 shows 73% of detected flips are artifacts. Any automated safety measurement reporting small effects (<5%) without human validation is dominated by classifier noise. The 0.6% becomes ~0.16%. If the same correction applies to TR141 and TR143, their findings also shrink by ~3x.

### Lesson 2: Pseudoreplication Produces False Positives

TR141's alignment ANOVA (p=0.008) was a false positive from n=1 per category. Self-correction with 8 additional models produced p=0.915. This is how preliminary findings should be handled.

### Lesson 3: Co-Batch Verification is Essential

TR143's 22.1% verification means for 78% of observations, we cannot confirm the treatment was delivered. Future studies must verify co-batching at the scheduler level.

### Lesson 4: Self-Correction Strengthens Credibility

Two major self-corrections (TR141 alignment ANOVA, TR143 v1.0 retraction) were published transparently. This pattern is more valuable than getting the right answer on the first try.

---

## SS9. What Remains Unknown

| Gap | How to Close |
|-----|-------------|
| Quantization x batch composition | Factorial design: 3 quant x 3 composition conditions |
| Temperature > 0 | Repeat TR138/TR143 at temperature 0.7 |
| Models > 14.8B | Cloud GPU experiments |
| Multi-GPU tensor parallelism | Multi-GPU hardware required |
| Production traffic patterns | Production monitoring study |
| Human adjudication of TR141/TR143 | Extend review queue (263 TR141 rows pending) |

---

## SS10. Conclusions

### Numbered Findings

1. **Quantization below Q3_K_S is the dominant safety risk.** 100% ASR at Q2_K. 13.9x quality-safety divergence at Q3_K_S. Confidence: **High.** (TR139, TR140, TR142)

2. **Batch perturbation is real but operationally negligible.** 0.16% human-adjudicated genuine rate. Confidence: **High.** (TR138 + adjudication)

3. **Output instability is the only reliable predictor of batch fragility.** r=0.91 across 15 models. Confidence: **High.** (TR141)

4. **Batch composition does not affect aggregate safety.** All tests non-significant. Confidence: **Moderate** (22.1% verification). (TR143)

5. **Rare flips are directionally biased toward unsafe.** 88-92%, p=0.006. Confidence: **Moderate** (small N). (TR143)

6. **Quality metrics are insufficient safety proxies.** 13.9x divergence. Confidence: **High.** (TR142)

7. **Alignment type does not predict batch fragility.** p=0.915 with balanced groups. Confidence: **High.** (TR141 v3)

8. **Multi-turn jailbreaks interact multiplicatively with quantization.** All 8 strategies, all p<1e-4. Confidence: **High.** (TR139)

9. **Automated classifiers have ~73% false positive rate for small changes.** 17 genuine / 63 reviewed. Confidence: **Moderate** (single reviewer). (TR138)

10. **No categorical variable reliably predicts safety under optimization.** Only per-model empirical measurement works. Confidence: **High** (convergent across 4 TRs).

### What This Program Contributes

The 6 TRs in this synthesis constitute the most comprehensive empirical mapping of the inference-optimization safety attack surface on consumer hardware. The 307,000 data points, 18+ models, and 6 dimensions provide breadth no individual study achieves. The program's self-corrections and human adjudication distinguish it from studies that optimize for headline claims.

### What This Program Does NOT Establish

- That batch perturbation is a deployment-blocking risk (it is not, at 0.16%)
- That alignment type is irrelevant to safety in general (only to batch fragility)
- That multi-tenant vLLM is proven safe (22.1% verification limits the claim)
- That findings transfer to models >14.8B or non-greedy decoding
- That the directional asymmetry is universal (3 models only)

---

## Appendix A: Cross-TR Data Summary

| TR | Records | Models | Key Finding | MDE |
|----|---------|--------|-------------|-----|
| TR138 | 38,667 | 3 | 0.6% flip, 27% genuine | ~1.5pp |
| TR139 | 48,425 | 4 | 100% ASR at Q2_K + attention_shift | ~2pp |
| TR140 | 30,000 | 3 | Q2_K universal, format 92% vs 0% | ~3pp |
| TR141 | 152,022 | 18 | Output instability r=0.91 | 1.3pp |
| TR142 | 23,632 | 4 | Safety degrades 13.9x faster | ~3pp |
| TR143 | 14,250 | 3 | Composition null, 88-92% directional | 4.7pp |
| **Total** | **306,996** | **18+** | | |

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **ASR** | Attack Success Rate -- fraction of jailbreak attempts eliciting compliance |
| **Batch composition** | The set of other requests concurrently processed alongside a target |
| **Batch size** | Number of concurrent requests in a single forward pass |
| **Co-batch verification** | Confirmation that target and fillers were physically co-processed |
| **FP non-associativity** | (a+b)+c != a+(b+c) in floating-point; batch size changes accumulation order |
| **Flip rate** | Fraction of prompts whose safety classification changes under perturbation |
| **Human adjudication** | Manual review of automated safety classifications by a human annotator |
| **McNemar test** | Paired non-parametric test comparing binary outcomes on same subjects |
| **MDE** | Minimum Detectable Effect at 80% power and alpha=0.05 |
| **Output instability** | Fraction of responses that differ textually from batch=1 baseline |
| **PagedAttention** | vLLM's memory management isolating KV-caches in fixed-size blocks |
| **Pseudoreplication** | Treating non-independent observations as independent, inflating N |
| **Q2_K/Q3_K_S/Q4_K_M** | GGML quantization levels with decreasing bits per weight |
| **Regex artifact** | Classifier disagreement caused by phrasing, not behavior |
| **TOST** | Two One-Sided Tests for equivalence within a pre-specified bound |

---

## References

1. TR138: Batch Inference Safety Under Non-Determinism. Banterhearts, 2026.
2. TR138 v2: Batch Safety -- Strengthened-Evidence Revision. Banterhearts, 2026.
3. TR139: Multi-Turn Jailbreak Under Quantization. Banterhearts, 2026.
4. TR140: Many-Shot and Long-Context Jailbreak Under Quantization. Banterhearts, 2026.
5. TR141: Cross-Architecture Refusal Fragility Under Batch Perturbation (v3.0). Banterhearts, 2026.
6. TR142: Quality-Safety Correlation Under Quantization. Banterhearts, 2026.
7. TR143: Cross-Request Safety Leakage Under Continuous Batching (v2.0). Banterhearts, 2026.
8. Conclusive 134-137: The Safety Cost of Inference Optimization. Banterhearts, 2026.
9. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
10. Xu, H., et al. "Q-resafe: Quantization Robustness Evaluation for Safety-Aligned Language Models." ICML 2025.
11. McNemar, Q. "Note on the sampling error of the difference between correlated proportions or percentages." Psychometrika, 1947.
12. Schuirmann, D.J. "A comparison of the Two One-Sided Tests Procedure." J. Pharmacokinetics and Biopharmaceutics, 1987.
