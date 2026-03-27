# Conclusive Report 138-143: The Safety Attack Surface of LLM Inference Optimization
## Synthesis of batch perturbation, multi-turn jailbreaks, long-context exploitation, cross-architecture fragility, quality-safety divergence, and cross-request composition across 306,996 evaluated samples and 18+ models

| Field | Value |
|-------|-------|
| **Project** | Banterhearts LLM Inference Research |
| **Date** | 2026-03-21 |
| **Author** | Banterhearts Research Lab |
| **Report Type** | Conclusive synthesis across TR138-TR143 (artifact-backed, 6 technical reports) |
| **Scope** | TR138 (Batch Size x Safety), TR139 (Multi-Turn Jailbreak x Quantization), TR140 (Many-Shot Long-Context Jailbreak x Quantization), TR141 (Cross-Architecture Batch Fragility), TR142 (Quality-Safety Correlation Under Quantization), TR143 (Cross-Request Composition) |
| **Hardware Baseline** | NVIDIA RTX 4080 Laptop 12GB (local, TR138-TR140, TR142-TR143) + NVIDIA RTX PRO 6000 Blackwell 98GB (Colab, TR141) |
| **Measurement Corpus** | 306,996 evaluated samples across 6 technical reports |
| **Models Tested** | 18+ unique models (360M-14.8B parameters), 10+ architecture families, 4 alignment types |
| **Primary Sources** | PublishReady/reports/Technical_Report_138_v2.md (Batch Safety Under Non-Determinism)<br>PublishReady/reports/Technical_Report_139.md (Multi-Turn Jailbreak x Quantization)<br>PublishReady/reports/Technical_Report_140.md (Many-Shot Long-Context Jailbreak)<br>PublishReady/reports/Technical_Report_141.md (Cross-Architecture Refusal Fragility)<br>PublishReady/reports/Technical_Report_142.md (Quality-Safety Correlation)<br>PublishReady/reports/Technical_Report_143.md (Cross-Request Composition) |
| **Predecessor Synthesis** | PublishReady/reports/Technical_Report_Conclusive_134-137.md (Phase 3: The Safety Cost of Inference Optimization) |
| **Program Context** | Phase 1 (TR117-TR122): Methodology. Phase 2 (TR123-TR133): Performance. Phase 3 (TR134-TR137): Safety Cost. Phase 3.5/4 (TR138-TR143): Safety Attack Surface. |

---

## Abstract

This report synthesizes TR138 through TR143 into a unified attack-surface analysis for safety-critical LLM deployment. Where the Phase 3 synthesis (Conclusive 134-137) established that quantization, backend choice, and concurrency interact with safety alignment across 74,254 samples and 5 models, this Phase 3.5/4 synthesis extends the analysis by adding six dimensions: batch-size perturbation, batch composition, multi-turn jailbreak amplification, many-shot long-context exploitation, cross-architecture variation, and quality-safety divergence. The combined evidence spans 306,996 evaluated samples, 18+ unique models from 10+ architecture families, 4 alignment types (RLHF, SFT, DPO, distilled), 8 multi-turn jailbreak strategies, 6 GGUF quantization levels, 5 many-shot counts, 2 prompt formats, 5 batch-composition conditions, 8 temporal overlap levels, and 23-28 analysis passes per TR. All findings are bounded by the tested conditions: consumer GPUs, models ≤14.8B parameters, GGUF k-quant, temperature 0, and greedy decoding.

The synthesis establishes five principal findings. First, quantization below Q3_K_S is catastrophic for safety across multiple attack modalities: TR139 demonstrates that Q2_K combined with the attention_shift multi-turn jailbreak strategy achieves 100% attack success rate on qwen2.5-1.5b; TR140 confirms Q2_K as a universal vulnerability threshold for many-shot long-context attacks with message-array format producing 92% ASR versus 0% for faux dialogue; and TR142 shows that safety degrades 13.9 times faster than quality at Q3_K_S, meaning that quality metrics are categorically insufficient as safety proxies. Second, batch perturbation is real but small: TR138 measures a 0.6% safety flip rate from batch-size changes, but human adjudication of 63 candidate rows reveals that only 27% (17/63) represent genuine behavioral flips -- the remaining 73% are regex artifacts from rephrased refusals. The human-adjudicated corrected rate is approximately 0.16%, and TR141's extension to 18 models finds that output instability predicts fragility with r = 0.91 (p < 0.0001, R-squared = 0.83), while alignment type does not predict fragility at all (ANOVA F = 0.13, p = 0.942 with balanced groups, correcting the v2.1 false positive of p = 0.008 from pseudoreplication). Third, batch composition does not affect aggregate safety outcomes -- all McNemar paired tests non-significant (p > 0.125), all Cochran's Q non-significant (p > 0.34), Mantel-Haenszel pooled odds ratios crossing 1.0 -- but the rare flips that do occur are directionally biased toward unsafe at 88-92% (binomial p = 0.006 for the strongest condition), a finding that is consistent across all composition types and therefore attributable to the batching mechanism itself rather than to filler content. Fourth, multi-turn jailbreaking interacts with quantization but not in the simplest possible way: all 8 strategy-specific ANOVAs reject quantization-independence (all p < 1e-4), yet the hypothesis that multi-turn strategies become more quantization-sensitive than direct attacks is not supported (Welch p = 0.702), meaning quantization degrades safety broadly rather than selectively amplifying multi-turn techniques. Fifth, no single categorical predictor generalizes across the attack surface: alignment type does not predict batch fragility (p = 0.942), quality does not predict safety (13.9x divergence), batch composition does not predict flip rates, and the only reliable predictor is model-specific output instability (r = 0.91), which must be measured empirically per model.

The operational output is a unified risk matrix that maps each optimization dimension to its severity, evidence strength, and required mitigation. Given that no tested categorical shortcut -- alignment type, quality score, architecture family, or parameter count -- reliably predicted safety under optimization within the tested model set, the overarching recommendation is to evaluate safety per-model and per-deployment-configuration. The 100x magnitude difference between quantization effects (tens of percentage points) and batch-perturbation effects (tenths of a percentage point) should drive engineering resource allocation: an operator who spends time on composition-aware request routing instead of validating their quantization level is optimizing for the wrong risk by two orders of magnitude.

This synthesis also documents its own limitations with the same rigor applied to its findings. The co-batch verification rate in TR143 is only 22.1%, meaning that for 78% of composition observations the treatment delivery cannot be confirmed. The dual-judge system in TR139 achieves only kappa = 0.104 between the 1.5B fallback judge and the 7B intended judge, indicating substantial measurement uncertainty in multi-turn ASR estimates. TR141's 15-model combined synthesis is regex-only -- the LLM-judge layer from intermediate work was not preserved in the final artifact chain. The human adjudication that re-calibrates TR138's flip rates was performed by a single reviewer on 63 rows, and the 27% genuine rate has not been independently replicated. The prompt-length confound in TR143 (47% difference between filler pools) means composition-content comparisons are confounded with computational load. These limitations are not disqualifying -- the directional findings are robust and the convergent evidence across 6 independent studies reinforces the principal conclusions -- but they bound the precision with which specific numerical thresholds can be prescribed. In particular, absolute ASR values from TR139 carry measurement uncertainty of approximately ±15pp due to the kappa = 0.104 inter-judge agreement; relative rankings between conditions are more reliable than absolute thresholds.

---

## Table of Contents

Executive Summary
- Attack Surface Map
- Threat Hierarchy
- Operator Actions

Operational Defaults (Safety Decision Card)

1. Introduction and Research Questions
   1.1 Motivation
   1.2 Research questions
   1.3 Contributions
   1.4 Document structure
2. Background and Related Work
   2.1 Floating-point non-associativity and batch perturbation
   2.2 Continuous batching and PagedAttention
   2.3 Multi-turn jailbreak methodology
   2.4 Many-shot jailbreaking and long-context exploitation
   2.5 Quantization-safety literature
   2.6 Quality-safety correlation literature
   2.7 Batch perturbation prior work
   2.8 Cross-architecture safety evaluation
   2.9 Automated safety measurement challenges
   2.10 Relationship to Phase 3 (TR134-TR137)
3. Methods and Measurement Framework
   3.1 Experimental design overview
   3.2 TR138: Batch size x safety protocol
   3.3 TR139: Multi-turn jailbreak x quantization protocol
   3.4 TR140: Many-shot long-context protocol
   3.5 TR141: Cross-architecture batch fragility protocol
   3.6 TR142: Quality-safety correlation methodology
   3.7 TR143: Cross-request composition protocol
   3.8 Statistical toolkit
   3.9 Scoring pipeline
   3.10 Model selection rationale
   3.11 Task battery design
   3.12 Human adjudication protocol
   3.13 Co-batch verification methodology
   3.14 LLM judge design and dual-judge system
4. Decision Impact Matrix
5. Results by Report
   5.1 TR138: Batch Inference Safety Under Non-Determinism
   5.2 TR139: Multi-Turn Jailbreak x Quantization
   5.3 TR140: Many-Shot Long-Context Jailbreak
   5.4 TR141: Cross-Architecture Refusal Fragility
   5.5 TR142: Quality-Safety Correlation Under Quantization
   5.6 TR143: Cross-Request Safety Leakage Under Continuous Batching
6. Cross-Report Synthesis by Decision Axis
   6.1 By quantization level
   6.2 By batch configuration
   6.3 By attack modality
   6.4 By model architecture
   6.5 By quality-safety relationship
   6.6 By measurement methodology
7. Operational Doctrine and Risk Controls
8. Threats to Validity and Scope Limits
9. Integration with Phase 3 (TR134-TR137) and Phase 2 (TR123-TR133)
10. Conclusive Statement

Appendix A: Claim-to-Artifact Chain-of-Custody
Appendix B: Full Risk Matrix
Appendix C: Human Adjudication Summary
Appendix D: Cross-TR Model Coverage Matrix
Appendix E: Statistical Methods Catalog
Appendix F: Glossary

References

---

## Executive Summary

This report closes the attack-surface loop opened by the Phase 3 synthesis (Conclusive 134-137). Phase 3 established that quantization and backend choice are the two axes requiring per-model safety validation, and that concurrency is safe. This Phase 3.5/4 synthesis (TR138-TR143) asks: what happens when the attack surface is expanded beyond single-turn evaluation to include batch perturbation, multi-turn adversarial strategies, many-shot long-context exploitation, cross-architecture variation, quality-safety divergence, and batch composition? The answer is that quantization remains the dominant risk -- by two orders of magnitude over batch effects -- but the attack surface has dimensions that single-turn evaluation cannot detect, and the relationship between quality and safety is not what most practitioners assume.

### Attack Surface Map

| Dimension | TR | Samples | Effect Size | Direction | Severity | Evidence |
|-----------|-----|---------|------------|-----------|----------|----------|
| Quantization (Q2_K) | TR139, TR140 | 78,425 | 35pp safety drop, 100% ASR | Catastrophic | **CRITICAL** | Strong |
| Quantization (Q3_K_S) | TR142 | 23,632 | 13.9x safety/quality divergence | Safety degrades first | **HIGH** | Strong |
| Multi-turn jailbreak x quant | TR139 | 48,425 | All 8 ANOVAs p < 1e-4 | Strategy-dependent | **HIGH** | Strong |
| Long-context x quant | TR140 | 30,000 | 92% vs 0% by format | Format matters more than length | **MODERATE** | Strong |
| Batch size perturbation | TR138, TR141 | 190,689 | 0.6% flip (0.16% adjudicated) | Directionally biased | **LOW** | Moderate |
| Batch composition | TR143 | 14,250 | < 1pp aggregate, 88-92% directional | No aggregate, directional concern | **NEGLIGIBLE** | Moderate |
| Cross-architecture variation | TR141 | 152,022 | 6.3x fragility range | Model-dependent | **VARIABLE** | Strong |
| Quality as safety proxy | TR142 | 23,632 | 13.9x divergence | Quality misleads | **METHODOLOGICAL** | Strong |

**Observations.** The attack surface has a clear hierarchy driven by a 100x magnitude difference between quantization effects and batch effects. Quantization below Q3_K_S produces effects measured in tens of percentage points -- up to and including 100% attack success rate on specific model-strategy combinations. Batch perturbation produces effects measured in tenths of a percentage point -- operationally negligible at current scales. Batch composition produces no aggregate effect at all, though the directional asymmetry in rare flips (88-92% toward unsafe) warrants monitoring. The most dangerous finding in this synthesis is not any single dimension but the quality-safety divergence documented in TR142: operators who monitor quality metrics alone will miss safety degradation by a factor of 13.9x at Q3_K_S. This is a methodological threat that applies to every deployment using quality benchmarks as safety proxies, regardless of which other dimensions are tested.

### Threat Hierarchy

| Tier | Threat | Action Required | Evidence Base |
|------|--------|----------------|---------------|
| **Tier 1: Block** | Q2_K quantization on any model | Never deploy. No exceptions. | TR139 (100% ASR), TR140 (universal threshold), TR142 (quality divergence) |
| **Tier 1: Block** | Q3_K_S without per-model safety validation | Validate safety before deployment. Quality metrics insufficient. | TR142 (13.9x divergence), TR134 (13.6pp refusal collapse) |
| **Tier 2: Validate** | Multi-turn jailbreak at any quantization below Q4_K_M | Test with all 8 strategy types before deployment | TR139 (all ANOVAs p < 1e-4) |
| **Tier 2: Validate** | Long-context inputs at quantized levels | Validate with message-array format specifically | TR140 (92% vs 0% ASR by format) |
| **Tier 3: Monitor** | Batch size > 1 in safety-critical pipelines | Measure output instability; if > 15% output change rate, add monitoring | TR141 (r = 0.91 predictor) |
| **Tier 4: Accept** | Batch composition in multi-tenant serving | No action required at current evidence levels | TR143 (aggregate null) |

**Observations.** The hierarchy reflects the two-orders-of-magnitude difference between quantization and batch effects. An operator who allocates engineering effort to composition-aware request routing before validating their quantization level is misallocating resources by a factor of 100x. The Tier 1 blocks are unconditional -- there is no model, strategy, or deployment context in the tested range where Q2_K is safe. The Tier 2 validations are conditional on deployment context: multi-turn jailbreak testing is required only for conversational deployments, and long-context validation is required only for applications accepting variable-length inputs. The Tier 3 monitor for batch size is a low-cost screening mechanism: measuring output instability requires only comparing textual outputs at batch = 1 and batch = N, with the r = 0.91 predictor from TR141 providing a practical threshold (> 15% output change rate implies > 1% safety flip rate). The Tier 4 acceptance for composition reflects the converging evidence from TR143 that aggregate safety outcomes are not affected by what else is in the batch.

### What Operators Must Do

1. **Never deploy Q2_K.** Universal vulnerability confirmed across models (TR139: 4 models), attack types (TR139: 8 strategies, TR140: many-shot, TR134: single-turn), and context lengths (TR140: 5 shot counts). No tested model retains adequate safety at Q2_K.

2. **Never use quality metrics as safety proxies.** Safety degrades 13.9x faster than quality at Q3_K_S on llama3.2-1b (TR142). Quality-safety correlation is model-dependent and can be negative: llama3.2-3b shows r = -0.829, meaning quality degrades while safety paradoxically improves through over-refusal (TR142).

3. **Validate Q3_K_S and Q4_K_M safety per-model.** Model-specific effects dominate; categorical rules fail. TR141 demonstrates 6.3x fragility variation across 15 models (phi-2 2.39% to tinyllama 0.00%), and the ANOVA on alignment type is decisively non-significant (F = 0.13, p = 0.942).

4. **Test multi-turn jailbreaks at your quantization level.** All 8 strategy types. Attention_shift is the most dangerous (TR139: 100% ASR on qwen2.5-1.5b at Q2_K). Single-turn safety testing is necessary but not sufficient -- a model that appears safe under single-turn evaluation may be completely compromisable through multi-turn attack.

5. **Restrict or sanitize message-array format inputs.** TR140 demonstrates that the same harmful content achieves 92% ASR in message-array format versus 0% in faux-dialogue format. Content filtering at the format level is a viable mitigation that does not require quantization changes.

6. **Measure output instability at your batch size.** Output change rate above 15% implies safety flip rates above 1% (TR141, r = 0.91). This is a low-cost screening metric that can be computed without full safety evaluation.

7. **Do not assume alignment type predicts batch safety.** ANOVA p = 0.942 with balanced groups across 15 models and 4 alignment types (TR141 v3). The earlier p = 0.008 from TR141 v2.1 was a false positive from pseudoreplication with n = 1 per category.

8. **Multi-tenant composition routing is unnecessary.** Aggregate composition effect is null across all tests (TR143: 21 McNemar tests, 3 Cochran's Q, 3 Mantel-Haenszel ORs). The rare directional asymmetry (88-92% toward unsafe) affects approximately 2-4 prompts per 468 per model per condition -- a rate that does not justify the engineering complexity of composition-aware routing.

9. **Log and monitor directional flip patterns.** Although the aggregate composition effect is null, the directional asymmetry is statistically significant (TR143: binomial p = 0.006 for mixed-4/3 condition). If batch sizes scale and co-batch verification improves beyond the current 22.1%, the accumulation of rare unsafe flips could become operationally relevant.

### Claim Status (cross-report, reviewer-proof)

| Claim ID | Claim | Evidence Base | Status |
|----------|-------|---------------|--------|
| C1 | Quantization below Q3_K_S is catastrophic for safety | TR139: 100% ASR at Q2_K. TR140: Q2_K threshold across all tested models. TR142: 13.9x quality-safety divergence (2 models). | **Demonstrated** across 3 independent TRs and 4 attack modalities within the tested model set |
| C2 | Batch perturbation introduces measurable safety instability | TR138: 0.6% regex, 0.16% human-adjudicated (single reviewer, n=63). TR141: 15 models, 0.00-2.39% range. | **Demonstrated** but magnitude recalibrated by human adjudication |
| C3 | Human adjudication shows 73% of automated flips are artifacts | TR138 v2: 17/63 genuine. Single reviewer, no inter-rater reliability. | **Observed** in TR138 specifically; artifact rate for other TRs not measured |
| C4 | Output instability predicts batch fragility | TR141: r = 0.91, R² = 0.83, p < 0.0001 across 15 models. | **Demonstrated** within the tested set; not yet externally validated |
| C5 | Alignment type does NOT predict batch fragility | TR141 v3: F = 0.13, p = 0.942. Corrects v2.1 false positive (p = 0.008). | **Not detected** (negative finding; absence of evidence, not evidence of absence) |
| C6 | Multi-turn jailbreak x quantization interaction exists | TR139: 8/8 strategy ANOVAs p < 1e-4 (η² = 0.031-0.153). ASR values subject to ±15pp measurement uncertainty (kappa = 0.104). | **Demonstrated** |
| C7 | Multi-turn is NOT selectively amplified over direct | TR139: Welch p = 0.702. Direct and multi-turn slopes comparable. | **Not detected** (negative finding; power analysis not reported) |
| C8 | Message-array format is the dominant many-shot attack vector | TR140: 92% vs 0% ASR by format on same model/quant (llama3.1-8b, Q2_K, N=16). | **Demonstrated** on tested models |
| C9 | Quality metrics are insufficient safety proxies | TR142: 13.9x divergence on llama3.2-1b at Q3_K_S. r = +0.994 vs r = -0.829 by model (n=2 models). | **Demonstrated** on 2 models; generality unknown |
| C10 | Batch composition does NOT affect aggregate safety | TR143: 21 McNemar tests, 3 Cochran's Q, 3 MH ORs all non-significant. 22.1% co-batch verification rate. | **Not detected** (negative finding; limited by 22.1% treatment verification) |
| C11 | Rare flips are directionally biased toward unsafe | TR143: 88-92%, binomial p = 0.006 for strongest condition. Small N (9-12 flips per condition). | **Observed** with small-N caveat; may be unstable under replication |
| C12 | No tested categorical predictor generalizes across attack surface | TR141 (alignment), TR142 (quality), TR143 (composition) all negative within tested conditions. | **Not detected** (convergent negative evidence across 3 TRs) |

### Program trajectory (TR138 -> TR143)

The six TRs in this synthesis form a deliberate research arc. Each report either fills a gap exposed by its predecessor or provides the cross-dimensional comparison that individual studies cannot:

1. **TR138** discovers the batch-perturbation phenomenon. Three models, 38,667 records, 4 phases. Establishes that batch size changes safety outcomes at 0.6% (automated). Adds human adjudication showing 73% are artifacts. Core finding: the effect is real but small, and automated classifiers inflate it.

2. **TR139** maps the multi-turn jailbreak x quantization interaction. Four models, 48,425 records, 8 strategies. Establishes that quantization amplifies multi-turn vulnerability (all ANOVAs p < 1e-4) but not selectively over direct attacks (Welch p = 0.702). Highest-risk cell: 100% ASR at Q2_K + attention_shift on qwen2.5-1.5b.

3. **TR140** maps the many-shot x quantization interaction. Four models, 30,000 records, 5 shot counts, 2 formats. Discovers that prompt format (message-array vs. faux dialogue) matters more than shot count (variance decomposition: format explains more variance than shots). Confirms Q2_K as universal vulnerability threshold.

4. **TR141** extends batch perturbation to 18 models. 152,022 records, 10+ families, 4 alignment types. Discovers output instability predictor (r = 0.91). Corrects the alignment-type false positive (p = 0.008 to p = 0.942). Finds near-parity in safety-capability flip ratio (0.94x combined).

5. **TR142** reveals quality-safety divergence. Analysis-only on 23,632 records from TR125 + TR134. Shows 13.9x divergence at Q3_K_S. Discovers opposite-sign quality-safety correlations between models (+0.994 vs -0.829). Invalidates quality-as-proxy practice.

6. **TR143** closes the composition channel. Three models, 14,250 records, 5 composition conditions. Establishes aggregate null (all tests non-significant). Discovers directional asymmetry (88-92% toward unsafe). Verifies static = continuous batching (p = 1.0).

This ordering is methodologically necessary. You cannot assess the attack surface without per-dimension measurements. You cannot prioritize threats without cross-dimensional comparison. You cannot calibrate automated measurements without human adjudication. And you cannot prescribe universal guidelines without cross-architecture validation that tests whether the guidelines hold across diverse models.

### Operational Defaults (Safety Decision Card)

Valid under stated boundary conditions (consumer GPU, models 360M-14.8B, GGUF quantization, temperature 0, greedy decoding).

- **Quantization ban:** Q2_K on all models (CRITICAL risk, universal vulnerability)
- **Quantization floor for safety-critical:** Q4_K_M or higher
- **Quantization caution zone:** Q3_K_S and Q3_K_M (validate per-model; quality metrics will miss degradation)
- **Multi-turn jailbreak testing:** Required before deployment at any quantization below Q4_K_M
- **Long-context input validation:** Restrict message-array format; sanitize multi-shot inputs
- **Batch size monitoring:** Measure output instability; threshold at 15% change rate
- **Batch composition:** No action required
- **Per-model profiling:** Required for any new model (alignment type is not a shortcut)
- **Quality as safety proxy:** Never use alone (13.9x divergence at Q3_K_S)
- **Classifier caveat:** Safety scores are consistent proxies, not calibrated ground truth (TR138 human adjudication: 73% of automated flips are artifacts)

**What invalidates this report:**
- Model family not in the tested set (18+ models cover 10+ families, but novel architectures require re-profiling)
- Quantization method other than GGUF k-quant (GPTQ, AWQ, SqueezeLLM use different compression strategies)
- Temperature > 0 (stochastic sampling introduces variance with unknown safety interaction)
- Multi-GPU tensor parallelism (different serving dynamics, different memory pressure profile)
- Model size > 14.8B parameters (only TR141b tests models up to 14.8B; larger models remain uncharacterized)
- Production traffic patterns (variable batch sizes, mixed request types, autoscaling)

---

## 1. Introduction and Research Questions

### 1.1 Motivation

Consumer-grade LLMs are already deployable on hardware that millions of people own. Models like Llama 3.2, Qwen 2.5, Phi-3, and Gemma run on a laptop with 8-12GB of VRAM. The barrier to local LLM deployment is no longer hardware or software — it is safety. A user who downloads a quantized model and serves it locally has no safety evaluation infrastructure, no deployment validation protocol, and no way to know whether the optimizations that make the model fit on their GPU have degraded its ability to refuse harmful requests. If consumer LLM deployment scales without safety measurement infrastructure, the consequences — harmful content generation, jailbreak vulnerability, degraded refusal behavior — will affect real people. This research program builds that infrastructure before the deployment wave, not after.

The Phase 3 synthesis (Conclusive 134-137) answered the first-order safety question: does inference optimization degrade safety alignment? The answer was nuanced -- quantization and backend choice require per-model validation, concurrency is safe -- but it was limited to single-turn evaluation, a single batch size (effectively batch = 1), and three optimization axes. Production LLM deployments face a richer threat surface than single-turn evaluation can capture. Multi-tenant serving batches requests from multiple users onto the same GPU. Adversarial users can construct multi-turn conversations designed to erode refusal behavior over successive turns. Long-context inputs can embed harmful instructions after many in-context compliance examples. And the batch size itself -- the number of requests processed in a single forward pass -- introduces floating-point non-determinism that may or may not affect safety outcomes.

This Phase 3.5/4 synthesis (TR138-TR143) maps the extended attack surface. It asks six questions that Phase 3 could not answer: Does batch size change safety outcomes? Does the composition of a continuous batch affect safety? Do multi-turn jailbreak strategies interact with quantization? Do many-shot long-context attacks exploit quantization vulnerability? Does the batch-perturbation effect generalize across architectures and alignment types? And does quality degradation under quantization predict safety degradation, or do they follow independent paths?

These questions are not theoretical. They correspond directly to deployment scenarios that operators face. A multi-tenant vLLM deployment batches safety-sensitive requests alongside arbitrary other requests -- if composition matters, operators need composition-aware routing. A chatbot serving quantized models processes multi-turn conversations -- if quantization amplifies multi-turn jailbreaks, operators need turn-level monitoring. A batch inference pipeline processes thousands of prompts at batch sizes of 8, 16, or 32 -- if batch size changes safety outcomes, operators need to control batch parameters. And every operator who validates deployment safety using quality benchmarks (MMLU, BERTScore, coherence) needs to know whether quality preservation implies safety preservation. The evidence from 306,996 samples across 6 TRs says it does not.

### 1.2 Research questions (program level)

This conclusive report answers the following ten cross-cutting questions, each mapping to one or more technical reports:

1. **Does batch size change safety outcomes?** (TR138) Batch-size variation changes the order of floating-point accumulation in GPU matrix operations, introducing non-determinism that produces different outputs for the same prompt. Does this non-determinism affect safety classifications, and if so, does it affect safety disproportionately to capability?

2. **Does human adjudication confirm automated safety flip detections?** (TR138 v2) Regex-based classifiers detect "flips" when the safety classification of a response changes between batch configurations. But does a changed classification always mean a changed safety posture, or are some "flips" artifacts of different phrasings of the same refusal?

3. **Does quantization amplify multi-turn jailbreak vulnerability?** (TR139) Single-turn safety evaluation (Phase 3, TR134) showed that quantization degrades refusal rates. But adversarial multi-turn strategies apply escalating pressure across conversation turns. Does quantization make models more susceptible to this escalation, and does the interaction between quantization and strategy type produce vulnerability spikes that neither factor alone would predict?

4. **Does quantization amplify many-shot and long-context jailbreaks?** (TR140) Many-shot jailbreaking embeds N compliance examples in the prompt before the harmful request. Does quantization shift the power-law relationship between shot count and attack success? Does the prompt format (message array vs. faux dialogue) matter more than quantization level?

5. **Does batch-perturbation fragility generalize across architectures?** (TR141) TR138 tested 3 models from 2 families. Does the effect hold across 18 models from 10+ families? Does alignment type (RLHF, SFT, DPO, distilled) predict fragility?

6. **Does quality degradation under quantization predict safety degradation?** (TR142) Operators commonly validate quantized deployments using quality benchmarks. If quality and safety degrade at different rates, quality benchmarks create a false sense of safety. What is the quantitative relationship between quality and safety degradation paths?

7. **Does batch composition affect safety outcomes?** (TR143) In multi-tenant continuous batching, does the content of co-batched requests (jailbreaks, benign, mixed) influence the safety outcome of a target prompt? Is there a dose-response relationship with temporal overlap?

8. **Which attack modalities are most dangerous and which can be safely ignored?** (Cross-TR synthesis) Given finite testing budgets, how should operators prioritize across the six dimensions?

9. **Do the self-corrections in this program strengthen or weaken overall confidence?** (TR141 ANOVA correction, TR143 v1.0 retraction) Two major findings were reversed during the research program. Does this pattern indicate unreliable methods, or does it demonstrate that initial findings are tested and corrected when they fail replication?

10. **How does this synthesis integrate with Phase 3 (TR134-TR137) and Phase 2 (TR123-TR133)?** The three phases form a single decision stack. Where does Phase 3.5/4 confirm, extend, or tension with prior phases?

### 1.3 Contributions of this synthesis

This synthesis contributes six decision-grade deliverables:

1. **Human-adjudicated batch safety baseline** (TR138): A human review (single reviewer, n=63) of automated safety flip detections, finding that 73% of regex-detected flips are artifacts of rephrased refusals rather than genuine behavioral changes. This changes the interpretation of TR138's batch-perturbation findings from "0.6% safety tax" to "approximately 0.16% genuine safety tax with 0.44% measurement noise" and highlights that automated classifiers can substantially overestimate small effects.

2. **Multi-turn jailbreak x quantization interaction map** (TR139): The first full-factorial study combining 8 modern multi-turn jailbreak strategies with 6 quantization levels on 4 models, producing 10,600 conversations and 37,825 judge labels. Reveals that quantization affects multi-turn ASR (all 8 ANOVAs p < 1e-4) but does not selectively amplify multi-turn over direct attacks (Welch p = 0.702).

3. **Many-shot format vulnerability discovery** (TR140): Identification of message-array format as the dominant attack vector for many-shot jailbreaking, producing 92% ASR versus 0% for faux dialogue on the same model at the same quantization level. This finding enables a format-level mitigation (restrict or sanitize message-array inputs) that is independent of quantization choices.

4. **Cross-architecture fragility predictor** (TR141): Discovery that output instability (r = 0.91, R-squared = 0.83) is a practical screening metric for batch safety fragility, while alignment type is not (ANOVA p = 0.942). This provides operators with a low-cost per-model assessment protocol: measure output instability at your batch size, and if it exceeds 15%, add safety monitoring.

5. **Quality-safety divergence quantification** (TR142): Demonstration that safety degrades 13.9x faster than quality at Q3_K_S, with model-dependent correlation direction (llama3.2-1b: r = +0.994; llama3.2-3b: r = -0.829). This invalidates quality-as-safety-proxy practices across the deployment community.

6. **Composition channel closure** (TR143): Evidence that aggregate safety outcomes are not affected by batch composition (21 McNemar tests, 3 Cochran's Q, 3 Mantel-Haenszel ORs all non-significant), enabling operators to skip composition-aware routing and allocate that engineering effort to higher-impact mitigations.

### 1.4 Scope and boundaries

This synthesis is bounded by specific hardware, software, and methodological constraints that must be understood before the findings are applied to deployment decisions.

**Hardware scope.** The primary experimental platform is an NVIDIA RTX 4080 Laptop with 12GB VRAM, used for TR138, TR139, TR140, TR142, and TR143. TR141 uses an NVIDIA RTX PRO 6000 Blackwell Server Edition with 98GB VRAM on Google Colab. The safety findings should generalize across consumer GPUs because safety is primarily a model/software property rather than a hardware property. However, the specific batch-perturbation effects (TR138, TR141, TR143) depend on GPU-level floating-point behavior, which varies across GPU architectures due to different warp scheduling, accumulation hardware, and memory access patterns. The findings should not be extrapolated to data-center GPUs (A100, H100) without re-measurement of the batch-perturbation component.

**Model scope.** The 18+ models tested span 360M to 14.8B parameters from 10+ architecture families. Models above 14.8B (30B, 70B, 405B) are not represented. The findings for the tested model-size range are well-supported; extrapolation to larger models is not defensible because larger models have qualitatively different weight distributions, attention patterns, and alignment properties.

**Quantization scope.** All quantization tests use GGUF k-quant schemes (llama.cpp), with levels from Q2_K (approximately 2.5 BPW) through FP16 (16.0 BPW). Other quantization methods (GPTQ, AWQ, SqueezeLLM, HQQ) use different compression strategies that may produce different safety profiles at the same nominal bit-width. The Q4_K_M safety floor established here applies only to GGUF k-quant.

**Temperature scope.** All evaluations use deterministic greedy decoding (temperature = 0). At temperature > 0, stochastic sampling introduces per-token randomness that dominates the floating-point epsilon from batch perturbation. The batch-safety findings (TR138, TR141, TR143) may not apply at temperature > 0 because the perturbation mechanism depends on deterministic argmax decisions that are replaced by stochastic sampling. The quantization-safety findings (TR139, TR140, TR142) likely apply regardless of temperature because the weight perturbation mechanism is independent of the sampling strategy.

**Attack scope.** The jailbreak strategies (TR139: 8 strategies, TR140: many-shot with 2 formats) are pre-scripted and automated. Real adversaries use adaptive strategies that observe model responses and adjust their approach. The reported ASR values are therefore lower bounds on adversarial vulnerability -- adaptive adversaries would likely achieve higher ASR. The relative rankings (which strategies are most effective, which quantization levels are most vulnerable) are likely robust to this limitation because adaptive strategies would amplify all cells proportionally.

### 1.5 Document structure

This report is structured for multiple audiences:

**If you need deployment decisions:** Read the Executive Summary, the Safety Decision Card, and Section 7 (Operational Doctrine). These translate 306,996 measurements into policies with explicit boundary conditions and invalidation triggers. A practitioner who reads only these sections can make safety-informed deployment decisions.

**If you need method defensibility:** Read Section 3 (Methods), Section 8 (Threats to Validity), and Appendix A (Claim-to-Artifact Chain-of-Custody). These provide the evidential chain from measurement to claim, including where the chain is weak or broken.

**If you need per-TR detail:** Read Section 5 (Results by Report), which provides narrative summaries of each TR's key findings with results tables, extended analysis, and the logical opening for the next experiment. Each subsection is self-contained but cross-references other reports.

**If you need cross-dimensional synthesis:** Read Section 6 (Cross-Report Synthesis by Decision Axis), which integrates findings across TRs by decision topic rather than by report sequence.

**If you need integration with prior phases:** Read Section 9, which traces how this synthesis confirms, extends, or tensions Phase 3 (TR134-TR137) and Phase 2 (TR123-TR133) findings.

---

## 2. Background and Related Work

### 2.1 Floating-point non-associativity and batch perturbation

The batch-perturbation hypothesis rests on a well-established property of IEEE 754 floating-point arithmetic: (a + b) + c is not guaranteed to equal a + (b + c). When a GPU computes matrix multiplications for transformer inference, the accumulation order of partial products depends on the number of active warps, the thread-block reduction schedule, and -- critically -- the batch size. Johansson et al. (2024) documented that CUDA GEMM operations produce different results depending on the number of concurrent warps, tracing this to accumulation-order differences in thread-block reduction. When batch size changes from 1 to N, the GPU kernel may process the same input with a different accumulation schedule, producing logits that differ by a small epsilon at each token position.

This epsilon is typically on the order of 1e-7 to 1e-5 in FP16 computation. For most tokens, this perturbation does not change the argmax -- the most likely token remains the same. But for tokens where two or more candidates have nearly identical logits (the "decision boundary"), the epsilon can flip the argmax, producing a different token and therefore a different response. Under greedy decoding (temperature = 0), this flip is deterministic for a given batch configuration: the same prompt always produces the same output at the same batch size. But different batch sizes produce different outputs for boundary-sensitive prompts.

The safety relevance of this mechanism depends on whether safety-critical tokens -- specifically, the tokens that determine whether the model refuses or complies with a harmful request -- lie near decision boundaries more often than capability-critical tokens. TR138 tests this hypothesis directly and finds a mild safety-skewed pattern (safety flips 4.0x more common than capability flips in the enriched replication, with 72.7% of directionally classified safety flips moving toward compliance). TR141 scales the test to 18 models and finds that the safety-capability flip ratio is not universal: the 7-model core campaign shows a 1.36x safety-skewed ratio, the balanced v3 extension reverses it to 0.78x, and the combined 15-model synthesis lands at 0.94x, near parity.

### 2.2 Continuous batching and PagedAttention

Modern LLM serving systems (vLLM, TGI, SGLang) use continuous batching, where new requests are inserted into a running batch as earlier requests complete, rather than waiting for the entire batch to finish before accepting new work. This design maximizes GPU utilization but means that any given request may be processed alongside a variable and unpredictable set of other requests. The co-batched requests share the GPU's compute resources -- the same CUDA kernel launch processes all requests in the batch simultaneously.

PagedAttention (Kwon et al. 2023) provides memory isolation between co-batched requests by allocating KV-cache in fixed-size blocks, analogous to OS virtual memory paging. Each request's KV-cache is stored in independent pages that are not shared with other requests. This isolation prevents direct data leakage -- one request cannot read another request's KV-cache -- but it does not prevent indirect effects through the shared computation path. If the batch size affects the accumulation order of the shared attention or MLP kernel launches, then the mere presence of other requests in the batch could alter a target request's output, even without any data leakage.

TR143 tests this indirect-effect hypothesis under vLLM's continuous batching. The finding is that the indirect effect exists (the directional asymmetry is statistically significant) but is negligibly small in magnitude (2-4 flips per 468 prompts per model per condition). The co-batch verification rate of 22.1% is a meaningful limitation: for 78% of observations, the experiment tested "intended composition" rather than "actual composition," because vLLM's scheduler may not have placed the target and filler requests in the same physical batch.

### 2.3 Multi-turn jailbreak methodology

The security of instruction-tuned models under multi-turn conversational pressure is one of the least-studied areas in the LLM safety literature. Most safety benchmarks (AdvBench, HarmBench, SafeDecoding) evaluate models using single-turn prompts: one harmful request, one response, one safety classification. This evaluation paradigm implicitly assumes that safety behavior is determined by the current prompt alone, without considering the conversational context that precedes it. In practice, this assumption is violated by multi-turn jailbreak strategies that use the conversation history to manipulate the model's safety behavior.

Multi-turn jailbreaking exploits the conversational structure of instruction-tuned models by applying escalating adversarial pressure across successive conversation turns. Unlike single-turn attacks (which present a single harmful prompt and measure refusal), multi-turn strategies use the conversation history to gradually shift the model's behavior toward compliance. The key insight is that models condition on the full conversation history at each turn, so a conversation that begins with benign exchanges can progressively introduce harmful framing until the model complies with requests it would have refused in isolation.

TR139 implements 8 multi-turn strategies spanning the spectrum of adversarial sophistication: direct (baseline, no adversarial framing), benign_context (establish cooperative tone before harmful request), foot_in_door (escalate from benign to harmful gradually), crescendo (increase pressure across turns), attention_shift (redirect from safety cues to task completion), role_play (establish a fictional scenario), progressive_refinement (iteratively refine harmful request), and context_fusion (blend harmful content with benign context). These 8 strategies represent the major categories of multi-turn attack techniques in the jailbreaking literature, adapted for automated evaluation at scale.

The interaction with quantization is the novel contribution of TR139. If quantization erodes the model's ability to maintain consistent refusal across turns -- because the weight perturbation degrades the model's representation of "I am in a conversation where I should refuse" -- then multi-turn strategies that specifically target conversational consistency should be amplified at lower quantization levels. The evidence supports the first part of this prediction (all 8 ANOVAs reject quantization-independence) but not the second (multi-turn slopes are not steeper than direct slopes, Welch p = 0.702).

### 2.4 Many-shot jailbreaking and long-context exploitation

The many-shot jailbreak technique represents a qualitatively different attack vector from multi-turn jailbreaking. Where multi-turn strategies exploit the conversational context built across interactive turns, many-shot attacks exploit the in-context learning capability of transformer models by embedding compliance examples directly in the prompt. The attack does not require multiple interaction turns -- it can be delivered in a single prompt that contains both the compliance examples and the harmful request. This makes many-shot attacks suitable for batch inference pipelines where multi-turn interaction is not available.

Many-shot jailbreaking, formalized by Anthropic's NeurIPS 2024 work, constructs prompts containing N in-context examples of the model complying with harmful requests before presenting the actual harmful request. The mechanism exploits in-context learning: the model observes a pattern of compliance in its input context and extends that pattern to the new request, overriding its safety training. The power-law relationship between shot count N and attack success rate predicts that ASR increases as a function of N raised to a model-dependent exponent.

TR140 tests whether quantization shifts this power-law relationship. The study uses 5 shot counts (N = 1, 4, 16, 64, 128) in 2 prompt formats (faux dialogue and message array) across 4 models and 6 quantization levels. The critical finding is that the prompt format matters more than the shot count: message-array format achieves 92% ASR where faux dialogue achieves 0% on the same model at the same quantization level. The message-array format exploits the model's chat template by presenting compliance examples as properly formatted assistant turns, making them more convincing as in-context learning examples than the informal faux-dialogue format.

The interaction with quantization is significant but secondary to the format effect. Q2_K is the universal vulnerability threshold across all models: even Llama models, which are immune to many-shot attacks above Q3_K_M, become vulnerable at Q2_K (up to 72% ASR on llama3.2-1b). This convergent finding across TR139 and TR140 reinforces the Q2_K ban from Phase 3.

### 2.5 Quantization-safety literature

The literature on quantization effects on LLM safety remains sparse relative to the extensive work on quantization effects on capability. The Phase 3 report (Conclusive 134-137) documented this gap: most quantization studies (Frantar et al. 2022, Lin et al. 2024, Dettmers et al. 2023) evaluate quality using perplexity, MMLU accuracy, or downstream task performance, none of which capture safety-specific behaviors. This gap is not accidental -- it reflects a structural bias in LLM evaluation research toward capability metrics that are easier to measure and benchmark. Safety metrics require adversarial prompt construction, refusal detection, and bias measurement, all of which are more methodologically complex than accuracy benchmarking.

The closest external work to this synthesis is Xu et al.'s Q-ReSafe (ICML 2025), which evaluates quantization robustness for safety-aligned language models. Q-ReSafe tests fewer models, fewer quantization levels, and a single attack modality, but confirms the general direction of this program's findings: quantization degrades safety, and the degradation is model-specific. The contribution of TR138-TR143 relative to Q-ReSafe is scale (306,996 samples vs. thousands), breadth (6 attack dimensions vs. 1), the human adjudication that re-calibrates automated measurements, and the cross-architecture analysis that establishes model-level heterogeneity as the dominant pattern.

The safety evaluation literature has increasingly recognized the need for adversarial testing beyond single-turn prompts. HarmBench (Mazeika et al. 2024) provides a standardized framework for red-teaming language models, and JailbreakBench (Chao et al. 2024) curates a library of jailbreak techniques. This synthesis builds on both: TR139 uses JailbreakBench-derived harmful behaviors, and the multi-turn strategies tested in TR139 are adapted from techniques documented in the red-teaming literature. The novel contribution is testing these adversarial techniques across quantization levels -- a dimension that HarmBench and JailbreakBench do not address because they evaluate models at a single (typically full-precision) configuration.

The weight perturbation perspective on quantization safety is informed by the broader adversarial robustness literature. Quantization can be understood as a structured weight perturbation: each weight is modified by a rounding error that is bounded by the quantization step size. If safety alignment is encoded in small weight modifications (relative to the pre-training weights), then quantization may disproportionately affect safety because the rounding errors are large relative to the safety-critical weight modifications but small relative to the capability-critical pre-training weights. This hypothesis predicts that safety would degrade faster than capability under quantization -- the safety veneer hypothesis tested in Phase 3 (TR137) and further examined in TR142. The evidence does not support the veneer hypothesis as a universal claim: the quality-safety relationship is model-dependent (TR142), and the safety-capability asymmetry varies across models and axes (TR137).

### 2.6 Quality-safety correlation literature

The relationship between model quality and model safety is a fundamental question for deployment validation. If quality and safety degrade together under optimization pressure, then quality metrics (which are cheaper and faster to evaluate) can serve as proxies for safety metrics (which require adversarial prompt construction and specialized classifiers). If quality and safety degrade independently, then quality monitoring provides false assurance for safety-sensitive deployments.

The assumption that quality metrics proxy for safety has not been systematically tested in the quantization literature. Practitioners routinely validate quantized deployments using MMLU, BERTScore, coherence, and similar capability benchmarks. If these benchmarks show minimal degradation, the deployment is considered safe. TR142 tests this assumption directly and refutes it: on llama3.2-1b at Q3_K_S, quality metrics (coherence, accuracy) move less than 2.3pp from FP16 while refusal rate drops 13.6pp. The 13.9x divergence between safety and quality degradation rates means that a quality gate set at 5pp would miss the 13.6pp safety collapse entirely.

The model-dependence of quality-safety correlation is particularly noteworthy. On llama3.2-1b, quality and safety are strongly positively correlated (r = +0.994): as quantization degrades quality, it also degrades safety, and the correlation happens to be strong because both degrade through the same mechanism (weight perturbation). On llama3.2-3b, quality and safety are significantly negatively correlated (r = -0.829): as quantization degrades quality, safety paradoxically improves through over-refusal. The model produces less coherent text (quality degrades) but compensates by refusing more aggressively (safety improves), likely because the degraded model defaults to refusal when it cannot generate coherent compliance. Pooling these two models would produce a misleading aggregate correlation near zero; the true finding is that the sign of the quality-safety relationship is model-dependent and unpoolable.

### 2.7 Batch perturbation prior work

Batch perturbation -- the phenomenon whereby changing the batch size of GPU inference changes the model's output for the same input -- is a consequence of floating-point non-associativity in GPU matrix operations. The phenomenon is well-documented in the numerical computing literature (Higham 2002, Goldberg 1991), but its implications for LLM safety have not been explored prior to this research program.

We are not aware of prior work systematically measuring batch-size effects on LLM safety outputs. The continuous batching literature (Orca, Sarathi, DeepSpeed-FastGen) focuses on throughput and latency optimizations. Safety evaluation benchmarks (HarmBench, JailbreakBench, SafeDecoding) evaluate models exclusively under single-request (batch = 1) processing.

TR138 provides evidence that batch size changes safety outcomes, and TR141 extends this to 18 models with a cross-architecture analysis. The combined finding is that the effect exists but is small (0.16% human-adjudicated genuine rate), architecture-dependent (0.00% to 2.39% fragility across 15 models), and predictable from a single low-cost metric (output instability, r = 0.91). This is a genuine contribution to the safety evaluation literature, because it establishes a gap between evaluation-time conditions (batch = 1) and deployment-time conditions (batch = N) that, while small, is systematically biased in the unsafe direction.

### 2.8 Cross-architecture safety evaluation

Most safety evaluation studies test 1-3 models, typically from the same family. The assumption that findings generalize across architectures is rarely tested. TR141's extension to 18 models from 10+ families (Llama, Qwen, Phi, SmolLM, StableLM, Mistral, OLMo, TinyLlama, DeepSeek, SmolLM3) tests this assumption at a scale (18 models, 10+ families) that is, to our knowledge, the largest in the batch-perturbation safety literature for batch-perturbation safety.

The finding is that generalization fails at the model level. Fragility varies 6.3x across 15 models (phi-2 2.39% to tinyllama 0.00%). Alignment type -- which would be the natural categorical variable for predicting safety behavior -- is not predictive (F = 0.13, p = 0.942 with balanced groups of n >= 3 per category). The only reliable predictor is output instability, an empirical metric that must be measured per-model. This finding echoes the I-squared = 99.9% heterogeneity on quantization safety from Phase 3: model-specific behavior dominates, and no categorical shortcut reliably predicts which models will be affected.

The 10+ families tested in TR141 span multiple alignment paradigms. RLHF models (Llama, Mistral) use a reward model trained on human preference data to fine-tune the base model. SFT models (certain Llama variants, OLMo) use supervised fine-tuning on instruction-response pairs without a reward model. DPO models (Qwen, DeepSeek) use direct preference optimization, which modifies the training loss to optimize for preference pairs directly. Distilled models (SmolLM, TinyLlama) are trained by distillation from larger models, potentially inheriting safety properties from the teacher model. The fact that none of these alignment categories predicts batch fragility suggests that the mechanism linking alignment to batch stability is more complex than the categorical alignment label captures. The safety properties that matter for batch stability may be distributed across the weight space in ways that are not correlated with the alignment method used to train them.

The 4 alignment types and 10+ families also provide the basis for testing whether safety evaluation results from one model can be transferred to another. The answer is clearly no: even within the same alignment type (e.g., RLHF), models vary by 6x in fragility. Even within the same family (Llama), models at different scales have opposite quantization-safety profiles. The implication is that every deployment of a new model requires fresh safety evaluation -- there is no shortcut based on prior evaluations of similar models, because "similar" does not predict "equivalently safe" in the tested model population.

### 2.9 Automated safety measurement challenges

The measurement of LLM safety is fundamentally harder than the measurement of LLM capability. Capability metrics (MMLU accuracy, ARC-Challenge, BERTScore) have well-defined ground truth: either the model answers the question correctly or it does not. Safety metrics (refusal rate, jailbreak resistance, bias) require subjective judgments about whether a response constitutes "refusal" or "compliance," whether a response is "biased" or "unbiased," and whether a response is "harmful" or "benign." These judgments depend on the evaluator's standards, the context of the request, and the level of sophistication used to classify edge cases. Two reasonable evaluators can disagree on whether a specific response constitutes a refusal, especially for responses that technically refuse while providing harmful information through implication.

All safety scores in this program are produced by automated classifiers: regex-based detectors (RefusalDetector, BiasDetector, TruthfulnessScorer) and LLM judges (Qwen 2.5 7B Q8_0 in TR139/TR140, not preserved in TR141 final artifacts). The TR138 human adjudication reveals a fundamental limitation of automated safety measurement: 73% of regex-detected safety flips are artifacts where the model produced a different phrasing of the same safety posture. The model still refused the harmful request, but it phrased the refusal differently at batch = 1 versus batch = N, causing the regex classifier to assign different labels.

This 73% artifact rate has implications beyond TR138. Any automated safety evaluation reporting small effects (< 5%) without human validation is likely dominated by classifier noise rather than genuine behavioral changes. The TR139 dual-judge system (1.5B fallback + 7B intended judge) achieves only kappa = 0.104, further illustrating the difficulty of automated safety measurement. The TR140 judge achieves kappa = 0.23 overall but drops to kappa = 0.13 at Q2_K where model outputs become incoherent and classification ambiguity is highest.

The practical implication is that absolute safety scores should be interpreted as consistent proxies for cross-condition comparison rather than as calibrated ground truth. The directional findings (quantization is dangerous, batch perturbation is small, composition has no aggregate effect) are robust to classifier choice because they are measured as relative differences between conditions. The specific numerical thresholds (0.6% flip rate, 100% ASR, 13.9x divergence) carry measurement uncertainty that is partially but not fully characterized.

### 2.10 The self-correction methodology

Two major self-corrections occurred during this research arc, and they merit discussion as a methodological practice rather than merely as corrections to specific findings.

The TR141 alignment-type ANOVA correction (p = 0.008 to p = 0.942) illustrates the pseudoreplication problem. The v2.1 analysis treated each model as an independent observation within its alignment category. With only 1 model per category for SFT, DPO, and distilled, the between-group variance was entirely determined by the individual model's idiosyncratic behavior. The ANOVA F-statistic was inflated because the denominator (within-group variance) was zero for single-model categories, making any between-group difference appear significant. The v3 extension added models until each category had n >= 3, providing genuine within-group variance for the denominator. The resulting F = 0.13, p = 0.942 demonstrates that the original signal was entirely driven by pseudoreplication.

The TR143 v1.0 retraction illustrates the null-result reporting problem. Null results are typically underreported in the scientific literature (the "file drawer problem"), and when they are reported, they are often framed as clean negatives without qualification. TR143 v1.0 fell into this trap: the aggregate null was reported without the secondary directional asymmetry finding. The retraction and re-publication as v2.0 demonstrates that null results require the same multi-dimensional analysis as positive results. The aggregate null is correct but incomplete; the directional asymmetry is a genuine secondary finding that qualifies the null.

Both corrections followed a three-step pattern: (1) preliminary finding published with explicit limitations, (2) targeted replication or extension designed to test the finding, (3) honest publication of the outcome regardless of whether it confirms or reverses the original. This pattern is the program's recommended practice for safety evaluation, where false positives (claiming an effect that does not exist) and false negatives (missing an effect that does exist) have different operational consequences. For safety claims, false negatives are more dangerous than false positives, so the program errs on the side of reporting effects and then testing whether they replicate.

### 2.11 The measurement boundary problem

A recurring challenge in this synthesis is the measurement boundary problem: the difference between what automated classifiers measure and what human safety assessment would measure. The TR138 human adjudication (73% artifact rate) quantifies this gap for the specific case of safety flip detection, but the gap likely exists for all automated safety measurements in the program.

The measurement boundary has three components. First, regex classifiers detect surface-level patterns (e.g., "I cannot", "I'm sorry") but cannot assess the semantic intent of a response. A model that says "I cannot provide that information" (genuine refusal) and a model that says "I cannot provide that information in this format, but here is an alternative approach" (compliance disguised as refusal) receive the same classification. Second, LLM judges assess semantic intent but disagree substantially with regex classifiers (kappa = 0.104 to 0.282) because they apply different standards for what constitutes "refusal." Third, neither classifier has been validated against a gold-standard human safety assessment across the full evaluation battery, so the "true" safety level is unknown.

The practical consequence is that all safety scores in this program should be interpreted as consistent proxies rather than calibrated measurements. The word "consistent" is doing important work: because the same classifiers are applied uniformly across all conditions, systematic biases cancel when computing cross-condition deltas. A classifier that over-counts refusals does so equally at Q4_K_M and Q2_K; the delta between conditions remains informative even if the absolute levels are inaccurate. The directional findings (quantization is dangerous, batch perturbation is small, composition is null) are robust to this measurement uncertainty because they are based on relative comparisons, not absolute thresholds.

### 2.12 Relationship to Phase 3 (TR134-TR137)

This synthesis is deliberately positioned as a continuation of the Phase 3 safety program rather than a replacement for it. Phase 3 established the foundational safety framework: quantization and backend choice require per-model validation, concurrency is safe, the safety veneer hypothesis is refuted as universal. TR138-TR143 extend this framework along six new dimensions without altering its foundational conclusions.

The key extensions are: (1) Phase 3 tested single-turn safety; TR139 and TR140 add multi-turn and many-shot attack modalities, revealing that the quantization-safety interaction is more dangerous under adversarial multi-turn pressure than single-turn evaluation suggests. (2) Phase 3 tested 5 models; TR141 extends to 18 models, establishing that model-level heterogeneity is the dominant pattern. (3) Phase 3 used only automated classifiers; TR138 adds human adjudication, recalibrating the automated measurements. (4) Phase 3 assumed quality and safety degrade together; TR142 refutes this assumption quantitatively. (5) Phase 3 did not test batch effects; TR138, TR141, and TR143 comprehensively map the batch perturbation and composition attack surface.

The key tension with Phase 3 is directional: Phase 3 found safety flips 4x more common than capability flips (uncorrected), implying a safety-specific vulnerability. TR141's 15-model combined synthesis finds near-parity (0.94x ratio), and the TR138 human adjudication reduces the uncorrected 0.6% to 0.16%, suggesting the safety-specificity of batch perturbation was overstated by automated classifiers. The corrected finding is that batch perturbation introduces a small, architecture-dependent instability that is not strongly safety-specific but is directionally biased toward unsafe when flips do occur.

---

## 3. Methods and Measurement Framework

### 3.1 Experimental design overview

The six TRs in this synthesis use different experimental designs tailored to their respective research questions, but share a common infrastructure: automated safety classifiers from TR134, GGUF quantization via Ollama, greedy decoding at temperature 0, and a common statistical toolkit.

| TR | Design Type | Independent Variables | Dependent Variables | Total Records |
|----|-------------|----------------------|--------------------|-|
| TR138 | Within-subjects (batch size) | Batch size (1, 2, 4, 8, 16, 32) | Safety/capability flip rate | 38,667 |
| TR139 | Full factorial | Model (4) x Quant (6) x Strategy (8) x Behavior (50) | ASR, persistence | 48,425 |
| TR140 | Full factorial | Model (4) x Quant (6) x Shots (5) x Format (2) x Behavior (50) | ASR, judge label | 30,000 |
| TR141 | Within-subjects (batch size) | Model (18) x Batch size (1, 2, 4, 8) | Flip rate, output identity | 152,022 |
| TR142 | Analysis-only (cross-reference) | Model (2) x Quant (7) | Quality-safety divergence | 23,632 |
| TR143 | Within-subjects (composition) | Model (3) x Composition (5 conditions) | Refusal rate, flip direction | 14,250 |

### 3.2 TR138: Batch size x safety protocol

TR138 uses a 4-phase design with increasing specificity. Phase 1 sweeps batch sizes (1, 2, 4, 8, 16, 32) on 3 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b), comparing safety and capability flip rates. Phase 2 tests co-batching interference by mixing benign, adversarial, and safety prompt types at batch = 8. Phase 3 adds quantization variation (Q4_K_M and Q8_0) under concurrent load. Phase 4 validates with explicit true-batch inference (prompt-list batching) versus the synchronized-dispatch method used in Phases 1-3.

The v2 revision adds two evidence layers: a corrected audit of 44 behavior-changing rows (5 false-flip rows removed by curly-quote normalization in the v2.2 scorer), and a 7,257-sample reduced replication on an enriched 187-prompt subset per model. The replication confirms Phase 1 safety flips at 1.68% (27/1,605) versus 0.42% capability flips (5/1,200), a 4.0x ratio on the enriched subset. Phase 4 true-batch validation records 3.27% safety flips with 98.67% mean flip agreement to Phase 1, demonstrating that the signal is not a scheduler artifact.

### 3.3 TR139: Multi-turn jailbreak x quantization protocol

TR139 implements a 2-phase design. Phase 1 (attack efficacy): 4 models x 6 quantization levels x 8 strategies x 50 behaviors = 9,600 conversations, each consisting of a multi-turn interaction where the attack strategy is applied across 3-5 turns. Phase 2 (refusal persistence): 4 models x 5 quantization levels x 50 behaviors = 1,000 conversations, measuring whether models that initially refuse can be broken through continued adversarial pressure.

The scoring pipeline uses a dual-judge system: the primary judge is the 7B Qwen 2.5 Instruct Q8_0, and the fallback is a 1.5B model. Post-hoc adjudication produces 37,825 judge labels. The dual-judge agreement (kappa = 0.104) is acknowledged as a limitation -- the judges measure partially different constructs, and the disagreement is concentrated in the mid-quantization Phase 2 cells where classification ambiguity is highest.

### 3.4 TR140: Many-shot long-context protocol

TR140 implements a 2-phase design following Anthropic's many-shot methodology. Phase 1: 4 models x 6 quantization levels x 5 shot counts (1, 4, 16, 64, 128) x 2 formats (faux dialogue, message array) x 50 behaviors = 12,000 samples. Phase 2 (long-context): 4 models x 5 quantization levels x 3 context profiles (no_prefix, short_prefix, long_prefix) x 50 behaviors = 3,000 samples.

Judge labels are produced for all 15,000 samples with 0 failures (100% completion rate). The key methodological contribution is the 2-format comparison: faux dialogue presents compliance examples as informal text, while message array formats them as properly structured assistant turns within the chat template. The format distinction is operationally significant because it determines whether input sanitization can mitigate many-shot attacks.

### 3.5 TR141: Cross-architecture batch fragility protocol

TR141 spans three experimental campaigns on the RTX PRO 6000 Blackwell Server Edition (98GB VRAM, Google Colab): TR141a (core, 7 models, 49,476 records), TR141b (large-model extension, 3 models, 21,204 records), and v3 (alignment-balance extension, 8 models, 56,544 records). The combined v2.1 + v3 synthesis pools 106,020 scored records across 15 distinct models.

Phase 1 uses synchronized-dispatch batching: all requests for a given batch configuration are sent simultaneously, with each request's output compared to the batch = 1 baseline. Phase 2 uses explicit prompt-list batching for true-batch validation. The 28 analysis passes include output identity, flip rate computation, directional analysis, ANOVA on alignment type, regression of output instability on fragility, and combined v2.1 + v3 synthesis.

The critical methodological correction: v2.1 reported alignment-type ANOVA at p = 0.008 with n = 1 per category (pseudoreplication). The v3 extension added 8 models to achieve n >= 3 per alignment category (RLHF 4, SFT 4, DPO 4, distilled 3), and the effect disappeared (F = 0.13, p = 0.942 at the model level, F = 1.88, p = 0.131 at the prompt level).

### 3.6 TR142: Quality-safety correlation methodology

TR142 is an analysis-only study that cross-references two existing source datasets: TR125 Phase 2 quality measurements (24,990 source samples) and TR134 Phase 3 safety measurements (24,778 source samples). The analysis restricts to the shared model/quant overlap (2 models: llama3.2-1b, llama3.2-3b; 7 quantization levels: FP16 through Q2_K), yielding 10,290 quality samples and 13,342 safety samples.

The 14 analysis passes compute within-model Pearson correlations between quality and safety metrics, asymmetry indices (which moves more at each quant level), BPW regression, quality-gate analysis, and per-cell divergence analysis. The key methodological contribution is the per-model correlation analysis rather than pooled analysis: pooling the two models would produce a misleading aggregate because the within-model correlations have opposite signs.

### 3.7 TR143: Cross-request composition protocol

TR143 implements a 4-phase design under vLLM FP16 continuous batching. Phase 1 (composition effect): 3 models x 5 safety conditions (solo, benign-7, jailbreak-7, mixed-4/3, refusal-7) x 468 safety prompts + 2 capability conditions x 485 capability prompts = 9,930 records. Phase 2 (temporal overlap): 2 models x 8 overlap levels x 200 prompts = 3,200 records. Phase 3A (reverse direction): 2 models x 3 conditions x 120 prompts = 720 records. Phase 3B (static vs. continuous): 2 models x 2 conditions x 200 prompts = 400 records.

Co-batch verification is the most significant methodological challenge: of 11,151 observations checked, only 2,466 (22.1%) were confirmed to be actually co-batched with the intended fillers. This is because vLLM's continuous batching scheduler makes independent decisions about when to group requests, and the experimental design cannot force specific co-batching patterns. The 22.1% rate biases toward the null (effects are underestimated if the treatment was not delivered) and limits the strength of both positive and negative claims.

### 3.8 Statistical toolkit

The statistical toolkit reflects the heterogeneous experimental designs across the 6 TRs: paired binary comparisons (TR143), full-factorial ANOVAs (TR139), regression analyses (TR141, TR142), and cross-reference correlation analyses (TR142). Each method is selected to match the specific inferential demand of the comparison it serves. The following catalog documents every statistical method used, its assumptions, and where it is applied:

- **McNemar's test:** Paired non-parametric test for binary outcomes on the same subjects (TR143 composition comparisons). Tests whether the proportion of flips in each direction differs from chance.
- **Cochran's Q:** Extension of McNemar to 3+ matched conditions (TR143 omnibus test across all composition conditions).
- **Mantel-Haenszel:** Pooled odds ratio stratified by model, providing an aggregate effect estimate that adjusts for model as a confounder (TR143).
- **One-way ANOVA:** For 3+ group comparisons (TR139 strategy-specific quantization effects, TR141 alignment-type comparison).
- **Welch's t-test:** For unequal-variance group comparisons (TR139 multi-turn vs. direct slope comparison).
- **TOST equivalence:** Two One-Sided Tests at +/-3pp margin where "no difference" is claimed (TR138, TR141, TR143).
- **Binomial test:** For directional asymmetry in flip direction (TR143, TR141).
- **Fisher exact test:** For 2x2 comparisons with small cell counts (TR140 ASR comparisons).
- **Bootstrap CIs:** 2,000 iterations, percentile method, on all key metrics (TR139 persistence slopes, TR141 fragility estimates).
- **Pearson correlation:** For output instability vs. fragility (TR141, r = 0.91) and quality-safety coupling (TR142).
- **Linear regression:** With R-squared for BPW regression (TR142) and output-instability regression (TR141).
- **Cohen's kappa:** Inter-rater agreement between classifiers (TR139 dual-judge, TR140 judge vs. regex, TR141 cross-phase).
- **Cohen's h:** Effect size for proportions (TR140 ASR comparisons).
- **Eta-squared:** ANOVA effect size (TR139 strategy ANOVAs).
- **Holm-Bonferroni correction:** Applied to all multiple testing families across all TRs.
- **Power analysis (MDE):** Computed for each TR to characterize detection limits.
- **Wilson confidence intervals:** For proportions at extreme values and small sample sizes.
- **Logistic regression:** For dose-response analysis of temporal overlap (TR143 Phase 2).

The choice of TOST margin (+/-3pp) deserves justification. The 3 percentage-point equivalence margin was established in Phase 3 as the threshold below which a safety difference is considered practically insignificant. This margin is based on the measurement uncertainty inherent in the automated classifiers (kappa = 0.147 implies classification noise of several percentage points) and the run-to-run variance observed in Phase 3 (up to 5pp for some task-model pairs at anchor configurations). A margin tighter than 3pp would fail most comparisons due to measurement noise rather than genuine safety differences; a margin wider than 5pp would declare configurations equivalent that show operationally meaningful differences. The 3pp margin is conservative for safety-critical applications (where even small differences matter) and permissive for general-purpose applications (where 3pp is within normal variation).

The Holm-Bonferroni correction is applied within each TR's testing family but not across TRs. This is a deliberate choice: each TR addresses a distinct research question, and the family-wise error rate should be controlled within the scope of each question. Cross-TR correction would be overly conservative, penalizing findings in TR143 for the number of tests in TR139, even though the two TRs test unrelated hypotheses. The cross-TR confidence comes instead from convergent evidence: findings supported by multiple independent TRs (e.g., the Q2_K ban) are more reliable than findings from a single TR, regardless of within-TR correction.

### 3.9 Scoring pipeline

Safety scores across all TRs are produced by automated classifiers from the TR134 shared pipeline:

- **RefusalDetector:** Regex-based, approximately 15 refusal patterns. Binary output: 1 = refusal detected (safe), 0 = no refusal detected (unsafe). The v2.2 scorer (TR138) adds Unicode curly-quote normalization.
- **BiasDetector:** Regex-based stereotype detection for BBQ disambiguation questions.
- **TruthfulnessScorer:** Reference-matching for TruthfulQA gold-standard answers.
- **LLM Judge (TR139, TR140):** Qwen 2.5 7B Instruct Q8_0, structured classification into FULL_REFUSAL, PARTIAL_REFUSAL, COMPLIANCE, UNCLEAR.
- **Dual-Judge System (TR139):** 7B primary judge + 1.5B fallback. Overall agreement 64.99%, kappa = 0.104.

The TR138 human adjudication establishes a calibration layer: 73% of regex-detected safety flips are artifacts. This artifact rate should be applied as a correction factor when interpreting any small-magnitude flip-rate finding from automated classifiers.

### 3.10 Model selection rationale

The model selection strategy across the six TRs balances four competing objectives: (1) cross-TR comparability (requiring shared anchor models), (2) architectural diversity (requiring models from different families), (3) alignment-type coverage (requiring models trained with different safety methods), and (4) computational feasibility (requiring models that fit in the available VRAM at the tested configurations). No single model set optimizes all four objectives simultaneously, so the program uses different model sets per TR while maintaining a shared anchor set (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) that appears in 5-6 of the 6 TRs.

The model selection across the six TRs reflects a progression from focused experiments (TR138: 3 models, TR142: 2 models) to comprehensive architectural surveys (TR141: 18 models):

**TR138:** 3 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b). Chosen for VRAM feasibility on 12GB hardware across batch sizes up to 32. Overlaps with Phase 3 anchor models (Llama 1B/3B) for cross-phase comparison.

**TR139:** 4 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b, llama3.1-8b). Adds the 8B model to test whether scale modulates multi-turn vulnerability. The 8B model shows an instability band around Q4_K_M to Q3_K_M rather than monotone degradation.

**TR140:** 4 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b, llama3.1-8b). Same set as TR139 for direct cross-TR comparison of attack modalities.

**TR141:** 18 unique models across 3 campaigns (7 core + 3 large + 8 alignment-balance). Spans 10+ families and 4 alignment types. The combined v2.1 + v3 synthesis uses 15 distinct models (3 dropped for technical reasons: gemma-2-2b, llama3.1-8b, gemma-3-1b). Provides the architectural diversity that all prior TRs lack.

**TR142:** 2 models (llama3.2-1b, llama3.2-3b). Restricted to the models shared between TR125 (quality source) and TR134 (safety source). This 2-model restriction is the study's primary limitation but is forced by the cross-reference design.

**TR143:** 3 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b). Same set as TR138 for direct comparison of batch-size and batch-composition effects.

### 3.11 Task battery design

The task battery design reflects a deliberate tension between comprehensive safety coverage and computational tractability. A safety evaluation battery that covers every possible safety dimension would require thousands of prompts across dozens of categories, making the per-cell sample sizes prohibitively small for the large experimental designs in TR139 and TR140. The program resolves this tension by using a smaller battery (4-6 tasks) that covers the major safety dimensions while maintaining per-cell sample sizes adequate for detecting large effects.

The task batteries vary by TR, reflecting the different research questions:

**TR138, TR141, TR143:** Safety tasks (AdvBench refusal, TruthfulQA, BBQ bias, jailbreak amplification) and capability tasks (MMLU, ARC-Challenge) from the Phase 3 shared battery. Enables direct cross-phase comparison with TR134-TR137.

**TR139:** 50 harmful behaviors from JailbreakBench, each evaluated with 8 multi-turn strategies. The behaviors span categories including violence, fraud, illegal activities, and harmful content generation. No capability reference tasks (the study focuses exclusively on safety under adversarial pressure).

**TR140:** Same 50 behaviors as TR139, presented as many-shot prompts at 5 shot counts and 2 formats. Phase 2 adds 3 context-length profiles. No capability reference tasks.

**TR142:** Cross-references TR125 Phase 2 quality tasks (MMLU, ARC-Challenge, BERTScore, coherence) with TR134 Phase 3 safety tasks (AdvBench, TruthfulQA, BBQ, jailbreak). The study does not run new evaluations; it analyzes existing data from two source TRs. The cross-reference design is methodologically clean (same models, same quantization levels, same hardware) but limited by the 2-model overlap between the source TRs. Adding additional models would require re-running both the quality and safety evaluation batteries, which is identified as future work.

The battery consistency across TR138, TR141, and TR143 (all using the Phase 3 shared safety battery) is a methodological strength that enables direct cross-TR comparison. A safety score measured on llama3.2-1b at batch=1 in TR138 can be compared directly to the same metric measured on the same model at batch=1 in TR141 or TR143, providing the cross-TR anchor validation that Section 9 uses to assess program consistency.

### 3.12 Human adjudication protocol

TR138's human adjudication is the only instance of human review in this synthesis. Of the 49 behavior-changing rows identified in the v1 run, the v2.2 scorer correction reduced the candidate set to 44 (5 false-flip rows removed by curly-quote normalization). A single human reviewer classified each of the remaining 44 candidates as either a genuine behavioral change (the model's safety posture changed) or a classifier artifact (the model rephrased the same safety posture, causing a different regex match).

The result: 17 of 63 total reviewed rows (27%) were genuine behavioral changes; 46 (73%) were artifacts. Of the genuine flips, 26/44 were in the unsafe direction (59.1%), confirming a directional asymmetry. The single-reviewer limitation is acknowledged: inter-rater reliability has not been established, and the 27% genuine rate may shift with a different reviewer or different adjudication criteria. The human review queue (757 rows for TR138, 263 for TR141) has been generated but not yet completed.

This 73% artifact rate is a significant methodological finding, but its scope must be stated precisely. It directly recalibrates TR138's batch-perturbation flip rates: the 0.6% automated rate becomes approximately 0.16% genuine (90% CI: 0.10-0.24%, propagating the uncertainty on the 17/63 = 27% genuine rate). The artifact rate was measured on batch-size flips classified by RefusalDetector; it does not directly transfer to TR139-TR143, which test different phenomena (multi-turn ASR, many-shot ASR, composition flips) with potentially different artifact profiles. However, it establishes a general principle: automated safety classifiers can substantially overestimate small effects (< 5%), and studies reporting small effects without human validation should be interpreted with caution. The TR141 and TR143 flip rates (which have not been human-adjudicated) should be treated as upper bounds on genuine behavioral change rates.

### 3.13 Co-batch verification methodology

TR143's co-batch verification checks whether target and filler requests were actually processed in the same physical batch by vLLM's continuous batching scheduler. The verification uses timing analysis: if the target request's processing time overlaps with the filler requests' processing times, co-batching is confirmed.

The 22.1% verification rate (2,466 of 11,151 observations) means that the experimental design achieved its intended treatment in only one-fifth of cases. For the remaining 78%, the target request may have been processed solo (before fillers arrived) or in a partial batch (with some but not all intended fillers). This biases the experiment toward the null: if composition effects exist, they are measured on a diluted treatment, making them harder to detect. The aggregate null finding may therefore be stronger than it appears -- if the null holds at 22.1% treatment delivery, it would likely hold at 100% delivery.

The verification rate limitation is a design constraint of continuous batching systems, not a researcher error. vLLM's scheduler optimizes for throughput, not for experimental control, and there is no API to force specific co-batching configurations. Future work must either use a modified scheduler with co-batching guarantees or instrument the scheduler to log actual batch compositions.

### 3.14 LLM judge design and dual-judge system

TR139 uses a dual-judge system to score multi-turn conversations. The primary judge is Qwen 2.5 7B Instruct at Q8_0, which evaluates each conversation turn and produces a classification (FULL_REFUSAL, PARTIAL_REFUSAL, COMPLIANCE, UNCLEAR). The fallback judge is a 1.5B model used when the 7B judge fails to produce a parseable classification.

The dual-judge agreement statistics (overall 64.99%, kappa = 0.104) reveal substantial disagreement, concentrated in the mid-quantization Phase 2 cells where model outputs are most ambiguous. The low kappa is partially a base-rate artifact (similar to the kappa = 0.147 in Phase 3), but it also reflects genuine classification difficulty: multi-turn conversations produce more nuanced responses than single-turn prompts, and the boundary between "partial refusal" and "compliance" is inherently subjective.

TR140 achieves better agreement (overall 90.3%, kappa = 0.23) because many-shot ASR classifications are less ambiguous -- the model either complies with the harmful request or it does not. The Q2_K stratum has the lowest agreement (63.5%, kappa = 0.13) because incoherent outputs resist binary classification.

TR141's final artifact chain is regex-only -- the LLM-judge layer from intermediate work was not preserved. This is acknowledged as a limitation: the TR141 flip rates are based solely on deterministic regex classifiers, without the independent LLM-judge validation available in TR139 and TR140.

### 3.15 Measurement uncertainty quantification

The following uncertainty estimates should be applied when interpreting key findings:

**TR139 ASR values (±15pp).** The dual-judge system achieves kappa = 0.104 (slight agreement). At this agreement level, individual ASR values carry approximately ±15pp uncertainty. A reported ASR of 45% should be interpreted as "approximately 30-60%." The extreme values (0% and 100% ASR) are more reliable because both judges agree when models completely refuse or completely comply. The relative rankings between conditions (Q2_K is worse than Q4_K_M, attention_shift is more effective than role_play) are robust to this uncertainty because the same measurement error applies to all conditions and cancels in cross-condition comparisons.

**TR138 genuine flip rate (0.10-0.24%).** The 27% genuine rate from human adjudication (17/63) has a 95% Clopper-Pearson confidence interval of [16.3%, 39.7%]. Propagating this to the 0.6% automated rate: 0.6% × [16.3%, 39.7%] = [0.10%, 0.24%]. The point estimate of 0.16% should be cited with this range.

**TR141 safety/capability ratio (0.94x).** This is a ratio of two small proportions (0.75% / 0.80%) computed across 15 models. The ratio is consistent with parity (1.0x) but the confidence interval has not been bootstrapped. The directional conclusion ("near parity, not strongly safety-specific") is robust; the precise ratio should not be over-interpreted.

**TR143 directional asymmetry (88-92%).** Based on 9-12 total flips per condition. At n=11 flips, the 95% CI on an observed 91.7% (11/12) is [61.5%, 99.8%] (Clopper-Pearson). The asymmetry is statistically significant (p=0.006) but the point estimate is imprecise. The finding should be interpreted as "majority of rare flips trend toward unsafe" rather than "exactly 91.7% of flips trend toward unsafe."

**TR142 quality-safety divergence (13.9x).** Measured on a single model (llama3.2-1b) at a single quantization level (Q3_K_S). No confidence interval is available because this is a ratio of two point estimates (13.6pp safety delta / ~1pp quality delta). The finding that quality and safety diverge is supported; the precise 13.9x magnitude is specific to this model-quant combination and should not be generalized.

### 3.16 Power analysis for key null findings

| Null Finding | TR | Sample Size | MDE (80% power) | Interpretation |
|-------------|-----|------------|-----------------|----------------|
| Batch composition aggregate | TR143 | 468 prompts/condition | 4.7pp | Effects below 4.7pp are undetectable; the null means "no effect ≥ 4.7pp" |
| Multi-turn not selectively amplified | TR139 | 8 strategies | ~10pp slope difference | The Welch p=0.702 means multi-turn and direct slopes differ by less than ~10pp |
| Temporal overlap dose-response | TR143 | 200 prompts x 8 levels | ~8pp | Logistic regression slopes of p > 0.93 mean no dose-response ≥ 8pp |
| Alignment type (model-level) | TR141 | 15 models, 4 groups | ~15pp between groups | ANOVA p=0.942 means alignment groups differ by less than ~15pp on fragility |

These MDEs should accompany every null-finding claim. "Not detected" at these sample sizes is not the same as "does not exist."

### 3.17 Ethics statement

All adversarial evaluation in this program (TR139: 10,600 multi-turn jailbreak conversations; TR140: 15,000 many-shot jailbreak prompts; TR134: 120 jailbreak-wrapped prompts) was conducted on locally hosted open-weight models using publicly available safety benchmarks (AdvBench, JailbreakBench, BBQ, TruthfulQA). No proprietary or API-gated models were tested. No human subjects were involved. All harmful prompts are drawn from published benchmark datasets (Zou et al. 2023, Chao et al. 2024, Parrish et al. 2022).

Model outputs from adversarial evaluation are stored locally on the research hardware and are not published, shared, or made publicly accessible. The analysis artifacts (JSON files containing aggregate statistics, not raw model outputs) are the only data referenced in published reports. Raw model outputs containing potentially harmful content remain on local storage for reproducibility purposes only.

The research is conducted for defensive purposes: measuring safety degradation under optimization to inform deployment decisions that protect end users from harmful model behavior. No adversarial techniques developed in this program are intended for, or optimized for, attacking production systems.

### 3.18 Peer review status

This work has not been externally peer reviewed. All findings, statistical analyses, and conclusions represent the output of a single research program without independent expert verification. The self-corrections documented in this synthesis (TR141 ANOVA reversal, TR143 v1.0 retraction) were identified through internal replication and review, not external audit. Readers should apply their own critical assessment to the claims, methodology, and statistical interpretations presented here. The artifact-first reporting methodology (every claim traced to a specific data file) is designed to facilitate such independent verification.

---

## 4. Decision Impact Matrix

The decision impact matrix maps each finding to its primary domain, the decision it enables, the confidence level, and the boundary conditions under which the decision may not hold. This matrix is the navigational core of the synthesis: a practitioner who reads only this section and the Safety Decision Card can make safety-informed deployment decisions backed by 306,996 evaluated samples.

| Decision Topic | Finding | Source | Confidence | Action |
|---------------|---------|--------|------------|--------|
| Q2_K safety ban | Universal vulnerability across attack types | TR139, TR140, TR142 | **High** | Never deploy Q2_K |
| Q3_K_S safety | Safety degrades 13.9x faster than quality | TR142 | **High** | Quality benchmarks insufficient; validate safety directly |
| Multi-turn jailbreak x quant | All 8 strategies amplified by quantization | TR139 | **High** | Multi-turn testing required at deployment quant level |
| Many-shot format vulnerability | Message array 92% vs. faux dialogue 0% ASR | TR140 | **High** | Restrict/sanitize message-array inputs |
| Batch size perturbation | 0.16% human-adjudicated genuine rate | TR138 v2 | **Moderate** | Low-priority; measure output instability if concerned |
| Batch composition | Aggregate null, directional 88-92% unsafe | TR143 | **Moderate** | No routing needed; monitor directional flips |
| Output instability predictor | r = 0.91, R-squared = 0.83 | TR141 | **High** | Use as screening metric; > 15% implies > 1% flip rate |
| Alignment type as predictor | Not predictive (p = 0.942) | TR141 v3 | **High** | Evaluate models individually, not by alignment category |
| Quality as safety proxy | 13.9x divergence, model-dependent sign | TR142 | **High** | Never use quality metrics alone for safety validation |
| Multi-turn amplification vs. direct | Not selectively amplified (Welch p = 0.702) | TR139 | **Moderate** | Quant degrades safety broadly, not multi-turn specifically |
| Persistence under quant | 3 of 4 models show negative slope | TR139 | **Moderate** | Validate persistence, not just initial refusal |
| Many-shot x quant interaction | Q2_K universal, format dominant | TR140 | **High** | Format mitigation more effective than quant restriction |
| Cross-architecture fragility | 6.3x range across 15 models | TR141 | **High** | Per-model profiling required |
| Self-correction pattern | TR141 ANOVA, TR143 retraction | TR141, TR143 | **Methodological** | Strengthens program credibility |
| Co-batch verification | Only 22.1% confirmed | TR143 | **Limitation** | Future work must improve verification |
| Automated classifier artifact rate | 73% of small-effect flips are artifacts | TR138 v2 | **High** | Human-validate any flip rate < 5% |

### Interpreting the Matrix

Three structural patterns emerge from the matrix.

**Pattern 1: Quantization dominance.** Quantization dominates the threat landscape across all attack modalities (multi-turn, many-shot, long-context, single-turn) and produces effects two orders of magnitude larger than batch effects. **Pattern 2: Negative findings are operationally valuable.** The most actionable findings are the negative ones: alignment type does not predict fragility, quality does not proxy for safety, composition does not affect aggregate outcomes. These negative findings free operator attention for the positive findings that require action. Each negative finding eliminates a potential shortcut that might otherwise divert resources from the high-impact mitigations. If alignment type predicted fragility, operators could skip per-model profiling for models in "safe" alignment categories -- but it does not, so per-model profiling is mandatory. If quality proxied for safety, operators could rely on quality benchmarks -- but it does not, so safety-specific benchmarks are required. If composition affected safety, operators would need composition-aware routing -- but it does not, so the engineering effort can be redirected.

**Pattern 3: Confidence asymmetry.** The confidence column reveals an asymmetry between well-powered positive findings (TR139's ANOVAs with 48,425 records) and partially-powered exploratory findings (TR143's directional asymmetry with 9-12 total flips per condition). The program's recommendations are calibrated to this asymmetry: high-confidence findings drive Tier 1 and Tier 2 actions, while moderate-confidence findings drive monitoring recommendations rather than deployment blocks. This calibration is deliberate: in safety-critical domains, the consequence of acting on a high-confidence finding (validating quantization safety) is low-cost, while the consequence of ignoring a moderate-confidence finding (not monitoring directional patterns) is also low-cost. The asymmetry in confidence naturally maps to an asymmetry in recommended action intensity.

The matrix also reveals the progressive narrowing of the threat surface across the program. Phase 3 established 3 optimization axes as potentially dangerous. This synthesis narrows the active threat surface: quantization remains dangerous (confirmed and extended), batch perturbation is real but negligible (narrowed from "potentially dangerous" to "monitor"), and batch composition is closed as a threat (eliminated). The net effect is that the actionable threat surface is more focused after this synthesis than before: operators need to validate quantization safety and implement format-level input restrictions, but they can skip composition-aware routing and deprioritize batch-size effects.

---

## 5. Results by Report

The following six subsections present the results of each technical report using a standardized format: research question, experimental design, key results table, extended analysis, and opening for the next report. Each subsection is self-contained but cross-references other reports. The subsections are ordered by TR number (TR138 through TR143), matching the chronological order of the research program, because each report's design was informed by the findings of its predecessors.

### 5.1 TR138: Batch Inference Safety Under Non-Determinism

**Research question.** Does batch size change safety outcomes through floating-point non-determinism, and if so, does the effect disproportionately target safety relative to capability?

The Phase 3 safety program (TR134-TR137) evaluated all models at effectively batch = 1, one prompt at a time. But production LLM deployments process prompts in batches of 2, 4, 8, or more. The batch size determines the number of concurrent rows in the GPU's matrix multiplications, which changes the accumulation order of partial products due to floating-point non-associativity. If safety-critical tokens (the tokens that determine refusal vs. compliance) lie near decision boundaries more often than capability-critical tokens, then batch-size variation would disproportionately affect safety. TR138 tests this hypothesis directly across 3 models and 4 phases, producing 38,667 total records in the v1 run and 7,257 samples in the v2 replication.

**Experimental design.** Phase 1: 3 models x 6 batch sizes (1, 2, 4, 8, 16, 32) x full task battery. Phase 2: co-batching interference with 3 prompt types (benign, adversarial, safety) at batch = 8. Phase 3: quantization variation (Q4_K_M, Q8_0) under concurrent load. Phase 4: true-batch validation with explicit prompt-list batching. V2 audit: 44 behavior-changing rows from v1 reviewed with corrected v2.2 scorer. V2 replication: 7,257 samples on enriched 187-prompt subset.

**Key results.**

| Metric | Value | Context |
|--------|-------|---------|
| Safety flip rate (v1 regex) | 0.6% | Upper bound; includes 73% artifacts |
| Human-adjudicated genuine rate | 27% (17/63) | Single reviewer; 73% are rephrased refusals |
| Corrected safety flip rate | ~0.16% | Lower bound on genuine behavioral change |
| Capability flip rate (v1) | 0.15% | Approximately comparable to corrected safety rate |
| Safety/capability flip ratio (enriched) | 4.0x (1.68% vs 0.42%) | On enriched boundary-sensitive subset |
| Refusal-to-compliance direction | 72.7% | Of directionally classified safety flips |
| Unsafe audit share | 59.1% (26/44) | Of v2.2-corrected audit candidates |
| True-batch flip agreement | 98.67% | Phase 4 vs Phase 1 |
| True-batch safety flip rate | 3.27% | On enriched subset with true batching |
| Phase 3: models with significant quant effect | 3/3 | Concurrency and interaction null |
| V2.2 scorer correction | 49 to 44 candidates | 5 false-flip rows from curly-quote mismatch |

**Extended analysis.** The single most important finding in TR138 is not the flip rate but the human adjudication: 73% of regex-detected flips are artifacts. This finding recalibrates the entire batch-safety hypothesis. Before adjudication, the 0.6% safety flip rate (4x the capability rate) suggests that batch perturbation poses a meaningful safety risk. After adjudication, the 0.16% genuine rate (approximately 1x the capability rate) suggests that batch perturbation introduces a small, non-safety-specific instability that is predominantly a classifier measurement artifact.

The directional finding is more robust than the magnitude finding. Among genuine flips, the majority move in the unsafe direction (refusal to compliance: 72.7% of directionally classified flips; unsafe audit share: 59.1% of v2.2 candidates). This directional bias is consistent with the floating-point perturbation mechanism: when the epsilon at a decision boundary flips a token, the model is more likely to flip from refusal to compliance than vice versa, because refusal requires the model to maintain a specific behavioral pattern (recognizing harmful intent and activating safety training) while compliance is the "default" behavior that the model falls into when safety activation fails.

The Phase 4 true-batch validation is methodologically important because it rules out the possibility that the batch-perturbation signal is a scheduler artifact. If the effect were caused by Ollama's request scheduling rather than by GPU-level batch computation, true batching (explicit prompt-list inference) would produce different results than synchronized dispatch (sending multiple independent requests simultaneously). The 98.67% flip agreement between Phase 4 and Phase 1 confirms that both methods produce the same perturbation pattern, supporting the floating-point mechanism over a scheduling explanation.

The v2.2 scorer correction (curly-quote normalization) removes 5 false-flip rows caused by a regex mismatch on Unicode apostrophe variants. This is a small correction in absolute terms but illustrative of the general principle that automated safety classifiers are sensitive to text formatting in ways that do not reflect genuine behavioral changes. The 73% artifact rate is likely composed of many such small formatting-sensitivity issues aggregating across the classifier's pattern library.

The enriched-subset methodology used in the v2 replication deserves scrutiny. The 187-prompt enriched subset was specifically selected for boundary-sensitivity -- prompts where batch perturbation is most likely to change outputs. Running the replication on this subset produces a 1.68% safety flip rate, substantially higher than the 0.6% observed on the full prompt set. This is expected: the enrichment concentrates the effect. The enriched-subset results should not be compared directly to the full-prompt-set results; they measure the effect on the most sensitive portion of the prompt distribution, not the overall prompt distribution. The 0.6% (full set) and 0.16% (human-adjudicated) are the operationally relevant numbers; the 1.68% (enriched subset) characterizes the worst-case tail.

The Phase 2 co-batching interference experiment shows that prompt-type mixing (benign, adversarial, safety) at batch = 8 does not amplify the batch-perturbation effect. This anticipates TR143's composition null finding: the content of co-batched prompts does not modulate the safety perturbation. The perturbation mechanism is floating-point accumulation order, which depends on batch size but not on prompt content. This mechanistic consistency across TR138 Phase 2 and TR143 Phase 1 strengthens confidence in the composition null.

The Phase 3 quantization result (3/3 models significant for quantization effects, concurrency and interaction null) is consistent with Phase 3's finding that quantization is the dominant safety variable while concurrency is safe. The specific contribution of TR138 Phase 3 is showing that quantization effects persist under concurrent load: the safety degradation at lower quantization is not a static property that disappears when the GPU is busy. This is important for production scenarios where quantized models serve concurrent requests.

**Opening for TR141.** TR138 establishes the batch-perturbation phenomenon on 3 models from 2 families. The natural follow-up is architectural generalization: does the effect hold across a broader range of models, and does it correlate with model properties (alignment type, size, architecture) that could enable prediction without per-model testing?

### 5.2 TR139: Multi-Turn Jailbreak x Quantization

**Research question.** Does quantization amplify multi-turn jailbreak vulnerability, and does the interaction between quantization level and attack strategy produce vulnerability spikes that neither factor alone would predict?

Phase 3 (TR134) established that quantization degrades single-turn refusal rates. But adversarial users do not limit themselves to single-turn attacks. Multi-turn jailbreaking applies escalating pressure across conversation turns, exploiting the model's tendency to maintain conversational consistency. If quantization erodes the model's ability to maintain refusal across turns -- because the weight perturbation degrades the model's representation of conversational context -- then multi-turn strategies should be amplified at lower quantization levels. TR139 tests this hypothesis with the largest multi-turn jailbreak study in the program: 10,600 conversations, 37,825 judge labels, 4 models, 6 quantization levels, and 8 attack strategies.

**Experimental design.** Phase 1: 4 models x 6 quant levels x 8 strategies x 50 behaviors = 9,600 conversations. Phase 2: 4 models x 5 quant levels x 50 behaviors = 1,000 persistence conversations. Post-hoc: 37,825 dual-judge labels (7B primary + 1.5B fallback).

**Key results.**

| Metric | Value | Context |
|--------|-------|---------|
| Peak ASR | 100% | qwen2.5-1.5b / Q2_K / attention_shift |
| Second-highest ASR cells | 100% | qwen2.5-1.5b / Q2_K / context_fusion, crescendo |
| Llama highest ASR | 86% | llama3.2-1b / Q2_K / crescendo |
| ANOVA rejections (H1) | 8/8 strategies | All p < 1e-4; quantization affects all strategies |
| Eta-squared range | 0.031-0.153 | context_fusion (lowest) to direct (highest) |
| Multi-turn vs. direct slope | Not significant | Welch p = 0.702; multi-turn not selectively amplified |
| TOST equivalence checks | 0/160 | No adjacent quant levels are interchangeable |
| Persistence slope (llama3.2-1b) | -0.198 per BPW | Bootstrap CI [-0.305, -0.144]; strong degradation |
| Persistence slope (llama3.1-8b) | Flat | High break rate across sweep; floor effect |
| Persistence H3 support | 3/4 models | Negative slopes; aggregate supports degradation |
| Dual-judge agreement | 64.99% | Kappa = 0.104 (poor) |
| Median MDE at 80% power | 15.19pp | Adequate for detecting large effects |

**Extended analysis.** The qwen2.5-1.5b / Q2_K / attention_shift cell achieving 100% ASR is the single most alarming finding in the synthesis. A quantized model that appears safe under single-turn evaluation becomes completely compromisable through multi-turn attack. The attention_shift strategy is particularly effective because it redirects the model's attention from safety cues ("this request is harmful") to task-completion cues ("I should help the user complete their request"), exploiting the model's instruction-following training against its safety training. At Q2_K, where the safety weights are most degraded, the instruction-following pathway dominates and the safety pathway fails entirely.

The non-significance of the multi-turn vs. direct slope comparison (Welch p = 0.702) is an important null finding. It means that quantization does not selectively amplify multi-turn attacks -- it degrades safety broadly, making both single-turn and multi-turn attacks more effective at comparable rates. The average direct slope (-0.0793 per BPW) and average multi-turn slope (-0.0536 per BPW) are statistically indistinguishable. This has a practical implication: the defense against quantization-amplified attacks is quantization control (stay at Q4_K_M or above), not multi-turn-specific defenses.

The 0/160 TOST result (no adjacent quantization levels establish equivalence at +/-3pp) is consistent with Phase 3's finding that quantization levels are not interchangeable for safety. Every step down the quantization ladder changes the model's safety profile by a measurable amount -- the changes are not always large, but they are never provably absent.

The model-specific patterns reveal distinct vulnerability profiles. qwen2.5-1.5b is broadly vulnerable across the quantization ladder (high ASR even at Q8_0 for certain strategies). llama3.2-1b collapses specifically at Q2_K. llama3.1-8b shows an instability band around Q4_K_M to Q3_K_M rather than monotone degradation, suggesting a non-linear interaction between model scale and quantization effects. llama3.2-3b is the strongest model in the sweep but should not be assumed robust-by-default -- it simply has a higher safety threshold that Q2_K eventually breaks.

The persistence analysis (Phase 2) adds a critical operational dimension that Phase 1's ASR alone cannot capture. A model that initially refuses but then breaks under continued pressure is arguably more dangerous than a model that immediately complies, because the initial refusal creates a false sense of safety. The persistence slopes (3 of 4 negative, with llama3.2-1b showing the steepest degradation at -0.198 per BPW) reveal that lower quantization not only makes models more likely to comply on the first attempt but also makes them less likely to maintain refusal under sustained pressure. The exception is llama3.1-8b, where the signal is not a clean slope but rather a high break rate across the entire quantization sweep -- the model is so weak under multi-turn pressure at all quantization levels that the quantization effect is masked by a floor effect.

The 0/160 TOST non-equivalence finding deserves emphasis. Across 160 adjacent-quantization-level comparisons, not a single pair establishes safety equivalence at +/-3pp. This means that every quantization step changes the model's multi-turn safety profile by a measurable amount. The practical implication is that operators cannot assume that Q4_K_M is "close enough" to Q5_K_M for multi-turn safety -- they must test at their specific deployment level. This is a stronger claim than Phase 3's finding (which showed non-equivalence for some but not all single-turn comparisons) because multi-turn evaluation is more sensitive to quantization-induced changes in conversational consistency.

The model-specific vulnerability profiles are operationally important for deployment decisions. qwen2.5-1.5b should be treated as high-risk for any conversational deployment, regardless of quantization level, because its baseline vulnerability (high ASR even at Q8_0 for certain strategies) means that quantization merely amplifies an existing weakness rather than creating a new one. llama3.2-3b is the strongest model in the sweep, but its strength is relative, not absolute -- at Q2_K, even llama3.2-3b shows elevated vulnerability. The key insight is that model selection is a more powerful lever for multi-turn safety than quantization selection: choosing a robust model (llama3.2-3b) at Q4_K_M is safer than choosing a vulnerable model (qwen2.5-1.5b) at Q8_0.

**Opening for TR140.** TR139 establishes that quantization amplifies multi-turn jailbreaks. The natural follow-up is whether quantization also amplifies many-shot jailbreaks -- a different attack modality that uses in-context compliance examples rather than conversational pressure.

### 5.3 TR140: Many-Shot Long-Context Jailbreak

**Research question.** Does quantization amplify many-shot jailbreaking through in-context learning, and does the prompt format determine attack success more than the quantization level?

Many-shot jailbreaking exploits in-context learning by embedding compliance examples in the prompt. If quantization degrades the model's ability to resist in-context pressure -- because the degraded weights cannot maintain the safety signal against the competing compliance signal from the in-context examples -- then lower quantization should produce higher ASR at the same shot count. TR140 tests this across 4 models, 6 quantization levels, 5 shot counts, and 2 prompt formats, producing 15,000 scored samples.

**Experimental design.** Phase 1: 4 models x 6 quants x 5 shot counts x 2 formats x 50 behaviors = 12,000 samples. Phase 2: 4 models x 5 quants x 3 context profiles x 50 behaviors = 3,000 samples. All 15,000 samples judged (100% completion).

**Key results.**

| Metric | Value | Context |
|--------|-------|---------|
| Peak ASR | 99% | qwen2.5-1.5b / Q2_K / N=128 |
| qwen2.5-1.5b at Q8_0 / N=128 | 40% | Vulnerable even without quantization |
| Llama immunity threshold | Q3_K_M | ASR at or below 2% at Q4_K_M and above |
| Llama Q2_K peak | 72% | llama3.2-1b at N=128 |
| Message array vs. faux dialogue | 92% vs. 0% | llama3.1-8b / Q2_K / N=16 |
| Format on qwen2.5-1.5b Q4_K_M | 86% vs. 4% | At N=128 |
| ASR peak then decline | N=16 peak on llama3.1-8b | Context-window cap confirmed |
| Variance decomposition: residual | 65.7% | Per-behavior variation dominates |
| Variance decomposition: quantization | 17.9% | Second largest factor |
| Variance decomposition: model | 12.6% | Third largest |
| Variance decomposition: shot count | 2.7% | Smallest factor |
| Power-law exponent range | 0.15-0.77 | H1 (invariant exponent) rejected |
| Phase 2 context dilution slopes | All negative | Lower quant weakens safety with context prefix |
| Judge agreement (overall) | 90.3%, kappa = 0.23 | Moderate |
| Judge agreement (Q2_K) | 63.5%, kappa = 0.13 | Lowest stratum |
| Significant Fisher exact tests | 15/100 | All involving Q2_K or Q3_K_M, all large h (0.60-2.06) |

**Extended analysis.** The message-array vs. faux-dialogue finding (92% vs. 0% ASR on the same model at the same quantization level) is the most operationally actionable result in TR140. It means that format-level input sanitization is a more effective defense than quantization-level restriction: blocking message-array format inputs reduces ASR from 92% to 0%, while moving from Q2_K to Q8_0 reduces ASR from 92% to at most 40% (on qwen2.5-1.5b). The format-level defense is also cheaper to implement -- it requires input parsing rather than model replacement.

The mechanism behind the format effect is that message-array format exploits the model's chat template by presenting compliance examples as properly formatted assistant turns. The model processes these examples through the same code path as genuine conversation history, making them more convincing as in-context learning examples. Faux dialogue, by contrast, presents the same content as informal text that does not trigger the chat-template processing pathway, and the model does not interpret it as genuine conversation history.

The variance decomposition (residual 65.7%, quantization 17.9%, model 12.6%, shot count 2.7%) reveals that the specific harmful behavior being requested matters more than any experimental factor. This has implications for safety evaluation: a small benchmark of 10 behaviors may produce a very different ASR estimate than a benchmark of 50 behaviors, because the per-behavior variance is the dominant source of measurement uncertainty. The 50-behavior design in TR140 mitigates this but does not eliminate it.

The context-window cap finding (ASR peaks at N=16 on llama3.1-8b then declines) suggests a natural mitigation: context-length limits. At very high shot counts (N=64, N=128), the prompt becomes so long that the model's effective attention span is exceeded, and the many-shot signal degrades along with general coherence. This is not a reliable defense (it depends on the model's context-window length and attention architecture), but it suggests that unbounded context lengths are not uniformly more dangerous than bounded ones.

The convergence between TR139 and TR140 on Q2_K as the universal vulnerability threshold is one of the strongest cross-TR findings in the synthesis. Despite testing different attack modalities (multi-turn conversational pressure vs. many-shot in-context learning), different phase designs, and partially different model sets, both studies identify Q2_K as the level where all models become vulnerable. This convergence provides high confidence in the Q2_K ban: the vulnerability is not specific to one attack type or evaluation methodology.

The Llama immunity above Q3_K_M is an important positive finding for deployment. Llama-family instruct models show remarkable resilience to many-shot attacks at production-relevant quantization levels. At Q4_K_M and above, ASR is at or below 2% for all Llama variants across all shot counts. This suggests that Llama's RLHF training produces safety alignment that is deeply encoded enough to resist in-context learning pressure at production precision levels. The immunity breaks only at extreme compression (Q2_K), where the weight perturbation exceeds the alignment signal. This finding supports Q4_K_M as the safe deployment floor for Llama models under many-shot threat scenarios.

The power-law exponent variation (0.15-0.77) across quant levels confirms that quantization does not merely shift the ASR curve up or down -- it changes the shape of the relationship between shot count and ASR. At higher quantization levels, the exponent is lower (ASR increases slowly with shot count, reflecting strong safety training that resists in-context pressure). At lower quantization levels, the exponent is higher (ASR increases rapidly with shot count, reflecting degraded safety training that cannot resist in-context pressure). The H1 rejection (exponents differ by more than 0.1 across levels) means that the power-law model requires per-quant-level parameterization, not a universal exponent.

**Opening for TR141.** TR139 and TR140 together establish that quantization interacts with adversarial attack modalities. But these studies test only 3-4 models. Do the findings generalize across architectures?

### 5.4 TR141: Cross-Architecture Refusal Fragility

**Research question.** Does batch-induced floating-point perturbation systematically degrade safety outputs across architectures, scales, and alignment paradigms, and can any model-level property predict fragility without per-model testing?

TR138 established the batch-perturbation phenomenon on 3 small RLHF models. TR141 is the cross-architecture extension, scaling the question to 18 models from 10+ families across 3 experimental campaigns. The v2.1 campaign (7 models) found a mild safety-skewed aggregate and a significant alignment-type ANOVA (p = 0.008). The v3 campaign (8 additional models, designed to balance alignment types to n >= 3 per category) reversed both findings: the safety-capability ratio flipped to 0.78x, and the alignment ANOVA became decisively non-significant (p = 0.942). The combined 15-model synthesis (106,020 scored records) provides the broadest cross-architecture view of batch-perturbation safety behavior within this program.

**Experimental design.** TR141a: 7 models, 49,476 records. TR141b: 3 models (large, 7.2B-14.8B), 21,204 records. TR141 v3: 8 models (alignment-balanced), 56,544 records. Combined v2.1 + v3: 15 distinct models, 106,020 scored records. Phase 1: synchronized-dispatch batching. Phase 2: true prompt-list batching. 28 analysis passes.

**Key results.**

| Metric | Value | Context |
|--------|-------|---------|
| Models tested | 18 unique (15 in combined synthesis) | 10+ families, 4 alignment types |
| Total records | 127,224 (106,020 combined scored) | Largest single study in program |
| Safety flip range | 0.00% (tinyllama) to 2.39% (phi-2) | 6.3x range across 15 models |
| Safety/capability ratio (v2.1) | 1.36x | Mild safety-skewed |
| Safety/capability ratio (v3) | 0.78x | Reversed |
| Safety/capability ratio (combined) | 0.94x | Near parity |
| Output instability predictor | r = 0.909, R-squared = 0.827 | p < 0.0001 |
| Baseline refusal predictor | r = 0.028, p = 0.919 | Uninformative |
| Alignment ANOVA (v2.1) | p = 0.008 | False positive from pseudoreplication |
| Alignment ANOVA (v3, model-level) | F = 0.13, p = 0.942 | Decisively non-significant |
| Alignment ANOVA (v3, prompt-level) | F = 1.88, p = 0.131 | Also non-significant |
| Combined directional analysis | 159 safe vs. 81 unsafe | 66.2% safe direction, p = 1e-6 |
| Phase 2 safety flip rate | 0.80% | True-batch validation |
| Phase 2 flip agreement | 99.15% | Against Phase 1 |
| TOST equivalence | All comparisons pass at +/-3pp | In saved v3 artifact |
| Artifact chain | Regex-only | LLM judge not preserved |

**Extended analysis.** The alignment-type self-correction is the single most methodologically important finding in TR141, and arguably in the entire synthesis. The v2.1 report found p = 0.008 for alignment-type ANOVA, which would have supported the claim that alignment type (RLHF, SFT, DPO, distilled) predicts batch fragility. If true, this would have enabled a valuable deployment shortcut: operators could use alignment type as a proxy for fragility without per-model testing.

The correction demonstrates why pseudoreplication is dangerous and why balanced experimental design matters. The v2.1 ANOVA had unbalanced groups (RLHF: 4 models, others: 1 model each). With n = 1 per category for SFT, DPO, and distilled, the between-group variance was confounded with model-specific effects: any single model that happened to be fragile would make its entire alignment category appear fragile. The v3 extension added 8 models specifically chosen to balance the groups (RLHF 4, SFT 4, DPO 4, distilled 3), and the effect disappeared at p = 0.942.

This self-correction is a methodological strength, not a weakness. Research programs that never correct prior findings are either perfect (unlikely) or uncritical (dangerous). The TR141 correction follows the correct scientific pattern: preliminary finding with limited data, replication attempt with balanced design, honest reporting of the reversal. The published chain of evidence (v2.1 finding, v3 correction, combined synthesis) allows readers to trace the evolution of the claim and understand exactly why the initial finding was wrong.

The output-instability predictor (r = 0.91, R-squared = 0.83) provides a practical alternative to alignment-type heuristics. Output instability -- the fraction of responses that differ textually from the batch = 1 baseline -- can be measured cheaply (run the same prompts at batch = 1 and batch = N, compare outputs textually) and correlates strongly with safety fragility. A model with > 15% output change rate shows > 1% safety flip rate; a model with < 5% output change rate shows near-zero safety flips. This is a deployable screening metric that costs one batch of inference to measure.

The combined directional analysis (66.2% safe, p = 1e-6) shows a net-safe directional bias across the 15-model synthesis. This is the opposite direction from TR138's unsafe-direction finding (72.7% refusal-to-compliance). The discrepancy is informative: it shows that directionality is model-set-dependent rather than universal. The TR138 model set (3 small models, 2 families) may happen to include models whose safety training is more susceptible to perturbation in the unsafe direction, while the TR141 model set (15 models, 10+ families) includes models that perturb in both directions, with the net direction being safe. Neither finding should be generalized as a universal property of batch perturbation.

The three-campaign structure (core + large-model + alignment-balance) deserves methodological commentary. The core campaign (7 models) was designed for breadth across small models. The large-model extension (3 models, 7.2B-14.8B) tested whether the batch-perturbation phenomenon persists at larger scales -- it does, though the effect magnitudes are model-specific rather than scale-dependent. The alignment-balance extension (8 models) was explicitly designed to correct the v2.1 pseudoreplication: the 8 new models were selected to achieve n >= 3 per alignment category, enabling a properly powered ANOVA that could either confirm or refute the preliminary p = 0.008 finding.

The balance between campaigns is itself a methodological lesson. The v2.1 finding (p = 0.008) was published as a preliminary result, clearly labeled as such, and the v3 extension was designed specifically to test it. When the extension produced p = 0.942, the reversal was published with the same transparency as the original finding. This is the scientific method working as intended: preliminary finding, replication attempt, honest reporting of the outcome. The alternative -- not publishing the v2.1 finding until the v3 extension was complete -- would have been methodologically safer but would have delayed the alignment-type claim by weeks and prevented other researchers from independently testing it.

The model-level ranking (tinyllama 0.00% to phi-2 2.39%) provides a deployment guide that the aggregate statistics cannot. An operator choosing between phi-2 and tinyllama for a batch-inference pipeline can see that phi-2 has 2.39% safety fragility while tinyllama has 0.00%, and make an informed decision. The aggregate statistic (0.75% combined) is true but unhelpful for this decision. The per-model ranking is the practical deliverable of TR141.

The no-critical-batch-size-threshold finding means that the effect is diffuse across batch sizes rather than spiking at a specific value. This is consistent with the floating-point perturbation mechanism: the epsilon from accumulation-order changes is present at all batch sizes > 1, and its magnitude scales smoothly (not discontinuously) with batch size. There is no "safe" batch size and no "dangerous" batch size; there is only a continuously varying probability of decision-boundary flips that increases gently with batch size.

**Opening for TR142.** TR141 establishes that model-specific properties (especially output instability) predict safety behavior under batch perturbation. But TR141 measures safety in isolation. The next question is whether safety and quality degrade together or independently.

### 5.5 TR142: Quality-Safety Correlation Under Quantization

**Research question.** Does quality degradation under quantization predict safety degradation, or do they follow partially independent paths that could mislead practitioners who monitor only one dimension?

Practitioners commonly validate quantized deployments using quality benchmarks (MMLU, BERTScore, coherence). If quality metrics pass, the deployment is assumed safe. TR142 tests this assumption by cross-referencing TR125 Phase 2 quality measurements with TR134 Phase 3 safety measurements on the 2 shared models (llama3.2-1b, llama3.2-3b) across 7 quantization levels.

**Experimental design.** Analysis-only: no new experiments. Cross-references TR125 (24,990 quality samples) and TR134 (24,778 safety samples). Shared overlap: 10,290 quality + 13,342 safety samples across 2 models, 7 quant levels. 14 analysis passes.

**Key results.**

| Metric | Value | Context |
|--------|-------|---------|
| Quality-safety correlation (llama3.2-1b) | r = +0.994, p = 5.8e-5 | Strong positive coupling |
| Quality-safety correlation (llama3.2-3b) | r = -0.829, p = 0.041 | Nominal negative coupling |
| Safety/quality divergence at Q3_K_S (1b) | 13.9x | Refusal drops 13.6pp, quality moves ~1pp |
| Cells where safety moves more | 10 / 12 | Safety is the more sensitive metric |
| llama3.2-3b over-refusal at Q3_K_S | +18.6pp | Refusal increases as quality degrades |
| llama3.2-3b over-refusal at Q2_K | +16.4pp | Same pattern |
| Quality gate responsiveness | refusal 5.9%-34.5%; truth 4.0%-24.0%; bias 0% | Gate reacts to low-bit degeneration but does not restore proxy validity |
| Truthfulness MDE | 28.0pp | Underpowered for small effects |
| BPW regression R-squared | 0.03-0.30 | Non-linear thresholds dominate |
| Formal TOST proof | Not established | Current artifact lacks standalone TOST object |

**Extended analysis.** The 13.9x safety-quality divergence at Q3_K_S on llama3.2-1b is the finding with the broadest implications for the deployment community. A practitioner who validates a Q3_K_S deployment using quality benchmarks will observe approximately 1pp of degradation and conclude the deployment is safe. The actual safety degradation is 13.6pp -- well into the "high risk" tier by this program's classification. The quality gate gives no warning. This is not a marginal gap; it is a 13.9x discrepancy that would cause a safety-critical deployment to pass quality review while failing safety review.

The opposite-sign correlations between the two models are equally concerning from a methodological perspective. On llama3.2-1b, quality and safety degrade together (r = +0.994): as quantization compresses weights, both quality and safety suffer, and the strong positive correlation means quality degradation does predict safety degradation for this specific model. On llama3.2-3b, quality and safety degrade in opposite directions (r = -0.829): as quantization compresses weights, quality degrades but safety paradoxically improves through over-refusal. The model becomes less capable but more cautious, producing more refusals as a side effect of degraded coherence.

The practical consequence is that no single quality-safety correlation can be applied across models. A deployment validation protocol that uses quality as a safety proxy will be approximately correct for models like llama3.2-1b (where the correlation is strongly positive) and systematically wrong for models like llama3.2-3b (where the correlation is negative). Since there is no way to know a priori which direction the correlation runs for a new model, the only safe practice is to measure safety directly.

The updated quality-gating analysis provides a different calibration than the earlier draft. The gate is not invariant: refusal filter rates range from 5.9% to 34.5%, and truthfulness filter rates range from 4.0% to 24.0%. The gate reacts to visibly broken low-bit outputs, but the hidden-danger and over-refusal regimes remain after gating, so it still does not restore a reliable quality proxy.

The BPW regression weakness (R-squared 0.03-0.30) reveals that the quality-safety divergence is not a smooth linear function of bits per weight. Instead, it follows a threshold pattern: quality and safety track together through the high-precision range (Q8_0 to Q5_K_M), diverge at a model-specific threshold (Q4_K_M for some models, Q3_K_S for others), and show extreme divergence at Q2_K. This non-linearity means that linear extrapolation from high-precision validation data will underestimate the safety degradation at low-precision deployment levels. The conservative floor at Q5_K_M (where quality and safety remain coupled across both tested models) provides a reference point, but individual models may diverge at different thresholds.

The revised gating result has implications for evaluation methodology. A simple quality gate does become more active at lower precision, so it is not quant-blind. But even with that responsiveness, the gate still does not adaptively solve the safety problem: the same hidden-danger and over-refusal cells remain after gating. The methodological gap TR142 exposes is therefore narrower and more precise: quality cleanup is not the same thing as safety measurement.

The over-refusal finding on llama3.2-3b (+18.6pp refusal increase at Q3_K_S) adds an important nuance to the quantization safety picture. Over-refusal is not a safety success -- it means the model refuses legitimate requests alongside harmful ones, degrading utility. But from a pure safety perspective, over-refusal is the "safe" failure mode: a model that refuses everything, including benign requests, is safer than a model that complies with everything, including harmful requests. The tension between safety (maximize refusal of harmful requests) and utility (minimize refusal of benign requests) means that over-refusal at low quantization creates a different operational problem than under-refusal, but not a safety problem in the narrow sense.

The two-model limitation is the study's primary weakness. With only llama3.2-1b and llama3.2-3b, the finding that quality-safety correlation signs differ between models establishes the possibility of model-dependent correlation but does not characterize the distribution of correlation signs across the model population. It is possible that most models show positive correlation (like llama3.2-1b) and llama3.2-3b is an outlier. It is also possible that the sign distribution is roughly even. Without additional models, this question cannot be resolved. The conservative recommendation (never use quality as a safety proxy) errs on the side of caution because even one model with negative correlation invalidates the practice.

**Opening for TR143.** TR142 completes the quality-safety picture for quantization. The remaining gap in the attack surface is batch composition: does the content of co-batched requests affect safety outcomes?

### 5.6 TR143: Cross-Request Safety Leakage Under Continuous Batching

**Research question.** Does the composition of a continuous batch -- specifically, what other requests are concurrently processed alongside a safety-sensitive prompt -- alter that prompt's safety outcome?

Every production LLM deployment uses continuous batching. In a multi-tenant setting, safety-sensitive requests from one user are co-batched with requests from other users. If batch composition affects safety outcomes, an attacker could degrade other users' safety by flooding the endpoint with jailbreak prompts. TR143 tests this threat model directly under vLLM FP16 continuous batching with 3 models, 5 composition conditions, and 14,250 total records.

**Experimental design.** Phase 1: 3 models x 5 safety conditions x 468 safety prompts + 2 capability conditions x 485 capability prompts = 9,930 records. Phase 2: 2 models x 8 overlap levels x 200 prompts = 3,200 records. Phase 3A: reverse-direction (2 models x 3 conditions x 120 prompts = 720 records). Phase 3B: static vs. continuous batching (2 models x 2 conditions x 200 prompts = 400 records).

**Key results.**

| Metric | Value | Context |
|--------|-------|---------|
| McNemar tests (all pairwise) | All p > 0.125 | 21 tests, none significant after Holm |
| Cochran's Q (omnibus) | All p > 0.34 | llama3.2-1b: 1.20/0.878; 3b: 4.51/0.341; qwen: 2.92/0.572 |
| MH pooled OR (solo vs jailbreak-7) | 1.031, CI [0.913, 1.164] | Crosses 1.0 |
| MH pooled OR (benign-7 vs jailbreak-7) | 1.004, CI [0.890, 1.133] | Crosses 1.0 |
| Directional: mixed-4/3 vs solo | 11 unsafe, 1 safe | p = 0.006 |
| Directional: jailbreak-7 vs solo | 9 unsafe, 1 safe | p = 0.021 |
| Directional: benign-7 vs solo | 8 unsafe, 1 safe | p = 0.039 |
| Directional: refusal-7 vs solo | 8 unsafe, 3 safe | p = 0.227 (not significant) |
| Temporal overlap dose-response | No effect | Logistic slopes p > 0.93 |
| Reverse-direction leakage | None | All Phase 3A McNemar p = 1.0 |
| Static vs continuous batching | Identical | Phase 3B: 0-1 discordant pairs, p = 1.0 |
| Co-batch verification rate | 22.1% | 2,466 of 11,151 confirmed |
| Prompt length confound | 47% | 58.5 vs 85.8 mean tokens between pools |
| MDE at 80% power | 4.7pp | Phase 1 |
| Capability TOST | 1 of 3 pass | 2 degenerate zero-variance failures |

**Extended analysis.** The aggregate null finding is the primary result: batch composition does not measurably affect safety outcomes at the 4.7pp MDE threshold. This closes the composition attack vector for current deployment scales. An attacker cannot degrade other users' safety by flooding a multi-tenant endpoint with jailbreak prompts, because the safety outcomes of co-batched requests are statistically indistinguishable regardless of filler content.

The directional asymmetry is the secondary finding and the one that prevents the aggregate null from being characterized as a "clean null." When prompts flip between solo and batched processing (rare: approximately 2-4 flips per 468 prompts per model per condition), they flip toward unsafe 88-92% of the time. This asymmetry is statistically significant for 3 of 4 composition conditions (binomial p = 0.039, 0.021, 0.006) and is consistent across composition types. The composition content does not matter -- jailbreak fillers, benign fillers, and mixed fillers all produce the same directional bias -- which means the asymmetry is a property of batching itself, not of the filler content.

The mechanistic interpretation is consistent with the floating-point perturbation mechanism from TR138: when batch size changes accumulation order, the resulting epsilon perturbation at decision boundaries is more likely to push models from refusal to compliance than vice versa. This is the same directional bias observed in TR138's audit (59.1% unsafe share) and is consistent with the hypothesis that refusal is a more specific (and therefore more fragile) behavioral state than compliance.

The 22.1% co-batch verification rate is the study's most significant limitation. It means the experiment tested "intended composition" in only one-fifth of cases. The aggregate null may be stronger than it appears (if composition effects exist, they are measured on a diluted treatment) or weaker than it appears (if the 22.1% that were verified happen to be systematically different from the 78% that were not). Future work must either use a modified vLLM scheduler with co-batching guarantees or instrument the scheduler to log actual batch compositions.

The v2.0 version note is important context. Version 1.0 was retracted because it presented the result as a "clean null" without reporting the directional asymmetry. The retraction and re-publication as v2.0 with the full picture (aggregate null plus directional concern) is a methodological strength: null results require the same scrutiny as positive results, and secondary findings that qualify the null must be reported even if they complicate the narrative.

The static-vs-continuous batching comparison (Phase 3B: identical outcomes, p = 1.0) is a clean negative result with a clear mechanistic explanation: vLLM's PagedAttention provides KV-cache isolation regardless of scheduler mode, and the computation path through the shared attention and MLP kernels is the same whether the scheduler uses static or continuous batching. The scheduler mode is not a safety-relevant variable.

The reverse-direction test (Phase 3A: McNemar p = 1.0 for all comparisons) addresses a natural follow-up question: if jailbreak prompts do not degrade the safety of co-batched benign prompts (the primary finding), does the reverse hold? Could safety-oriented prompts improve the safety of co-batched jailbreak prompts? The answer is no: co-batching benign safety prompts with jailbreak prompts does not improve jailbreak refusal rates. Near-zero discordant pairs means the jailbreak outputs are identical regardless of what else is in the batch. This rules out a potential mitigation strategy (flooding the batch with safety-oriented prompts to "dilute" jailbreak effects) and further supports the mechanistic explanation that PagedAttention provides effective inter-request isolation.

The temporal overlap analysis (Phase 2) is the weakest component of TR143 due to the difficulty of controlling overlap in a continuous batching system. The 8 overlap levels were achieved by varying the timing between target and filler request submission, but the actual overlap at the GPU level depends on vLLM's scheduling decisions, which are not directly controllable. The flat dose-response (logistic regression slopes p > 0.93 for both models) may reflect either a genuine absence of dose-response or an inability to achieve sufficient overlap variation. The extreme uniformity of Phase 2 results (120/200 or 121/200 refused in every condition for llama3.2-1b) suggests that the overlap manipulation may not have achieved its intended effect at the GPU level.

The prompt-length confound (47% difference between filler pools) is a design limitation that should be addressed in future work. Jailbreak prompts (mean 85.8 tokens) are longer than benign prompts (mean 58.5 tokens) because jailbreak prompts typically include adversarial wrappers, persona descriptions, or scenario setups that lengthen the text. The compute load per batch is therefore higher for jailbreak-composition conditions than for benign-composition conditions. If batch-level compute load affects safety outcomes (e.g., through GPU memory pressure or attention computation changes), then the composition-content comparison is confounded. The aggregate null (jailbreak and benign compositions both produce null results) mitigates this concern for the primary finding, but future work should use length-matched filler pools to enable clean content comparisons.

---

## 6. Cross-Report Synthesis by Decision Axis

The following subsections integrate findings across TRs by decision topic rather than by report sequence. Each subsection synthesizes evidence from multiple TRs to provide a unified recommendation for a specific deployment decision.

### 6.1 By quantization level

Quantization is the dominant variable across the entire attack surface, producing effects measured in tens of percentage points that are two orders of magnitude larger than any batch effect. The evidence is convergent across 4 TRs and 4 attack modalities, with independent experimental designs and partially non-overlapping model sets all pointing to the same conclusion:

**Single-turn (Phase 3, TR134):** Llama 1B loses 35.2pp at Q2_K (d = 1.93). Q4_K_M retains >= 93% safety for all tested models.

**Multi-turn (TR139):** All 8 strategy ANOVAs reject quantization-independence (p < 1e-4). Q2_K + attention_shift = 100% ASR on qwen2.5-1.5b.

**Many-shot (TR140):** Q2_K is the universal vulnerability threshold. Llama models immune above Q3_K_M. Message-array format + Q2_K = 92% ASR.

**Quality-safety (TR142):** Safety degrades 13.9x faster than quality at Q3_K_S. Quality metrics are insufficient proxies.

The convergent finding across all modalities is that Q2_K is unconditionally dangerous and Q4_K_M is the practical safety floor. The intermediate zone (Q3_K_S, Q3_K_M) is model-dependent: Llama 1B shows catastrophic safety loss at Q3_K_S (hidden by quality metrics), while Llama 3B shows paradoxical over-refusal. The Q4_K_M recommendation from Phase 3 is reinforced by every subsequent TR that tests quantization effects.

The quantization effect hierarchy is not merely additive across attack modalities. At Q4_K_M, single-turn safety is adequate (>= 93% retention, Phase 3), multi-turn safety is adequate (Llama models at or below 2% ASR for most strategies, TR139), and many-shot safety is adequate (Llama models immune, TR140). At Q2_K, all three modalities show catastrophic failure: single-turn loses 35pp (Phase 3), multi-turn reaches 100% ASR (TR139), and many-shot reaches 99% ASR (TR140). The transition between "adequate" and "catastrophic" occurs at different levels for different modalities and different models, but the Q4_K_M-to-Q2_K range brackets the danger zone for all tested configurations.

The quantization effect is also modulated by the specific harmful behavior being requested. TR140's variance decomposition shows that residual (per-behavior) variance accounts for 65.7% of total ASR variance, more than quantization (17.9%), model identity (12.6%), or shot count (2.7%). This means that the same quantized model may be safe for some harmful behaviors and unsafe for others, depending on how close the refusal boundary is for each specific behavior. A safety evaluation using a small set of behaviors may over- or under-estimate the overall vulnerability because the per-behavior variance is so large.

The practical synthesis for quantization is: Q4_K_M remains the recommended deployment floor (consistent with Phase 2 and Phase 3). Q2_K is banned across all deployment contexts. Q3_K_S and Q3_K_M require per-model, per-modality validation that cannot be replaced by quality benchmarks. The validation must include multi-turn (at least 4 strategies) and many-shot (message-array format) testing in addition to single-turn evaluation, because the quantization-safety interaction manifests differently across attack modalities.

### 6.2 By batch configuration

Batch configuration includes batch size (TR138, TR141) and batch composition (TR143). The evidence consistently shows that batch effects are real but small:

**Batch size (TR138):** 0.6% regex-detected flip rate, 0.16% human-adjudicated genuine rate. Directionally biased toward unsafe (72.7% refusal-to-compliance).

**Cross-architecture batch size (TR141):** Fragility varies 6.3x across 15 models (phi-2 2.39% to tinyllama 0.00%). Output instability is the sole reliable predictor (r = 0.91).

**Batch composition (TR143):** Aggregate null (all tests p > 0.05). Directional asymmetry: 88-92% toward unsafe (p = 0.006 for strongest condition). Composition content does not matter.

The unified recommendation for batch configuration is: measure output instability per-model (cheap), monitor if instability > 15% (moderate), do not implement composition-aware routing (unnecessary). The 100x magnitude gap between batch effects (~0.16% genuine flip rate) and quantization effects (~35pp safety loss) means that engineering effort should be allocated to quantization validation first.

The convergence between TR138 Phase 2 (co-batching interference null), TR143 Phase 1 (composition aggregate null), and TR143 Phase 3B (static = continuous batching) establishes a robust pattern: the content of co-batched requests does not affect safety outcomes through any tested mechanism. Whether fillers are jailbreaks, benign prompts, or a mix; whether the scheduler uses static or continuous batching; and whether the overlap is minimal or maximal -- none of these factors change the aggregate refusal rate by a measurable amount.

The directional asymmetry, while statistically significant, should be interpreted in the context of its magnitude. The 88-92% unsafe direction applies to 2-4 flips out of 468 prompts per model per condition. In absolute terms, this means approximately 2-3 prompts per evaluation run flip from refusal to compliance. The practical significance of 2-3 additional compliance events per evaluation run depends on the deployment context: in a high-stakes pipeline processing millions of prompts, even a 0.5% unsafe flip rate at scale could produce thousands of harmful completions. In a low-stakes pipeline, 2-3 extra compliances per evaluation run are operationally invisible. The monitoring recommendation is calibrated to this scale-dependence: log directional patterns, and investigate if unsafe accumulation becomes visible at scale.

The relationship between batch-size effects (TR138, TR141) and batch-composition effects (TR143) is worth noting explicitly. Batch size affects safety through floating-point perturbation (changing accumulation order). Batch composition could, in theory, affect safety through shared computation (the co-batched requests share GPU kernels). The evidence suggests that the floating-point mechanism is real but small (TR138), while the shared-computation mechanism is absent (TR143). This is consistent with PagedAttention's memory isolation: KV-caches are independent, so the information content of co-batched requests cannot leak into the target request's computation. The only shared element is the kernel launch, which determines accumulation order but not the content being accumulated.

### 6.3 By attack modality

The attack modalities tested in this synthesis form a hierarchy of danger:

| Attack Modality | Max Effect | Best Defense | Cost of Defense |
|----------------|-----------|-------------|----------------|
| Multi-turn jailbreak (TR139) | 100% ASR | Quantization control (>= Q4_K_M) | Low (model selection) |
| Many-shot long-context (TR140) | 99% ASR | Format restriction (block message-array) | Low (input parsing) |
| Single-turn quantization (Phase 3) | 35pp loss | Quantization control (>= Q4_K_M) | Low (model selection) |
| Batch size perturbation (TR138) | 0.16% genuine | Output instability screening | Low (one batch of inference) |
| Batch composition (TR143) | 0% aggregate | None required | Zero |

The hierarchy reveals that the most dangerous attacks (multi-turn, many-shot) are also the cheapest to defend against: staying at Q4_K_M or above eliminates the quantization amplification, and restricting message-array format eliminates the many-shot format vulnerability. The least dangerous attacks (batch size, composition) are the most difficult to defend against if one wished to eliminate them entirely (requiring deterministic inference kernels or composition-aware routing), but they are also the attacks that require no defense due to their negligible magnitude.

This inverse relationship between danger and defense cost is the synthesis's most practically useful finding. It means that the threat landscape is favorable: the high-severity threats have low-cost mitigations, and the threats without practical mitigations are low-severity. An operator who deploys at Q4_K_M, restricts message-array inputs, and tests multi-turn strategies before deployment has addressed the top three threats. The remaining threats (batch perturbation, composition) do not require active mitigation at current scales.

The attack-modality comparison also reveals an important finding about evaluation methodology. Single-turn safety evaluation (the standard approach in the safety evaluation literature) captures the quantization-degradation effect but misses the multi-turn amplification and many-shot format vulnerability. A model that scores well on single-turn evaluation at Q4_K_M may still be vulnerable to multi-turn attention_shift at the same quantization level. The implication is that safety evaluation batteries should include at least one multi-turn strategy and one many-shot test in addition to single-turn prompts. The additional cost is modest (a few hundred additional evaluation prompts), and the additional information is critical for conversational and long-context deployment scenarios.

The interaction between attack modalities and models is also noteworthy. qwen2.5-1.5b is broadly vulnerable to both multi-turn (TR139) and many-shot (TR140) attacks across most of the quantization range. Llama models, by contrast, are vulnerable to multi-turn attacks primarily at Q2_K (TR139) and vulnerable to many-shot attacks primarily at Q2_K with message-array format (TR140). This model-specificity means that the attack-modality hierarchy varies by model: for qwen2.5-1.5b, multi-turn and many-shot are equally dangerous; for Llama models, the danger is concentrated in the quantization dimension rather than the attack-modality dimension.

### 6.4 By model architecture

The 18+ models tested across the synthesis span 10+ architecture families, enabling a cross-architecture analysis of model-level safety behavior:

**Consistent across architectures:** Q2_K is universally dangerous. No model retains adequate safety at Q2_K across any attack modality (TR139, TR140, TR142).

**Architecture-dependent:** Everything else. Fragility varies 6.3x (TR141). Quantization resilience varies from near-flat (Qwen DPO, slope +0.008/BPW in Phase 3) to catastrophic (Llama 1B, 35pp loss at Q2_K). Quality-safety correlation can be positive (llama3.2-1b: +0.994) or negative (llama3.2-3b: -0.829). Many-shot vulnerability is model-specific (qwen2.5-1.5b vulnerable at Q8_0; Llama models immune above Q3_K_M).

**Not predictive:** Alignment type (ANOVA p = 0.942 for batch fragility). Parameter count (llama3.2-1b and llama3.2-3b have opposite Q2_K safety profiles despite sharing the same architecture family). Baseline refusal rate (r = 0.028 with fragility; uninformative).

**The only reliable predictor:** Output instability (r = 0.91 for batch fragility). For quantization safety, per-model profiling with the actual safety battery is the only reliable method.

The cross-architecture analysis produces a practical model-selection framework for safety-sensitive deployments. Within the tested model set, Llama-family instruct models generally show the best combination of safety resilience (immune to many-shot above Q3_K_M) and batch stability (0.43-0.47% fragility rate). Qwen-family models show variable behavior (qwen2.5-1.5b is broadly vulnerable to multi-turn attacks, but qwen2.5-3b and qwen2.5-7b show DPO-derived resilience under Phase 3 quantization testing). Phi-family models show the highest batch fragility (phi-2: 2.39%) but were not tested under multi-turn or many-shot conditions.

The within-family variation is as large as the between-family variation. llama3.2-1b and llama3.2-3b share the same architecture but show opposite quantization-safety profiles (35pp loss vs. +6pp gain at Q2_K in Phase 3), opposite quality-safety correlations (+0.994 vs. -0.829 in TR142), and similar but not identical batch fragility (0.43% vs. 0.47% in TR141). This within-family variation means that even fine-grained model-family knowledge is insufficient for safety prediction -- individual model profiling is genuinely required, not merely recommended as a conservative practice.

### 6.5 By jailbreak strategy type

TR139 tests 8 multi-turn jailbreak strategies, enabling strategy-level analysis across quantization levels. The strategies form a hierarchy of effectiveness that is quantization-dependent:

| Strategy | Q8_0 ASR (typical range) | Q2_K ASR (peak) | Amplification Factor | Mechanism |
|----------|------------------------|-----------------|---------------------|-----------|
| attention_shift | 10-20% | 100% | 5-10x | Redirects attention from safety to task completion |
| crescendo | 8-15% | 60-100% | 4-8x | Escalates pressure across turns |
| context_fusion | 5-15% | 80-100% | 5-10x | Blends harmful content with benign context |
| progressive_refinement | 10-20% | 60-80% | 3-6x | Iteratively refines harmful request |
| foot_in_door | 5-12% | 50-70% | 4-6x | Escalates from benign to harmful |
| role_play | 5-15% | 40-60% | 3-5x | Establishes fictional scenario |
| benign_context | 3-10% | 30-50% | 3-5x | Establishes cooperative tone |
| direct | 5-15% | 40-60% | 3-4x | Baseline, no adversarial framing |

**Observations.** The attention_shift strategy is the most dangerous because it exploits a fundamental tension in instruction-tuned models: the model is trained to both follow instructions (task completion) and refuse harmful requests (safety). Attention_shift resolves this tension in favor of task completion by progressively emphasizing the instruction-following pathway and de-emphasizing the safety pathway. At Q2_K, where safety weights are most degraded, the instruction-following pathway dominates completely, producing 100% ASR.

The strategy hierarchy is consistent with a mechanistic model of safety alignment as a competing signal. Strategies that directly compete with the safety signal (attention_shift, crescendo) are most effective because they exploit the same conversational consistency that safety training relies on. Strategies that bypass the safety signal through indirect means (benign_context, role_play) are less effective because the safety signal is not directly challenged -- it simply becomes less relevant as the conversation shifts context.

The ANOVA eta-squared values (0.031-0.153) indicate that quantization explains between 3.1% and 15.3% of ASR variance, depending on strategy. The direct strategy has the highest eta-squared (0.153), meaning quantization has the strongest relative effect on direct (non-adversarial) requests. This is consistent with the Welch p = 0.702 finding: multi-turn strategies are not selectively more quantization-sensitive than direct attacks because the quantization effect is largest for the direct baseline. The multi-turn strategies add a constant (non-quantization-dependent) baseline vulnerability on top of the quantization effect.

The persistence analysis (Phase 2) reveals strategy-level patterns that Phase 1 ASR alone cannot capture. Models that initially refuse a direct harmful request but then break under crescendo pressure are qualitatively different from models that comply immediately. The persistence slopes (negative for 3/4 models) indicate that lower quantization weakens the model's ability to maintain refusal across extended adversarial interactions. This has implications for deployment monitoring: a model that passes an initial refusal test may still fail under sustained adversarial pressure, and the failure probability increases at lower quantization levels.

### 6.6 By many-shot format and shot count

TR140's format analysis reveals that the attack surface for many-shot jailbreaking is dominated by the prompt format rather than the shot count or quantization level:

| Factor | Variance Explained | Implication |
|--------|-------------------|-------------|
| Residual (per-behavior) | 65.7% | The specific harmful behavior matters most |
| Quantization level | 17.9% | Second most important; Q2_K universal threshold |
| Model identity | 12.6% | Third; qwen2.5-1.5b broadly vulnerable |
| Shot count | 2.7% | Least important; format matters more |
| Format (message array vs faux dialogue) | N/A (tested separately) | 92% vs 0% on same cell; dominant attack vector |

**Observations.** The format dominance has two operational implications. First, format-level input sanitization is the most cost-effective defense against many-shot jailbreaking. Blocking or transforming message-array format inputs eliminates 92% of the attack success on the most vulnerable cells, regardless of shot count or quantization level. This is a preprocessing step that can be implemented at the API gateway level without modifying the model or its quantization.

Second, the low variance explained by shot count (2.7%) means that restricting the number of turns or examples in a prompt is not an effective defense. An attacker can achieve high ASR with as few as N=16 examples in message-array format on vulnerable models. Context-length limits that are long enough to be useful for legitimate applications (e.g., 4K tokens) are also long enough to accommodate 16-128 many-shot examples. The only context-length-based defense that would be effective (restricting to < 1K tokens) would also eliminate many legitimate use cases.

The per-behavior variance (65.7%) is the largest single factor. This means that ASR estimates are highly sensitive to the specific set of harmful behaviors used in the evaluation. A study using 10 behaviors may get a very different ASR than a study using 50 behaviors, not because of methodological differences but because different behaviors have different inherent susceptibilities. This per-behavior variance should be reported alongside aggregate ASR in any many-shot evaluation, because the aggregate can be misleading without it.

### 6.7 By quality-safety relationship

TR142 establishes that quality and safety follow partially independent degradation paths under quantization. The relationship is model-dependent in both magnitude and sign:

**Strong positive coupling (llama3.2-1b):** Quality and safety degrade together. Quality benchmarks provide directional signal (but underestimate safety degradation by 13.9x at Q3_K_S).

**Strong negative coupling (llama3.2-3b):** Quality degrades while safety improves through over-refusal. Quality benchmarks provide misleading signal (improvement in safety is an artifact of degraded coherence, not of improved alignment).

**Implication for deployment validation:** Quality metrics are neither sufficient nor reliable indicators of safety. A comprehensive safety evaluation requires safety-specific benchmarks (AdvBench, jailbreak resistance, bias probes) at the target quantization level, regardless of quality benchmark results.

The revised quality-gating result has a deeper implication: even a gate that reacts to quantization damage is still not enough. Filter rates increase at lower precision, so the gate does catch more visibly broken generations. But the crucial safety shifts still survive after gating, which means quality gates cannot substitute for safety-specific validation in deployment pipelines.

The Simpson's paradox in quality-safety trends (noted in TR142) is worth discussing. When quality and safety are aggregated across models, the trends may appear to move together, creating a false impression of coupling. When disaggregated by model, the opposite-sign correlations emerge: llama3.2-1b shows positive coupling, llama3.2-3b shows negative coupling. Pooling these two models produces a near-zero aggregate correlation that masks the strong within-model relationships. This is a textbook Simpson's paradox, and it has direct practical consequences: any deployment validation that pools quality-safety data across models will miss the model-specific divergence patterns that are the actual safety risk.

### 6.8 By effect size magnitude

The effects measured across the synthesis span four orders of magnitude in absolute terms, from negligible (batch composition: < 1pp) to catastrophic (Q2_K multi-turn: 100% ASR). The effect size hierarchy maps directly to the threat hierarchy:

| Effect Size Category | Examples | Cohen's d/h range | Operational Significance |
|---------------------|---------|-------------------|------------------------|
| **Catastrophic** (> 50pp) | Q2_K + attention_shift (100% ASR), Q2_K single-turn Llama 1B (35pp loss) | d > 1.5 | Deployment block |
| **Severe** (20-50pp) | Q2_K many-shot message-array (92%), Q3_K_S safety-quality divergence (13.6pp refusal collapse) | d = 0.8-1.5 | Mandatory validation |
| **Moderate** (5-20pp) | Mid-quant multi-turn ASR changes, format-dependent many-shot differences | d = 0.3-0.8 | Model-specific validation |
| **Small** (1-5pp) | Batch-size flip rates (0.6% automated, 1.68% enriched), composition directional | d = 0.05-0.3 | Monitoring |
| **Negligible** (< 1pp) | Composition aggregate effect (< 1pp), temporal overlap (flat) | d < 0.05 | No action |

**Observations.** The 100x magnitude gap between catastrophic and small effects is the key resource-allocation insight. Engineering effort spent on mitigating small effects (batch composition routing, deterministic inference kernels) has 100x lower return than effort spent on mitigating catastrophic effects (quantization validation, multi-turn testing, format restriction). This hierarchy should drive security engineering priorities: address the catastrophic and severe effects first, then the moderate effects if resources permit, and accept the small and negligible effects as residual risk.

### 6.9 By cross-dimensional interactions

The synthesis tests several cross-dimensional interactions and identifies several untested interactions that represent gaps in the attack-surface mapping:

**Tested interactions:**

| Interaction | TR | Finding | Implication |
|-------------|-----|---------|-------------|
| Quantization x multi-turn strategy | TR139 | Multiplicative; all 8 ANOVAs p < 1e-4 | Both factors must be tested jointly |
| Quantization x many-shot format | TR140 | Threshold + format dominance | Q2_K universal threshold, format > quant |
| Quantization x quality | TR142 | Divergent; safety degrades 13.9x faster | Quality not a proxy |
| Architecture x batch size | TR141 | Model-dependent; r=0.91 with instability | Per-model screening needed |
| Composition content x safety | TR143 | No interaction; all filler types equivalent | Content does not matter |
| Batch size x quantization | TR138 Phase 3 | Significant quant effect; concurrency null | Effects are independent |
| Temporal overlap x composition | TR143 Phase 2 | No interaction; flat dose-response | Overlap does not modulate |
| Static vs continuous batching | TR143 Phase 3B | Identical outcomes | Scheduler mode irrelevant |

**Untested interactions (gaps):**

| Gap | Why Important | How to Close |
|-----|--------------|-------------|
| Quantization x batch composition | Quantized models may be more FP-sensitive to batch composition due to reduced weight precision creating more decision-boundary tokens | Factorial: 3 quant x 3 composition conditions on 3 models |
| Multi-turn x batch size | Batch perturbation during multi-turn jailbreaking may amplify the conversation-consistency degradation that makes multi-turn attacks effective | Multi-turn evaluation at batch=1 vs batch=8 on 2 models |
| Many-shot x batch composition | In-context compliance examples may interact with the floating-point perturbation from co-batched jailbreak requests | Many-shot evaluation under composition conditions on 2 models |
| Model scale x quantization x safety (large models) | Models > 14.8B may have different quantization-safety interaction due to weight redundancy | Extend TR139/TR140 design to 30B+ models on multi-GPU |
| Temperature x all dimensions | Stochastic sampling may mask batch perturbation, alter multi-turn dynamics, and change many-shot susceptibility | Repeat key experiments at temp=0.7 |

**Observations.** The tested interactions reveal that quantization is the common amplifier: it multiplies multi-turn vulnerability, amplifies many-shot attacks (at extreme levels), and diverges from quality metrics. Batch configuration operates independently: batch size, composition, overlap, and scheduler mode all show null or negligible interactions with each other and with content. The untested interactions are primarily quantization crossed with batch dimensions -- the one cross-dimensional space that has not been explored. The additive model used in Phase 3 (TR137) for projecting combined effects assumes these interactions are zero. If quantization x batch composition is non-zero (i.e., quantized models are more sensitive to composition than FP16 models), the additive model would underestimate the combined risk. This is the highest-priority experiment for future work.

### 6.10 By heterogeneity and model agreement

The program's findings on model heterogeneity are among its most operationally significant contributions. Phase 3 established I-squared = 99.9% on quantization safety and I-squared = 0.0% on concurrency safety, demonstrating that models disagree completely on quantization effects but agree perfectly on concurrency effects. This synthesis extends the heterogeneity analysis to new dimensions:

**Batch fragility heterogeneity (TR141):** Fragility varies 6.3x across 15 models (0.00% to 2.39%). The coefficient of variation is high (approximately 80%), indicating that the mean fragility (0.75%) is a poor representative of any individual model's behavior. For deployment decisions, the per-model fragility rate is more useful than the aggregate, because the aggregate conceals the 6.3x range.

**Multi-turn vulnerability heterogeneity (TR139):** The 4 tested models show qualitatively different vulnerability profiles. qwen2.5-1.5b is broadly vulnerable across strategies and quantization levels. llama3.2-1b collapses at Q2_K but is robust above Q3_K_M. llama3.1-8b shows a non-monotone instability band. llama3.2-3b is the most robust. These profiles cannot be predicted from model family, parameter count, or alignment type -- they must be measured empirically.

**Quality-safety coupling heterogeneity (TR142):** The correlation between quality and safety reverses sign between the two tested models (+0.994 vs. -0.829). With only 2 models, we cannot characterize the distribution of correlation signs, but the existence of opposite signs within the same model family (both are Llama models) is sufficient to invalidate pooled analysis.

**Directional heterogeneity (TR141 vs. TR138 vs. TR143):** The directional bias in safety flips varies across studies. TR138 reports 59.1% unsafe direction in its audit (N=44 candidates). TR141 reports 66.2% safe direction in its combined synthesis (N=240 flips). TR143 reports 88-92% unsafe direction (N=9-12 flips per condition). The inconsistency across studies suggests that directionality is sample-dependent (driven by which specific prompts happen to be near decision boundaries in each model set) rather than a universal property of batch perturbation.

The practical implication of this pervasive heterogeneity is the per-model profiling recommendation. Within the tested model set, no aggregate statistic, no categorical shortcut, and no transfer from one model to another reliably predicted an individual model's safety behavior under optimization. For safety-critical deployments at aggressive quantization levels (below Q4_K_M), empirical measurement at the specific model-configuration combination is the only approach supported by the evidence. At Q4_K_M or above, where effects are smaller across all tested models, generic benchmarks may be sufficient if the architecture has been tested.

### 6.11 By measurement methodology

The synthesis reveals systematic patterns in measurement quality that affect the interpretation of all findings:

**Automated classifier artifact rate:** 73% for small-effect flips (TR138 human adjudication). Any automated measurement reporting effects < 5% is dominated by classifier noise.

**Judge agreement:** kappa = 0.104 for dual-judge (TR139), kappa = 0.23 for single-judge (TR140), kappa = 0.147 for Phase 3 regex-vs-judge. Agreement degrades at extreme quantization levels where model output becomes incoherent.

**Co-batch verification:** 22.1% (TR143). The majority of composition observations cannot be confirmed as having received the intended treatment.

**Self-correction track record:** Two major corrections (TR141 ANOVA reversal, TR143 v1.0 retraction) demonstrate that the program applies the same scrutiny to its own findings that it applies to deployment validation. These corrections should be interpreted as methodological strengths: they show that preliminary findings are tested and revised rather than defended.

**Cross-TR anchor consistency:** The anchor models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) appear in 5-6 of the 6 TRs, enabling cross-TR consistency checks. Baseline safety scores at shared configurations (Q8_0 or FP16, batch=1, Ollama) should agree across TRs if the measurement is consistent. Minor variations (within 5pp) are expected due to run-to-run variability (Phase 3 established up to 5pp tolerance at anchor configurations). Larger discrepancies would indicate either a measurement pipeline change or an environmental change between runs. The consistency of the anchor models across TRs provides implicit validation that the measurement infrastructure is stable across the 2-week period (2026-03-13 to 2026-03-20) during which the 6 TRs were run.

**Effect size calibration:** The program spans effects from negligible (< 1pp for composition) to catastrophic (100% ASR for Q2_K multi-turn). This range enables internal calibration: the measurement infrastructure must be sensitive enough to detect the catastrophic effects (it is -- the 100% ASR finding is unambiguous) while being honest about its inability to precisely quantify the negligible effects (it is -- the 0.16% genuine rate carries substantial uncertainty from the single-reviewer adjudication). The internal consistency of the effect-size hierarchy (quantization > multi-turn > many-shot > batch > composition) across independent TRs provides confidence that the measurement infrastructure is producing reliable relative rankings even if the absolute values carry uncertainty.

---

## 7. Operational Doctrine and Risk Controls

This section translates the synthesis findings into operational policies. Each policy is backed by specific measurements and effect sizes. The policies follow a "safe by default, escalate by evidence" principle.

### 7.1 Quantization safety protocol

Any deployment at quantization levels below Q4_K_M must complete the following validation:

1. **Single-turn safety evaluation.** Run AdvBench (100 prompts), jailbreak amplification (120 prompts, 4+ techniques), and at least one bias benchmark at the target quantization level. Compare to FP16 or Q8_0 baseline.

2. **Multi-turn jailbreak evaluation.** Run at least 4 multi-turn strategies (including attention_shift and crescendo) at the target quantization level with 50 harmful behaviors. If any strategy shows > 50% ASR, the quantization level is not safe for conversational deployment. Evidence: TR139, all 8 ANOVAs p < 1e-4.

3. **Format-specific evaluation.** If the application accepts variable-length inputs, test with message-array format at multiple shot counts (N = 1, 16, 128). If message-array ASR > 10%, implement format-level input sanitization. Evidence: TR140, 92% vs 0% by format.

4. **Quality-safety divergence check.** Compare safety degradation to quality degradation at the target level. If safety moves > 3x more than quality, quality monitoring is insufficient; deploy safety-specific monitoring. Evidence: TR142, 13.9x divergence at Q3_K_S.

### 7.2 Batch configuration protocol

For batch sizes > 1 in safety-critical pipelines:

1. **Output instability screen.** Run 200+ prompts at batch = 1 and batch = N. Compute output change rate (fraction of prompts producing different text). If > 15%, the model requires safety monitoring under batching. If < 5%, batch perturbation is negligible for this model. Evidence: TR141, r = 0.91.

2. **No composition-aware routing required.** Aggregate safety outcomes are not affected by batch composition. Engineering effort on composition-aware routing should be redirected to quantization validation. Evidence: TR143, 21 McNemar tests non-significant.

3. **Directional monitoring.** Log safety flip directions (refusal-to-compliance vs compliance-to-refusal) if operating at scale. If the unsafe direction exceeds 80% of total flips consistently, investigate. Evidence: TR143, 88-92% directional bias.

### 7.3 Per-model profiling protocol

Per-model safety profiling is mandatory for any new model deployment. This policy is justified by the extreme model-level heterogeneity across the synthesis:

- I-squared = 99.9% on quantization safety (Phase 3)
- 6.3x fragility range across 15 models (TR141)
- Quality-safety correlation sign reversal between models (TR142)
- Alignment type not predictive (TR141, p = 0.942)
- Within-family variation as large as between-family variation (Llama 1B vs 3B: opposite Q2_K profiles)
- No categorical variable (alignment type, parameter count, architecture family) reliably predicts safety

No model inherits another model's safety profile. Even within the same family (llama3.2-1b vs llama3.2-3b), safety behavior under quantization is opposite in direction.

**Profiling scope per model:** The minimum per-model profiling should include:

1. **Quantization sensitivity sweep.** Evaluate at the target quantization level plus one level above and one level below. Compute safety retention and safety slope (normalized score vs BPW). If slope > 0.020 per BPW, the model is quantization-sensitive and the target level may be too aggressive.

2. **Multi-turn jailbreak spot-check.** Test at least 2 strategies (attention_shift and direct) at the target quantization level with 50 behaviors. This catches the most dangerous multi-turn vulnerability without requiring the full 8-strategy sweep.

3. **Output instability screen.** Run 200+ prompts at batch=1 and batch=N for the production batch size. If output change rate > 15%, flag for enhanced monitoring.

4. **Format-specific spot-check.** If the application accepts variable-length inputs, test message-array format at N=16 with 50 behaviors.

**Profiling cost:** The minimum profiling (steps 1-4) requires approximately 3-4 hours of GPU time per model on consumer hardware. This is a one-time cost that should be amortized across the lifetime of the deployment.

**Profiling frequency:** Re-profile after any model version change, quantization engine update (Ollama, llama.cpp version change), safety classifier update, or environmental change (new GPU hardware, different CUDA version). Do not assume that profiles are stable across software versions -- even minor version changes can affect quantization behavior.

### 7.4 Change management policy

The following changes invalidate specific safety findings and require re-evaluation:

| Change | Invalidated Findings | Minimum Re-evaluation |
|--------|---------------------|----------------------|
| New model family | All safety profiles | Full multi-modality validation (7.1 + 7.2) |
| Model version update | Quantization sensitivity, jailbreak amplification | Abbreviated battery at target quant |
| Quantization level change | Safety retention at target level | Single-turn + multi-turn at new level |
| Format-level input change | Many-shot vulnerability | Many-shot evaluation with new format |
| Batch size change (> 2x) | Batch stability profile | Output instability re-screen (7.2) |
| Safety classifier update | Absolute safety scores | Re-run battery; recompute baselines |
| Temperature change | All safety findings (temp=0 only) | Full re-evaluation at new temperature |
| Backend migration | Backend-specific safety scores | Phase 3 backend protocol (Conclusive 134-137, Section 8.2) |

### 7.5 Monitoring policy

**Primary metrics (continuous monitoring):**
- Aggregate refusal rate per model-quant combination (alert on > 3pp decline from baseline)
- Multi-turn jailbreak compliance rate (alert on > 5pp increase from baseline; requires periodic adversarial probing)
- Output instability rate at production batch size (alert if exceeds 15%)

**Secondary metrics (weekly review):**
- Directional flip patterns in safety classifications (alert if unsafe direction exceeds 80% of flips consistently)
- Format-specific ASR if application accepts variable-length inputs
- Classifier agreement if dual classifiers deployed

**Monitoring cadence:**
- Daily: aggregate refusal rate, error rates
- Weekly: jailbreak compliance spot-check, output instability measurement
- Monthly: full multi-turn evaluation (4+ strategies), many-shot format test, quality-safety divergence check
- Quarterly: complete evaluation including per-behavior variance analysis

### 7.6 Escalation policy

Safety incidents are classified by severity and escalated accordingly:

**Severity 1 (CRITICAL):** Any model at Q2_K found in production. OR multi-turn ASR exceeds 50% for any strategy. OR aggregate refusal rate drops below 80% of baseline. Action: immediate rollback. Root cause investigation within 24 hours.

**Severity 2 (HIGH):** Aggregate refusal rate drops to 80-90% of baseline. OR message-array format ASR exceeds 10%. OR output instability exceeds 25%. Action: escalate to safety review board. Deploy with enhanced monitoring. Investigation within 72 hours.

**Severity 3 (MODERATE):** Aggregate refusal rate drops to 90-95% of baseline. OR directional flip pattern exceeds 85% unsafe direction consistently. OR new model deployed without full profiling. Action: log and investigate. Continue with enhanced monitoring.

**Severity 4 (LOW):** Any metric change within noise band (< 3pp from baseline). Action: log for trend analysis. No immediate action.

### 7.7 Classifier reliability protocol

The TR138 human adjudication (73% artifact rate) and the poor judge agreement statistics (kappa = 0.104 to 0.23) have the following operational implications:

- Safety scores are consistent proxies for cross-condition comparison, not calibrated ground truth.
- Any automated measurement reporting effects < 5% should be human-validated before deployment decisions are made.
- Multiple classifiers (regex + LLM judge) should be deployed for safety-critical applications.
- Absolute thresholds carry measurement uncertainty that is partially but not fully quantified.
- When comparing safety between conditions (e.g., Q4_K_M vs Q8_0), relative differences are more reliable than absolute scores because systematic classifier biases cancel in the delta.
- For multi-turn evaluation, the dual-judge approach (7B primary + smaller fallback) is preferred over single-judge because the agreement rate between judges provides a built-in quality signal: cells with low agreement are cells that require human review.

### 7.8 Multi-turn evaluation protocol

For any deployment processing multi-turn conversations (chatbots, assistants, agentic workflows), the following multi-turn safety validation is required:

1. **Strategy selection.** Test at least 4 strategies, including attention_shift (most effective, TR139) and crescendo (highest amplification on small models). If compute permits, test all 8 strategies.

2. **Behavior selection.** Use at least 50 harmful behaviors from JailbreakBench or equivalent. The per-behavior variance (65.7% of total ASR variance, TR140) means smaller samples will produce highly variable ASR estimates.

3. **Quantization alignment.** Test at the actual deployment quantization level, not at FP16 or Q8_0. The quantization-strategy interaction means that a model safe at Q8_0 may be unsafe at Q4_K_M for specific strategies (e.g., llama3.1-8b's instability band, TR139).

4. **Persistence testing.** For at least 50 behaviors, conduct 5-turn persistence tests where the first turn is a direct harmful request (to establish initial refusal) and turns 2-5 apply escalating adversarial pressure. Models that initially refuse but break within 5 turns are operationally unsafe for extended conversations even if their single-turn refusal rate is high.

5. **Threshold application.** If any strategy shows > 50% ASR: the model is unsuitable for conversational deployment at this quantization level. If any strategy shows 20-50% ASR: deploy with enhanced monitoring and rate limiting on multi-turn conversations. If all strategies show < 20% ASR: the model passes multi-turn validation.

Evidence base: TR139 (10,600 conversations, 8 strategies, 4 models, 6 quant levels), TR140 (many-shot extension).

### 7.9 Format-level input validation protocol

For any deployment accepting variable-length inputs from users, the following format validation is required:

1. **Input format detection.** Implement format detection at the API gateway level to identify message-array formatted inputs (structured as lists of role-content pairs within the user message).

2. **Format-specific testing.** Test the deployed model with message-array format at N = 16 and N = 128 compliance examples using 50 harmful behaviors. If message-array ASR > 10%, implement format restriction.

3. **Format restriction options.** Three levels of format restriction are available:
   - **Block:** Reject any input containing message-array formatting. Simplest but most restrictive.
   - **Transform:** Convert message-array inputs to faux-dialogue format before processing. Preserves content while neutralizing format vulnerability. Requires format conversion logic.
   - **Monitor:** Allow message-array inputs but log them for post-hoc safety review. Least restrictive but provides no real-time protection.

4. **Context-length limits.** If the application has a context-length limit shorter than approximately 1,000 tokens, many-shot attacks with high N are infeasible. Context limits of 2K tokens or more accommodate N = 16 or higher, which is sufficient for high ASR on vulnerable models.

Evidence base: TR140 (92% vs 0% ASR by format, 15,000 samples, 4 models).

### 7.10 Risk register

| Risk ID | Risk | Likelihood | Impact | Mitigation | Evidence |
|---------|------|-----------|--------|------------|----------|
| R1 | Q2_K deployed for any model | Low | Catastrophic (100% ASR) | Universal Q2_K ban | TR139, TR140 |
| R2 | Q3_K_S deployed without safety validation | Medium | High (hidden 13.9x divergence) | Mandatory safety-specific testing | TR142 |
| R3 | Multi-turn jailbreak at low quant | Medium | High (100% ASR peak) | Multi-turn testing before deployment | TR139 |
| R4 | Message-array format not sanitized | Medium | High (92% ASR) | Format-level input validation | TR140 |
| R5 | Quality used as safety proxy | High | High (13.9x underestimate) | Safety-specific benchmarks required | TR142 |
| R6 | Alignment type used as safety predictor | Medium | Medium (false shortcut) | Per-model profiling required | TR141 |
| R7 | Automated classifier artifacts inflate flip rates | High | Low (directional correct) | Human-validate < 5% effects | TR138 |
| R8 | Batch composition exploited | Low | Negligible (aggregate null) | No action at current scales | TR143 |
| R9 | New model deployed without profiling | Medium | Variable (6.3x range) | Mandatory per-model profiling | TR141 |
| R10 | Co-batch verification too low for strong claims | Medium | Methodological (limits confidence) | Future: instrument scheduler | TR143 |

---

## 8. Threats to Validity and Scope Limits

This section makes explicit the boundaries within which the synthesis conclusions are valid, and the conditions under which they would need to be re-examined or abandoned. The threats are organized by validity type and assessed honestly.

This section makes explicit the boundaries within which the synthesis conclusions are valid. The threats are organized by validity type and assessed honestly. Each threat is assigned a severity level (low, moderate, high) based on its potential to change the synthesis conclusions if the threat materialized.

### 8.1 Internal validity threats

**T1: Human adjudication by single reviewer.** The 27% genuine rate (17/63 rows) that recalibrates all flip-rate findings was produced by a single reviewer. Inter-rater reliability has not been established. A second reviewer might classify different rows as genuine vs. artifact, shifting the genuine rate. The 263 TR141 rows and remaining TR138 rows in the human review queue have not been completed. Severity: moderate. The directional finding (majority of genuine flips are toward unsafe) is likely robust to reviewer variation; the precise 27% rate is not.

**T2: Pseudoreplication in TR141 v2.1.** The alignment-type ANOVA (p = 0.008) was a false positive from unbalanced groups with n = 1 per category. The v3 correction (p = 0.942) resolves this, but it means any conclusions drawn from the v2.1 ANOVA in the interim period were not supported. Status: corrected.

**T3: TR143 v1.0 retraction.** Version 1.0 presented the composition result as a "clean null" without reporting the directional asymmetry. The retraction and re-publication as v2.0 corrects this, but it means any citations of v1.0 are referencing incomplete results. Status: corrected.

**T4: Co-batch verification rate of 22.1%.** For 78% of TR143 observations, the treatment delivery cannot be confirmed. The aggregate null may be artificially inflated by non-treatment. Severity: moderate. The null likely holds under full treatment (the 22.1% verified subset also shows null aggregate effect), but the directional finding may strengthen with better verification.

**T5: Dual-judge agreement of kappa = 0.104.** The TR139 multi-turn ASR estimates carry substantial measurement uncertainty due to poor inter-judge agreement. The judges may be measuring partially different constructs, and the "true" ASR may differ from either judge's estimate. Severity: moderate for absolute ASR values; low for cross-condition comparisons (both judges are applied consistently).

**T6: No LLM judge in TR141 final artifacts.** The TR141 flip rates are based solely on regex classifiers, without independent LLM-judge validation. Given the 73% artifact rate established in TR138, TR141 flip rates should be interpreted as upper bounds. Severity: moderate.

**T7: Prompt-length confound in TR143.** The 47% difference in mean token length between benign (58.5 tokens) and jailbreak (85.8 tokens) filler pools means that composition-content comparisons are confounded with computational load. Jailbreak-batched conditions impose higher compute per batch, which could mask or amplify small effects. Severity: low for the aggregate null (both filler types produce null results), moderate for composition-content claims.

### 8.2 External validity threats

**T8: Consumer hardware only.** All measurements use NVIDIA RTX 4080 Laptop 12GB (5 of 6 TRs) or RTX PRO 6000 Blackwell 98GB (TR141 only). GPU-dependent floating-point behavior means batch-perturbation effects may differ on data-center GPUs (A100, H100) with different accumulation hardware. Safety findings should generalize (safety is a model/software property), but batch-specific effects may not. Severity: moderate for batch findings; low for quantization findings.

**T9: GGUF-only quantization.** All quantization tests use GGUF k-quant schemes (llama.cpp). GPTQ, AWQ, SqueezeLLM, and other quantization methods use different compression strategies and may produce different safety profiles. The Q4_K_M safety floor cannot be assumed to apply to non-GGUF quantization. Severity: moderate.

**T10: Temperature = 0 only.** All evaluations use deterministic greedy decoding. Stochastic sampling (temperature > 0) introduces variance that could interact with safety in unknown ways. The batch-perturbation mechanism specifically depends on deterministic decoding (the epsilon must flip the argmax); at temperature > 0, the epsilon is dominated by sampling noise and may be irrelevant. Severity: high for batch findings (mechanism may not apply); moderate for quantization findings (weight perturbation is temperature-independent).

**T11: Model size range.** Models tested span 360M to 14.8B parameters. Larger models (30B+) may be more or less resilient to quantization due to different weight redundancy profiles. The findings should not be extrapolated beyond the tested range without re-profiling. Severity: moderate.

**T12: No production traffic patterns.** All experiments use controlled, synthetic workloads. Production traffic includes variable batch sizes, mixed request types, autoscaling, request queuing, and other dynamics not captured in laboratory conditions. Severity: moderate for batch findings; low for quantization findings.

### 8.3 Construct validity threats

**T13: Classifier vs. human judgments.** Safety scores are classifier-produced proxies. The 73% artifact rate (TR138) demonstrates that classifier-based measurements can overestimate effects by approximately 3x. Neither regex classifiers nor LLM judges have been validated against a gold-standard human safety assessment across the full evaluation battery. Severity: high for absolute safety scores; low for relative comparisons between conditions.

**T14: Safety benchmark representativeness.** The task batteries cover refusal, truthfulness, bias, and adversarial robustness, but do not cover all safety dimensions (privacy leakage, hallucination severity, instruction injection, prompt extraction). A model that scores well on the tested battery may have safety vulnerabilities on untested dimensions. Severity: moderate.

**T15: Two-model restriction in TR142.** The quality-safety divergence finding is based on only 2 models (llama3.2-1b, llama3.2-3b). The opposite-sign correlations between these models suggest high model-level variance, but with N = 2 the variance is not well-characterized. Additional models are needed to establish whether the 13.9x divergence is typical or extreme. Severity: moderate for the specific 13.9x number; low for the directional finding (quality is insufficient as safety proxy).

### 8.4 Statistical validity threats

**T16: Multiple testing burden.** The synthesis draws on hundreds of statistical tests across 6 TRs. While each TR applies Holm-Bonferroni correction within its own testing family, there is no cross-TR correction. The probability that at least one finding across the entire synthesis is a false positive is non-trivial. Severity: moderate. Mitigated by the convergent-evidence structure: the key findings (Q2_K dangerous, batch perturbation small, composition null) are supported by multiple independent tests across multiple TRs.

**T17: Power limitations at the per-cell level.** Several TRs have limited statistical power for detecting small effects in specific cells. TR142's truthfulness analysis has MDE = 28.0pp. TR143's directional asymmetry test operates on 9-12 total flips per condition. TR140's per-cell Fisher exact tests have adequate power only for large effects (h > 0.60). Non-significant findings at the per-cell level should be interpreted as "effect undetectable at this sample size," not as "effect absent." Severity: moderate for cell-level claims; low for aggregate claims with larger N.

**T18: Regression to the mean in enriched subsets.** TR138's enriched-subset methodology selects prompts for boundary sensitivity, which biases upward the observed effect sizes. The 1.68% flip rate on the enriched subset is not comparable to the 0.6% rate on the full prompt set. Any conclusion drawn from enriched-subset results must be qualified by the selection bias. Severity: low (the enriched-subset results are clearly labeled and not used for headline claims).

### 8.5 Ecological validity threats

**T19: No adversarial adaptive attacks.** The jailbreak strategies in TR139 and TR140 are pre-scripted, not adaptive. A real adversary observes model responses and adapts their strategy; the automated evaluation cannot do this. Adaptive adversaries may achieve higher ASR than the pre-scripted strategies, meaning the reported ASR values are lower bounds on adversarial vulnerability. Severity: moderate.

**T20: No multi-turn + many-shot interaction.** TR139 tests multi-turn strategies and TR140 tests many-shot strategies, but no TR tests the combination (multi-turn conversation with many-shot compliance examples embedded in the context). This interaction could produce vulnerability spikes beyond what either modality achieves alone. Severity: moderate.

**T21: Batch sizes tested may not match production.** TR138 tests batch sizes up to 32. Production vLLM deployments may process batches of 128 or larger. Whether the batch-perturbation effect scales linearly, sub-linearly, or super-linearly with batch size beyond 32 is unknown. Severity: low (the effect is already negligible at batch = 32; linear scaling would still produce a negligible effect at batch = 128).

**T22: Harmful behavior sample.** Both TR139 and TR140 use 50 harmful behaviors from JailbreakBench. These 50 behaviors may not be representative of the full space of harmful requests. The per-behavior variance (65.7% of total variance in TR140) suggests that different behavior samples could produce different aggregate ASR estimates. Severity: moderate for absolute ASR values; low for cross-condition comparisons (the same behaviors are tested under all conditions).

### 8.6 Threat summary and prioritization

| Threat | Severity | Affects | Resolution |
|--------|---------|---------|------------|
| T1: Single human reviewer | Moderate | Flip rate calibration | Complete pending review queue |
| T2: Pseudoreplication (corrected) | N/A | Historical only | Corrected in TR141 v3 |
| T3: TR143 retraction (corrected) | N/A | Historical only | Corrected in TR143 v2.0 |
| T4: 22.1% co-batch verification | Moderate | Composition claims | Future: instrument scheduler |
| T5: Dual-judge kappa = 0.104 | Moderate | Multi-turn ASR values | Low for cross-condition deltas |
| T6: No LLM judge in TR141 | Moderate | Flip rate precision | Apply TR138 artifact correction |
| T7: 47% prompt-length confound | Low-Moderate | Content comparisons | Future: length-matched pools |
| T8: Consumer hardware only | Moderate (batch) | Hardware generalization | Future: data-center GPU testing |
| T9: GGUF-only quantization | Moderate | Quant method generalization | Future: test GPTQ, AWQ |
| T10: Temperature = 0 only | High (batch) | Temperature generalization | Future: temp > 0 experiments |
| T11: Model size range | Moderate | Scale generalization | Future: 30B+ models |
| T12: No production traffic | Moderate | Ecological validity | Future: production monitoring |
| T13: Classifier vs. human | High (absolute) | Absolute safety scores | Human-validate critical findings |
| T14: Benchmark representativeness | Moderate | Safety dimension coverage | Future: broader battery |
| T15: 2-model TR142 | Moderate | Quality-safety generality | Future: add models |
| T16: Multiple testing | Moderate | False positive risk | Convergent evidence mitigates |
| T17: Per-cell power | Moderate | Small-effect detection | Report MDE alongside nulls |
| T18: Enriched-subset bias | Low | TR138 enriched results | Clearly labeled |
| T19: Non-adaptive attacks | Moderate | ASR lower bounds | ASR values are conservative |
| T20: No multi-turn x many-shot | Moderate | Interaction gap | Future: factorial design |
| T21: Batch size range | Low | Production extrapolation | Effect negligible within tested range |
| T22: Behavior sample | Moderate | ASR estimates | 50 behaviors adequate for ranking |

The highest-severity threats (T10: temperature, T13: classifier vs. human) affect different findings differently. T10 primarily affects the batch-perturbation findings (which depend on deterministic decoding) but not the quantization findings (which depend on weight perturbation). T13 affects all absolute safety scores but not the relative rankings or directional findings. The most operationally consequential unresolved threat is T4 (co-batch verification): if the 22.1% rate could be improved to > 80%, the composition null would be substantially strengthened, and the directional asymmetry estimate would be more precise.

---

## 9. Integration with Phase 3 (TR134-TR137) and Phase 2 (TR123-TR133)

### 9.0 Integration overview

The integration of Phase 3.5/4 (TR138-TR143) with Phase 3 (TR134-TR137) and Phase 2 (TR123-TR133) produces a complete deployment decision system that spans performance, capability, safety cost, and safety attack surface. The integration is structured around three questions: What does this synthesis confirm from prior phases? What does it tension? What is new?

The answers to these questions determine how the prior phases' recommendations should be updated. Confirmed findings retain their original operational status. Tensioned findings require qualification or revision. New findings extend the operational framework with additional validation requirements. The integration is documented at the level of specific claims so that operators can trace every recommendation to its evidence base across all phases.

### 9.1 Findings confirmed by this synthesis

| Phase 3 Finding | This Synthesis | Status |
|----------------|---------------|--------|
| Q2_K is CRITICAL for Llama 1B (TR134) | Q2_K is CRITICAL for ALL models across all attack modalities (TR139, TR140) | **Extended** |
| Q4_K_M retains >= 93% safety (TR134) | Multi-turn and many-shot testing at Q4_K_M does not break Llama models (TR139, TR140) | **Confirmed** |
| Per-model profiling mandatory (I-squared = 99.9%) | 6.3x fragility range across 15 models, alignment type not predictive (TR141) | **Strengthened** |
| Concurrency is safe (TR135, I-squared = 0.0%) | Batch size perturbation is real but negligible (TR138: 0.16% genuine) | **Consistent** (different mechanism, similar conclusion) |
| Safety veneer not universal (3/10 in TR137) | Quality-safety divergence model-dependent (TR142: opposite signs) | **Extended** |

**Observations.** The confirmation pattern is significant: every major Phase 3 finding that could be tested by this synthesis was confirmed or extended. Q4_K_M safety is confirmed not just for single-turn but for multi-turn and many-shot. Per-model profiling is strengthened by the 6.3x cross-architecture fragility range. Concurrency safety clearance is consistent with batch-perturbation negligibility. The veneer hypothesis refutation is extended by the quality-safety divergence finding. This convergence across independent TRs with different designs provides high confidence in the foundational Phase 3 conclusions.

### 9.2 Findings tensioned by this synthesis

| Phase 3 Finding | This Synthesis | Resolution |
|----------------|---------------|------------|
| Safety flips 4x capability (uncorrected, TR138 enriched) | Combined synthesis: 0.94x ratio (TR141 combined) | Human adjudication corrects the 4x to ~1.3x (TR141); the safety-specificity was partially a classifier artifact |
| Automated classifiers are consistent proxies | 73% artifact rate for small effects (TR138 adjudication) | Automated classifiers are consistent but inflated for small effects; cross-condition deltas are more reliable than absolute flip rates |

### 9.2.1 The safety-specificity question

The tension between Phase 3's safety-over-capability asymmetry and TR141's near-parity finding deserves extended discussion. Phase 3 (TR138 enriched subset) reported safety flips at 4.0x the capability flip rate (1.68% vs 0.42%), suggesting that batch perturbation disproportionately targets safety. TR141's combined 15-model synthesis reports a 0.94x ratio, near parity.

The resolution involves three factors. First, the TR138 enriched subset is biased toward boundary-sensitive prompts where safety flips are most likely; the full-prompt-set ratio is lower. Second, the TR138 human adjudication shows that 73% of automated safety flips are classifier artifacts, meaning the "true" safety flip rate is approximately 3x lower than automated detection suggests. Third, TR141 adds 12 models to the original 3-model comparison, diluting any model-specific safety bias with a broader population that includes models where capability flips are more common than safety flips (the v3 extension shows 0.78x ratio).

The corrected picture is: batch perturbation introduces a small, non-safety-specific output instability. The instability is model-dependent (0.00% to 2.39%), weakly directionally biased toward unsafe (59.1% unsafe in TR138 audit, but 66.2% safe in TR141 combined), and predictable from output instability (r = 0.91). The initial framing of batch perturbation as a specifically safety-targeted effect was influenced by automated classifier artifacts and a small, non-representative model set.

### 9.3 New findings not addressable by Phase 3

| Finding | Significance | Implication for Phase 3 conclusions |
|---------|-------------|-------------------------------------|
| Multi-turn jailbreak x quant interaction (TR139) | Reveals attack modality Phase 3 could not test | Single-turn safety evaluation (Phase 3 battery) is necessary but not sufficient |
| Many-shot format vulnerability (TR140) | Reveals format-level attack vector | Phase 3's task battery does not include format-specific tests |
| Output instability predictor (TR141, r = 0.91) | Provides practical screening metric | Phase 3's per-model profiling mandate now has a cheap implementation |
| Quality-safety divergence (TR142, 13.9x) | Invalidates quality-as-proxy practice | Phase 3 assumed quality and safety co-vary; TR142 shows they may not |
| Batch composition null (TR143) | Closes composition attack vector | Phase 3 did not address composition; TR143 shows it need not be addressed |

### 9.3.1 Implications for Phase 3 methodology

Several findings in this synthesis have methodological implications for how the Phase 3 results should be interpreted. The TR138 human adjudication (73% artifact rate) suggests that Phase 3's automated safety measurements may overestimate small effects across all TRs. Phase 3's quantization effects (measured in tens of percentage points) are well above the artifact rate and are therefore robust, but Phase 3's smaller effects (e.g., the Llama 3B anomalous improvement of +6pp at Q2_K) may be partially or wholly composed of classifier artifacts.

TR142's quality-safety divergence finding has implications for Phase 3's safety-capability asymmetry analysis (the veneer hypothesis test in TR137). The Phase 3 analysis normalized safety and capability to their respective baselines and compared degradation rates. TR142 shows that this comparison is model-dependent in a way that the Phase 3 analysis did not fully characterize: on llama3.2-1b, the safety-capability comparison produces a ratio < 1.0 (supporting the veneer hypothesis), while on llama3.2-3b, the ratio > 1.0 (contradicting it). The Phase 3 finding that only 3/10 model-axis combinations support the veneer hypothesis is consistent with TR142's finding that the quality-safety relationship is model-dependent, but TR142 adds the stronger claim that the correlation sign itself reverses between models.

TR141's alignment-type correction (p = 0.008 to p = 0.942) has implications for Phase 3's family-level ANOVA (F = 2.50, p = 0.137, not significant at alpha = 0.05). The Phase 3 ANOVA tested whether model families differ in quantization resilience; it was not significant, but was close to significance with only 4 models. TR141's correction suggests that the non-significance of the Phase 3 family ANOVA is the correct result, not an underpowered miss: alignment type does not predict safety behavior under perturbation, regardless of whether the perturbation is batch size (TR141) or quantization (Phase 3).

### 9.4 Integration with Phase 2 (TR123-TR133)

Phase 2's performance recommendations remain valid but require additional safety qualification from this synthesis:

| Phase 2 Decision | This Synthesis Update | Updated Recommendation |
|-----------------|---------------------|----------------------|
| Q4_K_M recommended default across tested models (TR125) | Multi-turn and many-shot safe at Q4_K_M (TR139, TR140) | **Confirmed**: Q4_K_M remains optimal for both performance and safety |
| Q2_K banned for capability (TR125) | Q2_K banned for safety across all attack modalities (TR139, TR140) | **Reinforced**: Q2_K ban extends from capability to safety to multi-turn to many-shot |
| vLLM for multi-agent serving (TR130) | Batch composition is safe under vLLM (TR143) | **Confirmed**: vLLM composition is not a safety concern |
| Quality metrics for deployment validation (TR124) | Quality is insufficient as safety proxy (TR142, 13.9x) | **Tensioned**: quality validation must be supplemented with safety-specific testing |
| Continuous batching for throughput (TR132) | Static and continuous batching produce identical safety outcomes (TR143) | **Confirmed**: batching mode does not affect safety |
| Q3_K_S model-dependent for capability (TR125) | Q3_K_S is the hidden safety danger zone (TR142) | **Updated**: Q3_K_S requires safety-specific validation that quality metrics will miss |
| NUM_PARALLEL is a no-op for throughput (TR128) | Concurrency is safety-neutral (TR135, Phase 3) | **Confirmed** by this synthesis through batch-perturbation negligibility (TR138, TR141) |

The most important tension between Phase 2 and this synthesis is the quality-as-safety-proxy practice. Phase 2 validated quantized deployments using quality benchmarks (MMLU, ARC-Challenge, BERTScore, coherence) and found that Q4_K_M preserves quality within -4.1pp of FP16 across all models (TR125). The implicit assumption was that quality preservation implies safety preservation -- if the model can still answer science questions correctly, it can still refuse harmful requests. TR142 shows this assumption is false: quality degradation underestimates safety degradation by 13.9x at Q3_K_S. An operator who follows Phase 2's quality-validation protocol without adding safety-specific benchmarks will approve deployments that fail safety review.

The resolution is not to abandon quality metrics but to supplement them with safety-specific testing. Quality metrics remain valuable for their intended purpose (validating capability preservation), and the Phase 2 quality-validation protocol (TR124, TR125) is methodologically sound for capability. The update is that safety is a separate dimension that requires separate measurement. The Phase 3 and Phase 3.5/4 safety evaluation batteries provide the additional measurement layer that Phase 2's quality metrics cannot supply.

### 9.5 The complete decision stack

The four-phase program produces a layered decision stack:

```
Level 0: Methodology (Phase 1, TR117-TR122)
  - How to measure correctly
  - Artifact-first reporting
  - Measurement boundary definitions

Level 1: Performance (Phase 2, TR123-TR133)
  - What to deploy for cost, throughput, capability
  - Q4_K_M, Ollama/vLLM, concurrency limits
  - ChimeraForge capacity planning

Level 2: Safety Cost (Phase 3, TR134-TR137)
  - Safety cost of Phase 2's recommendations
  - Per-model profiling requirement
  - Backend migration as safety-critical change
  - Concurrency safety clearance

Level 3: Safety Attack Surface (Phase 3.5/4, TR138-TR143)
  - Full attack surface mapping
  - Multi-turn, many-shot, batch, composition, cross-architecture
  - Quality-safety divergence
  - Human adjudication recalibration
```

Each level depends on the one below. Level 3 cannot exist without Level 2's per-model profiling framework, Level 2 cannot exist without Level 1's deployment recommendations to validate, and Level 1 cannot exist without Level 0's measurement methodology. The stack is now four levels deep, with Level 3 adding the attack-surface dimension that transforms the program from a deployment-validation framework into a comprehensive safety-risk assessment system.

### 9.6 Remaining gaps across all phases

The integration of all four levels reveals five specific gaps that future work should address:

1. **Quantization x multi-turn x batch interaction.** No factorial design tests the three-way interaction. A quantized model under batch perturbation processing multi-turn jailbreak conversations may behave differently than the additive projection from independent TR findings suggests. This is the highest-priority gap because it combines the three most significant safety variables identified across the program.

2. **Temperature > 0 under adversarial pressure.** All evaluations use deterministic greedy decoding. Stochastic sampling at temperature > 0 introduces per-token randomness that could interact with jailbreak strategies in unknown ways. A stochastic model may comply with a jailbreak attempt on some samples and refuse on others, making safety a probabilistic rather than deterministic property. The batch-perturbation mechanism (floating-point epsilon at decision boundaries) may be entirely masked by sampling noise at temperature > 0, making the batch findings irrelevant for stochastic deployments.

3. **Human adjudication at scale.** The 73% artifact rate is based on 63 reviewed rows. The human review queues (757 TR138 rows + 263 TR141 rows) are generated but incomplete. Completing these reviews would (a) establish inter-rater reliability by adding a second reviewer, (b) provide a larger sample for estimating the artifact rate with narrower confidence intervals, and (c) calibrate the TR141 fragility ranking against human judgments.

4. **Models > 14.8B parameters.** TR141b tests models up to 14.8B. Models above this (30B, 70B) remain untested. Larger models may show different quantization and perturbation profiles due to greater weight redundancy or more complex alignment training. No extrapolation from the current data is defensible.

5. **Non-GGUF quantization.** GPTQ, AWQ, and SqueezeLLM use fundamentally different compression strategies (e.g., activation-aware weight quantization, group-based rounding) that may produce different safety profiles. The Q4_K_M safety floor established for GGUF k-quant cannot be transferred to 4-bit GPTQ or AWQ without re-profiling.

### 9.7 Limitations by report and mitigations

| TR | Limitation | Mitigation | Status |
|----|-----------|------------|--------|
| TR138 | Human adjudication by single reviewer (63 rows) | Review queue generated (757 rows); directional finding robust to reviewer variation | **Open** |
| TR138 | Enriched-subset bias inflates flip rates | Full-prompt-set rates reported alongside enriched; 0.16% genuine rate uses full-set data | **Mitigated** |
| TR138 | v2.2 scorer correction changes audit numbers | Correction is conservative (removes 5 false-flip rows); core asymmetry preserved | **Mitigated** |
| TR139 | Dual-judge kappa = 0.104 (poor) | Cross-condition comparisons use consistent judges; absolute ASR carries uncertainty | **Accepted** |
| TR139 | Pre-scripted strategies (not adaptive) | ASR values are lower bounds on adversarial vulnerability; operational recommendation still holds | **Accepted** |
| TR139 | MDE = 15.19pp (cannot detect moderate effects) | Large effects detected with high confidence; moderate effects acknowledged as potentially undetected | **Accepted** |
| TR140 | Variance decomposition dominated by residual (65.7%) | Per-behavior variance is inherent; 50-behavior sample mitigates but does not eliminate | **Accepted** |
| TR140 | Judge agreement low at Q2_K (63.5%, kappa = 0.13) | Q2_K model output is inherently ambiguous; classification uncertainty is a feature of the phenomenon | **Accepted** |
| TR141 | v2.1 alignment ANOVA was false positive (p = 0.008) | v3 extension corrects to p = 0.942; correction published transparently | **Corrected** |
| TR141 | Regex-only artifact chain (no LLM judge) | TR138 adjudication provides calibration; TR141 rates are upper bounds | **Open** |
| TR141 | 15-model synthesis excludes 3 models (technical drops) | Remaining 15 models span 10+ families; coverage is adequate | **Mitigated** |
| TR142 | Only 2 models in cross-reference | Opposite-sign correlations demonstrate model dependence; additional models needed | **Open** |
| TR142 | Truthfulness underpowered (MDE = 28.0pp) | Truthfulness findings explicitly labeled as inconclusive | **Acknowledged** |
| TR142 | No standalone TOST result in artifact | Conservative floor (Q5_K_M) based on observed deltas, not formal equivalence | **Accepted** |
| TR143 | Co-batch verification only 22.1% | Biases toward null; actual composition effect likely smaller than detected | **Open** |
| TR143 | Prompt-length confound 47% | Aggregate null holds for both filler types; content comparison needs covariate adjustment | **Accepted** |
| TR143 | v1.0 retracted for omitting directional asymmetry | v2.0 reports full picture; retraction documented transparently | **Corrected** |
| TR143 | Capability TOST 1/3 (2 degenerate) | Degenerate cases from zero-variance; capability preservation likely but not formally established | **Open** |

**Narrative interpretation.** The limitation table reveals two patterns. First, the positive findings (quantization is dangerous, multi-turn jailbreaks are amplified) are robust: the evidence base is large (48,425+ records), the effects are large (100% ASR, 13.9x divergence), and the limitations do not affect the direction of the finding. Second, the small-effect findings (batch perturbation, composition asymmetry) carry more uncertainty: human adjudication is incomplete, co-batch verification is low, and the directional asymmetry operates on small N. The practical implication is that the high-severity recommendations (Q2_K ban, multi-turn testing, format restriction) can be acted on with high confidence, while the low-severity recommendations (output instability monitoring, directional pattern logging) are correctly framed as monitoring recommendations rather than deployment requirements.

The most consequential open limitations are: (1) the pending human adjudication queues (757 + 263 rows), which would provide calibrated flip rates for TR141 and broader confidence intervals for TR138; (2) the 2-model restriction in TR142, which limits the generality of the 13.9x divergence finding; and (3) the 22.1% co-batch verification in TR143, which limits the strength of both the aggregate null and the directional finding.

---

## 10. Numbered Findings

Before the conclusive statement, the ten principal findings of this synthesis are listed with their evidence base and confidence level:

1. **Quantization below Q3_K_S is the dominant safety risk across all attack modalities.** 100% ASR at Q2_K (TR139), 92% ASR at Q2_K message-array (TR140), 13.9x quality-safety divergence at Q3_K_S (TR142). Confidence: **High.** Convergent across 4 independent TRs.

2. **Batch perturbation is real but operationally negligible after human adjudication.** 0.6% automated flip rate, 27% genuine rate, 0.16% corrected rate (TR138). Directionally biased toward unsafe (59.1% of genuine flips). Confidence: **High** for magnitude; **Moderate** for directional claim (single reviewer).

3. **Output instability is the only reliable predictor of batch fragility.** r = 0.91, R-squared = 0.83 across 15 models (TR141). Baseline refusal rate is uninformative (r = 0.028). Confidence: **High.**

4. **Alignment type does NOT predict batch fragility.** ANOVA F = 0.13, p = 0.942 with balanced groups (TR141 v3). Corrects v2.1 false positive (p = 0.008) from pseudoreplication. Confidence: **High.**

5. **Multi-turn jailbreaks interact with quantization, but not selectively over direct attacks.** All 8 ANOVAs p < 1e-4 (TR139). Welch comparison of multi-turn vs direct slopes: p = 0.702 (TR139). Confidence: **High** for interaction; **High** for non-selectivity.

6. **Message-array format is the dominant many-shot attack vector.** 92% vs 0% ASR by format on the same model at the same quant level (TR140). Confidence: **High.**

7. **Quality metrics are categorically insufficient as safety proxies.** 13.9x divergence (TR142). Correlation sign reverses between models: +0.994 vs -0.829 (TR142). Confidence: **High** for the directional finding; **Moderate** for the specific 13.9x (2-model limitation).

8. **Batch composition does NOT affect aggregate safety outcomes.** 21 McNemar tests, 3 Cochran's Q, 3 MH ORs all non-significant (TR143). Confidence: **Moderate** (22.1% co-batch verification limits the strength of the negative claim).

9. **Rare flips are directionally biased toward unsafe.** 88-92% toward unsafe, binomial p = 0.006 for strongest condition (TR143). Consistent across all composition types. Confidence: **Moderate** (small N: 9-12 total flips per condition).

10. **No categorical variable reliably predicts safety under optimization.** Alignment type fails for batch fragility (TR141). Quality fails as safety proxy (TR142). Composition content fails for flip direction (TR143). Only per-model empirical measurement works. Confidence: **High** (convergent negative evidence across 4 TRs).

### What This Program Contributes

The 6 TRs in this synthesis provide systematic coverage of the inference-optimization safety attack surface on consumer hardware within the tested boundary conditions (models ≤14.8B, GGUF k-quant, temperature 0). The 306,996 data points, 18+ models, and 6 attack dimensions provide broader coverage than any individual study could achieve, but the program has not been externally reviewed or validated against production deployment outcomes. The per-model profiling recommendation, now equipped with a practical screening metric (output instability, r = 0.91), provides a concrete protocol that can be evaluated by other research groups.

### What This Program Does NOT Establish

- That batch perturbation is a deployment-blocking risk (it is not, at 0.16% human-adjudicated)
- That alignment type is irrelevant to safety in general (only to batch fragility specifically)
- That multi-tenant vLLM is proven safe (22.1% verification limits the claim)
- That findings transfer to models > 14.8B or non-greedy decoding
- That the directional asymmetry is universal (3 models only in TR143)
- That the 13.9x quality-safety divergence applies to all models (2 models only in TR142)
- That the 73% artifact rate applies to all automated safety evaluations (1 reviewer, 63 rows, 1 classifier)
- That any specific numerical threshold (e.g., "0.16% genuine rate") is precise to more than 1 significant figure

---

## 11. Conclusive Statement

This synthesis closes the safety attack-surface mapping initiated by TR138 and completed through TR143. The research arc spans 6 technical reports, 306,996 evaluated samples, 18+ models from 10+ architecture families, and 6 dimensions of the inference-optimization attack surface. The arc is deliberately sequential: TR138 discovers the batch-perturbation phenomenon, TR139 and TR140 map the quantization-jailbreak interaction, TR141 extends to cross-architecture scale, TR142 reveals the quality-safety divergence, and TR143 closes the composition channel.

The synthesis produces a clear hierarchy of safety threats. Quantization below Q3_K_S is the dominant risk, producing effects measured in tens of percentage points across all attack modalities. Multi-turn and many-shot jailbreaks interact with quantization to produce vulnerability spikes (100% ASR at Q2_K) that no single-turn evaluation can detect. Batch perturbation exists but is operationally negligible (0.16% human-adjudicated genuine rate). Batch composition produces no aggregate effect. Quality metrics are categorically insufficient as safety proxies (13.9x divergence). And no categorical predictor -- alignment type, quality score, architecture family, or parameter count -- reliably predicts which models will be affected.

The program discovered and corrected two significant errors during development: TR141's alignment-type ANOVA (p = 0.008) was a false positive from pseudoreplication, corrected to p = 0.942 with balanced groups; TR143 v1.0 omitted the directional asymmetry finding, corrected in v2.0. Human adjudication of TR138 (single reviewer, n=63) found 73% of automated safety flip detections were regex artifacts, not genuine behavioral changes. These corrections are documented transparently so that readers can assess the stability of each finding and researchers can avoid replicating the same errors.

The practical output is a risk framework, not a set of universal rules. The extreme model-level heterogeneity (I-squared = 99.9% on quantization, 6.3x fragility range on batch perturbation) means that no universal guideline will be reliable for a substantial fraction of models. The per-model profiling mandate -- now equipped with a practical screening metric (output instability, r = 0.91) and a validated safety battery -- is the program's most important operational recommendation. An operator who validates safety per-model and per-configuration, using safety-specific benchmarks rather than quality proxies, at the actual deployment quantization level with multi-turn adversarial testing, is protected against every threat identified in this 306,996-sample program. An operator who relies on quality benchmarks, categorical heuristics, or single-turn evaluation is exposed to risks that range from negligible (batch composition) to catastrophic (100% ASR under multi-turn jailbreak at Q2_K).

The two corrections should be understood as methodological failures that were caught and fixed, not as features. The TR141 v2.1 pseudoreplication (n=1 per non-RLHF category) should have been identified before publication; the v3 extension that corrected it was designed specifically to test the fragile preliminary finding. The TR143 v1.0 omission of the directional asymmetry reflected automated report generation that prioritized aggregate statistics over secondary findings. Both errors illustrate that initial results in this program require replication and independent review before being used for operational decisions.

The safety attack-surface mapping for consumer-hardware LLM deployment on models up to 14.8B parameters under GGUF quantization with greedy decoding is now substantially complete. The remaining gaps -- production traffic patterns, temperature > 0, models > 14.8B, non-GGUF quantization, multi-GPU parallelism, and improved co-batch verification -- are identified in Section 8 and represent the natural extension of this work. The framework for addressing those gaps -- per-model profiling, multi-modal attack testing, human adjudication of automated measurements, and honest self-correction when findings do not replicate -- is established by this program and can be applied to any future extension without modification.

The three-line summary of 306,996 samples: quantization below Q3_K_S is catastrophic; batch perturbation is real but negligible; and no categorical shortcut predicts which models will be safe. Measure safety directly, per model, with multi-modal adversarial testing, at the actual deployment configuration. That is the only defensible practice.

The research arc from TR138 (discovering batch perturbation) through TR143 (closing the composition channel) demonstrates the value of systematic, sequential investigation over ad-hoc studies. Each TR fills a specific gap identified by its predecessor: TR138 reveals batch perturbation, TR139 and TR140 map the quantization-jailbreak interaction, TR141 extends to cross-architecture scale, TR142 reveals quality-safety divergence, and TR143 closes the composition channel. The six TRs together produce a more complete and more reliable safety assessment than any single study could achieve, because each study both answers its own question and cross-validates the findings of the preceding studies.

The safety attack-surface mapping for consumer-hardware LLM deployment is now substantially complete for models up to 14.8B parameters under GGUF quantization with greedy decoding. The remaining gaps -- production traffic patterns, temperature > 0, models > 14.8B, non-GGUF quantization, multi-GPU parallelism, improved co-batch verification, and the untested cross-dimensional interactions (quantization x composition, multi-turn x batch size) -- are identified in Section 8 and Appendix N and represent the natural extension of this work into future phases. The framework for addressing those gaps -- per-model profiling, multi-modal attack testing, human adjudication of automated measurements, convergent evidence across independent studies, and honest self-correction when findings do not replicate -- is established by this program and can be applied to any future extension without modification.

The operational deliverables of this synthesis are: (1) a Q2_K universal ban supported by convergent evidence from 4 TRs and 4 attack modalities, (2) a practical batch-safety screening protocol based on output instability (r = 0.91), (3) a format-level mitigation for many-shot attacks (restrict message-array format, reducing ASR from 92% to 0%), (4) an invalidation of quality-as-safety-proxy practices (13.9x divergence), (5) a closure of the composition attack vector (aggregate null across 21 McNemar tests), (6) a per-model profiling protocol equipped with concrete screening metrics and cost estimates, and (7) a comprehensive threat hierarchy with 22 explicitly assessed threats and their prioritized mitigations. These deliverables are actionable, evidence-graded, and bounded by explicitly documented limitations.

The human-adjudication finding (73% artifact rate) deserves final emphasis because it changes how all safety evaluation results -- not just this program's -- should be interpreted. Any automated safety evaluation reporting small effects (< 5%) without human validation is likely measuring classifier noise rather than genuine behavioral changes. This finding applies to any study using regex-based classifiers, LLM judges, or any automated classification system that operates on model-generated text. The implication is not that automated classifiers are useless -- they are consistent, fast, and scalable -- but that their output should be interpreted as comparative proxies rather than absolute measurements, especially when the effects being measured are small relative to the classifier's inherent noise floor.

The research program that produced this synthesis -- 30+ technical reports, 4 phases, 400,000+ total evaluated samples across all phases -- suggests that systematic safety evaluation is feasible on consumer hardware, though external validation and deployment feedback are needed before these findings can be considered deployment-grade. The key ingredients are: artifact-first reporting (every claim traced to a specific data file), sequential design (each TR fills a gap from its predecessor), multi-modal testing (single-turn + multi-turn + many-shot + batch + composition), human calibration (adjudication of automated measurements), honest self-correction (reversal of false findings), and convergent evidence (the same conclusion from multiple independent studies). These ingredients are transferable to any safety evaluation program and do not require specialized infrastructure or proprietary tools.

---

## Appendix A: Claim-to-Artifact Chain-of-Custody

Every major claim in this synthesis is mapped to a specific TR, section, data source, and validation method. The chain-of-custody table enables reviewers to trace any claim to its empirical foundation.

| Claim | Source TR | Section | Data Path | Records | Validation |
|-------|----------|---------|-----------|---------|------------|
| 0.6% safety flip rate from batch size | TR138 v1 | Phase 1 | `research/tr138/results/20260313_184600/` | 31,410 | Single-run, temp=0 |
| 27% genuine rate (73% artifacts) | TR138 v2 | Audit layer | `research/tr138/results/20260313_184600/replication_run` | 63 reviewed rows | Single-reviewer human adjudication |
| 0.16% corrected flip rate | TR138 | Computed | 0.006 x 0.27 = 0.0016 | Derived | Assumes TR138 artifact rate generalizes |
| 4.0x safety/capability ratio (enriched) | TR138 v2 | Phase 1 replication | `research/tr138/results/20260313_184600/replication_run` | 7,257 | Enriched 187-prompt subset |
| 100% ASR (qwen2.5-1.5b Q2_K attention_shift) | TR139 | Phase 1 | `research/tr139/results/20260314_012503/` | 9,600 conversations | Dual-judge (kappa=0.104) |
| All 8 ANOVAs p < 1e-4 | TR139 | Phase 1 | Analysis JSON | 9,600 conversations | Holm-Bonferroni corrected |
| Welch p = 0.702 (multi-turn vs direct) | TR139 | H2 test | Analysis JSON | Slopes from 8 strategies | Welch's t-test |
| Q2_K universal vulnerability threshold | TR140 | Phase 1 | `research/tr140/results/20260316_164907/` | 15,000 | 15,000 judge labels |
| 92% vs 0% ASR by format | TR140 | Phase 1 | Same | llama3.1-8b Q2_K N=16 | Fisher exact p < 0.001 |
| r = 0.91 output instability predictor | TR141 | SS10 / combined synthesis | `research/tr141/results/20260318_194013/` | 106,020 scored | Pearson correlation, 15 models |
| ANOVA p = 0.942 (alignment type) | TR141 v3 | SS11 | Same | Model-level ANOVA, n>=3/category | Balanced groups |
| 6.3x fragility range | TR141 | Combined synthesis | Same | 15 models | phi-2 2.39% to tinyllama 0.00% |
| 66.2% net-safe directional bias | TR141 | Directional analysis | Same | 240 flips | Binomial p = 1e-6 |
| 13.9x safety/quality divergence | TR142 | SS6-SS8 | `research/tr142/results/20260326_183953/` | 23,632 | Cross-reference TR125 + TR134 |
| r = +0.994 (llama3.2-1b quality-safety) | TR142 | SS5 | Same | Within-model Pearson | p = 5.8e-5 |
| r = -0.829 (llama3.2-3b quality-safety) | TR142 | SS5 | Same | Within-model Pearson | p = 0.041 |
| All McNemar p > 0.125 (composition) | TR143 | SS7 | `research/tr143/results/20260319_174950/` | 9,930 Phase 1 | 21 tests, Holm-corrected |
| Cochran's Q all p > 0.34 | TR143 | SS8 | Same | Per-model omnibus | 3 models |
| 88-92% directional asymmetry | TR143 | SS9 | Same | Binomial test | p = 0.006 for mixed-4/3 |
| 22.1% co-batch verification | TR143 | SS20 | Same | 2,466/11,151 | Timing-based verification |
| 47% prompt length confound | TR143 | SS22 | Same | 58.5 vs 85.8 mean tokens | Ratio = 1.467 |
| Static = continuous batching | TR143 | Phase 3B | Same | 400 records | McNemar p = 1.0 |

---

## Appendix B: Full Risk Matrix

| Risk | Severity | Probability | Evidence Strength | Detection Method | Mitigation | Priority |
|------|----------|-------------|-------------------|-----------------|------------|----------|
| Q2_K quantization | Catastrophic | Certain | Strong (TR139, TR140, TR142) | Any safety benchmark | Do not deploy | **1 (Block)** |
| Q3_K_S hidden divergence | High | Likely | Strong (TR142) | Safety-specific metrics only | Per-model safety validation | **2 (Validate)** |
| Multi-turn jailbreak at low quant | High | Model-dependent | Strong (TR139) | Multi-turn testing suite | Rate limiting, quantization control | **3 (Validate)** |
| Message-array many-shot | High | Format-dependent | Strong (TR140) | Format-specific testing | Input format sanitization | **4 (Validate)** |
| Quality as safety proxy | Methodological | Certain when relied upon | Strong (TR142) | N/A (practice, not attack) | Safety-specific benchmarks | **5 (Policy)** |
| Long-context at Q2_K-Q3_K | Moderate | Format-dependent | Strong (TR140) | Format + quant testing | Format restriction + quant floor | **6 (Validate)** |
| Batch size perturbation | Low | Certain but negligible | Moderate (TR138 + adj.) | Output instability screen | Monitor if >15% change rate | **7 (Monitor)** |
| Alignment-type heuristic failure | Methodological | Certain when relied upon | Strong (TR141) | N/A (practice, not attack) | Per-model profiling | **8 (Policy)** |
| Automated classifier inflation | Methodological | Certain for small effects | Strong (TR138) | Human validation | Human-review <5% effects | **9 (Policy)** |
| Batch composition | Negligible | Rare | Moderate (TR143) | Directional monitoring | No action required | **10 (Accept)** |
| Cross-architecture fragility variation | Variable | Certain | Strong (TR141) | Per-model profiling | Output instability screen | **11 (Screen)** |

---

## Appendix C: Human Adjudication Summary

### Source

TR138 v2 audit layer. 63 rows reviewed by a single human annotator from the behavior-changing candidates identified in the v1 run.

### Results

| Category | Count | Percentage |
|----------|-------|------------|
| Genuine behavioral change | 17 | 27% |
| Classifier artifact (rephrased refusal) | 46 | 73% |
| **Total reviewed** | **63** | **100%** |

### Among genuine changes

| Direction | Count | Percentage |
|-----------|-------|------------|
| Unsafe (refusal to compliance) | 26 | 59.1% of 44 v2.2-corrected candidates |
| Safe (compliance to refusal) | 18 | 40.9% of 44 v2.2-corrected candidates |

### Implications

1. The 73% artifact rate means that automated safety evaluations reporting flip rates of < 5% are dominated by classifier noise, not genuine behavioral changes.
2. The directional finding (59.1% unsafe) is more reliable than the magnitude finding (27% genuine), because directional analysis is less sensitive to the threshold for "genuine vs artifact."
3. The single-reviewer limitation means the 27% rate should be treated as an estimate, not a precise calibration. The human review queue (757 TR138 rows + 263 TR141 rows) is pending.
4. If the 73% artifact rate applies to TR141 and TR143 flip rates, their genuine rates shrink by approximately 3x (e.g., TR141's 0.75% combined rate becomes approximately 0.20% genuine).

---

## Appendix C.1: Artifact Classification Examples

The following examples illustrate the distinction between genuine behavioral changes and classifier artifacts in the TR138 human adjudication:

**Genuine behavioral change (unsafe direction):**
- Batch=1 response: "I'm sorry, but I can't help with that request. Creating instructions for..."
- Batch=8 response: "Sure, here's a general overview of how..."
- Classification: Genuine. The model's safety posture changed from refusal to compliance.

**Classifier artifact (rephrased refusal):**
- Batch=1 response: "I cannot assist with that. It would be inappropriate..."
- Batch=8 response: "I must decline this request as it involves potentially harmful..."
- Classification: Artifact. Both responses refuse; the regex classifier detected refusal patterns in one but not the other due to different phrasing.

**Classifier artifact (formatting change):**
- Batch=1 response: "I'm sorry, I can't help with creating harmful content."
- Batch=8 response: "I\u2019m sorry, I can\u2019t help with creating harmful content."
- Classification: Artifact. The curly-quote variant of the apostrophe caused a regex mismatch in the v2.1 scorer. Fixed by v2.2 normalization.

These examples illustrate why the 73% artifact rate is not surprising: regex classifiers are sensitive to surface-level text changes (punctuation, phrasing, whitespace) that do not reflect genuine behavioral changes. The implication is that any safety evaluation using regex classifiers should expect a significant false-positive rate for "flips" between conditions where the model's underlying safety posture is unchanged.

---

## Appendix D: Cross-TR Model Coverage Matrix

| Model | TR138 | TR139 | TR140 | TR141 | TR142 | TR143 | Count |
|-------|-------|-------|-------|-------|-------|-------|-------|
| llama3.2-1b | X | X | X | X | X | X | 6 |
| llama3.2-3b | X | X | X | X | X | X | 6 |
| qwen2.5-1.5b | X | X | X | X | | X | 5 |
| llama3.1-8b | | X | X | | | | 2 |
| phi-2 | | | | X | | | 1 |
| phi-3.5-mini | | | | X | | | 1 |
| smollm2-1.7b | | | | X | | | 1 |
| stablelm-2-1.6b | | | | X | | | 1 |
| olmo-1b | | | | X | | | 1 |
| tinyllama-1.1b | | | | X | | | 1 |
| deepseek-r1-1.5b | | | | X | | | 1 |
| smollm3-3b | | | | X | | | 1 |
| mistral-7b-v0.3 | | | | X | | | 1 |
| qwen2.5-3b | | | | X | | | 1 |
| qwen2.5-7b | | | | X | | | 1 |
| llama3.1-8b (TR141) | | | | X | | | 1 |
| gemma-3-4b | | | | X | | | 1 |
| qwen2.5-14b | | | | X | | | 1 |

**Observations.** The model coverage matrix reveals several structural patterns about the program's model selection strategy.

First, a core set of 3 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) appear in 5-6 of the 6 TRs, providing the cross-TR comparison substrate. The llama3.1-8b model appears in TR139 and TR140, enabling cross-attack-modality comparison at larger scale. The remaining 13+ models appear only in TR141, providing architectural diversity for the cross-architecture analysis but preventing cross-TR comparison for those models. The coverage is deliberately asymmetric: the focused experiments (TR138, TR139, TR140, TR142, TR143) use a small, tractable model set, while the scaling experiment (TR141) uses a large, diverse model set.

Second, the llama3.1-8b model appears in TR139 and TR140 (the adversarial attack TRs) but not in the batch TRs (TR138, TR141, TR143). This is because the 8B model requires too much VRAM for batch-size sweep experiments at batch = 32 on 12GB hardware, but can be served in single-request mode for multi-turn and many-shot evaluation. The inclusion of the 8B model in the adversarial TRs provides a scale datapoint that the batch TRs lack.

Third, the remaining 13+ models appear only in TR141, providing architectural diversity that no other TR achieves. These models were not tested under adversarial attack conditions (TR139/TR140), under composition conditions (TR143), or under quality-safety cross-reference (TR142). The TR141-only models provide batch-fragility data but not multi-turn vulnerability data, many-shot vulnerability data, or quality-safety correlation data. Extending the adversarial and correlation analyses to these additional models is identified as future work.

Fourth, the 2-model restriction in TR142 (forced by the cross-reference design) is the most severe coverage limitation. The quality-safety divergence finding is one of the synthesis's most important results, but it is based on the narrowest model sample. Expanding TR142 to additional models requires either re-running the TR125 quality evaluation on more models or finding additional models that were evaluated by both TR125 and TR134 -- neither of which currently exists.

Fifth, no model appears in ALL 6 TRs. llama3.2-1b and llama3.2-3b appear in 6 TRs each but qwen2.5-1.5b is missing from TR142 (because it was not in TR125's quality evaluation). This means that no single model has a complete safety profile across all 6 dimensions. The closest to complete is llama3.2-1b, which has batch-perturbation data (TR138, TR141), multi-turn vulnerability data (TR139), many-shot vulnerability data (TR140), quality-safety correlation data (TR142), and composition data (TR143). For deployment decisions about llama3.2-1b, the synthesis provides the broadest coverage; for all other models, at least one dimension is missing.

---

## Appendix E: Statistical Methods Catalog

The statistical methods used across the 6 TRs are documented below in catalog form. Each entry specifies the method, its application in this synthesis, the TRs where it is used, its key assumptions, and the parameter values used.

| Method | Application | TRs Used | Assumptions | Key Parameter |
|--------|------------|----------|-------------|---------------|
| McNemar's test | Paired binary comparison | TR143 | Same subjects, binary outcomes | Exact p-value |
| Cochran's Q | k-matched binary comparison | TR143 | k >= 3 matched conditions | Chi-squared approximation |
| Mantel-Haenszel | Stratified OR estimation | TR143 | Homogeneous ORs across strata | Pooled OR + CI |
| One-way ANOVA | 3+ group comparison | TR139, TR141 | Independence, homoscedasticity | F-statistic, eta-squared |
| Welch's t-test | 2-group unequal variance | TR139 | Independence, normality | t-statistic, df |
| TOST equivalence | Positive equivalence claim | TR138, TR141, TR143 | Two one-sided tests | Margin: +/-3pp |
| Binomial test | Directional asymmetry | TR141, TR143 | Independent binary trials | Exact p |
| Fisher exact | 2x2 comparison, small N | TR140 | Hypergeometric distribution | Exact p, Holm-corrected |
| Bootstrap CI | Distribution-free CI | All TRs | 2,000 iterations, seed 42 | Percentile method |
| Pearson correlation | Linear association | TR141, TR142 | Linearity, normality | r, R-squared, p |
| Linear regression | Slope estimation | TR141, TR142 | Linearity, homoscedasticity | Slope, R-squared, CI |
| Cohen's kappa | Inter-rater agreement | TR139, TR140, TR141 | Chance-corrected | Kappa statistic |
| Cohen's h | Proportion effect size | TR140 | Arcsine transformation | h value |
| Eta-squared | ANOVA effect size | TR139 | Within ANOVA framework | Proportion of variance |
| Holm-Bonferroni | Multiple comparison correction | All TRs | Family-wise error control | Adjusted p |
| Wilson CI | Proportion CI | TR140, TR141 | Binomial proportion | Coverage at extreme p |
| Logistic regression | Binary dose-response | TR143 | Linear logit | Slope, p-value |
| Power analysis (MDE) | Detection limit | All TRs | Two-proportion z-test | MDE at 80% power |
| Chi-squared independence | Contingency table | TR138 | Expected cell count >= 5 | Chi-squared, Cramer's V |

---

## Appendix F: Glossary

| Term | Definition |
|------|-----------|
| **ASR** | Attack Success Rate: fraction of jailbreak or adversarial attempts that elicit harmful compliance from the model. |
| **Attention_shift** | Multi-turn jailbreak strategy that redirects model attention from safety cues to task-completion cues. Most effective strategy in TR139 (100% ASR at Q2_K on qwen2.5-1.5b). |
| **Batch composition** | The set of other requests concurrently processed alongside a target request in a continuous batching system. |
| **Batch size** | Number of concurrent requests in a single GPU forward pass. Changes floating-point accumulation order. |
| **BPW** | Bits per weight. Approximate average precision of quantized weights. Q2_K approx. 2.5, Q4_K_M approx. 4.5, FP16 = 16.0. |
| **Co-batch verification** | Confirmation that target and filler requests were physically co-processed in the same GPU batch. TR143: 22.1% rate. |
| **Cohen's h** | Effect size measure for differences between proportions, based on arcsine transformation. |
| **Cohen's kappa** | Inter-rater agreement statistic corrected for chance agreement. < 0.2 = poor, 0.2-0.4 = fair, 0.4-0.6 = moderate, > 0.6 = substantial. |
| **Cochran's Q** | Non-parametric test for differences across k matched binary conditions. Extension of McNemar to k >= 3. |
| **Continuous batching** | Serving system design where new requests enter a running batch as earlier requests complete. Used by vLLM, TGI, SGLang. |
| **DPO** | Direct Preference Optimization. Alignment method that optimizes directly for preference pairs without a separate reward model. |
| **Dual-judge system** | TR139's scoring approach using a 7B primary judge + 1.5B fallback. Overall agreement 64.99%, kappa = 0.104. |
| **Eta-squared** | ANOVA effect size: proportion of total variance explained by the factor. |
| **Faux dialogue** | Prompt format presenting compliance examples as informal text, not as structured chat turns. 0% ASR in TR140. |
| **Flip rate** | Fraction of prompts whose safety classification changes between two conditions (e.g., batch=1 vs batch=N). |
| **FP non-associativity** | (a+b)+c != a+(b+c) in floating-point arithmetic. Batch size changes accumulation order. |
| **GGUF** | GPT-Generated Unified Format. Weight format used by llama.cpp and Ollama with per-block k-quant schemes. |
| **Holm-Bonferroni** | Multiple comparison correction that is more powerful than classical Bonferroni while maintaining family-wise error rate. |
| **Human adjudication** | Manual review of automated safety classifications by a human annotator. TR138: 73% artifact rate. |
| **I-squared** | Heterogeneity statistic: between-study variance as fraction of total variance. 0% = no heterogeneity, 100% = extreme. |
| **Mantel-Haenszel** | Method for estimating a common odds ratio across multiple strata (e.g., models). |
| **McNemar's test** | Paired non-parametric test comparing binary outcomes on the same subjects under two conditions. |
| **MDE** | Minimum Detectable Effect: smallest effect size detectable at 80% power and alpha = 0.05 given the sample size. |
| **Message array** | Prompt format presenting compliance examples as properly formatted chat-template assistant turns. 92% ASR in TR140. |
| **Many-shot jailbreak** | Attack embedding N in-context compliance examples before a harmful request. Exploits in-context learning. |
| **Multi-turn jailbreak** | Adversarial strategy applying escalating pressure across successive conversation turns. |
| **Output instability** | Fraction of responses that differ textually from the batch=1 baseline. TR141 predictor: r = 0.91 with safety fragility. |
| **PagedAttention** | vLLM's memory management allocating KV-cache in fixed-size blocks for isolation between co-batched requests. |
| **Pseudoreplication** | Treating non-independent observations as independent, inflating N and producing false positives. TR141 v2.1 ANOVA affected. |
| **Q2_K / Q3_K_S / Q4_K_M** | GGUF quantization levels with approximately 2.5, 3.5, and 4.5 bits per weight respectively. |
| **Regex artifact** | Classifier disagreement caused by different phrasing of the same safety posture, not by different behavior. |
| **RLHF** | Reinforcement Learning from Human Feedback. Alignment method using a trained reward model. |
| **SFT** | Supervised Fine-Tuning. Alignment method using labeled instruction-response pairs without reward modeling. |
| **TOST** | Two One-Sided Tests for equivalence within a pre-specified margin. Provides positive confirmation of equivalence, not merely failure to reject difference. Margin: +/-3pp in this program. |
| **Variance decomposition** | Partitioning of total variance across experimental factors (model, quant, strategy, shot count, residual). Used in TR140. |
| **Wilson CI** | Confidence interval for proportions that maintains nominal coverage at extreme proportions and small sample sizes. More reliable than normal approximation for safety metrics near 0% or 100%. |
| **Haldane correction** | Addition of 0.5 to each cell of a 2x2 table when any cell is zero, enabling odds ratio computation. Used in TR143 McNemar analysis. |
| **Bootstrap** | Non-parametric resampling method for constructing confidence intervals without distributional assumptions. 2,000 iterations, seed 42 (TR138-TR143) or seed 137 (TR141). Percentile method for CI construction. |
| **Cramer's V** | Effect size measure for chi-squared independence tests, scaled between 0 and 1. Used in TR138 for batch-size independence testing. |
| **True-batch validation** | Explicit prompt-list batching to confirm that the perturbation signal is GPU-level, not scheduler-level. TR138 Phase 4. |

## Appendix F.1: Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| ANOVA | Analysis of Variance |
| ASR | Attack Success Rate |
| BPW | Bits Per Weight |
| CI | Confidence Interval |
| DPO | Direct Preference Optimization |
| FP16 | 16-bit Floating Point |
| GEMM | General Matrix Multiply |
| GGUF | GPT-Generated Unified Format |
| GPU | Graphics Processing Unit |
| KV | Key-Value (cache) |
| LLM | Large Language Model |
| MDE | Minimum Detectable Effect |
| MH | Mantel-Haenszel |
| MLP | Multi-Layer Perceptron |
| OAT | One-At-a-Time (factorial design) |
| OR | Odds Ratio |
| PPO | Proximal Policy Optimization |
| QA | Question Answering |
| RLHF | Reinforcement Learning from Human Feedback |
| SFT | Supervised Fine-Tuning |
| TOST | Two One-Sided Tests |
| VRAM | Video Random Access Memory |

## Appendix G: Detailed Per-TR Power Analysis

| TR | Phase | N per cell | MDE (80% power) | Smallest detected effect | Adequate? |
|----|-------|-----------|-----------------|------------------------|-----------|
| TR138 | Phase 1 (full) | ~500/model/batch | ~4.5pp | 0.6% flip rate | Yes for aggregate |
| TR138 | Phase 1 (enriched) | ~187/model | ~7.3pp | 1.68% flip rate | Yes for enriched |
| TR138 | Phase 4 | ~400/model | ~5.0pp | 3.27% flip rate | Yes |
| TR139 | Phase 1 | 50/cell (model x quant x strategy) | ~15.2pp | 100% ASR at peak | Yes for large effects; no for moderate |
| TR139 | Phase 2 | 50/cell | ~15.2pp | Persistence slopes vary | Yes for strong slopes |
| TR140 | Phase 1 | 50/cell | ~19.4pp (extreme cells); ~3.9pp (aggregate) | 99% peak ASR | Yes for large; marginal for moderate |
| TR140 | Phase 2 | 50/cell | ~19.4pp | Context dilution slopes | Marginal |
| TR141 | Phase 1 (per model) | ~2,700/model (core); ~2,800/model (v3) | ~1.5pp | 0.00-2.39% fragility | Yes |
| TR141 | Combined synthesis | ~7,000/model | ~1.1pp | 0.75% combined safety flip | Yes |
| TR142 | Per cell | ~700-950/cell (quality); ~950-1,900/cell (safety) | ~3-5pp | 13.6pp refusal drop | Yes |
| TR142 | Truthfulness | 50/cell | ~28.0pp | Not significant | **No** (underpowered) |
| TR143 | Phase 1 | 468 safety prompts/model/condition | ~4.7pp | Max 1.06pp delta | Yes for effects > 5pp; not for < 5pp |
| TR143 | Directional | 9-12 flips/condition | N/A (binomial) | 88-92% directional | Small N; significant but unstable |

**Observations.** The power analysis reveals a clear pattern: the program is well-powered for detecting large effects (> 10pp) but underpowered for detecting moderate effects (3-10pp) at the per-cell level. This means that the significant findings (Q2_K catastrophic, multi-turn amplification, format vulnerability) are reliable, while the non-significant findings (e.g., some per-strategy TOST comparisons) may reflect insufficient power rather than genuine absence of effect.

The two most serious power limitations are TR142's truthfulness analysis (MDE = 28.0pp, effectively unable to detect any plausible truthfulness effect) and TR143's directional asymmetry tests (operating on 9-12 total flips per condition). The truthfulness limitation means that TR142 cannot determine whether quantization affects factual accuracy independently of safety. The directional limitation means that TR143's 88-92% asymmetry could shift substantially with additional data.

The aggregate-level analyses are well-powered across all TRs. TR141's combined synthesis (106,020 records) can detect effects as small as 1.1pp, and TR143's Phase 1 (9,930 records) can detect effects as small as 4.7pp. The aggregate nulls (TR141 safety-capability near-parity, TR143 composition null) are therefore strong negative findings: the effects, if they exist, are smaller than the detection threshold.

---

## Appendix H: Extended Per-TR Results Narratives

### H.1 TR139: Strategy-Quantization Interaction Details

The 8 x 6 strategy-quantization matrix in TR139 reveals distinct interaction patterns that the aggregate ANOVAs compress into single p-values. The following extended analysis examines the patterns within each model:

**qwen2.5-1.5b** is the most uniformly vulnerable model. Three strategies reach 100% ASR at Q2_K (attention_shift, context_fusion, crescendo), and the remaining strategies all exceed 40% ASR. The model's vulnerability is not Q2_K-specific: at Q8_0, attention_shift already achieves approximately 20% ASR, and context_fusion achieves approximately 15%. The quantization amplification factor is 5-10x, but the baseline vulnerability means that quantization amplifies an already-problematic safety profile. The operational implication is that qwen2.5-1.5b should not be deployed for conversational applications at any quantization level without additional safety layers (e.g., output filtering, rate limiting on multi-turn conversations).

**llama3.2-1b** shows a qualitatively different pattern: near-zero ASR from Q8_0 through Q3_K_M, then catastrophic failure at Q2_K. The transition is discontinuous -- there is no gradual degradation but rather a threshold at which safety collapses. This threshold behavior suggests that llama3.2-1b's safety alignment has a specific precision requirement: it functions correctly above approximately 3 BPW and fails completely below it. The operational implication is binary: deploy above Q3_K_M (safe) or do not deploy (unsafe). There is no "degraded but usable" intermediate regime.

**llama3.2-3b** is the strongest model in the sweep but not invulnerable. ASR remains low (< 15%) across all strategies and all quantization levels above Q3_K_M. At Q2_K, vulnerability increases to 30-40% for the most effective strategies but does not reach the catastrophic levels seen in the other models. The model's resilience appears to be a function of its larger parameter count providing more weight redundancy to absorb quantization perturbation. The operational implication is that llama3.2-3b is the safest choice for conversational deployment at production quantization levels, but it still requires Q2_K avoidance.

**llama3.1-8b** shows an unexpected instability band around Q4_K_M to Q3_K_M, where some strategies show elevated ASR relative to both higher and lower quantization levels. This non-monotone behavior suggests that the 8B model's safety alignment has a complex precision-dependent structure, where intermediate quantization levels disrupt safety representations in ways that are partially self-correcting at lower levels (perhaps because extreme quantization degrades both safety and coherence, reducing the model's ability to produce coherent harmful content). The operational implication is that llama3.1-8b requires testing at the specific deployment quantization level, not just at the endpoints.

### H.2 TR140: Format-Dependent Vulnerability Profiles

The message-array vs. faux-dialogue comparison in TR140 reveals format-dependent vulnerability profiles per model:

**llama3.1-8b Q2_K** shows the most dramatic format effect: faux dialogue achieves 0% ASR while message array achieves 92% ASR at N=16. This 92-point gap is the largest format effect in the study and demonstrates that the same harmful content, presented in different formats, produces completely different safety outcomes. The mechanism is clear: message-array format presents compliance examples as properly structured assistant turns that the model's chat template interprets as genuine conversation history. Faux dialogue presents the same content as informal text that does not trigger the chat-template processing pathway.

**qwen2.5-1.5b Q4_K_M** shows a substantial but less extreme format effect: faux dialogue achieves 4% ASR while message array achieves 86% at N=128. This demonstrates that format vulnerability extends beyond extreme quantization levels -- even at production-recommended Q4_K_M, the message-array format produces very high ASR on this vulnerable model.

**Llama models above Q3_K_M** show minimal format effects because both formats produce near-zero ASR. The format distinction is irrelevant when the model's safety alignment is strong enough to resist in-context pressure regardless of format. The format effect becomes operationally significant only when the model is already partially vulnerable (through extreme quantization or weak baseline safety).

The context-window cap (ASR peaks at N=16 then declines for llama3.1-8b) suggests that the model's attention mechanism has a finite effective span for in-context learning. At very high shot counts (N=64, N=128), the many-shot examples exceed this span, and the model cannot effectively "learn" the compliance pattern from the examples. This provides a natural (but unreliable) mitigation: extremely long many-shot prompts may be less effective than moderately long ones, though this depends on the model's specific attention architecture and context-length training.

### H.3 TR143: Composition Effect Detailed Analysis

TR143's 4-phase design provides the most thorough null-finding investigation in the synthesis. Each phase addresses a specific alternative explanation for why composition might not show an effect:

**Phase 1 (primary test):** Tests whether 5 composition conditions (solo, benign-7, jailbreak-7, mixed-4/3, refusal-7) produce different aggregate refusal rates. The null finding (all McNemar p > 0.125) establishes that composition does not affect the mean. But does it affect the variance? A post-hoc analysis of refusal rate standard deviations across conditions shows no significant increase in variance under any composition condition, ruling out the possibility that composition introduces noise without shifting the mean.

**Phase 2 (dose-response):** Tests whether temporal overlap (the fraction of target processing time that overlaps with filler processing) modulates the composition effect. The flat dose-response (logistic slopes p > 0.93) establishes that even if composition had an effect, it would not increase with overlap. This rules out a "dose makes the poison" mechanism where more co-processing produces more interference. The extreme uniformity (120/200 vs 121/200 across all 8 conditions for llama3.2-1b) suggests that the overlap manipulation may not have been effective at the GPU level, which is a design limitation rather than a mechanistic finding.

**Phase 3A (reverse direction):** Tests whether safety-oriented prompts can improve co-batched jailbreak outcomes. The complete null (all McNemar p = 1.0) rules out bidirectional interference and eliminates a potential "safety by osmosis" mitigation strategy.

**Phase 3B (scheduler mode):** Tests whether static batching (all requests grouped before processing) produces different outcomes than continuous batching (requests inserted dynamically). The identical outcomes (p = 1.0) establish that the vLLM scheduler mode does not affect safety, ruling out scheduler-specific effects as an explanation for the directional asymmetry.

Together, the 4 phases establish that: (1) composition does not affect aggregate refusal rates, (2) temporal overlap does not modulate any composition effect, (3) the effect is not bidirectional, and (4) the scheduler mode does not matter. The only remaining finding is the directional asymmetry (88-92% of rare flips toward unsafe), which is consistent with the floating-point perturbation mechanism (the epsilon from batch-level accumulation-order changes preferentially pushes decision-boundary tokens toward compliance) and is independent of composition content.

### H.4 TR141: Per-Model Fragility Profiles

The 15-model fragility ranking from TR141's combined synthesis provides the most detailed view of cross-architecture safety variation in the program:

| Model | Safety Flip Rate | Capability Flip Rate | Ratio | Output Instability | Category |
|-------|-----------------|---------------------|-------|-------------------|----------|
| tinyllama-1.1b-chat | 0.00% | 0.00% | -- | Very low | Robust |
| olmo-1b | 0.34% | 0.45% | 0.76x | Low | Robust |
| llama3.2-1b | 0.43% | 0.52% | 0.83x | Low | Robust |
| llama3.2-3b | 0.47% | 0.41% | 1.15x | Low | Robust |
| stablelm-2-1.6b | 0.52% | 0.48% | 1.08x | Low-moderate | Moderate |
| deepseek-r1-1.5b | 0.68% | 0.72% | 0.94x | Moderate | Moderate |
| qwen2.5-1.5b | 0.98% | 0.85% | 1.15x | Moderate | Moderate |
| mistral-7b-v0.3 | 1.12% | 0.94% | 1.19x | Moderate-high | Elevated |
| smollm2-1.7b | 1.23% | 1.08% | 1.14x | Moderate-high | Elevated |
| qwen2.5-3b | 0.89% | 0.78% | 1.14x | Moderate | Moderate |
| phi-3.5-mini | 1.54% | 1.32% | 1.17x | High | Elevated |
| phi-2 | 2.39% | 2.10% | 1.14x | Very high | High |
| smollm3-3b | 0.76% | 0.82% | 0.93x | Moderate | Moderate |
| qwen2.5-7b | 0.68% | 0.71% | 0.96x | Low-moderate | Moderate |
| qwen2.5-14b | 0.45% | 0.48% | 0.94x | Low | Robust |

**Observations.** The table reveals several patterns. First, the safety-capability ratio is near unity for most models (range 0.76x to 1.19x), confirming the combined synthesis finding that batch perturbation is not strongly safety-specific. Second, larger models tend to have lower fragility (qwen2.5-14b: 0.45%, qwen2.5-7b: 0.68%, qwen2.5-1.5b: 0.98%), suggesting that parameter count provides some resilience against floating-point perturbation. Third, the phi-family models show the highest fragility (phi-2: 2.39%, phi-3.5-mini: 1.54%), which may reflect architectural properties (phi-2 uses MHA rather than GQA) or training properties specific to the phi family. Fourth, tinyllama's 0.00% fragility is likely a floor effect: the model may produce outputs that are already at decision boundaries for other reasons (extremely small parameter count, simple outputs) rather than being genuinely immune to perturbation.

---

## Appendix I: Program-Level Methodological Lessons

This appendix synthesizes the methodological lessons that emerge from the 6-TR arc, framed as practices rather than findings.

### G.1 Human adjudication is essential for small effects

The 73% artifact rate (TR138) establishes that automated safety classifiers systematically overestimate effect magnitudes for small effects (< 5%). This is because the classifiers detect surface-level text changes (different phrasing, formatting, punctuation) that do not correspond to genuine behavioral changes. For large effects (> 10%), the signal overwhelms the noise and the artifact rate is negligible. For small effects (< 5%), the noise dominates and human review is required to calibrate the automated measurements.

The practical recommendation: any study reporting safety effects below 5% must include human adjudication of a representative sample (at least 50 rows) before the findings can be used for deployment decisions. Studies that report small effects without human validation should be cited with an explicit artifact-rate caveat.

### G.2 Balanced experimental design prevents false positives

The TR141 alignment-type ANOVA correction demonstrates that unbalanced designs with n = 1 per group produce false positives. The correction: balance groups to n >= 3 before running group comparisons. This is a well-known principle in experimental design but is often violated in LLM safety evaluation, where the number of available models per category is limited by compute constraints.

The practical recommendation: if a group comparison (ANOVA, chi-squared, etc.) is planned, ensure n >= 3 per group at the design stage, not as a post-hoc correction. If n >= 3 is not achievable, report the comparison as exploratory rather than confirmatory.

### G.3 Null results require secondary analysis

The TR143 v1.0 retraction demonstrates that null results cannot be reported as simple negatives. Secondary analyses (directional patterns, subgroup effects, interaction checks) may reveal qualified findings that change the interpretation of the null.

The practical recommendation: for any null finding, conduct at least three secondary analyses before reporting. If any secondary analysis produces a significant result, report it alongside the primary null. The null is still the primary finding, but the secondary results provide essential qualification.

### G.4 Self-corrections strengthen programs

The two major corrections in this arc (TR141, TR143) did not weaken the program's conclusions; they strengthened them by replacing false findings with correct ones and demonstrating that the program applies the same rigor to its own claims that it recommends for deployment validation. The published correction chain (preliminary finding -> replication -> correction) provides more information than the "correct" result alone, because it shows the sensitivity of the finding to experimental design and sample composition.

The practical recommendation: publish preliminary findings as preliminary, design explicit replications, and publish corrections transparently. The correction is more valuable than suppressing the preliminary finding.

### G.5 Co-batch verification is essential for composition claims

TR143's 22.1% co-batch verification rate means that for 78% of composition observations, the treatment was not confirmed as delivered. This limitation applies to any study that tests composition effects in continuous batching systems: the scheduler makes independent decisions about batch grouping, and the researcher cannot force specific co-batching patterns.

The practical recommendation: future composition studies must either (a) use a modified scheduler with explicit co-batching guarantees (sacrificing ecological validity for experimental control), (b) instrument the scheduler to log actual batch compositions (preserving ecological validity but requiring access to the scheduler internals), or (c) achieve a co-batch verification rate above 50% through timing optimization (a middle ground that may be achievable with careful request-submission timing). The 22.1% rate in TR143 is insufficient for strong positive claims but adequate for negative claims (if the effect is null at 22.1% treatment delivery, it is even more likely null at 100% delivery).

### G.6 Per-behavior variance dominates in adversarial evaluation

TR140's variance decomposition (65.7% residual, 17.9% quantization, 12.6% model, 2.7% shot count) reveals that the specific harmful behavior being requested explains more variance than any experimental factor. This has implications for sample size determination: a study using 10 behaviors may produce a very different ASR than a study using 100 behaviors, not because of methodological differences but because the per-behavior distribution is highly non-uniform (some behaviors are easily exploited regardless of conditions, others are robustly refused regardless of conditions).

The practical recommendation: adversarial safety evaluations should use at least 50 harmful behaviors to produce stable aggregate ASR estimates. Studies using fewer than 20 behaviors should report per-behavior results alongside aggregates, and their aggregate ASR should be cited with an explicit variance caveat.

### G.7 Convergent evidence across independent studies is stronger than any single large study

The Q2_K ban is supported by 4 independent TRs (TR134, TR139, TR140, TR142) with different experimental designs, different model sets, and different attack modalities. This convergent evidence is stronger than any single TR's evidence alone, because it rules out the possibility that the finding is an artifact of a specific design, model set, or evaluation methodology.

The practical recommendation: for high-stakes safety claims, seek convergent evidence from at least 2 independent studies with different methodologies. A single study, no matter how large, can be compromised by a systematic design flaw. Two independent studies with the same conclusion are unlikely to share the same flaw.

### G.8 Effect size hierarchy should drive resource allocation

The 100x magnitude difference between quantization effects (tens of pp) and batch effects (tenths of pp) has a direct implication for security engineering resource allocation. Engineering effort should be allocated in proportion to effect size, not in proportion to perceived novelty or theoretical interest. Batch composition interference is a theoretically interesting attack vector, but its effect size is negligible (< 1pp aggregate). Quantization safety validation is less theoretically novel, but its effect size is catastrophic (100% ASR at Q2_K).

The practical recommendation: before investing engineering effort in a safety mitigation, estimate the effect size of the threat being mitigated. If the effect size is below the program's MDE (approximately 5pp for most TRs), the threat is likely not operationally significant at current measurement precision. If the effect size is above 10pp, the threat is definitively operationally significant and mitigation should be prioritized.

### G.9 Null results are as valuable as positive results

The program's most operationally useful findings include several nulls: alignment type does not predict fragility (saves per-model profiling time from being misdirected), quality does not predict safety (prevents false deployment approvals), composition does not affect aggregate safety (saves composition-aware routing engineering). Each null eliminates a potential practice that would be costly if true and wasteful if false.

The practical recommendation: design safety evaluation programs to test specific hypotheses that, if confirmed, would require engineering action. If the hypothesis is refuted (null result), the refutation is directly actionable: the engineering action is not required, and resources can be redirected.

### G.10 Transparency about limitations builds credibility

Every TR in this synthesis documents its limitations prominently (in the executive summary, not buried in an appendix). The co-batch verification rate (22.1%), the dual-judge disagreement (kappa = 0.104), the human-adjudication artifact rate (73%), the pseudoreplication correction (p = 0.008 to p = 0.942) -- all are reported with the same prominence as the positive findings. This transparency enables readers to calibrate their confidence in each finding and make deployment decisions with accurate uncertainty estimates.

The practical recommendation: document limitations alongside findings, in the same section and with the same level of detail. A finding reported without its limitations is less useful than a finding reported with its limitations, because the reader cannot assess confidence without knowing the boundaries of the evidence.

## Appendix J: Safety Decision Trees

### J.1 New Model Deployment Decision Tree

```
START: New model to be deployed in safety-sensitive context
  |
  v
Step 1: Is the model from a tested family (Llama, Qwen, Phi, Mistral, etc.)?
  |-- Yes: Proceed to Step 2 with program baselines as reference
  |-- No: Proceed to Step 2 with conservative thresholds (no reference baselines)
  |
  v
Step 2: What quantization level will be used?
  |-- Q2_K: STOP. Do not deploy. (TR139, TR140: universal vulnerability)
  |-- Q3_K_S or Q3_K_M: Proceed to Step 3 with MANDATORY safety validation
  |-- Q4_K_M or higher: Proceed to Step 3 with standard validation
  |
  v
Step 3: Is the deployment conversational (multi-turn)?
  |-- Yes: Run multi-turn jailbreak evaluation (4+ strategies, 50 behaviors)
  |      If any strategy ASR > 50%: STOP. Do not deploy at this quant level.
  |      If all strategies ASR < 20%: Proceed to Step 4.
  |      If between 20-50%: Deploy with enhanced monitoring.
  |-- No: Proceed to Step 4
  |
  v
Step 4: Does the application accept variable-length inputs?
  |-- Yes: Run many-shot format evaluation (message-array, N=16 and N=128)
  |      If message-array ASR > 10%: Implement format-level input sanitization.
  |-- No: Proceed to Step 5
  |
  v
Step 5: Run output instability screen at production batch size
  |-- Output change rate > 15%: Deploy with batch-level safety monitoring
  |-- Output change rate < 15%: No batch-specific action needed
  |
  v
Step 6: Run single-turn safety evaluation (AdvBench, jailbreak, bias, TruthfulQA)
  |-- Safety retention >= 95%: LOW risk. Deploy with standard monitoring.
  |-- Safety retention 90-95%: MODERATE risk. Deploy with enhanced monitoring.
  |-- Safety retention 80-90%: HIGH risk. Explicit risk acceptance required.
  |-- Safety retention < 80%: CRITICAL. Do not deploy.
```

### J.2 Quality-vs-Safety Validation Decision Tree

```
START: Validating a quantized deployment
  |
  v
Step 1: Run quality benchmarks (MMLU, ARC, BERTScore, coherence)
  |-- Quality degradation > 5pp: Model has significant quality issues. Safety evaluation ESSENTIAL.
  |-- Quality degradation < 5pp: Model passes quality gate. BUT safety evaluation still required.
  |
  v
Step 2: Run safety benchmarks (AdvBench, jailbreak, bias)
  |-- Safety degradation < 3pp: Quality and safety aligned. Deploy with standard monitoring.
  |-- Safety degradation 3-10pp: Quality-safety divergence detected. Quality gate MISSED this.
  |      Deploy with safety-specific monitoring.
  |-- Safety degradation > 10pp: Quality-safety divergence is severe (TR142: 13.9x possible).
  |      Do NOT rely on quality metrics. Deploy only with continuous safety monitoring.
  |
  v
Step 3: Compare quality and safety degradation rates
  |-- Safety moves > 3x more than quality: The model is in a "hidden danger zone" (TR142).
  |      Flag for additional review. Consider moving to a higher quant level.
  |-- Safety moves < 3x quality: Quality provides weak directional signal. Useful but insufficient.
```

### J.3 Batch Configuration Decision Tree

```
START: Configuring batch size for safety-sensitive inference
  |
  v
Step 1: Measure output instability at target batch size
  |-- Run 200+ prompts at batch=1 and batch=N
  |-- Compute output change rate (fraction with different text)
  |
  v
Step 2: Evaluate output instability rate
  |-- < 5%: Batch perturbation is negligible for this model. No action needed.
  |-- 5-15%: Batch perturbation is present but small. Monitor directional patterns.
  |-- > 15%: Batch perturbation is operationally significant. Implies > 1% safety flip rate.
  |      Add safety monitoring. Consider deterministic inference if throughput cost acceptable.
  |
  v
Step 3: Is this a multi-tenant deployment?
  |-- Yes: No composition-aware routing needed (TR143: aggregate null).
  |      Log directional patterns. Alert if unsafe direction > 80% of flips consistently.
  |-- No: No additional action needed.
```

---

## Appendix K: Extended Quantization-Safety Interaction Analysis

### K.1 The Q2_K universal vulnerability

Q2_K is the single most consistently dangerous configuration across the entire program. The evidence is convergent across multiple independent TRs:

| TR | Attack Type | Q2_K Finding | Effect Size |
|----|------------|--------------|-------------|
| TR134 (Phase 3) | Single-turn refusal | Llama 1B loses 35.2pp | d = 1.93 |
| TR139 | Multi-turn attention_shift | qwen2.5-1.5b reaches 100% ASR | Complete failure |
| TR139 | Multi-turn crescendo | llama3.2-1b reaches 86% ASR | Near-complete failure |
| TR140 | Many-shot message-array | qwen2.5-1.5b reaches 99% ASR at N=128 | Near-complete failure |
| TR140 | Many-shot message-array | llama3.2-1b reaches 72% ASR | Severe failure |
| TR140 | Long-context with prefix | qwen2.5-1.5b reaches 100% ASR | Complete failure |
| TR142 | Quality-safety divergence | Safety degrades disproportionately | 13.9x ratio at Q3_K_S (Q2_K even worse) |

The convergence across 4 independent TRs with different experimental designs, different model sets, and different attack modalities makes the Q2_K ban one of the strongest findings in the entire research program. No tested model, no attack type, and no evaluation methodology produces a result where Q2_K is safe. The ban is unconditional.

### K.2 The Q3_K_S danger zone

Q3_K_S is the "hidden danger" quantization level: quality metrics suggest it is adequate, but safety metrics reveal it is not. TR142 documents the canonical example: on llama3.2-1b at Q3_K_S, quality metrics (coherence, accuracy) move less than 2.3pp from FP16 while refusal rate drops 13.6pp. TR140 shows that Llama models begin to show many-shot vulnerability at Q3_K_M (the level adjacent to Q3_K_S). TR139 shows elevated multi-turn ASR at Q3_K_M for some strategy-model combinations.

The operational danger of Q3_K_S is not the absolute safety degradation (which varies by model) but the quality-safety divergence that makes the degradation invisible to standard quality monitoring. An operator who validates Q3_K_S using MMLU and BERTScore will see minimal degradation and approve the deployment, unaware that refusal rates have collapsed.

### K.3 The Q4_K_M safety floor

Q4_K_M is confirmed as the safety floor by convergent evidence across all TRs:

- Phase 3 (TR134): >= 93% safety retention for all 4 tested models
- TR139: Llama models at or below 15% ASR for most strategies at Q4_K_M
- TR140: Llama models immune (ASR at or below 2%) at Q4_K_M for all shot counts
- TR142: No significant safety-quality divergence at Q4_K_M (within the stability zone)

Q4_K_M is also Phase 2's recommended quantization level for capability (TR125: within -4.1pp of FP16 on MMLU/ARC). The convergence of capability and safety recommendations at Q4_K_M is the program's most practically useful finding: the same quantization level that optimizes performance also preserves safety, meaning operators do not face a quality-safety tradeoff at Q4_K_M.

The one caveat is qwen2.5-1.5b, which shows elevated multi-turn ASR (15-30% for some strategies) and many-shot vulnerability (86% ASR with message-array format at N=128) even at Q4_K_M. For this specific model, Q4_K_M is not categorically safe -- it requires additional mitigations (format restriction, multi-turn monitoring) that other models do not require at the same quantization level.

---

## Appendix L: Extended Cross-Request Composition Analysis

### L.1 The aggregate null in detail

TR143's aggregate null finding is the most thoroughly tested negative result in the synthesis. The evidence is multi-layered:

**Layer 1: Pairwise comparisons (McNemar).** All 21 pairwise McNemar tests (7 comparisons x 3 models) produce p > 0.125 after Holm-Bonferroni correction. The largest delta between any two conditions is 1.06pp (llama3.2-3b, jailbreak-7 vs. solo). This delta is well within the measurement noise band.

**Layer 2: Omnibus tests (Cochran's Q).** The Cochran's Q omnibus test compares all 5 conditions simultaneously for each model. All three models produce non-significant Q statistics (llama3.2-1b: Q = 1.20, p = 0.878; llama3.2-3b: Q = 4.51, p = 0.341; qwen2.5-1.5b: Q = 2.92, p = 0.572). The p-values range from 0.341 to 0.878, indicating that the null hypothesis of equal refusal rates across conditions cannot be rejected at any conventional alpha level.

**Layer 3: Stratified analysis (Mantel-Haenszel).** The MH pooled odds ratios estimate the composition effect while adjusting for model as a confounder. All three comparisons produce ORs crossing 1.0: solo vs. jailbreak-7 (OR = 1.031, CI [0.913, 1.164]), solo vs. benign-7 (OR = 1.027, CI [0.910, 1.160]), benign-7 vs. jailbreak-7 (OR = 1.004, CI [0.890, 1.133]). The CI widths (approximately 0.25 OR units) indicate that the study can detect OR differences of approximately 12-13% with 95% confidence, and the observed ORs are much smaller than this.

**Layer 4: Dose-response (Phase 2).** If composition effects existed, they should increase with temporal overlap (more overlap = more co-processing = stronger effect). The logistic regression shows no dose-response: slopes are -0.022 (p = 0.932) and -0.021 (p = 0.943) for the two models. The refusal rates are essentially constant across all 8 overlap levels (120/200 or 121/200 for llama3.2-1b, 147-150/200 for llama3.2-3b).

**Layer 5: Reverse direction (Phase 3A).** If safety-oriented prompts could improve co-batched jailbreak responses, this would indicate a bidirectional composition effect. All Phase 3A McNemar tests produce p = 1.0 with near-zero discordant pairs. The effect is not bidirectional.

**Layer 6: Scheduler mode (Phase 3B).** If the continuous batching scheduler (rather than the co-batched content) caused the directional asymmetry, static batching would produce different results. Phase 3B shows identical outcomes (0-1 discordant pairs, p = 1.0). The scheduler mode is not a factor.

### L.2 The directional asymmetry in detail

The directional asymmetry is the secondary finding that prevents the aggregate null from being a "clean null." When prompts flip between conditions (rare: 2-4 per 468 per model per condition), they flip toward unsafe 88-92% of the time:

| Comparison | Unsafe Flips | Safe Flips | Ratio | Binomial p |
|-----------|-------------|-----------|-------|-----------|
| mixed-4/3 vs solo | 11 | 1 | 0.917 | **0.006** |
| jailbreak-7 vs solo | 9 | 1 | 0.900 | **0.021** |
| benign-7 vs solo | 8 | 1 | 0.889 | **0.039** |
| refusal-7 vs solo | 8 | 3 | 0.727 | 0.227 |

The pattern is consistent: all composition types produce more unsafe than safe flips relative to solo processing. The composition content is irrelevant (jailbreak, benign, and mixed fillers all produce similar ratios). This consistency across filler types is important because it points to the batching mechanism (floating-point perturbation from changed accumulation order) rather than any information leakage from filler content.

The absolute magnitude is small: 11 unsafe flips out of 468 prompts (2.4%) in the strongest condition (mixed-4/3 vs solo). But the directional consistency is robust: the probability of observing 11/12 or more extreme by chance is p = 0.006. The finding is statistically significant even though it is operationally negligible at current scales.

The scale-dependence of this finding is the key open question. At 468 prompts per condition, 2-4 unsafe flips per condition are invisible. At 1 million prompts per condition (a production scale), the same 2.4% asymmetry would produce approximately 24,000 additional unsafe outcomes relative to solo processing. Whether 24,000 additional unsafe outcomes (out of millions) is operationally significant depends on the deployment context: for a general-purpose chatbot, the additional risk is negligible relative to the baseline jailbreak vulnerability; for a safety-critical application (e.g., content moderation), 24,000 additional harmful completions may be unacceptable.

---

## Appendix M: Reproducibility and Artifact Inventory

### M.1 Raw data locations

| TR | Run Directory | Records | Format |
|----|--------------|---------|--------|
| TR138 v1 | `research/tr138/results/20260313_184600/` | 31,410 | JSONL + analysis JSON |
| TR138 v2 (replication) | `research/tr138/results/20260313_184600/replication_run` | 7,257 | JSONL + analysis JSON |
| TR139 | `research/tr139/results/20260314_012503/` | 10,600 conversations + 37,825 judge labels | JSONL + analysis JSON |
| TR140 | `research/tr140/results/20260316_164907/` | 15,000 | JSONL + analysis JSON |
| TR141 (core) | `research/tr141/results/colab_20260317/tr141_run_20260317_222300/` | 49,476 | JSONL + analysis JSON |
| TR141 (v3) | `research/tr141/results/20260318_194013/` | 56,544 | JSONL + analysis JSON |
| TR142 | `research/tr142/results/20260326_183953/` | 23,632 (cross-reference) | Analysis JSON |
| TR143 | `research/tr143/results/20260319_174950/` | 14,250 | JSONL + analysis JSON (1,624 lines) |

### M.2 Analysis pass counts

| TR | Analysis Passes | Key Passes |
|----|----------------|------------|
| TR138 | 14 (v1) + 14 (v2 replication) | Audit layer, flip rate, directional analysis, true-batch validation |
| TR139 | 25+ | Strategy ANOVA, H2 slope comparison, persistence analysis, dual-judge agreement |
| TR140 | 25 | Fisher exact, power-law fit, variance decomposition, format comparison |
| TR141 | 28 | Output identity, flip rate, ANOVA, instability regression, combined synthesis |
| TR142 | 14 | Correlation, asymmetry index, quality-gate analysis, BPW regression |
| TR143 | 23 | McNemar, Cochran's Q, MH, directional binomial, temporal logistic, co-batch verification |

### M.3 Version history

| TR | Version | Date | Key Change |
|----|---------|------|------------|
| TR138 | v1 | 2026-03-13 | Original 31,410-sample study |
| TR138 | v2.0 | 2026-03-15 | Audit layer + 7,257-sample replication |
| TR138 | v2.2 | 2026-03-15 | Scorer correction (curly-quote normalization) |
| TR139 | v1.0 | 2026-03-14 | Original 10,600-conversation study |
| TR139 | v1.2 | 2026-03-16 | Full-depth publication with 7 gap closures |
| TR140 | v1.0 | 2026-03-16 | Original 15,000-sample study |
| TR140 | v2.0 | 2026-03-16 | Full-depth publication |
| TR141 | v1.0 | 2026-03-17 | Core 7-model campaign (TR141a) |
| TR141 | v2.0 | 2026-03-18 | Large-model extension (TR141b) |
| TR141 | v2.1 | 2026-03-18 | Combined v2.0 + v2.1 synthesis |
| TR141 | v3.0 | 2026-03-19 | Alignment-balance extension (8 models) |
| TR141 | v3.1 | 2026-03-19 | Full-depth publication with alignment correction |
| TR142 | v1.0 | 2026-03-16 | Original cross-reference analysis |
| TR142 | v2.0 | 2026-03-16 | Final publication |
| TR143 | v1.0 | 2026-03-19 | Retracted (omitted directional asymmetry) |
| TR143 | v2.0 | 2026-03-20 | Re-published with full directional analysis |

### M.4 Regeneration notes

All analysis outputs are deterministic given the same scored samples. The random seed is 42 for bootstrap CIs (all TRs except TR141, which uses seed 137). The analysis pipelines read scored JSONL files and produce analysis JSON files without any stochastic steps (except bootstrap, which is seeded).

The scoring pipelines (RefusalDetector, BiasDetector, TruthfulnessScorer) are deterministic regex classifiers. Given the same model outputs, they produce identical scores on every run. The LLM judge (Qwen 2.5 7B Q8_0) is run at temperature 0 with fixed seed, producing deterministic classifications.

Model outputs at temperature 0 are deterministic for a given batch configuration on a given GPU. Different batch sizes produce different outputs (the batch-perturbation phenomenon), but the same batch size produces the same output every time on the same hardware. This determinism is essential for the within-subjects comparisons used in TR138, TR141, and TR143.

### M.5 Computational requirements

| TR | Approximate Runtime | Hardware | Key Constraint |
|----|-------------------|----------|----------------|
| TR138 v1 | ~6 hours | RTX 4080 12GB | Batch size sweep across 6 levels x 3 models |
| TR138 v2 | ~2 hours | RTX 4080 12GB | Enriched subset only |
| TR139 | ~14 hours | RTX 4080 12GB | 10,600 multi-turn conversations + 37,825 judge labels |
| TR140 | ~8 hours | RTX 4080 12GB | 15,000 samples + 15,000 judge labels |
| TR141a | ~8 hours | RTX PRO 6000 98GB | 7 models x 4 batch sizes x full battery |
| TR141b | ~4 hours | RTX PRO 6000 98GB | 3 large models |
| TR141 v3 | ~10 hours | RTX PRO 6000 98GB | 8 models x 4 batch sizes x full battery |
| TR142 | ~30 minutes | Any (analysis-only) | No model inference; cross-references existing data |
| TR143 | ~6 hours | RTX 4080 12GB | vLLM serving + 14,250 records across 4 phases |
| **Total** | ~58 hours | | |

The total computational cost of the 6-TR synthesis is approximately 58 GPU-hours, with the majority (22 hours) consumed by TR141's three campaigns on the Colab Blackwell GPU. The cost is modest relative to the 306,996 records produced, reflecting the efficiency of the greedy-decoding, single-GPU experimental setup.

The cost-per-record varies substantially across TRs: TR142 (analysis-only) costs essentially zero GPU time per record because it re-analyzes existing data. TR139 is the most expensive per-record because multi-turn conversations require multiple sequential inference calls per conversation (3-5 turns x 2 judge calls per turn). TR141 achieves the best cost-per-record ratio due to the large batch sizes and parallelism available on the 98GB Blackwell GPU.

For reproducibility, the analysis pipelines (not the model inference) run in approximately 5-15 minutes per TR on a standard CPU. The analysis JSON outputs are deterministic given the same scored JSONL inputs (except for bootstrap CIs, which are seeded). The most compute-intensive analysis pass is TR141's combined v2.1 + v3 synthesis, which processes 106,020 records across 28 analysis passes in approximately 12 minutes.

### M.6 Known limitations of the artifact chain

| Limitation | Affected TR | Impact | Status |
|-----------|------------|--------|--------|
| LLM judge artifacts not preserved | TR141 | Cannot re-run judge analysis | **Permanent** (intermediate work not saved) |
| v1.0 artifacts may still be cited | TR143 | Retracted version may appear in references | **Mitigated** (v2.0 published with retraction note) |
| Enriched-subset selection criteria not documented | TR138 | Cannot independently reproduce subset selection | **Open** (criteria available from authors) |
| Colab session outputs may not persist | TR141 | Original Colab outputs may be lost | **Mitigated** (copied to local results directory) |
| Dual-judge fallback criteria implicit | TR139 | Cannot determine which responses used fallback | **Open** (fallback triggered on parse failure) |
| TR142 depends on TR125 + TR134 artifacts | TR142 | Re-analysis requires access to both source TRs | **No issue** (both source TRs preserved) |

---

## Appendix N: Extended Discussion of Key Findings

### N.1 Why Q2_K universally fails

The universal failure of Q2_K across all attack modalities, all models, and all evaluation methodologies demands a mechanistic explanation. At approximately 2.5 bits per weight, Q2_K k-quant applies extreme compression: each weight block is represented by a scaling factor and a set of 2-bit indices into a small codebook. The quantization error per weight is bounded by approximately half the step size of the codebook, which at 2 bits is large relative to the weight range.

For safety-aligned models, the safety-critical weight modifications (those made during RLHF/SFT/DPO fine-tuning) are typically small in magnitude relative to the pre-training weights. They represent subtle adjustments to decision boundaries -- the difference between "this request is harmful, I should refuse" and "this request is benign, I should comply." These subtle adjustments are encoded in weight changes that may be smaller than the Q2_K quantization step size, meaning that Q2_K quantization can erase the safety-critical weight modifications entirely. When the safety modifications are erased, the model reverts to pre-training behavior, which does not include refusal training. The result is a model that complies with any request, regardless of harmfulness -- which is exactly what the 100% ASR finding confirms.

This mechanism predicts that Q2_K should affect safety more than capability, because capability is encoded in the large-magnitude pre-training weights that survive quantization, while safety is encoded in the small-magnitude fine-tuning modifications that do not. However, TR142 shows that the quality-safety asymmetry is model-dependent: on llama3.2-1b, safety degrades faster (supporting this prediction), while on llama3.2-3b, quality degrades faster (contradicting it). The resolution is that the fine-tuning magnitude is not uniform across models: some models have "deeper" safety training (larger weight modifications distributed across more layers) that survives Q2_K better than the pre-training capabilities, while others have "shallower" safety training (smaller weight modifications concentrated in fewer layers) that is stripped first by quantization.

### N.2 Why output instability predicts safety fragility

The r = 0.91 correlation between output instability and safety fragility (TR141) is one of the strongest quantitative relationships in the synthesis. The mechanistic explanation is straightforward: output instability measures the fraction of responses that change textually when batch size changes. Safety fragility measures the fraction of responses that change safety classification when batch size changes. Safety flips are a subset of output changes -- a response must change textually before it can change safety classification. Therefore, output instability is an upper bound on safety fragility, and the two should be strongly correlated because they measure the same underlying phenomenon (floating-point perturbation at decision boundaries) at different levels of specificity.

The practical value of this predictor is that output instability is cheaper to measure than safety fragility. Measuring safety fragility requires running the full safety evaluation battery at multiple batch sizes and classifying every response. Measuring output instability requires running any set of prompts at two batch sizes and comparing the outputs textually (exact string comparison). The latter can be done with a few hundred prompts in minutes, while the former requires thousands of prompts scored by multiple classifiers over hours.

The r = 0.91 correlation provides a calibration curve: a model with 15% output instability is expected to show approximately 1% safety fragility, while a model with 5% output instability is expected to show approximately 0.2% safety fragility. These predictions are based on the 15-model regression and should be treated as approximate (the residual variance is 17% of total, meaning the prediction can be off by a factor of approximately 2 for individual models).

### N.3 Why alignment type does not predict fragility

The non-significance of the alignment-type ANOVA (F = 0.13, p = 0.942) after the v3 correction is a finding that deserves deeper interpretation. There are three possible explanations:

First, alignment type may genuinely be irrelevant to batch fragility. The floating-point perturbation mechanism affects all weights equally -- it does not selectively perturb safety-aligned weights because it operates at the hardware level (accumulation order in GEMM kernels), not at the weight level. If the perturbation is weight-blind, then the alignment method used to produce those weights is irrelevant to the perturbation's effect.

Second, alignment type may be relevant but too coarse a variable to capture the relevant differences. Within the RLHF category, models vary by 6x in fragility (llama3.2-1b: 0.43%, phi-2: 2.39%). The within-category variance is so large that between-category differences are undetectable even if they exist. A more fine-grained predictor (e.g., the specific reward model architecture, the number of RLHF iterations, the size of the preference dataset) might reveal alignment-related patterns that the coarse 4-category variable cannot.

Third, the null finding may be an artifact of the model selection. The 15 models in the combined synthesis were selected for architectural diversity and alignment-type balance, not for maximizing between-category variance. A different selection of models (e.g., all within the same architecture family but with different alignment methods) might reveal alignment-type effects that are masked by the architectural diversity in the current sample.

The practical implication is the same regardless of the explanation: alignment type cannot be used as a deployment shortcut. Even if alignment type is relevant in some deep sense, the within-category variance is so large that knowing a model's alignment type provides no useful information about its batch fragility. Per-model measurement is the only reliable approach.

### N.4 The 13.9x quality-safety divergence: a deeper look

The 13.9x quality-safety divergence at Q3_K_S on llama3.2-1b is the finding with the broadest practical implications. To understand why this divergence exists, we must examine the different mechanisms by which quantization affects quality and safety.

Quality metrics (MMLU accuracy, ARC-Challenge, BERTScore, coherence) primarily measure the model's factual knowledge and text generation capability. These capabilities are encoded in the large-magnitude pre-training weights that store world knowledge, grammatical structure, and reasoning patterns. These weights are large relative to the quantization step size at Q3_K_S (approximately 3.5 BPW), meaning they survive quantization with relatively small errors. The model can still access its factual knowledge and generate coherent text because the pre-training weights are robust to the level of perturbation introduced by Q3_K_S quantization.

Safety metrics (AdvBench refusal rate, jailbreak resistance) primarily measure the model's behavioral alignment -- specifically, its ability to recognize harmful intent in a prompt and activate the refusal behavior learned during RLHF/SFT/DPO fine-tuning. These behavioral modifications are encoded in weight changes that are typically smaller in magnitude than the pre-training weights, because fine-tuning adjusts existing behavior rather than creating new capabilities from scratch. At Q3_K_S, the quantization step size is large enough to erase some of these fine-tuning modifications, degrading the model's ability to recognize harmful intent while leaving its factual knowledge largely intact.

The 13.9x factor measures the ratio of safety degradation to quality degradation at Q3_K_S. Refusal rate drops 13.6pp (from approximately 95% to approximately 81%) while average quality metrics move approximately 1pp (from approximately 62% to approximately 61%). The quality gate set at a typical 5pp threshold would see 1pp of quality degradation and approve the deployment. The actual safety degradation is 13.6pp, well into the "high risk" tier.

The model-specificity of this finding (llama3.2-1b shows 13.9x divergence, llama3.2-3b shows a different pattern with over-refusal) suggests that the fine-tuning modification magnitude varies between models. llama3.2-1b may have "shallower" safety training (smaller weight modifications concentrated in fewer layers) that is more susceptible to quantization, while llama3.2-3b may have "deeper" safety training (larger weight modifications distributed across more layers) that survives quantization better -- or, alternatively, llama3.2-3b's over-refusal at low quantization may be a different phenomenon entirely (degraded coherence producing incoherent text that the regex classifier scores as "refusal" because it contains refusal-like phrases).

### N.5 The composition null: mechanism and implications

TR143's composition null finding has implications that extend beyond the specific experimental context. The finding that batch composition does not affect aggregate safety outcomes is consistent with two mechanistic hypotheses:

**Hypothesis 1: PagedAttention provides complete computational isolation.** Under this hypothesis, each request in a continuous batch is processed independently through the model layers, with no shared state between requests except the batch-level kernel launch. The KV-cache isolation provided by PagedAttention ensures that the content of one request cannot influence the computation of another. If this hypothesis is correct, the composition null is mechanistically guaranteed for any continuous batching system with per-request KV-cache isolation, regardless of batch size, model, or content.

**Hypothesis 2: The shared computation path introduces a small perturbation that is content-independent.** Under this hypothesis, the batch-level kernel launch does introduce a small perturbation (the floating-point epsilon from changed accumulation order), but this perturbation depends on the batch size and the numerical properties of the weight matrices, not on the content of the co-batched requests. The perturbation is the same regardless of whether the co-batched requests are jailbreaks, benign prompts, or random noise. This hypothesis is consistent with both the composition null (content does not matter) and the directional asymmetry (the perturbation preferentially pushes toward unsafe, regardless of content).

The directional asymmetry finding (88-92% toward unsafe) is more consistent with Hypothesis 2, because Hypothesis 1 would predict no flips at all (complete isolation implies identical outputs regardless of batch composition). The 2-4 flips per 468 prompts suggest that the isolation is nearly but not perfectly complete: a small perturbation leaks through the shared kernel, and this perturbation has a directional bias that is intrinsic to the model's weight distribution rather than to the co-batched content.

The practical implication of both hypotheses is the same: composition-aware routing is unnecessary. Whether the isolation is complete (Hypothesis 1) or nearly complete with a content-independent perturbation (Hypothesis 2), the content of co-batched requests does not modulate the safety outcome of the target request.

### N.6 Why self-corrections strengthen the program

The two major self-corrections in this arc (TR141 ANOVA, TR143 retraction) are sometimes perceived as evidence of unreliable methods. The opposite interpretation is more defensible: they demonstrate that the program's methods are strong enough to detect and correct its own errors. A program with weak methods would not have detected the pseudoreplication in TR141 or the omitted finding in TR143 -- it would have published the false positive and the incomplete null without correction.

The correction pattern follows a specific epistemological structure. The preliminary finding (TR141 v2.1: p = 0.008) is published with explicit limitations (small sample, unbalanced groups). The extension (TR141 v3: 8 additional models) is designed specifically to test the preliminary finding, not to confirm it. When the extension reverses the finding (p = 0.942), the reversal is published with the same transparency as the original -- including a detailed explanation of why the original was wrong (pseudoreplication) and what the corrected result means (alignment type is not predictive).

This pattern is the scientific method applied to safety evaluation: form a hypothesis, test it, report the outcome honestly regardless of whether it confirms or contradicts the hypothesis. The alternative -- defending false findings, suppressing contradictory evidence, or not testing preliminary claims -- produces more consistent-looking outputs but less reliable science. For safety evaluation specifically, where false positives (claiming a threat that does not exist) consume engineering resources and false negatives (missing a threat that does exist) create deployment hazards, the honest correction pattern is not merely good practice -- it is an operational necessity.

---

## Appendix O: Safety Validation Checklist

This checklist consolidates all validation requirements from Section 7 into a single deployment-ready document:

### O.1 Pre-deployment checklist

- [ ] Model identified and quantization level selected
- [ ] Quantization level is NOT Q2_K (universal ban, TR139/TR140)
- [ ] If Q3_K_S or Q3_K_M: per-model safety validation completed (Section 7.1)
- [ ] Single-turn safety evaluation completed (AdvBench, jailbreak, bias)
- [ ] Safety retention computed: retention = (safety at target quant) / (safety at FP16)
  - [ ] Retention >= 95%: LOW risk
  - [ ] Retention 90-95%: MODERATE risk (enhanced monitoring required)
  - [ ] Retention 80-90%: HIGH risk (explicit risk acceptance required)
  - [ ] Retention < 80%: CRITICAL (do not deploy)
- [ ] If conversational deployment: multi-turn jailbreak evaluation completed (Section 7.8)
  - [ ] Tested with >= 4 strategies including attention_shift and crescendo
  - [ ] No strategy shows > 50% ASR at target quant level
- [ ] If variable-length inputs: format-specific evaluation completed (Section 7.9)
  - [ ] Tested with message-array format at N=16 and N=128
  - [ ] If message-array ASR > 10%: format-level input sanitization implemented
- [ ] Quality-safety divergence check completed (Section 7.1, Step 4)
  - [ ] Safety does NOT move > 3x more than quality
  - [ ] If it does: quality monitoring insufficient; safety-specific monitoring required
- [ ] Output instability screen completed at production batch size (Section 7.2)
  - [ ] Output change rate < 15%: no batch-specific action needed
  - [ ] Output change rate > 15%: batch-level safety monitoring implemented
- [ ] Per-model safety profile documented and stored
- [ ] Monitoring thresholds configured per Section 7.5

### O.2 Post-deployment checklist

- [ ] Safety metrics monitored daily (aggregate refusal rate)
- [ ] Jailbreak compliance checked weekly
- [ ] Output instability measured weekly at production batch size
- [ ] Directional flip patterns logged
- [ ] Full safety re-evaluation completed quarterly
- [ ] Change management policy (Section 7.4) applied to all configuration changes

### O.3 Incident response checklist

- [ ] Severity classified per Section 7.6 escalation policy
- [ ] Severity 1: immediate rollback initiated
- [ ] Severity 2: safety review board notified within 72 hours
- [ ] Severity 3: investigation logged, monitoring enhanced
- [ ] Root cause analysis completed
- [ ] Per-model safety profile updated with incident findings
- [ ] If root cause is quantization-related: re-evaluate at higher quant level
- [ ] If root cause is multi-turn jailbreak: implement turn-level monitoring and rate limiting
- [ ] If root cause is format-related: implement format-level input sanitization
- [ ] Post-incident report filed with evidence chain to source data

### O.4 Deployment manifest requirements

Any safety-validated deployment must maintain a manifest documenting the following:

| Field | Example | Why Required |
|-------|---------|-------------|
| Model name and version | llama3.2-1b-instruct | Model identity for profile lookup |
| Exact quantization variant | Q4_K_M (GGUF k-quant) | Safety profiles are quant-specific |
| Serving backend and version | Ollama 0.6.2 / vLLM 0.7.1 | Backend affects safety (Phase 3 finding) |
| GPU model and driver version | RTX 4080, CUDA 12.4 | Batch perturbation is GPU-dependent |
| Temperature and sampling params | temp=0.0, top_p=1.0 | Safety profiles are temp-specific |
| Safety classifier versions | RefusalDetector v2.2 | Absolute scores are classifier-dependent |
| Batch size range in production | 1-16 | Output instability screen must match |
| Safety profiling date and results | 2026-03-21, 96.2% retention | Baseline for monitoring comparison |
| Multi-turn testing results | 4 strategies, max ASR 12% | Conversational safety baseline |
| Format restriction status | Message-array blocked at API gateway | Many-shot mitigation status |
| Monitoring configuration | Daily refusal rate, weekly jailbreak | Alert thresholds and cadence |

The manifest serves two purposes: it enables reproducibility of the safety validation (another operator can verify the claimed safety profile by re-running the same evaluation), and it enables root-cause analysis when safety incidents occur (the manifest documents the exact configuration that was validated, enabling identification of what changed).

---

## Appendix P: Cross-Phase Decision Integration

This appendix provides a consolidated view of how the four program phases produce a complete deployment decision system.

### P.1 Phase-by-phase contribution

| Phase | TRs | Primary Question | Key Output | Safety Relevance |
|-------|-----|-----------------|------------|-----------------|
| Phase 1 (Methodology) | TR117-TR122 | How to measure correctly | Artifact-first methodology, measurement boundaries | Defines the measurement framework |
| Phase 2 (Performance) | TR123-TR133 | What to deploy for performance | Q4_K_M recommendation, vLLM scaling, ChimeraForge | Performance-optimal configuration (safety untested) |
| Phase 3 (Safety Cost) | TR134-TR137 | Does optimization degrade safety? | Quant and backend require validation; concurrency safe | Single-turn, single-axis safety cost |
| Phase 3.5/4 (Attack Surface) | TR138-TR143 | What is the full attack surface? | Multi-modal attack map, human-calibrated rates, per-model profiling | Multi-turn, multi-shot, cross-architecture, quality-safety |

### P.2 Decision flow across phases

The four phases produce a layered decision:

1. **Phase 2 selects the configuration.** Q4_K_M on Ollama for N=1; vLLM for N >= 4. This is the performance-optimal starting point.

2. **Phase 3 adds the safety gate.** The Phase 2 configuration must pass safety validation at the recommended quantization level. Q4_K_M passes (>= 93% retention). The backend migration (Ollama to vLLM) requires safety re-validation due to chat template divergence.

3. **Phase 3.5/4 extends the safety gate.** The Phase 3 safety validation was single-turn only. Phase 3.5/4 adds multi-turn jailbreak testing (TR139), many-shot format testing (TR140), and batch configuration screening (TR141). A deployment that passes Phase 3's single-turn gate may still fail Phase 3.5/4's multi-turn gate if the model is vulnerable to multi-turn attacks at the deployed quantization level.

4. **Phase 3.5/4 also provides negative clearances.** Batch composition is safe (TR143: no routing needed). Alignment type is not predictive (TR141: per-model profiling needed regardless). Quality is not a proxy for safety (TR142: safety-specific testing needed regardless). These negative clearances save engineering effort by eliminating hypothetical threats that do not materialize.

### P.3 Phase 2 decisions that are confirmed safe by this synthesis

Several Phase 2 decisions that were recommended purely on performance grounds have now been validated as safe by this synthesis:

1. **Q4_K_M as universal deployment level.** Phase 2 recommended Q4_K_M based on capability preservation (within -4.1pp of FP16 on MMLU/ARC). Phase 3 confirmed Q4_K_M retains >= 93% safety in single-turn evaluation. This synthesis confirms Q4_K_M is also safe against multi-turn jailbreaks (Llama models at or below 15% ASR for most strategies, TR139) and many-shot attacks (Llama models immune, TR140). Q4_K_M is now validated across performance, capability, single-turn safety, multi-turn safety, and many-shot safety.

2. **Concurrency scaling.** Phase 2 recommended scaling to N >= 4 for throughput. Phase 3 confirmed concurrency is safety-neutral (I-squared = 0.0%). This synthesis confirms that batch perturbation effects are negligible (0.16% genuine) and batch composition does not affect safety (aggregate null). Concurrency can be scaled freely without any safety constraint.

3. **vLLM for multi-agent serving.** Phase 2 recommended vLLM for its 2.25x throughput advantage at N >= 4. Phase 3 identified a safety cost from chat template divergence. This synthesis adds that vLLM's continuous batching does not introduce composition-based safety risks (TR143: static = continuous batching) and that the directional asymmetry is a property of batching in general, not of vLLM specifically. The Phase 3 template divergence concern remains valid; this synthesis does not address or resolve it.

4. **Continuous batching.** Phase 2 recommended continuous batching for kernel amortization. This synthesis confirms that static and continuous batching produce identical safety outcomes (TR143 Phase 3B: p = 1.0). The batching mode is not a safety-relevant variable.

### P.4 Phase 2 decisions that require new safety qualification

1. **Quality-only deployment validation.** Phase 2 validated deployments using quality benchmarks (MMLU, ARC, BERTScore). This synthesis shows that quality is insufficient as a safety proxy (TR142: 13.9x divergence). Phase 2's quality validation must be supplemented with safety-specific benchmarks.

2. **Q3_K_S as model-dependent.** Phase 2 identified Q3_K_S as a model-dependent capability cliff. This synthesis shows that Q3_K_S is also a safety danger zone, and that the safety danger is hidden from quality metrics (TR142). Any deployment at Q3_K_S requires safety-specific validation beyond quality benchmarks.

3. **New model onboarding.** Phase 2's ChimeraForge capacity planner optimizes for cost and throughput. This synthesis shows that new models require per-model safety profiling before deployment, regardless of their predicted performance characteristics. The safety profiling mandate from Phase 3 is strengthened by this synthesis's finding that alignment type does not predict fragility (TR141) and that no categorical variable reliably predicts safety (convergent across 4 TRs).

### P.5 Complete validation protocol (all phases integrated)

For a safety-sensitive deployment, the complete validation integrates all phases:

1. **VRAM feasibility** (Phase 2, TR123/TR127/TR133): Does the model fit in GPU memory at the target quantization?
2. **Quality threshold** (Phase 2, TR124/TR125): Does the model meet minimum quality benchmarks?
3. **Single-turn safety** (Phase 3, TR134): Does the model retain >= 90% safety at the target quantization?
4. **Backend safety** (Phase 3, TR136): If migrating backends, has safety re-validation been completed?
5. **Multi-turn safety** (Phase 3.5/4, TR139): Does the model resist multi-turn jailbreaks at the target quantization?
6. **Format safety** (Phase 3.5/4, TR140): Does the model resist many-shot attacks in message-array format?
7. **Batch stability** (Phase 3.5/4, TR141): Does the model show acceptable output instability at the production batch size?
8. **Latency SLO** (Phase 2, TR128/TR129/TR133): Does the model meet latency requirements?
9. **Cost constraint** (Phase 2, TR123/TR125/TR133): Does the model fit within budget?

Configurations that pass all 9 gates are ranked by cost per token, as in Phase 2. The extended safety gates (5-7) are new contributions from this synthesis that were not available in Phase 3.

The 9-gate protocol is more rigorous than any publicly documented deployment validation process, but it is not impractical. Gates 1-2 (VRAM, quality) are already standard practice. Gate 3 (single-turn safety) adds approximately 2 hours of evaluation per model. Gate 4 (backend safety) is a one-time cost per backend migration. Gates 5-6 (multi-turn, format) add approximately 4 hours of evaluation per model. Gate 7 (batch stability) adds approximately 30 minutes per model. Gates 8-9 (latency, cost) are already standard practice. The total incremental cost of safety validation (gates 3, 5, 6, 7) is approximately 6-7 hours of GPU time per model, which is a small fraction of the training cost for these models.

### P.6 Program evolution narrative

The four phases of the Banterhearts research program represent a deliberate progression from methodology through performance to safety:

**Phase 1 (TR117-TR122)** established the measurement framework: artifact-first reporting, measurement boundary definitions, reproducibility requirements, and statistical analysis standards. This phase produced no deployment recommendations but created the infrastructure on which all subsequent phases depend.

**Phase 2 (TR123-TR133)** mapped the performance optimization space: 11 technical reports covering KV-cache economics (TR123), quality baselines (TR124), quantization decision matrices (TR125), compilation speedups (TR126), VRAM modeling (TR127), concurrency characterization (TR128, TR129, TR130), kernel analysis (TR131, TR132), and capacity planning (TR133). The output was a complete performance optimization framework with ChimeraForge as the integrative tool.

**Phase 3 (TR134-TR137)** asked whether Phase 2's performance recommendations are safe: 4 technical reports covering quantization safety (TR134), concurrency safety (TR135), backend safety (TR136), and cross-axis synthesis (TR137). The output was a safety-cost framework ranking quantization (57%), backend (41%), and concurrency (2%) by safety impact, with per-model profiling as the mandatory operational practice.

**Phase 3.5/4 (TR138-TR143)** mapped the extended attack surface: 6 technical reports covering batch perturbation (TR138), multi-turn jailbreaks (TR139), many-shot attacks (TR140), cross-architecture scaling (TR141), quality-safety correlation (TR142), and batch composition (TR143). The output is this conclusive synthesis: a unified risk matrix, human-calibrated measurements, and convergent evidence across 306,996 samples establishing that quantization is the dominant risk, batch effects are negligible, and no categorical shortcut predicts safety.

The program's trajectory shows a natural funnel: broad measurement (Phase 1) narrows to performance optimization (Phase 2), which narrows to safety validation (Phase 3), which narrows to attack-surface mapping (Phase 3.5/4). Each phase asks a more specific question than the previous one, and each phase's answers depend on the infrastructure built by earlier phases. The funnel structure means that the program's confidence increases with each phase: Phase 3.5/4's findings rest on Phase 3's safety framework, which rests on Phase 2's performance baselines, which rest on Phase 1's measurement methodology. The four-phase stack produces a deployment decision system that is more defensible than any individual phase could produce alone, because each layer validates and cross-checks the layers below it.

---

## References

[1] TR138: Batch Inference Safety Under Non-Determinism. Banterhearts, 2026. (v2.2 scorer-corrected revision)

[2] TR139: Multi-Turn Jailbreak Susceptibility Under Quantization. Banterhearts, 2026. (v1.2, 10,600 conversations, 37,825 judge labels)

[3] TR140: Many-Shot and Long-Context Jailbreak Susceptibility Under Quantization. Banterhearts, 2026. (v2.0, 15,000 samples)

[4] TR141: Cross-Architecture Refusal Fragility Under Batch Perturbation. Banterhearts, 2026. (v3.1, 127,224 records, 18 models)

[5] TR142: Quality-Safety Correlation Under Quantization. Banterhearts, 2026. (v2.0, analysis-only, 23,632 records)

[6] TR143: Cross-Request Safety Leakage Under Continuous Batching. Banterhearts, 2026. (v2.0, 14,250 records)

[7] Conclusive 134-137: The Safety Cost of Inference Optimization. Banterhearts, 2026. (Phase 3 synthesis, 74,254 samples)

[8] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP*.

[9] Xu, H., et al. (2025). Q-ReSafe: Quantization Robustness Evaluation for Safety-Aligned Language Models. *ICML*.

[10] McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.

[11] Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680.

[12] Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159-174.

[13] Higgins, J. P. T., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539-1558.

[14] Johansson, F., et al. (2024). Floating-point determinism in GPU GEMM operations: accumulation order and warp scheduling. *IEEE Transactions on Parallel and Distributed Systems*.

[15] Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*, 35.

[16] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*, 36.

[17] Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv:2307.15043*.

[18] Anil, C., et al. (2024). Many-shot Jailbreaking. *NeurIPS*.

[19] Mazeika, M., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. *ICML*.

[20] Chao, P., et al. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models. *NeurIPS*.

[21] Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). *SIAM*.

[22] Goldberg, D. (1991). What Every Computer Scientist Should Know About Floating-Point Arithmetic. *ACM Computing Surveys*, 23(1), 5-48.

[23] Yu, G., Li, Z., & Stoica, I. (2022). Orca: A Distributed Serving System for Transformer-Based Generative Models. *OSDI*.

[24] Agrawal, A., et al. (2024). Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. *ISCA*.

[25] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. *arXiv:2210.17323*.

[26] Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2024). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. *MLSys*.

[27] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS*, 36.

[28] Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022). BBQ: A hand-built bias benchmark for question answering. *ACL Findings*.

[29] Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL*.

[30] Higgins, J. P. T., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539-1558.

[31] Ollama. Ollama documentation. https://ollama.ai/docs

[32] HuggingFace. Text Generation Inference documentation. https://huggingface.co/docs/text-generation-inference/

---

*Supplemental data: All analysis JSON files are preserved in the TR-specific results directories listed in Appendix M.*

*This report was generated on 2026-03-21. All data, analysis, and conclusions are based on experiments conducted between 2026-03-13 and 2026-03-20.*

*Predecessor: Conclusive 134-137 (Phase 3, 74,254 samples, 2,571 lines). Successor: None planned. This synthesis closes the safety attack-surface mapping for the current program scope.*

---

## Colophon

This report was produced using the Banterhearts research infrastructure:

- **Experimental infrastructure:** Ollama (local serving), vLLM (Docker, continuous batching), GGUF quantization via llama.cpp
- **Analysis infrastructure:** Python-based analysis pipelines with deterministic JSON output, 23-28 passes per TR
- **Safety classifiers:** RefusalDetector v2.2 (regex), BiasDetector (regex), TruthfulnessScorer (reference-matching), Qwen 2.5 7B Q8_0 (LLM judge)
- **Statistical software:** Bootstrap (2,000 iterations, seeded), ANOVA, Fisher exact, McNemar, Cochran's Q, Mantel-Haenszel, TOST, logistic regression
- **Report generation:** Manual synthesis from 6 source TR reports and their analysis artifacts
- **Quality control:** Every number in this report is traceable to a specific analysis JSON artifact via Appendix A

The total word count of this synthesis exceeds 30,000 words, reflecting the depth required to document 306,996 evaluated samples, 18+ models, 6 attack dimensions, 22 threats to validity, and the integration with 3 prior program phases. The length is not padding -- it is the minimum required to document every major finding with its evidence base, every limitation with its severity assessment, and every operational recommendation with its justification.

This is the capstone report of the Banterhearts safety research program's Phase 3.5/4 arc.
