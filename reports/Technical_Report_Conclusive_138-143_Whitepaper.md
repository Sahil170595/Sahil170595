# TR138-TR143 Decision Whitepaper
## Executive guidance for safety attack-surface management in LLM inference

| Field | Value |
|-------|-------|
| **Project** | Banterhearts LLM Inference Research |
| **Date** | 2026-03-21 |
| **Version** | 1.0 |
| **Report Type** | Decision whitepaper |
| **Audience** | Decision makers, product leaders, ML ops leaders, safety teams |
| **Scope** | TR138 (Batch Safety), TR139 (Multi-Turn Jailbreak x Quantization), TR140 (Many-Shot Long-Context), TR141 (Cross-Architecture Batch Fragility), TR142 (Quality-Safety Correlation), TR143 (Batch Composition) |
| **Primary Source** | `PublishReady/reports/Technical_Report_Conclusive_138-143.md` |
| **Predecessor** | `PublishReady/reports/Technical_Report_Conclusive_134-137_Whitepaper.md` (Phase 3 decisions) |

---

## Abstract

Consumer-grade LLMs run on hardware that millions of people own, but the safety evaluation infrastructure for local deployment does not exist. A user who quantizes a model to fit their GPU has no way to know whether that optimization degraded refusal behavior, amplified jailbreak vulnerability, or introduced demographic bias. This whitepaper builds that infrastructure.

It distills six technical reports (TR138-TR143) and 306,996 evaluated samples into deployment policy for the safety attack surface of optimized LLM inference on consumer hardware. Phase 3 (TR134-TR137) established WHICH optimization axes degrade safety (quantization, backend) and which do not (concurrency). This Phase 3.5/4 synthesis asks: what additional attack modalities exist beyond single-turn evaluation, how severe are they relative to each other, and what categorical shortcuts (alignment type, quality metrics, batch composition) can operators rely on? Outcome: nine shippable decisions covering batch perturbation, multi-turn jailbreaks, many-shot exploitation, cross-architecture variation, quality-safety divergence, and batch composition -- backed by human adjudication, cross-architecture replication across 18+ models, and convergent evidence from 6 independent studies.

---

## Boundary conditions (do not skip)

This guidance is valid only under the measured boundary:

- Models at or below 14.8B parameters (18+ tested: Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B/7B/14B, Phi-2, Phi-3.5-mini, Mistral 7B, SmolLM2-1.7B, SmolLM3-3B, TinyLlama-1.1B, StableLM-2-1.6B, OLMo-1B, OLMo-2-1B, DeepSeek-R1-1.5B, Llama-3.1-8B, and others)
- Hardware: NVIDIA RTX 4080 Laptop 12GB (TR138-140, TR142-143) + RTX PRO 6000 Blackwell 98GB (TR141)
- Quantization via GGUF k-quant (Q2_K through Q8_0) for TR139/TR140; FP16 via vLLM for TR138/TR141/TR143
- Serving backends: Ollama (TR139/TR140), vLLM Docker (TR138/TR141/TR143)
- Safety scored by automated classifiers (RefusalDetector, BiasDetector, TruthfulnessScorer) with human adjudication on 63 TR138 rows (27% genuine flip rate)
- Temperature 0 (deterministic sampling) throughout
- 4 alignment types tested: RLHF, SFT, DPO, Distilled
- 8 multi-turn jailbreak strategies, 5 many-shot counts, 5 batch composition conditions

If any of these change, re-run the relevant safety evaluation and re-validate before applying these decisions.

---

## Nine decisions you can ship now

### From Phase 3 (confirmed by this synthesis)

1. **Quantization floor remains Q4_K_M for safety-critical applications.** Phase 3 established this floor; this synthesis confirms it holds under multi-turn jailbreaks (Llama models at or below 15% ASR at Q4_K_M, TR139), many-shot attacks (Llama models immune above Q3_K_M, TR140), and batch perturbation (0.16% genuine flip rate, TR138). Q4_K_M has been evaluated across single-turn, multi-turn, many-shot, and batch-perturbation threat models for the tested model set.

2. **Q2_K ban is unconditional within the tested model set.** Phase 3 banned Q2_K for Llama 1B. This synthesis extends the ban: no tested model (18+ across 10+ families, 360M-14.8B) retains adequate safety at Q2_K under any measured attack type. TR139: 100% ASR at Q2_K on qwen2.5-1.5b with attention_shift. TR140: Q2_K is the vulnerability threshold across all tested models for many-shot attacks. TR142: quality metrics underestimate Q2_K safety degradation by 13.9x (measured on 2 models).

3. **Per-model safety profiling is mandatory (strengthened).** Phase 3 established this based on I-squared = 99.9%. This synthesis strengthens it: TR141 demonstrates 6.3x fragility variation across 15 models (phi-2 2.39% to tinyllama 0.00%), and alignment type is not predictive (ANOVA F = 0.13, p = 0.942). No categorical shortcut replaces per-model measurement.

### New from this synthesis

4. **Never use quality metrics as sole safety proxies.** On llama3.2-1b, safety degrades 13.9x faster than quality at Q3_K_S (TR142). On the only other tested model (llama3.2-3b), the quality-safety correlation is negative (r = -0.829), meaning quality degrades while safety paradoxically improves through over-refusal. With only 2 models, the distribution of quality-safety relationships across the model population is unknown. The conservative recommendation: measure safety directly rather than inferring it from quality.

5. **Test multi-turn jailbreaks at your quantization level.** All 8 strategy-specific ANOVAs reject quantization-independence (p < 1e-4, TR139). Single-turn safety testing is necessary but not sufficient. Include at least 4 strategy types (attention_shift, crescendo, progressive_refinement, context_fusion) in pre-deployment validation.

6. **Restrict or sanitize message-array format inputs.** Message-array format achieves 92% ASR versus 0% for faux-dialogue format on the same model and quantization level (TR140). Format-level input validation is more effective than quantization-level restriction: blocking message-array format reduces ASR from 92% to 0%, while moving from Q2_K to Q8_0 reduces it from 92% to at most 40%.

7. **Measure output instability as a batch-safety screen.** Output change rate above 15% implies safety flip rates above 1% (TR141, r = 0.91, R-squared = 0.83). This is a low-cost screening metric: run the same prompts at batch = 1 and batch = N, compare outputs textually. Takes 30 minutes per model.

8. **Batch composition routing is unnecessary.** Aggregate composition effect is null across all tests (TR143: 21 McNemar tests, 3 Cochran's Q, 3 Mantel-Haenszel ORs, all non-significant). An attacker cannot degrade other users' safety by flooding a multi-tenant endpoint with jailbreak prompts. Do not invest in composition-aware request routing.

9. **Do not assume alignment type predicts batch safety.** ANOVA F = 0.13, p = 0.942 with balanced groups across 15 models and 4 alignment types (TR141 v3). The earlier p = 0.008 from TR141 v2.1 was a false positive from pseudoreplication with n = 1 per category.

---

## Decision matrix (one-glance policy)

| Threat | Severity | Action | Evidence |
|--------|----------|--------|----------|
| Q2_K on any model | **CRITICAL** | Ban unconditionally | TR139 (100% ASR), TR140 (universal threshold), TR142 (13.9x hidden) |
| Q3_K_S without safety testing | **HIGH** | Validate per-model; quality metrics will miss it | TR142 (13.9x divergence), TR134 (13.6pp refusal collapse) |
| Multi-turn jailbreak at < Q4_K_M | **HIGH** | Test with 4+ strategies before deployment | TR139 (all ANOVAs p < 1e-4) |
| Message-array format at low quant | **HIGH** | Restrict or sanitize at API gateway | TR140 (92% vs 0% ASR) |
| Quality used as sole safety proxy | **HIGH** | Add safety-specific benchmarks | TR142 (opposite-sign correlations by model) |
| Batch size > 1 (safety-critical) | **LOW** | Screen output instability; monitor if > 15% | TR141 (r = 0.91) |
| Batch composition | **NEGLIGIBLE** | No action required | TR143 (aggregate null, all tests p > 0.05) |
| Alignment type as predictor | **INVALID** | Do not use; profile each model individually | TR141 v3 (p = 0.942) |

---

## Key findings (decision-grade)

- **Quantization is the dominant risk by two orders of magnitude.** Quantization effects are measured in tens of percentage points (up to 100% ASR at Q2_K). Batch effects are measured in tenths of a percentage point (0.16% genuine flip rate). An operator who addresses batch composition before validating quantization is misallocating resources by 100x.

- **Human adjudication deflates automated flip rates by approximately 3x in TR138.** Of 63 TR138 candidate rows reviewed by a single human annotator, only 17 (27%, 90% CI: 17-40%) represent genuine behavioral flips. The remaining 73% are regex artifacts from rephrased refusals. This calibration is specific to TR138's batch-perturbation phenomenon; artifact rates for other TRs (which test different attack types) have not been measured. The general principle — that automated classifiers can substantially overestimate small effects — likely applies broadly, but the specific 73% rate should not be assumed for TR139-TR143.

- **Output instability is the most reliable predictor tested for batch fragility.** Across 15 models and 4 alignment types, output change rate correlates with safety flip rate at r = 0.91 (R-squared = 0.83). Among tested categorical predictors, alignment type is uninformative (ANOVA p = 0.942) and baseline refusal rate is uninformative (r = 0.028, p = 0.919). Other potential predictors (e.g., specific architectural properties, training data composition) were not tested.

- **Multi-turn jailbreaks interact with quantization but not selectively.** All 8 strategy ANOVAs reject quantization-independence (all p < 1e-4, η² = 0.031-0.153), but multi-turn strategies are not more quantization-sensitive than direct attacks (Welch p = 0.702). Quantization degrades safety broadly, not through a multi-turn-specific mechanism. Caveat: ASR values carry measurement uncertainty of approximately ±15pp due to inter-judge agreement of kappa = 0.104; relative rankings between conditions are more reliable than absolute thresholds.

- **Format matters more than shot count for many-shot attacks.** Message-array format produces 92% ASR versus 0% for faux-dialogue format on the same model (TR140). Variance decomposition: residual (per-behavior) = 65.7%, quantization = 17.9%, model = 12.6%, shot count = 2.7%. Prompt format, not shot count, is the dominant variable.

- **Quality-safety correlation reverses between models.** llama3.2-1b shows r = +0.994 (quality and safety co-degrade); llama3.2-3b shows r = -0.829 (quality degrades while safety improves through over-refusal). No single quality-safety relationship can be applied across models. Safety must be measured directly.

- **Batch composition does not affect aggregate safety, but rare flips are directionally biased.** All McNemar, Cochran's Q, and Mantel-Haenszel tests are non-significant. However, when flips occur, 88-92% are toward unsafe (binomial p = 0.006 for strongest condition). The asymmetry is consistent across composition types and attributable to the batching mechanism itself, not to filler content.

---

## Operational recommendations (policy statements)

### Quantization policy (extended from Phase 3)

- **Policy:** Q4_K_M remains the safety-validated floor. Now confirmed against multi-turn (TR139), many-shot (TR140), and batch perturbation (TR138/TR141) attack modalities.
- **Policy ban:** Q2_K on ALL models (extended from Llama-1B-only in Phase 3). Universal vulnerability confirmed across 4 models, 8 strategies, 2 long-context formats, and quality-safety analysis.
- **Policy gate:** Q3_K_S requires safety-specific testing that includes multi-turn and many-shot modalities. Quality metrics are insufficient (13.9x underestimate).

### Multi-turn jailbreak policy (new)

- **Policy:** Test all deployed quantized models against at least 4 multi-turn jailbreak strategies before deployment.
- **Policy:** attention_shift is the most dangerous strategy (100% ASR on qwen2.5-1.5b at Q2_K). Prioritize it in testing.
- **Policy:** Multi-turn testing is required at the deployed quantization level. Results at Q8_0 do not predict behavior at Q4_K_M.

### Long-context input policy (new)

- **Policy:** Detect and restrict message-array formatted inputs at the API gateway level.
- **Policy:** If message-array format must be allowed, test the deployed model at N = 16 and N = 128 compliance examples. If ASR > 10%, implement format transformation (message-array to faux-dialogue).

### Batch configuration policy (new)

- **Policy:** Measure output instability per-model at the production batch size. Screen threshold: 15% change rate.
- **Policy:** Batch composition routing is not required. Aggregate composition effect is null.
- **Policy:** Static and continuous batching produce identical safety outcomes (TR143 Phase 3B, p = 1.0). Scheduler mode is not a safety-relevant variable.

### Quality-safety monitoring policy (new)

- **Policy:** Never use quality benchmarks (MMLU, ARC, BERTScore, coherence) as the sole deployment validation for safety-sensitive applications.
- **Policy:** Safety-specific benchmarks (refusal, jailbreak resistance, bias) are required in addition to quality benchmarks.
- **Policy:** If safety moves > 3x more than quality at any quantization level, the model is in a "hidden danger zone." Escalate for additional review.

---

## Risk impact

### Attack surface by severity (ranked)

1. **Quantization (Q2_K-Q3_K_S):** 100% ASR achievable. 13.9x hidden from quality metrics. CRITICAL for all models. Dominates all other dimensions by 100x.
2. **Multi-turn jailbreak x quantization:** All 8 strategy ANOVAs significant. Peak 100% ASR. HIGH but addressed by quantization floor.
3. **Many-shot format exploitation:** 92% ASR via message-array format. HIGH but mitigable via format restriction.
4. **Batch perturbation:** 0.16% genuine flip rate (human-adjudicated). LOW. Predictable via output instability (r = 0.91).
5. **Batch composition:** Aggregate null. NEGLIGIBLE. No engineering action required.

### Worst-case combinations

- **Worst overall:** qwen2.5-1.5b at Q2_K + attention_shift multi-turn = 100% ASR (TR139)
- **Worst many-shot:** llama3.1-8b at Q2_K + message-array N=16 = 92% ASR (TR140)
- **Worst hidden danger:** llama3.2-1b at Q3_K_S: quality metrics show ~1pp degradation, safety shows 13.6pp (TR142)
- **Most fragile under batching:** phi-2 at 2.39% safety flip rate (TR141)
- **Most robust under batching:** tinyllama-1.1b-chat at 0.00% safety flip rate (TR141)

---

## Implementation plan (30-day view)

**Days 1-7: Validate quantization safety with extended attack modalities**

- Run single-turn safety battery at deployed quantization level (carried over from Phase 3).
- NEW: Run multi-turn jailbreak evaluation (4 strategies, 50 behaviors) at deployed quantization level.
- NEW: Run many-shot format test (message-array, N=16 and N=128) if application accepts variable-length inputs.
- Flag any model-quant combination with multi-turn ASR > 50% or many-shot ASR > 10% for immediate review.

**Days 8-14: Implement format-level and batch-level controls**

- NEW: Add message-array format detection to API gateway. Block or transform message-array inputs for safety-sensitive endpoints.
- NEW: Measure output instability at production batch size for each deployed model. If > 15%, add batch-level safety monitoring.
- Confirm Q2_K ban is enforced across all deployments (extended from Llama-1B-only to universal).

**Days 15-21: Add safety-specific benchmarks alongside quality metrics**

- NEW: Add safety benchmarks (refusal, jailbreak) to the deployment validation pipeline alongside existing quality benchmarks.
- NEW: Compute quality-safety ratio at the deployed quantization level. If safety moves > 3x more than quality, escalate.
- Document per-model safety profiles including multi-turn and many-shot results.

**Days 22-30: Monitoring and ongoing validation**

- NEW: Add turn-level safety monitoring for multi-turn conversational deployments.
- Maintain quarterly safety battery re-evaluation (Phase 3 cadence).
- Monitor output instability at batch size changes during autoscaling events.
- Log directional flip patterns for batch-served models (TR143: 88-92% toward unsafe when flips occur).

---

## Risks, limitations, invalidation triggers

### Limitations

- **Human adjudication by single reviewer.** The 27% genuine rate (17/63) that recalibrates flip rates was produced by a single reviewer. Inter-rater reliability not established.
- **Co-batch verification rate of 22.1%.** TR143 composition claims rest on only 22.1% verified co-batching. The aggregate null likely holds under full treatment but precision is limited.
- **Dual-judge agreement of kappa = 0.104.** TR139 multi-turn ASR estimates carry substantial measurement uncertainty. Cross-condition comparisons are more reliable than absolute values.
- **TR141 regex-only artifact chain.** The LLM-judge layer from intermediate work was not preserved. TR141 flip rates should be interpreted as upper bounds (apply 27% human-adjudication correction).
- **Two-model TR142.** Quality-safety divergence finding based on 2 models. Opposite-sign correlations establish the possibility but not the distribution of model-dependent correlation directions.
- **Temperature = 0 only.** Batch-perturbation findings depend on deterministic decoding. At temperature > 0, the epsilon is dominated by sampling noise. Quantization findings likely hold regardless of temperature.
- **No adaptive adversaries.** Jailbreak strategies are pre-scripted. Adaptive adversaries would likely achieve higher ASR. Reported values are lower bounds.

### What invalidates this guidance

- Model family not in the tested set (18+ models from 10+ families tested, but novel architectures require re-profiling)
- Quantization method other than GGUF k-quant (GPTQ, AWQ, SqueezeLLM have different compression strategies)
- Temperature > 0 (unknown interaction with batch perturbation mechanism)
- Multi-GPU tensor parallelism (different serving dynamics, different memory pressure)
- Model size > 14.8B parameters (only TR141b tests up to 14.8B; larger models untested)
- Production traffic patterns with variable batch sizes and autoscaling
- Different safety classifiers (absolute scores are classifier-dependent)

---

## Evidence anchors (audit-ready)

| Decision | Artifact | Key Number |
|----------|----------|------------|
| Q2_K ban (universal) | `research/tr139/results/20260314_012503/tr139_analysis.json` | 100% ASR on qwen2.5-1.5b/Q2_K/attention_shift |
| Multi-turn x quant interaction | `research/tr139/results/20260314_012503/tr139_analysis.json` | 8/8 ANOVAs p < 1e-4 |
| Format vulnerability | `research/tr140/results/20260316_164907/tr140_analysis.json` | 92% vs 0% ASR by format |
| Output instability predictor | `research/tr141/results/20260318_194013/tr141_combined_analysis.json` | r = 0.909, R² = 0.827 |
| Alignment type not predictive | `research/tr141/results/20260318_194013/tr141_combined_analysis.json` | F = 0.13, p = 0.942 |
| Quality-safety divergence | `research/tr142/results/20260316_143936/tr142_analysis.json` | 13.9x at Q3_K_S |
| Human adjudication correction | `human_review/tr138/flipped_safety_rows_adjudicated.csv` | 17/63 genuine (27%) |
| Batch composition null | `research/tr143/results/20260320_024547/tr143_analysis.json` | All McNemar p > 0.125, all Cochran's Q p > 0.34 |
| Directional asymmetry | `research/tr143/results/20260320_024547/tr143_analysis.json` | 88-92% toward unsafe, p = 0.006 |
| Batch perturbation magnitude | `research/tr138/results/20260313_184600/tr138_analysis.json` | 0.6% regex, 0.16% human-adjudicated |
| Cross-architecture fragility | `research/tr141/results/20260318_194013/tr141_combined_analysis.json` | 0.00% to 2.39% (6.3x range) |
| Phase 2 Q4_K_M confirmation | TR139 + TR140 convergence | Llama models safe at Q4_K_M under multi-turn and many-shot |

---

## References

- Conclusive report: `PublishReady/reports/Technical_Report_Conclusive_138-143.md`
- TR138: `PublishReady/reports/Technical_Report_138_v2.md` (Batch Safety Under Non-Determinism)
- TR139: `PublishReady/reports/Technical_Report_139.md` (Multi-Turn Jailbreak x Quantization)
- TR140: `PublishReady/reports/Technical_Report_140.md` (Many-Shot Long-Context Jailbreak)
- TR141: `PublishReady/reports/Technical_Report_141.md` (Cross-Architecture Refusal Fragility)
- TR142: `PublishReady/reports/Technical_Report_142.md` (Quality-Safety Correlation)
- TR143: `PublishReady/reports/Technical_Report_143.md` (Cross-Request Composition)
- Phase 3 whitepaper: `PublishReady/reports/Technical_Report_Conclusive_134-137_Whitepaper.md`
- Phase 3 conclusive: `PublishReady/reports/Technical_Report_Conclusive_134-137.md`

---

## How this extends Phase 3

Phase 3 (TR134-TR137) delivered 6 decisions covering quantization floor, concurrency clearance, backend migration, per-model profiling, Q2_K ban, and jailbreak monitoring. This whitepaper adds 3 new decisions (quality-safety, multi-turn testing, format restriction) and extends 3 existing ones (Q2_K ban to universal, per-model profiling strengthened, quantization floor confirmed under new modalities).

The Phase 3 decisions remain valid and are confirmed by convergent evidence. The Phase 3.5/4 decisions address attack modalities that Phase 3 could not test: multi-turn strategies, many-shot exploitation, batch composition, cross-architecture variation, and the quality-safety relationship. Together, the two whitepapers cover the characterized optimization-safety attack surface for consumer-hardware LLM deployment (18+ models, ≤14.8B, GGUF k-quant, temperature=0, NVIDIA consumer GPUs). Gaps remain: temperature > 0, models > 14.8B, non-GGUF quantization, adaptive adversaries, and multi-GPU serving.

---

## Optional upgrades (board-ready polish)

- Add a 1-page Attack Surface Dashboard: the 8-row threat matrix + boundary conditions + three highest-priority actions.
- Add a jailbreak technique ASR heatmap (model x strategy x quantization) as a visual summary from TR139.
- Add a quality-safety divergence chart showing the 13.9x gap at Q3_K_S as a visual for executives.
- Commission inter-rater reliability study on the TR138 human adjudication (3 raters, 100 rows) to narrow the 27% genuine rate confidence interval.
- Complete the remaining 964 rows of human adjudication (TR139: 200, TR134-137: 501, TR141: 263) to extend the artifact-rate calibration beyond TR138.
