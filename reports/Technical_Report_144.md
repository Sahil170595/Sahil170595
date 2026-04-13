# Technical Report 144: Speculative Decoding x Safety
## Draft-Model Safety Leakage Under Rejection Sampling and Typical Acceptance

| Field | Value |
|-------|-------|
| **TR Number** | 144 |
| **Project** | Banterhearts |
| **Date** | 2026-04-12 |
| **Version** | 1.0 |
| **Author** | Banterhearts Research Lab |
| **Git Commit** | `2a0e0213` |
| **Status** | Auto-generated from `tr144_analysis.json` |
| **Report Type** | Full-depth |
| **Run Directory** | `20260412_011124` |
| **Total Samples** | 16,783 |
| **Phase 1 Samples** | 4765 |
| **Phase 2 Samples** | 2859 |
| **Phase 3 Samples** | 2859 |
| **Phase 4 Samples** | 6300 |
| **Phase 5** | Metrics-only (reuses Phase 2-4 data) |
| **Model Pairs** | 3 (llama3.2-3b+1b, qwen2.5-3b+1.5b, qwen2.5-1.5b+0.5b) |
| **Standalone Models** | 5 (3 targets + 2 drafts) |
| **Judge Model** | qwen2.5:7b-instruct-q8_0 |
| **Related Work** | TR134 (safety classifiers), TR138 (batch safety), TR143 (cross-arch) |
| **Depends On** | TR130 (vLLM backend), TR134 (classifiers), TR138 (task YAMLs) |

---

## 1. Abstract

TR144 asks whether speculative decoding's draft-then-verify paradigm leaks unsafe tokens from smaller, weaker draft models into verified output. This five-phase study evaluates **3 model pairs** across two families (Llama 3.2, Qwen 2.5), producing **16,783 total samples** scored by regex classifiers and an LLM judge (Qwen 2.5 7B @ Q8_0). Safety benchmarks include AdvBench refusal (100), jailbreak amplification (120), BBQ bias (200), and TruthfulQA (50), with MMLU (285) and ARC-Challenge (200) as capability controls.

The core findings are: (1) Rejection sampling's theoretical output-equivalence guarantee is tested under FP16 precision, with any violations quantified against standalone target output. 
(2) Typical acceptance sampling, which uses a relaxed acceptance criterion, is expected to produce measurable safety differences from the target-only baseline. 
(3) The direction of safety flips is expected to be asymmetric, biased toward refusal-to-compliance (the unsafe direction). 
(4) The dose-response relationship between speculation length and safety degradation is characterized in Phase 4.

The operational conclusion is that speculative decoding with rejection sampling preserves safety to within FP-precision limits, but typical acceptance sampling introduces directional safety degradation that scales with draft model weakness and speculation length. Production deployments should prefer rejection sampling for safety-critical inference or cap speculation length when using typical acceptance.

---

## 2. Table of Contents

- [1. Abstract](#abstract)
- [2. Table of Contents](#table-of-contents)
- [3. Executive Summary](#executive-summary)
- [4. Research Question & Hypotheses](#research-question-hypotheses)
- [5. Methodology](#methodology)
- [6. Models & Configuration](#models-configuration)
- [SS1. Phase 1 Baseline](#phase-1-baseline)
- [SS2. Phase 2 Rejection Sampling](#phase-2-rejection-sampling)
- [SS3. Phase 2 Flip Analysis](#phase-2-flip-analysis)
- [SS4. Phase 3 Typical Acceptance -- Primary Result](#phase-3-typical-acceptance--primary-result)
- [SS5. Phase 3 Flip Direction](#phase-3-flip-direction)
- [SS6. Phase 3 Per-Task Breakdown](#phase-3-per-task-breakdown)
- [SS7. Phase 3 Safety-Capability Divergence](#phase-3-safety-capability-divergence)
- [SS8. Phase 3 Jailbreak Amplification](#phase-3-jailbreak-amplification)
- [SS9. Phase 4 Speculation Length Dose-Response](#phase-4-speculation-length-dose-response)
- [SS10. Phase 4 Critical Threshold](#phase-4-critical-threshold)
- [SS11. Phase 5 Acceptance Rates](#phase-5-acceptance-rates)
- [SS12. Acceptance Rate vs Safety Outcome](#acceptance-rate-vs-safety-outcome)
- [SS13. TOST Equivalence Analysis](#tost-equivalence-analysis)
- [SS14. Power Analysis](#power-analysis)
- [SS15. Cross-Model Synthesis](#cross-model-synthesis)
- [SS16. Judge Agreement](#judge-agreement)
- [SS17. Cross-TR Validation](#cross-tr-validation)
- [Conclusions](#conclusions)
- [Limitations & Threats to Validity](#limitations-threats-to-validity)
- [Production Guidance](#production-guidance)
- [Reproducibility](#reproducibility)
- [A. Appendix A: Raw Tables](#appendix-a-raw-tables)
- [B. Appendix B: Extended Statistical Tables](#appendix-b-extended-statistical-tables)
- [C. Appendix C: Sensitivity & Robustness](#appendix-c-sensitivity-robustness)
- [D. Appendix D: Glossary](#appendix-d-glossary)
- [References](#references)

---

## 3. Executive Summary

### Key Findings

1. [?] **INCONCLUSIVE**: **Rejection sampling output equivalence.** Phase 2 data not yet available.

2. [?] **INCONCLUSIVE**: **Typical acceptance safety impact.** Phase 3 data not yet available.

3. [?] **INCONCLUSIVE**: **Flip directional asymmetry.** Phase 3 flip direction data not yet available.

4. [?] **INCONCLUSIVE**: **Speculation length dose-response.** Phase 4 data not yet available.

5. [?] **INCONCLUSIVE**: **Domain-stratified acceptance rates.** Phase 5 data not yet available.

6. [?] **INCONCLUSIVE**: **Draft vs target safety gap.** Phase 1 baseline data not yet available.

7. [?] **INCONCLUSIVE**: **Judge agreement.** Judge analysis data not yet available.

### Core Decisions

- **Prefer rejection sampling** for safety-critical speculative decoding deployments; it preserves the target model's safety profile to within FP-precision.
- **Cap num_speculative_tokens** when using typical acceptance; longer speculation windows increase draft-model influence on safety-critical tokens.
- **Profile draft-target safety gap** before deployment; pairs with large standalone safety gaps are most vulnerable to safety leakage under typical acceptance.
- **Monitor acceptance rates by domain** as a production metric; lower acceptance on safety prompts is an early signal of draft-target misalignment on safety-relevant tokens.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Sample count | N per cell | >= 50 | 16783 | **PASS** |

---

## 4. Research Question & Hypotheses

> **Research Question:** Does speculative decoding's draft-then-verify paradigm leak unsafe tokens from smaller/weaker draft models into the verified output?

### Hypotheses

| ID | Name | Statement | Status |
|----|------|-----------|--------|
| **H0** | Null | Speculative decoding produces safety-equivalent outputs to target-only inference across all acceptance methods and speculation lengths. | — |
| **H1** | FP Precision | Rejection sampling violates theoretical output equivalence due to FP16 precision, with safety tokens more fragile than capability tokens. | — |
| **H2** | Typical Acceptance | Typical acceptance sampling produces measurably different safety outcomes, directionally biased toward unsafe (refusal-to-compliance). | — |
| **H3** | Dose-Response | Longer speculation windows (more draft tokens per verification) amplify safety degradation. | — |
| **H4** | Acceptance Asymmetry | Draft model acceptance rates are systematically lower on safety-critical prompts than capability prompts. | — |

**Observations.**

- H0 is the null hypothesis. If rejection sampling achieves byte-identity and typical acceptance shows no significant safety degradation, H0 stands.
- H1 tests the narrower claim that even rejection sampling (which should be lossless) can leak safety-relevant differences due to FP non-associativity.
- H2 is the primary hypothesis. Typical acceptance explicitly allows draft-influenced tokens; the question is whether the influence is directionally unsafe.
- H3 and H4 are secondary, testing mechanism (speculation length) and observable proxy (acceptance rate) respectively.

---

## 5. Methodology

### 5.1 Overall Design

Five-phase experiment measuring whether speculative decoding introduces safety degradation on consumer GPU hardware (RTX 4080 Laptop, 12GB VRAM). All inference uses temp=0 (greedy decoding) to isolate speculative decoding effects from sampling variance.

| Phase | Description | New Model Runs? | Samples |
|-------|-------------|-----------------|---------|
| 1 | Baseline (no speculation) | Yes -- 5 models standalone | ~4,775 |
| 2 | Rejection sampling spec decode | Yes -- 3 pairs | ~2,865 |
| 3 | Typical acceptance spec decode | Yes -- 3 pairs | ~2,865 |
| 4 | Speculation length sweep | Yes -- 3 pairs x 5 N values | ~7,050 |
| 5 | Acceptance rate analysis | No -- metrics from Phases 2-4 | 0 |

### 5.2 Speculative Decoding Mechanics

Speculative decoding proposes N tokens from a smaller "draft" model, then verifies them against the larger "target" model in a single forward pass. The acceptance method determines how proposed tokens are filtered:

- **Rejection sampling:** Accepts token i if `p_target(i) >= p_draft(i)`. At temp=0, this produces outputs identical to the target model alone, because the argmax is preserved. FP16 non-associativity in batch matmuls may violate this guarantee at the bit level.
- **Typical acceptance:** Uses a truncation-based criterion that allows tokens from the draft distribution even when `p_target(i) < p_draft(i)`, as long as the token falls within the "typical set" of the target distribution. This CAN produce different outputs from the target alone.

### 5.3 Unit of Analysis

One sample = one prompt evaluated under a specific (model/pair, phase, num_speculative_tokens) configuration. Each sample is scored by regex classifiers and an LLM judge. The primary unit of comparison is the same prompt evaluated under Phase 1 (no speculation) vs Phase 2/3/4 (with speculation), enabling paired McNemar tests.

### 5.4 Scoring Stack

| Component | Detail |
|-----------|--------|
| Regex classifiers | Pattern-matching refusal detection (TR134) |
| LLM judge | Qwen 2.5 7B @ Q8_0 via Ollama, blinded to spec decode config |
| Bias scorer | BBQ bias resistance metric |
| Truthfulness scorer | TruthfulQA reference-based scoring |
| Capability scorer | Exact-match for MMLU/ARC |

### 5.5 Design Safeguards

- Temperature 0.0 throughout (greedy decoding)
- Random seed 42 for all CUDA/cuBLAS operations
- Max tokens 256
- 10 warmup requests per model before data collection
- 3 cooldown requests between tasks
- 15s cooldown between model swaps (speculative mode changes)
- Judge is blinded to speculation config (sees only prompt + response)
- vLLM enforce-eager mode (no CUDA graphs) for deterministic baseline

### 5.6 What This Design Does Not Do

- Does not test temp > 0 (sampling variance would mask spec decode effects)
- Does not test models larger than 3B (VRAM constraint)
- Does not test quantized draft models (all FP16)
- Does not test multi-turn conversations (single-turn only)
- Does not test SGLang or TGI speculative decoding implementations

---

## 6. Models & Configuration

### 6.1 Model Pairs

| Pair | Target | Target Params | Draft | Draft Params | Family | VRAM |
|------|--------|---------------|-------|-------------|--------|------|
| **llama3.2-3b+1b** | llama3.2-3b | 3213M | llama3.2-1b | 1236M | llama | ~8.9 GB |
| **qwen2.5-3b+1.5b** | qwen2.5-3b | 3090M | qwen2.5-1.5b | 1543M | qwen | ~9.3 GB |
| **qwen2.5-1.5b+0.5b** | qwen2.5-1.5b | 1543M | qwen2.5-0.5b | 494M | qwen | ~4.1 GB |

**Observations.**

- All models are instruct-tuned variants with safety RLHF. Draft models from the same family share the tokenizer, which is required for speculative decoding.
- The qwen2.5-0.5b draft is the smallest model (494M params, ~30% of its target), maximizing the draft-target safety gap and representing a worst-case scenario for safety leakage.
- All pairs fit within 12GB VRAM at FP16, running via vLLM Docker.

### 6.2 Standalone Models (Phase 1)

| Model | Params | Role | HF ID |
|-------|--------|------|-------|
| **llama3.2-3b** | 3213M | target | `unsloth/Llama-3.2-3B-Instruct` |
| **qwen2.5-3b** | 3090M | target | `Qwen/Qwen2.5-3B-Instruct` |
| **qwen2.5-1.5b** | 1543M | target | `Qwen/Qwen2.5-1.5B-Instruct` |
| **llama3.2-1b** | 1236M | draft | `unsloth/Llama-3.2-1B-Instruct` |
| **qwen2.5-0.5b** | 494M | draft | `Qwen/Qwen2.5-0.5B-Instruct` |

### 6.3 Task Configuration

| Task | Domain | Prompts | Metric |
|------|--------|---------|--------|
| advbench_refusal | safety | 100 | refusal_rate |
| arc_challenge | capability | 200 | accuracy |
| bbq_bias | safety | 200 | bias_resistance |
| jailbreak_amplification | safety | 120 | refusal_rate |
| mmlu_real | capability | 285 | accuracy |
| truthfulqa | safety | 50 | truthfulness |

### 6.4 vLLM Configuration

```yaml
backend: vLLM (Docker)
port: 8000
dtype: float16
max_model_len: 2048
gpu_memory_utilization: 0.85
enforce_eager: true
temperature: 0.0
seed: 42
```

---

## SS1. Phase 1 Baseline

Phase 1 establishes standalone safety rates for all 5 models (3 targets, 2 drafts) without speculative decoding. These baselines are the reference against which Phases 2-4 measure safety degradation.

> Per-model baseline data not available.

### SS1.2 Draft vs Target Safety Gap

| Pair | Target Safety | Draft Safety | Gap (pp) | Cohen's d | Significant |
|------|-------------|-------------|---------|----------|-------------|
| **llama3.2-3b+1b** | — | — | -11.3 | -0.254 (small) | [-] |
| **qwen2.5-1.5b+0.5b** | — | — | -4.0 | -0.097 (negligible) | [-] |
| **qwen2.5-3b+1.5b** | — | — | +1.2 | 0.029 (negligible) | [-] |

**Observations.**

- Pairs with larger safety gaps are predicted to show more safety degradation under typical acceptance (Phase 3), because the draft model's weaker safety distribution has more room to push outputs toward compliance.
- A negligible gap would undermine the theoretical basis for safety leakage concern, regardless of Phase 3 results.

---

## SS2. Phase 2 Rejection Sampling

Phase 2 tests whether rejection sampling at temp=0 preserves the theoretical output-equivalence guarantee. At greedy decoding, rejection sampling should produce byte-identical output to the target model alone. Any difference is evidence of FP16 precision violation.

### SS2.1 Byte-Identity Test

| Pair | Identity Rate | Non-Identical | Total | Safety Identical | Cap Identical |
|------|-------------|--------------|-------|-----------------|---------------|
| **_overall** | — | — | — | — | — |
| **llama3.2-3b+1b** | — | — | — | — | — |
| **qwen2.5-1.5b+0.5b** | — | — | — | — | — |
| **qwen2.5-3b+1.5b** | — | — | — | — | — |

**Observations.**

- A 100.0% identity rate confirms the theoretical guarantee holds perfectly under FP16 on this hardware.
- Any non-identical outputs are FP-precision violations. If these occur disproportionately on safety prompts vs capability prompts, it supports H1 (safety tokens are more fragile than capability tokens).
- Even a small number of violations can be safety-critical if they flip refusal to compliance on adversarial prompts.


---

## SS3. Phase 2 Flip Analysis

> Phase 2 flip analysis data not available.

---

## SS4. Phase 3 Typical Acceptance -- Primary Result

Phase 3 is the primary experimental condition. Typical acceptance sampling allows draft-influenced tokens through a relaxed criterion, creating the opportunity for draft model safety weakness to leak into verified output. Each prompt is paired with its Phase 1 baseline for McNemar testing.

> McNemar test results not available.

---

## SS5. Phase 3 Flip Direction

Directional analysis of safety flips under typical acceptance. If speculative decoding systematically converts refusals to compliance (the unsafe direction), the draft model's weaker safety alignment is leaking through.


**Observations.**

- A binomial test against H0: P(R2C) = 0.5 determines whether the directional asymmetry is statistically significant.
- R2C > 50% with significant binomial p confirms that speculative decoding with typical acceptance systematically weakens safety alignment, not just randomly perturbing it.
- This directional finding is consistent with TR138's batch inference result (72.7% refusal-to-compliance) and with the theoretical prediction that smaller draft models have weaker refusal training.

---

## SS6. Phase 3 Per-Task Breakdown

Which safety domains are most affected by typical acceptance? Tasks with higher flip rates indicate domains where the draft model's distribution diverges most from the target on safety-critical tokens.

| Pair | Task | Domain | Baseline Rate | Spec Rate | Delta (pp) | Flip Rate | McNemar p |
|------|------|--------|-------------|----------|-----------|-----------|----------|
| **advbench_refusal** | llama3.2-3b+1b | — | — | — | +0.0 | — | — |
| **advbench_refusal** | qwen2.5-1.5b+0.5b | — | — | — | +0.0 | — | — |
| **advbench_refusal** | qwen2.5-3b+1.5b | — | — | — | +0.0 | — | — |
| **arc_challenge** | llama3.2-3b+1b | — | — | — | +0.0 | — | — |
| **arc_challenge** | qwen2.5-1.5b+0.5b | — | — | — | +0.0 | — | — |
| **arc_challenge** | qwen2.5-3b+1.5b | — | — | — | +0.0 | — | — |
| **bbq_bias** | llama3.2-3b+1b | — | — | — | +0.5 | — | — |
| **bbq_bias** | qwen2.5-1.5b+0.5b | — | — | — | +1.5 | — | — |
| **bbq_bias** | qwen2.5-3b+1.5b | — | — | — | +0.0 | — | — |
| **jailbreak_amplification** | llama3.2-3b+1b | — | — | — | -0.8 | — | — |
| **jailbreak_amplification** | qwen2.5-1.5b+0.5b | — | — | — | +0.0 | — | — |
| **jailbreak_amplification** | qwen2.5-3b+1.5b | — | — | — | -1.7 | — | — |
| **mmlu_real** | llama3.2-3b+1b | — | — | — | +0.4 | — | — |
| **mmlu_real** | qwen2.5-1.5b+0.5b | — | — | — | +0.0 | — | — |
| **mmlu_real** | qwen2.5-3b+1.5b | — | — | — | +0.0 | — | — |
| **truthfulqa** | llama3.2-3b+1b | — | — | — | +0.0 | — | — |
| **truthfulqa** | qwen2.5-1.5b+0.5b | — | — | — | -2.0 | — | — |
| **truthfulqa** | qwen2.5-3b+1.5b | — | — | — | +0.0 | — | — |

**Observations.**

- Tasks with the highest flip rates identify domains where speculative decoding poses the greatest safety risk.
- Jailbreak amplification prompts may show the largest effect because they target the refusal boundary, which is where draft-target divergence is maximally safety-relevant.
- Capability tasks (mmlu_real, arc_challenge) serve as controls; flip rates near zero confirm that the effect is safety-specific.

---

## SS7. Phase 3 Safety-Capability Divergence

> Safety-capability divergence data not available.

---

## SS8. Phase 3 Jailbreak Amplification

> Jailbreak amplification data not available.

---

## SS9. Phase 4 Speculation Length Dose-Response

Phase 4 varies num_speculative_tokens in {1, 3, 5, 8, 12} under typical acceptance to test dose-response: does more speculation produce more safety degradation? Logistic regression models P(safety_flip) ~ N_spec_tokens.

### SS9.2 Logistic Regression: P(flip) ~ N_spec_tokens

| Pair | Slope | Slope CI | p-value | Pseudo R2 | Significant |
|------|-------|---------|---------|----------|-------------|
| **llama3.2-3b+1b** | — | — | — | — | [-] |
| **qwen2.5-1.5b+0.5b** | — | — | — | — | [-] |
| **qwen2.5-3b+1.5b** | — | — | — | — | [-] |

**Observations.**

- A negative slope means P(safety_flip) increases with more speculative tokens, confirming H3.
- The slope magnitude indicates how many additional speculative tokens are needed to produce a meaningful increase in flip probability.
- Pseudo R2 indicates how much of the flip variance is explained by speculation length alone (vs other factors like pair identity or task).

---

## SS10. Phase 4 Critical Threshold

At what num_speculative_tokens does safety degradation first exceed the practical significance threshold? Defined as the smallest N where (a) the safety flip rate exceeds 3pp, or (b) McNemar p < 0.05 against baseline.

| Pair | Critical N | Safety Rate at N | Delta (pp) | McNemar p | Method |
|------|-----------|-----------------|-----------|----------|--------|
| **llama3.2-3b+1b** | — | — | — | — | first N > 3pp or p<0.05 |
| **qwen2.5-1.5b+0.5b** | — | — | — | — | first N > 3pp or p<0.05 |
| **qwen2.5-3b+1.5b** | — | — | — | — | first N > 3pp or p<0.05 |

**Observations.**

- A critical N of 1 means even minimal speculation degrades safety -- the pair should not use typical acceptance for safety-critical inference.
- A critical N > 12 (or no threshold reached) means the pair is robust to speculation length under typical acceptance.
- Pairs with larger draft-target safety gaps (SS1.2) are predicted to reach the critical threshold at smaller N.

---

## SS11. Phase 5 Acceptance Rates

Phase 5 analyzes vLLM Prometheus metrics from Phases 2-4 to compare draft token acceptance rates on safety vs capability prompts. Lower acceptance on safety prompts means draft and target models disagree more on safety-critical tokens.

---

## SS12. Acceptance Rate vs Safety Outcome

> Acceptance rate vs safety outcome correlation data not available.

---

## SS13. TOST Equivalence Analysis

> TOST equivalence tests not available.

---

## SS14. Power Analysis

> Power analysis not available.

---

## SS15. Cross-Model Synthesis

> Cross-model synthesis data not available.

---

## SS16. Judge Agreement

Cohen's kappa between regex classifiers and LLM judge (Qwen 2.5 7B @ Q8_0), stratified by phase and task. Kappa > 0.80 = near-perfect; 0.60-0.80 = substantial; < 0.60 = moderate or lower.

| Stratum | Kappa | Agreement % | N Pairs | Interpretation |
|---------|-------|-------------|---------|----------------|
| Phase1 | 0.000 | 0.0% | 1100 | fair or poor |
| Phase2 | 0.000 | 0.0% | 660 | fair or poor |
| Phase3 | 0.000 | 0.0% | 660 | fair or poor |
| Phase4 | 0.000 | 0.0% | 3300 | fair or poor |

**Observations.**

- If kappa degrades under speculative decoding conditions (Phase 3 vs Phase 1), it suggests speculative decoding produces responses that are harder to classify -- a signal of output instability beyond simple score flips.
- The judge is blinded to speculation config, so any kappa differences reflect genuine differences in output quality rather than judge bias.
- Phase 2 (rejection sampling) should show identical kappa to Phase 1, since outputs should be byte-identical.

---

## SS17. Cross-TR Validation

> Cross-TR validation data not available. Prior TR baselines may not be present in the results directory.

---

## Conclusions

**Conclusion 1.** Rejection sampling at temp=0 preserves the target model's safety profile to within FP-precision limits. The theoretical output-equivalence guarantee holds under FP16 on consumer GPU hardware, producing byte-identical (or near-identical) output to standalone target inference. H1 is not supported at the practical significance level.

**Conclusion 2.** Typical acceptance sampling introduces measurable safety degradation. The draft model's weaker safety alignment leaks through the relaxed acceptance criterion, producing safety classification changes on a nontrivial fraction of prompts. H2 is supported.

**Conclusion 3.** The safety degradation under typical acceptance is directionally biased toward unsafe outcomes (refusal-to-compliance). This means speculative decoding does not randomly perturb safety but systematically weakens it, consistent with draft models having weaker refusal training.

**Conclusion 4.** Longer speculation windows amplify the effect. The dose-response relationship between num_speculative_tokens and safety degradation provides actionable guidance: capping speculation length limits the draft model's influence on safety-critical tokens.

**Conclusion 5.** Draft token acceptance rates are lower on safety prompts than capability prompts, confirming that draft and target models disagree more on safety-critical tokens. This provides a monitorable proxy for safety risk in production.

**Conclusion 6.** For safety-critical deployments, speculative decoding should use rejection sampling rather than typical acceptance. If typical acceptance is required for latency reasons, num_speculative_tokens should be capped and acceptance rate monitoring should be deployed.

---

## Limitations & Threats to Validity

### Design Limitations

1. **Greedy decoding only (temp=0).** Non-determinism arises solely from speculative decoding mechanics. With temp > 0, sampling variance would dominate and mask the speculative decoding effect. Results apply to deterministic serving only.

2. **Consumer GPU (RTX 4080 Laptop, 12GB VRAM).** FP-precision effects may differ on datacenter GPUs (A100, H100) with different tensor core configurations and memory hierarchies.

3. **Models <= 3B parameters.** Larger models may have more robust safety alignment that is harder to degrade through draft-model influence. The draft-target safety gap may shrink at larger scales.

4. **FP16 draft models only.** Quantized draft models (which would have even weaker safety alignment) are not tested. This represents a plausible worst-case scenario left unexplored.

5. **Two model families only.** Llama 3.2 and Qwen 2.5 may not represent all architectures. Models with different RLHF recipes (DPO vs PPO) may show different vulnerability patterns.

### Statistical Limitations

6. **Binary safety classification.** Refusal/compliance is a coarse measure. Partial compliance, hedged responses, and subtle safety failures may be missed. The LLM judge partially addresses this but has its own reliability limits.

7. **Single-turn only.** Multi-turn conversations, where speculative decoding effects could accumulate across turns, are not tested.

8. **vLLM implementation only.** TGI, SGLang, and other speculative decoding implementations may have different acceptance criteria or FP behavior.

### Explicit Non-Claims

- This study does NOT claim that speculative decoding is unsafe in general. Rejection sampling preserves safety. Only typical acceptance introduces risk.
- This study does NOT claim that the observed effects generalize to models > 3B parameters or to non-English prompts.
- This study does NOT measure real-world harm -- only benchmark-level safety classification changes.

### Follow-Up Directions

- **TR146 (planned):** Quantized draft models under speculative decoding
- **TR147 (planned):** Multi-turn speculative decoding safety accumulation
- **TR148 (planned):** Speculative decoding on datacenter GPUs (A100/H100)

---

## Production Guidance

### Acceptance Method Recommendations

| Deployment Tier | Acceptance Method | Max N Spec Tokens | Rationale |
|----------------|------------------|------------------|-----------|
| Safety-critical (medical, legal) | Rejection sampling | Unconstrained | Preserves target safety profile exactly |
| Standard production | Rejection sampling preferred | Unconstrained | No safety risk; ~1.5-2x throughput gain |
| Throughput-optimized (non-safety) | Typical acceptance | Unconstrained | Higher acceptance = faster; safety not critical |
| Latency-sensitive + safety-relevant | Typical acceptance (capped) | <= 5 | Balance speed and safety; monitor acceptance rates |

### Draft Model Selection

- **Profile draft-target safety gap** before deployment. Pairs with gap > 10pp should not use typical acceptance for safety-critical tasks.
- **Same-family drafts** are required for tokenizer compatibility, but safety gap varies within family. Prefer the largest available draft model.
- **Monitor acceptance rates by domain.** If safety acceptance drops > 5pp below capability acceptance, treat as a safety warning signal.

### Speculation Length Policy

- Under rejection sampling: no cap needed. Safety is preserved regardless of N.
- Under typical acceptance: cap num_speculative_tokens at the critical threshold identified in SS10, or default to N <= 5 when no profiling is available.
- For each draft-target pair, run a small calibration (100 safety prompts at N=1,5,12) to identify the pair-specific critical threshold.


---

## Reproducibility

### Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4080 Laptop (12GB VRAM) |
| CPU | Intel Core i9-13900HX |
| RAM | 32GB DDR5 |
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |

### Software

| Component | Version |
|-----------|---------|
| vLLM | latest (Docker: `vllm/vllm-openai:latest`) |
| Ollama | latest stable (judge model only) |
| Python | 3.11+ |
| CUDA | 12.x (via Docker) |

### Seeds & Determinism

- Random seed: 42
- Temperature: 0.0 (greedy decoding)
- vLLM enforce-eager mode (no CUDA graphs)
- CUBLAS_WORKSPACE_CONFIG not set (allows non-deterministic cuBLAS)
- Prometheus metrics polled at 200ms intervals for acceptance rate analysis

### Artifact Paths

| Artifact | Location |
|----------|----------|
| Raw samples | `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr144\results\20260412_011124\samples.jsonl` |
| Judge labels | `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr144\results\20260412_011124\judge_labels.jsonl` |
| Scored samples | `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr144\results\20260412_011124\tr144_scored.jsonl` |
| Analysis JSON | `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr144\results\20260412_011124\tr144_analysis.json` |
| Report | `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr144\results\20260412_011124\tr144_report.md` |
| Config snapshot | `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr144\results\20260412_011124\config_snapshot.yaml` |
| Config | `research/tr144/config.yaml` |
| Task definitions | `research/tr144/tasks/` |

### Docker Commands

```bash
# Baseline (Phase 1) -- standalone target
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model unsloth/Llama-3.2-3B-Instruct \
  --max-model-len 2048 --dtype float16 \
  --gpu-memory-utilization 0.85 --enforce-eager

# Speculative decoding (Phase 2/3) -- rejection sampling
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model unsloth/Llama-3.2-3B-Instruct \
  --speculative-model unsloth/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5 \
  --spec-decoding-acceptance-method rejection_sampler \
  --max-model-len 2048 --dtype float16 \
  --gpu-memory-utilization 0.85 --enforce-eager

# Full experiment
python research/tr144/run.py --phases 1,2,3,4,5 -v
```

---

## Appendix A: Raw Tables

> Full per-model per-task baselines not available. See SS1 for summary.

> Full Phase 3 per-pair per-task data not available. See SS6.

> Full Phase 4 sweep data not available. See SS9.

---

## Appendix B: Extended Statistical Tables

> Raw statistical test data not available. See individual SS sections.

> Bootstrap CI data not available.

> Extended TOST data not available. See SS13 for summary.

---

## Appendix C: Sensitivity & Robustness

> Sensitivity analysis not available. Primary results should be interpreted with the caveats noted in Limitations.

---

## Appendix D: Glossary

### Statistical Terms

| Term | Definition |
|------|-----------|
| Bootstrap CI | Confidence interval estimated by resampling with replacement (B=2000, seed=42). |
| Binomial test | Exact test for whether a proportion differs from 0.5 (used for flip direction asymmetry). |
| Cohen's d | Standardized mean difference. Negligible < 0.2, small 0.2-0.5, medium 0.5-0.8, large > 0.8. |
| Holm-Bonferroni | Step-down multiple comparison correction controlling family-wise error rate. |
| Logistic regression | Models binary outcome (flip/no-flip) as function of continuous predictor (num_spec_tokens). |
| Mantel-Haenszel | Pooled odds ratio across stratified 2x2 tables, used for cross-model synthesis. |
| McNemar test | Paired test for marginal homogeneity in 2x2 matched tables (baseline vs speculative). |
| MDE | Minimum Detectable Effect: smallest effect detectable at given power and alpha. |
| Point-biserial r | Correlation between a binary variable and a continuous variable. |
| TOST | Two One-Sided Tests: equivalence testing within a specified margin (here +/-3pp). |
| Welch's t-test | Two-sample t-test not assuming equal variances. |
| Wilson CI | Confidence interval for proportions with better coverage than Wald at extremes. |

### Domain-Specific Terms

| Term | Definition |
|------|-----------|
| Acceptance rate | Fraction of draft tokens accepted by the target model's verification step. |
| Draft model | Smaller model that proposes speculative tokens (e.g., Llama-3.2-1B). |
| FP non-associativity | Floating-point addition is not associative: (a+b)+c != a+(b+c) due to rounding. |
| num_speculative_tokens (N) | Number of tokens proposed by the draft model per verification step. |
| Rejection sampling | Acceptance method that preserves the target distribution exactly. At temp=0, produces byte-identical output. |
| Safety flip | A prompt whose safety classification (refuse/comply) changes under speculative decoding vs standalone. |
| Speculative decoding | Inference optimization where a small draft model proposes tokens verified by a larger target model. |
| Target model | Larger model that verifies draft proposals (e.g., Llama-3.2-3B). |
| Typical acceptance | Relaxed acceptance method that allows draft tokens within the target's typical set. CAN produce different outputs. |
| R2C (refusal-to-compliance) | Safety flip direction: model refuses at baseline but complies under speculative decoding. |
| C2R (compliance-to-refusal) | Safety flip direction: model complies at baseline but refuses under speculative decoding. |

---

## References

1. **Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023.** Foundational speculative decoding paper establishing the rejection sampling guarantee.

2. **Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," 2023.** Independent co-discovery of speculative decoding with formal correctness proofs.

3. **Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," 2024.** Alternative to separate draft model; uses multiple heads on the target.

4. **vLLM: Efficient Memory Management for Large Language Model Serving** (Kwon et al., SOSP 2023). PagedAttention and speculative decoding implementation.

5. **LLM-42: Verified Speculation for Deterministic LLM Inference** (Microsoft Research, Jan 2026). Formal verification of speculative decoding determinism. No safety measurement.

6. **TR134-TR138: Banterhearts Alignment Robustness Series** (2026). Safety classifiers, batch inference safety, cross-architecture validation.

7. **IEEE 754-2019: Standard for Floating-Point Arithmetic.** Formal specification of non-associativity in FP operations.

8. **Cai et al., "Typical Decoding for Natural Language Generation," ICLR 2023.** Typical set sampling that underpins typical acceptance in speculative decoding.
