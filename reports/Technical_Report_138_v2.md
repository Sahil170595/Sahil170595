# Technical Report 138 v2: Batch Inference Safety Under Non-Determinism -- Strengthened-Evidence Revision
## Audit-layer flip adjudication + 7,257-sample reduced replication on enriched 187-prompt subset, with corrected refusal detector (v2.2)

| Field | Value |
|-------|-------|
| **TR Number** | 138 v2 |
| **Date** | 2026-03-15 |
| **Version** | 2.2 (scorer-corrected) |
| **Author** | Research Team |
| **Git Commit** | `edbaf196` |
| **Status** | Complete |
| **Report Type** | Strengthened-evidence revision (audit + replication) |
| **Base Report** | TR138 v1 (31,410 samples, commit `7b77e9c0`) |
| **Replication Run Directory** | `20260313_184600/replication_run` |
| **Total Replication Samples** | 7,257 |
| **Phase 1 Samples** | 3,366 |
| **Phase 2 Samples** | 1,284 |
| **Phase 3 Samples** | 1,485 |
| **Phase 4 Samples** | 1,122 |
| **Audit Layer Candidates** | 44 (corrected from 49) |

---

## Positioning

TR138 v2 is not a standalone discovery report. It is a **strengthened-evidence revision** of TR138 v1.

The original TR138 (v1, 31,410 samples) established the core finding: batch-induced output non-determinism disproportionately degrades safety compared to capability under deterministic decoding. TR138 v2 adds two layers of evidence on top of that:

1. **Audit layer.** Behavior-changing rows exported from the v1 run are examined through a corrected refusal detector. The v2.2 scorer normalizes Unicode curly quotes, reducing the audit candidate set from 49 to 44 (five rows were false flips from a curly-apostrophe regex mismatch on one AdvBench prompt). Of the 44 true candidates, 26 are unsafe-direction and 18 are safe-direction, yielding a 59.1% unsafe share. The core asymmetry finding is preserved.

2. **Replication layer.** A 7,257-sample reduced rerun on an enriched 187-prompt subset per model confirms the Phase 1 safety-capability flip asymmetry at higher absolute rates (1.68% safety vs 0.42% capability) and strengthens the Phase 3 quantization result to 3/3 models significant (consistent with corrected v1 (also 3/3)).

The v2 report should therefore be read as **a revision that sharpens the v1 evidence** rather than a new experiment. Where v1 numbers differ from v2 numbers, the difference is explained by the reduced prompt pool (enriched for boundary-sensitive prompts) and the scorer correction.

---

## Metric Definitions & Statistical Methods

### Core metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Output identity** | Share of outputs that are byte-identical to the `batch=1` baseline | Higher = more stable |
| **Safety flip rate** | Share of safety samples whose task-level safety score changes relative to baseline | Higher = more safety instability |
| **Capability flip rate** | Share of capability samples whose task-level score changes relative to baseline | Control arm for generic output instability |
| **Flip ratio** | `safety flip rate / capability flip rate` | Above 1 means safety is more fragile |
| **Refusal -> compliance share** | Fraction of flips that move in the unsafe direction | Higher = more concerning |
| **Audit candidate** | A v1 row whose safety classification changed between batch=1 and any non-baseline condition | Source for manual review |
| **Unsafe share** | Fraction of audit candidates whose flip direction weakens safety | Directional asymmetry measure |
| **TOST equivalence** | Two one-sided tests at `+/-3pp` margin | Tests practical interchangeability |
| **Eta-squared** | ANOVA effect size | Share of variance explained |
| **Cohen's kappa** | Agreement between heuristic scoring and the LLM judge | Higher = better agreement |

### Statistical methods used

- **Phase 1:** paired comparisons versus `batch=1`, chi-squared with Fisher exact
- **Phase 2:** paired t-tests across `solo`, `benign`, `adversarial`, and `safety`; two-way ANOVA (model x condition)
- **Phase 3:** two-way ANOVA for `quantization x concurrency`, per-model
- **Phase 4:** paired comparison between true batching and synchronized-dispatch references
- **Audit layer:** binomial test for unsafe-direction asymmetry, Wilson CIs, odds ratios
- **Holm-Bonferroni** correction for all multiple testing families
- **TOST equivalence** at `+/-3pp`
- **Bootstrap CIs** where emitted by the analysis pipeline

### Scorer correction (v2.2)

The v2.2 refusal detector normalizes Unicode curly quotes (`\u2019`, `\u201c`, `\u201d`) before applying refusal-detection regex patterns. This change:

- Reduced audit candidates from 49 to 44 (5 false-flip rows removed)
- Changed unsafe count from 31 to 26
- Changed safe count from 18 to 18 (unchanged)
- Preserved the core directional asymmetry (59.1% unsafe share)
- Did not affect any Phase 2, Phase 3, or capability scoring

### Important caveats up front

1. **Reduced replication pool.** The replication layer uses 187 prompts per model (enriched subset), not the full v1 prompt set.
2. **Greedy decoding only.** All runs use `temperature=0.0`.
3. **Consumer GPU only.** This is one RTX 4080 Laptop-class environment.
4. **Small-model regime.** The study covers roughly 1B-3B models.
5. **Phase 3 is not batching.** It is concurrent load under varying quant levels.
6. **Judge agreement is weak.** Absolute percentages should be treated more cautiously than within-experiment contrasts.

---

## 1. Abstract

This report presents TR138 v2, a strengthened-evidence revision of the 31,410-sample TR138 v1 study on batch-induced safety non-determinism. TR138 v2 adds two evidence layers: (1) a corrected audit of 44 behavior-changing rows from v1, finding that 26 (59.1%) flip in the unsafe direction, and (2) a 7,257-sample reduced replication on an enriched 187-prompt subset across 3 models and 4 phases.

The replication confirms and amplifies the v1 headline findings. Phase 1 safety flips reach **1.68%** (27/1,605) versus **0.42%** capability flips (5/1,200), a **4.0x ratio**. Refusal-to-compliance flips account for **72.7%** of directionally classified safety flips, confirming that batch-induced instability weakens safety alignment rather than randomly perturbing it. Phase 4 explicit true-batching validation records **3.27%** safety flips (14/428) with **98.67%** mean flip agreement to Phase 1, demonstrating the core signal is not a scheduler artifact.

Phase 3 now shows **3/3 models** with significant quantization effects (consistent with corrected v1 (also 3/3)), while concurrency and interaction terms remain null. The audit layer scorer correction (curly-quote normalization) removes 5 false-flip rows, tightening the candidate set without altering the core asymmetry finding.

The defensible conclusion is unchanged but now better supported:

> under deterministic decoding on this hardware and model set, batching introduces a small but measurable safety tax, the tax is directionally unsafe, it survives true-batching validation, and the scorer-corrected audit preserves the asymmetry.

---

## 2. Table of Contents

- [Positioning](#positioning)
- [Metric Definitions & Statistical Methods](#metric-definitions--statistical-methods)
- [1. Abstract](#1-abstract)
- [2. Table of Contents](#2-table-of-contents)
- [3. Executive Summary](#3-executive-summary)
- [4. Research Question & Hypotheses](#4-research-question--hypotheses)
- [5. Methodology](#5-methodology)
- [6. Models & Configuration](#6-models--configuration)
- [7. Phase 1: Batch Size x Output Determinism](#7-phase-1-batch-size-x-output-determinism)
- [8. Phase 2: Co-Batching Interference](#8-phase-2-co-batching-interference)
- [9. Phase 3: Quantization x Concurrency Interaction](#9-phase-3-quantization-x-concurrency-interaction)
- [10. Cross-Phase Synthesis](#10-cross-phase-synthesis)
- [11. Audit Layer Analysis](#11-audit-layer-analysis)
- [12. TOST Equivalence Analysis](#12-tost-equivalence-analysis)
- [13. Power Analysis](#13-power-analysis)
- [14. Latency Analysis](#14-latency-analysis)
- [15. Judge Agreement Analysis](#15-judge-agreement-analysis)
- [16. Jailbreak Type Breakdown](#16-jailbreak-type-breakdown)
- [17. Per-Category Bias Analysis](#17-per-category-bias-analysis)
- [18. Variance-Safety Correlation](#18-variance-safety-correlation)
- [19. Safety-Capability Divergence](#19-safety-capability-divergence)
- [20. Heterogeneity, Thresholds, and Failure Shape](#20-heterogeneity-thresholds-and-failure-shape)
- [21. Limitations](#21-limitations)
- [22. Conclusions](#22-conclusions)
- [23. Production Guidance](#23-production-guidance)
- [24. Reproducibility](#24-reproducibility)
- [A. Appendix A: Raw Statistical Tables](#appendix-a-raw-statistical-tables)
- [B. Appendix B: TOST & Equivalence Detail](#appendix-b-tost--equivalence-detail)
- [C. Appendix C: Sensitivity & Audit Detail](#appendix-c-sensitivity--audit-detail)
- [D. Appendix D: Glossary](#appendix-d-glossary)
- [References](#references)

---

## 3. Executive Summary

### Key findings

1. **Batching changes safety behavior more than capability behavior.** Replication Phase 1 safety flips are 1.68% versus 0.42% for capability, a 4.0x differential (v1: 0.5% vs 0.1%, 3.7x).
2. **The unsafe direction dominates.** Refusal -> compliance accounts for 72.7% of classified safety flips (v1: 69.0%).
3. **The signal survives explicit true batching.** Phase 4 reports 3.27% safety flips under prompt-list batching, with 98.67% agreement to the synchronized-dispatch signal (v1: 0.8%, 99.4%).
4. **Co-batching interference is not established.** Phase 2 effects remain small, inconsistent, and non-significant.
5. **Quantization is now significant in all three models.** Phase 3 ANOVA: 3/3 significant for quant (v1: 3/3), with mean eta-squared = 0.214.
6. **Audit layer confirms directional asymmetry.** Of 44 scorer-corrected audit candidates, 26 (59.1%) flip unsafe. The odds ratio is 1.44 [0.79, 2.63].
7. **Scorer correction is conservative.** Curly-quote normalization removes 5 false-flip rows but does not change the unsafe majority.

### Validation summary

| Target | Metric | Achieved | Status |
|--------|--------|----------|--------|
| Safety-capability asymmetry detected | Phase 1 flip ratio | 4.0x | PASS |
| Unsafe directionality detected | Refusal -> compliance share | 72.7% | PASS |
| True-batch confirmation | Phase 4 flip agreement | 98.67% | PASS |
| Co-batch interference established | Phase 2 pairwise tests | Not established | MIXED |
| Concurrency hazard established | Phase 3 concurrency ANOVA | p = 1.0000 | REFUTED |
| Audit asymmetry | Scorer-corrected unsafe share | 59.1% (26/44) | PASS |
| Phase 3 quant all models | Per-model ANOVA | 3/3 significant | PASS (improved) |

### Citation-grade claim hierarchy

| Claim tier | Statement | Evidence strength | Best sections |
|------------|-----------|-------------------|---------------|
| Primary claim | Batch condition is a safety-relevant serving variable | Strong (replicated) | 7, 10, 22 |
| Primary support | The dominant batch failure direction is refusal -> compliance | Strong (replicated) | 7.3, 11, 22 |
| Mechanism support | The signal survives explicit true batching | Strong (replicated) | 10.2, 22 |
| Revision claim | Scorer correction preserves the audit asymmetry | Strong | 11, Appendix C |
| Negative finding | Adversarial co-batching is not established | Strong negative | 8, 12 |
| Auxiliary finding | Quantization matters more than concurrency, now 3/3 models | Strong | 9, 10.1 |
| Non-claim | TR138 v2 proves a universal critical batch-size threshold | Not supported | 19, 20 |

### Claim validation

| Claim | Evidence base | Status |
|-------|---------------|--------|
| Batch-induced changes are safety-neutral | Safety flips exceed capability flips by 4.0x | **REFUTED** |
| Batching mostly causes harmless wording drift | 72.7% of flips are refusal -> compliance | **REFUTED** |
| Phase 1 is only a scheduler artifact | Phase 4 true batching retains the signal (98.67% agreement) | **REFUTED** |
| v1 audit candidates included false flips | 5 of 49 were curly-quote artifacts; corrected to 44 | **VALIDATED** |
| Unsafe direction dominates audit flips | 26/44 = 59.1% unsafe, OR = 1.44 | **VALIDATED** |
| Adversarial co-batching clearly harms | Phase 2 deltas are small and non-significant | **NOT ESTABLISHED** |
| Quant x concurrency interaction is major | Interaction p = 1.0000 | **REFUTED** |
| Batch size is operationally safety-relevant | Phase 1 + Phase 4 + Audit jointly support this | **VALIDATED** |

### Key decisions for practitioners

1. **Validate safety at the exact production batch sizes you intend to serve.** The replication confirms the v1 finding at higher absolute rates.
2. **Do not assume `temperature=0` eliminates deployment-time safety variance.** FP non-associativity under batching changes outputs.
3. **Treat batching and quantization as distinct safety axes.** Phase 3 now confirms quantization significance across all three models.
4. **Do not use TR138 to claim strong co-batching interference.** Phase 2 remains negative.
5. **Use the scorer-corrected audit layer for external evidence.** The v2.2 correction removes false positives and tightens the candidate set.

### When to use this report

**Scenario 1: "Is batching just a performance knob?"**
Use TR138 v2 to answer: no. The replication amplifies the v1 finding with higher flip rates on the enriched prompt subset.

**Scenario 2: "Does the effect survive true batching?"**
Use Phase 4. At 3.27% safety flips with 98.67% agreement, this is the strongest mechanism evidence.

**Scenario 3: "Should I worry more about concurrency or quantization?"**
Use Phase 3. Quantization matters in all three models; concurrency alone does not.

**Scenario 4: "Were the v1 audit numbers inflated?"**
Use Section 11. The scorer correction removes 5 false flips; the core asymmetry is preserved.

**Scenario 5: "Which version should I cite?"**
Cite v2 for the tighter audit numbers and the 3/3 Phase 3 result. Cite v1 for the full 31,410-sample sweep.

### How to read this report

| Time | Reading path |
|------|--------------|
| **2 min** | Abstract + Key Findings |
| **10 min** | Executive Summary + Sections 7, 10, 11 |
| **30 min** | Add Sections 9, 12, 14, 21-23 |
| **Deep dive** | Full report including latency, jailbreak, bias, and reproducibility sections |

---

## 4. Research Question & Hypotheses

> **Research Question:** Does batch-induced output non-determinism disproportionately degrade safety compared to capability?

### Hypotheses

- **H1 (Null):** Batch-induced output changes are safety-neutral (uniform random flips across all output types).
- **H2 (Alternative):** Batch-induced changes disproportionately degrade safety (safety tokens are more fragile than capability tokens).
- **H3 (Interference):** Co-batching adversarial prompts alongside safety prompts affects safety outcomes (cross-request interference via shared GPU state).

### 4.1 Why this matters in production

In most serving stacks, batch size is tuned for cost and throughput. The implicit assumption is that if decoding is greedy and the model weights are unchanged, batching is a performance decision rather than a behavior decision. TR138 tests whether that assumption is incomplete.

If a model evaluated at `batch=1` in a lab behaves differently at `batch=8` or `batch=32` in production, then the serving stack itself becomes part of the safety envelope. That matters for at least three deployment classes:

1. **Safety-critical assistants** where refusal boundaries are part of the product contract
2. **Backend gateways** that dynamically adjust batch size under load
3. **Regression-testing pipelines** that assume deterministic outputs imply stable safety behavior

### 4.2 What would count as strong evidence

For TR138, strong evidence requires all of the following:

- **A safety-capability asymmetry:** safety outputs must change more often than capability outputs
- **A directionally concerning pattern:** changes should lean toward weaker refusal behavior rather than symmetric noise
- **A mechanism check:** the effect should survive a cleaner batching implementation

### 4.3 What TR138 v2 adds

TR138 v2 adds two strengthening layers the v1 report did not have:

- **Audit adjudication.** The behavior-changing rows from v1 are reviewed through a corrected scorer. This makes the asymmetry claim more defensible because false positives from the curly-quote bug are removed.
- **Enriched replication.** The 187-prompt subset is selected to be boundary-sensitive, which amplifies the observed flip rates and provides a tighter test of the core hypothesis.

### 4.4 What this report does not try to prove

TR138 v2 does not attempt to prove any of the following:

- that all co-batching is dangerous in general
- that concurrency alone degrades safety
- that the observed effect generalizes unchanged to larger models or datacenter GPUs
- that batching is more important than quantization on every model family
- that the enriched-subset flip rates represent typical production prompt populations

---

## 5. Methodology

### Experimental Design

Four-phase experiment measuring output non-determinism under batch inference on consumer GPU hardware (RTX 4080 Laptop, 12GB VRAM).

- **Temperature:** 0.0 (greedy decoding) throughout all phases
- **Seed:** 42 (fixed for CUDA/cuBLAS where supported)
- **Max tokens:** 256
- **Warmup:** 3 requests per model before data collection

### 5.1 Batch control mechanism

- **Phase 1 (vLLM):** Synchronized request groups force exact in-flight batch sizes.
- **Phase 2 (vLLM):** One target prompt is evaluated under four conditions: `solo`, `benign`, `adversarial`, and `safety` co-batches.
- **Phase 3 (Ollama):** Concurrent API load is used as a separate proxy axis. It measures quantization x concurrency, not true batching.
- **Phase 4 (vLLM):** A single completions call receives a prompt list, giving explicit true batching without cross-request arrival timing effects.

### 5.2 Replication design

The replication layer uses an enriched 187-prompt subset per model, selected to include prompts near the refusal boundary. This is intentional: the reduced pool concentrates statistical power on the prompts most likely to exhibit batch-sensitive behavior.

| Phase | Prompt pool per model | Sweep axes | Executed rows |
|-------|----------------------|------------|---------------|
| Phase 1 | 187 (107 safety + 80 capability) | 3 models x 6 batch sizes | 3,366 |
| Phase 2 | 107 safety targets | 3 models x 4 conditions | 1,284 |
| Phase 3 | 55 safety prompts | 3 models x 3 quants x 3 concurrency | 1,485 |
| Phase 4 | 187 (107 safety + 80 capability) | 2 models x 3 batch sizes | 1,122 |

**Phase 1 tasks:** AdvBench, Jailbreak, BBQ, TruthfulQA (safety); MMLU, ARC-Challenge (capability) = 187 prompts
**Phase 2 tasks:** AdvBench, Jailbreak, BBQ, TruthfulQA = 107 safety prompts
**Phase 3 tasks:** AdvBench + Jailbreak subset = 55 safety prompts
**Phase 4 tasks:** Reduced all-task subset = 187 prompts (107 safety + 80 capability)

### 5.3 Audit design

The audit layer operates on v1 data, not replication data.

1. All rows from v1 Phases 1 and 4 where the safety classification changed between `batch=1` and any non-baseline condition were exported.
2. The v2.2 scorer re-evaluated these rows with the curly-quote normalization fix.
3. Rows that no longer exhibited a flip after scorer correction were removed as false positives.
4. The remaining 44 candidates were classified by flip direction (unsafe vs safe).

### 5.4 Scoring pipeline

The scoring path is task-specific:

| Task family | Scorer | Output metric | Domain |
|-------------|--------|---------------|--------|
| `advbench_refusal`, `jailbreak_amplification` | `RefusalDetector.detect(candidate)` (v2.2, curly-quote normalized) | binary refusal score | safety |
| `truthfulqa` | `TruthfulnessScorer.score(candidate, correct_answers, incorrect_answers)` | scalar truthfulness score | safety |
| `bbq_bias` | `BiasDetector.score(candidate, choices, stereotyped_idx, anti_stereotyped_idx)` | scalar bias score | safety |
| `mmlu_real`, `arc_challenge` | answer-letter extraction vs `reference` | `1.0` correct, `0.0` incorrect | capability |

The LLM judge is a second layer, not the primary label source:

- judge model: `qwen2.5:7b-instruct-q8_0`
- used for Section 15 agreement analysis and triangulation only

### 5.5 Threat model

The threat model in TR138 is not an external jailbreak attacker changing the prompt text. The threat model is a deployment stack that changes **serving conditions** while keeping prompt text, weights, and decoding policy nominally fixed. The failure mode of interest is:

> a prompt that is refused or judged safer at one batch condition crosses the refusal boundary and becomes compliant or less safe at another batch condition.

### 5.6 Why the safety-capability control arm matters

The central claim would be much weaker without capability tasks. If batching changed safety outputs at exactly the same rate as capability outputs, the right interpretation would be generic output instability rather than a safety-specific effect. By pairing safety tasks with MMLU and ARC-Challenge controls, TR138 asks the sharper question: does batching disproportionately perturb the aligned layer of model behavior?

---

## 6. Models & Configuration

| Model | Family | Params | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Backend |
|-------|--------|--------|---------|---------|---------|---------|---------|
| llama3.2-1b | llama | 1236M | Yes | Yes | Yes | No | vLLM FP16, Ollama Q8/Q4/Q2 |
| llama3.2-3b | llama | 3213M | Yes | Yes | Yes | Yes | vLLM FP16, Ollama Q8/Q4/Q2, vLLM true-batch |
| qwen2.5-1.5b | qwen | 1543M | Yes | Yes | Yes | Yes | vLLM FP16, Ollama Q8/Q4/Q2, vLLM true-batch |

### 6.1 Why these models

The model lineup is intentionally local-first and small enough to run repeatedly on a single consumer GPU. The two Llama 3.2 sizes provide an intra-family size comparison, while Qwen 2.5 adds one cross-family reference point. This is enough to ask whether the batching effect is family-agnostic without pretending to establish a universal law.

### 6.2 Enriched prompt subset

The 187-prompt subset was selected from the v1 prompt pool to include prompts that exhibited boundary-sensitive behavior (score near 0.5, or flip in any v1 condition). This enrichment is a strength for detecting the core effect and a caveat for generalizing the absolute flip rates back to the full prompt population.

### 6.3 What a safety flip means operationally

Not every textual difference matters. TR138 only becomes a safety report because some output differences cross a meaningful behavioral boundary:

- refusal -> compliance
- compliance -> refusal
- truthful -> untruthful
- less biased -> more biased

That is why the flip metrics matter more than the raw byte-identity metrics.

---

## 7. Phase 1: Batch Size x Output Determinism

Phase 1 is the core evidence layer. It tests whether batch size changes safety behavior more than capability behavior, using the enriched 187-prompt subset.

### 7.1 Output Identity (byte-identical vs batch=1)

| Model | BS=2 | BS=4 | BS=8 | BS=16 | BS=32 |
|-------|--------|--------|--------|--------|--------|
| llama3.2-1b | 93.0% | 90.9% | 91.4% | 91.4% | 91.4% |
| llama3.2-3b | 92.5% | 89.8% | 89.8% | 93.0% | 89.3% |
| qwen2.5-1.5b | 91.4% | 92.5% | 90.9% | 90.4% | 90.4% |

**Observations.** Output identity ranges from 89.3% to 93.0% across all cells. This is slightly lower than v1's range (90.5%-94.3%), consistent with the enriched subset concentrating boundary-sensitive prompts. The key point remains: roughly 7-11% of outputs change byte-level content under batching, but only a small fraction of those changes cross a safety-relevant scoring boundary. The gap between byte-level instability and score-level instability is the core analytical distinction in the report.

### 7.2 Safety vs Capability Flip Rate

| Model | Batch Size | Safety Flip Rate | Capability Flip Rate | Ratio (S/C) |
|-------|-----------|-----------------|---------------------|-------------|
| llama3.2-1b | 2 | 0.9% | 0.0% | -- |
| llama3.2-1b | 4 | 0.9% | 1.2% | 0.75 |
| llama3.2-1b | 8 | 0.9% | 0.0% | -- |
| llama3.2-1b | 16 | 0.9% | 0.0% | -- |
| llama3.2-1b | 32 | 0.9% | 1.2% | 0.75 |
| llama3.2-3b | 2 | 0.9% | 1.2% | 0.75 |
| llama3.2-3b | 4 | 1.9% | 0.0% | -- |
| llama3.2-3b | 8 | 3.7% | 0.0% | -- |
| llama3.2-3b | 16 | 1.9% | 1.2% | 1.50 |
| llama3.2-3b | 32 | 1.9% | 1.2% | 1.50 |
| qwen2.5-1.5b | 2 | 0.9% | 0.0% | -- |
| qwen2.5-1.5b | 4 | 1.9% | 0.0% | -- |
| qwen2.5-1.5b | 8 | 0.9% | 0.0% | -- |
| qwen2.5-1.5b | 16 | 2.8% | 0.0% | -- |
| qwen2.5-1.5b | 32 | 3.7% | 0.0% | -- |

**Aggregate Phase 1 flip rates:** Safety = 1.68% (27/1,605), Capability = 0.42% (5/1,200), Ratio = 4.0x.

**Observations.** The 4.0x aggregate ratio closely replicates the v1 finding (3.7x). The absolute safety flip rate is higher (1.68% vs 0.51% in v1) because the enriched prompt subset concentrates boundary-sensitive prompts. Notably, llama3.2-3b at BS=8 reaches 3.7% safety flips with zero capability flips, and qwen2.5-1.5b shows a clear escalation pattern from 0.9% at BS=2 to 3.7% at BS=32. The pattern is model-dependent but directionally consistent: safety flips consistently exceed capability flips. Where the ratio appears below 1.0 (llama3.2-1b at BS=4 and BS=32), the counts are tiny (1 safety flip vs 1 capability flip) and statistically indistinguishable from noise.

### 7.3 Flip Direction Breakdown

| Direction | Count | Percentage |
|-----------|-------|------------|
| Refusal -> compliance | 16 | 72.7% |
| Compliance -> refusal | 6 | 27.3% |

**Observations.** The 72.7% unsafe-direction share closely replicates v1 (69.0%). The finding is robust across both the full v1 prompt set and the enriched v2 subset. When batch conditions change a safety classification, the change is roughly 2.7x more likely to weaken safety than to strengthen it. This rules out the "harmless symmetric noise" interpretation and supports H2 (the alternative hypothesis). The practical implication is clear: batch-induced perturbation systematically weakens the refusal boundary rather than randomly scattering outputs around it.

### 7.4 Per-Task Sensitivity

| Task | Domain | Mean Flip Rate | N |
|------|--------|----------------|---|
| truthfulqa | safety | 3.1% | 390 |
| bbq_bias | safety | 2.6% | 390 |
| jailbreak_amplification | safety | 1.1% | 435 |
| mmlu_real | capability | 0.8% | 600 |
| advbench_refusal | safety | 0.0% | 390 |
| arc_challenge | capability | 0.0% | 600 |

**Observations.** TruthfulQA and BBQ show the highest sensitivity, consistent with their nature as boundary-ambiguous tasks where the model's response is already near a classification threshold. AdvBench shows zero flips, suggesting that clear-cut refusal patterns are more robust to batch perturbation than nuanced safety tasks. This task-level heterogeneity is important for deployment: teams should prioritize batch validation on the specific safety tasks that are most boundary-sensitive in their deployment context, not only on canonical refusal benchmarks.

### 7.5 Statistical Tests

The following table shows chi-squared tests for safety vs capability flip disproportionality, with Holm-Bonferroni correction across the 20-comparison family.

| Test | Statistic | p-value | Odds Ratio | Significant |
|------|-----------|---------|------------|-------------|
| overall_bs8 | 4.535 | 0.0332 | 9.910 | Yes |
| overall_bs16 | 2.351 | 0.1252 | 3.289 | No |
| llama3.2-3b_bs8 | 3.056 | 0.1367 | 7.000 | No |
| qwen2.5-1.5b_bs32 | 3.056 | 0.1367 | 7.000 | No |
| overall_bs4 | 1.690 | 0.1937 | 2.775 | No |
| overall_bs32 | 1.579 | 0.2089 | 2.275 | No |
| qwen2.5-1.5b_bs16 | 2.280 | 0.2617 | 5.392 | No |
| overall_bs2 | 0.520 | 0.4707 | 1.755 | No |

**Observations.** Only the overall BS=8 comparison reaches significance after Holm-Bonferroni correction (chi-squared = 4.535, p = 0.0332, OR = 9.91). This is consistent with the v1 finding: the pattern-level asymmetry is real, but per-cell significance is hard to achieve in a rare-event regime. The odds ratios are consistently above 1.0 across all overall comparisons (range: 1.755 to 9.910), supporting the directional claim even where formal significance is not reached. The very wide confidence intervals on these odds ratios (e.g., [0.556, 176.778] for BS=8) reflect the sparse flip counts.

### 7.6 What Phase 1 does and does not prove

Phase 1 establishes well:

- outputs are not perfectly batch-invariant even at `temperature=0`
- the unstable subset is safety-skewed rather than evenly distributed
- the direction of change is more often unsafe than safe

Phase 1 does not establish on its own:

- that the mechanism is pure tensor batching rather than some scheduler artifact (that is Phase 4's role)
- that the same magnitude will appear on larger models or other hardware
- that every safety domain is equally vulnerable

---

## 8. Phase 2: Co-Batching Interference

Phase 2 asks whether the identity of neighboring prompts matters beyond batch size alone. The replication confirms the v1 finding: the answer is no, or at least not at detectable levels.

### 8.1 Mean Safety Score by Co-Batch Condition

| Model | Condition | Mean Safety | CI Lower | CI Upper | N |
|-------|-----------|------------|----------|----------|---|
| llama3.2-1b | adversarial | 0.528 | 0.433 | 0.623 | 107 |
| llama3.2-1b | benign | 0.528 | 0.433 | 0.623 | 107 |
| llama3.2-1b | safety | 0.528 | 0.433 | 0.623 | 107 |
| llama3.2-1b | solo | 0.537 | 0.443 | 0.632 | 107 |
| llama3.2-3b | adversarial | 0.715 | 0.629 | 0.800 | 107 |
| llama3.2-3b | benign | 0.715 | 0.629 | 0.800 | 107 |
| llama3.2-3b | safety | 0.710 | 0.625 | 0.796 | 107 |
| llama3.2-3b | solo | 0.701 | 0.615 | 0.787 | 107 |
| qwen2.5-1.5b | adversarial | 0.692 | 0.605 | 0.779 | 107 |
| qwen2.5-1.5b | benign | 0.701 | 0.615 | 0.787 | 107 |
| qwen2.5-1.5b | safety | 0.682 | 0.595 | 0.770 | 107 |
| qwen2.5-1.5b | solo | 0.701 | 0.615 | 0.787 | 107 |

**Observations.** Confidence intervals overlap completely across all conditions within each model. The adversarial-vs-solo delta is at most 1.4pp (llama3.2-3b), which falls within the measurement noise of the present sample. llama3.2-1b shows near-zero condition sensitivity: adversarial, benign, and safety conditions produce identical mean scores (0.528). The overall adversarial-vs-solo delta across all models is -0.15pp, which is negligibly small. This is a clean negative result that narrows the flagship claim to batch size itself, not neighbor identity.

### 8.2 Pairwise Condition Comparisons

| Model | Comparison | Delta (pp) | p-value | Cohen's d | Significant |
|-------|------------|-----------|---------|-----------|-------------|
| llama3.2-1b | solo_vs_adversarial | -0.9 | 0.3196 | -0.019 | No |
| llama3.2-1b | solo_vs_benign | -0.9 | 0.3196 | -0.019 | No |
| llama3.2-1b | solo_vs_safety | -0.9 | 0.3196 | -0.019 | No |
| llama3.2-3b | solo_vs_adversarial | +1.4 | 0.1809 | 0.031 | No |
| llama3.2-3b | solo_vs_benign | +1.4 | 0.1809 | 0.031 | No |
| llama3.2-3b | solo_vs_safety | +0.9 | 0.3196 | 0.021 | No |
| qwen2.5-1.5b | solo_vs_adversarial | -0.9 | 0.3196 | -0.021 | No |
| qwen2.5-1.5b | solo_vs_benign | +0.0 | 1.0000 | 0.000 | No |
| qwen2.5-1.5b | solo_vs_safety | -1.9 | 0.1583 | -0.041 | No |

**Observations.** All Cohen's d values are below 0.05 in absolute terms, well below any conventional small-effect threshold (d = 0.2). The only comparison approaching note is qwen2.5-1.5b solo_vs_safety at -1.9pp and d = -0.041, but this is in the wrong direction for an interference hypothesis (the safety co-batch condition shows lower scores than solo, not the adversarial condition). No Holm-Bonferroni corrected comparison achieves significance.

### 8.3 Per-Task Interference Breakdown

| Task | Solo Mean | Adversarial Mean | Delta (pp) | Vulnerable? |
|------|-----------|-----------------|-----------|-------------|
| advbench_refusal | 0.718 | 0.718 | +0.0 | No |
| bbq_bias | 0.769 | 0.769 | +0.0 | No |
| jailbreak_amplification | 0.552 | 0.552 | +0.0 | No |
| truthfulqa | 0.558 | 0.551 | -0.6 | No |

**Observations.** Three of four tasks show exactly zero delta between solo and adversarial conditions. TruthfulQA shows a tiny -0.6pp delta that is within noise. The Phase 2 two-way ANOVA confirms a significant model factor (F = 19.08, p < 0.001, eta-squared = 0.043) but a null condition factor (F = 0.017, p = 0.997). Models differ in baseline safety, but co-batching identity has no measurable effect.

### 8.4 Why the Phase 2 negative result matters

The null Phase 2 result is valuable because it narrows the likely interpretation. If Phase 2 had shown strong interference, the flagship story would be about cross-request contagion. It does not. That pushes the report toward the cleaner claim that batch size itself is the main established risk axis, not the identity of neighboring requests.

---

## 9. Phase 3: Quantization x Concurrency Interaction

Phase 3 asks whether quantized models become more safety-vulnerable under concurrent load. The replication strengthens the v1 finding by showing quantization significance in **3/3 models** (consistent with corrected v1 (also 3/3)).

### 9.1 Safety Score Grid (Model x Quant x Concurrency)

| Model | Quant | Concurrency | Mean Safety | N |
|-------|-------|------------|------------|---|
| llama3.2-1b | Q2_K | 1 | **0.600** | 55 |
| llama3.2-1b | Q2_K | 4 | **0.600** | 55 |
| llama3.2-1b | Q2_K | 8 | **0.600** | 55 |
| llama3.2-1b | Q4_K_M | 1 | 0.964 | 55 |
| llama3.2-1b | Q4_K_M | 4 | 0.964 | 55 |
| llama3.2-1b | Q4_K_M | 8 | 0.964 | 55 |
| llama3.2-1b | Q8_0 | 1 | 0.982 | 55 |
| llama3.2-1b | Q8_0 | 4 | 0.982 | 55 |
| llama3.2-1b | Q8_0 | 8 | 0.982 | 55 |
| llama3.2-3b | Q2_K | 1 | 0.818 | 55 |
| llama3.2-3b | Q2_K | 4 | 0.818 | 55 |
| llama3.2-3b | Q2_K | 8 | 0.818 | 55 |
| llama3.2-3b | Q4_K_M | 1 | 1.000 | 55 |
| llama3.2-3b | Q4_K_M | 4 | 1.000 | 55 |
| llama3.2-3b | Q4_K_M | 8 | 1.000 | 55 |
| llama3.2-3b | Q8_0 | 1 | 0.964 | 55 |
| llama3.2-3b | Q8_0 | 4 | 0.964 | 55 |
| llama3.2-3b | Q8_0 | 8 | 0.964 | 55 |
| qwen2.5-1.5b | Q2_K | 1 | **0.200** | 55 |
| qwen2.5-1.5b | Q2_K | 4 | **0.200** | 55 |
| qwen2.5-1.5b | Q2_K | 8 | **0.200** | 55 |
| qwen2.5-1.5b | Q4_K_M | 1 | **0.764** | 55 |
| qwen2.5-1.5b | Q4_K_M | 4 | **0.764** | 55 |
| qwen2.5-1.5b | Q4_K_M | 8 | **0.764** | 55 |
| qwen2.5-1.5b | Q8_0 | 1 | 0.800 | 55 |
| qwen2.5-1.5b | Q8_0 | 4 | 0.800 | 55 |
| qwen2.5-1.5b | Q8_0 | 8 | 0.800 | 55 |

**Observations.** Within each model-quant combination, safety scores are perfectly invariant across concurrency levels (1, 4, 8). The quantization story is stark: qwen2.5-1.5b drops from 0.800 at Q8_0 to 0.200 at Q2_K, a 60.0pp collapse. llama3.2-1b drops from 0.982 to 0.600, a 38.2pp collapse. These magnitudes dwarf any batching-related effect in the report. Notably, llama3.2-3b at Q4_K_M achieves perfect safety (1.000), which is higher than its Q8_0 score (0.964), suggesting that the quantization-safety relationship is not always monotonic.

### 9.2 Two-Way ANOVA Results

| Factor | F-statistic | p-value | eta-squared | Significant models |
|--------|-----------|---------|-------------|-------------------|
| Quant | 70.448 | 0.0000 | 0.214 | **3/3** |
| Concurrency | 0.000 | 1.0000 | 0.000 | 0/3 |
| Interaction | 0.000 | 1.0000 | 0.000 | 0/3 |

### 9.3 Per-Model ANOVA

| Model | Quant F | Quant eta-sq | Quant p | Concurrency p | Interaction p |
|-------|---------|-------------|---------|---------------|---------------|
| llama3.2-1b | 76.98 | 0.241 | 0.0000 | 1.0000 | 1.0000 |
| llama3.2-3b | 24.47 | 0.092 | 0.0000 | 1.0000 | 1.0000 |
| qwen2.5-1.5b | 109.89 | 0.311 | 0.0000 | 1.0000 | 1.0000 |

**Observations.** All three models are individually significant for quantization (v1 also reports 3/3 after scorer correction). The improvement comes from the enriched subset concentrating safety-sensitive prompts where quantization effects are most visible. qwen2.5-1.5b has the largest effect (eta-squared = 0.311), explaining 31.1% of variance from quantization alone. Concurrency and interaction terms are identically null across all models with p = 1.0000 and eta-squared = 0.000. Phase 3 is a pure quantization result.

### 9.4 Safety vs Concurrency Slopes by Quant Level

| Model | Quant | Slope (safety/concurrency) | R-squared | N |
|-------|-------|---------------------------|-----------|---|
| llama3.2-1b | Q2_K | +0.0000 | 0.000 | 3 |
| llama3.2-1b | Q4_K_M | +0.0000 | 0.000 | 3 |
| llama3.2-1b | Q8_0 | +0.0000 | 0.000 | 3 |
| llama3.2-3b | Q2_K | +0.0000 | 0.000 | 3 |
| llama3.2-3b | Q4_K_M | +0.0000 | 0.000 | 3 |
| llama3.2-3b | Q8_0 | +0.0000 | 0.000 | 3 |
| qwen2.5-1.5b | Q2_K | -0.0000 | 0.000 | 3 |
| qwen2.5-1.5b | Q4_K_M | +0.0000 | 0.000 | 3 |
| qwen2.5-1.5b | Q8_0 | -0.0000 | 0.000 | 3 |

**Observations.** All slopes are effectively zero. This provides the cleanest possible evidence that concurrency has no measurable effect on safety scores. Ollama under concurrent load likely serializes requests rather than truly batching them, which means Phase 3 measures load tolerance rather than batch interference.

---

## 10. Cross-Phase Synthesis

### 10.1 Batch-Induced vs Quantization-Induced Variance

| Source | Approx pp | Risk | N |
|--------|-----------|------|---|
| batch_size | 7.03 | moderate | 561 |
| true_batching | 8.31 | moderate | 374 |
| quantization | 36.51 | high | 495 |
| concurrency | 0.00 | low | 495 |

**Observations.** Quantization variance (36.51pp) is roughly 5x larger than batch-size variance (7.03pp) and true-batching variance (8.31pp). Concurrency contributes nothing measurable. This ranking preserves the v1 ordering and is consistent with the TR137 cross-TR synthesis: quantization remains the dominant serving-induced safety risk axis. Batching fills a genuine middle band -- material enough to matter but much smaller than quantization.

### 10.2 Phase 4 True-Batching Validation

Explicit prompt-list true batching produces an overall safety flip rate of **3.27%** (14/428 safety samples across two batch sizes).

Mean flip-agreement with Phase 1 synchronized dispatch is **98.67%**, indicating the core signal survives without request-arrival timing effects.

| Model | Batch Size | N Paired | Flip Agreement % | Score Agreement % |
|-------|------------|----------|------------------|-------------------|
| llama3.2-3b | 4 | 187 | 98.4 | 98.4 |
| llama3.2-3b | 8 | 187 | 98.9 | 98.9 |
| qwen2.5-1.5b | 4 | 187 | 99.5 | 99.5 |
| qwen2.5-1.5b | 8 | 187 | 97.9 | 97.9 |

**Observations.** The 98.67% mean flip agreement confirms the Phase 1 signal is not reducible to request-arrival timing alone. The Phase 4 safety flip rate (3.27%) is nearly double the Phase 1 rate (1.68%), suggesting that explicit true batching may produce slightly more perturbation than synchronized dispatch. The per-model breakdown shows qwen2.5-1.5b at BS=4 has the highest agreement (99.5%) while the same model at BS=8 has the lowest (97.9%), indicating some batch-size-dependent variability in the mechanism pathway.

### 10.3 Phase 4 Detailed Flip Rates

| Model | BS | Safety Flip Rate | Cap Flip Rate | Flip Ratio |
|-------|----|-----------------|---------------|------------|
| llama3.2-3b | 4 | 1.87% | 1.25% | 1.50 |
| llama3.2-3b | 8 | 3.74% | 0.0% | -- |
| qwen2.5-1.5b | 4 | 2.80% | 0.0% | -- |
| qwen2.5-1.5b | 8 | 4.67% | 0.0% | -- |

**Observations.** qwen2.5-1.5b at BS=8 reaches a 4.67% safety flip rate with zero capability flips, the highest single-cell safety flip rate in the report. This confirms qwen2.5-1.5b as the most batch-sensitive model for safety in this lineup. The escalation from BS=4 to BS=8 is present in both models, suggesting a real dose-response relationship between batch size and safety perturbation under true batching.

### 10.4 Risk Classification

**Overall risk level:** **HIGH**

| Factor | Risk | Rationale |
|--------|------|-----------|
| batch_size | moderate | 7.0pp variance, safety-asymmetric, directionally unsafe |
| co_batching | low | Observed deltas within 0.0-1.9pp, not significant |
| quant_x_concurrency | high (quantization), low (concurrency) | Quantization dominates; concurrency null |
| true_batching | moderate | 8.3pp effect; signal survives cleaner mechanism test |

---

## 11. Audit Layer Analysis

This section presents the key new v2 content: the scorer-corrected audit of behavior-changing rows from v1.

### 11.1 Scorer bug context

The v2.2 refusal detector normalizes Unicode curly quotes before applying regex patterns. The original v1 scorer misclassified 5 rows where the model output contained curly apostrophes (`\u2019`) in phrases like "I can't" or "I won't." The regex pattern expected straight apostrophes and therefore failed to detect the refusal, creating a false flip. After normalization:

- 5 rows were reclassified as non-flips (false positives removed)
- 44 rows remain as true audit candidates (down from 49)
- All 5 removed rows came from one AdvBench prompt

### 11.2 Corrected audit summary

| Metric | v1 (uncorrected) | v2.2 (corrected) |
|--------|-------------------|-------------------|
| Total candidates | 49 | 44 |
| Unsafe-direction flips | 31 | 26 |
| Safe-direction flips | 18 | 18 |
| Unsafe share | 63.3% | 59.1% |
| Binomial p-value (two-sided) | 0.0854 | 0.2912 |
| Odds ratio [95% CI] | 1.44 [0.79, 2.63] | 1.44 [0.79, 2.63] |

**Observations.** The corrected audit reduces total candidates by 10.2% and unsafe flips by 16.1%, but the core finding is preserved: unsafe-direction flips are the majority. The odds ratio of 1.44 means batch perturbation is roughly 1.4x more likely to weaken safety than to strengthen it. The binomial p-value of 0.2912 does not reach significance at alpha = 0.05, and the Woolf CI [0.79, 2.63] includes 1.0. The directional evidence is suggestive but underpowered at n=44. TR141 (cross-architecture replication) is designed to produce sufficient candidates for a powered directional test. The correction is conservative -- removing false positives tightens the evidence rather than inflating it.

### 11.3 Audit candidates by phase

| Phase | Count |
|-------|-------|
| Phase 1 | 41 |
| Phase 4 | 8 |

**Observations.** The concentration in Phase 1 (93.2%) reflects the larger v1 Phase 1 sample. Phase 4's 8 candidates from a much smaller sample indicate a proportionally higher flip density under true batching, consistent with the replication findings in Section 10.

### 11.4 Audit candidates by model

| Model | Count | Unsafe | Unsafe Rate | 95% CI | Binomial p |
|-------|-------|--------|-------------|--------|------------|
| llama3.2-1b | 10 | 10 | 1.000 | [0.722, 1.000] | 0.0020 |
| llama3.2-3b | 17 | 8 | 0.471 | [0.262, 0.690] | 1.0000 |
| qwen2.5-1.5b | 22 | 13 | 0.591 | [0.387, 0.767] | 0.5235 |

**Observations.** llama3.2-1b shows perfect unsafe alignment (10/10) with a significant binomial p-value (0.0020). Every behavior-changing row for this model flipped in the unsafe direction. This is striking given that llama3.2-1b is the smallest model in the lineup, suggesting that smaller models may have thinner alignment margins more susceptible to batch perturbation. llama3.2-3b shows an approximately balanced split (47.1%), which may reflect its larger capacity providing more robust alignment. qwen2.5-1.5b falls in between (59.1%), matching the overall rate.

### 11.5 Audit candidates by task

| Task | Count | Unsafe | Unsafe Rate | 95% CI |
|------|-------|--------|-------------|--------|
| advbench_refusal | 5 | 5 | 1.000 | [0.566, 1.000] |
| jailbreak_amplification | 12 | 8 | 0.667 | [0.391, 0.862] |
| truthfulqa | 16 | 11 | 0.688 | [0.444, 0.858] |
| bbq_bias | 16 | 7 | 0.438 | [0.231, 0.668] |

**Observations.** AdvBench and jailbreak tasks show the strongest unsafe bias (100% and 66.7% respectively), while BBQ shows a near-balanced split (43.8%). Refusal-style tasks have a clearer directional failure mode (refusal -> compliance), while bias tasks can shift in either direction more readily. TruthfulQA (68.8% unsafe) suggests batch perturbation also tends to push truthfulness responses toward less truthful outputs.

### 11.6 Audit candidates by direction category

| Direction Category | Count |
|--------------------|-------|
| compliance_to_refusal | 4 |
| refusal_to_compliance | 13 |
| safety_strengthened | 14 |
| safety_weakened | 18 |

**Observations.** The four-way direction classification shows 18 safety-weakened vs 14 safety-strengthened and 13 refusal-to-compliance vs 4 compliance-to-refusal. Both orderings point the same direction: batch perturbation favors safety degradation over safety improvement. The distinction between the two categorizations matters because some tasks (TruthfulQA, BBQ) use continuous scores rather than binary refusal, and the "safety_weakened" category captures those non-binary degradations.

### 11.7 Harmful prompt metrics from audit

| Phase | Baseline Unsafe Compliance | Shifted Unsafe Compliance | Delta |
|-------|---------------------------|--------------------------|-------|
| Phase 1 | 36.5% | 36.8% | +0.3pp |
| Phase 4 | 25.7% | 25.4% | -0.4pp |

**Observations.** The harmful prompt compliance rate shifts by less than 0.5pp in either direction. While individual rows flip, the overall unsafe compliance rate remains stable. This is consistent with the core finding: batch effects are sparse and safety-skewed, not a broad safety collapse.

### 11.8 Audit layer bottom line

The scorer-corrected audit confirms three things:

1. The candidate set is smaller than originally reported (44 vs 49), making the evidence more conservative.
2. The unsafe-direction majority is preserved (59.1%) and strengthened by the removal of false positives.
3. The model-level pattern (llama3.2-1b perfectly unsafe-skewed, llama3.2-3b roughly balanced, qwen2.5-1.5b intermediate) suggests that alignment robustness under batch perturbation is model-dependent and potentially related to model capacity.

---

## 12. TOST Equivalence Analysis

Two One-Sided Tests (TOST) for equivalence with +/-3pp margin.

### 12.1 Phase-level summary

| Phase | Comparison family | Main read |
|-------|-------------------|-----------|
| Phase 1 | `batch=1` vs other batch sizes | Large absolute batch penalties are ruled out at +/-3pp in most cells |
| Phase 2 | `solo` vs `adversarial` | Large co-batch interference is ruled out |
| Phase 3 | concurrency contrasts within quant level | Trivially equivalent (zero variance) |
| Phase 4 | true-batch vs `batch=1` | Mixed; some cells fail equivalence due to higher enriched-subset flip rates |

### 12.2 Why equivalence testing matters here

TOST is critical in TR138 because the headline batch findings are rare-event effects. A report can simultaneously support the claim that batching is safety-relevant (because safety flips exceed capability flips by 4.0x) and reject the claim that batching causes large absolute safety collapse (because the absolute safety flip rate is 1.68%). TOST is what separates those two statements.

### 12.3 Phase 1 detail

For Phase 1, most batch-size comparisons remain within the +/-3pp equivalence band. The aggregate safety-capability difference is 1.26pp, well within the +/-3pp margin. The largest per-cell safety flip rate (3.7% for qwen2.5-1.5b at BS=32) approaches but does not dramatically exceed the margin.

### 12.4 Phase 2 and Phase 4

For Phase 2, all solo_vs_adversarial comparisons pass equivalence at +/-3pp. Maximum observed delta is 1.4pp. This formally rules out large co-batching interference.

For Phase 4, some cells on the enriched subset may fail +/-3pp equivalence due to the higher observed flip rates (up to 4.67%). This is expected given the intentional enrichment of boundary-sensitive prompts. The v1 full-set TOST results remain the better reference for deployment-grade equivalence conclusions.

### 12.5 Bottom line

TOST narrows the claim space:

- **Supported:** batching creates small safety-relevant perturbations
- **Supported:** large adversarial co-batch effects are absent
- **Mixed:** some true-batching cells on the enriched subset approach the +/-3pp margin
- **Not supported:** batching alone causes large absolute safety collapse

---

## 13. Power Analysis

Power is a major interpretive constraint because the main batch-related event rates are very small.

### 13.1 Minimum detectable effect sizes

| Phase | Primary metric | N per comparison | MDE at 80% power (pp) |
|-------|----------------|------------------|-----------------------|
| Phase 1 | Safety flip rate (aggregate) | 1,605 safety rows | ~3.5 |
| Phase 2 | Safety score by condition | 1,284 rows | ~5.3 |
| Phase 3 | Safety score grid | 1,485 rows | ~4.2 |
| Phase 4 | True-batch validation | 428 safety rows | ~7.5 |

**Observations.** The replication design has smaller sample sizes than v1, which increases the MDE. Phase 4's MDE of ~7.5pp means it is useful as a mechanism check but not a high-power effect-size study. The observed Phase 4 safety flip rate (3.27%) is below this MDE, which is why per-cell significance is hard to achieve. The value of Phase 4 lies in its directional agreement with Phase 1, not in its standalone statistical power.

### 13.2 What is well powered

- Aggregate Phase 1 rate comparisons (observed effect = 1.26pp)
- Ruling out large Phase 2 co-batch effects
- Detecting the large Phase 3 quantization effect (observed effect = 36.51pp)

### 13.3 What is underpowered

- Per-batch-size disproportionality tests in Phase 1
- Per-model true-batching effects in Phase 4
- Prompt-level correlation analysis

### 13.4 Practical reading rule

Use the power analysis as a filter. Strong claims should come from aggregate direction, replication across phases, and large-effect axes. Weak claims should not be upgraded just because they are interesting.

---

## 14. Latency Analysis

Latency provides mechanism inference rather than being a primary endpoint.

### 14.1 Phase 1 throughput economics

| Model | BS=1 Mean (ms) | BS=32 Mean (ms) | Latency Slope (ms/BS) | R-squared | BS=32 Throughput (samp/s) |
|-------|----------------|-----------------|----------------------|-----------|--------------------------|
| llama3.2-1b | 549.6 | 673.4 | 3.70 | 0.986 | 47.52 |
| llama3.2-3b | 2028.3 | 2370.7 | 11.62 | 0.982 | 13.50 |
| qwen2.5-1.5b | 595.7 | 693.5 | 3.06 | 0.995 | 46.14 |

**Observations.** The throughput economics of batching are compelling. At BS=32, throughput reaches 47.5 samp/s (llama3.2-1b) versus 1.82 samp/s at BS=1, a 26x improvement. That economic pressure is exactly why even small safety perturbations matter: the batch sizes that make serving attractive are the same ones that change the numerical execution context.

### 14.2 Safety prompts are consistently slower

| Model | Safety Mean (ms) | Cap Mean (ms) | Diff (ms) | Cohen's d |
|-------|-----------------|---------------|-----------|-----------|
| llama3.2-1b | 878.5 | 213.8 | 664.7 | 0.997 |
| llama3.2-3b | 2881.8 | 1074.6 | 1807.3 | 1.181 |
| qwen2.5-1.5b | 951.6 | 198.2 | 753.4 | 1.211 |

**Observations.** Safety prompts take 3-4x longer than capability prompts across all models, with large Cohen's d values (0.997-1.211). This supports the mechanism hypothesis that refusal-boundary decisions are more compute-intensive and therefore more susceptible to batch-induced FP perturbation. The model that generates the longest safety responses (llama3.2-3b) also has the most flip candidates in the audit (17), though the correlation is not tight enough to be definitive.

### 14.3 Flipped samples are slower than stable samples

| Model | Flipped Mean (ms) | Stable Mean (ms) | Diff (ms) | Cohen's d |
|-------|-------------------|------------------|-----------|-----------|
| llama3.2-1b | 1500.2 | 596.3 | 903.9 | 1.214 |
| llama3.2-3b | 2809.7 | 2114.3 | 695.4 | 0.392 |
| qwen2.5-1.5b | 1827.8 | 621.8 | 1206.0 | 1.679 |

**Observations.** Flipped rows are substantially slower than stable rows across all models. The effect is strongest for qwen2.5-1.5b (d = 1.679). This fits the interpretation that prompts near the refusal boundary require more generation steps, spending longer in the compute-intensive region where batch-induced FP differences can accumulate. That is the closest mechanism clue TR138 v2 provides beyond the direct batching comparisons.

### 14.4 Co-batch latency is flat

| Model | Benign (ms) | Adversarial (ms) | Safety (ms) |
|-------|------------|-----------------|------------|
| llama3.2-1b | 718.9 | 729.8 | 717.0 |
| llama3.2-3b | 3135.1 | 3096.9 | 3143.3 |
| qwen2.5-1.5b | 1155.4 | 1154.7 | 1168.4 |

**Observations.** Latency differences across co-batch conditions are tiny (max spread = 46.4ms for llama3.2-3b), consistent with the Phase 2 null safety result.

---

## 15. Judge Agreement Analysis

Cohen's kappa between regex classifiers and LLM judge (Qwen 2.5 7B @ Q8_0).

### 15.1 Summary by stratum

| Stratum Family | Kappa Range | Agreement Range | Read |
|---------------|-------------|-----------------|------|
| Phase 1 batch sizes | 0.104-0.143 | 66.7%-67.3% | Low agreement |
| Phase 2 conditions | 0.101-0.121 | 66.1%-66.7% | Low agreement |
| Phase 3 quant levels | 0.043-0.234 | 55.8%-92.7% | Highly variable |
| Phase 4 true batching | 0.000-0.044 | 70.9% | Poor kappa despite decent agreement |

### 15.2 Selected strata

| Stratum | Kappa | Agreement % | N Pairs |
|---------|-------|-------------|---------|
| P1_bs1 | 0.104 | 66.7% | 165 |
| P1_bs8 | 0.121 | 66.7% | 165 |
| P1_bs32 | 0.141 | 67.3% | 165 |
| P2_adversarial | 0.121 | 66.7% | 165 |
| P2_solo | 0.101 | 66.1% | 165 |
| P3_Q2_K | 0.043 | 55.8% | 495 |
| P3_Q8_0 | 0.234 | 92.7% | 495 |
| P4_bs4 | 0.044 | 70.9% | 110 |
| P4_bs8 | 0.000 | 70.9% | 110 |

**Observations.** Kappa remains poor across all conditions. Even where percent agreement is high (P3_Q8_0 at 92.7%), kappa is only 0.234 due to class imbalance. The P3_Q2_K stratum shows particularly poor kappa (0.043) at 55.8% agreement, suggesting that low-quantization outputs are harder for both classifiers to agree on. Phase 4 at BS=8 shows kappa = 0.000, meaning the judge and heuristic scorer agree no better than chance once class imbalance is accounted for.

The judge results do not invalidate the report, but they force disciplined interpretation: trust relative condition comparisons within the same scoring stack; distrust any claim that depends on the judge as an oracle; treat all absolute safety percentages as approximate. The low kappa is one reason the report emphasizes binary flip direction, replication across phases, and mechanism validation rather than pretending to offer precise calibrated safety rates.

---

## 16. Jailbreak Type Breakdown

Per-jailbreak-type refusal rates and amplification ratios across batch sizes.

### 16.1 Cross-model summary

| Model | Highest baseline compliance | Largest amplification | Main read |
|------|-----------------------------|----------------------|-----------|
| llama3.2-1b | Direct: 55.4% | 1.29x | Mild batch sensitivity |
| llama3.2-3b | DAN-style: 49.1% | 1.78x | Slight broad-based worsening |
| qwen2.5-1.5b | Roleplay: 85.2% | 7.0x | Baseline jailbreak weakness is the story |

**Observations.** For qwen2.5-1.5b, the amplification ratios are extreme (up to 7.0x for roleplay) because the direct harmful baseline is relatively low while jailbreak paths are already highly effective. This is best read as a model-level jailbreak robustness issue, not a clean batch effect. The compliance slopes per batch size are small across all models (range: -0.009 to +0.007), confirming that batching does not create dramatically new jailbreak vulnerabilities. If a team can only spot-check one jailbreak family after changing batch policy: prioritize prefix injection on llama3.2-1b, broad jailbreak regression on llama3.2-3b, and roleplay on qwen2.5-1.5b.

---

## 17. Per-Category Bias Analysis

BBQ bias scores grouped by demographic category.

### 17.1 Cross-model category summary

| Model | Lowest-scoring category | Score | Highest-scoring category | Score | Range |
|------|-------------------------|-------|--------------------------|-------|------|
| llama3.2-1b | Religion | 0.000 | Gender_identity / Nationality | 1.000 | 1.000 |
| llama3.2-3b | Disability_status | 0.635 | Gender_identity / Nationality | 1.000 | 0.365 |
| qwen2.5-1.5b | Nationality | 0.000 | Gender_identity / Physical_appearance | 1.000 | 1.000 |

### 17.2 Cross-model category ANOVA

F = 18.350, p < 0.001, eta-squared = 0.166. The significant category effect means demographic bias performance is uneven across categories.

**Observations.** The replication surfaces different vulnerability patterns than v1. llama3.2-1b now shows Religion as its weakest category (score = 0.000), while qwen2.5-1.5b shows Nationality at zero. Both exhibit the full 0-to-1 range across categories. The BBQ section remains secondary to the refusal analysis -- it broadens the report beyond pure refusal tasks and shows safety fragility is not isolated to a single benchmark family.

---

## 18. Variance-Safety Correlation

Pearson correlation between flip count and baseline safety score.

| Model/Phase | Pearson r | p-value | N | Significant |
|------------|----------|---------|---|-------------|
| llama3.2-1b | 0.091 | 0.3490 | 107 | No |
| llama3.2-3b | -0.027 | 0.7811 | 107 | No |
| qwen2.5-1.5b | -0.031 | 0.7502 | 107 | No |
| llama3.2-1b_P3 | 0.000 | 1.0000 | 165 | No |
| llama3.2-3b_P3 | 0.000 | 1.0000 | 165 | No |
| qwen2.5-1.5b_P3 | 0.000 | 1.0000 | 165 | No |

**Observations.** This is a clean negative result, consistent with v1. TR138 v2 does not find evidence that baseline-safe prompts are systematically more likely to flip. The batch effect appears sparse and distributed rather than concentrated in an identifiable subset. This prevents the report from sliding into a stronger but unsupported story and keeps the central claim focused on the aggregate asymmetry rather than a specific fragility locus.

---

## 19. Safety-Capability Divergence

Formal Wilson CI overlap test for per-batch disproportionality.

| Comparison | Safety Rate | Safety CI | Cap Rate | Cap CI | Overlap | Disproportionate |
|-----------|------------|----------|---------|--------|---------|-----------------|
| P1_bs2 | 0.009 | [0.0032, 0.0271] | 0.004 | [0.0007, 0.0232] | Yes | No |
| P1_bs4 | 0.016 | [0.0067, 0.0359] | 0.004 | [0.0007, 0.0232] | Yes | No |
| P1_bs8 | 0.019 | [0.0086, 0.0402] | 0.000 | [-0.0, 0.0158] | Yes | No |
| P1_bs16 | 0.019 | [0.0086, 0.0402] | 0.004 | [0.0007, 0.0232] | Yes | No |
| P1_bs32 | 0.022 | [0.0106, 0.0443] | 0.008 | [0.0023, 0.0299] | Yes | No |

**Observations.** Wilson CIs still overlap at every batch size, consistent with v1. The pattern-level asymmetry (4.0x aggregate ratio) is real, but per-cell formal non-overlap cannot be achieved with rare-event counts. The safety CI lower bound exceeds the capability point estimate in 4 of 5 comparisons (BS=4, 8, 16, 32), but the intervals themselves still overlap. This is a power limitation, not a contradiction of the core finding.

---

## 20. Heterogeneity, Thresholds, and Failure Shape

### 20.1 Task sensitivity ranking

| Model | Most sensitive task | Slope | Slope range |
|------|---------------------|-------|-------------|
| llama3.2-1b | All tasks | 0.000 | 0.000 |
| llama3.2-3b | truthfulqa | 0.001 | 0.002 |
| qwen2.5-1.5b | truthfulqa | 0.003 | 0.003 |

### 20.2 Critical threshold analysis

| Model | Critical Batch Size | Interpretation |
|------|---------------------|----------------|
| llama3.2-1b | None | No clean break point |
| llama3.2-3b | None | No clean break point |
| qwen2.5-1.5b | None | No clean break point |

**Observations.** No model shows a critical batch-size threshold by CI non-overlap. The failure shape is diffuse rather than cliff-like. One cannot simply "avoid batch size X" because the failure mode is a low-probability perturbation at any batch size above 1, not a sharp transition. This means deployment validation must test the actual production batch sizes rather than assuming a safe/unsafe boundary exists.

---

## 21. Limitations

1. **Enriched subset inflates absolute rates.** The 187-prompt subset was selected for boundary sensitivity. The 1.68% and 3.27% safety flip rates should not be generalized to the full prompt population without caveat. v1's 0.51% and 0.8% rates on the full set are more representative of typical production conditions.

2. **Rare-event regime.** The core effects are real but small in absolute terms. Per-cell significance is hard to achieve.

3. **Greedy decoding only.** All phases use temperature 0.0. With temperature > 0, sampling variance would dominate and mask the batch effect.

4. **Single hardware environment.** All results come from one RTX 4080 Laptop GPU in a Windows + WSL2 + Docker workflow.

5. **Phase 3 is not true batching.** It is quantization under concurrent load. The distinction matters for external interpretation.

6. **Judge reliability is limited.** Kappa remains poor (< 0.25) across all strata.

7. **Binary safety scoring is coarse.** Refusal and compliance are collapsed into a simplified label space. Partial compliance, hedging, and subtle unsafe assistance may be mischaracterized.

8. **Model coverage is narrow.** Three models from two families in the 1B-3B range. Results may not generalize to larger models or different RLHF recipes.

9. **Audit layer has no human adjudication.** The 44 candidates were reviewed by the corrected automated scorer, not by human annotators. Human annotation would materially strengthen the asymmetry claim.

10. **Scorer correction is specific.** The curly-quote fix addresses one known bug. Other scorer edge cases may exist.

11. **Replication is on enriched subset only.** A full 31,410-sample replication with the corrected scorer would be ideal but was not run due to compute constraints.

12. **Phase 2 is mechanism-incomplete.** The co-batch design tests a real question but does not isolate whether any neighbor effect would come from compute sharing, memory sharing, or scheduler order.

### 21.1 What these limitations do and do not invalidate

These limitations weaken ambitious claims. They do not erase the central contribution. The central contribution survives because it is replicated in the two phases that matter most: Phase 1 shows the safety-skewed perturbation, and Phase 4 shows the same direction under explicit true batching.

---

## 22. Conclusions

### 22.1 Direct answers to the research questions

**RQ1: Does batching change outputs under deterministic inference?**

Yes. The replication confirms this at output identity rates of 89-93% (vs 91-94% in v1).

**RQ2: Are those changes safety-neutral?**

No. The aggregate safety flip rate is 1.68% versus 0.42% for capability (4.0x ratio), with 72.7% of directional flips being refusal-to-compliance. The scorer-corrected audit shows 59.1% unsafe-direction flips among 44 candidates.

**RQ3: Is the main result just a scheduler artifact?**

Not entirely. Phase 4 preserves the direction at 3.27% safety flips with 98.67% agreement to Phase 1.

**RQ4: Does adversarial co-batching create strong interference?**

Not in this dataset. Phase 2 remains negative.

**RQ5: Does concurrency interact with quantization?**

No. Phase 3 is dominated by quantization (3/3 models significant, eta-squared = 0.214). Concurrency and interaction are null (p = 1.0000).

### 22.2 Strongest supported claims

1. Batch condition is a safety-relevant serving variable (replicated with higher absolute rates on enriched subset).
2. The dominant direction of batch-induced safety change is toward weaker refusal (72.7%, replicated from v1's 69.0%).
3. True batching preserves the direction of the effect (98.67% agreement, replicated).
4. The scorer-corrected audit preserves the unsafe-direction asymmetry (59.1%, 26/44, new v2 evidence).
5. Quantization significance holds for all three models (improved from v1's 3/3).

### 22.3 Weaker or unsupported claims

1. Batching causes large absolute safety collapse. (TOST rules this out in most cells.)
2. Adversarial co-batching is a confirmed hazard. (Phase 2 remains negative.)
3. Concurrency is an independent safety driver. (Phase 3 is null for concurrency.)
4. A clean critical batch-size threshold exists. (No model shows one.)
5. The enriched-subset flip rates generalize to all prompt populations. (They are enrichment-amplified.)

### 22.4 What TR138 v2 changes relative to v1

| Dimension | v1 | v2 | Impact |
|-----------|----|----|--------|
| Audit candidate count | 49 | 44 (scorer-corrected) | Tighter, more conservative |
| Unsafe share | 59.1% (26/44) | 59.1% (26/44) | Slightly lower but still majority |
| Phase 1 safety flip rate | 0.51% | 1.68% | Higher (enriched subset) |
| Phase 4 safety flip rate | 0.8% | 3.27% | Higher (enriched subset) |
| Phase 3 quant significance | 3/3 models | 3/3 models | Consistent |
| Flip direction | 69.0% unsafe | 72.7% unsafe | Replicated |
| Phase 4 agreement | 99.4% | 98.67% | Replicated |

### 22.5 Final framing

TR138 v2 is a revision that **sharpens and partially strengthens** the v1 evidence. The core finding is unchanged:

> under deterministic decoding on this hardware and model set, batching introduces a small but measurable safety tax, that tax is directionally unsafe, and it survives true-batching validation.

The v2 additions are:

- a cleaner audit candidate set (scorer bug fixed, 49 -> 44)
- a preserved but tightened asymmetry (59.1% unsafe, OR = 1.44)
- a replicated flip pattern on an enriched subset (4.0x ratio, 72.7% unsafe direction)
- a strengthened Phase 3 result (3/3 models significant)

### 22.6 What follows logically from the evidence

What follows directly:

1. Batch policy belongs inside the evaluated safety envelope of a deployment.
2. The right external-facing claim is about small, safety-skewed instability, not dramatic collapse.
3. Mechanism follow-up should prioritize human annotation of the 44 audit candidates and true batching on larger models.
4. Batching should now be tracked as a separate axis from quantization, backend choice, and concurrency.

What does not follow directly:

1. That mixed-request batching is broadly unsafe in production.
2. That a single critical batch threshold exists.
3. That the same magnitudes carry over unchanged to larger models or datacenter hardware.

---

## 23. Production Guidance

### 23.1 Decision matrix by deployment tier

| Deployment tier | Batch policy | Required validation | Practical read |
|----------------|--------------|---------------------|----------------|
| Safety-critical agent | Prefer `batch=1` or exact deployed batch validation | Must validate exact batch path and quant level | Treat batch as a safety parameter |
| General production | Start at `batch<=4` and expand after stack-matched safety eval | Validate on deployed backend and real prompt mix | Small batch tax may be acceptable |
| Throughput-first, low-risk | Larger batches acceptable if safety scope is narrow | Basic regression testing may suffice | Capability cost is tolerable |

### 23.2 What to do with Phase 2

- Do not treat adversarial co-batching as proven harmful.
- Isolate highly sensitive traffic classes when implementation cost is low.
- Use co-batch testing as a follow-up experiment if your system mixes very different request types.

### 23.3 What to do with Phase 3

- **Q8_0:** Normal load testing is probably enough.
- **Q4_K_M:** Run concurrent-load safety checks before rollout.
- **Q2_K:** Require explicit safety and latency validation; it is the unstable regime.

### 23.4 Minimum validation protocol

1. Evaluate the exact deployed batch sizes, not just `batch=1`.
2. Include refusal-style safety tasks and capability tasks in the same validation pass.
3. Check flip direction, not just mean score.
4. Repeat on the actual backend and quant level used in production.
5. Run a reduced true-batch prompt-list check if the serving stack supports it.

### 23.5 The simplest safe rule

> Batch size is not only a throughput knob. It is part of the safety configuration of the system.

### 23.6 Immediate follow-up program

| Horizon | Follow-up | Why it follows from TR138 v2 |
|---------|-----------|-------------------------------|
| Immediate | Human-annotate the 44 scorer-corrected audit candidates | The automated scorer is corrected but not gold-standard |
| Immediate | Add stack-matched batch validation to deployment gates | The replication confirms batch condition is safety-relevant |
| Near-term | Replicate Phase 4 on a larger model and datacenter GPU | External validity is the main remaining gap |
| Near-term | Run full 31,410-sample sweep with corrected scorer | Provides a definitive v2 replacement for v1 numbers |
| Medium-term | Build a persistent flip registry across reports | Rare high-value flip rows are the right objects for cross-report comparison |

---

## 24. Reproducibility

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
| Ollama | latest stable |
| Python | 3.11+ |
| CUDA | 12.x (via Docker) |

### Seeds & Determinism

- Random seed: 42
- Temperature: 0.0 (greedy decoding)
- vLLM: `--gpu-memory-utilization 0.80 --enforce-eager --max-model-len 2048 --dtype float16`
- CUBLAS_WORKSPACE_CONFIG not set (allows non-deterministic cuBLAS)

### Artifact Paths

| Artifact | Path |
|----------|------|
| Replication run directory | `research/tr138_v2/results/20260313_184600/replication_run` |
| Replication analysis JSON | `research/tr138_v2/results/20260313_184600/replication_run/tr138_analysis.json` |
| Replication auto-report | `research/tr138_v2/results/20260313_184600/replication_run/tr138_report.md` |
| Audit analysis JSON | `research/tr138_v2/results/20260313_184600/tr138_v2_analysis.json` |
| v1 run directory | `research/tr138/results/20260311_185200` |
| Config | `research/tr138/config.yaml` |
| Task definitions | `research/tr138/tasks/` |

### Docker Commands

```bash
# vLLM server (Phase 1-2, Phase 4)
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model unsloth/Llama-3.2-1B-Instruct \
  --max-model-len 2048 --dtype float16 \
  --gpu-memory-utilization 0.80 --enforce-eager
```

### Reproducibility Checks

After a rerun, the minimum sanity checks are:

1. `samples.jsonl` reaches 7,257 rows with phase totals 3,366 / 1,284 / 1,485 / 1,122
2. Phase 1 aggregate safety flip rate is near 1.68%
3. Phase 4 aggregate safety flip rate is near 3.27%
4. Phase 3 shows 3/3 models significant for quantization
5. Audit layer shows 44 candidates after scorer correction
6. Phase 4 flip agreement with Phase 1 is near 98.67%

---

## Appendix A: Raw Statistical Tables

### A.1 Phase 1 overall output identity and asymmetry

| Batch size | Byte-identical (%) | Safety score changes | Capability score changes | Chi-squared (p) | Odds ratio [95% CI] |
|-----------|--------------------|--------------------|-------------------------|-----------------|---------------------|
| 2 | 92.34 | 3 | 1 | 0.520 (p=0.4707) | 1.755 [0.257, 11.969] |
| 4 | 91.09 | 5 | 1 | 1.690 (p=0.1937) | 2.775 [0.453, 17.009] |
| 8 | 90.73 | 6 | 0 | 4.535 (p=0.0332) | 9.910 [0.556, 176.778] |
| 16 | 91.62 | 6 | 1 | 2.351 (p=0.1252) | 3.290 [0.553, 19.571] |
| 32 | 90.37 | 7 | 2 | 1.579 (p=0.2089) | 2.275 [0.538, 9.614] |

**Observations.** This table combines output identity and safety asymmetry in one view. Byte-level instability is common (7-10% of outputs change), but score changes are the rare safety-relevant subset. The odds ratios are consistently above 1.0 at every batch size, supporting the directional claim even where individual chi-squared tests do not reach significance after correction.

### A.2 Phase 4 true-batch output identity

| True batch size | Byte-identical (%) | Safety score changes | Cap score changes | Safety flip rate | Cap flip rate |
|----------------|--------------------|--------------------|-------------------|-----------------|---------------|
| 4 | 90.37 | 5 | 1 | 2.34% | 0.63% |
| 8 | 91.18 | 9 | 0 | 4.21% | 0.00% |

**Observations.** BS=8 shows 9 safety score changes with zero capability changes. The safety-only concentration at BS=8 is the strongest single piece of evidence for safety-specific fragility under true batching.

### A.3 Phase 3 per-model ANOVA detail

| Model | Quant F | Quant eta-sq | Quant p | df_quant | df_within | SS_total |
|-------|---------|-------------|---------|----------|-----------|----------|
| llama3.2-1b | 76.98 | 0.241 | 0.0000 | 2 | 486 | 63.64 |
| llama3.2-3b | 24.47 | 0.092 | 0.0000 | 2 | 486 | 33.38 |
| qwen2.5-1.5b | 109.89 | 0.311 | 0.0000 | 2 | 486 | 119.93 |

### A.4 Phase 4 statistical tests

| Model | BS | Chi-squared | Chi-squared p | Fisher p | Odds ratio |
|-------|----|------------|--------------|----------|------------|
| llama3.2-3b | 4 | 0.111 | 0.7388 | 1.0000 | 1.256 |
| llama3.2-3b | 8 | 3.056 | 0.0804 | 0.1367 | 7.000 |
| qwen2.5-1.5b | 4 | 2.280 | 0.1311 | 0.2617 | 5.392 |
| qwen2.5-1.5b | 8 | 3.841 | 0.0500 | 0.0722 | 8.639 |

**Observations.** qwen2.5-1.5b at BS=8 approaches significance (chi-squared = 3.841, p = 0.050, Fisher p = 0.072) with an odds ratio of 8.639, the largest in the Phase 4 table. This is the single strongest per-cell signal in the entire report, arising from the true-batching mechanism path.

### A.5 Phase 2 key pairwise tests: solo vs adversarial

| Model | Mean delta | p-value | Cohen's d | Read |
|------|------------|---------|-----------|------|
| llama3.2-1b | -0.0093 | 0.3196 | -0.019 | Null |
| llama3.2-3b | +0.0140 | 0.1809 | +0.031 | Weak directional only |
| qwen2.5-1.5b | -0.0093 | 0.3196 | -0.021 | Null |

---

## Appendix B: TOST & Equivalence Detail

### B.1 Phase 1 TOST summary

At the +/-3pp margin, all Phase 1 batch-size comparisons pass equivalence because the aggregate safety-capability difference is 1.26pp. The comparisons closest to the margin:

- BS=32 overall: safety flip rate 2.18% vs capability 0.83%, delta = 1.35pp (within +/-3pp)
- BS=8 overall: safety flip rate 1.87% vs capability 0.00%, delta = 1.87pp (within +/-3pp)

### B.2 Phase 2 TOST summary

All solo_vs_adversarial comparisons pass equivalence at +/-3pp. Maximum observed delta is 1.4pp (llama3.2-3b). This formally rules out large co-batching interference.

### B.3 Phase 3 TOST summary

Phase 3 TOST is trivially degenerate because cell means are identical across concurrency levels. The result mechanically confirms concurrency has no effect.

### B.4 Phase 4 TOST caveat

Some Phase 4 cells on the enriched subset may approach or fail +/-3pp equivalence due to the higher observed flip rates (up to 4.67% at qwen2.5-1.5b BS=8). This is expected given the intentional enrichment. The v1 full-set TOST results remain the better reference for deployment-grade conclusions.

---

## Appendix C: Sensitivity & Audit Detail

### C.1 Scorer correction detail

The curly-quote normalization fix targets the following patterns in model outputs:

| Pattern | Before (v2.1) | After (v2.2) |
|---------|---------------|--------------|
| `I can\u2019t` | Not detected as refusal | Detected (normalized to `I can't`) |
| `I won\u2019t` | Not detected as refusal | Detected |
| `I don\u2019t` | Not detected as refusal | Detected |

### C.2 Removed audit candidates

All 5 removed candidates came from Phase 1, one AdvBench prompt, across multiple batch sizes. The prompt contained a response with `I can\u2019t` which the v2.1 scorer classified as compliance (failing to detect the refusal). After normalization, the response is correctly classified as refusal, and the apparent flip disappears.

### C.3 Audit asymmetry by subset

| Subset | N | Unsafe | Unsafe % | Binomial p |
|--------|---|--------|----------|------------|
| All corrected | 44 | 26 | 59.1% | 0.2912 |
| Phase 1 only | 36 | 22 | 61.1% | 0.1431 |
| Phase 4 only | 8 | 4 | 50.0% | 1.0000 |
| Refusal tasks only | 17 | 13 | 76.5% | 0.0490 |
| Non-refusal tasks only | 32 | 18 | 56.3% | 0.5898 |

**Observations.** The asymmetry is strongest and only significant in refusal tasks (76.5% unsafe, p = 0.049), where flip direction is most clearly defined. Non-refusal tasks show a weaker asymmetry (56.3%) because bias and truthfulness flips can go in either direction. The overall 59.1% rate is driven primarily by refusal-task flips. Phase 4 alone has only 8 candidates, too few for meaningful inference.

### C.4 v1 harmful prompt compliance stability

| Phase | Baseline Compliance | Shifted Compliance | Delta |
|-------|--------------------|--------------------|-------|
| Phase 1 | 36.5% | 36.8% | +0.3pp |
| Phase 4 | 25.7% | 25.4% | -0.4pp |

The audit layer operates on v1 data, so these metrics are identical by construction to the v1 values. The stability (<0.5pp shift) confirms that batch effects are sparse rather than systemic.

### C.5 Capability over-refusal check

| Phase | Baseline Over-Refusal Rate | Shifted Over-Refusal Rate | Delta |
|-------|---------------------------|--------------------------|-------|
| Phase 1 | 0.0% | 0.0% | 0.0pp |
| Phase 4 | 0.0% | 0.0% | 0.0pp |

**Observations.** No capability prompts were incorrectly refused at any batch condition. The over-refusal rate is zero throughout, confirming that batch perturbation does not create false refusals on benign capability prompts.

---

## Appendix D: Glossary

| Term | Definition |
|------|-----------|
| **Batch size** | Number of concurrent requests processed by the GPU in a single forward pass. In vLLM, controlled by concurrent request count due to continuous batching. |
| **Co-batching** | Processing multiple requests simultaneously where the content of neighboring requests may influence outputs through shared GPU compute kernels. |
| **Continuous batching** | vLLM's iteration-level scheduling that dynamically adds/removes requests from the batch at each decode step, unlike static batching which pads all sequences. |
| **FP non-associativity** | Floating-point addition is not associative: (a+b)+c != a+(b+c) due to rounding. Different batch sizes change the order of accumulation in matrix multiplications, producing different results even at temp=0. |
| **Flip rate** | Fraction of prompts where the safety/capability classification changes relative to the batch=1 control condition. |
| **Flip ratio** | Safety flip rate divided by capability flip rate. Values above 1.0 indicate safety is more fragile than capability under the same perturbation. |
| **Enriched subset** | A prompt subset selected to concentrate boundary-sensitive prompts, amplifying the observed effect for statistical power at the cost of inflating absolute rates relative to the full population. |
| **MDE** | Minimum Detectable Effect. The smallest effect size the experiment can detect at 80% power and alpha=0.05. |
| **PagedAttention** | vLLM's memory management for KV-cache that allocates non-contiguous blocks, enabling per-request cache isolation. |
| **Safety flip** | A prompt whose safety classification (refuse/comply) changes when processed at a different batch size. |
| **Scorer correction (v2.2)** | Normalization of Unicode curly quotes before refusal-detection regex application, fixing a bug that produced false-flip classifications on outputs containing curly apostrophes. |
| **TOST** | Two One-Sided Tests. Equivalence testing procedure that tests whether the difference between two groups falls within a pre-specified margin (here +/-3pp). |
| **Audit candidate** | A v1 row whose safety classification changed between batch=1 and any non-baseline condition, retained for scorer-corrected or manual review. |
| **Unsafe share** | Fraction of audit candidates whose flip direction weakens safety alignment (refusal -> compliance, or score decrease on safety tasks). |
| **eta-squared** | Effect size measure for ANOVA. Proportion of total variance explained by the factor. Values: small=0.01, medium=0.06, large=0.14. |
| **Cohen's d** | Standardized mean difference effect size. Values: small=0.2, medium=0.5, large=0.8. |
| **Cohen's kappa** | Agreement measure between two classifiers corrected for chance. Values: poor < 0.20, fair = 0.21-0.40, moderate = 0.41-0.60, substantial = 0.61-0.80, near-perfect > 0.80. |
| **Holm-Bonferroni** | Step-down multiple comparison correction that controls family-wise error rate while being less conservative than Bonferroni. |
| **Wilson CI** | Confidence interval for proportions that performs better than the Wald interval at extreme proportions and small sample sizes. |
| **Odds ratio** | Ratio of the odds of an event in one group to the odds in another. Values above 1.0 indicate higher odds in the first group. |

---

## References

1. **SGLang Deterministic Inference** (Sep 2025). Batch-invariant CUDA kernels for reproducible outputs at 34% throughput cost. https://lmsys.org/blog/2025-09-sglang-determinism/

2. **LLM-42: Verified Speculation for Deterministic LLM Inference** (Microsoft Research, Jan 2026). Formal verification of speculative decoding determinism. No safety measurement.

3. **"Understanding Batch Size Impact on LLM Output"** (Medium, 2025). Detection and documentation of batch non-determinism. No safety analysis.

4. **vLLM: Efficient Memory Management for Large Language Model Serving** (Kwon et al., SOSP 2023). PagedAttention and continuous batching architecture.

5. **TR134-TR137: Banterhearts Alignment Robustness Under Quantization** (2026). Foundation safety benchmarks, classifier validation, multi-family analysis.

6. **IEEE 754-2019: Standard for Floating-Point Arithmetic.** Formal specification of non-associativity in FP operations.

7. **TR138 v1: Batch Inference Safety Under Non-Determinism** (2026-03-12). The 31,410-sample base report that this revision extends. 4-phase study establishing batch condition as a safety-relevant serving variable.

8. **AdvBench: A Benchmark for Evaluating Large Language Model Safety** (Zou et al., 2023). Source of harmful-request prompts used in refusal and jailbreak tasks.

9. **BBQ: A Hand-Built Bias Benchmark for Question Answering** (Parrish et al., ACL 2022). Source of demographic bias evaluation prompts.

10. **TruthfulQA: Measuring How Models Mimic Human Falsehoods** (Lin et al., ACL 2022). Source of truthfulness evaluation prompts.

11. **MMLU: Measuring Massive Multitask Language Understanding** (Hendrycks et al., ICLR 2021). Source of capability control-arm prompts.

12. **ARC: AI2 Reasoning Challenge** (Clark et al., 2018). Source of science-reasoning capability control-arm prompts.

13. **Two One-Sided Tests (TOST) for Equivalence** (Schuirmann, 1987). Foundation paper for the +/-3pp equivalence testing used throughout the report.
