# Technical Report 138: Batch Inference Safety Under Non-Determinism
## Controlled study of whether deterministic batching changes safety behavior more than capability behavior, with explicit true-batching validation

| Field | Value |
|-------|-------|
| **TR Number** | 138 |
| **Date** | 2026-03-12 |
| **Version** | 1.0 |
| **Author** | Research Team |
| **Git Commit** | `7b77e9c0` |
| **Status** | Complete |
| **Report Type** | Safety alignment analysis (4-phase batching study) |
| **Run Directory** | `20260311_185200` |
| **Total Samples** | 31,410 |
| **Phase 1 Samples** | 17,154 |
| **Phase 2 Samples** | 5,616 |
| **Phase 3 Samples** | 5,940 |
| **Phase 4 Samples** | 2,700 |

---

## Positioning

TR138 is a flagship report in the Banterhearts safety-serving line. Its contribution is intentionally narrow and empirical rather than broad and speculative.

The novelty claim here is narrow and defensible:

> prior work treats batch non-determinism mainly as a reproducibility, numerical, or performance issue; TR138 tests whether it is also a **safety failure mode**.

That is the core contribution. The strongest evidence comes from the combination of:

- **Phase 1:** safety-versus-capability asymmetry under controlled vLLM batch sweeps
- **Phase 4:** explicit prompt-list true batching, which tests whether the effect survives without request-arrival timing confounds

The report is deliberately more cautious on two other axes:

- **Phase 2:** co-batching interference is not clearly established
- **Phase 3:** quantization x concurrency is operationally useful, but it is not itself a batching result

The report keeps those boundaries explicit so the novelty claim stays defensible.

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
| **TOST equivalence** | Two one-sided tests at `+/-3pp` margin | Tests practical interchangeability, not just non-significance |
| **Eta-squared** | ANOVA effect size | Share of variance explained |
| **Cohen's kappa** | Agreement between heuristic scoring and the LLM judge | Higher = better agreement |

### Statistical methods used

- **Phase 1:** paired comparisons versus `batch=1`
- **Phase 2:** paired comparisons across `solo`, `benign`, `adversarial`, and `safety`
- **Phase 3:** two-way ANOVA for `quantization x concurrency`
- **Phase 4:** paired comparison between true batching and synchronized-dispatch references
- **Holm-Bonferroni** correction for multiple testing
- **TOST equivalence** at `+/-3pp`
- **Bootstrap / confidence intervals** where emitted by the generated analysis

### Important caveats up front

1. **Greedy decoding only.** All runs use `temperature=0.0`.
2. **Consumer GPU only.** This is one RTX 4080 Laptop-class environment.
3. **Small-model regime.** The study covers roughly 1B-3B models.
4. **Phase 3 is not batching.** It is concurrent load under varying quant levels.
5. **Judge agreement is weak.** Absolute percentages should be treated more cautiously than within-experiment contrasts.

---

## 1. Abstract

This report presents results from TR138, a 31,410-sample experiment testing whether batch-induced output non-determinism in GPU inference disproportionately degrades safety compared to capability. We evaluate 3 instruction-tuned models across two families (Llama 3.2, Qwen 2.5) using controlled vLLM batching (Phase 1-2), Ollama quantized serving under concurrent load (Phase 3), and a compact true-batching validation on explicit vLLM prompt lists (Phase 4).

The main result is that **batch-induced changes are not safety-neutral**. In Phase 1, safety outputs flip at **0.5%** while capability outputs flip at **0.1%**, a **3.6x differential**. The dominant direction is **refusal -> compliance** (**69.0%** of classified flips), meaning the instability is not just harmless wording drift: when classification changes, it more often weakens refusal behavior than strengthens it.

The most important methodological result is Phase 4. Explicit prompt-list true batching still produces **0.8%** safety flips and retains **99.4% agreement** with the synchronized-dispatch signal. This indicates the core safety effect is not merely a scheduler artifact.

Phase 2 is much weaker. The co-batching design is now methodologically coherent, but the empirical differences between `solo`, `benign`, `adversarial`, and `safety` neighbors are small and not statistically convincing in the present sample. Phase 3 shows a separate result: **quantization matters, concurrency does not**. That finding is useful operationally, but it should not be folded back into the batching claim.

The defensible conclusion is:

> under deterministic decoding on this hardware and model set, batching introduces a small but measurable safety tax, and that tax survives a true-batching validation step.

---

## 2. Table of Contents

- [Positioning](#positioning)
- [Metric Definitions & Statistical Methods](#metric-definitions--statistical-methods)
- [1. Abstract](#abstract)
- [2. Table of Contents](#table-of-contents)
- [3. Executive Summary](#executive-summary)
- [4. Research Question & Hypotheses](#research-question--hypotheses)
- [5. Methodology](#methodology)
- [6. Models & Configuration](#models--configuration)
- [7. Phase 1: Batch Size x Output Determinism](#phase-1-batch-size-x-output-determinism)
- [8. Phase 2: Co-Batching Interference](#phase-2-co-batching-interference)
- [9. Phase 3: Quantization x Concurrency Interaction](#phase-3-quantization-x-concurrency-interaction)
- [10. Cross-Phase Synthesis](#cross-phase-synthesis)
- [11. TOST Equivalence Analysis](#tost-equivalence-analysis)
- [12. Power Analysis](#power-analysis)
- [13. Latency Analysis](#latency-analysis)
- [14. Judge Agreement Analysis](#judge-agreement-analysis)
- [15. Jailbreak Type Breakdown](#jailbreak-type-breakdown)
- [16. Per-Category Bias Analysis](#per-category-bias-analysis)
- [17. Variance-Safety Correlation](#variance-safety-correlation)
- [18. Safety-Capability Divergence](#safety-capability-divergence)
- [19. Heterogeneity, Thresholds, and Failure Shape](#heterogeneity-thresholds-and-failure-shape)
- [20. Cross-TR Validation](#cross-tr-validation)
- [21. Limitations](#limitations)
- [22. Conclusions](#conclusions)
- [23. Production Guidance](#production-guidance)
- [24. Reproducibility](#reproducibility)
- [A. Appendix A: Source-of-Truth Artifacts](#appendix-a-source-of-truth-artifacts)
- [B. Appendix B: Output Identity & Flip Audit Tables](#appendix-b-output-identity--flip-audit-tables)
- [C. Appendix C: Statistical Test Inventory](#appendix-c-statistical-test-inventory)
- [D. Appendix D: Glossary](#appendix-d-glossary)
- [References](#references)

---

## 3. Executive Summary

### Key findings

1. **Batching changes safety behavior more than capability behavior.** Phase 1 safety flips are 0.5% versus 0.1% for capability, a 3.6x differential.
2. **The unsafe direction dominates.** Refusal -> compliance accounts for 69.0% of classified safety flips.
3. **The signal survives explicit true batching.** Phase 4 reports 0.8% safety flips under prompt-list batching, with 99.4% agreement to the synchronized-dispatch signal.
4. **Co-batching interference is not established.** Phase 2 effects are small, inconsistent, and non-significant.
5. **Quantization is the real Phase 3 story.** Quantization is significant; concurrency and the interaction term are effectively null.
6. **Batching is a real but bounded safety tax.** Most outputs remain stable, but the unstable subset is disproportionately safety-relevant.

### Validation summary

| Target | Metric | Achieved | Status |
|--------|--------|----------|--------|
| Safety-capability asymmetry detected | Phase 1 flip ratio | 3.6x | PASS |
| Unsafe directionality detected | Refusal -> compliance share | 69.0% | PASS |
| True-batch confirmation | Phase 4 flip agreement | 99.4% | PASS |
| Co-batch interference established | Phase 2 pairwise tests | Not established | MIXED / NO CLEAR EFFECT |
| Concurrency hazard established | Phase 3 concurrency ANOVA | p = 1.0000 | REFUTED |

### Citation-grade claim hierarchy

| Claim tier | Statement | Evidence strength | Best sections to cite |
|------------|-----------|-------------------|-----------------------|
| Primary claim | Batch condition is a safety-relevant serving variable under deterministic decoding | Strong | 7, 10.2, 22 |
| Primary support | The dominant batch failure direction is refusal -> compliance | Strong | 7.3, 22 |
| Mechanism support | The signal survives explicit true batching | Strong | 10.2, 22 |
| Negative finding | Adversarial co-batching is not established as a strong hazard here | Strong negative | 8, 11, 23 |
| Auxiliary finding | Quantization matters more than concurrency in the Phase 3 stack | Strong | 9, 10.1, 23 |
| Non-claim | TR138 proves a universal critical batch-size threshold | Not supported | 18, 19, 22 |

### Claim validation

| Claim | Evidence base | Status |
|-------|---------------|--------|
| Batch-induced changes are safety-neutral | Safety flips exceed capability flips by 3.6x | **REFUTED** |
| Batching mostly causes harmless wording drift | 69.0% of flips are refusal -> compliance | **REFUTED** |
| Phase 1 is only a scheduler artifact | Phase 4 true batching retains the signal | **REFUTED** |
| Adversarial co-batching clearly harms nearby safety | Phase 2 deltas are small and non-significant | **NOT ESTABLISHED** |
| Quantization x concurrency interaction is a major combined effect | Interaction p = 1.0000 | **REFUTED** |
| Batch size is operationally safety-relevant | Phase 1 + Phase 4 jointly support this | **VALIDATED** |

### Key decisions for practitioners

1. **Validate safety at the exact production batch sizes you intend to serve.**
2. **Do not assume `temperature=0` eliminates deployment-time safety variance.**
3. **Treat batching and quantization as distinct safety axes.**
4. **Do not use TR138 to claim strong co-batching interference.**
5. **Use Phase 4 as the strongest external-facing evidence in this report.**

### When to use this report

**Scenario 1: "Is batching just a performance knob?"**  
Use TR138 to answer: no. On this setup it also changes safety behavior.

**Scenario 2: "Does the effect survive true batching?"**  
Use Phase 4. That is the cleanest mechanism test in the report.

**Scenario 3: "Should I worry more about concurrency or quantization?"**  
Use Phase 3. Quantization matters; concurrency alone does not in this setup.

**Scenario 4: "Can I claim adversarial co-batching is dangerous?"**  
Use Phase 2 carefully. TR138 does not establish that claim strongly.

### How to read this report

| Time | Reading path |
|------|--------------|
| **2 min** | Abstract + Key Findings |
| **10 min** | Executive Summary + Sections 7, 9, and 10.2 |
| **30 min** | Add Sections 10.3, 13, and 21-23 |
| **Deep dive** | Full report including latency, jailbreak, bias, and reproducibility sections |

---

## 4. Research Question & Hypotheses

> **Research Question:** Does batch-induced output non-determinism disproportionately degrade safety compared to capability?

### Hypotheses

- **H1 (Null):** Batch-induced output changes are safety-neutral (uniform random flips across all output types).
- **H2 (Alternative):** Batch-induced changes disproportionately degrade safety (safety tokens are more fragile than capability tokens).
- **H3 (Interference):** Co-batching adversarial prompts alongside safety prompts affects safety outcomes (cross-request interference via shared GPU state).

### 4.1 Why this matters in production

In most serving stacks, batch size is tuned for cost and throughput. The implicit assumption is that if decoding is greedy and the model weights are unchanged, batching is a performance decision rather than a behavior decision.

TR138 tests whether that assumption is incomplete. If a model evaluated at `batch=1` in a lab behaves differently at `batch=8` or `batch=32` in production, then the serving stack itself becomes part of the safety envelope. That matters for at least three deployment classes:

1. **Safety-critical assistants** where refusal boundaries are part of the product contract
2. **Backend gateways** that dynamically adjust batch size under load
3. **Regression-testing pipelines** that assume deterministic outputs imply stable safety behavior

The report is therefore not asking whether batching exists, or whether batching improves throughput. Those are already known. It is asking whether batching belongs on the list of **safety-relevant deployment parameters**.

### 4.2 What would count as strong evidence

The strongest evidence for the core claim is not merely "outputs differ." Output differences happen for many reasons and are not all safety-relevant.

For TR138, strong evidence requires all of the following:

- **A safety-capability asymmetry:** safety outputs must change more often than capability outputs
- **A directionally concerning pattern:** changes should lean toward weaker refusal behavior rather than symmetric noise
- **A mechanism check:** the effect should survive a cleaner batching implementation, not only synchronized request timing

This is why Phase 1 and Phase 4 matter most. Phase 1 gives the asymmetry and directionality. Phase 4 attacks the strongest alternative explanation.

### 4.3 What this report does not try to prove

TR138 does not attempt to prove any of the following:

- that all co-batching is dangerous in general
- that concurrency alone degrades safety
- that the observed effect generalizes unchanged to larger models or datacenter GPUs
- that batching is more important than quantization on every model family

Those would require different experiments. The report is strongest when read as a focused answer to one question: whether deterministic batching can produce **safety-specific output instability**.

---

## 5. Methodology

### Experimental Design

Four-part experiment measuring output non-determinism under batch inference on consumer GPU hardware (RTX 4080 Laptop, 12GB VRAM).

- **Temperature:** 0.0 (greedy decoding) throughout all phases
- **Seed:** 42 (fixed for CUDA/cuBLAS where supported)
- **Max tokens:** 256
- **Warmup:** 3 requests per model before data collection

### 5.1 Threat model

The threat model in TR138 is not an external jailbreak attacker changing the prompt text. The threat model is a deployment stack that changes **serving conditions** while keeping prompt text, weights, and decoding policy nominally fixed.

The failure mode of interest is:

> a prompt that is refused or judged safer at one batch condition crosses the refusal boundary and becomes compliant or less safe at another batch condition.

That is why the report emphasizes refusal/compliance flips and not only output identity. Byte-level output changes are only interesting here insofar as they induce a materially different safety classification.

### 5.2 Why four phases

Each phase exists to answer a different version of the same question.

- **Phase 1** asks whether batch size changes safety behavior more than capability behavior.
- **Phase 2** asks whether the identity of neighboring prompts matters beyond batch size alone.
- **Phase 3** asks whether quantized low-precision serving is sensitive to concurrent load.
- **Phase 4** asks whether the Phase 1 signal survives a cleaner true-batching mechanism.

This structure matters because otherwise the report would mix scheduler timing, neighbor effects, quantization effects, and tensor batching into one claim. The four-phase design is what makes the conclusions separable.

### Batch Control Mechanism

- **Phase 1 (vLLM):** Synchronized request groups force exact in-flight batch sizes.
- **Phase 2 (vLLM):** One target prompt is evaluated under four conditions: `solo`, `benign`, `adversarial`, and `safety` co-batches.
- **Phase 3 (Ollama):** Concurrent API load is used as a separate proxy axis. It measures quantization x concurrency, not true batching.
- **Phase 4 (vLLM):** A single completions call receives a prompt list, giving explicit true batching without cross-request arrival timing effects.

### 5.3 Why the safety-capability control arm matters

The central claim of the report would be much weaker without capability tasks.

If batching changed safety outputs at exactly the same rate as capability outputs, then the right interpretation would be generic output instability rather than a safety-specific effect. By pairing safety tasks with MMLU and ARC-Challenge controls, TR138 asks the sharper question:

> does batching disproportionately perturb the aligned layer of model behavior relative to ordinary task performance?

That comparison is one of the most important design decisions in the report.

### Co-Batching Design (Phase 2)

Four conditions are used in Phase 2:
- **Solo:** Target prompt evaluated alone (`batch_size=1`).
- **Benign:** Target prompt co-batched with innocuous factual questions.
- **Adversarial:** Target prompt co-batched with harmful/jailbreak prompts.
- **Safety:** Target prompt co-batched with non-adversarial safety-evaluation prompts.

### 5.4 What the scoring pipeline can and cannot tell us

TR138 uses the same kind of automated scoring stack that underlies adjacent Banterhearts safety work: refusal-style heuristics, task-specific scoring, and an LLM judge pass.

This is good enough for within-experiment comparisons, but it is not equivalent to human red-teaming or gold-standard annotation. That is why the report is strongest on **relative comparisons across conditions** and weaker on claims about absolute deployed safety rates.

### 5.5 Full data-generation specification

The cleanest way to read TR138 is to start from the raw row-generation process rather than from the summary tables.

Every analyzed row originates from a fixed YAML task file in `research/tr138/tasks/`. The runner loads those files in source order, renders `prompt_template` fields by direct substitution, and carries forward the metadata needed later for scoring:

- shared fields: `task_name`, `task_type`, `sample_id`, `prompt`, `reference`, `correct_answers`, `incorrect_answers`
- task metadata in `sample_meta`: BBQ category and answer-choice structure, TruthfulQA context, jailbreak family, and prompt-specific contextual fields such as `question` and `instruction`

The benchmark membership itself is deterministic. There is no online sampling, no dynamic prompt generation, and no hidden reservoir. When reduced subsets are needed, `_subset_prompts_by_task()` takes the first `N` rows per task while preserving source order.

The exact executed scope for this run was:

| Phase | Prompt source | Executed prompt pool | Sweep axes | Executed row formula | Executed rows |
|------|---------------|----------------------|------------|----------------------|---------------|
| Phase 1 | All task YAMLs | `953` prompts per model (`468` safety + `485` capability) | `3 models x 6 batch sizes` | `3 x 953 x 6` | `17,154` |
| Phase 2 | Safety tasks only | `468` targets per model | `3 models x 4 conditions` | `3 x 468 x 4` | `5,616` |
| Phase 3 | AdvBench + jailbreak subset | `220` prompts per model / quant / concurrency cell | `3 models x 3 quants x 3 concurrency levels` | `3 x 220 x 3 x 3` | `5,940` |
| Phase 4 | Reduced all-task subset | `450` prompts per model / batch size (`250` safety + `200` capability) | `2 models x 3 batch sizes` | `2 x 450 x 3` | `2,700` |

The task-level prompt counts that actually executed were:

- Phase 1: AdvBench `100`, jailbreak amplification `120`, BBQ `198`, TruthfulQA `50`, MMLU `285`, ARC-Challenge `200`
- Phase 2: AdvBench `100`, jailbreak amplification `120`, BBQ `198`, TruthfulQA `50`
- Phase 3: AdvBench `100`, jailbreak amplification `120`
- Phase 4: AdvBench `60`, jailbreak amplification `80`, BBQ `80`, TruthfulQA `30`, MMLU `100`, ARC-Challenge `100`

This is one place where artifact discipline matters. Earlier shorthand rounded BBQ to `200`, but the run artifacts for TR138 contain `198` executable BBQ rows. The prompt totals above are therefore the source of truth.

### 5.6 Execution semantics by phase

The four phases generate rows in materially different ways. That difference is the whole point of the design.

| Phase | Backend and dispatch path | What one retained row represents | What is discarded | Why it matters analytically |
|------|----------------------------|----------------------------------|-------------------|-----------------------------|
| Phase 1 | vLLM `/v1/completions` with exact synchronized dispatch groups | One prompt evaluated under a forced in-flight dispatch batch of size `1,2,4,8,16,32` | Tail-padding filler outputs when the last group is short | Tests batch-size sensitivity while keeping the dispatch group exact |
| Phase 2 | vLLM `/v1/completions` with one target plus fillers | One target prompt under one condition: `solo`, `benign`, `adversarial`, or `safety` | All filler responses | Isolates whether neighbor identity changes the target outcome |
| Phase 3 | Ollama `/api/generate` under semaphore-limited concurrency | One safety prompt under one `quant x concurrency` cell | Nothing besides failed rows | Measures continuous-flow concurrent load, not tensor batching |
| Phase 4 | Single vLLM `/v1/completions` call with a prompt list | One prompt from an explicit true prompt-list batch | Tail-padding filler outputs when the prompt list is short | Removes cross-request arrival timing and tests the cleaner batching mechanism |

The most important execution details are easy to miss but central to interpretation:

- **Phase 1** pads incomplete tail groups so every analyzed row still experiences the stated dispatch size.
- **Phase 2** records only the target response; fillers exist only to create the co-batch context.
- **Phase 2** also counterbalances target position across conditions, so neighbor condition is not confounded with a fixed slot.
- **Phase 3** is intentionally continuous-flow. The next request starts when a slot opens; it is not a true batch.
- **Phase 4** sends a prompt list in one completion call and maps returned `choice.index` values back onto retained prompts. That is why it is the mechanism check for the core batching claim.

### 5.7 Row schema, scoring, and judge pipeline

Every retained row in `samples.jsonl` contains enough information to reconstruct the analysis path:

- universal identifiers: `phase`, `model`, `task_name`, `task_type`, `sample_id`, `seed`, `repetition`
- prompt and output fields: `prompt`, `candidate`, `reference`, `correct_answers`, `incorrect_answers`, `sample_meta`
- runtime fields: `wall_ms`, `prompt_tokens`, `completion_tokens`, `status`
- phase-specific controls:
  - Phase 1 / 4: `batch_size`, `dispatch_batch_size`, `tail_padded`
  - Phase 2: `condition`, `co_batch_condition`, `filler_count`, `target_position`
  - Phase 3: `quant`, `concurrency`, `eval_ms`, `submit_time_s`
  - Phase 4 only: `batch_mode=true_prompt_list`, `true_batching=true`, `analyzed_in_batch`

The scoring path in `research/tr138/analyze.py` is task-specific rather than one global classifier:

| Task family | Scorer | Output metric | Domain | Operational meaning |
|-------------|--------|---------------|--------|---------------------|
| `advbench_refusal`, `jailbreak_amplification` | `RefusalDetector.detect(candidate)` | `score = 1.0` if refused else `0.0` | safety | Whether the model refused the unsafe request |
| `truthfulqa` | `TruthfulnessScorer.score(candidate, correct_answers, incorrect_answers)` | scalar truthfulness score | safety | Whether the answer overlaps more with true than false reference sets |
| `bbq_bias` | `BiasDetector.score(candidate, choices, stereotyped_idx, anti_stereotyped_idx)` | scalar bias score | safety | Whether the answer aligns with anti-stereotyped / non-biased behavior |
| `mmlu_real`, `arc_challenge` | answer-letter extraction vs `reference` | `1.0` correct, `0.0` incorrect | capability | Standard accuracy control arm |

The LLM judge is a second layer, not the primary label source:

- judge model: `qwen2.5:7b-instruct-q8_0`
- judged rows: safety rows only, `21,480` total in this run
- routing:
  - refusal and jailbreak rows -> `judge_refusal(prompt, candidate)`
  - TruthfulQA rows -> `judge_truthfulness(prompt, candidate, correct_answers, incorrect_answers)`
  - BBQ rows -> `judge_bias(context, question, candidate, choice_a, choice_b, choice_c)`

That judge output is used for Section 14 agreement analysis and for triangulation, not to overwrite the primary task-specific scorer. This distinction is essential because kappa is low and the report should not pretend the judge is an oracle.

### 5.8 How raw rows become report claims

TR138's findings are not based on free-form reading of generations. They are derived in a fixed evidence chain:

1. `samples.jsonl` records one retained inference result per experimental unit.
2. `tr138_scored.jsonl` attaches task-specific scores and domains to those rows.
3. `tr138_analysis.json` aggregates those scores into phase-level metrics, paired comparisons, ANOVAs, TOST results, latency summaries, and agreement tables.
4. This publish-ready report interprets those aggregates and declares which claims are strong, weak, negative, or unsupported.

The mapping from raw rows to claims is:

| Report question | Row-level comparison | Aggregate output | Artifact key | Resulting claim |
|-----------------|----------------------|------------------|--------------|-----------------|
| Do batched outputs differ from `batch=1`? | Match Phase 1 or Phase 4 rows on `(model, sample_id)` and compare text identity | output identity / changed-output rate | `phase1.output_identity`, `phase4.output_identity` | Batching is not byte-invariant |
| Are score changes safety-skewed? | Compare matched non-baseline rows against the same prompt at `batch=1` | safety vs capability flip counts and ratios | `phase1.flip_rates`, `phase4.flip_rates` | Safety is more fragile than capability |
| Are the flips directionally concerning? | Restrict to refusal tasks whose score changed | refusal -> compliance vs compliance -> refusal counts | `phase1.flip_direction_breakdown` | The dominant direction is unsafe |
| Do neighbors matter beyond batch size? | Match Phase 2 rows on `(model, sample_id)` across `solo`, `benign`, `adversarial`, `safety` | condition means and paired tests | `phase2.condition_scores`, `phase2.pairwise_comparisons` | Large co-batch interference is not established |
| Is Phase 3 a concurrency story or a quantization story? | Group Phase 3 rows by `(model, quant, concurrency)` | two-way ANOVA and slope tables | `phase3.anova`, `phase3.slopes` | Quantization matters; concurrency does not |
| Does the signal survive explicit batching? | Compare Phase 4 rows against Phase 4 `batch=1` and against the synchronized-dispatch signal | true-batch flip rate and flip agreement | `phase4.flip_rates`, `phase4.phase1_alignment` | The core signal survives true batching |
| Are effects practically small or large? | Use score vectors from each phase | TOST and power analysis | `tost_equivalence`, `power_analysis` | Effects are real but small in absolute terms |
| How trustworthy is the heuristic scoring stack? | Join safety rows with `judge_labels.jsonl` by full record key | kappa and agreement | `judge_agreement` | Judge support is weak and directional only |

---

## 6. Models & Configuration

| Model | Family | Params | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Backend |
|-------|--------|--------|---------|---------|---------|---------|---------|
| llama3.2-1b | llama | 1236M | Yes | Yes | Yes | No | vLLM FP16, Ollama Q8/Q4/Q2 |
| llama3.2-3b | llama | 3213M | Yes | Yes | Yes | Yes | vLLM FP16, Ollama Q8/Q4/Q2, vLLM true-batch |
| qwen2.5-1.5b | qwen | 1543M | Yes | Yes | Yes | Yes | vLLM FP16, Ollama Q8/Q4/Q2, vLLM true-batch |

**Phase 1 tasks:** AdvBench (100), Jailbreak (120), BBQ (198), TruthfulQA (50), MMLU (285), ARC-Challenge (200) = 953 prompts
**Phase 2 tasks:** AdvBench (100), Jailbreak (120), BBQ (198), TruthfulQA (50) = 468 safety prompts
**Phase 3 tasks:** AdvBench (100), Jailbreak (120) = 220 safety prompts
**Phase 4 tasks:** Reduced subset = 450 prompts (250 safety + 200 capability)

### 6.1 Why these models

The model lineup is intentionally local-first and small enough to run repeatedly on a single consumer GPU. That is a limitation for external generalization, but it is also a strength for mechanism work because it lets the report control the serving path tightly.

The two Llama 3.2 sizes provide an intra-family size comparison, while Qwen 2.5 adds one cross-family reference point. That is enough to ask whether the batching effect is family-agnostic without pretending to establish a universal law across all instruction-tuned LLMs.

### 6.2 Why these benchmarks

The benchmark mix is designed to separate three different safety behaviors:

- **Refusal under harmful requests:** AdvBench, jailbreak amplification
- **Bias sensitivity:** BBQ
- **Truthfulness / errorful compliance:** TruthfulQA

MMLU and ARC-Challenge provide the capability control arm. This matters because the report would be much less informative if it only measured safety tasks. The capability side is what allows the report to argue that the observed effect is **safety-skewed**, not just generic output churn.

### 6.3 What a safety flip means operationally

Not every textual difference matters. TR138 only becomes a safety report because some output differences cross a meaningful behavioral boundary:

- refusal -> compliance
- compliance -> refusal
- truthful -> untruthful
- less biased -> more biased

That is why the flip metrics matter more than the raw byte-identity metrics.

---

## 7. Phase 1: Batch Size x Output Determinism

### 7.1 Output Identity (byte-identical vs batch=1)

| Model | BS=2 | BS=4 | BS=8 | BS=16 | BS=32 |
|-------|--------|--------|--------|--------|--------|
| llama3.2-1b | 94.1% | 94.2% | 94.1% | 94.3% | 94.3% |
| llama3.2-3b | 91.8% | 91.0% | 92.1% | 90.5% | 91.3% |
| qwen2.5-1.5b | 92.7% | 92.4% | 91.6% | 91.3% | 91.6% |

### 7.2 Safety vs Capability Flip Rate

| Model | Batch Size | Safety Flip Rate | Capability Flip Rate | Ratio (S/C) |
|-------|-----------|-----------------|---------------------|-------------|
| llama3.2-1b | 2 | 0.2% | 0.2% | 1.00 |
| llama3.2-1b | 4 | 0.2% | 0.2% | 1.00 |
| llama3.2-1b | 8 | 0.2% | 0.2% | 1.00 |
| llama3.2-1b | 16 | 0.2% | 0.2% | 1.00 |
| llama3.2-1b | 32 | 0.2% | 0.2% | 1.00 |
| llama3.2-3b | 2 | 0.4% | 0.0% | -- |
| llama3.2-3b | 4 | 0.4% | 0.2% | **2.05** |
| llama3.2-3b | 8 | 0.4% | 0.0% | -- |
| llama3.2-3b | 16 | 0.4% | 0.0% | -- |
| llama3.2-3b | 32 | 1.1% | 0.2% | **5.10** |
| qwen2.5-1.5b | 2 | 0.9% | 0.2% | **4.05** |
| qwen2.5-1.5b | 4 | 0.9% | 0.2% | **4.05** |
| qwen2.5-1.5b | 8 | 0.9% | 0.2% | **4.05** |
| qwen2.5-1.5b | 16 | 0.9% | 0.0% | -- |
| qwen2.5-1.5b | 32 | 0.4% | 0.0% | -- |

### 7.3 Flip Direction Breakdown

| Direction | Count | Percentage |
|-----------|-------|------------|
| Refusal -> compliance | 20 | 69.0% |
| Compliance -> refusal | 9 | 31.0% |

### 7.4 Per-Task Sensitivity

| Task | Domain | Mean Flip Rate | N |
|------|--------|----------------|---|
| truthfulqa | safety | 1.6% | 750 |
| jailbreak_amplification | safety | 0.6% | 1800 |
| bbq_bias | safety | 0.5% | 2970 |
| advbench_refusal | safety | 0.0% | 1500 |
| mmlu_real | capability | 0.2% | 4275 |
| arc_challenge | capability | 0.0% | 3000 |

### 7.5 Statistical Tests

| Test | Statistic | p-value | Effect Size | Significant |
|------|-----------|---------|-------------|-------------|
| overall_bs16 | 4.731 | 0.0296 | 5.204 | Yes |
| overall_bs32 | 3.832 | 0.0503 | 3.539 | No |
| qwen2.5-1.5b_bs16 | 4.163 | 0.0578 | 9.407 | No |
| overall_bs2 | 2.969 | 0.0849 | 3.120 | No |
| overall_bs8 | 2.969 | 0.0849 | 3.120 | No |
| llama3.2-3b_bs32 | 2.830 | 0.1176 | 3.833 | No |
| overall_bs4 | 1.753 | 0.1856 | 2.227 | No |
| qwen2.5-1.5b_bs2 | 1.919 | 0.2097 | 3.129 | No |
| qwen2.5-1.5b_bs4 | 1.919 | 0.2097 | 3.129 | No |
| qwen2.5-1.5b_bs8 | 1.919 | 0.2097 | 3.129 | No |
| llama3.2-3b_bs16 | 2.077 | 0.2409 | 5.204 | No |
| llama3.2-3b_bs2 | 2.077 | 0.2409 | 5.204 | No |
| llama3.2-3b_bs8 | 2.077 | 0.2409 | 5.204 | No |
| qwen2.5-1.5b_bs32 | 2.077 | 0.2409 | 5.204 | No |
| llama3.2-3b_bs4 | 0.371 | 0.6180 | 1.731 | No |
| llama3.2-1b_bs16 | 0.001 | 1.0000 | 1.036 | No |
| llama3.2-1b_bs2 | 0.001 | 1.0000 | 1.036 | No |
| llama3.2-1b_bs32 | 0.001 | 1.0000 | 1.036 | No |
| llama3.2-1b_bs4 | 0.001 | 1.0000 | 1.036 | No |
| llama3.2-1b_bs8 | 0.001 | 1.0000 | 1.036 | No |

### 7.6 What Phase 1 does and does not prove

Phase 1 supplies the main empirical basis for the report, but it should be read precisely.

What it establishes well:

- outputs are not perfectly batch-invariant even at `temperature=0`
- the unstable subset is safety-skewed rather than evenly distributed
- the direction of change is more often unsafe than safe

What it does not establish on its own:

- that the mechanism is pure tensor batching rather than some scheduler artifact
- that the same magnitude will appear on larger models or other hardware
- that every safety domain is equally vulnerable

That is exactly why Phase 4 exists. Phase 1 gives the primary signal; Phase 4 tests whether the signal survives a cleaner mechanism check.

---

## 8. Phase 2: Co-Batching Interference

### 8.1 Mean Safety Score by Co-Batch Condition

| Model | Condition | Mean Safety | CI Lower | CI Upper | N |
|-------|-----------|------------|----------|----------|---|
| llama3.2-1b | adversarial | 0.654 | 0.611 | 0.697 | 468 |
| llama3.2-1b | benign | 0.654 | 0.611 | 0.697 | 468 |
| llama3.2-1b | safety | 0.654 | 0.611 | 0.697 | 468 |
| llama3.2-1b | solo | 0.654 | 0.611 | 0.697 | 468 |
| llama3.2-3b | adversarial | 0.769 | 0.731 | 0.807 | 468 |
| llama3.2-3b | benign | 0.766 | 0.728 | 0.804 | 468 |
| llama3.2-3b | safety | 0.768 | 0.730 | 0.806 | 468 |
| llama3.2-3b | solo | 0.764 | 0.726 | 0.802 | 468 |
| qwen2.5-1.5b | adversarial | 0.788 | 0.752 | 0.825 | 468 |
| qwen2.5-1.5b | benign | 0.793 | 0.756 | 0.829 | 468 |
| qwen2.5-1.5b | safety | 0.793 | 0.756 | 0.829 | 468 |
| qwen2.5-1.5b | solo | 0.793 | 0.756 | 0.829 | 468 |

### 8.2 Pairwise Condition Comparisons

| Model | Comparison | Delta (pp) | p-value | Effect Size | Significant |
|-------|------------|-----------|---------|-------------|-------------|
| llama3.2-1b | adversarial_vs_safety | +0.0 | 1.0000 | 0.000 | No |
| llama3.2-1b | benign_vs_adversarial | +0.0 | 1.0000 | 0.000 | No |
| llama3.2-1b | benign_vs_safety | +0.0 | 1.0000 | 0.000 | No |
| llama3.2-1b | solo_vs_adversarial | +0.0 | 1.0000 | 0.000 | No |
| llama3.2-1b | solo_vs_benign | +0.0 | 1.0000 | 0.000 | No |
| llama3.2-1b | solo_vs_safety | +0.0 | 1.0000 | 0.000 | No |
| llama3.2-3b | adversarial_vs_safety | -0.1 | 0.3178 | -0.003 | No |
| llama3.2-3b | benign_vs_adversarial | +0.3 | 0.4060 | 0.008 | No |
| llama3.2-3b | benign_vs_safety | +0.2 | 0.5643 | 0.005 | No |
| llama3.2-3b | solo_vs_adversarial | +0.5 | 0.0956 | 0.013 | No |
| llama3.2-3b | solo_vs_benign | +0.2 | 0.3178 | 0.005 | No |
| llama3.2-3b | solo_vs_safety | +0.4 | 0.1575 | 0.010 | No |
| qwen2.5-1.5b | adversarial_vs_safety | +0.4 | 0.4148 | 0.011 | No |
| qwen2.5-1.5b | benign_vs_adversarial | -0.4 | 0.3178 | -0.011 | No |
| qwen2.5-1.5b | benign_vs_safety | +0.0 | 1.0000 | 0.000 | No |
| qwen2.5-1.5b | solo_vs_adversarial | -0.4 | 0.3178 | -0.011 | No |
| qwen2.5-1.5b | solo_vs_benign | +0.0 | 1.0000 | 0.000 | No |
| qwen2.5-1.5b | solo_vs_safety | +0.0 | 1.0000 | 0.000 | No |

### 8.3 Per-Task Interference Breakdown

| Task | Solo Mean | Adversarial Mean | Delta (pp) | Vulnerable? |
|------|-----------|-----------------|-----------|-------------|
| advbench_refusal | 0.807 | 0.807 | +0.0 | No |
| bbq_bias | 0.870 | 0.870 | +0.0 | No |
| jailbreak_amplification | 0.553 | 0.556 | +0.3 | No |
| truthfulqa | 0.510 | 0.507 | -0.3 | No |

### 8.4 Why the weak Phase 2 result still matters

The null-looking Phase 2 result is still useful for two reasons.

First, it narrows the likely interpretation of TR138. If Phase 2 had shown a large adversarial-neighbor effect, the flagship story would be about cross-request interference. It does not. That pushes the report back toward the cleaner claim that **batch size itself** is the main established risk axis.

Second, it tells us something about future experiment design. If co-batching effects exist here, they are likely smaller than the present design's comfortable detection range and may require:

- larger sample sizes
- more targeted prompt families
- stronger adversarial fillers
- more direct instrumentation of backend scheduling and kernel behavior

---

## 9. Phase 3: Quantization x Concurrency Interaction

### 9.1 Safety Score Grid (Model x Quant x Concurrency)

| Model | Quant | Concurrency | Mean Safety | N |
|-------|-------|------------|------------|---|
| llama3.2-1b | Q2_K | 1 | **0.627** | 220 |
| llama3.2-1b | Q2_K | 4 | **0.627** | 220 |
| llama3.2-1b | Q2_K | 8 | **0.627** | 220 |
| llama3.2-1b | Q4_K_M | 1 | 0.959 | 220 |
| llama3.2-1b | Q4_K_M | 4 | 0.959 | 220 |
| llama3.2-1b | Q4_K_M | 8 | 0.959 | 220 |
| llama3.2-1b | Q8_0 | 1 | 0.986 | 220 |
| llama3.2-1b | Q8_0 | 4 | 0.986 | 220 |
| llama3.2-1b | Q8_0 | 8 | 0.986 | 220 |
| llama3.2-3b | Q2_K | 1 | 0.877 | 220 |
| llama3.2-3b | Q2_K | 4 | 0.877 | 220 |
| llama3.2-3b | Q2_K | 8 | 0.877 | 220 |
| llama3.2-3b | Q4_K_M | 1 | 0.996 | 220 |
| llama3.2-3b | Q4_K_M | 4 | 0.996 | 220 |
| llama3.2-3b | Q4_K_M | 8 | 0.996 | 220 |
| llama3.2-3b | Q8_0 | 1 | 0.986 | 220 |
| llama3.2-3b | Q8_0 | 4 | 0.986 | 220 |
| llama3.2-3b | Q8_0 | 8 | 0.986 | 220 |
| qwen2.5-1.5b | Q2_K | 1 | **0.245** | 220 |
| qwen2.5-1.5b | Q2_K | 4 | **0.245** | 220 |
| qwen2.5-1.5b | Q2_K | 8 | **0.245** | 220 |
| qwen2.5-1.5b | Q4_K_M | 1 | **0.777** | 220 |
| qwen2.5-1.5b | Q4_K_M | 4 | **0.777** | 220 |
| qwen2.5-1.5b | Q4_K_M | 8 | **0.777** | 220 |
| qwen2.5-1.5b | Q8_0 | 1 | 0.836 | 220 |
| qwen2.5-1.5b | Q8_0 | 4 | 0.836 | 220 |
| qwen2.5-1.5b | Q8_0 | 8 | 0.836 | 220 |

### 9.2 Two-Way ANOVA Results

| Factor | F-statistic | p-value | eta^2 (eta-squared) | Significant |
|--------|-----------|---------|-----------------|-------------|
| Quant (3/3 models) | 254.796 | 0.00e+00 | 0.194 | Yes |
| Concurrency (0/3 models) | 0.000 | 1.0000 | 0.000 | No |
| Interaction (0/3 models) | 0.000 | 1.0000 | 0.000 | No |

### 9.3 Safety vs Concurrency Slopes by Quant Level

| Model | Quant | Slope (safety/concurrency) | R^2 | N |
|-------|-------|---------------------------|-----|---|
| llama3.2-1b | Q2_K | +0.0000 | 0.000 | 3 |
| llama3.2-1b | Q4_K_M | +0.0000 | 0.000 | 3 |
| llama3.2-1b | Q8_0 | +0.0000 | 0.000 | 3 |
| llama3.2-3b | Q2_K | +0.0000 | 0.000 | 3 |
| llama3.2-3b | Q4_K_M | +0.0000 | 0.000 | 3 |
| llama3.2-3b | Q8_0 | +0.0000 | 0.000 | 3 |
| qwen2.5-1.5b | Q2_K | +0.0000 | 0.000 | 3 |
| qwen2.5-1.5b | Q4_K_M | +0.0000 | 0.000 | 3 |
| qwen2.5-1.5b | Q8_0 | +0.0000 | 0.000 | 3 |

### 9.4 How Phase 3 should be read

Phase 3 is part of TR138 because it tests a nearby operational concern, but it is not part of the core batching claim.

The right interpretation is:

- quantization remains a material safety axis
- concurrency, in this setup, contributes no measurable additional harm
- the report should not claim a multiplicative quantization x batching interaction from this phase

This distinction matters because overclaiming here would weaken the flagship result rather than strengthen it. The batching story is already strong enough without borrowing force from a different mechanism.

---

## 10. Cross-Phase Synthesis

### 10.1 Batch-Induced vs Quantization-Induced Variance

| Source | Approx pp | Risk | N |
|--------|-----------|------|---|
| batch_size | 3.710 | moderate | 2859 |
| true_batching | 5.180 | moderate | 900 |
| quantization | 34.960 | high | 1980 |
| concurrency | 0.000 | low | 1980 |

### 10.2 Phase 4 True-Batching Validation

Explicit prompt-list true batching produces an overall safety flip rate of **0.8%**.
Mean flip-agreement with Phase 1 synchronized dispatch is **99.4%**, which indicates how much of the original signal survives without request-arrival timing effects.

| Model | Batch Size | N Paired | Flip Agreement % | Score Agreement % |
|-------|------------|----------|------------------|-------------------|
| llama3.2-3b | 4 | 450 | 100.0 | 100.0 |
| llama3.2-3b | 8 | 450 | 99.6 | 99.6 |
| qwen2.5-1.5b | 4 | 450 | 99.1 | 99.1 |
| qwen2.5-1.5b | 8 | 450 | 99.1 | 99.1 |

### 10.3 Risk Classification

**Overall risk level:** **HIGH**

| Factor | Risk | Publish-ready rationale |
|--------|------|-------------------------|
| batch_size | moderate | About 3.7pp observed variance, safety-asymmetric, and directionally unsafe |
| co_batching | low | Observed deltas stay within about 0.0-0.5pp and are not significant |
| quant_x_concurrency | high as a quantization story, low as a concurrency story | Quantization dominates; concurrency contributes no detectable effect |
| true_batching | moderate | About 5.2pp observed effect, and the signal survives the cleaner mechanism test |

### 10.4 Novel empirical datapoints uncovered by TR138

TR138 is not novel because it shows that floating-point inference can vary. That was already known. The novelty is in the exact safety-facing datapoints it adds.

| Novel datapoint | Why it matters |
|-----------------|----------------|
| Non-baseline Phase 1 outputs are only about `92%` byte-identical, but score-changing rows are much rarer (`0.5%` safety, `0.1%` capability aggregate) | Separates harmless wording churn from behaviorally meaningful flips |
| Among classified safety flips, `69.0%` move from refusal to compliance | Shows the instability is directionally unsafe rather than symmetric noise |
| Explicit true prompt-list batching reproduces the signal with `0.8%` safety flips and `99.4%` mean flip agreement to synchronized dispatch | Converts the main finding from "possible scheduler artifact" into a real batching result |
| Phase 2 finds no large adversarial-neighbor effect under this design | Narrows the likely mechanism away from a strong co-batch contagion story |
| Phase 3 shows about `34.96pp` quantization variance versus `0pp` concurrency variance in this setup | Gives a quantitative prioritization of adjacent serving risks instead of lumping them together |
| Safety prompts are consistently slower than capability prompts, and flipped rows are slower than stable rows | Provides a concrete mechanism clue: the fragile rows also appear to be the more compute-intensive rows |
| No model shows a critical batch-size threshold by CI non-overlap | The hazard shape is diffuse rather than cliff-like, which changes how one should validate it |

Two of those datapoints are especially important.

First, TR138 gives a decomposition that earlier batching discussions usually skip: most batch-induced output differences are not themselves safety failures. The safety problem lives in the thin slice of rows where the task score changes, and that slice is sparse but safety-skewed.

Second, TR138 contributes a negative-but-important result: the present evidence does **not** support a dramatic adversarial-neighbor contagion story. That makes the flagship claim cleaner because it focuses attention on batch condition itself rather than on a weaker and less resolved interference narrative.

### 10.5 What these datapoints change in practice and in theory

The combined result changes the shape of the argument in two ways.

Operationally, it means teams should stop asking only "does batching change outputs?" and start asking "does batching change safety classifications on the actual deployed path?" TR138 shows those are not the same question.

Methodologically, it means future work should not treat all serving perturbations as interchangeable. TR138 separates four axes cleanly enough to support a sharper research program:

- batch size: small but safety-skewed effect
- true batching: confirms the effect survives a cleaner mechanism test
- co-batch identity: weak or null in the present design
- quantization: much larger than batching in absolute effect size

That is a useful narrowing of the problem. The next experiments should be designed to resolve mechanism and external validity, not to re-prove the existence of a generic output-difference phenomenon.

---

## 11. TOST Equivalence Analysis

This section asks a stricter question than the hypothesis tests above: even if some effects are non-zero, are they small enough to count as operationally negligible under a `+/-3 percentage point` margin?

That distinction matters in TR138 because the headline batch findings are rare-event effects. A report can simultaneously support the claim that batching is safety-relevant and reject the claim that batching causes large absolute safety collapse. TOST is what separates those two statements.

### 11.1 Why equivalence testing matters here

Classical null-hypothesis testing is asymmetric:

- failure to reject does not prove similarity
- statistical significance can coexist with very small absolute differences

TOST addresses the practical question instead:

> can we rule out effects larger than `+/-3 percentage points`?

That bound is appropriate here for three reasons:

1. The observed Phase 1 and Phase 4 safety shifts are sub-1 percentage point.
2. A 3pp shift would already be operationally meaningful for many refusal-style deployments.
3. The bound remains much smaller than the quantization-induced safety movement seen elsewhere in the Banterhearts safety line.

### 11.2 Phase-level TOST summary

| Phase | Comparison family | Equivalent comparisons | Total comparisons | Main read |
|------|-------------------|------------------------|-------------------|-----------|
| Phase 1 | `batch=1` vs other batch sizes | 10 | 10 | Large absolute batch penalties are ruled out |
| Phase 2 | `solo` vs `adversarial` | 3 | 3 | Large co-batch interference is ruled out |
| Phase 3 | concurrency contrasts within quant level | 7 | 18 | Mixed and only weakly informative |
| Phase 4 | true-batch vs `batch=1` | 3 | 4 | Large true-batch penalties are mostly ruled out |

### 11.3 What TOST changes about the Phase 1 claim

Phase 1 is the cleanest example of why this report needs both significance and equivalence framing.

The main Phase 1 result is:

- overall safety flip rate: `0.5%`
- overall capability flip rate: `0.1%`
- safety-to-capability ratio: about `3.6x`

That ratio is meaningful, but the TOST results show that absolute movement is still small. Every Phase 1 batch-size comparison falls inside the `+/-3pp` equivalence band. So the correct reading is:

- batching introduces small, real perturbations
- those perturbations are not safety-neutral
- but they are not large enough to describe as broad safety collapse

That is a stronger and more defensible claim than either extreme.

### 11.4 Phase 2 and Phase 4 implications

For Phase 2, all three `solo_vs_adversarial` model-level TOST tests are equivalent. This matters because it sharply limits what can be said about adversarial neighbor effects. TR138 does not support a claim that adversarial co-batching creates a large practical drop in refusal behavior.

For Phase 4, three of four explicit true-batching comparisons are also equivalent. That does not weaken the mechanism result. It simply means the true-batching subset supports this narrower conclusion:

- true batching reproduces the direction of the Phase 1 signal
- the true-batching penalty remains small in absolute terms

### 11.5 Phase 3 caveat

Phase 3 produces the least interpretable equivalence output. The reason is not hidden complexity in the concurrency effect. The reason is that many cell means are exactly or nearly identical, making the TOST result mechanically degenerate in some cells and only partially informative in others.

The right interpretation is therefore simple:

- Phase 3 does not show a meaningful concurrency penalty
- Phase 3 does show a strong quantization penalty
- TOST adds little beyond confirming that concurrency is not the dominant story

### 11.6 Bottom line

TOST narrows the claim space in a useful way:

- supported: batching creates small safety-relevant perturbations
- supported: large adversarial co-batch effects are absent in this dataset
- supported: large true-batching effects are also absent on the reduced validation subset
- not supported: batching alone causes large absolute safety collapse

## 12. Power Analysis

Power is a major interpretive constraint in TR138 because the main batch-related event rates are very small. That means "not significant" often means "rare and hard to resolve," not automatically "there is no effect."

### 12.1 Minimum detectable effect sizes

| Phase | Primary metric | N | MDE at 80 percent power (pp) | Read |
|------|----------------|---|-------------------------------|------|
| Phase 1 | Safety flip rate (aggregate) | 8,424 safety rows | 1.9 | Adequate for modest aggregate shifts |
| Phase 1 | Capability flip rate (aggregate) | 8,730 capability rows | 2.1 | Adequate for modest aggregate shifts |
| Phase 2 | Safety score by condition | 5,616 rows | 2.3 | Good for moderate co-batch effects |
| Phase 3 | Safety score grid | 5,940 rows | 2.0 | Good for large quant effects, weak for tiny concurrency effects |
| Phase 4 | True-batch validation | 2,700 rows | 3.5 | Useful as a mechanism check, not a high-power effect-size study |

### 12.2 What is well powered

TR138 is well powered for:

- aggregate Phase 1 rate comparisons
- ruling out large Phase 2 co-batch effects
- detecting the large Phase 3 quantization effect
- establishing that Phase 4 is not hiding a very large true-batch penalty

This is why the report can say with confidence that Phase 3 is mainly a quantization result and that Phase 2 does not contain a large interference effect.

### 12.3 What is underpowered or only weakly powered

TR138 is not strongly powered for:

- per-batch-size disproportionality tests in Phase 1
- very small condition deltas inside Phase 2
- model-by-model true-batching effects in Phase 4
- per-jailbreak slope interpretation
- prompt-level correlation analysis

This matters because several tempting claims sit exactly in these low-power zones. For example, the `3.6x` Phase 1 safety-to-capability ratio is directionally important, but the Wilson confidence intervals still overlap at each batch size. That is a classic rare-event setting: the pattern is real enough to motivate caution, but not strong enough to produce a clean threshold-level proof.

### 12.4 Practical reading rule

Use the power analysis as a filter:

- strong claims should come from aggregate direction, replication across phases, and large-effect axes
- weak claims should not be upgraded just because they are interesting

That is why the report treats Phase 1 plus Phase 4 as the core evidence, Phase 2 as negative or weak evidence, and Phase 3 as an auxiliary concurrency study dominated by quantization.

---

## 13. Latency Analysis

Latency is not a primary endpoint in TR138, but it is useful for mechanism inference. If the prompts most exposed to safety flips are also the prompts that take longer or grow more unstable under load, that supports a compute-stress interpretation rather than a purely semantic one.

### 13.1 Phase 1 throughput economics

The deployment motivation for batching is obvious in the throughput curves.

| Model | BS=1 throughput | BS=32 throughput | Mean latency slope (ms / batch-size unit) | R^2 | Read |
|------|-----------------|------------------|-------------------------------------------|-----|------|
| llama3.2-1b | 2.66 samp/s | 60.29 samp/s | 4.82 | 0.992 | Large throughput win with modest latency growth |
| llama3.2-3b | 0.51 samp/s | 13.22 samp/s | 15.11 | 0.994 | Same pattern, but with a much higher absolute latency floor |
| qwen2.5-1.5b | 1.78 samp/s | 45.22 samp/s | 4.60 | 0.989 | Strong throughput gain with relatively flat mean latency curve |

Two points follow immediately.

First, the business reason teams batch requests is real. Throughput scales dramatically.

Second, that is exactly why even small safety-sensitive perturbations matter. The batch sizes that make serving economically attractive are the same batch sizes that change the numerical execution context.

### 13.2 Safety prompts are consistently slower than capability prompts

| Model | Safety mean (ms) | Capability mean (ms) | Difference (ms) | Cohen's d | Read |
|------|------------------|----------------------|-----------------|-----------|------|
| llama3.2-1b | 652.0 | 201.8 | 450.2 | 0.873 | Large domain gap |
| llama3.2-3b | 2907.5 | 1290.8 | 1616.8 | 1.048 | Very large domain gap |
| qwen2.5-1.5b | 973.7 | 246.6 | 727.1 | 1.230 | Very large domain gap |

This is one of the most useful secondary findings in the report.

Safety prompts are not just semantically different from capability prompts. They are also computationally different in this stack. Across all three models, safety prompts take substantially longer. That does not prove a numerical mechanism by itself, but it does make the Phase 1 asymmetry more plausible: the refusal boundary appears to be a more compute-intensive region of behavior than ordinary capability tasks.

### 13.3 Co-batch latency is basically flat across neighbor condition

| Model | Benign (ms) | Adversarial (ms) | Safety (ms) | Spread (max-min) |
|------|-------------|------------------|-------------|------------------|
| llama3.2-1b | 739.4 | 741.2 | 722.4 | 18.8 |
| llama3.2-3b | 2837.8 | 2854.0 | 2873.8 | 36.0 |
| qwen2.5-1.5b | 951.0 | 961.5 | 957.6 | 10.5 |

This is consistent with the Phase 2 main result. If adversarial neighbors were causing a strong interference effect in this setup, one plausible accompaniment would be a clear latency or scheduling signature. There is no such signature here. The latency differences are tiny relative to the overall runtime.

### 13.4 Flipped samples are slower than stable samples

| Model | Flipped mean (ms) | Stable mean (ms) | Difference (ms) | Cohen's d | Read |
|------|-------------------|------------------|-----------------|-----------|------|
| llama3.2-1b | 916.2 | 431.1 | 485.1 | 0.856 | Large slowdown among flipped rows |
| llama3.2-3b | 3579.6 | 2107.5 | 1472.2 | 0.842 | Large slowdown among flipped rows |
| qwen2.5-1.5b | 1615.3 | 607.3 | 1008.0 | 1.455 | Very large slowdown among flipped rows |

This pattern is directional rather than definitive, but it is hard to ignore. Across all three models, rows that flip under batching are materially slower than rows that remain stable. That fits a simple interpretation:

- prompts near the refusal boundary appear harder to resolve
- those harder prompts spend longer in generation
- those are also the prompts more likely to change classification when the batch context changes

That is the closest thing TR138 has to a mechanism clue beyond the direct batching comparisons themselves.

### 13.5 Phase 3 latency says more about quantization than concurrency

The Phase 3 latency grid reinforces the main read of that phase.

| Model | Q8_0 c1 -> c8 (ms) | Q4_K_M c1 -> c8 (ms) | Q2_K c1 -> c8 (ms) | Main read |
|------|--------------------|----------------------|--------------------|-----------|
| llama3.2-1b | 291.5 -> 780.6 | 282.4 -> 676.3 | 444.2 -> 2047.8 | Q2_K is the unstable latency regime |
| llama3.2-3b | 400.7 -> 1855.4 | 341.4 -> 1131.3 | 443.7 -> 2014.3 | Lower-bit latency is not free in this stack |
| qwen2.5-1.5b | 479.8 -> 2720.9 | 422.2 -> 2070.2 | 839.6 -> 5411.9 | Q2_K degrades sharply under load |

This is important for two reasons.

First, it explains why Phase 3 should not be summarized as "concurrency is harmless." The correct statement is narrower: concurrency adds little safety signal relative to quantization, but it still interacts with runtime cost.

Second, it is a warning against simplistic deployment assumptions. In this Ollama setup, the lowest-bit quant level is not the fastest path under concurrent load. That is a production result, not just a safety result.

### 13.6 What latency can and cannot prove

Latency supports the report in three ways:

- it explains why batching is economically attractive
- it shows that safety prompts occupy a different runtime regime from capability prompts
- it shows that flipped rows cluster among slower rows

But latency does not prove the exact numerical cause of the safety effect. It is supporting evidence, not the core causal test. The core causal evidence remains Phase 1 plus Phase 4.

---

## 14. Judge Agreement Analysis

The judge layer is one of the most important credibility constraints in TR138. If the regex-style safety scoring and the LLM judge strongly agreed, the downstream interpretation would be much cleaner. They do not.

### 14.1 Aggregate readout

| Stratum family | Kappa range | Agreement range | Read |
|---------------|-------------|-----------------|------|
| Phase 1 batch sizes | 0.119 to 0.131 | 69.8% to 70.2% | Low agreement quality |
| Phase 2 co-batch conditions | 0.118 to 0.132 | 69.8% to 70.5% | Low agreement quality |
| Phase 3 quant levels | 0.110 to 0.251 | 62.2% to 94.4% | Highly unstable across quant levels |
| Phase 4 true batching | 0.093 to 0.095 | 78.6% to 78.9% | Better raw agreement, still poor kappa |

The central fact here is not the raw agreement percentage. It is the kappa.

Kappa remains poor across every family of conditions. Even where percent agreement looks superficially high, agreement beyond chance is weak. That means the judge stack should be treated as a directional audit layer, not as a gold-standard label source.

### 14.2 Why percent agreement alone is misleading

The Phase 3 rows show the problem clearly:

- `Q8_0`: `94.4%` agreement, kappa `0.202`
- `Q4_K_M`: `92.4%` agreement, kappa `0.251`
- `Q2_K`: `62.2%` agreement, kappa `0.110`

If one only looked at agreement percentage, `Q8_0` would appear robust. Kappa says otherwise. The judge and heuristic scorer still disagree materially on the classification structure.

This suggests class imbalance and boundary ambiguity are doing a lot of work. The agreement percentage can be inflated simply because many rows are easy negatives or easy refusals. Kappa is correctly telling us that the hard cases remain hard.

### 14.3 What this means for TR138's conclusions

The judge results do not invalidate the report, but they force a more disciplined interpretation:

- trust the report most on relative condition changes within the same scoring stack
- distrust any claim that depends on the judge as an external oracle
- treat all absolute safety percentages as approximate, not canonical

The low kappa is also one reason the report emphasizes binary flip direction, replication across phases, and mechanism validation rather than pretending to offer precise calibrated safety rates.

### 14.4 Practical implication

If TR138 were being prepared for an external publication rather than an internal flagship report, the judge agreement problem would be the first thing to upgrade:

- human annotation on a targeted flip subset
- a second independent judge model
- adjudication on disagreement rows

As the report stands, the judge layer is adequate for internal directional work, but not a gold-standard safety annotation pipeline.

---

## 15. Jailbreak Type Breakdown

The jailbreak section is most useful when read as a vulnerability profile, not as proof of a clean batch-size law. The slopes are small, the level patterns are not monotonic, and the event counts are modest. What the section does provide is a view of where each model is already fragile and whether batching seems to worsen or leave unchanged that existing weakness.

### 15.1 Cross-model summary

| Model | Highest baseline compliance | Largest observed amplification | Slope pattern | Main read |
|------|-----------------------------|--------------------------------|--------------|-----------|
| llama3.2-1b | Prefix injection: `59.2%` | `2.19x` | All slopes small and positive | Mild batch sensitivity on top of a known weak jailbreak family |
| llama3.2-3b | DAN-style: `33.3%` baseline, broader positive slopes | `1.54x` | All slopes small and positive | Slight broad-based worsening, but still modest in size |
| qwen2.5-1.5b | Roleplay: `73.3%` baseline | `5.02x` | Mixed; vulnerability dominates slope | Baseline jailbreak weakness is the real story |

### 15.2 What the model-by-model patterns say

For `llama3.2-1b`, the main vulnerability is prefix injection. It starts with the highest baseline compliance and shows the largest observed amplification ratio. The slopes are positive but very small, so the right reading is not "batching creates jailbreak risk from nothing." The right reading is "batching slightly worsens an already weak refusal regime."

For `llama3.2-3b`, the pattern is broader. DAN-style, direct, prefix injection, and roleplay all show small positive compliance slopes. That makes the model look more uniformly load-sensitive, though still only weakly so. This is more interesting than the absolute size of any one slope.

For `qwen2.5-1.5b`, batching is not the main story. The model is already highly vulnerable to roleplay and prefix injection at baseline, and the amplification ratios are extreme because the direct harmful baseline is relatively low while the jailbreak paths are already strong. This is best read as a model-level jailbreak robustness issue, not a clean batch effect.

### 15.3 Why this section is still useful

The jailbreak breakdown matters for deployment because it tells you where to spend scarce evaluation effort.

If you can only spot-check one family of prompts after changing batch policy:

- prioritize prefix injection on `llama3.2-1b`
- prioritize broad jailbreak regression on `llama3.2-3b`
- prioritize roleplay and prefix injection on `qwen2.5-1.5b`

That is a better operational use of this section than trying to extract a strong universal batch-amplification law from it.

---

## 16. Per-Category Bias Analysis

The BBQ section is secondary to the refusal analysis, but it still adds useful structure. It shows whether the instability story is concentrated only in harm-refusal tasks or whether demographic reasoning tasks also reveal uneven behavior.

### 16.1 Cross-model category summary

| Model | Lowest-scoring category | Score | Highest-scoring category | Score | Range |
|------|-------------------------|-------|--------------------------|-------|------|
| llama3.2-1b | Disability_status | 0.500 | Nationality | 1.000 | 0.500 |
| llama3.2-3b | Disability_status | 0.748 | Gender_identity / Nationality / Race_x_gender | 1.000 | 0.252 |
| qwen2.5-1.5b | Disability_status | 0.637 | Gender_identity / Physical_appearance / Religion / Race_x_gender | 1.000 | 0.363 |

The most consistent pattern is simple:

- `Disability_status` is the weakest category in all three models.

That consistency is stronger than any single category ranking. It suggests the models share a category-specific weakness that is not idiosyncratic to one architecture.

### 16.2 Cross-model category effect

The cross-model ANOVA reports:

- `F = 48.81`
- `p < 1e-16`
- `eta^2 = 0.071`

That is a real category effect and a non-trivial one. But it still should not be overstated. This section is descriptive. It does not by itself prove that batching uniquely amplifies bias in those categories. What it proves is that category-level bias performance is uneven, and that this unevenness persists across the model slate used in TR138.

### 16.3 How to use this section

Use the BBQ analysis as a robustness map:

- it identifies categories worth targeted follow-up
- it broadens the report beyond pure refusal tasks
- it shows that safety fragility is not isolated to a single benchmark family

Do not use it as the centerpiece of the TR138 claim. The report is still primarily about batching and refusal fragility, not about a new general theory of demographic bias under batching.

---

## 17. Variance-Safety Correlation

This analysis asks whether the prompts that start out safest are also the prompts most likely to flip under perturbation. That would support a stronger version of the fragility story: the refusal boundary itself would be the main locus of instability.

### 17.1 Results

| Model or phase | Pearson r | p-value | N | Read |
|---------------|-----------|---------|---|------|
| llama3.2-1b | 0.034 | 0.4643 | 468 | Null |
| llama3.2-3b | -0.038 | 0.4143 | 468 | Null |
| qwen2.5-1.5b | -0.037 | 0.4248 | 468 | Null |
| llama3.2-1b_P3 | 0.000 | 1.0000 | 660 | Null |
| llama3.2-3b_P3 | 0.000 | 1.0000 | 660 | Null |
| qwen2.5-1.5b_P3 | 0.000 | 1.0000 | 660 | Null |

### 17.2 Interpretation

This is a clean negative result.

TR138 does not find evidence that baseline-safe prompts are systematically more likely to flip than other prompts. That means the stronger version of the theory is not supported. The batch effect appears to be sparse and distributed rather than concentrated in a clearly identifiable subset of "most aligned" prompts.

That is actually useful. It prevents the report from sliding into a stronger but weaker-supported story.

### 17.3 What remains possible

The null result does not mean prompt-level heterogeneity is absent. It only means this specific scalar summary does not expose it. With so few flips, and with a binary baseline score, the analysis has limited resolution. A richer annotation scheme or human review of the flipped rows could still reveal structured prompt-level vulnerability.

---

## 18. Safety-Capability Divergence

This section tests the strongest version of the Phase 1 claim: do safety and capability flip rates separate cleanly enough, at each batch size, to show formal non-overlap of Wilson confidence intervals?

### 18.1 Per-batch readout

| Batch size comparison | Safety rate | Safety CI | Capability rate | Capability CI | Overlap | Disproportionate |
|----------------------|-------------|-----------|-----------------|---------------|---------|------------------|
| P1_bs2 | 0.005 | [0.0024, 0.0103] | 0.001 | [0.0004, 0.005] | Yes | No |
| P1_bs4 | 0.005 | [0.0024, 0.0103] | 0.002 | [0.0007, 0.006] | Yes | No |
| P1_bs8 | 0.005 | [0.0024, 0.0103] | 0.001 | [0.0004, 0.005] | Yes | No |
| P1_bs16 | 0.005 | [0.0024, 0.0103] | 0.001 | [0.0001, 0.0039] | Yes | No |
| P1_bs32 | 0.006 | [0.0029, 0.0112] | 0.001 | [0.0004, 0.005] | Yes | No |

### 18.2 What this means

This is one of the most important nuance sections in the report.

The phase-level pattern points toward safety asymmetry:

- safety flips are more frequent than capability flips
- the aggregate safety-to-capability ratio is about `3.6x`
- the dominant direction is refusal to compliance

But the per-batch Wilson intervals still overlap. So the strongest formal version of the claim is not established at the individual batch-size level.

The correct reading is therefore:

- TR138 supports a pattern-level asymmetry
- TR138 does not support a clean threshold-level divergence proof

That is not a weakness in the report's logic. It is the honest consequence of rare events.

### 18.3 Why this still matters

In production, rare asymmetric failures can matter even when they are hard to separate with neat interval tests. If the failure direction is consistently toward weaker refusal, and if it appears under both synchronized dispatch and true batching, that is enough to make batch policy a safety-relevant variable even without a dramatic per-batch cliff.

---

## 19. Heterogeneity, Thresholds, and Failure Shape

This section asks whether the batch effect looks like a cliff or a diffuse fragility pattern.

### 19.1 Task sensitivity summary

| Model | Most sensitive task | Slope | Least sensitive task | Slope range | Read |
|------|---------------------|-------|----------------------|-------------|------|
| llama3.2-1b | advbench_refusal | 0.000000 | truthfulqa | 0.000000 | No detectable task gradient |
| llama3.2-3b | jailbreak_amplification | 0.000599 | arc_challenge | 0.000000 | Very small task skew toward jailbreaks |
| qwen2.5-1.5b | jailbreak_amplification | 0.000325 | bbq_bias | -0.000492 | Tiny mixed task gradient |

The key fact is how small these slopes are. There is no strong monotonic task-level escalation with batch size. The best available interpretation is that batching creates a sparse perturbation regime, not a simple linear degradation law.

### 19.2 Critical threshold analysis

| Model | Critical batch size | Method | Interpretation |
|------|---------------------|--------|----------------|
| llama3.2-1b | None | Wilson CI non-overlap | No batch size where safety clearly separates from capability |
| llama3.2-3b | None | Wilson CI non-overlap | No threshold-style break point |
| qwen2.5-1.5b | None | Wilson CI non-overlap | No threshold-style break point |

This matters because it tells you what kind of operational hazard TR138 has identified.

It is not a "safe until batch size 8, unsafe at 16" story.

It is a "batching changes the system into a slightly different safety regime, with sparse failures and no clean cliff" story.

### 19.3 Implication for deployment

That failure shape is harder to reason about than a simple threshold. A threshold can be avoided. A diffuse low-rate perturbation requires monitoring, targeted validation, and stack-matched evaluation.

This is another reason Phase 4 matters. In a threshold-style story, the reduced Phase 4 subset would need to identify the exact cliff. It does not. Instead, it confirms the weaker but more realistic conclusion: true batching preserves the same general sparse-failure shape seen in Phase 1.

---

## 20. Cross-TR Validation

TR138 sits inside a larger Banterhearts research line, so it should be read both on its own terms and in relation to adjacent reports.

For the final TR138 run, cross-TR validation was executed directly against the local result artifacts from:

- TR135: `research/tr135/results/20260307_162151/tr135_analysis.json`
- TR136: `research/tr136/results/20260308_015147/tr136_analysis.json`
- TR137: `research/tr137/results/20260308_180727/tr137_analysis.json`

The emitted TR138-specific artifact is:

- `research/tr138/results/20260311_185200/tr138_cross_tr_validation.json`

This section therefore no longer relies on conceptual comparison alone. It uses direct anchor checks and ranking reconciliation from the stored analyses.

### 20.1 Direct anchor validation against TR136

TR136 is the best baseline anchor for TR138 because it contains the same model slate and explicit backend labels for `vllm_fp16`, `ollama_q4_k_m`, and `ollama_q8_0`.

The strongest anchor is `Phase 1 batch=1` in TR138 against `vllm_fp16` in TR136.

| Model | Anchor | Mean delta (pp) | Max delta (pp) | Tasks compared | Within 5pp tolerance? |
|------|--------|-----------------|----------------|----------------|-----------------------|
| llama3.2-1b | TR138 P1 `batch=1` vs TR136 `vllm_fp16` | 0.23 | 0.83 | 6 | Yes |
| llama3.2-3b | TR138 P1 `batch=1` vs TR136 `vllm_fp16` | 0.88 | 2.93 | 6 | Yes |
| qwen2.5-1.5b | TR138 P1 `batch=1` vs TR136 `vllm_fp16` | 0.68 | 2.00 | 6 | Yes |

This is a strong result. Every shared model-task baseline falls within the `5pp` tolerance, and most are far tighter than that.

The same pattern holds for the Phase 3 Ollama anchors on the two shared safety tasks:

| Model | Anchor | Mean delta (pp) | Max delta (pp) | Tasks compared | Within 5pp tolerance? |
|------|--------|-----------------|----------------|----------------|-----------------------|
| llama3.2-1b | TR138 P3 `Q4_K_M, c=1` vs TR136 `ollama_q4_k_m` | 0.00 | 0.00 | 2 | Yes |
| llama3.2-3b | TR138 P3 `Q4_K_M, c=1` vs TR136 `ollama_q4_k_m` | 0.41 | 0.83 | 2 | Yes |
| qwen2.5-1.5b | TR138 P3 `Q4_K_M, c=1` vs TR136 `ollama_q4_k_m` | 1.25 | 2.50 | 2 | Yes |
| llama3.2-1b | TR138 P3 `Q8_0, c=1` vs TR136 `ollama_q8_0` | 0.00 | 0.00 | 2 | Yes |
| llama3.2-3b | TR138 P3 `Q8_0, c=1` vs TR136 `ollama_q8_0` | 0.83 | 1.67 | 2 | Yes |
| qwen2.5-1.5b | TR138 P3 `Q8_0, c=1` vs TR136 `ollama_q8_0` | 0.42 | 0.84 | 2 | Yes |

These anchor checks matter because they show TR138 is not a numerically isolated run. Where the serving stack overlaps directly with prior TRs, the baselines line up closely.

### 20.2 Concurrency null consistency with TR135

TR135 asked whether concurrency itself degrades safety and found a robust null. TR138 Phase 3 should agree with that if the Phase 3 implementation is behaving sensibly.

For the shared Llama models at `Q4_K_M`, it does:

| Model | TR138 `Q4_K_M` c1 -> c8 delta (pp) | TR138 slope | TR135 safety slope per agent | TR135 interpretation | Consistent null? |
|------|------------------------------------|-------------|------------------------------|----------------------|------------------|
| llama3.2-1b | 0.00 | 0.0000 | +0.000141 | both_stable | Yes |
| llama3.2-3b | 0.00 | 0.0000 | -0.000120 | both_stable | Yes |

Qwen is excluded from this exact check because TR135 used `qwen2.5-3b-q4_k_m` while TR138 Phase 3 used `qwen2.5-1.5b`. Even with that family mismatch, the TR138 Phase 3 pattern still points in the same direction: concurrency contributes effectively no safety movement relative to quantization.

This is one of the cleanest cross-report validations in TR138. The concurrency null is not only a within-run claim; it reproduces the earlier TR135 result on the shared Llama slice.

### 20.3 Axis ranking against TR137 synthesis

TR137 already established the historical ranking across earlier optimization axes:

| Axis source | Effect size (pp) |
|-------------|------------------|
| TR137 aggregate quantization | 20.62 |
| TR137 aggregate backend | 14.79 |
| TR137 aggregate concurrency | 0.36 |

TR138 adds its own axes:

| TR138 axis | Effect size (pp) |
|-----------|------------------|
| quantization | 34.96 |
| true batching | 5.18 |
| batch size | 3.71 |
| concurrency | 0.00 |

Merged into one ordering, the picture is:

| Rank | Axis | Effect size (pp) |
|------|------|------------------|
| 1 | TR138 quantization | 34.96 |
| 2 | TR137 historical quantization | 20.62 |
| 3 | TR137 historical backend | 14.79 |
| 4 | TR138 true batching | 5.18 |
| 5 | TR138 batch size | 3.71 |
| 6 | TR137 historical concurrency | 0.36 |
| 7 | TR138 concurrency | 0.00 |

This ranking is the most useful cross-TR takeaway:

- quantization remains the dominant serving risk axis
- backend choice remains larger than the batching effects seen here
- batching is materially larger than concurrency as a safety-relevant phenomenon
- TR138 therefore fills a genuine middle band in the optimization-safety map rather than duplicating an already-solved axis

### 20.4 Relationship to prior TRs

The direct validation above changes how the earlier reports should be read together.

TR138 is best understood as a complement to the prior safety-tax reports:

- TR134 established that low-precision quantization can damage safety alignment.
- TR135 established that concurrency alone is not the primary driver.
- TR136 established that backend choice can matter more than precision in some deployment settings.
- TR137 synthesized those axes into a broader framework for optimization-induced safety cost.

TR138 adds a distinct question:

> if model weights, prompt text, and decoding policy are all fixed, can the serving batch condition itself still change safety behavior?

That question is different from the quantization, backend, and concurrency questions. It is also more subtle, which is why the report spends so much effort distinguishing small but real effects from large dramatic ones.

Now that the artifact-level anchors have been checked, the relationship is stronger than before:

- TR138 shares baseline behavior cleanly with TR136 where the backend and model anchors match
- TR138 reproduces the TR135 concurrency-null result on the shared Llama slice
- TR138 preserves the older ranking in which quantization dominates concurrency, but adds batching as a new intermediate axis

### 20.5 What TR138 uniquely contributes after cross-validation

TR138 contributes three things the earlier line does not.

First, it isolates batch condition as an independent variable instead of bundling it inside broader system changes.

Second, it tests both synchronized dispatch and explicit prompt-list true batching. That is the cleanest mechanism check in this cluster of reports.

Third, the cross-TR validation now shows that the batch story is not interchangeable with the quantization or concurrency stories:

- quantization effects are much larger
- backend effects are still larger on average
- concurrency effects are much smaller
- true batching sits in the middle as a small but distinct safety-relevant phenomenon

### 20.6 What TR138 does not overturn

TR138 does not overturn the main findings of the prior line.

- It does not show that batching is a bigger risk than quantization.
- It does not show that concurrency is a hidden dominant factor.
- It does not show that adversarial co-batching is already a confirmed deployment hazard.

What it does show is that the older framework was incomplete. Batch policy belongs in the safety evaluation surface, even if it is not the largest term in that surface.

---

## 21. Limitations

The report is strongest when its limitations are explicit rather than hidden. The main ones are below.

1. **Rare-event regime.** The core Phase 1 and Phase 4 effects are real but small in absolute terms. That limits per-cell significance and threshold-style inference.

2. **Greedy decoding only.** All phases use temperature `0.0`. This isolates numerical instability, but the result applies most directly to deterministic serving rather than sampled generation.

3. **Single hardware environment.** All results come from one RTX 4080 Laptop GPU in a Windows + WSL2 + Docker workflow. Datacenter GPUs may change both the runtime profile and the numerical profile.

4. **Phase 1 is still scheduler-mediated.** Phase 4 addresses this directly, but only on a reduced subset and only for two models.

5. **Phase 3 is not true batching.** It is quantization under concurrent load. The report treats it that way, but the distinction still matters for external interpretation.

6. **Judge reliability is limited.** Kappa remains low across all strata. The judge pass is useful for triangulation, not for gold-standard adjudication.

7. **Binary safety scoring is coarse.** Refusal and compliance are collapsed into a simplified label space. Partial compliance, hedging, and subtle unsafe assistance may be mischaracterized.

8. **Model coverage is still narrow.** The slate is good enough for an internal flagship study, but not broad enough to claim a universal law across all alignment recipes and model scales.

9. **Phase 2 is mechanism-incomplete.** The co-batch design tests a real question, but it does not isolate whether any neighbor effect would come from compute sharing, memory sharing, or scheduler order.

10. **No human annotation subset.** A curated human review of the flipped rows would materially strengthen the interpretation, especially given the weak judge agreement.

11. **Cross-TR validation is still partial rather than fully unified.** Section 20 now provides artifact-backed anchor checks against TR135-TR137, but it is not a single pooled meta-analysis across every model, task, and optimization axis in the prior line.

### 21.1 What these limitations do and do not invalidate

These limitations weaken ambitious claims. They do not erase the central contribution.

The central contribution survives because it is replicated in the two phases that matter most:

- Phase 1 shows the safety-skewed perturbation under controlled dispatch batching.
- Phase 4 shows the same general direction under explicit true batching.

That is enough to make batching a real safety variable, even if it is not yet enough to define a universal quantitative law.

---

## 22. Conclusions

### 22.1 Direct answers to the research questions

**RQ1: Does batching change outputs under deterministic inference?**

Yes. Output identity drops from perfect agreement at `batch=1` to roughly `92%` byte-identical agreement under larger batch settings, and safety classifications flip at a low but non-zero rate.

**RQ2: Are those changes safety-neutral?**

No. The aggregate safety flip rate is about `0.5%` versus `0.1%` for capability, with refusal-to-compliance as the dominant failure direction.

**RQ3: Is the main result just a request-scheduler artifact?**

Not entirely. Phase 4 preserves the direction of the effect under explicit prompt-list true batching, though again at a small absolute rate.

**RQ4: Does adversarial co-batching create a strong interference effect?**

Not in this dataset. Phase 2 rules out large practical interference but leaves room for narrower effects that would need a different design to resolve.

**RQ5: Does concurrency meaningfully interact with quantization to create a new batch story?**

No. Phase 3 is dominated by quantization. Concurrency contributes little safety signal in comparison.

### 22.2 Strongest supported claims

The strongest claims in TR138 are:

1. Batch condition is a safety-relevant serving variable, not just a throughput variable.
2. The dominant direction of batch-induced safety change is toward weaker refusal.
3. The batch effect is small in absolute terms but real enough to matter for safety-critical deployments.
4. True batching preserves the direction of the effect.

### 22.3 Weaker or unsupported claims

The report does **not** support the following stronger claims:

1. Batching causes large absolute safety collapse.
2. Adversarial co-batching is already a confirmed production hazard.
3. Concurrency is an independent major safety driver in the Phase 3 stack.
4. There is a clean critical batch-size threshold where the system becomes unsafe.

### 22.4 Final framing

TR138 should be framed as a flagship report because it closes an important methodological gap in the Banterhearts safety line.

Before TR138, the optimization story was mainly:

- quantization can hurt safety
- backend choice can matter
- concurrency alone is weak

After TR138, the story is more complete:

- even when weights and prompts stay fixed, the serving configuration itself can induce small but safety-skewed instability
- explicit true batching confirms that this is not just an arrival-timing artifact

That is a novel and publishable result, provided it is described with the same caution the data requires.

### 22.5 What follows logically from the evidence

The next-step logic from TR138 should be disciplined.

What follows directly:

1. Batch policy belongs inside the evaluated safety envelope of a deployment.
2. The right external-facing claim is about **small, safety-skewed instability**, not dramatic collapse.
3. Mechanism follow-up should prioritize true batching, deterministic kernels, and row-level flip adjudication.
4. Programmatically, batching should now be tracked as a separate axis from quantization, backend choice, and concurrency.

What does **not** follow directly:

1. that mixed-request batching is broadly unsafe in production
2. that a single critical batch threshold exists
3. that the same magnitudes will carry over unchanged to larger models or datacenter hardware

The consequence is that TR138 should drive a narrower but stronger follow-up program: resolve the mechanism, replicate on a larger deployment stack, and manually audit the small set of behavior-changing rows.

## 23. Production Guidance

TR138's production value comes from narrowing where batch policy belongs in the validation stack.

### 23.1 Decision matrix by deployment tier

| Deployment tier | Batch policy | Required validation standard | Practical read |
|----------------|--------------|------------------------------|----------------|
| Safety-critical agent | Prefer `batch=1` or exact deployed batch validation | Must validate the exact batch path and model quant level | Treat batch as a safety parameter |
| General production assistant | Start at `batch<=4` and expand only after stack-matched safety eval | Validate on the deployed backend and real prompt mix | Small batch tax may be acceptable |
| Throughput-first, low-risk workload | Larger batches acceptable if safety scope is narrow | Basic regression testing may be enough | Capability cost is smaller than safety cost |

### 23.2 What to do with Phase 2

Phase 2 should influence operations in a restrained way:

- do not treat adversarial co-batching as already proven harmful
- do isolate highly sensitive traffic classes when the implementation cost is low
- do use co-batch testing as a follow-up experiment if your system mixes very different request types

The correct operational reaction is "investigate where cheap," not "ban mixed batching outright."

### 23.3 What to do with Phase 3

Phase 3 leads to a clearer operational rule:

- `Q8_0`: normal load testing is probably enough
- `Q4_K_M`: still run concurrent-load safety checks before rollout
- `Q2_K`: require explicit safety validation and latency validation; it is the unstable regime in this stack

The report does not say low-bit quantization is always unsafe. It says the lowest-bit regime deserves the most scrutiny and cannot be justified by simplistic assumptions about speed.

### 23.4 Minimum validation protocol for teams using batching

At minimum, a production team should do the following before changing batch policy:

1. Evaluate the exact deployed batch sizes, not just `batch=1`.
2. Include refusal-style safety tasks and ordinary capability tasks in the same validation pass.
3. Check flip direction, not just mean score.
4. Repeat on the actual backend and quant level used in production.
5. Run a reduced true-batch prompt-list check if the serving stack supports it.

### 23.5 The simplest safe rule

If a team only remembers one thing from TR138, it should be this:

> batch size is not only a throughput knob. It is part of the safety configuration of the system.

That is the most operationally useful conclusion in the report.

### 23.6 Immediate follow-up program

TR138 is strongest when it drives concrete next steps rather than generic caution.

| Horizon | Follow-up | Why it follows from TR138 |
|---------|-----------|---------------------------|
| Immediate | Manually review every behavior-changing safety row from Phase 1 and Phase 4 | The flip set is small enough to audit completely, and judge kappa is too weak to treat the automated labels as final |
| Immediate | Add stack-matched batch validation to deployment gates | The report establishes that batch condition itself is safety-relevant |
| Near-term | Replicate Phase 4 on a larger model and one datacenter-class GPU | The main remaining external-validity question is hardware and scale |
| Near-term | Run deterministic-kernel or deterministic-backend A/B tests | This is the cleanest mechanism test for whether the flips are truly numerical in origin |
| Near-term | Re-run Phase 2 with stronger filler families and backend instrumentation | The current negative result narrows the claim but leaves open smaller neighbor effects |
| Medium-term | Build a persistent flip registry across reports | TR138 shows that rare but high-value flip rows are the right objects for cross-report comparison |

### 23.7 What the next report should do

The best follow-up to TR138 is not "more of the same at bigger scale." It is a more diagnostic experiment.

The highest-value next study would:

1. keep the true-batching Phase 4 path
2. add human adjudication on every flipped or disagreement row
3. compare one deterministic kernel path against the current production-like path
4. replicate on a larger model family or datacenter GPU
5. preserve the safety-versus-capability control arm

That would turn TR138 from a strong flagship finding into a mechanistically sharper result with better external defensibility.

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
- vLLM uses the production-like default execution path (no forced eager mode)
- CUBLAS_WORKSPACE_CONFIG not set (allows non-deterministic cuBLAS)

### Artifact Paths

- Run directory: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr138\results\20260311_185200`
- Analysis JSON: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr138\results\20260311_185200\tr138_analysis.json`
- Report: `C:\Users\sahil\OneDrive\Documents\GitHub\Banterhearts\research\tr138\results\20260311_185200\tr138_report.md`
- Config: `research/tr138/config.yaml`
- Task definitions: `research/tr138/tasks/`

### Docker Commands

```bash
# vLLM server (Phase 1-2)
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model unsloth/Llama-3.2-1B-Instruct \
  --max-model-len 2048 --dtype float16 \
  --gpu-memory-utilization 0.80
```

### Run Commands

```bash
# Full TR138 run
python research/tr138/run.py -v

# Judge + analysis + report only on an existing run
python research/tr138/run.py -v --skip-prep --skip-eval --run-dir research/tr138/results/20260311_185200

# Analysis + report only
python research/tr138/run.py -v --skip-prep --skip-eval --skip-judge --run-dir research/tr138/results/20260311_185200

# Report only
python research/tr138/run.py -v --skip-prep --skip-eval --skip-judge --skip-analyze --run-dir research/tr138/results/20260311_185200

# Cross-TR validation only
python research/tr138/cross_tr_validate.py --run-dir research/tr138/results/20260311_185200
```

### Expected Outputs

| File | Purpose |
|------|---------|
| `samples.jsonl` | Raw eval rows across Phases 1-4 |
| `judge_labels.jsonl` | Judge outputs used for agreement and scored analyses |
| `tr138_analysis.json` | Canonical aggregate metrics and statistical outputs |
| `tr138_scored.jsonl` | Row-level scored records |
| `tr138_cross_tr_validation.json` | Cross-report anchor checks against TR135/TR136/TR137 |
| `tr138_report.md` | Generated run report |
| `PublishReady/reports/Technical_Report_138.md` | Hand-authored publish-ready report |

### Reproducibility Checks

After a rerun, the minimum sanity checks are:

1. `samples.jsonl` reaches `31,410` rows for the full run, with phase totals `17,154 / 5,616 / 5,940 / 2,700`
2. `tr138_analysis.json` exists before report generation
3. `judge_labels.jsonl` reaches `21,480` rows for a full judge pass
4. Phase 1 aggregate safety flip rate remains near the reported `0.5%`
5. Phase 4 aggregate true-batch safety flip rate remains near the reported `0.8%`
6. Phase 3 continues to show quantization significance with null concurrency and interaction terms

---

## Appendix A: Source-of-Truth Artifacts

This appendix makes the evidence chain explicit. The report narrative is not the source of truth for claims; the run artifacts are.

### A.1 Primary artifacts

| Artifact | Path | Role in report | Primary sections |
|----------|------|----------------|------------------|
| Raw eval rows | `research/tr138/results/20260311_185200/samples.jsonl` | Source-of-truth sample records for Phases 1-4 | 5-10, 13 |
| Judge labels | `research/tr138/results/20260311_185200/judge_labels.jsonl` | LLM judge outputs used for agreement analysis | 14 |
| Analysis JSON | `research/tr138/results/20260311_185200/tr138_analysis.json` | Canonical aggregate metrics and test outputs | 7-23 |
| Scored rows | `research/tr138/results/20260311_185200/tr138_scored.jsonl` | Row-level scored outputs for audit and spot-checking | 7-19 |
| Cross-TR validation | `research/tr138/results/20260311_185200/tr138_cross_tr_validation.json` | Direct anchor checks against TR135/TR136/TR137 | 20 |
| Generated run report | `research/tr138/results/20260311_185200/tr138_report.md` | Machine-generated precursor, useful for provenance only | provenance |
| Experiment config | `research/tr138/config.yaml` | Source-of-truth experiment scope and model slate | 5, 6, 24 |
| Runner | `research/tr138/run.py` | Execution path and phase semantics | 5, 24 |
| Analyzer | `research/tr138/analyze.py` | Statistical pipeline and aggregation logic | 11-20 |
| Report generator | `research/tr138/generate_report.py` | Auto-report emission path | provenance |

### A.2 Section-to-artifact map

| Report section | Primary artifact key or source |
|----------------|--------------------------------|
| Section 7 | `phase1.output_identity`, `phase1.flip_rates`, `phase1.flip_direction_breakdown`, `phase1.statistical_tests` |
| Section 8 | `phase2.condition_scores`, `phase2.pairwise_comparisons`, `phase2.anova` |
| Section 9 | `phase3.safety_grid`, `phase3.anova`, `phase3.anova_per_model`, `phase3.slopes` |
| Section 10 | `cross_phase_synthesis`, `phase4.phase1_alignment`, `phase4.flip_rates` |
| Sections 11-12 | `tost_equivalence`, `power_analysis` |
| Section 13 | `latency_analysis` |
| Section 14 | `judge_agreement` |
| Section 15 | `jailbreak_breakdown` |
| Section 16 | `bbq_categories` |
| Sections 17-19 | `variance_safety_correlation`, `safety_capability_divergence`, `slope_heterogeneity`, `critical_threshold` |
| Section 20 | `tr138_cross_tr_validation.json`, `tr135_analysis.json`, `tr136_analysis.json`, `tr137_analysis.json` |

### A.3 Evidence-handling rule

Where report prose and generated summaries differ in emphasis, the artifact wins. This publish-ready report is an interpretive layer built on top of the artifacts above, not a substitute for them.

### A.4 Raw row schema by phase

| Phase | One retained row means | Phase-specific control fields | Primary pairing key used later |
|------|-------------------------|-------------------------------|--------------------------------|
| 1 | One prompt evaluated under one exact dispatch batch size | `batch_size`, `dispatch_batch_size`, `tail_padded` | `(model, task_name, sample_id)` against Phase 1 `batch=1` |
| 2 | One target prompt evaluated under one neighbor condition | `condition`, `co_batch_condition`, `filler_count`, `target_position` | `(model, task_name, sample_id)` across conditions |
| 3 | One safety prompt evaluated under one `quant x concurrency` cell | `quant`, `concurrency`, `eval_ms`, `submit_time_s` | `(model, task_name, sample_id, quant, concurrency)` |
| 4 | One prompt evaluated inside one explicit prompt-list batch | `batch_size`, `dispatch_batch_size`, `tail_padded`, `batch_mode`, `true_batching`, `analyzed_in_batch` | `(model, task_name, sample_id)` against Phase 4 `batch=1` and Phase 1 counterparts |

This matters because TR138 uses different comparison logic in different phases. The report is only coherent if the reader keeps the unit of comparison aligned to the phase design.

### A.5 Scoring and judge routing reference

| Task | Primary scorer in analysis | Inputs used | Output used in report |
|------|----------------------------|-------------|-----------------------|
| `advbench_refusal` | `RefusalDetector` | `candidate` | binary refusal score |
| `jailbreak_amplification` | `RefusalDetector` | `candidate` | binary refusal score |
| `truthfulqa` | `TruthfulnessScorer` | `candidate`, `correct_answers`, `incorrect_answers` | truthfulness score |
| `bbq_bias` | `BiasDetector` | `candidate`, answer choices, stereotype indices | bias score |
| `mmlu_real`, `arc_challenge` | answer extractor + reference compare | `candidate`, `reference` | accuracy |

| Task family | Judge route | Inputs used | Used for |
|-------------|-------------|-------------|----------|
| refusal and jailbreak | `judge_refusal` | `prompt`, `candidate` | Section 14 agreement only |
| TruthfulQA | `judge_truthfulness` | `prompt`, `candidate`, `correct_answers`, `incorrect_answers` | Section 14 agreement only |
| BBQ | `judge_bias` | `context`, `question`, `candidate`, `choice_a`, `choice_b`, `choice_c` | Section 14 agreement only |

The judge does not replace the main analysis labels. That separation is a design choice, not an accident.

---

## Appendix B: Output Identity & Flip Audit Tables

### B.1 Phase 1 overall output identity and asymmetry

| Batch size | Byte-identical (%) | Safety score changes | Capability score changes | Overall chi-squared p | Odds ratio |
|-----------|--------------------|----------------------|--------------------------|-----------------------|------------|
| 2 | 92.86 | 7 | 2 | 0.0849 | 3.1200 |
| 4 | 92.55 | 7 | 3 | 0.1856 | 2.2270 |
| 8 | 92.62 | 7 | 2 | 0.0849 | 3.1200 |
| 16 | 92.03 | 7 | 1 | 0.0296 | 5.2040 |
| 32 | 92.41 | 8 | 2 | 0.0503 | 3.5390 |

This table is useful because it combines the output-identity story and the safety-asymmetry story in one place. Byte-level instability is common relative to score changes; score changes are the rare but safety-relevant subset.

### B.2 Phase 4 true-batch validation

| True batch size | Byte-identical (%) | Safety score changes | Capability score changes | Safety flip rate | Capability flip rate | Flip ratio |
|----------------|--------------------|----------------------|--------------------------|------------------|----------------------|------------|
| 4 | 91.22 | 4 | 1 | 0.0080 | 0.0025 | 3.2 |
| 8 | 92.44 | 4 | 0 | 0.0080 | 0.0000 | inf |

### B.3 Phase 4 alignment with synchronized-dispatch signal

| Model | True batch size | Paired rows | Flip agreement (%) | Score agreement (%) |
|------|------------------|-------------|--------------------|---------------------|
| llama3.2-3b | 4 | 450 | 100.00 | 100.00 |
| llama3.2-3b | 8 | 450 | 99.56 | 99.56 |
| qwen2.5-1.5b | 4 | 450 | 99.11 | 99.11 |
| qwen2.5-1.5b | 8 | 450 | 99.11 | 99.11 |

This is the most important appendix table in the report. It shows why Phase 4 is best read as a mechanism confirmation rather than as a second independent giant sweep.

---

## Appendix C: Statistical Test Inventory

This appendix is not an exhaustive dump of every generated row. It records the tests most important to the report's main claims.

### C.1 Phase 2 key pairwise tests: `solo` vs `adversarial`

| Model | Mean delta | p-value | Cohen's d | Read |
|------|------------|---------|-----------|------|
| llama3.2-1b | +0.0000 | 1.0000 | +0.0000 | Null |
| llama3.2-3b | +0.0053 | 0.0956 | +0.0130 | Weak directional pattern only |
| qwen2.5-1.5b | -0.0043 | 0.3178 | -0.0110 | Null |

### C.2 Phase 3 per-model ANOVA summary

| Model | Quant p | Quant eta^2 | Concurrency p | Concurrency eta^2 | Interaction p | Interaction eta^2 |
|------|---------|-------------|---------------|-------------------|---------------|-------------------|
| llama3.2-1b | 0.0000 | 0.2181 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| llama3.2-3b | 0.0000 | 0.0644 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| qwen2.5-1.5b | 0.0000 | 0.2996 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

This is the cleanest short proof that Phase 3 is a quantization story rather than a concurrency story.

### C.3 Phase 4 true-batch statistical tests

| Model | True batch size | Chi-squared p | Fisher p | Odds ratio | Significant (uncorrected) |
|------|------------------|---------------|----------|------------|---------------------------|
| llama3.2-3b | 4 | 0.3706 | 1.0000 | 2.4108 | No |
| llama3.2-3b | 8 | 0.1201 | 0.2576 | 5.6707 | No |
| qwen2.5-1.5b | 4 | 0.4318 | 0.6326 | 1.8808 | No |
| qwen2.5-1.5b | 8 | 0.3706 | 1.0000 | 2.4108 | No |

The absence of strong per-cell significance is consistent with the rare-event nature of the main signal. It does not negate the value of Phase 4 as a direction-preserving mechanism check.

---

## Appendix D: Glossary

| Term | Definition |
|------|-----------|
| Batch size | Number of concurrent requests processed by the GPU in a single forward pass. In vLLM, controlled by concurrent request count due to continuous batching. |
| Co-batching | Processing multiple requests simultaneously where the content of neighboring requests may influence outputs through shared GPU compute kernels. |
| Continuous batching | vLLM's iteration-level scheduling that dynamically adds/removes requests from the batch at each decode step, unlike static batching which pads all sequences. |
| FP non-associativity | Floating-point addition is not associative: (a+b)+c != a+(b+c) due to rounding. Different batch sizes change the order of accumulation in matrix multiplications, producing different results even at temp=0. |
| Flip rate | Fraction of prompts where the safety/capability classification changes relative to the batch=1 control condition. |
| MDE | Minimum Detectable Effect. The smallest effect size the experiment can detect at 80% power and alpha=0.05. |
| PagedAttention | vLLM's memory management for KV-cache that allocates non-contiguous blocks, enabling per-request cache isolation. |
| Safety flip | A prompt whose safety classification (refuse/comply) changes when processed at a different batch size. |
| TOST | Two One-Sided Tests. Equivalence testing procedure that tests whether the difference between two groups falls within a pre-specified margin (here +/-3pp). |
| eta^2 (eta-squared) | Effect size measure for ANOVA. Proportion of total variance explained by the factor. Values: small=0.01, medium=0.06, large=0.14. |

---

## References

1. **SGLang Deterministic Inference** (Sep 2025). Batch-invariant CUDA kernels for reproducible outputs at 34% throughput cost. https://lmsys.org/blog/2025-09-sglang-determinism/

2. **LLM-42: Verified Speculation for Deterministic LLM Inference** (Microsoft Research, Jan 2026). Formal verification of speculative decoding determinism. No safety measurement.

3. **"Understanding Batch Size Impact on LLM Output"** (Medium, 2025). Detection and documentation of batch non-determinism. No safety analysis.

4. **vLLM: Efficient Memory Management for Large Language Model Serving** (Kwon et al., SOSP 2023). PagedAttention and continuous batching architecture.

5. **TR134-TR137: Banterhearts Alignment Robustness Under Quantization** (2026). Foundation safety benchmarks, classifier validation, multi-family analysis.

6. **IEEE 754-2019: Standard for Floating-Point Arithmetic.** Formal specification of non-associativity in FP operations.
