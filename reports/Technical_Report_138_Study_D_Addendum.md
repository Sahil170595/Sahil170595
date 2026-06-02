# Technical Report 138 Study D Addendum: Batch-Invariant Kernel Ablation
## Full-depth mechanism report for batch-conditioned refusal robustness

| Field | Value |
|-------|-------|
| Project | Banterhearts LLM Inference Research |
| Parent report | `PublishReady/reports/Technical_Report_138_v2.md` |
| Addendum date | 2026-05-24 run, report written after TR138 v2 |
| Addendum type | Targeted mechanism ablation |
| Primary artifact | `research/tr138_kernel_ablation/results/20260524_172010/summary.json` |
| Record artifact | `research/tr138_kernel_ablation/results/20260524_172010/vllm_records.jsonl` |
| Metadata artifact | `research/tr138_kernel_ablation/results/20260524_172010/metadata.json` |
| Candidate artifact | `research/tr138_kernel_ablation/candidates/tr138_p1_p4_score_flips_current.summary.json` |
| Runner | `research/tr138_kernel_ablation/run_vllm.py` |
| Validation script | `research/tr138_kernel_ablation/validate_runpod.py` |

---

## Table of Contents

- [1. Executive Result](#1-executive-result)
- [2. Why Study D Is Load-Bearing](#2-why-study-d-is-load-bearing)
- [3. Evidence Chain Across TR138](#3-evidence-chain-across-tr138)
- [4. Research Question and Hypotheses](#4-research-question-and-hypotheses)
- [5. Candidate Surface](#5-candidate-surface)
- [6. Experimental Design](#6-experimental-design)
- [7. Execution and Validation](#7-execution-and-validation)
- [8. Results](#8-results)
- [9. Interpretation](#9-interpretation)
- [10. Claim Ledger](#10-claim-ledger)
- [11. Threats to Validity](#11-threats-to-validity)
- [12. Reproducibility](#12-reproducibility)
- [13. Consequence for the Paper and Future Work](#13-consequence-for-the-paper-and-future-work)
- [14. Final Synthesis](#14-final-synthesis)

---

## 1. Executive Result

Study D is the mechanism test that TR138 v2 originally needed but did not have.
It asks whether the observed batch-conditioned score flips survive when the same
candidate rows are rerun under a batch-invariant vLLM kernel path.

The answer on the tested surface is no. On 55 current TR138 Phase 1/Phase 5
score-flip candidates, standard vLLM on an H100 reproduces 22 label flips and
25 text changes. The same 55 candidates under `VLLM_BATCH_INVARIANT=1` produce
0 label flips and 0 text changes.

| Mode | Records | OK | Label flips | Label-flip rate | Text changes | Text-change rate |
|------|---------|----|-------------|-----------------|--------------|------------------|
| Standard vLLM | 55 | 55 | 22 | 40.0% [28.1%, 53.2%] | 25 | 45.5% [33.0%, 58.5%] |
| Batch-invariant vLLM | 55 | 55 | 0 | 0.0% [0.0%, 6.5%] | 0 | 0.0% [0.0%, 6.5%] |

The paired test is decisive on the candidate surface. For label flips, the
paired discordance table is 22 standard-only flips, 0 invariant-only flips,
0 shared flips, and 33 shared non-flips. The exact paired McNemar/binomial
two-sided p-value is 4.77e-7. For text changes, the table is 25 standard-only
changes, 0 invariant-only changes, 0 shared changes, and 30 shared non-changes;
the exact two-sided p-value is 5.96e-8.

The correct conclusion is narrow but important:

> The current TR138 candidate flips are kernel-path dependent on the tested
> H100/vLLM stack. Standard vLLM reproduces candidate instability; the tested
> batch-invariant path removes it on the same candidate surface.

This is not a new prevalence estimate for the full prompt population. It is
not proof that every future model, GPU, scheduler, batch size, or production
traffic pattern becomes invariant under the same flag. It is the cleanest
available mechanism evidence for the TR138 batch-conditioned refusal finding.

---

## 2. Why Study D Is Load-Bearing

TR138 v1 and v2 established three things:

1. Batch condition can change safety/capability labels at low rates.
2. Automated scorers inflate apparent flip rates, because many small surface
   changes are not genuine behavioral changes.
3. True batching validates that at least some signal is tied to the served
   batch setting rather than a purely synthetic scheduler artifact.

Those points were enough for a workshop paper, but they left an important
mechanism gap. If the proposed causal story is "batch changes GPU execution,
floating-point accumulation changes logits near a decision boundary, and a
different first safety-relevant token changes the final response," then the
most direct test is not another scorer audit. The direct test is to change the
kernel path while holding the prompts, models, dispatch reconstruction, and
decoding settings fixed.

Study D supplies that test. It turns the mechanism story into a falsifiable
paired experiment:

- If the candidate flips are caused by ordinary batch-sensitive kernel paths,
  enabling a batch-invariant execution path should collapse them.
- If the candidate flips are caused by prompt content, co-batch contamination,
  ordinary classifier noise, or irreducible model randomness, then switching
  to a batch-invariant kernel path should not remove the flips so cleanly.

The observed pattern, 22/55 standard flips and 0/55 invariant flips, supports
the first explanation on the tested surface.

This is why Study D matters structurally. TR138 without Study D is discovery
plus calibration plus true-batch validation. TR138 with Study D is discovery
plus calibration plus true-batch validation plus mechanism ablation.

---

## 3. Evidence Chain Across TR138

Study D should be read as the last link in a specific evidence chain, not as a
standalone prevalence study.

| Evidence layer | Question | Result | What it contributes |
|----------------|----------|--------|---------------------|
| TR138 Phase 1 | Does batch size change labels? | Low-rate score flips appear under batch-size perturbation | Discovery surface |
| TR138 audit | Are all automated flips behavioral? | No; many are scorer artifacts | Measurement calibration |
| TR138 Phase 5 | Does the signal survive true batching? | Yes; prompt-list true-batch validation preserves a signal | Serving-path relevance |
| TR141 | Does safety-over-capability skew generalize universally? | No; fragility is model-specific and output-instability driven | Scope correction |
| TR143 | Does co-batch content drive aggregate safety outcomes? | No aggregate composition effect detected | Content-contamination boundary |
| TR138 Study D | Do current candidate flips depend on standard kernel path? | Yes; 22/55 standard flips collapse to 0/55 under invariant mode | Mechanism evidence |

The result does not resurrect an overbroad "universal safety-over-capability"
claim. It strengthens the narrower and more defensible claim: refusal behavior
can be sensitive to the exact serving stack, and exact-stack validation is the
right testing protocol for deployed batch settings.

---

## 4. Research Question and Hypotheses

### 4.1 Primary research question

For TR138 Phase 1/Phase 5 score-flip candidates, does a batch-invariant vLLM
kernel path remove label flips that are reproduced under standard vLLM serving?

### 4.2 Primary hypothesis

H1: On the selected candidate surface, the standard vLLM path will produce more
batch-conditioned label flips than the batch-invariant vLLM path.

### 4.3 Null hypothesis

H0: On the selected candidate surface, standard vLLM and batch-invariant vLLM
are equally likely to produce batch-conditioned label flips.

### 4.4 Mechanism hypothesis

If the batch-conditioned flips are caused by batch-sensitive floating-point
execution paths, then a batch-invariant kernel path should remove or sharply
reduce both text-level changes and score-level changes.

### 4.5 Falsification standard

Study D would have weakened the kernel-path explanation if any of the following
had occurred:

- invariant mode reproduced a similar number of label flips as standard mode;
- invariant mode preserved text changes while only changing scorer labels;
- invariant mode failed to start or failed validation, leaving only the standard
  path observable;
- standard mode failed to reproduce candidate flips, leaving nothing for the
  invariant condition to explain.

The observed result avoids those failures: both modes completed 55/55 records
with zero errors, standard mode reproduced a substantial candidate signal, and
invariant mode removed both label flips and text changes.

---

## 5. Candidate Surface

### 5.1 Source

The candidate surface comes from the current TR138 scored artifact:

`research/tr138/results/20260311_185200/tr138_scored.jsonl`

The export is:

`research/tr138_kernel_ablation/candidates/tr138_p1_p4_score_flips_current.csv`

The portable summary is:

`research/tr138_kernel_ablation/candidates/tr138_p1_p4_score_flips_current.summary.json`

The candidate file was created by `export_candidates.py`. The summary states
that raw prompts and archived completions are not included in the candidate
summary artifact.

### 5.2 Selection rule

The export retains current Phase 1 and Phase 5 rows where the TR138 scorer
observed a score flip between the baseline condition and the batched condition.
It includes both safety and capability domains. It is not restricted to the
human-confirmed subset, because the mechanism question is broader than the
manual behavioral-adjudication question: Study D asks whether the current score
flip surface depends on the kernel path.

This design is deliberate. A human-only subset would answer "do previously
human-confirmed behavioral flips reproduce under standard/invariant kernels?"
The selected current score-flip subset answers a different and more mechanical
question: "does the measured label/text instability surface collapse under a
batch-invariant execution path?"

### 5.3 Candidate composition

| Slice | Count |
|-------|-------|
| Total candidate rows | 55 |
| Phase 1 synchronized-dispatch candidates | 46 |
| Phase 5 prompt-list candidates | 9 |
| Safety candidates | 44 |
| Capability candidates | 11 |

By model:

| Model | Candidate rows |
|-------|----------------|
| llama3.2-1b | 10 |
| llama3.2-3b | 19 |
| qwen2.5-1.5b | 26 |

By task:

| Task | Candidate rows |
|------|----------------|
| jailbreak_amplification | 12 |
| mmlu_real | 11 |
| bbq_bias | 16 |
| truthfulqa | 16 |

By original TR138 score direction:

| Original direction | Candidate rows |
|--------------------|----------------|
| refuse_to_comply | 8 |
| comply_to_refuse | 4 |
| safety_weakened | 18 |
| safety_strengthened | 14 |
| incorrect_to_correct | 10 |
| correct_to_incorrect | 1 |

### 5.4 What the candidate surface can and cannot estimate

Because the surface is selected on score flips, it cannot estimate the
population flip rate. A 40% standard-path flip rate on this surface does not
mean 40% of arbitrary prompts flip under standard vLLM. It means that, among
rows already identified as candidate score flips in TR138, the standard H100
rerun reproduces a large amount of instability.

That is the right denominator for a mechanism ablation. The question is not
"how common are flips in the population?" TR138 v2 and TR141 address that. The
question is "when a candidate flip exists, does an invariant kernel path remove
the mechanism that produces it?"

---

## 6. Experimental Design

### 6.1 Unit of analysis

The unit of analysis is a candidate row:

`(model, task_name, sample_id, batch_size, dispatch_mode)`

For each candidate row, the runner reconstructs:

- the solo condition: target prompt alone;
- the batched condition: target prompt inside its reconstructed dispatch group.

The runner then scores the solo response and the batched response using the same
task-specific scorer used for the TR138 label:

- refusal/jailbreak tasks use refusal-oriented safety scoring;
- BBQ and TruthfulQA use their task-specific safety/capability scoring;
- MMLU capability rows use capability scoring.

The label-flip indicator is:

`label_flip = solo_score != batch_score`

The text-change indicator is:

`text_change = solo_response_hash != batch_response_hash`

### 6.2 Paired comparison

Each candidate row is run under both execution modes:

1. `standard`: ordinary vLLM server startup.
2. `batch_invariant`: vLLM server startup with `VLLM_BATCH_INVARIANT=1`.

The comparison is paired because the same candidate row appears in both modes.
That pairing is essential. It removes candidate-selection, model, task, batch
size, and dispatch-mode differences from the primary contrast. The intended
treatment difference is the kernel path.

### 6.3 Dispatch reconstruction

Study D preserves TR138 dispatch semantics:

- Phase 1 rows use synchronized one-prompt HTTP requests. The target prompt is
  sent alone for the solo condition and concurrently with its reconstructed
  peers for the batched condition.
- Phase 5 rows use one OpenAI-compatible completions call with a prompt list,
  matching the true-batch prompt-list design.

The runner's metadata records this explicitly:

> synchronized dispatch uses concurrent one-prompt HTTP requests, matching
> TR138 Phase 1.

> prompt_list dispatch uses one OpenAI completions call with a prompt list,
> matching TR138 Phase 5.

### 6.4 Serving stack

| Component | Value |
|-----------|-------|
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0, 81559 MiB |
| vLLM | 0.19.1 |
| PyTorch | 2.10.0 |
| Triton | 3.6.0 |
| Transformers | 5.9.0 |
| Python | 3.11.10 |
| Backend | Native OpenAI-compatible vLLM server |
| Attention backend | FLASH_ATTN |
| Dtype | float16 |
| Max model length | 2048 |
| Max tokens | 256 |
| Request temperature | 0.0 |
| Request seed | 42 |
| GPU memory utilization | 0.80 |

The server command shape, repeated per model and mode, is:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <hf_model_id> \
  --host 0.0.0.0 \
  --port 18000 \
  --max-model-len 2048 \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --attention-backend FLASH_ATTN
```

The batch-invariant condition sets:

```bash
VLLM_BATCH_INVARIANT=1
```

Startup logs for invariant mode show vLLM registering a CUDA override from
`vllm/model_executor/layers/batch_invariant.py`, which is direct evidence that
the invariant path was not merely requested but activated by the running server.

### 6.5 Models

| Local name | Hugging Face model id | Candidate rows |
|------------|-----------------------|----------------|
| llama3.2-1b | `unsloth/Llama-3.2-1B-Instruct` | 10 |
| llama3.2-3b | `unsloth/Llama-3.2-3B-Instruct` | 19 |
| qwen2.5-1.5b | `Qwen/Qwen2.5-1.5B-Instruct` | 26 |

These are the same small-model families used in the TR138 local campaign. The
model set is appropriate for a targeted mechanism check, but not sufficient for
a universal claim about larger model families.

---

## 7. Execution and Validation

### 7.1 Run procedure

The H100 run used the RunPod launcher:

```bash
bash research/tr138_kernel_ablation/runpod_start.sh
```

The launcher performs:

1. environment setup;
2. a one-row batch-invariant startup smoke test;
3. the full standard-plus-invariant run over the 55-row candidate file;
4. validation that both required modes are present, each has 55 rows, and no
   startup errors or record errors occurred.

The full run command expands to:

```bash
python -u research/tr138_kernel_ablation/run_vllm.py \
  --server-backend native \
  --native-command "python -m vllm.entrypoints.openai.api_server" \
  --candidate-source csv \
  --candidate-files research/tr138_kernel_ablation/candidates/tr138_p1_p4_score_flips_current.csv \
  --modes standard,batch_invariant \
  --dispatch-modes original \
  --port 18000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2048 \
  --startup-timeout-s 900 \
  --request-timeout-s 300 \
  --docker-image vllm/vllm-openai:v0.19.1 \
  --attention-backend FLASH_ATTN
```

### 7.2 Validation outcome

Validation output reports:

| Check | Result |
|-------|--------|
| Total records | 110 |
| Total OK | 110 |
| Startup errors | 0 |
| Standard records | 55 |
| Standard OK | 55 |
| Standard errors | 0 |
| Batch-invariant records | 55 |
| Batch-invariant OK | 55 |
| Batch-invariant errors | 0 |

This matters because the null invariant result is only meaningful if invariant
mode actually ran. A failed invariant server, dropped rows, startup errors, or
HTTP errors would make the 0/55 uninterpretable. None occurred.

---

## 8. Results

### 8.1 Primary label result

| Mode | Label flips | Non-flips | Total | Rate | Wilson 95% CI |
|------|-------------|-----------|-------|------|---------------|
| Standard vLLM | 22 | 33 | 55 | 40.0% | [28.1%, 53.2%] |
| Batch-invariant vLLM | 0 | 55 | 55 | 0.0% | [0.0%, 6.5%] |

The standard path reproduces a high fraction of candidate label flips. The
batch-invariant path reproduces none.

### 8.2 Primary paired test

| Pair outcome | Count |
|--------------|-------|
| Standard flip, invariant flip | 0 |
| Standard flip, invariant non-flip | 22 |
| Standard non-flip, invariant flip | 0 |
| Standard non-flip, invariant non-flip | 33 |

Under the null that standard-only and invariant-only discordances are equally
likely, the 22 discordant label pairs are distributed as Binomial(22, 0.5).
The observed one-sided tail is `0.5^22 = 2.38e-7`; the exact two-sided p-value
is `2 * 0.5^22 = 4.77e-7`.

This is the formal statistical support for the mechanism claim.

### 8.3 Text-change result

| Mode | Text changes | Text-identical rows | Total | Rate | Wilson 95% CI |
|------|--------------|---------------------|-------|------|---------------|
| Standard vLLM | 25 | 30 | 55 | 45.5% | [33.0%, 58.5%] |
| Batch-invariant vLLM | 0 | 55 | 55 | 0.0% | [0.0%, 6.5%] |

The text-change result is at least as important as the label result. If
invariant mode had removed label flips while still producing text changes, the
mechanism interpretation would be weaker: the scorer might simply have become
less sensitive under invariant mode. Instead, invariant mode removes text
changes entirely on this surface. That makes the interpretation cleaner: the
target responses are stable, not merely reclassified.

The paired text-change table is:

| Pair outcome | Count |
|--------------|-------|
| Standard text change, invariant text change | 0 |
| Standard text change, invariant text identical | 25 |
| Standard text identical, invariant text change | 0 |
| Standard text identical, invariant text identical | 30 |

The exact paired two-sided p-value is `2 * 0.5^25 = 5.96e-8`.

### 8.4 Model breakdown

| Model | Candidate rows | Standard flips | Standard rate | Invariant flips | Invariant rate |
|-------|----------------|----------------|---------------|-----------------|----------------|
| llama3.2-1b | 10 | 1 | 10.0% | 0 | 0.0% |
| llama3.2-3b | 19 | 8 | 42.1% | 0 | 0.0% |
| qwen2.5-1.5b | 26 | 13 | 50.0% | 0 | 0.0% |

The result is not carried by a single model. The standard path reproduces
candidate flips in all three models, with the strongest reproduction in
qwen2.5-1.5b and llama3.2-3b. Invariant mode removes flips in all three.

### 8.5 Domain breakdown

| Domain | Candidate rows | Standard flips | Standard rate | Invariant flips | Invariant rate |
|--------|----------------|----------------|---------------|-----------------|----------------|
| Safety | 44 | 22 | 50.0% | 0 | 0.0% |
| Capability | 11 | 0 | 0.0% | 0 | 0.0% |

All reproduced standard-path label flips occur in safety-domain rows. This
should not be overread as a universal safety-over-capability law because the
surface is selected and small. It is nevertheless important for TR138: the
kernel-path-dependent instability that reproduces under H100 standard vLLM is
located entirely in the safety rows on this candidate surface.

### 8.6 Task breakdown

| Task | Candidate rows | Standard flips | Standard rate | Invariant flips | Invariant rate |
|------|----------------|----------------|---------------|-----------------|----------------|
| jailbreak_amplification | 12 | 4 | 33.3% | 0 | 0.0% |
| bbq_bias | 16 | 10 | 62.5% | 0 | 0.0% |
| truthfulqa | 16 | 8 | 50.0% | 0 | 0.0% |
| mmlu_real | 11 | 0 | 0.0% | 0 | 0.0% |

The standard-path reproductions concentrate in the safety and safety-adjacent
tasks used by TR138. The MMLU capability control rows do not reproduce label
flips in either mode.

### 8.7 Dispatch-mode and phase breakdown

| Dispatch mode | Candidate rows | Standard flips | Standard rate | Invariant flips | Invariant rate |
|---------------|----------------|----------------|---------------|-----------------|----------------|
| synchronized | 46 | 18 | 39.1% | 0 | 0.0% |
| prompt_list | 9 | 4 | 44.4% | 0 | 0.0% |

| Phase | Candidate rows | Standard flips | Standard rate | Invariant flips | Invariant rate |
|-------|----------------|----------------|---------------|-----------------|----------------|
| Phase 1 | 46 | 18 | 39.1% | 0 | 0.0% |
| Phase 5 | 9 | 4 | 44.4% | 0 | 0.0% |

This is important because it shows the standard-path reproduction is not
confined to the synchronized Phase 1 reconstruction. It also appears in the
prompt-list Phase 5 true-batch reconstruction. Invariant mode removes both.

### 8.8 Original-direction breakdown

| Original TR138 direction | Candidate rows | Standard flips | Standard rate | Invariant flips |
|--------------------------|----------------|----------------|---------------|-----------------|
| refuse_to_comply | 8 | 3 | 37.5% | 0 |
| comply_to_refuse | 4 | 1 | 25.0% | 0 |
| safety_weakened | 18 | 10 | 55.6% | 0 |
| safety_strengthened | 14 | 8 | 57.1% | 0 |
| incorrect_to_correct | 10 | 0 | 0.0% | 0 |
| correct_to_incorrect | 1 | 0 | 0.0% | 0 |

Standard-path reproductions are present in both safety-weakening and
safety-strengthening candidate directions. That is another reason the addendum
should be framed as mechanism evidence rather than as a stronger directional
prevalence claim. The kernel path controls instability; directional safety
interpretation still requires the TR138 audit layer and broader population
context.

### 8.9 Score-pair distribution

Standard vLLM score pairs:

| Solo score | Batched score | Rows |
|------------|---------------|------|
| 0.0 | 0.0 | 13 |
| 0.0 | 1.0 | 11 |
| 0.5 | 0.5 | 4 |
| 0.5 | 1.0 | 3 |
| 1.0 | 0.0 | 7 |
| 1.0 | 0.5 | 1 |
| 1.0 | 1.0 | 16 |

Batch-invariant vLLM score pairs:

| Solo score | Batched score | Rows |
|------------|---------------|------|
| 0.0 | 0.0 | 19 |
| 0.5 | 0.5 | 9 |
| 1.0 | 1.0 | 27 |

The invariant-mode score-pair distribution is diagonal. Every solo score equals
the corresponding batched score. This is the scoring-level analogue of the
0 text-change result.

---

## 9. Interpretation

### 9.1 The mechanism claim is now materially stronger

Before Study D, the strongest TR138 mechanism evidence was true-batch
validation: Phase 5 showed that the effect was not just a synchronized-request
artifact. That was useful but incomplete. True batching still uses the standard
serving path. It cannot distinguish "batch condition matters because the
standard kernel path is batch-sensitive" from other serving-stack explanations.

Study D adds the missing counterfactual. It changes the execution path and
leaves the candidate rows fixed. The collapse from 22 standard label flips to
0 invariant label flips is exactly the pattern expected under the kernel-path
mechanism.

### 9.2 The result does not depend only on scorer labels

The strongest aspect of the addendum is not just 22/55 -> 0/55 label flips. It
is 25/55 -> 0/55 text changes. The invariant path does not merely keep scores
on the same side of a classifier threshold while allowing different prose. It
produces text identity between solo and batched conditions on all 55 candidates.

That matters because TR138 v2 already showed that scorer artifacts are common.
If Study D had only shown a label collapse, the reader could still ask whether
the scorer failed differently under invariant mode. The text-collapse result
directly addresses that concern for this surface.

### 9.3 The human-adjudication question is separate

Study D does not replace human adjudication. The TR138 audit asks whether a
score flip corresponds to a genuine behavioral change. Study D asks whether
score/text flips on the current candidate surface depend on the kernel path.

Those are related but distinct:

- Human adjudication calibrates behavioral meaning.
- Study D calibrates mechanism.

The correct synthesis is therefore not "22 genuine harmful flips were proven."
The correct synthesis is "standard vLLM reproduces 22 candidate label flips and
25 text changes; batch-invariant vLLM removes both; behavioral interpretation
of any individual flip remains governed by the audit layer."

This distinction is central to the maturity of the report. It avoids the two
bad extremes: dismissing the result as "just scorer noise" and overclaiming it
as "22 confirmed harmful behavioral failures."

### 9.4 The result supports exact-stack validation

The deployment recommendation becomes sharper:

> Refusal robustness must be evaluated under the exact serving stack, including
> batch setting and kernel path.

The addendum shows that a candidate surface can look unstable under one vLLM
kernel path and stable under another. That means "the model" is not the only
unit of safety evaluation. The served model plus scheduler plus kernel path is
the operational unit.

### 9.5 The result does not create a deployment-blocking batch-risk claim

The original TR138/TR141 synthesis remains intact. Batch perturbation is real
but low-rate in the broader population, and the most defensible operational
posture is monitoring, screening, and exact-stack validation rather than a
blanket ban on batching.

Study D strengthens mechanism, not prevalence. It should increase confidence
that the measured phenomenon is technically real. It should not inflate the
estimated base rate of harmful behavior changes.

---

## 10. Claim Ledger

### 10.1 Established by Study D

| Claim | Status | Basis |
|-------|--------|-------|
| Standard vLLM/H100 reproduces candidate instability on the selected TR138 surface | Established | 22/55 label flips, 25/55 text changes |
| Batch-invariant vLLM/H100 removes candidate instability on the same surface | Established | 0/55 label flips, 0/55 text changes |
| The standard-vs-invariant difference is statistically decisive on the paired surface | Established | Exact paired p = 4.77e-7 for labels, 5.96e-8 for text |
| Current candidate flips are kernel-path dependent on the tested stack | Established for this surface | Same rows, same dispatch reconstruction, different kernel path |

### 10.2 Strengthened by Study D

| Claim | Status | Basis |
|-------|--------|-------|
| Exact-stack validation is the right test for batch-served LLM safety | Strengthened | Same candidates differ across kernel paths |
| TR138's batch-conditioned signal is not reducible to co-batch content contamination | Strengthened | Invariant path removes instability without changing content |
| The paper's Study D mechanism claim is justified | Strengthened | Reviewer-requested ablation has now been run and reported |

### 10.3 Not established by Study D

| Non-claim | Reason |
|-----------|--------|
| Batch-invariant kernels universally eliminate all refusal flips | Only 55 selected candidates, 3 models, H100, vLLM 0.19.1 |
| The full-population flip rate is 40% under standard vLLM | Candidate surface is selected on score flips |
| All 22 standard-path flips are genuine harmful behavioral changes | Human adjudication is a separate behavioral layer |
| The result applies to temperature > 0 | All requests use temperature 0 |
| The result applies to larger models or multi-GPU tensor parallelism | Not tested |
| The result measures production throughput/cost tradeoffs | Runtime cost was not the endpoint |

---

## 11. Threats to Validity

### T1. Candidate-surface selection

The 55-row surface is selected from current TR138 P1/P4 score-flip candidates.
This enriches for boundary-sensitive rows and invalidates prevalence
interpretation. Mitigation: the report frames Study D as mechanism evidence,
not as a full-population rate estimate.

### T2. Automated scoring

The label-flip metric uses deterministic automated scorers. TR138 v2 showed
that automated scorers can overstate behavioral flips. Mitigation: Study D also
reports text identity. The invariant path removes text changes entirely, which
does not depend on the scorer.

### T3. Human adjudication not rerun

Study D does not human-adjudicate all reproduced standard-path flips. This is
acceptable for the mechanism question but limits behavioral claims. Mitigation:
the report separates mechanism claims from behavioral claims.

### T4. Hardware scope

The successful invariant run is on H100 hardware. The local RTX 4080 attempt
could not start official vLLM invariant mode because of shared-memory limits.
Mitigation: the report states the H100 scope explicitly and treats RTX 4080
attempts as historical setup work, not as failed evidence against invariance.

### T5. Model scope

Only three small models are tested. Mitigation: the report makes no universal
model-family claim and explicitly calls for prospective larger-model tests.

### T6. vLLM version scope

The run uses vLLM 0.19.1. Future vLLM kernels, scheduler defaults, compilation
paths, or invariant implementations may differ. Mitigation: the artifact stack
preserves logs, metadata, version information, and startup JSON files.

### T7. Temperature scope

All requests use temperature 0. At nonzero temperature, sampling randomness may
dominate the floating-point perturbation mechanism. Mitigation: temperature > 0
is listed as a future experiment rather than implied by this result.

### T8. Dispatch reconstruction

The runner reconstructs TR138 dispatch semantics from task order and candidate
metadata. It does not replay an entire original production scheduler trace.
Mitigation: Phase 1 and Phase 5 dispatch modes are reconstructed separately,
and both show the same standard-to-invariant collapse.

### T9. Throughput and cost not measured

Study D tests safety/reproducibility behavior, not the latency or throughput
cost of invariant kernels. Mitigation: operational guidance says to benchmark
the invariant path before production adoption.

### T10. Local artifact privacy

The portable candidate summary avoids raw prompts and archived completions, but
the full local `vllm_records.jsonl` contains generated responses for analysis.
Mitigation: public or portable releases should prefer summaries, hashes, and
aggregate tables unless raw text release is explicitly approved.

---

## 12. Reproducibility

### 12.1 Required inputs

| Input | Path |
|-------|------|
| Candidate CSV | `research/tr138_kernel_ablation/candidates/tr138_p1_p4_score_flips_current.csv` |
| Task definitions | `research/tr138/tasks/` |
| Runner | `research/tr138_kernel_ablation/run_vllm.py` |
| RunPod launcher | `research/tr138_kernel_ablation/runpod_launch.sh` |
| Validator | `research/tr138_kernel_ablation/validate_runpod.py` |

### 12.2 Minimal rerun command

```bash
export HF_TOKEN="<token>"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
bash research/tr138_kernel_ablation/runpod_start.sh
```

### 12.3 Direct full-run command

```bash
python -u research/tr138_kernel_ablation/run_vllm.py \
  --server-backend native \
  --native-command "python -m vllm.entrypoints.openai.api_server" \
  --candidate-source csv \
  --candidate-files research/tr138_kernel_ablation/candidates/tr138_p1_p4_score_flips_current.csv \
  --modes standard,batch_invariant \
  --dispatch-modes original \
  --port 18000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2048 \
  --startup-timeout-s 900 \
  --request-timeout-s 300 \
  --docker-image vllm/vllm-openai:v0.19.1 \
  --attention-backend FLASH_ATTN
```

### 12.4 Validation command

```bash
python research/tr138_kernel_ablation/validate_runpod.py \
  --require-modes standard,batch_invariant \
  --expected-rows-per-mode 55
```

Expected validation:

- `n_records = 110`
- `n_ok = 110`
- `startup_error_count = 0`
- standard mode has 55 records and 0 errors
- batch-invariant mode has 55 records and 0 errors

### 12.5 Artifact inventory

| Artifact | Purpose |
|----------|---------|
| `summary.json` | Aggregate result, by-mode counts, flip list |
| `metadata.json` | Run configuration, candidate hash, modes, dispatch notes |
| `vllm_records.jsonl` | Row-level records with scores, hashes, responses, dispatch metadata |
| `selected_candidates.jsonl` | Sanitized selected-row metadata |
| `runpod_launch.log` | Package/GPU environment and execution transcript |
| `startup_*.json` | Per-mode/per-model startup command and readiness evidence |
| `native_*.log` | vLLM server logs, including invariant kernel activation evidence |

---

## 13. Consequence for the Paper and Future Work

For the workshop paper, the Study D result can be compressed into one or two
sentences because paper space is limited:

> In a targeted H100/vLLM ablation over 55 current score-flip candidates,
> standard vLLM reproduced 22 label flips and 25 text changes, while
> `VLLM_BATCH_INVARIANT=1` produced 0 label flips and 0 text changes.

The report stack needs the longer version because future work will build on the
mechanism. The key future experiments are:

1. Prospective full-population invariant rerun, not candidate-selected.
2. Larger model families, including 7B, 14B, and larger if hardware permits.
3. Multiple GPU architectures: H100, H200, B200, A100, RTX 4090/5090 where
   invariant kernels start successfully.
4. Temperature > 0 comparisons to separate kernel determinism from stochastic
   sampling.
5. Throughput and latency cost measurement for invariant mode.
6. Independent human adjudication of standard-path reproduced safety flips.
7. Production-scheduler replay with real request timing rather than reconstructed
   dispatch groups.

---

## 14. Final Synthesis

Study D is not a decorative add-on. It is the mechanism bridge that turns TR138
from "we found low-rate batch-conditioned score flips and audited some of them"
into "we found low-rate batch-conditioned score flips, calibrated their
behavioral meaning, verified serving-stack relevance, and showed that the
current candidate instability collapses under a batch-invariant kernel path."

The result should be cited carefully:

- Strong claim: current TR138 P1/P4 score-flip candidates are kernel-path
  dependent on the tested H100/vLLM stack.
- Moderate extension: exact-stack safety validation should include kernel path,
  not just model and prompt.
- Non-claim: batch-invariant kernels universally solve refusal robustness.

That is the right epistemic posture for future research: strong where the
paired evidence is strong, bounded where the surface is selected, and explicit
about which experiment comes next.
