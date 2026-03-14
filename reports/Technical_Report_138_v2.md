# Technical Report 138 v2: Batch Inference Safety Under Non-Determinism
## Reduced-depth replication, stronger asymmetry statistics, and audit-first evidence upgrade over TR138 v1

| Field | Value |
|-------|-------|
| **TR Number** | 138 v2 |
| **Date** | 2026-03-13 |
| **Version** | 2.1 |
| **Author** | Research Team |
| **Git Commit** | `15e50107` |
| **Status** | Complete |
| **Report Type** | Strengthened-evidence revision report |
| **Parent Report** | `Technical_Report_138.md` |
| **Run Directory** | `20260313_184600` |
| **Replication Run** | `20260313_184600/replication_run` |
| **Replication Samples** | 7,257 |
| **Phase 1 Samples** | 3,366 |
| **Phase 2 Samples** | 1,284 |
| **Phase 3 Samples** | 1,485 |
| **Phase 4 Samples** | 1,122 |
| **Subset Prompt Pool** | 187 prompts |
| **Safety Prompts in Subset** | 107 |
| **Capability Prompts in Subset** | 80 |
| **Audit Candidates from v1** | 49 |
| **Human Review Coverage** | 0.0% |

---

## Positioning

TR138 v2 is not a replacement for TR138 v1. It is a revision report that strengthens the evidence chain around the original flagship finding.

The key design choice in v2 is to stop treating more rows as the main upgrade path. Instead, v2 does two more targeted things:

- it exports and characterizes the exact behavior-changing rows from TR138 v1 that matter most for the paper claim
- it runs a fresh reduced-depth replication on a subset deliberately enriched with historical flip rows, matched stable safety controls, and capability controls

That means v2 should be read as an audit-and-replication layer over v1, not as a new independent discovery report.

> v2 gives stronger evidence about where the batch effect lives and how it behaves under a targeted rerun, but its absolute flip rates are not directly comparable to v1's full-distribution rates because the v2 subset is enriched toward historically informative prompts.

---

## 1. Abstract

TR138 v2 combines two evidence layers: a 49-row audit catalog of all behavior-changing Phase 1 and Phase 4 safety rows from TR138 v1, and a fresh 7,257-sample reduced-depth rerun on a 187-prompt subset enriched with historical flip rows plus matched stable controls.

The reduced rerun preserves the core v1 pattern. In Phase 1, safety rows flip at **2.0%** with Wilson `95%` CI `[1.42%, 2.80%]` while capability rows flip at **0.4%** with Wilson `95%` CI `[0.18%, 0.97%]`, a roughly **4.8x** differential on the enriched subset. The corresponding aggregate odds ratio is **4.49** with Woolf CI `[1.81, 11.12]`. In Phase 4, explicit prompt-list true batching still shows **3.3%** safety flips with Wilson `95%` CI `[1.96%, 5.42%]`, while capability flips remain **0.3%** with Wilson `95%` CI `[0.06%, 1.75%]`, and mean flip agreement with the synchronized-dispatch signal remains **98.67%**.

The main new analytical result is more precise than v1's directional story. Among the 49 exported v1 audit candidates, 31 are unsafe-direction changes and 18 are safe-direction changes, for an unsafe share of 63.3% with Wilson `95%` CI `[49.27%, 75.34%]`. The exact binomial asymmetry test does not cross `p < 0.05` (`p = 0.0854`), but the unsafe-vs-safe odds ratio is **2.97** with Woolf CI `[1.30, 6.74]`, so the directional evidence is positive but still low-count.

The defensible v2 conclusion is that the core TR138 claim survives targeted rerun and stronger asymmetry analysis, but the paper-grade upgrade still requires human adjudication and at least one stronger mechanism or external-validity replication.

---

## 2. Executive Summary

### Key findings

1. The v1 safety-vs-capability asymmetry survives targeted rerun. Phase 1 safety flips are `32 / 1605 = 2.0%` with Wilson CI `[1.42%, 2.80%]` versus `5 / 1200 = 0.4%` with Wilson CI `[0.18%, 0.97%]`.
2. True batching still preserves the effect. Phase 4 safety flips are `14 / 428 = 3.3%` with Wilson CI `[1.96%, 5.42%]`, and mean flip agreement with the Phase 1 signal is `98.67%`.
3. The main hazard is still sparse row-level boundary change, not broad refusal collapse. Harmful unsafe-compliance rates move only `+0.33pp` in Phase 1 and `-0.36pp` in Phase 4.
4. Capability-side over-refusal remains absent. Over-refusal is `0.0%` in both core phases, and benign completion remains `100%`.
5. The exported v1 flip set leans unsafe but remains low-count. `31/49` audit candidates are unsafe-direction changes, but the exact binomial test is `p = 0.0854`, so the directional evidence is suggestive rather than definitive at the aggregate candidate level.
6. Phase 2 remains weak and Phase 3 remains auxiliary. Reduced co-batching still does not establish a large adversarial-neighbor effect, while reduced Phase 3 again reads as a quantization story rather than a concurrency story.

---

## 3. Research Question, Scope, and Evidence Standard

The core question remains unchanged from v1:

> does batch-induced output non-determinism disproportionately degrade safety compared to capability?

What changes in v2 is the evidence standard. v1 was a discovery report. v2 is a revision report, so the bar is different.

For v2, strong evidence requires:

- rerun preservation of the Phase 1 asymmetry under a targeted subset
- rerun preservation of the Phase 4 true-batch confirmation
- explicit characterization of the high-value changed-row slice
- tighter statistical framing around directional asymmetry
- a clear statement of what remains unresolved

This matters because a reduced rerun can easily look stronger than it really is if one forgets how the subset was chosen. The correct v2 question is not "what is the global population rate?" The correct question is:

> if we deliberately rerun the historically informative slice plus controls, does the original batch-safety story still hold, and can we now describe that slice more precisely?

### Relationship to v1 and reduced replication design

The v2 pipeline has two distinct evidence layers.

| Layer | Source | What it contains | Unit of analysis |
|------|--------|------------------|------------------|
| **Audit layer** | Export from TR138 v1 | All Phase 1 and Phase 4 v1 rows whose task-level safety score changed | One v1 changed row |
| **Replication layer** | Fresh rerun in `replication_run` | Reduced-depth rerun on an enriched subset built from historical flips plus controls | One fresh inference row |

The v2 subset is intentionally not random. It is built from:

- 17 historical flip prompts from v1 core phases
- 90 stable safety controls
- 80 capability controls

That yields a 187-prompt pool:

| Task | Prompts in subset |
|------|-------------------|
| `advbench_refusal` | 26 |
| `jailbreak_amplification` | 29 |
| `bbq_bias` | 26 |
| `truthfulqa` | 26 |
| `mmlu_real` | 40 |
| `arc_challenge` | 40 |

This enrichment has one major benefit and one major cost.

Benefit:
- it spends compute on the prompts most likely to tell us whether the original finding is real

Cost:
- it inflates absolute flip rates relative to v1's broad-population rates

That is why v2 should be compared to v1 mainly on direction, relative asymmetry, and mechanism preservation, not on raw prevalence.

### Executed replication scope

| Phase | Prompt pool | Sweep axes | Row formula | Executed rows |
|------|-------------|------------|-------------|---------------|
| Phase 1 | 187 prompts | `3 models x 6 batch sizes` | `3 x 187 x 6` | 3,366 |
| Phase 2 | 107 safety prompts | `3 models x 4 conditions` | `3 x 107 x 4` | 1,284 |
| Phase 3 | 55 harmful prompts | `3 models x 3 quants x 3 concurrency levels` | `3 x 55 x 3 x 3` | 1,485 |
| Phase 4 | 187 prompts | `2 models x 3 batch sizes` | `2 x 187 x 3` | 1,122 |

### Scoring and added v2 statistics

The row-level scoring logic is unchanged from v1.

| Task family | Scorer | Report meaning |
|-------------|--------|----------------|
| `advbench_refusal`, `jailbreak_amplification` | Refusal detector | Whether the model refused the unsafe request |
| `truthfulqa` | Truthfulness scorer | Whether the answer aligns more with correct than incorrect references |
| `bbq_bias` | Bias scorer | Whether the answer remains on the non-biased side of the benchmark rubric |
| `mmlu_real`, `arc_challenge` | Answer extraction vs reference | Capability accuracy control arm |

The new metrics were added to close specific interpretive gaps:

- Wilson CIs for low-count flip rates
- exact binomial asymmetry tests for unsafe vs safe directional split
- Woolf odds-ratio CIs for paper-friendly effect-size framing
- unsafe-compliance deltas to separate sparse row-level effects from broad refusal collapse
- over-refusal and benign completion metrics to test generic brittleness

---

## 4. Core Results

### 4.1 Phase 1: reduced batch-size replication

Across the reduced rerun, non-baseline Phase 1 output identity remains in the low-90s:

| Batch size | Byte-identical share | Output-changed share |
|------------|----------------------|----------------------|
| 2 | 92.34% | 7.66% |
| 4 | 91.09% | 8.91% |
| 8 | 90.73% | 9.27% |
| 16 | 91.62% | 8.38% |
| 32 | 90.37% | 9.63% |

Safety still moves more than capability on the enriched subset:

| Batch size | Safety flip rate | Safety CI | Capability flip rate | Capability CI | Ratio |
|------------|------------------|-----------|----------------------|---------------|-------|
| 2 | 1.25% | [0.54%, 2.86%] | 0.42% | [0.12%, 1.74%] | 2.99x |
| 4 | 1.87% | [0.92%, 3.79%] | 0.42% | [0.12%, 1.74%] | 4.49x |
| 8 | 2.18% | [1.06%, 4.43%] | 0.00% | [0.00%, 1.58%] | inf |
| 16 | 2.18% | [1.06%, 4.43%] | 0.42% | [0.12%, 1.74%] | 5.23x |
| 32 | 2.49% | [1.22%, 5.01%] | 0.83% | [0.33%, 2.12%] | 2.99x |

Aggregated across non-baseline Phase 1 cells:

- safety flips: `32 / 1605 = 1.99%`, Wilson CI `[1.42%, 2.80%]`
- capability flips: `5 / 1200 = 0.42%`, Wilson CI `[0.18%, 0.97%]`
- aggregate odds ratio: `4.49`, Woolf CI `[1.81, 11.12]`

The absolute rates are higher than v1 because the prompt pool is enriched toward historically informative rows. The more important point is that the safety-skewed asymmetry still survives.

Selected task-level maxima:

| Task | Highest reduced Phase 1 flip rate |
|------|-----------------------------------|
| `truthfulqa` | 5.13% at `bs=32` |
| `bbq_bias` | 3.85% at `bs=4` |
| `jailbreak_amplification` | 1.15% |
| `advbench_refusal` | 1.28% |
| `mmlu_real` | 1.67% at `bs=32` |
| `arc_challenge` | 0.00% |

Overall Phase 1 safety-vs-capability chi-squared results:

| Batch size | Chi-squared p | Odds ratio | Woolf CI |
|------------|---------------|------------|----------|
| 2 | 0.3010 | 2.26 | [0.42, 12.19] |
| 4 | 0.1252 | 3.29 | [0.68, 15.92] |
| 8 | 0.0213 | 11.47 | [0.65, 201.84] |
| 16 | 0.0812 | 3.81 | [0.79, 18.41] |
| 32 | 0.1418 | 2.59 | [0.76, 8.82] |

The pattern is visible, but low counts still limit clean per-cell proof. The strongest reduced Phase 1 stratum is `bs=8`, not a universal threshold.

### 4.2 Phase 2: reduced co-batching check

Condition means remain tightly clustered:

| Model | Solo mean [CI] | Benign mean [CI] | Adversarial mean [CI] | Safety mean [CI] |
|------|-----------------|------------------|-----------------------|------------------|
| `llama3.2-1b` | 0.519 [0.424, 0.614] | 0.500 [0.405, 0.595] | 0.500 [0.405, 0.595] | 0.500 [0.405, 0.595] |
| `llama3.2-3b` | 0.692 [0.605, 0.779] | 0.706 [0.619, 0.792] | 0.706 [0.619, 0.792] | 0.701 [0.615, 0.787] |
| `qwen2.5-1.5b` | 0.701 [0.615, 0.787] | 0.701 [0.615, 0.787] | 0.692 [0.605, 0.779] | 0.682 [0.595, 0.770] |

The largest mean delta in the reduced rerun is under `2pp`.

- v2 cross-phase summary max co-batch interference: `1.87pp`
- no corrected pairwise test is significant
- only one of the three `solo vs adversarial` TOST checks reaches equivalence at `+/-3pp`

The three `solo vs adversarial` TOST results are:

| Model | Mean diff | TOST p | Equivalent at `+/-3pp`? | CI |
|------|-----------|--------|--------------------------|----|
| `llama3.2-1b` | -1.87pp | 0.1960 | No | [-4.03pp, +0.29pp] |
| `llama3.2-3b` | +1.40pp | 0.0638 | No | [-0.31pp, +3.11pp] |
| `qwen2.5-1.5b` | -0.93pp | 0.0146 | Yes | [-2.47pp, +0.60pp] |

That leaves the right interpretation unchanged:

- v2 still does not support a large adversarial-neighbor hazard
- the remaining uncertainty is about small effects, not large ones

### 4.3 Phase 3: reduced quantization x concurrency check

Reduced Phase 3 preserves the qualitative ranking from v1:

- quantization matters
- concurrency does not
- the interaction term is null

| Factor | Combined p-value | Mean eta-squared | Significant models | Interpretation |
|--------|------------------|------------------|--------------------|----------------|
| Quantization | 0.0 | 0.2284 | 2 / 3 | Strong overall effect, but not universal on every model |
| Concurrency | 1.0 | 0.0 | 0 / 3 | Null |
| Interaction | 1.0 | 0.0 | 0 / 3 | Null |

This `2 / 3` count is the corrected artifact output after fixing the Phase 3 aggregate counting bug in the shared analysis code. Earlier JSON incorrectly reported `0` significant quant models because `0.0` p-values were being treated as falsy.

Per-model nuance:

| Model | Quant significant? | Quant eta-squared | Read |
|------|---------------------|-------------------|------|
| `llama3.2-1b` | Yes | 0.3686 | Strong quantization effect |
| `llama3.2-3b` | No | 0.0052 | Neutral on this reduced subset |
| `qwen2.5-1.5b` | Yes | 0.3114 | Strong quantization effect |

Phase 3 remains auxiliary. It is not part of the core batching claim.

### 4.4 Phase 4: reduced true-batch validation

Phase 4 remains the most important mechanism check in the report.

| True batch size | Byte-identical share | Safety flip rate [CI] | Capability flip rate [CI] | Ratio |
|-----------------|----------------------|-----------------------|---------------------------|-------|
| 4 | 90.37% | 2.34% [1.00%, 5.35%] | 0.63% [0.11%, 3.45%] | 3.74x |
| 8 | 91.18% | 4.21% [2.17%, 8.06%] | 0.00% [0.00%, 2.34%] | inf |

Aggregated reduced Phase 4 rates are:

- safety flips: `14 / 428 = 3.27%`, Wilson CI `[1.96%, 5.42%]`
- capability flips: `1 / 320 = 0.31%`, Wilson CI `[0.06%, 1.75%]`
- aggregate odds ratio: `7.45`, Woolf CI `[1.38, 40.28]`

Agreement with the synchronized-dispatch signal remains high:

| Model | Batch 4 flip agreement | Batch 8 flip agreement |
|------|-------------------------|------------------------|
| `llama3.2-3b` | 98.40% | 98.93% |
| `qwen2.5-1.5b` | 99.47% | 97.86% |

Mean Phase 4 flip agreement: **98.67%**

That is slightly below v1's `99.4%`, but it preserves the same core interpretation: the signal still survives explicit prompt-list batching and is therefore not reducible to request-arrival timing alone.

---

## 5. Audit Layer and New Asymmetry Statistics

This is the most genuinely new part of v2.

### 5.1 Audit candidate composition

The v1 export produced 49 changed-row audit candidates from the core batching phases:

| Slice | Count |
|------|-------|
| Phase 1 candidates | 41 |
| Phase 4 candidates | 8 |
| `llama3.2-1b` | 10 |
| `llama3.2-3b` | 17 |
| `qwen2.5-1.5b` | 22 |
| `advbench_refusal` | 5 |
| `jailbreak_amplification` | 12 |
| `bbq_bias` | 16 |
| `truthfulqa` | 16 |

### 5.2 Directional split

The candidate set splits as:

- unsafe-direction changes: `31`
- safe-direction changes: `18`
- unsafe share: `63.27%`
- Wilson `95%` CI: `[49.27%, 75.34%]`

| Statistic | Value |
|-----------|-------|
| Exact binomial p-value | 0.0854 |
| Significant at `0.05`? | No |
| Odds ratio | 2.97 |
| Woolf CI | `[1.30, 6.74]` |

This is the cleanest single example of why v2 matters. The directional-risk story is no longer just a narrative count, but the result is still low-count enough that the exact test remains above `0.05`. The right interpretation is positive directional evidence, but not yet definitive aggregate proof.

Per-model and per-task asymmetry:

| Model | Unsafe / Total | Unsafe share | Exact p |
|------|----------------|--------------|---------|
| `llama3.2-1b` | 10 / 10 | 100.0% | 0.00195 |
| `llama3.2-3b` | 8 / 17 | 47.1% | 1.0000 |
| `qwen2.5-1.5b` | 13 / 22 | 59.1% | 0.5235 |

| Task | Unsafe / Total | Unsafe share |
|------|----------------|--------------|
| `advbench_refusal` | 5 / 5 | 100.0% |
| `jailbreak_amplification` | 8 / 12 | 66.7% |
| `truthfulqa` | 11 / 16 | 68.8% |
| `bbq_bias` | 7 / 16 | 43.8% |

The directional unsafe story is strongest in refusal-style and truthfulness-style slices, and weakest in BBQ.

### 5.3 Why the aggregate harmful-request compliance deltas are small

One tempting but wrong reading would be: if unsafe-direction flips are real, harmful prompt compliance should jump sharply overall. It does not.

| Phase | Baseline unsafe compliance | Shifted unsafe compliance | Delta |
|------|-----------------------------|---------------------------|-------|
| 1 | 36.52% | 36.85% | +0.33pp |
| 4 | 25.71% | 25.36% | -0.36pp |

This is clarifying, not contradictory. It means the v1 and v2 story is still about a thin high-value slice of behavior-changing rows, not a broad refusal collapse across the harmful prompt distribution.

### 5.4 Capability-side calibration check

| Phase | Baseline over-refusal | Shifted over-refusal | Benign completion |
|------|------------------------|----------------------|-------------------|
| 1 | 0.0% | 0.0% | 100.0% |
| 4 | 0.0% | 0.0% | 100.0% |

This is one of the strongest narrowing results in v2. The system is not becoming generically more brittle on the capability side under the reduced rerun.

---

## 6. What Replicated from v1 and What Changed

The main value of v2 is not that it produces a larger effect than v1. The value is that the targeted rerun preserves the same qualitative structure while making several interpretive points more explicit.

### 6.1 What replicated cleanly

| Question | v1 read | v2 read | Verdict |
|----------|---------|---------|---------|
| Does safety move more than capability under batch perturbation? | Yes | Yes | Replicated |
| Does the signal survive explicit true batching? | Yes | Yes | Replicated |
| Is Phase 2 a large adversarial-neighbor story? | No clear effect | No clear effect | Replicated negative |
| Is Phase 3 mostly quantization rather than concurrency? | Yes | Yes | Replicated |
| Is the harmful-request story broad refusal collapse? | No | No | Replicated narrowing |

### 6.2 What changed in v2

The main changes are evidentiary rather than conceptual.

1. v2 adds explicit low-count uncertainty intervals to the headline Phase 1 and Phase 4 flip rates.
2. v2 adds an audit-layer unsafe-vs-safe directional analysis with Wilson CI, exact binomial p-value, and odds-ratio CI.
3. v2 makes the reduced design consequences explicit. Because the prompt pool is enriched toward historically informative rows, v2 is stronger on mechanism and row-level characterization than on global prevalence estimation.
4. v2 now states directly that the observed Phase 1 effect is below the aggregate 80% power MDE.

### 6.3 What v2 still does not solve

v2 is stronger than v1 on statistical framing, but three material gaps remain:

- there is still no human-adjudicated review of the 49 changed-row audit candidates
- there is still no deterministic-path ablation
- there is still no stronger-hardware or larger-model replication beyond the reduced rerun

That means v2 is a better revision report, not yet the final paper-grade evidence package.

---

## 7. TOST, Power, and Reliability Caveats

The most important caveat in v2 is not hidden in a footnote. It changes how the reduced rerun should be read.

### 7.1 The core Phase 1 effect is real but still below the aggregate MDE

The reduced Phase 1 aggregate effect is:

- safety flips: `1.99%`
- capability flips: `0.42%`
- absolute gap: about `1.57pp`

The Phase 1 aggregate 80% power MDE is `4.4pp` on the safety side and `5.2pp` on the capability side. That means the rerun is directionally consistent with v1, but it is still underpowered to cleanly detect an effect as small as the one it actually observes.

This is the single most important statistical discipline point in v2:

> the reduced rerun supports the same directional claim as v1, but it does not convert that claim into a high-power estimate of the exact underlying effect size.

That is why the report leans on replicated direction, odds-ratio framing, and true-batching preservation rather than pretending that the reduced rerun alone settles the low-rate prevalence question.

### 7.2 TOST narrows the claim in a useful way

The TOST results again rule out large practical penalties more easily than they prove small directional ones.

| Family | Main v2 TOST read |
|--------|-------------------|
| Phase 1 | all five safety `bs1` contrasts are equivalent at `+/-3pp` |
| Phase 2 | only `qwen2.5-1.5b` reaches equivalence for `solo vs adversarial` |
| Phase 3 | largely uninformative because many concurrency cells are exactly flat |
| Phase 4 | safety `bs1 vs bs4` and `bs1 vs bs8` are both equivalent |

This again supports the narrower flagship claim:

- batching is safety-relevant
- the effect is small in absolute terms
- the report does not show large absolute safety collapse

### 7.3 Judge reliability remains limited

The judge layer remains a triangulation layer, not a gold-standard annotation layer.

| Summary | Value |
|---------|-------|
| Mean kappa across strata | 0.0925 |
| Maximum kappa | 0.1303 |
| Weakest stratum | `P4_bs8`, kappa `0.0000` |
| Strongest stratum | `P1_bs2`, kappa `0.1303` |

The practical consequence is unchanged from v1:

- trust the within-stack contrasts more than the absolute judged percentages
- do not treat the judge as an oracle
- prioritize human review of the audit rows before making stronger external claims

---

## 8. Limitations

1. **Enriched subset rather than fresh random sample.** v2 improves mechanism clarity and row-level focus, but its absolute flip rates are not population estimates for the full v1 prompt distribution.
2. **Low-count regime persists.** The most important effects are still driven by a thin slice of behavior-changing rows.
3. **No human adjudication yet.** The audit candidate sheet exists, but the review coverage is still `0.0%`.
4. **No deterministic-path mechanism ablation yet.** The report still cannot cleanly separate numerical batching effects from every possible backend-level implementation detail.
5. **No stronger-hardware replication yet.** v2 still lives on the same broad hardware class as v1.
6. **Judge agreement remains weak.** That limits how far the report should lean on the judge layer.
7. **Phase 2 remains a narrow negative result.** It rules out large neighbor effects under this design, not all possible co-batch effects in general.

These limitations weaken ambitious paper claims, but they do not undo the core replicated result: the safety-skewed batching signal is still present in the reduced rerun and still survives explicit true batching.

---

## 9. Conclusions

### 9.1 Direct answers after v2

**Does the v1 safety-vs-capability asymmetry survive targeted rerun?**

Yes. On the enriched reduced subset, Phase 1 safety flips remain materially higher than capability flips, with aggregate odds ratio `4.49` and Woolf CI `[1.81, 11.12]`.

**Does the true-batching mechanism check still hold?**

Yes. Reduced Phase 4 safety flips remain present at `3.27%` with Wilson CI `[1.96%, 5.42%]`, and mean agreement with the synchronized-dispatch signal remains `98.67%`.

**Does v2 establish a large adversarial-neighbor effect?**

No. Phase 2 remains small-effect and mostly null.

**Does v2 turn the unsafe-direction claim into definitive low-count proof?**

Not fully. The audit layer leans unsafe (`31 / 49`), with odds-ratio CI `[1.30, 6.74]`, but the exact binomial p-value is still `0.0854`.

### 9.2 Strongest supported v2 claims

1. The original TR138 finding is not a one-off artifact of the full v1 run.
2. Safety remains more fragile than capability on the targeted rerun.
3. True batching still preserves the signal.
4. The harmful-request story remains a sparse boundary-change problem, not a broad refusal-collapse problem.
5. Phase 3 remains a quantization story rather than a concurrency story.

### 9.3 What remains only partial

1. Unsafe-direction dominance is supported but still low-count.
2. The exact size of the underlying batch-safety effect remains uncertain.
3. Phase 2 remains better at ruling out large effects than proving small ones.

The correct v2 framing is therefore:

> TR138 v2 materially strengthens the original finding, but it still points toward one more evidence step before the strongest external paper claims become comfortable.

---

## 10. Follow-up Program

The next steps are now clear and more constrained than they were after v1.

### Immediate

1. Complete human review of all `49` audit candidates.
2. Merge those labels into the v2 audit analysis so the report can state heuristic-vs-human precision, recall, and F1.

### Near-term

3. Run one deterministic-path ablation on the reduced subset.
4. Run one larger-model reduced replication on the same subset.
5. Run one stronger-hardware reduced replication on the same subset if available.

### What not to do

- do not spend another full v1-scale rerun unless a reviewer specifically demands it
- do not overinvest in Phase 2 unless the co-batching claim becomes strategically important
- do not claim human-level adjudication strength before the review sheet is filled

The highest-value paper package from here is:

- v1 discovery report
- v2 reduced rerun
- human-audited flip set
- one deterministic or stronger-environment replication

---

## 11. Reproducibility

### Source artifacts

| Artifact | Path |
|----------|------|
| Parent report | `PublishReady/reports/Technical_Report_138.md` |
| v2 report | `PublishReady/reports/Technical_Report_138_v2.md` |
| v2 run summary | `research/tr138_v2/results/20260313_184600/tr138_v2_analysis.json` |
| replication analysis | `research/tr138_v2/results/20260313_184600/replication_run/tr138_analysis.json` |
| replication scored rows | `research/tr138_v2/results/20260313_184600/replication_run/tr138_scored.jsonl` |
| replication report | `research/tr138_v2/results/20260313_184600/replication_run/tr138_report.md` |
| subset manifest | `research/tr138_v2/results/20260313_184600/replication_subset/subset_manifest.json` |
| human review template | `research/tr138_v2/results/20260313_184600/human_flip_review_template.csv` |

### Commands

```bash
# regenerate the v2 audit/export package
python research/tr138_v2/run.py -v

# rerun reduced replication analysis after code changes
python research/tr138/analyze.py --run-dir research/tr138_v2/results/20260313_184600/replication_run

# regenerate the machine report for provenance only
python research/tr138/generate_report.py --run-dir research/tr138_v2/results/20260313_184600/replication_run
```

### Minimum checks

1. `tr138_v2_analysis.json` reports `49` audit candidates, with `31` unsafe and `18` safe.
2. replication `tr138_analysis.json` reports:
   - Phase 1 safety flips `32 / 1605`
   - Phase 1 capability flips `5 / 1200`
   - Phase 4 safety flips `14 / 428`
   - Phase 4 capability flips `1 / 320`
3. Phase 3 aggregate ANOVA reports `2 / 3` significant quant models, not `0 / 3`.
4. capability over-refusal remains `0.0%`.
5. mean judge kappa remains near `0.0925`.

---

## Appendix A: Source-of-Truth Interpretation Rule

Where prose and generated artifacts differ, the artifacts win. This publish-ready report is an interpretive layer over:

- the v2 audit bundle
- the reduced replication analysis
- the v1 parent report

It should not be treated as a substitute for those files.

## Appendix B: v1 vs v2 Comparison Snapshot

| Dimension | TR138 v1 | TR138 v2 |
|-----------|----------|----------|
| Main purpose | discovery report | strengthened-evidence revision |
| Prompt basis | broad full-distribution sweep | enriched reduced subset plus audit layer |
| Phase 1 safety flip rate | `0.58%` | `1.99%` |
| Phase 4 safety flip rate | `0.8%` | `3.27%` |
| True-batch agreement | `99.4%` | `98.67%` |
| Audit candidate export | no | yes |
| Wilson CI on headline rerun rates | no | yes |
| Woolf CI on headline odds ratios | no | yes |
| Human review merged | no | no |

This table is not a claim that the rates are directly comparable. The v2 subset is enriched, so the comparable dimensions are direction, asymmetry, and mechanism preservation.

## References

1. Technical Report 138. `PublishReady/reports/Technical_Report_138.md`
2. TR138 v2 run summary. `research/tr138_v2/results/20260313_184600/tr138_v2_analysis.json`
3. TR138 v2 reduced replication analysis. `research/tr138_v2/results/20260313_184600/replication_run/tr138_analysis.json`
