# Technical Report 144: Speculative Decoding x Safety
## Protocol And Publication Contract For A Main-Track Empirical Study

| Field | Value |
|---|---|
| **TR Number** | 144 |
| **Date** | 2026-04-02 |
| **Version** | 1.0 |
| **Status** | Protocol locked; execution pending |
| **Report Type** | Publication-build report |
| **Source Directory** | `research/tr144/` |
| **Paper Package** | `papers/speculative_decoding_safety/` |
| **Current Implemented Core** | 17,555 primary model outputs |
| **Locked Submission Floor** | 55,895 primary model outputs |
| **Stretch Target** | 65,765 primary model outputs |
| **Adjudication Target** | 5,000 labels |

---

## Abstract

TR144 asks whether speculative decoding changes safety outcomes by allowing a
smaller draft model to shape the token sequence later verified by a larger
target model. The existing codebase already implements a five-phase confirmatory
study spanning baseline decoding, rejection-sampling equivalence, typical
acceptance, speculation-length dose response, and acceptance-rate telemetry.
What it did not yet contain was a publication contract that converts that
experiment into a main-track paper line.

This report supplies that missing layer. It freezes the paper thesis, defines a
machine-readable evidence contract, and locks a submission floor of `55,895`
primary outputs plus `5,000` adjudication labels. The minimum result-bearing
package consists of the existing `17,555`-output confirmatory core, a
`25,560`-output seed-replication block, and a `12,780`-output dtype-robustness
block. A `9,870`-output non-zero-temperature stretch block remains optional.

The key constraint is explicit: no TR144 run artifacts exist yet. This report
therefore does not claim empirical findings. Instead, it upgrades TR144 into a
paper-ready empirical program whose execution can later support submission-grade
claims without redesigning the study in midstream.

---

## Positioning

TR144 sits at a productive junction in the Banterhearts program.

- TR134 established that smaller models are generally more fragile on safety.
- TR138 and TR143 showed that serving-time controls can perturb safety
  outcomes.
- TR144 asks the next systems-safety question: if throughput gains now rely on
  draft-then-verify generation, is speculative decoding itself part of the
  deployment-time safety surface?

This line is narrower than a generic "serving is unsafe" story and more
operational than a pure systems paper. It links a widely deployed optimization
to a directly testable alignment question.

## Research Question

> Does speculative decoding's draft-then-verify paradigm leak unsafe behavior
> from weaker draft models into the final verified output?

The study is worth paperization because it combines three properties that
reviewers usually want but protocol drafts often lack:

1. a hard theoretical null for rejection sampling at temperature 0
2. an operationally realistic relaxed-acceptance condition that can actually
   diverge from target-only decoding
3. per-request telemetry that helps interpret when draft and target disagree

## Hypotheses

- **H0:** Speculative decoding is safety-equivalent to target-only decoding.
- **H1:** Rejection sampling violates theoretical output equivalence because of
  arithmetic or implementation effects.
- **H2:** Typical acceptance produces measurable safety drift.
- **H3:** Longer speculation windows amplify the drift.
- **H4:** Acceptance-rate telemetry differs between safety and capability
  prompts, indicating stronger draft-target disagreement on safety-critical
  tokens.

## What Is Already Implemented

The `research/tr144/` folder already supports:

- benchmark preparation
- full prompt loading across six tasks
- speculative and non-speculative vLLM execution
- per-request metrics polling for speculative phases
- 23-pass structured analysis
- markdown report generation
- cross-TR baseline validation against prior result lines

In other words, TR144 already had an experiment. What it lacked was the
publication layer that makes the eventual evidence tractable for a paper.

## Core Confirmatory Design

The current implemented core contains five phases:

| Phase | Purpose | Outputs |
|---|---|---:|
| 1 | Standalone target and draft baselines | 4,775 |
| 2 | Rejection-sampling equivalence audit | 2,865 |
| 3 | Typical-acceptance primary comparison | 2,865 |
| 4 | Speculation-length dose response | 7,050 |
| 5 | Acceptance-rate analysis | 0 new outputs |

This sums to `17,555` primary outputs.

## Main-Track Expansion

The paper should not be locked on the core alone. The minimum result-backed
submission package is:

| Block | Outputs | Why It Exists |
|---|---:|---|
| Confirmatory core | 17,555 | Establishes the causal question with the already implemented design |
| Seed replication | 25,560 | Tests whether rare-event flips survive additional seeds |
| Dtype robustness | 12,780 | Distinguishes float16-specific effects from broader speculative-decoding behavior |

Total minimum: `55,895` primary outputs.

Optional stretch:

| Block | Outputs | Why It Exists |
|---|---:|---|
| Temperature robustness | 9,870 | Evaluates whether the story survives outside greedy decoding |

Stretch total: `65,765` primary outputs.

## Adjudication Contract

The `5,000`-label target is designed to support claim-bearing interpretation
rather than raw scale for its own sake. The locked ordering is:

1. all speculative safety flips
2. all rejection-sampling equivalence violations
3. disagreement clusters near the refusal boundary
4. matched stable controls by pair, phase, and task

Human review is reserved for the highest-value subset: at least `300`
adjudications covering disagreement clusters and paper-bound examples.

## Reporting Constraint

Inspection of the current TR144 implementation surfaced a schema mismatch
between `analyze.py` and `generate_report.py`. That does not block the
paperization pass, but it does block any assumption that the generated markdown
report is already publication-grade. The report layer must be tightened against
real run artifacts before result prose is promoted into the manuscript.

## Claim Ladder

### Claimable After The Core Only

- the experiment runs end to end
- phase-2 byte-identity auditing is operational
- phase-3 paired comparison is operational
- phase-4 dose-response analysis is operational
- per-request acceptance telemetry is operational

### Claimable After The Submission Floor

- rare-event interpretation can be stabilized across seeds
- arithmetic framing becomes less tied to float16 alone
- a main-track manuscript can rely on confirmatory plus robustness evidence

### Claimable Only After The Stretch Block

- any stronger statement about non-zero-temperature behavior
- broader deployment claims outside greedy decoding

## Paper Package Deliverables

TR144 now has a formal paper package:

- `papers/speculative_decoding_safety/README.md`
- `papers/speculative_decoding_safety/paper_manifest.json`
- `papers/speculative_decoding_safety/draft.md`
- `papers/speculative_decoding_safety/scripts/build_assets.py`
- `papers/speculative_decoding_safety/latex/`

The generated figures and tables in that package are planning artifacts built
from the locked publication contract. They are intentionally not presented as
empirical result figures.

## Risks And Review Objections

### 1. "This is still a small-model artifact."

Answer:
the paper line explicitly adds seed replication and dtype robustness before
results are promoted to the main-track floor. The claim is not "all
speculative decoding is unsafe"; it is "this widely deployed optimization needs
to be treated as part of the evaluated safety envelope."

### 2. "The observed effects may be too rare to matter."

Answer:
that is precisely why the evidence bar was raised from `17,555` to `55,895`
outputs. Rare-event claims need more rows, more adjudication, and stronger
null-preservation discipline.

### 3. "Without run artifacts, this is not yet a result paper."

Answer:
correct. This report is explicit that TR144 is currently protocol-grade. The
deliverable of this pass is a paper-ready evidence contract, not fabricated
results.

## Reproducibility

The current publication layer is reproducible without running models:

```bash
python papers/speculative_decoding_safety/scripts/build_assets.py
python papers/validate_papers.py
```

Empirical execution remains unchanged:

```bash
python research/tr144/run.py --phases 1,2,3,4,5
```

## Conclusion

TR144 is now structurally ready to become a paper. The study has:

- a frozen thesis
- a locked observation budget that clears a main-track bar
- a machine-readable publication contract
- a protocol-grade technical report
- a repo-native manuscript scaffold

What it still does not have is the one thing no paperization pass can invent:
real TR144 run artifacts. The next meaningful step is execution, not more
rethinking of the paper structure.
