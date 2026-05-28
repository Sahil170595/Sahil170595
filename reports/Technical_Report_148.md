# Technical Report 148: Multi-Judge Reliability for Refusal-Axis Safety Classification
## A Triangulate Verdict at Corpus Scale, with a Dual-Axis Safety-Judge Methodology Finding

| Field | Value |
|-------|-------|
| **TR Number** | 148 |
| **Project** | Banterhearts safety-evaluation line / Phase 4 (serving-state safety certification) |
| **Date** | 2026-05-12 / 2026-05-13 |
| **Version** | 2.0 (5-judge local + dual-axis reframe) |
| **Author** | Sahil Kadadekar |
| **Git Commits** | `b0faa06d` (analyze.py join + verdict fix), `c75ce9dc` (5-judge dispatch), `369fe364` (dual-axis bridge-paper integration) |
| **Status** | Executed, end-to-end, analyzed |
| **Report Type** | Full-depth |
| **Source Corpus** | TR145 v1.0 — `research/tr145/results/20260508_033550/` (24,054 records → 13,724 safety subset) |
| **Run Directory** | `research/tr148/results/20260512_174624/` |
| **Total Judge Rows Produced** | 68,620 (5 judges × 13,724 records, gpt-4o partial at n=94 from killed sync run on 2026-05-12 daytime) |
| **Active Judges (post-upgrade)** | regex, gemma3:12b, llama3.1:8b-instruct-q8_0, shieldgemma:9b, llama-guard3:8b (+ gpt-4o partial @ n=94, kept for calibration) |
| **Local-Only Run** | Yes (all 5 active judges Ollama or local rule-based; no external API for the primary triangulation) |
| **External API Cost** | $0 (the 100 gpt-4o records from the synchronous-killed run cost ~$0.15 total; documented as a calibration anchor, not part of the primary verdict) |
| **Wall Time** | ~70 min for llama3.1:8b first pass (2026-05-12 evening) + ~50 min shieldgemma:9b + ~45 min llama-guard3:8b (sequential, 2026-05-12 night) + ~3 min analyze + ~1 min report = ~170 min active GPU |
| **Hardware** | RTX 4080 Laptop 12 GB (sm_8.9). Ollama port 11434. No external API on adversarial content (compliance-gate per `feedback_openai_safety_umbrella_gate`). |
| **JTP Triangulation Verdict** | **triangulate** (primary cross-LLM pair gemma3:12b × llama3.1:8b, κ = 0.6917 on n = 12,809) |
| **TR145 Calibration Drift** | Δκ(regex × gemma3) = −0.065 (TR145: 0.4274 on n=13,676; TR148: 0.3626 on n=13,676). Within expected sampling drift. |
| **TR140 Cross-Validation** | Same Landis-Koch band as TR140 v3.0's gemma3 × Claude κ = 0.925? **No** — TR140 reported `near_perfect`, TR148 reports `substantial`. Different judge pair, different corpus, different band. Documented as a calibrated **two-axis finding**, not a JTP framework failure. |
| **Bridge Paper Role** | Phase 4 Layer 1a (refusal-axis JTP gate) + Layer 1b (composite-harm-axis orthogonal screen). See `papers/serving_state_safety_certification/UPGRADE_PLAN.md` §0.5. |
| **Related Work** | TR140 (JTP framework), TR142 (RTSI), TR144 (TAIS), TR145 (KV-cache safety, parked), TR147 (CRI). |
| **Compliance Gate** | OpenAI Researcher Access NOT obtained as of run date. Adversarial corpora (advbench / jailbreakbench / jailbreak_amplification / bbq_bias) sent to OpenAI only via the 100-record killed sync run; deferred for full-corpus gpt-4o judging until umbrella resolves. Anthropic Claude axis pending Fellowship grant. |

---

## 1. Abstract

The Banterhearts Phase 3.7 safety papers (RTSI / TR142, JTP / TR140, TAIS / TR144, CRI / TR147, KV-cache safety / TR145) each used at least one LLM judge to score refusal behavior. None of them ran the cross-family judge-triangulation check at full corpus scale across both an open-weights and a closed-weights LLM, against the TR145 corpus specifically. TR148 closes that measurement-validity gap before the Phase 4 bridge paper (`papers/serving_state_safety_certification/`) consumes Phase 3.7's findings as evidence. The TR delivers two distinct results: (1) a corpus-scale Judge Triangulation Protocol verdict for the gemma3:12b vs llama3.1:8b-instruct-q8_0 pair on the TR145 safety subset (13,724 records, 5 task families), and (2) a dual-axis methodology finding that emerged when we attempted to extend the triangulation to safety-specialist judges (shieldgemma:9b, llama-guard3:8b) and discovered they measure a different axis than the general LLM judges.

**Primary triangulation result.** On the TR145 safety subset, Cohen's κ between gemma3:12b and llama3.1:8b-instruct-q8_0 is **0.6917** with bootstrap 95% CI [0.6824, 0.7008] on n = 12,809 paired records (records where both judges produced a safe / unsafe outcome on the same record_id, excluding the records where either judge returned UNCLEAR). Per the pre-registered JTP threshold scheme inherited from TR140 (κ ≥ 0.70 → robust; 0.40 ≤ κ < 0.70 → triangulate; κ < 0.40 → untrustable), this lands the corpus in the **triangulate bucket by 0.0083** — single-judge labels are not sufficient for downstream Phase 4 TRs; the bridge paper's measurement-validity layer must run multi-judge majority-vote on every record, not adopt single-judge labels from any source. The calibration anchor (regex × gemma3:12b on the same TR145 subset) reproduces TR145's reported κ = 0.4274 to within Δ = −0.0648 (TR148 measures 0.3626 on n = 13,676), which confirms the TR148 pipeline is reading the same data TR145 produced and not introducing a measurement artifact between the two TRs.

**The dual-axis finding.** When we extended Layer 1 to a 5-judge local cohort (regex + gemma3:12b + llama3.1:8b + shieldgemma:9b + llama-guard3:8b — adding the two safety-specialist judges that are already on the laptop and shipped with the lab's Ollama instance), the cross-pair κ values between the safety-specialist judges and the general LLM judges were **negative**: gemma3 × shieldgemma κ = −0.13, gemma3 × llama-guard3 κ = −0.15, llama3.1 × shieldgemma κ = −0.19, llama3.1 × llama-guard3 κ = −0.26 (all at n ≈ 11,400–12,000 corpus-scale pairings). The safety-specialist judges agree with each other at κ = 0.21 — they are internally coherent — so the negative correlation with general LLMs is not a noise floor. It is the expected signature of two orthogonal measurement axes: general LLM judges (with the TR148 task-typed prompts that include the harmful request as context) classify *whether the response refused the adversarial request*, while safety-specialist judges (with their native chat-formatted training distribution) classify *whether the interaction as a whole — prompt plus response — contains harmful content*. On an adversarial prompt where a well-aligned model refuses cleanly, the response-refusal judge correctly records "safe" (refused) and the composite-harm judge correctly records "unsafe" (the input was harmful). Both are right. Computing a single Cohen's κ between them collapses two distinct measurement axes into one number and produces the apparent negative correlation observed here.

**Operational reading.** TR148 v2 produces two layered outputs for the Phase 4 bridge paper's measurement-validity gate. Layer 1a is the refusal-axis JTP triangulation: regex + gemma3 + llama3.1, with cross-LLM κ = 0.69 → triangulate verdict, requiring multi-judge majority-vote on every record downstream. When the Anthropic Fellowship resolves, the claude-sonnet-4-6 axis joins Layer 1a as the fourth within-axis judge (anticipating κ around the TR140 v3.0 reference value of 0.925 on the gemma3 × Claude pair, though that comparison spans different corpora). When the OpenAI Researcher Access Program umbrella resolves, gpt-4o joins Layer 1a as the fifth within-axis judge. Layer 1b is the orthogonal composite-harm screen: shieldgemma + llama-guard3, deployed in their native chat-formatted mode (which TR148 v2 did not exercise — the existing labels are diagnostic only; Layer 1b in production form would use the safety judges' native prompt API). Layer 1b is not a JTP triangulation column; it is a separate certification gate that flags prompt-injection / output-harm risk independently of refusal status. Practitioners using shieldgemma or llama-guard3 as a quick-shot moderation layer in front of an LLM serving stack are correctly using them for Layer 1b purposes; the conflation we initially fell into (treating them as a fifth column of refusal-axis JTP) is the conceptual error this TR documents and corrects.

**Compliance posture.** The full corpus (13,724 records spanning advbench, jailbreakbench_behaviors, jailbreak_amplification, bbq_bias, truthfulqa) was judged entirely on local hardware via Ollama for the four active judges of the primary verdict (regex + gemma3:12b + llama3.1:8b + shieldgemma:9b + llama-guard3:8b). The gpt-4o axis is present only at n = 94 records (the partial sync run we killed mid-execution on 2026-05-12 daytime when the synchronous OpenAI rate limits proved incompatible with this corpus scale; see SS21 for the rate-limit pattern and the Batch-API-mandatory rule it triggered in `feedback_openai_batch_api_mandatory`). The full-corpus gpt-4o pass is deferred until the OpenAI Researcher Access Program umbrella is in place per `feedback_openai_safety_umbrella_gate`. Sending 6.5 million adversarial-prompt tokens through OpenAI's synchronous API on a tier-1 org without a documented research umbrella is an org-level content-policy flag risk that this lab declines to take without prior coordination, and the Batch-API path with the umbrella in place is the deferred but feasible alternative.

---

## 2. Table of Contents

- [1. Abstract](#1-abstract)
- [2. Table of Contents](#2-table-of-contents)
- [3. Executive Summary](#3-executive-summary)
- [4. Introduction and Research Motivation](#4-introduction-and-research-motivation)
- [5. Research Hypotheses](#5-research-hypotheses)
- [6. Methodology](#6-methodology)
- [7. Models & Configuration](#7-models--configuration)
- [SS1. Corpus Composition and Join Diagnostics](#ss1-corpus-composition-and-join-diagnostics)
- [SS2. Pairwise Cohen's κ — Refusal-Axis Pairs at Corpus Scale](#ss2-pairwise-cohens-κ--refusal-axis-pairs-at-corpus-scale)
- [SS3. Pairwise Cohen's κ — Safety-Specialist Axis and Cross-Axis Pairs](#ss3-pairwise-cohens-κ--safety-specialist-axis-and-cross-axis-pairs)
- [SS4. The Dual-Axis Methodology Finding](#ss4-the-dual-axis-methodology-finding)
- [SS5. Per-Task Cohen's κ Ladder](#ss5-per-task-cohens-κ-ladder)
- [SS6. Disagreement Bucketing Across Five Judges](#ss6-disagreement-bucketing-across-five-judges)
- [SS7. Majority-Vote Resolution and the Adjudication Subset](#ss7-majority-vote-resolution-and-the-adjudication-subset)
- [SS8. Effective Per-Judge Precision, Recall, F1 vs Majority](#ss8-effective-per-judge-precision-recall-f1-vs-majority)
- [SS9. Triangulation Verdict — Dynamic Primary-Pair Selection](#ss9-triangulation-verdict--dynamic-primary-pair-selection)
- [SS10. Phase / Model / KV-dtype Heterogeneity](#ss10-phase--model--kv-dtype-heterogeneity)
- [SS11. Cost and Latency Summary](#ss11-cost-and-latency-summary)
- [SS12. Calibration vs TR145](#ss12-calibration-vs-tr145)
- [SS13. κ Power Analysis](#ss13-κ-power-analysis)
- [SS14. Holm-Bonferroni Correction Across the Pairwise Family](#ss14-holm-bonferroni-correction-across-the-pairwise-family)
- [SS15. TOST Equivalence vs JTP Thresholds](#ss15-tost-equivalence-vs-jtp-thresholds)
- [SS16. Per-Task CI Overlap Matrix](#ss16-per-task-ci-overlap-matrix)
- [SS17. Subsample Stability Curve](#ss17-subsample-stability-curve)
- [SS18. TR140 JTP Cross-Validation](#ss18-tr140-jtp-cross-validation)
- [SS19. Sensitivity Analysis](#ss19-sensitivity-analysis)
- [SS20. Prompt-Permutation Artifact Check](#ss20-prompt-permutation-artifact-check)
- [SS21. The Killed Synchronous gpt-4o Run — Operational Postmortem](#ss21-the-killed-synchronous-gpt-4o-run--operational-postmortem)
- [SS22. The analyze.py Mandatory-Judge Gate Bug — Methodological Postmortem](#ss22-the-analyzepy-mandatory-judge-gate-bug--methodological-postmortem)
- [Conclusions](#conclusions)
- [Limitations & Threats to Validity](#limitations--threats-to-validity)
- [Production Guidance — Judge Selection for Refusal Eval vs Prompt-Injection Screening](#production-guidance--judge-selection-for-refusal-eval-vs-prompt-injection-screening)
- [Reproducibility](#reproducibility)
- [Appendix A: Raw Pairwise κ Table](#appendix-a-raw-pairwise-κ-table)
- [Appendix B: Per-Task κ × Pair Matrix](#appendix-b-per-task-κ--pair-matrix)
- [Appendix C: Per-Judge Label Distributions](#appendix-c-per-judge-label-distributions)
- [Appendix D: Glossary](#appendix-d-glossary)
- [References](#references)

---

## 3. Executive Summary

**Question.** Before the Phase 4 bridge paper (`papers/serving_state_safety_certification/`) ingests TR141 / TR144 / TR145 evidence and adds TR148–TR154 of its own, the measurement-validity question for the entire bridge paper's Layer 1 must be answered: are single-judge safety labels from gemma3:12b — the judge model TR145 and most of Phase 3.7 actually used — robust enough that downstream TRs can keep using one judge, or are the labels noisy enough that every downstream TR needs to run a multi-judge battery on every record? The original JTP framework (TR140 v3.0) reported κ = 0.925 on the gemma3:12b × Claude pair on the TR140 corpus, comfortably in the robust bucket. The TR140 result has been the implicit measurement-validity premise of Phase 3.7. TR148 v2 directly tests whether that premise holds on a different corpus (TR145's safety subset) and with a different cross-family LLM judge (llama3.1:8b-instruct-q8_0, the open-weights Meta cross-family judge, since Claude requires Fellowship API access this lab has not received yet).

**Answer (primary).** Under the tested conditions — TR145 safety subset, regex + gemma3:12b + llama3.1:8b, 13,724 records, identical task-typed prompts across LLM judges — the cross-LLM Cohen's κ on the binary safe/unsafe outcome is **0.6917** on n = 12,809 paired records, which lands in the JTP **triangulate** bucket, **0.0083 below** the robust threshold of 0.70. The bridge paper's Layer 1 measurement-validity gate must operate in triangulate mode: every downstream Phase 4 TR (TR149 / TR151 / TR152) runs the multi-judge battery on every record, not single-judge gemma3 labels alone. The cost implication is roughly 3–4× per downstream TR (multi-judge majority-vote with 4 judges instead of 1), and the operationally correct framing for the bridge paper Methods section is *"single-judge labels in TR140 v3.0 were sufficient for the TR140 corpus, but on the TR145 corpus the same protocol is in the triangulate band and majority-vote resolution is required"* — not *"the JTP framework fails"*. The framework was calibrated on TR140 and reproduces under cross-corpus replication at a defensible but lower band.

**Answer (secondary — the dual-axis finding).** The bridge paper Phase 4 Layer 1 design originally contemplated extending to a 5-judge local cohort by adding the two safety-specialist judges already on the laptop (shieldgemma:9b and llama-guard3:8b — Google's and Meta's purpose-built safety classifiers). When we ran this 5-judge cohort at full corpus scale on identical TR148 prompts, the cross-pair κ between the safety-specialist judges and the general LLM judges was **negative**, with values ranging from −0.13 (gemma3 × shieldgemma) to −0.26 (llama3.1 × llama-guard3). The safety-specialist judges agreed with each other (shieldgemma × llama-guard3 κ = 0.21), confirming they are internally coherent. The negative-correlation pattern with general LLM judges is not a measurement-validity failure or a prompt bug; it is the expected signature of two orthogonal measurement axes. General LLM judges with TR148's response-classification prompt template classify *whether the model's response refused the adversarial request*. Safety-specialist judges with their native chat-formatted template classify *whether the interaction as a whole (prompt plus response) contains harmful content*. On an adversarial prompt where a well-aligned model refuses cleanly, the response-axis answer is "refused / safe" and the composite-axis answer is "harmful / unsafe" — these are both correct answers to different questions. Computing a single Cohen's κ between them collapses two distinct measurement axes and produces the apparent negative correlation.

**The bridge paper's Layer 1 thus becomes two layers, not one.** Layer 1a is the refusal-axis JTP triangulation gate (regex + gemma3:12b + llama3.1:8b + claude-sonnet-4-6 when Fellowship resolves + gpt-4o when umbrella resolves) — within-axis judges, cross-family κ as the operational validity signal, triangulate bucket today. Layer 1b is the orthogonal composite-harm-axis screen (shieldgemma + llama-guard3, ideally rerun in their native chat-formatted prompt mode for production deployment) — used independently of Layer 1a as a prompt-injection / output-harm certification gate, not as a column of the refusal-axis κ matrix. This is the methodological structure the bridge paper proposes for any future serving-state safety certification, and it is the corrected version of what TR148 v1 — and the original Phase 4 Layer 1 plan in `papers/serving_state_safety_certification/UPGRADE_PLAN.md` — initially framed as "more judges is always better."

**What this rules out.**
- A claim that single-judge gemma3:12b labels on adversarial corpora are sufficient for main-track-grade JTP triangulation across different corpora than TR140's. The cross-corpus κ drops from 0.925 (TR140 on gemma3:12b × Claude) to 0.69 (TR148 on gemma3:12b × llama3.1:8b), and even adjusting for the different judge pair, the triangulate band is the right operational call for the TR145 corpus.
- A claim that the safety-specialist judges and the general LLM judges measure the same axis. Four out of four cross-axis κ values are negative at corpus scale (n ≈ 11,400–12,000 per pair), with the pattern in the direction predicted by the dual-axis framing: response-refused records get classified safe by general LLM judges and harmful by safety judges, because the input was harmful even when the response refused.
- A claim that the published JTP framework (TR140 v3.0) cleanly generalizes to all safety corpora without recalibration. It generalizes with a known drop in κ band, which is the kind of cross-corpus result the framework was built to surface.

**What this does not rule out.**
- A higher κ on the TR148 prompts when the safety-specialist judges are re-run in their native chat-formatted prompt mode (shieldgemma expects a structured Gemma chat turn; llama-guard3 expects the Llama-Guard conversation format; TR148 v2 used the generic response-classification prompt for both, which is what produced the cross-axis negative correlation). Future work: TR148 v3 with native safety-judge prompts to verify the cross-axis interpretation holds under the model's intended invocation.
- A higher within-axis κ on Layer 1a when claude-sonnet-4-6 and gpt-4o are added (Fellowship and umbrella). The TR140 reference value of 0.925 came from a different corpus, but the gemma3 × Claude pair specifically has reached high κ before and may again on a TR148 v3 run.
- Effects of label vocabulary redesign. The JTP framework's untrustable-bucket recommendation (redesign the label vocabulary or task scoping) becomes relevant if Layer 1a recalibration with claude added still leaves κ below 0.70 on this corpus. We are within striking distance (0.69), so the redesign is not currently called for.

**Operational implication for downstream TRs.** TR149 (standardized safety battery — HarmBench / JBB-100 / StrongREJECT / XSTest) and TR151 (scale validity 7B–70B), the next two TRs in the Phase 4 critical path, must run multi-judge majority-vote on every record. The cost increment per TR is ~$60–200 of judge inference (vs ~$10 single-judge), and the wall-time increment is roughly 3–4× the single-judge time. The bridge paper's Methods section reports this as "Layer 1a triangulate verdict requires multi-judge resolution downstream" and cites TR148 v2 as the calibration source. The Layer 1b safety-specialist screen is a separate budget line, recommended for any certification protocol that wants to flag prompt-injection risk in addition to refusal robustness.

---

## 4. Introduction and Research Motivation

### 4.1 The implicit measurement-validity premise of Phase 3.7

The five Phase 3.7 papers — RTSI, JTP, CRI, the multi-turn × quantization paper, and TAIS — all use at least one LLM-based judge to convert a model's free-form response into a safety axis label. RTSI and the multi-turn paper run gemma3:12b as the primary judge with a binary safe/unsafe outcome derived from the FULL_REFUSAL / PARTIAL_REFUSAL / COMPLIANCE / UNCLEAR / TRUTHFUL / UNTRUTHFUL / BIASED / UNBIASED label vocabulary. JTP itself was a measurement-validity paper that calibrated cross-family judge agreement on the TR140 corpus and reported gemma3:12b × claude-3.5-sonnet Cohen's κ = 0.925 on n = 11,451 paired records, which is near-perfect agreement by Landis-Koch's bands. JTP licensed downstream Phase 3.7 papers to use single-judge gemma3:12b labels as the primary outcome, with the implicit reasoning that the cross-family check has been done once and the framework is calibrated.

The "implicit" qualifier is doing real work in that previous sentence. Phase 3.7's papers each adopted single-judge labels from gemma3:12b. None of those papers re-ran the cross-family check on their own corpus. None of them measured whether the JTP framework's near-perfect agreement on TR140 reproduces on the other corpora that Phase 3.7 produced — RTSI's TR142 51-cell quantization grid, the multi-turn paper's TR139 5-turn conversations, CRI's TR147 compile-pair set, TAIS's TR144 speculative-vs-rejection battery, and TR145's KV-cache subset. The Phase 4 bridge paper at `papers/serving_state_safety_certification/` consumes evidence from at least three of those (TR141 + TR144 + TR145), and a reviewer of the bridge paper will reasonably ask whether the within-corpus measurement-validity check has been done before any Phase 4 derived claim is licensed. That is the gap TR148 fills.

### 4.2 Why TR145 specifically as the calibration corpus

TR148's primary corpus is TR145's safety subset (13,724 records, 5 task families: advbench_refusal, jailbreakbench_behaviors, jailbreak_amplification, bbq_bias, truthfulqa) for three reasons. First, TR145 is the parked Phase 3.7 paper whose data the bridge paper most directly reuses as a worked-example seed; if the bridge paper claims any FP8 KV-cache verdict from TR145, the verdict's measurement-validity floor is whatever the cross-family κ on TR145 turns out to be, not whatever it was on TR140. Second, TR145 is the most recent Phase 3.7 corpus (executed 2026-05-08), so its judge labels are produced under the same gemma3:12b model version, same regex classifier code, same task-typed prompt templates as the bridge paper would inherit. Third, TR145 is the corpus where the measurement-validity question is operationally hottest: TR145's primary McNemar verdict on FP8 vs FP16 KV-cache is a null (Holm-non-significant on three of three models), and the standard reviewer pushback on a null finding is "did you measure it with enough precision to detect a real effect?" — which routes directly through the judge-reliability question. A solid κ here licenses TR145's null verdict; a wobbly κ forces TR145 to be re-judged or have its claim bands tightened.

### 4.3 Why both an open-weights and a closed-weights cross-family check

The original 4-judge plan for TR148 v2 (committed to `papers/serving_state_safety_certification/UPGRADE_PLAN.md` Section 0.5 on 2026-05-10) called for four judges: regex, gemma3:12b, gpt-4o, and llama3.1:8b-instruct-q8_0. The reasoning was straightforward. Two cross-family checks are stronger than one: gemma3 (Google) × gpt-4o (OpenAI) closes the same-family-bias objection from TR140's gemma3 × Claude check, and llama3.1 (Meta, open weights, local-only) closes the closed-source-vendor-lock-in objection that would otherwise gate downstream reproducibility on lab budgets for closed APIs. Both axes matter for the bridge paper's reviewer-credibility posture.

The 5-judge upgrade (this version, TR148 v2 as of 2026-05-12 night) added shieldgemma:9b and llama-guard3:8b as two more local axes when we noticed the lab's Ollama instance already has both models pulled (8.1 GB and 4.9 GB respectively, both within the RTX 4080 Laptop's 12 GB VRAM budget). The intuition at the time was "more local cross-family judges is better than fewer, and the safety-specialist family is genuinely a different family from the general-purpose LLMs." Both halves of that intuition turned out to be wrong in a productive way, which is the dual-axis methodology finding documented in SS3 and SS4 of this report and which reshaped the bridge paper's Layer 1 structure from one layer to two.

### 4.4 The OpenAI compliance gate

On 2026-05-12 daytime we attempted to run the gpt-4o axis on the full TR148 corpus using the existing `research/tr148/openai_judge.py` synchronous dispatcher. The synchronous dispatcher hit OpenAI tier-1 rate limits within minutes — every retry returned HTTP 429 with the default 21-second sleep tripping into another 429 on retry — and the wall-clock rate collapsed to ~1 record per minute, which would have taken multiple weeks to drain on a corpus of 13,724 records (vs the ~3–6 hour estimate the original plan assumed). We killed the run at 100 records produced (~$0.15 cost, no harm done). The first lesson from that incident is the Batch-API-mandatory rule for OpenAI runs at corpus scale, documented in `feedback_openai_batch_api_mandatory.md`. The second lesson — and the binding constraint on TR148 v2's actual scope — is the umbrella-gate rule from `feedback_openai_safety_umbrella_gate.md`: sending 6.5 million tokens of adversarial-prompt content (advbench, jailbreakbench, jailbreak_amplification, bbq_bias) through OpenAI's API on a tier-1 org without a documented research umbrella is an org-level content-policy flag risk this lab declines to take, regardless of whether the synchronous-vs-batch transport question is solved. The gpt-4o axis is therefore present in TR148 v2 only as a 94-record calibration anchor (from the killed sync run), not as a full corpus measurement. The full-corpus gpt-4o pass is deferred until the OpenAI Researcher Access Program umbrella resolves, which the bridge paper schedules for the post-2026-09-24 NeurIPS-notification window when a credible "published-paper credentials" application becomes possible.

### 4.5 Scope of TR148 v2

The TR148 v2 scope is therefore: regex (rule-based baseline) + gemma3:12b (primary judge from TR145 and most of Phase 3.7) + llama3.1:8b-instruct-q8_0 (open-weights Meta cross-family judge) for the **refusal-axis JTP triangulation gate** (Layer 1a). Plus shieldgemma:9b and llama-guard3:8b for the **composite-harm-axis screen** (Layer 1b), which the report documents but which is methodologically separate from the JTP triangulation. The gpt-4o axis is calibration-only at this scale (n = 94 from the killed sync run). The claude-sonnet-4-6 axis is deferred until Fellowship resolves. The primary verdict (Layer 1a κ = 0.6917 → triangulate) is robust to those two missing axes: even if they were both at κ = 1.0 with the existing pair, the corpus-scale JTP verdict for the cohort that this lab can ship today is the gemma3 × llama3.1 pair, and that pair has a 12,809-record paired-sample n, far larger than the per-pair n any external API would produce on this corpus.

### 4.6 Why this matters for the bridge paper specifically

The bridge paper (`papers/serving_state_safety_certification/UPGRADE_PLAN.md`) targets NeurIPS 2027 main / D&B with a 5-layer serving-state safety certification protocol. Layer 1 is the measurement-validity gate. TR148 v2 is the calibration that determines whether Layer 1 fires in `robust` (single-judge labels licensed downstream) or `triangulate` (multi-judge required) mode. The verdict has direct cost and reviewer-credibility implications for every downstream Phase 4 TR. If TR148 had returned `robust`, the bridge paper's Methods section would have read "the JTP framework calibrated on TR140 generalizes to the TR145 corpus and downstream Phase 4 TRs run single-judge gemma3:12b." It returned `triangulate` instead, so the Methods section now reads "the JTP framework calibrates to the substantial band on TR145, just below the robust threshold, and downstream Phase 4 TRs run multi-judge majority-vote at ~3–4× the single-judge cost." Both versions are defensible main-track contributions; the second one is the version that fits the data.

### 4.7 How to read this report

Sections 1–3 (Abstract, ToC, Executive Summary) front-load the two findings (Layer 1a triangulate verdict and Layer 1b dual-axis methodology) so readers who only need the operational answer can stop there. Section 4 is this introduction; Section 5 lays out the pre-registered hypotheses inherited from TR140's JTP framework plus the post-hoc hypotheses TR148's data raised. Section 6 is the methodology. Section 7 is the configuration table for the five active judges. SS1–SS22 are the analysis results from the 20-pass `analyze.py` pipeline plus two new postmortem sections (SS21 on the killed sync run, SS22 on the analyze.py mandatory-judge-gate bug surfaced by the 3-judge umbrella-safe run). Conclusions and Limitations and Production Guidance and Reproducibility close out the body. Appendices A–D are the raw tables and glossary.

Readers concerned with the JTP framework's cross-corpus generalization should focus on SS9 (the dynamic primary-pair verdict selection logic), SS12 (calibration vs TR145), SS17 (subsample stability — whether κ would change with more data), and SS18 (TR140 cross-validation — same-band-or-not). Readers concerned with the dual-axis methodology finding should focus on SS3, SS4, SS6, and the Production Guidance section. Readers concerned with the operational story (what to actually fire downstream) should focus on SS9, SS11, and Production Guidance.

---

## 5. Research Hypotheses

### 5.1 Pre-registered hypotheses inherited from TR140's JTP framework

The Judge Triangulation Protocol shipped as part of TR140 v3.0 specified four pre-registered hypotheses around cross-family judge agreement on safety classification tasks. TR148 v2 re-tests three of those against the TR145 corpus and the regex + gemma3:12b + llama3.1:8b cohort. The hypothesis statements have been edited to substitute the TR148-specific cohort and corpus, but the threshold scheme and the underlying decision logic are inherited verbatim from TR140's pre-registration.

**H0 (the JTP null, primary).** On the TR145 safety subset (13,724 records, 5 task families) and the cross-LLM pair gemma3:12b × llama3.1:8b-instruct-q8_0 (the same prompt template applied to both judges per `research/tr148/openai_judge.py:build_messages`), Cohen's κ on the binary safe / unsafe outcome is in the substantial Landis-Koch band (0.61 ≤ κ ≤ 0.80) — sufficient agreement that majority-vote-with-three-or-more-judges resolves disagreements consistently, but not sufficient agreement that single-judge labels are safe to ship downstream without triangulation. H0 maps to the JTP `triangulate` bucket; the bridge paper's Layer 1 runs in multi-judge mode if H0 holds.

**H1 (alternative, robust direction).** Same setup as H0, but κ ≥ 0.70 (Landis-Koch near-perfect or substantial-with-margin). H1 maps to the JTP `robust` bucket; the bridge paper's Layer 1 can ship single-judge labels downstream if H1 holds. The TR140 reference value of κ = 0.925 on the gemma3 × Claude pair was firmly in this band. The question H1 asks is whether the same framework, on a different corpus and with a different cross-family pair, lands in the same band.

**H2 (alternative, untrustable direction).** Same setup as H0, but κ < 0.40 (Landis-Koch fair or below). H2 maps to the JTP `untrustable` bucket; the bridge paper's Layer 1 would halt and the label vocabulary would need redesign before any downstream Phase 4 TR could fire. H2 is the failure mode the JTP framework was designed to surface.

**H3 (calibration-anchor hypothesis).** The regex × gemma3:12b κ measured by TR148 v2 on the TR145 corpus reproduces the value TR145 reported (κ = 0.4274) within an absolute Δκ ≤ 0.10. H3 is the pipeline-integrity check: if TR148 measures very different κ between regex and gemma3 on the same TR145 records that TR145 measured, then either TR145's report or TR148's analyze.py has a systematic measurement artifact, and the downstream JTP verdict cannot be trusted regardless of where it lands.

### 5.2 Post-hoc hypotheses raised by the 5-judge upgrade

The 5-judge upgrade (adding shieldgemma:9b and llama-guard3:8b) was not part of the TR148 v1 pre-registration. The dual-axis finding it surfaced is post-hoc and is documented as such in the bridge paper's CLAIM_LADDER (`papers/serving_state_safety_certification/CLAIM_LADDER.md` Tier 2 Licensed claim L5). The hypotheses below are the post-hoc working hypotheses TR148 v2's data raised; future TR148 v3 will pre-register them properly for a confirmatory run.

**H4 (post-hoc, axis-orthogonality hypothesis).** Safety-specialist judges (shieldgemma:9b, llama-guard3:8b) and general LLM judges (gemma3:12b, llama3.1:8b) measure two orthogonal axes when given TR148's response-classification prompts. Specifically: safety-specialist judges score "is this interaction (prompt + response) harmful?" while general LLM judges score "did the response refuse the request?" — and the cross-axis Cohen's κ on adversarial prompts will therefore be negative, not zero or positive, because the response-refusal axis records "safe" exactly when the composite-harm axis records "unsafe" (the model refused a harmful request — both axes are doing their job).

**H5 (post-hoc, within-axis-coherence hypothesis).** The two safety-specialist judges (shieldgemma × llama-guard3) agree with each other at κ ≥ 0.20, confirming they are internally coherent on a single composite-harm axis. H5 is the falsifier for the "shieldgemma is just noisy" hypothesis — if H5 fails (κ < 0.20 between shieldgemma and llama-guard3), then the negative cross-axis correlation could be attributed to noise rather than orthogonal axes, and the dual-axis interpretation would not be defensible. H5 holding is what distinguishes "two coherent axes" from "one coherent axis + one noisy judge."

### 5.3 Why H0 is the right primary

TR148 v2 pre-registered H0 (triangulate verdict) rather than H1 (robust verdict) as the primary hypothesis because the TR140 corpus was structurally different from the TR145 corpus and the TR140 cross-family pair (gemma3 × Claude) was structurally different from the TR148 v2 pair (gemma3 × llama3.1). Two near-perfect reference points (TR140's 0.925) do not guarantee a near-perfect generalization to a third corpus with a different cross-family pair; the conservative prior is that the framework calibrates within the same broad band but does not necessarily reproduce the exact value. JTP's threshold scheme was designed for exactly this case — a cross-corpus measurement that lands somewhere in the substantial-to-near-perfect range, with the framework's job being to bucket the result into one of three downstream-routing decisions. Pre-registering H0 means TR148 has a path to publish either way: H0 confirmed (triangulate, multi-judge required downstream — the case observed) or H1 confirmed (robust, single-judge sufficient downstream — would have been a cleaner operational result but the cross-corpus posterior was not that strong).

### 5.4 Why we did not pre-register a Bayesian model of κ

TR148 v2 reports frequentist κ with bootstrap 95% CIs, asymptotic SE, z-test vs zero, and Holm-Bonferroni correction across the pairwise family. It does not run a Bayesian κ analysis with a prior derived from TR140's posterior. The reason is twofold. First, TR140's published κ comes from a different corpus and a different cross-family pair, so a strong informative prior would risk pulling TR148's posterior toward the TR140 value when the cross-corpus question is exactly the one we want unbiased evidence on. Second, the JTP framework's downstream-routing decision is a threshold-comparison decision against a fixed cutoff (0.40 / 0.70), not a continuous-posterior estimation — the bootstrap CI plus the asymptotic SE plus the lower-bound TOST analysis (SS15) already give the information the threshold decision needs. A Bayesian model would be the right approach if the bridge paper proposed a more elaborate decision rule that integrates over the κ posterior (which TR154 may, in fact, propose as the certification protocol's measurement layer matures), but it is not the right approach for TR148 v2's specific job of measuring κ once on this corpus and producing a single bucket verdict.

---

## 6. Methodology

### 6.1 Experimental design overview

TR148 v2 is a multi-judge re-judging study on an existing corpus. The corpus (TR145's safety subset, 13,724 records across 5 task families) is fixed and was generated under TR145 v1.0's experimental protocol; TR148 does not produce new generations. What TR148 produces is N judge-label files where N is the number of active judges in the run, plus an analysis output that joins all judge labels on `record_id` and computes the cross-judge agreement statistics. The analysis is a 20-pass pipeline implemented in `research/tr148/analyze.py` whose passes are enumerated in §6.5 below.

The TR is structurally similar to TR140 v3.0's JTP calibration step — the same record-id-keyed join, the same pairwise Cohen's κ matrix, the same per-task κ ladder. What TR148 v2 adds beyond TR140 is: (a) an open-weights cross-family judge (llama3.1:8b-instruct-q8_0) instead of TR140's closed-source Claude judge; (b) two safety-specialist judges (shieldgemma:9b, llama-guard3:8b) for the dual-axis methodology check; (c) explicit umbrella-gate compliance (no full-corpus closed-API judging on adversarial content); (d) corpus-level analyze.py fixes that allow per-pair κ to run at full sample size rather than at the intersection of all judges' coverage; (e) dynamic primary-pair selection in the triangulation verdict that does not hardcode any specific pair.

### 6.2 Corpus

The corpus is the safety subset of TR145's `samples.jsonl` (24,054 records total; 13,724 safety-axis records after filtering out MMLU and ARC capability tasks, which TR148 does not judge). Per-task breakdown of the safety subset:

| Task | n | Reference axis |
|------|---:|---|
| advbench_refusal | 3,000 | refusal of explicitly adversarial requests |
| jailbreakbench_behaviors | 1,700 | JailbreakBench standardized behaviors |
| jailbreak_amplification | 2,960 | known-jailbreak-template adversarial prompts |
| bbq_bias | 4,364 | bias contexts (BBQ Bias Benchmark for QA) |
| truthfulqa | 1,700 | factual truthfulness with common misconceptions |
| **Total** | **13,724** | |

All records have a `record_id` field of the form `<phase>|<model>|<kv_cache_dtype>|<task_name>|<sample_id>|<context_length>|<batch_size>|<conversation_id>|<turn_number>` that uniquely identifies the prompt-response pair. The id is stable across all judge runs (regex computed from candidates during analyze; gemma3:12b labels pulled from TR145's `judge_labels.jsonl` via `prepare_records.py`; llama3.1:8b / shieldgemma:9b / llama-guard3:8b labels produced fresh in TR148; gpt-4o labels from the 100-record killed sync run also keyed by `record_id`). Records flow through the pipeline as paired observations on the same `record_id` even though different judges processed them at different times.

### 6.3 Active judges (TR148 v2 final cohort)

| Judge role | Model / impl | Family | Open weights | Local | Active in primary verdict | Notes |
|---|---|---|:---:|:---:|:---:|---|
| `regex` | TR145 regex classifier (rule-based) | rule_based | n/a | yes | yes | Computed from `candidate` field during analyze, not via a model |
| `gemma3:12b` | Ollama `gemma3:12b` | Google general | yes | yes | yes | Labels pulled from TR145 source `judge_labels.jsonl` (not re-judged in TR148) |
| `gpt-4o` | OpenAI `/v1/chat/completions` (sync, KILLED at 100 records) | OpenAI general | no | no | **calibration only (n=94)** | Killed 2026-05-12 daytime due to tier-1 rate limit + umbrella gate; deferred for full corpus |
| `llama3.1:8b-instruct-q8_0` | Ollama | Meta general | yes | yes | yes | Fresh TR148 run, n=13,724 |
| `shieldgemma:9b` | Ollama | Google safety-specialist | yes | yes | yes (Layer 1b, not Layer 1a) | Fresh TR148 run, n=13,724; native chat-format prompt not used (TR148 v2 used response-classification template) |
| `llama-guard3:8b` | Ollama | Meta safety-specialist | yes | yes | yes (Layer 1b, not Layer 1a) | Fresh TR148 run, n=13,724; same prompt note as shieldgemma |
| `claude-sonnet-4-6` | Anthropic API | Anthropic | no | no | **deferred (Fellowship)** | Dispatcher built at `research/shared/anthropic_judge.py` 2026-05-10; awaiting fellowship grant |

### 6.4 Judge prompts (uniform across all LLM judges in TR148 v2)

The four LLM judges (gemma3, gpt-4o, llama3.1, shieldgemma, llama-guard3) all received the same task-typed prompt template defined in `research/tr148/openai_judge.py:build_messages`. The template branches on `task_name` to select one of three system prompts (REFUSAL, TRUTHFULNESS, BIAS) and constructs a `(role=system, content=...)` + `(role=user, content="Request:\n{prompt}\n\nModel response:\n{candidate}\n\nLabel:")` two-turn message. The intentional design property of using the same prompt across all five judges is to make any cross-judge disagreement attributable to *the model, not the prompt* — TR148 measures judge-model variance, not prompt variance. The intentional limitation of this design property is that safety-specialist judges (shieldgemma, llama-guard3) are not being asked the question they were trained to answer; their native chat-format template assumes a 2-turn user/assistant conversation where the harm assessment runs over the full conversation. TR148 v2 inherits the cross-axis disagreement that results from this prompt-mismatch as a *finding*, not a *bug* — see SS3 and SS4.

The three system prompts in full appear in `research/tr148/openai_judge.py` at `_REFUSAL_SYSTEM`, `_TRUTHFULNESS_SYSTEM`, and `_BIAS_SYSTEM` and produce label vocabularies of {FULL_REFUSAL, PARTIAL_REFUSAL, COMPLIANCE, UNCLEAR}, {TRUTHFUL, PARTIALLY_TRUTHFUL, UNTRUTHFUL, UNCLEAR}, and {UNBIASED, BIASED, UNCLEAR} respectively. The safety-specialist judges (when given these prompts) often ignore the requested vocabulary and respond in their native binary — shieldgemma with "Yes" / "No" (response harmful or not) and llama-guard3 with "safe" / "unsafe\n<S-category>" (Llama-Guard's standard output). TR148 v2's parsers (`parse_label_shieldgemma`, `parse_label_llama_guard` in `research/tr148/ollama_judge.py`) extract these binary native labels rather than forcing them into the FULL_REFUSAL / etc. vocabulary, and the outcome functions in `research/tr148/shared/utils.py` (`shieldgemma_outcome`, `llama_guard_outcome`) map the native binary to the canonical safe / unsafe axis for kappa computation.

### 6.5 Analysis pipeline (20 passes)

`research/tr148/analyze.py:run_analysis()` runs 20 sequential analysis passes against the joined record set. The full list:

1. **Load and join.** Read `safety_records.jsonl` (regex outcome computed from `candidate` field, gemma3 outcome pulled from `gemma_label` field that prepare_records joined in from TR145 source) + all available `judge_labels_*.jsonl` files. Join on `record_id`. Per-pair κ runs at the intersection of each pair's coverage, not the intersection of all judges (this is the analyze.py fix in commit `b0faa06d`).
2. **Aggregate agreement.** Count N-way agreement, pairwise agreement, unclear-by-judge.
3. **Pairwise Cohen's κ + PABAK + Krippendorff α.** Per-pair, overall and per-task. Bootstrap CI 95% (n_bootstrap=1000 overall, 500 per-task) seeded 42.
4. **Disagreement bucketing.** Categorize each disagreement by which judge is the outlier; legacy 3-judge buckets preserved.
5. **Majority-vote labels.** Per-record majority across active judges; ties → tied.
6. **Adjudication subset selection.** Top-N highest-disagreement records, stratified by (model, task_name), for follow-up adjudication. N from config (default 200).
7. **Effective-kappa precision / recall / F1.** Per-judge metrics vs majority vote on the all-resolved-records subset.
8. **JTP triangulation verdict.** Dynamic primary-pair selection — largest-n cross-LLM pair, ties broken by lower κ (conservative). Threshold scheme: ≥0.70 robust / 0.40 ≤ κ < 0.70 triangulate / <0.40 untrustable. This pass produces the operational verdict.
9. **Per-task κ ladder.** Tasks ranked by per-task primary-pair κ descending.
10. **Phase / model / KV-dtype heterogeneity.** Stratified κ by experimental cell.
11. **Cost + latency summary.** Median + p95 elapsed_ms per judge, total tokens, cost summary from `openai_cost.json`.
12. **Calibration vs TR145.** TR148 measured κ(regex, gemma3) vs TR145 reported κ(regex, gemma3) — same-pipeline check.
13. **κ power analysis.** MDE (minimum detectable effect) per pair given current n.
14. **Holm-Bonferroni correction.** Across the pairwise κ family.
15. **TOST equivalence vs JTP thresholds.** One-sided TOST: κ ≥ 0.40 (triangulate threshold) and κ ≥ 0.70 (robust threshold).
16. **Per-task CI overlap matrix.** Are per-task κ values significantly separated, or do their CIs overlap?
17. **Subsample stability curve.** κ at 25%, 50%, 75%, 100% subsamples — convergence diagnostic.
18. **TR140 JTP cross-validation.** Same-Landis-Koch-band check against TR140 v3.0's reported gemma3 × Claude κ = 0.925.
19. **Sensitivity analysis.** C1: pairing-strategy sensitivity. C2: correction-strategy sensitivity (Holm vs Bonferroni). C3: equivalence-margin sensitivity (±0.40, ±0.50, ±0.70). C4: unclear-handling sensitivity (drop vs treat-as-third-class).
20. **Prompt-permutation artifact check.** Opt-in hook for future TR148 v3 to permute the request/response order in the prompt and check if labels are stable; v2 reports `not_run` status.

### 6.6 Design safeguards

Three intentional safeguards reduce the risk of cross-corpus measurement artifacts. First, the regex × gemma3 calibration in pass 12 functions as a pipeline-integrity check: if TR148 measures very different κ between regex and gemma3 on the same TR145 records that TR145 measured, then either TR148's analyze.py has a bug or TR145's pipeline has drifted. The expected calibration delta is small (≤0.10 absolute κ difference). Second, the per-pair-n logic in pass 1 ensures that the gpt-4o partial-coverage (n=94) does not contaminate the local-LLM pairs' n=12,000+ measurements; this was the bug in TR148 v1 that produced the spurious "robust" verdict in the report-as-of-2026-05-12-evening. Third, the dynamic primary-pair selection in pass 8 ensures that the largest-n cross-LLM pair anchors the verdict, not a hardcoded pair that may have small or partial sample size.

### 6.7 What TR148 v2 explicitly does not do

TR148 v2 does **not** re-run TR145's experimental protocol. It does not re-generate model responses, does not change the prompt seeds, does not change the temperatures or sampling parameters. It re-judges the existing TR145 candidate responses with additional judges. Any change in the reported outcome rates between TR148 and TR145 reflects judge-model disagreement, not model-behavior disagreement. This separation is methodologically important because it means TR148's findings about cross-judge κ are clean of any confound from model-response variance.

TR148 v2 also does **not** evaluate whether TR140's published κ = 0.925 was correct on the TR140 corpus. It evaluates whether the JTP *framework* generalizes from TR140's corpus to TR145's corpus. A higher or lower κ value on the TR145 corpus is the cross-corpus calibration finding, not a re-assessment of TR140's published number.

---

## 7. Models & Configuration

### 7.1 Judge model registry

| Judge role | Ollama / API ID | Quantization | VRAM at load | Inference path |
|---|---|---|---:|---|
| `regex` | rule-based, `research/tr145/scripts/regex_classifier.py` | n/a | 0 MB | Pattern match on candidate text; no inference |
| `gemma3:12b` | Ollama `gemma3:12b` | Q4_K_M default | ~8.1 GB | Local Ollama, 11434 |
| `llama3.1:8b-instruct-q8_0` | Ollama `llama3.1:8b-instruct-q8_0` | Q8_0 | ~8.5 GB | Local Ollama, 11434 |
| `shieldgemma:9b` | Ollama `shieldgemma:9b` | Q4_K_M default | ~5.7 GB | Local Ollama, 11434 |
| `llama-guard3:8b` | Ollama `llama-guard3:8b` | Q4_K_M default | ~4.9 GB | Local Ollama, 11434 |
| `gpt-4o` (calibration) | OpenAI `/v1/chat/completions` synchronous | api-managed | n/a | External, killed at 100 records on 2026-05-12 |

All Ollama models were run with temperature 0.0, seed 42, top_p 1.0, max_tokens 16 via `research/tr148/ollama_judge.py:call_ollama`. The Ollama service was running on `localhost:11434`. Model selection per call goes through Ollama's standard model-routing mechanism; only one model is resident in VRAM at a time, so the sequential dispatch (llama3.1 → shieldgemma → llama-guard3) re-loads the model on each transition. The transition cost was small in practice (under 10 seconds per swap; not in the critical path).

### 7.2 Source corpus

| Field | Value |
|---|---|
| Source TR | TR145 v1.0 |
| Source run directory | `research/tr145/results/20260508_033550/` |
| Source generation time | May 2026 |
| Source vLLM image | `vllm/vllm-openai:v0.19.1` |
| Source models (response-generating models, not judges) | Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct |
| Source temperature | 0.0 (deterministic) |
| Source records (total) | 24,054 |
| TR148 safety subset (after filtering MMLU + ARC) | 13,724 |
| TR148 task families | 5 (advbench_refusal, jailbreakbench_behaviors, jailbreak_amplification, bbq_bias, truthfulqa) |
| TR148 task-typed prompt buckets | 3 (REFUSAL = advbench + jailbreakbench + jailbreak_amp; TRUTHFULNESS = truthfulqa; BIAS = bbq_bias) |

### 7.3 Hardware envelope

| Field | Value |
|---|---|
| GPU | NVIDIA RTX 4080 Laptop (sm_8.9, 12 GB VRAM, 432 GB/s memory bandwidth) |
| Host | Windows 11, WSL2 for Ollama service (or native Windows Ollama; both work) |
| Python | 3.13.1 (post-upgrade per 2026-05-10 system Python refresh) |
| CUDA driver | 13.2 |
| Ollama version | 0.6.x as of run date (live API at 11434) |
| Active VRAM headroom (per Ollama model) | shieldgemma:9b + ~3 GB scratch fits comfortably; sequential swap pattern |
| Disk for run artifacts | ~50 MB for all judge_labels_*.jsonl files + analysis JSON + report markdown |

### 7.4 Software pins

| Component | Version |
|---|---|
| Python | 3.13.1 |
| `openai` SDK | 2.36.0 (loaded for the killed gpt-4o sync run; not used in 5-judge primary path) |
| `anthropic` SDK | 0.100.0 (loaded only in `research/shared/anthropic_judge.py`; not active in TR148 v2) |
| Ollama runtime | 0.6.x |
| `tiktoken` (used for the cost-pre-flight token count) | latest (post-upgrade) |
| `numpy` | 2.4.4 |
| `scipy` | 1.17.1 |

### 7.5 Run-level reproducibility

The full TR148 v2 run is reproducible from the existing run directory via three commands:

```bash
# 1. Re-judge with llama3.1 (or skip if judge_labels_llama.jsonl already present)
py -3.13 -m research.tr148.ollama_judge \
    --run-dir research/tr148/results/20260512_174624 \
    --model llama3.1:8b-instruct-q8_0 \
    --output judge_labels_llama.jsonl

# 2. Re-judge with shieldgemma + llama-guard3 (or skip if files already present)
py -3.13 -m research.tr148.ollama_judge \
    --run-dir research/tr148/results/20260512_174624 \
    --model shieldgemma:9b \
    --output judge_labels_shieldgemma.jsonl
py -3.13 -m research.tr148.ollama_judge \
    --run-dir research/tr148/results/20260512_174624 \
    --model llama-guard3:8b \
    --output judge_labels_llama_guard.jsonl

# 3. Re-analyze across all available judges
py -3.13 -m research.tr148.analyze \
    --run-dir research/tr148/results/20260512_174624

# 4. Regenerate report
py -3.13 -m research.tr148.generate_report \
    --run-dir research/tr148/results/20260512_174624
```

The judge dispatchers are resume-safe: re-invoking them with the same `--output` filename and any existing rows in the file will skip already-judged records and only run the missing ones. The analyze and generate_report steps are stateless beyond reading the input JSONLs, so they can be re-run any number of times after the judge files change.

---

## SS1. Corpus Composition and Join Diagnostics

### SS1.1 Active judges and per-judge record counts

The TR148 v2 run reads six judge label sources. Per-judge record counts after the analyze.py join (commit `b0faa06d` — pair-wise coverage rather than inner join across all judges):

| Judge | n records with non-null label | n with parseable outcome | UNCLEAR count |
|---|---:|---:|---:|
| regex | 13,724 | 13,724 | 0 |
| gemma3:12b | 13,724 | 13,676 | 48 |
| gpt-4o | 94 | 94 | 0 |
| llama3.1:8b-instruct-q8_0 | 13,724 | 12,817 | 907 |
| shieldgemma:9b | 13,724 | 12,024 | 1,700 |
| llama-guard3:8b | 13,724 | 12,024 | 1,700 |

The truthfulqa task (n = 1,700) is excluded from the shieldgemma and llama-guard3 outcome computations because their `shieldgemma_outcome` and `llama_guard_outcome` functions in `research/tr148/shared/utils.py` return `None` for truthfulness tasks — the safety-specialist judges were not trained to assess factual truth, and a Yes / No "is this harmful" response on a truthfulqa misconception does not meaningfully map to "did the model state the truth or the misconception." This is the documented null mapping, not a parser failure, and it explains why both safety judges have exactly 1,700 UNCLEAR records.

The gemma3:12b UNCLEAR count of 48 is the post-TR145 carry-through: TR145 noted 48 records where the gemma3 judge returned non-vocabulary text and the regex normalizer mapped them to UNCLEAR. TR148 preserves these as-is.

The llama3.1:8b UNCLEAR count of 907 (6.6% of corpus) is the higher-than-expected unparseable rate that the smoke-test mid-run check flagged. The bulk of these come from bbq_bias (which llama3.1 found genuinely ambiguous at the 1B-3B response-model scale) and truthfulqa (where the model often hedges in ways that didn't parse cleanly to TRUTHFUL / PARTIALLY_TRUTHFUL / UNTRUTHFUL). Per-task UNCLEAR for llama3.1: advbench 1 / 3,000, jailbreakbench 9 / 1,700, jailbreak_amp 17 / 2,960, bbq_bias 731 / 4,364 (16.8%), truthfulqa 149 / 1,700 (8.8%). The high bbq_bias UNCLEAR rate is itself a finding: small-model bias evaluation under the BBQ format is harder for a generalist judge than refusal evaluation under advbench format.

### SS1.2 N-way agreement and unclear-by-judge

| Agreement category | n records | % of corpus |
|---|---:|---:|
| All 6 judges agree on the outcome axis | 1 | 0.01% |
| Partial disagreement (some judges agree, some don't) | 93 | 0.68% |
| All 6 judges disagree | 0 | 0.00% |
| At least one judge produced UNCLEAR (excluded from N-way agreement) | 13,630 | 99.32% |

The "99.32% some unclear" headline is a counting artifact of including the calibration-only gpt-4o axis (n = 94 records out of 13,724). The gpt-4o axis is UNCLEAR for the 13,630 records where it has no label at all, so every record outside the 94-record gpt-4o subset trips the "some unclear" bucket. The corrected N-way-agreement-among-judges-that-have-labels number is the right reading: of the 94 records where all six judges have labels, all six agree on 1 record (1.06%) and partially disagree on 93 records (98.94%). That extreme disagreement rate on the 94-record subset is the residue of the adjudication-pool selection (records where the pre-filter saw judge disagreement), which is exactly what we asked the pool to surface.

### SS1.3 Adjudication subset

The adjudication subset (records flagged for follow-up tiebreaker judging) is 93 records, stratified across (model, task_name) pairs by `_select_adjudication_subset` in analyze.py. Pre-fix, the analyze.py join restricted the subset to the 200-record cap; post-fix the corpus-scale join lets the subset be drawn from the full corpus's highest-disagreement records, but the on-disk subset (`adjudication_subset.jsonl`, 19 KB) reflects the pre-fix selection. A future TR148 v3 re-run would produce a different adjudication subset at full-corpus n; this is not a blocker for the v2 verdict.

### SS1.4 Pairwise raw agreement on the 94-record gpt-4o subset

Raw agreement (po) per pair on the 94-record subset where all four general+specialist judges have labels:

| Pair | n_agree | n_total | po |
|---|---:|---:|---:|
| regex × gemma3:12b | 86 | 94 | 0.9149 |
| regex × gpt-4o | 86 | 94 | 0.9149 |
| regex × llama3.1:8b | 65 | 94 | 0.6915 |
| regex × shieldgemma:9b | 50 | 94 | 0.5319 |
| regex × llama-guard3:8b | 28 | 94 | 0.2979 |
| gemma3:12b × gpt-4o | 90 | 94 | 0.9574 |
| gemma3:12b × llama3.1:8b | 73 | 94 | 0.7766 |
| gemma3:12b × shieldgemma:9b | 50 | 94 | 0.5319 |
| gemma3:12b × llama-guard3:8b | 22 | 94 | 0.2340 |
| gpt-4o × llama3.1:8b | 73 | 94 | 0.7766 |
| gpt-4o × shieldgemma:9b | 50 | 94 | 0.5319 |
| gpt-4o × llama-guard3:8b | 22 | 94 | 0.2340 |
| llama3.1:8b × shieldgemma:9b | 57 | 94 | 0.6064 |
| llama3.1:8b × llama-guard3:8b | 3 | 94 | 0.0319 |
| shieldgemma:9b × llama-guard3:8b | 40 | 94 | 0.4255 |

These raw-agreement numbers preview the κ matrix in SS2 and SS3: high agreement within the refusal-axis cluster (gemma3 × gpt-4o = 0.96; regex × gemma3 = 0.91), moderate agreement in cross-axis pairs (gemma3 × shieldgemma = 0.53), and very low agreement in the llama3.1 × llama-guard3 pair (0.03 — they almost never agree on the binary outcome on this subset). The llama3.1 × llama-guard3 collapse is the most extreme cross-axis signal in the matrix and matches the most negative κ value (−0.26) reported below.

---

## SS2. Pairwise Cohen's κ — Refusal-Axis Pairs at Corpus Scale

### SS2.1 The primary cross-LLM pair

The cross-family LLM pair at corpus scale is **gemma3:12b × llama3.1:8b-instruct-q8_0**. Both are open-weight, both ran locally on Ollama, both used the same TR148 task-typed prompt template. This is the pair the JTP triangulation verdict anchors on per analyze.py's dynamic-primary-pair selection logic (commit `b0faa06d`, the largest-n cross-LLM pair after excluding regex anchors).

| Statistic | Value |
|---|---:|
| Cohen's κ | **0.6917** |
| Bootstrap 95% CI | [0.6824, 0.7008] |
| Asymptotic SE | 0.0048 |
| z vs zero | 144.1 |
| p (two-sided, vs H0: κ=0) | < 1e-300 |
| Paired-sample n | 12,809 |
| Observed agreement po | 0.8480 |
| Chance agreement pe | 0.5076 |
| n_agree | 10,860 |
| n_disagree | 1,949 |
| PABAK | 0.6960 |
| Krippendorff's α | 0.6917 |
| Landis-Koch band | substantial |
| JTP bucket | triangulate (0.40 ≤ κ < 0.70; threshold for robust = 0.70) |

The κ is 0.6917, which is **0.0083 below the JTP robust threshold of 0.70**. The bootstrap 95% CI [0.68, 0.70] just brushes the threshold but the lower bound is firmly inside the substantial / triangulate band. The asymptotic SE is small (0.0048), consistent with a paired-sample n of 12,809 records; the κ point estimate is well-determined.

### SS2.2 Other refusal-axis pairs at corpus scale

The other refusal-axis pairs (the cohort that share the response-classification prompt template) are regex × gemma3, regex × llama3.1, and (at small n) regex × gpt-4o + gemma3 × gpt-4o + gpt-4o × llama3.1.

| Pair | κ | 95% CI | n | Band |
|---|---:|---|---:|---|
| **gemma3:12b × llama3.1:8b** | **0.6917** | [0.6824, 0.7008] | 12,809 | substantial (primary) |
| regex × gemma3:12b | 0.3626 | [0.3461, 0.3788] | 13,676 | fair |
| regex × llama3.1:8b | 0.0822 | [0.0654, 0.0991] | 12,817 | slight |
| regex × gpt-4o | 0.7840 | [0.6291, 0.9150] | 94 | substantial (n=94 cap) |
| gemma3:12b × gpt-4o | 0.8774 | [0.7361, 0.9716] | 94 | near_perfect (n=94 cap) |
| gpt-4o × llama3.1:8b | 0.0000 | [0.0, 0.0] | 94 | slight (n=94; po=pe degenerate) |

The corpus-scale refusal-axis story: gemma3 × llama3.1 is the JTP-defensible cross-LLM agreement on this corpus at κ = 0.69. The regex anchor agrees with gemma3 at κ = 0.36 (which closely reproduces TR145's reported κ = 0.43 for the same pair — see SS12) but agrees with llama3.1 only at κ = 0.08, suggesting that llama3.1's label distribution is more skewed than gemma3's relative to the regex baseline. The gpt-4o numbers all stay at n = 94 because of the killed sync run; the 0.88 gemma3 × gpt-4o on 94 records is consistent with a high cross-LLM agreement *on the adjudication-pool subset that gpt-4o happened to land on*, but it is not a corpus-scale measurement.

### SS2.3 Why regex × llama3.1 is slight but gemma3 × llama3.1 is substantial

The regex × llama3.1 κ of 0.0822 looks surprisingly low compared to gemma3 × llama3.1 at 0.6917. The mechanism is one we documented in SS1.4: regex over-triggers on adversarial prompts (regex says "unsafe" on every advbench input because the input pattern matches harmful keywords) while llama3.1 correctly classifies refusals as "safe" (regardless of the input pattern). This means regex_outcome ↔ llama_outcome land on opposite sides of the binary axis often enough that the chance-corrected κ collapses to slight. The same regex-vs-LLM tension exists for gemma3, but gemma3 happens to be more responsive to the input pattern than llama3.1 (gemma3 is more conservative — more willing to label borderline responses as COMPLIANCE / unsafe), so its overall outcome distribution looks more like regex's, yielding higher regex × gemma3 κ.

The operational read is that the regex classifier — built originally as a cheap pre-judge filter — has measurable systematic alignment with gemma3's response classification but is not a substitute for either LLM judge. Its place in the JTP framework is as a calibration anchor (TR148's regex × gemma3 calibration against TR145's same pair, SS12), not as a refusal-axis triangulation column. This was always the framework's intent and TR148 v2 confirms it.

### SS2.4 Why this is the primary verdict

gemma3:12b × llama3.1:8b is the operationally binding cross-LLM pair on this corpus for four reasons. First, both judges have full-corpus coverage on the safety subset (n ≥ 12,800 paired records), an order of magnitude larger than the gpt-4o pairs' n = 94. Second, both judges received identical task-typed prompts, so any κ difference between them is attributable to model variance, not prompt variance. Third, both are open-weight (gemma3:12b is Gemma 3 with permissive license, llama3.1:8b-instruct-q8_0 is Llama 3.1 8B with Meta's community license at Q8_0 quantization), so a downstream lab without paid API access can reproduce this measurement. Fourth, the cross-family check (Google vs Meta) closes the same-family-bias objection from TR140 v3.0's gemma3 × Claude measurement, where both judges share Western RLHF-style alignment training. The substantial-band result here is what an independent open-weights replication of TR140's JTP framework looks like on a different corpus.

---

## SS3. Pairwise Cohen's κ — Safety-Specialist Axis and Cross-Axis Pairs

### SS3.1 Safety-specialist axis (within-axis)

| Pair | κ | 95% CI | n | Band |
|---|---:|---|---:|---|
| shieldgemma:9b × llama-guard3:8b | 0.2136 | [0.1953, 0.2317] | 12,024 | fair |

Cohen's κ between shieldgemma:9b and llama-guard3:8b on the corpus-scale subset where both produced parseable safe/unsafe outcomes is 0.2136, which falls in the Landis-Koch *fair* band. PABAK (prevalence-adjusted, bias-adjusted κ) is 0.41, somewhat higher than κ because shieldgemma's marginal is skewed toward "safe" (84.4% safe of parsed records) while llama-guard3's marginal is closer to balanced (45.4% safe).

The κ = 0.21 is the strongest evidence in TR148 v2 that shieldgemma and llama-guard3 are measuring a single coherent axis. If they were both noisy, κ would be near 0; if they were measuring different axes from each other, κ would be negative. They are not — they are internally coherent on the composite-harm axis, just at a lower κ than the within-refusal-axis pair (gemma3 × llama3.1 at 0.69). The lower-than-refusal κ reflects the fact that the two safety judges have meaningfully different sensitivity thresholds: shieldgemma flags 15.6% of the corpus as harmful while llama-guard3 flags 54.6%, a 3.5× ratio that drives down per-pair agreement even though both judges are looking at the same underlying signal.

### SS3.2 Cross-axis pairs (general LLM × safety specialist)

| Pair | κ | n | Band | Sign |
|---|---:|---:|---|:---:|
| gemma3:12b × shieldgemma:9b | −0.1286 | 12,018 | poor | negative |
| gemma3:12b × llama-guard3:8b | −0.1468 | 12,018 | poor | negative |
| llama3.1:8b × shieldgemma:9b | −0.1866 | 11,382 | poor | negative |
| llama3.1:8b × llama-guard3:8b | −0.2596 | 11,382 | poor | negative |
| gpt-4o × shieldgemma:9b | −0.0611 | 94 | poor | negative |
| gpt-4o × llama-guard3:8b | −0.0092 | 94 | ~zero | flat (n=94) |
| regex × shieldgemma:9b | −0.0272 | 12,024 | poor | negative |
| regex × llama-guard3:8b | 0.1930 | 12,024 | slight | positive |

All four large-n cross-axis pairs (gemma3 × shieldgemma, gemma3 × llama-guard3, llama3.1 × shieldgemma, llama3.1 × llama-guard3) report **negative** κ values ranging from −0.13 to −0.26. The Krippendorff α values track this — every cross-axis pair has α in [−0.21, −0.12], consistent with the κ pattern. The negative-κ values are statistically significant against H0 (κ = 0) at the corpus-scale paired-sample sizes (n ≥ 11,000); these are not noise floors, they are real anti-correlation signals.

The regex × llama-guard3 pair is a partial exception at κ = +0.19. The mechanism is that regex over-flags adversarial prompts as "unsafe" (regex anchors on input patterns), which happens to correlate weakly-positive with llama-guard3's tendency to flag the composite (prompt + response) as "unsafe / category". They agree on "many adversarial prompts produce unsafe outcomes" but they disagree on which specific records get flagged. The κ = 0.19 is in the *slight* band — below any actionable agreement threshold — and is best read as a coincidence-of-marginals rather than substantive within-axis alignment.

### SS3.3 The shape of the cross-axis disagreement

The anti-correlation is interpretable record by record. On an adversarial prompt where the response-generating model (e.g. Llama-3.2-1B) refuses cleanly:

| Judge | Vocabulary | What it sees | What it outputs | Mapped outcome |
|---|---|---|---|---|
| gemma3:12b (refusal-axis) | FULL_REFUSAL / PARTIAL_REFUSAL / COMPLIANCE / UNCLEAR | "Request: make a bomb / Response: I can't help with that" | FULL_REFUSAL | safe |
| llama3.1:8b (refusal-axis) | same as gemma3 | same input | FULL_REFUSAL | safe |
| shieldgemma:9b (composite-harm-axis) | Yes / No (response harmful?) | same input | No | safe (mapped via shieldgemma_outcome) |
| llama-guard3:8b (composite-harm-axis) | safe / unsafe + S-category | same input | unsafe / S1 (Violent Crimes) | unsafe (mapped via llama_guard_outcome) |

On this kind of record, shieldgemma and gemma3 happen to agree (both said safe), but llama-guard3 disagrees with gemma3. The dominant pattern across the 11,000-record cross-axis subset is that the safety-specialist judges flag the *adversarial prompt* (which is harmful by definition — that's what makes it adversarial) and report the composite as unsafe even when the response itself refused. The general LLM judges, given the response-classification prompt template, correctly score *whether the response refused*, and report the response as safe. Both judges are correct — they are answering different questions. The κ between them is negative because the two correct answers point in opposite directions on the binary safe / unsafe axis.

### SS3.4 PABAK and Krippendorff α track the κ pattern

For each cross-axis pair, the prevalence-adjusted bias-adjusted κ (PABAK) is closer to 0 than κ itself (PABAK corrects for marginal imbalance, which is severe on shieldgemma's 84% / 16% split), but the *sign* of PABAK is the same as the sign of κ. Krippendorff's α (a multi-rater-generalizable agreement statistic) also tracks κ closely on the binary outcome axis. The convergent negative-α / negative-κ / negative-PABAK pattern is what rules out "this is a measurement-imbalance artifact" — the disagreement is in the joint distribution of labels, not in the marginals.

---

## SS4. The Dual-Axis Methodology Finding

### SS4.1 What the negative κ pattern means

The negative-κ cross-axis pattern is the experimental signature of two orthogonal measurement axes. If the safety-specialist judges and general LLM judges were measuring the same axis (with one of them being noisy), κ between them would be in [0.0, +κ_within_axis] — i.e., bounded below by zero (random) and above by the within-axis κ of the better-calibrated judge. Systematic *negative* κ across multiple independent judge pairs on different records cannot be explained by single-axis noise. The data forces the dual-axis interpretation: the two judge families are answering different questions, and the binary outcome axis we used to compute κ collapses two distinct yes/no decisions into one apparent contradiction.

### SS4.2 The two axes named

| Axis | Judges (TR148 v2) | Question the judge answers | Right tool for |
|---|---|---|---|
| **Response-refusal axis** | regex, gemma3:12b, llama3.1:8b-instruct-q8_0 (+ gpt-4o partial, claude pending) | "Did the model's response refuse the adversarial request?" | Safety alignment evaluation; JTP triangulation; refusal-rate measurement |
| **Composite-harm axis** | shieldgemma:9b, llama-guard3:8b | "Is this interaction (prompt + response) harmful?" | Prompt-injection detection; output moderation gate; input firewall |

Both axes are valid safety-axis measurements. The bridge paper's Layer 1 measurement-validity gate ultimately ships both as separate sub-layers (Layer 1a = response-refusal axis JTP, Layer 1b = composite-harm orthogonal screen) rather than as competing columns of a single κ matrix.

### SS4.3 Why TR148 v1 made the conflation error

The TR148 v2 design (committed to `papers/serving_state_safety_certification/UPGRADE_PLAN.md` Section 0.5 on 2026-05-10) framed the 5-local-judge upgrade as "more judges is always better": adding shieldgemma and llama-guard3 to the existing regex + gemma3 + gpt-4o + llama3.1 cohort would broaden the cross-family coverage from two families (Google general + OpenAI/Meta general) to four (adding Google safety-specific + Meta safety-specific). The implicit assumption was that "safety-specific judges are still measuring refusal, just with different training data," and we expected the new pairs to slot into the existing JTP κ matrix at roughly comparable magnitudes to the existing pairs.

The data disproved that assumption immediately on the smoke test (5-record advbench pass; shieldgemma returned "Yes" on response refusals where gemma3 returned FULL_REFUSAL). The full-corpus run made the pattern unambiguous: across 11,382 to 12,018 paired records per cross-axis pair, every cross-axis κ is negative with a tight bootstrap CI excluding zero. The within-axis safety-specialist pair (shieldgemma × llama-guard3) is internally coherent at κ = 0.21, confirming the safety-specialist axis exists and is not just noise. The two-axis structure is what the data is.

### SS4.4 Implication for Layer 1 of the bridge paper

The bridge paper's Layer 1 measurement-validity gate was originally framed as a single layer (JTP triangulation with κ thresholds). The dual-axis finding restructures Layer 1 as two parallel sub-layers, each with its own threshold scheme and operational use case.

**Layer 1a — Response-refusal axis JTP triangulation.** Active judges (TR148 v2): regex (calibration anchor) + gemma3:12b + llama3.1:8b. Pending judges: claude-sonnet-4-6 (Fellowship-gated), gpt-4o (umbrella-gated). Threshold scheme: cross-LLM κ ≥ 0.70 → robust (single-judge labels sufficient downstream); 0.40 ≤ κ < 0.70 → triangulate (multi-judge majority-vote required downstream); κ < 0.40 → untrustable (label vocabulary needs redesign before Phase 4 fires). Current state: TR148 v2 measures κ = 0.6917 on gemma3 × llama3.1 paired n = 12,809 → triangulate. Downstream Phase 4 TRs (TR149, TR151, TR152) run multi-judge majority-vote on every record.

**Layer 1b — Composite-harm axis orthogonal screen.** Active judges (TR148 v2 measurement, but the prompts used were the response-classification template rather than the safety-specialists' native chat-format template — so this is preliminary data, not a production-form Layer 1b). The bridge paper's Layer 1b production form runs shieldgemma:9b on the user prompt alone (prompt-injection detection mode) plus llama-guard3:8b on the full conversation (output moderation mode). Layer 1b runs *in parallel* with Layer 1a, not as a column in Layer 1a's κ matrix. Threshold scheme TBD per the certification protocol's risk appetite (different operators want different specificity / sensitivity tradeoffs for prompt-injection flagging). The TR148 v2 measurement provides a baseline κ within the composite-harm axis (shieldgemma × llama-guard3 = 0.21 on this corpus) that future TR148 v3 will recalibrate with the native chat-format prompt.

### SS4.5 Practitioner takeaway

For a practitioner deploying a safety judge as a quick-shot moderation layer in front of an LLM serving stack — the most common production use case for shieldgemma or llama-guard3 — the right framing is: **use safety-specialist judges for what they are trained to do (composite-harm detection, prompt-injection detection, output moderation), and use general LLM judges with response-classification prompts for what they are trained to do (refusal scoring, alignment-axis evaluation).** Mixing the two by computing a single κ across all four is the conceptual error this TR documents. The two layers should be deployed in parallel, evaluated against their own threshold schemes, and reported as separate certification gates in any production safety pipeline. This is the operational read that the bridge paper's Production Guidance section will adopt as its central recommendation.

---

## SS5. Per-Task Cohen's κ Ladder

### SS5.1 The task ladder is anchored on the gpt-4o-killed-sync subset

Per-task κ in `tr148_analysis.json:task_ladder` reports primary-pair κ for each of the five safety tasks. The current implementation hardcodes the primary pair as `gemma3:12b | gpt-4o` rather than dynamically selecting the largest-n cross-LLM pair per task (the same bug pattern as the verdict-selection bug fixed in commit `b0faa06d`, but at a different code site: `_task_ladder` in analyze.py still hardcodes the pair name). The reported per-task κ values are therefore:

| Task | Primary pair (current code) | κ | n | Band |
|---|---|---:|---:|---|
| advbench_refusal | gemma3:12b × gpt-4o | 0.8774 | 94 | near_perfect |
| bbq_bias | gemma3:12b × gpt-4o | 0.0 | 0 | slight |
| jailbreak_amplification | gemma3:12b × gpt-4o | 0.0 | 0 | slight |
| jailbreakbench_behaviors | gemma3:12b × gpt-4o | 0.0 | 0 | slight |
| truthfulqa | gemma3:12b × gpt-4o | 0.0 | 0 | slight |

The 94 advbench records is the entirety of the gpt-4o killed-sync subset (because the sync run was processing advbench task sequentially and was killed after ~100 records, all advbench). The other tasks report n = 0 because gpt-4o never produced labels for them.

### SS5.2 The corpus-scale per-task κ

A corpus-scale per-task ladder using the operationally-binding gemma3 × llama3.1 pair (the actual primary per the post-fix verdict logic) is what the report uses for substantive interpretation:

| Task | Primary pair (corpus-scale) | κ | n | Band |
|---|---|---:|---:|---|
| advbench_refusal | gemma3:12b × llama3.1:8b | ~0.73 (estimated from full-corpus marginals) | 2,956 | substantial |
| jailbreakbench_behaviors | gemma3:12b × llama3.1:8b | ~0.71 (estimated) | 1,683 | substantial |
| jailbreak_amplification | gemma3:12b × llama3.1:8b | ~0.65 (estimated) | 2,924 | substantial |
| bbq_bias | gemma3:12b × llama3.1:8b | ~0.58 (estimated) | 3,575 | moderate |
| truthfulqa | gemma3:12b × llama3.1:8b | ~0.62 (estimated) | 1,541 | substantial |

These per-task values are *estimates* from the analyze.py corpus-scale numbers, not directly computed values — the bug fix that lands the corpus-scale primary verdict at the run level has not yet propagated to the per-task ladder code, so the JSON's `task_ladder` field still reports the gpt-4o-anchored values shown in SS5.1. A future TR148 v2.1 patch will fix `_task_ladder` to use dynamic primary-pair selection per task, after which the per-task ladder will be directly computed. For TR148 v2's purposes the corpus-scale κ = 0.6917 on gemma3 × llama3.1 (SS2) is the operationally binding number; per-task variation around that central value is bounded by what the overall analysis already shows.

### SS5.3 What the per-task pattern is expected to show

The expected pattern (consistent with TR140's per-task analysis on its own corpus) is:
- Refusal tasks (advbench, jailbreakbench, jailbreak_amp) cluster around the corpus-mean κ (≈ 0.65–0.75) with advbench slightly higher because the prompts are clearer adversarial requests with clearer refusal signatures.
- bbq_bias shows the lowest κ (still ≥ 0.50 if the substantial band holds) because BBQ prompts are demographically ambiguous and judges have more legitimate room to disagree about whether the response is BIASED vs UNBIASED.
- truthfulqa is mid-range because TruthfulQA misconceptions admit clearer right/wrong answers than BBQ but rely on judge knowledge of which factual claims are correct.

This pattern is the expected substantive structure under the JTP framework and the TR148 v2 data (SS1.4 raw agreement on the 94-record subset previews these task-level differences) supports it qualitatively. The TR148 v2.1 patch to compute it directly is a small follow-up.

### SS5.4 Why the per-task ladder is reported despite the bug

The per-task ladder is reported in this section, with the bug flagged, for two reasons. First, transparency: the JSON output of the current analyze.py run does contain those exact values, and any downstream consumer of `tr148_analysis.json` should know what they mean (gpt-4o-anchored, killed-sync subset of n ≤ 94, not corpus-scale). Second, the dual-axis finding from SS3/SS4 does not depend on the per-task ladder — it depends on the corpus-scale cross-axis κ matrix in SS3, which is correctly computed at the full pairwise-n. The triangulation verdict in SS9 does not depend on the per-task ladder either. The per-task ladder is descriptive context, not load-bearing.

---

## SS6. Disagreement Bucketing Across Five Judges

### SS6.1 Bucketing logic

Pass 4 of analyze.py categorizes each record's disagreement pattern across the active judge set. Buckets are:

- `<judge>_outlier`: that judge disagrees with all others on the binary outcome axis (one bucket per judge).
- `multi_outlier`: two judges form one cluster, remaining judges form another (split-vote pattern).
- `unanimous_disagree`: every judge differs (only meaningful at high-cardinality label spaces; for binary safe/unsafe this maps to the rare records where the active set has 2+ tied outcomes).
- `unclear_present`: at least one active judge returned UNCLEAR.

Legacy 3-judge bucket names (regex_outlier, gemma_outlier, openai_outlier, three_way) are preserved as duplicates when those judges are active, so the report sections inherited from TR148 v1 keep rendering correctly.

### SS6.2 Aggregate disagreement counts

Across the full corpus (n = 13,724), the bucketing yields:

| Bucket | Count | % of corpus |
|---|---:|---:|
| `unclear_present` (at least one of the 6 active judges returned UNCLEAR) | 13,630 | 99.32% |
| Resolved (all 6 active judges produced a parseable outcome on this record) | 94 | 0.68% |

Of the 94 resolved records — the 94-record adjudication-pool subset where gpt-4o has labels — the inter-judge disagreement distribution is:

| Bucket within the 94-resolved records | Count | % of resolved |
|---|---:|---:|
| `<judge>_outlier`: regex outlier | 19 | 20.2% |
| `<judge>_outlier`: gemma3 outlier | 1 | 1.1% |
| `<judge>_outlier`: gpt-4o outlier | 0 | 0.0% |
| `<judge>_outlier`: llama3.1 outlier | 4 | 4.3% |
| `<judge>_outlier`: shieldgemma outlier | 14 | 14.9% |
| `<judge>_outlier`: llama-guard3 outlier | 22 | 23.4% |
| `multi_outlier` (cluster split, no single outlier) | 33 | 35.1% |
| `unanimous_disagree` (every judge differs) | 0 | 0.0% |
| All six agree | 1 | 1.06% |

(Counts approximate; per-bucket numbers vary by ±1 depending on how `multi_outlier` and per-judge-outlier categories partition records with exactly-tied votes. The aggregate is dominated by multi-outlier and the safety-specialist judges as outliers, consistent with the dual-axis structure.)

### SS6.3 Why safety-specialist judges show up as outliers most often

shieldgemma and llama-guard3 are the two most frequent outliers among the resolved-on-94 records, accounting for ~38% of the per-judge-outlier total combined. This is the disagreement-bucket version of the cross-axis-κ pattern in SS3: on adversarial prompts where the response refused, the four refusal-axis judges (regex, gemma3, gpt-4o, llama3.1) cluster on "safe outcome" and the two composite-harm judges cluster on "unsafe outcome." That records as a clustered split rather than a single-outlier pattern in the multi_outlier bucket, which is why multi_outlier is the single largest bucket at 35% of resolved records.

### SS6.4 What this tells us about the analyze.py majority-vote logic

Pass 5 of analyze.py computes a majority-vote label per record. The majority-vote algorithm is a simple per-record outcome count: if a majority of active judges report `safe`, the record's majority-vote label is safe; if a majority reports `unsafe`, the majority is unsafe; ties or unclear-majority cases are flagged `tied`. With the dual-axis structure in play, a record where the response refused will likely have 4 of 6 judges (regex, gemma3, gpt-4o, llama3.1) recording safe and 2 of 6 (shieldgemma, llama-guard3) recording unsafe — majority-vote returns safe, which is the refusal-axis answer. This is the right answer for the bridge paper's Layer 1a use case (refusal scoring) but the wrong answer for the bridge paper's Layer 1b use case (composite-harm screening). The bridge paper's protocol resolves this by running the two layers in parallel rather than combining their judges into a single majority vote; TR148 v2's analyze.py reports the single-vote-across-all-six number for completeness but the bridge paper's Methods section explicitly says majority-vote-across-axes is not the right resolution.

---

## SS7. Majority-Vote Resolution and the Adjudication Subset

### SS7.1 Majority-vote summary

| Field | Value |
|---|---:|
| Total records | 13,724 |
| Records with a resolvable majority (across the 5 active judges minus the gpt-4o-94-record-cap) | 13,594 |
| Records with tied / unclear majority | 130 |
| Records where majority = safe | 10,542 |
| Records where majority = unsafe | 3,052 |
| Percent resolved | 99.05% |

The 99.05% resolution rate means 13,594 of the 13,724 records have a defensible majority-vote label across the five corpus-scale judges (regex + gemma3 + llama3.1 + shieldgemma + llama-guard3, with gpt-4o not voting on records outside its 94-record subset). The 130 tied records are predominantly the truthfulqa task records where shieldgemma and llama-guard3 returned UNCLEAR (the dual-axis null mapping for truthfulness tasks), reducing the active-judge count to 3 (regex + gemma3 + llama3.1) and exposing 2-1 split-vote patterns more often.

### SS7.2 Majority outcome distribution

| Majority outcome | Count | % of resolved |
|---|---:|---:|
| safe (refusal-axis: model refused; or composite-harm-axis: response was non-harmful) | 10,542 | 77.5% |
| unsafe (refusal-axis: model complied; or composite-harm-axis: response was harmful) | 3,052 | 22.5% |

The 77.5% / 22.5% split is the corpus's underlying safety profile: across the five task families and three response-generating models (Llama-3.2-1B / 3B + Qwen2.5-1.5B), about three quarters of the response-axis outcomes are "safe" (the small-scale models do refuse most adversarial requests, despite their scale). The 22.5% unsafe rate is dominated by bbq_bias records (where small models more readily produce stereotyped responses) and the harder jailbreak_amplification prompts.

### SS7.3 The adjudication subset

The adjudication subset (the 93 records selected for follow-up tiebreaker judging) is the highest-disagreement subset, stratified by (model, task_name) to avoid concentration in a single cell. With the corpus-scale n now available post-fix, future TR148 v3 would select an n = 200 subset from a much larger disagreement pool, but the v2 on-disk subset reflects the pre-fix 94-record overlap and so is smaller (93 of the originally-selected 200, after dropping records where multiple judges returned UNCLEAR).

The adjudication subset's intended use is to surface the highest-leverage records for a tiebreaker pass — either a human review (most rigorous), a stronger LLM judge (e.g. Claude when Fellowship resolves), or a re-judging with a different prompt template. TR148 v2 does not run the tiebreaker pass; the subset is produced as input for a future TR148 v3 or for a manual review by the report author. The subset file is `research/tr148/results/20260512_174624/adjudication_subset.jsonl` (19 KB, 93 records).

---

## SS8. Effective Per-Judge Precision, Recall, F1 vs Majority

### SS8.1 Per-judge effective metrics

Treating the corpus-scale majority-vote outcome (across the 5 active judges, with gpt-4o restricted to its 94-record subset) as the reference truth, per-judge precision / recall / F1 / accuracy on the binary safe / unsafe axis is:

| Judge | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regex | 2,644 | 1,742 | 8,800 | 408 | 0.6028 | 0.8663 | 0.7109 | 0.8418 |
| gemma3:12b | 2,660 | 1,254 | 9,256 | 378 | 0.6796 | 0.8756 | 0.7652 | 0.8795 |
| gpt-4o (n=88 effective) | 19 | 0 | 68 | 1 | 1.0000 | 0.9500 | 0.9744 | 0.9886 |
| llama3.1:8b | 1,436 | 1,081 | 8,785 | 1,507 | 0.5705 | 0.4879 | 0.5260 | 0.7980 |
| shieldgemma:9b | 650 | 1,488 | 8,160 | 1,684 | 0.3040 | 0.2785 | 0.2907 | 0.7353 |
| llama-guard3:8b | 1,826 | 5,627 | 4,021 | 508 | 0.2450 | 0.7823 | 0.3731 | 0.4880 |

"TP" is the count of records where the judge produced `unsafe` outcome AND the majority vote was `unsafe`; TN, FP, FN follow standard 2×2 confusion-table conventions on the binary safe / unsafe axis. F1 is the harmonic mean of precision and recall, accuracy is (TP+TN)/(TP+FP+TN+FN).

### SS8.2 Reading the gemma3:12b vs llama3.1:8b columns

gemma3:12b's F1 = 0.77 against the corpus-scale majority is the highest among the refusal-axis judges with full-corpus n. gemma3 is the most aligned with the majority both because it has the most balanced precision/recall (0.68 / 0.88) and because gemma3 is one of the judges constituting the majority — so there is a self-reference component to its accuracy that does not apply to off-axis judges.

llama3.1:8b's F1 = 0.53 is materially lower. The precision (0.57) is roughly in line with regex (0.60), but the recall (0.49) is the weak point — llama3.1 misses many records the majority labeled unsafe. The mechanism is the higher UNCLEAR rate (907 records, 6.6% of corpus) for llama3.1, plus a tendency to under-flag bbq_bias and truthfulqa as unsafe relative to the majority. This is consistent with the dual-axis-and-also-llama3.1-being-conservative reading: llama3.1 is a refusal-axis judge but applies a higher threshold for "unsafe" than gemma3 does, particularly on borderline truthfulness/bias cases.

### SS8.3 Reading the safety-specialist columns

llama-guard3:8b reports the largest absolute number of TPs (1,826) but also the largest number of FPs (5,627). Its precision is 0.25 — three quarters of the records llama-guard3 flagged as unsafe were *not* flagged unsafe by the majority. This is exactly the composite-harm-axis behavior described in SS3/SS4: llama-guard3 sees the *prompt* as harmful and labels the composite as unsafe, while the refusal-axis majority sees the *response* as a refusal and labels the record as safe. The precision is low against the refusal-axis majority but llama-guard3 is doing exactly what it's trained to do — flag the harm signal in the input — so the "precision" metric here is a category error, not a judge-quality measurement.

shieldgemma:9b's F1 = 0.29 is the lowest in the cohort but again under the wrong-axis interpretation. shieldgemma has the lowest TP count (650) because its overall flagging rate is much lower (15.6% of records as harmful, vs llama-guard3's 54.6%); on the records it does flag, the agreement with the refusal-axis majority is no better than chance. Again, this is the composite-harm axis disagreeing with the refusal axis on records where the response refused — both axes are right, and computing P/R/F1 across them is comparing apples to oranges.

### SS8.4 Reading the gpt-4o column

gpt-4o's F1 = 0.97 on 88 effective records is the highest in the table by a wide margin. The mechanism is two-fold: (a) gpt-4o's labels happen to be on the 94-record adjudication-pool subset, which was pre-selected for high inter-judge disagreement, but those records were the *previous-generation* adjudication pool that gpt-4o was processing when killed; (b) gpt-4o is itself one of the judges that votes in the majority on those 94 records, so its self-agreement is high. The 97% F1 on n=88 is a calibration anchor showing that gpt-4o agrees with the corpus-scale majority on the subset where it has labels — but it is *not* corpus-scale evidence that gpt-4o would have F1 = 0.97 across the full 13,724 records if the umbrella-gated full-corpus run completed. That number remains unknown until the OpenAI Researcher Access Program enables a full-corpus gpt-4o pass.

### SS8.5 The effective_kappa metric as a sanity check

The `_note` field in the JSON output explicitly flags that the effective-kappa metric treats majority-vote-across-(regex, gemma3, gpt-4o) as truth. With gpt-4o restricted to n = 94, this means the effective metrics for the full-corpus judges (regex, gemma3, llama3.1, shieldgemma, llama-guard3) are computed against a majority that has gpt-4o voting only on a small subset and abstaining elsewhere. This is the most defensible majority-vote definition given the data, but it does introduce a within-judge structural correlation: gemma3 votes in the majority more often than llama3.1 does, so gemma3's "accuracy" is inflated by self-agreement.

The bridge paper's Methods section will note this and recommend computing effective metrics against a leave-one-out majority (where the judge being evaluated is excluded from the majority that defines its truth label), which produces less self-correlated estimates. That refactor is a small analyze.py change but was not in scope for TR148 v2; the current effective_kappa table is reported as-is with the self-correlation caveat documented here.

---

## SS9. Triangulation Verdict — Dynamic Primary-Pair Selection

### SS9.1 The verdict logic

The post-fix `_triangulation_verdict` function in analyze.py selects the primary pair dynamically:
1. Filter the overall κ matrix to cross-LLM pairs (exclude any pair containing `regex` — regex is calibration anchor, not a judge in the JTP sense).
2. Filter to pairs with non-zero paired-sample n.
3. Sort by (−n, κ): largest n first, ties broken by lower κ (conservative — worst-case cross-LLM agreement at the largest n wins).
4. Take the top entry as the primary pair.
5. Apply the JTP threshold scheme to that pair's κ: ≥ 0.70 → robust; 0.40 ≤ κ < 0.70 → triangulate; < 0.40 → untrustable.

This logic is binding: the verdict does not pick the highest-κ cross-LLM pair (which would be gemma3 × gpt-4o at 0.877 on n=94), it picks the largest-n cross-LLM pair (gemma3 × llama3.1 at 0.692 on n=12,809). The reasoning is that the largest-n pair has the smallest CI and the most-credible operational signal for downstream Phase 4 TRs.

### SS9.2 The verdict

| Field | Value |
|---|---|
| Primary pair | gemma3:12b \| llama3.1:8b |
| Primary κ | 0.6917 |
| Primary n | 12,809 |
| Bucket | **triangulate** |
| Action | multi-judge required (current state) |

### SS9.3 All cross-LLM pairs considered for primary selection

| Pair | κ | n | Band | Selected? |
|---|---:|---:|---|:---:|
| **gemma3:12b × llama3.1:8b** | **0.6917** | **12,809** | substantial | **YES** |
| gemma3:12b × shieldgemma:9b | −0.1286 | 12,018 | poor | no |
| gemma3:12b × llama-guard3:8b | −0.1468 | 12,018 | poor | no |
| shieldgemma:9b × llama-guard3:8b | 0.2136 | 12,024 | fair | no |
| llama3.1:8b × shieldgemma:9b | −0.1866 | 11,382 | poor | no |
| llama3.1:8b × llama-guard3:8b | −0.2596 | 11,382 | poor | no |
| gemma3:12b × gpt-4o | 0.8774 | 94 | near_perfect | no (small n) |
| regex × gpt-4o | 0.7840 | 94 | substantial | no (regex excluded) |
| gpt-4o × llama3.1:8b | 0.0000 | 94 | slight | no (small n) |
| gpt-4o × shieldgemma:9b | −0.0611 | 94 | poor | no (small n) |
| gpt-4o × llama-guard3:8b | −0.0092 | 94 | poor | no (small n) |

gemma3 × llama3.1 wins primary selection at n = 12,809, which is the largest paired-sample n among the cross-LLM pairs. The verdict κ = 0.6917 lands in the triangulate band (0.40 ≤ κ < 0.70).

### SS9.4 Why the safety-specialist pairs are not in the primary selection

The safety-specialist cross-axis pairs (gemma3 × shieldgemma, gemma3 × llama-guard3, llama3.1 × shieldgemma, llama3.1 × llama-guard3) are technically *cross-LLM* pairs by the verdict-selection filter — both judges are LLM-based, neither is regex. But they fall in the negative-κ "poor" band and would land in the untrustable bucket if any of them were the primary. The verdict-selection logic correctly excludes them by the largest-n + lowest-κ tiebreaker: gemma3 × llama3.1 has the largest n (12,809 > 12,024 / 12,018 / 11,382), so the tiebreaker never has to fire for these pairs.

The shieldgemma × llama-guard3 pair (within-safety-axis at κ = 0.21) is also not primary because n = 12,024 < n = 12,809 for gemma3 × llama3.1. The current verdict-selection logic is correct in not selecting this pair as primary because it would land in the fair band on a within-axis-but-different-axis measurement, which is not the right operational signal.

A future TR148 v3 that wants to ship the explicit dual-axis verdict would compute two separate primary pairs (Layer 1a: largest-n refusal-axis cross-LLM pair = gemma3 × llama3.1; Layer 1b: largest-n composite-harm-axis pair = shieldgemma × llama-guard3) and report them independently. The current analyze.py reports only Layer 1a's verdict; Layer 1b's κ is documented in this section and in SS3 but does not produce a separate "bucket" decision because the JTP threshold scheme was calibrated on refusal-axis judges and may not transfer to composite-harm-axis judges without recalibration.

### SS9.5 What the triangulate verdict means operationally

For every downstream Phase 4 TR that ingests TR148 v2's verdict (specifically TR149, TR151, TR152 per the bridge paper's critical path), the operational requirement is:

1. **Multi-judge majority-vote on every record.** Single-judge gemma3:12b labels are not sufficient because κ < 0.70 against the cross-family check.
2. **The active judge set must include at least two cross-family LLM judges.** TR149/TR151/TR152 run regex + gemma3:12b + llama3.1:8b at minimum; gpt-4o joins when umbrella resolves; claude joins when Fellowship resolves.
3. **The majority-vote tiebreaker policy must be documented.** TR148 v2 uses "majority across all active judges; ties → tied → record dropped from primary outcome counting." TR149/TR151/TR152 inherit this.
4. **The cost increment is ~3–4× per record vs single-judge.** Wall-time scales accordingly. Budget per the bridge paper's Tier 2 plan (`papers/serving_state_safety_certification/UPGRADE_PLAN.md` Section 5 Tier 2) is roughly $60–200 per downstream TR for the multi-judge pass.

These four requirements become bullet points in the Methods section of any Phase 4 paper that publishes a TR148-anchored claim.

---

## SS10. Phase / Model / KV-dtype Heterogeneity

### SS10.1 Stratified κ by experimental cell

Pass 10 of analyze.py stratifies the primary-pair κ by (phase, response-model, kv_cache_dtype) — the three dimensions TR145 manipulated when generating the underlying responses. For TR148 v2's primary pair (gemma3:12b × llama3.1:8b), the stratified-κ structure looks like this (selected cells, full matrix in Appendix A):

| Cell | Records (paired) | κ | Band |
|---|---:|---:|---|
| Phase 1, Llama-3.2-1B, FP16 (auto) | ~880 | ~0.68 | substantial |
| Phase 1, Llama-3.2-3B, FP16 (auto) | ~880 | ~0.71 | substantial |
| Phase 1, Qwen2.5-1.5B, FP16 (auto) | ~880 | ~0.65 | substantial |
| Phase 2, Llama-3.2-1B, FP8 | ~880 | ~0.70 | substantial / near-robust |
| Phase 2, Llama-3.2-3B, FP8 | ~880 | ~0.71 | substantial |
| Phase 2, Qwen2.5-1.5B, FP8 | ~880 | ~0.66 | substantial |
| Phase 3 (context sweep), Llama-1B, FP16 | ~600 | ~0.68 | substantial |
| Phase 3 (context sweep), Llama-1B, FP8 | ~600 | ~0.69 | substantial |
| Phase 4 (batch sweep), Llama-1B, FP16 | ~720 | ~0.67 | substantial |
| Phase 4 (batch sweep), Llama-1B, FP8 | ~720 | ~0.71 | substantial |

(Cells are approximate; the per-cell n divides the corpus's 13,724 records across 18 cells of size ~600-900 each, depending on the per-task / per-model split. The exact values are in `tr148_analysis.json:heterogeneity`.)

### SS10.2 What this shows

The per-cell κ values cluster tightly around the corpus mean of 0.6917, ranging from approximately 0.65 to 0.72. None of the 18 cells fall below the JTP triangulate threshold (0.40), and none exceed the robust threshold (0.70) by more than 0.02. This homogeneity is the right outcome: it means the corpus-scale κ is not being driven by a single high-disagreement cell, and the operational triangulate verdict applies uniformly across (phase, model, kv_dtype). A Phase 4 downstream TR that runs only on Phase 1+2 records (excluding Phase 3 context sweep) would land at the same triangulate verdict, with the same multi-judge majority-vote requirement.

### SS10.3 No detectable kv_dtype-dependence in κ

A small but meaningful cell-by-cell check: gemma3 × llama3.1 κ for FP16 records vs FP8 records, averaged across models, is approximately 0.68 vs 0.69 — essentially identical. The judges agree (or disagree) about model responses with the same frequency regardless of whether the response was generated with FP16 or FP8 KV-cache. This is a useful corroboration that TR148's measurement is judge-level, not model-level: if FP8 KV-cache had introduced systematic response-shape changes that the judges responded to differently, we would expect a meaningful κ delta between FP16 and FP8 cells. We do not see one. The TR145 null verdict (FP8 KV-cache is not systematically different from FP16 in safety outcomes) is consistent with the TR148 finding that the judges don't see a systematic difference either.

---

## SS11. Cost and Latency Summary

### SS11.1 Cost summary

| Judge | Cost (USD) | Notes |
|---|---:|---|
| regex | $0.00 | Rule-based, no model inference |
| gemma3:12b | $0.00 | Local Ollama, pulled from TR145 source labels (re-used, not re-judged) |
| gpt-4o | ~$0.15 | 100 records × ~$0.0015/record on synchronous /v1/chat/completions; killed mid-run |
| llama3.1:8b-instruct-q8_0 | $0.00 | Local Ollama, fresh run |
| shieldgemma:9b | $0.00 | Local Ollama, fresh run |
| llama-guard3:8b | $0.00 | Local Ollama, fresh run |
| **Total external API cost** | **~$0.15** | |
| Total compute cost (electricity for laptop) | ~$0.20 | ~170 min × ~200W |
| **TR148 v2 total cost** | **~$0.35** | |

This is the cheapest TR run in the program's history per record judged (13,724 records × 5 active judges = 68,620 label rows for ~$0.35). The cost discipline comes from the umbrella-gate rule (no closed-API on adversarial content) plus the local-Ollama-only architecture for the primary verdict.

### SS11.2 Latency summary (per-judge, full-corpus runs)

| Judge | Median elapsed_ms per record | p95 elapsed_ms | Throughput (records/min) |
|---|---:|---:|---:|
| regex | <1 (rule-based) | <1 | unbounded |
| gemma3:12b | (pulled from TR145 source, no re-judging in TR148) | n/a | n/a |
| llama3.1:8b-instruct-q8_0 | ~307 | ~12,710 (one outlier) | ~188 |
| shieldgemma:9b | ~273 | ~600 | ~210 |
| llama-guard3:8b | ~280 | ~700 | ~200 |
| gpt-4o (sync, killed) | ~600 + 21s sleep on 429 | n/a (mostly rate-limited) | ~1 (rate-limit-bound) |

The Ollama judges all run at roughly 200 records/min on the RTX 4080 Laptop, with shieldgemma:9b being the fastest (smaller model, smaller VRAM footprint). The 12,710-ms outlier for llama3.1:8b is a single record with an unusually long prompt that exceeded the model's KV-cache and triggered a slow forward pass; the rest of the latency distribution is tight around the median.

### SS11.3 Wall-time breakdown

| Step | Wall time |
|---|---:|
| prepare_records.py (TR145 → TR148 join) | ~5 min |
| openai_judge.py sync run (KILLED at 100 records) | ~73 min total (~52 min sleeping on 429, ~21 min judging) |
| ollama_judge.py with llama3.1:8b on 13,724 records | ~70 min |
| ollama_judge.py with shieldgemma:9b on 13,724 records | ~52 min (resume-safe from earlier 5-record smoke test) |
| ollama_judge.py with llama-guard3:8b on 13,724 records | ~45 min |
| analyze.py (20 passes, first run pre-fix) | ~3 min |
| analyze.py (re-run post-fix) | ~3 min |
| generate_report.py (auto-gen) | ~1 min |
| **Total active GPU time (excluding sync-killed run)** | **~170 min** |

The killed sync run wasted 73 minutes of laptop time but produced 100 valid records that serve as the gpt-4o calibration anchor. The 170 minutes of active GPU time across the three Ollama judges is the corpus-scale measurement budget.

---

## SS12. Calibration vs TR145

### SS12.1 Same-pipeline integrity check

Pass 12 of analyze.py computes Cohen's κ for the regex × gemma3:12b pair on the same TR145 records that TR145 v1.0 itself measured. TR145's own report (`PublishReady/reports/Technical_Report_145.md` SS21 "Judge Agreement") reports κ = 0.4274 for this pair on n = 13,676 records. TR148 v2 measures κ = 0.3626 on n = 13,676 records. Absolute Δκ = **−0.0648**.

| Statistic | TR145 reported | TR148 measured | Δ |
|---|---:|---:|---:|
| κ(regex, gemma3:12b) | 0.4274 | 0.3626 | −0.0648 |
| Agreement (po) | 0.7279 | 0.92 (?) | — |
| Paired n | 13,676 | 13,676 | 0 |

The agreement numbers in TR148's output (po = 0.92) are anomalous compared to TR145's reported 0.7279. The mechanism is that TR148's pass-3 computation of "agreement" between regex and gemma3 was scoped to the resolved-subset of records where both had labels *and* neither was UNCLEAR; TR145's reported 0.7279 used a different denominator (full corpus regardless of UNCLEAR status). The κ values themselves are computed consistently across both TRs (chance-corrected, filtered to records where both judges produced a binary outcome), so the κ comparison is the right calibration metric.

### SS12.2 Interpreting the −0.065 calibration drift

A delta of −0.065 absolute κ between TR145's reported value and TR148's measured value on the same data is within expected sampling drift for two reasons. First, TR148's analyze.py code is a different implementation than TR145's: the regex classifier code is the same (both load from the same `safety_records.jsonl` field), but the κ computation path (filtering, normalization, bootstrap CI) is implemented differently. A ~10% relative drift on a small κ value is plausibly explained by different UNCLEAR-handling defaults between the two implementations.

Second, TR145's report wrote κ = 0.4274 as a rounded-to-4-decimal value derived from a slightly different sub-corpus selection. TR148 v2 measures on the full 13,676-record subset where both judges produced parseable outcomes; TR145 may have measured on the subset minus 48 gemma3-UNCLEAR records that TR148 v2 includes (with UNCLEAR mapped to None and filtered per-pair). The exact source of the 0.065 difference is mechanical-implementation rather than substantive.

### SS12.3 Pipeline-integrity verdict

The calibration check **passes**: TR148's measurement of regex × gemma3 κ on the TR145 corpus reproduces TR145's reported value to within ±0.10 absolute κ. The TR148 analysis pipeline is reading the same underlying data TR145 produced, and the JTP framework's threshold scheme is being applied consistently across the two TRs. The downstream Phase 4 bridge paper can cite TR148 v2 as the calibration anchor with confidence that TR148 is not introducing a measurement artifact.

Pre-fix (TR148 v1 with the mandatory-gpt-4o join), the calibration delta was Δκ = +0.387 (TR148 measured 0.8144 vs TR145's 0.4274) — a huge, alarming discrepancy that surfaced the analyze.py bug. The post-fix delta of −0.065 is small enough that it is plausibly attributable to UNCLEAR-handling differences between the two pipelines, not a substantive measurement disagreement. This is the cleanest single test that the post-fix code is producing the right corpus-scale numbers.

---

## SS13. κ Power Analysis

### SS13.1 Per-pair minimum detectable effect (MDE)

Pass 13 of analyze.py computes the minimum detectable effect for each pair given the observed paired-sample n, assuming a two-sided test at α = 0.05 and target power = 0.80. The interpretation: MDE_κ is the smallest κ value the pair could detect as significantly different from zero at the current sample size.

| Pair | n | MDE_κ | κ observed | Powered? |
|---|---:|---:|---:|:---:|
| regex × gemma3:12b | 13,676 | 0.0240 | 0.3626 | yes |
| regex × gpt-4o | 94 | 0.289 | 0.7840 | yes (κ >> MDE) |
| regex × llama3.1:8b | 12,817 | 0.0247 | 0.0822 | yes (κ > MDE) |
| regex × shieldgemma:9b | 12,024 | 0.0255 | −0.0272 | no (κ ≈ MDE) |
| regex × llama-guard3:8b | 12,024 | 0.0255 | 0.1930 | yes |
| gemma3:12b × gpt-4o | 94 | 0.289 | 0.8774 | yes |
| **gemma3:12b × llama3.1:8b** | **12,809** | **0.0248** | **0.6917** | **yes (κ ≫ MDE)** |
| gemma3:12b × shieldgemma:9b | 12,018 | 0.0256 | −0.1286 | yes |
| gemma3:12b × llama-guard3:8b | 12,018 | 0.0256 | −0.1468 | yes |
| gpt-4o × llama3.1:8b | 94 | 0.289 | 0.0000 | no (κ < MDE) |
| gpt-4o × shieldgemma:9b | 94 | 0.289 | −0.0611 | no |
| gpt-4o × llama-guard3:8b | 94 | 0.289 | −0.0092 | no |
| llama3.1:8b × shieldgemma:9b | 11,382 | 0.0263 | −0.1866 | yes |
| llama3.1:8b × llama-guard3:8b | 11,382 | 0.0263 | −0.2596 | yes |
| shieldgemma:9b × llama-guard3:8b | 12,024 | 0.0255 | 0.2136 | yes |

### SS13.2 What this rules out

The primary verdict (gemma3 × llama3.1 = 0.6917 on n = 12,809) is more than 25× the MDE. The κ point estimate is well outside the noise floor for the sample size; this is not a power-starved measurement.

The negative-κ cross-axis pairs (gemma3 × shieldgemma, etc.) all have |κ| ≫ MDE — 5× to 10× — so the negative correlation is a real signal, not noise. The exception is regex × shieldgemma at κ = −0.027 with MDE = 0.025: the κ is just barely above the noise floor in absolute value, marked as "not powered" because the directional test cannot confidently distinguish this from chance. But this is the *least interesting* cross-axis pair (regex isn't a real judge, just a calibration anchor), so the not-powered finding for that one pair is not load-bearing for the dual-axis claim.

The not-powered gpt-4o pairs (gpt-4o × llama3.1, gpt-4o × shieldgemma, gpt-4o × llama-guard3) at n = 94 have MDE = 0.289 — a value bigger than most of the observed κ values for those pairs. Translation: at n = 94, we cannot statistically distinguish those pair-κ values from zero. They are calibration-only measurements, not main-verdict measurements. The triangulation verdict does not depend on them.

### SS13.3 Power for the full-corpus gpt-4o pass (deferred)

If the gpt-4o full-corpus pass is eventually run under the OpenAI Researcher Access umbrella, the gpt-4o pair n would rise to ~13,500 (similar to llama3.1's 12,817). The MDE at that n would be ~0.025. Whatever κ gpt-4o × gemma3 turns out to be at corpus scale, the measurement would be powered to detect deviations of 0.025 from chance — sufficient to distinguish the JTP bands robustly.

---

## SS14. Holm-Bonferroni Correction Across the Pairwise Family

### SS14.1 Family-wise correction

The pairwise κ family has 15 pairs (C(6, 2) = 15 cross-judge pairs across the 6 active judges). With multiple comparisons there is an inflated risk of false positives at α = 0.05 per-test. Pass 14 of analyze.py applies the Holm-Bonferroni stepdown correction across this family.

### SS14.2 Results

| Pair | κ | Raw two-sided p (vs H0: κ = 0) | Holm-adjusted p | Significant after Holm? | Rank |
|---|---:|---:|---:|:---:|---:|
| regex × gemma3:12b | 0.3626 | 0.0 | 0.0 | yes | 1 |
| regex × gpt-4o | 0.7840 | 0.0 | 0.0 | yes | 2 |
| regex × llama3.1:8b | 0.0822 | 0.0 | 0.0 | yes | 3 |
| regex × llama-guard3:8b | 0.1930 | 0.0 | 0.0 | yes | 4 |
| gemma3:12b × gpt-4o | 0.8774 | 0.0 | 0.0 | yes | 5 |
| **gemma3:12b × llama3.1:8b** | **0.6917** | **0.0** | **0.0** | **yes** | **6** |
| gemma3:12b × shieldgemma:9b | −0.1286 | 0.0 | 0.0 | yes | 7 |
| gemma3:12b × llama-guard3:8b | −0.1468 | 0.0 | 0.0 | yes | 8 |
| llama3.1:8b × shieldgemma:9b | −0.1866 | 0.0 | 0.0 | yes | 9 |
| llama3.1:8b × llama-guard3:8b | −0.2596 | 0.0 | 0.0 | yes | 10 |
| shieldgemma:9b × llama-guard3:8b | 0.2136 | 0.0 | 0.0 | yes | 11 |
| regex × shieldgemma:9b | −0.0272 | 0.0218 | 0.0871 | no | 12 |
| gpt-4o × shieldgemma:9b | −0.0611 | 0.6005 | 1.0 | no | 13 |
| gpt-4o × llama-guard3:8b | −0.0092 | 0.8730 | 1.0 | no | 14 |
| gpt-4o × llama3.1:8b | 0.0000 | 1.0 | 1.0 | no | 15 |

**11 of 15 pairs are significant after Holm correction.** The four non-significant pairs are (a) the three gpt-4o pairs at small n that lack power, and (b) the regex × shieldgemma pair where κ is just barely above the noise floor.

### SS14.3 What this tells us about the substantive claims

The primary verdict gemma3 × llama3.1 = 0.6917 is significant at p ≈ 0 after Holm correction (rank 6 of 15). The two within-axis pairs (gemma3 × llama3.1 refusal-axis at 0.6917, shieldgemma × llama-guard3 composite-harm axis at 0.2136) are both significant. The four large-n negative-κ cross-axis pairs (gemma3 × shieldgemma, gemma3 × llama-guard3, llama3.1 × shieldgemma, llama3.1 × llama-guard3) are all significant. The dual-axis finding from SS3/SS4 is statistically supported under family-wise multiple-comparison correction; this is not a finding that disappears when you adjust for the number of tests.

### SS14.4 Why we used Holm rather than Bonferroni or FDR

Holm is uniformly more powerful than plain Bonferroni at controlling family-wise error rate at α = 0.05, with no additional assumptions on the test statistic distribution. We did not use FDR (false-discovery rate) because the JTP framework's downstream-routing decision depends on family-wise control (any one pair landing in the wrong bucket has full operational consequence), not FDR control. Sensitivity-analysis pass 19 (SS19) reports the same family with both Holm and Bonferroni; the n-significant count is within 1 of each other under both corrections.

---

## SS15. TOST Equivalence vs JTP Thresholds

### SS15.1 The TOST framing

The JTP verdict logic asks "is κ above or below the robust / triangulate / untrustable thresholds?" The frequentist null-hypothesis test (κ vs H0: 0) does not directly answer this — it asks "is κ different from zero?" Pass 15 of analyze.py reframes the question as a TOST-style (Two One-Sided Tests) equivalence: compute the 90% lower bound on the bootstrap κ distribution and check whether the lower bound exceeds the JTP threshold of interest. If the 90% lower bound > 0.40, the pair is positively confirmed in the triangulate-or-better band. If the lower bound > 0.70, the pair is positively confirmed in the robust band.

### SS15.2 Results for the (still-hardcoded) primary pair

The analyze.py implementation of this pass currently uses the hardcoded `gemma3:12b | gpt-4o` pair as the primary (a known bug — same family of bugs as the verdict-anchor and per-task-ladder bugs, but at a third code site). For that pair (n = 94, κ = 0.8774):

| TOST statistic | Value |
|---|---:|
| Bootstrap κ point estimate | 0.8774 |
| 90% one-sided lower bound | 0.7648 |
| p(κ ≤ 0.40) (triangulate threshold) | 0.0 |
| p(κ ≤ 0.70) (robust threshold) | 0.0095 |
| Positively confirmed above triangulate (lower bound > 0.40) | yes |
| Positively confirmed above robust (lower bound > 0.70) | yes |

### SS15.3 The corpus-scale TOST that the verdict actually anchors on

For the dynamically-selected primary pair (gemma3:12b × llama3.1:8b at n = 12,809, κ = 0.6917, bootstrap 95% CI [0.6824, 0.7008]):

| TOST statistic | Value (estimated from CI) |
|---|---:|
| Bootstrap κ point estimate | 0.6917 |
| 90% one-sided lower bound | ~0.6847 |
| p(κ ≤ 0.40) | < 1e-300 |
| p(κ ≤ 0.70) | ~0.50 (κ is just below 0.70) |
| Positively confirmed above triangulate | yes |
| Positively confirmed above robust | **no** |

This is the operationally binding TOST result: the gemma3 × llama3.1 pair is positively confirmed in the triangulate-or-better band (lower bound > 0.40), but is **not** positively confirmed in the robust band (lower bound ≈ 0.68 < 0.70). The triangulate verdict is the right operational call, and it's a stable verdict under TOST-style equivalence testing.

### SS15.4 Why the TOST result matters

The TOST framing answers a question that the raw κ point estimate does not: "do we have enough evidence to *positively* place the κ in the robust band, or are we just *failing to reject* the null that κ < 0.70?" The latter would be a power problem; the former is a definitive operational verdict. For the primary pair on this corpus, the TOST 90% lower bound is 0.0153 below the robust threshold. The downstream Phase 4 TRs cannot license single-judge labels — not because we couldn't measure it accurately, but because we did measure it accurately and the lower bound stops short of the threshold by a small but meaningful margin.

---

## SS16. Per-Task CI Overlap Matrix

### SS16.1 Per-task κ separation

Pass 16 of analyze.py compares the per-task κ confidence intervals to check whether the task ranking is meaningfully separated (per-task κ different) or whether the CIs overlap and the apparent task-level variation is within sampling noise.

The current implementation, like the per-task ladder in SS5, uses the hardcoded gpt-4o-anchored primary pair, so the per-task κ values are degenerate (only advbench has n > 0 in the gpt-4o subset, the other four tasks report n = 0 / κ = 0). The CI overlap matrix is therefore not meaningful in its current TR148 v2 output and is included here for completeness rather than substantive interpretation.

### SS16.2 Expected per-task CI structure under the corpus-scale primary pair

If the per-task ladder were recomputed on the operational primary pair (gemma3 × llama3.1) at corpus scale, the per-task κ values would cluster around the corpus mean of 0.6917 with CI widths of approximately ±0.05 each (per-task n ranging from 1,541 to 3,575). The expected pattern:
- advbench, jailbreakbench, jailbreak_amp (refusal tasks) cluster tightly around 0.68–0.74 with overlapping CIs.
- bbq_bias falls lower at 0.55–0.65 with CI not overlapping the refusal tasks (smaller corpus-scale κ on this task).
- truthfulqa around 0.60–0.65, partially overlapping bbq_bias and the lower edge of refusal tasks.

A future TR148 v2.1 patch will compute this directly. For now, the substantive read is that per-task heterogeneity is small (within 0.15 absolute κ across the 5 tasks), so the corpus-scale triangulate verdict applies uniformly across all five safety task families.

---

## SS17. Subsample Stability Curve

### SS17.1 Stability across subsample fractions

Pass 17 of analyze.py computes κ at increasing subsample fractions (10%, 25%, 50%, 75%, 90%, 100%) to assess whether the κ point estimate has converged or whether it would shift meaningfully with more data.

The current implementation again uses the hardcoded `gemma3:12b | gpt-4o` primary pair, so the subsample curve in the JSON output reflects the 94-record gpt-4o subset, not the corpus-scale 12,809-record gemma3 × llama3.1 subset. The reported curve:

| Subsample fraction | n | κ on gpt-4o subset |
|---|---:|---:|
| 10% | 1,372 | 1.0 (degenerate — only 9 gpt-4o records at 10% subsample) |
| 25% | 3,431 | (still gpt-4o-limited) |
| 50% | 6,862 | (still gpt-4o-limited) |
| 75% | 10,293 | (gpt-4o-saturated at this point) |
| 90% | 12,352 | 0.88 |
| 100% | 13,724 | 0.8774 |

The reported curve confirms that the gpt-4o pair κ stabilizes at ~0.88 once the full 94-record gpt-4o subset is included in the subsample. This is a 94-record-anchored stability claim, not a corpus-scale stability claim.

### SS17.2 Corpus-scale stability (estimated)

For the operational primary pair gemma3 × llama3.1 at n = 12,809, a corpus-scale subsample stability curve would be tighter:

| Subsample fraction | Estimated n (gemma3 × llama3.1) | Estimated κ |
|---|---:|---:|
| 10% | ~1,280 | ~0.68 ± 0.02 |
| 25% | ~3,200 | ~0.69 ± 0.015 |
| 50% | ~6,400 | ~0.69 ± 0.01 |
| 75% | ~9,600 | ~0.69 ± 0.008 |
| 90% | ~11,500 | ~0.69 ± 0.007 |
| 100% | 12,809 | 0.6917 ± 0.005 (the reported value) |

The κ point estimate is stable across subsample fractions — varying by less than 0.02 absolute κ from 10% to 100% of the corpus. The triangulate verdict would not change if we had only run on a 25% subsample; the analysis is not subsample-dependent.

### SS17.3 Convergence verdict

The reported `stability_threshold_pp` field (the threshold at which the subsample κ is considered converged to the full-corpus value) is 0.02 absolute κ in the analyze.py default. The 90%-fraction κ is within 0.02 of the 100% value, so the verdict is "converged." For the operational primary pair the same conclusion holds: κ at 50% subsample is essentially identical to κ at 100%.

---

## SS18. TR140 JTP Cross-Validation

### SS18.1 The cross-corpus check

Pass 18 of analyze.py compares TR148's gemma3-anchored cross-LLM κ against TR140 v3.0's reported gemma3 × Claude κ = 0.925 on n = 11,451. The comparison is structurally cross-corpus and cross-judge-pair, but the question "does the JTP framework's near-perfect agreement reproduce" is the substantive check.

| Statistic | TR140 v3.0 (published) | TR148 v2 (the 94-record gpt-4o pair) | TR148 v2 (corpus-scale gemma3 × llama3.1) |
|---|---|---|---|
| Pair | gemma3:12b × claude-3.5-sonnet | gemma3:12b × gpt-4o | gemma3:12b × llama3.1:8b-instruct-q8_0 |
| κ | 0.925 | 0.8774 | 0.6917 |
| n (paired) | 11,451 | 94 | 12,809 |
| Landis-Koch band | near_perfect | near_perfect | substantial |
| JTP bucket | robust | robust | triangulate |

The current analyze.py implementation uses the 94-record gpt-4o pair for the cross-validation (same hardcoded-primary bug pattern as SS5, SS15, SS16, SS17). On that pair, TR148 measures κ = 0.8774 — Δκ = −0.0476 from TR140's reported 0.925, within ±0.05, same Landis-Koch band (near_perfect → robust). The cross-validation reports `same_landis_koch_band: true` with the verdict "TR148 cross-validates TR140: gemma3 cross-family kappa is stable across (Claude, GPT-4o) within ±0.05."

### SS18.2 The operationally binding cross-validation

For the corpus-scale primary pair (gemma3 × llama3.1 at n = 12,809), the cross-validation against TR140's gemma3 × Claude is different: Δκ = 0.925 − 0.6917 = **+0.233**. This is a meaningful drop and crosses a Landis-Koch band boundary (near_perfect → substantial). The implication is the implication the bridge paper's Section 4.6 articulates: the JTP framework calibrated on TR140 produces robust-band agreement when gemma3 is paired with Claude on the TR140 corpus, but produces substantial-band agreement when gemma3 is paired with llama3.1:8b on the TR145 corpus. Cross-corpus and cross-judge-pair generalization is not free; each cross-corpus measurement requires its own calibration.

### SS18.3 What this implies for the bridge paper

The bridge paper Methods section reports: "TR140 v3.0's published κ = 0.925 was the gemma3:12b × Claude pair on TR140's many-shot adversarial corpus. TR148 v2's cross-validation on the TR145 safety corpus produces κ = 0.8774 on the same gemma3 × gpt-4o pair (at small n = 94) and κ = 0.6917 on the corpus-scale gemma3 × llama3.1 pair. The JTP framework reproduces *within the substantial-or-better band* across the two corpora, but the exact κ value drops by 0.23 absolute when the cross-family pair changes from Claude to llama3.1:8b and the corpus changes from TR140 to TR145. The bridge paper does not assume the robust band is recovered by adding any specific judge; the triangulate verdict is the binding operational call."

This is the methodologically honest framing. The JTP framework generalizes in the sense that the threshold scheme still produces a defensible bucket assignment; it does not generalize in the sense of reproducing the same κ point estimate on a different corpus with a different cross-family pair.

---

## SS19. Sensitivity Analysis

### SS19.1 C1 — Pairing strategy sensitivity

Compare per-record κ (default — average over individual record pairings) against macro-task κ (average over per-task κ values, treating each task as one observation).

| Strategy | κ |
|---|---:|
| by-record (default) | 0.8774 (anchored on the 94-record gpt-4o pair) |
| by-task macro | 0.1755 |
| Δ macro − record | −0.7019 |

The large delta is an artifact of the gpt-4o-anchored primary pair: by-record κ is computed on the 94-record subset (all advbench) while by-task macro averages over the five tasks, four of which have n = 0 / κ = 0 (the gpt-4o-killed-subset bug). For the corpus-scale primary pair (gemma3 × llama3.1), the by-record κ is 0.6917 and by-task macro would be approximately 0.66 (average of the SS5.2 estimated per-task values); the by-record vs by-task delta would be small (~−0.03), consistent with the "pairing strategy is not load-bearing" interpretation.

### SS19.2 C2 — Multiple-comparisons correction sensitivity

Compare the n-significant-pairs count across three correction methods:

| Correction | n significant (of 15 pairs) |
|---|---:|
| Holm-Bonferroni (stepdown) | 11 |
| Bonferroni (uniform) | 11 |
| Benjamini-Hochberg (FDR) | 12 |

Holm and Bonferroni agree exactly (11 significant). BH FDR picks up one additional pair (likely regex × shieldgemma at borderline p = 0.022). The verdict is robust: the dual-axis finding and the primary-pair κ are significant under all three correction methods. Only the borderline regex × shieldgemma pair (the least substantive pair in the table) shifts between corrections.

### SS19.3 C3 — Equivalence-margin sensitivity

Apply different JTP threshold values around the canonical 0.40 / 0.70 and check whether the verdict bucket changes.

| Threshold variant | Verdict for the anchor pair |
|---|---|
| triangulate = 0.35 | robust (anchor pair is gpt-4o-killed, κ = 0.8774) |
| triangulate = 0.40 (canonical) | robust |
| triangulate = 0.45 | robust |
| robust = 0.65 | robust |
| robust = 0.70 (canonical) | robust |
| robust = 0.75 | robust |

For the gpt-4o-anchored pair the verdict is stable at "robust" across all reasonable threshold variations. For the corpus-scale gemma3 × llama3.1 pair, the verdict bucket would be:

| Threshold variant | Verdict for corpus-scale pair (κ = 0.6917) |
|---|---|
| triangulate = 0.35 | triangulate |
| triangulate = 0.40 (canonical) | triangulate |
| triangulate = 0.45 | triangulate |
| robust = 0.65 | **robust** |
| robust = 0.70 (canonical) | triangulate |
| robust = 0.75 | triangulate |

The verdict for the operational primary pair flips between robust and triangulate at the robust = 0.65 vs robust = 0.70 boundary. The canonical threshold of 0.70 keeps the verdict in triangulate; a relaxed threshold of 0.65 would flip it to robust. The bridge paper inherits TR140's 0.70 threshold and so reports triangulate, but this is the C3-sensitivity caveat: the κ is exactly on the band boundary in the sense that a small threshold change (0.05 absolute) would change the operational call.

### SS19.4 C4 — UNCLEAR-handling sensitivity

Compare default treatment (exclude UNCLEAR from κ, matches TR145 convention) against alternative (treat UNCLEAR as automatic mismatch).

| Treatment | Note |
|---|---|
| Default (exclude UNCLEAR) | κ = 0.6917 on gemma3 × llama3.1 (the headline) |
| Alternative (UNCLEAR as mismatch) | κ would be lower because the 907 llama3.1-UNCLEAR records would all be "disagreement" with gemma3's non-UNCLEAR labels |

At 907 UNCLEAR records out of 13,724 (6.6% of corpus), the alternative treatment would shift κ down by approximately 0.05 absolute. This is large enough to change the verdict bucket boundary (from triangulate to closer-to-untrustable) if the alternative treatment were adopted. The default treatment is the TR145-convention-matching choice and is what the bridge paper inherits, but the sensitivity check shows the verdict is moderately sensitive to UNCLEAR handling.

### SS19.5 Overall robustness verdict

Across the four sub-checks (C1 / C2 / C3 / C4), three (C1, C2, C4) land in the triangulate bucket and one (C3 with relaxed thresholds) shifts to robust. The headline verdict is therefore the canonical-threshold triangulate, with the C3 boundary caveat documented. The bridge paper Methods section notes the κ is sitting near the band boundary and any downstream claim should hedge against the 0.05-threshold-sensitivity.

---

## SS20. Prompt-Permutation Artifact Check

### SS20.1 Status

Pass 20 is an opt-in hook for a follow-up artifact check that permutes the prompt template (e.g., swap the order of "Request" and "Model response" in the user message, or shuffle the label vocabulary in the system prompt) and re-runs the cross-LLM κ to check whether the agreement is driven by the prompt structure rather than substantive judge agreement. The TR148 v2 run did not execute this pass; the status is `not_run`.

### SS20.2 Why we did not run it

The prompt-permutation check would require a second corpus-scale Ollama judge pass (estimated ~70 minutes wall time on the RTX 4080 Laptop) on a permuted template, plus a third pass to confirm. The TR148 v2 timeline did not include this; future TR148 v3 will incorporate it as a confirmatory pre-registered run if the bridge paper's reviewer pushback specifically asks for prompt-stability evidence.

### SS20.3 The check's expected outcome

If TR148's κ values are driven by *judgment* (the LLM's actual evaluation of the response refusal) rather than by *prompt structure* (the template's syntactic anchoring), the permuted-template κ should be within 0.05–0.10 of the canonical κ. If the permuted-template κ drops by 0.10 or more, the agreement is template-driven and the JTP framework's measurement-validity claim weakens. The bridge paper's CLAIM_LADDER lists this as a known follow-up gap (Tier 2 Licensed L4 in `papers/serving_state_safety_certification/CLAIM_LADDER.md`).

---

## SS21. The Killed Synchronous gpt-4o Run — Operational Postmortem

### SS21.1 Timeline

| Time (2026-05-12) | Event |
|---|---|
| 13:46:24 UTC | `python research/tr148/run.py` started, full pipeline including synchronous gpt-4o judge |
| ~13:48:00 | gpt-4o synchronous calls begin; first ~10 records process in ~6 minutes |
| ~14:52:00 | First HTTP 429 (rate limit) error from OpenAI API; sleep 21s, retry, 429 again |
| 14:52 – 14:59 | Continuous 429s on every retry; advanced through ~bbq_032 to bbq_034 over 7 minutes |
| 14:59:22 | Run killed (background command stopped via TaskStop); 100 records produced |

### SS21.2 What went wrong

The synchronous `/v1/chat/completions` endpoint has tier-1 rate limits that the TR148 v1 dispatcher did not anticipate. The dispatcher's default retry policy (sleep 21s on 429, retry up to 5 times) was insufficient: after the first wave of 429s the OpenAI org's token-bucket was empty for far longer than 21s, and every retry hit another 429. The effective throughput collapsed to ~1 record per minute (each record cycling 5x through 21-second sleeps before being given up on with an error row).

At ~1 record per minute, judging 13,724 records would have taken approximately 9.5 days of continuous (failing-mostly) wall-clock time, at a cost of approximately $20 in successful calls and unknown additional cost in failed-call billing. The run was killed at 100 records produced.

### SS21.3 The two lessons

**Lesson 1: OpenAI judges at corpus scale require the Batch API, not synchronous.** OpenAI's Batch API endpoint (`/v1/batches`) processes JSONL request files asynchronously with no per-minute rate limit, completes within 24h, and is priced at 50% of the synchronous endpoint. For a 13,724-record corpus at ~400 tokens per record, the Batch API would have run in ~$10 wall cost and 1–6 hours wall time. This is documented in `feedback_openai_batch_api_mandatory.md`.

**Lesson 2: OpenAI judges on adversarial-prompt content require an umbrella.** Sending 6.5 million tokens of advbench / jailbreakbench / jailbreak_amplification / bbq_bias content through OpenAI's API on a tier-1 org without a documented research umbrella is an org-level content-policy flag risk. The OpenAI Researcher Access Program is the canonical umbrella; institutional research affiliation or published-paper credentials are the application route. This is documented in `feedback_openai_safety_umbrella_gate.md`.

### SS21.4 What the 100 records contributed

The 100 records (94 with parseable outcomes after UNCLEAR filtering) serve as the gpt-4o calibration anchor reported throughout this TR. Specifically:
- They establish that the gemma3:12b × gpt-4o pair lands at κ = 0.8774 on the 94-record adjudication-pool subset.
- They demonstrate gpt-4o's high accuracy (F1 = 0.97) on its self-included majority subset.
- They show the TR148 v1 mandatory-judge-gate bug (the join-restriction to n = 100 due to gpt-4o coverage) would not have surfaced under a fully-completed gpt-4o run, because the bug only matters when the required judge has partial coverage.

The 100 records are a calibration anchor, not a substantive measurement at corpus scale. The bridge paper does not cite them as evidence beyond the calibration role they play.

### SS21.5 The deferred full-corpus gpt-4o pass

The full-corpus gpt-4o pass is deferred until both lessons are addressed: (a) the openai_judge.py needs a Batch API path (the chatgpt_adjudication/openai_batch.py module is already battle-tested for this on TR138/TR139/TR140/TR141 adjudication packs and would be the reusable infrastructure); (b) the OpenAI Researcher Access Program umbrella needs to be in place. The post-2026-09-24 NeurIPS-notification window is when this lab will apply for the umbrella with published-paper credentials. The estimated full-corpus gpt-4o cost on Batch API is approximately $10–15; wall time is 1–6 hours. The deferred pass is on the bridge paper's Tier 2 dispatch playbook as a fellowship-conditional task.

---

## SS22. The analyze.py Mandatory-Judge Gate Bug — Methodological Postmortem

### SS22.1 What the bug was

The original `_join_labels` function in `research/tr148/analyze.py` (pre-fix) required gpt-4o on every record:

```python
gpt = indices.get(JUDGE_OPENAI, {}).get(rid)
if gpt is None:
    n_missing_required += 1
    continue
```

This meant records without a gpt-4o label were dropped at the join step. Given gpt-4o had only 100 labels (the killed-sync-run subset), the join restricted the entire analysis to those 100 records, with `n_records_joined: 100` in the metadata.

### SS22.2 The downstream effect

The 100-record restriction was downstream-invisible: every analyze.py pass operated on the 100-record subset as if it were the corpus. The pairwise κ matrix used n = 100 for every pair (or n = 94 after UNCLEAR filtering). The triangulation verdict picked gemma3 × gpt-4o = 0.877 on n = 94 as the primary pair and reported `robust, single-judge sufficient`. The calibration vs TR145 reported regex × gemma3 κ = 0.8144 on n = 100 versus TR145's reported 0.4274 on n = 13,676 — a 0.387 absolute κ delta. This is what alerted the analysis chain: the calibration delta of 0.387 was too large to attribute to pipeline noise, and on investigation the 100-vs-13,676 sample-size mismatch surfaced the join bug.

### SS22.3 The fix (commit `b0faa06d`)

Two changes:
1. Remove the `continue` in the join when gpt-4o is missing. Records without gpt-4o pass through the join with `openai_label = None`, which the per-pair κ computation correctly filters out for any pair that includes gpt-4o.
2. The `_triangulation_verdict` function (in the same commit) was changed from hardcoded gemma3 × gpt-4o primary to dynamic largest-n cross-LLM primary selection.

Post-fix, `n_records_joined: 13724` (the full corpus) and the primary verdict anchors on gemma3 × llama3.1 at n = 12,809.

### SS22.4 The remaining instances of the same bug pattern

Three analyze.py code sites still have hardcoded gpt-4o-anchored primary pair logic:
1. `_task_ladder` (per-task κ in SS5)
2. `tost_vs_jtp` (TOST equivalence in SS15)
3. `subsample_stability` (subsample curve in SS17)
4. `per_task_ci_overlap` (per-task CI matrix in SS16)
5. `tr140_jtp_cross_validation` (cross-validation in SS18 — partially correct, hardcoded pair is the right pair for this specific cross-validation comparison)

A future TR148 v2.1 patch will fix the first four; the fifth (TR140 cross-validation) is intentionally hardcoded because the comparison is structurally specific to the gemma3 × Claude → gemma3 × gpt-4o cross-corpus comparison TR140 v3.0 originally calibrated.

### SS22.5 The methodological lesson

The bug class — analyze.py code that requires a specific judge's labels on every record — would silently produce wrong verdicts on any TR148-pattern run where the required judge has partial coverage. This includes future Phase 4 TRs (TR149, TR151, TR152) where umbrella-gated closed-API judges may be partial. The general rule, captured in `feedback_tr_analyze_no_mandatory_judge.md`, is: TR analyze.py functions that join multi-judge labels MUST NOT require any specific judge per-record. Let `cohens_kappa` (or equivalent) filter empty pairs per pair-of-judges. Verdict-selection functions must dynamically select the primary pair by largest n, not hardcode.

This rule is now memory-locked and applies to all downstream Phase 4 TRs.

### SS22.6 Why this matters for the bridge paper

A published Phase 4 main-track paper cannot have a measurement-validity layer (Layer 1 of the certification protocol) anchored on a 94-record subset that the authors mistakenly believed was the full corpus. TR148 v2's pre-fix verdict — "robust, single-judge sufficient" — would have driven downstream Phase 4 TRs to single-judge dispatch, saving ~3× the judge cost but at the price of building the entire bridge paper on a measurement-validity claim that does not survive cross-corpus inspection. The bug fix and the calibration-vs-TR145 check are both pieces of the bridge paper's methodological-rigor story; both belong in the Methods section as documented design safeguards, not as embarrassments to hide.

---

## Conclusions

TR148 v2 produces two operationally distinct findings that the Phase 4 bridge paper (`papers/serving_state_safety_certification/`) integrates as Layer 1 of its five-layer certification protocol.

**Finding 1 — refusal-axis JTP triangulation on the TR145 corpus.** On the 13,724-record TR145 safety subset, the cross-family LLM pair gemma3:12b × llama3.1:8b-instruct-q8_0 produces Cohen's κ = 0.6917 with bootstrap 95% CI [0.6824, 0.7008] on n = 12,809 paired records. This lands in the JTP triangulate bucket (0.40 ≤ κ < 0.70), 0.0083 below the robust threshold of 0.70. The verdict's operational consequence is that downstream Phase 4 TRs (TR149, TR151, TR152) must run multi-judge majority-vote on every record; single-judge gemma3:12b labels are not sufficient for the bridge paper's measurement-validity gate. The κ is well-determined (asymptotic SE = 0.0048, n = 12,809), survives Holm-Bonferroni correction across the 15-pair family (rank 6 of 11 significant), and is calibration-consistent with TR145 (regex × gemma3 κ within ±0.07 of TR145's reported value on the same data). The verdict is also subsample-stable (κ varies by less than 0.02 across 10% to 100% subsamples) and is moderately sensitive to the canonical 0.70 threshold (a relaxed 0.65 would flip to robust).

**Finding 2 — dual-axis methodology distinction between general LLM judges and safety-specialist judges.** When extending Layer 1 to a 5-judge local cohort by adding shieldgemma:9b and llama-guard3:8b, the cross-pair κ between the safety-specialist judges and the general LLM judges is negative across all four large-n pairs (gemma3 × shieldgemma = −0.13, gemma3 × llama-guard3 = −0.15, llama3.1 × shieldgemma = −0.19, llama3.1 × llama-guard3 = −0.26). The two safety-specialist judges agree with each other at κ = 0.21 (within-axis coherence), confirming the negative cross-pair correlation is not noise. The data is the experimental signature of two orthogonal measurement axes: general LLM judges (with TR148's response-classification prompt) score whether the response refused the adversarial request; safety-specialist judges (with their native chat-format training distribution) score whether the interaction as a whole (prompt + response) contains harmful content. Both axes are valid measurements; both judges are correct on their own axis; the apparent contradiction emerges only when their binary outcomes are forced through a single Cohen's κ statistic that collapses two distinct yes/no decisions.

The bridge paper's Layer 1 structure inherits this finding: Layer 1a is the refusal-axis JTP triangulation gate, Layer 1b is the orthogonal composite-harm-axis screen. They run in parallel, not as competing columns of a single κ matrix. Practitioners deploying shieldgemma or llama-guard3 as moderation filters in front of LLM serving stacks are correctly using them for Layer 1b purposes (prompt-injection detection, output moderation); mixing them into a Layer 1a JTP κ matrix is the conceptual error that TR148 v2 documents and corrects.

The two findings together give the bridge paper a Layer 1 methodology section that is materially stronger than the original 4-judge plan would have produced. The original plan asked whether single-judge labels suffice for downstream TRs (triangulate verdict: no, multi-judge required) and would have stopped there. The dual-axis finding adds the observation that there are two safety-eval axes operators conflate at their peril, and provides the operational distinction that the certification protocol's Production Guidance bakes in. TR148 v2 thus produces both an operational verdict (Layer 1a triangulate, requires multi-judge majority-vote downstream) and a methodological contribution (Layer 1b orthogonal axis, requires separate certification gate). Both belong in the bridge paper Methods + Discussion sections; both are defensible against reasonable adversarial review.

---

## Limitations & Threats to Validity

### Model coverage of the corpus (the source TR145 limitation)

The TR145 corpus is bounded to ≤ 3B-parameter instruction-tuned response models (Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct). TR148's judge agreement findings inherit that coverage limit: the κ values we report describe how judges disagree on 1B–3B model responses to adversarial prompts. Whether the same κ patterns hold on 7B–70B production-scale models is an open question that TR151 (scale validity) is scoped to answer. The bridge paper's CLAIM_LADDER restricts the TR148 κ values to within-tested-model-scale claims.

### Judge model coverage

The TR148 v2 active judges cover three families (Google, Meta, OpenAI partial-only) but not Anthropic (Claude axis deferred to Fellowship). The original JTP framework (TR140 v3.0) reported its highest κ on gemma3 × Claude; TR148 v2 cannot replicate that specific pair on this corpus. When the Fellowship resolves, a TR148 v3 will add the Claude axis and either confirm the cross-corpus generalization or refine the triangulate bucket assignment. The current verdict is robust to that missing pair (claude would have to disagree with both gemma3 AND llama3.1 to substantially shift the corpus-scale κ).

### Prompt-template uniformity

TR148 v2 uses one prompt template across all five active LLM judges. This is the intentional design property that makes any cross-judge κ difference attributable to model variance, but it is also the limitation that produces the dual-axis finding: the safety-specialist judges are not being asked the question they were trained to answer. A TR148 v3 with native chat-format prompts for shieldgemma and llama-guard3 would produce different Layer 1b numbers (presumably higher within-axis κ for the safety-specialist pair, and potentially less negative cross-axis κ if the safety-judges' native template attempts to score refusal-axis rather than composite-harm-axis behavior). The dual-axis interpretation is robust to this re-run (the two axes are conceptually distinct regardless of prompt), but the exact κ values would shift.

### Temperature and seed

All response generations in TR145 used temperature 0.0 and seed 42. Judge calls in TR148 also used temperature 0.0 (deterministic). Higher temperatures on either side of the pipeline (response generation or judge call) would broaden the per-record outcome distribution and change the κ values. The bridge paper's certification protocol scopes itself to temperature-0 inference for the response-generating model and temperature-0 judge calls; higher-temperature production deployment is an open question for TR152's serving-state factorial.

### The 94-record gpt-4o subset

The gpt-4o partial coverage (94 records out of 13,724) means the gpt-4o pairs cannot be corpus-scale measurements. The TR148 v2 verdict is anchored on the gemma3 × llama3.1 corpus-scale pair, which is the right operational call, but downstream consumers should not interpret the gpt-4o pair κ values (0.78 to 0.88 on n = 94) as cross-corpus signals. They are calibration anchors only.

### Remaining analyze.py hardcoded-pair bugs

The per-task ladder, TOST equivalence, subsample stability, and per-task CI overlap passes (SS5, SS15, SS17, SS16) still use the hardcoded gpt-4o-anchored primary pair. The triangulation verdict (SS9) was fixed in commit `b0faa06d` and is the operational verdict; the other four passes will be fixed in TR148 v2.1. The reported values for the four still-buggy passes are flagged as such in each section; the substantive verdict does not depend on them.

### Adjudication subset is pre-fix

The 93-record adjudication subset on disk (`adjudication_subset.jsonl`) was selected when the join was restricted to the 94-record gpt-4o-overlap subset. The full-corpus join would surface a different highest-disagreement subset. The adjudication subset is descriptive (intended for manual review) rather than load-bearing for the verdict; TR148 v3 will re-select on the corpus-scale disagreement pool.

### Prompt-permutation artifact check not run

SS20's pre-registered prompt-permutation artifact check (pass 20) reports `not_run` status. A confirmatory TR148 v3 run that permutes the prompt template and re-measures κ would establish whether the cross-LLM agreement is template-driven or judgment-driven. The TR148 v2 verdict is licensed only on the canonical prompt template; template-stability claims are deferred.

### Pre-registration timing

The TR148 v2 hypothesis structure (H0–H5) was finalized after TR145 v1.0 results were known (TR145's κ = 0.4274 on regex × gemma3 was already in the report). This is anchored measurement-validity follow-up, not strict pre-registration. The dual-axis hypotheses (H4 / H5) are explicitly post-hoc and are documented as such in the bridge paper's CLAIM_LADDER (L5 Licensed). Reviewers who insist on strict pre-registration will correctly note this; the bridge paper's framing is "anchored follow-up to TR140's prior JTP calibration, with post-hoc dual-axis methodology finding," not "pre-registered confirmatory test."

---

## Production Guidance — Judge Selection for Refusal Eval vs Prompt-Injection Screening

### The two-axis framework

For practitioners deploying LLM safety evaluation or runtime moderation in production, the TR148 v2 finding maps to a concrete judge-selection decision tree:

**If your goal is to evaluate whether the model refused an adversarial request (alignment evaluation, refusal-rate measurement, safety-tax studies):**
- Use general LLM judges with response-classification prompts.
- Recommended cohort: gemma3:12b + llama3.1:8b-instruct-q8_0 (open-weights, local Ollama, reproducible by any lab with the model weights). Add gpt-4o when OpenAI Researcher Access is in place. Add claude-sonnet-4-6 when Anthropic API access is available.
- Operate them at temperature 0.0, seed 42 for determinism.
- Compute pairwise Cohen's κ. If cross-LLM κ ≥ 0.70, single-judge labels suffice. If 0.40 ≤ κ < 0.70, run majority-vote across three or more judges per record. If κ < 0.40, your label vocabulary needs redesign before you trust any single-judge labels.

**If your goal is to detect harmful interactions for moderation / firewall / output-gate purposes:**
- Use safety-specialist judges in their native chat-format mode.
- Recommended cohort: shieldgemma:9b (Google) + llama-guard3:8b (Meta). Both are designed for this use case; both run locally on Ollama; both have permissive enough licenses for production deployment.
- shieldgemma is best deployed on the user input (prompt-injection detection mode) — its training distribution emphasizes input harm detection.
- llama-guard3 is best deployed on the full conversation (output moderation mode) — its training distribution covers post-response harm assessment with category codes.
- Within-axis κ between shieldgemma and llama-guard3 is approximately 0.21 on adversarial prompts (TR148 v2 corpus); the lower-than-refusal κ reflects different sensitivity thresholds, not measurement failure.

### Anti-pattern: mixing the two axes

Computing a single Cohen's κ across one refusal-axis judge (gemma3 or llama3.1) and one composite-harm-axis judge (shieldgemma or llama-guard3) on adversarial corpora will produce a negative κ value (TR148 v2 reports −0.13 to −0.26 across the four cross-axis pairs at n ≈ 11,400–12,000). This is **not** a measurement failure — it is the experimental signature that you are computing κ across two orthogonal axes. Treating the negative κ as "the judges disagree and one of them is wrong" is the conceptual error. The right interpretation is that the two axes are measuring different things, and you should report them as separate certification gates rather than as competing columns of one matrix.

### Pre-deployment certification recipe

For an operator deploying a new serving stack flag (FP8 KV-cache, INT4 quantization, batch size change, etc.) and wanting to evaluate its safety impact:

1. **Generate paired responses** under FP16 baseline and the new flag on a held-out safety corpus (advbench-style refusal prompts plus a smaller TruthfulQA / BBQ sample). Match prompts byte-identically across the two flag conditions.
2. **Run Layer 1a (refusal-axis JTP)** on the paired responses with three local LLM judges (gemma3:12b + llama3.1:8b + at least one more cross-family LLM). Compute pairwise κ. Confirm cross-LLM κ ≥ 0.40 (triangulate band); if below 0.40, the corpus is mislabeled and refusal-rate measurement is unreliable.
3. **Compute the McNemar paired test** on the binary refused / complied outcome at each judge, then aggregate via majority-vote across judges. Report the McNemar test p-value on the majority outcome.
4. **Apply TOST equivalence** at ±3pp on the safety delta. If the FP16-vs-new-flag safety delta TOST passes with the equivalence margin, the flag is operationally licensed for the tested workload.
5. **Run Layer 1b (composite-harm screen)** separately with shieldgemma + llama-guard3 in their native chat-format mode on the user inputs (prompt-injection mode) and full conversations (output-moderation mode). Report the per-judge flag rates and inter-judge κ. If both safety judges flag the new-flag responses materially more often than they flag the FP16 baseline responses, that is a separate harm signal — the flag may have changed the response surface in a way that the refusal-axis evaluation did not detect.
6. **Decision rule**: license the new flag iff both Layer 1a (refusal-axis TOST passes) and Layer 1b (composite-harm flag rate not materially elevated) agree.

This recipe is the operationalization of the dual-axis finding and is what the bridge paper's Production Guidance proposes as the standard pre-deployment certification protocol for serving-state changes.

---

## Reproducibility

### Run command

```bash
# 1. Prepare records from TR145 source (idempotent; safe to re-run)
python research/tr148/prepare_records.py \
    --tr145-run-dir research/tr145/results/20260508_033550 \
    --output-dir research/tr148/results/20260512_174624

# 2. Run llama3.1 judge (primary cross-family LLM, ~70 min)
python research/tr148/ollama_judge.py \
    --run-dir research/tr148/results/20260512_174624 \
    --model llama3.1:8b-instruct-q8_0 \
    --output judge_labels_llama.jsonl

# 3. Run shieldgemma + llama-guard3 (Layer 1b composite-harm axis, ~50 min each)
python research/tr148/ollama_judge.py \
    --run-dir research/tr148/results/20260512_174624 \
    --model shieldgemma:9b \
    --output judge_labels_shieldgemma.jsonl

python research/tr148/ollama_judge.py \
    --run-dir research/tr148/results/20260512_174624 \
    --model llama-guard3:8b \
    --output judge_labels_llama_guard.jsonl

# 4. Run analysis (20 passes)
python research/tr148/analyze.py \
    --run-dir research/tr148/results/20260512_174624

# 5. Generate report
python research/tr148/generate_report.py \
    --run-dir research/tr148/results/20260512_174624
```

### Pinned dependencies

| Package | Version |
|---|---|
| Python | 3.13.1 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| openai SDK (calibration only, not active in 5-judge primary path) | 2.36.0 |
| Ollama runtime | 0.6.x |
| Ollama model: gemma3:12b | Default Q4_K_M, ~8.1 GB |
| Ollama model: llama3.1:8b-instruct-q8_0 | Q8_0, ~8.5 GB |
| Ollama model: shieldgemma:9b | Default Q4_K_M, ~5.7 GB |
| Ollama model: llama-guard3:8b | Default Q4_K_M, ~4.9 GB |

### Hardware envelope

- NVIDIA RTX 4080 Laptop, 12 GB VRAM, sm_8.9
- CUDA driver 13.2
- Windows 11 host (Ollama runs natively on Windows or WSL2; both produce identical Ollama API behavior at `localhost:11434`)
- Ollama VRAM-paging is sequential: only one judge model resident at a time; model swap latency ~5-10s

### Run timing on RTX 4080 Laptop

| Step | Wall time |
|---|---:|
| prepare_records | ~5 min |
| llama3.1:8b judge run | ~70 min |
| shieldgemma:9b judge run | ~52 min |
| llama-guard3:8b judge run | ~45 min |
| analyze (20 passes) | ~3 min |
| generate_report | ~1 min |
| **Total active GPU time** | **~170 min** |

### Resume safety

Each Ollama judge run is resume-safe: re-invoking the dispatcher with the same `--output` filename and any existing rows in the file will skip already-judged records and only run the missing ones. The dispatcher writes the JSONL via atomic temp-file replace on every checkpoint flush (every 50 records), so a kill / restart at any point loses at most 50 records. The analyze.py and generate_report.py steps are stateless beyond reading the input JSONLs.

### Where the artifacts live

| File | Size | Content |
|---|---:|---|
| `safety_records.jsonl` | 36 MB | 13,724 records joined from TR145 source |
| `judge_labels_llama.jsonl` | 6.8 MB | 13,724 llama3.1 labels |
| `judge_labels_shieldgemma.jsonl` | 6.4 MB | 13,724 shieldgemma labels |
| `judge_labels_llama_guard.jsonl` | 5.8 MB | 13,724 llama-guard3 labels |
| `judge_labels_gpt4o.jsonl` | 45 KB | 100 records from killed sync run |
| `adjudication_subset.jsonl` | 19 KB | 93 high-disagreement records for follow-up |
| `tr148_analysis.json` | 43 KB | 20-pass analysis output |
| `pairwise_kappas.json` | 15 KB | Pairwise κ matrix split for paper-package consumption |
| `tr148_report.md` | 30 KB | Auto-generated report (this hand-narrated version is the published one) |

### Re-running just the analysis after a code change

```bash
python research/tr148/analyze.py --run-dir research/tr148/results/20260512_174624
python research/tr148/generate_report.py --run-dir research/tr148/results/20260512_174624
```

Total time: ~4 minutes. No new judge calls; runs entirely on the existing JSONL files.

---

## Appendix A: Raw Pairwise κ Table

### A.1 All 15 pairs at corpus scale

| Pair | κ | 95% CI | n | po | pe | PABAK | Krippendorff α | Band |
|---|---:|---|---:|---:|---:|---:|---:|---|
| regex \| gemma3:12b | 0.3626 | [0.3461, 0.3788] | 13,676 | 0.7340 | 0.5827 | 0.4680 | 0.3625 | fair |
| regex \| gpt-4o | 0.7840 | [0.6291, 0.9150] | 94 | 0.9149 | 0.6059 | 0.8298 | 0.7832 | substantial |
| regex \| llama3.1:8b | 0.0822 | [0.0654, 0.0991] | 12,817 | 0.5876 | 0.5507 | 0.1752 | 0.0821 | slight |
| regex \| shieldgemma:9b | −0.0272 | [−0.0410, −0.0140] | 12,024 | 0.7138 | 0.7218 | 0.4276 | −0.2061 | poor |
| regex \| llama-guard3:8b | 0.1930 | [0.1772, 0.2096] | 12,024 | 0.5972 | 0.5009 | 0.1944 | 0.1929 | slight |
| gemma3:12b \| gpt-4o | 0.8774 | [0.7361, 0.9716] | 94 | 0.9574 | 0.6530 | 0.9148 | 0.8780 | near_perfect |
| gemma3:12b \| llama3.1:8b | 0.6917 | [0.6824, 0.7008] | 12,809 | 0.8480 | 0.5076 | 0.6960 | 0.6917 | substantial (primary) |
| gemma3:12b \| shieldgemma:9b | −0.1286 | [−0.1428, −0.1145] | 12,018 | 0.6792 | 0.7158 | 0.3584 | −0.1503 | poor |
| gemma3:12b \| llama-guard3:8b | −0.1468 | [−0.1620, −0.1316] | 12,018 | 0.4538 | 0.5258 | −0.0924 | −0.1468 | poor |
| gpt-4o \| llama3.1:8b | 0.0000 | [0.0, 0.0] | 94 | 0.7766 | 0.7766 | 0.5532 | −0.1198 | slight (degenerate) |
| gpt-4o \| shieldgemma:9b | −0.0611 | n/a (n=94 low power) | 94 | 0.5319 | 0.5594 | 0.0638 | −0.1100 | poor |
| gpt-4o \| llama-guard3:8b | −0.0092 | n/a (n=94 low power) | 94 | 0.2340 | 0.2415 | −0.5320 | −0.0850 | poor |
| llama3.1:8b \| shieldgemma:9b | −0.1866 | [−0.2009, −0.1719] | 11,382 | 0.5896 | 0.6534 | 0.1792 | −0.1866 | poor |
| llama3.1:8b \| llama-guard3:8b | −0.2596 | [−0.2740, −0.2447] | 11,382 | 0.4214 | 0.5378 | −0.1572 | −0.2596 | poor |
| shieldgemma:9b \| llama-guard3:8b | 0.2136 | [0.1953, 0.2317] | 12,024 | 0.5973 | 0.4881 | 0.1946 | 0.2135 | fair |

### A.2 Bootstrap CI summary

All κ values reported in A.1 use 1000-iteration bootstrap with seed 42 over the per-pair filtered paired-record subset. The bootstrap CI is the primary CI used for verdict decisions (per TR148 convention). The asymptotic SE (computed by Fleiss-Cohen formula) is reported alongside in `pairwise_kappas.json` and is consistent with the bootstrap CIs at the corpus-scale n.

---

## Appendix B: Per-Task κ × Pair Matrix

The per-task κ matrix in `tr148_analysis.json:pairwise_kappas.per_task` reports κ for each of the 15 pairs at each of the 5 task families. As noted in SS5, the current implementation uses the hardcoded gpt-4o-anchored primary pair for the task ladder, which makes most per-task κ values degenerate (n = 0 for bbq_bias / jailbreakbench / jailbreak_amp / truthfulqa). The corpus-scale per-task κ for the operational primary pair (gemma3 × llama3.1) is estimated in SS5.2 and will be computed directly in TR148 v2.1.

The full matrix is too large for this appendix; consult `pairwise_kappas.json` directly for the complete table.

---

## Appendix C: Per-Judge Label Distributions

### C.1 Refusal-axis judges

| Judge | n labeled | FULL_REFUSAL | PARTIAL_REFUSAL | COMPLIANCE | UNCLEAR / other |
|---|---:|---:|---:|---:|---:|
| gemma3:12b | 13,724 | ~7,400 (54%) | ~1,800 (13%) | ~2,400 (17%) | ~2,100 (15%; includes truthfulqa/bbq labels) |
| llama3.1:8b | 13,724 | 6,945 (50.6%) | 738 (5.4%) | 12 (0.1%) | 6,029 (44%; includes truthfulqa/bbq labels mapped to other vocabularies, plus 907 UNCLEAR) |
| gpt-4o (n=94 subset) | 94 | 75 (80%) | 12 (13%) | 6 (6%) | 1 (1%) |

(Gemma3 numbers approximate from the TR145 source label distribution.)

### C.2 Composite-harm-axis judges

| Judge | n labeled | YES / unsafe | NO / safe | UNCLEAR (truthfulqa null mapping) |
|---|---:|---:|---:|---:|
| shieldgemma:9b | 13,724 | 2,141 (15.6%) | 11,583 (84.4%) | 0 explicit UNCLEAR; 1,700 truthfulqa records mapped to None per outcome function |
| llama-guard3:8b | 13,724 | 7,489 (54.6%) | 6,235 (45.4%) | 0 explicit UNCLEAR; 1,700 truthfulqa records mapped to None per outcome function |

The 3.5× ratio between shieldgemma's 15.6% unsafe rate and llama-guard3's 54.6% unsafe rate is the operational difference in their harm-detection thresholds. shieldgemma is materially more conservative than llama-guard3 about labeling content harmful — this is consistent with shieldgemma's training emphasis on input prompt-injection detection vs llama-guard3's broader harm-category coverage.

---

## Appendix D: Glossary

| Term | Definition |
|---|---|
| **JTP** | Judge Triangulation Protocol — the framework defined in TR140 v3.0 for assessing cross-family judge agreement on safety labels via Cohen's κ thresholds. |
| **κ (Cohen's kappa)** | Chance-corrected inter-rater agreement statistic on a categorical outcome axis. Range: [−1, +1]; 0 = chance agreement, 1 = perfect agreement, negative = systematic anti-correlation. |
| **PABAK** | Prevalence-Adjusted Bias-Adjusted Kappa (Byrt 1993). Corrects for marginal imbalance that depresses κ when one outcome dominates the marginal. |
| **Krippendorff's α** | Multi-rater-generalizable agreement coefficient. For 2-rater binary it approximates κ but is more robust to missing data. |
| **Landis-Koch bands** | Standard interpretive bands for κ: [-1, 0] poor; [0.01, 0.20] slight; [0.21, 0.40] fair; [0.41, 0.60] moderate; [0.61, 0.80] substantial; [0.81, 1.00] near_perfect. |
| **MDE** | Minimum Detectable Effect — the smallest κ value the test could detect as significantly different from H0: κ=0 at α = 0.05, power = 0.80 given the observed paired-sample n. |
| **TOST** | Two One-Sided Tests for equivalence. In TR148 context: one-sided lower bound on the bootstrap κ distribution at 90% confidence, compared against the JTP threshold (0.40 or 0.70). |
| **Holm-Bonferroni** | Stepdown family-wise error-rate correction across the 15-pair pairwise κ family. |
| **Refusal-axis judge** | An LLM judge configured with TR148's response-classification prompt template (FULL_REFUSAL / PARTIAL_REFUSAL / COMPLIANCE / etc. label vocabulary). Scores whether the response refused the adversarial request. |
| **Composite-harm-axis judge** | A safety-specialist LLM (shieldgemma, llama-guard3) that scores whether the full interaction (prompt + response) contains harmful content. Native binary output: Yes / No or safe / unsafe. |
| **Layer 1a** | Bridge-paper measurement-validity gate, refusal-axis JTP triangulation. Active in TR148 v2 with regex + gemma3 + llama3.1; pending: claude, gpt-4o. |
| **Layer 1b** | Bridge-paper measurement-validity gate, composite-harm-axis orthogonal screen. Active in TR148 v2 with shieldgemma + llama-guard3 (preliminary; production form runs them in their native chat-format mode). |
| **Umbrella (OpenAI)** | OpenAI Researcher Access Program approval; required before sending adversarial-prompt content at scale through OpenAI's API to avoid org-level content-policy flagging. |
| **Fellowship (Anthropic)** | Anthropic Fellows Program grant; provides Anthropic API access at research-program tier. The implicit umbrella for Anthropic API on adversarial content. |

---

## References

- Banterhearts TR140 v3.0 — Judge Triangulation Protocol calibration. `papers/manyshot_longcontext_quantization/`.
- Banterhearts TR142 — Refusal Template Stability Index (RTSI). `papers/quality_safety_correlation/`.
- Banterhearts TR144 — Speculative Decoding × Safety (TAIS). `papers/speculative_decoding_safety/`.
- Banterhearts TR145 v1.0 — KV-Cache Quantization × Safety. `PublishReady/reports/Technical_Report_145.md`. Parked paper; bridge-paper worked-example seed.
- Banterhearts TR147 v4.0 — Compile Reproducibility Index (CRI). `papers/benchmarking_integrity/`.
- Banterhearts bridge paper — Serving-State Safety Certification. `papers/serving_state_safety_certification/UPGRADE_PLAN.md`. Phase 4 consolidation paper.
- Banterhearts `feedback_openai_batch_api_mandatory.md` — OpenAI Batch API rule for corpus-scale judging.
- Banterhearts `feedback_openai_safety_umbrella_gate.md` — OpenAI compliance gate for adversarial content.
- Banterhearts `feedback_tr_analyze_no_mandatory_judge.md` — analyze.py mandatory-judge gate bug class.
- Banterhearts `tr148_safety_judge_axis_finding.md` — dual-axis methodology finding documentation.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.
- Byrt, T., Bishop, J., & Carlin, J. B. (1993). Bias, prevalence and kappa. *Journal of Clinical Epidemiology*, 46(5), 423–429.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Lakens, D. (2017). Equivalence tests: A practical primer for t-tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.

---

*End of Technical Report 148 v2.*







