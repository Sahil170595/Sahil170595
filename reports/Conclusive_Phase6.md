# Technical Report — Conclusive Synthesis, Phase 6

## Measurement-Validity Substrate and the Inference-Flag Safety Null Line (TR144–TR149 + TR152)

**Author:** Sahil Kadadekar
**Phase:** 4.5 (measurement-validity → Layer 5 anchor)
**Scope:** TR144, TR145, TR146, TR147, TR148, TR149, TR152
**Status:** Conclusive synthesis. All constituent technical reports executed end-to-end on local / RunPod hardware; cloud-gated extensions (TR150, TR151, TR153) deferred and documented in Chapter 6.
**Companion files:** `Conclusive_Phase6_Extended_Appendices.md` (cross-TR tables, named-method registry, glossary), `Conclusive_Phase6_Whitepaper.md` (condensed external-facing version).

---

## Abstract

Phase 6 of the Banterhearts research program does two things at once. First, it builds a **measurement-validity substrate** — the set of instruments and validity gates that any serving-state safety claim must pass before it is allowed to count as evidence. That substrate has three legs: a *compile-reproducibility* gate (TR147's Compile Reproducibility Index, CRI, and its Triton stack-version kill-shot), a *judge-triangulation* gate (TR148 v2's Judge Triangulation Protocol, JTP, and its dual-axis finding that safety-specialist judges and general LLM judges measure orthogonal constructs), and a *corpus-standardization* gate (TR149's adoption of the four literature-canonical safety batteries — HarmBench, JailbreakBench, StrongREJECT, XSTest — as the program's reference corpus). Second, on top of that substrate, Phase 6 runs an **inference-flag safety null line**: a sequence of paired, matched-pair studies each asking whether a serving-state knob moves refusal behaviour, and each returning a calibrated null. The members of that null line are TR144 (speculative decoding at temperature 0, screened by the Typical-Acceptance Invariance Screen, TAIS), TR145 (FP8 KV-cache at a single configuration), TR149 (FP8 KV-cache re-measured on the standardized batteries), and TR152 (FP8 KV-cache swept across the serving state in a star factorial).

The headline results are unambiguous and conservatively bounded. Speculative decoding is behaviourally inert with respect to safety at greedy decoding (maximum |Cohen's h| = 0.024 across 18 adversarial contrasts, ~4× below the 0.1 triviality cutoff). FP8 KV-cache produces no statistically supported safety degradation at a single configuration (Mantel–Haenszel pooled OR = 1.05, 95% CI [0.90, 1.23]), the same null replicates on the standardized batteries (pooled OR = 0.8065, CI [0.3828, 1.6989]; 0 of 12 cells significant after Holm–Bonferroni; 12 of 12 cells positively equivalent at ±3pp), and the null survives a serving-state factorial (0 discordant pairs across 8,976 matched harmful-battery pairs; all measurable footprint confined to a sub-one-percentage-point over-refusal lean on the Qwen 2.5 family at the XSTest benign edge). A deliberate negative result anchors the methodology: TR146's four mechanistic probes all fail to distinguish safe from dangerous quantized configurations, which is precisely why the safety verdict across the entire arc is carried by behavioural measurement rather than by interpretability artefacts.

The deliverable of Phase 6 is not a single number but a *certificate shape*. The downstream serving-state safety certification paper (the "bridge paper") inherits a five-layer protocol — Layer 1 measurement validity (TR148), Layer 2 behavioural screens (TR142/TR144), Layer 3 compile integrity (TR147), Layer 4 scale validity (TR149 for 1B–3B, TR151 cloud-gated for 7B–72B), Layer 5 serving-state validity (TR152) — and Phase 6 fills Layers 1, 3, and 5 with executed evidence, supplies the 1B–3B anchor for Layer 4, and contributes the TAIS member of Layer 2. Every claim is stated against the claim ladder: "passes certification under tested conditions," never "FP8 is safe."

---

## Table of Contents

1. Executive Summary
2. Research Motivation and Hypothesis Structure
3. Background and Related Work
4. Methodology Overview
5. Certificate Impact Matrix
6. Chapter 1 — Measurement-Validity Foundation (TR147 CRI + TR148 JTP)
7. Chapter 2 — Corpus Standardization (TR149)
8. Chapter 3 — The Inference-Flag Safety Null Line (TR144 + TR145 + TR149 + TR152)
9. Chapter 4 — The Mechanistic Dead-End (TR146)
10. Chapter 5 — Cross-Axis Composition and the Layer 5 Anchor (TR152)
11. Chapter 6 — Cloud-Gated Deferrals (TR150, TR151, TR153)
12. Cross-TR Synthesis — From Measurement Validity to the Layer 5 Certificate
13. Integration with Phase 5 (TR138–143) and Phase 5 (TR155 / TR163)
14. Numbered Findings
15. Threats to Validity (Phase 6-wide)
16. Production Guidance and Operational Doctrine
17. Conclusion
18. References

---

## 1. Executive Summary

Phase 6 is the measurement-validity phase of the safety line. The phases before it mapped an attack surface (Phase 5, TR138–143); the phase after it turns to mitigation (Phase 5, TR155/TR163 onward). Sandwiched between, Phase 6 answers a prior question that both neighbours depend on: *can we trust the instruments at all?* If a compiled-latency benchmark flips its qualitative verdict when a Triton minor version changes, if a single LLM judge's safety labels do not survive cross-family triangulation, or if a "null result" is an artefact of a lab-assembled task set, then no downstream safety or performance claim is defensible. Phase 6 builds the gates that make those failure modes detectable, then runs a battery of inference-flag studies through those gates and reports the verdicts.

**The seven constituent reports decompose into two threads.**

The **measurement-validity thread** is three reports:

- **TR147 (CRI).** On one fixed RTX 6000 Ada GPU running one fixed model, bumping only the Triton minor version (3.3.1 → 3.4.0 → 3.6.0) collapses the `torch.compile` prefill speedup from 62–77% to a near-zero neutral result and simultaneously erases an 80% `reduce-overhead` decode crash. The cross-version effect sizes on the compile-vs-eager contrast span |Cohen's d| ≈ 14–49 — essentially no distributional overlap. The Compile Reproducibility Index formalizes this as the maximum pairwise |Cohen's d| across a stack-perturbation set, with bands robust (<0.5) / sensitive (<2) / fragile (≥2) / catastrophic (≥10). The lesson — pin the Triton version (and the repository code SHA) as part of the benchmark identity, or the conclusion is not reproducible — is the performance-measurement leg of the substrate.
- **TR148 (JTP).** The Judge Triangulation Protocol converts cross-family judge agreement (Cohen's κ) into a downstream-routing verdict: κ ≥ 0.70 robust (single judge sufficient), 0.40–0.70 triangulate (multi-judge majority vote mandatory), <0.40 untrustable. On the 13,724-record TR145 safety subset, the cross-family pair gemma3:12b × llama3.1:8b scores κ = 0.6917 (CI [0.6824, 0.7008], n = 12,809) — the triangulate bucket, 0.0083 below robust. The decisive secondary finding is *dual-axis*: the safety-specialist judges (shieldgemma:9b, llama-guard3:8b) anti-correlate with the general judges (κ = −0.13 to −0.26 across four large-n pairs) while remaining internally coherent (κ = 0.21), because the specialists score composite prompt+response harm while the general judges score response refusal. They are two orthogonal axes, not one noisy axis.
- **TR149 (corpus standardization).** Re-running the FP8 KV-cache contrast on the four literature-canonical batteries, holding everything else fixed, replicates the TR145 null with positive ±3pp equivalence on all 12 (battery × model) cells and a pooled OR of 0.8065. As a by-product it shows the JTP verdict is *corpus-specific*: the same judge pair that scored κ = 0.69 on TR145's mixed task set scores κ = 0.83 on the clean adversarial batteries.

The **inference-flag null-line thread** is four reports (TR144, TR145, TR149, TR152), each asking "does this serving-state knob move safety?" and each returning a calibrated null:

- **TR144 (TAIS).** Speculative decoding at temperature 0 is behaviourally inert: 90.66% byte-identity under rejection sampling, literal-zero per-task safety deltas under typical acceptance, flat dose-response across speculation length, and a maximum |Cohen's h| of 0.024 across the E1–E5 expansion (production-scale 70B, adversarial draft, quantized draft, two seeds, bfloat16). The temperature > 0 carve-out is explicit and load-bearing.
- **TR145 (FP8 single-config).** The base case of the null line: 24,054 paired records across five phases, every primary McNemar non-significant, every interaction ANOVA approximately additive, pooled OR 1.05 [0.90, 1.23]. The claim is bounded — "no detectable FP8 effect under tested conditions," not "FP8 is safe" — and that scope discipline is what the rest of the line extends.
- **TR149 (FP8 standardized).** The literature-comparable replication, described above.
- **TR152 (FP8 serving-state factorial).** The capstone: 45,000 responses, 20,754 matched pairs across 120 strata, 0 discordant pairs on 8,976 harmful-battery pairs, all discordance on XSTest and 91% of it Qwen-bound, max |Cohen's h| = 0.0458, and a flat FP8-interaction spread (2.99pp, inside the ±3pp band). The cross-context pooled OR of 1.8817 [1.3185, 2.6855] is a *located* over-refusal lean on the benign edge, not a harmful-compliance failure.

**A deliberate negative result governs both threads.** TR146 ran the most attractive shortcut — replace expensive behavioural evaluation with a cheap forward-pass mechanistic probe — across four probes (first-token entropy, refusal-direction geometry, calibration drift, safety-neuron quantization error) and 5,100 forward passes, and falsified it on every axis. The safety mechanism is real (safety-critical neurons absorb 1.40× disproportionate quantization error, p < 0.0001) but *necessary-not-sufficient*: every quantized model suffers the damage while only some cross the behavioural failure threshold. The operational consequence is that the safety verdict across Phase 6 lives in measured refusal behaviour, not in weights or residual streams.

**What Phase 6 hands forward.** The five-layer certification protocol, with executed evidence in Layers 1, 3, and 5; the 1B–3B anchor for Layer 4; the TAIS member of Layer 2; and a worked example of a serving-state flag (FP8 KV-cache) that *passes* the protocol, stated as the bounded claim the evidence supports. The cloud-gated extensions (TR150 long context, TR151 7B–72B scale, TR153 KV-method sweep) are the documented widenings of the certified envelope, all deferred to the bridge-paper GO/NO-GO trigger.

### 1.1 Claim status — cross-report, reviewer-proof

The single discipline that holds the arc together is that every headline is tagged to a tier on the project claim ladder and stated against its bound. The table below is the reviewer's-eye summary: what is claimed, on what evidence, and where the claim stops.

| # | Headline claim | Tier | Primary evidence | Stated bound |
|---|---|---|---|---|
| C1 | FP8 KV-cache is harmful-refusal-neutral across the tested serving state | **Supported** | TR152: 0 / 8,976 discordant harmful-battery pairs; interaction spread 2.99pp inside ±3pp | 1B–4B, vLLM v0.19.1, tested {batch, prefix, temperature} only |
| C2 | Speculative decoding is behaviourally inert at greedy decoding | **Supported** | TR144: max \|Cohen's h\| = 0.024 over 18 AdvBench contrasts (64,855 paired samples) | temperature 0 only; linear draft–verify only |
| C3 | The FP8 null replicates on the field's standard batteries | **Supported** | TR149: 12/12 cells TOST-equivalent at ±3pp; pooled OR 0.8065 [0.3828, 1.6989] | 1B–3B; discriminating power is effectively a two-battery claim |
| C4 | The judge cohort has a *measured* validity floor | **Supported** | TR148: gemma3 × llama3.1 κ = 0.6917 [0.6824, 0.7008], n = 12,809 → triangulate | TR145 corpus; two-LLM-judge (gpt-4o/Claude deferred) |
| C5 | Safety-specialist and general judges measure orthogonal axes | **Supported** | TR148: four cross-axis κ = −0.13 to −0.26; within-specialist κ = +0.21 | corpus-scale; refusal vs composite-harm construct |
| C6 | A compiled-latency conclusion can flip on a dependency bump | **Supported** | TR147: Triton-only ablation 62.82% → 0.84% prefill; cross-version \|d\| ≈ 14–49 | RTX 6000 Ada, one model; vLLM-stack replication pending |
| C7 | FP8 induces a located over-refusal lean on one model family | **Licensed** | TR152: per-family MH OR Qwen 3.878 [2.386, 6.302] vs Llama 0.484, Phi-3 0.130 | benign-edge (XSTest) only; sub-1pp; temperature axes |
| C8 | The JTP verdict is corpus-specific | **Licensed** | TR149: same pair κ = 0.8306 on clean batteries vs 0.6917 on the mixed set | clean refusals are easier to judge; do not self-select dispatch |
| C9 | Mechanistic probes predict safety degradation under quantization | **Forbidden (F3)** | TR146: four probes all \|r\| < 0.15; danger-vs-neutral Mann–Whitney p = 0.979 | AWQ/GPTQ INT4; falsified, not merely unconfirmed |
| C10 | "FP8 is safe" unconditionally | **Forbidden** | — | never claimed; the supported form is always "no detectable effect under tested conditions" |

**Observations.** The ten claims partition cleanly: six Supported (each carries an executed run with adequate pooled power), two Licensed (each is evidenced but bounded by an explicit scope limit), and two Forbidden (one the evidence specifically *rejects*, one the program refuses to ever assert). No Supported claim is stated without its bound in the same row, and the two Licensed claims (C7, C8) are the ones a careless synthesis would over-state into "FP8 degrades Qwen safety" (it degrades benign-edge *over-refusal*, not harmful compliance) and "judges are robust" (they are robust *on clean adversarial corpora*, triangulate on mixed sets).

> The reviewer-proof posture is the deliverable, not a presentational nicety. A bridge paper that inherits C1–C10 as a ladder can defend each claim against its most likely objection in a single sentence, because the bound is already attached. The arc never spends credibility on an unbounded claim it would have to retract under questioning.

### 1.2 Operational defaults — the Phase 6 certificate at a glance

| Decision | Phase 6 default | Conditioning evidence |
|---|---|---|
| **Enable FP8 (E4M3) KV-cache for 1B–4B serving?** | **Yes**, across batch ∈ {1, 8, 32}, prefix-cache {on, off}, temperature ∈ {0.0, 0.7, 1.0}, without per-config re-validation | TR152 harmful-core invariance + flat interaction; TR145/TR149 base + standardized nulls |
| **Running Qwen 2.5 at temperature > 0 on a benign-edge-heavy workload?** | Enable FP8 **but monitor the over-refusal rate** | TR152 per-family OR (Qwen 3.878; effect absent on Llama/Phi-3) |
| **Trust a cross-stack latency number?** | Only after a CRI gate returns `robust`/`sensitive`; pin Triton minor version **and** repository code SHA | TR147 Triton kill-shot + gpt-fast code-SHA axis |
| **Trust a single LLM judge's safety label?** | Only if the JTP κ on your calibration corpus clears 0.70; otherwise triangulate (≥3 judges, majority vote) | TR148 triangulate verdict at κ = 0.6917 |
| **Read safety off weights / logits / a forward-pass probe?** | **No** — behavioural measurement only | TR146 four-probe falsification + GPTQ confidence paradox |
| **Treat speculative decoding as safety-neutral?** | Only at greedy decoding (temperature 0); re-test above it | TR144 temperature-0 carve-out |

**Observations.** The card is deliberately a mix of one green light (enable FP8 in the validated envelope) and five gates, because the contribution of Phase 6 is not "FP8 is fine" — it is the *gates* that let a deployment decide whether its own configuration is inside the certified envelope. The single most consequential default is the last-but-one row: the program's most attractive cost-saving shortcut (predict safety from a cheap forward pass) is explicitly contraindicated, which is what keeps every other default behavioural and therefore measurable.

> A deployment that follows this card and stays inside 1B–4B / vLLM v0.19.1 / the tested serving axes inherits a *single* FP8 certificate it does not have to re-earn per configuration. A deployment outside that envelope (7B–72B, long context, other KV-quant methods) is told exactly which scaffold (TR151, TR150, TR153) names its missing evidence — it is never silently extrapolated to.

### 1.3 Program trajectory — Phase 5 → Phase 6 → Phase 5

Phase 6 is the hinge of a three-phase safety arc. Phase 5 (TR138–143) *mapped the attack surface*: it asked where inference optimization and adversarial pressure can move refusal behaviour at all — batch-size non-determinism, multi-turn jailbreaks, many-shot long-context exploitation, cross-architecture refusal fragility, the quality-safety divergence, and cross-request composition under continuous batching. Phase 6 (this synthesis: TR144–149, TR152) then asks the prior question both neighbours depend on — *can the instruments that produced those maps be trusted, and does the most-deployed serving flag (FP8 KV-cache) move safety once the instruments are validated?* Phase 7 (TR155/TR163 onward) turns to *mitigation*: TR163, the program's first MVP feasibility demo (offline LOOCV reanalysis over TR142's 51-cell table, not a paper-grade campaign), shows an RTSI-gated quantization-routing policy recovers ~76% of the refusal gap at roughly 22% of requests routed to the safe path, with a leave-one-out cross-validated routing AUC of 0.84.

| Phase | Question | Members | Phase 6's relationship |
|---|---|---|---|
| **Phase 5** | *Where* can the serving state move safety? | TR138–143 (batch, multi-turn, long-context, cross-arch, quality-safety, composition) | Phase 6 validates the instruments those maps were drawn with (JTP, CRI) and re-tests the FP8 flag under them |
| **Phase 6** | *Can we trust the instruments, and does FP8 move safety under them?* | TR144–149, TR152 | this synthesis |
| **Phase 5** | *How do we mitigate the located vulnerabilities?* | TR155, TR163 (RTSI-gated routing) | Phase 6's RTSI (Layer 2) and behavioural-measurement discipline are the substrate the mitigations are built and validated on |

**Observations.** The trajectory explains why a *measurement-validity* phase is worth a full conclusive synthesis of its own: a mitigation phase that routes on RTSI (TR163) inherits its right to trust RTSI from the behavioural-screen discipline established here, and an attack-surface phase whose nulls were measured by a single un-triangulated judge would not survive the JTP gate Phase 6 introduces. Phase 6 is the phase that converts the program's findings from "one lab's measurements" into "measurements under validated instruments."

> The deeper integration with both neighbouring phases — which Phase 5 findings Phase 6 confirms, tensions, or re-grounds, and how Phase 6's substrate licenses the Phase 5 mitigations — is developed in §13. The point of stating it up front is that Phase 6's value is *relational*: it is the validity layer the rest of the arc stands on.

---

## 2. Research Motivation and Hypothesis Structure

### 2.1 Why a measurement-validity phase at all

The safety line had, by the end of Phase 5, accumulated a large body of null and located results on the safety cost of inference optimization. Those results were only as trustworthy as the instruments that produced them, and three latent failure modes threatened all of them simultaneously:

1. **Performance-measurement non-reproducibility.** Compiled-latency conclusions (the basis for "this optimization is worth its safety risk" trade-offs) might be artefacts of an unpinned software stack rather than properties of the optimization.
2. **Judge-label non-reproducibility.** Safety verdicts produced by a single LLM judge might not survive cross-family triangulation, in which case every "no safety change" finding is one judge's opinion rather than a measurement.
3. **Corpus-artefact non-reproducibility.** A null measured on a lab-assembled task set might be a property of that idiosyncratic mix rather than of the intervention, and would not survive a reviewer's "your null is an instrument artefact" objection.

Phase 6 builds one gate against each failure mode (CRI, JTP, corpus standardization) and then demonstrates the gates by passing a real intervention (FP8 KV-cache and speculative decoding) through them.

### 2.2 The hypothesis structure of the null line

Each member of the inference-flag null line is framed so that the *safety-degrading* hypothesis is the one that requires evidence to act on, and equivalence is a *positive* small-effect finding rather than mere failure to reject. Concretely, the line shares a three-hypothesis structure:

- **H0 (no effect):** the serving flag does not change refusal behaviour beyond the ±3pp deployment-relevance margin.
- **H1 (located effect):** the flag changes refusal behaviour in a specific, attributable corner (a model family, a battery slice, a temperature axis) without a generic serving-state interaction.
- **H2 (interaction):** the flag's safety effect is modulated by serving-state axes (batch, context, concurrency), so no single certificate can hold across configurations.

TR144 and TR145 return H0. TR149 returns H0 on the standardized corpus. TR152 returns H1 (a located Qwen-XSTest over-refusal lean) with H2 explicitly rejected (flat interaction spread). The distinction between H1 and H2 is the entire value of the serving-state factorial: a located effect can be carried as a per-family footnote on a single certificate, whereas an interaction would force per-configuration re-validation.

### 2.3 The five-layer certification protocol (the target artefact)

Phase 6 is organized around the bridge paper's five-layer serving-state safety certification protocol. The canonical layer assignment, taken from the project claim ladder (`papers/serving_state_safety_certification/CLAIM_LADDER.md`, S4), is:

| Layer | Question it answers | Phase 6 anchor | Status after Phase 6 |
|---|---|---|---|
| **Layer 1 — Measurement validity** | Can you trust the safety *label*? | TR148 (JTP), split 1a refusal-axis / 1b composite-harm | Evidenced |
| **Layer 2 — Behavioural screens** | Does a cheap behavioural screen flag the dangerous configs? | TR142 (RTSI, shipped) + TR144 (TAIS) | TAIS member evidenced |
| **Layer 3 — Compile integrity** | Can you trust the latency *number*? | TR147 (CRI) | Evidenced (vLLM-stack replication pending) |
| **Layer 4 — Scale validity** | Does the null hold across model scale? | TR149 (1B–3B) + TR151 (7B–72B, cloud-gated) | 1B–3B anchored; scale pending |
| **Layer 5 — Serving-state validity** | Does the null hold across the serving state? | TR152 (star factorial) | Evidenced (1B–4B) |

> **Interpretive note on layer numbering.** The "inference-flag safety null line" (TR144 / TR145 / TR149 / TR152) is a *thematic thread* that runs *across* layers — TAIS sits in Layer 2, the single-config null TR145 is the base case feeding Layers 4 and 5, the standardized replication TR149 is the Layer 4 anchor, and the serving-state factorial TR152 is the Layer 5 anchor. It is not itself a numbered layer. This synthesis uses the claim-ladder numbering above throughout; where a constituent report's own forward-pointer used a different ad-hoc numbering, it is reconciled to this table.

### 2.4 The research questions, stated at the program level

The three failure modes (§2.1), the hypothesis structure (§2.2), and the target artefact (§2.3) reduce to five research questions that the seven executed reports answer between them. Stating them as a program — rather than letting each report carry its own question in isolation — is what makes the synthesis a certificate rather than a digest.

| RQ | The question | Reports that answer it | Verdict |
|---|---|---|---|
| **RQ1 — Instrument validity** | Can the latency and safety instruments be trusted to reproduce, or are the measurements stack/judge artefacts? | TR147 (CRI), TR148 (JTP) | Latency reproduces only under a pinned stack (catastrophic CRI otherwise); judge labels reproduce within band (triangulate κ) and split into two orthogonal axes |
| **RQ2 — The harmful-core null** | Does any inference-serving flag — FP8 KV-cache, speculative decoding, batch size, prefix caching, temperature — move the *harmful refusal core* beyond ±3pp? | TR144, TR145, TR149, TR152 | No: harmful-core refusal is invariant across every tested flag (max \|h\| 0.024 / 0.0458; 0 of 8,976 harmful pairs discordant under serving-state composition) |
| **RQ3 — The mechanistic shortcut** | Can a cheap forward-pass probe substitute for behavioural measurement and predict which quantized config is dangerous? | TR146 | No (Forbidden, F6): four probes all fail; safety cannot be read off the weights |
| **RQ4 — Scale and serving-state validity** | Does the null hold across model scale and across the *composed* serving state, or is it a single-config artefact? | TR149 (scale, 1B–3B), TR152 (serving-state factorial) | Holds on standardized batteries (1B–3B) and across the single-axis serving state; the only located lean is benign-edge over-refusal on one family (H1, not H2) |
| **RQ5 — Cross-phase grounding** | How does the validated substrate re-ground the prior attack-surface phase and license the downstream mitigation phase? | §13 (integration) | Phase 6 converts Phase 5's provisional framings into gated claims and supplies every instrument Phase 7's first MVP mitigation feasibility demo (TR163) consumes |

**Observations.** The five questions are not independent — they are ordered by dependency, which is the same order the certificate composes in. RQ1 must be answered before RQ2 can mean anything (a null measured by an un-triangulated judge on an unpinned stack is not a null, it is an opinion), and RQ3 sits underneath RQ2 as the reason the answer to RQ2 has to be *behavioural*. RQ4 is RQ2 asked again at the boundaries (scale, composed serving state) where a single-config null is most likely to break, and RQ5 is the question that turns four reports' worth of "no effect" into a phase that does work for the phases on either side of it. The verdict column reads as a single sentence: the instruments reproduce within stated bounds, the harmful core does not move, the shortcut to skip the instruments does not exist, the null survives the boundaries that would most likely break it, and the whole thing grounds the map behind it and the mitigations ahead of it.

### 2.5 Contributions of this synthesis

Each constituent report contributes its own result; the conclusive synthesis contributes things that *no single report states*, because they only exist when the seven are read together. These are the deliverables of the synthesis as a document:

1. **The composed five-layer certificate.** No individual TR claims the five-layer protocol; each anchors one layer. The synthesis is where the layers are stated as one artefact with one chain of custody (§2.3, §12.1, §13.4), so a deployment decision can cite "the certificate" rather than seven separate forward-pointers.
2. **The inference-flag null line as a four-member resolution ladder.** TR144/TR145/TR149/TR152 are presented elsewhere as four separate nulls-and-locateds; the synthesis shows they are four resolutions of *one underlying truth* — harmful refusal is invariant to the serving state, and the over-refusal edge is faintly, locally perturbed on one family (§8.3, §12.7).
3. **The dual-axis refinement of Layer 1.** The synthesis carries TR148's finding that the validity gate is two orthogonal axes (refusal vs composite-harm) into the protocol as the 1a/1b split, which changes how every downstream judge cohort must be assembled (§6.2, F4).
4. **The negative control as cross-cutting governance.** TR146 is reframed from "a mechanistic study that found nulls" into the result that *governs both pillars* — the reason the entire arc is behavioural rather than mechanistic (§9.3, §13.4 step 6).
5. **The H1-vs-H2 operationalization.** The synthesis makes explicit that the value of the serving-state factorial is the distinction between a located effect (carry as a per-family footnote) and an interaction (force per-config re-validation), and that TR152 returns the former (§2.2, §10).
6. **The cross-phase grounding.** The synthesis states how Phase 6 re-grounds Phase 5's attack-surface maps and forward-licenses Phase 7's first MVP mitigation feasibility demo (paper-grade expansion queued, not promoted) — the integration that makes the program a single arc rather than three phases (§13).
7. **A reproducible $0-cost substrate.** The synthesis documents that the entire certification substrate was built under the umbrella gate on a local judge cohort at $0 external API cost, which is itself a methodological contribution: the validity layer can be built without a cloud safety-judge budget (§4.4, §4.10, §13.3).

**Observations.** The seven contributions split into two kinds. The first four (composed certificate, resolution ladder, dual-axis split, negative control) are *synthetic* — they are claims about the relationships *between* reports that no report could make alone. The last three (H1/H2 operationalization, cross-phase grounding, reproducible substrate) are *programmatic* — they situate the phase in the arc and in the program's cost-and-process model. A reader who only read the seven TRs would have all the numbers and none of these seven contributions; that gap is precisely what a conclusive synthesis exists to close, and it is why the synthesis is the backbone the downstream papers are built from rather than a summary of them.

> The contribution most likely to be cited downstream is the first: the composed five-layer certificate. A bridge paper does not cite "TR147 and TR148 and TR149 and TR152"; it cites the certificate, with each layer's anchor and bound attached. The synthesis is the document that makes that single citation possible, which is the operational meaning of "this research doc is the backbone for the papers" — the papers compress the certificate to a venue page limit, but they compress *this* statement of it, not seven separate reports.

---

## 3. Background and Related Work

This synthesis sits at the intersection of three literatures that rarely cite one another: low-precision numerics and serving-systems engineering, automated safety evaluation, and the statistics of equivalence. Phase 6 of the program is deliberately a *measurement-validity substrate* rather than a safety claim in its own right — its purpose is to establish that the instruments used to score "is this served configuration safe?" are trustworthy, and only then to run an inference-flag safety contrast through those instruments. Accordingly, the related work below is organized around the seven measurement objects that the constituent technical reports interrogate, plus the statistical machinery that binds them and a brief positioning against the program's prior attack-surface phase. Each subsection establishes what the prior literature actually claims, what it leaves underdetermined for deployed serving state, and why that gap motivates a *behavioural* check in place of a *bit-exactness* or *static-analysis* check.

A recurring theme deserves stating up front. Most of the upstream work — the speculative-decoding distribution-preservation theorems, the FP8 format specifications, the refusal-direction geometry — establishes properties about a *model* or an *algorithm in isolation*. Deployment, by contrast, composes those properties with continuous batching, paged memory, dynamic load, and a particular compiler toolchain. The central methodological wager of Phase 6 is that this composition can break or obscure properties that hold component-wise, and that the only defensible way to certify the composed system is to measure its *behaviour* under the conditions it will actually run in, with statistical instruments whose reliability has been independently established.

### 3.1 FP8 numerics, KV-cache quantization, and floating-point non-associativity

The 8-bit floating-point formats used throughout the program follow the two encodings introduced in *FP8 Formats for Deep Learning* (Micikevicius et al., 2022): **E4M3** (one sign bit, four exponent bits, three mantissa bits) and **E5M2** (one sign bit, five exponent bits, two mantissa bits). E4M3 trades dynamic range for precision and, per the specification, departs from strict IEEE-754 conventions by not representing infinities and reserving only a single NaN bit-pattern, which buys it a slightly wider representable magnitude range; E5M2 retains IEEE-754-style special-value handling. The recommended convention places E4M3 on weights and activations and E5M2 on gradients, with exponent biases of 7 and 15 respectively. For an inference-only safety study the gradient path is irrelevant, so the quantized object of interest is an activation-like tensor in the E4M3 encoding.

The program's distinctive choice is to quantize the **KV-cache** rather than the weights. This matters for the safety question. Weight quantization perturbs the parameters once, at load time, and every forward pass thereafter sees the same perturbed model; the resulting behaviour is at least *deterministic* given fixed inputs. KV-cache quantization, by contrast, perturbs the *running state* — the keys and values accumulated across the prefill and decode of an actual request — and the magnitude of that perturbation depends on the sequence content, the cache layout, and the order in which partial attention reductions are summed. The safety-relevant consequence is that the same prompt can, in principle, receive a different completion depending on serving conditions that have nothing to do with the prompt itself.

The usual mechanistic story for this sensitivity is **floating-point non-associativity**: because `(a + b) + c` need not equal `a + (b + c)` in finite precision, a reduction whose summation order changes across batch sizes or kernel-tiling decisions can yield bit-different results, and those differences can cross a token-selection boundary. We adopt this framing but flag an important refinement from recent serving-systems work. *Defeating Nondeterminism in LLM Inference* (Thinking Machines Lab, 2025) argues that non-associativity combined with concurrent atomic accumulation is *not* in fact the dominant source of run-to-run nondeterminism in production LLM endpoints, because typical forward passes contain few atomic adds; the dominant source is **batch-invariance failure** — kernels that compute numerically different reductions at different batch sizes, composed with a server that varies batch size with load. This sharpens, rather than undermines, the motivation here: whether the proximate cause is reduction-order within a kernel or batch-size-dependent kernel selection, the deployed numerical result is a function of serving state, not just of the prompt and the nominal precision flag.

> The takeaway that drives the whole program is that "FP8 KV-cache is safe" is not a statement that can be settled by inspecting the format specification or checking a bit-exact reproduction. Bit-exactness is the wrong success criterion, because the served system is *expected* to be bit-nondeterministic; the right criterion is whether the *distribution of safety-relevant behaviour* is indistinguishable from the FP16 baseline. That reframing — from a numerics question to a behavioural-equivalence question — is the hinge on which TR145, TR149, and TR152 all turn.

### 3.2 Continuous batching, PagedAttention, and serving-state composition

Modern LLM serving is dominated by the **PagedAttention** design of *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., SOSP 2023), which underlies the vLLM engine used in this program. PagedAttention borrows virtual-memory paging from operating systems: the KV-cache is fragmented into fixed-size blocks that need not be physically contiguous, which nearly eliminates the internal and external fragmentation that plagued earlier serving stacks and enables flexible KV-cache sharing within and across requests. On top of this sits **continuous (iteration-level) batching**, in which requests join and leave the running batch at token granularity rather than waiting for a fixed batch to drain, and **prefix caching**, in which the KV-cache for a shared prompt prefix is computed once and reused across requests that share it.

These mechanisms are what make a single-configuration safety measurement *underdetermined* for deployment. Three serving-state factors compose with the FP8 KV-cache perturbation of Section 3.1 in ways that a one-shot benchmark cannot see. **Batch size and concurrency** determine which kernel tiling and reduction order the attention computation uses, and therefore — per the batch-invariance argument above — the exact numerical state a request observes. **Prefix caching** changes whether a request's early KV entries were quantized in the context of its own prefill or inherited from a shared, separately-quantized prefix, altering the quantization error profile at the start of the sequence. **Decoding temperature** sets how close the next-token distribution sits to a selection boundary, governing how often a small numerical perturbation actually flips a sampled token. None of these is exotic; all three are routine knobs in a production deployment.

> The structural lesson is that a safety verdict measured at, say, batch size 1, greedy decoding, no prefix sharing, does not transfer to a deployment at batch size 32, temperature 0.7, with prefix caching on. Each factor is individually plausible-to-be-null, but a deployed system runs them *jointly*, and interactions are exactly what a single-config study cannot rule out. This is the precise gap TR152 is built to close: a star/factorial design over batch size, prefix caching, and temperature, so that main effects and their interactions on the safety contrast can be estimated rather than assumed away. TR145 establishes the single-config null; TR152 asks whether that null survives serving-state composition.

### 3.3 Speculative decoding sampling theory

Speculative decoding accelerates autoregressive generation by having a small **draft** model propose several tokens that the large **target** model then verifies in a single parallel forward pass. Two near-simultaneous papers established the technique and its central correctness guarantee. *Fast Inference from Transformers via Speculative Decoding* (Leviathan, Kalman, and Matias, ICML 2023; arXiv 2211.17192) introduced the speculative-execution framing and a verification rule. *Accelerating Large Language Model Decoding with Speculative Sampling* (Chen, Borgeaud, Irving, Lespiau, Sifre, and Jumper, DeepMind, 2023; arXiv 2302.01318) introduced a **modified rejection-sampling** scheme for the verification step and demonstrated 2–2.5× decoding speedups on a 70B-parameter model.

The property that matters for safety is the **distribution-preservation guarantee**: under the modified rejection-sampling acceptance rule, the tokens emitted by the speculative procedure are distributed *identically* to tokens that would have been sampled directly from the target model at the same temperature. When a drafted token is rejected, it is resampled from an adjusted residual distribution constructed precisely so that the marginal output distribution is unchanged. Chen et al. are careful to state this preservation holds "within hardware numerics" — i.e., up to the same finite-precision caveats discussed in Section 3.1.

> The theoretical prediction this hands to a safety study is sharp and falsifiable. If speculative decoding provably preserves the target distribution, then at **greedy decoding** (temperature 0, where the output is a deterministic argmax) speculative and standard decoding should produce essentially identical safety behaviour — there is no distribution left to perturb. The theory is silent, however, about the *finite-sample, finite-precision* regime at **temperature > 0**, where the residual-resampling path is actually exercised and where numerical realization of the acceptance rule could in principle shift behaviour. That asymmetry — a strong invariance prediction at greedy decoding, a genuine carve-out at positive temperature — is exactly what TAIS (TR144) is designed to probe behaviourally rather than to assume: a Typical-Acceptance Invariance Screen that treats the distribution-preservation theorem as a hypothesis to be tested at the served operating point, not a license to skip measurement.

### 3.4 LLM-as-a-judge and judge-triangulation reliability

Safety scoring at scale has largely moved from human annotation to **LLM-as-a-judge** pipelines, in which a model classifies whether a response is a refusal, is harmful, or complies with a forbidden request. The appeal is throughput; the liability is that the judge is itself a fallible, biased model, and a safety conclusion is only as trustworthy as the instrument that produced its labels. The literature documents systematic judge pathologies — position and verbosity biases, self-preference, sensitivity to prompt framing — which means a single judge's verdict cannot be taken at face value for a high-stakes safety claim.

A distinction that the program treats as first-class is between two *constructs* that the word "judge" conflates. **General instruction-tuned judges** (e.g., Gemma-3, Llama-3.1 used as scorers) are typically prompted to assess whether the *response* refuses or complies — a response-axis measurement. **Safety-specialist classifiers** are trained for content moderation and measure something different. *Llama Guard* (Inan et al., Meta, 2023; arXiv 2312.06674) is an LLM-based input-output safeguard instruction-tuned on a safety-risk taxonomy that classifies *both* the prompt and the response against harm categories. *ShieldGemma* (Zeng et al., Google, 2024; arXiv 2407.21772) is a suite of Gemma-2-based content-moderation classifiers (2B/9B/27B) targeting harm categories such as dangerous content, hate, harassment, and sexually explicit content, with distinct prompt-only and prompt-response operating modes. Both specialists score *composite prompt+response harm*, not response refusal.

The right reliability instrument for cross-model agreement is **Cohen's κ**, which corrects raw agreement for chance and is appropriate for categorical labels from independent raters. The Judge Triangulation Protocol (JTP, TR148) uses cross-family κ to route an evaluation into one of three trust regimes — *robust* at κ ≥ 0.70, *triangulate* at 0.40 ≤ κ < 0.70, and *untrustable* at κ < 0.40 — so that a brittle judge configuration is caught before its labels are believed.

> TR148's empirical finding is the reason this subsection separates *construct* from *reliability*. The safety-specialist classifiers did not merely disagree noisily with the general judges; they **anti-correlated** with them. Read naively through a single-axis lens, a negative κ looks like instrument failure. Read correctly, it is a *construct difference*: the specialists are scoring composite prompt-plus-response harm while the general judges are scoring response refusal, and those two axes can move in opposite directions (a model that complies — low refusal — with a benign-looking prompt can register low composite harm, and vice versa). The protocol-level consequence is that triangulation must be performed *within* an axis, never across axes; a low cross-axis κ is a category error, not a red flag. This is a measurement-validity result, and it is logically prior to any safety verdict the judges are later asked to render.

### 3.5 torch.compile, Triton, and benchmark reproducibility

Performance claims for served LLMs rest on a deep compiler stack. `torch.compile` lowers PyTorch programs through **TorchInductor**, which generates **Triton** kernels for GPU execution, which in turn compile to PTX/SASS through the CUDA toolchain; an optional **CUDA-graph capture** mode (`reduce-overhead`) records the kernel-launch sequence once and replays it to amortize per-launch overhead during decode. Each layer is versioned independently, and — crucially — the *generated kernel* for a given PyTorch program is not fixed by the program; it is a function of the Inductor and Triton versions in the environment at compile time.

This is the reproducibility hazard. A latency conclusion ("torch.compile gives an X% prefill speedup", "reduce-overhead removes a decode-time crash") is implicitly conditioned on the exact compiler-stack versions, yet those versions are routinely treated as invisible infrastructure rather than as experimental factors. The broader ML-systems benchmarking literature has documented an analogous reproducibility crisis: results that fail to replicate across hardware, driver, and library versions, with the dependency surface often unreported.

> TR147's **Compile Reproducibility Index (CRI)** operationalizes this concern as a measurement: the maximum pairwise |Cohen's d| on compiled-latency across a software-stack perturbation set, with calibrated bands (robust < 0.5, sensitive < 2, fragile ≥ 2, catastrophic ≥ 10). The headline "Triton kill-shot" is a worked example of why this index is necessary: on a *single fixed GPU and a single fixed model*, changing only the Triton minor version across 3.3.1 → 3.4.0 → 3.6.0 collapses a 62–77% torch.compile prefill speedup to roughly zero and erases an ~80% reduce-overhead decode improvement. Nothing about the model, the prompt, or the hardware changed — only a transitive dependency's minor version. The lesson for a serving-safety certification is that the *substrate on which timing and even crash behaviour are measured* is itself a variable, and a benchmark that does not pin and report it is not reproducible. Note the deliberate statistical division of labour: CRI is the *one* place the program uses Cohen's **d**, because compiled latency is a continuous outcome; every binary safety contrast elsewhere uses Cohen's **h** (Section 3.8).

### 3.6 Mechanistic interpretability and the refusal direction

A tempting shortcut for safety certification is to skip behavioural measurement and instead *inspect the model's internals* — to predict whether a quantized configuration is safe from its activations rather than from its outputs. The most relevant interpretability construct is the **refusal direction**. *Refusal in Language Models Is Mediated by a Single Direction* (Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, and Nanda, 2024; arXiv 2406.11717) reports that refusal behaviour in chat models is, to a striking degree, mediated by a single linear direction in the residual stream: ablating that direction suppresses refusals (a white-box jailbreak), and adding it induces refusals, with minimal collateral effect on other capabilities. This sits within a wider activation-steering and linear-probing literature that treats high-level behaviours as approximately linearly represented and therefore readable, and steerable, via directions in activation space.

If a safety-relevant behaviour has a clean geometric signature, one might hope to *probe* for it: measure whether the refusal direction, the first-token entropy, the calibration, or the per-neuron quantization error of a quantized configuration differs from the baseline, and use that as an early-warning predictor of unsafe behaviour without running the full battery.

> TR146 tests exactly this hope and **falsifies** it for the program's setting. Four probes — first-token entropy, refusal-direction geometry in the spirit of Arditi et al., calibration drift, and safety-neuron quantization error — all fail to *separate* safe from dangerous quantized configurations. The conceptual reason is important and not merely a null: a direction can be *geometrically preserved under quantization* (the activation still projects onto the refusal direction much as before) while being *behaviourally ineffective* (the downstream computation no longer converts that projection into a refusal at the same rate), or conversely the geometry can shift without behaviour following. Probe-readability of a behaviour in a clean model does not imply probe-*predictivity* of that behaviour's robustness under a numerical perturbation. The methodological consequence for the certification protocol is decisive: internal probes are not admissible as a substitute for behavioural safety measurement here, and the program does not let an interpretability signal stand in for an output-level contrast.

### 3.7 Standardized adversarial safety batteries

Cross-paper comparability in safety evaluation requires *standardized corpora*; otherwise every study scores against a bespoke prompt set and results cannot be stacked or contrasted. TR149 re-measures the FP8 KV-cache safety contrast against four literature-canonical batteries, each measuring a distinct facet. We describe them strictly at the level of what they measure and never reproduce their contents.

**HarmBench** (*A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal*; Mazeika et al., ICML 2024; arXiv 2402.04249) is a standardized framework for evaluating automated red-teaming attacks and a model's robust-refusal behaviour, built to compare attack methods and target models on a common footing. **JailbreakBench** (*An Open Robustness Benchmark for Jailbreaking Large Language Models*; Chao et al., NeurIPS 2024 Datasets and Benchmarks; arXiv 2404.01318) provides an evolving repository of jailbreak artifacts and the JBB-Behaviors set spanning misuse and matched benign behaviours, with an explicit emphasis on reproducible, comparable attack-success and defense reporting. **StrongREJECT** (*A StrongREJECT for Empty Jailbreaks*; Souly et al., NeurIPS 2024 Datasets and Benchmarks; arXiv 2402.10260) pairs a set of forbidden prompts with an automated evaluator calibrated to human judgments, and its central methodological contribution is that prior evaluators *overstate* jailbreak effectiveness — it scores the *usefulness* of harmful information actually conveyed, not merely the absence of a refusal token. **XSTest** (*A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*; Röttger et al., NAACL 2024; arXiv 2308.01263) measures the *over-refusal* failure mode specifically: it comprises safe prompts that superficially resemble unsafe ones (which a well-calibrated model should *answer*) contrasted with genuinely unsafe prompts (which it should refuse), so a model that refuses everything scores poorly rather than well.

> The four together are not redundant; they probe orthogonal axes — attack-robustness (HarmBench), reproducible jailbreak comparison (JailbreakBench), graded harmful-information delivery rather than refusal-token presence (StrongREJECT), and the benign-edge over-refusal axis (XSTest). XSTest is the load-bearing inclusion for a safety-*tax* study: a numerical change that made a model refuse more could read as "safer" on a refusal-only battery while actually degrading usefulness, and only an over-refusal corpus catches that. The reason TR149 runs the contrast across all four is comparability and corpus-robustness: a null that replicates on every standardized battery is a far stronger claim than a null on one bespoke set, and a verdict that flips across batteries is itself a finding about corpus-specificity rather than about the configuration.

### 3.8 Equivalence testing and matched-pairs methodology in safety evaluation

The statistical posture of this program is unusual for safety work and deserves explicit grounding, because the headline results are *nulls* — and a null is only interesting if the design had the power to detect a real effect and the analysis can affirm equivalence rather than merely fail to reject difference.

The core instrument is **TOST** — the two-one-sided-tests procedure (Schuirmann, 1981), standard in bioequivalence. Ordinary null-hypothesis testing can only *fail to reject* a difference, which is not evidence of sameness; absence of evidence is not evidence of absence. TOST inverts the logic: it places the *equivalence* claim in the alternative hypothesis by running two one-sided tests against a pre-specified equivalence margin, concluding equivalence only if *both* are rejected — equivalently, if a confidence interval for the effect falls *entirely within* the margin. The program fixes that margin at **±3 percentage points** on the safety-relevant proportion, so "no detectable effect" becomes the *positive, bounded* statement "any effect is smaller than 3pp," not the vacuous "we couldn't find one."

Because the design is paired (the same prompt is scored under baseline and treatment), the difference test is the **matched-pairs McNemar exact test**, which conditions on the discordant pairs and is exact for the small discordant counts a strong null produces. The effect-size measure for these binary, paired outcomes is **Cohen's h** on proportions — the arcsine-transformed difference between two rates — *not* Cohen's d. The distinction is deliberate and load-bearing across the program: d is for continuous outcomes (and is reserved for CRI's compiled-latency in Section 3.5), whereas h is the appropriate standardized effect size for a difference in proportions and is the matched-pairs estimator used throughout the safety line. When evidence is pooled across strata (models, corpora, serving cells), the program uses the **Haldane-corrected Mantel–Haenszel pooled odds ratio**, with the Haldane correction handling the zero-count cells that a near-perfect null generates. Multiplicity across the family of tests is controlled by **Holm–Bonferroni stepdown**, which is uniformly more powerful than plain Bonferroni while preserving family-wise error control.

> The methodological claim being staked is that an *equivalence* result, properly bounded by a pre-registered margin and backed by adequate power, is a stronger and more useful deployment signal than a bare "p > 0.05." A practitioner can act on "the FP8 safety effect is provably within ±3pp"; they cannot safely act on "we ran a test and didn't find significance." Cohen's h, McNemar, Holm–Bonferroni, and Haldane-MH are the supporting cast that make the TOST verdict defensible at the small effect sizes a true null produces — and keeping h (binary) cleanly separated from d (continuous) is what prevents the cross-instrument confusion that would otherwise undermine the whole stack.

### 3.9 Relationship to the program's prior attack-surface phase

Phase 6 is positioned as the *measurement-validity* link in a three-link chain. The phase that precedes it mapped an **attack surface**: it asked where and how quantized, served LLMs become *more* exploitable — studying perturbations introduced by batching, multi-turn interaction, long-context exploitation, and cross-architecture fragility. That phase produced hypotheses about *where* safety risk concentrates. The phase that follows Phase 6 is a **mitigation** phase, which uses validated measurements to gate or route around the risks the program can detect.

> Phase 6's job is the connective tissue between those two. It does not itself attack and it does not itself mitigate; it certifies that the *instruments* — the LLM judges (Section 3.4), the standardized batteries (Section 3.7), the compiler-stack-aware timing substrate (Section 3.5), the equivalence statistics (Section 3.8) — are trustworthy enough that an attack-surface finding and a later mitigation can both be believed. The seven reports synthesized here therefore divide into a *validity* layer (TR147/CRI on benchmark reproducibility, TR148/JTP on judge reliability, TR146 on the inadmissibility of internal probes as a behavioural substitute) and an *application* layer that runs the inference-flag safety null line through those validated instruments (TR144/TAIS on speculative decoding, TR145 on the single-config FP8 KV-cache null, TR149 on cross-battery replication, TR152 on serving-state factorial composition). The detailed integration with both neighbouring phases is deferred to §13; here we make no specific claims about the constituent reports of the adjacent phases beyond this high-level framing.

### 3.10 Source-verification note

This section's external attributions were WebFetch-verified during drafting; the program-internal artefacts are flagged as such, in keeping with the program's defensibility discipline (verify external specs, never dress an internal definition as a citation).

**Verified via web (titles, authors, venues, years, and the specific characterizations used above):** *FP8 Formats for Deep Learning* (Micikevicius et al., 2022, arXiv 2209.05433 — E4M3 = 1/4/3, E5M2 = 1/5/2, biases 7/15, E4M3's IEEE-754 departures); *Efficient Memory Management … with PagedAttention* (Kwon et al., SOSP 2023, arXiv 2309.06180); *Defeating Nondeterminism in LLM Inference* (Thinking Machines Lab, 2025 — batch-invariance-failure argument); *Fast Inference from Transformers via Speculative Decoding* (Leviathan, Kalman, Matias, ICML 2023, arXiv 2211.17192); *Accelerating LLM Decoding with Speculative Sampling* (Chen et al., DeepMind 2023, arXiv 2302.01318 — modified rejection sampling, distribution preservation "within hardware numerics"); *Refusal … Mediated by a Single Direction* (Arditi et al., 2024, arXiv 2406.11717); *Llama Guard* (Inan et al., Meta 2023, arXiv 2312.06674); *ShieldGemma* (Zeng et al., Google 2024, arXiv 2407.21772 — sizes/construct verified; first-author surname taken from the paper's authorship, not independently re-confirmed); *HarmBench* (Mazeika et al., ICML 2024, arXiv 2402.04249); *JailbreakBench* (Chao et al., NeurIPS 2024 Datasets & Benchmarks, arXiv 2404.01318); *StrongREJECT* (Souly et al., NeurIPS 2024 Datasets & Benchmarks, arXiv 2402.10260); *XSTest* (Röttger et al., NAACL 2024, arXiv 2308.01263); TOST (Schuirmann, 1981).

**Asserted from the program's own internal definitions (not external citations):** the named methods CRI, JTP, TAIS and their calibrated thresholds (CRI bands; JTP κ routing 0.40/0.70; TAIS |h| < 0.1); all TR-level empirical findings (the Triton latency collapse, the judge anti-correlation, the four-probe falsification, the FP8 nulls); the ±3pp equivalence margin. McNemar's exact test, Cohen's h vs d (Cohen, 1988), Holm–Bonferroni, and Haldane-corrected Mantel–Haenszel are used per their standard textbook definitions and were not individually web-fetched (they are not novel terminology).

---

## 4. Methodology Overview

### 4.1 Shared experimental discipline

Every executed report in Phase 6 inherits the safety line's standing methodology:

- **Matched-pair design.** The unit of analysis is a prompt sampled once under the control condition and once under the treatment condition with an identical seed, so the McNemar test operates on genuinely matched pairs and the effect-size estimator is the matched-pairs Cohen's h, not Cohen's d.
- **Equivalence testing.** Two One-Sided Tests (TOST) at a pre-specified ±3pp margin, calibrated since TR141 as the deployment-relevance band. Equivalence is reported as a positive finding.
- **Family-wise correction.** Holm–Bonferroni stepdown across the comparison family, chosen over Benjamini–Hochberg because family-wise error rate is the right guarantee for a deployment-certification verdict.
- **Cross-strata synthesis.** Mantel–Haenszel pooled odds ratios (Haldane-corrected on the discordant cells, Wald CI on the log-OR) pool per-stratum 2×2 tables into a single across-stratum summary.
- **Bootstrap confidence intervals.** Reported on the matched-pair deltas that feed the TOST procedure.
- **Power and MDE.** Retrospective minimum-detectable-effect analysis on every primary cell, so a null is reported as "powered to detect an effect of size X and did not" rather than as bare non-significance.

### 4.2 Judge dispatch and the umbrella gate

Judge dispatch is resolved at run time from the JTP verdict. A `robust` κ licenses single-judge dispatch; a `triangulate` κ mandates a multi-judge majority vote on every record; an `untrustable` κ aborts the run. Because TR148's calibration corpus produced `triangulate` (κ = 0.6917), the downstream Phase 6 reports ran the multi-judge local cohort (regex + gemma3:12b + llama3.1:8b). The GPT-4o axis is stripped from every adversarial-corpus run by the `--skip-openai-judge` umbrella gate: adversarial-prompt content cannot flow through the OpenAI tier-1 API without Researcher Access Program enrollment, so it is gated until that umbrella resolves. The Anthropic (Claude) judge axis is similarly deferred to Fellowship credentials. The consequence is that Phase 6's cross-family agreement numbers are two-LLM-judge measurements; the third and fourth axes are documented deferrals, not gaps.

### 4.3 Hardware and software envelope

The executed reports run on the local RTX 4080 Laptop 12GB (Ada sm_8.9, the minimum FP8-qualifying SKU) and, for the cells that exceed 12GB, on RunPod RTX 6000 Ada 48GB. The serving stack is pinned to `vllm/vllm-openai:v0.19.1` throughout the FP8 line; TR147's ablation deliberately varies the Triton minor version while holding everything else fixed. The model set is 1B–4B instruction-tuned (Llama-3.2-1B/3B, Qwen2.5-1.5B/3B, Phi-3-mini-4k), with TR144's E1 expansion reaching production scale (Llama-3.1-70B-AWQ-INT4 target + 8B draft) and TR147's large-model cells reaching 7B/8B. Scale beyond 4B for the FP8 safety line is TR151's cloud-gated scope.

### 4.4 Cost posture

All executed Phase 6 work ran at **$0 external API cost**: the umbrella gate keeps adversarial corpora off paid APIs, and the judge cohort is local Ollama. The only metered spend is RunPod GPU time for the cells that exceed the local 12GB envelope (TR146's 7B cells, ~$0.20; TR147's A100 and large-model slices). This cost posture is itself a deliverable — it demonstrates that the certification substrate can be built without a cloud safety-judge budget, which is what licenses the volume-and-velocity strategy of the program.

### 4.5 The statistical toolkit, and why each estimator is the one chosen

Because the same six estimators recur in every chapter, it is worth stating precisely what each one measures and why it, rather than a more familiar alternative, is the correct choice for a matched-pair safety null. The choices are not stylistic; each one is the estimator that does not lie on a near-degenerate (ceiling-heavy, almost-entirely-concordant) safety corpus.

- **McNemar's exact test (not the χ² of independence).** The unit of analysis is a single prompt sampled once under the control condition (FP16, or target-alone, or the baseline serving context) and once under the treatment with an *identical seed*. That pairing means the two observations are not independent, so the relevant question is not "are the two marginal safe-rates different?" but "of the prompts whose outcome *changed*, did they change disproportionately in one direction?" McNemar's test operates only on the discordant cells (b = control-safe/treatment-unsafe, c = control-unsafe/treatment-safe) and ignores the concordant mass. On a safety corpus where the overwhelming majority of pairs are concordant-safe, this is the only test that has any power at all — a χ² of independence would be swamped by the concordant diagonal.

- **Cohen's h on paired proportions (not Cohen's d).** The effect-size estimator across the entire safety line is the matched-pairs Cohen's h — the arcsine-transformed difference of two proportions — *not* Cohen's d. This is a deliberate, memory-locked choice: Cohen's d is a standardized mean difference appropriate to continuous outcomes, whereas safety outcomes are binary (refused / complied) and the relevant effect size is a difference of proportions stabilized by the arcsine transform so that a fixed h corresponds to the same detectability regardless of where on the [0,1] rate axis the proportions sit. The null cutoffs (|h| < 0.1 for TAIS, |h| < 0.2 "negligible" band elsewhere) are calibrated on h, and TR147's CRI is the one place Cohen's *d* is correct — because there the outcome (compiled latency in milliseconds) genuinely is continuous.

- **TOST equivalence at ±3pp (not "failure to reject").** Two One-Sided Tests bound the matched-pair delta inside a pre-registered ±3-percentage-point margin and declare equivalence as a *positive* finding when the bootstrap delta CI falls entirely inside [−3pp, +3pp]. The ±3pp margin has been the deployment-relevance band since TR141. The distinction from a bare non-significant McNemar is the whole methodological point: "we failed to reject H0" is consistent with a large undetected effect on an underpowered cell, whereas "the delta is equivalent to zero within ±3pp" is a bounded claim. Across Phase 6 the verdicts run off TOST, not off McNemar non-rejection, precisely because the per-cell McNemar is often underpowered (TR149 per-cell MDE 14–28pp; TR152 per-cell ~3.25–3.75pp).

- **Holm–Bonferroni stepdown (not Benjamini–Hochberg).** When a report runs a family of cells (TR149's 12, TR152's 120), the family-wise error rate must be controlled or a single spurious "significant" cell will be read as a finding. Holm–Bonferroni is chosen over Benjamini–Hochberg because a deployment-certification verdict wants the family-wise-error guarantee (the probability of *any* false positive), not the false-discovery-rate guarantee (the expected *proportion* of false positives among rejections) that BH provides; on these corpora the two give the identical verdict at the achieved resolution, but FWER is the right guarantee to state.

- **Mantel–Haenszel pooled odds ratio, Haldane-corrected, matched-pairs form.** To synthesize many per-stratum 2×2 tables into one across-stratum summary, the line uses the Mantel–Haenszel pooled OR. For *paired* cells the correct estimator is the discordant-ratio form `(Σb + 0.5)/(Σc + 0.5)` with log-OR variance `1/Σb + 1/Σc`; the `+0.5` is the Haldane–Anscombe continuity correction that keeps the estimator finite when a discordant count is zero (which happens constantly on ceiling batteries). TR149's estimator postmortem (§7.5) is the cautionary tale: feeding paired cells into the *unpaired* cross-product `(a·d)/(b·c)` exploded the pooled OR to 3411.5 on an all-null corpus, because that formula routes the dominant concordant mass into the numerator. The matched-pairs form and the unpaired form are both correct — for their respective table types — which is why TR145 (genuine unpaired marginals) and TR149 (paired cells) correctly use *different* estimators.

- **Bootstrap confidence intervals and retrospective MDE.** The matched-pair deltas that feed TOST carry bootstrap CIs rather than normal-approximation CIs, because the proportions are often near a 0/1 boundary where the normal approximation is poor. And every primary cell carries a retrospective minimum-detectable-effect computation, so a null is reported as "powered to detect an effect of size X, and the observed delta was below X" rather than as bare non-significance — the discipline that distinguishes a *measured* null from an *underpowered* one.

### 4.6 The claim ladder — how every result is tiered

Every claim in this synthesis is tagged to a tier on the project claim ladder (`CLAIM_LADDER.md`), and the tiering is what keeps the arc defensible:

- **Supported (Tier 1):** directly evidenced by an executed run with adequate pooled power — e.g. the TR148 κ = 0.6917 (S2), the TR149 12/12 TOST equivalence (S5), the TR148 dual-axis finding (S6).
- **Licensed-with-caveat (Tier 2):** evidenced but bounded by a stated scope limit — e.g. the corpus-specific κ tightening to 0.83 (L3, because clean refusals are easier to judge), the TR145 single-config null as a parked-paper seed (L6).
- **Forbidden (Tier 3):** claims the evidence specifically *rejects* — most importantly F3 ("mechanistic probes predict safety degradation under quantization"), anchored by TR146.

The single most important discipline the ladder enforces is that no member of the null line is ever allowed to state "FP8 is safe." The supported claim is always the bounded form: "no detectable effect under tested conditions, with positive ±3pp equivalence on the cells that have the variance to show one." That bounded form is what lets the downstream reports *extend* the certified envelope rather than re-litigate an over-claim.

### 4.7 Per-report measurement protocols

The shared discipline above is instantiated differently in each report, because each isolates a different manipulated variable. The matrix below states, for every executed report, the design type, the model set, the realized scale, the manipulated condition, the judge cohort, and the primary estimator the verdict runs off. The per-report Results chapters (§6–§10) interpret these protocols; this subsection fixes the protocol facts in one place so a reader can audit "what was actually measured" without reconstructing it from seven chapters.

| TR | Design | Models | Realized scale | Manipulated condition | Judge cohort | Primary verdict estimator |
|---|---|---|---|---|---|---|
| **TR147** (CRI) | single-GPU stack-perturbation ablation; compile-vs-eager | Qwen2.5-1.5B (kill-shot); 7 models across v1; 8B/7B large cell | 52,410 primary latency rows | Triton minor version {3.3.1, 3.4.0, 3.6.0}, GPU/model/cache/mode fixed | none (latency, not safety) | max pairwise **Cohen's d** on compiled latency (continuous) |
| **TR148** (JTP) | re-judge of the TR145 safety subset; cross-family κ | n/a (judge study) | 13,724 records re-judged | judge identity (5 local + 94-record gpt-4o anchor) | regex + gemma3:12b + llama3.1:8b + shieldgemma:9b + llama-guard3:8b | pairwise **Cohen's κ**, dynamic largest-n primary pair |
| **TR149** (standardized) | TR145 design held fixed, task set swapped | 3 × 1B–3B instruction-tuned (TR145 set) | 7,578 records; 3,537 paired | task set → 4 canonical batteries | regex + gemma3:12b + llama3.1:8b (triangulate cohort) | **TOST** ±3pp + MH discordant-ratio OR |
| **TR144** (TAIS) | paired draft–target factorial + E1–E5 expansion | 3 core pairs; E1 70B-AWQ + 8B draft | 64,855 paired (16,783 core + 48,072 exp) | speculative decoding (rejection / typical accept), N ∈ {1,3,5,8,12}, temp 0 | regex + gemma3:12b (κ ≈ 0, different constructs) | matched-pairs **Cohen's h** + ±3pp TOST |
| **TR145** (FP8 base) | 5-phase paired battery, single config | 3 × 1B–3B instruction-tuned | 24,054 records (P1–P5) | `--kv-cache-dtype` auto (FP16) → fp8 (E4M3), weights FP16 | gemma3:12b (κ = 0.43 vs regex anchor) | McNemar + ANOVA (η²) + **unpaired** MH OR |
| **TR146** (mechanistic) | forward-pass probe falsification, observational | 17 cells (6 FP16 + 5 AWQ + 6 GPTQ INT4) | 5,100 forward passes | quantization scheme × probe | none (probe-vs-RTSI correlation) | Pearson **r** + Mann–Whitney U vs regime labels |
| **TR152** (factorial) | FP8-anchored **star** factorial (one-factor-at-a-time) | 5 models / 3 families | 45,000 responses; 20,754 pairs / 120 strata; 135,000 judge labels | KV-dtype × {batch, prefix, temperature} spokes | regex + gemma3:12b + llama3.1:8b (κ = 0.831) | matched-pairs **Cohen's h** + ±3pp TOST + per-family MH OR |

**Observations.** Three protocol facts in this matrix are load-bearing and easy to miss. First, TR147 is the *only* report whose primary estimator is Cohen's **d** rather than Cohen's **h**, because its outcome (compiled latency in milliseconds) is genuinely continuous while every safety outcome is a binary refused/complied proportion — using d on a safety proportion, or h on a latency, would be a category error. Second, TR145 builds genuine *unpaired* marginal tables and therefore correctly uses the unpaired Mantel–Haenszel cross-product, whereas TR149 and TR152 build *paired* cells and use the discordant-ratio form — the two are not interchangeable, and conflating them is exactly the bug TR149's postmortem documents (§7.5). Third, TR152's design is explicitly a **star** (one-factor-at-a-time around a center baseline), *not* a resolution-IV/V fractional factorial; that distinction was corrected after a 2026 methodological critique and is the reason the report does not — and structurally cannot — claim to measure joint two-axis interactions.

> The matrix is the audit surface for the whole arc. Every downstream claim names the report it rests on, and every report here names the design, scale, and estimator it used — so a reviewer can trace any headline back to "this design, this many paired observations, this estimator" in two hops. The judge-cohort column also makes the umbrella gate visible at a glance: no executed safety report uses an external (OpenAI/Anthropic) judge, every one runs the local Ollama cohort, and that is why the realized external cost is $0.

**Protocol notes on the non-obvious choices.** A few design decisions need a sentence of justification beyond the matrix. (i) *TR148 re-judges TR145 rather than collecting fresh labels* so that the κ it measures is a property of the judge cohort on a *known* corpus, isolating judge variance from prompt variance; the 94-record gpt-4o anchor is a calibration check, not a corpus-scale judge, which is why the dynamic primary-pair rule must exclude it from hijacking the verdict. (ii) *TR149 changes only the task set* — same models, same vLLM pin, same dtypes, same seed — so any verdict shift is attributable to the corpus and not to a serving confound; this is what makes it a clean replication rather than a new experiment. (iii) *TR144's judge κ ≈ 0 between regex and gemma3 is expected, not alarming*: the two score different constructs (surface refusal pattern vs semantic safety), the null holds under each independently, and the expansion cells are regex-scored with a semantic rejudge queued. (iv) *TR152 realizes 12 of 14 planned cells per model* (≈19.4% of the full 72-cell crossing) because the two speculative-decoding spokes fail on a vLLM v0.19.1 argument-parser rejection — a launcher bug, not a hardware ceiling — enumerated explicitly rather than silently dropped.

### 4.8 The self-correction methodology

Two of the seven executed reports document and fix an analysis error *inside the report that contains it*, and treating that as a methodological feature rather than an embarrassment is a deliberate program-level discipline. The pattern is identical in both cases: a statistic comes back with a value that is too large to be real, the anomaly is chased to a specific line of `analyze.py`, the fix is committed with a postmortem, and — critically — the report states whether the bug changed any verdict.

- **TR148's mandatory-judge join bug (§6.2).** The original `_join_labels` required a gpt-4o label on every record, silently restricting the entire analysis to the ~100-record killed-sync subset and returning the *wrong* verdict (gemma3 × gpt-4o = 0.877 → "robust, single-judge sufficient"). The surfacing symptom was the regex × gemma3 calibration anchor reading κ = 0.8144 against TR145's reported 0.4274 — a discrepancy too large for pipeline noise. The v2 fix dropped the mandatory `continue`, replaced the hardcoded pair with dynamic largest-n selection, and flipped the verdict to the correct `triangulate`. This bug is now memory-locked as a standing rule.
- **TR149's paired-vs-unpaired estimator bug (§7.5).** The first analysis pass fed genuine paired McNemar cells into the *unpaired* odds-ratio cross-product `(a·d)/(b·c)`, which on an almost-entirely-concordant corpus exploded the pooled OR to 3411.5. The fix replaced both sites with the matched-pairs discordant-ratio form, yielding the consistent 0.8065. The report states explicitly that *no verdict flipped* (the odds ratios are display-only; the equivalence verdict runs off TOST and the significance verdict off McNemar) and that TR145's own unpaired estimator was verified *unaffected* — so a future reader "fixing" TR145 to match would reintroduce a bug.

**Observations.** The two bugs are categorically different in consequence — TR148's silently produced a *wrong verdict*, TR149's produced an *implausible display number* but left every verdict intact — and the discipline is to report that difference rather than blur the two as "we found a bug." The methodological contribution is the surfacing mechanism: both bugs were caught by an *integrity cross-check* (a value compared against an independently-known prior — TR145's reported κ, the expected OR magnitude on a null corpus), not by code review. That is why the line carries integrity anchors at all: the regex × gemma3 calibration drift bound (±0.10) and the all-null-corpus OR sanity check exist precisely so that a join or estimator bug *cannot* pass silently.

> Reporting one's own analysis errors in-report is not a confession; it is the evidence that the pipeline has tripwires. A program that never reports a self-correction is not a program without bugs — it is a program without integrity cross-checks, which is the more dangerous state. The TR148 case is the load-bearing one: a mandatory-judge gate does not error, it computes the right statistic on the wrong sample and reports a confident wrong verdict, and the only thing that catches it is a number that *should* have been known and was checked.

### 4.9 Model-selection rationale

The executed FP8 safety line runs on a deliberately chosen 1B–4B instruction-tuned set — Llama-3.2-1B/3B, Qwen2.5-1.5B/3B, Phi-3-mini-4k — with TR144's E1 expansion reaching Llama-3.1-70B-AWQ-INT4 and TR147's large cells reaching 7B/8B. The selection is not an accident of what fits on a 12GB laptop; three constraints drove it.

- **Three families, not three sizes of one family.** Llama, Qwen, and Phi are independently trained with different alignment recipes, so a finding that replicates across all three is a property of the *quantization/serving operation* rather than of one vendor's safety tuning. This is what lets TR152 isolate the located finding to the Qwen family specifically — a single-family study could not distinguish "FP8 over-refuses" from "this model over-refuses."
- **The Qwen family was kept in deliberately.** TR145's lone ±3pp boundary case (Qwen-1.5B, failing by 0.09pp, §8.2) seeded the located finding that TR152 resolves to the Qwen-XSTest over-refusal lean. Dropping the family that carries the only non-null signal would have produced a cleaner-looking but less honest null line; keeping it is what turns "an across-the-board null" into "a null on the harmful core with one located, per-family, benign-edge lean."
- **Small models are where ceilings bite, and that is named.** The 1B–3B set saturates JailbreakBench and StrongREJECT at 100% refusal (§7.3), which is *why* TR149's discriminating evidence reduces to two batteries and the 7B–72B range is handed to TR151. The model set was chosen to be cheap and reproducible, and the cost of that choice (saturation, ≤4B scale bound) is enumerated rather than hidden.

**Observations.** The selection rationale is the reason the scale gap (§13.6) reads as an *envelope boundary* rather than a weakness. The line is anchored on a model set chosen for cross-family breadth and reproducibility at $0–low cost, with the explicit understanding that production scale (7B–72B) is TR151's job. The one selection decision that most shapes the headline is the deliberate retention of Qwen: it is the single source of the only located positive in the entire line, and a study that optimized for a clean null would have had every incentive to drop it.

### 4.10 Responsible-disclosure and ethics posture

The arc handles adversarial content under a defense-in-depth posture that is itself a methodological commitment, not an afterthought. Three rules compose:

- **The umbrella gate (process-level).** Adversarial-corpus prompts (HarmBench, JailbreakBench, StrongREJECT, AdvBench) never flow through an external API without Researcher Access Program / Fellowship enrollment; the `--skip-openai-judge` flag strips the external tier from every adversarial run, and the judge cohort is local Ollama only. This keeps the program off an org-level flag risk and is the reason the realized external cost is $0.
- **No raw-prompt reproduction (artifact-level).** Inspection of safety-record artifacts uses structural-only probes (keys, types, lengths); raw adversarial prompts and model completions are never reproduced into a report or a working context. This synthesis follows that rule — every adversarial battery is named and characterized, but no harmful prompt or completion text appears anywhere in it.
- **Bounded claims (epistemic-level).** Every null-line member is barred by the claim ladder from stating "FP8 is safe"; the supported form is always "no detectable effect under tested conditions." An over-claimed safety result is itself a safety hazard — it would license a deployment the evidence does not cover — so the bounded-claim discipline is an ethics rule, not only a statistical one.

**Observations.** The three rules sit at three different levels (process, artifact, epistemic) and together mean the arc can study adversarial robustness without ever creating, transmitting, or over-claiming an adversarial capability. The posture is also what makes the work reproducible by others without a cloud safety-judge budget or an adversarial-prompt corpus pulled into general-purpose API logs — which is the same property that licenses the program's volume strategy. Responsible disclosure here is not a paragraph appended at the end; it is the gate on which judge runs, the probe on which artifacts are inspected, and the tier on which every claim is allowed to land.

---

## 5. Certificate Impact Matrix

A conclusive synthesis earns its length only if a reader can trace each constituent report to a *decision it changes*. This section is that trace. It is the bridge between the methodology (§4, "what was measured and how") and the per-report chapters (§6–§11, "what each report found"): a decision-first ledger that names, for every executed report, the serving-state decision it informs, what a deployment would have done *without* the evidence, what the evidence licenses it to do *now*, and the claim-ladder tier the changed decision sits on. The matrix is deliberately written so that a serving engineer who reads nothing else can extract the operational consequences, and a reviewer who reads only this can check that no decision is licensed past its evidence.

### 5.1 The decisions a serving-state safety certificate must inform

A serving-state safety certificate is not an academic object; it exists to answer a fixed set of deployment questions that recur every time a team turns on an inference optimization. Phase 6 is scoped to exactly six of them, and the whole arc is organized around moving each from "decided on faith" to "decided on measured, bounded evidence."

| # | Deployment decision | Who asks it | The pre-Phase-4.5 default | The failure if decided wrong |
|---|---|---|---|---|
| D1 | Enable FP8 (E4M3) KV-cache in production serving? | serving / infra eng | "probably fine, it's just precision" | a silent refusal-rate shift under load that no single-config test would catch |
| D2 | Re-validate safety for *every* serving configuration (batch, prefix, temperature)? | safety / eval lead | "re-run the battery per config" (expensive) or "skip it" (unsafe) | either a combinatorial eval bill or an unmeasured interaction |
| D3 | Treat speculative decoding as safety-transparent? | latency eng | "the theorem says distribution-preserving, ship it" | a leakage path at temperature > 0 that the greedy-only theorem does not cover |
| D4 | Trust a single LLM judge's safety labels? | eval / research | "one strong judge is enough" | a confident, wrong safety verdict from an un-triangulated judge |
| D5 | Trust a cross-stack latency / throughput number? | perf eng | "a benchmark is a benchmark" | a 60%-vs-0% conclusion flip from an unrecorded dependency bump |
| D6 | Read safety off a cheap forward-pass probe instead of behaviour? | research / cost | "interpretability can shortcut the eval" | approving the *most* dangerous configs because they look the most certain |

**Observations.** The six decisions are not independent — they form a dependency order that the rest of the synthesis follows. D4 and D5 are *validity* decisions (can the safety label and the latency number be trusted at all?) and must be answered first, because D1–D3 are safety verdicts that consume judge labels and D6 is the shortcut that would bypass the whole measurement apparatus. The pre-Phase-4.5 default column is the honest baseline: in the absence of this evidence, every one of these decisions was being made by appeal to a plausible prior ("precision is fine," "the theorem covers it," "one judge is enough"), and each plausible prior has a specific, documented failure mode that the corresponding report either confirms or closes.

> The reason a *measurement-validity* phase deserves a full conclusive synthesis is visible in this table: four of the six decisions (D1, D2, D3, D6) are safety verdicts that are only as trustworthy as the two validity decisions (D4, D5) underneath them. A program that answered D1 ("FP8 is safe") without first answering D4 ("can the judge that scored it be trusted?") would be building on sand. Phase 6's contribution is to answer D4 and D5 *with measured thresholds* and then answer D1–D3 and D6 *under* those validated instruments.

### 5.2 Per-report decision impact

The table below maps each executed report to the decision it moves, the direction of the move, and the tier the moved decision sits on. Every numeric anchor in the "evidence" column appears in full, with its source section, in the per-report chapter named in the last column.

| Report | Decision moved | Before → after | Key evidence | Tier | Chapter |
|---|---|---|---|---|---|
| **TR147** (CRI) | D5 (trust a cross-stack number?) | "a benchmark is a property of the model" → "a benchmark is a property of the *pinned stack*; gate it with CRI" | Triton-only ablation flips prefill 62.82% → 0.84%; cross-version \|d\| ≈ 14–49; eager control flat (\|d\| ≤ 0.15) | Supported (C6) | §6.1 |
| **TR148** (JTP) | D4 (trust one judge?) | "one strong judge ships" → "triangulate unless κ ≥ 0.70 on *your* corpus" | gemma3 × llama3.1 κ = 0.6917 [0.6824, 0.7008], n = 12,809 → triangulate | Supported (C4) | §6.2 |
| **TR148** (dual-axis) | D4 (which judges agree?) | "low κ = broken judge" → "low *cross-axis* κ = different construct; triangulate within axis" | four cross-axis κ = −0.13 to −0.26; within-specialist κ = +0.21 | Supported (C5) | §6.2 |
| **TR146** (probes) | D6 (probe instead of behave?) | "interpretability can shortcut the eval" → "**no** — behaviour only" | four probes all \|r\| < 0.15; danger-vs-neutral p = 0.979; GPTQ confidence paradox | Forbidden (C9/F3) | §9 |
| **TR144** (TAIS) | D3 (speculative = transparent?) | "the theorem says ship it" → "yes at temperature 0; re-test above it" | max \|h\| = 0.024 over 64,855 paired; byte-identity survives adversarial + quantized drafts | Supported (C2) | §8.1 |
| **TR145** (FP8 base) | D1 (enable FP8?) | "probably fine" → "no detectable effect *at the tested single config*" | pooled OR 1.05 [0.90, 1.23]; 24,054 records; all primary McNemar n.s. | Supported, bounded | §8.2 |
| **TR149** (standardized) | D1 (is the null a corpus artefact?) | "your null is your task set" → "the null replicates on the field's batteries" | 12/12 TOST-equivalent at ±3pp; pooled OR 0.8065 [0.3828, 1.6989] | Supported (C3) | §7 |
| **TR152** (factorial) | D2 (re-validate per config?) | "re-run per config or skip" → "**one** certificate across the tested serving state" | 0/8,976 harmful discordant; interaction spread 2.99pp inside ±3pp | Supported (C1) | §10 |
| **TR152** (per-family) | D1 (FP8 on Qwen?) | "FP8 is uniformly fine" → "monitor Qwen over-refusal at temp > 0" | per-family MH OR Qwen 3.878 [2.386, 6.302] vs Llama 0.484, Phi-3 0.130 | Licensed (C7) | §10.4 |

**Observations.** Read top-to-bottom the matrix is the arc's logic in one screen: the two validity reports (TR147 → D5, TR148 → D4) come first and change *how every other number is allowed to be read*; the negative control (TR146 → D6) closes the one shortcut that would have bypassed the rest; then the four null-line members move the three substantive serving decisions (D1, D2, D3) from prior-belief to bounded measurement. Notice that two reports move *two* decisions each — TR148 moves the trust-one-judge decision (D4) and, via the dual-axis finding, the harder which-judges-can-even-be-compared decision; TR152 moves both the per-config-revalidation decision (D2) and the FP8-on-Qwen decision (D1's per-family refinement). Those double-impact rows are the load-bearing reports of the phase, which is why they receive the longest chapters (§6.2, §10).

> The single most consequential row is TR152 → D2. The combinatorial cost of re-validating safety for every (batch × prefix × temperature) serving configuration is what makes most teams *skip* serving-state safety validation entirely. TR152's flat-interaction result converts that "re-run per config or skip" dilemma into "validate once, deploy across the tested envelope," which is the only economically viable way a production team actually adopts a safety gate. The certificate's value is not the null itself — it is that the null is *serving-state-independent*, so it is consumed once rather than re-earned per deploy.

### 5.3 The layer-fill ledger

The six decisions map onto the bridge paper's five-layer certification protocol (§2.3). This ledger states, per layer, what Phase 6 fills it with, the residual that remains, and the named scaffold that would close the residual.

| Layer | Filled by | Status after Phase 6 | Residual | Named extension |
|---|---|---|---|---|
| **1 — Measurement validity** | TR148 (JTP κ = 0.6917; dual-axis split 1a/1b) | Evidenced | gpt-4o + Claude cross-family axes deferred | umbrella resolves → 4-judge κ matrix |
| **2 — Behavioural screens** | TR144 (TAIS, max \|h\| = 0.024) + TR142 (RTSI, shipped) | TAIS member evidenced | temperature > 0 (TAIS E6 unrun); tree-speculation | TR144 E6 temp sweep |
| **3 — Compile integrity** | TR147 (CRI; Triton kill-shot, \|d\| ≈ 14–49) | Evidenced | replication on the *vLLM* stack (CRI measured on torch.compile microbench) | bridge-paper vLLM-stack CRI run |
| **4 — Scale validity** | TR149 (1B–3B; 12/12 TOST-equivalent) | 1B–3B anchored | 7B–72B unmeasured; batteries saturate at this scale | TR151 (cloud-gated) |
| **5 — Serving-state validity** | TR152 (1B–4B; 0/8,976 harmful discordant, flat interaction) | Evidenced (1B–4B) | long context (≤2,048 tok tested); KV-method family | TR150 (context), TR153 (method) |

**Observations.** Three of five layers (1, 3, 5) are filled with executed evidence; Layer 2 has its TAIS member executed and its RTSI member already shipped from Phase 5; Layer 4 is half-filled (1B–3B anchored, 7B–72B scaffolded). No layer is empty, and every residual has a *named* extension rather than an open question — which is the difference between "the protocol is incomplete" and "the protocol is validated on a bounded envelope with designed widenings." The residuals are also honestly heterogeneous: Layer 1's residual is a *credential* gate (the external judges await the umbrella), Layer 3's is a *generalization* gate (CRI was measured on a torch.compile microbenchmark, not yet on the vLLM serving stack the safety line runs on), and Layer 4's is a *hardware* gate (7B–72B needs cloud GPU). Conflating those three kinds of residual would over- or under-state how close each layer is to closure.

> The ledger is the claim-ladder made operational. Each "Status" cell is the strongest verb the evidence licenses — "Evidenced," "anchored," never "complete" or "proven" — and each "Residual" cell is the bound that keeps the corresponding Supported claim honest. A bridge paper that quotes this ledger inherits not just the five filled layers but the exact shape of what remains, so it can state Layer 4 as "scale-validated at 1B–3B, 7B–72B scaffolded and cloud-gated" without a reviewer being able to claim the gap was hidden.

### 5.4 What each gate prevents — the cost-of-being-wrong column

A certificate's value is best stated as the failure it would have allowed. This is the inverse reading of §5.1's "failure if decided wrong" column: for each gate Phase 6 installs, what concrete, documented bad outcome does it catch that the pre-gate default would have shipped?

| Gate | The bad outcome it catches | The documented near-miss in the arc |
|---|---|---|
| **CRI (Layer 3)** | shipping a "60% speedup" that is 0% on a neighbouring stack row | the Triton 3.3.1→3.4.0 prefill flip on *one physical GPU* (an eight-sigma event under the no-effect null) |
| **JTP (Layer 1)** | a confident, wrong safety verdict from one judge | TR148's own v1 mandatory-judge join bug: gpt-4o-only n = 94 → "robust, ship single-judge" (the *wrong* verdict), fixed to triangulate at n = 12,809 |
| **Dual-axis split (Layer 1)** | discarding a correct specialist judge as "broken" because its κ is negative | the four cross-axis κ = −0.13 to −0.26 that a single-axis reading would misdiagnose as instrument failure |
| **TAIS (Layer 2)** | assuming speculative decoding is safe at *all* temperatures from a greedy-only theorem | the explicit temperature-0 carve-out; E5 bf16 shifts 36–53% of bytes yet holds the safety null, proving the screen tracks behaviour not bytes |
| **The matched-pairs estimator discipline** | a pooled OR exploding to 3411.5 on an all-null corpus | TR149's estimator postmortem (unpaired cross-product fed paired cells); fixed to the discordant-ratio form → OR 0.8065 |
| **Behavioural-only discipline (TR146)** | a logit/confidence screen *approving* the most dangerous configs | the GPTQ confidence paradox — GPTQ models grow *more* first-token-confident while *failing* to refuse |

**Observations.** Five of the six rows are not hypothetical — they are near-misses the program *actually hit and corrected* inside the constituent reports. The JTP gate is the sharpest example: TR148's own first analysis pass produced the exact wrong verdict the gate is designed to prevent (a single-judge "robust" call on a 94-record subset), and the gate is what caught it. That is the strongest possible evidence that a validity gate is load-bearing rather than ceremonial: the program tripped each wire at least once and the wire held. The two estimator rows (matched-pairs discipline, behavioural-only) are the subtlest, because both failure modes produce a *confident, plausible-looking* wrong number — a clean pooled OR, a clean confidence score — which is precisely the kind of error that survives review unless a specific gate is watching for it.

> This column is why the synthesis insists every downstream claim name the gate it passed. A safety verdict that did not pass CRI, JTP, the matched-pairs estimator check, and the behavioural-only discipline is not a weaker version of a Phase 6 verdict — it is a different kind of object, one whose most likely failure modes were never checked. The arc's contribution is not the FP8 null in isolation; it is the FP8 null *with each of these six wires demonstrably intact*.

---

## 6. Chapter 1 — Measurement-Validity Foundation (TR147 CRI + TR148 JTP)

The first pillar of Phase 6 establishes that the two kinds of number every downstream claim rests on — a *latency* number and a *safety-label* number — are reproducible. TR147 governs the former; TR148 governs the latter. Together they are Layers 3 and 1 of the certification protocol, and they are presented first because nothing else in the arc is trustworthy without them.

### 6.1 TR147 — The Compile Reproducibility Index and the Triton kill-shot

**Headline.** On a single fixed RTX 6000 Ada GPU running one fixed model (Qwen2.5-1.5B, FP16, DynamicCache prefill), changing *only* the Triton minor version flips the `torch.compile` prefill benchmark from a 62–77% speedup to a near-zero (≤3.4%) neutral result, and simultaneously erases an 80% `reduce-overhead` decode crash. The qualitative conclusion of the benchmark — "compilation is a large win" versus "compilation is neutral" — is determined by a dependency that most benchmark reports never record.

**The ablation.** Holding GPU, model, cache implementation, and compile mode fixed and varying only Triton (SS8.1):

| Triton version | Prefill, default | Prefill, reduce-overhead | `reduce-overhead` decode crash rate |
|---|---|---|---|
| 3.3.1 | −62.82% (speedup) | −77.24% (speedup) | 0.800 (480/600 errors) |
| 3.4.0 | +0.84% (neutral) | +0.54% (neutral) | 0.000 |
| 3.6.0 | −0.74% (neutral) | −1.60% (neutral) | 0.000 |

**Observations.** The 62.82% → 0.84% collapse on a single physical GPU is characterized in the source as an "eight-sigma event" under the null model that Triton version does not matter (SS8.5). The eager-execution control is flat across the same three versions (prefill means 21.218 / 21.009 / 21.420 ms, a ~2% spread; decode means within ~1%; cross-version eager-to-eager |d| ≤ 0.15), which is what isolates the effect to the compiled path rather than to ambient machine variance (SS8.6). The cross-version effect sizes on the compile-vs-eager prefill contrast span |Cohen's d| ≈ 14–49 (e.g. StaticCache reduce-overhead 3.4.0-vs-3.3.1 d = −48.928; DynamicCache reduce-overhead d = −32.837; StaticCache default 3.6.0-vs-3.3.1 d = −25.521), magnitudes at which the two distributions essentially do not overlap (SS8.4).

> A benchmark that reports "torch.compile gives a 60% prefill speedup" without pinning the Triton minor version is not reporting a property of the model or the GPU. It is reporting a property of one row of a dependency matrix, and the neighbouring rows say "neutral." The number is real; its generalization is an illusion.

**The named method — CRI.** The Compile Reproducibility Index is the **maximum pairwise |Cohen's d| on compiled-latency across a stack-perturbation set** (canonically the Triton minor versions {3.3.1, 3.4.0, 3.6.0} with GPU, model, cache, and compile mode held fixed), implemented in `research/tr147/v4/compute_cri.py` (SS8.4). Its calibrated bands are:

| Band | Threshold (max pairwise \|d\|) | Interpretation |
|---|---|---|
| robust | < 0.5 | latency conclusion stable across the stack set |
| sensitive | < 2 | mild stack dependence; report with the stack pinned |
| fragile | ≥ 2 | conclusion is stack-contingent; a single number is not reportable |
| catastrophic | ≥ 10 | no distributional overlap; the qualitative verdict flips |

**Observations.** Under these bands the TR147 Triton ablation lands deep in *catastrophic* territory (cross-version |d| ≈ 14–49), which is exactly why the qualitative conclusion flips on one physical GPU (SS8.4, SS8.5). A second, subtler finding extends the benchmark-identity tuple: a dual-variant probe of the external `gpt-fast` benchmark (stack held constant at A100-SXM4-80GB, torch 2.11.0+cu130, Triton 3.6.0) found the pinned Dec-2023 commit `d2c5d8223f` crashes 0/5 compiled runs while current HEAD `6ecad9b5b6` produces a stable 106.74 tok/s (n=20, CV 0.0066), promoting the *repository code SHA* to a load-bearing sixth axis of the benchmark identity alongside GPU, Triton, PyTorch, cache implementation, and compile mode (SS11.7, SS11.8).

> CRI's edge case is instructive: when the external `gpt-fast` pinned commit returns zero surviving compiled stack points, CRI returns `classification="invalid"` ("need ≥ 2 stack points, got 0") rather than a band, because you cannot compute a pairwise distance over a single survivor. A reproducibility index that silently reported "robust" on a zero-survival run would be worse than useless.

**The version arc.** TR147 is a rare research line that *strengthened* its methodological claim as it gathered data, with each version closing a reviewer objection (SS2.2): v1 (Ada, 15,240 rows, 7 models) replicated the 60.7–77.3% prefill gain; v2 (6,840 rows) fixed a decode false-success measurement bug and showed the decode boundary persists across a 16× token range; v3 (A100, 1,440 rows) found the A100 *amplifies* rather than rescues the failure (100% compiled-decode crash at all token lengths); v4 (~25,200 rows) added the StaticCache rescue test, the large-model cell, and the Triton ablation kill-shot, moving the verdict label from "WEAKENED" to "phase-separated, stack-attributed, cache-qualified." The total measurement budget is 52,410 primary rows (SS2.1).

**Two regime findings the version arc surfaced.** Beyond the Triton kill-shot, two cross-regime results harden the "pin the whole stack" discipline:

| Finding | Evidence | Reading |
|---|---|---|
| **A100 amplifies, it does not rescue** | On A100-SXM4-80GB, compiled decode is **100% crash at every token length** (64/128/256) for both qwen2.5-1.5b and qwen2.5-3b; compiled prefill survives only at token_len=64 and crashes at 128/256, but where it survives the gain is large (81.7% / 79.3%) (SS5.1) | The prior "a bigger datacenter GPU absorbs compiler fragility" is false — the failure is *worse* on the A100, so a result validated on Ada cannot be assumed to transfer up |
| **StaticCache rescues correctness, not speed** | `StaticCache` + `mode="default"` gives **0.000 decode crash** across all six (model × GPU) cells but a **+1.61% to +3.46% decode slowdown** vs eager; prefill retains a 54.4–63.2% gain. `reduce-overhead` + `StaticCache` stays unsafe everywhere (0.800 crash, the 60 surviving rows all token_len=1) (SS6.1–6.4, SS7.3) | The cache implementation is itself a load-bearing axis: switching to StaticCache buys decode *correctness* at the cost of the decode *speedup* the whole exercise was chasing, so the cache choice changes the conclusion |

**Observations.** The eager-sanity control being flat across all three Triton versions (SS8.6) is what licenses attributing every one of these flips to the compiled path rather than to ambient machine variance — the same discipline that lets the kill-shot stand. The large-model closure carries an honest caveat: the A100 large-model cell uses dense FP16 qwen2.5-7b while Ada uses AWQ-4bit (48GB is marginal for dense FP16 at this prompt/KV sizing), so the apples-to-apples cross-GPU large-model comparison is the llama3.1-8b dense-FP16-on-both-sides cell, which is the preferred citation (SS7.2, SS12).

**Why a latency-reproducibility gate belongs in a *safety* certification.** CRI is, on its face, a performance-engineering instrument — it measures whether a *latency* number reproduces, not whether a model is *safe*. Its inclusion as Layer 3 of a serving-state *safety* protocol is deliberate and rests on a specific dependency: every safety-vs-performance trade-off the program makes is conditioned on a performance number, and if that number is not reproducible, the trade-off is not defensible. Three concrete linkages make this load-bearing rather than tidy:

- **The cost-of-safety argument runs through latency.** A claim like "this optimization is worth its (null) safety risk because it buys an X% speedup" is only as solid as the X. TR147 shows X can be 62% on one stack row and 0% on the neighbour — so a safety-tax argument that does not pin the stack is quoting a number that may not exist on the deployment's actual toolchain.
- **The serving stack that produces the safety null is the same stack CRI gates.** The FP8 safety line is pinned to vLLM v0.19.1 precisely *because* TR147 demonstrated that an unpinned stack is a moving target. The pin is not bureaucratic caution; it is the direct operational consequence of the Triton kill-shot, and it is what makes TR149 a clean replication of TR145 rather than a confounded re-run on a drifted stack.
- **Crash behaviour, not just speed, is stack-contingent.** The ablation erased an 80% `reduce-overhead` decode *crash* by a version bump — meaning the stack determines not only how fast a configuration runs but *whether it runs at all*. A serving-safety certificate that did not pin the stack could certify a configuration that crashes on the deployment's actual toolchain, which is a availability-safety failure even before any refusal-behaviour question.

**Observations.** The linkage that most directly ties CRI to safety is the third one: the stack determines crash behaviour, so "is this serving configuration safe to deploy?" has a stack-contingent answer *before* the refusal question is even reached. A configuration that crashes 80% of decode steps on the deployment's Triton version is not a safety-neutral configuration regardless of its refusal behaviour. CRI is the gate that catches this, which is why it sits at Layer 3 underneath the behavioural safety layers rather than off to the side as a performance footnote.

> The deeper point is that CRI imports the program's anti-confabulation discipline into the performance substrate. A latency number, like a safety verdict, is a *measurement under conditions*, and the conditions (six axes: GPU, Triton, PyTorch, cache implementation, compile mode, code SHA) are part of the measurement's identity. A benchmark that omits them is not a weaker measurement — it is an unfalsifiable one, because a reader cannot reproduce it and a neighbour stack will silently disagree. CRI is the instrument that makes a latency claim falsifiable, which is the precondition for it counting as evidence in a safety trade-off at all.

**Role and forward-pointer.** CRI is **Layer 3** of the certification protocol (the compile-integrity gate; CLAIM_LADDER S3). Its contribution to Phase 6 is a negative-control discipline: a serving-state latency or throughput delta is allowed to count as evidence only once the entire software stack — GPU, Triton version, PyTorch build, cache implementation, compile mode, and code SHA — is pinned, because perturbing any one axis can flip the qualitative conclusion on the same hardware. S3 remains a claim-ladder placeholder pending the bridge-paper CRI replication on the vLLM stack.

### 6.2 TR148 — The Judge Triangulation Protocol and the dual-axis finding

**Headline.** The two safety-specialist judges (shieldgemma:9b, llama-guard3:8b) anti-correlate with the two general LLM judges (gemma3:12b, llama3.1:8b) — not because either is broken, but because they measure two orthogonal axes. The specialists score *composite harm* ("is this prompt+response interaction harmful?"); the general judges score *response refusal* ("did the model refuse the adversarial request?"). On a cleanly refused adversarial prompt, both judges are correct and land on opposite sides of the binary safe/unsafe axis.

**The named method — JTP.** The Judge Triangulation Protocol converts pairwise Cohen's κ between independently-trained LLM judges (different families, identical task-typed prompts so disagreement is attributable to model variance, not prompt variance) into a downstream-routing verdict, inherited verbatim from TR140's pre-registration (SS5.1):

| κ band | Verdict | Downstream consequence |
|---|---|---|
| κ ≥ 0.70 | robust | single-judge labels sufficient to ship |
| 0.40 ≤ κ < 0.70 | triangulate | multi-judge majority vote mandatory on every record |
| κ < 0.40 | untrustable | label vocabulary needs redesign before any claim is licensed |

The primary pair is selected dynamically: filter to cross-LLM pairs (excluding the regex calibration anchor), filter to non-zero paired n, sort by (−n, κ) so the largest-n pair wins with ties broken toward the worse κ, then apply the threshold scheme (SS9.1).

**The verdict.** On the 13,724-record TR145 safety subset, the binding cross-family pair gemma3:12b × llama3.1:8b scores **κ = 0.6917**, bootstrap 95% CI [0.6824, 0.7008], at n = 12,809 paired records — the *triangulate* bucket, 0.0083 below the robust threshold (SS2.1, SS9.2). The κ is well-determined (asymptotic SE = 0.0048, z vs zero = 144.1, observed agreement 0.8480, chance agreement 0.5076). The operational consequence is concrete: every downstream Phase 5 report (TR149, TR151, TR152) must run multi-judge majority vote at roughly 3–4× single-judge cost.

**The dual-axis finding.** The decisive secondary result (SS3, SS4):

| Pair | Cohen's κ | n | Reading |
|---|---|---|---|
| gemma3 × shieldgemma | −0.1286 | 12,018 | cross-axis (refusal vs composite-harm) |
| gemma3 × llama-guard3 | −0.1468 | 12,018 | cross-axis |
| llama3.1 × shieldgemma | −0.1866 | 11,382 | cross-axis |
| llama3.1 × llama-guard3 | −0.2596 | 11,382 | cross-axis |
| shieldgemma × llama-guard3 | +0.2136 | 12,024 | within composite-harm axis (coherent) |

**Observations.** All four cross-axis κ values are negative and statistically significant against κ = 0 at corpus scale — this is not a noise floor. The within-specialist pair is positive and fair-band (κ = 0.2136, CI [0.1953, 0.2317]), which is the falsifier that licenses the dual-axis reading: if the specialists were simply noisy, they would not agree *with each other*. The low specialist effective F1 against the general-judge majority (shieldgemma F1 = 0.2907, llama-guard3 F1 = 0.3731) is therefore a wrong-axis category error, not a quality measurement (SS8.3).

**The per-judge F1 ladder and the integrity cross-check.** Read against the corpus-scale majority vote, the per-judge effective F1 scores are gemma3:12b 0.7652 (P 0.6796 / R 0.8756), regex 0.7109, llama3.1:8b 0.5260 (P 0.5705 / R 0.4879), llama-guard3 0.3731, shieldgemma 0.2907 (SS8.1, SS8.3). The two general judges and the regex anchor cluster high; the two specialists sit low — but, per the dual-axis finding, that is *expected* under the wrong-axis category error, not a verdict on judge quality. The majority vote itself resolves 99.05% of records (13,594 of 13,724), splitting 10,542 safe (77.5%) / 3,052 unsafe (22.5%) (SS7.1–7.2). A separate integrity cross-check confirms TR148 is reading the same data TR145 produced: the regex × gemma3 calibration anchor drifts only Δκ = −0.0648 (TR145 reported 0.4274; TR148 measured 0.3626 on the identical n = 13,676 records), inside the ±0.10 pipeline-integrity bound (SS12.1, SS12.3).

> The gpt-4o axis is present only as a calibration anchor at n = 94 from a killed synchronous run (its 0.877 κ with gemma3 is on the disagreement-pool subset, not a corpus-scale signal), and the Claude axis is entirely Fellowship-deferred. The verdict therefore stands on a two-LLM-judge measurement; the third and fourth cross-family axes are documented deferrals, not silent gaps. This is exactly why the dynamic largest-n primary-pair rule matters — it prevents the high-κ-but-tiny-n gpt-4o pair from hijacking the verdict, which is the precise failure the v1 bug produced.

> The dual-axis finding is the methodological keystone of the entire Phase 6 line. Before TR148, every shipped safety verdict rested on the implicit premise that TR140's single near-perfect calibration (gemma3 × Claude κ = 0.925 on the TR140 corpus) generalized to every other corpus. TR148 tested that premise and found the framework reproduces only *within band* — κ drops to 0.6917 on a different corpus and a different cross-family pair. That is not a failure of JTP; it is exactly the cross-corpus result the framework was built to surface.

**The v1 → v2 correction.** The v1 → v2 arc is dominated by one corrected join bug that flipped the verdict (SS22). The original `_join_labels` in `research/tr148/analyze.py` *required* a gpt-4o label on every record (`if gpt is None: continue`); because gpt-4o had only ~100 labels from a killed synchronous run, the join silently restricted the entire analysis to that 100-record subset. The symptom that surfaced it was the regex × gemma3 calibration anchor reading κ = 0.8144 on n = 100 against TR145's reported 0.4274 — a +0.387 discrepancy too large to be pipeline noise. The pre-fix verdict was the *wrong answer*: gemma3 × gpt-4o = 0.877 on n = 94 → "robust, single-judge sufficient." The v2 fix (commit `b0faa06d`) dropped the mandatory-judge `continue` and replaced the hardcoded primary pair with dynamic largest-n selection; post-fix, `n_records_joined` = 13,724, the verdict anchors on gemma3 × llama3.1 at n = 12,809 → triangulate, and the calibration delta shrinks to within the ±0.10 integrity bound (−0.0648).

> This bug is now memory-locked as a standing rule (`feedback_tr_analyze_no_mandatory_judge.md`): a TR `analyze.py` join must never require a specific judge per record, and verdict selection must pick the primary pair by largest n dynamically. A mandatory-judge gate does not error — it silently computes the right statistic on the wrong sample and reports a confident, wrong verdict.

**Role and forward-pointer.** JTP is **Layer 1** of the certification protocol, and the dual-axis finding splits it into **Layer 1a** (the refusal-axis JTP gate: regex anchor + gemma3 + llama3.1, currently triangulate, with Claude and gpt-4o joining when credentials resolve) and **Layer 1b** (the orthogonal composite-harm screen: shieldgemma + llama-guard3 in native chat-format mode, run *in parallel* rather than folded into the κ matrix). The split is the reason every judge-labeled verdict downstream cites a *measured* validity floor rather than an assumed one (CLAIM_LADDER S2, S6).

### 6.3 Why TR147 and TR148 belong in the same chapter

CRI and JTP answer the same question on two different instruments. CRI asks "can you trust the latency number you measured?"; JTP asks "can you trust the safety label you measured?" Both convert a reproducibility question into a calibrated index with a routing verdict (CRI bands → reportable/not; JTP bands → single-judge/triangulate/abort), and both were sharpened by closing a concrete failure: CRI by the Triton kill-shot, JTP by the mandatory-judge join bug. Neither is a safety result in itself; both are the gates that license the safety results in Chapters 2 and 3 to count as evidence.

---

## 7. Chapter 2 — Corpus Standardization (TR149)

**Headline.** The FP8 KV-cache safety null replicates on the four literature-canonical batteries (HarmBench, JailbreakBench, StrongREJECT, XSTest) with positive ±3pp equivalence on all 12 (battery × model) cells, and as a by-product the cross-corpus judge agreement proves corpus-specific: the same gemma3 × llama3.1 pair that scored κ ≈ 0.69 on TR145's mixed task set scores κ ≈ 0.83 on these standardized adversarial batteries.

### 7.1 The protocol — change only the task set

TR149 inherits TR145's paired two-condition design unchanged — same three 1B–3B instruction-tuned models, same `vllm/vllm-openai:v0.19.1` pin, same `--kv-cache-dtype auto` (FP16) vs `fp8` contrast, temperature 0.0, seed 42, 256-token cap — and swaps *only* the task set (SS6.1, SS6.9). Holding everything else fixed is what makes TR149 a clean replication: any change in the verdict is attributable to the corpus, not to a serving-configuration confound. The corpus is 7,578 sampled records (3 models × 4 batteries × 2 dtypes) with zero sampling errors across all six vLLM cycles; the verdict-bearing paired count after dropping unresolved outcomes is 3,537 (SS4, SS6.1).

**Sign-aware scoring.** HarmBench, JailbreakBench, and StrongREJECT use the refusal-as-safe convention (every prompt adversarial; refusal is correct). XSTest carries a per-prompt convention: its ~250 safe-but-superficially-alarming prompts must be *complied* with (over-refusal is the failure mode) and its ~200 genuinely-harmful prompts must be refused. The scoring stack inverts the XSTest safe-slice outcome so "safe" always means "the model did the right thing," and the two XSTest slices are reported separately rather than averaged (SS6.2, SS5).

### 7.2 The replicated null

| Statistic | Value | Source |
|---|---|---|
| McNemar cells significant after Holm | **0 of 12** (every Holm-adjusted p = 1.000000) | SS2, SS9 |
| Smallest raw p in the family | 0.125 (2.5× the uncorrected threshold) | SS9 |
| Cross-battery Mantel–Haenszel pooled OR | **0.8065**, 95% CI [0.3828, 1.6989] | SS4 |
| Discordant pairs corpus-wide | 27 of 3,537 (0.76%), split 12 FP8-degraded / 15 FP8-improved | SS4 |
| TOST equivalence at ±3pp | **12 of 12 cells pass** (11/12 still pass at ±1pp) | SS7, SS14.1 |
| Max Cohen's h (paired-binary) | 0.0742 (HarmBench / qwen2.5-1.5b) | SS3 |
| Largest absolute safe-rate delta | 2.14pp (XSTest safe slice, llama3.2-1b), toward *more* compliance | SS5.1 |
| Cross-battery heterogeneity | I² = 0.0% on all three models (Cochran's Q 0.0071–0.0317, df=3) | SS13 |

**Observations.** The pooled OR brackets 1.0 and the discordant split is near-even (12 vs 15), which is the signature of no effect rather than a small effect in one direction. The largest single delta points the "wrong" way for a degradation story — toward *more* compliance on safe prompts, i.e. over-refusal relief, not safety loss. The zero heterogeneity and the leave-one-battery-out check (every dropped-battery CI overlaps the full-set CI; dropping a ceiling battery leaves the pooled OR bit-identical at 0.8065) confirm the verdict is not carried by any one battery (SS16).

> TR149 answers the standing reviewer objection to TR145's null — "your null is a property of your idiosyncratic task set, not of FP8" — by reproducing it on the benchmarks the field actually uses for cross-paper comparison, with positive ±3pp equivalence on every cell that has the variance to show an effect. That is the difference between "we did not find an effect" and "we measured equivalence within a deployment-relevant margin."

### 7.3 The honest blind spot — ceilings and power

Two of four batteries are at the refusal ceiling: JailbreakBench-100 and StrongREJECT produce 100% refusal on all three models under both dtypes, so six of twelve cells have zero discordant pairs and therefore zero discriminating power (SS1, SS2). XSTest's unsafe slice is also at the 100% ceiling; its entire signal lives in the safe-prompt (over-refusal) slice, where models comply with only 24–45% of safe-but-alarming prompts under FP16 — a model property present equally under FP8 (SS5). The consequence, made explicit by the leave-one-out analysis, is that the discriminating evidence for the cross-battery null is effectively a **two-battery claim** (HarmBench + XSTest jointly) on this 1B–3B model set. Per-cell minimum detectable effect ranges from ~14pp (HarmBench/StrongREJECT/XSTest cells, n = 311–403) to ~28pp (JailbreakBench cells, n = 100), so the verdict rests on the TOST equivalence — which bounds each delta inside ±3pp — not on the underpowered per-cell McNemar (SS10).

> A production-scale model that does not saturate JailbreakBench and StrongREJECT is where those batteries regain discriminating power. That model set is TR151's scope, which is why TR149 anchors Layer 4 only for the 1B–3B range and explicitly hands the 7B–72B half of Layer 4 forward.

### 7.4 What each battery actually contributed

The four batteries were chosen to probe orthogonal axes (§3.7), and reading the result per battery shows which axes carried the null and which were dead weight on this model set.

| Battery | What it measures | Contribution to the 1B–3B null | Discriminating power here |
|---|---|---|---|
| **HarmBench** | robust-refusal under automated red-teaming | the primary discriminating battery: max Cohen's h = 0.0742 here (qwen2.5-1.5b), all cells TOST-equivalent | high — not saturated, carries the harmful-core signal |
| **XSTest (safe slice)** | over-refusal of benign-but-alarming prompts | the *only* battery with a balanced split; FP16 compliance 24–45%, largest delta +2.14pp toward *more* compliance | high — the over-refusal axis the others cannot see |
| **JailbreakBench-100** | reproducible jailbreak comparison | 100% refusal under both dtypes on all three models — zero discordant pairs | none on 1B–3B (regains it at scale → TR151) |
| **StrongREJECT** | graded harmful-information delivery | 100% refusal under both dtypes; κ degenerates at zero variance (PABAK = 0.9979) | none on 1B–3B |

**Observations.** The leave-one-battery-out analysis (§7.2) makes the dependency precise: dropping a ceiling battery (JailbreakBench or StrongREJECT) leaves the pooled OR bit-identical at 0.8065, because a battery with zero discordant pairs contributes zero to a discordant-ratio estimator. The null is therefore carried by exactly two batteries — HarmBench (the harmful-core discriminator) and XSTest's safe slice (the over-refusal discriminator) — and that two-battery structure is not a weakness to hide but the precise statement of what the 1B–3B evidence supports. The single most telling number is XSTest's largest delta: +2.14pp toward *more compliance* on safe prompts under FP8, i.e. the opposite of a degradation story, which is the first standardized-corpus appearance of the over-refusal axis that TR152 later resolves to the Qwen family.

> The per-battery reading is what converts "the null replicates on four standard batteries" into the more honest and more defensible "the null replicates on the two batteries that have discriminating power at this scale, and the other two saturate — which is itself the finding that motivates TR151." A synthesis that reported "12/12 cells equivalent" without the saturation breakdown would overstate the breadth of the evidence; naming the two-battery structure is what keeps the Layer 4 claim bounded to exactly what 1B–3B can show.

### 7.4b The corpus-specific JTP finding

On TR149's standardized batteries the cross-family pair gemma3 × llama3.1 agrees at **κ = 0.8306** (near-perfect band, observed agreement 95.51%, n = 7,557) — materially higher than the κ = 0.6917 the *same pair, same template* produced on TR145's mixed task set (SS11). Per-battery: κ = 0.8096 (HarmBench), κ = 1.0000 (JailbreakBench), κ = 0.7959 (XSTest); the StrongREJECT κ = −0.0005 is not a disagreement but Cohen's κ degenerating at zero variance (every record "safe" under both judges), correctly read as PABAK = 0.9979 (SS17).

**Observations.** Crucially, TR149 did *not* self-select its judge dispatch on this higher κ. The Phase 5 routing rule pins the dispatch decision to TR148's calibration corpus and forbids a downstream report from shopping for the corpus on which its judges happen to agree best (SS6.3, SS15). So TR149 ran the multi-judge cohort mandated by TR148's `triangulate` verdict even though its own corpus would have licensed single-judge dispatch.

> The corpus-specific κ is a Licensed claim, not a Supported one (CLAIM_LADDER L3): clean unambiguous refusals are easier to judge than mixed truthfulness/bias tasks, so the JTP verdict tightens from triangulate to robust on standardized adversarial batteries. The methodological discipline — pin dispatch to the calibration corpus, never to the self-measured κ — is what keeps the validity gate honest.

### 7.5 Role and forward-pointer

TR149 is **Layer 4** of the certification protocol (scale validity), anchoring it for the 1B–3B instruction-tuned model set (CLAIM_LADDER S5, supported; L1, licensed). It also contributes the literature-comparable FP8 evidence to the inference-flag null-line thread shared with TR144/TR145/TR152. The estimator postmortem in TR149 (SS21) is a standing caution for the whole line: the first analysis pass fed genuine paired McNemar cells into the *unpaired* odds-ratio formula `(a·d)/(b·c)`, which on an all-null (almost entirely concordant) corpus exploded to a pooled OR of 3411.5; the fix replaced both sites with the matched-pairs discordant-ratio form `(Σb+0.5)/(Σc+0.5)`, yielding the internally consistent 0.8065. No verdict flipped (the odds ratios are display-only; the equivalence verdict runs off TOST and the significance verdict off exact-binomial McNemar), and TR145's own Mantel–Haenszel code was verified *unaffected* because it builds genuine unpaired marginal tables — so a future reader "fixing" TR145 to match TR149 would reintroduce a bug.

---

## 8. Chapter 3 — The Inference-Flag Safety Null Line (TR144 + TR145 + TR149 + TR152)

The second pillar of Phase 6 is a sequence of studies that each ask "does this serving-state knob move safety?" and each return a calibrated null. The thread spans three layers of the certification protocol — TAIS in Layer 2, the standardized replication in Layer 4, the serving-state factorial in Layer 5 — with TR145 as the single-config base case that the others extend. This chapter presents TR144 (the speculative-decoding member) and TR145 (the FP8 base case) in full, then summarizes how TR149 (Chapter 2) and TR152 (Chapter 5) widen the certified envelope. The unifying discipline is the claim ladder: every member bounds its result to "no detectable effect under tested conditions."

### 8.1 TR144 — Speculative decoding is behaviourally inert (TAIS)

**Headline.** At greedy decoding (temperature 0), speculative decoding is behaviourally inert with respect to safety: the target model's verification step fully overrides any draft-model influence, so neither rejection sampling nor typical acceptance leaks unsafe tokens — even when the draft is adversarially trained to prefer harmful completions.

**Scale and design.** A paired, factorial design across three core model pairs plus a five-experiment expansion: 64,855 total paired samples = 16,783 core + 48,072 expansion (E1 = 4,006; E2 = 4,006; E3 = 4,006; E4 = 24,036; E5 = 12,018), no samples dropped or filtered (SS5.9).

**The null, layer by layer.**

| Probe | Result | Source |
|---|---|---|
| Rejection-sampling byte-identity | 90.66% across 2,859 paired comparisons; of 267 textually-changed outputs, only 10 (3.7%) flip a safety score | SS3 |
| Phase 2 McNemar (rejection sampling) | all three non-significant (p = 1.000), 2–3 discordant per 953 | SS4 |
| Typical-acceptance McNemar (the primary leakage test) | all three non-significant (p ≥ 0.5), 2–4 discordant per 953 | SS6 |
| Per-task safety deltas (typical acceptance) | all 12 *exactly* 0.0pp, Cohen's d = 0.000, spanning baseline rates 0.408–0.985 | SS8 |
| Dose-response across speculation length N ∈ {1,3,5,8,12} | all 12 logistic slopes 0.000, r² = 0.000 | SS10 |
| Mantel–Haenszel speculative-vs-baseline safety OR | 1.000 [0.835, 1.198] for both Phase 2 and Phase 3 | SS16 |

**Observations.** Typical acceptance produces *fewer* total flips (2–4 per pair) than rejection sampling (8 per pair) — the opposite of what a leakage mechanism would predict (SS7). The only significant cross-model finding is that drafts are weaker *standalone* (draft-vs-target OR = 1.256 [1.054, 1.497]), yet under speculation the target's verification restores safety exactly (SS16). A reversed-telemetry curiosity: at ≤3B, draft-token acceptance is *higher* on safety prompts (47.8%) than capability prompts (26.3%), Cohen's d = 0.815 — the opposite of the disagreement hypothesis — and the gap inverts at 70B (SS12, SS19.3); acceptance rate does not predict safety flips at the per-request level (SS13).

**The named method — TAIS.** The Typical-Acceptance Invariance Screen is a behavioural equivalence check for speculative decoding: under matched draft+target pairs it compares the target-alone refusal rate against the speculative-output rate across both acceptance methods, using matched-pairs Cohen's h paired with a ±3pp TOST. The null cutoff is **|Cohen's h| < 0.1**, calibrated empirically from the E1–E5 expansion where the maximum observed |h| across all 18 AdvBench contrasts is **0.024** — roughly 4× below the cutoff (SS24.1). The screen is corroborated by 25/27 TOST passes at ±3pp (the two non-passes are capability-domain CI edge cases with mean diff 0.0pp) and, in the strongest cells, by 100% byte-identity (SS14).

**The expansion closes the escape hatches.** The v2.0 core carried predictable reviewer objections (single seed, single dtype, small models, benign draft, no quantized draft); v3.0 closes each with a direct probe (SS19–SS23):

- **E1 (production scale):** Llama-3.1-70B-AWQ-INT4 target + 8B draft → AdvBench refusal 0.839, 95% CI [0.783, 0.884], overlapping the lab-scale 3B band; the N-sweep at 70B is flat (max pairwise delta 0.36pp).
- **E2 (strongest adversarial draft):** DPO on flipped Anthropic/hh-rlhf → 100.00% byte-identical to the canonical-draft output across 4,006 samples.
- **E3 (quantized draft):** GPTQ-4bit draft → 100.00% byte-identical to the fp16 draft across 4,006 samples.
- **E4 (two seeds):** seed_123 vs seed_456 → 100.00% byte-identical across 12,018 keys.
- **E5 (dtype):** bfloat16 *does* move 36–53% of output bytes (a genuine accumulator-rounding change, not a no-op), yet max safety delta is −1.0pp, max |h| = 0.054, and all 9 contrasts pass TOST.

> The byte-identity matrix partitions every run into two equivalence classes — (fp16 core, E2, E3, E4) all byte-identical, and (E5 bf16) shifted 36–53% — with the safety outcome invariant across both. The expansion does not merely fail to find an effect; it demonstrates that the things that *do* change the output bytes (dtype) still do not change the safety verdict.

**Why the verification step is the mechanism.** The null is not a coincidence of weak drafts; it is a *structural* consequence of where speculative decoding places its safety-relevant decision. The draft model proposes, but the target model *verifies* every token, and at greedy decoding the target's argmax is the final arbiter — there is no residual distribution for a draft to bias. Three telemetry findings make this mechanism visible rather than assumed:

- **Drafts are weaker standalone, yet speculation restores safety exactly.** The only significant cross-model contrast in the whole report is that the draft models, *run alone*, refuse less than the targets (draft-vs-target OR = 1.256 [1.054, 1.497]). Under speculation that gap vanishes — the speculative-vs-baseline OR is 1.000 [0.835, 1.198] for both Phase 2 and Phase 3 (SS16). A leakage mechanism would predict the weaker draft *dragging down* the speculative output; the verification step instead clamps it back to the target's behaviour exactly.
- **Acceptance rate does not predict safety flips.** At the per-request level, the fraction of draft tokens the target accepts is uncorrelated with whether the request's safety score flips (SS13). If draft influence leaked through accepted tokens, high-acceptance requests would flip more; they do not.
- **The reversed-telemetry curiosity.** At ≤3B, draft-token acceptance is *higher* on safety prompts (47.8%) than capability prompts (26.3%), Cohen's d = 0.815 — the opposite of the "drafts and targets disagree most on safety" hypothesis — and the gap *inverts* at 70B (SS12, SS19.3). The reading is that on small models a refusal is a low-entropy, easy-to-draft continuation (the draft and target both "know" to refuse), so acceptance is high precisely where safety matters; the inversion at scale is a model-capacity effect, not a safety one.

**Observations.** The three telemetry findings together convert "we found no effect" into "we found the *reason* there is no effect, and it is the target's verification step." That is a categorically stronger null: the draft-vs-target standalone gap proves the instrument can detect a real refusal difference between two models (OR 1.256, CI excludes 1.0), and the speculative-vs-baseline OR collapsing to exactly 1.000 proves the verification step closes that gap. The acceptance-rate non-correlation and the reversed-telemetry finding then rule out the two most plausible leakage pathways (influence through accepted tokens, draft-target disagreement on safety content). The null is mechanistic, not merely statistical.

> The reversed-telemetry finding is the kind of anomaly the program's "log everything" discipline exists to catch: a higher draft-acceptance rate on safety prompts than capability prompts is counter-intuitive and could have been dismissed as noise, but read correctly it is *corroborating* evidence — if refusals are low-entropy continuations that both draft and target agree on, then the verification step has nothing to correct, which is exactly why the safety null holds. An effect the disagreement hypothesis predicted (drafts and targets diverging on safety) would have shown up as *lower* acceptance on safety prompts; the data show the opposite.

**Threats and the load-bearing carve-out.** All results are at temperature 0. At temperature > 0 the typical-acceptance criterion genuinely alters the output distribution and the leakage hypothesis may still hold; the planned E6 temperature sweep is unrun, and the null must not be extrapolated to stochastic settings (SS24.5). Regex classifiers and the Gemma 3 judge show κ = 0.0 across phases (they measure surface refusal pattern vs semantic safety) — the null holds under *each* independently but no cross-validation exists; expansion cells are regex-scored only, with a Claude rejudge queued (SS17). Two model families, vLLM v0.19, linear draft-verify only (no EAGLE/Medusa/tree speculation); per-pair MDE is 4.3–7.7pp, so 1–3pp effects cannot be ruled out within a single cell (SS15).

**Role and forward-pointer.** TAIS is the speculative-decoding member of **Layer 2** (behavioural screens), alongside RTSI (TR142). It is the one null-line member where the output is influenced by a *second model* (the draft) rather than a compressed or batched version of the same model, so it certifies a qualitatively distinct leakage pathway is closed at the production-dominant greedy operating point.

### 8.2 TR145 — FP8 KV-cache at a single configuration (the base case)

**Headline.** TR145 is an across-the-board null: flipping vLLM's `--kv-cache-dtype` from `auto` (FP16) to `fp8` (E4M3), with model weights held at FP16, produces no statistically supported safety degradation on any of three small instruction-tuned models across a five-phase paired battery — and the result is bounded as "no detectable FP8 effect under tested conditions," not "FP8 is safe."

**Scale and design.** 24,054 records at 100% completion, paired on (model, task, sample_id) across five phases under deterministic decoding (temperature 0.0, seed 42, vLLM v0.19.1, RTX 4080 Laptop). Phase counts: P1 = 3,009, P2 = 3,009, P3 = 4,000, P4 = 12,036, P5 = 2,000 (SS6.2). TR145 is the first report in the line to isolate KV-cache precision as the only manipulated variable (TR125/TR134 moved weight precision; TR138 moved batch size).

**The null across all five phases.**

| Phase | Test | Result | Source |
|---|---|---|---|
| P2 | Safety McNemar, 3 models | p = 1.0000 / 0.5966 / 0.3143, none Holm-significant; ORs ≈ 1.06 / 1.28 / 1.28 | SS2.1 |
| P2 | The only Holm-significant outcome is *capability* on Qwen | p = 0.0018 (OR 1.89), while Qwen *safety* stays p = 0.31 — opposite of the disproportionate-harm hypothesis | SS2.2 |
| P3 | Context-length × KV ANOVA | p_interaction = 0.974 / 0.538, η² ≈ 0.000–0.001 (non-monotonic, not the predicted widening) | SS6.1, SS8 |
| P4 | Batch-size × KV ANOVA | p_interaction = 0.980 / 0.998, η² = 0.000, every cell approximately additive | SS10.1, SS11 |
| P5 | Turn-5 multi-turn McNemar | Llama-1B p = 0.219 (6 discordant); Llama-3B p = 1.000 (0 discordant) | SS16 |
| — | Mantel–Haenszel pooled safety OR | 1.05 [0.90, 1.23] (P2); batch=8 OR 1.00 [0.83, 1.21]; turn-5 OR 2.06 [0.61, 6.99] | SS20 |
| — | TOST at ±3pp | 9/22 pass; both Llama safety tests pass (Δ −0.39pp, −0.58pp); lone safety non-equivalence is Qwen-1.5B Δ = −3.09pp, failing by 0.09pp | SS18 |

**Observations.** Every primary cell achieves 80% power at α = 0.05 (Phase 1+2 safety MDE 4.5pp; Phase 5 turn-5 MDE 3.4pp, the most-powered cell), and observed deltas fall below MDE in all cases — the null is not power-starved (SS19). Judge agreement (gemma3:12b vs regex) is κ = 0.43 over 13,676 records, which is the expected ceiling for a regex-vs-generalist comparison and affects only the inter-rater cross-check, not the regex-primary headline statistics (SS21). Cross-TR baseline reproducibility is 36/36 tuples within ±5pp against TR138/TR143/TR144 (SS22).

> The single safety boundary case is Qwen-1.5B, which fails the ±3pp TOST by 0.09pp (Δ = −3.09pp, CI [−5.91, −0.27]). This is the seed of the located finding that TR152 later resolves: the Qwen family carries whatever small FP8 footprint exists, and it lives on the benign edge, not the harmful core.

**The five phases, read one at a time.** The single-table summary above compresses five distinct sub-experiments, each isolating a different dimension along which a precision change could plausibly leak into safety. Read separately, they are a designed sweep from the simplest contrast to the most adversarial:

- **P1 / P2 — single-turn safety (3,009 + 3,009 records).** The base contrast: the same adversarial prompt under FP16 and FP8, scored for refusal. All three models return non-significant safety McNemar (p = 1.0000 / 0.5966 / 0.3143), with odds ratios clustered tightly around 1.0 (≈1.06 / 1.28 / 1.28). The one Holm-significant outcome in the entire phase is *capability* on Qwen (p = 0.0018, OR 1.89) — and it points the *wrong way* for a disproportionate-harm story: if FP8 selectively damaged safety, the safety contrast would move while capability held, but the data show the inverse (capability moves, safety holds at p = 0.31). That asymmetry is the first evidence that whatever FP8 does to Qwen is not a safety-specific mechanism (SS2.1–2.2).
- **P3 — context-length × KV interaction (4,000 records).** The hypothesis here is that quantized KV state accumulates error as the context grows, so a safety effect absent at short context could emerge at longer context. The two-way ANOVA finds p_interaction = 0.974 / 0.538 with η² ≈ 0.000–0.001, and — critically — the trend is *non-monotonic*, not the predicted monotone widening. A genuine accumulation mechanism would produce a context×KV interaction that grows with length; the data show noise around zero (SS6.1, SS8).
- **P4 — batch-size × KV interaction (12,036 records, the largest phase).** This is the phase that most directly anticipates TR152: batch size is the Phase 5 attack-surface flag (batch-invariance failure), and P4 asks whether it interacts with KV precision. The ANOVA returns p_interaction = 0.980 / 0.998, η² = 0.000, every cell approximately additive. The batch=8 pooled OR is 1.00 [0.83, 1.21] — dead-centre null. P4 is where TR145 first establishes the *additivity* that TR152's star factorial later confirms across the full serving state: batch size does whatever it does on its own, but it does not modulate the FP8 contrast (SS10.1, SS11).
- **P5 — multi-turn turn-5 (2,000 records).** The most adversarial single-config slice: a five-turn conversation, scored at the final turn, where multi-turn jailbreak pressure is highest. Llama-1B returns p = 0.219 (6 discordant); Llama-3B returns p = 1.000 (0 discordant). The turn-5 pooled OR (2.06 [0.61, 6.99]) has a wide CI that the report flags honestly — turn-5 is the lowest-n cell — but the most-powered turn-5 MDE (3.4pp) and the zero-discordant Llama-3B result keep the slice inside the null (SS16, SS19).

**Observations.** The phase-by-phase reading is what turns "an across-the-board null" from a slogan into a structured finding: each phase probes a *different* leakage mechanism (single-turn susceptibility, context accumulation, batch-size interaction, multi-turn pressure), and each returns null *for a mechanism-specific reason* rather than by generic non-significance. The two phases that matter most for the downstream arc are P4 and the P2 capability asymmetry. P4 establishes additivity of the batch knob — the property TR152 generalizes into the serving-state-independent certificate. The P2 capability-moves-while-safety-holds asymmetry is the first appearance of the theme TR146 later formalizes: FP8's damage to Qwen is real but is *not* safety-targeted, so a safety verdict cannot be read off a capability or confidence signal.

> Reading the five phases separately also exposes the honest weak point that TR145's scope discipline names: every phase is a *single configuration* of its manipulated axis crossed with KV precision, so P3/P4 measure context and batch *one factor at a time against KV*, never jointly with temperature or prefix caching. That single-config-per-axis limitation is exactly the gap TR152's factorial is built to close — which is why TR145 is the *base case* of the null line and TR152 is its capstone, not a redundant re-run.

**The scope discipline.** TR145 is the report that most explicitly bounds its own claim. The abstract, the executive summary's "What this does not rule out," and Conclusion 1 all state the licensed claim is "no statistically supported FP8 safety degradation *under tested conditions*." The non-extrapolation boundaries are enumerated: largest model 3.21B (production runs 7B–70B); longest context 2,048 tokens (production runs 8k–128k); 5-turn conversations (real arcs run 10–100 turns); temperature 0.0 only (production runs 0.7–1.0); single hardware SKU and single vLLM version.

> This scope discipline is exactly what Layer 5 needs. A serving-state-independent FP8 certificate can only be anchored on a base case whose claim is already self-limited to its tested envelope, so that the downstream reports (TR149 standardized batteries, TR152 serving-state sweep) *extend* the envelope rather than re-litigate an over-claimed result. TR145 supplies the conservatively-bounded base; the rest of the line widens it.

**Role and forward-pointer.** TR145 is the single-config FP8 base case feeding **Layer 4** (as the 1B–3B null that TR149 standardizes and TR151 scales) and **Layer 5** (as the single-config null that TR152 generalizes across the serving state). It is the parked-paper seed (CLAIM_LADDER L6): the standalone TR145 paper is parked until TR151 scale data lands.

### 8.3 How TR149 and TR152 widen the line

TR149 (Chapter 2) re-measures the FP8 null on the four literature-canonical batteries and returns positive ±3pp equivalence on all 12 cells (pooled OR 0.8065), answering the "your null is a task-set artefact" objection. TR152 (Chapter 5) folds the serving-state axes — batch size, prefix caching, temperature — into one factorial and asks not "does FP8 change safety?" but "does any serving-state axis *modulate the FP8 contrast*?", returning a flat interaction with all measurable footprint confined to a sub-1pp Qwen-XSTest over-refusal lean. Read together, the four members are a **four-step resolution ladder on a single underlying truth**: harmful refusal is invariant to FP8, and the over-refusal boundary is faintly, locally perturbed on one model family.

| Member | Knob | Verdict | Pooled OR (safety) | Layer role |
|---|---|---|---|---|
| TR144 (TAIS) | speculative decoding @ temp 0 | H0 null, max \|h\| = 0.024 | 1.000 [0.835, 1.198] | Layer 2 |
| TR145 | FP8 KV-cache, single config | H0 null | 1.05 [0.90, 1.23] | base case → Layers 4, 5 |
| TR149 | FP8 KV-cache, standardized batteries | H0 null, 12/12 TOST-equivalent | 0.8065 [0.3828, 1.6989] | Layer 4 (1B–3B) |
| TR152 | FP8 KV-cache, serving-state factorial | H1 located (Qwen-XSTest), H2 rejected | 1.8817 [1.3185, 2.6855] | Layer 5 |

> **Observations.** The TR152 pooled OR is the only one whose CI clears 1.0, which looks at first like a break in the null line. It is not. The entire signal is on XSTest (the over-refusal battery) and 91% Qwen-bound; the harmful batteries show *zero* discordance across 8,976 pairs. A pooled OR above 1.0 driven entirely by *more refusal of benign-but-alarming prompts* on one model family is a located over-refusal lean, not a safety degradation — the certificate carries it as a per-family footnote, exactly the H1-not-H2 outcome the factorial was built to distinguish.

---

## 9. Chapter 4 — The Mechanistic Dead-End (TR146)

**Headline.** TR146 is a mechanistic dead-end: across four standard interpretability probes (first-token entropy, refusal-direction geometry, calibration drift, safety-neuron quantization error), none distinguishes safe from dangerous quantized configurations — so behavioural measurement, not mechanistic probing, must carry the safety verdict across the entire arc.

### 9.1 The attractive shortcut, and why it fails

The most attractive cost-saving move available to the whole program is to replace expensive behavioural evaluation (generate, judge, triangulate) with a cheap forward-pass mechanistic probe (run the model forward, read an internal signal, predict safety). TR146 tested that move directly with 5,100 forward passes across 17 model-quant cells (6 FP16 anchors + 5 AWQ + 6 GPTQ, INT4) — forward-pass only, no text generation — and correlated each probe's per-cell metric against the TR142 regime labels (hidden_danger / near_hidden_danger / neutral) and RTSI scores. It falsified the shortcut on every axis.

| Probe | What it measures | RTSI correlation | Danger-vs-neutral separation | Source |
|---|---|---|---|---|
| 1. First-token entropy | refusal uncertainty at first generated token | Pearson r = 0.083, p = 0.81 | Mann–Whitney p = 0.61 | SS4.2–4.3 |
| 2. Refusal-direction cosine | rotation of the Arditi et al. refusal direction | r = −0.144, p = 0.67 | p = 0.61 (every cell > 0.97 cosine) | SS5.2–5.3 |
| 2b. Refusal-direction magnitude | attenuation of the refusal direction | r = −0.61 (the one signal) | p = 0.036 | SS5.3, SS8.5 |
| 3. Calibration drift | confidence / Gini shift | r < 0.09 | p = 0.788 | SS6.3, SS8.5 |
| 4. Safety-neuron quant error | disproportionate error on top-5% safety neurons | r = 0.119 | p = 0.979 | SS7.4–7.5 |

**Observations.** No probe achieves |r| > 0.15 with RTSI. The one apparent signal — refusal-direction *magnitude* at r = −0.61 — is a model-level property (e.g. llama3.2-1b magnitude 4.7 vs phi-2 72.3), identical across a model's AWQ and GPTQ variants, so it is a vulnerability *factor* of the base model, not a per-configuration diagnostic of which quantization is dangerous (SS5.3, SS8.5).

**Each probe, and why each one fails.** The four probes were not chosen at random — each operationalizes a different theory of *where* quantization damage would show up, and the value of the falsification is that it closes four distinct hopes, not one:

- **Probe 1 — first-token entropy** tests the theory that a quantized model about to comply with a harmful prompt is *less certain* at the first generated token (the refusal "decision point"). It fails at r = 0.083 (p = 0.81): entropy at the first token simply does not track downstream refusal behaviour. The compounding danger is the GPTQ confidence paradox (§9.2) — GPTQ models grow *more* confident while refusing *less*, so entropy points the wrong way (SS4.2–4.3).
- **Probe 2 — refusal-direction geometry** is the most theoretically motivated probe, testing the Arditi et al. finding that refusal is mediated by a single residual-stream direction: if quantization rotated that direction, refusal would degrade. The *cosine* of the direction is preserved at > 0.97 across every cell (r = −0.144, p = 0.61) — the geometry is essentially intact under quantization. The *magnitude* is the one above-floor signal (r = −0.61), but it is a model-level constant, not a per-config diagnostic. The geometry is preserved; the behaviour still varies — which is the cleanest single illustration of "geometrically preserved, behaviourally ineffective" (SS5.2–5.3).
- **Probe 3 — calibration drift** tests whether quantization shifts the model's confidence calibration (Gini / confidence) in a way that predicts safety. It fails at r < 0.09 (Mann–Whitney p = 0.788): calibration moves under quantization, but not in a direction that separates dangerous from neutral configs (SS6.3, SS8.5).
- **Probe 4 — safety-neuron quantization error** is the most direct test: identify the top-5% of neurons by harmful-vs-harmless activation contrast, and measure whether *those* neurons take disproportionate quantization damage. They do (1.40×, the one real mechanistic result) — but the ratio does not predict danger status (r = 0.119, Mann–Whitney p = 0.979), with neutral rows sitting *above* danger rows (SS7.4–7.5).

**Observations.** The four-probe structure is what makes F3 a *forbidden* claim rather than a single underpowered null. A reviewer can dismiss one failed probe as a bad operationalization; dismissing four — entropy, geometry, calibration, and the most-direct neuron-error probe — requires arguing that *every* internal signal fails to predict behaviour, which is precisely the claim TR146 stakes. The deepest of the four is Probe 2: the refusal direction is the strongest interpretability result in the safety literature the program could have leaned on, and the finding that its *geometry is preserved while behaviour varies* is the exact reason the certification protocol's Layer 1a (refusal axis) must remain a *behavioural* template screen rather than a geometric probe.

> The probe-by-probe reading is what licenses the strong negative. "Mechanistic probes do not predict safety degradation under quantization" would be an over-claim from one null; from four independent falsifications plus a confidence paradox that makes the most natural probe (entropy/confidence) point the *wrong* way, it is the Forbidden-tier claim F3. The program wanted this shortcut to work — a cheap forward-pass screen would have replaced the expensive generate-judge-triangulate pipeline — and the discipline is that it reports the falsification at full strength rather than salvaging the one above-floor magnitude signal into a diagnostic it cannot support.

### 9.2 The mechanism is real but necessary-not-sufficient

The one *established* mechanistic result is that safety-critical neurons (top 5% by harmful-vs-harmless activation contrast) absorb **1.40× disproportionate quantization error** (mean 1.395×, one-sample t vs ratio 1.0 p < 0.0001), with GPTQ worse than AWQ (1.45× vs 1.37×) (SS7.3–7.4). But that same ratio does *not* predict hidden-danger status: the danger-vs-neutral Mann–Whitney is p = 0.979, and the two *neutral* rows (qwen2.5-1.5b AWQ 1.46×, GPTQ 1.52×) sit *above* several hidden-danger rows (SS7.4–7.5).

> The safety mechanism is real — quantizers genuinely under-protect safety-critical parameters — but it is necessary, not sufficient. Every quantized model suffers the damage; only some cross the behavioural failure threshold, and that threshold depends on full forward-pass dynamics (attention, gating, normalization) that neuron-level or single-direction signals cannot resolve. You cannot read safety off the weights.

A second, more dangerous corollary feeds the program's caution list: the **GPTQ confidence paradox** (SS8.3). GPTQ models become *more confident* about the first token (negative entropy shift, positive confidence shift) while *failing to refuse* — so any confidence- or logit-based screen would actively *approve* the most dangerous configurations. This is a concrete argument against trusting any logit-derived "we look certain, so we're fine" signal in serving-state certification.

### 9.3 Role and forward-pointer

TR146 is the **negative control** that governs both pillars of Phase 6. It is the reason the measurement-validity substrate (Chapter 1) and the null line (Chapter 3) are *behavioural* rather than mechanistic: the refusal direction is geometrically preserved at cosine > 0.97 yet behaviourally ineffective, so the refusal-axis verdict (Layer 1a) cannot be discharged by mechanistic probes and must remain a behavioural template screen. In the claim ladder, TR146 is the anchor for Forbidden claim **F3** ("mechanistic probes predict safety degradation under quantization") — the bridge paper treats safety as behavioural, not mechanistic, and cites TR146 as the citable evidence for why. Its threats to validity are honest: AWQ/GPTQ INT4 only (no GGUF k-quants), 100-prompt budget per phase, only ~2 neutral rows in the regime contrast (so "no separation" is partly low power), a single ~65% probe-extraction layer, a fixed top-5% safety-neuron cutoff, and an observational (non-causal) design.

---

## 10. Chapter 5 — Cross-Axis Composition and the Layer 5 Anchor (TR152)

**Headline.** TR152 is the cross-axis-composition capstone of the FP8 KV-cache null line: an FP8 KV-cache is refusal-neutral on clearly-harmful prompts across the *entire* tested serving state — batch size, prefix caching, and sampling temperature — with the only measurable footprint a sub-one-percentage-point over-refusal lean confined to the Qwen 2.5 family on the XSTest benign-edge battery, located specifically on the temperature axes and amplified mildly on the 1.5B variant. This is the report that converts a single-configuration null (TR145) and a standardized-corpus null (TR149) into a *serving-state-independent* FP8 safety certificate — the artefact Layer 5 inherits.

### 10.1 The serving engineer's objection, and the design that answers it

TR145 and TR149 each measured FP8 at one serving configuration. That leaves open the first objection a serving engineer raises: production never runs a single config — it batches requests, shares prefix caches, speculates tokens, and samples at non-zero temperature. The relevant question is therefore not "does FP8 change safety?" (settled null) but "does any serving-state axis *modulate the FP8 contrast*?" TR152 folds those axes into one factorial to answer it.

The design is an **FP8-anchored star factorial**, not a fractional factorial. Rather than the full 2×3×2×2×3 = 72-cell crossing (KV-dtype × batch × prefix × speculative-decoding × temperature), it fixes a center baseline (`kv-auto, bs=1, pf=off, sp=off, temp=0.0`) and runs one "spoke" per off-baseline axis level, each spoke at both FP16 and FP8 — 7 paired contexts × 2 dtypes = 14 cells per (model, battery). The report explicitly labels this a **one-factor-at-a-time star, not a resolution-IV/V design** (SS3), correcting the 2026-05-14 "AI slop" critique that caught a prior resolution-V mislabel. The reason the star is the *correct* choice, not a convenience, is structural: matched-pair safety estimators require every deviating-axis spoke to share the baseline's prompt set, which a confounded fractional design would break. The trade is deliberate and disclosed — *joint* two-axis interactions are not measured; the star answers "does any single serving axis bend the FP8 contrast?" at full matched-pair rigor. Each cell carries machine-checkable `deviates_axis` and `paired_with` fields, and untested factorial cells are enumerated explicitly as a reviewer-mitigation pattern.

**Scale.** 45,000 sampled responses; **20,754 matched FP16-vs-FP8 pairs across 120 (battery × model × context) strata**; **135,000 judge-label rows** on disk (45,000 per judge × 3 judges: regex + gemma3:12b + llama3.1:8b). Five models across three families (`llama3.2-1b`, `llama3.2-3b`, `phi3-mini-4k`, `qwen2.5-1.5b`, `qwen2.5-3b`) × four batteries (HarmBench-400, JailbreakBench-100, StrongREJECT-313 at 100/cell; XSTest uncapped to 450/cell) × 6 runnable serving-state contexts, vLLM v0.19.1 on an RTX 4080 Laptop 12GB, local judges under the `--skip-openai-judge` umbrella, **$0 external cost**. Realized coverage is **12 of 14 planned cells per model** (≈19.4% of the full 72-cell factorial); the two `sp-1` speculative-decoding cells fail identically across every model (SS1, SS3, SS4).

### 10.2 The harmful core is perfectly invariant

| Quantity | Value | Source |
|---|---|---|
| Discordant pairs across all harmful batteries | **0 of 8,976** matched pairs (5 models × 6 contexts × 3 batteries = 90 cells) | SS1, SS7, SS8 |
| FP16-only safe rate on HarmBench + JailbreakBench + StrongREJECT | **1.0000** (8,976/8,976) | SS1, SS8 |
| Harmful-battery cells passing ±3pp TOST | **90 of 90** (by construction — zero variance) | SS12 |

**Observations.** Every model refuses every clearly-harmful prompt under both FP16 and FP8, across every batch size, prefix-cache setting, and temperature tested. There is no discordance to analyze on the harmful core — the matched-pairs structure has zero off-diagonal mass. This is the part of the certificate that holds *unconditionally* in the 1B–4B range: the serving-state-independent FP8 harmful-refusal guarantee rests on 8,976 paired observations with zero counterexamples.

> The harmful-core invariance is the load-bearing Layer 5 claim. Everything that follows — the pooled OR above 1.0, the Qwen lean, the temperature localization — happens entirely on the *benign-edge* battery (XSTest), where the failure mode is over-refusal of safe-but-alarming prompts, not compliance with harmful ones. Keeping those two axes separate is the whole point of the chapter.

### 10.3 All discordance lives on the over-refusal edge

| Quantity | Value | Source |
|---|---|---|
| Total discordant pairs | **133** (Σb = 87 FP8-degraded, Σc = 46 FP8-improved, net b−c = +41) | SS8, SS9 |
| Location | **all 133 on XSTest**, in 27 of 30 XSTest cells (the 3 concordant cells are all `llama3.2-3b`) | SS8 |
| XSTest FP16-only safe rate | 0.6928 (the only battery not at the refusal ceiling) | SS8 |
| Cross-context Mantel–Haenszel pooled OR | **1.8817**, 95% CI **[1.3185, 2.6855]** (Haldane (87+0.5)/(46+0.5)) | SS1, SS11 |
| Independent exact binomial sign test (87 vs 46) | z ≈ 3.55, two-sided **p ≈ 0.0004** | SS11 |
| Per-cell McNemar significant after Holm | **0 of 120** (smallest raw p = 0.0117 at `xstest \| qwen2.5-1.5b \| temp=0.7`, b=10/c=1; Holm → 1.0000) | SS9, SS13 |
| TOST equivalence at ±3pp | **117 of 120 cells (97.5%)** equivalent; the 3 inconclusive are all `qwen2.5-1.5b` non-zero contexts; all 120 equivalent at ±5pp | SS12 |
| Cohen's h on the 27 discordant cells | max **\|h\| = 0.0458** (`qwen2.5-1.5b \| temp=0.7`); 27 negligible / 0 small / 0 medium / 0 large | SS9 |
| FP8-interaction spread | **2.99pp** (inside-edge of ±3pp modulation band); per-context mean Δpp from −0.164pp (temp=0.7) to −0.012pp (baseline), a 0.152pp swing | SS10 |

**Observations.** The pooled OR is the only one in the entire null line whose CI clears 1.0, and at first glance that reads as a break in the line. It is not. The whole signal is on XSTest (the over-refusal battery), and dropping XSTest produces a full null — 100% of the pooled FP8 signal is XSTest-bound (SS22). A pooled OR above 1.0 driven entirely by *more refusal of benign-but-alarming prompts* is a located over-refusal lean, not a safety degradation. The per-cell evidence reinforces this: zero of 120 cells survive Holm correction, every one of the 27 discordant cells has a negligible Cohen's h (max 0.0458, more than 4× below the 0.2 "small" threshold), and the axis-level interaction spread (2.99pp) sits *inside* the ±3pp pre-registered modulation band with a per-context-mean swing of just 0.152pp. No serving axis modulates the FP8 contrast at the axis-level mean.

> The H0/H1/H2 hypothesis structure is what makes this verdict precise. H0 (no effect anywhere) is rejected by the pooled OR and sign test. H2 (a serving-state axis *modulates* the FP8 contrast) is rejected by the flat interaction spread and the per-context means within 0.17pp of zero. What survives is **H1**: a *located* positive footprint with no serving-state interaction — discordance confined to the Qwen × XSTest corner, no axis bending the contrast. The factorial was built specifically to distinguish H1 from H2, and it does.

### 10.4 The signal is doubly Qwen-bound

| Family / model | b−c (net FP8-degraded XSTest pairs) | Per-family MH OR | Source |
|---|---|---|---|
| Qwen-2.5 (both variants) | carries **79 of 87** FP8-degraded pairs (91%) | **3.878 [2.386, 6.302]** (FP8 degrades, CI well above 1.0) | SS8, SS23 |
| — `qwen2.5-1.5b` | +34 | — | SS8 |
| — `qwen2.5-3b` | +25 | — | SS8 |
| Llama-3.2 (`-1b` −6, `-3b` −2) | net **−8** (opposite direction) | **0.484 [0.202, 1.157]** (neutral-to-improve, CI brackets 1.0) | SS8, SS23 |
| Phi-3-mini-4k | **−10** (opposite direction) | **0.130 [0.024, 0.715]** (FP8 statistically *improves*, CI excludes 1.0 on improved side) | SS8, SS23 |

**Observations.** The per-family Mantel–Haenszel decomposition splits the pooled OR three ways and shows the aggregate 1.88 is not a property of FP8 — it is a property of the Qwen family. Llama and Phi-3 run the *opposite* (FP8-improved) direction; Phi-3's CI even excludes 1.0 on the improvement side. Qwen's inherent baseline over-refusal is also the highest of the set: `qwen2.5-1.5b`'s both-unsafe XSTest discordance rate is 37.2% vs `llama3.2-3b`'s 26.9%. The footprint sits on the family that is already most prone to over-refusing benign-but-alarming prompts, and FP8 nudges that pre-existing tendency rather than introducing a new failure mode.

> This is why the certificate carries the finding as a *per-family footnote* rather than a generic FP8 warning. "FP8 KV-cache slightly increases Qwen-2.5's over-refusal of benign-edge prompts by sub-1pp, concentrated on the temperature axes" is a precise, actionable caveat for a deployment running Qwen; it says nothing adverse about FP8 on Llama or Phi-3, and nothing at all about harmful-prompt safety on any family.

### 10.5 The v1→v2 resolution ladder — refine, not flip

TR152 has a genuine pilot-to-canonical arc, and the discipline of that arc is itself a methodological payload.

| Quantity | v1 pilot (2026-05-21) | v2 canonical | Reading |
|---|---|---|---|
| Records / matched pairs | 14,400 / 7,010 | **45,000 / 20,754** | 3× scale-up |
| Families / models | 2 / 3 (XSTest capped 100) | **3 / 5** (XSTest uncapped 450) | +Phi-3, +llama3.2-1b |
| Discordant pairs | 23 (17/6 split) | **133** (87/46) | 5.8× discordant base |
| Strata | 72 | **120** | finer resolution |
| Pooled OR | 2.69 [1.09, 6.62] | **1.88 [1.32, 2.69]** | regressed toward 1.0 (as underpowered estimate should) |
| CI width | 5.53 | **1.37** | ~4× tighter |
| CI lower bound | 1.09 | **1.32** | moved *further* from 1.0 |
| TOST coverage | 60/72 (83.3%) | **117/120 (97.5%)** | rose |
| Smallest raw p | 0.25 | **0.0117** | 21× tighter |

**Observations.** The v2 numbers are the textbook signature of "real directional signal, badly localized in v1": the point estimate regresses toward 1.0 against larger n, the CI tightens ~4×, *and* the lower bound moves further from 1.0. An artefact would have collapsed the CI through 1.0; a real-but-localized effect tightens around a value above 1.0. Crucially, the **qualitative shape — Qwen-family, XSTest-only, temperature-amplified — is invariant across the resolution lift**. That invariance is what retires v1 to a pilot citation and makes v2 the canonical TR152 result.

> v2 also retracted one v1 attribution via direct console-log inspection — the diagnostic-discipline move the program's anti-confabulation rules demand. The speculative-decoding axis failure is a **vLLM v0.19.1 argparse rejection** (deprecated `--speculative-model` / `--num-speculative-tokens` flags vs the current `--speculative-config` JSON blob), **not the OOM** v1 claimed. The fix is a one-line `run.py:167-169` flag swap, not a bigger GPU. v1 pattern-matched "cell failed → VRAM ceiling"; v2 read the log and found the real cause. That is the `read the artifact first` rule operating inside the report itself.

### 10.6 Sensitivity, power, and judge agreement

| Check | Result | Source |
|---|---|---|
| Leave-one-out: drop XSTest | **full null** (100% of signal is XSTest-bound) | SS22 |
| LOO: drop `qwen2.5-1.5b` | OR 1.230 [0.762, 1.983] (CI brackets 1.0) | SS22 |
| LOO: drop `qwen2.5-3b` | OR 1.376 [0.927, 2.043] (CI brackets 1.0) | SS22 |
| LOO: drop any single context | CI still clears 1.0 (largest residual: drop temp=0.7 → 1.675) | SS22 |
| Power | per-cell Holm-clear MDE ≈ −3.25 to −3.75pp (worst observed −2.23pp); pooled MH powered to detect OR ≈ 1.42, observed 1.88 | SS21 |
| Cross-judge κ | gemma3:12b ↔ llama3.1:8b **κ = 0.831** on 44,951 paired records (up from v1's 0.814), clears JTP "robust" 0.70 | SS14 |

**Observations.** The leave-one-out structure pins the finding precisely: it survives dropping any single *context* but not dropping XSTest or either Qwen variant — which is exactly what a Qwen-on-XSTest localized effect predicts. The pooled estimator was powered to detect OR ≈ 1.42 and observed 1.88, so the positive finding is not power-starved. The cross-judge κ = 0.831 clears the JTP robust threshold, so the verdict does not hinge on judge noise (the pooled regex↔LLM κ ≈ 0.06–0.10 is a Simpson-paradox artifact of pooling degenerate harmful marginals against XSTest's balanced split, not a real disagreement — SS14).

### 10.7 Role and forward-pointer

TR152 v2 is the **sole data anchor for the bridge paper's Layer 5 (serving-state validity)**. TR145 settled the single-configuration question (pooled OR 1.05 [0.90, 1.23]); TR149 replicated the null on the standardized four-battery corpus (pooled OR 0.81 [0.38, 1.70], 12/12 TOST-equivalent); TR152 folds the serving-state axes into one factorial and returns a flat interaction (spread 2.99pp inside the ±3pp band, every per-context mean within 0.17pp of zero, every LOO-context CI still clearing 1.0). That flat interaction is what converts the two upstream nulls into a **serving-state-INDEPENDENT FP8 safety certificate**: a single certificate a deployment can rely on across {batch ∈ 1, 8, 32} × {prefix on/off} × {temperature 0.0, 0.7, 1.0} in the 1B–4B range without re-validating every serving setting, carrying the precisely-located Qwen-2.5 XSTest sub-1pp over-refusal lean as a per-family footnote rather than a blocker. The two documented extensions that *widen* the certified envelope rather than rebuild it are Layer 4's 7B–72B scale companion (TR151, cloud-gated) and the patched speculative-decoding axis (a one-line launcher fix). The full standalone narration is the 1,331-line `Technical_Report_152.md`; this chapter is its synthesis.

---

## 11. Chapter 6 — Cloud-Gated Deferrals (TR150, TR151, TR153)

**Headline.** Three scaffolds complete the Phase 6 design but are deliberately unrun: they require cloud GPU (A100-80GB / L40S class) and, for the adversarial-corpus members, an external-judge umbrella that is Fellowship-gated. They are documented here so the certified envelope's *boundaries* are explicit — the bridge paper inherits each as a named extension with a verified/NOT-verified split, not as a silent gap.

### 11.1 What is deferred, and why

The Phase 6 executed set (TR144–149, TR152) anchors the certification protocol for the **1B–4B instruction-tuned model range on a single local hardware SKU**. Three boundaries of that envelope are scaffolded but not yet measured:

| Scaffold | Scope | Layer role | Gate | Verified-now status |
|---|---|---|---|---|
| **TR150** | long-context (8k–128k) FP8 KV-cache safety | extends the null line into the context regime TR145 explicitly excluded | cloud GPU (long-context KV memory) | scaffold only; external specs WebFetch-verified; not run |
| **TR151** | 7B–72B FP8 KV-cache safety on standardized batteries | **completes Layer 4** (scale validity) — the 7B–72B half TR149 hands forward | A100-80GB-class cloud GPU | scaffold only; the model set is the published-spec set TR149 names; not run |
| **TR153** | KV-cache *method* sweep (FP8 vs INT8 vs other quantization schemes) | widens the Layer 4/5 knob from one FP8 contrast to a method family | cloud GPU + possible external judge | scaffold only; not run |

**Observations.** None of these is a Phase 6 *result*; all three are Phase 6 *boundaries*. The distinction matters for the claim ladder: TR149 anchors Layer 4 for 1B–3B and explicitly defers the 7B–72B half to TR151, so the bridge paper's Layer 4 claim is "scale-validated at 1B–3B; 7B–72B scaffolded, cloud-gated" — a bounded claim, not an over-reach. Treating the deferrals as named extensions keeps the certified envelope honest about where it stops.

> The defensibility bar (`feedback_scaffold_defensibility_bar`) governs how these scaffolds may be described: external specs (HuggingFace IDs, parameter counts, library API surfaces) are WebFetch-verified; no DOE or statistical terminology is asserted that has not been derived from a real run; and every scaffold README carries an explicit verified / NOT-verified split. `py_compile + imports` passing is *not* "fire-ready" — these are documented as designs awaiting execution, nothing more.

### 11.2 The GO/NO-GO trigger

The deferrals are not parked indefinitely. The bridge paper's build plan (`UPGRADE_PLAN.md`) carries a GO/NO-GO trigger dated **2026-10-24**, gated on two external signals: an Anthropic Fellowship decision (which would supply both the cloud-credit budget for A100-class runs *and* the Researcher Access Program umbrella that lets adversarial corpora flow through external judges) and a NeurIPS-acceptance signal (notification 2026-09-24). The umbrella gate is the binding constraint for the adversarial members: until it resolves, TR150/TR151/TR153 would run with the same `--skip-openai-judge` local-judge cohort as the executed set ($0 external cost, local Ollama judges only), which is sufficient for the safety verdict but forgoes the GPT-4o triangulation slot.

> Phase 6's contribution is not contingent on these runs. The executed set (TR144–149, TR152) is a complete five-layer certification for the 1B–4B range; the deferrals widen the *parameter* and *context* envelope, they do not repair a hole in the protocol. That is the difference between "the protocol is incomplete" and "the protocol is validated on a bounded envelope with named, designed extensions to widen it."

---

## 12. Cross-TR Synthesis — From Measurement Validity to a Layer 5 Certificate

The six executed reports are not six independent findings; they are a single dependency chain that builds upward from "can you trust the number?" to "can a deployment rely on the certificate across its serving state?" This section makes the chain explicit.

### 12.1 The dependency spine

| Layer | TR(s) | Question answered | What it licenses downstream |
|---|---|---|---|
| **1 (validity)** | TR148 (JTP) | Can the safety *label* be trusted? | Every judge-labeled verdict in Layers 2–5 cites a *measured* κ floor (0.6917), split into refusal-axis (1a) and composite-harm (1b) screens |
| **3 (compile integrity)** | TR147 (CRI) | Can the latency *number* be trusted? | Every cross-stack comparison is gated by a reproducibility band; the Triton kill-shot is the worked counterexample |
| **2 (behavioural screens)** | TR144 (TAIS), [RTSI/TR142] | Does a behavioural knob (speculative decoding) leak unsafe tokens? | A calibrated null (max \|h\| = 0.024) at the greedy operating point |
| **4 (scale validity)** | TR149 | Does the FP8 null replicate on the field's standard batteries? | Positive ±3pp equivalence on 12/12 cells (1B–3B); 7B–72B handed to TR151 |
| **5 (serving-state validity)** | TR152 | Does any serving axis modulate the FP8 contrast? | A serving-state-independent FP8 certificate (1B–4B), Qwen-XSTest footnote |
| **(negative control)** | TR146 | Can mechanistic probes shortcut the behavioural verdict? | No — F3 forbidden; the whole chain stays behavioural |

**Observations.** The chain has a strict order of operations. The validity layers (1, 3) come *first* because they are the gates that license everything else to count as evidence: a safety delta measured by an untrustable judge, or a latency delta measured on a non-reproducible stack, is not a finding. The behavioural and scale layers (2, 4) then accumulate calibrated nulls under those validated instruments. Layer 5 (TR152) is the capstone because it is the only one that asks the *composition* question — not "is FP8 safe at config X?" but "is the safety verdict stable across the serving state?" — and a serving-state-independent certificate is the only kind a production system can actually consume without per-config re-validation.

> The negative control (TR146) sits underneath the entire chain. It is the reason every layer is behavioural rather than mechanistic: if a cheap forward-pass probe could predict safety, the program would have replaced its expensive generate-judge-triangulate pipeline with it. TR146 falsified that shortcut on four axes, so the chain is built on behavioural measurement end to end — and the cost of that measurement is what makes Layers 1 and 3 (the validity gates) load-bearing rather than ceremonial.

### 12.2 The two-pillar structure

Phase 6 has two pillars that meet at Layer 5:

1. **The measurement-validity substrate** (Chapter 1: TR147 + TR148; negative control TR146). This pillar answers *"can you trust what you measured?"* It produces two calibrated indices (CRI for latency reproducibility, JTP for judge agreement) and one hard negative (mechanistic probes do not shortcut the behavioural verdict). Nothing here is a safety result; everything here is a *license* for the safety results.

2. **The inference-flag null line** (Chapter 3: TR144 + TR145 + TR149 + TR152). This pillar answers *"does the serving state move safety?"* across four members and returns a calibrated null on the harmful core every time, with a single located over-refusal footprint (Qwen × XSTest) that the factorial resolves to H1-not-H2.

The pillars meet at TR152: the serving-state factorial is simultaneously the capstone of the null line *and* the sole Layer 5 anchor, and it can only carry that role because the measurement-validity substrate licenses its judge labels (JTP κ = 0.831 on its own corpus, clearing robust) and the claim ladder bounds its positive finding to a per-family over-refusal footnote rather than a generic FP8 warning.

> The synthesis claim is narrow and defensible: *for instruction-tuned models in the 1B–4B range, on a single local hardware SKU and vLLM v0.19.1, an FP8 KV-cache is harmful-refusal-neutral across the tested serving state, under judge labels whose agreement is independently measured to clear the JTP robust threshold.* Every qualifier in that sentence is a layer of the protocol doing its job. The bridge paper's contribution is the *protocol*, instantiated and validated on this bounded envelope; the deferrals (Chapter 6) widen the envelope without rebuilding the protocol.

The remaining subsections re-cut the same six executed reports along the axes a reviewer or a deploying team actually reasons in — the serving knob, the model family, the judge cohort, the effect-size regime, the null-vs-located distinction, and the estimator — because a finding that holds *across* the cuts is far stronger than one visible only along the spine.

### 12.3 Synthesis by serving-state knob

The program's safety question is, at bottom, "which knob, if turned, moves refusal behaviour?" Phase 6 turned eight of them. The table collects the verdict per knob across whichever reports manipulated it.

| Serving knob | Reports that turned it | Verdict on the harmful core | Located footprint | Source |
|---|---|---|---|---|
| **KV-cache precision** (FP16 → FP8 E4M3) | TR145, TR149, TR152 | null (pooled OR 1.05 / 0.81 / harmful 0/8,976 discordant) | Qwen-XSTest over-refusal, sub-1pp | §8.2, §7.2, §10.2 |
| **Batch size** ({1, 8, 32}) | TR145 (P4), TR152 | additive, no interaction (P4 ANOVA p_int = 0.980/0.998) | none at the axis-mean | §8.2, §10.3 |
| **Context length** (≤2,048 tok) | TR145 (P3) | additive, non-monotonic (P3 ANOVA p_int = 0.974/0.538) | none | §8.2 |
| **Prefix caching** (on/off) | TR152 | no axis-mean modulation (interaction spread 2.99pp) | none | §10.3 |
| **Sampling temperature** ({0.0, 0.7, 1.0}) | TR152 | no axis-mean modulation; per-context mean swing 0.152pp | Qwen lean *localizes* here | §10.3, §10.4 |
| **Speculative decoding** (reject / typical accept, N≤12) | TR144 | inert at temp 0 (max \|h\| = 0.024) | none (temp > 0 carved out) | §8.1 |
| **Multi-turn depth** (turn 5) | TR145 (P5) | null (turn-5 McNemar p = 0.22 / 1.0) | wide CI flagged | §8.2 |
| **Compile stack** (Triton minor version) | TR147 | *latency* flips catastrophically (\|d\| ≈ 14–49) | n/a (not a safety knob) | §6.1 |

**Observations.** Seven of the eight knobs are safety-null on the harmful core; the eighth (compile stack) is not a safety knob at all but a *latency* knob, and it is the one that moves catastrophically — which is the arc's quiet structural joke: the knob everyone worries about for safety (precision, batching, speculation) is inert, while the knob nobody records (the Triton minor version) is the one that flips a headline. The only safety footprint anywhere in the table localizes to a single knob (temperature) on a single family (Qwen), and even there it is sub-1pp and on the benign edge. The batch-size and context-length rows are worth dwelling on: these are the two knobs the Phase 5 attack-surface work flagged as the most plausible movers (batch-size non-determinism, long-context exploitation), and TR145's ANOVA finds both approximately *additive* with KV-precision — i.e. whatever each does on its own, it does not *interact* with FP8.

> The knob-by-knob cut is what licenses the word "independent" in "serving-state-independent certificate." It is not that FP8 was tested once and assumed to generalize; it is that each serving knob was turned, separately, against the FP8 contrast, and none of them bent it at the axis-mean. The certificate holds across the *Cartesian product* of the tested knob settings precisely because no knob showed an interaction — the one empirical fact that would have forced per-configuration re-validation.

### 12.4 Synthesis by model family

The arc spans three small-model families plus two large-scale probes. Cutting by family exposes where the single located footprint actually lives.

| Family | Members in the arc | Harmful-core verdict | Benign-edge (XSTest) behaviour | Per-family FP8 MH OR (TR152) |
|---|---|---|---|---|
| **Llama-3.2** | 1B, 3B (TR145/149/152); 8B, 70B-AWQ (TR144/147 probes) | invariant | neutral-to-improving under FP8 (net −8 discordant) | 0.484 [0.202, 1.157] (CI brackets 1.0) |
| **Qwen-2.5** | 1.5B, 3B (TR145/149/152) | invariant | carries 91% of FP8-degraded XSTest pairs; highest baseline over-refusal (37.2% both-unsafe rate on 1.5B) | **3.878 [2.386, 6.302]** (above 1.0) |
| **Phi-3-mini-4k** | 4B (TR152) | invariant | FP8 *improves* (net −10 discordant) | 0.130 [0.024, 0.715] (below 1.0) |
| **Large-scale probes** | Llama-3.1-70B-AWQ (TR144 E1), qwen2.5-7b / llama3.1-8b (TR147) | TR144 E1 refusal 0.839 [0.783, 0.884], overlaps lab-scale band | not measured at scale (TR151 scope) | n/a |

**Observations.** The harmful-core column is uniform — every family refuses every clearly-harmful prompt under FP8 — so the entire family-level story lives on the benign edge, and there the three families *diverge in direction*: Qwen leans toward more over-refusal, Phi-3 leans toward less, Llama is neutral. That divergence is the single most important reason the pooled TR152 OR of 1.88 must not be read as "FP8 degrades safety": pooling three families whose effects point in three directions produces an aggregate that is a property of the *family mix*, not of FP8. The per-family Mantel–Haenszel decomposition (§10.4) is what dissolves the aggregate back into its parts, and the parts say FP8's only adverse footprint is a nudge to a pre-existing Qwen tendency. The large-scale row is deliberately thin: TR144's E1 confirms the *speculative-decoding* null survives to 70B, but the *FP8* line's scale validity above 4B is unmeasured and explicitly handed to TR151 — the family cut makes that boundary impossible to paper over.

> The family-by-family reading converts a scary-looking pooled OR into an actionable per-family footnote. "FP8 KV-cache slightly increases Qwen-2.5's over-refusal of benign-edge prompts" is something a team running Qwen can monitor and a team running Llama or Phi-3 can ignore. A synthesis that reported only the pooled 1.88 would have manufactured a generic FP8 warning out of a Qwen-specific, benign-edge, sub-1pp lean — exactly the over-claim the claim ladder's Licensed tier exists to prevent.

### 12.5 Synthesis by judge cohort

Every safety verdict in the arc is a judge measurement, so the verdicts are only as stable as the cohort that produced them. Cutting by judge shows where agreement is load-bearing and where it is expected to be low.

| Judge / pair | Role | Agreement observed | Reading |
|---|---|---|---|
| **regex (anchor)** | surface refusal-pattern detector | F1 = 0.7109 vs corpus majority (TR148); κ ≈ 0 vs gemma3 on TR144 | a stable cheap anchor, not a semantic judge; its κ-with-LLM varies by corpus balance |
| **gemma3:12b** | primary general judge | F1 = 0.7652 (highest); κ = 0.6917 with llama3.1 (TR145 corpus), 0.8306 (TR149), 0.831 (TR152) | the workhorse; clears JTP robust on clean corpora, triangulate on mixed |
| **llama3.1:8b** | second general judge | F1 = 0.5260; the cross-family partner in the binding κ | independent family → the κ-with-gemma3 *is* the JTP verdict |
| **shieldgemma:9b / llama-guard3:8b** | composite-harm specialists | cross-axis κ = −0.13 to −0.26 vs general judges; +0.21 with each other | a *different construct* (Layer 1b), not a broken judge — triangulate within axis only |
| **gpt-4o (anchor only)** | external calibration | κ = 0.877 with gemma3 at n = 94 (disagreement-pool subset) | umbrella-gated; present as a calibration anchor, excluded from the corpus-scale verdict |
| **Claude** | external (deferred) | — | Fellowship-gated; the named fourth axis |

**Observations.** The judge cut surfaces the dual-axis finding's full consequence: of the five executed judges, three (regex, gemma3, llama3.1) measure *response refusal* and two (shieldgemma, llama-guard3) measure *composite harm*, and the program's entire JTP verdict rests on the cross-family κ *within* the refusal axis (gemma3 × llama3.1). The specialists' negative cross-axis κ is not noise to be averaged away — it is a category boundary, and the protocol's rule is to triangulate within an axis and run the composite-harm screen in parallel (Layer 1b), never to fold a specialist into the refusal-axis κ matrix. The corpus-dependence of gemma3 × llama3.1 (0.6917 → 0.83) is the second load-bearing fact: the same pair, same prompt template, clears robust on clean adversarial batteries and only triangulates on mixed task sets, which is why dispatch is pinned to the *calibration* corpus and never self-selected on the corpus that flatters the judges.

> The judge cohort is the one place the umbrella gate is most visible in the synthesis. Two of the four cross-family axes the full JTP matrix would use (gpt-4o, Claude) are credential-gated, so every executed verdict is a two-LLM-judge measurement. That this two-judge agreement clears κ ≥ 0.83 on the standardized and serving-state corpora is what lets the arc proceed at $0 external cost without the verdict hinging on the missing axes — but the synthesis states the missing axes as named deferrals, never as closed questions, because a reviewer is entitled to know the κ matrix is two-wide, not four-wide.

### 12.6 Synthesis by effect-size regime

Because the arc is a null line, the most honest single-number summary is the *largest* effect anyone found. Collecting the maxima per report shows how far below the triviality threshold the whole line sits.

| Report | Largest standardized effect observed | Threshold | Headroom |
|---|---|---|---|
| TR144 (TAIS) | max \|Cohen's h\| = 0.024 (18 AdvBench contrasts) | 0.1 (TAIS null cutoff) | ~4× below |
| TR152 (factorial) | max \|Cohen's h\| = 0.0458 (qwen2.5-1.5b, temp=0.7) | 0.2 ("small") | ~4× below |
| TR149 (standardized) | max \|Cohen's h\| = 0.0742 (HarmBench, qwen2.5-1.5b) | 0.2 ("small") | ~3× below |
| TR145 (base) | lone TOST failure Δ = −3.09pp (Qwen-1.5B), fails ±3pp by 0.09pp | ±3pp | misses by 0.09pp |
| TR147 (CRI) | cross-version \|Cohen's d\| ≈ 48.9 (latency, *not* safety) | 10 (catastrophic) | ~5× above |
| TR146 (probes) | refusal-direction *magnitude* r = −0.61 (model-level, not per-config) | \|r\| > 0.15 (any signal) | the one probe signal, but not a per-config diagnostic |

**Observations.** Two regimes are visible and they could not be further apart. The safety contrasts (TR144/149/152, and TR145's near-miss) all sit *well inside* the triviality bands — the largest safety effect in the entire arc is a Cohen's h of 0.0742, more than 2.5× below the "small" threshold, and the single TOST failure misses equivalence by nine *hundredths* of a percentage point. The latency contrast (TR147) sits 5× *above* the catastrophic threshold. Reading the two regimes together is the arc's central irony made quantitative: the standardized-effect-size telescope, pointed at the safety question, sees nothing; pointed at the reproducibility question, it sees an effect so large the distributions do not overlap. TR146's row is the careful exception — its one above-floor signal (refusal-direction magnitude, r = −0.61) is a *model-level* property identical across a model's quant variants, so it is a vulnerability factor of the base model, not the per-configuration diagnostic the probe was hoping for, and it does not separate dangerous from safe configs (Mann–Whitney p = 0.979).

> An effect-size cut is the most reviewer-resistant way to summarize a null line, because it forecloses the "absence of evidence" objection. The line does not merely "fail to reach significance"; its *largest* observed safety effect, across 64,855 + 7,578 + 20,754 paired observations, is a Cohen's h of 0.0742. When the biggest thing you can find is a quarter of the threshold for "small," the null is measured, not assumed — and the same telescope finding a |d| of 49 on the latency axis proves the instrument can see an effect when one is there.

### 12.7 Synthesis by null-vs-located outcome

The arc is not uniformly null — it has exactly one located positive, and stating precisely where the line crosses from H0 to H1 is what keeps the certificate from over-claiming in either direction.

| Contrast | Outcome | Why it lands there |
|---|---|---|
| Speculative decoding @ temp 0 (TR144) | **H0** (null) | target verification overrides the draft; distribution-preservation holds at greedy argmax |
| FP8 single-config harmful core (TR145) | **H0** (null) | every primary McNemar non-significant; pooled OR brackets 1.0 |
| FP8 standardized harmful core (TR149) | **H0** (null) | 12/12 TOST-equivalent; discordant split near-even (12 vs 15) |
| FP8 serving-state harmful core (TR152) | **H0** (null, exact) | 0/8,976 discordant — zero off-diagonal mass on the harmful batteries |
| FP8 serving-state benign edge (TR152 XSTest) | **H1** (located) | pooled OR 1.88 clears 1.0, but 100% XSTest-bound, 91% Qwen-bound, sub-1pp |
| FP8 serving-state interaction (TR152) | **H2 rejected** | interaction spread 2.99pp inside ±3pp; per-context mean within 0.17pp of zero |
| Mechanistic-probe predictivity (TR146) | **H0 affirmed as F3** | four probes all |r| < 0.15; the shortcut is falsified, not merely unconfirmed |

**Observations.** The line crosses from H0 to H1 at exactly one place — the benign-edge battery under FP8 — and crucially does *not* cross to H2 (a serving-state interaction). That two-part precision is the whole payload of the factorial: H1 (a located effect) can be carried as a per-family footnote on a single certificate, whereas H2 (an interaction) would have forced per-configuration re-validation and destroyed the "serving-state-independent" claim. The factorial was built specifically to separate these two, and the separation is clean: the located effect is real (sign test p ≈ 0.0004) but confined, and no axis bends the contrast at its mean. TR146's row is the inverse case — a hypothesis the program *wanted* to be true (cheap probes predict safety) and the evidence rejects, which is why it sits on the Forbidden tier rather than as an open null.

> "One located positive, no interaction, everything else null" is the most defensible shape a serving-safety arc can have, because it is neither the implausible "everything is perfectly fine" nor the alarmist "FP8 degrades safety." It says: the harmful core is invariant, there is exactly one small benign-edge footprint on one family, and no serving knob modulates the contrast — so a deployment gets one certificate plus one monitorable footnote, which is precisely what an operations team can act on.

### 12.8 Synthesis by measurement methodology

Finally, cutting by estimator shows that the verdicts are not artefacts of a single statistical choice — each verdict is carried by the estimator appropriate to its table type, and the arc documents what happens when the wrong one is used.

| Verdict | Estimator that carries it | Why this estimator | Documented failure of the wrong one |
|---|---|---|---|
| "no significant per-cell effect" | McNemar exact (paired) | conditions on discordant pairs; the only test with power on a concordant-heavy corpus | a χ² of independence would be swamped by the concordant diagonal |
| "equivalent within ±3pp" | TOST | converts a null into a *positive, bounded* claim | bare McNemar non-rejection is consistent with a large undetected effect |
| "pooled across strata" | Haldane-MH, discordant-ratio form for paired cells | finite under zero discordant counts; correct for paired tables | TR149's unpaired cross-product fed paired cells → pooled OR 3411.5 |
| "effect size negligible" | Cohen's h (paired-binary) | the standardized effect for a difference of proportions | Cohen's d would mis-scale a proportion difference; reserved for TR147 latency |
| "reproducibility band" | max pairwise Cohen's d (continuous) | latency is genuinely continuous | Cohen's h on latency, or d on a safety proportion, is a category error |
| "judge trust regime" | Cohen's κ (chance-corrected) | corrects raw agreement for chance; PABAK handles zero-variance corpora | raw agreement would read a degenerate all-safe corpus as perfect |

**Observations.** The methodological discipline is that the *table type* dictates the estimator, and the arc contains a worked counterexample for the two most error-prone choices. The matched-pairs-vs-unpaired OR distinction is the sharpest: TR145 builds genuine unpaired marginal tables and correctly uses the cross-product MH form, while TR149 and TR152 build paired cells and use the discordant-ratio form — and TR149's postmortem documents what happens when paired cells are fed the unpaired formula (the pooled OR explodes to 3411.5 on an all-null corpus, because the dominant concordant mass routes into the numerator). The h-vs-d distinction is the second: every binary safety contrast uses h, the one continuous latency contrast (CRI) uses d, and the synthesis flags that "fixing" one to match the other would *introduce* a bug. The κ row closes the loop with PABAK handling the degenerate-corpus case (StrongREJECT's all-safe slice, where κ = −0.0005 is correctly read as PABAK = 0.9979, not as disagreement).

> The estimator cut is the reviewer's deepest audit surface. It demonstrates that no verdict in the arc rests on a single statistical convenience: each is carried by the estimator matched to its data structure, the program documents the failure mode of the obvious wrong choice for the two riskiest decisions, and the cross-estimator consistency (McNemar non-significance, TOST equivalence, pooled OR bracketing 1.0, and negligible h all agreeing on the harmful core) is what makes the null robust to "you picked the test that gave you the answer you wanted."

---

## 13. Integration with Phase 5 (TR138–143) and Phase 5 (TR155 / TR163)

Phase 6 is the middle link of a three-phase safety arc, and a conclusive synthesis is incomplete until it states, concretely, which prior-phase findings it confirms or re-grounds and how its substrate licenses the phase that follows. §1.3 sketched the trajectory; this section develops it at the level of specific reports and specific dependencies.

### 13.1 What Phase 5 established, and what Phase 6 does with it

Phase 5 (TR138–143) mapped the *attack surface*: the set of places where inference optimization and adversarial pressure can move refusal behaviour. Its members and their bearing on Phase 6:

| Phase 5 report | What it established | Phase 6's relationship |
|---|---|---|
| **TR138** (batch-inference safety) | batch-size non-determinism can shift refusal outputs; the seed of the inference-flag-null-line question | Phase 6's TR152 turns the batch knob *against the FP8 contrast* and finds it additive (no interaction) — the attack-surface flag, re-grounded under the validity gates |
| **TR141** (refusal fragility) | cross-architecture refusal fragility; 18 models, 127,224 records | supplies the fragility prior the RTSI screen (Layer 2) operationalizes; TR152's per-family Qwen lean is consistent with its fragility ranking |
| **TR139** (multi-turn × quant) | multi-turn jailbreak pressure under quantization | TR145's P5 turn-5 null is the FP8-KV-cache slice of the same question, bounded to 5 turns |
| **TR142** (RTSI) | the Refusal Template Stability Index behavioural screen | **directly inherited** as the Layer 2 partner of TAIS; Phase 6 does not re-derive it, it composes with it |
| **TR140** (JTP lineage / ML-CQ) | cross-family judge κ calibration (gemma3 × Claude κ = 0.925 on the TR140 corpus) | the *pre-registration* TR148's JTP thresholds inherit verbatim; TR148 tests whether that single near-perfect calibration generalizes (it does, within band: κ drops to 0.6917 on a different corpus) |
| **TR143** | cross-request composition under continuous batching | the composition prior TR152's star factorial probes at the serving-state level |

**Observations.** The relationship is not "Phase 6 supersedes Phase 5"; it is "Phase 6 validates the instruments Phase 5's maps were drawn with, and re-tests the most-deployed flag under them." The clearest case is TR140 → TR148: Phase 5 produced a single near-perfect judge calibration (κ = 0.925), and the implicit premise of every shipped Phase 5 safety verdict was that this calibration generalized. TR148 tested that premise and found it holds *within band* — the same framework reproduces a κ of 0.69 on a different corpus and a different cross-family pair, which is exactly the cross-corpus result the JTP framework was built to surface, not a failure of it. The second clear case is TR138 → TR152: Phase 5 flagged batch-size as a plausible safety mover, and Phase 6's factorial turns that exact knob against the FP8 contrast and finds it additive — the attack-surface hypothesis is not refuted (batch-size non-determinism is real) but its *interaction* with FP8 is bounded null.

> Phase 6's integration with Phase 5 is the reason the program's earlier nulls are now defensible rather than provisional. A Phase 5 "no safety change under batching" finding measured by a single un-triangulated judge on a bespoke corpus would not survive the JTP gate or the corpus-standardization gate Phase 6 introduces. Phase 6 retroactively *grounds* the Phase 5 maps by supplying the validity layer they were drawn without — which is why a measurement-validity phase is worth a conclusive synthesis even though it produces "only" nulls.

### 13.2 What Phase 6 licenses for Phase 5

Phase 7 (TR155/TR163 onward) is the *mitigation* phase. Its first MVP feasibility demo, TR163, is an RTSI-gated quantization-routing policy run as offline LOOCV reanalysis over TR142's 51-cell RTSI table — a proof-of-mechanism showing the routing idea works, not a paper-grade campaign. Paper-scale expansion required before any promotion or downstream synthesis. The demo recovers ~76% of the refusal gap at roughly 22% of requests routed to the safe path, with a leave-one-out cross-validated routing AUC of 0.84. That instrument inherits three things directly from Phase 6:

1. **The right to trust RTSI.** TR163 routes on RTSI; RTSI is a Layer 2 behavioural screen whose discipline (behavioural, not mechanistic) is exactly what TR146's negative control established. A mitigation that routed on a *mechanistic* probe would be building on the shortcut TR146 falsified.
2. **The judge cohort and its validated floor.** TR163's recovery number is a judge measurement, and it is only meaningful because the JTP gate (Layer 1) established that the cohort scoring it clears the triangulate threshold. The mitigation does not get to claim "76% recovery" without the κ floor underneath it.
3. **The behavioural-measurement discipline end to end.** Phase 6's central methodological wager — certify the *composed, served* system by measuring its behaviour, not by inspecting its internals — is the substrate the mitigation is both built and validated on.

**Observations.** The dependency runs strictly one way: Phase 6 does not mitigate anything (it is a validity-and-null phase), but every Phase 5 mitigation consumes a Phase 6 instrument. TR163's RTSI gate is the cleanest example — RTSI's *right to be trusted as a routing signal* comes from the behavioural-screen discipline Phase 6's negative control (TR146) established, and its *recovery number* comes from a judge cohort whose validity floor Phase 6's JTP gate measured. Without Phase 6, TR163 would be "a routing policy that recovers 76% of the gap according to one judge we didn't validate, gated on a screen we didn't establish as behavioural" — which is not a defensible mitigation claim.

> The three-phase arc reads as a single sentence: Phase 5 found *where* the serving state can move safety, Phase 6 validated the instruments and showed the most-deployed flag (FP8 KV-cache) does not move the harmful core under them, and Phase 5 is now mitigating the located vulnerabilities using screens and judges that Phase 6 certified. Phase 6 is the phase that converts the program from "one lab's measurements" into "measurements under validated instruments" — and that conversion is what every downstream mitigation stands on.

### 13.3 The measurement-count and cost ledger across the integrated arc

Stating the realized scale across the three phases makes the integration concrete and guards against the in-memory totals that drift fast.

| Phase | Representative scale (this arc's members) | External judge cost |
|---|---|---|
| Phase 5 (TR138–143) | TR141 alone: 127,224 records / 18 models; TR139: 10,600 convos / 37,825 labels; TR140: ~91K controls | mixed (pre-umbrella adjudication packs via OpenAI Batch API) |
| **Phase 6 (this synthesis)** | TR144 64,855 paired; TR145 24,054; TR146 5,100 passes; TR147 52,410 latency rows; TR148 13,724 re-judged; TR149 7,578; TR152 45,000 responses / 135,000 judge labels | **$0** (umbrella gate; local Ollama cohort) |
| Phase 7 (TR155/TR163) | TR163 MVP feasibility demo (offline LOOCV reanalysis, ~16 KB JSON, AUC 0.84); TR155 single-axis pilot (936 records) — both NOT paper-grade; expansions queued | local cohort |

**Observations.** The cost row is itself a deliverable: Phase 6's entire executed set ran at $0 external API cost because the umbrella gate keeps adversarial corpora on the local Ollama cohort, with the only metered spend being RunPod GPU time for the cells that exceed the local 12GB envelope. That cost posture is what licenses the program's volume-and-velocity strategy — a certification substrate that required a cloud safety-judge budget per corpus could not have been built at this scale on this timeline. The scale row guards the canonical-count discipline: the in-memory program total drifts fast, so per-report realized counts are stated here and the canonical aggregate is deferred to `BANTERHEARTS_MEASUREMENT_COUNT.md`.

> The integrated ledger is the answer to "why does a null-heavy validity phase justify this much measurement?" The seven Phase 6 reports together put well over 200,000 primary and judge measurements on disk at $0 external cost, and the reason that volume is necessary is the same reason the phase exists: a *measured* null — powered, bounded, triangulated — requires far more evidence than an unmeasured "we didn't see anything," and the whole point of Phase 6 is that its nulls are measured.

### 13.4 The complete decision stack across the three phases

Read end to end, the three phases compose into a single deployment decision flow rather than three disconnected result sets. Phase 5 supplies the *map* (where the serving state can move refusal), Phase 6 supplies the *gates* (whether the instruments that drew the map can be trusted, and whether the most-deployed flag moves the harmful core under them), and Phase 5 supplies the *mitigations* (what to route or retrain when a gate fires). The stack below states, for an operator standing up a quantized serving deployment, which phase answers each question in sequence.

| Decision step | Question the operator asks | Phase that answers it | Verdict instrument |
|---|---|---|---|
| 1. Is my latency number real? | "Did the compile stack reproduce, or is my benchmark a Triton-version artefact?" | Phase 6 (TR147) | CRI band (robust < 0.5 / fragile ≥ 2 / catastrophic ≥ 10) |
| 2. Is my safety label real? | "Do my judges agree enough to ship a single-judge verdict, or must I triangulate?" | Phase 6 (TR148) | JTP κ band (robust ≥ 0.70 / triangulate / untrustable) |
| 3. Does the serving state move the harmful core? | "Will FP8, batching, prefix caching, or speculation leak unsafe completions?" | Phase 6 (TR144/TR145/TR149/TR152) + Phase 5 (TR138/TR143) | matched-pairs Cohen's h + ±3pp TOST; H1-vs-H2 factorial |
| 4. Does the serving state move the *benign edge*? | "Will the deployment over-refuse safe-but-alarming prompts, and on which family?" | Phase 6 (TR152 XSTest slice) + Phase 5 (TR141 fragility) | per-family MH OR on the over-refusal battery |
| 5. Is the model fragile under quantization at all? | "Which (model, quant) cells are template-unstable before I even serve them?" | Phase 5 (TR142 RTSI) inherited as Layer 2 | RTSI score (0.10 / 0.40 cutoffs) |
| 6. Can I read safety off the weights to skip the judge? | "Is there a cheap forward-pass probe that predicts danger?" | Phase 6 (TR146) | **No** — F3 forbidden; behavioural measurement is mandatory |
| 7. When a gate fires, what do I do? | "How do I recover refusal without abandoning quantization?" | Phase 5 (TR163) | RTSI-gated routing (≈76% gap recovery, LOOCV AUC 0.84) |

**Observations.** The stack is strictly ordered: each step is only meaningful if the step above it passed. Step 3's "the harmful core is invariant" claim is worthless if step 2's judge floor was never measured, and step 2's κ is worthless if step 1's latency-reproducibility gate was never run (a fragile compile stack means the system under measurement is not even the system that will be served). The single most important structural fact the stack encodes is the position of step 6: the temptation to skip steps 2–4 by reading a cheap internal signal is exactly what TR146 forecloses, which is why the negative control sits *across* the whole stack rather than at one layer. The operator does not get a shortcut; the price of a trustworthy verdict is the full behavioural pipeline, and the arc's contribution is to make that price defensible rather than optional.

> The decision stack is also the cleanest statement of why Phase 6 is a *phase* and not a collection of TRs. A reader could take TR147, TR148, TR149, and TR152 as four separate nulls-and-calibrations; the stack shows they are four steps of one certificate, and that the certificate composes upward with Phase 5's map and downward into Phase 5's mitigations. Removing any one step does not weaken one result — it breaks the chain of custody for every result below it.

### 13.5 Findings Phase 6 refines or re-grounds from Phase 5

Integration is not only confirmation; in three places Phase 6's gates *refine* a Phase 5 framing that was defensible but under-qualified, and stating the refinements explicitly is what keeps the arc honest about what changed.

- **The safety-specificity question, sharpened.** Phase 5's attack-surface work raised the "disproportionate-harm" hypothesis — that inference optimization might damage safety *more* than capability. Phase 6 supplies the cleanest refutation of the strong form on the FP8 flag: TR145's P2 returns the one Holm-significant outcome of the phase as *capability* on Qwen (p = 0.0018, OR 1.89) while Qwen *safety* holds at p = 0.31 (§8.2), and TR146 then shows the safety-neuron damage is real (1.40×) but non-predictive of danger status (p = 0.979, §9.2). The refined statement is precise: FP8 does have a measurable footprint on some models, but it is *not safety-targeted* — capability moves while the harmful core holds, and the internal mechanism that would make it safety-targeted does not separate dangerous from neutral configs.
- **The single-calibration premise, re-grounded.** Phase 5's TR140 produced one near-perfect judge calibration (gemma3 × Claude κ = 0.925), and every shipped Phase 5 safety verdict implicitly assumed it generalized. TR148 tests the premise directly and finds it holds *within band* — the same framework reproduces κ = 0.6917 on a different corpus and a different cross-family pair (§6.2), and TR149 then shows the κ is itself corpus-specific, tightening to 0.83 on clean adversarial batteries (§7.4b). The refinement is that judge agreement is not a fixed property of the framework; it is a property of the *(judge cohort × corpus)* pair, which is why the dispatch decision must be pinned to a calibration corpus and never self-selected.
- **The "no safety change under batching" finding, validated.** Phase 5's TR138/TR143 reported that batch-size non-determinism and cross-request composition do not change aggregate safety. Phase 6's TR152 re-runs the batch knob *against the FP8 contrast* under the JTP gate and the corpus-standardization gate, and finds the interaction additive (H2 rejected, §10) — so the Phase 5 batching null is now a *gated* null rather than a single-judge bespoke-corpus null. The finding did not change; its defensibility did.

**Observations.** None of these three is a reversal — Phase 6 does not overturn a Phase 5 result. What it does is convert three *provisional* Phase 5 framings (disproportionate harm plausible-but-untested, single calibration assumed-to-generalize, batching null measured-without-a-validity-gate) into *bounded* statements with the qualifier attached. The pattern is the same in all three: Phase 5 asked the right question and got a defensible first answer; Phase 6 supplied the instrument that says how far that answer travels. This is exactly the work a measurement-validity phase is supposed to do, and it is why the synthesis treats Phase 6 as the phase that "grounds" Phase 5 rather than the phase that "supersedes" it.

> The refinement most likely to matter downstream is the second one. A program that assumed a single κ = 0.925 calibration generalized would have shipped single-judge verdicts on every corpus; the TR148/TR149 finding that κ is corpus-specific is what mandates the triangulate cohort on mixed corpora and licenses single-judge dispatch only where the corpus has been shown to support it. That is a standing operational rule, not a one-report result, and it propagates to every safety verdict the program ships after Phase 6.

### 13.6 Remaining gaps across the integrated arc

A conclusive synthesis must state what the three phases together have *not* closed, so the deferrals are documented rather than silent. The gaps below are the union of the per-report threats (§15) read across the arc, ordered by how directly each bounds a shipped claim.

| Gap | What is unclosed | Which claim it bounds | Where it is scheduled |
|---|---|---|---|
| **Scale beyond ~4B** | the FP8 safety null is anchored only on 1B–3B (TR149) and 70B-AWQ on the speculation axis (TR144 E1); the 7B–72B FP8-KV-cache range is unmeasured | the Layer 4 scale-validity claim is explicitly bounded to 1B–3B | TR151 (cloud-gated, GO/NO-GO 2026-10-24) |
| **Stochastic decoding** | every null-line result is at temperature 0 (greedy); at temperature > 0 the typical-acceptance criterion genuinely alters the output distribution | the TAIS null must not be extrapolated to sampling settings | TR144 E6 temperature sweep (unrun) |
| **Speculation topology** | only linear draft–verify is tested; EAGLE / Medusa / tree speculation is uncovered | the TAIS certificate covers linear speculation only | future expansion |
| **Quant-format coverage** | mechanistic probing is AWQ/GPTQ INT4 only; no GGUF k-quants (the Q2_K/Q3_K_S danger zone Phase 5 flagged) | the F3 mechanistic-dead-end claim is bounded to INT4 schemes | future TR146 extension |
| **Joint serving-state interactions** | TR152 is a *star* factorial (one factor at a time around a center), so two-axis interactions (e.g. batch × temperature jointly) are unmeasured | the Layer 5 "additive" claim covers single-axis modulation only | resolution-IV/V design (not yet scoped) |
| **External judge axes** | the gpt-4o and Claude cross-family axes are credential-deferred; the JTP verdict stands on two LLM judges | the κ floor is a two-LLM-judge measurement | Researcher Access Program / Fellowship |
| **Semantic rejudge of expansion cells** | TR144's E1–E5 expansion is regex-scored; a semantic (gemma3/Claude) rejudge is queued | the expansion's byte-identity claims are strong; its semantic-equivalence claims await rejudge | queued |

**Observations.** The gaps cluster into two kinds, and the distinction matters for how an operator should read the certificate. The first kind is *envelope* gaps — scale, decoding temperature, speculation topology, quant format — where the claim is sound *within* the tested envelope and the gap is simply the boundary of that envelope, scheduled forward (most prominently TR151 for scale). The second kind is *instrument* gaps — the two missing judge axes, the queued semantic rejudge — where the verdict is established but its validity floor would tighten with more instruments. Neither kind is a hidden weakness: every one is enumerated in a per-report threats section and tagged to where it is scheduled. The single largest gap by downstream consequence is scale: the entire FP8 safety line is anchored on ≤3B (plus the 70B speculation cell), and the production-relevant 7B–72B range is the explicit scope of TR151, which is why the certificate's scale-validity layer is the one most prominently marked "anchored for 1B–3B, handed forward."

> The honest statement of remaining gaps is what separates this synthesis from an over-claim. A weaker report would present the five-layer certificate as complete; the truthful version is that Layers 1–3 and 5 are anchored on the executed envelope, Layer 4 is anchored for 1B–3B with the 7B–72B half explicitly deferred to TR151, and the whole line is bounded to greedy decoding and linear speculation. Stating the gaps in one table, each tagged to its scheduled closure, is the same discipline as the claim ladder: it is the reason a reviewer can trust the supported claims, because the unsupported ones are named rather than blurred.

---

## 14. Numbered Findings

The synthesis distils to a set of numbered findings, each tagged to its claim-ladder tier and its source chapter. They are ordered by the dependency spine — validity findings first, then the null line, then the located positive, then the integration findings — so that a reader can cite any one in isolation and a reviewer can check each against its evidence.

**F1 (Supported, §6.1).** On a single fixed RTX 6000 Ada GPU running one fixed model, changing only the Triton minor version (3.3.1 → 3.4.0 → 3.6.0) collapses the `torch.compile` prefill speedup from 62–77% to ≤3.4% and erases an 80% `reduce-overhead` decode crash; the cross-version compile-vs-eager effect sizes span |Cohen's d| ≈ 14–49, while the eager control is flat (|d| ≤ 0.15). A cross-stack latency conclusion is therefore not reportable without pinning the Triton version *and* the repository code SHA.

**F2 (Supported, §6.1).** The CRI formalizes F1 as the maximum pairwise |Cohen's d| on compiled latency across a stack-perturbation set, with bands robust (<0.5) / sensitive (<2) / fragile (≥2) / catastrophic (≥10); the Triton ablation lands in catastrophic. CRI returns `invalid` (not a band) when fewer than two stack points survive, as in the `gpt-fast` pinned-commit edge case.

**F3 (Supported, §6.2).** On the 13,724-record TR145 safety subset, the binding cross-family judge pair gemma3:12b × llama3.1:8b agrees at Cohen's κ = 0.6917 (95% CI [0.6824, 0.7008], n = 12,809) — the JTP *triangulate* bucket, 0.0083 below robust — so every downstream judge-labeled verdict must run multi-judge majority vote rather than trust a single judge.

**F4 (Supported, §6.2).** Safety-specialist judges (shieldgemma:9b, llama-guard3:8b) anti-correlate with general judges (four cross-axis κ = −0.13 to −0.26) while agreeing with each other (κ = +0.21), because the specialists score composite prompt+response harm and the general judges score response refusal. They are two orthogonal axes; triangulation must be performed *within* an axis, splitting Layer 1 into a refusal-axis gate (1a) and a composite-harm screen (1b).

**F5 (Supported, §6.2).** A mandatory-judge join is a silent verdict-flipping bug: TR148 v1 required a gpt-4o label per record, restricting the analysis to a 94-record subset and producing the wrong verdict ("robust, single-judge"); the v2 fix (drop the mandatory `continue`, select the primary pair by largest n) recovers the correct triangulate verdict at n = 12,809.

**F6 (Forbidden / F3-tier, §9).** Four mechanistic probes — first-token entropy, refusal-direction geometry, calibration drift, and safety-neuron quantization error — all fail to distinguish safe from dangerous quantized configurations (all |r| < 0.15 with RTSI; danger-vs-neutral Mann–Whitney p = 0.979). Safety cannot be read off the weights; the verdict must be behavioural.

**F7 (Supported, §9.2).** The safety mechanism is real but necessary-not-sufficient: safety-critical neurons absorb 1.40× disproportionate quantization error (p < 0.0001, GPTQ 1.45× vs AWQ 1.37×), yet that ratio does not predict hidden-danger status (neutral rows sit above several danger rows). The GPTQ confidence paradox compounds this — GPTQ models grow more first-token-confident while failing to refuse, so any logit/confidence screen would approve the most dangerous configs.

**F8 (Supported, §8.1).** Speculative decoding is behaviourally inert at greedy decoding: max |Cohen's h| = 0.024 across 18 AdvBench contrasts (64,855 paired samples), 90.66% byte-identity under rejection sampling, literal-zero per-task deltas under typical acceptance, and flat dose-response across speculation length. Byte-identity survives an adversarially-DPO'd draft (E2, 100%), a quantized draft (E3, 100%), and seed changes (E4, 100%); bf16 (E5) shifts 36–53% of bytes yet holds the safety null (max |h| = 0.054).

**F9 (Supported, bounded, §8.2).** FP8 (E4M3) KV-cache at a single serving configuration produces no statistically supported safety degradation across three 1B–3B models and a five-phase paired battery (24,054 records): every primary McNemar non-significant, both context- and batch-interaction ANOVAs approximately additive, pooled OR 1.05 [0.90, 1.23]. The lone TOST non-equivalence is Qwen-1.5B (Δ = −3.09pp, missing ±3pp by 0.09pp) — the seed of the located finding TR152 resolves.

**F10 (Supported, §7).** The FP8 null replicates on the four literature-canonical batteries (HarmBench, JailbreakBench, StrongREJECT, XSTest): 0 of 12 cells significant after Holm, 12 of 12 TOST-equivalent at ±3pp, pooled OR 0.8065 [0.3828, 1.6989], I² = 0.0%. Discriminating power is effectively a two-battery claim (HarmBench + XSTest) because the other two saturate at the refusal ceiling on 1B–3B models.

**F11 (Licensed, §7.4).** The JTP verdict is corpus-specific: the same gemma3 × llama3.1 pair scores κ = 0.8306 on the clean standardized batteries versus 0.6917 on TR145's mixed task set. Clean unambiguous refusals are easier to judge; dispatch must be pinned to the calibration corpus, never self-selected on the corpus where the judges happen to agree best.

**F12 (Supported, §10.2).** FP8 KV-cache is *exactly* invariant on the harmful core across the tested serving state: 0 of 8,976 matched harmful-battery pairs discordant across 5 models × 6 serving contexts × 3 batteries, FP16-only safe rate 1.0000. This is the serving-state-independent harmful-refusal guarantee, resting on 8,976 paired observations with zero counterexamples.

**F13 (Licensed, §10.3–10.4).** The only measurable FP8 footprint is a located benign-edge over-refusal lean: all 133 discordant pairs are on XSTest (pooled OR 1.8817 [1.3185, 2.6855], sign-test p ≈ 0.0004), 91% Qwen-bound (per-family MH OR Qwen 3.878 vs Llama 0.484, Phi-3 0.130), max |h| = 0.0458, concentrated on the temperature axes. It is H1-located, not H2-interaction: the interaction spread is 2.99pp inside the ±3pp band, so no serving axis modulates the contrast at its mean.

**F14 (Supported, §10.5).** The TR152 v1→v2 arc is a textbook resolution lift, not a flip: scaling 14,400→45,000 records regressed the pooled OR toward 1.0 (2.69→1.88), tightened the CI ~4× (5.53→1.37), and moved the lower bound *further* from 1.0 (1.09→1.32), while the qualitative shape (Qwen-family, XSTest-only, temperature-amplified) stayed invariant. v2 also corrected a v1 attribution by log inspection: the speculative-decoding cell failure is a vLLM argparse rejection, not the OOM v1 claimed.

**F15 (Supported, §13).** Phase 6 retroactively grounds Phase 5: TR148 confirms the TR140 judge calibration generalizes within band (0.925 → 0.69 cross-corpus), and TR152 re-grounds the TR138 batch-size attack-surface flag as additive (no FP8 interaction). It forward-licenses Phase 5: TR163's RTSI-gated mitigation (76% gap recovery, LOOCV AUC 0.84) inherits its right to trust RTSI from TR146's behavioural-only discipline and its recovery number from the JTP-validated judge cohort.

**Observations.** The fifteen findings partition exactly as the claim ladder requires: eleven Supported, two Licensed (F11, F13 — each bounded by an explicit scope limit), one Forbidden (F6), and one Supported-but-bounded base case (F9). The two Licensed findings are the ones a careless reader would over-state — F11 into "judges are robust" (they are robust *on clean corpora*) and F13 into "FP8 degrades Qwen safety" (it nudges benign-edge *over-refusal*, not harmful compliance). The single Forbidden finding (F6) is the one the program *wanted* to be true and the evidence rejects.

> The numbered findings are the citable surface of the synthesis: a bridge paper can lift any F-number with its tier and its bound already attached, and a reviewer can trace each to the chapter and the source-section that backs it. No finding is stated without its envelope in the same sentence, which is the discipline that lets the arc extend its claims downstream rather than retract them under questioning.

---

## 15. Threats to Validity (Phase 6-Wide)

The per-TR threats are enumerated in each chapter; this section consolidates the threats that cut *across* the arc and states how the design contains each.

### 15.1 Scope boundaries shared by the whole line

| Threat | Extent | Containment |
|---|---|---|
| **Model scale** | every executed TR is 1B–4B; production runs 7B–70B | bounded claim ("1B–4B range"); TR151 scaffolded to extend Layer 4 to 7B–72B |
| **Single hardware SKU** | RTX 4080 Laptop 12GB (local) for TR145/149/152; Ada/cloud for TR147/144 expansion | CRI (TR147) is the explicit cross-stack-reproducibility gate; cross-SKU validity is a named TR151 prerequisite |
| **Single serving stack** | vLLM v0.19.1 pin across the FP8 line | the pin is what makes TR149 a clean replication of TR145; cross-version validity is out of scope by design |
| **Temperature** | TR144's null and TR145's base case are temperature-0 only | TR152 *extends* to temp 0.7/1.0 and finds the Qwen lean lives on the temperature axes; TR144's E6 temp sweep is the named unrun extension |
| **Context length** | ≤2,048 tokens across the line | TR150 scaffolded for 8k–128k; the long-context regime is explicitly excluded, not silently assumed |

**Observations.** The shared boundary is consistent: *1B–4B, single SKU, vLLM v0.19.1, short context, greedy-to-moderate temperature.* Every claim in the arc is bounded to that envelope, and every boundary has a named scaffold (TR150/TR151/TR153) designed to extend it. None of the boundaries is a hidden assumption — they are enumerated in each report's scope-discipline section and re-stated here.

### 15.2 Statistical and measurement threats

- **Ceiling effects on harmful batteries.** Three of four standardized batteries (JailbreakBench, StrongREJECT, and XSTest's unsafe slice) saturate at 100% refusal on 1B–4B models, so they carry zero discriminating power on this model set (TR149 SS1; TR152 SS8). The cross-battery nulls therefore rest on the *non-saturated* batteries (HarmBench + XSTest's over-refusal slice) and on TOST equivalence, not on underpowered per-cell McNemar. This is the single most important reason TR151 (production-scale models that do *not* saturate these batteries) matters — it is where the saturated batteries regain power.

- **Per-cell power vs pooled power.** Individual cells are often underpowered (TR149 per-cell MDE 14–28pp; TR152 per-cell Holm-clear MDE ~3.25–3.75pp), so no per-cell McNemar survives Holm in either report (0/12 and 0/120). The verdicts run off the *pooled* estimators (Mantel–Haenszel OR + TOST equivalence), which are adequately powered (TR152 pooled MDE OR ≈ 1.42, observed 1.88). Reading a per-cell non-significance as "no effect" would be the power-starvation fallacy; the design avoids it by anchoring verdicts on the pooled equivalence test.

- **The matched-pairs estimator gotcha.** TR149's estimator postmortem (feeding paired cells into the unpaired OR formula, which exploded to 3411.5 on an all-null corpus) is a standing caution for the whole line: the safety-line estimator is matched-pairs McNemar / Cohen's h / discordant-ratio OR, *not* the unpaired forms. A future reader "fixing" TR145's Mantel–Haenszel code to match TR149's would *reintroduce* a bug, because TR145 builds genuine unpaired marginal tables and TR149 builds paired cells — they correctly use different estimators.

- **Judge-cohort completeness.** The entire arc runs under the `--skip-openai-judge` umbrella gate, so the GPT-4o triangulation slot is empty (adversarial corpora must not flow through external APIs without Researcher Access Program enrollment). The two local LLM judges agree at κ ≥ 0.83 on the standardized and serving-state corpora (clearing JTP robust), so no verdict hinges on the missing slot — but the third-judge triangulation is a named Fellowship-gated extension, not a closed question.

- **Mechanistic probes are a dead end, not a noisy signal.** TR146's negative result is itself partly low-power (only ~2 neutral rows in the regime contrast), so "no separation" is bounded as "no separation detectable at this budget" rather than "proven impossible." The arc treats F3 as forbidden (mechanistic probes do not predict safety) on the strength of *four independent* falsifications plus the GPTQ confidence paradox, not on a single underpowered test.

> The honest summary is that Phase 6 is a **bounded, well-powered-in-aggregate null line with one located positive footprint**, validated under independently-measured judge agreement, on a 1B–4B envelope with named extensions. It is not a universal "FP8 is safe" claim, and every report says so in its own scope-discipline section. The strength of the arc is precisely that it never over-claims past its envelope.

### 15.3 Construct-validity threats — what "safe" actually measures

The deepest threat to a safety null is not that the measurement is noisy but that it measures the *wrong construct*: a result can be a clean, well-powered null on a quantity that is not the quantity a reviewer cares about. Phase 6 has three construct-validity exposures, each contained by an explicit design choice.

| Construct threat | Where it bites | Containment |
|---|---|---|
| **"Refusal" ≠ "harmlessness"** | the general judges (gemma3, llama3.1) score *response refusal*, not whether the response is actually harmful | TR148's dual-axis finding makes the construct split explicit: Layer 1a is the refusal axis, Layer 1b (shieldgemma + llama-guard3) is the orthogonal composite-harm screen, run in parallel rather than averaged |
| **"Safe rate" sign convention** | on XSTest, a *higher* refusal rate is *worse* (over-refusal), the opposite of the harmful batteries | TR149/TR152 invert the XSTest safe-slice outcome so "safe" always means "did the right thing," and report the two XSTest slices separately rather than averaging (§7.1, SS6.2) |
| **Regex refusal ≠ semantic refusal** | the regex anchor matches surface refusal templates, not semantic compliance; its κ with the LLM judges is corpus-balance-dependent | the regex anchor is a *cheap stable detector*, never the semantic verdict; the headline statistics run off the LLM-judge majority, and the regex–LLM κ is reported as a calibration cross-check, not a trust signal (§12.5) |

**Observations.** The refusal-vs-harmlessness split is the most consequential construct threat in the entire arc, and it is the one TR148 was built to surface rather than to hide. A naive single-axis pipeline that folded the composite-harm specialists into the refusal κ matrix would have read their −0.13 to −0.26 cross-axis agreement as instrument failure and discarded two *correct* judges — the precise category error the dual-axis finding names. The sign-convention threat is subtler and battery-specific: a degradation story told only on refusal-as-safe batteries (HarmBench/JBB/StrongREJECT) could read a numerical change that made a model refuse *more* as "safer," when on the benign edge that same change is the over-refusal failure mode. XSTest's inclusion is the load-bearing construct guard against exactly that misreading, which is why the single located positive in the arc surfaces *there* and nowhere else.

> The construct discipline is what makes the harmful-core null and the benign-edge located positive two different findings rather than a contradiction. "FP8 does not change refusal of harmful prompts" (HarmBench/JBB/StrongREJECT, 0/8,976 discordant) and "FP8 slightly increases over-refusal of benign-but-alarming prompts on Qwen" (XSTest, OR 1.88) are statements about two orthogonal constructs measured on two disjoint prompt slices — and the arc keeps them apart by design, never averaging a refusal-as-safe battery against an over-refusal battery into a single misleading "safe rate."

### 15.4 External-validity threats — generalization beyond the tested envelope

External validity is the threat that the result is true *here* but does not transfer to where it will be deployed. The arc's external-validity exposures are the same boundaries enumerated in §15.1, but the threat framing adds *why each boundary is plausibly load-bearing* rather than merely *that it exists*.

| Generalization gap | Why it might matter (the mechanism that could break the null) | What the arc does about it |
|---|---|---|
| **Scale (1B–4B → 7B–72B)** | larger models have larger residual streams and may quantize the KV-cache with a different error profile; the standardized batteries also *saturate* at 1B–4B and regain discriminating power only at scale | bounded claim + TR151 scaffolded; the saturation argument (§15.2) makes scale the *single most important* unmeasured axis, not a routine caveat |
| **Stack version (vLLM v0.19.1 → other)** | TR147 proves a *latency* conclusion can flip on a transitive-dependency bump; a safety conclusion is more robust (it runs off behaviour, not timing) but the FP8 *kernel* itself is stack-versioned | the pin is what makes TR149 a clean replication; CRI (Layer 3) is the named gate for cross-stack timing, and the FP8 safety claim is explicitly bounded to the tested stack |
| **Hardware SKU (Ada laptop → datacenter)** | TR147's A100 finding shows a datacenter GPU can *amplify* rather than rescue compiler fragility; batch-invariance failure is hardware-kernel-specific | the harmful-core null is a behavioural invariant unlikely to be SKU-specific, but cross-SKU validity is named as a TR151 prerequisite, not assumed |
| **Context length (≤2,048 → 8k–128k)** | long context accumulates more quantized KV state, and the Phase 5 long-context attack-surface work flagged many-shot exploitation as a real mover | explicitly excluded; TR150 scaffolded for 8k–128k; the null is never extrapolated into the long-context regime |
| **Temperature (0.0 → production sampling)** | TR144's speculative-decoding null is *greedy-only* by theorem; above temperature 0 the typical-acceptance path is actually exercised | TR152 *does* extend FP8 to temp 0.7/1.0 (and finds the Qwen lean lives there); TR144's E6 temp sweep is the named unrun extension for the speculative member |

**Observations.** The external-validity threats are not symmetric in severity, and the arc is careful to say so. The scale gap is the most serious because it is *double*: the model behaviour itself may differ at 7B–72B, and — independently — the standardized batteries lose their discriminating power at 1B–4B, so the cleanest test of the null is precisely the one that has not been run. The stack and SKU gaps are partially *self-mitigating*: the safety verdict runs off behaviour rather than timing, and a harmful-refusal invariant is far less likely to be hardware-specific than a latency speedup, so TR147's catastrophic latency fragility does *not* imply a fragile safety verdict — but the arc declines to lean on that intuition and names cross-SKU validity as a prerequisite anyway. The temperature gap is the one the arc has *already partially closed*: TR152 carried FP8 across temp 0.7/1.0 and that is exactly where the Qwen lean localized, so for the FP8 line temperature is a *measured* axis, not an open one; it remains open only for the speculative-decoding member (TR144).

> External validity is where a safety arc most often over-reaches, and the discipline that prevents it here is the same one that runs through the claim ladder: every headline names its envelope in the same sentence. The arc's contribution is not "FP8 KV-cache is safe" — a statement whose external validity is unbounded and therefore indefensible — but "FP8 KV-cache is harmful-refusal-neutral *for 1B–4B instruction-tuned models on vLLM v0.19.1 at the tested serving axes*," whose external validity is exactly as wide as the evidence and no wider. The scaffolds (TR150/TR151/TR153) are the designed widenings, each one naming a specific generalization gap above.

### 15.5 Internal-validity threats — confounds within the paired design

Internal validity is the threat that something *other than* the manipulated variable produced the observed (non-)effect. A null is especially exposed here, because a confound that *masks* a real effect would manufacture a false null.

- **Seed-locked determinism could mask a stochastic effect.** The FP8 line runs at temperature 0, seed 42, deterministic decoding — which is what makes the pairing exact, but also means a safety effect that only manifests under sampling stochasticity would be invisible at the base case. *Containment:* TR152 breaks determinism by carrying the contrast to temperature 0.7/1.0, and the located Qwen lean appears precisely there — so the design does *not* rest on determinism masking the effect; the one place an effect surfaces is the one place determinism is relaxed.
- **Judge contamination across conditions.** If the same judge scored FP16 and FP8 outputs with any condition-leakage (e.g. seeing the dtype label), the agreement could be inflated. *Containment:* judges score completions blind to the dtype; the matched-pairs structure scores the *same prompt* under both conditions, so any judge bias is differenced out of the discordant-pair count.
- **Prompt-set leakage between paired conditions.** The matched-pairs estimator is only valid if every deviating-axis spoke shares the baseline's exact prompt set. *Containment:* TR152's star design enforces this with machine-checkable `deviates_axis` and `paired_with` fields (§10.1); a confounded fractional design that broke the shared-prompt invariant is exactly why the star, not a resolution-IV/V fractional factorial, is the correct choice.
- **Estimator-choice confound.** The pooled OR is sensitive to whether the table is treated as paired or unpaired — the wrong choice manufactures a spurious effect (TR149's 3411.5 explosion) or masks a real one. *Containment:* §12.8's estimator-by-table-type discipline, with the matched-pairs discordant-ratio form for paired cells and the unpaired cross-product only for genuine marginal tables.
- **Baseline reproducibility drift.** A null measured against a baseline that has silently drifted from prior reports would be uninterpretable. *Containment:* TR145's 36/36 cross-TR baseline tuples within ±5pp against TR138/TR143/TR144 (§8.2, SS22), and TR148's regex × gemma3 calibration anchor drifting only Δκ = −0.0648 from TR145's reported value (§6.2, SS12), both confirm the pipeline reads the same data the prior reports produced.

**Observations.** The internal-validity threats are the ones a null is *most* exposed to, because every one of them, if uncontained, would manufacture a false null — and a false null is exactly what a safety reviewer fears. The strongest single containment is the temperature one: the most natural objection to a temperature-0 null ("you froze out the stochasticity where the effect lives") is answered not by argument but by TR152 actually carrying the contrast into the stochastic regime and finding the one located effect *there*, which simultaneously closes the internal-validity gap and produces the arc's only positive finding. The baseline-reproducibility checks are the quiet workhorses: 36/36 tuples within ±5pp and a calibration-anchor drift inside ±0.10 are what license treating the seven reports as measurements of *one* underlying system rather than seven independent experiments that happen to agree.

> Internal validity is the axis on which the matched-pairs design earns its cost. A between-subjects safety comparison (different prompts under FP16 vs FP8) would confound the dtype effect with prompt-difficulty variance and could not difference out judge bias; the matched-pairs structure — same prompt, same seed, same judge, differenced — is what converts "the two conditions had similar safe rates" into "the *same prompts* changed outcome in only N discordant pairs, split near-evenly." That is a categorically stronger internal-validity claim, and it is the reason the harmful-core result can be stated as *zero discordant pairs* rather than as *indistinguishable marginal rates*.

### 15.6 Threats specific to the single located positive

The arc's one non-null finding — the Qwen × XSTest over-refusal lean (F13) — deserves its own threat analysis, because a located positive is exposed to the *mirror-image* risks of a null: it could be an artefact (a false positive), or it could be real but mis-attributed (right effect, wrong cause).

| Threat to F13 | The worry | What rules it out |
|---|---|---|
| **It is a multiple-comparisons artefact** | 120 cells × several tests is a lot of chances for one to look significant | the pooled sign test (87 vs 46, p ≈ 0.0004) is a *single* aggregate test, not a cherry-picked cell; and 0/120 cells survive Holm — the finding lives at the pooled level, not in a lucky cell |
| **It is a judge artefact** | the lean could be one judge's idiosyncrasy on benign-edge prompts | cross-judge κ = 0.831 on TR152's corpus clears JTP robust; the discordance is agreed across the cohort, not one judge's |
| **It is a power illusion (really null)** | maybe the OR > 1.0 is noise around a true null | the pooled estimator was powered to detect OR ≈ 1.42 and observed 1.88; the v1→v2 lift *tightened* the CI around a value above 1.0 and moved the lower bound further from 1.0 — the artefact signature (CI collapsing through 1.0) did not occur (§10.5) |
| **It is real but not Qwen-specific** | maybe FP8 leans over-refusal on all families and Qwen just shows it most | the per-family MH decomposition has Llama (0.484) and Phi-3 (0.130) running the *opposite* direction, Phi-3's CI excluding 1.0 on the improvement side — the effect is directional-by-family, not a uniform lean Qwen amplifies (§10.4) |
| **It is real but not FP8-caused** | maybe Qwen's baseline over-refusal drives it regardless of dtype | the finding is a *matched-pairs discordance* — the same prompt flipping FP16→FP8 — so Qwen's high baseline over-refusal (37.2% both-unsafe) is differenced out; what remains is the FP8-attributable *delta* on top of that baseline |

**Observations.** The located positive is held to a *higher* evidentiary bar than the nulls around it, which is the correct asymmetry: a null that is wrong understates risk, a positive that is wrong manufactures a phantom vulnerability, and both failure modes are worth closing. The two sharpest defenses are the per-family decomposition and the matched-pairs structure. The decomposition rules out "FP8 leans over-refusal generically" by showing two of three families run the *opposite* direction — a generic FP8 lean cannot produce Phi-3's CI excluding 1.0 on the *improvement* side. The matched-pairs structure rules out "it's just Qwen's baseline over-refusal" by construction: the estimator counts only prompts that *flipped* between FP16 and FP8, so a high but dtype-invariant baseline contributes zero discordant pairs and cannot produce the signal.

> The located positive is the arc's proof that the instrument can see an effect when one is present — the same telescope that finds nothing on the harmful core finds a real, sign-test-significant, judge-agreed, per-family-localized lean on the benign edge. That is what makes the harmful-core null *credible* rather than merely *consistent with low power*: a measurement apparatus that returns "null everywhere" is suspect, while one that returns "null on the harmful core, one located positive on the benign edge of one family, no serving-state interaction" is reporting structure it actually resolved.

---

## 16. Production Guidance and Operational Doctrine

This section states the operational guidance Phase 6 licenses, in the form the bridge paper's Layer 5 inherits. Each statement is tagged with its claim-ladder tier.

### 16.1 The serving-state-independent FP8 certificate (Supported, bounded)

> **For instruction-tuned models in the 1B–4B parameter range, served on vLLM v0.19.1, an FP8 (E4M3) KV-cache is harmful-refusal-neutral across the tested serving state — batch size ∈ {1, 8, 32}, prefix caching {on, off}, and sampling temperature ∈ {0.0, 0.7, 1.0} — and a deployment may enable it without per-serving-configuration safety re-validation, subject to the per-family caveat below.**

This is the Layer 5 certificate. It rests on TR152's 8,976 zero-discordance harmful-battery pairs and flat interaction spread (2.99pp inside the ±3pp band), standing on TR145's single-config base case and TR149's standardized-corpus replication, under judge labels whose agreement clears the JTP robust threshold (κ = 0.831). It is bounded to the 1B–4B envelope and the single tested stack; it is *not* a claim about 7B–72B models (TR151), long context (TR150), or other KV-quantization methods (TR153).

### 16.2 The per-family over-refusal caveat (Licensed)

> **The Qwen 2.5 family carries a sub-1-percentage-point increase in over-refusal of benign-but-alarming prompts under FP8, concentrated on the non-zero temperature axes and amplified mildly on the 1.5B variant. A deployment running Qwen 2.5 at temperature > 0 on a benign-edge-heavy workload should monitor the over-refusal rate; the effect is absent on Llama-3.2 and Phi-3 (where FP8 is neutral-to-improving on the same battery).**

This caveat is the H1-located finding (per-family MH OR: Qwen 3.878, Llama 0.484, Phi-3 0.130). It is a quality-of-service note about declining safe requests, *not* a safety-degradation warning — the harmful core is invariant on every family.

### 16.3 What the guidance explicitly does NOT say (Forbidden)

- It does **not** say "FP8 is safe" unconditionally — the claim is bounded to the tested envelope (TR145 scope discipline; CLAIM_LADDER F-tier).
- It does **not** license reading safety off the weights — mechanistic probes (entropy, refusal-direction, calibration, safety-neuron error) do *not* predict safety degradation (TR146; F3). Any logit- or confidence-based screen is specifically contraindicated by the GPTQ confidence paradox, where the most dangerous configurations look the *most* certain.
- It does **not** license speculative decoding as safety-neutral at temperature > 0 — TR144's null is greedy-only; the typical-acceptance criterion genuinely alters the output distribution above temperature 0, and the leakage hypothesis is untested there.
- It does **not** extend to cross-stack latency comparisons without a CRI gate — the Triton kill-shot (62.82% → 0.84% prefill collapse) is the worked example of why an ungated cross-stack number is untrustable.

### 16.4 The deployment-time protocol shape

The five layers compose into a deployment-time checklist the bridge paper formalizes:

1. **Layer 1 (validity):** run the JTP κ check on your judge cohort against the calibration corpus; if κ < 0.70, triangulate (≥3 judges) or abort the labeled verdict.
2. **Layer 3 (compile integrity):** compute CRI across your candidate serving stacks; if the band is `fragile`/`catastrophic` (max pairwise |d| ≥ 2), the latency comparison is not reportable.
3. **Layer 2 (behavioural screens):** run RTSI on the refusal-template features and TAIS if speculative decoding is enabled at greedy decoding.
4. **Layer 4 (scale validity):** confirm your model scale is inside the validated 1B–4B envelope, or treat the certificate as scaffolded-only above 4B pending TR151.
5. **Layer 5 (serving-state validity):** the FP8 certificate above applies; carry the Qwen-family over-refusal footnote if applicable.

> The protocol's value is that it is *behavioural, calibrated, and bounded*. Every gate has a measured threshold (not a heuristic), every verdict is bounded to a stated envelope, and the one shortcut everyone reaches for — mechanistic probing — is explicitly closed. That is the artefact Phase 6 hands to the bridge paper.

### 16.5 The cost model of running the protocol

A certification protocol is only adopted if a team can afford to run it. Phase 6's design choices are, in large part, cost-driven, and stating the cost of each gate is what makes the protocol a realistic deployment artefact rather than an academic ideal.

| Gate | What it costs to run once | What it would cost the naive way | The saving |
|---|---|---|---|
| **Layer 1 (JTP κ check)** | one re-judge pass of a calibration corpus across ≥2 local judges (gemma3:12b + llama3.1:8b on Ollama, $0 external) | a fresh human-annotation round, or a paid frontier-judge corpus | the κ floor is measured once on the calibration corpus and *inherited* by every downstream report — not re-run per study |
| **Layer 3 (CRI)** | a stack-perturbation micro-benchmark (a few hundred latency rows per stack point) | shipping a latency number and discovering on a neighbour stack that it was 0% | one micro-benchmark vs a production-visible reproducibility failure |
| **Layer 2 (RTSI/TAIS)** | RTSI is forward-pass-cheap (refusal-template features); TAIS is a paired byte/refusal comparison | a full behavioural battery per speculative configuration | a screen, not a full eval — RTSI flags which configs *need* the full battery |
| **Layer 4 (scale validity)** | one standardized-battery pass per model scale (TR149 = 7,578 records, $0 external) | re-deriving safety per model from scratch with bespoke corpora | a fixed battery reused across scales for cross-paper comparability |
| **Layer 5 (serving-state factorial)** | the star factorial — 7 spokes × 2 dtypes per (model, battery), $0 external (local judges) | the full 72-cell crossing, or per-config re-validation forever | one factorial certificate consumed across the Cartesian product of tested knobs |

**Observations.** The single largest cost lever in the entire protocol is the umbrella gate's local-judge cohort: every executed safety report ran at **$0 external API cost** because adversarial corpora are scored by local Ollama judges (regex + gemma3:12b + llama3.1:8b) rather than a paid frontier API, with the only metered spend being RunPod GPU time for cells that exceed the local 12GB envelope (§4.4). The second lever is the *inherit-don't-re-run* discipline: the JTP κ floor is measured once on the calibration corpus and consumed by every downstream report rather than re-derived, and the Layer 5 certificate is consumed across the tested serving Cartesian product rather than re-earned per configuration. The third is the star-vs-full-factorial choice: the star answers "does any single serving axis bend the FP8 contrast?" at 12 cells per model where the full crossing would need 72 — a 6× saving that the flat-interaction result retroactively justifies (if any axis *had* bent the contrast, the joint interactions the star cannot see would have mattered; none did).

> The cost model is why a *measurement-validity* phase is economically rational rather than a luxury. The naive alternatives in the right-hand column are not hypothetical — they are what a team does *without* this protocol: re-validate per config (combinatorial bill), ship un-reproducible latency numbers (the Triton failure), or trust a single un-triangulated judge (the verdict-flipping bug). Each gate replaces a recurring or catastrophic cost with a one-time bounded one, and the whole protocol runs at $0 external cost — which is the precondition for the program's volume-and-velocity strategy.

### 16.6 The deployment decision tree

The five-layer checklist (§16.4) is a linear walkthrough; a deploying team actually reasons as a *decision tree* with early exits. Stating it as a tree makes the gates' dependency order operational.

1. **Is your model scale inside 1B–4B instruction-tuned?**
   - *No (7B–72B):* the FP8 certificate is **scaffolded-only** above 4B — treat FP8 KV-cache as unvalidated pending TR151; do not inherit the Layer 5 certificate. Stop here for the FP8 claim.
   - *Yes:* proceed.
2. **Is your serving stack vLLM v0.19.1 (or a CRI-validated neighbour)?**
   - *Unsure / different version:* run the **Layer 3 CRI gate** on your candidate stacks before trusting *any* cross-stack latency number; the safety verdict is more robust but the FP8 kernel is stack-versioned, so re-confirm the behavioural null on your stack if it differs materially.
   - *Yes:* proceed.
3. **Can you trust your judge cohort?**
   - Run the **Layer 1 JTP κ check** against the calibration corpus. If κ < 0.70, **triangulate** (≥3 judges, majority vote) or do not license the labeled verdict. Never self-select dispatch on the corpus where your judges happen to agree best (§7.4).
4. **Are you enabling FP8 KV-cache?**
   - *Yes:* the **Layer 5 certificate** applies across batch ∈ {1, 8, 32}, prefix {on, off}, temperature ∈ {0.0, 0.7, 1.0} — *without* per-config re-validation.
   - **Is your model the Qwen 2.5 family, served at temperature > 0, on a benign-edge-heavy workload?** If yes, **monitor the over-refusal rate** (the F13 caveat); if no, no per-family action needed.
5. **Are you enabling speculative decoding?**
   - *At greedy decoding (temp 0):* TAIS licenses it as safety-neutral.
   - *At temperature > 0:* **untested** — TR144's null is greedy-only; re-test before relying on it.
6. **Are you tempted to read safety off a cheap forward-pass probe instead of behaviour?**
   - **No.** TR146 forbids it (F3); the GPTQ confidence paradox means a logit/confidence screen would approve the *most* dangerous configs.

**Observations.** The tree's early-exit structure is what keeps it cheap: a team outside the 1B–4B envelope exits at step 1 without running any gate (it simply learns its FP8 claim is scaffolded-only), and a team inside the envelope on the validated stack reaches the Layer 5 certificate in three gate-checks. The two branch points that most teams get wrong are step 3 (trusting a single judge without the κ check — the verdict-flipping failure) and step 6 (reaching for a mechanistic shortcut — the falsified probe). Both are placed as explicit branches rather than buried in prose precisely because they are the high-frequency errors the arc's negative results were built to catch.

> The decision tree is the protocol's *adoption surface*. A team does not adopt a five-layer checklist by reading a synthesis; it adopts a tree it can walk in an afternoon, exiting early where its configuration is out of scope and reaching a bounded certificate where it is in scope. The tree encodes the same dependency order as the claim ladder — validity gates (1, 3) before safety verdicts (5), with the forbidden shortcut (6) closed at the end — so a team that walks it inherits the arc's discipline without having to reconstruct it.

### 16.7 What to monitor in production

A certificate is a point-in-time validation; production is continuous. The arc's findings translate into a small set of monitorable signals, each tied to the specific finding that motivates it.

| Signal to monitor | Why (the finding behind it) | Trigger to act |
|---|---|---|
| **Over-refusal rate on benign-edge traffic** (Qwen 2.5, temp > 0) | F13: the one located FP8 footprint is a sub-1pp Qwen over-refusal lean on XSTest-like prompts, concentrated on the temperature axes | a rising decline rate on known-benign requests, especially after a temperature or batch change on Qwen |
| **Serving-stack version drift** (Triton, PyTorch, vLLM, code SHA) | F1/F2: a latency conclusion can flip catastrophically on a transitive-dependency bump; the FP8 kernel is itself stack-versioned | any change to the six-axis benchmark identity (GPU, Triton, PyTorch, cache impl, compile mode, code SHA) — re-run CRI |
| **Judge-cohort agreement** (periodic JTP κ re-check) | F3/F11: the κ floor is corpus-specific (0.69 mixed, 0.83 clean); a drift in traffic distribution can move it | κ on a rolling calibration sample dropping below 0.70 → revert to triangulation |
| **Harmful-refusal rate on the standardized batteries** | F12: the harmful core is the unconditional guarantee; any movement is the signal that matters most | *any* non-zero discordance on harmful-battery prompts under a config inside the certified envelope — this should be flat at 100% refusal |

**Observations.** The monitoring set is deliberately asymmetric in priority. The harmful-refusal rate is the *only* signal whose movement would invalidate the core certificate — F12 says it is exactly 1.0000 across 8,976 pairs in the tested envelope, so any non-zero discordance in production on a config inside that envelope is a first-order alarm, not a quality-of-service note. The over-refusal rate is a *second-order* signal: a rising Qwen over-refusal rate degrades usefulness (declined benign requests) but says nothing adverse about harmful-prompt safety, so it is a quality-of-service dashboard, not a safety alarm. The stack-version and judge-κ signals are *meta-monitors*: they watch the instruments rather than the model, and they are the production analogues of Layers 3 and 1 — the same gates, run continuously instead of once.

> The monitoring doctrine inverts the usual safety-dashboard instinct. The signal a team is tempted to watch — model confidence, logit margins, a cheap internal probe — is the one TR146 forbids, because the GPTQ confidence paradox means rising confidence can *co-occur* with failing refusals. The signals the arc licenses are all *behavioural and external*: the refusal rate on a known corpus, the judge-cohort agreement, the stack identity. You monitor what you can measure behaviourally, never what you can only read off the weights.

### 16.8 Graceful degradation — what to do when a layer cannot be run

Not every deployment can run every gate. A protocol that demanded all five layers or nothing would not be adopted; the arc's findings imply a *graceful-degradation* order — which layers are load-bearing and which can be substituted by a conservative default.

- **If you cannot run Layer 1 (JTP):** default to **triangulation** (≥3 judges, majority vote) unconditionally. The κ check exists to *license single-judge dispatch*; without it, you pay the 3–4× triangulation cost but never risk the single-judge verdict-flip. This is the safe degradation.
- **If you cannot run Layer 3 (CRI):** do not report *any* cross-stack latency comparison; pin and publish your full stack identity (six axes) and report only within-stack numbers. The degradation is a *scope restriction*, not a risk — you simply stop making the claim CRI would have gated.
- **If you cannot run Layer 2 (RTSI/TAIS):** fall back to the full behavioural battery for every configuration rather than the cheap screen. More expensive, not less safe — the screen's job was to tell you *which* configs need the full battery, so without it you run them all.
- **If you cannot run Layer 4/5 at your scale or serving state:** treat the FP8 certificate as **scaffolded-only** and run the standardized battery on your own configuration before enabling FP8. The certificate is an inheritance, not an entitlement; outside the validated envelope you re-earn it.

**Observations.** Every degradation path is *toward more cost and never toward more risk* — which is the property a safety protocol must have to be honestly adoptable. The JTP degradation (default to triangulation) is the cleanest example: the gate's entire purpose is to *cheapen* the common case (license single-judge dispatch when κ clears robust), so its absence costs throughput, not safety. The same shape holds for every layer: the gates are *cost-reducers gated on a validity check*, so failing to run a gate forfeits the cost reduction but never the safety floor. This is the difference between a protocol whose layers are *safety requirements* (all-or-nothing) and one whose layers are *validated cost optimizations* (degrade gracefully) — Phase 6's are the latter.

> The graceful-degradation order is what makes the protocol a *spectrum* rather than a gate. A resource-constrained team triangulates every judge, runs the full battery on every config, and reports only within-stack numbers — slower, but exactly as safe, and still inside the claim ladder. A fully-instrumented team runs all five gates and inherits the single Layer 5 certificate. Both are following the same protocol; they differ only in how much of the cost optimization they can afford, never in the safety floor they stand on.

### 16.9 Anti-patterns the arc specifically warns against

Finally, the arc's negative results and near-misses crystallize into a short list of anti-patterns — the specific wrong moves the program *made and corrected*, or *tested and falsified*, so that a downstream consumer does not repeat them.

| Anti-pattern | Why it is tempting | The finding that forbids it |
|---|---|---|
| **"One strong judge is enough."** | a single frontier judge is cheap and feels authoritative | F3/F5: the mandatory-judge join restricted TR148 v1 to 94 records and produced the *wrong* verdict; triangulate unless κ ≥ 0.70 on *your* corpus |
| **"Low κ means a broken judge — drop it."** | a negative agreement looks like instrument failure | F4: the specialists' −0.13 to −0.26 cross-axis κ is a *different construct*, not a broken judge; triangulate within axis, screen composite-harm in parallel |
| **"A benchmark is a property of the model."** | latency numbers are reported without their stack | F1/F2: the Triton-only ablation flips prefill 62.82% → 0.84% on one physical GPU; pin the six-axis stack identity or the number is not reportable |
| **"Re-validate safety per serving config."** | the cautious-looking move; or its twin, "skip it" | F12/F14: TR152's flat interaction converts per-config re-validation into one serving-state-independent certificate; the combinatorial bill was unnecessary |
| **"Read safety off the weights / logits / a probe."** | interpretability promises a cheap shortcut | F6/F7: four probes all falsify; the GPTQ confidence paradox means a logit screen *approves* the most dangerous configs |
| **"Fix TR145's OR code to match TR149's."** | the two reports use different OR formulas — looks like an inconsistency | §7.5/§15.2: TR145 builds *unpaired* marginal tables, TR149 *paired* cells; they correctly use different estimators — "fixing" one reintroduces the 3411.5 bug |
| **"Pattern-match the failure cause."** | a failed cell *looks* like a VRAM ceiling | §10.5: TR152 v1 blamed OOM; the log showed a vLLM argparse rejection (a one-line flag fix). Read the artifact before attributing. |

**Observations.** Five of the seven anti-patterns are not hypothetical — they are mistakes the program *actually made inside the constituent reports and caught* (the mandatory-judge join, the unpaired-OR explosion, the OOM mis-attribution) or shortcuts it *tested and falsified* (the mechanistic probes, the single-judge sufficiency). That is the strongest possible provenance for an anti-pattern list: each entry is a wire the program tripped at least once, and the corresponding finding is the wire holding. The last two are the subtlest and the most program-specific — the estimator-consistency trap (where the "obvious fix" *introduces* a bug) and the failure-attribution trap (where the plausible cause is the wrong one) — and both are direct expressions of the project's anti-confabulation discipline: read the artifact, match the estimator to the table type, never let a plausible pattern-match stand in for an inspected cause.

> The anti-pattern list is the arc's hard-won operational wisdom in its most transferable form. A downstream team cannot re-derive the dual-axis finding or the estimator postmortem from first principles, but it can read "low κ may mean a different construct, not a broken judge" and "match the OR formula to the table type" and avoid the two errors that cost the program a re-analysis cycle each. The list is short by design — seven entries, each tied to a numbered finding — because an anti-pattern catalogue is only useful if it is memorable, and the most memorable form is "here is the specific wrong move, here is the finding that forbids it."

---

## 17. Conclusion

Phase 6 consolidates six executed technical reports into a single five-layer serving-state safety certification protocol, validated on the 1B–4B instruction-tuned model range. The arc has two pillars that meet at Layer 5.

The **measurement-validity substrate** (Chapter 1) supplies two calibrated indices and one hard negative. CRI (TR147) converts cross-stack latency reproducibility into a routing band, sharpened by the Triton kill-shot where a compiler-version change collapsed prefill throughput by a factor of ~75. JTP (TR148) converts cross-family judge agreement into a calibrated κ floor (0.6917), split into a refusal-axis gate (1a) and an orthogonal composite-harm screen (1b) after the dual-axis finding showed specialist safety judges anti-correlate with generalist judges because they measure a different construct. TR146 is the negative control underneath both: four mechanistic probes all fail to distinguish safe from dangerous quantized configurations, so the entire arc stays behavioural — you cannot read safety off the weights, and the GPTQ confidence paradox shows that any logit-based screen would actively approve the most dangerous configurations.

The **inference-flag null line** (Chapter 3) accumulates a calibrated null on the harmful core across four members. TR144 (TAIS) shows speculative decoding is behaviourally inert at greedy decoding, max |Cohen's h| = 0.024 across 64,855 paired samples, with byte-identity surviving adversarial drafts, quantized drafts, and seed changes. TR145 supplies the single-config FP8 base case (pooled OR 1.05), conservatively bounded to its tested envelope. TR149 (Chapter 2) replicates the FP8 null on the four literature-canonical batteries with positive ±3pp equivalence on all 12 cells, and discovers the JTP verdict is corpus-specific (κ tightens from 0.69 to 0.83 on standardized adversarial batteries). TR152 (Chapter 5) folds the serving-state axes into one factorial and returns the capstone result: zero discordance across 8,976 harmful-battery pairs, a flat interaction (spread 2.99pp inside the ±3pp band), and a single located footprint — a sub-1pp Qwen-2.5 over-refusal lean on the benign-edge battery, resolved to H1-not-H2 by the per-family Mantel–Haenszel decomposition.

Read together, the four null-line members are a **four-step resolution ladder on one underlying truth**: harmful refusal is invariant to FP8 across the serving state, and the over-refusal boundary is faintly, locally perturbed on one model family. That truth, measured under judge labels whose agreement independently clears the JTP robust threshold, is the **serving-state-independent FP8 safety certificate** the bridge paper inherits as Layer 5.

The protocol's boundaries are explicit (Chapter 6): 1B–4B scale, single local SKU, vLLM v0.19.1, short context, with three named scaffolds (TR150 long-context, TR151 7B–72B scale, TR153 KV-method sweep) designed to widen the envelope rather than rebuild the protocol, gated on the 2026-10-24 GO/NO-GO trigger. Phase 6's contribution is the protocol itself, instantiated and validated on a bounded envelope, with every claim tagged to its tier on the claim ladder and every gate carrying a measured threshold. It does not claim FP8 is universally safe; it certifies a precise, bounded, behaviourally-measured operating envelope and names exactly where that envelope stops.

### What Phase 6 establishes

Stated as the strongest verbs the evidence licenses, and no stronger:

1. **A composed, five-layer serving-state safety certification protocol exists and is instantiated.** It is not a proposal; four of its five layers are anchored by executed reports (Layer 1 JTP, Layer 3 CRI, Layer 4 1B–3B, Layer 5 serving-state), the fifth (Layer 2) has its TAIS member executed alongside the shipped RTSI member, and each layer carries a *measured* threshold rather than an asserted one.
2. **The harmful-refusal core is invariant to the tested serving state.** Across FP8 KV-cache, speculative decoding, batch size, context length, prefix caching, and sampling temperature, no knob moves harmful-refusal beyond ±3pp; the capstone factorial finds zero discordance across 8,976 harmful-battery pairs.
3. **Judge agreement is measurable, corpus-specific, and two-axis.** The JTP floor (κ = 0.6917, tightening to 0.83 on standardized batteries) is a property of the (cohort × corpus) pair, and the refusal axis is orthogonal to the composite-harm axis — both established, not assumed.
4. **Safety cannot be read off the weights.** Four mechanistic probes falsify the cheap-shortcut hypothesis; the verdict must be behavioural, and a logit-confidence screen would actively approve the most dangerous configurations.
5. **The whole substrate is reproducible at $0 external cost.** Every executed safety verdict ran on a local judge cohort under the umbrella gate, which is itself a methodological result: the validity layer does not require a cloud safety-judge budget.

### What Phase 6 explicitly does NOT establish

The same evidence, read for its boundaries:

1. **It does not establish that "FP8 is safe."** The supported claim is "no detectable harmful-core effect under tested conditions, with positive ±3pp equivalence on the cells that have the variance to show one" — a bounded envelope, not a universal property.
2. **It does not establish the null beyond ~4B.** The FP8 safety line is anchored on 1B–3B (plus the 70B speculation cell); the production-relevant 7B–72B range is TR151's explicit, cloud-gated scope.
3. **It does not establish the null under stochastic decoding.** Every result is at temperature 0; the typical-acceptance criterion genuinely alters the output distribution above it, and the TAIS null must not be extrapolated there.
4. **It does not establish a four-judge κ floor.** The verdict stands on two LLM judges; the gpt-4o and Claude cross-family axes are credential-deferred.
5. **It does not establish joint two-axis serving interactions.** TR152 is a star factorial (one factor at a time around a center), so the "additive" claim covers single-axis modulation only.

**Observations.** The two lists are the same evidence read in two directions, and the discipline of stating them side by side is the whole epistemic posture of the phase. Every "establishes" item has a matching "does not" that bounds it: the protocol exists *but* its Layer 4 is half-filled; the harmful core is invariant *but* within ≤4B and at greedy decoding; judge agreement is measured *but* on two axes by two judges. A reader who takes only the left column over-claims; a reader who takes only the right column under-claims; the synthesis is the document that forces them to be read together, which is the same job the claim ladder does at the level of an individual sentence.

> The closing statement of the arc is one sentence with every qualifier load-bearing: *for instruction-tuned models in the 1B–4B range, on a single local hardware SKU and vLLM v0.19.1, at greedy decoding, an FP8 KV-cache is harmful-refusal-neutral across the tested serving state, under judge labels whose agreement is independently measured to clear the JTP robust threshold, with a single located sub-1pp Qwen over-refusal lean on the benign edge.* Phase 5 found where the serving state could move safety; Phase 6 built the instruments to measure it and showed the most-deployed flag does not move the harmful core under them; Phase 5 is now mitigating the located vulnerabilities with the screens and judges Phase 6 certified. That is the arc, and the certificate this synthesis hands forward is the middle link that makes the other two defensible.

---

## 18. References

### 18.1 TR-internal source reports

- **Technical Report 144** — *Speculative Decoding × Safety: the Typical-Acceptance Invariance Screen (TAIS).* 64,855 paired samples; max |Cohen's h| = 0.024. (`PublishReady/reports/Technical_Report_144.md`)
- **Technical Report 145** — *FP8 KV-Cache Safety at a Single Configuration.* 24,054 records, five-phase paired battery; pooled OR 1.05 [0.90, 1.23]. (`PublishReady/reports/Technical_Report_145.md`)
- **Technical Report 146** — *Mechanistic Probing of Quantization Safety: a Four-Probe Falsification.* 5,100 forward passes; 1.40× safety-neuron quantization error, necessary-not-sufficient. (`PublishReady/reports/Technical_Report_146.md`)
- **Technical Report 147** — *The Compile Reproducibility Index (CRI) and the Triton Kill-Shot.* 52,410 primary rows; prefill collapse 62.82% → 0.84%, |d| ≈ 14–49. (`PublishReady/reports/Technical_Report_147.md`)
- **Technical Report 148** — *The Judge Triangulation Protocol (JTP) and the Dual-Axis Finding.* κ = 0.6917; refusal-axis vs composite-harm axis split. (`PublishReady/reports/Technical_Report_148.md`)
- **Technical Report 149** — *FP8 KV-Cache Safety on Standardized Batteries.* 7,578 records, 4 batteries; pooled OR 0.8065, 12/12 TOST-equivalent, corpus-specific κ = 0.83. (`PublishReady/reports/Technical_Report_149.md`)
- **Technical Report 152** — *FP8 KV-Cache Serving-State Safety Factorial.* 45,000 responses, 20,754 pairs, 135,000 judge labels; pooled OR 1.8817 [1.3185, 2.6855], serving-state-independent certificate with Qwen-XSTest footnote. (`PublishReady/reports/Technical_Report_152.md`)

### 18.2 Related Banterhearts reports (cited, not synthesized here)

- **Technical Report 138 / 141** — batch-inference safety and refusal fragility; the inference-flag-null-line antecedents (accepted, ICML 2026 Workshop on Hypothesis Testing).
- **Technical Report 142** — the Refusal Template Stability Index (RTSI), the Layer 2 behavioural screen alongside TAIS.
- **Technical Report 140** — the ML-CQ judge-triangulation lineage feeding JTP's κ-threshold calibration.

### 18.3 Protocol and claim-ladder anchors

- `papers/serving_state_safety_certification/CLAIM_LADDER.md` — the supported/licensed/forbidden claim ladder and the canonical five-layer mapping (S4). Supported anchors S2 (TR148 κ), S5 (TR149 equivalence), S6 (dual-axis). Forbidden F3 (mechanistic probes).
- `papers/serving_state_safety_certification/UPGRADE_PLAN.md` — the bridge-paper build plan and the 2026-10-24 GO/NO-GO trigger.

### 18.4 External method lineage

- Arditi et al. — the refusal-direction construct probed (and found behaviourally ineffective under quantization) in TR146.
- HarmBench, JailbreakBench, StrongREJECT, XSTest — the four literature-canonical safety batteries standardized in TR149 and reused in TR152.
- The matched-pairs statistical toolkit — McNemar's exact test, Cohen's h (paired-binary), TOST equivalence at ±3pp, Holm–Bonferroni stepdown, Haldane-corrected Mantel–Haenszel pooled OR, bootstrap confidence intervals.

---

*End of Conclusive_Phase6.md. Companion documents: `Conclusive_Phase6_Extended_Appendices.md` (per-TR appendix tables + cross-TR appendices) and `Conclusive_Phase6_Whitepaper.md` (condensed external-facing synthesis).*

