# Phase 6 Decision Whitepaper
## Executive guidance for serving-state safety certification of optimized LLM inference

| Field | Value |
|-------|-------|
| **Project** | Banterhearts LLM Inference Research |
| **Date** | 2026-05-28 |
| **Version** | 1.0 |
| **Report Type** | Decision whitepaper |
| **Audience** | Decision makers, ML ops leaders, safety teams, serving-infrastructure engineers |
| **Scope** | TR144 (Speculative Decoding × Safety / TAIS), TR145 (FP8 KV-cache single-config), TR146 (Mechanistic-probe falsification), TR147 (Compile Reproducibility Index / CRI), TR148 (Judge Triangulation Protocol / JTP), TR149 (FP8 standardized batteries), TR152 (FP8 serving-state factorial) — plus inherited TR142 (RTSI, Layer 2) and the forward link to TR163 (Phase 5 mitigation) |
| **Primary Source** | `PublishReady/reports/Conclusive_Phase6.md` |
| **Predecessor** | `PublishReady/reports/Technical_Report_Conclusive_Phase5_Whitepaper.md` (Phase 5 attack-surface decisions) |

---

## Abstract

Phase 5 (TR138–143) mapped the safety attack surface of optimized LLM inference: *where* batching, multi-turn pressure, long-context exploitation, and cross-architecture fragility can move refusal behaviour. This whitepaper distils the phase that comes next. Phase 6 answers the prior question both neighbours depend on — *can the instruments that drew those maps be trusted, and does the most-deployed serving flag (FP8 KV-cache) actually move safety once the instruments are validated?*

It consolidates seven technical reports into a single **five-layer serving-state safety certification protocol** and a worked example of a serving flag that passes it. Three reports build the measurement substrate: CRI (TR147) gates whether a compiled-latency number is reproducible, JTP (TR148) gates whether a safety label survives cross-family judge triangulation, and a hard negative control (TR146) proves you cannot read safety off the weights. Four reports run the inference-flag null line through those gates: speculative decoding (TR144), FP8 KV-cache at a single config (TR145), FP8 on the field's standardized batteries (TR149), and FP8 across a serving-state factorial (TR152). The outcome is a **serving-state-independent FP8 certificate** for 1B–4B instruction-tuned models on vLLM v0.19.1, with one located footnote (a sub-1pp Qwen-2.5 over-refusal lean on benign-but-alarming prompts). The entire executed substrate ran at **$0 external API cost** on a local judge cohort.

This is not a claim that "FP8 is safe." It is a bounded, behaviourally-measured operating envelope, with every gate carrying a measured threshold and the boundary stated explicitly.

---

## Boundary conditions (do not skip)

This guidance is valid only under the measured boundary:

- **Model scale:** instruction-tuned models in the **1B–4B** parameter range (Llama-3.2, Qwen-2.5, Phi-3 families). The 7B–72B range is *not* certified — it is the explicit, cloud-gated scope of TR151.
- **Serving stack:** vLLM **v0.19.1** (or a CRI-validated neighbour). The FP8 kernel is stack-versioned; a materially different version requires re-confirming the behavioural null.
- **Precision:** FP8 **E4M3** KV-cache vs FP16 baseline. Other KV-quantization methods (the TR153 sweep) are uncovered.
- **Serving-state axes (the certified envelope):** batch size ∈ {1, 8, 32}, prefix caching {on, off}, sampling temperature ∈ {0.0, 0.7, 1.0}.
- **Decoding:** greedy and the three tested temperatures only. The speculative-decoding null (TAIS) is **temperature-0 only** and must not be extrapolated to sampling settings.
- **Context length:** short context. Long-context FP8 (the TR150 scaffold) is uncovered.
- **Judges:** two local LLM judges (gemma3:12b + llama3.1:8b on Ollama) plus regex, under the umbrella gate. The gpt-4o and Claude cross-family axes are credential-deferred.
- **Factorial design:** TR152 is a *star* factorial (one axis varied at a time around a center), so the "additive" verdict covers **single-axis** modulation only; joint two-axis interactions are unmeasured.

If any of these change, re-run the relevant layer and re-validate before applying these decisions.

---

## Definitions (one-time)

The decisions below are stated against named instruments and estimators. Each is a measured object with a calibrated threshold, not a label of convenience.

- **CRI (Compile Reproducibility Index):** the maximum pairwise |Cohen's d| on compiled-latency across a stack-perturbation set, banded `robust` (<0.5) / `sensitive` (<2) / `fragile` (≥2) / `catastrophic` (≥10). It answers: *is this latency number a property of the optimization, or of one unrecorded toolchain?*
- **JTP (Judge Triangulation Protocol):** the cross-family Cohen's κ on the largest-n judge pair, banded `robust` (≥0.70) / `triangulate` (0.40–0.70) / `untrustable` (<0.40). It answers: *can a single judge's safety label be trusted, or must it be triangulated?*
- **TAIS (Typical-Acceptance Invariance Screen):** a matched-pairs Cohen's h with a ±3pp TOST equivalence test over draft–verify decoding pairs. Null cutoff |h| < 0.1. It answers: *does speculative decoding alter the safety-relevant output distribution?*
- **RTSI (Refusal-Template Stability Index):** a four-feature behavioural screen over refusal-template drift, banded low (<0.10) / moderate (0.10–0.40) / high (≥0.40). Inherited from Phase 5 (TR142) as the Layer 2 screen.
- **Matched-pairs McNemar / discordance:** the same prompt run under FP16 and FP8; a *discordant* pair is one where exactly one arm refuses. `b` counts FP8-degraded (FP8 refuses a benign prompt the baseline answered), `c` counts FP8-improved. Concordant pairs carry no information and are differenced out — so a model's high baseline over-refusal does not contaminate the FP8 delta.
- **Cohen's h:** the effect size for the difference between two proportions (paired-binary). Reported instead of Cohen's d throughout the safety line because the outcome is a refusal rate, not a continuous score.
- **TOST at ±3pp:** two one-sided tests against a ±3-percentage-point equivalence margin. Passing TOST is a *positive* claim of equivalence, not merely a failure to reject — the ±3pp band is the deployment-relevance threshold below which a refusal-rate move is operationally inert.
- **Mantel–Haenszel pooled OR:** the matched-pairs discordant-ratio odds ratio pooled across strata (battery × model × serving context), Haldane-corrected. OR ≈ 1.0 with a CI bracketing 1.0 is the pooled null.
- **Star factorial:** a fractional design that varies *one* serving axis at a time around a fixed center cell. It licenses single-axis modulation claims only; joint two-axis interactions are not covered (TR152 samples 12 of the 72 full-factorial cells, 19.4%).
- **H0 / H1 / H2:** the three verdicts of the serving-state factorial. **H0** = no effect within ±3pp; **H1** = a *located* effect (a specific family/battery moves, but no serving knob modulates it); **H2** = a *serving-state interaction* (a knob like batch or temperature changes the FP8 effect). H1 is a per-family footnote on one certificate; H2 would force per-configuration re-validation. TR152 returns **H1 with H2 rejected**.

---

## The five-layer protocol (one table)

The abstract introduced "the five-layer serving-state safety certification protocol." Each layer gates a specific question. A claim is *Supported* only when every layer below it has passed. A deployment that skips a layer trades a measured guarantee for an assumed one.

| Layer | Question it answers | Gate | Phase 6 status | Anchor TR |
|---|---|---|---|---|
| **Layer 1 — Judge validity** | Can this judge's safety label be trusted alone? | JTP κ ≥ 0.70 on calibration corpus → `robust`; 0.40–0.70 → `triangulate`; < 0.40 → `untrustable` | **Passed in triangulate band** (κ = 0.6917 on the TR145 mixed corpus) — protocol defaults to ≥ 3-judge majority vote | TR148 |
| **Layer 2 — Behavioural screens** | Is this (model, quant) combination template-stable and decoding-inert? | RTSI band (low / moderate / high) on refusal-template drift + TAIS \|Cohen's h\| < 0.1 at greedy decoding | **Inherited from Phase 5 (TR142 RTSI) + Phase 6 (TR144 TAIS)**. Negative-control rule (no mechanistic gates) backed by TR146. | TR142, TR144, TR146 |
| **Layer 3 — Stack reproducibility** | Is this latency / behavioural number a property of the optimization or of one unrecorded toolchain? | CRI band `robust` (max pairwise \|Cohen's d\| < 0.5); reportable if `sensitive` (< 2); blocked if `fragile`/`catastrophic` (≥ 2 / ≥ 10) | **`Catastrophic` measured** on `torch.compile` across Triton 3.3.1 / 3.4.0 / 3.6.0 (\|d\| ≈ 14–49). Blocks any cross-version compile-prefill latency claim; behavioural safety null is decoupled from this band. | TR147 |
| **Layer 4 — Scale validity** | Does this hold at production scale (7B–72B)? | Pooled OR brackets 1.0 + TOST coverage ≥ 95% at the target scale on the standardized batteries | **Cloud-gated** — TR151 scaffolded, not run; 1B–4B carries forward as the only certified band. Deployments above 4B revert to scaffolded-only until TR151 lands. | TR151 (deferred) |
| **Layer 5 — Serving-state independence** | Does this hold across the runtime serving configuration (batch / prefix / temperature / speculative decoding)? | Cross-context MH pooled OR with CI clearing 1.0 by a defensible margin **and** FP8-interaction spread within ±3 pp (no H2 modulation) | **Passed (H1, H2 rejected)** — pooled OR 1.88 [1.32, 2.69] (sign-test p ≈ 0.0004), spread 2.99 pp inside ±3 pp, all modulation single-axis. The Qwen footnote attaches to Layer 5, not to the layers below. | TR152 |

The protocol is *inheritance-based*: Layer 5 inherits from Layer 4, which inherits from Layer 3, etc. A deployment outside Layer 4's certified band (i.e., > 4B parameters) reverts to scaffolded-only status at Layer 5 and must re-earn it on its own configuration; the lower layers carry forward unchanged. The protocol never has an all-or-nothing rejection mode — every gate either certifies or downgrades to a documented graceful-degradation path (see *Graceful-degradation policy* below).

> **Why each layer is a *gate* rather than a label.** A label says "this passes" or "this fails" against a fixed threshold. A gate says "this passes *given* a measured threshold, *or* it forces a documented downgrade path." Layer 1 is the canonical example: κ < 0.70 doesn't fail certification — it changes the dispatch policy to triangulation. Layer 3 is the second: CRI `fragile` doesn't fail certification — it blocks the *latency* claim and leaves the behavioural safety claim intact. The framing matters because it lets a deployment honestly adopt the protocol in partial form: every shortfall has a documented more-expensive-but-equally-safe path, never an all-or-nothing reject.

---

## Six decisions you can ship now

These are the Phase 6 operational defaults: one green light and five gates. The gates are the contribution — they let a deployment decide whether its own configuration is inside the certified envelope.

1. **Enable FP8 (E4M3) KV-cache for 1B–4B serving without per-configuration re-validation.** Across batch ∈ {1, 8, 32}, prefix caching {on, off}, and temperature ∈ {0.0, 0.7, 1.0}, FP8 is harmful-refusal-neutral. Evidence: TR152 finds **0 of 8,976** matched harmful-battery pairs discordant (FP16 safe rate 1.0000) and a flat FP8-interaction spread of **2.99pp**, inside the ±3pp deployment-relevance band; this stands on TR145's single-config base case (pooled OR 1.05 [0.90, 1.23]) and TR149's standardized-corpus replication (12/12 cells TOST-equivalent at ±3pp). The certificate is *inherited once* across the tested serving Cartesian product, not re-earned per config.

2. **Carry the Qwen-2.5 over-refusal footnote.** If you run Qwen-2.5 at temperature > 0 on a benign-edge-heavy workload, **monitor the over-refusal rate**. FP8 induces a sub-1pp lean toward declining benign-but-alarming prompts on this family (per-family Mantel–Haenszel OR Qwen **3.878** [2.386, 6.302] vs Llama 0.484, Phi-3 0.130). This is a quality-of-service note — the harmful core is invariant on *every* family — not a safety-degradation warning.

3. **Gate every cross-stack latency number on CRI before you trust it.** Pin the Triton minor version **and** the repository code SHA as part of the benchmark identity. Bumping only the Triton minor version collapsed a compile prefill speedup from 62.82% to 0.84% on one fixed GPU (TR147), with cross-version |Cohen's d| ≈ 14–49. If the CRI band is `fragile`/`catastrophic` (max pairwise |d| ≥ 2), the latency comparison is not reportable.

4. **Triangulate judges unless κ clears 0.70 on your calibration corpus.** A single LLM judge's safety label is only trustworthy if the Judge Triangulation Protocol κ on *your* corpus clears the robust threshold. On the TR145 mixed task set the binding cross-family pair scored κ = **0.6917** [0.6824, 0.7008] — the triangulate bucket — so default to ≥3 judges with majority vote. Never self-select dispatch on the corpus where your judges happen to agree best.

5. **Keep the safety verdict behavioural — never read it off weights, logits, or a forward-pass probe.** Four mechanistic probes (first-token entropy, refusal-direction geometry, calibration drift, safety-neuron quantization error) all fail to distinguish safe from dangerous quantized configs (TR146; all |r| < 0.15, danger-vs-neutral Mann–Whitney p = 0.979). The GPTQ confidence paradox makes this worse: the most dangerous configs look the *most* certain, so a logit/confidence screen would approve them.

6. **Treat speculative decoding as safety-neutral only at greedy decoding.** At temperature 0, speculative decoding is behaviourally inert (TR144 TAIS: max |Cohen's h| = 0.024 across 64,855 paired samples, 90.66% byte-identity). Above temperature 0 the typical-acceptance criterion genuinely alters the output distribution — re-test before relying on it.

> A deployment that follows this card and stays inside 1B–4B / vLLM v0.19.1 / the tested serving axes inherits a *single* FP8 certificate it does not have to re-earn per configuration. A deployment outside that envelope is told exactly which scaffold names its missing evidence — TR151 (scale), TR150 (long context), TR153 (KV-method) — and is never silently extrapolated to.

---

## Decision matrix (one-glance policy)

| Decision point | Verdict | Action | Evidence |
|----------------|---------|--------|----------|
| FP8 KV-cache, 1B–4B, vLLM v0.19.1, tested axes | **CERTIFIED** | Enable without per-config re-validation | TR152 (0/8,976 harmful discordant; interaction spread 2.99pp) |
| FP8 on Qwen-2.5, temp > 0, benign-edge workload | **MONITOR** | Watch over-refusal rate (QoS, not safety) | TR152 (per-family MH OR Qwen 3.878) |
| Cross-stack latency comparison | **GATE on CRI** | Pin Triton + code SHA; abort report if band ≥ fragile | TR147 (62.82% → 0.84% prefill; \|d\| ≈ 14–49) |
| Single-judge safety verdict | **GATE on JTP** | Triangulate unless κ ≥ 0.70 on your corpus | TR148 (κ = 0.6917 → triangulate) |
| Safety read off weights/logits/probe | **FORBIDDEN** | Behavioural measurement only | TR146 (4 probes \|r\| < 0.15; p = 0.979) |
| Speculative decoding at temp > 0 | **UNTESTED** | Re-test; TAIS null is greedy-only | TR144 (max \|h\| = 0.024 at temp 0) |
| FP8 above 4B / long context / other KV-quant | **SCAFFOLDED-ONLY** | Re-earn the certificate on your config | TR151 / TR150 / TR153 (cloud-gated) |
| "FP8 is safe" unconditionally | **NEVER CLAIMED** | Supported form is always bounded | Claim ladder F-tier |

---

## Key findings (decision-grade)

Each finding leads with its headline number and closes with the artifact that carries it.

- **0 of 8,976 matched harmful-battery pairs are discordant** — the harmful refusal core is exactly invariant to the tested serving state. Across FP8 KV-cache, speculative decoding, batch ∈ {1, 8, 32}, prefix caching {on, off}, and temperature ∈ {0.0, 0.7, 1.0}, no knob moves harmful-refusal at all: FP16-only safe rate 1.0000, FP8-only safe rate 1.0000. This is the unconditional guarantee underneath the entire certificate. *Evidence: TR152, 8,976 paired harmful-battery observations (HarmBench + JailbreakBench + StrongREJECT).*

- **The one located FP8 footprint is a sub-1pp benign-edge over-refusal lean, max |Cohen's h| = 0.0458.** All 133 discordant pairs sit on XSTest (the benign-but-alarming battery); the cross-context pooled OR is 1.8817 [1.3185, 2.6855] (sign-test p ≈ 0.0004). This is a *located* effect (H1), not a serving-state interaction (H2): the FP8-interaction spread across all serving contexts is 2.99pp, inside the ±3pp band. The H1/H2 distinction sets the consequence — a located effect is a per-family footnote on one certificate; an interaction would have forced per-configuration re-validation. *Evidence: TR152, 133 discordant of 11,778 XSTest pairs; 20,754 matched pairs total.*

- **A compiled-latency speedup collapsed from 62.82% to 0.84% on a Triton minor-version bump alone.** Holding the GPU and model fixed and changing only Triton (3.3.1 → 3.4.0 → 3.6.0) erased the `torch.compile` prefill win and an 80% decode crash with it; cross-version |Cohen's d| ≈ 14–49 means essentially no distributional overlap, while the eager-mode control stayed flat (|d| ≤ 0.15). A latency number without its pinned six-axis stack identity is not a property of the optimization — it is a property of one unrecorded toolchain. *Evidence: TR147, RTX 6000 Ada, single model.*

- **The binding judge pair agrees at κ = 0.6917 [0.6824, 0.7008] — inside the triangulate band, not robust.** The JTP floor is a property of the *(judge cohort × corpus)* pair, not a constant: the same gemma3:12b × llama3.1:8b pair tightens to κ = 0.8306 on clean standardized batteries. Separately, safety-specialist judges (shieldgemma, llama-guard3) *anti-correlate* with general judges (cross-axis κ = −0.13 to −0.26) while agreeing with each other (κ = +0.21) — they score composite prompt+response harm, not response refusal. Two orthogonal axes, not one noisy axis; triangulation must be performed within an axis. *Evidence: TR148 (n = 12,809) + TR149.*

- **Four mechanistic probes all score |r| < 0.15; danger-vs-neutral Mann–Whitney p = 0.979 — safety cannot be read off the weights.** First-token entropy, refusal-direction geometry, calibration drift, and safety-neuron quantization error all fail to distinguish safe from dangerous quantized configs. The mechanism is *real but necessary-not-sufficient*: safety-critical neurons absorb 1.40× disproportionate quantization error (p < 0.0001; GPTQ 1.45× vs AWQ 1.37×), yet that damage does not predict which configs cross the behavioural failure threshold — every quantized model suffers it, only some fail. The GPTQ confidence paradox makes a logit screen actively dangerous: the worst configs look the most certain. *Evidence: TR146, AWQ/GPTQ INT4.*

- **Speculative decoding holds max |Cohen's h| = 0.024 over 64,855 paired samples — but only at temperature 0.** The byte-identity (90.66%) survives an adversarially-DPO'd draft, a quantized draft, and seed changes across 18 AdvBench contrasts (TR144 E1–E5). Above temperature 0 the typical-acceptance criterion genuinely alters the output distribution and must be re-tested. *Evidence: TR144, greedy decoding only.*

- **The entire executed substrate ran at $0 external API cost.** Every safety verdict came from a local Ollama judge cohort (regex + gemma3:12b + llama3.1:8b) under the umbrella gate; the only metered spend was RunPod GPU time for cells exceeding the local 12 GB envelope. A validity layer that does not require a cloud safety-judge budget is itself a methodological result — and it is why the null is *measured* (powered, bounded, triangulated) rather than merely absent. *Evidence: TR144/145/148/149/152, local judge cohort.*

---

## What is claimed — and where each claim stops

Every headline above is tagged to a tier on the project claim ladder: **Supported** (directly evidenced, CI-bounded, no caveat), **Licensed** (evidenced within an explicit scope stated in the main text), or **Forbidden** (the evidence specifically rejects it). The bound travels with the claim so a deployment — or a reviewer — can defend each one against its most likely objection in a single sentence.

| # | Claim | Tier | Evidence | Where it stops |
|---|-------|------|----------|----------------|
| C1 | FP8 KV-cache is harmful-refusal-neutral across the tested serving state | **Supported** | TR152: 0/8,976 discordant harmful pairs; interaction spread 2.99pp inside ±3pp | 1B–4B, vLLM v0.19.1, tested {batch, prefix, temperature} only |
| C2 | Speculative decoding is behaviourally inert at greedy decoding | **Supported** | TR144: max \|h\| = 0.024 over 64,855 paired samples | temperature 0 only; linear draft–verify only |
| C3 | The FP8 null replicates on the field's standard batteries | **Supported** | TR149: 12/12 cells TOST-equivalent at ±3pp; pooled OR 0.8065 [0.3828, 1.6989] | 1B–3B; discriminating power is effectively a two-battery claim |
| C4 | The judge cohort has a *measured* validity floor | **Supported** | TR148: κ = 0.6917 [0.6824, 0.7008], n = 12,809 → triangulate | TR145 corpus; two-LLM-judge (gpt-4o/Claude deferred) |
| C5 | Safety-specialist and general judges measure orthogonal axes | **Supported** | TR148: four cross-axis κ = −0.13 to −0.26; within-specialist κ = +0.21 | corpus-scale; refusal vs composite-harm construct |
| C6 | A compiled-latency conclusion can flip on a dependency bump | **Supported** | TR147: Triton-only ablation 62.82% → 0.84%; cross-version \|d\| ≈ 14–49 | RTX 6000 Ada, one model; vLLM-stack replication pending |
| C7 | FP8 induces a located over-refusal lean on one model family | **Licensed** | TR152: per-family MH OR Qwen 3.878 [2.386, 6.302] vs Llama 0.484, Phi-3 0.130 | benign-edge (XSTest) only; sub-1pp; temperature axes |
| C8 | The JTP verdict is corpus-specific | **Licensed** | TR149: same pair κ = 0.8306 on clean batteries vs 0.6917 on the mixed set | clean refusals are easier to judge; do not self-select dispatch |
| C9 | Mechanistic probes predict safety degradation under quantization | **Forbidden** | TR146: four probes all \|r\| < 0.15; danger-vs-neutral p = 0.979 | falsified, not merely unconfirmed (AWQ/GPTQ INT4) |
| C10 | "FP8 is safe" unconditionally | **Forbidden** | — | never claimed; supported form is always "no detectable effect under tested conditions" |

Six Supported, two Licensed, two Forbidden. The two Forbidden rows are load-bearing: C9 is why the protocol forbids logit/confidence shortcuts, and C10 is the line the whole envelope is written to never cross.

---

## Worked example: certifying a Qwen 2.5 1.5B FP8 deployment

To make the protocol concrete, walk a real production-shape decision through the five gates. The example is chosen deliberately to land on the most-caveated cell in the envelope — a Qwen-family variant at non-zero temperature with a mixed-intent workload — so the gates exercise both the green-light path and the located-footnote path simultaneously.

**Scenario.** A serving team wants to enable FP8 (E4M3) KV-cache on **Qwen 2.5 1.5B-Instruct** served via **vLLM v0.19.1** on RTX 4080 / A100 hardware, at **batch size 8**, **prefix caching on**, and **sampling at temperature 0.7**. Workload is mixed-intent: routine chat, code assistance, and a non-trivial fraction of benign-edge prompts ("how do I kill a Python process," "what household chemicals should I never mix," etc.) that look adversarial on a regex match.

| Step | Gate | Walk | Verdict |
|---|---|---|---|
| 1 | **Layer 1 (JTP)** — judge cohort calibration | Configure local Ollama cohort: regex + gemma3:12b + llama3.1:8b. Score ≥ 500 calibration prompts; compute pairwise Cohen's κ on the binding (largest-n) cross-family pair. If κ ≥ 0.70 → single-judge mode; 0.40–0.70 → triangulate; < 0.40 → block. | **Triangulate** (default for the Phase 6 cohort on a mixed-intent corpus: κ = 0.6917 on the TR145 reference). Pipeline runs majority-of-3 on adversarial-corpus traffic. |
| 2 | **Layer 2 (RTSI + TAIS + negative control)** — behavioural screens | Run RTSI on Qwen 2.5 1.5B / FP16 baseline; check template-stability band. Run TAIS if speculative decoding is enabled — not in this scenario, skip. Verify no logit / confidence / mechanistic probe is on the safety path (negative control). | RTSI: Qwen 2.5 1.5B is in TR142 v3's evaluated set; not flagged in the high-risk band. TAIS: N/A (greedy not used; sampling temperature 0.7 is the operational decoding mode). Negative control: no probes proposed. **Pass.** |
| 3 | **Layer 3 (CRI)** — stack reproducibility | Pin the six-axis stack identity: GPU sm_N, Triton minor version, PyTorch build, FlashAttention version, vLLM version, code SHA. Compute CRI by perturbing the most-frequently-bumped axis (Triton minor or vLLM patch) on a representative compiled-path benchmark. | If `robust` / `sensitive` → safe to report cross-stack latency. If `fragile` / `catastrophic` → block cross-version latency claims; the FP8 *safety* null still holds (it is behavioural, not latency-keyed). For this deployment: **report behavioural verdict regardless of CRI band**; only the latency-comparison claim depends on it. |
| 4 | **Layer 4 (Scale validity)** — is 1.5B inside the certified band? | The Phase 6 certificate is bounded to 1.2 B – 3.8 B. 1.5B is inside. | **Inside band.** Inherit Layer 4 unchanged. (Had the team chosen Qwen 2.5 7B, this step would have returned scaffolded-only and required re-earning the certificate on production hardware via the TR149 standardized battery.) |
| 5 | **Layer 5 (Serving-state independence)** — does the configuration fall in the tested envelope? | batch ∈ {1, 8, 32} → 8 ✓; prefix ∈ {on, off} → on ✓; temperature ∈ {0.0, 0.7, 1.0} → 0.7 ✓; speculative decoding disabled ✓. All four runnable axes inside the tested envelope. | **Inside envelope.** FP8 certificate applies. |
| 6 | **Per-family caveat check (the located footnote)** | Qwen 2.5 at temp = 0.7 lands in the located footnote — the sub-1pp over-refusal lean on benign-edge prompts. The deployment workload contains benign-edge prompts. | **Carry the QoS footnote.** Wire a benign-edge over-refusal monitor (rolling decline rate on a small held-out canary set of XSTest-style prompts). This is a quality-of-service alarm, not a safety alarm — harmful refusal remains exactly invariant. |
| 7 | **Production-monitoring add-ons** | Add: (a) first-order alarm on harmful-refusal discordance (any non-zero on a config inside the envelope is a real alarm), (b) rolling JTP κ on a calibration sample (revert to triangulation if it drops below 0.70), (c) CRI re-run on any stack-identity change. | Wire all three. |

**Resulting deployment decision.**

- ✅ **Enable FP8 KV-cache** on Qwen 2.5 1.5B / vLLM v0.19.1 / batch 8 / prefix on / temp 0.7.
- ✅ **Triangulated judge pipeline** (majority-of-3 across regex + gemma3:12b + llama3.1:8b).
- ⚠️ **Benign-edge over-refusal dashboard** for Qwen at temperature > 0 (QoS, not safety).
- 📌 **Stack identity pinned** in deployment manifest; CRI re-run on any version bump.
- ❌ **No logit-based or mechanistic safety screens** anywhere on the dispatch path.

**Cost of compliance.** Every gate above runs on local infrastructure with zero external API spend. The judge cohort under triangulation adds ~50 ms median latency per scored response, amortized into the existing safety-eval budget. The CRI calibration runs once per stack change, not per request. The whole protocol is a one-time configuration cost plus thin steady-state monitoring — and the team holds a *measured*, not assumed, certificate for their exact configuration. If any single gate later fails, the documented downgrade path applies: triangulation expands to 4-5 judges, latency claims narrow to within-stack only, scale extension reverts to scaffolded-only.

> A team that completes this walkthrough has a measured FP8 certificate for their exact serving configuration and a documented graceful-degradation path for every gate. The seven steps are reproducible from the configuration alone; a different team running the same scenario on the same stack inherits the same verdict. That reproducibility is what separates a *protocol* from a *one-off audit*.

---

## The located finding, decomposed

The single non-null result — the Qwen-XSTest over-refusal lean — is worth decomposing exactly, because a pooled OR of 1.88 read without its structure would manufacture a generic "FP8 over-refuses" warning out of a family-specific, benign-edge, sub-1pp effect. Three cuts show where the 133 discordant pairs live.

**By battery — all discordance is on the benign-edge battery; the harmful core is clean.**

| Battery | Discordant / paired | Reading |
|---------|--------------------|---------|
| HarmBench-400 | **0 / 2,995** | harmful core invariant |
| JailbreakBench-100 | **0 / 2,996** | harmful core invariant |
| StrongREJECT-313 | **0 / 2,985** | harmful core invariant |
| XSTest-450 (benign-edge) | **133 / 11,778** | the entire FP8 footprint |

**By family — the footprint is concentrated on Qwen, and runs the *opposite* way on the others.** Discordance counts both directions; the per-family MH odds ratio reads the direction.

| Family (per-variant discordant) | XSTest discordant | Per-family MH OR | Direction |
|---------------------------------|-------------------|------------------|-----------|
| Qwen-2.5 (1.5B: 66, 3B: 33) | 99 | **3.878 [2.386, 6.302]** | FP8 *degrades* (CI well above 1.0) |
| Llama-3.2 (1B: 16, 3B: 6) | 22 | 0.484 | FP8 *improves* |
| Phi-3-mini (4k: 12) | 12 | 0.130 | FP8 *improves* (CI excludes 1.0) |

Of the 87 FP8-*degraded* pairs (the over-refusal direction), **79 are Qwen — 91%**. The 46 FP8-*improved* pairs run the opposite way on Llama and Phi-3, so the effect is directional-by-family, not a uniform lean that Qwen merely amplifies. Because the test is matched-pairs discordance, Qwen's high baseline over-refusal (37.2% both-refuse rate on 1.5B) is differenced out — what remains is the FP8-attributable delta on top of it.

**By serving axis — no single axis modulates the FP8 contrast at its mean (H2 rejected).**

| Serving context (single-axis deviation) | Mean FP8 Δ (pp) | Min / max cell Δ (pp) | n paired |
|------------------------------------------|-----------------|------------------------|----------|
| baseline (batch 1, prefix off, greedy) | −0.012 | −1.02 / +0.76 | 3,466 |
| batch_size = 8 | −0.075 | −1.02 / +0.26 | 3,469 |
| batch_size = 32 | −0.126 | −1.51 / +0.26 | 3,468 |
| prefix_caching = on | −0.025 | −1.02 / +0.76 | 3,468 |
| temperature = 0.7 | −0.164 | −2.23 / +0.26 | 3,432 |
| temperature = 1.0 | −0.110 | −2.21 / +0.51 | 3,451 |

Every per-context mean delta sits within ±0.17pp of zero, and the full spread of per-cell deltas (max − min across all contexts) is **2.99pp — inside the ±3pp band**. Temperature is the most active axis, which is exactly why the operational caveat names *Qwen at temperature > 0*; even there the modulation does not clear deployment-relevance. The speculative-decoding axis is covered separately by TR144's TAIS null (greedy-only).

---

## Operational recommendations (policy statements)

### FP8 KV-cache policy (Supported, bounded)

- **Policy:** For instruction-tuned 1B–4B models on vLLM v0.19.1, enable FP8 (E4M3) KV-cache across the certified serving axes (see *Boundary conditions*) without per-configuration safety re-validation.
- **Policy caveat:** For the Qwen-2.5 family at temperature > 0 on benign-edge-heavy traffic, monitor the over-refusal rate. The effect is absent on Llama-3.2 and Phi-3.
- **Policy boundary:** Above 4B, in long context, or under a different KV-quant method, the certificate is scaffolded-only. Run the standardized battery on your own configuration before enabling FP8.

### Measurement-validity policy (the gates)

- **Policy (Layer 1, JTP):** Run the κ check on your judge cohort against a fixed calibration corpus. If κ < 0.70, triangulate (≥3 judges, majority vote) or do not license the labeled verdict. Periodically re-check κ as traffic distribution drifts.
- **Policy (Layer 3, CRI):** Compute CRI across candidate serving stacks before reporting any cross-stack latency number. Pin and publish the six-axis stack identity (GPU, Triton, PyTorch, cache impl, compile mode, code SHA). If the band is ≥ fragile, report only within-stack numbers.
- **Policy (Layer 2, screens):** Run RTSI on refusal-template features to flag template-unstable (model, quant) cells before serving; run TAIS if speculative decoding is enabled at greedy decoding.
- **Policy (negative control):** Never substitute a mechanistic/logit/confidence probe for behavioural measurement. The shortcut is falsified and the confidence signal is actively misleading.

### Graceful-degradation policy

- **If you cannot run Layer 1:** default to unconditional triangulation. You pay 3–4× judge cost but never risk the single-judge verdict-flip. Cost up, risk unchanged.
- **If you cannot run Layer 3:** stop making cross-stack latency claims; report within-stack only. A scope restriction, not a risk.
- **If you cannot run Layer 2:** run the full behavioural battery on every config rather than the cheap screen. More expensive, equally safe.
- **If you cannot run Layer 4/5 at your scale or serving state:** treat the FP8 certificate as scaffolded-only and re-earn it on your configuration.

> Every degradation path moves toward more cost and never toward more risk — the property a safety protocol must have to be honestly adoptable. The gates are *validated cost optimizations*, not all-or-nothing safety requirements: failing to run one forfeits the cost reduction, never the safety floor.

---

## Risk impact

### Ranked by certificate consequence

1. **Harmful-refusal movement (the core guarantee).** Exactly invariant in the tested envelope (0/8,976 discordant). *Any* non-zero discordance in production on a config inside the envelope is a first-order alarm, not a quality note. This is the only signal whose movement invalidates the core certificate.
2. **Serving-stack version drift.** A latency conclusion can flip catastrophically on a transitive-dependency bump, and the FP8 kernel is itself stack-versioned. Re-run CRI on any change to the six-axis benchmark identity.
3. **Judge-cohort agreement drift.** The κ floor is corpus-specific (0.69 mixed, 0.83 clean); a traffic-distribution shift can move it below 0.70 and silently invalidate single-judge verdicts. Re-check periodically; revert to triangulation when it drops.
4. **Qwen-2.5 benign-edge over-refusal.** A rising decline rate on known-benign requests (especially after a temperature or batch change on Qwen) degrades usefulness. Second-order: a QoS dashboard, not a safety alarm.

### Worst-case and best-case anchors

- **Strongest located footprint:** Qwen-2.5 over-refusal on XSTest under FP8 at temperature > 0 — per-family MH OR 3.878, max |h| 0.0458, still sub-1pp absolute.
- **The shortcut that fails worst:** a logit/confidence screen under GPTQ, which grows *more* confident while *failing* to refuse (TR146).
- **The cleanest invariance:** the harmful core under serving-state composition — 8,976 paired observations, zero counterexamples (TR152).

---

## Reviewer objections — anticipated and answered

Decision-grade work absorbs the most-likely pushback up front. Each objection below is paired with the evidence that closes it — the answers travel with the whitepaper so a deployment, advisor, or external auditor can defend the position in a single exchange. Section references point into the Conclusive (`Conclusive_Phase6.md`) and the source TR (`Technical_Report_<NNN>.md`) for the load-bearing evidence.

**Objection 1: "Pooled OR 1.88 on FP8 sounds like a non-trivial degradation. Why is this not a safety alarm?"**

Decompose by battery before reading the pooled number. All 133 discordant pairs sit on XSTest (the benign-edge battery); the harmful core is 0 / 8,976 discordant. The "degradation" is over-refusal on benign-but-alarming prompts, not under-refusal on harmful prompts — absolute rate change is sub-1pp. The headline pooled OR is a Qwen-2.5 / XSTest signal pooled into one number; the **per-family Mantel–Haenszel decomposition (Qwen 3.878, Llama 0.484 [improving], Phi-3 0.130 [statistically improving]) is the right read**. The headline pooled OR is a generalization artifact of mixing three opposite-direction family effects; the per-family table is the verdict. *(Evidence: TR152 SS23, Conclusive §6.)*

**Objection 2: "You're calling FP8 'safe' but you only tested vLLM v0.19.1 on 12 GB hardware. Production runs at 80 GB and a newer vLLM."**

The whitepaper never claims FP8 is unconditionally safe — it claims FP8 is harmful-refusal-neutral *within* the stated boundary (Boundary conditions section), and explicitly lists which layers are scaffolded-only outside that boundary (Layer 4 for scale, the TR153 sweep for KV-quant method, TR150 for long context). A deployment at 7B–72B on a newer vLLM treats the certificate as scaffolded-only and re-earns it via the standardized battery on its production stack. **The downgrade path is explicit; the boundary is the contribution, not a defect.** A protocol whose boundary is silent is the dangerous one — Phase 6's boundary is named in every claim row. *(Evidence: Boundary conditions; Claim ladder C1's "Where it stops" column.)*

**Objection 3: "How do I know the judge cohort isn't itself compromised by the same quantization the model uses?"**

Two safeguards. **First**, the judges are LLM-judges (gemma3:12b + llama3.1:8b on Ollama Q4_K_M) evaluating *target-model outputs*, not their own outputs — so judge-quantization effects are decoupled from target-model effects. **Second**, TR148 measured cross-family κ on a fixed corpus (κ = 0.6917 on the TR145 mixed set) and TR149 re-measured on a different corpus (κ = 0.8306 on the standardized clean-refusal batteries); both numbers are reported and the *triangulate* posture (≥ 3 judges, majority vote) defaults under κ < 0.70. The judge cohort is **in the same protocol as the target — Layer 1 (JTP) gates it explicitly**. The verdict is not "trust the judge"; it is "use the judge if and only if Layer 1 passes." *(Evidence: TR148, TR149; Layer 1 row in the five-layer protocol table.)*

**Objection 4: "You report 0 / 8,976 harmful discordant. Isn't that just a floor effect — every prompt refuses regardless?"**

Yes — and that is the design feature, not a confound. The matched-pairs McNemar estimator differences out concordant pairs (both refuse or both comply) and reports only discordant ones (FP8 refuses but FP16 didn't, or vice versa). Floor effects on the harmful batteries mean every FP16 sample refused; the question the design asks is **whether *any* FP8 sample failed to** under any of the 6 × 5 × 3 = 90 (context × model × battery) cells. **Zero counterexamples across 8,976 paired observations is the strongest invariance statement the design can produce**. Adding more harmful prompts buys near-zero additional information; v2 correctly spent its expansion budget on the XSTest battery, which had non-degenerate discordance and was the locus of v1's signal. *(Evidence: TR152 SS4, SS8, SS17; matched-pairs McNemar in the Definitions section.)*

**Objection 5: "The five-layer protocol looks bureaucratic. What's the minimum a deployment can do and still claim something defensible?"**

The graceful-degradation policy in *Operational recommendations* makes this explicit. The minimum is: **(a) triangulate judges unconditionally** (Layer 1 fallback — pay 3× judge cost, never risk single-judge verdict flip); **(b) stop making cross-stack latency claims** (Layer 3 fallback — report within-stack only); **(c) treat FP8 as scaffolded-only and re-run the standardized battery on the production configuration** (Layer 4/5 fallback — re-earn the certificate on your stack). Every fallback is more expensive but no less safe than the gated path; **the protocol cannot be partially adopted into an unsafe state**. A deployment that runs no Phase 6 gate at all is left with the unconditional triangulation + scaffolded-only baseline — slower and more expensive than the certificate, but the safety guarantee is identical. *(Evidence: Operational recommendations / Graceful-degradation policy.)*

**Objection 6: "Mechanistic interpretability is improving. Why insist on never using a logit / probe gate?"**

Because the *exact* failure mode TR146 documented is the one a future logit / probe gate would embed: quantization damages safety-critical neurons proportionally more than benign neurons, but **that damage does not predict which configs cross the behavioural failure threshold**. The GPTQ confidence paradox actively misleads — the most dangerous configs grow the most certain. Until a mechanistic gate produces a behavioural-equivalent verdict on the same falsification set (4 probes × n configs, all behavioural-out-of-band) the protocol must read safety from behavioural evidence. The Forbidden tier (claim C9) is **not a permanent ban on mechanistic work; it is a ban on *deploying* a mechanistic gate as a safety substitute**. The behavioural gate remains; the mechanistic probe is permitted as a *research artifact* and forbidden as a *deployment gate*. The day a mechanistic probe passes the TR146 falsification suite is the day the protocol revisits the C9 row. *(Evidence: TR146 four-probe falsification; Conclusive §4.)*

**Objection 7: "Phase 6 forward-points to TR163 as a mitigation demo. What's its current status?"**

TR163 is an **MVP feasibility demo** — an offline LOOCV reanalysis over the TR142 51-cell RTSI table (no new GPU sampling, ~16 KB of analysis JSON) showing the RTSI-gated routing concept recovers 76% of the refusal gap at ~22% routed, LOOCV AUC 0.84. **It is a proof-of-mechanism, NOT a paper-grade campaign and NOT a Phase 6 deliverable.** A paper-grade expansion — online RTSI-gated routing across 5 models × standardized 4-battery cohort × 3 routing policies × 2 KV-cache dtypes (~75K primary records + ~225K judge labels under JTP triangulation + hold-out LOOCV) — is queued. No downstream synthesis or external promotion will land until that expansion executes. The Phase 7 paper synthesis (three-layer mitigation comparison: behavioural-screen routing / serving-state sink policy / representation-direction patching) is blocked on at least two of those three layers reaching paper-grade substrate. The mitigation concept is **selective — it does not blanket-disable FP8** — and inherits its right to use RTSI from TR146's behavioural-discipline finding (mechanistic probes are *forbidden* as gates; RTSI is *behavioural* and passes Layer 2). *(Evidence: TR163 (in PublishReady forward-link); Conclusive §13.2.)*

**Objection 8: "Why ICML Hypothesis Testing rather than a flagship safety venue?"**

Phase 6's contribution is **methodological-first**: a five-layer protocol that gates *whether* a serving flag can be certified, with FP8 KV-cache as the worked example. The Hypothesis Testing framing — pre-registered H0 / H1 / H2 with operational thresholds, matched-pairs estimators with calibrated equivalence bands, claim-ladder tiering tied to evidence — is the venue-native vocabulary. A flagship safety venue would correctly ask "what does FP8 do to safety," and the answer (Conclusive §4) is "nothing, except this one located footnote." The Hypothesis Testing venue asks the harder methodological question Phase 6 actually answers: "how do you know?" The flagship-safety submission lane is for the Phase 5 mitigation work (TR163 + future), not the validity substrate Phase 6 builds. *(Reference: `papers/serving_state_safety_certification/`; Phase 5 batch-inference-safety paper accepted at ICML 2026 Workshop on Hypothesis Testing — the precedent for this venue alignment.)*

> The objection set above is meant to be *defensive armament*, not authoritative answers. A reviewer is free to push deeper on any of these; the linked evidence is the contact point. Decisions in this whitepaper survive these objections as written; if a reviewer finds a path the evidence does not close, that path is a Phase 5 (or later) research question, not a Phase 6 retraction.

---

## Implementation plan (30-day view)

**Days 1–7: Stand up the two validity gates.**

- Compute CRI across your candidate serving stacks; pin the six-axis stack identity into the deployment manifest. Stop reporting any cross-stack latency number whose CRI band is ≥ fragile.
- Run the JTP κ check on your judge cohort against a fixed calibration corpus. If κ < 0.70, switch the safety-verdict pipeline to ≥3-judge majority vote.

**Days 8–14: Apply the FP8 certificate inside the envelope.**

- Confirm your model scale is inside 1B–4B and your stack is vLLM v0.19.1 (or CRI-validated). If yes, enable FP8 KV-cache across the certified axes without per-config re-validation.
- If outside the envelope (7B–72B, long context, other KV-quant), run the standardized battery on your own configuration before enabling FP8; treat the certificate as scaffolded-only.

**Days 15–21: Wire the behavioural screens and the negative-control rule.**

- Run RTSI on refusal-template features to flag template-unstable (model, quant) cells. Run TAIS if speculative decoding is enabled at greedy decoding.
- Remove any logit/confidence/mechanistic safety screen from the deployment path; replace with behavioural measurement on a known corpus.

**Days 22–30: Production monitoring.**

- Add a first-order alarm on harmful-refusal discordance for any config inside the certified envelope (target: flat at 100% refusal).
- Add a QoS dashboard for Qwen-2.5 benign-edge over-refusal at temperature > 0.
- Add meta-monitors: rolling JTP κ on a calibration sample (revert to triangulation below 0.70) and stack-identity drift (re-run CRI on any change).

---

## Risks, limitations, invalidation triggers

### Limitations

- **Scale anchored at 1B–3B (plus one 70B speculation cell).** The production-relevant 7B–72B FP8-KV-cache range is unmeasured and is TR151's explicit, cloud-gated scope.
- **Greedy / three-temperature only.** Every null-line result is at temperature ∈ {0.0, 0.7, 1.0}; the TAIS speculative-decoding null is temperature-0 only and must not be extrapolated to sampling settings.
- **Star factorial, not full crossing.** TR152 varies one serving axis at a time around a center, so the "additive" verdict covers single-axis modulation only; joint two-axis interactions (e.g. batch × temperature jointly) are unmeasured.
- **Two LLM judges.** The JTP floor stands on gemma3:12b + llama3.1:8b; the gpt-4o and Claude cross-family axes are credential-deferred. The κ floor is a two-LLM-judge measurement.
- **Single local SKU.** CRI's Triton kill-shot is demonstrated on one RTX 6000 Ada GPU; vLLM-stack replication of the compile-integrity finding is pending.
- **Quant-format coverage.** Mechanistic probing is AWQ/GPTQ INT4 only; the GGUF k-quant danger zone Phase 5 flagged is not re-probed.

### What invalidates this guidance

- Model scale outside 1B–4B (re-earn via TR151).
- Serving stack materially different from vLLM v0.19.1 (re-confirm the behavioural null on your stack).
- KV-quantization method other than FP8 E4M3 (the TR153 sweep).
- Decoding temperature outside the tested set, or speculative decoding above temperature 0.
- Long context (the TR150 scaffold).
- A different judge cohort or calibration corpus (re-measure the κ floor — it is corpus-specific).

---

## Evidence anchors (audit-ready)

| Decision | Artifact (primary run-dir) | Key Number |
|----------|----------------------------|------------|
| FP8 serving-state certificate | `research/tr152/results/20260526_232600/tr152_analysis.json` | 0/8,976 harmful discordant; FP8-interaction spread 2.99pp inside ±3pp; 20,754 matched pairs |
| Qwen over-refusal footnote | `research/tr152/results/20260526_232600/tr152_analysis.json` | per-family MH OR Qwen 3.878 [2.386, 6.302]; all 133 discordant on XSTest |
| FP8 single-config base case | `research/tr145/results/20260508_033550/tr145_analysis.json` | 24,054 records; pooled OR 1.05 [0.90, 1.23] |
| FP8 standardized replication | `research/tr149/results/20260514_001356/tr149_analysis.json` | 12/12 TOST-equivalent at ±3pp; pooled OR 0.8065; I² = 0.0% |
| Speculative-decoding inertia (TAIS) | `research/tr144/results/20260412_metrics_rerun/tr144_analysis.json` (E1–E5: `tr144/results/e1_70b_pair/`, `e2_adversarial/`) | max \|Cohen's h\| = 0.024 over 64,855 paired samples |
| Compile reproducibility (CRI) | `research/tr147/results/20260412_195222/tr147_analysis.json` | prefill 62.82% → 0.84%; cross-version \|d\| ≈ 14–49 |
| Judge triangulation floor (JTP) | `research/tr148/results/20260512_174624/tr148_analysis.json` | κ = 0.6917 [0.6824, 0.7008] → triangulate |
| Dual-axis judge finding | `research/tr148/results/20260512_174624/pairwise_kappas.json` | cross-axis κ = −0.13 to −0.26; within-specialist κ = +0.21 |
| Mechanistic-probe falsification | `research/tr146/results/phase4_gptq_docker/regime_tests.csv` (+ `rtsi_correlation.json`) | 4 probes all \|r\| < 0.15; danger-vs-neutral Mann–Whitney p = 0.979 |
| Full certificate + claim ladder | `PublishReady/reports/Conclusive_Phase6.md` | §1.1 (C1–C10), §16 (production doctrine) |
| Phase 7 forward link (MVP mitigation demo, paper-grade expansion queued) | `PublishReady/reports/Conclusive_Phase6.md` | §13.2: TR163 ≈76% gap recovery, LOOCV AUC 0.84 |

---

## References

- Conclusive synthesis: `PublishReady/reports/Conclusive_Phase6.md`
- Extended appendices: `PublishReady/reports/Conclusive_Phase6_Extended_Appendices.md`
- TR144: `PublishReady/reports/Technical_Report_144.md` (Speculative Decoding × Safety / TAIS)
- TR145: `PublishReady/reports/Technical_Report_145.md` (FP8 KV-cache single-config)
- TR146: `PublishReady/reports/Technical_Report_146.md` (Mechanistic-probe falsification)
- TR147: `PublishReady/reports/Technical_Report_147.md` (Compile Reproducibility Index / CRI)
- TR148: `PublishReady/reports/Technical_Report_148.md` (Judge Triangulation Protocol / JTP)
- TR149: `PublishReady/reports/Technical_Report_149.md` (FP8 standardized batteries)
- TR152: `PublishReady/reports/Technical_Report_152.md` (FP8 serving-state factorial)
- Phase 5 whitepaper (predecessor): `PublishReady/reports/Technical_Report_Conclusive_Phase5_Whitepaper.md`
- Phase 5 conclusive: `PublishReady/reports/Technical_Report_Conclusive_Phase5.md`
- Claim ladder: `papers/serving_state_safety_certification/CLAIM_LADDER.md`

---

## How this extends Phase 5

Phase 5 (TR138–143) mapped the attack surface — where batching, multi-turn pressure, many-shot exploitation, cross-architecture fragility, and cross-request composition can move refusal behaviour. This whitepaper does not supersede those maps; it **validates the instruments they were drawn with and re-tests the most-deployed flag under them.**

- **Phase 5 produced one near-perfect judge calibration** (κ = 0.925 on its corpus) and every shipped Phase 5 verdict implicitly assumed it generalized. JTP (TR148) tested the premise and found it holds *within band* — κ = 0.6917 on a different corpus and pair — and TR149 showed κ is itself corpus-specific (0.83 on clean batteries). Judge agreement is a property of the *(cohort × corpus)* pair, not a fixed constant. This is a standing operational rule now, not a one-report result.
- **Phase 5 flagged batch-size non-determinism as a plausible safety mover** (TR138). The TR152 factorial turns that exact knob against the FP8 contrast and finds it additive (H2 rejected). The Phase 5 batching null is now a *gated* null rather than a single-judge bespoke-corpus null — the finding did not change, its defensibility did.
- **Phase 5's RTSI (TR142)** is inherited verbatim as the Layer 2 behavioural screen; Phase 6 composes with it rather than re-deriving it, and pairs it with TAIS (TR144).

Together the two whitepapers cover the characterized optimization-safety surface for the bounded envelope: Phase 5 says where the serving state can move safety, Phase 6 builds the gates that measure it and shows the most-deployed flag does not move the harmful core under them. The next link — **Phase 7** — uses these certified screens and judges to *mitigate* the located vulnerabilities; its first MVP feasibility demo, TR163, is an RTSI-gated quantization-routing policy run as offline LOOCV reanalysis over TR142's 51-cell RTSI table — a proof-of-mechanism showing the routing concept recovers ≈76% of the refusal gap at ~22% routed, LOOCV AUC 0.84 — **not a paper-grade campaign**. Paper-scale expansion required before any external promotion. TR163 inherits its right to trust RTSI from TR146's behavioural-only discipline.

Gaps remain, each tagged to its scheduled closure: scale beyond 4B (TR151), long context (TR150), other KV-quant methods (TR153), stochastic decoding and non-linear speculation topology, and the two deferred external judge axes. All are cloud-gated on the bridge-paper GO/NO-GO trigger (2026-10-24).

---

## Optional upgrades (board-ready polish)

- Add a one-page **certificate card**: the five-layer protocol table + boundary conditions + the single FP8 green-light statement, for distribution to deploying teams.
- Add a **decision-tree poster** of the §16.6 walkthrough (scale → stack → judge → FP8 → speculation → no-shortcut), so a team can walk the gates in an afternoon.
- Add a **CRI dashboard** that recomputes the band on every stack-identity change and blocks latency reports above the `fragile` threshold.
- Commission the deferred **frontier-judge axes** (gpt-4o, Claude) once credentials land, to tighten the two-judge κ floor toward a four-judge measurement.
- Run the **TR151 scale extension** (7B–72B) at the GO/NO-GO trigger to widen the certificate's most consequential boundary — the scale-validity layer is the one most prominently marked "anchored for 1B–3B, handed forward."
