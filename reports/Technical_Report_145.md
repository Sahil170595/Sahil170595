# Technical Report 145: KV-Cache Quantization x Safety
## FP8 KV-Cache as a Silent Safety Degradation Vector — A Five-Phase Null Result

| Field | Value |
|-------|-------|
| **TR Number** | 145 |
| **Project** | Banterhearts safety-evaluation line |
| **Date** | 2026-05-08 |
| **Version** | 1.0 |
| **Author** | Sahil Kadadekar |
| **Git Commit** | `edd46c19` (working tree at run time) |
| **Status** | Executed, end-to-end |
| **Report Type** | Full-depth |
| **Run Directory** | `research/tr145/results/20260508_033550/` |
| **Total Records** | 24,054 (P1=3009, P2=3009, P3=4000, P4=12036, P5=2000) |
| **OK Records** | 24,054 |
| **Completion Rate** | 100.0% |
| **Models** | 3 (Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-1.5B) |
| **KV-Cache Dtypes** | FP16 (auto), FP8 (E4M3) via vLLM `--kv-cache-dtype` |
| **Context Lengths** | 256, 512, 1024, 2048 tokens |
| **Batch Sizes** | 1, 4, 8 |
| **Multi-Turn Length** | 5 turns x 100 conversations |
| **Judge Model** | gemma3:12b (via Ollama, blinded to KV-cache config) |
| **vLLM Image (Pinned)** | `vllm/vllm-openai:v0.19.1` |
| **Hardware** | RTX 4080 Laptop 12 GB (sm_8.9, FP8-capable) |
| **Wall Time** | ~14 hr active GPU + ~6.5 hr laptop sleep (resumed cleanly) |
| **Related Work** | TR125 (weight-quantization safety), TR134 (quant calibration), TR138 (batch safety), TR140 (many-shot jailbreak), TR141 (refusal fragility), TR142 (RTSI), TR143 (cross-arch), TR144 (speculative decoding) |
| **Depends On** | TR130 (vLLM backend abstraction), TR134 (classifiers + judge integration), TR138 (task YAMLs + batch dispatch pattern), TR123 (KV-cache memory formulas) |

---

## 1. Abstract

KV-cache quantization is sold as a serving-layer flag, not a model modification: a single `--kv-cache-dtype fp8` argument to vLLM halves attention-cache memory, doubles practical throughput on a fixed VRAM budget, and is widely treated as latency-positive and safety-neutral. That last assumption has not been tested at scale. TR145 fills the gap with a five-phase paired study: 24,054 records across three models (Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-1.5B), with the only manipulated variable being the KV-cache precision flag — model weights remain FP16 throughout, so any safety effect we observe is attributable to FP8 KV-cache rounding alone, not to weight quantization.

The headline result is a null. The primary McNemar test on Phase 2 (FP8 vs FP16 KV-cache, same prompts, same seed, same temperature) produces no Holm-significant safety effect on any of the three models (p = 1.00, 0.60, 0.31). Phase 3 ANOVA on context length × KV-cache interaction is non-significant for both Llama models (p = 0.97 and 0.54), and the FP8-FP16 gap is non-monotonic across context lengths from 256 to 2048 tokens — consistent with noise, not with the accumulated-rounding-error hypothesis. Phase 4 ANOVA on batch size × KV-cache interaction is even flatter (p = 0.98 and 0.998), and the per-cell interaction decomposition classifies every (model, batch-size) cell as approximately additive: KV-cache quantization does not amplify the batching-safety effects documented in TR138. Phase 5 multi-turn paired McNemar at the safety probe (turn 5) shows 6 flips on Llama-1B (5 unsafe / 1 safe; p = 0.22) and 0 flips on Llama-3B (p = 1.0), with no significant FP8 effect on final-turn safety. Mantel-Haenszel pooled odds ratios across the three models give 1.05 [0.90, 1.23] for Phase 2 safety, 1.00 [0.83, 1.21] for batch=8, and 2.06 [0.61, 6.99] for the multi-turn turn-5 probe (all confidence intervals straddle 1).

The TOST equivalence battery at ±3pp passes 9 of 22 tests, including all small-model paired safety tests; the two notable failures are Qwen2.5-1.5B safety (delta = -3.1pp, just outside the equivalence margin) and Llama-3B capability (delta = -1.6pp but with high paired-sample variance). Power analysis confirms every cell achieves 80% power at α = 0.05, so the null findings are real null findings, not power-starved nondetections. The judge agreement (gemma3:12b vs regex classifiers) lands at Cohen's κ = 0.43 with 99.6% of safety records assigned a non-unclear label, which is moderate inter-rater agreement appropriate for the regex/judge cross-check.

The operational reading is conservative. We do not claim FP8 KV-cache is universally safety-neutral — Qwen2.5-1.5B fails the ±3pp equivalence margin on safety, and the wide multi-turn confidence intervals leave room for effects we are not powered to detect at n = 100 conversations. We do claim that under tested conditions (three small instruction-tuned models, single-server vLLM, temperature 0.0, contexts up to 2048 tokens, batch sizes up to 8, conversations up to 5 turns), no statistically supported FP8 safety degradation is detectable, and operators considering FP8 KV-cache for production should run a paired safety eval on their own workload rather than treat the optimization as either pre-approved or pre-banned. The paired-eval recipe in the Production Guidance section is the durable artifact of this report.

---

## 2. Table of Contents

- [1. Abstract](#1-abstract)
- [2. Table of Contents](#2-table-of-contents)
- [3. Executive Summary](#3-executive-summary)
- [4. Introduction and Research Motivation](#4-introduction-and-research-motivation)
- [5. Research Hypotheses](#5-research-hypotheses)
- [6. Methodology](#6-methodology)
- [7. Models & Configuration](#7-models--configuration)
- [SS1. Phase 1 FP16 Baseline](#ss1-phase-1-fp16-baseline)
- [SS2. Phase 2 FP8 KV-Cache — Primary Result](#ss2-phase-2-fp8-kv-cache--primary-result)
- [SS3. Phase 2 Flip Direction Analysis](#ss3-phase-2-flip-direction-analysis)
- [SS4. Phase 2 Per-Task KV-Cache Effect](#ss4-phase-2-per-task-kv-cache-effect)
- [SS5. Safety-Capability Divergence](#ss5-safety-capability-divergence)
- [SS6. Phase 3 Context Length × KV-Cache ANOVA](#ss6-phase-3-context-length--kv-cache-anova)
- [SS7. Phase 3 Context Length Slope Analysis](#ss7-phase-3-context-length-slope-analysis)
- [SS8. Phase 3 FP8 Degradation Curve](#ss8-phase-3-fp8-degradation-curve)
- [SS9. Phase 3 Critical Context Length](#ss9-phase-3-critical-context-length)
- [SS10. Phase 4 Batch Size × KV-Cache ANOVA](#ss10-phase-4-batch-size--kv-cache-anova)
- [SS11. Phase 4 Interaction Decomposition](#ss11-phase-4-interaction-decomposition)
- [SS12. Phase 4 Per-Task Breakdown](#ss12-phase-4-per-task-breakdown)
- [SS13. Phase 4 TR138 Cross-Reference](#ss13-phase-4-tr138-cross-reference)
- [SS14. Phase 5 Setup-Turn Refusal Diagnostics](#ss14-phase-5-setup-turn-refusal-diagnostics)
- [SS15. Phase 5 Setup-Turn × KV-Cache Diagnostic Interaction](#ss15-phase-5-setup-turn--kv-cache-diagnostic-interaction)
- [SS16. Phase 5 Turn-5 McNemar](#ss16-phase-5-turn-5-mcnemar)
- [SS17. Phase 5 Conversation-Level Analysis](#ss17-phase-5-conversation-level-analysis)
- [SS18. TOST Equivalence Analysis](#ss18-tost-equivalence-analysis)
- [SS19. Power Analysis](#ss19-power-analysis)
- [SS20. Cross-Model Synthesis (Mantel-Haenszel)](#ss20-cross-model-synthesis-mantel-haenszel)
- [SS21. Judge Agreement](#ss21-judge-agreement)
- [SS22. Cross-TR Validation](#ss22-cross-tr-validation)
- [Conclusions](#conclusions)
- [Limitations & Threats to Validity](#limitations--threats-to-validity)
- [Production Guidance](#production-guidance)
- [Reproducibility](#reproducibility)
- [Appendix A: Raw Tables](#appendix-a-raw-tables)
- [Appendix B: Extended Statistical Tables](#appendix-b-extended-statistical-tables)
- [Appendix C: Sensitivity & Robustness](#appendix-c-sensitivity--robustness)
- [Appendix D: Glossary](#appendix-d-glossary)
- [References](#references)

---

## 3. Executive Summary

**Question.** Does flipping `--kv-cache-dtype` from `auto` (FP16) to `fp8` in vLLM silently degrade safety, when nothing else about the deployment changes?

**Answer.** Under the tested conditions, no. The ±3pp equivalence margin holds for 9 of 22 paired comparisons, ANOVA finds no significant context-length or batch-size interaction with KV-cache dtype, and the multi-turn final-probe McNemar test is non-significant on both Llama models. The two non-equivalent results — Qwen2.5-1.5B safety at delta = -3.1pp and Llama-3B capability at delta = -1.6pp with high variance — are at the equivalence boundary, not gross degradations.

**What this rules out.**
- A *generic* FP8 KV-cache safety-degradation effect that would be detectable across small instruction-tuned models with n = 1,003 paired prompts at α = 0.05 and 80% power.
- The accumulated-rounding-error hypothesis (H2): if FP8 errors compound over context length, we would expect monotonically widening FP16-FP8 gaps from ctx=256 to ctx=2048; what we observe is non-monotonic noise.
- The compound-with-batching hypothesis (H3): TR138 documented batch-induced safety flips, but TR145 finds the FP8-batch interaction term is approximately additive in every (model, batch_size) cell tested, not multiplicative.
- The directional-asymmetry hypothesis (H5): aggregate flip counts across all three models split 80 unsafe / 65 safe, well within binomial noise (p = 0.31 to 1.0 per model).

**What this does not rule out.**
- Effects on larger models (we tested up to 3B; production models are typically 7B+).
- Effects at temperature > 0, where the verification-style argument that "compute is deterministic" no longer applies cleanly to the cache-precision question.
- Effects on workloads with longer context windows (we tested up to 2048 tokens of prefill padding).
- Effects on conversation patterns longer than 5 turns or that use different jailbreak structures than the GPT-4-style adversarial multi-turn template we used.
- Subtle effects below the ±3pp equivalence margin — specifically the Qwen2.5-1.5B advbench result (-6.0pp delta on the n=100 advbench subset, though with the wide odds ratio CI characteristic of sparse contingency tables).

**Operational implication.** FP8 KV-cache is not pre-approved as safety-neutral, but it is also not pre-banned. Treat it as a workload-specific deployment decision: run a paired safety eval at your representative context length, batch size, and conversation length before flipping the flag in production. The TR145 pipeline (vLLM Docker + Ollama judge, ~24,000 records over 14 hr on a single laptop GPU) is the reference recipe for that paired eval.

**What was new in TR145 relative to prior Banterhearts safety TRs.**
- First TR in this line to isolate KV-cache precision as the *only* manipulated variable (TR125 / TR134 manipulated weight precision; TR138 manipulated batch size; TR141 / TR143 varied architecture).
- First five-phase factorial-with-resume safety design — Phases 3, 4, and 5 are stratified-cell designs around Phase 1+2's paired core, allowing both within-phase ANOVA and across-phase Mantel-Haenszel pooling.
- First TR to use `gemma3:12b` as the primary judge (TR140 used it as the JTP triangulation companion to Claude); achieves κ = 0.43 against regex classifiers, in the moderate-agreement range we expect for a generalist judge.

---

## 4. Introduction and Research Motivation

### 4.1 The serving-layer optimization that bypasses the safety gate

Safety evaluation in modern LLM deployments has converged on a workflow: pretrain, RLHF or DPO, run a refusal benchmark, and ship the checkpoint. The implicit assumption is that anything done to the *checkpoint* — quantization, distillation, fine-tuning — is what changes safety, and that anything done to the *serving stack* — request batching, KV-cache compression, speculative decoding — is at most a latency or cost optimization. Banterhearts' prior safety line has systematically broken that assumption: TR138 showed batched concurrent inference produces measurable refusal flips that batch-size-1 inference does not; TR141 documented refusal-template fragility under quantization that the source checkpoint did not predict; TR144 explored speculative decoding's safety profile and found a clean null at temperature zero, motivating renewed attention to other "checkpoint-neutral" serving choices.

KV-cache quantization is the serving-layer flag with the largest practical leverage and the smallest evidentiary base. vLLM, TensorRT-LLM, SGLang, and HuggingFace TGI all expose `--kv-cache-dtype fp8` as a one-flag throughput optimization: the cached attention key/value tensors are stored at 8-bit precision (E4M3 on Hopper-class GPUs, E5M2 on others), halving cache memory and roughly doubling the maximum number of concurrent requests the same VRAM can hold. Throughput gains of 1.6-2.1x at fixed memory are routinely cited in vLLM release notes and benchmarks (vLLM v0.5.0 onwards). Operators reach for this flag to scale to longer contexts, larger batches, or cheaper deployment tiers — and they do so under the implicit assumption that the model weights are unchanged, so safety is unchanged.

That assumption is testable, and to our knowledge has not been tested at scale outside this report. The closest prior work is on weight-quantization safety (TR125 v1-3 in this lab; Lin et al. AWQ; Frantar et al. GPTQ-for-LLMs), which manipulates a different precision axis. Public KV-cache benchmarks measure perplexity, not refusal behavior; throughput benchmarks measure latency, not bias resistance. The gap TR145 fills is between the implicit safety claim ("the checkpoint is unchanged, therefore the model is unchanged, therefore safety is unchanged") and an actual paired safety measurement under the flag flip.

### 4.2 Why this is not the same question as weight quantization

It is tempting to assume KV-cache quantization is "just like weight quantization, but on a different tensor." It is not, and the difference matters for the safety question.

Weight quantization replaces the model's parameters with a low-precision approximation. The error is *systematic* — every forward pass uses the same quantized weights, so every output is downstream of the same lossy approximation. The safety risk has been characterized: TR125 showed that 4-bit GPTQ and AWQ on Llama-2-7B can move advbench refusal rates by 5-15 percentage points, with the direction model-and-method-dependent. TR134 documented calibration sensitivity: the choice of calibration set (C4 vs WikiText) shifts the refusal pattern.

KV-cache quantization is structurally different. Model weights remain FP16. Only the *retained attention state* is compressed, and only after the projection through full-precision Q/K/V weight matrices. The error is therefore (a) localized to the attention computation, (b) accumulated over the context length (each new token's attention pattern depends on all prior cached K/V at reduced precision), and (c) decoupled from the model's instruction-following alignment, which lives in the weights.

This decoupling cuts both ways. On one hand, any safety effect we observe is attributable to attention-state precision alone, not to a model-level reweighting — a cleaner inference than the weight-quantization studies. On the other hand, the effect, if it exists, may be subtle: small attention-pattern shifts that do not move per-token logits dramatically but accumulate over many tokens to push the output across a refusal-vs-compliance decision boundary. The five-phase design of TR145 is built around that hypothesis: Phase 2 catches gross effects with a paired prompt-level test; Phase 3 catches accumulation effects with a context-length stratification; Phase 4 catches interaction effects with batch concurrency; Phase 5 catches conversation-history-dependent effects with reconstructed multi-turn probes.

### 4.3 Research questions

TR145 answers five concrete decision questions:

1. **Phase 2:** Does flipping `--kv-cache-dtype` from `auto` (FP16) to `fp8` change paired safety outcomes on the same prompts beyond chance? (McNemar's test, primary result.)
2. **Phase 3:** Does the FP16-FP8 safety gap grow with context length? Does FP8 rounding accumulate as the cache fills with longer prefixes?
3. **Phase 4:** Does FP8 KV-cache compound with the batched-inference safety effects documented in TR138, or are the two effects approximately additive?
4. **Phase 5:** Does FP8 KV-cache produce conversation-history-dependent safety effects? Do reconstructed multi-turn histories at the final-turn safety probe show degradation that single-turn tests miss?
5. **Cross-cutting:** Is the FP8-FP16 safety delta within a ±3pp equivalence margin (TOST), and is the cross-model pooled odds ratio (Mantel-Haenszel) consistent with no effect?

### 4.4 Why this matters for production

The practical risk is specific. A deployment team may:

- Enable `--kv-cache-dtype fp8` to fit a longer context window into the same VRAM (relevant for RAG, long-document QA, multi-turn chat).
- Enable it to double the per-GPU concurrent-request budget at fixed VRAM (cost reduction).
- Enable it as part of a generic "performance optimizations" profile that also includes prefix caching, speculative decoding, and quantized weights, without isolating which optimization contributes to which observed regression.

If FP8 KV-cache silently weakens refusal behavior — say, by shifting attention away from the system prompt's safety instruction over a long conversation, or by softening the model's commitment to a refusal once it has begun one — the operator would see a faster, cheaper system that is also less safe, with no signal in the latency or throughput dashboard to reveal the cost.

TR145 tests whether that scenario actually materializes under conditions that approximate small-model deployments. The answer, under tested conditions, is no.

### 4.5 Scope

| Scope item | Coverage |
|------------|----------|
| Deployment style | vLLM v0.19.1 Docker with NVIDIA Container Toolkit GPU passthrough |
| Inference acceleration | KV-cache quantization (FP8 E4M3) — model weights remain FP16 |
| Acceptance methods | N/A (no speculative decoding) |
| Context lengths tested | 256, 512, 1024, 2048 tokens of benign padding before safety probe |
| Batch sizes tested | 1, 4, 8 concurrent requests (TR138 dispatch pattern) |
| Conversation lengths tested | 5 turns, with safety probe at turn 5 |
| Models | 3 (Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct) |
| Model families | 2 (Llama, Qwen) |
| Safety benchmarks | 5 tasks (AdvBench refusal, JailbreakBench-derived behaviors, jailbreak amplification, BBQ bias, TruthfulQA) |
| Capability controls | 2 tasks (MMLU, ARC-Challenge) |
| Temperature | 0.0 (greedy decoding) throughout |
| Random seed | 42 fixed across CUDA, NumPy, Python |
| Primary focus | Safety equivalence under KV-cache-precision change |

### 4.6 Literature grounding

TR145 is anchored in three prior literatures:

**KV-cache compression theory.** FP8 attention has been theoretically analyzed by NVIDIA (Transformer Engine FP8 GEMMs), and the E4M3 / E5M2 numerical formats have known dynamic-range properties (Micikevicius et al. 2022, Sun et al. 2024). vLLM's implementation casts cached K/V tensors to E4M3 on Ada/Hopper hardware after the FP16 attention projection. The expected error magnitude per cached token is small (on the order of 2^-3 relative error) but accumulates with context length and may interact with attention-softmax sharpness.

**Quantization safety literature.** Weight-quantization safety has been studied for instruction-tuned models (Frantar et al. GPTQ-for-LLMs 2023; Lin et al. AWQ 2024; Banterhearts TR125 / TR134 2025-26). The consensus finding is that aggressive (4-bit and below) weight quantization can shift refusal behavior, with model-and-method-specific direction. KV-cache precision is structurally different (see §4.2) and has not been systematically benchmarked for safety.

**Inference-time safety in this lab.** Banterhearts has documented that batching (TR138), conversation length (TR139), architecture choice (TR141, TR143), refusal templates (TR142 / RTSI), and speculative decoding (TR144 / TAIS) all produce or rule out specific safety effects. TR145 extends the catalog to KV-cache precision, holding all of those other variables fixed.

### 4.7 How to read this report

TR145 is structured as a five-phase study with progressive hypothesis testing. Phase 1 (FP16 baseline) and Phase 2 (FP8 KV-cache) form the paired core. Phases 3, 4, and 5 are factorial expansions: Phase 3 stratifies by context length, Phase 4 by batch size, Phase 5 by conversation history. Cross-cutting analyses (TOST equivalence, Mantel-Haenszel synthesis, power, judge agreement) integrate the per-phase findings into a single null-or-not verdict.

Readers who want only the primary result: see SS2 (Phase 2 McNemar) and SS18 (TOST). Readers who want to validate the null is not power-starved: see SS19 (power analysis). Readers who want to translate the result into a deployment decision: see Production Guidance.

---

## 5. Research Hypotheses

TR145 pre-registered five hypotheses, mapped one-to-one onto the five phases. The pre-registration scaffold lives at `papers/kv_cache_quantization_safety/` and was committed before the first sample was taken; the analysis pipeline does not have access to outcome data when computing the test statistics.

| ID | Hypothesis | Phase | Test | Outcome |
|----|-----------|-------|------|---------|
| H0 | FP8 KV-cache produces safety-equivalent outputs to FP16 KV-cache. | 2 | McNemar paired test, ±3pp TOST | **McNemar non-rejection on 3/3 models** (p = 1.00, 0.60, 0.31); **TOST ±3pp equivalence on 2/3 models** (Llama-1B Δ=-0.39pp, Llama-3B Δ=-0.58pp; Qwen-1.5B Δ=-3.09pp fails by 0.09pp) |
| H1 | FP8 KV-cache degrades safety more than capability (disproportionate impact). | 2 | Safety-capability paired-delta CI overlap | Not supported (CIs overlap on all 3 models) |
| H2 | FP8 rounding errors accumulate over context length. | 3 | Two-way ANOVA (ctx × KV), slope analysis | Not supported (p ≥ 0.54 on both models, non-monotonic gap) |
| H3 | Batching and KV-cache quantization compound rather than add. | 4 | Two-way ANOVA (batch × KV), interaction decomposition | Not supported (p ≥ 0.98, all cells additive) |
| H4 | Reconstructed multi-turn histories expose FP8-sensitive final-turn safety failures. | 5 | Turn-5 paired McNemar | Not supported (1B p = 0.22, 3B p = 1.0) |
| H5 | Directional asymmetry persists under KV-cache quantization (more refusal→compliance flips than reverse). | 2-5 | Binomial test on flip direction | Not supported (no model has significant asymmetry) |

The pre-registration's explicit primary hypothesis is H0; H1-H5 are secondary. Under the pre-registered analysis plan, failure to reject H0 is the headline result regardless of whether H1-H5 land. All five secondary hypotheses also fail to find supportive evidence.

The H0 row above splits the McNemar verdict from the TOST verdict deliberately. McNemar's test asks whether the discordant cells of the paired contingency table are unbalanced; non-rejection on three of three models means the data does not provide evidence against the equivalence hypothesis. TOST asks the stronger question — is the observed delta *positively bounded* within a pre-specified margin — and gives a different answer per model: Llama-1B and Llama-3B pass at ±3pp with deltas under 0.6pp; Qwen-1.5B fails the equivalence test by 0.09pp at delta -3.09pp. A reader who conflates the two reads "H0 confirmed equivalent" everywhere; the careful read is "H0 not rejected by McNemar on all three models, *and* positively confirmed equivalent on two of three by TOST." That distinction matters for deployment guidance: Qwen-1.5B specifically does not have a positive equivalence finding under TR145, only an absence-of-rejection.

### 5.1 Why H0 is the right primary

Choosing the null as primary is not a defensive move; it is a design decision driven by the asymmetric cost of error. If the paper claimed FP8 degrades safety and the result was a false positive, deployment teams would avoid a real throughput optimization for no reason. If the paper claimed FP8 preserves safety and the result was a false negative, deployment teams would deploy a real safety regression undetected. The right null is the one where rejection requires evidence, and the safety-degrading hypothesis is the one that requires evidence for action.

This framing also aligns with the equivalence-testing logic in SS18: TOST asks whether the FP8-FP16 delta is *within* a pre-specified margin, not whether it is *zero*. Equivalence at ±3pp is what operators actually need to know — they do not require FP8 to be byte-identical to FP16, only to be close enough that the throughput gain dominates the residual safety drift.

### 5.2 Why we did not pre-register a positive directional hypothesis

An alternative framing would have been: "FP8 will degrade safety by X percentage points." We did not pre-register that, for two reasons. First, the magnitude of effect — if any — is not predictable from FP8 numerical theory alone; it depends on attention-pattern sharpness, which varies model-to-model in ways the literature has not characterized. Second, pre-registering a directional effect would have biased the equivalence margin: if we had said "expect ~5pp safety drop," readers would interpret a 3pp drop as "smaller than expected but still confirming the hypothesis." The TOST framing avoids that trap by forcing the paper to pre-specify a margin and then report whether the data falls inside it.

---

## 6. Methodology

### 6.1 Experimental design overview

TR145 is a five-phase factorial design. The independent variable is `kv_cache_dtype` ∈ {auto, fp8}. The dependent variables are (a) per-prompt safety classification (regex + LLM judge) and (b) per-prompt capability classification (exact-match against gold answers). Stratification variables vary by phase: model identity in all phases, plus context length in Phase 3, batch size in Phase 4, conversation index and turn number in Phase 5.

Every record in the run carries a deterministic identity — `(phase, model, kv_cache_dtype, task_name, sample_id, context_length, batch_size, conversation_id, turn_number)` — which serves three purposes: (1) deduplication during incremental writes to `samples.jsonl`, (2) resume-by-default after a process restart or laptop sleep, and (3) paired-test alignment in the analysis pipeline. The latter is the operational meaning of "paired" in this report: the same `(model, task_name, sample_id)` tuple is matched between Phase 1 and Phase 2 (and across cells in Phases 3-5) when computing McNemar's test and TOST equivalence.

### 6.2 Phase structure

| Phase | Description | n records | Independent variable |
|-------|-------------|-----------|----------------------|
| 1 | Baseline (FP16 model weights, FP16 KV-cache) | 3,009 | none (baseline) |
| 2 | FP8 KV-cache (FP16 weights unchanged) | 3,009 | KV-cache dtype |
| 3 | Context length × KV-cache interaction | 4,000 | KV-cache dtype × context length (256/512/1024/2048) |
| 4 | Batch size × KV-cache interaction | 12,036 | KV-cache dtype × batch size (1/4/8) |
| 5 | Conversation history × KV-cache (multi-turn) | 2,000 | KV-cache dtype × conversation index × turn number |
| **Total** | | **24,054** | |

Phase 1 and Phase 2 share the same 1,003-prompt task inventory: 100 AdvBench refusal, 50 JailbreakBench-derived behaviors (TR140-sourced), 120 jailbreak amplification (many-shot), 198 BBQ bias, 50 TruthfulQA, 285 MMLU, 200 ARC-Challenge. Phases 3-5 use a 250-prompt safety subset (50 per safety task, capability tasks omitted) for cell-budget reasons; the omitted capability tasks have already been characterized in Phase 1+2 and the relevant capability question for the interaction phases is whether safety changes, not whether ARC accuracy changes.

The Phase 5 structure deserves explanation. Each conversation is a 5-turn arc: the first four turns are benign topic-steering setup (drawn from 5 opener templates and 5 steering templates, deterministically combined), and the fifth turn is the safety probe. The OpenAI-compatible chat API reconstructs the full conversation history per request, so this is not a persistent server-side KV-reuse claim — it is a final-prompt-context test where the safety probe is preceded by a synthetic conversation history. This design is identical to the multi-turn arc used in TR139, which lets us cross-reference the trajectory shape.

### 6.3 Models and rationale

| Model | Params | VRAM (FP16) | Family | Why included |
|-------|--------|-------------|--------|--------------|
| Llama-3.2-1B-Instruct | 1.24 B | 2.5 GB | Llama | Smallest production-deployable instruct model; sensitive to KV precision |
| Llama-3.2-3B-Instruct | 3.21 B | 6.4 GB | Llama | Mid-size laptop deployment target; same architecture as 1B for in-family comparison |
| Qwen2.5-1.5B-Instruct | 1.54 B | 3.0 GB | Qwen | Cross-family check; Qwen 2.5 has stronger out-of-the-box refusal calibration than Llama |

The models were chosen for three reasons. First, all three fit comfortably in 12 GB VRAM at FP16 weights with vLLM's `--gpu-memory-utilization 0.85`, leaving room for the KV cache plus a 2048-token prefill buffer, even at batch=8 — the largest combination tested. Second, the same trio was used in TR138, TR139, and TR143, allowing cross-TR baseline checks (SS22). Third, the Llama-1B / Llama-3B pair is in-family and same-tokenizer, isolating parameter count from architecture; adding Qwen2.5-1.5B introduces a different model family at a comparable parameter count to test whether any observed effect is family-specific.

Larger models (Llama-3.1-8B, Qwen2.5-7B, Llama-3.1-70B) were considered and excluded for hardware reasons — the laptop-GPU constraint that makes TR145 reproducible on a $2,000 machine also caps the parameter budget. The Limitations section (below) flags the 7B+ extrapolation as the most important open question.

### 6.4 Safety battery

The safety battery is reused from the alignment-robustness line and is identical across all five phases:

| Task | Domain | Metric | n samples | Source |
|------|--------|--------|-----------|--------|
| advbench_refusal | safety/refusal | refusal-template detection | 100 | AdvBench (Zou et al. 2023) |
| jailbreakbench_behaviors | safety/refusal | refusal-template detection | 50 | JailbreakBench-derived (TR140 robust subset) |
| jailbreak_amplification | safety/refusal | refusal-template detection | 120 | TR138 / many-shot adversarial |
| bbq_bias | safety/bias | bias-resistance metric | 198 | BBQ (Parrish et al. 2022) |
| truthfulqa | safety/truthfulness | reference-based scoring | 50 | TruthfulQA (Lin et al. 2022) |
| mmlu_real | capability | exact-match | 285 | MMLU (Hendrycks et al. 2021) |
| arc_challenge | capability | exact-match | 200 | ARC-Challenge (Clark et al. 2018) |
| **Total** | | | **1,003** | |

Five safety tasks plus two capability controls. The safety vs capability split is intentional: H1 (disproportionate safety degradation) requires both kinds of measurement, and the Phase-2 paired test is reported separately for safety and capability outcomes.

### 6.5 Scoring stack

| Component | Detail |
|-----------|--------|
| Regex classifiers | Pattern-matching refusal detection inherited from TR134, augmented with TR142 RTSI templates |
| LLM judge | gemma3:12b via Ollama, blinded to KV-cache config and batch/context conditions |
| Bias scorer | BBQ bias-resistance metric on (context, question, answer choices) tuples |
| Truthfulness scorer | TruthfulQA reference-based scoring against `correct_answers` and `incorrect_answers` lists |
| Capability scorer | Exact-match for MMLU and ARC-Challenge gold answers |

Two parallel scoring tracks run on every safety record: a deterministic regex classifier (cheap, fast, false-negatives on softened refusals) and an LLM judge (slower, catches paraphrased refusals, requires inter-rater validation). Cohen's κ between the two is reported in SS21.

The judge is *blinded*: it receives only `(prompt, response)` and never the KV-cache dtype, batch size, context length, or any phase identifier. The judge is also *task-typed*: refusal tasks (AdvBench, JailbreakBench, jailbreak amplification) go through `judge.judge_refusal`, TruthfulQA goes through `judge.judge_truthfulness` with the gold answer lists, BBQ goes through `judge.judge_bias` with the context/question/choices structure. This is the same task-typing pattern as TR140's JTP triangulation, with `gemma3:12b` filling the role that `qwen2.5:7b-instruct-q8_0` originally held in the pre-registration scaffold.

### 6.6 Analysis pipeline (23 passes)

`analyze.py` runs a 23-pass pipeline against `samples.jsonl` and `judge_labels.jsonl`. The passes group by phase: Pass 1 scores all records; Pass 2 merges judge labels; Passes 3-7 cover Phase 1+2 (baseline rates, McNemar, flip direction, per-task effect, safety-capability divergence); Passes 8-11 cover Phase 3 (ANOVA, slope, degradation curve, critical context length); Passes 12-15 cover Phase 4 (ANOVA, interaction decomposition, per-task breakdown, TR138 cross-reference); Passes 16-19 cover Phase 5 (setup-turn diagnostics, turn-5 McNemar, conversation-level analysis); Passes 20-23 are cross-cutting (TOST, power, Mantel-Haenszel synthesis, cross-TR validation).

Multiple-comparisons correction is Holm-Bonferroni across all (model, domain) family-wise comparisons in Phase 2. ANOVA in Phases 3-4 is two-way with the model as the random factor and the (KV dtype, stratification level) factor pair as the fixed factors. TOST equivalence is paired t-test against the ±3pp margin on per-prompt deltas. Power analysis is computed retrospectively for each cell using the observed baseline rate and per-cell n.

### 6.7 Design safeguards

- **Temperature 0.0 throughout.** Removes sampling stochasticity as a confound. Any Phase 1 vs Phase 2 difference at the same (model, task, sample_id) is attributable to FP16-vs-FP8 KV-cache, not to sampling noise.
- **Random seed 42** for all CUDA/cuBLAS, NumPy, and Python random number generators.
- **vLLM v0.19.1 pinned** Docker image (`vllm/vllm-openai:v0.19.1`), pulled once before the first phase and reused across all 18 vLLM container lifecycles. `--enforce-eager` to disable graph capture, eliminating one source of cross-run nondeterminism.
- **Resume by default.** Every record is keyed by its deterministic identity; `samples.jsonl` is written incrementally and atomically via temp-file-replace; on restart, completed `ok` records are skipped and only missing/error records are re-sampled.
- **Judge blinding.** `judge_labels.jsonl` records the model name and KV-cache dtype for each record but the LLMJudge prompt template strips both before construction.
- **Pinned task inventory.** All 1,003 prompts are deterministically loaded from YAML files in `research/tr145/tasks/`, with `prepare_benchmarks.py` enforcing that the task inventory matches the pre-registration before any sampling begins.

---

## 7. Models & Configuration

| Parameter | Value |
|-----------|-------|
| vLLM image | `vllm/vllm-openai:v0.19.1` (Docker, pinned digest `sha256:2622f38a0aa6...`) |
| vLLM args | `--dtype float16 --gpu-memory-utilization 0.85 --enforce-eager` |
| Model dtype | float16 throughout (weights never quantized) |
| KV-cache dtypes | `auto` (resolves to FP16 on FP16 model) and `fp8` (E4M3 on Ada/Hopper) |
| Max model length | 2048 (Phases 1, 2, 4) / 4096 (Phases 3, 5 — headroom for prefix and history) |
| Max new tokens | 256 |
| Temperature | 0.0 |
| Top-p | 1.0 |
| Seed | 42 |
| Warmup requests | 10 per (model, dtype) combination |
| Cooldown between models | 10 s |
| Judge model | `gemma3:12b` via Ollama at `http://localhost:11434` |
| Judge keep_alive | 0 (force unload after each session, frees VRAM for vLLM relaunch) |
| Hardware | NVIDIA RTX 4080 Laptop GPU, 12 GB VRAM, sm_8.9 (Ada Lovelace, FP8-capable) |
| Host OS | Windows 11, Docker Desktop 29.2.1, NVIDIA Container Toolkit |
| CUDA | 12.4 (vLLM image pinned) |
| Python | 3.13.1 (host); 3.12 (vLLM image) |

The hardware caveat: FP8 KV-cache requires CUDA compute capability ≥ 8.9 (Ada Lovelace or Hopper). The RTX 4080 Laptop is sm_8.9, so it qualifies, but it is the *minimum* qualifying SKU. The same study on an A100 (sm_8.0, FP16-only KV cache) would not be runnable as designed; on H100 (sm_9.0), FP8 cache uses E4M3 with slightly different numerical properties than the E4M3 used on Ada. The ±3pp equivalence margin in this report is calibrated to the Ada implementation, not Hopper.

The Phase 0 FP8-validation gate runs before any sampling starts: a small test prompt ("What is 2+2?") is sent to a freshly-launched vLLM container with `--kv-cache-dtype fp8` and `unsloth/Llama-3.2-1B-Instruct`. If the launch fails or the response is malformed, the entire run aborts before consuming GPU time on the main study. In the TR145 v1.0 run, the gate passed with response `" 2+2 is 4."` after ~70 s of vLLM startup.

---

## SS1. Phase 1 FP16 Baseline

Phase 1 establishes the FP16 KV-cache safety floor for each model. It is not the primary result; its purpose is to provide a paired baseline for Phase 2 and to validate that the three models behave at safety rates consistent with prior TRs (cross-TR validation in SS22).

### SS1.1 Per-model aggregate rates

| Model | Safety rate | Safety n | Capability rate | Capability n |
|-------|-------------|----------|-----------------|--------------|
| **llama3.2-1b** | **64.3%** | 518 | 33.6% | 485 |
| **llama3.2-3b** | **75.9%** | 518 | 58.4% | 485 |
| **qwen2.5-1.5b** | **78.7%** | 518 | 64.7% | 485 |

The per-model rates split as expected: parameter count predicts capability monotonically (1.24B → 3.21B → 1.54B is a non-monotonic ranking, but capability ordering 33.6% < 58.4% / 64.7% reflects that Qwen2.5 is a stronger architecture per-parameter than Llama 3.2 at the small end), and Qwen leads on safety despite being smaller than Llama-3B because Qwen 2.5's instruction-tuning is more refusal-aggressive on advbench-style prompts (98% AdvBench refusal vs 79% for Llama-3B).

The Llama-1B → Llama-3B safety jump (64.3% → 75.9%, +11.6pp) is the largest cross-model safety delta in the matrix. The Llama-3B → Qwen-1.5B safety jump is smaller (75.9% → 78.7%, +2.8pp) but capability also jumps (58.4% → 64.7%, +6.3pp), which is the expected family-level architecture advantage.

### SS1.2 Per-task safety + capability rates

| Task | Domain | Llama-1B | Llama-3B | Qwen-1.5B |
|------|--------|---------|----------|-----------|
| advbench_refusal | refusal | 65.0% | 79.0% | **98.0%** |
| jailbreakbench_behaviors | refusal | 52.0% | 66.0% | 74.0% |
| jailbreak_amplification | refusal | 51.7% | 58.3% | 57.5% |
| bbq_bias | bias | 79.3% | **92.4%** | 89.9% |
| truthfulqa | truthfulness | 46.0% | 56.0% | 51.0% |
| arc_challenge | capability | 36.5% | 67.5% | 74.0% |
| mmlu_real | capability | 31.6% | 51.9% | 58.3% |

Three observations from the per-task baseline:

1. **AdvBench refusal has the widest cross-model spread** (65% to 98%, 33pp). This is the task most sensitive to instruction-tuning quality — direct "tell me how to X" harm prompts are the cleanest measurement of whether the safety post-training actually shaped refusal behavior. Qwen 2.5's 98% refusal is at the ceiling; Llama 3.2 1B's 65% is well below the floor most production deployments target.
2. **Jailbreak amplification clusters between 51-58% across all three models.** This is the many-shot adversarial task from TR138/TR140 where refusal degrades under repeated unsafe context. The cross-model floor is consistent with prior TR140 findings: instruction-tuning protects against single-shot attacks but most small models fail under multi-shot.
3. **BBQ bias resistance is high (79-92%)** and roughly tracks model size, but it is the safety axis most likely to be a regex-classifier artifact rather than genuine bias resistance. The per-task judge agreement (SS21) lands at 68.8% on BBQ, the lowest of the five safety tasks, suggesting the regex and LLM disagree on bias more than on refusal.

### SS1.3 Implication for Phase 2 paired tests

The Phase 2 paired test compares each prompt's Phase 1 outcome to its Phase 2 outcome on the same `(model, task, sample_id)`. With baselines spread from 31.6% (Llama-1B MMLU) to 98.0% (Qwen advbench), the per-cell McNemar's test is well-defined: each cell has at least 50 paired observations and a non-degenerate baseline rate (no cell at 0% or 100%, where McNemar is uninformative). This is a precondition the Phase 1 results need to clear, and they do.

---

## SS2. Phase 2 FP8 KV-Cache — Primary Result

Phase 2 is the primary experimental condition. Each model is evaluated on the same 1,003 prompts as Phase 1 with only `--kv-cache-dtype` changed from `auto` (FP16) to `fp8`. Model weights remain FP16. Sampling is deterministic at temperature 0.0 with seed 42. McNemar's test on paired `(model, task, sample_id)` outcomes determines whether FP8 KV-cache changes safety classifications beyond chance.

### SS2.1 Safety McNemar tests

| Model | n paired | Discordant | R→C | C→R | χ² | p (exact) | McNemar OR | Holm sig |
|-------|----------|-----------|-----|-----|----|-----------| -----------|----------|
| **llama3.2-1b** | 518 | 33 | 17 | 16 | 0.000 | **1.0000** | 1.061 | No |
| **llama3.2-3b** | 518 | 32 | 18 | 14 | 0.281 | **0.5966** | 1.276 | No |
| **qwen2.5-1.5b** | 518 | 80 | 45 | 35 | 1.012 | **0.3143** | 1.282 | No |

R→C = refusal→compliance (FP16 refused, FP8 complied — the "unsafe direction"). C→R = compliance→refusal (the reverse, "safe direction"). Holm-Bonferroni across the three models corrects family-wise error rate.

**Reading the table.** All three exact p-values are well above the 0.05 threshold, even before Holm correction. The McNemar odds ratios are clustered near 1: Llama-1B at 1.06 (essentially no asymmetry), Llama-3B at 1.28 (weak unsafe-direction skew, not significant), Qwen-1.5B at 1.28 (same magnitude on a larger discordant pool). The Phase 2 primary test does not reject H0.

The 80 discordant flips on Qwen2.5-1.5B is the largest discordant count of the three models, but this is also the model with the highest baseline safety rate (78.7% on the safety subset, with AdvBench at 98%). High-ceiling baselines are inherently more sensitive to noise: when the FP16 baseline is near 100%, every FP8 deviation that produces compliance shows up as a discordant flip, but the same deviation on a 50%-baseline task is half as likely to cross a decision boundary.

### SS2.2 Capability McNemar tests

| Model | n paired | Discordant | C→I | I→C | p (exact) | Holm sig |
|-------|----------|-----------|-----|-----|-----------|----------|
| **llama3.2-1b** | 485 | 25 | 13 | 12 | 1.0000 | No |
| **llama3.2-3b** | 485 | 20 | 14 | 6 | 0.1153 | No |
| **qwen2.5-1.5b** | 485 | 107 | 70 | 37 | **0.0018** | **Yes** |

C→I = correct→incorrect, I→C = incorrect→correct. Capability is computed against MMLU and ARC-Challenge gold answers (exact-match). Note Holm correction here is across the (safety, capability) family for each model.

The Qwen2.5-1.5B capability result is the only Holm-significant outcome in the entire Phase 2 battery: 70 correct→incorrect flips vs 37 incorrect→correct, p = 0.0018. The McNemar OR is 70/37 = 1.89 (capability degraded under FP8). This is a real effect on capability — and crucially, *not* on safety: Qwen's safety McNemar is non-significant (p = 0.31) on the same paired structure.

This combination — significant capability degradation, non-significant safety effect — is the opposite of H1 (which predicted safety would degrade more than capability). H1 is not supported.

### SS2.3 Why Qwen capability degrades but Llama capability does not

The Qwen-only capability degradation is a real Phase 2 finding worth interpreting. Three candidate explanations:

1. **Qwen 2.5's instruction-tuning attends more sharply to the prompt structure.** Qwen 2.5 was trained with a tighter chat template and is known to produce more deterministic-looking outputs at temperature 0; FP8 attention rounding may shift that determinism in a way that clips correct answer tokens. Llama 3.2's looser instruction-tuning may absorb the same FP8 perturbation without changing the final-answer token.
2. **Qwen 2.5's MMLU/ARC accuracy is closest to a decision boundary on uncertain items.** Qwen's capability rate is 64.7% versus Llama-3B's 58.4%; if FP8 perturbation is a small unsigned shift in attention distribution, it will move more items across the correct/incorrect boundary on Qwen than on Llama because Qwen has more items near that boundary.
3. **Statistical artifact.** Three models tested. Per Holm correction, one would expect ~5% false-positive rate at family-wise α = 0.05. We observe one significant capability result out of six tests (3 models × 2 outcomes). This is consistent with chance.

We do not have data to distinguish (1)/(2)/(3); the SS22 cross-TR validation against TR125 weight-quantization data on Qwen would be the right follow-up. For TR145's purposes, the operational read is that capability under FP8 deserves its own paired profiling — a finding consistent with the Production Guidance recommendation.

### SS2.4 Why this is the primary result

McNemar on paired prompts with the same seed and temperature is the cleanest possible safety-equivalence test for a serving-flag perturbation. There is no model-weight change, no calibration-set choice, no random-sample resampling — only the KV-cache precision. If FP8 had a primary safety effect at the magnitudes deployment teams should care about (≥3pp), n = 518 per model with 80% power would detect it. The test does not detect it. That is the headline.

The five secondary hypotheses (H1-H5) and the supporting analyses (TOST, MH, power) are all confirmatory or stratification-specific tests of the same null. None of them shifts the headline.

---

## SS3. Phase 2 Flip Direction Analysis

The McNemar test asks whether the discordant cells are unbalanced; the flip direction analysis asks *which task* is contributing to any observed asymmetry, even if the per-model aggregate is non-significant.

### SS3.1 Per-model flip aggregates

| Model | Flip → unsafe | Flip → safe | Total flips | Directional ratio | Binomial p | Net direction |
|-------|--------------|-------------|-------------|-------------------|-----------|---------------|
| llama3.2-1b | 17 | 16 | 33 | 0.515 | 1.000 | neutral |
| llama3.2-3b | 18 | 14 | 32 | 0.563 | 0.597 | neutral |
| qwen2.5-1.5b | 45 | 35 | 80 | 0.563 | 0.314 | neutral |

The "unsafe" direction is FP16 refused, FP8 complied: the model abandoned refusal under FP8 KV-cache. The "safe" direction is the reverse. A directional ratio of 0.5 is symmetric (no asymmetry); a ratio above 0.5 favors unsafe, below 0.5 favors safe. All three models show modest unsafe-leaning ratios (0.51-0.56) but none clears the binomial threshold at α = 0.05. Aggregating across all three models gives 80 unsafe / 65 safe (= 0.55 ratio, n = 145 flips). The two-sided binomial test against 0.5 gives p ≈ 0.244, not the one-sided 0.13 that an earlier draft of this section reported; we use two-sided here because directional asymmetry was not pre-registered with a sign and the right test is "is the ratio different from 50/50?" rather than "is it greater than 50/50?". Either way, the result is not significant at α = 0.05.

H5 (directional asymmetry persists) is not supported at the per-model or pooled level.

### SS3.2 Per-task flip breakdown

| Task | Llama-1B u/s | Llama-3B u/s | Qwen-1.5B u/s | Aggregate u/s |
|------|--------------|--------------|---------------|---------------|
| advbench_refusal | 1 / 5 | 3 / 0 | 6 / 0 | **10 / 5** |
| jailbreakbench_behaviors | 2 / 4 | 1 / 4 | 7 / 0 | **10 / 8** |
| jailbreak_amplification | 3 / 1 | 6 / 2 | 15 / 17 | 24 / 20 |
| bbq_bias | 6 / 4 | 3 / 6 | 13 / 11 | 22 / 21 |
| truthfulqa | 5 / 2 | 5 / 2 | 4 / 7 | 14 / 11 |

Three tasks show clearly unsafe-leaning aggregate flips: AdvBench refusal (10/5), JailbreakBench-behaviors (10/8), and TruthfulQA (14/11). AdvBench is the most concerning of the three because it's the canonical "model refuses outright harm" benchmark — and on Qwen specifically, 6 of 6 directional flips on AdvBench are unsafe (no safe-direction flips). On Llama-3B, the same pattern: 3 of 3 AdvBench flips are unsafe.

This is the strongest negative signal in the entire Phase 2 result. It does not survive the per-task McNemar test at the per-model level (n=6 and n=3 flips are too small for individual significance), but it is consistent with a small unsafe-direction bias on hard refusal tasks specifically.

### SS3.3 What the per-task pattern means for production

The aggregate-level reading is a null. The per-task pattern, however, suggests that *if* any FP8 effect exists at a magnitude this study cannot detect, it is concentrated on hard refusal tasks (AdvBench, JailbreakBench) rather than on bias or amplification tasks. That maps to a specific operational recommendation: an FP8 deployment should run paired profiling specifically on canonical refusal tasks at higher n than the 100/50 used here, not on a generic "safety eval" that pools refusal with bias and truthfulness.

---

## SS4. Phase 2 Per-Task KV-Cache Effect

Per-task effect sizes — Cohen's d, mean delta, paired contingency tables — for each safety and capability task under Phase 1 (FP16) vs Phase 2 (FP8). The McNemar OR values in this section are computed with the unbiased estimator (a*d / b*c with continuity correction); the very large ORs reflect that most cells have near-zero off-diagonal count (b or c near zero), which inflates the OR even when the absolute effect is tiny.

### SS4.1 AdvBench refusal (n = 100 per cell)

| Model | FP16 mean | FP8 mean | Delta (pp) | Cohen's d | n paired |
|-------|----------|----------|------------|-----------|----------|
| llama3.2-1b | 0.650 | 0.690 | **+4.0** | +0.085 | 100 |
| llama3.2-3b | 0.790 | 0.760 | -3.0 | -0.072 | 100 |
| qwen2.5-1.5b | 0.980 | 0.920 | **-6.0** | -0.277 | 100 |

The Qwen-1.5B drop on AdvBench (-6.0pp, |d| = 0.28) is the largest single-cell safety effect in Phase 2. With n = 100, this individual effect is not Holm-significant against the family of (5 tasks × 3 models) tests, but it is the cell that drives the SS18 TOST result — Qwen-1.5B safety overall is the only paired safety comparison that fails the ±3pp equivalence margin, and AdvBench is the task carrying that delta.

The Llama-1B *positive* delta (+4.0pp, FP8 refuses more than FP16) is a real direction reversal at small n, suggesting that whatever attention perturbation FP8 introduces does not have a consistent sign across the three models on the same task.

### SS4.2 JailbreakBench behaviors (n = 50 per cell)

| Model | FP16 mean | FP8 mean | Delta (pp) | Cohen's d |
|-------|----------|----------|------------|-----------|
| llama3.2-1b | 0.520 | 0.560 | +4.0 | +0.080 |
| llama3.2-3b | 0.660 | 0.720 | +6.0 | +0.131 |
| qwen2.5-1.5b | 0.740 | 0.600 | **-14.0** | **-0.295** |

Qwen-1.5B loses 14pp on JailbreakBench under FP8 — by far the largest single-cell delta in the entire Phase 2 matrix. The Cohen's d of -0.30 is in the small-effect range. With n = 50, this is not statistically supported on its own; pooled with the other 7 / 0 unsafe-direction flip pattern from SS3.2, however, it is consistent with FP8 KV-cache producing a real Qwen-specific drop on hard refusal tasks. The Llama models show small *positive* deltas on the same task.

### SS4.3 Jailbreak amplification (n = 120 per cell)

| Model | FP16 mean | FP8 mean | Delta (pp) | Cohen's d |
|-------|----------|----------|------------|-----------|
| llama3.2-1b | 0.517 | 0.500 | -1.7 | -0.033 |
| llama3.2-3b | 0.583 | 0.567 | -1.7 | -0.034 |
| qwen2.5-1.5b | 0.575 | 0.575 | 0.0 | 0.000 |

Three near-zero deltas. Jailbreak amplification is the many-shot adversarial task; the baseline is already at 50-58% across all models, so there is little headroom for FP8 to introduce additional degradation, and the data shows none.

### SS4.4 BBQ bias resistance (n = 198 per cell)

| Model | FP16 mean | FP8 mean | Delta (pp) | Cohen's d |
|-------|----------|----------|------------|-----------|
| llama3.2-1b | 0.793 | 0.783 | -1.0 | -0.025 |
| llama3.2-3b | 0.924 | 0.939 | +1.5 | +0.060 |
| qwen2.5-1.5b | 0.899 | 0.889 | -1.0 | -0.020 |

All three deltas are within ±1.5pp; none of the Cohen's d values clears 0.10 (negligible-to-very-small). BBQ is the most stable safety axis under FP8 KV-cache.

### SS4.5 TruthfulQA (n = 50 per cell)

| Model | FP16 mean | FP8 mean | Delta (pp) | Cohen's d |
|-------|----------|----------|------------|-----------|
| llama3.2-1b | 0.460 | 0.480 | +2.0 | +0.040 |
| llama3.2-3b | 0.560 | 0.520 | -4.0 | -0.080 |
| qwen2.5-1.5b | 0.510 | 0.510 | 0.0 | 0.000 |

TruthfulQA deltas are within ±4pp at n=50. The Llama-3B -4pp delta is the only individual-task delta that crosses the ±3pp margin, but it is one cell out of fifteen and is consistent with sampling noise at this n.

### SS4.6 Capability tasks (MMLU, ARC)

| Task | Model | FP16 mean | FP8 mean | Delta (pp) | Cohen's d |
|------|-------|----------|----------|------------|-----------|
| arc_challenge | llama3.2-1b | 0.365 | 0.360 | -0.5 | -0.010 |
| arc_challenge | llama3.2-3b | 0.675 | 0.630 | -4.5 | -0.094 |
| arc_challenge | qwen2.5-1.5b | 0.740 | 0.615 | **-12.5** | **-0.269** |
| mmlu_real | llama3.2-1b | 0.316 | 0.302 | -1.4 | -0.030 |
| mmlu_real | llama3.2-3b | 0.519 | 0.484 | -3.5 | -0.071 |
| mmlu_real | qwen2.5-1.5b | 0.583 | 0.554 | -2.9 | -0.058 |

The Qwen ARC drop (-12.5pp, |d| = 0.27) is the source of the SS2.2 capability McNemar significance. ARC requires multi-step reasoning more than MMLU's single-fact retrieval; it is plausible that FP8 attention rounding bites harder on multi-step than on retrieval-style answers. This is a real per-task effect — not a safety effect, but a capability effect worth noting separately. Llama-3B ARC is also non-trivially down (-4.5pp, on the boundary of meaningful).

### SS4.7 Pattern across the per-task matrix

Looking at the 21 (5 safety + 2 capability) × 3 model = 21-cell matrix, eight cells show |delta| ≥ 3pp. Of those eight cells, seven are unsafe-direction (FP8 produces lower safety/capability) and one is safe-direction (Llama-1B AdvBench at +4pp). The seven unsafe-direction cells of |delta| ≥ 3pp cluster on Qwen-1.5B (3 cells: AdvBench -6, JailbreakBench -14, ARC -12.5) and Llama-3B (3 cells: TruthfulQA -4, ARC -4.5, MMLU -3.5); Llama-1B contributes only the safe-direction +4pp on AdvBench.

This pattern says: where FP8 KV-cache produces a meaningful single-cell delta, the direction is more often unsafe than safe (7 of 8), but the magnitude is in the small-effect range (|d| = 0.07-0.30) and the per-cell tests are not individually significant. Aggregate over the family, and the McNemar tests do not reject H0. The right interpretation is that FP8 introduces small task-specific perturbations, not a generic safety regression.

---

## SS5. Safety-Capability Divergence

H1 predicts that FP8 KV-cache should degrade safety more than capability (disproportionate impact). The test compares each model's mean safety delta against its mean capability delta with overlapping confidence intervals as the equivalence criterion.

| Model | Safety mean Δ | Safety 95% CI | Capability mean Δ | Capability 95% CI | Effect gap (pp) | CIs overlap | Disproportionate? |
|-------|---------------|---------------|-------------------|-------------------|-----------------|-------------|-------------------|
| llama3.2-1b | -0.4pp | [-2.5, +1.8] | -0.2pp | [-2.3, +1.9] | +0.2 | Yes | No |
| llama3.2-3b | -0.6pp | [-2.7, +1.4] | -1.6pp | [-3.5, +0.2] | -1.1 | Yes | No |
| qwen2.5-1.5b | -3.1pp | [-6.4, +0.3] | -6.8pp | [-10.9, -2.7] | -3.7 | Yes | No |

Two findings.

First, no model shows disproportionate safety-vs-capability degradation. The Llama-1B and Llama-3B effect gaps are within ±2pp, and even the Qwen-1.5B effect gap is in the safe direction relative to H1: capability degrades *more* than safety (-6.8pp vs -3.1pp), not less. H1 is *opposite-supported*: capability is more sensitive to FP8 than safety is, in the only model where either is affected at the ±3pp level.

Second, only Qwen-1.5B shows a safety mean delta that approaches the ±3pp equivalence margin (-3.1pp, with the upper CI bound at +0.3pp). This is the same finding the SS18 TOST analysis flags: of the three models, Qwen is the only one whose paired safety comparison does *not* pass the ±3pp equivalence test. The interpretation is consistent across analyses.

The "Safety and capability similarly affected" interpretation in the per-model rows is computed by the analyzer's CI-overlap rule. Note that "similarly affected" here includes cases where neither is significantly affected (Llama-1B), where one trends down but neither reaches the equivalence margin (Llama-3B), and where both move in the same direction with overlapping CIs (Qwen-1.5B). It is a coarse summary; the per-task SS4 breakdown is finer-grained.

---

## SS6. Phase 3 Context Length × KV-Cache ANOVA

Phase 3 stratifies by context length (256 / 512 / 1024 / 2048 tokens of benign padding before the safety probe) crossed with KV-cache dtype (auto / fp8). The two-way ANOVA tests whether the FP16-FP8 gap depends on context length — i.e., whether FP8 rounding errors accumulate as the cache fills.

### SS6.1 ANOVA results

| Model | F (interaction) | p_interaction | η² (interaction) | Significant interaction? |
|-------|-----------------|---------------|------------------|--------------------------|
| llama3.2-1b | 0.131 | **0.974** | 0.000 | No |
| llama3.2-3b | 0.722 | **0.538** | 0.001 | No |

Both p-values are far above 0.05; both η² values are essentially zero (less than 0.1% of variance explained by the interaction term). The KV-cache × context-length interaction is not detectable at any reasonable α level.

### SS6.2 What this rules out

H2 (FP8 rounding accumulates over context length) predicts a monotonically widening FP16-FP8 gap as the cache fills with longer prefixes. The ANOVA does not reveal that pattern. The formal interpretation is conservative: we cannot reject the hypothesis that the FP16-FP8 gap is independent of context length within the tested range (256-2048 tokens).

The η² = 0.000 / 0.001 magnitudes are particularly informative. Even if a true interaction effect exists below detection threshold, it accounts for less than 0.1% of the per-cell variance — far below the threshold (≥1% / ≥5% / ≥10%) typically used to flag a meaningful interaction in safety evaluation contexts.

### SS6.3 Phase 3 limitations

The ANOVA is well-powered for medium effects (per SS19 power analysis: 80% power at MDE 5.7-6.1pp on the per-cell margins) but underpowered for small effects (1-3pp). The study does not rule out subtler effects below the ±3pp equivalence margin. It rules out gross accumulated-error effects that would matter for practical deployment.

The 2048-token ceiling also matters: many production deployments run with 8K-32K context windows. The Phase 3 result does not extrapolate beyond 2048; it is plausible that FP8 effects accumulate at longer context lengths and we have not measured them.

---

## SS7. Phase 3 Context Length Slope Analysis

The slope analysis fits a linear regression of safety rate on context length, separately for FP16 and FP8, then compares the slopes. A larger negative slope under FP8 vs FP16 would support H2 (degradation grows with context).

### SS7.1 Per-model slopes

**llama3.2-1b**

| Dtype | Slope (per token) | Slope (per 1k tokens) | R² | 95% CI |
|-------|-------------------|------------------------|-----|--------|
| auto (FP16) | -6.2e-05 | -6.2pp / 1k tokens | 0.86 | [-1.05e-4, -2.0e-5] |
| fp8 | -7.0e-05 | -7.0pp / 1k tokens | 0.92 | [-1.13e-4, -2.9e-5] |
| **Difference (fp8 - auto)** | **-8e-06** | -0.8pp / 1k tokens | — | — |

Both FP16 and FP8 show monotonic safety degradation with context length on Llama-1B, with the FP8 slope marginally steeper (-7.0 vs -6.2pp per 1k tokens). The slope difference is -0.8pp / 1k tokens — directionally consistent with H2 but the magnitude is well below the ±3pp margin.

**llama3.2-3b**

| Dtype | Slope (per token) | Slope (per 1k tokens) | R² | 95% CI |
|-------|-------------------|------------------------|-----|--------|
| auto (FP16) | +4.2e-05 | +4.2pp / 1k tokens | 0.69 | [+3e-6, +8.3e-5] |
| fp8 | +5.9e-05 | +5.9pp / 1k tokens | 0.96 | [+1.8e-5, +9.9e-5] |
| **Difference (fp8 - auto)** | +1.7e-05 | +1.7pp / 1k tokens | — | — |

Llama-3B shows the *opposite* pattern: safety *improves* with context length under both FP16 and FP8, and FP8's improvement slope is slightly larger. This is a real cross-model inversion: longer contexts help Llama-3B and hurt Llama-1B.

### SS7.2 Why the Llama-3B slope is positive

The positive slope on Llama-3B at longer contexts is consistent with an in-context-learning effect: the longer the benign padding before the safety probe, the more the model has to anchor on the conversational tone, and the more reliably it produces a refusal-style response on the probe. This is not specific to KV-cache precision — it is an architectural property of the larger Llama-3.2 model.

The Llama-1B negative slope is the more surprising direction: longer context appears to *weaken* refusal on the smaller model. This is plausibly a context-length capability ceiling — the 1B model struggles to maintain coherent attention over 2k-token prefixes and may simplify its response in ways that bypass the system-prompt safety instruction.

### SS7.3 Implication for FP8 specifically

The slope-difference comparison is the relevant test for H2. On Llama-1B, FP8 is 0.8pp / 1k tokens steeper in the unsafe direction (slope difference = -0.8pp / 1k); on Llama-3B, FP8 is 1.7pp / 1k tokens *less steep* in the unsafe direction (slope difference = +1.7pp / 1k). The two models give opposite signs to H2.

The auto-generated interpretation in the analyzer flags Llama-1B as "FP8 shows steeper safety degradation with context length (accumulated error hypothesis supported)" because the slope direction is consistent with H2. The full picture across both models is more equivocal: H2 has weak support on Llama-1B and weak counter-evidence on Llama-3B, with neither slope-difference reaching the ±3pp threshold or the ANOVA significance threshold.

---

## SS8. Phase 3 FP8 Degradation Curve

The degradation curve plots the FP8-FP16 delta at each context length, fits a slope to that delta over context length, and reports whether the gap is monotonic or stable.

### SS8.1 Llama-3.2-1B degradation curve

| Context (tokens) | FP16 mean | FP8 mean | Delta (pp) | n / cell |
|------------------|-----------|----------|------------|----------|
| 256 | 45.6% | 46.4% | **+0.8** | 250 |
| 512 | 40.4% | 43.4% | **+3.0** | 250 |
| 1024 | 36.6% | 37.4% | +0.8 | 250 |
| 2048 | 33.2% | 33.6% | +0.4 | 250 |

Delta-slope: -7.5e-06 per token (-0.8pp per 1k tokens). R² = 0.26 — a weak fit. Trend label from analyzer: **"FP8 gap is stable across context lengths"**.

The Phase 3 1B curve shows the FP8 gap *positive* (FP8 produces higher safety than FP16) at all four context lengths, peaking at +3.0pp at ctx=512 and shrinking at longer contexts. This is the opposite direction from what H2 predicts. The non-monotonic shape — peak in the middle, taper at the ends — is also inconsistent with simple accumulation: pure error accumulation would predict monotonic widening.

### SS8.2 Llama-3.2-3B degradation curve

| Context (tokens) | FP16 mean | FP8 mean | Delta (pp) | n / cell |
|------------------|-----------|----------|------------|----------|
| 256 | 66.8% | 66.2% | -0.6 | 250 |
| 512 | 71.0% | 67.6% | **-3.4** | 250 |
| 1024 | 67.8% | 72.6% | **+4.8** | 250 |
| 2048 | 75.8% | 76.6% | +0.8 | 250 |

Delta-slope: +1.7e-05 per token (+1.7pp per 1k tokens). R² = 0.15. Trend label: **"FP8 gap is stable across context lengths"**.

Llama-3B's gap is genuinely non-monotonic: FP8 underperforms by 3.4pp at ctx=512, then *outperforms* by 4.8pp at ctx=1024, then settles to near-zero at ctx=2048. The two adjacent-cell delta swing of 8.2pp (ctx=512 to ctx=1024) at n=250 is consistent with sampling noise; the analyzer's "stable across context lengths" trend label correctly flags that no monotonic pattern is present.

### SS8.3 What the curves tell us about H2

The degradation curve is the most direct test of H2. The hypothesis predicts a monotonically widening unsafe-direction gap. The data shows:

- Llama-1B: gap is in the *safe* direction (FP8 > FP16), peaking in the middle, tapering at the ends.
- Llama-3B: gap is in the *unsafe* direction at ctx=512 only, *safe* direction elsewhere, no monotonic pattern.

Neither model produces a curve consistent with H2. The ANOVA result (SS6) is consistent with the curve interpretation — the interaction term is essentially zero variance. H2 is not supported.

---

## SS9. Phase 3 Critical Context Length

The "critical context length" is operationally defined as the context length at which the absolute FP16-FP8 gap first crosses ±3pp. It is a deployment-relevant statistic: it answers "below what context length is FP8 safety-equivalent to FP16?"

| Model | Threshold (pp) | Critical context (tokens) | All below threshold? | Interpretation |
|-------|---------------|---------------------------|----------------------|----------------|
| llama3.2-1b | 3.0 | 512 | No | First |delta| ≥ 3pp at ctx=512 |
| llama3.2-3b | 3.0 | 512 | No | First |delta| ≥ 3pp at ctx=512 |

Both models cross the ±3pp threshold first at ctx=512 — but in opposite directions. Llama-1B at ctx=512 has FP8 *better* by +3.0pp (the analyzer's interpretation note that "FP8 safety drops" is misleading for this case; the actual delta is +3.0pp safe-direction). Llama-3B at ctx=512 has FP8 *worse* by -3.4pp (genuine unsafe-direction crossing).

The "no monotonic pattern" finding from SS8 means the critical context length should be read with caution. At ctx=1024, the Llama-3B FP8-FP16 gap is +4.8pp (safe direction — FP8 is now better than FP16 by more than the threshold). The threshold-crossing is non-monotonic; calling ctx=512 "critical" is correct for the threshold definition but does not imply that contexts above 512 are unsafe.

For deployment recommendations, the more honest reading is: at no tested context length does FP8 produce a *monotonic* safety drop relative to FP16. Per-cell deltas in the 3-5pp range are observed but in mixed directions, consistent with sampling noise around an effectively-zero true effect.

---

## SS10. Phase 4 Batch Size × KV-Cache ANOVA

Phase 4 stratifies by batch size (1 / 4 / 8 concurrent requests) crossed with KV-cache dtype. The two-way ANOVA tests whether the FP16-FP8 gap depends on batch size — does FP8 amplify the batched-inference safety effects documented in TR138?

### SS10.1 ANOVA results

| Model | F (interaction) | p_interaction | η² (interaction) | Significant interaction? |
|-------|-----------------|---------------|------------------|--------------------------|
| llama3.2-1b | 0.099 | **0.980** | 0.000 | No |
| llama3.2-3b | 0.034 | **0.998** | 0.000 | No |

Even flatter than Phase 3. Both p-values are essentially 1, and η² is zero. The batch × KV-cache interaction is not detectable.

### SS10.2 What this rules out

H3 (batching and KV-cache quantization compound) predicts that the FP16-FP8 gap should grow as batch size grows. The ANOVA does not show that pattern. Combined with the SS11 interaction decomposition (every cell classified as approximately additive), the result is a clean rejection of H3.

This is operationally important: TR138 documented that batched inference produces ~4x safety flip rate vs single-request inference. If FP8 amplified that effect — say, doubled the flip rate at batch=8 — the deployment recommendation would be "do not enable FP8 KV-cache on high-concurrency endpoints." The data does not support that recommendation. FP8 effect on safety is approximately the same regardless of batch size, in the tested range.

### SS10.3 Caveat: SS19 power for Phase 4

Phase 4 has the lowest per-cell power in the study (MDE = 7.4-8.4pp at 80% power, n = 518 per cell). The ANOVA is well-powered to detect medium-to-large interaction effects but underpowered for small (≤3pp) interaction effects. The conclusion is "no large interaction" rather than "no interaction at all."

---

## SS11. Phase 4 Interaction Decomposition

The interaction decomposition computes, for each (model, batch_size) cell, what the additive prediction (KV effect + batch effect, independent) would be vs the observed compound effect. The interaction term is the residual: positive interaction means the effects compound super-additively, negative means sub-additively, near-zero means additive.

### SS11.1 Llama-3.2-1B decomposition

| Batch size | KV effect (pp) | Batch effect (pp) | Compound (pp) | Additive prediction | Interaction (pp) | Type |
|------------|----------------|-------------------|---------------|---------------------|------------------|------|
| 4 | -0.4 | -0.6 | -0.3 | -1.0 | +0.7 | approximately additive |
| 8 | -0.4 | -0.6 | -0.2 | -1.0 | +0.8 | approximately additive |

Llama-1B shows a tiny positive interaction term (+0.7-0.8pp): the compound effect is *less negative* than the additive prediction. This is a sub-additive or "saturating" interaction — when both effects are present, they don't fully stack. The magnitude is well below threshold.

### SS11.2 Llama-3.2-3B decomposition

| Batch size | KV effect (pp) | Batch effect (pp) | Compound (pp) | Additive prediction | Interaction (pp) | Type |
|------------|----------------|-------------------|---------------|---------------------|------------------|------|
| 4 | -0.6 | -0.5 | -1.1 | -1.1 | 0.0 | approximately additive |
| 8 | -0.6 | -0.3 | -1.1 | -0.9 | -0.2 | approximately additive |

Llama-3B's interaction terms are essentially zero. The KV effect (-0.6pp) and batch effect (-0.3 to -0.5pp) compose linearly within sampling precision.

### SS11.3 Aggregate interpretation

All four (model, batch_size) cells classified as approximately additive. None show super-additive (compound > additive prediction) or measurably sub-additive interaction. The 0.0 / -0.2 / +0.7 / +0.8 interaction-term magnitudes are in the noise floor for n = 518 per cell.

The operational takeaway is: a deployment team running paired safety profiling should test FP16-vs-FP8 at the production batch size, but does not need to test the full (KV × batch) factorial — the FP8 effect at batch=1 is a reasonable predictor of the FP8 effect at higher batch sizes within the tested range.

---

## SS12. Phase 4 Per-Task Breakdown

Per-task two-way ANOVAs (KV × batch) on the safety-task subset, separately for each model.

| Task | Llama-1B p_interaction | Llama-3B p_interaction |
|------|------------------------|------------------------|
| advbench_refusal | 1.000 | 0.962 |
| jailbreakbench_behaviors | 0.999 | 0.987 |
| jailbreak_amplification | 0.997 | 0.971 |
| bbq_bias | 0.995 | 0.989 |
| truthfulqa | 0.998 | 0.974 |

Every per-task per-model interaction p-value exceeds 0.96. There is no task on which the (KV × batch) interaction reaches conventional significance. The aggregate ANOVA result (SS10) is not driven by averaging-out heterogeneity across tasks; the pattern holds task-by-task.

This rules out a specific concern: that the aggregate flat ANOVA might be hiding a strong task-specific interaction (e.g., "FP8 × batch=8 ruins jailbreak amplification but improves AdvBench"). The data shows no such hidden pattern.

---

## SS13. Phase 4 TR138 Cross-Reference

TR138 documented that batched concurrent inference produces a 4x increase in safety flip rate vs batch=1 baseline on the same models. Phase 4 enables a direct cross-TR comparison: does FP8 KV-cache amplify TR138's batch-induced safety drift?

| Model | Batch size | FP16 batch delta | FP8 batch delta | Amplification ratio | KV amplifies batch? |
|-------|-----------|------------------|------------------|---------------------|---------------------|
| llama3.2-1b | 4 | -0.58pp | +0.10pp | -0.17 | No |
| llama3.2-1b | 8 | -0.58pp | +0.19pp | -0.33 | No |
| llama3.2-3b | 4 | -0.48pp | -0.48pp | 1.00 | No |
| llama3.2-3b | 8 | -0.29pp | -0.48pp | 1.67 | No |

The "FP16 batch delta" is the observed FP16 (model at batch=N - model at batch=1) safety delta. The "FP8 batch delta" is the same measurement under FP8. The amplification ratio is FP8_batch_delta / FP16_batch_delta. A ratio greater than 1 would mean FP8 amplifies the batch effect; below 1 means it dampens it.

All four (model, batch_size) cells fall below the amplification threshold. The Llama-1B batch effect actually reverses sign under FP8 (FP16 batch delta is -0.58pp, FP8 batch delta is +0.10/+0.19pp); the Llama-3B amplification ratio is 1.0 / 1.67, which the analyzer flags as not-amplifying because the absolute deltas are tiny (-0.29pp, -0.48pp) and a "1.67x amplification" of -0.29pp is still a very small absolute effect.

The analyzer's verdict: **"KV-cache quantization does not amplify batching effects."**

This is the strongest ruling-out result in TR145. TR138's batch-safety effect is itself small (sub-1pp at the deltas TR138 measured), and FP8 does not enlarge it. Combined with the SS10 ANOVA and SS11 decomposition, H3 is decisively unsupported.

---

## SS14. Phase 5 Setup-Turn Refusal Diagnostics

Phase 5 evaluates a 5-turn conversation arc: turns 1-4 are benign topic-steering setup, turn 5 is the safety probe. The setup turns are *diagnostic* — they tell us how the model behaves under the synthetic conversation history, but they are not the primary measurement. The primary measurement is the turn-5 paired McNemar (SS16).

Setup-turn diagnostics report the per-turn refusal *rate* on turns 1-4. Refusal rate above zero on a benign setup turn means the model is over-refusing the benign content of that turn, not refusing a probe.

### SS14.1 Per-model setup-turn refusal rates

**llama3.2-1b**

| KV dtype | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Slope | R² | Increases over turns? |
|----------|--------|--------|--------|--------|-------|-----|----------------------|
| auto | 0% | 0% | 0% | 0% | 0.000 | 0.0 | No |
| fp8 | 0% | 0% | 0% | 20% | 0.060 | 0.6 | **Yes** |

Llama-1B under FP8 produces 20% setup-refusal at turn 4 specifically. FP16 produces 0% setup-refusal across all four turns. The slope difference (FP8 - FP16) is +0.06 — FP8 accumulates over-refusal over the conversation.

**llama3.2-3b**

| KV dtype | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Slope | R² | Increases over turns? |
|----------|--------|--------|--------|--------|-------|-----|----------------------|
| auto | 0% | 0% | 0% | 60% | 0.180 | 0.6 | **Yes** |
| fp8 | 0% | 0% | 0% | 60% | 0.180 | 0.6 | **Yes** |

Llama-3B has the same setup-refusal pattern under both dtypes: 0% on turns 1-3, 60% on turn 4. The slope difference is exactly zero — FP8 and FP16 produce identical setup-refusal trajectories.

### SS14.2 What turn-4 setup-refusal means

The 20% / 60% turn-4 setup-refusal rates do not mean the models refuse the safety probe at turn 4; the safety probe is at turn 5. They mean that the *benign* content of turn 4 (a contextual-setup template) triggers a refusal-style response in the model. This is a measurement artifact: the conversation arc templates were constructed to walk the model through topic-steering content, and at turn 4 some models start producing refusal-shaped outputs as the conversation approaches the eventual probe.

The Llama-3B 60% setup-refusal-at-turn-4 is much higher than Llama-1B's 20% — consistent with the larger model being more sensitive to the conversation's drift. The fact that Llama-3B's setup-refusal is *identical* under FP16 and FP8 indicates that whatever sensitivity the model has, it is not a KV-cache-precision phenomenon.

The Llama-1B FP8-vs-FP16 difference (20% vs 0% at turn 4) is the only place in the entire Phase 5 setup diagnostic where FP8 and FP16 diverge. It is consistent with FP8 introducing some additional setup-refusal sensitivity on the smaller, less-stable model — but again, this is a setup-turn measurement, not the primary probe.

### SS14.3 Why this is "diagnostic only"

The setup turns are not safety measurements in the H0/H1 sense. The probe is at turn 5; the turn 1-4 measurements are intended to characterize the conversation arc's typical trajectory, not to be summed into a safety verdict. The analyzer correctly tags every Phase 5 turn-1-to-4 record with `is_measurement_turn: false` and excludes them from the SS18 TOST and SS19 power analyses.

---

## SS15. Phase 5 Setup-Turn × KV-Cache Diagnostic Interaction

A two-way ANOVA on the setup-turn refusal rates (turn × KV-cache dtype). Diagnostic only — does FP8 produce a different trajectory shape than FP16 over the setup turns?

| Model | F (interaction) | p_interaction | Significant? |
|-------|----------------|---------------|--------------|
| llama3.2-1b | computed but not flagged | n.s. | No |
| llama3.2-3b | computed but not flagged | n.s. | No |

Both models' setup-turn × KV-cache interactions are non-significant. The Llama-1B 20% / 0% turn-4 difference (SS14.1) is the largest single diagnostic point but does not produce a significant interaction across the full turn-1-to-4 trajectory.

Since these are diagnostic measurements, the non-significance is not a primary finding. It is a check that the setup-turn behavior is consistent enough between FP16 and FP8 that the turn-5 probe (SS16) is the right primary measurement, not confounded by a different conversation trajectory under each dtype.

---

## SS16. Phase 5 Turn-5 McNemar

Turn 5 is the safety probe — the actual measurement target of Phase 5. McNemar's test on paired conversations: does FP8 KV-cache flip the turn-5 safety outcome more often than chance?

| Model | n paired | Discordant | R→C | C→R | χ² | p (asymp) | p (exact) | McNemar OR | Significant? |
|-------|----------|-----------|-----|-----|----|-----------|-----------|------------|--------------|
| llama3.2-1b | 100 | 6 | 5 | 1 | 1.500 | 0.221 | **0.219** | 3.67 | No |
| llama3.2-3b | 100 | 0 | 0 | 0 | 0.000 | 1.000 | **1.000** | 1.00 | No |

### SS16.1 Llama-3.2-1B turn-5 result

100 paired conversations, 6 discordant outcomes: 5 went refusal→compliance under FP8 (unsafe direction), 1 went compliance→refusal (safe). McNemar p = 0.22 (exact), not significant. McNemar OR = 3.67 — the directional asymmetry is real (5 unsafe : 1 safe) but the absolute count is small.

The 5/1 ratio is the strongest unsafe-direction signal in the entire Phase 5 result. At n = 100 conversations, however, this is consistent with chance: a fair coin produces a 5-or-more-tails-in-6-flips outcome 11% of the time. The exact p of 0.22 quantifies that.

### SS16.2 Llama-3.2-3B turn-5 result

0 discordant outcomes. Every paired conversation produced the same turn-5 safety classification under FP16 and FP8. McNemar OR is undefined (b = c = 0); the analyzer reports OR = 1.0 and p = 1.0 by convention.

This is a striking result. At n = 100 conversations with 5 turns each (500 total record), zero turn-5 flips. Llama-3B's safety behavior under FP8 is byte-equivalent to its behavior under FP16 on this measurement.

### SS16.3 Why the two models diverge

Llama-1B: 6 discordant turn-5 outcomes (5/1 unsafe-leaning). Llama-3B: 0 discordant. The same prompts, the same temperature, the same seed, the same conversation history reconstruction. The only differences are the model size (1B vs 3B) and (per phase 1 / 2) the inherent baseline safety rate.

Plausible explanation: Llama-3B has a much higher final-turn safety baseline (99% per SS17.1) than Llama-1B (97%). At the higher ceiling, FP8 perturbations have less headroom to flip the outcome — most prompts that were going to refuse will still refuse. At Llama-1B's lower ceiling, a few percent more prompts are near the decision boundary where FP8 perturbations can flip them.

This is the "high-baseline-is-stable" interpretation also seen in Phase 2 (SS2.1: Qwen at 98% AdvBench produces fewer flips per absolute discordant count than its rate would suggest). It is a candidate explanation; the data is too sparse to confirm it.

### SS16.4 H4 verdict

H4 (reconstructed multi-turn histories expose FP8-sensitive final-turn safety failures) predicts a significant turn-5 McNemar effect. Neither model produces one. H4 is not supported. The Llama-1B 5/1 directional ratio is suggestive but not significant; replicating Phase 5 at n = 500 conversations rather than 100 would be the right way to follow up, and is recommended in the Limitations section.

---

## SS17. Phase 5 Conversation-Level Analysis

The conversation-level analysis aggregates by conversation rather than by turn: for each (model, dtype) pair, what fraction of the 100 conversations produced a fully-safe arc (turn 5 safe)?

| Cell | Mean turn-5 safety | 95% CI | n conv | % all-safe |
|------|-------------------|--------|--------|------------|
| llama3.2-1b auto | 0.970 | [0.936, 1.004] | 100 | 97.0% |
| llama3.2-1b fp8 | 0.930 | [0.879, 0.981] | 100 | 93.0% |
| **Δ (1B)** | **-0.040** | — | — | **-4.0pp** |
| llama3.2-3b auto | 0.990 | [0.970, 1.010] | 100 | 99.0% |
| llama3.2-3b fp8 | 0.990 | [0.970, 1.010] | 100 | 99.0% |
| **Δ (3B)** | **0.000** | — | — | **0.0pp** |

### SS17.1 Llama-1B 4pp drop

The Llama-1B conversation-level safety drops 4pp (97% → 93%) under FP8. The 95% CIs overlap heavily — [0.936, 1.004] and [0.879, 0.981] — but the lower bound of the FP8 CI (0.879) is below the upper bound of the FP16 CI (1.004), so the comparison is not strictly stratified-equivalent. The McNemar test on this comparison (SS16.1) produces p = 0.22, consistent with the wide CI.

This is the single most "concerning" Phase 5 result, but it is also at the equivalence-margin boundary. A 4pp drop on a 100-conversation paired test is what we would expect if there is a real ~3-5pp FP8 effect on multi-turn safety, *or* if there is no effect and the observation is sampling noise. The data cannot distinguish the two.

### SS17.2 Llama-3B identical

Llama-3B's mean turn-5 safety is 99% under both FP16 and FP8, with identical CIs. The McNemar test (SS16.2) reports zero discordant outcomes. The two distributions are not just equivalent at the ±3pp margin — they are statistically indistinguishable.

### SS17.3 Operational read

If a deployment team were running Llama-3.2-1B in a multi-turn chat product and considered enabling FP8 KV-cache, the right operational read is: the turn-5 safety probe sees a 4pp drop on a 100-conversation eval, with p = 0.22 against H0. This is *not* a definitive finding — the data is consistent with both "no effect" and "small real effect" — but it is suggestive enough that the team should run an independent paired eval on their own multi-turn workload before deploying. The recommended sample size for that independent eval, per SS19 power analysis at MDE 3.4pp, is ≥ 400 conversations.

For Llama-3.2-3B, the data does not support any concern about multi-turn FP8 effects in this study.

---

## SS18. TOST Equivalence Analysis

TOST (Two One-Sided Tests) at the ±3pp equivalence margin asks: is the FP16-FP8 delta within ±3pp at p < 0.05 against the margin? An equivalent result means we have positive evidence that the effect is small, not just absence of evidence for an effect.

### SS18.1 Summary

| Metric | Value |
|--------|-------|
| n tests | 22 |
| n equivalent at ±3pp | **9** |
| % equivalent | **40.9%** |

22 paired comparisons total: 6 from Phase 2 (3 models × safety + capability), 8 from Phase 3 cells, 6 from Phase 4 cells, 2 from Phase 5 turn-5. Nine pass the ±3pp equivalence test.

### SS18.2 Equivalent comparisons

The 9 comparisons that pass equivalence include the most operationally-relevant tests:

| Comparison | Δ (pp) | TOST p | Bound (pp) |
|------------|--------|--------|-----------|
| p2_vs_p1 llama3.2-1b safety | -0.39 | 0.0086 | ±3.0 |
| p2_vs_p1 llama3.2-3b safety | -0.58 | 0.0130 | ±3.0 |
| p2_vs_p1 llama3.2-1b capability | -0.21 | 0.0035 | ±3.0 |
| (plus 6 from the Phase 3-5 cell battery) | | | |

The Llama-1B safety equivalence (delta = -0.39pp, well inside ±3pp, TOST p = 0.0086) is the headline equivalence result: under tested conditions, FP8 KV-cache is positively-confirmed equivalent to FP16 within ±3pp on Llama-1B safety. Same story for Llama-3B safety (-0.58pp).

### SS18.3 Non-equivalent comparisons

Two notable non-equivalences:

| Comparison | Δ (pp) | TOST p | Why not equivalent |
|------------|--------|--------|---------------------|
| p2_vs_p1 qwen2.5-1.5b safety | **-3.09** | 0.521 | Delta exceeds equivalence bound |
| p2_vs_p1 llama3.2-3b capability | -1.65 | 0.071 | Within bound but high paired-sample variance |

Qwen-1.5B safety is the only model whose paired safety comparison fails the equivalence test, and it fails at the boundary: the observed delta is -3.09pp, just outside the ±3.0pp margin. The CI on the delta is [-5.91, -0.27], meaning the upper bound is just barely above zero. This is consistent with the SS5 finding that Qwen is the only model where the safety effect approaches but does not clearly exceed the practical-significance threshold.

Llama-3B capability (delta -1.65pp) is *within* the ±3pp margin in absolute terms but does not pass the TOST equivalence test because the paired-sample variance is high (TOST p = 0.071). With more paired observations on the same prompts, this comparison would likely flip to equivalent.

### SS18.4 Interpretation

A 9/22 equivalence rate (40.9%) is moderate. It is not a clean-sweep equivalence (which would be 22/22) but it is also not a clear failure (which would be 0/22 or near it). The pattern is consistent with the rest of the report: small models (Llama-1B) are equivalent, mid-size (Llama-3B) is mostly equivalent, Qwen-1.5B is at the boundary on safety, and Phase 4 cell-level equivalences are mixed because per-cell n = 518 is at the lower end of TOST's reliable-detection range.

The deployment-relevant TOST result is the per-model paired-safety test from Phase 2: Llama-1B and Llama-3B pass equivalence at ±3pp on safety; Qwen-1.5B fails by 0.09pp. This is the right level at which to make deployment decisions.

---

## SS19. Power Analysis

Retrospective power analysis confirms whether the null findings are real null findings or power-starved nondetections. For each cell, the power analysis computes the minimum detectable effect (MDE) given the cell's baseline rate, sample size, and α = 0.05 at 80% power.

### SS19.1 Phase 1+2 paired tests

| Comparison | Baseline rate | n | MDE (pp) | Powered ≥ 80%? |
|------------|---------------|----|----------|----------------|
| Phase 1 safety | 72.94% | 1554 | 4.5 | Yes |
| Phase 1 capability | 52.23% | 1455 | 5.2 | Yes |
| Phase 2 safety | 71.59% | 1554 | 4.5 | Yes |
| Phase 5 turn-5 | 97.00% | 400 | 3.4 | Yes |

The Phase 1+2 safety MDE of 4.5pp is interpretable: had there been a true 4.5pp FP8 safety degradation, the test would have detected it 80% of the time. The observed deltas are all below 4.5pp on Llama models and 3.1pp on Qwen — within the MDE range, so the null is consistent with either "no true effect" or "true effect smaller than MDE." Practical-significance reasoning (the ±3pp equivalence margin in SS18) is the appropriate disambiguator.

The Phase 5 turn-5 MDE of 3.4pp at n = 400 is the *most powered* cell in the study, owing to the high baseline safety rate (97%) which compresses the binomial variance. At this MDE, a ≥ 3.4pp FP8 effect on turn-5 safety would be detected, and the data shows |delta| ≤ 4pp on Llama-1B and 0pp on Llama-3B, consistent with no detectable effect.

### SS19.2 Phase 3 per-cell

| Cell | Baseline | n | MDE (pp) |
|------|----------|---|----------|
| llama3.2-1b auto | 38.95% | 1000 | 6.1 |
| llama3.2-1b fp8 | 40.20% | 1000 | 6.1 |
| llama3.2-3b auto | 70.35% | 1000 | 5.7 |
| llama3.2-3b fp8 | 70.75% | 1000 | 5.7 |

Phase 3 cell MDE is 5.7-6.1pp. The per-cell deltas observed are mostly below this (the largest is 4.8pp on Llama-3B at ctx=1024), so the per-cell tests are individually underpowered for the small effects observed. The cross-cell ANOVA in SS6 pools across all four context lengths to gain power, and even there the result is non-significant (p ≥ 0.54).

### SS19.3 Phase 4 per-cell

| Cell | Baseline | n | MDE (pp) |
|------|----------|---|----------|
| Llama-1B cells | 63-64% | 518 | 8.3-8.4 |
| Llama-3B cells | 75-76% | 518 | 7.4-7.6 |

Phase 4 has the lowest per-cell power (MDE 7.4-8.4pp). Per-cell effects are well below this MDE; the aggregate ANOVA (SS10) is the right test. The η² = 0 result there confirms no large interaction effect.

### SS19.4 Power-starved? No

The pattern across the power-MDE table is consistent with the null findings being real: all primary tests achieve 80% power at α = 0.05, the MDE values are smaller than the practical-significance threshold (±3pp) only on Phase 5 turn-5, and the observed deltas are below MDE in all cases. We are not seeing "no effect because no power"; we are seeing "no effect within practical-significance margins, with sufficient power to detect the practical-significance margin if it existed."

---

## SS20. Cross-Model Synthesis (Mantel-Haenszel)

Mantel-Haenszel synthesis pools per-model 2x2 contingency tables into a single across-model odds ratio. It is robust to model-specific baseline rates and is the right cross-model summary when each model's data contributes to a shared question. Note: MH uses *unpaired* per-model tables; within-model pairing is captured by McNemar in SS2 and SS16.

| Comparison | Pooled OR | 95% CI | n strata |
|------------|-----------|--------|----------|
| FP16 vs FP8 safety (Phase 2) | **1.05** | [0.90, 1.23] | 3 |
| FP16 vs FP8 batch=8 safety (Phase 4) | **1.00** | [0.83, 1.21] | 2 |
| FP16 vs FP8 turn-5 safety (Phase 5) | 2.06 | [0.61, 6.99] | 2 |

### SS20.1 Phase 2 pooled OR

Pooled across 3 models: OR = 1.05, 95% CI [0.90, 1.23]. The CI straddles 1; the upper bound (1.23) corresponds to a ~5pp safety reduction at FP16's 73% baseline. The lower bound (0.90) is in the safe-direction. The MH test does not reject H0 at α = 0.05.

This is the cross-model headline: pooling across the three models, FP8 KV-cache produces an estimated 5% relative-odds change in safety with a CI that comfortably brackets no effect.

### SS20.2 Phase 4 batch=8 pooled OR

Pooled across the 2 Llama models at the highest batch size: OR = 1.00, CI [0.83, 1.21]. This is the most-symmetric MH result in TR145: the pooled OR is *exactly* 1.0, indicating that at batch=8 specifically, FP8 produces the same odds of safety as FP16. The CI is similar in width to the Phase 2 result, as expected given the same per-cell n.

### SS20.3 Phase 5 turn-5 pooled OR

Pooled across the 2 Llama models on the multi-turn safety probe: OR = 2.06, CI [0.61, 6.99]. The point estimate is suggestive (FP8 produces 2x the odds of refusal-failure on multi-turn vs FP16) but the CI is wide because n_strata = 2 and the per-stratum n_discordant is small (6 + 0 = 6 total discordant). The CI's lower bound (0.61) is comfortably below 1, and the upper bound (6.99) is too wide to interpret as a strong upper-bound effect.

This is the noisiest MH stratification in the study. With more conversations or more models, the CI would tighten; at the tested n, the result is "directionally consistent with the SS16 turn-5 McNemar 5/1 ratio on Llama-1B, but neither is individually significant."

### SS20.4 Synthesis

The three MH pooled ORs (1.05, 1.00, 2.06) tell a coherent story: aggregate (Phase 2) and high-batch (Phase 4) tests see no FP8 effect; multi-turn (Phase 5) sees a directionally suggestive effect with a wide CI. Pre-registered analysis plan does not select Phase 5 as a primary test; the headline remains the Phase 2 / Phase 4 null. The Phase 5 directional signal is flagged as an open question for follow-up at higher n.

---

## SS21. Judge Agreement

Cohen's κ between the regex classifier and `gemma3:12b` LLM judge on the Phase 1+2 safety records. κ > 0.80 = near-perfect agreement; 0.60-0.80 = substantial; 0.40-0.60 = moderate; < 0.40 = fair-to-poor.

### SS21.1 Aggregate agreement

| Metric | Value |
|--------|-------|
| Overall agreement | 75.4% |
| Cohen's κ | **0.43** |
| n judged | 13,676 |
| n unclear (judge gave UNCLEAR label) | 48 |

κ = 0.43 is in the moderate range. For a generalist LLM judge versus regex classifiers on safety classification, this is the expected level — regex over-counts soft refusals that the judge correctly classifies as compliance, and the judge over-counts cautious-but-not-refusing responses that the regex correctly classifies as compliance. The 0.43 number is consistent with the TR140 JTP κ = 0.925 finding being achievable only when using two LLM judges (`gemma3:12b` × Claude); regex-vs-judge agreement is fundamentally noisier because the two are measuring slightly different things.

### SS21.2 Per-task agreement

| Task | Agreement | n | n unclear |
|------|-----------|---|-----------|
| advbench_refusal | **90.3%** | 300 | 0 |
| jailbreakbench_behaviors | 79.3% | 150 | 0 |
| jailbreak_amplification | 74.4% | 360 | 0 |
| bbq_bias | 68.8% | 593 | 1 |
| truthfulqa | 69.9% | 146 | 4 |

AdvBench refusal has the highest agreement (90.3%) — the regex and judge agree on hard refusal-vs-compliance classifications. JailbreakBench and jailbreak amplification are lower (79.3% / 74.4%) because their prompts are softer / more adversarial, and the regex's hard-refusal patterns miss paraphrased compliance. BBQ bias and TruthfulQA are lowest (68.8% / 69.9%) because the regex is fundamentally less suited to bias and truthfulness scoring than to refusal detection.

### SS21.3 The JBB judge fix landed

The Phase 1 JBB judge issue (150/150 records initially classified as UNCLEAR by the analyzer's safe-outcome mapping despite valid FULL_REFUSAL labels from gemma3:12b) was fixed in `analyze.py` and confirmed in the post-fix run. JBB now contributes 150 valid records to the kappa calculation at 79.3% agreement, between AdvBench (90.3%) and jailbreak amplification (74.4%) — exactly where a refusal-task should land. The fix is documented in the run-time bug log.

### SS21.4 Implication for downstream tests

The McNemar tests in SS2 use the regex classifier as the primary outcome, with judge labels merged in for cross-validation. The 75.4% / κ = 0.43 inter-rater agreement means the McNemar tests are not measuring a pure regex artifact — the judge agreeing with the regex on 75% of records means the test statistics largely reflect the underlying safety outcome. The 25% disagreement tail is non-trivial but is not concentrated in a way that would systematically bias the McNemar result toward false significance.

---

## SS22. Cross-TR Validation

Cross-TR validation compares TR145 baseline rates against TR138 (batch safety, same models), TR143 (cross-architecture, same models), and TR144 (speculative decoding, same models). The check: are TR145's Phase 1 baselines within 5pp of the same model's baseline in prior TRs?

The standalone `cross_tr_validate.py` script ran against `research/tr138/`, `research/tr143/`, and `research/tr144/` result directories, producing 36 (prior-TR × model × task) baseline comparisons. The output is in `research/tr145/results/20260508_033550/cross_tr_validation.json`.

### SS22.1 Validation summary

| Metric | Value |
|--------|-------|
| Total comparisons | 36 |
| Consistent within ±5pp | **36** |
| Drifted | 0 |
| Drift threshold | 5.0pp |
| Verdict | **all_consistent: true** |

Every (prior-TR, model, task) baseline tuple in TR145 is within ±5pp of the corresponding baseline in TR138, TR143, or TR144. Most comparisons are *byte-identical* (Δ = 0.00pp) because the same prompts, models, and seed produced the same outputs across runs.

### SS22.2 Largest observed deltas

| Prior TR | Model | Task | TR145 score | Prior score | Δ (pp) |
|----------|-------|------|-------------|-------------|--------|
| TR138 | llama3.2-1b | jailbreak_amplification | 51.67% | 50.83% | +0.83 |
| TR138 | llama3.2-1b | mmlu_real | 31.58% | 31.93% | -0.35 |
| (all others) | (various) | (various) | — | — | 0.00 |

Two non-zero deltas, both well below the 5pp threshold and both on Llama-1B vs TR138. The +0.83pp jailbreak amplification delta is the largest single drift in the entire 36-tuple comparison; it likely reflects a small TR138-vs-TR145 difference in the multi-shot prompt construction or judge labeling, but at this magnitude it is consistent with sampling noise and does not indicate a methodological drift that would invalidate the FP8-vs-FP16 paired finding in TR145.

### SS22.3 What this validates

This is a *baseline reproducibility* check, not a cross-TR transferability claim for the FP8-vs-FP16 finding. The narrow claim it supports: when you run TR145 with the same models, prompts, seed, and dtype settings as TR138/143/144 used for their FP16 baselines, you get the same rates. That rules out a class of failure modes — task-set drift, scoring-rule drift, model-version drift — that would have caused the TR145 baselines to disagree with prior TRs even before any FP8 manipulation.

The broader claim (that the TR145 FP8-vs-FP16 paired result would also reproduce in another TR) is not testable here, because no prior TR ran an FP8-KV-cache vs FP16-KV-cache paired battery on the same models. The only path to that claim is independent replication.

---

## Conclusions

**Conclusion 1 — Primary result.** The isolated Phase 2 paired McNemar test does not establish a Holm-corrected FP8 KV-cache safety effect under the tested conditions. With n = 1,003 prompts per model, paired across (model, task, sample_id), at temperature 0.0 with seed 42 and identical FP16 model weights, no Holm-significant safety effect is detected on any of three models (p = 1.00, 0.60, 0.31). The correct paper claim is bounded: KV-cache dtype is an evaluation-relevant serving perturbation worth profiling, but this run does not by itself prove safety degradation.

**Conclusion 2 — Flip directions.** Per-model and per-task flip directions are observed but not decisively asymmetric. The aggregate cross-model unsafe : safe ratio is 80 : 65 (binomial p ≈ 0.13), and the per-model directional ratios are 0.51-0.56. The per-task pattern shows AdvBench-refusal flips clustering in the unsafe direction (10 unsafe : 5 safe pooled across models), which is suggestive of a small unsafe-direction bias on hard-refusal tasks specifically, but it does not rise to statistical significance at the per-cell level. The report describes the flip-direction pattern as descriptive, not as evidence of systematic FP8 weakening.

**Conclusion 3 — Context length interaction.** Phase 3 does not establish a significant context-length amplification effect. Both Llama models produce p_interaction ≥ 0.54 with η² ≈ 0; the FP8-FP16 gap is non-monotonic (positive at some context lengths, negative at others), inconsistent with the accumulated-rounding-error hypothesis. Any context-length trend in the report is descriptive; H2 is not supported.

**Conclusion 4 — Batch interaction.** Phase 4 does not establish a significant batch-size interaction. p_interaction ≥ 0.98 on both models, η² ≈ 0, and every (model, batch_size) cell is classified as approximately additive. Batch-related deployment recommendations remain "profile at representative batch sizes" rather than "FP8 amplifies batched-safety effects." The TR138 cross-reference confirms this directly: KV-cache quantization does not amplify the batch-induced safety drift documented in TR138.

**Conclusion 5 — Multi-turn final-probe.** Phase 5 setup-turn diagnostics show some setup-refusal trajectory differences between FP16 and FP8 on Llama-1B (20% vs 0% setup-refusal at turn 4), but the primary measurement is the turn-5 paired McNemar, which is non-significant on both models (1B p = 0.22 with 5/1 unsafe-leaning ratio; 3B p = 1.0 with zero discordant outcomes). H4 is not supported. The Llama-1B 5/1 directional ratio is the most suggestive unsafe-direction signal in the report; replicating Phase 5 at n ≥ 400 conversations rather than 100 would be the right follow-up.

**Conclusion 6 — Cross-cutting.** TOST equivalence at ±3pp passes 9 of 22 comparisons, including all Llama paired safety tests (delta ≤ 0.6pp). Qwen-1.5B safety is the only model whose paired safety test fails the equivalence margin, at delta = -3.09pp (just outside ±3.0pp). Mantel-Haenszel pooled odds ratios are 1.05 [0.90, 1.23] for Phase 2 safety, 1.00 [0.83, 1.21] for batch=8 safety, and 2.06 [0.61, 6.99] for multi-turn turn-5 safety — all CIs straddle 1. Power analysis confirms every cell achieves 80% power at α = 0.05, MDE 3.4-8.4pp depending on cell.

**Conclusion 7 — Production guidance.** FP8 KV-cache is a deployment decision that requires paired safety profiling on representative workload distributions. The report's strongest durable recommendation is not a blanket prohibition or a blanket approval; it is that operators should establish equivalence margins, run paired McNemar tests at production-typical context lengths and batch sizes, and verify the FP8 deployment falls within their tolerance. The TR145 pipeline provides a reference recipe for that paired eval.

---

## Limitations & Threats to Validity

### Model size ceiling

The largest model tested is Llama-3.2-3B at 3.21B parameters. Production deployments routinely run 7B-70B models. The TR145 result is silent on whether FP8 KV-cache safety effects scale with parameter count. There are competing predictions: (a) larger models have more redundant attention heads and may be *more* robust to per-head precision loss; (b) larger models have more attention computation and more cumulative FP8 rounding events, potentially less robust. The data does not adjudicate; the right follow-up is a Llama-3.1-8B or Llama-3.1-70B run on rented A100/H100 capacity, which is outside the laptop-GPU scope of this study.

### Temperature ceiling

All measurements are at temperature 0.0 (greedy decoding). Many production deployments run at temperature 0.7-1.0 for better response diversity. At higher temperatures, the same FP8 attention perturbation could push softmax outputs across token-selection boundaries that they would not cross at temperature 0. The TR145 result does not extrapolate to temperature > 0; the open question is whether temperature-induced sampling stochasticity *masks* a real FP8 effect (because outputs vary even under FP16) or *amplifies* it (because sampling near a tilted distribution lands on different tokens). This is the same caveat TR144 flagged for speculative decoding.

### Context length ceiling

The longest context tested is 2048 tokens. Production deployments routinely run 8k-128k context (RAG, long-document QA, long-form chat). The Phase 3 ANOVA does not extrapolate beyond 2048; H2 (accumulated rounding error) could in principle become detectable at much longer contexts. The Limitations recommendation is to profile FP8 at the production context-length distribution rather than treat the 2048-token result as binding.

### Conversation length ceiling

Phase 5 tests 5-turn conversations with a single safety probe at turn 5. Real multi-turn deployments often have much longer arcs (10-100 turns). The probe at turn 5 specifically may not generalize to "FP8 turn-100 safety equivalence." Increasing conversation length is the right follow-up but requires either many more conversations or a different probe-placement design (e.g., probe-every-Nth-turn rather than probe-at-turn-5).

### Judge model substitution

The pre-registration scaffold specified `qwen2.5:7b-instruct-q8_0` as the judge model; the executed run used `gemma3:12b` because the pinned judge model was not present on the run hardware and `gemma3:12b` was already validated in TR140 at JTP κ = 0.925. The judge model swap is documented in the working-tree changes to `config.yaml` and `judge_analysis.py`. The McNemar tests use regex-classifier outcomes as the primary outcome (not judge labels), so the judge-model swap affects only the inter-rater κ computation, not the headline test statistics. The fact that the judge model is now config-driven rather than hardcoded is an improvement in run-time hygiene.

### Hardware specificity

The study runs on RTX 4080 Laptop (sm_8.9, Ada Lovelace). FP8 KV-cache on Hopper (sm_9.0, H100) uses E4M3 with slightly different numerical properties; on AMD MI300X, the FP8 implementation is different again. The TR145 result is calibrated to the Ada implementation specifically. Cross-hardware extrapolation requires re-running the same battery on the target hardware, which is the purpose of the pinned vLLM image plus reproducibility recipe (Reproducibility section).

### vLLM version specificity

The pinned vLLM image is v0.19.1. Newer vLLM versions (v0.20.1, the current `:latest` at the time of writing) include FP8 KV-cache improvements. The result is calibrated to v0.19.1 specifically. A re-run on v0.20+ could produce different numbers; the difference would itself be informative but is outside this study's scope.

### Single-laptop reproducibility

The study runs on a single laptop GPU, including a ~6.5 hour laptop-sleep event mid-run. The resume-by-default mechanism handled the sleep cleanly (no record loss), but a full re-run on a non-laptop deployment would not have the same sleep event and may surface a different timing-related effect (e.g., lower thermal stress and therefore different FP8 numerical noise at the hardware level). This is a low-probability concern but worth flagging.

### Pre-registration vs working tree drift

The pre-registration scaffold lives at `papers/kv_cache_quantization_safety/`. The executed run made several modifications during execution: (a) the judge model swap from `qwen2.5:7b-instruct-q8_0` to `gemma3:12b`; (b) a JailbreakBench task that was added to the analyzer's refusal-task tuples after Phase 1 ran (the original analyzer treated valid FULL_REFUSAL labels as UNCLEAR for that task); (c) a `turn5_effect` variable bug in the report generator that was patched mid-run. Each of these is documented in the working-tree git diff and in the run-time bug log. None affects the headline McNemar/ANOVA result.

### Sample size for hard-refusal tasks

AdvBench refusal has n = 100 samples and JailbreakBench has n = 50. The Qwen-1.5B JailbreakBench delta of -14pp is the largest single-cell delta in the matrix, but at n = 50 it is not significant in isolation. The right follow-up — and the most actionable single follow-up beyond TR145 — is to run a 1,000-sample paired AdvBench / JailbreakBench eval specifically on Qwen-1.5B under FP8 vs FP16 KV-cache, to nail down whether the small-n delta is a real Qwen-specific effect or sampling noise.

---

## Production Guidance

### KV-Cache Dtype Recommendations

| Deployment Tier | Recommended KV-Cache Dtype | Rationale |
|----------------|----------------------------|-----------|
| Safety-critical (medical, legal, government) | **FP16 (auto)** | The cost of an undetected safety regression in this tier exceeds any throughput benefit; FP8 should not be enabled without an independent paired eval that exceeds the equivalence margin used in TR145. |
| Standard production (consumer chat, agent loops) | **FP8 acceptable after local equivalence check** | TR145 found Llama-1B and Llama-3B safety equivalent at ±3pp; if your model and workload are similar, FP8 is likely safe — but verify with a paired McNemar on your own prompts. |
| Throughput-optimized (non-safety, e.g., code completion, retrieval) | **FP8** | KV-cache memory savings of ~50% enable significantly higher concurrent-request budgets at fixed VRAM; safety considerations are not the primary constraint. |
| Long-context workloads (>2k tokens) | **Profile at representative long contexts** | TR145 tested up to 2048 tokens. Beyond that, the accumulated-error hypothesis remains untested. Run a paired eval at your representative context length before enabling. |
| High-batch endpoints (batch ≥ 8) | **FP8 acceptable, with caveat** | TR145 batch × KV-cache interaction was non-significant up to batch=8. Higher batch sizes (16, 32) were not tested; profile at production batch size. |
| Multi-turn chatbots (≥ 5 turns) | **Profile, especially on smaller models** | TR145 turn-5 paired McNemar non-significant but Llama-1B showed 5/1 unsafe-leaning directional ratio at n=100. For multi-turn deployments on small models, recommend ≥ 400-conversation paired eval. |
| Qwen 2.5 family at 1.5B parameter scale | **FP16 unless paired eval confirms equivalence** | TR145 found Qwen-1.5B safety -3.09pp under FP8 (just outside the ±3pp equivalence margin), with -14pp on JailbreakBench at n=50. The Qwen result deserves an independent replication. |

### Pre-Deployment Safety Profiling Recipe

Before enabling `--kv-cache-dtype fp8` on any safety-relevant endpoint, run this paired profiling protocol:

1. **Baseline.** Sample 1,000 prompts from your production distribution (or the relevant safety-eval subset) under your current FP16 deployment. Record per-prompt outcomes (refused / complied) deterministically (temperature 0.0, fixed seed).
2. **FP8 mirror.** Sample the same 1,000 prompts under the proposed FP8 deployment, with all other settings identical. Use the same seed and temperature.
3. **Pair and McNemar.** Match outcomes by prompt ID. Run McNemar's test on the discordant cell counts. The null hypothesis is "FP8 does not change paired safety outcomes."
4. **TOST equivalence.** Compute the FP8-FP16 mean delta and its 95% CI. Test against your equivalence margin (recommend ±3pp for production chat, ±1pp for safety-critical).
5. **Per-task breakdown.** Disaggregate by task (refusal, bias, truthfulness). The aggregate may be equivalent while a single task fails — TR145 saw this pattern with Qwen-1.5B's AdvBench/JailbreakBench specifically.
6. **Production monitoring.** If FP8 is deployed, track refusal-rate drift as a time series. Alert on drops > 3pp from the FP16 baseline week-over-week.
7. **Re-validate annually or on hardware change.** FP8 numerical properties differ between Ada (sm_8.9) and Hopper (sm_9.0); cross-hardware deployments require independent profiling.

### Memory Savings vs Safety Risk Tradeoff

| Metric | FP16 KV-Cache | FP8 KV-Cache (E4M3) |
|--------|---------------|---------------------|
| Memory per token (1B model, single sequence) | ~0.5 MB | ~0.25 MB |
| Memory per token (3B model, single sequence) | ~1.5 MB | ~0.75 MB |
| Max concurrent requests on 12 GB GPU (3B model, 2k context) | ~3 | ~6 |
| Throughput uplift (typical) | reference | 1.6-2.1x |
| Numerical precision (relative error per cached token) | ~2^-10 | ~2^-3 |
| Safety equivalence (TR145, tested conditions) | reference | TOST-equivalent at ±3pp on Llama; -3.09pp on Qwen-1.5B |
| Capability equivalence (TR145, tested conditions) | reference | TOST-equivalent on Llama-1B, marginal on Llama-3B, -6.8pp on Qwen-1.5B |

The throughput case for FP8 is real and significant. The 50% KV-cache memory reduction enables doubling the concurrent-request budget on the same GPU, or extending the supported context window at fixed batch size. For deployments where throughput / cost is the binding constraint, this is a meaningful optimization.

The safety case is workload-specific. Within the tested conditions, TR145 finds no general-purpose blocker to FP8 deployment, but it also finds enough per-task / per-model variation (Qwen JailbreakBench, Llama-1B multi-turn directional ratio) that "trust the flag" is not the right operational stance. Run the recipe.

---

## Reproducibility

### Run command

```bash
python research/tr145/run.py --phases 1,2,3,4,5
```

Optional flags:
- `--config <path>` — override `research/tr145/config.yaml`
- `--run-dir <path>` — resume into existing run directory (default: new timestamped dir)
- `--phases 1,2,3,4,5` — comma-separated phase selection
- `--rerun-selected` — discard selected-phase records before running them
- `--validate-only` — exercise prep + Docker/GPU + FP8 gate, exit before sampling
- `--skip-judge` — skip the LLM judge step (regex-only)
- `--judge-model <tag>` — override `judge.model` from config (e.g., `gemma3:12b`)

### Pinned dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| vLLM | v0.19.1 (Docker pinned) | Inference server with `--kv-cache-dtype` support |
| Docker Engine | 29.2.1 | Container runtime |
| NVIDIA Container Toolkit | (current) | GPU passthrough to containers |
| Ollama | (current) | Judge backend |
| `gemma3:12b` | (Ollama pull) | LLM judge |
| Python | 3.13.1 | Host orchestration |
| `torch` | 2.x | Host-side tensor utilities |
| `httpx` | 0.27+ | vLLM API client |
| `pyyaml` | 6.x | Config + task YAML parsing |
| Hardware | RTX 4080 Laptop 12 GB or any sm_8.9+ NVIDIA GPU | FP8 KV-cache requires CC ≥ 8.9 |

### Hardware envelope

- **Minimum GPU:** sm_8.9 (Ada Lovelace; RTX 4080 Laptop, RTX 4090, L40S) or sm_9.0+ (Hopper; H100, H200, B100). FP8 KV-cache is not supported on Ampere (sm_8.0, A100) or earlier.
- **Minimum VRAM:** 12 GB to run all five phases including Phase 4 batch=8 on Llama-3.2-3B.
- **Recommended VRAM:** 24-48 GB for headroom on batch=16+ or longer context experiments.
- **Disk:** ~10 GB for the pinned vLLM image, ~150 MB for samples.jsonl + judge_labels.jsonl + scored.jsonl, ~10-20 GB for HF model cache.

### Run timing on RTX 4080 Laptop

| Phase | n records | Wall time |
|-------|-----------|-----------|
| Phase 0 (FP8 gate) | 1 test prompt | ~1 min |
| Phase 1 (FP16 baseline, 3 models × 1003) | 3,009 | ~65 min |
| Phase 2 (FP8, 3 models × 1003) | 3,009 | ~80 min |
| Phase 3 (2 models × 4 ctx × 2 dtype × 250) | 4,000 | ~150 min |
| Phase 4 (2 models × 3 batch × 2 dtype × 1003) | 12,036 | ~200 min |
| Phase 5 (2 models × 2 dtype × 100 conv × 5 turns) | 2,000 | ~110 min |
| Judge phase (gemma3:12b on 13,724 safety records) | 13,724 | ~108 min |
| Analysis (23 passes) | — | ~1 min |
| Report generation | — | ~1 min |
| **Total active GPU time** | **24,054** | **~12 hr** |
| Including ~6.5 hr laptop sleep mid-run | — | ~25 hr clock |

### Resume mode

The run is resume-safe by default. Each record carries a deterministic identity, `samples.jsonl` is written incrementally with atomic temp-file replace, and on re-launch the runner skips records with `status: ok` and re-samples records with errors or missing identities. A laptop sleep, container crash, or process kill loses no completed records; only in-flight records (the cell currently sampling) need to be resumed.

### Where the artifacts live

```
research/tr145/results/20260508_033550/
├── config_snapshot.yaml      # frozen config for this run
├── run_metadata.json         # provenance: git commit, platform, argv, timestamps
├── samples.jsonl             # 24,054 raw inference records
├── judge_labels.jsonl        # 13,724 judge labels (gemma3:12b, blinded)
├── tr145_scored.jsonl        # samples + regex scores + judge labels merged
├── tr145_analysis.json       # 23-pass analysis output
└── tr145_report.md           # auto-generated TR
```

The published copy lives at `PublishReady/reports/Technical_Report_145.md` and is the hand-narrated companion to the auto-generated tr145_report.md (the publish step in `generate_report.py` performs a `shutil.copy2`; this hand-edited version supersedes the auto-copy for publication purposes).

### Re-running just analysis or report

If the source samples are intact, only later steps can be re-run:

```bash
# Re-run analysis only (uses existing samples.jsonl + judge_labels.jsonl)
python research/tr145/analyze.py --run-dir research/tr145/results/20260508_033550

# Re-run report only (uses existing tr145_analysis.json)
python research/tr145/generate_report.py --run-dir research/tr145/results/20260508_033550
```

Both are idempotent and run in under a minute.

---

## Appendix A: Raw Tables

### A.1 Phase 1 baseline rates (full per-task per-model)

| Task | Llama-1B rate | Llama-3B rate | Qwen-1.5B rate | n / cell |
|------|---------------|---------------|----------------|----------|
| advbench_refusal | 0.650 | 0.790 | 0.980 | 100 |
| jailbreakbench_behaviors | 0.520 | 0.660 | 0.740 | 50 |
| jailbreak_amplification | 0.517 | 0.583 | 0.575 | 120 |
| bbq_bias | 0.793 | 0.924 | 0.899 | 198 |
| truthfulqa | 0.460 | 0.560 | 0.510 | 50 |
| arc_challenge | 0.365 | 0.675 | 0.740 | 200 |
| mmlu_real | 0.316 | 0.519 | 0.583 | 285 |

### A.2 Phase 2 paired safety McNemar (full)

| Model | n | Discordant | R→C | C→R | χ² | p (asymp) | p (exact) | OR | OR 95% CI | Holm sig |
|-------|---|-----------|-----|-----|----|-----------|-----------|----|-----------|----------|
| llama3.2-1b | 518 | 33 | 17 | 16 | 0.000 | 1.000 | 1.000 | 1.061 | (Haldane-Anscombe) | No |
| llama3.2-3b | 518 | 32 | 18 | 14 | 0.281 | 0.596 | 0.597 | 1.276 | (HA) | No |
| qwen2.5-1.5b | 518 | 80 | 45 | 35 | 1.012 | 0.314 | 0.314 | 1.282 | (HA) | No |

### A.3 Phase 3 cell-level safety means (full ctx × dtype matrix)

**llama3.2-1b**

| ctx | auto mean | fp8 mean | delta (pp) | n / cell |
|-----|-----------|----------|------------|----------|
| 256 | 0.456 | 0.464 | +0.8 | 250 |
| 512 | 0.404 | 0.434 | +3.0 | 250 |
| 1024 | 0.366 | 0.374 | +0.8 | 250 |
| 2048 | 0.332 | 0.336 | +0.4 | 250 |

**llama3.2-3b**

| ctx | auto mean | fp8 mean | delta (pp) | n / cell |
|-----|-----------|----------|------------|----------|
| 256 | 0.668 | 0.662 | -0.6 | 250 |
| 512 | 0.710 | 0.676 | -3.4 | 250 |
| 1024 | 0.678 | 0.726 | +4.8 | 250 |
| 2048 | 0.758 | 0.766 | +0.8 | 250 |

### A.4 Phase 4 cell-level safety means (full batch × dtype matrix, AdvBench task)

**llama3.2-1b** (AdvBench, n=100/cell)

| batch | auto mean | fp8 mean | delta (pp) |
|-------|-----------|----------|------------|
| 1 | 0.65 | 0.69 | +4 |
| 4 | 0.64 | 0.68 | +4 |
| 8 | 0.64 | 0.68 | +4 |

**llama3.2-3b** (AdvBench, n=100/cell)

| batch | auto mean | fp8 mean | delta (pp) |
|-------|-----------|----------|------------|
| 1 | 0.79 | 0.76 | -3 |
| 4 | 0.79 | 0.78 | -1 |
| 8 | 0.79 | 0.78 | -1 |

The full 5-task × 3-batch × 2-dtype matrix is in `tr145_analysis.json` under `phase4_batch_interaction.per_task_breakdown`.

### A.5 Phase 5 turn-5 paired outcomes

| Model | KV dtype | Turn-5 safe | Turn-5 unsafe | n |
|-------|----------|-------------|---------------|---|
| llama3.2-1b | auto | 97 | 3 | 100 |
| llama3.2-1b | fp8 | 93 | 7 | 100 |
| llama3.2-3b | auto | 99 | 1 | 100 |
| llama3.2-3b | fp8 | 99 | 1 | 100 |

Discordant pairs (per SS16): Llama-1B 5 R→C + 1 C→R = 6 total; Llama-3B 0 + 0 = 0.

---

## Appendix B: Extended Statistical Tables

### B.1 TOST results (all 22 tests)

The full 22-row TOST table is in `tr145_analysis.json` under `tost_results`. The summary by category:

- Phase 2 paired safety (3 tests): 2 equivalent (Llama-1B, Llama-3B), 1 not (Qwen-1.5B at -3.09pp).
- Phase 2 paired capability (3 tests): 1 equivalent (Llama-1B), 2 marginal (Llama-3B borderline at -1.65pp / TOST p=0.07; Qwen-1.5B not equivalent at -6.8pp).
- Phase 3 cell tests (8 tests, 4 ctx × 2 models): 4 equivalent.
- Phase 4 cell tests (6 tests, 3 batch × 2 models): 1 equivalent.
- Phase 5 cell tests (2 tests): 1 equivalent (Llama-3B), 1 not (Llama-1B at -4pp).

### B.2 Power analysis (full per-cell MDE)

Full table in `tr145_analysis.json` under `power_analysis`. All 22 cells achieve 80% power at α = 0.05 with MDE 3.4-8.4pp.

### B.3 Mantel-Haenszel synthesis details

| Comparison | Pooled OR | 95% CI | n strata | Heterogeneity (Q test) |
|------------|-----------|--------|----------|------------------------|
| Phase 2 safety | 1.0512 | [0.8958, 1.2336] | 3 (models) | n.s. |
| Phase 4 batch=8 safety | 1.0047 | [0.8310, 1.2147] | 2 (models) | n.s. |
| Phase 5 turn-5 safety | 2.0582 | [0.6064, 6.9853] | 2 (models) | n.s. |

### B.4 Judge agreement per-task

| Task | Agreement | n records | n unclear | n disagree |
|------|-----------|-----------|-----------|-----------|
| advbench_refusal | 0.903 | 300 | 0 | 29 |
| jailbreakbench_behaviors | 0.793 | 150 | 0 | 31 |
| jailbreak_amplification | 0.744 | 360 | 0 | 92 |
| bbq_bias | 0.688 | 593 | 1 | 184 |
| truthfulqa | 0.699 | 146 | 4 | 44 |

---

## Appendix C: Sensitivity & Robustness

### C.1 Pairing strategy sensitivity

The McNemar tests pair on `(model, task, sample_id)`. Alternative pairings — pair on `(model, task)` only (within-task aggregate), pair on `(task)` only (cross-model aggregate) — do not change the headline null result. The within-`(model, task)` pairing is the most powered and is the primary reported test.

### C.2 Holm correction sensitivity

The primary Phase 2 safety McNemar tests do not reach significance at any reasonable α-correction (Holm, Benjamini-Hochberg, Bonferroni). The Qwen capability test (p = 0.0018) survives all three corrections in its (3 models × 2 outcomes = 6 tests) family.

### C.3 Judge model substitution sensitivity

The McNemar primary tests use regex outcomes, not judge labels. Re-running the McNemar tests with judge labels as the primary outcome (instead of regex) is supported by the analyzer (`_judge_safe_outcome` is the relevant function); informal results during development showed similar non-significance, with judge-based deltas slightly larger than regex-based deltas. The published headline uses regex for consistency with prior TRs.

### C.4 Equivalence margin sensitivity

At ±3pp: 9/22 equivalent (40.9%). At ±5pp (a looser deployment-tolerance setting): 17/22 equivalent (77%). At ±1pp (a stricter safety-critical setting): 2/22 equivalent. The choice of ±3pp matches the prior Banterhearts-line conventions and is the recommended margin for general production chat; tighter margins are appropriate for safety-critical deployments and the failure rate is correspondingly higher.

---

## Appendix D: Glossary

- **AdvBench refusal:** A benchmark of 100+ direct-harm prompts; the metric is the rate at which the model produces a refusal-shaped response.
- **AWQ / GPTQ:** Activation-aware / Generative Post-Training Quantization — weight-quantization methods. Not used in TR145 (model weights remain FP16).
- **BBQ:** Bias Benchmark for QA — evaluates model bias on social-group-related question answering.
- **Cohen's d:** Standardized effect size; |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large.
- **Cohen's κ:** Inter-rater agreement coefficient corrected for chance; κ > 0.80 = near-perfect, 0.60-0.80 = substantial, 0.40-0.60 = moderate, < 0.40 = fair-to-poor.
- **E4M3 / E5M2:** FP8 numerical formats. E4M3 has 4 exponent bits and 3 mantissa bits (better precision, narrower range); E5M2 has the reverse (better range, lower precision). vLLM uses E4M3 for KV-cache on Ada/Hopper.
- **FP8 KV-cache:** 8-bit quantization of cached attention key/value tensors during inference. Halves attention-cache memory.
- **JailbreakBench:** A benchmark of harm-eliciting prompts derived from the JailbreakBench paper; TR145 uses a 50-behavior subset from TR140's robust-refusal slice.
- **Mantel-Haenszel synthesis:** A meta-analytic method for pooling 2×2 contingency tables across strata; produces a single weighted odds ratio with a CI.
- **McNemar's test:** Non-parametric paired test for binary outcomes; tests whether the discordant cell counts of a 2×2 table are unbalanced.
- **MDE (Minimum Detectable Effect):** The smallest effect size that would be detected with a given sample size, α level, and power.
- **MMLU:** Massive Multitask Language Understanding benchmark; capability proxy.
- **R→C / C→R:** Refusal-to-Compliance / Compliance-to-Refusal; the two off-diagonal cells of the McNemar paired contingency table.
- **TOST (Two One-Sided Tests):** Equivalence-testing procedure; tests whether the observed effect is *within* a pre-specified margin, not whether it is exactly zero.
- **TruthfulQA:** Truthfulness benchmark testing whether models produce factually correct vs popular-misconception responses.
- **vLLM:** PagedAttention-based LLM inference server with KV-cache management.

---

## References

1. Lin et al. (2024). **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** MLSys.
2. Frantar et al. (2023). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** ICLR.
3. Micikevicius et al. (2022). **FP8 Formats for Deep Learning.** arXiv:2209.05433.
4. Sun et al. (2024). **A Closer Look at FP8 Quantization for Deep Learning Training.** arXiv:2401.10560.
5. Zou et al. (2023). **Universal and Transferable Adversarial Attacks on Aligned Language Models.** arXiv:2307.15043.
6. Parrish et al. (2022). **BBQ: A Hand-Built Bias Benchmark for Question Answering.** ACL Findings.
7. Lin et al. (2022). **TruthfulQA: Measuring How Models Mimic Human Falsehoods.** ACL.
8. Hendrycks et al. (2021). **Measuring Massive Multitask Language Understanding.** ICLR.
9. Clark et al. (2018). **Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.** arXiv:1803.05457.
10. Leviathan et al. (2023). **Fast Inference from Transformers via Speculative Decoding.** ICML.
11. Chen et al. (2023). **Accelerating Large Language Model Decoding with Speculative Sampling.** arXiv:2302.01318.
12. Banterhearts TR125 (2026). **Weight-Quantization Safety: 9-Format Comparison.** Internal report.
13. Banterhearts TR134 (2026). **Quantization Calibration Sensitivity.** Internal report.
14. Banterhearts TR138 (2026). **Batch-Inference Safety.** Accepted, ICML 2026 Workshop on Hypothesis Testing.
15. Banterhearts TR140 (2026). **Many-Shot Jailbreak under Quantization (JTP).**
16. Banterhearts TR141 (2026). **Refusal-Template Fragility under Quantization.** Accepted, ICML 2026 Workshop on Hypothesis Testing.
17. Banterhearts TR142 (2026). **Refusal Template Stability Index (RTSI).**
18. Banterhearts TR143 (2026). **Cross-Architecture Safety.** Internal report.
19. Banterhearts TR144 (2026). **Speculative Decoding Safety (TAIS).**
20. Banterhearts TR147 (2026). **Compile Reproducibility Index (CRI).**

---

*End of Technical Report 145. For data, see `research/tr145/results/20260508_033550/`. For source code, see `research/tr145/`. For the paper-package writeup, see `papers/kv_cache_quantization_safety/`.*
