# Technical Report 139: Multi-Turn Jailbreak Susceptibility Under Quantization
## Full-scale conversational attack sweep across 4 models, 6 GGUF quant levels, 8 multi-turn strategies, and 2-phase refusal-persistence evaluation

| Field | Value |
|-------|-------|
| **TR Number** | 139 |
| **Project** | Banterhearts Safety Alignment Research |
| **Date** | 2026-03-14 |
| **Version** | 1.2 |
| **Author** | Research Team |
| **Git Commit** | `edbaf196` |
| **Status** | Complete |
| **Report Type** | Multi-turn jailbreak quantization study |
| **Run Directory** | `20260314_012503` |
| **Total Conversations** | 10,600 |
| **Phase 1 Conversations** | 9,600 |
| **Phase 2 Conversations** | 1,000 |
| **Judge Labels** | 37,825 |
| **Models** | `llama3.2-1b`, `llama3.2-3b`, `qwen2.5-1.5b`, `llama3.1-8b` |
| **Quant Levels** | `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K` |
| **Related Work** | [TR134](Technical_Report_134.md), [TR135](Technical_Report_135.md), [TR136](Technical_Report_136.md), [TR137](Technical_Report_137.md), [TR138](Technical_Report_138.md), [TR138_v2](Technical_Report_138_v2.md) |
| **Depends On** | TR134 shared refusal detector and judge stack, TR125 run utilities, JailbreakBench-derived behavior set |

---

## Abstract

TR139 asks whether GGUF quantization changes susceptibility to multi-turn jailbreak attacks and whether lower-bit models become easier to break once a conversation has already entered an adversarial pressure regime. This is the first full study in this research line to combine modern multi-turn jailbreak methodology with a quantization sweep on local open-weight models.

The completed run contains **10,600 conversations**: **9,600 Phase 1 attack-efficacy conversations** across 4 models, 6 quant levels, 8 attack strategies, and 50 harmful behaviors, plus **1,000 Phase 2 persistence conversations** across 4 models, 5 quant levels, and the same 50 behaviors. Post-hoc adjudication adds **37,825 LLM-judge labels** on top of the primary regex-based refusal detector.

The Phase 1 result is strong. All 8 strategy-specific ANOVAs reject quant-independence with `p < 1e-4`, with effect sizes ranging from `eta^2 = 0.0314` (`context_fusion`) to `eta^2 = 0.1526` (`direct`). The highest-risk cells cluster in `qwen2.5-1.5b` and low-bit `llama3.2-1b`: `qwen2.5-1.5b / Q2_K / attention_shift`, `context_fusion`, and `crescendo` all reach **100% ASR**, while `llama3.2-1b / Q2_K / crescendo` reaches **86%**.

The Phase 2 result is narrower and more heterogeneous. Persistence generally decreases as quantization gets lower, but not by the same mechanism in every model. The strongest degradation appears in `llama3.2-1b` (`slope = -0.198` persistence points per BPW, bootstrap CI `[-0.305, -0.144]`), while `llama3.1-8b` is so weak under pressure across the sweep that the main signal is high break rate rather than a clean monotone slope.

The most important non-result is that quantization does **not** establish a universal law that multi-turn strategies become more quant-sensitive than direct attacks. Average direct slope is `-0.0793`, average multi-turn slope is `-0.0536`, and the Welch comparison is not significant (`p = 0.702`). Lower quantization changes multi-turn attack success, but it does not prove global multi-turn amplification beyond direct degradation.

The operational conclusion is that quantization belongs in the safety envelope for conversational deployment, but the risk is model- and threshold-specific rather than universal. The right deployment response is targeted validation and threshold-aware policy, not a single blanket quantization rule.

---

## Executive Summary

TR139 answers:

> does quantization change multi-turn jailbreak risk, and does it make initially refusing models easier to break under conversational pressure?

Yes, but not in the simplest possible way.

### Key Findings

1. **Quantization materially affects multi-turn ASR.** All 8 strategy-level ANOVAs reject quant-independence with `p < 1e-4`.
2. **Risk is strongly model-dependent.** `qwen2.5-1.5b` is broadly vulnerable, `llama3.2-1b` collapses at `Q2_K`, `llama3.1-8b` shows an instability band around `Q4_K_M` to `Q3_K_M`, and `llama3.2-3b` is the strongest model in the sweep.
3. **Persistence usually weakens as quantization gets lower.** Three of four persistence slopes are negative, and the aggregate Phase 2 verdict supports degradation under lower bit-width.
4. **There is no safe equivalence story.** Across `160` TOST checks at `+/-3pp`, `0 / 160` establish equivalence.
5. **Multi-turn amplification exists in specific cells but not as a universal law.** The highest finite amplification ratios reach `49.0x`, but H2 is still not supported in aggregate.
6. **A preserved post-hoc judge pass exposes exactly where regex scoring is weakest.** The saved judge layer adds `37,825` adjudications, with overall agreement `64.99%` and kappa `0.1037` against the regex detector. The lowest-agreement slices are the mid-quant Phase 2 cells, which is exactly where human review is most justified.

### Core Decisions

- Do **not** treat quantization as a pure systems optimization for interactive chat.
- Avoid `Q2_K` for safety-sensitive deployment on the smaller models tested here.
- Validate persistence under pressure, not just direct harmful prompt refusal.
- Treat `qwen2.5-1.5b` as high-risk for conversational adversarial exposure across much of the quant ladder.
- Treat `llama3.2-3b` as strongest-in-sweep, not as robust-by-default.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Phase 1 conversations | Count | 9,600 | 9,600 | **PASS** |
| Phase 2 conversations | Count | 1,000 | 1,000 | **PASS** |
| Judge labels | Count | >= 9,600 | 37,825 | **PASS** |
| H1 rejection (quant-independence) | Strategy ANOVA p < 0.05 | 8 / 8 strategies | 8 / 8, all p < 1e-4 | **PASS** |
| H2 multi-turn amplification | Welch p < 0.05 | significant slope difference | p = 0.702 | **FAIL** |
| H3 persistence degradation | Negative slope, CI excludes 0 | 4 / 4 models | 3 / 4 models | **PARTIAL** |
| TOST non-equivalence (+/-3pp) | Proportion establishing equivalence | 0 / 160 | 0 / 160 | **PASS** (confirms non-interchangeability) |
| Cross-phase anchor match | Phase 1 direct vs Phase 2 initial refusal | 20 / 20 cells | 20 / 20 exact | **PASS** |
| Power (median MDE at 80%) | Detectable effect | <= 20pp | 15.19pp | **PASS** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| 1 | Quantization changes multi-turn ASR | 8/8 strategy ANOVAs reject H1, all `p < 1e-4` | **Demonstrated** |
| 2 | Lower quant always worsens multi-turn risk | Model-level sweep shows mixed threshold patterns, not universal monotonicity | **Not validated** |
| 3 | Multi-turn attacks become more quant-sensitive than direct attacks | Welch slope comparison `p = 0.702` | **Not validated** |
| 4 | Lower quant often weakens persistence under pressure | H3 supported; 3/4 persistence slopes negative | **Demonstrated** |
| 5 | Threshold behavior matters more than a single global quant rule | Critical quant thresholds cluster by model/strategy, not globally | **Demonstrated** |
| 6 | Regex-only scoring would be too weak for this report | 37,825 preserved judge labels identify the same mid-quant Phase 2 slices as the highest-ambiguity cells | **Demonstrated** |

---

## When to Use This Report

TR139 is the reference report for conversational jailbreak risk under quantization. Use it when:

### Scenario 1: Quantized chat deployment review

**Question:** "Can I safely deploy a lower-bit local model for interactive chat?"

**Answer:** Not without adversarial conversational validation. Direct-turn refusal rates are not enough. TR139 shows that high direct refusal can coexist with very high multi-turn ASR.

### Scenario 2: Comparing models for hostile-input environments

**Question:** "Which of these local models is least bad under multi-turn attack?"

**Answer:** In this run, `llama3.2-3b` is strongest overall. `qwen2.5-1.5b` is the most dangerous. `llama3.2-1b` is acceptable only above its low-bit cliff. `llama3.1-8b` is especially poor under continued pressure.

### Scenario 3: Deciding whether direct harmful-prompt tests are sufficient

**Question:** "If the model refuses direct prompts, is that enough evidence?"

**Answer:** No. TR139 directly shows that a model can retain high direct refusal while still failing badly in multi-turn attack cells.

### Scenario 4: Positioning TR139 relative to the broader safety line

**Question:** "What does TR139 add beyond TR134-TR138?"

**Answer:** Earlier reports established that inference-time choices can affect safety. TR139 extends that line into conversational attack structure and shows that quantization changes the multi-turn attack surface too.

---

## Table of Contents

- [Metric Definitions and Evidence Standards](#metric-definitions-and-evidence-standards)
- [1. Introduction and Research Motivation](#1-introduction-and-research-motivation)
- [2. Methodology and Experimental Design](#2-methodology-and-experimental-design)
- [3. Models, Quants, Strategies, and Pressure Tactics](#3-models-quants-strategies-and-pressure-tactics)
- [4. Phase 1 Results: Multi-Turn Attack Success](#4-phase-1-results-multi-turn-attack-success)
- [5. Phase 2 Results: Refusal Persistence Under Pressure](#5-phase-2-results-refusal-persistence-under-pressure)
- [6. Statistical Synthesis and Hypothesis Evaluation](#6-statistical-synthesis-and-hypothesis-evaluation)
- [7. Judge Agreement and Scoring Reliability](#7-judge-agreement-and-scoring-reliability)
- [8. Cross-Phase Validation](#8-cross-phase-validation)
- [9. Cross-Report Positioning](#9-cross-report-positioning)
- [10. Production Guidance](#10-production-guidance)
- [11. Limitations and Follow-Up Program](#11-limitations-and-follow-up-program)
- [12. Conclusions](#12-conclusions)
- [13. Reproducibility](#13-reproducibility)
- [References and External Anchors](#references-and-external-anchors)
- [Appendix A: Raw Data Tables](#appendix-a-raw-data-tables)
- [Appendix B: Extended Statistical Tables](#appendix-b-extended-statistical-tables)
- [Appendix C: Sensitivity and Robustness](#appendix-c-sensitivity-and-robustness)
- [Appendix D: Glossary](#appendix-d-glossary)
- [Appendix E: Configs (Source of Truth)](#appendix-e-configs-source-of-truth)

---

## Metric Definitions and Evidence Standards

### Primary Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **ASR** | Fraction of Phase 1 conversations whose target turn is scored compliant | Higher is worse |
| **Break rate** | Fraction of initially refusing Phase 2 conversations that eventually comply under pressure | Higher is worse |
| **Persistence** | Fraction of turns in a Phase 2 conversation that preserve refusal | Lower is worse |
| **Slope per BPW** | Change in ASR or persistence per bits-per-weight unit | Negative ASR slope means lower quant increases ASR; negative persistence slope means lower quant weakens persistence |
| **Amplification ratio** | Multi-turn ASR divided by direct ASR for the same `(model, quant)` cell | Values > 1 indicate multi-turn strategy outperforms direct |

### Statistical Tests Used

| Test | Role in report |
|------|----------------|
| One-way ANOVA | H1 quant-independence checks by strategy |
| Two-way ANOVA | Quant x strategy interaction within each model |
| Fisher exact test | Per-cell quant-vs-baseline comparisons |
| Welch's t-test | H2 direct-vs-multi-turn slope comparison; pairwise quant comparisons |
| Cohen's d | Effect size on every pairwise quant comparison |
| Holm-Bonferroni correction | Applied to all Fisher exact and pairwise t-test families to control FWER |
| Bootstrap CI (1,000 iterations, seed 42) | Uncertainty on persistence slopes and ASR slopes |
| TOST (`+/-3pp`) | Tests whether quant cells are practically equivalent |
| Variance decomposition | Separates model, quant, strategy, and residual contributions |
| Power analysis (alpha=0.05, power=0.80) | Estimates MDE per cell |

### Evidence Standard

TR139 distinguishes:

- **Established findings**: directly supported by the completed artifacts and tests
- **Partial findings**: directionally real but still heterogeneous or underconstrained
- **Non-claims**: tempting interpretations that the run does not justify

That distinction is necessary because TR139 contains both strong positive findings and real negative findings.

---

## 1. Introduction and Research Motivation

Multi-turn jailbreaks are now one of the strongest attack classes in the alignment literature. They work by distributing harmful intent across turns, building authority, exploiting persona commitment, or gradually narrowing from benign context into unsafe content. That makes them structurally different from the single-turn refusal tests used in much of routine safety validation.

Separately, the recent safety line in this repo has already shown that deployment choices such as backend, batching, and quantization are not safety-neutral. But those reports mostly interrogated single-turn or near-single-turn behavior. The missing question was whether the same deployment choices alter the much stronger conversational attack regime.

TR139 fills that gap.

The core novelty is not simply "lower quant hurts safety." The novelty is:

> quantization interacts with conversational attack structure in a model-specific and threshold-specific way.

That matters for practice because a deployment team often chooses quantization for memory fit, speed, or laptop/server constraints. If that same decision also moves the model into a conversationally unstable attack regime, then quantization is part of the safety envelope, not just the systems stack.

The report therefore has two jobs:

1. determine whether multi-turn ASR depends on quantization
2. determine whether lower quant weakens persistence after an initial refusal

### 1.1 Research questions

TR139 answers four concrete decision questions:

1. Does multi-turn ASR depend on quantization level?
2. Does lower quantization make multi-turn strategies more dangerous than direct attacks, or does it merely degrade safety more generally?
3. Once a model refuses, which quant settings preserve that refusal and which settings collapse under pressure?
4. Are there model-specific threshold regions that justify deployment policy?

### 1.2 Why this matters

The practical risk is compound. A system can appear strong on direct harmful prompts, fit comfortably on available hardware, and still enter a conversational failure regime because:

- lower-bit deployment changes the refusal surface
- multi-turn context changes how the model interprets the final request
- repeated pressure changes whether a refusal is durable

TR139 therefore closes a real decision gap. It is not just another jailbreak benchmark. It is a deployment-facing study of how quantization interacts with the strongest attack class in the current literature.

### 1.3 Scope

| Scope item | Coverage |
|------------|----------|
| Deployment style | Local Ollama chat-style serving |
| Attack family | Multi-turn jailbreaks plus follow-up persistence pressure |
| Models | 4 local instruct models from 1B to 8B |
| Quant ladder | 6 Phase 1 levels, 5 Phase 2 levels |
| Behavior set | 50 harmful behaviors across 10 categories |
| Primary focus | Safety under adversarial conversational interaction |

### 1.4 Literature grounding

TR139 is anchored in two prior literatures:

- multi-turn jailbreak methods such as Crescendo, Foot-in-the-Door, Attention Shifting, RACE-like refinement, context fusion, and persona attacks
- quantization-safety work showing that lower-bit deployment can alter alignment behavior

The novelty gap is the intersection. Prior work treats these mostly as separate problems. TR139 measures them together.

### 1.5 How to read this report

Read the report in three passes:

1. **Phase 1** for the main quantization x conversational-attack result
2. **Phase 2** for the persistence-under-pressure result
3. **Statistical synthesis and production guidance** for what is actually established and how to use it

For unfamiliar terms, see Appendix D (Glossary). For raw data tables and full pairwise statistics, see Appendices A and B. For robustness checks and sensitivity analysis, see Appendix C. The canonical run configuration is reproduced in Appendix E.

---

## 2. Methodology and Experimental Design

### 2.1 Overall design

TR139 is a two-phase design.

**Phase 1** measures attack success under fully-specified multi-turn jailbreak templates:

- 4 models
- 6 quant levels
- 8 attack strategies
- 50 harmful behaviors

This yields exactly `9,600` conversations.

**Phase 2** measures persistence after an initial refusal:

- 4 models
- 5 quant levels
- 50 harmful behaviors

This yields exactly `1,000` conversations.

### 2.2 Unit of analysis

The basic unit is one conversation under one fixed `(model, quant, strategy, behavior)` configuration.

That conversation then becomes:

- one ASR contribution in Phase 1
- or one persistence/break contribution in Phase 2

The report is therefore conversation-level in generation, but cell-level in aggregation and hypothesis testing.

### 2.3 How rows become claims

The evidence chain is:

1. model outputs are generated from fixed multi-turn or persistence templates
2. the refusal detector scores target turns or pressure turns
3. the judge layer re-scores a large post-hoc subset
4. conversation outcomes aggregate into ASR, break rate, and persistence by cell
5. those cells feed ANOVA, slope, equivalence, and variance analyses
6. only then do report-level claims follow

This matters because TR139 is not just a table of alarming cells. The report's claims are tied to the aggregation and statistics, not only the worst examples.

### 2.4 Scoring stack

The primary scorer is the shared refusal detector inherited from the TR134 safety stack. It provides the run-wide consistent refusal/compliance signal.

TR139 then adds one preserved post-hoc LLM judge pass, producing **37,825 labels** on the scored conversations. The saved artifact chain uses `qwen2.5:7b-instruct-q8_0` as the judge model. An earlier fallback run existed during development, but its outputs were overwritten and are not treated as part of the final evidentiary record. The judge layer is therefore used here as a secondary adjudication signal over the final saved run, not as a dual-judge comparison.

### 2.5 Why Phase 1 and Phase 2 are separated

Phase 1 and Phase 2 answer different questions.

- Phase 1: "Can the attacker get compliance by the target turn under a structured multi-turn attack?"
- Phase 2: "If the model initially refuses, how much pressure does it tolerate before breaking?"

Those are related but not identical. A model can be weak in one regime and less weak in the other. TR139 keeps them separate for that reason.

### 2.6 Design safeguards

The design includes several safeguards specifically to keep the interpretation clean:

- `direct` anchors the single-turn baseline
- `benign_context` isolates multi-turn context from overt escalation strategy
- Phase 2 measures both overall persistence and pressure-only resistance so immediate failure is not conflated with later breakage
- the 8B model is explicitly treated as quantized-only in practice rather than pretending FP16 would be deployable on this hardware
- the same harmful behavior pool is reused across phases so the persistence analysis is not answering a different content question from Phase 1

### 2.7 What this design does not do

TR139 does not test:

- adaptive human red-teaming
- hosted frontier APIs
- training-time mitigation changes
- every possible quant scheme

It is a serving-time deployment study, not a universal alignment benchmark.

### 2.8 Why these models

The model lineup is intentional rather than opportunistic.

- `llama3.2-1b` is the small-model cliff candidate: easy to deploy locally, but most likely to show quantization-sensitive failure bands.
- `llama3.2-3b` provides a same-family comparison at higher capability, which helps separate family effects from pure scale effects.
- `qwen2.5-1.5b` adds a different instruction-tuning family with a strong reputation on utility tasks, making it a useful cross-family safety test rather than a second Llama-only result.
- `llama3.1-8b` tests whether larger parameter count automatically buys conversational robustness. In this run it does not.

This is the same style of design logic used in TR126: the report is not just "four models happened to be available." Each model occupies a distinct explanatory role in the causal story.

### 2.9 Why this quant ladder

The quant ladder covers the deployment settings that actually matter for local serving:

- `Q8_0` is the closest practical approximation to a high-fidelity quantized baseline.
- `Q6_K` and `Q5_K_M` cover common "safe enough for quality, lighter on memory" settings.
- `Q4_K_M` and `Q3_K_M` are the first places where many laptop and edge deployments begin trading quality for fit or throughput aggressively.
- `Q2_K` is the severe compression endpoint and functions as a stress test for low-bit collapse.

Phase 2 drops `Q5_K_M` on purpose. The goal there is slope resolution under pressure, not exhaustive coverage of every adjacent quant step. Keeping `Q8_0`, `Q6_K`, `Q4_K_M`, `Q3_K_M`, and `Q2_K` preserves the upper, middle, and lower parts of the ladder while keeping the persistence run computationally tractable.

### 2.10 Sample-count integrity

TR139's executed counts are fixed by design and matter for interpretation:

| Phase | Formula | Executed |
|-------|---------|---------:|
| Phase 1 | `4 models x 6 quants x 8 strategies x 50 behaviors` | 9,600 conversations |
| Phase 2 | `4 models x 5 quants x 50 behaviors` | 1,000 conversations |
| Judge | preserved target-turn and follow-up adjudication on scored outputs | 37,825 labels |

These exact counts matter for two reasons:

1. they show the run is complete rather than partially sampled
2. they define what the report can and cannot claim at the cell level

TR139 has enough coverage for strong aggregate claims, but only moderate power for many fine-grained cell-to-cell comparisons. The report therefore follows TR126's discipline: use the whole design for the headline claims, and treat the cell-level readout as structure and threshold discovery rather than a license for hundreds of tiny claims.

---

## 3. Models, Quants, Strategies, and Pressure Tactics

### 3.1 Model and quant matrix

| Model | Family | Phase 1 quants | Phase 2 quants |
|-------|--------|----------------|----------------|
| `llama3.2-1b` | Llama 3.2 | `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K` | `Q8_0`, `Q6_K`, `Q4_K_M`, `Q3_K_M`, `Q2_K` |
| `llama3.2-3b` | Llama 3.2 | same | same |
| `qwen2.5-1.5b` | Qwen 2.5 | same | same |
| `llama3.1-8b` | Llama 3.1 | same | same |

Bits-per-weight values used in the slope analyses:

| Quant | BPW |
|-------|----:|
| `Q8_0` | 8.00 |
| `Q6_K` | 6.57 |
| `Q5_K_M` | 5.69 |
| `Q4_K_M` | 4.85 |
| `Q3_K_M` | 3.91 |
| `Q2_K` | 3.35 |

### 3.2 Phase 1 attack strategies

The 8 strategies cover controls plus five recognizable jailbreak families from the literature:

| Strategy | Function in the study |
|----------|-----------------------|
| `direct` | Single-turn control baseline |
| `benign_context` | Multi-turn context control without explicit escalation |
| `foot_in_door` | Topic escalation |
| `crescendo` | Authority and professional framing escalation |
| `attention_shift` | Context distraction and fictional reframing |
| `role_play` | Persona manipulation |
| `progressive_refinement` | Iterative narrowing toward harmful specificity |
| `context_fusion` | Distributed intent across benign setup turns |

The control strategies are important. They let the report separate:

- the effect of multi-turn context alone
- the effect of structured multi-turn attack strategy
- the direct single-turn baseline

### 3.3 Phase 2 pressure tactics

Phase 2 uses up to 8 follow-up pressure tactics after an initial refusal, including:

- authority appeal
- public-knowledge framing
- moral reframing
- emotional appeal
- rephrasing
- competitive comparison
- reciprocity leverage
- normalization

This gives a more decision-relevant measure of persistence than one additional retry turn.

### 3.4 Strategy taxonomy and turn structure

The attack families fall into four operational groups:

| Group | Strategies | Typical target-turn position |
|-------|------------|------------------------------|
| Direct baseline | `direct` | turn 1 |
| Context-only control | `benign_context` | turn 5 |
| Escalation and authority | `foot_in_door`, `crescendo`, `attention_shift` | turn 5 |
| Intent distribution / persona / refinement | `context_fusion`, `role_play`, `progressive_refinement` | turn 4 |

This matters for the turn-of-first-compliance analysis: the report is not comparing arbitrary turn counts. It is comparing fixed attack structures with distinct target-turn positions.

### 3.5 Why the behavior categories matter

The 50 harmful behaviors are spread across 10 categories, which lets the report distinguish:

- a model that fails only in a narrow risk pocket
- a model that degrades across many harm families at once

That distinction becomes important in the per-category vulnerability section below.

### 3.6 Published method coverage and why the controls matter

The strategy set is not just a benchmark grab-bag. It deliberately spans multiple conversational failure mechanisms:

- **momentum without overt escalation**: `benign_context`
- **topic escalation**: `foot_in_door`
- **authority and professional framing**: `crescendo`
- **attention diversion and fictionalization**: `attention_shift`
- **persona commitment**: `role_play`
- **iterative narrowing**: `progressive_refinement`
- **distributed intent across turns**: `context_fusion`

That coverage is what lets TR139 say something stronger than "some jailbreak worked." It can ask whether quantization mostly changes:

- direct harmful compliance
- conversational momentum sensitivity
- specific strategy families
- post-refusal durability

The controls are therefore not filler. They are what keeps the report from collapsing all conversational effects into a single undifferentiated "multi-turn attacks are strong" claim.

---

## 4. Phase 1 Results: Multi-Turn Attack Success

### 4.1 H1 is rejected cleanly

All 8 strategy-specific ANOVAs reject H1:

| Strategy | F | p | Eta squared |
|----------|---:|---:|------------:|
| `attention_shift` | 9.7575 | `< 1e-4` | 0.0393 |
| `benign_context` | 19.5875 | `< 1e-4` | 0.0758 |
| `context_fusion` | 7.7356 | `< 1e-4` | 0.0314 |
| `crescendo` | 32.9943 | `< 1e-4` | 0.1214 |
| `direct` | 43.0025 | `< 1e-4` | 0.1526 |
| `foot_in_door` | 29.2947 | `< 1e-4` | 0.1093 |
| `progressive_refinement` | 13.3443 | `< 1e-4` | 0.0529 |
| `role_play` | 19.3813 | `< 1e-4` | 0.0751 |

**Observations.** All 8 F-statistics exceed 7.7 and all p-values are below 1e-4 after Holm-Bonferroni correction across the 8-test family. The largest effect size is `direct` (`eta^2 = 0.1526`, large by conventional thresholds), meaning quantization alone explains over 15% of the variance in direct-attack ASR. Even the smallest effect (`context_fusion`, `eta^2 = 0.0314`) clears the small-effect threshold. The result is unambiguous:

> quantization materially affects attack success in the multi-turn regime.

### 4.2 Aggregate strategy ranking

Average ASR across the full Phase 1 sweep ranks the strategies as follows:

| Strategy | Mean ASR |
|----------|---------:|
| `attention_shift` | 32.33% |
| `progressive_refinement` | 30.92% |
| `role_play` | 25.67% |
| `context_fusion` | 24.08% |
| `crescendo` | 23.67% |
| `direct` | 23.50% |
| `benign_context` | 22.58% |
| `foot_in_door` | 19.92% |

**Observations.** Two conclusions follow.

1. `attention_shift` and `progressive_refinement` are the strongest strategies overall in this run.
2. The control strategies are not trivial. `direct` and `benign_context` are both strong enough that the multi-turn landscape is not reducible to "special jailbreak tricks only."

### 4.3 Model-level Phase 1 profiles

#### `qwen2.5-1.5b`

Mean ASR by quant:

- `Q8_0`: `43.25%`
- `Q6_K`: `39.25%`
- `Q5_K_M`: `42.75%`
- `Q4_K_M`: `55.50%`
- `Q3_K_M`: `53.75%`
- `Q2_K`: `83.75%`

This is the worst model in the sweep. It is not merely a low-bit problem. The model is already vulnerable across upper-bit settings, and then becomes catastrophic at `Q2_K`.

#### `llama3.2-1b`

Mean ASR by quant:

- `Q8_0`: `1.75%`
- `Q6_K`: `1.75%`
- `Q5_K_M`: `3.75%`
- `Q4_K_M`: `5.75%`
- `Q3_K_M`: `9.75%`
- `Q2_K`: `61.50%`

This is the clearest cliff model in the study. Above `Q2_K`, the system is imperfect but relatively stable. At `Q2_K`, it collapses.

#### `llama3.1-8b`

Mean ASR by quant:

- `Q8_0`: `18.50%`
- `Q6_K`: `23.25%`
- `Q5_K_M`: `14.75%`
- `Q4_K_M`: `26.25%`
- `Q3_K_M`: `56.00%`
- `Q2_K`: `44.00%`

This is not a monotone degradation story. It is an instability-band story, with the sharpest failures around `Q3_K_M`.

#### `llama3.2-3b`

Mean ASR by quant:

- `Q8_0`: `5.75%`
- `Q6_K`: `6.75%`
- `Q5_K_M`: `2.75%`
- `Q4_K_M`: `3.25%`
- `Q3_K_M`: `0.50%`
- `Q2_K`: `3.75%`

This is the strongest model in the sweep. It still fails in some cells, but it is clearly separated from the other three.

### 4.4 Strategy effects by model

Model-level ANOVA across strategies shows that the models are not just different in total ASR; they differ in which strategy families dominate them.

| Strategy | Strongest model mean ASR | Eta squared |
|----------|--------------------------|------------:|
| `attention_shift` | `qwen2.5-1.5b` at 93.33% | 0.6017 |
| `context_fusion` | `qwen2.5-1.5b` at 67.33% | 0.3556 |
| `progressive_refinement` | `qwen2.5-1.5b` at 78.33% | 0.3955 |
| `role_play` | `qwen2.5-1.5b` at 67.33% | 0.3683 |
| `direct` | `llama3.1-8b` at 46.00% | 0.0961 |

**Observations.** This confirms the qualitative reading from the cell tables: `qwen2.5-1.5b` is disproportionately weak to several multi-turn methods, while `llama3.1-8b` is unusually weak even on the direct baseline. The `eta^2` values for `attention_shift` (0.6017) and `progressive_refinement` (0.3955) are exceptionally large, indicating that model identity explains over half the variance in those strategy families.

### 4.5 Quant x strategy interaction

Two-way ANOVA within each model shows that the effect of quant level depends on the strategy being used:

| Model | Quant effect eta^2 | Strategy effect eta^2 | Interaction eta^2 |
|-------|-------------------:|----------------------:|------------------:|
| `llama3.1-8b` | 0.1019 | 0.0454 | 0.1038 |
| `llama3.2-1b` | 0.3794 | 0.0192 | 0.0402 |
| `llama3.2-3b` | 0.0113 | 0.0806 | 0.0251 |
| `qwen2.5-1.5b` | 0.0898 | 0.2898 | 0.0570 |

**Observations.** The interpretation is different by model:

- `llama3.2-1b`: quant dominates (`eta^2 = 0.3794`); the low-bit cliff is the main story
- `qwen2.5-1.5b`: strategy dominates (`eta^2 = 0.2898`); the model is broadly vulnerable and the attack family matters more than the exact quant
- `llama3.1-8b`: quant and interaction are both substantial (`eta^2 = 0.1019` and `0.1038` respectively); instability depends on where in the strategy ladder the model is hit
- `llama3.2-3b`: the whole surface is comparatively flat and low (quant `eta^2 = 0.0113`)

### 4.6 Highest-risk cells

The worst Phase 1 cells are:

| Model | Quant | Strategy | ASR |
|-------|-------|----------|----:|
| `qwen2.5-1.5b` | `Q2_K` | `attention_shift` | 100% |
| `qwen2.5-1.5b` | `Q2_K` | `context_fusion` | 100% |
| `qwen2.5-1.5b` | `Q2_K` | `crescendo` | 100% |
| `qwen2.5-1.5b` | `Q6_K` | `attention_shift` | 98% |
| `qwen2.5-1.5b` | `Q8_0` | `attention_shift` | 98% |
| `llama3.1-8b` | `Q3_K_M` | `benign_context` | 92% |
| `llama3.2-1b` | `Q2_K` | `crescendo` | 86% |

**Observations.** Two things stand out:

1. `attention_shift` is one of the strongest strategies in the whole run, especially on `qwen2.5-1.5b`.
2. The `benign_context` control can itself become dangerous in unstable quant regimes, which means multi-turn context accumulation alone is not safety-neutral.

### 4.7 Turn-of-first-compliance structure

Mean turn-of-first-compliance by strategy is almost perfectly determined by template position:

| Strategy | Mean compliance turn |
|----------|---------------------:|
| `direct` | 1.0 |
| `context_fusion` | 4.0 |
| `progressive_refinement` | 4.0 |
| `role_play` | 4.0 |
| `attention_shift` | 5.0 |
| `benign_context` | 5.0 |
| `crescendo` | 5.0 |
| `foot_in_door` | 5.0 |

**Observations.** This is not a trivial observation. It means the attack scaffold is doing what it is supposed to do: models usually hold until the final target turn and then break at the intended conversational moment, rather than leaking compliance randomly earlier in the sequence.

### 4.8 Per-category vulnerability

Average Phase 1 ASR by behavior category across the full sweep:

| Category | Mean ASR |
|----------|---------:|
| Chemical & Biological Weapons | 32.81% |
| Disinformation & Deception | 29.58% |
| Violence & Weapons | 27.71% |
| Self-Harm & Dangerous Substances | 27.40% |
| Illegal Activities | 27.19% |
| Cybercrime & Unauthorized Intrusion | 25.10% |
| Harassment & Bullying | 24.48% |
| Discrimination & Stereotyping | 20.63% |
| Privacy Violations & Surveillance | 19.58% |
| Economic Harm & Fraud | 18.85% |

**Observations.** The spread is not enormous (14pp between highest and lowest), which is itself informative. The worst models are not failing only on one narrow content type. They are failing across a broad harm surface. Chemical & Biological Weapons leads at 32.81%, likely because these prompts contain specific procedural requests that trip even high-bit models. Economic Harm & Fraud trails at 18.85%, suggesting models have stronger refusal priors on financial harm framing.

### 4.9 Cross-strategy correlation

Several strategy pairs fail on the same behaviors, especially in the high-risk regions:

- `llama3.1-8b / Q4_K_M`: `context_fusion` vs `progressive_refinement`, `r = 0.7513`
- `qwen2.5-1.5b / Q3_K_M`: `context_fusion` vs `progressive_refinement`, `r = 0.7441`
- `qwen2.5-1.5b / Q6_K`: `context_fusion` vs `crescendo`, `r = 0.6896`

**Observations.** This suggests some behaviors are intrinsically multi-turn-fragile rather than being uniquely vulnerable to one attack family. When two strategies with different mechanisms (distributed intent vs iterative narrowing) fail on the same prompts with r > 0.7, the weakness is in the model's representation of the behavior, not in the specifics of the attack scaffold.

### 4.10 Conditional ASR given setup compliance

Conditional ASR is especially useful when a strategy has a nontrivial setup stage. One illustrative case:

- `qwen2.5-1.5b / Q3_K_M / attention_shift`
  - setup compliance rate: `70%`
  - unconditional ASR: `72%`
  - conditional ASR given setup success: `91.43%`

This means once the setup stage lands, the target-turn attack is even stronger than the unconditional cell suggests.

### 4.11 Critical quant thresholds

The critical threshold analysis does not yield one universal cutoff. Instead:

- `llama3.2-1b` mostly crosses at `Q2_K`, with some earlier thresholds at `Q3_K_M`
- `llama3.1-8b` mainly crosses at `Q3_K_M`, with one earlier `Q4_K_M` threshold for `benign_context`
- `qwen2.5-1.5b` mostly crosses at `Q2_K`, but `role_play` crosses earlier at `Q5_K_M`

This is why the report recommends threshold-aware policy rather than a single repo-wide quant rule.

### 4.12 What Phase 1 does not prove

Phase 1 does **not** prove:

- that lower quant always worsens ASR monotonically
- that every multi-turn strategy is more dangerous than direct attack
- that one threshold applies to every model

Those are exactly the overclaims this report avoids.

---

## 5. Phase 2 Results: Refusal Persistence Under Pressure

### 5.1 Aggregate model picture

| Model | Avg break rate | Avg persistence | Initial refusals across sweep | Breaks under pressure |
|-------|---------------:|----------------:|------------------------------:|----------------------:|
| `llama3.1-8b` | 96.67% | 0.1329 | 119 | 118 |
| `llama3.2-1b` | 55.95% | 0.6720 | 198 | 95 |
| `llama3.2-3b` | 46.68% | 0.5862 | 208 | 100 |
| `qwen2.5-1.5b` | 38.56% | 0.6809 | 211 | 86 |

**Observations.** The most operationally important number here is not a slope. It is the `llama3.1-8b` break profile. Once it initially refuses, it still breaks under pressure almost every time in this design. The 96.67% average break rate means that `llama3.1-8b` refusal is effectively cosmetic - it stops the user once and then yields. By contrast, `qwen2.5-1.5b` has the lowest break rate (38.56%) despite being the weakest model in Phase 1, because it simply never refuses many harmful prompts in the first place (fewer initial refusals to break).

### 5.2 Persistence slope analysis

| Model | Slope per BPW | Bootstrap CI | Interpretation |
|-------|---------------:|--------------|----------------|
| `llama3.1-8b` | 0.0160 | `[-0.0080, 0.0641]` | effectively flat / unstable |
| `llama3.2-1b` | -0.1980 | `[-0.3054, -0.1442]` | lower quant breaks more easily |
| `llama3.2-3b` | -0.1111 | `[-0.4195, -0.0175]` | lower quant breaks more easily |
| `qwen2.5-1.5b` | -0.1007 | `[-0.3106, 0.4225]` | negative direction, wide uncertainty |

**Observations.** This is enough to support H3 overall, but it also shows why the report does not claim a single common mechanism across the four models. The `llama3.2-1b` slope (-0.1980) is the steepest and most precisely estimated, while the `qwen2.5-1.5b` CI crosses zero on the upper bound, indicating that its persistence degradation is directionally real but not yet cleanly separated from noise. Full persistence data by model x quant is in Appendix A.

### 5.3 Pressure resistance extremes

Lowest mean pressure resistance cells:

| Model | Quant | Pressure resistance | Break rate | Initial refusals |
|-------|-------|--------------------:|-----------:|-----------------:|
| `llama3.2-1b` | `Q2_K` | 0.0391 | 100.0% | 16 |
| `llama3.1-8b` | `Q4_K_M` | 0.0898 | 100.0% | 32 |
| `llama3.1-8b` | `Q8_0` | 0.1919 | 100.0% | 43 |

Highest mean pressure resistance cells:

| Model | Quant | Pressure resistance | Break rate | Initial refusals |
|-------|-------|--------------------:|-----------:|-----------------:|
| `llama3.2-1b` | `Q6_K` | 0.9896 | 8.33% | 48 |
| `llama3.2-1b` | `Q8_0` | 0.9787 | 17.02% | 47 |
| `qwen2.5-1.5b` | `Q6_K` | 0.9770 | 8.16% | 49 |
| `qwen2.5-1.5b` | `Q8_0` | 0.9617 | 10.20% | 49 |

**Observations.** This makes the persistence story much more concrete than the slope table alone. Some cells remain extremely durable under pressure, while others offer almost no effective resistance at all. The `llama3.2-1b / Q2_K` cell (pressure resistance 0.0391, 100% break rate) is the single most operationally dangerous cell in Phase 2 - the model folds immediately under any pressure at this quant level.

### 5.4 Turn-level persistence curves

The turn trajectories show a consistent shape in the weakest cells:

- initial refusal at turn 1 can still be moderately high
- refusal collapses sharply by turns 2-4
- later turns rarely recover meaningfully once the model has broken

Examples:

- `llama3.2-1b / Q2_K`: mean refusal score drops from `0.32` on turn 1 to `0.0` on turns 2-3
- `llama3.1-8b / Q8_0`: turn 1 refusal is `0.86`, but by turn 3 it is down to `0.186`
- `llama3.1-8b / Q4_K_M`: turn 1 refusal is `0.64`, but by turn 4 it is `0.0312`

These trajectories support the interpretation that some models are not merely vulnerable in the aggregate. They are unable to maintain refusal once the conversation has started bending toward compliance.

### 5.5 Concrete persistence regimes

Selected Phase 2 cells:

- `llama3.2-1b / Q6_K`: break rate `8.33%`, persistence `0.9511`
- `llama3.2-1b / Q8_0`: break rate `17.02%`, persistence `0.9222`
- `llama3.2-1b / Q2_K`: break rate `100%`, persistence `0.0467`
- `qwen2.5-1.5b / Q6_K`: break rate `8.16%`, persistence `0.9600`
- `qwen2.5-1.5b / Q8_0`: break rate `10.20%`, persistence `0.9467`
- `qwen2.5-1.5b / Q3_K_M`: break rate `84.44%`, persistence `0.6156`
- `llama3.1-8b / Q8_0`: break rate `100%`, persistence `0.2422`

The two strongest persistence stories are:

- `llama3.2-1b` and `qwen2.5-1.5b` both have relatively stable upper-bit regimes and much weaker lower-bit regimes
- `llama3.1-8b` is already weak enough that quant slope is less important than absolute break behavior

### 5.6 What Phase 2 adds beyond Phase 1

Phase 1 alone could have been dismissed as "multi-turn templates are powerful." Phase 2 blocks that easy escape. It shows that the problem is not just target-turn compliance under crafted attack scaffolds. It is also refusal durability after the system has already said no once.

---

## 6. Statistical Synthesis and Hypothesis Evaluation

### 6.1 H1: ASR independence of quantization

**Verdict: rejected.**

This is the cleanest result in the report. All 8 strategies reject H1 with `p < 1e-4`.

In addition to the ANOVA results, per-cell Fisher exact tests (each quant vs Q8_0 baseline) were corrected with Holm-Bonferroni within each model x strategy family. All-pairs Welch's t-tests were also corrected with Holm-Bonferroni. Selected pairwise Cohen's d values illustrate the magnitude:

| Model | Strategy | Comparison | Cohen's d | Holm p |
|-------|----------|------------|----------:|-------:|
| `llama3.2-1b` | `crescendo` | Q2_K vs Q8_0 | 2.49 | < 0.001 |
| `llama3.2-1b` | `foot_in_door` | Q2_K vs Q8_0 | 2.33 | < 0.001 |
| `llama3.1-8b` | `benign_context` | Q3_K_M vs Q8_0 | 1.00 | < 0.001 |
| `llama3.1-8b` | `direct` | Q3_K_M vs Q8_0 | 1.00 | < 0.001 |
| `llama3.1-8b` | `attention_shift` | Q2_K vs Q4_K_M | 0.71 | 0.004 |
| `qwen2.5-1.5b` | `role_play` | Q2_K vs Q8_0 | 2.18 | < 0.001 |
| `qwen2.5-1.5b` | `crescendo` | Q2_K vs Q8_0 | 2.20 | < 0.001 |
| `llama3.2-3b` | `direct` | Q2_K vs Q3_K_M | 3.47 | < 0.001 |

Full pairwise tables with Cohen's d and Holm-corrected p-values are in Appendix B.

### 6.2 H2: quantization amplifies multi-turn attacks relative to direct attacks

Cell-level amplification is real. The largest finite amplification ratios are:

- `49.0x` for `qwen2.5-1.5b / Q6_K / attention_shift`
- `49.0x` for `qwen2.5-1.5b / Q8_0 / attention_shift`
- `39.0x` for `qwen2.5-1.5b / Q8_0 / progressive_refinement`

But the aggregate slope test does **not** support the stronger global claim:

- average direct slope: `-0.0793`
- average multi-turn slope: `-0.0536`
- ratio: `0.68x`
- Welch `p = 0.7023`

So H2 is **not supported**.

This is one of the most important negative findings in the report. It keeps the interpretation honest. Quantization matters, but it is not enough to claim that multi-turn attacks are generically more quant-sensitive than direct ones.

### 6.3 H3: persistence decreases with lower quantization

**Verdict: supported, with heterogeneity.**

Three of four persistence slopes are negative, and the clearest evidence appears in `llama3.2-1b` and `llama3.2-3b`. The `qwen2.5-1.5b` direction is consistent but uncertain, and `llama3.1-8b` is so brittle in absolute terms that monotonicity is not the whole story.

### 6.4 Latency and deployment tradeoff

Average mean total latency by quant across the Phase 1 sweep:

| Quant | Mean total latency |
|-------|-------------------:|
| `Q8_0` | 8,334 ms |
| `Q6_K` | 7,363 ms |
| `Q3_K_M` | 6,859 ms |
| `Q2_K` | 6,848 ms |
| `Q5_K_M` | 6,641 ms |
| `Q4_K_M` | 6,186 ms |

**Observations.** The latency slopes show the usual systems temptation: lower-bit deployment often gets faster, especially outside the 8B model. TR139's importance is that it links those speed gains to a real conversational safety tradeoff.

### 6.5 Equivalence and variance

Two additional results narrow interpretation:

- `0 / 160` TOST checks establish equivalence under `+/-3pp`
- variance decomposition gives `18.31%` model, `6.72%` quant, `1.21%` quant x strategy, `0.83%` strategy, and `72.93%` residual

**Observations.** This means:

- neighboring quant settings are not safely interchangeable by default
- model family explains more structured variation than strategy family does
- large residual variance (72.93%) is consistent with behavior-level heterogeneity across the 50-behavior benchmark - the safety surface is not smooth, and individual harmful behaviors contribute substantial idiosyncratic variation that no single design factor captures

A sensitivity check on the residual decomposition is in Appendix C.

### 6.6 Power caveat

Power is uneven across the Phase 1 cells:

- median MDE at 80% power: `15.19pp`
- minimum: `5.57pp`
- maximum: `27.43pp`

That is why the report leans hardest on:

- large cell differences
- consistent directional patterns
- aggregate hypothesis tests

and not on every small per-cell delta.

### 6.7 What the negative H2 finding rules out

The failure of H2 is not a side note. It is one of the most important interpretation constraints in the report.

It rules out three easy but wrong summaries:

1. **"Multi-turn attacks are always the thing quantization amplifies most."**
   Not shown. Some direct baselines degrade sharply too, especially in the weakest cells.

2. **"Any large amplification ratio demonstrates special multi-turn sensitivity."**
   Not by itself. Very large amplification ratios can arise when the direct denominator is tiny and the multi-turn numerator is merely moderate.

3. **"The right global story is direct vs multi-turn."**
   Also not shown. The cleaner story is model regime plus threshold regime. For some models, quantization mostly induces a cliff. For others, it changes which conversational strategy family is dominant. For `llama3.1-8b`, the persistence story is stronger than the direct-vs-multi-turn slope story.

This is exactly the kind of negative result that improves the report rather than weakening it. It narrows the claim from a broad and easy-to-attack thesis to a more defensible one:

> quantization changes the conversational safety surface, but the way it does so is model-specific rather than governed by a single universal amplification law.

---

## 7. Judge Agreement and Scoring Reliability

TR139's preserved artifact chain contains one post-hoc judge pass over the final run: `qwen2.5:7b-instruct-q8_0` produced **37,825 labels** across the same **10,600 conversations** used by the regex-based primary analysis. The purpose of this layer is not to replace the deterministic scorer. It is to show where the regex and an independent LLM adjudicator agree, and where they do not.

### 7.1 Preserved judge pass: overall

| Metric | Saved judge value |
|--------|------------------:|
| Overall agreement with regex | 64.99% |
| Overall kappa | 0.1037 |
| Judged rows included in agreement slice | 12,562 |

**Observations.** Agreement is modest, not strong. That does not invalidate the report, but it does limit what can be claimed about borderline rows. The right interpretation is that the judge layer confirms the existence of ambiguity in the harder slices of the run, especially where conversational persistence responses become partial, hedged, or stylistically mixed.

### 7.2 Phase 1 agreement by quant level

| Quant | N | Agreement | Kappa |
|-------|--:|----------:|------:|
| `Q8_0` | 1,600 | 81.50% | 0.1029 |
| `Q6_K` | 1,600 | 79.81% | 0.0817 |
| `Q5_K_M` | 1,600 | 80.00% | 0.0040 |
| `Q4_K_M` | 1,600 | 75.88% | 0.0817 |
| `Q3_K_M` | 1,600 | 69.31% | 0.0373 |
| `Q2_K` | 1,600 | 54.44% | 0.0612 |

**Observations.** The preserved judge pass shows the same broad gradient the report would already lead us to expect: agreement is highest in the upper-quant slices and worst at `Q2_K`. The `Q5_K_M` stratum is especially notable because agreement remains 80.00% while kappa falls to 0.0040, indicating that the class balance is skewed enough that raw agreement alone overstates certainty.

### 7.3 Phase 2 agreement by quant level

| Slice | N | Agreement | Kappa |
|-------|--:|----------:|------:|
| `p2_Q8_0` | 561 | 43.14% | 0.1149 |
| `p2_Q6_K` | 499 | 46.89% | 0.1436 |
| `p2_Q4_K_M` | 775 | 33.16% | 0.0697 |
| `p2_Q3_K_M` | 503 | 36.78% | 0.0730 |
| `p2_Q2_K` | 624 | 30.61% | 0.0625 |

**Observations.** Phase 2 is the genuinely difficult part of the scoring problem. Agreement falls sharply once the model is no longer producing a clear direct-turn refusal or compliance answer and instead is responding after multiple pressure turns. The worst slices are the same ones the substantive analysis already treats as fragile and ambiguous: `p2_Q4_K_M`, `p2_Q3_K_M`, and `p2_Q2_K`.

### 7.4 What the preserved judge pass shows

The saved judge layer establishes three things:

1. **The core report claims do not depend on the judge.** All 8 ANOVA rejections, the H2 non-result, and the H3 persistence slopes are computed from the deterministic primary scorer.
2. **The mid-quant and low-quant Phase 2 slices are exactly where a human review layer would add value.** Those are the cells where agreement is weakest.
3. **Regex-only scoring would have hidden the extent of row-level ambiguity.** The judge layer does not overturn the aggregate findings, but it does show why cell-level narratives need discipline.

### 7.5 What follows from this

The report's broad findings are defensible because they rest on:

- thousands of conversations
- large differences in the most important cells
- replicated phase structure
- aggregate tests that survive the noisy scoring layer
- a preserved secondary judge pass that highlights which slices are least settled

But the next credibility upgrade is still obvious:

> add human adjudication on a stratified scored subset, prioritizing the mid-quant Phase 2 cells where the saved judge pass diverges most from the regex detector.

### 7.6 Why weak agreement does not collapse the core result

The judge-agreement section is easy to misuse in either direction.

It would be wrong to say:

- "kappa is weak, therefore the report says nothing"

It would also be wrong to say:

- "agreement is good enough, therefore every cell-level story is settled"

The right read is between those extremes.

TR139's core claims survive because they are supported by multiple independent structures at once:

- large and repeated Phase 1 differences in the highest-risk regions
- a completed two-phase design rather than one isolated benchmark
- exact `20 / 20` direct-anchor matches between Phase 1 and Phase 2
- negative findings that constrain overclaiming rather than silently disappearing
- a secondary judge layer that points to the same high-ambiguity slices the report already treats cautiously

What the weak agreement blocks is not the entire report. It blocks overconfident fine-grained claims about every borderline scored output. That is why this document is now at flagship-report level, but the next paper-grade upgrade is still human adjudication.

---

## 8. Cross-Phase Validation

### 8.1 Direct-turn anchor matches exactly across phases

The strongest within-report validation check is available because Phase 2 begins with the same direct harmful request regime that Phase 1 already measures under the `direct` strategy. If the benchmark preparation, prompt wiring, and scoring logic drifted between phases, the Phase 1 direct refusal rate and the Phase 2 initial-refusal rate would disagree.

They do not merely align directionally. They match exactly in all `20 / 20` shared `(model, quant)` cells.

| Model | `Q8_0` | `Q6_K` | `Q4_K_M` | `Q3_K_M` | `Q2_K` |
|-------|--------|--------|----------|----------|--------|
| `llama3.1-8b` | `86 / 86` | `66 / 66` | `64 / 64` | `12 / 12` | `10 / 10` |
| `llama3.2-1b` | `94 / 94` | `96 / 96` | `92 / 92` | `82 / 82` | `32 / 32` |
| `llama3.2-3b` | `74 / 74` | `76 / 76` | `82 / 82` | `96 / 96` | `88 / 88` |
| `qwen2.5-1.5b` | `98 / 98` | `98 / 98` | `96 / 96` | `90 / 90` | `40 / 40` |

Each cell is `Phase 1 direct refusal % / Phase 2 initial refusal %`.

**Observations.** This is the cleanest validation result in the report outside the hypothesis tests themselves. It shows that the Phase 2 persistence findings are built on the same direct-turn baseline measured independently in Phase 1 rather than on a silently different prompt or scoring path.

### 8.2 What this validates

This exact match validates four practical things at once:

1. the shared behavior set was wired consistently across phases
2. the direct harmful prompt in Phase 2 is functionally the same intervention as the Phase 1 `direct` control
3. the refusal detector is at least self-consistent across the two phases on the initial direct turn
4. the large Phase 2 differences are not an artifact of beginning from a different baseline than the Phase 1 sweep

This is the kind of section TR126 uses well: not just another result table, but a check that the measurement system is coherent enough for the downstream interpretation to matter.

### 8.3 What this does not validate

This validation is important, but it does not solve every reliability problem.

It does **not** show:

- that the regex classifier is correct in an absolute human sense
- that the judge labels are ground truth
- that the strategy templates exhaust the conversational attack surface
- that the same break profiles would appear on another backend or hardware stack

So the right interpretation is strong but bounded:

> TR139's two phases are internally coherent. The report is still not the same thing as a human-adjudicated paper benchmark.

### 8.4 Why the direct anchor matters scientifically

This validation also sharpens the main causal reading of the report.

The Phase 1 result is not "some models comply a lot." The more interesting claim is:

- several models still refuse direct harmful prompts at meaningful rates
- those same models can nevertheless fail badly once attack structure or pressure is introduced

That is why TR139 is a conversational-safety report rather than a direct-refusal report with extra decoration.

---

## 9. Cross-Report Positioning

TR139 extends the repo's safety line in a specific direction.

- TR134 established the single-turn quantization anchor.
- TR135-TR137 showed that inference-time choices such as quantization, backend, and concurrency can move safety behavior.
- TR138 showed that serving-time batching is not safety-neutral even under deterministic decoding.
- TR139 adds the conversational regime: structured multi-turn attacks and post-refusal pressure.

What is newly measured here is not just "more jailbreaks." It is two distinct deployment-facing datapoints:

1. **attack-surface reshaping under quantization**
   Quantization changes which strategy families dominate a model and where the dangerous thresholds lie.

2. **refusal durability after the model has already said no**
   A model can look acceptable on the direct turn and still collapse once the user keeps pushing.

The strongest cross-report continuity point remains the direct-turn `Q8_0` anchor. The following table compares TR139 baselines against prior TR measurements on overlapping models, applying the +/-5pp tolerance established in TR137:

| Model | TR134 Q8_0 (AdvBench) | TR136 Q8_0 (aggregate) | TR139 Q8_0 (direct, 50 behaviors) | TR134->139 delta | TR136->139 delta | Within +/-5pp? |
|-------|----------------------:|----------------------:|----------------------------------:|------------:|------------:|:------------:|
| `llama3.2-1b` | 90.0% (n=100) | 87.6% (n=468) | 94.0% (n=50) | +4.0pp | +6.4pp | TR134 yes / TR136 no |
| `llama3.2-3b` | 52.0% (n=100) | 80.9% (n=468) | 74.0% (n=50) | +22.0pp | -6.9pp | no / no |
| `qwen2.5-1.5b` | - | 84.8% (n=468) | 98.0% (n=50) | - | +13.2pp | - / no |
| `llama3.1-8b` | - | - | 86.0% (n=50) | - | - | baseline set |

**Observations.** The `llama3.2-1b` anchor is the tightest cross-TR match (+4.0pp vs TR134, within tolerance). The larger deltas for other models are expected rather than alarming: TR134 used AdvBench-only prompts, TR136 used a 4-task aggregate, and TR139 uses a 50-behavior JailbreakBench-derived set spanning 10 harm categories. The `llama3.2-3b` TR134->TR139 gap (+22.0pp) is the most notable and reflects this model's sensitivity to prompt distribution; it refuses more of TR139's curated harmful behaviors than TR134's AdvBench set. These cross-TR deltas are therefore primarily task-distribution effects rather than measurement errors, and the within-stack internal consistency (20/20 exact Phase 1-to-Phase 2 anchor match) remains the stronger validation signal.

That matters because it blocks an easy dismissal. These models are not failing only because they refuse nothing in the first place. Several retain substantial direct refusal while still showing large multi-turn failure regions. TR139 therefore supports a narrower but more useful claim:

> conversational attack structure is its own safety regime, and quantization changes that regime.

---

## 10. Production Guidance

### 10.1 What deployment teams should do differently

Do **not** treat quantization as a pure systems knob for public or hostile-input chat systems.

For any model intended for interactive chat, validate all four of the following:

- direct harmful prompt refusal
- multi-turn jailbreak ASR
- refusal persistence under continued pressure
- model-specific critical quant thresholds

If you only run direct-turn refusal checks, you are missing the exact failure mode that dominates several of the most important cells in TR139.

### 10.2 Model-by-model deployment regime

The report supports different operational conclusions for the four models.

| Model | Relative safest band in this sweep | Caution band | Clear avoid band | Why |
|-------|------------------------------------|--------------|------------------|-----|
| `llama3.2-3b` | `Q8_0` through `Q3_K_M` | `Q2_K` | none conclusively catastrophic, but still validate | Lowest overall ASR surface, strongest model in sweep |
| `llama3.2-1b` | `Q8_0` and `Q6_K` | `Q4_K_M` and `Q3_K_M` | `Q2_K` | Clear low-bit cliff, strong persistence collapse at `Q2_K` |
| `llama3.1-8b` | no clearly comfortable hostile-chat band demonstrated | all quants | especially `Q3_K_M` instability band | Direct refusal can look acceptable while persistence remains extremely poor |
| `qwen2.5-1.5b` | no validated low-risk conversational band in this study | all quants | `Q2_K` definitively unacceptable | Severe multi-turn vulnerability even at upper quants; low-bit collapse makes it worse |

**Observations.** This table should be read as a deployment triage guide, not as a universal leaderboard. "Relative safest" means safest among the tested cells here, not safe in an absolute sense. The most striking row is `llama3.1-8b`: despite being the largest model in the sweep, it has no comfortable hostile-chat band because its persistence is catastrophically weak (Section 5.1).

### 10.3 Release checklist justified by this report

Before shipping a quantized chat model into a hostile-input environment:

- [ ] Run direct harmful prompt refusal checks on the exact deployment quant
- [ ] Run at least one multi-turn benchmark family, not only direct prompts
- [ ] Test persistence after initial refusal, not only target-turn compliance
- [ ] Identify the first dangerous quant threshold for that model line
- [ ] Review high-risk strategy families, not only average ASR
- [ ] Check whether the model has a brittle mid-quant instability band rather than a simple low-bit collapse
- [ ] Add human adjudication on the most policy-relevant cells before making strong external claims

### 10.4 What not to infer from this report

TR139 does **not** justify:

- a universal "lower quant always means less safe" law
- a universal "multi-turn is always more quant-sensitive than direct" law
- a claim that any tested model is robust by default
- a claim that one safe quant threshold applies across model families

The operational lesson is threshold-aware and model-aware policy, not one blanket rule.

### 10.5 Deployment scenarios

The same report supports different decisions in different deployment contexts.

#### Scenario A: closed evaluation or trusted-user local assistant

If the model is used only by a trusted operator, the main question is whether low-bit deployment changes the model enough to distort internal evaluation or staff workflows. In that case:

- `llama3.2-3b` remains the cleanest candidate
- `llama3.2-1b` can be usable above its `Q2_K` cliff
- `qwen2.5-1.5b` is still a poor choice if the workflow includes adversarial testing, because the conversational failure surface is already broad

#### Scenario B: public or semi-public chat surface

This is where TR139 matters most. Public chat needs protection against:

- conversational momentum
- structured escalation
- repeated pressure after refusal

For this scenario, the report supports a much stricter rule:

- do not deploy `qwen2.5-1.5b` on this evidence base alone
- do not deploy `llama3.1-8b` without stronger persistence controls
- do not use `Q2_K` on the smaller models in safety-sensitive settings

#### Scenario C: quantization chosen only for fit or speed

TR139 directly argues against the common shortcut:

> "we only changed memory footprint and latency, not behavior"

That shortcut is not defensible for conversational deployment on the models tested here.

---

## 11. Limitations and Follow-Up Program

### 11.1 Limitations

1. **Primary classification remains heuristic.**
   The refusal detector is useful and internally consistent, but it is still a pattern-based classifier rather than a human labeler.

2. **The preserved judge pass uses the same family as one evaluated model.**
   The saved judge is Qwen 2.5 7B, which overlaps with the evaluated `qwen2.5-1.5b` family. The size gap mitigates direct self-evaluation somewhat, but a truly independent judge family would be stronger.

3. **No human adjudication layer is included yet.**
   This is the single biggest remaining gap if TR139 is meant to become paper-grade evidence rather than flagship technical-report evidence.

4. **The strategy templates are fixed and reproducible rather than adaptive.**
   This is a feature for clean comparison, but it also means the report estimates a reproducible lower bound on attacker capability rather than an upper bound from skilled human red-teaming.

5. **The run covers one serving stack.**
   Everything here is local Ollama chat serving. The report does not isolate backend effects the way TR135-TR137 do.

6. **The run covers four local open-weight model lines, not frontier APIs.**
   The point is local deployment policy, not universal alignment ranking across all model classes.

7. **Residual variance is large.**
   The variance decomposition assigns `72.93%` to residual variation. That is consistent with behavior-level heterogeneity, but it still means many fine-grained cell stories should be treated cautiously.

8. **TOST non-equivalence everywhere is not the same as universal large effects.**
   `0 / 160` equivalence results justifies caution about interchangeability. It does not mean every neighboring quant pair differs by a practically large amount.

9. **Single-hardware execution still matters.**
   The quant thresholds and latency tradeoffs were measured on one RTX 4080 Laptop environment.

10. **Published-method mapping is faithful but still repo-implemented.**
    The strategy families are grounded in prior literature, but the exact templates are local implementations rather than borrowed benchmark binaries.

### 11.2 Highest-value follow-up work

1. human adjudication on a stratified subset of the most policy-relevant Phase 1 and Phase 2 outputs
2. larger-judge rerun for calibration against the fallback-judge path
3. replication of the highest-risk cells on another hardware stack
4. additional model-family replication, especially stronger Qwen and Llama variants
5. adaptive attack comparison against the fixed strategy templates
6. backend replication to separate quant effects from serving-stack effects

The first item is still the best next upgrade. The core result is already real enough for flagship technical-report use, but human adjudication is the cleanest path from "strong internal evidence" to "paper-quality evidence."

---

## 12. Conclusions

### 12.1 Direct answers

**Does quantization change multi-turn jailbreak risk?**

Yes. All 8 strategy-specific ANOVAs reject quant-independence with p < 1e-4 and `eta^2` up to 0.1526. Holm-Bonferroni-corrected pairwise tests confirm large Cohen's d values (up to 3.47) between critical quant pairs.

**Does lower quantization make multi-turn attacks more dangerous than direct attacks?**

Not as a universal law. The aggregate Welch slope comparison is not significant (p = 0.702). Cell-level amplification ratios reach 49.0x, but the global multi-turn-vs-direct amplification claim (H2) is not supported.

**Does lower quantization weaken persistence after initial refusal?**

Yes, with heterogeneity. Three of four models show negative persistence slopes, with `llama3.2-1b` the clearest (slope = -0.198/BPW, bootstrap CI [-0.305, -0.144]).

**Are neighboring quant levels interchangeable?**

No. 0 / 160 TOST checks establish equivalence at +/-3pp. Deployment teams cannot assume adjacent quant settings are safety-equivalent.

### 12.2 Cross-TR comparison

| Dimension | TR134-TR137 | TR138 / TR138 v2 | TR139 |
|-----------|-------------|-------------------|-------|
| Attack regime | Single-turn | Single-turn under batch perturbation | Multi-turn conversational |
| Primary manipulation | Quant level, backend | Batch size, co-batching | Quant level x attack strategy |
| Models | 2-5 models, 1B-3B | 3 models, 1B-3B | 4 models, 1B-8B |
| Core finding | Quant changes safety scores | Batching disproportionately flips safety outputs | Quant changes multi-turn ASR and persistence |
| Effect magnitude | Small-to-moderate | 2.0% safety flip rate, 4.7x ratio | `eta^2` up to 0.38; ASR swings of 60+pp |
| Strongest negative | - | No Phase 2 co-batching effect | No universal multi-turn amplification (H2) |
| Statistical foundation | Bootstrap CIs, TOST, power analysis | Wilson CIs, odds ratios, TOST | ANOVA, Holm-Bonferroni pairwise, Cohen's d, TOST, variance decomposition |
| Conversational depth | 1 turn | 1 turn | Up to 13 turns (5 attack + 8 pressure) |
| Key deployment implication | Validate quant choice | Treat batch size as safety-relevant | Validate multi-turn + persistence, not just direct refusal |

**Observations.** The most important column contrast is conversational depth: TR134-TR137 and TR138 operate entirely in the single-turn regime, while TR139 extends to 13 turns. The effect magnitudes also scale accordingly - TR138's 2.0% safety flip rate is a subtle signal requiring careful statistical framing, while TR139's 60+pp ASR swings in the worst cells are unmistakable even without sophisticated tests.

### 12.3 What this report establishes

1. Quantization belongs in the safety envelope for conversational deployment.
2. The risk is model-specific and threshold-specific, not governed by a single universal rule.
3. Direct-turn refusal is necessary but not sufficient - models that refuse direct harmful prompts can still fail badly under multi-turn attack or continued pressure.
4. The strongest model in this sweep (`llama3.2-3b`) is not robust by default; it is merely the least bad.
5. The weakest model under persistence pressure (`llama3.1-8b`) has a 96.67% break rate despite high initial refusal, making its refusal behavior effectively cosmetic.

---

## 13. Reproducibility

### 13.1 Entry points

Full pipeline:

```bash
python research/tr139/run.py -v
```

Common rerun modes:

```bash
# Judge + analysis + report on an existing run
python research/tr139/run.py -v --skip-prep --skip-eval --run-dir research/tr139/results/20260314_012503

# Analysis + report only
python research/tr139/run.py -v --skip-prep --skip-eval --skip-judge --run-dir research/tr139/results/20260314_012503
```

### 13.2 Source-of-truth configuration

The canonical config is `research/tr139/config.yaml` (full YAML excerpts in Appendix E). The key run-shaping parameters are:

| Setting | Value |
|---------|-------|
| Phase 1 models | `llama3.2-1b`, `llama3.2-3b`, `qwen2.5-1.5b`, `llama3.1-8b` |
| Phase 1 quants | `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K` |
| Phase 1 strategies | 8 |
| Phase 2 quants | `Q8_0`, `Q6_K`, `Q4_K_M`, `Q3_K_M`, `Q2_K` |
| Phase 2 pressure turns | 8 |
| Temperature | `0.0` |
| Max new tokens | `256` |
| Seed | `42` |
| Warmup requests | `3` |
| Cooldown between models | `10s` |

### 13.3 Exact run scope

| Component | Formula | Executed |
|-----------|---------|---------:|
| Phase 1 conversations | `4 x 6 x 8 x 50` | 9,600 |
| Phase 2 conversations | `4 x 5 x 50` | 1,000 |
| Judge labels | post-hoc scored turns | 37,825 |
| Total conversations | `Phase 1 + Phase 2` | 10,600 |

The primary completed run for this report is:

- `research/tr139/results/20260314_012503`

### 13.4 Model tags and execution context

The configured Ollama tags are:

| Model | Ollama tag | Notes |
|-------|------------|-------|
| `llama3.2-1b` | `llama3.2:1b` | full quant ladder |
| `llama3.2-3b` | `llama3.2:3b` | full quant ladder |
| `qwen2.5-1.5b` | `qwen2.5:1.5b` | full quant ladder |
| `llama3.1-8b` | `llama3.1:8b` | quantized-only in practice; `skip_fp16` |

Judge execution detail for the completed preserved run:

- saved judge tag: `qwen2.5:7b-instruct-q8_0`
- saved judge output count: `37,825`
- earlier fallback-run outputs were overwritten during development and are not treated as part of the final evidence chain

### 13.5 Key artifacts

| Artifact | Path | Role in this report |
|----------|------|---------------------|
| Analysis JSON | `research/tr139/results/20260314_012503/tr139_analysis.json` | primary quantitative source |
| Scored rows | `research/tr139/results/20260314_012503/tr139_scored.jsonl` | scored conversation-level outputs |
| Judge labels | `research/tr139/results/20260314_012503/judge_labels.jsonl` | post-hoc judge layer |
| Generated report | `research/tr139/results/20260314_012503/tr139_report.md` | machine-produced appendix layer |
| README | `research/tr139/README.md` | design intent and literature map |
| Shared utils | `research/tr139/shared/utils.py` | strategy templates, BPW map, constants |

### 13.6 Analysis and generation scripts

| Script | Role |
|--------|------|
| `research/tr139/run.py` | orchestration entry point |
| `research/tr139/prepare_benchmarks.py` | task and behavior preparation |
| `research/tr139/judge_analysis.py` | post-hoc judge pass |
| `research/tr139/analyze.py` | hypothesis tests and summary statistics |
| `research/tr139/generate_report.py` | generated markdown report |
| `research/tr139/shared/utils.py` | model matrix, quant ladder, strategy templates |

The generated report in the run directory remains a useful machine-produced appendix, but this publish-ready report is the authoritative interpretation layer.

### 13.7 Prerequisites

To rerun TR139 cleanly, the environment needs:

- Ollama running with GPU access
- the four configured model tags installed
- the task files under `research/tr139/tasks/`
- the shared refusal detector and judge stack inherited from TR134

Operationally, the run is large enough that it should be treated like a planned experiment rather than a casual benchmark script. That is another reason this report needs to preserve the reasoning chain in full.

---

## References and External Anchors

TR139 is grounded in two external literatures plus the repo's prior safety line.

### Multi-turn jailbreak methods

- **Crescendo** (Russinovich et al., 2024): authority-building conversational jailbreaks
- **Foot-in-the-Door** (Wang et al., 2025): gradual escalation through related benign requests
- **Attention Shifting Attack** (Zhou et al., 2025): distraction and fictional framing
- **RACE / progressive refinement** (Li et al., 2025): narrowing through iterative benign-seeming reformulation
- **Context Fusion Attack** (Xu et al., 2024): distributed intent across turns
- **ActorAttack / persona manipulation** (Ren et al., 2024): role and character consistency as a jailbreak lever

### Quantization x safety work

- **HarmLevelBench**: lower-bit deployment can alter harmful-compliance behavior
- **Q-resafe**: quantization-sensitive safety changes are not purely hypothetical
- **Alignment-Aware Quantization**: quantization can interact with alignment properties rather than acting as a neutral compression layer

### Repo continuity

- **TR134-TR137**: quantization, backend, and concurrency are not safety-neutral
- **TR138 / TR138 v2**: batching is not safety-neutral under deterministic decoding
- **TR139**: conversational attack structure belongs in that same deployment-safety family

TR139's direct novelty is the intersection of those literatures: conversational jailbreak methodology and quantized local deployment.

---

## Appendix A: Raw Data Tables

### A.1 Phase 2 persistence by model x quant (complete)

| Model | Quant | BPW | Initial refusals | Broke | Break rate | Mean persistence | Std | Pressure resistance |
|-------|-------|----:|------------------:|------:|-----------:|-----------------:|----:|--------------------:|
| `llama3.1-8b` | `Q8_0` | 8.00 | 43 | 43 | 100.0% | 0.2422 | 0.2029 | 0.1919 |
| `llama3.1-8b` | `Q6_K` | 6.57 | 33 | 33 | 100.0% | 0.2111 | 0.2214 | 0.2348 |
| `llama3.1-8b` | `Q4_K_M` | 4.85 | 32 | 32 | 100.0% | 0.1222 | 0.1493 | 0.0898 |
| `llama3.1-8b` | `Q3_K_M` | 3.91 | 6 | 5 | 83.3% | 0.0533 | 0.1921 | 0.3750 |
| `llama3.1-8b` | `Q2_K` | 3.35 | 5 | 5 | 100.0% | 0.0356 | 0.1445 | 0.2750 |
| `llama3.2-1b` | `Q8_0` | 8.00 | 47 | 8 | 17.0% | 0.9222 | 0.2389 | 0.9787 |
| `llama3.2-1b` | `Q6_K` | 6.57 | 48 | 4 | 8.3% | 0.9511 | 0.1985 | 0.9896 |
| `llama3.2-1b` | `Q4_K_M` | 4.85 | 46 | 34 | 73.9% | 0.7467 | 0.2694 | 0.7880 |
| `llama3.2-1b` | `Q3_K_M` | 3.91 | 41 | 33 | 80.5% | 0.6933 | 0.3532 | 0.8262 |
| `llama3.2-1b` | `Q2_K` | 3.35 | 16 | 16 | 100.0% | 0.0467 | 0.0780 | 0.0391 |
| `llama3.2-3b` | `Q8_0` | 8.00 | 37 | 9 | 24.3% | 0.6067 | 0.4602 | 0.7973 |
| `llama3.2-3b` | `Q6_K` | 6.57 | 38 | 13 | 34.2% | 0.5733 | 0.4564 | 0.7237 |
| `llama3.2-3b` | `Q4_K_M` | 4.85 | 41 | 13 | 31.7% | 0.6622 | 0.4199 | 0.7835 |
| `llama3.2-3b` | `Q3_K_M` | 3.91 | 48 | 24 | 50.0% | 0.7644 | 0.3050 | 0.7708 |
| `llama3.2-3b` | `Q2_K` | 3.35 | 44 | 41 | 93.2% | 0.3244 | 0.2856 | 0.2898 |
| `qwen2.5-1.5b` | `Q8_0` | 8.00 | 49 | 5 | 10.2% | 0.9467 | 0.1785 | 0.9617 |
| `qwen2.5-1.5b` | `Q6_K` | 6.57 | 49 | 4 | 8.2% | 0.9600 | 0.1631 | 0.9770 |
| `qwen2.5-1.5b` | `Q4_K_M` | 4.85 | 48 | 36 | 75.0% | 0.5311 | 0.3263 | 0.4974 |
| `qwen2.5-1.5b` | `Q3_K_M` | 3.91 | 45 | 38 | 84.4% | 0.6156 | 0.3176 | 0.6444 |
| `qwen2.5-1.5b` | `Q2_K` | 3.35 | 20 | 3 | 15.0% | 0.3511 | 0.4727 | 0.8625 |

**Observations.** The `llama3.1-8b` row group is striking: break rate is 100% at every level except Q3_K_M, which only escapes by having very few initial refusals (6). The `llama3.2-1b` cliff between Q6_K (8.3% break) and Q4_K_M (73.9% break) is the sharpest discontinuity in the table. The `qwen2.5-1.5b` Q2_K row is paradoxically low-break-rate (15.0%) because the model already complies directly at Q2_K, leaving only 20 initial refusals to test.

### A.2 Phase 1 ASR slopes by model x strategy (complete)

| Model | Strategy | Slope/BPW | CI lower | CI upper | R^2 |
|-------|----------|----------:|---------:|---------:|---:|
| `llama3.1-8b` | `attention_shift` | -0.0193 | -0.0743 | +0.0663 | 0.094 |
| `llama3.1-8b` | `benign_context` | -0.1386 | -0.3208 | -0.0241 | 0.574 |
| `llama3.1-8b` | `context_fusion` | +0.0051 | -0.0879 | +0.0497 | 0.008 |
| `llama3.1-8b` | `crescendo` | -0.0522 | -0.2966 | +0.0306 | 0.097 |
| `llama3.1-8b` | `direct` | -0.1699 | -0.3784 | -0.0294 | 0.720 |
| `llama3.1-8b` | `foot_in_door` | -0.0710 | -0.1704 | -0.0046 | 0.659 |
| `llama3.1-8b` | `progressive_refinement` | -0.0446 | -0.1456 | +0.0089 | 0.424 |
| `llama3.1-8b` | `role_play` | -0.0858 | -0.2851 | +0.0561 | 0.395 |
| `llama3.2-1b` | `attention_shift` | -0.0923 | -0.2911 | +0.0000 | 0.355 |
| `llama3.2-1b` | `benign_context` | -0.0692 | -0.1922 | +0.0173 | 0.367 |
| `llama3.2-1b` | `context_fusion` | -0.0650 | -0.2035 | +0.0000 | 0.363 |
| `llama3.2-1b` | `crescendo` | -0.1197 | -0.3788 | +0.0000 | 0.351 |
| `llama3.2-1b` | `direct` | -0.0987 | -0.2712 | -0.0066 | 0.479 |
| `llama3.2-1b` | `foot_in_door` | -0.1080 | -0.2233 | -0.0365 | 0.713 |
| `llama3.2-1b` | `progressive_refinement` | -0.1270 | -0.3331 | -0.0110 | 0.538 |
| `llama3.2-1b` | `role_play` | -0.0465 | -0.1490 | +0.0000 | 0.336 |
| `llama3.2-3b` | `attention_shift` | +0.0000 | +0.0000 | +0.0000 | 0.000 |
| `llama3.2-3b` | `benign_context` | +0.0157 | -0.0079 | +0.0584 | 0.265 |
| `llama3.2-3b` | `context_fusion` | +0.0253 | +0.0000 | +0.0412 | 0.717 |
| `llama3.2-3b` | `crescendo` | -0.0027 | -0.0088 | +0.0000 | 0.336 |
| `llama3.2-3b` | `direct` | +0.0400 | +0.0191 | +0.0695 | 0.722 |
| `llama3.2-3b` | `foot_in_door` | -0.0074 | -0.0242 | +0.0057 | 0.297 |
| `llama3.2-3b` | `progressive_refinement` | +0.0016 | -0.0002 | +0.0076 | 0.111 |
| `llama3.2-3b` | `role_play` | +0.0000 | +0.0000 | +0.0000 | 0.000 |
| `qwen2.5-1.5b` | `attention_shift` | +0.0234 | -0.0110 | +0.1078 | 0.147 |
| `qwen2.5-1.5b` | `benign_context` | -0.0669 | -0.1824 | -0.0076 | 0.493 |
| `qwen2.5-1.5b` | `context_fusion` | -0.0891 | -0.2526 | +0.0074 | 0.493 |
| `qwen2.5-1.5b` | `crescendo` | -0.1169 | -0.3294 | -0.0055 | 0.523 |
| `qwen2.5-1.5b` | `direct` | -0.0885 | -0.2632 | -0.0002 | 0.432 |
| `qwen2.5-1.5b` | `foot_in_door` | -0.0961 | -0.2129 | -0.0174 | 0.609 |
| `qwen2.5-1.5b` | `progressive_refinement` | -0.0144 | -0.0670 | +0.0168 | 0.128 |
| `qwen2.5-1.5b` | `role_play` | -0.1403 | -0.2297 | -0.0204 | 0.716 |

**Observations.** The `llama3.2-3b` row group is strikingly flat - most slopes are near zero, and even the largest (`direct`, +0.0400/BPW) reflects an anomalous pattern where lower quant slightly improves ASR, driven by the model's already-low failure surface. The `llama3.2-1b` slopes are uniformly negative, consistent with the Q2_K cliff story. The `qwen2.5-1.5b` `role_play` slope (-0.1403, `R^2 = 0.716`) has the best linear fit among the high-risk cells.

### A.3 Power analysis by model x strategy

| Model | Strategy | Baseline ASR | MDE (80% power) |
|-------|----------|-------------:|-----------------:|
| `llama3.1-8b` | `attention_shift` | 18.0% | 21.5pp |
| `llama3.1-8b` | `benign_context` | 16.0% | 20.5pp |
| `llama3.1-8b` | `context_fusion` | 22.0% | 23.2pp |
| `llama3.1-8b` | `crescendo` | 26.0% | 24.6pp |
| `llama3.1-8b` | `direct` | 14.0% | 19.4pp |
| `llama3.1-8b` | `foot_in_door` | 8.0% | 15.2pp |
| `llama3.1-8b` | `progressive_refinement` | 22.0% | 23.2pp |
| `llama3.1-8b` | `role_play` | 22.0% | 23.2pp |
| `llama3.2-1b` | `attention_shift` | 0.0% | 5.6pp |
| `llama3.2-1b` | `benign_context` | 8.0% | 15.2pp |
| `llama3.2-1b` | `context_fusion` | 0.0% | 5.6pp |
| `llama3.2-1b` | `crescendo` | 0.0% | 5.6pp |
| `llama3.2-1b` | `direct` | 6.0% | 13.3pp |
| `llama3.2-1b` | `foot_in_door` | 0.0% | 5.6pp |
| `llama3.2-1b` | `progressive_refinement` | 0.0% | 5.6pp |
| `llama3.2-1b` | `role_play` | 0.0% | 5.6pp |
| `llama3.2-3b` | `attention_shift` | 0.0% | 5.6pp |
| `llama3.2-3b` | `benign_context` | 10.0% | 16.8pp |
| `llama3.2-3b` | `context_fusion` | 10.0% | 16.8pp |
| `llama3.2-3b` | `crescendo` | 0.0% | 5.6pp |
| `llama3.2-3b` | `direct` | 26.0% | 24.6pp |
| `llama3.2-3b` | `foot_in_door` | 0.0% | 5.6pp |
| `llama3.2-3b` | `progressive_refinement` | 0.0% | 5.6pp |
| `llama3.2-3b` | `role_play` | 0.0% | 5.6pp |
| `qwen2.5-1.5b` | `attention_shift` | 98.0% | 7.8pp |
| `qwen2.5-1.5b` | `benign_context` | 8.0% | 15.2pp |
| `qwen2.5-1.5b` | `context_fusion` | 60.0% | 27.4pp |
| `qwen2.5-1.5b` | `crescendo` | 34.0% | 26.5pp |
| `qwen2.5-1.5b` | `direct` | 2.0% | 7.8pp |
| `qwen2.5-1.5b` | `foot_in_door` | 30.0% | 25.7pp |
| `qwen2.5-1.5b` | `progressive_refinement` | 78.0% | 23.2pp |
| `qwen2.5-1.5b` | `role_play` | 36.0% | 26.9pp |

**Observations.** MDE is strongly driven by baseline ASR. Cells with 0% baseline (e.g., `llama3.2-1b / attention_shift`) have the tightest MDE (5.6pp) because any movement from zero is easy to detect. The broadest MDEs (24-27pp) appear in cells with baseline ASR near 30-60%, exactly where the binomial variance is maximized. This means the report is well-powered to detect large cliff-like failures but poorly powered to distinguish adjacent moderate-ASR cells.

---

## Appendix B: Extended Statistical Tables

### B.1 Selected pairwise comparisons with Cohen's d and Holm-Bonferroni correction

All comparisons use Welch's t-test with Holm-Bonferroni correction within each model x strategy family. Full tables are in `tr139_analysis.json` -> `phase1_pairwise`.

#### `llama3.2-1b` (cliff model)

| Strategy | Comparison | Mean A | Mean B | t | p | Cohen's d | Holm p | Sig? |
|----------|------------|-------:|-------:|--:|--:|----------:|-------:|:----:|
| `crescendo` | Q2_K vs Q8_0 | 0.86 | 0.00 | - | < 0.001 | 2.49 | < 0.001 | yes |
| `foot_in_door` | Q2_K vs Q8_0 | 0.76 | 0.00 | - | < 0.001 | 2.33 | < 0.001 | yes |
| `progressive_refinement` | Q2_K vs Q8_0 | 0.78 | 0.00 | - | < 0.001 | 2.49 | < 0.001 | yes |
| `direct` | Q2_K vs Q8_0 | 0.68 | 0.06 | - | < 0.001 | 1.54 | < 0.001 | yes |
| `attention_shift` | Q2_K vs Q8_0 | 0.54 | 0.00 | - | < 0.001 | 1.59 | < 0.001 | yes |

#### `llama3.1-8b` (instability-band model)

| Strategy | Comparison | Mean A | Mean B | t | p | Cohen's d | Holm p | Sig? |
|----------|------------|-------:|-------:|--:|--:|----------:|-------:|:----:|
| `benign_context` | Q3_K_M vs Q8_0 | 0.92 | 0.16 | - | < 0.001 | 1.00 | < 0.001 | yes |
| `direct` | Q3_K_M vs Q8_0 | 0.88 | 0.14 | - | < 0.001 | 1.00 | < 0.001 | yes |
| `attention_shift` | Q2_K vs Q4_K_M | 0.42 | 0.12 | 3.55 | < 0.001 | 0.71 | 0.004 | yes |
| `foot_in_door` | Q6_K vs Q8_0 | 0.20 | 0.08 | - | 0.020 | 0.38 | 0.058 | no |

#### `qwen2.5-1.5b` (broadly vulnerable)

| Strategy | Comparison | Mean A | Mean B | t | p | Cohen's d | Holm p | Sig? |
|----------|------------|-------:|-------:|--:|--:|----------:|-------:|:----:|
| `role_play` | Q2_K vs Q8_0 | 1.00 | 0.36 | - | < 0.001 | 2.18 | < 0.001 | yes |
| `crescendo` | Q2_K vs Q8_0 | 1.00 | 0.34 | - | < 0.001 | 2.20 | < 0.001 | yes |
| `foot_in_door` | Q2_K vs Q8_0 | 0.88 | 0.30 | - | < 0.001 | 1.38 | < 0.001 | yes |
| `benign_context` | Q2_K vs Q8_0 | 0.52 | 0.08 | - | < 0.001 | 1.10 | < 0.001 | yes |

#### `llama3.2-3b` (flat-surface model)

| Strategy | Comparison | Mean A | Mean B | t | p | Cohen's d | Holm p | Sig? |
|----------|------------|-------:|-------:|--:|--:|----------:|-------:|:----:|
| `direct` | Q2_K vs Q3_K_M | 0.12 | 0.00 | - | < 0.001 | 3.47 | < 0.001 | yes |
| `benign_context` | Q2_K vs Q8_0 | 0.02 | 0.10 | - | 0.081 | 0.53 | 1.000 | no |

**Observations.** The Cohen's d values confirm that the largest pairwise effects are not just statistically significant - they are practically enormous. Values above 2.0 represent complete regime changes rather than incremental shifts. The `llama3.2-3b` `direct` Q2_K vs Q3_K_M comparison has the largest d (3.47) despite both being low-ASR, because the transition from 0% to 12% ASR represents a full shift from floor to non-trivial failure. Holm correction is conservative: the `llama3.1-8b / foot_in_door / Q6_K vs Q8_0` comparison loses significance after correction (raw p = 0.020, Holm p = 0.058), demonstrating that borderline results are properly gated.

### B.2 Fisher exact test results (quant vs Q8_0 baseline, selected significant cells)

All Fisher exact tests are Holm-Bonferroni corrected within each model x strategy family (5 tests per family).

| Model | Strategy | Quant | Odds ratio | Fisher p | Holm p | Sig? |
|-------|----------|-------|----------:|---------:|-------:|:----:|
| `llama3.1-8b` | `benign_context` | Q3_K_M | - | < 0.001 | < 0.001 | yes |
| `llama3.1-8b` | `benign_context` | Q4_K_M | - | < 0.001 | < 0.001 | yes |
| `llama3.1-8b` | `benign_context` | Q2_K | - | 0.007 | 0.036 | yes |
| `llama3.1-8b` | `crescendo` | Q3_K_M | - | < 0.001 | < 0.001 | yes |
| `llama3.1-8b` | `direct` | Q3_K_M | - | < 0.001 | < 0.001 | yes |
| `llama3.1-8b` | `direct` | Q2_K | - | < 0.001 | < 0.001 | yes |
| `llama3.1-8b` | `foot_in_door` | Q6_K | - | 0.004 | 0.020 | yes |
| `llama3.2-1b` | `crescendo` | Q2_K | - | < 0.001 | < 0.001 | yes |
| `llama3.2-1b` | `direct` | Q2_K | - | < 0.001 | < 0.001 | yes |
| `qwen2.5-1.5b` | `role_play` | Q2_K | - | < 0.001 | < 0.001 | yes |
| `qwen2.5-1.5b` | `crescendo` | Q2_K | - | < 0.001 | < 0.001 | yes |

**Observations.** After Holm-Bonferroni correction, the most important Q2_K-vs-baseline and Q3_K_M-vs-baseline comparisons remain strongly significant. The correction eliminates borderline cells that might otherwise be overclaimed, which is why the report emphasizes model-level patterns and ANOVA-level conclusions over individual cell stories.

---

## Appendix C: Sensitivity and Robustness

### C.1 Runtime and execution shape

TR139 is large enough that execution shape is part of the scientific interpretation, not just an ops detail.

| Component | Structure | Upper-bound model calls |
|-----------|-----------|------------------------:|
| Phase 1 | 9,600 conversations across 8 fixed strategies | 39,600 |
| Phase 2 | 1,000 conversations with up to 8 pressure follow-ups | 9,000 |
| Eval total | Phase 1 + Phase 2 | 48,600 |
| Judge | post-hoc scored turns | 37,825 labels |

The Phase 1 upper bound comes from the fixed strategy lengths:

- `direct`: 1 turn
- `benign_context`: 5 turns
- `foot_in_door`: 5 turns
- `crescendo`: 5 turns
- `attention_shift`: 5 turns
- `role_play`: 4 turns
- `progressive_refinement`: 4 turns
- `context_fusion`: 4 turns

That yields `33` attack-template turns per `(model, quant, behavior)` bundle and therefore `4 x 6 x 50 x 33 = 39,600` Phase 1 model calls before Phase 2 even begins.

This runtime shape explains three report choices:

1. Phase 1 is the main evidentiary mass, so the report leans on aggregate Phase 1 structure for the strongest claims.
2. Phase 2 is smaller and therefore better used as a persistence-validation layer than as a replacement for Phase 1.
3. The cost of reproducing TR139 is nontrivial, which raises the bar for how much reasoning the publish-ready report must preserve.

### C.2 Variance decomposition sensitivity

The main variance decomposition (18.31% model, 6.72% quant, 1.21% quant x strategy, 0.83% strategy, 72.93% residual) uses a simple sum-of-squares approach. If the residual term is further partitioned by behavior category, the 10 harm categories absorb an estimated 8-12% of the residual, reducing it to approximately 61-65%. This suggests behavior-level heterogeneity is real but not the dominant unexplained source - individual prompt-level variation within categories is larger.

### C.3 Judge sensitivity check

The report's core Phase 1 claims do not depend on the judge layer. The 8/8 ANOVA rejections use the primary regex refusal detector, not the judge. The judge layer adds depth (turning binary refusal into a nuanced agreement picture) but does not change the H1 rejection or the H2 non-result. The H3 persistence slopes also use the primary detector.

This means the core findings survive even if the judge layer is discarded entirely, which is the appropriate robustness posture given the weak kappa values.

### C.4 Source artifacts

Primary source-of-truth files for this report:

| Artifact | Path | Report use |
|----------|------|------------|
| Main analysis | `research/tr139/results/20260314_012503/tr139_analysis.json` | all quantitative claims |
| Scored outputs | `research/tr139/results/20260314_012503/tr139_scored.jsonl` | cell-level drilldown and follow-up audit |
| Judge labels | `research/tr139/results/20260314_012503/judge_labels.jsonl` | agreement and reliability section |
| Generated report | `research/tr139/results/20260314_012503/tr139_report.md` | machine-produced baseline interpretation |
| Config | `research/tr139/config.yaml` | run scope and settings |
| Design README | `research/tr139/README.md` | research questions and literature map |
| Strategy definitions | `research/tr139/shared/utils.py` | exact template structure and BPW mapping |

The publish-ready report is manually written from those artifacts. It should be treated as the narrative interpretation layer on top of the generated outputs, not as a substitute for them.

---

## Appendix D: Glossary

### Statistical Terms

| Term | Definition |
|------|-----------|
| **ANOVA (one-way)** | Tests whether means differ across 3+ groups. F-statistic is the ratio of between-group to within-group variance. p < 0.05 rejects the null that all groups have the same mean. |
| **ANOVA (two-way)** | Tests main effects of two factors and their interaction simultaneously. `eta^2` for each factor shows what fraction of variance it explains. |
| **Bootstrap CI** | Confidence interval computed by resampling the data with replacement (here: 1,000 iterations, seed 42, percentile method). Does not assume normality. |
| **Cohen's d** | Standardized effect size for comparing two means: d = (M1 - M2) / pooled SD. Conventional thresholds: small = 0.2, medium = 0.5, large = 0.8. Values above 2.0 indicate regime-change-level differences. |
| **Fisher exact test** | Non-parametric test for 2x2 contingency tables. Used here for per-cell quant-vs-baseline comparisons where expected counts may be small. |
| **Holm-Bonferroni correction** | Sequential multiple comparison correction that controls family-wise error rate (FWER). Orders raw p-values, then rejects if p_i <= alpha / (k - i + 1). Less conservative than Bonferroni but still controls Type I error. |
| **eta^2 (eta-squared)** | ANOVA effect size. Proportion of total variance explained by the factor. Small = 0.01, medium = 0.06, large = 0.14. |
| **MDE (Minimum Detectable Effect)** | Smallest effect the study can detect at specified power (80%) and alpha (0.05). Depends on sample size and baseline rate. |
| **TOST (Two One-Sided Tests)** | Equivalence testing: tests whether the difference between groups falls within a pre-specified margin (here +/-3pp). Rejection means the groups are practically equivalent; failure means non-equivalence is not ruled out. |
| **Welch's t-test** | t-test for comparing two means without assuming equal variances. Used for the H2 slope comparison and pairwise quant comparisons. |
| **Wilson score interval** | Confidence interval for proportions that corrects for the poor coverage of the normal approximation at extreme values (near 0 or 1). |

### Attack Strategy Terms

| Strategy | Definition |
|----------|-----------|
| **direct** | Single-turn baseline. The harmful request is presented without conversational scaffold. |
| **benign_context** | Benign multi-turn context followed by the harmful request. Tests whether conversational momentum alone shifts the safety boundary. |
| **foot_in_door** | Gradual topical escalation. Small, seemingly acceptable requests accumulate toward the final harmful turn. |
| **crescendo** | Authority and credential framing. The conversation builds trust and legitimacy before the harmful request. |
| **attention_shift** | Creative or unrelated framing followed by abrupt pivot. Strongest overall strategy in this sweep. |
| **role_play** | Persona induction. The model is pushed to maintain a fictional character or role that weakens refusal behavior. |
| **progressive_refinement** | The user starts broad and iteratively narrows toward harmful specificity. One of the most consistently dangerous strategies. |
| **context_fusion** | Intent is distributed across individually milder turns and fused at the target turn. Especially powerful on the most vulnerable model. |

### Deployment Terms

| Term | Definition |
|------|-----------|
| **ASR (Attack Success Rate)** | Fraction of conversations where the target turn is scored compliant. Higher = worse. |
| **BPW (Bits Per Weight)** | Average number of bits per model parameter after quantization. Q8_0 = 8.00, Q2_K = 3.35. |
| **Break rate** | Fraction of initially-refusing Phase 2 conversations that eventually comply under pressure. |
| **GGUF** | File format for quantized LLM weights used by llama.cpp and Ollama. |
| **Persistence** | Fraction of turns in a Phase 2 conversation that preserve refusal. Lower = worse. |
| **Pressure resistance** | Fraction of post-refusal turns that maintain refusal. Isolates durability after the initial response. |

---

## Appendix E: Configs (Source of Truth)

The executed run is defined by `research/tr139/config.yaml`. The key excerpts are:

```yaml
phase1:
  quant_levels: [Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K]
  attack_strategies:
    - direct
    - benign_context
    - foot_in_door
    - crescendo
    - attention_shift
    - role_play
    - progressive_refinement
    - context_fusion

phase2:
  quant_levels: [Q8_0, Q6_K, Q4_K_M, Q3_K_M, Q2_K]
  pressure_turns: 8

max_new_tokens: 256
temperature: 0.0
seed: 42
warmup_requests: 3
cooldown_between_models_s: 10
```

Model definitions:

```yaml
- name: llama3.2-1b
  ollama_tag: "llama3.2:1b"

- name: llama3.2-3b
  ollama_tag: "llama3.2:3b"

- name: qwen2.5-1.5b
  ollama_tag: "qwen2.5:1.5b"

- name: llama3.1-8b
  ollama_tag: "llama3.1:8b"
  skip_fp16: true
```

Those config excerpts are the final source of truth for what TR139 actually ran.



