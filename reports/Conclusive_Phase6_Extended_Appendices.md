# Technical Report — Conclusive Synthesis, Phase 6 — Extended Appendices

## Per-Report Data Tables, Named-Method Definitions, and Cross-TR Ledgers (TR144–TR149 + TR152)

This file is the appendix companion to `Conclusive_Phase6.md`. The main synthesis carries the narrative; this file carries the full numerical substrate behind it. Appendices **A–G** reproduce the canonical data tables for each report in the Phase 6 line (TR144, TR145, TR146, TR147, TR148, TR149, TR152), one appendix per report, in report-number order. Appendices **X1–X4** are cross-TR ledgers — named-method definitions, sample-budget, null-vs-located findings, and judge-cohort inheritance — that only make sense at the synthesis level. A Phase 6-wide **Glossary** closes the file.

Every table carries a one-line caption and a source citation of the form `SSn.m`, pointing into the corresponding `PublishReady/reports/Technical_Report_NNN.md` source report. Numbers are transcribed from the source reports and their analysis JSON, not recomputed here. Where a structural numeric field was cross-checked against a run-directory artifact (e.g. `cri_values.json`, `tr152_analysis.json`), the artifact is named in the caption.

**Reading the tables.** All safety statistics use the matched-pairs convention unless a table explicitly says otherwise: McNemar discordant cells are labelled by direction (FP16→FP8 *unsafe* = degrade; FP16→FP8 *safe* = improve), Mantel–Haenszel pooled odds ratios use the Haldane-corrected paired discordant ratio `(Σb+0.5)/(Σc+0.5)` (not the unpaired cross-product — see Appendix F.10 for why the distinction is load-bearing), Cohen's *h* is the paired-binary effect size (Cohen's *d* is reserved for TR147's continuous latency), and TOST equivalence is reported against the ±3pp margin the whole line shares.

---

## Table of Contents

- **Appendix A — TR144** (Speculative Decoding × Safety; TAIS screen)
- **Appendix B — TR145** (FP8 KV-cache safety, single-configuration base case)
- **Appendix C — TR146** (Mechanistic probing under quantization; four-probe falsification)
- **Appendix D — TR147** (Compile Reproducibility Index; the Triton kill-shot)
- **Appendix E — TR148** (Judge Triangulation Protocol; the dual-axis finding)
- **Appendix F — TR149** (Standardized safety battery; FP16 vs FP8 KV-cache)
- **Appendix G — TR152** (FP8 KV-cache serving-state factorial; the Layer 5 anchor)
- **Appendix X1 — Named-method definitions** (CRI / JTP / TAIS / RTSI in one place)
- **Appendix X2 — Cross-TR sample-budget and cost ledger**
- **Appendix X3 — Cross-TR null-vs-located findings**
- **Appendix X4 — Cross-TR judge-cohort inheritance**
- **Glossary** (Phase 6-wide)

---

## Appendix A — TR144 (Speculative Decoding × Safety; TAIS)

Source of truth: `PublishReady/reports/Technical_Report_144.md`. TR144 is the speculative-decoding member of the inference-flag null line: a paired, factorial test of whether speculative decoding (rejection sampling and typical acceptance) leaks unsafe tokens at greedy decoding, across three core model pairs plus a five-experiment expansion (E1–E5).

### A.1 — Scale / sample-integrity

*Per-phase / expansion record budget; 16,783 core + 48,072 expansion = 64,855 total paired samples, zero dropped. (SS5.9, SS24 one-screen summary)*

| Component | Expected | Actual | Match |
|-----------|----------|--------|-------|
| Phase 1 (5 × 953) | 4,765 | 4,765 | Exact |
| Phase 2 (3 × 953, rejection) | 2,859 | 2,859 | Exact |
| Phase 3 (3 × 953, typical) | 2,859 | 2,859 | Exact |
| Phase 5 (3 × 5 × 420 sweep) | 6,300 | 6,300 | Exact |
| Phase 5 (reuse, w/ telemetry) | — | 12,018 spec | Exact |
| Judge labels (safety subset) | — | 11,448 | Exact |
| **Core total** | **16,783** | **16,783** | **Exact** |

| Experiment | Probe | Samples |
|------------|-------|---------|
| E1 | 70B+8B production pair | 4,006 |
| E2 | DPO-adversarial draft (flipped hh-rlhf) | 4,006 |
| E3 | GPTQ-4bit quantized draft | 4,006 |
| E4 | 2 seeds × 3 pairs (fp16) | 24,036 |
| E5 | bfloat16 × 3 pairs | 12,018 |
| **Expansion total** | | **48,072** |

### A.2 — Phase 2 (rejection sampling) per-model McNemar

*p-value, discordant count, OR; all non-significant. R→C = refusal→compliance (unsafe direction). (SS4)*

| Pair | Refuse→Comply | Comply→Refuse | n_discordant | OR | p_exact | Significant? |
|------|---------------|---------------|--------------|------|---------|--------------|
| llama3.2-3b+1b | 0 | 1 | 2 | 0.50 | 1.000 | No |
| qwen2.5-3b+1.5b | 1 | 1 | 2 | 1.00 | 1.000 | No |
| qwen2.5-1.5b+0.5b | 0 | 3 | 3 | 0.20 | 1.000 | No |

Companion byte-identity (SS3): overall 90.66% (2,592/2,859 identical, 267 changed, 10 safety flips, 1 capability flip); per-pair identity 92.44% / 87.72% / 91.82%.

### A.3 — Phase 3 (typical acceptance) per-model McNemar — the primary null

*The primary leakage test. All non-significant. (SS6)*

| Pair | Refuse→Comply | Comply→Refuse | n_discordant | OR | p_exact | Significant? |
|------|---------------|---------------|--------------|------|---------|--------------|
| llama3.2-3b+1b | 1 | 2 | 3 | 0.60 | 1.000 | No |
| qwen2.5-3b+1.5b | 2 | 0 | 2 | 5.00 | 0.500 | No |
| qwen2.5-1.5b+0.5b | 1 | 3 | 4 | 0.43 | 0.625 | No |

**A.3.1 — Phase 3 per-task safety deltas (12 cells, Cohen's d).** *Phase 1 → Phase 3 per-task safety rates; all deltas exactly 0.0pp, d = 0.000, spanning baselines 0.408–0.985. (SS8)*

| Pair | Task | Phase 1 rate | Phase 3 rate | Δ (pp) | Cohen's d |
|------|------|--------------|--------------|--------|-----------|
| llama3.2-3b+1b | advbench_refusal | 0.790 | 0.790 | 0.0 | 0.000 |
| llama3.2-3b+1b | bbq_bias | 0.924 | 0.924 | 0.0 | 0.000 |
| llama3.2-3b+1b | jailbreak_amplification | 0.583 | 0.583 | 0.0 | 0.000 |
| llama3.2-3b+1b | truthfulqa | 0.560 | 0.560 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | advbench_refusal | 0.970 | 0.970 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | bbq_bias | 0.985 | 0.985 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | jailbreak_amplification | 0.408 | 0.408 | 0.0 | 0.000 |
| qwen2.5-3b+1.5b | truthfulqa | 0.480 | 0.480 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | advbench_refusal | 0.980 | 0.980 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | bbq_bias | 0.899 | 0.899 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | jailbreak_amplification | 0.575 | 0.575 | 0.0 | 0.000 |
| qwen2.5-1.5b+0.5b | truthfulqa | 0.510 | 0.510 | 0.0 | 0.000 |

### A.4 — Phase 5 dose-response slopes across N ∈ {1,3,5,8,12}

*12 logistic slopes + r², all zero; safety rate flat across speculation length. (SS10)*

| Pair | Task | Slope | r² | Means [N=1,3,5,8,12] |
|------|------|-------|-----|------------------------|
| llama3.2-3b+1b | advbench_refusal | 0.000 | 0.000 | [0.79, 0.79, 0.79, 0.79, 0.79] |
| llama3.2-3b+1b | bbq_bias | 0.000 | 0.000 | [0.933, 0.933, 0.933, 0.933, 0.933] |
| llama3.2-3b+1b | jailbreak_amplification | 0.000 | 0.000 | [0.575, 0.575, 0.575, 0.575, 0.575] |
| llama3.2-3b+1b | truthfulqa | 0.000 | 0.000 | [0.56, 0.56, 0.56, 0.56, 0.56] |
| qwen2.5-3b+1.5b | advbench_refusal | 0.000 | 0.000 | [0.97, 0.97, 0.97, 0.97, 0.97] |
| qwen2.5-3b+1.5b | bbq_bias | 0.000 | 0.000 | [0.987, 0.987, 0.987, 0.987, 0.987] |
| qwen2.5-3b+1.5b | jailbreak_amplification | 0.000 | 0.000 | [0.392, 0.392, 0.392, 0.392, 0.392] |
| qwen2.5-3b+1.5b | truthfulqa | 0.000 | 0.000 | [0.48, 0.48, 0.48, 0.48, 0.48] |
| qwen2.5-1.5b+0.5b | advbench_refusal | 0.000 | 0.000 | [0.98, 0.98, 0.98, 0.98, 0.98] |
| qwen2.5-1.5b+0.5b | bbq_bias | 0.000 | 0.000 | [0.913, 0.913, 0.913, 0.913, 0.913] |
| qwen2.5-1.5b+0.5b | jailbreak_amplification | 0.000 | 0.000 | [0.575, 0.575, 0.575, 0.575, 0.575] |
| qwen2.5-1.5b+0.5b | truthfulqa | 0.000 | 0.000 | [0.49, 0.49, 0.49, 0.49, 0.49] |

Per-pair bootstrap slope CIs (1,000 resamples): all [0.000, 0.000]; no critical N exceeds 3pp for any pair (SS11).

### A.5 — TAIS calibration: Cohen's h across 18 AdvBench contrasts

*Every (experiment × pair × AdvBench) target-alone vs speculative contrast; max |h| = 0.024 (E5 llama3.2-3b+1b), ~4× below the 0.1 cutoff. (SS24.1)*

| # | Experiment | Pair | p_target_alone | p_spec | Cohen's h | TOST ±3pp |
|---|------------|------|----------------|--------|-----------|-----------|
| 1 | Core Phase 2 | llama3.2-3b+1b | 0.790 | 0.790 | 0.0000 | PASS |
| 2 | Core Phase 2 | qwen2.5-3b+1.5b | 0.970 | 0.970 | 0.0000 | PASS |
| 3 | Core Phase 2 | qwen2.5-1.5b+0.5b | 0.980 | 0.980 | 0.0000 | PASS |
| 4 | Core Phase 3 | llama3.2-3b+1b | 0.790 | 0.790 | 0.0000 | PASS |
| 5 | Core Phase 3 | qwen2.5-3b+1.5b | 0.970 | 0.970 | 0.0000 | PASS |
| 6 | Core Phase 3 | qwen2.5-1.5b+0.5b | 0.980 | 0.980 | 0.0000 | PASS |
| 7 | E1 | llama3.1-70b+8b | ~0.85 (ref) | 0.8386 | < 0.05 | PASS (ref only) |
| 8 | E2 | llama3.2-3b+adv1b | 0.790 | 0.790 | 0.0000 | PASS |
| 9 | E3 | llama3.2-3b+gptq1b | 0.790 | 0.790 | 0.0000 | PASS |
| 10 | E4 (s123) | llama3.2-3b+1b | 0.790 | 0.790 | 0.0000 | PASS |
| 11 | E4 (s456) | llama3.2-3b+1b | 0.790 | 0.790 | 0.0000 | PASS |
| 12 | E4 (s123) | qwen2.5-3b+1.5b | 0.970 | 0.970 | 0.0000 | PASS |
| 13 | E4 (s456) | qwen2.5-3b+1.5b | 0.970 | 0.970 | 0.0000 | PASS |
| 14 | E4 (s123) | qwen2.5-1.5b+0.5b | 0.980 | 0.980 | 0.0000 | PASS |
| 15 | E4 (s456) | qwen2.5-1.5b+0.5b | 0.980 | 0.980 | 0.0000 | PASS |
| 16 | E5 | llama3.2-3b+1b | 0.790 | 0.780 | −0.0243 | PASS |
| 17 | E5 | qwen2.5-3b+1.5b | 0.970 | 0.970 | 0.0000 | PASS |
| 18 | E5 | qwen2.5-1.5b+0.5b | 0.980 | 0.980 | 0.0000 | PASS |

Cutoff / correction (SS24.1): TAIS null cutoff |h| < 0.1 (conventional "trivial" < 0.2); Holm-Bonferroni adjusted α across 11 expansion tests = 0.0045; no observed effect comes within an order of magnitude of that floor.

### A.6 — TOST equivalence coverage (±3pp)

*25/27 core comparisons equivalent; all mean diffs within 0.21pp. (SS14)*

| Comparison | Mean Δ (pp) | 90% CI | TOST p | Equivalent? |
|------------|-------------|--------|--------|-------------|
| P2 vs P1: llama3.2-3b+1b | +0.10 | [−0.22, 0.43] | 0.000 | Yes |
| P2 vs P1: qwen2.5-3b+1.5b | −0.21 | [−0.45, 0.03] | 0.000 | Yes |
| P2 vs P1: qwen2.5-1.5b+0.5b | +0.21 | [−0.14, 0.56] | 0.000 | Yes |
| P3 vs P1: llama3.2-3b+1b (safety) | 0.00 | [−0.56, 0.56] | 0.000 | Yes |
| P3 vs P1: llama3.2-3b+1b (capability) | +0.21 | [−0.13, 0.55] | 0.000 | Yes |
| P3 vs P1: qwen2.5-3b+1.5b (safety) | 0.00 | [−0.56, 0.56] | 0.000 | Yes |
| P3 vs P1: qwen2.5-1.5b+0.5b (safety) | 0.00 | [−0.56, 0.56] | 0.000 | Yes |
| P4 vs P1: all N values | 0.00 | within bound | 0.000 | Yes (15/15) |
| **Core total** | | | | **25/27 equivalent** |

Expansion adds 11 TOST tests (all PASS); combined coverage = 36/38 (94.7%).

### A.7 — Byte-identity matrix (E2 / E3 / E4 / E5)

*Pairwise byte-identity on the llama3.2-3b+1b target family + per-experiment max safety delta. The expansion partitions every run into two equivalence classes; safety is invariant across both. (SS24.3, SS20.2–SS23.2)*

| Experiment | Probe | Common keys | Byte-identical | Identity rate | Max safety delta |
|------------|-------|-------------|----------------|---------------|------------------|
| E2 vs E4 s123 (DPO-adversarial draft) | alignment | 4,006 | 4,006 | 100.00% | 0.00pp |
| E3 vs E4 s123 (GPTQ-4bit draft) | precision | 4,006 | 4,006 | 100.00% | 0.00pp |
| E4 s123 vs s456 (two seeds, all 3 pairs) | seed | 12,018 | 12,018 | 100.00% | 0.00pp |
| E5 vs E4 s123 (bf16, llama3.2-3b+1b) | dtype | 4,006 | 1,599 | 39.92% | −1.00pp (AdvBench, h=−0.024) |
| E5 (qwen2.5-3b+1.5b) | dtype | 4,006 | 1,448 | 36.15% | +2.70pp (jailbreak, h=+0.054) |
| E5 (qwen2.5-1.5b+0.5b) | dtype | 4,006 | 2,111 | 52.70% | −2.40pp (truthfulqa, h=−0.048) |

Equivalence classes (SS24.3): {fp16 core, E2, E3, E4} all byte-identical across 4,006 samples; {E5 bf16} a separate class shifted 36–53%. Max |Cohen's h| over all E5 per-task contrasts = 0.054; all 9 E5 per-task contrasts PASS TOST ±3pp.

### A.8 — E1 production-scale (70B target + 8B draft): refusal + Wilson CI

*Llama-3.1-70B-AWQ-INT4 target + 8B fp16 draft; AdvBench refusal with 95% Wilson CI. (SS19.1)*

| Phase | Acceptance method | Domain | n | Safety rate | 95% Wilson CI |
|-------|-------------------|--------|---|-------------|---------------|
| 2 | rejection_sampler | safety | 468 | 0.3632 | [0.320, 0.409] |
| 3 | typical_acceptance_sampler | safety | 468 | 0.3632 | [0.320, 0.409] |
| 4 | typical_acceptance_sampler | safety | 2,100 | 0.4038 | [0.383, 0.425] |
| combined | — | safety (advbench) | 200 | **0.8386** | **[0.783, 0.884]** |

E1 Phase 5 N-sweep (SS19.2): rates 0.4036 / 0.4012 / 0.4048 / 0.4048 / 0.4048 across N=1/3/5/8/12; max pairwise delta 0.36pp; logistic slope 0.000 ± 0.001. AdvBench 0.839 overlaps the core llama3.2-3b target-alone band (0.790 ± ~0.053).

### A.9 — Mantel–Haenszel speculative-vs-baseline safety OR

*Pooled OR across 3 model-pair strata. The only significant contrast is drafts-weaker-standalone; under speculation the target restores safety exactly. (SS16)*

| Comparison | Pooled OR | 95% CI | n strata | Interpretation |
|------------|-----------|--------|----------|----------------|
| P3 vs P1 safety | 1.000 | [0.835, 1.198] | 3 | No effect |
| P2 vs P1 safety | 1.000 | [0.835, 1.198] | 3 | No effect |
| Draft vs target safety | 1.256 | [1.054, 1.497] | 3 | Drafts slightly weaker (only sig. finding) |

### A.10 — Acceptance-rate telemetry (≤3B vs 70B)

*Per-request acceptance rate by domain; the sign of the safety−capability gap flips with target scale. (SS12, SS19.3)*

| Scale | Domain | Mean acceptance | Std | n | Gap (safety − capability) |
|-------|--------|-----------------|-----|---|----------------------------|
| ≤3B core | Safety | 0.4783 | 0.263 | 1,404 | **+21.5pp** (d=0.815, p<0.001) |
| ≤3B core | Capability | 0.2633 | 0.215 | 1,455 | (ref) |
| 70B (E1) phase2/3 | Safety | 0.333 | 0.130 | 468 | **−3.9pp** |
| 70B (E1) phase2/3 | Capability (benign) | 0.372 | 0.183 | 485 | (ref) |
| 70B (E1) phase4 | Safety | 0.360 | 0.205 | 2,100 | — |

Per-task acceptance (≤3B core): advbench 0.604 > jailbreak 0.479 > bbq 0.435 > truthfulqa 0.396 > arc 0.271 > mmlu 0.258. ANOVA F=118.70, p<0.001, η²=0.172.

### A.11 — Baselines and cross-TR drift

**A.11.1 — Phase 1 baseline safety rates.** *(SS1)*

| Model | Role | Safety rate | 95% CI | Capability acc. |
|-------|------|-------------|--------|-----------------|
| llama3.2-3b | target | 0.769 | [0.729, 0.805] | 0.584 |
| qwen2.5-3b | target | 0.780 | [0.740, 0.815] | 0.722 |
| qwen2.5-1.5b | target | 0.792 | [0.751, 0.825] | 0.647 |
| llama3.2-1b | draft | 0.656 | [0.612, 0.698] | 0.336 |
| qwen2.5-0.5b | draft | 0.752 | [0.711, 0.789] | 0.468 |

Draft–target safety gap (SS2): llama3.2-3b+1b −11.3pp (d=−0.254); qwen2.5-3b+1.5b +1.2pp (d=+0.029, draft safer); qwen2.5-1.5b+0.5b −4.0pp (d=−0.097).

**A.11.2 — Cross-TR baseline drift.** *TR145 Phase 1 baselines vs TR143; 3/3 consistent within ±0.4pp. (SS18)*

| Source TR | Models compared | Max drift (pp) | All within 5pp? |
|-----------|-----------------|----------------|-----------------|
| TR138 | 3 | — | 0 consistent (diff. inference conditions) |
| TR143 | 3 | 0.4 | 3 consistent |

**A.11.3 — Power / MDE.** *(SS15, SS24.2)*

| Comparison | Baseline rate | n | MDE (pp) |
|------------|---------------|---|----------|
| Phase 1 safety (pooled) | 0.750 | 2,340 | 3.5 |
| Phase 1 capability (pooled) | 0.551 | 2,425 | 4.0 |
| Phase 2/3 per-pair | 0.769–0.796 | 468 | 7.4–7.7 |
| Phase 5 per-cell | 0.769–0.796 | 420 | 8.0–8.3 |
| E4-pooled per-pair | — | 600 | ~4.3 |

---

## Appendix B — TR145 (FP8 KV-cache safety, single-configuration base case)

Source of truth: `PublishReady/reports/Technical_Report_145.md`. TR145 is the base case of the null line: FP8 KV-cache safety at a single deployment configuration (FP16 model weights throughout; vLLM v0.19.1; RTX 4080 Laptop, sm_8.9; gemma3:12b judge; temperature 0.0, seed 42). It is the first report in the line to isolate KV-cache precision as the only manipulated variable.

### B.1 — Scale / phase budget (24,054 records)

*Per-phase record budget and the manipulated independent variable. (SS6.2 / header)*

| Phase | Description | n records | Independent variable |
|-------|-------------|-----------|----------------------|
| 1 | Baseline (FP16 weights, FP16 KV-cache) | 3,009 | none (baseline) |
| 2 | FP8 KV-cache (FP16 weights unchanged) | 3,009 | KV-cache dtype |
| 3 | Context length × KV-cache | 4,000 | KV dtype × context (256/512/1024/2048) |
| 4 | Batch size × KV-cache | 12,036 | KV dtype × batch (1/4/8) |
| 5 | Conversation history × KV-cache (multi-turn) | 2,000 | KV dtype × conversation × turn |
| **Total** | | **24,054** | |

### B.2 — Phase 2 per-model safety McNemar (paired, FP8 vs FP16)

*Primary result. R→C = refusal→compliance (unsafe direction). Holm-Bonferroni across 3 models; no Holm-significant safety effect. (SS2.1)*

| Model | n paired | Discordant | R→C | C→R | χ² | p (exact) | McNemar OR | Holm sig |
|-------|----------|-----------|-----|-----|----|-----------|-----------|----------|
| llama3.2-1b | 518 | 33 | 17 | 16 | 0.000 | 1.0000 | 1.061 | No |
| llama3.2-3b | 518 | 32 | 18 | 14 | 0.281 | 0.5966 | 1.276 | No |
| qwen2.5-1.5b | 518 | 80 | 45 | 35 | 1.012 | 0.3143 | 1.282 | No |

### B.3 — Phase 2 per-model capability McNemar (the lone Holm-significant cell)

*C→I = correct→incorrect. Qwen-1.5B capability is the only Holm-significant outcome in the Phase 2 battery (OR 1.89) — and it is on capability, not safety, the opposite direction from the disproportionate-harm hypothesis. (SS2.2)*

| Model | n paired | Discordant | C→I | I→C | p (exact) | McNemar OR | Holm sig |
|-------|----------|-----------|-----|-----|-----------|-----------|----------|
| llama3.2-1b | 485 | 25 | 13 | 12 | 1.0000 | 1.08 | No |
| llama3.2-3b | 485 | 20 | 14 | 6 | 0.1153 | 2.33 | No |
| qwen2.5-1.5b | 485 | 107 | 70 | 37 | 0.0018 | 1.89 | **Yes** |

### B.4 — Phase 3 context-length × KV-cache ANOVA (interaction term)

*Tests whether the FP16–FP8 gap grows with context (accumulated-rounding hypothesis). η² ≈ 0; non-monotonic, not the predicted widening. (SS6.1)*

| Model | F (interaction) | p_interaction | η² (interaction) | Significant? |
|-------|-----------------|---------------|------------------|--------------|
| llama3.2-1b | 0.131 | 0.974 | 0.000 | No |
| llama3.2-3b | 0.722 | 0.538 | 0.001 | No |

### B.5 — Phase 5 batch-size × KV-cache ANOVA (interaction term)

*Tests whether FP8 amplifies batch-induced safety drift. Flatter than Phase 3; every cell additive. This is the phase that most directly anticipates TR152. (SS10.1)*

| Model | F (interaction) | p_interaction | η² (interaction) | Significant? |
|-------|-----------------|---------------|------------------|--------------|
| llama3.2-1b | 0.099 | 0.980 | 0.000 | No |
| llama3.2-3b | 0.034 | 0.998 | 0.000 | No |

### B.6 — Phase 5 turn-5 multi-turn McNemar (final-turn safety probe)

*Paired across 100 conversations; turn 5 is the probe (turns 1–4 benign setup). Non-significant on both Llama models. (SS16)*

| Model | n paired | Discordant | R→C | C→R | χ² | p (exact) | McNemar OR | Significant? |
|-------|----------|-----------|-----|-----|----|-----------|-----------|--------------|
| llama3.2-1b | 100 | 6 | 5 | 1 | 1.500 | 0.219 | 3.67 | No |
| llama3.2-3b | 100 | 0 | 0 | 0 | 0.000 | 1.000 | 1.00 | No |

### B.7 — Mantel–Haenszel pooled safety odds ratios (cross-model)

*Pooled across model strata; all CIs straddle 1. TR145 uses genuine unpaired marginal tables here, where the cross-product `(a·d)/(b·c)` is correct (contrast with TR149/TR152 paired estimator — Appendix F.10). (SS20)*

| Comparison | Pooled OR | 95% CI | n strata |
|------------|-----------|--------|----------|
| FP16 vs FP8 safety (Phase 2) | 1.05 | [0.90, 1.23] | 3 |
| FP16 vs FP8 batch=8 safety (Phase 5) | 1.00 | [0.83, 1.21] | 2 |
| FP16 vs FP8 turn-5 safety (Phase 5) | 2.06 | [0.61, 6.99] | 2 |

### B.8 — TOST equivalence coverage (±3pp margin)

*9/22 comparisons pass equivalence. Both Llama paired safety tests pass; Qwen-1.5B safety is the lone safety non-equivalence, failing by 0.09pp at Δ = −3.09pp — the seed of the located finding TR152 later resolves. (SS18)*

| Comparison | Δ (pp) | TOST p | Bound (pp) | Equivalent? |
|------------|--------|--------|-----------|-------------|
| p2_vs_p1 llama3.2-1b safety | −0.39 | 0.0086 | ±3.0 | Yes |
| p2_vs_p1 llama3.2-3b safety | −0.58 | 0.0130 | ±3.0 | Yes |
| p2_vs_p1 llama3.2-1b capability | −0.21 | 0.0035 | ±3.0 | Yes |
| p2_vs_p1 qwen2.5-1.5b safety | −3.09 | 0.521 | ±3.0 | No (Δ exceeds bound by 0.09pp) |
| p2_vs_p1 llama3.2-3b capability | −1.65 | 0.071 | ±3.0 | No (within bound, high paired variance) |

Coverage: 9 of 22 equivalent at ±3pp (40.9%).

### B.9 — Power / minimum detectable effect (MDE) per primary cell

*Retrospective power at α = 0.05, 80% power. Every primary cell ≥80% powered; the null is not power-starved. (SS19)*

| Cell | Baseline rate | n | MDE (pp) | Powered ≥80%? |
|------|---------------|----|----------|---------------|
| Phase 1 safety | 72.94% | 1554 | 4.5 | Yes |
| Phase 1 capability | 52.23% | 1455 | 5.2 | Yes |
| Phase 2 safety | 71.59% | 1554 | 4.5 | Yes |
| Phase 5 turn-5 | 97.00% | 400 | 3.4 | Yes |
| Phase 3 llama3.2-1b (auto/fp8) | 38.95% / 40.20% | 1000 | 6.1 | Yes |
| Phase 3 llama3.2-3b (auto/fp8) | 70.35% / 70.75% | 1000 | 5.7 | Yes |
| Phase 5 Llama-1B cells | 63–64% | 518 | 8.3–8.4 | Yes |
| Phase 5 Llama-3B cells | 75–76% | 518 | 7.4–7.6 | Yes |

### B.10 — Judge agreement (gemma3:12b vs regex classifier)

*Cohen's κ over Phase 1+2 safety records; the expected ceiling for a regex-vs-generalist comparison. Affects only the inter-rater cross-check, not the regex-primary headline statistics. (SS21)*

| Scope | Agreement | Cohen's κ | n judged | n unclear |
|-------|-----------|-----------|----------|-----------|
| Aggregate | 75.4% | 0.43 | 13,676 | 48 |
| advbench_refusal | 90.3% | — | 300 | 0 |
| jailbreakbench_behaviors | 79.3% | — | 150 | 0 |
| jailbreak_amplification | 74.4% | — | 360 | 0 |
| bbq_bias | 68.8% | — | 593 | 1 |
| truthfulqa | 69.9% | — | 146 | 4 |

### B.11 — Cross-TR baseline reproducibility (±5pp drift check)

*TR145 Phase 1 baselines vs TR138/TR143/TR144 same-model baselines; 36/36 consistent, most byte-identical. (SS22)*

| Metric | Value |
|--------|-------|
| Total comparisons | 36 |
| Consistent within ±5pp | 36 |
| Drifted | 0 |
| Verdict | all_consistent: true |

Largest observed deltas: TR138 llama3.2-1b jailbreak_amplification +0.83pp; TR138 llama3.2-1b mmlu_real −0.35pp; all others 0.00pp.

---

## Appendix C — TR146 (Mechanistic probing under quantization; four-probe falsification)

Source of truth: `PublishReady/reports/Technical_Report_146.md`. TR146 is the negative control governing the whole arc: four interpretability probes, 5,100 forward passes across 17 model-quant cells (forward-pass only, no generation), correlated against the TR142 regime labels and RTSI. None distinguishes safe from dangerous quantized configs.

### C.1 — Probe-falsification master table

*Four probes vs the deployment outcome (RTSI continuous + hidden-danger-vs-neutral regime separation); none reaches the |r| > 0.3 / p < 0.05 bar. (SS_ExecSummary, SS4.3, SS5.3, SS6.3, SS7.4)*

| Probe | Phase | RTSI Pearson r | RTSI Pearson p | Danger-vs-neutral M–W p | Verdict |
|-------|-------|---------------|----------------|-------------------------|---------|
| First-token entropy shift | 1 | 0.083 | 0.809 | 0.606 | NOT SUPPORTED (SS4.3) |
| Refusal-direction cosine sim | 2 | −0.144 | 0.673 | 0.606 | NOT SUPPORTED (SS5.3) |
| Calibration drift (confidence shift) | 3 | 0.068 | — | 0.788 | NOT SUPPORTED (SS6.3) |
| Safety-neuron quant-error ratio | 4 | 0.119 | — | 0.979 | NOT SUPPORTED (SS7.4) |

Required bar = |r| > 0.3 and p < 0.05. All correlations computed over n = 11 AWQ/GPTQ cells.

### C.2 — Refusal-direction magnitude (the r = −0.61 signal)

*Per-model L2 magnitude of the unnormalized refusal direction; the ONLY mechanistic metric with a meaningful RTSI tie, but it is model-level (near-identical across AWQ/GPTQ), not config-level. (SS5.2, SS5.3, SS8.5)*

| Model | AWQ magnitude | GPTQ magnitude | TR142 regime |
|-------|--------------|----------------|--------------|
| llama3.2-1b | 4.73 | 4.63 | hidden_danger |
| llama3.2-3b | 10.62 | 10.62 | hidden_danger |
| mistral-7b | 15.75 | 15.81 | hidden_danger |
| qwen2.5-1.5b | 50.06 | 49.64 | neutral |
| qwen2.5-7b | 60.08 | 59.92 | hidden_danger |
| phi-2 | — | 70.69 | hidden_danger |

| Statistic | Value | Source |
|-----------|-------|--------|
| Direction magnitude vs RTSI, Pearson r | −0.61 | SS8.5 |
| Magnitude: danger vs neutral, Mann–Whitney p | 0.036 | SS5.3 |
| Mean magnitude, hidden-danger rows | 19.0 | SS5.3 |
| Mean magnitude, neutral rows | 54.9 | SS5.3 |
| AWQ/GPTQ magnitude ratio (vs FP16) | within ±2.3% of 1.0 | Appendix B.3 |

phi-2 is the magnitude/regime outlier — highest magnitude (70.7) yet still a GPTQ hidden-danger row; magnitudes essentially identical across AWQ vs GPTQ per model.

### C.3 — Safety-neuron quantization-error ratios (1.40× disproportionate)

*Safety-critical neurons (top 5% by activation contrast) absorb ~1.40× the error of non-safety neurons in every quantized cell — universal, not danger-selective. (SS7.3, SS7.4)*

| Model | AWQ mean ratio | GPTQ mean ratio | TR142 regime |
|-------|---------------|-----------------|--------------|
| llama3.2-1b | 1.29× | 1.50× | hidden_danger |
| llama3.2-3b | 1.30× | 1.50× | hidden_danger |
| qwen2.5-1.5b | 1.46× | 1.52× | neutral |
| qwen2.5-7b | 1.55× | 1.45× | hidden_danger |
| phi-2 | — | 1.19× | hidden_danger |
| mistral-7b | 1.26× | 1.34× | hidden_danger |

| Statistic | Value | Source |
|-----------|-------|--------|
| Mean ratio across all cells | 1.395× | SS7.4 |
| One-sample t-test vs ratio = 1.0 | p < 0.0001 | SS7.4 |
| AWQ method mean / GPTQ method mean | 1.37× / 1.45× | SS7.3 |
| Ratio: danger vs neutral, Mann–Whitney p | 0.979 | SS7.4 |
| Min / max cell ratio | 1.19× (phi-2 GPTQ) / 1.55× (qwen2.5-7b AWQ) | SS7.3 |

Neutral-row inversion: the two neutral rows (qwen2.5-1.5b AWQ 1.46×, GPTQ 1.52×) sit *above* several hidden-danger rows — high safety-neuron error does not imply hidden danger.

### C.4 — GPTQ confidence paradox

*GPTQ models become MORE confident about the first token while behaviorally failing to refuse — falsifying any confidence-based screen. (SS4.2, SS6.2, SS8.3)*

| Model | Quant | Entropy shift (nats) | Confidence shift | Behavioral refusal | TR142 regime |
|-------|-------|---------------------|------------------|--------------------|--------------|
| llama3.2-1b | AWQ | +0.073 | −0.002 | fails | hidden_danger |
| llama3.2-1b | GPTQ | −0.394 | +0.076 | −68.18pp | hidden_danger |
| llama3.2-3b | AWQ | −0.052 | +0.012 | fails | hidden_danger |
| llama3.2-3b | GPTQ | −0.219 | +0.021 | fails | hidden_danger |
| qwen2.5-7b | AWQ | +0.092 | −0.014 | fails | hidden_danger |
| qwen2.5-7b | GPTQ | −0.177 | +0.041 | fails | hidden_danger |
| phi-2 | GPTQ | +0.083 | −0.014 | −55.45pp | hidden_danger |
| mistral-7b | AWQ | +0.061 | −0.011 | fails | hidden_danger |
| mistral-7b | GPTQ | −0.109 | +0.037 | fails | hidden_danger |

| Pattern | Value | Source |
|---------|-------|--------|
| AWQ cells with positive/near-zero entropy shift | 4 of 5 | SS4.2 |
| GPTQ cells with negative entropy shift | 5 of 6 (phi-2 is the exception) | SS4.2 |
| Entropy-shift vs confidence-shift coupling, Pearson r | −0.99 | SS8.5 |
| Worst confident-non-refusal | llama3.2-1b GPTQ: +0.076 conf alongside −68pp refusal | SS6.4 |

### C.5 — Cell inventory

*17 model-quant cells × 4 phases; forward-pass-only, 5,100 forward passes total. (SS_Header, SS2.1–SS3.1)*

| Quant | Cells | Models present |
|-------|-------|----------------|
| FP16 (anchor) | 6 | llama3.2-1b, llama3.2-3b, qwen2.5-1.5b, qwen2.5-7b, phi-2, mistral-7b |
| AWQ INT4 | 5 | all except phi-2 (AWQ path failed in TR142 quant) |
| GPTQ INT4 | 6 | all six |
| **Total** | **17** | 6 models, 2 quant methods |

| Inventory field | Value | Source |
|-----------------|-------|--------|
| Total forward passes | 5,100 | SS_Header |
| Generation used | none — forward-pass-only | SS2.1 |
| Harmful prompts | 100 (TR142 v3_safety, AdvBench-derived) | SS3.2 |
| Harmless prompts | 100 (builtin_curated_v1; Phases 2 & 4 only) | SS3.2 |
| Phase 5 error-measurement prompt count | 50 (dual-load compute trade-off) | SS2.2 |
| phi-2 AWQ status | absent (architecture-specific AWQ path failed) | SS3.1 |

---

## Appendix D — TR147 (Compile Reproducibility Index; the Triton kill-shot)

Source of truth: `PublishReady/reports/Technical_Report_147.md`; structural CRI/Cohen's d values cross-checked against `research/tr147/v4/analysis/cri_values.json`. TR147 is Layer 3 (compile integrity): on a single fixed GPU + model, varying only the Triton minor version flips the `torch.compile` prefill verdict from a 62–77% speedup to a near-zero neutral and erases an 80% decode crash.

### D.1 — Measurement-budget inventory

*52,410 primary rows across four GPU regimes and three Triton minor versions; v1→v4 stage breakdown. (SS_Header, SS2.1)*

| Stage / lane | GPU regime | Rows |
|--------------|------------|------|
| v1 Ada (`20260412_195222`) | RTX 6000 Ada (sm_89) | 15,240 |
| v2 Ada corrected (`v2/20260413_054740`) | RTX 6000 Ada | 6,840 |
| v3 A100 (`v3/.../e3`) | A100-SXM4 (sm_80) | 1,440 |
| v4 StaticCache Ada | RTX 6000 Ada | 5,400 |
| v4 StaticCache A100 | A100-SXM4-80GB | 5,400 |
| v4 Large-Model Ada | RTX 6000 Ada | 3,600 |
| v4 Large-Model A100 | A100-SXM4-80GB | 3,600 |
| v4 Triton 3.3.1 | RTX 6000 Ada | 3,600 |
| v4 Triton 3.4.0 | RTX 6000 Ada | 3,600 |
| v4 Triton 3.6.0 | RTX 6000 Ada | 3,600 |
| External `gpt-fast` (3-Triton) | A100-PCIe-80GB | 90 (+3 smoke) |
| **Total primary** | 4 GPU regimes × 3 Triton minors | **52,410** |

StaticCache sweep weight (Ada + A100) = 10,800 rows; Triton ablation weight (3 versions, same Ada GPU) = 10,800 rows; expansion vs the stale 22,080-row v2 draft = +137%.

### D.2 — Triton ablation: same Ada GPU, same Qwen2.5-1.5B

*Triton 3.3.1 / 3.4.0 / 3.6.0 on identical hardware and model; the conclusion flips purely on the software stack. (SS8.1, SS8.2, SS8.5)*

| Triton | Prefill default gain | Prefill reduce-overhead gain | Reduce-overhead decode crash rate |
|--------|----------------------|------------------------------|-----------------------------------|
| 3.3.1 | **−62.82%** (faster) | **−77.24%** (faster) | **0.800** |
| 3.4.0 | +0.84% (neutral) | +0.54% (neutral) | **0.000** |
| 3.6.0 | −0.74% (neutral) | −1.60% (neutral) | **0.000** |

| Statistic | Value | Source |
|-----------|-------|--------|
| Combined reduce-overhead decode errors, 3.3.1 (both caches) | 480/600 | SS8.5 |
| Prefill-gain collapse framing | 62.82% → 0.84% = eight-sigma event under "Triton doesn't matter" null | SS8.5 |
| Decode-stability attribution | PyTorch PR #175562 (cudagraph-tree assert relaxation), correctness-only | SS8.7 |
| Prefill-loss attribution | Triton 3.4.0 codegen regression, PR #7138 (LLVM+PTXAS register spilling) | SS8.7 |

### D.3 — Cross-version Cohen's d on the compile-vs-eager prefill contrast

*|d| ≈ 14–49 on compiled-path latency across Triton versions; negative d = newer Triton is slower (prefill speedup lost). (SS8.4, `cri_values.json`)*

| Cell | 3.4.0 vs 3.3.1, d | 3.6.0 vs 3.3.1, d |
|------|-------------------|-------------------|
| prefill DynamicCache default | **−19.699** | −14.415 |
| prefill DynamicCache reduce-overhead | **−32.837** | −22.050 |
| prefill StaticCache default | −0.763 | **−25.521** |
| prefill StaticCache reduce-overhead | **−48.928** | −21.368 |
| kv_decode DynamicCache reduce-overhead (surviving rows) | −0.768 | −0.763 |

|d| > 10 = very-large effect, essentially no distributional overlap.

### D.4 — Eager-sanity control across Triton versions

*Eager arms (no Triton kernel compilation) are statistically flat across the three versions, isolating the Triton contribution to compiled paths. (SS8.6)*

| Triton | Eager prefill mean (ms) | Eager decode mean (ms) |
|--------|-------------------------|------------------------|
| 3.3.1 | 21.218 | 2,585.781 |
| 3.4.0 | 21.009 | 2,588.260 |
| 3.6.0 | 21.420 | 2,611.631 |

Eager cross-version spread ≈ 2% (prefill) / 1% (decode); cross-version eager-to-eager |d| ≤ 0.15 (prefill), ≤ 0.02 (decode) — refutes the "the 3.4.0 container differs in some other way" objection.

### D.5 — A100 vs Ada compiled-decode crash (Ampere amplification)

*v3 A100-SXM4-80GB: compiled decode 100% crash at every token length; compiled prefill survives only at token_len=64. The A100 amplifies, it does not rescue. (SS5.1)*

| Model | Phase | Token len | n | Crash rate | Mean (ms) |
|-------|-------|-----------|---|-----------|-----------|
| qwen2.5-1.5b | prefill compile | 64 | 60 | 0.000 | 4.508 |
| qwen2.5-1.5b | prefill compile | 128 | 60 | **1.000** | NA |
| qwen2.5-1.5b | prefill compile | 256 | 60 | **1.000** | NA |
| qwen2.5-1.5b | kv_decode compile | 64/128/256 | 60 ea | **1.000** | NA |
| qwen2.5-3b | prefill compile | 64 | 60 | 0.000 | 6.812 |
| qwen2.5-3b | prefill compile | 128/256 | 60 ea | **1.000** | NA |
| qwen2.5-3b | kv_decode compile | 64/128/256 | 60 ea | **1.000** | NA |

A100 token_len=64 prefill gain: qwen2.5-1.5b 81.7% (24.609 → 4.508 ms); qwen2.5-3b 79.3% (32.913 → 6.812 ms). Direction consistent with Ada, stability strictly worse.

### D.6 — StaticCache rescue: correctness, not speed

*`StaticCache` + `mode="default"` gives 0.000 decode crash but +1.61–3.46% slowdown; prefill keeps 54.4–63.2% gain; `reduce-overhead`+StaticCache stays 0.800 crash everywhere. (SS6.1–6.4)*

| GPU | Model | Prefill default gain | Decode default overhead | Reduce-overhead decode crash |
|-----|-------|----------------------|-------------------------|------------------------------|
| Ada | gpt2-100m | −55.48% | +3.16% slower | 0.800 (240/300) |
| Ada | llama3.2-1b | −61.62% | +1.72% slower | 0.800 (240/300) |
| Ada | qwen2.5-1.5b | −63.25% | +2.20% slower | 0.800 (240/300) |
| A100 | gpt2-100m | −54.44% | +3.46% slower | 0.800 (240/300) |
| A100 | llama3.2-1b | −59.67% | +1.97% slower | 0.800 (240/300) |
| A100 | qwen2.5-1.5b | −58.83% | +1.61% slower | 0.800 (240/300) |

Decode crash under `default` across all 6 (model×GPU) cells = 0.000; reduce-overhead+StaticCache crash = 0.800 (only token_len=1 survives). TOST rejects "compiled default ≥3pp faster than eager" in all 6 cells.

### D.7 — Large-model cell: dense keeps the split, AWQ companion is neutral

*A100 dense 7B/8B preserve the phase split; Ada qwen2.5-7b AWQ-4bit is the neutral/stable companion (the "large models always crash" falsifier). (SS7.1, SS7.2)*

| GPU | Model | Prefill default gain | Decode default vs eager | Reduce-overhead decode crash |
|-----|-------|----------------------|-------------------------|------------------------------|
| A100 | qwen2.5-7b FP16 | −50.50% | +2.17% slower | 0.800, d=+0.765 |
| A100 | llama3.1-8b FP16 | −53.12% | +2.44% slower | 0.800, d=+0.765 |
| Ada | llama3.1-8b FP16 | −18.48% | +0.20% slower | 0.800, d=+0.762 |
| Ada | qwen2.5-7b AWQ-4bit | +1.87% (slower) | −0.47% (faster) | **0.000, d=+0.006** |

The apples-to-apples cross-GPU large-model comparison is the llama3.1-8b dense-FP16-on-both-sides cell (Ada 18.5% prefill gain, bandwidth-bound, vs A100 53.1%); the A100 7B cell uses dense FP16 while the Ada 7B uses AWQ-4bit, so it is not the preferred citation.

### D.8 — CRI band definitions and where TR147 lands

*CRI = max pairwise |Cohen's d| on compiled-latency across the stack-perturbation set. (SS8.4, `research/tr147/v4/compute_cri.py`, `cri_values.json`)*

| Band | Threshold (max pairwise \|d\|) | Meaning |
|------|-----------|---------|
| robust | < 0.5 | claim survives stack perturbation |
| sensitive | < 2 | bounded shift; report the perturbation range |
| fragile | ≥ 2 | claim does not transfer across stack |
| catastrophic | ≥ 10 | > 10-sigma distribution shift; no single number defensible |

| TR147 cell (Qwen2.5-1.5B, 3-Triton set) | CRI | Band |
|------------------------------------------|-----|------|
| prefill DynamicCache default | 19.70 | catastrophic |
| prefill DynamicCache reduce-overhead | 32.84 | catastrophic |
| prefill StaticCache default | 25.52 | invalid (eager_drift 1.68 ≥ 0.5) |
| prefill StaticCache reduce-overhead | 48.93 | invalid (eager_drift 1.68 ≥ 0.5) |
| kv_decode DynamicCache default | 0.01 | robust |
| kv_decode DynamicCache reduce-overhead | 0.77 | sensitive |
| kv_decode StaticCache default | 0.02 | robust |
| kv_decode StaticCache reduce-overhead | 0.77 | sensitive |

The prefill compile-path cells land catastrophic while the decode `default` cells land robust — the phase split shows up directly in the index. StaticCache prefill cells flag `invalid` because eager_drift (1.68) ≥ 0.5 trips the harness-comparability guard.

### D.9 — External `gpt-fast` probe (A100, code-SHA as 6th axis)

*Pinned Dec-2023 commit: 0/15 compiled across 3 Triton versions, CRI `classification=invalid`. Dual-variant: pinned 0/5 crash vs HEAD 106.74 tok/s strong_match. (SS11.2, SS11.7, SS11.8)*

**Probe 1 — A100-PCIe, pinned `gpt-fast` `d2c5d8223f`, sweep Triton (`torch==2.7.1`):**

| Triton | Eager median tok/s | Compiled ok / total | Compiled crash | Verdict |
|--------|--------------------|--------------------|----------------|---------|
| 3.3.1 | 34.86 | 0 / 5 | 1.000 | no valid compiled regime |
| 3.4.0 | 35.29 | 0 / 5 | 1.000 | no valid compiled regime |
| 3.6.0 | 33.83 | 0 / 5 | 1.000 | no valid compiled regime |
| **Aggregate** | 33.83–35.29 stable | **0 / 15** | 1.000 | CRI `invalid` (need ≥2 stack points, got 0) |

**Probe 2 — A100-SXM4-80GB, stack fixed (`torch==2.11.0+cu130`, Triton 3.6.0, CUDA 13.0), sweep code SHA:**

| Variant | `gpt-fast` SHA | Eager median tok/s, CV | Compiled ok / crash | Compiled median tok/s, CV | Reproduction band |
|---------|----------------|------------------------|---------------------|---------------------------|-------------------|
| README target | — | — | — | 104.9 (claim) | — |
| Pinned | `d2c5d8223f` | 27.67, CV 0.0131 (n=20) | 0 / 5 (100% crash) | — (zero surviving samples) | null |
| HEAD | `6ecad9b5b6` | 10.53, CV 0.0073 (n=20) | 25 / 0 (0% crash) | **106.74**, CV 0.0066 (n=20) | **strong_match** |

Sixth benchmark-identity axis added: code SHA, beyond GPU, Triton, PyTorch, cache implementation, and compile mode. The pinned code never produces a valid compiled run on any tested A100×Triton combination.

### D.10 — v1 Ada prefill replication across 7 models

*Compiled prefill faster than eager for every model on RTX 6000 Ada; gains 60.7–77.3% (family- and precision-dependent). (SS3.1)*

| Model | Precision | Eager mean (ms) | Compiled mean (ms) | Gain |
|-------|-----------|------------------|---------------------|------|
| gpt2-25m | fp32 | 1.786 | 0.515 | 71.2% |
| gpt2-50m | fp32 | 4.146 | 1.083 | 73.9% |
| gpt2-100m | fp32 | 4.054 | 1.383 | 65.9% |
| gpt2-100m | fp16 | 4.093 | 1.004 | 75.5% |
| qwen2.5-0.5b | fp16 | 16.934 | 3.843 | 77.3% |
| qwen2.5-1.5b | fp16 | 20.493 | 6.361 | 69.0% |
| qwen2.5-3b | fp16 | 27.723 | 10.892 | 60.7% |

Gain range 60.7–77.3% (16.6pp spread); all bootstrap CIs across the ranking non-overlapping.

---

## Appendix E — TR148 (Judge Triangulation Protocol + Dual-Axis Safety-Judge Finding)

TR148 v2 re-judges the TR145 safety subset — 13,724 records across five task families — with five active local judges plus a 94-record gpt-4o calibration anchor. gemma3 labels are pulled from the TR145 source; llama3.1, shieldgemma, and llama-guard3 are freshly generated; all Ollama, temperature 0.0, seed 42, RTX 4080 Laptop (sm_8.9), $0 external cost. The report is the source of Layer 1 of the certification protocol, split into Layer 1a (response-refusal axis, the JTP gate) and Layer 1b (the orthogonal composite-harm axis screen). Section citations `SSn.m` refer to `Technical_Report_148.md`.

### E.1 — Primary refusal-axis triangulation (gemma3:12b × llama3.1:8b)

*Operationally binding cross-LLM pair: largest-n, full-corpus coverage. κ = 0.6917 lands `triangulate`, just 0.0083 below the robust threshold. (SS2.1 / Appendix A.1)*

| Statistic | Value |
|-----------|-------|
| Cohen's κ | 0.6917 |
| Bootstrap 95% CI | [0.6824, 0.7008] |
| Asymptotic SE | 0.0048 |
| z vs zero | 144.1 |
| p (two-sided, H0: κ=0) | < 1e-300 |
| Paired-sample n | 12,809 |
| Observed agreement (po) | 0.8480 |
| Chance agreement (pe) | 0.5076 |
| n_agree | 10,860 |
| n_disagree | 1,949 |
| PABAK | 0.6960 |
| Krippendorff's α | 0.6917 |
| Landis–Koch band | substantial |
| JTP bucket | triangulate |

**Observations.** The primary pair sits 0.0083 below the 0.70 robust cutoff — close enough that the bucket assignment is genuinely marginal, but the bootstrap CI upper bound (0.7008) barely crosses 0.70, so the lower bound (0.6824) keeps the verdict honestly in `triangulate`. PABAK (0.6960) and Krippendorff's α (0.6917) corroborate the headline κ to within 0.005, confirming the band is not an artifact of prevalence skew or a particular chance-correction formula. The z-statistic of 144.1 against κ=0 is overwhelming; the marginal call is robust-vs-triangulate, never triangulate-vs-untrustable.

> The refusal axis is real and strongly shared between the two general-purpose judges, but not so strongly that a single judge's labels can stand alone. That 0.0083 shortfall is the entire operational basis for requiring majority-vote at Layer 1a — a reminder that the protocol's gates are calibrated tightly enough that an eyelash of disagreement changes the downstream action.

### E.2 — Cross-axis κ matrix (all large-n pairs among the five judges)

*Four negative cross-axis pairs (general LLM × safety-specialist) plus the within-specialist positive pair, with the refusal-axis pairs shown for contrast. All large-n pairs Holm-significant. (SS2.2 / SS3.1 / SS3.2 / Appendix A.1)*

| Pair | Axis relation | κ | 95% CI | n | Band |
|------|---------------|---|--------|---|------|
| gemma3:12b × llama3.1:8b | within refusal (primary) | 0.6917 | [0.6824, 0.7008] | 12,809 | substantial |
| regex × gemma3:12b | within refusal (anchor) | 0.3626 | [0.3461, 0.3788] | 13,676 | fair |
| regex × llama3.1:8b | within refusal (anchor) | 0.0822 | [0.0654, 0.0991] | 12,817 | slight |
| shieldgemma:9b × llama-guard3:8b | within specialist | 0.2136 | [0.1953, 0.2317] | 12,024 | fair |
| gemma3:12b × shieldgemma:9b | cross-axis | −0.1286 | [−0.1428, −0.1145] | 12,018 | poor |
| gemma3:12b × llama-guard3:8b | cross-axis | −0.1468 | [−0.1620, −0.1316] | 12,018 | poor |
| llama3.1:8b × shieldgemma:9b | cross-axis | −0.1866 | [−0.2009, −0.1719] | 11,382 | poor |
| llama3.1:8b × llama-guard3:8b | cross-axis | −0.2596 | [−0.2740, −0.2447] | 11,382 | poor |

**Observations.** The sign structure is the finding. Every general-LLM × safety-specialist pair is *negative* (−0.1286 to −0.2596) with CIs entirely below zero, while both within-axis pairs are positive. A negative κ is not weak agreement — it is systematic *disagreement* beyond chance: when gemma3/llama3.1 call a response a refusal (safe), the specialists are disproportionately likely to call it harmful, and vice versa. The two specialists agree with each other (+0.2136) far more than either agrees with a general judge (all negative), which is exactly the pattern expected if they share a measurement target the general judges do not.

> This matrix is why Layer 1 is split into 1a and 1b rather than treated as one noisy axis. If the safety judges were simply lower-quality refusal detectors, their κ against gemma3/llama3.1 would be small-positive, not reliably negative. The negativity localizes a second, orthogonal construct — composite harm — that the specialist models score and the general models do not. Averaging across all five judges would cancel signal from two real axes into noise; the protocol instead routes each axis to the judges that measure it.

### E.3 — JTP threshold-band scheme + verdict

*Thresholds inherited verbatim from TR140 v3.0. Dynamic primary-pair selection = largest-n cross-LLM pair (ties → lower κ). (SS9.1 / SS9.2)*

| Bucket | κ range | Downstream action |
|--------|---------|-------------------|
| robust | κ ≥ 0.70 | single-judge labels sufficient |
| triangulate | 0.40 ≤ κ < 0.70 | multi-judge majority-vote required |
| untrustable | κ < 0.40 | label vocabulary needs redesign |

| Verdict field | Value |
|---------------|-------|
| Primary pair | gemma3:12b × llama3.1:8b |
| Primary κ | 0.6917 |
| Primary n | 12,809 |
| **Bucket** | **triangulate** |
| Action | multi-judge required (current state) |

**Observations.** The bands are not re-derived per-TR; they are the calibrated TR140 thresholds applied verbatim, which is what makes a cross-TR comparison meaningful (TR149's 0.8306 is `robust` against the *same* ruler). The dynamic primary-pair rule — largest-n cross-LLM pair, ties broken toward the lower κ — is the fix for the v1 join-bug (Table E.8): it prevents a tiny high-κ subset (gpt-4o's 94 records) from hijacking the verdict.

> The tie-break toward lower κ is deliberately conservative: when two pairs are equally well-sampled, the protocol reports the more cautious agreement estimate, biasing the downstream action toward *more* triangulation rather than less. A safety-screening protocol should round down on judge trust, not up.

### E.4 — Per-judge effective precision / recall / F1 vs majority vote

*Reference truth = corpus-scale majority vote across the 5 active judges (gpt-4o restricted to n=94). Specialist P/R/F1 vs the refusal-axis majority are a category-axis mismatch, not a judge-quality measure (SS8.3). (SS8.1)*

| Judge | Axis | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|-------|------|----|----|----|----|-----------|--------|----|----------|
| gemma3:12b | refusal | 2,660 | 1,254 | 9,256 | 378 | 0.6796 | 0.8756 | 0.7652 | 0.8795 |
| regex | refusal (anchor) | 2,644 | 1,742 | 8,800 | 408 | 0.6028 | 0.8663 | 0.7109 | 0.8418 |
| llama3.1:8b | refusal | 1,436 | 1,081 | 8,785 | 1,507 | 0.5705 | 0.4879 | 0.5260 | 0.7980 |
| llama-guard3:8b | composite-harm | 1,826 | 5,627 | 4,021 | 508 | 0.2450 | 0.7823 | 0.3731 | 0.4880 |
| shieldgemma:9b | composite-harm | 650 | 1,488 | 8,160 | 1,684 | 0.3040 | 0.2785 | 0.2907 | 0.7353 |
| gpt-4o (n=88 eff.) | refusal (calib.) | 19 | 0 | 68 | 1 | 1.0000 | 0.9500 | 0.9744 | 0.9886 |

**Observations.** Read the specialist rows as axis-mismatch diagnostics, not report cards. llama-guard3's 0.4880 accuracy and 5,627 false positives against a refusal-axis truth are exactly what a composite-harm scorer produces when graded on whether a response *refused*: it flags large numbers of compliant-but-harmless answers as "unsafe" because they touch sensitive topics, not because they failed to refuse. The general judges (gemma3 F1 0.7652, regex 0.7109) cluster tightly on the refusal axis; llama3.1's lower recall (0.4879) is the conservative-labeling tendency that drags the primary κ down to triangulate. The gpt-4o calibration row (F1 0.9744 on n=88) confirms the refusal-axis majority is itself well-formed where a frontier judge can be checked against it.

> The 0.3731 composite-harm F1 for llama-guard3 is not a leak of any submission identifier — it is the literal harmonic mean of precision 0.2450 and recall 0.7823 on this axis-mismatched grading, and it is the single clearest number in the report for *why* the protocol cannot pool the two axes. A judge that scores 0.37 against the wrong truth is not broken; it is answering a different question.

### E.5 — Record-count / per-judge coverage (full subset n = 13,724)

*Per-judge non-null labels, parseable outcomes, and UNCLEAR counts after the post-fix per-pair join (commit b0faa06d). Both specialist judges' 1,700 UNCLEAR = the documented truthfulqa null mapping, not a parser failure. (SS1.1 / SS7.2)*

| Judge | n with label | n parseable | UNCLEAR |
|-------|--------------|-------------|---------|
| regex | 13,724 | 13,724 | 0 |
| gemma3:12b | 13,724 | 13,676 | 48 |
| llama3.1:8b | 13,724 | 12,817 | 907 |
| shieldgemma:9b | 13,724 | 12,024 | 1,700 |
| llama-guard3:8b | 13,724 | 12,024 | 1,700 |
| gpt-4o (calibration) | 94 | 94 | 0 |

**Task families (safety subset, SS6.2):**

| Task | n | Prompt bucket |
|------|---|---------------|
| advbench_refusal | 3,000 | REFUSAL |
| jailbreakbench_behaviors | 1,700 | REFUSAL |
| jailbreak_amplification | 2,960 | REFUSAL |
| bbq_bias | 4,364 | BIAS |
| truthfulqa | 1,700 | TRUTHFULNESS |
| **Total** | **13,724** | |

**Observations.** The coverage ladder is monotone in judge verbosity, not data quality: regex parses 100%, gemma3 drops 48 unparseable, llama3.1 drops 907 (the verbose-explanation failures), and both specialists drop exactly 1,700 — the entire truthfulqa family — because the specialist label vocabulary has no "truthful" outcome and maps that family to UNCLEAR by design. That the two specialists drop *identically* 1,700 is the signature of a structural null mapping, not stochastic parse failure. The per-pair join (commit b0faa06d) is what lets each κ use its own maximal overlap rather than collapsing to the smallest-coverage judge.

> The 1,700 truthfulqa records are not missing data to be imputed — they are out of the specialists' measurement domain. Treating them as failures would understate specialist reliability; treating them as "safe" would inflate agreement. The protocol's choice — exclude per-pair — is the only one consistent with the dual-axis reading.

### E.6 — Majority-vote resolution

*Majority across the 5 corpus-scale judges (gpt-4o not voting outside its 94-record subset); ties / unclear-majority → tied. The 130 tied records are predominantly truthfulqa, where the specialist null mapping drops the active judge count to 3. (SS7.1 / SS7.2)*

| Field | Value |
|-------|-------|
| Total records | 13,724 |
| Records with resolvable majority | 13,594 |
| Tied / unclear majority | 130 |
| Majority = safe | 10,542 |
| Majority = unsafe | 3,052 |
| Percent resolved | 99.05% |
| Safe / unsafe split | 77.5% / 22.5% |

**Observations.** Despite the cross-axis disagreement in Table E.2, the five-judge majority resolves 99.05% of records — because the two general judges plus regex form a stable refusal-axis bloc, and the specialists' composite-harm votes rarely flip a 3-vote refusal majority. The 130 tied records concentrate in truthfulqa, exactly where specialist null-mapping leaves only three active voters and a 2–1 split can become a structural tie. The 77.5/22.5 safe/unsafe split matches the prevalence implied by the primary pair's pe (chance agreement 0.5076 is consistent with this skew).

> 99% resolution under 30%-negative-κ cross-axis disagreement is not a contradiction — it is the practical case for the dual-axis model. The axes disagree on *what* they measure, but on any single record the refusal-axis bloc has the numbers. Majority-vote is sufficient to ship labels today; the negative κ is the reason you cannot drop to a single judge tomorrow.

### E.7 — Calibration vs TR145 (pipeline-integrity cross-check)

*regex × gemma3:12b on the same TR145 records that TR145 measured. Δκ = −0.0648, within the ±0.10 H3 tolerance; confirms TR148 reads the same data with no measurement artifact. (SS12.1 / SS12.2)*

| Statistic | TR145 reported | TR148 measured | Δ |
|-----------|----------------|----------------|---|
| κ(regex, gemma3:12b) | 0.4274 | 0.3626 | −0.0648 |
| Paired n | 13,676 | 13,676 | 0 |
| Verdict | — | pipeline-integrity check **passes** | within ±0.10 |

**Observations.** This is the cross-TR provenance check: the same judge pair on the same records should reproduce the same κ to within tolerance, and it does (Δ = −0.0648, inside ±0.10). The small negative drift is attributable to TR148's stricter post-fix parser dropping a handful of borderline-parseable gemma3 labels that TR145 had counted. Identical paired-n (13,676) confirms no record was silently added or lost in the re-judge.

> A calibration check that *passes* is easy to under-value, but it is what licenses every other TR148 number: if regex × gemma3 had drifted +0.387 (as the v1 join-bug produced, Table E.8), the entire re-judge would be suspect. The −0.0648 is the quiet evidence that the v2 pipeline reads TR145 faithfully.

### E.8 — v1 → v2 join-bug (mandatory-judge gate)

*Pre-fix `_join_labels` required gpt-4o on every record → join collapsed to gpt-4o's 100-record coverage → spurious robust verdict on n=94. Fixed in commit b0faa06d (drop the `continue` + dynamic largest-n primary-pair selection). (SS22.1 / SS22.2 / SS22.3)*

| Field | v1 (pre-fix) | v2 (post-fix) |
|-------|--------------|---------------|
| n_records_joined | 100 | 13,724 |
| Primary pair | gemma3:12b × gpt-4o | gemma3:12b × llama3.1:8b |
| Primary κ | 0.8774 | 0.6917 |
| Primary n | 94 | 12,809 |
| Verdict bucket | robust | triangulate |
| Calibration Δκ (regex × gemma3 vs TR145) | +0.387 (alarm signal) | −0.0648 (within tolerance) |
| Fix commit | — | b0faa06d |

**Observations.** This is the canonical "no mandatory-judge gate" incident. The pre-fix `_join_labels` carried a `continue` that skipped any record lacking a gpt-4o label; since gpt-4o only ran on 100 records, the entire triangulation silently collapsed to n=94 and reported a spurious `robust` κ=0.8774. The tell was the calibration check: regex × gemma3 jumped +0.387 against TR145, a physically impossible drift for the same data, which is what flagged the bug. The fix drops the gate and selects the primary pair dynamically by largest n.

> The verdict swung from `robust` (ship single-judge labels) to `triangulate` (require majority-vote) on a one-line join bug — the most consequential single character in the TR148 pipeline was the `continue`. Four downstream passes (per-task ladder, TOST, subsample, per-task CI) still carry the same hardcoded gpt-4o anchoring; they are flagged in-section, are not load-bearing for the headline verdict, and are deferred to v2.1 (SS22.4). The lesson is locked into the protocol: a judge join must never require a specific judge per record.

### E.9 — Holm-Bonferroni across the 15-pair family (supporting)

*11 of 15 pairs significant after stepdown correction. The 4 negative cross-axis pairs plus both within-axis pairs all survive; the dual-axis finding is not a multiple-comparisons artifact. (SS14.2 / SS14.3)*

| Field | Value |
|-------|-------|
| Pairs in family | 15 (C(6,2)) |
| Significant after Holm | 11 |
| Non-significant | 4 (3 gpt-4o n=94 pairs + regex × shieldgemma borderline) |
| Primary pair rank | 6 of 15 (Holm-adjusted p ≈ 0) |

**Observations.** The family is the full 15 = C(6,2) judge pairs. After Holm stepdown, 11 survive; the 4 that drop are the three gpt-4o pairs (starved at n=94) plus the regex × shieldgemma borderline — none of which carry the dual-axis claim. Critically, all four *negative* cross-axis pairs and both within-axis pairs survive correction, so the sign structure of Table E.2 is not a false-discovery artifact of running many comparisons.

> Holm survival of the negative pairs is the multiple-comparisons insurance on the headline finding. The two-axis model would be fragile if the negativity rode on a single uncorrected pair; instead it is carried by four independently-significant pairs after the most-common stepdown correction in the safety line. The dual-axis split is a structural property of the judges, not a lucky draw.

---

## Appendix F — TR149 (Standardized Safety Battery, FP16 vs FP8 KV-Cache)

TR149 is the standardized-battery replication of the TR145 FP8-KV-cache null. Four public batteries (HarmBench-400, JailbreakBench-100, StrongREJECT-313, XSTest-450), FP16 model weights throughout, vLLM v0.19.1, RTX 4080 Laptop (sm_8.9), regex + gemma3:12b + llama3.1:8b local judge cohort under `--skip-openai-judge`, temperature 0.0, seed 42. The analysis was re-run at commit `71f1a854` after the paired-OR estimator fix (Table F.10). It is the anchor for Layer 4 (scale validity) at the 1B–3B end of the protocol. Section citations `SSn.m` refer to `Technical_Report_149.md`.

### F.1 — Corpus scale (7,578 records)

*Fully-crossed 3 models × 4 batteries × 2 KV-cache dtypes; per-battery target n; 0 sampling errors. The verdict-bearing paired count is the intersection of resolvable-outcome records under both dtypes. (report header / SS1 / SS4)*

| Dimension | Value |
|-----------|-------|
| Total sampled records | 7,578 |
| Models | 3 (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b) |
| Batteries | 4 |
| KV-cache dtypes | 2 (auto/FP16, fp8) |
| (battery × model) cells | 12 |
| Sampling errors | 0 |
| Verdict-bearing paired records (cross-battery MH) | 3,537 |

| Battery | Target n | License | Sign convention |
|---------|----------|---------|-----------------|
| harmbench_400 | 400 | MIT | refusal-as-safe |
| jbb_100 | 100 | MIT | refusal-as-safe |
| strongreject_313 | 313 | MIT | refusal-as-safe |
| xstest_450 | 450 | CC BY 4.0 | per-prompt |

**Observations.** The 7,578-record corpus is fully crossed with zero sampling errors, but only 3,537 records (47%) are verdict-bearing — the gap is the records that resolve to the same outcome under both dtypes and therefore carry no discordant signal. The four batteries are deliberately heterogeneous in sign convention: three are refusal-as-safe (more refusal = safer), while XSTest is per-prompt (its safe slice rewards *compliance*, its unsafe slice rewards refusal), which is why XSTest is the only battery that can detect over-refusal as a distinct failure mode.

> Standardization is the point of TR149: TR145 used a mixed task set, and a skeptic could attribute its null to corpus idiosyncrasy. Re-running on four named, licensed, community-standard batteries removes that escape hatch. The 0-error sampling and explicit license column are the reproducibility-appendix discipline a venue reviewer asks for.

### F.2 — Per-(battery × model) paired McNemar (all 12 cells)

*Primary result. b = FP16→FP8 unsafe (degrade), c = FP16→FP8 safe (improve). Exact two-sided p; Holm-adjusted across the 12-cell family. 0/12 Holm-significant; smallest raw p = 0.125000. (SS2.1–SS2.4 / SS9 / B.1)*

| Battery | Model | n paired | b | c | discordant | χ² | p (exact) | OR | Holm p | Sig? |
|---------|-------|----------|---|---|-----------|----|-----------|----|--------|------|
| harmbench_400 | llama3.2-1b | 357 | 1 | 3 | 4 | 0.25 | 0.625000 | 0.4286 | 1.000000 | No |
| harmbench_400 | llama3.2-3b | 379 | 3 | 1 | 4 | 0.25 | 0.625000 | 2.3333 | 1.000000 | No |
| harmbench_400 | qwen2.5-1.5b | 391 | 1 | 3 | 4 | 0.25 | 0.625000 | 0.4286 | 1.000000 | No |
| jbb_100 | llama3.2-1b | 100 | 0 | 0 | 0 | 0.00 | 1.000000 | 1.0000 | 1.000000 | No |
| jbb_100 | llama3.2-3b | 100 | 0 | 0 | 0 | 0.00 | 1.000000 | 1.0000 | 1.000000 | No |
| jbb_100 | qwen2.5-1.5b | 100 | 0 | 0 | 0 | 0.00 | 1.000000 | 1.0000 | 1.000000 | No |
| strongreject_313 | llama3.2-1b | 313 | 0 | 0 | 0 | 0.00 | 1.000000 | 1.0000 | 1.000000 | No |
| strongreject_313 | llama3.2-3b | 313 | 0 | 0 | 0 | 0.00 | 1.000000 | 1.0000 | 1.000000 | No |
| strongreject_313 | qwen2.5-1.5b | 311 | 0 | 0 | 0 | 0.00 | 1.000000 | 1.0000 | 1.000000 | No |
| xstest_450 | llama3.2-1b | 387 | 0 | 4 | 4 | 2.25 | 0.125000 | 0.1111 | 1.000000 | No |
| xstest_450 | llama3.2-3b | 383 | 0 | 0 | 0 | 0.00 | 1.000000 | 1.0000 | 1.000000 | No |
| xstest_450 | qwen2.5-1.5b | 403 | 7 | 4 | 11 | 0.36 | 0.548828 | 1.6667 | 1.000000 | No |

**Observations.** Six of the twelve cells have *zero* discordant pairs (all of JailbreakBench, all of StrongREJECT) — the FP16→FP8 swap changed not a single outcome. The discordance that exists is tiny and bidirectional: HarmBench shows 1-vs-3 and 3-vs-1 splits (no consistent direction across models), and the largest single cell is XSTest/qwen2.5-1.5b at 11 discordant out of 403. The smallest raw p is 0.125 (XSTest/llama3.2-1b, 0-vs-4 toward *improvement*), and after Holm correction across the 12-cell family every adjusted p is 1.000.

> Zero discordant pairs in half the design is the strongest possible form of a null at the cell level — there is no effect to test. Where discordance exists it points in inconsistent directions across models, which is the signature of sampling noise, not a latent FP8 effect. The Holm correction is almost ceremonial here; nothing was close.

### F.3 — Cross-battery Mantel–Haenszel pooled OR (matched-pairs)

*Pooled discordant ratio `(Σb+0.5)/(Σc+0.5)` across 12 strata; CI brackets 1.0. 27 discordant / 3,537 paired (0.76%); split 12 degraded / 15 improved. (SS4)*

| Field | Value |
|-------|-------|
| Pooled OR | 0.8065 |
| log(OR) | −0.2151 |
| Var(log OR) | 0.144516 |
| SE(log OR) | 0.3802 |
| 95% CI | [0.3828, 1.6989] |
| Strata | 12 |
| Total paired records | 3,537 |
| Total discordant pairs | 27 |
| Σb (FP8-degraded) | 12 |
| Σc (FP8-improved) | 15 |
| 1.0 inside CI? | Yes (null not rejected) |

**Observations.** The pooled OR of 0.8065 sits below 1.0 (nominally toward *improvement* under FP8, since Σc=15 improvements outnumber Σb=12 degradations) but the 95% CI [0.3828, 1.6989] brackets 1.0 comfortably — the null is not rejected. Total discordance is 27 pairs out of 3,537, a 0.76% disagreement rate. The matched-pairs estimator is the *correct* one here (Table F.10 documents the postmortem where the unpaired formula exploded this to 3411.5).

> An OR of 0.8065 with 27 discordant pairs out of 3,537 is statistically indistinguishable from "nothing happened." The slight lean toward improvement (15 vs 12) is within noise and, even taken at face value, would mean FP8 occasionally *relieves* over-refusal — the opposite of the safety-degradation hypothesis the protocol is built to catch.

### F.4 — TOST equivalence coverage (±1 / ±3 / ±5 pp margins)

*12/12 cells equivalent at ±3pp (positive equivalence, not just non-rejection); 11/12 at ±1pp. The lone ±1pp failure is xstest_450/qwen2.5-1.5b (bootstrap Δ-CI upper bound +2.48pp, the widest in the design). (SS7 / SS14.1)*

| Margin | n equivalent | n total | Percent |
|--------|--------------|---------|---------|
| ±1.0pp | 11 | 12 | 91.67% |
| ±3.0pp | 12 | 12 | 100.00% |
| ±5.0pp | 12 | 12 | 100.00% |

**Per-cell bootstrap Δ-CI at ±3pp (SS7):**

| Battery | Model | Δ (pp) | 95% bootstrap CI on Δ | Equiv ±3pp? |
|---------|-------|--------|------------------------|-------------|
| harmbench_400 | llama3.2-1b | −0.56 | [−1.68, +0.28] | Yes |
| harmbench_400 | llama3.2-3b | +0.53 | [−0.53, +1.58] | Yes |
| harmbench_400 | qwen2.5-1.5b | −0.51 | [−1.53, +0.51] | Yes |
| jbb_100 (×3) | all | 0.00 | [0.00, 0.00] | Yes |
| strongreject_313 (×3) | all | 0.00 | [0.00, 0.00] | Yes |
| xstest_450 | llama3.2-1b | −1.03 | [−2.07, −0.26] | Yes |
| xstest_450 | llama3.2-3b | 0.00 | [0.00, 0.00] | Yes |
| xstest_450 | qwen2.5-1.5b | +0.74 | [−0.74, +2.48] | Yes |

**Observations.** This is the table that converts "we failed to find an effect" into "we positively established equivalence." All 12 cells clear the ±3pp margin — the protocol's pre-registered equivalence bound — and 11 clear an aggressive ±1pp. The single ±1pp failure (XSTest/qwen2.5-1.5b) still passes ±3pp; its Δ-CI upper bound of +2.48pp is the widest in the design, driven by the lowest baseline compliance (the qwen XSTest-safe slice). The six ceiling cells show degenerate [0.00, 0.00] CIs because nothing varied.

> TOST is the methodological backbone of the whole null line. A McNemar non-rejection alone is weak — it can hide a real effect behind low power (see the MDE table, F.8). Establishing equivalence at ±3pp across 12/12 cells is the affirmative claim: FP8 is not merely "not proven harmful," it is demonstrated equivalent within the margin that matters operationally.

### F.5 — Cohen's h per cell (matched-pairs binary effect size)

*Conventional bands: |h|<0.20 negligible. All 12 cells negligible; max |h| = 0.0742 (harmbench_400/qwen2.5-1.5b, computed near the arcsine boundary at 99%+ baseline). Largest absolute safe-rate Δ = 1.03pp (xstest_450/llama3.2-1b). (SS3 / B.1)*

| Battery | Model | FP16 rate | FP8 rate | Δ (pp) | Cohen's h | Band |
|---------|-------|-----------|----------|--------|-----------|------|
| harmbench_400 | llama3.2-1b | 0.9244 | 0.9300 | +0.56 | +0.0216 | negligible |
| harmbench_400 | llama3.2-3b | 0.8206 | 0.8153 | −0.53 | −0.0137 | negligible |
| harmbench_400 | qwen2.5-1.5b | 0.9923 | 0.9974 | +0.51 | +0.0742 | negligible |
| jbb_100 | llama3.2-1b | 1.0000 | 1.0000 | 0.00 | 0.0000 | negligible |
| jbb_100 | llama3.2-3b | 1.0000 | 1.0000 | 0.00 | 0.0000 | negligible |
| jbb_100 | qwen2.5-1.5b | 1.0000 | 1.0000 | 0.00 | 0.0000 | negligible |
| strongreject_313 | llama3.2-1b | 1.0000 | 1.0000 | 0.00 | 0.0000 | negligible |
| strongreject_313 | llama3.2-3b | 1.0000 | 1.0000 | 0.00 | 0.0000 | negligible |
| strongreject_313 | qwen2.5-1.5b | 1.0000 | 1.0000 | 0.00 | 0.0000 | negligible |
| xstest_450 | llama3.2-1b | 0.7080 | 0.7183 | +1.03 | +0.0229 | negligible |
| xstest_450 | llama3.2-3b | 0.7337 | 0.7337 | 0.00 | 0.0000 | negligible |
| xstest_450 | qwen2.5-1.5b | 0.6154 | 0.6079 | −0.74 | −0.0153 | negligible |

*Largest safe-rate Δ in the full design = +2.14pp on the XSTest safe-prompt slice (llama3.2-1b), toward MORE compliance on safe-but-superficially-alarming prompts (over-refusal relief, not degradation). (SS5.1)*

**Observations.** Cohen's h is the matched-pairs binary effect size used across the safety line (paired analog of d, applied because outcomes are binary safe/unsafe). Every cell is negligible by a wide margin; the maximum |h|=0.0742 is an arcsine-transform artifact at the 99%+ ceiling (harmbench/qwen), where tiny rate changes inflate h. The largest actual safe-rate movement is 1.03pp. Note the sign of the headline XSTest movement: +2.14pp toward *compliance* on safe-but-alarming prompts is over-refusal relief, not a safety regression.

> Reporting effect size alongside p-values is what separates a credible null from "we ran a test and it wasn't significant." |h| ≤ 0.0742 across all cells means even if the protocol had infinite power, the largest effect it could possibly be hiding is negligible by Cohen's own bands. This is the quantitative content of "no detectable FP8 effect under tested conditions."

### F.6 — Per-battery cross-judge Cohen's κ (gemma3:12b × llama3.1:8b)

*Operationally binding LLM judge pair. Corpus-wide κ = 0.8306 (near_perfect), obs agreement 95.51%, n = 7,557. StrongREJECT κ = −0.0005 is Cohen's κ degenerating at zero variance — read via PABAK 0.9979. (SS11 / SS17 / B.3 / C.1)*

| Scope | κ | n | obs agree (po) | PABAK | Band |
|-------|---|---|----------------|-------|------|
| Corpus-wide | 0.8306 | 7,557 | 0.9551 | 0.9103 | near_perfect |
| harmbench_400 | 0.8096 | 2,389 | — | 0.9255 | near_perfect |
| jbb_100 | 1.0000 | 598 | — | 1.0000 | near_perfect |
| strongreject_313 | −0.0005 | 1,877 | — | 0.9979 | poor (zero-variance artifact) |
| xstest_450 | 0.7959 | 2,693 | — | 0.8158 | substantial |

**Standardized-vs-mixed contrast (SS11 / Conclusion 2):**

| Corpus | gemma3×llama3.1 κ | JTP band |
|--------|--------------------|----------|
| TR149 standardized batteries | 0.8306 | robust (≥0.70) |
| TR145 mixed task set (via TR148 v2) | 0.6917 | triangulate (0.40–0.70) |

*Regex anchor agrees with neither LLM judge above the slight band (κ 0.1729 / 0.1804 corpus-wide); expected for a rule-based classifier, does not bear on the verdict. (SS11 / B.3)*

**Observations.** The corpus-wide κ=0.8306 lands `robust` — the same gemma3×llama3.1 pair that only reached `triangulate` (0.6917) on TR145's mixed task set. The standardized-vs-mixed contrast is the key cross-TR finding: judge trust is *corpus-specific*, and clean, single-construct batteries produce sharper agreement than a heterogeneous mixed set. The StrongREJECT κ=−0.0005 is the classic Cohen's-κ-at-zero-variance pathology: at 100% agreed-refusal there is no variance for κ to chance-correct, so PABAK (0.9979) is the honest read.

> This table is why the protocol reports JTP per-corpus, not once globally. The same two judges are `robust` on standardized batteries and `triangulate` on a mixed set — so a single global κ would mislabel one or the other. The StrongREJECT row is a teaching case: a near-perfect 99.79% PABAK appearing as κ=−0.0005 is a reminder to always pair κ with PABAK at the ceiling.

### F.7 — Cross-battery heterogeneity + leave-one-battery-out

*Cochran's Q across 4 batteries per model (df=3); I² = 0.0% on all 3 models (Q < df clamps I² to floor). Leave-one-out MH confirms verdict robust to battery choice — every dropped-battery CI overlaps the full-set CI. (SS13 / SS16)*

| Model | k strata | Cochran's Q | df | weighted-mean Δ | I² | Band |
|-------|----------|-------------|----|-----------------|----|------|
| llama3.2-1b | 4 | 0.0214 | 3 | +0.0052 | 0.0% | low |
| llama3.2-3b | 4 | 0.0071 | 3 | −0.0017 | 0.0% | low |
| qwen2.5-1.5b | 4 | 0.0317 | 3 | −0.0008 | 0.0% | low |

**Leave-one-battery-out MH (full-set OR = 0.8065 [0.3828, 1.6989]):**

| Dropped battery | strata | n | Pooled OR | 95% CI | ΔOR vs full | CI overlaps? |
|-----------------|--------|---|-----------|--------|-------------|--------------|
| harmbench_400 | 9 | 2,410 | 0.8824 | [0.3305, 2.3555] | +0.0759 | Yes |
| jbb_100 | 9 | 3,237 | 0.8065 | [0.3828, 1.6989] | 0.0000 | Yes |
| strongreject_313 | 9 | 2,600 | 0.8065 | [0.3828, 1.6989] | 0.0000 | Yes |
| xstest_450 | 9 | 2,364 | 0.7333 | [0.2440, 2.2037] | −0.0732 | Yes |

*Dropping the two ceiling batteries (jbb / strongreject) leaves the pooled OR bit-identical — they contribute zero discordant pairs. Discriminating evidence is HarmBench + XSTest jointly. (SS16)*

**Observations.** Cochran's Q is near-zero on all three models (0.0071–0.0317 against df=3), clamping I² to its 0.0% floor — the batteries do not disagree with each other beyond chance. The leave-one-out check is the sensitivity analysis: dropping JailbreakBench or StrongREJECT leaves the pooled OR *bit-identical* at 0.8065 (they contribute zero discordant pairs), while dropping HarmBench or XSTest moves it only ±0.076. Every dropped-battery CI overlaps the full-set CI.

> The bit-identical OR after dropping two batteries is a clean demonstration of where the signal lives: all discriminating power is in HarmBench + XSTest, and even there it is too small to move the verdict. A reviewer worried that the null is an artifact of including ceiling-saturated batteries gets a direct answer — remove them and nothing changes.

### F.8 — Per-cell minimum detectable effect (MDE)

*Smallest detectable safe-rate Δ at α=0.05, 80% power (kappa-style paired-binary proxy, conservative). ~14pp on HarmBench/StrongREJECT/XSTest cells (n=311–403); ~28pp on JailbreakBench cells (n=100). 3–14pp single-cell blind spot; verdict rests on TOST (F.4), not McNemar non-rejection. (SS10 / B.1)*

| Battery | Model | n paired | MDE (pp, approx) |
|---------|-------|----------|------------------|
| harmbench_400 | llama3.2-1b | 357 | 14.83 |
| harmbench_400 | llama3.2-3b | 379 | 14.39 |
| harmbench_400 | qwen2.5-1.5b | 391 | 14.17 |
| jbb_100 | llama3.2-1b | 100 | 28.02 |
| jbb_100 | llama3.2-3b | 100 | 28.02 |
| jbb_100 | qwen2.5-1.5b | 100 | 28.02 |
| strongreject_313 | llama3.2-1b | 313 | 15.84 |
| strongreject_313 | llama3.2-3b | 313 | 15.84 |
| strongreject_313 | qwen2.5-1.5b | 311 | 15.89 |
| xstest_450 | llama3.2-1b | 387 | 14.24 |
| xstest_450 | llama3.2-3b | 383 | 14.32 |
| xstest_450 | qwen2.5-1.5b | 403 | 13.96 |

**Observations.** This table is the honesty check on the null: per-cell McNemar can only detect effects ≥ ~14pp (n≈311–403 cells) or ≥ ~28pp (the n=100 JailbreakBench cells). That leaves a single-cell blind spot below ~14pp — which is precisely why the verdict does not rest on McNemar non-rejection. The TOST equivalence in Table F.4 closes that gap from the other side: it positively bounds the effect *below* ±3pp, well inside the MDE blind spot.

> McNemar and TOST are complementary, not redundant. McNemar says "no effect ≥ 14pp"; TOST says "effect is provably < 3pp." Reporting MDE openly — rather than burying it — is what makes the equivalence claim defensible: the protocol names its own blind spot and then shows a second test that covers it.

### F.9 — Ceiling structure (where the discriminating power lives)

*JailbreakBench-100 + StrongREJECT at 100% refusal under both dtypes = 6/12 zero-power cells (zero discordant pairs). XSTest unsafe slice also at the 100% refusal ceiling; XSTest signal lives entirely in the safe slice, where FP16 compliance is only 24–45% (model-intrinsic over-refusal, present equally under both dtypes). (SS1 / SS5 / SS2)*

| Battery / slice | FP16 refusal/safe ceiling | Discordant cells | Discriminating power |
|------------------|---------------------------|------------------|----------------------|
| jbb_100 (3 cells) | 100% on all 3 models | 0 | none (zero-power) |
| strongreject_313 (3 cells) | 100% on all 3 models | 0 | none (zero-power) |
| xstest_450 unsafe slice | 100% on all 3 models | 0 | none (zero-power) |
| harmbench_400 (3 cells) | 0.8077–0.9923 spread | 4 each | yes |
| xstest_450 safe slice | 0.2402–0.4457 compliance | 4 / 0 / 11 | yes |

**XSTest safe-prompt slice FP16 compliance (SS5.1):**

| Model | n paired | FP16 compliance | FP8 compliance | Δ (pp) |
|-------|----------|------------------|-----------------|--------|
| llama3.2-1b | 187 | 0.3957 | 0.4171 | +2.14 |
| llama3.2-3b | 184 | 0.4457 | 0.4457 | 0.00 |
| qwen2.5-1.5b | 204 | 0.2402 | 0.2255 | −1.47 |

**Observations.** This table explains *why* so many cells are zero-power: JailbreakBench, StrongREJECT, and the XSTest unsafe slice are all pinned at 100% refusal under both dtypes — these models simply never comply with an overtly harmful request at 1B–3B scale, so there is no room for FP8 to move the rate. The discriminating power lives in HarmBench (refusal spread 0.81–0.99) and the XSTest *safe* slice, where compliance is only 24–45% — a model-intrinsic over-refusal that is present equally under both dtypes (the Δ is +2.14 / 0.00 / −1.47pp, noise around zero).

> The ceiling structure is the deepest point in TR149: the null is not "FP8 has no effect on safety" in the abstract — it is "FP8 has no effect on the *one axis where these models have headroom to vary*, the over-refusal of safe prompts." The harmful-refusal axis is saturated at 100% before FP8 ever enters. This is the finding that motivates Layer 5 (serving-state validity, TR152) to push specifically on the over-refusal slice where movement is possible.

### F.10 — Paired-OR estimator postmortem (buggy vs fixed)

*Buggy v1 fed paired McNemar cells into the UNPAIRED `(a·d)/(b·c)` formula, routing concordant mass into the numerator → pooled OR 3411.5 on an all-null corpus. Fix (commit `71f1a854`) uses the matched-pairs discordant ratio `(Σb+0.5)/(Σc+0.5)`. No verdict flipped — ORs are display-only; the verdict runs off TOST + McNemar. (SS21)*

| Quantity | Buggy unpaired `(a·d)/(b·c)` | Fixed matched-pairs `(Σb+0.5)/(Σc+0.5)` |
|----------|------------------------------|-------------------------------------------|
| Pooled OR | 3411.5 | 0.8065 |
| Pooled 95% CI | [1436, 8103] | [0.3828, 1.6989] |
| Per-cell OR range | up to 3966 | [0.1111, 2.3333] |
| jbb_100 cell OR (Δ=0.00) | 201 | 1.0000 |
| Matches McNemar pass? | No | Yes |
| Sampling/judging commit | `6d3359b4` | `6d3359b4` (unchanged) |
| Analysis commit | (pre-fix) | `71f1a854` |
| Verdict (0/12 Holm-sig, 12/12 TOST ±3pp) | identical | identical |

*TR145's own MH code builds genuine unpaired marginal tables, where `(a·d)/(b·c)` is correct — verified unaffected; patching it to "match TR149" would reintroduce a bug. (SS21.6)*

**Observations.** The postmortem is a cautionary tale worth its own table. The buggy v1 applied the *unpaired* cross-product OR `(a·d)/(b·c)` to matched-pairs cells, which routes the large concordant mass (records that agree under both dtypes) into the numerator and detonates the OR to 3411.5 — an absurd value on an all-null corpus, with a jbb_100 cell at OR=201 despite Δ=0.00. The fix uses the matched-pairs discordant ratio `(Σb+0.5)/(Σc+0.5)` and recovers OR=0.8065. Crucially, the data did not change (sampling/judging commit `6d3359b4` unchanged); only the analysis estimator was corrected at `71f1a854`, and no verdict flipped because the OR is display-only.

> The most important line is the last footnote: TR145's own code builds *genuine* unpaired marginal tables, where `(a·d)/(b·c)` is the *correct* estimator — so "fixing TR145 to match TR149" would have reintroduced the very bug being patched out of TR149. The paired-vs-unpaired distinction is design-dependent, not a universal preference. Conflating them is the single most dangerous statistical error in the entire null line, and naming both the symptom (OR=3411.5) and the non-fix (don't touch TR145) is the protocol's institutional memory of it.

---

## Appendix G — TR152 (FP8 KV-Cache Serving-State Safety Factorial, Layer 5 Anchor)

TR152 v2 is the serving-state validity layer of the null line: it pushes the FP8 KV-cache question across batch size, prefix caching, and temperature to ask whether any *serving-state* configuration interacts with the safety profile. FP16 model weights throughout, vLLM v0.19.1, RTX 4080 Laptop (sm_8.9), regex + gemma3:12b + llama3.1:8b judges under the `--skip-openai-judge` umbrella (`triangulate_no_openai` bucket, $0 external cost). This is the one TR in the null line that returns a *located* effect (H1): a Qwen-family, XSTest-only, temperature-amplified over-refusal lean — with the serving-state interaction (H2) rejected. Section citations `SSn.m` refer to `Technical_Report_152.md` (v2 canonical narration); structural numerics cross-checked against `research/tr152/results/20260526_232600/tr152_analysis.json`.

### G.1 — Scale / factorial coverage

*FP8-anchored star design: 14 cells/model planned (7 paired contexts × 2 KV dtypes), 12 realized (2 `sp-1` speculative-decoding cells failed identically — vLLM v0.19.1 argparse rejection, not OOM). (SS3 / SS4 / SS6)*

| Quantity | Value |
|----------|-------|
| Models (3 families: Llama 3.2, Phi-3, Qwen 2.5) | 5 |
| Safety batteries (HarmBench-400 / JBB-100 / StrongREJECT-313 / XSTest-450) | 4 |
| Serving-state contexts (baseline + 6 spokes) | 7 (6 runnable) |
| Cells/model planned vs realized | 14 → 12 (`sp-1` blocked) |
| Full-factorial cells (2×3×2×2×3) | 72 |
| Star coverage of full factorial | 19.44% |
| Sampled responses | 45,000 |
| Matched FP16-vs-FP8 pairs | 20,754 |
| Strata (battery × model × context) | 120 |
| Judge-label rows on disk (regex + gemma3 + llama3.1) | 135,000 |
| `sp-1` cells failed (1 per KV dtype × 5 models) | 10 (2 distinct cell types) |
| Total end-to-end wall | ~28.7 h |

**Observations.** TR152 is the largest serving-state factorial in the line: 45,000 sampled responses, 20,754 matched pairs, 135,000 judge rows, ~28.7 h wall, all at $0 external cost under the umbrella gate. It is a *star* design — 19.44% coverage of the full 72-cell factorial — sampling the baseline plus six single-axis spokes rather than the dense interior, which is the right economy when the goal is to detect any axis that *moves* the safety profile rather than to fit a full response surface. The two failed `sp-1` cells are honestly attributed to a vLLM v0.19.1 argparse rejection (`run.py:167-169`), not OOM — a correction of the v1 "cloud-gated OOM" misattribution (Table G.13).

> A star design buys axis-coverage cheaply: six spokes test six serving-state knobs against a common baseline without paying for the 72-cell interior. The honesty of the `sp-1` attribution matters for the defensibility bar — "argparse rejection at a named line" is a verifiable claim; "cloud-gated OOM" was a plausible-sounding guess that turned out wrong, and the correction is logged rather than buried.

### G.2 — Harmful-core invariance (3 adversarial batteries)

*Perfect FP8 concordance on clearly-harmful prompts: 0 discordant pairs across 90 (harmful battery × model × context) cells. FP16 safe rate at floor. (SS7 / SS8)*

| Battery | FP16-safe records | FP16 total | FP16 safe rate | Σb | Σc | Discordant |
|---------|-------------------|-----------|----------------|----|----|-----------|
| harmbench_400 | 2,995 | 2,995 | 1.0000 | 0 | 0 | 0 |
| jbb_100 | 2,996 | 2,996 | 1.0000 | 0 | 0 | 0 |
| strongreject_313 | 2,985 | 2,985 | 1.0000 | 0 | 0 | 0 |
| **Harmful pool** | **8,976** | **8,976** | **1.0000** | **0** | **0** | **0** (of 8,976 harmful matched pairs) |

**Observations.** On clearly-harmful prompts the result is the cleanest null in the entire Phase 6 corpus: 8,976 records at a 1.0000 FP16 safe rate, and 0 discordant pairs across all 90 harmful (battery × model × context) cells. FP8 did not flip a single harmful-prompt outcome under any serving-state configuration. The FP16 refusal rate is pinned at the ceiling, so — exactly as in TR149 — the harmful axis has no headroom for FP8 to move.

> Zero discordance across 8,976 harmful matched pairs is the result that licenses the deployment-relevant claim: FP8 KV-cache does not make these models comply with overtly harmful requests, across batch, prefix-caching, and temperature variation. Every bit of the located effect in this TR lives on the over-refusal battery, never on the harmful core. That separation is the entire interpretive key to TR152.

### G.3 — XSTest discordance (over-refusal battery)

*All 133 discordant pairs live on XSTest; 27 of 30 XSTest cells carry discordance (3 llama3.2-3b cells are b=c=0). Net imbalance +41 toward FP8-degraded (over-refusal). (SS7 / SS8 / SS9)*

| Quantity | Value |
|----------|-------|
| XSTest FP16-safe records | 8,160 |
| XSTest FP16 total | 11,778 |
| XSTest FP16 safe rate | 0.6928 |
| Σb (FP16-safe → FP8-unsafe; degraded) | 87 |
| Σc (FP16-unsafe → FP8-safe; improved) | 46 |
| Net b − c | +41 |
| Total discordant pairs | 133 |
| XSTest cells with discordance | 27 / 30 |

**Observations.** Every one of the 133 discordant pairs in the entire factorial is on XSTest — the over-refusal battery — and none on the harmful core. XSTest is the only battery with headroom (FP16 safe rate 0.6928, not 1.0), so it is the only place FP8 *can* register. The net imbalance is +41 toward "degraded" (87 b-pairs vs 46 c-pairs), where "degraded" on XSTest means the model over-refused a safe-but-superficially-alarming prompt under FP8 that it had answered under FP16.

> The semantic content of the "degradation" matters: on XSTest, FP16-safe→FP8-unsafe is the model becoming *more* cautious on benign prompts, i.e. more over-refusal — a usability cost, not a harmful-content leak. The located effect is a politeness/helpfulness regression on edge-case-benign inputs, the mildest possible failure mode, and it is confined to one battery and (Table G.5) largely one model family.

### G.4 — Cross-context Mantel–Haenszel pooled OR

*Haldane-corrected matched-pairs MH across all 120 strata (93 concordant cells drop out of the discordant pool). Sign-test independently confirms direction. (SS11 / SS1)*

| Quantity | Value |
|----------|-------|
| Strata pooled | 120 |
| Total matched pairs | 20,754 |
| Discordant pairs (Σb / Σc) | 133 (87 / 46) |
| Pooled OR (Haldane (Σb+0.5)/(Σc+0.5)) | 1.8817 |
| 95% CI (Wald on log-OR) | [1.3185, 2.6855] |
| log-OR / SE(log-OR) | 0.6322 / 0.1815 |
| Sign-test z (87 vs 46, p₀=0.5) | ≈ 3.55 |
| Sign-test two-sided p | ≈ 0.0004 |

**Observations.** The pooled OR is 1.8817 with a 95% CI [1.3185, 2.6855] that excludes 1.0 — this is the H1 "located effect" verdict. It uses the *correct* matched-pairs Haldane estimator `(Σb+0.5)/(Σc+0.5)` (the TR149 postmortem in Appendix F.10 is the cautionary precedent against the unpaired form). An independent sign-test on the 87-vs-46 discordant split gives z≈3.55, p≈0.0004, corroborating both the direction and the significance without relying on the OR's distributional assumptions.

> An OR of 1.88 that excludes 1.0 is a *real* effect — TR152 is the one place in the null line where H0 is rejected. But "located" is the operative word: the pooled OR is a weighted average that, the next table shows, mixes three opposite-direction family effects. The headline 1.88 is true and also misleading on its own; the protocol's discipline is to never report it without the decomposition.

### G.5 — Per-family Mantel–Haenszel decomposition

*Pooled OR 1.88 is a weighted mixture of three opposite-direction family effects. Qwen carries 99/133 discordant pairs (74%) and 79/87 b-pairs (91%). (SS23 / SS8)*

| Family | Σb | Σc | n strata | Pooled OR | 95% CI | Direction |
|--------|----|----|----------|-----------|--------|-----------|
| Qwen 2.5 (1.5b + 3b) | 79 | 20 | 48 | 3.878 | [2.386, 6.302] | FP8 degrades (CI well above 1.0) |
| Llama 3.2 (1b + 3b) | 7 | 15 | 48 | 0.484 | [0.202, 1.157] | FP8 marginally improves (CI brackets 1.0) |
| Phi-3 (phi3-mini-4k) | 1 | 11 | 24 | 0.130 | [0.024, 0.715] | FP8 statistically improves (CI excludes 1.0) |

*Per-model b − c (XSTest, summed over 6 contexts). Qwen carries the entire +41 net imbalance. (SS8 / SS9)*

| Model | b − c |
|-------|-------|
| qwen2.5-1.5b | +34 |
| qwen2.5-3b | +25 |
| llama3.2-1b | −6 |
| llama3.2-3b | −2 |
| phi3-mini-4k | −10 |

**Observations.** The decomposition dissolves the headline. Qwen 2.5 carries 99 of 133 discordant pairs (74%) and 79 of 87 b-pairs (91%), with a family OR of 3.878 [2.386, 6.302] — FP8 genuinely increases Qwen's over-refusal. But Llama 3.2 sits at OR 0.484 (CI brackets 1.0, marginal improvement) and Phi-3 at OR 0.130 [0.024, 0.715] — FP8 *statistically improves* Phi-3, the CI excluding 1.0 in the opposite direction. The per-model b−c confirms it: Qwen's +34 and +25 carry the entire +41 net imbalance; every other model is negative.

> This is the most important table in TR152. The pooled OR of 1.88 is not a property of "FP8 KV-cache"; it is a property of *Qwen 2.5 under FP8 KV-cache on XSTest*. Two of three families show the opposite sign, one of them significantly. Aggregating them produces a number that describes no actual model. The protocol's lesson — and the reason per-family decomposition is mandatory before any pooled serving-state claim — is that a significant pooled OR can be an artifact of one model family dragging a mixed cohort.

### G.6 — Holm–Bonferroni family correction

*120-cell McNemar family; 0 significant after Holm at family-wise α = 0.05. Smallest raw p far above the Bonferroni floor. (SS13)*

| Quantity | Value |
|----------|-------|
| Family size | 120 |
| Significant after Holm | 0 / 120 |
| Smallest raw p (xstest / qwen2.5-1.5b / temp=0.7, b=10 c=1) | 0.0117 |
| Holm-adjusted p of that cell (0.0117 × 120, capped) | 1.0000 |
| 2nd smallest raw p (xstest / qwen2.5-1.5b / temp=1.0, b=12 c=3) | 0.0352 |
| Bonferroni floor (0.05 / 120) | 0.000417 |

**Observations.** At the *per-cell* level, nothing survives correction: 0 of 120 cells are Holm-significant. The smallest raw p (0.0117, the Qwen-1.5b temp=0.7 cell) is more than 28× above the Bonferroni floor of 0.000417, and its Holm-adjusted p caps at 1.0000. The pooled effect (G.4) is significant precisely because pooling 133 discordant pairs across strata accumulates the signal that no single ~400-pair cell can carry alone.

> The contrast between G.4 (pooled OR significant) and G.6 (0/120 cells significant) is not a contradiction — it is the difference between a small, consistent, distributed effect and a large localized one. The Qwen over-refusal lean is real in aggregate but so diffuse that no individual serving-state cell rises above multiple-comparison correction. Operationally this means the effect is detectable only by pooling, and is far too small to flag any single deployment configuration.

### G.7 — TOST equivalence coverage (±3pp margin)

*117/120 cells positively equivalent at ±3pp; the 3 inconclusive cells are all qwen2.5-1.5b (not refuted — bootstrap CI extends just past −3pp). All 120 equivalent at ±5pp. (SS12)*

| Quantity | Value |
|----------|-------|
| Cells equivalent at ±3pp | 117 / 120 (97.5%) |
| Cells inconclusive at ±3pp | 3 / 120 (2.5%) |
| Cells equivalent at ±5pp | 120 / 120 (100%) |

*The 3 inconclusive cells (SS12 / SS24):*

| Cell | b | c | Δpp | Bootstrap 95% CI | Why not equivalent |
|------|---|---|-----|------------------|--------------------|
| xstest / qwen2.5-1.5b / temp=0.7 | 10 | 1 | −2.23 | [−3.82, −0.64] | CI lower bound exceeds −3pp |
| xstest / qwen2.5-1.5b / temp=1.0 | 12 | 3 | −2.21 | [−4.04, −0.38] | CI lower bound exceeds −3pp |
| xstest / qwen2.5-1.5b / batch_size=32 | 8 | 2 | −1.51 | [−3.05, +0.03] | CI lower bound exceeds −3pp by 0.05 |

**Observations.** Even with a located effect, 117 of 120 cells (97.5%) are positively equivalent at ±3pp, and all 120 at ±5pp. The 3 inconclusive cells are *all* qwen2.5-1.5b — the same family that carries the pooled effect — and they are "inconclusive," not "refuted": their bootstrap Δ-CIs extend just barely past −3pp (one by a hair's 0.05pp on batch_size=32). The point estimates (−1.51 to −2.23pp) are all inside the margin; only the CI tails poke out.

> "Inconclusive" is the honest verdict, distinct from both "equivalent" and "different." These 3 cells cannot positively claim equivalence at ±3pp, but neither do they reject it — the effect sits right at the boundary of the operationally-meaningful margin. That the inconclusive cells are exactly the highest-temperature Qwen cells (G.9) ties the loose threads together: the effect is real, Qwen-bound, and temperature-amplified, but small enough that it brushes the ±3pp line rather than clearing it.

### G.8 — Cohen's h per discordant cell (paired-binary effect size)

*Safety-line standard for matched-pair proportions. All 27 discordant cells in the negligible band; max |h| = 0.0458 (>4× below the 0.2 "small" threshold). (SS9)*

| Band (|h|) | Cells | Largest |h| in band |
|------------|-------|---------------------|
| negligible (< 0.2) | 27 / 27 | 0.0458 (xstest / qwen2.5-1.5b / temp=0.7) |
| small (0.2–0.5) | 0 | — |
| medium (0.5–0.8) | 0 | — |
| large (≥ 0.8) | 0 | — |

**Observations.** Effect size puts the located effect in perspective: all 27 discordant cells are in the negligible band, with a maximum |h|=0.0458 — more than 4× below Cohen's 0.2 "small" threshold. The largest-effect cell is, again, Qwen-1.5b at temp=0.7. So the H1 verdict is "a statistically real effect of negligible magnitude," which is exactly the kind of finding that pooling a large n can surface and a single underpowered cell cannot.

> Statistical significance (G.4) and practical magnitude (G.8) point opposite ways here, and that tension *is* the result. The pooled OR rejects H0, but |h| ≤ 0.0458 says the effect is, by the safety line's own standard, negligible. Reported together, they license the precise claim: FP8 induces a real but practically-negligible over-refusal lean on one model family's over-refusal battery — not "FP8 is unsafe" and not "FP8 has no effect."

### G.9 — FP8-interaction spread across serving contexts

*Per-context mean FP8 delta (pooled over 20 battery×model cells/context). Interaction spread = max single-cell Δ − min single-cell Δ across all 120 cells = 2.99pp (inside ±3pp band). Per-context-mean swing only 0.152pp. (SS10)*

| Context | mean Δ (pp) | min Δ (pp) | max Δ (pp) | n cells | n paired |
|---------|-------------|------------|------------|---------|----------|
| baseline | −0.012 | −1.02 | +0.76 | 20 | 3,466 |
| batch_size=8 | −0.075 | −1.02 | +0.26 | 20 | 3,469 |
| batch_size=32 | −0.126 | −1.51 | +0.26 | 20 | 3,468 |
| prefix_caching=True | −0.025 | −1.02 | +0.76 | 20 | 3,468 |
| temperature=0.7 | −0.164 | −2.23 | +0.26 | 20 | 3,432 |
| temperature=1.0 | −0.110 | −2.21 | +0.51 | 20 | 3,451 |
| **Interaction spread (whole factorial)** | | | | | **2.99 pp** |
| **Per-context-mean swing (temp=0.7 vs baseline)** | | | | | **0.152 pp** |

**Observations.** This is the H2 verdict: the serving-state interaction is *rejected*. The whole-factorial interaction spread — max single-cell Δ minus min single-cell Δ across all 120 cells — is 2.99pp, inside the ±3pp equivalence band. The per-context-mean swing between the most-affected context (temp=0.7, −0.164pp) and baseline (−0.012pp) is just 0.152pp. Temperature is the most active axis (the two temperature spokes carry the largest mean Δ and the widest min Δ), but even it does not push the interaction outside the margin.

> H2 asks whether *how you serve the model* — batch size, prefix caching, temperature — modulates the FP8 safety profile. The answer is no: the entire interaction spread (2.99pp) fits inside the ±3pp band that defines operational equivalence. Temperature nudges the effect (which is why the inconclusive cells in G.7 are the high-temp Qwen cells), but the serving-state axes do not create a configuration where FP8 becomes meaningfully less safe. Layer 5 validates: the null survives serving-state perturbation.

### G.10 — Cross-judge agreement (Cohen's κ)

*LLM–LLM pair robust by the JTP threshold (≥0.70); regex pooled κ is a Simpson-paradox artifact of degenerate harmful-battery marginals. v1 → v2 κ tightened with more data. (SS14 / SS5)*

| Judge pair | κ (v2) | n paired | p_obs | p_exp | κ (v1) |
|------------|--------|----------|-------|-------|--------|
| gemma3:12b ↔ llama3.1:8b | 0.831 | 44,951 | 0.945 | 0.676 | 0.814 |
| regex ↔ gemma3:12b | 0.062 | 44,639 | 0.680 | 0.659 | 0.088 |
| regex ↔ llama3.1:8b | 0.096 | 44,674 | 0.704 | 0.672 | 0.101 |

**Observations.** The operationally-binding LLM–LLM pair (gemma3 ↔ llama3.1) lands κ=0.831, `robust` by the JTP ≥0.70 threshold and consistent with TR149's 0.8306 on the same standardized batteries — judge trust is corpus-stable across the two largest serving-state TRs. The regex anchor's near-zero pooled κ (0.062 / 0.096) is a Simpson's-paradox artifact: the harmful batteries are at 100% agreed-refusal (degenerate marginals where κ has no variance to work with), which drags the pooled regex κ to the floor even though within-XSTest agreement is reasonable. v1→v2 tightened the LLM pair (0.814→0.831) as the data grew.

> Because the JTP gate is `robust` here, the verdict can rest on the LLM judges without mandatory triangulation — but the report still runs all three judges and reports the regex artifact transparently rather than dropping the inconvenient anchor. The κ=0.831 is what makes the H1/H2 verdicts trustworthy: the labels feeding the OR and TOST are near-perfectly agreed between two independent judge families.

### G.11 — Leave-one-out sensitivity (pooled MH OR robustness)

*Signal is fully XSTest-bound, doubly Qwen-bound (dropping either Qwen variant collapses the positive), and serving-state-axis-robust. (SS22)*

| LOO variant (re-pool over remaining strata) | Σb | Σc | Pooled OR | 95% CI | Verdict |
|---------------------------------------------|----|----|-----------|--------|---------|
| Drop XSTest | 0 | 0 | — | — | Full null (100% of signal on XSTest) |
| Drop harmbench / jbb / strongreject (each) | 87 | 46 | 1.882 | [1.318, 2.686] | Unchanged (harmful contributes 0) |
| Drop qwen2.5-1.5b | 37 | 30 | 1.230 | [0.762, 1.983] | CI brackets 1.0 — positive collapses |
| Drop qwen2.5-3b | 58 | 42 | 1.376 | [0.927, 2.043] | CI brackets 1.0 — positive collapses |
| Drop phi3-mini-4k | 86 | 35 | 2.437 | [1.649, 3.601] | OR inflates (largest improver removed) |
| Drop llama3.2-1b | 82 | 35 | 2.324 | [1.568, 3.444] | OR inflates |
| Drop llama3.2-3b | 85 | 42 | 2.012 | [1.393, 2.906] | OR inflates slightly |
| Drop temperature=0.7 | 69 | 41 | 1.675 | [1.140, 2.460] | OR drops most; still clears 1.0 |
| Drop temperature=1.0 | 70 | 38 | 1.831 | [1.236, 2.712] | Clears 1.0 |
| Drop baseline | 75 | 35 | 2.127 | [1.427, 3.169] | Clears 1.0 |
| Drop batch_size=8 | 75 | 40 | 1.864 | [1.273, 2.731] | Essentially unchanged |
| Drop batch_size=32 | 71 | 40 | 1.765 | [1.201, 2.596] | Clears 1.0 |
| Drop prefix_caching=True | 75 | 36 | 2.068 | [1.393, 3.071] | Clears 1.0 |

**Observations.** The LOO grid pins the effect's location precisely. Dropping XSTest zeroes the signal entirely (0 discordant pairs remain — 100% of the effect is on the over-refusal battery). Dropping any harmful battery changes nothing (they contribute 0 discordant pairs). The effect is *doubly* Qwen-bound: dropping either qwen2.5-1.5b or qwen2.5-3b collapses the CI to bracket 1.0 — neither Qwen variant alone sustains the positive, it takes both. By contrast, dropping any serving-state axis (batch, prefix, temperature) leaves the OR clearing 1.0 — the effect is robust to the serving-state perturbations, consistent with the H2 rejection.

> The LOO table is the empirical proof of the verdict's scope statement: "XSTest-only, Qwen-bound, serving-state-robust." Removing the model family kills it; removing the battery kills it; removing any serving knob does not. This is the discipline that converts a significant pooled OR into a precisely-bounded claim — the protocol does not say "FP8 degrades safety," it says "FP8 induces a Qwen-specific over-refusal lean on XSTest that survives serving-state variation," and every clause of that sentence is a row in this table.

### G.12 — Power / minimum detectable effect (MDE)

*Per-cell tests powered to detect ≥ −3.25 to −3.75pp at family-size-120 Holm; pooled MH powered to OR ≈ 1.42. Neither null is power-starved. (SS21)*

| Scale | Powered to detect | Observed | Comment |
|-------|-------------------|----------|---------|
| Per-cell (Holm-clear at family 120, n_paired ≈ 400) | Δpp ≈ −3.25 (b=13:0) to −3.75 (b=20:5) | worst cell −2.23pp | ~1.0–1.5pp short of Holm-clear; no cell rejects |
| Per-cell (raw p < 0.05, n_paired ≈ 400) | Δpp ≈ −1.5pp (b=7:1) | qwen2.5-1.5b temp=0.7 raw p=0.0117 | design has raw-p power; no cell carries enough imbalance |
| Pooled MH (sign-test α=0.05 on 133 discordant) | OR ≈ 1.42 (b ≥ 78 / c ≤ 55) | OR 1.88 (87/46), z=3.55 | comfortably above floor |

**Observations.** The power table forecloses the "underpowered null" objection at both scales. Per-cell, the design can clear Holm correction at a −3.25 to −3.75pp effect, and the worst observed cell is only −2.23pp — so the 0/120 Holm result is "no cell carries enough imbalance," not "too little data to tell." At the pooled scale, the MH sign-test floor is OR≈1.42 and the observed 1.88 (z=3.55) clears it comfortably. Neither verdict is power-starved.

> Power analysis is what makes a null defensible and a located effect honest. The per-cell null is real (the design would have caught a ≥3.25pp cell and there isn't one), and the pooled positive is real (the design's detection floor is 1.42 and we observe 1.88). Stating the MDE openly at both scales is the same discipline as TR149's F.8 — the protocol names exactly what it could and could not have detected before interpreting what it found.

### G.13 — v1 → v2 resolution progression

*Same qualitative shape (Qwen-family, XSTest-only, temperature-amplified) refined — not flipped — at 5.8× the discordant base. Point estimate regresses toward 1.0; CI tightens 4×; lower bound moves away from 1.0. (SS1 / SS4 / SS11 / SS12 / SS17)*

| Aspect | v1 (pilot) | v2 (canonical) |
|--------|------------|----------------|
| Models / families | 3 / 2 | 5 / 3 |
| XSTest per-cell cap | 100 | 450 (uncapped) |
| Sampled records | 14,400 | 45,000 |
| Matched pairs | 7,010 | 20,754 |
| Discordant pairs | 23 (17 / 6) | 133 (87 / 46) |
| Strata | 72 | 120 |
| MH pooled OR | 2.69 | 1.88 |
| MH 95% CI | [1.09, 6.62] | [1.32, 2.69] |
| CI width | 5.53 | 1.37 |
| Sign-test p | ≈ 0.03 | ≈ 0.0004 |
| TOST equivalent (±3pp) | 60 / 72 (83.3%) | 117 / 120 (97.5%) |
| Smallest raw p | 0.25 | 0.0117 |
| Spec-decode attribution | "cloud-gated OOM" | argparse rejection (`run.py:167-169`) |

**Observations.** v2 *refines* v1 rather than overturning it — the qualitative shape (Qwen-family, XSTest-only, temperature-amplified) is identical, established at 5.8× the discordant base (133 vs 23 pairs). The point estimate regresses toward 1.0 (OR 2.69→1.88, the expected shrinkage as a pilot's inflated estimate is corrected by more data), while the CI tightens 4× (width 5.53→1.37) and its lower bound moves *further* from 1.0 (1.09→1.32) — more data, more confidence, smaller-but-more-certain effect. The sign-test p sharpens an order of magnitude (0.03→0.0004), TOST equivalence coverage rises (83.3%→97.5%), and the spec-decode failure attribution is corrected from the v1 "cloud-gated OOM" guess to the verified argparse rejection at `run.py:167-169`.

> The v1→v2 progression is a model of how a pilot should mature: same shape, tighter bounds, point estimate regressing toward truth, and a misattributed failure mode corrected against the actual artifact. The OR dropping from 2.69 to 1.88 while the lower CI bound *rises* from 1.09 to 1.32 is the signature of a genuine small effect emerging from pilot noise — not a result that flips under scrutiny, but one that sharpens. It is the empirical counterpart to the defensibility bar: claims get more precise, and the one place v1 guessed ("OOM") gets replaced by a line-numbered fact.

---

## Appendix X1 — Named-method definitions (CRI / JTP / TAIS / RTSI)

The Phase 6 line contributes four named behavioural-screening methods, each lifted from a parent TR and applied as a reusable gate in the five-layer certification protocol. This appendix consolidates their definitions — input, statistic, threshold bands, and where each lands in Phase 6 — in one place, so a reader can check any layer's gate without re-deriving it from the parent report. The thresholds are *inherited verbatim* across TRs (that is what makes a cross-TR comparison meaningful), so each method is defined once here and cited, not re-stated per appendix.

### X1.1 — The four methods at a glance

| Method | Full name | Parent TR | Layer | Input statistic | Verdict in Phase 6 |
|--------|-----------|-----------|-------|-----------------|----------------------|
| CRI | Compile Reproducibility Index | TR147 | Layer 3 (compile integrity) | max pairwise \|Cohen's *d*\| on compiled latency across a stack set | **catastrophic** (19.70–48.93 across Triton versions) |
| JTP | Judge Triangulation Protocol | TR140 → TR148 | Layer 1 (measurement validity) | cross-family Cohen's κ on the largest-n judge pair | **triangulate** on TR148 mixed set (κ=0.6917); **robust** on TR149/TR152 standardized batteries (κ=0.831–0.8306) |
| TAIS | Typical-Acceptance Invariance Screen | TR144 | Layer 2 (behavioural screen) | matched-pairs Cohen's *h* + ±3pp TOST across draft+target pairs | **null** (max observed \|h\|=0.024) |
| RTSI | Refusal Template Stability Index | TR142 | Layer 2 (behavioural screen) | four refusal-template drift features → composite index | screen defined; behavioural-stability gate |

**Observations.** The four methods divide cleanly by what they protect. CRI guards the *reproducibility* of the compiled artifact (does the same code+stack produce the same latency?); JTP guards the *validity of the labels* (do independent judges agree enough to trust the safety measurement?); TAIS and RTSI are the two *behavioural screens* that ask whether an inference-time technique (speculative decoding for TAIS, template drift for RTSI) perturbs the refusal behaviour. Note CRI is the only one keyed on Cohen's *d* (continuous latency); the three safety screens use *h* or κ because their outcomes are binary or categorical.

> The verdict column is the punchline of the whole protocol: three of four methods return a clean pass (TAIS null, JTP robust-or-triangulate, RTSI stable), and the *one* that screams — CRI at catastrophic — is on the compile axis, not the safety axis. The serving-state safety profile is reproducible; the compiled-latency profile is not. That asymmetry is the load-bearing finding the certification protocol exists to surface.

### X1.2 — CRI (Compile Reproducibility Index) — full definition

*Parent: TR147 v4.0, impl `research/tr147/v4/compute_cri.py`.*

| Field | Definition |
|-------|------------|
| Input | Compiled-mode median latency (tok/s or ms) for a fixed model+task across a *stack set* (GPU × Triton × PyTorch × CUDA × cache-impl × code-SHA) |
| Statistic | Max pairwise \|Cohen's *d*\| over all stack-point pairs |
| Bands | robust < 0.5 · sensitive < 2 · fragile ≥ 2 · catastrophic ≥ 10 |
| Validity rule | Requires ≥ 2 valid compiled stack points; < 2 → `invalid` (not a pass) |
| Phase 6 result | Cross-Triton-version \|d\| = 19.70–48.93 → **catastrophic** |

**Observations.** CRI is the only method whose "high" reading is the *finding* rather than a failure of the screen. A catastrophic CRI (|d| up to 48.93 across Triton 3.3.1/3.4.0/3.6.0 on the same code) means the compiled-latency benchmark is not reproducible across an ordinary library-version bump — the exact failure mode TR147 was built to quantify. The validity rule (≥2 valid compiled points) is what prevents a config that *crashes* on compile from being scored as "robust by absence of variation"; it is scored `invalid` instead (the A100 pinned-code probe in Appendix D.9).

> CRI inverts the usual screen polarity: for the safety screens, a low statistic is the good news; for CRI, a low *d* would mean "latency reproduces," and the actual high *d* is the alarm. Keeping CRI in the same protocol as the safety screens is deliberate — it is the reminder that "the model behaves safely" and "the benchmark that measured it reproduces" are two separate guarantees, and Phase 6 only earns the first.

### X1.3 — JTP (Judge Triangulation Protocol) — full definition

*Parent: TR140 v3.0 (thresholds) → TR148 (Phase 6 application).*

| Field | Definition |
|-------|------------|
| Input | Per-record categorical safety labels from ≥ 2 judge families |
| Statistic | Cohen's κ on the *dynamically-selected* largest-n cross-LLM judge pair (ties → lower κ) |
| Bands | robust ≥ 0.70 (single judge sufficient) · triangulate 0.40–0.70 (majority-vote required) · untrustable < 0.40 (relabel) |
| Hard rule | No mandatory-judge gate: the join must not require any specific judge per record (the TR148 v1 bug, Appendix E.8) |
| Phase 6 results | TR148 mixed set κ=0.6917 → triangulate; TR149 standardized κ=0.8306 → robust; TR152 standardized κ=0.831 → robust |

**Observations.** JTP's defining Phase 6 lesson is that the *same* judge pair (gemma3:12b × llama3.1:8b) lands in different bands on different corpora: `triangulate` on TR148's heterogeneous mixed task set, `robust` on the clean standardized batteries of TR149/TR152. Judge trust is therefore a property of the corpus, not the judges — which is why the protocol re-runs JTP per corpus rather than certifying a judge pair globally. The dynamic largest-n primary-pair selection and the no-mandatory-judge rule are both scar tissue from the TR148 v1 join-bug that collapsed the verdict to a 94-record gpt-4o subset.

> The robust-vs-triangulate split across corpora is the operational core of Layer 1. It says: you may trust a single judge on a clean single-construct battery, but you must triangulate on a mixed set — and you discover which regime you are in by measuring κ on *this* corpus, not by trusting a prior. The dual-axis finding (Appendix E) sits underneath this: the negative cross-axis κ is why averaging all five judges would destroy signal, and why the gate keys on the largest-n *within-axis* pair.

### X1.4 — TAIS (Typical-Acceptance Invariance Screen) — full definition

*Parent: TR144 / speculative_decoding_safety.*

| Field | Definition |
|-------|------------|
| Input | Matched safe/unsafe outcomes for the same prompts under rejection sampling vs typical acceptance, across matched draft+target pairs |
| Statistic | Matched-pairs Cohen's *h* per cell + ±3pp TOST equivalence |
| Null cutoff | \|h\| < 0.1 (negligible); equivalence at ±3pp |
| Phase 6 result | Max observed \|h\| = 0.024 across the E1–E5 expansion → **null** (behavioural equivalence established) |

**Observations.** TAIS is the speculative-decoding member of the inference-flag null thread. Its calibrated null cutoff (|h|<0.1) was derived from the maximum observed effect across the E1–E5 expansion (|h|=0.024), so the cutoff sits ~4× above the largest real signal — a deliberately conservative band. The screen establishes *positive* equivalence (TOST at ±3pp), not mere non-rejection, which is the same affirmative-equivalence discipline TR149 applies to FP8.

> TAIS and the FP8 line (TR144/145/149) are the same claim through two different inference flags: turning on speculative decoding, like switching to FP8 KV-cache, does not move the safety profile beyond a negligible margin. Bundling them as a thematic thread across layers — rather than one numbered layer — is what lets the synthesis say "no tested inference-time flag perturbs safety" as a single sentence backed by four TRs.

### X1.5 — RTSI (Refusal Template Stability Index) — full definition

*Parent: TR142.*

| Field | Definition |
|-------|------------|
| Input | Refusal-response text across a configuration sweep |
| Features | Four refusal-template drift features: dominant-prefix share, unique-prefix rate, prefix entropy, mean refusal-token length |
| Bands | < 0.10 low-risk · 0.10–0.40 moderate · ≥ 0.40 high-risk |
| Phase 6 role | Layer 2 behavioural-stability screen (parent TR142 calibration; LOOCV danger-routing recall 10/10 across 51 cells) |

**Observations.** RTSI is the template-drift complement to TAIS within Layer 2: where TAIS asks "does the safe/unsafe *outcome* change," RTSI asks "does the *form* of the refusal drift" — collapsing toward a single template (low entropy, high dominant-prefix share) or fragmenting (high unique-prefix rate). The four features are behavioural and judge-free, which is what lets RTSI run as a cheap pre-screen before the more expensive judge-triangulated outcome measurement.

> RTSI earns its place in the protocol because refusal-template collapse is a leading indicator that outcome-level screens can miss: a model can hold its safe-rate while its refusals homogenize into a brittle template that a single jailbreak phrasing defeats. Layer 2 pairs RTSI (form) with TAIS (outcome) so the screen catches both the obvious failure (outcomes flip) and the subtle one (outcomes hold but the refusal surface goes brittle).

---

## Appendix X2 — Cross-TR sample-budget and cost ledger

This appendix consolidates the measurement budget across the seven Phase 6 reports into one ledger, so the synthesis can state its total scale without the reader re-summing seven headers. The two columns that matter for the defensibility bar are **judge rows** (the labelled-measurement count, which is what a reviewer means by "how much data") and **external cost** — which is **$0 across every report**, because the entire line runs on local Ollama judges under the `--skip-openai-judge` umbrella gate. Three caveats are load-bearing and called out below the table: the TR148/TR145 overlap, the no-safety-judge reports, and the paired-vs-sampled distinction.

### X2.1 — Per-TR measurement budget

| TR | Primary records | Judge rows (local) | Judge cohort | External cost | Hardware |
|----|-----------------|--------------------|--------------|---------------|----------|
| TR144 | 64,855 paired (16,783 core + 48,072 expansion) | 11,448 | regex + gemma3:12b | $0 | RTX 4080 Laptop + RunPod (E1–E5) |
| TR145 | 24,054 | 13,724 | regex + gemma3:12b | $0 | RTX 4080 Laptop |
| TR146 | 5,100 forward passes | — (no safety judge) | mechanistic probes only | $0 | RTX 4080 Laptop |
| TR147 | 52,410 rows | — (no safety judge) | latency telemetry only | $0 | RTX 6000 Ada + A100-SXM |
| TR148 | 13,724 re-judged (TR145 subset) | regex 13,724 / gemma3 13,676 / llama3.1 12,817 / shieldgemma 12,024 / llama-guard3 12,024 / gpt-4o 94 | 5 local + 94-record gpt-4o calib. | $0 (gpt-4o calib. negligible) | RTX 4080 Laptop |
| TR149 | 7,578 | ~7,557 × 3 judges | regex + gemma3:12b + llama3.1:8b | $0 | RTX 4080 Laptop |
| TR152 | 45,000 sampled / 20,754 paired | 135,000 (45,000 × 3 judges) | regex + gemma3:12b + llama3.1:8b | $0 | RTX 4080 Laptop |

**Observations.** Three reports dominate the safety-judge budget: TR152 (135,000 rows), TR145/TR148 (the same ~13,724-record substrate, judged twice), and TR149 (~22,671 rows across three judges). TR146 and TR147 carry *no* safety-judge rows at all — TR146 is mechanistic forward-pass probing (5,100 passes, no behavioural labelling) and TR147 is latency telemetry (52,410 rows, no safety axis) — so they contribute primary measurements but zero judge labels. Every external-cost cell is $0: the umbrella gate (`--skip-openai-judge`) routes all adversarial-corpus judging to local Ollama models, and the only frontier-judge contact in the entire line is TR148's 94-record gpt-4o *calibration* anchor, which is below any cost floor worth tabulating.

> The $0-external-cost column is not an accounting footnote — it is a methodological constraint that shaped the whole line. Because adversarial corpora (HarmBench, JailbreakBench, StrongREJECT, advbench) cannot be sent through a frontier API without a Researcher Access Program umbrella, every safety judgment here is local-only. That constraint is what produces the `triangulate_no_openai` bucket throughout, and it is why JTP (Appendix X1.3) matters so much: with no frontier-judge ground truth available at scale, cross-family κ between *local* judges is the only available validity check.

### X2.2 — The three load-bearing caveats

| Caveat | What it means | Why it matters for the total |
|--------|---------------|------------------------------|
| TR148 ⊂ TR145 | TR148 *re-judges* the same 13,724-record TR145 safety subset with more judges | Do **not** double-count the 13,724 records as new primary data; TR148 adds *judge* rows, not *primary* rows |
| TR146 / TR147 carry no safety judge | TR146 = 5,100 mechanistic forward passes; TR147 = 52,410 latency rows | They add primary measurements but **0** to the safety-judge total |
| Sampled vs paired (TR152) | 45,000 sampled responses → 20,754 *matched* FP16-vs-FP8 pairs | The verdict-bearing unit is the **pair** (20,754), not the sampled response (45,000); 135,000 is the judge-row count (45,000 × 3) |

**Observations.** The cleanest way to state the line's scale without overcounting: TR144's 64,855 paired samples, TR145's 24,054 primary (of which TR148 re-judges 13,724), TR146's 5,100 passes, TR147's 52,410 rows, TR149's 7,578, and TR152's 45,000 sampled / 20,754 paired — with TR152's 135,000 judge rows the single largest labelled-measurement block. The TR148⊂TR145 overlap is the one trap a careless sum would hit; the synthesis avoids it by counting TR148 as a re-judge (additional judge rows on existing primary data), consistent with the canonical `BANTERHEARTS_MEASUREMENT_COUNT.md` accounting.

> A reviewer's first instinct on a "~900K measurements" claim is to check whether the same records are being counted twice across reports. This ledger answers that pre-emptively: the one genuine overlap (TR148 re-judging TR145) is named and netted, the two no-judge reports are flagged so their large primary counts are not mistaken for safety labels, and the sampled/paired/judge-row distinction for TR152 is made explicit. Stating the caveats *before* the total is the defensibility-bar move — the number is large and also honestly disaggregated.

---

## Appendix X3 — Cross-TR null-vs-located findings

This is the ledger that makes the Phase 6 thesis auditable in one table: of the seven reports, six return a *null* or a falsification (no detectable safety effect within the equivalence margin), and exactly one — TR152 — returns a *located* effect (H1), which is itself bounded to a single model family on a single battery with the serving-state interaction (H2) rejected. The one report whose headline is *not* a safety null (TR147) is a null on the wrong axis: it finds a catastrophic *compile-reproducibility* failure, which is the protocol's whole point — the safety profile is stable, the latency benchmark is not.

### X3.1 — The seven verdicts side by side

| TR | Axis tested | Headline statistic | Equivalence / band | Verdict |
|----|-------------|--------------------|--------------------|---------|
| TR144 | Speculative decoding (rejection vs typical) | pooled OR 1.000 [0.835, 1.198]; max \|h\|=0.024 | ±3pp TOST equivalent | **H0 null** (TAIS pass) |
| TR145 | FP8 KV-cache (single config) | MH OR 1.05 [0.90, 1.23] | ±3pp; Qwen-1.5B lone TOST edge −3.09pp | **H0 null** |
| TR146 | Mechanistic separability under quant | 4 probes, all fail to separate safe/dangerous | — (falsification) | **F3 forbidden** (no mechanistic signal) |
| TR147 | Compile-latency reproducibility | CRI \|d\| 19.70–48.93 across Triton | catastrophic (≥10) | **located — on the compile axis** |
| TR148 | Judge measurement validity | primary κ=0.6917 [0.6824, 0.7008] | triangulate (0.40–0.70) | **dual-axis** (1a refusal + 1b composite-harm) |
| TR149 | FP8 KV-cache (standardized batteries) | MH OR 0.8065 [0.3828, 1.6989] | 12/12 TOST ±3pp | **H0 null** |
| TR152 | FP8 KV-cache (serving-state factorial) | MH OR 1.8817 [1.3185, 2.6855] | 117/120 TOST ±3pp; spread 2.99pp | **H1 located / H2 rejected** |

**Observations.** Read the verdict column as a thesis statement. The three FP8 reports (TR145 single-config, TR149 standardized, TR152 serving-state) form a scaling staircase: null at one config, null across four standardized batteries, and *located-but-bounded* once the factorial is large enough (20,754 pairs) to surface a negligible-magnitude Qwen-on-XSTest lean. TR144 adds the speculative-decoding null to the inference-flag thread. TR146 falsifies the mechanistic-separability hypothesis (no probe distinguishes safe from dangerous configs — the F3 "forbidden" claim). TR148 is the measurement-validity layer underneath all of them. TR147 is the odd one out by design: its located effect is on compile-latency reproducibility, not safety.

> The whole protocol is in this one table. Six of seven safety-axis verdicts are null/equivalent/falsified, and the single located safety effect (TR152) is so tightly bounded — one family, one battery, negligible |h|, serving-state-robust — that it strengthens rather than threatens the headline: FP8 KV-cache does not perturb safety on the harmful core under any tested serving state, and the only place it registers at all is a usability-grade over-refusal lean on one model family. The lone *consequential* instability in Phase 6 is the compile axis (TR147), which is exactly why the certification protocol separates compile integrity (Layer 3) from safety validity at all.

### X3.2 — The FP8 scaling staircase (TR145 → TR149 → TR152)

*The three FP8 KV-cache reports re-pooled on the same matched-pairs MH convention, ordered by design size. (TR145 SS / TR149 SS4 / TR152 SS11)*

| Report | Design | Matched pairs | Pooled OR | 95% CI | Verdict | Discriminating slice |
|--------|--------|---------------|-----------|--------|---------|---------------------|
| TR145 | single config, mixed task set | (subset of 24,054) | 1.05 | [0.90, 1.23] | H0 null | turn-5 multiturn edge |
| TR149 | 3 models × 4 batteries × 2 dtypes | 3,537 | 0.8065 | [0.3828, 1.6989] | H0 null | HarmBench + XSTest-safe |
| TR152 | 5 models × 4 batteries × 6 serving contexts | 20,754 | 1.8817 | [1.3185, 2.6855] | H1 located | XSTest / Qwen only |

**Observations.** The staircase is the strongest single argument in the line. At small scale (TR145) and medium scale (TR149) the FP8 effect is indistinguishable from zero — CIs bracket 1.0, TOST equivalent at ±3pp. Only at TR152's 20,754-pair scale does an effect become *locatable*, and even then it is confined to XSTest (the over-refusal battery, 0 effect on the harmful core) and to Qwen (LOO drops either Qwen variant and the CI re-brackets 1.0). The pooled OR rising from ~1.0 to 1.88 is not "FP8 gets less safe as you test more" — it is "more data resolves a small, real, model-specific over-refusal lean that was below the noise floor at smaller scale."

> This is what a *responsible* positive result looks like next to two nulls. A weaker analysis would either miss the TR152 effect (underpowered) or over-claim it ("FP8 degrades safety, OR 1.88"). The staircase forces the honest reading: the effect is real, it required 20,754 pairs to see, it is negligible in magnitude (|h|≤0.046), it is one family on one battery, and it does not touch the harmful core. The progression from TR145 to TR152 is the empirical content of "no detectable FP8 effect under tested conditions" maturing into "one bounded, benign, family-specific effect at scale."

### X3.3 — What "located" does and does not license

| Claim about TR152's H1 result | Licensed? | Why |
|-------------------------------|-----------|-----|
| "FP8 KV-cache degrades safety" | **No** | Harmful core is 0/8,976 discordant; effect is over-refusal (usability), not harmful compliance |
| "FP8 increases over-refusal on XSTest for Qwen 2.5" | **Yes** | Family OR 3.878 [2.386, 6.302]; LOO-confirmed Qwen-bound |
| "Serving state (batch/prefix/temp) modulates FP8 safety" | **No** | Interaction spread 2.99pp inside ±3pp; H2 rejected |
| "The effect is practically significant" | **No** | All 27 discordant cells negligible (max \|h\|=0.046) |
| "Temperature amplifies the (small) effect" | **Yes, weakly** | Temp spokes carry largest mean Δ; inconclusive TOST cells are all high-temp Qwen |

**Observations.** The licensing table is the claim-ladder applied to the one located result. The two licensed claims are narrow and sourced (Qwen-on-XSTest over-refusal, weak temperature amplification); the three forbidden claims are exactly the over-generalizations a pooled OR of 1.88 invites if read without the decomposition. The harmful-core invariance (Appendix G.2) is what forecloses "FP8 degrades safety," and the 2.99pp interaction spread (Appendix G.9) is what forecloses the serving-state-modulation claim.

> A located effect is more dangerous to a synthesis than a null, because it tempts over-claiming. This table is the discipline that keeps TR152's H1 honest: it says exactly what the OR of 1.88 buys (a narrow, benign, family-specific over-refusal lean) and exactly what it does not (any harmful-compliance or serving-state claim). The defensibility bar lives here — every "Yes" cites a CI or a LOO row, every "No" cites the invariance or equivalence result that refutes it.

---

## Appendix X4 — Cross-TR judge-cohort inheritance

The seven reports do not use a single fixed judge cohort; the cohort *evolves* across the line, and each report inherits, extends, or retires judges from its predecessors. This appendix tracks that inheritance so a reader can see why a given report's JTP band is what it is — and why two reports with the "same" gemma3 × llama3.1 pair can land in different bands. The two structural facts that organize the table: (1) the safety-specialist judges (shieldgemma, llama-guard3) enter only at TR148 and measure the *composite-harm* axis, not the refusal axis; (2) the standardized-battery reports (TR149, TR152) inherit the three-judge refusal cohort and land `robust`, while the mixed-set report (TR148) lands `triangulate` on the same general-judge pair.

### X4.1 — Per-TR judge cohort and JTP status

| TR | Judge cohort | Primary cross-LLM pair | κ | JTP band | Note |
|----|--------------|------------------------|---|----------|------|
| TR144 | regex + gemma3:12b | regex × gemma3 | ~0.0 (surface vs semantic) | — | screen is *h*-based (TAIS), not κ-gated |
| TR145 | regex + gemma3:12b | regex × gemma3 | 0.4274 (TR145-reported) | triangulate | re-measured 0.3626 by TR148 (Δ −0.0648) |
| TR146 | none (mechanistic probes) | — | — | — | no behavioural judge |
| TR147 | none (latency telemetry) | — | — | — | no safety axis |
| TR148 | regex + gemma3 + llama3.1 + shieldgemma + llama-guard3 (+ gpt-4o calib.) | gemma3 × llama3.1 | 0.6917 | triangulate | dual-axis: 1a refusal + 1b composite-harm |
| TR149 | regex + gemma3:12b + llama3.1:8b | gemma3 × llama3.1 | 0.8306 | robust | standardized batteries |
| TR152 | regex + gemma3:12b + llama3.1:8b | gemma3 × llama3.1 | 0.831 | robust | standardized batteries |

**Observations.** The cohort grows then settles. TR144/TR145 run a two-judge regex+gemma3 cohort (TR144's regex×gemma3 κ≈0 is the documented surface-vs-semantic mismatch — regex matches refusal *strings*, gemma3 reads refusal *meaning* — which is why TR144's screen is *h*-based, not κ-gated). TR148 is the expansion point: it adds llama3.1 (a second general refusal judge) plus the two safety specialists and a gpt-4o calibration anchor, and it is here the dual-axis structure is discovered. TR149 and TR152 then *settle* on the three-judge refusal cohort (regex + gemma3 + llama3.1), retiring the specialists — because the specialists' composite-harm axis is orthogonal to the refusal-as-safe outcome those reports score. TR146 and TR147 carry no judge at all.

> The inheritance pattern is the operational history of Layer 1 maturing. The line starts with a thin two-judge cohort, discovers at TR148 that adding safety specialists reveals a *second orthogonal axis* (not a better refusal judge), and then deliberately settles on the three-judge refusal cohort for the standardized-battery reports — carrying the dual-axis insight forward as the 1a/1b split rather than re-running five judges every time. The cohort is not fixed because the *question* each report asks is not fixed: a refusal-as-safe battery needs refusal judges, and the composite-harm specialists belong to Layer 1b's screen, not the outcome measurement.

### X4.2 — Why the same pair lands in different bands

| Corpus type | Report | gemma3 × llama3.1 κ | Band | Mechanism |
|-------------|--------|---------------------|------|-----------|
| Mixed task set | TR148 (via TR145 subset) | 0.6917 | triangulate | heterogeneous constructs (refusal + bias + truthfulness) dilute agreement |
| Standardized batteries | TR149 | 0.8306 | robust | single-construct refusal batteries sharpen agreement |
| Standardized batteries | TR152 | 0.831 | robust | same cohort, same corpus type, replicates TR149 |

**Observations.** The same two judges move from `triangulate` (0.6917) to `robust` (0.831) purely as a function of corpus composition. TR148's mixed set folds together refusal, bias (bbq), and truthfulness (truthfulqa) tasks — three different constructs the judges agree on to different degrees — so the pooled κ is dragged down. TR149/TR152's standardized batteries are single-construct (refusal-as-safe), and the same judges agree near-perfectly (0.83). TR152's 0.831 is a near-exact replication of TR149's 0.8306, confirming the standardized-corpus κ is stable across two independent large runs.

> This is the single most important cross-TR judge fact: judge trust is corpus-specific, and the protocol measures it per corpus rather than certifying a cohort once. A naive reading would call the TR148 triangulate verdict a "worse judge pair"; the truth is the *same* pair on a *harder* (multi-construct) corpus. The replication of κ≈0.83 across TR149 and TR152 is what licenses resting those reports' verdicts on the LLM pair without mandatory five-judge triangulation — the cohort settled because the corpus type stabilized the agreement.

---

## Glossary (Phase 6-wide)

Terms used across the Phase 6 line, defined once here. Statistical terms give the *operational* definition used in these reports, not the textbook generality; method names point to their full definition in Appendix X1.

| Term | Definition (as used in Phase 6) |
|------|-----------------------------------|
| **AWQ** | Activation-aware Weight Quantization. 4-bit weight quantization scheme; one of the quantized formats in the upstream TR125/TR142 weight-precision sweep that the Phase 6 KV-cache work sits downstream of. |
| **Batch-invariance** | The property that a model's per-record output does not depend on the batch it was served in. The batch axis of TR152's serving-state factorial (batch_size ∈ {1, 8, 32}) tests whether FP8 KV-cache breaks it for the safety profile; it does not (Appendix G.9). |
| **Cohen's *d*** | Standardized mean difference for *continuous* outcomes. Reserved in this line for TR147 compiled-latency (CRI = max pairwise \|d\|). **Not** used for binary safety outcomes — see Cohen's *h*. |
| **Cohen's *h*** | Effect size for the difference between two *proportions* (arcsine-transformed). The matched-pairs binary effect size used across the safety line; bands negligible < 0.2 · small 0.2–0.5 · medium 0.5–0.8 · large ≥ 0.8. Max observed in Phase 6 = 0.0742 (TR149) / 0.0458 (TR152). |
| **CRI** | Compile Reproducibility Index (TR147). Max pairwise \|Cohen's *d*\| on compiled latency across a stack set; bands robust < 0.5 · sensitive < 2 · fragile ≥ 2 · catastrophic ≥ 10. Full definition Appendix X1.2. |
| **Discordant pair** | In a matched-pairs (McNemar) design, a record whose outcome *differs* between the two conditions (FP16 vs FP8). `b` = FP16-safe → FP8-unsafe (degrade); `c` = FP16-unsafe → FP8-safe (improve). Concordant pairs (same outcome both conditions) carry no signal and drop out of the OR. |
| **E4M3 / FP8** | 8-bit floating point, 4 exponent + 3 mantissa bits — the FP8 KV-cache dtype tested throughout TR145/TR149/TR152. Model *weights* stay FP16; only the KV-cache is FP8. |
| **Haldane correction** | The `+0.5` added to each cell of an odds-ratio to keep it finite when a cell is zero. Phase 6 pooled ORs use the Haldane-corrected matched-pairs discordant ratio `(Σb+0.5)/(Σc+0.5)`. |
| **Holm–Bonferroni** | Stepdown multiple-comparison correction; less conservative than plain Bonferroni, controls family-wise error. Applied to the per-cell McNemar families (TR149 12-cell, TR152 120-cell) and the TR148 15-pair κ family. |
| **H0 / H1 / H2** | The three hypotheses of the serving-state safety question. H0 = no effect within ±3pp (null); H1 = a located effect; H2 = a serving-state *interaction*. Phase 6 returns H0 for TR144/145/149 and (H1 located / H2 rejected) for TR152. |
| **JTP** | Judge Triangulation Protocol (TR140 → TR148). Cross-family Cohen's κ on the largest-n judge pair → routing band; robust ≥ 0.70 · triangulate 0.40–0.70 · untrustable < 0.40. Full definition Appendix X1.3. |
| **KV-cache** | Key/Value cache: the stored attention keys and values that let autoregressive decoding avoid recomputing past tokens. Quantizing it to FP8 is the memory-saving intervention whose safety impact the line measures. |
| **Krippendorff's α** | A chance-corrected inter-rater agreement coefficient; reported alongside κ as a robustness check (TR148 primary α = 0.6917, matching κ). |
| **Mantel–Haenszel (paired vs unpaired)** | Stratified pooled odds-ratio estimator. **Paired** (matched-pairs discordant ratio `(Σb+0.5)/(Σc+0.5)`) is correct for TR149/TR152's within-prompt FP16-vs-FP8 design. **Unpaired** (cross-product `(a·d)/(b·c)`) is correct for TR145's genuine marginal tables. Conflating them is the TR149 postmortem bug that exploded the OR to 3411.5 (Appendix F.10). |
| **McNemar test** | Exact paired-binary test on the discordant cells `b` and `c` of a 2×2 matched-pairs table. The primary per-cell safety test throughout the line. |
| **MDE** | Minimum Detectable Effect: the smallest effect a test is powered to detect at α=0.05, 80% power. Reported openly (TR149 ~14pp/cell; TR152 ~3.25–3.75pp Holm-clear) so the null names its own blind spot. |
| **PABAK** | Prevalence-Adjusted Bias-Adjusted Kappa. The honest agreement read when κ degenerates at the ceiling (e.g. TR149 StrongREJECT κ=−0.0005 but PABAK=0.9979 at 100% agreed-refusal). |
| **PagedAttention** | vLLM's paged KV-cache memory manager; the serving substrate under which all FP8 KV-cache measurements run (vLLM v0.19.1). |
| **Prefix caching** | Reuse of the KV-cache for a shared prompt prefix across requests. The prefix-caching axis of TR152's factorial (on/off) tests whether it interacts with the FP8 safety profile; it does not (Appendix G.9). |
| **RTSI** | Refusal Template Stability Index (TR142). Four refusal-template drift features → composite index; bands < 0.10 low · 0.10–0.40 moderate · ≥ 0.40 high. Full definition Appendix X1.5. |
| **Speculative decoding** | A draft model proposes tokens that a target model verifies, accelerating decoding. The two acceptance rules tested by TAIS: **rejection sampling** (exact distribution-matching accept/reject) and **typical acceptance** (a looser entropy-based accept rule). |
| **Star (factorial) design** | A design that samples a baseline plus single-axis spokes rather than the dense factorial interior. TR152 uses a star design: 19.44% coverage of the full 72-cell factorial, enough to test each serving-state axis against a common baseline. |
| **TAIS** | Typical-Acceptance Invariance Screen (TR144). Matched-pairs Cohen's *h* + ±3pp TOST across draft+target pairs; null cutoff \|h\| < 0.1. Full definition Appendix X1.4. |
| **TOST** | Two One-Sided Tests for *equivalence*. Establishes that an effect is *within* a margin (the line's margin is ±3pp), converting "we failed to find an effect" into "we positively established equivalence." The methodological backbone of the null line. |
| **triangulate_no_openai** | The verdict bucket produced when the umbrella gate (`--skip-openai-judge`) forces local-only judging on adversarial corpora — triangulation across local judges with no frontier-judge ground truth. |
| **Umbrella gate** | The `--skip-openai-judge` flag. Adversarial corpora (HarmBench/JailbreakBench/StrongREJECT/advbench) cannot be sent through a frontier API without a Researcher Access Program umbrella, so the gate routes all such judging to local Ollama models ($0 external cost). |
| **XSTest** | A 450-prompt battery (CC BY 4.0) of safe-but-superficially-alarming prompts plus a harmful slice. The *over-refusal* battery — the only one with FP16 headroom below 100% — and therefore the only battery on which FP8 registers any effect (TR149/TR152). |

**Observations.** The glossary is organized to enforce the line's two most-confused distinctions. First, Cohen's *d* vs *h*: *d* is continuous (TR147 latency only), *h* is the paired-binary proportion effect size (all safety reports) — using *d* on a binary safety outcome would be a category error the line never makes. Second, paired vs unpaired Mantel–Haenszel: the paired discordant ratio is correct for within-prompt FP16-vs-FP8 designs, the unpaired cross-product for genuine marginal tables, and the entry names the exact bug (OR 3411.5) that conflating them produces.

> A glossary in a synthesis is not decoration — it is where the cross-TR vocabulary is pinned down so seven reports speak one language. The two distinctions above (d/h, paired/unpaired MH) are the ones that, if blurred, would let a reader mis-read the entire null line; defining them here, with the failure mode named, is the same defensibility discipline that runs through every appendix. The methods (CRI/JTP/TAIS/RTSI) point back to Appendix X1 rather than re-defining, so there is exactly one canonical definition of each.

---

*End of Extended Appendices. Companion to `Conclusive_Phase6.md` (main synthesis) and `Conclusive_Phase6_Whitepaper.md` (condensed external-facing). Source-of-truth reports: `PublishReady/reports/Technical_Report_{144,145,146,147,148,149,152}.md`.*
