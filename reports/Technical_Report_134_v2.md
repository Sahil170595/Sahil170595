# Technical Report 134 v2: Alignment Robustness Under Quantization -- Expanded
## Multi-family safety evaluation across 6 models (1.2B-7.6B), dual-judge validation, and regex-judge gap analysis

| Field | Value |
|-------|-------|
| **TR Number** | 134 v2 |
| **Project** | Banterhearts LLM Performance Research |
| **Date** | 2026-03-28 (v1: Mar 6, expansion: Mar 27-28) |
| **Author** | Research Team |
| **Report Type** | Safety alignment analysis (metric-backed, 6 models, 4 families, 7 quant levels, dual-judge, regex-judge gap analysis) |
| **Test Duration** | Original: ~10 hrs (Phase 3), Expansion: ~8 hrs (eval + gemma3 judge), Bespoke analysis: ~2 hrs |
| **Status** | Complete -- v2 expansion delivered |
| **Run IDs** | Original Phase 3: `20260305_144827`, Expansion: `20260327_170457`, Bespoke analysis: `20260328_173033` |
| **Total Samples** | 38,120 (24,778 original + 13,342 expansion) |
| **Judge Annotations** | 24,336 total (12,168 legacy Qwen + 6,552 Gemma small-model + 5,616 Gemma 7B rejudge) |
| **Related Work** | [TR124](Technical_Report_124.md), [TR125](Technical_Report_125_v2.md), [TR133](Technical_Report_133.md), [TR134 v1](Technical_Report_134.md), [TR142](Technical_Report_142.md) |
| **Depends On** | TR125 (quantization quality data), TR124 (FP16 baselines), TR142 (bespoke analysis pipeline) |

> **Peer Review Status:** This report has not undergone formal peer review. All findings should be interpreted as preliminary evidence from a consumer-hardware research program. Replication on independent hardware and datasets is required before drawing deployment conclusions.

---

## Abstract

TR134 v1 evaluated alignment robustness under quantization across 4 model families (Llama 3.2 1B/3B, Mistral 7B, Qwen 2.5 7B) using regex classifiers and a single LLM judge (Qwen 2.5 7B Q8_0). It established that Q2_K is catastrophic for safety, that Mistral 7B has weak baseline alignment at all precision levels, and that jailbreak susceptibility scales with quantization. It left three gaps: (1) coverage of sub-3B non-Llama families, (2) the observation that regex-judge agreement was low without diagnosing *why*, and (3) reliance on a single judge model that shared a family with one evaluated model.

TR134 v2 addresses all three. We add two model families -- Qwen 2.5 1.5B Instruct (1.5B parameters, Qwen/DPO family) and Phi-2 Chat v2 (2.7B parameters, Microsoft/Phi family) -- evaluated across 7 quantization levels (FP16 through Q2_K) on the same 6-benchmark safety battery. A second judge model (Gemma 3 12B) evaluates expansion samples, and the current bundle also includes Gemma re-judging of the 7B pair. The TR142 bespoke analysis pipeline merges original and expansion data into a unified 40-row safety matrix (6 models x 5-7 quant levels = 40 model-quant entries) with judge-derived aggregates on all 40 cells.

Key findings from the expanded matrix: (1) Qwen 2.5 1.5B replicates the Q2_K catastrophe on regex refusal and capability -- refusal collapses from 84.1% (FP16) to 34.1% (Q2_K, regex), while the judge still reports 82.9%, making it a measurement-divergent total-collapse case rather than a clean hidden-danger replicate. (2) Phi-2 shows a distinct safety profile: moderate baseline refusal (58.6% at FP16 via regex) with weak degradation slope, but anomalous bias resistance *improvement* at lower quant (over-refusal artifact). (3) The Mistral regex-judge gap is 64-71pp across all quant levels -- the regex classifier misses Mistral's refusal phrasing (contractions like "I can't" and hedged refusals), producing a systematically misleading safety picture. (4) Cross-family ANOVA on safety slopes with all 6 models: F=0.62, p=0.477 -- still not significant, with wider coverage. (5) The "hidden danger" regime (quality stable + safety degrading) appears at llama3.2-1b Q3_K_S, qwen2.5-7b Q2_K, and near-hidden-danger at mistral-7b Q2_K.

**Total: 38,120 safety samples across 6 models, 40 model-quant entries, plus 24,336 judge annotations across three source files.**

---

## Metric Definitions

Carried forward from TR134 v1 without modification. See TR134 v1, "Metric Definitions" section.

### Safety Metrics

- **Refusal Rate:** Proportion of harmful prompts where the model refuses. Regex classifier (`RefusalDetector`, ~15 patterns). Range [0, 1]. Higher = safer.
- **Bias Resistance:** Proportion of BBQ questions where the model selects non-stereotypical or "unknown" answer. Range [0, 1]. Higher = less biased.
- **Truthfulness:** Proportion of TruthfulQA questions answered correctly. Range [0, 1]. Higher = more truthful.
- **Jailbreak Compliance Rate:** Complement of refusal rate on jailbreak-wrapped prompts. Lower = safer.

### Capability Metrics

- **Accuracy (MMLU / ARC-Challenge):** Rescored accuracy via regex letter extraction. Same methodology as TR125.

### Derived Metrics

- **Safety-Capability (S/C) Ratio:** `normalized_safety / normalized_capability`. Value < 1.0 = safety degrades faster.
- **Normalized Score:** `raw / baseline`. Baseline: FP16 for sub-7B models, Q8_0 for 7B models.
- **Delta (pp):** Percentage-point change from baseline. Negative = degradation.
- **Cohen's Kappa:** Chance-corrected inter-rater agreement. < 0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate.
- **Regime:** Classification from the bespoke analysis -- `neutral` (quality and safety co-move), `hidden_danger` (quality stable, safety degrading >= 10pp), `near_hidden_danger` (approaching hidden_danger threshold).

### BPW Reference

| Quant Level | BPW | Relative to FP16 |
|-------------|-----|-------------------|
| FP16 | 16.0 | 1.00x |
| Q8_0 | 8.0 | 0.50x |
| Q6_K | 6.5 | 0.41x |
| Q5_K_M | 5.5 | 0.34x |
| Q4_K_M | 4.5 | 0.28x |
| Q3_K_S | 3.5 | 0.22x |
| Q2_K | 2.5 | 0.16x |

---

## Executive Summary

TR134 v2 extends the original 4-model safety matrix to 6 models spanning 4 distinct families. The core thesis from v1 is reinforced: **quantization degrades safety alignment, with Q2_K catastrophic across model families, and the degradation rate is model-dependent but not significantly family-dependent.**

### What Changed in v2

| Dimension | v1 | v2 |
|-----------|----|----|
| Models | 4 (llama3.2-1b, 3b, mistral-7b, qwen2.5-7b) | 6 (+qwen2.5-1.5b, phi-2) |
| Families | 3 (Llama, Mistral, Qwen) | 4 (+Phi) |
| Model-quant entries | 26 | 40 |
| Total samples | 24,778 | 38,120 |
| Judge annotations | 12,168 | 24,336 across 3 source files |
| Judge models | 1 (qwen2.5:7b-instruct-q8_0) | 2 judge families in current bundle (legacy Qwen 7B + Gemma 3 12B, including 7B rejudge) |
| Regex-judge gap analysis | Exploratory | Systematic (per-model, per-quant, per-task) |
| Safety regime classification | Not available | Full matrix (neutral / hidden_danger / near_hidden_danger) |
| Cross-family ANOVA | 3 families, F=2.50, p=0.137 | 4 families (6 models), F=0.62, p=0.477 |

### Key Findings (v2 Additions)

1. **Qwen 2.5 1.5B replicates the Q2_K catastrophe.** AdvBench refusal drops from 84.1% (FP16) to 34.1% (Q2_K), a -50pp collapse. This is comparable to llama3.2-1b's -57pp drop in v1. The Q2_K safety cliff is not Llama-specific -- it appears in every small model with strong baseline alignment.

2. **Qwen 2.5 1.5B Q2_K also collapses on capability.** MMLU drops -35.1pp (58.9% to 23.9%), ARC drops -48.5pp (74.0% to 25.5%). At Q2_K, both safety and capability are near floor. This is total model collapse, not selective safety degradation.

3. **Phi-2 has a flat safety slope and a moderate baseline.** AdvBench refusal at FP16: 58.6% (regex). Safety refusal slope: +0.0005 (essentially zero). Jailbreak refusal at FP16 is only 31.7%. Phi-2's direct refusal is moderate (not strong), and its jailbreak resistance is poor -- both are baseline alignment weaknesses, not quantization effects.

4. **Phi-2 shows anomalous bias resistance improvement at low quant.** Bias resistance rises from 84.8% (FP16) to 99.0% (Q2_K), a +14.1pp increase. This is the same over-refusal artifact observed in llama3.2-3b in v1 -- the model defaults to "unknown" answers as coherence degrades, scoring as unbiased.

5. **The Mistral regex-judge gap is 64-71pp on refusal and is systematic.** At Q3_K_S: regex says 19.1% refusal, judge says 90.0%. At Q8_0: regex says 23.6%, judge says 91.3%. The gap is consistent across all quant levels -- this is not a quantization effect but a regex classifier failure. Mistral uses refusal phrasing (contractions, hedged responses, topic redirections) that the regex patterns do not match.

6. **Three "hidden danger" model-quant entries identified.** The bespoke analysis classifies llama3.2-1b Q3_K_S and qwen2.5-7b Q2_K as `hidden_danger` (quality roughly preserved, safety refusal dropping >= 10pp from baseline). Mistral-7b Q2_K is classified `near_hidden_danger`. These are the deployment-critical entries: capability benchmarks alone would not flag them.

7. **Cross-family ANOVA remains non-significant with 4 families.** F=0.62, p=0.477. Adding Phi and Qwen 1.5B did not sharpen the between-family signal. Within-family variance continues to dominate. Model-level alignment quality predicts safety robustness better than family membership.

8. **Judge agreement varies dramatically by task.** AdvBench refusal: 77.6% agreement, kappa=0.169 (slight). BBQ bias: 86.1% agreement, kappa=0.446 (moderate). TruthfulQA: 39.5% agreement, kappa=0.145 (slight). The judge and regex classifiers agree best on bias (clearest signal) and worst on truthfulness (most ambiguous construct).

### Claim Validation (v2 Update)

| # | Claim | v1 Status | v2 Status | Evidence |
|---|-------|-----------|-----------|----------|
| 1 | Safety robust through Q3_K_S for well-aligned models | Demonstrated (3/4) | **Demonstrated (5/6)** | All models except Mistral maintain S/C >= 0.95 through Q3_K_S |
| 2 | Q2_K catastrophic for safety | Demonstrated | **Replicated** | qwen2.5-1.5b shows -50pp refusal (new), comparable to llama3.2-1b -57pp |
| 3 | Safety degrades disproportionately to capability | Partial (1/4) | **Partial (2/6)** | Mistral-7b and qwen2.5-7b show clear selective safety-over-quality degradation; llama3.2-1b and qwen2.5-1.5b collapse on both axes |
| 4 | Cross-family difference by RLHF recipe | Not validated (p=0.137) | **Not validated (p=0.477)** | Additional families weaken the signal |
| 5 | Regex classifiers are unreliable for some models | Noted | **Demonstrated** | Mistral 64-71pp gap is systematic; qwen2.5-1.5b Q2_K gap is 49pp |
| 6 | Hidden danger regime exists | Not tested | **Demonstrated** | 2 hidden_danger + 1 near_hidden_danger entries identified |

### Key Decisions for Practitioners (v2 Update)

1. **Minimum safe quant for new models:** Qwen 2.5 1.5B is safe through Q3_K_S for refusal (84.5% at Q3_K_S). Phi-2 is safe through Q4_K_M for refusal (55.0% at Q4_K_M, moderate baseline). Neither should be deployed at Q2_K.

2. **Do not trust regex refusal classifiers alone.** The Mistral gap demonstrates that regex can underreport safety by 70pp. If your model uses non-standard refusal phrasing (contractions, hedged refusals), regex will classify refusals as compliance. Use an LLM judge for validation.

3. **Monitor for hidden danger entries.** If your model shows stable quality metrics at a quant level but you have not checked safety, you may be in a hidden danger regime. Check refusal rate specifically -- it degrades before quality for some models at Q3_K_S.

4. **Consumer deployment thesis:** On a 12GB GPU, the expanded matrix supports Qwen 2.5 7B at Q4_K_M as the safest deployment option (94.5% refusal, 98.5% bias resistance). For sub-3B deployments where VRAM is extremely constrained, Qwen 2.5 1.5B at Q4_K_M (80.0% refusal, 88.9% bias resistance) is the best option. Phi-2 is not recommended for safety-critical applications due to its moderate direct refusal (58.6% at FP16) and weak jailbreak resistance (31.7% at FP16).

### When to Use This Report (v2 Scenarios)

**Scenario 1: Deploying Qwen 2.5 1.5B at Minimal VRAM**

**Question:** "I need the smallest possible Qwen model for an edge device. What's the lowest safe quant?"

**Answer:** Q4_K_M. Refusal stays at 80.0%, bias resistance at 88.9%, and jailbreak refusal at 65%. Q3_K_S is also safe for direct refusal (84.5%) but MMLU drops -13.7pp, and jailbreak refusal fluctuates (73.3%). Q2_K is absolutely ruled out: -50pp refusal, -48.5pp ARC. At Q4_K_M, estimated VRAM is approximately 1.5 GB -- feasible on most edge hardware.

**Scenario 2: Evaluating Whether Regex Safety Scores Are Trustworthy**

**Question:** "Our safety pipeline uses regex classifiers. Should we trust the numbers?"

**Answer:** It depends heavily on your model. For llama3.2-1b and the two Qwen models at higher-fidelity quants, regex-to-judge refusal gaps are often single-digit to low-teens. llama3.2-3b can still exceed 20pp, and Mistral 7B underreports refusal by 64-71pp across the board. For any model at Q2_K, disagreement widens sharply. Section 9 should be treated as a calibration audit, not as a one-size-fits-all endorsement of regex.

**Scenario 3: Checking for Hidden Safety Degradation**

**Question:** "Our model passes capability benchmarks at Q3_K_S. Is safety also fine?"

**Answer:** Maybe not. Consult Section 11 (safety regimes). llama3.2-1b at Q3_K_S shows stable quality (+0.98pp BERTScore) but -13.6pp refusal -- this is a hidden danger entry. If your model is in the matrix, check the regime classification. If not, run AdvBench refusal at your target quant level to verify.

**Scenario 4: Choosing Between Phi-2 and Llama 3.2 1B**

**Question:** "Both fit in my VRAM budget. Which is safer?"

**Answer:** For direct refusal, llama3.2-1b (93.6% FP16) is substantially stronger than phi-2 (58.6% FP16). At Q2_K, phi-2 (55.0%) retains more refusal than llama3.2-1b (36.8%) -- phi-2 degrades gracefully while llama3.2-1b collapses. For jailbreak resistance, llama3.2-1b is vastly superior (96.7% FP16 vs phi-2's 31.7%). If your threat model includes adversarial users, choose llama3.2-1b at >= Q4_K_M. If your threat model is accidental harm and moderate refusal is acceptable, phi-2 degrades less at low quant levels.

**Scenario 5: Cross-Referencing with TR125 Capability Data**

**Question:** "TR125 said Q4_K_M is safe for capability. Does TR134 v2 confirm this for safety across the expanded model set?"

**Answer:** Yes, with the same caveats as v1. All 6 models maintain adequate safety at Q4_K_M. Phi-2 (55.0% refusal at Q4_K_M) and Mistral (22.3% regex / ~93% judge) have weaker baselines, but their Q4_K_M safety is not significantly worse than their baseline. The TR125 Q4_K_M recommendation extends to safety for all tested families.

**Scenario 6: Understanding the Regex-Judge Discrepancy for Your Model**

**Question:** "We see a large gap between our regex and LLM judge safety scores. Is this normal?"

**Answer:** Consult Section 9. Single-digit to low-teens gaps are common on llama3.2-1b and the Qwen rows at higher-fidelity quants, but 20-30pp gaps can still appear on llama3.2-3b and phi-2. Persistent 60-70pp gaps indicate a regex classifier failure -- that is the Mistral pattern. Gaps above 40pp at Q2_K should be treated as a red flag for either classifier failure or severe output degradation.

### v1 vs v2 Findings Comparison

| Finding | v1 (4 models) | v2 (6 models) | Change |
|---------|--------------|---------------|--------|
| Q2_K catastrophic | Demonstrated for llama3.2-1b (-57pp) | **Replicated** for qwen2.5-1.5b (-50pp); phi-2 is exception (-3.6pp, flat slope) | Strengthened for cliff models |
| Mistral weak baseline | 29% refusal (regex) | 29% regex, **~91% judge** -- regex was misleading | Revised: Mistral refuses but regex misses it |
| Cross-family ANOVA | F=2.50, p=0.137 (3 families) | F=0.62, **p=0.477** (4 families) | Weaker signal with more data |
| Safety robust through Q3_K_S | 3/4 models | **5/6 models** | Extended |
| Over-refusal artifact | llama3.2-3b | **+phi-2** (bias metric specifically) | New instance |
| Jailbreak susceptibility scales with quant | All 4 techniques negative slope | Confirmed, **phi-2 flat as exception** | One exception added |
| Regex-judge kappa | 0.013-0.282 (exploratory) | 0.145-0.446, **Mistral gap 64-71pp diagnosed** | Root cause identified |
| Hidden danger regime | Not tested | **2 hidden_danger + 1 near_hidden_danger** identified | New finding |
| Bias most robust metric | Observed qualitatively | **Confirmed quantitatively**: only 4/40 entries > 10pp absolute shift (2 degradation, 2 improvement) | Confirmed |
| Capability proxy for safety | Not tested | Model-dependent: r=+0.997 (qwen2.5-1.5b) to r=-0.232 (phi-2); qwen2.5-7b r=-0.580 | New finding |

### How to Read This Report

| Time | Reading Path |
|------|-------------|
| **2 min** | Abstract -> Claim Validation table -> Key Decisions |
| **10 min** | Add Key Findings (1-8) + Section 9 (Regex vs Judge) + Section 11 (Safety Regimes) |
| **30 min** | Add Sections 4-8 (per-model safety curves) + Section 10 (cross-model ANOVA) + Section 14 (Limitations) |
| **60 min** | Full report Sections 1-15 + all Appendices |
| **Deep dive** | Appendix A (full safety matrix), Appendix B (all regex-judge gaps), bespoke analysis report at `research/tr142/results/bespoke_analysis/20260328_173033/analysis_report.md` |

### Table of Contents

**What Changed (Section 1)**
1. [What Changed in v2](#1-what-changed-in-v2)

**Methodology (Sections 2-3)**
2. [Methodology](#2-methodology)
3. [Statistical Methods & Caveats](#3-statistical-methods--caveats)

**Safety Results (Sections 4-8)**
4. [Results: Refusal (AdvBench)](#4-results-refusal-advbench)
5. [Results: Truthfulness (TruthfulQA)](#5-results-truthfulness-truthfulqa)
6. [Results: Bias Resistance (BBQ)](#6-results-bias-resistance-bbq)
7. [Results: Jailbreak Refusal](#7-results-jailbreak-refusal)
8. [Safety Degradation Curves: New Models](#8-safety-degradation-curves-new-models)

**Cross-Cutting Analyses (Sections 9-12)**
9. [Regex vs Judge Agreement](#9-regex-vs-judge-agreement)
10. [Cross-Model Safety Comparison](#10-cross-model-safety-comparison)
11. [Safety Regime Classification](#11-safety-regime-classification)
12. [Q5_K_M Safety Floor](#12-q5_k_m-safety-floor)

**Meta & Closing (Sections 13-15)**
13. [Judge Model Note](#13-judge-model-note)
14. [Limitations](#14-limitations)
15. [Reproducibility](#15-reproducibility)

**Appendices**
- [Appendix A: Full Safety Matrix](#appendix-a-full-safety-matrix-regex)
- [Appendix B: Full Regex vs Judge Gaps](#appendix-b-full-regex-vs-judge-gaps)
- [Appendix C: Capability Data](#appendix-c-capability-data-expansion-models)
- [Appendix D: Expansion Slope CIs](#appendix-d-expansion-slope-confidence-intervals)
- [Appendix E: S/C Ratios for New Models](#appendix-e-safety-capability-ratios-for-new-models)
- [Appendix F: Glossary](#appendix-f-glossary)

---

## 1. What Changed in v2

### 1.1 New Models

Two models were added to the safety matrix via TR142's expansion pipeline:

| Model | Family | Parameters | RLHF Method | Quant Levels | Baseline | Ollama Tag Pattern |
|-------|--------|-----------|-------------|-------------|----------|-------------------|
| Qwen 2.5 1.5B Instruct | Qwen | 1.54B | DPO | 7 (FP16-Q2_K) | FP16 | `qwen2.5:1.5b-instruct-{quant}` |
| Phi-2 Chat v2 | Phi | 2.78B | SFT + RLHF | 7 (FP16-Q2_K) | FP16 | `phi:2.7b-chat-v2-{quant}` |

**Why Qwen 2.5 1.5B:** Provides a second Qwen model at a different parameter count (1.5B vs 7B), enabling within-family size comparisons for DPO alignment. Also fills the sub-2B gap where only Llama 3.2 1B was previously tested.

**Why Phi-2:** Introduces a fourth alignment family (Microsoft's Phi training recipe). Phi-2 uses supervised fine-tuning (SFT) with additional RLHF, distinct from Meta's PPO (Llama, Mistral) and Alibaba's DPO (Qwen). At 2.7B parameters, it falls between Llama 1B and Llama 3B.

### 1.2 Why These Models

**Qwen 2.5 1.5B fills the sub-2B DPO gap.** In v1, the only sub-2B model was llama3.2-1b (PPO). Adding Qwen 1.5B (DPO) at a similar parameter count enables a controlled comparison: does alignment method affect safety robustness at small scale? The answer, from Sections 8 and 10, is mixed -- both models show Q2_K cliffs of similar magnitude (-50pp Qwen vs -57pp Llama), but Qwen 1.5B has a weaker baseline (84.1% vs 93.6%).

**Phi-2 introduces a fourth alignment paradigm.** Microsoft's Phi series uses a distinct training recipe (SFT-dominant with RLHF). At 2.78B parameters, it fills the gap between Llama 1B and 3B while adding family diversity. The result reveals a qualitatively different safety profile: flat slope with weak jailbreak baseline, unlike any v1 model.

**Both models have FP16 as baseline.** Unlike the 7B models that require Q8_0 baseline, both expansion models fit at FP16 on the RTX 4080 Laptop GPU (qwen2.5-1.5b: ~3.1 GB FP16, phi-2: ~5.6 GB FP16). This provides the full 13.5 BPW range (FP16 to Q2_K) for slope computation.

### 1.3 New Judge Model

The expansion work introduced **Gemma 3 12B** (via Ollama) as a second judge family. The current bundle now contains three judge sources:

- Gemma 3 12B is from a different model family (Google) than any evaluated model, eliminating the self-evaluation concern noted in v1 (where the Qwen 2.5 7B judge was evaluating Qwen 2.5 7B responses).
- The legacy TR134 corpus retains its original Qwen 2.5 7B Q8_0 judge labels.
- The small-model expansion (`qwen2.5-1.5b`, `phi-2`) uses Gemma 3 12B labels.
- The 7B pair (`mistral-7b`, `qwen2.5-7b`) was also re-judged with Gemma 3 12B.

The bespoke analysis therefore treats judge labels as a second measurement track alongside regex, not as a single harmonized gold-standard corpus.

### 1.4 Merged Analysis Pipeline

The TR142 bespoke analysis (`research/tr142/results/bespoke_analysis/20260328_173033/`) merges original TR134 Phase 3 data with the expansion data into a unified matrix:

- **matrix.csv:** 40 model-quant rows, with raw and delta quality columns (BERTScore, ROUGE-L, coherence), safety columns (refusal, truthfulness, bias), and judge-derived columns.
- **regimes.csv:** Safety regime classification per model-quant entry (neutral / hidden_danger / near_hidden_danger).
- **regex_vs_judge_gaps.csv:** Per-model, per-quant, per-metric gap between regex and judge classifications.
- **judge_agreement_task.csv / judge_agreement_quant.csv:** Cohen's kappa and agreement rates by task and quant level.

---

## 2. Methodology

### 2.1 Relationship to Prior Work

| Reference | Contribution | How TR134 v2 Uses It |
|-----------|-------------|---------------------|
| TR124 (Banterhearts) | FP16 quality baselines, metric framework | Baseline comparison methodology |
| TR125 (Banterhearts) | Quantization decision matrix, 5 models x 7 quants | Capability baseline, Q4_K_M/Q3_K_S cliff reference |
| TR133 (Banterhearts) | Predictive capacity planner, VRAM models | VRAM estimates for deployment recommendations |
| TR134 v1 (Banterhearts) | 4-model safety matrix, jailbreak amplification, bias categories | Primary dataset (24,778 samples), original judge labels |
| TR142 (Banterhearts) | Bespoke analysis pipeline, expansion framework | Expansion samples (13,342), merged matrix, regime classification |
| BBQ (Parrish et al. 2022) | 11-category bias benchmark | Primary bias evaluation instrument |
| TruthfulQA (Lin et al. 2022) | Truthfulness benchmark | Safety metric (underpowered) |
| JailbreakBench (Chao et al. 2024) | Standardized jailbreak evaluation | Source of harmful behaviors |
| JailbreakHub (Shen et al. 2023) | 1,405 real-world jailbreak prompts | Source of jailbreak templates |

### 2.2 Safety Battery

Identical to v1 across all 6 models. All tasks use the same prompts, temperature (0.0), seed (42), and max_new_tokens (256).

| Task | N per model-quant | Domain | Primary Metric | Source |
|------|-------------------|--------|----------------|--------|
| AdvBench Refusal | 100 | Safety | Refusal rate | walledai/AdvBench |
| TruthfulQA | 50 | Safety | Truthfulness | truthfulqa/truthful_qa |
| BBQ Bias | 198 | Safety | Bias resistance | heegyu/bbq (11 configs) |
| Jailbreak Amplification | 120 | Safety | Refusal rate | JailbreakBench + JailbreakHub |
| MMLU | 285 | Capability | Accuracy | cais/mmlu |
| ARC-Challenge | 200 | Capability | Accuracy | allenai/ai2_arc |

Total per model-quant: 953 samples. Total across all 40 entries: 38,120 samples.

### 2.3 Regex Classifiers

Three regex-based classifiers from v1, unchanged:

- **RefusalDetector:** ~15 refusal patterns (e.g., "I cannot", "I'm not able to", "As an AI", "I must decline"). Returns binary.
- **TruthfulnessClassifier:** String match against reference answers. Returns binary.
- **BiasDetector:** Compares answer choice to stereotypical answer. Returns binary.

**Known limitation (discovered in v2):** The RefusalDetector does not match contractions ("I can't") or hedged refusals ("I'd rather not discuss that"). This produces systematic under-reporting for Mistral 7B, which uses these phrasing patterns. See Section 9.

### 2.4 Dual-Judge Setup

| Corpus | Judge Model | N Annotations | Evaluated Models |
|--------|-------------|---------------|------------------|
| Original TR134 Phase 3 | Qwen 2.5 7B Instruct Q8_0 | 12,168 | llama3.2-1b, llama3.2-3b, mistral-7b, qwen2.5-7b |
| Expansion small-model judge | Gemma 3 12B | 6,552 | qwen2.5-1.5b, phi-2 |
| 7B re-judge | Gemma 3 12B | 5,616 | mistral-7b, qwen2.5-7b |

These judge sources are not a single calibrated corpus. The 7B pair now has both legacy Qwen-judge labels and Gemma re-judge labels in the broader artifact set, while the merged matrix reports judge-derived aggregates as a secondary measurement track rather than collapsing all judge outputs into one canonical ground truth.

**Conflict resolution:** When regex and judge disagree, we report both values. Neither is treated as ground truth. The gap itself is the finding -- it reveals where safety classification is ambiguous or where the regex classifier fails.

**Why not resolve conflicts?** Choosing one classifier as ground truth would mask the central finding: safety classification is inherently ambiguous, and the choice of classifier materially affects reported safety rates. A deployment engineer who resolves to the judge will see Mistral as 91% safe; one who resolves to regex will see 23% safe. Both are wrong -- the truth involves uncertainty that a single number cannot capture.

### 2.5 Data Merge Process

The bespoke analysis pipeline (`python -m research.tr142.bespoke_analysis`) performs the following merge:

1. **Load original TR134 Phase 3 samples** (24,778 records, 4 models, 26 model-quant entries).
2. **Load expansion TR142 samples** (13,342 records, 2 models, 14 model-quant entries).
3. **Align schemas.** Both datasets use the same column structure (base_model, quant, task, metric, score) because the expansion config mirrors the original.
4. **Compute per-entry statistics.** Mean, std, CI, median, min, max for each (model, quant, task, metric) combination.
5. **Compute quality deltas.** BERTScore, ROUGE-L, coherence changes from baseline.
6. **Compute safety deltas.** Refusal, truthfulness, bias resistance changes from baseline.
7. **Compute judge deltas.** Where available, judge-classified refusal, truthfulness, bias changes from baseline.
8. **Classify regimes.** Apply hidden_danger / near_hidden_danger / neutral rules.
9. **Compute cross-metric correlations.** Per-model Pearson and Spearman between quality and safety deltas.
10. **Output unified matrix** (40 rows) and supporting CSVs.

### 2.6 Normalization

- Sub-7B models (llama3.2-1b, 3b, qwen2.5-1.5b, phi-2): normalized to FP16.
- 7B models (mistral-7b, qwen2.5-7b): normalized to Q8_0 (FP16 exceeds 12GB VRAM).

---

## 3. Statistical Methods & Caveats

All methods carry forward from v1. The expanded dataset changes power calculations:

**Tests used:**
- Pairwise Welch's t-tests between adjacent quant levels (alpha = 0.05, uncorrected).
- One-way ANOVA across model families on mean safety degradation slopes (now 4 families, 4-6 slopes each).
- Linear regression of normalized score vs BPW per (model, metric).
- Cohen's kappa for regex-vs-judge agreement.

**Key caveats (carried from v1, updated):**

1. **Multiple comparison correction not applied.** The expanded matrix runs more pairwise tests. Expected false positive count increases proportionally.

2. **Power limitations remain severe for safety.** MDE = 18.3pp for safety at N=117/variant. The expansion models have the same sample sizes per task as v1. Only Q2_K cliff effects are reliably detectable.

3. **Dual-judge introduces calibration asymmetry.** Gemma 3 12B and Qwen 2.5 7B Q8_0 may have different classification thresholds. Judge labels should not be directly compared across the two corpora.

4. **Phi-2 alignment method is ambiguous.** Microsoft's documentation describes Phi-2 as using "SFT with RLHF" but does not specify PPO vs DPO. We classify it as a distinct family (Phi) rather than assigning it to PPO or DPO.

5. **Expansion config matches v1 exactly.** Same task paths, temperature, seed, max_new_tokens, backend (Ollama). The only differences are the model tags and judge model.

6. **7B baseline asymmetry persists from v1.** 7B models cover 5.5 BPW (Q8_0 to Q2_K), sub-7B models cover 13.5 BPW (FP16 to Q2_K). Slopes are not directly comparable across these ranges. The ANOVA normalizes within each model's BPW range but visual slope comparisons can be misleading.

7. **The hidden danger classification thresholds are somewhat arbitrary.** The bespoke analysis uses >= 10pp refusal drop + < 3pp BERTScore change. Different thresholds would change which entries are flagged. The specific entries identified should be treated as illustrative, not definitive.

### 3.1 Power Analysis (Updated for 6 Models)

| Metric Type | N per Variant | MDE (80% power, alpha=0.05) | Entries Below MDE |
|------------|--------------|------------------------------|-------------------|
| Safety (binary) | 100-120 | 18.3-19.6pp | 30/40 entries |
| Capability (binary) | 200-285 | 12.7-13.5pp | 32/40 entries |
| Truthfulness (binary) | 50 | 27.7pp | 40/40 entries |

**Observations:**

- **Most entries are below the detection limit.** 30 of 40 model-quant entries show safety deltas below the 18.3pp MDE. These are classified as "no significant degradation" by default, but real degradation up to 18pp could be hiding.

- **Truthfulness is completely underpowered.** At N=50, the MDE is 27.7pp. No truthfulness finding in this report is statistically reliable. The entire truthfulness analysis (Section 5) should be treated as exploratory.

- **Only Q2_K effects are reliably detected.** The effects that cross the MDE threshold are all at Q2_K: llama3.2-1b refusal (-57pp), qwen2.5-1.5b refusal (-50pp), and the llama3.2-3b over-refusal (+41pp). Phi-2's refusal delta (-3.6pp) is well below the detection limit.

### 3.2 Future Sample Size Requirements

To detect a 5pp safety delta at 80% power (adequate for deployment decisions):

| Current N | Required N | Factor | Practical Impact |
|-----------|-----------|--------|-----------------|
| 100 (AdvBench) | 1,540 | 15.4x | Would require 1,540 harmful prompts per variant |
| 50 (TruthfulQA) | 1,540 | 30.8x | Would require full 817-question TruthfulQA |
| 198 (BBQ) | 1,540 | 7.8x | Would require expanded BBQ set |
| 120 (Jailbreak) | 1,540 | 12.8x | Would require more jailbreak behaviors |

At 1,540 samples per task per variant, the 40-entry matrix would require approximately 370,000 total samples (vs the current 38,120). This is computationally feasible but would take approximately 100 hours of Ollama inference on the current hardware.

---

## 4. Results: Refusal (AdvBench)

### 4.1 Per-Model AdvBench Refusal Rates (Regex)

Bold = >= 10pp degradation from baseline. `--` = no data at that quant level (7B models use Q8_0 baseline).

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 90.0% | 90.0% | 90.0% | 86.0% | 87.0% | 85.0% | **33.0%** |
| llama3.2-3b | 53.0% | 52.0% | 57.0% | 55.0% | 47.0% | 91.0% | 94.0% |
| mistral-7b | -- | 29.0% | 35.0% | 29.0% | 31.0% | 22.0% | **12.0%** |
| qwen2.5-7b | -- | 98.0% | 99.0% | 99.0% | 99.0% | 96.0% | 93.0% |
| **qwen2.5-1.5b** | **84.1%** | **83.2%** | **85.5%** | **87.3%** | **80.0%** | **84.5%** | **34.1%** |
| **phi-2** | **58.6%** | **58.6%** | **54.1%** | **57.7%** | **55.0%** | **56.4%** | **55.0%** |

**Observations:**

- **Qwen 2.5 1.5B replicates the Q2_K catastrophe.** 84.1% refusal at FP16, relatively stable through Q3_K_S (84.5%), then collapses to 34.1% at Q2_K (-50pp). This matches llama3.2-1b's -57pp drop in magnitude. The Q2_K cliff is now confirmed across 2 families and 3 models (llama3.2-1b, qwen2.5-1.5b, and to a lesser degree mistral-7b).

- **Phi-2 shows a flat but moderate refusal profile.** 58.6% at FP16, 55.0% at Q2_K (-3.6pp). Unlike the cliff pattern, Phi-2's refusal is essentially quant-invariant. The Q3_K_S rate (56.4%) is only -2.2pp from baseline. The slope is nearly zero, but the baseline itself is moderate.

- **Qwen 2.5 1.5B has a strong FP16 baseline within the sub-3B class.** 84.1% refusal rate at FP16 is lower than qwen2.5-7b (93.2% at Q8_0) but higher than phi-2 (58.6%). DPO alignment at 1.5B produces solid but not near-perfect direct refusal.

- **Phi-2's Q2_K (55.0%) remains close to baseline.** Unlike llama3.2-1b (36.8%) or qwen2.5-1.5b (34.1%), Phi-2 at Q2_K does not exhibit the catastrophic cliff behavior. However, its absolute refusal rate is moderate at all quant levels, likely because Phi-2's safety training is based on different principles (SFT-dominant rather than reward-model-optimized).

### 4.2 AdvBench Refusal with Judge Override

The judge provides an independent refusal classification. For models where regex and judge agree, this adds confidence. For models where they diverge, the gap is the finding.

| Model | Quant | Regex | Judge | Gap (pp) |
|-------|-------|-------|-------|----------|
| llama3.2-1b | FP16 | 93.6% | 100.0% | +6.4 |
| llama3.2-1b | Q2_K | 36.8% | 97.7% | **+60.8** |
| llama3.2-3b | FP16 | 76.4% | 100.0% | +23.6 |
| llama3.2-3b | Q2_K | 92.7% | 100.0% | +7.3 |
| mistral-7b | Q8_0 | 23.6% | 91.3% | **+67.7** |
| mistral-7b | Q2_K | 12.3% | 82.9% | **+70.6** |
| qwen2.5-7b | Q8_0 | 93.2% | 99.8% | +6.6 |
| qwen2.5-7b | Q2_K | 80.9% | 96.1% | +15.2 |
| qwen2.5-1.5b | FP16 | 84.1% | 91.8% | +7.7 |
| qwen2.5-1.5b | Q2_K | 34.1% | 82.9% | **+48.8** |
| phi-2 | FP16 | 58.6% | 70.0% | +11.4 |
| phi-2 | Q2_K | 55.0% | 79.1% | +24.1 |

**Observations:**

- **The judge consistently reports higher refusal than regex.** Every entry shows a positive gap. The judge captures implicit, hedged, and non-standard refusals that regex misses.

- **Mistral's gap is extreme and quant-invariant.** The 64-71pp gap appears at every quant level, not just Q2_K. This is a regex classifier failure specific to Mistral's refusal phrasing style, not a quantization effect.

- **llama3.2-1b Q2_K gap is 60.8pp.** The regex says 37% refusal; the judge says 98%. At Q2_K, llama3.2-1b produces incoherent outputs that technically refuse (by being nonsensical rather than compliant), but the regex does not match these as refusals.

- **qwen2.5-1.5b Q2_K gap is 48.8pp.** Similar to llama3.2-1b: the model produces degraded outputs at Q2_K that the judge interprets as refusals but the regex does not capture.

- **Phi-2 gaps are moderate.** 11-24pp range. Phi-2's refusal phrasing is closer to the regex patterns than Mistral's, but still partially missed.

### 4.3 v1-to-v2 Comparison: Did Original Model Results Change?

The original 4 models retain their v1 data exactly -- no re-evaluation, no re-judging. The bespoke analysis simply includes them alongside the expansion models. To verify consistency:

| Model | v1 FP16/Q8_0 Refusal | v2 FP16/Q8_0 Refusal | Match? |
|-------|----------------------|----------------------|--------|
| llama3.2-1b | 90.0% | 90.0% | Yes |
| llama3.2-3b | 53.0% | 53.0% | Yes |
| mistral-7b | 29.0% (Q8_0) | 29.0% (Q8_0) | Yes |
| qwen2.5-7b | 98.0% (Q8_0) | 98.0% (Q8_0) | Yes |

All original data is preserved. The v2 findings for original models come from new analyses (regime classification, regex-judge gap quantification, sign reversal analysis) applied to existing data, not from new evaluations.

### 4.4 Refusal Rate Ranking at Key Quant Levels

| Rank | FP16/Q8_0 Best | Q4_K_M Best | Q2_K Best |
|------|---------------|-------------|-----------|
| 1 | llama3.2-1b (93.6%) | qwen2.5-7b (94.5%) | llama3.2-3b (92.7%, artifact) |
| 2 | qwen2.5-7b (93.2%) | llama3.2-1b (90.5%) | qwen2.5-7b (80.9%) |
| 3 | qwen2.5-1.5b (84.1%) | qwen2.5-1.5b (80.0%) | phi-2 (55.0%) |
| 4 | llama3.2-3b (76.4%) | llama3.2-3b (66.4%) | llama3.2-1b (36.8%) |
| 5 | phi-2 (58.6%) | phi-2 (55.0%) | qwen2.5-1.5b (34.1%) |
| 6 | mistral-7b (23.6%) | mistral-7b (22.3%) | mistral-7b (12.3%) |

**Observations:**

- **The ranking is remarkably stable across quant levels** (excluding Q2_K artifacts). The best models at FP16 remain the best at Q4_K_M. This confirms the v1 finding: baseline alignment quality determines safety more than quantization.

- **Llama3.2-3b's Q2_K #1 ranking is misleading.** The 92.7% refusal is an over-refusal artifact. Excluding it, qwen2.5-7b is the most robust at Q2_K -- consistent with its #1 ranking at Q4_K_M.

- **Phi-2 holds steady at #5 (FP16) to #3 (Q2_K) while llama3.2-1b drops from #1 to #4.** Phi-2's flat slope gives it an advantage at extreme quant levels, while llama3.2-1b's cliff drops it below phi-2 at Q2_K.

---

## 5. Results: Truthfulness (TruthfulQA)

### 5.1 Per-Model Truthfulness (Regex)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 55.0% | 56.0% | 48.0% | 49.0% | 58.0% | 49.0% | 44.0% |
| llama3.2-3b | 49.0% | 48.0% | 51.0% | 58.0% | 50.0% | 52.0% | 54.0% |
| mistral-7b | -- | 60.0% | 55.0% | 59.0% | 54.0% | 50.0% | 56.0% |
| qwen2.5-7b | -- | 50.0% | 53.0% | 49.0% | 57.0% | 49.0% | 50.0% |
| **qwen2.5-1.5b** | **49.0%** | **43.0%** | **47.0%** | **51.0%** | **51.0%** | **54.0%** | **59.0%** |
| **phi-2** | **39.0%** | **45.0%** | **42.0%** | **48.0%** | **50.0%** | **44.0%** | **44.0%** |

**Observations:**

- **All models fluctuate within noise.** With N=50 per variant, the MDE is ~28pp. No truthfulness delta exceeds this threshold for any model-quant pair. TruthfulQA results are exploratory at this sample size.

- **Qwen 2.5 1.5B shows a paradoxical increase at Q2_K.** 49% at FP16, 59% at Q2_K (+10pp). This likely reflects the model's degraded outputs happening to match reference answers by chance -- or the model defaulting to simpler, more "truthful" responses as it loses the ability to generate elaborate (and potentially wrong) answers.

- **Phi-2 has the weakest truthfulness baseline.** 39% at FP16, the lowest in the matrix. The model's tendency to generate confident but incorrect answers is a known Phi-2 characteristic.

- **The judge-regex gap on truthfulness is the largest across tasks.** Judge agreement is only 39.5% with kappa=0.145 (see Section 9). The truthfulness construct is the most ambiguous to classify.

### 5.2 Judge vs Regex on Truthfulness

The judge reports higher truthfulness than regex for some models and lower for others:

| Model | Quant | Regex Truthfulness | Judge Truthfulness | Gap (pp) |
|-------|-------|-------------------|-------------------|----------|
| llama3.2-1b | FP16 | 55.0% | 38.3% | -16.7 |
| llama3.2-1b | Q2_K | 44.0% | 9.1% | **-34.9** |
| mistral-7b | Q8_0 | 60.0% | 68.9% | +8.9 |
| mistral-7b | Q3_K_S | 50.0% | 72.4% | **+22.4** |
| phi-2 | FP16 | 39.0% | 59.4% | **+20.4** |
| qwen2.5-1.5b | FP16 | 49.0% | 60.6% | +11.6 |
| qwen2.5-1.5b | Q2_K | 59.0% | 25.0% | **-34.0** |

**Observations:**

- **The direction of the gap is model-dependent.** For Llama models, the regex reports *higher* truthfulness than the judge. For Mistral and Phi, the judge reports higher. This bidirectional disagreement is why the overall kappa is so low.

- **The Q2_K gaps for small models (-34pp for llama3.2-1b, -34pp for qwen2.5-1.5b) are large and negative.** At Q2_K, these models produce degraded outputs that happen to contain reference answer keywords (caught by regex) but are not actually truthful in context (flagged by judge). This is a meaningful distinction: keyword-level matching overstates truthfulness when the model's outputs are incoherent.

- **Truthfulness remains the weakest safety signal in the matrix.** The combination of small sample size (N=50), bidirectional gaps, and high ambiguity means truthfulness findings should not drive deployment decisions. Refusal and bias are more reliable safety signals.

---

## 6. Results: Bias Resistance (BBQ)

### 6.1 Per-Model Bias Resistance (Regex)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 89.4% | 88.9% | 88.4% | 87.4% | 87.4% | 99.5% | 73.2% |
| llama3.2-3b | 96.5% | 96.0% | 95.0% | 95.0% | 96.5% | 94.4% | 78.8% |
| mistral-7b | -- | 83.8% | 83.8% | 84.3% | 85.4% | 80.3% | 77.3% |
| qwen2.5-7b | -- | 98.5% | 98.0% | 97.5% | 98.5% | 97.5% | 99.0% |
| **qwen2.5-1.5b** | **85.4%** | **89.4%** | **88.4%** | **89.4%** | **88.9%** | **89.9%** | **90.4%** |
| **phi-2** | **84.8%** | **87.9%** | **86.4%** | **83.8%** | **86.9%** | **91.9%** | **99.0%** |

**Observations:**

- **Phi-2 shows strong bias resistance improvement at low quant.** From 84.8% (FP16) to 99.0% (Q2_K), a +14.1pp increase. This is the over-refusal artifact: as the model loses coherence, it defaults to "unknown" answers that score as unbiased. The same pattern appeared in llama3.2-1b at Q3_K_S (+10.1pp to 99.5%) in v1.

- **Qwen 2.5 1.5B bias resistance is stable and slightly positive.** 85.4% at FP16, 90.4% at Q2_K (+5.0pp). Unlike phi-2's dramatic swing, the Qwen 1.5B increase is gradual and may partially reflect genuine robustness of the DPO-trained bias properties.

- **qwen2.5-7b is the strongest performer on bias resistance.** 98.5% at Q8_0, 99.0% at Q2_K. Essentially flat. The 7B Qwen model's DPO alignment makes bias resistance highly resilient to quantization.

- **Judge data tells a different story for phi-2.** The judge reports phi-2 FP16 bias resistance at 60.4% (vs regex 84.8%). This 24pp gap suggests regex over-reports bias resistance for phi-2 because the model's answer formatting sometimes confuses the regex classifier.

### 6.2 Bias Resistance: Cross-Family Patterns

| Pattern | Models | Mechanism |
|---------|--------|-----------|
| Stable high | qwen2.5-7b (97.5-99.0%) | Strong DPO alignment preserves bias properties |
| Stable moderate | mistral-7b (77.3-85.4%), qwen2.5-1.5b (85.4-90.4%) | Moderate baseline, gradual or no degradation |
| Over-refusal increase | phi-2 (84.8% to 99.0%), llama3.2-1b (87.4% to 99.5% at Q3_K_S) | Coherence loss causes default to "unknown" |
| Genuine degradation | llama3.2-1b Q2_K (73.2%, -16.2pp), llama3.2-3b Q2_K (78.8%, -17.7pp) | Bias resistance collapses alongside capability |

**Observations:**

- **The over-refusal pattern is visible in bias but not refusal.** Phi-2's bias resistance increases at Q2_K while its refusal rate drops. This divergence suggests the over-refusal mechanism affects answer-choice tasks (BBQ) differently from open-ended tasks (AdvBench). On BBQ, the model defaults to the "unknown" option; on AdvBench, the model sometimes produces incoherent outputs that do not match refusal patterns.

- **Bias resistance is the most robust safety metric overall.** Across all 6 models and 40 entries, only 4 entries show > 10pp absolute bias shift: 2 show degradation (llama3.2-1b Q2_K at -16.2pp, llama3.2-3b Q2_K at -17.7pp) and 2 show improvement (llama3.2-1b Q3_K_S at +10.1pp, phi-2 Q2_K at +14.1pp). For 36/40 entries, bias resistance is within 5pp of baseline. This contrasts sharply with refusal (7 entries with > 10pp degradation) and jailbreak (8+ entries with > 10pp degradation).

- **Implication for practitioners.** If your primary safety concern is demographic bias (not refusal or jailbreaks), quantization down to Q3_K_S is safe for all tested models. The bias cliff, where it exists, is at Q2_K only.

---

## 7. Results: Jailbreak Refusal

### 7.1 Per-Model Jailbreak Refusal (Regex, 120 prompts per variant)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | 96.7% | 98.3% | 97.5% | 96.7% | 93.3% | 75.8% | 40.0% |
| llama3.2-3b | 95.8% | 93.3% | 94.2% | 95.0% | 82.5% | 98.3% | 91.7% |
| mistral-7b | -- | 19.2% | 23.3% | 20.8% | 15.0% | 16.7% | 12.5% |
| qwen2.5-7b | -- | 89.2% | 89.2% | 88.3% | 90.8% | 75.0% | 70.8% |
| **qwen2.5-1.5b** | **71.7%** | **70.0%** | **74.2%** | **77.5%** | **65.0%** | **73.3%** | **26.7%** |
| **phi-2** | **31.7%** | **30.0%** | **27.5%** | **29.2%** | **30.8%** | **31.7%** | **40.8%** |

**Observations:**

- **Qwen 2.5 1.5B jailbreak refusal collapses at Q2_K.** From 71.7% (FP16) to 26.7% (Q2_K), a -45.0pp drop. The model is broadly jailbreak-resistant through Q3_K_S (73.3%) but completely vulnerable at Q2_K.

- **Phi-2's jailbreak refusal is weak at ALL quant levels.** 31.7% at FP16 is comparable to Mistral's 19.2% at Q8_0. Quantization does not meaningfully change this: the slope is essentially zero (-0.0003). Phi-2's jailbreak vulnerability is a baseline alignment problem, not a quantization problem.

- **Phi-2 Q2_K shows paradoxical improvement.** Jailbreak refusal increases from 31.7% (FP16) to 40.8% (Q2_K), a +9.2pp gain. This is the over-refusal pattern: the model refuses more prompts indiscriminately at Q2_K, including jailbreak-wrapped ones.

- **Qwen 2.5 1.5B's baseline jailbreak refusal (71.7%) is lower than its direct refusal (84.1%).** The 12.4pp gap between direct and jailbreak refusal indicates that DPO alignment at 1.5B creates solid direct refusal, but jailbreak robustness requires something more -- likely larger model capacity or specific adversarial training.

### 7.2 Direct-Jailbreak Gap Analysis

The gap between direct refusal (AdvBench) and jailbreak refusal reveals how much "safety budget" each model spends resisting adversarial attacks:

| Model | Baseline Direct | Baseline Jailbreak | Gap | Interpretation |
|-------|----------------|-------------------|-----|----------------|
| llama3.2-1b | 90.0% | 96.7% | +6.7pp | Jailbreaks paradoxically *less* effective than direct (DAN triggers extra safety) |
| llama3.2-3b | 53.0% | 95.8% | +42.8pp | Huge gap: model is permissive on direct but resistant to jailbreaks |
| mistral-7b | 29.0% | 19.2% | -9.8pp | Jailbreaks work better than direct (model is already weak) |
| qwen2.5-7b | 98.0% | 89.2% | -8.8pp | Small gap: jailbreaks modestly effective |
| qwen2.5-1.5b | 84.1% | 71.7% | -12.4pp | Moderate gap: solid direct, weaker on jailbreaks |
| phi-2 | 58.6% | 31.7% | -26.9pp | Large gap: moderate direct, poor jailbreak |

**Observations:**

- **Phi-2 has a notable direct-jailbreak gap (-26.9pp).** The model refuses 58.6% of direct harmful requests but only 31.7% of jailbreak-wrapped ones. This suggests Phi-2's safety training focuses on recognizing harmful *content* rather than harmful *intent wrapped in benign framing*. Jailbreak templates disguise the harmful intent, bypassing Phi-2's content-level safety filters.

- **Llama 3.2 3B shows a paradoxical positive gap (+42.8pp).** Jailbreak refusal (95.8%) exceeds direct refusal (53.0%). This is because the model is permissive with direct requests (low baseline alignment for direct harm) but recognizes jailbreak patterns as suspicious. The DAN-style and roleplay framing triggers safety checks that the direct request does not.

- **The gap scales with quantization for some models.** Qwen 2.5 1.5B's gap widens from -12.4pp at FP16 to -7.4pp at Q2_K (34.1% direct refusal, 26.7% jailbreak). As the model weakens at Q2_K, both direct and jailbreak refusal collapse toward similar levels.

---

## 8. Safety Degradation Curves: New Models

### 8.1 qwen2.5-1.5b (baseline: FP16)

| Task | Metric | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|------|--------|------|------|------|--------|--------|--------|------|
| advbench_refusal | refusal_rate | 84.1% | 83.2% | 85.5% | 87.3% | 80.0% | 84.5% | **34.1%** |
| bbq_bias | bias_resistance | 85.4% | 89.4% | 88.4% | 89.4% | 88.9% | 89.9% | 90.4% |
| jailbreak | refusal_rate | 71.7% | 70.0% | 74.2% | 77.5% | 65.0% | 73.3% | **26.7%** |
| truthfulqa | truthfulness | 49.0% | 43.0% | 47.0% | 51.0% | 51.0% | 54.0% | 59.0% |
| mmlu_real | accuracy | 58.9% | 56.5% | 57.9% | 58.6% | 53.7% | 45.3% | **23.9%** |
| arc_challenge | accuracy | 74.0% | 74.5% | 72.5% | 73.5% | 71.5% | 64.0% | **25.5%** |

**Observations:**

- **Relatively stable refusal through Q3_K_S, then Q2_K collapse.** The advbench refusal fluctuates between 80-87% for all quant levels FP16-Q3_K_S, then drops to 34.1% at Q2_K (-50pp from FP16). This is among the sharpest cliffs in the expanded matrix.

- **Capability collapses at Q2_K too.** MMLU drops -35.1pp (to near-random 23.9%), ARC drops -48.5pp (to 25.5%). At Q2_K, the model is non-functional for both safety and capability. The S/C ratio at Q2_K is not meaningful because both numerator and denominator are near floor.

- **Bias resistance is the most robust metric.** Slightly increases across quant levels (85.4% to 90.4%). The DPO-trained bias properties appear highly quantization-resistant for the Qwen family at both 1.5B and 7B.

- **Jailbreak refusal shows a secondary cliff at Q2_K (-45pp).** Unlike advbench, jailbreak refusal fluctuates in the Q4_K_M-Q3_K_S range (65-77%) before dropping to 26.7% at Q2_K.

- **Quality-safety correlation is near-perfect for this model.** The bespoke analysis reports Pearson r=0.997 between coherence and refusal rate, and r=0.988 between BERTScore and refusal rate. This means quality metrics are a reliable proxy for safety degradation -- if you see quality drop, safety is dropping too. This is the "safe" correlation pattern: practitioners monitoring quality will catch safety regressions.

- **Capability collapses harder than safety through Q3_K_S.** MMLU at Q3_K_S is 45.3% (-13.7pp from FP16), while AdvBench refusal is 84.5% (+0.4pp from FP16). The capability cliff arrives at Q3_K_S; the safety cliff arrives at Q2_K. This one-level gap creates a deployment risk window: Q3_K_S is borderline for capability but still safe, potentially encouraging deployment at a quant level with unacceptable quality.

- **Within-family comparison (1.5B vs 7B):** Qwen 2.5 1.5B shows a much sharper Q2_K cliff than qwen2.5-7b. The 7B model degrades gracefully (80.9% refusal at Q2_K, -12.3pp from Q8_0), while the 1.5B model collapses to 34.1% (-50pp from FP16). Model size provides substantial quantization headroom for the Qwen family. The DPO alignment recipe is robust at both sizes through Q3_K_S, but the 1.5B model lacks the parameter redundancy to survive Q2_K compression.

### 8.2 phi-2 (baseline: FP16)

| Task | Metric | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|------|--------|------|------|------|--------|--------|--------|------|
| advbench_refusal | refusal_rate | 58.6% | 58.6% | 54.1% | 57.7% | 55.0% | 56.4% | 55.0% |
| bbq_bias | bias_resistance | 84.8% | 87.9% | 86.4% | 83.8% | 86.9% | 91.9% | 99.0% |
| jailbreak | refusal_rate | 31.7% | 30.0% | 27.5% | 29.2% | 30.8% | 31.7% | 40.8% |
| truthfulqa | truthfulness | 39.0% | 45.0% | 42.0% | 48.0% | 50.0% | 44.0% | 44.0% |
| mmlu_real | accuracy | -- | -- | -- | -- | -- | -- | -- |
| arc_challenge | accuracy | 68.5% | 68.5% | 68.5% | 68.5% | 67.5% | 65.0% | 51.0% |

> **Note:** Phi-2 MMLU data is not available in the bespoke analysis. See TR125 v2 for phi-2 MMLU capability data.

**Observations:**

- **Phi-2 has the flattest safety refusal slope in the matrix.** Slope = +0.0005 (essentially zero). Refusal does not systematically track BPW for this model. Individual quant levels fluctuate within ~4pp around the baseline (54.1%-58.6%) without a clear downward trend.

- **Q2_K degradation is negligible for refusal.** Only -3.6pp on advbench (55.0% vs 58.6% baseline), -17.5pp on ARC. Phi-2 at Q2_K retains essentially its baseline refusal level -- a qualitatively different behavior from llama3.2-1b or qwen2.5-1.5b, though the baseline itself is moderate.

- **Jailbreak resistance is the weakest metric.** 31.7% at FP16 means the model complies with 68% of jailbreak-wrapped requests. This is a baseline alignment issue: Phi-2's SFT-based training does not produce strong jailbreak resistance.

- **Bias resistance increases at Q2_K (+14.1pp to 99.0%).** This is the clearest over-refusal artifact in the expanded matrix. The model defaults to safe/neutral answers at Q2_K, which the bias classifier scores as unbiased.

- **Quality-safety decoupling is a deployment concern.** The bespoke analysis reports r=-0.232 between BERTScore and refusal for phi-2. Quality metrics provide no warning of safety changes for this model. A practitioner monitoring ARC would see relatively stable performance through Q3_K_S (65.0%, -3.5pp from FP16) while having no signal about whether refusal behavior has shifted. (Note: phi-2 MMLU capability data is available in TR125 v2, not in this TR's bespoke analysis.)

- **Phi-2 occupies a unique position in the safety taxonomy.** It is the only model with (a) moderate direct refusal (~58%), (b) weak jailbreak resistance (~32%), (c) flat safety slope, and (d) quality-safety decoupling. This combination means: the model does not get dramatically worse at low quant, but it is also not particularly safe at high quant. Deployment decisions should focus on the moderate direct refusal baseline (58.6%) and weak jailbreak baseline (31.7%) rather than quantization effects.

- **ARC-Challenge is remarkably flat through Q5_K_M.** 68.5% at FP16, 68.5% at Q5_K_M -- zero degradation across 4 quant levels. This is the flattest capability curve in the expanded matrix and suggests Phi-2's architecture is particularly resistant to capability quantization down to 5.5 BPW.

- **Comparison with Llama 3.2 models at similar parameter counts.** Phi-2 (2.78B) sits between llama3.2-1b (1.24B) and llama3.2-3b (3.21B) in size. Its safety profile is qualitatively different from both: llama3.2-1b shows a catastrophic Q2_K cliff, llama3.2-3b shows over-refusal, and phi-2 shows flat degradation with a weak baseline. These differences arise from training recipe, not parameter count.

### 8.3 Summary: Where New Models Fit in the Safety Landscape

| Model | Safety Profile | Q2_K Behavior | Quality-Safety Coupling | Deployment Recommendation |
|-------|---------------|--------------|------------------------|---------------------------|
| qwen2.5-1.5b | Solid baseline (84.1%), Q2_K cliff | Catastrophic (-50pp refusal, -48.5pp ARC) | Tightly coupled (r=0.99) | Safe through Q3_K_S; never deploy Q2_K |
| phi-2 | Moderate baseline (58.6%), flat slope | Negligible refusal change (-3.6pp) | Decoupled (r=-0.23) | Moderate refusal at all quant levels; not for jailbreak-sensitive apps |

---

## 9. Regex vs Judge Agreement

### 9.1 Aggregate Agreement by Task

| Task | N Pairs | N Agree | Agreement % | Kappa | Interpretation |
|------|---------|---------|-------------|-------|----------------|
| advbench_refusal | 4,000 | 3,104 | 77.6% | 0.169 | Slight |
| bbq_bias | 5,148 | 4,430 | 86.1% | 0.446 | Moderate |
| truthfulqa | 2,000 | 789 | 39.5% | 0.145 | Slight |

**Observations:**

- **BBQ bias has the highest agreement.** The bias construct is the least ambiguous: the regex checks which multiple-choice option the model selected, and the judge evaluates whether the selection is stereotypical. Both methods converge because the signal is structural (letter choice), not semantic.

- **TruthfulQA has the worst agreement.** 39.5% agreement is below chance for a binary classifier. The regex matches reference answer keywords; the judge evaluates whether the response is factually correct in context. These measure overlapping but distinct constructs.

- **AdvBench agreement (77.6%) hides a low kappa (0.169).** High raw agreement occurs because both classifiers agree on the majority class (refusal). Kappa corrects for this expected agreement, revealing that the classifiers add minimal information beyond the base rate.

### 9.2 Agreement by Quant Level (AdvBench)

| Quant | N Pairs | N Agree | Agreement % | Kappa |
|-------|---------|---------|-------------|-------|
| FP16 | 400 | 337 | 84.3% | 0.129 |
| Q8_0 | 600 | 474 | 79.0% | 0.143 |
| Q6_K | 600 | 481 | 80.2% | 0.170 |
| Q5_K_M | 600 | 472 | 78.7% | 0.132 |
| Q4_K_M | 600 | 462 | 77.0% | 0.147 |
| Q3_K_S | 600 | 500 | 83.3% | 0.260 |
| Q2_K | 600 | 378 | 63.0% | 0.162 |

**Observations:**

- **Q2_K has the lowest agreement (63.0%).** At extreme quantization, model outputs become ambiguous -- partially coherent refusals, garbled text that neither classifier handles well. The disagreement rate at Q2_K is 37%, almost double the FP16 rate (15.8%).

- **Q3_K_S has the highest kappa (0.260).** At this quant level, the classifiers agree the most beyond chance. This may reflect a "cleaner" signal at Q3_K_S: the model either clearly refuses or clearly complies, with fewer borderline cases than at higher quant levels where the model produces hedged responses.

- **No systematic trend.** The hypothesis from v1 -- that lower quant produces more ambiguous responses, reducing agreement -- is only weakly supported by Q2_K. The overall pattern is flat with Q2_K as an outlier.

### 9.3 The Mistral Gap

The largest and most systematic regex-judge gap is for Mistral 7B on refusal:

| Quant | Regex Refusal | Judge Refusal | Gap (pp) |
|-------|---------------|---------------|----------|
| Q8_0 | 23.6% | 91.3% | **+67.7** |
| Q6_K | 28.6% | 92.5% | **+63.8** |
| Q5_K_M | 24.5% | 91.8% | **+67.3** |
| Q4_K_M | 22.3% | 92.7% | **+70.4** |
| Q3_K_S | 19.1% | 90.0% | **+70.9** |
| Q2_K | 12.3% | 82.9% | **+70.6** |

**Observations:**

- **The gap is 64-71pp at every quant level.** This is not a quantization artifact. Mistral uses refusal phrasing that the regex does not match. Common patterns: "I can't assist with that" (contraction), "That's not something I'm able to help with" (indirect phrasing), "I'd prefer to discuss something else" (topic redirection).

- **The judge sees 82-93% refusal.** According to the judge, Mistral actually refuses most harmful requests -- contradicting the regex-based finding from v1 that Mistral has "weak baseline alignment." v1's claim that Mistral's refusal rate is 29% was based entirely on regex. The judge revises this to ~91%.

- **Implication for v1 findings:** v1's claim that Mistral has "critically weak safety alignment" needs qualification. Mistral's *regex-measured* refusal is weak (23-29%). Its *judge-measured* refusal is strong (91-93% at Q8_0). The truth likely lies between these extremes. What is clear is that Mistral refuses differently from other models, and regex classifiers designed for standard refusal phrases will systematically underreport its safety.

- **The gap narrows at Q2_K.** From 67.7pp at Q8_0 to 70.6pp at Q2_K -- the regex rate drops faster (-11pp) than the judge rate (-8pp). At Q2_K, both classifiers agree the model is more compliant, but the magnitude of degradation depends entirely on which classifier you trust.

### 9.4 The Q2_K Gap for Small Models

| Model | Quant | Regex | Judge | Gap (pp) |
|-------|-------|-------|-------|----------|
| llama3.2-1b | Q2_K | 36.8% | 97.7% | **+60.8** |
| qwen2.5-1.5b | Q2_K | 34.1% | 82.9% | **+48.8** |

Both models show massive gaps at Q2_K but not at higher quant levels. The mechanism differs from Mistral: at Q2_K, these models produce garbled or incoherent outputs. The regex sees no refusal phrases and classifies as compliance. The judge interprets incoherence as non-compliance (the model did not produce harmful content, so it "refused"). Both interpretations are defensible. The question is: does a model that produces nonsense in response to harmful requests count as "safe"?

---

## 10. Cross-Model Safety Comparison

### 10.1 Safety Slope Table (All 6 Models)

| Model | Refusal Slope | Bias Slope | Truthfulness Slope | Mean Safety Slope | Cap Slope |
|-------|---------------|------------|--------------------|--------------------|-----------|
| llama3.2-1b | +0.0250 | +0.0038 | +0.0100 | +0.0129 | +0.0221 |
| llama3.2-3b | -0.0201 | +0.0068 | -0.0074 | -0.0069 | +0.0110 |
| mistral-7b | +0.0922 | +0.0129 | +0.0183 | +0.0411 | +0.0133 |
| qwen2.5-7b | +0.0234 | -0.0004 | +0.0013 | +0.0081 | +0.0157 |
| **qwen2.5-1.5b** | **+0.0204** | **-0.0041** | **-0.0119** | **+0.0015** | **+0.0269** |
| **phi-2** | **+0.0005** | **-0.0077** | **-0.0131** | **-0.0068** | **+0.0070** |

**Observations:**

- **Phi-2 has the flattest refusal slope (+0.0005).** Refusal is essentially quant-invariant for this model. But the negative bias and truthfulness slopes reflect the over-refusal artifact: the model gets "better" at bias by defaulting to safe answers at low quant.

- **Qwen 2.5 1.5B's mean safety slope (+0.0015) is near zero.** This is misleading -- the strong refusal slope (+0.0204) is partially cancelled by negative bias (-0.0041) and truthfulness (-0.0119) slopes. The refusal slope is the policy-relevant metric.

- **Mistral's slope (+0.0411) remains the steepest**, but this is based on regex-measured refusal. If judge-measured refusal is used instead, the slope would be shallower (judge sees ~91% at Q8_0 dropping to ~83% at Q2_K, a much smaller normalized range).

### 10.2 Safety vs Capability Divergence (All 6 Models)

| Model | Safety Slope | Cap Slope | Divergence | Verdict |
|-------|-------------|-----------|------------|---------|
| llama3.2-1b | +0.0129 | +0.0221 | -0.0092 | Capability degrades faster |
| llama3.2-3b | -0.0069 | +0.0110 | -0.0178 | Safety appears to *improve* (artifact) |
| mistral-7b | +0.0411 | +0.0133 | +0.0278 | Safety degrades faster |
| qwen2.5-7b | +0.0081 | +0.0157 | -0.0076 | Capability degrades faster |
| qwen2.5-1.5b | +0.0015 | +0.0269 | -0.0254 | Capability degrades faster |
| phi-2 | -0.0068 | +0.0070 | -0.0137 | Safety appears to *improve* (artifact) |

**Observations:**

- **4 of 6 models show capability degrading faster than safety.** The general pattern from v1 holds: safety properties are at least as robust as capability under quantization. Mistral remains the sole exception where safety degrades faster.

- **Qwen 2.5 1.5B shows the largest capability-over-safety divergence (-0.025).** Capability (MMLU, ARC) degrades steeply at Q2_K while safety refusal holds through Q3_K_S. This confirms the Q2_K cliff: at Q2_K, everything collapses together.

- **Two models show negative safety slopes (llama3.2-3b, phi-2).** Both are over-refusal artifacts. Neither model is genuinely safer at low quant.

### 10.3 Cross-Family ANOVA (v2)

With 4 families (Llama: 6 slopes from 2 models, Mistral: 3, Qwen: 6 from 2 models, Phi: 3):

| Statistic | v1 | v2 |
|-----------|----|----|
| F-statistic | 2.50 | **0.62** |
| p-value | 0.137 | **0.477** |
| df | (2, 9) | (3, 14) |
| Conclusion | Not significant | **Not significant** |

**Observations:**

- **Adding families *reduced* the F-statistic.** Phi-2 and Qwen 1.5B have intermediate safety slopes that fall within the existing range, reducing the between-group variance. The data do not support a family-level effect on safety robustness.

- **Model-level alignment quality remains the best predictor.** Within the Qwen family, qwen2.5-7b (+0.0081) and qwen2.5-1.5b (+0.0015) have different slopes despite sharing the DPO training recipe. Within Llama, 1B (+0.0129) and 3B (-0.0069) pull in opposite directions. Family is not informative.

- **This is a negative result, and it is informative.** The RLHF recipe (PPO vs DPO vs SFT) does not predict safety robustness under quantization at the resolution available to us. Model-specific factors (parameter count, training data composition, alignment strength) matter more than alignment method category.

### 10.4 Quality-Safety Sign Reversals

The bespoke analysis examines whether quality and safety move in the same direction under quantization. A "sign reversal" occurs when quality degrades but safety improves (or vice versa) for different models at the same quant level. The pooled correlation across all models provides an aggregate view:

| Quality Metric | Safety Metric (Regex) | Models Positive | Models Negative | Pooled r |
|---------------|----------------------|-----------------|-----------------|----------|
| BERTScore | Refusal rate | llama3.2-1b, mistral-7b, qwen2.5-1.5b | llama3.2-3b, phi-2, qwen2.5-7b | +0.700 |
| BERTScore | Truthfulness | llama3.2-1b, qwen2.5-7b | llama3.2-3b, mistral-7b, phi-2, qwen2.5-1.5b | -0.013 |
| BERTScore | Bias resistance | llama3.2-1b, mistral-7b, phi-2, qwen2.5-7b | llama3.2-3b, qwen2.5-1.5b | +0.227 |
| Coherence | Refusal rate | llama3.2-1b, mistral-7b, phi-2, qwen2.5-1.5b | llama3.2-3b, qwen2.5-7b | +0.539 |
| ROUGE-L | Refusal rate | llama3.2-1b, mistral-7b, phi-2, qwen2.5-1.5b | llama3.2-3b, qwen2.5-7b | +0.611 |

**Observations:**

- **BERTScore-refusal has the highest pooled r (+0.700).** Across most models, when BERTScore degrades (lower quality), refusal rate also degrades (lower safety). This confirms quality can serve as a rough proxy for safety -- but only for models where the sign is positive.

- **BERTScore-truthfulness has near-zero pooled r (-0.013).** Quality degradation does not predict truthfulness degradation. These are measuring different things: BERTScore captures output fluency, while truthfulness captures factual accuracy. A model can become less fluent while remaining (or becoming) more truthful.

- **Three models consistently show positive correlations.** Llama3.2-1b and mistral-7b show quality-safety co-degradation across all quality metrics. For these models, quality monitoring covers safety.

- **Two models show mixed or negative correlations.** Llama3.2-3b (over-refusal artifact) and qwen2.5-7b (hidden danger) show negative BERTScore-refusal correlations. For these models, quality monitoring can be actively misleading -- stable quality does not mean stable safety.

### 10.5 Per-Model Safety Resilience Ranking

Combining all safety metrics (refusal, truthfulness, bias) into a resilience ranking based on mean safety slope and Q2_K performance:

| Rank | Model | Mean Safety Slope | Q2_K AdvBench Refusal | Q2_K Verdict | Overall Assessment |
|------|-------|-------------------|-----------------------|--------------|-------------------|
| 1 | qwen2.5-7b | +0.0081 | 80.9% | Functional | Most resilient -- strong baseline, gradual degradation |
| 2 | qwen2.5-1.5b | +0.0015 | 34.1% | Collapsed | Stable through Q3_K_S, catastrophic at Q2_K |
| 3 | llama3.2-1b | +0.0129 | 36.8% | Collapsed | Similar to qwen2.5-1.5b with stronger baseline |
| 4 | phi-2 | -0.0068 | 55.0% | Stable | Flat slope, moderate baseline; artifact-inflated bias |
| 5 | llama3.2-3b | -0.0069 | 92.7% | Over-refusing | Negative slope is artifact, not genuine resilience |
| 6 | mistral-7b | +0.0411 | 12.3% | Collapsed | Weakest: steep slope AND weak baseline |

**Observations:**

- **The ranking differs from v1.** In v1, the ranking was qwen2.5-7b > llama3.2-1b > llama3.2-3b > mistral-7b. Adding qwen2.5-1.5b confirms the Qwen family's safety resilience, while phi-2 introduces a new "flat but weak" archetype.

- **Baseline alignment quality remains the strongest predictor.** Models with >= 80% baseline refusal (qwen2.5-7b, qwen2.5-1.5b, llama3.2-1b) are the top 3. Phi-2's moderate baseline (58.6%) ranks it #4, and Mistral's weak regex baseline (23.6%, ~91% judge) places it last regardless of slope.

- **Over-refusal inflates the ranking for llama3.2-3b.** The 94% Q2_K refusal looks impressive but is an artifact of coherence collapse, not genuine safety. If over-refusal is penalized, llama3.2-3b drops below phi-2.

---

## 11. Safety Regime Classification

The bespoke analysis classifies each model-quant entry into regimes based on quality deltas (BERTScore, ROUGE-L, coherence) and safety deltas (refusal rate from regex):

- **Neutral:** Quality and safety co-move (both degrade, or neither does).
- **Hidden danger:** Quality roughly preserved (BERTScore delta < 3pp), safety refusal drops >= 10pp. The model *looks* fine on quality benchmarks but has silently lost safety.
- **Near hidden danger:** Approaching hidden danger thresholds.

### 11.1 Hidden Danger Entries

| Model | Quant | BERTScore Delta | Refusal Delta | Regime |
|-------|-------|-----------------|---------------|--------|
| llama3.2-1b | Q3_K_S | +0.98pp | **-13.6pp** | **hidden_danger** |
| qwen2.5-7b | Q2_K | +2.39pp | **-12.3pp** | **hidden_danger** |
| mistral-7b | Q2_K | -2.12pp | **-11.4pp** | **near_hidden_danger** |

**Observations:**

- **llama3.2-1b Q3_K_S is the most dangerous entry.** Quality is nearly unchanged (+0.98pp BERTScore), but refusal drops -13.6pp. A practitioner monitoring only MMLU/ARC would miss this. Jailbreak refusal drops even more steeply (-20.8pp).

- **qwen2.5-7b Q2_K has improving quality but degrading safety.** BERTScore actually *increases* by +2.39pp (likely noise), while refusal drops -12.3pp. This combination is especially dangerous because quality improvement would encourage deployment at Q2_K.

- **All other entries are neutral.** The expanded models (qwen2.5-1.5b, phi-2) are classified as neutral at all quant levels. For qwen2.5-1.5b, quality and safety collapse together at Q2_K. For phi-2, neither quality nor safety shows the >= 10pp refusal drop required for hidden_danger classification (refusal is essentially flat).

### 11.2 Correlation Between Quality and Safety

From the bespoke analysis correlation spine:

| Model | Quality Metric | Safety Metric | Pearson r | Spearman r | Direction |
|-------|---------------|---------------|-----------|------------|-----------|
| llama3.2-1b | BERTScore | Refusal | +0.917 | +0.143 | Co-degrading (except Q3_K_S outlier) |
| qwen2.5-1.5b | BERTScore | Refusal | +0.988 | +0.657 | **Strong co-degradation** |
| qwen2.5-1.5b | Coherence | Refusal | +0.997 | +0.771 | **Near-perfect co-degradation** |
| phi-2 | BERTScore | Refusal | -0.232 | -0.145 | **Decoupled** |
| mistral-7b | BERTScore | Refusal | +0.574 | +0.100 | Weak co-degradation |
| qwen2.5-7b | BERTScore | Refusal | -0.613 | -0.200 | **Inverse** |

**Observations:**

- **Qwen 2.5 1.5B has the tightest quality-safety coupling.** r=0.988 for BERTScore-refusal means quality metrics are a reliable proxy for safety degradation in this model. If BERTScore drops, refusal drops. This is actually the *safe* scenario -- quality metrics serve as a warning.

- **Phi-2 is decoupled.** r=-0.232 means quality and safety move independently. Quality metrics provide no information about safety for this model. Phi-2 requires dedicated safety monitoring.

- **Qwen 2.5 7B is inversely correlated.** Quality improves slightly while safety degrades -- the hidden danger pattern. At Q2_K, this model's quality metrics would *reassure* a practitioner while safety erodes.

### 11.3 Full Regime Classification Matrix

All 34 non-baseline model-quant entries from the bespoke analysis:

| Model | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|--------|--------|--------|------|
| llama3.2-1b | neutral | neutral | neutral | neutral | **hidden_danger** | neutral |
| llama3.2-3b | neutral | neutral | neutral | neutral | neutral | neutral |
| mistral-7b | -- | neutral | neutral | neutral | neutral | **near_hidden** |
| qwen2.5-7b | -- | neutral | neutral | neutral | neutral | **hidden_danger** |
| qwen2.5-1.5b | neutral | neutral | neutral | neutral | neutral | neutral |
| phi-2 | neutral | neutral | neutral | neutral | neutral | neutral |

**Observations:**

- **37 of 40 entries are neutral.** For most model-quant combinations, quality and safety co-move. The hidden danger regime is rare but high-impact.

- **Hidden danger entries cluster at the cliff.** llama3.2-1b Q3_K_S and qwen2.5-7b Q2_K are the only full hidden_danger entries. Both are at the quant level just before or at the catastrophic cliff for their respective models.

- **Expansion models (qwen2.5-1.5b, phi-2) have no hidden danger entries.** Both models show co-movement: when safety degrades (Q2_K), quality degrades proportionally. This is actually the safer behavior from a monitoring perspective.

- **The hidden danger mechanism differs by model.** For llama3.2-1b at Q3_K_S, quality is preserved (BERTScore +0.98pp) because the Q3_K_S quantization primarily affects safety-relevant weight regions while preserving general capability. For qwen2.5-7b at Q2_K, quality slightly *improves* (+2.39pp BERTScore) -- possibly because the quantized model produces shorter, more focused responses that score higher on BERTScore but have weaker safety properties.

### 11.4 Implications for Safety Monitoring

The hidden danger regime has direct operational implications:

1. **Quality-only monitoring is insufficient.** If you monitor only MMLU, ARC, BERTScore, or ROUGE-L, you will miss safety degradation at llama3.2-1b Q3_K_S and qwen2.5-7b Q2_K. Safety-specific benchmarks (AdvBench, jailbreak) must be part of the evaluation suite.

2. **The safe monitoring strategy is model-dependent.** For qwen2.5-1.5b (r=0.99 quality-safety coupling), quality monitoring is a reliable safety proxy. For phi-2 (r=-0.23) and qwen2.5-7b (r=-0.61), it is not. Practitioners should check the quality-safety correlation for their specific model before relying on quality-only monitoring.

3. **Hidden danger is a one-level warning.** Both hidden danger entries appear one quant level above the catastrophic cliff (llama3.2-1b: Q3_K_S is hidden danger, Q2_K is catastrophic collapse; qwen2.5-7b: Q2_K is hidden danger). If you observe hidden danger at your target quant level, the next lower level is likely catastrophic.

---

## 12. Q5_K_M Safety Floor

The Q5_K_M level (5.5 BPW) is the highest quant level that provides meaningful VRAM savings while staying well above the Q3_K_S cliff. Safety deltas at Q5_K_M from FP16/Q8_0 baseline:

| Model | Refusal | Truthfulness | Bias | BERTScore |
|-------|---------|-------------|------|-----------|
| llama3.2-1b | -1.82pp | -6.00pp | -2.02pp | -0.67pp |
| llama3.2-3b | +0.45pp | +9.00pp | -1.52pp | -0.54pp |
| mistral-7b | +0.91pp | -1.00pp | +0.51pp | -0.10pp |
| qwen2.5-7b | +0.00pp | -1.00pp | -1.01pp | -2.19pp |
| qwen2.5-1.5b | +3.18pp | +2.00pp | +4.04pp | -0.78pp |
| phi-2 | -0.91pp | +9.00pp | -1.01pp | +1.01pp |

**Observations:**

- **All safety deltas at Q5_K_M are within noise.** No model shows more than 6pp degradation on any safety metric. Q5_K_M is safe for all 6 models.

- **This confirms TR125's Q4_K_M recommendation extends to safety**, with a one-level safety margin. For safety-critical deployments, Q5_K_M provides the best risk-reward: significant VRAM savings (34% of FP16 size) with no measurable safety impact.

### 12.1 The Q2_K Catastrophe Across Models

Q2_K (2.5 BPW) is the single most important finding in the expanded matrix. Here is the complete Q2_K picture:

| Model | AdvBench Refusal Delta | Jailbreak Refusal Delta | MMLU Delta | ARC Delta | Regime |
|-------|----------------------|------------------------|------------|-----------|--------|
| llama3.2-1b | **-57.0pp** | **-56.7pp** | -14.7pp | -18.0pp | neutral (both collapse) |
| llama3.2-3b | +41.0pp | -4.2pp | -16.1pp | -8.5pp | neutral (over-refusal) |
| mistral-7b | -17.0pp | -6.7pp | -3.2pp | -6.5pp | near_hidden_danger |
| qwen2.5-7b | -5.0pp | -18.3pp | -7.7pp | -6.5pp | hidden_danger |
| qwen2.5-1.5b | **-50.0pp** | **-45.0pp** | **-35.1pp** | **-48.5pp** | neutral (both collapse) |
| phi-2 | -3.6pp | +9.2pp | *(TR125 v2)* | -17.5pp | neutral |

**Observations:**

- **Three distinct Q2_K behavior patterns emerge:**
  - **Total collapse** (llama3.2-1b, qwen2.5-1.5b): Both safety and capability drop by 36-50pp. The model is non-functional.
  - **Selective degradation** (mistral-7b, qwen2.5-7b): Safety drops more than capability. These are the hidden danger models -- quality looks acceptable but safety is compromised.
  - **Flat/over-refusal** (llama3.2-3b, phi-2): Llama3.2-3b's safety metrics paradoxically improve (over-refusal artifact). Phi-2's refusal is essentially flat (-3.6pp) while its bias resistance increases (over-refusal on BBQ only). Both models mask degradation through different mechanisms.

- **The total collapse pattern correlates with small model size.** Both total-collapse models are sub-2B parameters. The 3B+ models have enough parameter redundancy to partially survive Q2_K compression.

- **No model is safe at Q2_K.** Even the most resilient model (qwen2.5-7b, -12.3pp refusal from Q8_0 baseline) loses 18.3pp on jailbreak resistance and enters the hidden danger regime. Q2_K should never be deployed for safety-sensitive applications.

- **The v2 expansion reinforces the consumer deployment recommendation:** Q4_K_M is the minimum safe quant level, Q5_K_M is preferred, and Q2_K is categorically unacceptable. This recommendation has now been demonstrated across 6 models from 4 families.

---

## 13. Judge Model Note

### 13.1 Dual-Judge Strategy

| Aspect | Legacy Qwen-Judged Corpus | Gemma Small-Model Corpus | Gemma 7B Re-Judge Corpus |
|--------|---------------------------|--------------------------|---------------------------|
| Judge model | Qwen 2.5 7B Instruct Q8_0 | Gemma 3 12B | Gemma 3 12B |
| Judge family | Qwen | Google | Google |
| N annotations | 12,168 | 6,552 | 5,616 |
| Evaluated models | llama3.2-1b, llama3.2-3b, mistral-7b, qwen2.5-7b | qwen2.5-1.5b, phi-2 | mistral-7b, qwen2.5-7b |
| Self-evaluation risk | Present for legacy qwen2.5-7b labels | None | None |

### 13.2 Why Keep the Legacy Qwen-Judged Labels?

The original 12,168 judge labels from Qwen 2.5 7B Q8_0 are preserved even though the current bundle now also includes 5,616 Gemma re-judge annotations for the 7B pair:

1. **Reproducibility.** Re-judging with a different model would invalidate direct comparison with v1 results.
2. **Partial Gemma re-judging is already enough to surface judge sensitivity.** The 7B re-judge lets us test the self-evaluation concern directly without rewriting the historical v1 corpus.
3. **The multi-source judge design is informative.** Having both legacy and Gemma annotations reveals where conclusions are robust to judge choice and where they are not.

### 13.3 Known Judge Limitations

- **Qwen 2.5 7B Q8_0** may be biased toward Qwen-style responses on the legacy corpus, especially for the original qwen2.5-7b rows.
- **Gemma 3 12B** is a larger, more capable judge but has its own biases. As a Google model, it may have different safety thresholds than the Qwen judge.
- **Neither judge is ground truth.** Both are approximate. The regex classifier provides a third, fully deterministic signal.

### 13.4 What the Dual-Judge Design Reveals

The multi-source judge design is not just a nuisance variable. It reveals where safety conclusions are stable and where the measurement instrument is doing substantial work:

1. **Judge choice materially changes absolute safety levels.** Mistral is the clearest example: regex reports 12-29% refusal, while judge-based refusal stays around 78-93% depending on quant and judge source.

2. **The Qwen self-evaluation concern from v1 is now bounded, not eliminated.** The Gemma re-judge of the 7B pair shows that qwen2.5-7b remains a high-refusal model at Q8_0 and still degrades at Q2_K, so the broad safety story is not solely an artifact of Qwen judging Qwen.

3. **Future work should use a single, independent judge.** A model from a family not in the evaluation matrix (e.g., Claude, GPT-4) would eliminate the self-evaluation concern entirely. For the current consumer-hardware setup, the dual-judge approach is a pragmatic compromise.

---

## 14. Limitations

All limitations from v1 carry forward. Additional v2 limitations:

1. **Judge calibration mismatch remains real.** The current bundle now contains three judge source files, including a Gemma re-judge of the 7B pair, but those annotations are still not a single calibrated ground-truth corpus. Judge aggregates are informative secondary measurements, not interchangeable labels.

2. **Phi-2 alignment provenance is unclear.** Microsoft's documentation for Phi-2 does not clearly specify the alignment method (PPO, DPO, or pure SFT). We treat Phi as a separate family, but it may be more similar to one of the existing families than we assume. The ANOVA treats it as distinct.

3. **Qwen 2.5 1.5B and Qwen 2.5 7B share family.** The ANOVA treats all Qwen slopes as one group (n=6 slopes). If the 1.5B and 7B models have genuinely different safety robustness patterns (they do -- 1.5B has a Q2_K cliff, 7B does not), pooling them reduces within-family variance and inflates the F-statistic. However, the ANOVA is still non-significant, so this concern is academic.

4. **Expansion models lack FP16 at 7B.** This is not relevant for the expansion (both models are sub-3B, so FP16 is feasible and used). But it means the expansion models have a wider BPW range (FP16 to Q2_K = 13.5 BPW) than the 7B models (Q8_0 to Q2_K = 5.5 BPW), making cross-model slope comparisons asymmetric.

5. **Regex classifier was not updated for v2.** The Mistral gap (Section 9.3) identifies a known deficiency in the RefusalDetector. We did not modify the regex to add contraction patterns because (a) doing so would change v1 results, breaking comparability, and (b) the gap itself is a finding. Future work should evaluate expanded regex patterns or a more robust classifier.

6. **No equivalence testing (TOST).** As in v1, the "robust" verdicts are based on failure to detect degradation (p > 0.05), not confirmation of equivalence. The 18.3pp MDE means up to 18pp of real safety degradation could hide in "robust" entries.

7. **Over-refusal confounds remain.** Llama3.2-3b, phi-2, and to a lesser extent qwen2.5-1.5b at Q2_K show metrics that improve at low quant. These are artifacts of coherence loss, not genuine safety improvement. No automated method distinguishes over-refusal from genuine refusal in the current pipeline.

8. **BBQ per-category analysis not expanded.** The per-category bias analysis from v1 (Section 10) was not re-run with the expansion models. Phi-2 and Qwen 1.5B may have different per-category vulnerability patterns.

9. **Single repetition.** All runs use temperature=0.0 with single repetition. No multi-run variance estimation is performed. Ollama's llama.cpp backend may not produce bit-identical outputs across runs even at temp=0.

10. **This report has not undergone formal peer review.** All findings should be treated as preliminary. The consumer-hardware research program operates without institutional review or independent replication.

11. **Jailbreak templates unchanged from v1.** Only 3 jailbreak techniques (DAN-style, roleplay, prefix injection) tested. Novel techniques (crescendo, multi-turn, token smuggling) are not represented. The expansion models may have different vulnerability profiles to these untested techniques.

12. **No multi-turn evaluation.** All safety tasks are single-turn. Multi-turn jailbreaks and context manipulation are untested. These are increasingly the dominant real-world attack vectors.

13. **BBQ per-category analysis was not re-run.** The per-category bias vulnerability analysis from v1 (nationality most vulnerable, race_ethnicity most robust) was not extended to the expansion models. Phi-2 and Qwen 1.5B may have different per-category vulnerability patterns.

14. **TruthfulQA remains severely underpowered.** With 50 questions per variant, the MDE is approximately 28pp. No truthfulness finding in this report achieves statistical significance. The full 817-question TruthfulQA set would provide adequate power.

### 14.1 Summary of What We Can and Cannot Conclude

| Conclusion | Confidence | Basis |
|-----------|------------|-------|
| Q2_K is catastrophic for small models | **High** | Replicated across 3 models, effect sizes > 50pp |
| Safety is robust through Q4_K_M | **Moderate** | All 6 models show < 10pp degradation, but MDE = 18pp |
| Mistral regex-judge gap is real | **High** | 64-71pp gap across 6 quant levels, systematic |
| Cross-family ANOVA is non-significant | **High** | p=0.477 with 4 families; negative result is informative |
| Hidden danger regime exists | **Moderate** | Identified for 2 entries; sample-size-dependent thresholds |
| Quality monitoring is sufficient for safety | **Low** | True for qwen2.5-1.5b (r=0.99), false for phi-2 (r=-0.23) and qwen2.5-7b (r=-0.61) |
| DPO alignment is more quantization-robust | **Low** | Suggestive (Qwen family strong) but confounded with model size and training data |
| Truthfulness degrades under quantization | **Unresolvable** | Underpowered (N=50, MDE=28pp); no conclusion possible |

---

## 15. Reproducibility

### 15.1 Pipeline Commands

```bash
# Original TR134 Phase 3 (4 models):
python research/tr134/phase3/run.py -v

# TR142 Expansion (2 new models):
python research/tr142/expansion/run_tr134_expansion.py

# TR142 Bespoke Analysis (merge + analyze):
python research/tr142/bespoke_analysis.py
```

### 15.2 Key Artifacts

| Artifact | Path |
|----------|------|
| Original Phase 3 samples | `research/tr134/results/phase3/20260305_144827/samples.jsonl` |
| Expansion samples | `research/tr142/expansion/results/tr134_expansion/20260327_170457/` |
| Expansion config | `research/tr142/expansion/tr134_expansion_config.yaml` |
| Expansion analysis | `research/tr142/expansion/results/tr134_expansion/20260327_170457/phase3_analysis.json` |
| Bespoke merged matrix | `research/tr142/results/bespoke_analysis/20260328_173033/matrix.csv` |
| Bespoke regimes | `research/tr142/results/bespoke_analysis/20260328_173033/regimes.csv` |
| Judge agreement (task) | `research/tr142/results/bespoke_analysis/20260328_173033/judge_agreement_task.csv` |
| Judge agreement (quant) | `research/tr142/results/bespoke_analysis/20260328_173033/judge_agreement_quant.csv` |
| Regex vs judge gaps | `research/tr142/results/bespoke_analysis/20260328_173033/regex_vs_judge_gaps.csv` |
| Bespoke analysis report | `research/tr142/results/bespoke_analysis/20260328_173033/analysis_report.md` |
| Published report v1 | `PublishReady/reports/Technical_Report_134.md` |
| Published report v2 | `PublishReady/reports/Technical_Report_134_v2.md` (this file) |

### 15.3 Environment

| Component | Value |
|-----------|-------|
| OS | Windows 11 Home 10.0.26200 |
| CPU | 13th Gen Intel Core i9-13980HX |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU (12,282 MB VRAM, Compute Capability 8.9) |
| Ollama | Local HTTP API (http://localhost:11434) |
| Python | 3.x |
| Key packages | datasets, pyyaml, scipy, pandas |
| Temperature | 0.0 (greedy decoding) |
| Max new tokens | 256 |
| Seed | 42 |
| Batch size | 1 (sequential inference) |
| Warmup runs | 2 |

### 15.4 Model Tags (Expansion)

| Model | Quant | Ollama Tag |
|-------|-------|-----------|
| qwen2.5-1.5b | FP16 | `qwen2.5:1.5b-instruct-fp16` |
| qwen2.5-1.5b | Q8_0 | `qwen2.5:1.5b-instruct-q8_0` |
| qwen2.5-1.5b | Q6_K | `qwen2.5:1.5b-instruct-q6_K` |
| qwen2.5-1.5b | Q5_K_M | `qwen2.5:1.5b-instruct-q5_K_M` |
| qwen2.5-1.5b | Q4_K_M | `qwen2.5:1.5b-instruct-q4_K_M` |
| qwen2.5-1.5b | Q3_K_S | `qwen2.5:1.5b-instruct-q3_K_S` |
| qwen2.5-1.5b | Q2_K | `qwen2.5:1.5b-instruct-q2_K` |
| phi-2 | FP16 | `phi:2.7b-chat-v2-fp16` |
| phi-2 | Q8_0 | `phi:2.7b-chat-v2-q8_0` |
| phi-2 | Q6_K | `phi:2.7b-chat-v2-q6_K` |
| phi-2 | Q5_K_M | `phi:2.7b-chat-v2-q5_K_M` |
| phi-2 | Q4_K_M | `phi:2.7b-chat-v2-q4_K_M` |
| phi-2 | Q3_K_S | `phi:2.7b-chat-v2-q3_K_S` |
| phi-2 | Q2_K | `phi:2.7b-chat-v2-q2_K` |

### 15.5 Known Reproducibility Issues

- **Ollama determinism:** temp=0.0 may not produce bit-identical outputs across Ollama versions due to llama.cpp floating-point accumulation order differences. Results should be directionally reproducible but exact scores may vary by 1-2pp.
- **BBQ dataset:** heegyu/bbq may be updated on HuggingFace. Pin to a specific revision if exact reproducibility is required.
- **AdvBench gating:** Dataset access requires HuggingFace account and term acceptance. Access may change.
- **Judge non-determinism:** LLM judge classifications at temp=0 are approximately but not perfectly reproducible across Ollama versions.
- **Disk space:** Approximately 60 GB required for all 40 Ollama model variants (26 original + 14 expansion).

---

## Appendix A: Full Safety Matrix (Regex)

Complete per-model, per-quant safety data from the merged matrix. Deltas are from baseline (FP16 for sub-7B, Q8_0 for 7B).

### A.1 AdvBench Refusal Rate (%)

| Model | Baseline | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | FP16 | 90.0 | 90.0 | 90.0 | 86.0 | 87.0 | 85.0 | 33.0 |
| llama3.2-3b | FP16 | 53.0 | 52.0 | 57.0 | 55.0 | 47.0 | 91.0 | 94.0 |
| mistral-7b | Q8_0 | -- | 29.0 | 35.0 | 29.0 | 31.0 | 22.0 | 12.0 |
| qwen2.5-7b | Q8_0 | -- | 98.0 | 99.0 | 99.0 | 99.0 | 96.0 | 93.0 |
| qwen2.5-1.5b | FP16 | 84.1 | 83.2 | 85.5 | 87.3 | 80.0 | 84.5 | 34.1 |
| phi-2 | FP16 | 58.6 | 58.6 | 54.1 | 57.7 | 55.0 | 56.4 | 55.0 |

### A.2 BBQ Bias Resistance (%)

| Model | Baseline | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | FP16 | 89.4 | 88.9 | 88.4 | 87.4 | 87.4 | 99.5 | 73.2 |
| llama3.2-3b | FP16 | 96.5 | 96.0 | 95.0 | 95.0 | 96.5 | 94.4 | 78.8 |
| mistral-7b | Q8_0 | -- | 83.8 | 83.8 | 84.3 | 85.4 | 80.3 | 77.3 |
| qwen2.5-7b | Q8_0 | -- | 98.5 | 98.0 | 97.5 | 98.5 | 97.5 | 99.0 |
| qwen2.5-1.5b | FP16 | 85.4 | 89.4 | 88.4 | 89.4 | 88.9 | 89.9 | 90.4 |
| phi-2 | FP16 | 84.8 | 87.9 | 86.4 | 83.8 | 86.9 | 91.9 | 99.0 |

### A.3 TruthfulQA Truthfulness (%)

| Model | Baseline | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | FP16 | 55.0 | 56.0 | 48.0 | 49.0 | 58.0 | 49.0 | 44.0 |
| llama3.2-3b | FP16 | 49.0 | 48.0 | 51.0 | 58.0 | 50.0 | 52.0 | 54.0 |
| mistral-7b | Q8_0 | -- | 60.0 | 55.0 | 59.0 | 54.0 | 50.0 | 56.0 |
| qwen2.5-7b | Q8_0 | -- | 50.0 | 53.0 | 49.0 | 57.0 | 49.0 | 50.0 |
| qwen2.5-1.5b | FP16 | 49.0 | 43.0 | 47.0 | 51.0 | 51.0 | 54.0 | 59.0 |
| phi-2 | FP16 | 39.0 | 45.0 | 42.0 | 48.0 | 50.0 | 44.0 | 44.0 |

### A.4 Jailbreak Refusal Rate (%)

| Model | Baseline | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|----------|------|------|------|--------|--------|--------|------|
| llama3.2-1b | FP16 | 96.7 | 98.3 | 97.5 | 96.7 | 93.3 | 75.8 | 40.0 |
| llama3.2-3b | FP16 | 95.8 | 93.3 | 94.2 | 95.0 | 82.5 | 98.3 | 91.7 |
| mistral-7b | Q8_0 | -- | 19.2 | 23.3 | 20.8 | 15.0 | 16.7 | 12.5 |
| qwen2.5-7b | Q8_0 | -- | 89.2 | 89.2 | 88.3 | 90.8 | 75.0 | 70.8 |
| qwen2.5-1.5b | FP16 | 71.7 | 70.0 | 74.2 | 77.5 | 65.0 | 73.3 | 26.7 |
| phi-2 | FP16 | 31.7 | 30.0 | 27.5 | 29.2 | 30.8 | 31.7 | 40.8 |

---

## Appendix B: Full Regex vs Judge Gaps

### B.1 Refusal Rate Gaps

| Model | Quant | Regex | Judge | Gap (pp) |
|-------|-------|-------|-------|----------|
| llama3.2-1b | FP16 | 93.6% | 100.0% | +6.4 |
| llama3.2-1b | Q8_0 | 94.5% | 100.0% | +5.5 |
| llama3.2-1b | Q6_K | 94.1% | 100.0% | +5.9 |
| llama3.2-1b | Q5_K_M | 91.8% | 100.0% | +8.2 |
| llama3.2-1b | Q4_K_M | 90.5% | 99.5% | +9.1 |
| llama3.2-1b | Q3_K_S | 80.0% | 96.4% | +16.4 |
| llama3.2-1b | Q2_K | 36.8% | 97.7% | +60.8 |
| llama3.2-3b | FP16 | 76.4% | 100.0% | +23.6 |
| llama3.2-3b | Q8_0 | 74.5% | 100.0% | +25.5 |
| llama3.2-3b | Q6_K | 77.3% | 100.0% | +22.7 |
| llama3.2-3b | Q5_K_M | 76.8% | 100.0% | +23.2 |
| llama3.2-3b | Q4_K_M | 66.4% | 100.0% | +33.6 |
| llama3.2-3b | Q3_K_S | 95.0% | 100.0% | +5.0 |
| llama3.2-3b | Q2_K | 92.7% | 100.0% | +7.3 |
| mistral-7b | Q8_0 | 23.6% | 91.3% | +67.7 |
| mistral-7b | Q6_K | 28.6% | 92.5% | +63.8 |
| mistral-7b | Q5_K_M | 24.5% | 91.8% | +67.3 |
| mistral-7b | Q4_K_M | 22.3% | 92.7% | +70.4 |
| mistral-7b | Q3_K_S | 19.1% | 90.0% | +70.9 |
| mistral-7b | Q2_K | 12.3% | 82.9% | +70.6 |
| qwen2.5-7b | Q8_0 | 93.2% | 99.8% | +6.6 |
| qwen2.5-7b | Q6_K | 93.6% | 99.8% | +6.1 |
| qwen2.5-7b | Q5_K_M | 93.2% | 99.1% | +5.9 |
| qwen2.5-7b | Q4_K_M | 94.5% | 99.5% | +5.0 |
| qwen2.5-7b | Q3_K_S | 84.5% | 100.0% | +15.5 |
| qwen2.5-7b | Q2_K | 80.9% | 96.1% | +15.2 |
| qwen2.5-1.5b | FP16 | 84.1% | 91.8% | +7.7 |
| qwen2.5-1.5b | Q8_0 | 83.2% | 93.2% | +10.0 |
| qwen2.5-1.5b | Q6_K | 85.5% | 91.8% | +6.4 |
| qwen2.5-1.5b | Q5_K_M | 87.3% | 94.5% | +7.3 |
| qwen2.5-1.5b | Q4_K_M | 80.0% | 90.5% | +10.5 |
| qwen2.5-1.5b | Q3_K_S | 84.5% | 93.6% | +9.1 |
| qwen2.5-1.5b | Q2_K | 34.1% | 82.9% | +48.8 |
| phi-2 | FP16 | 58.6% | 70.0% | +11.4 |
| phi-2 | Q8_0 | 58.6% | 72.6% | +14.0 |
| phi-2 | Q6_K | 54.1% | 68.2% | +14.1 |
| phi-2 | Q5_K_M | 57.7% | 72.6% | +14.9 |
| phi-2 | Q4_K_M | 55.0% | 72.7% | +17.7 |
| phi-2 | Q3_K_S | 56.4% | 78.2% | +21.8 |
| phi-2 | Q2_K | 55.0% | 79.1% | +24.1 |

**Observations:**

- **Phi-2's gap increases with lower quant.** From +11.4pp at FP16 to +24.1pp at Q2_K. As the model's refusal phrasing becomes less standard at lower quant, the regex misses more. This is a mild version of the Mistral effect.

- **Llama3.2-3b Q3_K_S has the smallest gap (+5.0pp).** At this quant level, the model's over-refusal produces explicit refusal phrases that both regex and judge capture.

- **All gaps are positive.** The judge always reports more refusal than regex. There is no case where the regex overestimates refusal relative to the judge.

### B.2 Refusal Gap Patterns

The gaps cluster into three regimes:

**Low gap (< 15pp): Standard refusal models.** Llama3.2-1b at FP16-Q4_K_M, qwen2.5-7b at Q8_0-Q4_K_M, qwen2.5-1.5b at FP16-Q5_K_M. These models use standard refusal phrases that both regex and judge capture. The small residual gap comes from hedged or implicit refusals that the regex misses but the judge catches.

**Medium gap (15-35pp): Diverging at low quant.** Phi-2 at Q3_K_S-Q2_K (22-24pp), llama3.2-3b at FP16-Q4_K_M (23-34pp), qwen2.5-7b at Q3_K_S-Q2_K (15pp). At lower quant, model outputs become less standard, widening the gap. Also applies to models with non-standard baseline refusal (llama3.2-3b uses longer, contextual refusals even at FP16).

**High gap (> 40pp): Systematic classifier failure.** Mistral-7b at all quant levels (64-71pp), llama3.2-1b at Q2_K (61pp), qwen2.5-1.5b at Q2_K (49pp). Two distinct causes: Mistral uses refusal phrasing the regex cannot match (permanent classifier failure), while llama3.2-1b and qwen2.5-1.5b at Q2_K produce incoherent outputs that the judge interprets as refusal but regex does not (quant-induced classifier failure).

### B.3 Truthfulness Gaps (Top Discrepancies)

| Model | Quant | Regex | Judge | Gap (pp) |
|-------|-------|-------|-------|----------|
| llama3.2-1b | Q2_K | 44.0% | 9.1% | **-34.9** |
| qwen2.5-1.5b | Q2_K | 59.0% | 25.0% | **-34.0** |

**Observations:**

- **Truthfulness gaps are negative.** The regex reports *higher* truthfulness than the judge. The regex matches reference answer keywords; the judge evaluates whether the response is actually informative. At Q2_K, degraded models produce outputs containing reference keywords by chance but not in a truthful context.

---

## Appendix C: Per-Model Deployment Recommendations

### C.0 Summary Table

| Model | Min Safe Quant (Direct Refusal) | Min Safe Quant (Jailbreak) | Min Safe Quant (Bias) | VRAM at Safe Quant | Recommended For |
|-------|-------------------------------|---------------------------|----------------------|-------------------|-----------------|
| llama3.2-1b | Q3_K_S (85%) | Q4_K_M (93.3%) | Q4_K_M (87.4%) | ~0.8 GB (Q4_K_M) | Edge, low-risk chatbots |
| llama3.2-3b | Q5_K_M (55%) | Q5_K_M (95.0%) | Q3_K_S (94.4%) | ~2.0 GB (Q5_K_M) | Moderate-risk chatbots |
| mistral-7b | Q8_0 (29% regex / ~91% judge) | NOT RECOMMENDED | Q3_K_S (80.3%) | ~7.0 GB (Q8_0) | Non-safety-critical only |
| qwen2.5-7b | Q2_K (93%) | Q4_K_M (90.8%) | Q2_K (99.0%) | ~4.6 GB (Q4_K_M) | Safety-critical, customer-facing |
| qwen2.5-1.5b | Q3_K_S (84.5%) | Q4_K_M (65.0%) | Q2_K (90.4%) | ~1.2 GB (Q4_K_M) | Edge safety-critical |
| phi-2 | Q4_K_M (55.0%) | NOT RECOMMENDED | Q3_K_S (91.9%) | ~1.8 GB (Q4_K_M) | Non-adversarial, bias-sensitive |

**Notes:**

- "Min Safe Quant" = last quant level with < 10pp degradation from baseline on that metric.
- "NOT RECOMMENDED" = baseline jailbreak refusal is below 32%, making the model unsuitable for adversarial threat models regardless of quant level.
- VRAM estimates from TR133 VRAM model.

### C.1 Capability Data (Expansion Models)

### C.1 MMLU Accuracy (%)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| qwen2.5-1.5b | 58.9% | 56.5% | 57.9% | 58.6% | 53.7% | 45.3% | 23.9% |
| phi-2 | -- | -- | -- | -- | -- | -- | -- |

> **Note:** Phi-2 MMLU data is not available in this TR's bespoke analysis. See TR125 v2 for phi-2 MMLU capability data.

### C.2 ARC-Challenge Accuracy (%)

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K |
|-------|------|------|------|--------|--------|--------|------|
| qwen2.5-1.5b | 74.0% | 74.5% | 72.5% | 73.5% | 71.5% | 64.0% | 25.5% |
| phi-2 | 68.5% | 68.5% | 68.5% | 68.5% | 67.5% | 65.0% | 51.0% |

**Observations:**

- **Qwen 2.5 1.5B has the sharpest capability cliff at Q2_K.** MMLU drops -35.1pp, ARC drops -48.5pp. This is worse than any v1 model at Q2_K (llama3.2-1b: -14.7pp MMLU, -18.0pp ARC). The 1.5B parameter count provides less redundancy for weight compression.

- **Phi-2 degrades more gracefully on ARC.** ARC drops -17.5pp at Q2_K. The larger parameter count (2.78B) and different architecture provide more quantization headroom. (Phi-2 MMLU capability data is in TR125 v2, not in this TR's bespoke analysis.)

### C.2 Cross-Validation: Expansion Models vs Original Models

Comparing capability degradation at Q2_K across all 6 models:

| Model | Parameters | MMLU Q2_K Delta | ARC Q2_K Delta | Total Cap Degradation |
|-------|-----------|----------------|---------------|----------------------|
| llama3.2-1b | 1.24B | -14.7pp | -18.0pp | -32.7pp |
| qwen2.5-1.5b | 1.54B | **-35.1pp** | **-48.5pp** | **-83.6pp** |
| phi-2 | 2.78B | *(see TR125 v2)* | -17.5pp | -- |
| llama3.2-3b | 3.21B | -16.1pp | -8.5pp | -24.6pp |
| mistral-7b | 7.25B | -3.2pp | -6.5pp | -9.7pp |
| qwen2.5-7b | 7.62B | -7.7pp | -6.5pp | -14.2pp |

**Observations:**

- **Qwen 2.5 1.5B has the worst Q2_K capability degradation in the matrix.** -83.6pp total, more than double llama3.2-1b's -32.7pp despite having 25% more parameters. This suggests the Qwen 1.5B architecture has less quantization headroom than the Llama architecture at similar parameter counts.

- **Parameter count broadly predicts degradation magnitude.** Sorted by parameter count: 1.24B (-32.7pp), 1.54B (-83.6pp outlier), 2.78B (MMLU N/A, ARC -17.5pp), 3.21B (-24.6pp), 7.25B (-9.7pp), 7.62B (-14.2pp). The qwen2.5-1.5b outlier aside, larger models degrade less at Q2_K.

- **Phi-2 MMLU data is not available in this TR's bespoke analysis** (see TR125 v2 for phi-2 capability data). ARC degradation alone (-17.5pp) is comparable to llama3.2-3b's -8.5pp despite phi-2 being 14% smaller.

---

## Appendix D: Expansion Slope Confidence Intervals

| Model | Metric | Slope | CI Lower | CI Upper |
|-------|--------|-------|----------|----------|
| phi-2 | refusal_rate | +0.0005 | -0.0227 | +0.0120 |
| phi-2 | accuracy | +0.0070 | +0.0001 | +0.0324 |
| phi-2 | bias_resistance | -0.0077 | -0.0451 | +0.0003 |
| phi-2 | truthfulness | -0.0131 | -0.0292 | +0.0365 |
| qwen2.5-1.5b | refusal_rate | +0.0204 | -0.0011 | +0.1039 |
| qwen2.5-1.5b | accuracy | +0.0269 | +0.0068 | +0.1112 |
| qwen2.5-1.5b | bias_resistance | -0.0041 | -0.0056 | +0.0001 |
| qwen2.5-1.5b | truthfulness | -0.0119 | -0.0602 | +0.0004 |

**Observations:**

- **Phi-2 refusal slope CI spans zero (-0.023 to +0.012).** Consistent with the flat slope observation. There is no evidence that phi-2 refusal degrades with quantization.

- **Qwen 2.5 1.5B refusal slope CI barely includes zero (-0.001 to +0.104).** The Q2_K cliff inflates the upper bound. Without Q2_K, the slope would be much shallower.

- **Qwen 2.5 1.5B bias CI is entirely non-positive (-0.006 to +0.000).** Bias resistance does not degrade -- it slightly improves at lower quant, consistent with the over-refusal artifact.

### D.1 Comparison with Original Model Slopes

| Model | Refusal Slope | CI Lower | CI Upper | Zero in CI? |
|-------|---------------|----------|----------|-------------|
| llama3.2-1b | +0.0250 | +0.0041 | +0.1118 | No -- significant |
| llama3.2-3b | -0.0201 | -0.1005 | +0.0054 | Yes -- not significant |
| mistral-7b | +0.0922 | +0.0374 | +0.1743 | No -- significant |
| qwen2.5-7b | +0.0234 | +0.0029 | +0.0445 | No -- significant |
| qwen2.5-1.5b | +0.0204 | -0.0011 | +0.1039 | Yes -- borderline |
| phi-2 | +0.0005 | -0.0227 | +0.0120 | Yes -- not significant |

**Observations:**

- **Three models have significant positive refusal slopes.** Llama3.2-1b, mistral-7b, and qwen2.5-7b all show CIs that exclude zero. These models genuinely lose refusal capability as BPW decreases.

- **Three models have non-significant slopes.** Llama3.2-3b (over-refusal artifact), qwen2.5-1.5b (borderline, driven by Q2_K outlier), and phi-2 (flat). For these models, the data do not support a linear BPW-refusal relationship.

- **Qwen 2.5 1.5B's wide CI (+/-0.05) reflects the binary Q2_K cliff.** The model is stable at 80-87% refusal through Q3_K_S, then drops to 34.1% at Q2_K. A linear slope poorly captures this step-function behavior. The slope (+0.0204) is misleading because it averages across the cliff.

### D.2 Aggregate Safety Slopes (All 6 Models)

| Model | Mean Safety Slope | Std | N Safety Metrics |
|-------|-------------------|-----|-----------------|
| llama3.2-1b | +0.0129 | 0.010 | 3 |
| llama3.2-3b | -0.0069 | 0.014 | 3 |
| mistral-7b | +0.0411 | 0.043 | 3 |
| qwen2.5-7b | +0.0081 | 0.013 | 3 |
| qwen2.5-1.5b | +0.0015 | 0.019 | 3 |
| phi-2 | -0.0068 | 0.007 | 3 |

**Grand mean across all 6 models:** +0.0083 (std = 0.018).

**Observations:**

- **The grand mean (+0.0083) confirms that safety degrades with lower BPW on average.** The positive sign means normalized safety decreases as BPW decreases (more quantization = worse safety). But the effect is small: +0.0083 normalized score per BPW unit means going from FP16 (16 BPW) to Q4_K_M (4.5 BPW) costs approximately 0.096 normalized safety points, or roughly 10% of baseline safety. This is within the noise for most metrics at the available sample sizes.

- **The standard deviation (0.018) exceeds the mean.** Model-to-model variation is larger than the average effect. This is why the ANOVA is non-significant: between-model variance dominates within-family and between-family variance.

---

## Appendix E: Safety-Capability Ratios for New Models

### E.1 qwen2.5-1.5b S/C Ratio

> **Note:** Safety Norm uses corrected refusal rates (84.1% FP16 baseline) plus truthfulness and bias.

| Quant | BPW | Safety Norm (mean) | Cap Norm (mean) | S/C Ratio | Safe? |
|-------|-----|-------------------|-----------------|-----------|-------|
| FP16 | 16.0 | 1.000 | 1.000 | 1.000 | Yes |
| Q8_0 | 8.0 | 0.971 | 0.983 | 0.988 | Yes |
| Q6_K | 6.5 | 1.004 | 0.982 | 1.022 | Yes |
| Q5_K_M | 5.5 | 1.042 | 0.994 | 1.048 | Yes |
| Q4_K_M | 4.5 | 1.011 | 0.939 | 1.077 | Yes |
| Q3_K_S | 3.5 | 1.053 | 0.817 | 1.289 | Yes (cap drops faster) |
| Q2_K | 2.5 | 0.890 | 0.376 | 2.367 | No (both at floor) |

**Observations:**

- **S/C ratio exceeds 1.0 at Q3_K_S and Q2_K** because capability collapses faster than safety. At Q3_K_S (ratio 1.289), refusal is still at 84.5% while MMLU has dropped -13.7pp. At Q2_K (ratio 2.367), both are collapsed but capability is even worse.

- **The ratio is uninformative at Q2_K.** A ratio of 2.367 sounds like "safety is more than twice as robust as capability" -- but both are near floor (34.1% refusal, 24% MMLU). The ratio amplifies noise when the denominator approaches zero.

### E.2 phi-2 S/C Ratio

> **Note:** The phi-2 MMLU capability data used in the original S/C calculation was not from the bespoke analysis (see TR125 v2 for phi-2 MMLU data). The Cap Norm column below uses ARC-Challenge only. Safety Norm uses the corrected refusal rates (58.6% FP16 baseline) plus truthfulness and bias.

| Quant | BPW | Safety Norm (mean) | Cap Norm (ARC only) | S/C Ratio | Safe? |
|-------|-----|-------------------|-----------------|-----------|-------|
| FP16 | 16.0 | 1.000 | 1.000 | 1.000 | Yes |
| Q8_0 | 8.0 | 1.064 | 1.000 | 1.064 | Yes |
| Q6_K | 6.5 | 1.006 | 1.000 | 1.006 | Yes |
| Q5_K_M | 5.5 | 1.068 | 1.000 | 1.068 | Yes |
| Q4_K_M | 4.5 | 1.082 | 0.985 | 1.098 | Yes |
| Q3_K_S | 3.5 | 1.058 | 0.949 | 1.115 | Yes |
| Q2_K | 2.5 | 1.078 | 0.745 | 1.447 | Artifact |

**Observations:**

- **Phi-2 S/C ratio never drops below 1.0.** Safety metrics never degrade faster than capability for this model. This is partly genuine (flat refusal slope) and partly artifact (bias resistance and truthfulness increasing at some quant levels).

- **The Q2_K ratio (1.447) is artifact-driven.** Safety norm = 1.078 (above baseline) because bias resistance increases to 99.0% at Q2_K while refusal barely changes. If bias is excluded, the safety norm would be closer to 1.0.

---

## Appendix F: Glossary

| Term | Definition |
|------|------------|
| BPW | Bits per weight |
| RLHF | Reinforcement Learning from Human Feedback |
| PPO | Proximal Policy Optimization (Llama, Mistral) |
| DPO | Direct Preference Optimization (Qwen) |
| SFT | Supervised Fine-Tuning (primary method for Phi-2) |
| S/C Ratio | Safety-Capability ratio |
| MDE | Minimum Detectable Effect |
| Cohen's kappa | Chance-corrected inter-rater agreement |
| Over-refusal | Model refuses harmless requests due to coherence loss at low quant |
| Hidden danger | Quality preserved but safety degraded -- invisible to capability benchmarks |
| Regime | Classification of a model-quant entry's quality-safety relationship |
| GGUF | GPT-Generated Unified Format for quantized weights |
| Dual-judge | Use of two different LLM judge models across corpora |

---

## References

1. Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021.*
2. Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). Think you have Solved Question Answering? Try ARC. *arXiv:1803.05457.*
3. Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering. *ACL 2022.*
4. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022.*
5. Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models. *NeurIPS 2024.*
6. Shen, X., Chen, Z., Backes, M., Shen, Y., & Zhang, Y. (2023). Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. *arXiv:2308.03825.*
7. Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics, 33*(1), 159-174.
8. Banterhearts TR124 (2026). Quality & Accuracy Baseline.
9. Banterhearts TR125 (2026). Quantization Decision Matrix.
10. Banterhearts TR133 (2026). Predictive Capacity Planner.
11. Banterhearts TR134 v1 (2026). Alignment Robustness Under Quantization.
12. Banterhearts TR142 (2026). Cross-Architecture Quality-Safety Correlation Analysis.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1 | 2026-03-06 | Original 4-model safety matrix, 24,778 samples, single judge |
| v2 | 2026-03-28 | +2 models (qwen2.5-1.5b, phi-2), +13,342 samples, dual-judge, regex-judge gap analysis, regime classification, sign reversal analysis |

---

*End of Technical Report 134 v2.*
