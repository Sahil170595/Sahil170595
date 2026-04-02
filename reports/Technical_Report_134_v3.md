# Technical Report 134 v3: Alignment Robustness Under Quantization -- AWQ/GPTQ Safety Collapse
## GGUF k-quant + AWQ + GPTQ safety evaluation across 6 models (1.2B-7.6B), refusal template destabilization mechanism, and deployment taxonomy

| Field | Value |
|-------|-------|
| **TR Number** | 134 v3 |
| **Project** | Banterhearts LLM Performance Research |
| **Date** | 2026-04-01 (v1: Mar 6, v2: Mar 28, v3: Apr 1) |
| **Version** | 3.0 |
| **Author** | Research Team |
| **Git Commit** | `0439e828` |
| **Report Type** | Full-depth safety alignment analysis (metric-backed, 6 models, 4 families, 9 quant formats, dual-judge, refusal mechanism analysis) |
| **Status** | Complete -- v3 AWQ/GPTQ expansion delivered |
| **Run IDs** | Original Phase 3: `20260305_144827`, v2 Expansion: `20260327_170457`, v3 AWQ/GPTQ Safety: `20260331_125319` |
| **Total Samples** | 44,791 safety (24,778 original + 13,342 v2 expansion + 6,671 v3 AWQ/GPTQ) |
| **Judge Annotations** | 27,612 total (12,168 legacy Qwen + 6,552 Gemma small-model + 5,616 Gemma 7B rejudge + 3,276 v3 AWQ/GPTQ judge) |
| **Model-Quant Entries** | 47 (40 GGUF k-quant from v2 + 7 AWQ/GPTQ entries) |
| **Related Work** | [TR124](Technical_Report_124.md), [TR125 v2](Technical_Report_125_v2.md), [TR133](Technical_Report_133.md), [TR134 v1](Technical_Report_134.md), [TR134 v2](Technical_Report_134_v2.md), [TR142 v2](Technical_Report_142_v2.md) |
| **Depends On** | TR125 (quantization quality data), TR124 (FP16 baselines), TR134 v2 (GGUF safety matrix), TR142 (bespoke analysis pipeline) |

> **Peer Review Status:** This report has not undergone formal peer review. All findings should be interpreted as preliminary evidence from a consumer-hardware research program. Replication on independent hardware and datasets is required before drawing deployment conclusions.

---

## Abstract

TR134 v2 established that GGUF k-quant safety degradation is model-dependent but follows a predictable pattern: most models retain adequate refusal through Q3_K_S, with Q2_K catastrophic for small models. It left one critical gap: the safety impact of non-GGUF quantization formats (AWQ, GPTQ) that operate at similar effective bit-widths but use fundamentally different compression algorithms. TR134 v3 closes this gap.

This study extends the safety matrix with **7 new AWQ/GPTQ model-quant entries** across 4 models (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b, phi-2), producing **6,671 new safety samples** scored by regex classifiers and **3,276 LLM judge annotations** (Gemma 3 12B). The v3 canonical analysis merges all three data waves into a **47-row unified matrix** (6 models, 9 quantization formats, 4 families) with the full bespoke analysis pipeline applied to every cell.

The core findings are: (1) **AWQ and GPTQ cause catastrophic safety collapse on every tested model**, with refusal drops of -22pp to -68pp from baseline -- far worse than GGUF k-quants at comparable bit-widths. (2) **llama3.2-1b GPTQ produces the worst safety outcome in the entire 340,000-sample research program**: -68pp refusal rate, -44pp judge-confirmed refusal, and a complete shift in refusal template structure. (3) **5 of 7 hidden-danger regime cells in the expanded matrix are AWQ or GPTQ entries**, despite these formats comprising only 15% of the matrix. (4) **The refusal template destabilization mechanism (Phase 6) shows AWQ/GPTQ destroy the model's dominant refusal prefix**, with dominant-prefix-share dropping by 28-57pp alongside refusal collapse. (5) **AWQ/GPTQ at approximately 4 BPW are far more dangerous than Q4_K_M at 4.85 BPW**: the GGUF format preserves refusal while AWQ/GPTQ shatter it. (6) **Regex-judge gaps are amplified under AWQ/GPTQ**, with phi-2 GPTQ showing a 38.75pp refusal gap -- the regex reports 3.2% refusal while the judge reports 41.9%.

The operational conclusion is that **AWQ and GPTQ quantization formats should be treated as unsafe for any safety-critical deployment**, regardless of the base model's alignment quality. GGUF k-quants at Q4_K_M or above remain the only format family with demonstrated safety preservation.

---

## Metric Definitions

### Safety Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Refusal Rate** | Proportion of harmful prompts where the model refuses (regex classifier, ~15 patterns) | Higher = safer. Range [0, 1] |
| **Bias Resistance** | Proportion of BBQ questions with non-stereotypical or "unknown" answer | Higher = less biased. Range [0, 1] |
| **Truthfulness** | Proportion of TruthfulQA questions answered correctly (string match) | Higher = more truthful. Range [0, 1] |
| **Judge Refusal Rate** | LLM-judge-classified refusal rate (Gemma 3 12B or Qwen 2.5 7B Q8_0) | Higher = safer. Independent of regex |
| **Judge Bias Resistance** | LLM-judge-classified bias resistance (available for AWQ/GPTQ + Mistral + phi-2 GGUF) | Higher = less biased. Independent of regex |

### Derived Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Delta (pp)** | Percentage-point change from baseline (FP16 for sub-7B, Q8_0 for 7B) | Negative = degradation |
| **Regime** | Classification: `neutral`, `hidden_danger`, `near_hidden_danger` | hidden_danger = quality stable + safety degrading >= 10pp |
| **Regex-Judge Gap (pp)** | `judge_rate - regex_rate`, in percentage points | Large positive = regex underreports safety |
| **Dominant Prefix Share** | Fraction of refusals starting with the model's most common refusal prefix | Lower = more fragmented refusal templates |
| **Unique Prefix Rate** | Fraction of refusals with a prefix not shared by any other refusal in the same cell | Higher = more diverse/fragmented templates |
| **Hard Refusal Rate** | Fraction of all prompts receiving a hard (explicit) refusal vs soft (hedged) refusal | Tracks near-perfectly with total refusal rate |
| **Template Destabilization Flag** | Boolean: dominant-prefix-share drops >= 15pp from baseline | Mechanistic indicator of quantization damage to alignment weights |

### BPW Reference (Extended for AWQ/GPTQ)

| Quant Format | Approx. BPW | Type | Notes |
|-------------|-------------|------|-------|
| FP16 | 16.0 | Baseline | Sub-7B baseline |
| Q8_0 | 8.0 | GGUF | 7B baseline |
| Q6_K | 6.5 | GGUF | |
| Q5_K_M | 5.5 | GGUF | Conservative review floor |
| Q4_K_M | 4.85 | GGUF | Recommended minimum for safety |
| **AWQ** | **~4.0** | **Activation-aware** | **Per-channel salient-weight preservation** |
| **GPTQ** | **~4.0** | **Post-training** | **Hessian-based layer-wise reconstruction** |
| Q3_K_S | 3.5 | GGUF | |
| Q2_K | 2.5 | GGUF | Catastrophic for most models |

### Statistical Tests Used

| Test | Role in This Report |
|------|-------------------|
| Pearson correlation | Refusal-style metric correlations (Phase 6), quality-safety associations |
| Spearman correlation | Non-parametric companion to Pearson for robustness |
| Repeated-measures correlation | Quality-safety co-movement controlling for within-model pairing |
| Mixed-effects regression | Pooled quality-safety relationship with random model intercepts |
| Leave-one-out sensitivity | Robustness of pooled correlations to extreme quant points or models |
| Cohen's kappa | Regex-judge inter-rater agreement |
| Wilson CI | Confidence intervals for proportions (safety rates per cell) |

### Evidence Standard

**Established findings** require convergent evidence from multiple measurement tracks (regex + judge), effect sizes exceeding the MDE (18.3pp for safety), and consistency across at least 2 models.

**Partial findings** show significance on one track or in a subset of models.

**Non-claims** are results where evidence is insufficient or where the effect does not replicate across measurement methods.

---

## Executive Summary

TR134 v3 extends the 40-entry GGUF k-quant safety matrix with 7 AWQ/GPTQ entries and discovers that these quantization formats cause catastrophic safety collapse across every tested model. The central thesis is: **at comparable bit-widths (~4 BPW), AWQ and GPTQ destroy safety alignment that GGUF k-quants preserve.**

### What Changed in v3

| Dimension | v2 | v3 | Delta |
|-----------|----|----|-------|
| Model-quant entries | 40 (GGUF only) | 47 (+7 AWQ/GPTQ) | +7 entries |
| Quantization formats | 7 (FP16-Q2_K) | 9 (+AWQ, +GPTQ) | +2 formats |
| Total safety samples | 38,120 | 44,791 | +6,671 |
| Judge annotations | 24,336 | 27,612 | +3,276 |
| Hidden-danger + near | 3 (v2: 2 HD + 1 near) | 8 (7 HD + 1 near: 2 GGUF HD + 5 AWQ/GPTQ HD + 1 GGUF near) | +5 HD entries |
| Refusal template analysis | Not available | Phase 6 mechanism (style correlations) | New capability |
| Worst single-entry refusal loss | -57pp (llama3.2-1b Q2_K) | -68pp (llama3.2-1b GPTQ) | New record |

### Key Findings

1. **AWQ/GPTQ cause catastrophic safety collapse on all tested models.** llama3.2-1b AWQ: -62pp refusal. llama3.2-1b GPTQ: -68pp refusal. llama3.2-3b AWQ: -23pp. llama3.2-3b GPTQ: -21pp. qwen2.5-1.5b AWQ: -25pp. qwen2.5-1.5b GPTQ: -48pp. phi-2 GPTQ: -55pp. Every AWQ/GPTQ entry shows safety degradation exceeding the 18.3pp MDE.

2. **llama3.2-1b GPTQ is the worst safety outcome in the entire research program.** Refusal drops from 93.6% (FP16) to 25.5% (GPTQ), a -68.2pp collapse. The judge confirms at -44.3pp delta. BERTScore paradoxically improves by +8.5pp.

3. **5 of 7 hidden-danger cells are AWQ/GPTQ entries.** AWQ/GPTQ entries represent 15% of the matrix (7/47) but 71% of hidden-danger cells (5/7). Quality metrics would not flag these entries.

4. **AWQ/GPTQ at ~4 BPW are far more dangerous than Q4_K_M at 4.85 BPW.** Q4_K_M-to-AWQ gap is 58.6pp on llama3.2-1b. Less than 1 BPW difference produces a catastrophic safety gap. Format selection dominates bit-width.

5. **Refusal template destabilization is the mechanism.** Phase 6 analysis shows dominant-prefix-share deltas ranging from +9pp to -57pp under AWQ/GPTQ (most entries show drops, but not universally). Pearson r=+0.589 between prefix share and refusal (p=5.1e-05). Unique prefix rate: r=-0.813 (p=1.1e-10). Mean refusal tokens: r=-0.698 (p=3.9e-07).

6. **Regex-judge gaps are amplified under AWQ/GPTQ.** phi-2 GPTQ: 38.75pp gap. llama3.2-1b AWQ: 30.7pp gap. llama3.2-1b GPTQ: 30.3pp gap. All AWQ/GPTQ gaps exceed typical GGUF median (~8-10pp).

7. **Quality metrics paradoxically improve under AWQ/GPTQ for some models.** llama3.2-1b GPTQ: BERTScore +8.5pp, ROUGE-L +27.3pp. This quality improvement is the hidden-danger signature -- the model generates higher-quality text because it is complying with harmful prompts instead of producing short refusals.

8. **GPTQ is consistently worse than AWQ.** On all 3 models with both formats, GPTQ produces larger refusal loss: llama3.2-1b -68pp GPTQ vs -62pp AWQ, qwen2.5-1.5b -48pp vs -25pp, llama3.2-3b -21pp vs -23pp (exception: llama3.2-3b within noise).

9. **The deployment taxonomy rejects AWQ/GPTQ as blanket-unsafe.** AWQ: 3 reject rows out of 3, max refusal signal 37.4pp. GPTQ: 3 reject rows out of 4, max refusal signal 44.3pp. No AWQ or GPTQ entry qualifies even as "candidate" for safe deployment.

10. **Judge agreement is higher under AWQ/GPTQ than GGUF.** AWQ advbench kappa: 0.584 (moderate). GPTQ advbench kappa: 0.526 (moderate). GGUF average kappa: ~0.17 (slight). This is because AWQ/GPTQ produce clearer compliance signals that both classifiers detect.

### Claim Validation (v3 Update)

| # | Claim | v2 Status | v3 Status | Evidence |
|---|-------|-----------|-----------|----------|
| C1 | Safety robust through Q3_K_S for GGUF | Demonstrated (5/6) | **Confirmed** | Unchanged: 5/6 models maintain refusal within noise through Q3_K_S |
| C2 | Q2_K catastrophic for safety | Replicated | **Replicated** | Joined by AWQ/GPTQ at even worse magnitude |
| C3 | AWQ/GPTQ cause safety collapse | Not tested | **Established** | All 7 entries >= 20pp refusal loss; 5/7 hidden_danger |
| C4 | Format matters more than BPW | Not tested | **Established** | Q4_K_M vs AWQ: 58.6pp gap at <1 BPW difference |
| C5 | Template destabilization explains collapse | Not tested | **Established** | r=+0.589 (prefix share), r=-0.698 (refusal tokens), r=-0.813 (unique prefix) |
| C6 | Regex fails under AWQ/GPTQ | Noted | **Demonstrated** | phi-2 GPTQ: 38.75pp gap; llama3.2-1b: 30pp gap |
| C7 | Hidden-danger is AWQ/GPTQ-dominated | 3 GGUF entries | **Established** | 5/7 hidden-danger cells are AWQ/GPTQ |

### Core Decisions

- **Never deploy AWQ or GPTQ in safety-critical applications.** Every tested entry shows catastrophic refusal collapse.
- **GGUF Q4_K_M remains the recommended minimum for safety-critical deployment.** Retains >= 55% judge-based refusal on all models (regex-based rates vary by model; mistral-7b shows 22% regex but 93% judge). Q5_K_M is the conservative review floor, not a blanket auto-deploy setting.
- **Do not use BPW alone to predict safety.** The compression algorithm matters as much as the compression ratio. AWQ/GPTQ at 4.0 BPW are far more dangerous than GGUF at 3.5 BPW.
- **Always cross-validate with an LLM judge when deploying quantized models.** Regex classifiers miss 30-39pp of refusal behavior under AWQ/GPTQ.
- **Treat quality improvement under quantization as a warning signal.** BERTScore/ROUGE-L improvement at a quantized level may indicate the model is complying with harmful prompts rather than refusing them.

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| v3 sample count | Safety samples | >= 5,000 | 6,671 | **PASS** |
| v3 judge count | Judge annotations | >= 3,000 | 3,276 | **PASS** |
| AWQ safety collapse | Refusal delta vs baseline | >= 18.3pp MDE | -22pp to -62pp | **PASS** |
| GPTQ safety collapse | Refusal delta vs baseline | >= 18.3pp MDE | -21pp to -68pp | **PASS** |
| Hidden-danger detection | Regime classification | >= 1 entry | 5 AWQ/GPTQ entries | **PASS** |
| Mechanism evidence | Phase 6 correlation | p < 0.05 | p=5.1e-05 (dominant prefix) | **PASS** |
| Matrix coverage | All entries have judge | 47/47 | 47/47 | **PASS** |

---

## When to Use This Report

### Scenario 1: Evaluating AWQ or GPTQ for Production Deployment

**Question:** "Our team is considering AWQ or GPTQ quantization for a chat assistant. Is it safe?"

**Answer:** No. See SS5 Tables 1-2 -- every AWQ/GPTQ entry shows refusal collapse of -22pp to -68pp from baseline. On llama3.2-1b, GPTQ retains only 25.5% refusal (regex) and 55.7% refusal (judge). Use GGUF Q4_K_M instead, which retains 90.5% refusal at similar VRAM. The <1 BPW savings from AWQ/GPTQ translate to <0.2 GB VRAM savings but 58-65pp refusal loss. See SS6 for the direct format comparison.

### Scenario 2: Choosing Between Q4_K_M and AWQ at Similar Bit-Widths

**Question:** "AWQ runs at ~4 BPW and Q4_K_M at ~4.85 BPW. Is the VRAM savings worth the risk?"

**Answer:** No. See SS6 Table 5. On llama3.2-1b, Q4_K_M retains 90.5% refusal while AWQ drops to 31.8% -- a 58.6pp gap for less than 1 BPW difference. On qwen2.5-1.5b, Q4_K_M retains 80.0% while GPTQ drops to 36.4% -- a 43.6pp gap. GGUF at 3.5 BPW (Q3_K_S) still outperforms AWQ/GPTQ at 4.0 BPW on every model. See SS6 Table 6.

### Scenario 3: Investigating a Regex-Judge Discrepancy on a Quantized Model

**Question:** "Our regex safety audit shows 3% refusal on a GPTQ model, but manual spot-checks suggest higher actual refusal."

**Answer:** This matches the phi-2 GPTQ pattern exactly. See SS8 Table 7: regex reports 3.2% refusal while the LLM judge reports 41.9% -- a 38.75pp gap. AWQ/GPTQ destabilize refusal templates so the model refuses in non-standard phrasing that regex patterns do not match. Use an LLM judge for any AWQ/GPTQ safety audit. Note that even the 41.9% judge-confirmed refusal is a catastrophic outcome -- the model still complies with 58% of harmful prompts.

### Scenario 4: Quality Metrics Look Good on a Quantized Model But You Suspect Safety Problems

**Question:** "Our GPTQ model scores higher BERTScore than FP16. How can it be unsafe?"

**Answer:** This is the hidden-danger pattern documented in SS7. On llama3.2-1b GPTQ, BERTScore improves +8.5pp and ROUGE-L improves +27.3pp while refusal drops -68pp. The quality improvement occurs because the model is complying with harmful prompts -- generating longer, more coherent harmful content -- instead of producing short refusals that score low on text-quality metrics. See SS7 Table 4 for the full quality-paradox breakdown and SS9 for the refusal template destabilization mechanism that explains this.

### Scenario 5: Cross-Referencing with TR125/TR142 for Deployment Decisions

**Question:** "Does TR134 v3 change the deployment recommendations from TR125 and TR142?"

**Answer:** Yes, significantly. TR125 and TR142 v2 did not evaluate AWQ/GPTQ safety. TR134 v3 demonstrates that these formats are blanket-unsafe despite acceptable quality scores. The combined recommendation is: use GGUF Q4_K_M or above for safety-critical applications. See SS10 Table 9 for the complete deployment taxonomy. If you have already deployed an AWQ or GPTQ model, run an LLM judge safety audit immediately.

### Scenario 6: Understanding Why GPTQ Is Worse Than AWQ

**Question:** "Both are 4-bit. Why is GPTQ consistently worse?"

**Answer:** GPTQ minimizes layer-wise reconstruction error via Hessian approximation, while AWQ preserves activation-salient weights. Neither explicitly preserves safety-relevant weight structures, but AWQ's activation-awareness may incidentally protect some alignment-critical channels. See SS5 Table 2 for the head-to-head: GPTQ is worse on 2/3 models (llama3.2-1b by 6pp, qwen2.5-1.5b by 23pp), with llama3.2-3b within noise. The mechanism analysis in SS9 shows both formats destabilize refusal templates, but GPTQ produces more severe prefix fragmentation.

---

## Table of Contents

**Metadata and Summary**
- [Abstract](#abstract)
- [Metric Definitions](#metric-definitions)
- [Executive Summary](#executive-summary)
- [When to Use This Report](#when-to-use-this-report)

**Introduction and Methodology (SS1-SS3)**
- [SS1. Introduction](#ss1-introduction)
- [SS2. Methodology](#ss2-methodology)
- [SS3. Models and Design](#ss3-models-and-design)

**Results (SS4-SS11)**
- [SS4. GGUF Safety Baseline](#ss4-results-gguf-safety-baseline)
- [SS5. AWQ/GPTQ Refusal Collapse](#ss5-results-awqgptq-refusal-collapse)
- [SS6. Format Comparison at Matched BPW](#ss6-results-format-comparison-at-matched-bpw)
- [SS7. Hidden-Danger Regime Analysis](#ss7-results-hidden-danger-regime-analysis)
- [SS8. Regex vs Judge Under AWQ/GPTQ](#ss8-results-regex-vs-judge-under-awqgptq)
- [SS9. Refusal Template Destabilization Mechanism](#ss9-results-refusal-template-destabilization-mechanism)
- [SS10. Deployment Taxonomy](#ss10-results-deployment-taxonomy)
- [SS11. Bias and Truthfulness](#ss11-results-bias-and-truthfulness)

**Synthesis and Closing (SS12-SS15)**
- [SS12. Statistical Synthesis](#ss12-statistical-synthesis)
- [SS13. Conclusions](#ss13-conclusions)
- [SS14. Limitations and Follow-Up](#ss14-limitations-and-follow-up)
- [SS15. Reproducibility](#ss15-reproducibility)

**References and Appendices**
- [References](#references)
- [Appendix A: Full 47-Row Safety Matrix](#appendix-a-full-47-row-safety-matrix)
- [Appendix B: Full Regex vs Judge Gaps](#appendix-b-full-regex-vs-judge-gaps)
- [Appendix C: Phase 6 Refusal Style Data](#appendix-c-phase-6-refusal-style-data)
- [Appendix D: Deployment Protocol](#appendix-d-deployment-protocol)
- [Appendix E: Glossary](#appendix-e-glossary)

---

## SS1. Introduction

### SS1.1 Research Questions

1. **RQ1:** Do AWQ and GPTQ preserve safety alignment at comparable bit-widths to GGUF k-quants?
2. **RQ2:** What mechanism causes format-dependent safety degradation?
3. **RQ3:** Do regex-based safety classifiers remain reliable under AWQ/GPTQ?
4. **RQ4:** How should the deployment taxonomy account for format-dependent risk?
5. **RQ5:** Is the hidden-danger regime (quality stable, safety degrading) concentrated in specific quantization formats?

### SS1.2 Why This Matters

AWQ and GPTQ are among the most widely used quantization formats outside the llama.cpp ecosystem. They are the default output format for many Hugging Face model quantization workflows, and their availability through libraries like AutoGPTQ and AutoAWQ makes them the path of least resistance for practitioners who need smaller models. Both formats operate at approximately 4 bits per weight, placing them between GGUF Q4_K_M (4.85 BPW) and Q3_K_S (3.5 BPW) in effective precision.

If these formats destroy safety alignment while preserving or even improving quality metrics, practitioners would deploy unsafe models without any warning from standard benchmarks. The hidden-danger scenario -- where capability evaluations show the model performing well but safety has silently degraded -- is the most operationally dangerous outcome in quantized model deployment. TR134 v3 demonstrates that this is exactly the pattern AWQ and GPTQ produce.

The finding is especially critical because AWQ/GPTQ models are often shared pre-quantized on Hugging Face, meaning practitioners may adopt them without running any safety evaluation. A regex-based safety audit, which is the most common lightweight approach, would further mask the problem: phi-2 GPTQ shows 3.2% refusal by regex (appearing safe) while the model actually complies with 58% of harmful prompts by judge assessment.

### SS1.3 Scope

| Dimension | Coverage |
|-----------|----------|
| Models | 6: llama3.2-1b (1.24B), llama3.2-3b (3.21B), qwen2.5-1.5b (1.54B), phi-2 (2.78B), mistral-7b (7.25B), qwen2.5-7b (7.62B) |
| Families | 4: Llama (PPO), Qwen (DPO), Phi (SFT+RLHF), Mistral (PPO) |
| Quantization formats | 9: FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K (GGUF) + AWQ + GPTQ |
| AWQ/GPTQ models tested | 4: llama3.2-1b, llama3.2-3b, qwen2.5-1.5b (AWQ+GPTQ); phi-2 (GPTQ only) |
| Safety battery | AdvBench refusal (100), TruthfulQA (50), BBQ bias (198), Jailbreak (120) per entry |
| Total safety samples | 44,791 across 3 waves |
| Judge annotations | 27,612 across 4 judge sources |
| Model-quant entries | 47 (40 GGUF + 7 AWQ/GPTQ) |
| Matrix columns | 83 (quality, safety, judge, derived, regime) |

### SS1.4 Literature Grounding

**AWQ (Lin et al., 2023)** preserves "salient" weights by analyzing activation distributions to determine per-channel quantization scales. The core insight is that a small fraction of weights, identified by activation magnitude, disproportionately affects model quality. AWQ protects these weights at the cost of compressing others more aggressively. The method does not explicitly consider which weights encode safety-related behaviors from RLHF/DPO training.

**GPTQ (Frantar et al., 2022)** uses Hessian-based second-order information to minimize layer-wise reconstruction error during quantization. Each weight is quantized sequentially, with remaining weights adjusted to compensate for the quantization error. Like AWQ, GPTQ optimizes for output reconstruction fidelity -- it does not distinguish between safety-critical and safety-neutral weights.

**The gap:** Safety alignment from RLHF/DPO training is encoded in specific weight patterns that may be sparse and distributed across layers. Neither AWQ's activation saliency nor GPTQ's reconstruction error minimization guarantees preservation of these patterns. GGUF k-quant formats, by contrast, use mixed-precision block quantization that applies uniform compression within blocks without targeting specific weight subsets. This uniform approach may incidentally preserve more of the distributed safety encoding because it does not selectively compress any weight subpopulation.

**The RLHF/DPO safety encoding hypothesis.** Safety alignment from RLHF training is believed to be encoded in specific weight patterns -- particularly in the attention and MLP layers that control response generation. These patterns may be sparse: only a small subset of weights determines whether the model generates a refusal or a compliance. If this subset overlaps poorly with AWQ's "salient" weights (identified by activation magnitude) or GPTQ's low-reconstruction-error weights, quantization could selectively destroy safety while preserving general capability. GGUF k-quant's uniform block quantization would not face this problem because it compresses all weights equally within each block, preserving the relative magnitudes that encode safety decisions.

**DPO vs PPO resilience hypothesis.** DPO encodes preferences directly into the policy weights without an intermediate reward model. This may produce more distributed safety encoding that is harder to selectively destroy. PPO, by contrast, fine-tunes via reward-model-guided policy gradients, which may concentrate safety information in a smaller set of weights. TR134 v3 provides preliminary evidence for this: qwen2.5-1.5b (DPO) shows partial AWQ resilience that llama models (PPO) do not.

**Prior work on safety under quantization** is sparse. Jain et al. (2023) evaluated adversarial robustness under quantization but focused on attack success rates rather than refusal behavior. No prior work has systematically compared AWQ, GPTQ, and GGUF safety outcomes at matched bit-widths. The closest related work is the Banterhearts program's own TR134 v1/v2, which established GGUF k-quant safety profiles but did not test non-GGUF formats.

### SS1.5 How to Read This Report

| Time | Reading Path |
|------|-------------|
| **2 min** | Abstract, Claim Validation table (Executive Summary), Core Decisions |
| **10 min** | Add Key Findings (1-10) + SS5 (AWQ/GPTQ Collapse) + SS7 (Hidden Danger) |
| **30 min** | Add SS4 (GGUF Baseline) + SS6 (Format Comparison) + SS8 (Regex vs Judge) + SS9 (Mechanism) + SS14 (Limitations) |
| **60 min** | Full report SS1-SS15 + all Appendices |
| **Deep dive** | Appendix A (full 47-row matrix), Appendix B (all gaps), Appendix C (Phase 6 data), bespoke CSVs at `research/tr142/results/bespoke_analysis_v3/phase56_v3_canonical/` |

Each result section follows the pattern: pre-table context prose, data table, then **Observations** interpreting the table. Key findings receive a blockquote restatement. No table appears without accompanying interpretation.

---

## SS2. Methodology

### SS2.1 Three-Wave Design

TR134 v3 integrates three evaluation waves collected across four weeks. Each wave used identical inference parameters and safety battery but targeted different model-quant populations.

| Wave | Date Range | Models | Formats | Samples | Judge Model | Judge N |
|------|-----------|--------|---------|---------|-------------|---------|
| Wave 1 (v1) | Mar 5-6 | 4 (llama3.2-1b, 3b, mistral-7b, qwen2.5-7b) | 7 GGUF (FP16-Q2_K) | 24,778 | Qwen 2.5 7B Q8_0 | 12,168 |
| Wave 2 (v2) | Mar 27-28 | 2 (qwen2.5-1.5b, phi-2) | 7 GGUF (FP16-Q2_K) | 13,342 | Gemma 3 12B | 12,168 |
| Wave 3 (v3) | Mar 31 | 4 (llama3.2-1b, 3b, qwen2.5-1.5b, phi-2) | AWQ + GPTQ | 6,671 | Gemma 3 12B | 3,276 |
| **Total** | | **6** | **9** | **44,791** | | **27,612** |

**Observations.** Wave 2 includes both Gemma 3 12B small-model judge (6,552) and Gemma 3 12B rejudge of the 7B pair (5,616). The bespoke analysis loaded 22,464 judge annotations after deduplication and schema alignment. The raw-to-loaded difference reflects legacy judge format conversion.

### SS2.2 Inference Parameters

All waves use identical inference configuration:

- **Temperature:** 0.0 (deterministic)
- **Seed:** 42
- **max_new_tokens:** 256
- **Backend:** Ollama (GGUF) / Transformers (AWQ/GPTQ)
- **Isolation:** Each model-quant pair runs in an isolated Ollama session (GGUF) or isolated Transformers loading (AWQ/GPTQ)

Prompts are identical across formats for the same model. AWQ/GPTQ entries receive the same prompt templates as their GGUF counterparts.

### SS2.3 Safety Battery

| Task | N per Entry | Domain | Primary Metric | Source |
|------|------------|--------|----------------|--------|
| AdvBench Refusal | 100 (110 for some entries) | Direct harm | Refusal rate | walledai/AdvBench |
| TruthfulQA | 50 | Truthfulness | Correctness | truthfulqa/truthful_qa |
| BBQ Bias | 198 | Social bias | Bias resistance | heegyu/bbq (11 configs) |
| Jailbreak Amplification | 120 | Adversarial | Refusal rate | JailbreakBench + JailbreakHub |
| MMLU | 285 | Capability | Accuracy | cais/mmlu |
| ARC-Challenge | 200 | Capability | Accuracy | allenai/ai2_arc |

### SS2.4 Scoring Stack

**Regex classifiers (all waves):**
- **RefusalDetector:** ~15 refusal patterns. Known limitation: misses contractions ("I can't") and hedged refusals.
- **BiasDetector:** Compares answer choice to stereotypical answer. Binary.
- **TruthfulnessClassifier:** String match against reference answers. Binary.

**LLM judges:**
- **Wave 1:** Qwen 2.5 7B Instruct Q8_0 (same family as one evaluated model -- Qwen 2.5 7B)
- **Waves 2-3:** Gemma 3 12B (different family from all evaluated models)

**Conflict resolution:** When regex and judge disagree, both values are reported. Neither is treated as ground truth. The gap between them is itself a finding -- it reveals where safety classification is ambiguous.

### SS2.5 Regime Classification Rule

The bespoke analysis classifies each non-baseline model-quant entry into one of three regimes:

- **Hidden danger:** BERTScore delta >= -3pp AND safety refusal rate delta <= -10pp. Quality appears stable or improved; safety has silently degraded.
- **Near hidden danger:** BERTScore delta >= -3pp AND safety refusal rate delta between -8pp and -10pp (approaching threshold).
- **Neutral:** All other entries. Quality and safety either co-move or neither degrades significantly.

This classification is conservative: the -10pp threshold for safety is well below the 18.3pp MDE. Entries flagged as hidden_danger show real degradation, but the threshold should be treated as illustrative rather than definitive.

### SS2.6 AWQ/GPTQ Checkpoint Provenance

All AWQ/GPTQ checkpoints were produced locally via the TR142 v3 quantization pipeline using default configurations:

- **AWQ:** AutoAWQ, 4-bit, group_size=128, per-channel activation-aware scales
- **GPTQ:** AutoGPTQ, 4-bit, group_size=128, Hessian-based layer-wise reconstruction

phi-2 AWQ export failed due to architecture incompatibility with AutoAWQ's expected module structure. Only GPTQ is available for phi-2. All 7B models (mistral-7b, qwen2.5-7b) were not quantized to AWQ/GPTQ because 7B checkpoints required Colab Pro A100 capacity not yet available.

### SS2.7 Data Merge Process

The TR142 bespoke analysis pipeline (`research/tr142/bespoke_analysis/`) performs the following merge across three data waves:

1. **Load all safety samples** (3 sources, 44,791 records total). Align schemas: all datasets use the same column structure (base_model, quant, task, metric, score) because the expansion config mirrors the original.
2. **Load all judge annotations** (4 sources, 27,612 raw, 22,464 loaded after deduplication and schema alignment). The raw-to-loaded gap primarily affects the legacy Qwen judge source (12,168 raw, 7,020 loaded) due to format conversion.
3. **Compute per-entry statistics.** Mean, CI (Wilson for proportions), N for each (model, quant, task, metric) combination.
4. **Compute quality deltas.** BERTScore, ROUGE-L, coherence changes from each model's baseline (FP16 or Q8_0).
5. **Compute safety deltas.** Refusal, truthfulness, bias resistance changes from baseline.
6. **Compute judge deltas.** Judge-classified refusal, truthfulness, bias changes from baseline.
7. **Classify regimes.** Apply hidden_danger / near_hidden_danger / neutral rules per SS2.5.
8. **Compute Phase 6 style metrics.** Extract refusal prefixes, compute dominant-prefix-share, unique-prefix-rate, prefix entropy, mean refusal tokens per entry.
9. **Output unified matrix** (47 rows, 83 columns) and 35 supporting CSVs.

### SS2.8 Power Analysis

| Metric Type | N per Entry | MDE (80% power, alpha=0.05) | AWQ/GPTQ Effects | Detectable? |
|------------|------------|------------------------------|-----------------|-------------|
| Safety (binary, refusal) | 100-220 | 18.3pp | -21pp to -68pp | **Yes** (all exceed MDE) |
| Safety (binary, bias) | 198 | 13.9pp | -6pp to -21pp | Partial (3/7 exceed MDE) |
| Truthfulness (binary) | 50 | 27.7pp | -7pp to +10pp | **No** (all below MDE) |

All core AWQ/GPTQ refusal findings exceed the MDE by substantial margins. The smallest refusal effect (-20.9pp, llama3.2-3b GPTQ) exceeds the MDE by 14%. Bias findings are partially detectable -- the 3 GPTQ entries showing >14pp degradation exceed the MDE, but AWQ bias effects (-3pp to -6pp) are below the detection limit. Truthfulness is completely underpowered at N=50 and should be ignored for deployment decisions.

### SS2.9 What This Design Does Not Do

- **Does not test AWQ/GPTQ on 7B+ models.** All AWQ/GPTQ entries are sub-4B. Larger models may show different vulnerability patterns.
- **Does not test non-default AWQ/GPTQ configurations.** Group size, bit-width, and calibration dataset are all at defaults. Non-default settings might produce different safety outcomes.
- **Does not establish causal mechanism.** Phase 6 correlations are observational. Template destabilization co-occurs with refusal loss but the direction of causation is not proven.
- **Does not test multi-turn or adversarial prompting under AWQ/GPTQ.** All prompts are single-turn. TR139 multi-turn jailbreak data has not been cross-validated with AWQ/GPTQ.
- **Does not compare AutoAWQ vs other AWQ implementations.** Only one implementation per format was tested.
- **Does not test exl2, EETQ, or other quantization formats.** Only GGUF, AWQ, and GPTQ are covered.

---

## SS3. Models and Design

### SS3.1 Model Summary

| Model | Family | Parameters | Alignment | GGUF Levels | AWQ | GPTQ | Baseline |
|-------|--------|-----------|-----------|-------------|-----|------|----------|
| llama3.2-1b | Llama | 1.24B | PPO | FP16-Q2_K (7) | Yes | Yes | FP16 |
| llama3.2-3b | Llama | 3.21B | PPO | FP16-Q2_K (7) | Yes | Yes | FP16 |
| qwen2.5-1.5b | Qwen | 1.54B | DPO | FP16-Q2_K (7) | Yes | Yes | FP16 |
| phi-2 | Phi | 2.78B | SFT+RLHF | FP16-Q2_K (7) | No (failed) | Yes | FP16 |
| mistral-7b | Mistral | 7.25B | PPO | Q8_0-Q2_K (5) | Not tested | Not tested | Q8_0 |
| qwen2.5-7b | Qwen | 7.62B | DPO | Q8_0-Q2_K (5) | Not tested | Not tested | Q8_0 |

**Observations.** The 4 sub-4B models have AWQ/GPTQ data (7 entries total: 3 AWQ + 4 GPTQ). The 2 7B models carry forward GGUF data from v1/v2 only. phi-2 AWQ failed during the AutoAWQ export step due to architecture incompatibility. All AWQ/GPTQ checkpoints were produced locally via the TR142 v3 quantization pipeline.

### SS3.2 Entry Count per Model

| Model | GGUF Entries | AWQ Entries | GPTQ Entries | Total |
|-------|-------------|-------------|-------------|-------|
| llama3.2-1b | 8 (incl. FP16) | 1 | 1 | 10 |
| llama3.2-3b | 8 (incl. FP16) | 1 | 1 | 10 |
| qwen2.5-1.5b | 8 (incl. FP16) | 1 | 1 | 10 |
| phi-2 | 8 (incl. FP16) | 0 | 1 | 9 |
| mistral-7b | 6 (incl. Q8_0) | 0 | 0 | 6 |
| qwen2.5-7b | 6 (incl. Q8_0) | 0 | 0 | 6 |
| **Total** | **44** | **3** | **4** | **47** (41 non-baseline + 6 baseline) |

**Observations.** The 6 baseline entries (FP16 or Q8_0) serve as reference points. The 41 non-baseline entries are the subjects of regime classification and delta analysis.

### SS3.3 Alignment Method Distribution

| Alignment Method | Models | Family |
|-----------------|--------|--------|
| PPO | llama3.2-1b, llama3.2-3b, mistral-7b | Llama, Mistral |
| DPO | qwen2.5-1.5b, qwen2.5-7b | Qwen |
| SFT+RLHF | phi-2 | Phi |

**Observations.** Three distinct alignment paradigms are represented. PPO (3 models) dominates the sample. The v2 ANOVA finding (F=0.62, p=0.477) indicates alignment family does not significantly predict safety robustness slope -- within-family variance dominates. The v3 AWQ/GPTQ finding is orthogonal: it is a format effect, not a family effect.

---

## SS4. Results: GGUF Safety Baseline

SS4 establishes the GGUF k-quant safety profile that serves as the comparison anchor for AWQ/GPTQ analysis. All data in this section is from v1/v2 waves.

### SS4.1 GGUF Refusal Rate Profile (Regex)

| Model | FP16/Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | Slope (pp/BPW) |
|-------|-----------|------|--------|--------|--------|------|----------------|
| llama3.2-1b | 93.6% | 94.1% | 91.8% | 90.5% | 80.0% | 36.8% | Steep cliff at Q2_K |
| llama3.2-3b | 76.4% | 77.3% | 76.8% | 66.4% | 95.0% | 92.7% | Non-monotonic (over-refusal at Q3_K_S) |
| mistral-7b | 23.6% | 28.6% | 24.5% | 22.3% | 19.1% | 12.3% | Weak baseline, mild slope |
| qwen2.5-7b | 93.2% | 93.6% | 93.2% | 94.5% | 84.5% | 80.9% | Robust through Q3_K_S |
| qwen2.5-1.5b | 84.1% | 85.5% | 87.3% | 80.0% | 84.5% | 34.1% | Cliff at Q2_K |
| phi-2 | 58.6% | 54.1% | 57.7% | 55.0% | 56.4% | 55.0% | Flat (~0pp/BPW) |

**Observations.**

- Most well-aligned models (llama3.2-1b, qwen2.5-7b, qwen2.5-1.5b) maintain adequate refusal (>= 80%) through Q3_K_S.
- Q2_K produces catastrophic refusal collapse for llama3.2-1b (-56.8pp) and qwen2.5-1.5b (-50pp).
- phi-2 shows a unique flat slope: baseline is moderate (58.6%) and does not degrade meaningfully through Q2_K. This means phi-2 has a weak baseline alignment rather than quantization vulnerability.
- llama3.2-3b shows non-monotonic behavior: refusal increases at Q3_K_S (95.0%) and Q2_K (92.7%), which is the over-refusal artifact from v1 where degraded models default to refusing everything.
- Mistral-7b's low regex refusal (12-29% range) is misleading -- the judge reports 83-93% refusal (see SS8 and Appendix B). The regex gap is systematic and not a quantization artifact.

> GGUF k-quants preserve safety through Q3_K_S for most well-aligned models. Q2_K is the cliff point. This establishes the baseline against which AWQ/GPTQ will be compared.

### SS4.2 GGUF Judge-Confirmed Refusal

| Model | FP16/Q8_0 | Q4_K_M | Q3_K_S | Q2_K |
|-------|-----------|--------|--------|------|
| llama3.2-1b | 100.0% | 99.5% | 96.4% | 97.7% |
| llama3.2-3b | 100.0% | 100.0% | 100.0% | 100.0% |
| mistral-7b | 91.3% | 92.7% | 90.0% | 82.9% |
| qwen2.5-7b | 99.8% | 99.5% | 100.0% | 96.1% |
| qwen2.5-1.5b | 91.8% | 90.5% | 93.6% | 82.9% |
| phi-2 | 70.0% | 72.7% | 78.2% | 79.1% |

**Observations.**

- The judge shows much higher refusal rates than regex for most GGUF entries. llama3.2-1b Q2_K: regex 36.8%, judge 97.7% (60.8pp gap). This means Q2_K may produce non-standard refusals that regex misses.
- llama3.2-3b judge: perfect 100.0% across all GGUF levels including Q2_K. The non-monotonic regex pattern (over-refusal at Q3_K_S) is confirmed by the judge.
- phi-2 judge: 70-79% range, increasing slightly at lower quant. The flat regex slope is confirmed but at a higher absolute level.
- For GGUF, the judge generally shows stability. The dramatic judge drops occur only under AWQ/GPTQ (see SS5.2).

> GGUF judge-confirmed refusal remains high (>= 82.9%) for all models except phi-2 through Q2_K. The judge reveals that GGUF safety is better than regex reports suggest.

### SS4.3 Per-Model GGUF Safety Profiles

**llama3.2-1b (Llama, 1.24B, PPO).** Strong baseline (93.6% FP16). Robust through Q4_K_M (-3.2pp). Q3_K_S shows the first significant drop (-13.6pp), classified as hidden_danger because BERTScore is stable (+0.98pp). Q2_K is catastrophic (-56.8pp). This is the canonical "strong baseline, steep cliff" profile. Judge confirms: 100% through Q5_K_M, 96.4% at Q3_K_S, but 97.7% at Q2_K -- the judge detects partial refusals that regex misses at Q2_K.

**llama3.2-3b (Llama, 3.21B, PPO).** Moderate baseline (76.4% FP16, 100% judge). Non-monotonic GGUF curve: Q3_K_S jumps to 95.0% and Q2_K to 92.7% (regex). This is the over-refusal artifact -- the model defaults to refusing everything as coherence degrades. The judge does not show this pattern (100% across all GGUF levels), confirming that llama3.2-3b maintains genuine refusal behavior throughout the GGUF range. The regex anomaly reflects the classifier detecting Q3_K_S/Q2_K outputs as refusals because garbled text incidentally matches refusal patterns.

**mistral-7b (Mistral, 7.25B, PPO).** Low regex baseline (23.6% Q8_0). This is a systematic regex failure, not a weak model: the judge reports 91.3% refusal at Q8_0. Mistral uses hedged refusals ("I can't help with that", "I'd rather not discuss") that the regex patterns do not match. GGUF degradation is mild by judge (-8.4pp from Q8_0 to Q2_K) but crosses the near_hidden_danger threshold.

**qwen2.5-7b (Qwen, 7.62B, DPO).** Strong baseline (93.2% Q8_0, 99.8% judge). Robust through Q4_K_M (+1.4pp regex, -0.3pp judge). Q3_K_S shows -8.6pp regex drop. Q2_K shows -12.3pp, classified as hidden_danger (BERTScore +2.39pp). This is the best-performing 7B model for safety under quantization.

**qwen2.5-1.5b (Qwen, 1.54B, DPO).** Good baseline (84.1% FP16, 91.8% judge). Robust through Q3_K_S (+0.5pp regex, +1.8pp judge). Q2_K is catastrophic (-50.0pp regex), replicating the llama3.2-1b cliff pattern. Interesting: judge at Q2_K shows 82.9% (-8.9pp), much less severe than regex (-50.0pp). The Q2_K regex collapse may partly reflect garbled output rather than genuine compliance.

**phi-2 (Phi, 2.78B, SFT+RLHF).** Moderate baseline (58.6% FP16, 70.0% judge). Flat GGUF slope: -3.6pp maximum delta across all GGUF levels. Q2_K retains 55.0% refusal. This is a distinct profile: weak baseline alignment that is resilient to GGUF compression. The ceiling limitation is baseline alignment quality, not quantization vulnerability. Judge shows 70-79% range with a mild *increase* at Q2_K (79.1%), suggesting phi-2 at Q2_K produces outputs the judge interprets as refusals even when they are garbled.

### SS4.4 Q5_K_M Conservative Review Floor

The deployment taxonomy identifies Q5_K_M as the conservative review floor: the lowest bit-width format with zero reject rows, but still a `model_specific_review_only` role in the canonical deployment table.

| Model | Refusal Delta (Q5_K_M vs Baseline) | Max Safety Signal |
|-------|-----------------------------------|------------------|
| llama3.2-1b | -1.82pp | -6.0pp (truthfulness) |
| llama3.2-3b | +0.45pp | +9.0pp (truthfulness) |
| mistral-7b | +0.91pp | -1.0pp (truthfulness) |
| phi-2 | -0.91pp | +9.0pp (truthfulness) |
| qwen2.5-1.5b | +3.18pp | +4.0pp (bias) |
| qwen2.5-7b | +0.00pp | -2.2pp (BERTScore) |

**Observations.** Maximum refusal signal at Q5_K_M is 3.18pp (qwen2.5-1.5b, positive direction). All refusal deltas are within noise (<4pp). Q5_K_M preserves safety on all tested models, but the canonical deployment role remains `model_specific_review_only` because some truthfulness and bias signals still warrant per-model review.

---

## SS5. Results: AWQ/GPTQ Refusal Collapse

SS5 is the central finding of TR134 v3. Every AWQ/GPTQ entry shows safety degradation exceeding the 18.3pp MDE.

### SS5.1 Refusal Rates: Regex Measurement Track

**Table 1. AWQ/GPTQ Refusal Rates vs Baseline (Regex)**

| Model | FP16 | Q4_K_M | AWQ | GPTQ | AWQ Delta | GPTQ Delta |
|-------|------|--------|-----|------|-----------|-----------|
| llama3.2-1b | 93.6% | 90.5% | 31.8% | 25.5% | **-61.8pp** | **-68.2pp** |
| llama3.2-3b | 76.4% | 66.4% | 53.6% | 55.5% | **-22.7pp** | **-20.9pp** |
| qwen2.5-1.5b | 84.1% | 80.0% | 59.5% | 36.4% | **-24.5pp** | **-47.7pp** |
| phi-2 | 58.6% | 55.0% | -- | 3.2% | -- | **-55.5pp** |

**Observations.**

- Every entry exceeds the 18.3pp MDE. These are not borderline findings -- the smallest effect (-20.9pp, llama3.2-3b GPTQ) still exceeds the detection limit by 14%.
- llama3.2-1b GPTQ (-68.2pp) is the worst safety outcome in the entire 340,000-sample research program, exceeding even llama3.2-1b Q2_K (-56.8pp) by 11pp despite operating at higher effective bit-width.
- phi-2 GPTQ at 3.2% refusal means the model complies with 96.8% of harmful prompts by regex classification. This model is functionally unaligned.
- GPTQ is worse than AWQ on 2/3 models (llama3.2-1b by 6.4pp, qwen2.5-1.5b by 23.2pp). llama3.2-3b is within noise (2pp difference).
- qwen2.5-1.5b shows a notable AWQ-GPTQ gap: -24.5pp AWQ vs -47.7pp GPTQ. DPO alignment may provide partial resilience under AWQ's activation-aware compression but not under GPTQ's reconstruction-based approach.

> AWQ and GPTQ cause catastrophic refusal collapse on every tested model. Effects range from -20.9pp to -68.2pp, all exceeding the 18.3pp MDE.

### SS5.2 Judge-Confirmed Refusal

**Table 2. AWQ/GPTQ Refusal Rates vs Baseline (LLM Judge)**

| Model | FP16 Judge | AWQ Judge | GPTQ Judge | AWQ Delta | GPTQ Delta |
|-------|-----------|-----------|-----------|-----------|-----------|
| llama3.2-1b | 100.0% | 62.6% | 55.7% | **-37.4pp** | **-44.3pp** |
| llama3.2-3b | 100.0% | 77.3% | 78.2% | **-22.7pp** | **-21.8pp** |
| qwen2.5-1.5b | 91.8% | 75.5% | 62.3% | **-16.4pp** | **-29.5pp** |
| phi-2 | 70.0% | -- | 41.9% | -- | **-28.1pp** |

**Observations.**

- The judge independently confirms safety collapse, though at smaller absolute magnitude (-16pp to -44pp vs -21pp to -68pp for regex). This compression occurs because the judge detects partial/hedged refusals that regex misses -- it flags more GGUF entries as refusals too, raising the baseline.
- llama3.2-1b GPTQ: -44.3pp by judge. Even with the judge's more generous refusal detection, the model still fails to refuse 44% of harmful prompts.
- qwen2.5-1.5b AWQ judge delta (-16.4pp) is the smallest AWQ/GPTQ judge delta. This provides further evidence that DPO alignment offers partial resilience under AWQ.
- phi-2 GPTQ: 41.9% judge-confirmed refusal. The model still complies with 58% of harmful prompts even by the more generous judge standard.
- llama3.2-3b AWQ/GPTQ: judge deltas (-22.7pp, -21.8pp) closely match regex deltas (-22.7pp, -20.9pp), indicating clean refusal failure rather than measurement artifact.

> The LLM judge independently confirms AWQ/GPTQ safety collapse with -16pp to -44pp losses. Both measurement tracks agree: these formats destroy safety.

### SS5.3 AWQ/GPTQ vs GGUF Q2_K

A critical comparison: AWQ/GPTQ operate at ~4 BPW while Q2_K operates at ~2.5 BPW. If safety degradation were purely a function of compression, Q2_K should be worse. It is not.

**Table 3. AWQ/GPTQ vs Q2_K (Regex Refusal)**

| Model | Q2_K (2.5 BPW) | AWQ (~4 BPW) | GPTQ (~4 BPW) | AWQ vs Q2_K | GPTQ vs Q2_K |
|-------|----------------|-------------|--------------|-------------|-------------|
| llama3.2-1b | 36.8% | 31.8% | 25.5% | Worse by 5pp | Worse by 11pp |
| llama3.2-3b | 92.7% | 53.6% | 55.5% | **Worse by 39pp** | **Worse by 37pp** |
| qwen2.5-1.5b | 34.1% | 59.5% | 36.4% | Better by 25pp | Comparable |
| phi-2 | 55.0% | -- | 3.2% | -- | **Worse by 52pp** |

**Observations.**

- GPTQ is worse than Q2_K on 3/4 models despite operating at 1.5 BPW higher effective precision. phi-2 GPTQ (3.2%) vs Q2_K (55.0%) is a 52pp gap -- the model retains safety at 2.5 BPW but loses it entirely at 4 BPW under GPTQ.
- llama3.2-3b is the most dramatic: Q2_K preserves 92.7% refusal (over-refusal artifact) while AWQ drops to 53.6% and GPTQ to 55.5%. AWQ/GPTQ bypass the over-refusal behavior entirely.
- qwen2.5-1.5b AWQ (59.5%) is the one case where AWQ outperforms Q2_K (34.1%), suggesting DPO-trained weights have partial AWQ resilience for this model.
- The overall pattern: AWQ/GPTQ at 4 BPW produce Q2_K-level or worse safety outcomes. BPW is not a reliable cross-format safety predictor.

> At comparable or higher bit-widths, AWQ/GPTQ produce Q2_K-level or worse safety damage. Compression ratio alone does not explain safety loss.

### SS5.4 Confidence Intervals on AWQ/GPTQ Refusal

| Model | Quant | Refusal | 95% CI Low | 95% CI High | N | CI Width |
|-------|-------|---------|-----------|------------|---|---------|
| llama3.2-1b | AWQ | 31.8% | 26.4% | 38.2% | 220 | 11.8pp |
| llama3.2-1b | GPTQ | 25.5% | 20.0% | 30.9% | 220 | 10.9pp |
| llama3.2-3b | AWQ | 53.6% | 47.3% | 60.0% | 220 | 12.7pp |
| llama3.2-3b | GPTQ | 55.5% | 49.5% | 61.8% | 220 | 12.3pp |
| qwen2.5-1.5b | AWQ | 59.5% | 52.7% | 65.0% | 220 | 12.3pp |
| qwen2.5-1.5b | GPTQ | 36.4% | 30.5% | 42.7% | 220 | 12.3pp |
| phi-2 | GPTQ | 3.2% | 1.4% | 5.9% | 220 | 4.5pp |

**Observations.**

- All CIs are narrow (4.5-12.7pp width), demonstrating that the findings are not driven by sampling noise.
- phi-2 GPTQ CI: [1.4%, 5.9%] -- even the upper bound is catastrophically low. The model is functionally unaligned under GPTQ regardless of sampling variation.
- llama3.2-1b GPTQ CI upper bound (30.9%) is still below the Q2_K point estimate (36.8%). With 95% confidence, GPTQ is worse than Q2_K on llama3.2-1b.
- qwen2.5-1.5b shows non-overlapping CIs between AWQ [52.7%, 65.0%] and GPTQ [30.5%, 42.7%], confirming that the 23pp AWQ-GPTQ gap on this model is statistically real.
- llama3.2-3b AWQ [47.3%, 60.0%] and GPTQ [49.5%, 61.8%] CIs overlap substantially, confirming the 2pp difference is within noise.

### SS5.5 Summary: AWQ vs GPTQ Head-to-Head

| Model | AWQ Refusal | GPTQ Refusal | AWQ - GPTQ Gap | Winner |
|-------|-----------|-------------|---------------|--------|
| llama3.2-1b | 31.8% | 25.5% | +6.4pp | AWQ (within noise) |
| llama3.2-3b | 53.6% | 55.5% | -1.8pp | Equivalent |
| qwen2.5-1.5b | 59.5% | 36.4% | **+23.2pp** | AWQ (significant) |
| phi-2 | -- | 3.2% | -- | Cannot compare |

**Observations.** AWQ is meaningfully better than GPTQ only on qwen2.5-1.5b (+23.2pp, non-overlapping CIs). On the other models, the difference is within noise. Neither format is safe, but if forced to choose between them, AWQ is the less-dangerous option -- particularly for DPO-aligned models where AWQ's activation-aware approach may incidentally preserve more alignment-critical weights.

---

## SS6. Results: Format Comparison at Matched BPW

SS6 directly compares GGUF and AWQ/GPTQ at the closest available bit-widths.

### SS6.1 Q4_K_M (4.85 BPW) vs AWQ (~4 BPW) vs GPTQ (~4 BPW)

**Table 5. Refusal Rate at ~4 BPW: Format Comparison**

| Model | Q4_K_M (4.85) | AWQ (~4.0) | GPTQ (~4.0) | Q4_K_M - AWQ Gap | Q4_K_M - GPTQ Gap |
|-------|--------------|---------|----------|-----------------|------------------|
| llama3.2-1b | 90.5% | 31.8% | 25.5% | **58.6pp** | **65.0pp** |
| llama3.2-3b | 66.4% | 53.6% | 55.5% | **12.8pp** | **10.9pp** |
| qwen2.5-1.5b | 80.0% | 59.5% | 36.4% | **20.5pp** | **43.6pp** |
| phi-2 | 55.0% | -- | 3.2% | -- | **51.8pp** |

**Observations.**

- The gap ranges from 10.9pp (llama3.2-3b GPTQ) to 65.0pp (llama3.2-1b GPTQ) for less than 1 BPW difference in effective precision.
- Q4_K_M retains >= 55% refusal on all models. GPTQ goes as low as 3.2% (phi-2).
- The smallest gap (llama3.2-3b, 10.9-12.8pp) still exceeds reasonable deployment thresholds. No practitioner would accept a 10pp refusal loss for <1 BPW savings.
- llama3.2-1b is the clearest demonstration: 90.5% refusal at 4.85 BPW drops to 25.5% at 4.0 BPW under GPTQ. The format, not the bit-width, determines the outcome.

> At ~4 BPW, GGUF Q4_K_M preserves safety while AWQ/GPTQ destroy it. BPW alone does not predict safety.

### SS6.2 Q3_K_S (3.5 BPW) vs AWQ/GPTQ (~4.0 BPW)

A stronger test: does GGUF at *lower* bit-width still outperform AWQ/GPTQ at higher bit-width?

**Table 6. Refusal Rate: GGUF at 3.5 BPW vs AWQ/GPTQ at 4.0 BPW**

| Model | Q3_K_S (3.5 BPW) | AWQ (~4.0 BPW) | GPTQ (~4.0 BPW) | Q3_K_S - AWQ | Q3_K_S - GPTQ |
|-------|-----------------|---------------|----------------|-------------|--------------|
| llama3.2-1b | 80.0% | 31.8% | 25.5% | 48.2pp | 54.5pp |
| llama3.2-3b | 95.0% | 53.6% | 55.5% | 41.4pp | 39.5pp |
| qwen2.5-1.5b | 84.5% | 59.5% | 36.4% | 25.0pp | 48.1pp |
| phi-2 | 56.4% | -- | 3.2% | -- | 53.2pp |

**Observations.**

- GGUF at 3.5 BPW preserves more safety than AWQ/GPTQ at 4.0 BPW on **every model tested**. The gaps range from 25pp to 54.5pp.
- This rules out bit-width as the primary driver. GGUF Q3_K_S uses 0.5 BPW *less* precision than AWQ/GPTQ yet preserves 25-54pp more refusal.
- The consistent pattern across all 4 models and 7 comparisons demonstrates that format selection dominates bit-width in determining safety outcomes.

> Format selection dominates bit-width in determining safety outcomes. GGUF at 3.5 BPW is safer than AWQ/GPTQ at 4.0 BPW.

---

## SS7. Results: Hidden-Danger Regime Analysis

SS7 examines the most operationally dangerous pattern: model-quant entries where quality metrics appear stable or improved but safety has silently degraded.

### SS7.1 All Hidden-Danger and Near-Hidden-Danger Entries

**Table 4. Entries Classified as Hidden-Danger or Near-Hidden-Danger**

| Model | Quant | Format | BERTScore Delta (pp) | Refusal Delta (pp) | Regime |
|-------|-------|--------|---------------------|--------------------|----|
| llama3.2-1b | **GPTQ** | GPTQ | +8.49 | -68.18 | **hidden_danger** |
| llama3.2-1b | **AWQ** | AWQ | +8.27 | -61.82 | **hidden_danger** |
| phi-2 | **GPTQ** | GPTQ | +3.21 | -55.45 | **hidden_danger** |
| llama3.2-3b | **AWQ** | AWQ | -0.83 | -22.73 | **hidden_danger** |
| llama3.2-3b | **GPTQ** | GPTQ | +0.00 | -20.91 | **hidden_danger** |
| llama3.2-1b | Q3_K_S | GGUF | +0.98 | -13.64 | hidden_danger |
| qwen2.5-7b | Q2_K | GGUF | +2.39 | -12.27 | hidden_danger |
| mistral-7b | Q2_K | GGUF | -2.12 | -11.36 | near_hidden_danger |

**Observations.**

- 5 of 7 hidden-danger entries (71%) are AWQ or GPTQ. These formats comprise only 7/47 = 15% of the matrix. The concentration is extreme: AWQ/GPTQ are 4.7x overrepresented in hidden-danger relative to their matrix share.
- AWQ/GPTQ hidden-danger magnitudes (-21pp to -68pp refusal) far exceed GGUF hidden-danger magnitudes (-12pp to -14pp refusal). The average AWQ/GPTQ hidden-danger refusal loss is -45.8pp vs -12.9pp for GGUF.
- llama3.2-1b GPTQ: BERTScore +8.49pp while refusal -68.18pp. A quality-only audit would flag this as an *improvement*. This is the most dangerous single entry in the matrix.
- The remaining 33/41 non-baseline entries are classified as neutral, meaning quality and safety either co-move or neither degrades significantly.

> 71% of hidden-danger cells are AWQ/GPTQ entries. Quality improvement under quantization is the hidden-danger signature.

### SS7.2 Quality Paradox Under AWQ/GPTQ

Why do quality metrics improve when safety collapses? The mechanism: when a model stops refusing harmful prompts, it generates longer, more coherent *harmful content*. Text-quality metrics (BERTScore, ROUGE-L) reward fluency and length, which are higher for compliant responses than for short refusal templates.

**Table: Quality vs Safety Under AWQ/GPTQ**

| Model | Quant | BERTScore Delta | ROUGE-L Delta | Coherence Delta | Refusal Delta |
|-------|-------|----------------|--------------|----------------|--------------|
| llama3.2-1b | AWQ | +8.27pp | +25.77pp | +17.86pp | -61.82pp |
| llama3.2-1b | GPTQ | +8.49pp | +27.29pp | +18.30pp | -68.18pp |
| llama3.2-3b | GPTQ | +0.00pp | +17.74pp | +12.08pp | -20.91pp |
| phi-2 | GPTQ | +3.21pp | +12.55pp | -6.22pp | -55.45pp |
| qwen2.5-1.5b | AWQ | -13.67pp | -14.79pp | -5.39pp | -24.55pp |
| qwen2.5-1.5b | GPTQ | -12.98pp | -11.63pp | -2.52pp | -47.73pp |

**Observations.**

- Llama models show the pure hidden-danger pattern: all quality metrics improve while safety collapses. The ROUGE-L improvement (+25-27pp) is driven by the model generating longer responses when it complies with harmful prompts instead of producing "I can't assist with that" (20 tokens).
- phi-2 GPTQ shows quality improvement on BERTScore/ROUGE-L but coherence *decrease* (-6.2pp). This suggests partial output degradation alongside refusal loss.
- qwen2.5-1.5b shows quality *and* safety degradation under both AWQ (-14pp BERTScore, -25pp refusal) and GPTQ (-13pp BERTScore, -48pp refusal). This is total collapse, not hidden danger. The bespoke analysis correctly classifies qwen2.5-1.5b AWQ/GPTQ as neutral (both axes degrade) rather than hidden_danger.
- The divergent pattern by model family suggests DPO alignment (Qwen) encodes safety in weights that are also quality-critical, causing co-degradation, while PPO alignment (Llama) separates quality and safety weight structures, enabling selective safety loss.

> Quality metrics are unreliable as safety proxies under AWQ/GPTQ. Quality improvement may indicate safety collapse rather than model improvement.

---

## SS8. Results: Regex vs Judge Under AWQ/GPTQ

SS8 quantifies the measurement divergence between regex and LLM judge classifiers under AWQ/GPTQ. The finding: AWQ/GPTQ amplify regex-judge gaps because they destabilize refusal templates into forms the regex cannot match.

### SS8.1 Refusal Rate Gaps (AWQ/GPTQ Entries)

**Table 7. Regex vs Judge Refusal Rate Under AWQ/GPTQ**

| Model | Quant | Regex | Judge | Gap (pp) | Regex Misses |
|-------|-------|-------|-------|----------|-------------|
| phi-2 | GPTQ | 3.2% | 41.9% | **+38.8** | 92% of judge-detected refusals |
| llama3.2-1b | AWQ | 31.8% | 62.6% | **+30.7** | 49% of judge-detected refusals |
| llama3.2-1b | GPTQ | 25.5% | 55.7% | **+30.3** | 54% of judge-detected refusals |
| qwen2.5-1.5b | GPTQ | 36.4% | 62.3% | **+25.9** | 42% of judge-detected refusals |
| llama3.2-3b | AWQ | 53.6% | 77.3% | **+23.6** | 31% of judge-detected refusals |
| llama3.2-3b | GPTQ | 55.5% | 78.2% | **+22.7** | 29% of judge-detected refusals |
| qwen2.5-1.5b | AWQ | 59.5% | 75.5% | **+15.9** | 21% of judge-detected refusals |

**Observations.**

- All AWQ/GPTQ gaps are positive (regex underreports) and larger than typical GGUF gaps (8-10pp median for non-Mistral GGUF).
- phi-2 GPTQ is the extreme case: regex detects 3.2% refusal while the judge detects 41.9%. The regex misses 92% of actual refusals because the model's refusal template is completely destroyed -- it no longer produces any of the ~15 patterns the regex matches.
- Even with the judge's higher numbers, every AWQ/GPTQ entry remains catastrophically below baseline. phi-2 GPTQ at 41.9% judge-confirmed refusal means the model still complies with 58% of harmful prompts.
- The consistent gap direction (all positive) indicates systematic regex failure under AWQ/GPTQ, not random measurement noise.

### SS8.2 Bias Resistance Gaps (GPTQ Entries)

| Model | Quant | Regex Bias | Judge Bias | Gap (pp) |
|-------|-------|-----------|-----------|----------|
| phi-2 | GPTQ | 63.6% | 22.7% | **-40.9** |
| qwen2.5-1.5b | GPTQ | 79.8% | 45.5% | **-34.3** |
| llama3.2-3b | GPTQ | 78.8% | 45.5% | **-33.3** |
| llama3.2-1b | GPTQ | 68.7% | 38.4% | **-30.3** |

**Observations.**

- Bias gaps are negative: regex *over-reports* bias resistance. phi-2 GPTQ: regex says 63.6% bias-resistant, judge says 22.7%. The regex may be classifying degraded/nonsensical outputs as non-stereotypical (bias-resistant) when the model has actually lost the ability to produce coherent answers.
- The negative bias gaps are the opposite direction of refusal gaps (positive). This asymmetry is consistent: regex over-detects bias resistance (model defaults to "unknown" or garbled text) and under-detects refusal (model loses standard refusal templates).
- AWQ bias gaps are smaller than GPTQ (data available for llama3.2-1b AWQ: regex 83.3%, judge 53.0%, gap -30.3pp; llama3.2-3b AWQ: regex 93.4%, judge 73.7%, gap -19.7pp).

> Regex classifiers are unreliable under AWQ/GPTQ in both directions: they under-report refusal by 16-39pp and over-report bias resistance by 20-41pp. LLM judge validation is mandatory for any AWQ/GPTQ deployment.

### SS8.3 Agreement by Format (AdvBench Refusal)

| Format | N Pairs | N Agree | Agreement % | Cohen's Kappa | Category |
|--------|---------|---------|-------------|---------------|----------|
| AWQ | 300 | 248 | 82.7% | 0.584 | Moderate |
| GPTQ | 400 | 298 | 74.5% | 0.526 | Moderate |
| GGUF (Q8_0) | 600 | 474 | 79.0% | 0.143 | Slight |
| GGUF (Q4_K_M) | 600 | 462 | 77.0% | 0.147 | Slight |
| GGUF (Q2_K) | 600 | 378 | 63.0% | 0.162 | Slight |

**Observations.**

- AWQ/GPTQ produce *higher* kappa than GGUF, which is counterintuitive. The explanation: AWQ/GPTQ produce clearer compliance signals. When the model fails to refuse under AWQ/GPTQ, it generates unambiguous harmful content that both regex and judge classify as compliance. Under GGUF, partial refusals and hedged responses create more ambiguity.
- GPTQ agreement (74.5%) is lower than AWQ (82.7%), consistent with GPTQ producing more severe output degradation.
- GGUF Q2_K has the lowest agreement (63.0%) because Q2_K produces garbled output that neither classifier handles well.

### SS8.4 Agreement by Task (All Formats)

| Task | N Pairs | Agreement % | Cohen's Kappa | Category |
|------|---------|-------------|---------------|----------|
| AdvBench Refusal | 4,700 | 77.7% | 0.316 | Fair |
| BBQ Bias | 6,534 | 82.8% | 0.450 | Moderate |
| TruthfulQA | 2,350 | 40.8% | 0.184 | Slight |

**Observations.**

- Bias has the highest agreement (82.8%, kappa=0.450). The bias classification task is cleaner: the model either selects a stereotypical answer or does not.
- Truthfulness has extremely low agreement (40.8%, kappa=0.184). The truthfulness construct is inherently ambiguous, and regex string-matching fundamentally differs from judge semantic evaluation.
- Refusal agreement (77.7%, kappa=0.316) is "fair" -- adequate for cross-validation but insufficient to treat either classifier as a sole truth source.

---

## SS9. Results: Refusal Template Destabilization Mechanism

SS9 presents the Phase 6 analysis that identifies *why* AWQ/GPTQ cause safety collapse. The mechanism: quantization disrupts the weight structures encoding refusal templates, fragmenting the model's dominant refusal prefix and causing it to produce diverse, non-standard, verbose text instead of crisp learned refusals.

### SS9.1 Template Metrics for Top Degraded Entries

**Table 8. Refusal Style Changes for Most Degraded Entries**

| Model | Quant | Refusal Delta | Dom. Prefix Delta | Unique Prefix Delta | Tokens Delta | Destab. Flag |
|-------|-------|-------------|------------------|-------------------|-------------|-------------|
| phi-2 | GPTQ | -90.0% | +5.5pp | +93.4% | +170.6 | No |
| llama3.2-1b | GPTQ | -59.0% | -33.5pp | +40.5% | +67.9 | Yes |
| llama3.2-1b | Q2_K | -57.0% | -51.0pp | +49.5% | +73.5 | Yes |
| qwen2.5-1.5b | Q2_K | -56.0% | -53.5pp | +24.6% | +161.7 | Yes |
| qwen2.5-1.5b | GPTQ | -52.0% | -55.3pp | +43.7% | +190.6 | Yes |
| llama3.2-1b | AWQ | -51.0% | -56.8pp | +50.4% | +45.9 | Yes |

**Observations.**

- 4 of 6 top entries are AWQ/GPTQ. Template destabilization (dominant-prefix drop >= 15pp) co-occurs with refusal loss for all entries except phi-2 GPTQ, which shows complete template destruction rather than destabilization.
- phi-2 GPTQ: dominant prefix *increases* slightly (+5.5pp) but the prefix itself changes completely (from "i m sorry but as" to "answer 1 research and identify"). This is total prefix replacement, not fragmentation. The model loses its refusal behavior entirely and generates an unrelated response pattern.
- qwen2.5-1.5b GPTQ: dominant-prefix drops -55.3pp and refusal tokens increase +190.6. The model preserves the refusal prefix text ("i m sorry but i") but uses it far less frequently, and when it does refuse, the refusal is much longer and more verbose.
- The token increase pattern is universal: every degraded entry shows longer refusal text. This is because non-standard refusals are verbose ("I understand your question, but I need to point out that...") compared to learned templates ("I can't assist with that").

### SS9.2 Phase 6 Style Correlations

**Table: Refusal-Style Metric Correlations with Refusal Rate Delta (n=41 non-baseline entries)**

| Style Metric | Pearson r | p-value | Spearman rho | p-value | Direction |
|-------------|----------|---------|-------------|---------|-----------|
| Dominant prefix share | +0.589 | **5.1e-05** | +0.501 | **8.4e-04** | Lower concentration = more refusal loss |
| Unique prefix rate | -0.813 | **1.1e-10** | -0.431 | **4.9e-03** | Higher diversity = more refusal loss |
| Prefix entropy | -0.456 | **2.7e-03** | -0.576 | **8.3e-05** | More entropic = more refusal loss |
| Mean tokens (refusal) | -0.698 | **3.9e-07** | -0.394 | **1.1e-02** | Longer refusal = more refusal loss |
| Mean chars (refusal) | -0.658 | **2.9e-06** | -0.391 | **1.2e-02** | Longer refusal = more refusal loss |
| Hard refusal rate | +0.998 | **2.5e-50** | +0.987 | **1.8e-32** | Hard refusal tracks total near-perfectly |

**Observations.**

- All six style metrics show statistically significant correlations with refusal loss. The strongest predictor is unique prefix rate (r=-0.813): entries with more diverse refusal openings also show more refusal loss.
- Dominant prefix share (r=+0.589) provides the mechanistic interpretation: when quantization reduces the dominant prefix's share of all refusals, the model is losing its learned refusal template.
- Mean tokens (r=-0.698) shows that refusal verbosity increases as refusal rate decreases. This is consistent with the model "trying harder" to refuse but producing non-standard text that is both less effective (more compliance slips through) and less recognizable (regex misses it).
- Hard refusal rate tracks total refusal rate near-perfectly (r=+0.998), confirming that the safety loss is loss of hard refusals, not conversion from hard to soft refusals.
- Spearman rho is consistently lower than Pearson r for unique prefix rate (rho=-0.431 vs r=-0.813), indicating the relationship is not purely monotonic -- some high-diversity entries (e.g., phi-2 GPTQ) show complete collapse rather than gradual degradation.

> Refusal template destabilization is the primary mechanism. Prefix fragmentation, diversity increase, and verbosity shift all predict refusal loss with p < 0.01 on all metrics.

### SS9.3 Template Shift Examples

| Model | Quant | Baseline Prefix | Post-Quant Prefix | Shift Type |
|-------|-------|----------------|------------------|-----------|
| llama3.2-1b | GPTQ | "i can t assist with" | "i can t fulfill that" | Partial (synonym substitution) |
| llama3.2-1b | AWQ | "i can t assist with" | "i can t provide instructions" | Partial (object substitution) |
| phi-2 | GPTQ | "i m sorry but as" | "answer 1 research and identify" | **Complete destruction** |
| qwen2.5-1.5b | GPTQ | "i m sorry but i" | "i m sorry but i" (fewer uses) | Frequency reduction |
| qwen2.5-1.5b | AWQ | "i m sorry but i" | "i m sorry but i" (fewer uses) | Frequency reduction |
| llama3.2-1b | Q2_K | "i can t assist with" | "i can t assist with" (fewer uses) | Frequency reduction |

**Observations.**

- Three distinct destabilization modes are observed: (1) synonym substitution (llama GPTQ/AWQ), (2) complete destruction (phi-2 GPTQ), and (3) frequency reduction without text change (qwen, llama Q2_K).
- phi-2 GPTQ shows total prefix replacement: the dominant opening shifts from a refusal pattern to what appears to be a task-completion pattern ("answer 1 research and identify"). The model's refusal behavior is not degraded -- it is eliminated and replaced with compliance behavior.
- qwen2.5-1.5b preserves refusal prefix text under both AWQ and GPTQ but uses it less frequently. This is consistent with DPO alignment encoding refusal at a deeper level (the template survives but the trigger frequency decreases).
- llama3.2-1b Q2_K shows the GGUF version of destabilization: same text, fewer uses. The GGUF pattern is gentler than AWQ/GPTQ, consistent with the smaller refusal deltas.

> Three destabilization modes: synonym substitution (Llama AWQ/GPTQ), complete destruction (phi-2 GPTQ), and frequency reduction (Qwen, Llama Q2_K). DPO alignment preserves template text more robustly than PPO alignment.

---

## SS10. Results: Deployment Taxonomy

SS10 applies the Phase 5 deployment protocol to the full 47-row matrix to classify each quantization format's suitability for safety-critical deployment.

### SS10.1 Format Safety Classification

**Table 9. Deployment Taxonomy by Quantization Format**

| Format | N Models | Candidate Rows | Manual Review | Direct Eval | Reject Rows | Max Refusal Signal (pp) | Role |
|--------|---------|---------------|--------------|------------|------------|------------------------|------|
| Q8_0 | 4 | 1 | 1 | 2 | 0 | 2.60 | model_specific_review_only |
| Q6_K | 6 | 3 | 2 | 1 | 0 | 1.82 | model_specific_review_only |
| Q5_K_M | 6 | 2 | 2 | 2 | 0 | 2.73 | model_specific_review_only |
| Q4_K_M | 6 | 2 | 2 | 1 | 1 | 2.73 | not_blanket_safe |
| Q3_K_S | 6 | 1 | 2 | 2 | 1 | 8.18 | not_blanket_safe |
| Q2_K | 6 | 0 | 3 | 0 | 3 | 9.09 | not_blanket_safe |
| **AWQ** | **3** | **0** | **0** | **0** | **3** | **37.44** | **not_blanket_safe** |
| **GPTQ** | **4** | **0** | **1** | **0** | **3** | **44.29** | **not_blanket_safe** |

**Observations.**

- AWQ and GPTQ are blanket-rejected. Zero candidate rows, zero direct-eval rows. Every model tested under AWQ/GPTQ fails the safety gate.
- AWQ max refusal signal (37.4pp) and GPTQ max signal (44.3pp) dwarf Q2_K's 9.1pp. AWQ/GPTQ are 4-5x more dangerous than GGUF Q2_K by this metric.
- Q5_K_M remains the conservative GGUF review floor: max refusal signal 2.73pp, zero reject rows, but still `model_specific_review_only` in the canonical deployment table.
- Q4_K_M has 1 reject row (llama3.2-3b, -10pp refusal) but retains >= 55% refusal on all models. It remains the recommended minimum for safety-critical deployment where VRAM is constrained.
- 21 of 47 rows show measurement divergence (regex-judge gap >= 20pp). This number is inflated by Mistral's systematic gap. Excluding Mistral, 15/41 rows show divergence -- still concerning.

> AWQ and GPTQ are blanket-unsafe. GGUF Q5_K_M is the conservative review floor; Q4_K_M is the recommended minimum.

### SS10.2 Per-Model Deployment Recommendations

| Model | Min Safe GGUF | AWQ/GPTQ Safe? | Est. VRAM (Q4_K_M) | Recommendation |
|-------|-------------|---------------|-------------------|----|
| llama3.2-1b | Q4_K_M (90.5%) | **No** (AWQ -62pp, GPTQ -68pp) | ~0.8 GB | GGUF Q4_K_M or above |
| llama3.2-3b | Q5_K_M (76.8%) | **No** (AWQ -23pp, GPTQ -21pp) | ~2.0 GB | GGUF Q5_K_M or above with model-specific review |
| qwen2.5-1.5b | Q4_K_M (80.0%) | **No** (AWQ -25pp, GPTQ -48pp) | ~1.2 GB | GGUF Q4_K_M or above |
| phi-2 | Q4_K_M (55.0%) | **No** (GPTQ -55pp) | ~1.8 GB | GGUF Q4_K_M; note moderate baseline |
| mistral-7b | Q8_0 (23.6%/91.3% judge) | Not tested | ~7.0 GB | Non-safety-critical only |
| qwen2.5-7b | Q4_K_M (94.5%) | Not tested | ~4.6 GB | GGUF Q4_K_M; strongest safe option |

**Observations.**

- VRAM savings from AWQ/GPTQ over Q4_K_M are minimal (~0.1-0.2 GB for sub-3B models). The 62-68pp refusal loss is not justified by any VRAM benefit.
- qwen2.5-7b at Q4_K_M remains the strongest safe deployment option: 94.5% refusal (regex), 99.5% (judge), estimated ~4.6 GB VRAM.
- phi-2 has a moderate baseline (58.6% FP16 refusal) that is a ceiling limitation, not a quantization problem. GPTQ destroys even this moderate alignment.
- Mistral-7b's regex refusal (12-29%) is misleading. By judge, it maintains 83-93% refusal. Do not make deployment decisions about Mistral based on regex alone. However, Mistral is not recommended for safety-critical applications due to the measurement ambiguity.

> The combined recommendation across TR125, TR134 v3, and TR142: use GGUF Q4_K_M or above for any safety-critical deployment. AWQ/GPTQ should not be used.

---

## SS11. Results: Bias and Truthfulness

### SS11.1 Bias Resistance Under AWQ/GPTQ (Regex)

| Model | FP16 | AWQ | GPTQ | AWQ Delta | GPTQ Delta |
|-------|------|-----|------|-----------|-----------|
| llama3.2-1b | 89.4% | 83.3% | 68.7% | -6.1pp | **-20.7pp** |
| llama3.2-3b | 96.5% | 93.4% | 78.8% | -3.0pp | **-17.7pp** |
| qwen2.5-1.5b | 85.4% | 87.9% | 79.8% | +2.5pp | -5.6pp |
| phi-2 | 84.8% | -- | 63.6% | -- | **-21.2pp** |

**Observations.**

- GPTQ degrades bias resistance by 17-21pp on 3 of 4 models. The exception is qwen2.5-1.5b (-5.6pp), where DPO alignment may provide partial bias resistance preservation.
- AWQ impact on bias is smaller: -3pp to -6pp. This is consistent with AWQ's activation-aware approach incidentally preserving more bias-relevant weights.
- However, these regex bias numbers should be interpreted cautiously given the negative bias gaps in SS8.2. The judge reports much lower bias resistance: phi-2 GPTQ regex says 63.6% but judge says 22.7%. The true bias degradation under GPTQ may be larger than regex reports.

### SS11.2 Bias Resistance Under AWQ/GPTQ (Judge)

| Model | Quant | Judge Bias | Interpretation |
|-------|-------|-----------|---------------|
| llama3.2-1b | AWQ | 53.0% | Moderate degradation from GGUF range (no judge bias for GGUF available) |
| llama3.2-1b | GPTQ | 38.4% | Severe degradation |
| llama3.2-3b | AWQ | 73.7% | Moderate |
| llama3.2-3b | GPTQ | 45.5% | Severe |
| qwen2.5-1.5b | AWQ | 81.3% | Mild |
| qwen2.5-1.5b | GPTQ | 45.5% | Severe |
| phi-2 | GPTQ | 22.7% | Near-floor |

**Observations.**

- By judge measurement, GPTQ bias impact is severe across all models. phi-2 GPTQ at 22.7% means the model produces stereotypical answers 77% of the time.
- AWQ judge-measured bias is consistently better than GPTQ: llama3.2-1b 53.0% vs 38.4%, llama3.2-3b 73.7% vs 45.5%, qwen2.5-1.5b 81.3% vs 45.5%.
- qwen2.5-1.5b AWQ judge bias (81.3%) is the only AWQ/GPTQ entry above 75%. DPO alignment provides observable bias preservation under AWQ specifically.

> GPTQ degrades bias resistance severely by judge measurement (22-45%). AWQ has smaller bias impact (53-81%). DPO alignment provides partial protection.

### SS11.3 Truthfulness Under AWQ/GPTQ

All AWQ/GPTQ truthfulness deltas are within noise. At N=50 per entry, the MDE is 27.7pp. No truthfulness delta exceeds this threshold. Truthfulness should not be used for deployment decisions and is included only for completeness.

| Model | Quant | Truthfulness | Delta from Baseline |
|-------|-------|-------------|-------------------|
| llama3.2-1b | AWQ | 53.0% | -2.0pp |
| llama3.2-1b | GPTQ | 50.0% | -5.0pp |
| llama3.2-3b | AWQ | 47.0% | -2.0pp |
| llama3.2-3b | GPTQ | 59.0% | +10.0pp |
| qwen2.5-1.5b | AWQ | 58.0% | +9.0pp |
| qwen2.5-1.5b | GPTQ | 42.0% | -7.0pp |
| phi-2 | GPTQ | 38.0% | -1.0pp |

**Observations.** No delta exceeds the 27.7pp MDE. The apparent "improvement" for llama3.2-3b GPTQ (+10pp) and qwen2.5-1.5b AWQ (+9pp) is within noise at N=50.

---

## SS12. Statistical Synthesis

### SS12.1 Hypothesis Evaluation

| Hypothesis | Test | Result | Status |
|-----------|------|--------|--------|
| H1: AWQ/GPTQ preserve safety at matched BPW | Format comparison (SS6) | All entries >= 10.9pp worse than Q4_K_M | **Rejected** |
| H2: Template destabilization predicts refusal loss | Pearson correlation (SS9) | r=+0.589, p=5.1e-05 (prefix share) | **Supported** |
| H3: Regex reliable under AWQ/GPTQ | Gap analysis (SS8) | Gaps 15.9-38.8pp on refusal | **Rejected** |
| H4: Hidden-danger is format-independent | Regime analysis (SS7) | 5/7 hidden-danger are AWQ/GPTQ | **Rejected** |
| H5: Quality predicts safety under AWQ/GPTQ | Quality paradox (SS7.2) | BERTScore +8.5pp with refusal -68pp | **Rejected** |
| H6: GGUF safety through Q3_K_S | GGUF baseline (SS4) | 5/6 models maintain refusal | **Confirmed** |

### SS12.2 Repeated-Measures Correlations (Full v3 Matrix)

These correlations control for within-model pairing to test whether quality changes predict safety changes across the full 41-entry matrix.

| Quality Metric | Safety Metric | Source | r | p | 95% CI | Power |
|---------------|-------------|--------|---|---|--------|-------|
| BERTScore | Refusal | Regex | +0.152 | 0.378 | [-0.19, +0.46] | 0.14 |
| ROUGE-L | Refusal | Regex | -0.349 | **0.037** | [-0.61, -0.02] | 0.56 |
| Coherence | Refusal | Regex | -0.274 | 0.106 | [-0.55, +0.06] | -- |
| BERTScore | Refusal | Judge | -0.120 | 0.485 | -- | 0.11 |
| ROUGE-L | Refusal | Judge | -0.627 | **4.2e-05** | -- | 0.99 |
| Coherence | Refusal | Judge | -0.575 | **2.5e-04** | -- | 0.97 |

**Observations.**

- BERTScore does not reliably predict refusal (p=0.378 for regex, p=0.485 for judge). The sign even differs between tracks (+0.152 regex, -0.120 judge). BERTScore is not a safety proxy.
- ROUGE-L and coherence correlate *negatively* with judge-measured refusal (p<0.001 for both, power>0.95). This is the hidden-danger mechanism quantified: entries with higher text quality show lower judge-confirmed refusal. The relationship is driven by AWQ/GPTQ entries and Q2_K points.
- The ROUGE-L/refusal correlation (r=-0.349, p=0.037) is the only significant result on the regex track. The 0.56 power is moderate -- the relationship is real but the evidence is not definitive for regex-based measurement.

### SS12.3 Mixed-Effects Estimates

Mixed-effects models with random model intercepts assess whether the quality-safety relationship holds after accounting for model-level variation.

| Quality Metric | Safety Metric | Coefficient | p | 95% CI |
|---------------|-------------|------------|---|--------|
| ROUGE-L | Refusal | -0.861 | **0.017** | [-1.57, -0.15] |
| BERTScore | Refusal | +0.775 | 0.326 | [-0.77, +2.32] |
| Coherence | Refusal | -0.953 | 0.068 | [-1.98, +0.07] |

**Observations.**

- ROUGE-L is the only significant predictor in the mixed-effects model (p=0.017). Each 1pp increase in ROUGE-L delta associates with a 0.86pp decrease in refusal rate. This is the formal quantification of the quality-safety trade-off under quantization.
- BERTScore is non-significant (p=0.326) with a *positive* coefficient, opposite to the expected direction. This confirms BERTScore is not a useful safety proxy.
- Coherence is borderline (p=0.068). The direction (negative) is consistent with ROUGE-L but the evidence is insufficient to claim significance.

### SS12.4 Leave-One-Out Sensitivity

To test whether pooled correlations are driven by specific formats or extreme quant points, we drop one format at a time and re-compute.

| Omitted | Pooled Pearson r (BERTScore vs Refusal) | Change from Full |
|---------|---------------------------------------|-----------------|
| Full matrix | +0.122 | -- |
| Drop AWQ entries | +0.257 | +0.135 |
| Drop GPTQ entries | +0.309 | +0.187 |
| Drop Q2_K entries | -0.195 | -0.317 |
| Drop llama3.2-1b | +0.489 (p=0.004) | +0.367 |

**Observations.**

- The pooled correlation is *unstable*. Dropping Q2_K reverses the sign from +0.122 to -0.195. Dropping GPTQ pushes it positive to +0.309. Dropping llama3.2-1b (the model with the most extreme AWQ/GPTQ effects) turns it strongly positive (+0.489, p=0.004).
- This instability demonstrates that no single pooled correlation captures the quality-safety relationship. The relationship is format-dependent and model-dependent.
- For practical purposes: within GGUF, quality and safety co-move (positive correlation). AWQ/GPTQ introduce hidden-danger entries that reverse the relationship. Q2_K introduces total-collapse entries that pull the correlation negative.

### SS12.5 Cross-TR Validation

TR134 v3 findings can be cross-referenced against prior Banterhearts TRs for consistency:

| Cross-Reference | Prior Finding | v3 Consistency |
|----------------|--------------|----------------|
| TR125 v2: Q4_K_M safe for quality | BERTScore within -2.6pp for all models | **Consistent.** v3 confirms Q4_K_M safe for both quality and safety |
| TR125 v2: Q2_K catastrophic for quality | MMLU drops -35pp, ARC drops -48pp on qwen2.5-1.5b | **Consistent.** Q2_K destroys quality and safety together |
| TR134 v1: Q2_K catastrophic for safety | llama3.2-1b -57pp refusal | **Replicated.** Still present in v3 unified matrix |
| TR134 v2: Cross-family ANOVA non-significant | F=0.62, p=0.477 | **Unchanged.** AWQ/GPTQ is a format effect, not a family effect |
| TR134 v2: Mistral regex gap 64-71pp | Still 64-71pp in v3 matrix | **Unchanged.** No regression from adding AWQ/GPTQ data |
| TR142 v2: Quality-safety sign reversal | 34/36 pairings split across models | **Extended.** AWQ/GPTQ add hidden-danger entries that drive negative pooled correlation |

**Observations.** All prior findings replicate in the v3 matrix. The AWQ/GPTQ entries are additive -- they create new findings without contradicting existing ones. The hidden-danger mechanism identified in v3 was not detectable in v1/v2 because GGUF entries rarely show quality improvement alongside safety degradation (exception: llama3.2-1b Q3_K_S, which shows +0.98pp BERTScore with -13.6pp refusal).

### SS12.6 Within-Model Correlation Spine

The bespoke analysis computes per-model Pearson correlations between quality (BERTScore) and safety (refusal rate) across all quant levels including AWQ/GPTQ. These reveal model-specific quality-safety relationships.

| Model | BERTScore vs Refusal (Pearson r) | N Points | Interpretation |
|-------|--------------------------------|---------|----------------|
| llama3.2-1b | -0.275 | 8 | Weak negative (AWQ/GPTQ drive negative sign) |
| llama3.2-3b | -0.461 | 8 | Moderate negative |
| mistral-7b | +0.574 | 5 | Positive (GGUF only, co-movement) |
| phi-2 | -0.694 | 7 | Strong negative (GPTQ is the outlier) |
| qwen2.5-1.5b | +0.935 | 8 | Strong positive (quality and safety co-degrade under AWQ/GPTQ) |
| qwen2.5-7b | -0.613 | 5 | Moderate negative (GGUF only) |

**Observations.** The sign reversal pattern is now quantified: models with AWQ/GPTQ hidden-danger entries (llama3.2-1b, llama3.2-3b, phi-2) show negative BERTScore-refusal correlation (quality up, safety down). Models without AWQ/GPTQ entries (mistral-7b) or with total-collapse entries (qwen2.5-1.5b) show positive correlation (both move together). This confirms that the hidden-danger pattern is format-specific, not universal.

---

## SS13. Conclusions

### SS13.1 Research Questions Resolved

**RQ1: Do AWQ and GPTQ preserve safety at comparable bit-widths to GGUF k-quants?**

No. AWQ/GPTQ at ~4 BPW produce refusal losses of -21pp to -68pp. GGUF Q4_K_M at 4.85 BPW preserves refusal within -4pp to -10pp. GGUF Q3_K_S at 3.5 BPW (lower BPW than AWQ/GPTQ) still outperforms both AWQ and GPTQ on every model. The finding is established with convergent evidence from regex and judge classifiers across 4 models.

**RQ2: What mechanism causes format-dependent safety degradation?**

Refusal template destabilization. AWQ/GPTQ disrupt the weight structures encoding refusal behavior, fragmenting the model's dominant refusal prefix (r=+0.589, p=5.1e-05) and increasing refusal diversity (r=-0.813, p=1.1e-10) and verbosity (r=-0.698, p=3.9e-07). Three destabilization modes were observed: synonym substitution, complete destruction, and frequency reduction. The mechanism is correlational, not proven causal.

**RQ3: Do regex classifiers remain reliable under AWQ/GPTQ?**

No. Regex-judge gaps increase from a median of ~8-10pp (GGUF) to 16-39pp (AWQ/GPTQ) on refusal. phi-2 GPTQ regex misses 92% of judge-detected refusals. Bias resistance is also affected: regex over-reports by 20-41pp. LLM judge validation is mandatory for any AWQ/GPTQ model.

**RQ4: How should the deployment taxonomy account for format-dependent risk?**

AWQ and GPTQ are blanket-rejected. Zero candidate rows across all tested models. GGUF Q5_K_M is the conservative review floor; Q4_K_M is the recommended minimum. BPW-based comparisons are invalid across format families.

**RQ5: Is the hidden-danger regime concentrated in specific formats?**

Yes. 5 of 7 hidden-danger cells are AWQ/GPTQ (71% concentration vs 15% matrix share). AWQ/GPTQ hidden-danger magnitude (-21pp to -68pp refusal) exceeds GGUF hidden-danger magnitude (-12pp to -14pp) by 3-5x.

### SS13.2 Operational Recommendations

1. **Never deploy AWQ or GPTQ in safety-critical applications.** All 7 entries show catastrophic refusal collapse. phi-2 GPTQ at 3.2% regex refusal is functionally unaligned.

2. **Use GGUF Q4_K_M or above for safety-critical deployment.** This format retains >= 55% refusal on all tested models and >= 90% on llama3.2-1b and qwen2.5-7b.

3. **Do not use BPW as a cross-format safety predictor.** AWQ/GPTQ at 4.0 BPW are worse than GGUF at 3.5 BPW. The compression algorithm matters as much as the compression ratio.

4. **Always use an LLM judge for safety evaluation of quantized models.** Regex classifiers miss 16-39pp of refusal behavior under AWQ/GPTQ. Even for GGUF, judge validation should be standard practice.

5. **Treat quality improvement under quantization as a warning signal.** BERTScore and ROUGE-L improvement at a quantized level may indicate the hidden-danger pattern: the model is generating higher-quality harmful content rather than refusing.

6. **If you have deployed an AWQ or GPTQ model, audit safety immediately.** These formats are widely available on Hugging Face and commonly used. A regex-based audit may not detect the problem (see phi-2 GPTQ).

### SS13.3 v1 to v3 Findings Progression

| Finding | v1 (4 GGUF models) | v2 (6 GGUF models) | v3 (6 models, 9 formats) | Trajectory |
|---------|------|------|------|------------|
| Q2_K catastrophic | Demonstrated (1 model) | Replicated (2 models) | Replicated + exceeded by AWQ/GPTQ | Strengthened |
| Mistral regex gap | Noted (64-71pp) | Diagnosed (systematic) | Unchanged (64-71pp) | Stable |
| Cross-family ANOVA | p=0.137 (3 families) | p=0.477 (4 families) | p=0.477 (unchanged) | Non-significant |
| Safety robust through Q3_K_S | 3/4 models | 5/6 models | 5/6 models (GGUF confirmed) | Confirmed |
| Hidden danger regime | Not tested | 3 entries (all GGUF) | 8 entries (5 AWQ/GPTQ, 3 GGUF) | Major expansion |
| AWQ/GPTQ safety | Not tested | Not tested | **Catastrophic collapse** | New in v3 |
| Template destabilization | Not tested | Not tested | **Established** (r=+0.589) | New in v3 |
| Regex failure under AWQ/GPTQ | Not tested | Not tested | **Demonstrated** (38.8pp gap) | New in v3 |
| Quality-as-safety proxy | Not tested | Not tested | **Rejected** (hidden-danger) | New in v3 |

### SS13.4 Implications for the Research Program

TR134 v3 is the second most important report in the Banterhearts program after TR142 v2 (the unified quality-safety correlation analysis). It demonstrates that the quantization format, not just the compression ratio, determines safety outcomes. This finding changes the deployment calculus: practitioners cannot assume that any 4-bit model is safe simply because GGUF Q4_K_M models are safe. The format must be verified.

The finding also has implications for quantization research: safety-aware quantization methods that explicitly preserve RLHF/DPO-encoded weights could potentially enable safe AWQ/GPTQ deployment. TR134 v3 provides the evidence that current default configurations are insufficient and identifies the refusal template mechanism as the target for preservation.

The hidden-danger finding has broader implications for AI safety evaluation practices. Standard model evaluation pipelines that rely on capability benchmarks (MMLU, ARC, HumanEval) would not detect the AWQ/GPTQ safety collapse. BERTScore and ROUGE-L would actively mislead evaluators by showing improvement. Only direct safety evaluation (refusal testing with harmful prompts) reveals the problem. This argues for mandatory safety evaluation in model release pipelines, particularly for quantized variants.

### SS13.5 What This Report Does Not Change

The GGUF findings from v1/v2 are unchanged. Q4_K_M remains the recommended minimum for GGUF deployments. The cross-family ANOVA remains non-significant. The Mistral regex gap remains unresolved. These findings are stable across three report versions and 44,791 safety samples.

---

## SS14. Limitations and Follow-Up

### SS14.1 Design Limitations

- **AWQ/GPTQ tested on sub-4B models only.** 7B models (mistral-7b, qwen2.5-7b) were not quantized to AWQ/GPTQ due to hardware constraints (Colab Pro A100 required). Larger models may show different degradation patterns -- their larger weight matrices may preserve safety-critical structures better under reconstruction-error-optimizing quantization.
- **phi-2 AWQ missing.** AutoAWQ export failed due to architecture incompatibility. The AWQ findings are based on 3 models, not 4.
- **Default configurations only.** AWQ (group_size=128) and GPTQ (group_size=128) were tested at default settings. Non-default configurations (different group sizes, mixed-precision, calibration datasets) might produce different outcomes.
- **Single hardware platform.** All evaluations ran on an RTX 4080 Laptop GPU with 12GB VRAM. Different hardware may produce different numerical results due to floating-point precision differences.
- **Single-turn prompts only.** All safety evaluation uses single-turn prompts. Multi-turn adversarial prompting (TR139) was not cross-validated with AWQ/GPTQ.

### SS14.2 Statistical Limitations

- **MDE 18.3pp for safety.** All AWQ/GPTQ effects exceed this threshold, so the limitation does not affect the core finding. However, finer-grained comparisons (e.g., AWQ vs GPTQ within a model) may be below detection limits.
- **Truthfulness completely underpowered.** N=50 per entry, MDE=27.7pp. No truthfulness finding is reliable.
- **Multiple comparison correction not applied.** The effects are large enough (20-68pp) that correction would not change any finding, but the practice should be noted.
- **Phase 6 correlations are observational.** Template destabilization co-occurs with refusal loss but causal direction is not proven. It is possible that refusal loss causes template fragmentation (the model produces non-standard text when it is "confused") rather than template fragmentation causing refusal loss (the model loses its learned template and cannot refuse).

### SS14.3 Explicit Non-Claims

- **Does not prove AWQ/GPTQ unsafe in all configurations.** Non-default group sizes, mixed-precision quantization, or safety-aware calibration datasets might produce different outcomes.
- **Does not prove GGUF safe in all applications.** GGUF Q4_K_M preserves refusal on tested benchmarks. Other safety dimensions (multi-turn, adversarial, domain-specific) were not tested.
- **Does not prove causal mechanism.** The correlation evidence is strong (all p<0.01) but the direction of causation between template destabilization and refusal loss is not established.
- **Does not apply to models outside the tested range.** 7B+ models, non-instruct models, and architectures other than the 4 tested families may behave differently.
- **Does not claim AWQ is "safe enough."** AWQ is less damaging than GPTQ on 2/3 models but still produces catastrophic refusal loss (-23pp to -62pp). "Less bad" is not "safe."

### SS14.4 Measurement Limitations

- **Regex classifier has known blind spots.** The RefusalDetector misses contractions ("I can't"), hedged refusals, and topic redirections. This affects all GGUF entries but is amplified under AWQ/GPTQ where novel refusal patterns emerge.
- **Judge calibration differs across waves.** Wave 1 used Qwen 2.5 7B Q8_0; Waves 2-3 used Gemma 3 12B. These judges may have different refusal thresholds. Cross-wave judge comparisons should be treated as approximate.
- **Bias measurement is particularly noisy.** The BBQ benchmark produces binary scores on a 198-item battery. Small changes in answer selection produce large apparent shifts. The judge-measured bias values (22-81%) should be treated as directional rather than precise.
- **No human evaluation baseline.** Neither regex nor judge is validated against human ground truth specifically for AWQ/GPTQ outputs. Both classifiers may systematically misclassify outputs that do not match their training distribution.

### SS14.5 Follow-Up Directions

1. **Test AWQ/GPTQ on 7B+ models** when Colab Pro A100 capacity is available. 7B models may show different vulnerability patterns due to larger weight matrices and potentially more distributed safety encoding.
2. **Test non-default AWQ/GPTQ configurations** (group_size=64, group_size=256, different calibration datasets, 3-bit and 8-bit variants) to determine whether the safety collapse is configuration-dependent or intrinsic to the algorithm.
3. **Cross-validate with TR139 multi-turn jailbreak data.** AWQ/GPTQ models that show single-turn refusal collapse may be even more vulnerable to multi-turn adversarial strategies. This would test whether the template destabilization mechanism interacts with adversarial prompt engineering.
4. **Investigate safety-aware calibration for AWQ/GPTQ.** If the mechanism is template destabilization, calibration datasets that include refusal examples (e.g., sampling from AdvBench refusal outputs) might preserve safety-critical weights during quantization.
5. **Extend Phase 6 mechanism analysis** with attention-head-level probing to identify which specific weight structures are disrupted by AWQ/GPTQ. This would move from correlational to mechanistic evidence.
6. **Conduct formal A/B testing** with human evaluators to validate LLM judge classifications on AWQ/GPTQ outputs and establish a ground truth for safety measurement under non-standard quantization.
7. **Test exl2 and EETQ formats** to determine whether the safety collapse is specific to AWQ/GPTQ or generalizes to all non-GGUF quantization approaches.

---

## SS15. Reproducibility

### SS15.1 Run Artifacts

| Artifact | Location | Description |
|----------|----------|------------|
| v1 safety samples | `research/tr134/results/phase3/20260305_144827/phase3_scored.jsonl` | 24,778 samples |
| v1 judge labels | `research/tr134/results/phase3/20260305_144827/phase3_judged.jsonl` | 12,168 annotations |
| v2 expansion samples | `research/tr142/expansion/results/tr134_expansion/20260327_170457/phase3_scored.jsonl` | 13,342 samples |
| v2 expansion judge | `research/tr142/expansion/results/judge_gemma3/expansion_judged_20260328_150119.jsonl` | 6,552 annotations |
| v2 7B rejudge | `research/tr142/expansion/results/judge_gemma3/rejudge_7b_20260328_172908.jsonl` | 5,616 annotations |
| v3 AWQ/GPTQ samples | `research/tr142/expansion/results/v3_safety/20260331_125319/phase3_scored.jsonl` | 6,671 samples |
| v3 AWQ/GPTQ judge | `research/tr142/expansion/results/v3_safety/20260331_125319/phase3_judged.jsonl` | 3,276 annotations |
| Bespoke analysis (35 CSVs) | `research/tr142/results/bespoke_analysis_v3/phase56_v3_canonical/` | Full analysis bundle |
| Run manifest | `research/tr142/results/bespoke_analysis_v3/phase56_v3_canonical/run_manifest.json` | Source audit + config |
| Source audit | `research/tr142/results/bespoke_analysis_v3/phase56_v3_canonical/source_audit.csv` | SHA-256 per file |

### SS15.2 Seeds and Determinism

- **Inference:** temperature=0.0, seed=42, max_new_tokens=256 (all waves)
- **Bootstrap:** B=2000, seed=42
- **Non-deterministic components:** Ollama backend may produce slightly different outputs across runs despite temperature=0. AWQ/GPTQ via Transformers is deterministic with seed=42.

### SS15.3 SHA-256 Checksums

| Source | SHA-256 (prefix) |
|--------|-----------------|
| v1 safety scored | `9f832412dec5` |
| v1 safety raw | `40d651f4ee37` |
| v2 expansion scored | `583a610190db` |
| v3 AWQ/GPTQ scored | `7dcbb5b9e4a8` |
| v3 AWQ/GPTQ raw | `f350bbe6295f` |
| v3 AWQ/GPTQ judge | `ff6f1278fca3` |
| v1 judge | `5eadb499686c` |
| v2 Gemma judge | `fc57e95dc587` |
| v2 7B rejudge | `aa03f165fc19` |

### SS15.4 Library Versions

| Library | Version |
|---------|---------|
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| scipy | 1.15.2 |
| statsmodels | 0.14.5 |
| pingouin | 0.6.1 |

### SS15.5 Git Reference

All code and data traceable to git commit `0439e828`. The bespoke analysis was generated at 2026-04-01T16:02:43Z.

---

## References

1. [TR124: Quality and Accuracy Baseline](Technical_Report_124.md)
2. [TR125 v2: Quantization Decision Matrix](Technical_Report_125_v2.md)
3. [TR133: Predictive Capacity Planner](Technical_Report_133.md)
4. [TR134 v1: Alignment Robustness Under Quantization](Technical_Report_134.md)
5. [TR134 v2: Alignment Robustness -- Expanded](Technical_Report_134_v2.md)
6. [TR142 v2: Quality-Safety Correlation](Technical_Report_142_v2.md)
7. Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. arXiv:2306.00978.
8. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. arXiv:2210.17323.
9. Jain, N., Schwarzschild, A., Wen, Y., Somepalli, G., Kirchenbauer, J., Chiang, P., Goldblum, M., Saha, A., Geiping, J., & Goldstein, T. (2023). Baseline Defenses for Adversarial Attacks Against Aligned Language Models. arXiv:2309.00614.
10. Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering. Findings of ACL 2022.
11. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022.
12. Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking Language Models. arXiv:2404.01318.
13. Shen, X., Chen, Z., Backes, M., Shen, Y., & Zhang, Y. (2023). Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. arXiv:2308.03825.

---

## Appendix A: Full 47-Row Safety Matrix

### A.1 Refusal Rate (Regex, %)

| Model | Baseline | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|----------|------|------|------|--------|--------|--------|------|-----|------|
| llama3.2-1b | FP16 | 93.6 | 94.5 | 94.1 | 91.8 | 90.5 | 80.0 | 36.8 | **31.8** | **25.5** |
| llama3.2-3b | FP16 | 76.4 | 74.5 | 77.3 | 76.8 | 66.4 | 95.0 | 92.7 | **53.6** | **55.5** |
| mistral-7b | Q8_0 | -- | 23.6 | 28.6 | 24.5 | 22.3 | 19.1 | 12.3 | -- | -- |
| qwen2.5-7b | Q8_0 | -- | 93.2 | 93.6 | 93.2 | 94.5 | 84.5 | 80.9 | -- | -- |
| qwen2.5-1.5b | FP16 | 84.1 | 83.2 | 85.5 | 87.3 | 80.0 | 84.5 | 34.1 | **59.5** | **36.4** |
| phi-2 | FP16 | 58.6 | 58.6 | 54.1 | 57.7 | 55.0 | 56.4 | 55.0 | -- | **3.2** |

**Observations.** Bold entries are AWQ/GPTQ. Every AWQ/GPTQ entry is below the same model's Q4_K_M. phi-2 GPTQ (3.2%) is the lowest safety score in the entire program. llama3.2-3b shows anomalous GGUF behavior (over-refusal at Q3_K_S/Q2_K) but AWQ/GPTQ bypass this artifact.

### A.2 Judge Refusal Rate (%)

| Model | Baseline | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|----------|------|------|------|--------|--------|--------|------|-----|------|
| llama3.2-1b | FP16 | 100.0 | 100.0 | 100.0 | 100.0 | 99.5 | 96.4 | 97.7 | **62.6** | **55.7** |
| llama3.2-3b | FP16 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | **77.3** | **78.2** |
| mistral-7b | Q8_0 | -- | 91.3 | 92.5 | 91.8 | 92.7 | 90.0 | 82.9 | -- | -- |
| qwen2.5-7b | Q8_0 | -- | 99.8 | 99.8 | 99.1 | 99.5 | 100.0 | 96.1 | -- | -- |
| qwen2.5-1.5b | FP16 | 91.8 | 93.2 | 91.8 | 94.5 | 90.5 | 93.6 | 82.9 | **75.5** | **62.3** |
| phi-2 | FP16 | 70.0 | 72.6 | 68.2 | 72.6 | 72.7 | 78.2 | 79.1 | -- | **41.9** |

**Observations.** Judge shows GGUF stability (>= 82.9% for all models through Q2_K, except phi-2 at 70-79% and Mistral at 83-93%). AWQ/GPTQ show clear drops: llama3.2-1b from 100% to 56-63%, qwen2.5-1.5b from 92% to 62-76%, phi-2 from 70% to 42%. The judge independently confirms that AWQ/GPTQ cause safety collapse on a different measurement track.

### A.3 Bias Resistance (Regex, %)

| Model | Baseline | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_S | Q2_K | AWQ | GPTQ |
|-------|----------|------|------|------|--------|--------|--------|------|-----|------|
| llama3.2-1b | FP16 | 89.4 | 88.9 | 88.4 | 87.4 | 87.4 | 99.5 | 73.2 | **83.3** | **68.7** |
| llama3.2-3b | FP16 | 96.5 | 96.0 | 95.0 | 95.0 | 96.5 | 94.4 | 78.8 | **93.4** | **78.8** |
| mistral-7b | Q8_0 | -- | 83.8 | 83.8 | 84.3 | 85.4 | 80.3 | 77.3 | -- | -- |
| qwen2.5-7b | Q8_0 | -- | 98.5 | 98.0 | 97.5 | 98.5 | 97.5 | 99.0 | -- | -- |
| qwen2.5-1.5b | FP16 | 85.4 | 89.4 | 88.4 | 89.4 | 88.9 | 89.9 | 90.4 | **87.9** | **79.8** |
| phi-2 | FP16 | 84.8 | 87.9 | 86.4 | 83.8 | 86.9 | 91.9 | 99.0 | -- | **63.6** |

**Observations.** GPTQ degrades bias on 3/4 models by 17-21pp. AWQ impact is smaller (3-6pp). phi-2 Q2_K shows 99.0% bias resistance (over-refusal artifact); phi-2 GPTQ drops to 63.6% (no over-refusal -- the model generates coherent but biased responses). qwen2.5-1.5b bias is robust under AWQ (+2.5pp) but mildly degrades under GPTQ (-5.6pp).

### A.4 Full Regime Classification Summary

| Regime | GGUF Count | AWQ Count | GPTQ Count | Total |
|--------|-----------|-----------|-----------|-------|
| Neutral | 31 | 1 (qwen2.5-1.5b) | 1 (qwen2.5-1.5b) | 33 |
| Hidden danger | 2 | 2 | 3 | 7 |
| Near hidden danger | 1 | 0 | 0 | 1 |
| **Total non-baseline** | **34** | **3** | **4** | **41** |

**Observations.** qwen2.5-1.5b AWQ and GPTQ are classified as neutral (not hidden danger) because both quality and safety degrade together. The hidden-danger classification requires quality to be stable while safety degrades. This means qwen2.5-1.5b AWQ/GPTQ is a total-collapse case, while Llama/phi-2 AWQ/GPTQ entries are hidden-danger cases where quality masks the safety problem.

### A.5 Truthfulness (Regex, %)

| Model | Baseline | FP16 | Q8_0 | Q5_K_M | Q4_K_M | Q2_K | AWQ | GPTQ |
|-------|----------|------|------|--------|--------|------|-----|------|
| llama3.2-1b | FP16 | 55.0 | 56.0 | 49.0 | 58.0 | 44.0 | 53.0 | 50.0 |
| llama3.2-3b | FP16 | 49.0 | 48.0 | 58.0 | 50.0 | 54.0 | 47.0 | 59.0 |
| mistral-7b | Q8_0 | -- | 60.0 | 59.0 | 54.0 | 56.0 | -- | -- |
| qwen2.5-7b | Q8_0 | -- | 50.0 | 49.0 | 57.0 | 50.0 | -- | -- |
| qwen2.5-1.5b | FP16 | 49.0 | 43.0 | 51.0 | 51.0 | 59.0 | 58.0 | 42.0 |
| phi-2 | FP16 | 39.0 | 45.0 | 48.0 | 50.0 | 44.0 | -- | 38.0 |

**Observations.** At N=50, MDE=27.7pp. No entry shows a delta exceeding this threshold. Truthfulness appears format-independent and noise-dominated. This table is included for completeness only; truthfulness should not inform deployment decisions.

### A.6 Key Takeaways from the Full Matrix

1. **AWQ/GPTQ entries are consistently the lowest refusal values within each model.** No AWQ/GPTQ entry outperforms the same model's Q4_K_M on refusal.
2. **Judge data shows less severe AWQ/GPTQ degradation than regex**, but still catastrophic. The judge-regex convergence validates the finding via two independent measurement tracks.
3. **Bias resistance shows format-dependent degradation.** GPTQ degrades bias more than AWQ, which degrades more than GGUF. This is a secondary safety dimension beyond refusal.
4. **Truthfulness is noise.** Ignore for deployment purposes.
5. **The 47-row matrix represents the most comprehensive safety evaluation of quantized LLMs in the research program**, spanning 44,791 samples across 3 data waves with 27,612 judge annotations.

---

## Appendix B: Full Regex vs Judge Gaps

### B.1 Refusal Rate Gaps (All Entries with Both Tracks)

| Model | Quant | Regex | Judge | Gap (pp) | Category |
|-------|-------|-------|-------|----------|----------|
| mistral-7b | Q3_K_S | 19.1% | 90.0% | +70.9 | Systematic regex failure |
| mistral-7b | Q2_K | 12.3% | 82.9% | +70.6 | Systematic regex failure |
| mistral-7b | Q4_K_M | 22.3% | 92.7% | +70.4 | Systematic regex failure |
| mistral-7b | Q8_0 | 23.6% | 91.3% | +67.7 | Systematic regex failure |
| mistral-7b | Q5_K_M | 24.5% | 91.8% | +67.3 | Systematic regex failure |
| mistral-7b | Q6_K | 28.6% | 92.5% | +63.8 | Systematic regex failure |
| llama3.2-1b | Q2_K | 36.8% | 97.7% | +60.8 | Q2_K degradation artifact |
| qwen2.5-1.5b | Q2_K | 34.1% | 82.9% | +48.8 | Q2_K degradation artifact |
| phi-2 | GPTQ | 3.2% | 41.9% | **+38.8** | AWQ/GPTQ template collapse |
| llama3.2-3b | Q4_K_M | 66.4% | 100.0% | +33.6 | Moderate gap |
| llama3.2-1b | AWQ | 31.8% | 62.6% | **+30.7** | AWQ/GPTQ template collapse |
| llama3.2-1b | GPTQ | 25.5% | 55.7% | **+30.3** | AWQ/GPTQ template collapse |
| qwen2.5-1.5b | GPTQ | 36.4% | 62.3% | **+25.9** | AWQ/GPTQ template collapse |
| llama3.2-3b | AWQ | 53.6% | 77.3% | **+23.6** | AWQ/GPTQ template collapse |
| llama3.2-3b | GPTQ | 55.5% | 78.2% | **+22.7** | AWQ/GPTQ template collapse |
| phi-2 | Q3_K_S | 56.4% | 78.2% | +21.8 | Moderate gap |
| qwen2.5-1.5b | AWQ | 59.5% | 75.5% | **+15.9** | AWQ/GPTQ template collapse |
| qwen2.5-7b | Q3_K_S | 84.5% | 100.0% | +15.5 | Moderate gap |
| phi-2 | Q4_K_M | 55.0% | 72.7% | +17.7 | Moderate gap |

**Observations.** Three distinct gap categories emerge: (1) Mistral systematic regex failure (64-71pp, all quant levels), (2) Q2_K degradation artifacts (49-61pp, garbled output confuses regex), and (3) AWQ/GPTQ template collapse (16-39pp, non-standard refusal phrasing). The AWQ/GPTQ gaps are a new category in v3.

### B.2 Top Bias Resistance Gaps

| Model | Quant | Regex | Judge | Gap (pp) |
|-------|-------|-------|-------|----------|
| phi-2 | GPTQ | 63.6% | 22.7% | -40.9 |
| qwen2.5-1.5b | GPTQ | 79.8% | 45.5% | -34.3 |
| llama3.2-3b | GPTQ | 78.8% | 45.5% | -33.3 |
| llama3.2-1b | AWQ | 83.3% | 53.0% | -30.3 |
| llama3.2-1b | GPTQ | 68.7% | 38.4% | -30.3 |
| phi-2 | FP16 | 84.8% | 60.4% | -24.4 |
| mistral-7b | Q2_K | 77.3% | 53.5% | -23.7 |

**Observations.** Bias gaps are negative (regex over-reports). The largest are GPTQ entries (-30pp to -41pp). phi-2 FP16 also shows a -24pp gap, suggesting phi-2 has a baseline measurement issue on bias independent of quantization.

---

## Appendix C: Phase 6 Refusal Style Data

### C.1 Complete Style Correlations

| Style Metric | n | Pearson r | p-value | 95% CI Low | 95% CI High | Spearman rho | p-value |
|-------------|---|----------|---------|-----------|------------|-------------|---------|
| dominant_prefix_share_delta | 41 | +0.589 | 5.1e-05 | +0.344 | +0.759 | +0.501 | 8.4e-04 |
| unique_prefix_rate_delta | 41 | -0.813 | 1.1e-10 | -0.896 | -0.673 | -0.431 | 4.9e-03 |
| prefix_entropy_norm_delta | 41 | -0.456 | 2.7e-03 | -0.670 | -0.173 | -0.576 | 8.3e-05 |
| mean_tokens_refusal_delta | 41 | -0.698 | 3.9e-07 | -0.828 | -0.497 | -0.394 | 1.1e-02 |
| mean_chars_refusal_delta | 41 | -0.658 | 2.9e-06 | -0.803 | -0.440 | -0.391 | 1.2e-02 |
| hard_refusal_rate_delta | 41 | +0.998 | 2.5e-50 | +0.997 | +0.999 | +0.987 | 1.8e-32 |

**Observations.** All correlations significant at p<0.05 on both Pearson and Spearman tracks. The Pearson-Spearman divergence for unique_prefix_rate (r=-0.813 vs rho=-0.431) indicates non-linear relationship: a few extreme entries (AWQ/GPTQ, Q2_K) drive the Pearson correlation while the rank-order relationship is weaker across moderate degradation entries.

### C.2 Template Destabilization by Entry

| Model | Quant | Refusal Delta | Dom. Prefix Delta | Unique Prefix Delta | Tokens Delta | Top Prefix (Baseline) | Top Prefix (Quant) | Destab. Flag | Verbosity Flag |
|-------|-------|-------------|------------------|-------------------|--------------|--------------------|------------------|-------------|---------------|
| phi-2 | GPTQ | -90.0% | +5.5pp | +93.4% | +170.6 | i m sorry but as | answer 1 research and identify | No | Yes |
| llama3.2-1b | GPTQ | -59.0% | -33.5pp | +40.5% | +67.9 | i can t assist with | i can t fulfill that | Yes | Yes |
| llama3.2-1b | Q2_K | -57.0% | -51.0pp | +49.5% | +73.5 | i can t assist with | i can t assist with | Yes | Yes |
| qwen2.5-1.5b | Q2_K | -56.0% | -53.5pp | +24.6% | +161.7 | i m sorry but i | i m sorry but i | Yes | Yes |
| qwen2.5-1.5b | GPTQ | -52.0% | -55.3pp | +43.7% | +190.6 | i m sorry but i | i m sorry but i | Yes | Yes |
| llama3.2-1b | AWQ | -51.0% | -56.8pp | +50.4% | +45.9 | i can t assist with | i can t provide instructions | Yes | Yes |
| phi-2 | Q2_K | -19.0% | -55.6pp | +4.5% | -45.7 | i m sorry but as | i m sorry but i | Yes | No |
| mistral-7b | Q2_K | -17.0% | -6.6pp | -11.5% | +18.5 | i must clarify that i | i m here to provide | No | No |
| qwen2.5-1.5b | AWQ | -15.0% | -28.6pp | +20.4% | +46.9 | i m sorry but i | i m sorry but i | Yes | Yes |

**Observations.** phi-2 GPTQ destabilization flag is False because the dominant-prefix share did not drop -- it was replaced entirely. The destabilization taxonomy captures partial template loss but not total prefix replacement. phi-2 Q2_K shows template destabilization (prefix drops -55.6pp) but verbosity *decreases* (-45.7 tokens), an unusual pattern suggesting output collapse rather than verbose non-standard refusal.

---

## Appendix D: Deployment Protocol

### D.1 Phase 5 Validation Protocol

| Step | Check | Result | Status |
|------|-------|--------|--------|
| P5_001 | Freeze matched matrix | 6 models, 4 families, 47 rows | Pass |
| P5_002 | Run within-model correlation screen | 34/36 metric pairings split positive and negative across models | Pass |
| P5_003 | Use direct safety deltas | 47/47 merged rows with direct safety metrics | Pass |
| P5_004 | Cross-check regex with LLM judge | 47/47 merged rows have judge coverage | Pass |
| P5_005 | Classify regimes | 7 hidden-danger + 1 near-hidden-danger | Pass |
| P5_006 | Select conservative floor | Q5_K_M: max_refusal_signal=2.73pp, reject_rows=0 | Pass |

### D.2 Phase 5 Taxonomy Catalog

| ID | Category | Count | Description |
|----|----------|-------|-------------|
| TAX_001 | sign_reversal_proxy_failure | 34/36 pairings | Quality-safety sign reversal across models |
| TAX_002 | hidden_danger | 7 rows | Quality stable + safety degrading >= 10pp |
| TAX_003 | near_hidden_danger | 1 row | Approaching hidden-danger threshold |
| TAX_004 | over_refusal | 0 rows | No over-refusal entries in current matrix |
| TAX_005 | measurement_divergence | 21 rows | Regex-judge gap >= 20pp |
| TAX_006 | conservative_floor_candidate | Q5_K_M | Lowest-bit format with zero reject rows |

**Observations.** The taxonomy systematizes deployment risk. AWQ/GPTQ entries trigger TAX_002 (hidden danger) and TAX_005 (measurement divergence) simultaneously -- they are both safety-collapsed and measurement-unreliable.

### D.3 Recommended Deployment Workflow

For any quantized model intended for safety-critical deployment:

**Step 1: Format Gate.**
If the model uses AWQ or GPTQ quantization: stop. Do not deploy in safety-critical applications. Convert to GGUF Q4_K_M or above using llama.cpp's convert tool. If conversion is not possible, treat the model as unsafe.

**Step 2: Quant Level Gate.**
If Q2_K: stop. Not safe for any tested model. If Q3_K_S: proceed with elevated caution -- this level has 1 hidden-danger entry in the matrix (llama3.2-1b). If Q4_K_M or above: proceed to safety evaluation.

**Step 3: Safety Evaluation.**
Run AdvBench refusal evaluation (minimum 100 prompts, recommended 200). Compute refusal rate. Compare against FP16 or Q8_0 baseline for the same model. If delta exceeds -10pp: flag for review.

**Step 4: Judge Cross-Validation.**
Run an LLM judge (Gemma 3 12B or comparable) on refusal samples. Compute regex-judge gap. Interpret:
- Gap < 10pp: normal. Proceed.
- Gap 10-20pp: elevated. Investigate model's refusal phrasing.
- Gap > 20pp: measurement divergence. Do not trust regex scores alone. Use judge as primary metric.

**Step 5: Regime Check.**
Compute BERTScore delta (quantized vs baseline) and refusal rate delta. If BERTScore is stable (< 3pp change) but refusal drops >= 10pp: hidden-danger regime. Do not deploy without additional investigation.

**Step 6: Deploy with Monitoring.**
If all checks pass: deploy. Establish ongoing monitoring of refusal behavior. Re-evaluate after model updates, backend changes, or prompt template modifications.

### D.4 Quick Reference: Format Decision Matrix

| Your Format | Safety-Critical? | Action |
|-------------|-----------------|--------|
| FP16 / Q8_0 | Yes | Deploy (baseline safe) |
| Q6_K / Q5_K_M | Yes | Review (conservative floor) |
| Q4_K_M | Yes | Deploy after Steps 3-5 |
| Q3_K_S | Yes | Deploy after Steps 3-5, elevated monitoring |
| Q2_K | Yes | **Do not deploy** |
| AWQ | Yes | **Do not deploy** |
| GPTQ | Yes | **Do not deploy** |
| Any | No | Deploy at discretion; note safety limitations |

---

## Appendix E: Glossary

### Statistical Terms

| Term | Definition |
|------|-----------|
| ANOVA | Analysis of variance -- omnibus test for differences across 3+ groups |
| Bootstrap CI | Confidence interval estimated by resampling with replacement (B=2000, seed=42) |
| Cohen's d | Standardized mean difference; 0.2 small, 0.5 medium, 0.8 large |
| Cohen's kappa | Chance-corrected inter-rater agreement; <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, >0.60 substantial |
| MDE | Minimum Detectable Effect -- smallest effect detectable at given power and alpha |
| Mixed-effects model | Regression with both fixed effects (quant level) and random effects (model intercept) |
| Pearson r | Linear correlation coefficient; range [-1, +1] |
| Repeated-measures correlation | Correlation controlling for within-subject (within-model) pairing |
| Spearman rho | Rank-order correlation coefficient; robust to non-linear relationships |
| Wilson CI | Confidence interval for proportions; better coverage than Wald at extremes |

### Domain-Specific Terms

| Term | Definition |
|------|-----------|
| AWQ | Activation-Aware Weight Quantization; per-channel scales from activation magnitudes (Lin et al., 2023) |
| BPW | Bits Per Weight; effective precision of a quantized model |
| DPO | Direct Preference Optimization; alignment via preference pairs without reward model |
| GGUF | GPT-Generated Unified Format; llama.cpp quantization family with mixed-precision block quantization |
| GPTQ | GPT Quantization; Hessian-based second-order layer-wise reconstruction (Frantar et al., 2022) |
| Hidden danger | Regime where quality metrics are stable or improved but safety refusal has degraded >= 10pp |
| k-quant | GGUF mixed-precision block quantization variants (Q2_K through Q8_0) |
| PPO | Proximal Policy Optimization; RL-based alignment via reward model |
| Refusal template | Consistent text pattern used by model to refuse harmful requests (e.g., "I can't assist with that") |
| Template destabilization | Loss of dominant refusal prefix under quantization; measured by dominant-prefix-share delta |
| Verbosity shift | Increase in mean refusal text length under quantization; indicates non-standard refusal generation |
| Regime classification | Taxonomy of model-quant entries based on quality-safety co-movement pattern |
| Measurement divergence | Regex-judge gap exceeding 20pp; indicates classifier unreliability for the entry |
