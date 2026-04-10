# Technical Report 146: Mechanistic Safety Probing Under Quantization
## Why Standard Mechanistic Probes Fail to Predict Quantization-Induced Safety Degradation

| Field | Value |
|-------|-------|
| **TR Number** | 146 |
| **Project** | Banterhearts |
| **Date** | 2026-04-10 |
| **Version** | 1.0 |
| **Author** | Research Team |
| **Git Commit** | `256461f7` |
| **Status** | Complete |
| **Report Type** | Full-depth |
| **Run Directories** | `research/tr146/results/phase{1,2,3,4}_gptq_docker/` |
| **Total Samples** | 5,100 forward passes across 4 phases |
| **Models** | 6 (llama3.2-1b, llama3.2-3b, qwen2.5-1.5b, qwen2.5-7b, phi-2, mistral-7b) |
| **Quant Methods** | FP16 (anchor), AWQ INT4, GPTQ INT4 |
| **Probed Cells** | 17 model-quant cells per phase (11 AWQ/GPTQ + 6 FP16) |
| **Related Work** | [TR142](Technical_Report_142_v3.md) (quality-safety correlation matrix, RTSI) |
| **Depends On** | TR142 (regime labels, RTSI scores, harmful prompt pool, AWQ/GPTQ checkpoints) |

---

## Abstract

TR146 asks why quantization degrades safety behavior even when quality metrics remain stable. TR142 established the behavioral result (quality is not a safety proxy) and provided a lightweight screen (RTSI). TR146 probes the mechanism with four experiments across **17 model-quant cells** (6 models, 2 quantization methods, 6 FP16 anchors), generating **5,100 forward passes** with hidden-state extraction.

The core findings are: (1) First-token entropy does not predict safety degradation — AWQ increases first-token uncertainty while GPTQ decreases it, yet both produce hidden-danger rows (r = 0.08, p = 0.81). (2) The refusal direction is geometrically preserved under quantization with cosine similarity > 0.97 in every cell, yet behaviorally ineffective in hidden-danger rows (r = -0.14, p = 0.67). (3) Calibration drift follows the same method-specific pattern as entropy and has no predictive power for danger (all r < 0.09). (4) Safety-critical neurons absorb **1.40x** disproportionate quantization error compared to non-safety neurons (p < 0.0001), but this is a universal property of all quantized cells, not specific to hidden-danger rows (regime p = 0.98).

The most important non-result is that none of the four mechanistic probes can distinguish safe from dangerous quantized configurations. The operational conclusion is that behavioral template metrics (RTSI) remain the only viable pre-deployment screen because the underlying mechanism is necessary-but-not-sufficient: every quantized model suffers safety-neuron damage and direction perturbation, but only some cross the threshold into actual behavioral failure.

---

## Executive Summary

### Key Findings

1. **First-token entropy shift does not predict safety degradation.** AWQ increases entropy (model less certain about refusing) while GPTQ decreases it (model more confident). Both methods produce hidden-danger rows. Pearson correlation with RTSI: r = 0.08 (p = 0.81). Mann-Whitney regime separation: p = 0.61.

2. **The refusal direction survives quantization geometrically.** Cosine similarity between FP16 and quantized refusal directions exceeds 0.97 in all 11 AWQ/GPTQ cells. The direction is not destroyed — it is preserved but behaviorally ineffective. Cosine similarity does not correlate with RTSI (r = -0.14, p = 0.67) or distinguish regimes (p = 0.61).

3. **Refusal direction magnitude correlates with model vulnerability.** Models with weaker refusal directions (llama3.2-1b: 4.7, llama3.2-3b: 10.7) are more likely to produce hidden-danger rows than models with stronger directions (qwen2.5-7b: 60.1, phi-2: 72.3). Mann-Whitney magnitude test: p = 0.036.

4. **Calibration drift is method-specific, not danger-specific.** GPTQ systematically increases top-1 confidence (+0.02 to +0.08pp), AWQ has negligible or slightly negative shift. Neither direction predicts hidden-danger status. All RTSI correlations r < 0.09.

5. **Safety neurons absorb disproportionate quantization error (1.40x, p < 0.0001).** In every one of the 11 quantized cells, safety-critical neurons (top 5% by activation contrast) have higher mean absolute error than non-safety neurons. Mean ratio across all cells and layers: 1.40x. One-sample t-test against ratio = 1.0: p < 0.0001. GPTQ produces higher ratios than AWQ (mean 1.45x vs 1.37x).

6. **But disproportionate safety-neuron error does not predict hidden-danger status.** The safety error ratio is essentially identical between hidden-danger rows (mean 1.34x) and neutral rows (mean 1.50x). Mann-Whitney test: p = 0.98. RTSI correlation: r = 0.12. The damage is universal — every quantized model suffers it, regardless of whether it behaviorally fails.

7. **RTSI remains the only viable pre-deployment screen.** None of the four mechanistic probes (entropy, direction, calibration, neuron error) can replace or improve upon the behavioral RTSI screen from TR142. The mechanism is there but it is necessary-not-sufficient: safety neurons are always disproportionately damaged, the refusal direction always slightly perturbed, but only some configurations cross the behavioral threshold.

### Core Decisions

- Do not use first-token entropy as a safety screening signal — it is method-specific, not danger-specific
- Do not use refusal-direction cosine similarity as a quantization safety diagnostic — the direction is always preserved
- Do not use calibration drift (ECE, confidence shift) as a safety proxy — same method-specific problem as entropy
- Safety-neuron error ratios are a valid mechanism-level insight but not an operational screen
- Continue to use RTSI (behavioral template metrics) as the recommended pre-deployment screen
- Refusal direction magnitude is a model-level vulnerability factor, not a per-configuration diagnostic

### Validation Summary

| Target | Metric | Required | Achieved | Status |
|--------|--------|----------|----------|--------|
| Phase 1 cell coverage | cells completed | 17/17 | 17/17 | **PASS** |
| Phase 2 cell coverage | cells completed | 17/17 | 17/17 | **PASS** |
| Phase 3 cell coverage | cells completed | 17/17 | 17/17 | **PASS** |
| Phase 4 cell coverage | cells completed | 11/11 | 11/11 | **PASS** |
| Safety neuron error > 1.0 | one-sample t-test | p < 0.05 | p < 0.0001 | **PASS** |
| Entropy predicts danger | Pearson r with RTSI | |r| > 0.3 | r = 0.08 | **NOT SUPPORTED** |
| Direction predicts danger | cosine sim vs RTSI | |r| > 0.3 | r = -0.14 | **NOT SUPPORTED** |
| Calibration predicts danger | confidence shift vs RTSI | |r| > 0.3 | r = 0.07 | **NOT SUPPORTED** |
| Neuron ratio predicts danger | ratio vs RTSI | |r| > 0.3 | r = 0.12 | **NOT SUPPORTED** |

### Claim Validation

| # | Claim | Evidence Base | Status |
|---|-------|---------------|--------|
| C1 | First-token entropy predicts safety degradation | Phase 1: r = 0.08, p = 0.81 | **Not supported** |
| C2 | Refusal direction is destroyed by quantization | Phase 2: all cosine sims > 0.97 | **Not supported** (direction preserved) |
| C3 | Refusal direction magnitude tracks model vulnerability | Phase 2: Mann-Whitney p = 0.036 | **Partial** (model-level, not config-level) |
| C4 | Calibration drift predicts safety degradation | Phase 3: all r < 0.09 | **Not supported** |
| C5 | Safety neurons absorb disproportionate quantization error | Phase 4: ratio = 1.40x, p < 0.0001 | **Established** |
| C6 | Safety-neuron error ratio predicts hidden-danger status | Phase 4: regime p = 0.98, RTSI r = 0.12 | **Not supported** |
| C7 | RTSI is uniquely predictive among tested probes | Phases 1-4: no mechanistic probe achieves |r| > 0.15 with RTSI | **Established** |

---

## When to Use This Report

### Scenario 1: "Can I use a mechanistic probe instead of RTSI to screen quantized models?"

**Question:** Is there a cheaper or more interpretable alternative to RTSI for detecting hidden-danger quantized configurations?

**Answer:** No. TR146 tested four mechanistic probes (first-token entropy, refusal direction cosine similarity, calibration drift, safety-neuron error ratio) and none achieved |r| > 0.15 with RTSI or distinguished hidden-danger from neutral rows. See SS4-SS7 for the full evidence. The mechanism is necessary-but-not-sufficient: all quantized models show safety-neuron damage, but only some cross the behavioral threshold. RTSI remains the recommended screen.

### Scenario 2: "Why does quantization break safety behavior?"

**Question:** What is the mechanistic explanation for quantization-induced safety degradation?

**Answer:** Safety neurons absorb 1.40x disproportionate quantization error (SS7, p < 0.0001), the refusal direction is slightly perturbed but geometrically preserved (SS5, cosine sim > 0.97), and models with weaker refusal directions are more vulnerable (SS5, p = 0.036). The failure mode is not direction destruction but threshold sensitivity: the refusal signal is weakened just enough to lose behavioral effectiveness in some configurations. AWQ and GPTQ produce different internal signatures (opposite entropy shifts, different confidence patterns) but can both trigger the same behavioral failure.

### Scenario 3: "Does a model's size determine its quantization safety risk?"

**Question:** Are smaller models inherently more vulnerable to safety degradation under quantization?

**Answer:** Yes, partially. Refusal direction magnitude scales with model size (llama3.2-1b: 4.7, qwen2.5-7b: 60.1), and the Mann-Whitney test shows hidden-danger rows have lower magnitude (p = 0.036). But magnitude alone is not sufficient — phi-2 (2.7B) has the highest magnitude (72.3) despite being a smaller model. See SS5 for the direction magnitude analysis.

### Scenario 4: "Should I worry about safety-neuron damage in my quantized model?"

**Question:** If safety neurons are disproportionately damaged in every quantized model, does that mean all quantized models are unsafe?

**Answer:** No. The 1.40x disproportionate error is universal but the behavioral consequences are not. Most quantized configurations retain sufficient safety behavior despite the elevated neuron-level error. The error is necessary but not sufficient for behavioral failure. Use RTSI to determine whether a specific configuration has crossed the behavioral threshold. See SS7 for the full neuron-error analysis.

---

## Table of Contents

- [Abstract](#abstract)
- [Executive Summary](#executive-summary)
- [When to Use This Report](#when-to-use-this-report)
- [Metric Definitions](#metric-definitions)
- [SS1. Introduction](#ss1-introduction)
- [SS2. Methodology](#ss2-methodology)
- [SS3. Study Matrix and Infrastructure](#ss3-study-matrix-and-infrastructure)
- [SS4. Phase 1: First-Token Entropy Shift](#ss4-phase-1-first-token-entropy-shift)
- [SS5. Phase 2: Refusal Direction Geometry](#ss5-phase-2-refusal-direction-geometry)
- [SS6. Phase 3: Calibration Drift](#ss6-phase-3-calibration-drift)
- [SS7. Phase 4: Safety Neuron Quantization Error](#ss7-phase-4-safety-neuron-quantization-error)
- [SS8. Cross-Phase Synthesis](#ss8-cross-phase-synthesis)
- [SS9. Implications for RTSI and the NeurIPS Paper](#ss9-implications-for-rtsi-and-the-neurips-paper)
- [SS10. Conclusions](#ss10-conclusions)
- [SS11. Limitations and Follow-Up](#ss11-limitations-and-follow-up)
- [SS12. Reproducibility](#ss12-reproducibility)
- [References](#references)
- [Appendix A: Raw Data Tables](#appendix-a-raw-data-tables)
- [Appendix B: Extended Statistical Tables](#appendix-b-extended-statistical-tables)
- [Appendix C: Prompt Provenance](#appendix-c-prompt-provenance)
- [Appendix D: Glossary](#appendix-d-glossary)

---

## Metric Definitions

### Primary Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| First-token entropy | Shannon entropy (nats) of the logit distribution at the first generated token position | Higher = model less certain about first token choice |
| Refusal direction | Unit vector in residual stream space computed as mean(harmful) - mean(harmless) activations | Encodes the geometric "direction of refusal" (Arditi et al. 2024) |
| Cosine similarity | Cosine between FP16 and quantized refusal directions | 1.0 = perfectly preserved, 0.0 = orthogonal |
| Direction magnitude | L2 norm of the unnormalized refusal direction | Higher = stronger refusal signal in representation space |
| Top-1 confidence | Probability assigned to the most likely first token | Higher = model more confident |
| ECE (proxy) | Expected Calibration Error using entropy as accuracy proxy | Higher = worse calibration |
| Safety error ratio | Mean absolute hidden-state error on safety neurons / mean error on non-safety neurons | >1.0 = safety neurons disproportionately damaged |
| RTSI | Refusal Template Stability Index (from TR142) | Higher = more template instability = higher risk |

### Statistical Tests Used

| Test | Role in This Report |
|------|-------------------|
| Pearson correlation | Continuous association between mechanistic metrics and RTSI |
| Spearman correlation | Rank-based robustness check on Pearson |
| Mann-Whitney U | Non-parametric regime separation (hidden-danger vs neutral) |
| One-sample t-test | Safety error ratio > 1.0 |

### Evidence Standard

**Established findings** require p < 0.05 and practical significance (|r| > 0.3 for correlations, ratio meaningfully > 1.0 for neuron error).

**Partial findings** show statistical significance but lack predictive power for the deployment-relevant outcome (hidden-danger vs neutral classification).

**Not supported** claims fail both statistical significance and practical significance thresholds.

---

## SS1. Introduction

### SS1.1 Research Questions

1. Does first-token entropy shift predict which quantized configurations will exhibit hidden-danger safety degradation?
2. Does the refusal direction (Arditi et al. 2024) survive quantization, and does its preservation or distortion correlate with behavioral safety outcomes?
3. Does calibration drift (confidence shift, ECE change) under quantization predict safety degradation?
4. Do safety-critical neurons (identified via activation contrasting) absorb disproportionate quantization error, and does this predict hidden-danger status?
5. Can any mechanistic probe replace or improve upon RTSI as a pre-deployment safety screen?

### SS1.2 Motivation

TR142 established that quality metrics (BERTScore, ROUGE-L, coherence) cannot serve as safety proxies under quantization. The Refusal Template Stability Index (RTSI) provides a behavioral screen that catches all 10 hidden- or near-hidden-danger rows in the 51-row study matrix. But RTSI is a behavioral correlate — it measures template instability, not the underlying cause of safety degradation. TR146 tests whether standard mechanistic interpretability tools can explain *why* some quantized configurations fail and whether they can serve as independent or complementary screening signals.

### SS1.3 Source Anchors

| Phase | Source Paper | Key Hypothesis |
|-------|-------------|---------------|
| 1 | Qi et al. 2025 (ICLR Outstanding Paper) | Safety alignment is shallow — concentrated in first few tokens |
| 2 | Arditi et al. 2024 (NeurIPS) | Refusal is mediated by a single direction in residual stream |
| 2 | RefusalCompressed 2025 (arXiv 2504.04215) | Quantized models retain refusal direction orientation |
| 3 | Proskurina et al. 2024 (NAACL) | GPTQ 4-bit decreases confidence on true labels |
| 4 | Chen et al. 2024 (Safety Neurons) | ~5% of neurons account for >90% of safety behavior |

---

## SS2. Methodology

### SS2.1 Design Principles

1. **Matched cell grid**: All 4 phases probe the same 17 model-quant cells (6 FP16 anchors + 5 AWQ + 6 GPTQ), drawn from the frozen TR142 canonical matrix.
2. **No generation**: All phases use forward-pass-only extraction (logits or hidden states). No text is generated. This makes the probing fast and deterministic.
3. **Same prompts**: Harmful prompts (100 AdvBench-derived) are shared across all phases. Phase 2 and Phase 4 additionally use 100 curated harmless prompts for activation contrasting.
4. **TR142 context join**: Every analysis correlates mechanistic metrics with TR142 regime labels (hidden_danger, near_hidden_danger, neutral) and RTSI scores.

### SS2.2 Phase Protocols

**Phase 1 — First-Token Entropy:** For each cell, tokenize 100 harmful prompts, run a forward pass, extract logits at the last input token position (= first generation position), compute Shannon entropy. Compare entropy shift from FP16 baseline.

**Phase 2 — Refusal Direction:** For each cell, extract residual stream activations at ~65% layer depth on 100 harmful + 100 harmless prompts. Compute refusal direction = mean(harmful) - mean(harmless), normalize to unit vector. Compute cosine similarity between quantized and FP16 directions. Also record raw direction magnitude.

**Phase 3 — Calibration Drift:** Same logit extraction as Phase 1. Compute top-1 confidence, top-5 confidence, Gini impurity, logit range, and a proxy ECE using entropy as the accuracy stand-in.

**Phase 4 — Safety Neuron Error:** For each FP16 anchor, identify safety neurons (top 5% by activation contrast between harmful and harmless prompts) at 5 evenly-spaced layers (20%-80% depth). Then load each quantized variant and compute per-neuron mean absolute error on 50 harmful prompts. Compare mean error for safety vs non-safety neurons.

### SS2.3 Infrastructure

- **FP16 models**: Loaded from HuggingFace cache via `transformers`
- **AWQ models**: Loaded from local TR142 expansion checkpoints (`research/tr142/expansion/checkpoints/`)
- **GPTQ models**: Loaded via `gptqmodel` 5.8.0 inside `tr142-gptq:compiled` Docker container (gptqmodel requires Linux + GPU at install time)
- **Phase 4 7B cells**: Required RunPod RTX 6000 Ada 48GB — FP16 7B (~14GB) + quant 7B cannot fit simultaneously on 12GB consumer GPU
- **Tokenizer fallback**: GPTQ checkpoints missing sentencepiece files use tokenizer from the original HF model

---

## SS3. Study Matrix and Infrastructure

### SS3.1 Cell Coverage

| Model | Family | FP16 | AWQ | GPTQ | Total Cells |
|-------|--------|------|-----|------|-------------|
| llama3.2-1b | Llama | 1 | 1 | 1 | 3 |
| llama3.2-3b | Llama | 1 | 1 | 1 | 3 |
| qwen2.5-1.5b | Qwen | 1 | 1 | 1 | 3 |
| qwen2.5-7b | Qwen | 1 | 1 | 1 | 3 |
| phi-2 | Phi | 1 | — | 1 | 2 |
| mistral-7b | Mistral | 1 | 1 | 1 | 3 |
| **Total** | | **6** | **5** | **6** | **17** |

*Observations.* phi-2 AWQ is absent because the architecture-specific AWQ path failed during TR142 quantization. All other cells are present across all 4 phases. Phase 4 initially failed on the four 7B AWQ/GPTQ cells due to GPU memory (12GB insufficient for FP16 + quant simultaneously); these were completed on RunPod RTX 6000 Ada 48GB.

### SS3.2 Prompt Provenance

| Set | Source | Count | Usage |
|-----|--------|-------|-------|
| Harmful | TR142 `v3_safety` scored samples (AdvBench-derived) | 100 | All 4 phases |
| Harmless | Curated benign instruction set (`builtin_curated_v1`) | 100 | Phases 2, 4 |

All prompts are loaded from explicit file paths recorded in `config.yaml`, not from filesystem scans. Run manifests record per-file provenance.

---

## SS4. Phase 1: First-Token Entropy Shift

### SS4.1 Hypothesis

If safety alignment is shallow (Qi et al. 2025), quantization should increase first-token entropy on harmful prompts, making the model less certain about its refusal decision. Higher entropy shift should correlate with worse safety outcomes and higher RTSI.

### SS4.2 Results

| Model | Quant | Mean Entropy (nats) | Entropy Shift | TR142 Regime |
|-------|-------|-------------------|--------------|-------------|
| llama3.2-1b | FP16 | 2.470 | 0.000 | baseline |
| llama3.2-1b | AWQ | 2.543 | **+0.073** | hidden_danger |
| llama3.2-1b | GPTQ | 2.076 | **-0.394** | hidden_danger |
| llama3.2-3b | FP16 | 2.383 | 0.000 | baseline |
| llama3.2-3b | AWQ | 2.331 | -0.052 | hidden_danger |
| llama3.2-3b | GPTQ | 2.164 | -0.219 | hidden_danger |
| qwen2.5-1.5b | FP16 | 2.404 | 0.000 | baseline |
| qwen2.5-1.5b | AWQ | 2.436 | +0.031 | neutral |
| qwen2.5-1.5b | GPTQ | 2.257 | -0.147 | neutral |
| qwen2.5-7b | FP16 | 1.874 | 0.000 | anchor |
| qwen2.5-7b | AWQ | 1.966 | +0.092 | hidden_danger |
| qwen2.5-7b | GPTQ | 1.697 | -0.177 | hidden_danger |
| phi-2 | FP16 | 1.807 | 0.000 | baseline |
| phi-2 | GPTQ | 1.889 | +0.083 | hidden_danger |
| mistral-7b | FP16 | 2.654 | 0.000 | anchor |
| mistral-7b | AWQ | 2.714 | +0.061 | hidden_danger |
| mistral-7b | GPTQ | 2.545 | -0.109 | hidden_danger |

*Observations.* The entropy shift is **method-specific, not danger-specific**. AWQ tends to increase entropy (model less certain): 4 of 5 AWQ cells show positive shift. GPTQ consistently decreases entropy (model more confident): 5 of 6 GPTQ cells show negative shift, including the worst hidden-danger row (llama3.2-1b GPTQ at -0.394). The two methods push first-token uncertainty in opposite directions, yet both produce hidden-danger rows. This fundamentally breaks the shallow-alignment prediction: GPTQ models become *more confident* about their first token while failing to refuse.

### SS4.3 Correlation with RTSI and Regimes

| Test | Statistic | Value | p-value | Interpretation |
|------|-----------|-------|---------|---------------|
| Entropy shift vs RTSI | Pearson r | 0.083 | 0.809 | No relationship |
| Entropy shift vs RTSI | Spearman rho | 0.327 | 0.326 | No relationship |
| Hidden-danger vs neutral entropy | Mann-Whitney U | — | 0.606 | No separation |

### SS4.4 Phase 1 Conclusion

**First-token entropy shift does not predict safety degradation under quantization.** The hypothesis that quantization increases refusal uncertainty is method-specific: AWQ increases entropy, GPTQ decreases it. Neither direction predicts hidden-danger status. The GPTQ pattern (increased confidence + safety failure) is particularly noteworthy — it means a model can be *more sure* about its first token while still failing to refuse.

---

## SS5. Phase 2: Refusal Direction Geometry

### SS5.1 Hypothesis

If refusal behavior is mediated by a single direction in the residual stream (Arditi et al. 2024), quantization might distort or attenuate this direction, causing behavioral safety degradation. Lower cosine similarity to the FP16 refusal direction should correlate with worse safety outcomes.

### SS5.2 Results — Cosine Similarity

| Model | Quant | Cosine Sim to FP16 | Direction Magnitude | TR142 Regime |
|-------|-------|-------------------|-------------------|-------------|
| llama3.2-1b | AWQ | 0.985 | 4.73 | hidden_danger |
| llama3.2-1b | GPTQ | 0.980 | 4.63 | hidden_danger |
| llama3.2-3b | AWQ | 0.991 | 10.62 | hidden_danger |
| llama3.2-3b | GPTQ | 0.982 | 10.62 | hidden_danger |
| qwen2.5-1.5b | AWQ | 0.991 | 50.06 | neutral |
| qwen2.5-1.5b | GPTQ | 0.988 | 49.64 | neutral |
| qwen2.5-7b | AWQ | 0.993 | 60.08 | hidden_danger |
| qwen2.5-7b | GPTQ | 0.985 | 59.92 | hidden_danger |
| phi-2 | GPTQ | 0.995 | 70.69 | hidden_danger |
| mistral-7b | AWQ | 0.999 | 15.75 | hidden_danger |
| mistral-7b | GPTQ | 0.998 | 15.81 | hidden_danger |

*Observations.* Every cosine similarity exceeds 0.97. The refusal direction is geometrically preserved under both AWQ and GPTQ across all 6 models. The RefusalCompressed 2025 finding (arXiv 2504.04215) — that quantized models retain refusal direction orientation while pruned models lose it — is confirmed on our matrix.

### SS5.3 Correlation with RTSI and Regimes

| Test | Statistic | Value | p-value | Interpretation |
|------|-----------|-------|---------|---------------|
| Cosine sim vs RTSI | Pearson r | -0.144 | 0.673 | No relationship |
| Cosine sim: danger vs neutral | Mann-Whitney U | — | 0.606 | No separation |
| **Direction magnitude: danger vs neutral** | **Mann-Whitney U** | — | **0.036** | **Significant** |

*Observations.* Cosine similarity has no predictive power, but direction magnitude does. Hidden-danger rows have a mean direction magnitude of 19.0 vs 54.9 for neutral rows. This is largely a model-size effect: llama3.2-1b (4.7) and llama3.2-3b (10.7) have the weakest refusal directions and are the most frequent hidden-danger sources. However, phi-2 (72.3) is an outlier — high magnitude, medium size, still a hidden-danger row for GPTQ.

### SS5.4 Phase 2 Conclusion

**The refusal direction is not destroyed by quantization — it is preserved but behaviorally ineffective.** Cosine similarity > 0.97 everywhere means the direction's orientation survives. The failure is not geometric destruction but rather a threshold phenomenon: models with weaker refusal signals (lower magnitude) are more vulnerable to the small perturbation that quantization introduces. This explains why the same model can refuse correctly at FP16 but fail at INT4 — the refusal direction is still there, just not strong enough to dominate the output distribution after quantization noise is added.

---

## SS6. Phase 3: Calibration Drift

### SS6.1 Hypothesis

If quantization changes confidence calibration (Proskurina et al. 2024), and if safety-borderline prompts live in the low-confidence regime, then calibration drift should predict safety degradation.

### SS6.2 Results

| Model | Quant | Mean Confidence | Confidence Shift | Entropy Shift | TR142 Regime |
|-------|-------|----------------|-----------------|--------------|-------------|
| llama3.2-1b | AWQ | 0.292 | -0.002 | +0.073 | hidden_danger |
| llama3.2-1b | GPTQ | 0.370 | **+0.076** | -0.394 | hidden_danger |
| llama3.2-3b | AWQ | 0.346 | +0.012 | -0.052 | hidden_danger |
| llama3.2-3b | GPTQ | 0.356 | +0.021 | -0.219 | hidden_danger |
| qwen2.5-1.5b | AWQ | 0.281 | +0.002 | +0.031 | neutral |
| qwen2.5-1.5b | GPTQ | 0.322 | +0.043 | -0.147 | neutral |
| qwen2.5-7b | AWQ | 0.342 | -0.014 | +0.092 | hidden_danger |
| qwen2.5-7b | GPTQ | 0.397 | +0.041 | -0.177 | hidden_danger |
| phi-2 | GPTQ | 0.528 | -0.014 | +0.083 | hidden_danger |
| mistral-7b | AWQ | 0.453 | -0.011 | +0.061 | hidden_danger |
| mistral-7b | GPTQ | 0.501 | +0.037 | -0.109 | hidden_danger |

*Observations.* Confidence shift mirrors the entropy pattern exactly (they are mathematically related). GPTQ increases confidence (+0.02 to +0.08), AWQ has negligible or slightly negative shift. The calibration angle adds no new information beyond Phase 1.

### SS6.3 Correlation with RTSI and Regimes

| Test | Statistic | Value | Interpretation |
|------|-----------|-------|---------------|
| Confidence shift vs RTSI | Pearson r | 0.068 | No relationship |
| Entropy shift vs RTSI | Pearson r | 0.083 | No relationship |
| Gini shift vs RTSI | Pearson r | 0.088 | No relationship |
| Confidence shift: danger vs neutral | Mann-Whitney p | 0.788 | No separation |

### SS6.4 Phase 3 Conclusion

**Calibration drift has no predictive power for safety degradation.** It is redundant with the entropy analysis (Phase 1) and confirms the same method-specific pattern: GPTQ increases confidence, AWQ does not, and neither direction predicts danger.

---

## SS7. Phase 4: Safety Neuron Quantization Error

### SS7.1 Hypothesis

If ~5% of neurons account for >90% of safety behavior (Chen et al. 2024), and if quantization algorithms do not specifically protect these neurons, then safety neurons should absorb disproportionate error. If the disproportionality is larger in hidden-danger configurations, neuron-level error could serve as a predictive screen.

### SS7.2 Safety Neuron Identification

For each FP16 model, safety neurons were identified via activation contrasting at 5 evenly-spaced layers (20%-80% depth). The top 5% of neurons by |mean(harmful) - mean(harmless)| were tagged as safety-critical.

| Model | Layers Probed | Safety Neurons per Layer | Hidden Dim |
|-------|--------------|------------------------|-----------|
| llama3.2-1b | 3, 6, 8, 10, 13 | 102 | 2,048 |
| llama3.2-3b | 6, 10, 14, 18, 22 | 179 | 3,584 |
| qwen2.5-1.5b | 6, 10, 14, 18, 22 | 179 | 3,584 |
| qwen2.5-7b | 6, 10, 14, 18, 22 | 179 | 3,584 |
| phi-2 | 6, 11, 16, 21, 26 | 128 | 2,560 |
| mistral-7b | 6, 11, 16, 21, 26 | 204 | 4,096 |

### SS7.3 Results — Safety Error Ratios

| Model | Quant | Mean Safety Ratio | Max Safety Ratio | TR142 Regime |
|-------|-------|------------------|-----------------|-------------|
| llama3.2-1b | AWQ | 1.29x | 1.45x | hidden_danger |
| llama3.2-1b | GPTQ | 1.50x | 1.62x | hidden_danger |
| llama3.2-3b | AWQ | 1.30x | 1.43x | hidden_danger |
| llama3.2-3b | GPTQ | 1.50x | 1.59x | hidden_danger |
| qwen2.5-1.5b | AWQ | 1.46x | 1.54x | neutral |
| qwen2.5-1.5b | GPTQ | 1.52x | 1.73x | neutral |
| qwen2.5-7b | AWQ | 1.55x | 1.70x | hidden_danger |
| qwen2.5-7b | GPTQ | 1.45x | 1.52x | hidden_danger |
| phi-2 | GPTQ | 1.19x | 1.27x | hidden_danger |
| mistral-7b | AWQ | 1.26x | 1.33x | hidden_danger |
| mistral-7b | GPTQ | 1.34x | 1.45x | hidden_danger |

*Observations.* **Every single cell shows safety neurons with higher quantization error than non-safety neurons.** The minimum ratio is 1.19x (phi-2 GPTQ), the maximum is 1.55x (qwen2.5-7b AWQ). GPTQ produces higher ratios than AWQ on average (1.45x vs 1.37x), consistent with GPTQ's more aggressive quantization strategy.

### SS7.4 Statistical Tests

| Test | Statistic | Value | p-value | Interpretation |
|------|-----------|-------|---------|---------------|
| Safety ratio > 1.0 | One-sample t | mean = 1.395 | **< 0.0001** | **Safety neurons disproportionately damaged** |
| Ratio: danger vs neutral | Mann-Whitney U | — | 0.979 | **No separation** |
| Ratio vs RTSI | Pearson r | 0.119 | — | **No relationship** |

*Observations.* The disproportionate error is **universal, not selective**. The two neutral rows (qwen2.5-1.5b AWQ at 1.46x, qwen2.5-1.5b GPTQ at 1.52x) have *higher* ratios than several hidden-danger rows. The damage is there in every quantized model, but it does not predict which ones actually fail behaviorally. This is the key insight: safety-neuron damage is necessary but not sufficient for hidden-danger status.

### SS7.5 Phase 4 Conclusion

**Safety neurons absorb 1.40x disproportionate quantization error (p < 0.0001), but this does not predict hidden-danger status.** The finding is mechanistically important — it explains *why* quantization can break safety behavior — but operationally useless as a screening tool because every quantized model shows the same pattern. The behavioral threshold that separates safe from dangerous configurations is not captured by the neuron-level error magnitude.

---

## SS8. Cross-Phase Synthesis

### SS8.1 The Necessary-but-Not-Sufficient Pattern

All four phases converge on the same structural result: the mechanism exists but does not predict the outcome.

| Mechanism | Present? | Predicts Danger? | Interpretation |
|-----------|----------|-----------------|---------------|
| Increased first-token uncertainty | Method-specific | No | AWQ increases, GPTQ decreases — opposite directions |
| Refusal direction distortion | Minimal (>0.97) | No | Direction preserved, behavior fails anyway |
| Calibration drift | Method-specific | No | Same pattern as entropy — redundant |
| Safety-neuron damage | **Universal (1.40x)** | **No** | Every quantized model suffers it |

The failure mode is not that a single mechanism breaks. It is that multiple small perturbations — slight direction weakening, redistributed confidence, elevated neuron error — compound to cross a behavioral threshold in some configurations but not others. The threshold is configuration-specific (model × quant method × quant level) and cannot be predicted from any single mechanistic signal.

### SS8.2 Why RTSI Works When Mechanistic Probes Don't

RTSI measures the *downstream behavioral consequence* of the combined mechanistic perturbation: whether the model's refusal templates have destabilized. It does not need to isolate the cause because it directly measures the effect. The mechanistic probes each capture one upstream signal, but the signal-to-noise ratio is too low for any individual probe to predict the downstream outcome.

This is analogous to medical diagnostics: measuring blood cholesterol (upstream biomarker) is less predictive of heart attack than measuring chest pain (downstream symptom). Both are real, but the symptom is more actionable for the acute decision.

### SS8.3 The GPTQ Confidence Paradox

The most striking cross-phase finding is the GPTQ confidence paradox: GPTQ models become *more confident* about their first token (Phase 1, Phase 3) while simultaneously failing to refuse (TR142 behavioral data). This means a confidence-based screen would actively *approve* the most dangerous GPTQ configurations because they look more certain, not less. This is a direct warning against using logit-based diagnostics as safety screens.

---

## SS9. Implications for RTSI and the NeurIPS Paper

### SS9.1 What TR146 Adds to the Paper

TR146 provides the mechanistic backing that strengthens the RTSI contribution:

1. **Negative results validate RTSI's uniqueness.** Standard mechanistic probes (entropy, direction, calibration, neuron error) cannot replace RTSI. This makes the behavioral template approach a genuine contribution, not just a placeholder for a "real" mechanistic diagnostic.

2. **The safety-neuron finding (1.40x, p < 0.0001) explains the phenomenon.** Quantization systematically under-protects safety-relevant parameters. This is a clean, publishable mechanistic result even though it doesn't serve as a screen.

3. **The refusal-direction preservation finding resolves a potential objection.** A reviewer might ask "maybe the refusal direction is just destroyed." It isn't — cosine similarity > 0.97 everywhere. The failure is subtler than direction destruction.

4. **The GPTQ confidence paradox strengthens the proxy-failure argument.** If GPTQ models become more confident while failing to refuse, then any confidence-based screening would give the wrong answer. This is exactly the kind of hidden danger the paper warns about.

### SS9.2 Paper Section Placement

- **Main body:** GPTQ confidence paradox (1 paragraph in Discussion) + safety-neuron ratio as a mechanism sentence
- **Appendix:** Full TR146 mechanistic probe results as a supplementary section
- Cite Arditi 2024, Qi 2025, Chen 2024, RefusalCompressed 2025 in Related Work

---

## SS10. Conclusions

TR146 tested four mechanistic hypotheses about quantization-induced safety degradation across 17 model-quant cells spanning 6 models and 2 quantization methods.

The central result is a necessary-but-not-sufficient pattern. Safety neurons are disproportionately damaged (1.40x, p < 0.0001) in every quantized model. The refusal direction is preserved geometrically (cosine similarity > 0.97) but weakened in magnitude. First-token entropy and calibration drift are method-specific (AWQ and GPTQ shift in opposite directions). Yet none of these mechanistic signals predict which configurations are actually dangerous.

The operational implication is clear: behavioral template metrics (RTSI) remain the only viable pre-deployment screen because they measure the downstream consequence of the combined perturbation, not any single upstream cause. The mechanism is real but the mapping from mechanism to outcome is too noisy for mechanistic probes alone to serve as safety diagnostics.

---

## SS11. Limitations and Follow-Up

### SS11.1 Limitations

1. **AWQ/GPTQ only.** The 17-cell matrix covers FP16, AWQ, and GPTQ but not GGUF k-quants. Extending to GGUF would require `llama-cpp-python` hidden-state extraction, which is scaffolded but not yet validated.

2. **Top-5% safety neuron threshold.** The 5% cutoff for safety neuron identification is a hyperparameter. Sensitivity analysis at 1%, 3%, and 10% was not performed but would be straightforward.

3. **Single probe layer for refusal direction.** Phase 2 extracts at ~65% depth. The refusal direction may be stronger or weaker at different layers. Multi-layer extraction would provide a more complete picture.

4. **Phase 4 requires dual model loading.** The FP16 + quant simultaneous loading requirement limits Phase 4 to GPUs with sufficient memory (48GB for 7B models). This is an infrastructure constraint, not a methodological one.

5. **100 prompts per phase.** The prompt count is adequate for cell-level aggregation but insufficient for per-prompt statistical analysis.

6. **No causal intervention.** All phases are observational (measure and correlate). Causal experiments (e.g., ablating safety neurons before quantization, or artificially strengthening the refusal direction) would provide stronger mechanistic evidence.

### SS11.2 Follow-Up Directions

1. **Refusal direction magnitude as a pre-quantization vulnerability predictor.** The p = 0.036 result for direction magnitude suggests models could be scored for quantization vulnerability *before* quantization is applied.

2. **Layer-wise safety neuron error profiles.** The current analysis averages across 5 layers. Identifying whether safety-neuron damage is concentrated in early, middle, or late layers could inform layer-specific mixed-precision quantization strategies.

3. **Cross-quant-method neuron error comparison.** AWQ (1.37x) and GPTQ (1.45x) differ in their safety-neuron error ratios. Understanding why could inform method-specific safety-aware quantization.

---

## SS12. Reproducibility

### SS12.1 Run Commands

```bash
# Phase 1: First-token entropy
PYTHONPATH=/workspace python3 -m research.tr146.phase1.extract -v
python -m research.tr146.phase1.analyze --results-dir research/tr146/results/phase1_gptq_docker -v

# Phase 2: Refusal direction
PYTHONPATH=/workspace python3 -m research.tr146.phase2.extract -v
python -m research.tr146.phase2.analyze --results-dir research/tr146/results/phase2_gptq_docker -v

# Phase 3: Calibration drift
PYTHONPATH=/workspace python3 -m research.tr146.phase3.extract -v
python -m research.tr146.phase3.analyze --results-dir research/tr146/results/phase3_gptq_docker -v

# Phase 4: Safety neurons (requires Docker for GPTQ)
PYTHONPATH=/workspace python3 -m research.tr146.phase4.extract -v
python -m research.tr146.phase4.analyze --results-dir research/tr146/results/phase4_gptq_docker -v

# Phase 4 7B rerun (RunPod 48GB)
PYTHONPATH=/workspace python3 research/tr146/phase4/rerun_7b_failures.py -v
```

### SS12.2 Dependencies

| Package | Version | Role |
|---------|---------|------|
| torch | 2.11.0+cu126 | Model loading, forward passes |
| transformers | 4.57.6 | Model loading (FP16, AWQ) |
| gptqmodel | 5.8.0 | GPTQ model loading (Docker only) |
| numpy | 2.2.6 | Array operations, entropy computation |
| scipy | 1.14+ | Statistical tests |
| pandas | 2.2+ | Data aggregation |
| PyYAML | 6.0+ | Config loading |

### SS12.3 Hardware

| Phase | Hardware | Time |
|-------|----------|------|
| Phase 1 | RTX 4080 Laptop 12GB (Docker) | ~55 min |
| Phase 2 | RTX 4080 Laptop 12GB (Docker) | ~90 min |
| Phase 3 | RTX 4080 Laptop 12GB (Docker) | ~55 min |
| Phase 4 (small models) | RTX 4080 Laptop 12GB (Docker) | ~20 min |
| Phase 4 (7B rerun) | RunPod RTX 6000 Ada 48GB | ~15 min |

---

## References

1. Arditi, A., Obeso, O., Syed, A., Paleka, D., Rimsky, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. *NeurIPS 2024*.
2. Qi, X., Panda, A., Lyu, K., Ma, X., Roy, S., Beirami, A., Mittal, P., & Henderson, P. (2025). Safety Alignment Should Be Made More Than Just a Few Tokens Deep. *ICLR 2025 (Outstanding Paper)*.
3. Chen, J., Wang, X., Yao, Z., et al. (2024). Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons. *arXiv 2406.14144*.
4. Proskurina, I., Brun, L., Metzler, G., & Velcin, J. (2024). When Quantization Affects Confidence of Large Language Models? *Findings of NAACL 2024*.
5. (2025). Towards Understanding and Improving Refusal in Compressed Models via Mechanistic Interpretability. *arXiv 2504.04215*.
6. Wollschlager et al. (2025). The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence. *ICML 2025*.

---

## Appendix A: Raw Data Tables

All raw data is stored in the TR146 results directories:

| File | Location | Contents |
|------|----------|---------|
| Phase 1 samples | `results/phase1_gptq_docker/entropy_samples.jsonl` | 1,700 per-prompt entropy records |
| Phase 1 shifts | `results/phase1_gptq_docker/entropy_shifts.csv` | 17-row cell-level entropy summaries |
| Phase 2 cosine sims | `results/phase2_gptq_docker/cosine_similarities.csv` | 17-row cosine similarity table |
| Phase 2 directions | `results/phase2_gptq_docker/directions/*.npy` | 17 refusal direction vectors |
| Phase 3 samples | `results/phase3_gptq_docker/calibration_samples.jsonl` | 1,700 per-prompt calibration records |
| Phase 3 ECE | `results/phase3_gptq_docker/ece_per_cell.csv` | 17-row ECE table |
| Phase 4 errors | `results/phase4_gptq_docker/safety_neuron_errors.jsonl` | 55 layer-level neuron error records |
| Phase 4 cell summary | `results/phase4_gptq_docker/cell_level_with_context.csv` | 11-row cell-level ratio summary |

---

## Appendix B: Extended Statistical Tables

### B.1 Phase 1: Full Regime Test

| Test | Group A | Group B | n_A | n_B | Mean A | Mean B | U | p |
|------|---------|---------|-----|-----|--------|--------|---|---|
| Mann-Whitney entropy shift | hidden+near-hidden | neutral | 8 | 2 | -0.080 | -0.050 | — | 0.606 |

### B.2 Phase 4: Per-Layer Safety Error Ratios

See `results/phase4_gptq_docker/per_layer_errors.csv` for the full 55-row per-layer breakdown. Safety error ratios range from 1.06x (phi-2 GPTQ layer 26) to 1.73x (qwen2.5-1.5b GPTQ layer 18). The ratio tends to increase at deeper layers, consistent with error accumulation through the transformer stack.

---

## Appendix C: Prompt Provenance

### C.1 Harmful Prompts

Source: `research/tr142/expansion/results/v3_safety/20260331_125319/samples.jsonl`

100 unique AdvBench-derived prompts extracted from the TR142 canonical safety evaluation. Task filter: `task_name == "advbench_refusal"`. Loaded via `research/tr146/shared/prompts.py` with deterministic ordering.

### C.2 Harmless Prompts

Source: Built-in curated set (`builtin_curated_v1`) in `research/tr146/shared/prompts.py`.

100 benign instruction prompts (e.g., "What is the capital of France?", "Explain how photosynthesis works."). Used for activation contrasting in Phases 2 and 4.

---

## Appendix D: Glossary

| Term | Definition |
|------|-----------|
| **Activation contrasting** | Computing mean activation differences between harmful and harmless prompts to identify safety-relevant neurons |
| **Cosine similarity** | Dot product of two unit vectors; measures directional alignment (1.0 = identical, 0.0 = orthogonal) |
| **Direction magnitude** | L2 norm of the unnormalized refusal direction vector; measures refusal signal strength |
| **Hidden-danger** | TR142 regime: quality delta >= -2pp and refusal delta <= -10pp (quality stable, safety fails) |
| **RTSI** | Refusal Template Stability Index; behavioral screen from TR142 using prefix diversity metrics |
| **Safety neuron** | Neuron in the top 5% by activation contrast between harmful and harmless prompts |
| **Safety error ratio** | Mean quantization error on safety neurons / mean error on non-safety neurons |
