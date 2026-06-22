# Technical Report 164 V5: Evidence Closure for Serving-Stack Physics

**Status:** V5 evidence-closure report. This report updates the TR164 V4 amortization model with three reviewer-facing closure layers: (1) explicit predictor baselines and failure-mode accounting for the 672-cell static-batch grid, (2) served SGLang validation against the static-knee upper-bound claim, and (3) true-decode weight-axis matrices on a single H100 for Qwen2.5, Mistral, and Gemma 3. The report is written from the generated artifacts, not from manual spreadsheet transcription.

**Primary source artifacts.**

- Static-grid predictor audit: `research/tr164/v5/v5_predictor_baselines.json`
- Static-grid failure audit: `research/tr164/v5/v5_failure_audit.json`
- SGLang served validation: `research/tr164/v5/v5_sglang_served_summary.json`
- Weight-axis analyzer: `research/tr164/analyze_weight_axis_decode.py`
- Weight-axis summary: `research/tr164/weight_axis_decode/weight_axis_decode_summary.json`
- Weight-axis per-cell table: `research/tr164/weight_axis_decode/weight_axis_decode_cells.csv`
- Weight-axis per-batch table: `research/tr164/weight_axis_decode/weight_axis_decode_batch_measurements.csv`
- Weight-axis per-replicate table: `research/tr164/weight_axis_decode/weight_axis_decode_replicates.csv`

**One-paragraph V5 verdict.** V5 strengthens the TR164 paper package materially, but it also narrows the strongest claim. The cleanest positive evidence is now cross-family: Qwen2.5 14B->32B succeeds in 3/3 broad matched contexts and 2/2 confirm contexts, Mistral 7B->24B succeeds in 3/3 matched contexts, and Gemma 3 12B->27B succeeds in 3/3 matched contexts. The predictor audit shows that Ck/W plus a per-GPU alpha is the best static-grid predictor (finite-only Spearman 0.8779; censored-as-128 Spearman 0.8462), while KV-only and context-only baselines are weaker or less explanatory. The served SGLang validation supports the static-knee-as-upper-bound framing in 8/8 ok cells. The caveat is important: Gemma 3 is not monotone across all three sizes. Its 4B->12B contrast is inverted in 0/3 contexts, and 4B->27B is mixed at 2/3, so V5 should not claim a universal "larger model always later knee" law.

---

## 1. Research Question

TR164 V4 established a static-batch amortization model for vLLM decode:

`eta(B) = (1 + r) / (1 + B r)`, with `r = C k / W`,

where `C` is context length, `k` is KV bytes per token, and `W` is weight bytes. The V4 claim was that continuous batching stops amortizing when per-request KV traffic becomes large relative to the shared weight read.

V5 asks four closure questions:

1. Does Ck/W still beat simpler baselines when the predictor audit is made explicit?
2. Where does the static model fail, and are failures concentrated in interpretable regimes?
3. Is the static knee a useful upper bound for served closed-loop behavior under SGLang?
4. Does moving the weight axis inside model families shift the true-decode knee in the predicted direction?

The fourth question is the most reviewer-facing addition. V4's main static grid used mostly 7B/8B models, so W moved weakly. V5 adds larger same-family contrasts on a fixed H100 serving protocol.

---

## 2. Evidence Chain

The unit of analysis depends on the layer:

- Static predictor audit: one efficiency curve per `(model, GPU, decode length, precision, context)` in the 672-cell V4 grid. There are 96 curves. Each curve contains batch sizes 1, 2, 4, 8, 16, 32, 64, with three timed repetitions per cell.
- Failure audit: the same 96 static curves, classified by fit quality and failure morphology.
- SGLang served validation: one served closed-loop cell per `(backend, model, GPU, precision, context, prompt regime)` in the compact validation slice. There are 8 ok cells.
- Weight-axis true-decode runs: one row per `(model, context)` for fixed GPU/backend/precision/decode protocol. The analyzer currently contains 28 cell rows across Qwen smoke/full/confirm, Mistral smoke/matrix, and Gemma 3 smoke/matrix, plus 263 per-batch rows and 708 per-replicate rows.

Rows become claims as follows:

1. Per-replicate streamed decode timings are aggregated into per-batch per-request decode throughput.
2. Per-batch throughput is normalized by the batch-1 throughput to form `eta(B)`.
3. The knee is the first batch/concurrency where `eta(B) < 0.65`. Continuous knees use log2-batch interpolation at the 0.65 crossing; censored curves are marked when the last tested batch remains above threshold.
4. Predictor baselines are ranked against the observed continuous knee using Spearman rank correlation and, where meaningful, log2-space error.
5. Family contrasts compare the continuous knee of the larger model to the smaller model at matched context.

The V5 report uses the same tau threshold as V4: `eta = 0.65`.

---

## 3. Headline Results

### 3.1 Predictor baselines

The V5 predictor audit makes the competing explanations explicit.

| Evaluation | Curves | Censored | Best predictor | Spearman | Pearson log2 | MAE log2 | Median factor error |
|---|---:|---:|---|---:|---:|---:|---:|
| Censored as 128 | 96 | 31 | Ck/W + per-GPU alpha | 0.8462 | 0.8497 | 0.8532 | 1.806 |
| Finite only | 65 | 0 | Ck/W + per-GPU alpha | 0.8779 | 0.8414 | 0.7015 | 1.626 |

The closed-form Ck/W predictor remains strong without the alpha correction:

| Evaluation | Ck/W closed-form Spearman | Context-only Spearman | VRAM ceiling Spearman | KV-traffic-only Spearman | KV-bytes-only Spearman |
|---|---:|---:|---:|---:|---:|
| Censored as 128 | 0.8138 | 0.7890 | 0.7537 | 0.7210 | 0.0941 |
| Finite only | 0.8384 | 0.8377 | 0.8321 | 0.6962 | 0.1473 |

Interpretation: the model is not merely "longer context means earlier knee." Context-only is competitive on finite curves because context is the largest lever, but it cannot explain model-family movement at fixed context. KV-only is weaker because it omits the shared weight-read term. The per-GPU alpha version improves calibration without changing the core mechanism.

### 3.2 Failure audit

The failure audit prevents the model from being overstated as a universal rational curve.

| Audit item | Count |
|---|---:|
| Total curves | 96 |
| Low-fit curves, `R^2 < 0.70` | 14 |
| Negative-`R^2` curves | 4 |

Failure categories are concentrated:

| Category | Count |
|---|---:|
| non_monotone | 13 |
| tested_beyond_vram_ceiling | 10 |
| long_context_long_decode_tail | 6 |
| negative_r2 | 4 |
| smooth_low_fit | 1 |

The important read is that most failures are not random noise. They occur where the experiment pushes beyond the smooth amortization regime: long context, long decode, H100 wave-batching behavior, or tested batches beyond the VRAM ceiling. V5 therefore supports a two-regime framing: the Ck/W model is a strong predictor in the smooth decode-amortization regime, while capacity-tail and non-monotone curves should be labeled rather than forced into the model.

### 3.3 Served SGLang validation

The compact SGLang served validation ran 8/8 ok cells with zero errors. The metric is the served knee: the smallest concurrency `N` where parallel efficiency falls below 0.65, with `None` meaning no crossing by `N = 64`.

| Served validation claim | Result |
|---|---:|
| SGLang served knee <= static knee | 8/8 |
| Distinct-prompt served knee <= static knee | 4/4 |
| Shared-prefix served knee within one static ladder step | 2/4 |
| SGLang equal to matched vLLM served knee | 3 |
| SGLang lower than matched vLLM served knee | 5 |
| SGLang higher than matched vLLM served knee | 0 |

Cell-level served table:

| Model | Context | Regime | Static knee | SGLang knee | vLLM knee | SGLang vs static | SGLang vs vLLM |
|---|---:|---|---:|---:|---:|---|---|
| llama3.1-8b | 512 | distinct | >64 | 12 | 12 | lower | equal |
| llama3.1-8b | 512 | shared | >64 | 48 | >64 | lower | lower |
| llama3.1-8b | 8192 | distinct | 16 | 4 | 4 | lower | equal |
| llama3.1-8b | 8192 | shared | 16 | 16 | 32 | equal | lower |
| qwen2.5-7b | 512 | distinct | >64 | 12 | 16 | lower | lower |
| qwen2.5-7b | 512 | shared | >64 | 48 | >64 | lower | lower |
| qwen2.5-7b | 8192 | distinct | 32 | 4 | 4 | lower | equal |
| qwen2.5-7b | 8192 | shared | 32 | 24 | 32 | lower | lower |

Interpretation: this supports the paper's protocol distinction. The static-batch knee is not a served latency measurement; it is an amortization ceiling. In the served SGLang slice, real scheduling reaches the knee at or before the static ceiling in every ok cell.

### 3.4 Weight-axis true-decode matrices

The weight-axis protocol fixes the serving setup and moves model size inside families. It uses vLLM OpenAI streaming, one H100, fp8 weights, fp16 KV cache, 64-token decode, exact prompt contexts, and the same `eta = 0.65` knee rule. Timing starts after the first non-empty streamed token from every request, so the measurement is a true-decode window rather than prefill or TTFT.

Generated table sizes:

| Table | Rows |
|---|---:|
| `weight_axis_decode_cells.csv` | 28 |
| `weight_axis_decode_pair_contrasts.csv` | 17 |
| `weight_axis_decode_batch_measurements.csv` | 263 |
| `weight_axis_decode_replicates.csv` | 708 |

Cost estimate from the analyzer:

| Run group | Measured H100 seconds |
|---|---:|
| Qwen smoke | 243.2 |
| Qwen full | 1991.9 |
| Qwen confirm | 1295.7 |
| Mistral smoke | 666.2 |
| Mistral matrix | 1700.4 |
| Gemma 3 smoke | 991.9 |
| Gemma 3 matrix | 1977.1 |
| Total H100 seconds | 8866.4 |
| Total analyzer-estimated H100 cost | USD 9.73 |

The cost estimate uses measured function load time plus per-context elapsed time and excludes small CPU/memory overhead.

---

## 4. Weight-Axis Results by Family

### 4.1 Qwen2.5: clean positive result

Qwen2.5 provides the cleanest same-family test in the V5 data. The broad run compares Qwen2.5-14B to Qwen2.5-32B at three contexts; the confirm run repeats the two most operationally important long-context cells with denser ladders.

| Run | Context | 14B knee | 32B knee | Observed ratio | Predicted ratio | Observed larger later |
|---|---:|---:|---:|---:|---:|---|
| qwen_full | 2048 | 26.0711 | 31.4907 | 1.2079 | 1.6157 | yes |
| qwen_full | 8192 | 8.7606 | 9.7646 | 1.1146 | 1.5060 | yes |
| qwen_full | 32000 | 3.5514 | 3.7923 | 1.0678 | 1.2994 | yes |
| qwen_confirm | 8192 | 8.4162 | 10.8535 | 1.2896 | 1.5060 | yes |
| qwen_confirm | 32000 | 3.4990 | 4.0011 | 1.1435 | 1.2994 | yes |

Success rates:

| Contrast | Success |
|---|---:|
| Qwen broad 14B->32B | 3/3 |
| Qwen confirm 14B->32B | 2/2 |

Qwen full-run predictor ladder:

| Predictor | Spearman |
|---|---:|
| Ck/W closed-form | 0.9524 |
| context only 1/C | 0.8063 |
| KV traffic only 1/(Ck) | 0.7857 |
| W only | -0.0126 |

Interpretation: at fixed context and fixed serving protocol, Qwen2.5-32B keeps the true-decode amortization knee later than Qwen2.5-14B even though the larger model has larger KV bytes per token. This is exactly the contrast that KV-only should get wrong and Ck/W should get right.

### 4.2 Mistral: second clean positive family

Mistral was added because Gemma2 and Gemma4 were blocked by clean serving support in the pinned vLLM/H100 environment. The Mistral pair is Mistral-7B-Instruct-v0.3 against Mistral-Small-24B-Instruct-2501 on the same H100/fp8 protocol.

| Context | 7B knee | 24B knee | Observed ratio | Predicted ratio | Observed larger later |
|---:|---:|---:|---:|---:|---|
| 2048 | 24.4343 | 38.2735 | 1.5664 | 2.4485 | yes |
| 8192 | 8.7898 | 11.3806 | 1.2948 | 2.1254 | yes |
| 32000 | 3.4922 | 4.1799 | 1.1969 | 1.6037 | yes |

Success rate:

| Contrast | Success |
|---|---:|
| Mistral 7B->24B | 3/3 |

Mistral predictor ladder:

| Predictor | Spearman |
|---|---:|
| Ck/W closed-form | 1.0000 |
| context only 1/C | 0.9562 |
| KV traffic only 1/(Ck) | 0.8286 |
| W only | 0.2928 |

Interpretation: Mistral independently reproduces the weight-axis direction. The larger model's knee is later at every matched context, and Ck/W ranks the six finite knees perfectly in this small matrix.

### 4.3 Gemma 3: useful stress test, not a clean monotone replication

Gemma 3 ran successfully across 4B, 12B, and 27B. This is important because earlier Gemma attempts had environment-level blockers: Gemma2 hit the H100 flash-attention softcapping path, and Gemma4 was not recognized by the pinned Transformers/vLLM stack. Gemma 3 avoids those serving failures and gives a real data point.

Gemma 3, however, is mixed:

| Contrast | Success |
|---|---:|
| Gemma 3 4B->12B | 0/3 |
| Gemma 3 12B->27B | 3/3 |
| Gemma 3 4B->27B | 2/3 |
| All Gemma 3 pairwise contrasts | 5/9 |

Cell-level Gemma 3 contrasts:

| Contrast | Context | Small knee | Large knee | Observed ratio | Predicted ratio | Observed larger later |
|---|---:|---:|---:|---:|---:|---|
| 4B->12B | 2048 | 23.5190 | 21.2198 | 0.9022 | 1.0032 | no |
| 4B->12B | 8192 | 15.1243 | 12.0466 | 0.7965 | 1.0022 | no |
| 4B->12B | 32000 | 6.7530 | 5.3175 | 0.7874 | 1.0010 | no |
| 12B->27B | 2048 | 21.2198 | 28.6155 | 1.3485 | 1.6247 | yes |
| 12B->27B | 8192 | 12.0466 | 16.1607 | 1.3415 | 1.4231 | yes |
| 12B->27B | 32000 | 5.3175 | 6.2294 | 1.1715 | 1.1880 | yes |
| 4B->27B | 2048 | 23.5190 | 28.6155 | 1.2167 | 1.6299 | yes |
| 4B->27B | 8192 | 15.1243 | 16.1607 | 1.0685 | 1.4262 | yes |
| 4B->27B | 32000 | 6.7530 | 6.2294 | 0.9225 | 1.1891 | no |

Gemma 3 predictor ladder:

| Predictor | Spearman |
|---|---:|
| Ck/W closed-form | 0.9167 |
| context only 1/C | 0.9487 |
| KV traffic only 1/(Ck) | 0.8833 |
| W only | 0.0527 |

Interpretation: Gemma 3 should be reported as a stress test. The 12B->27B contrast agrees cleanly with the weight-axis prediction, and the rank correlation is still high. But 4B->12B is inverted at every context, and 4B->27B fails at 32K. The likely paper framing is that Ck/W captures a major architectural pressure, not that model size alone gives a universal monotone knee ordering. Gemma 3's attention configuration and implementation details remain material.

---

## 5. What V5 Establishes

### Established findings

1. **Ck/W remains the best mechanistic predictor in the static grid.** With per-GPU alpha, it is the top predictor under both finite-only and censored-as-128 evaluations. The closed-form version remains strong and beats KV-only.
2. **The static model's failures are interpretable.** Low-fit curves are concentrated in non-monotone, tested-beyond-capacity, long-context, and long-decode regimes.
3. **The static knee is a useful served upper bound in the compact SGLang validation.** SGLang served knees are at or below the static knee in 8/8 ok cells.
4. **Weight-axis movement is real in at least two independent families.** Qwen2.5 and Mistral both show the larger model retaining a later true-decode knee at every matched context.
5. **KV-only is not sufficient.** In the Qwen and Mistral family contrasts, larger models have larger or non-smaller KV traffic, yet the larger model often has a later knee because W also moved.

### Partial findings

1. **Gemma 3 supports the mechanism only partially.** Ck/W ranks Gemma 3 knees well overall, and 12B->27B is clean, but 4B->12B is inverted and 4B->27B is mixed.
2. **Cross-family rank correlation is not the cleanest metric.** The combined Qwen/Mistral/Gemma 3 finite set has Ck/W Spearman 0.8142, but context-only is higher at 0.8948 because context dominates the pooled matrix. The better claim is within-family, fixed-context movement, not pooled cross-family monotonicity.
3. **Served validation is compact, not exhaustive.** SGLang covers 8 ok cells in this closure run. It supports the upper-bound framing but does not replace the full cross-backend V2/V3 served corpus.

### Negative findings and non-claims

1. **Do not claim "larger model always shifts the knee later."** Gemma 3 4B->12B directly falsifies that simple statement.
2. **Do not claim the rational curve fits every tested regime.** Four curves have negative `R^2`, and 14/96 are low-fit under `R^2 < 0.70`.
3. **Do not claim static-batch throughput is served latency.** V5 supports the static knee as an upper bound or ceiling, not as a direct served operating point.
4. **Do not claim Gemma2/Gemma4 evidence.** Gemma2 and Gemma4 were not cleanly usable under the pinned H100/vLLM stack. Gemma 3 is the valid Gemma evidence in V5.

---

## 6. Paper-Ready Claim Ladder

### Strong claims

- "Across the 672-cell static grid, Ck/W with a per-GPU alpha is the strongest predictor of the observed amortization knee among the tested baselines, with finite-only Spearman 0.8779 and censored-as-128 Spearman 0.8462."
- "Static-knee failures concentrate in identifiable non-smooth regimes: 14/96 low-fit curves, mostly non-monotone and often beyond the VRAM ceiling or in the long-context long-decode tail."
- "In the compact SGLang served validation, every served knee is at or below the corresponding static knee, supporting the static knee as an upper-bound surface rather than a served-latency measurement."
- "In true-decode H100 runs, Qwen2.5-32B shifts the knee later than Qwen2.5-14B in 5/5 matched-context comparisons across broad and confirm runs."
- "In true-decode H100 runs, Mistral-Small-24B shifts the knee later than Mistral-7B in 3/3 matched-context comparisons."

### Claims with caveats

- "Gemma 3 mostly preserves the rank-level Ck/W signal, but it is not monotone across all sizes." Use the exact 5/9 pairwise result.
- "The V5 family evidence supports a weight-axis effect." Pair this with the caveat that model-family architecture and implementation details remain material.
- "Context-only is strong but incomplete." It is competitive in static and pooled matrices because context dominates, but it cannot explain fixed-context model-size shifts.

### Forbidden claims

- "The model is universal."
- "Larger models always have later knees."
- "Gemma validates the same monotone story as Qwen and Mistral."
- "Static-batch knee equals served latency knee."
- "KV traffic alone explains the surface."

---

## 7. Reproducibility Notes

The V5 weight-axis runs used Modal H100 execution through `research/tr164/modal_weight_axis_decode.py`. The successful matrix modes were:

- `mistral_smoke`
- `mistral_matrix`
- `gemma3_smoke`
- `gemma3_matrix`

The known blocked Gemma modes are documented by run artifacts:

- Gemma2 smoke rows failed at request time because the H100 flash-attention build did not support tanh softcapping in the pinned vLLM path.
- Gemma4 smoke rows failed at startup because the pinned Transformers/vLLM stack did not recognize the `gemma4` / `gemma4_unified` architectures.

Those are environment-support facts, not negative model results.

The analysis script `research/tr164/analyze_weight_axis_decode.py` emits:

- `weight_axis_decode_summary.json`
- `weight_axis_decode_cells.csv`
- `weight_axis_decode_pair_contrasts.csv`
- `weight_axis_decode_batch_measurements.csv`
- `weight_axis_decode_replicates.csv`
- `WEIGHT_AXIS_DECODE_FINDINGS.md`

The report numbers above trace to those files or to the V5 JSON files listed in the source-artifact block.

---

## 8. Bottom Line

V5 is worth integrating into the paper. It does not make TR164 a universal law paper, and the Gemma 3 result should stop the draft from overclaiming. What it does provide is exactly the reviewer-facing closure the V4-only version lacked: explicit baseline comparison, explicit failure accounting, a served-backend upper-bound check, and two clean independent weight-axis families plus one useful stress-test family. The right final paper claim is therefore:

> Continuous-batching breakdown is governed primarily by the ratio of per-request KV traffic to shared weight traffic. Ck/W predicts the static amortization knee better than KV-only baselines, served knees fall at or below the static ceiling in compact validation, and true-decode weight-axis shifts reproduce cleanly in Qwen2.5 and Mistral while Gemma 3 exposes the remaining architecture-specific boundary conditions.
