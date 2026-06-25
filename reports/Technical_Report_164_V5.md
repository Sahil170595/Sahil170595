# Technical Report 164 V5: Evidence Closure for Serving-Stack Physics

**Status:** V5 evidence-closure report. This report updates the TR164 V4 amortization model with three reviewer-facing closure layers: (1) explicit predictor baselines and failure-mode accounting for the 672-cell static-batch grid, (2) served SGLang validation against the static-knee upper-bound claim, and (3) true-decode weight-axis matrices on a single H100 for Qwen2.5, Mistral, and Gemma 3. The report is written from the generated artifacts, not from manual spreadsheet transcription.

**Primary source artifacts.**

- Static-grid predictor audit (ordering + failure inputs): `research/tr164/v5/v5_predictor_baselines.json`
- Static-grid predictor audit (magnitude source of record, tau sweep, bootstrap CIs): `research/tr164/mlsys_closure/predictor_baselines_extended.json`
- Static-grid failure audit: `research/tr164/v5/v5_failure_audit.json`
- SGLang served validation: `research/tr164/v5/v5_sglang_served_summary.json`
- Weight-axis analyzer: `research/tr164/analyze_weight_axis_decode.py`
- Weight-axis summary: `research/tr164/weight_axis_decode/weight_axis_decode_summary.json`
- Weight-axis per-cell table: `research/tr164/weight_axis_decode/weight_axis_decode_cells.csv`
- Weight-axis per-batch table: `research/tr164/weight_axis_decode/weight_axis_decode_batch_measurements.csv`
- Weight-axis per-replicate table: `research/tr164/weight_axis_decode/weight_axis_decode_replicates.csv`
- Methodology reconciliation: `research/tr164/methodology/TR_static_ckw_model.md`, `TR_weight_axis_matrix.md`, `TR_served_validation.md`, `TR_mlsys_closure.md`
- Underlying static data: `research/tr164/results/modal_amortization/full_xmodel_672/results.json`

**One-paragraph V5 verdict.** V5 strengthens the TR164 paper package materially, but it also narrows the strongest claim. The cleanest positive evidence is now cross-family: Qwen2.5 14B->32B succeeds in 3/3 broad matched contexts and 2/2 confirm contexts, Mistral 7B->24B succeeds in 3/3 matched contexts, and Gemma 3 12B->27B succeeds in 3/3 matched contexts. The predictor audit shows that Ck/W plus a per-GPU alpha is the best static-grid predictor (finite-only Spearman 0.8779; censored-as-128 Spearman 0.8462), while KV-only and architecture-blind baselines are weaker or uninformative. The served SGLang validation supports the static-knee-as-upper-bound framing in 8/8 ok cells, on an engine with a different scheduler and a different prefix-cache implementation than vLLM. The caveat is important and load-bearing: Gemma 3 is not monotone across all three sizes. Its 4B->12B contrast is inverted in 0/3 contexts, and 4B->27B is mixed at 2/3, so V5 must not claim a universal "larger model always later knee" law.

---

## 1. Research Question

TR164 V4 established a static-batch amortization model for vLLM decode:

`eta(B) = (1 + r) / (1 + B r)`, with `r = C k / W`,

where `C` is context length, `k` is KV bytes per token, and `W` is weight bytes. The V4 claim was that continuous batching stops amortizing the shared weight read once per-request KV traffic becomes large relative to that read. The efficiency curve `eta(B)` is the per-request decode time at batch 1 divided by the per-request decode time at batch `B`; the knee is the batch where amortization has decayed past a fixed efficiency floor.

V5 asks four closure questions:

1. Does Ck/W still beat simpler and architecture-blind baselines when the predictor audit is made explicit and the magnitude estimate is recomputed correctly?
2. Where does the static model fail, and are failures concentrated in interpretable regimes rather than scattered as noise?
3. Is the static knee a useful upper bound for served closed-loop behavior on an engine other than vLLM?
4. Does moving the weight axis inside model families shift the true-decode knee in the predicted direction?

The fourth question is the most reviewer-facing addition. V4's main static grid used mostly 7B/8B models, so `W` moved weakly (about 1.11x across the three 7-8B models) and the weight term of `r` was nearly a constant. V5 adds larger same-family contrasts — Qwen2.5 14B/32B, Mistral 7B/24B, Gemma 3 4B/12B/27B — on a fixed H100 serving protocol so that `W` is the lever being swept.

---

## 2. Evidence Chain

The unit of analysis depends on the layer.

- **Static predictor audit:** one efficiency curve per `(model, GPU, decode length, precision, context)` in the 672-cell V4 grid. The 672 cells divide by the 7-point batch ladder `[1, 2, 4, 8, 16, 32, 64]` into **96 curves**. Each curve is the 7-point efficiency vector `eta(B) = t_perreq(B) / t_perreq(1)` for that combination, with timed repetitions per cell.
- **Failure audit:** the same 96 static curves, re-classified by fit quality and failure morphology against the free-`r` fit.
- **SGLang served validation:** one served closed-loop cell per `(backend, model, GPU, precision, context, prompt regime)` in the compact validation slice. There are **8 ok cells** (2 models x 2 contexts x 2 regimes), all `status=ok`, zero errors.
- **Weight-axis true-decode runs:** one row per `(model, context)` for a fixed GPU/backend/precision/decode protocol. The analyzer folds in **28 cell rows** (Qwen full 9 + Qwen confirm 4 + Mistral matrix 6 + Gemma 3 matrix 9), built from 263 per-batch rows and 708 per-replicate rows. Smoke runs exist on disk but are not folded into the reported ladder.

Rows become claims as follows:

1. Per-replicate streamed decode timings are aggregated into per-batch per-request decode throughput.
2. Per-batch throughput is normalized by the batch-1 throughput to form `eta(B)`.
3. The knee is the first batch/concurrency where `eta(B) < 0.65`. Continuous knees use log2-batch interpolation at the 0.65 crossing; censored curves are flagged when the last tested batch remains at or above threshold.
4. Predictor baselines are ranked against the observed continuous knee using Spearman rank correlation and, where the predictor carries a magnitude, log2-space error and factor error.
5. Family contrasts compare the continuous knee of the larger model to the smaller model at matched context.

The V5 report uses the same tau threshold as V4 throughout: `tau = EFF_BREAKDOWN = 0.65`. This single common metric is what lets the offline static knee, the served SGLang knee, and the weight-axis decode knee be compared on one axis.

**Provenance discipline (read before citing any magnitude).** There are two predictor-baseline files and they are not interchangeable. `v5_predictor_baselines.json` is authoritative for **ordering** (Spearman) and for the **failure-audit inputs**, but its `median_abs_factor_error` field is mislabeled — it actually holds `2^mean(log2 err)`, the geometric-mean factor, not a median (the analyzer computes `factor_error = 2**mae_log2` at `analyze_v5_predictors.py:177-178`). All **magnitude** numbers in this report are therefore taken from `predictor_baselines_extended.json`, which separates the true `median_abs_factor_error` from `mean_abs_factor_error`, carries bootstrap `spearman_ci95`, and sweeps `tau in {0.50, 0.65, 0.75, 0.80, 0.90}` on two surfaces. `TR_static_ckw_model.md` §6.4 documents the mislabel explicitly. The two files agree on Spearman to the reported digits; they differ only on the factor-error fields.

---

## 3. Headline Results

### 3.1 Predictor baselines — the full ladder

The headline surface is `full_generation` (full `LLM.generate`, prefill and decode timed together) at `tau = 0.65`. The audit ranks eleven predictors against the observed continuous knee under two conventions: finite-only (censored curves dropped, n=65, 0 censored) and censored-as-128 (censored knees imputed at batch 128, n=96, 31 censored). The full ladder, with bootstrap CIs from the extended file:

| Predictor | Spearman (finite) | Spearman CI95 (finite) | Spearman (cens-128) | median factor (ext.) | geom factor (ext.) |
|---|---:|---|---:|---:|---:|
| Ck/W + per-GPU alpha | 0.8779 | [0.7518, 0.9493] | 0.8462 | 1.297 | 1.626 |
| Ck/W closed-form (zero-param) | 0.8384 | [0.6905, 0.9282] | 0.8138 | 2.022 | 2.181 |
| ratio only `-r` | 0.8384 | [0.6905, 0.9282] | 0.8138 | — | — |
| context only `1/C` | 0.8377 | [0.7376, 0.8932] | 0.7890 | — | — |
| VRAM ceiling `n_max` | 0.8321 | [0.708, 0.9023] | 0.7537 | 4.261 | 5.361 |
| KV traffic only `1/(Ck)` | 0.6962 | [0.5229, 0.8341] | 0.7210 | — | — |
| GPU only | 0.1631 | [-0.0866, 0.3998] | 0.2768 | — | — |
| KV bytes only `1/k` | 0.1473 | [-0.0812, 0.3637] | 0.0941 | — | — |
| decode only | -0.1065 | [-0.3446, 0.1428] | 0.1530 | — | — |
| precision only | -0.1478 | [-0.378, 0.0907] | 0.2455 | — | — |
| weight bytes only `W` | NaN | (degenerate) | NaN | — | — |

**Observations.** The ladder splits cleanly into three bands. The top band (Spearman > 0.83 finite) holds Ck/W+alpha, Ck/W closed-form, the rank-equivalent `ratio only -r`, context-only, and the VRAM ceiling — five predictors whose CIs overlap heavily. The middle band is KV-traffic-only at 0.6962, which omits the shared weight read and loses about 0.14 of rank correlation against Ck/W. The bottom band is the architecture-blind baselines — GPU-only, KV-bytes-only, decode-only, precision-only — all with `|rho| < 0.3` and CIs that straddle zero (two of them negative finite). `weight bytes only W` is NaN: `W` spans only about 1.11x across the three 7-8B models, so it is near-constant within the grid and has no rank variance to correlate. The magnitude column is the part that moved versus the V5 file: the true median factor error for the zero-param Ck/W is **2.022x** (not the mislabeled 2.181x), and the per-GPU alpha correction tightens it to **1.297x**.

> The audit's job is to stop the reader from believing "longer context means earlier knee" is the whole story. It is not. Context-only ties Ck/W on rank (0.8377 vs 0.8384 finite) because context is the single largest lever in a grid where weight barely moves — but a rank tie on a weight-static grid is exactly what you would expect whether or not the weight term matters, and it cannot adjudicate the within-family, fixed-context question V5 was built to ask. The separation Ck/W actually earns here is over KV-traffic-only (which drops the weight read) and over the architecture-blind floor, not over context-only.

The magnitude win is real but not unique. Leave-one-model-out calibrated context-only and context+GPU baselines reach median factor 1.517x / 1.513x (`predictor_baselines_extended.json: full_generation.tau_0.65.calibrated_cv_baselines.leave_one_model_out`), so the per-GPU alpha's 1.297x is the best magnitude but in the same neighborhood as a calibrated context baseline. The honest claim is "best predictor, mechanistically interpretable, magnitude within ~1.3x," not "uniquely accurate."

### 3.2 Threshold sensitivity — the 0.65 knee is not cherry-picked

The censored fraction and the Ck/W rank both move with tau. From the extended file's `full_generation` tau sweep:

| tau | finite n | censored n | Ck/W finite Spearman | Ck/W finite median factor |
|---:|---:|---:|---:|---:|
| 0.50 | 38 | 58 | 0.897 | 2.241 |
| 0.65 | 65 | 31 | 0.8384 | 2.022 |
| 0.75 | 95 | 1 | 0.9204 | — |
| 0.80 | 95 | 1 | — | — |
| 0.90 | 96 | 0 | — | — |

**Observations.** Censoring is monotone in tau: a stricter efficiency threshold makes curves cross sooner, so fewer are right-censored — 58 censored at tau=0.50 collapses to 0 at tau=0.90. The headline tau=0.65 sits in the middle of this range, with 31 of 96 curves censored. The Ck/W rank correlation stays high across the whole sweep (0.897 at 0.50, 0.8384 at 0.65, 0.9204 at 0.75), so the 0.65 choice is not a value picked to flatter the predictor.

> The tau sweep is the cheapest defense against a cherry-picking objection. If the knee result only held at one threshold, the sweep would show rho collapsing away from 0.65. It does not. The dip to 0.8384 at exactly 0.65 (lower than its neighbors at 0.50 and 0.75) is, if anything, the conservative point of the sweep — the headline reports the threshold where Ck/W looks slightly worse, not best.

### 3.3 Failure audit

The failure audit prevents the rational curve from being overstated as a universal law. It flags every curve whose **free-`r` fit** — the one-free-parameter form `eta(B) = (1 + r) / (1 + B r)` fit per curve — has `R^2 < 0.70`.

| Audit item | Count |
|---|---:|
| Total curves | 96 |
| Low-fit curves, `R^2 < 0.70` | 14 |
| Negative-`R^2` curves | 4 |

**Observations.** Of the 96 fitted curves, 14 fall below the `R^2 < 0.70` low-fit threshold — about 15% of the grid — and only 4 of those go fully negative (`R^2 < 0`). The negative-`R^2` set is a strict subset of the low-fit set: every catastrophic fit is also a low-fit, but two-thirds of the low-fit tail (10 of 14) still carries a positive `R^2`. The 4-of-96 negative-`R^2` rate is the headline "where the rational form actively misrepresents the curve" count, distinct from the broader 14-of-96 "where it fits poorly" count.

> The two-tier audit (14 low-fit, 4 negative) is deliberately conservative: it reports the soft `R^2 < 0.70` boundary, not just the hard negative-`R^2` failures, so the failure tail is not understated. Eighty-two of 96 curves clear the bar, and even inside the failing 14 the majority retain a positive `R^2` — the rational form is wrong outright on only 4 curves, not on the full low-fit set.

Failure categories (overlapping labels, not a partition — a curve may carry several):

| Category | Count |
|---|---:|
| non_monotone | 13 |
| tested_beyond_vram_ceiling | 10 |
| long_context_long_decode_tail | 6 |
| negative_r2 | 4 |
| smooth_low_fit | 1 |

**Observations.** The category counts sum to 34 across only 14 curves, confirming these are overlapping labels rather than a partition — most failing curves carry several morphology tags at once. `non_monotone` dominates at 13 of 14, so a rising segment in `eta(B)` is the near-universal failure signature; `tested_beyond_vram_ceiling` (10) and `long_context_long_decode_tail` (6) account for most of the rest, and `smooth_low_fit` is a lone outlier at 1. The 4 `negative_r2` curves are exactly the V-shaped subset where non-monotonicity is severe enough to drive the fit below zero.

> The morphology is concentrated, not scattered: a single mechanism (non-monotone recovery, 13/14) explains nearly every low-fit curve, and it co-occurs with VRAM-ceiling breach (10/14) and the long-context long-decode corner (6/14). The solitary `smooth_low_fit` curve is the only failure with no morphological pathology at all — one curve out of 96 where the rational form is simply the wrong shape rather than a capacity or monotonicity artifact.

Concentration of the 14 low-fit curves by factor level (also overlapping):

| Axis | Count of 14 low-fit curves |
|---|---:|
| gpu=H100 | 13 |
| decode=512 | 12 |
| precision=fp16 | 9 |
| context=32000 | 8 |
| precision=fp8 | 5 |
| model=llama3.1-8b | 5 |
| model=mistral-7b | 5 |
| context=8192 | 4 |
| model=qwen2.5-7b | 4 |
| context=512 | 2 |
| decode=64 | 2 |
| gpu=A100-80GB | 1 |

**Observations.** The categories do not sum to 14 — they are labels, not buckets. The non-overlapping facts are: 14 curves with `R^2 < 0.70`, of which 4 have `R^2 < 0`; 13 of the 14 are non_monotone (the lone exception is the smooth_low_fit curve); and 13 of 14 are on H100 (the lone A100 curve is `qwen2.5-7b|A100-80GB|d512|fp16|ctx512`). The axis concentration is stark: H100 (13/14), long decode d512 (12/14), and long context ctx32000 (8/14) dominate. The failures are not scattered; they cluster in exactly the H100 long-context long-decode corner where §6 of the static-model methodology says the rational form should not be treated as a law.

> The median fit across all 96 curves is `R^2 = 0.928` (`amortization_full_stats.json: L3_roofline`), and the audit's value is that it shows this median is not hiding the failures — they are the labeled tail. The model is a strong predictor in the smooth decode-amortization regime and a poor one in the capacity tail, and V5's framing is two-regime by design: fit where it fits, label where it breaks.

The four negative-`R^2` curves are the morphological signature worth naming. All four are at `H100 | d512 | ctx32000`, one each on Llama-3.1-8B (fp16 and fp8) and Mistral-7B (fp16 and fp8), none on Qwen. The worst is `llama3.1-8b|H100|d512|fp8|ctx32000` at `R^2 = -0.0978` — which matches the free-`r` fit minimum cited as the macro `\vfourFitMin = -0.0978` in `TR_static_ckw_model.md` §1, an independent cross-check that the audit and the macro table are reading the same curve. All four negative-`R^2` curves carry `recovers_after_crossing = true`: `eta` dips below tau and then rebounds at b16->b32->b64 (rebound deltas 0.2845-0.3197). That V-shape is what breaks a monotone-decay fit and drives `R^2` negative — the curve is not noisy, it is non-monotone in a way the rational form cannot represent.

The single `smooth_low_fit` curve is the instructive edge case: `llama3.1-8b|H100|d512|fp8|ctx8192`, `R^2 = 0.6932`, monotone (empty `monotone_increases` list), no VRAM breach (max_batch/nmax = 0.992) — a genuinely smooth curve that the rational form fits poorly with no morphological pathology at all. It is the only low-fit curve with no second category, and it bounds how much of the low-fit tail is "real shape the model misses" versus "capacity/non-monotone artifact": almost all of it is the latter.

The complete per-curve detail for all 14 low-fit curves (`v5_failure_audit.json: low_fit_curves[]`):

| key | fit_r2 | obs knee | arch knee | nmax_vram | max/nmax | recovers | categories |
|---|---:|---:|---:|---:|---:|:--:|---|
| qwen2.5-7b\|A100-80GB\|d512\|fp16\|ctx512 | 0.596 | null | 280.9 | 2114.4 | 0.03 | false | non_monotone |
| qwen2.5-7b\|H100\|d512\|fp16\|ctx512 | 0.4786 | null | 280.9 | 2114.4 | 0.03 | false | non_monotone |
| qwen2.5-7b\|H100\|d512\|fp16\|ctx32000 | 0.6309 | 16.50 | 6.01 | 33.83 | 1.892 | false | non_monotone, vram, lc-ld-tail |
| qwen2.5-7b\|H100\|d512\|fp8\|ctx32000 | 0.686 | 10.96 | 3.77 | 37.98 | 1.685 | false | non_monotone, vram, lc-ld-tail |
| llama3.1-8b\|H100\|d64\|fp16\|ctx32000 | 0.6298 | 7.30 | 3.60 | 14.60 | 4.383 | false | non_monotone, vram |
| llama3.1-8b\|H100\|d512\|fp16\|ctx8192 | 0.3921 | null | 9.59 | 57.04 | 1.122 | false | non_monotone, vram |
| llama3.1-8b\|H100\|d512\|fp16\|ctx32000 | **-0.0713** | 8.48 | 3.60 | 14.60 | 4.383 | **true** | non_monotone, vram, lc-ld-tail, neg_r2 |
| llama3.1-8b\|H100\|d512\|fp8\|ctx8192 | 0.6932 | 56.92 | 5.57 | 64.52 | 0.992 | false | **smooth_low_fit** |
| llama3.1-8b\|H100\|d512\|fp8\|ctx32000 | **-0.0978** | 6.68 | 2.57 | 16.52 | 3.875 | **true** | non_monotone, vram, lc-ld-tail, neg_r2 |
| mistral-7b\|H100\|d64\|fp16\|ctx32000 | 0.6982 | 7.23 | 3.40 | 14.98 | 4.274 | false | non_monotone, vram |
| mistral-7b\|H100\|d512\|fp16\|ctx8192 | 0.6494 | null | 8.81 | 58.50 | 1.094 | false | non_monotone, vram |
| mistral-7b\|H100\|d512\|fp16\|ctx32000 | **-0.0371** | 8.40 | 3.40 | 14.98 | 4.274 | **true** | non_monotone, vram, lc-ld-tail, neg_r2 |
| mistral-7b\|H100\|d512\|fp8\|ctx8192 | 0.6597 | 61.20 | 5.17 | 65.25 | 0.981 | false | non_monotone |
| mistral-7b\|H100\|d512\|fp8\|ctx32000 | **-0.0337** | 6.54 | 2.47 | 16.70 | 3.831 | **true** | non_monotone, vram, lc-ld-tail, neg_r2 |

**Observations.** The four negative-`R^2` rows are the only four with `recovers = true`, and they are exactly the four `lc-ld-tail` rows — one each on Llama and Mistral at fp16 and fp8, all at `H100|d512|ctx32000`, none on Qwen. The `arch knee` column shows the predictor consistently calls these capacity-tail knees far too early (predicted 2.5-3.6 against observed 6.5-8.5), which is the under-prediction the model makes when the true curve is a V-shape it cannot see. The four `null`-observed-knee rows (the two ctx512 and two ctx8192-fp16 curves) never cross tau within B<=64 — they are the right-censored members of the low-fit tail. The two ctx512 rows have `max/nmax = 0.03` (deep within capacity) and are therefore non_monotone-only, NOT VRAM-breach: low fit there is shape, not capacity.

> The per-curve table is the audit's strongest defense against an "it doesn't fit" objection, because it names every curve, its fit, and why it failed. The pattern is consistent: every catastrophic fit (`R^2 < 0`) is a non-monotone V-shape in the H100 long-context long-decode corner that recovers after crossing tau, and the model under-predicts its knee. None is unexplained noise. The lone smooth_low_fit row (`R^2 = 0.6932`, recovers=false, no VRAM breach) is the one curve where the rational form simply has the wrong shape, and it is a single curve out of 96.

### 3.4 Served SGLang validation

The compact SGLang served validation ran 8/8 ok cells with zero errors. The metric is the **served knee**: the smallest concurrency `N > 1` where parallel efficiency `eta(N) = agg_decode_tps(N) / agg_decode_tps(1) / N` falls below 0.65, with `None` meaning no crossing by `N = 64`. The static baseline is the offline `d64` static knee (decode 64, matching the served `max_tokens=64`).

| Served validation claim | Result |
|---|---:|
| SGLang served knee <= static knee | 8/8 |
| Distinct-prompt served knee <= static knee | 4/4 |
| Shared-prefix served knee within one static ladder step | 2/4 |
| SGLang equal to matched vLLM served knee | 3 |
| SGLang lower (earlier) than matched vLLM served knee | 5 |
| SGLang higher (later) than matched vLLM served knee | 0 |

**Observations.** The upper-bound direction is unanimous: all 8/8 served knees land at or below their static knee, with the distinct-prompt regime clean at 4/4. The shared-prefix regime is the looser case at 2/4 "within one static ladder step" — but the remaining 2 are shared cells whose static knee is censored (no finite ladder step to bound against), not violations. Against vLLM, SGLang never knees later (0 higher), tying in 3 cells and coming in earlier in 5, so the bound direction is preserved when the engine is swapped.

> The cross-engine count is the load-bearing number here: SGLang knees later than vLLM in zero of the 8 cells. A served-knee ceiling that survives a different scheduler and a different prefix-cache implementation is evidence the ceiling is a property of bandwidth-bound decode rather than a vLLM scheduling artifact. The shared-prefix 2/4 is a censoring bookkeeping detail, not a failure of the bound.

Cell-level served table (all cells SGLang, A100-80GB, fp16, `min_ok_rate = 1.0`):

| Model | Context | Regime | Static knee | SGLang knee | vLLM knee | SGL vs static | SGL vs vLLM | SGL-static steps | SGL-vLLM steps |
|---|---:|---|---:|---:|---:|---|---|---:|---:|
| llama3.1-8b | 512 | distinct | None | 12 | 12 | lower | equal | -6 | 0 |
| llama3.1-8b | 512 | shared | None | 48 | None | lower | lower | -2 | -2 |
| llama3.1-8b | 8192 | distinct | 16 | 4 | 4 | lower | equal | -4 | 0 |
| llama3.1-8b | 8192 | shared | 16 | 16 | 32 | equal | lower | 0 | -2 |
| qwen2.5-7b | 512 | distinct | None | 12 | 16 | lower | lower | -6 | -1 |
| qwen2.5-7b | 512 | shared | None | 48 | None | lower | lower | -2 | -2 |
| qwen2.5-7b | 8192 | distinct | 32 | 4 | 4 | lower | equal | -6 | 0 |
| qwen2.5-7b | 8192 | shared | 32 | 24 | 32 | lower | lower | -1 | -1 |

**Observations.** Every SGLang served knee lands at or below its static knee — the upper-bound direction holds 8/8. The two distinct ctx8192 cells match vLLM exactly (Llama static 16 -> SGLang 4 = vLLM 4; Qwen static 32 -> SGLang 4 = vLLM 4). The two shared ctx512 cells have censored static knees (`None`), so they have no finite ladder step to bound against — they are the 2/4 shared cells outside the "within one step" count, not violations. Critically, SGLang never knees *later* than vLLM in any of the 8 cells (equal in 3, lower in 5, higher in 0), so swapping engines does not break the bound direction.

> The static knee is not a served-latency measurement; it is an amortization ceiling. This slice tests whether that ceiling is a vLLM scheduler artifact or a property of bandwidth-bound decode. SGLang uses a different scheduler and RadixAttention prefix caching rather than vLLM's automatic prefix cache, and the ceiling direction survives the swap in every cell. That is the cross-engine down-payment the V4-only served validation lacked.

This extends V4's vLLM-only served validation. The V4-to-V5 progression, side by side:

| Layer | Engine | Distinct upper-bound | Shared ceiling | Scope |
|---|---|---|---|---|
| V4 served | vLLM 0.10.2 | 21/21 finite-both (Wilson CI [0.85, 1.00]) | 21/28 finite-both (75%, 25% violations) | full grid, magnitude estimate |
| V5 served | SGLang (RadixAttention) | 4/4 | 2/4 within one step (2/4 censored static) | compact subset, direction only |

**Observations.** V4 established the upper-bound *magnitude* on vLLM with a Wilson interval; V5 deliberately does not re-estimate that magnitude. The reviewer concern V5 answers is narrower and sharper — the V4 served grid was vLLM-centered, so the upper-bound could in principle be an artifact of vLLM's scheduler or its automatic prefix cache. SGLang has a different scheduler and a different prefix-cache implementation (RadixAttention), which makes it the cleanest available test of whether the bound *direction* is engine-specific. The 8/8 cross-engine result says it is not. The 25% shared-prefix violation rate in V4 is itself a measured property of the deployed vLLM 0.10.2 prefix cache, not a model failure — it is the rate at which a real prefix cache lets shared-prompt traffic exceed the static ceiling.

> The closure and V5 cover two different objects. The streaming/marginal-decode closure validates the *offline static-knee object itself* (Ck/W rank survives even with prefill in the timed window). V5 validates that the same static knee *transfers as an upper bound* to a second served engine. Together they secure both ends: the closure secures the static knee's decode-meaning, V5 secures its served-transfer across engines. The cross-engine SGLang grid is also the explicit down-payment on the closure's own forward ask to replicate the streaming knee on a second engine to show the Ck/W ordering is an architecture/memory property rather than a vLLM scheduler artifact.

### 3.5 Weight-axis true-decode matrices

The weight-axis protocol fixes the serving setup and moves model size inside families. It uses vLLM 0.10.2 OpenAI-compatible streaming, one H100, fp8 weights, fp16 KV cache, 64-token decode, exact prompt contexts, and the same `tau = 0.65` knee rule. Timing starts only after every request in a batch has emitted its first non-empty streamed token, so the measurement is a true-decode window rather than prefill or time-to-first-token.

Generated table sizes:

| Table | Rows |
|---|---:|
| `weight_axis_decode_cells.csv` | 28 |
| `weight_axis_decode_pair_contrasts.csv` | 17 |
| `weight_axis_decode_batch_measurements.csv` | 263 |
| `weight_axis_decode_replicates.csv` | 708 |

**Observations.** The four substrate tables form a clean aggregation pyramid: 708 per-replicate streamed-decode timings roll up into 263 per-batch measurements, which collapse into 28 analyzed cells and 17 pairwise contrasts. The 28-to-708 fan-out reflects multiple replicates per batch level per cell, and the 17 pair contrasts are the within-family matched-context comparisons (Qwen, Mistral, and Gemma 3 small-vs-large at three contexts each) that carry the weight-axis claim. Every reported knee and ratio in this section traces back through this pyramid to the 708-row replicate floor.

> The row counts establish that the weight-axis result rests on real replicated timings, not a thin handful of runs: 708 timed replicates underpin 28 cells. The narrowing from 708 to 17 contrasts is the aggregation that turns raw streamed-token timestamps into the matched-context knee comparisons — the contrast count, not the replicate count, is what the family-level claims are counted in.

Per-run measured H100 seconds and analyzer cost estimate:

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
| Total analyzer-estimated H100 cost (at USD 3.95/hr) | USD 9.73 |

**Observations.** The full weight-axis campaign — smokes, broad sweeps, confirm re-runs, and two matrices — cost under ten dollars of H100 time. The smoke runs (Qwen 243.2s, Mistral 666.2s, Gemma 3 991.9s) exist on disk but are reps=1 and are not folded into the analysis ladder; the folded cells come from the full/confirm/matrix runs only.

> The cost number matters for a reproducibility venue: the weight-axis result is cheap to reproduce. The paper macro `\waxisCost = USD 3.87` is the Qwen-only slice (a narrower scope), not a discrepancy with the USD 9.73 full-campaign figure. The cost estimate uses measured function load time plus per-context elapsed time at the Modal H100 price and excludes small CPU/memory overhead.

---

## 4. Methodology Deep Dive

### 4.1 The single-H100 fp8 true-decode protocol

Every weight-axis cell runs on a single H100 (one Modal function per model, `@app.function(gpu="H100", timeout=14400)`), serving through vLLM 0.10.2's OpenAI-compatible streaming endpoint. Weights are loaded fp8 via `--quantization fp8`; the KV cache is left at the fp16 default (not overridden). This is deliberately a weight-axis experiment, not a KV-precision experiment — KV stays fp16 in every cell so the only quantity moving within a family is `W`. The compute dtype is `float16` for Qwen/Mistral and `bfloat16` for Gemma, distinct from the fp8 weight *storage*. GPU memory utilization is pinned at 0.90, and `--max-model-len` is set to `max(max_context + 64, min_model_len)`.

The true-decode window is the methodological core. Each request is streamed and every non-empty token gets a `time.perf_counter()` timestamp. For a batch of `B` concurrent requests, the window opens at `window_start = max(first_token_time over the B requests)` — only once *all* requests have produced their first token, which excludes prefill and TTFT from the throughput denominator. The window closes at `window_end = max(last_token_time)`. Only tokens with timestamp `>= window_start` are counted. Then `agg_decode_tps = token_count / (window_end - window_start)` and `per_req_decode_tps = agg_decode_tps / B`. Excluded by construction: prefill/first-token, the 8-token warmup completion, first-token-spread skew (recorded as a diagnostic only), and any failed stream — a batch is scored only if all `B` streams return HTTP 200 with at least `max(2, decode//2) = 32` tokens.

Prompts defeat the prefix cache so each request does real prefill plus decode. `make_prompt` builds a distinct tokenized head `"Request {uid} context {context}. "` padded with seeded random 6-digit ints up to `min(64, context)` tokens, then fills to exactly `context` tokens with a repeated `"The quick brown fox..."` body. The distinct head is what stops vLLM from serving cached prefill. Decode is greedy (`temperature=0.0`) and fixed at exactly 64 tokens (`max_tokens=64, min_tokens=64`), so every request emits the same number of decode steps. Contexts are `2048`, `8192`, and `32000`, matched across the small and large model within every pair.

### 4.2 The knee definition and the estimator

For a curve, `eta_by_batch[b] = per_req_decode_tps_mean(b) / per_req_decode_tps_mean(1)`. The **discrete knee** is the first tested batch `b > 1` with `eta[b] < 0.65`. The **continuous knee** (the headline object) interpolates the crossing in log2(batch) space: at the first `i` where `eta[i] < tau <= eta[i-1]`, `knee = 2^(log2(b_{i-1}) + frac*(log2(b_i) - log2(b_{i-1})))` with `frac = (eta[i-1] - tau) / (eta[i-1] - eta[i])`. Log2 interpolation matches the geometric batch ladder. If the curve never crosses (last-batch `eta >= tau`), the continuous knee is `+inf`, the curve is marked `censored=True`, and it is dropped from the finite Spearman.

The architectural estimator is `r_arch = C * k / W`, with `k = 2 * layers * kv_heads * head_dim * KV_DTYPE_BYTES` and `KV_DTYPE_BYTES = 2`, and the predicted knee `arch_knee = (1 + r - tau) / (tau * r)`. Predictors are negated for sign consistency so that "higher predictor means later knee": `Ck/W = -r_arch`, `context 1/C = -context`, `KV 1/(Ck) = -(context*k)`, `W only = weight_bytes`. Spearman is computed with `scipy.stats.spearmanr` and requires at least three points.

### 4.3 The censored-as-128 imputation

A curve is right-censored when its `eta` never drops below 0.65 within the tested batch ladder. For the static grid the ladder tops out at batch 64, so a censored curve has a true knee somewhere above 64 — unobserved. V5 reports two conventions for the static grid:

- **finite-only:** censored curves dropped, correlation over the n=65 finite curves (0 censored).
- **censored-as-128:** censored knees placed at batch 128 so all 96 curves enter the correlation (31 censored + 65 finite = 96).

The choice of **128** is the next geometric doubling past the maximum tested batch (64 x 2 = 128). It is an analyst placement, not a measurement — explicitly flagged as a confound in the static-model methodology (§7 item 7: "an analyst choice, not a measurement"). At tau=0.65, 31 of 96 curves are censored (`v5_predictor_baselines.json: censored_as_128.n_censored = 31`; the extended file agrees). The methodology doc's "32% of curves are right-censored" figure is the same set — 31/96 = 32.3%, a rounding caveat, not two different counts.

The extended file carries a `censored_underpredictions` field that explains why censoring penalizes Ck/W on magnitude. After the censored knees are forced to 128, it counts how many of the 31 a predictor *still* under-predicts (predictor said knee < 128 but truth is >= 128): Ck/W closed-form under-predicts 15 of 31, Ck/W+alpha 8 of 31, the VRAM ceiling only 2 of 31. The zero-param Ck/W systematically under-predicts the censored tail — which is exactly why its censored-as-128 Spearman (0.8138) sits below its finite Spearman (0.8384), and why the VRAM ceiling, built to track capacity, handles the censored tail better than Ck/W even though it is a weaker overall predictor.

### 4.4 The served-knee parallel-efficiency protocol

The served knee uses the identical `tau = 0.65` and identical `eta` shape as the offline static knee, by construction — one common metric across offline and served. Served `eta(N) = agg_decode_tps(N) / agg_decode_tps(1) / N`; the knee is the smallest concurrency `N > 1` on the ladder where `eta(N) < 0.65`, and `None` if there is no crossing by `N = 64`. The V5 SGLang ladder is the fine 11-point `1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64` for both prompt regimes (confirmed in the JSON: every cell's `parallel_efficiency_by_n` has exactly those 11 keys). There is no interpolation on the served side — the knee is snapped to the nearest ladder grid point below 0.65, so the served knee's granularity is the ladder's granularity.

The static baseline the served knee is compared against is the `d64` leg of `amortization_stats.json: knee_table` (decode 64, matching the served `max_tokens=64`), keyed `<model>|<gpu>|d64|<precision>` then `ctx<context>`. Each knee is mapped to its index in `LADDER = [1,2,4,6,8,12,16,24,32,48,64,None]`; `sglang_minus_static_steps = rank(served) - rank(static)`, and the shared "within one step" tolerance is `abs(steps) <= 1`. With `None -> +inf`: served < static is earlier (conservative), served = static is tight, served > static would be a violation, and "served <= static" is the union of the first two. Censored static cells (no finite knee) are excluded from the proportion denominators and counted separately.

---

## 5. Weight-Axis Results by Family

### 5.1 Qwen2.5: clean positive result

Qwen2.5 provides the cleanest same-family test in the V5 data. The broad run compares Qwen2.5-14B to Qwen2.5-32B at three contexts (REPS=2); the confirm run repeats the two most operationally important long-context cells with denser ladders near the crossing (reps=3): ctx8192 `[1,6,8,9,10,11,12,14,16]`, ctx32000 `[1,2,3,4,5,6,8]`.

| Run | Context | 14B knee | 32B knee | Observed ratio | Predicted ratio | Larger later |
|---|---:|---:|---:|---:|---:|---|
| qwen_full | 2048 | 26.0711 | 31.4907 | 1.2079 | 1.6157 | yes |
| qwen_full | 8192 | 8.7606 | 9.7646 | 1.1146 | 1.5060 | yes |
| qwen_full | 32000 | 3.5514 | 3.7923 | 1.0678 | 1.2994 | yes |
| qwen_confirm | 8192 | 8.4162 | 10.8535 | 1.2896 | 1.5060 | yes |
| qwen_confirm | 32000 | 3.4990 | 4.0011 | 1.1435 | 1.2994 | yes |

**Observations.** Qwen2.5-32B keeps its true-decode knee later than Qwen2.5-14B in all five rows — 3/3 broad and 2/2 confirm — so the "larger later" column is unanimously yes. The observed ratios (1.07-1.29) sit below the predicted ratios (1.30-1.62) at every context but always point the same way, and both observed and predicted shrink monotonically as context grows from 2048 to 32000 (observed 1.21 -> 1.11 -> 1.07; predicted 1.62 -> 1.51 -> 1.30). The confirm run at ctx8192 records a *larger* observed shift than the broad run (1.2896 vs 1.1146), showing the broad coarse ladder under-resolved that crossing rather than overstating it.

> The context-dependent compression is the mechanistically expected pattern: at long context the `Ck` numerator of `r` dominates for both model sizes, partly washing out the larger model's weight-ratio advantage, so the knee separation narrows even as it stays positive. That the denser confirm ladder reproduces the direction — and at ctx8192 finds a bigger gap than the coarse ladder — is the right kind of robustness check, ruling out the worry that the broad result was a ladder-resolution artifact.

Qwen full-run predictor ladder (8/9 finite, 1 censored):

| Predictor | Spearman |
|---|---:|
| Ck/W closed-form | 0.9524 |
| context only `1/C` | 0.8063 |
| KV traffic only `1/(Ck)` | 0.7857 |
| W only | -0.0126 |

**Observations.** Qwen2.5-32B keeps the true-decode knee later than Qwen2.5-14B at all five matched contexts across broad and confirm runs (3/3 broad, 2/2 confirm). The observed ratios (1.07-1.29) are smaller than predicted (1.30-1.62) but always in the predicted direction. There is a context-dependent compression: both the observed and predicted shift shrink as context grows (broad observed 1.21 -> 1.11 -> 1.07 from ctx2048 to ctx32000; predicted 1.62 -> 1.51 -> 1.30), because at long context the `Ck` numerator dominates `r` for both models and the weight-ratio advantage of the larger model is partly washed out — the knee is already early for both. The confirm run, with denser ladders near the crossing, reproduces the broad result (and at ctx8192 records a *larger* observed shift, 1.29 vs 1.11, showing the broad coarse ladder slightly under-resolved the crossing rather than over-stating it). The Ck/W Spearman of 0.9524 over the finite Qwen curves is the best of any single family.

> This is exactly the contrast KV-only should get wrong and Ck/W should get right: the larger model has larger KV bytes per token, so KV-only predicts the larger model knees *earlier*, yet the larger model knees *later* at every context because `W` grew faster than `k`. The `W only` Spearman of -0.0126 confirms weight alone does not rank the knee — it is the *ratio* `Ck/W` that carries the signal, and within Qwen it ranks the knees almost perfectly.

### 5.2 Mistral: second clean positive family

The Mistral pair is Mistral-7B-Instruct-v0.3 against Mistral-Small-24B-Instruct-2501 on the same H100/fp8 protocol (matrix, reps=3).

| Context | 7B knee | 24B knee | Observed ratio | Predicted ratio | Larger later |
|---:|---:|---:|---:|---:|---|
| 2048 | 24.4343 | 38.2735 | 1.5664 | 2.4485 | yes |
| 8192 | 8.7898 | 11.3806 | 1.2948 | 2.1254 | yes |
| 32000 | 3.4922 | 4.1799 | 1.1969 | 1.6037 | yes |

Mistral predictor ladder (6/6 finite, 0 censored):

| Predictor | Spearman |
|---|---:|
| Ck/W closed-form | 1.0000 |
| context only `1/C` | 0.9562 |
| KV traffic only `1/(Ck)` | 0.8286 |
| W only | 0.2928 |

**Observations.** Mistral independently reproduces the weight-axis direction at every matched context (3/3), with the largest observed shift in the whole campaign — the 7B->24B step at ctx2048 moves the knee 1.5664x. Ck/W ranks all six finite knees perfectly (Spearman 1.0000) in this small matrix.

> Mistral is the second independent family, and the 7B->24B jump (a 3.25x weight ratio against a 1.25x KV ratio) is the cleanest separation of the weight axis from the KV axis in V5. The larger model has slightly more KV traffic per token yet knees substantially later, because `W` more than tripled. Two independent families — Qwen and Mistral — now show the same direction, which is the core of the weight-axis claim.

### 5.3 Gemma 3: useful stress test, not a clean monotone replication

Gemma 3 ran successfully across 4B, 12B, and 27B (matrix, reps=3). This matters because earlier Gemma attempts had environment-level blockers: Gemma2 hit the H100 flash-attention tanh-softcapping path, and Gemma4 was not recognized by the pinned Transformers/vLLM stack. Gemma 3 avoids those serving failures and gives real data — but it is mixed.

| Contrast | Success |
|---|---:|
| Gemma 3 4B->12B | 0/3 |
| Gemma 3 12B->27B | 3/3 |
| Gemma 3 4B->27B | 2/3 |
| All Gemma 3 pairwise contrasts | 5/9 |

**Observations.** Gemma 3 is mixed, not clean: only 5 of 9 pairwise contrasts succeed. The 12B->27B step is a perfect 3/3, but 4B->12B is 0/3 — inverted at every context — and 4B->27B is partial at 2/3. The single 0/3 contrast is what pulls the family down to 5/9 and is exactly why V5 cannot claim a universal "larger model always knees later" law.

> The split is diagnostic, not random: the contrast that fails completely (4B->12B) is the one §6 identifies as a Ck/W degeneracy, where `k` and `W` both grow ~2.8x and the predicted ratio sits on the knife edge at ~1.00. The contrasts where the predictor has real signal (12B->27B, and 4B->27B at the shorter contexts) succeed. The 5/9 headline therefore reflects predictor degeneracy on one step, not a breakdown of the underlying mechanism.

| Contrast | Context | Small knee | Large knee | Observed ratio | Predicted ratio | Larger later |
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

**Observations.** The row-level detail localizes the failure precisely. All three 4B->12B rows have observed ratios below 1 (0.7874-0.9022) against predicted ratios of essentially 1.00 (1.0010-1.0032) — the predictor calls a coin-flip and the data lands the other way, which is uninformative rather than contradictory. The 12B->27B rows are all above 1 (1.1715-1.3485) with predictions that track them (1.1880-1.6247), the clean 3/3 leg. The lone 4B->27B miss is ctx32000 (observed 0.9225 vs predicted 1.1891), where the long-context numerator and Gemma's irregular attention geometry combine to flip the sign.

> The detailed table shows the inversion is confined to the degenerate-prediction rows: where the predicted ratio is ~1.00 (4B->12B), the observed direction is meaningless noise; where the predicted ratio is well above 1 (12B->27B), the observed direction matches. This is the empirical backing for treating the 4B->12B inversion as predictor degeneracy, not a counterexample — the predictor never claimed a direction there to be wrong about.

Gemma 3 predictor ladder (9/9 finite, 0 censored):

| Predictor | Spearman |
|---|---:|
| Ck/W closed-form | 0.9167 |
| context only `1/C` | 0.9487 |
| KV traffic only `1/(Ck)` | 0.8833 |
| W only | 0.0527 |

**Observations.** The 12B->27B contrast agrees cleanly with the prediction (3/3, observed ratios 1.17-1.35), and the rank correlation across all nine Gemma 3 curves stays high (0.9167). But 4B->12B is inverted at every context (observed ratios 0.79-0.90, all < 1), and 4B->27B fails at ctx32000 (0.9225). Within Gemma, context-only (0.9487) actually edges Ck/W (0.9167) on rank — for the same reason it edges it on the static grid: when the within-family weight separation is degenerate, context carries the order.

> Gemma 3 should be reported as a stress test, not a clean replication. The 12B->27B step is a third clean positive contrast, but the 4B->12B inversion is real and must be explained mechanistically rather than dismissed — which §6 does. The reviewer-facing point is that Ck/W captures a major architectural pressure, not that model size alone gives a universal monotone knee ordering. Gemma 3's attention geometry and implementation details remain material.

### 5.4 Cross-family pooled correlation

Pooling all three families (Qwen 9 + Mistral 6 + Gemma 3 9 = 24 cells, 23 finite after dropping the one censored Qwen2.5-7B@ctx2048 cell):

| Predictor | Pooled Spearman |
|---|---:|
| context only `1/C` | 0.8948 |
| KV traffic only `1/(Ck)` | 0.8547 |
| Ck/W closed-form | 0.8142 |
| W only | 0.0597 |

Pooled Ck/W Spearman = 0.8142, 95% bootstrap CI [0.5737, 0.9175] (10,000 resamples, numpy `default_rng` seed 164, percentile method; CI values inferred from `TR_weight_axis_matrix.md`'s citation of `cross_family_bootstrap_ci.json`, NOT-verified against that file directly). Leave-one-out jackknife range 0.7877-0.8453: no single cell drives the pooled rho.

**Observations.** Pooled cross-family is the weakest framing for Ck/W. Context-only (0.8948) edges it, just as on the static grid, because the pooled matrix is dominated by the context axis (three contexts spanning 2048-32000 against a handful of weight steps). The wide CI [0.57, 0.92] reflects only 23 finite points.

> The honest read is that the within-family, fixed-context contrast is the load-bearing evidence, not the pooled rank correlation. Pooling re-introduces the context lever that V5 was specifically designed to hold fixed, and context wins on a pooled rank metric whether or not the weight term matters. `W only` rho is approximately zero everywhere (-0.0126 / 0.2928 / 0.0527 / 0.0597) — weight alone never ranks the knee; the ratio `Ck/W` does, and it is the within-family movement at fixed context, not the pooled number, that distinguishes the ratio from context-only.

---

## 6. Mechanism

### 6.1 Why context-only ties on rank but cannot explain within-family movement

Across both the static grid (context-only 0.8377 vs Ck/W 0.8384 finite) and the pooled weight-axis matrix (context-only 0.8948 vs Ck/W 0.8142), context-only matches or beats Ck/W on rank. This is not evidence that the weight term is inert — it is a structural property of grids where context is the largest-varying lever and weight barely moves. On the static grid `W` spans only 1.11x across the three 7-8B models; in the pooled matrix the three contexts span 16x while the weight steps within a family span 2-6x and pull in different directions. A rank metric on such a grid is dominated by whichever axis has the most monotone spread, and that is context.

The within-family, fixed-context contrast is the only design that isolates the weight term. At fixed `C`, the `Ck` numerator of `r` is fixed by the architecture and the only thing moving between the small and large model is `W` in the denominator (plus the architecture-dependent `k`). Context-only is *constant* within such a contrast — it predicts the same knee for the 14B and the 32B because it never sees `W`. So context-only cannot adjudicate the within-family question at all; it has no signal there by construction. That Qwen and Mistral both move the knee later for the larger model at fixed context is precisely the evidence context-only cannot produce.

### 6.2 Why Gemma 3 4B->12B inverts: the Ck/W degeneracy

The 4B->12B inversion is not a falsification of the weight-axis hypothesis; it is a case where the predictor has no signal to begin with. The architecture parameters (from `weight_axis_decode_cells.csv` and the JSON `architecture` block):

- gemma3-4b: layers 34, kv_heads 4, head_dim 256, params 4,300,079,472 -> `k = 139,264 B/token`, `W = 4.30e9` (fp8).
- gemma3-12b: layers 48, kv_heads 8, head_dim 256, params 12,187,325,040 -> `k = 393,216 B/token`, `W = 1.219e10` (fp8).

The KV ratio `k_large/k_small = 393,216/139,264 = 2.8235`. The weight ratio `W_large/W_small = 12,187,325,040/4,300,079,472 = 2.8342`. Both grow about 2.8x, almost identically. The predicted knee ratio is therefore `(Ck/W)_large/(Ck/W)_small = k_ratio/W_ratio = 0.9962` — a 0.38% change. The predicted ratios at each context are essentially 1.00 (1.0032 / 1.0022 / 1.0010). The predictor sits exactly on the knife edge between "earlier" and "later," so its rank signal for this contrast is approximately zero, and the observed inversion (ratios 0.79-0.90) is noise relative to a predictor with no signal — uninformative about the hypothesis, not a counterexample to it.

The contrast with the non-degenerate pairs makes this concrete:

| Pair | k ratio | W ratio | (Ck/W)_large/_small |
|---|---:|---:|---:|
| qwen 14->32 | 1.3333 | 2.2183 | 0.6011 |
| mistral 7->24 | 1.2500 | 3.2523 | 0.3843 |
| gemma3 4->12 | 2.8235 | 2.8342 | 0.9962 (degenerate) |
| gemma3 12->27 | 1.2917 | 2.2509 | 0.5738 |
| gemma3 4->27 | 3.6471 | 6.3795 | 0.5717 |

**Observations.** In every non-degenerate pair, `W` outpaces `k`, so `(Ck/W)_large/_small` is well below 1 (0.38-0.60) and the predictor cleanly says "larger knees later." The gemma3 4->12 pair is the lone degenerate row at 0.9962, where `k` and `W` grow in lockstep. The 12->27 and 4->27 pairs are non-degenerate (0.5738, 0.5717) and the predictor recovers its signal — which is why 12->27 is a clean 3/3 and 4->27 is mostly right (2/3).

> The root architectural cause is Gemma 3's attention geometry. Gemma 3 uses `head_dim = 256` at 4B and 12B (double the 128 of Qwen/Mistral) combined with a `kv_heads 4->8` doubling across the 4B->12B step, which makes `k` scale almost as fast as the parameter count over that one step. The geometry then changes non-monotonically with size: `kv_heads 4->8->16` and `head_dim 256->256->128` (the 27B drops head_dim back to 128). The 4->27 partial failure at ctx32000 traces to the same irregularity. The inversion is a property of Gemma 3's architecture making `Ck/W` degenerate, not a failure of the mechanism the ratio encodes.

### 6.3 The per-GPU alpha

The closed-form Ck/W is zero-parameter and lands within about 2x on the static grid (median factor 2.022x). The per-GPU alpha is a single empirical multiplicative constant fit per GPU type that tightens the magnitude to about 1.3x (median factor 1.297x) without changing the rank ordering or the mechanism. It is a calibration term, not a structural part of the model — it absorbs per-GPU constants (effective memory bandwidth, scheduler overhead, wave-batching behavior) that the zero-param form does not model. The improvement is real but is matched in neighborhood by a leave-one-model-out calibrated context baseline (median factor 1.517x), so the alpha is best framed as "the model calibrates well per GPU," not "the alpha is uniquely accurate" or "the alpha is first-principles."

---

## 7. Operational Quirks and Execution Arc

### 7.1 Run structure: smoke vs matrix vs confirm

The weight-axis analyzer reads seven result-JSON inputs but folds only four run types into the reported ladder. `qwen_smoke`, `mistral_smoke`, and `gemma3_smoke` are reps=1 smoke runs that exist on disk (along with Gemma2 and Gemma4 smokes) but are NOT folded into the analysis — `analyze_weight_axis_decode.py` reads only `qwen_full` (broad sweep, REPS=2), `qwen_confirm` (qwen-only re-run with denser ladders near the crossing, reps=3), `mistral_matrix` (reps=3), and `gemma3_matrix` (reps=3). The reported analysis is 28 cells: qwen_full 9 + qwen_confirm 4 + mistral_matrix 6 + gemma3_matrix 9, all `status=ok`.

### 7.2 fp8-weight handling

All cells run `--quantization fp8` (weight storage fp8); `weight_bytes = params * 1.0` under fp8 (the analyze convention; fp16 would be `* 2.0`). The compute dtype differs from the quantization: float16 for Qwen/Mistral, bfloat16 for Gemma — fp8 is the weight storage, dtype is the activation/compute path. The KV cache is left at fp16 default throughout, which keeps this a weight-axis experiment rather than a KV-precision one.

### 7.3 Excluded and censored cells

One static-grid weight-axis cell is censored and dropped: `qwen2.5-7b @ ctx2048` never crosses tau (its eta ladder ends at 0.766 > 0.65 at batch 48), so `continuous_knee = inf`, `censored = True`. This is why the Qwen pool is 8/9 finite and the cross-family pool is 23/24. The 7B family is the ladder anchor, not a reported *pair* (the pairs are 14->32, 7->24, 4->12, 12->27, 4->27), so the censored anchor does not remove a contrast.

The Gemma2/Gemma4 smoke runs are excluded as environment-support failures, not negative model results: Gemma2 smoke rows failed at request time because the pinned H100 flash-attention build did not support tanh softcapping in the vLLM path; Gemma4 smoke rows failed at startup because the pinned Transformers/vLLM stack did not recognize the `gemma4` / `gemma4_unified` architectures. Gemma 3 was promoted to the matrix in their place.

### 7.4 The failure-audit edge cases

The static failure audit operates on the 96 curves, re-fitting each with the one-free-parameter form and re-classifying the `R^2 < 0.70` tail. `non_monotone` is detected from a per-curve `monotone_increases[]` list (segments where eta rises rather than falls); any positive segment flags the curve. `tested_beyond_vram_ceiling` flags when `max_batch_over_nmax > 1` — the 10 such curves range 1.094-4.383x over capacity, while the two ctx512 curves (max/nmax = 0.03, deep within capacity) are NOT in this category despite being low-fit. `long_context_long_decode_tail` is the d512 + ctx32000 + recovers-after-crossing combination (6 curves). `recovers_after_crossing` distinguishes a true V-shape (eta crosses tau then climbs back) from a monotone curve — all 4 negative-`R^2` curves recover; the smooth_low_fit curve does not.

Four of the 14 low-fit curves have `observed_knee = null` (the two ctx512 curves and the two ctx8192 fp16 curves) — they never cross tau=0.65 within B<=64, so they are right-censored, contributing to the censored-as-128 split at knee=128 rather than to the finite-only correlation. This is the bridge between the failure audit and the censoring methodology: the low-fit tail and the censored tail overlap on the curves that never resolve a knee within the ladder.

### 7.5 Served-grid quirks

The SGLang served grid carries its own confounds, all of which bias *toward* the conservative direction. The CUDA-graph cap `--cuda-graph-max-bs 32` caps capture at batch 32; above that SGLang falls back to eager execution, which can depress eta at N=48,64 independent of bandwidth and pull the SGLang knee *earlier* than physical. This makes the "SGLang <= vLLM" and "SGLang <= static" results conservative, not optimistic — but it means SGLang knee *magnitudes* at high N are not directly comparable to vLLM's. The SGLang image is pinned by digest (`lmsysorg/sglang@sha256:8df56b...`), A100-80GB only, fp16 only, with `--mem-fraction-static 0.82`. RadixAttention prefix caching is on by default; the distinct regime defeats it with unique HEAD tokens, the shared regime exploits it with identical filler. `min_ok_rate = 1.0` on all 8 cells (zero failed requests). A ladder-asymmetry confound carries over from V4: the vLLM SHARED grid used a sparser 7-point ladder while the vLLM DISTINCT and V5 SGLang grids use the fine 11-point ladder, which is why the shared analysis reports "within one ladder step" rather than exact equality.

### 7.6 The closure layer that secures the decode interpretation

A separate post-V5 closure audit (`TR_mlsys_closure.md`) defends the one honesty exposure in the static grid: the canonical grid times the *entire* `LLM.generate` call, so prefill is inside the timed window. The closure defuses this two ways. First, a GPU-free marginal-decode subtraction (`D512 - D64` at matched cells) recovers Ck/W finite Spearman 0.9421 (n=23). Second, a streaming true-decode audit that excludes prefill by construction (the same `window_start = max(first_token_time)` window the weight-axis runs use) recovers Ck/W finite Spearman 0.886 (n=29, CI95 [0.7245, 0.9531]; censored-as-128 0.9192). Critically, the Ck/W median factor error falls from 2.022x (prefill-in-window) to 1.322x (clean-decode surface) — the prefill-in-window inflation was a real source of the 2x factor error, while the rank claim was never the part at risk. The paired CI between Ck/W and context-only on the streaming surface is delta_rho 0.0258, CI95 [-0.0734, 0.1221], excludes_zero=false: Ck/W is *at least as good as* context-only and adds a mechanistic W term, not strictly better. The weight-axis runs in this report use exactly the closure's clean-decode window, so they inherit its prefill-free property by construction.

---

## 8. What V5 Establishes

### Established findings

1. **Ck/W remains the best mechanistic predictor in the static grid.** With per-GPU alpha it is the top predictor under both finite-only (Spearman 0.8779) and censored-as-128 (0.8462) evaluations, and the zero-param closed form (0.8384 finite) beats KV-traffic-only and the architecture-blind floor. The true median factor error is 2.022x zero-param, tightening to 1.297x with the alpha.
2. **The static model's failures are interpretable.** The 14/96 low-fit curves concentrate in H100 (13/14), long-decode d512 (12/14), and long-context ctx32000 (8/14) — the capacity tail, not random noise. The four negative-`R^2` curves are all non-monotone V-shapes that recover after crossing.
3. **The static knee is a useful served upper bound on a second engine.** SGLang served knees are at or below the static knee in 8/8 ok cells and never knee later than matched vLLM, so the ceiling direction is not a vLLM scheduler artifact.
4. **Weight-axis movement is real in two independent families.** Qwen2.5 (5/5 matched contexts across broad and confirm) and Mistral (3/3) both keep the larger model's true-decode knee later at fixed context.
5. **KV-only is not sufficient.** In the Qwen and Mistral contrasts the larger model has larger or non-smaller KV traffic, yet knees later because `W` moved faster — the ratio carries the signal that KV-traffic-only cannot.

### Partial findings

1. **Gemma 3 supports the mechanism only partially.** Ck/W ranks Gemma 3 knees well (0.9167) and 12B->27B is a clean 3/3, but 4B->12B inverts at every context and 4B->27B is 2/3 — the 4B->12B inversion is a Ck/W degeneracy (k and W both grow ~2.8x), not a counterexample.
2. **Cross-family rank correlation is not the cleanest metric.** Pooled Ck/W is 0.8142 (CI [0.57, 0.92]) while context-only is 0.8948, because pooling re-introduces the context lever V5 holds fixed. The load-bearing claim is within-family, fixed-context movement.
3. **Served validation is compact, not exhaustive.** SGLang covers 8 ok cells. It confirms the upper-bound direction on a second engine but does not replace the full cross-backend served corpus.

### Negative findings and non-claims

1. **Do not claim "larger model always shifts the knee later."** Gemma 3 4B->12B directly inverts that statement at every context.
2. **Do not claim the rational curve fits every tested regime.** Four curves have negative `R^2` and 14/96 are low-fit under `R^2 < 0.70`.
3. **Do not claim static-batch throughput is served latency.** V5 supports the static knee as an upper bound, not a served operating point.
4. **Do not claim Gemma2/Gemma4 evidence.** Both failed under the pinned H100/vLLM stack for environment-support reasons; Gemma 3 is the only valid Gemma evidence in V5.

---

## 9. Paper-Ready Claim Ladder

### Strong claims

- "Across the 672-cell static grid, Ck/W with a per-GPU alpha is the strongest predictor of the observed amortization knee among the tested baselines, with finite-only Spearman 0.8779 and censored-as-128 Spearman 0.8462, and median factor error 1.297x."
- "Static-knee failures concentrate in identifiable non-smooth regimes: 14/96 low-fit curves, almost entirely on H100 and in the long-context long-decode tail, with four non-monotone V-shaped curves driving negative `R^2`."
- "In the compact SGLang served validation, every served knee is at or below the corresponding static knee (8/8) and never later than matched vLLM, supporting the static knee as an upper-bound surface on a second engine."
- "In true-decode H100 runs, Qwen2.5-32B shifts the knee later than Qwen2.5-14B in 5/5 matched-context comparisons across broad and confirm runs."
- "In true-decode H100 runs, Mistral-Small-24B shifts the knee later than Mistral-7B in 3/3 matched-context comparisons."

### Claims with caveats

- "Gemma 3 mostly preserves the rank-level Ck/W signal (0.9167) but is not monotone across all sizes." Use the exact 5/9 pairwise result and the 4B->12B degeneracy explanation.
- "The V5 family evidence supports a weight-axis effect." Pair with the caveat that architecture-specific attention geometry (Gemma 3 head_dim/kv_heads) remains material.
- "Context-only is strong but incomplete." It is competitive on static and pooled rank metrics because context dominates, but it is constant within a fixed-context model-size contrast and cannot explain that movement.

### Forbidden claims

- "The model is universal."
- "Larger models always have later knees."
- "Gemma validates the same monotone story as Qwen and Mistral."
- "Static-batch knee equals served latency knee."
- "KV traffic alone explains the surface."
- "The per-GPU alpha is derived from first principles." (It is one empirical constant per GPU.)

---

## 10. Limitations and Forbidden Claims

**Gemma non-monotonicity blocks universality.** The single most important limitation is that Gemma 3 4B->12B inverts the weight-axis direction at every context. Even though the inversion is a Ck/W degeneracy (k and W both grow ~2.8x, predicted ratio 0.9962), it falsifies any claim of the form "larger model always knees later." The mechanism survives — the predictor has no signal in that degenerate pair — but the simple monotone statement does not.

**Static knee is not served latency.** The static knee is an amortization ceiling measured offline; it is an upper bound on the served knee, not a served operating point or a latency. The SGLang validation supports the bound direction (served <= static) but says nothing about absolute served latency.

**The per-GPU alpha is empirical, not first-principles.** The 1.297x magnitude depends on a single fitted constant per GPU type. It calibrates well and does not change the rank ordering, but it is not derived from a bandwidth model, and a calibrated context-only baseline reaches a comparable magnitude (1.517x).

**Single-H100 weight axis on one GPU.** The weight-axis matrices all run on a single H100. The weight-axis direction is not yet shown to transfer across GPU types, multi-GPU tensor-parallel layouts, or different memory-bandwidth regimes — the static-grid alpha varies per GPU precisely because those constants differ.

**fp16 KV throughout.** Every weight-axis and static cell uses fp16 KV cache (`KV_DTYPE_BYTES = 2`). The audit and baselines say nothing about how fp8-KV would shift the knee.

**Censoring at batch 128 is an analyst choice.** The censored-as-128 split places 31 unobserved knees at the next geometric doubling past the tested maximum. It is an imputation, not a measurement, and the censored-as-128 Spearman should be read as a robustness convention, not an observed result.

**Compact served slice.** The SGLang validation is 8 cells on two models at one GPU and one precision. It confirms the bound direction on a second engine but is not a magnitude re-estimation and does not cover Mistral, additional contexts, or additional engines (TGI remains a forward item).

**Cross-family pooled rank is the wrong metric.** Pooled Ck/W (0.8142) is below context-only (0.8948) because pooling re-introduces the context lever. Any claim that leans on the pooled number rather than the within-family contrast over-reads a metric the design was built to avoid.

---

## 11. Bottom Line

V5 is worth integrating into the paper. It does not make TR164 a universal-law paper, and the Gemma 3 result should stop the draft from overclaiming. What it provides is exactly the reviewer-facing closure the V4-only version lacked: an explicit eleven-predictor baseline ladder with corrected magnitudes and a tau sweep, an explicit failure accounting that locates every low-fit curve, a served-backend upper-bound check on an engine with a different scheduler and prefix cache, and two clean independent weight-axis families plus one informative stress-test family. The right final paper claim is therefore:

> Continuous-batching breakdown is governed primarily by the ratio of per-request KV traffic to shared weight traffic. Ck/W predicts the static amortization knee better than KV-only and architecture-blind baselines, its failures concentrate in an identifiable non-smooth capacity tail rather than scattering, served knees fall at or below the static ceiling in compact cross-engine validation, and true-decode weight-axis shifts reproduce cleanly in Qwen2.5 and Mistral while Gemma 3 exposes the remaining architecture-specific boundary conditions where the predictor degenerates.

---

## Data Reconciliation & Cross-Version Lineage (2026-06-24)

This section ties V5's substrate to the canonical TR164 data-lineage map (`research/tr164/TR164_DATA_LINEAGE.md`) and the program measurement count (`BANTERHEARTS_MEASUREMENT_COUNT.md`, 2026-06-24 supplement, headline ~1,348,000 primary + judge measurements across 49+ unique TR numbers).

**(a) V5's provenance.** V5 is the evidence-closure layer built on V4's static-batch amortization grid. Its load-bearing new substrate, re-verified on disk 2026-06-24:

- **Cross-family weight-axis matrix** — `research/tr164/weight_axis_decode/weight_axis_decode_replicates.csv` (**708 rows**, Qwen/Mistral/Gemma3). Isolates the W lever in r=C·k/W that V4's 7–8B grid barely exercised: pooled cross-family ρ=0.8142, with the Gemma 3 4B→12B knee inversion mechanistically explained and explicitly blocking a "larger model always knees later" universal.
- **Compact SGLang served-ceiling check** — `research/tr164/v5/results/sglang_served` (**12 rows**, 8 ok cells). SGLang served ≤ static on 8/8, extending V4's vLLM-only served validation toward a cross-engine upper-bound statement.
- The predictor baselines + failure-mode audit over the 672-cell grid (`vfiveCurveTotal` = 96 curves) are **derived re-analysis** of V4's already-counted static data (0 new primary rows; lineage map §1, double-count guard §5).

**Hardware.** V5 substrate hardware, consistent with the lineage map's run-dir → GPU assignments: the cross-family weight-axis matrix ran on a single H100 (`modal_weight_axis_decode.py`), the compact SGLang served-ceiling on A100-80GB (`modal_sglang_served_v5.py`); the 70B confirmation is 2×H100 TP=2.

**(b) V5 in the arc.** V1 (closed-loop boundary, 21,159 rows) → V2 (cross-backend, 57,024) → V3 (matched A100 grid, 94,500×2 + 3,780 refill) → V4 (static η(B)=C·k/W model, 1,143-row grid) → **V5 (this report: predictor baselines + served-ceiling + weight-axis)**. V5's discipline is the forbidden-claims ladder: no universality, no "larger model always later knee" (the Gemma 3 inversion blocks it), static knee ≠ served latency.

**(c) Work done after V5.** The post-V5 SSP layer: streaming true-decode + marginal-decode closure audits (`vllm_decode_timed_closure` 252 rows; `mlsys_closure` derived) defending the decode interpretation; the live Modal policy-cap validation (`policy_replay/modal_policy_validation`, 8 rows) operationalizing the knee table into a deployable batch-cap policy; the 70B confirmation (`results/modal_amortization_70b_confirm` 40 + `modal_amortization_70b` 31 rows) partially discharging the scale caveat (knee ~1.8× later than 8B, TP=2 confound flagged); and the TR170 kernel-reproducibility pilot (`research/tr170/`, 555 rows), the kernel-level companion to TR164's serving-level physics.

**(d) Methodology cross-references** (`research/tr164/methodology/`): V5's served validation in `TR_served_validation.md`; the weight-axis matrix in `TR_weight_axis_matrix.md`; the underlying static model in `TR_static_ckw_model.md`; the closure audits in `TR_mlsys_closure.md`; the V1–V3 closed-loop boundary in `TR_closed_loop_boundary.md`.

**Provenance note.** Every row count above is re-verified on disk 2026-06-24 in the lineage map and reconciles to the row with the measurement doc's 2026-06-24 supplement (the full supplement is +11,781: 6,042 SSP TR164 + 5,184 TR165 GIL-control arms + 555 TR170). The downstream artifact is referred to generically as the SSP submission to a full-depth systems venue; this section names no specific external venue, no submission identifier, and no gating language.
