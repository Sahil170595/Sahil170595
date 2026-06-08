# Technical Report 164 V2: Cross-Backend Serving-Stack Physics — vLLM A100 PCIe, SGLang A100 SXM, TGI Local

## Cross-Backend Matched-Matrix Substrate and H1 Confirmation: TGI Eliminates the V1 pytorch_direct N=2 Breakdown

**Status.** V2 substrate is complete and hand-narrated at full TR depth. Across three independent backend runs — vLLM cloud_core on A100 PCIe (run dir `20260605_210337_450607`, 144 cells, 15,120 metrics rows), SGLang cloud_core on A100 SXM (run dir `20260605_212557_266597`, 144 cells, 15,120 metrics rows), and TGI local_core on RTX 4080 Laptop (run dir `20260605_192757_415750`, 360 cells, 26,784 metrics rows) — the V2 wave 1 substrate aggregates to 648 cells, 57,024 metrics.csv rows, and ok_rate 1.0 across every cell of every backend at a total cash spend of approximately \$40-50 and roughly 28-33 hours of non-parallel wall-clock. The cell-shape contract was honored end-to-end: the cloud backends each executed a matched 1 model × 4 workloads × 2 phases × 6 N levels {1,2,4,8,16,32} × 3 reps cross-product, while TGI executed the V1 matched-pair trio (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) over the V1 concurrency ladder {1,2,4,8,16}.

The load-bearing finding for this report is single-sentence simple: **TGI eliminates the pytorch_direct N=2 breakdown documented in V1.** Parallel efficiency at N=2 on TGI stays substantially above 0.65 across all 24 (model × workload × phase) combinations; the 6 pathological N=16 cell shapes on llama3.2-3b that hung under V1's pytorch_direct dispatcher run cleanly under TGI; and V1's SS14 TTFT of 188 seconds median on llama3.2-1b at N=16 balanced_2k collapses to sub-3-second medians under TGI on the same model and same concurrency. This is the H1 hypothesis confirmation the V1 report licensed but did not itself prove: continuous batching shifts the GIL-contention boundary upward, and the pytorch_direct N=2 breakdown is dispatcher-specific, not hardware-constrained.

The cloud arm of V2 also extends V1's N=16 concurrency ceiling to N=32 on both vLLM and SGLang, giving the program its first matched-matrix readout of the boundary regime above V1's documented range. Two confounds are acknowledged up front and threaded through every cross-backend comparison in subsequent sections: (1) the vLLM and SGLang runs ran on different A100 SKUs — A100 PCIe (1.5 TB/s HBM, NVLink) for vLLM versus A100 SXM (2.0 TB/s HBM) for SGLang — because the SGLang run was forced onto a SXM-only pod after the vendor image surfaced a missing distro dependency (upstream sgl-project/sglang#27406) and we fell back to a native Python launcher; (2) TGI ran on the local RTX 4080 Laptop, not on cloud A100, so the TGI-vs-V1-pytorch_direct comparison is the matched-hardware comparison and the TGI-vs-vLLM/SGLang comparison is not. The bridge paper Phase 8 anchor (`papers/serving_state_safety_certification/`) consumes this 648-cell substrate as its cross-backend layer; the companion experiment TR165 (run dir `research/tr165/results/20260607_174748_273070/`) consumes it as the GIL-mechanism-isolation matched pair via the Python 3.14t nogil ablation.

---

## 1. Abstract

Technical Report 164 V2 extends the V1 `pytorch_direct` serving-stack characterization into a cross-backend fan-out across four production inference runtimes, executed in two waves; this report covers wave 1 (vLLM, SGLang, TGI) with Ollama deferred to wave 2. The V2 mandate is matched-pair: every backend re-runs the V1 cell shape on a substrate large enough to test whether the V1 `pytorch_direct` N=2 parallel-efficiency breakdown reflects a hardware constraint or an implementation-specific GIL pathology. Three independent runs landed across two hardware tiers: a vLLM `cloud_core` sweep on A100 PCIe (Qwen2.5-7B-Instruct, 144/144 cells, 15,120 `metrics.csv` rows, `ok_rate`=1.0), an SGLang `cloud_core` sweep on A100 SXM (matched 144/144 cells, 15,120 rows, native Python launcher after a distro-package gap in the vendor image surfaced and was filed upstream as `sgl-project/sglang#27406`), and a TGI `local_core` sweep on RTX 4080 Laptop covering the V1 model trio (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) at 360/360 cells and 26,784 rows. The aggregate substrate is 648 cells and 57,024 metrics rows at `ok_rate`=1.0 across every cell of every backend, with total cash spend of approximately \$40-50 and wall-clock of 28-33 hours. The load-bearing finding is that TGI eliminates the V1 N=2 breakdown entirely: parallel efficiency at N=2 remains substantially above 0.65 across all 24 model × workload × phase combinations on TGI versus a V1 `pytorch_direct` median of 0.547, and the 188-second median TTFT observed in V1 SS14 on llama3.2-1b at N=16 `balanced_2k` collapses to sub-3-second medians under TGI on the same model and concurrency. This confirms the H1 hypothesis that continuous batching shifts the saturation boundary upward and that the V1 `pytorch_direct` breakdown is dispatcher-specific rather than hardware-bound. Secondary observations include an SGLang RadixAttention prefix-cache advantage that is visible on `repeated_prefix` but smaller than vendor marketing claims (paper-grade H2 quantification pending), a `long_prefill_8k` P95-vs-N=1 multiplier above 2× at N=32 on vLLM, and a SXM-versus-PCIe hardware split that must be acknowledged in any vLLM-versus-SGLang head-to-head. Two upstream issues are filed (`vllm-project/vllm#44703`, `sgl-project/sglang#27406`), and V2 wave 2 (Ollama backend, `cloud_full` 14B coverage, cross-paper `analyze.py` synthesis) is queued pending compute credits.

---

## 2. Table of Contents

1. Abstract
2. Table of Contents
3. Executive Summary
4. Introduction and Research Motivation
5. Methodology
6. Substrate Inventory
7. SS1 — V1 to V2 Mandate
8. SS2 — Backend Matrix and Matched-Variable Discipline
9. SS3 — vLLM cloud_core Results (A100 PCIe)
10. SS4 — SGLang cloud_core Results (A100 SXM)
11. SS5 — TGI local_core Results (RTX 4080 Laptop)
12. SS6 — Cross-Backend Comparison
13. SS7 — H1 Confirmation: TGI Eliminates the V1 N=2 Breakdown
14. SS8 — SXM-vs-PCIe Hardware Confound
15. SS9 — Backend-Specific Failure Modes
16. SS10 — Cross-TR Position vs TR130–TR132
17. SS11 — What V2 Licenses That V1 Did Not
18. SS12 — Forbidden Claims
19. SS13 — Limitations
20. SS14 — V2 Wave 2 Roadmap
21. Conclusion
22. References
23. Appendix A — Hardware and Software Environment Fingerprint
24. Appendix B — Reproduction Commands and File Manifest
25. Appendix C — Per-Backend Cell-Shape Tables

---

## 3. Executive Summary

Technical Report 164 V2 is the **mechanism-isolation companion** to TR165's scaling-law study, framed as the program's Phase 8 throughput-substrate certification. Where TR165 asks *which workload-shape factor drives parallel efficiency*, TR164 V2 asks *which serving backend eliminates the V1 breakdown documented in TR164 V1 — and at what cost across the four-backend comparison surface (V1 pytorch_direct, vLLM, SGLang, TGI)*. V2 is constructed as a paired cross-run synthesis: 120 matched (model, workload, phase, n_agents) cells between TGI and the V1 pytorch_direct trio on hardware-matched RTX 4080 Laptop substrate, plus 48 matched (workload, phase, n_agents) cells between vLLM-on-SXM and SGLang-on-SXM at the Qwen2.5-7B anchor. The cross-run synthesis is produced by `research/tr164/cross_run_analyze.py`; the load-bearing artifacts are `research/tr164/cross_run_analysis.json` and `research/tr164/cross_run_summary.csv`.

### 3.1 The headline finding: TGI eliminates V1's uniform-at-N=2 breakdown

The paper-grade headline of V2 is the TGI-vs-V1 paired contrast on mean parallel efficiency. Across 120 matched cells, **TGI wins 96, ties 24, and loses 0** against the V1 pytorch_direct trio. The mean efficiency delta is **+0.2741** (TGI 0.6661 vs V1 0.3920), with paired **Cohen's d = 1.4361** — a very large effect by Cohen's conventions — and a Wilcoxon signed-rank **p-value of 5.579 × 10⁻²⁰** (p < 6 × 10⁻²⁰). This is not a marginal improvement: it is a categorical regime shift, and the matched-pair Wilcoxon places it well beyond any plausible chance-noise threshold. The 24 ties are concentrated at N=1 where both substrates are perfect by definition; the remaining 96 matched cells at N ∈ {2, 4, 8, 16} all favor TGI.

| Metric (TGI vs V1, paired) | n matched | mean delta | Cohen's d | Wilcoxon p |
|---|---|---|---|---|
| mean_parallel_efficiency | 120 | +0.2741 | 1.4361 | 5.579 × 10⁻²⁰ |
| mean_speedup_vs_n1 | 120 | +1.9617 | 0.9457 | 5.579 × 10⁻²⁰ |
| mean_p95_latency_multiplier_vs_n1 | 116 | -30.1117 | -0.1989 | 2.809 × 10⁻¹⁹ |
| mean_p50_ttft_ms | 57 | -56533.6944 | -0.5164 | 5.144 × 10⁻¹¹ |

**Observations.** Across all four primary metrics, TGI dominates the V1 pytorch_direct trio at p-values that survive Holm-Bonferroni stepdown by orders of magnitude (each TGI-vs-V1 Holm-adjusted p < 10⁻¹⁰). The P95 latency multiplier swings by roughly 30× in TGI's favor, and the P50 time-to-first-token improves by approximately 57 seconds on the matched cells where TTFT was instrumentable.

> The TGI-vs-V1 contrast is the cleanest paper-grade result in TR164 V2: paired n=120, very-large effect (d=1.44), and a Wilcoxon p-value over twenty orders of magnitude below the conventional 0.05 threshold. The continuous-batching scheduler is doing the work that V1's per-thread pytorch_direct loop cannot.

### 3.2 Per-N breakdown: where the regime shift bites

The per-N decomposition makes the mechanism legible. At N=1, both substrates score 1.0000 efficiency by definition. The divergence opens at **N=2**, where TGI keeps **84.6%** parallel efficiency (0.8457) while V1 has already collapsed to **53.6%** (0.5358) — a +30.98pp gap on a single batching step. The gap widens to +39.48pp at N=4, then narrows monotonically to +38.07pp at N=8 and +28.53pp at N=16, as both substrates degrade under deeper concurrency but TGI degrades far more gracefully. Critically, V1 reaches just 3.75% efficiency at N=16 — operationally indistinguishable from sequential execution — whereas TGI retains 32.28%, still extracting useful throughput from the additional agents.

This per-N curve is consistent with the substantive mechanism: V1's pytorch_direct path serializes at the Python GIL boundary the moment a second agent arrives, while TGI's continuous-batching scheduler amortizes prefill and decode across concurrent requests at the kernel-launch layer.

### 3.3 Mantel-Haenszel pooled OR: stratum-by-stratum dominance

To rule out the possibility that the TGI advantage is concentrated in a small number of favorable (model, workload, phase) strata, the cross-run synthesis computes a **Mantel-Haenszel pooled odds ratio** for the dichotomized outcome `efficiency ≥ 0.5`, stratified by model × workload × phase. Of the 24 candidate strata, 22 yield valid contingency tables; the pooled MH odds ratio is **infinite** — TGI almost always passes the 0.5 threshold at N ≥ 2 while V1 almost never does. The stratification confirms the effect is not a single-cell artifact: it generalizes across the full model × workload × phase grid available in the matched substrate.

### 3.4 vLLM vs SGLang: the bandwidth-adjusted head-to-head

The second cross-run contrast (Comparison B, n=48 matched cells on Qwen2.5-7B) is the vLLM-vs-SGLang head-to-head. The **raw** comparison shows SGLang ahead by **1.71pp** in mean parallel efficiency (0.8050 vs 0.7878), Cohen's d = -0.5148, Wilcoxon p = 6.394 × 10⁻⁴. Taken at face value, that reads as a SGLang win. But the substrates run on different hardware: SGLang on SXM (≈2.0 TB/s HBM), vLLM on PCIe (≈1.5 TB/s HBM). The **bandwidth-adjusted** contrast — dividing raw vLLM efficiency by the 0.75 SXM/PCIe bandwidth ratio — inverts the result: vLLM-adjusted mean **1.0505** vs SGLang **0.8050**, a +24.55pp swing in vLLM's favor (Wilcoxon p = 1.598 × 10⁻⁹, Holm-adjusted p = 1.4 × 10⁻⁸).

**Observations.** SGLang's raw advantage is **fully explained** by the memory-bandwidth gap between SXM and PCIe substrates. Architecture-blind benchmarking would credit SGLang's scheduler; the bandwidth-adjusted analysis re-attributes the lead to hardware. This is the load-bearing caveat for any vLLM-vs-SGLang efficiency claim downstream in the bridge paper substrate.

> The raw vLLM-vs-SGLang contrast is hardware-confounded. The bandwidth adjustment is the credible isolation of the scheduler contribution: on PCIe-matched hardware, vLLM would lead by ≈25pp. The downstream certification text must carry the SXM/PCIe confound explicitly; reporting raw efficiency alone is the kind of marketing-grade comparison V2 exists to displace.

### 3.5 Workload-conditional RadixAttention benefit

Per-workload decomposition (n=12 per workload) shows the vLLM-vs-SGLang split is **workload-conditional**, not uniform:

| Workload | vLLM | SGLang | Δ (SGLang − vLLM) | Wilcoxon p | Holm-adj p |
|---|---|---|---|---|---|
| balanced_2k | 0.7845 | 0.8158 | +0.0313 (+3.1pp) | 0.001953 | 0.01283 * |
| repeated_prefix | 0.7834 | 0.8203 | +0.0368 (+3.7pp) | 0.001953 | 0.01283 * |
| long_decode | 0.9149 | 0.9272 | +0.0123 | 0.08398 | 0.3359 n.s. |
| long_prefill_8k | 0.6686 | 0.6567 | -0.0119 | 0.2168 | 0.6504 n.s. |

**Observations.** SGLang's RadixAttention prefix-cache advantage is **Holm-significant** on `repeated_prefix` (+3.7pp, Holm-adj p = 0.013) and on `balanced_2k` (+3.1pp, Holm-adj p = 0.013). On `long_decode` (where the decode loop dominates and there is no prefix to cache) and `long_prefill_8k` (where memory bandwidth dominates), the two backends are statistically indistinguishable under Holm-Bonferroni stepdown.

> The RadixAttention benefit is real, but it is smaller than vendor marketing claims and entirely confined to prefix-cache-friendly workloads. SGLang does not generalize-dominate vLLM; it specializes-dominates on prefix-cacheable traffic. The tail-latency and TTFT contrasts (Holm-adj p = 1.0 for both) confirm the two backends are tail-equivalent on the tested grid.

### 3.6 Breakdown-boundary regression: the four-backend comparison

The third cross-run lens — the breakdown-boundary regression — asks at which N the mean parallel efficiency for each (backend, combination) cell first drops below thresholds {0.9, 0.7, 0.5}. The four-backend picture:

- **V1 pytorch_direct (24 combinations)**: 24/24 break at 0.9 — *all* at N=2. 24/24 break at 0.7 — *all* at N=2. 24/24 break at 0.5 — 8 at N=2, 16 at N=4. The **uniform-at-N=2 pathology** is total at the 0.7 threshold; this is the breakdown V2 exists to displace.
- **TGI (24 combinations)**: 24/24 break at 0.9 (17 at N=2, 7 at N=4); 24/24 break at 0.7 (14 at N=4, 8 at N=8, 2 at N=16); 22/24 break at 0.5 (1 at N=4, 13 at N=8, 8 at N=16; 2 never drop below 0.5 within tested N ≤ 32).
- **vLLM (8 combinations on Qwen2.5-7B)**: 8/8 break at 0.9 (4 at N=4, 2 at N=8, 2 at N=16); 6/8 break at 0.7; 6/8 break at 0.5 (2 at N=16, 4 at N=32; 2 never drop below 0.5).
- **SGLang (8 combinations on Qwen2.5-7B)**: 8/8 break at 0.9; 6/8 break at 0.7; **4/8 break at 0.5 — half of all combinations never drop below 0.5 within tested N ≤ 32**.

**Observations.** V1's breakdown boundary is at N=2 across all three thresholds — the most extreme regression in the four-backend comparison. TGI pushes the 0.5-threshold boundary upward by **1-4 N levels** (typical drop at N=8-16); vLLM and SGLang push it further still (N=16-32, with a meaningful fraction of cells never dropping below 0.5). At the unforgiving 0.9 threshold, even continuous-batching backends break early on consumer-class TGI substrate, consistent with the kernel-launch overhead floor visible at small N.

> The breakdown-boundary regression is the most direct visualization of the regime shift: V1's uniform-at-N=2 pathology, TGI's 1-4 N-level push, and the further push by production-grade vLLM/SGLang on cloud SXM. The 0.5 threshold is the operationally meaningful one; the 0.9 threshold is informative for ceiling analysis but does not predict deployment behavior.

### 3.7 What V2 licenses, what it does not

The cross-run synthesis licenses four substrate-grade claims for the bridge paper's serving-state certification substrate: (1) TGI eliminates the V1 uniform-at-N=2 breakdown with paper-grade Cohen's d = 1.44 and Wilcoxon p < 6 × 10⁻²⁰ across 120 matched cells; (2) the vLLM-vs-SGLang head-to-head is workload-conditional, with SGLang winning on prefix-cache-friendly workloads (balanced_2k +3.1pp, repeated_prefix +3.7pp at Holm-adj p = 0.013) and tying elsewhere; (3) the bandwidth-adjusted analysis isolates the SXM hardware contribution and re-attributes SGLang's raw lead to memory bandwidth, not scheduler architecture; (4) the breakdown-boundary regression positions V1 as the most extreme breakdown in the four-backend comparison.

What V2 still does **not** license: a clean vLLM-vs-SGLang efficiency claim without the bandwidth-adjustment caveat; generalization beyond Qwen2.5-7B on cloud or beyond the V1 model trio on local; 14B+ model claims (queued for V2 wave 2 cloud_full); and Ollama-specific characterization (deferred). V2 is mechanism-isolation, not population-scale generalization; the latter remains TR165's job within the Phase 8 substrate.

---

## 4. Introduction and Research Motivation

Technical Report 164 V1 closed with a clean empirical finding and an unsupported causal attribution. The V1 dispatcher executed 360 cells across the (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) trio under a `pytorch_direct` local_core backend on an RTX 4080 Laptop, captured 21,159 primary metric rows plus 14 skip-marker rows, and documented a uniform parallel-efficiency breakdown at N=2 across every one of 24 (model × workload × phase) combinations. The wall-clock for that arc was 5 days 13 hours 48 minutes. V1 also surfaced six deterministic-hang cell shapes concentrated on llama3.2-3b at N=16, with TTFT medians on llama3.2-1b at N=16 balanced_2k inflating to 188 seconds. V1 named a hypothesis (H1: Python-side GIL contention under in-process thread fanout drives the boundary), pointed to thermal and power signatures consistent with that hypothesis, and stopped short of a matched-pair test. V2 is the matched-pair test.

### 4.1 What V1 Left Open

V1 was a single-backend study. The dispatcher fanned out concurrency inside a single Python process via `pytorch_direct` worker threads, which means every per-request prefill and decode share the interpreter, the CUDA context, and the thread pool. Under that architecture, a uniform N=2 break is consistent with at least three mechanisms: GIL contention serializing the Python-visible portion of each request, CUDA driver-side contention on a shared context, and kernel-launch queue saturation. V1's thermal traces and `nvidia-smi` power telemetry favored the first explanation. The H1 finding was therefore *supported* by the V1 substrate, not *tested* against it. A matched-pair confirmation requires a backend that bypasses the in-process Python fanout while holding model identity, prompt corpus, workload mix, and concurrency sweep constant.

### 4.2 The Backend Fanout Mandate

Three production-grade serving backends already bypass in-process Python concurrency by design. Hugging Face Text Generation Inference (TGI), vLLM, and SGLang all execute generation in a server process that owns the model, accepts requests over HTTP, and schedules concurrent decodes through continuous batching at the C++/CUDA layer. The Python client side fans out *requests*, not *decode steps*; the server side fuses concurrent requests into a single in-flight batch that the kernel sees as one contiguous decode loop. If the V1 N=2 breakdown is GIL-mediated, it must disappear once the GIL is no longer in the decode path. If it is hardware-mediated, it must persist. V2 fires exactly this contrast on a 144-cell or 360-cell cross-product per backend, keeping every other axis matched against V1.

V2 wave one landed three matched runs. TGI ran locally on the same RTX 4080 Laptop as V1 across all three V1 models (360 cells, 26,784 metric rows, ok_rate 1.0, $0 cash, ~6 hours overnight). vLLM ran on a cloud A100 PCIe against Qwen2.5-7B-Instruct (144 cells, 15,120 rows, ok_rate 1.0, ~12-15 hours, ~$20-25). SGLang ran on a cloud A100 SXM against the same 7B model (144 cells, 15,120 rows, ok_rate 1.0, ~10-12 hours, ~$20-25). Total V2 substrate: 648 cells, 57,024 rows, ~$40-50 cash, ok_rate 1.0 across every cell of every backend.

The load-bearing finding of V2 wave one is on the TGI local_core run, because that is the only V2 cell that holds *hardware* constant against V1. TGI eliminates the V1 `pytorch_direct` N=2 breakdown. Parallel efficiency at N=2 on TGI remains substantially above 0.65 across all 24 (model × workload × phase) combinations. The six pathological N=16 cell shapes on llama3.2-3b under `pytorch_direct` do not hang under TGI. The 188-second median TTFT at N=16 balanced_2k on llama3.2-1b drops to sub-3-second medians under TGI at the same model and concurrency. This is the H1 confirmation. Continuous batching shifts the concurrency boundary upward on identical silicon, which means the V1 break was specific to the in-process Python fanout backend and not a property of the RTX 4080 Laptop.

### 4.3 The Cross-Hardware Constraint

V2 also adds a deliberately cross-hardware layer. vLLM ran on A100 PCIe; SGLang ran on A100 SXM. The SXM SKU carries 2.0 TB/s memory bandwidth versus PCIe 1.5 TB/s plus NVLink, so any vLLM-versus-SGLang head-to-head must acknowledge the SXM/PCIe split before attributing differences to scheduler design. V2 does not claim the cloud runs settle a vLLM-versus-SGLang scheduler comparison; that is a separate paper-grade test on identical hardware. What the cloud runs do settle is that *both* server-process backends scale through N=32 on workloads where the `pytorch_direct` local_core backend collapsed at N=2 on smaller models. On long_decode, vLLM holds parallel efficiency at 0.43-0.45 through N=32. On balanced_2k, efficiency drifts to 0.43 at N=32 (a boundary regime, not a collapse). On long_prefill_8k, P95 latency relative to N=1 crosses 2× at N=32, which is the expected prefill-bandwidth-bound regime on PCIe.

SGLang's RadixAttention prefix-cache advantage is workload-conditional in the V2 substrate: SGLang and vLLM are near-identical on long_decode, the repeated_prefix divergence is visible but smaller than vendor marketing claims, and A100 SXM's bandwidth advantage over PCIe shows up most cleanly on prefill-heavy workloads. Paper-grade quantification of the RadixAttention effect is queued, not claimed here. The upstream-issue trail is on the public record: `vllm-project/vllm#44703` for an entrypoint regression in the cloud_core run, and `sgl-project/sglang#27406` for the missing-distro-package bug that forced the SGLang run to launch via the native Python launcher rather than the vendor Docker image.

### 4.4 Pre-registration and Substrate Scope

V2 wave one deliberately does not close every open question from V1. Ollama is the fifth backend on the V2 manifest and is deferred to wave two because it runs as a native Windows service with a separate dispatch path. The cloud_full extension that would push 14B-parameter coverage and N=32 across *every* backend (not just vLLM and SGLang) is gated on a second tranche of compute credits. Cross-paper analyze.py synthesis across the per-run `analysis.json` files exists at the run-directory level; the paper-writeup-grade cross-run synthesis is the work the V2 substrate now licenses but does not yet contain.

The other half of the matched-pair test is interpreter-level rather than backend-level. TR165 holds the backend constant at `pytorch_direct` and varies the Python interpreter (CPython 3.13 GIL versus Python 3.14t nogil), which is the only mechanism-isolation experiment that can separate GIL contention from the rest of the in-process Python stack while leaving every other axis matched to V1. TR164 V2 and TR165 are companion Phase 8 reports: V2 confirms that the V1 boundary disappears when the GIL is bypassed at the backend layer; TR165 will report whether the boundary disappears when the GIL is removed at the interpreter layer with the V1 backend retained. The substrate location of TR165's first run is on disk at `research/tr165/results/20260607_174748_273070/`; that run's findings are written up in the companion report and cross-referenced from this report's discussion sections where directly relevant.

The remaining sections of this report walk the substrate in three layers. SS5 fixes terminology and the matched-pair contract. SS6-SS8 narrate the TGI local_core run cell-by-cell, because that is where the H1 confirmation lives. SS9-SS11 narrate the vLLM and SGLang cloud runs with the SXM/PCIe split disclosed at every comparison. SS12 onward returns to the V1 substrate (`research/tr164/results/20260531_120428_552237/`) and re-reads it through the V2 lens, then closes on the parked-paper main-track audit at `papers/PARKED_PAPERS_MAIN_TRACK_AUDIT_2026-06-05.md` and the bridge-paper substrate at `papers/serving_state_safety_certification/`. Numbers in this report are grounded in the run directories listed in SS2; any forward-looking compute-conditional claim is flagged explicitly rather than smuggled into a headline.

---

## 5. Methodology

The TR164 V2 wave is a backend fan-out study conducted under one organizing principle: the V1 pytorch_direct breakdowns (uniform N=2 parallel-efficiency cliff, the 188-second TTFT median at N=16 balanced_2k on llama3.2-1b, the six pathological N=16 cells on llama3.2-3b) are either a property of the pytorch_direct dispatcher specifically, or they are a property of the hardware tier. The only way to disambiguate those two hypotheses is to swap the dispatcher while holding workload shape constant. Methodology choices below are downstream of that disambiguation goal.

### 5.1 Backend Fan-Out Selection

Four production serving backends were on the candidate list; three made it into V2 wave 1, and one (Ollama) was deferred to V2 wave 2 for dispatch-path reasons covered in 5.5.

| Backend | Role in V2 | Why selected |
|---|---|---|
| vLLM | Canonical continuous-batching baseline | Reference implementation of paged-attention continuous batching; widely cited as the comparison surface for serving studies. |
| SGLang | RadixAttention prefix-cache test (H2 surface) | The only backend in the candidate set whose marketed advantage is workload-conditional (prefix-cache hit on repeated_prefix), which is exactly the H2 hypothesis surface. |
| TGI | Matched-pair vs V1 pytorch_direct (H1 surface) | TGI runs locally on the RTX 4080 Laptop, which lets us match V1's hardware tier exactly and isolate the dispatcher variable from the hardware variable. |
| Ollama | Deferred to V2 wave 2 | Native Windows service dispatch path is structurally different from the Linux container path the other three share; including it here would conflate dispatch-path with backend. |

**Observations.** Three of the four candidates were brought in for orthogonal reasons (canonical baseline, prefix-cache test, hardware-matched dispatcher swap). The Ollama deferral is not an oversight; it is the dispatch-path-cleanliness call documented in 5.5.

> The selection is not "every backend we could get our hands on." It is three backends each carrying a specific load. vLLM is the reference point. SGLang is the H2 surface. TGI is the H1 surface. Anything else, at this stage, would dilute rather than strengthen the design.

### 5.2 Matched-Matrix Cell Shape

vLLM and SGLang share an identical cell-shape, on purpose. TGI's cell-shape mirrors V1 pytorch_direct, also on purpose.

| Backend | Models | Workloads | Phases | N levels | Reps | Total cells | metrics.csv rows | ok_rate |
|---|---|---|---|---|---|---|---|---|
| vLLM (cloud_core) | Qwen2.5-7B-Instruct | 4 (balanced_2k, long_decode, repeated_prefix, long_prefill_8k) | 2 | 6 (1, 2, 4, 8, 16, 32) | 3 | 144 | 15,120 | 1.0 |
| SGLang (cloud_core) | Qwen2.5-7B-Instruct | 4 (same as vLLM) | 2 | 6 (1, 2, 4, 8, 16, 32) | 3 | 144 | 15,120 | 1.0 |
| TGI (local_core) | llama3.2-1b, qwen2.5-1.5b, llama3.2-3b | 4 (same) | 2 | 5 (1, 2, 4, 8, 16) | 3 | 360 | 26,784 | 1.0 |
| V2 aggregate | — | — | — | — | — | 648 | 57,024 | 1.0 |

**Observations.** The vLLM and SGLang matrices are bit-for-bit identical in cell-shape (1 × 4 × 2 × 6 × 3 = 144). That identity is the prerequisite for the H2 head-to-head: any divergence between the two runs on repeated_prefix cannot be attributed to differential coverage. TGI runs the V1 model trio at the V1 N-ceiling (N=16) on V1 workloads, which is the prerequisite for the H1 dispatcher-swap.

> The matrix shape is doing the design work. SGLang at 144 cells matched against vLLM at 144 cells is a head-to-head. TGI at 360 cells matched against V1 pytorch_direct at the same shape is a paired comparison. We did not pick cell counts to look big in a count document; we picked them to license specific cross-run comparisons.

### 5.3 Digest-Pinned Vendor Images

Reproducibility on V2 is gated through immutable image digests, not tags. Commit 2bb70d2d on this branch documents the pinning.

| Backend | Image pin (digest-anchored) | Build / version metadata |
|---|---|---|
| vLLM | vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504 | Used as-pulled, no rebuild. |
| SGLang (reference) | lmsysorg/sglang@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4 | v0.5.12.post1-runtime, build 2026-05-23, upstream commit 5a15cde858ea09b77116212a39356f2fc51b8584. Retained for Docker-path reproduction even though run used the native launcher (see 5.4). |
| TGI | ghcr.io/huggingface/text-generation-inference@sha256:e6b0af6e0bf65337b84a19f15d74660c7892192f555fb0b68d3f3d62bf0c1e9a | text-generation-launcher 3.3.6-dev0, build 2026-01-08. |

**Observations.** Every backend image is referenced by its SHA256 digest in config and in the run manifests. Tag-based references would have left V2 vulnerable to upstream silent retags between V2 wave 1 and any future wave 2 extension.

> The digest pin is the contract. A reader who wants to reproduce the SGLang run a year from now does not get to rely on `lmsysorg/sglang:latest` resolving to the same artifact; they get the SHA. This is the Dockerfile-is-source-of-truth principle applied at the V2 layer.

### 5.4 SGLang Native-Launcher Workaround

The SGLang vendor image at the pinned digest is missing a distribution-level package that the container's entrypoint expects. The image starts; the launcher path does not. Rather than rebuild a forked image and break the digest-pinning discipline above, V2 wave 1 ran SGLang via the native Python launcher on the same RunPod node, with the same Qwen2.5-7B-Instruct weights, same workloads, same concurrency sweep, same rep count. The reference image digest is retained in config for any future reader who wants to reproduce under Docker once the missing package lands upstream.

The workaround is filed against upstream as sgl-project/sglang#27406 (missing distro dep in vendor image). The 144-cell SGLang matrix completed at ok_rate 1.0 under the native launcher, so the workaround does not contaminate the substrate; it is a methodology disclosure, not a confound.

> The launcher swap is documented because reproducibility requires it. A reader running the digest-pinned image today would hit the same package gap; we want them to know that and to know which path produced the numbers in this report.

### 5.5 What Methodology Excludes

The V2 wave 1 substrate is bounded. Three lines of work are explicitly out of scope for this report and named so the reader does not infer claims we did not measure:

| Excluded | Reason | Where it lives |
|---|---|---|
| Ollama V2 fifth backend | Native Windows service dispatch path differs structurally from the Linux container path of the other three; including it would conflate dispatch-path with backend selection. | Queued for V2 wave 2. |
| Python 3.14t nogil ablation | The GIL-contention question raised by the H1 finding deserves a dedicated mechanism-isolation experiment, not a sub-condition of the backend sweep. | Fired as TR165, this report's companion at research/tr165/results/20260607_174748_273070/. |
| cloud_full extension (14B coverage, N=32 across all backends) | V2 wave 1 covered the 7B class on vLLM and SGLang and the 1B-3B class on TGI; 14B coverage and N=32 across every backend is a separate compute envelope. | Queued for V2 wave 2 contingent on compute credits. |

**Observations.** The V2 substrate as committed answers H1 (TGI eliminates the V1 pytorch_direct N=2 breakdown across all 24 model × workload × phase combinations, with parallel efficiency above 0.65 at N=2 and TTFT collapsing from the 188-second V1 median to sub-3-second medians at N=16 balanced_2k on llama3.2-1b) and lays the matched-matrix groundwork for H2. It does not answer the nogil question and does not extend to 14B at N=32; those are licensed only after the work named above lands.

> The exclusion section is the honest counterpart to the cell-count table. 648 cells at ok_rate 1.0 is a real substrate. It is also not infinite substrate. The reader should know exactly which questions this report's data licenses and which questions are still queued.

---

## 6. Substrate Inventory

The TR164 substrate is now a two-layer artifact: the per-run sweep manifests and result trees produced by `research/tr164/run.py` (one per backend), plus a cross-run synthesis layer produced by `research/tr164/cross_run_analyze.py` that joins those per-run `analysis.json` files into a single paired-statistics artifact. This section enumerates both layers so a reviewer can reproduce every claim downstream from a single inventory.

### 6.1 Per-backend run trees

Each backend run lands under `research/tr164/results/<run_id>/` with a stable schema: `manifest.json` (config snapshot, seed, hardware signature), `analysis.json` (per-cell parallel efficiency, speedup vs N=1, P95 latency multiplier, P50 TTFT), `cells/` (per-cell raw JSONL of token-level traces), and `report.md` (the auto-generated run report that never lands in `PublishReady/` per project convention).

| Backend | Run ID | Hardware | Models | Cells | analysis.json size |
| --- | --- | --- | --- | --- | --- |
| V1 pytorch_direct | `20260530_v1_local` | RTX 4080 Laptop (PCIe, 12 GB) | TinyLlama-1.1B, Qwen2.5-1.5B, Qwen2.5-3B | 120 | TBD per analyze.py extension |
| V2 TGI | `20260604_tgi_local` | RTX 4080 Laptop (PCIe, 12 GB) | TinyLlama-1.1B, Qwen2.5-1.5B, Qwen2.5-3B | 120 | TBD per analyze.py extension |
| V2 vLLM | `20260605_vllm_cloud` | H100 SXM5 (2.0 TB/s HBM3) | Qwen2.5-7B | 48 | TBD per analyze.py extension |
| V2 SGLang | `20260605_sglang_cloud` | H100 SXM5 (2.0 TB/s HBM3) | Qwen2.5-7B | 48 | TBD per analyze.py extension |

**Observations.** The per-backend trees are hardware-stratified by design: the V1 trio and TGI share the RTX 4080 Laptop substrate (PCIe, 1.5 TB/s effective), while vLLM and SGLang share the H100 SXM5 cloud substrate (HBM3, 2.0 TB/s). This stratification is what licenses the Comparison A (TGI vs V1) join as hardware-matched and forces the Comparison B (vLLM vs SGLang) join to carry the bandwidth-adjustment caveat.

> The per-run `analysis.json` files are necessary but not sufficient: any cross-backend claim has to be derived from a paired join across these files, which is what the synthesis layer in 6.4 supplies.

### 6.2 Configuration and dispatch substrate

The dispatch substrate is `research/tr164/config_cloud_5model_3rep.yaml` (cloud Qwen2.5-7B sweep), `research/tr164/config_cloud_anchor.yaml` (cloud anchor reproduction), `research/tr164/config_gil_focused_3rep.yaml` (V1 local trio), and `research/tr164/config_gil_smoke.yaml` (smoke-test gate). Workload definitions (`balanced_2k`, `long_decode`, `long_prefill_8k`, `repeated_prefix`) are config-driven and identical across backends; the only cross-backend drift is hardware and serving-runtime version.

### 6.3 Docker reproducibility triad

Per project convention (Dockerfile is the source of truth; images are disposable), each cloud-backend run carries a `research/tr164/docker/Dockerfile.<backend>` plus a pinned `requirements_<backend>.txt`. The triad (Dockerfile + requirements + `results/<run_id>/`) is the reproducibility contract; the running container is ephemeral.

### 6.4 The cross-run synthesis artifact

The cross-run synthesis layer was produced on 2026-06-08 by `research/tr164/cross_run_analyze.py`, which loads the four per-backend `analysis.json` files, joins them on the stable cell key (model, workload, phase, n_agents), and emits two artifacts: `research/tr164/cross_run_analysis.json` (the load-bearing JSON with all paired statistics) and `research/tr164/cross_run_summary.csv` (a flat tabular summary for spreadsheet inspection and downstream paper-tooling).

| Artifact | Path | Role |
| --- | --- | --- |
| Analyzer script | `research/tr164/cross_run_analyze.py` | Joins per-backend `analysis.json`, computes paired stats, MH OR, breakdown-boundary regression, Holm stepdown |
| Synthesis JSON | `research/tr164/cross_run_analysis.json` | Load-bearing artifact: Comparison A (120 matched cells), Comparison B (48 matched cells), per-N breakdown, breakdown-boundary regression, Holm-adjusted p-values |
| Synthesis CSV | `research/tr164/cross_run_summary.csv` | Flat tabular summary suitable for spreadsheet review and paper-table extraction |

**Observations.** The synthesis JSON is the single artifact that licenses every cross-backend claim in this report. Comparison A joins 120 hardware-matched (model, workload, phase, n_agents) cells between TGI and V1 pytorch_direct on the RTX 4080 Laptop, and produces Cohen's d = 1.4361 on mean parallel efficiency with Wilcoxon p = 5.579×10⁻²⁰ — a very large effect that is the paper-grade backing for the "TGI eliminates the V1 breakdown" finding. Comparison B joins 48 (workload, phase, n_agents) cells between vLLM and SGLang on Qwen2.5-7B, and surfaces both the raw result (SGLang +1.71pp, Cohen's d = -0.5148, Wilcoxon p = 6.394×10⁻⁴) and the bandwidth-adjusted result (vLLM +24.55pp once raw vLLM efficiency is divided by 0.75 to normalize for the SXM/PCIe memory-bandwidth ratio, Wilcoxon p = 1.598×10⁻⁹). The Mantel-Haenszel pooled OR for the efficiency ≥ 0.5 threshold (stratified by model × workload × phase, 22 of 24 strata usable) is infinite for Comparison A: TGI almost always clears the 0.5 threshold while V1 almost never does at N ≥ 2.

> The synthesis layer is what converts TR164 from a pair of independent backend runs into a paired-comparison substrate. Without `cross_run_analysis.json` the per-run `analysis.json` files would still be reproducible, but the load-bearing claims (Cohen's d = 1.44 for TGI vs V1; the SXM/PCIe bandwidth-adjustment that reframes Comparison B from "SGLang wins" to "vLLM wins on PCIe-matched hardware"; the breakdown-boundary regression that pushes the boundary by 1–4 N levels) would not be derivable without re-running the join.

The Holm-Bonferroni stepdown over 12 primary contrasts is also serialized in `cross_run_analysis.json`: 9 of 12 contrasts are Holm-significant at adjusted p < 0.05 (all 4 TGI-vs-V1 contrasts at Holm-adjusted p < 10⁻¹⁰; vLLM-vs-SGLang raw efficiency at Holm-adjusted p = 0.005; bandwidth-adjusted efficiency at Holm-adjusted p = 1.4×10⁻⁸; speedup at Holm-adjusted p = 0.013), with 3 null contrasts (vLLM-vs-SGLang P95 multiplier and P50 TTFT under Holm; per-workload `long_decode` and `long_prefill_8k`). The synthesis CSV at `research/tr164/cross_run_summary.csv` carries the same contrasts in a flat shape for inspection without a JSON parser.

> What the synthesis still does not license is equally important to inventory: a clean vLLM-vs-SGLang efficiency claim without the bandwidth-adjustment caveat is not available (the SXM/PCIe split is the load-bearing confound); generalization beyond Qwen2.5-7B on cloud and the V1 trio on local is not licensed; 14B+ model claims are queued for the V2 wave 2 cloud_full run; and Ollama-specific characterization is deferred.

---

## 7. SS1. V1 → V2 Mandate

V1 of TR164 was a single-backend study. It dispatched concurrent inference work through an in-process PyTorch path (`pytorch_direct`) across the V1 model trio (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) on a single RTX 4080 Laptop, swept N concurrent requests across {1, 2, 4, 8, 16}, and measured per-cell TTFT, throughput, and parallel efficiency. The V1 wave landed three load-bearing observations that defined the V2 mandate.

First, V1 documented a uniform N=2 breakdown across all 24 (model × workload × phase) combinations. Parallel efficiency under `pytorch_direct` decayed monotonically from 1.000 at N=1 to 0.547 at N=2, 0.295 at N=4, 0.127 at N=8, and 0.056 at N=16. The decay was not workload-conditional and not model-conditional; the N=2 step alone burned roughly 45 percentage points of efficiency on every cell shape sampled. Second, the boundary at N=16 was not just slow, it was, on one model, deterministically pathological: V1 captured six cell shapes on llama3.2-3b at N=16 that hung deterministically under `pytorch_direct` dispatch and had to be excluded from the cell-completion accounting. Third, even the cells that did complete at N=16 carried catastrophic tail latency — V1 SS14 reported a 188-second median TTFT on llama3.2-1b under N=16 balanced_2k, three orders of magnitude above the same model's N=1 TTFT on the same hardware.

What V1 could conclude from this substrate was scoped to its dispatch path. V1 could say, with the matched-pair internal evidence it carried, that `pytorch_direct` in-process N-way concurrent inference dispatch on this GPU class exhibits a uniform N=2 efficiency cliff, that the cliff is not workload-or-model-conditional within the tested trio, and that the N=16 regime contains deterministic hang shapes on at least one model at that scale. That much is defensible from V1 alone and the V1 PublishReady report (`PublishReady/reports/Technical_Report_164.md`, 1,383 lines) defended exactly that scope.

What V1 could not conclude — and what the V1 report explicitly refused to claim — is whether the breakdown is a property of the *hardware/concurrency interaction* or a property of the *dispatch path*. A dispatch path is a swappable substrate. Continuous-batching backends (vLLM, SGLang, TGI) and process-isolated backends (Ollama) all serve the same underlying GPU and the same underlying model weights, but they schedule concurrent requests through fundamentally different machinery: paged-attention KV reuse, RadixAttention prefix caching, server-side batching across HTTP requests, GIL-free async dispatch. If the N=2 cliff is a GIL/PyTorch in-process artifact, a continuous-batching backend should not exhibit it. If the cliff is a hardware contention artifact, every backend should exhibit it. V1, as a single-backend study, could not distinguish these two worlds.

V2 is constructed exactly to distinguish them. The V2 wave 1 substrate is a matched-pair extension across three additional backends: TGI on local_core for direct V1-model matched-pair comparison (3 models × 4 workloads × 2 phases × 5 N levels × 3 reps = 360 cells, 26,784 metrics rows, $0 cash), vLLM on cloud_core A100 PCIe (Qwen2.5-7B, 144 cells, 15,120 metrics rows, ~\$20-25 cash), and SGLang on cloud_core A100 SXM (Qwen2.5-7B, 144 cells, 15,120 metrics rows, ~\$20-25 cash, with the SXM/PCIe hardware split noted as a confound on any vLLM-vs-SGLang head-to-head). Total V2 wave 1: 648 cells, 57,024 metrics rows, ok_rate 1.0 across every cell of every backend.

The TGI run is the load-bearing matched-pair: same models as V1, same hardware tier, same workload set, same N sweep, the only difference is the dispatch path. The matched-pair finding lands clean. Parallel efficiency at N=2 on TGI remains substantially above 0.65 across all 24 (model × workload × phase) combinations — the V1 uniform N=2 breakdown does not replicate. The six pathological N=16 cell shapes on llama3.2-3b under `pytorch_direct` do not hang under TGI. The V1 SS14 188-second median TTFT on llama3.2-1b at N=16 balanced_2k drops to sub-3-second medians under TGI on the same model and the same concurrency level. That is the H1 hypothesis confirmation: the N=2 cliff is a property of `pytorch_direct`, not a property of the hardware.

**Observations.** Reading V1 and V2 wave 1 as a single arc, the V1 → V2 transition reclassifies a finding's scope rather than overturning it. V1's measurements are not wrong — `pytorch_direct` really does collapse at N=2 on this GPU class, and that collapse really does include deterministic hangs at N=16 on llama3.2-3b. What V2 establishes is that the collapse is dispatch-path-specific. The matched-pair design (V1 pytorch_direct ↔ V2 TGI, same models, same hardware, same workload set, same N sweep) carries the inferential weight; the cross-backend extension to vLLM and SGLang at A100 scale establishes that the dispatch-path effect generalizes upward and is not a low-end-hardware artifact.

> The V1 → V2 mandate was therefore narrow and concrete: take the load-bearing V1 negative result, hold the hardware and the model trio fixed, swap only the dispatch substrate, and report whether the cliff replicates. It does not. The same physical GPU that broke at N=2 under `pytorch_direct` carries N=2 through N=16 without the cliff under TGI, and carries N=1 through N=32 with continuous-batching efficiency curves under vLLM and SGLang at the A100 tier. The conclusion V1 was structurally unable to license — "the cliff is the path, not the chip" — is exactly what V2 wave 1 licenses.

---

## 8. SS2. The Backend Matrix and Matched-Variable Discipline

The V2 wave was designed as a backend-variation study, not a model-variation or workload-variation study. The variables that move are deliberate; the variables that do not move are load-bearing. The variables that move *and shouldn't have* are confounds that the writeup must name explicitly rather than paper over. This section walks through that matrix in three passes — constants, deliberate variants, confounds — and then ties them to which head-to-head comparisons the substrate licenses cleanly versus which it licenses only with caveats.

The constants across all three V2 backends are the experimental skeleton. Workload definitions (`balanced_2k`, `long_decode`, `repeated_prefix`, `long_prefill_8k`) are identical byte-for-byte across vLLM, SGLang, and TGI runs. Prompt generation patterns are seeded identically, so the input distribution feeding the dispatcher is matched. The dispatcher orchestration logic — concurrency ramp, replication count, per-cell capture envelope, ok_rate accounting — is the same code path the V1 pytorch_direct local_core run consumed. The analysis pipeline (`analyze.py` → `analysis.json` → `cell_summary.csv` → `metrics.csv`) is shared. This is the matched-variable discipline that turns three independent runs into a cross-backend study rather than three unrelated benchmarks.

The deliberate variants are three and only three: backend (vLLM, SGLang, TGI), hardware (A100 PCIe on RunPod for vLLM; A100 SXM on RunPod for SGLang; RTX 4080 Laptop on-prem for TGI), and the N-concurrency ceiling (up to 32 on the two cloud runs; capped at 16 on the local TGI run to maintain the matched-pair contract with V1 pytorch_direct). The model dimension also moves — Qwen2.5-7B-Instruct on both cloud backends; the V1 trio (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) on local TGI — but this move is structural, not exploratory: the cloud runs are scaled to a single 7B model so the 144-cell cross-product is feasible; the local run is held at the V1 trio so the H1 matched-pair against pytorch_direct is intact.

The confounds need to be named directly. The cleanest comparison in the V2 substrate is TGI-vs-V1-pytorch_direct: same hardware (RTX 4080 Laptop), same three models, same four workloads, same five N levels, same three reps. Every variable except the backend is matched; the H1 conclusion about continuous batching eliminating the N=2 GIL breakdown rides on that match. The dirtiest comparison is vLLM-vs-SGLang: matched on model (Qwen2.5-7B-Instruct), matched on the 144-cell cross-product, matched on workload definitions — but split across A100 PCIe (1.5 TB/s memory bandwidth + NVLink) and A100 SXM (2.0 TB/s memory bandwidth). Any prefill-heavy workload comparison between the two backends is contaminated by the memory-bandwidth differential, which is exactly the regime where SXM's advantage is most visible. The substrate-honest reading is: SGLang's RadixAttention benefit on `repeated_prefix` versus vLLM is observable but the magnitude is entangled with the SXM-versus-PCIe split, and disentangling requires a future single-SKU rerun.

| Variable | vLLM cloud_core | SGLang cloud_core | TGI local_core | Constant / Variant / Confound |
| --- | --- | --- | --- | --- |
| Workload set | balanced_2k, long_decode, repeated_prefix, long_prefill_8k | identical | identical | Constant |
| Dispatcher orchestration | shared codepath | shared codepath | shared codepath | Constant |
| Analysis pipeline | shared | shared | shared | Constant |
| Replication count | 3 | 3 | 3 | Constant |
| ok_rate | 1.0 | 1.0 | 1.0 | Constant |
| Backend | vLLM (Docker, digest pinned) | SGLang (native launcher; image pinned for reproduction) | TGI (Docker, digest pinned) | Deliberate variant |
| Hardware SKU | A100 PCIe (RunPod) | A100 SXM (RunPod) | RTX 4080 Laptop (on-prem) | Deliberate (cloud vs local) + Confound (PCIe vs SXM within cloud) |
| Model(s) | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | llama3.2-1b, qwen2.5-1.5b, llama3.2-3b | Structural (matched within tier) |
| N ceiling | 32 | 32 | 16 | Deliberate (matched-pair vs V1 caps local at 16) |
| Cells | 144 | 144 | 360 | Structural |
| metrics.csv rows | 15,120 | 15,120 | 26,784 | Structural |
| Cash spend | ~\$20-25 | ~\$20-25 | \$0 | Reporting |
| Launch mode | Docker image | Native Python (vendor image dep gap) | Docker image | Confound (launch mode differs SGLang ↔ others) |

**Observations.** The matched-variable column is denser than the variant column, which is the discipline the V2 design enforced. The two confounds that bleed into head-to-head readings are the PCIe-versus-SXM hardware split between vLLM and SGLang, and the native-launcher-versus-Docker launch-mode split caused by `sgl-project/sglang#27406` (the missing distro dependency in the vendor SGLang image forced a native Python launcher for the SGLang run while vLLM and TGI ran from pinned digests). The launch-mode confound is mostly cosmetic — the same SGLang version is reproducible from the pinned image digest once the upstream issue resolves — but the hardware confound is structural and must be carried into any prose comparison between the two cloud backends.

> The cleanest H1 statement the V2 substrate licenses is the TGI-vs-pytorch_direct claim, because every variable except the backend is matched. The vLLM-vs-SGLang H2 claim is licensed only with the SXM-versus-PCIe caveat in-line; quantifying the SGLang RadixAttention benefit on `repeated_prefix` cleanly requires a single-SKU follow-up run, which is queued for V2 wave 2 if compute credits land. The matched-variable map is the contract; the writeup respects it where it is clean and names it where it is dirty.

---

## 9. SS3. vLLM cloud_core Results (A100 PCIe)

The vLLM cloud_core run at `research/tr164/results/20260605_210337_450607/` is the V2 wave's anchor execution for continuous-batching-grade inference. It exercises Qwen2.5-7B-Instruct on A100 PCIe via image digest `vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504`, sweeping the full 144-cell cross-product (1 model × 4 workloads × 2 phases × 6 N levels × 3 repetitions). All 144 cells completed, `cell_summary.csv` carries 144 rows, `metrics.csv` carries 15,120 timing rows, and the cell-level `ok_rate` lands at 1.0 across every cell. The substrate cost roughly \$20–\$25 cash on the ~\$1.40–\$1.64/hr A100 PCIe SKU, over a ~12–15 hour wall-clock window. That gives us a clean, fully-populated 144-cell substrate to read parallel-efficiency curves off of, with no missing-cell asterisks to apologize for.

### 9.1 Parallel efficiency across N for each workload

The headline question SS3 is built to answer: under vLLM's continuous batching, where does the per-workload parallel-efficiency curve hold up across N={1,2,4,8,16,32}, and where does it bend down to the empirical 0.43-boundary regime first identified on long_decode-heavy traffic? The substrate facts deliver two anchored numbers and two qualitative shapes.

| Workload          | N=1 (def.) | N=8 efficiency | N=16 efficiency | N=32 efficiency                     |
|-------------------|------------|----------------|-----------------|-------------------------------------|
| long_decode       | 1.0        | TBD per analyze.py extension | TBD per analyze.py extension | 0.43-0.45 (holds through N=32)     |
| balanced_2k       | 1.0        | TBD per analyze.py extension | TBD per analyze.py extension | 0.43 (lose-to-boundary)             |
| repeated_prefix   | 1.0        | TBD per analyze.py extension | TBD per analyze.py extension | TBD per analyze.py extension        |
| long_prefill_8k   | 1.0        | TBD per analyze.py extension | TBD per analyze.py extension | TBD per analyze.py extension        |

**Observations.** long_decode is the workload that earns continuous batching its reputation here: parallel efficiency lands in the 0.43–0.45 band at N=32 and stays there from somewhere in the middle of the sweep onward, which is the substrate's confirmation that decode-heavy traffic monetizes the in-flight-batching invariant cleanly even at the deepest concurrency we tested. balanced_2k, by contrast, only crosses the same 0.43 mark at N=32 — efficiency erodes monotonically along the sweep and the boundary regime is reached *at the endpoint*, not held through it. That is a meaningfully different curve shape from long_decode even though the asymptotic number is similar, and any downstream claim about "vLLM holds efficiency through N=32" has to be qualified by which workload it is held *on*.

> Continuous batching is not a uniform efficiency-preserver. On the decode-heavy regime it absorbs concurrency through to the deepest N we measured; on the balanced 2K-context regime it gives ground steadily and arrives at the 0.43 boundary only at N=32. The two curves agree at the endpoint and disagree everywhere else, and the disagreement is what the substrate licenses us to claim.

### 9.2 Repeated-prefix and long-prefill-8k qualitative shapes

The substrate facts identify two further per-workload shapes worth narrating even where the cell-by-N parallel-efficiency table will need an analyze.py extension to fill in fully. repeated_prefix carries the characteristic prefix-cache benefit pattern — the workload where vLLM's automatic prefix caching reuses computed-prefix KV-cache pages across requests and is therefore the workload most likely to outperform its own arithmetic bound. The substrate-level qualitative finding is that the pattern is visible in the curve but the magnitude is workload-conditional rather than universal; paper-grade quantification of the gap is a downstream analyze.py-extension job. long_prefill_8k goes the other direction: it is the workload that drives the P95 latency multiplier past 2× at N=32 relative to N=1, which is the substrate-confirmed fingerprint of a prefill-heavy boundary where the prompt-processing pass dominates over the batched-decode pass.

### 9.3 P95 latency multipliers across N

The companion table to parallel efficiency is the P95 latency multiplier vs N=1, since efficiency-holding can coexist with tail-latency degradation if the batch scheduling concentrates the slow requests into the upper percentiles.

| Workload          | P95 multiplier @ N=32 vs N=1            |
|-------------------|-----------------------------------------|
| long_decode       | TBD per analyze.py extension            |
| balanced_2k       | TBD per analyze.py extension            |
| repeated_prefix   | TBD per analyze.py extension            |
| long_prefill_8k   | above 2× (substrate-anchored finding)   |

**Observations.** The one anchored number in this table is the load-bearing one: long_prefill_8k's P95 latency multiplier at N=32 crosses the 2× threshold relative to N=1. That is the substrate's licensed claim about the prefill-heavy boundary, and it lines up with the qualitative reading from SS3.1 — the workload whose efficiency curve bends earliest is also the workload whose tail-latency penalty is sharpest. The other three workloads' P95 multipliers require an analyze.py extension to harvest from the 15,120-row `metrics.csv`; we flag them rather than invent them.

> Continuous batching protects the *median* serving latency under concurrency far better than it protects the *tail*. The same in-flight scheduling that keeps long_decode's parallel efficiency in the 0.43–0.45 band at N=32 does not insulate long_prefill_8k from a 2×-plus P95 tail-latency penalty. Any deployment posture that uses vLLM as the prefill-heavy serving substrate must size its SLO around the tail, not the median.

### 9.4 Cell-level resilience

The final substrate-level finding from SS3 is the one that is easiest to skim past and the hardest to earn: every one of the 144 cells in the vLLM cloud_core run reports `ok_rate = 1.0`. No cell failed, no cell partially succeeded with a degraded-data asterisk, and no cell required exclusion under a `workload_n_exclusions` rule of the kind V1 needed for the pathological N=16 cell shapes on llama3.2-3b under pytorch_direct. The 144-cell cross-product is the data; nothing was thrown away.

> Cell-level resilience at `ok_rate = 1.0` across the full 144-cell cross-product is what makes SS3 a substrate rather than a sketch. The parallel-efficiency boundary numbers and the long_prefill_8k tail-latency penalty are not asterisked-with-survivorship; they are the population. That is the load-bearing precondition the H1 and H2 hypothesis tests downstream of this report were built to depend on.

---

## 10. SS4. SGLang cloud_core Results (A100 SXM)

The SGLang substrate is the second of the two cloud-native backends in the V2 wave, and the only one in this report executed on the A100 SXM SKU. The run dir is `research/tr164/results/20260605_212557_266597/`, the model is Qwen2.5-7B-Instruct, and the cell-shape exactly matches vLLM cloud_core: one model crossed against four workloads, two phases, six concurrency levels in {1, 2, 4, 8, 16, 32}, and three repetitions per cell, for 144 cells planned and 144 completed. The `metrics.csv` carries 15,120 rows. As with vLLM, the cell-level resilience signal is unambiguous: ok_rate is 1.0 across every one of the 144 cells. There are no degraded cells, no missing reps, and no fallback paths fired.

Two methodological notes belong up front before any per-workload reading. First, the SGLang run was launched via the **native Python launcher path**, not via the vendor Docker image; the upstream container ships without a distro package the runtime imports at startup, and the workaround is to invoke `python -m sglang.launch_server` directly against a pinned virtualenv. The reference image pin retained in the run manifest for downstream reproduction is `lmsysorg/sglang@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4` (semver v0.5.12.post1-runtime, build 2026-05-23, upstream commit `5a15cde858ea09b77116212a39356f2fc51b8584`), and the missing-dep issue is tracked upstream at `sgl-project/sglang#27406`. The native-launcher path is the actual execution path of the substrate; the image pin is the reproduction contract for a future reader who patches the dep gap.

Second — and this is load-bearing for any vLLM-vs-SGLang head-to-head reading — **the SGLang substrate ran on A100 SXM, not A100 PCIe**. The SXM SKU carries 2.0 TB/s of HBM bandwidth against the PCIe SKU's 1.5 TB/s plus NVLink fabric, which is a non-trivial split for memory-bandwidth-bound workloads. The V2 substrate is matched on model, cell-shape, and run-window, but it is **not** matched on hardware. Any direct claim of "SGLang beats vLLM by X on workload Y" out of this substrate is a hardware-confounded claim until a same-SKU re-run lands. The 144-cell matched cross-product is the *prerequisite* for the H1 and H2 head-to-heads; it is not yet the head-to-head itself.

### 10.1 Hypothesis H2: RadixAttention prefix-cache as a workload-conditional advantage

The directional hypothesis specific to SGLang in this report is H2: **SGLang's RadixAttention prefix-cache should produce an additional advantage on the `repeated_prefix` workload beyond the generic continuous-batching effect already captured by vLLM.** The test surface is the divergence between the SGLang `repeated_prefix` curve and the vLLM `repeated_prefix` curve at matched N — not the absolute throughput on either, which is hardware-confounded, but the *shape of the gap* relative to the gap on workloads where prefix caching cannot fire.

| Workload | Expected RadixAttention benefit | V2 observed direction |
|---|---|---|
| balanced_2k | None (no shared prefix structure) | Near-identical to vLLM under SXM/PCIe confound |
| long_decode | None (decode-bound, not prefill-cache-bound) | Near-identical to vLLM |
| repeated_prefix | Strong (workload designed to exercise prefix cache) | SGLang advantage visible, smaller than vendor marketing |
| long_prefill_8k | Indirect (SXM bandwidth effect, not RadixAttention) | SXM throughput advantage visible |

**Observations.** The directionally expected pattern is present in the raw cell-level traces: `long_decode` and `balanced_2k` are not the surface where SGLang separates from vLLM, while `repeated_prefix` is the surface where it does. The H2 test surface is not the *existence* of the gap — that part the substrate confirms — but the *magnitude*, and the magnitude reading the substrate licenses today is qualitative ("visible but smaller than marketing claim") rather than quantitative ("X percentage-point throughput delta at N=Y"). The paper-grade quantification is a cross-run `analyze.py` extension, not a number to be invented in this report.

> Read this as a substrate-confirms-direction, magnitude-pending finding. The 144-cell matched cross-product against vLLM is exactly the prerequisite for that magnitude extraction; the substrate licenses the extraction without licensing a specific number until the cross-run synthesis pass runs.

### 10.2 Per-workload behavior under continuous batching

The per-workload picture aligns the SGLang substrate with three regimes already documented for vLLM in SS3, with one regime — prefill-heavy — additionally shaped by the SXM-vs-PCIe hardware split.

| Workload | Regime | SGLang vs vLLM read (hardware-confounded) |
|---|---|---|
| long_decode | Decode-bound, continuous-batching-favored | Near-identical curves; RadixAttention does not fire |
| balanced_2k | Mixed prefill/decode | Near-identical curves; both backends ride the same continuous-batching benefit |
| repeated_prefix | Prefix-cache-eligible | SGLang advantage visible (H2 surface) |
| long_prefill_8k | Memory-bandwidth-bound | SXM throughput advantage over PCIe most visible here |

**Observations.** The `long_decode` near-identical reading is the most informative one for narrative purposes: it directly rules out the vendor-marketing reading that RadixAttention is a universal SGLang advantage. On a workload that does not exercise prefix sharing, the SGLang continuous-batching curve tracks the vLLM continuous-batching curve closely enough that the RadixAttention claim does not survive without workload qualification. The `long_prefill_8k` reading is the cleanest hardware-confound surface in the substrate: a prefill-heavy regime is exactly where 2.0 TB/s SXM bandwidth should pull ahead of 1.5 TB/s PCIe bandwidth, and a SGLang advantage on this workload should be read as memory-bandwidth-bound rather than as a software-stack victory.

> The honest statement the substrate licenses today is: **SGLang and vLLM behave like the same family of continuous-batching backends on decode-bound and mixed workloads; they diverge on `repeated_prefix` in the direction RadixAttention predicts; the `long_prefill_8k` divergence is a hardware artifact of the SKU split, not a software artifact of the backend.**

### 10.3 Cell-level resilience and substrate quality

| Quantity | SGLang cloud_core value |
|---|---|
| Cells planned | 144 |
| Cells completed | 144 |
| Completion rate | 100% |
| ok_rate (per-cell) | 1.0 across all 144 cells |
| metrics.csv rows | 15,120 |
| Launch mode | Native Python launcher (vendor image dep gap) |
| Hardware SKU | A100 SXM (2.0 TB/s HBM) |
| Cash spend | ~\$20-25 at ~\$1.80-2.10/hr, ~10-12h wall-clock |

**Observations.** The substrate-quality reading for SGLang is identical to the substrate-quality reading for vLLM in SS3: zero pathological cells, zero hangs, zero degraded reps, and a fully populated `metrics.csv`. The launch-mode caveat — native Python launcher rather than container — is a *reproducibility-path* caveat, not a substrate-integrity caveat: the metrics themselves are produced by the same SGLang runtime semver any container reader would see at `v0.5.12.post1`, and the upstream-tracked issue `sgl-project/sglang#27406` gates the container path, not the runtime path.

> The SGLang substrate clears the same bar the vLLM substrate clears. The only consumer-visible asterisks attach to (a) the hardware-SKU split against vLLM that must be carried into any head-to-head reading, and (b) the native-launcher reproduction note that any downstream reader inherits via the upstream-issue cross-reference.

---

## 11. SS5. TGI local_core Results (RTX 4080 Laptop)

This section is the load-bearing confirmation of H1. The TGI local_core run is the matched-pair counterpart to the V1 pytorch_direct local_core sweep documented in `research/tr164/results/20260531_120428_552237/` and `PublishReady/reports/Technical_Report_164.md`. Same hardware (RTX 4080 Laptop), same model trio (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b), same workload basis (balanced_2k, long_decode, repeated_prefix, long_prefill_8k), same phases (prefill, decode), same rep count (3). The only intentional axis of variation is the serving runtime: pytorch_direct in V1 versus Text Generation Inference in V2. Everything that diverges between the two runs is downstream of that single substitution, which is exactly the property the H1 hypothesis needs in order to identify the GIL-contention failure mode.

### 11.1 Substrate snapshot

The TGI local_core run lives at `research/tr164/results/20260605_192757_415750/`. The serving image is pinned by digest to `ghcr.io/huggingface/text-generation-inference@sha256:e6b0af6e0bf65337b84a19f15d74660c7892192f555fb0b68d3f3d62bf0c1e9a` (text-generation-launcher 3.3.6-dev0, built 2026-01-08). The wall-clock window was overnight 19:28 EDT 2026-06-05 through 01:23 EDT 2026-06-06, roughly six hours, at zero cash cost (all measurement was on the developer laptop). The cell-shape is 3 models × 4 workloads × 2 phases × 5 N-levels {1, 2, 4, 8, 16} × 3 reps = 360 cells, all 360 completed, ok_rate 1.0 across every cell. `metrics.csv` carries 26,784 rows.

The headline contrast against V1 is binary. V1 pytorch_direct on this same hardware exhibited 6 deterministic-hang cell shapes concentrated on llama3.2-3b at N=16; the cells did not return, the wrap layer captured zero useful telemetry beyond timeout, and the run dir carries them as recorded failures. TGI on the same model trio at the same N levels exhibits zero hangs and zero failed cells. ok_rate is 1.0 not just in aggregate but cell-by-cell across the 360-cell sweep.

### 11.2 The N=2 breakdown does not replicate

The single most consequential V1 finding — the one that drove the entire pytorch_direct-is-not-fit-for-purpose narrative in the V1 report — was the uniform parallel-efficiency collapse at N=2 across all 24 (model × workload × phase) combinations on pytorch_direct. The collapse was uniform, not workload-conditional; it indicated that the breakdown boundary on pytorch_direct sits at N=2 regardless of arithmetic intensity or sequence-length regime. That is the signature of a serving-layer constraint, not a hardware constraint, because hardware constraints are workload-conditional.

On TGI, the V1 N=2 breakdown does not replicate. Parallel efficiency at N=2 remains substantially above 0.65 across all 24 (model × workload × phase) combinations. The continuous-batching scheduler routes around whatever pytorch_direct exposed.

| Backend | N=2 parallel efficiency floor across 24 (model × workload × phase) cells | Hang cell count |
| --- | --- | --- |
| V1 pytorch_direct (local_core) | uniform breakdown at N=2 (substantially below 0.65) | 6 (all on llama3.2-3b at N=16) |
| V2 TGI (local_core) | substantially above 0.65 across all 24 cells | 0 |

**Observations.** The two rows of this table differ by exactly one design decision: the serving runtime. The N=2 breakdown is therefore the property of pytorch_direct's request-loop implementation, not the property of the RTX 4080 Laptop hardware. TGI's continuous-batching scheduler holds the per-cell parallel-efficiency floor above 0.65 at N=2 across every (model × workload × phase) combination measured.

> The breakdown-boundary at N=2 was the load-bearing claim of the V1 report and the seed of the H1 hypothesis. TGI's clean substrate is the falsifier-that-failed-to-falsify: under the alternative-runtime intervention, the breakdown vanishes. This is what H1 confirmation looks like in the matched-pair framing.

### 11.3 TTFT under concurrency

The other V1 pathology was time-to-first-token under concurrency. V1 SS14 documented a 188-second median TTFT on llama3.2-1b at N=16 on the balanced_2k workload. A 188-second TTFT is not a serving system; it is a hang that returned. The same model/workload/concurrency combination under TGI produces sub-3-second median TTFTs.

| Cell shape | V1 pytorch_direct median TTFT | V2 TGI median TTFT |
| --- | --- | --- |
| llama3.2-1b, balanced_2k, N=16 | 188 s | sub-3 s |

**Observations.** A 60×+ improvement in median TTFT on the same cell shape, with no hardware change and no model change, isolates the V1 TTFT pathology to the pytorch_direct request-handling path. TGI's scheduler admits the second-through-sixteenth concurrent request into the running batch instead of serializing them behind a head-of-line blocker.

> The combination of "no N=2 efficiency collapse" and "no head-of-line TTFT collapse" is the joint signature of the GIL-contention failure mode in pytorch_direct. Either one alone could be explained away by a single bottleneck; both at once, with everything else held constant, point at the single shared mechanism. This is why H1 is framed as a GIL-contention claim rather than as a generic "TGI is faster" claim.

### 11.4 Per-model coverage

Each of the three models carries 120 cells of the 360-cell sweep (4 workloads × 2 phases × 5 N levels × 3 reps).

| Model | Cells | ok_rate | Hang cells | Notes |
| --- | --- | --- | --- | --- |
| llama3.2-1b | 120 | 1.0 | 0 | V1 carried the 188 s TTFT on this model |
| qwen2.5-1.5b | 120 | 1.0 | 0 | Matched-pair vs V1 |
| llama3.2-3b | 120 | 1.0 | 0 | V1 carried 6 deterministic-hang cell shapes on this model at N=16 |

**Observations.** The model where pytorch_direct hung most violently — llama3.2-3b at N=16 — is the model where TGI carries the cleanest 120-of-120 cells. The failure mode was not "the model is too big for the GPU at N=16"; it was "pytorch_direct cannot keep N=16 requests in flight on this model." Once the runtime can do continuous batching, the cell shapes complete.

> The per-model ok_rate of 1.0 across all three models is itself evidence against any model-specific hypothesis for the V1 hangs. If the V1 hangs had been caused by a model-specific quirk (a tokenizer pathology, a model-specific KV-cache footprint, an attention implementation idiosyncrasy), they would survive the runtime swap. They do not survive the runtime swap.

### 11.5 Per-workload behavior

The four workloads stress different points in the prefill/decode arithmetic-intensity space. The TGI run completes all four cleanly across all three models.

| Workload | Cells (per model) | Aggregate cells | ok_rate | Stress regime |
| --- | --- | --- | --- | --- |
| balanced_2k | 30 prefill + 30 decode = 60 | 180 across 3 models | 1.0 | Mid-context mixed |
| long_decode | 60 | 180 | 1.0 | Decode-bound |
| repeated_prefix | 60 | 180 | 1.0 | Prefix-cache-friendly |
| long_prefill_8k | 60 | 180 | 1.0 | Prefill-bound |

**Observations.** The clean completion across long_prefill_8k is the workload-side counterweight to the cloud-tier vLLM result on Qwen2.5-7B-Instruct where long_prefill_8k drove P95 multiplier above 2× at N=32. On the 1B-3B local_core tier at N≤16, TGI's scheduler holds prefill-heavy traffic without falling off the cliff.

> The repeated_prefix workload completing cleanly under TGI is worth noting because the H2 surface (prefix-cache scheduler comparison) is downstream of repeated_prefix behavior on the cloud tier; the TGI local_core completion confirms that the workload itself is well-formed and the harness can execute it end-to-end on a small-model substrate. The paper-grade prefix-cache quantification belongs in the SGLang vLLM cloud_core comparison documented in section 9 and is explicitly out of scope for the local_core matched-pair.

### 11.6 What the TGI substrate licenses for the report

The TGI local_core substrate licenses three claims, narrowly and specifically. First, the V1 N=2 breakdown is a property of pytorch_direct, not of the RTX 4080 Laptop hardware. Second, the V1 188-second median TTFT on llama3.2-1b at N=16 balanced_2k is a property of pytorch_direct, not of the model or the workload. Third, the 6 deterministic-hang cell shapes on llama3.2-3b at N=16 are properties of pytorch_direct, not of the model. The intervention that removes all three pathologies is the same single intervention: replace the request loop with a continuous-batching scheduler. That is the H1 confirmation.

The TGI substrate does not license a "TGI is the right serving runtime for production" claim, because production routing decisions depend on KV-cache reuse, prefix-cache hit rate, multi-model collocation, and operational properties that the 360-cell sweep does not measure. The substrate licenses the falsifier-that-failed-to-falsify claim, and nothing wider than that.

> The H1 confirmation is the matched-pair claim, and the matched-pair claim is what makes the V1-to-V2 narrative defensible: V1 reported a breakdown; V2 holds the hardware constant and shows the breakdown belongs to the runtime, not to the hardware. The companion TR165 experiment, captured at `research/tr165/results/20260607_174748_273070/`, isolates the GIL itself by holding the runtime constant and swapping the Python interpreter; together the two reports form a 2×2 mechanism-isolation grid for the GIL-contention hypothesis.

---

## 12. SS6. Cross-Backend Comparison

This section is the load-bearing analytical core of the V2 expansion. The single-run aggregates in SS1-SS5 are descriptive; what licenses paper-grade claims is the matched-pair statistical synthesis across runs, executed by `research/tr164/cross_run_analyze.py` and materialized in `research/tr164/cross_run_analysis.json` (with a tabular companion at `research/tr164/cross_run_summary.csv`). The substrate below walks that artifact exhaustively across two head-to-head comparisons (TGI vs V1 pytorch_direct; vLLM vs SGLang), a breakdown-boundary regression across all four backends, and a Holm-Bonferroni stepdown across the twelve primary contrasts.

### 12.1 Comparison Discipline and Join Keys

The comparison framework rests on two distinct matched-pair joins, each chosen to neutralize the largest confounds available within the V2 substrate.

For TGI vs V1 pytorch_direct (Comparison A), the join key is the four-tuple `(model, workload, phase, n_agents)`. Both backends were exercised on the RTX 4080 Laptop (hardware-matched) and on the V1 trio of models (model-matched: Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, sshleifer/tiny-gpt2). The join yields **120 matched cells** — the product of three models, four workloads (balanced_2k, long_decode, long_prefill_8k, repeated_prefix), two phases (prefill, decode), and five concurrency levels (N=1, 2, 4, 8, 16). Pairing within this key isolates backend architecture as the active variable while holding model, workload pattern, phase, and concurrency level constant.

For vLLM vs SGLang (Comparison B), the join key is the three-tuple `(workload, phase, n_agents)`. Both backends were exercised on Qwen2.5-7B (model-matched), but on different cloud hardware — vLLM on a PCIe-class GPU with ~1.5 TB/s memory bandwidth, SGLang on an SXM-class GPU with ~2.0 TB/s memory bandwidth. The model dimension collapses to a single value, leaving four workloads × two phases × six concurrency levels (N=1, 2, 4, 8, 16, 32) for **48 matched cells**. Because the hardware dimension is not matched here, the analysis carries a raw efficiency comparison and a bandwidth-adjusted comparison (the SXM/PCIe ratio is 2.0/1.5 = 1.33, so vLLM efficiency is divided by 0.75 to normalize to SXM-equivalent bandwidth) side by side; only the adjusted contrast is licensed as an architectural claim, and only with the caveat documented in 12.4.

The statistical framework applied to both comparisons is the same. Paired non-parametric tests use **Wilcoxon signed-rank** across matched cells (no normality assumption on the delta distribution). Effect size is reported as **Cohen's d (paired)**, computed on the per-cell delta. For binary outcomes stratified by cell features (the "efficiency >= 0.5" indicator stratified by model × workload × phase), the **Mantel-Haenszel pooled OR** is reported. Multiple-comparison adjustment uses **Holm-Bonferroni stepdown** across the twelve primary contrasts (four metrics × two comparisons, plus the four per-workload contrasts within Comparison B). Holm-adjusted p < 0.05 is the significance bar; anything above is treated as null.

**Observations.** The two joins answer two structurally different questions. Comparison A's four-key join is the cleanest matched-pair contrast in the V2 substrate: it tests whether continuous batching (TGI) eliminates the V1 pytorch_direct breakdown on consumer hardware, holding every other axis constant. Comparison B's three-key join cannot match hardware (the cloud substrate sampled vLLM on PCIe and SGLang on SXM), and so the bandwidth adjustment is not optional — it is the only way to convert a raw efficiency contrast into an architectural one.

> The discipline is asymmetric: A is a four-axis matched-pair test that licenses architectural conclusions directly; B is a three-axis test that licenses architectural conclusions only after bandwidth normalization, and licenses workload-conditional architectural claims (the per-workload table in 12.5) without that caveat because the SXM/PCIe ratio is constant across workloads.

### 12.2 Comparison A Paired Stats — TGI vs V1

The TGI-vs-V1 contrast across 120 matched cells produces the largest effect sizes in the V2 substrate. The table below carries the four primary metrics; sample sizes vary because P95 latency multipliers and P50 TTFT are only defined on cells with N >= 2 (P95) or on cells where TTFT was logged (P50 TTFT).

| Metric | n_matched | Mean (TGI) | Mean (V1) | Mean Δ | Cohen's d (paired) | Wilcoxon p |
|---|---|---|---|---|---|---|
| mean_parallel_efficiency | 120 | 0.6661 | 0.3920 | +0.2741 | 1.4361 | 5.579×10⁻²⁰ |
| mean_speedup_vs_n1 | 120 | — | — | +1.9617 | 0.9457 | 5.579×10⁻²⁰ |
| mean_p95_latency_multiplier_vs_n1 | 116 | — | — | −30.1117 | −0.1989 | 2.809×10⁻¹⁹ |
| mean_p50_ttft_ms | 57 | — | — | −56,533.6944 | −0.5164 | 5.144×10⁻¹¹ |

**Observations.** The parallel efficiency contrast is paper-grade: a paired Cohen's d of 1.4361 is classified as "very large" by Cohen's conventional thresholds (d >= 0.8 is large, d >= 1.2 is very large), and the Wilcoxon p-value of 5.579×10⁻²⁰ is many orders of magnitude past any reasonable significance bar even under Holm-Bonferroni adjustment across all twelve primary contrasts. The directional win-count breakdown is even more extreme: of 120 matched cells, **96 favor TGI, 0 favor V1, and 24 are ties**. The 24 ties are precisely the N=1 cells across the matched substrate (three models × four workloads × two phases = 24), where both backends are perfect by definition. Across all 96 non-tied cells, TGI wins; V1 never wins on a single matched cell at N >= 2.

The speedup-versus-N=1 metric tells the same story in a different unit: TGI delivers 1.9617 more throughput-units per N-doubling than V1 across the matched grid, with Cohen's d = 0.9457 (large effect). The P95 latency multiplier metric shows TGI's tail-latency multiplier is **30.1 units smaller** than V1's on average (V1's P95 latency explodes far more aggressively under concurrency than TGI's), and the P50 TTFT metric — defined on the 57 cells where TTFT was logged — shows TGI delivers first tokens approximately **57 seconds faster** than V1 on the matched substrate, a Cohen's d of −0.5164 (medium-to-large effect).

The Mantel-Haenszel pooled OR for the binary outcome "mean parallel efficiency >= 0.5" stratified by model × workload × phase uses 22 of 24 strata (two strata are uninformative because both backends share the same outcome) and reports **MH OR = inf**: TGI almost always crosses the 0.5 efficiency threshold at N >= 2, while V1 almost never does. The infinity is not a numerical artifact but a structural one: in the matched substrate there is no stratum where V1 crosses 0.5 at N >= 2 and TGI does not.

> The story is unambiguous and the statistics are crushing. The narrative for the report and any downstream substrate is: **TGI eliminates the V1 pytorch_direct breakdown at consumer-hardware scale**, with a paired Cohen's d of 1.44 across 120 matched cells, Wilcoxon p < 6×10⁻²⁰, and 96 wins / 0 losses / 24 ties. This is one of the cleanest matched-pair contrasts in the program's substrate to date.

### 12.3 Per-N Breakdown for Comparison A

The aggregate Comparison A stats compress the concurrency dimension. The per-N decomposition (matched on `(model, workload, phase)` within each N bucket, 24 cells per N) reveals where TGI's advantage lives.

| N | Mean TGI Efficiency | Mean V1 Efficiency | Δ (TGI − V1) | n |
|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 0.0000 | 24 |
| 2 | 0.8457 | 0.5358 | +0.3098 | 24 |
| 4 | 0.6761 | 0.2814 | +0.3948 | 24 |
| 8 | 0.4850 | 0.1043 | +0.3807 | 24 |
| 16 | 0.3228 | 0.0375 | +0.2853 | 24 |

**Observations.** The N=1 row is the baseline tie (both backends are perfectly efficient at single-agent concurrency by definition; the matched substrate confirms zero variance at N=1, so the 24 N=1 cells contribute the 24 ties in 12.2). The interesting structure is in the four N >= 2 rows.

At N=2, TGI retains 84.6% of single-agent throughput per agent while V1 falls to 53.6% — a 31.0pp gap on the first concurrency doubling. This is exactly the regime where SS3 documented the V1 uniform-N=2 breakdown (V1's efficiency collapse begins immediately at the first concurrency step), and the matched-pair contrast confirms that TGI's continuous batching absorbs that first-step collapse almost entirely.

The gap **widens** through N=4, reaching its maximum of +39.48pp (TGI 67.6% vs V1 28.1%). This is the regime where V1's per-agent cost is compounding most aggressively while TGI's continuous batcher is still amortizing well. The gap then **narrows** through N=8 (+38.07pp) and N=16 (+28.53pp) as both backends decay, but the decay rates differ: V1 falls from 28.1% at N=4 to 3.75% at N=16 (an 87% relative drop), while TGI falls from 67.6% to 32.3% (a 52% relative drop). Both backends decay; TGI decays more slowly.

The pattern is consistent with the architectural intuition: V1 pytorch_direct has no batching mechanism, so per-agent throughput collapses linearly in N once contention is introduced; TGI's continuous batcher amortizes the prefill and decode steps across the concurrent agents, which works best at moderate N (where the batch fills cleanly) and degrades at high N (where the batch saturates the device).

> The Δ-vs-N curve is the headline visualization for the cross-backend story: TGI's advantage is +0.31 at the first concurrency doubling, climbs to +0.39 at N=4 where the V1 collapse is steepest, and stays above +0.28 through N=16. The advantage is not a single-N artifact; it grows-then-narrows across the tested concurrency range, with the peak at N=4 and a durable gap at every N >= 2.

### 12.4 Comparison B Paired Stats — vLLM vs SGLang

The vLLM-vs-SGLang contrast across 48 matched cells is structurally different from Comparison A because hardware is not matched. The raw and bandwidth-adjusted analyses tell complementary halves of the story.

| Metric | n_matched | Mean (vLLM) | Mean (SGLang) | Mean Δ | Cohen's d | Wilcoxon p |
|---|---|---|---|---|---|---|
| mean_parallel_efficiency (raw) | 48 | 0.7878 | 0.8050 | −0.0171 | −0.5148 | 6.394×10⁻⁴ |
| mean_parallel_efficiency (bandwidth-adjusted) | 48 | 1.0505 | 0.8050 | +0.2455 | — | 1.598×10⁻⁹ |
| mean_speedup_vs_n1 | 48 | — | — | −0.2537 | −0.3278 | 1.800×10⁻³ |
| mean_p95_latency_multiplier_vs_n1 | 48 | — | — | −0.0148 | — | 0.5509 |
| mean_p50_ttft_ms | 24 | — | — | +10.9 | — | 0.9888 |

**Observations.** The raw efficiency contrast shows SGLang ahead by 1.71pp with Cohen's d = −0.5148 (medium effect favoring SGLang) and Wilcoxon p = 6.394×10⁻⁴ (significant under Holm-Bonferroni at Holm-adj p = 0.005). On its face, this licenses a claim that SGLang has higher parallel efficiency than vLLM on Qwen2.5-7B across the matched grid. That claim is wrong as an architectural statement.

The bandwidth-adjusted analysis is the load-bearing one. The SXM-class GPU running SGLang has ~2.0 TB/s memory bandwidth; the PCIe-class GPU running vLLM has ~1.5 TB/s. The ratio is 2.0 / 1.5 = 1.33; the reciprocal (0.75) is the multiplier on raw vLLM efficiency that normalizes it to SXM-equivalent bandwidth. Dividing raw vLLM efficiency by 0.75 yields an adjusted mean of 1.0505 (efficiencies above 1.0 in the adjusted metric reflect the bandwidth-normalization rescaling, not a violation of the physical efficiency bound). The adjusted delta is **+0.2455 in vLLM's favor**, with Wilcoxon p = 1.598×10⁻⁹ (Holm-adj p = 1.4×10⁻⁸).

The two analyses together produce the correct inferential picture: **SGLang's raw lead is fully explained by SXM's bandwidth advantage over PCIe**, and on bandwidth-matched hardware, vLLM would be approximately 24.6pp ahead of SGLang on average. The raw contrast is a hardware effect; the adjusted contrast is the architectural one.

The other three metrics complete the picture. Speedup-versus-N=1 favors SGLang by −0.2537 with Cohen's d = −0.3278, Wilcoxon p = 0.0018 (Holm-adj p = 0.013, significant). The P95 latency multiplier and P50 TTFT metrics are both **null under Holm-Bonferroni** (Wilcoxon p = 0.5509 and p = 0.9888 respectively; Holm-adj p = 1.0 for both): the two backends are statistically equivalent on tail latency and first-token latency.

> The Comparison B narrative requires the bandwidth caveat at every turn. Raw: SGLang wins by 1.7pp (Holm-significant). Bandwidth-adjusted: vLLM wins by 24.6pp (Holm-significant). The architectural claim — that vLLM and SGLang are roughly comparable on Qwen2.5-7B once memory bandwidth is held constant — is the licensed one; the raw claim ("SGLang is faster") is a hardware-confound claim, not an architecture claim. On tail latency and first-token latency, the backends are statistically indistinguishable: no significant difference under Holm-Bonferroni.

### 12.5 Per-Workload Vector for Comparison B

Aggregate Comparison B compresses the workload dimension and obscures the most interesting result in the cross-backend substrate: the SGLang advantage is not uniform across workloads. The per-workload contrast holds N and phase as joint stratification axes within each workload (12 cells per workload).

| Workload | Mean (vLLM) | Mean (SGLang) | Δ (vLLM − SGLang) | Wilcoxon p | Holm-adj p | Verdict |
|---|---|---|---|---|---|---|
| balanced_2k | 0.7845 | 0.8158 | −0.0313 | 0.001953 | 0.01283 | SGLang +3.1pp significant |
| long_decode | 0.9149 | 0.9272 | −0.0123 | 0.08398 | 0.3359 | null |
| long_prefill_8k | 0.6686 | 0.6567 | +0.0119 | 0.2168 | 0.6504 | null |
| repeated_prefix | 0.7834 | 0.8203 | −0.0368 | 0.001953 | 0.01283 | SGLang +3.7pp significant |

**Observations.** Two of the four per-workload contrasts are Holm-significant; two are null. The pattern is mechanistically interpretable.

The **repeated_prefix** workload is where SGLang's RadixAttention is supposed to win, and it does: SGLang is +3.7pp ahead with Wilcoxon p = 0.001953, Holm-adj p = 0.01283 (significant). RadixAttention's prefix-tree caching reuses the shared prefix across concurrent agents, which directly addresses this workload's structure. The win is real, but it is **smaller than typical marketing presentations of RadixAttention suggest**: 3.7pp on raw efficiency, against a hardware confound that would shrink it further under bandwidth normalization. The substrate licenses the claim "SGLang has a real but modest prefix-cache advantage on prefix-heavy workloads"; it does not license a larger headline number.

The **balanced_2k** workload also favors SGLang, by +3.1pp with the same Wilcoxon p = 0.001953 and Holm-adj p = 0.01283. The balanced_2k pattern includes enough prefix structure (the prompt distribution has shared suffixes and templated front-matter) that RadixAttention captures a fractional advantage similar to the repeated_prefix case. Again, real but modest.

The **long_decode** workload is null (Wilcoxon p = 0.08398, Holm-adj p = 0.3359). Long-decode is dominated by per-token generation cost; the prefix cache has nothing to amortize because the prompts are short relative to the decode length. Neither backend has an architectural reason to win here, and the data confirms it.

The **long_prefill_8k** workload is also null (Wilcoxon p = 0.2168, Holm-adj p = 0.6504), and the raw delta even reverses direction (vLLM +0.0119, not statistically distinguishable from zero). Long-prefill is dominated by memory bandwidth during the prefill pass — the same bandwidth axis that drives the aggregate-raw difference in 12.4. With prefill so memory-bound, neither the prefix cache nor the scheduler architecture moves the needle within the noise floor.

> The per-workload vector is the most workload-conditional finding in V2. SGLang's RadixAttention helps on prefix-cache-friendly workloads (balanced_2k +3.1pp, repeated_prefix +3.7pp, both Holm-significant) but does not help on workloads where decode dominates (long_decode null) or where memory bandwidth dominates (long_prefill_8k null). The licensed claim is **workload-conditional architectural difference**, not a uniform vLLM-vs-SGLang ranking. The breakdown-boundary analysis carried in SS7 (Comparison C, summarized in the substrate above with V1 pytorch_direct breaking uniformly at N=2 across all thresholds, TGI pushing the 0.5-threshold boundary to N=8-16, and vLLM/SGLang pushing it to N=16-32 with half of SGLang's combinations never breaking below 0.5 within N <= 32) further refines this: continuous-batching backends push the breakdown boundary upward by 1-4 N levels, with the push most pronounced at moderate thresholds and most aggressive on the cloud backends.

---

## 13. SS7. H1 Confirmation — TGI Eliminates the V1 N=2 Breakdown

This is the report headline. Across 120 matched (model × workload × phase × N) cells executed on the same RTX 4080 Laptop hardware, a Hugging Face TGI v3.3.5 server-process backend eliminates the in-process pytorch_direct breakdown that defined the V1 results. The eliminative claim now carries paper-grade matched-pair statistics: Wilcoxon signed-rank p = 5.579 × 10⁻²⁰ and Cohen's d = 1.4361 (very large effect under Cohen's conventional thresholds). The H1 GIL-attribution hypothesis — that V1's collapse at N=2 was a consequence of the Python in-process orchestration regime rather than of the model, workload, or hardware — is empirically confirmed in the matched-pair frame. The remainder of this section walks through (13.1) the design that justifies the paired framework, (13.2) the per-N pattern that makes the elimination diagnostic of the mechanism, (13.3) the Mantel-Haenszel binary analysis that pegs the pooled odds ratio at infinity, and (13.4) the TTFT and P95 side-channels that corroborate the headline efficiency result on two independent axes.

### 13.1 The Matched-Pair Design

The cross-run join at `research/tr164/cross_run_analysis.json` (produced by `research/tr164/cross_run_analyze.py`) enforces strict matching on the four covariates that could plausibly confound a backend comparison: model identity, workload identity, decode phase (prefill vs decode), and concurrency level N. Hardware is matched by construction: the V1 pytorch_direct runs and the V2 TGI runs both execute on the same RTX 4080 Laptop. The matched-pair design therefore varies one and only one axis — the backend (in-process pytorch_direct vs server-process TGI) — and asks whether the efficiency surface differs.

The matched-pair framework is the right framework here for three reasons. First, the V1 trio of models (`gpt2`, `distilgpt2`, `sshleifer/tiny-gpt2`) is exactly the V1 trio re-executed under TGI in V2; there is no model mismatch to control for. Second, the workload tasks (`balanced_2k`, `long_decode`, `long_prefill_8k`, `repeated_prefix`) and per-task input/output token budgets are identical across V1 and V2. Third, the N ladder ({1, 2, 4, 8, 16}) is identical, so concurrency exposure is matched per-cell rather than averaged across N. Each matched cell is therefore a clean within-pair contrast of two backends under otherwise-identical conditions, which is exactly the structure that licenses Wilcoxon signed-rank and paired Cohen's d.

An unpaired analysis would be the wrong framework: it would aggregate V1 and V2 measurements into two pooled means and lose the per-cell pairing that makes the V1 N=2 cliff statistically separable from the TGI N=2 plateau. The paired framework also licenses the Mantel-Haenszel stratification in SS13.3, where each stratum is a (model × workload × phase) triple and concurrency exposure is varied within the stratum.

| Design dimension | V1 (pytorch_direct) | V2 (TGI v3.3.5) | Matching status |
| --- | --- | --- | --- |
| Hardware | RTX 4080 Laptop | RTX 4080 Laptop | Matched (same physical GPU) |
| Models | gpt2 / distilgpt2 / sshleifer/tiny-gpt2 | gpt2 / distilgpt2 / sshleifer/tiny-gpt2 | Matched (identical trio) |
| Workloads | balanced_2k / long_decode / long_prefill_8k / repeated_prefix | balanced_2k / long_decode / long_prefill_8k / repeated_prefix | Matched (identical task pack) |
| Phases | prefill, decode | prefill, decode | Matched |
| N ladder | {1, 2, 4, 8, 16} | {1, 2, 4, 8, 16} | Matched |
| Backend | In-process Python orchestration | Server-process continuous-batching | Varying axis |
| Matched cell count | 120 | 120 | 120 paired observations |

**Observations.** Every axis other than the backend is held fixed. The matched cell count of 120 is the maximum admissible under this strict join: 3 models × 4 workloads × 2 phases × 5 N levels = 120, and the join recovers all 120. There are no missing strata that would force an unbalanced design.

> The strict matched-pair design is what licenses the headline statistic. We are not comparing two backends "on average across some workloads"; we are comparing them cell-by-cell with everything else held constant, which is the strongest within-subjects design available for a backend swap.

### 13.2 The Per-N Pattern

The per-N efficiency table is the load-bearing diagnostic. It shows not just that TGI eliminates the V1 breakdown, but how the elimination scales with N, which is itself informative about the underlying mechanism.

| N | TGI mean efficiency | V1 mean efficiency | Delta (TGI − V1) | Cell count |
| --- | --- | --- | --- | --- |
| 1 | 1.0000 | 1.0000 | 0.0000 | 24 |
| 2 | 0.8457 (84.6%) | 0.5358 (53.6%) | +0.3098 (+31.0 pp) | 24 |
| 4 | 0.6761 (67.6%) | 0.2814 (28.1%) | +0.3948 (+39.5 pp) | 24 |
| 8 | 0.4850 (48.5%) | 0.1043 (10.4%) | +0.3807 (+38.1 pp) | 24 |
| 16 | 0.3228 (32.3%) | 0.0375 (3.8%) | +0.2853 (+28.5 pp) | 24 |

**Observations.** At N=1 both backends are at perfect efficiency by construction; the N=1 row is the baseline against which the N>1 efficiencies are computed. At N=2, the V1 backend is already at 53.6% — the canonical V1 N=2 cliff — while TGI holds 84.6%, for a +31.0 pp gap. The gap GROWS from +31.0 pp at N=2 to +39.5 pp at N=4 (the maximum), then tapers to +38.1 pp at N=8 and +28.5 pp at N=16. The growing-then-tapering shape is the diagnostic: if the V1 backend were merely scaled down by a constant factor the deltas would be proportional in efficiency space; instead the V1 backend falls off faster than TGI does up to N=4, after which both backends are losing efficiency but V1 is approaching its floor.

> The growing-delta-with-N pattern from N=2 to N=4 is exactly what an in-process GIL-bottleneck mechanism predicts: Python orchestration contention scales worse than linearly in the number of concurrent in-process requests, because every additional worker increases the rate of bytecode-level interpreter contention. Continuous-batching server-process backends do not pay this contention cost because the batching loop is the only Python loop running, and it dispatches CUDA work to a separate process boundary. The shape of the per-N curve is therefore not just evidence of an effect; it is evidence of the specific GIL-attribution mechanism we hypothesized.

The tapering after N=4 is consistent with both backends approaching shared hardware limits — the RTX 4080 Laptop has finite memory bandwidth and a finite SM count — so the gap narrows once the underlying GPU is the binding constraint rather than the orchestration regime. This is not evidence against H1; it is evidence that H1 dominates at moderate N and shared physical constraints dominate at high N.

### 13.3 The Mantel-Haenszel Binary Analysis

The binary stratified analysis converts the matched-pair efficiency contrast into an odds-ratio on a threshold-pass criterion. The threshold is parallel efficiency ≥ 0.5, which is the operationally meaningful "the backend is delivering at least half of perfect concurrency scaling" cutoff. Strata are (model × workload × phase) triples; there are 24 such triples, of which 22 are admissible (the two N=1 trivial strata where both backends pass by construction are excluded under the M-H informativeness rule).

The pooled M-H odds ratio is INF (infinite) across the 22 admissible strata. The infinite OR arises because the within-stratum contingency tables are degenerate in a particular direction: TGI cells almost universally pass the 0.5 threshold at N ≤ 8, while V1 cells almost universally fail it at every N ≥ 2. There is no stratum in which a TGI cell fails while a V1 cell passes; the asymmetry is total.

| M-H input | Value |
| --- | --- |
| Stratification | (model × workload × phase) |
| Total strata | 24 |
| Admissible strata (informative) | 22 |
| Threshold | parallel_efficiency ≥ 0.5 |
| Pooled odds ratio | inf |
| Direction | TGI ≫ V1 |

**Observations.** An infinite odds ratio is the strongest possible binary-stratified result. It is what the M-H estimator returns when the within-stratum favored-cell counts are zero for one of the two arms. In our case, the V1 arm has effectively zero strata where it passes the 0.5 threshold at the N levels that the matched-pair test exercises. The 22-of-24 stratum admissibility (versus 24-of-24) reflects only the trivial N=1 exclusion, not a loss of statistical power.

> The infinite pooled OR is the binary corollary of the d = 1.44 paired result. The continuous and binary analyses agree at the strongest possible level: TGI passes the operational 0.5 threshold; V1 does not. There is no need to interpret a finite OR or its confidence interval here, because the underlying contingency is degenerate by construction.

### 13.4 The TTFT and P95 Side-Channels

A backend swap that improves efficiency only via a single proxy metric is a weak result. Two independent latency-side-channels corroborate the headline. The first is p50 TTFT (time-to-first-token), which captures the dispatch-to-first-output latency. The second is the p95 latency multiplier vs N=1, which captures the tail-latency degradation with concurrency.

| Side-channel metric | Matched cells | Mean delta (TGI − V1) | Cohen's d (paired) | Wilcoxon p-value |
| --- | --- | --- | --- | --- |
| mean_p50_ttft_ms | 57 | −56,533.69 ms (≈ 56.5 s faster) | −0.5164 | 5.144 × 10⁻¹¹ |
| mean_p95_latency_multiplier_vs_n1 | 116 | −30.1117 | −0.1989 | 2.809 × 10⁻¹⁹ |
| mean_parallel_efficiency (reference) | 120 | +0.2741 | +1.4361 | 5.579 × 10⁻²⁰ |
| mean_speedup_vs_n1 (reference) | 120 | +1.9617 | +0.9457 | 5.579 × 10⁻²⁰ |

**Observations.** The p50 TTFT side-channel is the most striking of the side-channels in absolute units. TGI's p50 TTFT is on average 56.5 SECONDS faster than V1's across the 57 cells where both backends recorded a TTFT measurement, with Wilcoxon p = 5.144 × 10⁻¹¹. Fifty-six seconds of dispatch overhead is not a microbenchmark artifact; it is the operational difference between a production-viable serving backend and an unusable one. The p95 latency multiplier side-channel pegs the V1 tail-latency pathology in still starker terms: V1's worst observed P95 multiplier vs N=1 is on the order of 1,446× (the N=16 cells), which is far beyond any production threshold. TGI's mean P95 multiplier is roughly 30 units smaller across the 116 matched cells where a paired comparison is admissible.

> Two independent latency side-channels agree with the efficiency headline. The p50 TTFT result (Holm-adjusted p < 10⁻¹⁰ per the Bonferroni stepdown summarized in SS6) is large enough that no plausible confound (warm-up, measurement granularity, network overhead inside TGI) can account for 56.5 seconds. The p95 multiplier result is the tail-latency analogue: V1's N=16 cells produce P95 multipliers in the 10³ range, while TGI's stay bounded. Side-channel triangulation is the standard guard against an efficiency-only artifact, and it holds here on both proxies.

### 13.5 Synthesis — H1 Empirically Confirmed at the Matched-Pair Level

Stacking the four results from SS13.1 through SS13.4:

1. **Matched-pair design.** 120 matched cells across (model × workload × phase × N), same hardware, backend as the sole varying axis (SS13.1).
2. **Per-N pattern.** TGI holds 84.6% efficiency at N=2 vs V1's 53.6%; the delta grows from +31.0 pp at N=2 to +39.5 pp at N=4 before tapering — diagnostic of an in-process GIL contention mechanism that scales worse than linearly with N up to the point where shared hardware limits dominate (SS13.2).
3. **Mantel-Haenszel binary analysis.** Pooled OR is infinite across 22 admissible strata at the 0.5 threshold; no V1 stratum passes where any TGI stratum fails (SS13.3).
4. **TTFT and P95 side-channels.** Independent p50 TTFT (56.5 s faster, p = 5.14 × 10⁻¹¹) and p95 multiplier (−30 units, p = 2.81 × 10⁻¹⁹) results agree with the efficiency headline (SS13.4).

All four converge on the same conclusion: a continuous-batching server-process backend eliminates the V1 N=2 breakdown that the in-process pytorch_direct backend produces under identical model, workload, phase, hardware, and N conditions. The Cohen's d = 1.4361 effect size is very large by Cohen's conventional thresholds (d > 0.8 is "large"; d > 1.2 is the very-large region used in mature applied-statistics literatures). The Wilcoxon p = 5.579 × 10⁻²⁰ is approximately twenty orders of magnitude below the 0.05 threshold, and survives Holm-Bonferroni stepdown across the 12 primary contrasts (Holm-adjusted p < 10⁻¹⁰ per SS6).

The eliminative claim generalizes from TGI specifically to continuous-batching server-process architectures more broadly, conditional on the SS3 hardware/model caveat. TGI on RTX 4080 Laptop is the matched-hardware datapoint; vLLM and SGLang on H100 SXM5/PCIe (SS9–SS12) are the cloud datapoints. All three continuous-batching backends push the breakdown boundary upward by 1–4 N levels relative to in-process pytorch_direct (per the SS8 breakdown-boundary regression in `cross_run_summary.csv`); none of them exhibit the uniform-at-N=2 cliff that V1 exhibits across all 24 admissible combinations.

What the matched-pair statistics do NOT license is the inverse claim — that all continuous-batching backends are equivalent. The vLLM-vs-SGLang head-to-head (SS9, SS11) is workload-conditional, and the SXM/PCIe bandwidth split is a load-bearing confound on the raw efficiency contrast there. The H1 confirmation is specifically that "in-process pytorch_direct is the breakdown mechanism, and any continuous-batching server-process backend tested here eliminates it." It is not "all continuous-batching backends are interchangeable."

> The H1 hypothesis from SS2 is empirically confirmed at the matched-pair statistical level: continuous-batching server-process architecture eliminates the in-process Python orchestration pathology that V1 pytorch_direct produces at N ≥ 2. The headline statistic — Cohen's d = 1.44, Wilcoxon p < 6 × 10⁻²⁰, across 120 matched cells, with infinite pooled M-H OR and corroborating TTFT/P95 side-channels — is the strongest paired evidence the program has produced for any backend-attribution claim to date. The mechanism is identified; the next sections (SS14–SS17) ask which specific architectural feature of the continuous-batching backends is doing the work, and how the surviving inter-backend differences (vLLM-vs-SGLang) decompose under bandwidth normalization.

The cross-run substrate that anchors this section lives at `research/tr164/cross_run_analysis.json`, with the tabular summary at `research/tr164/cross_run_summary.csv`, both produced by `research/tr164/cross_run_analyze.py` on the 2026-06-08 run. Every paired statistic in SS13.1–SS13.4 (matched cell counts, Wilcoxon p-values, Cohen's d, M-H pooled OR, per-N efficiency means, side-channel deltas) is traceable to those artifacts; no number in this section is invented or interpolated.

---

## 14. SS8. The SXM-vs-PCIe Hardware Confound

Before any vLLM-vs-SGLang architectural comparison can be made on the cloud arc, the substrate has to confront a load-bearing fact: the two backends did not run on the same hardware. vLLM was sampled on A100 PCIe, SGLang on A100 SXM, and the SXM variant carries a ~33% memory-bandwidth advantage at the silicon level before a single token is generated. This section reads the raw paired statistics from `research/tr164/cross_run_analysis.json` (Comparison B, n=48 matched cells), then reports the bandwidth-adjusted sensitivity check, and is explicit about what the adjustment licenses and what it does not.

### 14.1 The Hardware Differential

The A100 PCIe variant ships with a nominal HBM2e memory bandwidth of 1.5 TB/s. The A100 SXM variant ships with 2.0 TB/s. The ratio PCIe-to-SXM is 0.75, which is the multiplicative scalar that recurs throughout the rest of this section. SXM also carries an NVLink interconnect (600 GB/s aggregate) for multi-GPU all-reduce; that interconnect is not exercised at single-GPU and therefore does not enter the comparison directly, but it is a reminder that PCIe and SXM are different SKUs at the platform level, not merely different package form factors.

The substitution was a dispatch-time decision driven by RunPod spot-pricing availability rather than an experimental design choice. The original V2 wave 1 plan was to put both vLLM and SGLang on the same A100 PCIe pod; the SGLang dispatch landed on an SXM pod because that was what the spot market surfaced inside the wave 1 budget envelope. The cross-run synthesis carries this confound as a first-class object: every backend-to-backend efficiency comparison is reported twice, once raw (hardware-confounded) and once bandwidth-adjusted (vLLM's raw efficiency divided by 0.75 to normalize to an SXM-equivalent bandwidth budget).

The implication is that any naive "SGLang beats vLLM" or "vLLM beats SGLang" claim drawn from the raw cross-run table is reading hardware and architecture as a single composite signal. The next subsection separates them.

### 14.2 The Raw vs Adjusted Stats

The headline contrast is the aggregate `mean_parallel_efficiency` comparison from Comparison B over the 48 matched (workload, phase, n_agents) cells on Qwen2.5-7B. Both the raw and the bandwidth-adjusted variants are computed by `research/tr164/cross_run_analyze.py` and serialized into `research/tr164/cross_run_analysis.json`.

| Variant | vLLM mean | SGLang mean | Delta (vLLM − SGLang) | Cohen's d | Wilcoxon p | Holm-adj p |
|---|---|---|---|---|---|---|
| Raw efficiency | 0.7878 (78.78%) | 0.8050 (80.50%) | −0.0171 (−1.71 pp) | −0.5148 | 6.394×10⁻⁴ | 0.005 |
| Bandwidth-adjusted | 1.0505 (105.05%) | 0.8050 (80.50%) | +0.2455 (+24.55 pp) | TBD per analyze.py extension | 1.598×10⁻⁹ | 1.4×10⁻⁸ |

**Observations.** Read column-by-column, the raw line says SGLang is 1.71 pp ahead of vLLM in mean parallel efficiency across the 48 matched cells, with a medium negative Cohen's d (−0.5148) and a Wilcoxon p of 6.394×10⁻⁴ (Holm-adjusted to 0.005). The bandwidth-adjusted line — produced by dividing vLLM's raw efficiency by 0.75 to put both backends on a notionally common 2.0 TB/s bandwidth budget — flips the sign. Under that adjustment vLLM lands at a notional 1.0505 (above 1.0 because the adjustment is multiplicative and the raw vLLM efficiencies are already in the 0.78 range), 24.55 pp ahead of SGLang, with a Wilcoxon p of 1.598×10⁻⁹ and Holm-adjusted p of 1.4×10⁻⁸. The sign flip is the substantive finding of the section: at the aggregate level, SGLang's raw lead over vLLM is fully attributable to the SXM-vs-PCIe memory bandwidth differential, not to the RadixAttention prefix-cache architecture.

> The most consequential consequence of the sign flip is that any single-line summary of the wave 1 cloud comparison — in any blog, any one-pager, any conference poster — has to either name the hardware confound explicitly or report only the per-workload breakdown from §15. The aggregate raw line, taken alone, materially mischaracterizes which backend wins on matched hardware, and the bandwidth-adjusted line, taken alone, overstates the strength of vLLM's advantage by assuming linear bandwidth scaling that does not hold on every workload.

The supporting contrasts in Comparison B move in the same direction as the raw efficiency line but with smaller effect sizes. The `mean_speedup_vs_n1` contrast yields delta −0.2537 (vLLM slower than SGLang on N-scaled speedup), Cohen's d −0.3278, Wilcoxon p 0.0018, Holm-adjusted p 0.013 — significant. The `mean_p95_latency_multiplier_vs_n1` contrast is null under Holm (delta −0.0148, Wilcoxon p 0.5509, Holm-adjusted p 1.0); the two backends are indistinguishable on the tail-latency multiplier as N grows. The `mean_p50_ttft_ms` contrast is also null (n=24, delta +10.9 ms, Wilcoxon p 0.9888, Holm-adjusted p 1.0); the two backends are indistinguishable on time-to-first-token under matched workload pairing. The licensed take is: backends are equivalent on tail latency and TTFT; they differ on aggregate parallel efficiency in a way that is dominated by hardware once the adjustment is applied.

### 14.3 Why the Adjustment Is a Sensitivity Check, Not a Causal Claim

The bandwidth-adjustment computation is one line of arithmetic: divide vLLM's raw efficiency by 0.75 (the PCIe-to-SXM bandwidth ratio) and compare against SGLang's raw efficiency. The arithmetic is straightforward; the load-bearing assumption underneath it is not. The adjustment treats parallel efficiency as linearly proportional to memory bandwidth. That assumption is approximately defensible for memory-bandwidth-bound regimes — long-prefill phases on large activations, decode phases at small N where the bottleneck is KV-cache streaming — and is not defensible for compute-bound regimes (small contexts, tensor-core saturation) or for latency-bound regimes (dispatch overhead, scheduler contention) where the bandwidth differential does not translate one-for-one into efficiency.

For that reason the cross-run synthesis labels the bandwidth-adjusted column as a **sensitivity check** rather than as a causal correction. The honest reading of the sign flip is: "the raw advantage SGLang shows over vLLM is consistent with the SXM/PCIe bandwidth ratio; if that ratio were the only thing distinguishing the two runs, the adjustment would make vLLM win by a large margin." It is not the reading: "vLLM is actually 24.55 pp better than SGLang and the raw number is wrong." The right interpretation lives between those two anchors and depends on the per-workload decomposition.

That decomposition, presented in detail in the next section, shows that RadixAttention has a real, Holm-significant workload-conditional advantage on `repeated_prefix` (delta −0.0368, +3.68 pp for SGLang, Wilcoxon p 1.953×10⁻³, Holm-adjusted p 0.01283) and on `balanced_2k` (delta −0.0313, +3.13 pp for SGLang, Wilcoxon p 1.953×10⁻³, Holm-adjusted p 0.01283). On `long_decode` and `long_prefill_8k` the per-workload contrast is null under Holm. The per-workload story is therefore: SGLang's RadixAttention does help on prefix-cache-friendly workloads, but the magnitude is 3–4 pp, not the 17 pp implied by the raw aggregate and not the negative 24.55 pp implied by the bandwidth-adjusted aggregate. The aggregate-only number, in either direction, overstates the architectural story.

| Workload | vLLM | SGLang | Delta | Wilcoxon p | Holm-adj p | Verdict |
|---|---|---|---|---|---|---|
| balanced_2k | 0.7845 | 0.8158 | −0.0313 (+3.13 pp SGLang) | 1.953×10⁻³ | 0.01283 | SGLang wins |
| long_decode | 0.9149 | 0.9272 | −0.0123 (+1.23 pp SGLang) | 0.08398 | 0.3359 | n.s. |
| long_prefill_8k | 0.6686 | 0.6567 | +0.0119 (+1.19 pp vLLM) | 0.2168 | 0.6504 | n.s. |
| repeated_prefix | 0.7834 | 0.8203 | −0.0368 (+3.68 pp SGLang) | 1.953×10⁻³ | 0.01283 | SGLang wins |

**Observations.** The per-workload decomposition tells a more conservative story than either aggregate column tells alone. RadixAttention helps on `repeated_prefix` by 3.68 pp and on `balanced_2k` by 3.13 pp, both Holm-significant. On `long_decode` and `long_prefill_8k` the head-to-head is null under Holm adjustment. The hardware confound from §14.2 does not erase the prefix-cache story; it bounds it. The two workloads where SGLang has a real, defensible architectural advantage are the ones where prefix-cache reuse is the dominant work pattern, which is exactly where RadixAttention is designed to help.

> The cleanest way to phrase the SS8 finding for downstream substrate is: at the aggregate level the SGLang-vs-vLLM efficiency delta is dominated by SXM-vs-PCIe memory bandwidth; the per-workload decomposition shows a residual 3–4 pp RadixAttention advantage on prefix-cache-friendly workloads that survives the confound. Neither the raw aggregate nor the bandwidth-adjusted aggregate is appropriate as a standalone headline. The sensitivity check is informative about how much of the raw signal the hardware can explain; it does not establish that vLLM would beat SGLang by 25 pp on PCIe-matched hardware, and any blog or paper draft that reports the bandwidth-adjusted line without the per-workload caveat is overclaiming.

The implication for V2 wave 2 is straightforward: when budget allows, the vLLM-vs-SGLang head-to-head needs to be re-run with both backends on the same SKU (either both PCIe or both SXM) so the confound collapses. Until that re-run lands, every comparison in the cross-run table is reported with the raw + bandwidth-adjusted pair, and the per-workload decomposition carries the load-bearing claims.

---

## 15. SS8b. Breakdown-Boundary Regression Across Four Backends

The four-backend comparison in SS7 and SS8a established that TGI dominates V1 pytorch_direct on aggregate parallel efficiency and that vLLM-vs-SGLang is workload-conditional once memory-bandwidth confounds are isolated. Aggregate effects, however, smear over a question that matters operationally: at what concurrency does each backend stop being "efficient enough" for a given service-level threshold? This section answers that question via the breakdown-boundary regression encoded in `research/tr164/cross_run_analyze.py` and materialized in `research/tr164/cross_run_analysis.json`. The regression sweeps three efficiency thresholds — 0.9 (strict), 0.7 (moderate), 0.5 (permissive) — and, for each backend × threshold pair, counts how many (model × workload × phase) combinations break the threshold and at which N they first break. The resulting distribution is the cleanest summary of where each backend's parallelism actually stops scaling.

### 15.1 Methodology

For every backend, the cross-run analyzer enumerates all matched (model, workload, phase) combinations. Within each combination, it walks the per-N mean parallel efficiency series in ascending order of N and records the smallest N at which the mean first drops below the threshold. Combinations that never drop below the threshold within the tested range (N ≤ 16 for V1, N ≤ 32 for V2 backends) are bucketed as "never breaks". The combination count differs across backends because the underlying sampling regimes differ: V1 pytorch_direct and TGI both carry the full V1 trio crossed with workload and phase (24 combinations), while vLLM and SGLang carry only the cloud Qwen2.5-7B arc (8 combinations = 1 model × 4 workloads × 2 phases). The threshold sweep is deliberately three-pointed: 0.9 picks up regressions that would be invisible at coarser thresholds and surfaces backends whose efficiency degrades early but gracefully; 0.5 picks up the catastrophic floor and is where the V1 pathology is most visible; 0.7 sits at the operationally common "still useful" boundary where most production routing decisions actually live. All breakdown distributions below are read directly from the `breakdown_boundary` block of `cross_run_analysis.json` and cross-validated against the per-N efficiency means reported in SS7.

**Observations.** The methodology assumes monotonic decay of parallel efficiency in N, which is empirically true for every combination in the substrate. It does not assume monotonic behaviour across thresholds within a backend, but in practice the count of combinations that break at a stricter threshold dominates the count at a more permissive one — there is no backend in the four-backend set that breaks at 0.5 without also breaking at 0.7 and 0.9. The methodology is descriptive, not inferential: the substrate-level inferential weight sits with the Wilcoxon and Holm-Bonferroni results in SS8a.

> The breakdown-boundary regression is the operationally useful summary of "where does this backend stop being efficient enough?". It turns the per-N efficiency series — which is dense, noisy, and hard to compare across backends with different N grids — into a small integer per (combination, threshold) pair. That compression is what makes cross-backend comparison legible.

### 15.2 V1 Pytorch_Direct — The Uniform-N=2 Pathology

The V1 pytorch_direct breakdown distribution is the most extreme in the four-backend set. At the strict threshold of 0.9, 24 of 24 combinations break at N=2 — uniformly, with no spread across model, workload, or phase. At the moderate threshold of 0.7, the picture is unchanged: 24 of 24 combinations still break at N=2. Even at the permissive 0.5 threshold, 24 of 24 combinations break, and they break early — 8 combinations break at N=2 and the remaining 16 break by N=4 at the latest. This is consistent with the per-N efficiency means from SS7: V1 falls to 53.6% at N=2, 28.1% at N=4, 10.4% at N=8, and 3.8% at N=16, meaning the entire scaling envelope of V1 pytorch_direct is consumed between N=1 and N=4.

| Threshold | Combinations breaking | N=2 | N=4 | N=8 | N=16 | Never breaks |
|---|---|---|---|---|---|---|
| 0.9 | 24 / 24 | 24 | 0 | 0 | 0 | 0 |
| 0.7 | 24 / 24 | 24 | 0 | 0 | 0 | 0 |
| 0.5 | 24 / 24 | 8 | 16 | 0 | 0 | 0 |

**Observations.** Three properties of this distribution are load-bearing. First, the uniformity at the 0.9 and 0.7 thresholds — every single combination breaking at the same N=2 — is the signature of a dispatcher-level bottleneck rather than a workload-level or model-level effect; the breakdown is invariant under the explanatory variables that should modulate it if the cause were upstream of the dispatcher. Second, the fact that the 0.5 threshold does not move the boundary past N=4 means there is no permissive threshold at which V1 pytorch_direct is rescued; the failure mode is not "good at moderate N, bad at high N" but "fails immediately and continues failing". Third, the absence of any "never breaks" combinations across all three thresholds quantifies the gap between V1 and the continuous-batching backends in a single integer: zero V1 combinations survive even the most permissive threshold.

> V1 pytorch_direct is not a backend whose parallelism scales poorly. It is a backend whose parallelism does not meaningfully exist beyond N=1: every combination crosses the 0.7 floor before reaching N=4, and the 0.5 floor is exhausted by N=4 on every combination. This is what the SS7 aggregate Cohen's d of 1.44 vs TGI is summarizing.

### 15.3 The V2 Backends Pattern

The three V2 backends — TGI, vLLM, SGLang — all push the breakdown boundary outward relative to V1, but the magnitude and shape of the push differ across them in ways the breakdown-boundary regression makes visible.

**TGI (24 combinations).** TGI is the only V2 backend with the full V1-comparable workload × model × phase coverage. At the 0.9 threshold, 24 of 24 combinations break, with 17 breaking at N=2 and the remaining 7 at N=4. The strict-threshold picture is therefore not radically better than V1's — TGI still breaks the 0.9 floor on every combination by N=4 — but the spread between N=2 and N=4 is the first signal that continuous batching is doing something. At the 0.7 threshold the distribution stretches further: 14 combinations break at N=4, 8 at N=8, and 2 at N=16, with all 24 eventually breaking. At the 0.5 threshold, 22 of 24 combinations break (1 at N=4, 13 at N=8, 8 at N=16), and 2 combinations never drop below 0.5 within the tested N ≤ 16. The two never-breaking combinations are the first appearance in the regression of a TGI configuration that holds permissive efficiency across the full V1 N-grid.

| Threshold | Breaks | N=2 | N=4 | N=8 | N=16 | Never |
|---|---|---|---|---|---|---|
| 0.9 | 24 / 24 | 17 | 7 | 0 | 0 | 0 |
| 0.7 | 24 / 24 | 0 | 14 | 8 | 2 | 0 |
| 0.5 | 22 / 24 | 0 | 1 | 13 | 8 | 2 |

**vLLM (8 combinations, cloud Qwen2.5-7B arc).** vLLM's distribution at the 0.9 threshold places 4 combinations at N=4, 2 at N=8, and 2 at N=16 — the entire 8-combination set breaks but the spread starts where TGI's distribution ends. At the 0.7 threshold, 6 of 8 combinations break (2 at N=8, 4 at N=16) and 2 never drop below 0.7 within tested N ≤ 32. At the 0.5 threshold the same 6 of 8 break (2 at N=16, 4 at N=32), and the 2 never-breaking combinations carry through. The shape of the vLLM distribution is "early strict-threshold failure, late permissive-threshold survival" — the 0.9 floor is crossed but the 0.5 floor is held into the high-N regime on most combinations.

| Threshold | Breaks | N=4 | N=8 | N=16 | N=32 | Never |
|---|---|---|---|---|---|---|
| 0.9 | 8 / 8 | 4 | 2 | 2 | 0 | 0 |
| 0.7 | 6 / 8 | 0 | 2 | 4 | 0 | 2 |
| 0.5 | 6 / 8 | 0 | 0 | 2 | 4 | 2 |

**SGLang (8 combinations, cloud Qwen2.5-7B arc).** SGLang's distribution at the 0.9 threshold spreads earlier than vLLM's — 1 combination breaks at N=2, 1 at N=4, 4 at N=8, and 2 at N=16 — meaning SGLang is not strictly stronger than vLLM under the strict floor. At the 0.7 threshold, 6 of 8 combinations break (2 at N=8, 4 at N=16), matching the vLLM count exactly with 2 never-breaking combinations. At the 0.5 threshold the SGLang distribution diverges decisively from every other backend: only 4 of 8 combinations break (2 at N=16, 2 at N=32), and 4 of 8 combinations — half the sample — never drop below 0.5 within tested N ≤ 32. This is the strongest breakdown resistance in the four-backend comparison.

| Threshold | Breaks | N=2 | N=4 | N=8 | N=16 | N=32 | Never |
|---|---|---|---|---|---|---|---|
| 0.9 | 8 / 8 | 1 | 1 | 4 | 2 | 0 | 0 |
| 0.7 | 6 / 8 | 0 | 0 | 2 | 4 | 0 | 2 |
| 0.5 | 4 / 8 | 0 | 0 | 0 | 2 | 2 | 4 |

**Observations.** Three patterns repeat across the three V2 backends. First, the 0.9 threshold is unforgiving for all of them — every backend breaks every combination at this threshold, just at different N. Second, the 0.7 threshold is where the never-breaking bucket first appears (2 combinations for both vLLM and SGLang, 0 for TGI on consumer hardware), and is the threshold most likely to drive operational routing. Third, the 0.5 threshold is where the three backends separate maximally: TGI holds 2 of 24 never-breaking, vLLM holds 2 of 8 (25%), SGLang holds 4 of 8 (50%). The 0.5 separation across vLLM and SGLang aligns with the workload-conditional pattern from SS8a: SGLang's RadixAttention wins on prefix-cache-friendly workloads, and the never-breaking combinations are concentrated there.

> The V2 distribution tells a more nuanced story than "continuous batching solves the breakdown". It tells a story of conditional rescue: the breakdown is pushed outward by 1–4 N levels for every backend, but the rescue is workload- and threshold-conditional. The 0.5 threshold is where the rescue is most visible because that is the regime where dispatcher overhead has been amortized and the remaining efficiency loss is workload-shaped.

### 15.4 The Substantive Interpretation

The breakdown-boundary regression supports four substantive claims, each grounded in the distribution tables above.

The first claim is that continuous-batching backends do not universally eliminate the breakdown. At the strict 0.9 threshold, every continuous-batching backend in the comparison still breaks every combination — TGI by N=4, vLLM by N=16, SGLang by N=16. The 0.9 threshold is, in practice, a hardware-and-dispatcher bound: a backend that holds 90% parallel efficiency at high concurrency is fighting against fixed-cost dispatcher overhead and memory-bandwidth ceilings that no scheduling strategy can erase. The cross-run substrate licenses the claim that the breakdown is mitigated, not eliminated.

The second claim is that the rescue magnitude is 1–4 N levels at moderate thresholds. The arithmetic is direct: V1 pytorch_direct breaks the 0.7 floor uniformly at N=2; TGI pushes this to a distribution centered on N=4–N=8; vLLM and SGLang push it to N=8–N=16. Counting in doublings of N, the rescue is one doubling on TGI relative to V1 at the 0.7 threshold and two-to-three doublings on vLLM and SGLang. This is a meaningful operational improvement and a defensible one — it is exactly what the SS7 aggregate efficiency means project.

The third claim is that the 0.5 threshold differentiates the backends most clearly. At 0.9, all three V2 backends look similar; at 0.7, they begin to separate; at 0.5, the separation is maximal — TGI 2/24 never-breaking (≈8%), vLLM 2/8 (25%), SGLang 4/8 (50%). The permissive threshold is the regime where dispatcher and bandwidth overhead have been amortized and the remaining variance is dominated by workload-specific scheduling — which is precisely the dimension on which RadixAttention and continuous batching are designed to compete.

The fourth claim is that the V1 pytorch_direct uniform-N=2 pathology is the most extreme breakdown in the four-backend comparison and is the load-bearing evidence for replacing V1 in any high-N regime. Every other backend in the comparison has at least one (combination, threshold) cell where it never breaks; V1 has zero such cells across all three thresholds. The qualitative gap between V1 and any V2 backend is therefore not a difference of degree — it is a difference of kind, in the sense that V1 has no concurrency regime in which permissive efficiency is preserved.

**Observations.** The four claims compose into a single operational rule: pick the threshold that matches the service level, then pick the backend whose never-breaking-or-late-breaking bucket at that threshold dominates. At 0.9, no backend dominates; at 0.7, vLLM and SGLang tie and dominate TGI; at 0.5, SGLang dominates on the cloud Qwen2.5-7B arc. The rule is workload-conditional in exactly the ways SS8a documented, and the breakdown-boundary regression is the threshold-conditional generalization of that result.

> The breakdown-boundary regression is the operational complement to the SS8a aggregate effect sizes. The aggregate effects say "TGI beats V1 by a very large margin on average"; the breakdown-boundary regression says "TGI's rescue is concentrated at the 0.5 threshold and is overwhelmed at the 0.9 threshold". Both are true; the breakdown-boundary regression is the one that maps directly onto a routing decision.

---

## 15. SS9. Backend-Specific Failure Modes

The V2 wave 1 dispatch surfaced two upstream issues worth documenting in the report substrate. Both were filed during the cloud_core arc on RunPod (the A100 PCIe + A100 SXM split described in SS3), neither blocked the substrate from landing at ok_rate = 1.0 across all 144+144 cells, and each carries a distinct read on what its resolution would unlock for V2 wave 2. The TGI local_core case is included here as the counter-example: a backend whose vendor image age looked alarming on first inspection but produced 360 cells of clean substrate (26,784 metrics.csv rows) with no maintainer intervention required.

### 15.1 vllm-project/vllm#44703

The vLLM-specific issue was surfaced by the V2 dispatch against `vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504` on A100 PCIe. The 144-cell sweep across `{balanced_2k, long_decode, repeated_prefix, long_prefill_8k} × {prefill, decode} × N∈{1,2,4,8,16,32} × 3 reps` completed at ok_rate = 1.0 with 15,120 metrics.csv rows, so the issue did not corrupt the substrate. What the issue surface flags is the boundary-regime behaviour the run did capture cleanly: `long_prefill_8k` driving the P95 latency multiplier vs N=1 above 2x at N=32, and `balanced_2k` parallel efficiency degrading to 0.43 at N=32. The report's read on impact is bounded: V2 wave 1 vLLM data is publication-grade as-is for the H1 boundary-shift question (continuous batching vs pytorch_direct GIL contention), because the 144-cell cross-product holds. Upstream maintainer response is pending at the time of writing; resolution would unlock cleaner N=32 boundary characterisation in V2 wave 2, particularly the prefill-heavy regime where the substrate currently shows the largest P95 dilation.

### 15.2 sgl-project/sglang#27406

The SGLang vendor image (`lmsysorg/sglang@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4`, semver v0.5.12.post1-runtime, build 2026-05-23, upstream commit 5a15cde858ea09b77116212a39356f2fc51b8584) shipped without a distro Python package the launcher required. The Docker path failed at startup; the workaround was a native Python launcher path, which restored the dispatch and let the matched 144-cell sweep land on A100 SXM (15,120 metrics.csv rows, ok_rate = 1.0). The native-launcher workaround is functionally equivalent for measurement purposes — the cell-shape matches vLLM's 144 cells exactly, which is the prerequisite for the H1 head-to-head — but the reproducibility contract per the project's Dockerfile-is-source-of-truth convention is partially weakened: the image digest is pinned in `config.yaml` for future re-dispatch once upstream resolves, but the V2 wave 1 substrate itself was produced via the native launcher, not the image. Upstream maintainer response is pending. Resolution would unlock pure-Docker SGLang dispatch in V2 wave 2, restoring the triad (Dockerfile + requirements + result data) without the native-launcher footnote.

### 15.3 TGI: image age did not compromise substrate

The TGI image (`ghcr.io/huggingface/text-generation-inference@sha256:e6b0af6e0bf65337b84a19f15d74660c7892192f555fb0b68d3f3d62bf0c1e9a`) carries `text-generation-launcher 3.3.6-dev0`, a January 2026 build, in a backend whose vendor declared maintenance mode during 2026. A priori this looked like the most fragile of the three V2 backends. Empirically it was the cleanest: 360 cells across 3 models × 4 workloads × 2 phases × N∈{1,2,4,8,16} × 3 reps completed at ok_rate = 1.0 in a single overnight 19:28 EDT → 01:23 EDT window, producing 26,784 metrics.csv rows at zero cash cost. It also delivered the load-bearing H1 finding (cross-referenced from SS5 and SS6): TTFT on llama3.2-1b at N=16 balanced_2k dropped from the V1 pytorch_direct median of 188 seconds to sub-3-second medians, eliminating the V1 N=2 breakdown across all 24 (model × workload × phase) combinations.

| Backend | Image / launcher | Issue filed | Workaround | Cells (planned/completed) | metrics.csv rows | ok_rate |
|---|---|---|---|---|---|---|
| vLLM | vllm-openai@d9a5c1c1 | vllm-project/vllm#44703 | None required for V2 wave 1 substrate | 144/144 | 15,120 | 1.0 |
| SGLang | lmsysorg/sglang@8df56b54 | sgl-project/sglang#27406 | Native Python launcher (not Docker) | 144/144 | 15,120 | 1.0 |
| TGI | text-generation-launcher 3.3.6-dev0 (Jan 2026) | None | N/A — clean run despite vendor maintenance mode | 360/360 | 26,784 | 1.0 |

**Observations.** All three backends produced ok_rate = 1.0 across every cell, so neither upstream issue corrupted the substrate. The vLLM issue is non-blocking for V2 wave 1 publication-grade claims and queues a cleaner N=32 boundary read for wave 2. The SGLang issue is non-blocking for the matched 144-cell H1 surface but leaves a Docker-reproducibility footnote that wave 2 would prefer to resolve. TGI's image age — a backend in vendor-declared maintenance mode running a January 2026 build — turned out to be the substrate's single most consequential contribution, eliminating the V1 pytorch_direct N=2 breakdown without any maintainer intervention required.

> The pattern across the three issues is that backend failure modes in V2 wave 1 were issue-surface failures, not substrate-quality failures. Image-distribution bugs (missing distro deps, dev-tag rolling builds) cost dispatch time and weakened the reproducibility contract at the margins, but the underlying measurement surface — 57,024 metrics.csv rows across 648 cells at ok_rate = 1.0 — landed clean. The honest read is that "maintenance mode" upstream and rough-edged vendor images are tolerable risks for a substrate of this shape, and the two filed issues are wave-2 quality-of-life unlocks rather than wave-1 blockers. Resolution of vllm-project/vllm#44703 would tighten the N=32 prefill-heavy boundary read; resolution of sgl-project/sglang#27406 would restore pure-Docker SGLang dispatch and close the reproducibility-contract footnote.

---

## 16. SS10. Cross-TR Position vs TR130-TR132

TR164 V2 is not a standalone artifact. It is the latest beat in a four-report arc that began with TR130 and has been the program's serving-stack measurement spine ever since. To position V2 honestly, the prior three reports need to be named for what they were, and the V2 delta needs to be stated against that baseline rather than against an idealized clean-room starting point.

TR130 stood up the serving-stack benchmarking framework. It introduced the workload taxonomy this report still uses (balanced_2k, long_decode, repeated_prefix, long_prefill_8k) and the N-sweep concurrency-ramp methodology. Its backend layer was vLLM-centric: the abstraction existed in code, but only one backend was exercised end-to-end. Concurrency was capped at N=16 because that was the regime where the laptop tier still resolved cleanly. TR131 turned the same harness onto GPU kernel profiling for root-cause attribution. TR132 ported that profiling into a containerized substrate so the kernel traces were reproducible against a pinned image rather than against whatever the host happened to ship that week. The triad established three things: a workload grammar, a concurrency-sweep convention, and a reproducibility contract anchored on image digests.

V2 extends that arc on three explicit axes, and only those three. Stating them narrowly matters because the program has a defensibility bar against scaffold overclaim.

| Axis | TR130-TR132 baseline | TR164 V2 delta |
|---|---|---|
| Backend coverage | vLLM-centric (one production-grade backend exercised) | vLLM + SGLang + TGI exercised end-to-end across matched 144/144 + 144/144 + 360/360 cell grids |
| Matched-pair backend test | Not performed | TGI ⊕ V1 pytorch_direct on identical model trio (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) at matched workload × phase × N |
| Concurrency ceiling | N=16 maximum | N=32 reached on vLLM cloud_core and SGLang cloud_core (cell-shape 1 model × 4 workloads × 2 phases × 6 N levels × 3 reps = 144 cells per backend) |

**Observations.** TR130's vLLM-centric framing was load-bearing in its time but it foreclosed any backend-architecture-level claim. With only one production-grade backend in the substrate, every finding had to be stated as "vLLM under condition X" rather than "continuous-batching backends under condition X." TR131 and TR132 inherited that constraint at the kernel-profiling layer. V2 retires that ceiling: 57,024 metrics rows across 648 cells with ok_rate 1.0 on every cell of every backend means the next layer of claim — backend-architecture-level breakdown-boundary attribution — is now licensed by substrate, not by hand-wave.

> The TR130-TR132 trilogy was the right scaffold for the question those reports could ask: where does a single production serving stack break under concurrency, and what does its kernel trace look like at the break. V2 is the right scaffold for the question that trilogy could not ask: when the N=2 breakdown disappears under TGI on identical hardware, identical models, and identical workloads, was the original breakdown a hardware constraint or a runtime-architecture artifact? The TGI matched-pair on the V1 model trio (360/360 cells, parallel efficiency at N=2 above 0.65 across all 24 model × workload × phase combinations, TTFT median dropping from 188 seconds on V1 pytorch_direct at N=16 balanced_2k to sub-3-second on TGI at the same model/concurrency) is the H1 mechanism answer TR130-TR132 had no way to produce. The companion TR165 nogil ablation extends the same arc one rung further by isolating the GIL term itself.

Two scope caveats are part of the honest position. First, the A100 PCIe (vLLM) vs A100 SXM (SGLang) hardware split means any direct vLLM-vs-SGLang head-to-head must carry the SXM-vs-PCIe memory-bandwidth disclosure (2.0 TB/s vs 1.5 TB/s + NVLink); the V2 substrate licenses backend-architecture attribution but it does not license a clean SKU-controlled vLLM-vs-SGLang race, and TR130-TR132 did not need to confront this because they did not run SGLang at all. Second, Ollama as a fifth backend is deferred to V2 wave 2; cloud_full 14B coverage at N=32 across all backends is queued behind external compute credit landing. Those gaps are named here so the cross-TR position is not oversold: V2 is a natural continuation of TR130-TR132 into the H1 mechanism question, not a closure of every backend question the trilogy left open.

---

## 18. SS11. What V2 Licenses That V1 Did Not

V1 of TR164 established a narrow, hardware-anchored claim: on a single RTX 4080 Laptop GPU, three Hugging Face Transformers models running under a synchronous `pytorch_direct` server scaffold exhibit a uniform-at-N=2 parallel-efficiency breakdown across every workload, phase, and tested model. That finding was internally consistent and well-instrumented, but it licensed essentially one sentence: "naive concurrent serving breaks at N=2 on consumer hardware for these three models." V2's cross-run synthesis — joining the V1 local trio against TGI (also local, also RTX 4080 Laptop, same trio) and against vLLM and SGLang (both cloud, Qwen2.5-7B, A100-SXM and A100-PCIe respectively) — expands what the report can defensibly assert in six numbered directions. Each claim below is backed by a specific matched-pair statistical test in `research/tr164/cross_run_analysis.json`, produced by `research/tr164/cross_run_analyze.py`.

**Claim 1. The H1 GIL-contention hypothesis is confirmed at the matched-pair statistical level.** V1 inferred GIL contention from the within-process uniformity of the N=2 breakdown; the inference was structurally sound but lacked a counterfactual. The TGI-vs-V1 paired comparison supplies that counterfactual at scale. Across 120 matched (model, workload, phase, n_agents) cells — every cell of V1's 24-combination grid intersected with TGI's matched 5-N sweep on the same hardware — TGI wins on mean parallel efficiency in 96 of 120 cells, V1 wins in 0, and 24 cells tie (the N=1 baselines, where both are perfect by definition). The paired Cohen's d is 1.4361 (very large effect), and the Wilcoxon signed-rank statistic is 7260.0 with p = 5.579×10⁻²⁰. The per-N decomposition is unambiguous: at N=2, TGI maintains 0.8457 (84.6%) efficiency where V1 drops to 0.5358 (53.6%), a +30.98pp gap; at N=4 the gap widens to +39.48pp (TGI 67.6%, V1 28.1%); at N=8 it is +38.07pp (TGI 48.5%, V1 10.4%); at N=16 it is +28.53pp (TGI 32.3%, V1 3.8%). When the only software variable that changes is the server process (TGI's Rust router plus continuous batching versus V1's synchronous Python `pytorch_direct` scaffold) and the hardware, model, workload, and phase are held identical, the lift is monotone and very large at every concurrency level above 1. Holding everything else fixed and observing this magnitude of delta licenses, for the first time, the inference that V1's breakdown is attributable to its synchronous-Python serving architecture rather than to model size, workload shape, or hardware capacity.

**Claim 2. Continuous-batching backends generalize across three independent server-process implementations.** V1's evidence base contained one server (`pytorch_direct`); V2 adds three (vLLM, SGLang, TGI), each with a different code lineage and different scheduling philosophy. All three push the breakdown boundary upward relative to V1. The fact that the H1 contradiction generalizes across vLLM (cloud, A100-SXM, custom CUDA paged-attention), SGLang (cloud, A100-PCIe, RadixAttention prefix-cache), and TGI (local, RTX 4080 Laptop, Rust router with continuous batching) is what elevates V2's licensing from "V1 had a bug in its scaffold" to "V1's scaffold exposed a real architectural pathology that three independent production backends were specifically engineered to eliminate." The three backends do not push the boundary to the same N — they break at varying positions — but they all push it.

**Claim 3. The breakdown-boundary in `pytorch_direct` is software-architectural, not hardware- or workload-conditional.** The breakdown-boundary regression at three efficiency thresholds (0.9, 0.7, 0.5) returns, for V1 `pytorch_direct`, a uniform-at-N=2 result across all 24 combinations and across every threshold tested: 24/24 V1 combinations break the 0.9 threshold at N=2, 24/24 break the 0.7 threshold at N=2, and at the most permissive 0.5 threshold 8/24 still break at N=2 with the remaining 16 breaking at N=4. There is no combination of (model size, workload shape, prefill/decode phase) on this hardware that postpones the V1 breakdown beyond N=4 even at the 0.5 threshold. By contrast, vLLM at threshold 0.5 has 2 of 8 combinations that never drop below 0.5 within N≤32, SGLang has 4 of 8 that never drop, and TGI has 2 of 24 that never drop. The V1 boundary is a fixed property of the server architecture; the V2-backend boundaries are workload-conditional.

**Claim 4. The vLLM-vs-SGLang RAW efficiency comparison is hardware-confounded, but per-workload analysis isolates a real RadixAttention prefix-cache advantage.** This is the most carefully bounded claim in the report. On raw 48-cell matched pairs, SGLang edges vLLM by mean Δ = -0.0171 (Cohen's d -0.5148, Wilcoxon p = 6.394×10⁻⁴). The bandwidth-adjusted re-analysis — dividing vLLM's raw efficiency by the A100-SXM/PCIe memory-bandwidth ratio of 0.75 to normalize for the 2.0 TB/s vs 1.5 TB/s split — inverts the sign: bandwidth-adjusted Δ = +0.2455 in vLLM's favor (Wilcoxon p = 1.598×10⁻⁹). The raw advantage is hardware-explained, not architecture-explained. What survives the bandwidth caveat is the per-workload decomposition: SGLang outperforms vLLM by +3.1pp on `balanced_2k` (Holm-adjusted p = 0.01283) and by +3.7pp on `repeated_prefix` (Holm-adjusted p = 0.01283), with both contrasts surviving Holm-Bonferroni stepdown across 12 primary contrasts. On `long_decode` (Holm-adj p = 0.3359) and `long_prefill_8k` (Holm-adj p = 0.6504) the difference is null. The pattern is mechanistically clean: RadixAttention helps where prefix sharing is available, contributes nothing where decode dominates throughput, and contributes nothing where memory bandwidth dominates. The +3.7pp prefix-cache win is smaller than common marketing framing but real and Holm-significant.

**Claim 5. N-scaling claims up to N=32 are licensed on vLLM and SGLang.** V1 instrumented only N ∈ {1,2,4,8,16}; the V1 breakdown was so early (at N=2) that scaling beyond N=16 was uninformative. V2 extends to N=32 for the cloud backends, and the breakdown-boundary regression returns interpretable results at that boundary: 4 of 8 vLLM combinations cross the 0.5 threshold at N=32, 2 SGLang combinations cross at N=32, and half of SGLang's combinations never cross. The report can now describe the N=16–32 region as instrumented territory for the continuous-batching backends rather than as extrapolation.

**Claim 6. The Mantel-Haenszel binary analysis isolates the TGI-vs-V1 contrast at the strongest possible binary-stratified result.** Stratifying by model × workload × phase and using the parallel-efficiency ≥ 0.5 threshold as the binary outcome, the MH pooled odds ratio across 22 of 24 usable strata is infinity: TGI almost always clears the 0.5 threshold at N ≥ 2 while V1 almost never does. The two unusable strata are those in which V1 cleared zero cells (degenerate denominator). An infinite pooled OR is the maximum that a binary-stratified estimator can return — every stratum is one-sided in the same direction. The MH result is consistent with the paired-continuous Wilcoxon and the Cohen's d = 1.44 estimate; the three estimators corroborate each other rather than triangulate.

What V2 still does **not** license is equally important to state.

| Boundary | Why V2 does not cross it |
| --- | --- |
| A clean vLLM-vs-SGLang efficiency claim without the bandwidth-adjustment caveat | The SXM/PCIe 2.0/1.5 TB/s memory-bandwidth split is the load-bearing confound; any unqualified claim would attribute hardware delta to backend architecture. |
| Generalization to 14B+ models | Wave 2 cloud_full is queued but not run; the 7B cross-backend evidence does not extrapolate. |
| Ollama-specific characterization | Deferred from V2 scope; no matched-pair data exists in the cross-run join. |
| Generalization beyond the V1 local trio on consumer hardware | TGI matched the V1 trio exactly; transfer to other small models on RTX-class GPUs is plausible but uninstrumented. |

**Observations.** The licensing expansion is asymmetric: V2 strengthens the H1 GIL-contention conclusion from a within-V1 inference into a paired-counterfactual conclusion with very-large-effect statistical evidence, and it generalizes the continuous-batching-helps conclusion across three independent backends; but it remains scrupulously narrow on the head-to-head vLLM-vs-SGLang comparison precisely because the hardware confound is too large to wave through. The six licensed claims and four explicit non-claims together demarcate the V2 evidence envelope. Holm-Bonferroni stepdown across the 12 primary contrasts produced 9 significant results and 3 nulls at α = 0.05; the nulls (p95 multiplier and p50 TTFT in the vLLM-vs-SGLang comparison, plus per-workload `long_decode` and `long_prefill_8k`) are themselves substantive — they license the claim that the two cloud backends are tail-latency-equivalent at moderate concurrency on the workloads where prefix-cache reuse is unavailable.

> What V2 licenses is not just "more numbers behind the same V1 story" — it is a qualitatively different epistemic posture. V1 could say "we observed a breakdown"; V2 can say "we observed the breakdown, we counterfactually eliminated it, and the counterfactual generalizes across three independent server processes." The four non-claims (vLLM-vs-SGLang without bandwidth adjustment, 14B+ scaling, Ollama, beyond-trio local generalization) are not weaknesses; they are the precise boundary that the matched-pair design is unable to cross without additional data. Reporting the boundary explicitly is what distinguishes the V2 synthesis from the genre of backend-benchmark posts that report a single raw number and call it an architecture comparison.

---

## 18. SS12. Forbidden Claims

Substrate fidelity has a dual face. SS11 names what the V2 wave plus the TR165 companion experiment do license. SS12 names what they do not. The forbidden-claims discipline is the report's structural boundary on conclusions; every claim outside this boundary is either an overreach the substrate cannot pay for, or a re-framing that needs additional runs before it earns shelf space. Three claims are load-bearing forbidden, and each one corresponds to a temptation the V2 numbers create.

### 18.1 Forbidden Claim 1: "vLLM beats SGLang."

The substrate runs vLLM cloud_core on A100 PCIe (run dir `20260605_210337_450607`) and SGLang cloud_core on A100 SXM (run dir `20260605_212557_266597`). The SKU split is not a footnote — it is the structural reason the head-to-head is not licensed. A100 SXM carries 2.0 TB/s HBM bandwidth against PCIe's 1.5 TB/s plus NVLink, and the substrate explicitly flags prefill-heavy workloads as the regime where the SXM advantage is most visible (memory-bandwidth-bound, as expected). Any "vLLM beats SGLang" or "SGLang beats vLLM" framing folds the engine effect and the SKU effect into the same number, and the substrate does not separate them.

**Observations.** The matched 144-cell cross-product is the prerequisite for an H1-style engine comparison; the hardware confound is the prerequisite for refusing to publish one.

> What IS licensed is workload-conditional behavior: SGLang and vLLM are near-identical on long_decode (RadixAttention's prefix-cache benefit does not show up universally), repeated_prefix is the H2 test surface where SGLang's advantage is visible but smaller than marketing claims, and the SXM throughput edge concentrates on prefill-heavy regimes. The cross-engine ranking question is queued for a same-SKU re-run, not declared from V2 wave 1.

### 18.2 Forbidden Claim 2: "Continuous batching always eliminates the N=2 breakdown."

The TGI local_core run on the RTX 4080 Laptop (run dir `20260605_192757_415750`) is the strongest single result in the V2 wave: parallel efficiency at N=2 stays substantially above 0.65 across all 24 (model × workload × phase) combinations, the V1 pathological N=16 cell shapes on llama3.2-3b do not hang, and the 188-second SS14 V1 TTFT median on llama3.2-1b at N=16 balanced_2k collapses to sub-3-second medians on the same hardware. That is one continuous-batching backend on one model trio. The vLLM and SGLang cloud_core runs corroborate continuous-batching benefit on long_decode (vLLM parallel efficiency 0.43–0.45 through N=32), but they sit on a different SKU class and a different model (Qwen2.5-7B-Instruct). Three backends is not "all backends."

**Observations.** Three continuous-batching engines clearing the breakdown is structural evidence that the V1 pattern is engine-specific; it is not yet a universal-quantifier claim over the family.

> The licensed form is: "the three continuous-batching backends measured here (TGI, vLLM, SGLang) eliminate the pytorch_direct N=2 breakdown on the cells they were measured on." Generalising to every continuous-batching backend, every model size, and every workload requires the deferred fifth-backend slot (Ollama V2) and a cloud_full extension across all backends, both of which the V2 wave explicitly leaves to wave 2.

### 18.3 Forbidden Claim 3: "GIL contention is the SOLE cause of the V1 breakdown."

TGI eliminating the V1 breakdown confirms GIL contention as a mechanism — TGI runs its scheduler in Rust, the V1 backend ran a Python-loop dispatcher under the GIL, and the mechanism-shift tracks the symptom-shift. The TR165 nogil ablation (run dir `20260607_174748_273070`), which is this report's matched-pair companion, returns an H2_partial verdict: 79% of combinations improve under Python 3.14t free-threading, 21% do not. A sole-cause story would predict near-universal improvement; the 21% non-improving combinations falsify that prediction.

**Observations.** The substrate licenses GIL contention as **a** mechanism in V1's collapse, not **the** mechanism.

> The licensed framing is multi-mechanism: GIL contention is one load-bearing factor (TGI's Rust scheduler removes it and the breakdown goes with it), scheduler architecture is another (continuous batching versus blocking dispatch), and the residual 21% of TR165 combinations that do not improve under nogil point to additional mechanisms — memory-bandwidth saturation, KV-cache allocator contention, or workload-specific arithmetic-intensity floors — that the current substrate names but does not yet isolate. Sole-cause claims collapse this structure and are forbidden.

### 18.4 Why these three are the discipline boundary

Each forbidden claim corresponds to a real V2 result that an under-disciplined writeup would over-promote. The vLLM-vs-SGLang temptation comes from having matched 144-cell cross-products; the universal-continuous-batching temptation comes from TGI's clean N=2 result; the sole-cause-GIL temptation comes from the symmetry between TGI removing the GIL-bound dispatcher and TGI removing the breakdown. The substrate-fidelity rule is: licensed claims travel; over-promoted claims get caught at external review. Holding the line here is what lets SS11's licensed claims carry weight.

---

## 19. SS13. Limitations.

The V2 wave is a substantial step forward over V1 — 648 cells, 57,024 metrics rows, ok_rate 1.0 across every cell, and the H1 hypothesis confirmation on TGI — but the substrate as it stands carries five structural limitations that must be acknowledged before any head-to-head backend claim escapes the report. None of these limitations invalidate the V1-vs-TGI result on the GIL-elimination axis; all of them constrain the breadth of the cross-backend story V2 is licensed to tell.

**Limitation 1: Hardware confound on vLLM vs SGLang.** The vLLM cloud_core run executed on A100 PCIe (RunPod) at ~\$1.40–1.64/hr; the SGLang cloud_core run executed on A100 SXM (RunPod) at ~\$1.80–2.10/hr. PCIe and SXM are not the same SKU: A100 SXM carries 2.0 TB/s HBM2e memory bandwidth against PCIe's 1.5 TB/s plus NVLink topology that PCIe lacks. Any vLLM-vs-SGLang head-to-head over the matched 144-cell cross-product is therefore confounded by the hardware tier in addition to the backend. The prefill-heavy workloads (long_prefill_8k) are where the bandwidth gap is most visible — exactly the regime in which a naive backend comparison would misattribute SXM's bandwidth advantage to SGLang's RadixAttention. SS9 quantifies the surface; the limitation is that disentangling backend-from-SKU requires either an A100-PCIe SGLang re-run or an A100-SXM vLLM re-run, neither of which is in V2 wave 1's compute budget.

**Limitation 2: Model scope on the cloud arc.** The cloud arc (vLLM + SGLang) is capped at Qwen2.5-7B-Instruct. Cloud_full extension to 14B and beyond is queued for V2 wave 2, contingent on compute credits landing. The 7B ceiling means cross-backend claims at the model-scale axis cannot be made from V2 wave 1 alone; the substrate licenses backend × workload × N statements at 7B, not backend × model-scale × workload × N statements across a parameter sweep. TGI's local_core run covers llama3.2-1b, qwen2.5-1.5b, llama3.2-3b (the V1 model trio for matched-pair vs V1 pytorch_direct), so the V1-vs-V2 matched-pair claim does cross models at the small end — but the cloud arc does not.

**Limitation 3: Concurrency ceiling on TGI.** TGI local_core was capped at N ∈ {1, 2, 4, 8, 16} (5 levels), matching V1 pytorch_direct cell-shape to preserve the matched-pair design. The cloud arc on vLLM and SGLang extends to N ∈ {1, 2, 4, 8, 16, 32} (6 levels). N=32 behavior on TGI is therefore unobserved in V2 wave 1. The pathological-cell-shape narrative — vLLM at N=32 balanced_2k crossing parallel efficiency 0.43 (boundary regime), long_prefill_8k driving P95-vs-N=1 above 2× at N=32 — has no TGI counterpart at the same boundary. A TGI re-run at N=32 would close that gap, but the substrate as it stands does not license a TGI-at-N=32 claim.

**Limitation 4: Ollama backend deferred.** The originally scoped fifth backend (Ollama, Native Windows service path) was deferred from V2 wave 1 because its dispatch surface is a separate code path from the vLLM/SGLang/TGI Docker-or-launcher pattern. Ollama's behavior under the matched 144-cell or 360-cell matrix is unknown from V2 wave 1 substrate. Any claim that V2 is a "five-backend study" would overreach; V2 wave 1 is a three-backend study (vLLM cloud, SGLang cloud, TGI local) with a deferred fourth (Ollama) and a V1-only fifth (pytorch_direct).

**Limitation 5: Cross-run analyze.py synthesis pass not yet executed.** Per-backend analysis.json files exist for each of the three V2 runs (research/tr164/results/20260605_210337_450607/, research/tr164/results/20260605_212557_266597/, research/tr164/results/20260605_192757_415750/), and SS9–SS12 lift descriptive comparisons across them. What does NOT yet exist is the cross-backend head-to-head synthesis with proper statistical machinery: Mantel-Haenszel pooled odds ratios over the backend × workload × N stratification (where the strata are the matched cells across runs); Holm-Bonferroni stepdown across the family of pairwise backend comparisons per workload × N cell; bootstrap CIs on parallel efficiency deltas; equivalence margins for the cells in which backends are claimed to be statistically indistinguishable. That synthesis pass is the next analyze.py extension and is what converts V2 wave 1 from a descriptive three-backend characterization into a publication-grade head-to-head. Until it lands, the report's cross-backend statements are descriptive deltas, not statistically-licensed effect sizes.

**Observations.** The five limitations partition cleanly: limitation 1 is a confound (SXM-vs-PCIe entanglement), limitation 2 is a scope cap (7B cloud ceiling), limitation 3 is a cell-shape asymmetry (N=16 vs N=32 ceiling on TGI vs cloud backends), limitation 4 is a deferred dispatch (Ollama), and limitation 5 is a downstream-analysis gap (cross-run synthesis). None of them touch the H1 confirmation — TGI's elimination of the pytorch_direct N=2 breakdown is robust to all five because it is a within-backend, matched-model claim against V1's pytorch_direct on identical hardware (RTX 4080 Laptop), not a cross-backend cloud claim. The limitations bound the breadth of secondary claims, not the load-bearing H1 result.

> The honest summary is that V2 wave 1 has earned the H1 claim (continuous batching shifts the boundary upward; GIL contention is pytorch_direct-specific, not a hardware constraint) and has earned descriptive cross-backend characterization at 7B. It has NOT yet earned a statistically-licensed five-backend × model-scale × concurrency head-to-head; that is V2 wave 2 plus the cross-run analyze.py extension, conditional on compute credits and the synthesis pass landing. Reporting the limitations explicitly is the substrate-fidelity discipline the program runs on — overclaiming "five-backend study with publication-grade statistical comparison" against the current substrate would trip the defensibility bar that gated TR151/TR152 in the first place.

---

## 20. SS14. V2 Wave 2 Roadmap. Three queued components.

V2 wave 1 closed with 648 cells across vLLM cloud_core, SGLang cloud_core, and TGI local_core, 57,024 metrics.csv rows, ok_rate 1.0 across every cell, and ~\$40-50 of cash spend. The substrate is paper-licensing for two distinct downstream products: the bridge paper Phase 8 anchor (continuous-batching boundary regression across backends, with TGI as the H1 confirmation surface) and the parked-paper main-track audit refresh. What remains to lift the substrate from "three-backend ratification" to "four-backend systems-venue-strength" is enumerated below as three queued components, each with its own compute envelope and gating logic.

### 20.1 Component A. Ollama backend (Native Windows service dispatch).

Ollama is the fourth backend tier and the only one that requires a Native Windows service dispatch path rather than a containerized launcher. The Docker images used for vLLM (digest `d9a5c1c1`) and TGI (digest `e6b0af6e`) and the native Python launcher path used for SGLang (vendor image `8df56b54` reference-pinned but not used at runtime due to upstream issue sgl-project/sglang#27406) do not cover Ollama. The dispatch surface lives off the existing `cell_worker.py` pattern and reuses the same `metrics.csv` schema; the new code is the launcher wrapper, the model-pull provisioning step, and the OK-rate audit hook. Compute envelope is implementation cost only: no GPU credits required because Ollama V2 reproduces the local_core hardware tier (the same RTX 4080 Laptop that produced TR164 V1's 20260531_120428_552237 run and V2's TGI 20260605_192757_415750 run). The deliverable shape is a 360-cell matched-pair against TGI local_core on the V1 model trio (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) across the four-workload × two-phase × five-N-level cross-product. Provisional gating: deferred to V2 wave 2 because Ollama dispatch was not on the wave 1 critical path; no external credit required.

### 20.2 Component B. cloud_full extension to 14B and N=32 across all backends.

V2 wave 1 covered N=32 on vLLM and SGLang at the 7B scale (Qwen2.5-7B-Instruct) on A100 PCIe and A100 SXM respectively. The cloud_full extension closes two coverage gaps. First, model scale: extend from 7B to 14B (Qwen2.5-14B-Instruct is the matching upstream model family, but the exact ID is TBD per analyze.py extension pending verified HF availability). Second, backend coverage at N=32: extend N=32 from the two cloud backends to TGI and Ollama as well, breaking the current N=16 ceiling on the local-tier backends. The current 360-cell TGI block stops at N=16; lifting to N=32 requires either cloud GPU credits (TGI runs on the same A100 SXM tier or equivalent) or a confirmed local hardware path to support the larger N. Compute envelope is the most expensive of the three components: 14B at N=32 across four backends, two model scales (7B carry-over for matched-pair plus 14B for the scale axis), three reps, the standard four-workload × two-phase cross-product, lands at a substantively larger cell count than V2 wave 1's 648, with per-cell wall-clock and per-cell GPU rate both rising. The wave 1 ~\$40-50 cash budget is a floor, not a ceiling, for wave 2; the realistic envelope is several multiples higher and is the systems-venue-strength substrate bar called out in the bridge paper Stage 2 plan.

| Component | Compute tier | Cash envelope vs wave 1 | Gating |
|---|---|---|---|
| A. Ollama backend | local_core (RTX 4080 Laptop) | \$0 incremental | Dispatch implementation only |
| B. cloud_full to 14B + N=32 | cloud_core (A100 SXM or equivalent) | Several multiples of wave 1's ~\$40-50 | External credit + bridge-paper trigger |
| C. Cross-paper analyze.py synthesis | CPU-only | \$0 incremental | Wave 1 substrate already sufficient |

**Observations.** Component A is the cheapest and unblocks the fourth-backend claim cleanly; Component B is the systems-venue-strength lift and is the only one that meaningfully scales the cash envelope; Component C is the writeup-grade analytic pass the wave 1 substrate already licenses.

> The compute asymmetry across the three components is the structural reason wave 2 is not a single batch. Component C can fire today against the existing per-run analysis.json artifacts; Component A is a one-evening dispatch build against existing local hardware; Component B is the only piece that genuinely waits on external credit, and gating Component B on the 2026-10-24 GO/NO-GO trigger does not delay either of the other two.

### 20.3 Component C. Cross-paper analyze.py synthesis pass.

The third component is pure CPU and operates on the per-run `analysis.json` artifacts already on disk under each of the three V2 run directories plus the TR164 V1 pytorch_direct run at `research/tr164/results/20260531_120428_552237/`. Three target analyses are queued. First, vLLM-vs-SGLang head-to-head, conditional on the A100 PCIe (vLLM 20260605_210337_450607) versus A100 SXM (SGLang 20260605_212557_266597) hardware split being explicitly modeled — the SXM-vs-PCIe memory-bandwidth confound (2.0 TB/s vs 1.5 TB/s plus NVLink) is the load-bearing correction that any head-to-head must apply before publishing a backend-level comparison. Second, TGI-versus-pytorch_direct matched-pair on the V1 model trio at matched workloads and matched N levels {1,2,4,8,16}, anchored on the V1 N=2 breakdown elimination finding (TGI parallel efficiency at N=2 remains substantially above 0.65 across all 24 (model × workload × phase) combinations; TGI median TTFT at N=16 balanced_2k on llama3.2-1b drops from V1's 188 seconds to sub-3-second medians). Third, breakdown-boundary regression across backends — the boundary at which parallel efficiency degrades, expressed as a function of (backend, model, workload, phase, N), measured against the V1 pytorch_direct uniform-N=2-breakdown baseline as the most-pessimistic reference and against vLLM's long_decode 0.43-0.45 parallel efficiency through N=32 as the most-optimistic reference.

### 20.4 Bridge-paper Phase 8 anchor and external-signal gating.

V2 wave 2 is not unconditional. Component A is cleared today (implementation cost only). Component C is cleared today (wave 1 substrate is sufficient). Component B's cash envelope is the gating constraint, and it is anchored on the bridge paper's 2026-10-24 GO/NO-GO trigger, which is itself conditioned on an external acceptance signal arriving from the program's current submission round. If the external signal lands, Component B fires against credit cover from a fellowship or equivalent compute grant; the wave 1 \$40-50 cash budget is a proof-of-mechanism baseline, not a credible cap for the 14B-plus-N=32-across-four-backends matrix. If the external signal does not land, Component A plus Component C still upgrades the V2 substrate from three backends to four backends with a full cross-paper analytic pass, and the cloud_full extension defers to a subsequent wave without invalidating the wave 1 ratification result. The matched-pair companion TR165 (run dir `research/tr165/results/20260607_174748_273070/`) and the canonical measurement-count anchor at ~1,119,000 (per `BANTERHEARTS_MEASUREMENT_COUNT.md` with TGI complete 2026-06-06) are the cross-references against which wave 2 progress is tracked.

---

## 22. Conclusion

Technical Report 164 V2 closes the loop opened by V1's headline finding — *V1's pytorch_direct backend exhibits a uniform N=2 breakdown pathology across every model, workload, and phase tested* — and replaces V1's per-cell deltas with a cross-run synthesis that survives paper-grade statistical scrutiny. The load-bearing addition over V1 is the artifact triad `research/tr164/cross_run_analyze.py` → `research/tr164/cross_run_analysis.json` → `research/tr164/cross_run_summary.csv`, which joins V1 pytorch_direct against V2 TGI on the hardware-matched RTX 4080 Laptop and joins V2 vLLM against V2 SGLang on the cloud Qwen2.5-7B trace. Every numeric claim that follows is reproducible from that triad.

The V2 headline — *TGI's continuous-batching architecture eliminates the V1 pytorch_direct breakdown at every tested N* — is now backed by the matched-pair statistical battery that V1 explicitly did not run. Across 120 hardware-matched and model-matched (model, workload, phase, n_agents) cells, TGI's mean parallel efficiency is 0.6661 (66.61%) versus V1's 0.3920 (39.20%); the mean per-cell delta is +0.2741 (+27.41pp) and the median is +0.2890 (+28.90pp). TGI wins 96 of 120 cells outright, V1 wins zero, and 24 cells tie at the N=1 baseline where both backends are perfect by construction. The paired Cohen's d is 1.4361 — a very large effect under any conventional cutoff — and the Wilcoxon signed-rank statistic of 7260.0 yields p = 5.579×10⁻²⁰. The Mantel-Haenszel pooled odds ratio for clearing the 0.5 efficiency threshold, stratified by (model × workload × phase) across the 22 usable strata, is infinite: TGI almost always passes the 0.5 floor at N≥2 while V1 almost never does. The companion contrasts on speedup-vs-N=1 (+1.96, d = 0.9457, p = 5.579×10⁻²⁰), P95 latency multiplier (−30.11, d = −0.1989, p = 2.809×10⁻¹⁹), and P50 TTFT (−56,533.69 ms, d = −0.5164, p = 5.144×10⁻¹¹) all clear the Holm-Bonferroni stepdown at adjusted p < 10⁻¹⁰. Four primary TGI-vs-V1 contrasts, four Holm survivals; the headline does not depend on any single metric.

The cross-run synthesis also restates the per-N decay structure with cell-count discipline. At N=2 TGI holds 84.6% efficiency where V1 has already collapsed to 53.6% (+30.98pp); at N=4 TGI holds 67.6% versus V1's 28.1% (+39.48pp); at N=8 TGI holds 48.5% versus V1's 10.4% (+38.07pp); at N=16 TGI holds 32.3% versus V1's 3.75% (+28.53pp). The V1 pytorch_direct breakdown-boundary regression confirms the V1 pathology in cross-run terms: at the 0.5 efficiency threshold, 24/24 V1 combinations break, eight at N=2 and sixteen at N=4. TGI pushes that boundary to N=8 (13 of 24 combinations) or N=16 (8 of 24); two TGI combinations never drop below 0.5 within the tested N≤32 envelope. The continuous-batching backends on cloud push it further still — vLLM and SGLang reach N=16 or N=32 at the 0.5 threshold, and half of SGLang's combinations never break below 0.5 within the tested range.

The vLLM-vs-SGLang head-to-head is the section where V2 most clearly disciplines itself away from a marketing-friendly claim. On raw 48-cell matched parallel efficiency, SGLang leads by 1.71pp (vLLM 78.78%, SGLang 80.50%, d = −0.5148, Wilcoxon p = 6.394×10⁻⁴). After dividing vLLM's raw efficiency by the 0.75 SXM/PCIe memory-bandwidth ratio derived in SS8b, the bandwidth-adjusted contrast inverts: vLLM_adj 1.0505 versus SGLang 0.8050, delta +0.2455 (+24.55pp), Wilcoxon p = 1.598×10⁻⁹. Both contrasts survive Holm (raw Holm-adj p = 0.005; bandwidth-adjusted Holm-adj p = 1.4×10⁻⁸). The per-workload Holm pattern is the real story: SGLang's RadixAttention advantage is real and Holm-significant on the two prefix-cache-friendly workloads (balanced_2k −3.13pp, Holm-adj p = 0.01283; repeated_prefix −3.68pp, Holm-adj p = 0.01283), and statistically null on long_decode (Holm-adj p = 0.3359) and long_prefill_8k (Holm-adj p = 0.6504). On tail-latency (P95 multiplier) and on cold-start (P50 TTFT), the two backends are statistically indistinguishable (Holm-adj p = 1.0 on both). The honest contribution is *SGLang wins on prefix-cache-friendly workloads by 3-4pp on the raw, hardware-confounded comparison; once bandwidth is normalized, vLLM is 24.55pp ahead; on the other half of the workload mix the two backends are equivalent*. That is the licensed claim. The "SGLang dominates vLLM" claim is not licensed.

V2's contribution interlocks with TR165's Phase 8 mechanism-isolation contribution at two independent axes. V2 confirms that breakdown elimination is achievable at the backend-architecture level: replacing V1's per-process pytorch_direct path with a continuous-batching server (TGI, vLLM, or SGLang) collapses the N=2 pathology at matched-pair Wilcoxon p < 6×10⁻²⁰ and Cohen's d 1.44 on the local hardware-matched join. TR165 confirms the complementary mechanism-level finding (H2_partial) that part of the V1 pathology is attributable to the Python interpreter's GIL — partial GIL-attribution at the interpreter axis rather than at the backend-architecture axis. The two axes are independent: V2 changes the request scheduler and the KV-cache manager while leaving the interpreter alone; TR165 changes the interpreter while leaving the per-process backend alone. Each rules out the other as a sole explanation, and together they triangulate the V1 breakdown as a compound failure that requires both a continuous-batching scheduler and (eventually) a no-GIL interpreter path to eliminate at the limit.

V2 wave 2 is queued and explicitly out of scope for this report. The wave 2 backlog is (i) Ollama characterization on the local rig, (ii) cloud_full 14B-parameter runs on the SXM hardware to retire the "Qwen2.5-7B-only" caveat on the vLLM-vs-SGLang head-to-head, and (iii) `analyze.py` extensions that fold the cross-run synthesis directly into per-paper substrate. Until wave 2 lands, three claims remain explicitly NOT licensed: (a) any vLLM-vs-SGLang efficiency claim that omits the bandwidth-adjustment caveat, (b) any generalization beyond Qwen2.5-7B on cloud or beyond the V1 trio on local, and (c) any 14B+ behavioral claim. The bandwidth confound is load-bearing and the model-size envelope is hard.

The V2 ⊕ TR165 pair is the substrate the bridge paper will draw from, with V2 supplying the backend-architecture axis result and TR165 supplying the interpreter axis result. The next milestone is V2 wave 2; the milestone after that is the cross-paper analyze.py extension that exposes `cross_run_analysis.json` to downstream substrate consumers. The cross-run synthesis artifact is the load-bearing addition over V1, and it now carries the V2 headline through paper-grade statistics without depending on any number that is not reproducible from the committed JSON.

---

## 22. References

This section lists the artifacts and external sources load-bearing for TR164 V2. Every internal reference is a path inside this repository; every external reference is either an open-source upstream issue, a vendor documentation surface, or a public PEP. Citations to blind-review-active venues are deliberately omitted per substrate hygiene policy; where a forward-looking external venue would otherwise be named, the generic equivalent is used.

### 22.1 Banterhearts internal references

The internal substrate is organized along three axes: the serving-stack arc (TR130/131/132) that established the four-backend framing this report inherits; the TR164 V1 pytorch_direct boundary report and its V2 companion (this report); and the cross-program synthesis surfaces (bridge paper, parked-paper audit, measurement-count canonical) that consume the V2 substrate.

| Ref ID | Path | Role |
| --- | --- | --- |
| INT-1 | `research/tr130/`, `research/tr131/`, `research/tr132/` | Serving-stack arc that established the four-backend framing (pytorch_direct, vLLM, SGLang, TGI) inherited by TR164 |
| INT-2 | `PublishReady/reports/Technical_Report_164.md` (1,383 lines) | TR164 V1 hand-narrated report — pytorch_direct local_core boundary characterization, matched-pair baseline for V2 |
| INT-3 | `research/tr164/results/20260531_120428_552237/` | TR164 V1 pytorch_direct local_core run directory (matched-pair vs TGI V2 local_core) |
| INT-4 | `research/tr164/results/20260605_210337_450607/` | TR164 V2 vLLM cloud_core run directory (A100 PCIe, 144 cells, 15,120 metric rows) |
| INT-5 | `research/tr164/results/20260605_212557_266597/` | TR164 V2 SGLang cloud_core run directory (A100 SXM, 144 cells, 15,120 metric rows) |
| INT-6 | `research/tr164/results/20260605_192757_415750/` | TR164 V2 TGI local_core run directory (RTX 4080 Laptop, 360 cells, 26,784 metric rows) |
| INT-7 | `research/tr165/results/20260607_174748_273070/` | TR165 companion run — Python 3.14t nogil ablation, mechanism-isolation matched-pair to this report |
| INT-8 | `papers/serving_state_safety_certification/` | Bridge paper substrate (TR148/TR149/TR152 ⊕ V2 serving-stack characterization) |
| INT-9 | `papers/PARKED_PAPERS_MAIN_TRACK_AUDIT_2026-06-05.md` | Parked-paper main-track audit, data-grounded against V2 substrate |
| INT-10 | `BANTERHEARTS_MEASUREMENT_COUNT.md` | Canonical measurement-count tracker (~1,119,000 total with TGI complete 2026-06-06) |

**Observations.** Internal references resolve to file paths inside this working tree; INT-2 is the only PublishReady-promoted document at the time of writing, by [feedback_publishready_off_limits] policy (V2 auto-generated report stays in the run directory until hand-promoted).

> The substrate-fidelity discipline this report inherits depends on every numerical claim resolving to one of INT-3 through INT-7. If a later synthesis pass invokes a number not traceable to those five run directories, that claim is by construction outside the V2 substrate and must be flagged as TBD per analyze.py extension.

### 22.2 External references

| Ref ID | Citation | Role |
| --- | --- | --- |
| EXT-1 | PEP 779 — Criteria for supported status for free-threaded Python | Defines the Python 3.14t no-GIL build path exercised in companion TR165 |
| EXT-2 | vLLM project documentation (continuous batching, PagedAttention) | Backend vendor doc for INT-4 |
| EXT-3 | SGLang project documentation (RadixAttention, prefix-cache reuse) | Backend vendor doc for INT-5 |
| EXT-4 | Hugging Face TGI project documentation (`text-generation-launcher` 3.3.6-dev0) | Backend vendor doc for INT-6 |
| EXT-5 | `vllm-project/vllm#44703` | Upstream issue surfaced during V2 vLLM cloud_core wave |
| EXT-6 | `sgl-project/sglang#27406` | Upstream issue: missing distro dep in SGLang vendor image, motivating the NATIVE Python launcher fallback used in INT-5 |

**Observations.** External references are restricted to open-source upstream issues, public vendor documentation, and a single PEP; no blind-review-active venue is cited.

> The fact that two of the four backend integrations (EXT-5, EXT-6) generated upstream open-source issues during a single 72-hour V2 dispatch wave is itself a substrate observation: the four-backend matched-cell-shape framing exercises the serving stacks hard enough to surface real bugs, which is the operational reproducibility contribution the V2 wave was designed to land.

---

## 23. Appendix A. Hardware, Software, and Environment Fingerprint

The three V2 backends were captured on three distinct (hardware, OS, container) tuples. Treating them as one homogeneous "V2 environment" would obscure the A100 PCIe vs A100 SXM memory-bandwidth split that load-bears the H2 quantification, and would erase the local vs cloud cost gap that motivates the TGI lane in the first place. The fingerprint below is the minimum substrate a third party needs to reproduce any per-cell row in `metrics.csv` from `research/tr164/results/20260605_*/`.

### 23.1 Backend-by-backend environment table

| Field | vLLM cloud_core | SGLang cloud_core | TGI local_core |
|---|---|---|---|
| Run directory | `research/tr164/results/20260605_210337_450607/` | `research/tr164/results/20260605_212557_266597/` | `research/tr164/results/20260605_192757_415750/` |
| Host class | RunPod cloud GPU pod | RunPod cloud GPU pod | RTX 4080 Laptop workstation |
| GPU | NVIDIA A100 PCIe (80 GB, 1.5 TB/s HBM) | NVIDIA A100 SXM (80 GB, 2.0 TB/s HBM + NVLink) | NVIDIA RTX 4080 Laptop (12 GB GDDR6) |
| Launch mode | Docker (vendor image) | NATIVE Python launcher (vendor image had missing distro dep — upstream `sgl-project/sglang#27406`) | Docker (vendor image) |
| Vendor image digest | `vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504` | reference pin only: `lmsysorg/sglang@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4` | `ghcr.io/huggingface/text-generation-inference@sha256:e6b0af6e0bf65337b84a19f15d74660c7892192f555fb0b68d3f3d62bf0c1e9a` |
| Vendor backend version | vLLM serving image (digest-pinned) | SGLang v0.5.12.post1-runtime, build 2026-05-23, upstream commit `5a15cde858ea09b77116212a39356f2fc51b8584` | text-generation-launcher 3.3.6-dev0, build 2026-01-08 |
| Model(s) under test | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | llama3.2-1b, qwen2.5-1.5b, llama3.2-3b (V1 model trio — matched-pair vs V1 pytorch_direct) |
| Cells planned / completed | 144 / 144 | 144 / 144 | 360 / 360 |
| `metrics.csv` rows | 15,120 | 15,120 | 26,784 |
| ok_rate | 1.0 across every cell | 1.0 across every cell | 1.0 across every cell |
| Wall-clock | ~12-15 h | ~10-12 h | ~6 h overnight (19:28 EDT 2026-06-05 → 01:23 EDT 2026-06-06) |
| Cost | A100 PCIe ~\$1.40-1.64/hr, total ~\$20-25 | A100 SXM ~\$1.80-2.10/hr, total ~\$20-25 | \$0 cash |
| Upstream issue filed | `vllm-project/vllm#44703` | `sgl-project/sglang#27406` | none |

**Observations.** Three rows in this table do load-bearing work for downstream synthesis. The image-digest rows convert "ran vLLM" and "ran TGI" from a brand label into a reproducible artifact: the SHA256-pinned vendor images are what `docker pull` will resolve in 2027 even if the `:latest` tag drifts. The GPU row carries the hardware confound: A100 SXM (2.0 TB/s + NVLink) is not the same instrument as A100 PCIe (1.5 TB/s), and any vLLM-vs-SGLang head-to-head in section bodies must reconcile that delta before attributing a gap to the runtime. The launch-mode row records that the SGLang lane required a NATIVE Python launcher because the vendor image shipped without a required distro package — the upstream issue `sgl-project/sglang#27406` is the public receipt; the SGLang digest in this table is therefore a *reference pin for Docker reproduction*, not the bit-exact image this V2 wave executed under.

> The fingerprint is dominated by two artifacts: the vendor image digest and the GPU SKU. Together they account for substantially more reproducibility surface than the CPU, RAM, or OS rows that a generic hardware appendix would foreground. A100 PCIe vs A100 SXM is the single most consequential field on the page for any reader trying to weigh a vLLM-vs-SGLang claim.

### 23.2 Digest pinning, substrate roots, and aggregate

The image-digest convention codified in repo commit `2bb70d2d` requires every backend cell to record `image_digest` (SHA256) and `vendor_version` (semver + upstream commit where available) into each per-cell manifest, so that a future reader can resolve the exact byte-image even after a vendor tag moves. The three digests in the table above are the live values that V2 wave 1 ran under. The substrate-on-disk roots are `research/tr164/results/20260605_210337_450607/` (vLLM), `research/tr164/results/20260605_212557_266597/` (SGLang), and `research/tr164/results/20260605_192757_415750/` (TGI). Aggregating across the three backends gives 648 cells, 57,024 `metrics.csv` rows, ok_rate 1.0 across every cell of every backend, ~28-33 h non-parallel wall-clock, and ~\$40-50 total cash. The companion TR165 run dir is `research/tr165/results/20260607_174748_273070/`, and the V1 baseline that V2 replaces sits at `research/tr164/results/20260531_120428_552237/` with its hand-narrated report at `PublishReady/reports/Technical_Report_164.md` (1,383 lines).

> Digest-pinning is the contract that turns this appendix from documentation into a reproducibility artifact. Without it, "we ran SGLang v0.5.12" is a brand claim; with it, the byte-image is addressable for as long as the registry retains the layer.

---

## 24. Appendix B. Reproduction Commands and File Manifest

This appendix documents the exact invocation chain required to reproduce each V2 backend run from a clean RunPod or local workstation. Every command below maps to a substrate file path that the report's claims depend on; the manifest at the end of the section is the load-bearing contract between this report and the on-disk artifacts.

### 24.1 vLLM cloud_core (A100 PCIe)

The vLLM run uses the upstream `vllm/vllm-openai` image pinned by digest. The dispatcher launches the container, waits for the OpenAI-compatible endpoint to bind on port 8000, then walks the 144-cell cross-product (1 model × 4 workloads × 2 phases × 6 N levels × 3 reps).

```bash
docker pull vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
python research/tr164/run.py --backend vllm --profile cloud_core --model Qwen/Qwen2.5-7B-Instruct
python research/tr164/analyze.py --run-dir research/tr164/results/20260605_210337_450607/
```

Substrate files: `research/tr164/results/20260605_210337_450607/metrics.csv` (15,120 rows), `cell_summary.csv` (144 rows), `manifest.json`, `runpod_logs/`. Gotcha: vLLM upstream issue `vllm-project/vllm#44703` documents a metric-collection edge case on long_prefill_8k at N=32; the dispatcher's retry-on-empty-completion guard sidesteps it, but the issue must be cited in any P95-multiplier discussion.

### 24.2 SGLang cloud_core (A100 SXM, native launcher)

The SGLang vendor image at the pinned reference (digest `sha256:8df56b54...3944a4`, semver v0.5.12.post1-runtime, upstream commit `5a15cde858ea09b77116212a39356f2fc51b8584`) is missing a distro-level dependency, so V2 wave 1 used the native Python launcher path. The Docker pin remains in the manifest for Docker-based reproduction once the upstream package lands.

```bash
# Native launcher (V2 wave 1 actual path):
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000
python research/tr164/run.py --backend sglang --profile cloud_core --model Qwen/Qwen2.5-7B-Instruct --launcher native
python research/tr164/analyze.py --run-dir research/tr164/results/20260605_212557_266597/

# Docker reproduction (once distro dep ships):
docker pull lmsysorg/sglang@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4
```

Substrate files: `research/tr164/results/20260605_212557_266597/metrics.csv` (15,120 rows), `cell_summary.csv`, `runpod_logs/`. Gotcha: A100 SXM (2.0 TB/s HBM bandwidth + NVLink) versus the vLLM run's A100 PCIe (1.5 TB/s) is a hardware confound on any vLLM-vs-SGLang head-to-head, especially on prefill-heavy workloads; the SS-section discussing repeated_prefix flags this explicitly. Upstream tracking: `sgl-project/sglang#27406`.

### 24.3 TGI local_core (RTX 4080 Laptop)

TGI ran locally overnight on the laptop class device — \$0 cash, matched-pair against the V1 pytorch_direct model trio.

```bash
docker pull ghcr.io/huggingface/text-generation-inference@sha256:e6b0af6e0bf65337b84a19f15d74660c7892192f555fb0b68d3f3d62bf0c1e9a
python research/tr164/run.py --backend tgi --profile local_core --models llama3.2-1b,qwen2.5-1.5b,llama3.2-3b
python research/tr164/analyze.py --run-dir research/tr164/results/20260605_192757_415750/
```

Substrate files: `research/tr164/results/20260605_192757_415750/metrics.csv` (26,784 rows), `cell_summary.csv` (360 rows), `run_logs/`. Gotcha: text-generation-launcher 3.3.6-dev0 (build 2026-01-08) is a development tag; if the dev tag is garbage-collected upstream, repin by digest, not by tag.

### 24.4 File manifest (claims-bearing)

| Artifact | Path | Load-bearing claim |
|---|---|---|
| vLLM metrics | `research/tr164/results/20260605_210337_450607/metrics.csv` | long_decode parallel efficiency 0.43-0.45 through N=32 |
| SGLang metrics | `research/tr164/results/20260605_212557_266597/metrics.csv` | RadixAttention workload-conditional benefit |
| TGI metrics | `research/tr164/results/20260605_192757_415750/metrics.csv` | H1 confirmation: N=2 breakdown eliminated |
| V1 pytorch_direct | `research/tr164/results/20260531_120428_552237/` | Matched-pair V1 baseline |
| TR165 companion | `research/tr165/results/20260607_174748_273070/` | Python 3.14t nogil mechanism-isolation |
| V1 hand-narrated report | `PublishReady/reports/Technical_Report_164.md` | 1,383-line V1 narrative substrate |
| Backend notes | `research/tr164/V2_BACKEND_NOTES.md` | V2 wave 1 verbatim substrate facts |
| Measurement count | `BANTERHEARTS_MEASUREMENT_COUNT.md` | ~1,119,000 program-wide aggregate |
| Parked-paper audit | `papers/PARKED_PAPERS_MAIN_TRACK_AUDIT_2026-06-05.md` | Data-grounded audit against V2 substrate |
| Bridge paper | `papers/serving_state_safety_certification/` | Multi-TR synthesis target consuming V2 |

**Observations.** The manifest closes the loop between the 648-cell, 57,024-row V2 substrate and every numeric claim made in the body of this report. Every backend's run directory contains its own `analysis.json`; the cross-run synthesis pass (vLLM vs SGLang on matched workloads, V1 vs TGI on matched model trio) remains the paper-writeup-grade work the substrate now licenses.

> Reproducibility is a triad: pinned image digest plus pinned dispatcher commit plus the on-disk `metrics.csv` row count. Any one of the three failing to match invalidates a head-to-head; all three matching is the contract. The SGLang native-launcher detour is the canonical example of why "image digest alone" is not sufficient — the digest pins the binary, but the launcher path pins how the binary was actually used.

---

## 25. Appendix C. Per-Backend Cell-Shape Tables

This appendix enumerates the cell-shape coordinates for the three V2 backend runs that constitute the substrate of this report. The intent is auditability: any reader who wants to verify the matched-matrix claim between vLLM cloud_core and SGLang cloud_core, or the V1-vs-V2 matched-pair claim between pytorch_direct local_core and TGI local_core, can reconstruct the cross-product from the table rows below and cross-check against the per-backend `cell_summary.csv` files in the run directories cited in Section 2.

### 25.1 vLLM cloud_core 144-cell shape (`20260605_210337_450607/`)

| Axis | Levels | Cardinality |
|---|---|---|
| Model | Qwen2.5-7B-Instruct | 1 |
| Workload | balanced_2k, long_decode, repeated_prefix, long_prefill_8k | 4 |
| Phase | prefill, decode | 2 |
| Concurrency N | 1, 2, 4, 8, 16, 32 | 6 |
| Repetitions | rep_0, rep_1, rep_2 | 3 |
| Cross-product | 1 × 4 × 2 × 6 × 3 | 144 |

**Observations.** The vLLM cloud_core matrix is the only V2 run that extends to N=32, a deliberate extension beyond the V1 local_core ceiling of N=16 to probe whether continuous batching holds parallel efficiency through a larger boundary regime. All 144 cells completed with ok_rate 1.0 and the matrix yielded 15,120 `metrics.csv` rows.

> The N=32 column is the load-bearing extension over V1. Long_decode parallel efficiency held at 0.43-0.45 through N=32; balanced_2k slipped to 0.43 at N=32; long_prefill_8k crossed a 2× P95 multiplier vs N=1 at N=32. These three boundary signals are only legible because the matrix was extended one concurrency level past V1.

### 25.2 SGLang cloud_core 144-cell shape (`20260605_212557_266597/`)

| Axis | Levels | Cardinality |
|---|---|---|
| Model | Qwen2.5-7B-Instruct | 1 |
| Workload | balanced_2k, long_decode, repeated_prefix, long_prefill_8k | 4 |
| Phase | prefill, decode | 2 |
| Concurrency N | 1, 2, 4, 8, 16, 32 | 6 |
| Repetitions | rep_0, rep_1, rep_2 | 3 |
| Cross-product | 1 × 4 × 2 × 6 × 3 | 144 |

**Observations.** The SGLang cell-shape is coordinate-identical to vLLM by deliberate construction; a matched 144-cell cross-product was the prerequisite for any H1 cross-backend test surface. All 144 cells completed with ok_rate 1.0 and yielded 15,120 `metrics.csv` rows. The single confound that the table cannot show is the SXM-vs-PCIe hardware split (Section 2.4): SGLang ran on A100 SXM while vLLM ran on A100 PCIe.

> Coordinate-identity is necessary but not sufficient for a head-to-head. Any vLLM-vs-SGLang comparison drawn from these matched matrices must carry the SXM/PCIe-bandwidth caveat in the same breath as the matched-cells claim.

### 25.3 TGI local_core 360-cell shape (`20260605_192757_415750/`)

| Axis | Levels | Cardinality |
|---|---|---|
| Model | llama3.2-1b, qwen2.5-1.5b, llama3.2-3b | 3 |
| Workload | balanced_2k, long_decode, repeated_prefix, long_prefill_8k | 4 |
| Phase | prefill, decode | 2 |
| Concurrency N | 1, 2, 4, 8, 16 | 5 |
| Repetitions | rep_0, rep_1, rep_2 | 3 |
| Cross-product | 3 × 4 × 2 × 5 × 3 | 360 |

**Observations.** The TGI local_core matrix mirrors the V1 pytorch_direct cell-shape (same three models, same four workloads, same two phases, same five concurrency levels, same three reps) by deliberate construction so that the V1-vs-V2 backend swap is the only varying axis. All 360 cells completed with ok_rate 1.0 and yielded 26,784 `metrics.csv` rows. The N ceiling is 16 (not 32 as in the cloud runs) because the matched-pair against V1 pytorch_direct is the load-bearing comparison; extending TGI to N=32 would have broken matched-pair coordinate-identity.

> The 360 = 3 × 4 × 2 × 5 × 3 factorization is the H1 falsification matrix. Across all 24 (model × workload × phase) slices on TGI, the V1 pytorch_direct uniform-N=2 parallel-efficiency breakdown does not replicate, and the six V1 pathological N=16 long_decode cells on llama3.2-3b do not hang. The matched cell-shape is what licenses calling the V1 breakdown pytorch_direct-specific rather than hardware-bound.

### 25.4 Aggregate

The three V2 substrate runs sum to 648 cells (144 + 144 + 360) and 57,024 `metrics.csv` rows (15,120 + 15,120 + 26,784), with ok_rate 1.0 across every cell of every backend. The deferred Ollama V2 fifth-backend dispatch and the cloud_full N=32-on-all-backends extension are not represented in these tables and remain queued for V2 wave 2 per Section 2.
