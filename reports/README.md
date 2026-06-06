# Banterhearts Technical Reports
## LLM Inference Research & Safety Alignment — ~1,119,000 Primary + Judge Measurements (TR164 V2 vLLM+SGLang+TGI all complete 2026-06-06), 46+ Technical Reports

**Latest (2026-06-05):** **TR164 V1 — Serving-Stack Physics on Consumer GPU** shipped — `Technical_Report_164.md` (1,383 lines, full-depth with Appendices A-M) — closes the local_core leg of the SGLang + precision-controlled serving-stack expansion of the TR130-132 line. 346 real cells across 3 model tiers (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b) × 2 phases (scaling, ttft) × 4 workloads (short_decode, balanced_2k, long_decode, repeated_prefix) × 5 concurrency tiers (N=1/2/4/8/16) × 3 reps, plus 14 skip-marker rows documenting six deterministic-hang cell shapes on llama3.2-3b at N=16. **Headline finding: parallel efficiency on the in-process pytorch_direct backend breaks down at N=2 uniformly across all 24 (model × workload × phase) combinations** — median efficiency 1.000 → 0.547 → 0.295 → 0.127 → 0.056 at N=1/2/4/8/16, P95 latency multiplier up to **1446.67× at N=16**, TTFT on llama3.2-1b N=16 balanced_2k at **188.4 seconds** for median request. **Six (phase × workload × N=16) cell shapes on llama3.2-3b proved deterministically pathological** — the dispatcher hung for ≥3.3 hours with GPU at 99% SM-occupancy but thermal/power readings consistent with GIL-starved idle (50-54°C, 23-49 W on a part rated 80-150 W sustained). Mechanism hypothesis: Python GIL serialization of in-process N-way concurrent inference dispatch, supported by (a) the thermal/power signature, (b) clean recovery via subprocess timeout + in-process fallback at lower N, (c) cross-model asymmetry where **qwen2.5-1.5b retains 0.92 mean speedup at N=16 while llama3.2-3b retains 0.03** on balanced_2k — consistent with model-architecture differences in GIL-release frequency. Substrate: 21,159 metric rows + 14 clean nsys-rep kernel captures (6.12 GB) + 3 orphan .qdstrm files (2.61 GB, NVIDIA Nsight Systems 2025.5.1 EventCollection bug). 7-generation dispatcher arc V1→V7 spanning 5d 13h 48m wall-clock (2026-05-31 12:04 → 2026-06-05 01:52). Methodological contributions: nsys-wrapped subprocess pattern (cell_worker.py), subprocess-timeout-and-fallback recovery, skip-row append protocol, capture_environment() Windows-WMI defensive guard. V2 cross-backend RunPod fan-out (vLLM, SGLang, Ollama, optionally TGI) + Python 3.14t nogil ablation are the mechanism-isolation experiments queued for the next compute window. Previous: **2026-05-28** **Conclusive Phase 6 synthesis trilogy** shipped — `Conclusive_Phase6.md` (1,453-line main synthesis), `Conclusive_Phase6_Extended_Appendices.md` (1,585-line per-TR appendix volume: Appendices A–G + cross-cutting X1–X4 + glossary), and `Conclusive_Phase6_Whitepaper.md` (408-line external decision whitepaper including the 5-layer protocol table, Qwen 2.5 1.5B worked example, and 8-objection reviewer FAQ). Consolidates seven executed reports — **TR144** (Speculative Decoding × Safety / TAIS), **TR145** (FP8 KV-cache single-config), **TR146** (mechanistic-probe falsification), **TR147** (Compile Reproducibility Index / CRI), **TR148** (Judge Triangulation Protocol / JTP), **TR149** (FP8 standardized batteries), **TR152** (FP8 serving-state factorial) — into a single **five-layer serving-state safety certification protocol** (Layer 1 JTP / Layer 2 RTSI + TAIS / Layer 3 CRI / Layer 4 scale validity / Layer 5 serving-state validity). Headline deliverable: a **serving-state-independent FP8 (E4M3) KV-cache certificate** for 1B–4B instruction-tuned models on vLLM v0.19.1 — harmful-refusal-neutral across batch ∈ {1, 8, 32}, prefix-caching {on, off}, and temperature ∈ {0.0, 0.7, 1.0} (TR152: **0 / 8,976 harmful-battery pairs discordant**; FP8-interaction spread 2.99 pp inside the ±3 pp band), with one located footnote — a sub-1 pp Qwen-2.5 over-refusal lean on the XSTest benign-edge battery (per-family Mantel–Haenszel OR 3.878 [2.386, 6.302]). The negative control (TR146) is load-bearing: four mechanistic probes all fail to distinguish safe from dangerous configs, so the entire arc stays behavioural — safety cannot be read off the weights. Claim ladder **C1–C10** (6 Supported / 2 Licensed / 2 Forbidden); the whole executed substrate ran at **$0 external API cost** under the umbrella gate on a local Ollama judge cohort. Cloud-gated envelope widenings (TR150 long-context, TR151 7B–72B scale, TR153 KV-method sweep) are gated on the 2026-10-24 GO/NO-GO trigger. The trilogy's most recent constituent report is TR152 v2 (Serving-State Safety Factorial — **canonical v2 local expansion**), shipped to `Technical_Report_152.md` (**1,341 lines full-depth narration with Appendices A-D**, supersedes the v1 pilot's 348 lines — matches TR148 / TR149 gold-standard template). SS14 tightened 2026-05-28 with v2's per-battery cross-judge κ recomputed directly from the sign-aware `per_judge_outcome` substrate (XSTest gemma↔llama κ = +0.7989 robust by JTP; harmful batteries degenerate-via-chance-correction-not-disagreement; pooled κ sanity-matches the analysis JSON exactly at 0.8310 / 0.0622 / 0.0962). Bridge-paper Layer 5: folds the FP8 KV-cache, batch, prefix-caching, speculative-decoding, and temperature axes into one `fp8_anchored_star` factorial (12 of 14 cells per model runnable; the 2 speculative-decoding cells fail uniformly on vLLM v0.19.1 argparse, not OOM — **v1's "cloud-gated by VRAM" attribution retracted**). **45,000 records, 20,754 matched FP16-vs-FP8 pairs across 5 models × 4 batteries (3 families)**, with XSTest uncapped to 450 / cell. Verdict: FP8 is **refusal-neutral on harmful prompts across the whole factorial** — HarmBench / JBB / StrongREJECT perfectly concordant (0 / 8,976 discordant pairs across 5 models × 6 contexts × 3 batteries), 0 / 120 cells Holm-significant, **117 / 120 TOST-equivalent at ±3 pp (97.5%, up from v1's 83.3%)**. The only footprint is a sub-1 pp over-refusal lean on the **Qwen family's XSTest** — **Mantel-Haenszel pooled OR 1.88 [1.32, 2.69] on 133 discordant pairs, sign-test p ≈ 0.0004**, refining v1's barely-clearing 2.69 [1.09, 6.62] on 23 pairs by 5.8 × the discordant base and tightening the CI by 4 ×. phi3-mini-4k and llama3.2-1b (both new in v2) pattern with the Llama-3.2-3b family-side: clean on XSTest, FP8 marginally improves their over-refusal behavior. The family signal localizes to Qwen 2.5 (both variants), amplified mildly by temperature on the 1.5 B variant. Run local-only under `--skip-openai-judge`, $0 external cost, 28.7 h end-to-end on RTX 4080 Laptop 12 GB. Previous: **2026-05-22** TR152 v1 (pilot, 348 lines — now retired in favor of v2 narration); **2026-05-14** TR149 (Standardized Safety Battery) → `Technical_Report_149.md` (1,303 lines) — FP8 KV-cache null replicates on the four standardized batteries (cross-battery MH pooled OR 0.8065 [0.38, 1.70], 0 / 12 Holm-significant, 12 / 12 TOST-equivalent); and **2026-05-13** TR148 v2 (Multi-Judge Reliability) → `Technical_Report_148.md` (1,556 lines) — triangulate verdict κ = 0.6917, dual-axis methodology finding. TR152 v2 judging completed 2026-05-28 00:06 ET — 135,000 judge-label rows now on disk, locking the program total at ~1,041,000 primary + judge measurements (see `BANTERHEARTS_MEASUREMENT_COUNT.md` 2026-05-28 supplement for verified row counts).

This directory contains the complete research program documenting LLM inference performance, optimization, multi-agent orchestration, cross-language analysis, deployment policy, safety alignment under inference optimizations, mechanistic safety probing, and **benchmarking-integrity reproducibility** — spanning consumer hardware (NVIDIA RTX 4080 Laptop GPU, 12 GB VRAM), cloud GPUs (Colab T4 16 GB, RunPod RTX 6000 Ada 48 GB, A100 80 GB, L40S 48 GB), Docker-based quantization pipelines (AWQ, GPTQ), and cross-version software-stack ablations (Triton 3.3.1 / 3.4.0 / 3.6.0).

---

## Featured Finding — TR147 Triton Stack Attribution (the "Kill-Shot")

A `torch.compile` benchmark on the *same GPU*, *same model*, and *same Python code* produces **opposite conclusions** depending on which Triton minor version is installed. This is documented in TR147 v4 with 10,800 rows of direct ablation (Triton 3.3.1 → 3.4.0 → 3.6.0 on RTX 6000 Ada, Qwen 2.5 1.5B FP16):

| Triton | Prefill Δ vs eager (reduce-overhead) | Decode crash rate |
|---|---|---|
| **3.3.1** | **−77.2%** (fast) | **80%** (unstable) |
| 3.4.0 | +0.5% (neutral) | 0% (stable) |
| 3.6.0 | −1.6% (neutral) | 0% (stable) |

Cross-version Cohen's d on the compile-vs-eager prefill contrast: **15 to 49** (|d|>10 indicates no distributional overlap). The eager arm is flat across versions (spread ≤2%, |d|≤0.15), confirming the variance lives entirely on the compiled path — not on the harness.

**Attribution (decoupled):**
- **Decode crash fix** → `pytorch/pytorch` PR #175562 / Issue #175557 (assertion relaxation in `dealloc_current_path_weakrefs()`, drafted during this research program). Stability fix; does not alter codegen.
- **Prefill speedup loss** → `triton-lang/triton` v3.4.0 codegen regression, documented in PR #7138 (LLVM + PTXAS register-spilling interaction) and compounded by PRs #6877 / #6694 / #6407 (warp-specialization register reallocation) and #6982 (generic swizzling rewrite for `convert_layout`). Codegen regression; does not touch stability.

These are **two independent upstream changes** by different authors that happen to land at the same software-stack boundary. The resulting phenomenon invalidates any published `torch.compile` benchmark number that does not report its Triton minor version as part of the benchmark identity — which, at the time of writing, includes virtually all of them. TR147 proposes a five-gate protocol for reporting compiled-path benchmarks: GPU sm_N, Triton minor version, PyTorch build, cache type (Dynamic/Static), and compile mode must all be explicit for a claim to be reproducible.

Full detail: [Technical_Report_147.md](Technical_Report_147.md), SS8.1–SS8.7.

---

## Research Program Overview

### Phase 1: Foundation (TR108-TR122)
**15 technical reports. ~8,000 benchmark runs.**

Single-agent and multi-agent performance analysis, Rust vs Python cross-language evaluation, ONNX/TensorRT backend exploration, cost/energy analysis, the compile paradox root-cause audit, model scaling study, and resource profiling.

### Phase 3: Deployment Framework (TR123-TR133)
**11 technical reports. ~105,945 primary measurements (per `BANTERHEARTS_MEASUREMENT_COUNT.md`).**

KV-cache economics, quality baselines, quantization decision matrix, Linux/Triton compile validation, long-context characterization, production workload analysis, N-agent scaling laws, serving stack comparison, GPU kernel profiling (host + in-container), and a predictive capacity planner shipped as the `chimeraforge plan` CLI.

### Phase 4: Safety Core (TR134-TR137) + Phase 5: Attack Surface (TR138-TR143, TR146) + v2/v3 Expansions
**Phase 4:** TR134–TR137 (~74,254; TR137 synthesis IS TR134+135+136 — do not double-count). **Phase 5:** TR138–TR143 synthesized in Conclusive_Phase5 (306,996 evaluated samples). Plus v2/v3 AWQ/GPTQ expansions.

Alignment robustness under quantization, multi-agent concurrency safety, cross-backend safety consistency, the safety tax synthesis, batch inference safety under non-determinism (+ strengthened-evidence revision), multi-turn jailbreak susceptibility under quantization, many-shot and long-context jailbreak, cross-architecture refusal fragility (largest study: 18 models, 10+ families, 127,224 records v3.1), quality-safety correlation, cross-request safety leakage under continuous batching (TR143), and mechanistic safety probing under quantization (TR146: safety neurons absorb 1.39x disproportionate quantization error).

The v3 expansion (TR125 v3, TR134 v3, TR142 v3) extends the evaluation matrix from GGUF k-quants to include AWQ and GPTQ 4-bit formats across 6 models, adding 11 new model-quant cells, 18,568 AWQ/GPTQ primary samples, and 5,148 AWQ/GPTQ judge annotations. A second-judge robustness check (Claude Sonnet 4, 11,470 rows) validates consistency across the full matrix.

### Phase 5: Benchmarking Integrity (TR147)
**1 technical report, 4 sub-versions. 62,280 measurements.**

Cross-GPU, cross-cache, cross-compile-mode, cross-Triton-version reproducibility ablation on `torch.compile`. V1 established the portability regression on RTX 6000 Ada; V2 hardened it with phase separation and StaticCache; V3 extended to A100 80 GB (where compiled decode crashes 100% at every token length tested); V4 closed the reviewer kill-shot with a 10,800-row same-GPU Triton-version ablation that flips the qualitative conclusion purely from a Triton 3.3.1 → 3.4.0 upgrade, and a 7B AWQ probe showing the only compile path that survives `reduce-overhead` decode on Ada.

### Phase 6: Serving-State Safety Certification (TR144-TR152)
**5 technical reports (TR144, TR145, TR148, TR149, TR152) + TR146 negative control. ~130,000+ primary + judge measurements.**

Five-layer serving-state safety certification protocol synthesised across speculative decoding × safety (TR144 / TAIS), FP8 KV-cache single-config (TR145), mechanistic-probe falsification as negative control (TR146), multi-judge reliability + dual-axis methodology (TR148 / JTP), standardized safety batteries (TR149), and the FP8 serving-state factorial (TR152). Layer 1 = JTP measurement validity (1a refusal-axis, 1b composite-harm-axis); Layer 2 = behavioural screens (RTSI inherited from TR142 + TAIS); Layer 3 = CRI compile integrity (inherited from TR147); Layer 4 = scale validity (TR149 1B–3B anchor; TR151 7B–72B cloud-gated); Layer 5 = serving-state validity (TR152 fp8_anchored_star factorial).

Headline deliverable: a serving-state-independent FP8 (E4M3) KV-cache certificate for 1B–4B instruction-tuned models on vLLM v0.19.1 — harmful-refusal-neutral across batch ∈ {1, 8, 32}, prefix-caching {on, off}, and temperature ∈ {0.0, 0.7, 1.0} (TR152: **0 / 8,976 harmful-battery pairs discordant**; FP8-interaction spread 2.99 pp inside the ±3 pp band), with one located footnote — a sub-1 pp Qwen-2.5 over-refusal lean on the XSTest benign-edge battery (per-family Mantel–Haenszel OR 3.878 [2.386, 6.302]). The TR146 negative control is load-bearing: four mechanistic probes (first-token entropy, refusal-direction cosine, calibration drift, safety-neuron error magnitude) all fail to distinguish safe from dangerous configs, so the entire arc stays behavioural. Whole executed substrate ran at **$0 external API cost** under the umbrella gate on a local Ollama judge cohort. Cloud-gated envelope widenings (TR150 long-context, TR151 7B–72B scale, TR153 KV-method sweep) are gated on the 2026-10-24 GO/NO-GO trigger.

### Phase 7: Mitigation Turn (TR155, TR160, TR163) — MVP Pilots Only

Per `EXPERIMENTS_STATUS.md` 2026-05-28: **0/3 paper-grade deliverables.** TR155 (936 + 2,808 judge, kvpress), TR163 (offline LOOCV over TR142 RTSI table), TR160 (built, not run).

### Serving-Stack Expansion: TR164 — In Flight

SGLang + precision-controlled expansion of TR130–TR132; V1 active since 2026-05-31 (`research/tr164/results/20260531_120428_552237/`).

### Conclusive Reports
**21 synthesis documents spanning Phases 1–6** (6 phase groups × main + extended appendices + whitepaper).

Six conclusive reports (TR108-TR116, TR117-TR122, TR123-TR133, TR134-TR137, TR138-TR143, and Phase 6 / TR144-TR152), six extended appendices volumes, and six executive whitepapers — providing decision guidance with artifact provenance. The Phase 6 trilogy consolidates the seven executed serving-state safety reports into a single five-layer certification protocol.

---

## Technical Reports Index

### Phase 1: Foundation (TR108-TR122)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR108** | Single-Agent LLM Performance Analysis | 158 runs | Complete | Optimal configs for single-agent inference |
| **TR109** | Agent Workflow Optimization | 20 configs | Complete | GPU=60, CTX=512, TEMP=0.8 optimal for workflows |
| **TR110** | Concurrent Multi-Agent Performance (Python) | 150 runs | Complete | 99.25% parallel efficiency achieved |
| **TR111_v2** | Rust Single-Agent Performance | 57 runs | Complete | 114.54 tok/s baseline, 15.2% faster than Python |
| **TR112_v2** | Rust vs Python Comparison | 111 runs | Complete | Rust: +15.2% throughput, -58% TTFT, -67% memory |
| **TR113** | Rust Multi-Agent (Single Ollama) | 150 runs | Complete | 82.2% peak; server contention identified |
| **TR114_v2** | Rust Multi-Agent (Dual Ollama) | 150 runs | Complete | 98.281% mean efficiency, 0.74% contention |
| **TR115_v2** | Rust Runtime Optimization | 150 runs | Complete | Tokio-default recommended (98.72% mean, 1.21pp sigma) |
| **TR116** | Cross-Model Multi-Agent Benchmarks | 60 runs | Complete | Rust + Gemma 3 is king (99.2%); Qwen shows imbalance |
| **TR117** | Cross-Backend Inference Benchmark | 2,471 (546 degraded) | Complete | GPU-compile best mean; ONNX/TRT failures documented |
| **TR117_multi** | Multi-Agent Root Cause Analysis | — | Complete | Python event loop saturation (5.33ms mean lag) |
| **TR118_v2.2** | ONNX Runtime + TensorRT Deep Dive | 360 | Complete | TensorRT-fp16 best prefill (2.48ms, -87% vs baseline) |
| **TR119** | Cost & Energy Analysis | 360 | Complete | onnxruntime-gpu best cost ($0.1279/1M tok on-demand) |
| **TR120** | The "Compile Paradox" Root-Cause Audit | 546 | Complete | TR117 compile label misattributed; shape stability critical |
| **TR121v1** | Model Scaling Study | 684 | Complete | Scaling pipeline from 5M to 20B parameters established |
| **TR122** | Resource Profiling Deep Dive | 2,041 | Complete | Baseline power (20.71W), poller scheduling, thermal equilibrium |

### Phase 2: Deployment Framework (TR123-TR133)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR123** | KV-Cache Production Economics | 900 | Complete | Best cost $0.013/1M tokens; cached decode 2-8x cheaper |
| **TR124** | Quality & Accuracy Baseline | 24,990 | Complete | Backend choice does not affect quality (0/7 ANOVA significant) |
| **TR125** | Quantization Decision Matrix | 41,895 quality across 51 model-quant variants (v3: 9 formats incl. AWQ/GPTQ) | Complete (v3) | Q4_K_M remains GGUF default; AWQ matches on 4/5 models, GPTQ on 5/6; Q2_K universally unacceptable |
| **TR126** | Docker/Linux + Triton Validation | 25,400 | Complete (decode crash fixed upstream 2026-06) | Compile paradox resolved: 24-60% prefill speedup on Linux; decode crashed on stock PyTorch, now fixed upstream by #184102 (in review), validated on real decode; #175562 landed |
| **TR127** | Long-Context Characterization | 1,144 | Complete | VRAM spillover (25-105x cliffs), not quadratic attention, is the bottleneck |
| **TR128** | Production Workloads | 3,172 | Complete | NUM_PARALLEL is a no-op (0/30 significant); M/D/1 deviates 20.4x |
| **TR129** | N-Agent Scaling Laws | 5,310 | Complete | Amdahl s=0.39-0.54; throughput plateaus at N=2 |
| **TR130** | Serving Stack Benchmarking | 4,797 | Complete | vLLM 2.25x advantage at N=8 via continuous batching |
| **TR131** | GPU Kernel Profiling | 26 runs | Complete | Overturns TR130: GPU memory bandwidth, not serving stack, is the bottleneck |
| **TR132** | In-Container GPU Profiling | 25 runs | Complete | Continuous batching amortizes kernels 77-80%, bandwidth 79-83% |
| **TR133** | Predictive Capacity Planner | 19,676 | Complete | 4/4 validation targets met; `chimeraforge plan` CLI shipped |

### Phase 3: Safety Alignment (TR134-TR143)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR134** | Alignment Robustness Under Quantization | 48,603 safety + 21,096 judge (v3: 51 cells, 9 formats) | Complete (v3) | Safety broadly robust through Q3_K_S; Q2_K catastrophic; AWQ/GPTQ introduce format-specific hidden-danger regimes; refusal template destabilization mechanism identified |
| **TR135** | Safety Under Multi-Agent Concurrency | 20,316 | Complete | Null finding confirmed: concurrency has zero detectable effect on safety (I-squared = 0.0%) |
| **TR136** | Cross-Backend Safety Consistency | 16,032 | Complete | Backend matters more than quant: Llama 1B shows 23pp safety drop Ollama→FP16; no TOST equivalence |
| **TR137** | The Safety Tax of Inference Optimization | 74,254 | Complete | Quantization 57% of safety cost, backend 41%, concurrency 2%; worst config retains only 57.5% baseline safety |
| **TR138** | Batch Inference Safety Under Non-Determinism | 31,410 | Complete | Batch non-determinism produces 0.6% automated flip rate (0.16% human-adjudicated genuine); 73% of automated detections are regex artifacts |
| **TR138 v2** | Batch Safety — Strengthened-Evidence Revision | 7,257 | Complete | Audit layer confirms 59.1% unsafe flip direction; replication yields 1.68% safety vs 0.42% capability flip rate |
| **TR139** | Multi-Turn Jailbreak Under Quantization | 48,425 | Complete | All 8 strategy ANOVAs reject quant-independence (p < 1e-4); qwen2.5-1.5b/Q2_K/attention_shift reaches 100% ASR |
| **TR140** | Many-Shot & Long-Context Jailbreak Under Quantization | 30,000 | Complete | Llama immune above Q3_K_M; Q2_K universal vulnerability threshold; message array format 92% vs 0% faux dialogue |
| **TR141** | Cross-Architecture Refusal Fragility Under Batch Perturbation | 127,224 (v3.1) | Complete (v3.1) | 18 models, 10+ families; 0.94x safety/capability ratio (near parity); alignment type not predictive (p=0.942); output instability predicts fragility (r=0.91) |
| **TR142** | Quality-Safety Correlation Under Quantization | 51 cells; 41,895 quality + 48,603 safety + 21,096 judge (v3: GGUF+AWQ+GPTQ) | Complete (v3) | Simpson's paradox extends to AWQ/GPTQ; RTSI mitigator calibrated with LOOCV (recall=1.0) across all 51 cells; behavioral screen is the only viable pre-deployment check |
| **TR143** | Cross-Request Safety Leakage Under Continuous Batching | 14,250 | Complete (v2.0) | Aggregate composition effect not significant; directional asymmetry IS significant — 88-92% of flips trend unsafe (p=0.006) |
| **TR146** | Mechanistic Safety Probing Under Quantization | 5,100 forward passes | Complete | Safety neurons absorb 1.40x disproportionate quant error (p<0.0001), but none of 4 mechanistic probes distinguish safe from dangerous configs; behavioral screens (RTSI) remain the only viable pre-deployment check |

### Phase 5: Benchmarking Integrity (TR147)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR147 v1** | Cross-Regime Portability (Ada 48 GB) | 18,600 | Complete | Compiled decode crash reproduced on a second GPU family; phase-separation protocol holds |
| **TR147 v2** | Five-Experiment Ada Deep-Dive (E1–E5) | 6,840 | Complete | StaticCache rescues decode correctness at ~5.8x latency tax; DynamicCache + `reduce-overhead` decode unsafe |
| **TR147 v3** | A100 80 GB Ampere Second-Regime | 1,440 | Complete | Ampere amplifies, does not rescue: 100% decode crash at every token length tested on A100 |
| **TR147 v4** | StaticCache Retest + 7B AWQ + Triton Ablation (Ada + A100) | 35,400 | Complete | **Triton kill-shot**: same GPU + same code + Triton 3.3.1 → 3.4.0 flips prefill gain 62–77% → 0–3% and decode crash 80% → 0%, |d|>10; 7B AWQ is the only `reduce-overhead` decode path with 0% crash |

### Phase 6: Serving-State Safety Certification (TR144, TR145, TR148, TR149, TR152)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR144** | Speculative Decoding × Safety / TAIS | 16,783 core + 48,072 E1–E5 expansion = 64,855 total | Complete | Typical-Acceptance Invariance Screen (TAIS) named method; max |Cohen's h| = 0.024 across matched draft+target pairs under rejection vs typical acceptance; null cutoff |h| < 0.1 calibrated; integrated into NeurIPS submission #3738 |
| **TR145** | FP8 KV-Cache Single-Config Safety | 24,054 records (P1–P5); 13,724 judge-labelled (gemma3:12b, κ=0.43, 0 errors) | Complete | Across-the-board null: Phase 2 McNemar all 3 models p≥0.31, Phase 3 ctx×KV ANOVA p≥0.54, Phase 4 batch×KV ANOVA p≥0.98, Phase 5 turn-5 paired McNemar 1B p=0.22 / 3B p=1.0; MH pooled OR safety=1.05 [0.90, 1.23]; bound claims to "no detectable FP8 effect under tested conditions" |
| **TR148** | Multi-Judge Reliability / JTP Dual-Axis | 13,724 records × 5 judges = 41,272 fresh judge-label rows (regex + gemma3:12b + llama3.1:8b + shieldgemma:9b + llama-guard3:8b + 100-record gpt-4o calibration anchor) | Complete (v2, 1,556 lines) | TRIANGULATE verdict gemma3 × llama3.1 κ = 0.6917 at n=12,809, 0.0083 below JTP robust threshold; **dual-axis finding**: safety-specialist judges anti-correlate with general LLM judges (κ = −0.13 to −0.26) because they measure composite prompt+response harm, not response-only refusal; bridge paper Layer 1 splits 1a refusal-axis vs 1b composite-harm-axis |
| **TR149** | FP8 KV-Cache on Standardized Safety Batteries | 7,578 primary + 22,734 judge across 3 models × 4 batteries (HarmBench-400, JBB-100, StrongREJECT-313, XSTest-450) × 2 KV-cache dtypes | Complete (1,303 lines) | TR145 null replicates on the literature-canonical batteries: cross-battery MH pooled OR 0.8065 [0.38, 1.70], **0 / 12 cells Holm-significant**, **12 / 12 TOST-equivalent at ±3pp**; cross-corpus judge κ 0.83 (vs TR145's 0.69) confirms triangulate verdict is corpus-specific; first run under `--skip-openai-judge` umbrella gate ($0 external cost) |
| **TR152** | FP8 Serving-State Factorial (v2) | **45,000 primary records, 20,754 matched FP16-vs-FP8 pairs** across 5 models × `fp8_anchored_star` 14-cell star × 4 batteries (XSTest uncapped to 450, harmful at 100); 135,000 judge-label rows on disk (regex + gemma3:12b + llama3.1:8b, completed 2026-05-28 00:06 ET) | Complete (1,341 lines) | FP8 **refusal-neutral on harmful prompts across the entire factorial**: HarmBench/JBB/StrongREJECT perfectly concordant (0 / 8,976 discordant pairs across 5 models × 6 contexts × 3 batteries); 0 / 120 cells Holm-significant; **117 / 120 TOST-equivalent at ±3pp (97.5%, up from v1's 83.3%)**. Footprint = sub-1pp Qwen XSTest over-refusal lean: MH pooled OR 1.88 [1.32, 2.69] on 133 discordant pairs, sign-test p ≈ 0.0004. Speculative-decoding cells fail at vLLM v0.19.1 argparse (not OOM — v1's "cloud-gated by VRAM" attribution retracted). $0 external cost; 28.7 h end-to-end on RTX 4080 Laptop 12 GB |

### Phase 6 Companion: TR138 Camera-Ready Study D

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR138 Study D Addendum** | Batch-Invariant Kernel Ablation (ICML 2026 Workshop on Hypothesis Testing camera-ready supplement) | 110 primary (standard 22/55 label flips, batch-invariant 0/55) + 2 pilot runs (12 each) on run `20260524_172010` | Complete (870 lines) | Batch-invariance kernel ablation eliminates the flip signal end-to-end; supports the reviewer-requested batch-invariant-kernel control; folded into the accepted batch_inference_safety camera-ready **UPLOADED 2026-05-29** (workshop 2026-07-11) |

### Expansion v2 Reports (TR142 Matrix Expansion)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR125 v2** | Quality Evaluation — Expanded Matrix | 8,820 expansion + 24,990 original | Complete | 7 models across 4 families; Q4_K_M sweet spot confirmed cross-family; mistral-7b MMLU 58.9%→55.1% at Q2_K |
| **TR134 v2** | Safety Alignment — Expanded Matrix | 13,342 expansion + 24,778 original; 24,336 judge annotations across 3 sources | Complete | 6 models across 4 families; Q2_K catastrophe replicates on qwen2.5-1.5b (-50pp); Mistral regex-judge gap up to 71pp; multi-source judge coverage (legacy Qwen + Gemma 3 12B) |
| **TR142 v2** | Quality-Safety Correlation — 6-Model Synthesis | 40 cells from TR125+TR134 expanded | Complete | 34/36 sign reversals (Simpson's paradox at scale); 26/34 cells safety degrades faster; Q5_K_M floor holds all 6 models; per-model r from +0.997 to -0.829 |

### Expansion v3 Reports (AWQ/GPTQ Cross-Method)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR125 v3** | Quality Evaluation — AWQ/GPTQ Expansion | 41,895 quality total (37,485 loaded); 51 model-quant variants across 9 formats | Complete | AWQ matches Q4_K_M quality on 4/5 models; GPTQ matches on 5/6; phi-2 AWQ excluded (architecture incompatibility); mistral-7b GPTQ retains 97.8% quality |
| **TR134 v3** | Safety Alignment — AWQ/GPTQ Collapse | 48,603 safety + 21,096 judge across 51 cells | Complete | AWQ/GPTQ safety varies by model: llama3.2-1b GPTQ safe, qwen2.5-7b GPTQ hidden-danger; refusal template destabilization mechanism identified |
| **TR142 v3** | Quality-Safety Correlation — Multi-Format Synthesis | 51 cells (40 GGUF + 11 AWQ/GPTQ), 83-column matrix | Complete | Simpson's paradox extends to non-GGUF formats; AWQ and GPTQ introduce new hidden-danger rows not predicted by GGUF thresholds; RTSI mitigator calibrated across all formats |

### Conclusive Reports

| Report | Scope | Size |
|--------|-------|------|
| **Conclusive 108-116** | Phase 1 Synthesis (Python→Rust Migration) | 2,826 lines |
| **Conclusive 108-116 Extended Appendices** | Phase 1 Deep-Dive Appendices (108-116) | 1,171 lines |
| **Conclusive 108-116 Whitepaper** | Phase 1 Executive Guidance (108-116) | 214 lines |
| **Conclusive 117-122** | Phase 1 Synthesis (Benchmarking→Decision-Grade) | 208KB |
| **Conclusive 117-122 Extended Appendices** | Phase 1 Deep-Dive Appendices | 89KB |
| **Conclusive 117-122 Whitepaper** | Phase 1 Executive Guidance | 8KB |
| **Conclusive 123-133** | Phase 2 Synthesis | 433KB, 3,327 lines, 60 appendices |
| **Conclusive 123-133 Extended Appendices** | Phase 2 Deep-Dive Appendices | 62KB |
| **Conclusive 123-133 Whitepaper** | Phase 2 Executive Guidance | 15KB |
| **Conclusive 134-137** | Phase 3a Synthesis (Safety Cost of Inference Optimization) | 2,571 lines, 74,254 samples |
| **Conclusive 134-137 Extended Appendices** | Phase 3a Deep-Dive Appendices | 908 lines |
| **Conclusive 134-137 Whitepaper** | Phase 3a Executive Guidance | 228 lines |
| **Conclusive 138-143** | Phase 3b Synthesis (Safety Attack Surface) — file: `Technical_Report_Conclusive_Phase5.md` | 2,590 lines, 306,996 samples |
| **Conclusive 138-143 Extended Appendices** | Phase 3b Deep-Dive Appendices — file: `Technical_Report_Conclusive_Phase5_Extended_Appendices.md` | 744 lines |
| **Conclusive 138-143 Whitepaper** | Phase 3b Executive Guidance — file: `Technical_Report_Conclusive_Phase5_Whitepaper.md` | 285 lines |
| **Conclusive Phase 6** (144-152) | Serving-State Safety Certification Synthesis (5-layer protocol; TR144/145/146/147/148/149/152) | 1,453 lines |
| **Conclusive Phase 6 Extended Appendices** | Per-TR appendix volume (A–G + cross-cutting X1–X4 + glossary) | 1,585 lines |
| **Conclusive Phase 6 Whitepaper** | Serving-State Certification Executive Guidance (5-layer protocol table, Qwen 2.5 1.5B worked example, 8-objection reviewer FAQ) | 408 lines |

### Historical Reports (Superseded)

| Report | Superseded By | Reason |
|--------|---------------|--------|
| TR111 | TR111_v2 | Micro-benchmark replaced with full workflow |
| TR112 | TR112_v2 | Flawed comparison methodology |
| TR114 | TR114_v2 | Incorrect statistics (97.5% -> 98.281%) |
| TR115 | TR115_v2 | Incomplete data analysis (30 -> 150 runs) |
| TR118 | TR118_v2.2 | Multiple revisions; v2.2 is latest |
| TR119v1 | TR119 | Early draft superseded |
| TR121 | TR121v1 | Early draft superseded |

---

## Phase 2 Deployment Decisions

Six shippable decisions backed by ~105,945 primary measurements (Phase 3, per `BANTERHEARTS_MEASUREMENT_COUNT.md`):

1. **Single-agent:** Ollama Q4_K_M (highest throughput per dollar)
2. **Multi-agent (N>=4):** vLLM FP16 (2.25x advantage from continuous batching)
3. **Compile:** Prefill only, Linux only, Inductor+Triton; never compile decode
4. **Quantization:** Q4_K_M default; Q8_0 for quality-critical; never Q2_K
5. **Context budget:** Ollama for >4K tokens on 12 GB; HF FP16 spills at 8-16K
6. **Capacity planning:** `chimeraforge plan` (empirical lookup tables, R-squared >= 0.859)

### Decision Matrix

| Condition | Backend | Quantization | Compile | Streaming |
|-----------|---------|-------------|---------|-----------|
| N=1, decode-heavy | Ollama | Q4_K_M | N/A | Always on |
| N=1, prefill-heavy (Linux) | Compiled HF | FP16 | Prefill only | N/A |
| N=2-3, any workload | Ollama or vLLM (benchmark your mix) | Q4_K_M or FP16 | Per backend default | Always on |
| N>=4, any workload | vLLM FP16 | FP16 | N/A (vLLM manages internally) | Always on |
| Quality-critical | phi-2 or llama3.1-8b | Q8_0 | No | Always on |
| VRAM-constrained (12 GB) | Ollama | Q4_K_M; Q3_K_S for phi-2 only | No | Always on |

---

## Phase 1 Key Findings

### Single-Agent (TR111_v2, TR112_v2)

| Metric | Python | Rust | Rust Advantage |
|--------|--------|------|----------------|
| Throughput | 99.34 tok/s | 114.54 tok/s | +15.2% |
| TTFT (cold) | 1437 ms | 603 ms | -58.0% |
| Memory | ~250 MB | ~75 MB | -67% |
| Startup | 1.5s | 0.2s | -83% |
| CV (throughput) | 4.8% | 2.6% | -46% (more consistent) |

### Multi-Agent (TR110, TR114_v2)

| Metric | Python (TR110) | Rust (TR114_v2) | Rust Advantage |
|--------|---------------|-----------------|----------------|
| Peak Config Avg | 99.25% | 99.396% | +0.15pp |
| Peak Single Run | 99.25% | 99.992% | +0.74pp |
| Mean Efficiency | 95.8% | 98.281% | +2.48pp |
| Contention Rate | 10-15% | 0.74% | -10-14pp |

### Runtime Optimization (TR115_v2)

| Runtime | Mean (%) | StdDev (pp) | Recommendation |
|---------|----------|-------------|----------------|
| Tokio-default | 98.72 | 1.21 | Production |
| Smol-1KB | 98.61 | 1.32 | Small binary alternative |
| Tokio-localset | 97.95 | 4.03 | Unstable |
| Smol | 97.72 | 4.87 | Avoid |
| Async-std | 50.00 | 0.00 | Unusable |

---

## Phase 2 Key Findings

- **Quantization is the dominant lever.** Q4_K_M saves 30-67% cost versus FP16 while losing at most -4.1pp accuracy. Q2_K is universally unacceptable (all models >11pp loss; qwen2.5-1.5b collapses -40.6pp).

- **Backend choice does not affect quality.** 7 quality metrics across 5 models and 2 backends at temp=0; none showed statistically significant differences (TR124).

- **The GPU, not the serving stack, is the scaling bottleneck.** PyTorch Direct (no serving stack) degrades 86.4% at N=8, worse than Ollama's 82.1%. Memory bandwidth stress increases +74% at N=8 (TR131).

- **Continuous batching amortizes the bandwidth bottleneck.** 77-80% kernel count reduction and 79-83% bandwidth-per-token reduction at N=8. Amortization ratio is 4.7-5.8x (TR132).

- **VRAM spillover, not quadratic attention, is the practical long-context bottleneck.** 25-105x latency cliffs when KV-cache pushes VRAM past capacity. Below spillover, Ollama prefill scaling is sub-linear (b = 0.083-0.158). GQA models sustain 3-11x longer contexts than MHA models (TR127).

- **NUM_PARALLEL is a no-op.** 0/30 pairwise comparisons significant (mean absolute change 4.0%). M/D/1 queueing theory deviates up to 20.4x from observed latency (TR128).

- **Multi-agent throughput plateaus at N=2.** Amdahl's Law with serial fractions s = 0.39-0.54 (R-squared > 0.97). Per-agent throughput at N=8 is 17-20% of solo throughput (TR129).

- **Consumer hardware is 95.4% cheaper than cloud.** TCO at 1B tokens/month: $153/yr consumer versus $2,880/yr AWS on-demand. Break-even for an RTX 4080 occurs at 0.3-2.7 months at 10M requests/month (TR123).

---

## Phase 3 Key Findings

- **Quantization is the dominant safety cost axis, but not universally.** Quantization accounts for 57% of total safety cost (TR137), safety is broadly robust through Q3_K_S for well-aligned models, and Q2_K triggers catastrophic failure across all tested families (TR134). However, model heterogeneity is extreme (I-squared = 99.9%) — universal thresholds are unreliable.

- **Serving backend is the second largest safety variable.** Backend choice accounts for 41% of total safety cost (TR137). Llama 1B shows a 23pp safety drop between Ollama Q4_K_M and vLLM FP16 despite higher precision — driven by chat template divergence between GGUF-embedded and HuggingFace tokenizer configs (TR136). Safety evaluations on one backend do not transfer to another.

- **Concurrency is the one safe axis.** 39,060 samples, all safety slopes indistinguishable from zero, I-squared = 0.0%. Concurrent Ollama requests queue rather than interfere. You can safely scale agents without safety degradation (TR135).

- **Batch non-determinism introduces small safety instability.** Automated detection shows 0.6% flip rate; human adjudication (n=63, single reviewer) reduces this to 0.16% genuine (TR138). Across 15 models, fragility varies from 0.00% to 2.39% (TR141). Alignment type does not predict fragility (p=0.942); output instability is the sole reliable predictor tested (r=0.91).

- **Multi-turn jailbreaks are systematically amplified by lower quantization.** All 8 strategy ANOVAs reject quant-independence (p < 1e-4). qwen2.5-1.5b at Q2_K reaches 100% ASR on three attack strategies (TR139). Persistence of initial refusals degrades monotonically with lower bit-width for 3 of 4 models.

- **Quality metrics alone are insufficient safety proxies.** Safety degrades up to 13.9x faster than quality at Q3_K_S on llama3.2-1b. The quality-safety correlation reverses sign between architectures (r = +0.994 on 1b, r = -0.829 on 3b) — pooled analysis is misleading (TR142).

- **Q2_K is the universal vulnerability threshold for many-shot attacks.** Llama models are immune above Q3_K_M; at Q2_K every tested model shows significantly elevated attack success rates. Message array format is dramatically more effective than faux dialogue at the same quantization level (TR140).

- **AWQ and GPTQ introduce safety risks not predicted by GGUF thresholds.** At matched effective bit-widths (~4-bit), AWQ and GPTQ produce different safety profiles than GGUF Q4_K_M. Some models (llama3.2-1b) are safe under GPTQ; others (qwen2.5-7b) enter hidden-danger regimes. Format-specific safety evaluation is required (TR134 v3, TR142 v3).

- **Mechanistic probes cannot predict quantization-induced safety failure.** First-token entropy, refusal direction cosine similarity, calibration drift, and safety-neuron error magnitude are all insufficient to distinguish safe from dangerous quantized configurations. Safety neurons absorb 1.40x disproportionate error (p < 0.0001), but this is universal across all quantized cells, not specific to hidden-danger rows. Behavioral screens (RTSI) remain the only viable pre-deployment check (TR146).

---

## Phase 5 Key Findings — Benchmarking Integrity

- **`torch.compile` benchmark identity is a 5-tuple, not a single number.** The combination (GPU sm_N, Triton minor version, PyTorch build, cache implementation, compile mode) determines the result. Changing any one axis with all others held constant can flip the qualitative conclusion. A published benchmark missing any of these five axes is point-in-time evidence, not a reproducible claim (TR147).

- **Triton minor version alone can flip the conclusion on the same GPU.** Triton 3.3.1 → 3.4.0 on the same RTX 6000 Ada with the same Qwen 2.5 1.5B FP16 model and same Python code: prefill compile gain collapses from −77% to 0–3%, and decode crash rate collapses from 80% to 0%. Cross-version Cohen's d on compile-vs-eager contrasts: 15–49. The eager arm is flat across versions (|d|≤0.15), so the variance lives on the compiled path only (TR147 v4 A1–A3).

- **The stability win and the speedup loss are independent upstream changes.** Decode stability comes from `pytorch/pytorch` PR #175562 (assertion relaxation, drafted during this research program). Prefill speedup loss comes from `triton-lang/triton` PR #7138 (LLVM + PTXAS register-spilling regression) plus register-allocator rewrites in the same Triton release. They are two separate changes by different authors that coincidentally landed at the same software-line boundary. The correctness fix is free; the speedup loss is a separable, remediable Triton regression — partial recovery is already visible in Triton 3.6.0 (TR147 v4 SS8.7).

- **Ampere (A100 sm_80) amplifies the compiled-decode failure rather than rescuing it.** Compiled decode crashes 100% at every tested token length on A100 (128, 256, 512), worse than Ada's 3.6% floor under the same harness. The "maybe a bigger GPU fixes it" objection is rejected (TR147 v3).

- **StaticCache restores compiled-decode correctness at a measurable latency cost.** Swapping DynamicCache → StaticCache gives 0% decode crash under `mode="default"` on both Ada and A100, at the cost of compiled decode being 1.6–3.5% *slower* than eager. TOST rejects any decode speedup claim. Compiled prefill retains 54–63% gains across the stable cells (TR147 v4 B1–B2).

- **AWQ-4bit 7B is the only compile path with zero `reduce-overhead` decode crash.** Across every (GPU, model, cache, compile mode, Triton) combination TR147 evaluated, the single cell that survives `reduce-overhead` decode with 0% crash is Ada + qwen2.5-7b-AWQ-4bit + StaticCache. Large dense FP16 7B and 8B models still show the phase split (50–57% prefill gain, 80% decode crash). Quantization via AWQ accidentally stabilizes compilation in a way that dense FP16 at the same parameter count does not (TR147 v4 SS7).

---

## Phase 6 Key Findings — Serving-State Safety Certification

- **FP8 KV-cache is refusal-neutral on harmful prompts across the entire factorial.** TR152 v2 records **0 / 8,976 discordant pairs** on HarmBench / JailbreakBench / StrongREJECT across 5 models × 6 contexts × 3 harmful batteries, **0 / 120 cells Holm-significant**, and **117 / 120 TOST-equivalent at ±3 pp** under the matched-pairs Cohen's h paired-binary effect-size protocol. FP8-interaction spread is 2.99 pp — inside the ±3 pp band. The TR145 null replicates on standardized batteries (TR149: cross-battery MH pooled OR 0.8065 [0.38, 1.70], 12/12 TOST-equivalent) and on the full serving-state factorial (TR152). FP8 KV-cache safety claims are bounded to 1B–4B instruction-tuned models on vLLM v0.19.1 at temperature ∈ {0.0, 0.7, 1.0}, batch ∈ {1, 8, 32}, prefix-caching {on, off}.

- **The only located footprint is a sub-1 pp Qwen-2.5 over-refusal lean on the XSTest benign-edge battery.** Per-family Mantel–Haenszel OR 3.878 [2.386, 6.302] on 133 discordant pairs, sign-test p ≈ 0.0004, refining v1's barely-clearing 2.69 [1.09, 6.62] by 5.8 × the discordant base. Llama 3.2-{1B, 3B} and phi-3-mini-4k pattern with the Llama-3.2-3b family side: clean on XSTest, FP8 marginally improves their over-refusal behavior. The signal localizes to the Qwen family, amplified mildly by temperature on the 1.5 B variant (TR152).

- **Mechanistic probes cannot predict the serving-state behavior — RTSI remains the only viable pre-deployment screen.** TR146's four probes (first-token entropy, refusal-direction cosine, calibration drift, safety-neuron error magnitude) all fail to distinguish safe from dangerous configs. Safety neurons absorb 1.40 × disproportionate quantization error universally (p<0.0001), but the damage is necessary, not sufficient, for behavioral failure. The Phase 6 protocol therefore stays behavioural end-to-end (TR146).

- **JTP triangulation requires multi-judge majority-vote at corpus scale.** TR148 v2's cross-LLM pair gemma3 × llama3.1 reaches κ = 0.6917 at n = 12,809 — 0.0083 below the JTP robust threshold of 0.70 → TRIANGULATE bucket. Downstream Phase 6 TRs (TR149, TR151, TR152) require multi-judge majority-vote rather than single-judge sign-off. The bridge paper Layer 1a anchors on this verdict (TR148 v2).

- **Safety judges measure two orthogonal axes, not one noisy axis.** Adding safety-specialist judges (shieldgemma:9b + llama-guard3:8b) at corpus scale reveals they **anti-correlate** with general LLM judges across 4 large-n pairs (cross-axis κ = −0.13 to −0.26) because they measure composite prompt+response harm, not response-only refusal. The bridge paper Layer 1 now splits the safety-specialist axis as an orthogonal screen (Layer 1b), not a fifth column of the refusal-axis JTP (Layer 1a). This is a methodological contribution that emerged only at TR148 v2's 5-judge scale (TR148 v2 SS4).

- **Typical-acceptance speculative decoding is safety-invariant under matched draft+target pairs.** TR144's E1–E5 expansion (48,072 expansion-probe samples on top of the 16,783-sample core) finds max |Cohen's h| = 0.024 across rejection sampling vs typical acceptance — well below the ±0.1 null cutoff calibrated from the same expansion. TAIS named method (Typical-Acceptance Invariance Screen) ships as the Phase 6 Layer 2 behavioural complement to RTSI; integrated into NeurIPS submission #3738 (TR144).

- **The whole locally-runnable arc ran at $0 external API cost.** TR145 + TR146 + TR148 + TR149 + TR152 + TR144 expansion all under the `--skip-openai-judge` umbrella gate on a local Ollama judge cohort (regex + gemma3:12b + llama3.1:8b + shieldgemma:9b + llama-guard3:8b). No adversarial-prompt content sent to any external API. Cloud-gated envelope widenings (TR150 long-context, TR151 7B–72B scale, TR153 KV-method sweep) wait on the 2026-10-24 GO/NO-GO trigger (Anthropic Fellowship + NeurIPS-acceptance signal).

---

## Report Details

### Phase 1

#### TR108: Single-Agent LLM Performance Analysis
**File:** `Technical_Report_108.md`
- Models: gemma3:latest, llama3.1:8b-instruct variants
- Hardware: NVIDIA RTX 4080 (12GB VRAM), i9-13980HX
- Test Matrix: 150+ benchmark runs across parameter sweeps

#### TR109: Agent Workflow Optimization
**File:** `Technical_Report_109.md`
- Optimal Config: GPU=60, CTX=512, TEMP=0.8 for agent workflows
- Methodology: Process isolation, forced cold starts, statistical validation

#### TR110: Concurrent Multi-Agent Performance (Python)
**File:** `Technical_Report_110.md`
- Test Matrix: 30 configurations x 5 runs = 150 benchmark runs
- Key Finding: 99.25% parallel efficiency with homogeneous Chimera agents

#### TR111_v2: Rust Single-Agent Performance
**File:** `Technical_Report_111_v2.md`
- Test Matrix: 19 configurations x 3 runs = 57 benchmark runs
- Baseline: 114.54 tok/s (15.2% faster than Python)
- Supersedes: TR111 (micro-benchmark)

#### TR112_v2: Rust vs Python Comparison
**File:** `Technical_Report_112_v2.md`
- Test Matrix: 37 configurations (19 Rust + 18 Python), 111 total runs
- Rust: +15.2% throughput, -58% TTFT, -67% memory, -83% startup
- Supersedes: TR112

#### TR113: Rust Multi-Agent (Single Ollama)
**File:** `Technical_Report_113.md`
- 82.2% peak efficiency, 63% contention rate
- Finding: Server-level serialization bottleneck; dual Ollama required

#### TR114_v2: Rust Multi-Agent (Dual Ollama)
**File:** `Technical_Report_114_v2.md`
- Test Matrix: 27 configurations x 5 runs = 135 benchmark runs
- Peak single run: 99.992%; Mean: 98.281% (+2.48pp vs Python)
- Supersedes: TR114

#### TR115_v2: Rust Runtime Optimization
**File:** `Technical_Report_115_v2.md`
- Test Matrix: 5 runtimes x 6 configs x 5 runs = 150 benchmark runs
- Recommendation: Standard Tokio — no custom config needed
- Supersedes: TR115

#### TR116: Cross-Model Multi-Agent Benchmarks
**File:** `Technical_Report_116.md`
- Test Matrix: 3 models x 2 runtimes x 2 scenarios x 5 runs = 60 runs
- Rust dominates across all models (+12-17pp efficiency vs Python)
- Gemma 3 is the scaling king (99.2% efficiency in Rust)

#### TR117: Cross-Backend Inference Benchmark
**File:** `Technical_Report_117.md`
- Test Matrix: 3,017 runs, 2,471 successful (82%)
- GPU-compile wins on mean latency (389ms); plain GPU wins on median (323ms)

#### TR117_multi_agent: Multi-Agent Root Cause Analysis
**File:** `Technical_Report_117_multi_agent.md`
- Python event loop saturation: Mean lag 5.33ms, p99 12.13ms, max 15.22ms
- For >100 tok/s multi-agent systems, Rust is mandatory

#### TR118_v2.2: ONNX Runtime + TensorRT Deep Dive
**File:** `Technical_Report_118_v2.2.md`
- Test Matrix: 360 run-level records across prefill and generate modes
- Best prefill: TensorRT-fp16 (2.48ms, -87% vs baseline)
- Supersedes: TR118, TR118_v2.1

#### TR119: Cost & Energy Analysis
**File:** `Technical_Report_119.md`
- Test Matrix: 5 backends x 5 scenarios x 7 reps x 2 modes = 350 runs
- Best cost: onnxruntime-gpu at $0.1279/1M tokens (on-demand)
- Lowest carbon: ~1.0 gCO2e/1M tokens

#### TR120: The "Compile Paradox" Root-Cause Audit
**File:** `Technical_Report_120.md`
- TR117's "compile paradox" is real but misattributed (label-only, no actual torch.compile)
- Shape stability fix: Padding/bucketing collapses compiled tail

#### TR121v1: Model Scaling Study
**File:** `Technical_Report_121v1.md`
- Scaling pipeline from 5M to 20B parameters (HF + Ollama)
- Three distinct regimes identified (small GPU, CPU, large-model serving)

#### TR122: Resource Profiling Deep Dive
**File:** `Technical_Report_122.md`
- Baseline power: RTX 4080 Laptop GPU idles at 20.71W (sigma=9.97W)
- V2 strict poller scheduling achieves 100ms grid adherence
- Thermal equilibrium: Small models reach equilibrium at 48C

### Phase 3

#### TR134: Alignment Robustness Under Quantization
**File:** `Technical_Report_134_v2.md`
- 6 models across 4 families, 40 model-quant entries, 6 benchmarks
- 38,120 evaluated safety samples plus 24,336 judge annotations across three source files
- Safety is broadly robust through Q3_K_S for well-aligned models; qwen2.5-1.5b replicates the Q2_K cliff, while qwen2.5-7b and mistral-7b expose hidden-danger / near-hidden-danger regimes
- Regex-vs-judge disagreement is strongly model-dependent: higher-fidelity Llama and Qwen rows are often low-gap, while Mistral underreports refusal by 64-71pp

#### TR135: Safety Under Multi-Agent Concurrency
**File:** `Technical_Report_135.md`
- 3 models, 4 concurrency levels (N=1,2,4,8), 6 benchmarks
- 39,060 raw records, 10,416 prompt-level observations, 9,900 judged samples
- Null finding: all safety slopes indistinguishable from zero; 8/9 TOST equivalent
- Latency scales linearly with N; safety does not degrade

#### TR136: Cross-Backend Safety Consistency
**File:** `Technical_Report_136.md`
- 3 models, 4 backends (Ollama Q4/Q8, vLLM FP16, TGI FP16), 6 benchmarks
- 10,416 evaluated samples, 5,616 judged samples
- Llama 1B: 23pp safety drop Ollama→FP16 (Cohen's d = -0.55 to -0.61, p < 0.0001)
- Mechanism: GGUF-embedded chat template vs HuggingFace tokenizer_config divergence

#### TR137: The Safety Tax of Inference Optimization
**File:** `Technical_Report_137.md`
- Meta-analysis synthesis of TR134-TR136; 74,254 total samples, 18 analysis passes
- Quantization: 57% of safety cost; Backend: 41%; Concurrency: 2%
- Worst combined config (Llama 1B Q2_K): 57.5% baseline safety retained (CRITICAL tier)
- I-squared = 99.9% on quant axis — universal guidelines are unreliable

#### TR138: Batch Inference Safety Under Non-Determinism
**File:** `Technical_Report_138.md`
- 4-phase batching study, 31,410 total samples; vLLM + Ollama backends
- Safety flip rate 4x higher than capability flip rate under batch non-determinism
- Phase 5 explicit true-batch validates effect is not a timing artifact

#### TR138 v2: Batch Safety — Strengthened-Evidence Revision
**File:** `Technical_Report_138_v2.md`
- Audit + replication of TR138 v1; 7,257 replication samples on enriched 187-prompt subset
- Audit layer: 44 true flip candidates; 59.1% in unsafe direction
- Replication: 1.68% safety flip rate vs 0.42% capability flip rate (vs v1's 4x ratio)
- Scorer corrected (v2.2): Unicode curly-quote normalization removed 5 false flip candidates
- Supersedes: TR138 v1 for quantitative flip-rate claims

#### TR139: Multi-Turn Jailbreak Under Quantization
**File:** `Technical_Report_139.md`
- 4 models, 6 GGUF quant levels, 8 attack strategies, 50 harmful behaviors
- 10,600 conversations (9,600 Phase 1 + 1,000 Phase 2), 37,825 judge labels
- All 8 strategy ANOVAs reject quant-independence (p < 1e-4); eta-squared 0.031-0.153
- Highest ASR: qwen2.5-1.5b/Q2_K/attention_shift, context_fusion, crescendo all at 100%

#### TR140: Many-Shot & Long-Context Jailbreak Under Quantization
**File:** `Technical_Report_140.md`
- 4 models, 6 quant levels, 5 shot counts, 2 prompt formats, 3 context profiles
- 15,000 scored samples (12,000 Phase 1 + 3,000 Phase 2), 15,000 judge labels
- Llama models immune above Q3_K_M; Q2_K is the universal vulnerability threshold
- Message array format 92% vs 0% faux dialogue on llama3.1-8b Q2_K at N=16
- Variance decomposition: residual 65.7%, quantization 17.9%, model 12.6%, shot count 2.7%

#### TR141: Cross-Architecture Refusal Fragility Under Batch Perturbation
**File:** `Technical_Report_141.md`
- **Cross-architecture batch safety report. v3.1, 1,726 lines.**
- 18 models (360M-14.8B), 10+ families, 4 alignment types (RLHF, SFT, DPO, Distilled)
- 127,224 evaluation records across three campaigns; combined v2.1+v3 synthesis: 106,020 scored records
- Combined synthesis: 0.75% safety vs 0.80% capability (0.94x ratio, near parity)
- Alignment type NOT predictive (F=0.13, p=0.942 model-level); v2.1 p=0.008 was false positive from pseudoreplication
- Output instability predicts fragility (r=0.91, R²=0.83); cross-architecture spread 0.00%-2.39%
- Net-safe directional bias: 159 compliance→refusal vs 81 refusal→compliance flips (p=1e-6)

#### TR142: Quality-Safety Correlation Under Quantization
**File:** `Technical_Report_142_v2.md`
- Merged analysis of 33,810 quality samples, 38,120 safety samples, and 24,336 judge annotations
- 6 matched models across 4 families, 40 model-quant cells, 14 core analysis passes plus supporting diagnostics
- Sign reversal persists in the expanded matrix: 34/36 quality-safety pairings split positive and negative across models
- Safety degrades 13.9x faster than quality at llama3.2-1b Q3_K_S, while qwen2.5-7b Q2_K reproduces the hidden-danger pattern outside Llama
- Q5_K_M remains the conservative floor; Q4_K_M is still model-dependent and ambiguous

#### TR146: Mechanistic Safety Probing Under Quantization
**File:** `Technical_Report_146.md`
- 6 models, 3 quant methods (FP16, AWQ INT4, GPTQ INT4), 17 model-quant cells per phase
- 5,100 forward passes with hidden-state extraction across 4 phases
- Phase 1 (first-token entropy): AWQ increases, GPTQ decreases uncertainty — no predictive power (r=0.08)
- Phase 2 (refusal direction): cosine similarity >0.97 in every cell, yet behaviorally ineffective in hidden-danger rows
- Phase 3 (calibration drift): negative drift — quantization doesn't degrade calibration, no predictive power
- Phase 5 (safety neuron error): 1.40x disproportionate error (p<0.0001), but universal, not danger-specific (regime p=0.98)
- Key conclusion: all four mechanistic probes fail to distinguish safe from dangerous configs; RTSI remains the only viable screen

### Phase 6: Serving-State Safety Certification (TR144, TR145, TR148, TR149, TR152)

#### TR144: Speculative Decoding × Safety / TAIS
**File:** `Technical_Report_144.md` (1,925 lines)
- 16,783-sample core paired study + 48,072-sample E1–E5 expansion = 64,855 total samples
- Hardware: RunPod A100-SXM4-80GB for the E1–E5 expansion; RTX 4080 Laptop for core paired runs
- Named method: **TAIS** (Typical-Acceptance Invariance Screen) — matched-pairs Cohen's h ±0.1 null cutoff calibrated from expansion max observed 0.024
- Verdict: speculative decoding via rejection sampling vs typical acceptance under matched draft+target pairs is safety-invariant
- Integrated into NeurIPS 2026 E&D Track submission #3738 (TAIS paper, under blind review; notification 2026-09-24 AoE)

#### TR145: FP8 KV-Cache Single-Config Safety
**File:** `Technical_Report_145.md` (1,465 lines)
- 24,054 records across 5 phases (P1=3009 FP16 baseline, P2=3009 FP8 paired, P3=4000 ctx×KV interaction, P4=12036 batch×KV interaction, P5=2000 multi-turn)
- 13,724 judge-labelled records (gemma3:12b, κ=0.43, 0 judge errors)
- Hardware: RTX 4080 Laptop 12GB, vLLM v0.19.1 Docker pinned, port 8801
- Across-the-board null: Phase 2 McNemar all 3 models p≥0.31, Phase 3 ctx×KV ANOVA p≥0.54, Phase 4 batch×KV ANOVA p≥0.98 (interactions ~additive), Phase 5 turn-5 paired McNemar 1B p=0.22 / 3B p=1.0
- MH pooled OR safety=1.05 [0.90, 1.23], batch=8 OR=1.00, turn-5 OR=2.06 (CI [0.61, 6.99])
- Claims bounded to "no detectable FP8 effect under tested conditions" with paired-profiling deployment guidance — not "FP8 is safe"

#### TR148: Multi-Judge Reliability / JTP Dual-Axis
**File:** `Technical_Report_148.md` (1,556 lines)
- 13,724 records re-judged with 5 fresh judges (regex + gemma3:12b + llama3.1:8b + shieldgemma:9b + llama-guard3:8b) + 100-record gpt-4o calibration anchor from a killed synchronous run
- 41,272 fresh judge-label rows added on top of TR145's gemma3:12b baseline
- TRIANGULATE verdict: gemma3 × llama3.1 cross-LLM κ = 0.6917, n=12,809 paired records, 0.0083 below JTP robust threshold of 0.70 → downstream Phase 6 TRs require multi-judge majority-vote
- **Dual-axis finding (SS4):** safety-specialist judges anti-correlate with general LLM judges (cross-axis κ = −0.13 to −0.26 across 4 large-n pairs) because they measure composite prompt+response harm, not response-only refusal
- Bridge paper Layer 1 splits: 1a refusal-axis JTP (general judges) + 1b composite-harm-axis screen (safety specialists)
- Integrated into NeurIPS 2026 E&D Track submission #3724 (JTP paper) + bridge-paper Layer 1 anchor

#### TR149: FP8 KV-Cache on Standardized Safety Batteries
**File:** `Technical_Report_149.md` (1,303 lines)
- 7,578 primary records across 3 models × 4 batteries (HarmBench-400, JailbreakBench-100, StrongREJECT-313, XSTest-450) × 2 KV-cache dtypes
- 22,734 judge-label rows (3 judges — regex + gemma3:12b + llama3.1:8b — × 7,578 records, 0 judge errors)
- Hardware: RTX 4080 Laptop 12GB, vLLM v0.19.1 Docker pinned, port 8801
- TR145 null **replicates on the literature-canonical batteries**: cross-battery MH pooled OR 0.8065 [0.38, 1.70]; **0 / 12 cells Holm-significant**; **12 / 12 TOST-equivalent at ±3pp**
- Cross-corpus judge κ = 0.83 (vs TR145's 0.69) confirms triangulate verdict is corpus-specific, not a universal noise floor
- First run under the new `--skip-openai-judge` umbrella gate (`triangulate_no_openai` bucket): $0 external API cost, no adversarial-prompt content sent to any external API

#### TR152: FP8 Serving-State Factorial (v2)
**File:** `Technical_Report_152.md` (1,341 lines — supersedes v1 pilot's 348 lines; matches TR148/TR149 gold-standard template depth)
- **v2 local expansion run `20260526_232600`**: 45,000 primary records, 20,754 matched FP16-vs-FP8 pairs across 5 models × `fp8_anchored_star` 14-cell star × 4 batteries (XSTest uncapped to 450, harmful at 100); judging completed 2026-05-28 00:06 ET — 135,000 judge-label rows on disk
- Models: qwen2.5-1.5b-instruct, qwen2.5-3b-instruct, llama3.2-1b-instruct (new in v2), llama3.2-3b-instruct, phi-3-mini-4k-instruct (new in v2)
- Hardware: RTX 4080 Laptop 12GB, $0 external cost, 28.7 h end-to-end
- Bridge paper Layer 5: folds FP8 KV-cache + batch + prefix-caching + speculative-decoding + temperature axes into one factorial (12 of 14 cells per model runnable; 2 speculative-decoding cells fail uniformly on vLLM v0.19.1 argparse — **v1's "cloud-gated by VRAM" attribution retracted**)
- **FP8 is refusal-neutral on harmful prompts across the entire factorial.** HarmBench / JBB / StrongREJECT perfectly concordant: **0 / 8,976 discordant pairs across 5 models × 6 contexts × 3 batteries**, 0 / 120 cells Holm-significant, **117 / 120 TOST-equivalent at ±3 pp (97.5%, up from v1's 83.3%)**
- The only footprint: sub-1 pp Qwen-2.5 over-refusal lean on XSTest — Mantel-Haenszel pooled OR 1.88 [1.32, 2.69] on 133 discordant pairs, sign-test p ≈ 0.0004, refining v1's 2.69 [1.09, 6.62] on 23 pairs by 5.8 × the discordant base and tightening the CI by 4 ×
- SS14 tightened 2026-05-28 with v2's per-battery cross-judge κ recomputed directly from the sign-aware `per_judge_outcome` substrate (XSTest gemma↔llama κ = +0.7989 robust by JTP; harmful batteries degenerate-via-chance-correction-not-disagreement; pooled κ sanity-matches the analysis JSON at 0.8310 / 0.0622 / 0.0962)

#### TR138 Study D Addendum: Batch-Invariant Kernel Ablation
**File:** `Technical_Report_138_Study_D_Addendum.md` (870 lines)
- ICML 2026 Workshop on Hypothesis Testing camera-ready supplement for the accepted batch_inference_safety paper (acceptance rating 7, "clearly in-scope"; camera-ready **UPLOADED 2026-05-29** ahead of 2026-06-17 deadline; workshop 2026-07-11)
- Canonical run `20260524_172010`: 110 primary records (standard 22/55 label flips, batch-invariant 0/55) + 2 pilot runs (12 each)
- Reviewer-requested batch-invariant-kernel ablation eliminates the flip signal end-to-end, supporting the paper's mechanistic claim
- Companion to the TR138 v2.2 / TR141 / TR143 evidence base

### v3 AWQ/GPTQ Cross-Method Reports

#### TR125 v3: Quantization Decision Matrix (AWQ/GPTQ Expansion)
**File:** `Technical_Report_125_v3.md`
- 6 models, 4 families, 51 model-quant variants across 9 formats (7 GGUF + AWQ + GPTQ)
- 41,895 total quality samples (37,485 loaded) across v1 + v2 + v3 waves
- Hardware: RTX 4080 Laptop (small models), RunPod RTX 6000 Ada 48GB (7B models)
- AWQ matches Q4_K_M quality on 4/5 models; GPTQ matches on 5/6; phi-2 AWQ excluded (parallel attention incompatibility)

#### TR134 v3: Alignment Robustness Under Quantization (AWQ/GPTQ Safety)
**File:** `Technical_Report_134_v3.md`
- 48,603 total safety samples + 21,096 judge annotations across 51 cells
- 11 AWQ/GPTQ entries add 10,483 safety samples and 5,148 judge annotations
- Refusal template destabilization mechanism identified: AWQ/GPTQ disrupt refusal surface differently from GGUF
- Second-judge robustness check (Claude Sonnet 4, 11,470 rows) validates consistency

#### TR142 v3: Quality-Safety Correlation — Multi-Format Synthesis
**File:** `Technical_Report_142_v3.md`
- 51-row × 83-column unified matrix (40 GGUF + 11 AWQ/GPTQ)
- Simpson's paradox extends to non-GGUF: AWQ and GPTQ introduce hidden-danger rows not predicted by GGUF thresholds
- RTSI mitigator calibrated with LOOCV (recall=1.0) across all 51 cells
- Supersedes TR142 v2 (40-cell GGUF-only analysis)

### Phase 2

#### TR123: KV-Cache Production Economics
**File:** `Technical_Report_123.md`
- 5 models (124M-3.2B), 5 backends, 5 cost blends
- Best cost: $0.013/1M tokens (GPT-2/compile)
- Consumer vs cloud: 95.4% savings ($153/yr vs $2,880/yr at 1B tokens/month)

#### TR124: Quality & Accuracy Baseline
**File:** `Technical_Report_124.md`
- 5 models x 2 backends, temp=0, 7 quality metrics
- Backend choice does not affect quality: 0/7 ANOVA significant

#### TR125: Quantization Decision Matrix
**File:** `Technical_Report_125_v2.md`
- 33,810 quality samples across 7 models, 4 families, and 46 model-quant variants
- Q4_K_M remains the quality default across all 7 models, with qwen2.5-7b and mistral-7b extending the result to 7B scale
- Q2_K remains universally unacceptable, with the worst collapse on qwen2.5-1.5b (-35.1pp MMLU, -48.5pp ARC)

#### TR126: Docker/Linux + Triton Validation
**File:** `Technical_Report_126.md`
- Compile paradox resolved: 24-60% prefill speedup on Linux with Inductor+Triton
- Decode compile crashes 100% of the time in all tested modes (on stock PyTorch)
- PyTorch bug discovered and reported upstream (pytorch/pytorch#175557, PR #175562)

**Update (2026-06-05):** The compiled-decode crash is fixable from PyTorch internals
after all. PR #175562 (assertion relaxation in `dealloc_current_path_weakrefs`) landed
in PyTorch main on 2026-06-04 (commit `be90a1495310`); it is a defensive cleanup and
does not by itself fix the decode crash. The decode fix is jansel's PR #184102
(Fixes #175557), which preserves/clones cudagraph-pool-aliased inputs before the
dealloc frees them. We validated #184102 on a real `gpt2` `reduce-overhead` decode
(128 tokens, growing KV cache) on two torch builds: baseline crashes at the first
decode step, #184102 completes all tokens with cudagraphs still active. It also held
across 7 synthetic feedback topologies. One coverage gap remains (synthetic, source-
verified): when a compiled function splits into multiple cudagraph partitions and a
top-level input consumed only by a later partition is fed back from the pool, #184102
still crashes; we reported this. No real model has been shown to hit this topology, so
practical blast radius is unproven / likely small. #184102 is under active maintainer
review (eellison), not merged. So the earlier "decode is an unpatchable dead end"
framing is superseded: the never-free approach fails, but input-preservation patches
it. Validation gist: https://gist.github.com/Sahil170595/062d40cb18e2b2e27e99c1efbfa3ccdb

#### TR127: Long-Context Performance Characterization
**File:** `Technical_Report_127.md`
- VRAM spillover causes 25-105x latency cliffs (not quadratic attention)
- Ollama prefill scaling is sub-linear below spillover (b = 0.083-0.158)
- GQA models sustain 3-11x longer contexts than MHA models

#### TR128: Production Workload Characterization
**File:** `Technical_Report_128.md`
- NUM_PARALLEL is a no-op: 0/30 pairwise comparisons significant
- M/D/1 queueing theory deviates up to 20.4x from reality
- Streaming adds zero overhead (0/9 tests significant)

#### TR129: N-Agent Scaling Laws
**File:** `Technical_Report_129.md`
- Amdahl's Law fit: serial fractions s = 0.39-0.54 (R-squared > 0.97)
- Per-agent throughput at N=8: 17-20% of solo throughput
- Fairness: Jain's index >= 0.997

#### TR130: Serving Stack Comparison
**File:** `Technical_Report_130.md`
- vLLM 2.25x throughput advantage at N=8 via continuous batching
- TGI provides equivalent amortization but lower absolute throughput
- vLLM/TGI deliver 6-8x faster TTFT (22-35ms vs 163-194ms)

#### TR131: GPU Kernel Profiling
**File:** `Technical_Report_131.md`
- Overturns TR130: GPU memory bandwidth, not serving stack, is the bottleneck
- PyTorch Direct degrades 86.4% at N=8 (worse than Ollama's 82.1%)
- Memory bandwidth stress increases +74% at N=8 (Holm-surviving test)

#### TR132: In-Container GPU Profiling
**File:** `Technical_Report_132.md`
- Continuous batching mechanism quantified: 77-80% kernel count reduction
- Bandwidth-per-token reduction: 79-83% at N=8
- Amortization ratio: 4.7-5.8x (59-72% of theoretical 8:1 maximum)

#### TR133: Predictive Capacity Planner
**File:** `Technical_Report_133.md`
- 19,676 empirical records feeding lookup tables
- 4-gate pipeline: VRAM feasibility, quality gate, latency gate, budget gate
- 4/4 validation targets met: VRAM R-squared=0.968, throughput R-squared=0.859
- `chimeraforge plan` CLI shipped

---

## Conclusive Report Details

### Phase 1a Synthesis: Technical_Report_Conclusive_Phase1
**File:** `Technical_Report_Conclusive_Phase1.md` (2,826 lines)
- Covers TR108 through TR116: Python-to-Rust migration, multi-agent architecture, runtime selection, cross-model validation
- Six shippable decisions: Rust for production, dual Ollama mandatory, Tokio-default runtime, Gemma 3 for scaling, Python ceiling at ~86%, config transfer failure
- Extended Appendices: `Technical_Report_Conclusive_Phase1_Extended_Appendices.md` (1,171 lines)
- Executive Whitepaper: `Technical_Report_Conclusive_Phase1_Whitepaper.md` (214 lines)

### Phase 1b Synthesis: Technical_Report_Conclusive_Phase2
**File:** `Technical_Report_Conclusive_Phase2.md` (208KB)
- Covers TR117 through TR122
- Extended Appendices: `Technical_Report_Conclusive_Phase2_Extended_Appendices.md` (89KB)
- Executive Whitepaper: `Technical_Report_Conclusive_Phase2_Whitepaper.md` (8KB)

### Phase 2 Synthesis: Technical_Report_Conclusive_Phase3
**File:** `Technical_Report_Conclusive_Phase3.md` (433KB, 3,327 lines, 60 appendices)
- Covers TR123 through TR133
- Three stable conclusions, six shippable decisions, full artifact provenance
- Extended Appendices: `Technical_Report_Conclusive_Phase3_Extended_Appendices.md` (62KB)
- Executive Whitepaper: `Technical_Report_Conclusive_Phase3_Whitepaper.md` (15KB)

### Phase 3a Synthesis: Technical_Report_Conclusive_Phase4
**File:** `Technical_Report_Conclusive_Phase4.md` (2,571 lines)
- Covers TR134 through TR137: quantization-induced alignment erosion, concurrency invariance, backend-driven template divergence, cross-axis safety taxonomy
- 74,254 evaluated samples, 24-configuration deployment risk matrix (3 CRITICAL, 3 moderate, 18 low)
- Key finding: serving backend choice is a previously uncharted safety variable (41% of safety cost)
- Extended Appendices: `Technical_Report_Conclusive_Phase4_Extended_Appendices.md` (908 lines)
- Executive Whitepaper: `Technical_Report_Conclusive_Phase4_Whitepaper.md` (228 lines)

### Phase 3b Synthesis: Technical_Report_Conclusive_Phase5
**File:** `Technical_Report_Conclusive_Phase5.md` (2,590 lines)
- Covers TR138 through TR143: batch-inference safety attack surface — batch non-determinism, multi-turn jailbreak under quantization, many-shot/long-context jailbreak, cross-architecture refusal fragility (largest study), quality-safety correlation, cross-request leakage under continuous batching
- 306,996 samples synthesised across 6 reports; cross-cutting findings on alignment-type non-predictivity (p=0.942), output-instability predictivity (r=0.91), Q2_K universal vulnerability threshold, 88–92% directional unsafe-flip asymmetry
- Three shippable behavioural-screen results: RTSI calibrated (TR142), refusal fragility ≤0.94× safety-capability ratio (TR141), batch-perturbation safety floor (TR138 v2)
- Extended Appendices: `Technical_Report_Conclusive_Phase5_Extended_Appendices.md` (744 lines)
- Executive Whitepaper: `Technical_Report_Conclusive_Phase5_Whitepaper.md` (285 lines)
- Note: file naming reflects the 2026-05-28 integer-clean phase rename (Phase 4 file → Phase 5 file, Phase 4.5 file → Phase 6 file). Narrative content unchanged.

### Phase 6 Synthesis: Serving-State Safety Certification
**File:** `Conclusive_Phase6.md` (1,453 lines)
- Covers the seven executed serving-state safety reports — TR144 (Speculative Decoding × Safety / TAIS), TR145 (FP8 KV-cache single-config), TR146 (mechanistic-probe falsification, negative control), TR147 (Compile Reproducibility Index / CRI), TR148 (Judge Triangulation Protocol / JTP), TR149 (FP8 standardized batteries), TR152 (FP8 serving-state factorial)
- Consolidates them into a **five-layer serving-state safety certification protocol**: Layer 1 measurement validity (JTP, split 1a refusal-axis / 1b composite-harm), Layer 2 behavioural screens (RTSI inherited from TR142 + TAIS), Layer 3 compile integrity (CRI), Layer 4 scale validity (TR149 1B–3B anchor; TR151 7B–72B cloud-gated), Layer 5 serving-state validity (TR152 factorial)
- Headline deliverable: a serving-state-independent FP8 (E4M3) KV-cache certificate for 1B–4B instruction-tuned models on vLLM v0.19.1 — harmful-refusal-neutral across batch ∈ {1, 8, 32}, prefix-caching {on, off}, temperature ∈ {0.0, 0.7, 1.0} (TR152: 0 / 8,976 harmful-battery pairs discordant; FP8-interaction spread 2.99 pp inside ±3 pp), with one located footnote — a sub-1 pp Qwen-2.5 over-refusal lean on XSTest (per-family MH OR 3.878 [2.386, 6.302])
- TR146 is load-bearing as a negative control: four mechanistic probes all fail to distinguish safe from dangerous configs, so the entire arc stays behavioural
- Claim ladder C1–C10 (6 Supported / 2 Licensed / 2 Forbidden); whole executed substrate ran at $0 external API cost under the umbrella gate on a local Ollama judge cohort
- Cloud-gated envelope widenings (TR150 long-context, TR151 7B–72B scale, TR153 KV-method sweep) gated on the 2026-10-24 GO/NO-GO trigger
- Extended Appendices: `Conclusive_Phase6_Extended_Appendices.md` (1,585 lines) — per-TR appendix volume (Appendices A–G + cross-cutting X1–X4 + glossary)
- Executive Whitepaper: `Conclusive_Phase6_Whitepaper.md` (408 lines — includes the 5-layer protocol table, Qwen 2.5 1.5B worked example, and 8-objection reviewer FAQ)

---

## Economic Impact

### Phase 1 (TR119)
- Best cost: onnxruntime-gpu at $0.1279/1M tokens (on-demand)
- Spot pricing: $0.03868/1M tokens (69.8% savings)
- Lowest carbon: ~1.0 gCO2e/1M tokens

### Phase 2 (TR123)
- Best cost: $0.013/1M tokens (GPT-2/compile, chat blend)
- Best cost above 1B params: $0.047/1M tokens (LLaMA-3.2-1B/compile)
- Consumer TCO at 1B tokens/month: $153/yr to $561/yr
- AWS TCO at 1B tokens/month: $2,880/yr to $8,584/yr
- Consumer hardware saves 95.4% versus cloud on-demand
- Break-even: RTX 4080 ($1,200) pays for itself in 0.3-2.7 months at 10M requests/month

### Production Throughput Ceiling (Phase 2)
- Single-agent peak: 1.17 req/s (Ollama, llama3.2-1b)
- Multi-agent peak: 559 tok/s total (vLLM, llama3.2-1b, N=8)
- Per-agent at N=8: 17-20% of solo (Ollama); 56% efficiency retained (vLLM)

---

## Repository Structure

```
PublishReady/reports/
├── README.md (this file)
│
├── Phase 1: Foundation (TR108-TR122)
│   ├── Technical_Report_108.md
│   ├── Technical_Report_109.md
│   ├── Technical_Report_110.md
│   ├── Technical_Report_111_v2.md
│   ├── Technical_Report_112_v2.md
│   ├── Technical_Report_113.md
│   ├── Technical_Report_114_v2.md
│   ├── Technical_Report_115_v2.md
│   ├── Technical_Report_116.md
│   ├── Technical_Report_117.md
│   ├── Technical_Report_117_multi_agent.md
│   ├── Technical_Report_118_v2.2.md
│   ├── Technical_Report_119.md
│   ├── Technical_Report_120.md
│   ├── Technical_Report_121v1.md
│   └── Technical_Report_122.md
│
├── Phase 2: Deployment Framework (TR123-TR133)
│   ├── Technical_Report_123.md
│   ├── Technical_Report_124.md
│   ├── Technical_Report_125_v2.md
│   ├── Technical_Report_126.md
│   ├── Technical_Report_127.md
│   ├── Technical_Report_128.md
│   ├── Technical_Report_129.md
│   ├── Technical_Report_130.md
│   ├── Technical_Report_131.md
│   ├── Technical_Report_132.md
│   └── Technical_Report_133.md
│
├── Phase 3: Safety Alignment (TR134-TR143, TR146)
│   ├── Technical_Report_134_v2.md
│   ├── Technical_Report_135.md
│   ├── Technical_Report_136.md
│   ├── Technical_Report_137.md
│   ├── Technical_Report_138.md
│   ├── Technical_Report_138_v2.md
│   ├── Technical_Report_138_Study_D_Addendum.md  ← ICML camera-ready supplement (batch-invariant kernel ablation)
│   ├── Technical_Report_139.md
│   ├── Technical_Report_140.md
│   ├── Technical_Report_141.md  ← largest study (127,224 records v3.1, 18 models)
│   ├── Technical_Report_142_v2.md
│   ├── Technical_Report_143.md
│   └── Technical_Report_146.md  ← mechanistic safety probing (5,100 forward passes); negative control for Phase 6
│
├── Phase 6: Serving-State Safety Certification (TR144, TR145, TR148, TR149, TR152)
│   ├── Technical_Report_144.md  ← TAIS / speculative decoding × safety (1,925 lines; 64,855 samples)
│   ├── Technical_Report_145.md  ← FP8 KV-cache single-config null (1,465 lines; 24,054 records)
│   ├── Technical_Report_148.md  ← JTP dual-axis multi-judge reliability (1,556 lines; 41,272 fresh judge rows)
│   ├── Technical_Report_149.md  ← FP8 standardized batteries (1,303 lines; 7,578 primary + 22,734 judge)
│   └── Technical_Report_152.md  ← FP8 serving-state factorial v2 (1,341 lines; 45,000 records, 20,754 paired)
│
├── v2 Expansion Reports
│   ├── Technical_Report_125_v2.md
│   ├── Technical_Report_134_v2.md
│   └── Technical_Report_142_v2.md
│
├── v3 AWQ/GPTQ Cross-Method Reports
│   ├── Technical_Report_125_v3.md
│   ├── Technical_Report_134_v3.md
│   └── Technical_Report_142_v3.md
│
├── Conclusive Reports
│   ├── Technical_Report_Conclusive_Phase1.md
│   ├── Technical_Report_Conclusive_Phase1_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_Phase1_Whitepaper.md
│   ├── Technical_Report_Conclusive_Phase2.md
│   ├── Technical_Report_Conclusive_Phase2_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_Phase2_Whitepaper.md
│   ├── Technical_Report_Conclusive_Phase3.md
│   ├── Technical_Report_Conclusive_Phase3_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_Phase3_Whitepaper.md
│   ├── Technical_Report_Conclusive_Phase4.md            ← Phase 3a Synthesis (TR134-137); file renamed 2026-05-28 (was Phase 3.5)
│   ├── Technical_Report_Conclusive_Phase4_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_Phase4_Whitepaper.md
│   ├── Technical_Report_Conclusive_Phase5.md            ← Phase 3b Synthesis (TR138-143 attack surface); file renamed 2026-05-28 (was Phase 4)
│   ├── Technical_Report_Conclusive_Phase5_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_Phase5_Whitepaper.md
│   ├── Conclusive_Phase6.md                             ← Phase 6 Synthesis: serving-state safety certification (5-layer protocol; TR144-152); file renamed 2026-05-28 (was Phase 4.5)
│   ├── Conclusive_Phase6_Extended_Appendices.md
│   └── Conclusive_Phase6_Whitepaper.md
│
├── Historical (Superseded)
│   ├── Technical_Report_111.md
│   ├── Technical_Report_112.md
│   ├── Technical_Report_114.md
│   ├── Technical_Report_115.md
│   ├── Technical_Report_118.md (+ v2.1)
│   ├── Technical_Report_119v1.md
│   └── Technical_Report_121.md
│
├── Legacy (moved to legacy/)
│   ├── Technical_Report_118.md
│   └── Technical_Report_118_v2.1.md
│
└── Model Benchmarks
    └── gemma3/
        └── Gemma3_Benchmark_Report.md
```

---

## Hardware Baseline

All measurements on a single fixed baseline:
- **GPU:** NVIDIA RTX 4080 Laptop GPU (12 GB GDDR6, 256-bit, 432 GB/s, AD104)
- **CPU:** Intel Core i9-13980HX (24 cores, 32 threads)
- **RAM:** 64 GB DDR5-4800
- **OS:** Windows 11 + WSL2/Ubuntu 22.04 for Docker/Linux workloads

---

## Research Questions Answered

### Phase 1

1. **Is Rust faster than Python for LLM inference?**
   Yes — 15.2% faster throughput, 58% faster TTFT, 67% less memory (TR112_v2)

2. **Does Rust's single-agent advantage carry over to multi-agent?**
   Yes — Rust exceeds Python by +2.48pp mean efficiency (TR114_v2)

3. **Which Rust async runtime is optimal?**
   Tokio-default — 98.72% mean, 1.21pp sigma (TR115_v2)

4. **Is dual Ollama architecture necessary?**
   Yes — reduces contention from 63% to 0.74% (TR113 -> TR114_v2)

### Phase 2

5. **Does backend choice affect quality?**
   No — 0/7 ANOVA significant at temp=0 across 5 models (TR124)

6. **What quantization level should you default to?**
   Q4_K_M — recommended default across tested models, at most -4.1pp accuracy loss, 30-67% cost savings (TR125)

7. **Does torch.compile help?**
   Prefill only, Linux only, 24-60% speedup; decode crashed 100% of the time on stock PyTorch, now fixed upstream by #184102 (in review), validated on real decode; #175562 landed (TR126)

8. **What limits long-context performance?**
   VRAM spillover, not quadratic attention — 25-105x latency cliffs at capacity (TR127)

9. **Is the serving stack or the GPU the bottleneck at scale?**
   The GPU — memory bandwidth, not serving stack logic, is the constraint. Continuous batching amortizes it 4.7-5.8x (TR131, TR132)

10. **Can you predict deployment configurations without running benchmarks?**
    Yes — `chimeraforge plan` uses empirical lookup tables with VRAM R-squared=0.968, throughput R-squared=0.859 (TR133)

### Phase 3

11. **Does quantization degrade safety disproportionately to capability?**
    Yes, at extreme quant levels — safety degrades up to 13.9x faster than quality at Q3_K_S; effect is model-family-specific (TR134, TR142)

12. **Does running multiple concurrent agents degrade safety?**
    No — concurrency is the one safe axis; all slopes zero, I-squared = 0.0%, confirmed across 39,060 samples (TR135)

13. **Does the serving backend affect model safety?**
    Yes — backend accounts for 41% of total safety cost; Llama 1B shows 23pp safety drop Ollama→FP16 from chat template divergence (TR136)

14. **Does batch non-determinism introduce safety failures?**
    Yes — batch perturbation produces measurable safety instability (0.16% human-adjudicated genuine rate, TR138). Fragility varies 0.00%-2.39% across 15 models; alignment type is not predictive (p=0.942), output instability is (r=0.91) (TR141)

15. **Does quantization amplify multi-turn jailbreak susceptibility?**
    Yes — all 8 strategy ANOVAs reject quant-independence; qwen2.5-1.5b/Q2_K reaches 100% ASR on three attack strategies (TR139)

16. **Are quality benchmarks sufficient to monitor safety?**
    No — quality and safety degradation paths are uncorrelated and direction-reversed across architectures (TR142)

17. **Do AWQ and GPTQ preserve the same safety guarantees as GGUF k-quants?**
    Not reliably — at matched ~4-bit precision, AWQ and GPTQ introduce model-specific hidden-danger regimes not predicted by GGUF thresholds. Format-specific safety evaluation is required (TR134 v3, TR142 v3)

18. **Can mechanistic interpretability probes predict quantization-induced safety failure?**
    No — first-token entropy, refusal direction preservation, calibration drift, and safety-neuron error magnitude all fail to distinguish safe from dangerous configs. Safety neurons absorb 1.40x disproportionate error universally; the damage is necessary but not sufficient for behavioral failure (TR146)

### Phase 5

19. **Is a `torch.compile` benchmark number reproducible across software-stack bumps?**
    Not without pinning Triton minor version. On the same RTX 6000 Ada with the same model and code, a Triton 3.3.1 → 3.4.0 upgrade flips compiled prefill from −77% (fast) to 0% (neutral) and compiled decode from 80% crash to 0% crash, with Cohen's d of 15–49 on compile-vs-eager prefill contrasts. The eager arm is flat (|d|≤0.15) across versions, ruling out harness drift. Any `torch.compile` benchmark that does not report its Triton minor version is an unfalsifiable claim (TR147 v4 SS8).

20. **Are the decode-stability and prefill-speedup-loss effects coupled or independent?**
    Independent. The decode stability win is a PyTorch-side assertion fix (PR #175562); the prefill speedup loss is a Triton-side codegen regression (PR #7138 + register-allocator rewrites). They landed at the same software-line boundary by coincidence, not causation. The stability fix is free; the speedup loss is a separable Triton regression already partially remediated in 3.6.0 (TR147 v4 SS8.7).

21. **Does a bigger GPU rescue the compiled-decode failure?**
    No. A100 80 GB (sm_80 Ampere) amplifies the failure — 100% decode crash at every tested token length, worse than Ada's 3.6% floor. Hardware class alone is not the variable (TR147 v3).

22. **What combination of cache and compile mode delivers both speed and stability?**
    On dense FP16: none of the tested cells. StaticCache + default is stable but ~2% slower than eager on decode. DynamicCache + reduce-overhead is fast on prefill but crashes 80% on decode. The only 0%-crash `reduce-overhead` decode cell in the entire TR147 matrix is Ada + qwen2.5-7b-AWQ-4bit + StaticCache — 4-bit quantization accidentally stabilizes compilation (TR147 v4 SS7, SS9.0).

### Phase 6

23. **Is speculative decoding via typical-acceptance sampling safety-equivalent to rejection sampling?**
    Yes — TR144's E1–E5 expansion (48,072 expansion-probe samples on top of 16,783 core paired) finds max |Cohen's h| = 0.024 across matched draft+target pairs under rejection vs typical acceptance, well below the ±0.1 null cutoff calibrated from the same expansion. TAIS (Typical-Acceptance Invariance Screen) ships as the named method; integrated into NeurIPS submission #3738 (TR144).

24. **Is FP8 (E4M3) KV-cache quantization safety-neutral under realistic serving states?**
    Yes for harmful prompts, with one located footnote on benign-edge content. TR152 v2's 45,000-record `fp8_anchored_star` factorial finds **0 / 8,976 discordant pairs** on HarmBench / JBB / StrongREJECT across 5 models × 6 contexts × 3 harmful batteries, 0 / 120 cells Holm-significant, 117 / 120 TOST-equivalent at ±3 pp, FP8-interaction spread 2.99 pp inside the band. TR149's standardized-battery replication finds 12 / 12 TOST-equivalent and cross-battery MH pooled OR 0.8065 [0.38, 1.70]. The only footprint is a sub-1 pp Qwen-2.5 over-refusal lean on XSTest (per-family MH OR 3.878 [2.386, 6.302], sign-test p ≈ 0.0004). Claims bounded to 1B–4B instruction-tuned models on vLLM v0.19.1 at batch ∈ {1, 8, 32}, prefix-caching {on, off}, temperature ∈ {0.0, 0.7, 1.0} (TR145 / TR149 / TR152).

25. **Do safety-specialist judges (Llama-Guard, ShieldGemma) measure the same thing as general LLM judges (Gemma 3, Llama 3.1)?**
    No — they measure two orthogonal axes. TR148 v2's 5-judge corpus-scale re-judging of TR145 reveals safety-specialist judges **anti-correlate** with general LLM judges across 4 large-n pairs (cross-axis κ = −0.13 to −0.26) because they label composite prompt+response harm, not response-only refusal. The Phase 6 protocol therefore splits Layer 1 into 1a refusal-axis JTP (general judges) + 1b composite-harm-axis screen (safety specialists). Single-axis judge triangulation is insufficient at the corpus scale where this finding first emerges (TR148 v2 SS4).

26. **Can multi-judge triangulation be done with one cross-LLM pair?**
    Only if that pair clears the JTP robust threshold of κ ≥ 0.70. TR148 v2's gemma3 × llama3.1 pair lands at κ = 0.6917 at n = 12,809 — 0.0083 below threshold → TRIANGULATE bucket. Downstream Phase 6 reports (TR149, TR152) therefore require multi-judge majority-vote rather than single cross-LLM sign-off. The bridge paper Layer 1a anchors on this verdict (TR148 v2).

27. **Can mechanistic interpretability probes substitute for behavioural screens under serving-state perturbation?**
    No. TR146's four probes (first-token entropy, refusal-direction cosine, calibration drift, safety-neuron error magnitude) all fail to distinguish safe from dangerous configs. Safety neurons absorb 1.40× disproportionate quantization error universally (p<0.0001), but the damage is necessary, not sufficient, for behavioral failure (regime p=0.98). TR146 is the load-bearing negative control for Phase 6 — the entire arc stays behavioural because mechanistic probing was checked first and failed (TR146).

---

## Reading Guide

### For Researchers
1. Start with **TR108-TR110** (Python baselines)
2. Study **TR112_v2** and **TR114_v2** (Rust vs Python)
3. Review Phase 2 starting with **TR123** (economics) through **TR133** (capacity planning)
4. Review Phase 3 starting with **TR134** (quantization x safety) through **TR142** (quality-safety correlation)
5. Review Phase 6 starting with **TR148** (judge methodology / JTP dual-axis) → **TR149** (standardized batteries) → **TR152** (serving-state factorial) → **TR144** (TAIS); TR146 is the load-bearing mechanistic negative control
6. Read the **Conclusive Reports** for synthesis and cross-TR analysis (Phase 6 main: `Conclusive_Phase6.md`)

### For Engineers
1. **Single-agent deployment:** TR112_v2 (language choice) + TR125 (quantization)
2. **Multi-agent deployment:** TR129 (scaling laws) + TR130 (serving stacks)
3. **Capacity planning:** TR133 + `chimeraforge plan` CLI
4. **Compilation:** TR126 (what works, what crashes)
5. **Safety-critical deployment:** TR137 (safety tax synthesis) + TR141 (batch safety, cross-architecture) + TR146 (mechanistic probing)
6. **Backend safety validation:** TR136 (backend safety consistency)
7. **Jailbreak risk assessment:** TR139 (multi-turn) + TR140 (many-shot)
8. **AWQ/GPTQ safety:** TR134 v3 + TR142 v3 (format-specific safety evaluation)
9. **FP8 KV-cache serving-state certification:** TR149 (standardized batteries) + TR152 (full serving-state factorial v2) — bounds for 1B–4B instruction-tuned on vLLM v0.19.1
10. **Speculative-decoding safety:** TR144 (TAIS named method, matched draft+target Cohen's h ±0.1 null cutoff)
11. **Judge selection for safety eval:** TR148 v2 (5-judge JTP, dual-axis methodology — choose general LLM judges for refusal-axis, safety specialists for composite-harm-axis; do not pool the two)

### For Decision Makers
1. **Rust vs Python Decision:** `Technical_Report_Conclusive_Phase1_Whitepaper.md` (language, architecture, runtime, model)
2. **Phase 2 Whitepaper:** `Technical_Report_Conclusive_Phase3_Whitepaper.md` (15KB, 6 decisions)
3. **Phase 3a Whitepaper (safety tax):** `Technical_Report_Conclusive_Phase4_Whitepaper.md` (228 lines, safety decision card)
4. **Phase 3b Whitepaper (attack surface):** `Technical_Report_Conclusive_Phase5_Whitepaper.md` (285 lines)
5. **Serving-State Safety Certification (Phase 6):** `Conclusive_Phase6_Whitepaper.md` (408 lines — FP8 KV-cache certificate, 5-layer protocol, Qwen 2.5 1.5B worked example, 8-objection reviewer FAQ, six shippable decisions)
6. **Decision Matrix:** See Phase 2 Deployment Decisions table above
7. **Cost Analysis:** TR123 ($/token) + TR119 (energy/carbon)

---

**Last Updated:** 2026-05-28
**Total Reports:** 45+ completed TR versions including the Phase 6 serving-state safety arc (TR144, TR145, TR148, TR149, TR152) + TR147 four-subversion benchmarking-integrity track + 3 v2 expansions + 3 v3 AWQ/GPTQ expansions + TR138 Study D Addendum (ICML camera-ready supplement) + 21 conclusive/whitepaper documents (7 phase groups × report / appendices / whitepaper) + 7 historical/superseded + 3 legacy + model benchmarks
**Total Measurements:** **~1,041,000 primary + judge measurements on disk across 45+ TRs** (TR152 v2 judging completed 2026-05-28 00:06 ET; the prior "≈ ~1.04M once judging completes" projection has landed as a firm number); see `BANTERHEARTS_MEASUREMENT_COUNT.md` (2026-05-28 supplement) for canonical reconciliation with verified row counts. Prior baselines: ~906K (2026-05-27 with TR152 v2 judging pending), ~800K (2026-05-14 post-TR149), ~770K (2026-05-13 post-TR148 v2), ~728K (2026-05-08 post-TR145).
**Quantization Formats Evaluated:** GGUF (FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_S, Q2_K), AWQ INT4, GPTQ INT4, FP8 E4M3 KV-cache (vLLM v0.19.1)
**Models Evaluated:** 20+ unique models across 10+ architecture families (360M to 14.8B parameters), including phi-3-mini-4k and llama3.2-1b added in TR152 v2
**Software Stacks Evaluated:** PyTorch 2.8 / 2.10 / nightly; Triton 3.3.1 / 3.4.0 / 3.6.0; vLLM v0.19.1 Docker pinned (Phase 6 standard); Ollama via native runner; Docker for AWQ/GPTQ pipelines
**Safety Judges Evaluated:** regex baseline + gemma3:12b + llama3.1:8b-instruct-q8_0 + shieldgemma:9b + llama-guard3:8b + 100-record gpt-4o calibration anchor (TR148 v2); all Phase 6 batteries gated via `--skip-openai-judge` umbrella → $0 external API cost
**Safety Corpora Evaluated:** HarmBench-400, JailbreakBench-100, StrongREJECT-313, XSTest-450 (TR149 / TR152 standardized batteries); advbench, jailbreakbench, jailbreak_amplification, bbq_bias (Phase 3 attack surface)
**Hardware:** NVIDIA RTX 4080 Laptop 12GB (primary; TR145 / TR149 / TR152 v1+v2 / Phase 3 attack surface), Colab T4 16GB (v2 expansion), RunPod RTX 6000 Ada 48GB (7B quantization + TR147 v1-v2-v4 Ada), RunPod A100-SXM4-80GB (TR147 v3 + v4 B1-B2 + TR144 E1-E5 expansion + 70B safety gates per DATA_INVENTORY), RunPod L40S 48GB (TR140 v2 controls), Docker (AWQ/GPTQ pipelines)
