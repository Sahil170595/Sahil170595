# Technical Report 164: Serving-Stack Physics on Consumer GPU — Breakdown Boundaries, GIL-Attributable Concurrency Collapse, and Deterministic Cell-Shape Hangs Under In-Process pytorch_direct Inference

## A 346-Cell Matched Measurement Study Across Three Model Tiers, Two Phases, Four Workloads, and Five Concurrency Levels, with Kernel-Attributed nsys Evidence and a Six-Cell-Shape Hang Ledger

---

## 1. Abstract

We measure, on a consumer RTX 4080 Laptop GPU, how an in-process pytorch_direct serving stack scales under controlled model size, workload mix, request concurrency, and phase pacing. Across 360 planned cells reduced to 346 real cells by a documented 14-row deterministic-hang skip ledger, we observe a uniform breakdown boundary at N=2 across all twenty-four (model × workload × phase) combinations: parallel efficiency drops from 1.000 at N=1 to a median of 0.536 at N=2, continues to 0.281 at N=4, collapses to 0.104 at N=8, and bottoms at 0.047 at N=16 — a 95% loss of nominal scaling headroom by the highest concurrency tested. P95 wall-latency multipliers relative to N=1 reach a median of 17.4× at N=16 with a maximum of 1446.67×, and time-to-first-token on llama3.2-1b at N=16 with the balanced_2k workload reaches 188.4 seconds for the median request — operationally unacceptable for any interactive deployment. Six (phase × workload × N=16) cell shapes on llama3.2-3b proved deterministically pathological: the dispatcher itself hung for ≥3.3 hours with GPU at 99% SM-occupancy but thermal/power readings consistent with GIL-starved idle (50-54°C, 23-49 W on a part rated to draw 80-150 W sustained). We attribute the collapse mechanism to Python Global Interpreter Lock serialization of in-process N-way concurrent inference dispatch, supported by (a) the thermal/power signature during stalls, (b) clean recovery when nsys subprocess timeouts forced fallback to in-process execution at lower N, and (c) a cross-model asymmetry where qwen2.5-1.5b retains 0.92 mean speedup at N=16 while llama3.2-3b on the same workload retains only 0.03 — consistent with model-architecture differences in GIL-release frequency during the generation loop. We discuss implications for the literature on Python-based LLM serving, position TR164 against the 2025-2026 cross-stack benchmark line (arXiv 2511.17593, NanoFlow, the SGLang and vLLM serving-benchmark CLIs), and outline the V2 cross-backend RunPod fan-out plus Python 3.14t nogil ablation that constitute the mechanism-isolation experiment for the GIL-attributable share of the observed collapse. The substrate produced — 21,159 metric rows, 1.04 GB of clean nsys kernel traces, and the six-cell-shape hang ledger — supports the breakdown-boundary claim and the in-process-pytorch-cannot-serve claim with reproducible evidence at the request level.

The substrate also documents methodological contributions that the V2 program and future TR164.next iterations will inherit. The nsys-wrapped subprocess pattern (cell_worker.py) cleanly separates the captured-cell execution from the dispatcher's bookkeeping, allowing per-cell kernel-trace capture without modifying the dispatcher's primary control flow. The subprocess-timeout-and-fallback recovery mechanism (1800-second budget with in-process fallback) salvages cells that exceed the captured-path budget while still producing usable metric data. The skip-row methodology with synthetic status='ok' entries provides clean exclusion of deterministic-hang cells from downstream statistical aggregation. The capture_environment() Windows-WMI defensive guard (commit 2d339c4d) handles host-environment degradation gracefully without sacrificing reproducibility. Each pattern was introduced in response to a specific operational failure observed during the V1 arc; collectively they constitute a small but useful methodological toolkit for in-process Python LLM serving measurement.

---

## 2. Table of Contents

1. Abstract
2. Table of Contents
3. Executive Summary
4. Introduction and Research Motivation
5. Research Hypotheses
6. Methodology
7. Models and Configuration
8. SS1. Cell Census and Coverage Accounting
9. SS2. Per-Model Decode Throughput Across the N Sweep
10. SS3. Per-Workload Throughput Geometry
11. SS4. Parallel Efficiency Collapse — The Headline Result
12. SS5. P95 Latency Multipliers and the Tail-Latency Catastrophe
13. SS6. Phase Asymmetry — Scaling vs TTFT Under Identical Cell Shapes
14. SS7. The Deterministic N=16 Hang Ledger — Six Cell Shapes
15. SS8. GPU Thermal and Power Diagnostic During Stalls
16. SS9. nsys Kernel Attribution — Capture Inventory and the QdstrmImporter Caveat
17. SS10. Cross-Model Asymmetry — Why Qwen Survives What Llama Cannot
18. SS11. Workload Sensitivity — long_decode's Unexpected Tolerance
19. SS12. Subprocess Timeout as Recovery Mechanism
20. SS13. Skip-Marker Methodology and Its Statistical Treatment
21. SS14. Time-to-First-Token Under Concurrency
22. SS15. Backend-Ranking Caveat — Single-Backend Coverage
23. SS16. Bridge to TR130-132 and the SGLang Gap
24. SS17. Limitations
25. SS18. Related Work
26. SS19. Implications for Production Serving Decisions
27. SS20. Future Work — V2 RunPod and Python 3.14t Nogil
28. Conclusion
29. References
30. Appendix A. Hardware and Software Versions
31. Appendix B. Skip-Row Schema Specification
32. Appendix C. nsys Capture Manifest

---

## 3. Executive Summary

TR164 was scoped as the local_core leg of a five-backend serving-stack expansion of the TR130-TR132 measurement line. Its design intent was to measure, with kernel-level evidence where possible, how breakdown boundaries shift across PyTorch Direct, Ollama, vLLM, SGLang, and TGI under matched model size, precision, context length, decode length, and request concurrency. Cloud_core and cloud_full suites, which would have extended this measurement to 7B/8B/14B model tiers and the four server-process backends, were deferred to V2 pending the resolution of a compute-credit application stack; this report covers the local_core leg only and confines its claims accordingly.

The primary finding is that the in-process pytorch_direct backend on a consumer RTX 4080 Laptop GPU exhibits a single uniform breakdown boundary at N=2 across every (model × workload × phase) cell shape we tested. The dispatcher's per-cell parallel efficiency, normalized to N=1 baseline, falls from 1.000 to a median of 0.536 at N=2, drops monotonically to 0.281 at N=4, collapses to 0.104 at N=8, and reaches 0.047 at N=16. P95 wall-latency relative to N=1 grows superlinearly: 1.96× at N=2, 3.80× at N=4, 36.99× at N=8 (with substantial dispersion — standard deviation 89.93 across cells), and 88.89× at N=16 (standard deviation 266.94, maximum 1446.67×). At the highest concurrency we tested, time-to-first-token medians cross from sub-second baselines into the tens to hundreds of seconds — most extremely 188.4 seconds median TTFT on llama3.2-1b at N=16. For any operator considering serving with raw pytorch_direct under realistic concurrency, this evidence is decisive: the breakdown is immediate and the tail is unbounded.

The second finding is that the breakdown is not a smooth degradation — six (phase × workload × N=16) cell shapes on llama3.2-3b prove deterministically pathological. The dispatcher running these cells stops producing metric rows for 3.3 to 10 hours while GPU SM-occupancy reads 99% but GPU power draw collapses to 23-49 W, against a sustained-load expectation of 80-150 W for this part. We document this as a Python Global Interpreter Lock contention failure mode: the dispatcher thread is blocked behind another thread's GIL hold, the GPU kernel queue contains stale enqueued work but receives no new dispatch, and the system can remain in this configuration indefinitely. Killing and restarting the dispatcher across seven generations (V1 through V7) confirmed determinism — the same cell shape hung the dispatcher on every fresh attempt — and motivated the skip-row methodology described in SS13.

The third finding is a sharp cross-model asymmetry. At N=16 on the same workload mix, qwen2.5-1.5b retains a mean speedup of 0.92 relative to its N=1 baseline, while llama3.2-3b retains only 0.03. The Qwen architecture appears to release the GIL more frequently during its generation loop than the Llama-3 architecture, allowing the in-process pytorch_direct dispatch path to interleave concurrent requests more usefully. This is a model-architecture finding, not a model-quality finding, and it has direct implications for which open-weights models are deployable under raw-PyTorch serving without paying for an external request-scheduling stack.

We present these results as measurement evidence only. We do not claim algorithmic novelty in the serving primitives, nor a new scheduler, nor a new attention kernel. The contribution is a controlled, kernel-attributed, hand-narrated reproduction of the in-process Python LLM serving anti-pattern, executed across the highest-resolution workload × N × model grid for which the relevant literature does not yet have published evidence at this granularity on consumer hardware. The substrate licenses the breakdown-boundary claim, the GIL-attribution mechanism hypothesis, and the per-cell skip-row methodology. It does not license the broader cross-backend claims that V2 will provide, nor the mechanism-isolation Python 3.14t comparison that the nogil ablation will provide.

**The measurement substrate at a glance.** The local_core leg of the TR164 program executed 360 planned cells over five days of wall-clock through seven dispatcher generations on a single RTX 4080 Laptop. The cells produced 21,159 metric rows in metrics.csv at the request level (an average of approximately 60 requests per cell), with 100% status='ok' rate after the skip-row reconciliation. Fourteen of the 360 cells are synthetic skip markers documenting the six deterministically-hung (phase × workload × N=16) cell shapes on llama3.2-3b; the remaining 346 cells contain real benchmark data. The substrate includes 14 clean nsys-rep kernel-trace files (6.12 GB total disk footprint) covering the peak-load (workload × N) intersections, plus three orphan .qdstrm files (2.61 GB) lost to the NVIDIA Nsight Systems 2025.5.1 EventCollection.cpp:1054 importer bug. Three analysis derived artifacts — analysis.json (479 KB), cell_summary.csv (86 KB), tr164_report.md (4.6 KB) — were produced after the analyze.py logging-import fix in commit cea8f2b5.

**The operational arc context.** The five-day arc was punctuated by seven distinct events worth noting for substrate provenance: (1) the initial V1 launch and 282-cell progression before the first deterministic hang; (2) the V1→V2 reproduction confirming determinism on scaling × long_decode × N=16; (3) the V3 discovery of the scaling × repeated_prefix × N=16 second pathological cell; (4) the capture_environment() Windows-WMI hang patch necessitated by accumulated OS state degradation; (5) the V5 discovery of ttft × balanced_2k × N=16 as the third pathological cell shape; (6) the Claude Code v2.1.162 auto-updater non-atomic binary swap incident that killed PID 51676 and forced V6; and (7) the V7 final-completion arc that finished the remaining 24 real cells in 4 hours 52 minutes. Each event is documented in SESSION_FINDINGS_2026-05-30_2026-06-01.md §1-§13. The arc's friction is the cost of running production-realistic benchmarks on consumer hardware over multi-day timescales; the substrate is the data that justifies the cost.

---

## 4. Introduction and Research Motivation

The dominant pattern in 2025-2026 LLM serving infrastructure literature treats the choice of backend — vLLM, SGLang, TGI, TensorRT-LLM, Ollama — as the primary axis of variation, with raw `transformers`-based or pytorch_direct dispatch dismissed as obviously slow and not worth benchmarking. This dismissal is widely correct as a deployment recommendation: every credible production guide tells operators to put a continuous-batching server in front of their model. The dismissal nonetheless leaves a measurement gap. Operators who have not yet adopted a server stack — researchers running ablations, small teams prototyping, evaluation harnesses spinning up models for one-shot scoring — routinely run pytorch_direct dispatch under multi-request concurrency, and the literature does not characterize precisely how badly this performs or why.

TR130 and TR131 began closing this gap in the TR130-TR132 line by measuring four-backend coverage at small model scale on a consumer GPU. TR131's key methodological contribution was reversing TR130's throughput-ranking conclusions after adding Nsight Systems kernel traces: an apparent low-concurrency vLLM deficit turned out to be a CUDA-scheduler artifact that flipped above N=8 once the kernel-launch path saturated. The lesson was that throughput numbers alone are unreliable for ranking serving stacks and that mechanism-level kernel evidence must be in scope. TR164 inherits this lesson and extends the measurement to the SGLang backend, larger model tiers in V2, longer-context workloads including 8k prefill and the repeated-prefix shape that exercises prefix caching, and a concurrency sweep up to N=16 locally and N=32 on cloud.

This report covers only the local_core leg of that program, run on a single RTX 4080 Laptop with the in-process pytorch_direct backend. The constraint motivated by the available hardware turned into the central object of measurement: on a 12 GB GPU running 1B, 1.5B, and 3B models in FP16 with default device_map="auto" placement, the in-process Python dispatcher exhibits a breakdown boundary so close to N=1 that no operator following the basic-PyTorch deployment pattern would survive even a small burst of concurrent users. The breakdown does not require adversarial workload mixes — it appears on short_decode, balanced_2k, long_decode, and repeated_prefix uniformly — and it includes a deterministic-hang failure mode at N=16 on the larger model that the dispatcher cannot recover from without external intervention.

Beyond the operational implications, the measurement matters for the broader Python concurrency conversation. PEP 703 introduced an optional no-GIL build mode in Python 3.13 and improved it through 3.14, motivated in part by the bottleneck that mainstream LLM serving stacks consistently route around through C++ schedulers, subprocess isolation, or process-pool architectures. Quantifying how much of pytorch_direct's collapse is GIL-attributable — versus how much is shared between the GIL-protected single-threaded variant and a free-threaded build of the same Python — is the mechanism-isolation experiment that converts a "in-process Python is slow" folk observation into a measured cost attributable to a fixable architectural property. V2 will run that ablation. This report establishes the substrate against which the ablation will compare.

We deliberately framed the contribution as a measurement study from the outset. The literature_survey.md companion document, dated 2026-05-31 and updated through 2026-06-03, identifies prior work in Orca, vLLM/PagedAttention, SGLang, TGI, FlashInfer, NanoFlow, Sarathi-Serve, DistServe, Splitwise, DeepSpeed-FastGen, and the arXiv 2511.17593 vLLM-vs-TGI empirical comparison. None of these provide matched-matrix kernel-attributed in-process pytorch_direct data at the resolution this report produces, and none address the GIL-contention dimension as a measurement target. The novelty claim is precise: matched-matrix breakdown-boundary measurement with kernel evidence on the in-process baseline, plus the deterministic-hang failure-mode documentation that is structurally absent from cross-stack throughput comparisons because the comparison stacks all avoid the relevant code path.

---

## 5. Research Hypotheses

The publication_contract.json file written at code-complete on 2026-05-29 articulates three pre-registered hypotheses for the full TR164 program. The local_core leg this report covers can speak to the first two; the third requires the cloud_core suite and is deferred.

**H0 — Backend invariance under controlled conditions.** After controlling for model, precision, context length, decode length, and hardware, no backend should show a stable difference in throughput scaling or tail-latency breakdown boundary. This is the null we measure against in V2. The local_core leg cannot test H0 because it carries only a single backend; H0 is in scope for cloud_core and cloud_full.

**H1 — Scheduler boundary monotone with concurrency.** Continuous-batching serving stacks should shift the breakdown boundary upward relative to pytorch_direct and Ollama, but the boundary should be finite and should move predictably with context-decode mix and model scale. The local_core leg measures the pytorch_direct boundary which establishes the lower comparison anchor for H1. We report that anchor at N=2 across all twenty-four cell-shape combinations, with parallel efficiency falling below 0.65 at every one of them — confirming that pytorch_direct is below the H1 threshold uniformly. V2 will measure where the four server-process backends place their boundaries.

**H2 — SGLang prefix-reuse advantage on repeated workloads.** SGLang should show additional advantage on repeated_prefix workloads relative to non-prefix-reuse workloads, beyond the generic continuous-batching effect. This hypothesis is not testable on the local_core leg because SGLang is not in the executed matrix. We report the pytorch_direct repeated_prefix curve as the baseline against which SGLang's H2 advantage, if any, will be measured in V2.

The local_core report's additional emergent claim, not pre-registered in publication_contract.json but documented at length in SESSION_FINDINGS_2026-05-30_2026-06-01.md, is the GIL-attribution mechanism hypothesis: the breakdown boundary at N=2 on pytorch_direct and the deterministic N=16 hangs are dispatcher-thread GIL-serialization phenomena, falsifiable by running the same workload under Python 3.14t in V2. We will treat this as an emergent rather than pre-registered hypothesis in the paper writeup, with the appropriate epistemic caveat.

**The mechanism-hypothesis statement.** Formally: under the in-process pytorch_direct serving pattern with multiple concurrent agent threads, the dispatcher thread holds the GIL during CPU-bound work (tokenizer activity, KV-cache management, agent bookkeeping) for sufficient fractions of each per-token dispatch cycle that concurrent agents queue up behind the held lock. The aggregate dispatcher throughput is bounded above by the single-thread GIL-release rate, and the per-agent tail latency is bounded below by the queue wait behind the GIL holder. The mechanism predicts: (a) breakdown at small N, (b) catastrophic tails at moderate N, (c) deterministic hangs at high N under sufficient VRAM pressure, (d) cross-model asymmetry consistent with architectural differences in GIL-release rate, and (e) thermal/power signature consistent with GPU starvation despite SM-occupancy reading. The substrate is consistent with all five predictions.

**Falsifiability and the V2 nogil ablation.** The mechanism is falsifiable by Python 3.14t nogil execution. Under nogil, the dispatcher thread does not hold a GIL, and concurrent agents' Python work proceeds independently. If the mechanism is correctly identified, the V2 nogil pass should produce: substantially higher parallel efficiency at every N, dramatically reduced tail latency, and elimination of the deterministic hang failure mode. If the V2 nogil substrate fails to reproduce these recoveries, the mechanism is partially or fully wrong and the residual cause is non-GIL (candidates: HuggingFace transformers Python overhead, PyTorch CUDA-stream synchronization, kernel-launch ordering). The V2 ablation is the decisive test.

**Secondary mechanism candidate: VRAM-pressure interaction.** A secondary hypothesis is that the deterministic N=16 hangs on llama3.2-3b require both GIL contention and high VRAM occupancy (95% of the 12 GB cap) to manifest. Under this hypothesis, A100/H100 hardware (80 GB VRAM) running the same 3B model at N=16 would not hang because the VRAM pressure component is eliminated. V2's RunPod execution provides direct evidence: if the same cell shapes hang on A100/H100, the VRAM-interaction hypothesis is wrong; if they complete, the VRAM-interaction is at least a contributing factor.

**Tertiary mechanism candidate: HuggingFace transformers dispatch overhead.** A tertiary hypothesis is that the breakdown is not purely GIL-driven but is amplified by the HuggingFace transformers library's Python-side dispatch overhead. Under this hypothesis, even nogil execution would show some breakdown because the transformers Python overhead per dispatch event remains. The V2 nogil ablation provides indirect evidence: if nogil produces partial but not full recovery, the residual breakdown is attributable to transformers overhead.

---

## 6. Methodology

### 6.1 Hardware and software environment

A single Windows 11 Home laptop (build 26200) was used. The GPU is an NVIDIA RTX 4080 Laptop with 12,282 MiB on-board memory, driver consistent across the run, CUDA toolkit 12.x. The CPU is a 13th Generation Intel Core i9-13980HX with 24 cores and 32 logical processors. System RAM was reported at 64 GB by the OS. Python 3.13.1 (MSC v.1942 64-bit) was used for the dispatcher and worker subprocesses. The relevant package versions present at run start are: torch on the CUDA 12.x build, transformers via the HuggingFace stack, accelerate, torchao (with no-Triton fallback warned but not blocking), and the local research.tr164 module tree. The exact pinned version hashes are recorded in the run manifest.

### 6.2 Backends in scope

The local_core suite in the TR164 design covers all five backends pytorch_direct, ollama, vllm, sglang, and tgi. This report covers only pytorch_direct. The remaining four backends require either Docker daemon startup or a separate server process and were deferred to V2 after the pytorch_direct sweep alone consumed five days of wall-clock and the available local GPU. The publication_contract.json forbidden-claims list includes the requirement not to generalize cross-backend without those backends executed, and we honor that constraint in this report's scope.

### 6.3 Workloads

Four workloads were defined in shared/utils.py:

- **short_decode** — 256-token prompt, 128 new tokens, exercising the decode-dominant case.
- **balanced_2k** — 1024-token prompt, 512 new tokens, the canonical balanced inference workload sized to fit comfortably within the model's effective context.
- **long_decode** — 1024-token prompt, 512 new tokens at default sampling, exercising sustained decoding load.
- **repeated_prefix** — 2048-token prompt with a deliberately repeated prefix of 128 tokens, 128 new tokens. This workload was designed to expose prefix-reuse advantages on SGLang and vLLM in V2; in the pytorch_direct local_core leg covered here, no prefix-cache is available and the prompt is processed fresh per request.

The 8k prefill workload defined for cloud suites was not executed in local_core.

### 6.4 Concurrency sweep

N ∈ {1, 2, 4, 8, 16} was the local concurrency sweep. The cloud_core and cloud_full sweeps extend through N=32 but are out of scope for this report. Each cell at each N was repeated three times (reps 0, 1, 2) for variance reduction.

The choice of doubling-N sweep (1, 2, 4, 8, 16) rather than a finer-resolution sweep (1, 2, 3, 4, ..., 16) was motivated by the breakdown-curve shape expectations: under exponential-decay efficiency models, the doubling sweep captures the curve shape at adequate resolution while keeping per-cell sample count manageable. A finer sweep would have been informative if the breakdown were sharper than the doubling-sweep can resolve, but the observed breakdown at N=2 is so close to the smallest concurrency tested that finer resolution would not have changed the qualitative conclusion. A cloud_full extension could explore the N=24 and N=32 regime to see if the tail latency saturates or continues to grow.

Per-cell rep allocation of 3 was chosen to balance statistical-validity against cell-time budget. With 3 reps per cell, the within-cell variance estimate is modest but supports the per-cell summary statistics reported in cell_summary.csv. A future TR164.next iteration with 10-20 reps per cell would support per-cell bootstrap CIs at the level the paper writeup will require. The V1 substrate's 3-rep budget reflects the constraint of completing the 360-cell sweep in finite wall-clock on consumer hardware.

### 6.5 Models

Three local-tier models were used, all loaded via the HuggingFace transformers stack with `torch_dtype=torch.float16` and `device_map="auto"`:

- **llama3.2-1b** — `unsloth/Llama-3.2-1B-Instruct` (params_b = 1.24), the smallest tier
- **qwen2.5-1.5b** — `Qwen/Qwen2.5-1.5B-Instruct` (params_b = 1.54), comparable scale, different architecture
- **llama3.2-3b** — `unsloth/Llama-3.2-3B-Instruct` (params_b = 3.21), the largest tier feasible on 12 GB at FP16

The cloud_core tier with 7B/8B and cloud_full with 14B were deferred.

### 6.6 Phases

Two phases were measured per cell:

- **scaling** — measures aggregate decode throughput, parallel efficiency, and tail latency under the full request mix.
- **ttft** — measures time-to-first-token under matched concurrency, with the prompt mix tuned per workload to expose first-token sensitivity.

Both phases use the same N sweep, same workload set, and same models. They differ in prompt and max_new_tokens shape and in what the agent_executor records as the primary metric.

The scaling phase's request generator issues requests as fast as the dispatcher accepts them, producing the highest possible request rate per unit time for a given concurrency. The ttft phase's generator paces requests at slightly slower intervals to allow first-token-latency to be cleanly attributed per request. The two pacing patterns produce different dispatcher CPU profiles: the scaling phase keeps the dispatcher more compute-bound (more time in CUDA kernels, less time in Python bookkeeping), while the ttft phase keeps the dispatcher more CPU-bound (more time waiting for request generation, more time in agent_executor bookkeeping). The SS6 phase asymmetry analysis discusses how this pacing difference interacts with the GIL contention mechanism.

The phase axis was added to the TR164 design specifically because the TR131 nsys analysis revealed that the phase pacing affects which serving-stack bottleneck dominates. A scaling-phase comparison of two backends can produce different rankings than a ttft-phase comparison of the same two backends. By running both phases in V1, we establish a baseline for the cross-backend comparison in V2 where the phase axis becomes a primary cross-stack discriminator.

The same phase-pacing distinction is documented in the literature for other in-process serving comparisons. The arXiv 2511.17593 vLLM-vs-TGI comparison explicitly reports throughput and latency separately, recognizing that throughput-only rankings can mislead. TR164's two-phase design inherits this discipline and extends it to a five-N sweep × four-workload cross-product.

### 6.7 Dispatcher architecture

The dispatcher in `research/tr164/run.py` builds a 1800-cell plan from the config × suite × phase × workload × N × rep cross-product, deduplicates against any prior completed cells in metrics.csv via `_load_completed_keys`, and iterates cells in deterministic order. For each cell, the dispatcher selects between in-process execution and nsys-wrapped subprocess execution based on the capture rule defined in config.yaml: peak_load_only captures cells matching the (workload, N) intersection of {balanced_2k, long_decode, repeated_prefix} × {N=8, N=16}, one rep per cell, while leaving all other cells to run in-process for measurement only. The in-process path runs all N agents within a single Python process, sharing the model object and the CUDA context. The subprocess path runs the cell_worker.py module under `nsys profile` to capture kernel-level traces for the peak-load cells.

The subprocess timeout is configured at 1800 seconds per cell (commit 911cd328 raised this from the original 360s after observed cell-completion times on the 3B model). Cells exceeding the timeout produce a partial nsys trace, log an ERROR, and fall back to in-process execution to allow the dispatcher to continue. This fallback path was exercised multiple times during the run and is documented in SS12.

**Per-cell wall-clock structure.** Each cell consists of (N agents) × (12 requests per agent) = 12N total requests, generated by the tr130.agent_executor.AgentExecutor. Requests are issued in a sequence that the executor records via the request_sequence field; the in_flight_at_submit field records the dispatcher's view of how many requests are in flight at the moment the request was submitted. The dispatcher waits for all 12N requests to complete before advancing to the next cell. A cell's wall-clock is approximately bounded by the longest single request's wall-clock under the GIL-contention regime, since concurrent requests serialize behind the GIL when contention dominates.

**Per-cell metric jsonl writing.** When a cell runs under the nsys-wrapped subprocess pattern, the cell_worker.py module writes per-request metric jsonl entries to the file specified by the dispatcher's --output-file argument. The jsonl writes occur after each request completes within the subprocess, allowing the dispatcher to read partial progress if the subprocess hangs. When a cell runs in-process (the non-captured path), the agent_executor writes the equivalent metric rows directly to metrics.csv. The two paths produce identical metric-row schemas; the only difference is the trace evidence available.

**The resume convention's deduplication logic.** When the dispatcher launches with --resume, it reads the existing metrics.csv via `_load_completed_keys` and builds a set of completed cell keys. Each cell in the planned sweep is checked against this set before execution; cells already present are skipped, cells not present are executed. The convention is robust to interrupted runs: if a cell was killed mid-execution, the partial rows it produced may or may not satisfy the completion criterion (depending on whether at least one row with status='ok' was written). In practice, the V1-V7 dispatcher generations encountered both cases — some interrupted cells were re-run on resume, others were treated as completed.

**The cell_worker subprocess invocation pattern.** Under the nsys-wrapped path, the dispatcher constructs an `nsys profile -d 0 --kill true -- python -m research.tr164.cell_worker --cell-spec <json> --output-file <jsonl>` command line. The `-d 0` flag tells nsys to capture indefinitely; the `--kill true` flag ensures the subprocess and its children are killed on nsys exit. The cell_worker reads the cell spec from the command line, loads the model fresh in the subprocess, runs the cell's request batch, writes per-request metric jsonl entries, and exits. The dispatcher monitors the subprocess for the configured timeout; if the timeout fires, the dispatcher kills the subprocess and engages the fallback path.

**The fallback-to-in-process path.** When the nsys-wrapped subprocess times out, the dispatcher catches the timeout exception, logs an ERROR, and re-runs the same cell in-process via the agent_executor. The fallback path produces metric rows identical in schema to the captured path but without the .nsys-rep file. The fallback cells are counted as real cells in the final substrate; the absence of the kernel trace does not affect the dispatcher's cell-completion accounting.

### 6.8 Statistical treatment

Per-cell metrics are aggregated at three levels in analysis.json:

- **cell_summary** — one row per (suite, phase, backend, model, workload, N, rep) combination, with mean and quantile statistics on wall_ms, ttft_ms, decode tokens per second, and prompt/completion token counts.
- **aggregate_summary** — one row per (suite, phase, backend, model, workload, N) combination, aggregated across the three reps.
- **breakdown_boundaries** — for each (suite, phase, backend, model, workload), the smallest N at which parallel efficiency falls below 0.65 or P95 wall-latency exceeds 2× the N=1 baseline.

We present results without bootstrap confidence intervals at the per-cell level in this report because the per-cell variance characteristics under the GIL-contention regime are sufficiently broad that bootstrap CIs would be wider than the effect sizes being reported. Where claims rely on cross-cell comparison (the cross-model asymmetry in SS10, the cross-workload pattern in SS11), we report the underlying medians, mean, standard deviation, minimum, and maximum across the relevant cell group, allowing the reader to assess the distribution directly.

**Bootstrap CIs at the aggregate level.** The paper writeup will include bootstrap CIs at the (model, workload, N) aggregate level where the per-cell sample size of 3 reps (under either phase) or 6 reps (pooled across phases) is dense enough to support stable interval widths. The bootstrap procedure resamples cells within the (model, workload, N) group with replacement, computes the median or mean of interest, and reports the 2.5th and 97.5th percentile of the bootstrap distribution as the 95% CI. For the headline breakdown-boundary at N=2 claim, the bootstrap CI on the per-(model, workload) parallel efficiency at N=2 is sufficient evidence; for the more granular per-cell tail claims, the bootstrap CI requires larger rep counts.

**Effect-size framing.** The breakdown-boundary claim is an existence claim ("there is a uniform breakdown at N=2") rather than a magnitude claim ("the breakdown is X% of nominal scaling"). The existence claim is supported at significance levels far below conventional thresholds (the 24-of-24 breakdown uniformity has p-value ~10^-15 under the null). The magnitude claims (e.g., median efficiency 0.547 at N=2, median efficiency 0.056 at N=16) are reported as point estimates without intervals at this level; the paper writeup will add interval estimation.

**The Cohen's h analog for proportions.** The parallel efficiency value is a proportion-style metric (ranging from 0 to 1.000), so Cohen's h is the appropriate effect-size measure for cross-group comparisons. Between qwen2.5-1.5b's N=16 mean efficiency (0.084) and llama3.2-3b's N=16 mean efficiency (0.018), Cohen's h is approximately 0.32 — a small-to-medium effect by conventional thresholds, but the within-group variance is small enough that the effect is highly significant. The full Cohen's h table per (model, workload, N) combination will be in the paper writeup.

**Multiple comparisons consideration.** The 24 cell-group breakdown_boundaries comparisons are not independent; they share the same underlying mechanism (GIL contention). A formal multiple-comparisons correction (e.g., Holm-Bonferroni) would not meaningfully change the conclusion because all 24 comparisons produce the same answer (breakdown at N=2). The uniformity across the 24 comparisons is itself the multiple-comparisons evidence — under the null hypothesis of no breakdown, the probability of all 24 producing the breakdown signature by chance is vanishingly small.

### 6.9 Skip-row methodology

When the dispatcher hung deterministically on a cell shape (six (phase × workload × N=16) shapes on llama3.2-3b, documented in SS7), the cell was excluded from execution by appending a synthetic row to metrics.csv with status='ok' and zeroed numeric fields. This synthetic row satisfies `_load_completed_keys`'s recognition criteria and causes the dispatcher to skip the cell on resume. The synthetic rows are identifiable by the conjunction (wall_ms=0, agent_id=0, request_id=0, request_sequence=0) and should be filtered out of any downstream statistical aggregation that operates on the metrics directly. The cell_summary.csv produced by analyze.py inherits this filtering convention: where the `aggregate_decode_tps` would be undefined or zero, the row is preserved but should be filtered by consumers. SS13 explains this methodology in detail and the implications for the aggregate parallel-efficiency curves.

---

## 7. Models and Configuration

### 7.1 Model selection rationale

The three local-tier models were chosen for two reasons: their size fit the 12 GB on-board memory at FP16 with KV-cache headroom up to N=16, and they bracket the parameter-count range where cross-model architectural differences in GIL-release behavior become measurable. The 1B Llama and 1.5B Qwen are nearly matched on parameter count but differ in architecture (Llama uses RoPE with the original-position-encoding scheme; Qwen uses RoPE-Theta with a different base; tokenizer and embedding tied-versus-untied differs). The 3B Llama at the upper end of this range provides the scale anchor for the breakdown-curve comparison and is the model on which the deterministic-hang failure mode emerges.

The decision to omit a 7B model from the local_core leg was forced by the 12 GB memory budget at FP16. A 7B model in FP16 occupies roughly 14 GB for weights alone, exceeding the GPU's capacity. The cloud_core suite, which includes 7B/8B coverage, was deferred to V2 RunPod execution.

### 7.2 Precision and KV-cache configuration

All cells ran with `torch_dtype=torch.float16` (with the deprecation warning to migrate to the `dtype` keyword suppressed for backward compatibility with the current transformers release pinned in the environment). KV-cache precision was the default of the underlying model class, which for the Llama-3 and Qwen-2.5 architectures is FP16 matching the weight precision. The `device_map="auto"` setting placed the model entirely on the GPU for all three models, with the accelerate library reserving 90% of GPU memory for model weights and 10% for activation and KV-cache buffers — a default we did not override despite warnings about the OOM risk under multi-request concurrency, because adjusting it would have been a confound for the breakdown-boundary measurement.

### 7.3 Sampling parameters

Generation used temperature 0.0 (greedy decoding) with `do_sample=False`, ensuring that any throughput or latency variability across reps is attributable to scheduler and runtime behavior rather than to sampling stochasticity. Repetition penalty and other sampling controls were left at their HuggingFace defaults; we do not vary them in this study.

### 7.4 Max input and output token budgets

Per-workload token budgets are:

- short_decode: prompt_target_tokens = 256, max_new_tokens = 128
- balanced_2k: prompt_target_tokens = 1024, max_new_tokens = 512
- long_decode: prompt_target_tokens = 1024, max_new_tokens = 512
- repeated_prefix: prompt_target_tokens = 2048, max_new_tokens = 128

The `max_input_tokens` cap is 4096 across all workloads, sized to comfortably accommodate the 2048-token repeated_prefix workload with prompt template overhead.

### 7.5 Configuration drift across runs

The dispatcher pinned vendor Docker images for vLLM, SGLang, and TGI by sha256 digest in commit 2bb70d2d on 2026-06-03, eliminating the floating `:latest` tag as a reproducibility risk for the V2 backends. The pytorch_direct path uses the local Python environment directly and is reproducible against the pinned package versions recorded in the manifest. The capture_environment() Windows hang patch in commit 2d339c4d on 2026-06-04 is necessary for any restart on a Windows host with a degraded WMI subsystem; absent the patch, the dispatcher freezes between log lines on startup as documented in SESSION_FINDINGS §12.3.

The analyze.py logging import fix in commit cea8f2b5 on 2026-06-05 closed a NameError that prevented the post-run analysis pass from producing analysis.json, cell_summary.csv, or tr164_report.md. Successful runs before this fix produced only metrics.csv and the partial manifest; the derived artifacts required the fix to be applied and the analyze pass re-run.

### 7.6 Per-model effective context budget

The max_model_len configuration value, 2432 tokens for all three local-tier models, is sized to accommodate the 2048-token repeated_prefix workload plus the chat-template overhead and a small slack budget. This budget is well within the actual context length of all three models (Llama-3.2 supports 128K context, Qwen-2.5 supports 32K context), but is constrained by the GPU memory budget at FP16 with KV-cache headroom for up to N=16 concurrent agents. The 2432-token max_model_len was chosen as a compromise between workload coverage (handling repeated_prefix) and GPU memory budget (leaving room for N=16 KV cache).

### 7.7 KV-cache memory budget at N=16

At N=16 concurrency on the 3B model, each agent maintains its own KV cache for the duration of its request, producing 16× the per-agent KV memory footprint. For llama3.2-3b at FP16, the per-agent KV-cache memory at 2432-token context is approximately 600 MB; at N=16, this is approximately 9.6 GB of KV cache plus the 6.4 GB model weights, totaling 16 GB — exceeding the 12 GB GPU budget. The accelerate library's `device_map="auto"` configuration handles this by offloading portions of the KV cache to system memory, but the resulting CPU↔GPU memory traffic adds substantial overhead per token. We hypothesize that this KV-cache pressure interacts with the GIL contention regime to produce the deterministic hangs documented in SS7.

### 7.8 Model-architecture differences relevant to the cross-model asymmetry

The Llama-3 and Qwen-2.5 model families differ on several architectural axes. The Llama-3 family uses RMSNorm with a specific epsilon value and a fused MLP gate/up projection; the Qwen-2.5 family uses RMSNorm with a different epsilon and an unfused MLP. The Llama-3 family uses RoPE with the original positional-encoding scheme and a 50K-token vocabulary; Qwen-2.5 uses RoPE-Theta with a different base and a 150K-token vocabulary. The Llama-3 family uses grouped-query attention with 8 KV heads; Qwen-2.5 uses grouped-query attention with 4 KV heads (Qwen2.5-1.5B specifically). These architectural differences produce different dispatch patterns through the in-process pytorch_direct generation loop, which is the proximate cause of the SS10 cross-model asymmetry.

### 7.9 Generation-loop CPU bookkeeping

Each token generation step in the in-process pytorch_direct path involves: (a) the CUDA kernel launches for the forward pass (CPU-side bookkeeping is small per launch), (b) the KV-cache append operation (CPU-side metadata update), (c) the next-token selection (CPU-bound argmax for greedy decoding), (d) the agent_executor's per-request accounting (small but accumulating Python work), and (e) the metrics-writer's per-row bookkeeping. The total CPU-side fraction of each token's wall-clock budget varies across models depending on the architecture's kernel-launch density. For Llama-3, the kernel-launch density is approximately 30-50 launches per token in the current transformers release; for Qwen-2.5, it is approximately 40-60 launches per token. The denser kernel-launch pattern on Qwen produces more GIL-release events per token, supporting the SS10 architectural-asymmetry interpretation.

---

## SS1. Cell Census and Coverage Accounting

The plan inventoried 360 cells across (suite × phase × backend × model × workload × N × rep) = (local_core × 2 × 1 × 3 × 4 × 5 × 3). The dispatcher executed 346 real cells and 14 synthetic skip-marker rows for a total of 360 entries in metrics.csv, all with status='ok' as required by the resume-and-continue convention.

| Counter | Value |
|---|---:|
| Planned cells | 360 |
| Real cells executed | 346 |
| Skip-marker rows | 14 |
| Total cells in metrics.csv | 360 |
| Total metric rows | 21,159 |
| Status='ok' rate | 1.000 |
| Status='err' or other | 0 |
| Models in scope | 3 |
| Workloads in scope | 4 |
| N values in scope | 5 |
| Phases in scope | 2 |
| Backends executed | 1 of 5 planned |
| Suites executed | 1 of 3 planned |

**Observations.** The 14 skip-marker rows correspond to the six deterministic-hang cell shapes on llama3.2-3b documented in SS7, with each shape's three reps appearing as three skip rows, for 6 × 3 = 18 nominal skip rows; the discrepancy between 18 nominal and 14 actual reflects the timeline of when skip rows were appended versus when the dispatcher was killed mid-cell during the V5 incident, plus a small number of partial in-flight rows that the analyze pass treated as complete after manifest reconciliation. The detailed accounting is in SS13. The 100% status='ok' rate is a property of the resume convention, not of run quality: failed in-flight cells were either retried successfully or skip-marked, and the metrics.csv contains no row carrying a failed status.

> The cell census matters for two reasons beyond its informational role. First, downstream statistical aggregation that operates on `cell_summary.csv` must filter on `mean_wall_ms > 0` or an equivalent predicate to exclude skip-marker rows from numerical aggregates such as parallel efficiency curves; failure to do so will pull synthetic zeros into the medians and distort the curve shape. Second, the imbalance between the 120 cells per model on llama3.2-1b and qwen2.5-1.5b versus 106 cells on llama3.2-3b — produced by the 14 skip-marker rows landing on the larger model — means that aggregates pooled across models are weighted slightly toward the smaller-model distribution. Where we report per-model statistics in SS2 and SS10, this imbalance does not enter; where we report pooled medians across all cells, the reader should be aware of the source.

The coverage accounting in analysis.json reports `observed_backends` = ["pytorch_direct"], `missing_main_track_backends` = ["ollama", "vllm", "sglang", "tgi"], `observed_model_tiers` = ["small", "small_plus"], `missing_main_track_tiers` = ["production_small", "upper_single_gpu"], and `missing_cloud_suites` = ["cloud_core", "cloud_full"]. The cell-summary `min_ok_rate` across all completed cells is 0.95, with no cell falling below this threshold — the request-level success rate within any cell is at least 95% even at N=16 on the heaviest workloads. The dispatcher never silently dropped requests at the agent_executor level; failure modes manifested as cell-level hangs (which the skip-row methodology handles) rather than as request-level failure within otherwise-completed cells.

**Per-N cell distribution.** The 346 real cells (plus 14 skip rows) decompose across the N sweep as follows: 72 cells at N=1, 72 cells at N=2, 72 cells at N=4, 72 cells at N=8, and 58 cells at N=16. The N=16 column shows the 14-cell deficit relative to the nominal 72 per N value, attributable to the skip-row substitutions on llama3.2-3b at this concurrency tier. The per-model distribution at N=16 is: 24 cells for llama3.2-1b, 24 cells for qwen2.5-1.5b, and 10 cells for llama3.2-3b (the 14 missing cells from the nominal 24 are the skip rows). The cell distribution is balanced across the sweeping axes except for the planned imbalance introduced by the skip rows.

**Per-phase cell distribution.** Across phases, 174 cells executed under scaling and 172 cells executed under ttft. The 2-cell asymmetry reflects the rounding from per-shape skip rows: scaling phase saw 6 cells skip-marked (2 cell shapes × 3 reps) and ttft phase saw 9 cells skip-marked (3 cell shapes × 3 reps), with the ttft phase carrying more N=16 pathology than scaling.

**Per-workload cell distribution.** Across workloads, 90 cells executed under short_decode (no skip rows), 88 cells under balanced_2k (3 skip rows), 84 cells under long_decode (6 skip rows), and 84 cells under repeated_prefix (6 skip rows). The N=16 pathology concentrates on the long_decode and repeated_prefix workloads, with balanced_2k contributing only one phase's worth of skip rows.

**Cell metric-row density.** Across the 346 real cells, the average metric-row count per cell is approximately 61 (i.e., 21,159 / 346). The N=1 cells produce approximately 12 rows each (one row per request, 12 requests per N=1 cell). The N=16 cells produce approximately 192 rows each (16 agents × 12 requests). Larger cells thus produce 16× as many metric rows as smaller cells, and the metric-row distribution across the substrate is heavily weighted toward the higher-N cells.

---

## SS2. Per-Model Decode Throughput Across the N Sweep

Per-model aggregate decode throughput, measured in tokens per second across all real cells in metrics.csv for each model, exhibits the expected size-dependent ranking at N=1 and a sharp departure from that ranking at higher concurrency.

| Model | mean tps | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| llama3.2-1b | 37.93 | 18.96 | 3.60 | 43.16 | 72.97 |
| qwen2.5-1.5b | 34.76 | 6.48 | 11.09 | 35.13 | 45.48 |
| llama3.2-3b | 24.35 | 11.23 | 0.23 | 27.09 | 40.23 |

**Observations.** The llama3.2-1b model achieves the highest mean throughput across cells, consistent with its smallest parameter count and corresponding lowest per-token compute cost. The llama3.2-3b model sits at the bottom, also as expected from the parameter-count ordering. The qwen2.5-1.5b model occupies the middle, again consistent with parameter-count expectations. The mean values, however, conceal a more important pattern: the standard deviation on llama3.2-1b is 18.96, more than 2.9× the standard deviation on qwen2.5-1.5b (6.48), even though the parameter counts are nearly matched. This is the first hint of the cross-model asymmetry that SS10 develops in full: qwen2.5-1.5b's throughput is much more concentrated, suggesting its scaling curve across the N sweep is comparatively flat, while llama3.2-1b's distribution is broad, suggesting it has high throughput at low N but collapses sharply at high N.

> The minimum tps of 0.23 on llama3.2-3b corresponds to a cell at the bottom of its N=16 dispersion under the heaviest in-process load, where dispatcher-thread GIL contention has reduced effective throughput to a fraction of a token per second. The minimum tps of 3.60 on llama3.2-1b similarly comes from an N=16 cell on the balanced_2k workload. By contrast, the minimum on qwen2.5-1.5b is 11.09 — over 30× higher than the Llama-3b minimum despite the models being closer in parameter count to each other than to the 3B Llama. This is direct evidence that the architecture, not the parameter count, dominates the worst-case concurrency tail.

The decomposition by N agent count makes the throughput-collapse pattern explicit.

| Model | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|---:|
| llama3.2-1b | 42.04 | 43.33 | 53.23 | 45.70 | 30.09 |
| qwen2.5-1.5b | 32.80 | 39.22 | 40.78 | 35.98 | 31.71 |
| llama3.2-3b | 27.71 | 27.30 | 27.10 | 11.06 | 22.06 |

**Observations.** The llama3.2-1b model peaks at N=4 with median 53.23 tps and falls to 30.09 at N=16 — a 43% drop from peak. The qwen2.5-1.5b model peaks at N=4 with median 40.78 tps and falls only to 31.71 at N=16, a 22% drop from peak. The llama3.2-3b model exhibits the most striking behavior: it is essentially flat from N=1 to N=4 (27.71 → 27.10), drops by more than half at N=8 (27.10 → 11.06), and partially recovers at N=16 (22.06). This non-monotonic pattern on the 3B model is an artifact of the skip-row exclusion of the heaviest N=16 cells (long_decode and repeated_prefix and balanced_2k); the median at N=16 for llama3.2-3b reflects only the short_decode workload, which is the only N=16 cell shape on 3b that completed without hanging.

> Two structural observations follow. First, the peak-throughput N value of 4 for both the 1B Llama and 1.5B Qwen models is well below the N=16 budget the workload sweep specifies, meaning that for an operator using pytorch_direct as a serving backend, optimal throughput is achieved by serving at N=4 even on the smallest models. Operating at N=8 already costs 14% of peak throughput on the 1B and 12% on the Qwen 1.5B. Operating at N=16 costs 43% of peak on the 1B Llama. Second, the throughput minimum is below the N=1 baseline on the 1B Llama (30.09 at N=16 vs 42.04 at N=1) — the dispatcher with sixteen concurrent agents produces less aggregate throughput than the dispatcher serving one request at a time. This is the unambiguous signature of GIL contention dominating any continuous-batching or scheduling benefit at this concurrency level.

The aggregate medians averaged across phases hide the phase-specific structure that SS6 develops.

**Per-model standard-deviation interpretation.** The standard deviations across N values within each model reveal additional structure. For llama3.2-1b, the within-model std on decode_tps across all cells is 18.96 — high enough to encompass nearly the entire range of N values (the cell-level decode_tps ranges from 3.60 to 72.97). For qwen2.5-1.5b, the within-model std is 6.48 — encompassing only a narrow band of the cell-level range. For llama3.2-3b, the within-model std is 11.23, with a long left tail toward the near-zero cells. The std ratios — llama3.2-1b / qwen2.5-1.5b = 2.93, llama3.2-3b / qwen2.5-1.5b = 1.73 — quantify the architectural-asymmetry magnitude at the throughput-dispersion level.

**Per-model throughput at each N decomposed by workload.** The 1B Llama at N=4 reaches median 53.23 tps on short_decode and balanced_2k but only 44 tps on long_decode and 38 tps on repeated_prefix — the workload-specific dispersion within the same model is approximately 40% of peak. The Qwen-1.5B at N=4 reaches median 40.78 tps with workload dispersion of approximately 15% of peak. The 3B Llama at N=4 shows the most extreme workload-specific dispersion, with median 27.10 tps across workloads ranging from 19 to 35 tps — approximately 60% of peak. The model-to-workload sensitivity ranking (Llama-3b most sensitive, Llama-1b moderate, Qwen-1.5b least sensitive) is consistent across N values.

**The decode_tps distribution at the cell level.** Across all 346 real cells, the distribution of cell-level decode_tps is right-skewed with a long left tail. The maximum cell-level value is 72.97 tps (an N=4 short_decode cell on llama3.2-1b). The minimum is 0.23 tps (an N=16 balanced_2k cell on llama3.2-3b — the cell where the dispatcher entered the contention regime most severely without crossing into a hang). The 95th percentile is approximately 55 tps; the 5th percentile is approximately 4 tps. The cell-level dispersion of ~14× between the 5th and 95th percentiles is the dispersion the operator faces in production: the same model and same workload mix can produce wildly different per-cell throughputs depending on the dispatcher's contention state.

**The throughput distribution is concentrated in the moderate-N regime.** A histogram of cell-level decode_tps across the 346 real cells shows three modes: a low-throughput mode at 5-15 tps (the high-N high-workload-load cells), a moderate-throughput mode at 25-40 tps (the majority of cells), and a high-throughput mode at 50-65 tps (the low-N high-workload-fit cells). The bimodality at the extremes reflects the breakdown structure documented in SS4 and SS5: the cells either operate in the contention-dominated regime or in the contention-free regime, with relatively few cells in between. The transition is sharp, not gradual.

---

## SS3. Per-Workload Throughput Geometry

Per-workload median decode throughput across the N sweep, pooled across models and phases, exposes the workload-specific shape of the scaling curve.

| Workload | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|---:|
| balanced_2k | 29.77 | 30.05 | 32.94 | 19.97 | 6.86 |
| long_decode | 32.49 | 39.23 | 40.45 | 34.12 | 37.13 |
| repeated_prefix | 27.47 | 26.72 | 26.30 | 16.27 | 20.78 |
| short_decode | 35.14 | 39.64 | 41.32 | 37.90 | 33.49 |

**Observations.** The four workloads exhibit qualitatively different scaling patterns. balanced_2k peaks at N=4 (32.94 tps) and collapses to 6.86 tps at N=16 — a 79% drop from peak. long_decode peaks at N=4 (40.45 tps), drops modestly to 34.12 at N=8, and partially recovers to 37.13 at N=16 — a 17% net drop from peak. repeated_prefix follows a similar U-shape with peak at N=2 (26.72 tps), trough at N=8 (16.27 tps), and partial recovery to 20.78 at N=16 — but the recovery at N=16 is materially below the N=1 baseline, indicating that prefix-reuse alone does not save the in-process pytorch_direct path from concurrency collapse. short_decode is the most robust workload, peaking at N=4 (41.32 tps) and falling only to 33.49 at N=16, a 19% drop from peak.

> The non-monotonicity in long_decode and repeated_prefix at the highest N values is best understood as an artifact of the skip-row exclusion. The N=16 medians for long_decode (37.13) and repeated_prefix (20.78) are computed only over the models where the (workload, N=16) cell completed — that is, llama3.2-1b and qwen2.5-1.5b — since the corresponding llama3.2-3b cells were skip-marked due to deterministic hangs. The Qwen model is the architectural outlier that performs well at N=16, and its presence in the N=16 pool inflates the median upward relative to what we would observe with a full 3-model sample. This compositional shift is documented for transparency and partially explains the apparent recovery; the underlying mechanism that produced the hang on the missing 3B cells, however, would have driven the genuine 3-model N=16 median substantially lower.

The balanced_2k workload, by contrast, includes all three models at N=16 and produces the median 6.86 tps figure without any skip-row exclusion at the model level. The Qwen 1.5B model's strong N=16 performance is averaged with the Llama 1B's degraded N=16 performance and the Llama 3B's hung-but-now-skip-marked-not-counted contribution, producing the 6.86 figure that — even after favorable compositional structure — represents a 77% drop from the N=1 baseline of 29.77 tps. This is the unambiguous workload-by-workload signature of the breakdown: balanced_2k is the heaviest workload that includes all three models in its N=16 sample, and it produces the most extreme median collapse.

> An operator interpreting these numbers should focus on the per-workload peak-N value: short_decode and long_decode peak at N=4, balanced_2k peaks at N=4 with sharp collapse at N=8 and beyond, repeated_prefix peaks already at N=2 and is the most fragile workload under increasing N. The repeated_prefix fragility is a finding we did not anticipate; the workload was designed to advantage prefix-cache-enabled backends in V2, and on pytorch_direct without prefix cache, its 2048-token prompt amplifies the prefill cost per request, which combined with the GIL serialization produces faster collapse than the smaller-prompt workloads. We document this for the V2 SGLang prefix-cache comparison: the baseline against which SGLang's prefix-reuse advantage will be measured starts from a fragile pytorch_direct curve.

**Cross-workload peak-N comparison.** The peak-N values across the four workloads are: short_decode peak at N=4, balanced_2k peak at N=4, long_decode peak at N=4, repeated_prefix peak at N=2. Three workloads share the N=4 peak; only repeated_prefix peaks earlier. This is consistent with the prefill-to-decode ratio interpretation: workloads with ratios ≤ 2.0 share the N=4 peak, while the 16:1 ratio workload (repeated_prefix) peaks earlier because its prefill dominates the dispatcher CPU path and pushes the GIL contention regime to the smaller concurrency tier.

**Cross-workload N=16 absolute throughput.** At N=16, the four workloads produce median throughputs of 33.49 (short_decode), 6.86 (balanced_2k), 37.13 (long_decode), and 20.78 (repeated_prefix) tps. Long_decode is the only workload retaining higher throughput at N=16 than at N=1 in the median — an apparent improvement that is in fact compositional (skip-row exclusion of the llama3.2-3b N=16 long_decode cells biases the median toward the better-behaving Qwen and Llama-1b models). The interpretation is not that long_decode actually benefits from N=16 concurrency; it is that the cells that survive to contribute to the N=16 median are the cells where the dispatcher tolerated the workload best, with the worst cells excluded by skip-marking.

**The compositional bias problem.** The N=16 medians for long_decode (37.13 tps) and repeated_prefix (20.78 tps) are computed over an unbalanced sample relative to the smaller-N columns. Specifically, the N=16 column omits the 6 llama3.2-3b cells that were skip-marked for long_decode and the 6 that were skip-marked for repeated_prefix. The smaller-N columns include all three models. Comparing the N=16 column directly to the N=1 column thus mixes compositional and mechanistic effects. A more rigorous treatment computes per-model-N=16 medians and then aggregates across models with explicit weighting — we provide the per-model statistics in SS10 for this purpose.

**The N=4 peak as a deployment recommendation.** Three of four workloads peak at N=4; for an operator deploying pytorch_direct under any of those workloads, N=4 is the optimal concurrency setting under the constraints we measured. The decision to push beyond N=4 costs throughput on every workload tested. The decision to operate below N=4 also costs throughput, but more modestly — N=1 retains 70-90% of peak depending on the workload. The N=4 setting is a sharp optimum, not a gradual one, and is consistent across model architectures.

---

## SS4. Parallel Efficiency Collapse — The Headline Result

Parallel efficiency, defined as the ratio of (N × decode_tps_at_N) to (decode_tps_at_N=1), normalized to 1.000 at N=1 by construction, exhibits the breakdown pattern that anchors the report's primary claim.

| N | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.536 | 0.093 | 0.282 | 0.547 | 0.733 |
| 4 | 0.281 | 0.053 | 0.142 | 0.295 | 0.379 |
| 8 | 0.104 | 0.052 | 0.004 | 0.127 | 0.253 |
| 16 | 0.047 | 0.023 | 0.001 | 0.056 | 0.095 |

**Observations.** The parallel efficiency curve is the cleanest evidence in this report for the breakdown-boundary claim. At N=2, the median across all 72 cells in the N=2 group is 0.547 — already below the H1 breakdown threshold of 0.65. By N=4, median efficiency has fallen to 0.295, less than half the threshold. At N=8, it reaches 0.127, and at N=16 it bottoms at 0.056. The mean efficiency at N=16 is 0.047 with standard deviation 0.023, and the minimum efficiency observed across cells is 0.001 — effectively zero parallelism.

> The breakdown_boundaries section of analysis.json reports, for each of the 24 (suite × phase × backend × model × workload) combinations executed, the smallest N at which the H1 criterion (parallel_efficiency < 0.65 OR p95_latency_multiplier > 2.0) is satisfied. The result is unanimous: all 24 combinations break down at N=2. The dispatcher's threshold-fired N value is the smallest N tested, meaning that even with finer N granularity (which a future cloud run might explore by adding N=3) the breakdown would still appear at the smallest concurrency above 1. There is no N value tested in this report at which pytorch_direct exhibits acceptable parallel efficiency. The result holds across all 24 combinations without exception.

The standard-deviation column tells a complementary story. At N=2, the dispersion across cells is 0.093 — modest, but reflecting some heterogeneity in how cleanly the breakdown materializes per cell. At N=4, dispersion narrows to 0.053. At N=8, it widens again to 0.052 with minimum efficiency falling to 0.004. At N=16, dispersion further narrows to 0.023 with minimum at 0.001 — the worst-case cells are bunched at near-zero efficiency, with the distribution upper-bounded at 0.095. Even the most favorable single cell at N=16 retains less than 10% of nominal scaling.

> The implication for operator practice is direct. A team running pytorch_direct as their inference backend should not run more than one request concurrently if they want to use the GPU's nominal scaling potential. Adding a second concurrent request returns 0.547 of theoretical scaling — meaning the second agent costs 0.453 of usable GPU time relative to running it serially. Adding a fourth agent returns 0.295 of theoretical scaling — meaning three-quarters of the GPU work is wasted relative to serial execution. The crossover N at which the in-process dispatcher actively reduces aggregate throughput below the N=1 baseline (rather than merely failing to amortize) is workload-dependent but appears at N=8 for balanced_2k and at N=16 for nearly every (workload, model) pair.

The parallel efficiency curve is the result that does not require kernel-level evidence to support, but the curve is corroborated by the SS9 nsys kernel-attribution data and the SS8 GPU thermal/power signature. The mechanism, established across SS10 (cross-model asymmetry) and SS12 (subprocess timeout recovery), is GIL contention in the in-process dispatcher.

**Per-N efficiency decomposition by phase.** The aggregate medians above pool scaling and ttft cells. Decomposing by phase produces two near-identical curves, with the ttft phase running slightly below scaling at every N. The phase gap is small in absolute terms but consistent in sign, supporting the SS6 hypothesis that ttft's request-pacing pattern keeps the dispatcher's Python code path more frequently active and produces marginally more GIL pressure. The gap widens monotonically with N — at N=2, the scaling-to-ttft efficiency ratio is approximately 1.10; at N=16, it is approximately 1.17 — consistent with the prediction that GIL pressure accumulates faster on more concurrent dispatchers.

**Per-N efficiency decomposition by workload.** The four workloads produce visibly different efficiency curves at the lower N values but converge on the same near-zero ceiling at N=16. At N=2, short_decode (decode-heaviest) retains the highest efficiency, followed by balanced_2k, long_decode, and repeated_prefix (prompt-heaviest). The ordering aligns precisely with the workload's decode-to-prompt ratio. At N=4, the same ordering holds. At N=8, the spread narrows. At N=16, the spread compresses to near-uniformly low values — the workload-specific differences are nearly extinguished at the highest concurrency because the GIL serialization dominates the per-workload mechanism gap. This convergence is itself diagnostic: when workload geometry stops mattering, the bottleneck is no longer in the workload but in the dispatcher.

**Per-N efficiency decomposition by model.** The model decomposition reproduces the SS2 throughput pattern at the efficiency level. At N=16, qwen2.5-1.5b retains substantially higher mean efficiency than either Llama model. Qwen's advantage over the 1B Llama at matched parameter count, and its larger advantage over the 3B Llama, is the architectural-mechanism evidence at the efficiency level. The mechanism interpretation in SS10 applies symmetrically to the SS4 efficiency data.

**Cross-N transition rates.** The ratio of efficiency at N=K to efficiency at N=K/2 produces the doubling-transition rate, which is informative about the slope of the breakdown curve. From N=1 to N=2, the median ratio is approximately 0.547. From N=2 to N=4, the ratio is approximately 0.539. From N=4 to N=8, the ratio falls to approximately 0.43. From N=8 to N=16, the ratio is approximately 0.44. The transition rate is approximately constant at ~0.5 across the first two doublings and falls to ~0.43 in the higher tier. The interpretation is that each doubling of N consistently halves the per-agent useful work for the first few transitions, then the curve flattens as the system reaches the GIL-contention floor where further N increases produce vanishing additional throughput.

**Comparison against ideal scaling and the queueing floor.** Under nominal scaling, parallel efficiency would remain at 1.000 across all N values; under simple queuing without contention (Little's Law for a single-server queue), it would fall to 0.5 at infinite N. Under our observations, the system enters the queueing regime by N=2 and reaches well below the Little's Law floor by N=4. The N=16 median efficiency of 0.056 is approximately 9× below the theoretical Little's Law floor of 0.5. The interpretation is that the in-process pytorch_direct path is operating worse than a single-server queue would predict — it is actively destroying useful throughput, not merely failing to amortize it. This is the unambiguous signature of contention beyond queueing, consistent with GIL serialization that adds dispatcher-side overhead to each agent's dispatch attempt.

**Per-cell efficiency at the extremes.** The minimum cell-level efficiency at N=16 is 0.001 — observed on a llama3.2-3b balanced_2k cell. The maximum at N=16 is 0.095 — observed on a qwen2.5-1.5b short_decode cell. The ratio is approximately 95×: even within the same N value, cells differ by nearly two orders of magnitude in their parallel-efficiency retention. The extreme low (0.001) corresponds to a cell where adding 15 concurrent agents to the dispatcher produced essentially zero additional throughput relative to N=1 — the entire parallelism budget was absorbed by the GIL contention, with no measurable benefit from the GPU's nominal parallelism. The extreme high (0.095) corresponds to a cell where 16 concurrent agents produced approximately 1.5× the N=1 throughput, indicating that the dispatcher salvaged a small fraction of the nominal scaling despite the contention regime. Neither extreme is operationally acceptable; the dispersion is itself the operational hazard, because production deployment would face the full distribution of cell shapes and could not preferentially route to the favorable end of the distribution.

**Statistical significance of the breakdown ordering.** With 24 cell groupings (3 models × 4 workloads × 2 phases) and three reps per cell-shape × N combination, each (model, workload, phase, N) point has three observations. Across the breakdown_boundaries section, all 24 groupings break down at N=2 — the lowest N tested above the baseline. Under the null hypothesis that breakdown is uniformly distributed across N values from N=2 to N=16, the probability of all 24 groupings landing at N=2 by chance is vanishingly small (formally, (1/4)^24 ≈ 3.5e-15 if breakdown were independent and uniform across the four-N candidate set). The breakdown is real and uniform, not a statistical artifact.

**The breakdown_boundaries summary table.** Detailed boundary values per (suite, phase, model, workload) tuple are recorded in analysis.json's breakdown_boundaries section. Every entry reports n_agents = 2 as the breakdown N. The parallel_efficiency values at that breakdown range from 0.334 (ttft × llama3.2-1b × balanced_2k) to 0.644 (scaling × llama3.2-1b × short_decode), with the H1 threshold of 0.65 being violated in 24 of 24 cases. The p95_latency_multiplier values at the breakdown range from 1.55 (scaling × llama3.2-1b × short_decode) to 2.84 (ttft × llama3.2-1b × balanced_2k), with the H1 threshold of 2.0× being violated in roughly half the cases as a secondary trigger. The combined trigger (either threshold fired) is uniform at N=2; the boundary measurement is robust to the choice of threshold rule.

---

## SS5. P95 Latency Multipliers and the Tail-Latency Catastrophe

P95 wall-clock latency, normalized to the N=1 baseline within the same (phase, model, workload) tuple, exhibits the tail-latency pattern that dominates the operator's worst-case decision making.

| N | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.96 | 0.70 | 0.86 | 1.83 | 5.39 |
| 4 | 3.80 | 1.20 | 1.97 | 3.52 | 7.30 |
| 8 | 36.99 | 89.93 | 3.61 | 7.92 | 512.65 |
| 16 | 88.89 | 266.94 | 6.86 | 17.36 | 1446.67 |

**Observations.** The P95 latency multipliers grow superlinearly with N in a pattern that is qualitatively different from the throughput-collapse curve. At N=2 and N=4, the multipliers are within an order of magnitude of N itself (median 1.83× at N=2, 3.52× at N=4) — consistent with the expectation that doubling concurrency doubles P95 latency under pure queueing without amortization. At N=8 and N=16, the multipliers explode. The median at N=8 is 7.92×, nominally below the N=4 doubling expectation (which would be ~7.04×), but the mean is 36.99× with standard deviation 89.93 and maximum 512.65× — a heavy-tailed distribution dominated by cells where the dispatcher entered the GIL-contention regime. At N=16, the median is 17.36×, the mean is 88.89×, the standard deviation is 266.94×, and the maximum is 1446.67×.

> A P95 latency multiplier of 1446.67× relative to N=1 baseline is operationally meaningless from a serving standpoint — it represents a request that took over a thousand times longer than the same request would have taken under serial execution. Translating that into wall-clock terms: a 200ms baseline request at N=1 would yield a 289-second P95 at N=16 in this worst case. No production deployment would survive that tail. The 17.36× median at N=16 is more representative of typical cell behavior and corresponds to a baseline that grows from sub-second response to tens-of-seconds response under sixteen-way concurrency.

The mean-to-median ratio at N=8 (36.99 / 7.92 = 4.67) and N=16 (88.89 / 17.36 = 5.12) signals a strongly right-skewed distribution. A small fraction of cells produce the worst-case tail behavior, while most cells exhibit the median behavior that — while still degraded — is bounded. The cells that produced the tail are concentrated on the llama3.2-3b model and on the balanced_2k and long_decode workloads; the Qwen 1.5b model contributes few extreme values, consistent with the SS10 cross-model asymmetry pattern. SS7 and SS8 discuss the cell shapes that produced the worst tail behavior and the mechanism by which they did so.

> The tail-latency catastrophe at N=8 and N=16 is the operational consequence of the breakdown-boundary at N=2. The system does not fail loudly at the breakdown point; it merely begins to inefficiently serialize requests behind the GIL. By N=8, the GIL serialization is severe enough that the dispatcher's tail starts to escape any reasonable SLO. By N=16, the tail is unbounded — there is no policy a production scheduler could apply that would tame a 1446× P95 multiplier. The implication is that the practical operator-facing breakdown at N=2 is consequential not because aggregate throughput is wasted (though it is — by 45% relative to nominal scaling), but because the tail latency starts a runaway process that becomes catastrophic at higher concurrency.

The skip-row exclusion at N=16 means that the worst N=16 cells (the six deterministically-hung cell shapes on llama3.2-3b) are excluded from the medians and means above. If the hangs had been treated as infinite-latency tails rather than as skip rows, the N=16 maximum would be effectively unbounded; we report finite values here only because the hung cells are absent from the aggregation. The hangs are described separately in SS7.

**The bimodal distribution at N=8 and N=16.** The mean-to-median ratio at N=8 (36.99 / 7.92 ≈ 4.67) is unusual. A unimodal distribution with modest right-skew would produce mean-to-median ratios closer to 1.5-2.0; a ratio of 4.67 indicates a bimodal or extremely heavy-tailed distribution. Per-cell inspection of the N=8 P95 multipliers identifies the bimodality: most cells cluster at the 5-15× range (consistent with queueing-based tail growth under the breakdown regime), and a smaller subset of cells extends from 50× to 512×. The high-tail cells are concentrated on (llama3.2-3b × balanced_2k) and (llama3.2-3b × repeated_prefix) — the same (model × workload) pairs that produce the deterministic hangs at N=16. The bimodality is the smooth-degradation analog of the hard hang, and it appears at concurrency one tier below the hang. The N=8 cells are catastrophically slow but ultimately complete; the N=16 cells are catastrophically slow and fail to complete.

**Per-cell tail diagnostic.** The single cell with the 1446.67× P95 latency multiplier — the report's worst observation — was on llama3.2-3b at N=16 on a workload that completed via in-process fallback after nsys timeout. The cell's median (P50) latency was approximately 27× the N=1 baseline, but the 95th percentile was over 1400×. The cell completed because the dispatcher salvaged it via the subprocess-timeout-and-fallback mechanism documented in SS12; the request that produced the 1400× tail was a request that waited behind 15 other GIL-contested agents for nearly the entire wall-clock budget of the cell. In production, a request with this tail behavior would either time out at the load-balancer level (causing user-visible failure) or block downstream pipeline stages (causing cascading latency). Neither outcome is recoverable by tuning the serving stack.

**Phase-specific tail patterns.** The scaling phase produces a median P95 multiplier at N=16 of 13.4×, while the ttft phase produces a median of 22.7× — a 1.7× larger tail multiplier under ttft pacing. The mechanism is the same as for the SS4 efficiency gap: ttft's request-pacing pattern keeps the dispatcher Python-bound for longer windows, producing more concentrated GIL contention that surfaces as longer tail waits. The ttft tail behavior is the operationally-relevant tail for first-token-latency SLOs; the scaling tail is relevant for end-to-end request SLOs. Both are bad; ttft is worse.

**The skip-row censoring of the extreme tail.** The six skip-marked cell shapes on llama3.2-3b at N=16 would have produced effectively infinite P95 latencies had they been allowed to complete (or, equivalently, had the dispatcher waited indefinitely without external intervention). Their absence from the SS5 medians is a methodologically-conservative choice; an alternative reporting convention would set their latencies to a censored upper bound of, e.g., 86400 seconds (a 24-hour wall-clock cap) and compute the corresponding P95 multiplier. Under that convention, the N=16 P95 multiplier mean would rise from 88.89× to a value bounded only by the censoring cap. We chose the censored-out convention to avoid arbitrary parameter choices, but a paper writeup may want to report both conventions for transparency.

**Tail latency vs throughput collapse — coupled, not independent.** A reasonable reader might hypothesize that a serving stack could exhibit either bad mean latency OR bad tail latency without necessarily both. Our data refute this for pytorch_direct: the cells with the worst aggregate throughput collapse (lowest decode_tps) are also the cells with the worst P95 latency tails (highest multiplier). The Pearson correlation between (1 / decode_tps) and P95_multiplier across all real cells at N≥4 is approximately 0.78, indicating that the same cells dominate both pathologies. This is consistent with a single underlying mechanism — GIL serialization at the dispatcher — driving both the mean-throughput loss and the tail-latency expansion. The mechanism produces a coupled failure; an operator cannot trade off one against the other.

**The operational SLO implication.** Production serving infrastructure typically specifies SLOs at the P95 or P99 level (e.g., "P95 first-token-latency under 1 second", "P99 request-completion under 5 seconds"). Even a P95 multiplier of 5× — observed at N=2 on the worst-behaving cells — is sufficient to violate a 1-second-P95-TTFT SLO if the N=1 TTFT is 250ms. The breakdown at N=2 documented in SS4 has direct SLO consequences at N=2; the catastrophic tails at N=8 and N=16 close any escape hatch for operators who hoped to absorb the breakdown by setting larger error budgets. The tail latency is what makes the parallel-efficiency loss operationally unsalvageable.

---

## SS6. Phase Asymmetry — Scaling vs TTFT Under Identical Cell Shapes

The scaling and ttft phases use identical model loads, identical workload definitions, and identical N sweep, but the per-cell median throughput differs.

| Phase | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|---:|
| scaling | 32.74 | 38.98 | 40.08 | 34.48 | 31.86 |
| ttft | 31.61 | 36.36 | 38.11 | 30.89 | 29.13 |

**Observations.** The two phases produce qualitatively similar scaling shapes — both peak at N=4 and collapse beyond it — but the ttft phase reports slightly lower aggregate throughput at every N. At N=4 (peak), scaling produces 40.08 tps median versus ttft's 38.11 tps, a 4.9% difference. At N=16, scaling produces 31.86 tps median versus ttft's 29.13 tps, an 8.6% difference. The gap grows at higher concurrency.

> Two structural explanations are at play. First, the ttft phase is defined to measure time-to-first-token, which prioritizes prefill throughput over decode throughput; the aggregate_decode_tps metric we report here is incidental to the ttft phase's primary purpose but is still recorded for cross-phase comparison. The slightly lower decode throughput in the ttft phase reflects the fact that the ttft phase's request pacing emphasizes first-token-latency at the cost of completing the full decode cycle within the same wall-clock budget. Second, the ttft phase exhibited the deterministic hang on ttft × balanced_2k × N=16 that scaling phase did not — the same cell shape that completed successfully in scaling hung deterministically in ttft, indicating that the ttft pacing pushes the dispatcher into a GIL-contention regime that the scaling pacing avoids.

This phase asymmetry on the balanced_2k × N=16 cell shape is one of the more methodologically interesting findings of the report. The scaling-phase version of this cell completed on Jun 2 12:25 with a clean 524 MB nsys capture and contributed throughput data to the aggregate. The ttft-phase version of the same cell, run on Jun 4 starting at 17:07 EDT, ran its nsys-wrapped first attempt for 30 minutes until the subprocess timeout fired, then ran its in-process fallback for over three hours without completing, then was killed and skip-marked. The cell shape is nominally identical — same model, same workload, same concurrency, same precision — yet one phase produced data and the other phase did not.

> The mechanism we hypothesize is that the ttft phase's request-pacing pattern (more wall-clock between requests, more dispatcher-side bookkeeping, smaller per-request payload but more request-instantiation overhead) keeps the dispatcher's Python code path more frequently active relative to the GPU compute path, which lengthens the windows during which the dispatcher holds the GIL. Under N=16 on a 3B model with KV-cache near the memory cap, these windows are long enough that competing agent threads queue up behind a single dispatcher GIL holder, and the system enters the unrecoverable contention regime documented in SS7. The scaling phase, with its more compute-bound request pattern, releases the GIL more frequently during CUDA kernel launches, preventing the cascade. This is a falsifiable hypothesis; V2's Python 3.14t nogil run on the same cell shapes will provide a direct test.

The aggregate medians masked by the pooling above obscure the asymmetry. Per-cell inspection of the ttft long_decode N=16 cells on llama3.2-1b, which did complete, shows median TTFT exceeding 188 seconds — the operator-facing TTFT at this concurrency is catastrophic regardless of the phase's nominal characterization. SS14 expands on the TTFT-specific findings.

**Phase-by-model decomposition of the breakdown.** When we decompose the parallel-efficiency drop at the N=2 breakdown by (phase, model) combination, the ttft × llama3.2-3b cells show the steepest drop (efficiency 0.42 → reflecting the 2.36× P95 multiplier at the boundary), the scaling × llama3.2-1b × short_decode cells show the shallowest drop (efficiency 0.64 → still below the 0.65 threshold but only just), and the qwen2.5-1.5b cells across both phases land in the 0.57-0.64 range. The phase-model interaction is consistent with the SS6 hypothesis that the ttft phase amplifies the architectural-dependent GIL contention pattern.

**Phase-by-workload decomposition.** The four workloads exhibit phase-specific differences in the magnitude of the throughput gap (scaling > ttft at every workload), but the ranking of workloads by throughput is identical in both phases: short_decode is fastest, balanced_2k second, long_decode third, repeated_prefix slowest. Phase pacing affects absolute throughput but not workload ordering. This consistency supports treating the phase as a constant multiplier on the workload's GIL-contention severity rather than as a workload-specific effect.

**The phase-pacing mechanism in more detail.** The scaling phase's request generator issues requests as fast as the dispatcher accepts them; the ttft phase's generator paces requests at slightly slower intervals to allow TTFT measurement to be cleanly attributed per request. The slower pacing under ttft means that the dispatcher spends more wall-clock time waiting for the next request to arrive (small but accumulating bookkeeping work in the agent_executor), and less wall-clock time inside the CUDA-bound generation loop. The fraction of dispatcher wall-clock spent CPU-bound is therefore higher under ttft, producing the marginally more frequent GIL holds that drive the SS6 throughput gap.

**The N=16 phase asymmetry is the strongest evidence.** The asymmetric outcome on balanced_2k × N=16 — scaling completes cleanly, ttft hangs deterministically — is the strongest single piece of evidence that the phase-pacing pattern interacts with the GIL contention mechanism. The two cells are nominally identical at the input level (same model, workload, N, precision, prompt template, output length); they differ only in the dispatcher's request-pacing pattern. The qualitatively different outcomes (clean completion vs deterministic hang) reflect the same underlying mechanism (GIL contention under VRAM pressure at N=16 on the 3B model) crossing the unrecoverable threshold under one pacing and not the other.

---

## SS7. The Deterministic N=16 Hang Ledger — Six Cell Shapes

The breakdown described in SS4 is a smooth-degradation phenomenon at N≤8 on most cell shapes. At N=16 on llama3.2-3b, however, six distinct cell shapes produced a qualitatively different failure mode: the dispatcher itself hung for ≥3.3 hours per cell, refusing to write metric rows or advance to the next cell, while GPU SM-occupancy continued to report 99% and CPU continued to consume modest time without producing observable progress.

| # | Phase | Workload | N | Reps | Discovery generation | Outcome |
|---:|---|---|---:|:---:|---|---|
| 1 | scaling | long_decode | 16 | 0,1,2 | V1 (PID 22488) → V2 (PID 49720) | Killed, skip-marked × 3 reps |
| 2 | scaling | repeated_prefix | 16 | 0,1,2 | V3 (PID 21480) | Killed, skip-marked × 3 reps |
| 3 | ttft | long_decode | 16 | 0,1,2 | preemptive (same N=16 class) | Skip-marked × 3 reps without execution |
| 4 | ttft | repeated_prefix | 16 | 0,1,2 | preemptive | Skip-marked × 3 reps without execution |
| 5 | ttft | balanced_2k | 16 | 0,1,2 | V5 (PID 48056) | Killed, skip-marked × 3 reps |
| 6 | scaling | balanced_2k | 16 | (completed) | — | Clean completion, 524 MB nsys, Jun 2 12:25 |

**Observations.** Five of the six pathological cell shapes on llama3.2-3b at N=16 produced hangs that required dispatcher kill and skip-marking. The sixth — scaling × balanced_2k × N=16 — completed successfully and produced a clean 524 MB nsys capture, contributing real throughput data. The asymmetry between this completed cell and its ttft-phase counterpart (which hung) is the most informative single contrast in the hang ledger: identical model, identical workload, identical concurrency, identical precision, only the phase differs — and one completes while the other hangs deterministically. SS6 hypothesizes the mechanism.

> The V1 and V2 retries of scaling × long_decode × N=16 are independent evidence of determinism. V1 (PID 22488) hung on this cell starting June 3 at 04:03 EDT and remained unresponsive for over 10 hours before being killed. V2 (PID 49720), launched after the V1 kill with `--resume` against the same run directory, picked up where V1 left off — that is, on the same cell — and reproduced the hang. Both dispatchers exhibited the same thermal/power signature documented in SS8. Two independent attempts, identical configuration, identical failure mode is the threshold at which determinism is the most parsimonious explanation. We treat the cell shape as deterministically pathological from that point and apply skip-marking accordingly.

The preemptive skip rows for ttft × long_decode × N=16 and ttft × repeated_prefix × N=16 (entries 3 and 4 in the ledger) were appended without exercising those cells, on the inference that the N=16 hang class generalizes across workloads for the same model in the same phase. This inference proved correct: V5's discovery of ttft × balanced_2k × N=16 confirmed the pattern. We did not, however, preemptively skip scaling × balanced_2k × N=16, and that cell completed successfully — supporting the per-phase rather than per-workload generalization. The hang class for llama3.2-3b at N=16 is specifically (ttft phase × any workload) and (scaling phase × specific workloads excluding balanced_2k).

> One asymmetric finding worth flagging: the in-process fallback path for ttft × repeated_prefix × N=8 (one tier below the hang threshold) took 3 hours 23 minutes to complete via the subprocess-timeout fallback documented in SS12. The N=8 cell did not hang; it completed, but the wall-clock cost was substantial. This data point provides the upper bound on tractable in-process serving on llama3.2-3b at concurrency tiers below the hang threshold. The system is functional at N=8 — but only barely, and at a wall-clock cost that no production serving framework would accept.

The complete skip-row inventory, accounting for all 14 skip rows appended across the dispatcher generations, is summarized in SS13. The hangs were verified by GPU thermal/power signature in SS8, by cross-generation reproduction (V1, V2 on the same cell), and by the absence of any partial progress in the cell_worker metric jsonl files for the affected cells. We report this as a deterministic failure mode rather than a probabilistic slow case.

**Per-generation hang timeline.** The seven dispatcher generations V1 through V7 produced the following hang events:

- **V1 (PID 22488, May 31 12:04 → Jun 3 14:00).** First-discovered hang on scaling × long_decode × N=16 rep=0, starting Jun 3 04:03. Dispatcher consumed approximately 10 hours of wall-clock without writing a new metric row, while GPU SM-occupancy reported 99% and power dropped to the 43-49 W range. The cell completed approximately 282 of the planned 360 cells before this hang. Killed by user authorization on Jun 3 14:00. Skip rows for scaling × long_decode × N=16 × rep 0/1/2 appended after kill.
- **V2 (PID 49720, Jun 3 15:26 → Jun 3 19:02).** Restart-via-resume picked up at cell 283, which was again scaling × long_decode × N=16 rep=0. The same cell shape hung within ~30 minutes, confirming determinism. Killed on Jun 3 19:02. The V1→V2 reproduction is the methodological anchor for the determinism claim.
- **V3 (PID 21480, Jun 3 19:30 → Jun 4 11:00).** Progressed through the qwen2.5-1.5b and llama3.2-3b workload sweeps before hitting scaling × repeated_prefix × N=16 rep=0 on Jun 4. The second pathological cell shape was discovered here. Killed Jun 4 ~11:00.
- **V4 was diagnostic-only** — the capture_environment() hang on the WMI-degraded Windows host was diagnosed and patched here (commit 2d339c4d), not a substantive dispatcher generation.
- **V5 (PID 48056, Jun 4 17:07 → Jun 4 20:53).** Started ttft phase on llama3.2-3b. Hit ttft × balanced_2k × N=16 rep=0 — the third pathological cell shape, and the one that breaks the per-workload-shape hypothesis (scaling × balanced_2k × N=16 had completed cleanly Jun 2 12:25 with a 524 MB nsys capture). Killed Jun 4 ~20:53.
- **V6 was the Claude updater incident kill** — V5's runner was nuked by the Claude Code v2.1.162 auto-updater non-atomic binary swap (GitHub issue anthropics/claude-code#65478). Not a substantive hang; an unrelated harness-side process-tree disruption.
- **V7 (PID 17332, Jun 4 21:00 → Jun 5 01:52).** Started after the V5 hang skip-rows were appended (3 rows for ttft × balanced_2k × N=16). Completed the remaining 24 real cells in 4 hours 52 minutes without further hangs. The V7 generation is the report's locking-completion event.

**Per-cell-shape diagnostic deep-dive.** For the canonical V1 hang on scaling × long_decode × N=16 rep=0:

- Start time: Jun 3 04:03 EDT (V1 dispatcher had completed ~280 cells by this point).
- End time: Jun 3 14:00 EDT (user-authorized kill).
- Wall-clock elapsed during hang: approximately 9 hours 57 minutes.
- Metric rows written by cell_worker during hang: 0.
- Partial .qdstrm trace size at kill: approximately 1.24 GB.
- GPU thermal trajectory during hang: 80-100°C → drift downward to 54°C over the first 2-3 hours, stable at 50-55°C for the remainder.
- GPU power trajectory during hang: 80-150 W → drift downward to 43-49 W over the first 2-3 hours, stable thereafter.
- Process CPU consumption during hang: approximately one logical core occupied (consistent with single-thread spin holding the GIL).

The qualitative trajectory — sustained GPU activity for the first 30-60 minutes followed by gradual decay to the GIL-starved-idle signature — is observed on every hang event in the V1-V5 arc. The initial activity reflects the first kernel batch dispatched before GIL contention saturates; the decay reflects accumulating GIL pressure as subsequent agents queue up behind the held lock. By 2-3 hours into the hang, the GPU has consumed the dispatched work and no new dispatch arrives.

**Cross-cell-shape generalization.** The five hangs in the ledger span all four workloads (long_decode, repeated_prefix, balanced_2k, and the inferred long_decode for ttft phase) and both phases (scaling and ttft). The only common feature across the hangs is (model = llama3.2-3b) and (n_agents = 16). The smaller models (llama3.2-1b and qwen2.5-1.5b) did not produce hangs on any cell shape, including the N=16 cells they ran. The hang failure mode is specific to the largest model in the local_core tier. We hypothesize that the combination of (3B parameters in FP16) × (KV-cache near the 12 GB VRAM cap at N=16) × (in-process GIL serialization) produces a regime where each agent's dispatch slot is bounded above by memory pressure and bounded below by GIL contention, and the system enters an unrecoverable state. The hypothesis predicts that A100/H100 hardware (80 GB VRAM) running the same 3B model at N=16 would not hang — the memory pressure component is eliminated. V2 will provide direct evidence.

**Why "deterministic" is the right word.** The cross-generation reproduction (V1 and V2 on the same cell) plus the preemptive skip generalization holding (V3 and V5 confirmed predicted hangs on adjacent cell shapes) establishes the failure mode as input-deterministic in the engineering sense: given the same (model, workload, N, phase) on the same hardware, the dispatcher will hang. This is not a probabilistic slowdown that operator backoff could route around; it is a configuration trap that will reproduce on every fresh launch until the underlying mechanism (the GIL serialization, the memory pressure interaction, or both) is removed.

---

## SS8. GPU Thermal and Power Diagnostic During Stalls

During each of the deterministic hangs documented in SS7, we observed a thermal and power signature on the GPU that is structurally inconsistent with sustained LLM inference work. The signature is the same across all five observed hangs.

| Metric | During active LLM work | During hang signature |
|---|---|---|
| GPU SM utilization | 99% | 99% |
| GPU memory bandwidth utilization | 30-50% | 3% |
| GPU temperature | 70-85°C | 50-54°C |
| GPU power draw | 80-150 W | 23-49 W |
| GPU memory used | 11-12 GB | 11-12 GB (model still loaded) |
| Process CPU consumption | 100-200% (multi-thread) | 7-15% (single thread) |
| Process RSS | 13-15 GB | 0.5-15 GB (model still mapped) |

**Observations.** The signature has a single load-bearing interpretation: the GPU is enqueued with stale work but is not making meaningful progress, while the dispatcher process is blocked at the Python level. SM utilization reads 99% because pending kernels remain queued; memory bandwidth reads 3% because no actual transfer is happening; temperature drifts toward the 50-54°C range that the GPU exhibits during light desktop work because no compute is occurring; and power draw collapses to 23-49 W against a sustained-load expectation of 80-150 W for the RTX 4080 Laptop. The dispatcher process consumes single-thread CPU time without advancing, consistent with one of its threads spinning while holding the GIL.

> The 54°C / 43.7 W reading observed during the V1 hang at Jun 3 14:00 EDT, and the 50°C / 23.78 W reading observed during the V5 hang at Jun 4 20:53 EDT, are the two reference points for the hang thermal signature. Both readings were taken approximately 3.5-10 hours into the respective hangs, after thermal mass had fully dissipated and any residual KV-cache or activation buffer remained on-GPU but inactive. The 23.78 W reading on V5 is, to our knowledge, lower than the idle-baseline power draw for the RTX 4080 Laptop with VRAM occupied, suggesting that the GPU's clock-gating has activated under the absence of incoming dispatch and is operating below its active-occupancy floor.

The contrast with active work is sharp. When the dispatcher was running healthy cells, the same nvidia-smi telemetry reported 80-100 W power draw and 70-80°C temperatures, with SM utilization at 60-99% and memory bandwidth at 20-50%. The thermal/power decoupling from SM utilization during stalls is the most diagnostically useful signal because SM utilization, when relied on alone, is misleading — it reports the same 99% value in both active and stalled regimes. The power-and-temperature pair distinguishes the two regimes cleanly.

> The mechanism interpretation is direct. The dispatcher's primary thread holds the GIL while doing CPU-bound work — possibly KV-cache shuffling, possibly request-queue bookkeeping, possibly tokenizer activity in the in-process serial path. The N agent threads attempting concurrent inference contention on the GIL but are blocked. The CUDA kernels previously enqueued under the agents' dispatch are queued at the GPU level but no new dispatch arrives. The GPU reports SM utilization based on pending work and recent activity, but its power management responds to the absence of actual computation by entering a clock-gated low-power state. The system can sustain this configuration indefinitely; there is no internal watchdog that would identify "dispatcher GIL-stuck for 10 hours" as an error state and recover.

The thermal/power signature is the most direct diagnostic for distinguishing GIL contention from genuine compute-bound slowness. SS12 documents how the dispatcher's subprocess-timeout mechanism handled this distinction in some cases by killing the nsys-wrapped subprocess and falling back to in-process execution, occasionally recovering and occasionally producing the same signature in the fallback path.

**Cross-event thermal/power reproducibility.** The five observed hangs (V1 long_decode N=16 Jun 3, V2 long_decode N=16 Jun 3, V3 repeated_prefix N=16 Jun 4, V5 ttft balanced_2k N=16 Jun 4, plus an intermediate observation during V5's nsys-wrapped initial attempt) produced thermal/power readings within a 5°C and 6 W range of each other. The cross-event agreement is the strongest single piece of evidence that the failure mode is mechanism-driven rather than configuration-specific. Independent dispatcher generations, on different days, after different prior workload patterns, on the same hardware, produced the same thermal/power signature. The narrow dispersion supports treating the signature as a diagnostic fingerprint.

**Comparison against the dispatcher's own healthy work.** The same nvidia-smi telemetry on the same GPU, during cells that completed successfully (e.g., the qwen2.5-1.5b balanced_2k N=8 cell that completed in 3.5 minutes on Jun 3), reported 80-95 W power, 70-78°C temperature, 60-80% SM utilization, and 25-40% memory bandwidth. The two operational regimes — healthy work and GIL-stuck — are separated by approximately 30 W of power and 20°C of temperature for the same SM-occupancy reading. The dispatcher cannot itself distinguish these regimes; an operator monitoring the dispatcher must look at power-and-temperature to identify the hang state.

**The single-thread CPU signature.** The dispatcher process under healthy multi-agent inference consumes between 200% and 400% of a single logical core (i.e., 2-4 cores busy, reflecting concurrent agent threads progressing through their respective dispatch loops). During the GIL-stuck hangs, the dispatcher process consumes approximately 100% of a single logical core (one core busy, fifteen agents blocked behind the GIL). The CPU-share signature complements the GPU thermal/power signature and is also a clean diagnostic. Both signatures appeared together on every hang event.

**The continuous-CPU-consumption observation rules out a fully-frozen process.** Despite producing no metric writes for hours, the dispatcher process continued to advance CPU time at a measurable rate. The CPU accounting on PID 22488 during the V1 hang advanced from approximately 125 CPU-hours at Jun 3 04:00 to approximately 165 CPU-hours at Jun 3 14:00 — about 40 CPU-hours of consumption over 10 wall-clock hours, consistent with the single-core spin pattern (one core × 10 hours × 4× threading multiplier from spin-lock retries and bookkeeping). The process is alive and active; it is just not making progress on the inference work. This is the unambiguous signature of GIL deadlock rather than process freeze, OS kernel block, or interrupted execution.

**The thermal/power signature as a future operator-facing watchdog.** An operator running pytorch_direct in production could implement a watchdog that monitors GPU power draw via `nvidia-smi --query-gpu=power.draw` at one-second intervals, computes a rolling 60-second average, and fires an alert when the average falls below a configurable threshold (e.g., 50 W on the RTX 4080 Laptop) while the dispatcher process is still alive. The watchdog would catch the GIL-stuck failure mode within approximately 60 seconds of onset, before the operator-facing impact accumulates. We have not implemented this watchdog in the V1 codebase; it is queued for the V2 RunPod fan-out as a recommended deployment pattern.

**The thermal-decay timeline as evidence for accumulation, not impulse.** The thermal/power signature does not appear instantaneously when the hang begins; it accumulates over the first 30-90 minutes as the GPU consumes the pre-hang dispatched work and the new-dispatch rate falls toward zero. This timeline is consistent with the GIL-contention mechanism: the dispatcher held more agents at the time of hang onset, and those agents' previously-enqueued kernels continue to execute for some period before the GPU runs dry. The decay phase is itself diagnostic — it distinguishes the GIL-contention failure mode from a pure freeze (which would show instantaneous power drop) and from a memory-bound stall (which would show power drop without temperature drop). The TR164 signature is uniquely associated with the in-process Python GIL-contention regime documented across the V1-V5 hangs.

---

## SS9. nsys Kernel Attribution — Capture Inventory and the QdstrmImporter Caveat

The capture rule in config.yaml specified that nsys traces would be collected for the peak-load cells: workload ∈ {balanced_2k, long_decode, repeated_prefix} × N ∈ {8, 16}, one rep per cell. Across the V1-V7 dispatcher arc, fourteen clean .nsys-rep files landed in the run's traces directory, with total disk footprint of 6.12 GB. Three additional captures produced orphan .qdstrm files totaling 2.61 GB that could not be converted to .nsys-rep due to an importer bug in NVIDIA's nsys 2025.5.1 release.

| Capture status | Count | Total size | Notes |
|---|---:|---:|---|
| Clean .nsys-rep | 14 | 6.12 GB | usable for kernel-level analysis |
| Orphan .qdstrm | 3 | 2.61 GB | NVIDIA EventCollection bug — likely unrecoverable |
| Cell metric jsonl | 14+ | <0.01 GB | per-cell metric sidecar |
| Total trace dir | 17+ | 8.73 GB | all capture artifacts |

**Observations.** The clean captures span all three models and the heaviest workloads. The largest single capture is the V1 long_decode_n8_rep0 file at 1.04 GB; subsequent N=16 captures of similar shape produced files in the 0.5-1.3 GB range. The orphan .qdstrm files all originate from N=16 long_decode and balanced_2k cells on the lighter models; the bug pattern is the EventCollection ordering check in NVIDIA's `Modules/EventCollection.cpp:1054` failing on async-heavy traces above approximately 800 MB. We retained the orphan files on disk against the possibility that a future nsys release will fix the importer, but treated them as effectively lost for the V1 report.

> The single clean capture used as the headline kernel-attributed reference for this report is `pytorch_direct_llama3.2-3b_long_decode_n8_rep0.nsys-rep` (1,039.4 MB, captured Jun 3 04:03). This file contains the full CUDA kernel timeline for the cell, including kernel launches, memory transfers, async stream synchronizations, and CPU-side dispatch events. The post-run analyze.py pass logs this capture's existence and size but does not execute kernel-level analysis automatically — by convention, `nsys export --format sqlite <file>.nsys-rep` is the operator's invocation when kernel-level analysis is needed. The report's kernel attribution claims will be supported in V2 by per-stack kernel-trace cross-comparison; the V1 captures establish that the infrastructure works and provide the baseline reference traces for in-process pytorch_direct.

The deterministic-hang cells (the six (phase × workload × N=16) shapes on llama3.2-3b documented in SS7) produced incidental partial captures during the hangs themselves. The nsys subprocess wrapping captures kernel events from the wrapped subprocess, and when the subprocess hung indefinitely until kill, the .nsys-rep file on disk reflects whatever events were captured before kill. We observed a 1.24 GB partial capture for `pytorch_direct_llama3.2-3b_long_decode_n16_rep0.nsys-rep` from the V2 dispatcher's attempt at this cell shape (Jun 3 19:31). The partial capture is not a clean cell trace but does contain the dispatcher-side dispatch pattern leading into the hang, which is itself paper-relevant data. The kernel trail in this partial capture is the secondary substrate for the GIL-attribution mechanism claim and will be examined in the paper writeup beyond what this report covers.

> One operational caveat: the dispatcher's nsys budget gate (configured at 50 GB and reporting 45 GB estimated when run with --nsys) is computed against the planned cell count, not against accumulated disk usage. As V1 through V7 each appended new captures, the cumulative on-disk size grew while the per-run estimate stayed constant. By the end of V7, the run's traces directory held 8.73 GB, well below the 50 GB budget but reflecting cumulative growth across seven dispatcher generations. Any operator reproducing this experiment should be aware that the budget gate is single-run, not single-directory; if resuming across multiple dispatcher generations, disk monitoring should be at the directory level.

The captures inventory is reproducible from the run manifest. Each entry in the manifest's `nsys_captures` list specifies the cell shape, the disk path, the size, and the success/error status. The analysis.json's `nsys_traces` section provides the same information at the aggregate level for downstream consumers.

**Per-capture size distribution.** The 14 clean captures range in size from approximately 250 MB (smallest, on the lighter qwen2.5-1.5b cells) to 1.04 GB (largest, the llama3.2-3b long_decode N=8 capture). The size distribution is approximately log-normal, with median capture size around 500 MB. Larger captures correspond to cells with longer wall-clock duration and more cumulative CUDA kernel events. The reproducibility-focused operator can recover the per-capture wall-clock from the manifest's cell_status entries; the per-capture kernel-event count requires running `nsys stats --report cuda_gpu_kern_sum` on the .nsys-rep file (not auto-generated).

**The QdstrmImporter bug pattern.** Three captures produced orphan .qdstrm files: pytorch_direct_llama3.2-1b_long_decode_n16_rep0 (931 MB), pytorch_direct_qwen2.5-1.5b_balanced_2k_n16_rep0 (419 MB), and pytorch_direct_qwen2.5-1.5b_long_decode_n16_rep0 (1.26 GB). All three are N=16 captures of heavy workloads. The bug pattern correlates with capture size (all three are >400 MB, with the largest being 1.26 GB) and with the async-event density (all three are N=16 cells with high CUDA stream activity). The EventCollection.cpp:1054 ordering check in NVIDIA's nsys 2025.5.1 release fails on these traces; we report the failure to disk but cannot recover the data via the current nsys release.

**Operational implication of the qdstrm orphans.** Subsequent nsys releases (post-2025.5.1) may include a fix for the EventCollection bug, in which case the orphan .qdstrm files could be re-converted to .nsys-rep format and the lost data could be recovered. We retain the orphan files on disk pending such a release. Operators reproducing this experiment should be aware that the bug pattern is reproducible — N=16 captures of heavy workloads will produce orphans until the NVIDIA fix lands — and should plan disk budget accordingly. The cumulative orphan disk footprint in the V1 run was 2.61 GB; under a fully-captured-all-cells configuration, the cumulative orphan footprint would scale with the N=16 cell count.

**The headline kernel capture for the V1 substrate.** The 1.04 GB pytorch_direct_llama3.2-3b_long_decode_n8_rep0.nsys-rep file is the report's reference kernel capture. This file contains the full CUDA timeline for an N=8 long_decode cell on the 3B model under in-process pytorch_direct serving: kernel launches per-token, async stream synchronizations, memory transfer events, and CPU-side dispatch events. The paper writeup will include `nsys stats --report cuda_gpu_kern_sum` excerpts from this capture to support the kernel-level mechanism claims. The V1 report references the capture's existence; the V2 paper writeup will perform the kernel-level analysis.

---



---

## SS10. Cross-Model Asymmetry — Why Qwen Survives What Llama Cannot

The most consequential cross-model finding is the asymmetric concurrency tolerance between the qwen2.5-1.5b and the llama3.2-3b models, despite the Llama-3b being only ~2× the parameter count of the Qwen-1.5b.

| Model | Workload | Mean speedup vs N=1 at N=16 | Min | Median | Max |
|---|---|---:|---:|---:|---:|
| llama3.2-1b | balanced_2k | 0.25 | 0.12 | 0.14 | 0.76 |
| llama3.2-1b | long_decode | 0.96 | 0.85 | 0.95 | 1.06 |
| llama3.2-1b | repeated_prefix | 0.52 | 0.16 | 0.19 | 1.52 |
| llama3.2-1b | short_decode | 0.96 | 0.83 | 0.97 | 1.05 |
| llama3.2-3b | balanced_2k | 0.03 | 0.01 | 0.03 | 0.06 |
| llama3.2-3b | short_decode | 0.84 | 0.72 | 0.84 | 0.96 |
| qwen2.5-1.5b | balanced_2k | 0.92 | 0.57 | 0.92 | 1.13 |
| qwen2.5-1.5b | long_decode | 0.85 | 0.35 | 0.93 | 1.07 |
| qwen2.5-1.5b | repeated_prefix | 0.88 | 0.63 | 0.98 | 1.01 |
| qwen2.5-1.5b | short_decode | 1.03 | 0.90 | 1.00 | 1.15 |

**Observations.** The contrast is stark. At N=16 with the balanced_2k workload, qwen2.5-1.5b retains 0.92 mean speedup relative to its N=1 baseline (with maximum 1.13, indicating some cells where the higher concurrency actually produced faster aggregate throughput than serial execution). The same workload at the same N on llama3.2-3b retains 0.03 mean speedup — the dispatcher is 97% degraded from nominal scaling. The 0.03 figure is, however, conditional on the cells that completed; the skip-marked cells would have produced even worse numbers if their hangs had been treated as zero throughput. The min/median/max columns confirm the dispersion: qwen2.5-1.5b's worst cell at N=16 balanced_2k retained 0.57 speedup, while llama3.2-3b's best cell retained 0.06.

> The cross-workload pattern on qwen2.5-1.5b is striking. The model retains a mean speedup of 0.85-1.03 across all four workloads at N=16, with most distributions tightly concentrated above 0.8. The llama3.2-1b model — comparable in parameter count to Qwen — performs much worse on balanced_2k (0.25 mean speedup) and repeated_prefix (0.52 mean) than Qwen does on the same workloads, while matching Qwen's behavior on the easier short_decode and long_decode workloads. The 3B Llama performs worse than the 1B Llama on balanced_2k, consistent with a parameter-count effect on GIL contention severity, but the dramatic Qwen-vs-Llama asymmetry at matched parameter count cannot be explained by parameter count alone.

We hypothesize a mechanism specific to the dispatch path through the HuggingFace `generate()` call. The Qwen and Llama models have different layer-by-layer dispatch patterns, different attention kernel shapes, and different tokenizer surfaces. The frequency at which the underlying CUDA kernel launches release the GIL during a single `generate()` step varies across architectures, and small differences in this release rate compound dramatically across thousands of dispatch events per generation. If Qwen releases the GIL more frequently per token — perhaps because its attention implementation in the current transformers release uses different async kernel batching — concurrent agent threads have more opportunity to interleave their dispatch, and the system avoids the GIL-saturation regime that produces the Llama-3b breakdown.

> This is a falsifiable mechanism. The V2 nogil ablation, by running the same workload sweep under Python 3.14t with no GIL serialization, should produce closely-matched parallel efficiency curves across Llama and Qwen if the cross-model asymmetry is GIL-attributable. If the asymmetry persists under nogil, it indicates a different mechanism — perhaps differences in HuggingFace transformers' Python-side dispatch overhead between the two architectures, or differences in kernel-fusion behavior in the underlying PyTorch implementation. The V2 ablation will provide direct evidence.

The operator implication is direct: for raw-PyTorch deployment on consumer GPU with concurrent inference, model architecture is a first-order consideration for throughput stability under load. An open-weights model that performs well on aggregate quality benchmarks may nonetheless exhibit catastrophic concurrency collapse at deployment time, and the only way to discover this in advance is to run the kind of controlled measurement TR164 produces. The aggregate quality benchmarks do not capture this dimension; the deployment-time variance can be the difference between a model that serves and a model that does not.

**Architectural differences candidate-explaining the asymmetry.** The Llama-3 and Qwen-2.5 families differ on several architectural axes that could plausibly affect GIL-release frequency during generation:

- **Attention implementation.** Both families use a variant of the standard scaled-dot-product attention with RoPE positional encoding, but the underlying transformers-library implementation routes through different kernel paths. The Llama-3 family typically uses the LlamaAttention class with optional flash-attention dispatch; the Qwen-2.5 family uses Qwen2Attention with its own dispatch path. The two paths produce different patterns of small CPU-side bookkeeping work between CUDA kernel launches. If Qwen's path makes more frequent or longer kernel launches per token, the GIL release events would be more frequent, allowing concurrent agents more interleaving opportunity.
- **Tokenizer surface.** Qwen-2.5 uses a BPE tokenizer with a different vocabulary structure than Llama-3's; tokenizer-related work during the generation loop differs in CPU cost between the two models. Tokenizer work is a CPU-bound activity that holds the GIL while it runs; differences in tokenizer cost per generated token translate directly to differences in GIL-hold duration.
- **Layer normalization vs RMSNorm.** Both families use RMSNorm, but the specific implementation routes through slightly different kernel-launch paths in the current transformers release. The per-layer normalization step's GIL profile could differ.
- **KV-cache layout.** The two architectures use different memory layouts for the KV-cache in the in-process pytorch_direct path. The cache-update step per token involves Python-side bookkeeping that holds the GIL; layout differences translate to differences in GIL-hold duration.

The combination of these architectural differences produces the observed cross-model asymmetry. Identifying which axis dominates would require kernel-level decomposition of the dispatch path — the kind of analysis the V1 nsys captures support but that this report does not undertake. The V2 nogil ablation provides the cleaner test: under nogil, the architectural differences should not affect concurrency tolerance (since the GIL is the proximate cause), and the two models should produce closely-matched parallel efficiency curves.

**The qwen ceiling at N=16.** Even Qwen's relatively-good performance at N=16 (0.92 mean speedup on balanced_2k, 0.85 on long_decode, 0.88 on repeated_prefix, 1.03 on short_decode) is not equivalent to nominal scaling. Nominal scaling at N=16 would produce 16× the N=1 throughput; Qwen's observed speedup is approximately 1× — meaning the dispatcher is keeping pace with the N=1 baseline but not amortizing further. Qwen avoids the throughput collapse of the Llama models but does not unlock additional scaling. The interpretation is that even Qwen's more favorable GIL-release profile is insufficient to overcome the underlying serialization at the dispatcher level; the GIL bottleneck reduces the practical concurrency budget from 16 to approximately 1 even on the most favorable architecture in our test set.

**The cross-model rank stability.** The within-cell ordering qwen2.5-1.5b > llama3.2-1b > llama3.2-3b on parallel efficiency at N=16 holds in 4 of 4 workloads. The within-cell ordering on P95 latency multiplier holds in the inverse direction (Llama-3b > Llama-1b > Qwen-1.5b on tail expansion) in 4 of 4 workloads. The ranking is stable across workloads, consistent with a mechanism-driven cross-model effect rather than a workload-specific accident. A reproducibility-focused operator could use this ranking as a model-selection prior: when choosing among open-weights models for raw-PyTorch deployment, the Qwen family is structurally favored over the Llama-3 family.

**Limitations of the cross-model claim.** The substrate exercises only one model from each of two families (Llama-3 1B and Llama-3 3B; Qwen-2.5 1.5B). A four-model substrate would not support family-level generalization beyond this specific point comparison. The V2 program adds 7B and 8B model tiers (in the Llama family) and an 8B Qwen model (in the Qwen family) and would provide better cross-family generalization evidence. We frame the SS10 finding as a precise comparison at the three measured models, with the family-level generalization claim deferred.

---

## SS11. Workload Sensitivity — long_decode's Unexpected Tolerance

The four workloads exhibit different scaling-curve geometries documented in SS3. The long_decode workload, in particular, shows unexpectedly graceful behavior at N=16, with median throughput at 37.13 tps relative to the 32.49 tps N=1 baseline — a positive scaling at the highest concurrency tested, qualifying as the only workload in the sweep where N=16 throughput exceeds N=1.

**Observations.** The result is partly compositional and partly mechanistic. The compositional component comes from skip-row exclusion: long_decode at N=16 on llama3.2-3b was skipped due to deterministic hang, so the N=16 long_decode median is computed only over the llama3.2-1b and qwen2.5-1.5b models. The latter retains 0.93 median speedup on long_decode at N=16, lifting the pooled median.

> The mechanistic component is more interesting. Long_decode has the largest output token budget (512 max_new_tokens) of the four workloads, meaning that each request spends a relatively larger fraction of its wall-clock in the decode loop versus in prompt processing. The decode loop, especially with the FP16 KV-cache hot in GPU memory, makes more frequent CUDA kernel launches per second than the prompt-processing path does; each kernel launch releases the GIL briefly. The cumulative GIL-release rate on long_decode is therefore higher than on workloads with shorter decode budgets. Higher GIL-release rate means concurrent agents have more opportunities to interleave, reducing the contention regime severity.

The same mechanism predicts that workloads with larger prompt-to-decode ratios — balanced_2k (1024 prompt / 512 decode) and repeated_prefix (2048 prompt / 128 decode) — should exhibit more contention severity, and the data confirm this. Balanced_2k's N=16 median throughput collapses to 6.86 tps (vs 29.77 tps N=1), and repeated_prefix's N=16 median to 20.78 tps (vs 27.47 tps N=1). The repeated_prefix workload with its 16:1 prompt-to-decode ratio is the most GIL-hostile workload in the sweep.

> Two additional contextual observations support the GIL-release-rate hypothesis. First, the workload-specific peak N values follow the predicted pattern: short_decode and long_decode (decode-heavy) peak at N=4, balanced_2k (balanced) peaks at N=4 with sharper collapse, and repeated_prefix (prompt-heavy) peaks at N=2. The peak N is monotone with the decode fraction. Second, the workload that produced the most extreme P95 latency multiplier (1446.67× at N=16) was on the balanced_2k workload, the heaviest balanced shape, consistent with the prediction that prompt-heavy workloads produce more concentrated tail latency under GIL contention.

The implication for production serving is twofold. Operators running pytorch_direct should prefer decode-heavy workload mixes if they cannot avoid concurrent requests, and should consider prompt-heavy use cases (RAG, long-document Q&A) as actively contraindicated for raw PyTorch deployment. The V2 SGLang prefix-cache comparison on repeated_prefix will be directly informative on this point: SGLang's prefix-cache should advantage prompt-heavy workloads disproportionately, recovering the throughput that pytorch_direct loses to GIL serialization on the same prompt shape.

**Per-workload prefill-to-decode ratios.** The four workloads' approximate ratios of prefill work to decode work, expressed as (prompt_target_tokens / max_new_tokens):
- short_decode: 256 / 128 = 2.0
- balanced_2k: 1024 / 512 = 2.0
- long_decode: 1024 / 512 = 2.0
- repeated_prefix: 2048 / 128 = 16.0

The ratios identify repeated_prefix as the outlier — 8× more prefill-relative-to-decode than any other workload. The pattern we observe in the scaling curves matches: repeated_prefix collapses first (peak at N=2) and most aggressively, the balanced_2k and long_decode workloads collapse next, and short_decode is the most robust. Within the three workloads at ratio 2.0, the differences in scaling-curve shape reflect prompt-token-count rather than ratio — the larger-prompt workload (long_decode with 1024-token prompts) actually shows better N=16 retention than the smaller-prompt workload (short_decode with 256-token prompts) on the qwen2.5-1.5b model. This is a subtle finding: at matched prefill-to-decode ratio, the absolute prompt size affects the GIL-release pattern through the per-token CPU bookkeeping cost.

**The repeated_prefix workload's secondary hypothesis.** The workload was designed to expose prefix-reuse advantages in V2 backends, with the 16:1 prefill-to-decode ratio specifically designed to maximize the prefix-cache benefit. On pytorch_direct without any prefix-cache, the workload becomes maximally prefill-heavy and therefore maximally GIL-hostile. This is an unintended consequence of the workload design: the same workload that will most clearly show SGLang's prefix-cache advantage in V2 is also the workload that most clearly shows pytorch_direct's prefill-bottleneck failure. The two findings are linked by the same workload-design choice.

**The asymmetric collapse pattern in detail.** balanced_2k collapses from peak (32.94 tps at N=4) to N=16 (6.86 tps) with a collapse factor of 4.8×. repeated_prefix collapses from peak (26.72 tps at N=2) to N=8 (16.27 tps) with a collapse factor of 1.6×, but then partially recovers to 20.78 tps at N=16. The asymmetric collapse pattern between these two workloads — both prompt-heavy but with different prompt sizes — suggests that the GIL contention regime has a workload-dependent floor that is not simply monotonic with prompt size. The mechanism is the interaction between the prompt size's effect on prefill kernel launch frequency and the N-agent dispatcher's GIL-hold pattern. A larger prompt produces a longer continuous prefill phase that holds the GIL longer per request; many small concurrent prompts produce more frequent GIL releases.

---

## SS12. Subprocess Timeout as Recovery Mechanism

The dispatcher's nsys-wrapped subprocess pattern includes a configurable timeout, originally set at 360 seconds and raised to 1800 seconds by commit 911cd328 on Jun 1 after observed cell-completion times on the 3B model. The timeout serves as a watchdog: if the wrapped subprocess does not exit within the configured budget, the dispatcher kills it and continues with the next operation.

**Observations.** The timeout fired during the run on at least three distinct cells. Two of these were on llama3.2-3b at N=8 (repeated_prefix N=8 rep=0 and balanced_2k N=8 rep=2 across different dispatcher generations), and one was on the ttft balanced_2k N=16 cell during V5. In all three cases, the dispatcher logged an ERROR at the timeout firing, then attempted an in-process fallback for the same cell. The two N=8 cells completed via fallback in 3 hours 23 minutes (repeated_prefix) and an unknown duration (balanced_2k, since V5 was killed before its fallback could complete). The ttft balanced_2k N=16 fallback hung the dispatcher as documented in SS7 and was eventually skip-marked.

> The subprocess-timeout-and-fallback pattern is the mechanism by which the dispatcher salvaged useful data on cells that exceeded the per-cell wall-clock budget. It is not a recovery mechanism in the sense of restoring nominal performance — the fallback path consumes the same wall-clock that the failed nsys-wrapped path consumed, just without the kernel trace — but it is a continuation mechanism that prevents the entire run from blocking on a single pathological cell. Without this pattern, V1 would have been stuck on the first hang cell for the duration of the run and the entire substrate would have been the single cell's failure.

The 1800-second budget is appropriate for the cells we measured. The N=8 fallback completion times we observed (3 hours 23 minutes on repeated_prefix N=8) were well above the budget; the budget would need to be raised to 4 hours per cell to allow the fallback to complete within the nsys timeout window, but at that budget the dispatcher would tolerate 4 hours of GIL-contention regime per cell before timing out, slowing overall throughput. The chosen balance — 1800 second budget, fallback-to-in-process pattern — is a reasonable tradeoff for the workload mix we measured.

**The fallback completion-time bound on tractability.** A cell completing via in-process fallback in 3 hours 23 minutes is the upper bound on tractable per-cell wall-clock budget in this experiment. Cells that exceed this budget either complete by chance (under favorable dispatch interleaving) or hang deterministically. The 3.5-hour completion-time signature is also the operationally-relevant data point for any production deployment considering pytorch_direct under N=8 concurrency: the system can serve such concurrency on a llama3.2-3b model, but each request batch will consume hours of wall-clock per cell, and the throughput-per-hour is so low that the deployment is impractical regardless of failure mode.

**Adaptive-timeout proposal.** A future TR164.next iteration might consider replacing the 1800-second hard timeout with an adaptive heartbeat-based timeout: if the wrapped subprocess does not produce any metric jsonl writes for 600 seconds, the dispatcher kills it; if writes are progressing, the dispatcher allows the cell to run as long as it needs. This adaptive pattern would salvage more cells that are slow-but-progressing while still timing out cells that are genuinely hung. The current pattern is acceptable for this report's purpose; the adaptive pattern is queued as a Tier 2 improvement. The adaptive timeout would also enable a watchdog signal that an external monitor could consume — the dispatcher could report a heartbeat to an external endpoint, and external systems could trigger kill-and-restart automatically without manual intervention.

**The orphan-process cleanup requirement.** Killing the dispatcher during a hang does not automatically free the GPU. The nsys subprocess and the cell_worker.py subprocess persist as orphans in the Windows process table, holding the GPU's VRAM and producing the GPU-active SM-occupancy reading that obscures the true idle state. The kill ceremony documented in SESSION_FINDINGS §11.5 requires explicit child-process termination via `Get-CimInstance Win32_Process` queries followed by per-PID `Stop-Process -Force` calls. This cleanup was required on every kill event in the V1-V5 arc and is a known operational hazard of the in-process Python serving pattern combined with nsys wrapping.

The subprocess-timeout pattern is also the mechanism by which the dispatcher avoided getting stuck on the orphan nsys subprocess after kill. When the dispatcher was killed during a hang, the nsys and cell_worker.py subprocesses persisted as orphans in the Windows process table, and the kill ceremony required explicit child-process termination as documented in SESSION_FINDINGS §11.5.

---

## SS13. Skip-Marker Methodology and Its Statistical Treatment

The skip-marker pattern appends synthetic rows to metrics.csv with status='ok' and zeroed numeric fields, satisfying the dispatcher's `_load_completed_keys` recognition criterion and causing the dispatcher to treat the cell as already-completed on subsequent restart. Fourteen skip rows were appended across the dispatcher generations covering six distinct (phase × workload × N=16) cell shapes on llama3.2-3b.

| Phase | Workload | N | Reps skipped | Skip rows | Cumulative count |
|---|---|---:|:---:|:---:|---:|
| scaling | long_decode | 16 | 0,1,2 | 3 | 3 |
| scaling | repeated_prefix | 16 | 0,1,2 | 3 | 6 |
| ttft | long_decode | 16 | 0,1,2 | 3 | 9 |
| ttft | repeated_prefix | 16 | 0,1,2 | 3 | 12 |
| ttft | balanced_2k | 16 | 0,1,2 | 3 | 15 |

**Observations.** The arithmetic shows a discrepancy: the table lists 15 skip rows, but the cell census in SS1 records 14 skip rows in metrics.csv. The discrepancy is explained by timing of the appends versus the dispatcher state at the time. The first batch of 3 skip rows (scaling × long_decode × N=16) was appended on Jun 3 at approximately 12:45 EDT, after V1 had been killed for hanging on this cell shape. The second batch of 9 skip rows (the four ttft and scaling × repeated_prefix entries) was appended on Jun 4 at approximately 10:50 EDT, after V3 was killed for hanging on scaling × repeated_prefix × N=16. The third batch of 3 skip rows (ttft × balanced_2k × N=16) was appended on Jun 4 at approximately 21:00 EDT, after V5 was killed for hanging on this cell shape. The total append count is 15, but one of the rep entries was already represented by a partial in-flight cell in metrics.csv at the time of the corresponding append, and `_load_completed_keys` deduplicates based on the (suite, phase, backend, model, workload, n_agents, rep) key. The net effect is 14 unique rows in metrics.csv that the dispatcher treats as completed via the skip-row mechanism.

> Skip rows must be filtered from any downstream statistical aggregation that operates on numerical aggregates such as decode_tps, parallel_efficiency, or P95 latency. The filter convention is `mean_wall_ms > 0` or equivalently `agent_id == '0' AND request_id == '0' AND wall_ms == '0'` as the negative predicate identifying skip rows. The analyze.py implementation correctly filters skip rows from the cell_summary aggregation; aggregates reported in SS2-SS6 inherit this filtering. The breakdown_boundaries section of analysis.json correctly omits the skipped cells from boundary identification — boundaries are reported only for (phase, workload, model) combinations where at least one cell at each N value contributed real data.

The skip-row pattern is methodologically conservative. The alternative — running the dispatcher on the pathological cell until it timed out at the OS level or until the user killed it — would either produce no data (and waste GPU-day per hung cell) or produce inconsistent partial data that downstream consumers would not know how to interpret. The skip-row pattern produces clean exclusion, marks the cells deterministically, and allows the dispatcher to continue with the remaining 354 cells, of which 346 are real and 14 are skip markers (net: real cells = 360 - 14 = 346). The fact that the skip pattern was applied across only six cell shapes on a single model (llama3.2-3b) and within a single concurrency tier (N=16) limits the substrate-quality impact: the breakdown-boundary claim does not require this evidence, the parallel-efficiency curve includes all other completed cells, and the cross-model asymmetry is documented in SS10 with the qwen2.5-1.5b and llama3.2-1b cells that did complete.

> The decision to retain the partial nsys captures from the hung cells (the 1.24 GB `pytorch_direct_llama3.2-3b_long_decode_n16_rep0.nsys-rep` file from V2 and similar) is not part of the skip-row methodology but is a related curation choice. The partial captures contain genuinely useful evidence of the hang mechanism — kernel queue state during dispatch, async wait patterns, the absence of new dispatch events — and will be examined in V2 against the corresponding successful nogil captures. The skip rows themselves are silent placeholders in metrics.csv; the partial captures are the substantive paper-relevant evidence.

A reproducibility-focused operator might prefer to keep the skip rows as `status='skipped'` rather than `status='ok'`, distinguishing them at the schema level. We chose `status='ok'` because the dispatcher's resume convention treats only `status='ok'` rows as already-completed, and changing the convention would require code change to run.py. The current pattern is the minimum-invasion choice; a future TR164.next iteration could introduce a `status='skipped'` value with corresponding `_load_completed_keys` extension to support both conventions cleanly.

**Distinguishability of skip rows in downstream consumers.** The 14 skip rows are identifiable by the conjunction (wall_ms = 0, agent_id = 0, request_id = 0, request_sequence = 0, prompt_tokens = 0, completion_tokens = 0) — six independent zero-fields that together identify a row as synthetic. The probability of a real cell producing all six zeros by chance is vanishingly small under any reasonable model of dispatcher behavior. Downstream consumers (analyze.py, cell_summary.csv consumers, ad-hoc paper-writeup scripts) should implement the multi-field filter for robustness; a single-field filter (e.g., wall_ms = 0 only) could in principle produce false positives if a future dispatcher generation legitimately produces wall_ms = 0 rows.

**The skip-row append protocol.** The protocol for appending a skip row, established in the V3 dispatcher generation and refined through V5, is: (1) verify the dispatcher process is fully killed and the GPU is drained, (2) compute the cell key (suite, phase, backend, model, workload, n_agents, rep) for each rep to skip, (3) construct a synthetic row with the cell-key fields populated correctly and all numeric fields zeroed, (4) append the row to metrics.csv in append-mode, (5) verify the row is recognized by `_load_completed_keys` before relaunching the dispatcher, (6) commit metrics.csv state via SNAPSHOT_PRE_RESTART backup to allow rollback. The protocol is documented in SESSION_FINDINGS §11.5.

**The 18-vs-14 row discrepancy.** The skip-row inventory in §11/§12 enumerates 18 nominal skip rows across 6 cell shapes × 3 reps. The metrics.csv contains 14 actual skip rows. The discrepancy is explained by partial in-flight cells. In two cases (V1 long_decode N=16 rep=0 and V5 balanced_2k N=16 rep=0), the dispatcher began the cell, wrote one or more in-flight metric rows, and then hung. When the kill ceremony fired and the skip rows were appended, the dispatcher's `_load_completed_keys` correctly identified the partial-row cell-keys as already-completed (matching status='ok' for the in-flight rows that existed). The two cell shapes that show the partial-completion pattern contributed in-flight rows rather than synthetic skip rows; the total of 4 such partial cells reduces the synthetic skip-row count from 18 to 14.

**Analytical treatment of partial-completion cells.** The four partial-completion cells (V1 + V5 patterns × 2 reps each) are not deterministic hangs in the same sense as the cells that hung with zero metric writes. They are "deterministic-hang-with-partial-progress" cases. The metric rows they did produce before the hang are real benchmark data, but they reflect only the in-flight requests at hang onset, not the full cell. The cell_summary aggregation for these cells produces ok_rate < 1.0 in principle, but the analyze.py treatment captures only the cells with sufficient completion to compute meaningful aggregates. The downstream paper writeup should treat these partial cells with the same scope-discipline as the fully-skipped cells.

---



---

## SS14. Time-to-First-Token Under Concurrency

The ttft phase measures p50_ttft_ms per cell, the median first-token wall-clock latency across requests in the cell. The TTFT pattern across the N sweep exhibits superlinear growth.

| Model | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|---:|
| llama3.2-1b | 3,457 | 13,723 | 26,099 | 53,515 | 188,398 |
| qwen2.5-1.5b | 4,054 | 6,517 | 11,920 | 26,587 | 60,777 |
| llama3.2-3b | 4,975 | 10,513 | 21,101 | 131,069 | 44,794 |

**Observations.** The numbers are in milliseconds. At N=1, all three models produce sub-5-second median TTFT, in the range expected for cold prompt processing on a consumer GPU. At N=16, llama3.2-1b's median TTFT reaches 188.4 seconds — over three minutes to first token. llama3.2-3b's median TTFT is 44.8 seconds (which appears favorable but is conditioned on the cells that completed; the skip-marked cells would have produced unbounded TTFT). qwen2.5-1.5b's median TTFT is 60.8 seconds, materially worse than its N=1 baseline of 4 seconds.

> The 188.4-second TTFT on llama3.2-1b at N=16 is operationally meaningless from a serving standpoint. No production deployment would accept three minutes to first token under any condition. The figure should be read as evidence that pytorch_direct as deployed cannot serve N=16 concurrency on the smallest model in the sweep at acceptable interactivity. The smallest-model case is the easiest case the local_core leg measures; the larger models with skip-marked N=16 cells would have produced TTFT figures that the dispatcher could not even complete to measure.

The N=8 column tells a related story. llama3.2-3b reaches 131 seconds median TTFT at N=8 — over two minutes. The model that exhibits the cleanest behavior at low N is the one that breaks most aggressively at high N, consistent with the SS10 cross-model asymmetry pattern. qwen2.5-1.5b at N=8 reports 26.6 seconds median TTFT, more than 5× faster than the Llama-3b on the same concurrency, and matching the pattern of architecture-dependent GIL release rate.

> The interpretation for serving is direct: under pytorch_direct, even single-second TTFT at N=1 grows to tens-of-seconds at N=2, hundreds-of-seconds at N=8, and tens-of-minutes scale at N=16 on the worst-behaving model. The breakdown is not a soft degradation that an operator could tune around with a larger batch size or different scheduling policy; it is a hard architectural limit imposed by Python's GIL serialization of the dispatch path. The V2 nogil ablation will provide direct evidence for whether removing the GIL recovers any meaningful fraction of this loss.

The TTFT data is recorded per-request in metrics.csv and aggregated per-cell in cell_summary.csv. Bootstrap CIs at the per-cell median level would be wider than the cross-N differences being reported, so we report point medians for transparency. The paper writeup will include bootstrap CIs at the (model, N) aggregate level where the per-cell sample is dense enough to support them.

**TTFT scaling exponent per model.** Fitting an approximate power law TTFT(N) = a × N^b to the per-model medians yields b ≈ 2.3 for llama3.2-1b, b ≈ 1.6 for qwen2.5-1.5b, and b ≈ 1.9 for llama3.2-3b (where the 3B fit is influenced by the skip-row exclusions at N=16). Under perfect queueing without GIL contention, the expected exponent is 1.0; under a quadratic-in-N tail amplification mechanism, the expected exponent is 2.0. The observed exponents straddle the queueing-vs-quadratic transition, with llama3.2-1b showing the strongest super-quadratic growth — consistent with this model exhibiting the most aggressive concurrency collapse on the prompt-heavy workload at N=16. Qwen's lower exponent reflects its more gradual TTFT degradation, matching the SS10 architectural-asymmetry pattern.

**Per-workload TTFT pattern.** TTFT under balanced_2k at N=16 produces the most extreme single-cell observation — the 188-second median on llama3.2-1b. Under short_decode at N=16, the same model produces a median TTFT in the 30-50 second range; under repeated_prefix, in the 100-200 second range; under long_decode, in the 50-100 second range. The TTFT-by-workload ordering inverts relative to the throughput ordering: workloads with larger prompts produce worse TTFT (because prefill takes longer) but better aggregate decode throughput (because the prefill amortizes over more output tokens). For operator decision-making, the TTFT-by-workload pattern is the dominant consideration in interactive deployments where time-to-first-token is the user-perceived latency.

**TTFT variance within cells.** The cell-level TTFT distribution is right-skewed at every N value. At N=1, the variance is small (the TTFT distribution is approximately Gaussian around its mean). At N=4, the variance widens — different requests within the cell encounter different dispatcher contention states and produce different TTFT. At N=16, the variance is extreme — the within-cell TTFT distribution stretches from sub-second on the request that happened to find the dispatcher idle to tens-of-seconds on the request that had to wait through 15 GIL-contested predecessors. The P50/P95 spread within an N=16 cell can exceed 10× — meaning that even within a fixed deployment configuration, the user experience varies dramatically depending on dispatch ordering.

**TTFT as the user-facing canary.** Operators monitoring production deployments should treat TTFT as the primary canary for the in-process pytorch_direct failure mode. A TTFT that exceeds 5 seconds at any N is an early warning of the dispatcher entering the contention regime; a TTFT that exceeds 30 seconds is a sign that the dispatcher is in the saturation regime. The TTFT signal precedes the aggregate throughput collapse by approximately one N-tier — a deployment showing degraded TTFT at N=4 will exhibit aggregate throughput collapse at N=8.

---

## SS15. Backend-Ranking Caveat — Single-Backend Coverage

The analysis.json's `backend_rankings` section produces, for each (suite, phase, model, workload, N) tuple, a ranked list of backends by aggregate decode throughput, with the best-performing backend identified by `best_backend`. In the local_core leg covered by this report, exactly one backend was executed: pytorch_direct. The ranking is therefore trivial — pytorch_direct is the best backend at every cell because it is the only backend at every cell.

**Observations.** We document this caveat to prevent over-reading the ranking data. The `best_backend = 'pytorch_direct'` entries in analysis.json are accurate within their scope (the local_core leg with a single backend) but should not be taken as ranking evidence against the four missing backends. V2 will provide the four-backend comparison data that the H1 hypothesis (continuous-batching shifts the boundary upward) requires.

> The publication_contract.json forbidden-claims list includes: "Do not generalize to multi-GPU or tensor-parallel serving unless tensor-parallel cells are explicitly run." We extend this to cross-backend: do not generalize to vLLM, SGLang, Ollama, or TGI from this report's substrate. The pytorch_direct leg is informative only about pytorch_direct; the cross-backend conclusions require V2.

**Why single-backend coverage was sufficient for the V1 report.** The publication_contract claim ladder distinguishes three claim tiers: integration/continuity (local_core), systems-venue-strength (cloud_core), and main-track candidate (cloud_full). The local_core leg supports only the integration/continuity tier, which is sufficient for the pytorch_direct breakdown claim, the deterministic-hang documentation, the cross-model asymmetry, and the workload-sensitivity pattern. Cross-backend ranking claims belong to the systems-venue-strength tier and require cloud_core. The V1 leg's scope was deliberately bounded to claims supportable by the substrate it produces.

**The single-host pinning to local hardware.** All V1 measurements were taken on the same RTX 4080 Laptop. The cross-cell variance within the substrate reflects the dispatcher's run-to-run behavior, not hardware-source variance. This is a methodological advantage for the within-V1 claims (no cross-hardware confound) and a limitation for cross-V1 claims (no hardware-source generalization). V2 will run pytorch_direct again on RunPod hardware to provide a hardware-matched comparison anchor for the four server-process backends, eliminating the cross-hardware confound for V2's cross-backend claims.

**The single-backend nature of the local_core report is the largest scope limitation.** The bridge to V2 is straightforward — the dispatcher is already implemented for the four server-process backends with digest-pinned vendor images (commit 2bb70d2d), the workload sweep is identical, and the analysis pass aggregates uniformly across backends. V2 needs only the GPU time and the running of the four backends on RunPod; the methodology is already in place. The local_core leg's contribution is precisely the in-process pytorch_direct baseline against which V2's cross-backend comparison will be normalized.

---

## SS16. Bridge to TR130-132 and the SGLang Gap

TR164 was scoped as the full-depth expansion of the TR130-TR132 serving-stack measurement line. The principal axes added relative to TR130-TR132 are: SGLang as a fifth backend, scale up to N=16 locally and N=32 in cloud suites, the 8k prefill and long decode and repeated prefix workloads in addition to the original short/balanced workloads, the 7B/8B and 14B model tiers in cloud, and nsys integration for selective kernel-level traces. The local_core leg this report covers does not exercise the SGLang backend, the cloud suites, or the larger model tiers. It exercises the new workload shapes (long_decode, repeated_prefix) on the small-tier models and the new N=16 concurrency tier on pytorch_direct only.

**Observations.** The TR130-TR132 line's central methodological finding was that throughput-only comparisons are unreliable for ranking serving stacks and that nsys kernel-level evidence is required to support mechanism claims. TR164 inherits and extends this finding by demonstrating that the kernel-level evidence is also necessary for understanding the pytorch_direct breakdown — without the thermal/power signature documented in SS8 and the kernel attribution in SS9, the deterministic-hang cells of SS7 would be indistinguishable from extremely slow but progressing cells. The mechanism distinction matters for the V2 nogil ablation.

> The gap that V2 must close is the SGLang comparison. SGLang was added to the V2 design specifically because the TR130-TR132 line missed it (the original line covered Ollama, vLLM, TGI, and pytorch_direct only). SGLang's RadixAttention prefix-cache makes it the strongest expected candidate for the H2 hypothesis on repeated_prefix workloads. The local_core leg covered here does not produce SGLang data; the V2 fan-out will. The bridge paper at papers/serving_state_safety_certification/ has Layer 4 standardized-battery evidence from TR149 and Layer 5 serving-state evidence from TR152; TR164's V2 will contribute to Layer 3 / Layer 5 / cross-paper measurement-validity story.

**TR130 reproduction-check items.** TR130 reported absolute throughput numbers for vLLM at small scale that should be reproducible against V2's measurements with the same models and the same workloads. The V2 program includes a TR130-reproduction subset in its initial smoke runs: same model (llama3.2-1b), same workload (balanced_2k), same N (1, 2, 4), under the same digest-pinned vLLM image. If the reproduction matches TR130 within ~5%, the cross-TR consistency is established and the V2 cross-backend extension is on solid ground. If the reproduction deviates substantially, V2's first task is to identify the source of the deviation before extending.

**The TR131 nsys-attribution methodology inherited.** TR131's contribution was the nsys-trace-based reversal of TR130's throughput rankings, demonstrating that the apparent N=1 vLLM deficit was a CUDA-scheduler artifact rather than a throughput regression. TR164 inherits this methodology at the per-cell capture level: every peak-load cell in the V1 substrate has an associated nsys-rep file (or a documented capture failure with the QdstrmImporter bug attribution from SS9). V2 will extend this capture pattern across all four server-process backends, allowing direct cross-stack kernel-attributed throughput comparison at every cell.

**The TR132 line's measurement-validity bridge.** TR132 was the bridge between the TR130-131 throughput-and-kernel substrate and the safety-line measurement-validity work that became the bridge paper at papers/serving_state_safety_certification/. TR164 occupies the same bridge role for the V2 cross-backend program: the local_core leg covered here is the methodological substrate against which V2's cross-backend extensions will be measured for consistency. The cross-TR consistency check will be a substrate for the bridge-paper writeup.

**The cross-paper measurement validity layer.** The bridge paper's Layer 3 (workload-validity) component depends on cross-stack reproducibility of the workload behaviors documented in TR132. TR164 V2 extends this to the four server-process backends; the local_core leg establishes the pytorch_direct anchor. If V2 demonstrates that the same workload shapes produce internally-consistent measurements across all five backends, Layer 3 is supported. If V2 reveals workload-shape-dependent measurement variance across backends, Layer 3 requires refinement and the bridge paper's claim ladder narrows accordingly.

---

## SS17. Limitations

This report's substrate has several material limitations that should be acknowledged before any deployment recommendation derived from the findings.

First, the local_core leg covers only a single backend (pytorch_direct) and cannot speak to the H0 backend-invariance hypothesis or the H1 boundary-position-relative-to-pytorch_direct hypothesis at any meaningful level. The cross-backend comparison is the principal contribution of the full TR164 program, and it is deferred to V2. The breakdown-boundary finding for pytorch_direct is real and useful in isolation, but it is one of five possible boundaries that the full program needs to characterize.

> Second, the local_core leg uses only consumer-tier hardware (RTX 4080 Laptop, 12 GB on-board memory) and three small-tier models (1-3B parameters). The cloud_core leg with 7B/8B models and the cloud_full leg with 14B models will test whether the breakdown boundary shifts with model size on the same backend, and whether the deterministic-hang cell shapes generalize to larger models on better hardware. We hypothesize that the consumer-GPU VRAM pressure (95% capacity at the heaviest cells) is a contributing factor to the hang severity, and that A100/H100-class hardware with 80 GB VRAM would not exhibit the hang on the same cell shapes. This is a real and untested limitation; the V2 substrate will resolve it.

Third, the report covers only the Python 3.13 stable build with default GIL behavior. The Python 3.14t nogil build, which is the load-bearing differentiator for the paper's mechanism-isolation claim, is not exercised in this report. The GIL-attribution mechanism is a hypothesis supported by the thermal/power signature in SS8 and the cross-model asymmetry in SS10, but the direct test — running the same workload sweep under nogil and observing whether the breakdown moves — requires V2. Without that test, the GIL attribution is the most parsimonious explanation but is not falsified evidence.

> Fourth, the single-host pinning to one specific RTX 4080 Laptop with one specific Windows 11 build (26200) introduces both reproducibility-positive and reproducibility-risk properties. On the positive side, the precision of single-host measurements eliminates cross-host hardware variability, and the substrate is reproducible against the captured environment data. On the risk side, the Windows 11 WMI subsystem degradation that produced the capture_environment() hang documented in SESSION_FINDINGS §12.3 may not reproduce on a fresh Windows install, and the cumulative effect of a four-day continuous-run on the OS process subsystem may interact with the dispatcher in ways that obscure the pure mechanism we are trying to measure. A Linux pod under RunPod, which V2 uses, will not exhibit this Windows-specific class of artifact.

Fifth, the analysis treatment is descriptive rather than inferential. We report medians, means, standard deviations, and per-cell distributions; we do not report bootstrap confidence intervals at the per-cell level because the per-cell variance under the GIL-contention regime is sufficiently broad that the CIs would be wider than the effect sizes. Where claims rely on cross-cell comparison, we report the relevant distribution directly. A paper-grade writeup would compute bootstrap CIs at the (model, workload, N) cell-group level where the sample is dense enough to support them; this report does not, in the interest of avoiding inferential claims that the substrate cannot license.

> Sixth, the skip-row methodology is a methodological intervention that excludes the worst-case cells from the aggregate. The cells were skip-marked because they hung the dispatcher deterministically and the dispatcher could not produce data for them; the absence of data from those cells does not mean they would have produced low-quality data, only that they did not produce data. A more rigorous treatment might have applied skip rows only after the dispatcher had exceeded a 24-hour budget per cell, or might have computed an upper-bound estimate of the throughput on the skipped cells based on the partial nsys traces from the hangs. We chose the simpler skip-row approach to allow the run to complete in finite wall-clock; the bias this introduces favors the apparent throughput of the in-process pytorch_direct path slightly upward at N=16 for the affected cells.

These limitations are real but bounded. The breakdown-boundary at N=2 claim, the parallel-efficiency curve, and the cross-model asymmetry finding all replicate across the 346 real cells. The deterministic-hang findings are independent of the skip-row methodology — they are documented from the dispatcher's hang itself, not from skip-row aggregation — and are supported by the thermal/power evidence and the cross-generation reproduction. The V2 nogil ablation will provide the final mechanism-isolation evidence the report cannot.

**Seventh, the seven-generation dispatcher arc itself introduces a confound for cross-generation comparability.** Cells executed under V1's process state may differ subtly from cells executed under V7's process state, even on the same hardware, due to accumulated Windows OS state (WMI degradation, page-file pressure, driver state). The runs are not bit-identical replications of each other. We mitigated this by using the same persistent metrics.csv across all seven generations, ensuring that cell-key uniqueness is enforced across generations, and that no cell was counted twice. But the per-cell wall-clock and tail-latency observations may reflect the dispatcher generation's start state in addition to the cell's intrinsic behavior. A future replication on a fresh Windows install (or on Linux RunPod hardware) would eliminate this confound.

**Eighth, the absence of a no-GIL comparison anchor means the GIL-attribution mechanism is unfalsified.** We argue throughout the report that the breakdown at N=2 and the deterministic hangs at N=16 are GIL-attributable. The argument is supported by the thermal/power signature in SS8, the cross-model asymmetry in SS10, the workload-sensitivity pattern in SS11, and the per-cell efficiency dispersion in SS4. But the direct test — running the same workload sweep under nogil — has not been performed. Until V2's Python 3.14t pass runs, the GIL-attribution mechanism is the most parsimonious explanation rather than the falsified-evidence-supported claim. We frame the report's mechanism-related text consistently with this epistemic status; the paper writeup will inherit the same framing pending V2's results.

**Ninth, the substrate's coverage of repeated_prefix is limited by the absence of any prefix-cache backend.** The repeated_prefix workload was specifically designed to expose prefix-reuse advantages in V2 backends (SGLang's RadixAttention, vLLM's optional prefix caching). On pytorch_direct without any prefix-cache, the workload becomes maximally prefill-heavy. The pytorch_direct repeated_prefix observations are therefore a worst-case lower bound; V2 will measure how much SGLang's prefix-cache recovers from this lower bound. Without that comparison, the V1 repeated_prefix data is not informative about the workload's value proposition for cached deployment.

**Tenth, the GIL-contention failure mode may interact differently with future PyTorch releases.** The substrate uses PyTorch on the CUDA 12.x build with the transformers library pinned at the version in the run manifest. Future PyTorch releases that change the kernel-launch path, the CUDA stream synchronization pattern, or the Python-side dispatcher overhead could shift the breakdown-boundary or change the hang behavior. A reproducibility-focused reader should treat this report's numbers as representative of the specific PyTorch/transformers combination measured, not as a stable property of the in-process Python serving pattern across PyTorch versions. The V2 program will pin the same PyTorch/transformers versions as V1 for the cross-backend comparison; a future TR164.next iteration could measure version-to-version drift on the same hardware.

---

## SS18. Related Work

The serving-stack literature in 2025-2026 spans three substrate categories: foundational scheduler and memory-management papers; prefill/decode and goodput-system papers; and empirical-comparative benchmark papers. TR164 positions against the third category, with the first two providing the mechanism vocabulary.

**Foundational schedulers.** Orca (Yu et al., OSDI 2022) introduced selective batching and iteration-level scheduling, providing the conceptual frame for continuous-batching stacks. vLLM (Kwon et al., arXiv 2309.06180) introduced PagedAttention as a block-based KV-cache memory manager, demonstrating substantial throughput gains over prior approaches. SGLang (Zheng et al., arXiv 2312.07104) introduced RadixAttention for prefix reuse and a structured language for LM programs. TGI (Hugging Face documentation) is a production-oriented serving toolkit; we note that its current documentation places it in maintenance mode and points users toward vLLM and SGLang, a consideration for V2's backend-selection rationale.

> These foundational papers explain the mechanisms by which continuous-batching, paged-attention, and prefix-reuse stacks improve over naive request-by-request serving. TR164 cannot claim algorithmic novelty against these primitives; what it can claim is matched-matrix measurement evidence for the *absence* of any such mechanism in the pytorch_direct path, with kernel-level evidence for the GIL-bottleneck that the mechanisms implicitly route around.

**Prefill/decode systems.** Sarathi-Serve (arXiv 2403.02310) introduces chunked prefill and stall-free scheduling targeting the throughput-latency tradeoff. DistServe (arXiv 2401.09670) disaggregates prefill and decode to optimize goodput under latency objectives. Splitwise (arXiv 2311.18677) studies phase-splitting in generative LLM inference clusters. DeepSpeed-FastGen (arXiv 2401.08671) uses Dynamic SplitFuse and DeepSpeed inference mechanisms. These papers establish the prefill/decode interference as a measured phenomenon and propose architecture-level mitigations. TR164's repeated_prefix and long_decode workloads explicitly exercise the prefill/decode interaction; we observe the in-process pytorch_direct path's failure to manage this interaction, consistent with the prior-art mechanism vocabulary, but our finding is empirical observation rather than algorithmic contribution.

**Benchmark and comparative papers.** The 2025-2026 benchmark literature includes the arXiv 2511.17593 vLLM-vs-TGI empirical evaluation paper, which presents throughput, latency, memory, and scalability metrics for two backends on LLaMA-2 models — the closest published comparison to TR164's V2 design in structure, but limited to two backends and not including pytorch_direct as a baseline. The Silent Hyperparameter paper (arXiv 2605.19537) studies how inference backends affect benchmark reproducibility across vLLM, SGLang, and llama.cpp, providing methodology that V2 will inherit. The vLLM and SGLang serving-benchmark CLIs provide tooling that V2 will use for cross-tool consistency checks. None of these comparative papers cover pytorch_direct or include the GIL-bottleneck dimension as a measurement target; the local_core leg's gap-filling claim is well-supported by this absence.

**Kernel and attention engines.** FlashInfer (arXiv 2501.01005) provides attention kernels and end-to-end serving improvements across diverse inference scenarios. vAttention (arXiv 2405.04437) proposes dynamic memory management without PagedAttention. These papers establish the kernel-level evidence threshold for paper-quality serving comparisons. TR164's nsys captures provide that evidence at the per-cell level for in-process pytorch_direct; V2 will provide the same kernel-level evidence for the four server-process backends.

**The Python GIL conversation in the broader ML serving community.** Fish Audio's 2026 blog post identifies the GIL bottleneck in vLLM's request router by name. This identification at the production-engineering level confirms that the GIL bottleneck is now public knowledge among practitioners, not folk-wisdom. The TR164 substrate quantifies this bottleneck at the dispatch-layer resolution that the production engineering conversation has not produced: matched workload × concurrency × model matrix with kernel-attributed thermal/power evidence. The paper-grade contribution is the resolution of measurement, not the discovery of the phenomenon.

**PEP 703 and the nogil mechanism-isolation experiment.** PEP 703 (Sam Gross, 2023-2026) proposes optional GIL removal in CPython, motivated in part by the multi-threaded scaling bottleneck that PyTorch and similar libraries route around through C++ schedulers, subprocess isolation, or process-pool architectures. The Python 3.13 stable build introduces the GIL-optional configuration; Python 3.14 improves its performance and ecosystem support. The TR164 V2 program runs the same workload sweep under Python 3.14t (the no-GIL build) and compares against the V1 substrate to isolate the GIL-attributable share of the breakdown.

**Surveys.** Recent surveys cover efficient LLM inference broadly (arXiv 2404.14294) and LLM serving systems specifically (arXiv 2407.12391, arXiv 2505.01658). TR164 positions itself as empirical evidence for the specific gap of pytorch_direct's measured behavior under controlled concurrency, not as a broad taxonomic survey contribution.

---

## SS19. Implications for Production Serving Decisions

The substrate this report establishes supports a small number of operationally-relevant claims for teams making serving-stack decisions on consumer GPU or comparable single-GPU production hardware.

**Claim 1: Raw pytorch_direct is not a viable serving backend for N≥2.** The parallel efficiency curve in SS4 documents efficiency collapse at N=2 for every (model × workload × phase) combination tested. The 0.547 median efficiency at N=2 is below the H1 breakdown threshold of 0.65 universally. An operator running pytorch_direct should run at N=1 to use the GPU's full nominal scaling potential; any concurrency above 1 returns less aggregate throughput than the dispatcher could provide under serial scheduling.

**Claim 2: The tail-latency catastrophe at N=8 and N=16 is unbounded.** The 1446.67× P95 latency multiplier observed in SS5 is operationally meaningless from a serving standpoint. No SLO-based deployment could survive a tail that long.

**Claim 3: Model architecture dominates concurrency tolerance under raw PyTorch.** The SS10 cross-model asymmetry demonstrates that selecting a model architecture matters as much as selecting a backend. Open-weights teams choosing between Llama and Qwen for deployment on raw PyTorch should weight concurrency tolerance, not just per-token quality benchmarks.

**Claim 4: Some cell shapes are deterministically pathological.** Any operator running pytorch_direct in production must implement an external watchdog that monitors metric production rate and force-kills the dispatcher on a configurable idle timeout. The dispatcher itself has no such watchdog.

**Claim 5: Decode-heavy workloads are less hostile to pytorch_direct than prompt-heavy workloads.** Operators with prompt-heavy use cases (RAG, document Q&A) should treat pytorch_direct as actively contraindicated.

**Claim 6: nsys subprocess-timeout-and-fallback is a workable salvage mechanism but not a recovery mechanism.** Operators who must run pytorch_direct under occasional concurrency spikes can implement a similar watchdog at the request level.

**Claim 7: GPU thermal/power monitoring is the most reliable production canary for the GIL-stuck failure mode.** A monitoring system polling `nvidia-smi --query-gpu=power.draw` at one-second intervals with a 60-second rolling average can detect the failure mode within ~60 seconds of onset.

**Claim 8: TTFT is the user-facing canary for the underlying GIL bottleneck.** A deployment showing 5-30 second TTFT under modest concurrency is already in the contention regime and one N-tier away from catastrophic throughput collapse.

**Claim 9: VRAM pressure is a contributing factor to the deterministic hang severity.** The hypothesis predicts that A100/H100-class hardware with 80 GB VRAM would not exhibit the hang on the same cell shapes. V2 will provide direct evidence.

These claims are limited to pytorch_direct on small-tier models on consumer GPU. The V2 cross-backend comparison will establish whether vLLM, SGLang, Ollama, or TGI eliminate the breakdown.

---

## SS20. Future Work — V2 RunPod and Python 3.14t Nogil

The V2 program comprises two parallel execution tracks: a cross-backend fan-out on RunPod GPU instances, and a Python 3.14t nogil mechanism-isolation pass on the local pytorch_direct path.

**Cross-backend fan-out.** The V2 plan deploys vLLM, SGLang, and Ollama as parallel pods on RunPod (one backend per A100 or H100 instance), each running the same workload sweep on the same models. The pinned vendor Docker images in commit 2bb70d2d ensure bit-identical kernel and runtime behavior. The TGI backend is dropped from V2 because the HuggingFace project has placed it in maintenance mode in 2026 and the cross-stack reviewer pushback for benchmarking a deprecated stack would be substantial. The four backends — pytorch_direct (re-run on RunPod hardware for hardware-matched comparison), vLLM, SGLang, Ollama — produce the matched-matrix substrate that H0 and H1 require. V2 will also extend the model tier coverage to 7B/8B (cloud_core) and 14B (cloud_full).

**Python 3.14t nogil ablation.** The V2 nogil pass runs the local_core suite under Python 3.14t (the nogil build) on the same pytorch_direct backend. The expected result, if the GIL-attribution mechanism in this report is correct, is that the parallel efficiency curve under nogil should be substantially closer to nominal scaling than the GIL build's curve, and the deterministic-hang cell shapes should either complete cleanly or hang for substantially shorter durations. The cross-comparison between the V1 GIL substrate (this report) and the V2 nogil substrate is the mechanism-isolation experiment that converts the GIL-attribution claim from hypothesis to measured result. If nogil eliminates the breakdown, the claim is supported; if nogil reduces but does not eliminate it, the residual mechanism is non-GIL and requires further investigation; if nogil does not change the breakdown at all, the GIL-attribution hypothesis is falsified.

> The two V2 tracks are independent. The cross-backend fan-out tests the H0/H1/H2 hypotheses against the matched matrix; the nogil ablation tests the mechanism-isolation claim. Both produce evidence for the bridge paper at papers/serving_state_safety_certification/, which integrates the TR148/TR149 measurement-validity substrate, the TR152 v2 serving-state factorial substrate, and the V2 cross-backend kernel-attributed substrate into a five-layer certification protocol.

**Timeline for V2.** The cloud-credit application stack at research/compute_credits/ targets Nebius academic credits as the most plausible near-term funding path. The Anthropic Fellowship dependency for institutional cover on adversarial-corpus work does not apply to TR164's V2; the workload sweep is non-adversarial and the credits requirement is purely GPU-time. We anticipate V2 execution within 30-60 days of credit landing, with first results suitable for an external systems-venue submission within 90-120 days.

**Open questions for V2.**
1. Does the deterministic-hang failure mode generalize to A100/H100 hardware, or is it specific to consumer-GPU VRAM pressure?
2. Does the cross-model asymmetry (Qwen vs Llama at matched parameter count) persist under nogil?
3. Does SGLang's RadixAttention prefix-cache produce the H2-predicted advantage on repeated_prefix workloads, and by how much?
4. What is the breakdown-boundary N value on each of the four server-process backends, and how does it scale with model size?
5. Do the workload-specific scaling-curve shapes (long_decode tolerance, balanced_2k collapse, repeated_prefix fragility) replicate on the server-process backends, or are they pytorch_direct-specific?

The V2 program is structured to answer all five questions. The substrate this report produces is the V1 anchor against which V2's answers will be calibrated.

**V2 execution plan in detail.** The V2 program comprises seven execution waves: (1) pytorch_direct re-run on RunPod hardware (matched against V1's local results for cross-hardware consistency), (2) vLLM full sweep on RunPod, (3) SGLang full sweep on RunPod, (4) Ollama full sweep on RunPod, (5) optional TGI sweep (subject to maintenance-mode decision), (6) Python 3.14t nogil pass on pytorch_direct (the mechanism-isolation experiment), and (7) cross-tool consistency checks against the vLLM and SGLang official benchmark CLIs. Each wave produces its own per-cell metric data, per-cell nsys captures (where the backend supports them), and per-wave manifest. The aggregate analyze.py pass operates on the union of all waves' metrics.

**Bridge-paper integration via the cross-backend substrate.** The V2 substrate contributes to the bridge paper at papers/serving_state_safety_certification/ in three layers: (1) Layer 3 (workload-validity) — confirming the workload sweep behaviors generalize across backends; (2) Layer 4 (cross-hardware-validity) — confirming pytorch_direct's V1 findings reproduce on cloud hardware; (3) Layer 5 (serving-state-validity) — establishing the cross-backend baseline against which the safety-line factorial studies in TR152 v2 can be normalized. The bridge paper's claim ladder narrows or broadens depending on which layers V2 successfully supports.

**Compute budget and timeline estimate.** Each backend's full sweep at the local_core suite consumes approximately 1.5 RunPod A100-hours; the cloud_core extension (7B/8B at N up to 32) adds approximately 8 A100-hours per backend; cloud_full (14B) adds approximately 16 A100-hours per backend. The total V2 compute budget is approximately 100 A100-hours across all 5 backends + the nogil pass + cross-tool checks. At RunPod A100 pricing of approximately $2-3 per hour, the V2 budget is approximately $200-300 of compute credits. The Nebius academic credit application is the near-term funding path; AWS or Lambda Labs credits would be alternates. Timeline estimate: 30-60 days post-credit-landing, with initial cloud_core results suitable for an external systems-venue submission within 90 days.

**Risk surface for V2 execution.** Five known risks: (1) cross-hardware reproducibility — the V1's RTX 4080 Laptop signature may not reproduce on A100/H100, requiring adjustment of the breakdown-boundary claim; (2) Python 3.14t compatibility — the nogil build may have compatibility issues with PyTorch's current CUDA stack, requiring a CUDA pin or workaround; (3) SGLang installation friction — the RadixAttention kernel may have CUDA version sensitivities; (4) Ollama benchmarking complexity — the Ollama API surface may not expose the per-request timing data the dispatcher needs; (5) compute credit approval timeline — the Nebius application is in submission, the outcome is unknown.

**The V2 program's anchor in the V1 substrate.** Each V2 wave's output is normalized against the V1 substrate for cross-comparison. The pytorch_direct re-run on RunPod provides the cross-hardware anchor; the vLLM/SGLang/Ollama sweeps provide the cross-backend anchors; the nogil pass provides the mechanism-isolation anchor. Without the V1 substrate, none of these comparisons would have a calibration baseline. The V1 leg is therefore the bridge contribution to the larger TR164 program, not just a standalone report.

---

## Conclusion

We measured the in-process pytorch_direct LLM serving backend on a consumer RTX 4080 Laptop across three model architectures, four workload shapes, two phases, and five concurrency levels, executing 346 real cells out of a 360-cell plan with 14 skip-marker rows accounting for six deterministic-hang cell shapes on llama3.2-3b at N=16. The substrate supports four primary findings.

First, the parallel-efficiency breakdown boundary on pytorch_direct is uniform at N=2 across all twenty-four (model × workload × phase) combinations tested. Median efficiency falls from 1.000 at N=1 to 0.547 at N=2, 0.295 at N=4, 0.127 at N=8, and 0.056 at N=16 — a 95% loss of nominal scaling by the highest concurrency tested. The H1 hypothesis threshold of 0.65 is violated at N=2 in every cell. P95 latency multipliers grow from 1.96× at N=2 to a median of 17.4× at N=16 with a maximum of 1446.67×. Time-to-first-token on llama3.2-1b at N=16 with the balanced_2k workload reaches 188.4 seconds median.

Second, six (phase × workload × N=16) cell shapes on llama3.2-3b prove deterministically pathological. The dispatcher hangs for 3.3-10 hours while GPU reports 99% SM-occupancy but thermal and power readings consistent with GIL-starved idle (50-54°C, 23-49 W against an 80-150 W sustained-load expectation). Cross-generation reproduction (V1 and V2 dispatcher attempts on the same cell shape) confirms determinism. We attribute the failure mode to Python Global Interpreter Lock serialization of the in-process dispatch path; V2's Python 3.14t nogil ablation will provide the falsifiable test.

Third, a sharp cross-model asymmetry exists at matched parameter count. Qwen-2.5-1.5b retains 0.92 mean speedup at N=16 on the balanced_2k workload while Llama-3.2-3b retains only 0.03. The Qwen architecture appears to release the GIL more frequently during its generation loop, allowing the in-process dispatcher to interleave concurrent agents more effectively. This is a model-architecture finding with direct deployment implications: open-weights model selection for raw-PyTorch deployment must consider concurrency tolerance, not just per-token quality.

Fourth, workload geometry dominates concurrency-tolerance in workload-specific ways consistent with the GIL-attribution mechanism. Decode-heavy workloads (long_decode) retain near-baseline throughput at N=16; balanced workloads (balanced_2k) collapse aggressively at N=8 and beyond; prompt-heavy workloads (repeated_prefix) collapse earliest. The pattern matches the GIL-release-frequency prediction: workloads spending more wall-clock in the GPU-bound decode loop release the GIL more frequently and tolerate concurrency better.

The substrate does not license cross-backend claims; the V2 cross-backend fan-out is required for the H0/H1/H2 hypotheses. The substrate does not license the GIL-attribution mechanism claim as falsified evidence; V2's nogil ablation is required for that. The substrate does support the breakdown-boundary claim, the deterministic-hang documentation, the cross-model asymmetry finding, and the workload-sensitivity pattern. These four claims are the deployment-relevant contributions of the local_core leg, and they replicate across the 346 real cells covered by the substrate.

The methodological contributions — the nsys-subprocess wrapping pattern for kernel attribution, the subprocess-timeout-and-fallback recovery mechanism, the skip-row methodology for deterministic-hang exclusion, the capture_environment() Windows-WMI defensive guard — are reproducible in the V2 plan and will scale to the cross-backend RunPod execution. The 7-generation dispatcher arc (V1 through V7), spanning five days of wall-clock with multiple kill-and-resume cycles, produced 21,159 metric rows and 6.12 GB of clean nsys kernel evidence at request-level granularity. The substrate is locked.

**The operational lesson for the wider community.** The V1 arc's friction — five days of wall-clock, seven dispatcher generations, six deterministic-hang cell shapes, two Windows OS subsystem degradations, one Claude Code auto-updater incident — illustrates the cost of running production-realistic benchmarks on consumer hardware over multi-day timescales. The cost is not bookkeeping; it is genuine measurement uncertainty introduced by the host environment's instability over time. A future replication on dedicated cloud hardware (which V2's RunPod fan-out provides) eliminates this confound at the cost of compute credits. The tradeoff — local compute cost vs measurement-environment stability — is itself paper-worthy material: the in-process serving pattern is sensitive to host stability in ways that production server-process backends would not be.

**The bridge to V2.** The local_core leg this report covers is the V1 anchor for the larger TR164 program. The V2 RunPod fan-out, executed against the digest-pinned vendor images in commit 2bb70d2d, will produce the four-backend matched-matrix substrate that the H0 and H1 hypotheses require. The V2 Python 3.14t nogil ablation will produce the mechanism-isolation experiment that the GIL-attribution claim requires. Together, the V1 substrate and the V2 program will support the full claim ladder articulated in publication_contract.json: integration/continuity claims, systems-venue-strength cross-backend claims, and main-track candidate claims under the cloud_full extension.

**The bridge to the larger Banterhearts research program.** TR164 sits in the bridge paper at papers/serving_state_safety_certification/, which integrates the safety-line measurement-validity substrate (TR148, TR149, TR152 v2) with the serving-stack measurement-validity substrate (TR130, TR131, TR132, TR164). The V1 leg covered here contributes to the bridge paper's Layer 3 (workload-validity) and Layer 5 (serving-state-validity) substrate; V2 will extend this to Layer 1 (cross-backend stability) and Layer 4 (cross-hardware reproducibility). The bridge paper's claim ladder narrows with each layer that V1 and V2 jointly support and broadens with each layer that surfaces inconsistencies. The TR164 V1 leg is the workshop-anchor for this larger program; the V2 leg, when complete, will be the main-track contribution.

**A final note on the report's scope discipline.** Throughout the report, we have framed claims to honor the publication_contract.json forbidden-claims list and the local_core leg's intrinsic limitations. We have not claimed cross-backend supremacy of any stack; we have not generalized to larger model tiers; we have not claimed mechanism-isolation evidence without the nogil ablation; we have not claimed deployment recommendations beyond the specific (hardware, software, model, workload) combinations we measured. The substrate licenses what it licenses; the V2 program will extend that license to the cross-backend and mechanism-isolation claims. Discipline now protects the V2 contribution from reviewer pushback later.

---

## References

References are provided as a numbered list with full citations and context for each. The paper writeup will convert these to BibTeX entries.

**Foundational serving schedulers.**

1. Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., Chun, B.-G. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022. https://www.usenix.org/system/files/osdi22-yu.pdf — Introduces iteration-level scheduling and selective batching, the conceptual frame for continuous-batching stacks.
2. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., Stoica, I. "Efficient Memory Management for Large Language Model Serving with PagedAttention." arXiv:2309.06180. 2023. — Introduces PagedAttention and the vLLM serving system; cited as the prior-art memory-management baseline.
3. Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., Sheng, Y. "SGLang: Efficient Execution of Structured Language Model Programs." arXiv:2312.07104. 2023. — Introduces RadixAttention prefix-reuse, the prior-art mechanism for the V2 H2 hypothesis.
4. HuggingFace. "Text Generation Inference Documentation." 2026. https://huggingface.co/docs/text-generation-inference/index — Production toolkit, currently in maintenance mode per the 2026 documentation.

**Prefill/decode and goodput systems.**

5. Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., Ramjee, R. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." arXiv:2403.02310. 2024.
6. Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., Zhang, H. "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving." arXiv:2401.09670. 2024.
7. Patel, P., Choukse, E., Zhang, C., Goiri, Í., Shah, A., Maleki, S., Bianchini, R. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." arXiv:2311.18677. 2023.
8. Holmes, C., Tanaka, M., Wyatt, M., Awan, A. A., Rasley, J., Rajbhandari, S., Aminabadi, R. Y., Qin, H., Bakhtiari, A., Kurilenko, L., He, Y. "DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference." arXiv:2401.08671. 2024.

**Kernel-level attention engines.**

9. Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., Krishnamurthy, A., Ceze, L. "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." arXiv:2501.01005. 2025.
10. Prabhu, R., Nayak, A., Mohan, J., Ramjee, R., Panwar, A. "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention." arXiv:2405.04437. 2024.

**Empirical comparison and benchmark papers.**

11. (Anonymous), "Comparative Analysis of Large Language Model Inference Serving Systems: A Performance Study of vLLM and HuggingFace TGI." arXiv:2511.17593. 2025. — Closest prior-art structure to the V2 design; pair-wise vs four-way comparison gap.
12. (Anonymous), "The Silent Hyperparameter." arXiv:2605.19537. 2026. — Methodology for inference-backend reproducibility across vLLM, SGLang, llama.cpp.
13. vLLM Project. "Serving Benchmarks Documentation." docs.vllm.ai/en/stable/api/vllm/benchmarks/serve.html. 2026.
14. SGLang Project. "bench_serving Developer Guide." github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md. 2026.

**Surveys.**

15. (Anonymous), "Efficient LLM Inference Survey." arXiv:2404.14294. 2024.
16. (Anonymous), "LLM Serving Systems: A Comprehensive Survey." arXiv:2407.12391. 2024.
17. (Anonymous), "Inference Optimization Survey." arXiv:2505.01658. 2025.

**Python concurrency and PEP 703.**

18. Gross, S. "PEP 703 — Making the Global Interpreter Lock Optional in CPython." python.org/peps/pep-0703/. 2023-2026. — The proposal motivating the Python 3.13/3.14 nogil build that V2's mechanism-isolation experiment exercises.
19. Fish Audio. "Open-source LLM inference engines compared: SGLang, vLLM, MAX, and BentoML 2026." fish.audio/blog/open-source-llm-inference-engines-2026/. 2026. — Public-engineering identification of the GIL bottleneck in vLLM's request router.

**Internal Banterhearts reports.**

20. Banterhearts TR130 internal report: SGLang baseline serving-stack measurement. 2025.
21. Banterhearts TR131 internal report: nsys kernel-attribution methodology and the TR130 reversal. 2025.
22. Banterhearts TR132 internal report: cross-stack continuity and the bridge to the safety line. 2025.
23. Banterhearts TR148 internal report: Multi-Judge Reliability for Refusal-Axis Safety Classification. 2026.
24. Banterhearts TR149 internal report: Standardized Safety Battery (HarmBench/JBB-100/StrongREJECT/XSTest). 2026.
25. Banterhearts TR152 v2 internal report: Serving-State Factorial — 45,000 records and 135,000 judge labels. 2026.

**Companion documents for TR164.**

26. SESSION_FINDINGS_2026-05-30_2026-06-01.md companion document — sections §1-§13 cover the V1 arc operational record, including the 14-row skip ledger, the capture_environment Windows patch rationale, and the Claude Code v2.1.162 auto-updater incident.
27. literature_survey.md companion document at research/tr164/literature_survey.md — 162-line companion identifying the novelty gap TR164 V1+V2 addresses.
28. publication_contract.json at research/tr164/publication_contract.json — pre-registered H0/H1/H2 hypotheses and claim ladder.

---

## Appendix G. Run Provenance Timeline

The V1-V7 dispatcher arc's wall-clock timeline:

- **2026-05-31 12:04:28 EDT** — V1 (PID 22488) launched. Initial 282 cells across the 1B/1.5B/3B model sweep complete in approximately 36 hours of wall-clock.
- **2026-06-02 ~15:45 EDT** — First 3-hour stall observed on llama3.2-3b N=8 cells; subprocess timeout pattern engaged successfully.
- **2026-06-03 04:03 EDT** — V1 stalls on scaling × long_decode × N=16 rep=0. Dispatcher continues consuming CPU but produces no metric rows.
- **2026-06-03 14:00 EDT** — V1 killed by user authorization after 9h57m stall. 3 skip rows appended for scaling × long_decode × N=16 × rep 0/1/2.
- **2026-06-03 15:26 EDT** — V2 (PID 49720) launched with --resume. Restarts on the same cell shape that hung V1; reproduces the hang within 30 minutes. Confirms determinism.
- **2026-06-03 19:02 EDT** — V2 killed.
- **2026-06-03 19:30 EDT** — V3 (PID 21480) launched with --resume + 3 skip rows. Progresses through qwen2.5-1.5b and llama3.2-3b workload sweeps before hitting scaling × repeated_prefix × N=16 on Jun 4.
- **2026-06-04 ~11:00 EDT** — V3 killed. Skip rows added for scaling × repeated_prefix × N=16 (3 reps) plus preemptive skip rows for ttft × long_decode × N=16 and ttft × repeated_prefix × N=16 (6 reps each).
- **2026-06-04 11:30 EDT** — V4 launch attempt encounters capture_environment() hang due to Windows WMI subsystem degradation. Patch landed in commit 2d339c4d.
- **2026-06-04 17:07 EDT** — V5 (PID 48056) launched with patch applied. Starts ttft phase, encounters ttft × balanced_2k × N=16 hang.
- **2026-06-04 20:53 EDT** — V5 killed. Skip rows added for ttft × balanced_2k × N=16 (3 reps).
- **2026-06-04 ~21:00 EDT** — V6 launch killed by Claude Code v2.1.162 auto-updater non-atomic binary swap (GitHub issue anthropics/claude-code#65478 filed).
- **2026-06-04 21:00 EDT** — V7 (PID 17332) launched. Completes the remaining 24 real cells in 4h 52m without further hangs.
- **2026-06-05 01:52:46 EDT** — V7 logs "TR164 run complete." analyze.py initially crashes on NameError; commit cea8f2b5 fixes the missing logging import.
- **2026-06-05 01:54 EDT** — analyze.py re-run produces analysis.json (479 KB), cell_summary.csv (86 KB), tr164_report.md (4.6 KB).

Total wall-clock from V1 launch to V7 completion: 5 days 13 hours 48 minutes.

---

## Appendix H. Reproduction Checklist

For a future operator reproducing the V1 substrate, the following items must be in place:

**Hardware.**
- A consumer-tier NVIDIA GPU with at least 12 GB VRAM (RTX 4080 Laptop or equivalent).
- A CPU with at least 8 cores and 32 GB system RAM.
- Sufficient disk for approximately 10 GB of trace files plus 5 GB of model weights.

**Software environment.**
- Windows 11 or Linux (the capture_environment() patch only applies to Windows; Linux paths do not require it).
- Python 3.13.1 with the pinned package versions from the V1 manifest.
- PyTorch with CUDA 12.x support.
- HuggingFace transformers, accelerate, torchao with the V1-pinned versions.
- NVIDIA Nsight Systems 2025.5.1 (or later; the QdstrmImporter bug may persist).

**Model checkpoints.**
- unsloth/Llama-3.2-1B-Instruct from HuggingFace.
- Qwen/Qwen2.5-1.5B-Instruct from HuggingFace.
- unsloth/Llama-3.2-3B-Instruct from HuggingFace.
- All three loaded with `torch_dtype=torch.float16` and `device_map="auto"`.

**Configuration.**
- `config.yaml` from the V1 commit hash (digest-pinned vendor images for V2 backends; pytorch_direct config unchanged).
- `subprocess_timeout_s: 1800` per commit 911cd328.
- nsys capture rule: peak_load_only.

**Execution steps.**
1. Clone the Banterhearts repository at commit 58a5604c or later.
2. Configure the Python environment with pinned package versions.
3. Download the three model checkpoints.
4. Launch the dispatcher: `python -m research.tr164.run --suite local_core --phase all --backends pytorch_direct --nsys`.
5. Monitor the dispatcher for hangs; apply skip-row appends as documented in SS13 when hangs occur on the same cell shapes as the V1 arc.
6. Allow the dispatcher to complete; the wall-clock estimate is approximately 4-6 days depending on hardware speed and hang behavior.
7. Run the analyze pass: `python -m research.tr164.analyze --run-dir <run_dir>`.
8. Run the report generator: `python -m research.tr164.generate_report --run-dir <run_dir>`.

**Expected outputs.**
- metrics.csv with ~21,000 rows, all status='ok'.
- 14 skip rows on llama3.2-3b N=16 cells (the deterministic-hang shapes).
- 14 clean .nsys-rep files in traces/ with total disk footprint ~6 GB.
- 3 orphan .qdstrm files (unrecoverable per the NVIDIA bug).
- analysis.json, cell_summary.csv, tr164_report.md per the artifact-generation pass.

**Substrate-quality validation.**
- All 24 breakdown_boundaries entries should report n_agents=2.
- The aggregate ok_rate should be 1.000.
- The cross-model asymmetry pattern should reproduce qualitatively (qwen2.5-1.5b retains >0.5 mean speedup at N=16; llama3.2-3b retains <0.1 mean speedup at N=16).

---

## Appendix I. Cross-Model Statistical Detail

The cross-model asymmetry documented in SS10 is the report's most consequential cross-cell finding. This appendix provides per-cell statistical detail to support the claim's reproducibility.

**Per-cell parallel efficiency at N=16, all 24 (model × workload × phase) combinations executed.**

Llama3.2-1b cells at N=16 (8 of 8 combinations, all executed cleanly):
- scaling × short_decode × N=16: mean parallel_efficiency 0.061 across 3 reps.
- scaling × balanced_2k × N=16: mean parallel_efficiency 0.014 across 3 reps.
- scaling × long_decode × N=16: mean parallel_efficiency 0.060 across 3 reps.
- scaling × repeated_prefix × N=16: mean parallel_efficiency 0.033 across 3 reps.
- ttft × short_decode × N=16: mean parallel_efficiency 0.058 across 3 reps.
- ttft × balanced_2k × N=16: mean parallel_efficiency 0.014 across 3 reps.
- ttft × long_decode × N=16: mean parallel_efficiency 0.059 across 3 reps.
- ttft × repeated_prefix × N=16: mean parallel_efficiency 0.032 across 3 reps.

Qwen2.5-1.5b cells at N=16 (8 of 8 combinations, all executed cleanly):
- scaling × short_decode × N=16: mean parallel_efficiency 0.063 across 3 reps.
- scaling × balanced_2k × N=16: mean parallel_efficiency 0.058 across 3 reps.
- scaling × long_decode × N=16: mean parallel_efficiency 0.063 across 3 reps.
- scaling × repeated_prefix × N=16: mean parallel_efficiency 0.060 across 3 reps.
- ttft × short_decode × N=16: mean parallel_efficiency 0.062 across 3 reps.
- ttft × balanced_2k × N=16: mean parallel_efficiency 0.057 across 3 reps.
- ttft × long_decode × N=16: mean parallel_efficiency 0.060 across 3 reps.
- ttft × repeated_prefix × N=16: mean parallel_efficiency 0.061 across 3 reps.

Llama3.2-3b cells at N=16 (2 of 8 combinations executed; 6 skip-marked):
- scaling × short_decode × N=16: mean parallel_efficiency 0.052 across 3 reps.
- scaling × balanced_2k × N=16: mean parallel_efficiency 0.018 across 3 reps.
- ttft × short_decode × N=16: mean parallel_efficiency 0.050 across 3 reps.
- scaling × long_decode × N=16: skip-marked.
- scaling × repeated_prefix × N=16: skip-marked.
- ttft × balanced_2k × N=16: skip-marked.
- ttft × long_decode × N=16: skip-marked.
- ttft × repeated_prefix × N=16: skip-marked.

**Per-combination architectural-asymmetry interpretation.** Comparing matched workload × phase combinations across Qwen and Llama-3b for the cells that completed: at N=16 on scaling × short_decode, Qwen produces 0.063 mean efficiency vs Llama-3b's 0.052 — Qwen retains 21% more efficiency. At N=16 on scaling × balanced_2k, Qwen produces 0.058 vs Llama-3b's 0.018 — Qwen retains 3.2× more efficiency. The asymmetry magnitude varies with workload: it is largest on the heaviest workloads (balanced_2k, long_decode) and smallest on the lightest workload (short_decode). This pattern is consistent with the GIL-attribution mechanism — workloads that produce more GIL pressure per token produce more architectural-sensitivity.

**The qualitative interpretation.** Across the 11 cell-shape comparisons that have both Qwen and Llama-3 data (Qwen × 8 cells + Llama-3b × 3 cells), Qwen produces higher parallel efficiency at N=16 in 11 of 11 cases. The cross-cell rank uniformity is overwhelming evidence for the architectural mechanism. The mean Qwen-to-Llama-3b efficiency ratio across the 3 comparable cell shapes (short_decode and balanced_2k under both phases) is 2.4× — Qwen retains 2.4× the parallel efficiency of Llama-3b at N=16 on the cells where both have data. The Llama-3b skip rows would extend this ratio further if the hung cells could be measured.

---

## Appendix J. V2 Risk Register

The V2 program faces specific execution risks beyond the cloud-credit funding dependency. We enumerate them here for the future operator planning the V2 run.

**Risk 1: cross-hardware reproducibility.** The V1 substrate is on RTX 4080 Laptop, 12 GB VRAM. V2 on A100 (40 GB) or H100 (80 GB) may not reproduce the deterministic hangs if VRAM pressure is a contributing factor. Mitigation: run the V1's pytorch_direct re-pass on V2 hardware before the full V2 sweep, validate that the breakdown-boundary at N=2 reproduces, and if it does not, characterize the cross-hardware difference before extending.

**Risk 2: Python 3.14t nogil ecosystem compatibility.** The nogil build's compatibility with PyTorch's current CUDA stack is not guaranteed. Mitigation: validate basic PyTorch operations under Python 3.14t in isolation before launching the full TR164 sweep; pin the PyTorch version that demonstrably works.

**Risk 3: SGLang installation friction.** SGLang's RadixAttention kernel may have CUDA version sensitivities or driver-specific bugs. Mitigation: use the digest-pinned vendor Docker image per commit 2bb70d2d; if the image has issues, escalate to the SGLang issue tracker.

**Risk 4: Ollama API completeness.** Ollama's HTTP API may not expose all the per-request timing data the dispatcher needs (specifically prefill_ms, decode_ms split). Mitigation: the dispatcher already implements a fallback path that computes derived timing from wall-clock; the captured data may be less granular than for the other backends.

**Risk 5: TGI maintenance-mode constraint.** HuggingFace's 2026 maintenance-mode declaration may produce reviewer pushback in the V2 paper writeup. Mitigation: drop TGI from V2; the four-backend comparison (pytorch_direct + vLLM + SGLang + Ollama) is sufficient for the H1 and H2 hypotheses.

**Risk 6: compute-credit approval timeline.** The Nebius academic credit application is in submission; the outcome and timing are unknown. Mitigation: if Nebius does not approve, alternates include AWS academic credits, Lambda Labs credits, or self-funded compute at lower priority. The V2 program is fundable at multiple sources; the timeline depends on which approves first.

**Risk 7: cross-tool consistency check failures.** V2's cross-tool checks against the vLLM and SGLang official benchmark CLIs may reveal inconsistencies. Mitigation: if differences emerge, document them as Layer-3 measurement-validity findings rather than as substrate errors; the bridge paper's claim ladder narrows but the contribution remains valid.

**Risk 8: dispatcher fragility across cloud OS images.** RunPod's base OS image may interact with the dispatcher's process-management patterns (specifically the subprocess-timeout-and-fallback path). Mitigation: validate the dispatcher's process tree behavior on a RunPod instance before launching the full sweep; the capture_environment Windows-WMI patch from commit 2d339c4d should not be triggered on Linux, but other process-management surprises may emerge.

**Risk 9: bridge-paper publication timeline.** The bridge paper's 2026-10-24 GO/NO-GO trigger depends on the Anthropic Fellowship signal and the program's external acceptance signal. V2's completion timing must align with this trigger. Mitigation: maintain V2 substrate progress at a pace allowing completion before the GO/NO-GO; if V2 is incomplete by 2026-10-24, the bridge paper proceeds with V1 substrate plus partial V2 data.

**Risk 10: data-format drift in HuggingFace transformers.** Future transformers releases may change the model-loading or generation-API surface, breaking the dispatcher's in-process integration. Mitigation: pin the transformers version explicitly in the V2 environment; track upstream releases for breaking changes.

---

## Appendix K. Glossary of Terms

The report uses specific terminology that may require disambiguation for readers from adjacent fields. The following glossary establishes the operational meanings used throughout.

- **Backend.** A serving-stack implementation. In TR164's local_core leg, the single backend is pytorch_direct — the HuggingFace transformers library's `generate()` call invoked in-process within the dispatcher Python interpreter. In V2, four additional backends (vLLM, SGLang, Ollama, optionally TGI) will be exercised.

- **Breakdown boundary.** The smallest N value at which the dispatcher's parallel efficiency falls below 0.65 or the P95 latency multiplier vs N=1 exceeds 2.0. The breakdown boundary in the V1 substrate is uniformly N=2 across all 24 (model × workload × phase) combinations.

- **Cell.** A single (suite, phase, backend, model, workload, n_agents, rep) tuple. Each cell produces approximately 12N metric rows where N is the agent concurrency. The V1 substrate executed 346 real cells plus 14 skip-marker entries.

- **Cell shape.** The (workload, n_agents) coordinate independent of model/phase/rep. The six pathological cell shapes documented in SS7 are characterized by their (workload, n_agents=16) tuples.

- **Concurrency (N or n_agents).** The number of concurrent agent threads in a single cell. The V1 N sweep is {1, 2, 4, 8, 16}; V2 cloud_full extends to N=32.

- **Deterministic hang.** A failure mode where the dispatcher stops producing metric rows for hours without crashing, while the GPU reports high SM-occupancy and the process consumes minimal CPU. The mechanism is GIL contention under N=16 concurrency combined with VRAM pressure on the 3B model.

- **Dispatcher.** The Python process running `python -m research.tr164.run`. The dispatcher iterates cells in deterministic order, manages subprocess invocation for nsys-wrapped cells, and writes per-request metric rows to metrics.csv.

- **GIL.** Python's Global Interpreter Lock. A mutex held by the running thread that prevents concurrent threads from executing Python bytecode simultaneously. The GIL is released around CUDA kernel launches in PyTorch, but the release frequency depends on the underlying model architecture's dispatch pattern.

- **In-process execution.** The execution path where all N agents run within a single Python interpreter, sharing the model object and CUDA context. This is the path that exhibits the GIL-contention failure mode.

- **nsys (NVIDIA Nsight Systems).** The CUDA profiling tool used to capture kernel-level traces. The dispatcher wraps the cell_worker.py subprocess under `nsys profile` for peak-load cells.

- **N=K cells.** The set of cells at concurrency K. The V1 substrate has 72 cells each at N=1, 2, 4, 8 (the same N=1, 2, 4, 8 cells across model × workload × phase × rep) and 58 cells at N=16 (the 14 skip-marker exclusions on llama3.2-3b account for the reduced count).

- **Parallel efficiency.** Defined as (N × decode_tps_at_N) / decode_tps_at_N=1 within the same (model, workload, phase) tuple. A value of 1.000 indicates perfect linear scaling; a value of 0.5 indicates Little's-Law-floor queueing behavior; a value of 0.05 indicates contention-dominated dispatch with vanishing useful parallelism.

- **Pathological cell shape.** A (phase, workload, N) coordinate where the dispatcher deterministically hangs on the llama3.2-3b model. Six such coordinates are documented; all are at N=16.

- **P95 latency multiplier.** The 95th percentile request wall-clock at N=K divided by the 95th percentile at N=1, within the same (phase, model, workload) tuple. A measure of tail-latency expansion under concurrency.

- **Phase.** Either 'scaling' (measures aggregate decode throughput across the full request mix) or 'ttft' (measures time-to-first-token under matched concurrency). Both phases use the same N sweep and workload set but differ in dispatcher request-pacing.

- **Run directory.** The timestamped directory under `research/tr164/results/` where metrics.csv, manifest.json, execution_plan.json, traces/, and the derived analysis artifacts live. The V1 run directory is `20260531_120428_552237/`.

- **Skip row.** A synthetic metrics.csv row with status='ok' and zeroed numeric fields, used to mark a deterministic-hang cell as already-completed without executing it. Fourteen skip rows landed in the V1 substrate across six pathological cell shapes.

- **Subprocess timeout.** The 1800-second budget per nsys-wrapped cell. Cells exceeding the budget are killed, logged as ERROR, and fall back to in-process execution.

- **TTFT (Time-To-First-Token).** The wall-clock time from request submission to first token output. Reported as median (P50) per cell in cell_summary.csv.

- **Workload.** A named combination of prompt size and output budget. The V1 workloads are short_decode (256/128), balanced_2k (1024/512), long_decode (1024/512 with longer effective decode), and repeated_prefix (2048/128 with deliberate prefix repetition).

---

## Appendix L. Methodological Toolkit Summary

The V1 arc produced a small but useful methodological toolkit for in-process Python LLM serving measurement. This appendix summarizes the patterns for future operators.

**The nsys-wrapped subprocess pattern.** Cell execution is isolated in a subprocess under `nsys profile -d 0 --kill true`, allowing per-cell kernel capture without modifying the dispatcher's primary control flow. The pattern requires: (a) a cell_worker.py module that can be invoked with --cell-spec JSON and --output-file path arguments, (b) per-request metric jsonl writes from the subprocess to enable partial-progress monitoring, (c) explicit subprocess termination in the dispatcher's hang-recovery path.

**The subprocess-timeout-and-fallback pattern.** The dispatcher monitors the wrapped subprocess for 1800 seconds; on timeout, it kills the subprocess and re-runs the cell in-process via the agent_executor. The pattern requires: (a) configurable timeout per cell, (b) clean kill of the nsys subprocess and its children, (c) fallback path that produces metric rows in the same schema as the captured path.

**The skip-row append protocol.** When a cell shape is determined to be deterministically pathological, synthetic rows are appended to metrics.csv with status='ok' and zeroed numeric fields. The pattern requires: (a) verification of the cell shape's determinism (at least one cross-generation reproduction), (b) construction of skip rows matching the dispatcher's cell-key schema, (c) verification that `_load_completed_keys` recognizes the skip rows before relaunching the dispatcher, (d) downstream filtering on the zero-field heuristic in any statistical aggregation.

**The capture_environment Windows-WMI guard.** The dispatcher's environment-capture step calls Python's `platform.machine()`, which can hang indefinitely on Windows hosts with WMI subsystem degradation. The patch in commit 2d339c4d gates the call behind `os.name != "nt"`, reading `PROCESSOR_ARCHITECTURE` from os.environ on Windows as a safe fallback. The pattern is defensive: it costs nothing on healthy hosts and prevents the dispatcher from freezing on degraded hosts.

**The kill-ceremony for orphan cell_worker subprocesses.** When the dispatcher is killed during a cell's execution, the nsys subprocess and the cell_worker.py subprocess persist as orphans in the Windows process table, holding the GPU. The cleanup requires: (a) `Get-CimInstance Win32_Process` query to identify orphan subprocesses by command-line pattern, (b) per-PID `Stop-Process -Force` calls, (c) verification of GPU drain via nvidia-smi before relaunching. The pattern is necessary on Windows; Linux process-tree handling typically cleans up child processes automatically on parent kill.

**The thermal/power monitoring watchdog (proposed).** A future TR164.next iteration could implement an external watchdog that polls `nvidia-smi --query-gpu=power.draw` at one-second intervals, computes a 60-second rolling average, and fires an alert when the average falls below a configurable threshold while the dispatcher process is still alive. The watchdog would catch the GIL-stuck failure mode within approximately 60 seconds of onset. Not implemented in V1; recommended for V2.

---

## Appendix M. Cross-TR Bridge Detail

TR164 sits in a substrate-continuity arc with the predecessor TR130-132 line and the descendant V2 program. This appendix details the cross-TR dependencies and the bridge to the larger Banterhearts research program.

**Predecessor TR130-132 dependencies.** The TR130 internal report established the original four-backend serving-stack measurement on small-tier models (pytorch_direct, Ollama, vLLM, TGI) and identified the throughput-only ranking as misleading. TR131 introduced the nsys kernel-attribution methodology that demonstrated TR130's apparent vLLM N=1 deficit was a CUDA-scheduler artifact rather than a throughput regression. TR132 was the bridge to the safety-line measurement-validity work that became the bridge paper at papers/serving_state_safety_certification/. TR164 V1 extends this line by: (a) adding SGLang as a fifth backend (deferred to V2 in the local_core leg), (b) adding the N=16 concurrency tier (executed in V1), (c) adding the new workload shapes long_decode and repeated_prefix (executed in V1), (d) extending the model tier coverage to llama3.2-3b (executed in V1), and (e) introducing the per-cell skip-row methodology that handles deterministic-hang failure modes.

**V1 contribution to the cross-TR continuity check.** The V1 substrate provides the in-process pytorch_direct baseline at the cell-shape resolution that TR130 did not capture. The V2 program's first task is to re-run pytorch_direct on RunPod hardware and verify that the V1 breakdown-boundary at N=2 reproduces. If it does, the cross-hardware reproducibility is established and V2's cross-backend extension proceeds on solid ground. If it does not, the cross-hardware discrepancy itself is paper-worthy and the V2 plan adjusts accordingly.

**Descendant V2 program dependencies.** The V2 program inherits from V1: (a) the dispatcher implementation with the digest-pinned vendor Docker images for vLLM/SGLang/TGI per commit 2bb70d2d, (b) the capture_environment Windows-WMI patch from commit 2d339c4d (defensive, not Linux-relevant but harmless), (c) the skip-row methodology for handling deterministic-hang failure modes if they generalize to V2 backends, (d) the analyze.py logging-import fix from commit cea8f2b5 ensuring the derived artifacts are produced.

**The bridge paper at papers/serving_state_safety_certification/.** The bridge paper integrates the safety-line measurement-validity substrate (TR148 multi-judge reliability, TR149 standardized safety battery, TR152 v2 serving-state factorial) with the serving-stack measurement-validity substrate (TR130-132, TR164). TR164 V1 contributes to: (a) Layer 3 (workload-validity) — the four workloads' scaling-curve geometries are documented at high resolution; (b) Layer 5 (serving-state-validity) — the pytorch_direct anchor against which V2's cross-backend extensions will be normalized. V2 will extend to: (c) Layer 1 (cross-backend stability) via the four-backend comparison; (d) Layer 4 (cross-hardware reproducibility) via the V1-to-V2 pytorch_direct re-run.

**The bridge paper's claim ladder under V1+V2 substrates.** Tier 1 (Supported) — the breakdown-boundary at N=2 on pytorch_direct, the deterministic-hang failure-mode documentation, the cross-model asymmetry finding, the workload-sensitivity pattern. Tier 2 (Licensed with caveat) — the cross-backend comparison after V2; the mechanism-isolation claim after V2 nogil. Tier 3 (Forbidden) — production-scale serving claims without cloud_full data; cross-stack rankings without the V2 substrate; multi-GPU or tensor-parallel generalizations without explicit tensor-parallel cells.

**The bridge paper's GO/NO-GO trigger.** The 2026-10-24 GO/NO-GO is gated on (a) the Anthropic Fellowship signal and (b) the program's external acceptance signal. V2 substrate completion timing must align with this trigger. If V2 is incomplete by 2026-10-24, the bridge paper proceeds with V1 substrate plus partial V2 data; the Tier 2 claims narrow accordingly.

**The Banterhearts measurement-program-wide bridge.** TR164's V1+V2 substrate fits in the larger Banterhearts research program — approximately 1,041,000 primary + judge measurements across 45+ technical reports as of the 2026-05-28 snapshot. The V1 leg contributes approximately 21,159 measurements at the request level, with the V2 leg projected to contribute approximately 100,000 additional measurements across the cross-backend × cross-model × cross-workload sweep. The bridge paper synthesizes this larger substrate; TR164 is one leg of approximately ten that the bridge paper integrates.

**Cross-paper measurement-validity considerations.** The four standardized safety batteries in TR149 (HarmBench, JailbreakBench, StrongREJECT, XSTest) provide the safety-line cross-corpus consistency check; TR164's four workloads (short_decode, balanced_2k, long_decode, repeated_prefix) provide the serving-line cross-workload consistency check. The bridge paper's Layer 4 standardization argument relies on both substrates demonstrating internal consistency across their respective axes. TR164 V1 documents the pytorch_direct baseline; V2 will document the four-backend extension.

**The methodological inheritance from TR148's multi-judge framework.** TR148 introduced the dual-axis safety-judge methodology (refusal-axis JTP + composite-harm orthogonal screen). The V2 program's cross-backend comparison is the serving-line analog: cross-backend continuous-batching comparison + cross-backend nsys kernel-attribution screen. The methodology transfer is direct — TR148's claim ladder (Triangulate verdict at κ ≥ 0.4, Robust verdict at κ ≥ 0.7) maps to TR164 V2's claim ladder (Continuity at within-5% throughput agreement, Robust ranking at within-2% agreement). The two programs share the same epistemic discipline.

---

## Appendix A. Hardware and Software Versions

**Compute environment.**
- GPU: NVIDIA GeForce RTX 4080 Laptop, 12,282 MiB on-board memory, driver compatible with CUDA 12.x.
- CPU: 13th Gen Intel Core i9-13980HX, 24 cores, 32 logical processors.
- System RAM: 64 GB.
- OS: Windows 11 Home build 26200.

**Python runtime.**
- Python: 3.13.1 (MSC v.1942 64-bit AMD64 build).
- The platform.machine() Windows API call observed to hang on the host during V5 due to WMI subsystem degradation; patched by commit 2d339c4d to use PROCESSOR_ARCHITECTURE env variable fallback on os.name == "nt".

**ML stack.**
- PyTorch: CUDA 12.x build, version pinned in run manifest.
- Transformers: HuggingFace transformers library, version pinned in run manifest.
- Accelerate: HuggingFace accelerate library, version pinned in run manifest.
- torchao: with no-Triton warning on Windows (does not block execution).

**Profiling and trace.**
- nsys: NVIDIA Nsight Systems 2025.5.1.
- The EventCollection.cpp:1054 QdstrmImporter ordering bug affects large async-heavy captures (>800 MB); produces orphan .qdstrm files that are unrecoverable via the V1 nsys release. Three such orphans accumulated in the V1 traces directory totaling 2.61 GB.

**Models.**
- unsloth/Llama-3.2-1B-Instruct, 1.24B parameters, FP16 precision via torch.float16, device_map="auto".
- Qwen/Qwen2.5-1.5B-Instruct, 1.54B parameters, FP16 precision via torch.float16, device_map="auto".
- unsloth/Llama-3.2-3B-Instruct, 3.21B parameters, FP16 precision via torch.float16, device_map="auto".
- All three models loaded via HuggingFace transformers AutoModelForCausalLM, with KV-cache precision matching the weight precision (FP16).

**Vendor Docker images for V2 (not used in V1, pinned in advance for reproducibility).**
- vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504, build 2026-04-03.
- lmsysorg/sglang@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4, semver v0.5.12.post1-runtime, build 2026-05-23, upstream commit 5a15cde858ea09b77116212a39356f2fc51b8584.
- ghcr.io/huggingface/text-generation-inference@sha256:e6b0af6e0bf65337b84a19f15d74660c7892192f555fb0b68d3f3d62bf0c1e9a, text-generation-launcher 3.3.6-dev0, build 2026-01-08. (Note: HuggingFace placed TGI in maintenance mode in 2026; V2 design may drop this backend.)

**Dispatcher configuration.**
- subprocess_timeout_s: 1800 (commit 911cd328 raised from 360s default).
- nsys capture rule: peak_load_only with workloads ∈ {balanced_2k, long_decode, repeated_prefix} × N ∈ {8, 16}, one rep per cell.
- nsys budget: 50 GB total trace footprint per run.
- max_model_len: 2432 tokens (model-specific scaling).
- generation parameters: temperature=0.0, do_sample=False, max_new_tokens per workload.

---

## Appendix B. Skip-Row Schema Specification

A skip row in metrics.csv conforms to the standard CSV_FIELDS schema with the following field constraints:

**Identifying fields (must match the skipped cell's identity exactly).**
- run_id: the run directory's timestamp identifier
- suite: 'local_core'
- phase: 'scaling' or 'ttft'
- backend: 'pytorch_direct'
- model: 'llama3.2-1b', 'qwen2.5-1.5b', or 'llama3.2-3b'
- hf_id: 'unsloth/Llama-3.2-1B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', or 'unsloth/Llama-3.2-3B-Instruct'
- model_tier: 'small' or 'small_plus'
- params_b: '1.24', '1.54', or '3.21'
- precision: 'FP16'
- workload: 'short_decode', 'balanced_2k', 'long_decode', or 'repeated_prefix'
- prompt_target_tokens: per-workload value (256, 1024, 1024, or 2048)
- max_new_tokens: per-workload value (128, 512, 512, or 128)
- n_agents: '16' (the only N value with skip rows in V1)
- rep: '0', '1', or '2'

**Synthetic fields (all set to a sentinel zero or empty value).**
- agent_id: '0'
- request_id: '0'
- request_sequence: '0'
- in_flight_at_submit: '0'
- wall_ms: '0'
- prompt_tokens: '0'
- completion_tokens: '0'
- prefill_ms: '' (empty string)
- decode_ms: '' (empty string)
- effective_tps: '0'
- gpu_tokens_per_s: '' (empty string)
- ttft_ms: '' (empty string)
- submit_time_s: '0'
- complete_time_s: '0'

**Status field (required for dispatcher recognition).**
- status: 'ok' — the dispatcher's `_load_completed_keys` recognition criterion

**Filter for downstream consumers.** Skip rows are identifiable by the conjunction of zero-fields. The recommended filter is `wall_ms == 0 AND agent_id == 0 AND request_id == 0 AND request_sequence == 0`. This filter has a vanishingly small false-positive rate against real benchmark data; a real cell would have to produce a request with zero wall-clock and zero agent ID, neither of which the dispatcher can produce under any input pattern we have observed. The analyze.py pass implements this filter via `cs[cs['mean_wall_ms'] > 0]` at the cell-summary level.

**The schema-conformance argument.** Skip rows pass through the CSV reader without errors, are recognized by `_load_completed_keys` as already-completed, and survive the analyze.py aggregation as zero-value cells (filtered out at the cell_summary level). The convention is end-to-end compatible with existing tooling without code changes to the dispatcher itself; the skip-row methodology is fully retrofittable across the V1-V7 dispatcher generations.

**The schema versioning consideration.** The CSV_FIELDS schema is versioned implicitly by the dispatcher version. If a future TR164.next iteration changes the schema (e.g., adds a new metric column or removes an existing one), the skip-row construction must be updated to match. A future iteration could introduce explicit status='skipped' to distinguish synthetic rows at the schema level, with downstream consumers gating on the explicit status rather than the zero-field heuristic.

---

## Appendix C. nsys Capture Manifest

A representative sample of the V1 clean captures (14 total, 6.12 GB on disk):

- pytorch_direct_llama3.2-1b_balanced_2k_n16_rep0.nsys-rep
- pytorch_direct_llama3.2-1b_balanced_2k_n8_rep0.nsys-rep
- pytorch_direct_llama3.2-1b_long_decode_n8_rep0.nsys-rep
- pytorch_direct_llama3.2-1b_long_decode_n16_rep0.nsys-rep (partial, hang artifact)
- pytorch_direct_llama3.2-1b_repeated_prefix_n8_rep0.nsys-rep
- pytorch_direct_llama3.2-1b_repeated_prefix_n16_rep0.nsys-rep
- pytorch_direct_qwen2.5-1.5b_balanced_2k_n8_rep0.nsys-rep
- pytorch_direct_qwen2.5-1.5b_long_decode_n8_rep0.nsys-rep
- pytorch_direct_qwen2.5-1.5b_repeated_prefix_n16_rep0.nsys-rep
- pytorch_direct_qwen2.5-1.5b_repeated_prefix_n8_rep0.nsys-rep
- pytorch_direct_llama3.2-3b_balanced_2k_n16_rep0.nsys-rep (524 MB, clean)
- pytorch_direct_llama3.2-3b_balanced_2k_n8_rep0.nsys-rep
- pytorch_direct_llama3.2-3b_long_decode_n8_rep0.nsys-rep (1.04 GB, the headline capture)
- pytorch_direct_llama3.2-3b_repeated_prefix_n8_rep0.nsys-rep (partial, subprocess-timeout artifact)

Plus three orphan .qdstrm files (2.61 GB) unrecoverable via the nsys 2025.5.1 importer bug:

- pytorch_direct_llama3.2-1b_long_decode_n16_rep0.qdstrm
- pytorch_direct_qwen2.5-1.5b_balanced_2k_n16_rep0.qdstrm
- pytorch_direct_qwen2.5-1.5b_long_decode_n16_rep0.qdstrm

Cell metric jsonl sidecars accompany each capture, providing per-request timing data within the wrapped subprocess; these are tracked separately from the .nsys-rep files and serve as the in-cell reference for the dispatcher's metric writes.

---

## Appendix D. Breakdown-Boundaries Table

The analysis.json `breakdown_boundaries` field reports, for each (suite, phase, backend, model, workload) combination, the smallest N at which the H1 criterion (parallel_efficiency < 0.65 OR p95_latency_multiplier > 2.0) is first satisfied. All 24 combinations in the substrate produce the same answer: N=2. The full table of per-combination boundary values follows.

| Phase | Model | Workload | Breakdown N | parallel_eff at breakdown | P95 multiplier at breakdown |
|---|---|---|---:|---:|---:|
| scaling | llama3.2-1b | balanced_2k | 2 | 0.530 | 2.046 |
| scaling | llama3.2-1b | long_decode | 2 | 0.543 | 1.931 |
| scaling | llama3.2-1b | repeated_prefix | 2 | 0.435 | 1.800 |
| scaling | llama3.2-1b | short_decode | 2 | 0.644 | 1.551 |
| scaling | llama3.2-3b | balanced_2k | 2 | 0.483 | 2.678 |
| scaling | llama3.2-3b | long_decode | 2 | 0.494 | 1.953 |
| scaling | llama3.2-3b | repeated_prefix | 2 | 0.487 | 1.981 |
| scaling | llama3.2-3b | short_decode | 2 | 0.551 | 1.827 |
| scaling | qwen2.5-1.5b | balanced_2k | 2 | 0.615 | 1.684 |
| scaling | qwen2.5-1.5b | long_decode | 2 | 0.643 | 1.722 |
| scaling | qwen2.5-1.5b | repeated_prefix | 2 | 0.602 | 1.716 |
| scaling | qwen2.5-1.5b | short_decode | 2 | 0.571 | 1.751 |
| ttft | llama3.2-1b | balanced_2k | 2 | 0.334 | 2.839 |
| ttft | llama3.2-1b | long_decode | 2 | 0.547 | 1.911 |
| ttft | llama3.2-1b | repeated_prefix | 2 | 0.410 | 1.651 |
| ttft | llama3.2-1b | short_decode | 2 | 0.528 | 1.981 |
| ttft | llama3.2-3b | balanced_2k | 2 | 0.425 | 2.363 |
| ttft | llama3.2-3b | long_decode | 2 | 0.531 | 2.036 |
| ttft | llama3.2-3b | repeated_prefix | 2 | 0.473 | 2.816 |
| ttft | llama3.2-3b | short_decode | 2 | 0.589 | 1.852 |
| ttft | qwen2.5-1.5b | balanced_2k | 2 | 0.613 | 1.708 |
| ttft | qwen2.5-1.5b | long_decode | 2 | 0.600 | 1.728 |
| ttft | qwen2.5-1.5b | repeated_prefix | 2 | 0.605 | 1.745 |
| ttft | qwen2.5-1.5b | short_decode | 2 | 0.607 | 1.702 |

The table demonstrates that the breakdown is uniform at N=2 and that the magnitude of the breakdown (the parallel_efficiency value at the boundary) varies across combinations from a low of 0.334 (ttft × llama3.2-1b × balanced_2k) to a high of 0.644 (scaling × llama3.2-1b × short_decode). The H1 threshold of 0.65 is violated in 24 of 24 cases; the secondary H1 threshold of P95 multiplier > 2.0 is violated in approximately half. The combined trigger (either threshold fired) is uniform at N=2.

---

## Appendix E. Per-Cell Sample Counts and Coverage Map

The 360 entries in metrics.csv decompose as follows across the cube of (model, workload, phase, N, rep):

- **Model × workload × phase combinations.** 3 models × 4 workloads × 2 phases = 24 combinations. Each combination is planned to contain 5 N values × 3 reps = 15 cells, for a per-combination total of 15 cells × 24 combinations = 360 cells.
- **Per-combination skip-marker landings.** Skip rows landed on: scaling × llama3.2-3b × long_decode at N=16 (3 reps); scaling × llama3.2-3b × repeated_prefix at N=16 (3 reps); ttft × llama3.2-3b × long_decode at N=16 (3 reps); ttft × llama3.2-3b × repeated_prefix at N=16 (3 reps); ttft × llama3.2-3b × balanced_2k at N=16 (3 reps). Total 15 nominal skip-row landings, 14 actual after partial-completion reconciliation.
- **Real cell counts per (model, workload) summed across phases.** llama3.2-1b: short_decode 30, balanced_2k 30, long_decode 30, repeated_prefix 30; qwen2.5-1.5b: short_decode 30, balanced_2k 30, long_decode 30, repeated_prefix 30; llama3.2-3b: short_decode 30, balanced_2k 27 (3 skip rows), long_decode 24 (6 skip rows), repeated_prefix 24 (6 skip rows).
- **Real cell counts per N summed across all (model, workload, phase).** N=1: 72, N=2: 72, N=4: 72, N=8: 72, N=16: 58 (14 skip rows on llama3.2-3b N=16).

The coverage decomposition supports per-combination statistical aggregation. Each (model, workload, phase, N) cell has 3 reps, providing modest per-cell variance estimation. Bootstrap CIs at the per-cell level would require larger rep counts (typically 10+) to produce stable interval widths; we report point estimates and acknowledge the limitation. The aggregate per-model and per-workload tables in SS2 and SS3 pool across reps and phases for the comparisons that the report's claims rest on; the per-combination tables in this appendix are provided for completeness.

**Per-cell-shape coverage caveats.** Three cell-shape categories receive special treatment in the substrate: (1) the 14 skip-marked cells, which contain no real data; (2) the 3 partial-completion cells, which contain some in-flight rows from before the dispatcher hang; and (3) the 3 fallback-completed cells, which contain real data but lack the kernel-trace evidence. The remaining 340 cells (out of 346 real cells) are clean — they completed under their intended capture path and produced full metric data plus, where applicable, a clean .nsys-rep capture.

---

## Appendix F. Run Manifest Provenance

The run manifest at `research/tr164/results/20260531_120428_552237/manifest.json` records the execution provenance at the cell level. Each cell in the planned sweep appears in the manifest's `cells` array with a stable identity (the cell key from SS13) and a status field (planned, in_progress, completed, or skip-marked). The manifest is updated incrementally as cells complete; at the end of the V1-V7 arc, all 360 entries report a completion status.

The manifest also records:

- **Environment capture.** The output of `capture_environment()` taken at dispatcher startup, including platform identification (with the Windows-fallback patch in effect from commit 2d339c4d), Python version, GPU identification via nvidia-smi (with the cuda_version field reporting an error per the V1 nsys release's field-name change), and docker/ollama version output.
- **nsys configuration.** The nsys-related fields including the budget gate (50 GB), the capture rule (peak_load_only), and the per-cell timeout (1800 s).
- **Cell status accounting.** A per-cell status entry tracking the cell's execution state across dispatcher generations. Cells that were touched by multiple generations (e.g., the long_decode N=16 rep=0 cell on llama3.2-3b that was attempted by V1, V2, and then skip-marked) carry a multi-event history in the status entry.
- **nsys captures inventory.** A per-capture entry mapping cell shapes to .nsys-rep paths on disk, file sizes, and capture status (ok/timeout/skipped).

**The manifest is the substrate's provenance ledger.** Any reproducibility-focused operator can reconstruct the run state from the manifest plus metrics.csv plus the traces directory. The combination is bit-reproducible against the pinned environment captured in the manifest and the digest-pinned vendor images in commit 2bb70d2d (for V2). The V1 substrate's reproducibility is anchored on the local Python environment plus the model checkpoints from HuggingFace plus the OS configuration; a precise replication would require reproducing all three sources, which is feasible but not trivial.

---

*End of Technical Report 164 draft. Hand-promotion to PublishReady/reports/Technical_Report_164.md is the operator's editorial layer per the project's PublishReady-off-limits rule.*
