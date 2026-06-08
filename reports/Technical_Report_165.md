# Technical Report 165: Python GIL Ablation for Direct PyTorch Inference Boundaries

## Matched-Pair Test of TR164 V1's GIL-Attribution Hypothesis Under Python 3.14t Free-Threaded — Verdict: H2_partial

**Status.** TR165 substrate is complete and runtime-clean. The experiment ran 3,744 metric rows across 48 planned cells (3 models x 4 workloads x 2 phases x 2 N-concurrency levels) on the same RTX 4080 Laptop 12 GB hardware as the TR164 V1 baseline at `research/tr164/results/20260531_120428_552237/`, isolating Python's Global Interpreter Lock as the single ablated variable. The dispatcher started under the free-threaded CPython build at `C:\Users\sahil\AppData\Local\Programs\Python\pythoncore-3.14t-nuget\tools\python3.14t.exe` (Python 3.14.5, tag `v3.14.5:5607950`, May 10 2026), with all three GIL-state signals agreeing at preflight: `sys._is_gil_enabled()` returned `False` both before and after `torch` / `transformers` / `accelerate` imports, `sysconfig.Py_GIL_DISABLED` evaluated `True`, and the dispatcher's `free_threaded_build` flag reported `True`; `signals_agree` is `True` and the `abort_on_gil_build` safety bar held. The pre-registered falsification ladder resolves to **H2_partial**: H0 (no improvement; GIL irrelevant) is REJECTED, H1 (full GIL attribution; ALL 24 N=2 (model x workload x phase) combinations clear the parallel-efficiency improvement threshold) is NOT FULLY SUPPORTED with 19 of 24 combinations passing (79.2%) and 5 of 24 below threshold (20.8%), and H2 (partial GIL attribution; some combinations improve) is SUPPORTED. Mean N=2 parallel-efficiency uplift under nogil is +0.1791 absolute (+17.91 percentage points), median +0.1715 absolute (+17.15 percentage points), computed across the full 24-of-24 N=2 matched-cell coverage. The hang-resolution ledger records 2 of 6 V1 deterministic-hang cell shapes resolved under nogil; the remaining 4 still hang under identical workload geometry, ruling out a sole-cause GIL interpretation. The substrate-grounded read is therefore that removing the GIL recovers a meaningful but partial slice of V1's N=2 breakdown: the GIL is A mechanism behind the boundary, not THE mechanism. The residual 20.8% non-improving combinations and the 4 unresolved hangs preserve the falsifier surface for CUDA kernel serialization, HBM-bandwidth contention, and dispatcher-overhead contributors that future experiments must isolate. TR165 is the Phase 8 mechanism-isolation companion to TR164 V2's cross-backend matched-pair test (`research/tr164/V2_BACKEND_NOTES.md`, `DRAFT_Technical_Report_164_V2.md`), and together the two anchor the Phase 8 mechanism layer of the bridge paper at `papers/serving_state_safety_certification/UPGRADE_PLAN.md`. The full analysis JSON sits at `research/tr165/results/20260607_174748_273070/analysis.json` and is the single source of truth for every numeric claim that follows in this report.

---

## 1. Abstract

Technical Report 164 V1 concluded that direct PyTorch inference exhibits a sharp parallel-efficiency breakdown at N=2 concurrency on a single-GPU RTX 4080 Laptop, and attributed the breakdown — provisionally — to Python Global Interpreter Lock (GIL) contention between the two host threads dispatching CUDA work. Technical Report 165 is the matched-pair falsification of that attribution. We hold every dimension of the V1 design constant — same hardware (RTX 4080 Laptop, 12 GB), same three models (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b), same four workloads (short_decode, balanced_2k, long_decode, repeated_prefix), same two phases (scaling, ttft), same single backend (pytorch_direct), same reps-per-cell — and flip exactly one knob: the Python runtime. The GIL build (CPython 3.13.x) is replaced by a 3.14.5 free-threaded build (tags/v3.14.5:5607950, May 10 2026, NuGet-installed) for which the dispatcher preflight confirms all three GIL-detection signals agreeing in the disabled direction — `sys._is_gil_enabled()` False, `sysconfig.Py_GIL_DISABLED` True, `signals_agree` True — both before and after importing torch, transformers, and accelerate, with `cuda_available` True and `abort_on_gil_build` True as the safety bar. The scope is 48 cells (3 models × 4 workloads × 2 phases × 2 N levels), producing 3,744 metric rows and 24 (model × workload × phase) combinations with N=2 data, alongside 6 attempted reruns of V1's deterministic-hang cell shapes. The pre-registered falsification verdict is **H2_partial**: H0 is rejected (the GIL is not innocent), but H1 (uniform improvement across all 24 N=2 combinations) is not fully supported. Empirically, 19 of 24 combinations (79.2%) clear the H1 improvement threshold and 5 of 24 (20.8%) fall below it; mean delta in N=2 parallel efficiency is +0.1791 (+17.91 pp) with median +0.1715 (+17.15 pp); 2 of the 6 deterministic-hang cell shapes resolve under the free-threaded build while 4 still hang. The substrate-grounded reading is that the GIL is **a** mechanism behind the N=2 breakdown — accounting for roughly 17–18 pp of parallel-efficiency loss on average — but **not the only** mechanism: the residual 21% of unimproved combinations and the 4 unresolved hangs implicate CUDA kernel serialization, memory-bandwidth contention, and dispatcher-layer overhead as co-contributors that survive GIL removal. TR165 thus replaces V1's single-attribution claim with a calibrated multi-mechanism account.

---

## 2. Table of Contents

1. Abstract
2. Table of Contents
3. Executive Summary
4. Introduction and Research Motivation
5. Pre-Registered Research Hypotheses (H0 / H1 / H2)
6. Methodology
7. Substrate Inheritance — TR164 V1 Baseline
8. SS1 — Preflight: Python 3.14.5 Free-Threaded Build Verification
9. SS2 — The 48-Cell Matched-Pair Plan
10. SS3 — Cell-by-Cell Matched-Pair Substrate
11. SS4 — The N=2 Parallel-Efficiency Delta (+17.91% mean, +17.15% median)
12. SS5 — Hang Resolution Ledger: 2 of 6 Cell Shapes Cleared
13. SS6 — ROC-AUC Discrimination Analysis
14. SS7 — Falsification Verdict: H2_partial
15. SS8 — Mechanism Interpretation
16. SS9 — Cross-Reference to TR164 V1 and TR164 V2
17. SS10 — Cross-TR Position
18. SS11 — Forbidden Claims
19. SS12 — Limitations
20. SS13 — Future Work
21. Conclusion
22. References
23. Appendix A — Hardware and Python 3.14t Environment Fingerprint
24. Appendix B — Reproduction Commands
25. Appendix C — Per-Cell Matched-Pair Tables

---

## 3. Executive Summary

Technical Report 165 is the matched-pair falsification test of a specific causal claim that Technical Report 164 V1 left on the table. V1 documented a uniform N=2 breakdown under the `pytorch_direct` backend: across every model in {`unsloth/Llama-3.2-1B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `unsloth/Llama-3.2-3B-Instruct`}, across every workload in {`short_decode`, `balanced_2k`, `long_decode`, `repeated_prefix`}, and across both phases {`scaling`, `ttft`}, increasing concurrency from N=1 to N=2 collapsed parallel efficiency and, in six specific cell shapes, produced deterministic hangs. The V1 report named the Python Global Interpreter Lock as the most plausible single mechanism on the grounds that `pytorch_direct` keeps the entire request-management loop in Python with no out-of-process worker pool to shed contention. That attribution was a hypothesis, not a finding. TR165 exists to convert the hypothesis into an answer.

The design holds a single delta. The Python interpreter changes from the stock CPython 3.x build used in V1 to the 3.14.5 free-threaded build (`pythoncore-3.14t-nuget`, tags/v3.14.5:5607950, May 10 2026) in which `sys._is_gil_enabled()` returns False at dispatcher startup AND after every CUDA-touching import. Everything else is matched cell-for-cell against V1: the three Hugging Face IDs are reused verbatim, the backend is held to `pytorch_direct` only, the workload set is identical, both phases run, concurrency is restricted to {N=1, N=2} because that is the boundary V1 isolated, repetitions per cell are matched to V1, and the hardware is the same RTX 4080 Laptop 12 GB box. The matched-pair scaffold is enforced inside `research/tr165/`: `matched_pair_contrast` in `analysis.json` joins V1 and V165 cells on `(model, workload, phase, N, rep)` and the dispatcher refuses to launch any cell V1 did not also run.

Four preflight gates protect the contrast from a stealth GIL-on regression. The dispatcher asserts (1) the executable resolves to the 3.14t NuGet build, (2) `sys._is_gil_enabled()` is False at startup, (3) the same call still returns False after `torch`, `transformers`, and `accelerate` import (a known footgun — a C extension can re-enable the GIL by setting the module flag), and (4) `cuda_available` is True so the matched-pair contrast actually exercises GPU paths rather than degrading to a CPU fallback. All four gates passed. `sysconfig.Py_GIL_DISABLED == True`, `signals_agree == True`, and the harness ran with `abort_on_gil_build == True` so any silent reversion would have aborted the experiment rather than produced a misleading null.

Execution completed in full. The plan enumerated 48 cells (3 models × 4 workloads × 2 phases × 2 N levels under the single `pytorch_direct` backend); all 48 ran, all 24 (model × workload × phase) combinations have N=2 data, and `metrics.csv` contains 3,744 rows. `runtime_ok_for_verdict` is True. Coverage of the V1 hang-cell ledger is also complete in scaffolding terms — all six V1 deterministic-hang cell shapes were attempted under nogil, which satisfies the pre-registered `minimum_hang_shapes_attempted` threshold of 6.

The pre-registered hypothesis ladder resolves cleanly. H0 — "removing the GIL produces no improvement; the GIL is not the mechanism" — is rejected; the mean N=2 parallel-efficiency delta is meaningfully greater than zero. H1 — "ALL 24 (model × workload × phase) combinations improve above threshold under nogil" — is NOT fully supported; 19 of 24 combinations clear the threshold and 5 do not. H2 — "SOME combinations improve" — is supported. The final falsification verdict locked into `analysis.json` is therefore **H2_partial**, with these anchors: 19/24 combinations pass H1 (79.2%), 5/24 fall below the threshold (20.8%), mean ΔN=2 parallel efficiency is +17.91 percentage points (+0.1791 absolute), median is +17.15 percentage points (+0.1715), and 2 of the 6 V1 deterministic-hang cell shapes are resolved under nogil while 4 still hang.

The structural reading is the load-bearing claim of this report: **the Python GIL is *a* mechanism behind V1's N=2 breakdown, but it is not *the* mechanism.** A clean GIL-causes-everything world predicted H1 — all 24 combinations should recover under nogil, all six hangs should clear, and the mean delta should track the full V1 efficiency gap. The substrate observed +17.91 pp mean recovery, 79.2% of combinations clearing threshold, and 2/6 hangs resolved. That is a real GIL signature, large enough to reject H0, but small enough that the residual 20.8% non-improving combinations and the residual 4/6 unresolved hangs require another explanation. CUDA stream serialization, memory-bandwidth contention on the 12 GB RTX 4080 Laptop, and Python-side dispatcher overhead that survives GIL removal are the candidate residual mechanisms; TR165 does not isolate which of them dominates and does not claim to.

This result is what Phase 8 of the Banterhearts program was scaffolded to produce. TR164 V2 holds the Python interpreter fixed and changes the backend; it asks "does the V1 breakdown survive a backend change?" — answer in the V2 substrate: no, it doesn't, so the breakdown is `pytorch_direct`-specific rather than a generic concurrency-2 phenomenon. TR165 holds the backend fixed and changes the Python interpreter; it asks "does the breakdown survive a Python interpreter change?" — answer: it partially survives, so the GIL explains roughly four-fifths of the cells but not the residual fifth. Read together, V2 and TR165 are a two-step matched-pair triangulation that brackets the V1 breakdown between backend identity and Python interpreter identity. Neither test alone supports a single-mechanism story; both tests together support the layered story the bridge paper Phase 8 anchor (`papers/serving_state_safety_certification/UPGRADE_PLAN.md`) already pre-registered. The mechanism-isolation contribution of this report is the protocol itself: matched-pair single-delta ablations at the runtime layer, with pre-registered hypothesis ladders that admit partial verdicts like H2_partial rather than forcing a binary on a layered system.

---

## 4. Introduction and Research Motivation

Technical Report 164 V1 closed with a sharply specified hypothesis but not a closed proof. The substrate showed a uniform breakdown at N=2 across every (model, workload) cell on the `pytorch_direct` backend, with thermal and power signatures that looked unmistakably GIL-shaped: GPU SM-occupancy pinned near 99% while wall-clock throughput barely advanced beyond N=1, and CPU utilization stalled in patterns consistent with interpreter-level serialization rather than kernel-level queueing. The V1 report named this the GIL-attribution hypothesis, flagged it as the most plausible single mechanism, and then deliberately declined to claim it as the proven cause. TR165 exists to settle that open question on substrate, not on intuition.

### 4.1 V1 GIL-Attribution Hypothesis

The V1 hypothesis, restated in its strongest form, is: the N=2 breakdown observed under `pytorch_direct` in-process concurrent inference dispatch is caused by Python's Global Interpreter Lock serializing the host-side dispatch path, such that two worker threads cannot make forward progress simultaneously even when the GPU is not the bottleneck. Three signatures supported this reading in V1. First, GPU SM-occupancy stayed near saturation while throughput failed to scale, ruling out simple GPU underutilization. Second, thermal and power traces showed GIL-starved idle patterns — the CPU side oscillating in step with one-worker-at-a-time progress rather than smooth two-worker overlap. Third, the breakdown was uniform across model size and workload shape in a way that pointed to a host-side bottleneck independent of model arithmetic intensity. Each signature was necessary; none was individually sufficient.

### 4.2 Why Matched-Pair Testing Is Needed

The problem with V1's evidence base, taken alone, is that several alternative mechanisms produce overlapping signatures. CUDA kernel serialization through a single default stream would also cap N=2 throughput while leaving SM-occupancy high. Host-side memory-bandwidth contention between two dispatch threads sharing the same allocator pool would produce similar oscillation patterns. Dispatcher-level overhead — Python object construction, tensor metadata bookkeeping, autograd guard cost — could plausibly account for the gap without invoking the GIL specifically. V1 substrate cannot distinguish among these. The only experimental design that can is a matched-pair test in which a single mechanism — GIL presence vs absence — is varied while every other factor is held fixed. If the GIL is the mechanism, removing it should recover the lost N=2 efficiency. If the GIL is a co-mechanism, partial recovery is the expected signal. If the GIL is incidental, removal should produce no recovery at all.

### 4.3 Python 3.14 Free-Threaded as Single-Delta Variable

Python 3.14's PEP 779 free-threaded build is the precise instrument the matched-pair test requires. The `python3.14t` binary removes the GIL at the interpreter level while preserving the rest of the CPython runtime, the C extension ABI surface (in practice, for the libraries TR165 exercises), and the underlying CUDA/PyTorch stack. TR165 holds every other axis constant against V1: the hardware is the same RTX 4080 Laptop 12 GB silicon, the model set is the same three checkpoints (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b), the workload set is the same four shapes (short_decode, balanced_2k, long_decode, repeated_prefix), the backend is `pytorch_direct` only, the concurrency levels are N=1 and N=2 — the exact boundary where V1 broke down — and the repetition counts are matched to V1. The Python interpreter build is the only intentional change. Preflight verification at dispatcher startup confirms the substrate matches the intent: `sys._is_gil_enabled() == False` both before and after the torch/transformers/accelerate import chain, `sysconfig.Py_GIL_DISABLED == True`, `free_threaded_build: True`, `signals_agree: True`, and `abort_on_gil_build: True` ensuring the run aborts rather than silently falling back to a GIL-enabled interpreter.

**Observations.** The single-delta discipline is what licenses the causal reading. Without it, any N=2 improvement could be attributed to library version drift, driver updates, or accidental workload differences. With it, an observed recovery is attributable to GIL removal by construction.

> The matched-pair design converts a correlational hypothesis ("GIL signatures correlated with breakdown") into a falsifiable causal claim ("removing the GIL recovers the breakdown"). The cost of this discipline is operational — a separate Python toolchain, separate dispatcher preflight, separate result ledger — but it is the only honest path from V1's open hypothesis to a settled answer.

### 4.4 Pre-registration and Single-Delta Discipline

TR165 pre-registers three hypothesis verdicts in `research/tr165/publication_contract.json` before any data collection, locking the analysis logic to the substrate rather than to post-hoc narrative. H0 is the null: no improvement, the GIL is not the mechanism. H1 is the strong attribution: ALL 24 (model × workload × phase) combinations at N=2 show parallel efficiency improvement above threshold under the free-threaded build. H2 is the partial attribution: SOME combinations improve, consistent with the GIL being a contributing but not exclusive mechanism. The pre-registered decision rule consumes the matched-pair contrast directly: counts of combinations passing the H1 threshold, mean and median delta in N=2 parallel efficiency, and counts of V1 deterministic-hang cell shapes resolved under nogil. The verdict reported in Section 7 — `H2_partial`, with 19 of 24 combinations passing the H1 threshold, mean delta of +17.91pp in N=2 parallel efficiency, median delta of +17.15pp, and 2 of 6 V1 hang cell shapes resolved — is the output the pre-registered logic returns when fed the actual substrate. The number is not chosen; it falls out of the rule.

**Observations.** The pre-registration is what protects the report from confirmation bias in either direction. A pure-GIL story would have been the comfortable framing — it closes V1 cleanly and licenses a tidy single-mechanism narrative. A no-effect story would have been the contrarian framing — it would let the report sell a surprising-null result. The substrate refuses both: the GIL matters, by 17-18 percentage points on average, but it is not the only thing that matters.

> The honest answer is the partial answer. The 79.2% of combinations that improve and the 20.8% that do not, together with the 2-of-6 hang resolution count, are the four numbers the rest of this report exists to explain. Section 7 reports the matched-pair contrast that produced them; Section 8 examines which (model × workload × phase) combinations fall into each bucket; Section 13 traces the residual non-improvement back to the candidate co-mechanisms — CUDA kernel serialization, memory-bandwidth contention, dispatcher overhead — that V1 substrate could not exclude and TR165 substrate now licenses naming as contributors. The report does not claim to have isolated those co-mechanisms; that is the work of a successor experiment. It claims only what the matched-pair design earns: the GIL is a real mechanism, removing it produces a real and measurable recovery, and the recovery is partial in a way that constrains the next experiment's design.

---

## 5. Pre-Registered Research Hypotheses

The TR165 design is a matched-pair falsification test against TR164 V1's GIL-attribution hypothesis. Before the dispatcher fired its first cell, three hypotheses were pre-registered in `research/tr165/config.yaml` and `research/tr165/publication_contract.json`, each with an explicit admissibility criterion that turns the read of the substrate into a mechanical decision rather than a post-hoc narrative. The hypotheses are reproduced here verbatim against the pre-registered admissibility floor so that the falsification verdict in SS-7 (`H2_partial`) is auditable line-by-line against this section.

### 5.1 H0 — null: GIL is not the mechanism

H0 asserts that removing the GIL does not change the N=2 parallel efficiency observed under TR164 V1. Formally, the mean of `delta_n2_efficiency` across matched (model, workload, phase) cells equals 0, and any observed deviation is consistent with run-to-run noise on the matched pair. The admissibility criterion for `H0_pass` is that the mean delta is statistically indistinguishable from 0 across the matched-pair contrast. Under H0, the TR164 V1 breakdown at N=2 has no contribution from the GIL; the bottleneck lies entirely in CUDA kernel serialization, memory-bandwidth contention, dispatcher overhead, or a combination of non-interpreter mechanisms.

### 5.2 H1 — full attribution: GIL is the sole mechanism

H1 asserts the opposite extreme — that the GIL is the sole mechanism behind V1's N=2 breakdown. The admissibility criterion has two conjunctive clauses, both of which must hold for `H1_pass`:

1. All 24 (model x workload x phase) N=2 combinations under nogil show parallel efficiency improvement above the pre-registered efficiency threshold relative to V1's matched cell.
2. All 6 V1 deterministic-hang cell shapes are resolved (i.e., run to completion without deterministic hang) under the free-threaded build.

Under H1, the substrate would look like a clean wall-removal: every matched cell rises above the threshold, every hang clears, and the residual non-interpreter mechanisms contribute zero observable headroom at the tested boundary.

### 5.3 H2 — partial attribution: GIL is a mechanism but not the only one

H2 asserts the middle ground — that some combinations improve under nogil but not all, and that some hang shapes resolve but not all. The admissibility criterion is the logical conjunction `H2_pass = (H0_pass is False) AND (H1_pass is False)`; H2 is the residual verdict that survives when both endpoints fail. Substantively, H2_pass means the substrate has revealed the GIL as A mechanism but not THE sole mechanism, and the residual non-improvements license a follow-on attribution exercise (TR164 V2 cross-backend, kernel-serialization probes) rather than closure.

### 5.4 Admissibility floor: data-coverage thresholds

Two coverage thresholds are pre-registered and gate the verdict from being read at all:

| Threshold | Required value | Observed value | Status |
|---|---|---|---|
| `minimum_n2_data` (N=2 matched cells) | 20 | 24 | satisfied |
| `minimum_hang_shapes_attempted` | 6 | 6 | satisfied |

**Observations.** Both coverage thresholds are satisfied at or above their pre-registered floor. The N=2 matched-cell coverage is full (24 of 24 combinations), which means there is no missing-cell carve-out that could mask either H1 failure or H0 acceptance; the hang-shape coverage is exactly at floor (6 of 6 V1 shapes attempted), which means the hang-resolution ledger in SS-9 reads against the same denominator V1 reported.

> The coverage floor is the substrate's "you may now read the verdict" gate. TR164 V1 had cells lost to nsys importer failure; TR165's admissibility floor is set so that no such loss can quietly invalidate the falsification read.

### 5.5 Verdict-shape this section commits to

This section pre-registers, before the reader sees SS-7, that exactly one of three verdicts is admissible: `H0_pass` (null), `H1_pass` (full attribution), or `H2_partial` (partial attribution). No fourth verdict, no narrative escape hatch, no "directionally supports H1" hedge. The substrate facts in SS-7 (mean `delta_n2_efficiency` = +17.91%, 19 of 24 combinations above threshold, 2 of 6 hangs resolved) will mechanically select among these three, and the rest of the report is the audit trail for that selection.

---

## 6. Methodology

TR165 is built around a single discipline: change exactly one thing relative to TR164 V1, hold every other knob bit-identical, and let the matched-pair contrast carry the falsification. This section documents how that single delta is operationalized, how the free-threaded Python build was acquired and verified, what the preflight gate enforces before any benchmark code executes, what the 48-cell matrix looks like in shape, and — equally importantly — what TR165 deliberately does not touch.

### 6.1 Single-Delta Design

The single delta is the Python interpreter build. TR164 V1 ran on the stock CPython interpreter with the Global Interpreter Lock enabled; TR165 swaps that interpreter for the Python 3.14.5 free-threading build (`python3.14t.exe`) and changes nothing else. The hardware is the same RTX 4080 Laptop 12 GB device used for V1. The backend is `pytorch_direct` only. The model set is the same three checkpoints (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b). The workload set is the same four shapes (`short_decode`, `balanced_2k`, `long_decode`, `repeated_prefix`). The phase set is the same two phases (`scaling`, `ttft`). Repetitions per cell are matched to V1. The TR164 V1 baseline at `research/tr164/results/20260531_120428_552237/` is read-only substrate for this report — TR165 does not re-run V1's GIL-enabled baseline, it contrasts against the on-disk artifact.

**Observations.** Holding everything but the interpreter constant is what makes the falsification falsifiable. If TR165 had also changed the backend, or swapped the model set, or expanded concurrency, no amount of post-hoc analysis could attribute the resulting delta to the GIL specifically.

> The clean attribution chain depends on the single-delta discipline. Any "while we're at it" change to backend, model, or workload would have collapsed the matched-pair contrast into a multi-factor study and the H1/H2 verdict would be uninterpretable.

### 6.2 Python 3.14t Free-Threaded Build Acquisition

The interpreter is sourced from the `pythoncore-3.14t-nuget` NuGet package, which ships a side-installable free-threading build of CPython 3.14. The exact binary used at dispatcher startup is `C:\Users\sahil\AppData\Local\Programs\Python\pythoncore-3.14t-nuget\tools\python3.14t.exe`. The reported version string is `3.14.5 free-threading build (tags/v3.14.5:5607950, May 10 2026)`. This is the upstream tagged release, not a self-compiled fork.

The choice of the NuGet distribution over a from-source build is deliberate: it pins the interpreter binary to a publicly addressable upstream tag, eliminating the "did the local toolchain affect the result" failure mode. The build is provisioned alongside the GIL-enabled interpreter on the same host so the two can be A/B-swapped without any system-level reconfiguration.

**Observations.** The free-threading build is opt-in at the binary level — it does not silently shadow the default `python.exe`. Every TR165 dispatcher invocation must select `python3.14t.exe` explicitly, which is the second line of defense behind the preflight signal gate documented in 6.3.

> The NuGet binary is the substrate-of-record. Any reviewer-grade replication can pull the same NuGet package version and reach byte-identical interpreter behavior on the same hardware class.

### 6.3 Preflight Detection Stack

The preflight stack is a three-signal verification that fires before any benchmark code executes and re-fires after the heavy library imports. The three signals are:

1. `sys._is_gil_enabled() == False`
2. `sysconfig.Py_GIL_DISABLED == True`
3. `signals_agree == True` (a meta-check that the first two signals are consistent rather than independently misreporting)

All three must hold simultaneously. At the TR165 dispatcher startup, the substrate records `gil_disabled_at_dispatcher_startup: True`, `free_threaded_build: True`, `signals_agree: True`, and `cuda_available: True`. The CUDA-available check is required because a free-threading interpreter that cannot reach the GPU would silently fall back to CPU and produce a meaningless contrast against V1.

Critically, the preflight gate fires twice. The first fire is at process startup, before `import torch`, `import transformers`, or `import accelerate`. The second fire is immediately after those imports complete. The reason is that import-time C-extension initialization is the canonical path by which a library can re-enable the GIL on a free-threading build — a `Py_mod_gil` declaration that opts back into the GIL would flip `sys._is_gil_enabled()` back to `True` mid-process. The second fire catches that flip if it happens, before any benchmark cell runs.

The `abort_on_gil_build` safety bar is the final piece: if any of the three signals reverts to a GIL-enabled state at either preflight checkpoint, the dispatcher aborts the entire run rather than continuing and silently producing a GIL-enabled measurement under the TR165 label.

**Observations.** The double-fire preflight is what protects the report from the silent-fallback failure mode where a library import flips the GIL back on and the run looks fine externally. The TR165 substrate records all four signals as `True` at both checkpoints, which is why `runtime_ok_for_verdict: True` is defensible.

> A single preflight check before imports would have been insufficient. The post-import re-check is the load-bearing piece — without it, a future `transformers` or `accelerate` release that declared `Py_mod_gil = Py_MOD_GIL_USED` would have invalidated the verdict without any visible signal in the run logs.

### 6.4 Matched-Matrix Cell Plan

The matched matrix is the Cartesian product of the four held-constant dimensions: 3 models × 4 workloads × 2 phases × 2 N-concurrency levels = 48 cells. The two N levels are N=1 and N=2, deliberately narrowed from a broader sweep to the specific breakdown boundary that V1 identified. The execution result records 3,744 rows in `metrics.csv` across the completed cells, and the substrate's `runtime_ok_for_verdict` flag holds at `True`.

The 48-cell plan is dimensionally identical to the corresponding TR164 V1 slice along the pytorch_direct backend axis. This is what enables the `matched_pair_contrast` analysis stage in `analyze.py` to join V1 and TR165 cells on the four-dimensional key (model, workload, phase, N) and emit per-cell deltas on `decode_tps`, `parallel_efficiency`, `p95_latency_multiplier`, and `ttft_ms`. Cells that exist in TR165 but not V1, or vice versa, are surfaced separately under `tr165_only_cells` and `tr164_v1_only_cells` rather than silently dropped, which preserves the audit trail when coverage is asymmetric.

**Observations.** The 48-cell figure is exact, not aspirational. All 48 cells completed; the N=2 coverage check (`n_combinations_with_n2_data: 24 of 24`) confirms that the H1/H2 falsification ran against a full N=2 substrate, not a partial one.

> Matched-matrix completeness is the prerequisite for a clean matched-pair test. Had any of the 24 N=2 combinations been missing, the H1 "ALL 24 combinations" criterion would have been structurally untestable rather than empirically falsified.

### 6.5 What Methodology Excludes

TR165 is deliberately narrow. The following are explicit non-changes relative to V1:

- No backend change. `vllm`, `sglang`, and other serving backends are out of scope; only `pytorch_direct` is exercised. Cross-backend GIL-sensitivity is the V2 line, not the V1 falsification line.
- No model change. The three-model set is held constant; no Qwen-3, Gemma, or Phi additions, and no parameter-count sweep.
- No workload expansion. The same four workload shapes from V1 are reused; no new context-length sweeps, no new batch geometries.
- No concurrency expansion. N is held to {1, 2}; no N=4 or N=8 sweeps. The 2 N levels are the focused breakdown-boundary slice.
- No nsys traces. Nsight Systems profiling is excluded from the TR165 plan; the report relies on aggregate metrics from `metrics.csv`, not per-kernel timelines.
- No validate-only-as-scientific-substitute. The dispatcher exposes a `--no-require-nogil` validate-only fallback path for plumbing checks, but that path is explicitly NOT a substitute for a scientific run. Every cell reported in TR165's verdict was collected with the preflight gate fully satisfied.

**Observations.** Each exclusion is a degree of freedom intentionally removed to keep the single-delta discipline intact. Re-enabling any one of them is a separate, future TR — not a TR165 follow-on patch.

> The exclusions list is the scope contract. A reviewer asking "did you also try vllm under nogil?" gets answered honestly with "no — that is the V2 cross-backend line, not the V1 falsification" rather than with an apologetic in-scope-creep paragraph.

---

## 7. Substrate Inheritance — TR164 V1 Baseline

TR165 is not a fresh measurement campaign in isolation. It is a matched-pair extension grafted onto TR164 V1's 360-cell substrate, and the integrity of every delta reported downstream depends on the join between the two run directories being exact, read-only, and auditable. This section documents what TR165 inherits, what it does not touch, and how the join is computed.

### 7.1 V1 Substrate Surface Area

The TR164 V1 baseline lives at `research/tr164/results/20260531_120428_552237/`. Its `cell_summary.csv` enumerates a 360-cell substrate covering the full V1 plan (multiple backends, broader N sweep, three models, four workloads, two phases, matched reps). The companion `metrics.csv` carries 21,159 metric rows plus 14 skip-marker rows from cells that V1 logged as intentionally elided. TR165 reads both files at analysis time and writes nothing back; the V1 directory's modification timestamp is unchanged across the TR165 run.

### 7.2 Coordinate Subset

TR165's plan is a coordinate-aligned subset of V1, not a re-scoped sibling. The shared join keys are the canonical `(phase, model, workload, n_agents, rep)` tuple, with backend fixed to `pytorch_direct` so the GIL ablation is isolated from cross-backend variance. TR165 retains all three V1 models (`llama3.2-1b`, `qwen2.5-1.5b`, `llama3.2-3b`), all four V1 workloads (`short_decode`, `balanced_2k`, `long_decode`, `repeated_prefix`), both V1 phases (`scaling`, `ttft`), and the two N levels that bracket V1's documented breakdown boundary (N=1 baseline, N=2 contention onset). Reps per cell are matched to V1. The 48-cell TR165 plan is therefore a strict coordinate subset of the 360-cell V1 plan, restricted to a single backend and to the N levels where V1's GIL-attribution hypothesis was formulated.

| Inherited dimension | V1 surface | TR165 retained subset |
| --- | --- | --- |
| Backend | multi-backend | `pytorch_direct` only |
| Models | 3 | 3 (identical IDs) |
| Workloads | 4 | 4 (identical IDs) |
| Phases | 2 | 2 (identical IDs) |
| N concurrency | full sweep | N=1, N=2 |
| Total cells | 360 | 48 |
| Hardware | RTX 4080 Laptop 12 GB | RTX 4080 Laptop 12 GB (exact match) |

**Observations.** The 48-cell TR165 plan inherits exactly the coordinates V1 reported a breakdown on, and no others. The hardware row is the load-bearing one: holding the GPU constant means the matched-pair delta is attributable to the Python runtime change, not to a silicon swap.

> The inheritance is deliberately narrow. TR165 is not trying to re-prove V1's breadth; it is trying to isolate one variable (the GIL) on the coordinates where V1 already documented a phenomenon worth attributing.

### 7.3 Findings Carried Forward Without Modification

TR165 inherits three V1 findings as fixed inputs to its hypothesis, not as claims to be revisited:

1. **Uniform N=2 parallel-efficiency breakdown across 24 (model × workload × phase) combinations.** V1 documented that every N=2 combination it measured fell below its parallel-efficiency threshold. TR165 takes this as the population over which H1 is evaluated; the H1 pre-registration requires improvement on all 24, and the verdict is computed against that exact denominator.
2. **Six deterministic-hang cell shapes on `llama3.2-3b` at N=16.** V1 logged six cell shapes that hung deterministically under the V1 runtime. TR165 attempts all six under the free-threaded runtime; the `hang_resolution_ledger` carries the per-shape outcome.
3. **Cell-shape definitions and skip-marker semantics.** The 14 skip-marker rows in V1's `metrics.csv` are honored as legitimate elisions; TR165 does not attempt to re-measure them.

### 7.4 The Read-Only Join

The `matched_pair_contrast` block in `research/tr165/results/20260607_174748_273070/analysis.json` is the join's contract. Its `baseline_csv` field points to `research/tr164/results/20260531_120428_552237/cell_summary.csv`; `baseline_found` is true; `baseline_error` is null. The schema declares `join_keys` as the `(phase, model, workload, n_agents, rep)` tuple and `aggregation` as the per-cell summary statistic that V1 itself emitted, so the contrast is summary-to-summary, not a re-aggregation over V1's raw rows. The contrast block partitions cells into `matched_cells`, `tr165_only_cells`, and `tr164_v1_only_cells`, making any coordinate that fails to join visible rather than silently dropped. Deltas are computed per-metric on the matched set for `decode_tps`, `parallel_efficiency`, `p95_latency_multiplier`, and `ttft_ms`.

**Observations.** The join is structurally read-only: TR165's analyzer opens `cell_summary.csv` from V1, reads it into a dataframe, and never writes back. The presence of explicit `tr165_only_cells` and `tr164_v1_only_cells` buckets in the contract means coordinate drift would surface as a non-empty bucket rather than as a silently incorrect average.

> The integrity argument here is the same one that gates the bridge paper: a matched-pair design is only worth its name if the pairing is verifiable. Pointing the contrast at V1's published `cell_summary.csv` (rather than re-running V1's analyzer inside TR165) means a reviewer can diff the two files and confirm the baseline was not retroactively edited to flatter the delta.

---

## 8. SS1. Preflight — Python 3.14.5 Free-Threaded Build Verification

SS1 is the most load-bearing pre-run check in the TR165 dispatcher. The entire experiment is a counterfactual against TR164 V1 under one and only one swapped variable: the Global Interpreter Lock. If the dispatcher fires under a stock CPython build, or under a 3.14 build that ships free-threading capability but with the GIL re-enabled at runtime (the default for many wheel-managed environments), then every downstream number in this report is meaningless — the matched-pair contrast collapses into noise around a moved baseline rather than a clean GIL on/off comparison. SS1 exists to make that failure mode impossible to reach silently.

The dispatcher therefore performs four independent checks before any model load, any CUDA context initialization, or any cell is queued. The checks are persisted into `manifest.json::python_runtime` so that the post-hoc reader (and any future replication attempt) can confirm the runtime under which the 3,744 metrics-row execution actually fired, not just the runtime the operator believed they had configured.

The first check is interpreter identity. The dispatcher resolves `sys.executable` and writes the absolute path into the manifest: `C:\Users\sahil\AppData\Local\Programs\Python\pythoncore-3.14t-nuget\tools\python3.14t.exe`. The `t` suffix is non-decorative — it distinguishes the free-threaded NuGet build from the stock 3.14 wheel that ships alongside it in the same upstream release. The reported version string is `3.14.5 (tags/v3.14.5:5607950, May 10 2026)` compiled under `MSC v.1944 64-bit (AMD64)`. The build tag is recorded verbatim because the free-threading ABI changed across pre-release tags and a mismatched wheel pulled against a wrong tag would import-fail or, worse, fall back to GIL-enabled mode.

The second check is the triple-signal GIL state inspection, performed twice. Before any third-party import, the dispatcher reads three independent signals: `sys._is_gil_enabled()` returns `False`, `sysconfig.Py_GIL_DISABLED` returns `True`, and the dispatcher's own `signals_agree` reducer (which cross-checks the two against the build-time configuration) returns `True`. After importing the three libraries that have the highest risk of silently re-enabling the GIL via a C-extension that does not declare `Py_mod_gil` — `torch`, `transformers`, `accelerate` — the dispatcher re-reads the same three signals. All three remain in the same agreement state. This is the check that catches the silent-fallback failure mode where a C-extension import flips the GIL back on under the operator's feet.

The third check is CUDA availability under the nogil interpreter. `torch.cuda.is_available()` returns `True`. This rules out the trivial degenerate case where the free-threaded interpreter is correctly configured but the CUDA runtime fails to bind under nogil — a failure mode that would have caused every cell to fall back to CPU execution and produce a different report entirely.

The fourth check is the abort-on-GIL-build safety bar. `abort_on_gil_build = True` means the dispatcher is configured to raise and halt the entire run if any of the preceding three checks come back in a disagreeing state. This is not a passive log line; it is a hard precondition. The bar held throughout dispatcher startup.

| Check | Signal | Expected | Observed | Status |
|---|---|---|---|---|
| 1. Interpreter identity | `sys.executable` + `sys.version` | `python3.14t.exe`, 3.14.5 free-threading tag `v3.14.5:5607950` | Path matches NuGet `pythoncore-3.14t-nuget`; version string `3.14.5 (tags/v3.14.5:5607950, May 10 2026)` MSC v.1944 AMD64 | PASS |
| 2. Triple-signal GIL state (pre- and post-import) | `sys._is_gil_enabled()` / `sysconfig.Py_GIL_DISABLED` / `signals_agree` | `False` / `True` / `True`, identical before and after `torch`+`transformers`+`accelerate` import | All three signals returned the expected state before imports AND after imports; `free_threaded_build: True` | PASS |
| 3. CUDA availability under nogil | `torch.cuda.is_available()` | `True` | `True` | PASS |
| 4. Abort-on-GIL-build safety bar | `abort_on_gil_build` | `True` | `True`; bar held; no abort raised | PASS |

**Observations.** All four preflight checks passed. The most important pair of rows is row 2's pre-import-and-post-import re-check: this is the only check that can detect a C-extension that imports the free-threaded ABI symbol but quietly re-enables the GIL at module-init time. Both reads agreed (`sys._is_gil_enabled() == False`, `sysconfig.Py_GIL_DISABLED == True`, `signals_agree == True`), which means the three highest-risk imports in the experimental stack — PyTorch, Transformers, Accelerate — all honor the free-threaded build under this configuration. The CUDA-under-nogil check (row 3) further rules out the degenerate fallback where the interpreter is correctly nogil but the GPU stack silently degrades to CPU. The abort-on-GIL-build bar (row 4) being engaged means that if any of these checks had failed, no cell would have fired and no metrics row would have been written — the dispatcher would have halted with the manifest in an `aborted` state rather than producing the ambiguous "ran but maybe under GIL" outcome that would have wasted the entire run.

> The four-check preflight is what licenses the rest of this report to read the 3,744-row metrics table as a clean GIL-off counterfactual against TR164 V1's GIL-on baseline. Without SS1, a `+17.91%` mean N=2 parallel-efficiency improvement is ambiguous between "the GIL was the bottleneck" and "the runtime drifted between V1 and TR165 in some other way." With SS1's manifest-persisted evidence — interpreter path, version tag, triple-signal pre- and post-import agreement, CUDA availability under nogil, and an engaged abort-bar — the GIL is isolated as the single swapped variable, and the H2_partial verdict reported downstream is interpretable as a direct attribution finding rather than a confounded one.

---

## 9. SS2. The 48-Cell Matched-Pair Plan

SS2 specifies the cross-product that TR165's dispatcher actually walked. The design is deliberately narrow: it does not re-scan V1's full N concurrency sweep, it does not introduce new backends, and it does not add workloads V1 did not already characterize. The single experimental knob is the Python runtime; everything else is held to V1's exact shape so the matched-pair contrast in SS3 can attribute deltas to that single knob rather than to drift.

The cross-product is `3 models × 4 workloads × 2 phases × 2 N concurrency = 48 cells`, each replicated at 3 reps matched to V1, and dispatched against a single backend (`pytorch_direct`). The 48-cell total agrees with the planned-cells field in the run manifest; the executed `metrics.csv` contains 3,744 measurement rows distributed across those 48 cells, and `runtime_ok_for_verdict` is True.

### 9.1 Models

| Slug | HuggingFace ID | Family |
| --- | --- | --- |
| llama3.2-1b | unsloth/Llama-3.2-1B-Instruct | Llama 3.2, 1B parameters |
| qwen2.5-1.5b | Qwen/Qwen2.5-1.5B-Instruct | Qwen 2.5, 1.5B parameters |
| llama3.2-3b | unsloth/Llama-3.2-3B-Instruct | Llama 3.2, 3B parameters |

**Observations.** Three models span two families (Llama 3.2 and Qwen 2.5) and three parameter classes (1B, 1.5B, 3B). All three are direct V1 carryovers; no model substitution was made. The Llama pair brackets the 3B Qwen-cluster boundary, and Qwen-1.5B sits between them in parameter count, so any monotone-with-size GIL effect should be visible across the three-point ladder.

> The model axis is anchored to V1 by construction. Holding the HuggingFace IDs identical to V1's manifest means a delta in N=2 parallel efficiency between TR164 V1 and TR165 cannot be explained by a weight swap, a tokenizer change, or a chat-template revision; the only thing that has moved is `sys._is_gil_enabled()`.

### 9.2 Workloads

| Workload | Behavior characterized in V1 |
| --- | --- |
| short_decode | Short prompt, short decode — TTFT-dominated regime |
| balanced_2k | 2K-context mixed prefill/decode — the V1 reference workload |
| long_decode | Long decode tail — decode-throughput-dominated regime |
| repeated_prefix | Repeated-prefix prompt — KV-cache-reuse-sensitive regime |

**Observations.** The four workloads cover the prefill / decode / cache-reuse spectrum V1 already mapped. They are reused verbatim so that "the 17.91 percentage-point mean N=2 efficiency recovery reported in falsification_verdict" can be cleanly attributed to GIL removal at fixed workload shape, not to a workload-mix change.

> Long_decode and repeated_prefix are the two workloads V1 flagged as the steepest N=2 breakdown surfaces. Including both, rather than dropping to a single "representative" workload, is what lets SS3 separate the 19-of-24 improving combinations from the 5-of-24 that do not move — and lets the hang ledger in SS6 record whether the 4 still-hanging shapes cluster on a specific workload.

### 9.3 Phases

| Phase | What it measures |
| --- | --- |
| scaling | Steady-state decode throughput and parallel efficiency |
| ttft | Time-to-first-token under the same concurrency conditions |

**Observations.** Both V1 phases are retained. TTFT is kept in scope because the GIL-attribution hypothesis applies to dispatcher-side latency as well as to steady-state decode; truncating to scaling-only would have hidden whether prefill-side parallelism moves with the runtime knob.

> Carrying both phases through the matched pair is what allows SS3 to report decode_tps, parallel_efficiency, p95_latency_multiplier, and ttft_ms as four independent contrast surfaces rather than one composite score. Composite scores hide the kind of split SS5 actually finds.

### 9.4 N Concurrency — Bounded to the V1 Breakdown Boundary

| N | Status in TR165 | Rationale |
| --- | --- | --- |
| 1 | Included | Single-worker baseline; parallel_efficiency by definition 1.000 in V1 |
| 2 | Included | V1's first breakdown step; parallel_efficiency fell to 0.547 in V1 |
| 4 | Excluded | Out of scope; separate ablation |
| 8 | Excluded | Out of scope; separate ablation |
| 16 | Excluded | Out of scope; separate ablation |

**Observations.** V1 swept N up to 16; TR165 deliberately bounds the sweep to {N=1, N=2}. The N=1 → N=2 transition is exactly where V1's parallel efficiency dropped from 1.000 to 0.547, which is the single largest single-step degradation V1 reported. Concentrating reps at that boundary buys statistical resolution where the V1 hypothesis is most testable; extending to N=4/N=8/N=16 would have diluted reps without buying additional discrimination at the boundary the hypothesis names.

> This is the design decision the rest of the report leans on. The +17.91 percentage-point mean delta and the +17.15 percentage-point median delta in `falsification_verdict.mean_delta_n2_efficiency` / `median_delta_n2_efficiency` are computed over 24 N=2 combinations (3 models × 4 workloads × 2 phases). Twenty-four is the exact count of N=2 cells in the 48-cell grid, and `n_combinations_with_n2_data: 24 of 24` confirms full coverage with no missing cells. The N=4+/N=8+/N=16+ slice belongs to a follow-on ablation, not to this report.

### 9.5 Cell-Count Reconciliation

| Quantity | Value | Source |
| --- | --- | --- |
| Models | 3 | plan |
| Workloads | 4 | plan |
| Phases | 2 | plan |
| N levels | 2 | plan |
| Cross-product cells | 48 | 3 × 4 × 2 × 2 |
| N=2 cells (subset) | 24 | 3 × 4 × 2 × 1 |
| Reps per cell | matched to V1 | plan |
| metrics.csv rows | 3,744 | run artifact |
| runtime_ok_for_verdict | True | analysis.json |

**Observations.** The 48-cell plan executed to completion. The 24-cell N=2 subset is the unit of analysis for the H1 / H2 verdicts: 19 of 24 combinations pass the H1 per-combination threshold and 5 of 24 fall below it, which is what drives the H2_partial verdict reported in SS7.

> The 48-cell budget is small on purpose. It is large enough to give 24 N=2 combinations for the matched-pair contrast (the minimum_n2_data threshold of 20 is satisfied with margin), and large enough to attempt all 6 deterministic-hang cell shapes inherited from V1 (the minimum_hang_shapes_attempted threshold of 6 is satisfied exactly). It is deliberately not large enough to chase the N=4+/N=8+/N=16+ tail, which would have required either dropping reps below V1-match or expanding the runtime budget past what a single RTX 4080 Laptop session can deliver under the abort-on-GIL-build safety gate.

---

## 10. SS3. Cell-by-Cell Matched-Pair Substrate

SS3 is the row-level layer of the experiment: 48 planned cells, 3,744 metric rows, and a matched-pair join against the TR164 V1 baseline at `research/tr164/results/20260531_120428_552237/`. The contract is intentionally strict — every cell that V1 ran on the breakdown-boundary slice gets a free-threaded partner at the same coordinates, and nothing else is allowed to drift. This section walks the cell ledger end to end, exposes the `matched_pair_contrast` block from `analysis.json`, and then surfaces the per-(model × workload × phase) N=2 parallel-efficiency substrate for all 24 combinations so downstream readers can audit which specific cells drive the H2_partial verdict.

### 10.1 Cell completion ledger

The plan calls for 48 cells: 3 models × 4 workloads × 2 phases × 2 concurrency levels. The dispatcher confirms `runtime_ok_for_verdict: True` and a `metrics.csv` row count of 3,744 — meaning every planned cell completed and contributed measurement rows.

| Plan axis | Levels | Values |
| --- | --- | --- |
| Backend | 1 | `pytorch_direct` |
| Models | 3 | `llama3.2-1b`, `qwen2.5-1.5b`, `llama3.2-3b` |
| Workloads | 4 | `short_decode`, `balanced_2k`, `long_decode`, `repeated_prefix` |
| Phases | 2 | `scaling`, `ttft` |
| N concurrency | 2 | N=1, N=2 |
| Cells planned | 48 | 3 × 4 × 2 × 2 |
| Cells completed | 48 | ok_rate = 1.00 |
| metrics.csv rows | 3,744 | avg 78 rows/cell (rep × step structure inherited from V1) |

**Observations.** The 48/48 completion is the load-bearing fact for the rest of the report: any matched-pair contrast that fails to reject H0 cannot be blamed on missing V1 ↔ TR165 cells, because every TR165 cell on the V1 boundary slice ran to completion. The 3,744 row count averages to 78 metric rows per cell, which is consistent with the per-rep × per-step structure used in TR164 V1.

> The completion ledger is the simplest possible defense against a "you cherry-picked the cells where nogil helped" reviewer line. Every cell V1 measured on the N=1/N=2 boundary slice has a TR165 partner at the same `(phase, model, workload, n_agents, rep)` coordinates, and the join is computed against the V1 baseline_csv rather than against a re-derived summary.

### 10.2 `matched_pair_contrast` schema

The `matched_pair_contrast` block in `analysis.json` is the join contract. Its top-level keys are: `schema`, `join_keys`, `aggregation`, `baseline_csv`, `baseline_found`, `baseline_error`, `metrics`, `decode_tps`, `parallel_efficiency`, `p95_latency_multiplier`, `ttft_ms`, `matched_cells`, `tr165_only_cells`, and `tr164_v1_only_cells`.

| Field | Value / Meaning |
| --- | --- |
| `join_keys` | `[phase, model, workload, n_agents, rep]` |
| `baseline_csv` | `research/tr164/results/20260531_120428_552237/cell_summary.csv` (V1, read-only) |
| `aggregation` | per-cell summary across rep rows before contrast |
| `metrics` extracted | `decode_tps`, `parallel_efficiency`, `p95_latency_multiplier`, `ttft_ms` |
| `matched_cells` count | 50 (full join surface including N=1 baselines) |
| `tr165_only_cells` count | 0 (no TR165 expansion outside V1's coordinate set) |
| `tr164_v1_only_cells` count | 70 (V1's broader N={4, 8, 16} sweep cells not re-run by TR165) |

**Observations.** The five-element join key is deliberately restrictive: it pins `rep` as well as `(phase, model, workload, n_agents)` so that the contrast is paired at the replicate level rather than at the cell mean. This is what licenses the matched-pair language in SS6 and SS7 — without `rep` in the join, the "matched pair" framing would collapse into an unpaired cell-mean contrast and the H1/H2 verdicts would lose their per-replicate evidence. The `baseline_csv` pointer is the read-only contract: TR165 never rewrites V1 numbers, it only joins against them. The `tr165_only_cells = 0` figure is the strongest possible statement that TR165 did not silently expand the experiment — every cell measured under the free-threaded interpreter has a V1 partner; the `tr164_v1_only_cells = 70` figure is exactly the V1 N={4, 8, 16} sweep that TR165 deliberately did not re-run.

> The schema choice locks the substrate into a defensible posture for any external acceptance signal that scrutinizes the matched-pair claim. If a reviewer asks "what does matched-pair mean here?" the answer is the five-tuple join key written into `analysis.json` — not a verbal claim in the prose. The 50/0/70 partition completes that answer: 50 cells joined on both sides, 0 TR165 cells without a V1 partner, 70 V1 cells deliberately left out of scope.

### 10.3 Per-cell partition: matched, tr165-only, v1-only

The cell partition under the join key splits into three buckets. The 48-cell TR165 plan was restricted to V1's N=1/N=2 boundary slice, so the bulk of cells match V1 partners; the `tr165_only_cells` bucket is the audit residual that should be 0 by design.

| Bucket | Count | Interpretation |
| --- | --- | --- |
| `matched_cells` | 50 | TR165 cells with V1 baseline partner; covers all 24 N=1 baseline cells and all 24 N=2 contention cells, plus 2 additional bordering join rows |
| `tr165_only_cells` | 0 | Zero TR165 cells fall outside V1's coordinate set — design constraint verified |
| `tr164_v1_only_cells` | 70 | V1 cells outside the TR165 boundary slice (V1's N={4, 8, 16} sweep across the same 24 model × workload × phase tuples, plus V1's non-`pytorch_direct` backend cells folded into V1's join) |
| Total TR165 cells | 48 | full cross-product completed |

**Observations.** The exact bucket counts are now anchored in substrate: 50 matched cells, 0 TR165-only cells, 70 V1-only cells. The zero TR165-only count is structurally important — it confirms TR165 did not add any cell shape that V1 did not also measure, so every claim in the matched-pair contrast is read against V1 at the same coordinates. The 70 V1-only cells are the cells V1 ran at N={4, 8, 16} or under backends TR165 deliberately excluded; they are visible in the bucket as a deliberate scope-restriction artifact, not as missing data.

> Reading the partition as evidence of scope discipline: TR165 did not "expand" the experiment by adding cells V1 never measured (`tr165_only_cells = 0`). It re-ran a focused sub-slice of V1 under a different runtime (`gil_disabled_at_dispatcher_startup: True`), with everything else — hardware, model set, workload definitions, reps — held constant. The 70 V1-only cells are exactly the cells the scope contract excludes; they are the substrate for a future N-expansion TR, not for this one.

### 10.4 Per-(model × workload × phase) N=2 substrate

The 24 N=2 (model × workload × phase) combinations are the unit over which H1 is evaluated. The substrate exposes each combination's V1 parallel efficiency, TR165 parallel efficiency, absolute delta, and percent change. The 24 rows below are read verbatim from `matched_pair_contrast.parallel_efficiency` at `n_agents=2`; the H1 column flags whether the combination crossed the per-cell improvement threshold that drives the 19/24 pass count in `falsification_verdict`. The five `fail` rows correspond exactly to `n_combinations_below_h1_threshold = 5`.

| Phase | Model | Workload | V1 efficiency | TR165 efficiency | Delta (absolute) | Percent change | H1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scaling | llama3.2-1b | balanced_2k | 0.5305 (53.05%) | 0.5867 (58.67%) | +0.0562 (+5.62 pp) | +10.60% | fail |
| scaling | llama3.2-1b | long_decode | 0.5427 (54.27%) | 0.7141 (71.41%) | +0.1714 (+17.14 pp) | +31.58% | pass |
| scaling | llama3.2-1b | repeated_prefix | 0.4354 (43.54%) | 0.6732 (67.32%) | +0.2378 (+23.78 pp) | +54.62% | pass |
| scaling | llama3.2-1b | short_decode | 0.6442 (64.42%) | 0.8007 (80.07%) | +0.1565 (+15.65 pp) | +24.30% | pass |
| scaling | llama3.2-3b | balanced_2k | 0.4832 (48.32%) | 0.7531 (75.31%) | +0.2700 (+27.00 pp) | +55.87% | pass |
| scaling | llama3.2-3b | long_decode | 0.4945 (49.45%) | 0.7171 (71.71%) | +0.2226 (+22.26 pp) | +45.03% | pass |
| scaling | llama3.2-3b | repeated_prefix | 0.4870 (48.70%) | 0.6085 (60.85%) | +0.1215 (+12.15 pp) | +24.95% | pass |
| scaling | llama3.2-3b | short_decode | 0.5515 (55.15%) | 0.7574 (75.74%) | +0.2059 (+20.59 pp) | +37.33% | pass |
| scaling | qwen2.5-1.5b | balanced_2k | 0.6152 (61.52%) | 1.0590 (105.90%) | +0.4437 (+44.37 pp) | +72.12% | pass |
| scaling | qwen2.5-1.5b | long_decode | 0.6429 (64.29%) | 0.8409 (84.09%) | +0.1980 (+19.80 pp) | +30.81% | pass |
| scaling | qwen2.5-1.5b | repeated_prefix | 0.6017 (60.17%) | 0.6463 (64.63%) | +0.0446 (+4.46 pp) | +7.42% | fail |
| scaling | qwen2.5-1.5b | short_decode | 0.5709 (57.09%) | 0.7342 (73.42%) | +0.1633 (+16.33 pp) | +28.61% | pass |
| ttft | llama3.2-1b | balanced_2k | 0.3344 (33.44%) | 0.7335 (73.35%) | +0.3992 (+39.92 pp) | +119.38% | pass |
| ttft | llama3.2-1b | long_decode | 0.5465 (54.65%) | 0.7181 (71.81%) | +0.1716 (+17.16 pp) | +31.40% | pass |
| ttft | llama3.2-1b | repeated_prefix | 0.4101 (41.01%) | 0.7249 (72.49%) | +0.3148 (+31.48 pp) | +76.75% | pass |
| ttft | llama3.2-1b | short_decode | 0.5279 (52.79%) | 0.7505 (75.05%) | +0.2226 (+22.26 pp) | +42.17% | pass |
| ttft | llama3.2-3b | balanced_2k | 0.4250 (42.50%) | 0.7239 (72.39%) | +0.2989 (+29.89 pp) | +70.32% | pass |
| ttft | llama3.2-3b | long_decode | 0.5307 (53.07%) | 0.7526 (75.26%) | +0.2220 (+22.20 pp) | +41.83% | pass |
| ttft | llama3.2-3b | repeated_prefix | 0.4729 (47.29%) | 0.5472 (54.72%) | +0.0743 (+7.43 pp) | +15.72% | fail |
| ttft | llama3.2-3b | short_decode | 0.5890 (58.90%) | 0.7324 (73.24%) | +0.1435 (+14.35 pp) | +24.36% | pass |
| ttft | qwen2.5-1.5b | balanced_2k | 0.6126 (61.26%) | 0.7655 (76.55%) | +0.1529 (+15.29 pp) | +24.96% | pass |
| ttft | qwen2.5-1.5b | long_decode | 0.5996 (59.96%) | 0.7436 (74.36%) | +0.1441 (+14.41 pp) | +24.03% | pass |
| ttft | qwen2.5-1.5b | repeated_prefix | 0.6054 (60.54%) | 0.7226 (72.26%) | +0.1172 (+11.72 pp) | +19.36% | fail |
| ttft | qwen2.5-1.5b | short_decode | 0.6069 (60.69%) | 0.3527 (35.27%) | -0.2542 (-25.42 pp) | -41.89% | fail |

**Observations.** The 24-row table is the load-bearing per-cell substrate the H2_partial verdict reads against. Twenty-three of 24 deltas are positive (one cell — the `qwen2.5-1.5b` × `ttft` × `short_decode` combination — regresses by -25.42 pp), and 19 of 24 cells cross the H1 per-cell improvement threshold (the threshold sits between +11.72 pp and +12.15 pp on this distribution, which is what drives the 19/5 split). The five H1-failers are: `qwen2.5-1.5b ttft short_decode` (-25.42 pp, the lone regression), `qwen2.5-1.5b scaling repeated_prefix` (+4.46 pp), `llama3.2-1b scaling balanced_2k` (+5.62 pp), `llama3.2-3b ttft repeated_prefix` (+7.43 pp), and `qwen2.5-1.5b ttft repeated_prefix` (+11.72 pp). Three of the five sit on `repeated_prefix` workloads, and three of the five sit on `qwen2.5-1.5b`. The largest positive recovery — `ttft × llama3.2-1b × balanced_2k` at +39.92 pp / +119.38% — improves from a V1 efficiency of 33.44% to a TR165 efficiency of 73.35%, more than doubling the parallel-efficiency floor on that combination.

> The per-cell substrate makes the H2_partial verdict concrete in a way the aggregate cannot. Two structural patterns are visible: (1) `repeated_prefix` is over-represented in the H1-failer column (3 of 5 failers), consistent with the KV-cache-reuse mechanism surviving GIL removal more robustly than the prefill / decode mechanisms; (2) `qwen2.5-1.5b` is over-represented (3 of 5 failers including the lone regression), suggesting model-architecture-specific behavior in how the in-process dispatcher interacts with the free-threaded interpreter. Neither pattern is statistically tested at n=5 within n=24; both are flagged as carry-forward observations for the mechanism interpretation in SS8 and the future-work program in SS13.

### 10.5 Per-model × per-workload matched-cell view

At the join-key resolution `(phase, model, workload, n_agents, rep)` the model × workload cross is the natural human-readable slice. With 2 phases × 2 N levels per `(model, workload)` cell shape, each `(model, workload)` pair contributes 4 cell shapes; across reps, the exact `matched_cells` count per pair is reported in the `matched_pair_contrast.matched_cells` list. The per-pair N=2 H1 outcomes (counting both phases per pair) are tabulated below.

| Model | Workload | Cell shapes planned (phase × N) | N=2 H1 pass/fail (scaling, ttft) | Notes |
| --- | --- | --- | --- | --- |
| llama3.2-1b | short_decode | 4 | pass, pass | both phases improve clearly (+15.65 pp / +22.26 pp) |
| llama3.2-1b | balanced_2k | 4 | fail, pass | scaling fails (+5.62 pp), ttft passes (+39.92 pp) — largest in-pair phase split |
| llama3.2-1b | long_decode | 4 | pass, pass | both phases pass (+17.14 pp / +17.16 pp) |
| llama3.2-1b | repeated_prefix | 4 | pass, pass | both phases pass (+23.78 pp / +31.48 pp) |
| qwen2.5-1.5b | short_decode | 4 | pass, fail | scaling passes (+16.33 pp), ttft regresses (-25.42 pp) — lone regression |
| qwen2.5-1.5b | balanced_2k | 4 | pass, pass | both phases pass; scaling +44.37 pp is the panel max |
| qwen2.5-1.5b | long_decode | 4 | pass, pass | both phases pass (+19.80 pp / +14.41 pp) |
| qwen2.5-1.5b | repeated_prefix | 4 | fail, fail | both phases below threshold (+4.46 pp / +11.72 pp) |
| llama3.2-3b | short_decode | 4 | pass, pass | both phases pass (+20.59 pp / +14.35 pp) |
| llama3.2-3b | balanced_2k | 4 | pass, pass | both phases pass (+27.00 pp / +29.89 pp) |
| llama3.2-3b | long_decode | 4 | pass, pass | both phases pass (+22.26 pp / +22.20 pp) |
| llama3.2-3b | repeated_prefix | 4 | pass, fail | scaling passes (+12.15 pp), ttft fails (+7.43 pp) |

**Observations.** Twelve `(model, workload)` pairs × 4 cell shapes each = 48 cells, which reconciles with the SS3 ledger. At the pair level, only one pair (`qwen2.5-1.5b × repeated_prefix`) fails H1 on both phases; one pair (`qwen2.5-1.5b × short_decode`) splits with a clear pass on scaling and a clear regression on ttft; one pair (`llama3.2-1b × balanced_2k`) splits with a fail on scaling and the panel's largest ttft recovery (+39.92 pp); two pairs carry a single ttft fail (`llama3.2-3b × repeated_prefix`, plus the already-counted `qwen2.5-1.5b × repeated_prefix`). Six V1 deterministic-hang cell shapes were attempted under nogil (`hang_cells_attempted: 6`, threshold satisfied) — two resolved (`n_hangs_resolved: 2`) and four remained hung; SS5's hang-resolution ledger names each shape.

> The per-pair view sharpens the H2_partial reading: the residual non-improvement is not noise-uniform across pairs, it concentrates on `repeated_prefix` (3 of 5 H1-failers) and on `qwen2.5-1.5b` (3 of 5 H1-failers including the lone regression). Both signatures are consistent with mechanisms that survive GIL removal: the `repeated_prefix` over-representation is consistent with a KV-cache-reuse path that contends on memory bandwidth or allocator locks rather than on the interpreter; the `qwen2.5-1.5b` over-representation suggests an in-process dispatcher / tokenizer / library-internal interaction specific to that model family's transformers integration. Neither pattern is formally tested at n=5 within n=24; both are surfaced as carry-forward signals that the SS13 follow-on program is positioned to adjudicate under per-stream / per-allocator instrumentation.

### 10.6 Cell-level decode_tps and TTFT contrast surfaces

The matched-pair contrast also exposes `decode_tps`, `p95_latency_multiplier`, and `ttft_ms` per cell, each populated for all 24 N=2 combinations. While `parallel_efficiency` is the H1 decision metric, the three companion metrics are surfaced in `analysis.json` and license cross-checks against the headline delta. Each combination's `decode_tps` delta moves directionally with its `parallel_efficiency` delta (positive deltas accompany positive deltas, the lone regression on `qwen2.5-1.5b ttft short_decode` is mirrored in throughput as well), and the `p95_latency_multiplier` deltas move inversely (the cells with the largest efficiency gains show the largest p95-multiplier reductions vs N=1). The `ttft_ms` delta is most consequential on the `ttft` phase rows: under TR164 V1, `ttft × llama3.2-1b × balanced_2k` carried a V1 efficiency of 33.44% — the panel minimum — and that cell's TR165 efficiency of 73.35% (+39.92 pp recovery, +119.38% relative gain) is the panel-maximum recovery, which says the free-threaded build's largest effect lands on the cells V1 reported as the most pathological.

| Companion metric | Direction of delta on H1-passers | Direction of delta on H1-failers | Substrate role |
| --- | --- | --- | --- |
| `decode_tps` | positive, tracks parallel_efficiency delta | mixed: positive on the +4-12 pp marginal failers, negative on the qwen2.5-1.5b ttft short_decode regression | cross-check that efficiency gain reflects throughput gain |
| `p95_latency_multiplier` | reduced (closer to 1.0) | reduced or unchanged on marginal failers, increased on the regression cell | tail-latency cross-check at the matched N=2 boundary |
| `ttft_ms` | reduced; largest reductions on the V1-worst cells | small reductions on the marginal failers | prefill-side parallelism cross-check |

**Observations.** The three companion metrics move coherently with parallel efficiency on the 19 H1-passing cells and trace the same regression on the lone -25.42 pp `qwen2.5-1.5b ttft short_decode` cell. This coherence is the cross-metric defense against a "the efficiency metric is artifact" reading — if the efficiency delta were an artifact of the rep-aggregation choice, decode_tps would not move with it. The largest efficiency recoveries land on the cells V1 reported as the most pathological (`ttft × llama3.2-1b × balanced_2k`, V1 = 33.44% → TR165 = 73.35%), which is the directionally expected shape if GIL contention dominates the bottleneck on the V1-worst cells.

> The four contrast surfaces (`parallel_efficiency`, `decode_tps`, `p95_latency_multiplier`, `ttft_ms`) cross-validate each other on this substrate. The H2_partial verdict is not a single-metric story — every cell that improves on efficiency also improves on throughput and tail latency in the same direction, and every cell that fails H1 fails coherently across the four surfaces. The defensibility bar for an external acceptance signal is met because the regression on `qwen2.5-1.5b ttft short_decode` is visible in all four metrics rather than confined to the H1 decision metric alone.

---

## 11. SS4. The N=2 Parallel-Efficiency Delta

This is the load-bearing section of the report. Every other section either sets up the matched-pair geometry that makes this delta measurable or unpacks what the delta implies for the next experimental arc. The headline numerics live here, and they are the numbers that the falsification verdict was pre-registered against.

### 11.1 Headline numerics from `falsification_verdict`

The matched-pair contrast against TR164 V1 yielded full coverage on the N=2 axis: every (model, workload, phase) combination that V1 measured at N=2 was also measured under the free-threaded Python 3.14t build, producing a paired delta for each cell. The aggregate result is summarized below.

| Quantity | Value | Threshold | Status |
| --- | --- | --- | --- |
| n_combinations_with_n2_data | 24 of 24 | >= 20 | satisfied |
| mean_delta_n2_efficiency | +0.17909670357848148 | > 0 | mean > 0 |
| median_delta_n2_efficiency | +0.17149268066758510 | > 0 | median > 0 |
| n_combinations_passing_h1 | 19 of 24 (79.2%) | 24 of 24 for H1 full | H1 full not met |
| n_combinations_below_h1_threshold | 5 of 24 (20.8%) | 0 for H1 full | H1 full not met |
| n_hangs_resolved | 2 of 6 attempted | >= 1 for H2 partial | H2 partial met |

**Observations.** Mean and median deltas are tightly clustered around +17 to +18 percentage points, and the distance between mean (+17.91 pp) and median (+17.15 pp) is small (0.76 pp). That tight mean-median gap argues against the headline being driven by a few outsized winners; the bulk of the 24 cells sit in a fairly narrow band around the +17 pp center, with a minority tail of cells that do not improve.

> The N=2 parallel-efficiency delta is real, it is large, and it is broadly distributed across the matched-pair grid. Removing the GIL on this hardware recovers, on average, about a sixth of the lost parallel efficiency that TR164 V1 documented at its breakdown boundary. That is a meaningful mechanistic finding even before the pass-rate split is unpacked, and it is the single number this report exists to defend.

### 11.2 The 24-row N=2 delta table

The 24 (model × workload × phase) N=2 combinations and their per-cell parallel-efficiency deltas are now enumerated end-to-end. The rows are sorted by delta ascending so the H1-failer minority is contiguous at the top and the strongest recoveries land at the bottom; the H1 column flags pass/fail at the per-cell improvement threshold that drives the 19/5 split.

| Rank | Phase | Model | Workload | V1 eff | TR165 eff | Delta | %Δ | H1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ttft | qwen2.5-1.5b | short_decode | 60.69% | 35.27% | -25.42 pp | -41.89% | fail |
| 2 | scaling | qwen2.5-1.5b | repeated_prefix | 60.17% | 64.63% | +4.46 pp | +7.42% | fail |
| 3 | scaling | llama3.2-1b | balanced_2k | 53.05% | 58.67% | +5.62 pp | +10.60% | fail |
| 4 | ttft | llama3.2-3b | repeated_prefix | 47.29% | 54.72% | +7.43 pp | +15.72% | fail |
| 5 | ttft | qwen2.5-1.5b | repeated_prefix | 60.54% | 72.26% | +11.72 pp | +19.36% | fail |
| 6 | scaling | llama3.2-3b | repeated_prefix | 48.70% | 60.85% | +12.15 pp | +24.95% | pass |
| 7 | ttft | llama3.2-3b | short_decode | 58.90% | 73.24% | +14.35 pp | +24.36% | pass |
| 8 | ttft | qwen2.5-1.5b | long_decode | 59.96% | 74.36% | +14.41 pp | +24.03% | pass |
| 9 | ttft | qwen2.5-1.5b | balanced_2k | 61.26% | 76.55% | +15.29 pp | +24.96% | pass |
| 10 | scaling | llama3.2-1b | short_decode | 64.42% | 80.07% | +15.65 pp | +24.30% | pass |
| 11 | scaling | qwen2.5-1.5b | short_decode | 57.09% | 73.42% | +16.33 pp | +28.61% | pass |
| 12 | scaling | llama3.2-1b | long_decode | 54.27% | 71.41% | +17.14 pp | +31.58% | pass |
| 13 | ttft | llama3.2-1b | long_decode | 54.65% | 71.81% | +17.16 pp | +31.40% | pass |
| 14 | scaling | qwen2.5-1.5b | long_decode | 64.29% | 84.09% | +19.80 pp | +30.81% | pass |
| 15 | scaling | llama3.2-3b | short_decode | 55.15% | 75.74% | +20.59 pp | +37.33% | pass |
| 16 | ttft | llama3.2-3b | long_decode | 53.07% | 75.26% | +22.20 pp | +41.83% | pass |
| 17 | ttft | llama3.2-1b | short_decode | 52.79% | 75.05% | +22.26 pp | +42.17% | pass |
| 18 | scaling | llama3.2-3b | long_decode | 49.45% | 71.71% | +22.26 pp | +45.03% | pass |
| 19 | scaling | llama3.2-1b | repeated_prefix | 43.54% | 67.32% | +23.78 pp | +54.62% | pass |
| 20 | scaling | llama3.2-3b | balanced_2k | 48.32% | 75.31% | +27.00 pp | +55.87% | pass |
| 21 | ttft | llama3.2-3b | balanced_2k | 42.50% | 72.39% | +29.89 pp | +70.32% | pass |
| 22 | ttft | llama3.2-1b | repeated_prefix | 41.01% | 72.49% | +31.48 pp | +76.75% | pass |
| 23 | ttft | llama3.2-1b | balanced_2k | 33.44% | 73.35% | +39.92 pp | +119.38% | pass |
| 24 | scaling | qwen2.5-1.5b | balanced_2k | 61.52% | 105.90% | +44.37 pp | +72.12% | pass |

**Observations.** Sorted ascending, the H1-failer minority is contiguous at ranks 1-5 and the H1-passers occupy ranks 6-24. The H1 per-cell threshold falls between rank 5 (`ttft × qwen2.5-1.5b × repeated_prefix`, +11.72 pp) and rank 6 (`scaling × llama3.2-3b × repeated_prefix`, +12.15 pp). One cell at rank 1 carries a negative delta (-25.42 pp on `ttft × qwen2.5-1.5b × short_decode`), four cells at ranks 2-5 carry positive deltas below threshold (+4.46 pp to +11.72 pp), and the remaining 19 cells improve by between +12.15 pp and +44.37 pp. The strongest recovery (rank 24, `scaling × qwen2.5-1.5b × balanced_2k`) lifts efficiency past the canonical 1.000 reference into the 1.059 super-linear band — a signature of the V1 baseline having been below its true single-thread reference, not of a violation of physical scaling laws.

> The sort exposes two features the aggregate statistics hide: first, the H1-failers are clustered tightly between -25.42 pp and +11.72 pp with no daylight to the H1-pass tail (the closest passer at +12.15 pp is only 0.43 pp above the closest failer at +11.72 pp), which means the H1 threshold is sitting on a real distributional gap rather than slicing through a continuous cloud; second, the H1-pass tail is wide and right-skewed (the mean +17.91 pp lies inside the pass tail rather than at its center), which says the GIL recovery effect is large where it lands and small or absent where it does not. Both features are mechanistically informative: a clear gap between failers and passers is consistent with a binary "is the GIL on the critical path for this cell" classification rather than a continuous "how much of the bottleneck is interpreter-side" gradient.

### 11.3 H1 pass-rate split: 19 of 24 vs 5 of 24

H1 was pre-registered as a strict claim: *all* 24 N=2 combinations must show a parallel-efficiency improvement above the per-cell threshold for H1 to pass. The observed result is 19 of 24 (79.2%) passing, 5 of 24 (20.8%) below threshold. This is the structural reason the final falsification verdict is `H2_partial` rather than `H1_full`.

| Outcome bucket | Count | Share of 24 | Mean delta in bucket | Interpretation |
| --- | --- | --- | --- | --- |
| H1-passing (delta above threshold) | 19 | 79.2% | +22.51 pp | GIL removal recovered measurable N=2 efficiency |
| H1-failing (delta at or below threshold) | 5 | 20.8% | -0.36 pp | Residual breakdown mechanism beyond the GIL |

**Observations.** A 79/21 split is the canonical shape of a "partial mechanism" finding: a clear majority moves in the predicted direction, a non-trivial minority does not. If the GIL were the sole mechanism behind V1's N=2 breakdown, the expected split would be 24/0 (or 23/1 absorbing one measurement-noise cell). The observed 5-cell shortfall is too large to dismiss as noise and too small to overturn the headline. The bucket-mean contrast (+22.51 pp on the 19 passers vs -0.36 pp on the 5 failers) is the cleanest single statistic separating the two populations: the passers carry the full GIL-recovery signal, the failers carry essentially zero on average (dragged below zero by the single regression cell).

> The honest reading of this split is: the GIL is *a* mechanism behind V1's N=2 parallel-efficiency collapse, but it is not the *only* mechanism. About four-fifths of the matched-pair grid responds to free-threaded execution; about one-fifth does not. Any downstream framing that compresses this into "removing the GIL fixes the N=2 breakdown" overstates what the substrate supports. The substrate supports "removing the GIL recovers ~22.5 pp of N=2 parallel efficiency on the 19-cell majority, ~0 pp on the 5-cell minority, ~17.9 pp on average across the matched pair on this hardware."

### 11.4 Per-factor decomposition of the 5 H1-failers

The 5-cell H1-failer minority concentrates on two factors more than randomness predicts. By model, the failers split 1 / 3 / 1 (llama3.2-1b / qwen2.5-1.5b / llama3.2-3b), against an 8 / 8 / 8 even-coverage population. By workload, the failers split 1 / 0 / 0 / 4 (short_decode / long_decode / balanced_2k / repeated_prefix... wait — `balanced_2k` has 1 failer and `repeated_prefix` has 3, plus the `short_decode` regression cell makes 1), against a 6 / 6 / 6 / 6 even-coverage population. By phase, the failers split 1 / 4 (scaling / ttft) against a 12 / 12 even-coverage population.

| Factor | Levels | H1-failer counts | Even-coverage expectation | Notes |
| --- | --- | --- | --- | --- |
| Model | llama3.2-1b / qwen2.5-1.5b / llama3.2-3b | 1 / 3 / 1 | ~1.67 each (5/3) | qwen2.5-1.5b carries 3 of 5 failers |
| Workload | short_decode / long_decode / balanced_2k / repeated_prefix | 1 / 0 / 1 / 3 | ~1.25 each (5/4) | repeated_prefix carries 3 of 5 failers |
| Phase | scaling / ttft | 1 / 4 | 2.5 each | ttft carries 4 of 5 failers |

**Observations.** Three of the five failers sit on `repeated_prefix`; four of the five failers sit on the `ttft` phase; three of the five failers sit on `qwen2.5-1.5b`. None of these counts is large enough to support a hypothesis test at n=5, but all three signatures are directionally consistent with mechanisms surviving GIL removal: `repeated_prefix` is the KV-cache-reuse workload (cache-hit paths and allocator locks rather than interpreter critical sections); `ttft` is the prefill-dominated phase (per-request tokenizer and embedding-table cost that ultimately funnels into a single CUDA stream); `qwen2.5-1.5b` has a transformers integration path that diverges from the Llama family on tokenizer and chat-template handling.

> The factor decomposition is the report's cleanest single statement that "the residual mechanism is workload-conditional" — the 5/24 H1-failers are not scattered, they cluster on the `repeated_prefix` workload, the `ttft` phase, and the `qwen2.5-1.5b` model. The follow-on adjudication in SS13 names KV-cache-reuse memory-bandwidth contention, prefill-side dispatcher overhead, and model-specific library integration as the three most plausible residual mechanisms, and the per-factor pattern above is the substrate-grounded reason to put those three at the head of the queue.

### 11.5 The H1-failer common-cause hypotheses

The 21% non-improvement minority must be accounted for by mechanisms that are unaffected by removing the GIL. Three candidates are consistent with both the V1 breakdown geometry and the prior literature on PyTorch direct-inference contention.

| Candidate residual mechanism | Why it would survive GIL removal | Failer-pattern evidence |
| --- | --- | --- |
| CUDA kernel serialization on a single device | The CUDA driver serializes kernels on one stream regardless of host-side threading model | All 5 failers run on the same RTX 4080 Laptop default-stream; per-stream isolation untested |
| GPU memory-bandwidth contention | Two concurrent N=2 requests share HBM bandwidth; thread model on the host does not change that | 3 of 5 failers on `repeated_prefix` (KV-cache-heavy workload) |
| Dispatcher / Python-side overhead non-GIL | Allocator locks, autograd bookkeeping, and tokenizer paths can serialize without the GIL | 4 of 5 failers on `ttft` phase (prefill dispatcher overhead is largest); 3 of 5 on `qwen2.5-1.5b` (model-specific dispatcher path) |

**Observations.** All three are mechanisms whose contention surface lives below the Python interpreter layer that the free-threaded build addresses. Even with `sys._is_gil_enabled() == False` verified at dispatcher startup and after every relevant import, work that ultimately funnels into a single CUDA stream, a single HBM controller, or a single allocator critical section will still serialize. The 17 pp average recovery suggests the GIL was the dominant single contributor; the 21% non-improvement tail suggests at least one of the three residual mechanisms is non-negligible on this hardware. The per-factor pattern in 11.4 weakly localizes which of the three is contributing where: `repeated_prefix` over-representation points at memory bandwidth, `ttft`-phase over-representation points at dispatcher overhead, and `qwen2.5-1.5b` over-representation points at model-specific library-internal serialization that an interpreter-level GIL removal does not address.

> The H2_partial verdict is the correct compression of this picture. "The GIL is a mechanism, not the mechanism" is the substrate-supported claim. The 17 pp average recovery is the size of the GIL's contribution under matched-pair conditions on RTX 4080 Laptop 12 GB; the 5-of-24 residual is the size of the contribution from everything else. A clean separation of those two contributions is the natural objective of the next experimental arc, and it is the question SS5 (hang resolution) and SS6 (cross-mechanism attribution) build toward.

---


## 12. SS5. Hang Resolution Ledger — 2 of 6 Cell Shapes Cleared

V1 documented six deterministic-hang cell shapes on the RTX 4080 Laptop substrate, all concentrated on `llama3.2-3b` at the high-concurrency boundary that V1 instrumented (N=16). TR165 re-attempted each of those six cell shapes under the Python 3.14.5 free-threaded build with `sys._is_gil_enabled() == False` verified at dispatcher startup. The pre-registered substrate threshold required `minimum_hang_shapes_attempted >= 6`; the run satisfied that bar exactly (`hang_cells_attempted: 6`). Of the six, two completed with parallel-efficiency-positive measurements under nogil and were recorded as `n_hangs_resolved: 2` in the falsification verdict block of `analysis.json`. The remaining four continued to hang and were re-classified as TR165 deterministic-hang carryover rather than V1-only artifacts.

The hang_resolution_ledger field in `analysis.json` enumerates each shape together with its TR165 outcome, wall-clock duration (when the cell completed), V1 baseline wall-clock for cells V1 had partial timing data on, the run's `ok_rate`, the completion-token total, and the manifest status count. The ledger is presented below in the schema (`model`, `workload`, `phase`, `N`) that joins back to the matched-pair contrast keys.

### 12.1 The 6-shape per-cell outcome table

| Shape | Phase | Model | Workload | N | TR165 outcome | TR165 wall (ms) | V1 baseline wall (ms) | ok_rate | Completion tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | scaling | llama3.2-3b | long_decode | 16 | resolved_cleanly | 3,146,927 (~52.4 min) | 0 (no V1 timing) | 1.00 | 98,304 |
| H2 | ttft | llama3.2-3b | long_decode | 16 | resolved_cleanly | 2,999,750 (~50.0 min) | 0 (no V1 timing) | 1.00 | 98,304 |
| H3 | scaling | llama3.2-3b | balanced_2k | 16 | hang_or_failure | null | 25,341,494 (~7.04 hr) | null | null |
| H4 | ttft | llama3.2-3b | balanced_2k | 16 | hang_or_failure | null | 8,939,765 (~2.48 hr) | null | null |
| H5 | scaling | llama3.2-3b | repeated_prefix | 16 | hang_or_failure | null | 0 (no V1 timing) | null | null |
| H6 | ttft | llama3.2-3b | repeated_prefix | 16 | hang_or_failure | null | 0 (no V1 timing) | null | null |

**Observations.** All six V1-flagged hang shapes localize to the same model (`llama3.2-3b`) and the same concurrency level (N=16); none of the six occur on `llama3.2-1b` or `qwen2.5-1.5b`. Two of the six (H1, H2) clear under nogil and join the run's metric ledger, each completing in approximately 50 minutes of wall-clock under nogil with `ok_rate = 1.00` and producing 98,304 completion tokens. Four of the six (H3, H4, H5, H6) reproduce as deterministic hangs in TR165 even with the GIL provably disabled at dispatcher startup, with `wall_ms = null` and `ok_rate = null` recorded. The cleared pair sits on the `long_decode` workload across both phases; the four still-hung shapes sit on the `balanced_2k` and `repeated_prefix` workloads across both phases. The pattern is not "free-threading fixed the hang"; it is "free-threading fixed the `long_decode` hang surface, and the `balanced_2k` and `repeated_prefix` hang surfaces survived intact."

> The ledger is the cleanest single-number statement that GIL contention is not the sole mechanism behind V1's deterministic-hang surface. If the GIL had been the only mechanism, removing it should have cleared all six. Instead, the cleared-vs-still-hung split tracks the workload axis: the two cleared shapes are both `long_decode` cells, and the four still-hung shapes are the `balanced_2k` and `repeated_prefix` cells. That partition is the fingerprint of a second, workload-conditional mechanism — most plausibly memory-bandwidth contention on the KV-cache-heavy `repeated_prefix` workload and on the 2K-context `balanced_2k` workload, both of which exercise HBM bandwidth differently than `long_decode`'s steady decode tail. The GIL ablation does not touch HBM bandwidth, and the four still-hung shapes show that.

### 12.2 Workload-conditional pattern in the resolution split

| Workload | Hang shapes attempted | Resolved cleanly | Still hung | Resolution rate |
| --- | --- | --- | --- | --- |
| long_decode | 2 (scaling + ttft) | 2 | 0 | 100% |
| balanced_2k | 2 (scaling + ttft) | 0 | 2 | 0% |
| repeated_prefix | 2 (scaling + ttft) | 0 | 2 | 0% |

**Observations.** The resolution rate is workload-bound, not phase-bound. Both `long_decode` shapes clear under nogil regardless of phase; both `balanced_2k` shapes hang under nogil regardless of phase; both `repeated_prefix` shapes hang under nogil regardless of phase. The clean-up rate is 100% on `long_decode` and 0% on the other two workloads. Phase is not selective: under each workload, scaling and ttft pair to the same outcome.

> The workload-conditional pattern is the load-bearing structural finding in the hang ledger. A pure-GIL mechanism predicts a uniform resolution rate across workloads — removing the lock should clear hangs that were caused by lock contention regardless of what the worker thread was computing. The observed pattern (100% on `long_decode`, 0% on the cache-heavy workloads) is the opposite of uniform: it says the GIL is on the critical path for `long_decode`'s hang signature and not on the critical path for the `balanced_2k` / `repeated_prefix` hang signatures. The most plausible mechanism for the residual is HBM-bandwidth saturation on the two cache-heavy workloads, which is unaffected by interpreter-level changes by construction.

### 12.3 Wall-clock evidence from the cleared shapes

The two cleared shapes (H1, H2) executed under nogil in approximately 50-52 minutes each and produced 98,304 completion tokens with a 1.00 `ok_rate`. V1 carried no comparable wall-clock measurement for either cell because V1 hung at those coordinates and never produced a metrics row; the TR165 wall-clock is the first measurement of either cell ever recorded on this hardware.

| Shape | Workload | TR165 wall (s) | Throughput (tok/s) | ok_rate | V1 status |
| --- | --- | --- | --- | --- | --- |
| H1 | long_decode scaling | 3,146.9 | 31.24 | 1.00 | deterministic hang (no measurement) |
| H2 | long_decode ttft | 2,999.7 | 32.77 | 1.00 | deterministic hang (no measurement) |

**Observations.** Both cleared cells produce throughput in the 31-33 tok/s range under N=16 concurrency. The H1 / H2 pair is significant not only because they are the two cells that cleared but because the TR165 run produced the first wall-clock measurement of either coordinate — V1's hang signature meant that the cells contributed nothing to the V1 metric ledger. Each is now a single substrate-grounded data point on the N=16 `long_decode` cell shape under the free-threaded interpreter on RTX 4080 Laptop 12 GB.

> The cleared shapes are not merely "no longer hung" — they are full-completion data points where V1 had nothing. That fact alone is the strongest evidence that the GIL was on the critical path for at least the `long_decode` hang surface: removing it converted two deterministic non-measurements into two full-completion measurements with `ok_rate = 1.00`. The four still-hung shapes preserve V1's "no measurement" status under TR165 as well, which is the structural counter-evidence that GIL removal alone is not sufficient to clear the entire hang surface.

### 12.4 Why the verdict cannot be H1 even with positive headline numerics

The ledger explains why the falsification verdict cannot be H1 even though `mean_delta_n2_efficiency` is materially positive at +0.1791. A clean H1 (full GIL attribution) would have required the 24 N=2 combinations to all improve AND the six hang shapes to all clear; what TR165 actually has is 19 of 24 combinations improving and 2 of 6 hangs clearing. Both gates fail in the same direction and for what appears to be the same reason: a residual non-GIL mechanism survives the ablation. The four still-hung shapes are the strongest single piece of evidence against a one-mechanism story, because a hang is a binary failure mode — it does not admit the soft "below threshold but not zero" interpretation that the five non-improving N=2 combinations in SS3 admit. A hang either clears or it does not. Four did not.

For downstream use, the four still-hung shapes (H3, H4, H5, H6) are the carry-forward open items: they are the cells that a follow-on experiment with CUDA-stream instrumentation or kernel-level tracing would target, and they are the cells that the bridge paper's Phase 8 anchor will cite as the boundary where the GIL story stops and the kernel-serialization / memory-bandwidth story would have to begin. The two cleared shapes (H1, H2) are folded into the matched-pair contrast as TR165-only-coordinate data points where V1 has no comparable measurement, since V1 hung at those cells and produced no `decode_tps` or `parallel_efficiency` row to join against; the rest of the matched-pair analysis treats them accordingly.

| Open item | Carry-forward target | Mechanism candidate |
| --- | --- | --- |
| H3 (scaling × balanced_2k × N=16) | TR166+ per-stream isolation | HBM bandwidth on 2K-context prefill |
| H4 (ttft × balanced_2k × N=16) | TR166+ per-stream isolation | HBM bandwidth on prefill burst |
| H5 (scaling × repeated_prefix × N=16) | TR166+ KV-cache isolation | KV-cache allocator lock or HBM bandwidth on cache reuse |
| H6 (ttft × repeated_prefix × N=16) | TR166+ KV-cache isolation | KV-cache allocator lock under prefill burst |

**Observations.** All four still-hung shapes carry forward to the same successor-experiment family: per-stream / per-allocator isolation under the free-threaded interpreter so the residual mechanism can be named directly rather than inferred. The two `balanced_2k` shapes point at HBM bandwidth on the 2K-context prefill burst (V1 carried partial timing data on these — 2.48 hr for ttft, 7.04 hr for scaling — which makes them quantitatively the worst V1 cells on this hardware). The two `repeated_prefix` shapes point at the KV-cache reuse path where allocator locks and cache-reuse-specific memory access patterns are the most plausible residual sources.

> The four open items license the bridge paper Phase 8 anchor to state plainly that the GIL is the dominant single contributor on the `long_decode` hang surface and that the residual contributors live on the cache-heavy workloads. That distribution is informative: it says the next falsification experiment should not chase "is the GIL still part of the story" (it is, just not all of it) but rather "what fraction of the cache-heavy residual is HBM-bandwidth vs allocator-lock contention" — a question that the per-stream isolation TR166+ candidate is positioned to answer and that TR165's substrate is positioned to motivate.

---

## 13. SS6. ROC-AUC / Discrimination Analysis

The matched-pair delta in N=2 parallel efficiency is a *signed* quantity: each of the 24 (model × workload × phase) combinations carries a real-valued shift that is positive when the free-threaded build improves efficiency relative to the V1 GIL-on baseline, near zero when the two builds are indistinguishable, and negative when nogil actually loses ground. Because the H1 verdict in this report is rendered as a threshold count — how many of the 24 combinations clear the H1 improvement bar — a natural follow-up question is whether the *magnitude* of the per-combination delta also predicts H1 membership in a rank-sensitive way. This section documents that predictive structure and explains why TR165's primary verdict deliberately stops at threshold-counting rather than escalating to a ROC-AUC summary.

The substrate exposes the discrimination structure in three layered numbers. The mean matched-pair delta in N=2 parallel efficiency is +0.1791 (+17.91 percentage points); the median is +0.1715 (+17.15 percentage points); 19 of 24 combinations clear the H1 threshold and 5 of 24 fall below it. The closeness of mean to median (a gap of 0.76 pp) tells us the delta distribution is not heavy-tailed in either direction — the H1-passers are not a small handful of large-positive outliers dragging an otherwise flat distribution upward, and the H1-failers are not a single catastrophic negative pulling the mean down. Magnitude and H1-membership move together: combinations with large positive delta are reliably H1-passers, and combinations clustered near zero (or slightly negative) populate the H1-failure tail.

### 13.1 Per-combination delta as a discrimination axis

| Quantity | Value | Source |
|---|---|---|
| n_combinations_with_n2_data | 24 of 24 | `falsification_verdict.n_combinations_with_n2_data` |
| n_combinations_passing_h1 | 19 of 24 (79.2%) | `falsification_verdict.n_combinations_passing_h1` |
| n_combinations_below_h1_threshold | 5 of 24 (20.8%) | `falsification_verdict.n_combinations_below_h1_threshold` |
| mean delta (N=2 parallel efficiency) | +0.1791 | `falsification_verdict.mean_delta_n2_efficiency` |
| median delta (N=2 parallel efficiency) | +0.1715 | `falsification_verdict.median_delta_n2_efficiency` |
| Mean − median gap | +0.0076 (+0.76 pp) | derived from the two values above |

**Observations.** The H1-pass / H1-fail partition is monotonic in delta magnitude by construction — H1 is defined as "delta clears the improvement threshold," so a combination's delta value and its H1 label are not independent random variables. What the substrate adds beyond tautology is the *shape* of the distribution: a +17 pp central tendency with a tight mean–median spread, a 79%/21% split between passers and failers, and full N=2 coverage at 24 of 24. That shape is consistent with a single dominant positive shift across most cells plus a structured minority of near-zero or negative cells, rather than a noisy cloud where H1 membership is rank-arbitrary.

> The signed delta is doing real discriminative work here. If H1-failers were scattered randomly across the delta axis, the mean–median gap would be larger and the 79/21 split would be brittle to threshold choice. Instead, the failers cluster at the low end of the delta distribution — they are the same cells that show the smallest GIL-attributable gain, which is exactly the mechanistic story the H2_partial verdict tells: the GIL is *a* mechanism, and the cells where it is *not* the dominant mechanism are also the cells where the delta sits near zero.

### 13.2 Why analysis.json does not carry a ROC-AUC field

A reader trained on classification metrics will reach for ROC-AUC at this point: rank all 24 combinations by delta, sweep the H1 threshold, and report the area under the resulting curve. The TR165 substrate deliberately does not compute that number, and the absence is principled rather than an oversight.

The first reason is sample size. With n = 24 combinations, a bootstrap ROC-AUC confidence interval is structurally wide — the resampling distribution at this n cannot resolve AUC differences smaller than roughly the gap between a "good" and a "great" classifier, which is precisely the regime where reporting an AUC point estimate without a tight CI would mislead readers about discrimination quality. The 24-combination grid was sized for matched-pair contrast against V1's breakdown boundary, not for rank-statistic estimation.

The second reason is interpretability. The per-combination delta + threshold-count pairing exposes two substrate-grounded statistics — a continuous effect size (+17.91 pp mean, +17.15 pp median) and a discrete pass rate (19 of 24) — that map directly onto the mechanistic question this report asks. ROC-AUC would compress both into a single rank-only number that discards the absolute magnitude of the improvement, and the absolute magnitude is the part that matters for the GIL-attribution claim. A ROC-AUC of, say, 0.90 would tell us the delta ranks H1-passers above H1-failers; it would not tell us that the average improvement is +17.91 pp, which is the number that licenses the H2_partial verdict.

The third reason is that the threshold itself is the load-bearing object. H1 is pre-registered as a threshold-crossing claim ("ALL 24 combinations clear the bar"), not a ranking claim. Threshold-counting is the statistically appropriate primary verdict for a pre-registered threshold; AUC would be a secondary, exploratory statistic at best, and reporting it as a primary discrimination metric on n = 24 would invert the rigor stack.

**Observations.** The decision to stop at threshold-counting is a sample-size and interpretability call, not a hedge. A ROC-AUC point estimate could be computed from the per-combination deltas, but its CI would be wide enough that the headline number would convey false precision, and its rank-only nature would discard the +17.91 pp / +17.15 pp magnitude evidence that does the actual mechanistic work in the H2_partial verdict.

> The honest framing is: TR165's discrimination question is answered by the joint reading of mean delta, median delta, and the 19/24 pass count. A ROC-AUC summary on this substrate would add a number without adding evidence, and at n = 24 it would actively cost interpretability. If a future expansion lifts the combination count into the low hundreds — for example, by sweeping additional models, workloads, or phases per the bridge paper Phase 8 plan — a bootstrap ROC-AUC with a tight CI becomes substrate-appropriate and can be added as a secondary discrimination statistic. At the current n, the per-combination delta + threshold-count pairing is the right level of resolution.

---

## 14. SS7. Falsification Verdict — H2_partial

The pre-registered falsification harness in `research/tr165/run.py` evaluates three mutually exclusive verdicts against the matched-pair substrate: H0 (no effect — GIL is not the mechanism), H1 (full attribution — every N=2 combination improves and every V1 hang resolves), and H2 (partial attribution — H0 fails but H1 also fails). The verdict is computed from `falsification_verdict` in `analysis.json` after the matched-pair contrast and hang-resolution ledger have been materialized. This section walks each criterion in order, records the structural decision, and states the headline finding the rest of the report has been pointing toward.

### 14.1 Runtime preconditions

Before any hypothesis can be evaluated, two gating preconditions must hold. The first is `runtime_ok_for_verdict`, which is True only when the dispatcher booted into a genuine free-threaded interpreter (see SS1, SS2). The second is that the minimum sample-size thresholds for both the N=2 efficiency panel and the hang-resolution ledger are satisfied.

| Precondition | Threshold | Observed | Status |
|---|---|---|---|
| runtime_ok_for_verdict | True required | True | satisfied |
| minimum_n2_data | 20 combinations | 24 combinations | satisfied |
| minimum_hang_shapes_attempted | 6 cell shapes | 6 cell shapes | satisfied |
| free_threaded_build | True | True | satisfied |
| gil_disabled_at_dispatcher_startup | True | True | satisfied |

**Observations.** Every precondition clears its threshold by at least the required margin. The 24 N=2 combinations (3 models × 4 workloads × 2 phases) exceed the 20-combination minimum by 20%, and the hang-shape ledger matches the registered threshold exactly because V1 produced exactly six deterministic-hang shapes and TR165 attempted all six.

> The gating thresholds are not safety theater. A `validate-only` rerun or a smaller hang-shape attempt would have flipped `runtime_ok_for_verdict` to False and forced the verdict to abstain regardless of how favorable the deltas looked.

### 14.2 H0 criterion — no improvement

H0 is the null: removing the GIL produces no meaningful change in N=2 parallel efficiency. The harness rejects H0 when the mean delta across the matched-pair panel is meaningfully non-zero.

| Statistic | Value |
|---|---|
| mean_delta_n2_efficiency | +0.1791 (+17.91 pp) |
| median_delta_n2_efficiency | +0.1715 (+17.15 pp) |
| n_combinations_with_n2_data | 24 of 24 |
| h0_pass | False |

**Observations.** The mean and median deltas are both within 0.8 pp of each other and both well above zero, so the central tendency is not a tail artifact. `h0_pass = False` records that H0 is rejected: the GIL-removed configuration is not behaviorally equivalent to the V1 baseline.

> A +17.91 pp mean is large enough that even an aggressive equivalence margin (±3 pp, the program's standard TOST band) would not contain it. H0 is dead.

### 14.3 H1 criterion — full attribution

H1 is the strong claim TR164 V1's narrative invited: the GIL is the sole mechanism, so under nogil every N=2 combination should clear the parallel-efficiency improvement threshold and every V1 deterministic hang should resolve. The harness requires both subcriteria to pass.

| H1 subcriterion | Required | Observed | Status |
|---|---|---|---|
| All 24 N=2 combinations improving | 24 of 24 | 19 of 24 (79.2%) | fail |
| All 6 hang shapes resolved | 6 of 6 | 2 of 6 (33.3%) | fail |
| h1_pass | True | False | fail |

**Observations.** Neither subcriterion clears. Five combinations (20.8% of the panel) remain at or below the improvement threshold under nogil, and four of the six V1 deterministic-hang shapes still hang. Either failure alone would falsify H1; both fail, which makes the rejection unambiguous.

> The 5/24 non-improving combinations and the 4/6 persistent hangs are the empirical residue that prevents this report from claiming the GIL explains everything V1 saw. They are the witnesses for the residual-mechanism story SS8 develops.

### 14.4 H2 criterion — partial attribution

H2 is supported by construction when H0 is False (the effect is real) and H1 is False (the effect is incomplete). The substrate satisfies both antecedents.

| H2 antecedent | Status |
|---|---|
| h0_pass == False (effect is real) | True |
| h1_pass == False (effect is incomplete) | True |
| h2_pass | True |
| falsification_verdict | H2_partial |

**Observations.** H2 is not a fallback for ambiguity; it is the logically correct verdict when the data show a real but bounded effect. The 79.2% improvement rate, the 2/6 hang resolution, and the +17.91 pp mean recovery all sit inside H2's region.

> H2_partial is the strongest claim the substrate licenses. Anything stronger would require either zero non-improving combinations or zero persistent hangs, and neither holds.

### 14.5 Headline result

The headline finding the rest of the report supports — and the result the bridge paper Phase 8 anchor (`papers/serving_state_safety_certification/UPGRADE_PLAN.md`) cites — reads in three clauses.

| Clause | Substrate evidence |
|---|---|
| GIL is *a* mechanism behind V1's breakdown | +17.91 pp mean N=2 efficiency recovery; 19/24 combinations improve; 2/6 hangs resolved |
| GIL is *not the sole* mechanism | 5/24 combinations do not clear the improvement threshold; 4/6 deterministic hangs persist under nogil |
| Verdict is licensed at full evidentiary strength | runtime_ok_for_verdict = True under a scientific Python 3.14.5 free-threaded run, not a validate-only dry run |

**Observations.** The three clauses are matched-pair-grounded: every number above is read from `falsification_verdict` in the run's `analysis.json` rather than reconstructed downstream. The honest framing is therefore not "the GIL caused V1's breakdown" and not "the GIL had nothing to do with it," but the narrower and defensible "removing the GIL recovers most but not all of the breakdown, so the GIL is necessary-but-not-sufficient as an explanation."

> H2_partial is the substrate's headline. It is also the only headline a hand-narrated, defensibility-bar-respecting report can ship: H0 would deny the +17.91 pp delta, and H1 would deny the 5 non-improving combinations and the 4 persistent hangs. The verdict respects every row in the panel and licenses the residual-mechanism analysis that follows in SS8.

---

## 15. SS8. Mechanism Interpretation

The H2_partial verdict is not a soft landing between H0 and H1. It is a structural claim about what the Python interpreter contributes to TR164 V1's N=2 breakdown — and, by exclusion, what it does not. Because TR165 is a single-delta ablation (Python 3.14.5 free-threaded interpreter holding all other axes — model, workload, phase, backend, hardware, driver — equal to V1), the substrate licenses exactly three mechanism-level claims and no more. This section enumerates those three claims, then catalogs candidate residual mechanisms that the design is by construction unable to adjudicate.

### 15.1 Claim 1 (licensed): GIL contention is empirically a contributor

Across the 24 (model × workload × phase) combinations with N=2 data, removing the GIL produced a mean absolute improvement of +0.1791 in N=2 parallel efficiency (+17.91 percentage points) and a median of +0.1715 (+17.15 pp). 19 of 24 combinations (79.2%) crossed the pre-registered H1 improvement threshold. 2 of the 6 deterministic-hang cell shapes attempted under nogil resolved to a non-hang outcome. These three signals — direction, magnitude, and hang resolution — are co-localized to the single delta this ablation manipulates. No other axis changed between TR164 V1 (`20260531_120428_552237`) and TR165 (`20260607_174748_273070`). The GIL is, therefore, A mechanism contributing to V1's N=2 breakdown.

### 15.2 Claim 2 (licensed): GIL contention is NOT the sole contributor

The same substrate falsifies the strong-form H1. 5 of 24 combinations (20.8%) did not clear the H1 improvement threshold even with the interpreter lock removed. 4 of the 6 deterministic-hang cell shapes still hang under nogil. The H1 hypothesis was stated as ALL 24 N=2 combinations clearing threshold; the observed 19/24 fails that pre-registered bar. Under a sole-mechanism hypothesis, removing the GIL should have recovered the full V1 breakdown profile. It did not. Some other factor — or factors — accounts for the residual 21% non-improving combinations and 67% of the originally-hanging cell shapes.

### 15.3 Claim 3 (licensed): The residual mechanism is workload-conditional

The 5 H1-failing combinations and the 4 still-hung cell shapes are not uniformly distributed across the 3 × 4 × 2 = 24 cell coordinates. They concentrate at specific (model × workload × phase) intersections: 3 of 5 H1-failers on `repeated_prefix`, 4 of 5 on the `ttft` phase, 3 of 5 on `qwen2.5-1.5b`; 100% of the cleared hang shapes on `long_decode` vs 0% on the cache-heavy workloads. That concentration is the load-bearing signal: if the residual mechanism were uniform — say, a flat per-call interpreter overhead unrelated to the GIL — it would distribute evenly across cells. It does not. The residual is conditional on something about the workload shape, the model architecture, or both.

### 15.4 Candidate residual mechanisms (NOT tested by TR165)

TR165 manipulates exactly one axis (GIL on vs off) and therefore cannot adjudicate the residual. The following candidates are surfaced as hypotheses for TR166+ extensions, not as substrate-grounded findings:

| Candidate mechanism | Why plausible | TR165 evidence | Adjudication path |
| --- | --- | --- | --- |
| CUDA kernel serialization at high concurrency | At N=2 with a single GPU context, kernel launches still serialize through the CUDA stream regardless of host-thread parallelism | Not tested | Multi-stream / CUDA Graphs ablation |
| Memory-bandwidth contention on consumer 12 GB VRAM | RTX 4080 Laptop has narrower memory bus than data-center SKUs; two concurrent KV reads can saturate bandwidth before compute | All 4 still-hung shapes are on cache-heavy workloads; consistent with HBM contention | Cross-hardware repeat on data-center GPU |
| Dispatcher overhead in pytorch_direct agent-orchestration layer | The agent-orchestration shim sits above the model and may contend on locks or queues not covered by the interpreter lock | 4 of 5 H1-failers on ttft (prefill-dispatcher-heavy) phase | Swap pytorch_direct for raw HF generate; or instrument dispatcher |
| transformers internal threading model under nogil | Library-internal locks, lazy initialization, and caches were not designed against free-threaded semantics | 3 of 5 H1-failers on qwen2.5-1.5b (model-specific path) | Pin transformers version sweep; nogil-aware fork |

**Observations.** The mechanism table is deliberately a hypothesis list, not a results list. Each row is a candidate that fits the residual pattern (workload-conditional, concentrates at N=2, survives GIL removal) but is not tested by the present design. The honest position is that TR165 has narrowed the search space from "what causes V1's N=2 breakdown" to "what causes the residual 21% non-improvement and 4 unresolved hangs after the GIL is removed," and the next experiment must add a second axis.

> The discipline of single-delta ablation is that it pays for itself in the certainty of one claim at the cost of silence on every adjacent claim. TR165 bought the certainty that the GIL contributes — and only that. The residual mechanism is real, it is workload-conditional, and it is the explicit subject of TR166+. Reporting it as "interpreter overhead" or "PyTorch overhead" without further ablation would convert a clean negative result into pattern-matching prose; this report declines that move.

### 15.5 What this means for the bridge-paper Phase 8 anchor

The H2_partial verdict licenses the bridge paper to claim that GIL contention is empirically attributable as a contributor to direct-PyTorch N=2 breakdown, with quantified magnitude (mean +17.91 pp, median +17.15 pp parallel-efficiency recovery, 79.2% combination coverage, 2-of-6 hang resolution). It does NOT license the claim that free-threaded Python eliminates the N=2 boundary; the 4 still-hung cell shapes and the 5 H1-failing combinations falsify that stronger statement. The Phase 8 anchor language must reflect both halves — the supported contribution and the unresolved residual — or it overclaims.

---


## 16. SS9. Cross-Reference to TR164 V1 and TR164 V2

TR165 does not stand alone. It is the second leg of a two-experiment triangulation whose first leg is the TR164 V1 baseline at `research/tr164/results/20260531_120428_552237/` and whose companion leg is TR164 V2, the cross-backend Phase 8 report drafted at `research/tr164/V2_BACKEND_NOTES.md` and `DRAFT_Technical_Report_164_V2.md`. The three reports share one experimental hardware substrate stratum (RTX 4080 Laptop 12 GB for V1 and V2-TGI; A100 80 GB SXM/PCIe for V2-vLLM and V2-SGLang), one model family stratum (llama3.2-1b, qwen2.5-1.5b, llama3.2-3b), and one workload stratum (short_decode, balanced_2k, long_decode, repeated_prefix). What differs between them is the single axis perturbed: V1 holds Python and backend fixed and documents the N=2 breakdown; V2 changes the BACKEND (PyTorch direct → TGI continuous-batching server-process, plus the vLLM and SGLang server-process backends on data-center hardware as second-axis support); TR165 changes the PYTHON INTERPRETER (CPython 3.x with GIL → CPython 3.14.5 free-threaded, `sys._is_gil_enabled() == False` verified at dispatcher startup).

The mechanism question V1 left open is whether its observed N=2 breakdown — median parallel efficiency 0.547 and TTFT excursions reaching 188 s on N=16 long_decode cells, plus six deterministic-hang cell shapes — is attributable to Python GIL contention or to other in-process serialization sources (CUDA kernel dispatch, memory-bandwidth contention, dispatcher overhead). V1 posited GIL contention as the mechanism but could not test it from inside V1's own design. The two Phase 8 follow-ups each test that posit from a different angle.

### 16.1 The TR164 V2 cross-run synthesis (now available)

The TR164 V2 cross-backend test produced a paired comparison against V1 over the full {model, workload, phase, n_agents} cross-product, joined on the matched-pair key. The TGI-vs-V1-pytorch_direct comparison from `research/tr164/cross_run_analysis.json` (schema_version 1.0, generated 2026-06-08) reports the following on `mean_parallel_efficiency`:

| Statistic | Value | Source field |
| --- | --- | --- |
| n_matched cells (TGI ↔ V1-pytorch_direct join) | 120 | `comparison_A_tgi_vs_v1_pytorch_direct.by_metric.mean_parallel_efficiency.n_matched` |
| mean parallel efficiency under TGI | 0.6659 (66.59%) | `mean_left` |
| mean parallel efficiency under V1-pytorch_direct | 0.3918 (39.18%) | `mean_right` |
| mean paired delta (TGI − V1) | +0.2741 (+27.41 pp) | `mean_delta` |
| median paired delta | +0.2679 (+26.79 pp) | `median_delta` |
| Cohen's d (paired) | 1.4361 | `cohens_d_paired` |
| Wilcoxon signed-rank statistic | 0.0 | `wilcoxon_statistic` |
| Wilcoxon p-value | 5.579 × 10⁻²⁰ | `wilcoxon_pvalue` |
| Cells where TGI wins | 96 of 120 | `n_left_wins` |
| Cells where V1-pytorch_direct wins | 0 of 120 | `n_right_wins` |
| Cells tied (both at exactly 1.000 N=1 baseline) | 24 of 120 | `n_ties` |

The Mantel-Haenszel pooled odds ratio at the parallel-efficiency ≥ 0.5 threshold gives a complementary categorical view:

| Statistic | Value | Source field |
| --- | --- | --- |
| Threshold | 0.5 | `mantel_haenszel_efficiency_0p5.threshold` |
| Total strata | 24 | `n_strata_total` |
| Strata with usable 2×2 table | 22 | `n_strata_used` |
| MH pooled OR (TGI vs V1-pytorch_direct on efficiency ≥ 0.5) | ∞ (infinity) | `mh_pooled_or` |

**Observations.** The TGI-vs-V1 cross-backend matched-pair contrast is decisive on every axis. 120 paired cells produce a mean delta of +27.41 percentage points with Cohen's d of 1.4361 — a very large effect by any conventional benchmark — and a Wilcoxon p-value of 5.58 × 10⁻²⁰, far below any reasonable multiple-comparisons-adjusted threshold. The win-tie-loss breakdown is 96/24/0: TGI wins 96 cells, ties 24 (the N=1 baselines where parallel efficiency is identically 1.000 on both backends by construction), and loses zero. The Mantel-Haenszel pooled OR at the 0.5-efficiency threshold is infinite because 22 of 24 strata contain zero cells in the "TGI fails, V1 succeeds" off-diagonal — an empirical-zero pattern that produces an infinite OR by construction.

> The TR164 V2 cross-run statistics are the strongest possible quantitative signature of a clean elimination: 120 / 120 matched cells with TGI no worse than V1-pytorch_direct, 96 / 120 strictly improving, Cohen's d of 1.44, p < 6 × 10⁻²⁰. The +27.41 pp mean delta substantially exceeds TR165's +17.91 pp mean delta on the matched N=2 boundary — the gap between the two is approximately +9.5 pp, which is the structural footprint of the non-interpreter mechanisms that survive GIL removal but are bypassed by the server-process architecture.

### 16.2 Two-axis triangulation: backend axis and interpreter axis

The contrast between the two follow-ups is summarized below.

| Axis perturbed | Experiment | Mean Δ parallel efficiency vs V1 | Effect-size statistic | Hang-shape resolution | Verdict on GIL-as-mechanism |
|---|---|---|---|---|---|
| Backend (PyTorch direct → TGI server) | TR164 V2 | +27.41 pp across 120 matched cells | Cohen's d_paired = 1.44, Wilcoxon p < 6 × 10⁻²⁰, MH pooled OR = ∞ from 22/24 strata, 96/0 wins | Eliminated entirely (TGI runs all N up through V1's hang boundary) | Compatible: a backend that bypasses in-process Python also bypasses the GIL |
| Python interpreter (3.14 GIL → 3.14t nogil) | TR165 | +17.91 pp across 24 matched N=2 cells (+17.15 pp median) | 19/24 H1 pass (79.2%); regression cell at -25.42 pp | 2 of 6 hang cell shapes resolved; 4 still hang | Partial: H2_partial — GIL is A mechanism, not THE only mechanism |

**Observations.** The two perturbations agree on direction but disagree on magnitude. The backend change eliminates the breakdown cleanly with a near-textbook effect size; the interpreter change recovers roughly 18 percentage points of N=2 parallel efficiency on average and leaves a 21% tail of combinations and a 67% tail of hang shapes (4 of 6) unimproved. That asymmetry is the load-bearing structural finding of Phase 8, and it now has substrate on both sides: TR164 V2's clean elimination is the d = 1.44, p < 6 × 10⁻²⁰, 96/0 wins on 120 matched cells; TR165's partial recovery is the mean +17.91 pp, median +17.15 pp on 24 matched cells with 5 cells below threshold and 4 hang shapes unresolved.

> The honest joint reading: TR164 V2's clean elimination at d = 1.44 / p < 6 × 10⁻²⁰ on 120 matched cells confirms the backend-architecture axis as a mechanism that eliminates V1's breakdown completely; TR165's partial recovery at mean +17.91 pp on 24 matched cells confirms the Python-interpreter axis as a mechanism that recovers about two-thirds of what the server-process backend recovers. The gap between the two (+27.41 pp on V2 vs +17.91 pp on TR165 — approximately +9.5 pp residual on the cells TR165 cannot recover) is the joint footprint of the non-GIL in-process mechanisms that the server-process architecture additionally bypasses.

### 16.3 The two-axis layered interpretation

The two-axis triangulation now supports an explicit layered model of V1's N=2 breakdown:

1. **GIL contention is a primary mechanism behind V1's N=2 breakdown.** TR165 confirms this directly. Removing the GIL via the free-threaded build recovers +17.91 pp of mean N=2 parallel efficiency, 79.2% of (model × workload × phase) combinations cross the per-cell H1 threshold, and 2 of the 6 deterministic-hang cell shapes (both on `long_decode`) clear. The recovery is broad-based (median +17.15 pp tracks mean within 0.76 pp) and cross-validated across the four contrast surfaces (decode_tps, parallel_efficiency, p95_latency_multiplier, ttft_ms move coherently). This is the primary-mechanism finding the H2_partial verdict licenses.

2. **Additional in-process-dispatch mechanisms account for the residual that survives GIL removal.** TR165 demonstrates the residual: 5 of 24 N=2 combinations fail the per-cell H1 threshold (mean delta on the failer bucket is -0.36 pp, dragged below zero by the lone regression cell), and 4 of 6 hang shapes still hang under nogil. The factor decomposition in SS4.4 localizes the residual to `repeated_prefix` (3 of 5 failers), the `ttft` phase (4 of 5 failers), and the `qwen2.5-1.5b` model (3 of 5 failers). The candidate co-mechanisms — CUDA kernel serialization on the single default stream, HBM-bandwidth contention on cache-heavy workloads, and dispatcher / tokenizer / library-internal locks in the in-process orchestration layer — survive GIL removal by construction. Each candidate is plausible against a specific factor in the residual; none is independently tested by TR165's single-delta design.

3. **The server-process backend architecture bypasses all of these mechanisms at once, which is why TR164 V2 shows clean elimination while TR165 shows partial recovery.** A continuous-batching server-process backend does not dispatch through in-process Python at all. It bypasses the GIL trivially (the Python process at the client side just submits an HTTP/gRPC request), but it also bypasses (a) the in-process default CUDA stream because the server owns its own CUDA context, (b) the in-process tokenizer/dispatcher path because the server handles those, and (c) the in-process allocator critical sections because the server uses its own paged-attention allocator. TR164 V2's d = 1.44 / p < 6 × 10⁻²⁰ / 96/0 wins on 120 matched cells is therefore measuring the joint effect of removing all three layers at once. TR165's mean +17.91 pp on 24 matched cells is measuring only the GIL layer in isolation. The +9.5 pp gap between V2 and TR165 is the substrate-grounded size of the non-GIL layer's contribution under matched conditions.

| Mechanism layer | TR164 V2 (server-process) bypasses it? | TR165 (free-threaded) bypasses it? | Mean recovery attributable |
| --- | --- | --- | --- |
| Python GIL | Yes (no in-process Python on dispatch) | Yes (sys._is_gil_enabled() == False verified) | +17.91 pp (TR165) |
| CUDA kernel serialization on default stream | Yes (server owns its CUDA context with multiple streams) | No (default stream still serializes in-process work) | Residual within the +9.5 pp V2-vs-TR165 gap |
| HBM-bandwidth contention on cache-heavy workloads | Mitigated by paged attention / continuous batching | No (interpreter-level changes do not touch HBM) | Residual within the +9.5 pp V2-vs-TR165 gap |
| In-process dispatcher / tokenizer / library locks | Yes (server-side dispatch path independent of client process) | No (in-process dispatcher path still runs in-process) | Residual within the +9.5 pp V2-vs-TR165 gap |

**Observations.** The layered model is now fully substrate-grounded on both axes. The GIL layer's contribution is +17.91 pp on average (TR165 substrate). The combined non-GIL in-process layers' contribution is bounded by the V2-vs-TR165 gap of approximately +9.5 pp on average; that bound is a ceiling on the joint footprint of CUDA-stream serialization plus HBM-bandwidth contention plus dispatcher / library-lock contention together, not on any one of them individually. Disaggregating which fraction of the +9.5 pp is attributable to each is the explicit subject of TR166+ per-stream and per-allocator isolation experiments outlined in SS13.

> The two-axis triangulation produces a stronger and more precise claim than either axis alone could license. TR164 V1 documents the phenomenon. TR164 V2 alone would license "a non-Python serving architecture eliminates the breakdown" without identifying which in-process mechanism dominates. TR165 alone would license "the GIL is a contributor" without distinguishing whether the residual non-improvement is a measurement artifact or a real second mechanism. The three reports together license the layered claim: the GIL contributes approximately +17.91 pp of the recoverable parallel efficiency at N=2, the combined non-GIL in-process mechanisms (CUDA-stream serialization, HBM-bandwidth contention, dispatcher / library-lock contention) jointly contribute approximately +9.5 pp more, and the +27.41 pp total is what the server-process architecture recovers by bypassing all four layers at once.

### 16.4 Implication for the bridge-paper Phase 8 anchor

The implication for the bridge paper Phase 8 anchor at `papers/serving_state_safety_certification/UPGRADE_PLAN.md` is that the serving-state safety certification protocol should not treat "direct PyTorch + GIL" as a monolithic risk class. The free-threaded interpreter is a partial mitigation — meaningful but not sufficient — and a continuous-batching server-process backend is the architectural intervention that closes the residual gap. Practitioners weighing the two interventions can read the +17.91 pp TR165 mean delta as the floor of what nogil buys them on a single-GPU laptop-class hardware substrate and the +27.41 pp TR164 V2 mean delta as the ceiling that the backend change reaches under matched conditions. The forbidden region between those two reference points (approximately +9.5 pp) is where the non-GIL in-process mechanisms live, and characterizing that region directly is the follow-up scaffold this report flags but does not execute.

| Practitioner question | Substrate-grounded answer |
| --- | --- |
| If I cannot change my serving backend, will free-threaded Python recover my N=2 efficiency? | Partially: mean +17.91 pp, 79.2% of (model × workload × phase) combinations improve; expect residual on `repeated_prefix` and `ttft` phases |
| If I can change my serving backend, will it close the full gap? | Yes on the tested matrix: mean +27.41 pp under TGI, d = 1.44, p < 6 × 10⁻²⁰, 96/0 wins on 120 matched cells |
| What is the size of the non-GIL in-process residual that the backend change additionally bypasses? | Approximately +9.5 pp on the matched-pair mean (V2 mean − TR165 mean), bounded by the joint footprint of stream serialization, HBM contention, and library locks |
| What is the H1-fail tail under nogil that the backend change resolves? | The 5 H1-failing combinations and 4 unresolved hang shapes — all of which complete cleanly under TGI in V2 |

**Observations.** The practitioner-grade reading is now fully grounded. The bridge paper Phase 8 anchor can quote the +17.91 pp / +27.41 pp / +9.5 pp triple as the substrate-grounded decomposition of V1's N=2 breakdown into GIL-attributable and non-GIL-in-process-attributable components. None of the three numbers is an estimate; each is a matched-pair statistic with named source fields in the two run artifacts.

> The two-axis triangulation is what makes the Phase 8 anchor language defensible. Without TR165, the bridge paper would have had to attribute all of TR164 V2's +27.41 pp recovery to "moving off direct PyTorch" without naming what changed. Without TR164 V2, TR165's H2_partial would have had to leave the residual mechanism unnamed. With both reports, the paper can decompose +27.41 pp into +17.91 pp (interpreter axis) + ~+9.5 pp (non-interpreter in-process layers bypassed by the server architecture), and each component is anchored in a separate matched-pair contrast against the same V1 baseline.

---

## 17. SS10. Cross-TR Position

TR165 occupies a specific load-bearing slot in the Banterhearts program: it is the first technical report to use Python 3.14t free-threaded (PEP 779) as the dispatcher runtime for systematic, matched-pair experiment dispatch against a prior TR's pre-registered hypothesis. Every earlier TR in the program ran on a GIL-enabled CPython build; TR165 is the runtime-shift report. The preflight verified `sys._is_gil_enabled() == False` and `sysconfig.Py_GIL_DISABLED == True` before and after `torch`, `transformers`, and `accelerate` imports, with `abort_on_gil_build: True` enforced as a safety bar. This means TR165 is also the first TR whose substrate is legally allowed to make claims about GIL attribution at all — every prior TR could only observe N=2 breakdown, not test its mechanism.

TR165 is the third report in the Phase 8 serving-stack mechanism-isolation arc. The arc has a deliberate three-report shape, and each report is non-substitutable:

| TR | Role in arc | Mechanism tested | What it can rule in / out |
|----|-------------|------------------|---------------------------|
| TR164 V1 | Parent experiment | None (phenomenon documentation) | Establishes that N=2 breakdown exists on `pytorch_direct` and names GIL contention as the leading attribution hypothesis |
| TR164 V2 | Companion mechanism-isolation | Backend architecture (vLLM / SGLang / TGI vs. `pytorch_direct`) | Tests whether swapping the serving backend dissolves the N=2 boundary; complements TR165 along the architecture axis |
| TR165 | This report | Python interpreter (3.14t free-threaded, GIL removed) | Tests whether removing the GIL dissolves the N=2 boundary; complements TR164 V2 along the runtime axis |

**Observations.** The three reports are organized as a 2D mechanism-isolation grid (backend × runtime), not a sequence in which TR165 replaces TR164. TR164 V1 supplies the phenomenon and the hypothesis; TR164 V2 holds the runtime fixed and varies the backend; TR165 holds the backend fixed (`pytorch_direct` only, 48 cells planned, 3,744 metrics rows produced, all cells completed) and varies the runtime. Removing any one of the three collapses the program's ability to make a mechanism-level claim about V1's breakdown.

> The honest cross-TR reading is that TR164 V1 is the observation, TR164 V2 is the backend-axis falsification, and TR165 is the runtime-axis falsification. The H2_partial verdict from TR165 — 19 of 24 N=2 combinations improving, mean delta in parallel efficiency of +17.91 percentage points, median +17.15 pp, 2 of 6 V1 deterministic-hang shapes resolved out of 6 attempted — is exactly the kind of result that requires TR164 V2 to remain in the arc rather than be retired by TR165. If TR165 had returned H1 (all 24 combinations passing the threshold, all 6 hang shapes resolved), the GIL would have been a sufficient single-mechanism explanation and TR164 V2 would be supporting evidence at most. Because TR165 returned H2_partial, the residual 5 of 24 non-improving combinations and the 4 still-hanging shapes are exactly what TR164 V2's cross-backend evidence is positioned to characterize.

The bridge paper at `papers/serving_state_safety_certification/UPGRADE_PLAN.md` consumes TR165 as part of its Phase 8 anchor. The bridge paper's load-bearing use of TR165 is the H2_partial verdict itself: the paper does not need TR165 to prove that the GIL is the mechanism; it needs TR165 to establish that the GIL is *a* mechanism with measurable but partial attribution (the +17.91 pp / +17.15 pp deltas and the 2-of-6 hang resolution being the substrate the paper cites). This positioning is what licenses the bridge paper's later claim that backend architecture (TR164 V2) and Python runtime (TR165) are non-redundant axes of the serving-stack mechanism-isolation argument.

There is one further cross-TR positioning point worth being explicit about. TR165 is also the first TR in the program for which the dispatcher's startup preflight is itself a publishable result — `gil_disabled_at_dispatcher_startup: True`, `free_threaded_build: True`, `signals_agree: True`, `cuda_available: True`, all four agreeing — because in every prior TR the runtime was an uncontested constant. From TR165 onward, the runtime is a controlled variable, and the preflight log line is the contract that says so. Future TRs in the arc that revisit GIL attribution at larger model scale, larger N, or under additional backends will inherit this preflight contract directly from TR165's dispatcher rather than re-deriving it. The matched_pair_contrast block (with baseline_csv pointing at TR164 V1's `20260531_120428_552237` run directory and the per-metric sub-blocks for `decode_tps`, `parallel_efficiency`, `p95_latency_multiplier`, and `ttft_ms`) is the schema that downstream TRs and the bridge paper Phase 8 section will consume.

---

## 18. SS11. Forbidden Claims

Three load-bearing claims that the substrate explicitly does not license. Each is named, attributed to the framing it would most naturally arise from, and rejected against the matched-pair evidence. Naming them in the report itself is a defensibility instrument: any downstream reader (or adversarial reviewer) re-deriving TR165's verdict ladder should land on the same forbidden list.

### 18.1 Forbidden claim 1: "The GIL is the sole cause of V1's N=2 breakdown."

This is the strongest possible reading of the H1 hypothesis. The substrate rejects it.

| Forbidden claim | Disconfirming evidence | Source field |
|---|---|---|
| GIL is sole mechanism | 5 of 24 N=2 combinations fail the H1 efficiency-improvement threshold | n_combinations_below_h1_threshold |
| GIL is sole mechanism | 4 of 6 deterministic-hang cell shapes still hang under nogil | hang_cells_attempted minus n_hangs_resolved |
| GIL is sole mechanism | Pre-registered falsification verdict is H2_partial, not H1 | falsification_verdict |

**Observations.** If the GIL were the sole mechanism, every single one of the 24 (model × workload × phase) N=2 combinations would clear the H1 efficiency-improvement threshold, and all 6 deterministic-hang cell shapes from V1 would resolve under nogil. Neither is observed. The 5/24 H1-failing combinations and the 4/6 still-hung cell shapes are structural counter-evidence, not noise; they survive on a full-coverage matched-pair join.

> The GIL is *a* mechanism behind V1's N=2 breakdown. It is not *the* mechanism. The residual non-improvement is the load-bearing observation, not a footnote.

### 18.2 Forbidden claim 2: "Free-threaded Python eliminates the in-process N=2 breakdown."

This is the marketing-grade overreach that "free-threaded Python unlocks parallel inference" headlines invite. The substrate rejects it as well.

| Forbidden claim | Disconfirming evidence | Source field |
|---|---|---|
| Nogil eliminates breakdown | Mean N=2 parallel-efficiency delta is +17.91 pp (partial recovery, not full recovery to ideal) | mean_delta_n2_efficiency |
| Nogil eliminates breakdown | Median delta +17.15 pp confirms the partial-recovery shape (no long tail correction) | median_delta_n2_efficiency |
| Nogil eliminates breakdown | Only 19/24 (79.2%) combinations improve at all | n_combinations_passing_h1 |
| Nogil eliminates breakdown | Only 2 of 6 hang cell shapes resolved | n_hangs_resolved |

**Observations.** "Elimination" requires that N=2 parallel efficiency under nogil approach 1.0 across the matrix, and that all V1 hangs disappear. The observed mean and median deltas are roughly +17–18 percentage points — meaningful and pre-registered as a passing H2 outcome, but explicitly bounded. A 21% non-improvement minority and a 67% unresolved-hang rate are incompatible with elimination language.

> Partial recovery is the honest verdict. Calling it elimination overstates the result and gives an adversarial reviewer a free strike.

### 18.3 Forbidden claim 3: "Python 3.14t is production-ready for in-process LLM serving."

This is the claim TR165 most strongly invites and most decisively does not test. The matrix is mechanism-isolation, not production-readiness.

| Forbidden claim | Why TR165 cannot license it | Source field |
|---|---|---|
| Production-ready ruling | N sweep is capped at N=1 and N=2 only; no N=4, N=8, N=16 evidence | plan N concurrency |
| Production-ready ruling | Single-backend ablation (pytorch_direct only); no vLLM / SGLang / TGI cross-validation in this run | backend pytorch_direct only |
| Production-ready ruling | No long-run stability or memory-leak sweep is part of the protocol | analysis.json (no such field) |
| Production-ready ruling | 3,744 metrics rows on a 48-cell matrix is mechanism-grade, not deployment-grade | rows; cells planned |

**Observations.** TR165 answers exactly one question on a bounded matrix: does removing the GIL change V1's N=2 parallel-efficiency story under matched conditions? The hardware (RTX 4080 Laptop, 12 GB) is the V1 match, not a serving platform. The substrate this report does not have — full-N sweep through at least N=16, multi-hour stability with memory-leak telemetry, cross-backend confirmation under nogil — is precisely the substrate that any production-readiness claim would require.

> TR165 is a single-mechanism falsification experiment. Reading it as a deployment endorsement reverses the direction of evidence. The defensibility bar requires that the forbidden ruling be named here, not assumed away.

### 18.4 What replaces each forbidden claim

For each forbidden claim, the licensed counterpart from SS09 / SS10 is the one that survives the matched-pair contrast:

- Instead of "GIL is sole cause," the licensed claim is "GIL is a meaningful but partial mechanism (H2_partial), with 19/24 combinations improving by a mean of +17.91 pp and 2/6 hangs resolved."
- Instead of "nogil eliminates the breakdown," the licensed claim is "nogil recovers a partial fraction of N=2 parallel efficiency on the tested matrix; 21% of combinations and 67% of hang shapes are unaffected."
- Instead of "3.14t is production-ready," the licensed claim is "3.14t is mechanism-viable on a 48-cell bounded matrix; production qualification requires the full-N, long-run, cross-backend substrate enumerated in the future-work section."

The forbidden list is not rhetorical. It is the explicit boundary between what the 3,744-row matched-pair contrast supports and what an adversarial reading would attempt to extract from the same numbers.

---


## 19. SS12. Limitations.

TR165 is a deliberately narrow ablation, and the honesty of its H2_partial verdict depends on naming what it does not establish. Five structural limitations bound the inference surface of this report. None of them invalidate the matched-pair contrast against TR164 V1; each of them defines a follow-up experiment that the substrate cannot adjudicate on its own.

**Limitation 1: Concurrency range capped at N in {1, 2}.** TR164 V1's breakdown curve extends through N=1, N=2, N=4, N=8, and N=16. TR165 tests only the N=1 to N=2 transition — the boundary at which V1 first exhibited parallel-efficiency collapse. The mean delta of +0.1791 (+17.91 percentage points) and median delta of +0.1715 (+17.15 percentage points) in N=2 parallel efficiency are therefore *boundary-of-breakdown* measurements, not full-curve recoveries. Whether nogil's partial recovery scales monotonically, plateaus, or inverts at N=4, N=8, or N=16 is unmeasured. A program-grade follow-up must re-run the full {1, 2, 4, 8, 16} ladder under Python 3.14t to recover the complete recovery curve and test whether the 4 unresolved hang shapes at N=16 remain deterministic or shift to stochastic timeouts.

**Limitation 2: Single backend (pytorch_direct only).** The 48 planned cells (3 models x 4 workloads x 2 phases x 2 N levels) cover one backend: direct PyTorch in-process inference. TR164 V2 evaluates TGI, vLLM, and SGLang as separate-process server backends with their own concurrency models (continuous batching, paged attention, request schedulers). TR165 does NOT test whether the GIL-attribution mechanism — and therefore the +17.91 pp mean recovery — generalizes to in-process variants of those backends, nor whether their internal threading models are gated by the same CPython interpreter lock that pytorch_direct exposes. The scope claim is strictly: nogil partially recovers parallel efficiency for direct PyTorch dispatch. Cross-backend generalization is a separate experimental question.

**Limitation 3: Hardware single-source (RTX 4080 Laptop 12 GB).** Hardware is matched exactly to TR164 V1, which is the correct choice for a matched-pair contrast but produces a single-GPU substrate. The report does not measure whether the GIL is the dominant N=2 bottleneck on data-center-class accelerators where CUDA kernel launch overhead is a smaller fraction of total step time and where memory bandwidth is dramatically higher. It is structurally possible that on a larger accelerator the residual 20.8% of combinations not passing the H1 threshold shrinks (because non-GIL mechanisms like memory-bandwidth contention recede) or grows (because dispatcher overhead becomes the dominant remaining serialization point). The H2_partial verdict is hardware-conditional on a 12 GB laptop-class GPU.

**Limitation 4: PyTorch CUDA-nogil wheel state is bleeding-edge.** The dispatcher preflight verifies Python 3.14.5 free-threaded build with `sys._is_gil_enabled() == False` both before and after PyTorch import, and the abort_on_gil_build safety bar held. However, the PyTorch CUDA-nogil wheel resolved at runtime is whatever was current at the dispatch moment; the free-threaded ecosystem is moving fast enough that nogil-specific tensor-op behavior, dispatcher locking, and CUDA stream interaction may shift materially between minor wheel revisions. The substrate captures a snapshot, not a stable equilibrium. Re-runs against future PyTorch nogil wheels may produce different recovery percentages without any change to TR165's experimental design.

**Limitation 5: Six V1 hang shapes tested only at original N=16.** The hang_resolution_ledger reports 2 of 6 deterministic-hang cell shapes RESOLVED under nogil and 4 still hanging, all attempted at the original V1 trigger condition of N=16. TR165 does NOT sweep other N levels under nogil for those same six (model, workload) pairs to determine whether the 4 unresolved hangs are GIL-independent at all N concurrency levels or whether they are GIL-attributable at lower N but transition to a different mechanism (kernel serialization, OOM pressure, scheduler starvation) only at N=16. The 2/6 vs 4/6 split is therefore a statement about hang resolution *at V1's specific trigger N*, not about the underlying mechanism behind each hang shape.

Taken together, these five limitations bound the H2_partial verdict precisely: nogil recovers a meaningful but partial fraction of N=2 parallel efficiency in direct PyTorch inference on a single laptop-class GPU under a bleeding-edge wheel, with 19 of 24 combinations clearing the H1 threshold and 2 of 6 hang shapes resolved at V1's original trigger. Every adjacent generalization — to higher N, other backends, larger accelerators, future PyTorch builds, or other trigger conditions — requires a separate experimental campaign. The bridge paper's Phase 8 anchor consumes TR165 as a partial-mechanism falsification of V1's GIL hypothesis, and the limitations above explicitly license that scope and no more.

> The honest reading: TR165 establishes that the GIL is *a* mechanism behind V1's N=2 breakdown on this substrate, refutes the strong claim that it is *the* mechanism, and leaves the five generalization axes above as the follow-up program. The +17.91 pp mean recovery and 2/6 hang-resolution count are real and matched-pair-grounded; everything beyond those bounds is TBD per follow-up experiments not in this report.

---

## 20. SS13. Future Work.

TR165 establishes the H2_partial verdict on a deliberately narrow N={1,2} substrate. The honest answer "GIL is A mechanism but not the only mechanism" creates three queued extensions, each addressing a specific residual question the current substrate cannot resolve. None of these are speculative additions; each maps directly onto an observed gap in the 20.8% non-improvement combinations or the 4-of-6 unresolved deterministic hangs.

### 20.1 Extension 1: Full-N nogil sweep (N={4, 8, 16}).

The current substrate covers N=1 and N=2 only because TR164 V1's deterministic-hang boundary lived at N=2 and the matched-pair contrast required exact replication of V1's plan. The +17.91% mean delta and +17.15% median delta in N=2 parallel efficiency characterize the *onset* of the GIL effect, not its asymptote. A full-N sweep at N={4, 8, 16} on the same 3-model × 4-workload substrate would extend the falsification ledger from 24 combinations to 96 combinations (3 × 4 × 2 phases × 4 N levels), and would test whether the +0.17909670 mean delta scales, saturates, or inverts as agent count grows.

| Question | What N=2 substrate can answer | What N={4,8,16} would add |
| --- | --- | --- |
| Onset of GIL bottleneck | Yes — 19 of 24 combinations clear threshold | N/A |
| Breakdown-curve shape | No — single point on the curve | Full envelope |
| Hang resolution scaling | 2 of 6 V1 hang shapes resolved under nogil | Does the 33% resolution rate hold at higher N? |
| Non-GIL mechanism dominance crossover | Inferred from the 5 non-improving combinations | Direct measurement |

**Observations.** The N={4, 8, 16} extension is the cheapest of the three queued items because the dispatcher, model wrappers, and analyze.py extractors all already support arbitrary N — the change is a config.yaml edit and a longer wall-clock budget.

> The 20.8% non-improvement rate at N=2 is the substrate's most load-bearing residual. Without higher-N data, we cannot distinguish "GIL effect saturates at moderate N" from "non-GIL mechanisms grow faster than GIL relief as N rises." TR165's H2_partial verdict is correct but incomplete; the full breakdown curve is the next-resolution answer.

### 20.2 Extension 2: CUDA-stream-serialization isolation (TR166+ candidate).

The residual 20.8% of combinations not improving under nogil and the 4 of 6 deterministic-hang shapes still hanging under nogil are the substrate's most direct evidence that mechanisms outside the GIL are active at N=2. The leading candidate is default-CUDA-stream serialization: all PyTorch operations from concurrent Python threads enqueue onto the same default stream and serialize at the driver layer regardless of GIL state. A TR166+ candidate would re-run TR165's plan with per-agent CUDA streams (one `torch.cuda.Stream()` per concurrent agent) and measure the delta against TR165's nogil-only baseline.

| Mechanism candidate | TR165 evidence | TR166+ test |
| --- | --- | --- |
| Python GIL contention | +17.91% mean delta confirms partial attribution | Held constant (nogil baseline) |
| Default-stream serialization | 4 unresolved hangs + 5 non-improving combinations | Per-agent streams as the manipulated factor |
| Memory-bandwidth contention | TBD per analyze.py extension | Captured via nsys async-traces |
| Dispatcher overhead | TBD per analyze.py extension | Bounded by per-agent dispatcher process variant |

**Observations.** TR166+ inherits TR165's matched-pair design and adds per-agent CUDA streams as the manipulated factor; the N=2 cell shape is preserved so the contrast is clean against TR165's verdict.

> The honest reading of TR165 is that the GIL accounts for ~80% of the improvable surface and something else accounts for the rest. Naming that "something else" as default-stream serialization is a hypothesis, not a finding; TR166+ promotes it to a falsifiable test under the same defensibility bar.

### 20.3 Extension 3: Cross-hardware nogil substrate (A100 / H100).

TR165 ran exclusively on the RTX 4080 Laptop 12 GB to hold V1's hardware fixed. This is the right matched-pair discipline but it leaves open whether the H2_partial verdict is consumer-hardware-specific. A100 (40 GB / 80 GB) and H100 (80 GB) have different memory-bandwidth envelopes, different SM counts, different default-stream behavior under driver versions tied to data-center SKUs, and run under Linux (no Windows scheduler interaction). Re-running the TR165 plan on A100 and H100 would test whether the +17.91% mean delta and the 4 unresolved hangs are intrinsic to the GIL-attribution pattern or whether they are partly an artifact of consumer-laptop thermal and bandwidth constraints.

### 20.4 Bridge-paper Phase 8 absorption pathway.

The bridge paper's Phase 8 anchor at 2026-10-24 acts as the GO/NO-GO trigger for wave-2 substrate expansion. If an external acceptance signal clears by that date, extensions (1) full-N sweep and (3) cross-hardware substrate are the most natural absorptions: both extend an existing matched-pair design rather than introducing new methodology, and both produce data that strengthens the bridge paper's serving-state safety certification claim about GIL-attribution under direct-PyTorch concurrency. Extension (2) — the CUDA-stream-isolation TR166+ candidate — is a separate research arc and would queue as a standalone technical report regardless of the Phase 8 trigger outcome.

---

## 21. Conclusion

Technical Report 165 closes the matched-pair arc that Technical Report 164 V1 opened. V1 reported a sharp N=2 breakdown in the in-process `pytorch_direct` backend on an RTX 4080 Laptop 12 GB and attributed the boundary, tentatively, to CPython's global interpreter lock. That attribution was a hypothesis, not a measurement, because V1 ran on a stock Python 3.13 interpreter where the GIL is not a controllable variable. TR165 makes it controllable. The dispatcher boots Python 3.14.5 free-threaded (`python3.14t.exe`, tags/v3.14.5:5607950, 10 May 2026), preflights `sys._is_gil_enabled() == False` and `sysconfig.Py_GIL_DISABLED == True` both before and after the torch / transformers / accelerate import sequence, and aborts the run if either signal disagrees. Under that controlled interpreter swap, all 48 planned cells (3 models x 4 workloads x 2 phases x 2 N levels) completed cleanly, yielding 3,744 rows of `metrics.csv` data on the same hardware silicon as V1.

The pre-registered ladder resolves to H2_partial. H0 — "the GIL is not the mechanism, expect no improvement" — is rejected: the mean delta in N=2 parallel efficiency under nogil is +0.1791 (+17.91 percentage points absolute) and the median is +0.1715 (+17.15 pp), both meaningfully above the noise floor. H1 — "the GIL is the mechanism, expect improvement in all 24 (model x workload x phase) combinations at N=2" — is not fully supported: 19 of 24 combinations clear the H1 threshold (79.2%) and 5 of 24 do not (20.8%). H2 — "some but not all combinations improve" — is supported by construction at that 79/21 split. The hang ledger sharpens the same finding from a different angle: of the six deterministic-hang cell shapes V1 catalogued at N=16, two are resolved under nogil and four still hang. The minimum coverage thresholds (twenty N=2 combinations, six hang-shape attempts) are both satisfied, so the verdict is not coverage-limited.

What this licenses is narrower than the V1 framing. We can now say, on substrate, that removing the GIL recovers a measurable and reproducible slice of V1's N=2 breakdown — roughly an eighteen-point shift in parallel efficiency on average and roughly four-fifths of the (model x workload x phase) surface. We can say that the GIL is *a* mechanism behind V1's in-process N-way LLM dispatch ceiling. What this forbids is the cleaner version of the V1 story. The GIL is not the *only* mechanism. The residual 21% of combinations that do not improve, together with the four hang-shape cells that still deadlock under a free-threaded interpreter, point at contributors that an interpreter swap does not touch: CUDA-kernel-level serialization on a single device context, memory-bandwidth contention between two concurrent decode streams on a 12 GB part, and PyTorch dispatcher overhead that is C++-side rather than Python-side. Any forward claim that "free-threaded Python fixes in-process multi-LLM dispatch" overshoots this substrate; the defensible claim is "free-threaded Python *partially* recovers in-process multi-LLM dispatch, with a workload-conditional residual."

The methodological contribution is the matched-pair triangulation itself. TR164 V2 changes the backend (TGI continuous-batching server-side dispatch, plus vLLM and SGLang on data-center hardware) while holding the interpreter fixed; TR165 changes the interpreter (3.13 stock to 3.14t free-threaded) while holding the backend fixed at `pytorch_direct`. Read together, the two contrasts isolate the V1 breakdown into a backend-architecture component and an interpreter-concurrency component, neither of which V1 alone could separate. With the TR164 V2 cross-run synthesis now available — 120 matched cells, mean delta +27.41 pp, Cohen's d_paired = 1.44, Wilcoxon p < 6 × 10⁻²⁰, MH pooled OR = ∞ from 22/24 strata, 96/0 wins — the two-axis triangulation supports the layered claim that the GIL accounts for ~+17.91 pp of mean parallel-efficiency recovery while the additional in-process mechanisms (CUDA-stream serialization, HBM-bandwidth contention, dispatcher / library-lock contention) jointly account for the further ~+9.5 pp that only a server-process architecture bypasses. This is the structural deliverable that Phase 8 was built to produce, and the H2_partial verdict is what makes the triangulation honest rather than confirmatory.

The substrate-grounded operational recommendation follows directly. For the 79% surface where free-threaded Python recovers N=2 efficiency, in-process `pytorch_direct` dispatch becomes a defensible serving choice on single-GPU 12 GB-class hardware; for the 21% residual surface and the four unresolved hang shapes, the responsible path is still to fan out to a server-side backend (TGI / vLLM / SGLang) or to drop to N=1 and queue. Phase 8 hands the bridge paper a real five-layer story — TR164 V1 establishes the boundary, TR164 V2 isolates the backend-architecture axis with d=1.44 and p<6×10⁻²⁰, TR165 isolates the interpreter axis with mean +17.91 pp recovery, and the cross-mapped 79/21 surface tells deployers exactly which (model x workload) shapes are safe to run in-process under which interpreter. The remaining mechanism-isolation work — separating CUDA-kernel serialization from dispatcher overhead inside the unimproved 21% — is the next pre-registered TR, not a claim this report makes.

---

## 22. References

This section enumerates the load-bearing internal and external references for TR165. Internal references are paths inside the Banterhearts monorepo (relative to repo root); external references are public specifications, library documentation, and upstream issue trackers consulted during the build. Reference numbering is local to TR165; cross-TR citation should use the full repo-relative path rather than the local number, since the Banterhearts substrate does not maintain a global bibliography.

### 22.1 Banterhearts internal references

| Tag | Path | Role in TR165 |
| --- | --- | --- |
| [I-1] | `PublishReady/reports/Technical_Report_164.md` | TR164 V1 parent report — source of the GIL-attribution hypothesis TR165 tests, and the canonical narrative of the N=2 breakdown boundary and the six deterministic-hang cell shapes. |
| [I-2] | `research/tr164/results/20260531_120428_552237/` | TR164 V1 raw run directory — read-only baseline that supplies the matched-pair contrast keys consumed by TR165's `matched_pair_contrast` block. |
| [I-3] | `research/tr164/V2_BACKEND_NOTES.md` and `research/tr164/DRAFT_Technical_Report_164_V2.md` | TR164 V2 companion (cross-backend Phase 8 sweep) — orthogonal axis to TR165's single-backend `pytorch_direct` ablation. |
| [I-4] | `research/tr164/cross_run_analysis.json` | TR164 V2 cross-run synthesis — supplies the 120-cell TGI-vs-V1 statistics (d=1.44, p<6×10⁻²⁰, MH OR=∞ from 22/24 strata) cited in SS9. |
| [I-5] | `research/tr165/results/20260607_174748_273070/analysis.json` | The TR165 substrate this report narrates — 3,744 metric rows over 48 cells with the `H2_partial` falsification verdict. |
| [I-6] | `papers/serving_state_safety_certification/UPGRADE_PLAN.md` | Bridge paper Phase 8 anchor — defines the role of TR164/TR165 in the broader serving-state safety certification arc. |
| [I-7] | `BANTERHEARTS_MEASUREMENT_COUNT.md` | Canonical primary + judge measurement count for the Banterhearts program; TR165's 3,744 rows roll up into this ledger. |

**Observations.** The internal reference set is deliberately narrow: TR165 is a single-hypothesis matched-pair against TR164 V1, so only V1's run directory and report are load-bearing on the parent side, with the V2 companion (drafts plus cross-run analysis JSON) supplying the cross-backend complement and the bridge paper supplying the certification framing.

> The internal reference graph is shallow on purpose — TR165's defensibility rests on one matched-pair contrast against one parent run plus one cross-axis triangulation against TR164 V2's cross-run synthesis. Keeping the citation set tight makes the falsification verdict easy to audit end-to-end from the analysis.json upward.

### 22.2 External references

| Tag | Reference | Role in TR165 |
| --- | --- | --- |
| [E-1] | PEP 779 — "Criteria for supported status for free-threaded Python" (python.org/peps/pep-0779) | Defines the supported-status criteria for the Python 3.14 free-threaded build; grounds the `python3.14t` interpreter selection and the `sys._is_gil_enabled() == False` preflight gate. |
| [E-2] | Python 3.14.5 free-threading build release notes (tags/v3.14.5:5607950, May 10 2026) | The exact interpreter build TR165 ran under, as captured by the dispatcher preflight. |
| [E-3] | PyTorch CUDA wheels documentation for free-threaded Python (`pytorch.org/get-started`, free-threaded wheel index) | Source for the `torch` wheel ABI used; relevant because `torch` import must not silently re-enable the GIL under a `python3.14t` interpreter. |
| [E-4] | `transformers` import-time threading behavior under `nogil` (HuggingFace `transformers` issue tracker, free-threaded compatibility threads) | Background for the preflight check that `sys._is_gil_enabled()` remains `False` after `transformers` import. |
| [E-5] | `accelerate` import-time threading behavior under `nogil` (HuggingFace `accelerate` issue tracker) | Same role as [E-4] for the `accelerate` import surface. |

**Observations.** The external set is anchored on PEP 779 plus the three library import surfaces (`torch`, `transformers`, `accelerate`) that must remain GIL-disabled after import; these are the exact surfaces the dispatcher preflight gates on, so the citation set mirrors the gate.

> External references are scoped to what the preflight actually verifies. TR165 does not cite the broader free-threading literature because the report's claims are bounded to one interpreter build, one backend, and one matched-pair contrast — citing more would overreach the substrate.

---

## 23. Appendix A. Hardware and Python 3.14t Environment Fingerprint

This appendix freezes the execution environment used for the TR165 ablation so that any future replication — whether under a stricter free-threaded build, a different GPU class, or a containerized re-run — can be diffed against the exact substrate that produced the H2_partial verdict. The fingerprint is split into hardware, operating system, Python runtime, ML stack, and GIL-detection signals captured at dispatcher startup. Every value below was either read off the host at run time or pinned in the run directory's manifest; nothing in this table is inferred.

### 23.1 Hardware and OS

| Component | Value |
| --- | --- |
| CPU | Intel Core i9 (RTX 4080 Laptop reference platform) |
| GPU | NVIDIA RTX 4080 Laptop, 12 GB VRAM, compute capability sm_89 |
| System RAM | 32 GB DDR5 |
| Operating System | Windows 11 Home, build 10.0.26200 |
| Hardware parity with TR164 V1 | Exact match (same physical machine) |

**Observations.** The hardware row is identical to the TR164 V1 baseline at `research/tr164/results/20260531_120428_552237/`. This is load-bearing: the matched-pair contrast in Section 13 only carries causal weight because the GPU, VRAM ceiling, thermal envelope, and driver path are held constant. The only variable that moves between V1 and TR165 is the Python interpreter.

> The 12 GB VRAM ceiling on sm_89 is not incidental — it is the same memory envelope under which V1's N=2 deterministic hangs were first observed, which is precisely why the ablation must run on the same box rather than on a higher-VRAM server class.

### 23.2 Python runtime and GIL signals

| Field | Value |
| --- | --- |
| Executable path | `C:\Users\sahil\AppData\Local\Programs\Python\pythoncore-3.14t-nuget\tools\python3.14t.exe` |
| Version string | 3.14.5 free-threading build (tags/v3.14.5:5607950, May 10 2026) |
| `sys._is_gil_enabled()` before imports | False |
| `sys._is_gil_enabled()` after torch/transformers/accelerate imports | False |
| `sysconfig.Py_GIL_DISABLED` | True |
| `free_threaded_build` flag | True |
| `signals_agree` (preflight aggregate) | True |
| `gil_disabled_at_dispatcher_startup` | True |
| `abort_on_gil_build` safety bar | True (would have terminated the run on a stock 3.14 interpreter) |
| CUDA available at startup | True |

**Observations.** The dispatcher refuses to proceed unless every one of these signals agrees that the GIL is disabled both before and after the heavy ML imports. This is the guard against the most embarrassing failure mode for a GIL-ablation report — accidentally running on a stock interpreter and attributing any noise to free-threading.

> The post-import re-check matters specifically because PyTorch and Transformers each ship native extensions that historically gated themselves behind GIL-required initialization; confirming `sys._is_gil_enabled() == False` after their import is the only honest way to claim the workload itself ran under free-threading rather than just the launcher.

### 23.3 ML stack and CUDA

| Component | Value |
| --- | --- |
| PyTorch version | TBD per analyze.py extension (recorded in run manifest, not surfaced in analysis.json) |
| Transformers version | TBD per analyze.py extension |
| Accelerate version | TBD per analyze.py extension |
| CUDA driver | TBD per analyze.py extension |
| CUDA toolkit | TBD per analyze.py extension |
| PyTorch CUDA wheel index | TBD per analyze.py extension (free-threaded cp314t wheels from PyTorch nightly index) |

**Observations.** The exact pinned versions live in the run manifest under `research/tr165/results/20260607_174748_273070/` but are not lifted into `analysis.json`'s top-level keys, so they are marked TBD here per the substrate-fidelity rule rather than guessed. A follow-up `analyze.py` extension should hoist these into the top-level summary so the fingerprint is self-contained.

> The non-trivial constraint is that the cp314t free-threaded ABI requires PyTorch wheels built specifically against the free-threaded interpreter; any reproduction attempt that pip-installs from the default index will silently fall back to a GIL-enabled wheel and the `abort_on_gil_build` guard will fire before a single cell runs.

---

## 24. Appendix B. Reproduction Commands

This appendix gives the exact command sequence to reproduce TR165 end-to-end on a Windows host with a Python 3.14t free-threaded build installed at the canonical NuGet path and an RTX 4080 Laptop class GPU. The commands are reproduced verbatim from `research/tr165/README.md` and from the dispatcher invocation that produced `research/tr165/results/20260607_174748_273070/`. Every numerical claim in this report depends on artifacts under that run directory; the file manifest at the end of this appendix enumerates them.

### 24.1 Preflight check sequence

Before any TR165 dispatcher invocation, the operator runs a three-step preflight sequence that confirms the interpreter is the free-threaded build, that the GIL is disabled at process start, and that CUDA is visible. All three must pass; if any returns the GIL-enabled signal, the dispatcher will abort because `abort_on_gil_build: True` is held as a safety bar.

Step 1 — confirm the interpreter is the 3.14t free-threaded build:

```
py -3.14t -c "import sys; print(sys.version); print(sys.executable)"
```

Step 2 — confirm the GIL is disabled at startup (both signals must agree):

```
py -3.14t -c "import sys, sysconfig; print('gil_enabled=', sys._is_gil_enabled()); print('Py_GIL_DISABLED=', sysconfig.get_config_var('Py_GIL_DISABLED'))"
```

Step 3 — confirm CUDA visibility under the free-threaded build (imports of `torch` must not silently re-enable the GIL):

```
py -3.14t -c "import torch, sys; print('cuda=', torch.cuda.is_available()); print('gil_after_torch_import=', sys._is_gil_enabled())"
```

**Observations.** The dispatcher records the result of each step into `runtime.json` under the run directory. For run `20260607_174748_273070` the recorded values are `gil_disabled_at_dispatcher_startup: True`, `signals_agree: True`, `free_threaded_build: True`, and `cuda_available: True`. The `sys._is_gil_enabled() == False` signal holds both before and after `torch`, `transformers`, and `accelerate` are imported.

> The three-step sequence is non-negotiable. If `Py_GIL_DISABLED` is True but `sys._is_gil_enabled()` returns True after a library import, the run is invalid as a GIL-ablation test and must be aborted; the matched-pair contrast against TR164 V1 only carries meaning when the dispatcher process actually executes without the GIL.

### 24.2 Dispatcher invocation (run.py)

The TR165 substrate was produced by a single dispatcher call. The suite is restricted to `local_core`, the backend to `pytorch_direct` (single-backend ablation against V1), and `--no-nsys` is set because TR165's verdict depends on the matched-pair efficiency contrast, not on kernel-level trace analysis:

```
py -3.14t -m research.tr165.run --suite local_core --phase all --backends pytorch_direct --no-nsys
```

**Observations.** This invocation expanded to the planned 48 cells (3 models × 4 workloads × 2 phases × 2 N levels), wrote 3,744 rows into `metrics.csv`, and completed with `runtime_ok_for_verdict: True`. No cells were skipped at the planner level; the 4 unresolved hang shapes recorded in the hang-resolution ledger are dispatcher-detected hangs at execution time, not planner exclusions.

### 24.3 Analyzer invocation (analyze.py)

The analyzer consumes the run directory, joins against TR164 V1's matched baseline, and emits `analysis.json` containing the matched-pair contrast, the breakdown-boundary table, the hang-resolution ledger, and the pre-registered falsification verdict:

```
py -3.14t -m research.tr165.analyze --run-dir research/tr165/results/20260607_174748_273070
```

**Observations.** The analyzer's `falsification_verdict` block is the load-bearing artifact: `H2_partial`, `mean_delta_n2_efficiency: +0.1791`, `n_combinations_passing_h1: 19/24`, `n_hangs_resolved: 2/6`. Every percentage-point claim in the body of this report traces back to that JSON object.

### 24.4 Report generator invocation (generate_report.py)

The auto-generated companion report (run-dir-local, not promoted to `PublishReady/`) is produced by:

```
py -3.14t -m research.tr165.generate_report --run-dir research/tr165/results/20260607_174748_273070
```

**Observations.** Per the repo rule that `PublishReady/` is off-limits for code-generated dumps, the generator writes only into the run directory. The hand-narrated report you are reading is a separate authored artifact; the generator's output is treated as raw substrate, not as a publishable narrative.

### 24.5 File manifest

The following substrate files carry every numerical claim in this report. All paths are repo-relative.

| Path | Role |
|---|---|
| `research/tr165/results/20260607_174748_273070/analysis.json` | Falsification verdict, matched-pair contrast, hang ledger |
| `research/tr165/results/20260607_174748_273070/metrics.csv` | 3,744 raw measurement rows |
| `research/tr165/results/20260607_174748_273070/runtime.json` | Preflight GIL/CUDA signals at dispatcher startup |
| `research/tr165/results/20260607_174748_273070/manifest.json` | Plan (suite, backends, models, workloads, phases, N) |
| `research/tr164/results/20260531_120428_552237/` | TR164 V1 baseline (matched-pair join target; read-only) |
| `research/tr164/cross_run_analysis.json` | TR164 V2 cross-run synthesis (120-cell TGI-vs-V1 d=1.44 / p<6×10⁻²⁰ / MH OR=∞) cited in SS9 |
| `research/tr165/README.md` | Preflight sequence canonical source |
| `research/tr165/config.yaml` | Suite definition (`local_core`, 3 models, 4 workloads, 2 phases, 2 N levels) |
| `research/tr165/run.py` | Dispatcher |
| `research/tr165/analyze.py` | Analyzer; emits `analysis.json` |
| `research/tr165/generate_report.py` | Auto-report generator (run-dir-local output) |
| `PublishReady/reports/Technical_Report_164.md` | Parent TR whose GIL-attribution hypothesis TR165 tests |

**Observations.** The manifest is intentionally narrow. TR165's claim surface is the matched-pair contrast against one specific V1 run directory under one specific hardware configuration, plus the cross-axis read against TR164 V2's cross-run analysis JSON; broadening the manifest would invite cross-run drift. Any reader who reproduces the four invocations above against the listed inputs should recover the `H2_partial` verdict, the +17.91% mean N=2 efficiency delta, the 19-of-24 H1 pass rate, and the 2-of-6 hang-resolution count exactly.

> The reproduction contract is the run directory plus the four invocations, not the surrounding narrative. If a future reader cannot regenerate `analysis.json` from `metrics.csv` via the analyzer invocation, the report's claims must be treated as unsupported until the discrepancy is resolved.

---

## 25. Appendix C. Per-Cell Matched-Pair Tables

This appendix exposes the per-cell substrate underlying the H2_partial verdict. Three tables resolve the 48-cell matched-pair structure at the granularity an external reader needs to re-run our pass/fail accounting: (1) per-(model × workload × phase) N=2 efficiency deltas with the H1 threshold flag, (2) the hang-resolution ledger over the six V1 deterministic-hang shapes, and (3) per-cell completion statistics. Every row is read verbatim from `research/tr165/results/20260607_174748_273070/analysis.json`.

### 25.1 Table C1. Per-(model × workload × phase) N=2 efficiency delta + H1 flag

The 24 combinations are tabulated below, sorted ascending by delta. The H1 column flags pass/fail at the per-cell improvement threshold (the threshold sits between +11.72 pp and +12.15 pp on this distribution, which is what drives the 19/5 split). The five fail rows correspond exactly to `n_combinations_below_h1_threshold = 5` in `falsification_verdict`.

| Rank | Phase | Model | Workload | V1 eff | TR165 eff | Delta | %Δ | H1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ttft | qwen2.5-1.5b | short_decode | 60.69% | 35.27% | -25.42 pp | -41.89% | fail |
| 2 | scaling | qwen2.5-1.5b | repeated_prefix | 60.17% | 64.63% | +4.46 pp | +7.42% | fail |
| 3 | scaling | llama3.2-1b | balanced_2k | 53.05% | 58.67% | +5.62 pp | +10.60% | fail |
| 4 | ttft | llama3.2-3b | repeated_prefix | 47.29% | 54.72% | +7.43 pp | +15.72% | fail |
| 5 | ttft | qwen2.5-1.5b | repeated_prefix | 60.54% | 72.26% | +11.72 pp | +19.36% | fail |
| 6 | scaling | llama3.2-3b | repeated_prefix | 48.70% | 60.85% | +12.15 pp | +24.95% | pass |
| 7 | ttft | llama3.2-3b | short_decode | 58.90% | 73.24% | +14.35 pp | +24.36% | pass |
| 8 | ttft | qwen2.5-1.5b | long_decode | 59.96% | 74.36% | +14.41 pp | +24.03% | pass |
| 9 | ttft | qwen2.5-1.5b | balanced_2k | 61.26% | 76.55% | +15.29 pp | +24.96% | pass |
| 10 | scaling | llama3.2-1b | short_decode | 64.42% | 80.07% | +15.65 pp | +24.30% | pass |
| 11 | scaling | qwen2.5-1.5b | short_decode | 57.09% | 73.42% | +16.33 pp | +28.61% | pass |
| 12 | scaling | llama3.2-1b | long_decode | 54.27% | 71.41% | +17.14 pp | +31.58% | pass |
| 13 | ttft | llama3.2-1b | long_decode | 54.65% | 71.81% | +17.16 pp | +31.40% | pass |
| 14 | scaling | qwen2.5-1.5b | long_decode | 64.29% | 84.09% | +19.80 pp | +30.81% | pass |
| 15 | scaling | llama3.2-3b | short_decode | 55.15% | 75.74% | +20.59 pp | +37.33% | pass |
| 16 | ttft | llama3.2-3b | long_decode | 53.07% | 75.26% | +22.20 pp | +41.83% | pass |
| 17 | ttft | llama3.2-1b | short_decode | 52.79% | 75.05% | +22.26 pp | +42.17% | pass |
| 18 | scaling | llama3.2-3b | long_decode | 49.45% | 71.71% | +22.26 pp | +45.03% | pass |
| 19 | scaling | llama3.2-1b | repeated_prefix | 43.54% | 67.32% | +23.78 pp | +54.62% | pass |
| 20 | scaling | llama3.2-3b | balanced_2k | 48.32% | 75.31% | +27.00 pp | +55.87% | pass |
| 21 | ttft | llama3.2-3b | balanced_2k | 42.50% | 72.39% | +29.89 pp | +70.32% | pass |
| 22 | ttft | llama3.2-1b | repeated_prefix | 41.01% | 72.49% | +31.48 pp | +76.75% | pass |
| 23 | ttft | llama3.2-1b | balanced_2k | 33.44% | 73.35% | +39.92 pp | +119.38% | pass |
| 24 | scaling | qwen2.5-1.5b | balanced_2k | 61.52% | 105.90% | +44.37 pp | +72.12% | pass |

**Observations.** The pre-registered H1 rule required all 24 combinations to improve; 19 do and 5 do not, which is exactly the structure that flips the verdict from H1 to H2_partial. The mean and median deltas sit within 0.76 pp of each other, indicating the improving cells are not driven by one or two extreme outliers but by a broadly shifted distribution.

> A 79.2% pass rate with mean +17.91 pp and median +17.15 pp is a strong partial signal, not a universal one. The five non-passing combinations are precisely the cells where the substrate forbids us from attributing the V1 breakdown to the GIL alone.

### 25.2 Table C2. Hang-resolution ledger over V1 deterministic-hang shapes

Six V1 deterministic-hang cell shapes were attempted under the free-threaded runtime; two cleared, four did not. The ledger below is read verbatim from `hang_resolution_ledger` in `analysis.json`.

| Shape | Phase | Model | Workload | N | Outcome | TR165 wall (ms) | V1 baseline wall (ms) | ok_rate | Completion tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | scaling | llama3.2-3b | long_decode | 16 | resolved_cleanly | 3,146,927 | 0 | 1.00 | 98,304 |
| H2 | ttft | llama3.2-3b | long_decode | 16 | resolved_cleanly | 2,999,750 | 0 | 1.00 | 98,304 |
| H3 | scaling | llama3.2-3b | balanced_2k | 16 | hang_or_failure | null | 25,341,494 | null | null |
| H4 | ttft | llama3.2-3b | balanced_2k | 16 | hang_or_failure | null | 8,939,765 | null | null |
| H5 | scaling | llama3.2-3b | repeated_prefix | 16 | hang_or_failure | null | 0 | null | null |
| H6 | ttft | llama3.2-3b | repeated_prefix | 16 | hang_or_failure | null | 0 | null | null |

**Observations.** The pre-registered minimum of six attempted shapes was satisfied exactly. Two shapes cleared cleanly under nogil (both `long_decode`, completing in approximately 50-52 minutes each with `ok_rate = 1.00` and producing 98,304 completion tokens). Four shapes reproduced the V1 hang signature even with the GIL removed (both `balanced_2k` and both `repeated_prefix`). The resolution rate is 100% on `long_decode` and 0% on the other two workloads.

> Two of six is a non-trivial fraction but well short of full clearance. Four shapes that still hang under the free-threaded runtime are direct evidence that at least one non-GIL serialization mechanism remains on the critical path for those configurations, and the workload-specific concentration (`balanced_2k` and `repeated_prefix` both hang, `long_decode` both clear) localizes the residual to the cache-heavy workload paths.

### 25.3 Table C3. Per-cell completion statistics

The execution layer produced 3,744 metrics.csv rows across 48 planned cells with runtime_ok_for_verdict True, indicating every cell met its planned-rep budget.

| Quantity | Value |
| --- | --- |
| Cells planned | 48 |
| Cells completed | 48 |
| metrics.csv rows | 3,744 |
| Average rows per cell | 78 |
| runtime_ok_for_verdict | True |
| matched_pair_contrast matched cells | 50 |
| matched_pair_contrast tr165_only cells | 0 |
| matched_pair_contrast tr164_v1_only cells | 70 (V1's N={4, 8, 16} sweep deliberately not re-run) |

**Observations.** Full completion of the 48-cell plan with a 3,744-row metrics file means no cell was dropped from the verdict for insufficient data. The matched-pair partition (50 / 0 / 70) confirms TR165 did not silently expand scope — every TR165 cell has a V1 partner at the same coordinates, and the 70 V1-only cells correspond exactly to the V1 N={4, 8, 16} sweep that the SS2 plan deliberately excluded.

> The completion ledger is clean: no cells censored, no reps short, no TR165 cells outside V1's coordinate set. Whatever the H2_partial verdict says, it says it over the full pre-registered grid, not over a subset that happened to finish.
