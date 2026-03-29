# Banterhearts Technical Reports
## LLM Inference Research & Safety Alignment — 593,134+ Primary Measurements, 39 Technical Reports

This directory contains the complete research program documenting LLM inference performance, optimization, multi-agent orchestration, cross-language analysis, deployment policy, and safety alignment under inference optimizations — spanning consumer hardware (NVIDIA RTX 4080 Laptop GPU, 12 GB VRAM) and cloud GPUs (NVIDIA RTX PRO 6000 Blackwell, 98 GB VRAM via Google Colab).

---

## Research Program Overview

### Phase 1: Foundation (TR108-TR122)
**15 technical reports. ~8,000 benchmark runs.**

Single-agent and multi-agent performance analysis, Rust vs Python cross-language evaluation, ONNX/TensorRT backend exploration, cost/energy analysis, the compile paradox root-cause audit, model scaling study, and resource profiling.

### Phase 2: Deployment Framework (TR123-TR133)
**11 technical reports. ~62,000+ measurements.**

KV-cache economics, quality baselines, quantization decision matrix, Linux/Triton compile validation, long-context characterization, production workload analysis, N-agent scaling laws, serving stack comparison, GPU kernel profiling (host + in-container), and a predictive capacity planner shipped as the `chimeraforge plan` CLI.

### Phase 3: Safety Alignment (TR134-TR143) + v2 Expansion
**10 technical reports + 3 v2 expansion reports. ~399,000+ data points.**

Alignment robustness under quantization, multi-agent concurrency safety, cross-backend safety consistency, the safety tax synthesis, batch inference safety under non-determinism (+ strengthened-evidence revision), multi-turn jailbreak susceptibility under quantization, many-shot and long-context jailbreak, cross-architecture refusal fragility (largest study: 18 models, 10+ families, 152,022 data points), quality-safety correlation, and cross-request safety leakage under continuous batching (TR143).

### Conclusive Reports
**12 synthesis documents spanning all phases.**

Five conclusive reports (TR108-TR116, TR117-TR122, TR123-TR133, TR134-TR137, TR138-TR143), five extended appendices volumes, and five executive whitepapers — providing decision guidance with artifact provenance.

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
| **TR125** | Quantization Decision Matrix | 33,810 | Complete | Q4_K_M remains the quality default across 7 models; Q2_K is universally unacceptable |
| **TR126** | Docker/Linux + Triton Validation | 25,400 | Complete | Compile paradox resolved: 24-60% prefill speedup on Linux; crashes decode |
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
| **TR134** | Alignment Robustness Under Quantization | 38,120 + 24,336 judge | Complete | Safety is broadly robust through Q3_K_S, qwen2.5-1.5b replicates the Q2_K cliff, and judge gaps are strongly model-dependent |
| **TR135** | Safety Under Multi-Agent Concurrency | 20,316 | Complete | Null finding confirmed: concurrency has zero detectable effect on safety (I-squared = 0.0%) |
| **TR136** | Cross-Backend Safety Consistency | 16,032 | Complete | Backend matters more than quant: Llama 1B shows 23pp safety drop Ollama→FP16; no TOST equivalence |
| **TR137** | The Safety Tax of Inference Optimization | 74,254 | Complete | Quantization 57% of safety cost, backend 41%, concurrency 2%; worst config retains only 57.5% baseline safety |
| **TR138** | Batch Inference Safety Under Non-Determinism | 31,410 | Complete | Batch non-determinism produces 0.6% automated flip rate (0.16% human-adjudicated genuine); 73% of automated detections are regex artifacts |
| **TR138 v2** | Batch Safety — Strengthened-Evidence Revision | 7,257 | Complete | Audit layer confirms 59.1% unsafe flip direction; replication yields 1.68% safety vs 0.42% capability flip rate |
| **TR139** | Multi-Turn Jailbreak Under Quantization | 48,425 | Complete | All 8 strategy ANOVAs reject quant-independence (p < 1e-4); qwen2.5-1.5b/Q2_K/attention_shift reaches 100% ASR |
| **TR140** | Many-Shot & Long-Context Jailbreak Under Quantization | 30,000 | Complete | Llama immune above Q3_K_M; Q2_K universal vulnerability threshold; message array format 92% vs 0% faux dialogue |
| **TR141** | Cross-Architecture Refusal Fragility Under Batch Perturbation | 152,022 | Complete (v3.1) | 18 models, 10+ families; 0.94x safety/capability ratio (near parity); alignment type not predictive (p=0.942); output instability predicts fragility (r=0.91) |
| **TR142** | Quality-Safety Correlation Under Quantization | 40 cells; 33,810 quality + 38,120 safety + 24,336 judge | Complete | Sign reversal persists across the expanded 6-model matrix; pooled quality metrics remain unreliable safety proxies |
| **TR143** | Cross-Request Safety Leakage Under Continuous Batching | 14,250 | Complete (v2.0) | Aggregate composition effect not significant; directional asymmetry IS significant — 88-92% of flips trend unsafe (p=0.006) |

### Expansion v2 Reports (TR142 Matrix Expansion)

| Report | Title | Samples | Status | Key Finding |
|--------|-------|---------|--------|-------------|
| **TR125 v2** | Quality Evaluation — Expanded Matrix | 8,820 expansion + 24,990 original | Complete | 7 models across 4 families; Q4_K_M sweet spot confirmed cross-family; mistral-7b MMLU 58.9%→55.1% at Q2_K |
| **TR134 v2** | Safety Alignment — Expanded Matrix | 13,342 expansion + 24,778 original; 12,168 gemma3 judge | Complete | 6 models across 4 families; Q2_K catastrophe replicates on qwen2.5-1.5b (-50pp); Mistral regex-judge gap up to 71pp; dual-judge (qwen2.5-7b + gemma3:12b) |
| **TR142 v2** | Quality-Safety Correlation — 6-Model Synthesis | 40 cells from TR125+TR134 expanded | Complete | 34/36 sign reversals (Simpson's paradox at scale); 26/34 cells safety degrades faster; Q5_K_M floor holds all 6 models; per-model r from +0.997 to -0.829 |

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
| **Conclusive 138-143** | Phase 3.5/4 Synthesis (Safety Attack Surface) | 2,497 lines, 306,996 samples |
| **Conclusive 138-143 Extended Appendices** | Phase 3.5/4 Deep-Dive Appendices | 722 lines |
| **Conclusive 138-143 Whitepaper** | Phase 3.5/4 Executive Guidance | 253 lines |

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

Six shippable decisions backed by ~62,000 measurements:

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
- Phase 4 explicit true-batch validates effect is not a timing artifact

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
- 6 matched models across 3 families, 40 model-quant cells, 14 core analysis passes plus supporting diagnostics
- Sign reversal persists in the expanded matrix: 34/36 quality-safety pairings split positive and negative across models
- Safety degrades 13.9x faster than quality at llama3.2-1b Q3_K_S, while qwen2.5-7b Q2_K reproduces the hidden-danger pattern outside Llama
- Q5_K_M remains the conservative floor; Q4_K_M is still model-dependent and ambiguous

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
- Decode compile crashes 100% of the time in all tested modes
- PyTorch bug discovered and reported upstream (pytorch/pytorch#175557, PR #175562)

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

### Phase 1a Synthesis: Technical_Report_Conclusive_108-116
**File:** `Technical_Report_Conclusive_108-116.md` (2,826 lines)
- Covers TR108 through TR116: Python-to-Rust migration, multi-agent architecture, runtime selection, cross-model validation
- Six shippable decisions: Rust for production, dual Ollama mandatory, Tokio-default runtime, Gemma 3 for scaling, Python ceiling at ~86%, config transfer failure
- Extended Appendices: `Technical_Report_Conclusive_108-116_Extended_Appendices.md` (1,171 lines)
- Executive Whitepaper: `Technical_Report_Conclusive_108-116_Whitepaper.md` (214 lines)

### Phase 1b Synthesis: Technical_Report_Conclusive_117-122
**File:** `Technical_Report_Conclusive_117-122.md` (208KB)
- Covers TR117 through TR122
- Extended Appendices: `Technical_Report_Conclusive_117-122_Extended_Appendices.md` (89KB)
- Executive Whitepaper: `Technical_Report_Conclusive_117-122_Whitepaper.md` (8KB)

### Phase 2 Synthesis: Technical_Report_Conclusive_123-133
**File:** `Technical_Report_Conclusive_123-133.md` (433KB, 3,327 lines, 60 appendices)
- Covers TR123 through TR133
- Three stable conclusions, six shippable decisions, full artifact provenance
- Extended Appendices: `Technical_Report_Conclusive_123-133_Extended_Appendices.md` (62KB)
- Executive Whitepaper: `Technical_Report_Conclusive_123-133_Whitepaper.md` (15KB)

### Phase 3a Synthesis: Technical_Report_Conclusive_134-137
**File:** `Technical_Report_Conclusive_134-137.md` (2,571 lines)
- Covers TR134 through TR137: quantization-induced alignment erosion, concurrency invariance, backend-driven template divergence, cross-axis safety taxonomy
- 74,254 evaluated samples, 24-configuration deployment risk matrix (3 CRITICAL, 3 moderate, 18 low)
- Key finding: serving backend choice is a previously uncharted safety variable (41% of safety cost)
- Extended Appendices: `Technical_Report_Conclusive_134-137_Extended_Appendices.md` (908 lines)
- Executive Whitepaper: `Technical_Report_Conclusive_134-137_Whitepaper.md` (228 lines)

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
├── Phase 3: Safety Alignment (TR134-TR143)
│   ├── Technical_Report_134_v2.md
│   ├── Technical_Report_135.md
│   ├── Technical_Report_136.md
│   ├── Technical_Report_137.md
│   ├── Technical_Report_138.md
│   ├── Technical_Report_138_v2.md
│   ├── Technical_Report_139.md
│   ├── Technical_Report_140.md
│   ├── Technical_Report_141.md  ← largest study (152,022 data points, 18 models)
│   ├── Technical_Report_142_v2.md
│   └── Technical_Report_143.md
│
├── Conclusive Reports
│   ├── Technical_Report_Conclusive_108-116.md
│   ├── Technical_Report_Conclusive_108-116_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_108-116_Whitepaper.md
│   ├── Technical_Report_Conclusive_117-122.md
│   ├── Technical_Report_Conclusive_117-122_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_117-122_Whitepaper.md
│   ├── Technical_Report_Conclusive_123-133.md
│   ├── Technical_Report_Conclusive_123-133_Extended_Appendices.md
│   ├── Technical_Report_Conclusive_123-133_Whitepaper.md
│   ├── Technical_Report_Conclusive_134-137.md
│   ├── Technical_Report_Conclusive_134-137_Extended_Appendices.md
│   └── Technical_Report_Conclusive_134-137_Whitepaper.md
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
   Prefill only, Linux only, 24-60% speedup; decode crashes 100% of the time (TR126)

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

---

## Reading Guide

### For Researchers
1. Start with **TR108-TR110** (Python baselines)
2. Study **TR112_v2** and **TR114_v2** (Rust vs Python)
3. Review Phase 2 starting with **TR123** (economics) through **TR133** (capacity planning)
4. Review Phase 3 starting with **TR134** (quantization x safety) through **TR142** (quality-safety correlation)
5. Read the **Conclusive Reports** for synthesis and cross-TR analysis

### For Engineers
1. **Single-agent deployment:** TR112_v2 (language choice) + TR125 (quantization)
2. **Multi-agent deployment:** TR129 (scaling laws) + TR130 (serving stacks)
3. **Capacity planning:** TR133 + `chimeraforge plan` CLI
4. **Compilation:** TR126 (what works, what crashes)
5. **Safety-critical deployment:** TR137 (safety tax synthesis) + TR141 (batch safety, cross-architecture)
6. **Backend safety validation:** TR136 (backend safety consistency)
7. **Jailbreak risk assessment:** TR139 (multi-turn) + TR140 (many-shot)

### For Decision Makers
1. **Rust vs Python Decision:** `Technical_Report_Conclusive_108-116_Whitepaper.md` (language, architecture, runtime, model)
2. **Phase 2 Whitepaper:** `Technical_Report_Conclusive_123-133_Whitepaper.md` (15KB, 6 decisions)
3. **Phase 3 Whitepaper:** `Technical_Report_Conclusive_134-137_Whitepaper.md` (228 lines, safety decision card)
4. **Decision Matrix:** See Phase 2 Deployment Decisions table above
5. **Cost Analysis:** TR123 ($/token) + TR119 (energy/carbon)

---

**Last Updated:** 2026-03-28
**Total Reports:** 63 files (36 completed TRs + 12 conclusive/whitepaper documents + 7 historical/superseded + 3 legacy + TR143)
**Total Measurements:** 558,804+ primary measurements across report sample columns; secondary judge annotations and synthesis-layer matrix cells are reported separately within the relevant reports
