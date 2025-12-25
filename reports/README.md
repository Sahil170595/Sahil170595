# Chimeraforge Technical Reports

## Comprehensive LLM Performance Research & Cross-Language Analysis

This directory contains the complete research journey documenting LLM performance analysis, optimization strategies, multi-agent orchestration, and cross-language (Rust vs Python) performance evaluation in the Chimeraforge research project.

> **Note:** This is the **Chimeraforge** research repository. All technical reports are based on research conducted in this repository.

---

## 📊 Research Journey Overview

### Phase 1: Single-Agent Foundation (TR108-TR109)

**Objective:** Establish baseline LLM performance and agent workflow optimization

- **TR108:** Single-agent LLM performance analysis (Python)
- **TR109:** Agent workflow optimization (Python)

### Phase 2: Multi-Agent Python Baseline (TR110)

**Objective:** Establish Python multi-agent performance baseline

- **TR110:** Concurrent multi-agent performance analysis (Python)

### Phase 3: Rust Single-Agent Implementation (TR111)

**Objective:** Port Python agents to Rust and validate single-agent performance

- **TR111:** Initial Rust agent (micro-benchmark) - **SUPERSEDED**
- **TR111_v2:** ✅ **Production-grade Rust single-agent** (full workflow parity)

### Phase 4: Cross-Language Single-Agent Comparison (TR112)

**Objective:** Direct Rust vs Python single-agent performance comparison

- **TR112:** Initial comparison (flawed - used micro-benchmark) - **SUPERSEDED**
- **TR112_v2:** ✅ **Comprehensive Rust vs Python comparison** (production-grade)

### Phase 5: Rust Multi-Agent Implementation (TR113-TR114)

**Objective:** Port multi-agent orchestration to Rust and compare to Python

- **TR113:** Rust multi-agent (single Ollama) - identified server contention issue
- **TR114:** Rust multi-agent (dual Ollama) - initial analysis - **SUPERSEDED**
- **TR114_v2:** ✅ **Comprehensive Rust multi-agent** (dual Ollama, corrected statistics)

### Phase 6: Runtime Optimization (TR115)

**Objective:** Optimize Rust async runtime for multi-agent workloads

- **TR115:** Initial runtime analysis - **SUPERSEDED**
- **TR115_v2:** ✅ **Definitive runtime optimization** (5 runtimes, 150 runs)

### Phase 6.5: Cross-Model Multi-Agent Analysis (TR116)

**Objective:** Isolate model choice as independent variable in multi-agent performance

- **TR116:** ✅ **Cross-model benchmarks** (Qwen 2.5 vs Gemma 3 vs Llama 3.1, Rust vs Python, 60 runs)

### Phase 6.6: Multi-Agent Root-Cause Analysis (TR117 Multi-Agent)

**Objective:** Root-cause audit of Python ceiling, Qwen mystery, and ranking flip anomalies

- **TR117 Multi-Agent:** ✅ **Root-cause analysis** (event loop instrumentation, hardware forensics, flow dynamics)

### Phase 7: Cross-Backend Inference Benchmarking (TR117)

**Objective:** Compare local-first inference backends and identify production baselines

- **TR117:** ✅ **Cross-backend inference benchmark** (PyTorch eager/compile, Ollama; ONNX/TRT failures documented)

### Phase 8: ONNX Runtime + TensorRT Deep Dive (TR118)

**Objective:** Fix ONNX/TRT infrastructure and produce publishable, local-first results

- **TR118:** ✅ **ONNX/TRT deep dive** (test fixture model: 0.1M params, forensic document)
- **TR118v2.1:** ✅ **Model scale comparative analysis** (0.1M vs 124M params, 1,210× scaling study, corrected)
- **TR118v2.2:** ✅ **Definitive comparative analysis** (measurement rigor + pipeline validation)

### Phase 9: Cost & Energy Analysis (TR119)

**Objective:** Translate latency/throughput into dollars, kWh, and carbon per token

- **TR119:** ✅ **Cost & energy analysis** (v1.1, multi-tier pricing, carbon footprint)
- **TR119v1:** ✅ **Comprehensive cost/energy deep dive** (1,290 lines, artifact-backed, business impact)

### Phase 10: Compile Paradox Investigation (TR120)

**Objective:** Root-cause audit of why compile improves mean but degrades median latency

- **TR120:** ✅ **Compile paradox root-cause audit** (1,101 lines, controlled reproduction, production fixes)

### Phase 11: Model Scaling Study (TR121)

**Objective:** How inference behavior changes from ~0.1M to ~20.9B parameters

- **TR121:** ✅ **Model scaling study pipeline** (pipeline-complete scaffold)
- **TR121v1:** ✅ **Comprehensive scaling analysis** (1,231 lines, three regimes, mechanistic insights, capacity planning)

---

## 📋 Technical Reports Index

### ✅ **Current Production-Ready Reports (v2)**

| Report | Title | Status | Key Finding |
|--------|-------|--------|-------------|
| **TR108** | Single-Agent LLM Performance Analysis | ✅ Complete | Optimal configs for single-agent inference |
| **TR109** | Agent Workflow Optimization | ✅ Complete | GPU=60, CTX=512, TEMP=0.8 optimal for workflows |
| **TR110** | Concurrent Multi-Agent Performance (Python) | ✅ Complete | 99.25% parallel efficiency achieved |
| **TR111_v2** | Rust Single-Agent Performance | ✅ Complete | 114.54 tok/s baseline, 15.2% faster than Python |
| **TR112_v2** | Rust vs Python Comparison | ✅ Complete | Rust: +15.2% throughput, -58% TTFT, -67% memory |
| **TR114_v2** | Rust Multi-Agent Performance | ✅ Complete | 98.281% mean efficiency, 99.992% peak run |
| **TR115_v2** | Rust Runtime Optimization | ✅ Complete | Tokio-default recommended (98.72% mean, 1.21pp σ) |
| **TR116** | Cross-Model Multi-Agent Benchmarks | ✅ Complete | Gemma 3 champion (97.3% Rust), Rust +12-17pp vs Python |
| **TR117** | Cross-Backend Inference Benchmark | ✅ Complete | GPU-compile best mean; ONNX/TRT failures documented |
| **TR117 Multi-Agent** | Multi-Agent Root-Cause Analysis | ✅ Complete | Python ceiling (86%), Qwen mystery, ranking flip explained |
| **TR118** | ONNX Runtime vs TensorRT Deep Dive | ✅ Complete | Publish-ready prefill benchmark + TRT INT8 calibration + perplexity gate |
| **TR118v2.1** | Model Scale Comparative Analysis | ✅ Complete | 1,210× scaling (0.1M → 124M), crossover validation, empirical refutation |
| **TR118v2.2** | Definitive Comparative Analysis | ✅ Complete | Measurement rigor + pipeline validation, comprehensive statistical analysis |
| **TR119** | Cost & Energy Analysis | ✅ Complete | Multi-tier pricing, carbon footprint, token economics (v1.1) |
| **TR119v1** | Cost & Energy Deep Dive | ✅ Complete | Comprehensive analysis (1,290 lines), business impact, TCO projections |
| **TR120** | Compile Paradox Root-Cause Audit | ✅ Complete | Controlled reproduction, shape stability insights, production guidance |
| **TR121** | Model Scaling Study Pipeline | ✅ Complete | Pipeline infrastructure, decode sweep, boundary shift experiments |
| **TR121v1** | Comprehensive Scaling Analysis | ✅ Complete | Three regimes, mechanistic insights, capacity planning (1,231 lines) |

### 📚 **Historical Reports (Superseded)**

| Report | Status | Superseded By | Reason |
|--------|--------|---------------|--------|
| TR111 | ❌ Superseded | TR111_v2 | Micro-benchmark → Full workflow |
| TR112 | ❌ Superseded | TR112_v2 | Flawed comparison methodology |
| TR114 | ❌ Superseded | TR114_v2 | Incorrect statistics (97.5% → 98.281%) |
| TR115 | ❌ Superseded | TR115_v2 | Incomplete data analysis (30 → 150 runs) |

---

## 🎯 Key Research Findings

### **Single-Agent Performance (TR111_v2, TR112_v2)**

**Rust Advantages:**

- **Throughput:** 114.54 tok/s vs Python 99.34 tok/s (**+15.2%**)
- **TTFT (cold):** 603ms vs Python 1437ms (**-58%**)
- **Memory:** ~75 MB vs Python ~250 MB (**-67%**)
- **Startup:** 0.2s vs Python 1.5s (**-83%**)
- **Consistency:** 2.6% CV vs Python 4.8% CV (**46% more consistent**)

**Verdict:** Rust dominates single-agent performance across all metrics.

### **Multi-Agent Performance (TR110, TR114_v2)**

**Python Baseline (TR110):**

- **Peak Config Efficiency:** 99.25% (homogeneous Chimera)
- **Mean Efficiency:** 95.8% (all 150 runs)
- **Contention Rate:** 10-15%

**Rust Performance (TR114_v2):**

- **Peak Config Efficiency:** 99.396% (chimera-hetero test011)
- **Peak Single Run:** 99.992% (chimera-homo test108)
- **Mean Efficiency:** 98.281% (all 135 runs) - **+2.48pp vs Python**
- **Contention Rate:** 0.74% - **-10-14pp vs Python**

**Verdict:** Rust **exceeds** Python in multi-agent scenarios (+2.48pp mean, +0.15pp peak config).

### **Runtime Optimization (TR115_v2)**

**Runtime Ranking (By Consistency):**

1. **Tokio-default:** 98.72% mean, 1.21pp σ 🏆 **RECOMMENDED**
2. **Smol-1KB:** 98.61% mean, 1.32pp σ ✅ **Alternative (smaller binary)**
3. **Tokio-localset:** 97.95% mean, 4.03pp σ ⚠️ **Unstable (18.96pp range)**
4. **Smol:** 97.72% mean, 4.87pp σ ❌ **Pathological failures (72.80% min)**
5. **Async-std:** 50.00% ❌ **Catastrophic (Tokio bridge conflict)**

**Verdict:** Use **standard Tokio** (`#[tokio::main]`) - no custom configuration needed.

---

## 📈 Performance Summary Matrix

### **Single-Agent Comparison (TR112_v2)**

| Metric | Python | Rust | Rust Advantage |
|--------|--------|------|----------------|
| **Throughput** | 99.34 tok/s | 114.54 tok/s | **+15.2%** |
| **TTFT (cold)** | 1437 ms | 603 ms | **-58.0%** |
| **Memory** | ~250 MB | ~75 MB | **-67%** |
| **Startup** | 1.5s | 0.2s | **-83%** |
| **CV (throughput)** | 4.8% | 2.6% | **-46% (more consistent)** |

### **Multi-Agent Comparison (TR110 vs TR114_v2)**

| Metric | Python (TR110) | Rust (TR114_v2) | Rust Advantage |
|--------|---------------|-----------------|----------------|
| **Peak Config Avg** | 99.25% | 99.396% | **+0.15pp** |
| **Peak Single Run** | 99.25% | 99.992% | **+0.74pp** |
| **Mean Efficiency** | 95.8% | 98.281% | **+2.48pp** |
| **Median Efficiency** | ~96.5% | 98.6% | **+2.1pp** |
| **StdDev** | 7.4pp | 4.9pp | **-2.5pp (more stable)** |
| **Contention Rate** | 10-15% | 0.74% | **-10-14pp** |

### **Runtime Performance (TR115_v2)**

| Runtime | Peak (%) | Mean (%) | StdDev (pp) | Min (%) | Recommendation |
|---------|----------|----------|-------------|---------|----------------|
| **Tokio-default** | 99.89 | **98.72** | **1.21** | 94.80 | ✅ **Production** |
| **Smol-1KB** | 99.94 | 98.61 | 1.32 | 94.98 | ✅ **Small binary** |
| **Tokio-localset** | 99.99 | 97.95 | 4.03 | 81.03 | ⚠️ **Unstable** |
| **Smol** | 99.87 | 97.72 | 4.87 | 72.80 | ❌ **Avoid** |
| **Async-std** | 50.00 | 50.00 | 0.00 | 50.00 | ❌ **Unusable** |

---

## 🔬 Report Details

### **TR108: Single-Agent LLM Performance Analysis**

**File:** `Technical_Report_108.md`

- **Focus:** Comprehensive LLM performance benchmarking and optimization
- **Models:** gemma3:latest, llama3.1:8b-instruct variants
- **Hardware:** NVIDIA RTX 4080 (12GB VRAM), i9-13980HX
- **Test Matrix:** 150+ benchmark runs across parameter sweeps
- **Key Findings:** Optimal configurations for single-agent inference
- **Status:** ✅ Complete (Publication-ready)

### **TR109: Agent Workflow Optimization**

**File:** `Technical_Report_109.md`

- **Focus:** Agent workflow performance vs single-inference optimization
- **Methodology:** Process isolation, forced cold starts, statistical validation
- **Key Findings:** Agent tasks require different optimization than single-inference
- **Optimal Config:** GPU=60, CTX=512, TEMP=0.8 for agent workflows
- **Quality Analysis:** Automated scoring methodology for report quality
- **Status:** ✅ Complete (Publication-ready)

### **TR110: Concurrent Multi-Agent Performance Analysis (Python)**

**File:** `Technical_Report_110.md`

- **Focus:** Parallel agent execution with resource coordination
- **Test Matrix:** 30 configurations × 5 runs = 150 benchmark runs
- **Key Findings:** 99.25% parallel efficiency achieved with homogeneous Chimera agents
- **Scenarios:** Baseline vs Chimera, Heterogeneous, Homogeneous configurations
- **Resource Analysis:** VRAM utilization, memory bandwidth saturation, contention patterns
- **Status:** ✅ Complete (Publication-ready)

### **TR111_v2: Rust Single-Agent Performance Analysis**

**File:** `Technical_Report_111_v2.md`

- **Focus:** Comprehensive Rust agent workflow performance with full Python parity
- **Test Matrix:** 19 configurations × 3 runs = 57 benchmark runs
- **Key Findings:**
  - Baseline: 114.54 tok/s (15.2% faster than Python)
  - TTFT: 547-1354ms range (high variance, context-dependent)
  - Optimal: `gpu60_ctx256_temp0p6` (115.94 tok/s)
- **Critical Discovery:** TTFT shows 150× more variation than throughput
- **Status:** ✅ Complete (Supersedes TR111)

### **TR112_v2: Rust vs Python Agent Performance Comparison**

**File:** `Technical_Report_112_v2.md`

- **Focus:** Cross-language comprehensive comparison with workflow parity
- **Test Matrix:** 37 configurations (19 Rust + 18 Python), 111 total runs
- **Key Findings:**
  - Rust: +15.2% throughput, -58% TTFT, -67% memory, -83% startup
  - Rust: 46% more consistent (2.6% vs 4.8% CV)
  - Business Impact: ~$3,040/year savings, 20-month break-even
- **Status:** ✅ Complete (Supersedes TR112)

### **TR113: Rust Multi-Agent Initial Analysis**

**File:** `Technical_Report_113.md`

- **Focus:** First Rust multi-agent implementation (single Ollama instance)
- **Key Finding:** 82.2% peak efficiency, 63% contention rate
- **Critical Discovery:** Server-level serialization bottleneck identified
- **Recommendation:** Dual Ollama architecture required
- **Status:** ✅ Complete (Historical - led to TR114)

### **TR114_v2: Rust Concurrent Multi-Agent Performance**

**File:** `Technical_Report_114_v2.md`

- **Focus:** Comprehensive Rust multi-agent with dual Ollama architecture
- **Test Matrix:** 27 configurations × 5 runs = 135 benchmark runs
- **Key Findings:**
  - Peak single run: 99.992% (test108)
  - Best config avg: 99.396% (test011 chimera-hetero)
  - Mean efficiency: 98.281% (+2.48pp vs Python)
  - Contention rate: 0.74% (-10-14pp vs Python)
- **Critical Discovery:** Rust **exceeds** Python in multi-agent scenarios
- **Status:** ✅ Complete (Supersedes TR114)

### **TR115_v2: Rust Async Runtime Performance Deep Dive**

**File:** `Technical_Report_115_v2.md`

- **Focus:** Comprehensive multi-runtime analysis for multi-agent workloads
- **Test Matrix:** 5 runtimes × 6 configs × 5 runs = 150 benchmark runs
- **Key Findings:**
  - All 4 working runtimes achieve ~100% peak (99.87-99.99%)
  - Consistency matters more: Tokio-default (1.21pp σ) recommended
  - Async-std unusable (50% efficiency, Tokio bridge conflict)
  - Smol has pathological failures (72.80% min efficiency)
- **Production Recommendation:** Use standard `#[tokio::main]` - no custom config needed
- **Status:** ✅ Complete (Supersedes TR115)

### **TR116: Cross-Model Multi-Agent Benchmarks**

**File:** `Technical_Report_116.md`

- **Focus:** Cross-model analysis (Qwen 2.5 vs Gemma 3 vs Llama 3.1) in multi-agent workloads
- **Test Matrix:** 60 multi-agent runs across 6 model-runtime combinations (3 models × 2 runtimes × 2 scenarios × 5 runs)
- **Key Findings:**
  - **Rust Dominates:** +12-17pp higher efficiency than Python across all models
  - **Gemma 3 Champion:** 97.3% efficiency (Rust), 99.2% in chimera-homo (approaching theoretical max)
  - **Qwen Coordination Overhead:** 90.0% efficiency (Rust) vs 97.3% for Gemma, suggesting heavier KV cache
  - **Python Efficiency Ceiling:** Never exceeds 86% efficiency, while Rust hits 90-99%
  - **Model Choice Matters More in Python:** 6.2pp spread (77.6-83.8%) vs Rust's 7.3pp (90.0-97.3%)
- **Business Impact:**
  - **Optimal Stack:** Rust + Gemma 3 (best efficiency)
  - **Cost Implications:** Python + Gemma 3 = +24% cost, Python + Qwen = +33% cost
- **Status:** ✅ Complete (12+ hours of testing)

### **TR117 Multi-Agent: Root-Cause Analysis of Multi-Agent Efficiency Anomalies**

**File:** `Technical_Report_117_multi_agent.md`

- **Focus:** Root-cause audit of three critical anomalies from TR116: Python Ceiling, Qwen Mystery, Ranking Flip
- **Methodology:** Three invasive research phases:
  - **Phase 1:** Python MRI (Event Loop Instrumentation)
  - **Phase 2:** Hardware Forensics (GPU/PCIe Profiling)
  - **Phase 3:** Flow Dynamics (Artificial Throttling)
- **Key Findings:**
  - **Python Ceiling Explained:** Event loop saturation (5.33ms mean lag, p99 12.13ms) from per-token JSON parsing at ~200 events/sec
  - **Qwen Mystery:** Software-bound (tokenizer overhead, chunk amplification), not PCIe saturation
  - **Ranking Flip:** Slower models provide "breathing room" for event loop, but throttling reduces efficiency
  - **TR110 Retroactive Explanation:** 99.25% efficiency achieved with slower models (~68 tok/s) in "Goldilocks Zone"
- **Production Recommendations:**
  - **For Python (Mitigation):** Token bucket throttling, batch HTTP chunk processing, use `uvloop`
  - **For Production (Migration):** Rust mandatory for >100 tok/s multi-agent systems
- **Status:** ✅ Complete (8+ hours of invasive instrumentation)

### **TR117: Cross-Backend Inference Benchmark**

**File:** `Technical_Report_117.md`

- **Focus:** Compare local-first inference backends (PyTorch eager/compile, Ollama, ONNX Runtime, TensorRT)
- **Test Matrix:** 3,017 runs, 2,471 successful (82%)
- **Key Findings:**
  - GPU-compile wins on mean latency (389ms)
  - Plain GPU wins on median latency (323ms)
  - ONNX/TRT reliability gaps identified (18% degraded)
- **Hardware:** NVIDIA RTX 4080 (12GB VRAM), i9-13980HX
- **Artifacts:** `results/tr117_tier3/` (metrics.csv, plots, analysis)
- **Status:** ✅ Complete (with known limitations addressed in TR118)

### **TR118: ONNX Runtime + TensorRT Deep Dive (Test Fixture)**

**File:** `Technical_Report_118.md`

- **Focus:** Validate ONNX/TRT pipeline with test fixture model (0.1M params)
- **Test Matrix:** 360 run-level records (180/mode, 0% degraded)
- **Key Findings:**
  - Pipeline validated: ONNX export, TRT engine builds (FP32/FP16/INT8), perplexity gates
  - ONNX CPU: 97× faster than PyTorch for tiny models
- **Status:** ✅ Complete (Forensic document - superseded by TR118v2.1/v2.2)

### **TR118v2.1: Model Scale Comparative Analysis**

**File:** `Technical_Report_118_v2.1.md`

- **Focus:** Definitive comparative analysis of ONNX Runtime and TensorRT performance scaling
- **Test Matrix:** 360 benchmarks: 2 models × 6 backends × 6 scenarios × 5 repetitions
- **Models:** `sshleifer/tiny-gpt2` (0.103M params) vs `gpt2` (124.4M params)
- **Key Findings:**
  - **The Crossover Phenomenon:** ONNX CPU goes from 21.9× faster → 0.70× slower as model scales (0.1M → 124M params)
  - **TensorRT Perfect Scaling:** Consistent 60-75% speedup across 1,210× parameter increase
  - **INT8 Reality Check:** 20% slower than FP16 for 124M params (not faster as expected)
  - **Perplexity Preservation:** All backends < 0.022% delta (production-ready accuracy)
- **Status:** ✅ Complete (1,210× scaling study, crossover validation)

### **TR118v2.2: Definitive Comparative Analysis**

**File:** `Technical_Report_118_v2.2.md`

- **Focus:** Measurement rigor + pipeline validation with comprehensive statistical analysis
- **Key Findings:**
  - Complete measurement definitions (latency, throughput, formulas)
  - Evidence-based explanations (TRT degradation root cause)
  - Full statistical rigor (t-tests, confidence intervals, effect sizes)
  - Comprehensive appendices (detailed performance tables, perplexity analysis)
- **Status:** ✅ Complete (1,325 lines, frontier-grade depth)

### **TR119: Cost & Energy Analysis**

**File:** `Technical_Report_119.md` (v1.1)

- **Focus:** Translate latency/throughput into dollars, kWh, and carbon per token
- **Key Findings:**
  - **Best overall:** `onnxruntime-gpu` at $0.1279 / 1M tokens (prefill mean)
  - **Best generate:** `onnxruntime-gpu` at $1.204 / 1M tokens
  - **Best request cost:** `onnxruntime-gpu` at $0.0001475 per request
  - **Business impact:** ~$7.1k/year savings vs transformers-gpu, ~$57.8k/year vs transformers-cpu
  - **Energy share:** Small (0.33-1.7% of total cost), infra cost dominates
- **Methodology:**
  - Multi-tier pricing: On-demand, spot, reserved
  - Carbon footprint: Energy × carbon intensity
  - Telemetry-backed: GPU power via NVML, CPU package power via Windows Energy Meter
- **Status:** ✅ Complete (v1.1)

### **TR119v1: Cost & Energy Deep Dive**

**File:** `Technical_Report_119v1.md`

- **Focus:** Comprehensive cost/energy analysis with telemetry-backed calculations and business impact
- **Key Findings:**
  - **Production-focused question:** "Which backend minimizes dollars per token and energy per token?"
  - **Multi-tier pricing:** On-demand, spot, reserved pricing comparisons
  - **Carbon footprint:** Energy × carbon intensity calculations
  - **Business impact:** Scaled TCO projections (1B tokens/month for 12 months)
  - **Statistical rigor:** ANOVA, pairwise comparisons, confidence intervals
- **Status:** ✅ Complete (1,290 lines, publish-ready)

### **TR120: Compile Paradox Root-Cause Audit**

**File:** `Technical_Report_120.md`

- **Focus:** Root-cause audit of why `torch.compile()` improves mean but degrades median latency
- **Key Findings:**
  - **TR117's compile paradox is misattributed:** Label didn't actually invoke `torch.compile()`
  - **Shape instability:** Variable prompt lengths trigger repeated recompilations
  - **Padding/bucketing fix:** Collapses compiled tail (p99: multi-ms → sub-ms)
  - **KV-cached decode:** Different regime - Inductor improves prefill but regresses decode
  - **Production guidance:** Gate compile on Triton, stabilize shapes, split modes
- **Methodology:**
  - Controlled reproduction with explicit `torch.compile()` vs eager
  - Multiple artifact sets (Windows fallback, Triton Docker runs)
  - Compiler evidence recording (Dynamo counters, CUDA-event timing)
- **Status:** ✅ Complete (1,101 lines, artifact-backed)

### **TR121: Model Scaling Study Pipeline**

**File:** `Technical_Report_121.md`

- **Focus:** Pipeline infrastructure for scaling study (5M to 20B parameters)
- **Key Findings:**
  - Pipeline infrastructure complete
  - Decode sweep infrastructure
  - Boundary shift experiments
- **Status:** ✅ Pipeline Complete (superseded by TR121v1)

### **TR121v1: Comprehensive Scaling Analysis**

**File:** `Technical_Report_121v1.md`

- **Focus:** How latency, throughput, and cold-start risk change from ~0.1M to ~20.9B parameters
- **Test Matrix:** Primary run `scripts/tr121/results/20251224_002149/` (684 records), decode sweep, Gemma3 family check, boundary shift runs
- **Key Findings:**
  - **Three distinct regimes:**
    1. Small-model GPU (0.1M-96M): Weak scaling (R² ~0.03-0.06) - not identifiable
    2. CPU scaling (0.1M-96M): Predictable (prefill: ~0.281, decode: ~0.291, R² ~0.85-0.87)
    3. Large-model serving (268M-20.9B): Strong scaling (prefill: ~0.623, decode: ~0.649, R² ~0.90-0.93)
  - **Mechanistic insight:** GPU latency tracks `n_layer` (depth) more than parameter count
  - **Decode dominance:** Decode fraction reaches 0.98-0.99 by 64-128 tokens
  - **Cold-start risk:** Warmup ratios up to 194x (HF GPU prefill)
  - **Business impact:** Capacity planning and shadow-priced cost analysis
- **Methodology:**
  - Phase-split measurements: Prefill, KV-cached decode, end-to-end
  - Two-family study: HF local (5M-124M) + Ollama (270M-20B)
  - Scaling law fits: Log-log power law per backend/mode/scenario
  - Architecture correlation analysis: Spearman correlations for depth/width/params
- **Status:** ✅ Complete (1,231 lines, publish-ready)

---

## 🎓 Research Evolution & Key Insights

### **The Journey: From Python Baseline to Rust Excellence**

1. **TR108-TR109 (Python Foundation):** Established optimal single-agent and workflow configurations
2. **TR110 (Python Multi-Agent):** Achieved 99.25% parallel efficiency baseline
3. **TR111_v2 (Rust Single-Agent):** Discovered Rust's 15.2% throughput advantage
4. **TR112_v2 (Cross-Language):** Quantified Rust's comprehensive advantages (performance, memory, startup)
5. **TR113 (Rust Multi-Agent v1):** Identified server contention bottleneck (single Ollama)
6. **TR114_v2 (Rust Multi-Agent v2):** Proved Rust **exceeds** Python in multi-agent (+2.48pp mean)
7. **TR115_v2 (Runtime Optimization):** Established Tokio-default as optimal runtime
8. **TR116 (Cross-Model):** Isolated model choice impact - Gemma 3 champion, Rust +12-17pp vs Python
9. **TR117 Multi-Agent (Root-Cause):** Explained Python ceiling (86%), Qwen mystery, ranking flip

### **Critical Discoveries**

1. **Single-Agent:** Rust is **15.2% faster** than Python (TR112_v2)
2. **Multi-Agent:** Rust **exceeds** Python by +2.48pp mean efficiency (TR114_v2)
3. **Runtime:** Standard Tokio achieves 98.72% mean with 1.21pp σ (TR115_v2)
4. **Architecture:** Dual Ollama **mandatory** for multi-agent (reduces contention 63% → 0.74%)
5. **Model Choice:** Gemma 3 champion (97.3% Rust), Rust +12-17pp vs Python across all models (TR116)
6. **Python Ceiling:** Event loop saturation caps Python at 86% efficiency (TR117 Multi-Agent)
7. **Consistency:** Rust's lower variance (4.9pp vs 7.4pp) provides production reliability

### **The "Paradox" That Wasn't**

**Initial Hypothesis (TR113/TR114):** Rust's single-agent advantage would disappear in multi-agent scenarios due to coordination overhead.

**Actual Reality (TR114_v2):** Rust's advantages **carry over** to multi-agent:

- Single-agent: +15.2% throughput
- Multi-agent: +2.48pp mean efficiency, +0.15pp peak config
- **Conclusion:** No paradox - Rust maintains and extends advantages

---

## 🚀 Production Recommendations

### **Single-Agent Deployment**

**Choose Rust if:**

- ✅ Production reliability required
- ✅ Resource efficiency critical (67% less memory)
- ✅ Fast startup needed (83% faster)
- ✅ Consistent performance valued (46% lower variance)

**Choose Python if:**

- ✅ Rapid prototyping needed
- ✅ Development velocity prioritized
- ✅ Ecosystem richness required

**Verdict:** **Rust for production** (15% faster, 67% less memory, 83% faster startup)

### **Multi-Agent Deployment**

**Architecture:**

- ✅ **Dual Ollama instances** (mandatory - eliminates server contention)
- ✅ **Heterogeneous configs** optimal (asymmetric GPU allocation)
- ✅ **Tokio-default runtime** (best consistency: 1.21pp σ)

**Configuration:**

- **Best Config (TR114_v2):** GPU=120/140, CTX=512/1024 (test011) → 99.396% efficiency
- **Balanced Config:** GPU=80, CTX=1024 (test004) → 98.984% efficiency
- **Avoid:** GPU=120 in baseline-vs-chimera (test005: 91.60% with contention)

**Verdict:** **Rust multi-agent** (98.281% mean vs Python 95.8%, 0.74% contention vs 10-15%)

### **Runtime Selection (TR115_v2)**

**Production:**

```rust
// Use standard Tokio - no custom configuration needed
#[tokio::main]
async fn main() {
    let (r1, r2) = tokio::join!(agent_a(), agent_b());
}
```

**Why:**

- ✅ Highest consistency: 1.21pp σ (vs 4.03pp for localset)
- ✅ Best mean efficiency: 98.72% (vs 97.95% for localset)
- ✅ Simplest deployment: No custom runtime configuration
- ✅ Best ecosystem: Native reqwest, no bridges

**Alternatives:**

- **Smol-1KB:** If binary size <5MB critical (98.61% mean, 1.32pp σ, -0.11pp loss acceptable)
- **Never use:** Async-std (50% efficiency), Smol (pathological failures)

---

## 💰 Business Impact Summary

### **Infrastructure Savings (TR112_v2)**

**Single-Agent:**

- Memory: 67% reduction (75 MB vs 250 MB)
- Startup: 83% faster (0.2s vs 1.5s)
- **Cost:** ~$3,040/year savings at 1M requests/month
- **Break-even:** 20 months ($5k dev overhead)

### **Multi-Agent Advantages (TR114_v2)**

**Performance:**

- Mean efficiency: +2.48pp (98.281% vs 95.8%)
- Contention rate: -10-14pp (0.74% vs 10-15%)
- Consistency: -2.5pp StdDev (4.9pp vs 7.4pp)

**Cost:**

- 50% lower infrastructure cost (67% less memory per agent)
- 3× concurrent capacity (lower memory footprint)
- Reduced contention = fewer failed requests

### **Runtime Optimization (TR115_v2)**

**Impact:**

- Runtime choice: Marginal (<1% between best options)
- **Async-std cost:** 50% efficiency = **2× infrastructure cost** (avoid!)
- **Tokio-default cost:** Same as Python (both ~99% efficiency)

**Strategic Insight:** Standard Tokio requires zero custom configuration, achieving Python parity with Rust's resource advantages.

---

## 📁 Repository Structure

```
PublishReady/reports/
├── README.md (this file)
│
├── ✅ Production-Ready Reports (v2)
│   ├── Technical_Report_108.md - Single-agent LLM performance (Python)
│   ├── Technical_Report_109.md - Agent workflow optimization (Python)
│   ├── Technical_Report_110.md - Multi-agent performance (Python)
│   ├── Technical_Report_111_v2.md - Rust single-agent performance
│   ├── Technical_Report_112_v2.md - Rust vs Python comparison
│   ├── Technical_Report_114_v2.md - Rust multi-agent performance
│   ├── Technical_Report_115_v2.md - Rust runtime optimization
│   ├── Technical_Report_116.md - Cross-model multi-agent benchmarks
│   ├── Technical_Report_117.md - Cross-backend inference benchmark
│   ├── Technical_Report_117_multi_agent.md - Multi-agent root-cause analysis
│   ├── Technical_Report_118_v2.1.md - Model scale comparative analysis
│   ├── Technical_Report_118_v2.2.md - Definitive comparative analysis
│   ├── Technical_Report_119.md - Cost & energy analysis (v1.1)
│   ├── Technical_Report_119v1.md - Cost & energy deep dive (comprehensive)
│   ├── Technical_Report_120.md - Compile paradox root-cause audit
│   ├── Technical_Report_121.md - Model scaling study pipeline (superseded)
│   └── Technical_Report_121v1.md - Comprehensive scaling analysis
│
├── 📚 Historical Reports (Superseded)
│   ├── Technical_Report_111.md - Initial Rust (micro-benchmark)
│   ├── Technical_Report_112.md - Flawed comparison
│   ├── Technical_Report_113.md - Rust multi-agent (single Ollama)
│   ├── Technical_Report_114.md - Rust multi-agent (incorrect stats)
│   └── Technical_Report_115.md - Incomplete runtime analysis
│
└── 📊 Model Benchmarks
    └── gemma3/
        └── Gemma3_Benchmark_Report.md
```

---

## 🔗 Report Relationships

```
TR108 (Python Single-Agent)
    ↓
TR109 (Python Workflow Optimization)
    ↓
TR110 (Python Multi-Agent Baseline)
    ↓
TR111_v2 (Rust Single-Agent) ──→ TR112_v2 (Rust vs Python)
    ↓                                    ↓
TR113 (Rust Multi-Agent v1)      TR114_v2 (Rust Multi-Agent v2)
    ↓                                    ↓
TR114_v2 (Dual Ollama)            TR115_v2 (Runtime Optimization)
```

**Key Dependencies:**

- TR111_v2 → TR112_v2 (single-agent comparison)
- TR110 → TR114_v2 (multi-agent comparison)
- TR111_v2/TR112_v2/TR114_v2 → TR115_v2 (baseline references)
- TR113 → TR114_v2 (identified dual Ollama requirement)
- TR114_v2/TR115_v2 → TR116 (cross-model multi-agent analysis)
- TR116 → TR117 Multi-Agent (root-cause analysis of anomalies)
- TR117 → TR118 → TR118v2.1 → TR118v2.2 (ONNX/TRT pipeline evolution)
- TR117 → TR119 → TR119v1 (cost/energy analysis)
- TR117 → TR120 (compile paradox investigation)
- TR117/TR118/TR120 → TR121 → TR121v1 (scaling study)

---

## 📖 Quick Reference

### **Best Configurations**

**Single-Agent (TR111_v2):**

- **Optimal:** `gpu60_ctx256_temp0p6` (115.94 tok/s)
- **Baseline:** Ollama defaults (114.54 tok/s)

**Multi-Agent (TR114_v2):**

- **Best Config:** GPU=120/140, CTX=512/1024 (test011) → 99.396% efficiency
- **Balanced:** GPU=80, CTX=1024 (test004) → 98.984% efficiency

**Runtime (TR115_v2):**

- **Production:** Tokio-default (`#[tokio::main]`)
- **Small Binary:** Smol-1KB (if size critical)

### **Performance Targets**

**Single-Agent:**

- Rust: 114.54 tok/s baseline, 115.94 tok/s optimal
- Python: 99.34 tok/s baseline

**Multi-Agent:**

- Rust: 98.281% mean, 99.396% peak config, 99.992% peak run
- Python: 95.8% mean, 99.25% peak config

**Runtime:**

- Tokio-default: 98.72% mean, 1.21pp σ (recommended)
- Smol-1KB: 98.61% mean, 1.32pp σ (alternative)

---

## 🎯 Research Questions Answered

1. **Q: Is Rust faster than Python for LLM inference?**  
   **A: Yes - 15.2% faster throughput, 58% faster TTFT, 67% less memory (TR112_v2)**

2. **Q: Does Rust's single-agent advantage carry over to multi-agent?**  
   **A: Yes - Rust exceeds Python by +2.48pp mean efficiency (TR114_v2)**

3. **Q: Which Rust async runtime is optimal for multi-agent workloads?**  
   **A: Tokio-default - 98.72% mean, 1.21pp σ (best consistency) (TR115_v2)**

4. **Q: Is dual Ollama architecture necessary?**  
   **A: Yes - Reduces contention from 63% to 0.74% (TR113 → TR114_v2)**

5. **Q: What is the optimal multi-agent configuration?**  
   **A: Heterogeneous GPU allocation (120/140) with asymmetric context (512/1024) → 99.396% efficiency (TR114_v2)**

---

## 📊 Statistical Validation

All reports include:

- ✅ **Statistical rigor:** Mean, median, stddev, percentiles, CV
- ✅ **Multiple runs:** 3-5 runs per configuration for confidence
- ✅ **Comprehensive coverage:** 19-37 configurations per report
- ✅ **Cross-validation:** Results validated across multiple reports
- ✅ **Reproducibility:** Full methodology and data sources documented

**Total Benchmark Runs:**

- TR108: 150+ runs
- TR109: 90 runs
- TR110: 150 runs
- TR111_v2: 57 runs
- TR112_v2: 111 runs
- TR114_v2: 135 runs
- TR115_v2: 150 runs
- TR116: 60 runs (multi-agent cross-model)
- TR117: 3,017 runs (2,471 successful)
- TR117 Multi-Agent: 8+ hours (invasive instrumentation)
- TR118: 360 runs
- TR118v2.1: 360 runs
- TR118v2.2: 360 runs
- TR120: 546+ runs
- TR121v1: 684+ runs (primary) + decode sweep + boundary shifts
- **Total: 6,000+ benchmark runs** across all reports

---

## 🚀 Getting Started

### **For Researchers**

1. Start with **TR108** (Python single-agent baseline)
2. Review **TR109** (workflow optimization methodology)
3. Study **TR110** (Python multi-agent baseline)
4. Compare **TR112_v2** (Rust vs Python single-agent)
5. Analyze **TR114_v2** (Rust multi-agent performance)
6. Reference **TR115_v2** (runtime optimization guidance)
7. Study **TR116** (cross-model multi-agent analysis)
8. Investigate **TR117 Multi-Agent** (root-cause analysis)
9. Study **TR117** (cross-backend inference benchmark)
10. Review **TR118v2.2** (ONNX/TRT definitive analysis)
11. Analyze **TR119v1** (cost & energy deep dive)
12. Investigate **TR120** (compile paradox root-cause)
13. Explore **TR121v1** (comprehensive scaling analysis)

### **For Engineers**

1. **Single-Agent:** Read **TR112_v2** for Rust vs Python comparison
2. **Multi-Agent:** Read **TR114_v2** for deployment guidance
3. **Model Selection:** Read **TR116** for cross-model multi-agent performance
4. **Python Issues:** Read **TR117 Multi-Agent** for Python ceiling root-cause
5. **Runtime:** Read **TR115_v2** for production recommendations
6. **Backend Selection:** Read **TR117** and **TR118v2.2** for inference backend comparison
7. **Cost Optimization:** Read **TR119v1** for cost/energy analysis
8. **Compile Behavior:** Read **TR120** for torch.compile() guidance
9. **Scaling Planning:** Read **TR121v1** for capacity planning and model selection
10. **Configuration:** Use best configs from TR111_v2 (single) and TR114_v2 (multi)

### **For Decision Makers**

1. **Executive Summary:** Review "Key Findings" sections in each report
2. **Business Impact:** See "Business Impact" sections in TR112_v2, TR114_v2, TR119v1, TR121v1
3. **Cost Analysis:** Review break-even analysis in TR112_v2, cost projections in TR119v1, capacity planning in TR121v1
4. **Backend Selection:** Review TR117, TR118v2.2 for production backend recommendations
5. **Scaling Strategy:** Review TR121v1 for model size selection and capacity planning
6. **Recommendations:** See "Production Recommendations" in this README

---

## 📝 Report Status Legend

- ✅ **Complete (Publication-ready):** Fully validated, production-grade analysis
- 📚 **Historical (Superseded):** Replaced by v2 with corrected methodology/data
- 🔬 **In Progress:** Currently being developed

---

**Last Updated:** 2025-12-24  
**Repository:** Chimeraforge (Research)  
**Maintainer:** Chimeraforge Research Team  
**Total Reports:** 22+ (19+ production-ready, 5+ historical)  
**Total Benchmark Runs:** 6,000+ across all reports
