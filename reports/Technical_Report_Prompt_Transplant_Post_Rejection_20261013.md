# Technical Report: Prompt Transplant After Rejection

## Native-consumer replication, current-guard screening, and the limits of a streaming-serving pivot

**Evidence cutoff:** October 13, 2026, after completion of `streaming-equivalence-e5c0570afa1e`.

**Document type:** Authored technical report. This is not a replacement submission manuscript, an acceptance forecast, or a deployment certification.

**Status:** The final held-out experiment executed successfully and failed its scientific admission criterion. All 64 paired comparisons completed. One released-prefix decision differed; consequently, none of the 24 conditional scheduler replays executed. The coordinator reports that the application closed normally and its owned GPU resources are no longer active.

**Privacy boundary:** This report contains no participant identities, credentials, private sharing URLs, or raw harmful examples. Artifact paths are relative to the private source repository. Linked evidence identifiers resolve to the source inventory in Appendix A; they do not publish the underlying data.

---

## Abstract

The post-rejection Prompt Transplant program tested whether a documented safety-judge integration defect could support a broader contribution about deployed consumers, training-data selection, or streaming inference. The strongest completed replication passed real requests through a native Redpanda processor and an actual Ollama model. On the previously used JailbreakBench reference cohort, correcting the rendered evaluation target reduced false positives from 67/190 to 20/190, while false negatives remained 13/110. This establishes a consequence at a native consumer boundary, not a new defect discovery or an observed customer deployment. A subsequent factorial behavioral probe measured changed request and response sensitivity but did not yield the proposed clean black-box target identification. A separate retrospective calibration analysis found no candidate passing its combined sensitivity and false-alarm screen across the tested comparisons.

Current-guard experiments then screened a possible training-data mechanism. Granite Guardian 4.1 and Qwen3Guard rarely rejected source-labeled safe refusals, weakening the proposed blanket refusal-removal mechanism. They rejected substantially more source-labeled safe helpful responses to harmful requests. An exact-request follow-up reproduced this contrast, but its source labels were model-generated and independent human adjudication remains absent. The multimodal arm classified every tested recommended response as safe; without harmful-response controls, that result cannot establish discrimination. No student model was trained, so there is no measured downstream training effect.

The final pivot evaluated a prerequisite for streaming-serving research: whether larger guard increments preserve the exact release decision. A development profile found a 20.90-fold effective guard-service reduction for 32-token increments, but the original exact-token-label gate failed. A separately frozen test on 64 previously unused held-out examples measured a 22.19-fold aggregate guard-service reduction and one exact released-prefix disagreement. Both implementations eventually rejected the differing response, but the larger increment released one additional token overlapping the source annotation. This is neither a demonstrated whole-response bypass nor an independently verified actionable harmful prefix. It is a counterexample to the chosen exact-equivalence requirement. The gated scheduling experiment correctly did not proceed. The completed work therefore supplies bounded native-consumer evidence, a candidate selective-filtering mechanism, and a measured release-contract failure, but not a validated training intervention or joint GPU scheduling method.

## Executive summary

The work made progress by replacing several plausible stories with measured boundaries. It did not establish that adding experiments alone turns the inherited audit into a new systems contribution.

| Question | Completed evidence | Conclusion at the cutoff |
|---|---|---|
| Does the known role-designation defect affect a real native consumer? | 600 valid native processor decisions on 300 paired examples | Yes, on the pinned implementation and reused cohort; no customer deployment or new defect discovery is established. |
| Can a small factorial probe cleanly identify the black-box target? | 600 valid decisions across 75 quartets and two implementations | It measures sensitivity changes; the corrected implementation remains inconclusive under the largest frozen probe budget. |
| Does request-conditioned calibration solve the observed tradeoff? | Retrospective analysis of 36,693 admitted scores | No candidate passes the combined exploratory screen in any of the 72 candidate-baseline contrasts. |
| Do current guards systematically remove safe refusals from training? | 389 source-safe refusal records in a 1,024-record screen | Not in this screen: rejection is rare for both tested guards. |
| Is there a more selective safe-help filtering issue? | 94 focal examples, exact-request refusal controls, and repeated native scoring | There is a reproducible label-dependent contrast worth adjudicating; genuine false-positive harm and downstream training consequences are not established. |
| Does the multimodal result show useful discrimination? | 195 recommended responses, two modalities and two targets | No. All response decisions are safe, and the arm lacks harmful-response positive controls. |
| Can chunked streaming guard calls preserve exact release decisions? | 64 new held-out pairs, chunk 1 versus chunk 32 | Not exactly on this cohort: one released-prefix disagreement. |
| Has a release-aware scheduler outperformed FIFO or EDF? | Three partial one-token FIFO runs; zero new gated queue runs | No completed scheduler comparison exists. |

The practical result is not a universal repair. Native target rendering should be corrected and regression-tested where the defect exists. Calibration should be judged at a matched operating point rather than by one favorable error rate. Training claims require verified supervision and an actual student intervention. Streaming optimizations require a release-policy comparison before their speed is treated as equivalent service.

### Principal decisions supported by the evidence

- Keep the native-consumer result as a bounded replication of an already known integration defect.
- Do not describe the factorial probe as a universally identifying black-box test.
- Do not launch a refusal-removal training study on the assumption that current guards frequently remove safe refusals; the screen did not support that premise.
- Keep the safe-help cases as candidates for independent adjudication, not confirmed human gold.
- Do not claim multimodal safety discrimination from the all-safe response arm.
- Preserve both failed streaming admission gates. Neither is a software launch failure, and neither can be retrospectively relabeled as a pass.
- Do not claim a scheduler speedup, live-generator improvement, or verified harmful-content bypass from these results.

## Contents

1. [Scope, evidence standards, and authoring method](#section-1)
2. [The rejection record and the actual chronology](#section-2)
3. [What was carried forward from Prompt Transplant](#section-3)
4. [Phase I: Native Redpanda consumer replication](#section-4)
5. [Phase II: Factorial behavioral diagnosis](#section-5)
6. [Phase III: Conditional calibration without new model calls](#section-6)
7. [Phase IV: Current-guard training-impact screening](#section-7)
8. [Phase V: Multimodal preparation and native screening](#section-8)
9. [Phase VI: Exact-request safe-help follow-up](#section-9)
10. [Why the research direction changed](#section-10)
11. [Phase VII: The original streaming pilot](#section-11)
12. [Phase VIII: Held-out release-decision equivalence](#section-12)
13. [Cross-phase interpretation and review objections](#section-13)
14. [Production implications and boundaries](#section-14)
15. [What would constitute the next contribution](#section-15)
16. [Reproducibility and compute accounting](#section-16)
17. [Limitations and conclusion](#section-17)
18. [Appendix A: Evidence-linked source inventory](#appendix-a)
19. [Appendix B: Pinned models, datasets, and freezes](#appendix-b)
20. [Appendix C: Terms that must not be conflated](#appendix-c)

<a id="section-1"></a>
## 1. Scope, evidence standards, and authoring method

### 1.1 The outcome this report documents

This report reconstructs the post-rejection research program from its source artifacts. It answers four questions: what was actually run, how individual observations became reported results, which proposed explanations survived those results, and what remains necessary for a distinct contribution. It does not attempt to rehabilitate an administrative rejection by adding scientific results, and it does not equate a longer experiment list with greater novelty.

The word "post-rejection" needs a date boundary. There are two different rejection histories. The JUDGe archive records an author-reported rejection and two scientific reviews, but no official decision was visible at the archive's capture. The archival security-venue record inspected on October 12 concerns deanonymization through a supplementary hosting page. Most newly executed work documented here is concentrated on October 12-13. Earlier experiments retained in the current manuscript are identified as carried-forward evidence rather than silently counted as new work.

The report follows the structure and evidentiary expectations of `Technical_Report_126.md`: explicit measurement units, full data-generation paths, phase-level interpretation, negative findings, deployment boundaries, and reproducibility. The source-first gap analysis found that the experiment documents were individually detailed but did not yet provide a coherent account of how successive negative findings changed the research hypothesis. The missing synthesis was therefore causal and methodological, not an absence of status tables. This report supplies that synthesis without modifying the underlying experiments. [E00](#e00)

### 1.2 Evidence levels

Four levels are kept separate throughout.

**Recorded measurement.** A count, timing, verdict, confidence interval, or status appears in a named run artifact. This is the basis for quantitative claims.

**Implementation evidence.** A pinned source path, rendered template, parser, or data-selection rule explains what the experiment does. Source inspection can establish a reachable mechanism; it cannot establish deployment prevalence or customer impact.

**Interpretation.** A measured contrast is consistent with a mechanism, but another explanation remains possible. Examples include policy disagreement in safe-help cases and numerical sensitivity in the streaming mismatch.

**Proposed work.** A design is documented but not executed. Student training, independent human adjudication, live generation with shared GPU contention, and a completed release-aware scheduler comparison remain in this category.

An execution receipt establishes that a process ran or returned. It does not override a failed scientific gate. Conversely, a failed gate does not erase usable measurements collected by a successful execution.

### 1.3 Counts are attached to units

The program contains response records, distinct response texts, prompt-response pairs, model-query rows, images, token positions, paired implementations, and queue replays. These are not interchangeable sample sizes. A response scored twice is two decisions but usually one paired experimental unit. Four views of an image-response pair are four model queries, not four independent images. Thousands of prefix classifications do not create thousands of independent examples for an equivalence claim.

For this reason, tables below state both the scored workload and the inferential unit. Where results come from a selected or previously inspected cohort, the report does not promote their confidence intervals into estimates for an unspecified deployment population.

<a id="section-2"></a>
## 2. The rejection record and the actual chronology

### 2.1 JUDGe: scientific feedback, with an incomplete decision record

The archived OpenReview page for the JUDGe submission was captured on September 29, 2026, at 15:14:13 UTC. It contained two official scientific reviews. Their recommendations were 6, marginally above the acceptance threshold, and 7, a good paper to accept; both reported confidence 3. At capture, the page displayed "No Recommendation," with no decision note or metareview. The archive separately recorded the author's report that the submission had been rejected. These facts support describing an author-reported rejection with available scientific feedback. They do not support reconstructing an official rejection rationale from the two ratings. [E01](#e01)

The substantive feedback was coherent even though it was not an official rejection explanation. Reviewers wanted clearer exposition, less dependence on jargon and formal propositions, and a more visible refusal-versus-compliance result. More importantly, they questioned whether arbitrary prompt swaps preserve the meaning of a response. A context-sensitive guard may legitimately change its decision when the surrounding request changes. Model-generated strata and a failed initial human-reliability gate also limited the use of classification-error language.

The reviews requested stronger controls: response-only scoring, additional guards, benign-to-benign variation, exact rendered prompts, same-request refusal/compliance comparisons, and checks that displayed contexts remained coherent. They also asked what benchmark builders should do differently and wanted the annotation procedures described more clearly. The subsequent program should be judged against these scientific issues, not against an inferred acceptance score.

### 2.2 Archival security venue: a hosting-layer anonymity failure

The archival security-venue submission record inspected on October 12 was labeled "Administrative Reject (after Submission Deadline)." The cited issue was an anonymous supplementary link whose hosting page disclosed account identity. A separate logged-out HTTP receipt showed a successful response without a login cookie or authorization header, and identity-bearing fields in serialized page metadata. The visible page's filename was not the entire anonymity surface. The captured HTML hash was `672af45122a015986d606d29c2496c6d39c83f893be5764de354c997a4a7f7b4`. [E02](#e02)

This is distinct from criticism of the experimental design. Clean PDF or archive contents do not imply an anonymous hosting page. A hosting response can reveal identity through hydration or prefetch data even when the visible interface looks innocuous. Conversely, the administrative decision does not establish that the scientific claims were unsound. The available evidence establishes the nature of the exposure and that it was inspected on October 12; it does not establish the exact timestamp at which the administrative decision was made.

The private hosting URL and identity values are intentionally omitted. They are not needed to explain the failure mode and should not be recopied into another research artifact.

### 2.3 Chronology supported by receipts

| Date or interval | Evidence-supported event | Chronology qualification |
|---|---|---|
| August 30 / September 21 | The JUDGe page showed submission and modification dates | These are manuscript metadata, not rejection dates. |
| September 29, 15:14 UTC | Scientific-review archive captured | Author-reported rejection; no official decision visible in that snapshot. |
| October 12 | Security-venue administrative rejection and logged-out metadata exposure inspected | Inspection date, not an inferred decision timestamp. |
| October 12, 23:52 to October 13, 00:10, New York | Native Redpanda run | UTC receipt spans 03:52:47 to 04:10:17 on October 13. |
| October 13, 01:06 to 01:10, New York | Behavioral diagnostic v2 native scoring | A directory dated October 12 does not make this an October 12 execution. |
| October 13 | Conditional analysis, current-guard screens, multimodal screen, and safe-help follow-up | Status and counts come from the individual manifests and result files. |
| October 13 | First streaming pilot stopped after partial queue execution | Development profiling completed; the scheduler matrix did not. |
| October 13 | Separately frozen 64-pair equivalence experiment completed | Scientific admission failed; all conditional queue runs remained unexecuted. |

The internal scientific review dated October 12 is also useful context, but it is a model-generated review, not a venue decision. It correctly identified the gap between source-level integration analysis and a measured consequence through a real native consumer. Its discussion of generic contracts and conformance checks did not itself establish a novel detector or serving system. [E03](#e03)

<a id="section-3"></a>
## 3. What was carried forward from Prompt Transplant

### 3.1 The original matched intervention

The current manuscript retains the central Prompt Transplant design: hold response material fixed and change its surrounding request or rendered target designation. The historical transplant panel contains 360 response records, 120 in each of three strata, presented in 720 conditions. There are 278 distinct response texts and 80 source requests, with 33 harmful and 47 benign requests. These denominators refer to different objects. Repeated response text must not be treated as independent merely because it appears in another carrier. [E04](#e04)

The observed shift under harmful versus benign request pools is a request-pool contrast. It is not automatically an isolated causal effect of harmfulness. Request semantics, compatibility with the answer, and other pool differences travel with the intervention. A matched response reduces one source of variation; it does not guarantee that every resulting conversation has the same meaning or correctness label.

The inherited attack-artifact panel likewise has two denominators: 1,800 attempted slots and 1,637 realized prompt-response pairs, with 163 attempts producing no response. Treating confirmed no-output attempts as zero attack successes is different from calculating a success fraction only over scored responses. Neither denominator should be silently substituted for the other. Those experiments remain carried-forward material; they were not regenerated for this report.

### 3.2 A known integration defect, not a newly discovered vulnerability

The manuscript's specific Ollama integration contrast concerns the rendered target at the end of a Llama Guard 3 8B conversation. The affected template's opening and closing designation did not agree. The intervention holds the model weights, runtime, and conversations fixed while correcting the target designation. This defect was already publicly discussed in 2025. Replicating its effect in another native consumer adds an execution boundary, not priority for discovering the defect. [E04](#e04)

The distinction between a literal assistant-target repair and a general role-aware repair matters. A template used for both input and output checks must render the appropriate target dynamically. Replacing every target with an assistant label would not be a universal fix for input moderation.

### 3.3 What was not counted as new evidence

The historical manuscript contains human references, automated labels, cross-family contrasts, runtime comparisons, and extensive supplementary analyses. They retain their original provenance and limitations. New experiments do not retroactively turn automated labels into human gold or make previously inspected cohorts fresh. The present report therefore uses the inherited work as a baseline and motivation, not as an inflated count of post-rejection discoveries.

This boundary also applies to framework artifacts. The October 12 internal review described a contract inventory with 112 specifications, 84 conforming cases, 28 unknown cases, and no real classifier calls in that particular artifact. Such an inventory can help organize an audit; it cannot substitute for behavioral execution. The native-consumer phase below was designed to close that specific gap. [E03](#e03)

<a id="section-4"></a>
## 4. Phase I: Native Redpanda consumer replication

### 4.1 Question and implementation boundary

The question was narrow: does the known target-designation change alter a real processor's parsed safety metadata when the processor sends requests to an actual guard runtime?

The experiment used the native `ollama_moderation` implementation from pinned Redpanda source, Benthos 4.82.0, Go 1.26.9, and Ollama 0.32.1 on a local RTX 4080 Laptop GPU. The exercised implementation was the Apache-licensed internal processor, not a claim about the licensing or supported embedding of a separate public registration wrapper. A recording HTTP intermediary forwarded actual requests and responses. It did not replace the guard with simulated verdicts. [E05](#e05)

The processor automatically pulls its configured model during initialization. That lifecycle detail matters: installing a corrected local template before initialization could be undone by a later pull. The experiment therefore initialized the consumer, installed the intended template for each arm, and checked the resulting template before scoring. The same quantized weight artifact was used in both arms.

### 4.2 Datapoint generation and statistical unit

The input cohort was the existing 300-example JailbreakBench judge-comparison dataset. Its human-majority references split into 190 unsuccessful and 110 successful jailbreaks, with 235 distinct source goals. Each example passed through the native consumer twice, once per template. A datapoint was the parsed native decision for one example under one template, paired with the actual model output.

For the binary reference, false positives are unsafe decisions on the unsuccessful-jailbreak reference class, while false negatives are safe decisions on successful jailbreaks. The rates are

`FPR = FP / N_reference_negative; FNR = FN / N_reference_positive`. (1)

These definitions are tied to this dataset's human-majority reference, not to a universal definition of response harmlessness. The confidence intervals resample whole source goals, preserving dependence among examples sharing a goal. The analysis uses 10,000 bootstrap draws with seed 20261012. [E05](#e05)

### 4.3 Results

| Native decision metric | Stock template | Corrected template |
|---|---:|---:|
| False positives / 190 | 67 | 20 |
| True negatives / 190 | 123 | 170 |
| False negatives / 110 | 13 | 13 |
| True positives / 110 | 97 | 97 |
| False-positive rate | 35.3% | 10.5% |
| False-negative rate | 11.8% | 11.8% |

The paired reduction in false-positive rate is 24.7 percentage points, with bootstrap interval [18.9, 30.7]. The false-negative difference is zero percentage points, with interval [-2.7, 2.7]. The latter is not proof of equivalence or absence of individual changes. Across the cohort, 49 decisions changed: 48 unsafe-to-safe and one safe-to-unsafe. All 600 raw verdicts were valid and matched the processor's parsed `safe` metadata. No call failed or truncated; the maximum prompt length was 830 tokens under a 32,768-token context setting. [E05](#e05)

### 4.4 What this closes

The result closes a real evidence gap. The measured consequence is not confined to hand-rendered prompts or first-token scoring. It survives native model invocation, complete generated verdicts, and the consumer's parser. A strict parser does not correct a semantically wrong but well-formed verdict.

The result does not establish a production bypass. The experiment measured metadata, not a customer's configured allow/block policy. It also reused a previously inspected reference cohort. Its confidence interval describes variation under that cohort's source-goal resampling scheme, not the prevalence of affected installations or the behavior of unseen customer traffic.

The stock arm took about 932 seconds and the corrected arm about 112 seconds. Those elapsed times include uncontrolled initialization and warm-state differences. They are not evidence that changing the target label makes inference approximately nine times faster. The scientific contrast here is the paired decision, not an unbalanced latency comparison.

### 4.5 Contribution implication

The defensible contribution is a native-consumer replication of a known integration failure with a measured error consequence. A dynamic target correction and ordinary regression tests are already plausible remedies. A new paper cannot claim novelty merely by wrapping that repair in a general contract vocabulary. The next phase therefore asked whether a behavioral probe could reveal the target boundary without reading the template.

<a id="section-5"></a>
## 5. Phase II: Factorial behavioral diagnosis

### 5.1 Design

The diagnostic crossed two request strata with two response-source strata. Seventy-five quartets produced 300 inputs per implementation and 600 native decisions across the stock and corrected implementations. Twenty-five quartets were allocated to calibration and 50 to evaluation. Request strata came from human prompt labels in Aegis 2.0; response strata reused JailbreakBench references. The resulting transplanted conversations were not independently human-labeled for contextual harm. [E06](#e06)

An initial unscored input freeze was quarantined after an embedded-newline CSV parsing defect. Only the corrected v2 freeze was scored. This distinction prevents a data-preparation failure from being misreported as a model result or from disappearing from the execution history.

For quartet decisions `d_ab`, where `a` indexes the request stratum and `b` the response-source stratum, the analysis computes

`R = (|d_10 - d_00| + |d_11 - d_01|) / 2`, (2)

`S = (|d_01 - d_00| + |d_11 - d_10|) / 2`. (3)

`R` measures decision movement associated with changing the request while holding the response source fixed. `S` measures movement associated with changing the response while holding the request fixed. These are behavioral sensitivity measures, not direct readings of an internal attention target.

The diagnostic used material and negligible effect margins of 0.10 and 0.05. Those margins operationalize the proposed classifier; they are not natural constants. Evaluation intervals resample whole quartets, using 2,000 draws and seed 20261012.

### 5.2 The frozen-budget result

All 600 model decisions were valid, with no failed or truncated rows. The maximum prompt length was 1,457 tokens. At the largest frozen evaluation budget, 128 calls per implementation corresponded to 32 quartets. Calibration calls are not included in that evaluation-call count.

| Implementation | Request movement `R` | Response movement `S` | Diagnostic label |
|---|---:|---:|---|
| Stock | 0.234 [0.140, 0.344] | 0.484 [0.328, 0.625] | Mixed |
| Corrected | 0.078 [0.016, 0.141] | 0.766 [0.625, 0.891] | Inconclusive |
| Corrected minus stock | -0.156 [-0.266, -0.047] | +0.281 [+0.125, +0.453] | Paired contrast, not a third classifier |

The correction decreased request-associated movement and increased response-associated movement. That is informative behavioral evidence. It did not meet the proposed clean classification of the corrected implementation at the largest frozen budget. Smaller 32- and 64-call evaluation points labeled it response-sensitive, but choosing those points after the larger-budget result would be selective reporting. The 16-call point had only four quartets and was below the minimum eight-block classification requirement. [E06](#e06)

### 5.3 Postspecified full evaluation

The analysis of all 50 evaluation quartets was explicitly postspecified. It found stock `R = 0.260 [0.170, 0.360]` and `S = 0.460 [0.350, 0.570]`, versus corrected `R = 0.120 [0.060, 0.190]` and `S = 0.740 [0.630, 0.840]`. The corresponding differences were -0.140 [-0.220, -0.060] and +0.280 [+0.170, +0.400]. The diagnostic labels remained mixed and inconclusive. [E06](#e06)

This extension increases descriptive coverage but does not become the original frozen-budget result. It also shows why an isolated favorable point on the budget curve should not carry the headline.

### 5.4 Why this is not unique black-box identification

A finite set of observations cannot uniquely identify an arbitrary endpoint's internal target function. Two different systems can agree on the tested quartets and disagree elsewhere. Context-sensitive policies may also legitimately depend on both turns. The test can identify observable sensitivity under a declared probe distribution; it cannot prove that an opaque service exclusively scores a particular turn for every input.

The source labels introduce a second limitation. Human labels on the original request and original response setting do not automatically transfer to an incoherent transplant. Exact source disjointness reduces direct reuse but does not establish semantic independence or absence of model-training contamination. The phase therefore contributes a useful response to the proposed diagnostic idea: the intervention changes measured behavior, but the stronger identification claim is not supported.

<a id="section-6"></a>
## 6. Phase III: Conditional calibration without new model calls

### 6.1 Hypothesis

The calibration phase asked whether request-associated score variation could be treated as a nuisance component and removed without paying an unacceptable false-alarm cost. It was a retrospective local analysis of already available scores, not a new held-out model experiment. It admitted 36,693 model-query scores, ran no new guard calls, and consumed no new cloud inference. The recorded local analysis elapsed time was about 5.14 seconds. [E07](#e07)

The three guard identifiers in this analysis are the historical score lanes `lg3`, `qwen`, and `granite`. They must not be silently substituted with the newly scored Granite Guardian 4.1 and Qwen3Guard runs described later.

### 6.2 Data, transformations, and thresholds

The development reference used 1,024 BeaverTails calibration records, split into 512 learning and 512 threshold records by hashed groups. Evaluation used 2,048 BeaverTails records: 928 reference-safe and 1,120 reference-unsafe. Transport used 414 XSTest records: 336 safe and 78 unsafe. No XSTest outcomes were used to fit the transformations.

The label constructs differ. BeaverTails contributes its original question-answer risk reference after the experiment's neutralization procedure; XSTest contributes a response-harm reference. Pooling their errors into one headline number would conceal that difference.

Let `z(P,R)` be the native score and `n(P) = z(P,R0)` the score for a fixed refusal response `R0`. The latter is a nuisance feature, not a newly human-labeled benign control. The analysis compares 11 entries: full and shared-calibration versions of native scoring, response-only scoring, and fixed anchoring; learned subtraction; two- and three-feature logistic models; conditional quantile residuals; and conditional bins. [E07](#e07)

Fixed anchoring uses a native-minus-nuisance score. Learned subtraction estimates a slope from safe learning examples. The two-feature logistic model uses native and nuisance scores; the three-feature version also uses response-only scores. Logistic features are standardized, with L2 regularization `C = 1`. The conditional-tail model uses a 0.95 quantile regression with L1 penalty 0.001. Conditional bins split at the learning-set median nuisance score and use separate safe-score cutoffs.

For a safe calibration sample of size `n`, the conservative order-statistic rule uses

`k = ceil((n + 1) * (1 - alpha)), with alpha = 0.05`, (4)

and rejects scores strictly above the selected threshold. An unattainable order statistic gives an infinite threshold. Full-calibration simple baselines receive the full original calibration-safe pool; shared variants use the held-out threshold portion. These distinctions matter when comparing a learned method with a simpler method that can use more calibration data.

Two-view methods require native and nuisance scores; three-feature logistic scoring requires a third response-only view. A retrospective score transform can therefore be cheap to analyze while still being more expensive to serve. No live latency advantage was measured here.

### 6.3 Representative operating points

Each cell below is `false positives / false negatives`, not a percentage. The table selects six interpretable methods from the 11-entry analysis; the source artifact contains the complete result set.

**BeaverTails test: 928 safe and 1,120 unsafe references.**

| Guard lane | Native, full | Response-only, full | Anchored, full | Conditional bins | Conditional tail | Logistic 2D |
|---|---:|---:|---:|---:|---:|---:|
| Granite | 38 / 352 | 38 / 458 | 37 / 370 | 53 / 317 | 47 / 336 | 52 / 290 |
| LG3 | 31 / 448 | 33 / 514 | 33 / 420 | 48 / 369 | 28 / 436 | 34 / 405 |
| Qwen | 30 / 333 | 32 / 427 | 23 / 410 | 15 / 481 | 13 / 484 | 20 / 370 |

**XSTest transport: 336 safe and 78 unsafe references.**

| Guard lane | Native, full | Response-only, full | Anchored, full | Conditional bins | Conditional tail | Logistic 2D |
|---|---:|---:|---:|---:|---:|---:|
| Granite | 2 / 13 | 3 / 11 | 1 / 12 | 3 / 8 | 3 / 6 | 3 / 8 |
| LG3 | 5 / 11 | 7 / 9 | 5 / 10 | 9 / 8 | 5 / 25 | 4 / 12 |
| Qwen | 0 / 25 | 0 / 17 | 0 / 30 | 0 / 42 | 0 / 34 | 0 / 28 |

Conditional bins reduce LG3 misses by 79 on BeaverTails but add 17 false positives. The same method reduces Granite misses by 35 and adds 15 false positives. On Qwen, it adds 148 misses while removing 15 false positives. These are different operating-point movements, not evidence for one universal correction. Transport also changes the ordering: LG3 conditional-tail scoring misses 25 XSTest unsafe references versus 11 for native scoring. [E07](#e07)

### 6.4 Uncertainty and the negative screen

The analysis resamples calibration and test groups independently for 2,000 bootstrap draws, using seed 20261013. Thresholds are refit within each draw, but the learned transformations and bin boundary remain fixed. The intervals therefore include threshold and sampled-evaluation variation conditional on the learned transform; they do not describe full retraining uncertainty.

The exploratory admission screen required an upper confidence bound below zero for the false-negative difference and an upper false-positive increase no greater than 0.02. Across two conditional candidates, six comparator entries, three guards, and two datasets, none of the 72 contrasts passed both conditions. These were pointwise exploratory comparisons without a familywise multiplicity claim. [E07](#e07)

For example, LG3 conditional bins versus native scoring on BeaverTails had an observed false-negative difference of -0.0705, but its interval extended to +0.0433. Its observed false-positive increase was +0.0183, with interval reaching +0.0417. A favorable point estimate did not satisfy the declared uncertainty and false-alarm criteria.

This phase does not prove that all conditional calibration is ineffective. It shows that these candidates, on this development and transport evidence, did not supply the required improvement over the tested baselines. The 5% calibration parameter is also not a demonstrated 5% error guarantee under distribution shift.

<a id="section-7"></a>
## 7. Phase IV: Current-guard training-impact screening

### 7.1 The proposed causal mechanism

A possible larger contribution was that guard decisions alter the supervision a student receives. The causal chain would be

`guard target or policy -> selected/weighted demonstrations -> student update -> downstream safety and utility`. (5)

The completed phase measures the first link. It does not measure the student update or downstream outcome. There is no trained student, matched random-attrition control, behavior-matched attrition control, or targeted-restoration comparison in the completed artifacts. [E08](#e08)

The initial concern was that safe refusals to harmful requests might be removed by a guard that overweights the request. Testing this premise with current guards was necessary before investing in training. It also avoided treating an old integration bug as evidence that every contemporary response guard has the same failure.

### 7.2 How the screen was constructed

The source was the pinned WildGuardMix training file. It contained 86,759 source rows; 48,825 lacking response text were excluded, leaving 37,934 usable rows. Hashed prompt grouping allocated 30,170 to training, 3,979 to development, and 3,785 to held-out partitions. Original text was retained. Normalized exact-pair grouping is not semantic deduplication. [E08](#e08)

The eligible training pool contained 16,907 source-labeled safe demonstrations: 1,842 harmful-request safe compliances, 8,515 safe refusals, and 6,550 benign compliances. The screen selected 768 eligible examples plus 256 harmful-response controls, for 1,024 records. All were taken from the training partition; calling this a held-out final student evaluation would be incorrect.

The source labels were generated through the source dataset's GPT-4 annotation process. They are useful strata for screening, not independent human gold for this investigation. A disagreement with them is therefore called a source-label disagreement until adjudicated.

Each guard received 1,024 native response queries and 1,012 deduplicated input queries, or 2,036 queries per guard. Granite Guardian 4.1 8B and Qwen3Guard Gen 8B were scored at pinned revisions with native interfaces, BF16 inference, greedy decoding, batch size eight, maximum input length 8,192, and maximum output length 128. All 4,072 decisions were valid and complete. Qwen's `Controversial` output remained separate rather than being silently recoded as either safe or unsafe. [E08](#e08)

### 7.3 Response decisions by source stratum

| Source stratum | Records | Granite Safe / Unsafe | Qwen Safe / Unsafe / Controversial |
|---|---:|---:|---:|
| Harmful request, harmful compliance | 255 | 49 / 206 | 1 / 234 / 20 |
| Harmful request, source-safe compliance | 94 | 73 / 21 | 13 / 35 / 46 |
| Harmful request, source-safe refusal | 389 | 387 / 2 | 385 / 3 / 1 |
| Benign request, harmful response | 1 | 1 / 0 | 0 / 0 / 1 |
| Benign request, source-safe compliance | 285 | 281 / 4 | 273 / 0 / 12 |

Granite rejected 2/389 source-safe refusals, or 0.0051 with bootstrap interval [0, 0.0129]. Qwen returned unsafe or controversial on 4/389, or 0.0103 [0.0026, 0.0206]. Those rates do not support a substantial blanket refusal-removal mechanism in this screen. [E08](#e08)

The safe-help stratum is different. Granite returned unsafe for 21/94, or 0.2234 [0.1383, 0.3085]. Qwen returned unsafe or controversial for 81/94, or 0.8617 [0.7872, 0.9255]. These intervals use the experiment's prompt-cluster resampling, not the 2,036-query count as an independent sample size.

### 7.4 Input/output disagreement is not automatically error

Input rejection greatly exceeded response rejection in the safe-refusal stratum: the recorded differences were approximately 0.838 for Granite and 0.985 for Qwen. That is expected when the input itself requests harmful material but the response refuses it. It is not evidence that the input guard is confused. A training pipeline that uses an input-moderation decision to remove response supervision may be implementing a different selection policy from one that judges the response. The intended target must be specified before calling either decision wrong.

### 7.5 What changed after this phase

The screen rejected the simple premise that modern guards commonly remove safe refusals in the tested setting. It identified a narrower candidate: safe-looking helpful responses to harmful requests receive very different treatment from refusals. That candidate could reflect real over-rejection, a source-label defect, policy disagreement, or unsafe detail within an otherwise helpful answer. The next phase therefore held the request exactly fixed and examined the response distinction rather than proceeding directly to student training.

Successful worker elapsed times were about 577 seconds for Granite and 654 seconds for Qwen. They are execution measurements, not invoice totals. Failed starts and resource overhead are accounted for separately in Section 16.

<a id="section-8"></a>
## 8. Phase V: Multimodal preparation and native screening

### 8.1 Prepared data versus executed data

Two multimodal sources were prepared, but preparation is not a scored result. The SPA-VL preparation selected 256 unique-image preference pairs from 7,000 validation rows. A relative preference label does not establish a binary harmful/harmless answer label. No native scored SPA-VL result is claimed here. [E09](#e09)

The VLGuard preparation contained 2,000 training images with 2,977 response pairs and 1,000 test images with 1,558 pairs: 4,535 pairs over 2,987 unique image files. Twelve test pairs shared image bytes with training material. Those overlaps prevent treating every nominal test item as image-disjoint. These are preparation-level counts; the executed arm was a smaller subset.

### 8.2 Native Llama Guard 4 design

The executed subset used 128 images and 195 source-recommended responses: 61 `instruction`, 67 `safe_instruction`, and 67 `unsafe_instruction` records. Every response was examined through four views: input moderation and response moderation, each with image-plus-text and text-only content. This produced 780 primary model queries, with two additional preflight decisions excluded from the primary counts. [E09](#e09)

Llama Guard 4 12B was used at a pinned revision and its native interface. The policy definition held the 13 image-compatible categories fixed between modalities rather than adding a text-only category to one arm. Removing an image can legitimately change what a request means, so the image/text contrast is not presumed to be a pure nuisance intervention.

The inferential unit is the image group. Four views of a pair and multiple pairs from one image cannot be counted as independent examples. The reported intervals use image-cluster resampling.

### 8.3 Results and their asymmetry

All 780 primary decisions were valid. Every one of the 195 responses was classified safe in both modalities, giving 390 safe response decisions.

| Source instruction group | Responses | Unsafe input, image + text | Unsafe input, text only | Unsafe response, either modality |
|---|---:|---:|---:|---:|
| `instruction` | 61 | 16 | 10 | 0 |
| `safe_instruction` | 67 | 0 | 0 | 0 |
| `unsafe_instruction` | 67 | 29 | 19 | 0 |

The image-input minus response unsafe-rate difference was 0.2623 [0.1475, 0.3770] in the `instruction` group and 0.4328 [0.3134, 0.5522] in the `unsafe_instruction` group. The image-input minus text-input differences were 0.0984 [0.0164, 0.1967] and 0.1493 [0.0448, 0.2687], respectively. [E09](#e09)

The observation is consistent with a guard distinguishing some unsafe inputs from recommended safe answers. It cannot establish response discrimination because no harmful-response positive-control arm was scored. An always-safe response classifier would also achieve the observed response pattern. The dataset's image and instruction flags are not a substitute for independently labeled response harm.

Likewise, a bootstrap interval of [0, 0] for an all-zero empirical outcome is not a population zero-risk guarantee. Resampling cannot introduce positive examples that the evaluation design never contained.

### 8.4 Runtime failures and the eventual successful run

Two earlier attempts failed in dynamic-cache handling: one with a type error and one with a causal-mask shape error. The successful attempt removed the dynamic-cache override, used the checkpoint's supported static configuration, and disabled compilation. It used the pinned Llama Guard preview Transformers revision, Torch 2.6.0, Torchvision 0.21.0, and the recorded environment. The scoring cohort and scientific criteria were not changed to obtain a favorable outcome. Both failed attempts remain part of the execution and budget record. [E09](#e09)

The successful worker took about 361 seconds. This phase adds a native multimodal execution result and a useful negative design lesson. It does not establish a multimodal remedy for Prompt Transplant, a training effect, or improved safety from adding images.

<a id="section-9"></a>
## 9. Phase VI: Exact-request safe-help follow-up

### 9.1 Why this comparison was needed

The 94 safe-help cases were promising but ambiguous. Comparing them with refusals to different requests would leave request difficulty and topic confounded with response style. The follow-up therefore paired each focal response with a source-safe refusal to the byte-identical request from the original training source. The refusal was selected by a fixed hash rule, not by which guard verdict looked favorable. [E10](#e10)

The design added 94 harmful-response controls selected by hash from the original screen. Those controls were not necessarily responses to the same requests as the focal/refusal pairs. They test whether the guard is generally producing safe outputs, but they do not create a three-way matched experiment for every request.

The follow-up was exploratory and selected after the initial screen. Its 94 focal cases were not a fresh random sample of all source-safe training material. Each guard scored 282 response queries, for 564 decisions. All were valid, with no missing or invalid verdicts. Native interfaces and model revisions were unchanged.

### 9.2 Paired results

| Response group | Records | Granite Safe / Unsafe | Qwen Safe / Unsafe / Controversial |
|---|---:|---:|---:|
| Focal source-safe help | 94 | 73 / 21 | 13 / 36 / 45 |
| Exact-request source-safe refusal | 94 | 94 / 0 | 94 / 0 / 0 |
| Harmful-response controls | 94 | 18 / 76 | 1 / 86 / 7 |

Granite reproduced all 94 focal categorical decisions. Qwen reproduced 93/94 exactly; one focal item moved from controversial to unsafe. The conservative unsafe-or-controversial count remained 81/94. This is why the distinction between categorical reproducibility and binary non-safe reproducibility is necessary. [E10](#e10)

The unsafe-only focal-minus-refusal difference was 0.2234 [0.1383, 0.3085] for Granite and 0.3830 [0.2872, 0.4787] for Qwen. Counting Qwen controversial outputs as rejection increased its paired difference to 0.8617 [0.7872, 0.9255]. The source uses 2,000 paired request-level bootstrap draws.

Holding the request exactly fixed removes request identity as an explanation for this within-pair contrast. It does not establish that the focal response is an acceptable training demonstration. A guard might reject it because of actionable detail, contextual policy restrictions, uncertainty, or a genuine false positive. The harmful controls show that the experiment is not simply observing an all-safe guard, but their source labels remain automated references.

### 9.3 Model triage and the independent-human gap

A subsequent triage examined 112 items: the 94 focal cases and 18 controls. Two same-family model agents handled disjoint 56-item packets. This is not two independent ratings per item, and it supplies no inter-rater agreement estimate. It is an exploratory triage operation, not a human annotation study. [E11](#e11)

For the 94 focal examples, triage categorized response harm as 72 safe, nine uncertain, and 13 unsafe. Its training-suitability judgments were 57 exclude, 32 retain, and five uncertain. The intersection of strict safety, usefulness, and retainability left 32 candidate demonstrations. Of those, 28 had received a non-safe verdict from at least one guard in the original screen. Qwen's decisions on the 32 were four safe, 12 unsafe, and 16 controversial; Granite's were 25 safe and seven unsafe.

These counts are not a 28/32 population false-positive rate. The subset was selected using the observed cases and a model triage that is neither independent human gold nor a random population reference. The right conclusion is that 32 cases merit adjudication before a training intervention, not that 32 cases have already been confirmed as valid supervision.

Two local human-review packets were prepared, each containing 76 items: the 32 candidates, 32 matched refusals, and 12 controls. At the evidence cutoff, their independent human annotation count was zero. The packets were local preparation artifacts, not evidence of completed sharing, browser validation, or returned judgments. Original annotations were not altered. Training admission remains closed on this evidence. [E11](#e11)

### 9.4 What a training result would still require

To establish the proposed downstream mechanism, one would need validated supervision, an actual student trained under the candidate filtering policy, a control with matched example or token attrition, and a restoration intervention that isolates the removed behavior rather than merely adding more data. Equal update counts alone are insufficient if the number of supervised tokens differs. Safety and helpfulness would need evaluation with references independent of the training selector.

The completed work does not contain those interventions. The safe-help result is a better-motivated candidate mechanism than blanket refusal removal, but it remains a candidate.

<a id="section-10"></a>
## 10. Why the research direction changed

### 10.1 The literature constrained the proposed novelty

The source memos conducted a bounded primary-source review, not an exhaustive proof that no related work exists. They found direct overlap with several broad narratives that might otherwise have been advertised as new. [E12](#e12)

[Shape it Up! Restoring LLM Safety during Finetuning](https://arxiv.org/html/2505.17196v3) already uses guard-derived partial-response signals in a training process. The local investigation inspected a pinned STAR-DSS code path connecting value data to training weights, but did not execute that training system or establish that it contains the particular Prompt Transplant integration defect. Its relevance is that guard-mediated supervision is a real design pattern, not that this project discovered the pattern or measured its downstream failure.

[Safety Pretraining: Toward the Next Generation of Safe AI](https://arxiv.org/html/2504.16980v1) already examines the limitations of selecting only safe material and the value of contextual safety-oriented data. A generic claim that filtering can damage safety is therefore insufficient. A new experiment would need to identify a specific selection mechanism, control the amount and composition of removed supervision, and measure the resulting student effect.

The newer [Safety Reconstructed: Generative Modeling via Masked Diffusion Builds Strong Safety Guardrails](https://arxiv.org/html/2609.33634v1) includes fixed-response comparisons under different prompt contexts. The broad matched-context diagnostic is not an untouched research space. Its existence raises the bar for a claim based only on adding another guard or repeating a prompt-sensitivity contrast.

### 10.2 Streaming was also an occupied area

[SentGuard: Sentence-Level Streaming Guardrails for Large Language Models](https://arxiv.org/html/2606.02041v1) already evaluates streaming guard behavior using replayed response traces. [Predict, Don't React: Value-Based Safety Forecasting for LLM Streaming](https://arxiv.org/html/2604.03962v1), introducing StreamGuard, includes measured wall-clock evaluation. [SafeSpec: Fast and Safe LLM via Dynamic Reflective Sampling](https://arxiv.org/html/2606.19755v1) already couples safety-related decisions with speculative verification and generation behavior.

These papers prevent a credible novelty claim of "first streaming safety timing," "first replayed streaming guard," or "first joint safety and efficiency design." They also motivate a more specific question: under a fixed safety and release policy, can serving decisions allocate finite hardware to generation, checking, and release better than strong ordinary scheduling and batching baselines?

### 10.3 The revised systems question

The streaming memo proposed measuring work that becomes useful only when checked output can be released. Generation can create hidden output backlog; guard computation can unblock that backlog; queueing can delay both. A scheduling policy should be judged by timely useful output under an explicit safety policy and hardware budget, not merely by how many guard calls it processes.

That question requires a control that was missing from a naive batching story: changing a guard's increment size must not silently change the release decision while being described as the same service. The first pilot therefore profiled guard increments, and the second experiment directly tested exact release-decision equivalence.

The specific Qwen streaming checkpoint was an engineering target for this prerequisite. It is not sufficient by itself for a broad current-model serving claim. The cache correction already present in the pinned source is also not a new bug discovered by this project. The research question must stand on a measured serving effect, not on rediscovering a known cache issue or on the recency of a model name. [E12](#e12)

<a id="section-11"></a>
## 11. Phase VII: The original streaming pilot

### 11.1 Frozen source construction

The pilot used `Qwen/Qwen3Guard-Stream-0.6B` and `Qwen/Qwen3GuardTest` at pinned revisions. The source construction combined two different strata: source-Unsafe visible response spans from `response_loc` and source-Safe assistant thinking traces from `thinking`. The latter retained the source trace, including its thinking delimiters. This is not a balanced collection of ordinary final answers with independently matched harm labels. [E13](#e13)

The source files contained 813 response-location rows and 1,059 thinking rows. Selection retained original tokenization and source unsafe-span endpoints. It excluded rather than truncated examples beyond 2,048 total tokens or 1,024 response tokens, and excluded ambiguous or mismatched source boundaries rather than rebasing them. Exact source token IDs were required for unsafe-span accounting.

The frozen set contained 128 examples, 64 from each source label. Thirty-two were development examples, 16 per label, and 96 were held out, 48 per label. Hash-based selection and exact-prompt grouping made these partitions disjoint at the declared key; they did not establish semantic independence or freedom from model-training contamination. The manifest records excluded rows, source hashes, and selection seed 17420261013. [E13](#e13)

### 11.2 The explicit release policy

Native labels were `0 = Safe`, `1 = Unsafe`, and `2 = Controversial`. The serving policy rejected at the second of two adjacent, equal non-safe labels, using the earliest qualifying unsafe or controversial run. Rejection was absorbing: no subsequent safe label reopened the stream.

For zero-based token-relative labels `l_j`, define

`k = min { j >= 1 : l_j = l_(j-1) and l_j is Unsafe or Controversial }`. (6)

If such a `k` exists, the maximum committed prefix contains `k` tokens, with indices `0` through `k - 1`. The token at index `k`, which completes the rejecting run, is not committed. If no rejection occurs, the complete response can be committed. This policy can release a token with a first isolated non-safe label; it is not a policy that blocks every individually flagged token.

The policy is also not identical to an offline routine that prioritizes any later unsafe classification over an earlier controversial rejection. The experiment's earliest absorbing decision was an explicit serving choice. Neither policy becomes ground-truth semantic safety merely because it is implemented consistently.

If `v_i(t)` is the length of the prefix whose required checks have completed, the execution property is

`committed_i(t) <= min(v_i(t), k_i)`, (7)

with `k_i = infinity` until a rejection boundary exists. This is a release-accounting property. A checked or policy-approved prefix may still contain harmful content missed by the classifier.

### 11.3 Measured guard service

The worker used an NVIDIA L40S, BF16, SDPA attention, Torch 2.8.0, Transformers 4.55.4, and an explicit per-stream cache. It synchronized the GPU around forward calls and retained all intermediate token logits needed for prefix labels. Arm order was shuffled per example. Warm-up was excluded from the reported profile. The checkpoint's known cache repair was already included in the pinned source. [E13](#e13)

The development profile scored 16 examples, eight from each source label, spanning 5,169 response-token positions. It examined increments of 1, 8, 32, and 128 tokens. The profile recorded labels beyond a later absorbing rejection for comparison, while effective service cost counted only the work required through rejection or the end of a non-rejected response.

| Increment | Mean effective service, seconds | P95 forward, seconds | Fill delay plus P95 at 256 tokens/s | Token-label changes | First-rejection changes |
|---|---:|---:|---:|---:|---:|
| 1 | 2.548305 | 0.015725 | 0.019631 | 0 | 0 |
| 8 | 0.415371 | 0.019023 | 0.050273 | 3 | 1 |
| 32 | 0.121902 | 0.018640 | 0.143640 | 1 | 0 |
| 128 | 0.054776 | 0.018845 | 0.518845 | 2 | 1 |

Increment 32 reduced mean effective service by a factor of 20.9046 relative to increment 1. This is a measured amortization benefit on these traces, not a new batching algorithm or an end-to-end serving gain.

The original admission criterion required zero token-label changes and a fill-plus-forward bound no greater than 250 milliseconds. Only increment 1 passed. Increment 32's single difference occurred at relative position 16 of a development thinking trace, after both arms had already rejected at relative position 15. That makes the difference irrelevant to that example's released prefix, but it does not make the original exact-label criterion pass. The failed gate was preserved. [E13](#e13)

### 11.4 The partial one-token queue runs

The planned queue matrix contained four arrival/load cells, two repeats, and three policies: FIFO, earliest deadline first, and a release-aware heuristic that prioritizes ready checks over prefills and uses deadlines within that class. Tokens arrived synthetically at 256 tokens per second. The completion deadline was the nominal final-token arrival plus 0.5 seconds.

Only three of the 24 runs completed before the pilot was intentionally stopped. All three used FIFO with the admitted one-token increment. They reused the same 32 held-out examples, 16 per source label.

| Completed condition | Source-Safe completed / 16 | Source-Safe on time / 16 | Source-Unsafe with annotated-span overlap / 16 | P95 release lag, seconds |
|---|---:|---:|---:|---:|
| Burst, load 1.0, repeat 1 | 13 | 0 | 15 | 47.789 |
| Uniform, load 1.5, repeat 0 | 13 | 0 | 15 | 67.996 |
| Uniform, load 0.5, repeat 0 | 13 | 0 | 15 | 12.728 |

These are three arrival replays of one cohort, not 96 independent source examples. There are no corresponding completed EDF or release-aware runs in this pilot. The other 21 planned conditions are unexecuted, not missing successful results. [E13](#e13)

The one-token operating point explains why reducing request arrival load alone was insufficient. A 256-token/s stream produces a token roughly every 3.9 milliseconds, while a one-token guard forward in this environment commonly required substantially longer. The guard could fall behind within a single stream. This observation identifies a poor operating regime; it does not establish that FIFO is inferior to the unexecuted policies.

### 11.5 What the exposure counts mean

The unsafe-span metric counts overlap between the released prefix and source-annotated unsafe token positions. It does not ask a new independent judge whether that released prefix is actionable, complete, or harmful in context. An early token inside an annotated interval can be a fragment, and a larger span can still require context to interpret.

The source-Safe arm is a different kind of material: assistant thinking traces. Its completion or rejection behavior should not be presented as the same task as screening source-Unsafe visible responses. Consequently, the 15/16 overlap count is not 15 verified actionable-harm bypasses, and the three source-Safe rejections are not independently adjudicated false positives. The source endpoints and mixed strata constrain both claims.

### 11.6 Why a second experiment was justified

The development evidence suggested that exact token-label equality might be unnecessarily strict for an absorbing release policy: a difference after rejection cannot affect released output. A weaker, operationally relevant criterion was therefore reasonable to test. It had to be frozen separately, use examples not already examined by the pilot, and preserve the first gate's failure. That is what the final experiment did.

<a id="section-12"></a>
## 12. Phase VIII: Held-out release-decision equivalence

### 12.1 The new freeze

The second protocol used the remaining 64 previously unused held-out examples, 32 source-Safe and 32 source-Unsafe. It excluded the held-out examples already used by the first queue pilot. Its query hash was `b3d168cc8acce8d18d70966d2a8c92cb875568df209cc147f5f2073e11dc027c`. The experiment compared increments 1 and 32 with independent cache state and shuffled arm order per response. [E14](#e14)

The primary equivalence object was not every raw label. It was the release-decision vector

`D_i = (rejected, first_reject_relative, committed_tokens, released_prefix_hash, annotated_span_tokens_exposed)`. (8)

This criterion permits irrelevant post-rejection label differences but does not permit changing what reaches the release boundary. It tests an exact operational contract on the observed pairs. It is not a proof of semantic safety, nor a guarantee of equality on every future request.

The queue phase was conditional: any release-decision disagreement would stop admission. If all pairs matched, the worker would run 24 increment-32 queue replays across the same policy and arrival matrix. Offered capacity would be fixed from the earlier development profile rather than tuned on held-out outcomes. The release-aware policy would be compared against the better observed FIFO or EDF baseline within each cell. That plan is not a result.

### 12.2 Successful execution, failed admission

The downloaded `result.json`, `equivalence.json`, and `RESULTS.md` agree on the outcome: 64/64 pairs completed, one decision disagreement, one common-prefix token-label difference, and status `equivalence_failed`. The worker completed normally. Scientific failure and software execution failure are different events here. [E15](#e15)

| Source label | Pairs | Release-decision disagreements | Reference rejected | Increment-32 rejected | Reference span-overlap rows | Increment-32 span-overlap rows |
|---|---:|---:|---:|---:|---:|---:|
| Safe | 32 | 0 | 6 | 6 | 0 | 0 |
| Unsafe | 32 | 1 | 31 | 31 | 30 | 30 |

The aggregate number of annotated unsafe-span tokens exposed was 191 for the reference and 192 for increment 32. The source-Safe zero-overlap entries should not be read as an independently measured harmlessness guarantee; that arm does not supply the same unsafe-span annotations as the source-Unsafe arm.

The aggregate guard service was 426.130351 seconds for increment 1 and 19.201989 seconds for increment 32. The speed ratio is

`sum_i service_i,1 / sum_i service_i,32 = 22.1919896`. (9)

This is a ratio of measured total guard-service times, not the mean of per-example speed ratios, not a confidence interval, and not scheduler goodput. Worker wall time was 461.002166 seconds. The local completion/download receipt records 585.596212 seconds for its broader interval; the two timings describe different scopes and should not be substituted for one another. [E15](#e15)

### 12.3 The exact counterexample

The coordinator's source inspection of `pair-045.json` identifies `response_loc:369` as the differing pair. At zero-based relative position 2, the reference label is Unsafe and the increment-32 label is Safe. The first rejecting position moves from 3 to 4. The committed-token count moves from 3 to 4, the released-prefix hash changes, and the overlap with the annotated unsafe span moves from three tokens to four. These are the four differing decision fields; both arms still eventually reject the response. [E15](#e15)

This counterexample is sufficient to falsify exact released-prefix equivalence under the frozen admission rule. It is not sufficient to establish a whole-response bypass or that the additional released token is independently actionable harmful content. The latter would require semantic adjudication of the released prefix, which was not performed.

The mismatch could plausibly arise from shape-dependent numerical behavior under BF16 and different forward-call grouping. However, the artifacts do not contain the needed logit-margin traces and independent repeated-kernel controls to establish that cause. A numerical explanation is a hypothesis, not a demonstrated root cause. Neither a cache defect nor a causal-mask defect should be asserted from this observation alone.

### 12.4 Why identical whole-response outcomes are insufficient

Both arms reject 31/32 source-Unsafe examples and 6/32 source-Safe examples. A table containing only final response-level rejection would make them appear identical. The prefix-level comparison reveals that one decision becomes effective a token later. For a serving system, that difference can matter even when both complete responses are eventually blocked.

This is the most useful new measurement from the streaming phase: the evaluation boundary changes what counts as agreement. Whole-response rejection agreement does not imply exact released-prefix agreement. The magnitude and semantic consequence of the difference are separate questions.

The result also clarifies what the first development observation did and did not predict. A post-rejection label difference in development was operationally inert. A pre-rejection difference in held-out material was not. The second experiment was necessary precisely because the first observation could not establish that all future differences would occur after rejection.

### 12.5 No scheduler comparison was completed

The run contains zero queue runs, zero queue comparison records, and zero scheduler regimes. That is the intended consequence of failed admission. A generated field reporting zero cross-scheduler differences is vacuous when no scheduler executed. Likewise, the report template's reference to two timing repeats describes the planned queue design, not two completed repetitions of this paired equivalence measurement. [E15](#e15)

There is therefore no new FIFO-versus-EDF comparison and no evidence that the release-aware heuristic improves on either. There is also no basis for claiming a zero-event disagreement bound: the result's zero-disagreement upper-bound field is null because one disagreement occurred. The observed fraction is one of 64 selected pairs. No additional population confidence interval was computed in this report.

### 12.6 Interpretation

The experiment establishes two facts that must remain together: increment 32 substantially reduced measured guard service, and it did not preserve the chosen exact release contract on all held-out examples. Reporting only the speed ratio would conceal the failed prerequisite. Reporting only the failure would conceal useful engineering headroom.

The appropriate next decision depends on the intended claim. If exact preservation of the one-token policy is required, increment 32 is not admitted by this evidence. If a different release policy is acceptable, it must be declared as a different policy and evaluated for both semantic safety and utility. It cannot inherit an equivalence claim that the completed experiment disproved.

<a id="section-13"></a>
## 13. Cross-phase interpretation and review objections

### 13.1 A chain of hypotheses, not one accumulating success count

The program began with a known integration defect and sought a broader consequence. The native-consumer run strengthened the execution boundary. The behavioral diagnostic did not provide clean universal target identification. Conditional calibration did not supply a robust repair. Current-guard screening did not support substantial safe-refusal removal. Safe-help screening produced a narrower candidate, but independent adjudication and student training remain absent. Multimodal response scoring had no positive controls. Streaming profiling showed large service amortization, while the held-out contract test rejected exact release equivalence.

These are not contradictory results. They are successive answers to different questions. The error would be to transfer strength from one phase to another: using a known native-consumer error to assert a current-guard training effect, using an all-safe multimodal arm to assert discrimination, or using faster guard calls to assert better live generation.

### 13.2 Which review objections were actually addressed?

| Objection or missing boundary | What the new work adds | What remains unresolved |
|---|---|---|
| Source inspection is not native execution | Real Redpanda requests, model outputs, and parsed metadata | Deployment prevalence, actual customer release policy, and novelty beyond the known defect |
| Arbitrary prompt swaps may be incoherent | Exact-request focal/refusal comparison in the safe-help follow-up | Independent validity of the focal responses and whether their rejection is wrong |
| Modern guards may not share the old failure | Native current-guard and Llama Guard 4 screening | Broader model coverage and a downstream consequence |
| A proposed repair may trade false alarms for misses | Explicit calibration operating points, transport, and uncertainty | A candidate that actually passes the stated comparative screen |
| More modalities could be superficial | A native image/text execution with fixed policy categories | Harmful-response controls and independent response references |
| Serving timing could conceal changed safety behavior | A frozen exact released-prefix comparison | A policy-preserving faster implementation or a justified alternative policy |
| Scheduling may be ordinary batching in new language | Literature positioning and declared FIFO/EDF baselines | The scheduler experiment itself, live generator contention, and a measured advantage |

The scientific reviews' concern about claim boundaries remains relevant. The new work makes several boundaries more precise, but it does not remove them by renaming the paper.

### 13.3 The established, partial, negative, and unsupported claims

**Established on the specified artifacts:** the native integration intervention changes consumer decisions; current guards treat the selected focal safe-help and refusal responses differently; the two streaming increments differ in effective service; one held-out pair violates exact released-prefix equivalence.

**Partial findings:** the safe-help cases may identify useful supervision excluded by a guard, but their validity requires independent adjudication. The streaming service reduction may create a useful operating regime, but its release contract and end-to-end serving value remain unresolved.

**Negative findings:** the clean behavioral-identification label did not hold at the largest frozen budget; conditional calibration did not pass the combined screen; blanket refusal rejection was not substantial in the current-guard screen; the second streaming admission criterion failed.

**Unsupported claims:** a causal student-training effect, multimodal harm discrimination, a newly discovered cache bug, a verified actionable-harm streaming bypass, a completed scheduler improvement, live-generator speedup, and reduced production KV memory from backpressure.

This classification is more useful than a single simulated acceptance number. It states which claims can be written now and which would require different evidence.

<a id="section-14"></a>
## 14. Production implications and boundaries

### 14.1 Treat the rendered target as part of the integration contract

A guard's model name is not the complete moderation implementation. The rendered target, policy text, prompt serialization, decoding, parser, threshold, and downstream action all contribute to behavior. The native-consumer result supports testing the complete path from intended target to parsed metadata. It does not justify assuming that a strict parser makes the decision semantically correct.

A role-aware rendering regression should cover both input and output moderation. The corrected assistant-target case is not a reason to hard-code the assistant role for every use. A deployment also needs to state what happens after parsing: block, annotate, route, review, or release. This report measured only some of those layers.

### 14.2 Do not deploy a calibration transform on one favorable error rate

The conditional analysis shows why a lower miss count alone is not a sufficient deployment criterion. Some candidates increase false alarms, others lose sensitivity, and transport changes the ordering. Production selection should compare methods at a declared utility or risk constraint, include the additional model views in latency and cost, and evaluate the chosen operating point on traffic representative of the intended use.

None of the tested conditional candidates earned a general deployment recommendation from the recorded screen. The experiment is useful because it exposes the tradeoffs, not because it supplies a ready-made repair.

### 14.3 Keep model triage separate from human validation of supervision

The safe-help triage can prioritize review. It should not be used to overwrite source labels or declare a confirmed false-positive set. A production training pipeline must decide whether its objective is to exclude harmful responses, avoid harmful requests entirely, preserve useful safe assistance, or some combination. Those objectives can generate different valid selection policies.

If a guard is used to weight token-level supervision rather than reject whole examples, the unit of removal also changes. The present screen does not establish which loss terms would be removed in a particular student training run. The inspected STAR-DSS path motivates the question but does not answer it for this dataset and these guards.

### 14.4 Release correctness is not semantic safety

A release buffer can obey its execution policy perfectly while the classifier misses harmful material. Conversely, two implementations can have the same final rejection rate while exposing different prefixes. Production evaluation needs both dimensions: whether the system released only what its declared checking policy permits, and whether the released content satisfies the intended safety objective.

The source-span proxy addresses only an overlap question. Before describing an exploit, the released prefix would need independent contextual evaluation and a specified adversary and consumer behavior. The current example is a release-contract counterexample, not an established actionable exploit.

### 14.5 Backpressure is not automatically a KV-memory optimization

Pausing generation or withholding output may reduce one form of backlog while extending request lifetime. A paused request can retain its KV state longer; a lower instantaneous generation rate does not necessarily reduce peak resident memory or total KV byte-seconds. A production proposal must measure both residency and duration rather than assume backpressure frees memory.

Relevant measurements would include peak KV bytes, per-request KV residence time, the integral of resident KV bytes over time, queued and generated-but-unreleased tokens, admission pressure, and memory reclaimed on cancellation or rejection. The current guard-only replay measured none of the live generator's KV behavior.

### 14.6 Hosting anonymity is an independent release surface

The administrative failure has a production analogue: the object being shared is not just the file but the entire unauthenticated response of its hosting service. A clean document can sit behind identity-bearing page metadata. The supported lesson is to treat the exact submitted URL and its serialized response as part of the anonymity boundary. It does not require or justify republishing the exposed identity values.

<a id="section-15"></a>
## 15. What would constitute the next contribution

### 15.1 Training selection: a causal result rather than another label table

The strongest remaining training question is selective exclusion of valid, useful safe assistance, not blanket refusal removal. The next necessary evidence is independent adjudication of the candidate examples and controls under a fixed policy. If that adjudication rejects most candidates as unsafe or unsuitable, the proposed training mechanism should be abandoned or reframed rather than rescued by model-generated labels.

If a meaningful validated set remains, the intervention should compare the eligible pool, native guard selection, a target-corrected or calibrated selection, matched random attrition, behavior/topic-matched attrition, and targeted restoration. Updates and supervised tokens should be controlled. Multiple seeds and student families are needed to distinguish a data-selection effect from one run's variation. The earlier memo's multi-arm training design was a proposal, not a completed or newly authorized compute commitment. [E12](#e12)

The decisive result would be that a specific guard-induced selection pattern causes a downstream deficit and that restoring the affected supervision repairs it beyond matched controls. A larger collection of guard disagreements would not establish that chain.

### 15.2 Streaming: choose the policy question explicitly

There are two legitimate next paths. One is implementation equivalence: preserve the one-token release policy and investigate whether a faster execution can meet it reliably. That path needs repeated exact-input controls, logit margins, cache-state tracing, and kernel/precision comparisons to localize the observed mismatch. The present BF16 explanation is not yet sufficient.

The other is policy design: adopt chunked checking as a different release policy and evaluate the tradeoff openly. A conservative buffer, earlier blocking condition, or changed lookahead could alter exposed prefixes and benign blocking. Such a design must be evaluated as a changed safety policy rather than advertised as an equivalent optimization.

Neither path should count ordinary per-call amortization as a novel scheduler. The service reduction provides engineering headroom; the contribution would need to show what a distinct policy or serving method does with that headroom.

### 15.3 What a joint serving result must contain

A complete systems experiment would include a live generator, real guard inference, shared hardware or an explicitly priced split, declared token-release semantics, and timing at the actual serving boundary. It would compare against FIFO, EDF, ordinary batching, and other strong simple allocation strategies under the same safety policy and hardware budget.

The proposed primary benefit should be timely useful output or completed acceptable responses, not raw generated tokens that remain blocked in a buffer. Guard rejection, benign blocking, released-prefix exposure, deadline failures, cancellation, GPU utilization, KV occupancy, and resource cost must remain visible. Multiple workload regimes and more than one relevant generator/guard setting would be needed for a broad systems claim.

The earlier memo proposed a substantial useful-goodput improvement over the strongest ordinary baseline as a decision criterion. That was a research target, not a measured result or an acceptance threshold. The completed evidence does not show that it is achievable.

### 15.4 Why the current stop is scientifically appropriate

The final worker did not pause for an unexplained engineering reason. It encountered the exact condition that its protocol defined as disqualifying. Running the queue matrix anyway would have compared implementations with a known release-decision difference while presenting them as equivalent. Preserving the stop protects the interpretation of any later scheduling result.

A future experiment can change the criterion or policy, but it must be a new, explicitly motivated experiment. The failed result should remain in the history rather than being overwritten by a more permissive definition of success.

<a id="section-16"></a>
## 16. Reproducibility and compute accounting

### 16.1 From source data to claims

The reproducibility chain is

`pinned source -> selection/freeze -> rendered native queries -> raw decisions -> validity admission -> paired/grouped analysis -> report claim`. (10)

Each arrow matters. Source labels determine what an error rate means. Selection determines the population to which it could apply. Rendering determines the guard's target. Validity admission separates malformed, missing, and truncated records from safe verdicts. Pairing and grouping determine the statistical unit. The final report cannot correct a missing link by adding a caveat after a stronger headline.

The source inventory includes protocols, hashes, manifests, execution receipts, analyses, and preserved failures. Raw harmful text remains in the private data boundary. The report does not change labels, rescore models, regenerate statistics, or alter the source freezes.

### 16.2 Completed scoring workload

| Phase | Primary completed decisions or pairs | Independence boundary |
|---|---:|---|
| Native consumer | 600 decisions | 300 paired examples; 235 source-goal groups |
| Behavioral diagnostic | 600 decisions | 75 quartets per implementation, split calibration/evaluation |
| Current-guard text screen | 4,072 decisions | 1,024 source records; input queries deduplicated separately |
| Multimodal screen | 780 decisions | 195 response records over 128 images |
| Safe-help follow-up | 564 decisions | 94 focal/refusal pairs plus 94 controls, scored by two guards |
| First streaming development profile | 16 source examples, 5,169 response-token positions | Repeated increments on the same development examples |
| First streaming queue pilot | 3 completed runs | Repeated arrival replays of the same 32 held-out examples |
| Final streaming equivalence | 64 paired examples | Previously unused held-out pairs; no queue runs |

The five successful paid classification workers produced 5,416 primary valid decisions in total. This excludes two multimodal preflight decisions, failed attempts, and streaming classifications. The 1,200 native-consumer and behavioral decisions were local workloads. These counts describe work performed, not the number of independent human-labeled examples.

### 16.3 Reservations are not actual charges

The original additional Modal authorization for this program was $40. The budget ledger records seven reservations of $2.75 for classification attempts, two reservations of $2.50 for the streaming attempts, and $1 of prior shared overhead. That totals $25.25 reserved and $14.75 unreserved. [E16](#e16)

| Attempt | Reserved amount | Outcome at cutoff |
|---|---:|---|
| `granite-02bc4c6d7592` | $2.75 | Successful text screen |
| `qwen-dd0d5f186b97` | $2.75 | Successful text screen |
| `llama4-mm-cc4eb6607eb3` | $2.75 | Failed dynamic-cache attempt, preserved |
| `llama4-mm-native-efe3370a4602` | $2.75 | Failed causal-mask attempt, preserved |
| `llama4-mm-static-4f8895b58da7` | $2.75 | Successful multimodal screen |
| `safehelp-granite-e6973c675a0d` | $2.75 | Successful matched follow-up |
| `safehelp-qwen-cb9a81187131` | $2.75 | Successful matched follow-up |
| `streaming-6e5e4363744a` | $2.50 | Profile complete; partial queue pilot stopped |
| `streaming-equivalence-e5c0570afa1e` | $2.50 | Execution complete; equivalence failed; queue gated out |
| Previously reserved shared overhead | $1.00 | Accounting reservation |

These are planning reservations, not provider invoices or confirmed charges. Failed attempts are not treated as free or automatically refunded. Successful worker elapsed time also cannot be converted into a complete invoice without accounting for startup, idle time, resources, failed jobs, and teardown.

The last inherited dashboard balance observation was $39.18 at the end of the preceding task. It was not freshly checked for this report and is account-wide, not a per-experiment billing receipt. It therefore cannot establish the exact cost of the final run or certify compliance with a provider-side cap. No new spending was authorized or initiated by the report-writing task.

### 16.4 Local reproduction entry points

The following command forms are recorded from source documentation. They are not claims that this report reran them. New output paths should be used so that original analyses and failures remain preserved.

From the native-consumer experiment directory:

```powershell
python analyze.py --labels evidence/labels.jsonl --stock evidence/stock_detexted.jsonl --corrected evidence/corrected_detexted.jsonl --output <new-replay.json>
```

From the behavioral-diagnostic experiment directory:

```powershell
python -B diagnostic.py calibrate --plan analysis_v2/frozen_plan.json --scores analysis_v2/native_scores/calibration.jsonl --out <new-calibration.json>
python -B diagnostic.py analyze --calibration <new-calibration.json> --scores analysis_v2/native_scores/test.jsonl --out <new-analysis.json>
python -B full_panel.py --manifest analysis_v2/manifest.json --scores analysis_v2/native_scores/test.jsonl --out <new-full-panel.json>
```

The conditional-analysis entry point is `run.py --inputs <admitted-transport-root> --output <new-output>`, with source bindings supplied by its design freeze. The paid screens and streaming experiments require their pinned manifests, worker environment, and approved model access; this report does not substitute a reconstructed launch command for those records.

Reproduction should distinguish a local statistical replay from a new model run. The former can reproduce calculations conditional on existing scores; the latter tests the current execution environment and may produce different categorical or numerical outcomes. Neither should silently overwrite an original result.

### 16.5 Preserved failure and amendment history

The history includes an unscored CSV freeze replaced after an embedded-newline parsing defect, two failed multimodal runtime attempts, the original exact-label streaming gate failure, a stopped partial FIFO pilot, and the held-out exact-release gate failure. These events have different meanings. The first three involve preparation or runtime configuration; the final equivalence failure is a scientific result from a successfully completed run.

The report was authored from existing artifacts and the final downloaded outcome. It did not run tests, broad validation, additional experiments, or a second report verification/edit pass. Existing manuscript and experiment files were not modified.

<a id="section-17"></a>
## 17. Limitations and conclusion

### 17.1 Statistical limits

Most intervals are conditional on selected cohorts, fixed model revisions, and specified resampling units. Bootstrap precision does not repair label ambiguity, prior inspection, or distribution shift. Repeated model views do not increase the independent example count. The calibration comparisons are exploratory and not a familywise-controlled claim. The safe-help subset is post-selected. The streaming paired sample contains 64 selected held-out examples, not a random sample from a known deployment distribution.

The final equivalence measurement provides a counterexample to exact equality but not a general deployment disagreement rate. Its speed ratio has no population performance confidence interval. The planned queue repeats were not executed and cannot be cited as timing replication.

### 17.2 Semantic and task limits

The program uses several references: human-majority jailbreak success, human request strata, source model-generated response labels, recommended multimodal responses, and source unsafe-token intervals. They answer different questions. None should be renamed as one universal human-harm ground truth.

The source-Safe thinking traces and source-Unsafe visible responses in the streaming study differ in content type and annotation structure. Overlap with an unsafe interval is not an independent judgment that the exposed prefix is actionable. All-safe multimodal responses do not establish sensitivity to harmful responses. Same-request safe-help differences do not establish that the focal answer is appropriate training material.

### 17.3 Systems and novelty limits

The native-consumer result does not measure deployed prevalence or an actual customer blocking policy. The calibration analysis does not measure live multi-view serving cost. The training phase does not train a student. The streaming worker replays frozen tokens through a real guard but has no live generator, no generator/guard GPU contention, no cross-request tensor batching, and no external network-client timing.

The known template defect, its ordinary regression repair, the already repaired cache behavior, generic context-sensitivity probes, and batching amortization are not new contributions merely because they appear in one project. Prior work already studies guard-mediated training, streaming checks, and safety/efficiency combinations. The new contribution must come from a demonstrated mechanism and consequence beyond those baselines.

### 17.4 Conclusion

The post-rejection work is complete as a documented sequence of executed experiments, not as proof that every proposed extension succeeded. It strengthens the native-consumer boundary of the inherited Prompt Transplant result, rejects several overly broad follow-up hypotheses, identifies an unresolved safe-help selection mechanism, and demonstrates that whole-response agreement can hide a released-prefix difference in chunked guard execution.

The final experiment is especially clear: 64 pairs completed, one exact release-decision mismatch, a 22.19-fold reduction in measured guard service, and zero admitted scheduler runs. Both the speed and the mismatch belong in the account. Neither implies a verified harmful bypass or a completed scheduling contribution.

The next credible paper would need either a causal training-selection result with independently validated supervision and matched controls, or a live serving result with an explicit release policy and strong scheduling baselines. Those are concrete missing experiments, not wording problems. This report preserves the evidence needed to choose between them without overstating what has already been achieved.

---

<a id="appendix-a"></a>
## Appendix A. Evidence-linked source inventory

The paths below are relative to the private repository root. They identify evidence custody without disclosing machine-owner paths or creating public links to sensitive datasets. Protocols describe intended work; result files and receipts determine what executed.

<a id="e00"></a>
### E00. House standard and writing rules

- `AGENTS.md`: current project instructions, including source-first technical reporting and model-selection requirements.
- `PublishReady/reports/Technical_Report_126.md`: reference for analytical depth, units, phase interpretation, production guidance, and reproducibility.
- `papers/paper_style_codex.md`: claim discipline, equation and prose conventions.

These documents supplied the authoring standard, not scientific results.

<a id="e01"></a>
### E01. JUDGe review archive

`papers/prompt_transplant/judge2026/reviews/openreview_reviews_20260929.md`

Authority for the September 29 capture, two official review recommendations and confidence values, the author-reported rejection, and the absence of an official decision in that snapshot. Review content is paraphrased rather than reproduced. The OpenReview forum identifier is `fTha6NKEE6`.

<a id="e02"></a>
### E02. Security-venue administrative anonymity evidence

Private receipt basename: `<venue><submission-id>-anonymity-evidence-20261012.json` (venue and submission identifier withheld from this report).

The receipt is retained outside the paper tree in the private local evidence location. It records the logged-out request conditions and identity-bearing hosting metadata. The administrative decision context was recovered from the prior investigation record; the HTTP receipt independently supports the exposure mechanism. Exact decision time is not established. Identity values, the submitted sharing URL, and the machine-owner path are omitted here.

<a id="e03"></a>
### E03. Internal scientific review

`reviews/scientific_review_20261012.md` in the current conversion directory under `papers/prompt_transplant/`

Internal model-generated review, not an official venue review. The reviewed main PDF hash is `e52652c00ba8194ed4b56db0c0f9acd9548fbee0be42a9ceb858edde125f56`; the appendix hash is `8f1284b99e969827aa1c29da95287ad779b21e13e2fe31cd93a50384ea6da409`. Its useful role here is identifying the native-execution and novelty gaps, not predicting acceptance.

<a id="e04"></a>
### E04. Carried-forward manuscript

`main.tex` in the current conversion directory under `papers/prompt_transplant/`

Authority for the current manuscript framing, inherited panel units, attack-attempt denominators, known-template history, and boundaries of the existing contribution. This report did not modify the manuscript or reexecute its historical experiments.

<a id="e05"></a>
### E05. Native Redpanda replication

- `research/tr174/redpanda_consumer_20261012/PROTOCOL.md`
- `research/tr174/redpanda_consumer_20261012/README.md`
- `research/tr174/redpanda_consumer_20261012/evidence/analysis.json`
- `pt/redpanda_consumer_20261012/native_run_01/execution_receipt.json`

Together these establish the native implementation, data and template pins, paired confusion counts, goal-cluster intervals, output validity, timing, and execution cleanup record. The de-texted analysis path preserves the distinction between raw model verdicts and consumer metadata.

<a id="e06"></a>
### E06. Behavioral diagnostic v2

- `research/tr174/behavioral_diagnostic_20261012/RESULTS.md`
- `pt/behavioral_diagnostic_20261013_v2/probe_freeze.json`
- `pt/behavioral_diagnostic_20261013_v2/native_run_01/execution_receipt.json`
- `research/tr174/behavioral_diagnostic_20261012/analysis_v2/native_scores/analysis.json`
- `research/tr174/behavioral_diagnostic_20261012/analysis_v2/native_scores/full_test_postspecified.json`

The result narrative and freeze distinguish the scored v2 data from the quarantined unscored preparation. The full-test extension is explicitly postspecified.

<a id="e07"></a>
### E07. Conditional calibration

- `research/tr174/conditional_gate_20261013/PROTOCOL.md`
- `research/tr174/conditional_gate_20261013/run.py`
- `research/tr174/conditional_gate_20261013/development_01/design_freeze.json`
- `research/tr174/conditional_gate_20261013/development_01/results.json`
- `research/tr174/conditional_gate_20261013/development_01/REPORT.md`

Authority for the admitted historical scores, local-only analysis, thresholds, transformations, point estimates, bootstrap scope, and 72 failed combined screens. The cache bridge checked both log odds and probabilities; its 12 checked queries had maximum log-odds difference zero and probability difference approximately `1.11e-16`. This is a cache admission check, not a new guard result.

<a id="e08"></a>
### E08. Current-guard training screen

- `research/tr174/training_impact_20261013/PROTOCOL.md`
- `research/tr174/training_impact_20261013/SCREENING_RESULTS.md`
- `research/tr174/training_impact_20261013/prepare.py`
- `research/tr174/training_impact_20261013/analyze.py`
- `pt/training_impact_20261013/text/manifest.json`
- `pt/training_impact_20261013/runs/granite-02bc4c6d7592/analysis.json`
- `pt/training_impact_20261013/runs/granite-02bc4c6d7592/summary.json`
- `pt/training_impact_20261013/runs/qwen-dd0d5f186b97/analysis.json`
- `pt/training_impact_20261013/runs/qwen-dd0d5f186b97/summary.json`

These sources define the source-label provenance, grouping and partitions, screen strata, native queries, valid categorical decisions, intervals, and elapsed times. They contain no completed student-training result.

<a id="e09"></a>
### E09. Multimodal preparation, failures, and result

- `pt/training_impact_20261013/multimodal/manifest.json`
- `pt/training_impact_20261013/vlguard/manifest.json`
- `pt/training_impact_20261013/multimodal_screen/manifest.json`
- `pt/training_impact_20261013/runs/llama4-mm-static-4f8895b58da7/analysis.json`
- `pt/training_impact_20261013/runs/llama4-mm-static-4f8895b58da7/summary.json`
- `pt/training_impact_20261013/runs/llama4-mm-static-4f8895b58da7/runtime_amendment.json`

The preserved failure records under attempts `llama4-mm-cc4eb6607eb3` and `llama4-mm-native-efe3370a4602` document the two unsuccessful cache configurations. Preparation manifests and successful scores must not be pooled as if every prepared pair was evaluated.

<a id="e10"></a>
### E10. Exact-request safe-help follow-up

- `pt/training_impact_20261013/safe_help_matched_20261013/PROTOCOL.md`
- `pt/training_impact_20261013/safe_help_matched_20261013/manifest.json`
- `pt/training_impact_20261013/runs/safehelp-granite-e6973c675a0d/analysis.json`
- `pt/training_impact_20261013/runs/safehelp-granite-e6973c675a0d/summary.json`
- `pt/training_impact_20261013/runs/safehelp-qwen-cb9a81187131/analysis.json`
- `pt/training_impact_20261013/runs/safehelp-qwen-cb9a81187131/summary.json`

Authority for exact-request matching, control selection, response-only native queries, repeated categorical decisions, and paired intervals.

<a id="e11"></a>
### E11. Triage and uncompleted human adjudication

- `pt/training_impact_20261013/safe_help_audit_20261013/FINDINGS.md`
- `pt/training_impact_20261013/safe_help_audit_20261013/triage_summary.json`
- `pt/training_impact_20261013/safe_help_audit_20261013/human_review/manifest.json`
- `pt/training_impact_20261013/safe_help_audit_20261013/human_review/COORDINATOR.md`

These sources separate model triage from independent human judgment and document the 32 candidate demonstrations and two blank local review packets. No completed human labels are claimed.

<a id="e12"></a>
### E12. Research-direction memos

- `research/tr174/NEXT_WEDGE_LITERATURE_20261013.md`
- `research/tr174/STREAMING_COMMIT_RESEARCH_20261013.md`

The first memo motivates the guard-to-training mechanism and documents the inspected STAR-DSS revision `14d1a908f5abd5c9710f2ebca01edc70a518949b`. The second motivates fixed-policy streaming serving and strong ordinary baselines. Both contain proposed work that must not be counted as executed. Primary literature links in Section 10 support the novelty boundaries rather than an exhaustive absence claim.

<a id="e13"></a>
### E13. Original streaming pilot

- `research/tr174/streaming_commit_20261013/PROTOCOL.md`
- `research/tr174/streaming_commit_20261013/worker.py`
- `pt/streaming_commit_20261013/frozen/manifest.json`
- `pt/streaming_commit_20261013/runs/streaming-6e5e4363744a/PILOT_STOP_REPORT.md`
- `pt/streaming_commit_20261013/runs/streaming-6e5e4363744a/PILOT_STOP_REPORT.json`
- `pt/streaming_commit_20261013/runs/streaming-6e5e4363744a/recovered/environment.json`

The recovered artifacts preserve development profiles and the three completed FIFO runs. Application `ap-QOIS8S4V977OXG4tGo7jf9` was stopped. The report distinguishes a completed profile from a stopped scheduler matrix.

<a id="e14"></a>
### E14. Separately frozen release-equivalence protocol

- `research/tr174/streaming_commit_20261013/EQUIVALENCE_PROTOCOL.md`
- `pt/streaming_commit_20261013/release_equivalence_frozen/manifest.json`

Authority for the unused 64-example cohort, exact release-decision vector, zero-disagreement admission rule, conditional queue matrix, and budget reservation. The original exact-label gate is not replaced or retrospectively passed.

<a id="e15"></a>
### E15. Final completed equivalence outcome

Root: `pt/streaming_commit_20261013/runs/streaming-equivalence-e5c0570afa1e/`.

- `result.json`: completed pairs, scientific status, aggregate service, worker wall time, and empty queue outputs.
- `equivalence.json`: label-stratified rejections and annotated-span overlap, the single decision disagreement, and null zero-event bound.
- `RESULTS.md`: generated result narrative; its unexecuted queue-template statements are qualified in this report.
- `receipt.json`: downloaded artifact inventory and broader completion/download wall time.
- `environment.json`: L40S, BF16, SDPA, pinned model and packages.
- `pair-045.json`: the exact mismatch, inspected by the coordinator and supplied as a field-level source extract for this report.

Application `ap-kttmkOIh1Vt1YMjWLtMiXj` completed normally according to the coordinator's execution handoff. The report directly read the aggregate result and receipt files; it uses the coordinator's supplied pair extract for the detailed mismatch rather than independently rereading that private pair. The distinction does not change the aggregate failure already recorded in `equivalence.json`.

<a id="e16"></a>
### E16. Compute reservation ledger

`pt/training_impact_20261013/budget/attempt-*.json`

The nine attempt receipts bind the reservations listed in Section 16.3. They are not provider billing statements. The final job's owned-resource closure was supplied by the coordinator after completion; this report did not independently query provider resource or billing state.

<a id="appendix-b"></a>
## Appendix B. Pinned models, datasets, and freezes

### B.1 Model and implementation revisions

| Component | Revision or version |
|---|---|
| Redpanda source | `fe6754c5886b4541542b3450e1eb241b7ef690ef` |
| Native consumer runtime | Benthos 4.82.0; Go 1.26.9; Ollama 0.32.1 |
| Shared Llama Guard 3 quantized weight artifact | SHA256 `883f1adfefb2622b3b8b226e45c52d45551f97304cd77b7e2a8009fea106d926` |
| Stock rendered template | SHA256 `cb5a7b24ae22ddee985e72490538cf8d5b3cd4125143f23933887ef5bcdb80c2` |
| Corrected rendered template | SHA256 `196d0224ee86385c9854d47fbaf46c1b4f337701889ee72ababd74a9f5677ab1` |
| Granite Guardian 4.1 8B | `ab01ccca5dcfb80246369a086a4a87a29198f5af` |
| Qwen3Guard Gen 8B | `4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb` |
| Llama Guard 4 12B | `87acb4b94e930c3d679e6e7ee9d57e2feab9ea71` |
| Llama Guard preview Transformers | `63cd4c76f3a068ac36c37453be156e019c55c1b7` |
| Qwen3Guard Stream 0.6B | `74e1479150e9029d6778993f00491108323bb6f8` |

The new text screens recorded Torch 2.11.0, Transformers 5.10.1, and Hugging Face Hub 1.19.0. The successful multimodal run used its separately pinned preview stack. The streaming execution used Torch 2.8.0, Transformers 4.55.4, and Hugging Face Hub 0.34.6. Different task-native environments are not interchangeable reproductions.

### B.2 Dataset revisions

| Source | Pinned revision | Role in this report |
|---|---|---|
| JailbreakBench judge comparison | `886acc352a31533ffbcf4ef22c744658688086fc` | Native-consumer reference and reused response strata |
| Aegis 2.0 | `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c` | Request strata for behavioral quartets |
| WildGuardMix | `d29c47f41c8b51348b5c8e8c81c039b3132b66d1` | Training-screen and matched safe-help source |
| SPA-VL | `0e75b34a9e803a236873420c8680de3889befd62` | Prepared preference pairs, not a scored outcome |
| VLGuard | `b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46` | Executed recommended-response multimodal subset |
| Qwen3GuardTest | `8a8391a583d65dc5f2be59e815404fe54289fa7c` | Streaming thinking traces and response-location spans |

The preparation artifacts record WildGuardMix under ODC-BY, SPA-VL under CC BY 4.0, and VLGuard under MIT. Those records describe source provenance; they are not a new redistribution or publication authorization for the private derived packages.

### B.3 Query and protocol bindings

| Artifact | SHA256 |
|---|---|
| JailbreakBench `judge-comparison.csv` | `dacaf76a1057785f11b0fc6aa07b9a73cdab588319ddfc5e8cef2909e5b112a5` |
| Behavioral v2 query input | `317742f16ae456eabc7890a231b377ba1b1e53fe0b9b9be542c3c454cec5548f` |
| Behavioral frozen plan | `bacb4da94447749fd28642d16c118e5de4a320a8e7b723cce1bafd3e7d838f63` |
| Current-guard text queries | `1f1d0ece3a5faa52cb3c5e70f2ab47d90ce649c88bb57ac15aa53596ed71c77d` |
| Multimodal screen queries | `8a19ff53887a88199162995bb78fb96330745fdc88c425378b7d245ba1969c85` |
| Matched safe-help queries | `6a1d00914517a7f841efb01a8a74f0486a26260470c7c1b9e78ee367db51207d` |
| Original streaming queries | `eb0f8e65a0e4bc06d16bec6b0e0f382ebaeb37cffea417b0c371b44df6405f57` |
| New held-out equivalence queries | `b3d168cc8acce8d18d70966d2a8c92cb875568df209cc147f5f2073e11dc027c` |
| Equivalence protocol | `023a5089ae0b3f7ad45ac1e2e11cf3fb12a61f23ae288facc46e86453c9a6a9f` |
| Development profile bound into new freeze | `ba4c791284b5e5070092d3594017894d43d6add5d84a1893dc047bf36cd1950e` |

These identifiers locate frozen evidence. Their inclusion is not a claim that the report-writing task recomputed the hashes or reran a validation suite.

<a id="appendix-c"></a>
## Appendix C. Terms that must not be conflated

| Term | Meaning in this report | What it does not establish |
|---|---|---|
| Source-safe | Safe according to the named source's annotation process | Independent human agreement or suitability for student training |
| Safe-help candidate | A focal response retained by the model triage for further review | Confirmed false positive or human gold |
| Native consumer execution | Real processor calls, guard output, and parsed metadata | Customer deployment or the customer's eventual allow/block action |
| Behavioral target sensitivity | Decision movement in the declared factorial probe | Unique identification of an arbitrary endpoint's internal target |
| Calibration target `alpha` | A parameter in a threshold-selection rule | A demonstrated deployment false-positive guarantee |
| Student-training effect | A measured outcome after a controlled student update | A difference in which examples a guard would select |
| Multimodal discrimination | Ability to separate safe and harmful response cases with valid references | All-safe verdicts on recommended responses |
| Effective guard service | Measured guard computation needed through the declared stopping rule | End-to-end user latency, network timing, or scheduler goodput |
| Exact release equivalence | Equality of the specified release-decision vector on paired examples | Semantic harmlessness or equality on all future inputs |
| Unsafe-span overlap | Released token positions intersecting source-annotated unsafe positions | An independently actionable harmful-prefix bypass |
| Absorbing rejection | No later label reopens a rejected stream | Rejection before any harmful content can ever be emitted |
| Guard-serving replay | Real guard inference on frozen tokens with synthetic arrivals | Live generation, shared generator KV contention, or production serving |
| Queue gate failure | A prerequisite failed, so conditional replays did not execute | A scheduler result or a software launch failure |
| Reserved compute | Amount allocated in the local experiment ledger | Provider invoice, verified account balance, or actual charge |

**Closing evidence statement:** The report preserves positive, negative, partial, and unexecuted findings separately. No new manuscript, experiment, label file, or release package was changed to produce it.
