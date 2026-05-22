# Pre-Phase 5 Plan — From Operational Lab to Predictive Lab

**Goal:** Close the analytical gap on Stage 2, then build the architectural breadth Phase 5+ needs.

**Companion reference:** `MODEL_SIZING_SCALING_REFERENCE.md` — the framework these stages exercise.

**Working mode for Stage 2.5 (interview-week sprint):** direction-only. The plan tells you *what* to do and *what to produce*, never *what the answer is*. The math is done by hand.

---

## Stage 2.5 — Roofline Reconciliation Appendix (this week, interview blocker)

**Goal:** Turn the lab from "I characterized vLLM operationally" into "I can predict its behavior from physics."

**Deliverable:** New appendix section in `inference_lab_results.ipynb`. Six subsections (A–F). Each subsection produces a **(predicted, measured, gap, hypothesis)** quadruple.

**Working rule:** every number in this appendix comes from you, by hand, before any tool call. If you can't derive it on paper, you don't write it down.

### 2.5.A — Hardware reference card

**What to produce:** a markdown table written once and reused throughout the appendix.

**For each (chip, quant) combo you used in Stage 2:**
- Peak FLOPS — be explicit about which path (tensor core vs SIMT vs INT4)
- HBM bandwidth
- The roofline ratio α_hbm
- β (the bits-per-param / bits-per-activation ratio for your quant)
- B_crit derived from α and β

**Pointer:** Reference doc §1 has the formula and an example. Verify the chip specs against NVIDIA's datasheet (link the source in your card).

**Acceptance:** every number cites a source URL; the formula it came from is written in the cell.

### 2.5.B — Test 5 batch-sweep reconciliation

**What to produce:**
- The Pareto curve from your existing Test 5 data (latency on x, throughput on y).
- A vertical line at your predicted B_crit from 2.5.A.
- Roofline efficiency (%) annotated at each batch point.
- One paragraph explaining the gap between your measured knee, your peak, and the predicted B_crit. Hypothesize where each one comes from.

**Pointer:** Reference doc §6 has the closed-form throughput equation. You'll need KV-per-token from §3 (Qwen2.5-7B-AWQ — derive it from the model card's `n_kv_heads`, `head_dim`, `n_layers`, and the activation dtype).

**Acceptance:** plot exists with predicted line; efficiency at peak batch is a real number; gap hypothesis is written even if not yet confirmed.

### 2.5.C — Test 14 prefill/decode master-equation reconciliation

**What to produce:**
- Predict prefill time for 1000 tokens from first principles.
- Predict per-token decode time from first principles. Pay attention to your quant — AWQ-4bit weights mean the bytes-moved number is *not* the FP16 number.
- Compute efficiency (measured / predicted) for both phases.
- If decode efficiency is low, propose 2–3 candidate causes and design a one-shot follow-up experiment to discriminate between them.

**Pointer:** Reference doc §6, general-case step-time equation. The "missing term" your reviewer didn't mention is scheduler/Python overhead — your B and 2.5.B data can isolate it.

**Acceptance:** four numbers (prefill predicted, prefill measured, decode predicted, decode measured), gap %, ≤2 surviving root-cause candidates.

### 2.5.D — Long-context-decode KV-bandwidth-domination test (new test)

**What to produce:**
- A new test at batch=1, sweeping context length across at least 5 points spanning ~1k to ~32k.
- Decompose predicted decode-step time into two terms: one constant in context length, one linear.
- Plot measured TBT vs context length. Annotate predicted slope and predicted intercept.
- Identify the crossover context length where the KV term dominates the weight term.

**Pointer:** Reference doc §3 for KV math, §6 for the step-time decomposition. The crossover is a single inequality — solve it on paper before measuring.

**Acceptance:** measured slope within a factor your stated tolerance (you choose 2× or 30% — defend your tolerance); crossover point written down.

### 2.5.E — Test 25 cliff prediction

**What to produce:** predict the cliff RPS *before* re-reading Test 25's measured cliff.
- KV memory budget = HBM − weights − headroom. Define each term.
- Max in-flight tokens.
- Max concurrent requests from your workload's avg sequence length.
- Predicted cliff RPS from concurrent-requests / per-request generation time.
- Compare to Test 25's measured cliff. Explain the gap.

**Pointer:** Reference doc §3 (KV math) and §6 (step-time → request time).

**Acceptance:** prediction within a stated factor; gap explanation grounded in *one* mechanism (block allocator, chunked prefill, KV layout, etc.), not a list.

### 2.5.F — P:G ratio from existing data

**What to produce:**
- The disaggregation equilibrium ratio for your workload, computed from Test 14 + Test 5.
- An explicit caveat paragraph about the data heterogeneity (different load conditions when each was measured).
- A comparison to the Scaling Book's 3:1 for Llama-70B at 8k/512, explaining why a small AWQ model on T4 lands somewhere different.

**Pointer:** Reference doc §7 has the Llama-70B disagg ratio derivation. Apply the same formula to your numbers.

**Acceptance:** one number + one caveat paragraph + one comparison paragraph.

### Stage 2.5 acceptance gate

Before declaring 2.5 done:
- Six subsections present, each with the **(predicted, measured, gap, hypothesis)** quadruple.
- If any subsection's prediction is off by more than 5×, **stop** — the framework or measurement is wrong; diagnose before continuing to 2.7.
- One summary cell at the top of the appendix listing the six predictions and how close they landed. This is the interview-ready artifact.

---

## Stage 2.6 — Long-lead infrastructure prep (parallel with 2.5)

**Run in parallel with 2.5, do NOT wait until 2.5 is done.**

**Day-1 task:** file the spot G/VT quota increase via AWS Console. Default is 32 vCPU; request at least 96 vCPU. Approval is typically 0–24h.

**Days 2–3 (after 2.5 main work):**
- Provision g6.12xlarge spot, attach a persistent EBS volume.
- Download to that volume:
  - Qwen2.5-7B-Instruct-AWQ (already used in Stage 2)
  - Llama-3.1-8B-Instruct + Llama-3.2-1B-Instruct (spec-decoding pair)
  - DeepSeek-V2-Lite-Chat (MLA)
  - Qwen1.5-MoE-A2.7B-Chat (single-A10G-capable MoE)
- Smoke test: each model loads in vLLM, generates one valid response.

**Acceptance:** EBS snapshot exists with all four models; each tested loadable. Instance torn down (do NOT leave running — Phase 4.1 surprise lesson).

---

## Stage 2.7 — Architectural Breadth Experiments (~3 weeks after 2.5)

Serial, in this order — each builds intuition for the next.

### 2.7.a — Tensor Parallelism scaling (1 week)

Llama-3.1-8B at TP ∈ {1, 2, 4} on g6.12xlarge. Open-loop ramp.

**Predict first**, measure second:
- Latency floor at TP=N from the parallelism math in reference doc §5.
- The TP where interconnect overhead flips the latency benefit (g6 = PCIe gen4, *not* NVLink — verify).

**Acceptance:** plot of latency vs TP with predicted floor; identify the TP minimizing $/Mtok and the TP minimizing latency. Explain why they differ.

### 2.7.b — Speculative decoding (3–4 days)

Single A10G. Llama-3.1-8B target, Llama-3.2-1B draft.

Three workloads: chat, code, structured JSON.

**Predict first:** acceptance rate ordering; speedup from the acceptance-rate formula.

**Acceptance:** acceptance-rate ordering matches prediction; speedup-by-workload plot; one-line "when to enable spec decoding" rule for Phase 5.

### 2.7.c — MLA empirical comparison (3–4 days)

A10G. Qwen2.5-7B-AWQ (GQA) vs DeepSeek-V2-Lite (MLA), same prompt/output/context.

**Predict first:** max batch ratio from KV-per-token table in §3.

**Acceptance:** measured KV-per-token within 20% of formula; bar chart of max concurrent sessions GQA vs MLA.

### 2.7.d — MoE inference (4–5 days)

Qwen1.5-MoE-A2.7B on A10G (or TP=2 if needed).

**Predict first:** B_crit shift factor for MoE from reference doc §5.

**Acceptance:** measured throughput plateau within 2× of predicted B_crit; one-paragraph "why MoE needs aggregated batch" explanation.

---

## Stage 2.8 — Single profiled kernel on roofline (defer)

torch.profiler on a vLLM decode step, one FFN GEMM, real dot on a real roofline. **Bank for post-2.7 polish week.** Not before 2.5+2.7 are solid.

---

## Decision checkpoints

| When | Question | If yes → | If no → |
|---|---|---|---|
| End of 2.5 | Did predictions land within a factor-of-2 of measurements consistently? | Proceed to 2.7 confidently | Stop — diagnose framework or measurement before building more |
| End of 2.5.D | KV-BW domination plot shows the regime transition cleanly? | Phase 5 prefix caching has a clean story to attach to | Re-instrument before Phase 5 |
| End of 2.7.a | Measured TP scaling matches predicted? | Phase 8 disaggregation has grounded latency math | Investigate interconnect (PCIe ≠ NVLink confusion is common) |
| End of 2.7.b | Spec-decoding > 1.3× speedup on at least one workload? | Bake into Phase 5 | Deprioritize, revisit at Phase 7 |

## Cost & ops hygiene

- All g6.12xlarge usage on Spot, torn down nightly (Phase 4.1 surprise lesson).
- Sweep result files namespaced per run (`stage25_sweep_v1.json`, etc.).
- Each notebook section commits to git with the scripts that produced its numbers — reproducibility is a deliverable.

## Timeline

| Week | Stage |
|---|---|
| This week | 2.5 (A–F) + file quota bump |
| Next week | 2.6 (provision + smoke test) |
| +1 | 2.7.a (TP) |
| +2 | 2.7.b (spec decoding) + 2.7.c (MLA) |
| +3 | 2.7.d (MoE) + writeup |
| +4 | Phase 5 design |

## End-of-plan artifact

For each experiment: **prediction → measurement → gap → implication.** That's the structure you walk into a Mahesh/Shiva conversation with. Not "I ran these tests" but "I predicted, I measured, I closed the loop, and here's what I'd do differently in production."
