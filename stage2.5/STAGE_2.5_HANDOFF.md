# Stage 2.5 Handoff — Sizing Walkthrough Session

Paste this (or reference its path) at the start of the next session.

---

## TL;DR

We're mid-Stage 2.5 (the "roofline reconciliation" appendix that converts the inference lab from operational characterization to predictive understanding). Mohamed has the framework but needs to connect the dots into a coherent sizing workflow before continuing with empirical reconciliation.

**Immediate next task:** walk through a complete sizing exercise end-to-end on a hypothetical SLO + workload, using the 9-step sizing workflow. This builds the big-picture mental model that makes the per-test reconciliation in 2.5.B/C feel motivated rather than mechanical.

**Mode:** Socratic by default (derivations, math, decisions all done by Mohamed); drop Socratic when Mohamed explicitly asks to be walked through, then resume. Memory `feedback_socratic_math.md` governs this.

---

## Reference docs to load at session start

1. **`MODEL_SIZING_SCALING_REFERENCE.md`** (repo root) — the 13-section framework. §0 has the 9-step workflow this walkthrough exercises. §1, §3, §5, §6 contain the equations.
2. **`PRE_PHASE5_PLAN.md`** (repo root) — Stage 2.5 acceptance criteria. The walkthrough produces the *predictive* artifact that 2.5.B/C then *reconcile* against measurements.
3. **Memory files** (auto-loaded via `MEMORY.md`): `feedback_socratic_math`, `feedback_aws_spot_gpu_quota`, `project_sizing_reference`.

---

## Where we are in Stage 2.5

| Subsection | Status |
|---|---|
| 2.5.A — Hardware reference card | ✅ T4 α=203, A10G α=208 derived |
| Detour: derive `B_crit` from first principles | ✅ Done — `B_crit = α × bytes_w / 2`; doc form `α × β` is FP16-anchored shortcut |
| Detour: attention FLOPs vs weight FLOPs ratio | ✅ Done — at (B=32, T=4k, Qwen-7B) ratio ≈ 0.11, threshold `T_crit = P/(2DL)` |
| Detour: Test 5 knee diagnosis | ✅ Predicted B_crit = 51; measured peak ≈ 60 (good match); measured knee at 16 well below — KV crossover at 220 rules out KV bandwidth; remaining hypothesis: fixed-cost-per-step engine overhead |
| 2.5.B — Test 5 reconciliation appendix write-up | 🟡 Math done, write-up pending |
| 2.5.C — Test 14 prefill/decode reconciliation | ⏸ Not started |
| 2.5.D — Long-context KV-BW domination test | ⏸ Not started (new test, needs run) |
| 2.5.E — Test 25 cliff prediction | ⏸ Not started |
| 2.5.F — P:G ratio from existing data | ⏸ Not started |
| **THIS WALKTHROUGH** — sizing-workflow big-picture exercise | 🆕 To do first in next session |

---

## The walkthrough setup

**Hypothetical scenario to size:**

> Chat product. TBT_p99 ≤ 50 ms. Input p50 = 500 tokens, output p50 = 300 tokens. QPS target = 5. Running Qwen2.5-7B-AWQ on T4 spot.

We walk through the 9-step workflow end-to-end. Mohamed does all the math; the assistant points to where in the reference doc each formula lives and challenges assumptions. Outcome: a predicted (B, replica_count) sizing decision with quantified latency and throughput estimates — then ready to validate against Stage 2 measurements.

---

## Mohamed's open questions to triage first

> Mohamed has questions before starting the walkthrough. Fill these in before kicking off, so the session opens with answering them rather than diving into math cold.

- [ ] So the 9 steps framework's objective is to find the B and replica count, is that correct?
- [ ] How does parallelism calculations get factored in? 
- [ ] I feel that i might be misunderstanding what these paramters from TRT-LLM (and their vLLM equivalent mean), worth redefining them before we move on: max_batch_size, max_input_len, max_output_len, max_num_tokens and what will happen if any of them get exceeded?

Recommended order at session start:
1. Answer Mohamed's pre-flight questions one at a time.
2. Confirm the hypothetical scenario above (or modify).
3. Walk through steps 1–9 of the sizing workflow.
4. End with a written sizing artifact + decision on whether to continue with 2.5.B–F or take a break.

---

## Mode reminders for the assistant

- **Socratic by default.** Don't pre-compute α, B_crit, KV-per-token, T_step, etc. Point to where in the reference doc the formula lives; let Mohamed derive.
- **Drop Socratic when Mohamed explicitly asks** ("walk me through", "tell me the answer", "give me the formula"). Resume Socratic on the next derivation.
- **Always cite units** when handing back a number. Mohamed has caught two unit-related confusions in the prior session.
- **Don't conflate prefill and decode regimes** — every formula has a different shape in each. Always specify which when deriving.
- **No new Bash tool calls to verify quota** — last verified 2026-05-12, account at 64 vCPU spot G/VT in us-east-1. Plenty for the sizing walkthrough (no infra changes needed).
- **Keep the assistant from drifting into 2.5.B/C/D/E/F too early.** The walkthrough is the priority for the next session. Empirical reconciliation comes after.

---

## Context already established (don't re-derive)

These are settled in the prior session — reuse, don't re-litigate:

- **Hardware ratios:** T4 α_hbm = 203 FLOPs/byte (FP16 tensor); A10G α_hbm = 208.
- **B_crit derivation:** `B_crit = α × bytes_w / 2` is the first-principles form. Doc shortcut `α × β` is only valid with FP16 activation anchor; mixing conventions double-counts precision change.
- **T_crit for attention vs weight FLOPs:** `T_crit = P / (2 × D × L)`. Below it, weight matmul dominates; above, attention dominates.
- **Test 5 setup:** prompt=200 tok, output=200 tok, same prompt for all (prefix caching effects possible), `avg_tokens_generated`=200 everywhere (no early termination).
- **Test 5 measured:** peak 1105 tok/s at B=64; knee at B=16 (74% of peak).
- **Predicted (Qwen2.5-7B AWQ on T4):** B_crit = 51; KV-vs-weight crossover B = 220 at T=300; weight-load floor ≈ 11.88 ms.
- **Hypothesis for early knee at 16:** fixed-cost-per-step engine overhead amortization, not KV bandwidth.
