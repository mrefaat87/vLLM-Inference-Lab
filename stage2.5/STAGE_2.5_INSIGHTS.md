# Stage 2.5 — Key Insights

The non-obvious things that stood out from deriving the roofline framework by hand. Each is a place where the surface-level math hides an assumption or convention worth knowing.

---

## 1. α is FP16-anchored

The Scaling Book's compact form `B_crit ≈ α × β` only holds **if α is treated as a hardware constant defined at FP16** and precision changes flow through `bits_a` (the β denominator). Mixing conventions — recomputing α at FP8 *and* halving `bits_a` — double-counts the precision change and gives 4× when the right answer is 2×.

**The convention-free form, derived from first principles:**
```
B_crit = α × bytes_per_weight / 2
```
where α uses FLOPs peak at the *actual* compute precision. No hidden anchor. Harder to misuse.

**Why this matters in an interview:** if asked to derive B_crit for an FP8 workload, going through the first-principles form keeps you out of the trap. The `α × β` form is a shortcut that requires remembering its FP16 anchor.

---

## 2. B_crit is a hardware **ceiling**, not a sizing **target**

The big mental-model correction. B_crit is a property of `(hardware × model × quant)` — it's where throughput plateaus and latency starts climbing linearly. You don't *target* B_crit.

**The right sizing flow:**
1. Start from the **SLO**.
2. Invert `T_step(B) ≤ SLA` to find the largest B that respects latency.
3. Check that B is below B_crit (it almost always is in latency-sensitive serving).
4. Compute throughput at that B → if insufficient, **add replicas**, not more batch.

You never "search for max throughput with low latency" because those goals oppose each other. You pick a point on the Pareto curve set by your SLO.

---

## 3. Attention is **compute-bound in prefill, bandwidth-bound in decode**

Both regimes do many attention FLOPs, but their physics differ:

| Regime | Arithmetic intensity | Bottleneck |
|---|---|---|
| Prefill (T queries × T keys) | ~T/2 (high) | FLOPs |
| Decode (1 new query × T keys) | ~1 (low) | KV-read bandwidth |

In decode, every byte of KV streamed produces ~1 FLOP — the hardware idles through the math waiting on HBM. That's why §3 / §10 of the reference doc obsess about KV *size* (MQA, MLA, cross-layer sharing) rather than attention *FLOPs* for decode workloads. **The lever you actually have during decode is bandwidth, not compute.**

---

## 4. Engine overhead is a real term that pure roofline ignores

Test 5's measured throughput knee at B=16 vs predicted memory-bound regime extending to B=51 = ~3× gap. KV bandwidth doesn't explain it (KV-vs-weight crossover at B=220, far above the sweep). What explains it: **fixed-cost-per-step** — scheduler, Python orchestration, kernel launch, attention metadata setup.

This term doesn't appear in chip-level roofline math. But at small batch (B < ~16 for vLLM on T4), it dominates step time, suppressing the linear-scaling regime the master equation predicts.

**Sizing implication:** the master equation gives an upper bound, not a forecast. Real systems have a fixed-overhead floor that you can't reason away — you measure it. Test 5 reconciliation surfaces it; pure analytical sizing misses it.

---

## 5. "B = sequences vs B = tokens" depends on prefill vs decode

Same letter, different physical meaning:

- **Prefill:** B in `2 × B × P` = tokens-per-step = `sequences × prompt_length`.
- **Decode:** B in `2 × B × P` = tokens-per-step = `sequences` (one new token per sequence per step).

This trips up almost every first derivation. The reason: prefill processes all prompt tokens in parallel; decode processes one new token per sequence per step.

**Bonus subtlety: in GQA, attention FLOPs use `n_query_heads`, not `n_kv_heads`.** GQA shrinks the KV *cache* (memory), not the attention *compute* (FLOPs). Every query head still runs its own QK^T against its group's key head.

---

## 6. Variable-S workloads have a hidden coupling — the longest sequence in the batch sets TBT for everyone

The clean form of decode `T_step` is:

```
T_step(B, S) ≈ (weights_bytes + B × KV_per_token × S) / HBM_BW + T_overhead
```

Two things this exposes that the "single B" framing hides:

**1. `B_SLA` is a function of S, not a constant.** As S grows, `B_SLA(S) ≈ (SLO_TBT − T_overhead − weights/HBM_BW) × HBM_BW / (KV_per_token × S)` — **inversely proportional to context length**. Double the context, halve the safe concurrency. This is the long-context scaling cliff: long-context serving isn't just KV-capacity-bound, it's KV-bandwidth-bound, and bandwidth pressure scales linearly with S.

**Sizing implication:** Don't size B_SLA against `max_model_len`. Size against **expected p95 context length** from your traffic. If `max_model_len=4096` but typical S=800, real B_SLA is ~5× larger than worst-case math suggests. Sizing for the tail and serving everyone at that cap is the most common sizing mistake.

**2. Continuous batching means the longest sequence in the running batch sets `T_step` for the entire batch.** Each decode step reads KV for every sequence in the batch — total bytes scale with `Σ S_i`, not `B × mean(S)`. The slowest-to-finish sequence dominates the bandwidth term while it's still running.

So a single long-context request in a batch of short ones doesn't just hurt itself — it stretches TBT for every co-resident sequence. The "free" parallelism of continuous batching has a floor set by the longest-S resident.

**Production corollary — segregate long from short.** This is the hidden case for **length-aware routing**: route short and long requests to different replicas, not just for KV capacity reasons but for TBT consistency. Without segregation, a single 4K-context arrival can blow TBT for a batch of 400-context interactive requests sharing the replica. With segregation, the long-context replica runs at its own (lower) B_SLA without contaminating the interactive fleet.

This is also the deeper reason **prefill/decode disaggregation** (DistServe, Mooncake) helps: it removes one source of length-dependent coupling. Length-aware routing within the *decode* tier removes the other.

**AWS analogy:** like sharing an ASG between latency-sensitive and batch workloads. The batch jobs' long-tail latency pulls p99 up for the interactive traffic even though "average utilization is fine." The fix isn't a smarter scaling policy — it's **two ASGs with traffic shaping**, sized independently.

---

## The through-line

Clean equations look universal but each carries an unstated regime, convention, or approximation. The skill that matters is **spotting which assumption is breaking** when you apply a formula in unfamiliar territory — FP8 instead of FP16, MoE instead of dense, decode instead of prefill, real engine instead of idealized chip. Memorizing the textbook form gets you the right answer in the textbook example. Deriving it from first principles is what makes the framework usable in production.
