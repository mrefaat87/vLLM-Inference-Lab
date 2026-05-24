// Pure-function compute layer for the LLM inference sizing calculator.
//
// No DOM access, no globals, no I/O — every function takes plain numbers/objects
// and returns plain numbers/objects. This is what makes the formulas testable in
// node:test without a browser. Every formula cites a section in
// MODEL_SIZING_SCALING_REFERENCE.md so reviewers can trace numbers to source.

// ---------- constants ----------
// Reserve ~10% of HBM for activations + workspace + paging fragmentation.
// The reference doesn't pin this; it's a vLLM-style headroom. Note: actual
// vLLM default is gpu_memory_utilization=0.92 (vllm/config/cache.py), so our
// ACT_OVERHEAD=0.10 (→ 90% utilization) is intentionally conservative vs the
// shipped default. Tunable from the UI later if needed.
export const ACT_OVERHEAD = 0.10;
// Practical within-node TP fan-out. 8 is the standard NVLink island (HGX/DGX).
// The hardware ceiling. The *useful* TP ceiling is min(TP_CAP, Y_max) — see
// yMax() below.
export const TP_CAP = 8;
// max_num_batched_tokens hint band α from DJL TRT-LLM guidance: 0.1–0.2 × B × ISL.
// We use the midpoint as the centerpoint; the sweep widens around it.
export const ALPHA = 0.15;
// MFU midpoints. Reference §0.3 (with footnote citing sources) gives bands of
// ~30–50% prefill MFU and ~5–15% decode MFU, synthesized from:
//   - DeepSeek-V3 production inference system overview (Feb 2025): back-calculated
//     ~34% prefill MFU / ~7% decode MFU on H800 against FP8 peak (the prefill
//     number is inflated by ~56% disk KV-cache hits, so the true compute-only
//     prefill MFU is lower).
//   - "Investigation of FP8 Across Accelerators" (arXiv:2502.01070): square
//     (prefill-shaped) GEMM on H100 hits ~59% of FP8 peak; thin (decode-shaped)
//     GEMM collapses to <1% peak in adversarial batch=64 cases.
//   - Sarathi-Serve (OSDI '24): qualitative — "decode is memory-bound, low MFU"
//     — no published number, just the structural argument.
//   - Databricks LLM Inference Best Practices: deliberately reports MBU (Model
//     Bandwidth Utilization), not MFU, for decode (~55–60% at batch=1) because
//     MFU is the wrong axis for a memory-bound workload.
// No primary source publishes "the" canonical bands. The midpoints below are
// rule-of-thumb. The 4× ratio they imply drives the pdRatio multiplier (line
// ~313) — that ratio is *only* defensible to roughly ±2× (real range across
// sources: 3×–8×). Treat pdRatio as an order-of-magnitude disagg hint, not a
// calibrated capacity ratio. See §0.3 "On the MFU bands" footnote in
// MODEL_SIZING_SCALING_REFERENCE.md for the verbatim source quotes and URLs.
export const PREFILL_MFU = 0.40;
export const DECODE_MFU = 0.10;

// Byte costs by dtype name. Sub-byte precisions (INT4) stored as float so we
// never integer-divide a parameter count (caught as an edge-case bug class).
export const DTYPE_BYTES = Object.freeze({
  FP16: 2,
  BF16: 2,
  FP8: 1,
  INT8: 1,
  INT4: 0.5,
});

// ---------- low-level building blocks ----------

// Total weight memory across all GPUs (TP shards it, but the aggregate is what
// matters for the roofline since BW also aggregates). params_b_total is in
// billions, matching the data table; multiply by 1e9 here once so callers stay
// in "B". Note: this is the *total* parameter count — for MoE, all experts must
// sit in HBM regardless of routing. The compute-roofline formula uses active
// params via stepTime's paramCount_active, not this function.
export function weightsBytes(params_b_total, weight_bytes) {
  return params_b_total * 1e9 * weight_bytes;
}

// KV cache bytes per generated token, per sequence (reference §3).
// MHA/GQA/MQA share the generic formula:
//   2 (K+V) × bytes × n_kv_heads × head_dim × n_layers
// Factor of 2 covers K and V tensors; n_kv_heads (< n_heads for GQA, =1 for
// MQA) is what GQA actually compresses — which is why GQA's KV is so much
// smaller than MHA.
// MLA is structurally different (reference §3 table): a single low-rank latent
// of size d_c plus one rope key of size d_rope, shared across heads — so no
// factor of 2 (one cached tensor, not K+V separately), no n_kv_heads, no
// head_dim. Field names match DeepSeek HF config (kv_lora_rank, qk_rope_head_dim).
export function kvPerToken(model, kv_bytes) {
  if (model.attn_type === "MLA") {
    return (model.kv_lora_rank + model.qk_rope_head_dim) * kv_bytes * model.n_layers;
  }
  return 2 * kv_bytes * model.n_kv_heads * model.head_dim * model.n_layers;
}

// Roofline ridge-point batch (reference §1, Scaling Book inference chapter
// https://jax-ml.github.io/scaling-book/inference/):
//   B_crit = (C / W_hbm) × (bits_w / bits_act) × (N_total / N_active)
// where C is the *precision-adjusted* peak FLOPs — not a fixed FP16 rate, and
// the trailing factor is the Scaling Book inference Q5 "factor of E/k" MoE
// sparsity correction: weight memory streams ALL experts (N_total) per step
// while compute scales only with the ACTIVE experts per token (N_active), so
// the ridge batch shifts right by exactly that ratio. For dense models
// N_total = N_active, sparsity = 1, and the formula reduces to the original.
//
// The book's rule on precision: "If we do our FLOPs in int8 or fp8, B_crit
// increases by 2x." FP8 tensor cores fire only when BOTH operands are FP8 AND
// the hardware exposes an FP8 path (cuBLAS GEMM rule), so FP8/FP8 picks
// fp8_tflops (2× on H100 → 590) while BF16/BF16 stays on fp16_tflops (295).
// Asymmetric cases (e.g., INT8 weights + BF16 acts) ride the FP16 compute path
// and shift only through bits_w/bits_act (8/16 → halves B_crit, matching the
// book's TPU v5e int8-weights/bf16-acts → 120 example).
//
// Worked MoE example (DeepSeek-V3 on H100, BF16/BF16):
//   dense B_crit = 295; sparsity = 671/37 ≈ 18.135; B_crit_MoE ≈ 5347.
export function bCrit(hw, model, weight_prec, act_prec) {
  const useFp8 = weight_prec === "FP8" && act_prec === "FP8" && hw.fp8_tflops;
  const flops = (useFp8 ? hw.fp8_tflops : hw.fp16_tflops) * 1e12;
  const bw = hw.hbm_bw_gbs * 1e9;
  const bits_w = DTYPE_BYTES[weight_prec] * 8;
  const bits_act = DTYPE_BYTES[act_prec] * 8;
  // Scaling Book inference Q5: B_crit_MoE = B_crit_dense × (E/k) = N_total/N_active.
  // Dense models: params_b_total === params_b_active → sparsity = 1.
  const sparsity = model.params_b_total / model.params_b_active;
  return (flops / bw) * (bits_w / bits_act) * sparsity;
}

// Time per decode step (seconds), per-kernel form (reference §4 + Scaling Book
// inference chapter, https://jax-ml.github.io/scaling-book/inference/):
//
//   Step_time = (B · KV_per_seq) / BW                       ← attention kernel
//             + max( 2·B·N / FLOPs ,  W / BW )              ← MLP kernel
//
// Why additive, not a single max: attention and MLP run as *separate sequential
// CUDA kernels* per layer, so per-step wall time sums across the two kernels.
// The right-hand max picks the MLP kernel's bottleneck (compute roof vs weight
// load). The book's justification for the asymmetry: "the attention component
// (left) is never compute-bound, and thus doesn't need a FLOPs roofline" — so
// attention is modelled as pure HBM-bandwidth on the KV stream only.
//
// MoE contract (formerly a verbatim-from-book simplification): the two MLP-
// branch terms read DIFFERENT param counts. Memory branch uses
// `weight_bytes_total` (full HBM weight stream — all MoE experts); caller MUST
// pass `weightsBytes(model.params_b_total, ...)`. Compute branch uses
// `paramCount_active` (per-token FLOPs scale with active params only — MoE
// compute sparsity DOES reduce 2BN/FLOPs). For dense models both fields are
// equal and behavior is unchanged vs the original single-N formulation.
//
// seq is tokens-of-KV per active sequence during the step we're modelling. Use
// PEAK occupancy (ISL+OSL — the last decode step) rather than the mean. Reason:
// production SLAs (e.g., "TBT p99 < 50ms") bound EVERY step, and the last step
// is the slowest because KV is largest then. Sizing against the mean fits
// batches that meet TBT on average but blow SLO at end-of-decode; sizing
// against peak is the SLO-conservative answer. The asymmetry vs bKv's peak
// usage is gone — both bounds now agree on what "fits" means.
export function stepTime({ B, kv_per_token, seq, weight_bytes_total, paramCount_active, totalBW, totalFLOPs }) {
  // Attention kernel: stream the live KV of every sequence each step. No compute
  // roof — attention's AI is ~1, always memory-bound at decode (reference §0.3).
  const attnMemTime = (B * kv_per_token * seq) / totalBW;
  // MLP kernel: weight-load vs compute, whichever is slower. W/BW is constant
  // in B (the per-step weight stream is the same regardless of how many
  // tokens share it); 2BN/FLOPs grows linearly in B. N here = active params.
  const mlpMemTime = weight_bytes_total / totalBW;
  const computeTime = (2 * B * paramCount_active) / totalFLOPs;
  return attnMemTime + Math.max(mlpMemTime, computeTime);
}

// Aggregate decode throughput in tokens/sec at batch B. One decode step emits
// exactly one new token per active sequence (autoregressive invariant), so
// B tokens per step → tps = B / stepTime. This is the headline tok/s number
// vendors quote — NOT per-user rate (= 1/stepTime). Inherits stepTime's
// physics: rises ~linearly with B below B_crit, then saturates at the
// compute ceiling FLOPs/(2N) above (where B cancels in the asymptote because
// stepTime is itself linear in B). Delegating to stepTime keeps one source
// of truth — the per-kernel KV/MLP/compute breakdown lives there, not here.
export function throughputAtB(args) {
  const t = stepTime(args);
  return args.B / t;
}

// $/Mtok output at the given throughput. Pure unit conversion — no new physics.
//
//   fleet cost rate = price_per_hour_usd × ngpus               $/hr
//   fleet emits     = tps × 3600                                tokens/hr
//   $/token         = (price × ngpus) / (tps × 3600)            $/tok
//   $/Mtok          = (price × ngpus × 1e6) / (3600 × tps)
//
// Reference §0.1 Pareto framing (latency × throughput × $/Mtok) — closes the
// third axis of the cost surface the calculator was previously missing.
//
// Edge cases: price=null returns null (custom hardware with blank price → UI
// renders the m-cost tile as "—"); tps=0 returns Infinity (caller masks before
// display); price=0 returns 0 (free hardware = free tokens).
export function costPerMtok({ price_per_hour_usd, ngpus, tps }) {
  if (price_per_hour_usd === null || price_per_hour_usd === undefined) return null;
  if (tps === 0) return Infinity;
  return (price_per_hour_usd * ngpus * 1e6) / (3600 * tps);
}

// ---------- batch-size bounds ----------

// Largest B that keeps a decode step under TBT_SLO. Solves the per-kernel
// stepTime form (reference §4 + Scaling Book):
//   T(B) = (B·K·S)/BW + max(W/BW, 2BN/FLOPs)
//
// The right-hand max is monotone in B only through the compute branch
// (W/BW is constant in B). The MLP-ridge crossover is at:
//   B_ridge = (W · FLOPs) / (2N · BW)
//
// Two regimes, both linear in B:
//   (a) Below the ridge (W/BW dominates): T = (B·K·S)/BW + W/BW
//       Solve: B ≤ (T_slo − W/BW) · BW / (K·S)
//       Unreachable if T_slo·BW ≤ W (weight stream alone busts budget).
//   (b) Above the ridge (compute dominates): T = (B·K·S)/BW + 2BN/FLOPs
//       Solve: B ≤ T_slo / (K·S/BW + 2N/FLOPs)
//
// Strategy: compute both candidate bounds; the binding answer is the candidate
// whose B *actually lands in its own regime* (i.e., on its own side of B_ridge).
// At small B regime (a) is correct; at large B regime (b) is correct. The
// boundary B_ridge is also a valid answer when both candidates straddle it.
// Variables: W = weight_bytes_total (total HBM weight stream — for MoE, all
// experts); N = paramCount_active (per-token compute scales with active params
// only — the 2BN/FLOPs term). Dense models: total === active, asymmetry is a
// no-op.
//
// Note on B_ridge vs B_crit: these answer different questions. B_ridge is the
// MLP weight-load-vs-compute crossover *within this simplified stepTime model*
// (no activation streaming). B_crit (in bCrit() above) is the per-matmul
// arithmetic-intensity threshold that DOES include activation memory via the
// bits_w/bits_act factor. They coincide numerically for symmetric precision
// (BF16/BF16, INT8/INT8) by algebraic coincidence, but diverge for FP8/FP8
// (B_ridge ≈ α, B_crit ≈ 2α). The Scaling Book inference chapter explicitly
// omits activations from the stepTime model ("activation costs are largely
// ignored — negligible for both prefill and generate"), so B_ridge is the
// correct boundary for this solve. Do NOT replace B_ridge with bCrit() —
// the two regime expressions below would no longer match the boundary and
// the solve would silently report falsely-permissive values for FP8/FP8.
export function bSlo({ tbt_ms, kv_per_token, seq, weight_bytes_total, paramCount_active, totalBW, totalFLOPs }) {
  const T = tbt_ms / 1000;
  const W = weight_bytes_total;
  const N = paramCount_active;
  // KS uses PEAK seq (ISL+OSL) — see stepTime() comment for the SLO-vs-mean
  // rationale. This makes bSlo answer "what's the max B that meets TBT on the
  // WORST decode step?" rather than "...on the average step", which is the
  // production-correct framing for any p99-style SLA.
  const KS = kv_per_token * seq;

  const wOverBW = W / totalBW;
  // Regime-(a) candidate: weight-load dominates the MLP kernel.
  const numA = T - wOverBW;
  if (numA <= 0) {
    return { value: 0, unreachable: true, reason: "weights stream time alone exceeds TBT budget" };
  }
  const bA = (numA * totalBW) / KS;
  // Regime-(b) candidate: MLP compute dominates.
  const bB = T / (KS / totalBW + (2 * N) / totalFLOPs);
  // Ridge: the B at which W/BW == 2BN/FLOPs (MLP kernel crossover).
  const bRidge = (W * totalFLOPs) / (2 * N * totalBW);

  // Pick the candidate that lands in its own regime. Below the ridge: (a)
  // governs. Above: (b) governs. If they cross, the binding one is the smaller
  // of the two valid pieces (the step time is piecewise-linear and continuous
  // at the ridge).
  let value, binding;
  if (bA <= bRidge) {
    // Candidate (a) is consistent (its solution sits in the W/BW-dominated regime).
    value = bA;
    binding = "mlp-weight-load";
  } else if (bB >= bRidge) {
    // Candidate (b) is consistent (its solution sits in the compute-dominated regime).
    value = bB;
    binding = "mlp-compute";
  } else {
    // Neither candidate is self-consistent → the SLO is met exactly at the
    // ridge. Use B_ridge.
    value = bRidge;
    binding = "mlp-ridge";
  }

  // Attention term is the only remaining axis: if KS·B/BW alone already exceeds
  // T (would only happen at absurd KV·seq), flag binding as attention.
  if ((value * KS) / totalBW > T) {
    binding = "attention";
  }
  return { value: Math.max(0, Math.floor(value)), unreachable: false, binding };
}

// Largest B that fits the KV cache in HBM after weights + activation headroom
// (reference §3, §5). Uses peak occupancy ISL+OSL — end-of-decode is when KV
// is largest, so this is the conservative bound that won't OOM mid-generation.
//
// Owns the KV-per-token derivation: calls kvPerToken(model, kv_bytes)
// internally so the right per-attention-variant formula (MHA / GQA / MQA / MLA)
// is always used. Earlier shape accepted a pre-computed kv_per_token from the
// caller, which made it possible for a caller to silently pass a wrong number
// (e.g., the MHA formula on an MLA model). Returns the K it derived so callers
// that also need it (chart curve, stepTime) don't recompute.
//
// weight_bytes_total must be the TOTAL HBM weight stream — for MoE, all
// experts sit in HBM regardless of routing. Compute sparsity does NOT shrink
// HBM footprint.
export function bKv({ hw, model, ngpus, weight_bytes_total, kv_bytes, isl, osl }) {
  const totalHBM = hw.hbm_gb * ngpus * 1e9;
  const usable = totalHBM * (1 - ACT_OVERHEAD) - weight_bytes_total;
  const kv_per_token = kvPerToken(model, kv_bytes);
  if (usable <= 0) {
    return { value: 0, weights_overflow: true, kv_per_token };
  }
  const peakSeq = isl + osl;
  if (peakSeq <= 0 || kv_per_token <= 0) {
    return { value: 0, weights_overflow: false, kv_per_token };
  }
  return {
    value: Math.floor(usable / (kv_per_token * peakSeq)),
    weights_overflow: false,
    kv_per_token,
  };
}

// ---------- TP usefulness ceiling (Scaling Book Y_max) ----------

// Max useful TP degree per replica before ICI all-reduce dominates per-step
// decode latency. Past this many chips, adding more TP doesn't cut step time
// (and may make it worse). Reference §6 (Scaling Book derivation):
//
//   T_HBM = 2DF/(Y·W_hbm)   weight-stream per layer (Wup + Wdown = 2DF bytes)
//   T_ICI = 2BD/W_ici       all-reduce of activations per layer
//   T_ICI > T_HBM  ⇒  Y > F / (B · β)   with β = W_hbm/W_ici
//
// d_model and n_layers cancel out (they scale both terms equally) — only d_ff
// survives. β is hardware-specific: H100 ≈ 3.7, A100 ≈ 3.4, TPU v5e ≈ 8.
// Returns +Infinity when ici_bw_gbs is missing (treat as "no constraint
// modelled" — caller still applies the TP_CAP and memory-fit ceilings).
export function yMax(d_ff, batch, hw) {
  if (!hw.ici_bw_gbs || !d_ff || batch <= 0) return Infinity;
  const beta = hw.hbm_bw_gbs / hw.ici_bw_gbs;
  return d_ff / (batch * beta);
}

// ---------- parallelism ----------

// Recommend TP × PP × replicas. Two ceilings interact:
//   1. Memory fit: smallest TP ∈ {1,2,4,8} where the weight slice fits per GPU.
//   2. Latency usefulness (Y_max): adding TP past Y_max stops cutting decode
//      latency because ICI is now the floor. We don't *force* TP down to
//      Y_max (if memory needs TP=4 we use TP=4 even if Y_max=2), but we surface
//      a diagnostic so the user knows TP=4 was a memory-fit choice, not a
//      latency choice.
// ngpus is total GPUs across the deployment; replicas = serving units that fit.
export function recommendParallelism({ hw, weight_bytes_total, ngpus, d_ff, batch }) {
  // weight_bytes_total must be the TOTAL HBM weight stream — for MoE, all experts sit in HBM regardless of routing. Compute sparsity does NOT shrink HBM footprint.
  const hbm_per_gpu = hw.hbm_gb * 1e9 * (1 - ACT_OVERHEAD);
  const ymax = yMax(d_ff, batch, hw);
  // Useful TP ceiling: NVLink fan-out (8), GPU count, and Y_max combined.
  const usefulCap = Math.min(TP_CAP, ngpus, Math.max(1, Math.floor(ymax)));
  const tpCandidates = [1, 2, 4, 8].filter((t) => t <= ngpus);
  let tp = null;
  for (const t of tpCandidates) {
    if (weight_bytes_total / t <= hbm_per_gpu) { tp = t; break; }
  }
  // Build a structured diagnostic about ICI-binding so the UI can surface it.
  const ici_binds = Number.isFinite(ymax) && ymax < TP_CAP;
  if (tp !== null) {
    const replicas = Math.floor(ngpus / tp);
    let reason = tp === 1
      ? "weights fit on a single GPU; TP=1 maximizes replica count"
      : `weights need TP=${tp} to fit per-GPU HBM with ${Math.round(ACT_OVERHEAD * 100)}% headroom`;
    if (ici_binds && tp > usefulCap) {
      reason += ` (memory needs TP=${tp}, but Y_max=${ymax.toFixed(1)} — beyond ICI usefulness ceiling)`;
    } else if (ici_binds) {
      reason += ` (Y_max=${ymax.toFixed(1)} caps useful TP regardless of memory)`;
    }
    return { tp, pp: 1, replicas, fits: true, reason, y_max: ymax, useful_cap: usefulCap };
  }
  // No TP value in {1,2,4,8} fits — need pipeline parallelism. Use TP=8 per
  // stage (NVLink ceiling) and split layers across PP stages.
  const tpUsed = Math.min(8, ngpus);
  const pp = Math.ceil(weight_bytes_total / (tpUsed * hbm_per_gpu));
  const replicaGpus = tpUsed * pp;
  if (replicaGpus > ngpus) {
    return {
      tp: tpUsed, pp, replicas: 0, fits: false, y_max: ymax, useful_cap: usefulCap,
      reason: `needs TP=${tpUsed}×PP=${pp}=${replicaGpus} GPUs per replica; only ${ngpus} available`,
    };
  }
  return {
    tp: tpUsed, pp, replicas: Math.floor(ngpus / replicaGpus), fits: true,
    y_max: ymax, useful_cap: usefulCap,
    reason: `weights too large for TP-only; need PP=${pp} stages (TP=${tpUsed} per stage)`,
  };
}

// ---------- max_num_batched_tokens ----------

// Recommends a starting value for vLLM's --max-num-batched-tokens flag (the
// scheduler's token budget per forward pass — prefill + decode tokens combined,
// the chunked-prefill knob). Returns { value, band } where `band` is a label
// for the UI tile.
//
// This is a HEURISTIC, not a derived bound. It blends two pieces of guidance:
//
//   1. DJL TRT-LLM "sensible band" hint (α × batch × ISL, α∈[0.1, 0.2]):
//      bigger workloads need a bigger token budget so prefill chunks don't
//      starve. We use the midpoint α=0.15. The α value itself is rule-of-thumb,
//      not derived from a primary benchmark — treat as a starting hint.
//
//   2. Sarathi-Serve OSDI '24 (arXiv 2403.02310v3) §5.1 — token-budget bands
//      indexed by SLO strictness, quoted verbatim:
//        "We use token budget of 2048 and 512 for all models under the relaxed
//         and strict settings, respectively, except for the LLaMA2-70B relaxed
//         configuration where we use token budget of 1536 to reduce the impact
//         of pipeline bubbles."
//      → 512 strict / 2048 relaxed are the source-verified anchors; 1536 is
//      the PP-tuned variant. Strict band is triggered by tbt_ms ≤ 30ms (tight
//      SLOs can't afford fat forward passes; bigger token budgets slow every
//      decode step).
//
// Algorithm:
//   - Compute `hint = α × maxBatch × isl` (DJL band lower edge).
//   - Pick the smallest candidate from {512, 1024, 1536, 2048, 4096, 8192}
//     that is ≥ max(1024, hint). Candidates are powers-of-2 (kernel tile
//     alignment) with 1536 inserted as the Sarathi 70B-PP variant.
//   - If TBT is strict (≤30ms), cap at 2048 — Sarathi's relaxed-band ceiling.
//
// Not modelled: HBM cost of the token budget (activation memory at the chosen
// budget); chunked-prefill throughput tradeoff (Sarathi's actual optimization).
// This is a "what to put in the vLLM flag as a starting point" answer, not a
// "what's optimal" answer — empirical sweep around the recommendation, then
// pick the goodput knee.
// Sarathi-Serve token-budget bands. SINGLE SOURCE OF TRUTH for the picker
// AND the sweep grid — drift between the two used to mean the picker could
// recommend 1536 or 8192 while the sweep table only showed {512, 1024, 2048,
// 4096}, hiding the actual recommendation from the user's empirical sweep.
//   - 512: strict TBT regime (Sarathi-Serve §5.1).
//   - 1024: default starting point (Sarathi-Serve § passim, also a vLLM
//           kernel-friendly power of 2).
//   - 1536: paper-cited LLaMA2-70B PP variant (Sarathi-Serve §5.1 verbatim:
//           "except for the LLaMA2-70B relaxed configuration where we use
//           token budget of 1536 to reduce the impact of pipeline bubbles").
//   - 2048: relaxed TBT regime cap (Sarathi-Serve §5.1).
//   - 4096 / 8192: large-throughput variants for throughput-only deployments.
export const SARATHI_BANDS = [512, 1024, 1536, 2048, 4096, 8192];
// Strict TBT regime: when TBT_SLO ≤ this, cap the budget at SARATHI_STRICT_CAP.
// "Strict ~512" / "relaxed ~2048" framing from Sarathi-Serve §5.2.
export const SARATHI_STRICT_TBT_MS = 30;
export const SARATHI_STRICT_CAP = 2048;

export function recommendMaxBatchedTokens({ maxBatch, isl, tbt_ms }) {
  const hint = ALPHA * Math.max(1, maxBatch) * Math.max(1, isl);
  const sarathiBand = tbt_ms <= SARATHI_STRICT_TBT_MS ? "strict (~512)" : "relaxed (~2048)";
  let pick = SARATHI_BANDS.find((c) => c >= Math.max(1024, hint)) ?? SARATHI_BANDS[SARATHI_BANDS.length - 1];
  if (tbt_ms <= SARATHI_STRICT_TBT_MS && pick > SARATHI_STRICT_CAP) pick = SARATHI_STRICT_CAP;
  return { value: pick, band: sarathiBand };
}

// ---------- P:D ratio (disaggregation hint) ----------
// Prefill:Decode capacity ratio for sizing a disaggregated deployment
// (reference §5.2). MFU asymmetry amplifies the raw ISL/OSL ratio: prefill
// achieves much higher tensor-core utilization than decode does.
export function pdRatio({ isl, osl }) {
  if (osl <= 0) return null;
  return (isl / osl) * (PREFILL_MFU / DECODE_MFU);
}

// ---------- sweep ranges (analytical → empirical hand-off) ----------
// Turn the analytical batch bound into the experimental sweep grid. Brackets
// the bound so the empirical sweep can find the actual knee on either side.
function powersOf2In(minV, maxV) {
  const out = [];
  let p = 1;
  while (p < minV) p *= 2;
  while (p <= maxV) { out.push(p); p *= 2; }
  return out;
}

// Build the empirical sweep grid that hands the analytical recommendation
// off to a benchmark runner. The whole point of the calculator (reference §0.5
// "two-stage methodology"): use analytical bounds to rule out ~80% of the
// search space, then sweep a tight band around the recommendation instead of
// a wide blind grid. Output is the three vLLM-shaped knobs a runner needs.
//
//   max_num_seqs:           powers of 2 in [b/4, 2b]. The vLLM "concurrent
//                           request cap." Powers of 2 because vLLM's
//                           scheduler prefers them and Sarathi-Serve §4
//                           recommends the family. Brackets the analytical
//                           batch by ½ decade either side so a sweep can
//                           empirically locate the knee.
//   max_num_batched_tokens: Sarathi-Serve verified bands (512 / 1024 / 2048 /
//                           4096) truncated to ≤ 2× the recommended value.
//                           Token budget per step — controls how much prefill
//                           can backfill alongside decode. Smaller protects
//                           TBT; larger raises prefill throughput.
//   concurrency:            request-level load. Fractional steps around b
//                           ([1, b/4, b/2, b, 1.5b, 2b]) — deliberately NOT
//                           powers of 2 because the goal is to land exactly
//                           at sub-saturation (b/4, b/2: confirms the system
//                           doesn't saturate prematurely), at the analytical
//                           knee (b: confirms it matches reality), and at
//                           super-saturation (1.5b, 2b: confirms SLO breaks
//                           where the math says it should).
//
// Caller (compute()) feeds in recommendedBatch = min(b_slo, b_kv) and
// recommendedMnbt from recommendMaxBatchedTokens(). The UI's "patchbay"
// renders the three grids as chip rows for the user to copy into their
// benchmark harness.
export function sweepRanges({ recommendedBatch, recommendedMnbt, tbt_ms }) {
  const b = Math.max(1, Math.floor(recommendedBatch));
  const seqs = powersOf2In(Math.max(1, Math.floor(b / 4)), Math.max(2, b * 2));
  // Sweep candidates come from the same SARATHI_BANDS the picker uses, so the
  // sweep table will always include the recommended value (including 1536 for
  // PP and 8192 for throughput-only). Apply the same strict-TBT cap so we
  // don't suggest the user empirically test 4096 when the picker already ruled
  // it out for strict TBT.
  const strictCap = tbt_ms != null && tbt_ms <= SARATHI_STRICT_TBT_MS ? SARATHI_STRICT_CAP : Infinity;
  const upper = Math.min(recommendedMnbt * 2, strictCap);
  const tokens = SARATHI_BANDS.filter((v) => v <= upper);
  const concurrency = [1, Math.max(1, Math.floor(b / 4)), Math.max(1, Math.floor(b / 2)), b, Math.floor(b * 1.5), b * 2]
    .filter((v, i, a) => a.indexOf(v) === i)
    .sort((x, y) => x - y);
  return { max_num_seqs: seqs, max_num_batched_tokens: tokens, concurrency };
}

// ---------- top-level orchestrator ----------

// Single entry point for the UI. Takes raw form input, returns everything the
// UI needs: metrics, chart curve points, sweep grids, and a structured warnings
// list. Kept top-down readable; each delegate is a pure function above.
export function compute(input) {
  const warnings = [];
  const { hw, model, isl, osl, weight_prec, kv_prec, act_prec, tbt_ms, ttft_ms, ngpus } = input;
  // Cost defaults to the hardware row's per-GPU on-demand list price; the UI
  // override (price-per-hour input) takes precedence by passing it in `input`.
  // null/undefined are both allowed and propagate to a "—" tile + gapped cost
  // curve in the chart (see costPerMtok's null guard).
  const price_per_hour_usd = input.price_per_hour_usd !== undefined
    ? input.price_per_hour_usd
    : (hw.price_per_hour_usd ?? null);

  // Input sanitation. Clamp to safe minima; never propagate NaN/Infinity.
  const cleanIsl = Math.max(1, Math.floor(isl || 0));
  const cleanOsl = Math.max(1, Math.floor(osl || 0));
  const cleanNgpus = Math.max(1, Math.floor(ngpus || 0));
  const cleanTbt = Math.max(1, tbt_ms || 0);

  if (cleanIsl + cleanOsl > model.max_context) {
    warnings.push({ level: "warn", msg: `ISL+OSL (${cleanIsl + cleanOsl}) exceeds model max_context (${model.max_context}).` });
  }
  if (model.attn_type === "MLA") {
    warnings.push({ level: "info", msg: `${model.label} uses MLA: KV modelled as (d_c + d_rope) × bytes × n_layers per token (reference §3).` });
  }
  // MoE info note (independent of attention type): announce the sparsity factor.
  const isMoE = model.params_b_total > model.params_b_active;
  if (isMoE) {
    warnings.push({
      level: "info",
      msg: `${model.label} is MoE: ${model.params_b_total}B total weights stream from HBM, ${model.params_b_active}B execute per token. B_crit shifted by ${(model.params_b_total/model.params_b_active).toFixed(1)}× vs a dense ${model.params_b_active}B model.`,
    });
  }
  if (weight_prec === "FP8" && !hw.fp8_tflops) {
    warnings.push({ level: "warn", msg: `${hw.label} has no hardware FP8 path; using FP16 roofline for compute (FP8 byte savings still apply to weights/KV memory).` });
  }

  const wBytes = DTYPE_BYTES[weight_prec];
  const kvBytes = DTYPE_BYTES[kv_prec];

  const W_total           = weightsBytes(model.params_b_total, wBytes);
  const paramCount_active = model.params_b_active * 1e9;
  const K = kvPerToken(model, kvBytes);
  const totalBW = hw.hbm_bw_gbs * 1e9 * cleanNgpus;
  // Match bCrit: FP8 tensor cores only fire on FP8×FP8 GEMM on FP8-capable HW.
  const useFp8Compute = weight_prec === "FP8" && act_prec === "FP8" && hw.fp8_tflops;
  const flopsPerGpu = (useFp8Compute ? hw.fp8_tflops : hw.fp16_tflops) * 1e12;
  const totalFLOPs = flopsPerGpu * cleanNgpus;

  // bCrit only depends on a single GPU's roofline ratio (it's a per-chip
  // arithmetic-intensity threshold; aggregating ngpus would cancel out).
  const bcrit = bCrit(hw, model, weight_prec, act_prec);

  // Sequence-occupancy: peak (ISL+OSL — end-of-decode) is used for BOTH bounds.
  // bKv: peak is the OOM risk (must fit the worst step). bSlo / stepTime: peak
  // is the SLO risk (the worst step is the slowest; "TBT p99 < X" demands every
  // step meet bound). Using mean for stepTime would fit batches that meet TBT
  // on average but blow SLO at end-of-decode — wrong answer for a sizing tool.
  const peakSeq = cleanIsl + cleanOsl;

  const bslo = bSlo({ tbt_ms: cleanTbt, kv_per_token: K, seq: peakSeq, weight_bytes_total: W_total, paramCount_active, totalBW, totalFLOPs });
  const bkv = bKv({ hw, model, ngpus: cleanNgpus, weight_bytes_total: W_total, kv_bytes: kvBytes, isl: cleanIsl, osl: cleanOsl });

  if (bslo.unreachable) {
    warnings.push({ level: "error", msg: `b_slo unreachable: TBT=${cleanTbt}ms × BW < weight stream time. Raise TP or relax TBT.` });
  }
  if (bkv.weights_overflow) {
    warnings.push({ level: "error", msg: `Weights overflow available HBM (${(W_total / 1e9).toFixed(1)} GB > ${hw.hbm_gb * cleanNgpus} GB). Increase ngpus.` });
  }

  // ---- recommended operating batch ----
  //
  // The three Bs are NOT the same kind of bound:
  //   b_kv:  hard CEILING — exceed → OOM.
  //   b_slo: hard CEILING — exceed → TBT SLO violated.
  //   b_crit: NOT a ceiling. It's the roofline ridge — the batch above which
  //          the MLP kernel flips from memory-bound (FLOPs idle) to
  //          compute-bound (latency grows linearly with B). It's a *regime
  //          marker / target floor*, not a constraint: nothing breaks if you
  //          run above it, you just pay latency for throughput; nothing breaks
  //          if you run below it, you just leave FLOPs on the floor.
  //
  // Recommended operating point: push to the right edge of the feasible range
  // [b_crit, min(b_slo, b_kv)] — max out throughput within the ceilings,
  // even if that lands past b_crit (the rational compute-bound trade-off).
  // Hence min(b_slo, b_kv), NOT min(b_crit, b_slo, b_kv): including b_crit
  // would deliberately under-batch and waste throughput honoring a constraint
  // that doesn't exist. The b_crit vs recommended comparison is then surfaced
  // as a regime diagnostic below (memory-bound vs compute-bound).
  //
  // How parallelism enters the numbers above:
  //   b_crit: per-chip roofline ratio (FLOPs/BW × bits_w/bits_act × sparsity).
  //          TP/PP do NOT change it — sharding splits weights and compute
  //          proportionally, so the per-chip ratio is invariant. That's why
  //          bCrit() takes no ngpus.
  //   b_kv:  scales ~linearly with ngpus. bKv() uses (hw.hbm_gb × ngpus) for
  //          aggregate HBM; weights still occupy W_total (sharded but the sum
  //          is the same), so usable HBM for KV grows almost linearly.
  //   b_slo: scales ~linearly with ngpus via totalBW = hbm_bw × ngpus and
  //          totalFLOPs = flops × ngpus — both branches of the closed form
  //          (weight-load branch and compute branch) grow with ngpus.
  // Net effect: adding GPUs lifts both ceilings while b_crit stays put — so
  // more hardware moves you rightward through the regimes (memory-bound →
  // ridge → compute-bound). What we DO NOT model: the comm cost of crossing
  // the useful-TP ceiling (Y_max). Beyond Y_max the all-reduce floor inflates
  // step time and the linear-in-ngpus assumption breaks down. We surface that
  // as a diagnostic via the parallelism block below, but don't dock b_slo for
  // it — sizing remains optimistic past Y_max.
  //
  // Fall back to 1 if both bounds are zero/unreachable so the sweep grids
  // still render something sane and we never propagate Infinity.
  const bsloEff = bslo.value > 0 ? bslo.value : Infinity;
  const bkvEff = bkv.value > 0 ? bkv.value : Infinity;
  const rawBatch = Math.min(bsloEff, bkvEff);
  const recommendedBatch = Number.isFinite(rawBatch) ? Math.max(1, rawBatch) : 1;

  // Regime diagnostic: where does the recommended batch sit relative to the
  // roofline ridge? See the long comment above for why b_crit isn't a
  // constraint — it's a regime marker, and that's what we report here.
  if (bcrit > 0 && Number.isFinite(bcrit)) {
    const binder = bsloEff <= bkvEff ? "TBT SLO" : "KV cache";
    if (recommendedBatch < bcrit * 0.8) {
      warnings.push({
        level: "info",
        msg: `Memory-bound regime: recommended batch (${Math.round(recommendedBatch)}) is below B_crit (${Math.round(bcrit)}). ` +
             `FLOPs are idle — ${binder} is the binder. More HBM or relaxed TBT would unlock throughput; a faster GPU at same BW would NOT help.`,
      });
    } else if (recommendedBatch > bcrit * 1.2) {
      warnings.push({
        level: "info",
        msg: `Compute-bound regime: recommended batch (${Math.round(recommendedBatch)}) is past B_crit (${Math.round(bcrit)}). ` +
             `Each additional sequence trades linear latency for throughput — the rational regime when ${binder} headroom allows it.`,
      });
    } else {
      warnings.push({
        level: "info",
        msg: `Operating near the roofline ridge: recommended batch (${Math.round(recommendedBatch)}) ≈ B_crit (${Math.round(bcrit)}). ` +
             `Throughput-per-latency sweet spot — neither FLOPs nor BW is significantly underutilized.`,
      });
    }
  }

  // Parallelism depends on batch (for Y_max), so compute it after recommendedBatch.
  const parallelism = recommendParallelism({
    hw, weight_bytes_total: W_total, ngpus: cleanNgpus,
    d_ff: model.d_ff, batch: recommendedBatch,
  });
  if (!parallelism.fits) {
    warnings.push({ level: "error", msg: parallelism.reason });
  }
  if (Number.isFinite(parallelism.y_max) && parallelism.y_max < TP_CAP && cleanNgpus > 1) {
    warnings.push({
      level: "warn",
      msg: `ICI binds TP usefulness at Y_max=${parallelism.y_max.toFixed(1)} chips ` +
           `(β=${(hw.hbm_bw_gbs / hw.ici_bw_gbs).toFixed(1)}, B=${recommendedBatch}, d_ff=${model.d_ff}). ` +
           `TP beyond this stops cutting decode latency.`,
    });
  }
  const mnbt = recommendMaxBatchedTokens({ maxBatch: recommendedBatch, isl: cleanIsl, tbt_ms: cleanTbt });
  const pd = pdRatio({ isl: cleanIsl, osl: cleanOsl });
  const sweep = sweepRanges({ recommendedBatch, recommendedMnbt: mnbt.value, tbt_ms: cleanTbt });

  // Chart curve: log-spaced B from 1 to 4× max(bcrit, bkv). UI draws stepTime
  // and throughput from this; sized to comfortably show the ridge and the
  // KV-cliff regions both.
  const curveMax = Math.max(bcrit * 4, (bkv.value || 1) * 2, 64);
  const curve = [];
  for (let B = 1; B <= curveMax; B = Math.max(B + 1, Math.ceil(B * 1.2))) {
    const t_ms = stepTime({ B, kv_per_token: K, seq: peakSeq, weight_bytes_total: W_total, paramCount_active, totalBW, totalFLOPs }) * 1000;
    const tps = (B / (t_ms / 1000));
    // Cost overlay on throughput — unit conversion only (see costPerMtok).
    // Propagates null when price is unset so the chart can render gaps cleanly.
    const cost = costPerMtok({ price_per_hour_usd, ngpus: cleanNgpus, tps });
    curve.push({ B, step_ms: t_ms, tokens_per_sec: tps, cost_per_mtok: cost });
  }

  // Throughput at the recommended batch — headline tile number.
  const throughput = recommendedBatch > 0 && Number.isFinite(recommendedBatch)
    ? throughputAtB({ B: recommendedBatch, kv_per_token: K, seq: peakSeq, weight_bytes_total: W_total, paramCount_active, totalBW, totalFLOPs })
    : 0;
  // $/Mtok at the recommended batch — headline tile number. null if price
  // unset; the UI renders that as "—".
  const cost_per_mtok = throughput > 0
    ? costPerMtok({ price_per_hour_usd, ngpus: cleanNgpus, tps: throughput })
    : null;

  return {
    metrics: {
      b_crit: Math.round(bcrit),
      b_slo: bslo.value,
      b_kv: bkv.value,
      recommended_batch: recommendedBatch,
      max_num_seqs: recommendedBatch,
      max_num_batched_tokens: mnbt.value,
      max_num_batched_tokens_band: mnbt.band,
      throughput_tps: throughput,
      throughput_tps_per_gpu: throughput / cleanNgpus,
      cost_per_mtok,
      price_per_hour_usd,
      parallelism,
      pd_ratio: pd,
      weights_gb: W_total / 1e9,
      kv_per_token_kb: K / 1024,
      kv_per_seq_gb: (K * peakSeq) / 1e9,
    },
    chart: { curve, bcrit, bslo: bslo.value, bkv: bkv.value, tbt_ms: cleanTbt },
    sweep,
    warnings,
  };
}
