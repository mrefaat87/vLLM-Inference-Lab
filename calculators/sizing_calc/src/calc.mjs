// Pure-function compute layer for the LLM inference sizing calculator.
//
// No DOM access, no globals, no I/O — every function takes plain numbers/objects
// and returns plain numbers/objects. This is what makes the formulas testable in
// node:test without a browser. Every formula cites a section in
// MODEL_SIZING_SCALING_REFERENCE.md so reviewers can trace numbers to source.

// ---------- constants ----------
// Reserve ~10% of HBM for activations + workspace + paging fragmentation. The
// reference doesn't pin this; it's a vLLM-style headroom (gpu_memory_utilization
// defaults to 0.90). Tunable from the UI later if needed.
export const ACT_OVERHEAD = 0.10;
// Within-node TP is hard-capped by NVLink fan-out; beyond 8 the ICI cost
// dominates (reference §6, β ≈ 8).
export const TP_CAP = 8;
// HBM-to-ICI bandwidth ratio used by reference §6's Y_max formula. Surfaced as
// an exported constant so tests can assert specific roofline-vs-comm scenarios.
export const BETA = 8;
// max_num_batched_tokens hint band α from DJL TRT-LLM guidance: 0.1–0.2 × B × ISL.
// We use the midpoint as the centerpoint; the sweep widens around it.
export const ALPHA = 0.15;
// MFU defaults from reference §0.3 — prefill saturates more of the compute than
// decode does, so the prefill:decode ratio is amplified beyond just ISL/OSL.
export const PREFILL_MFU = 0.5;
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
// matters for the roofline since BW also aggregates). params_b is in billions,
// matching the data table; multiply by 1e9 here once so callers stay in "B".
export function weightsBytes(params_b, weight_bytes) {
  return params_b * 1e9 * weight_bytes;
}

// KV cache bytes per generated token, per sequence (reference §3).
// Factor of 2 covers the K and V tensors. n_kv_heads (< n_heads for GQA, =1 for
// MQA) is the dimension that GQA actually compresses, which is why GQA's KV is
// so much smaller than MHA. MLA breaks this formula — caller must flag MLA
// models because the real KV is ~10–20× smaller than this returns.
export function kvPerToken(model, kv_bytes) {
  return 2 * kv_bytes * model.n_kv_heads * model.head_dim * model.n_layers;
}

// Roofline ridge-point batch (reference §1). Below this, weights dominate and
// adding batch is free; above, compute dominates and latency grows linearly.
// Hardware FP8 tensor cores fire only when BOTH operands are FP8 (cuBLAS GEMM
// rule) AND the hardware exposes an FP8 path. INT8 weights with FP16 acts use
// the FP16 compute path with the weight/act ratio capturing the asymmetry.
export function bCrit(hw, weight_prec, act_prec) {
  const weight_bytes = DTYPE_BYTES[weight_prec];
  const act_bytes = DTYPE_BYTES[act_prec];
  const useFp8 = weight_prec === "FP8" && act_prec === "FP8" && hw.fp8_tflops;
  const flops = (useFp8 ? hw.fp8_tflops : hw.fp16_tflops) * 1e12;
  const bw_bytes_per_sec = hw.hbm_bw_gbs * 1e9;
  // bits_per_param / bits_per_activation: equivalent to weight_bytes/act_bytes
  // since both are ×8. Skip the round-trip to avoid float noise.
  return (flops / bw_bytes_per_sec) * (weight_bytes / act_bytes);
}

// Time per decode step (seconds) as the max of memory and compute bound
// (reference §4 general form). The two terms compete; max picks the bottleneck.
// avgSeq is mean tokens-of-KV per active sequence during the step we're modelling.
export function stepTime({ B, kv_per_token, avgSeq, weight_bytes_total, paramCount, totalBW, totalFLOPs }) {
  // Memory: every step we stream the whole weight tensor once + the full KV of
  // every active sequence. KV scales linearly with B and avgSeq.
  const memTime = (weight_bytes_total + B * kv_per_token * avgSeq) / totalBW;
  // Compute: B tokens produced per step, 2·N flops per token (reference §0.3).
  const computeTime = (2 * B * paramCount) / totalFLOPs;
  return Math.max(memTime, computeTime);
}

// Throughput at a given batch (tokens/sec). Derived from stepTime since
// stepTime accounts for both bottlenecks; the reference §4 closed form is only
// the memory-bound branch.
export function throughputAtB(args) {
  const t = stepTime(args);
  return args.B / t;
}

// ---------- batch-size bounds ----------

// Largest B that keeps a decode step under TBT_SLO. Solves both branches of the
// stepTime max and returns the binding minimum. Returns 0 + an unreachable flag
// if weights alone blow the budget — caller can surface "raise TP / relax TBT".
export function bSlo({ tbt_ms, kv_per_token, avgSeq, weight_bytes_total, paramCount, totalBW, totalFLOPs }) {
  const T = tbt_ms / 1000;
  // Memory branch: (W + B·K·S)/BW ≤ T  →  B ≤ (T·BW − W)/(K·S)
  const numerator = T * totalBW - weight_bytes_total;
  if (numerator <= 0) {
    return { value: 0, unreachable: true, reason: "weights stream time alone exceeds TBT budget" };
  }
  const memBound = numerator / (kv_per_token * avgSeq);
  // Compute branch: 2·B·N/FLOPs ≤ T  →  B ≤ T·FLOPs/(2·N)
  const computeBound = (T * totalFLOPs) / (2 * paramCount);
  const value = Math.floor(Math.min(memBound, computeBound));
  return { value: Math.max(0, value), unreachable: false, binding: memBound < computeBound ? "memory" : "compute" };
}

// Largest B that fits the KV cache in HBM after weights + activation headroom
// (reference §3, §5). Uses peak occupancy ISL+OSL — end-of-decode is when KV
// is largest, so this is the conservative bound that won't OOM mid-generation.
export function bKv({ hw, model, ngpus, weight_bytes_total, kv_per_token, isl, osl }) {
  const totalHBM = hw.hbm_gb * ngpus * 1e9;
  const usable = totalHBM * (1 - ACT_OVERHEAD) - weight_bytes_total;
  if (usable <= 0) {
    return { value: 0, weights_overflow: true };
  }
  const peakSeq = isl + osl;
  if (peakSeq <= 0 || kv_per_token <= 0) return { value: 0 };
  return { value: Math.floor(usable / (kv_per_token * peakSeq)), weights_overflow: false };
}

// ---------- parallelism ----------

// Recommend TP × PP × replicas. Strategy: smallest TP (≤8, dividing ngpus)
// such that weight slice + 10% headroom fits per GPU; if even TP=8 can't fit
// the weights, escalate to PP. ngpus is total GPUs across the deployment;
// replicas = how many independent serving units fit.
export function recommendParallelism({ hw, weight_bytes_total, ngpus }) {
  const hbm_per_gpu = hw.hbm_gb * 1e9 * (1 - ACT_OVERHEAD);
  const tpCandidates = [1, 2, 4, 8].filter((t) => t <= ngpus);
  let tp = null;
  for (const t of tpCandidates) {
    if (weight_bytes_total / t <= hbm_per_gpu) { tp = t; break; }
  }
  if (tp !== null) {
    // PP not needed for weight fit; use remaining GPUs as additional replicas.
    const replicas = Math.floor(ngpus / tp);
    const reason = tp === 1
      ? "weights fit on a single GPU; TP=1 maximizes replica count"
      : `weights need TP=${tp} to fit per-GPU HBM with ${Math.round(ACT_OVERHEAD * 100)}% headroom`;
    return { tp, pp: 1, replicas, fits: true, reason };
  }
  // No TP value in {1,2,4,8} fits — need pipeline parallelism. Use TP=8 per
  // stage (NVLink ceiling) and split layers across PP stages.
  const tpUsed = Math.min(8, ngpus);
  const pp = Math.ceil(weight_bytes_total / (tpUsed * hbm_per_gpu));
  const replicaGpus = tpUsed * pp;
  if (replicaGpus > ngpus) {
    return {
      tp: tpUsed, pp, replicas: 0, fits: false,
      reason: `needs TP=${tpUsed}×PP=${pp}=${replicaGpus} GPUs per replica; only ${ngpus} available`,
    };
  }
  return {
    tp: tpUsed, pp, replicas: Math.floor(ngpus / replicaGpus), fits: true,
    reason: `weights too large for TP-only; need PP=${pp} stages (TP=${tpUsed} per stage)`,
  };
}

// ---------- max_num_batched_tokens ----------

// vLLM's max_num_batched_tokens. Combines DJL's α × batch × ISL hint with
// Sarathi-Serve's strict/relaxed bands (reference §10). Snaps to a power of 2
// ≥ 1024 so it lines up with kernel tile sizes.
export function recommendMaxBatchedTokens({ maxBatch, isl, tbt_ms }) {
  const hint = ALPHA * Math.max(1, maxBatch) * Math.max(1, isl);
  const sarathiBand = tbt_ms <= 30 ? "strict (~512)" : "relaxed (~2048)";
  const candidates = [512, 1024, 1536, 2048, 4096, 8192];
  // Prefer the smallest candidate ≥ max(1024, hint) — Sarathi powers-of-2 rule.
  let pick = candidates.find((c) => c >= Math.max(1024, hint)) ?? 8192;
  // If the strict TBT band is in play, never exceed 2048.
  if (tbt_ms <= 30 && pick > 2048) pick = 2048;
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

export function sweepRanges({ recommendedBatch, recommendedMnbt }) {
  const b = Math.max(1, Math.floor(recommendedBatch));
  const seqs = powersOf2In(Math.max(1, Math.floor(b / 4)), Math.max(2, b * 2));
  const tokens = [512, 1024, 2048, 4096].filter((v) => v <= recommendedMnbt * 2);
  // Concurrency uses fractional steps around b to capture sub-/super-saturation.
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

  // Input sanitation. Clamp to safe minima; never propagate NaN/Infinity.
  const cleanIsl = Math.max(1, Math.floor(isl || 0));
  const cleanOsl = Math.max(1, Math.floor(osl || 0));
  const cleanNgpus = Math.max(1, Math.floor(ngpus || 0));
  const cleanTbt = Math.max(1, tbt_ms || 0);

  if (cleanIsl + cleanOsl > model.max_context) {
    warnings.push({ level: "warn", msg: `ISL+OSL (${cleanIsl + cleanOsl}) exceeds model max_context (${model.max_context}).` });
  }
  if (model.attn_type === "MLA") {
    warnings.push({ level: "warn", msg: `${model.label} uses MLA; b_kv is a GQA-equivalent upper bound (real MLA cache is ~10–20× smaller).` });
  }
  if (weight_prec === "FP8" && !hw.fp8_tflops) {
    warnings.push({ level: "warn", msg: `${hw.label} has no hardware FP8 path; using FP16 roofline for compute (FP8 byte savings still apply to weights/KV memory).` });
  }

  const wBytes = DTYPE_BYTES[weight_prec];
  const kvBytes = DTYPE_BYTES[kv_prec];

  const W = weightsBytes(model.params_b, wBytes);
  const paramCount = model.params_b * 1e9;
  const K = kvPerToken(model, kvBytes);
  const totalBW = hw.hbm_bw_gbs * 1e9 * cleanNgpus;
  // Match bCrit: FP8 tensor cores only fire on FP8×FP8 GEMM on FP8-capable HW.
  const useFp8Compute = weight_prec === "FP8" && act_prec === "FP8" && hw.fp8_tflops;
  const flopsPerGpu = (useFp8Compute ? hw.fp8_tflops : hw.fp16_tflops) * 1e12;
  const totalFLOPs = flopsPerGpu * cleanNgpus;

  // bCrit only depends on a single GPU's roofline ratio (it's a per-chip
  // arithmetic-intensity threshold; aggregating ngpus would cancel out).
  const bcrit = bCrit(hw, weight_prec, act_prec);

  // Sequence-occupancy: ISL+OSL for KV (peak end-of-decode is the OOM risk);
  // ISL+OSL/2 for stepTime (mean during decode, what TBT actually averages).
  const peakSeq = cleanIsl + cleanOsl;
  const avgSeq = cleanIsl + cleanOsl / 2;

  const bslo = bSlo({ tbt_ms: cleanTbt, kv_per_token: K, avgSeq, weight_bytes_total: W, paramCount, totalBW, totalFLOPs });
  const bkv = bKv({ hw, model, ngpus: cleanNgpus, weight_bytes_total: W, kv_per_token: K, isl: cleanIsl, osl: cleanOsl });

  if (bslo.unreachable) {
    warnings.push({ level: "error", msg: `b_slo unreachable: TBT=${cleanTbt}ms × BW < weight stream time. Raise TP or relax TBT.` });
  }
  if (bkv.weights_overflow) {
    warnings.push({ level: "error", msg: `Weights overflow available HBM (${(W / 1e9).toFixed(1)} GB > ${hw.hbm_gb * cleanNgpus} GB). Increase ngpus.` });
  }

  const parallelism = recommendParallelism({ hw, weight_bytes_total: W, ngpus: cleanNgpus });
  if (!parallelism.fits) {
    warnings.push({ level: "error", msg: parallelism.reason });
  }

  // The "recommended batch" = the binding analytical bound. max_num_seqs in
  // vLLM should be set to this so empirical sweeps start at the right scale.
  // If both bounds are zero/unreachable, fall back to 1 (we still want the
  // sweep table to render something sane and never propagate Infinity).
  const bsloEff = bslo.value > 0 ? bslo.value : Infinity;
  const bkvEff = bkv.value > 0 ? bkv.value : Infinity;
  const rawBatch = Math.min(bsloEff, bkvEff);
  const recommendedBatch = Number.isFinite(rawBatch) ? Math.max(1, rawBatch) : 1;
  const mnbt = recommendMaxBatchedTokens({ maxBatch: recommendedBatch, isl: cleanIsl, tbt_ms: cleanTbt });
  const pd = pdRatio({ isl: cleanIsl, osl: cleanOsl });
  const sweep = sweepRanges({ recommendedBatch, recommendedMnbt: mnbt.value });

  // Chart curve: log-spaced B from 1 to 4× max(bcrit, bkv). UI draws stepTime
  // and throughput from this; sized to comfortably show the ridge and the
  // KV-cliff regions both.
  const curveMax = Math.max(bcrit * 4, (bkv.value || 1) * 2, 64);
  const curve = [];
  for (let B = 1; B <= curveMax; B = Math.max(B + 1, Math.ceil(B * 1.2))) {
    const t_ms = stepTime({ B, kv_per_token: K, avgSeq, weight_bytes_total: W, paramCount, totalBW, totalFLOPs }) * 1000;
    const tps = (B / (t_ms / 1000));
    curve.push({ B, step_ms: t_ms, tokens_per_sec: tps });
  }

  // Throughput at the recommended batch — headline tile number.
  const throughput = recommendedBatch > 0 && Number.isFinite(recommendedBatch)
    ? throughputAtB({ B: recommendedBatch, kv_per_token: K, avgSeq, weight_bytes_total: W, paramCount, totalBW, totalFLOPs })
    : 0;

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
      parallelism,
      pd_ratio: pd,
      weights_gb: W / 1e9,
      kv_per_token_kb: K / 1024,
      kv_per_seq_gb: (K * peakSeq) / 1e9,
    },
    chart: { curve, bcrit, bslo: bslo.value, bkv: bkv.value, tbt_ms: cleanTbt },
    sweep,
    warnings,
  };
}
