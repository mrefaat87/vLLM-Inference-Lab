# EP_max — research, derivation, and implementation plan

**Status:** research complete, not implemented. Pick this up when ready to add expert-parallelism sizing to the calculator.

**Author trail:** consolidated 2026-05-23 from a sub-agent research dossier + the Reiner Pope (Scaling Book co-author) interview on Dwarkesh.

---

## 1. Context

Our calculator already models the **tensor-parallel** usefulness ceiling, `Y_max`, in `src/calc.mjs` (`yMax(d_ff, batch, hw)`):

```
Y_max ≈ d_ff / (B · β_calc)        with β_calc = HBM_BW / NVLink_BW
```

Derivation in `reference/MODEL_SIZING_SCALING_REFERENCE.md` §6. Beyond `Y_max` chips, the intra-NVLink TP all-reduce floor dominates and adding TP stops cutting decode latency.

**We want the analogous formula for expert parallelism (EP).** Differences from TP:

1. **Comm pattern:** all-to-all (dispatch + combine), not all-reduce.
2. **Two-tier fabric:** EP can stay inside the scale-up domain (NVLink within rack) or spill across racks onto the scale-out fabric (InfiniBand / RoCE). Both must be modelled — Pope explicitly frames this as "the rack bounds the expert layer."
3. **Sparsity coupling:** EP usefulness scales with `E_total / k_active` (Pope's "300 × sparsity" rule).

---

## 2. Primary sources

### 2.1 The Scaling Book — inference chapter
URL: https://jax-ml.github.io/scaling-book/inference/

- No explicit `E_max` derivation. Only MoE-relevant content is in Q5: *"`B_crit_MoE = B_crit_dense × (E/k)`"* — the batch-side sparsity multiplier.
- `[EZ, DX, FY]` dimensional notation for expert sharding without a comm-cost derivation.

### 2.2 Reiner Pope on Dwarkesh
URL: https://gist.github.com/dwarkeshsp/79100f0fdeed69d76241903bb0604dbe

Load-bearing quotes:
- **300 × sparsity rule:** *"batch size needs to be bigger than approximately 300 times sparsity. For example, in DeepSeek I activate 32 out of 256 experts, so this would be 8 for DeepSeek."* — Pope's "32 of 256" simplification; tech report says 8 routed + 1 shared of 256 routed + 1 shared. See §9 Caveats.
- **Scale-up vs scale-out:** *"The scale-out... tends to be about 8x slower in bandwidth."* — universal `β_rack ≈ 8`.
- **Rack as EP boundary:** *"The fundamental thing here is that one rack bounds the size of an expert layer you can do."*
- **EP traffic pattern:** *"any GPU will be talking to any other GPU... This is an all-to-all traffic pattern."*

### 2.3 DeepSeek-V3 Technical Report
URL: https://arxiv.org/abs/2412.19437

### 2.4 DeepSeek Open-Infra-Index Day 6 (production system overview)
URL: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md

Production topology:
- **Prefill:** Routed Expert EP32 / shared+MLA DP32 across **4×H800 nodes**; 9 routed experts + 1 shared per GPU.
- **Decode:** Routed Expert **EP144** / DP144 across **18 H800 nodes**; 2 routed + 1 shared per GPU.
- Dual-batch / 5-stage pipeline overlap hides all-to-all behind compute.

### 2.5 DeepEP (DeepSeek's open-source EP comm library)
URL: https://github.com/deepseek-ai/DeepEP

Effective per-GPU bandwidth (SM90/H800, post-kernel-overhead):
- **Intra-node (NVLink):** dispatch ~726 GB/s, combine ~740 GB/s (with 64 SMs tuned).
- **Inter-node (CX7 RDMA/IB):** dispatch ~61–90 GB/s, combine ~61–81 GB/s depending on topology (EP 8×2 vs EP 8×4).

### 2.6 SGLang large-scale EP blog
URL: https://www.lmsys.org/blog/2025-05-05-large-scale-ep/

Reports `EP72` on 9×8×H100 — note: this is the **SGLang OSS replication**, not DeepSeek prod. Per-layer dispatch ~0.17 ms with DeepEP.

### 2.7 NVIDIA "Wide EP on NVL72"
URL: https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/

NVL72 = 130 TB/s aggregate NVLink5 domain → ~1.8 TB/s per-GPU bidir over 72 GPUs.

### 2.8 Meta Engineering — scaling LLM inference (Oct 2025)
URL: https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/

Anchor quote: *"The all-to-all communication can contribute 10-30% to end-to-end latency, especially for decode messages (100KB to 2MB)."*

---

## 3. Derivation

### 3.1 Setup

One MoE layer with `E_total` experts, `k` activated per token. Decode step has `B` tokens per replica. Hidden width `D` (residual). EP degree `E_p` = number of GPUs sharding the experts (assume 1 expert per GPU for clarity; multi-expert-per-GPU just rescales `E_p`).

### 3.2 The MoE block does two all-to-all comms per layer
- **Dispatch** — route tokens to the GPUs owning their `k` activated experts.
- **Combine** — gather expert outputs back and weighted-sum.

Both move tensors shaped `[tokens × D]` in `bytes_act` precision.

### 3.3 Per-GPU bytes moved by one dispatch (uniform routing, large-`E_p` limit)

A given GPU originates `B` tokens. Each token picks `k` experts uniformly across `E_total`. With `E_p` GPUs and `E_total/E_p` experts per GPU, the probability any specific destination GPU is hit by a given token is `≈ k / E_p`. Across `B` tokens, expected tokens sent from this GPU to one specific peer ≈ `B · k / E_p`. The GPU sends to `~E_p − 1` peers:

```
egress_dispatch ≈ B · k · D · bytes_act     [bytes / layer / GPU]
```

`E_p` cancels — total egress is "every locally-originated token sent to its `k` experts." Symmetric ingress from peers. **Combine has the same shape**: each expert returns its `B · k / E_p` per-source weighted-sum, summed across `E_p` GPUs of incoming combine = `B · k · D · bytes_act` ingress.

### 3.4 Total comm bytes per GPU per MoE layer
(dispatch + combine, egress + ingress)

```
Bytes_a2a = 4 · B · k · D · bytes_act     [bytes / layer / GPU]
```

The factor 4 = `(dispatch egress) + (dispatch ingress) + (combine egress) + (combine ingress)`. Matches DeepEP's separate dispatch/combine kernel structure.

### 3.5 Per-layer all-to-all time, two-tier fabric

With `S` = scale-up domain size (8 for HGX, 72 for NVL72), and `E_p > S`, fraction of traffic that crosses scale-up boundary = `(E_p − S) / E_p`. The slow link dominates wherever traffic crosses.

- **All in one rack** (`E_p ≤ S`):
  `T_a2a ≈ 4·B·k·D·bytes_act / W_nvlink_eff`
- **Crosses rack** (`E_p > S`):
  `T_a2a ≈ 4·B·k·D·bytes_act · ((E_p−S)/E_p) / W_scaleout + 4·B·k·D·bytes_act · (S/E_p) / W_nvlink_eff`

For `E_p ≫ S`, the cross-rack term dominates → `T_a2a ≈ 4·B·k·D·bytes_act / W_scaleout`.

### 3.6 Per-layer weight-load time on the expert weights

Each GPU loads only the experts it owns. With `E_total/E_p` experts per GPU:

```
T_HBM_expert ≈ W_expert_bytes / W_hbm
             = (N_total · bytes_w / E_total) / (E_p · W_hbm)         (×1/E_p once you account for sharding)
             = N_total · bytes_w / (E_total · E_p · W_hbm)
```

Scales as `1/E_p`, exactly like TP's weight-load term.

### 3.7 Crossover (the EP_max condition)

EP keeps cutting decode step time only while weight-streaming is the floor, i.e. `T_HBM_expert > T_a2a`:

```
N_total · bytes_w                               4 · B · k · D · bytes_act
─────────────────────────────────────────  >    ─────────────────────────
   E_total · E_p · W_hbm                              W_fabric
```

Solving for `E_p`:

```
              N_total · bytes_w · W_fabric
E_p  <   ─────────────────────────────────────────────────
         4 · E_total · k · B · D · bytes_act · W_hbm
```

Let `N_per_expert = N_total / E_total` (params per expert in bytes). Then:

```
              N_per_expert · bytes_w           1
E_max  ≈   ─────────────────────────────  ·  ─────────
            4 · k · B · D · bytes_act       β_fabric
```

with **`β_fabric = W_hbm / W_fabric`** — analogous to TP's β, using whichever fabric binds:
- `β_nvlink` ≈ 3.7 on H100 (same as Y_max's β_calc).
- `β_scaleout` ≈ 30 on H100 (HBM 3,350 vs IB ~100 GB/s).

### 3.8 Dimensional check

- Numerator: `[params/expert] · [bytes/param] = bytes`.
- Denominator: `[1] · [1] · [tokens] · [hidden_dim] · [bytes/activation] = bytes`.
- Ratio: dimensionless ✓.
- `β_fabric`: dimensionless (BW/BW) ✓.

---

## 4. Closed-form formulas

### 4.1 In-scale-up (`E_p ≤ S`, NVLink-only)

```
              N_total · bytes_w
E_max_nvlink ≈ ─────────────────────────────────────────────────
              4 · E_total · k · B · D · bytes_act · β_nvlink
```

### 4.2 Crossing the scale-up boundary (`E_p > S`, IB-bound)

```
              N_total · bytes_w
E_max_scaleout ≈ ─────────────────────────────────────────────────
                 4 · E_total · k · B · D · bytes_act · β_scaleout
```

Same shape, different β. **Crossing the rack boundary shrinks E_max by `β_scaleout / β_nvlink ≈ 8×`** — that's the chip-side statement of "the rack bounds the expert layer."

### 4.3 Cleaner per-expert form

```
                N_per_expert · bytes_w
E_max  ≈   ──────────────────────────────────────
           4 · k · B · D · bytes_act · β_fabric
```

### 4.4 Recommendation rule

Keep `E_p ≤ S` whenever the model fits. Only cross when memory forces it (e.g., DeepSeek-V3 671B FP8 = 671 GB > 8 × 80 GB H100 = 640 GB).

---

## 5. Worked examples

DeepSeek-V3 defaults: `N_total = 671B`, `N_active = 37B`, `E_total = 256`, `k = 8` (DeepSeek paper: 8 routed of 256 + 1 shared), `D = 7168`, FP8 weights/acts → `bytes_w = bytes_act = 1`.

`N_per_expert ≈ 671B / 256 ≈ 2.62 GB` (FP8 weights).
Denominator coefficient `4 · k · D · bytes_act = 4 · 8 · 7168 · 1 ≈ 229 KB per token`.

### 5.1 Case 1: 1× NVL72 rack (Blackwell), B = 2,000

- NVL72: aggregate 130 TB/s ÷ 72 ≈ 1.8 TB/s per-GPU bidir NVLink5.
- HBM3e on B200 ≈ 8 TB/s → `β_nvlink ≈ 8000/1800 ≈ 4.4`.
- All 72 GPUs are scale-up → `β_fabric = β_nvlink`.

```
E_max ≈ 2.62e9 / (229e3 · 2000 · 4.4) ≈ 1.3
```

**Interpretation:** at B=2,000, EP_max ≈ 1 — the all-to-all has already eaten the weight-load gain. The formula is correctly telling you: on NVL72 at heavy batch, EP doesn't buy decode latency past trivial widths. To benefit, either drop B (spread thin) or accept that EP is for memory fit, not latency.

### 5.2 Case 2: DeepSeek prod (H800, EP32 prefill / EP144 decode)

- `β_nvlink ≈ 3350/450 ≈ 7.4` (H800 NVLink capped at ~400 GB/s).
- `β_scaleout ≈ 3350/100 ≈ 33`.
- Crosses rack at S=8 → `β_scaleout` binds.

Prefill (B per device ≈ 16,384):
```
E_max ≈ 2.62e9 / (229e3 · 16384 · 33) ≈ 0.02
```
Decode (B per replica ≈ 256–1024):
```
B = 1024:  E_max ≈ 0.34
B = 256:   E_max ≈ 1.36
```

**Interpretation:** DeepSeek's choice of EP144 decode is **NOT EP_max-driven**. It's driven by:
1. **Memory fit:** 671 GB FP8 weights → need ≥9 H800s just to hold; KV needs more.
2. **Per-expert batch saturation** ("300 × sparsity" rule — each expert wants ≥ ~2,400 tokens per step at k=8).
3. **Comm-compute overlap:** dual-batch / 5-stage pipeline hides the all-to-all behind compute — they choose `E_p > E_max` and pay for it with software overlap.

**This is a critical honest caveat for the doc and UI.** EP_max is a latency-floor formula. Production EP topology is rarely chosen on that floor alone.

### 5.3 Case 3: 1× HGX H100 (8 GPUs), DeepSeek-V3, B = 500

Doesn't fit — 671 GB FP8 weights > 8 × 80 GB = 640 GB even before KV/acts. Memory-infeasible; EP_max is moot. **The calculator should error on memory fit, not report EP_max** in this regime.

### 5.4 Case 4: Single H100, degenerate

`E_p = 1` always satisfies `E_p ≤ E_max` trivially (no all-to-all, comm term → 0). Formula returns `+∞` in the limit. ✓

---

## 6. Reconciliation with Pope's "300 × sparsity"

Both rules capture the **same constraint** — each expert needs enough tokens to saturate its slice of compute — from opposite sides:

- **Pope's batch-side rule:** fix `E_p ≈ E_total / k`, ask what `B` makes per-expert compute compute-bound. Answer: `B ≥ 300 · (E/k)`.
- **Chip-side EP_max:** fix `B`, ask what `E_p` keeps per-expert weight load above the all-to-all floor.

Combine. At EP_max equality with the in-scale-up formula, substitute Pope's minimum batch `B = 300 · (E_total/k)`:

```
E_max · 300 · (E_total/k) · 4 · k · D · bytes_act · β_nvlink ≈ N_total · bytes_w
E_max · 1200 · E_total · D · bytes_act · β_nvlink           ≈ N_total · bytes_w

           N_per_expert · bytes_w
E_max ≈ ───────────────────────────────────────
        1200 · D · bytes_act · β_nvlink
```

For DeepSeek (`N_per_expert ≈ 2.62 GB`, `D = 7168`, FP8, `β_nvlink ≈ 7.4`):
```
E_max ≈ 2.62e9 / (1200 · 7168 · 1 · 7.4) ≈ 41
```

**~32–48 depending on rounding — matches DeepSeek's EP32 prefill order of magnitude**, and matches the Pope quote *"we have a divisibility problem… we're only going to use 64 of them."* Both rules independently land in the EP32–EP72 sweet spot. They are coherent.

The chip-side formula's added value vs Pope's batch-side: it tells you **how the answer moves when you cross the rack** (`β` jumps 8×, so `E_max` drops 8× — Pope's "rack bounds the expert layer" stated as numbers).

---

## 7. Current repo state — what's already in place

### 7.1 MoE schema (✅ landed via parallel session, commits `73efeec` → `0fdb7b0`)

`src/data/models.json` DeepSeek-V3 entry has:
```json
{
  "params_b_total": 671,
  "params_b_active": 37,
  "n_experts": 256,
  "experts_per_token": 8,
  "attn_type": "MLA",
  "kv_lora_rank": 512,
  "qk_rope_head_dim": 64,
  ...
}
```

`src/calc.mjs`:
- `weightsBytes(params_b_total, ...)` — HBM math uses total ✓
- `bCrit` applies `sparsity = params_b_total / params_b_active` (Scaling Book Q5 form) ✓
- `stepTime` separates memory (total weights) from compute (active params) ✓
- MoE info diagnostic fires when `total > active` ✓

`src/data/hardware.json` `_schema` already documents `ici_bw_gbs` as NVLink-only (intra-rack) with `scaleout_bw_gbs` not yet present.

### 7.2 Three-tier β sidebar (✅ landed, commit `60d1453`)

`reference/MODEL_SIZING_SCALING_REFERENCE.md` §6 (lines ~555–567) already has a sidebar naming the three fabrics (HBM > NVLink > InfiniBand/RoCE) and the two β ratios (`β_calc ≈ 3.7` intra-NVLink, `β_rack ≈ 8` cross-rack). **EP_max builds on this scaffolding without contradiction.**

### 7.3 What's missing

| Need | Location |
|---|---|
| `scaleout_bw_gbs` per hardware row | `src/data/hardware.json` |
| `scaleup_size` per hardware row | `src/data/hardware.json` |
| `epMax()` function | `src/calc.mjs` |
| Integration into `compute()` | `src/calc.mjs` |
| Goldens for DeepSeek-V3 EP_max | `tests/fixtures/golden.json` |
| Unit + e2e tests | `tests/calc.test.mjs`, `tests/calculator.e2e.spec.mjs` |
| EP_max readout tile | `src/ui.mjs` + `tools/build_sizing_calculator_html.py` |
| §6 EP_max derivation subsection | `reference/MODEL_SIZING_SCALING_REFERENCE.md` |
| Fix EP72→EP144 mislabel | `reference/MODEL_SIZING_SCALING_REFERENCE.md` §6 line ~601, §7 line ~644 |

---

## 8. 🔴 Doc bug surfaced by this research (independent of EP_max)

`reference/MODEL_SIZING_SCALING_REFERENCE.md` §6 line ~601 and §7 line ~644 both say:

> *"Decode: EP72 over 9×H100 nodes"*

That's **SGLang/LMSYS's open-source replication**, not DeepSeek production. DeepSeek's actual decode (per their Feb 2025 Open-Infra-Index Day 6 publication) is **EP144 over 18 H800 nodes**.

**Fix:** re-label one as "DeepSeek prod" and the other as "SGLang OSS replication." This is independent of EP_max implementation — pure doc correction, can land any time.

---

## 9. Caveats and unverified claims

1. **Uniform routing assumption.** The `4·B·k·D` coefficient assumes uniform expert routing and ignores the shared-expert path. Production has hot experts; DeepSeek adds redundant copies. Real all-to-all is ~10–30% higher (consistent with Meta's 100KB–2MB decode-message number). Treat the formula as an idealized lower bound on comm traffic.

2. **β_scaleout / β_nvlink ≈ 8 is Pope's universal generalization.** Real per-hardware varies: H100 SXM5 → 400 Gbps IB → ratio ≈ 7. **Compute per-hardware, don't hardcode 8.**

3. **DeepSeek-V3 `k_active` discrepancy.** Tech report: 8 routed + 1 shared of 256. Pope's transcript: "32 of 256" (a simplification). **Use the tech-report value (k=8); footnote the discrepancy in the doc.**

4. **The "300" constant in Pope's rule is the GPU `B_crit` ridge** (FLOPs/HBM_BW), which our `bCrit()` already computes per-hardware (~295 H100 BF16, ~590 H100 FP8). The EP_max derivation should ideally use the same `bCrit(hw)` for the "300" — improvement over Pope's universal constant.

5. **DeepEP's "726 GB/s NVLink dispatch" is effective bandwidth post-kernel-overhead**, not raw NVLink BW (900 GB/s on H100). If we want a tuned EP_max, use the effective (~720) numbers in `scaleout_bw_gbs`/`ici_bw_gbs`. Footnote needed.

6. **NVL72 per-GPU NVLink5 bidir BW (~1.8 TB/s)** is derived (130 TB/s aggregate ÷ 72). NVIDIA quotes 1.8 TB/s elsewhere — pin a primary-source citation before locking into the hardware table.

7. **`T_HBM > T_a2a` framing only applies to the MoE FFW block.** Attention (MLA) doesn't enter the all-to-all. EP_max is a constraint on the MoE FFW contribution to step time, not the whole layer. Note in reference doc so readers don't over-interpret.

---

## 10. Implementation plan

Two phases. They don't share files — **can run in parallel** (one sub-agent each) or sequentially.

### Phase 1 — Doc-only (safe to land any time)

**File:** `reference/MODEL_SIZING_SCALING_REFERENCE.md`

1. **Fix the EP72/EP144 mislabel** in §6 (line ~601) and §7 (line ~644). Re-label one as "DeepSeek prod EP144 / 18 nodes" and the other as "SGLang OSS replication EP72 / 9 nodes." Cite Open-Infra-Index Day 6 for the prod numbers.

2. **Add §6 EP_max derivation subsection**, paralleling Y_max. Include:
   - Setup: dispatch + combine all-to-all, traffic shape.
   - Byte counts: `4 · B · k · D · bytes_act` per GPU per layer (derive).
   - Weight-load term: `N_per_expert · bytes_w / (E_p · W_hbm)` per GPU per layer.
   - Crossover: `E_max ≈ N_per_expert · bytes_w / (4 · k · B · D · bytes_act · β_fabric)`.
   - Two regimes (in-rack β_nvlink, cross-rack β_scaleout).
   - Worked DeepSeek-V3 example reproducing the EP32-ish number when combined with the 300×sparsity rule (§6 worked-example pattern).
   - Honest caveat: DeepSeek's EP144 is NOT EP_max-driven (memory + per-expert batch + overlap dominate).
   - Link back to the existing 3-tier β sidebar.
   - Footnote: formula only governs MoE FFW; attention (MLA) doesn't enter.

### Phase 2 — Code + data + tests + UI

**Files touched:**
- `src/data/hardware.json` (schema + every row)
- `src/calc.mjs` (new function + `compute()` integration)
- `src/ui.mjs` (readout tile)
- `tools/build_sizing_calculator_html.py` (HTML template tile)
- `tests/fixtures/golden.json` (new goldens)
- `tests/calc.test.mjs` (unit tests)
- `tests/data.test.mjs` (schema validation)
- `tests/test_build.py` (required-IDs list)
- `tests/calculator.e2e.spec.mjs` (e2e for the new tile)

**Steps:**

1. **Extend `hardware.json` schema** with two new fields:
   - `scaleout_bw_gbs` (number, GB/s; nullable for single-node-only cards).
   - `scaleup_size` (int; default 8 for HGX, 72 for NVL72, 1 for PCIe-only).

   Per-card values to use (verify against datasheets before locking):
   | Card | scaleout_bw_gbs | scaleup_size | Source |
   |---|---|---|---|
   | T4 (g4dn) | null | 1 | PCIe-only |
   | L4 (g6) | null | 1 | PCIe-only |
   | A10G (g5) | null | 1 | PCIe-only |
   | A100-40GB SXM4 | ~50 | 8 | HGX A100, 200G IB |
   | A100-80GB SXM4 | ~50 | 8 | HGX A100, 200G IB |
   | H100-80GB SXM5 | ~100 | 8 | HGX H100, 400G IB CX7 |
   | H200 SXM5 | ~100 | 8 | HGX H200, 400G IB CX7 |
   | (future) NVL72 GB200 | ~400 | 72 | rack-scale, NVLink5 within, 400G+ IB between |

   Update `_schema` docs. Update `_sources` block.

2. **Update `tests/data.test.mjs`** to validate the new fields:
   - `scaleout_bw_gbs` either null or positive number in plausible range.
   - `scaleup_size` positive integer, typically 1, 4, 8, or 72.
   - Sanity: `hbm_bw_gbs / scaleout_bw_gbs` (if non-null) should be in `[5, 50]`.

3. **Implement `epMax()` in `src/calc.mjs`:**

   ```js
   /**
    * Max useful EP degree per replica before the all-to-all floor dominates
    * per-layer decode time. Two regimes:
    *   in-rack:    E ≤ scaleup_size → β = β_nvlink
    *   cross-rack: E > scaleup_size → β = β_scaleout
    * Returns +Infinity for dense models (no constraint modelled).
    *
    * Source: see EP_MAX_PLAN.md §3 for the derivation, §4 for the closed form,
    * §5 for worked examples. Scaling Book Q5 reconciliation in §6.
    */
   export function epMax({ model, hw, batch, weight_prec, act_prec }) {
     // Dense models: no EP-vs-comm constraint.
     if (!model.n_experts || !model.experts_per_token) return { ep_max_in_rack: Infinity, ep_max_cross_rack: Infinity, regime: "dense" };
     const bytes_w = DTYPE_BYTES[weight_prec];
     const bytes_act = DTYPE_BYTES[act_prec];
     const N_per_expert_bytes = (model.params_b_total * 1e9 / model.n_experts) * bytes_w;
     const denom = 4 * model.experts_per_token * batch * model.d_model * bytes_act;
     const β_nvlink = hw.hbm_bw_gbs / hw.ici_bw_gbs;
     const β_scaleout = hw.scaleout_bw_gbs ? hw.hbm_bw_gbs / hw.scaleout_bw_gbs : Infinity;
     return {
       ep_max_in_rack: N_per_expert_bytes / (denom * β_nvlink),
       ep_max_cross_rack: N_per_expert_bytes / (denom * β_scaleout),
       β_nvlink, β_scaleout,
       regime: "moe",
     };
   }
   ```

4. **Wire into `compute()`:** add `ep_max_in_rack` + `ep_max_cross_rack` to `metrics`. Emit info-level diagnostic if MoE model + `ngpus > ep_max_in_rack`, naming the regime and quoting the 300×sparsity caveat. **Don't error** — DeepSeek's prod EP144 deliberately exceeds EP_max.

5. **Goldens (`tests/fixtures/golden.json`)** — minimum three:
   - DeepSeek-V3 + H100 + B=1024 + FP8 + in-rack → assert `ep_max_in_rack` in `[1, 5]` band.
   - Same model + cross-rack regime → assert `ep_max_cross_rack` ≈ `β_scaleout / β_nvlink` × in_rack (sanity-check β ratio).
   - DeepSeek-V3 + NVL72 + B=2,000 → assert `ep_max ≈ 1–2` (matches Case 1 above; documents the heavy-batch regime).

6. **Unit tests (`tests/calc.test.mjs`):**
   - Dense models return `+Infinity` for both EP_max fields.
   - DeepSeek-V3 MoE returns finite, positive EP_max.
   - β-ratio invariant: `ep_max_in_rack / ep_max_cross_rack ≈ β_scaleout / β_nvlink`.
   - Doubling B halves EP_max (linear-in-B invariant).
   - Diagnostic fires when `ngpus > ep_max_in_rack` for MoE; doesn't fire for dense.

7. **UI tile:** add a readout in the parallelism section showing `EP ≤ X in-rack, ≤ Y cross-rack` with a 1-line tooltip explaining the latency-floor framing. Hide / show "—" for dense models. Update `tools/build_sizing_calculator_html.py` HTML template + `src/ui.mjs` paint logic. Add the new ID to `tests/test_build.py` required-IDs list.

8. **e2e (`tests/calculator.e2e.spec.mjs`):** add a test that selects DeepSeek-V3 and asserts the EP_max tile renders with a finite number; switches to Llama-3-8B and asserts it shows "—" (dense).

9. **Rebuild + run all 4 layers green.**

---

## 11. Honest framing for the UI / doc

When this lands, surface this prominently in the EP_max tile tooltip and the doc subsection:

> **EP_max is the latency floor — not a topology prescription.** Production EP choices (e.g., DeepSeek-V3's EP144 decode) routinely exceed EP_max because they're driven by **memory fit** + **per-expert batch saturation** + **comm-compute overlap** (software hiding the all-to-all behind compute). EP_max tells you how wide EP can go before adding more chips stops cutting decode latency. It does *not* tell you the optimal EP for production.

Without this caveat, users will see EP_max=1.3 for DeepSeek-V3-on-NVL72 and conclude something is broken. It's not — it's correctly reporting that NVL72's heavy-batch decode is comm-bound and EP_max chooses 1.

---

## 12. Quick-start checklist for picking this up

When ready to implement:

1. Re-read this file end-to-end (15 min).
2. Re-read `reference/MODEL_SIZING_SCALING_REFERENCE.md` §6 Y_max derivation (5 min) — EP_max parallels it structurally.
3. Re-read `calculators/sizing_calc/src/calc.mjs` `yMax()` and `compute()` (5 min) — `epMax()` mirrors `yMax()`.
4. Decide: Phase 1 only, Phase 2 only, or both in parallel (sub-agents).
5. Spin up sub-agent(s) with mandate quoted from §10 above.
6. After their report: rebuild, run 4-layer test suite, commit per-phase, push.

Estimated effort: Phase 1 ≈ 30 min sub-agent + 10 min review. Phase 2 ≈ 90 min sub-agent + 20 min review.
