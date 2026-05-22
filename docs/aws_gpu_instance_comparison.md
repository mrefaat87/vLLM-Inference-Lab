# AWS GPU & Accelerator Instance Types — Full Comparison

## NVIDIA GPU Lineage Map

| AWS GPU | NVIDIA Architecture | NVIDIA Product Line | Process | Key Feature |
|---|---|---|---|---|
| T4 | Turing | GeForce RTX 20xx / Quadro RTX | 12nm | First INT8 tensor cores for inference |
| V100 | Volta | Tesla V100 | 12nm | First tensor cores, HBM2 |
| A10G | Ampere | GeForce RTX 30xx family | 8nm | 2nd gen tensor cores, inference-tuned |
| A100 (40/80GB) | Ampere | Data center flagship | 7nm | MIG, 3rd gen NVLink, HBM2e |
| L4 | Ada Lovelace | GeForce RTX 40xx family | 5nm | FP8 support, energy efficient inference |
| L40S | Ada Lovelace | Workstation/DC variant | 5nm | 48GB VRAM, replaces A40 |
| H100 | Hopper | Data center flagship | 4nm | Transformer Engine, FP8, NVLink 4.0 |
| H200 | Hopper | H100 refresh | 4nm | HBM3e (141GB), same compute as H100 |

**The naming pattern:**
- **T/L** = inference/efficiency line (T for Turing, L for Lovelace) → shows up in **G instances**
- **V/A/H** = training/flagship line (Volta, Ampere, Hopper) → shows up in **P instances**

---

## G Instances — Inference & Graphics (NVIDIA)

| Instance | Accelerator | Arch | VRAM | FP16 TFLOPS | vCPUs | RAM | Network | Spot ~$/hr |
|---|---|---|---|---|---|---|---|---|
| g4dn.xlarge | 1x T4 | Turing | 16 GB | 65 | 4 | 16 GB | Up to 25 Gbps | ~$0.16 |
| g4dn.12xlarge | 4x T4 | Turing | 64 GB | 260 | 48 | 192 GB | 50 Gbps | ~$1.20 |
| g5.xlarge | 1x A10G | Ampere | 24 GB | 125 | 4 | 16 GB | Up to 10 Gbps | ~$0.38 |
| g5.12xlarge | 4x A10G | Ampere | 96 GB | 500 | 48 | 192 GB | 40 Gbps | ~$1.70 |
| g5.48xlarge | 8x A10G | Ampere | 192 GB | 1000 | 192 | 768 GB | 100 Gbps | ~$4.80 |
| g6.xlarge | 1x L4 | Ada Lovelace | 24 GB | 121 | 4 | 16 GB | Up to 25 Gbps | ~$0.30 |
| g6.12xlarge | 4x L4 | Ada Lovelace | 96 GB | 484 | 48 | 192 GB | 40 Gbps | ~$1.40 |
| g6e.xlarge | 1x L40S | Ada Lovelace | 48 GB | 362 | 4 | 16 GB | Up to 25 Gbps | ~$0.45 |
| g6e.12xlarge | 4x L40S | Ada Lovelace | 192 GB | 1448 | 48 | 192 GB | 40 Gbps | ~$3.50 |

**Key traits:**
- T4/A10G/L4/L40S are inference-optimized GPUs — good throughput/$ for serving
- Lower power draw = cheaper to run
- FP16/INT8 inference is the sweet spot
- G6e with L40S is the newest — 48 GB VRAM makes it viable for 13B-30B models

---

## P Instances — Training & Large-Scale Inference (NVIDIA)

| Instance | Accelerator | Arch | VRAM | FP16 TFLOPS | vCPUs | RAM | Interconnect | Spot ~$/hr |
|---|---|---|---|---|---|---|---|---|
| p3.2xlarge | 1x V100 | Volta | 16 GB | 125 | 8 | 61 GB | — | ~$0.92 |
| p3.16xlarge | 8x V100 | Volta | 128 GB | 1000 | 64 | 488 GB | NVLink 2.0 | ~$7.34 |
| p4d.24xlarge | 8x A100-40GB | Ampere | 320 GB | 2496 | 96 | 1152 GB | NVLink 3.0 + EFA 400G | ~$9.80 |
| p4de.24xlarge | 8x A100-80GB | Ampere | 640 GB | 2496 | 96 | 1152 GB | NVLink 3.0 + EFA 400G | ~$12.50 |
| p5.48xlarge | 8x H100 | Hopper | 640 GB | 15840 | 192 | 2048 GB | NVSwitch + EFA 3200G | ~$20+ |
| p5e.48xlarge | 8x H200 | Hopper | 1128 GB | 15840 | 192 | 2048 GB | NVSwitch + EFA 3200G | ~$25+ |

**Key traits:**
- NVLink/NVSwitch between GPUs — critical for tensor parallelism across GPUs
- EFA (Elastic Fabric Adapter) on p4d+ — enables multi-node training with NCCL
- Designed for large model training (pre-training, RLHF)
- Also used for inference of very large models (70B+, 405B) that need multi-GPU VRAM

---

## Inf Instances — AWS Inferentia (Custom Silicon)

| Instance | Accelerator | Chip | NeuronCores | vCPUs | RAM | Network | On-Demand ~$/hr |
|---|---|---|---|---|---|---|---|
| inf1.xlarge | 1x Inferentia1 | Inferentia v1 | 4 | 4 | 8 GB | Up to 25 Gbps | ~$0.23 |
| inf1.6xlarge | 4x Inferentia1 | Inferentia v1 | 16 | 24 | 48 GB | 25 Gbps | ~$1.18 |
| inf1.24xlarge | 16x Inferentia1 | Inferentia v1 | 64 | 96 | 192 GB | 100 Gbps | ~$4.72 |
| **inf2.xlarge** | 1x Inferentia2 | Inferentia v2 | 2 | 4 | 16 GB | Up to 15 Gbps | ~$0.76 |
| **inf2.8xlarge** | 1x Inferentia2 | Inferentia v2 | 2 | 32 | 128 GB | Up to 25 Gbps | ~$1.97 |
| **inf2.24xlarge** | 6x Inferentia2 | Inferentia v2 | 12 | 96 | 384 GB | 100 Gbps | ~$6.49 |
| **inf2.48xlarge** | 12x Inferentia2 | Inferentia v2 | 24 | 192 | 768 GB | 100 Gbps | ~$12.98 |

**Key traits:**
- Inferentia v1 — optimized for small/medium models (BERT, ResNet), not great for LLMs
- Inferentia v2 — designed for LLM inference, supports models up to 175B+ with NeuronLink across chips
- Uses AWS Neuron SDK (not CUDA) — must compile models with `torch-neuronx`
- No CUDA = vLLM doesn't run natively (uses Neuron runtime instead)
- Best $/token for supported models when you're locked into the Neuron ecosystem

---

## Trn Instances — AWS Trainium (Custom Silicon)

| Instance | Accelerator | Chip | NeuronCores | HBM | vCPUs | RAM | Network | On-Demand ~$/hr |
|---|---|---|---|---|---|---|---|---|
| trn1.2xlarge | 1x Trainium1 | Trainium v1 | 2 | 32 GB | 8 | 32 GB | Up to 12.5 Gbps | ~$1.34 |
| trn1.32xlarge | 16x Trainium1 | Trainium v1 | 32 | 512 GB | 128 | 512 GB | 800 Gbps EFA | ~$21.50 |
| trn1n.32xlarge | 16x Trainium1 | Trainium v1 | 32 | 512 GB | 128 | 512 GB | 1600 Gbps EFA | ~$24.78 |
| **trn2.48xlarge** | 16x Trainium2 | Trainium v2 | 64 | 1536 GB | 192 | 2048 GB | 3200 Gbps EFA | ~$22+ |
| **trn2u.48xlarge** | 16x Trainium2 | Trainium v2 (UltraCluster) | 64 | 1536 GB | 192 | 2048 GB | 3200 Gbps EFA | ~$22+ |

**Key traits:**
- Trainium v1 — AWS's answer to A100 for training, ~50% cheaper $/TFLOP
- Trainium v2 — competes with H100/H200, designed for trillion-parameter training
- Also uses Neuron SDK — same ecosystem as Inferentia
- trn2 can also serve inference — Trainium v2 has inference-optimized modes
- UltraClusters (trn2u) — pre-provisioned multi-node clusters for massive training jobs

---

## Cross-Family Comparison

| Dimension | G (NVIDIA) | P (NVIDIA) | Inf (AWS) | Trn (AWS) |
|---|---|---|---|---|
| **Primary use** | Inference serving | Training + large inference | Inference only | Training + inference |
| **Software ecosystem** | CUDA (universal) | CUDA (universal) | Neuron SDK (AWS-only) | Neuron SDK (AWS-only) |
| **vLLM support** | Native | Native | Partial (Neuron backend) | Experimental |
| **Framework support** | PyTorch, TF, JAX, everything | Everything | PyTorch (via Neuron) | PyTorch (via Neuron) |
| **Model portability** | Run anywhere | Run anywhere | AWS-locked | AWS-locked |
| **$/token (inference)** | Good | Expensive | Best (when model compiles) | Good |
| **$/TFLOP (training)** | Not designed for it | Good | N/A | Best |
| **Spot availability** | Best (many pools) | Scarce | Limited pools | Very limited |
| **Karpenter scaling** | Excellent | Hard | Moderate | Hard |
| **Compilation overhead** | None (CUDA JIT) | None | High (Neuron compile step) | High |
| **Model support breadth** | Any model | Any model | Neuron-supported list | Neuron-supported list |

---

## Auto Scaling Mental Model

| Family | Analogy |
|---|---|
| **G instances** | **General-purpose ASG fleet** — broad capacity pools, fast scale-out, spot-friendly. Your t3/m5 equivalent for inference. |
| **P instances** | **Dedicated host / bare-metal tier** — vertical scale-up, premium, few pools. Your database or stateful tier. |
| **Inf instances** | **Graviton fleet** — cheaper per unit of work, but requires recompilation (like ARM migration). Lock-in trade for cost. |
| **Trn instances** | **HPC cluster** — purpose-built for batch training workloads. Think EMR with GPUs. The Neuron compile step is like your Spark job compilation. |

---

## Decision Tree

```
Can the model fit in 1 GPU VRAM?
├── Yes → G instance (g4dn for learning, g6/g6e for production)
└── No → How many GPUs needed?
    ├── 2-4 GPUs → G multi-GPU (g5.12xl) or Inf2 (if Neuron-compatible)
    └── 8+ GPUs → P instance (p4d/p5) or Trn (if Neuron-compatible)

Are you optimizing $/token at scale?
├── Inf2 — cheapest if your model compiles on Neuron
├── G6/G6e — cheapest CUDA option
└── P — only when model size forces it

Are you training?
├── Small fine-tune → G5 or single Trn1
├── Full pre-training → Trn1/Trn2 (cost) or P5 (ecosystem)
```

---

*Last updated: 2026-04-02*
*Note: Spot prices are approximate and vary by region/AZ. On-demand prices for Inf/Trn shown where spot is uncommon.*
