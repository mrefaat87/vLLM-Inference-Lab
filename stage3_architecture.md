# Stage 3+: Production Inference Platform — System Architecture

## Core Design Principle: Async-First / Queue-Based

GPU capacity is scarce and expensive. Unlike CPU services where you can scale reactively in seconds,
GPU nodes take 2-5 minutes to provision and models take 30-60s to load. The system must:
- **Never reject requests** — queue them instead, increase wait time
- **Never let GPUs idle** — RabbitMQ pushes work to workers via prefetch
- **Decouple ingestion from processing** — accept at any rate, serve at GPU speed

This is the SQS + ASG pattern applied to inference.

```
┌──────────┐     ┌──────────────┐     ┌─────────┐     ┌───────────────┐
│  Client  │────►│  API Gateway │────►│  Queue  │────►│  GPU Workers  │
│          │◄────│  (FastAPI)   │     │         │     │  (vLLM pods)  │
│          │     │              │     │ Priority│     │               │
│ POST /   │     │ • Accept req │     │ queues: │     │ • Receive job │
│ generate │     │ • Validate   │     │ • premium│    │ • Run vLLM    │
│          │     │ • Return     │     │ • standard│   │ • Stream to   │
│ GET /    │     │   job_id     │     │ • batch │     │   result store│
│ status/  │     │ • Rate limit │     │         │     │               │
│ {job_id} │     │   per client │     │ Scaling │     │ Broker PUSHES │
│          │     │              │     │ signal: │     │ (never idle)  │
│ SSE /    │     │ • Stream     │     │ depth   │     │               │
│ stream/  │     │   results    │     │         │     │               │
│ {job_id} │     │   back       │     └─────────┘     └───────────────┘
└──────────┘     └──────────────┘           │               │
                        ▲                   │               │
                        │            ┌──────▼───────────────▼──┐
                        └────────────│     Result Store         │
                                     │  (Redis)                 │
                                     │                          │
                                     │  • Job status            │
                                     │  • Streamed token chunks │
                                     │  • Final result          │
                                     │  • Metrics (latency,     │
                                     │    tokens, queue wait)   │
                                     └──────────────────────────┘
```

### Client Interaction Pattern (Phase 1: Streaming Only)

```
POST /generate → 202 {job_id: "abc123"}
GET /stream/abc123 → SSE stream
  data: {token: "The"}
  data: {token: " answer"}
  data: {token: " is"}
  ...
  data: {status: "complete"}
```

Token delivery uses Redis pub/sub (not store-and-forward). Worker publishes tokens
to a channel, API gateway subscribes and relays via SSE. Added latency: <1ms per token.

Result store + polling + fire-and-forget patterns added later when batch use cases arrive.

## Full System Architecture (End State — Phase 8)

```
                           ┌─────────────────────────────────┐
                           │         INTERNET / CLIENT        │
                           └──────────────┬──────────────────┘
                                          │
                           ┌──────────────▼──────────────────┐
                           │     AWS ALB / NLB (Ingress)      │
                           │     TLS termination, health      │
                           └──────────────┬──────────────────┘
                                          │
                    ┌─────────────────────────────────────────────┐
                    │                 EKS CLUSTER                  │
                    │                                              │
                    │  ┌───────────────────────────────────────┐  │
                    │  │       API GATEWAY (FastAPI pods)       │  │
                    │  │                                        │  │
                    │  │  • Accept requests, return job_id      │  │
                    │  │  • Rate limit per client               │  │
                    │  │  • Classify request → tier/priority    │  │
                    │  │  • Serve results (poll / SSE stream)   │  │
                    │  └───────────────┬────────────────────────┘  │
                    │                  │                            │
                    │  ┌───────────────▼────────────────────────┐  │
                    │  │       QUEUE + RESULT STORE (Redis)      │  │
                    │  │                                         │  │
                    │  │  Priority queues per tier:              │  │
                    │  │    queue:fast:premium                   │  │
                    │  │    queue:fast:standard                  │  │
                    │  │    queue:quality:premium                │  │
                    │  │    queue:quality:standard               │  │
                    │  │                                         │  │
                    │  │  Streaming (pub/sub, not stored):        │  │
                    │  │    channel:job:{id} (tokens → SSE)      │  │
                    │  │                                         │  │
                    │  │  Queue depth → scaling signal → KEDA    │  │
                    │  └───────────────┬────────────────────────┘  │
                    │                  │                            │
                    │  ┌───────────────▼────────────────────────┐  │
                    │  │       WORKER TIER (broker pushes jobs)   │  │
                    │  │                                         │  │
                    │  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ │  │
                    │  │  │ FAST    │ │ QUALITY │ │ EMBEDDING│ │  │
                    │  │  │ WORKERS │ │ WORKERS │ │ WORKERS  │ │  │
                    │  │  │         │ │         │ │          │ │  │
                    │  │  │ Qwen    │ │ Qwen    │ │ BGE /    │ │  │
                    │  │  │ 1.5B    │ │ 7B-AWQ  │ │ E5       │ │  │
                    │  │  │         │ │         │ │          │ │  │
                    │  │  │ Receive │ │ Receive │ │ Pull     │ │  │
                    │  │  │ from    │ │ from    │ │ from     │ │  │
                    │  │  │ queue:  │ │ queue:  │ │ queue:   │ │  │
                    │  │  │ fast:*  │ │ quality:│ │ embed:*  │ │  │
                    │  │  │         │ │ *       │ │          │ │  │
                    │  │  │ SLA:    │ │ SLA:    │ │ SLA:     │ │  │
                    │  │  │ P99<1s  │ │ P99<5s  │ │ P99<200ms│ │  │
                    │  │  └────┬────┘ └────┬────┘ └─────┬────┘ │  │
                    │  └───────┼───────────┼────────────┼───────┘  │
                    │          │           │            │           │
                    │  ┌───────▼───────────▼────────────▼───────┐  │
                    │  │       AUTOSCALING LAYER                 │  │
                    │  │                                         │  │
                    │  │  KEDA (ScaledObjects)                   │  │
                    │  │    ↕ worker pods per tier               │  │
                    │  │    Signal: Redis queue depth             │  │
                    │  │                                         │  │
                    │  │  Karpenter (NodePools)                  │  │
                    │  │    ↕ GPU nodes based on pending pods    │  │
                    │  │    spot → on-demand fallback            │  │
                    │  │    g4dn / g5 / g6 diversification      │  │
                    │  │                                         │  │
                    │  │  Pre-provisioned capacity:              │  │
                    │  │    Warm pool of GPU nodes (always on)   │  │
                    │  │    Prefetch keeps workers fed = 100% util│  │
                    │  └────────────────────────────────────────┘  │
                    │                                              │
                    │  ┌────────────────────────────────────────┐  │
                    │  │       OBSERVABILITY LAYER               │  │
                    │  │                                         │  │
                    │  │  Prometheus ◄── vLLM /metrics           │  │
                    │  │             ◄── DCGM Exporter (GPU)     │  │
                    │  │             ◄── Redis Exporter (queue)  │  │
                    │  │             ◄── kube-state-metrics       │  │
                    │  │             ◄── node-exporter            │  │
                    │  │             ◄── API Gateway metrics      │  │
                    │  │                                         │  │
                    │  │  Grafana                                │  │
                    │  │    Dashboard: Queue Pipeline             │  │
                    │  │    Dashboard: Latency SLAs (per tier)   │  │
                    │  │    Dashboard: GPU Hardware               │  │
                    │  │    Dashboard: Capacity & Scaling         │  │
                    │  │    Dashboard: Cost Tracking              │  │
                    │  │                                         │  │
                    │  │  Alertmanager                           │  │
                    │  │    Queue depth spike, P99 breach,       │  │
                    │  │    preemptions, KV > 80%, worker down   │  │
                    │  └────────────────────────────────────────┘  │
                    │                                              │
                    │  ┌────────────────────────────────────────┐  │
                    │  │       NODE LAYER (Karpenter-managed)    │  │
                    │  │                                         │  │
                    │  │  ┌──────────┐ ┌──────────┐ ┌────────┐ │  │
                    │  │  │ g4dn.xl  │ │ g5.xlarge│ │ CPU    │ │  │
                    │  │  │ T4 16GB  │ │ A10G 24GB│ │ nodes  │ │  │
                    │  │  │ spot     │ │ spot     │ │ (API GW│ │  │
                    │  │  │          │ │          │ │ Redis, │ │  │
                    │  │  │ Small    │ │ Large    │ │ Prom,  │ │  │
                    │  │  │ models   │ │ models   │ │ Grafana│ │  │
                    │  │  └──────────┘ └──────────┘ └────────┘ │  │
                    │  └────────────────────────────────────────┘  │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │       INFRASTRUCTURE (Terraform)             │
                    │                                              │
                    │  VPC + Subnets + Security Groups             │
                    │  EKS Cluster + Managed Node Group (CPU)      │
                    │  Karpenter Controller + NodePool CRDs        │
                    │  EBS CSI Driver (model cache persistence)    │
                    │  IAM Roles (IRSA for Karpenter, EBS, ECR)    │
                    └─────────────────────────────────────────────┘
```

## Phase-by-Phase Build Order

Each phase adds a layer and produces a working, testable system.

### Phase 1: EKS + Queue + Single Worker
```
Client → API Gateway (FastAPI) → Redis Queue → vLLM Worker (single replica)
              │                       │               │
              │                       └───► Result ◄──┘
              └──────── poll/stream ──────── Store
                                              │
                                    Karpenter provisions g4dn.xlarge spot
                                    NVIDIA device plugin exposes GPU
                                    EBS PV for HuggingFace model cache
```
The queue is part of the foundation from day one.

### Phase 2: Observability
```
Phase 1 + Prometheus + Grafana + DCGM Exporter + Redis Exporter
         (all as K8s pods on CPU node)
         SSH tunnel to Grafana :3000
         Dashboards: queue depth, TTFT/TBT/P99, GPU util, KV cache
```

### Phase 3: Autoscaling
```
Phase 2 + KEDA ScaledObject → scale workers on Redis queue depth
                            → Karpenter provisions more GPU nodes
         Pre-provisioned capacity: warm pool of N GPU nodes
         Load test: ramp up → watch scale → ramp down → watch contract
```

### Phase 4: Scaling Policy Comparison + Cold Start ✅
```
Phase 3 + 4 KEDA scaling policies (queue-only vs composite KV triggers)
        + Pareto-distributed workloads (80/15/4/1)
        + Cold start optimization (S3 model cache, CUDA warmup, graceful drain)
        + Karpenter hardening (disruption budgets, consolidation delay)
Result: Composite KV triggers differentiate under stress (4x throughput vs queue-only)
```

### Phase 5: Smart Routing & Inference Optimization
```
Phase 4 + Push-based routing (per-replica local queues behind smart router)
        + Prefix caching + cache-aware routing (hash system prompt → warm replica)
        + Prefill sharing (shared prefix computed once, branch into decode streams)
        + max_num_batched_tokens tuning for workload mix
        + Speculative decoding in pipeline (0.5B draft + 7B target)
        + SGLang comparison (predictive reservation vs admit-everything)
        + Deficit-based fairness (D2LPM) for multi-tenant routing
        + Evaluate Gateway API Inference Extension (K8s SIG)
```

### Phase 6: Multi-Model Serving & Graceful Degradation
```
Phase 5 + Second model tier (1.5B for classification, 7B for generation)
        + GPU bin-packing (both models fit on T4: ~2GB + ~4GB)
        + Independent KEDA scalers per model queue
        + Tiered model fallback (7B saturated → route to 1.5B)
        + CUDA Checkpoint/Restore for fast model swaps (NVMe SSD snapshots)
        + Per-model observability and cost tracking
```

### Phase 7: Production Hardening
```
Phase 6 + API gateway with token bucket rate limiting (RPM + input/output TPM)
        + Priority queues with starvation prevention (aging/boosting, not strict)
        + Prediction-based early rejection (Mooncake pattern)
        + Token-aware rate limiting per client (token cost, not request count)
        + Response length limiting under load
        + Queue duration as scaling signal (replaces queue depth in KEDA)
        + Cost attribution per tenant/model
        + Safety/guardrails (content filtering, audit logging)
        + KV cache eviction to CPU (QLM pattern)
```

### Phase 8: Disaggregated Inference
```
Phase 7 + Feasibility scoping (T4 hardware check, Ray Serve vs NVIDIA Dynamo)
        + Llumnix live migration as lighter alternative (12x P99 improvement)
        + Separate prefill pool (compute-optimized) + decode pool (memory-optimized)
        + KV cache transfer via framework-native mechanisms
        + Independent scaling per pool
        + Comparison: monolithic vs live-migration vs disaggregated
```

## AWS Analogy Map

| Inference Platform Component | AWS Equivalent |
|------------------------------|----------------|
| EKS Cluster | The "region" — your compute boundary |
| Karpenter NodePool | EC2 Auto Scaling Group (but smarter — no launch config needed) |
| GPU Node | EC2 instance in the ASG |
| vLLM Pod (worker) | Application process on the instance |
| Redis Queue | SQS queue (with priority = separate queues) |
| API Gateway (FastAPI) | API Gateway + Lambda that enqueues to SQS |
| KEDA ScaledObject | Target Tracking Scaling Policy on ApproximateNumberOfMessagesVisible |
| Ray Serve / Dynamo (Phase 8) | Application Load Balancer + target groups |
| Cache-aware router | Session-affine ALB (sticky sessions on prefix hash) |
| Prometheus | CloudWatch Metrics |
| Grafana | CloudWatch Dashboards |
| Alertmanager | CloudWatch Alarms → SNS |
| KV cache utilization | Memory utilization metric on an instance |
| Queue depth | ApproximateNumberOfMessagesVisible (SQS) — THE scaling signal |
| Preemption count | Spot interruption notices |
| EBS PV for model cache | EBS volume for warm AMI data |
| NVIDIA device plugin | Instance type metadata (vCPU/memory advertised to scheduler) |
| Pre-provisioned GPU nodes | Reserved Instances / Capacity Reservations |

## Scaling Signal Decision Tree

```
Is Redis queue depth > threshold?
  YES → Scale up worker pods (KEDA)
        Are there pending pods?
          YES → Scale up nodes (Karpenter)
                Is spot capacity available?
                  YES → Provision spot
                  NO  → Fall back to on-demand (or different instance type)
          NO → Pods fit on existing nodes ✓

Is KV cache > 80% on any worker?
  YES → This worker is near capacity cliff
        Scale up pods to spread load

Is P99 TTFT > SLA threshold?
  YES → Likely prefill bottleneck
        Check if it's a node scaling issue (cold start) vs overload

Are all workers idle for > cooldown period?
  YES → Scale down pods (KEDA)
        Are nodes underutilized?
          YES → Karpenter consolidates (drains + terminates)
```

## Pre-Provisioned vs Reactive Capacity

Reactive scaling doesn't always work for GPUs because:
1. Node provision = 2-5 minutes (vs 30s for CPU)
2. Model loading = 30-60 seconds on top of that
3. Spot capacity might not be available at all

### Strategy 1: Reactive (scale-to-zero, pay per use)
```
No traffic → 0 GPU nodes → 0 cost
Traffic arrives → queue builds → KEDA scales pods → Karpenter provisions node (2-5 min)
First request waits 3-6 minutes. Subsequent requests served immediately.
```
Good for: dev/test, batch workloads that tolerate delay.

### Strategy 2: Pre-provisioned warm pool (reserved capacity)
```
Always: N GPU nodes running + N worker pods consuming from queue (broker-push via prefetch)
Traffic arrives → served immediately (no cold start)
Burst → KEDA scales beyond N → Karpenter provisions additional nodes
Idle workers still cost money but guarantee SLA.
```
Good for: production workloads with latency SLAs.

### Strategy 3: Hybrid (warm base + reactive burst)
```
Base: 1 GPU node always warm (handles normal traffic)
Burst: KEDA + Karpenter scale up additional nodes as queue depth grows
Queue absorbs the burst while new capacity comes online
Scale back to base after cooldown
```
This is the pattern we'll implement. The queue makes it possible — without the queue,
burst traffic during scale-up would get rejected or timeout.

## Cache-Aware Routing (Phase 5+)

Three levels of KV cache reuse, analogous to caching tiers:

### Level 1: Request Router (load balancer layer)
Hash the system prompt / prefix → route to the replica that has it cached.
Like session-affine ALB routing. "The session is the prompt prefix."
SGLang pioneered this with RadixAttention. We implement this in Phase 5 with a smart router.

### Level 2: Within vLLM (automatic, single-replica)
vLLM's prefix caching — if two requests on the same replica share a prefix,
the second skips prefill for the shared portion. Automatic, no configuration needed.
Like in-process memcached on the instance.

### Level 3: Distributed KV Store (Phase 8, disaggregated)
Shared KV cache accessible by any worker (NVIDIA NIXL, RDMA, NVLink).
Eliminates the routing problem but adds network transfer cost.
Like ElastiCache — shared state accessible from any instance.
