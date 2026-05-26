# Runbook

Day-to-day operational steps for running experiments against the
`inference-lab` EKS cluster.

## First-time setup (per AWS account)

```bash
# 1. Create Terraform state backend (idempotent — safe to re-run).
./eks/bootstrap_backend.sh

# 2. Confirm spot G/VT vCPU quota >= 48 for g5.12xlarge TP=4. If not,
#    open a quota increase ticket OR set --tp 1 and run on g5.xlarge first.
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-3819A6DF --region us-east-1
```

## Calculator integration

The lab now consults the sizing calculator on every run and bakes the
prediction into the result JSON. See
[`../../calculators/sizing_calc/docs/validation.md`](../../calculators/sizing_calc/docs/validation.md)
for the full contract; the short version:

```bash
# Optional one-time: build the predictions grid (faster than live calls).
node ../calculators/sizing_calc/scripts/emit_predictions.mjs

# Plan a run grid based on the calc's b_crit / curve.
exp plan --model-ref llama-3-70b --hw-ref A10G --rows 5

# Pre-flight on by default; strict mode blocks on b_kv<1 / model-unknown.
exp run ... --preflight strict

# Publish runs to the calc's Validation tab.
exp build-portal --calc-bridge ../calculators/sizing_calc/lab_runs/
```

## Per-experiment cycle

```bash
# Bring up cluster
./eks/bringup.sh

# Smoke-test one config locally before paying for a real run
exp run --engine mock --workload chatbot --rate 5 --duration 30

# Real run (any engine)
exp run --engine vllm   --workload chatbot         --rate 8 --duration 300
exp run --engine vllm   --workload agentic_coding  --rate 2 --duration 300
exp run --engine vllm   --workload mix             --rate 6 --duration 300
exp run --engine sglang --workload chatbot         --rate 8 --duration 300
exp run --engine trtllm --workload chatbot         --rate 8 --duration 300

# Build the static portal so you can browse results
exp build-portal --results-dir results --out _site
python -m http.server -d _site 8080  # then open http://localhost:8080/

# Tear it all down
./eks/teardown.sh
```

## Common operations

| Task | Command |
| --- | --- |
| List runs | `exp list` |
| Show a result | `jq . results/runs/<RUN_ID>.json` |
| Verify cluster context | `KUBECONFIG=./eks/kubeconfig kubectl config current-context` |
| See engine pod logs | `kubectl -n engines logs -l app=vllm-engine -f` |
| Snapshot engine metrics | `kubectl -n engines port-forward svc/vllm-engine 8000 & curl :8000/metrics` |
| Cancel an idle GPU node | `kubectl drain <node> --ignore-daemonsets` (Karpenter then reaps) |

## Recovery

| Symptom | Action |
| --- | --- |
| Bringup hangs at "wait Available" | `kubectl describe deploy/<engine>-engine -n engines` — usually GPU scheduling, check the NodePool |
| TRT-LLM build Job runs forever | One-time cost (~20 min for 70B AWQ). Watch with `kubectl logs job/trtllm-build -f`. Artifact is reused on subsequent runs. |
| Spot interruption mid-run | Karpenter replaces the node. Run is marked `failed` in manifest. Re-submit. |
| Cluster left running overnight | `./eks/teardown.sh --yes` — your wallet will thank you |

## Adding a new engine (summary)

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md#adding-a-new-engine). Five
files: `drivers/<engine>_driver.py`, `eks/manifests/engines/<engine>.yaml`,
`tests/contract/test_<engine>_driver.py`, `cli/exp.py` entry, and a
`docs/engines/<engine>.md` page.
