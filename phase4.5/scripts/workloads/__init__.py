"""Workload generators for Phase 4.5 Dynamo experiments.

Each workload yields (prompt: str, max_tokens: int) tuples suitable for an
OpenAI-style chat completion. The four workloads isolate different Dynamo
properties:

  W1 (random)         — control. Random short prompts, no shared prefix.
                        Stresses balanced P/D, tests baseline vs router cost.
  W2 (shared_prefix)  — KV-routing's home turf. Fixed 2KB system prompt,
                        variable user message. Cache reuse should be high.
  W3 (bursty_long)    — disaggregation's home turf. 80% W1 + 20% 8K-input
                        bursts. Co-located ITL spikes; disagg should hold.
  W4 (cold_start)     — single request post-pod-start. Measures weight-load
                        time, not steady-state inference.

Generators are deterministic given a seed so re-runs are comparable.
"""
