#!/usr/bin/env python3
"""Run the Phase 4.1 3-way image-pull comparison matrix.

Variants:
    A. docker_hub      — baseline-pod.yaml + default gpu-pool NodePool (stock EKS GPU AMI)
    B. ecr_same_region — baseline-pod-ecr.yaml + default gpu-pool NodePool
    C. prebaked_ami    — baseline-pod-prebaked.yaml + gpu-pool-prebaked NodePool

Per variant: 3 fresh-node runs. Total: 9 cold-start runs. Each tears down the
GPU node beforehand so stages 1-3 (provision, EC2 launch, kubelet bootstrap)
run cleanly.

Pre-flight gates (fail-fast — burn $0 of cluster time if any tripped):
    * cluster context is inference-phase4-1
    * each variant manifest exists and has the expected imagePullPolicy
    * variant C has nodeSelector image-source=prebaked-stock
    * variant C's NodePool exists and has Variant=prebaked-stock AMI selector
    * AZ pinning matches across variants (copied from default NodePool)

Hardware fingerprint check across all 9 runs:
    * AZ, instance_type, CPU model, kernel — must match exactly
    * AMI ID — A and B must match each other; C must NOT match (proves the
      prebaked AMI was used). Mismatch in unexpected directions aborts.

Output:
    phase4.1/tests/pull_comparison_results.json — 9 run records keyed by
    variant + run-id, plus matrix-level metadata (timestamps, fingerprint
    summary, validation pass/fail per gate).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

PHASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RUNS_DIR = PHASE_DIR / "tests" / "baseline_runs"
RESULTS_PATH = PHASE_DIR / "tests" / "pull_comparison_results.json"

EXPECTED_CONTEXT_SUBSTR = "inference-phase4-1"


@dataclass
class Variant:
    name: str        # Stable key for results JSON (snake_case)
    slug: str        # DNS-safe slug used in run_id → pod name (no underscores)
    manifest: pathlib.Path
    expected_pull_policy: str
    requires_prebaked_nodepool: bool
    expected_image_source_label: str
    run_ids: list[str] = field(default_factory=list)


VARIANTS: list[Variant] = [
    Variant(
        name="docker_hub",
        slug="hub",
        # Use the routing-isolated copy (baseline-pod-hub.yaml) instead of the
        # original baseline-pod.yaml — the original lacks the
        # image-source=default-eks nodeSelector and would race the prebaked
        # NodePool nondeterministically.
        manifest=PHASE_DIR / "k8s" / "baseline" / "baseline-pod-hub.yaml",
        expected_pull_policy="Always",
        requires_prebaked_nodepool=False,
        expected_image_source_label="default-eks",
    ),
    Variant(
        name="ecr_same_region",
        slug="ecr",
        manifest=PHASE_DIR / "k8s" / "baseline" / "baseline-pod-ecr.yaml",
        expected_pull_policy="Always",
        requires_prebaked_nodepool=False,
        expected_image_source_label="ecr-same-region",
    ),
    Variant(
        name="prebaked_ami",
        slug="prebaked",
        manifest=PHASE_DIR / "k8s" / "baseline" / "baseline-pod-prebaked.yaml",
        # CRITICAL — Always here would re-pull from Hub and erase the experiment.
        expected_pull_policy="IfNotPresent",
        requires_prebaked_nodepool=True,
        expected_image_source_label="prebaked-ami",
    ),
    Variant(
        # Variant E — RunAI Model Streamer. Same NodePool kind as variant C/D
        # (prebaked) but uses the streamer-augmented AMI + a manifest with no
        # init container. See plan doc for design notes.
        name="prebaked_ami_streamer",
        slug="streamer",
        manifest=PHASE_DIR / "k8s" / "baseline" / "baseline-pod-prebaked-streamer.yaml",
        expected_pull_policy="IfNotPresent",
        requires_prebaked_nodepool=True,
        expected_image_source_label="prebaked-streamer",
    ),
    Variant(
        # Variant F — Prebake + FSR + multi-compiler JIT cache (torch.compile +
        # Triton + FlashInfer + vLLM). Same node hardware as D; differs only in
        # AMI (warmed) + pod env vars + hostPath mount of /opt/vllm-cache.
        name="prebaked_ami_warmed",
        slug="warmed",
        manifest=PHASE_DIR / "k8s" / "baseline" / "baseline-pod-prebaked-warmed.yaml",
        expected_pull_policy="IfNotPresent",
        requires_prebaked_nodepool=True,
        expected_image_source_label="prebaked-warmed",
    ),
]
RUNS_PER_VARIANT = 3


# ─── kubectl helpers ─────────────────────────────────────────────────────────

def kubectl(args: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, check=check)


def current_context() -> str:
    return kubectl(["config", "current-context"]).stdout.strip()


def get_gpu_nodes() -> list[str]:
    raw = kubectl(["get", "nodes", "-l", "node-role=gpu", "-o", "name"]).stdout
    return [line.strip().split("/", 1)[-1] for line in raw.splitlines() if line.strip()]


def teardown_gpu_node() -> None:
    """Reused logic from run_baseline_matrix.py — force-delete and wait."""
    nodes = get_gpu_nodes()
    if not nodes:
        print("  no GPU node to tear down")
        return
    for n in nodes:
        print(f"  deleting node {n}")
        kubectl(["delete", "node", n, "--wait=false"])
    deadline = time.time() + 240
    while time.time() < deadline:
        nc = kubectl(["get", "nodeclaims", "-o", "name"]).stdout.strip()
        if not nc:
            print("  NodeClaim gone — Karpenter terminated the instance")
            return
        time.sleep(5)
    print("  WARNING: NodeClaim still exists after 240s; proceeding anyway")


# ─── Pre-flight gates ────────────────────────────────────────────────────────

def assert_cluster_context() -> None:
    ctx = current_context()
    if EXPECTED_CONTEXT_SUBSTR not in ctx:
        raise SystemExit(
            f"FAIL: kubectl context is {ctx!r}; expected substring "
            f"{EXPECTED_CONTEXT_SUBSTR!r}. Use the phase4.1 kubeconfig."
        )
    print(f"[gate] kubectl context OK: {ctx}")


def assert_manifest_pull_policy(variant: Variant) -> None:
    """Read the rendered manifest and confirm imagePullPolicy + nodeSelector."""
    if not variant.manifest.is_file():
        raise SystemExit(f"FAIL: manifest missing for variant {variant.name}: {variant.manifest}")
    text = variant.manifest.read_text()
    # Cheap text grep — the manifests are simple enough that yaml parsing
    # adds dependencies for no real benefit. If the structure changes we want
    # to fail loudly anyway.
    if f"imagePullPolicy: {variant.expected_pull_policy}" not in text:
        raise SystemExit(
            f"FAIL: variant {variant.name} manifest must have "
            f"imagePullPolicy: {variant.expected_pull_policy}. "
            f"This guard prevents variant C from accidentally re-pulling."
        )
    # The nodeSelector value differs from the pod label for prebaked variants:
    # variant C/D's pod has label image-source=prebaked-ami but selects nodes
    # labeled image-source=prebaked-stock (the NodePool's template label).
    # Variant E selects prebaked-streamer. So this must key off variant.name.
    if variant.name == "prebaked_ami_streamer":
        expected_label = "prebaked-streamer"
    elif variant.name == "prebaked_ami_warmed":
        expected_label = "prebaked-warmed"
    elif variant.requires_prebaked_nodepool:
        expected_label = "prebaked-stock"
    else:
        expected_label = "default-eks"
    expected_selector = f"image-source: {expected_label}"
    if expected_selector not in text:
        raise SystemExit(
            f"FAIL: variant {variant.name} must have nodeSelector "
            f"{expected_selector} — without explicit routing, Karpenter "
            f"schedules nondeterministically across NodePools that share "
            f"node-role=gpu and the data is contaminated."
        )
    print(f"[gate] variant {variant.name}: pull policy + nodeSelector OK")


def assert_prebaked_nodepool_ready(variants: list[Variant] | None = None) -> None:
    """Verify the prebaked NodePool + EC2NodeClass exist and resolve to an AMI.

    Each prebaked-style variant has its own NodePool/EC2NodeClass pair (one
    per AMI). Pass `variants` to gate only on the ones being run; default
    behavior gates on the original prebaked-stock pool.
    """
    pools = []
    if variants is None:
        pools = [("gpu-pool-prebaked", "gpu-nodes-prebaked",
                  "prebaked-stock", "gpu-node-stock.pkr.hcl")]
    else:
        for v in variants:
            if not v.requires_prebaked_nodepool:
                continue
            if v.name == "prebaked_ami_streamer":
                pools.append(("gpu-pool-prebaked-streamer",
                              "gpu-nodes-prebaked-streamer",
                              "prebaked-streamer",
                              "gpu-node-streamer.pkr.hcl"))
            elif v.name == "prebaked_ami_warmed":
                pools.append(("gpu-pool-prebaked-warmed",
                              "gpu-nodes-prebaked-warmed",
                              "prebaked-stock-warmed",
                              "gpu-node-stock-warmed.pkr.hcl"))
            else:
                pools.append(("gpu-pool-prebaked", "gpu-nodes-prebaked",
                              "prebaked-stock", "gpu-node-stock.pkr.hcl"))

    for nodepool, ec2nc_name, variant_tag, packer_tpl in pools:
        nc = kubectl(["get", "nodepool", nodepool, "-o", "json"])
        if nc.returncode != 0:
            raise SystemExit(
                f"FAIL: NodePool {nodepool} is not applied. Apply the "
                f"matching k8s/karpenter/*.yaml first."
            )
        ec2nc = kubectl(["get", "ec2nodeclass", ec2nc_name, "-o", "json"])
        if ec2nc.returncode != 0:
            raise SystemExit(f"FAIL: EC2NodeClass {ec2nc_name} is not applied.")
        data = json.loads(ec2nc.stdout)
        amis = data.get("status", {}).get("amis", [])
        if not amis:
            raise SystemExit(
                f"FAIL: EC2NodeClass {ec2nc_name} has no resolved AMIs. "
                f"The Variant={variant_tag} tag did not match any AMI. Run "
                f"build-ami.sh --template {packer_tpl} first."
            )
        print(f"[gate] {nodepool} resolves to AMI(s): "
              f"{[a.get('id') for a in amis]}")


# ─── Run a single record ─────────────────────────────────────────────────────

def run_single(variant: Variant, run_id: str) -> dict[str, Any]:
    """Invoke run_baseline.sh with MANIFEST_PATH set to this variant's manifest."""
    out = RUNS_DIR / f"{run_id}.json"
    if out.is_file():
        out.unlink()
    print(f"  [variant={variant.name}] run_baseline.sh {run_id}")
    env_arg = f"MANIFEST_PATH={variant.manifest}"
    rc = subprocess.run(
        ["env", env_arg, "bash", str(PHASE_DIR / "scripts" / "run_baseline.sh"), run_id],
        check=False,
    ).returncode
    print(f"  rc={rc}")
    if not out.is_file():
        raise RuntimeError(f"run JSON missing for {run_id} (rc={rc})")
    return json.loads(out.read_text())


# ─── Fingerprint validation ──────────────────────────────────────────────────

def fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    n = record.get("node", {}) or {}
    return {
        "az": n.get("az"),
        "instance_type": n.get("instance_type"),
        "cpu_model": n.get("cpu_model"),
        "kernel": n.get("kernel"),
        "ami_id": n.get("ami_id"),
    }


def validate_fingerprints(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Cross-variant fingerprint validation.

    Rules:
      * AZ, instance_type, CPU model, kernel: must be identical across all 9 runs.
      * AMI ID: A and B must share an AMI ID (default EKS GPU AMI).
                C must have a different AMI ID (the prebaked one). If C's AMI
                matches A/B's, the prebake didn't take effect.
    """
    issues: list[str] = []
    invariant_keys = ["az", "instance_type", "cpu_model", "kernel"]

    all_fps = [
        (variant, run_idx, fingerprint(rec))
        for variant, recs in results.items()
        for run_idx, rec in enumerate(recs)
    ]

    if not all_fps:
        return {"passed": False, "issues": ["no runs to validate"]}

    ref_fp = all_fps[0][2]
    for variant, run_idx, fp in all_fps:
        for k in invariant_keys:
            if fp[k] != ref_fp[k]:
                issues.append(
                    f"{variant} run {run_idx}: {k}={fp[k]!r} differs from reference {ref_fp[k]!r}"
                )

    ab_amis = {fp["ami_id"] for v, _, fp in all_fps if v in ("docker_hub", "ecr_same_region")}
    c_amis = {fp["ami_id"] for v, _, fp in all_fps if v in ("prebaked_ami", "prebaked_ami_fsr")}
    e_amis = {fp["ami_id"] for v, _, fp in all_fps if v == "prebaked_ami_streamer"}
    f_amis = {fp["ami_id"] for v, _, fp in all_fps if v == "prebaked_ami_warmed"}
    if len(ab_amis) > 1:
        issues.append(f"variants A/B have inconsistent AMI IDs: {ab_amis}")
    if len(c_amis) > 1:
        issues.append(f"variants C/D have inconsistent AMI IDs: {c_amis}")
    if len(e_amis) > 1:
        issues.append(f"variant E has inconsistent AMI IDs: {e_amis}")
    if len(f_amis) > 1:
        issues.append(f"variant F has inconsistent AMI IDs: {f_amis}")
    if ab_amis and c_amis and ab_amis == c_amis:
        issues.append(
            f"variant C/D used the same AMI as A/B ({ab_amis}) — the prebake "
            f"did not take effect. NodeSelector likely landed on a default-pool node."
        )
    if e_amis and (ab_amis == e_amis or c_amis == e_amis):
        issues.append(
            f"variant E AMI ({e_amis}) matches another variant — the streamer "
            f"NodePool resolved to the wrong AMI. Check Variant=prebaked-streamer tag."
        )
    if f_amis and (ab_amis == f_amis or c_amis == f_amis or e_amis == f_amis):
        issues.append(
            f"variant F AMI ({f_amis}) matches another variant — the warmed "
            f"NodePool resolved to the wrong AMI. Check Variant=prebaked-stock-warmed tag."
        )

    return {
        "passed": not issues,
        "issues": issues,
        "reference": ref_fp,
        "ab_amis": sorted(ab_amis),
        "c_amis": sorted(c_amis),
        "e_amis": sorted(e_amis),
        "f_amis": sorted(f_amis),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def _parse_variants_arg(args: list[str]) -> tuple[list[Variant], bool]:
    """Honor `--variants name1,name2`; default = all VARIANTS.

    Returns (selected_variants, merge_into_existing). `merge_into_existing` is
    True iff a filter was given AND the results JSON already exists — caller
    should preserve unselected variants' records when writing the new file.
    """
    selected = VARIANTS
    explicit = False
    if "--variants" in args:
        i = args.index("--variants")
        if i + 1 >= len(args):
            raise SystemExit("FAIL: --variants requires a comma-separated argument.")
        wanted = {n.strip() for n in args[i + 1].split(",") if n.strip()}
        unknown = wanted - {v.name for v in VARIANTS}
        if unknown:
            raise SystemExit(
                f"FAIL: unknown variant name(s): {sorted(unknown)}. "
                f"Valid: {[v.name for v in VARIANTS]}"
            )
        selected = [v for v in VARIANTS if v.name in wanted]
        explicit = True
    return selected, explicit and RESULTS_PATH.is_file()


def main() -> int:
    selected_variants, merge = _parse_variants_arg(sys.argv[1:])
    if len(selected_variants) == len(VARIANTS):
        print("=== Phase 4.1: full pull comparison ===\n")
    else:
        print(f"=== Phase 4.1: subset run "
              f"[{', '.join(v.name for v in selected_variants)}] ===\n")
        if merge:
            print(f"  merge mode: preserving non-selected variants in {RESULTS_PATH}\n")

    # Pre-flight
    assert_cluster_context()
    for v in selected_variants:
        assert_manifest_pull_policy(v)
    assert_prebaked_nodepool_ready(selected_variants)
    print("\nAll pre-flight gates passed.\n")

    # Seed results from disk if we're in merge mode; otherwise start clean.
    if merge:
        existing = json.loads(RESULTS_PATH.read_text())
        results: dict[str, list[dict[str, Any]]] = dict(existing.get("results", {}))
        # Variants we're about to re-run get cleared so we don't double-count.
        for v in selected_variants:
            results[v.name] = []
        # Make sure every selected variant is keyed.
        for v in selected_variants:
            results.setdefault(v.name, [])
    else:
        results = {v.name: [] for v in selected_variants}

    for variant in selected_variants:
        print(f"\n=== Variant: {variant.name} ===")
        for i in range(1, RUNS_PER_VARIANT + 1):
            # Use slug not name — pod name is `baseline-<run_id>` and underscores
            # are invalid in k8s DNS-1123 names.
            run_id = f"{variant.slug}-{i:03d}"
            variant.run_ids.append(run_id)
            print(f"\n--- {run_id} ({i}/{RUNS_PER_VARIANT}) ---")
            print("[gate] tearing down GPU node before fresh run")
            teardown_gpu_node()
            try:
                rec = run_single(variant, run_id)
            except Exception as exc:
                print(f"  ERROR in {run_id}: {exc}")
                # Best-effort: keep going with other runs but flag the gap.
                continue
            results[variant.name].append(rec)
            fp = fingerprint(rec)
            print(f"  fingerprint: {fp}")

    fingerprint_validation = validate_fingerprints(results)
    print("\n=== Fingerprint validation ===")
    print(json.dumps(fingerprint_validation, indent=2))

    # When merging, preserve the existing variants block for variants we
    # didn't touch; only update the selected ones' run_ids.
    if merge:
        merged_variants = dict(existing.get("variants", {}))
        for v in selected_variants:
            merged_variants[v.name] = {"manifest": str(v.manifest), "run_ids": v.run_ids}
        variants_block = merged_variants
    else:
        variants_block = {v.name: {"manifest": str(v.manifest), "run_ids": v.run_ids} for v in selected_variants}

    matrix_record = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs_per_variant": RUNS_PER_VARIANT,
        "variants": variants_block,
        "results": results,
        "fingerprint_validation": fingerprint_validation,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(matrix_record, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    if not fingerprint_validation["passed"]:
        print("\nWARNING: fingerprint validation reported issues; "
              "results may not be cleanly attributable.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
