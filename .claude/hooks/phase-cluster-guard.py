#!/usr/bin/env python3
"""PreToolUse Bash guard: block cluster commands that target the wrong cluster
for the current phase.

Walks up from the command's cwd looking for a `.phase` marker file. If found,
reads the sibling `.phase-cluster` file for the expected cluster name and
verifies that any kubectl/helm/eksctl/aws-eks invocation in the command will
hit *that* cluster — by checking the command's --context/--cluster flags and
the KUBECONFIG env. Mismatch -> exit 2 (blocks the tool call).

If no `.phase` marker is found above cwd, the hook is silent (cross-phase
work from repo root is allowed).
"""
import json
import os
import re
import sys

# Commands that touch a cluster. Keep tight — false positives are annoying.
TRIGGER = re.compile(
    r"(?:^|[\s;&|`(])(kubectl|helm|eksctl|karpenterctl)\b"
    r"|(?:^|[\s;&|`(])aws\s+eks\b"
)


def find_phase_dir(start: str):
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, ".phase")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def read_first_line(path: str):
    try:
        with open(path) as f:
            return f.readline().strip()
    except OSError:
        return None


def cluster_from_flags(cmd: str):
    """Pull the cluster/context name out of a kubectl/helm/aws/eksctl command."""
    # --context=ARN or --context ARN  (kubectl)
    # --kube-context=ARN or --kube-context ARN  (helm)
    # \b ensures --kube-context is matched as a unit, not as `--context` inside it.
    m = re.search(r"(?:^|\s)--(?:kube-)?context[=\s]+(\S+)", cmd)
    if m:
        return m.group(1).rsplit("/", 1)[-1]
    # --cluster=NAME or --cluster NAME (eksctl, aws eks)
    m = re.search(r"--cluster(?:-name)?[=\s]+(\S+)", cmd)
    if m:
        return m.group(1).rsplit("/", 1)[-1]
    # aws eks ... --name NAME
    m = re.search(r"aws\s+eks\s+\S+.*?--name[=\s]+(\S+)", cmd)
    if m:
        return m.group(1)
    return None


def block(reason: str):
    print(f"[phase-cluster-guard] BLOCKED: {reason}", file=sys.stderr)
    print(
        "  Fix: cd into the matching phase dir (so .envrc sets KUBECONFIG), "
        "or pass --context/--cluster pointing at the expected cluster.",
        file=sys.stderr,
    )
    sys.exit(2)


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if data.get("tool_name") != "Bash":
        sys.exit(0)

    cmd = (data.get("tool_input") or {}).get("command", "") or ""
    if not TRIGGER.search(cmd):
        sys.exit(0)

    cwd = data.get("cwd") or os.getcwd()
    phase_dir = find_phase_dir(cwd)
    if not phase_dir:
        sys.exit(0)  # not under any phase — allow

    phase = read_first_line(os.path.join(phase_dir, ".phase")) or "?"
    expected = read_first_line(os.path.join(phase_dir, ".phase-cluster"))

    # If phase has no cluster mapping yet, block any cluster command from inside it.
    if not expected:
        block(
            f"phase '{phase}' has no .phase-cluster mapping yet — "
            f"create the cluster, then write its name to {phase_dir}/.phase-cluster"
        )

    # 1) Explicit flag in the command wins.
    flag_cluster = cluster_from_flags(cmd)
    if flag_cluster:
        if flag_cluster != expected:
            block(
                f"phase '{phase}' expects cluster '{expected}', "
                f"command targets '{flag_cluster}'"
            )
        sys.exit(0)

    # 2) No explicit flag -> KUBECONFIG must point at the phase's kubeconfig
    #    (which contains only the expected cluster). This is the case for
    #    plain `kubectl get pods` etc.
    expected_kc = os.path.join(phase_dir, ".kubeconfig")
    if "kubectl" in cmd or "helm" in cmd:
        actual_kc = os.environ.get("KUBECONFIG", "")
        if actual_kc != expected_kc:
            block(
                f"phase '{phase}' requires KUBECONFIG={expected_kc} "
                f"(got '{actual_kc or '~/.kube/config (default)'}'). "
                f"Run `source {phase_dir}/.envrc` or install direnv."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
