"""AWS EC2 instance specs needed by the K8s driver to size pod resources.

We can't read EC2 metadata at template-render time (we're outside the
cluster), so this table captures the bits the Karpenter node-affinity
selectors need: per-instance memory (for right-sizing pod requests) and
the family/size split for the ``karpenter.k8s.aws/instance-{family,size}``
selectors.

Memory values are NodeAllocatable estimates: we shave ~20% off the EC2
spec for kubelet + system reservations and round down to a Gi integer.
Pod requests should target ~70% of this to leave headroom for kube-proxy
and CNI without OOM-killing the engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstanceSpec:
    """Subset of EC2 instance attributes the K8s driver cares about."""

    family: str            # e.g. "g5"
    size: str              # e.g. "12xlarge"
    memory_gib: int        # advertised EC2 memory, GiB
    n_gpu: int             # GPUs on the box (informational; tp dictates the request)

    @property
    def allocatable_memory_gib(self) -> int:
        """Conservative NodeAllocatable estimate (~80% of advertised)."""
        # Kubelet reserves ~25% on small nodes, ~10% on huge ones — split
        # the difference at 20% which is safe across the GPU SKU range.
        return max(1, int(self.memory_gib * 0.8))

    @property
    def pod_memory_request_gib(self) -> int:
        """What to put in ``resources.requests.memory`` for a single-pod node.

        Target ~70% of allocatable so kube-proxy, DCGM, log shippers, and
        the CNI all have room. Below 4 GiB we floor at 2 GiB — vLLM won't
        start in less.
        """
        # 0.7 × allocatable is the usable slice; floor at 2 GiB for safety.
        raw = int(self.allocatable_memory_gib * 0.7)
        return max(2, raw)


# Hand-curated table of GPU instance types we actually run. Keep this
# narrow; add rows as new families come online rather than trying to
# enumerate every EC2 SKU. The "size" column matches Karpenter's
# ``karpenter.k8s.aws/instance-size`` label exactly.
_SPECS: dict[str, InstanceSpec] = {
    # g4dn (T4): 1× T4 except .12xlarge (4×) and .metal (8×).
    "g4dn.xlarge":    InstanceSpec("g4dn", "xlarge",     16,  1),
    "g4dn.2xlarge":   InstanceSpec("g4dn", "2xlarge",    32,  1),
    "g4dn.4xlarge":   InstanceSpec("g4dn", "4xlarge",    64,  1),
    "g4dn.8xlarge":   InstanceSpec("g4dn", "8xlarge",   128,  1),
    "g4dn.12xlarge":  InstanceSpec("g4dn", "12xlarge",  192,  4),
    "g4dn.16xlarge":  InstanceSpec("g4dn", "16xlarge",  256,  1),
    "g4dn.metal":     InstanceSpec("g4dn", "metal",     384,  8),

    # g5 (A10G): same shape as g4dn but with A10G.
    "g5.xlarge":      InstanceSpec("g5", "xlarge",       16,  1),
    "g5.2xlarge":     InstanceSpec("g5", "2xlarge",      32,  1),
    "g5.4xlarge":     InstanceSpec("g5", "4xlarge",      64,  1),
    "g5.8xlarge":     InstanceSpec("g5", "8xlarge",     128,  1),
    "g5.12xlarge":    InstanceSpec("g5", "12xlarge",    192,  4),
    "g5.16xlarge":    InstanceSpec("g5", "16xlarge",    256,  1),
    "g5.24xlarge":    InstanceSpec("g5", "24xlarge",    384,  4),
    "g5.48xlarge":    InstanceSpec("g5", "48xlarge",    768,  8),

    # g6 (L4) — same memory tiers as g5.
    "g6.xlarge":      InstanceSpec("g6", "xlarge",       16,  1),
    "g6.2xlarge":     InstanceSpec("g6", "2xlarge",      32,  1),
    "g6.4xlarge":     InstanceSpec("g6", "4xlarge",      64,  1),
    "g6.8xlarge":     InstanceSpec("g6", "8xlarge",     128,  1),
    "g6.12xlarge":    InstanceSpec("g6", "12xlarge",    192,  4),
    "g6.16xlarge":    InstanceSpec("g6", "16xlarge",    256,  1),
    "g6.24xlarge":    InstanceSpec("g6", "24xlarge",    384,  4),
    "g6.48xlarge":    InstanceSpec("g6", "48xlarge",    768,  8),

    # P-family (V100 / A100 / H100). Full-node SKUs only — partial
    # provisioning isn't supported on these families.
    "p3.2xlarge":     InstanceSpec("p3",  "2xlarge",     61,  1),
    "p3.8xlarge":     InstanceSpec("p3",  "8xlarge",    244,  4),
    "p3.16xlarge":    InstanceSpec("p3",  "16xlarge",   488,  8),
    "p4d.24xlarge":   InstanceSpec("p4d", "24xlarge",  1152,  8),
    "p4de.24xlarge":  InstanceSpec("p4de","24xlarge",  1152,  8),
    "p5.48xlarge":    InstanceSpec("p5",  "48xlarge",  2048,  8),
}


def parse_instance(instance: str) -> InstanceSpec:
    """Look up an EC2 instance by its API name (e.g. ``g5.12xlarge``).

    Raises ``KeyError`` if the instance isn't in the curated table —
    callers should add a row rather than guess (a wrong memory estimate
    silently misallocates pod resources).
    """
    spec = _SPECS.get(instance)
    if spec is None:
        raise KeyError(
            f"unknown instance type {instance!r}; add it to "
            "experiments/drivers/instance_specs.py"
        )
    return spec
