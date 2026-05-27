"""Tests for the EC2 instance spec lookup table."""

from __future__ import annotations

import pytest

from experiments.drivers.instance_specs import parse_instance

pytestmark = pytest.mark.unit


def test_parse_known_instance_returns_family_and_size() -> None:
    spec = parse_instance("g5.12xlarge")
    assert spec.family == "g5"
    assert spec.size == "12xlarge"
    assert spec.memory_gib == 192
    assert spec.n_gpu == 4


def test_unknown_instance_raises_keyerror_with_actionable_message() -> None:
    with pytest.raises(KeyError, match="instance_specs.py"):
        parse_instance("g99.42xlarge")


def test_allocatable_memory_below_advertised() -> None:
    spec = parse_instance("g5.xlarge")
    # 16 GiB advertised → 80% allocatable cap → 12 GiB.
    assert spec.allocatable_memory_gib == 12


def test_pod_memory_request_leaves_headroom() -> None:
    # g4dn.xlarge: 16 advertised → 12 allocatable → 70% → 8 GiB request.
    # Critical: this MUST fit on the 16 GiB box, which the old hardcoded
    # 32Gi request did not.
    spec = parse_instance("g4dn.xlarge")
    assert spec.pod_memory_request_gib < 16
    assert spec.pod_memory_request_gib >= 2  # floor


def test_small_instance_memory_request_floor() -> None:
    spec = parse_instance("g5.xlarge")  # smallest GPU-bearing g5
    # Sanity: request must be > 0 and well under allocatable.
    assert spec.pod_memory_request_gib >= 2
    assert spec.pod_memory_request_gib <= spec.allocatable_memory_gib
