"""--rate parsing: 'auto' → None, positive floats → float, else BadParameter."""

from __future__ import annotations

import click
import pytest

from experiments.cli.exp import _parse_rate

pytestmark = pytest.mark.unit


class TestParseRate:
    @pytest.mark.parametrize("v", ["auto", "AUTO", " auto ", "Auto"])
    def test_auto_returns_none(self, v: str) -> None:
        assert _parse_rate(v) is None

    @pytest.mark.parametrize(
        ("inp", "expected"),
        [("1", 1.0), ("8", 8.0), ("0.5", 0.5), ("12.5", 12.5), ("1e2", 100.0)],
    )
    def test_positive_floats_pass_through(self, inp: str, expected: float) -> None:
        assert _parse_rate(inp) == pytest.approx(expected)

    @pytest.mark.parametrize("v", ["0", "0.0", "-1", "-0.5"])
    def test_non_positive_rejected(self, v: str) -> None:
        with pytest.raises(click.BadParameter):
            _parse_rate(v)

    @pytest.mark.parametrize("v", ["", "nope", "auto5", "inf", "nan", "5abc"])
    def test_garbage_rejected(self, v: str) -> None:
        with pytest.raises(click.BadParameter):
            _parse_rate(v)
