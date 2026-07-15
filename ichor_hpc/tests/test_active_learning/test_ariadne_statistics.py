"""Order-statistic contracts used by ARIADNE stop diagnostics."""
import pytest

from ichor.hpc.active_learning.daemon.live_executor import _high_quantile


def test_high_quantile_returns_nearest_rank_observed_value():
    assert _high_quantile([0.0, 10.0], 0.90) == 10.0
    assert _high_quantile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_high_quantile_rejects_nonfinite_inputs(value):
    with pytest.raises(ValueError, match="must be finite"):
        _high_quantile([1.0, value], 0.90)
