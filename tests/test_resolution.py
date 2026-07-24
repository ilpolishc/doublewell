import pytest

from doublewell.units import L, TOY_M
from doublewell.observables import required_n, check_resolution, momentum_std_max


def test_delta_pi_max_matches_analytic():
    assert momentum_std_max(L) == pytest.approx(14.14, rel=0.01)
    assert momentum_std_max(TOY_M) == pytest.approx(14.14, rel=0.01)


def test_required_n_and_guard_for_L():
    assert required_n(L, 4500.0) == 131072
    with pytest.raises(ValueError):
        check_resolution(L, 32768, 4500.0)
    check_resolution(L, 131072, 4500.0)


def test_guard_flags_under_resolved_toyM():
    check_resolution(TOY_M, 8192, 450.0)
    with pytest.raises(ValueError):
        check_resolution(TOY_M, 2048, 450.0)
