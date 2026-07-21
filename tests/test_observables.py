import numpy as np
import pytest

from doublewell.splitstep import Grid1D, gaussian_state, normalize
from doublewell.observables import covariance, squeezing_eigs


def test_ground_state_covariance():
    g = Grid1D(n=2048, half_width=16.0)
    Vxx, Vpp, Vxp = covariance(gaussian_state(g), g)[2:]
    assert (Vxx, Vpp, Vxp) == pytest.approx((1.0, 1.0, 0.0), abs=1e-8)
    assert squeezing_eigs(gaussian_state(g), g) == pytest.approx((1.0, 1.0), abs=1e-8)


def test_squeezed_vacuum():
    g = Grid1D(n=4096, half_width=24.0)
    s = 4.0
    psi = gaussian_state(g, var=s)
    Vxx, Vpp, Vxp = covariance(psi, g)[2:]
    assert Vxx == pytest.approx(s, rel=1e-4)
    assert Vpp == pytest.approx(1.0 / s, rel=1e-4)
    assert Vxp == pytest.approx(0.0, abs=1e-6)
    lm, lp = squeezing_eigs(psi, g)
    assert (lm, lp) == pytest.approx((1.0 / s, s), rel=1e-4)
    assert lm * lp == pytest.approx(1.0, rel=1e-4)


def test_sheared_state_matches_symplectic():
    g = Grid1D(n=4096, half_width=24.0)
    c = 0.5
    psi = normalize(np.exp(-g.xi ** 2 / 4.0) * np.exp(1j * c * g.xi ** 2 / 2.0), g)
    Vxx, Vpp, Vxp = covariance(psi, g)[2:]
    assert Vxx == pytest.approx(1.0, rel=1e-4)
    assert Vxp == pytest.approx(2.0 * c, rel=1e-3)
    assert Vpp == pytest.approx(1.0 + 4.0 * c ** 2, rel=1e-3)
    lm, lp = squeezing_eigs(psi, g)
    assert lm * lp == pytest.approx(1.0, rel=1e-3)
    expected_lp = 0.5 * ((2.0 + 4.0 * c ** 2) + 4.0 * abs(c) * np.sqrt(c ** 2 + 1.0))
    assert lp == pytest.approx(expected_lp, rel=1e-3)
