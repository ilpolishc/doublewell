import numpy as np
import pytest

from doublewell.splitstep import Grid1D, gaussian_state
from doublewell.observables import cumulants, cubic_phase_thetas


def test_cumulants_of_gaussian():
    g = Grid1D(n=4096, half_width=40.0)
    psi = gaussian_state(g, xi0=3.0, var=4.0)
    m1, m2, m3 = cumulants(psi, g)
    assert m1 == pytest.approx(3.0, abs=1e-3)
    assert m2 == pytest.approx(4.0, rel=1e-3)
    assert m3 == pytest.approx(0.0, abs=1e-3)


def test_cubic_phase_theta_roundtrip():
    th1, th2, th3 = 71.9, 960.0, -5.4e5
    m1 = -th1
    m2 = th2 + 2.0 * th1 ** 2
    m3 = 2.0 * th3 - 8.0 * th1 ** 3
    assert cubic_phase_thetas(m1, m2, m3) == pytest.approx((th1, th2, th3), rel=1e-9)
