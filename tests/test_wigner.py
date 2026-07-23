import numpy as np
import pytest

from doublewell.splitstep import Grid1D, gaussian_state, normalize
from doublewell.wigner import wigner


def test_ground_state_positive_with_correct_marginal():
    g = Grid1D(n=512, half_width=16.0)
    phi = gaussian_state(g)
    xo, pi, W = wigner(phi, g.dxi, xi=g.xi, center=0.0, half=12.0,
                       nx=121, pi_max=8.0, n_pi=121)
    assert W.min() > -1e-9
    marg = W.sum(axis=1) * (pi[1] - pi[0])
    ref = np.interp(xo, g.xi, np.abs(phi) ** 2)
    assert np.allclose(marg, ref, atol=5e-3)
    assert abs(xo[W.max(axis=1).argmax()]) < 0.3
    assert abs(pi[W.max(axis=0).argmax()]) < 0.3


def test_momentum_axis_sign():
    g = Grid1D(n=512, half_width=16.0)
    psi = normalize(np.exp(-g.xi ** 2 / 4) * np.exp(1j * 2.0 * g.xi), g)
    xo, pi, W = wigner(psi, g.dxi, xi=g.xi, center=0.0, half=12.0, pi_max=10.0, n_pi=201)
    assert pi[np.unravel_index(W.argmax(), W.shape)[1]] == pytest.approx(4.0, abs=0.3)
    sh = normalize(np.exp(-g.xi ** 2 / 4) * np.exp(1j * 0.25 * g.xi ** 2), g)
    xo, pi, Ws = wigner(sh, g.dxi, xi=g.xi, center=0.0, half=10.0, pi_max=10.0, n_pi=201)
    j = np.argmin(np.abs(xo - 4.0))
    assert pi[Ws[j].argmax()] > 1.0


def test_cat_state_has_negativity():
    g = Grid1D(n=512, half_width=16.0)
    cat = normalize(np.exp(-(g.xi - 5.0) ** 2 / 4) + np.exp(-(g.xi + 5.0) ** 2 / 4), g)
    xo, pi, W = wigner(cat, g.dxi, xi=g.xi, center=0.0, half=12.0,
                       nx=161, pi_max=8.0, n_pi=161)
    assert W.min() < -0.05
    assert W[np.argmin(np.abs(xo))].min() < -0.05
