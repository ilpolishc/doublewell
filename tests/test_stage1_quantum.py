import numpy as np
import pytest

from doublewell.units import TOY_S
from doublewell.splitstep import (
    Grid1D, gaussian_state, propagate, propagate_comoving, residual_potential,
    norm, var_xi, expect_xi, energy,
)
from doublewell.classical import comoving_frame

DTAU = 1e-3


def test_residual_matches_vdw_taylor():
    p = TOY_S
    xt = np.linspace(-5.0, 5.0, 21)
    eps = 1e-5
    for xi_c in (0.0, 12.0, 37.0, 141.0):
        vc = p.v_dw(np.array([xi_c]))
        vprime = (p.v_dw(np.array([xi_c + eps])) - p.v_dw(np.array([xi_c - eps]))) / (2 * eps)
        w_def = p.v_dw(xi_c + xt) - vc - vprime * xt
        assert np.allclose(w_def, residual_potential(p, xt, xi_c), atol=1e-10, rtol=1e-6)


def test_comoving_velocity_relation():
    tau = np.array([5.0, 15.0, 25.0, 35.0])
    d = 1e-4
    xi_p, _ = comoving_frame(TOY_S, tau + d)
    xi_m, _ = comoving_frame(TOY_S, tau - d)
    _, k_c = comoving_frame(TOY_S, tau)
    assert np.allclose((xi_p - xi_m) / (2 * d), 2.0 * k_c, rtol=1e-5)


def test_labframe_conservation_toys():
    grid = Grid1D(n=4096, half_width=200.0, xi_center=50.0)
    psi0 = gaussian_state(grid, xi0=TOY_S.xi_s)
    v = TOY_S.v_dw(grid.xi)
    e0 = energy(psi0, grid, v)
    psi = propagate(psi0, grid, v, DTAU, 15000)
    assert norm(psi, grid) == pytest.approx(1.0, abs=1e-9)
    assert energy(psi, grid, v) - e0 == pytest.approx(0.0, abs=1e-4)


def test_lab_comoving_agree_toys():
    tau_end, n = 15.0, 15000
    lab = Grid1D(n=4096, half_width=200.0, xi_center=50.0)
    com = Grid1D(n=4096, half_width=200.0)
    psi = propagate(gaussian_state(lab, xi0=TOY_S.xi_s), lab,
                    TOY_S.v_dw(lab.xi), DTAU, n)
    phi = propagate_comoving(gaussian_state(com, xi0=0.0), com, TOY_S, tau_end, n)
    xi_c = float(comoving_frame(TOY_S, tau_end)[0][0])

    assert norm(phi, com) == pytest.approx(1.0, abs=1e-9)
    assert expect_xi(psi, lab) == pytest.approx(xi_c + expect_xi(phi, com), abs=0.02)
    dX_lab, dX_com = np.sqrt(var_xi(psi, lab)), np.sqrt(var_xi(phi, com))
    assert dX_lab == pytest.approx(dX_com, rel=5e-3)
    assert dX_lab > 15.0
