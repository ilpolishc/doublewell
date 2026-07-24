import numpy as np
import pytest

from doublewell.units import L, TOY_S, v_prep
from doublewell.splitstep import (
    Grid1D, gaussian_state, propagate, norm, expect_xi, var_xi, expect_k, energy,
)


def test_L_set_derived_quantities_match_paper():
    assert L.eta == pytest.approx(707.1068, rel=1e-4)
    assert L.S0_dB == pytest.approx(56.99, abs=0.01)
    assert L.omega_tm_asymptotic == pytest.approx(4.0355, abs=1e-3)
    assert L.omega_tm == pytest.approx(4.0469, abs=1e-3)
    assert L.xi_s == pytest.approx(1000.0)
    assert L.tau_m == pytest.approx(404.69, abs=0.1)


def test_toy_s_is_actually_small():
    assert TOY_S.xi_s == pytest.approx(10.0)
    assert TOY_S.tau_m == pytest.approx(40.469, abs=0.01)


def test_double_well_shape():
    xi_min = TOY_S.D_over_XOm
    depth = -0.125 * TOY_S.omega_over_Omega ** 2 * TOY_S.D_over_XOm ** 2
    assert TOY_S.v_dw(np.array([xi_min]))[0] == pytest.approx(depth)
    assert TOY_S.v_dw(np.array([0.0]))[0] == 0.0
    eps = 1e-4
    slope = (TOY_S.v_dw(np.array([xi_min + eps]))[0]
             - TOY_S.v_dw(np.array([xi_min - eps]))[0]) / (2 * eps)
    assert slope == pytest.approx(0.0, abs=1e-8)


@pytest.fixture
def grid():
    return Grid1D(n=2048, half_width=16.0)


def test_ground_state_variance(grid):
    psi = gaussian_state(grid)
    assert norm(psi, grid) == pytest.approx(1.0, abs=1e-12)
    assert var_xi(psi, grid) == pytest.approx(1.0, abs=1e-10)
    # <(P/P_Om)^2> = 4 <k^2> = 1 for the ground state (zero-point momentum)
    assert 4.0 * expect_k(psi, grid, power=2) == pytest.approx(1.0, abs=1e-8)
    # ground-state energy of v = xi^2/4 is 1/2 (in hbar*Omega units)
    assert energy(psi, grid, v_prep(grid.xi)) == pytest.approx(0.5, abs=1e-8)


def test_ground_state_stationary(grid):
    v = v_prep(grid.xi)
    psi0 = gaussian_state(grid)
    dtau = 1e-3
    nsteps = int(round(2.0 * np.pi / dtau))
    psi = propagate(psi0, grid, v, dtau, nsteps)
    assert norm(psi, grid) == pytest.approx(1.0, abs=1e-10)
    assert var_xi(psi, grid) == pytest.approx(1.0, abs=1e-6)
    fidelity = np.abs(np.sum(np.conj(psi0) * psi) * grid.dxi)
    assert fidelity == pytest.approx(1.0, abs=1e-6)
    assert energy(psi, grid, v) == pytest.approx(0.5, abs=1e-6)


def test_coherent_state_oscillates(grid):
    xi0 = 2.0
    v = v_prep(grid.xi)
    psi = gaussian_state(grid, xi0=xi0)
    dtau = 1e-3
    means, taus = [], []

    def record(p, step):
        means.append(expect_xi(p, grid))
        taus.append(step * dtau)

    nsteps = int(round(2.0 * np.pi / dtau))
    psi = propagate(psi, grid, v, dtau, nsteps, callback=record, callback_every=500)
    expected = xi0 * np.cos(np.array(taus))
    assert np.max(np.abs(np.array(means) - expected)) < 1e-4
    assert var_xi(psi, grid) == pytest.approx(1.0, abs=1e-6)
