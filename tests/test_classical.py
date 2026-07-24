import numpy as np
import pytest

from doublewell.units import L
from doublewell.classical import (
    trajectory, turning_point, half_period_quad, energy_u,
    thermal_initial_conditions, evolve_ensemble,
)

U0 = 0.1


def test_turning_point_value():
    assert turning_point(U0) == pytest.approx(1.41069, abs=1e-4)


def test_half_period_ode_vs_quadrature_vs_paper():
    sol = trajectory(U0, tau_max=10.0)
    tau_half_ode = sol.t_events[0][0]
    tau_half_int = half_period_quad(U0)
    assert tau_half_ode == pytest.approx(tau_half_int, rel=1e-6)
    assert tau_half_ode == pytest.approx(L.omega_tm, rel=1e-5)
    assert tau_half_ode == pytest.approx(L.omega_tm_asymptotic, rel=0.02)


def test_trajectory_reaches_turning_point_and_returns():
    sol = trajectory(U0, tau_max=10.0)
    tau_half = sol.t_events[0][0]
    u_at_turn = sol.sol(tau_half)[0]
    assert u_at_turn == pytest.approx(turning_point(U0), abs=1e-6)
    u_back, v_back = sol.sol(2.0 * tau_half)
    assert u_back == pytest.approx(U0, abs=1e-5)
    assert v_back == pytest.approx(0.0, abs=1e-5)


def test_energy_conservation():
    sol = trajectory(U0, tau_max=10.0)
    taus = np.linspace(0.0, 10.0, 500)
    u, v = sol.sol(taus)
    e = energy_u(u, v)
    assert np.max(np.abs(e - e[0])) < 1e-9


def test_ensemble_rk4_matches_rk45():
    tau_end = 5.0
    sol = trajectory(U0, tau_max=tau_end)
    u_ref, v_ref = sol.sol(tau_end)
    u, v = evolve_ensemble(np.array([U0]), np.array([0.0]), tau_end, dtau=1e-3)
    assert u[0] == pytest.approx(u_ref, abs=1e-8)
    assert v[0] == pytest.approx(v_ref, abs=1e-8)


def test_thermal_ensemble_stats():
    rng = np.random.default_rng(1)
    u0, v0 = thermal_initial_conditions(L, 200_000, rng)
    assert np.mean(u0) == pytest.approx(0.1, abs=1e-6)
    assert np.std(u0) == pytest.approx(1e-4, rel=0.02)
    assert np.std(v0) == pytest.approx(1e-2, rel=0.02)
