from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp

from doublewell.units import ParamSet


def potential_u(u):
    return -0.5 * u ** 2 + 0.25 * u ** 4


def energy_u(u, v):
    return 0.5 * v ** 2 + potential_u(u)


def turning_point(u0: float, v0: float = 0.0) -> float:
    e = energy_u(u0, v0)
    return float(np.sqrt(1.0 + np.sqrt(1.0 + 4.0 * e)))


def half_period_quad(u0: float) -> float:
    e = energy_u(u0, 0.0)
    u1 = turning_point(u0)

    def integrand(u):
        return 1.0 / np.sqrt(2.0 * (e - potential_u(u)))

    val, err = quad(integrand, u0, u1, limit=200)
    return float(val)


def trajectory(u0: float, v0: float = 0.0, tau_max: float = 10.0,
               rtol: float = 1e-11, atol: float = 1e-12):
    def rhs(tau, y):
        u, v = y
        return [v, u - u ** 3]

    def turning(tau, y):
        return y[1]

    turning.direction = -1
    return solve_ivp(rhs, (0.0, tau_max), [u0, v0], rtol=rtol, atol=atol,
                     dense_output=True, events=turning)


def thermal_initial_conditions(params: ParamSet, n_traj: int,
                               rng: np.random.Generator,
                               sigma_s_over_XOm: float = 0.0):
    s = np.sqrt(params.thermal_variance())
    pos_std = np.hypot(s, sigma_s_over_XOm) / params.D_over_XOm
    vel_std = s / (params.omega_over_Omega * params.D_over_XOm)
    u0 = params.xs_over_D + pos_std * rng.standard_normal(n_traj)
    v0 = vel_std * rng.standard_normal(n_traj)
    return u0, v0


def comoving_frame(params: ParamSet, tau):
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    tau_w = params.omega_over_Omega * tau
    sol = trajectory(params.xs_over_D, tau_max=float(tau_w.max()) * 1.001 + 1e-6)
    u, up = sol.sol(tau_w)
    xi_c = params.D_over_XOm * u
    k_c = 0.5 * params.D_over_XOm * params.omega_over_Omega * up
    return xi_c, k_c


def evolve_ensemble(u0: np.ndarray, v0: np.ndarray, tau_end: float,
                    dtau: float = 1e-3):
    def force(u):
        return u - u ** 3

    n = int(round(tau_end / dtau))
    dtau = tau_end / n
    u, v = u0.astype(float).copy(), v0.astype(float).copy()
    for _ in range(n):
        k1u, k1v = v, force(u)
        k2u, k2v = v + 0.5 * dtau * k1v, force(u + 0.5 * dtau * k1u)
        k3u, k3v = v + 0.5 * dtau * k2v, force(u + 0.5 * dtau * k2u)
        k4u, k4v = v + dtau * k3v, force(u + dtau * k3u)
        u += (dtau / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v += (dtau / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return u, v
