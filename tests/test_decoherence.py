import numpy as np
import pytest
from scipy.integrate import solve_ivp

from doublewell.units import L, TOY_M, ParamSet
from doublewell.splitstep import Grid1D, gaussian_state
from doublewell.observables import c2_along_orbit, transfer_matrix
from doublewell.decoherence import (
    PI_DIFFUSION_PER_GAMMA, pi_diffusion_rate, propagate_comoving_kicks, ensemble_density,
    blur_std, blur_density, shift_density, fringe_contrast_ratio,
    visibility_bins, visibility_density, visibility_counts, visibility_prl,
)


FREE = ParamSet(name="free", xs_over_D=0.1, omega_over_Omega=0.0, D_over_XOm=1e4)


def _momentum_density(phi, grid):
    p = np.mean(np.abs(np.fft.fft(np.atleast_2d(phi), axis=-1)) ** 2, axis=0)
    return np.fft.fftshift(2.0 * grid.k), np.fft.fftshift(p) / p.sum()


def _chi(pi, p, y):
    return complex(np.sum(p * np.exp(1j * pi * y)))


# ---- the two rate anchors ----------------------------------------------------


def test_coherence_decay_matches_master_equation():
    g = Grid1D(n=512, half_width=32.0)
    d, tau_end, nsteps, ntraj = 3.0, 1.0, 200, 4000
    gamma_over_Omega = 0.039

    cat = np.exp(-((g.xi - d) ** 2) / 4.0) + np.exp(-((g.xi + d) ** 2) / 4.0)
    cat = cat / np.sqrt(np.sum(np.abs(cat) ** 2) * g.dxi)
    batch = np.tile(cat, (ntraj, 1))

    pi0, p0 = _momentum_density(cat, g)
    out = propagate_comoving_kicks(batch, g, FREE, tau_end, nsteps, gamma_over_Omega,
                                   np.random.default_rng(7))
    pi1, p1 = _momentum_density(out, g)

    delta_xi = 2.0 * d
    ratio = abs(_chi(pi1, p1, 0.5 * delta_xi)) / abs(_chi(pi0, p0, 0.5 * delta_xi))
    expected = np.exp(-(gamma_over_Omega / 2.0) * delta_xi ** 2 * tau_end)
    assert ratio == pytest.approx(expected, rel=0.06)
    assert 0.2 < expected < 0.8


def test_momentum_variance_growth_rate():
    g = Grid1D(n=512, half_width=32.0)
    tau_end, nsteps, ntraj, gamma_over_Omega = 1.0, 200, 4000, 1.0
    batch = np.tile(gaussian_state(g, xi0=0.0), (ntraj, 1))
    out = propagate_comoving_kicks(batch, g, FREE, tau_end, nsteps, gamma_over_Omega,
                                   np.random.default_rng(3))
    pi, p = _momentum_density(out, g)
    var = float(np.sum(pi ** 2 * p) - np.sum(pi * p) ** 2)
    assert (var - 1.0) / tau_end == pytest.approx(
        pi_diffusion_rate(gamma_over_Omega), rel=0.05)


def test_kick_grouping_is_harmless():

    g = Grid1D(n=2048, half_width=100.0)
    nsteps, ntraj = 1000, 64
    noise = np.random.default_rng(13).standard_normal((nsteps, ntraj))
    batch = np.tile(gaussian_state(g, xi0=0.0), (ntraj, 1))
    dens = [ensemble_density(
        propagate_comoving_kicks(batch, g, TOY_M, 5.0, nsteps, 1e-3, kick_every=every,
                                 noise=noise), g.dxi)
        for every in (1, 10)]
    assert np.abs(dens[1] - dens[0]).max() < 1e-4 * dens[0].max()


def test_per_row_gamma_matches_separate_runs():

    g = Grid1D(n=512, half_width=32.0)
    nsteps, gammas = 200, np.array([0.2, 2.0])
    noise = np.random.default_rng(17).standard_normal((nsteps, 2))
    psi = gaussian_state(g, xi0=0.0)
    both = propagate_comoving_kicks(np.tile(psi, (2, 1)), g, FREE, 1.0, nsteps, gammas,
                                    noise=noise)
    for i, gam in enumerate(gammas):
        one = propagate_comoving_kicks(psi, g, FREE, 1.0, nsteps, gam,
                                       noise=noise[:, i])
        assert np.allclose(both[i], one[0])


# ---- Route B ------------------------------------------------------------------


def test_transfer_matrix_symplectic_and_turning_point():
    tau, M = transfer_matrix(L, L.tau_m)
    assert np.abs(np.linalg.det(M) - 1.0).max() < 1e-7
    assert M[0] == pytest.approx(np.eye(2), abs=1e-12)
    # t_m is a classical turning point: the phase mode has zero position response there
    assert M[-1, 0, 1] == pytest.approx(0.0, abs=1e-6)
    assert M[-1, 0, 0] == pytest.approx(-0.070888, rel=1e-4)


@pytest.mark.parametrize("params", [L, TOY_M], ids=["L", "toyM"])
def test_blur_std_matches_covariance_ode(params):

    gamma = 1e-8
    c2 = c2_along_orbit(params, params.tau_m)
    R = pi_diffusion_rate(gamma)

    def rhs(tau, y):
        Sxx, Sxp, Spp = y
        c = c2(tau)
        return [2.0 * Sxp, Spp - 4.0 * c * Sxx, -8.0 * c * Sxp + R]

    y = solve_ivp(rhs, (0.0, params.tau_m), [0.0, 0.0, 0.0], rtol=1e-10, atol=1e-18).y[:, -1]
    assert np.sqrt(y[0]) == pytest.approx(blur_std(params, gamma), rel=1e-4)


def test_blur_std_sqrt_scaling():
    s1, s4 = blur_std(L, 1e-8), blur_std(L, 4e-8)
    assert s4 / s1 == pytest.approx(2.0, rel=1e-6)
    assert s1 == pytest.approx(0.1149, rel=1e-3)


def test_blur_density_adds_variance():
    g = Grid1D(n=4096, half_width=60.0)
    P = np.exp(-g.xi ** 2 / 8.0)
    P = P / (P.sum() * g.dxi)
    sigma = 1.5
    Q = blur_density(P, g.dxi, sigma)
    var = lambda p: float(np.sum(g.xi ** 2 * p) / np.sum(p) - (np.sum(g.xi * p) / np.sum(p)) ** 2)
    assert var(Q) - var(P) == pytest.approx(sigma ** 2, rel=1e-4)
    assert np.sum(Q) * g.dxi == pytest.approx(1.0, rel=1e-12)


def test_blur_density_damps_fringes_by_gaussian_factor():
    g = Grid1D(n=8192, half_width=120.0)
    k0, v, sigma = 2.0, 0.8, 0.4
    P = np.exp(-g.xi ** 2 / 800.0) * (1.0 + v * np.cos(k0 * g.xi))
    P = P / (P.sum() * g.dxi)
    Q = blur_density(P, g.dxi, sigma)

    amp = lambda p: 2.0 * abs(np.sum(p * np.exp(-1j * k0 * g.xi))) / np.sum(p * np.exp(0j))
    assert amp(Q) / amp(P) == pytest.approx(np.exp(-0.5 * (k0 * sigma) ** 2), rel=1e-3)


def test_blur_density_refuses_to_wrap():
    g = Grid1D(n=256, half_width=8.0)
    P = np.exp(-g.xi ** 2 / 200.0)
    with pytest.raises(ValueError, match="grid edge"):
        blur_density(P, g.dxi, 1.0)


def test_shift_density_matches_explicit_average():
    g = Grid1D(n=4096, half_width=60.0)
    P = np.exp(-g.xi ** 2 / 8.0) * (1.0 + 0.5 * np.cos(3.0 * g.xi))
    P = P / (P.sum() * g.dxi)
    shifts = np.array([-1.3, -0.2, 0.7, 2.1])
    ref = np.mean([np.interp(g.xi - s, g.xi, P) for s in shifts], axis=0)
    ref = ref / (ref.sum() * g.dxi)
    assert np.allclose(shift_density(P, g.dxi, shifts), ref, atol=2e-4 * P.max())


def test_shift_density_gaussian_shifts_reproduce_blur():
    g = Grid1D(n=8192, half_width=120.0)
    P = np.exp(-g.xi ** 2 / 800.0) * (1.0 + 0.8 * np.cos(2.0 * g.xi))
    P = P / (P.sum() * g.dxi)
    sigma = 0.5
    shifts = sigma * np.random.default_rng(0).standard_normal(500_000)
    assert np.allclose(shift_density(P, g.dxi, shifts), blur_density(P, g.dxi, sigma),
                       atol=0.01 * P.max())


def test_shift_density_rejects_shifts_that_would_wrap():
    g = Grid1D(n=1024, half_width=20.0)
    P = np.exp(-g.xi ** 2 / 8.0)
    with pytest.raises(ValueError, match="wrap"):
        shift_density(P, g.dxi, np.array([0.0, 15.0]))


# ---- Appendix-C visibility ----------------------------------------------------


def _fringe_pattern(g, v, k0=2.0):
    P = np.exp(-g.xi ** 2 / 900.0) * (1.0 + v * np.cos(k0 * g.xi))
    return P / (P.sum() * g.dxi)


def test_visibility_bins_land_on_second_max_and_first_min():
    g = Grid1D(n=16384, half_width=150.0)
    k0 = 2.0
    bins = visibility_bins(_fringe_pattern(g, 0.8, k0), g.xi)
    assert abs(bins["y_max"]) == pytest.approx(2.0 * np.pi / k0, abs=2 * g.dxi)
    assert abs(bins["y_min"]) == pytest.approx(np.pi / k0, abs=2 * g.dxi)
    assert bins["delta"] == pytest.approx(np.pi / k0, abs=2 * g.dxi)


def test_visibility_density_is_two_over_pi_times_contrast():

    g = Grid1D(n=16384, half_width=150.0)
    for v in (0.3, 0.6, 0.9):
        P = _fringe_pattern(g, v)
        assert visibility_density(P, g.xi, visibility_bins(P, g.xi)) == pytest.approx(
            2.0 * v / np.pi, rel=0.02)


def test_visibility_tracks_the_gaussian_damping_law():
    g = Grid1D(n=16384, half_width=150.0)
    k0, v = 2.0, 0.9
    P = _fringe_pattern(g, v, k0)
    bins = visibility_bins(P, g.xi)
    for sigma in (0.2, 0.5, 0.9):
        Q = blur_density(P, g.dxi, sigma)
        assert visibility_density(Q, g.xi, bins) == pytest.approx(
            2.0 * v * np.exp(-0.5 * (k0 * sigma) ** 2) / np.pi, rel=0.03)


def test_visibility_counts_agrees_with_density():
    g = Grid1D(n=16384, half_width=150.0)
    P = _fringe_pattern(g, 0.8)
    bins = visibility_bins(P, g.xi)
    rng = np.random.default_rng(5)
    n = 2_000_000
    Y = rng.choice(g.xi, size=n, p=P / P.sum()) + rng.uniform(-g.dxi / 2, g.dxi / 2, n)
    v_counts = visibility_counts(Y, bins)[0]
    assert v_counts == pytest.approx(visibility_density(P, g.xi, bins), abs=0.005)


def test_fringe_contrast_ratio_ignores_translation_and_follows_the_blur_law():

    g = Grid1D(n=8192, half_width=120.0)
    k0, v = 2.0, 0.8
    env = lambda x: np.exp(-x ** 2 / 900.0) * (1.0 + v * np.cos(k0 * x))
    P = env(g.xi) / (env(g.xi).sum() * g.dxi)
    Q = env(g.xi - 7.3) / (env(g.xi - 7.3).sum() * g.dxi)
    assert fringe_contrast_ratio(Q, g.xi, P, g.xi, k0) == pytest.approx(1.0, abs=1e-6)
    for sigma in (0.2, 0.5, 0.9):
        assert fringe_contrast_ratio(blur_density(P, g.dxi, sigma), g.xi, P, g.xi,
                                     k0) == pytest.approx(np.exp(-0.5 * (k0 * sigma) ** 2),
                                                          rel=1e-4)


def test_prl_visibility_is_the_undiluted_modulation_depth():

    g = Grid1D(n=16384, half_width=150.0)
    for v in (0.3, 0.6, 0.9):
        P = _fringe_pattern(g, v)
        assert visibility_prl(P, g.xi) == pytest.approx(v, rel=0.02)
        assert visibility_density(P, g.xi, visibility_bins(P, g.xi)) == pytest.approx(
            2.0 * v / np.pi, rel=0.02)


def test_prl_visibility_floors_at_zero_not_below():

    g = Grid1D(n=16384, half_width=150.0)
    P = _fringe_pattern(g, 0.9)
    bins = visibility_bins(P, g.xi)
    washed = blur_density(P, g.dxi, 3.0)
    assert visibility_prl(washed, g.xi) == 0.0
    assert visibility_density(washed, g.xi, bins) < 0.0


def test_prl_visibility_is_translation_invariant():

    g = Grid1D(n=16384, half_width=150.0)
    P = _fringe_pattern(g, 0.8)
    shifted = shift_density(P, g.dxi, np.array([0.8]))
    assert visibility_prl(shifted, g.xi) == pytest.approx(visibility_prl(P, g.xi), rel=0.03)


def test_visibility_rejects_a_pattern_without_fringes():
    g = Grid1D(n=2048, half_width=60.0)
    P = np.exp(-g.xi ** 2 / 50.0)
    with pytest.raises(ValueError, match="secondary maximum"):
        visibility_bins(P, g.xi)


def test_diffusion_constant_is_four():
    assert PI_DIFFUSION_PER_GAMMA == 4.0
