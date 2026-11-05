"""Constant-angle analytical approximation (Riera-Campeny et al., Quantum 8, 1393)."""
import os

import numpy as np
import pytest
from scipy.integrate import quad

from doublewell.units import L
from doublewell.analytic import (_airy_integral, constant_angle_frame, position_density,
                                 initial_covariance_phi)
from doublewell.decoherence import visibility_prl, visibility_bins, blur_std

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TM = os.path.join(DATA, "L_fig2ac.npz")


@pytest.mark.parametrize("c1,c2,c3", [(0.3, 1.0, 0.8), (0.3, 1.0, -0.8),
                                      (-1.2, 1.5, 2.0), (-1.2, 1.5, -2.0), (2.0, 1.0, -0.3)])
def test_airy_integral_matches_quadrature(c1, c2, c3):
    """Eq. (96) of the Quantum paper, including the sign convention for c3 < 0."""
    f = lambda k: np.exp(-0.5 * c2 * k * k) * np.cos(c3 * k ** 3 / 3.0 + c1 * k)
    ref = quad(f, -60.0, 60.0, limit=2000)[0] / (2.0 * np.pi)
    assert _airy_integral(c1, c2, c3) == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_frame_quantities_L():
    fr = constant_angle_frame(L, n_tau=4001)
    assert fr.eta.max() == pytest.approx(L.eta, rel=1e-3)          # eta_* = eta of Eq. (1)
    assert fr.tau_d / L.tau_m == pytest.approx(0.782, abs=2e-3)
    assert abs(fr.phi / np.pi - 0.5) < 0.01                        # phi ~ pi/2 (paper: 0.499969)
    assert abs(np.linalg.det(fr.S_m) - 1.0) < 1e-6


def test_blur_width_equals_route_B_green_function():
    """The constant-angle blur, sigma_b |(S R^T e_p)_x|, is the exact linear-response blur
    of decoherence.blur_std with phi(tau') frozen; at L they agree to < 1%."""
    fr = constant_angle_frame(L, n_tau=4001)
    for g in (1e-8, 4e-7):
        assert fr.blur_std_x(g) == pytest.approx(blur_std(L, g), rel=1e-2)


@pytest.mark.skipif(not os.path.exists(TM), reason="needs data/L_fig2ac.npz (run_fig2ac.py)")
def test_analytic_pattern_matches_numerical_t_m():
    with np.load(TM) as d:
        j = int(np.argmin(np.abs(d["snap_fracs"] - 1.0)))
        xi, dxi = d["xi"], float(d["dxi"])
        P = np.abs(d["snaps"][j]) ** 2
    m = (xi > -40.0) & (xi < 10.0)
    xi, P = xi[m], P[m] / (P[m].sum() * dxi)
    fr = constant_angle_frame(L, n_tau=4001)
    Pa = position_density(fr, xi, p_half=15.0, dp=0.02)
    bn, ba = visibility_bins(P, xi), visibility_bins(Pa, xi)
    assert ba["x_f"] == pytest.approx(bn["x_f"], abs=2 * dxi)
    assert ba["y_min"] == pytest.approx(bn["y_min"], abs=2 * dxi)
    assert visibility_prl(Pa, xi) == pytest.approx(visibility_prl(P, xi), abs=2e-3)
    assert np.abs(Pa - P).max() / P.max() < 0.05


def test_initial_covariance_rotation():
    fr = constant_angle_frame(L, n_tau=2001)
    vx, vp, cxp = initial_covariance_phi(fr, sigma_s=10.0)
    # phi ~ pi/2: the lab position variance 1 + sigma_s^2 lands on the p_phi quadrature
    assert vp == pytest.approx(101.0, rel=2e-3)
    assert vx == pytest.approx(1.0, abs=0.05)
    vx, vp, cxp = initial_covariance_phi(fr, nbar=3.0)
    assert vx == vp == pytest.approx(7.0) and cxp == pytest.approx(0.0, abs=1e-12)
