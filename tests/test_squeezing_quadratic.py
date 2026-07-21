import numpy as np
import pytest

from doublewell.units import TOY_S, L
from doublewell.observables import squeezing_quadratic, evolve_covariance
from doublewell.classical import half_period_quad, comoving_frame


def test_prep_trap_keeps_ground_state():
    tau = np.linspace(0.0, 20.0, 200)
    Sxx, Sxp, Spp = evolve_covariance(lambda t: 0.25, tau)
    assert np.allclose(Sxx, 1.0, atol=1e-6)
    assert np.allclose(Spp, 1.0, atol=1e-6)
    assert np.allclose(Sxp, 0.0, atol=1e-6)


@pytest.mark.parametrize("P", [TOY_S, L], ids=["toyS", "L"])
def test_symplectic_and_eta_peak_and_recompression(P):
    tm = half_period_quad(0.1) / P.omega_over_Omega
    tau = np.linspace(0.0, 2.0 * tm, 4000)
    lam_m, lam_p = squeezing_quadratic(P, tau)

    assert np.allclose(lam_m * lam_p, 1.0, atol=5e-3)

    xi_c = comoving_frame(P, tau)[0]
    i_tD = np.argmin(np.abs(xi_c - P.D_over_XOm))
    assert np.sqrt(lam_p[i_tD]) == pytest.approx(P.eta, rel=0.02)
    assert 10.0 * np.log10(lam_p[i_tD]) == pytest.approx(P.S0_dB, abs=0.1)

    assert np.sqrt(lam_p[-1]) == pytest.approx(1.0, abs=0.2)


def test_turning_point_squeezing_depends_only_on_xs_over_D():
    vals = []
    for P in (TOY_S, L):
        tm = half_period_quad(0.1) / P.omega_over_Omega
        tau = np.linspace(0.0, 2.0 * tm, 4000)
        _, lam_p = squeezing_quadratic(P, tau)
        vals.append(np.sqrt(lam_p[np.argmin(np.abs(tau - tm))]))
    assert vals[0] == pytest.approx(vals[1], rel=0.02)
    assert vals[0] == pytest.approx(np.sqrt(2.0) / TOY_S.xs_over_D, rel=0.05)
