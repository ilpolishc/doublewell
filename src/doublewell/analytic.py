from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import airy
from scipy.fft import rfft, irfft, rfftfreq

from doublewell.units import ParamSet
from doublewell.observables import transfer_matrix

@dataclass(frozen=True)
class ConstantAngleFrame:
    params: ParamSet
    tau: np.ndarray
    S: np.ndarray
    eta: np.ndarray
    phi_t: np.ndarray
    phi: float
    tau_d: float
    kappa3: float
    kappa4: float
    I_eta2: float

    @property
    def S_m(self) -> np.ndarray:
        return self.S[-1]

    def sigma_b2(self, gamma_over_Omega: float) -> float:
        return 4.0 * gamma_over_Omega * self.I_eta2

    def blur_direction(self) -> np.ndarray:
        c, s = np.cos(self.phi), np.sin(self.phi)
        return self.S_m @ np.array([-s, c])

    def blur_std_x(self, gamma_over_Omega: float) -> float:
        return float(np.sqrt(self.sigma_b2(gamma_over_Omega)) * abs(self.blur_direction()[0]))


def constant_angle_frame(params: ParamSet, n_tau: int = 20001) -> ConstantAngleFrame:
    tau, S = transfer_matrix(params, params.tau_m, n_tau=n_tau)
    Sxx, Sxp = S[:, 0, 0], S[:, 0, 1]
    eta = np.hypot(Sxx, Sxp)
    phi_t = np.arctan2(Sxp, Sxx)
    i_d = int(np.argmax(eta))
    phi = float(phi_t[i_d])

    from doublewell.classical import comoving_frame
    xi_c = comoving_frame(params, tau)[0]
    r2 = params.omega_over_Omega ** 2
    D = params.D_over_XOm
    kappa3 = float(r2 * trapezoid(3.0 * xi_c * eta ** 3 / D ** 2, tau))
    kappa4 = float(r2 * trapezoid(eta ** 4 / D ** 2, tau))
    I_eta2 = float(trapezoid(eta ** 2, tau))
    return ConstantAngleFrame(params, tau, S, eta, phi_t, phi, float(tau[i_d]),
                              kappa3, kappa4, I_eta2)



def _airy_integral(c1, c2, c3):

    a3 = np.abs(c3)
    arg = (c2 ** 2 + 4.0 * c3 * c1) / (4.0 * a3 ** (4.0 / 3.0))
    ai = airy(arg)[0]
    expo = (c2 ** 3 + 6.0 * c3 * c2 * c1) / (12.0 * c3 ** 2)
    return np.exp(expo) * ai / a3 ** (1.0 / 3.0)


def wigner_gaussian_frame(x_phi, p_phi, frame: ConstantAngleFrame,
                          var_x_phi: float = 1.0, var_p_phi: float = 1.0,
                          cov_xp_phi: float = 0.0, sigma_b2: float = 0.0):

    k3, k4 = frame.kappa3, frame.kappa4
    rho = cov_xp_phi / var_x_phi
    var_p_cond = var_p_phi - cov_xp_phi ** 2 / var_x_phi
    c3 = k3 + 3.0 * k4 * x_phi
    c2 = var_p_cond + sigma_b2
    c1 = p_phi - rho * x_phi + k3 * x_phi ** 2 + k4 * x_phi ** 3
    gauss = np.exp(-x_phi ** 2 / (2.0 * var_x_phi)) / np.sqrt(2.0 * np.pi * var_x_phi)
    return gauss * _airy_integral(c1, c2, c3)


def initial_covariance_phi(frame: ConstantAngleFrame, nbar: float = 0.0,
                           sigma_s: float = 0.0):
    s2 = 2.0 * nbar + 1.0
    C = np.diag([s2 + sigma_s ** 2, s2])
    c, s = np.cos(frame.phi), np.sin(frame.phi)
    R = np.array([[c, s], [-s, c]])
    Cp = R @ C @ R.T
    return float(Cp[0, 0]), float(Cp[1, 1]), float(Cp[0, 1])


def position_density(frame: ConstantAngleFrame, xi, p_half: float = 25.0, dp: float = 0.01,
                     var_x_phi: float = 1.0, var_p_phi: float = 1.0, cov_xp_phi: float = 0.0,
                     sigma_b2: float = 0.0, chunk: int = 1024):
    xi = np.asarray(xi, dtype=float)
    p = np.arange(-p_half, p_half + 0.5 * dp, dp)
    Sinv = np.linalg.inv(frame.S_m)
    c, s = np.cos(frame.phi), np.sin(frame.phi)
    R = np.array([[c, s], [-s, c]])
    A = R @ Sinv
    out = np.empty_like(xi)
    for i0 in range(0, len(xi), chunk):
        xs = xi[i0:i0 + chunk]
        x_phi = A[0, 0] * xs[:, None] + A[0, 1] * p[None, :]
        p_phi = A[1, 0] * xs[:, None] + A[1, 1] * p[None, :]
        W = wigner_gaussian_frame(x_phi, p_phi, frame, var_x_phi, var_p_phi, cov_xp_phi,
                                  sigma_b2)
        out[i0:i0 + chunk] = W.sum(axis=1) * dp
    dxi = xi[1] - xi[0]
    return out / (out.sum() * dxi)


def blur(P, dxi, sigma):

    if sigma <= 0.0:
        return P / (P.sum() * dxi)
    k = 2.0 * np.pi * rfftfreq(len(P), d=dxi)
    out = irfft(rfft(P) * np.exp(-0.5 * (k * sigma) ** 2), n=len(P))
    return out / (out.sum() * dxi)


def decohered_density(frame: ConstantAngleFrame, xi, P_coh, gamma_over_Omega: float):

    dxi = xi[1] - xi[0]
    return blur(P_coh, dxi, frame.blur_std_x(gamma_over_Omega))


def relative_density(P, xi):

    dxi = xi[1] - xi[0]
    n = len(P)
    n2 = 1 << int(np.ceil(np.log2(4 * n)))
    Pp = np.zeros(n2)
    Pp[:n] = P
    R = irfft(rfft(Pp) ** 2, n=n2) * dxi
    r = xi[0] * 2.0 + np.arange(n2) * dxi
    return r, np.clip(R, 0.0, None)
