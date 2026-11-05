from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from doublewell.splitstep import residual_potential
from doublewell.classical import comoving_frame
from doublewell.observables import transfer_matrix

PI_DIFFUSION_PER_GAMMA = 4.0


def pi_diffusion_rate(gamma_over_Omega: float) -> float:
    return PI_DIFFUSION_PER_GAMMA * gamma_over_Omega



def propagate_comoving_kicks(phi, grid, params, tau_end, nsteps, gamma_over_Omega, rng=None,
                             kick_every: int = 1, noise=None,
                             callback=None, callback_every: int = 0):

    dtau = tau_end / nsteps
    xic_mid = comoving_frame(params, (np.arange(nsteps) + 0.5) * dtau)[0]
    drift = np.exp(-1j * grid.k ** 2 * dtau)
    phi = np.atleast_2d(np.asarray(phi, dtype=np.complex128)).copy()
    ntraj = phi.shape[0]
    xi = grid.xi
    unit = np.sqrt(pi_diffusion_rate(np.asarray(gamma_over_Omega, dtype=float)) * dtau)
    if noise is None and rng is None:
        raise ValueError("pass either rng or noise")
    per_group = False
    if noise is not None:
        noise = np.asarray(noise, dtype=float).reshape(-1, ntraj)
        n_kicks = -(-nsteps // kick_every)
        if noise.shape[0] == n_kicks and n_kicks != nsteps:
            per_group = True
        elif noise.shape[0] != nsteps:
            raise ValueError(f"noise must have {nsteps} (per step) or {n_kicks} (per group) "
                             f"rows, got {noise.shape[0]}")

    group_start, group = 0, 0
    for step in range(1, nsteps + 1):
        hk = np.exp(-0.5j * residual_potential(params, xi, xic_mid[step - 1]) * dtau)
        phi *= hk
        phi = ifft(drift * fft(phi, axis=-1, workers=-1), axis=-1, workers=-1)
        phi *= hk
        if step % kick_every == 0 or step == nsteps:
            n_acc = step - group_start
            if noise is None:
                z = np.sqrt(n_acc) * rng.standard_normal(ntraj)
            elif per_group:
                z = np.sqrt(n_acc) * noise[group]
            else:
                z = noise[group_start:step].sum(axis=0)
            phi *= np.exp(1j * (0.5 * unit * z)[:, None] * xi)
            group_start, group = step, group + 1
        if callback is not None and callback_every and (
                step % callback_every == 0 or step == nsteps):
            callback(phi, step)
    return phi


def ensemble_density(phi, dxi):
    P = np.mean(np.abs(np.atleast_2d(phi)) ** 2, axis=0)
    return P / (P.sum() * dxi)


# ---- Route B: Gaussian blur with a first-principles width --------------------


def blur_std(params, gamma_over_Omega, n_tau=8000):

    tau, M = transfer_matrix(params, params.tau_m, n_tau=n_tau)
    M11, M12 = M[:, 0, 0], M[:, 0, 1]
    G = M12[-1] * M11 - M11[-1] * M12
    integral = trapezoid(G ** 2, tau)
    return float(np.sqrt(pi_diffusion_rate(gamma_over_Omega) * integral))


def blur_density(P, dxi, sigma_xi, edge_tol=1e-6):

    P = np.asarray(P, dtype=float)
    if sigma_xi <= 0.0:
        return P / (P.sum() * dxi)
    edge = max(P[0], P[-1]) / P.max()
    if edge > edge_tol:
        raise ValueError(f"density is {edge:.1e} of peak at the grid edge (> {edge_tol:.0e}); "
                         "circular blur would wrap. Widen the box or crop first.")
    k = 2.0 * np.pi * rfftfreq(len(P), d=dxi)
    out = irfft(rfft(P) * np.exp(-0.5 * (k * sigma_xi) ** 2), n=len(P))
    return out / (out.sum() * dxi)


def shift_density(P, dxi, shifts):

    P = np.asarray(P, dtype=float)
    shifts = np.asarray(shifts, dtype=float).ravel()
    n = len(P)
    if np.abs(shifts).max() > 0.25 * n * dxi:
        raise ValueError("shifts exceed a quarter of the box; the circular convolution "
                         "would wrap. Widen the box.")
    pos = shifts / dxi
    j = np.floor(pos).astype(np.int64)
    frac = pos - j
    h = (np.bincount(np.mod(j, n), weights=1.0 - frac, minlength=n)
         + np.bincount(np.mod(j + 1, n), weights=frac, minlength=n)) / len(shifts)
    out = irfft(rfft(P) * rfft(h), n=n)
    return out / (out.sum() * dxi)


# ---- Visibility (Appendix C of arXiv:2506.22092) -----------------------------


def visibility_bins(P, xi, height=0.03, prominence=0.02):

    P = np.asarray(P, dtype=float)
    xi = np.asarray(xi, dtype=float)
    i_peak = int(np.argmax(P))
    peaks, _ = find_peaks(P, height=height * P[i_peak], prominence=prominence * P[i_peak])
    others = peaks[peaks != i_peak]
    if len(others) == 0:
        raise ValueError("no secondary maximum found; pattern has no resolvable fringes")
    i_second = int(others[np.argmin(np.abs(others - i_peak))])
    lo, hi = sorted((i_peak, i_second))
    i_min = lo + int(np.argmin(P[lo:hi + 1]))

    y_max, y_min, y_peak = float(xi[i_second]), float(xi[i_min]), float(xi[i_peak])
    delta = abs(y_max - y_min)

    return {"y_peak": y_peak, "y_max": y_max, "y_min": y_min, "delta": delta,
            "x_f": abs(y_max - y_peak),
            "i_max": (y_max - 0.5 * delta, y_max + 0.5 * delta),
            "i_min": (y_min - 0.5 * delta, y_min + 0.5 * delta)}


def _mass(P, xi, interval):

    lo, hi = interval
    m = (xi > lo) & (xi <= hi)
    return float(P[m].sum() * (xi[1] - xi[0]))


def visibility_density(P, xi, bins):

    n_max, n_min = _mass(P, xi, bins["i_max"]), _mass(P, xi, bins["i_min"])
    if n_max + n_min == 0.0:
        return 0.0
    return float((n_max - n_min) / (n_max + n_min))


def visibility_prl(P, xi, height=0.002, prominence=0.002):

    try:
        b = visibility_bins(P, xi, height=height, prominence=prominence)
    except ValueError:
        return 0.0
    P = np.asarray(P, dtype=float)
    xi = np.asarray(xi, dtype=float)
    Pa = float(P[int(np.argmin(np.abs(xi - b["y_max"])))])
    Pb = float(P[int(np.argmin(np.abs(xi - b["y_min"])))])
    if Pa + Pb <= 0.0:
        return 0.0
    return (Pa - Pb) / (Pa + Pb)


def fringe_contrast_ratio(P, xi, P_ref, xi_ref, k0):

    amp = lambda p, x: abs(np.sum(p * np.exp(-1j * k0 * x))) / np.sum(p)
    return float(amp(np.asarray(P, float), np.asarray(xi, float))
                 / amp(np.asarray(P_ref, float), np.asarray(xi_ref, float)))


def visibility_counts(Y, bins):

    Y = np.asarray(Y, dtype=float)
    lo_a, hi_a = bins["i_max"]
    lo_b, hi_b = bins["i_min"]
    n_max = int(np.count_nonzero((Y > lo_a) & (Y <= hi_a)))
    n_min = int(np.count_nonzero((Y > lo_b) & (Y <= hi_b)))
    if n_max + n_min == 0:
        return 0.0, n_max, n_min
    return float((n_max - n_min) / (n_max + n_min)), n_max, n_min
