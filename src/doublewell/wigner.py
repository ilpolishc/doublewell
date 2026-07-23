import numpy as np


def _support(phi, xi, frac=1e-4):
    p = np.abs(phi) ** 2
    keep = np.where(p > frac * p.max())[0]
    return xi[keep[0]], xi[keep[-1]]


def _auto_window(phi, xi):
    lo, hi = _support(phi, xi)
    pad = 0.15 * (hi - lo) + 5.0
    return 0.5 * (lo + hi), 0.5 * (hi - lo) + pad


def wigner(phi, dxi, xi=None, center=None, half=None, max_lag=None,
           nx=241, pi_max=12.0, n_pi=241, decimate=1):
    if xi is None:
        xi = np.arange(len(phi)) * dxi
    if decimate > 1:
        phi = phi[::decimate]
        xi = xi[::decimate]
        dxi = dxi * decimate
    if center is None or half is None:
        c, h = _auto_window(phi, xi)
        center = c if center is None else center
        half = h if half is None else half

    n = len(phi)
    if max_lag is None:
        lo, hi = _support(phi, xi)
        Mcap = min(n // 2, int(np.ceil((hi - lo) / dxi)))
    else:
        Mcap = min(n // 2, int(np.ceil(max_lag / dxi)))

    sel = np.where(np.abs(xi - center) <= half)[0]
    if len(sel) == 0:
        sel = np.array([int(np.argmin(np.abs(xi - center)))])
    j_out = sel[np.unique(np.linspace(0, len(sel) - 1, min(nx, len(sel))).round().astype(int))]

    pi_axis = np.linspace(-pi_max, pi_max, n_pi)
    lags = np.arange(-Mcap, Mcap + 1)
    E = np.exp(1j * np.outer(pi_axis, lags) * dxi)

    W = np.zeros((len(j_out), n_pi))
    for row, c in enumerate(j_out):
        mm = min(c, n - 1 - c, Mcap)
        idx = np.arange(-mm, mm + 1)
        g = np.zeros(2 * Mcap + 1, dtype=np.complex128)
        g[Mcap - mm:Mcap + mm + 1] = np.conj(phi[c + idx]) * phi[c - idx]
        W[row] = (dxi / (2.0 * np.pi)) * np.real(E @ g)
    return xi[j_out], pi_axis, W
