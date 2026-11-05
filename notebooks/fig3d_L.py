import os
import sys
import time

import numpy as np
from scipy.special import erf

from doublewell.units import L
from doublewell.classical import trajectory, evolve_ensemble
from doublewell.splitstep import Grid1D, gaussian_state
from doublewell.observables import check_resolution
from doublewell.decoherence import (propagate_comoving_kicks, visibility_bins,
                                    visibility_prl, fringe_contrast_ratio)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")

SIGMA_CFG = dict(n=131072, half_width=4500.0, dtau=0.25)
SIGMA_OUT = os.path.join(DATA, "stage3d_quantum_L_sigma_s.npz")
SIGMA_SCAN = os.path.join(DATA, "stage3d_L_sigma_s_scan.npz")
SIGMA_S = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 20.0])


NBAR_CFG = dict(n=524288, half_width=9000.0, dtau=0.25)
NBAR_OUT = os.path.join(DATA, "stage3d_hybrid_L_nbar.npz")
NBAR_SCAN = os.path.join(DATA, "stage3d_L_nbar_scan.npz")
NBAR = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
SPLIT = 8.0
CHUNK = 8


def _trapz_weights(x):
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def reweight(dens, nodes, quad_w, sigma, dxi):
    if sigma <= 0.0:
        P = dens[int(np.argmin(np.abs(nodes)))].copy()
    else:
        w = quad_w * np.exp(-0.5 * (nodes / sigma) ** 2)
        P = (w[:, None] * dens).sum(axis=0) / w.sum()
    return P / (P.sum() * dxi)


def xi_nodes():
    a = np.array([0.0, 0.5, 1.0, 2.0, 3.5, 5.5, 8.0, 11.0, 14.5, 19.0,
                  24.0, 30.0, 38.0, 48.0, 62.0, 80.0])
    return np.unique(np.concatenate([-a[::-1], a]))


def pi_nodes(fine=0.25, fine_max=5.0, coarse=0.75, coarse_max=40.0):

    a = np.arange(-fine_max, fine_max + 1e-9, fine)
    b = np.arange(fine_max + coarse, coarse_max + 1e-9, coarse)
    pn = np.unique(np.concatenate([-b[::-1], a, b]))
    return pn[np.abs(pn) <= SPLIT]


def batch_sigma():
    n, hw, dtau = SIGMA_CFG["n"], SIGMA_CFG["half_width"], SIGMA_CFG["dtau"]
    check_resolution(L, n, hw)
    com = Grid1D(n=n, half_width=hw)
    nsteps = int(round(L.tau_m / dtau))
    xn = xi_nodes()
    print(f"L sigma_s: {len(xn)} xi_0 nodes, {nsteps} steps on n={n}", flush=True)
    phi0 = np.stack([gaussian_state(com, xi0=float(x), k0=0.0) for x in xn])
    t0 = time.perf_counter()
    out = propagate_comoving_kicks(phi0, com, L, L.tau_m, nsteps, 0.0,
                                   rng=np.random.default_rng(0), kick_every=nsteps)
    print(f"  propagated in {time.perf_counter() - t0:.0f}s", flush=True)
    dens = np.abs(out) ** 2
    dens /= dens.sum(axis=1, keepdims=True)
    np.savez_compressed(SIGMA_OUT, xi=com.xi, dxi=com.dxi, xi_nodes=xn, dens_xi=dens,
                        nsteps=nsteps, **SIGMA_CFG)
    print(f"saved {os.path.basename(SIGMA_OUT)}", flush=True)


def _sigma_tables():
    d = np.load(SIGMA_OUT)
    xi, dxi, xn = d["xi"], float(d["dxi"]), d["xi_nodes"]
    dx = d["dens_xi"] / dxi
    wx = _trapz_weights(xn)
    P0 = reweight(dx, xn, wx, 0.0, dxi)
    bins = visibility_bins(P0, xi)
    k0 = 2.0 * np.pi / bins["x_f"]
    v0 = visibility_prl(P0, xi)
    V = [visibility_prl(reweight(dx, xn, wx, float(s), dxi), xi) for s in SIGMA_S]
    C = [fringe_contrast_ratio(reweight(dx, xn, wx, float(s), dxi), xi, P0, xi, k0)
         for s in SIGMA_S]
    return xi, dxi, dx, xn, wx, P0, bins, v0, np.array(V), np.array(C)


def sigma_visibility(sigma):

    d = np.load(SIGMA_OUT)
    xi, dxi, xn = d["xi"], float(d["dxi"]), d["xi_nodes"]
    dx = d["dens_xi"] / dxi
    wx = _trapz_weights(xn)
    v0 = visibility_prl(reweight(dx, xn, wx, 0.0, dxi), xi)
    if sigma <= 0.0:
        return 1.0
    return visibility_prl(reweight(dx, xn, wx, float(sigma), dxi), xi) / v0


def figure_sigma():
    xi, dxi, dx, xn, wx, P0, bins, v0, V, C = _sigma_tables()
    print(f"L sigma_s: X_f={bins['x_f']:.3f} X_Om  V0={v0:.4f}")
    print("  sigma_s   V/V0    contrast")
    for s, v, c in zip(SIGMA_S, V, C):
        print(f"  {s:6.1f}  {v / v0:6.3f}   {c:7.4f}")
    np.savez(SIGMA_SCAN, sigma_s=SIGMA_S, V=V, C=C, V0=v0, Xf=bins["x_f"])
    print(f"saved {os.path.basename(SIGMA_SCAN)}")

def batch_nbar():
    n, hw, dtau = NBAR_CFG["n"], NBAR_CFG["half_width"], NBAR_CFG["dtau"]
    check_resolution(L, n, hw)
    com = Grid1D(n=n, half_width=hw)
    nsteps = int(round(L.tau_m / dtau))
    pn = pi_nodes()
    print(f"L n-bar: {len(pn)} pi_0 nodes |pi_0|<={SPLIT}, {nsteps} steps on n={n}, "
          f"chunks of {CHUNK}", flush=True)
    dens = np.zeros((len(pn), n), dtype=np.float32)
    t0 = time.perf_counter()
    for lo in range(0, len(pn), CHUNK):
        hi = min(lo + CHUNK, len(pn))
        phi0 = np.stack([gaussian_state(com, xi0=0.0, k0=float(p) / 2.0) for p in pn[lo:hi]])
        out = propagate_comoving_kicks(phi0, com, L, L.tau_m, nsteps, 0.0,
                                       rng=np.random.default_rng(0), kick_every=nsteps)
        d = np.abs(out) ** 2
        d /= d.sum(axis=1, keepdims=True)
        dens[lo:hi] = d.astype(np.float32)
        del out, d, phi0
        print(f"  nodes {hi:3d}/{len(pn)}  {time.perf_counter() - t0:6.0f}s", flush=True)
    np.savez_compressed(NBAR_OUT, xi=com.xi, dxi=com.dxi, pi_nodes=pn, dens=dens,
                        split=SPLIT, nsteps=nsteps, **NBAR_CFG)
    print(f"saved {os.path.basename(NBAR_OUT)}", flush=True)


def _far_density(sigma, xi, dxi, n_samp=400_000, seed=0):

    tw = L.omega_tm
    ref = trajectory(L.xs_over_D, tau_max=tw + 1e-9, rtol=1e-12, atol=1e-14).sol(tw)[0]
    scale = L.D_over_XOm * L.omega_over_Omega
    rng = np.random.default_rng(seed)
    p0 = rng.normal(0.0, sigma, n_samp * 4)
    p0 = p0[np.abs(p0) > SPLIT][:n_samp]
    if len(p0) < 100:
        return np.zeros_like(xi)
    x0 = rng.normal(0.0, sigma, len(p0)) + rng.normal(0.0, 1.0, len(p0))
    u, _ = evolve_ensemble(L.xs_over_D + x0 / L.D_over_XOm,
                           (p0 + rng.normal(0.0, 1.0, len(p0))) / scale, tw, dtau=1e-3)
    d = (u - ref) * L.D_over_XOm
    span = max(np.abs(d).max(), np.abs(xi).max()) * 1.05
    hist, edges = np.histogram(d, bins=int(2 * span / dxi), range=(-span, span))
    dens = hist / (hist.sum() * (edges[1] - edges[0]))
    return np.interp(xi, 0.5 * (edges[:-1] + edges[1:]), dens, left=0.0, right=0.0)


def figure_nbar():
    d = np.load(NBAR_OUT)
    xi, dxi, pn, dens = d["xi"], float(d["dxi"]), d["pi_nodes"], d["dens"].astype(float)
    wq = _trapz_weights(pn)
    P0 = dens[int(np.argmin(np.abs(pn)))]
    P0 = P0 / (P0.sum() * dxi)
    bins = visibility_bins(P0, xi)
    k0 = 2.0 * np.pi / bins["x_f"]
    v0 = visibility_prl(P0, xi)
    print(f"L n-bar: X_f={bins['x_f']:.3f} X_Om  V0={v0:.4f}  split |pi_0|={SPLIT:.0f}")
    print("  n-bar   far wt    V_mom    V_pos    V/V0    contrast")

    V, C = [], []
    for nb in NBAR:
        s = np.sqrt(2.0 * nb)
        if s == 0.0:
            V.append(1.0)
            C.append(1.0)
            print(f"  {nb:5.1f}   {0.0:6.4f}   {1.0:6.4f}   {1.0:6.4f}  {1.0:6.4f}  {1.0:7.4f}")
            continue
        fw = 1.0 - float(erf(SPLIT / (s * np.sqrt(2.0))))
        near = (wq * np.exp(-0.5 * (pn / s) ** 2))[:, None] * dens
        near = near.sum(axis=0)
        near /= near.sum() * dxi
        P = (1.0 - fw) * near + fw * _far_density(s, xi, dxi)
        P /= P.sum() * dxi

        v_mom = visibility_prl(P, xi) / v0
        v_pos = sigma_visibility(s)
        V.append(v_mom * v_pos)
        C.append(fringe_contrast_ratio(P, xi, P0, xi, k0))
        print(f"  {nb:5.1f}   {fw:6.4f}   {v_mom:6.4f}   {v_pos:6.4f}  {V[-1]:6.4f}  "
              f"{C[-1]:7.4f}", flush=True)

    np.savez(NBAR_SCAN, nbar=NBAR, V_over_V0=np.array(V), C_momentum=np.array(C),
             V0=v0, Xf=bins["x_f"], split=SPLIT, sigma_x=np.sqrt(2.0 * NBAR + 1.0))
    print(f"saved {os.path.basename(NBAR_SCAN)}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "figure-nbar"
    {"batch-sigma": batch_sigma, "figure-sigma": figure_sigma,
     "batch-nbar": batch_nbar, "figure-nbar": figure_nbar}[stage]()
