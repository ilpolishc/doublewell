import os
import sys
import time

import numpy as np

from doublewell.units import TOY_M
from doublewell.classical import trajectory
from doublewell.splitstep import Grid1D, gaussian_state
from doublewell.observables import check_resolution
from doublewell.decoherence import (propagate_comoving_kicks, visibility_bins,
                                    visibility_prl, fringe_contrast_ratio)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "stage3d_quantum_toyM_wide.npz")
SCAN = os.path.join(DATA, "stage3d_wide_scan.npz")

CFG = dict(n=262144, half_width=5000.0, dtau=0.05)
NBAR = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
CHUNK = 24


def _trapz_weights(x):
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def pi_nodes(fine=0.25, fine_max=5.0, coarse=0.75, coarse_max=40.0):

    a = np.arange(-fine_max, fine_max + 1e-9, fine)
    b = np.arange(fine_max + coarse, coarse_max + 1e-9, coarse)
    return np.unique(np.concatenate([-b[::-1], a, b]))


def _nbar_weights(nodes):
    wq = _trapz_weights(nodes)
    W = np.zeros((len(NBAR), len(nodes)))
    for a, nb in enumerate(NBAR):
        s = np.sqrt(2.0 * nb)
        W[a] = ((np.abs(nodes) < 1e-12).astype(float) if s == 0.0
                else wq * np.exp(-0.5 * (nodes / s) ** 2))
    return W


def node_is_contained(nodes, dens, xi, half_width, tol=0.1):

    tw = TOY_M.omega_tm
    kw = dict(tau_max=tw + 1e-9, rtol=1e-12, atol=1e-14)
    ref = trajectory(TOY_M.xs_over_D, **kw).sol(tw)[0]
    scale = TOY_M.D_over_XOm * TOY_M.omega_over_Omega
    predicted = np.array(
        [(trajectory(TOY_M.xs_over_D, v0=float(p) / scale, **kw).sol(tw)[0] - ref)
         * TOY_M.D_over_XOm for p in nodes])
    centroid = (dens * xi).sum(axis=1) / dens.sum(axis=1)
    skew = centroid[int(np.argmin(np.abs(nodes)))]
    return np.abs(centroid - skew - predicted) < tol * half_width


def batch():
    n, hw, dtau = CFG["n"], CFG["half_width"], CFG["dtau"]
    check_resolution(TOY_M, n, hw)
    com = Grid1D(n=n, half_width=hw)
    nsteps = int(round(TOY_M.tau_m / dtau))
    pn = pi_nodes()
    W = _nbar_weights(pn)
    print(f"toy-M wide box: n={n} hw={hw:.0f} dxi={com.dxi:.5f} {len(pn)} nodes, "
          f"{nsteps} steps, chunks of {CHUNK}", flush=True)

    acc = np.zeros((len(NBAR), n))
    centroid = np.zeros(len(pn))
    dens_pi0 = None
    t0 = time.perf_counter()
    for lo in range(0, len(pn), CHUNK):
        hi = min(lo + CHUNK, len(pn))
        phi0 = np.stack([gaussian_state(com, xi0=0.0, k0=float(p) / 2.0) for p in pn[lo:hi]])
        out = propagate_comoving_kicks(phi0, com, TOY_M, TOY_M.tau_m, nsteps, 0.0,
                                       rng=np.random.default_rng(0), kick_every=nsteps)
        dens = np.abs(out) ** 2
        dens /= dens.sum(axis=1, keepdims=True)
        centroid[lo:hi] = (dens * com.xi).sum(axis=1)
        acc += W[:, lo:hi] @ dens
        for j, p in enumerate(pn[lo:hi]):
            if abs(p) < 1e-12:
                dens_pi0 = dens[j].copy()
        del out, dens, phi0
        print(f"  nodes {hi:3d}/{len(pn)}  {time.perf_counter() - t0:6.0f}s", flush=True)

    dn = acc / W.sum(axis=1)[:, None]
    dn /= dn.sum(axis=1, keepdims=True) * com.dxi
    np.savez_compressed(OUT, xi=com.xi, dxi=com.dxi, pi_nodes=pn, nbar=NBAR,
                        dens_nbar=dn, P0=dens_pi0 / (dens_pi0.sum() * com.dxi),
                        centroid=centroid, nsteps=nsteps, **CFG)
    print(f"saved {os.path.basename(OUT)}", flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    xi, dxi, P0, dens = d["xi"], float(d["dxi"]), d["P0"], d["dens_nbar"]
    pn, cent = d["pi_nodes"], d["centroid"]
    bins = visibility_bins(P0, xi)
    k0 = 2.0 * np.pi / bins["x_f"]
    v0 = visibility_prl(P0, xi)

    tw = TOY_M.omega_tm
    kw = dict(tau_max=tw + 1e-9, rtol=1e-12, atol=1e-14)
    ref = trajectory(TOY_M.xs_over_D, **kw).sol(tw)[0]
    scale = TOY_M.D_over_XOm * TOY_M.omega_over_Omega
    pred = np.array([(trajectory(TOY_M.xs_over_D, v0=float(p) / scale, **kw).sol(tw)[0]
                      - ref) * TOY_M.D_over_XOm for p in pn])
    skew = cent[int(np.argmin(np.abs(pn)))]
    ok = np.abs(cent - skew - pred) < 0.1 * float(d["half_width"])
    wq = _trapz_weights(pn)
    print(f"toy-M wide box: X_f={bins['x_f']:.3f} X_Om  V0={v0:.4f}  "
          f"contained {ok.sum()}/{len(pn)} nodes (|pi_0| <= {np.abs(pn[ok]).max():.2f})")
    print("  n-bar   wrapped wt    V/V0    contrast   envelope std")

    V, C, leak = [], [], []
    for nb, P in zip(d["nbar"], dens):
        s = np.sqrt(2.0 * nb)
        w = wq * np.exp(-0.5 * (pn / s) ** 2) if s > 0 else (np.abs(pn) < 1e-12) * 1.0
        lk = float(1.0 - (w * ok).sum() / w.sum())
        m = float((P * xi).sum() * dxi)
        V.append(visibility_prl(P, xi) / v0)
        C.append(fringe_contrast_ratio(P, xi, P0, xi, k0))
        leak.append(lk)
        print(f"  {nb:5.1f}   {lk:9.4f}   {V[-1]:7.4f}   {C[-1]:7.4f}   "
              f"{np.sqrt((P * xi ** 2).sum() * dxi - m ** 2):10.2f}"
              f"{'   <- WRAPPED' if lk > 0.01 else ''}")

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    good = np.array(leak) <= 0.01
    nb = np.clip(d["nbar"], 0.03, None)
    ax.semilogx(nb[good], np.array(V)[good], "o-", color="#c0392b", lw=2, ms=5,
                label=r"$V/V_0$")
    ax.semilogx(nb[good], np.array(C)[good], "s--", color="#7f8c8d", lw=1.5, ms=4,
                label=r"contrast at $k_0$")
    if (~good).any():
        ax.semilogx(nb[~good], np.array(V)[~good], "x", color="0.55", ms=7,
                    label="wrapped: box too small")
    ax.axvline(40.0, color="0.7", ls=":", lw=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\bar n$  (0 plotted at 0.03)")
    ax.set_ylabel("surviving visibility (momentum half only)")
    ax.set_title("toy-M wide box: momentum quadrature only (validation, D8)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "stage3_fig3d_toyM_wide.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    np.savez(SCAN, nbar=d["nbar"], V=np.array(V), C=np.array(C), leak=np.array(leak),
             V0=v0, Xf=bins["x_f"])
    print(f"\nsaved {os.path.basename(out)}")


if __name__ == "__main__":
    {"batch": batch, "figure": figure}[sys.argv[1] if len(sys.argv) > 1 else "figure"]()
