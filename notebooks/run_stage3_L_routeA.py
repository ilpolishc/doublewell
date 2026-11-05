import os
import sys
import time

import numpy as np

from doublewell.units import L
from doublewell.splitstep import Grid1D, gaussian_state, propagate_comoving
from doublewell.observables import check_resolution, transfer_matrix
from doublewell.decoherence import (propagate_comoving_kicks, ensemble_density, blur_std,
                                    blur_density, visibility_bins, visibility_prl,
                                    fringe_contrast_ratio, pi_diffusion_rate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "stage3_L_routeA.npz")

CFG = dict(n=131072, half_width=4500.0, dtau=0.25)
GAMMAS = np.array([1e-8, 4e-8, 10e-8, 40e-8])
NPAIR = 32
KICK_EVERY = 5
SEED = 7


def build_noise(n_kicks, ngamma, npair, seed=SEED):

    rng = np.random.default_rng(seed)
    z = np.repeat(rng.standard_normal((n_kicks, 1, npair)), ngamma, axis=1)
    return np.stack([z, -z], axis=3).reshape(n_kicks, ngamma * 2 * npair)


def linear_response_sigma(gammas, nsteps, dtau, npair=NPAIR, kick_every=KICK_EVERY):

    n_kicks = -(-nsteps // kick_every)
    noise = build_noise(n_kicks, len(gammas), npair)
    tau, M = transfer_matrix(L, L.tau_m, n_tau=40001)
    G = M[-1, 0, 1] * M[:, 0, 0] - M[-1, 0, 0] * M[:, 0, 1]
    steps = np.minimum(np.arange(1, n_kicks + 1) * kick_every, nsteps)
    n_acc = np.diff(np.concatenate([[0], steps]))
    Gg = np.interp(steps * dtau, tau, G)
    out = []
    for i, g in enumerate(gammas):
        unit = np.sqrt(pi_diffusion_rate(g) * dtau)
        dpi = unit * np.sqrt(n_acc)[:, None] * noise[:, i * 2 * npair:(i + 1) * 2 * npair]
        out.append(float((Gg[:, None] * dpi).sum(axis=0).std()))
    return np.array(out)


def stage_ensemble():
    n, hw, dtau = CFG["n"], CFG["half_width"], CFG["dtau"]
    check_resolution(L, n, hw)
    com = Grid1D(n=n, half_width=hw)
    nsteps = int(round(L.tau_m / dtau))
    row_gamma = np.repeat(GAMMAS, 2 * NPAIR)
    ntraj = len(row_gamma)
    n_kicks = -(-nsteps // KICK_EVERY)
    noise = build_noise(n_kicks, len(GAMMAS), NPAIR)
    print(f"L Route A: {ntraj} trajectories x {nsteps} steps (dtau={dtau}), "
          f"{n_kicks} kick groups", flush=True)

    t0 = time.perf_counter()
    coh = propagate_comoving(gaussian_state(com, xi0=0.0), com, L, L.tau_m, nsteps)
    P0 = np.abs(coh) ** 2
    P0 /= P0.sum() * com.dxi
    print(f"  coherent reference in {time.perf_counter() - t0:.0f}s", flush=True)

    phi0 = np.tile(gaussian_state(com, xi0=0.0), (ntraj, 1))
    t0 = time.perf_counter()
    out = propagate_comoving_kicks(phi0, com, L, L.tau_m, nsteps, row_gamma,
                                   kick_every=KICK_EVERY, noise=noise)
    print(f"  ensemble in {time.perf_counter() - t0:.0f}s", flush=True)

    blocks = [out[i * 2 * NPAIR:(i + 1) * 2 * NPAIR] for i in range(len(GAMMAS))]
    dens = np.stack([ensemble_density(b, com.dxi) for b in blocks])
    half = np.stack([np.stack([ensemble_density(b[:NPAIR], com.dxi),
                               ensemble_density(b[NPAIR:], com.dxi)]) for b in blocks])
    np.savez_compressed(OUT, xi=com.xi, dxi=com.dxi, gammas=GAMMAS, dens=dens, half=half,
                        P0=P0, nsteps=nsteps, npair=NPAIR, **CFG)
    print(f"saved {os.path.basename(OUT)}", flush=True)


def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    xi, dxi, P0, dens, half = d["xi"], float(d["dxi"]), d["P0"], d["dens"], d["half"]
    gammas, nsteps = d["gammas"], int(d["nsteps"])
    bins = visibility_bins(P0, xi)
    k0 = 2.0 * np.pi / bins["x_f"]
    V0 = visibility_prl(P0, xi)      # PRL statistic throughout (D11)
    sig_lr = linear_response_sigma(gammas, nsteps, float(d["dtau"]))

    sig_b = np.array([blur_std(L, g) for g in gammas])
    sig_a, V_a, V_b, err = [], [], [], []
    for i, g in enumerate(gammas):
        C = fringe_contrast_ratio(dens[i], xi, P0, xi, k0)
        sig_a.append(np.sqrt(max(-2.0 * np.log(max(C, 1e-12)), 0.0)) / k0)
        V_a.append(visibility_prl(dens[i], xi))
        V_b.append(visibility_prl(blur_density(P0, dxi, sig_b[i]), xi))
        err.append(abs(visibility_prl(half[i, 0], xi)
                       - visibility_prl(half[i, 1], xi)) / 2.0)

    print(f"L set, X_f={bins['x_f']:.3f} X_Om  V0={V0:.4f}")
    print("  Gamma/Omega  sigma_B  sigma_LR  sigma_A   A/LR    Delta_pi   V_A               V_B")
    for i, g in enumerate(gammas):
        dpi = np.sqrt(pi_diffusion_rate(g) * L.tau_m)
        print(f"  {g:.1e}    {sig_b[i]:7.4f} {sig_lr[i]:9.4f} {sig_a[i]:8.4f} "
              f"{sig_a[i] / sig_lr[i]:7.3f}  {dpi:8.4f}   {V_a[i]:+7.4f} +- {err[i]:.4f}"
              f"  {V_b[i]:+7.4f}")
    print(f"\n  realised-noise excess over nominal: {np.mean(sig_lr / sig_b):.3f}x")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    peak = bins["y_peak"]
    a.plot(xi, P0, color="0.72", lw=1.0, label="coherent")
    for i, g in enumerate(gammas):
        c = plt.cm.viridis(0.15 + 0.7 * i / (len(gammas) - 1))
        a.plot(xi, dens[i], color=c, lw=1.1, label=r"A: $%g\times10^{-8}$" % (g * 1e8))
        a.plot(xi, blur_density(P0, dxi, sig_b[i]), color=c, lw=2.2, ls=":", alpha=0.8)
    a.set_xlim(peak - 45.0, peak + 8.0)
    a.set_xlabel(r"$\tilde X/X_\Omega$"); a.set_ylabel(r"$P(X,t_m)$")
    a.set_title("L set: Route A (thin) vs Route B (dotted)"); a.legend(fontsize=7.5)

    b.errorbar(gammas, V_a, yerr=err, fmt="o", ms=6, color="#1f6fb2", label="Route A")
    b.plot(gammas, V_b, "-", color="#e07b00", lw=2, label="Route B")
    b.set_xscale("log"); b.set_xlabel(r"$\Gamma/\Omega$")
    b.set_ylabel("visibility of the first minimum")
    b.set_title("L set: A vs B at the PRL's own $\\Gamma$"); b.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "stage3_AvsB_L.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {os.path.basename(out)}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "ensemble"
    {"ensemble": stage_ensemble, "figure": stage_figure}[stage]()
