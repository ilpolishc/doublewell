import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doublewell.units import L, ParamSet
from doublewell.decoherence import visibility_prl, visibility_bins
from doublewell.analytic import (constant_angle_frame, position_density, blur,
                                 relative_density, initial_covariance_phi)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")

GAMMAS = np.array([1e-8, 4e-8, 10e-8, 40e-8])
GAMMA_COLORS = ["#f3a0c0", "#df5f93", "#ab2d60", "#6d1536"]
REL_COLOR = "#e8820c"
ANA = dict(color="k", ls="--", lw=1.0)
SETS = [("XXL", 1e-4, 1e8, "#9ecae1"), ("XL", 1e-3, 1e6, "#3182bd"), ("L", 1e-2, 1e4, "#08306b")]
WINDOW = (-60.0, 10.0)
SIGMA_S = np.linspace(0.0, 14.0, 57)
NBAR = np.concatenate([[0.0], np.logspace(-1, np.log10(100.0), 40)])
NBAR_TICKS = np.array([0, 20, 40, 60, 80])


def load_numeric():
    with np.load(os.path.join(DATA, "L_fig2ac.npz")) as d:
        j = int(np.argmin(np.abs(d["snap_fracs"] - 1.0)))
        xi, dxi = d["xi"], float(d["dxi"])
        P = np.abs(d["snaps"][j]) ** 2
    P = P / (P.sum() * dxi)
    m = (xi > WINDOW[0]) & (xi < WINDOW[1])
    return xi[m], dxi, P[m]


def analytic_curve_V(frame, xi, dxi, v0, **kw):
    P = position_density(frame, xi, **kw)
    return visibility_prl(P, xi) / v0, P


def main():
    xi, dxi, Pn = load_numeric()
    v0n = visibility_prl(Pn, xi)
    fr = constant_angle_frame(L)
    wmass = Pn.sum() * dxi
    Pa = position_density(fr, xi) * wmass
    v0a = visibility_prl(Pa, xi)
    ba, bn = visibility_bins(Pa, xi), visibility_bins(Pn, xi)
    print(f"L constant-angle frame: phi/pi = {fr.phi / np.pi:.6f}  kappa3 = {fr.kappa3:.2f}  "
          f"kappa4 = {fr.kappa4:.2f}  eta_max = {fr.eta.max():.1f} at tau_d/tau_m = "
          f"{fr.tau_d / L.tau_m:.4f}")
    print(f"coherent pattern   numeric: V0 = {v0n:.4f}  X_f = {bn['x_f']:.3f}   "
          f"analytic: V0 = {v0a:.4f}  X_f = {ba['x_f']:.3f}   "
          f"max|dP|/maxP = {np.abs(Pa - Pn).max() / Pn.max():.3f}")
    out = dict(xi=xi, P_numeric=Pn, P_analytic=Pa, V0_numeric=v0n, V0_analytic=v0a,
               phi=fr.phi, kappa3=fr.kappa3, kappa4=fr.kappa4, I_eta2=fr.I_eta2)

    with np.load(os.path.join(DATA, "stage3_L_routeA.npz")) as d:
        xiA, densA, gamA = d["xi"], d["dens"], d["gammas"]
    mA = (xiA > WINDOW[0]) & (xiA < WINDOW[1])
    densA = densA / (densA.sum(axis=1, keepdims=True) * dxi)
    xiA, densA = xiA[mA], densA[:, mA]
    print("\n(a)  Gamma/Omega   blur std (analytic)   V/V0 analytic   V/V0 Route A")
    for g, col, PA in zip(gamA, GAMMA_COLORS, densA):
        s = fr.blur_std_x(g)
        Pg = blur(Pa, dxi, s) * wmass
        va, vA = visibility_prl(Pg, xi) / v0a, visibility_prl(PA, xiA) / v0n
        print(f"     {g:.0e}        {s:6.4f}               {va:6.3f}          {vA:6.3f}")
        out[f"P_analytic_gamma_{g:.0e}"] = Pg

    with np.load(os.path.join(DATA, "stage3_L_visibility.npz")) as d:
        r_xi, r_P, v_rel_n = d["r_xi"], d["r_P"], float(d["V_relative"])
    ra, Ra = relative_density(Pa, xi)
    v_rel_a = visibility_prl(Ra, ra)
    print(f"\n(b)  P(r~) visibility: numeric {v_rel_n:.4f}   analytic {v_rel_a:.4f};  "
          f"peak 10^3 P: numeric {1e3 * r_P.max():.2f} at {r_xi[np.argmax(r_P)]:.2f}, "
          f"analytic {1e3 * Ra.max():.2f} at {ra[np.argmax(Ra)]:.2f}")
    out.update(r_analytic=ra, Pr_analytic=Ra, V_rel_numeric=v_rel_n, V_rel_analytic=v_rel_a)

    with np.load(os.path.join(DATA, "stage3_L_visibility.npz")) as d:
        sweep = {k: d[k] for k in d.files if k.startswith(("gamma_", "V_"))}
    with np.load(os.path.join(DATA, "stage3_L_routeA.npz")) as d:
        half = d["half"][:, :, mA]
    vA = np.array([visibility_prl(p, xiA) / v0n for p in densA])
    vH = np.array([[visibility_prl(h, xiA) / v0n for h in hh] for hh in half])
    errA = np.abs(vH[:, 0] - vH[:, 1]) / 2.0
    fc, c = plt.subplots(figsize=(5.4, 4.1))
    print("\n(c)  set   phi/pi   kappa3   X_f(analytic)  Gamma/Omega at V/V0 = 0.5")
    for name, w, D, col in SETS:
        p = ParamSet(name, xs_over_D=0.1, omega_over_Omega=w, D_over_XOm=D)
        f = fr if name == "L" else constant_angle_frame(p)
        P0 = Pa if name == "L" else position_density(f, xi)
        v0 = visibility_prl(P0, xi)
        gam = sweep[f"gamma_{name}"]
        gfine = np.logspace(np.log10(gam[0]), np.log10(gam[-1]), 60)
        V = np.array([visibility_prl(blur(P0, dxi, f.blur_std_x(g)), xi) / v0 for g in gfine])
        g_half = np.exp(np.interp(0.5, V[::-1], np.log(gfine[::-1])))
        print(f"     {name:4s} {f.phi / np.pi:.5f} {f.kappa3:9.2f}  "
              f"{visibility_bins(P0, xi)['x_f']:.3f}         {g_half:.2e}")
        c.semilogx(gam[::2], sweep[f"V_{name}"][::2], "o", ms=4, color=col, mew=0, label=name)
        c.semilogx(gfine, V, **ANA)
        out[f"gamma_analytic_{name}"], out[f"V_analytic_{name}"] = gfine, V
    c.errorbar(gamA, vA, yerr=errA, fmt="s", ms=5, color="#08306b", mfc="w", mew=1.2,
               capsize=2, lw=1, label="L, Route A", zorder=6)
    c.axhline(v_rel_n / v0n, color=REL_COLOR, lw=1.6)
    c.annotate(r"$P(\tilde r)$, collective $\Gamma/\Omega$",
               (1.3e-15, v_rel_n / v0n + 0.03), fontsize=8, color=REL_COLOR)
    c.plot([], [], **ANA, label="analytic")
    c.set_xlim(1e-15, 1e-5); c.set_ylim(0, 1.05)
    c.set_xlabel(r"$\Gamma/\Omega$"); c.set_ylabel("Visibility")
    c.set_title("(c)", loc="left", fontsize=10)
    c.legend(fontsize=8, loc="center left")
    fc.tight_layout(); fc.savefig(os.path.join(FIG, "stage3_fig3c_L_analytic.png"), dpi=150)
    plt.close(fc)
    out.update(gamma_routeA=gamA, V_routeA=vA, V_routeA_err=errA)

    s = np.load(os.path.join(DATA, "stage3d_L_sigma_s_scan.npz"))
    n = np.load(os.path.join(DATA, "stage3d_L_nbar_scan.npz"))
    sig_n, Vs_n = s["sigma_s"], s["V"] / s["V0"]
    nbar_n, Vn_n = n["nbar"], n["V_over_V0"]

    cache = os.path.join(DATA, "stage3_L_analytic.npz")
    if "--reuse" in sys.argv and os.path.exists(cache):
        with np.load(cache) as d:
            Vs_a, Vn_a = d["V_sigma_s_analytic"], d["V_nbar_analytic"]
    else:
        Vs_a, Vn_a = None, None
    Vs_a = [] if Vs_a is None else list(Vs_a)
    for ss in (SIGMA_S if not Vs_a else []):
        vx, vp, cxp = initial_covariance_phi(fr, sigma_s=ss)
        Vs_a.append(analytic_curve_V(fr, xi, dxi, v0a, var_x_phi=vx, var_p_phi=vp,
                                     cov_xp_phi=cxp)[0])
    Vn_a = [] if Vn_a is None else list(Vn_a)
    for nb in (NBAR if not Vn_a else []):
        vx, vp, cxp = initial_covariance_phi(fr, nbar=nb)
        Vn_a.append(analytic_curve_V(fr, xi, dxi, v0a, var_x_phi=vx, var_p_phi=vp,
                                     cov_xp_phi=cxp)[0])
    Vs_a, Vn_a = np.array(Vs_a), np.array(Vn_a)
    print("\n(d)  sigma_s   V/V0 numeric   V/V0 analytic")
    for ss, v in zip(sig_n, Vs_n):
        print(f"     {ss:5.1f}      {v:6.3f}        {np.interp(ss, SIGMA_S, Vs_a):6.3f}")
    print("     n-bar    V/V0 numeric   V/V0 analytic")
    for nb, v in zip(nbar_n, Vn_n):
        print(f"     {nb:5.1f}      {v:6.3f}        {np.interp(nb, NBAR, Vn_a):6.3f}")

    fd, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(sig_n, Vs_n, "o", ms=6, color="#1f6fb2", mew=0,
            label=r"$\sigma_s$: quantum nodes (numerical)")
    ax.plot(np.sqrt(2.0 * nbar_n), Vn_n, "o", ms=6, mfc="none", mec="#c0392b", mew=1.4,
            label=r"$\bar n$: quantum nodes (numerical)")
    ax.plot(SIGMA_S, Vs_a, "-", color="k", lw=1.2, label=r"$\sigma_s$: analytic")
    ax.plot(np.sqrt(2.0 * NBAR), Vn_a, "--", color="#c0392b", lw=1.2,
            label=r"$\bar n$: analytic")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\sigma_{\rm s}/X_\Omega$")
    ax.set_ylabel("Visibility of the first minimum,  $V/V_0$")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25, lw=0.5)
    top = ax.twiny(); top.set_xlim(ax.get_xlim())
    top.set_xticks(np.sqrt(2.0 * NBAR_TICKS)); top.set_xticklabels([str(t) for t in NBAR_TICKS])
    top.set_xlabel(r"$\bar n$", color="#c0392b"); top.tick_params(axis="x", colors="#c0392b")
    ax.axvline(np.sqrt(80.0), color="0.6", ls=":", lw=1.0)
    fd.suptitle("L set. PRL Fig. 3d", y=0.98, fontsize=11)
    fd.tight_layout(); fd.savefig(os.path.join(FIG, "stage3_fig3d_L_analytic.png"), dpi=150)
    plt.close(fd)
    out.update(sigma_s_analytic=SIGMA_S, V_sigma_s_analytic=Vs_a,
               nbar_analytic=NBAR, V_nbar_analytic=Vn_a)

    np.savez(os.path.join(DATA, "stage3_L_analytic.npz"), **out)
    print("\nsaved stage3_fig3c_L_analytic.png, stage3_fig3d_L_analytic.png, stage3_L_analytic.npz")


if __name__ == "__main__":
    main()
