import os

import numpy as np
from scipy.fft import rfft, irfft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doublewell.units import L, ParamSet
from doublewell.decoherence import (blur_std, blur_density, visibility_bins,
                                    visibility_density, visibility_prl, pi_diffusion_rate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(DATA, "L_fig2ac.npz")

GAMMAS = np.array([1e-8, 4e-8, 10e-8, 40e-8])
GAMMA_COLORS = ["#f3a0c0", "#df5f93", "#ab2d60", "#6d1536"]
REL_COLOR = "#e8820c"

SETS = [("XXL", 1e-4, 1e8, "-", "#9ecae1"),
        ("XL", 1e-3, 1e6, "--", "#3182bd"),
        ("L", 1e-2, 1e4, ":", "#08306b")]
SIGMA_SPAN = (0.05, 1.4)      # sigma range over which V falls from ~1 to ~0
N_SWEEP = 40


def load_tm():
    with np.load(SRC) as d:
        j = int(np.argmin(np.abs(d["snap_fracs"] - 1.0)))
        xi, dxi = d["xi"], float(d["dxi"])
        P = np.abs(d["snaps"][j]) ** 2
        xic = float(d["snap_xic"][j])
    return xi, dxi, P / (P.sum() * dxi), xic


def relative_density(P, xi, dxi, keep=5000.0):
    """P(r~) = P * P, padded so the circular convolution cannot wrap."""
    n2 = 1 << int(np.ceil(np.log2(4 * len(P))))
    Pp = np.zeros(n2)
    Pp[:len(P)] = P
    R = irfft(rfft(Pp) ** 2, n=n2) * dxi
    xr = 2.0 * xi[0] + np.arange(n2) * dxi
    m = np.abs(xr) < keep
    xr, R = xr[m], R[m]
    return xr, R / (R.sum() * dxi)


def set_sweep(p):
    ref_g = 1e-10
    ref_s = blur_std(p, ref_g)
    lo, hi = (ref_g * (s / ref_s) ** 2 for s in SIGMA_SPAN)
    return np.logspace(np.log10(lo), np.log10(hi), N_SWEEP)


def _mark(ax, x, y, labels=(r"$P_{\rm a}$", r"$P_{\rm b}$"), dy=(7, -13)):
    ax.plot(x, y, "o", ms=4, color="k", zorder=5)
    for xi_, yi_, lab, d in zip(x, y, labels, dy):
        ax.annotate(lab, (xi_, yi_), textcoords="offset points",
                    xytext=(3, d), fontsize=10)
        ax.axvline(xi_, color="0.6", ls=":", lw=0.7)


def main():
    xi, dxi, P0, xic = load_tm()
    bins = visibility_bins(P0, xi)
    v0 = visibility_prl(P0, xi)
    sig = np.array([blur_std(L, g) for g in GAMMAS])
    curves = [blur_density(P0, dxi, s) for s in sig]

    print(f"X_c(t_m) = {xic:.1f} X_Om = {xic / L.D_over_XOm:.5f} D")
    print(f"X_f = {bins['x_f']:.3f} X_Om    coherent V0 = {v0:.4f} "
          f"(Appendix-C binned {visibility_density(P0, xi, bins):.4f})\n")
    print("  Gamma/Omega   sigma/X_Om   V_PRL    V/V0    Delta_pi")
    for g, s, Pg in zip(GAMMAS, sig, curves):
        v = visibility_prl(Pg, xi)
        print(f"  {g:.1e}      {s:8.4f}  {v:6.4f}  {v / v0:6.4f}  "
              f"{np.sqrt(pi_diffusion_rate(g) * L.tau_m):8.4f} P_Om")

    # ---- (a) ----------------------------------------------------------
    fa, a = plt.subplots(figsize=(5.4, 4.1))
    for g, Pg, col in zip(GAMMAS, curves, GAMMA_COLORS):
        a.plot(xi, 100.0 * Pg, color=col, lw=1.6,
               label=r"$%g\times10^{-8}$" % (g * 1e8))
    bl = visibility_bins(curves[0], xi)      # P_a, P_b sit on the lightest curve
    ja, jb = (int(np.argmin(np.abs(xi - bl[k]))) for k in ("y_max", "y_min"))
    # P_b sits at ~0.14 here, so its label goes ABOVE the point: below would land on the
    # x-axis tick labels.
    _mark(a, [xi[ja], xi[jb]], [100.0 * curves[0][ja], 100.0 * curves[0][jb]],
          dy=(7, 9))
    a.set_xlim(-15, 5)
    a.set_ylim(0, 8)
    a.set_xlabel(r"$\tilde X/X_\Omega$")
    a.set_ylabel(r"$10^2\,P(X)\,X_\Omega$")
    a.set_title("(a)  L set", loc="left", fontsize=10)
    a.legend(title=r"$\Gamma/\Omega$", fontsize=8, title_fontsize=8, loc="upper left")
    fa.tight_layout()
    fa.savefig(os.path.join(FIG, "stage3_fig3a_L.png"), dpi=150)
    plt.close(fa)

    # ---- (b) ----------------------------------------------------------
    xr, R = relative_density(P0, xi, dxi)
    br = visibility_bins(R, xr)
    v_rel = visibility_prl(R, xr)
    ka, kb = (int(np.argmin(np.abs(xr - br[k]))) for k in ("y_max", "y_min"))
    print(f"\n(b) relative coordinate: peak 10^3 P = {1e3 * R.max():.2f} at "
          f"r~ = {xr[np.argmax(R)]:.2f},  X_f = {br['x_f']:.2f} X_Om,  V = {v_rel:.4f}")

    fb, b = plt.subplots(figsize=(5.4, 4.1))
    b.plot(xr, 1e3 * R, color=REL_COLOR, lw=1.6)
    _mark(b, [xr[ka], xr[kb]], [1e3 * R[ka], 1e3 * R[kb]])
    b.set_xlim(-25, 5)
    b.set_ylim(0, 9.5)
    b.set_xlabel(r"$\tilde r/X_\Omega$")
    b.set_ylabel(r"$10^3\,P(\tilde r)\,X_\Omega$")
    b.set_title("(b)  two-particle relative coordinate", loc="left", fontsize=10)
    fb.tight_layout()
    fb.savefig(os.path.join(FIG, "stage3_fig3b_L.png"), dpi=150)
    plt.close(fb)

    # ---- (c) ----------------------------------------------------------
    fc, c = plt.subplots(figsize=(5.4, 4.1))
    print("\n  set   Gamma/Omega spanned by the transition")
    scan = {}
    for name, w, D, ls, col in SETS:
        p = ParamSet(name, xs_over_D=0.1, omega_over_Omega=w, D_over_XOm=D)
        gam = set_sweep(p)
        V = np.array([visibility_prl(blur_density(P0, dxi, blur_std(p, g)), xi) / v0
                      for g in gam])
        c.semilogx(gam, V, ls, color=col, lw=1.6, label=name)
        c.semilogx(gam[::3], V[::3], "o", ms=4, color=col)
        scan[f"gamma_{name}"] = gam
        scan[f"V_{name}"] = V
        print(f"  {name:4s}  {gam[0]:.2e}  ->  {gam[-1]:.2e}   (V {V[0]:.3f} -> {V[-1]:.3f})")
    # r is decoherence free under collective noise (SM Eq. S6), so this is flat in Gamma
    c.axhline(v_rel / v0, color=REL_COLOR, lw=1.6)
    c.annotate(r"$P(\tilde r)$, collective $\Gamma/\Omega$",
               (1.3e-15, v_rel / v0 + 0.03), fontsize=8, color=REL_COLOR)
    c.set_xlim(1e-15, 1e-5)
    c.set_ylim(0, 1.05)
    c.set_xlabel(r"$\Gamma/\Omega$")
    c.set_ylabel("Visibility")
    c.set_title("(c)", loc="left", fontsize=10)
    c.legend(fontsize=9, loc="center left")
    fc.tight_layout()
    fc.savefig(os.path.join(FIG, "stage3_fig3c_L.png"), dpi=150)
    plt.close(fc)

    np.savez(os.path.join(DATA, "stage3_L_visibility.npz"), V0=v0, Xf=bins["x_f"],
             r_xi=xr, r_P=R, V_relative=v_rel, **scan)
    print("\nsaved stage3_fig3a_L.png, stage3_fig3b_L.png, stage3_fig3c_L.png")


if __name__ == "__main__":
    main()
