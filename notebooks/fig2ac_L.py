import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from doublewell.units import L
from doublewell.wigner import wigner
from doublewell.classical import comoving_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(DATA, "L_fig2ac.npz")

MOM = L.D_over_XOm * L.omega_over_Omega
MARKERS = ["o", "h", "^", "*", "s", "o"]
LABELS = ["$t=0$", "", "$t=t_D$", "$t=t_m$", "", "$t=2t_m$"]

LABEL_POS = [(7, 7, "left", "bottom"), None, (0, -15, "center", "top"),
             (-11, 0, "right", "center"), None, (0, -15, "center", "top")]

XLIM = [(-10, 10), (-1000, 1000), (-1000, 1000), (-15, 5), (-1000, 1000), (-10, 10)]

CLOCKWISE = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]

VMAX = 1.0e-2


def fig2a(d):
    tm = float(d["tm"])
    tau = np.linspace(0, 2 * tm, 4000)
    xic, kc = comoving_frame(L, tau)

    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    ax.plot(xic / L.D_over_XOm, 2 * kc / MOM, "--", color="0.45", lw=1.3,
            label="classical")
    ax.plot(d["X"] / L.D_over_XOm, d["P"] / MOM, "-", color="#1f6fb2", lw=1.6,
            label=r"$\langle\hat X\rangle,\langle\hat P\rangle$")
    for f, mk, lab, pos in zip(d["snap_fracs"], MARKERS, LABELS, LABEL_POS):
        i = int(np.argmin(np.abs(d["tau"] - f * tm)))
        x, y = d["X"][i] / L.D_over_XOm, d["P"][i] / MOM
        ax.plot(x, y, mk, mfc="none", mec="0.2", mew=1.4, ms=9)
        if lab:
            dx, dy, ha, va = pos
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va=va, fontsize=9)
    ax.set_xlabel(r"$\langle\hat X\rangle / D$")
    ax.set_ylabel(r"$\langle\hat P\rangle / (m\omega D)$")
    ax.set_title("(a)  L set", loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "stage1_fig2a_L.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  <X>/D at 2t_m = {d['X'][-1] / L.D_over_XOm:.5f}  "
          f"(started at {d['X'][0] / L.D_over_XOm:.5f})")
    return out


def fig2c(d):
    xi, dxi = d["xi"], float(d["dxi"])
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2))

    Ws = []
    for phi, (lo, hi) in zip(d["snaps"], XLIM):
        Ws.append(wigner(phi, dxi, xi=xi, center=0.5 * (lo + hi), half=0.5 * (hi - lo),
                         nx=261, pi_max=11.0, n_pi=221, decimate=1))
    vmax = VMAX
    print(f"  shared colour scale +-{vmax:.0e}; |W| peaks per panel:")
    for f, (_, _, W) in zip(d["snap_fracs"], Ws):
        print(f"    t/t_m={f:.3f}  |W|max={np.abs(W).max():.4f}  "
              f"saturates {np.abs(W).max() / vmax:5.1f}x")

    for cell, (xo, pi, W), f, mk, (lo, hi) in zip(CLOCKWISE, Ws, d["snap_fracs"],
                                                  MARKERS, XLIM):
        ax = axes[cell]
        im = ax.pcolormesh(xo, pi, W.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="gouraud")
        ax.plot(0.09, 0.86, marker=mk, transform=ax.transAxes, mfc="none", mec="k",
                mew=1.3, ms=10)
        ax.set_xlabel(r"$\tilde X/X_\Omega$", fontsize=8)
        ax.set_ylabel(r"$\tilde P/P_\Omega$", fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-10, 10)
        ax.tick_params(labelsize=7)
        print(f"  cell {cell}  t/t_m = {f:.3f}  marker {mk}  window ({lo}, {hi})")
    fig.suptitle("(c)  L set, ordered clockwise", fontsize=11)
    fig.tight_layout(rect=[0.11, 0, 1, 1])
    cax = fig.add_axes([0.035, 0.13, 0.016, 0.72])
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb = fig.colorbar(im, cax=cax, format=fmt, extend="both")
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cb.set_label(r"$W(\tilde X,\tilde P)\;\;[X_\Omega P_\Omega]^{-1}$", fontsize=9)
    cax.tick_params(labelsize=7)
    cax.yaxis.get_offset_text().set_fontsize(7)
    out = os.path.join(FIG, "stage1_fig2c_L.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    d = np.load(SRC)
    a = fig2a(d)
    print()
    c = fig2c(d)
    print(f"\nsaved {os.path.basename(a)} and {os.path.basename(c)}")
