import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doublewell.units import ParamSet
from doublewell.observables import covariance_quadratic, squeezing_quadratic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")

SETS = [("XXL", 1e-4, 1e8, "-"), ("XL", 1e-3, 1e6, "--"), ("L", 1e-2, 1e4, ":")]
PINK, BLUE = "#d6246e", "#1f6fb2"
N_TAU = 20001


def main(full=False):
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    print("  set   S/S0 peak   S/S0 at t_m   dX/eta peak   t_D/t_m")
    for name, w, D, ls in SETS:
        p = ParamSet(name, xs_over_D=0.1, omega_over_Omega=w, D_over_XOm=D)
        tau = np.linspace(0.0, 2.0 * p.tau_m, N_TAU)
        Sxx = covariance_quadratic(p, tau)[0]
        lam_p = squeezing_quadratic(p, tau)[1]
        SS = np.log10(lam_p) / (2.0 * np.log10(p.eta))
        dX = np.sqrt(Sxx) / p.eta
        t = tau / p.tau_m
        ax.plot(t, SS, ls, color=PINK, lw=1.6)
        ax.plot(t, dX, ls, color=BLUE, lw=1.6, label=name)
        i = int(np.argmin(np.abs(tau - p.tau_m)))
        print(f"  {name:4s}  {SS.max():9.5f}   {SS[i]:11.4f}   {dX.max():11.5f}   "
              f"{t[int(np.argmax(dX))]:7.4f}")

    if full:
        src = os.path.join(DATA, "L_orange.npz")
        if os.path.exists(src):
            d = np.load(src)
            ax.plot(d["tau"] / float(d["tm"]), d["dX"] / float(d["eta"]), "-",
                    color="#e07b00", lw=1.4, alpha=0.9,
                    label="L, full state (not in Fig. 2b)")


    p = ParamSet("L", xs_over_D=0.1, omega_over_Omega=1e-2, D_over_XOm=1e4)
    tau = np.linspace(0.0, 2.0 * p.tau_m, N_TAU)
    tD = tau[int(np.argmax(np.sqrt(covariance_quadratic(p, tau)[0])))] / p.tau_m
    for v in (tD, 2.0 - tD):
        ax.axvline(v, color="0.75", ls=":", lw=0.8)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$t/t_{\rm m}$")

    ax.set_ylabel("")
    ax.text(-0.135, 0.76, r"$\mathcal{S}/\mathcal{S}_0$", transform=ax.transAxes,
            color=PINK, fontsize=13, rotation=90, ha="center", va="center")
    ax.text(-0.135, 0.28, r"$\Delta X/\eta X_\Omega$", transform=ax.transAxes,
            color=BLUE, fontsize=13, rotation=90, ha="center", va="center")
    ax.set_title("(b)", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", title="Table I", title_fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(left=0.20)
    out = os.path.join(FIG, "stage1_fig2b.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved {os.path.basename(out)}")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
