import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")        # .png
DATA = os.path.join(ROOT, "data")          # .npz

NBAR_TICKS = np.array([0, 20, 40, 60, 80])


def main():
    s = np.load(os.path.join(DATA, "stage3d_L_sigma_s_scan.npz"))
    n = np.load(os.path.join(DATA, "stage3d_L_nbar_scan.npz"))
    sig, Vs = s["sigma_s"], s["V"] / s["V0"]
    nbar, Vn = n["nbar"], n["V_over_V0"]
    sig_of_nbar = np.sqrt(2.0 * nbar)

    print("Fig. 3d overlay: the two curves are independent calculations")
    print("  n-bar   sigma=sqrt(2n)   V(n-bar)   V(sigma_s) there   ratio")
    for nb, sx, v in zip(nbar, sig_of_nbar, Vn):
        if sx > sig.max():
            continue
        vi = np.interp(sx, sig, Vs)
        print(f"  {nb:5.1f}   {sx:12.2f}   {v:8.4f}   {vi:16.4f}   {v / vi:6.3f}"
              if vi > 1e-6 else f"  {nb:5.1f}   {sx:12.2f}   {v:8.4f}   {vi:16.4f}")

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(sig, Vs, "-", color="#1f6fb2", lw=2,
            label=r"initial position imprecision $\sigma_s$")
    ax.plot(sig_of_nbar, Vn, "--", color="#c0392b", lw=2,
            label=r"initial occupation $\bar n$")
    ax.plot(sig_of_nbar, Vn, "o", ms=5, mfc="none", mec="#c0392b", mew=1.3)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\sigma_{\rm s}/X_\Omega$")
    ax.set_ylabel("Visibility of the first minimum,  $V/V_0$")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, lw=0.5)

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(np.sqrt(2.0 * NBAR_TICKS))
    top.set_xticklabels([str(t) for t in NBAR_TICKS])
    top.set_xlabel(r"$\bar n$", color="#c0392b")
    top.tick_params(axis="x", colors="#c0392b")

    ax.axvline(np.sqrt(80.0), color="0.6", ls=":", lw=1.0)
    ax.annotate(r"PRL: $\sigma_s\sim10\,X_\Omega \equiv \bar n\sim40$",
                xy=(np.sqrt(80.0), 0.57), fontsize=8, rotation=90,
                ha="right", va="bottom", color="0.35")

    fig.suptitle("L set. PRL Fig. 3d", y=0.98, fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG, "stage3_fig3d_L.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved {os.path.basename(out)}")


if __name__ == "__main__":
    main()
