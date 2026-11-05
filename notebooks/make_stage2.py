import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from doublewell.units import L
from doublewell.observables import cumulants, cubic_phase_thetas
from doublewell.classical import thermal_initial_conditions, evolve_ensemble

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")


class _Grid:
    def __init__(self, xi):
        self.xi = xi


def main():
    d = np.load(os.path.join(DATA, "L_fig2ac.npz"))
    xi, fracs, snaps = d["xi"], d["snap_fracs"], d["snaps"]
    j = int(np.argmin(np.abs(fracs - 1.0)))
    phi, xic_tm = snaps[j], float(d["snap_xic"][j])
    dxi = xi[1] - xi[0]
    P = np.abs(phi) ** 2
    P = P / (P.sum() * dxi)

    m1, m2, m3 = cumulants(phi, _Grid(xi))
    th1, th2, th3 = cubic_phase_thetas(m1, m2, m3)
    print(f"cumulants: mean={m1:.2f}  var={m2:.1f} (std={np.sqrt(m2):.1f})  "
          f"kappa3={m3:.3e}  skew={m3/m2**1.5:.3f}")
    print(f"cubic-phase thetas (X_Om units): theta1={th1:.2f}  theta2={th2:.1f}  theta3={th3:.3e}")

    imax = int(np.argmax(P))
    pk, _ = find_peaks(P, height=0.03 * P[imax], prominence=0.02 * P[imax])
    others = pk[pk != imax]
    Xf = float(abs(xi[imax] - xi[others[np.argmin(np.abs(others - imax))]]))
    print(f"first-fringe spacing X_f/X_Om = {Xf:.2f}  (target ~2.5, exit +-10%)")

    cache = os.path.join(DATA, "L_classical_tm.npz")
    if os.path.exists(cache):
        xi_cl = np.load(cache)["xi_cl"]
    else:
        rng = np.random.default_rng(0)
        u0, v0 = thermal_initial_conditions(L, 200_000, rng)
        u_tm, _ = evolve_ensemble(u0, v0, L.omega_tm, dtau=5e-3)
        xi_cl = L.D_over_XOm * u_tm - xic_tm
        np.savez(cache, xi_cl=xi_cl)
    print(f"classical spread at t_m: xi_cl in [{xi_cl.min():.0f}, {xi_cl.max():.0f}] X_Om")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (lo, hi) in ((a1, (xi[imax] - 420, xi[imax] + 40)),
                         (a2, (xi[imax] - 42, xi[imax] + 10))):
        ax.plot(xi, P, color="#1f6fb2", lw=0.8, label=r"$p_1(X)$ quantum")
        m = (xi_cl > lo) & (xi_cl < hi)
        nb = 130 if ax is a1 else 30
        ax.hist(xi_cl[m], bins=nb, range=(lo, hi), density=True,
                histtype="step", color="#e07b00", lw=1.6, label=r"$p_0(X)$ classical MC")
        ax.set_xlim(lo, hi); ax.set_xlabel(r"$\tilde X/X_\Omega$")
    a1.set_ylabel(r"$P(X, t_m)$"); a1.set_title("cubic-phase interference vs classical envelope")
    a1.legend(fontsize=8)
    a2.set_title(f"zoom: fringes $X_f \\approx {Xf:.1f}\\,X_\\Omega$ on a common envelope")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "stage2_L.png"), dpi=150)
    plt.close(fig)
    print("saved stage2_L.png")


if __name__ == "__main__":
    main()
