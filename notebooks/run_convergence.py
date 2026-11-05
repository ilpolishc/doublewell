import os
import sys
import time

import numpy as np

from doublewell.units import L, TOY_M
from doublewell.splitstep import Grid1D, gaussian_state, propagate_comoving, norm
from doublewell.observables import check_resolution, cumulants, c2_along_orbit
from doublewell.decoherence import (visibility_bins, visibility_density,
                                    fringe_contrast_ratio)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

DTAU_CASES = [
    (L,     dict(n=131072, half_width=4500.0), 1.5e-3, (2.0, 1.0, 0.25, 0.05)),
    (TOY_M, dict(n=16384,  half_width=450.0),  2.0e-3, (0.25, 0.1, 0.05, 0.02)),
]
BOX_CASE = (L, dict(n=131072, half_width=4500.0), dict(n=262144, half_width=9000.0), 0.25)


class _G:
    def __init__(self, xi):
        self.xi = xi


def _pattern(params, n, half_width, dtau):
    check_resolution(params, n, half_width)
    com = Grid1D(n=n, half_width=half_width)
    nsteps = max(1, int(round(params.tau_m / dtau)))
    t0 = time.perf_counter()
    phi = propagate_comoving(gaussian_state(com, xi0=0.0), com, params, params.tau_m, nsteps)
    P = np.abs(phi) ** 2
    return com, P / (P.sum() * com.dxi), nsteps, time.perf_counter() - t0, norm(phi, com) - 1


def stage_dtau():
    for params, grid, ref_dtau, dtaus in DTAU_CASES:
        c2 = c2_along_orbit(params, params.tau_m)(params.tau_m)
        com, Pref, nref, tref, _ = _pattern(params, dtau=ref_dtau, **grid)
        bins = visibility_bins(Pref, com.xi)
        k0 = 2.0 * np.pi / bins["x_f"]
        print(f"\n{params.name}: c2(t_m)={c2:.4e} -> residual timescale "
              f"1/sqrt(4 c2) = {1 / np.sqrt(4 * abs(c2)):.1f} in tau")
        print(f"  reference dtau={ref_dtau:.1e} ({nref} steps, {tref:.0f}s): "
              f"V={visibility_density(Pref, com.xi, bins):.5f}  X_f={bins['x_f']:.4f}")
        print("    dtau   nsteps    wall     norm-1        V        X_f    max|dP|/maxP  contrast")
        for dtau in dtaus:
            _, P, nst, wall, nerr = _pattern(params, dtau=dtau, **grid)
            print(f"  {dtau:7.3f} {nst:7d} {wall:7.1f}s {nerr:10.1e}  "
                  f"{visibility_density(P, com.xi, bins):.5f}  "
                  f"{visibility_bins(P, com.xi)['x_f']:.4f}  "
                  f"{np.abs(P - Pref).max() / Pref.max():11.5f}  "
                  f"{fringe_contrast_ratio(P, com.xi, Pref, com.xi, k0):9.6f}", flush=True)


def stage_box():
    params, base, wide, dtau = BOX_CASE
    print(f"\n{params.name}: box doubling at fixed dtau={dtau} (isolates aliasing from "
          f"time-step error; dxi is unchanged)")
    cb, Pb, _, tb, nb = _pattern(params, dtau=dtau, **base)
    cw, Pw, _, tw, nw = _pattern(params, dtau=dtau, **wide)
    Pw = np.interp(cb.xi, cw.xi, Pw)
    Pw /= Pw.sum() * cb.dxi
    bins = visibility_bins(Pb, cb.xi)
    k0 = 2.0 * np.pi / bins["x_f"]
    for name, P, cfg, wall, nerr in (("base", Pb, base, tb, nb), ("wide", Pw, wide, tw, nw)):
        m1, m2, m3 = cumulants(np.sqrt(P), _G(cb.xi))
        print(f"  {name}  n={cfg['n']:7d} hw={cfg['half_width']:6.0f} {wall:5.0f}s "
              f"norm-1={nerr:8.1e}  V={visibility_density(P, cb.xi, bins):.5f}  "
              f"X_f={bins['x_f'] if name == 'base' else visibility_bins(P, cb.xi)['x_f']:.4f}  "
              f"mean={m1:9.3f}  var={m2:10.2f}  kappa3={m3:.4e}")
    print(f"  max|P_wide - P_base|/max = {np.abs(Pw - Pb).max() / Pb.max():.6f}"
          f"   contrast ratio = {fringe_contrast_ratio(Pw, cb.xi, Pb, cb.xi, k0):.6f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("dtau", "all"):
        stage_dtau()
    if stage in ("box", "all"):
        stage_box()
