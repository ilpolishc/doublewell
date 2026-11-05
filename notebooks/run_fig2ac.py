import os
import numpy as np
from scipy.integrate import quad

from doublewell.units import L
from doublewell.splitstep import (Grid1D, gaussian_state, norm, residual_potential,
                                   expect_xi, expect_k)
from doublewell.observables import delta_x, check_resolution
from doublewell.classical import comoving_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "L_fig2ac.npz")
CKPT = os.path.join(DATA, "L_fig2ac.ckpt.npz")

PARAMS, N, HW, DTAU = L, 131072, 4500.0, 1.5e-3
CHUNK = 20000


def _t_of_u(u_target):
    u0 = 0.1
    e = -0.5 * u0 ** 2 + 0.25 * u0 ** 4
    t = quad(lambda u: 1.0 / np.sqrt(u ** 2 - 0.5 * u ** 4 + 2 * e), u0, u_target, limit=200)[0]
    return t / PARAMS.omega_tm


_TD = _t_of_u(1.0)
SNAP_FRACS = (0.0, _t_of_u(0.6), _TD, 1.0, 2.0 - _TD, 2.0)


def _save_atomic(ckpt, phi, step, rec, snaps):
    tmp = ckpt[:-4] + ".tmp"
    np.savez(tmp, phi=phi, step=step,
             t=rec["t"], X=rec["X"], P=rec["P"], dX=rec["dX"],
             sf=snaps["f"], sp=snaps["phi"], sx=snaps["xic"], sk=snaps["kc"])
    os.replace(tmp + ".npz", ckpt)


def run(params=PARAMS, n=N, hw=HW, dtau=DTAU, chunk=CHUNK, out=OUT, ckpt=CKPT,
        stop_after=None, tau_end=None):

    check_resolution(params, n, hw)
    tm = params.tau_m
    if tau_end is None:
        tau_end = 2.0 * tm + 1.0
    nsteps = int(round(tau_end / dtau))
    com = Grid1D(n=n, half_width=hw)
    dtau = tau_end / nsteps
    xic_mid = comoving_frame(params, (np.arange(nsteps) + 0.5) * dtau)[0]
    xic_end, kc_end = comoving_frame(params, (np.arange(nsteps) + 1) * dtau)
    drift = np.exp(-1j * com.k ** 2 * dtau)
    every = max(1, nsteps // 5000)
    snap_steps = {int(round(f * tm / dtau)): f for f in SNAP_FRACS}

    if os.path.exists(ckpt):
        with np.load(ckpt) as d:
            phi = d["phi"].astype(np.complex128)
            step0 = int(d["step"])
            rec = {k: list(d[k]) for k in ("t", "X", "P", "dX")}
            snaps = {"f": list(d["sf"]), "phi": list(d["sp"]),
                     "xic": list(d["sx"]), "kc": list(d["sk"])}
        print(f"resuming from step {step0}/{nsteps}", flush=True)
    else:
        phi = gaussian_state(com, xi0=0.0).astype(np.complex128)
        step0 = 0
        rec = {"t": [], "X": [], "P": [], "dX": []}
        snaps = {"f": [], "phi": [], "xic": [], "kc": []}
        if 0 in snap_steps:                         # t = 0 ground state
            snaps["f"].append(0.0); snaps["phi"].append(phi.copy())
            snaps["xic"].append(float(xic_end[0])); snaps["kc"].append(float(kc_end[0]))

    for s in range(step0 + 1, nsteps + 1):
        hk = np.exp(-0.5j * residual_potential(params, com.xi, xic_mid[s - 1]) * dtau)
        phi *= hk; phi = np.fft.ifft(drift * np.fft.fft(phi)); phi *= hk
        if s % every == 0 or s == nsteps:
            rec["t"].append(s * dtau)
            rec["X"].append(float(xic_end[s - 1]) + expect_xi(phi, com))
            rec["P"].append(2.0 * float(kc_end[s - 1]) + 2.0 * expect_k(phi, com))
            rec["dX"].append(delta_x(phi, com))
        if s in snap_steps:
            snaps["f"].append(snap_steps[s]); snaps["phi"].append(phi.copy())
            snaps["xic"].append(float(xic_end[s - 1])); snaps["kc"].append(float(kc_end[s - 1]))
        if s % chunk == 0 or s == stop_after:
            _save_atomic(ckpt, phi, s, rec, snaps)
            print(f"progress {100*s//nsteps:3d}%  tau={s*dtau:7.1f}  "
                  f"dX/eta={rec['dX'][-1]/params.eta:.3f}  norm-1={norm(phi,com)-1:.1e}", flush=True)
            if s == stop_after:
                return

    order = np.argsort(snaps["f"])
    np.savez(out, xi=com.xi, dxi=com.dxi, tm=tm, eta=params.eta, D=params.D_over_XOm,
             tau=np.array(rec["t"]), X=np.array(rec["X"]), P=np.array(rec["P"]),
             dX=np.array(rec["dX"]),
             snap_fracs=np.array(snaps["f"])[order],
             snaps=np.array(snaps["phi"])[order],
             snap_xic=np.array(snaps["xic"])[order],
             snap_kc=np.array(snaps["kc"])[order])

    orange = out[:-4] + "_orange.npz" if out.endswith(".npz") else out + "_orange.npz"
    if out == OUT:
        orange = os.path.join(DATA, f"{params.name}_orange.npz")
    np.savez(orange,
             tau=np.array(rec["t"]), dX=np.array(rec["dX"]), tm=tm, eta=params.eta)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"saved {out} and {os.path.basename(orange)}")


CONVERGENCE = {
    "box":  dict(n=262144, hw=9000.0, dtau=1.5e-3),
    "step": dict(n=131072, hw=4500.0, dtau=0.75e-3),
}


def run_convergence(which):
    cfg = CONVERGENCE[which]
    out = os.path.join(DATA, f"L_fig2ac_conv_{which}.npz")
    run(out=out, ckpt=out[:-4] + ".ckpt.npz", tau_end=PARAMS.tau_m, **cfg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in CONVERGENCE:
        run_convergence(sys.argv[1])
    else:
        run()
