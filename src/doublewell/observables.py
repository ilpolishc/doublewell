import numpy as np
from scipy.integrate import solve_ivp

from doublewell.splitstep import Grid1D
from doublewell.classical import trajectory


def _prob_x(psi):
    p = np.abs(psi) ** 2
    return p / p.sum()


def covariance(psi: np.ndarray, grid: Grid1D):
    px = _prob_x(psi)
    xi = grid.xi
    mx = float((xi * px).sum())
    Vxx = float((xi * xi * px).sum()) - mx * mx

    fk = np.fft.fft(psi)
    pk = np.abs(fk) ** 2
    pk = pk / pk.sum()
    mk = float((grid.k * pk).sum())
    Vkk = float((grid.k * grid.k * pk).sum()) - mk * mk

    dpsi = np.fft.ifft(1j * grid.k * fk)
    nrm = float((np.abs(psi) ** 2).sum() * grid.dxi)
    sym = float(np.real(np.sum(np.conj(psi) * xi * (-2j) * dpsi)) * grid.dxi / nrm)
    mp = 2.0 * mk
    Vxp = sym - mx * mp
    return mx, mp, Vxx, 4.0 * Vkk, Vxp


def squeezing_eigs(psi: np.ndarray, grid: Grid1D):
    _, _, Vxx, Vpp, Vxp = covariance(psi, grid)
    tr = Vxx + Vpp
    disc = np.sqrt((Vxx - Vpp) ** 2 + 4.0 * Vxp ** 2)
    return 0.5 * (tr - disc), 0.5 * (tr + disc)


def squeezing_dB(psi: np.ndarray, grid: Grid1D):
    lam_m, lam_p = squeezing_eigs(psi, grid)
    return -10.0 * np.log10(lam_m), 10.0 * np.log10(lam_p)


def delta_x(psi: np.ndarray, grid: Grid1D) -> float:
    xi = grid.xi
    px = _prob_x(psi)
    mx = float((xi * px).sum())
    return float(np.sqrt((xi * xi * px).sum() - mx * mx))

def evolve_covariance(c2, tau_grid, sigma0=(1.0, 0.0, 1.0)):
    def rhs(tau, y):
        Sxx, Sxp, Spp = y
        c = c2(tau)
        return [2.0 * Sxp, Spp - 4.0 * c * Sxx, -8.0 * c * Sxp]

    sol = solve_ivp(rhs, (0.0, tau_grid[-1]), list(sigma0), t_eval=tau_grid,
                    rtol=1e-10, atol=1e-12)
    return sol.y[0], sol.y[1], sol.y[2]


def covariance_quadratic(params, tau_grid, nbar=0.0):
    a = -0.25 * params.omega_over_Omega ** 2
    b = 0.125 * params.omega_over_Omega ** 2 / params.D_over_XOm ** 2
    tau_grid = np.asarray(tau_grid, dtype=float)
    tw_max = params.omega_over_Omega * tau_grid[-1] * 1.001 + 1e-6
    sol = trajectory(params.xs_over_D, tau_max=tw_max)

    def c2(tau):
        u = sol.sol(params.omega_over_Omega * tau)[0]
        return a + 6.0 * b * (params.D_over_XOm * u) ** 2

    s = 2.0 * nbar + 1.0
    return evolve_covariance(c2, tau_grid, sigma0=(s, 0.0, s))


def squeezing_quadratic(params, tau_grid, nbar=0.0):
    Sxx, Sxp, Spp = covariance_quadratic(params, tau_grid, nbar)
    tr = Sxx + Spp
    disc = np.sqrt((Sxx - Spp) ** 2 + 4.0 * Sxp ** 2)
    return 0.5 * (tr - disc), 0.5 * (tr + disc)

def momentum_std_max(params, n_tau=4000):
    tau = np.linspace(0.0, 2.0 * params.tau_m, n_tau)
    Spp = covariance_quadratic(params, tau)[2]
    return float(np.sqrt(Spp).max())


def required_n(params, half_width, headroom=4.0):
    n_min = headroom * momentum_std_max(params) * half_width / np.pi
    return int(2 ** int(np.ceil(np.log2(n_min))))


def check_resolution(params, n, half_width, headroom=4.0):
    pi_max = np.pi * n / half_width               # = 2 pi / dxi, dxi = 2 half_width / n
    need = headroom * momentum_std_max(params)
    if pi_max < need:
        raise ValueError(
            f"{params.name}: momentum Nyquist pi_max={pi_max:.1f} P_Om < {headroom}x "
            f"Delta_pi_max={need:.1f}; need n>={required_n(params, half_width, headroom)} "
            f"at half_width={half_width} (have n={n}).")
    return pi_max, need
