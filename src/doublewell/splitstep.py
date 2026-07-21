from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from doublewell.units import ParamSet
from doublewell.classical import comoving_frame


@dataclass(frozen=True)
class Grid1D:

    n: int
    half_width: float
    xi_center: float = 0.0
    xi: np.ndarray = field(init=False, repr=False)
    k: np.ndarray = field(init=False, repr=False)
    dxi: float = field(init=False)

    def __post_init__(self) -> None:
        dxi = 2.0 * self.half_width / self.n
        xi = self.xi_center + (np.arange(self.n) - self.n // 2) * dxi
        k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=dxi)
        object.__setattr__(self, "dxi", dxi)
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "k", k)


# ---- states -----------------------------------------------------------------


def gaussian_state(grid: Grid1D, xi0: float = 0.0, k0: float = 0.0,
                   var: float = 1.0) -> np.ndarray:
    psi = np.exp(-((grid.xi - xi0) ** 2) / (4.0 * var) + 1j * k0 * grid.xi)
    return normalize(psi, grid)


def norm(psi: np.ndarray, grid: Grid1D) -> float:
    return float(np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dxi))


def normalize(psi: np.ndarray, grid: Grid1D) -> np.ndarray:
    return psi / norm(psi, grid)


def expect_xi(psi: np.ndarray, grid: Grid1D, power: int = 1) -> float:
    prob = np.abs(psi) ** 2
    return float(np.sum(grid.xi ** power * prob) / np.sum(prob))


def var_xi(psi: np.ndarray, grid: Grid1D) -> float:
    m1 = expect_xi(psi, grid, 1)
    return expect_xi(psi, grid, 2) - m1 ** 2


def expect_k(psi: np.ndarray, grid: Grid1D, power: int = 1) -> float:
    prob = np.abs(np.fft.fft(psi)) ** 2
    return float(np.sum(grid.k ** power * prob) / np.sum(prob))


def energy(psi: np.ndarray, grid: Grid1D, v: np.ndarray) -> float:
    psi = normalize(psi, grid)
    prob_k = np.abs(np.fft.fft(psi)) ** 2
    kinetic = float(np.sum(grid.k ** 2 * prob_k) / np.sum(prob_k))
    potential = float(np.sum(v * np.abs(psi) ** 2) * grid.dxi)
    return kinetic + potential


def propagate(psi: np.ndarray, grid: Grid1D, v: np.ndarray, dtau: float,
              nsteps: int, callback=None, callback_every: int = 0) -> np.ndarray:
    half_kick = np.exp(-0.5j * v * dtau)
    drift = np.exp(-1j * grid.k ** 2 * dtau)
    psi = psi.astype(np.complex128, copy=True)
    for step in range(1, nsteps + 1):
        psi *= half_kick
        psi = np.fft.ifft(drift * np.fft.fft(psi))
        psi *= half_kick
        if callback is not None and callback_every and (
                step % callback_every == 0 or step == nsteps):
            callback(psi, step)
    return psi


def residual_potential(params: ParamSet, xi: np.ndarray, xi_c: float) -> np.ndarray:
    r2 = params.omega_over_Omega ** 2
    a = -0.25 * r2
    b = 0.125 * r2 / params.D_over_XOm ** 2
    return (a + 6.0 * b * xi_c ** 2) * xi ** 2 + 4.0 * b * xi_c * xi ** 3 + b * xi ** 4


def propagate_comoving(phi: np.ndarray, grid: Grid1D, params: ParamSet,
                       tau_end: float, nsteps: int, callback=None,
                       callback_every: int = 0) -> np.ndarray:
    dtau = tau_end / nsteps
    tau_mid = (np.arange(nsteps) + 0.5) * dtau
    xi_c_mid, _ = comoving_frame(params, tau_mid)
    drift = np.exp(-1j * grid.k ** 2 * dtau)
    phi = phi.astype(np.complex128, copy=True)
    for step in range(1, nsteps + 1):
        half_kick = np.exp(-0.5j * residual_potential(params, grid.xi,
                                                       xi_c_mid[step - 1]) * dtau)
        phi *= half_kick
        phi = np.fft.ifft(drift * np.fft.fft(phi))
        phi *= half_kick
        if callback is not None and callback_every and (
                step % callback_every == 0 or step == nsteps):
            callback(phi, step)
    return phi
