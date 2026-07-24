from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class ParamSet:
    name: str
    xs_over_D: float
    omega_over_Omega: float
    D_over_XOm: float
    nbar: float = 0.0

    @property
    def xi_s(self) -> float:
        return self.xs_over_D * self.D_over_XOm

    @property
    def eta(self) -> float:
        return 1.0 / (SQRT2 * self.omega_over_Omega * self.xs_over_D)

    @property
    def S0_dB(self) -> float:
        return 20.0 * math.log10(self.eta)

    @property
    def omega_tm_asymptotic(self) -> float:
        return math.log(4.0 * SQRT2 / self.xs_over_D)

    @property
    def omega_tm(self) -> float:
        from doublewell.classical import half_period_quad
        return half_period_quad(self.xs_over_D)

    @property
    def tau_m(self) -> float:
        return self.omega_tm / self.omega_over_Omega

    def v_dw(self, xi: np.ndarray) -> np.ndarray:
        r2 = self.omega_over_Omega ** 2
        quart = (xi / self.D_over_XOm) ** 2
        return 0.25 * r2 * xi ** 2 * (0.5 * quart - 1.0)

    def thermal_variance(self) -> float:
        return 2.0 * self.nbar + 1.0


def v_prep(xi: np.ndarray) -> np.ndarray:
    return 0.25 * xi ** 2


L = ParamSet(name="L", xs_over_D=0.1, omega_over_Omega=1e-2, D_over_XOm=1e4)

TOY_M = ParamSet(name="toy-M", xs_over_D=0.1, omega_over_Omega=0.1, D_over_XOm=1000.0)
TOY_S = ParamSet(name="toy-S", xs_over_D=0.1, omega_over_Omega=0.1, D_over_XOm=100.0)
