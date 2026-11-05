# Discrepancy log

Every mismatch with the papers or the project guide, and every question they do not answer.
## D1. `ω t_m`: exact 4.0469 vs the paper's `ln(4√2 D/X_s) = 4.0355`. resolved. 31.07.2026

A 0.28% difference, because asymptotic formula is used. `units.ParamSet.omega_tm` uses
the exact quadrature; `omega_tm_asymptotic` keeps the paper's closed form for comparison.
All production runs use the exact value.
## D2 Wavepacket touches the box edge at `t_D`. Resolved. Nothing interesting. 31.07.2026


At `t/t_m = 0.782` (`⟨X⟩ = D`, where `ΔX` peaks at `η`) the production L run has
`|ψ|² = 2.3×10⁻⁶` of peak **at the grid edge** `ξ = ±4500`. Every other snapshot is clean
(`10⁻¹³` or below; `10⁻²⁹` at `t_m`). 

Convergence runs were launched: doubled box (`n = 262144`, `half_width = 9000`, same `dξ`) and
halved step (`dτ = 0.75e-3`), both to `t_m`, both resumable. Acceptance: `X_f`, the binned
visibility, and the first three cumulants of `P(X,t_m)` unchanged to plotting precision.

Resolved! Doubling the box at fixed `dτ` (so the test isolates aliasing from time-step
error) leaves the `t_m` pattern untouched: `max|ΔP|/max P = 1.3e-5`, contrast ratio
1.000000, `V = 0.63530` and `X_f = 2.4719` in both. The `2.3e-6`-of-peak edge amplitude at
`t_D` does not propagate into the observable. Only `κ₃` moves (1.3%), which is the wider box
legitimately holding more of the `ξ³`-weighted far tail, not aliasing.


## D3. `σ_s` panel reproduced at L. resolved. 31.07.2026

Full quantum propagation of 31 `ξ₀` nodes at L run_stage3d_quantum.py,
reweighted to each `σ_s`:

| `σ_s/X_Ω` | binned `V/V₀` | fringe contrast at `k₀` | rigid-translation prediction |
|---|---|---|---|
| 1 | 0.975 | 0.984 | 0.984 |
| 3 | 0.795 | 0.867 | 0.864 |
| 5 | 0.511 | 0.670 | 0.666 |
| 7 | 0.230 | 0.455 | 0.451 |
| **10** | −0.050 (floor) | 0.203| 0.197 |
| 14 | −0.184 | 0.047 | 0.042 |

At the PRL's quoted `σ_s ~ 10 X_Ω` the fringes retain 20% of their coherent contrast, which is atolerance limit rather than a comfortable margin, which is how the paper phrases it. The
binned `V` has already hit its negative floor there (D4), so the envelope-free contrast is
the meaningful column.

This undermines the rigid-translation model at L specifically, agreement is 0.0–0.9%
for `σ_s ≤ 7` and 3% at `σ_s = 10`, even though the same model is badly wrong at toy-M. The
difference is that `σ_s` is an absolute imprecision, so it perturbs the initial position by
`σ_s/(0.1 D)`: 1% at L, 10% at toy-M. Per-node residuals against a rigid shift are 0.18% at
`ξ₀ = 2`, 0.6% at 8, 1.5% at 19, 6.4% at 38, 26% at 80.


## D4. Route B reproduces the paper's `Γη²/ω` scaling. resolved. 2026-09-03.09.2026

Independent check of the decoherence machinery against a number in the PRL text, "the effect of decoherence scales with `Γη²/ω`". With `σ² = 4(Γ/Ω)·I`, `I = ∫G²dτ`, the
ratio `I / [η²/(ω/Ω)]` must be set-independent

| set | `η` | `ω/Ω` | `I/[η²/(ω/Ω)]` |
|---|---|---|---|
| L | 707.1 | 1e-2 | 6.6045e-03 |
| toy-M | 70.7 | 1e-1 | 6.6045e-03 |
| toy-S | 70.7 | 1e-1 | 6.6045e-03 |
| XL | 7071.1 | 1e-3 | 6.6045e-03 |

Identical to five significant figures across four decades of `η²/ω`. The constant does
depend on `X_s/D`, which is `10⁻¹` for the whole Table-I parameter sets. The Green's function is
therefore right in structure.

## D5. Fig. 3b was not initially considered based on a cost estimate that was wrong. Resolved.05.09.2026

Fig. 3b and the
horizontal line in Fig. 3c were not initially considered. It was assumed the two-particle
case meant propagating a second particle. It does not.

From the Supplemental Material: two non-interacting particles with mirror-symmetric initial
conditions keep `P_tot,coh` factorised at all times, so with `X̃₂` distributed as `−X̃₁` the
relative coordinate `r̃ = X̃₁ − X̃₂` is the sum of two independent draws from the
single-particle residual distribution —


`P(r̃) = P_coh ∗ P_coh`


whic is a self-convolution of a pattern already done at hte time it is written. Just one FFT without propagation. And the
collective dissipator of Eq. (S3) couples to `X̂₁ + X̂₂` with the rank-1 covariance
`σ_n²[[1,1],[1,1]]` of Eq. (S5), which lies wholly in the centre of mass, so `r` is
decoherence-free (Eq. S6), which is why the 3c line is flat in `Γ`

It reproduces:

| | this work | paper |
|---|---|---|
| peak `10³ P(r̃) X_Ω` | 8.86 at `r̃ = −2.06` | ~8–9 near `r̃ ≈ −1` |
| visibility of `P(r̃)` | **0.172** | horizontal line at ~0.18 |

The second is a real cross-check on the single-particle result too. in a sense

