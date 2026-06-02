"""
model.py
--------
Система "одна жертва – два хижаки" з функціональною відповіддю Холлінга II

  dN/dt  = r·N·(1 - N/K) - a₁·P₁·N/(1 + a₁·h₁·N) - a₂·P₂·N/(1 + a₂·h₂·N)
  dP₁/dt = e₁·a₁·P₁·N/(1 + a₁·h₁·N) - d₁·P₁
  dP₂/dt = e₂·a₂·P₂·N/(1 + a₂·h₂·N) - d₂·P₂

Вектор параметрів b (10 компонентів, індекси 0..9 у Python):
  b = [r, K, a₁, h₁, e₁, d₁, a₂, h₂, e₂, d₂]

Стан u = [N, P₁, P₂]

Вихідні характеристики:
  ψ₁(b) = Σ N(tⱼ)²    (сума квадратів чисельності жертви)
  ψ₂(b) = Σ P₁(tⱼ)²
  ψ₃(b) = Σ P₂(tⱼ)²
"""

import numpy as np
from scipy.integrate import solve_ivp


def holling2(t, u, params):
    """Права частина системи ОДУ Холлінга II."""
    N, P1, P2 = u
    r, K, a1, h1, e1, d1, a2, h2, e2, d2 = params

    phi1 = a1 * N / (1.0 + a1 * h1 * N)
    phi2 = a2 * N / (1.0 + a2 * h2 * N)

    dN  = r * N * (1.0 - N / K) - phi1 * P1 - phi2 * P2
    dP1 = e1 * phi1 * P1 - d1 * P1
    dP2 = e2 * phi2 * P2 - d2 * P2

    return [dN, dP1, dP2]


def simulate(params, tspan=(0.0, 50.0), u0=None):
    """
    Розв'язує ОДУ для вектора параметрів b ∈ ℝ¹⁰ і повертає
    три вихідні характеристики [ψ₁, ψ₂, ψ₃].

    Parameters
    ----------
    params : array-like, shape (10,)
        Вектор параметрів [r, K, a₁, h₁, e₁, d₁, a₂, h₂, e₂, d₂].
    tspan : tuple
        (t_start, t_end) — часовий інтервал інтегрування.
    u0 : array-like or None
        Початкові умови [N₀, P₁₀, P₂₀]. За замовчуванням [80, 15, 10].

    Returns
    -------
    np.ndarray, shape (3,)
        [ψ₁, ψ₂, ψ₃] — суми квадратів по всіх розрахованих моментах часу.
    """
    if u0 is None:
        u0 = [80.0, 15.0, 10.0]

    params = np.asarray(params, dtype=float)

    sol = solve_ivp(
        holling2,
        tspan,
        u0,
        args=(params,),
        method='RK45',
        rtol=1e-6,
        atol=1e-6,
        dense_output=True,
    )

    if not sol.success:
        return np.array([np.nan, np.nan, np.nan])

    M = 101
    t_out = np.linspace(tspan[0], tspan[1], M)
    y_out = sol.sol(t_out)

    N_vals  = y_out[0]
    P1_vals = y_out[1]
    P2_vals = y_out[2]

    psi1 = float(np.sum(N_vals  ** 2))
    psi2 = float(np.sum(P1_vals ** 2))
    psi3 = float(np.sum(P2_vals ** 2))

    return np.array([psi1, psi2, psi3])
