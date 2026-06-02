#!/usr/bin/env python3
"""
main.py
-------
Точка входу Python-проекту.

Режими запуску:
  python main.py           → реальна ОДУ-система Холлінга II
  python main.py rosenbrock → тестовий полігон функції Розенброка

"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# Додаємо src/python до шляху пошуку модулів
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))

from model import simulate as model_simulate
from sampling import SCENARIOS, BOUNDS, FIXED_VALUES, generate_samples, cached_samples
from rbf_method import (
    train_rbf, predict_rbf, adaptive_train_rbf, compute_metrics
)


# ─────────────────────────────────────────────────────────────────────────────
# Функція Розенброка (тестовий полігон)
# ─────────────────────────────────────────────────────────────────────────────

def rosenbrock(b):
    """f(b₁, b₂) = (1 - b₁)² + 100·(b₂ - b₁²)²"""
    b1, b2 = b[0], b[1]
    return np.array([(1.0 - b1) ** 2 + 100.0 * (b2 - b1 ** 2) ** 2])

def lhs_rosenbrock(n, seed=42):
    """LHS-вибірка в [-2, 2]² для тестового полігону."""
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit    = sampler.random(n=n)
    return qmc.scale(unit, [-2.0, -2.0], [2.0, 2.0])


# ─────────────────────────────────────────────────────────────────────────────
# Режим реальної ОДУ-системи
# ─────────────────────────────────────────────────────────────────────────────

def run_ode_mode():
    print("Режим: реальна ОДУ-система Холлінга II")
    print("Оракул: model.simulate (чисельний розв'язок RK45)")
    print()

    scenario = "r_K"
    varying, fixed = SCENARIOS[scenario]
    n_train = 20

    X_train = cached_samples(n_train, varying, fixed, name="train", scenario=scenario)
    y_train = np.array([model_simulate(X_train[i]) for i in range(n_train)])

    model, X_final, history = adaptive_train_rbf(
        scenario, n_train, "multiquadric",
        X_init=X_train,
        y_init=y_train,
        max_iter=3,
        lam=1e-6,
        test_n=50,
        use_gradients=True,
        auto_eps=False,
        eps=1.0,
        log_transform=True,
    )

    n_added = len(X_final) - n_train

    X_test = cached_samples(100, varying, fixed, name="test", scenario=scenario)
    y_test = np.array([model_simulate(X_test[i]) for i in range(100)])
    y_pred = np.array([predict_rbf(model, X_test[i]) for i in range(100)])

    m = compute_metrics(y_test[:, 0], y_pred[:, 0])
    print(f"RMSE={m['rmse_orig']:.2f}, RMSE_log={m['rmse_log']:.4f}, R²={m['r2_orig']:.4f}")

    os.makedirs("plots/ode_mode", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7, label="Тест")
    lim_min = min(y_test[:, 0].min(), y_pred[:, 0].min())
    lim_max = max(y_test[:, 0].max(), y_pred[:, 0].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label="ідеал")
    ax.set_xlabel("Істинне ψ₁(b)")
    ax.set_ylabel("Прогноз ψ₁(b)")
    ax.set_title(f"RBF vs ОДУ: ψ₁(b), сценарій {scenario}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/ode_mode/rbf_vs_ode_psi1.png", dpi=150)
    plt.close()
    print("Графік збережено → plots/ode_mode/rbf_vs_ode_psi1.png")
    print(f"Додано точок адаптивно: {n_added}")


# ─────────────────────────────────────────────────────────────────────────────
# Режим тестового полігону Розенброка
# ─────────────────────────────────────────────────────────────────────────────

def run_rosenbrock_mode():
    print("Режим: тестовий полігон — функція Розенброка")
    print("  f(b₁,b₂) = (1-b₁)² + 100·(b₂-b₁²)²")
    print("Оракул: аналітична формула")
    print()

    lb, ub  = -2.0, 2.0
    n_train = 40

    X_train = lhs_rosenbrock(n_train, seed=42)
    y_train = np.array([rosenbrock(X_train[i]) for i in range(n_train)])

    print(f"Навчальних точок: {n_train}")

    m_std = train_rbf(X_train, y_train, "multiquadric",
                      eps=1.0, lam=1e-6, use_gradients=False)

    m_grad = train_rbf(X_train, y_train, "multiquadric",
                       eps=1.0, lam=1e-6,
                       use_gradients=True,
                       sim_fn=rosenbrock)

    grid_n = 40
    b1r = np.linspace(lb, ub, grid_n)
    b2r = np.linspace(lb, ub, grid_n)
    B1, B2 = np.meshgrid(b1r, b2r)

    y_true  = np.array([rosenbrock([b1, b2])[0]           for b1 in b1r for b2 in b2r])
    y_p_std = np.array([predict_rbf(m_std,  [b1, b2])[0]  for b1 in b1r for b2 in b2r])
    y_p_gr  = np.array([predict_rbf(m_grad, [b1, b2])[0]  for b1 in b1r for b2 in b2r])

    rmse_std  = float(np.sqrt(np.mean((y_true - y_p_std) ** 2)))
    rmse_grad = float(np.sqrt(np.mean((y_true - y_p_gr)  ** 2)))

    def r2(yt, yp):
        return 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)

    print(f"Метрики на сітці {grid_n}×{grid_n}:")
    print(f"  Стандартна RBF:  RMSE={rmse_std:.4f}  R²={r2(y_true, y_p_std):.4f}")
    print(f"  Градієнтна RBF:  RMSE={rmse_grad:.4f}  R²={r2(y_true, y_p_gr):.4f}")

    Z_true = np.array([[rosenbrock([b1, b2])[0] for b1 in b1r] for b2 in b2r])
    Z_std  = np.array([[predict_rbf(m_std,  [b1, b2])[0] for b1 in b1r] for b2 in b2r])
    Z_grad = np.array([[predict_rbf(m_grad, [b1, b2])[0] for b1 in b1r] for b2 in b2r])

    fig = plt.figure(figsize=(14, 10))
    titles = ["Розенброк (аналітична)", f"RBF без градієнтів (N={n_train})",
              f"RBF з градієнтами (N={n_train})", "|Похибка| градієнтна RBF"]
    Zs = [Z_true, Z_std, Z_grad, np.abs(Z_grad - Z_true)]
    cmaps = ['viridis', 'viridis', 'viridis', 'hot']

    for k, (ttl, Z, cmap_name) in enumerate(zip(titles, Zs, cmaps)):
        ax = fig.add_subplot(2, 2, k + 1, projection='3d')
        ax.plot_surface(B1, B2, Z, cmap=cmap_name, alpha=0.85)
        ax.set_xlabel("b₁"); ax.set_ylabel("b₂"); ax.set_zlabel("f")
        ax.set_title(ttl, fontsize=10)
        ax.view_init(elev=30, azim=40)

    fig.suptitle("RBF-сурогат: функція Розенброка", fontsize=13)
    plt.tight_layout()
    os.makedirs("plots/rosenbrock_mode", exist_ok=True)
    plt.savefig("plots/rosenbrock_mode/surfaces.png", dpi=150)
    plt.close()
    print("Графік збережено → plots/rosenbrock_mode/surfaces.png")


# ─────────────────────────────────────────────────────────────────────────────
# Вибір режиму
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) >= 2 else "ode"
    if mode == "rosenbrock":
        run_rosenbrock_mode()
    else:
        run_ode_mode()
