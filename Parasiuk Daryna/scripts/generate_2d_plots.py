#!/usr/bin/env python3
"""
generate_2d_plots.py
--------------------
Будує 4-панельні 3D-графіки (true vs predicted) для всіх 2D-сценаріїв
з використанням Adaptive RBF (градієнти + адаптивне семплювання).

Зберігає PNG у plots/rbf_with_gradients_and_adaptive_sampling/true_vs_predicted/.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from model import simulate as model_simulate
from sampling import SCENARIOS, cached_samples
from rbf_method import adaptive_train_rbf, predict_rbf, compute_metrics

DIM2_SCENARIOS = ["r_K", "a1_d1", "a2_d2", "d1_d2", "a1_a2", "e1_e2"]

PARAM_SHORT = {
    0: "r",  1: "K",
    2: "a₁", 3: "h₁", 4: "e₁", 5: "d₁",
    6: "a₂", 7: "h₂", 8: "e₂", 9: "d₂",
}

FIXED_EPS = {
    "gaussian":             0.5,
    "multiquadric":         1.0,
    "inverse_multiquadric": 1.0,
}


def make_title(scenario: str) -> str:
    parts = scenario.split("_")
    return "Параметри " + " і ".join(parts)


def plot_2d_scenario(
    scenario,
    ns=(5, 10, 20, 40),
    kernel_type="multiquadric",
    lam=1e-6,
):
    varying, fixed = SCENARIOS[scenario]
    idx1, idx2 = varying
    eps = FIXED_EPS[kernel_type]
    x_label = PARAM_SHORT[idx1]
    y_label = PARAM_SHORT[idx2]
    z_label = "ψ₁(b) = ΣN²(b)"
    title_str = make_title(scenario)

    X_adapt_ref = cached_samples(100, varying, fixed, name="adapt_ref", scenario=scenario)

    fig = plt.figure(figsize=(11, 9))
    fig.suptitle(title_str, fontsize=13)

    for panel_idx, n in enumerate(ns):
        X_train = cached_samples(n, varying, fixed, name="train", scenario=scenario)
        y_train = np.array([model_simulate(X_train[i]) for i in range(n)])

        X_test = cached_samples(n, varying, fixed, name="test", scenario=scenario)
        y_test = np.array([model_simulate(X_test[i]) for i in range(len(X_test))])

        model, _, _ = adaptive_train_rbf(
            scenario, n, kernel_type,
            X_init=X_train,
            y_init=y_train,
            adapt_ref=X_adapt_ref,
            use_gradients=True,
            eps=eps,
            auto_eps=False,
            lam=lam,
            log_transform=True,
        )

        y_pred = np.array([predict_rbf(model, X_test[i]) for i in range(len(X_test))])

        ax = fig.add_subplot(2, 2, panel_idx + 1, projection='3d')
        ax.scatter(X_test[:, idx1], X_test[:, idx2], y_test[:, 0],
                   c='red', s=12, alpha=0.8, label="Істинні точки")
        ax.scatter(X_test[:, idx1], X_test[:, idx2], y_pred[:, 0],
                   c='blue', marker='^', s=12, alpha=0.7, label="Прогнозовані точки")

        m    = compute_metrics(y_test[:, 0], y_pred[:, 0])
        r2   = round(m['r2_orig'],  3)
        rlog = round(m['rmse_log'], 4)

        ax.set_title(f"{n} точок   RMSE_log={rlog}  R²={r2}", fontsize=9)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_zlabel(z_label, fontsize=8)
        ax.view_init(elev=20, azim=30)
        if panel_idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    out_dir = "plots/rbf_with_gradients_and_adaptive_sampling/true_vs_predicted"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{scenario}_true_vs_predicted.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Збережено: {filename}")


def main():
    print("=" * 55)
    print("Генерація 4-панельних 3D-графіків (true vs predicted)")
    print("=" * 55)

    for scenario in DIM2_SCENARIOS:
        print(f"\n▶ Сценарій: {scenario}  ({make_title(scenario)})")
        plot_2d_scenario(scenario)

    print("\n✓ Готово! Усі графіки збережені у папці: "
          "plots/rbf_with_gradients_and_adaptive_sampling/true_vs_predicted/")
    for scenario in DIM2_SCENARIOS:
        print(f"   • {scenario}_true_vs_predicted.png")


if __name__ == "__main__":
    main()
