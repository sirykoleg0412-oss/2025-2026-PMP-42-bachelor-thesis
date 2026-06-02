#!/usr/bin/env python3
"""
generate_comparison_plots.py
----------------------------
Порівняння трьох режимів сурогатної моделі:
  standard — звичайна RBF (без градієнтів)
  hermite  — Hermite RBF (з градієнтами)
  adaptive — Adaptive RBF (градієнти + адаптивне семплювання)

Для 1D сценаріїв: суцільна крива + навчальні точки.
Для 2D сценаріїв: 3D scatter «Істинні vs Прогнозовані».
Зберігає PNG у plots/comparison/true_vs_predicted/.
Також зберігає metrics_comparison.csv у data/results/.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from model import simulate as model_simulate
from sampling import SCENARIOS, BOUNDS, FIXED_VALUES, cached_samples
from rbf_method import (
    train_rbf, predict_rbf, adaptive_train_rbf, compute_metrics
)

DIM1_SCENARIOS = ["r", "K", "a1", "d1", "a2", "d2"]
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

MODE_LABEL = {
    "standard": "Standard RBF\n(без градієнтів)",
    "hermite":  "Gradient-based RBF\n(з градієнтами)",
    "adaptive": "Adaptive RBF\n(градієнти + адаптація)",
}

MODE_COLOR = {
    "standard": "steelblue",
    "hermite":  "seagreen",
    "adaptive": "mediumpurple",
}


def train_mode(mode, scenario, n, kernel_type, X_train, y_train, X_adapt_ref,
               eps=1.0, lam=1e-6):
    """Тренує модель у заданому режимі."""
    if mode == "standard":
        return train_rbf(X_train, y_train, kernel_type,
                         use_gradients=False, eps=eps, auto_eps=False,
                         lam=lam, log_transform=True)
    elif mode == "hermite":
        return train_rbf(X_train, y_train, kernel_type,
                         use_gradients=True, eps=eps, auto_eps=False,
                         lam=lam, log_transform=True)
    elif mode == "adaptive":
        model, _, _ = adaptive_train_rbf(
            scenario, n, kernel_type,
            X_init=X_train, y_init=y_train,
            adapt_ref=X_adapt_ref,
            use_gradients=True, eps=eps, auto_eps=False,
            lam=lam, log_transform=True,
        )
        return model
    else:
        raise ValueError(f"Невідомий режим: {mode}")


def plot_comparison_scenario_1d(
    scenario,
    ns=(5, 10, 20, 40),
    modes=("standard", "hermite", "adaptive"),
    kernel_type="multiquadric",
    lam=1e-6,
    n_test=50,
    n_curve=100,
):
    varying, fixed = SCENARIOS[scenario]
    idx = varying[0]
    eps = FIXED_EPS[kernel_type]
    lb, ub = BOUNDS[idx]
    x_label = PARAM_SHORT[idx]

    X_adapt_ref = cached_samples(100, varying, fixed, name="adapt_ref", scenario=scenario)
    X_metric    = cached_samples(n_test, varying, fixed, name="test", scenario=scenario)
    y_metric    = np.array([model_simulate(X_metric[i]) for i in range(n_test)])

    curve_vals = np.linspace(lb, ub, n_curve)
    X_curve = np.tile(FIXED_VALUES, (n_curve, 1))
    X_curve[:, idx] = curve_vals
    y_curve_true = np.array([model_simulate(X_curve[i]) for i in range(n_curve)])

    n_rows = len(ns)
    n_cols = len(modes)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    fig.suptitle(f"True vs Predicted — {scenario}", fontsize=12)

    for row, n in enumerate(ns):
        X_train = cached_samples(n, varying, fixed, name="train", scenario=scenario)
        y_train = np.array([model_simulate(X_train[i]) for i in range(n)])

        for col, mode in enumerate(modes):
            print(f"  [{scenario} 1D] mode={mode}  N={n} ...")
            ax = axes[row][col]

            model = train_mode(mode, scenario, n, kernel_type,
                               X_train, y_train, X_adapt_ref, eps=eps, lam=lam)

            y_pred_metric = np.array([predict_rbf(model, X_metric[i]) for i in range(n_test)])
            m    = compute_metrics(y_metric[:, 0], y_pred_metric[:, 0])
            r2   = round(m['r2_orig'],  3)
            rlog = round(m['rmse_log'], 4)

            y_curve_pred = np.array([predict_rbf(model, X_curve[i]) for i in range(n_curve)])

            ax.plot(curve_vals, y_curve_true[:, 0], color='gray', lw=1.5, ls='--',
                    label="Істинна" if row == 0 and col == 0 else "")
            ax.plot(curve_vals, y_curve_pred[:, 0], color=MODE_COLOR[mode], lw=2.0,
                    label=MODE_LABEL[mode].split('\n')[0] if row == 0 and col == 0 else "")
            ax.scatter(X_train[:, idx], y_train[:, 0],
                       color='red', s=30, zorder=5,
                       label="Навчальні" if row == 0 and col == 0 else "")
            ax.set_title(f"{MODE_LABEL[mode].split(chr(10))[0]}  N={n}\n"
                         f"RMSE_log={rlog}  R²={r2}", fontsize=7)
            ax.set_xlabel(x_label)
            ax.set_ylabel("ψ₁(b)")
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    plt.tight_layout()
    os.makedirs("plots/comparison/true_vs_predicted", exist_ok=True)
    out = f"plots/comparison/true_vs_predicted/{scenario}_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Збережено: {out}")


def plot_comparison_scenario_2d(
    scenario,
    ns=(5, 10, 20, 40),
    modes=("standard", "hermite", "adaptive"),
    kernel_type="multiquadric",
    lam=1e-6,
    n_test=50,
):
    varying, fixed = SCENARIOS[scenario]
    idx1, idx2 = varying
    eps = FIXED_EPS[kernel_type]
    x_label = PARAM_SHORT[idx1]
    y_label = PARAM_SHORT[idx2]

    X_adapt_ref = cached_samples(100, varying, fixed, name="adapt_ref", scenario=scenario)
    X_metric    = cached_samples(n_test, varying, fixed, name="test", scenario=scenario)
    y_metric    = np.array([model_simulate(X_metric[i]) for i in range(n_test)])

    n_rows = len(ns)
    n_cols = len(modes)
    fig = plt.figure(figsize=(3.8 * n_cols, 3.2 * n_rows))
    fig.suptitle(f"True vs Predicted — {scenario}", fontsize=12)

    for row, n in enumerate(ns):
        X_train = cached_samples(n, varying, fixed, name="train", scenario=scenario)
        y_train = np.array([model_simulate(X_train[i]) for i in range(n)])

        X_vis   = cached_samples(n, varying, fixed, name="vis_test", scenario=scenario)
        y_vis   = np.array([model_simulate(X_vis[i]) for i in range(n)])

        for col, mode in enumerate(modes):
            panel = row * n_cols + col + 1
            print(f"  [{scenario}] mode={mode}  N={n} ...")
            ax = fig.add_subplot(n_rows, n_cols, panel, projection='3d')

            model = train_mode(mode, scenario, n, kernel_type,
                               X_train, y_train, X_adapt_ref, eps=eps, lam=lam)

            y_pred_metric = np.array([predict_rbf(model, X_metric[i]) for i in range(n_test)])
            m    = compute_metrics(y_metric[:, 0], y_pred_metric[:, 0])
            r2   = round(m['r2_orig'],  3)
            rlog = round(m['rmse_log'], 4)

            y_pred_vis = np.array([predict_rbf(model, X_vis[i]) for i in range(n)])

            ax.scatter(X_vis[:, idx1], X_vis[:, idx2], y_vis[:, 0],
                       c='red', s=12, alpha=0.75, label="Істинні")
            ax.scatter(X_vis[:, idx1], X_vis[:, idx2], y_pred_vis[:, 0],
                       c=MODE_COLOR[mode], s=12, marker='^', alpha=0.65, label="Прогноз")
            ax.set_title(f"{MODE_LABEL[mode].split(chr(10))[0]}  N={n}\n"
                         f"RMSE_log={rlog}  R²={r2}", fontsize=7)
            ax.set_xlabel(x_label, fontsize=7)
            ax.set_ylabel(y_label, fontsize=7)
            ax.set_zlabel("ψ₁(b)", fontsize=7)
            ax.view_init(elev=20, azim=30)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    plt.tight_layout()
    os.makedirs("plots/comparison/true_vs_predicted", exist_ok=True)
    out = f"plots/comparison/true_vs_predicted/{scenario}_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Збережено: {out}")


def save_comparison_metrics(
    ns=(5, 10, 20, 40, 80),
    modes=("standard", "hermite", "adaptive"),
    n_test=50,
    kernel_type="multiquadric",
    lam=1e-6,
):
    """Зберігає metrics_comparison.csv."""
    rows = []
    all_scenarios = (
        [(s, 1) for s in DIM1_SCENARIOS] +
        [(s, 2) for s in DIM2_SCENARIOS]
    )

    for scenario, dim in all_scenarios:
        print(f"\n── Метрики: {scenario} ──")
        varying, fixed = SCENARIOS[scenario]
        eps = FIXED_EPS[kernel_type]

        X_test = cached_samples(n_test, varying, fixed, name="test", scenario=scenario)
        y_test = np.array([model_simulate(X_test[i]) for i in range(n_test)])
        X_adapt_ref = cached_samples(100, varying, fixed, name="adapt_ref", scenario=scenario)

        for n in ns:
            X_train = cached_samples(n, varying, fixed, name="train", scenario=scenario)
            y_train = np.array([model_simulate(X_train[i]) for i in range(n)])

            for mode in modes:
                model = train_mode(mode, scenario, n, kernel_type,
                                   X_train, y_train, X_adapt_ref, eps=eps, lam=lam)
                y_pred = np.array([predict_rbf(model, X_test[i]) for i in range(n_test)])
                m = compute_metrics(y_test[:, 0], y_pred[:, 0])
                rows.append({
                    'scenario': scenario, 'dim': dim, 'N': n, 'mode': mode,
                    'RMSE':     round(m['rmse_orig'], 5),
                    'R2':       round(m['r2_orig'],   4),
                    'RMSE_log': round(m['rmse_log'],  5),
                    'R2_log':   round(m['r2_log'],    4),
                })
                print(f"  {scenario} | N={n} | mode={mode} → "
                      f"R²={m['r2_orig']:.3f}  RMSE_log={m['rmse_log']:.4f}")

    os.makedirs("data/results", exist_ok=True)
    out = "data/results/metrics_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n✓ Метрики збережено: {out}")


def main():
    print("=" * 65)
    print("Порівняння 3 режимів: Standard / Hermite / Adaptive RBF")
    print("=" * 65)

    print("\n── 1D сценарії ─────────────────────────────────────────")
    for scenario in DIM1_SCENARIOS:
        print(f"\n▶ Сценарій: {scenario}")
        plot_comparison_scenario_1d(scenario)

    print("\n── 2D сценарії ─────────────────────────────────────────")
    for scenario in DIM2_SCENARIOS:
        print(f"\n▶ Сценарій: {scenario}")
        plot_comparison_scenario_2d(scenario)

    print("\n── Збереження метрик по режимах ────────────────────────")
    save_comparison_metrics()

    print("\n✓ Графіки збережені у: plots/comparison/true_vs_predicted/")


if __name__ == "__main__":
    main()
