#!/usr/bin/env python3
"""
generate_tables.py
------------------
Обчислює метрики (MAE, RMSE, RMSE_log, R²) для всіх сценаріїв, розмірів
вибірки N і ядер. Зберігає CSV у data/results/.

"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from model import simulate as model_simulate
from sampling import SCENARIOS, cached_samples
from rbf_method import adaptive_train_rbf, predict_rbf, compute_metrics

FIXED_EPS = {
    "gaussian":             0.5,
    "multiquadric":         1.0,
    "inverse_multiquadric": 1.0,
}

KERNELS = ["gaussian", "multiquadric", "inverse_multiquadric"]

KERNEL_LABELS = {
    "gaussian":             "Gaussian",
    "multiquadric":         "Multiquadric",
    "inverse_multiquadric": "Inv. Multiquadric",
}

ADAPT_MAX_ITER = 2


def raw_metrics(y_true, y_pred):
    errors = y_true - y_pred
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return mae, rmse, r2


def main():
    scenarios = sorted(SCENARIOS.keys())
    ns     = [5, 10, 20, 40, 80]
    n_test = 50

    os.makedirs("data/results", exist_ok=True)

    kernel_rows  = []
    summary_data = []

    for scenario in scenarios:
        print(f"\nСценарій: {scenario}")
        varying, fixed = SCENARIOS[scenario]

        X_test = cached_samples(n_test, varying, fixed, name="test", scenario=scenario)
        y_test = np.array([model_simulate(X_test[i]) for i in range(n_test)])

        X_adapt_ref = cached_samples(n_test, varying, fixed,
                                     name="adapt_ref", scenario=scenario)

        scenario_rows = []

        for n in ns:
            X_train = cached_samples(n, varying, fixed, name="train", scenario=scenario)
            y_train = np.array([model_simulate(X_train[i]) for i in range(n)])

            for kernel in KERNELS:
                eps = FIXED_EPS[kernel]

                model, _, _ = adaptive_train_rbf(
                    scenario, n, kernel,
                    X_init=X_train,
                    y_init=y_train,
                    adapt_ref=X_adapt_ref,
                    eps=eps,
                    lam=1e-6,
                    use_gradients=True,
                    auto_eps=False,
                    max_iter=ADAPT_MAX_ITER,
                    adapt_method="gradient",
                    log_transform=True,
                )

                y_pred = np.array([predict_rbf(model, X_test[i]) for i in range(n_test)])
                mae, _, _ = raw_metrics(y_test[:, 0], y_pred[:, 0])
                m = compute_metrics(y_test[:, 0], y_pred[:, 0])

                kernel_rows.append({
                    'scenario': scenario, 'n': n, 'kernel': kernel,
                    'mae':      mae,
                    'rmse':     m['rmse_orig'],
                    'r2':       m['r2_orig'],
                    'rmse_log': m['rmse_log'],
                    'r2_log':   m['r2_log'],
                })

                if kernel == "multiquadric":
                    scenario_rows.append({
                        'N':       n,
                        'MAE':     mae,
                        'RMSE':    m['rmse_orig'],
                        'R2':      m['r2_orig'],
                        'RMSE_log': m['rmse_log'],
                        'R2_log':  m['r2_log'],
                    })
                    summary_data.append((scenario, n, mae, m['rmse_orig'],
                                         m['r2_orig'], m['rmse_log']))

                print(f"  N={n:<3d}  {KERNEL_LABELS[kernel]:<22s}  "
                      f"RMSE={m['rmse_orig']:.2f}  "
                      f"RMSE_log={m['rmse_log']:.4f}  R²={m['r2_orig']:.4f}")

        pd.DataFrame(scenario_rows).to_csv(f"data/results/metrics_{scenario}.csv", index=False)

    df_kernels = pd.DataFrame(kernel_rows)
    df_kernels.to_csv("data/results/metrics_kernels.csv", index=False)
    print("\n✓ data/results/metrics_kernels.csv")

    # ─── Таблиця порівняння ядер ───────────────────────────────────────────────
    sep = "=" * 88
    print(f"\n{sep}")
    print("ПОРІВНЯННЯ ЯДЕР: RMSE / RMSE_log / R²  (усереднено по всіх сценаріях)")
    print("Вибірка: LHS,  градієнти: увімкнено,  log_transform: true,  ε: фіксований")
    print(sep)
    print(f"{'Ядро':<22} | {'N':<4} | {'MAE':>12} | {'RMSE':>12} | {'RMSE_log':>10} | {'R²':>8}")
    print("-" * 88)

    for n in ns:
        for ki, kernel in enumerate(KERNELS):
            sub = df_kernels[(df_kernels['n'] == n) & (df_kernels['kernel'] == kernel)]
            mae_m  = sub['mae'].mean()
            rmse_m = sub['rmse'].mean()
            rlog_m = sub['rmse_log'].mean()
            r2_m   = sub['r2'].replace([float('nan')], np.nan).mean()
            label  = f"N={n}" if ki == 0 else ""
            print(f"{KERNEL_LABELS[kernel]:<22} | {label:<4} | "
                  f"{mae_m:>12.2f} | {rmse_m:>12.2f} | "
                  f"{rlog_m:>10.4f} | {r2_m:>8.4f}")
        print("-" * 88)

    # ─── Зведена таблиця Multiquadric ──────────────────────────────────────────
    print(f"\n{sep}")
    print("ЗВЕДЕНА ТАБЛИЦЯ (Multiquadric, log_transform=true, всі сценарії, ψ₁)")
    print(sep)
    print(f"{'Сценарій':<14} | {'N':<4} | {'MAE':>12} | {'RMSE':>12} | {'RMSE_log':>10} | {'R²':>8}")
    print("-" * 88)
    for (scenario, n, mae, rmse, r2, rmse_log) in summary_data:
        print(f"{scenario:<14} | {n:<4d} | {mae:>12.2f} | {rmse:>12.2f} | "
              f"{rmse_log:>10.4f} | {r2:>8.4f}")
    print(sep)


if __name__ == "__main__":
    main()
