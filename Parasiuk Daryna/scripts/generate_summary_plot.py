#!/usr/bin/env python3
"""
generate_summary_plot.py
------------------------
Читає metrics_*.csv з data/results/ і будує графік RMSE_log vs N
для 1D, 2D і 3D сценаріїв.

Зберігає PNG у plots/rbf_with_gradients_and_adaptive_sampling/rmse_vs_n.png.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))


DIM1 = ["r", "K", "a1", "d1", "a2", "d2"]
DIM2 = ["r_K", "a1_d1", "a2_d2", "d1_d2", "a1_a2", "e1_e2"]
DIM3 = ["r_a1_a2"]


def load_rmse_log(scenario):
    path = f"data/results/metrics_{scenario}.csv"
    if not os.path.isfile(path):
        return [], []
    df = pd.read_csv(path)
    if 'RMSE_log' in df.columns:
        return df['N'].tolist(), df['RMSE_log'].tolist()
    elif 'RMSE' in df.columns:
        return df['N'].tolist(), df['RMSE'].tolist()
    return [], []


def main():
    fig, axes = plt.subplots(3, 1, figsize=(9, 13))

    groups = [
        (axes[0], DIM1, "1D сценарії (адаптивне семплювання, log_transform)", "RMSE_log (ln-простір)"),
        (axes[1], DIM2, "2D сценарії", "RMSE_log (ln-простір)"),
        (axes[2], DIM3, "3D сценарій", "RMSE_log (ln-простір)"),
    ]

    for ax, scenarios, title, ylabel in groups:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        for sc in scenarios:
            n_vals, rmse_vals = load_rmse_log(sc)
            if not n_vals:
                continue
            ax.plot(n_vals, rmse_vals, marker='o', label=sc)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = "plots/rbf_with_gradients_and_adaptive_sampling"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "rmse_vs_n.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✓ Збережено: {out}")


if __name__ == "__main__":
    main()
