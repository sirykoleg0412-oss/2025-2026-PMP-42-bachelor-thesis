#!/usr/bin/env python3
"""
rosenbrock_kernel_comparison.py
--------------------------------
Верифікація RBF-сурогата на аналітичній функції Розенброка:
  f(b1, b2) = (1 - b1)^2 + 100*(b2 - b1^2)^2   на [-2,2] x [-2,2]

Таблиця метрик: 3 ядра x 3 моделі x N in {5,10,20,40}
  Ядра:   gaussian (eps=0.5), multiquadric (eps=1.0), inverse_multiquadric (eps=1.0)
  Моделі: Standard RBF | Gradient-based RBF | Gradient-based RBF + Adaptive Sampling

Графік 1: RMSE vs N — 3 підграфіки (по ядру), 3 лінії (по моделі)
Графік 2: Апроксимація vs Істинна (multiquadric, N=20):
          4 панелі — True | Standard | Gradient-based | Adaptive
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))
from rbf_method import train_rbf, predict_rbf, compute_metrics

# ─── Розенброк і LHS ──────────────────────────────────────────────────────────

LB, UB = -2.0, 2.0

def rosenbrock(b):
    b1, b2 = b[0], b[1]
    return np.array([(1.0 - b1)**2 + 100.0 * (b2 - b1**2)**2])

def lhs_sample(n, seed=0):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    return qmc.scale(sampler.random(n=n), [LB, LB], [UB, UB])

# ─── Адаптивне навчання для Розенброка ────────────────────────────────────────

def adaptive_train_rosenbrock(X_train, y_train, kernel, eps, lam,
                               X_pool, max_iter=3):
    """Gradient-based adaptive sampling для 2D Розенброка."""
    X_tr = X_train.copy()
    y_tr = y_train.copy()
    X_pl = X_pool.copy()
    n_init = len(X_train)

    for _ in range(max_iter):
        model = train_rbf(X_tr, y_tr, kernel_type=kernel,
                          eps=eps, lam=lam, use_gradients=True,
                          auto_eps=False, log_transform=False,
                          sim_fn=rosenbrock)

        # Критерій: |grad f| по обох вимірах (центральні різниці)
        h = 1e-3
        criterion = []
        for x_i in X_pl:
            grads = []
            for vi in range(2):
                xp = x_i.copy(); xp[vi] += h
                xm = x_i.copy(); xm[vi] -= h
                diff = (rosenbrock(xp)[0] - rosenbrock(xm)[0]) / (2 * h)
                grads.append(diff)
            criterion.append(norm(grads))

        n_add = max(1, n_init // 10)
        top_idx = np.argsort(criterion)[::-1][:n_add]

        new_X = X_pl[top_idx]
        new_y = np.array([rosenbrock(x)[0] for x in new_X])

        mask = np.ones(len(X_pl), dtype=bool)
        mask[top_idx] = False
        X_pl = X_pl[mask]

        X_tr = np.vstack([X_tr, new_X])
        y_tr = np.concatenate([y_tr, new_y])

    return train_rbf(X_tr, y_tr, kernel_type=kernel,
                     eps=eps, lam=lam, use_gradients=True,
                     auto_eps=False, log_transform=False,
                     sim_fn=rosenbrock)

# ─── Налаштування ─────────────────────────────────────────────────────────────

KERNELS = {
    "gaussian":             0.5,
    "multiquadric":         1.0,
    "inverse_multiquadric": 1.0,
}
KERNEL_LABELS = {
    "gaussian":             "Gaussian",
    "multiquadric":         "Multiquadric",
    "inverse_multiquadric": "Inv. Multiquadric",
}
MODELS = ["standard", "gradient", "adaptive"]
MODEL_LABELS = {
    "standard": "Standard RBF",
    "gradient": "Gradient-based RBF",
    "adaptive": "Gradient-based RBF\n+ Adaptive Sampling",
}
MODEL_COLORS = {
    "standard": "steelblue",
    "gradient": "seagreen",
    "adaptive": "mediumpurple",
}
MODEL_MARKERS = {
    "standard": "o",
    "gradient": "s",
    "adaptive": "^",
}
MODEL_LINESTYLE = {
    "standard": ":",
    "gradient": "--",
    "adaptive": "-.",
}

N_VALUES = [5, 10, 20, 40]
N_TEST   = 50
N_POOL   = 100
LAM      = 1e-6

OUTDIR_TABLES = "data/results"
OUTDIR_PLOTS  = "plots/rosenbrock_kernel_comparison"
os.makedirs(OUTDIR_TABLES, exist_ok=True)
os.makedirs(OUTDIR_PLOTS,  exist_ok=True)

# Фіксовані тестові та пул точки
X_test = lhs_sample(N_TEST, seed=999)
y_test = np.array([rosenbrock(x)[0] for x in X_test])

# ─── Крок 1: Таблиця метрик ───────────────────────────────────────────────────

print("=" * 65)
print("Таблиця метрик: 3 ядра x 3 моделі x 4 значення N")
print("=" * 65)

results = {k: {m: {"RMSE": [], "R2": []} for m in MODELS} for k in KERNELS}
rows = []

for kernel, eps in KERNELS.items():
    for n in N_VALUES:
        X_train = lhs_sample(n, seed=42)
        y_train = np.array([rosenbrock(x)[0] for x in X_train])
        X_pool  = lhs_sample(N_POOL, seed=77)

        for mode in MODELS:
            if mode == "standard":
                model = train_rbf(X_train, y_train, kernel_type=kernel,
                                  eps=eps, lam=LAM, use_gradients=False,
                                  auto_eps=False, log_transform=False)
            elif mode == "gradient":
                model = train_rbf(X_train, y_train, kernel_type=kernel,
                                  eps=eps, lam=LAM, use_gradients=True,
                                  auto_eps=False, log_transform=False,
                                  sim_fn=rosenbrock)
            else:  # adaptive
                model = adaptive_train_rosenbrock(X_train, y_train,
                                                  kernel, eps, LAM, X_pool)

            y_pred = np.array([predict_rbf(model, x)[0] for x in X_test])
            m = compute_metrics(y_test, y_pred)

            results[kernel][mode]["RMSE"].append(m["rmse_orig"])
            results[kernel][mode]["R2"].append(m["r2_orig"])

            rows.append({
                "kernel":   kernel,
                "model":    mode,
                "N":        n,
                "RMSE":     round(m["rmse_orig"], 2),
                "R2":       round(m["r2_orig"],   4),
                "RMSE_log": round(m["rmse_log"],  4),
                "R2_log":   round(m["r2_log"],    4),
            })
            print(f"  {KERNEL_LABELS[kernel]:20s} | {mode:10s} | N={n:2d} | "
                  f"RMSE={m['rmse_orig']:8.2f}  R²={m['r2_orig']:.4f}")

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR_TABLES, "rosenbrock_kernel_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\nТаблиця → {csv_path}")

# ─── Крок 2: RMSE vs N (3 підграфіки по ядру, 3 лінії по моделі) ─────────────

print("\nБудую графік RMSE vs N ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, (kernel, eps) in zip(axes, KERNELS.items()):
    for mode in MODELS:
        vals = results[kernel][mode]["R2"]
        ax.plot(N_VALUES, vals,
                color=MODEL_COLORS[mode],
                marker=MODEL_MARKERS[mode],
                linestyle=MODEL_LINESTYLE[mode],
                linewidth=2, markersize=7,
                alpha=0.5 if mode in ("standard", "gradient") else 1.0,
                label=MODEL_LABELS[mode].replace("\n", " "))
    ax.set_title(KERNEL_LABELS[kernel], fontsize=12)
    ax.set_xlabel("N (тренувальних точок)")
    ax.set_ylabel("R²")
    ax.set_xticks(N_VALUES)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("R² vs N — функція Розенброка (N_test=50)", fontsize=13)
plt.tight_layout()
path1 = os.path.join(OUTDIR_PLOTS, "rosenbrock_rmse_vs_n.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"Збережено → {path1}")

# ─── Крок 3: Поверхні при N=20 (multiquadric) — True vs 3 моделі ──────────────

print("Будую графіки поверхонь (multiquadric, N=5/10/40) ...")

GRID_N = 50
b1r = np.linspace(LB, UB, GRID_N)
b2r = np.linspace(LB, UB, GRID_N)
B1, B2 = np.meshgrid(b1r, b2r)
Z_true = np.array([[rosenbrock([b1, b2])[0] for b1 in b1r] for b2 in b2r])

def eval_grid(model):
    return np.array([[predict_rbf(model, [b1, b2])[0] for b1 in b1r] for b2 in b2r])

def metrics_grid(Z_p):
    m = compute_metrics(Z_true.ravel(), Z_p.ravel())
    return m["r2_orig"]

kernel_vis, eps_vis = "multiquadric", 1.0
for n_vis in [5, 10, 40]:
    X_train_vis = lhs_sample(n_vis, seed=42)
    y_train_vis = np.array([rosenbrock(x)[0] for x in X_train_vis])
    X_pool_vis  = lhs_sample(N_POOL, seed=77)

    m_std  = train_rbf(X_train_vis, y_train_vis, kernel_type=kernel_vis,
                       eps=eps_vis, lam=LAM, use_gradients=False,
                       auto_eps=False, log_transform=False)
    m_grad = train_rbf(X_train_vis, y_train_vis, kernel_type=kernel_vis,
                       eps=eps_vis, lam=LAM, use_gradients=True,
                       auto_eps=False, log_transform=False, sim_fn=rosenbrock)
    m_adap = adaptive_train_rosenbrock(X_train_vis, y_train_vis,
                                        kernel_vis, eps_vis, LAM, X_pool_vis)

    Z_std  = eval_grid(m_std)
    Z_grad = eval_grid(m_grad)
    Z_adap = eval_grid(m_adap)

    rmse_std  = float(np.sqrt(np.mean((Z_true - Z_std)**2)))
    rmse_grad = float(np.sqrt(np.mean((Z_true - Z_grad)**2)))
    rmse_adap = float(np.sqrt(np.mean((Z_true - Z_adap)**2)))

    E_std  = np.abs(Z_true - Z_std)
    E_grad = np.abs(Z_true - Z_grad)
    E_adap = np.abs(Z_true - Z_adap)

    # Рядок 1: поверхні; рядок 2: карти похибки
    top_panels = [
        (Z_true, "Аналітична f(b₁,b₂)\n(Розенброк)",                                         "viridis", None),
        (Z_std,  f"Standard RBF\nRMSE={rmse_std:.1f}  R²={metrics_grid(Z_std):.3f}",         "viridis", m_std),
        (Z_grad, f"Gradient-based RBF\nRMSE={rmse_grad:.1f}  R²={metrics_grid(Z_grad):.3f}", "viridis", m_grad),
        (Z_adap, f"Adaptive RBF\nRMSE={rmse_adap:.1f}  R²={metrics_grid(Z_adap):.3f}",       "viridis", m_adap),
    ]
    bot_panels = [
        (None,   None),
        (E_std,  "|f − f̂|  Standard RBF"),
        (E_grad, "|f − f̂|  Gradient-based RBF"),
        (E_adap, "|f − f̂|  Adaptive RBF"),
    ]

    fig = plt.figure(figsize=(18, 9))
    train_pts_z = np.array([rosenbrock(x)[0] for x in X_train_vis])

    for k, (Z, title, cmap, mdl) in enumerate(top_panels):
        ax = fig.add_subplot(2, 4, k + 1, projection='3d')
        ax.plot_surface(B1, B2, Z, cmap=cmap, alpha=0.85, linewidth=0)
        if mdl is not None:
            ax.scatter(X_train_vis[:, 0], X_train_vis[:, 1],
                       train_pts_z, color='red', s=20, zorder=5)
        ax.set_xlabel("b₁"); ax.set_ylabel("b₂"); ax.set_zlabel("f")
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=30, azim=40)

    for k, (E, title) in enumerate(bot_panels):
        ax = fig.add_subplot(2, 4, k + 5, projection='3d')
        if E is None:
            ax.axis('off')
            continue
        ax.plot_surface(B1, B2, E, cmap='hot', alpha=0.85, linewidth=0)
        ax.set_xlabel("b₁"); ax.set_ylabel("b₂"); ax.set_zlabel("|помилка|")
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=30, azim=40)

    fig.suptitle(f"Розенброк: порівняння моделей (multiquadric, N={n_vis})", fontsize=12)
    plt.tight_layout()
    path2 = os.path.join(OUTDIR_PLOTS, f"rosenbrock_surfaces_N{n_vis}.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Збережено → {path2}")

print("\n✓ Готово!")
