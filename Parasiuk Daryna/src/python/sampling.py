"""
sampling.py
-----------
Генерація вибірок методом Latin Hypercube Sampling (LHS) для параметрів
моделі Холлінга II.

Параметри моделі (10 штук, індекси 0..9 у Python):
  0  r   — швидкість росту жертви
  1  K   — ємність середовища
  2  a₁  — частота атаки хижака 1
  3  h₁  — час обробки хижака 1
  4  e₁  — ефективність засвоєння хижака 1
  5  d₁  — смертність хижака 1
  6  a₂  — частота атаки хижака 2
  7  h₂  — час обробки хижака 2
  8  e₂  — ефективність засвоєння хижака 2
  9  d₂  — смертність хижака 2
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

PARAM_NAMES = ["r", "K", "a1", "h1", "e1", "d1", "a2", "h2", "e2", "d2"]

BOUNDS = [
    (0.5,  3.0),    # r
    (50.0, 300.0),  # K
    (0.05, 0.40),   # a₁
    (0.05, 0.50),   # h₁
    (0.20, 0.90),   # e₁
    (0.10, 0.70),   # d₁
    (0.05, 0.40),   # a₂
    (0.05, 0.50),   # h₂
    (0.20, 0.90),   # e₂
    (0.10, 0.70),   # d₂
]

FIXED_VALUES = [1.5, 150.0, 0.20, 0.10, 0.50, 0.30, 0.15, 0.10, 0.40, 0.35]

SCENARIOS = {
    # 1D — варіюємо один параметр
    "r":    ([0],    [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "K":    ([1],    [0, 2, 3, 4, 5, 6, 7, 8, 9]),
    "a1":   ([2],    [0, 1, 3, 4, 5, 6, 7, 8, 9]),
    "d1":   ([5],    [0, 1, 2, 3, 4, 6, 7, 8, 9]),
    "a2":   ([6],    [0, 1, 2, 3, 4, 5, 7, 8, 9]),
    "d2":   ([9],    [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    # 2D — варіюємо два параметри
    "r_K":    ([0, 1],  [2, 3, 4, 5, 6, 7, 8, 9]),
    "a1_d1":  ([2, 5],  [0, 1, 3, 4, 6, 7, 8, 9]),
    "a2_d2":  ([6, 9],  [0, 1, 2, 3, 4, 5, 7, 8]),
    "d1_d2":  ([5, 9],  [0, 1, 2, 3, 4, 6, 7, 8]),
    "a1_a2":  ([2, 6],  [0, 1, 3, 4, 5, 7, 8, 9]),
    "e1_e2":  ([4, 8],  [0, 1, 2, 3, 5, 6, 7, 9]),
    # 3D — варіюємо три параметри
    "r_a1_a2": ([0, 2, 6], [1, 3, 4, 5, 7, 8, 9]),
}

_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples")
SAMPLES_DIR = os.path.normpath(_SAMPLES_DIR)


def generate_samples(n: int, varying_indices: list, fixed_indices: list,
                     bounds=None, fixed_values=None, seed=None) -> np.ndarray:
    """
    Генерує n точок методом LHS для варійованих параметрів.

    Повертає матрицю (n, 10) зі стовпцями у порядку PARAM_NAMES.

    Parameters
    ----------
    n               : кількість точок
    varying_indices : індекси параметрів, що варіюються (0-based)
    fixed_indices   : індекси фіксованих параметрів (0-based)
    bounds          : список кортежів (lb, ub) для всіх 10 параметрів
    fixed_values    : список фіксованих значень для всіх 10 параметрів
    seed            : відтворюваний seed для LHS
    """
    if bounds is None:
        bounds = BOUNDS
    if fixed_values is None:
        fixed_values = FIXED_VALUES

    d = len(varying_indices)
    if d == 0:
        raise ValueError("Немає варійованих параметрів")

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_sample = sampler.random(n=n)

    lb = np.array([bounds[i][0] for i in varying_indices])
    ub = np.array([bounds[i][1] for i in varying_indices])
    scaled = qmc.scale(unit_sample, lb, ub)

    samples = np.zeros((n, 10))
    for j, idx in enumerate(varying_indices):
        samples[:, idx] = scaled[:, j]
    for idx in fixed_indices:
        samples[:, idx] = fixed_values[idx]

    return samples


def save_samples(n: int, scenario: str, folder: str = None) -> None:
    """Генерує і зберігає LHS-вибірку для сценарію у CSV."""
    if folder is None:
        folder = SAMPLES_DIR
    if scenario not in SCENARIOS:
        raise ValueError(f"Невідомий сценарій: {scenario}")
    varying, fixed = SCENARIOS[scenario]
    samples = generate_samples(n, varying, fixed)
    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    dir_path = os.path.join(folder, scenario)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"train_N{n}.csv")
    df.to_csv(path, index=False)
    print(f"Збережено {n} точок для {scenario} → {path}")


def generate_all_samples(sizes=None) -> None:
    """Генерує вибірки для всіх сценаріїв і розмірів."""
    if sizes is None:
        sizes = [5, 10, 20, 40, 80]
    for scenario in SCENARIOS:
        for n in sizes:
            save_samples(n, scenario)


def cached_samples(n: int, varying: list, fixed: list,
                   name: str, scenario: str,
                   basedir: str = None) -> np.ndarray:
    """
    Читає LHS-вибірку з диску; якщо кешу нема — генерує, зберігає і повертає.

    Структура файлів:
        basedir/scenario/{name}_N{n}.csv

    Parameters
    ----------
    n        : кількість точок
    varying  : індекси варійованих параметрів (0-based)
    fixed    : індекси фіксованих параметрів (0-based)
    name     : роль вибірки: "train", "test", "adapt_ref", тощо
    scenario : ключ зі SCENARIOS
    basedir  : базова папка (за замовчуванням SAMPLES_DIR)
    """
    if basedir is None:
        basedir = SAMPLES_DIR

    dir_path = os.path.join(basedir, scenario)
    path = os.path.join(dir_path, f"{name}_N{n}.csv")

    if os.path.isfile(path):
        df = pd.read_csv(path)
        return df.values.astype(float)

    X = generate_samples(n, varying, fixed)
    os.makedirs(dir_path, exist_ok=True)
    df = pd.DataFrame(X, columns=PARAM_NAMES)
    df.to_csv(path, index=False)
    print(f"  [sampling] Збережено нові LHS-точки: {path}")
    return X
