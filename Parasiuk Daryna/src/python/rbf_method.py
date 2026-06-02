"""
rbf_method.py
-------------
RBF-сурогатна модель для системи Холлінга II.

Підтримує три режими:
  :standard — звичайна RBF (без градієнтів)
  :hermite  — Hermite RBF з градієнтним розширенням (use_gradients=True)
  :adaptive — адаптивне додавання точок (adaptive_train_rbf)

Ядра:
  gaussian             k(r) = exp(-ε·r²)
  multiquadric         k(r) = √(1 + ε²·r²)
  inverse_multiquadric k(r) = 1/√(1 + ε²·r²)

Нормалізація входів і виходів: z-score по кожній координаті.
log_transform=True: навчаємо в log-просторі (ln ψ) → exp при передбаченні.
"""

import numpy as np
from numpy.linalg import norm, cond, lstsq
import time

try:
    from .sampling import SCENARIOS, generate_samples
    from .model import simulate as model_simulate
except ImportError:
    from sampling import SCENARIOS, generate_samples
    from model import simulate as model_simulate


# ─────────────────────────────────────────────────────────────────────────────
# Ядра та їх похідні
# ─────────────────────────────────────────────────────────────────────────────

def gaussian(r, eps):
    return np.exp(-eps * r ** 2)

def d_gaussian(r, eps):
    return -2 * eps * r * np.exp(-eps * r ** 2)

def multiquadric(r, eps):
    return np.sqrt(1.0 + eps ** 2 * r ** 2)

def d_multiquadric(r, eps):
    return (eps ** 2 * r) / np.sqrt(1.0 + eps ** 2 * r ** 2)

def inverse_multiquadric(r, eps):
    return 1.0 / np.sqrt(1.0 + eps ** 2 * r ** 2)

def d_inverse_multiquadric(r, eps):
    return -(eps ** 2 * r) / (1.0 + eps ** 2 * r ** 2) ** 1.5


def get_kernel(kernel_type: str, eps: float):
    """Повертає (k(r), dk/dr(r)) для заданого ядра і параметра форми ε."""
    if kernel_type == "gaussian":
        return (lambda r: gaussian(r, eps)), (lambda r: d_gaussian(r, eps))
    elif kernel_type == "multiquadric":
        return (lambda r: multiquadric(r, eps)), (lambda r: d_multiquadric(r, eps))
    elif kernel_type == "inverse_multiquadric":
        return (lambda r: inverse_multiquadric(r, eps)), (lambda r: d_inverse_multiquadric(r, eps))
    else:
        raise ValueError(f"Невідомий тип ядра: {kernel_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Структура моделі
# ─────────────────────────────────────────────────────────────────────────────

class RBFModel:
    """
    Навчена RBF-модель.

    Attributes
    ----------
    centers       : np.ndarray (n, d)  — нормалізовані навчальні точки
    weights       : np.ndarray (n, p)  — ваги (p = кількість виходів)
    kernel_fn     : Callable           — функція ядра k(r)
    eps           : float              — параметр форми ε
    train_time    : float              — час навчання (секунди)
    loocv_selected: bool               — чи підбирався ε через LOOCV
    x_mean        : np.ndarray (d,)
    x_std         : np.ndarray (d,)
    y_mean        : np.ndarray (p,)
    y_std         : np.ndarray (p,)
    log_transform : bool
    """
    def __init__(self, centers, weights, kernel_fn, eps,
                 train_time, loocv_selected,
                 x_mean, x_std, y_mean, y_std, log_transform):
        self.centers        = centers
        self.weights        = weights
        self.kernel_fn      = kernel_fn
        self.eps            = eps
        self.train_time     = train_time
        self.loocv_selected = loocv_selected
        self.x_mean         = x_mean
        self.x_std          = x_std
        self.y_mean         = y_mean
        self.y_std          = y_std
        self.log_transform  = log_transform


# ─────────────────────────────────────────────────────────────────────────────
# Обчислення градієнтів центральними різницями
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradients(f, x, h=1e-4, n_average=3, max_grad_norm=1e4):
    """
    Чисельний якобіан f у точці x центральними різницями.

    Крок адаптивний: hᵢ = max(h, 1e-2·|xᵢ|).
    Результат усереднюється по n_average незалежних обчисленнях.
    Норма обмежується max_grad_norm.

    Parameters
    ----------
    f   : Callable, x -> np.ndarray shape (p,)
    x   : np.ndarray shape (d,)

    Returns
    -------
    np.ndarray shape (d, p)  — якобіан ∂f/∂x
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    fx = np.asarray(f(x), dtype=float)
    p = len(fx)
    grad = np.zeros((d, p))

    for i in range(d):
        hi = max(h, 1e-2 * abs(x[i]))
        if hi == 0.0:
            hi = 1e-4

        avg = np.zeros(p)
        for _ in range(n_average):
            xp = x.copy(); xp[i] += hi
            xm = x.copy(); xm[i] -= hi
            avg += (np.asarray(f(xp), dtype=float) -
                    np.asarray(f(xm), dtype=float)) / (2 * hi)
        grad[i, :] = avg / n_average

    gnorm = np.linalg.norm(grad)
    if gnorm > max_grad_norm:
        grad *= max_grad_norm / gnorm

    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Регуляризований МНК
# ─────────────────────────────────────────────────────────────────────────────

def solve_regularized_least_squares(A, B, lam):
    """Тихонівська регуляризація: мінімізує ‖Aw − B‖² + λ‖w‖²."""
    if lam > 0:
        n = A.shape[1]
        A = np.vstack([A, np.sqrt(lam) * np.eye(n)])
        B = np.vstack([B, np.zeros((n, B.shape[1]))])
    W, _, _, _ = lstsq(A, B, rcond=None)
    return W


# ─────────────────────────────────────────────────────────────────────────────
# LOOCV для вибору ε
# ─────────────────────────────────────────────────────────────────────────────

def loocv_select_eps(X, y, kernel_type,
                     lam=1e-6,
                     candidates=None,
                     use_gradients=False,
                     grad_data=None,
                     w_grad=0.1):
    """
    Вибір оптимального ε методом залишення одного (LOOCV).

    Parameters
    ----------
    X           : np.ndarray (n, d) — нормалізовані входи
    y           : np.ndarray (n, p)
    kernel_type : str
    lam         : float — параметр регуляризації
    candidates  : list[float] — сітка ε для пошуку
    use_gradients: bool — якщо True, LOOCV на повній системі функцій + градієнтів
    grad_data   : dict з ключами 'Xn', 'grads', 'x_std', 'y_std', 'yn'
    w_grad      : float — вага градієнтних рядків

    Returns
    -------
    best_eps : float
    loocv_errors : np.ndarray
    """
    if candidates is None:
        candidates = np.logspace(-1, 1, 20)

    if not use_gradients:
        return _loocv_standard(X, y, kernel_type, lam, candidates)
    return _loocv_full(X, y, kernel_type, lam, candidates, grad_data, w_grad)


def _loocv_standard(X, y, kernel_type, lam, candidates):
    n = X.shape[0]
    best_eps = candidates[0]
    best_err = np.inf
    loocv_errors = np.full(len(candidates), np.inf)

    for ci, eps_cand in enumerate(candidates):
        kfn, _ = get_kernel(kernel_type, eps_cand)
        Phi = np.array([[kfn(norm(X[i] - X[j])) for j in range(n)] for i in range(n)])
        A = Phi + lam * np.eye(n)

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue

        w = A_inv @ y[:, 0]
        diag_inv = np.diag(A_inv)
        if np.any(np.abs(diag_inv) < 1e-14):
            continue

        loo_resid = w / diag_inv
        err = float(np.mean(loo_resid ** 2))
        loocv_errors[ci] = err
        if err < best_err:
            best_err = err
            best_eps = eps_cand

    return best_eps, loocv_errors


def _loocv_full(X, y, kernel_type, lam, candidates, grad_data, w_grad):
    n, d = X.shape
    best_eps = candidates[0]
    best_err = np.inf
    loocv_errors = np.full(len(candidates), np.inf)

    Xn    = grad_data['Xn']
    grads = grad_data['grads']
    y_std = grad_data['y_std']
    x_std = grad_data['x_std']
    yn    = grad_data['yn']

    for ci, eps_cand in enumerate(candidates):
        kfn, dkfn = get_kernel(kernel_type, eps_cand)
        Phi_f = np.array([[kfn(norm(Xn[i] - Xn[j])) for j in range(n)] for i in range(n)])
        Phi_g, y_g = _build_gradient_system(Xn, grads, x_std, y_std, kfn, dkfn)

        err_sum = 0.0
        for held in range(n):
            idx_f = [i for i in range(n) if i != held]
            idx_g = [i for i in range(n * d) if not (held * d <= i < (held + 1) * d)]

            A_f = Phi_f[np.ix_(idx_f, list(range(n)))]
            A_g = Phi_g[idx_g, :]
            b_f = yn[idx_f, :]
            b_g = y_g[idx_g, :]

            A_u = np.vstack([A_f, w_grad * A_g])
            b_u = np.vstack([b_f, w_grad * b_g])
            W   = solve_regularized_least_squares(A_u, b_u, lam)

            pred_f = Phi_f[held, :] @ W
            pred_g = Phi_g[held * d:(held + 1) * d, :] @ W

            err_sum += float(np.sum((pred_f - yn[held, :]) ** 2))
            err_sum += w_grad ** 2 * float(np.sum((pred_g - y_g[held * d:(held + 1) * d, :]) ** 2))

        loocv_errors[ci] = err_sum / n
        if loocv_errors[ci] < best_err:
            best_err = loocv_errors[ci]
            best_eps = eps_cand

    return best_eps, loocv_errors


def _build_gradient_system(Xn, grads, x_std, y_std, kfn, dkfn):
    """
    Будує матрицю Phi_g (n*d, n) і вектор y_g (n*d, p) для градієнтних рядків.

    Рядок (k*d + l) відповідає похідній ядра по l-й координаті у точці k:
      ∂φ(‖Xn[k] - Xn[j]‖)/∂x_l = dkfn(r) · (Xn[k,l] - Xn[j,l]) / r

    Градієнти масштабуємо: y_g[k*d+l, c] = grads[k][l, c] · x_std[l] / y_std[c]
    """
    n, d = Xn.shape
    p = grads[0].shape[1]
    Phi_g = np.zeros((n * d, n))
    y_g   = np.zeros((n * d, p))

    for k in range(n):
        for j in range(n):
            r = norm(Xn[k] - Xn[j])
            if r > 1e-14:
                dk = dkfn(r)
                for l in range(d):
                    Phi_g[k * d + l, j] = dk * (Xn[k, l] - Xn[j, l]) / r

        for l in range(d):
            for c in range(p):
                y_g[k * d + l, c] = grads[k][l, c] * x_std[l] / y_std[c]

    return Phi_g, y_g


# ─────────────────────────────────────────────────────────────────────────────
# Навчання моделі
# ─────────────────────────────────────────────────────────────────────────────

def train_rbf(X, y, kernel_type="multiquadric",
              eps=5.0, lam=1e-6,
              use_gradients=False,
              w_grad=-1.0,
              auto_eps=False,
              eps_candidates=None,
              grad_h=1e-4,
              grad_n_average=1,
              max_grad_norm=1e4,
              sim_fn=None,
              log_transform=False):
    """
    Навчання RBF-сурогата.

    Parameters
    ----------
    X             : np.ndarray (n, d) — вхідні точки
    y             : np.ndarray (n, p) — виходи
    kernel_type   : str — "gaussian", "multiquadric", "inverse_multiquadric"
    eps           : float — параметр форми (ігнорується якщо auto_eps=True)
    lam           : float — Тихонівська регуляризація
    use_gradients : bool  — Hermite-розширення (градієнтне)
    w_grad        : float — вага градієнтних рядків (<0 → авто)
    auto_eps      : bool  — вибір ε через LOOCV
    eps_candidates: list[float] — сітка для LOOCV
    grad_h        : float — крок центральної різниці
    grad_n_average: int   — кількість усереднень для градієнта
    max_grad_norm : float — обмеження норми градієнта
    sim_fn        : Callable or None — функція b->ψ(b); за замовчуванням model.simulate
    log_transform : bool  — навчати у log-просторі

    Returns
    -------
    RBFModel
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n, d    = X.shape
    n_out   = y.shape[1]

    if eps_candidates is None:
        eps_candidates = np.logspace(-2, 2, 30)

    # z-score нормалізація входів
    x_mean = X.mean(axis=0)
    x_std  = X.std(axis=0)
    x_std[x_std < 1e-10] = 1.0
    Xn = (X - x_mean) / x_std

    # log-трансформація виходів
    y_orig = y.copy()
    if log_transform:
        y = np.log1p(np.maximum(y, 0.0))

    # z-score нормалізація виходів
    y_mean = y.mean(axis=0)
    y_std  = y.std(axis=0)
    y_std[y_std < 1e-10] = 1.0
    yn = (y - y_mean) / y_std

    if use_gradients:
        _fn = sim_fn if sim_fn is not None else model_simulate
        grads = [compute_gradients(_fn, X[i],
                                   h=grad_h, n_average=grad_n_average,
                                   max_grad_norm=max_grad_norm)
                 for i in range(n)]
        if log_transform:
            for k in range(n):
                for c in range(n_out):
                    grads[k][:, c] /= max(1.0 + y_orig[k, c], 1e-10)

    if auto_eps:
        if use_gradients:
            grad_data = {'Xn': Xn, 'grads': grads, 'x_std': x_std, 'y_std': y_std, 'yn': yn}
            eps, _ = loocv_select_eps(Xn, yn, kernel_type,
                                      lam=lam, candidates=eps_candidates,
                                      use_gradients=True, grad_data=grad_data,
                                      w_grad=abs(w_grad) if w_grad >= 0 else 0.1)
        else:
            eps, _ = loocv_select_eps(Xn, yn, kernel_type,
                                      lam=lam, candidates=eps_candidates)
        print(f"  [LOOCV] Обраний ε = {eps:.4f}")

    t_start = time.time()
    kernel_fn, dkernel_fn = get_kernel(kernel_type, eps)

    if not use_gradients:
        Phi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Phi[i, j] = kernel_fn(norm(Xn[i] - Xn[j]))

        cond_phi  = cond(Phi)
        cond_gram = cond(Phi.T @ Phi)
        print(f"  [diag] cond(Φ)={cond_phi:.2g}, cond(Φ'Φ)={cond_gram:.2g}, "
              f"norm(y_f)={norm(yn.ravel()):.3g}, ε={eps:.4f}")

        A = Phi + lam * np.eye(n)
        try:
            weights = np.linalg.solve(A, yn)
        except np.linalg.LinAlgError:
            weights, _, _, _ = lstsq(A, yn, rcond=None)

    else:
        Phi_f = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Phi_f[i, j] = kernel_fn(norm(Xn[i] - Xn[j]))

        Phi_g, y_g = _build_gradient_system(Xn, grads, x_std, y_std, kernel_fn, dkernel_fn)

        norm_yf = norm(yn.ravel())
        norm_yg = norm(y_g.ravel())

        if w_grad < 0:
            pure_w = min(0.1, norm_yf / max(norm_yg, np.finfo(float).eps))
        else:
            pure_w = w_grad

        grad_scale = 1.0 / max(norm_yg, np.finfo(float).eps)
        Phi_g = Phi_g * grad_scale
        y_g   = y_g   * grad_scale

        print(f"  [diag] norm(y_f)={norm_yf:.3g}, norm(y_g)={norm_yg:.3g}, "
              f"w_grad={pure_w:.3g}, ε={eps:.4f}")

        A_u = np.vstack([Phi_f, pure_w * Phi_g])
        cond_A = cond(A_u)
        if cond_A > 1e8:
            lam = max(lam, min(1e-8 * cond_A / 1e8, 1e-4))
        print(f"  [diag] cond(A_u)={cond_A:.2g}, λ={lam:.3g}")

        b_u = np.vstack([yn, pure_w * y_g])
        weights = solve_regularized_least_squares(A_u, b_u, lam)

    t_elapsed = time.time() - t_start

    return RBFModel(
        centers=Xn,
        weights=weights,
        kernel_fn=kernel_fn,
        eps=eps,
        train_time=t_elapsed,
        loocv_selected=auto_eps,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        log_transform=log_transform,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Передбачення
# ─────────────────────────────────────────────────────────────────────────────

def predict_rbf(model: RBFModel, x) -> np.ndarray:
    """
    Передбачення у точці x.

    Parameters
    ----------
    model : RBFModel
    x     : array-like shape (d,)

    Returns
    -------
    np.ndarray shape (p,)
    """
    x  = np.asarray(x, dtype=float)
    xn = (x - model.x_mean) / model.x_std

    n    = model.centers.shape[0]
    pred = np.zeros(model.weights.shape[1])

    for i in range(n):
        r = norm(xn - model.centers[i])
        pred += model.weights[i] * model.kernel_fn(r)

    pred_denorm = pred * model.y_std + model.y_mean
    if model.log_transform:
        return np.expm1(pred_denorm)
    return np.maximum(0.0, pred_denorm)


# ─────────────────────────────────────────────────────────────────────────────
# Адаптивне навчання
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_train_rbf(
    scenario: str,
    initial_n: int,
    kernel_type: str = "multiquadric",
    X_init=None,
    y_init=None,
    max_iter: int = 2,
    lam: float = 1e-6,
    w_grad: float = 0.1,
    test_n: int = 100,
    use_gradients: bool = True,
    auto_eps: bool = True,
    eps: float = 5.0,
    eps_candidates=None,
    adapt_method: str = "gradient",
    grad_h: float = 1e-4,
    max_grad_norm: float = 1e4,
    log_transform: bool = False,
    adapt_ref=None,
    sim_fn=None,
):
    """
    Адаптивне навчання RBF з двома стратегіями додавання точок:
      adapt_method = "error"    — додає точки в зонах найбільшої похибки прогнозу
      adapt_method = "gradient" — додає точки в зонах найбільшого |∇ψ(b)|

    Parameters
    ----------
    scenario   : str
    initial_n  : int — початковий розмір вибірки
    kernel_type: str
    X_init, y_init : np.ndarray or None — початкові навчальні дані
    max_iter   : int — кількість адаптивних ітерацій
    adapt_ref  : np.ndarray or None — зовнішній пул для адаптивного семплювання
    sim_fn     : Callable or None — функція симуляції

    Returns
    -------
    (model, X_train, history)
    history : list of dict з ключами iter, n_total, rmse, criterion
    """
    if eps_candidates is None:
        eps_candidates = np.logspace(-2, 2, 30)

    _sim = sim_fn if sim_fn is not None else model_simulate

    varying, fixed = SCENARIOS[scenario]

    if X_init is not None and y_init is not None:
        X_train = np.array(X_init, dtype=float)
        y_train = np.array(y_init, dtype=float)
    else:
        X_train = generate_samples(initial_n, varying, fixed)
        y_train = np.array([_sim(X_train[i]) for i in range(initial_n)])

    if adapt_ref is not None:
        X_pool = np.array(adapt_ref, dtype=float)
    else:
        X_pool = generate_samples(test_n, varying, fixed)

    y_pool = np.array([_sim(X_pool[i]) for i in range(len(X_pool))])

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_pool.ndim == 1:
        y_pool = y_pool.reshape(-1, 1)

    X_ref_eval = X_pool.copy()
    y_ref_eval = y_pool.copy()

    history = []

    for it in range(1, max_iter + 1):
        model = train_rbf(
            X_train, y_train, kernel_type,
            eps=eps, lam=lam, w_grad=w_grad,
            use_gradients=use_gradients,
            auto_eps=auto_eps,
            eps_candidates=eps_candidates,
            log_transform=log_transform,
            sim_fn=sim_fn,
        )

        n_pool = len(X_pool)
        n_add  = min(max(1, initial_n // 10), n_pool)

        if adapt_method == "gradient":
            # Рахуємо градієнт тільки по varying-вимірах (2–3 виклики ОДУ замість 20)
            criterion = []
            for i in range(n_pool):
                x_i = X_pool[i]
                g_vals = []
                for vi in varying:
                    hi = max(grad_h, 1e-2 * abs(x_i[vi]))
                    xp = x_i.copy(); xp[vi] += hi
                    xm = x_i.copy(); xm[vi] -= hi
                    diff = (np.asarray(_sim(xp), dtype=float) -
                            np.asarray(_sim(xm), dtype=float)) / (2 * hi)
                    g_vals.append(diff[0])
                criterion.append(float(norm(g_vals)))
        else:
            criterion = [
                float(norm(predict_rbf(model, X_pool[i]) - y_pool[i]))
                for i in range(n_pool)
            ]

        top_idx = np.argsort(criterion)[::-1][:n_add]

        new_points = X_pool[top_idx]
        y_new      = y_pool[top_idx]

        mask   = np.ones(n_pool, dtype=bool)
        mask[top_idx] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]

        X_train = np.vstack([X_train, new_points])
        y_train = np.vstack([y_train, y_new])

        n_eval     = len(X_ref_eval)
        y_pred_ref = np.array([predict_rbf(model, X_ref_eval[i]) for i in range(n_eval)])
        rmse_iter  = float(np.sqrt(np.mean((y_ref_eval[:, 0] - y_pred_ref[:, 0]) ** 2)))

        history.append({
            'iter':      it,
            'n_total':   len(X_train),
            'rmse':      rmse_iter,
            'criterion': criterion,
        })
        print(f"  [адаптація {adapt_method} {it}/{max_iter}] "
              f"додано {n_add}, всього: {len(X_train)},  RMSE={rmse_iter:.2f}")

    model = train_rbf(
        X_train, y_train, kernel_type,
        eps=eps, lam=lam, w_grad=w_grad,
        use_gradients=use_gradients,
        auto_eps=auto_eps,
        eps_candidates=eps_candidates,
        log_transform=log_transform,
        sim_fn=sim_fn,
    )

    return model, X_train, history


# ─────────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Обчислює метрики точності.

    Parameters
    ----------
    y_true, y_pred : array-like shape (n,)

    Returns
    -------
    dict з ключами: rmse_orig, r2_orig, rmse_log, r2_log
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    err       = y_true - y_pred
    rmse_orig = float(np.sqrt(np.mean(err ** 2)))
    ss_res    = float(np.sum(err ** 2))
    ss_tot    = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_orig   = float('nan') if ss_tot < np.finfo(float).eps else 1.0 - ss_res / ss_tot

    lt = np.log1p(np.maximum(y_true, 0.0))
    lp = np.log1p(np.maximum(y_pred, 0.0))
    err_log   = lt - lp
    rmse_log  = float(np.sqrt(np.mean(err_log ** 2)))
    ss_res_l  = float(np.sum(err_log ** 2))
    ss_tot_l  = float(np.sum((lt - lt.mean()) ** 2))
    r2_log    = float('nan') if ss_tot_l < np.finfo(float).eps else 1.0 - ss_res_l / ss_tot_l

    return {
        'rmse_orig': rmse_orig,
        'r2_orig':   r2_orig,
        'rmse_log':  rmse_log,
        'r2_log':    r2_log,
    }
