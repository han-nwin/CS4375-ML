import numpy as np
from cvxopt import matrix, solvers

# -----------------------------
# GLOBAL MODELS
# -----------------------------
model = {}  # hard-margin (your original training) -> {"w": ..., "b": ...}
soft_model = (
    {}
)  # soft-margin (tuned C via 80/20)      -> {"w": ..., "b": ..., "C": ..., "alpha": ...}


# ========================================================
# Feature map (unchanged)
# ========================================================
def phi_quad(X):
    """
    Quadratic feature map over 4-D inputs.
    Input:
      X: (m,4)
    Return:
      Phi: (m, 14)
    """
    X = np.asarray(X, dtype=float)
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    Phi = np.column_stack(
        [
            x1,
            x2,
            x3,
            x4,
            x1**2,
            x2**2,
            x3**2,
            x4**2,
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x2 * x3,
            x2 * x4,
            x3 * x4,
        ]
    )
    return Phi


# ========================================================
# Utilities (unchanged helpers)
# ========================================================
def check_hard_margin_feasibility(Phi, y, w, b, tol=1e-6):
    margins = y * (Phi @ w + b)
    min_margin = float(margins.min()) if len(margins) else float("inf")
    num_viol = int((margins < 1 - tol).sum())
    return {
        "min_margin": min_margin,
        "num_violations": num_viol,
        "violating_indices": np.where(margins < 1 - tol)[0],
    }


def print_feasibility_report(name, report):
    print(f"--- {name} feasibility (hard-margin constraints) ---")
    print(f"Min y*(w^T phi + b): {report['min_margin']:.6f}")
    print(f"Violations (< 1):   {report['num_violations']}")


def support_vectors_from_margins(X, y, Phi, w, b, atol=1e-6):
    margins = y * (Phi @ w + b)
    sv_idx = np.where(np.isclose(margins, 1.0, atol=atol))[0]
    return sv_idx, margins[sv_idx], X[sv_idx], y[sv_idx]


# ========================================================
# Primal hard-margin SVM (your original)
# ========================================================
def _fit_primal(Phi, y):
    """
    Hard-Margin SVM (Primal Form):
      min_{w,b}  0.5 * ||w||^2
      s.t. y_i * (w^T phi_i + b) >= 1
    Decision variable: z = [w (n), b (1)]
    """
    M, n = Phi.shape

    H = np.zeros((n + 1, n + 1), dtype=np.float64)
    H[:n, :n] = np.eye(n, dtype=np.float64)
    f = np.zeros(n + 1, dtype=np.float64)

    # G z <= h   with G = - diag(y) [Phi, 1],  h = -1
    G = np.zeros((M, n + 1), dtype=np.float64)
    h = -np.ones(M, dtype=np.float64)
    for i in range(M):
        yi = float(y[i])
        G[i, :n] = -yi * Phi[i]
        G[i, n] = -yi

    def to_cvx(a):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return matrix(a)

    H_cvx, f_cvx = to_cvx(H), to_cvx(f)
    G_cvx, h_cvx = to_cvx(G), to_cvx(h)

    solvers.options["show_progress"] = False
    sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx)

    status = sol["status"]
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP solver failed (status: {status}). ")

    z = np.array(sol["x"]).reshape(-1)
    w = z[:n]
    b = float(z[n])
    return w, b


def _train(X, y):
    """
    Train hard-margin SVM in quadratic feature space.
    Stores learned params in global `model`.
    """
    global model
    y = np.where(y >= 0, 1, -1).astype(int)
    Phi = phi_quad(X.astype(float, copy=True))
    w, b = _fit_primal(Phi, y)
    model = {"w": w, "b": b}


def _eval_demo(X):
    """
    Uses current `model` (hard-margin) to predict.
    """
    if not model:
        raise RuntimeError("Model is empty. Call _train(X, y) first.")
    Phi = phi_quad(X.astype(float, copy=True))
    scores = Phi @ model["w"] + model["b"]
    return np.where(scores >= 0, 1, -1)


# ========================================================
# (Public) eval for TA — unchanged, frozen weights/bias
# ========================================================
def eval(X):
    """
    Public eval API for the assignment (unchanged from your version).
    """
    final_model = {
        "w": [
            31.64224203,
            -5.42858361,
            19.62638296,
            60.78431324,
            60.79719603,
            -101.40837995,
            -15.22046704,
            10.96144778,
            -29.88672501,
            10.58137356,
            -1.79809404,
            -34.84917827,
            -61.61341963,
            3.50684772,
        ],
        "b": 2.89888111,
    }
    Phi = phi_quad(X.astype(float, copy=True))
    scores = Phi @ final_model["w"] + final_model["b"]
    return np.where(scores >= 0, 1, -1)


# ========================================================
# NEW: Soft-margin SVM (dual, L1-slack), 80/20 C selection
# ========================================================
def _fit_softmargin_dual(Phi, y, C):
    """
    Dual L1-slack SVM in feature space Phi:
      max_a   1^T a - 0.5 * sum_ij a_i a_j y_i y_j K_ij
      s.t.    0 <= a_i <= C,   sum_i a_i y_i = 0
    Returns w, b, alpha
    """
    M = Phi.shape[0]
    Y = y.astype(float)
    K = Phi @ Phi.T  # linear kernel in Phi-space

    # cvxopt wants: min (1/2)x^T P x + q^T x
    # We maximize, so set q = -1, P = Y Y^T * K
    P = matrix(np.outer(Y, Y) * K)
    q = matrix(-np.ones(M))

    # Bounds 0 <= a <= C  ->  [I; -I] a <= [C; 0]
    G = matrix(np.vstack([np.eye(M), -np.eye(M)]))
    h = matrix(np.hstack([C * np.ones(M), np.zeros(M)]))

    # Equality: y^T a = 0
    A = matrix(Y.reshape(1, -1))
    b0 = matrix(0.0)

    solvers.options["show_progress"] = False
    sol = solvers.qp(P, q, G, h, A, b0)
    if sol["status"] not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP failed: {sol['status']}")

    alpha = np.array(sol["x"]).reshape(-1)

    # Recover primal w in Phi-space: w = Phi^T (alpha * y)
    w = Phi.T @ (alpha * Y)

    # Bias from margin SVs (0 < a_i < C)
    eps = 1e-6
    mask = (alpha > eps) & (alpha < C - eps)
    if not np.any(mask):
        # fallback: any alpha>0
        mask = alpha > eps
    if np.any(mask):
        b_vals = Y[mask] - (Phi[mask] @ w)
        b = float(b_vals.mean())
    else:
        b = 0.0

    return w, b, alpha


def _select_C_80_20(Phi, y, Cs=(0.01, 0.1, 1, 10, 100, 1000), seed=0):
    """
    Simple 80/20 split to pick C.
    Returns (best_C, best_acc, (w_val, b_val))
    """
    M = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(M)
    tr = idx[: int(0.8 * M)]
    va = idx[int(0.8 * M) :]

    best = (None, -1.0, None, None)
    for C in Cs:
        w, b, _ = _fit_softmargin_dual(Phi[tr], y[tr], C)
        pred = np.sign(Phi[va] @ w + b)
        acc = (pred == y[va]).mean()
        if acc > best[1]:
            best = (C, acc, w, b)
    return best


def train_soft_and_freeze(X, y, Cs=(0.01, 0.1, 1, 10, 100, 1000), seed=0):
    """
    Train soft-margin SVM:
      1) pick C on 80/20 split,
      2) retrain on ALL data with best C,
      3) store in global `soft_model`.
    """
    global soft_model
    y = np.where(y >= 0, 1, -1).astype(int)
    Phi = phi_quad(X.astype(float, copy=True))

    best_C, val_acc, _, _ = _select_C_80_20(Phi, y, Cs=Cs, seed=seed)
    w, b, alpha = _fit_softmargin_dual(Phi, y, best_C)

    soft_model = {"w": w, "b": b, "C": float(best_C), "alpha": alpha}
    return best_C, val_acc


def eval_soft(X):
    """
    Predict with the tuned soft-margin model (must call train_soft_and_freeze first).
    """
    if not soft_model:
        raise RuntimeError(
            "Soft-margin model is empty. Call train_soft_and_freeze(X, y) first."
        )
    Phi = phi_quad(X.astype(float, copy=True))
    scores = Phi @ soft_model["w"] + soft_model["b"]
    return np.where(scores >= 0, 1, -1)


# ========================================================
# Main: trains both models and compares outputs
# ========================================================
if __name__ == "__main__":
    # Load full dataset
    data = np.loadtxt("mystery.data", delimiter=",")
    X = data[:, :4].astype(float)
    y = np.where(data[:, 4].astype(int) >= 0, 1, -1)

    # ---------------- Hard-margin (original) ----------------
    print("==== HARD-MARGIN (PRIMAL) ====")
    _train(X, y)
    Phi = phi_quad(X.astype(float, copy=True))
    rep = check_hard_margin_feasibility(Phi, y, model["w"], model["b"], tol=1e-6)
    print_feasibility_report("ALL DATA", rep)

    w_h, b_h = model["w"], model["b"]
    norm_w_h = np.linalg.norm(w_h)
    margin_h = 1.0 / (norm_w_h + 1e-12)
    print(f"||w|| = {norm_w_h:.6f}, margin = {margin_h:.6f}")
    sv_idx_h, sv_marg_h, sv_X_h, sv_y_h = support_vectors_from_margins(
        X, y, Phi, w_h, b_h, atol=1e-6
    )
    print(f"# support vectors (≈ via margin=1): {len(sv_idx_h)}")
    # Print w,b in copy-paste form
    print("\n[HARD] w =", np.array2string(w_h, separator=", "))
    print("[HARD] b =", f"{b_h:.8f}")

    # ---------------- Soft-margin (dual, tuned C) ----------------
    print("\n==== SOFT-MARGIN (DUAL, L1-SLACK, 80/20 C PICK) ====")
    best_C, val_acc = train_soft_and_freeze(
        X, y, Cs=(0.01, 0.1, 1, 10, 100, 1000), seed=0
    )
    w_s, b_s = soft_model["w"], soft_model["b"]
    norm_w_s = np.linalg.norm(w_s)
    margin_s = 1.0 / (norm_w_s + 1e-12)
    print(f"Chosen C: {best_C}  (80/20 val acc = {val_acc:.3f})")
    print(f"||w|| = {norm_w_s:.6f}, margin = {margin_s:.6f}")
    # Support vectors via alpha (0<alpha<C)
    alpha = soft_model["alpha"]
    eps = 1e-6
    sv_dual_mask = (alpha > eps) & (alpha < best_C - eps)
    print(f"# support vectors (dual 0<alpha<C): {int(sv_dual_mask.sum())}")

    # ---------------- Compare predictions ----------------
    print("\n==== PREDICTION COMPARISON (on ALL labeled data) ====")
    pred_h = np.where(Phi @ w_h + b_h >= 0, 1, -1)
    pred_s = np.where(Phi @ w_s + b_s >= 0, 1, -1)

    acc_h = (pred_h == y).mean()
    acc_s = (pred_s == y).mean()
    diff = pred_h != pred_s
    n_diff = int(diff.sum())

    print(f"Hard-margin accuracy: {acc_h*100:.2f}%")
    print(f"Soft-margin accuracy: {acc_s*100:.2f}%")
    print(f"Predictions differ on {n_diff} / {len(y)} samples.")

    if n_diff > 0:
        diff_idx = np.where(diff)[0][:20]  # show up to 20 indices
        print("Indices where they differ (up to 20):", diff_idx.tolist())
