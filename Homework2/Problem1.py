# Problem1.py
import numpy as np
from cvxopt import matrix, solvers

# -----------------------------
# GLOBAL MODEL (populated by train)
# -----------------------------
model = {}  # {"w": ..., "b": ...}

# -----------------------------
# Quadratic feature map phi(x)
# -----------------------------
FEATURE_ORDER = [
    "x1",
    "x2",
    "x3",
    "x4",
    "x1^2",
    "x2^2",
    "x3^2",
    "x4^2",
    "x1*x2",
    "x1*x3",
    "x1*x4",
    "x2*x3",
    "x2*x4",
    "x3*x4",
]


def phi_quad(X):
    """
    X: (n,4) raw inputs
    return Phi: (n,d) quadratic features
    """
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


# -----------------------------
# Hard-margin primal SVM (QP)
# -----------------------------
def fit_primal_hard_margin(Phi, y, solver_opts=None):
    """
    Hard-margin SVM:
      min_{w,b} 0.5||w||^2
      such that: y_i (w^T phi_i + b) >= 1
    Decision vector z = [w (d), b (1)]
    """
    n, d = Phi.shape
    D = d + 1

    H = np.zeros((D, D), dtype=np.float64)
    H[:d, :d] = np.eye(d, dtype=np.float64)
    f = np.zeros(D, dtype=np.float64)

    # Constraints: -y_i*(phi_i^T w + b) <= -1
    G = np.zeros((n, D), dtype=np.float64)
    h = -np.ones(n, dtype=np.float64)
    for i in range(n):
        yi = float(y[i])
        G[i, :d] = -yi * Phi[i]
        G[i, d] = -yi

    def to_cvx(a):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return matrix(a)

    H_cvx, f_cvx = to_cvx(H), to_cvx(f)
    G_cvx, h_cvx = to_cvx(G), to_cvx(h)

    solvers.options["show_progress"] = True
    if solver_opts:
        solvers.options.update(solver_opts)

    sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx)
    status = sol["status"]
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"Hard-margin QP failed (status: {status}). "
            "Data may not be separable with this feature map; "
            "switch to soft-margin if needed."
        )

    z = np.array(sol["x"]).reshape(-1)
    w = z[:d]
    b = float(z[d])
    return w, b


# -----------------------------
# Public train API required by the assignment
# -----------------------------
def train(X, y):
    """
    Learn hard-margin SVM in quadratic feature space
    and store learned params in the module-level `model` dict.
    Inputs:
      X: (n,4) features
      y: (n,) labels in {+1, -1}
    """
    global model
    y = np.where(y >= 0, 1, -1).astype(int)

    # No standardization: use raw inputs
    Phi = phi_quad(X.astype(float, copy=True))

    w, b = fit_primal_hard_margin(
        Phi, y, solver_opts=dict(abstol=1e-9, reltol=1e-8, feastol=1e-9)
    )

    model = {"w": w, "b": b}


# -----------------------------
# Public eval API required by the assignment
# -----------------------------
def eval(X):
    """
    Input: X (m x 4) matrix.
    Output: +-1 predictions using the already learned weights/bias.
    Assumes train() has been called to populate model.
    """
    if not model:
        raise RuntimeError("Model is empty. Call train(X, y) first.")
    Phi = phi_quad(X.astype(float, copy=True))
    scores = Phi @ model["w"] + model["b"]
    return np.where(scores >= 0, 1, -1)


# -----------------------------
# Train/Test 80/20 split
# -----------------------------
def _train_test_split_stratified(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(test_size * len(idx))))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


if __name__ == "__main__":
    # Demo run
    data = np.loadtxt("mystery.data", delimiter=",")
    X_all = data[:, :4].astype(float)
    y_all = np.where(data[:, 4].astype(int) >= 0, 1, -1)

    train_indices, test_indices = _train_test_split_stratified(
        X_all, y_all, test_size=0.20, seed=42
    )
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    X_test, y_test = X_all[test_indices], y_all[test_indices]

    # train and evaluate
    train(X_train, y_train)
    y_train_pred = eval(X_train)
    y_test_pred = eval(X_test)

    acc_train = (y_train_pred == y_train).mean()
    acc_test = (y_test_pred == y_test).mean()
    # margin (1/||w||)
    margin_val = 1.0 / max(np.linalg.norm(model["w"]), 1e-12)

    print("------ DEMO RESULTS (80/20) -----")
    print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
    print(f"Margin (1/||w||): {margin_val:.6f}")
    print(f"Bias b: {model['b']:.6f}")
    print("Weights w:", model["w"])
    print(f"Train accuracy: {acc_train*100:.2f}%")
    print(f"Test accuracy: {acc_test*100:.2f}%")
