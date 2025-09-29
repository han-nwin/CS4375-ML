# Problem1.py
import numpy as np
from cvxopt import matrix, solvers

# -----------------------------
# GLOBAL MODEL (populated by train)
# -----------------------------
model = {}  # {"w": ..., "b": ...}


# Quadratic feature map phi(x)
def phi_quad(X):
    """
    X: raw inputs
    return Phi: quadratic features
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
# Primal SVM (Quadratic Programming)
# -----------------------------
def _fit_primal(Phi, y):
    """
    Hard-Margin SVM (Primal Form):
      min_{w,b}  0.5 * ||w||^2
      subject to y_i * (w^T phi_i + b) >= 1,  for all i

    Decision variable: z = [w (n-dim), b (scalar)]
    """

    # Number of samples (M) and feature dimension (n)
    M, n = Phi.shape

    # 1. Build quadratic program terms
    H = np.zeros((n + 1, n + 1), dtype=np.float64)
    # Create identity matrix for quadratic term in objective (0.5 * w^T w)
    H[:n, :n] = np.eye(n, dtype=np.float64)

    # Linear term f (size n+1):
    # - All zeros since we has no linear term
    f = np.zeros(n + 1, dtype=np.float64)

    # 2. Build constraints on the entire data set
    # Original constraint: y_i (w^T phi_i + b) >= 1
    # Rewritten for QP form
    # (Gz <= h): -y_i (phi_i^T w + b) <= -1
    G = np.zeros((M, n + 1), dtype=np.float64)
    h = -np.ones(M, dtype=np.float64)
    for i in range(M):
        yi = float(y[i])

        # Assign to the i-th row and columns from 0 to n-1 of matrix G
        # coefficients for w
        G[i, :n] = -yi * Phi[i]

        # coefficient for bias b
        G[i, n] = -yi

    # 3. Convert numpy arrays -> cvxopt format
    def to_cvx(a):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return matrix(a)

    H_cvx, f_cvx = to_cvx(H), to_cvx(f)
    G_cvx, h_cvx = to_cvx(G), to_cvx(h)

    # 4. Solve the quadratic program
    solvers.options["show_progress"] = True  # print solver iterations
    sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx)

    status = sol["status"]
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP solver failed (status: {status}). ")

    # -------------------------------
    # 5. Extract solution
    # -------------------------------
    z = np.array(sol["x"]).reshape(-1)
    w = z[:n]  # learned weights
    b = float(z[n])  # learned bias
    return w, b


# -----------------------------
# Train model on given data set
# -----------------------------
def _train(X, y):
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

    w, b = _fit_primal(Phi, y)

    model = {"w": w, "b": b}


# -----------------------------
# Private eval to get w and b
# -----------------------------
def _eval_demo(X):
    """
    Input: X (m x 4) matrix.
    Output: +-1 predictions using learned weights/bias from train().
    Assumes train() has been called to populate model.
    """
    if not model:
        raise RuntimeError("Model is empty. Call train(X, y) first.")
    Phi = phi_quad(X.astype(float, copy=True))
    scores = Phi @ model["w"] + model["b"]
    return np.where(scores >= 0, 1, -1)


# -----------------------------
# For TA: REQUIRED EXPORT
# Public eval API for the assignment
# -----------------------------
def eval(X):
    """
    Input: X (m x 4) matrix.
    Output: +-1 predictions using the already learned weights/bias.
    """
    final_model = {
        "w": [
            29.19814671,
            0.32730614,
            20.46508506,
            52.31347643,
            65.44290667,
            -93.4816874,
            -14.98304077,
            4.55277202,
            -43.1404392,
            13.26671215,
            -2.1563674,
            -39.70198944,
            -46.05307219,
            7.0634797,
        ],
        "b": 1.490485,
    }
    Phi = phi_quad(X.astype(float, copy=True))
    scores = Phi @ final_model["w"] + final_model["b"]
    return np.where(scores >= 0, 1, -1)


# Train/Test 80/20 split helper
def _train_test_split(X, y, test_size=0.2, seed=42):
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


# Demo Run
if __name__ == "__main__":
    data = np.loadtxt("mystery.data", delimiter=",")
    X_all = data[:, :4].astype(float)
    y_all = np.where(data[:, 4].astype(int) >= 0, 1, -1)

    train_indices, test_indices = _train_test_split(
        X_all, y_all, test_size=0.20, seed=42
    )
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    X_test, y_test = X_all[test_indices], y_all[test_indices]

    # train and evaluate
    _train(X_train, y_train)
    y_train_pred = _eval_demo(X_train)
    y_test_pred = _eval_demo(X_test)

    acc_train = (y_train_pred == y_train).mean()
    acc_test = (y_test_pred == y_test).mean()
    # margin (1/||w||)
    margin_val = 1.0 / max(np.linalg.norm(model["w"]), 1e-12)

    print("------ DEMO RESULTS (80/20) -----")
    print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
    print(f"Margin (1/||w||): {margin_val:.6f}")
    print(f"Bias b: {model['b']:.6f}")
    print("Weights w:", model["w"])
    print(f"Train accuracy: {acc_train * 100:.2f}%")
    print(f"Test accuracy: {acc_test * 100:.2f}%")

    print("------ Test public eval API -----")
    print(
        f"Test public eval API: {eval(np.array([[0.905791937075619, 0.892267188231107, 0.955372102971021, 0.739832227063707], [0.126986816293506, 0.24260338627967, 0.724247033520084, 0.88899234515285]]))}"
    )
