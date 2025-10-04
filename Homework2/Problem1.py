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

    # Convert to cvxopt format
    H = matrix(H)
    f = matrix(f)
    G = matrix(G)
    h = matrix(h)

    # 3. Solve the quadratic program
    solvers.options["show_progress"] = True  # print solver iterations
    sol = solvers.qp(H, f, G, h)

    status = sol["status"]
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP solver failed (status: {status}). ")

    # 4. Extract solution
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


def check_hard_margin_feasibility(Phi, y, w, b, tol=1e-6):
    """
    For hard-margin SVM:
      constraints: y_i (w^T phi_i + b) >= 1
    Returns a dict with min margin and number of violations (< 1 - tol).
    """
    margins = y * (Phi @ w + b)
    min_margin = float(margins.min()) if len(margins) else float("inf")
    num_viol = int((margins < 1 - tol).sum())
    return {
        "min_margin": min_margin,
        "num_violations": num_viol,
        "violating_indices": np.where(margins < 1 - tol)[0],
    }


def print_feasibility_report(name, report):
    print(f"--- {name} feasibility (hard-margin) ---")
    print(f"Min y*(w^T phi + b): {report['min_margin']:.6f}")
    print(f"Violations (< 1):   {report['num_violations']}")


def support_vectors(X, y, Phi, w, b, atol=1e-6):
    """
    Return indices, margins, and the original feature vectors (with labels)
    for support vectors.
    """
    margins = y * (Phi @ w + b)
    sv_idx = np.where(np.isclose(margins, 1.0, atol=atol))[0]
    return sv_idx, margins[sv_idx], X[sv_idx], y[sv_idx]


# Demo Run
if __name__ == "__main__":
    # Load full dataset
    data = np.loadtxt("mystery.data", delimiter=",")
    X = data[:, :4].astype(float)
    y = np.where(data[:, 4].astype(int) >= 0, 1, -1)

    # Train on alL data
    _train(X, y)

    # Diagnostics on ALL data
    Phi = phi_quad(X.astype(float, copy=True))

    # Feasibility (should be 0 violations if perfectly separable)
    report = check_hard_margin_feasibility(Phi, y, model["w"], model["b"], tol=1e-6)
    print_feasibility_report("ALL DATA", report)

    # Margin and weights
    nw = np.linalg.norm(model["w"])
    print("------ FINAL MODEL-----")
    print(f"Bias b: {model['b']}")
    print(f"||w||: {nw}")
    print(f"Optimal margin 1/||w||: {1.0/max(nw,1e-12)}")

    # Support vectors (indices, values, and labels)
    sv_idx, sv_margins, sv_X, sv_y = support_vectors(
        X, y, Phi, model["w"], model["b"], atol=1e-6
    )
    print(f"# support vectors: {len(sv_idx)}")
    for i, idx in enumerate(sv_idx):
        print(f"Index {idx}: x={sv_X[i]}, y={sv_y[i]}, margin = {sv_margins[i]}")

    # Print w,b in for eval()
    print("\nWeight and bias for eval():")
    print("w =", np.array2string(model["w"], separator=", "))
    print("b =", f"{model['b']:.8f}")
