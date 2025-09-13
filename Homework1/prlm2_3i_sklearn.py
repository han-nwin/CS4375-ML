import numpy as np


def standard_subgradient_descent(X, y, iters=10**5):
    M, n = X.shape

    # feature map
    phi = np.zeros((M, n + 1))
    for i in range(M):
        x1, x2 = X[i]
        phi[i] = [x1, x2, x1**2 + x2**2]

    # vector of zeros with length n+1 (ie x1, x2 -> x1, x2, x1^2, x2^2)
    w = np.zeros(n + 1)
    b = 0.0

    inter_stops = {1, 10, 100, 10**3, 10**4, 10**5}
    logs = []

    for t in range(1, iters + 1):
        # compute g_w g_b
        g_w = np.zeros(n + 1)  # n=3 since φ maps to R^3
        g_b = 0.0

        # sum
        for i in range(M):
            f_i = w @ phi[i] + b
            # check condition
            if y[i] * f_i <= 0:  # misclassified
                g_w += -y[i] * phi[i]
                g_b += -y[i]

        g_w /= -M  # * -1/M
        g_b /= -M  # * -1/M

        # Updates with step size = 1
        w = w + g_w
        b = b + g_b

        # add to logs every iter stop
        if t in inter_stops:
            logs.append((t, w.copy(), b))

    return logs


# load and read the file to produce X and y
def load_data(path):
    # expects lines: x1,x2,y   (y in {-1, +1}), comma-separated
    X, y = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            x1, x2, yy = line.strip().split(",")
            X.append([float(x1), float(x2)])
            y.append(int(float(yy)))
    X = np.array(X, dtype=float)  # (M,2)
    y = np.array(y, dtype=int)  # (M,)
    return X, y


# === sklearn baselines (no changes to your implementation above) ===
def _phi_map(X):
    # same feature map you used: [x1, x2, x1^2 + x2^2]
    M, n = X.shape
    assert n == 2, "This helper assumes X has 2 columns (x1, x2)."
    phi = np.zeros((M, n + 1))
    for i in range(M):
        x1, x2 = X[i]
        phi[i] = [x1, x2, x1**2 + x2**2]
    return phi


def sklearn_baselines(X, y):
    try:
        from sklearn.linear_model import Perceptron, LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.metrics import accuracy_score
    except ImportError:
        print(
            "scikit-learn not installed. `pip install scikit-learn` to run baselines."
        )
        return

    Phi = _phi_map(X)  # (M,3) to match your w in R^3

    # 1) Perceptron (closest to your update rule conceptually)
    perc = Perceptron(
        penalty=None,  # classic perceptron
        alpha=0.0001,  # unused when penalty=None
        fit_intercept=True,
        max_iter=100000,
        tol=1e-5,
        shuffle=True,
        eta0=1.0,  # initial learning rate (not strictly used by Perceptron)
        early_stopping=False,
    )
    perc.fit(Phi, y)
    w_perc = perc.coef_.ravel()
    b_perc = float(perc.intercept_)
    yhat_perc = perc.predict(Phi)
    acc_perc = accuracy_score(y, yhat_perc)

    # 2) Linear SVM (hinge loss / subgradient-based under the hood)
    # Note: C large ≈ harder margin; tune if needed.
    lsvm = LinearSVC(C=1.0, fit_intercept=True, max_iter=100000, tol=1e-5, dual=True)
    lsvm.fit(Phi, y)
    w_lsvm = lsvm.coef_.ravel()
    b_lsvm = float(lsvm.intercept_)
    yhat_lsvm = lsvm.predict(Phi)
    acc_lsvm = accuracy_score(y, yhat_lsvm)

    # 3) Logistic Regression (smooth surrogate; still linear in φ)
    logreg = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True, max_iter=100000
    )
    # LogisticRegression expects class labels; {-1,+1} is fine.
    logreg.fit(Phi, y)
    w_log = logreg.coef_.ravel()
    b_log = float(logreg.intercept_)
    yhat_log = logreg.predict(Phi)
    acc_log = accuracy_score(y, yhat_log)

    # Print in a format similar to your logs
    print("\n=== scikit-learn baselines on φ(x)=[x1, x2, x1^2+x2^2] ===")
    print(f"[Perceptron]     w={w_perc}, b={b_perc:.6f} | train acc={acc_perc:.4f}")
    print(f"[LinearSVC]      w={w_lsvm}, b={b_lsvm:.6f} | train acc={acc_lsvm:.4f}")
    print(f"[LogisticReg]    w={w_log},  b={b_log:.6f}  | train acc={acc_log:.4f}")


if __name__ == "__main__":
    X, y = load_data("perceptron.data")
    logs = standard_subgradient_descent(X, y)

    for t, w, b in logs:
        print(f"iter {t:>6}: w={w}, b={b}")

    # run sklearn on the same mapped features and print comparable results
    sklearn_baselines(X, y)
