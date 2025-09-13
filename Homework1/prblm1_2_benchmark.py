import numpy as np
from sklearn.linear_model import QuantileRegressor, LinearRegression


def subgradient_descent(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m, n = X.shape

    a = np.ones(n)
    b = 1.0
    epsilon = 1e-6
    max_iters = int(1e5)

    for t in range(1, max_iters + 1):
        r = X @ a + b - y
        s = np.sign(r)
        g_a = X.T @ s
        g_b = s.sum()
        gamma = 1 / (1 + np.sqrt(t))

        norm = np.sqrt(np.linalg.norm(g_a) ** 2 + g_b**2)
        if norm <= epsilon:
            break

        a = a - gamma * g_a
        b = b - gamma * g_b

    return np.concatenate([a, [b]])


# ==== Your dataset ====
X = np.array(
    [
        [1, 2, 4],
        [2, 1, 12],
        [0, 3, -2],
        [-1, 4, -9],
        [3, 0, 4],
        [2, 5, -1],
    ],
    dtype=float,
)
y = np.array([7, 15, 1, -6, 7, 6], dtype=float)

# ==== Run your solver ====
params = subgradient_descent(X, y)
print("Your subgradient [a; b] =", params)

# ==== Run scikit-learn LAD (L1 loss) ====
lad = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
lad.fit(X, y)
print("QuantileRegressor coef_ =", lad.coef_, "intercept_ =", lad.intercept_)
print(f"predict {lad.predict([[2,3,4]])}")

# ==== Run scikit-learn OLS (L2 loss) ====
ols = LinearRegression()
ols.fit(X, y)
print("LinearRegression coef_ =", ols.coef_, "intercept_ =", ols.intercept_)
