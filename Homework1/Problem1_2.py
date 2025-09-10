import numpy as np


def l1_subgradient_descent(X, y, epsilon=1e-6, max_iters=1e8):
    """
    Subgradient descent for the given absolute loss function.
    Input:
        X : (m, n) data matrix (rows = observations)
        y : (m,) labels
        epsilon : float, stopping tolerance on subgradient norm
        max_iters : int, safeguard maximum iterations
    Output:
        (n+1,) vector [a; b]
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m, n = X.shape
    a = np.zeros(n)
    b = 0.0

    for t in range(1, max_iters + 1):
        r = X @ a + b - y  # residuals (m,)
        s = np.sign(r)  # subgradient contributions

        g_a = X.T @ s  # shape (n,)
        g_b = s.sum()  # scalar

        # compute subgradient norm
        norm = np.sqrt(np.linalg.norm(g_a) ** 2 + g_b**2)
        if norm <= epsilon:
            print(f"Stopped at iteration {t}, grad_norm={norm:.2e}")
            break

        # step size: 1 / (1 + sqrt(t))
        alpha = 1.0 / (1.0 + np.sqrt(t))

        # update
        a = a - alpha * g_a
        b = b - alpha * g_b

        # debug print
        print(f"iter={t}, a={a}, b={b}, norm={norm:.4e}")

    return np.concatenate([a, [b]])


# Example run
result = l1_subgradient_descent(
    [[1, 2, 3], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
    [6, 3, 6, 9],
    epsilon=1e-6,
)
print("Final [a; b] =", result)
