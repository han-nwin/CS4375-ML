import numpy as np


def l1_subgradient_descent(X, y):
    """
    Subgradient descent for the given absolute loss function.
    Input:
        X : (m, n) data matrix (rows = observations)
        y : (m,) labels
    Output:
        theta : (n+1,) vector [a; b]
    """
    m, n = X.shape
    a = np.zeros(n)
    b = 0.0

    for t in range(1, 1001):  # fixed number of iterations
        r = X @ a + b - y  # residuals (m,)
        s = np.sign(r)  # subgradient contributions

        g_a = X.T @ s  # shape (n,)
        g_b = s.sum()  # scalar

        # step size: 1 / (1 + sqrt(t))
        alpha = 1.0 / (1.0 + np.sqrt(t))

        # update
        a = a - alpha * g_a
        b = b - alpha * g_b

    return np.concatenate([a, [b]])
