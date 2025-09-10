import numpy as np


def subgradient_descent(X, y, epsilon=1e-6, max_iters=int(1e5)):
    """
    Subgradient descent for the given absolute loss function.
    Input:
        X : (m, n) data matrix (rows = data observations)
        y : (m,) labels
        epsilon : float, stopping tolerance on subgradient norm
        max_iters : int, safeguard maximum iterations
    Output:
        (n+1,) vector [a; b]
    """
    # turn input into numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # number of data points (m) and the number of features (n)
    # from the shape of the input data matrix X
    m, n = X.shape

    a = np.ones(n)  # vector of ones with length n
    b = 1.0

    for t in range(1, max_iters + 1):
        # Preps
        r = X @ a + b - y
        s = np.sign(r)

        # subgradient
        g_a = X.T @ s
        g_b = s.sum()

        # subgradient norm limit checking
        norm = np.sqrt(np.linalg.norm(g_a) ** 2 + g_b**2)
        if norm <= epsilon:
            print(f"Stopped at iteration {t}, norm={norm:.2e}")
            break

        # step size: 1 / (1 + sqrt(t))
        gamma = 1 / (1 + np.sqrt(t))

        # update functions
        a = a - gamma * g_a
        b = b - gamma * g_b

        # debug print
        print(f"iter={t}, a={a}, b={b}, norm={norm:.4e}, g_a={g_a}, g_b={g_b}")

    return np.concatenate([a, [b]])


# Example run
result = subgradient_descent(
    [
        [1, 2],
        [2, 1],
        [0, 3],
        [-1, 4],
        [3, 0],
        [2, 5],
    ],
    [0, 5, -5, -10, 10, -7],
)

print("Final [a; b] =", result)
