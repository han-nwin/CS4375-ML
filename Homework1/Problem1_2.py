import numpy as np


def subgradient_descent(X, y):
    """
    Subgradient descent for the given absolute loss function.
    Input:
        X : (m, n) data matrix (rows = data observations)
        y : (m,) labels
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

    # Epsilon and maximum iteration
    epsilon = 1e-6
    max_iters = int(1e5)

    for t in range(1, max_iters + 1):

        """Sum form"""
        # g_a = np.zeros(n)
        # g_b = 0.0
        # for i in range(m):
        #     r_i = X[i] @ a + b - y[i]
        #     s_i = np.sign(r_i)  # sign(r_i), with sign(0) = 0
        #
        #     g_a += s_i * X[i]  # add x^(i) * sign(r_i)
        #     g_b += s_i  # add sign(r_i)

        """ Matrix form - more effiecient"""
        # Preps
        r = X @ a + b - y
        s = np.sign(r)  # sign(0) = 0

        # subgradient
        g_a = X.T @ s
        g_b = s.sum()

        # step size: 1 / (1 + sqrt(t))
        gamma = 1 / (1 + np.sqrt(t))

        # norm limit checking
        norm = np.sqrt(np.linalg.norm(g_a) ** 2 + g_b**2)
        if norm <= epsilon:
            print(f"Stopped at iteration {t}, norm={norm:.2e}")
            break

        # update functions
        a = a - gamma * g_a
        b = b - gamma * g_b

        # debug print
        print(f"iter={t}, a={a}, b={b}, norm={norm:.4e}, g_a={g_a}, g_b={g_b}")

    return np.concatenate([a, [b]])


# Example run
result = subgradient_descent(
    [
        [1, 2, 4],
        [2, 1, 12],
        [0, 3, -2],
        [-1, 4, -9],
        [3, 0, 4],
        [2, 5, -1],
    ],
    [7, 15, 1, -6, 7, 6],
)

print("Final [a; b] =", result)
