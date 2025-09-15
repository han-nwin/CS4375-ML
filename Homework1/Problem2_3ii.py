import numpy as np


def stochastic_subgradient_descent(X, y, iters=10**5):
    M, n = X.shape

    # feature map
    phi = np.zeros((M, n + 1))
    for i in range(M):
        x1, x2 = X[i]
        phi[i] = [x1, x2, x1**2 + x2**2]

    # vector of zeros with length n+1 (ie x1, x2 -> x1, x2, x1^2, x2^2)
    w = np.zeros(n + 1)
    b = 0.0

    iter_stops = {1, 10, 100, 10**3, 10**4, 10**5}
    logs = []

    for t in range(1, iters + 1):
        # determine i
        i = (t - 1) % M  # Use mod to cycle through samples

        f_i = w @ phi[i] + b

        # check condition
        if y[i] * f_i <= 0:  # misclassified
            w = w + y[i] * 1 * phi[i]  # since k = 1 -> 1/k = 1
            b = b + 1 * y[i]

        # add to logs every iter stop
        if t in iter_stops:
            logs.append((t, w.copy(), b))

    return logs


# load and read the file to produce X and y
def load_data(path):
    # expects lines: x1,x2,y   (y = -1 or 1), comma-separated
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


# run
if __name__ == "__main__":
    X, y = load_data("perceptron.data")
    logs = stochastic_subgradient_descent(X, y)

    for t, w, b in logs:
        print(f"iter {t}: w={w}, b={b}")
