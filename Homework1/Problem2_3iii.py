import numpy as np


def stochastic_subgradient_descent(X, y, iters=10**5, step_sizes=[1]):
    M, n = X.shape

    # feature map
    phi = np.zeros((M, n + 1))
    for i in range(M):
        x1, x2 = X[i]
        phi[i] = [x1, x2, x1**2 + x2**2]

    results = {}

    for step_size in step_sizes:
        # vector of zeros with length n+1 (ie x1, x2 -> x1, x2, x1^2, x2^2)
        w = np.zeros(n + 1)
        b = 0.0

        iter_stops = set(range(100, 1000, 10))
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
            w = w + g_w * step_size
            b = b + g_b * step_size

            # add to logs every iter stop
            if t in iter_stops:
                logs.append((t, w.copy(), b))

        results[step_size] = logs

    return results


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
    step_sizes = [10**-10, 1, 10**10]
    results = stochastic_subgradient_descent(X, y, step_sizes=step_sizes)

    for step_size, logs in results.items():
        print(f"Step size: {step_size}")
        for t, w, b in logs:
            print(f"  iter {t}: w={w}, b={b}")
