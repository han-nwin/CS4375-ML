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


# === tensorflow/keras baselines (linear model + hinge losses) ===
def tf_baselines(X, y):
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. `pip install tensorflow` to run TF baselines.")
        return

    # same feature map
    def _phi_map(X):
        M, n = X.shape
        assert n == 2
        phi = np.zeros((M, n + 1), dtype=np.float32)
        for i in range(M):
            x1, x2 = X[i]
            phi[i] = [x1, x2, x1**2 + x2**2]
        return phi

    Phi = _phi_map(X).astype(np.float32)  # (M, 3)
    y_pm1 = y.astype(np.float32).reshape(-1, 1)  # {−1,+1} for hinge

    tf.random.set_seed(42)

    def train_and_report(
        loss_name="hinge", lr=0.05, epochs=800, l1=0.0, l2=0.0, batch_size=None
    ):
        reg = None
        if l1 or l2:
            reg = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(Phi.shape[1],)),
                tf.keras.layers.Dense(1, use_bias=True, kernel_regularizer=reg),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=loss_name
        )
        model.fit(
            Phi, y_pm1, epochs=epochs, batch_size=batch_size or Phi.shape[0], verbose=0
        )

        W, b_vec = model.layers[0].get_weights()  # W:(3,1), b:(1,)
        w = W.ravel().astype(float)
        b = float(b_vec[0])

        # evaluate with your sign rule
        yhat = np.where(Phi @ w + b >= 0, 1, -1)
        acc = (yhat == y).mean()

        print(f"[TF {loss_name:<13}] w={w}, b={b:.6f} | train acc={acc:.4f}")
        return w, b, acc

    # plain hinge (closer to linear SVM objective)
    train_and_report(loss_name="hinge", lr=0.1, epochs=1000, l2=1e-4)

    # squared hinge (another common SVM surrogate)
    train_and_report(loss_name="squared_hinge", lr=0.1, epochs=1000, l2=1e-4)

    # (optional) logistic with BCE for reference; convert labels to {0,1}
    y01 = ((y + 1) // 2).astype(np.float32).reshape(-1, 1)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(Phi.shape[1],)),
            tf.keras.layers.Dense(
                1, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            ),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    model.fit(Phi, y01, epochs=1000, batch_size=Phi.shape[0], verbose=0)
    W, b_vec = model.layers[0].get_weights()
    w_log = W.ravel().astype(float)
    b_log = float(b_vec[0])
    yhat = np.where((Phi @ w_log + b_log) >= 0, 1, -1)
    acc = (yhat == y).mean()
    print(f"[TF logistic     ] w={w_log}, b={b_log:.6f} | train acc={acc:.4f}")


if __name__ == "__main__":
    X, y = load_data("perceptron.data")
    logs = standard_subgradient_descent(X, y)
    for t, w, b in logs:
        print(f"iter {t:>6}: w={w}, b={b}")
    # tensorflow baselines
    tf_baselines(X, y)
