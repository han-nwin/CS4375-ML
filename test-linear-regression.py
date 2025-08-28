import numpy as np

# ----------------------------
# 1) Toy data (1 feature)
# ----------------------------
# X: square footage, y: price ($1000s)
X = np.array([500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500], dtype=float).reshape(
    -1, 1
)  # Reshape the 1D array into a 2D column vector with one feature per row
y = np.array([150, 195, 240, 270, 300, 360, 420, 470, 520], dtype=float)

# Train/validation split
rng = np.random.default_rng(
    42
)  # Create a random number generator with a fixed seed for reproducibility
idx = rng.permutation(
    len(X)
)  # Generate a random permutation of indices from 0 to len(X)-1
split = int(
    0.8 * len(X)
)  # Calculate the index to split data into 80% training and 20% validation
train_idx, val_idx = (
    idx[:split],
    idx[split:],
)  # Split the indices into training and validation sets
X_train, y_train = (
    X[train_idx],
    y[train_idx],
)  # Select training samples and corresponding targets
X_val, y_val = (
    X[val_idx],
    y[val_idx],
)  # Select validation samples and corresponding targets


# ----------------------------
# 2) Helper functions
# ----------------------------
def standardize(X):
    """Return standardized X, along with mean and std for inverse/transform on new data."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    return (X - mu) / sigma, mu, sigma


def add_bias(X):
    """Add a column of ones for the intercept term."""
    # np.c_ is a convenient way to concatenate along the second axis (columns).
    # Here, it concatenates a column of ones with the input matrix X,
    # effectively adding a bias (intercept) term to the features.
    return np.c_[np.ones((X.shape[0], 1)), X]


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ----------------------------
# 3) Prepare data
# ----------------------------
X_train_std, mu, sigma = standardize(X_train)
X_val_std = (X_val - mu) / sigma
Xb_train = add_bias(X_train_std)
Xb_val = add_bias(X_val_std)


# ----------------------------
# 4) Gradient Descent training
# ----------------------------
def fit_gd(Xb, y, lr=0.1, n_iters=2000):
    n_features = Xb.shape[1]
    w = np.zeros(n_features)  # [bias, w1, w2, ...]
    history = []
    for t in range(n_iters):
        y_hat = Xb @ w
        grad = -(2 / len(y)) * (Xb.T @ (y - y_hat))
        w -= lr * grad
        if (t + 1) % 100 == 0 or t == 0:
            history.append(mse(y, y_hat))
    return w, history


w, history = fit_gd(Xb_train, y_train, lr=0.1, n_iters=3000)

# ----------------------------
# 5) Evaluate
# ----------------------------
y_train_pred = Xb_train @ w
y_val_pred = Xb_val @ w

print("Weights [bias, w_scaled]:", w)
print("Train MSE:", mse(y_train, y_train_pred))
print("Val   MSE:", mse(y_val, y_val_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Val   R^2:", r2_score(y_val, y_val_pred))


# ----------------------------
# 6) Use the model on new inputs
# ----------------------------
def predict(raw_X):
    raw_X = np.array(raw_X, dtype=float).reshape(-1, 1)
    X_std = (raw_X - mu) / sigma
    Xb = add_bias(X_std)
    return Xb @ w


X_new = [[1800], [2200], [3000]]
print("Predictions for", X_new, "", predict(X_new), "(in $1000s)")

# ----------------------------
# 7) (Optional) Closed-form solution (Normal Equation) for comparison
# ----------------------------
# For reference only: w_cf = (X^T X)^(-1) X^T y, using the standardized features + bias.
w_cf = np.linalg.pinv(Xb_train.T @ Xb_train) @ (Xb_train.T @ y_train)
print("Closed-form weights:", w_cf)
