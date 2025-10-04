import numpy as np
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False


def polynomial_features(X, degree=2):
    """
    Create polynomial features up to given degree.

    For degree=2 with features [x1, x2]:
    Returns [x1, x2, x1^2, x1*x2, x2^2]

    Args:
        X: M x n matrix
        degree: polynomial degree (default 2 for quadratic)

    Returns:
        X_poly: M x n_new matrix with polynomial features
    """
    M, n = X.shape

    if degree == 1:
        return X

    # Start with original features
    features = [X]

    if degree >= 2:
        # Add quadratic terms: x_i^2
        features.append(X**2)

        # Add interaction terms: x_i * x_j for i < j
        for i in range(n):
            for j in range(i + 1, n):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))

    if degree >= 3:
        # Add cubic terms
        features.append(X**3)

        # Add x_i^2 * x_j interactions
        for i in range(n):
            for j in range(n):
                if i != j:
                    features.append((X[:, i] ** 2 * X[:, j]).reshape(-1, 1))

    if degree >= 4:
        # Add quartic terms: x_i^4
        features.append(X**4)

        # Add x_i^3 * x_j interactions
        for i in range(n):
            for j in range(n):
                if i != j:
                    features.append((X[:, i] ** 3 * X[:, j]).reshape(-1, 1))

        # Add x_i^2 * x_j^2 interactions for i < j
        for i in range(n):
            for j in range(i + 1, n):
                features.append((X[:, i] ** 2 * X[:, j] ** 2).reshape(-1, 1))

        # Add x_i^2 * x_j * x_k interactions for distinct i, j, k
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len({i, j, k}) == 3:
                        features.append(
                            (X[:, i] ** 2 * X[:, j] * X[:, k]).reshape(-1, 1)
                        )

    # Concatenate all features
    X_poly = np.hstack(features)

    return X_poly


def train_primal_svm(X, y, C=1.0):
    """
    Train Primal SVM with slack variables using QP.

    Formulation:
    min (1/2)||w||^2 + C * sum(xi_i)
    s.t. y_i(w^T x_i + b) >= 1 - xi_i
         xi_i >= 0

    Args:
        X: Training data (M samples, n features) - M x n matrix
        y: Labels in {0, 1} (M samples)
        C: Regularization parameter

    Returns:
        w: Weight vector (n x 1)
        b: Bias term (scalar)
    """
    M, n = X.shape

    # Convert labels from {0, 1} to {-1, +1}
    y_train = np.where(y == 0, -1, y).astype(float)

    # Construct H matrix (n+1+M) x (n+1+M)
    H = np.zeros((n + 1 + M, n + 1 + M), dtype=np.float64)
    H[:n, :n] = np.eye(n)
    H += 1e-8 * np.eye(n + 1 + M)  # Numerical stability

    # Construct f vector (n+1+M) x 1
    f = np.zeros(n + 1 + M, dtype=np.float64)
    f[n + 1 :] = C

    # Construct inequality constraints G and h
    # G1: Margin constraints
    G1 = np.zeros((M, n + 1 + M), dtype=np.float64)
    for i in range(M):
        G1[i, :n] = -y_train[i] * X[i]
        G1[i, n] = -y_train[i]
        G1[i, n + 1 + i] = 1.0
    h1 = -np.ones(M, dtype=np.float64)

    # G2: Non-negativity of slack variables
    G2 = np.zeros((M, n + 1 + M), dtype=np.float64)
    G2[:, n + 1 :] = -np.eye(M)
    h2 = np.zeros(M, dtype=np.float64)

    # Combine constraints
    G = np.vstack([G1, G2])
    h = np.hstack([h1, h2])

    # Convert to cvxopt format and solve
    H_cvx = matrix(H)
    f_cvx = matrix(f)
    G_cvx = matrix(G)
    h_cvx = matrix(h)

    sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx)

    # Extract solution
    solution = np.array(sol["x"]).flatten()
    w = solution[:n]
    b = solution[n]

    return w, b


def predict_svm(X, w, b):
    """Predict class labels using trained SVM."""
    decision = X @ w + b
    predictions = np.where(decision >= 0, 1, 0)
    return predictions


def calculate_accuracy(X, y, w, b):
    """Calculate accuracy of SVM predictions."""
    predictions = predict_svm(X, w, b)
    accuracy = np.mean(predictions == y)
    return accuracy


def load_and_split_data(filename, use_poly=False, poly_degree=2):
    """Load data and split into train, validation, and test sets."""
    data = np.loadtxt(filename, delimiter=",")

    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # Standardize features (using only training data stats)
    X_train_raw = X[:1000]
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    # Apply polynomial feature map if requested
    if use_poly:
        print(f"Creating polynomial features (degree {poly_degree})...")
        X = polynomial_features(X, degree=poly_degree)
        print(f"New feature dimension: {X.shape[1]}")

    # Split data
    X_train = X[:1000]
    y_train = y[:1000]

    X_val = X[1000:2000]
    y_val = y[1000:2000]

    X_test = X[2000:]
    y_test = y[2000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Load and split data with polynomial features
    print("Loading data...")

    # Try with degree 3 ( 1 and 2 FAILED)
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(
        "magic.data", use_poly=True, poly_degree=3
    )

    M_train, n = X_train.shape
    print(f"Training set: {M_train} samples, {n} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()

    # C values to try
    C_values = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

    train_accuracies = []
    val_accuracies = []
    models = []

    print("Training SVMs for different C values...")
    print("=" * 70)

    for C in C_values:
        print(f"\nTraining with C = {C:.0e}")

        # Train SVM
        w, b = train_primal_svm(X_train, y_train, C=C)
        models.append((w, b))

        # Calculate accuracies
        train_acc = calculate_accuracy(X_train, y_train, w, b)
        val_acc = calculate_accuracy(X_val, y_val, w, b)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")

    print("\n" + "=" * 70)
    print("\nSUMMARY OF RESULTS")
    print("=" * 70)
    print(f"{'C':<12} {'Train Acc':<15} {'Val Acc':<15}")
    print("-" * 70)
    for i, C in enumerate(C_values):
        print(f"{C:<12.0e} {train_accuracies[i]:<15.4f} {val_accuracies[i]:<15.4f}")

    # Find best C based on validation accuracy
    best_idx = np.argmax(val_accuracies)
    best_C = C_values[best_idx]
    best_val_acc = val_accuracies[best_idx]

    print("\n" + "=" * 70)
    print(f"\nBest C value: {best_C:.0e} (Validation accuracy: {best_val_acc:.4f})")

    # Retrain with combined training and validation data
    print(f"\nRetraining with combined train+val data using C = {best_C:.0e}...")
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.hstack([y_train, y_val])

    w_final, b_final = train_primal_svm(X_train_combined, y_train_combined, C=best_C)

    # Test accuracy
    test_acc = calculate_accuracy(X_test, y_test, w_final, b_final)

    print(f"\nFINAL TEST ACCURACY: {test_acc:.4f}")
    print("=" * 70)
