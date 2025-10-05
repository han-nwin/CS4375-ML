import numpy as np
from cvxopt import matrix, solvers


def train_modified_dual_svm(X, y, C=1.0):
    """
    Train Modified Dual SVM with quadratic slack penalty using QP.
    """
    M = X.shape[0]

    # Convert labels from {0, 1} to {-1, +1}
    y_train = np.where(y == 0, -1, y).astype(float)

    # Construct H matrix for QP (minimize (1/2)x^T H x - f^T x)
    # H = y*y^T * (X*X^T) + (1/2C)*I
    # Note: The quadratic slack term adds (1/2C)*I to the kernel matrix
    K = X @ X.T  # Linear kernel
    H = np.outer(y_train, y_train) * K + (1 / (2 * C)) * np.eye(M)
    H += 1e-8 * np.eye(M)  # Numerical stability

    # Construct f vector
    f = -np.ones(M, dtype=np.float64)

    # Inequality constraints: lambda_i >= 0
    G = -np.eye(M)
    h = np.zeros(M)

    # Equality constraint: sum(lamda * y_i) = 0
    A = y_train.reshape(1, -1)
    b_eq = np.zeros(1)

    # Convert to cvxopt format and solve
    H_cvx = matrix(H)
    f_cvx = matrix(f)
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    A_cvx = matrix(A)
    b_cvx = matrix(b_eq)

    # Solver
    solvers.options["show_progress"] = False
    sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx, A_cvx, b_cvx)

    # Extract solution
    lambda_ = np.array(sol["x"]).flatten()

    # Find support vectors (lambda > threshold)
    threshold = 1e-5
    support_vectors = np.where(lambda_ > threshold)[0]

    # Compute bias b
    # b* = y_i(1 - lambda_i*/(2C)) - sum_j lambda_j* y_j x_j^T x_i
    b_values = []
    for i in range(M):
        # Use all training points (or just those with lambda_i > threshold for stability)
        if lambda_[i] > 1e-5:  # Optional: only use points with non-zero lambda
            b_i = y_train[i] * (1 - lambda_[i] / (2 * C)) - np.sum(
                lambda_ * y_train * K[i, :]
            )
            b_values.append(b_i)

    b = np.mean(b_values) if b_values else 0.0

    return lambda_, b, support_vectors


def predict_modified_dual_svm(X_train, y_train, lambda_, b, X_test):
    """Predict class labels using trained modified dual SVM."""
    # Convert labels from {0, 1} to {-1, +1}
    y_train_signed = np.where(y_train == 0, -1, y_train).astype(float)

    # Compute kernel between test and training data (linear kernel)
    K_test = X_test @ X_train.T

    # Decision function
    decision = K_test @ (lambda_ * y_train_signed) + b
    predictions = np.where(decision >= 0, 1, 0)

    return predictions


def calculate_accuracy_modified(X_train, y_train, lambda_, b, X_test, y_test):
    """Calculate accuracy of modified dual SVM predictions."""
    predictions = predict_modified_dual_svm(X_train, y_train, lambda_, b, X_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy


def load_and_split_data(filename):
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

    # Split data
    X_train = X[:1000]
    y_train = y[:1000]

    X_val = X[1000:2000]
    y_val = y[1000:2000]

    X_test = X[2000:]
    y_test = y[2000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Load and split data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data("magic.data")

    M_train, n = X_train.shape
    print(f"Training set: {M_train} samples, {n} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()

    # Hyperparameters to try
    C_values = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

    results = []

    print("Training Modified Dual SVMs (Quadratic Slack)...")
    print("=" * 80)

    for C in C_values:
        print(f"\nTraining with C = {C:.0e}")

        # Train SVM
        lambda_, b, sv_indices = train_modified_dual_svm(X_train, y_train, C=C)

        # Calculate accuracies
        train_acc = calculate_accuracy_modified(
            X_train, y_train, lambda_, b, X_train, y_train
        )
        val_acc = calculate_accuracy_modified(
            X_train, y_train, lambda_, b, X_val, y_val
        )

        results.append(
            {
                "C": C,
                "lambda": lambda_,
                "b": b,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "num_sv": len(sv_indices),
            }
        )

        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Number of support vectors: {len(sv_indices)}")

    print("\n" + "=" * 80)
    print("\nSUMMARY OF RESULTS")
    print("=" * 80)
    print(f"{'C':<12} {'Train Acc':<15} {'Val Acc':<15} {'# SV':<10}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['C']:<12.0e} {r['train_acc']:<15.4f} "
            f"{r['val_acc']:<15.4f} {r['num_sv']:<10}"
        )

    # Find best hyperparameters based on validation accuracy
    best_result = max(results, key=lambda x: x["val_acc"])
    best_C = best_result["C"]
    best_val_acc = best_result["val_acc"]

    print("\n" + "=" * 80)
    print(f"\nBest hyperparameter: C = {best_C:.0e}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Retrain with combined training and validation data
    print(f"\nRetraining with combined train+val data using C = {best_C:.0e}")
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.hstack([y_train, y_val])

    lambda_final, b_final, sv_final = train_modified_dual_svm(
        X_train_combined, y_train_combined, C=best_C
    )

    # Test accuracy
    test_acc = calculate_accuracy_modified(
        X_train_combined,
        y_train_combined,
        lambda_final,
        b_final,
        X_test,
        y_test,
    )

    print(f"\nFINAL TEST ACCURACY: {test_acc:.4f}")
    print(f"Number of support vectors: {len(sv_final)}")
    print("=" * 80)
