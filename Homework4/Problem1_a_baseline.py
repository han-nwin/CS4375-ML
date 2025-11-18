import numpy as np
from cvxopt import matrix, solvers


class SVM_Gaussian:
    """
    SVM with Gaussian kernel
    Implements the slack formulation (soft-margin SVM)
    """

    def __init__(self, C=1.0, sigma=1.0):
        """
        Parameters:
        C: Slack penalty parameter
        sigma: Gaussian kernel parameter (bandwidth)
        """
        self.C = C
        self.sigma = sigma
        self.lambda_ = None
        self.b = 0
        self.X_train = None
        self.y_train = None

    def gaussian_kernel(self, X1, X2):
        """
        Compute Gaussian kernel matrix between X1 and X2
        K(x, x') = exp(-||x - x'||^2 / (2 * sigma^2))
        """
        # Compute squared Euclidean distances
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        sq_dists = X1_sq + X2_sq - 2 * X1 @ X2.T

        # Compute kernel
        K = np.exp(-sq_dists / (2 * self.sigma**2))
        return K

    def fit(self, X, y):
        """
        Train the SVM using quadratic programming with cvxopt
        """
        M = X.shape[0]

        # Store training data
        self.X_train = X

        # Convert labels to {-1, 1}
        self.y_train = np.where(y <= 0, -1, y).astype(float)

        # Compute kernel matrix
        K = self.gaussian_kernel(X, X)

        # Construct H matrix for QP (minimize (1/2)x^T H x - f^T x)
        H = np.outer(self.y_train, self.y_train) * K
        H += 1e-8 * np.eye(M)  # Numerical stability

        # Construct f vector
        f = -np.ones(M, dtype=np.float64)

        # Inequality constraints: 0 <= lambda_i <= C
        # -lambda_i <= 0 and lambda_i <= C
        G = np.vstack([-np.eye(M), np.eye(M)])
        h = np.hstack([np.zeros(M), self.C * np.ones(M)])

        # Equality constraint: sum(lambda_i * y_i) = 0
        A = self.y_train.reshape(1, -1)
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
        self.lambda_ = np.array(sol["x"]).flatten()

        # Compute bias b using support vectors with 0 < lambda < C
        threshold = 1e-5
        margin_sv = np.where(
            (self.lambda_ > threshold) & (self.lambda_ < self.C - threshold)
        )[0]

        if len(margin_sv) > 0:
            # Use average over margin support vectors
            b_values = []
            for i in margin_sv:
                b_i = self.y_train[i] - np.sum(self.lambda_ * self.y_train * K[i, :])
                b_values.append(b_i)
            self.b = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            support_vectors = np.where(self.lambda_ > threshold)[0]
            b_values = []
            for i in support_vectors:
                b_i = self.y_train[i] - np.sum(self.lambda_ * self.y_train * K[i, :])
                b_values.append(b_i)
            self.b = np.mean(b_values) if b_values else 0.0

    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        # Compute kernel between test and training data
        K_test = self.gaussian_kernel(X, self.X_train)

        # Decision function
        decision = K_test @ (self.lambda_ * self.y_train) + self.b

        # Return predictions in original label format
        return np.where(decision >= 0, 1, -1)


# Load the Gisette dataset
def load_data():
    """Load training, validation, and test data"""
    train_data = np.loadtxt("gisette_train.data", delimiter=",")
    valid_data = np.loadtxt("gisette_valid.data", delimiter=",")
    test_data = np.loadtxt("gisette_test.data", delimiter=",")

    # First column is the label, rest are features
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    X_valid = valid_data[:, 1:]
    y_valid = valid_data[:, 0]

    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main():
    """Main function"""
    print("=" * 80)
    print("BASELINE SVM - NO FEATURE SELECTION (All Original Features)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_valid.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Hyperparameters to tune
    C_values = [0.1, 1, 10]
    sigma_values = [0.1, 1, 10]

    print("\n" + "-" * 80)
    print("Training SVM with ALL original features (no PCA)")
    print("-" * 80)

    best_accuracy = 0
    best_C = None
    best_sigma = None

    print("\nTuning hyperparameters on validation set...")
    for C in C_values:
        for sigma in sigma_values:
            print(f"  Trying C={C}, sigma={sigma}...", end=" ", flush=True)
            # Train SVM on original features
            svm = SVM_Gaussian(C=C, sigma=sigma)
            svm.fit(X_train, y_train)

            # Evaluate on validation set
            y_valid_pred = svm.predict(X_valid)
            accuracy = np.mean(y_valid_pred == y_valid)
            print(f"Val Acc: {accuracy*100:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C
                best_sigma = sigma

    print(f"\nBest hyperparameters: C={best_C}, sigma={best_sigma}")
    print(f"Validation accuracy: {best_accuracy*100:.2f}%")

    # Train final baseline model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    svm_baseline = SVM_Gaussian(C=best_C, sigma=best_sigma)
    svm_baseline.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = svm_baseline.predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    test_error = 1 - test_accuracy

    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(f"Features used: ALL {X_train.shape[1]} original features (no PCA)")
    print(f"Best C: {best_C}")
    print(f"Best sigma: {best_sigma}")
    print(f"Validation accuracy: {best_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Test error: {test_error*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
