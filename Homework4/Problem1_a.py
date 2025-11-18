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


def perform_pca(X):
    """
    Perform PCA on the data matrix X
    X has shape (M, n) where M = number of data points, n = number of features
    Returns: eigenvalues, eigenvectors, mean
    """
    print("Performing PCA...")

    # Step 1: Center the data (subtract mean)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    M, n = X.shape  # M = number of data points, n = number of features
    print(f"Data shape: ({M}, {n})")
    print(f"M (number of data points): {M}")
    print(f"n (number of features): {n}")

    # Step 2: Compute covariance matrix
    # Cov = (1/(M-1)) * X^T * X (unbiased estimator)
    # Covariance matrix will be (n x n)
    cov_matrix = (1 / (M - 1)) * np.dot(X_centered.T, X_centered)

    print(f"Covariance matrix shape: {cov_matrix.shape}")

    # Step 3: Compute eigenvalues and eigenvectors
    print("Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, mean


def main():
    """Main function"""
    print("=" * 80)
    print("PROBLEM 1_a: SVMs and PCA")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Part 1: Perform PCA
    print("\n" + "-" * 80)
    eigenvalues, eigenvectors, mean = perform_pca(X_train)

    # Display top 6 eigenvalues
    print("\n" + "-" * 80)
    print("Part 1: Top 6 EIGENVALUES OF THE COVARIANCE MATRIX:")
    print("-" * 80)
    for i in range(6):
        print(f"Eigenvalue {i+1}: {eigenvalues[i]:.2f}")

    # Calculate variance explained by top 6
    total_variance = np.sum(eigenvalues)
    variance_top6 = np.sum(eigenvalues[:6])
    print(f"\nTotal variance: {total_variance:.2f}")
    print(f"Variance explained by top 6: {variance_top6:.6f}")
    print(f"Percentage explained by top 6: {(variance_top6/total_variance)*100:.2f}%")

    # Part 2: Build set K = {k99, k95, k90, k80, k75}
    print("\n" + "-" * 80)
    print("Part 2: Building set K - VARIANCE THRESHOLDS")
    print("-" * 80)

    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(eigenvalues)
    variance_ratios = cumulative_variance / total_variance

    # Find k values for different variance thresholds
    thresholds = [0.99, 0.95, 0.90, 0.80, 0.75]
    K = {}

    for z in thresholds:
        # Find smallest k where cumulative variance >= z
        k = int(
            np.argmax(variance_ratios >= z) + 1
        )  # +1 because index starts at 0, convert to int
        K[f"k{int(z*100)}"] = k
        print(f"\nk{int(z*100)} (explains {z*100}% variance):")
        print(f"  k = {k} components")

    # Convert K values to regular Python ints for cleaner output
    K_clean = {key: int(value) for key, value in K.items()}
    print(f"\nSet K = {K_clean}")

    # Part 3: Project data and train SVM for each k in K
    print("\n" + "-" * 80)
    print("PART 3: PROJECTING DATA AND TRAINING SVM")
    print("-" * 80)

    # Store results for comparison
    results = {}

    # Hyperparameters to tune
    C_values = [0.1, 1, 10]
    sigma_values = [0.1, 1, 10]

    for k_name, k in K_clean.items():
        print(f"\n{'-'*80}")
        print(f"Training SVM with {k_name} = {k} components")
        print(f"{'-'*80}")

        # Project data to k-dimensional subspace
        # Use top k eigenvectors
        eigenvectors_k = eigenvectors[:, :k]

        # Project training, validation, and test data
        X_train_proj = np.dot(X_train - mean, eigenvectors_k)
        X_valid_proj = np.dot(X_valid - mean, eigenvectors_k)
        X_test_proj = np.dot(X_test - mean, eigenvectors_k)

        print(f"Projected training data shape: {X_train_proj.shape}")

        # Grid search for best hyperparameters using validation set
        best_accuracy = 0
        best_C = None
        best_sigma = None

        print("Tuning hyperparameters on validation set...")
        for C in C_values:
            for sigma in sigma_values:
                # Train SVM
                svm = SVM_Gaussian(C=C, sigma=sigma)
                svm.fit(X_train_proj, y_train)

                # Evaluate on validation set
                y_valid_pred = svm.predict(X_valid_proj)
                accuracy = np.mean(y_valid_pred == y_valid)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_C = C
                    best_sigma = sigma

        print(f"Best hyperparameters: C={best_C}, sigma={best_sigma}")
        print(f"Validation accuracy: {best_accuracy*100:.2f}%")

        # Train final model with best hyperparameters
        svm_final = SVM_Gaussian(C=best_C, sigma=best_sigma)
        svm_final.fit(X_train_proj, y_train)

        # Evaluate on test set
        y_test_pred = svm_final.predict(X_test_proj)
        test_accuracy = np.mean(y_test_pred == y_test)
        test_error = 1 - test_accuracy

        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        print(f"Test error: {test_error*100:.2f}%")

        # Store results
        results[k_name] = {
            "k": k,
            "C": best_C,
            "sigma": best_sigma,
            "valid_acc": best_accuracy,
            "test_acc": test_accuracy,
            "test_error": test_error,
        }

    # Print summary
    print("\n" + "-" * 80)
    print("RESULTS SUMMARY")
    print("-" * 80)
    print(
        f"{'Model':<10} {'k':<8} {'C':<8} {'sigma':<10} {'Valid Acc':<12} {'Test Acc':<12} {'Test Error':<12}"
    )
    print("-" * 80)
    for name, res in results.items():
        print(
            f"{name:<10} {res['k']:<8} {res['C']:<8.1f} {res['sigma']:<10.2f} {res['valid_acc']*100:<12.2f} {res['test_acc']*100:<12.2f} {res['test_error']*100:<12.2f}"
        )

    # Find best model
    best_model_name = min(results.items(), key=lambda x: x[1]["test_error"])[0]
    best_model = results[best_model_name]
    print(f"\n{'-'*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"  k = {best_model['k']}")
    print(f"  C = {best_model['C']}")
    print(f"  sigma = {best_model['sigma']}")
    print(f"  Test Error = {best_model['test_error']*100:.2f}%")
    print(f"{'-'*80}")


if __name__ == "__main__":
    main()
