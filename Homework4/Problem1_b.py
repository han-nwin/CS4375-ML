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
    Returns: eigenvalues, eigenvectors, mean
    """
    print("Performing PCA...")

    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    M, n = X.shape
    print(f"Data shape: ({M}, {n})")

    # Compute covariance matrix
    cov_matrix = (1 / (M - 1)) * np.dot(X_centered.T, X_centered)

    # Compute eigenvalues and eigenvectors
    print("Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, mean


def compute_pi_distribution(eigenvectors, k):
    """
    Compute π distribution for feature selection
    π_j = (1/k) * sum_{i=1..k} v(i)_j^2

    Parameters:
    eigenvectors: All eigenvectors (n_features x n_features)
    k: Number of top eigenvectors to use

    Returns:
    π: Probability distribution over features
    """
    # Get top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]  # (n_features x k)

    # Compute π_j = (1/k) * sum_{i=1..k} v(i)_j^2
    pi = np.mean(top_k_eigenvectors**2, axis=1)

    # Normalize to ensure it's a valid probability distribution
    pi = pi / np.sum(pi)

    return pi


def sample_features(pi, s):
    """
    Sample s features from the probability distribution π
    Remove duplicates (use each feature only once)

    Parameters:
    pi: Probability distribution over features
    s: Number of features to sample

    Returns:
    selected_features: Array of unique feature indices
    """
    # Sample s features with replacement
    sampled = np.random.choice(len(pi), size=s, replace=True, p=pi)

    # Remove duplicates
    selected_features = np.unique(sampled)

    return selected_features


def main():
    """Main function for Problem 1b: PCA Feature Selection"""
    print("=" * 80)
    print("PROBLEM 1_b: PCA for Feature Selection")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Perform PCA once
    print("\n" + "-" * 80)
    eigenvalues, eigenvectors, mean = perform_pca(X_train)

    # Calculate cumulative variance
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues)
    variance_ratios = cumulative_variance / total_variance

    # Build set K
    thresholds = [0.99, 0.95, 0.90, 0.80, 0.75]
    K = {}
    for z in thresholds:
        k = int(np.argmax(variance_ratios >= z) + 1)
        K[f"k{int(z*100)}"] = k

    print(f"\nSet K = {K}")

    # Values of s to test
    s_values = list(range(10, 101, 10))  # [10, 20, ..., 100]
    print(f"s values: {s_values}")

    # Number of experiments per (k, s) combination
    num_experiments = 10

    # Hyperparameters (use best from Problem 1a)
    C = 1.0
    sigma = 10000.0

    # Store results
    results = {}

    print("\n" + "=" * 80)
    print("Running Experiments")
    print("=" * 80)

    # For each k in K
    for k_name, k_value in K.items():
        print(f"\n{'-'*80}")
        print(f"Processing {k_name} = {k_value} components")
        print(f"{'-'*80}")

        # Compute π distribution for this k
        pi = compute_pi_distribution(eigenvectors, k_value)
        print(f"Computed π distribution (sum = {np.sum(pi):.6f})")

        results[k_name] = {}

        # For each s value
        for s in s_values:
            print(f"\n  Testing s = {s} features:")

            test_errors = []

            # Run num_experiments experiments
            for exp in range(num_experiments):
                # Sample features
                selected_features = sample_features(pi, s)
                n_selected = len(selected_features)

                # Extract selected features from data
                X_train_selected = X_train[:, selected_features]
                X_valid_selected = X_valid[:, selected_features]
                X_test_selected = X_test[:, selected_features]

                # Train SVM
                svm = SVM_Gaussian(C=C, sigma=sigma)
                svm.fit(X_train_selected, y_train)

                # Evaluate on test set
                y_test_pred = svm.predict(X_test_selected)
                test_accuracy = np.mean(y_test_pred == y_test)
                test_error = 1 - test_accuracy

                test_errors.append(test_error)

                if (exp + 1) % 5 == 0:
                    print(f"    Experiment {exp+1}/{num_experiments} done")

            # Calculate mean and std
            mean_error = np.mean(test_errors)
            std_error = np.std(test_errors)

            results[k_name][s] = {
                "mean_error": mean_error,
                "std_error": std_error,
                "test_errors": test_errors,
            }

            print(f"    Mean test error: {mean_error*100:.2f}% ± {std_error*100:.2f}%")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'k \\ s':<10}", end="")
    for s in s_values:
        print(f"{s:<12}", end="")
    print()
    print("-" * 140)

    for k_name in K.keys():
        print(f"{k_name:<10}", end="")
        for s in s_values:
            mean_err = results[k_name][s]["mean_error"] * 100
            std_err = results[k_name][s]["std_error"] * 100
            print(f"{mean_err:.2f}±{std_err:.2f}  ", end="")
        print()

    # Find best configuration
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    best_overall = None
    best_error = float('inf')

    for k_name in K.keys():
        best_s_for_k = None
        best_error_for_k = float('inf')

        for s in s_values:
            mean_err = results[k_name][s]["mean_error"]
            if mean_err < best_error_for_k:
                best_error_for_k = mean_err
                best_s_for_k = s

            if mean_err < best_error:
                best_error = mean_err
                best_overall = (k_name, s, mean_err)

        print(f"\nBest for {k_name} (k={K[k_name]}): s={best_s_for_k}, error={best_error_for_k*100:.2f}%")

    print(f"\n{'-'*80}")
    print(f"BEST OVERALL: {best_overall[0]} (k={K[best_overall[0]]}), s={best_overall[1]}")
    print(f"Mean test error: {best_overall[2]*100:.2f}%")
    print(f"{'-'*80}")

    # Compare with baseline (from Problem 1a)
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINES")
    print("=" * 80)
    print(f"\nFeature Selection Best: {best_overall[2]*100:.2f}% (k={K[best_overall[0]]}, s={best_overall[1]})")
    print(f"PCA Projection Best (from 1a): 1.80% (k=855)")
    print(f"No Feature Selection (from 1a): 2.00% (all 5000 features)")
    print("=" * 80)


if __name__ == "__main__":
    main()
