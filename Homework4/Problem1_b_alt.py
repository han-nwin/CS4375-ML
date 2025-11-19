import numpy as np
from cvxopt import matrix, solvers


class SVM_Linear_Primal:
    """
    Primal soft-margin linear SVM
    """

    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Solve primal SVM QP
        """
        m, d = X.shape

        y = np.where(y <= 0, -1, y).astype(float)
        self.y_train = y
        self.X_train = X

        n_vars = d + m

        H = np.zeros((n_vars, n_vars))
        H[:d, :d] = np.eye(d)

        f = np.zeros(n_vars)
        f[d:] = self.C

        G1 = np.zeros((m, n_vars))
        G1[:, :d] = -(y[:, None] * X)
        G1[:, d:] = -np.eye(m)
        h1 = -np.ones(m)

        G2 = np.zeros((m, n_vars))
        G2[:, d:] = -np.eye(m)
        h2 = np.zeros(m)

        G = np.vstack([G1, G2])
        h = np.hstack([h1, h2])

        H_cvx = matrix(H)
        f_cvx = matrix(f)
        G_cvx = matrix(G)
        h_cvx = matrix(h)

        solvers.options["show_progress"] = False
        sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx)

        z = np.array(sol["x"]).flatten()
        self.w = z[:d]

    def predict(self, X):
        decision = X @ self.w
        return np.where(decision >= 0, 1, -1)


def load_data():
    """Load training, validation, and test data"""
    train_data = np.loadtxt("gisette_train.data", delimiter=",")
    valid_data = np.loadtxt("gisette_valid.data", delimiter=",")
    test_data = np.loadtxt("gisette_test.data", delimiter=",")

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

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    M, n = X.shape
    print(f"Data shape: ({M}, {n})")

    cov_matrix = (1 / (M - 1)) * np.dot(X_centered.T, X_centered)

    print("Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, mean


def compute_pi_distribution(eigenvectors, k):
    """
    Compute pi distribution for feature selection
    pi_j = (1/k) * sum_{i=1..k} v(i)_j^2
    """
    top_k_eigenvectors = eigenvectors[:, :k]
    pi = np.mean(top_k_eigenvectors**2, axis=1)
    pi = pi / np.sum(pi)
    return pi


def sample_features(pi, s):
    """
    Sample s features from the probability distribution pi
    Remove duplicates
    """
    sampled = np.random.choice(len(pi), size=s, replace=True, p=pi)
    selected_features = np.unique(sampled)
    return selected_features


def main():
    """Main function for Problem 1b: PCA Feature Selection"""
    print("=" * 80)
    print("PROBLEM 1_b: PCA for Feature Selection (Linear SVM)")
    print("=" * 80)

    print("\nLoading data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    print("\n" + "-" * 80)
    eigenvalues, eigenvectors, mean = perform_pca(X_train)

    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues)
    variance_ratios = cumulative_variance / total_variance

    thresholds = [0.99, 0.95, 0.90, 0.80, 0.75]
    K = {}
    for z in thresholds:
        k = int(np.argmax(variance_ratios >= z) + 1)
        K[f"k{int(z*100)}"] = k

    print(f"\nSet K = {K}")

    s_values = list(range(10, 101, 10))
    print(f"s values: {s_values}")

    num_experiments = 10
    C = 1.0

    results = {}

    print("\n" + "=" * 80)
    print("Running Experiments")
    print("=" * 80)

    for k_name, k_value in K.items():
        print(f"\n{'-'*80}")
        print(f"Processing {k_name} = {k_value} components")
        print(f"{'-'*80}")

        pi = compute_pi_distribution(eigenvectors, k_value)
        print(f"Computed pi distribution (sum = {np.sum(pi):.6f})")

        results[k_name] = {}

        for s in s_values:
            print(f"\n  Testing s = {s} features:")

            test_errors = []

            for exp in range(num_experiments):
                selected_features = sample_features(pi, s)

                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]

                svm = SVM_Linear_Primal(C=C)
                svm.fit(X_train_selected, y_train)

                y_test_pred = svm.predict(X_test_selected)
                test_accuracy = np.mean(y_test_pred == y_test)
                test_error = 1 - test_accuracy

                test_errors.append(test_error)

                if (exp + 1) % 5 == 0:
                    print(f"    Experiment {exp+1}/{num_experiments} done")

            mean_error = np.mean(test_errors)
            std_error = np.std(test_errors)

            results[k_name][s] = {
                "mean_error": mean_error,
                "std_error": std_error,
                "test_errors": test_errors,
            }

            print(f"    Mean test error: {mean_error*100:.2f}% +/- {std_error*100:.2f}%")

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
            print(f"{mean_err:.2f}+/-{std_err:.2f}  ", end="")
        print()

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

    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINES")
    print("=" * 80)
    print(f"\nFeature Selection Best: {best_overall[2]*100:.2f}% (k={K[best_overall[0]]}, s={best_overall[1]})")
    print(f"PCA Projection Best (from 1a): 1.80% (k=855)")
    print(f"No Feature Selection (from 1a): 2.00% (all 5000 features)")
    print("=" * 80)


if __name__ == "__main__":
    main()
