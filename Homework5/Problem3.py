import numpy as np


def load_and_preprocess_data(filename):
    """
    Load the leaf data and preprocess it.

    Returns:
        X: Preprocessed features (M x n) with mean zero and variance one
        y: Class labels (M,)
    """
    data = np.loadtxt(filename, delimiter=",")

    # First column is the class label
    y = data[:, 0].astype(int)

    # Remaining columns are features
    X = data[:, 1:]

    # Preprocess: mean zero and variance one
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, y


def kmeans_plus_plus_init(X, k, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    M, n = X.shape
    centers = np.zeros((k, n))

    # Choose first center uniformly at random
    centers[0] = X[np.random.randint(M)]

    # Choose remaining k-1 centers
    for i in range(1, k):
        # Compute squared distances to nearest existing center
        distances = np.zeros(M)
        for j in range(M):
            min_dist = np.inf
            for c in range(i):
                dist = np.sum((X[j] - centers[c]) ** 2)
                if dist < min_dist:
                    min_dist = dist
            distances[j] = min_dist

        # Choose next center with probability proportional to squared distance
        probabilities = distances / np.sum(distances)
        centers[i] = X[np.random.choice(M, p=probabilities)]

    return centers


def fix_covariance_matrix(cov, epsilon=1e-6):
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Set eigenvalues below epsilon to epsilon
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Reconstruct covariance matrix: Q * F * Q^T
    cov_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return cov_fixed


def gaussian_pdf(X, mu, cov):
    n = X.shape[1]

    # Add small regularization to ensure numerical stability
    cov_reg = cov + 1e-10 * np.eye(n)

    diff = X - mu
    cov_inv = np.linalg.inv(cov_reg)
    cov_det = np.linalg.det(cov_reg)

    # Compute PDF
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    normalization = 1.0 / np.sqrt((2 * np.pi) ** n * cov_det)

    return normalization * np.exp(exponent)


def compute_log_likelihood(X, pi, mu, cov):
    M = X.shape[0]
    k = len(pi)

    # Compute weighted PDF for each component
    weighted_pdfs = np.zeros((M, k))
    for j in range(k):
        weighted_pdfs[:, j] = pi[j] * gaussian_pdf(X, mu[j], cov[j])

    # Sum over components and take log
    log_likelihood = np.sum(np.log(np.sum(weighted_pdfs, axis=1) + 1e-300))

    return log_likelihood


def em_algorithm(X, k, max_iter=100, tol=1e-6, epsilon=1e-6, random_state=None):
    M, n = X.shape

    # Initialize using k-means++
    mu = kmeans_plus_plus_init(X, k, random_state)

    # Initialize covariances to identity matrix
    cov = np.array([np.eye(n) for _ in range(k)])

    # Initialize mixing coefficients uniformly
    pi = np.ones(k) / k

    prev_log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-step: Compute responsibilities
        responsibilities = np.zeros((M, k))

        for j in range(k):
            responsibilities[:, j] = pi[j] * gaussian_pdf(X, mu[j], cov[j])

        # Normalize responsibilities
        responsibility_sums = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities = responsibilities / (responsibility_sums + 1e-300)

        # M-step: Update parameters
        N_k = np.sum(responsibilities, axis=0)

        # Update mixing coefficients
        pi = N_k / M

        # Update means
        for j in range(k):
            mu[j] = np.sum(responsibilities[:, j : j + 1] * X, axis=0) / (
                N_k[j] + 1e-300
            )

        # Update covariances
        for j in range(k):
            diff = X - mu[j]
            weighted_diff = responsibilities[:, j : j + 1] * diff
            cov[j] = (weighted_diff.T @ diff) / (N_k[j] + 1e-300)

            # Fix covariance matrix to ensure minimum eigenvalue
            cov[j] = fix_covariance_matrix(cov[j], epsilon)

        # Compute log-likelihood
        current_log_likelihood = compute_log_likelihood(X, pi, mu, cov)

        # Check for convergence
        if abs(current_log_likelihood - prev_log_likelihood) < tol:
            break

        prev_log_likelihood = current_log_likelihood

    return pi, mu, cov, current_log_likelihood


def run_experiments(X, k_values, num_trials=20, epsilon=1e-6):
    results = {}

    for k in k_values:
        print(f"\nRunning experiments for k={k}...")
        log_likelihoods = []

        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end="")

            # Run EM algorithm with different random seed
            _, _, _, log_likelihood = em_algorithm(
                X, k, max_iter=100, tol=1e-6, epsilon=epsilon, random_state=trial
            )

            log_likelihoods.append(log_likelihood)
            print(f" log-likelihood: {log_likelihood:.4f}")

        log_likelihoods = np.array(log_likelihoods)

        results[k] = {
            "mean": np.mean(log_likelihoods),
            "variance": np.var(log_likelihoods),
            "all_values": log_likelihoods,
        }

        print(f"\nResults for k={k}:")
        print(f"  Mean log-likelihood: {results[k]['mean']:.4f}")
        print(f"  Variance of log-likelihood: {results[k]['variance']:.4f}")

    return results


def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data("leaf.data")
    print(f"Data shape: M={X.shape[0]} samples, n={X.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y))}")

    # Run experiments for k in {10, 20, 30, 36}
    k_values = [10, 20, 30, 36]
    epsilon = 1e-6

    print(f"\nRunning Gaussian Mixture Model with epsilon={epsilon}")
    results = run_experiments(X, k_values, num_trials=20, epsilon=epsilon)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)

    for k in k_values:
        print(f"\nk = {k}:")
        print(f"  Mean log-likelihood:     {results[k]['mean']:.6f}")
        print(f"  Variance of log-likelihood: {results[k]['variance']:.6f}")


if __name__ == "__main__":
    main()
