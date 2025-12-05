import numpy as np


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    # Convert labels from {1, 2} to {0, 1}
    y = y - 1
    return X, y


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


class GaussianNaiveBayes:

    def __init__(self, use_prior=False, prior_alpha=3.0, prior_beta=0.5):
        self.use_prior = use_prior
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.class_priors = None  # p(y=c) for each class c
        self.means = None  # mu_{i,c} for each feature i and class c
        self.variances = None  # sigma_{i,c}^2 for each feature i and class c
        self.classes = None  # Unique class labels

    def fit(self, X, y):
        M, n = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize parameter arrays
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n))
        self.variances = np.zeros((n_classes, n))

        # Compute MLE/MAP for each class
        for idx, c in enumerate(self.classes):
            # Get samples for class c
            X_c = X[y == c]
            N_c = X_c.shape[0]

            # MLE for class prior: p(y=c) = N_c / M
            self.class_priors[idx] = N_c / M

            # MLE for mean: mu_{i,c} = mean of feature i in class c
            self.means[idx, :] = np.mean(X_c, axis=0)

            # Variance estimation: MLE or MAP with InverseGamma prior
            var_mle = np.var(X_c, axis=0)

            if self.use_prior:
                # MAP estimate with InverseGamma(alpha, beta) prior
                # Posterior: InverseGamma(alpha + N_c/2, beta + N_c * var_mle / 2)
                # MAP (mode): (beta + N_c * var_mle / 2) / (alpha + N_c/2 + 1)
                posterior_alpha = self.prior_alpha + N_c / 2
                posterior_beta = self.prior_beta + N_c * var_mle / 2
                self.variances[idx, :] = posterior_beta / (posterior_alpha + 1)
            else:
                # MLE with small constant to avoid division by zero
                self.variances[idx, :] = var_mle + 1e-9

        return self

    def _compute_log_likelihood(self, X, class_idx):
        # Get parameters for this class
        mean = self.means[class_idx]
        var = self.variances[class_idx]

        # Compute log probability for each feature (Gaussian PDF in log space)
        # log p(x_i | y) = -0.5 * log(2*pi*sigma^2) - (x_i - mu)^2 / (2*sigma^2)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_likelihood -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)

        return log_likelihood

    def predict_log_proba(self, X):
        M = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((M, n_classes))

        for idx in range(n_classes):
            # log p(y=c | x) = log p(y=c) + log p(x | y=c)
            log_probs[:, idx] = np.log(
                self.class_priors[idx]
            ) + self._compute_log_likelihood(X, idx)

        return log_probs

    def predict_proba(self, X):
        log_probs = self.predict_log_proba(X)

        # Convert log probabilities to probabilities using log-sum-exp trick
        # p(y=c | x) = exp(log p(y=c | x)) / sum_c exp(log p(y=c | x))
        max_log_prob = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_prob)
        probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        # Predict class with highest log probability
        class_indices = np.argmax(log_probs, axis=1)
        return self.classes[class_indices]


def main():
    print("=" * 70)
    print("Problem 2: Gaussian Naive Bayes")
    print("=" * 70)

    # Load data
    print("\nLoading Sonar dataset...")
    X_train, y_train = load_data("sonar_train.data")
    X_valid, y_valid = load_data("sonar_valid.data")
    X_test, y_test = load_data("sonar_test.data")

    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_valid.shape}")
    print(f"Test set size: {X_test.shape}")

    # -------------------------------------------------------------------------
    # Part 1: Log-Likelihood and MLE Computation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 1: Log-Likelihood and MLE Computation")
    print("=" * 70)

    print("\nCOMPUTED MLE PARAMETERS:")
    print("-" * 70)
    print(
        f"Training set: M = {X_train.shape[0]} data points, n = {X_train.shape[1]} features"
    )

    # Fit model to compute MLEs
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    # Display computed MLEs
    print(f"\n1. Class Priors p_hat(y=c):")
    print(f"   p_hat(y=0) = {gnb.class_priors[0]:.4f}")
    print(f"   p_hat(y=1) = {gnb.class_priors[1]:.4f}")
    print(
        f"   (N_0 = {int(gnb.class_priors[0] * X_train.shape[0])}, N_1 = {int(gnb.class_priors[1] * X_train.shape[0])})"
    )

    print(f"\n2. Feature Means mu_hat_{{i,c}}:")
    print(f"   Class 0: {gnb.means[0, :]}")
    print(f"   Class 1: {gnb.means[1, :]}")

    print(f"\n3. Feature Variances sigma_hat_{{i,c}}^2:")
    print(f"   Class 0: {gnb.variances[0, :]}")
    print(f"   Class 1: {gnb.variances[1, :]}")

    print(
        f"\nAll MLE parameters computed from training data using closed-form solutions."
    )

    # -------------------------------------------------------------------------
    # Part 2: Model Accuracy on Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 2: Model Accuracy on Test Set")
    print("=" * 70)

    print("\nEvaluating trained Gaussian Naive Bayes model...")

    # Evaluate on training set
    y_pred_train = gnb.predict(X_train)
    train_accuracy = compute_accuracy(y_train, y_pred_train)

    # Evaluate on validation set
    y_pred_valid = gnb.predict(X_valid)
    valid_accuracy = compute_accuracy(y_valid, y_pred_valid)

    # Evaluate on test set
    y_pred_test = gnb.predict(X_test)
    test_accuracy = compute_accuracy(y_test, y_pred_test)

    print("\nAccuracy Results:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {valid_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

    print(f"\nThe trained model achieves {test_accuracy:.4f} accuracy on the test set.")

    # -------------------------------------------------------------------------
    # Part 3: Bayesian Approach with InverseGamma Prior
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 3: Bayesian Approach with InverseGamma Prior on Variances")
    print("=" * 70)

    # Analyze data characteristics
    print("\nData characteristics:")
    print(f"  Feature value range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Mean of all features: {X_train.mean():.4f}")
    print(f"  Many small values < 0.01 in the data")
    print(f"  Small dataset: M={X_train.shape[0]}, n={X_train.shape[1]}")

    print("\nUsing InverseGamma(alpha=3, beta=0.5) prior on variances:")
    print("  - Prevents variance from approaching 0 (numerical stability)")
    print("  - Regularizes variance estimates with small sample sizes")
    print("  - Beta chosen based on data scale (mean roughly 0.25)")
    print("  - MAP estimate: (beta + N_c * var_MLE / 2) / (alpha + N_c/2 + 1)")

    # Fit Bayesian model with InverseGamma prior
    print("\nTraining Bayesian Gaussian NB with InverseGamma(3, 0.5) prior...")
    gnb_bayesian = GaussianNaiveBayes(use_prior=True, prior_alpha=3.0, prior_beta=0.5)
    gnb_bayesian.fit(X_train, y_train)

    # Evaluate Bayesian model
    y_pred_train_bayes = gnb_bayesian.predict(X_train)
    train_accuracy_bayes = compute_accuracy(y_train, y_pred_train_bayes)

    y_pred_valid_bayes = gnb_bayesian.predict(X_valid)
    valid_accuracy_bayes = compute_accuracy(y_valid, y_pred_valid_bayes)

    y_pred_test_bayes = gnb_bayesian.predict(X_test)
    test_accuracy_bayes = compute_accuracy(y_test, y_pred_test_bayes)

    print("\nCOMPARISON: MLE vs MAP (Bayesian with Prior)")
    print("=" * 70)

    print("\nMLE (No Prior):")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {valid_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Variance (Class 0, first 5 features): {gnb.variances[0, :5]}")
    print(f"  Variance (Class 1, first 5 features): {gnb.variances[1, :5]}")

    print("\nMAP with InverseGamma(3, 0.5) Prior:")
    print(f"  Training Accuracy: {train_accuracy_bayes:.4f}")
    print(f"  Validation Accuracy: {valid_accuracy_bayes:.4f}")
    print(f"  Test Accuracy: {test_accuracy_bayes:.4f}")
    print(f"  Variance (Class 0, first 5 features): {gnb_bayesian.variances[0, :5]}")
    print(f"  Variance (Class 1, first 5 features): {gnb_bayesian.variances[1, :5]}")

    print("\nDifference in Variances (MAP - MLE):")
    var_diff_0 = gnb_bayesian.variances[0, :] - gnb.variances[0, :]
    var_diff_1 = gnb_bayesian.variances[1, :] - gnb.variances[1, :]
    print(f"  Class 0 (first 5): {var_diff_0[:5]}")
    print(f"  Class 1 (first 5): {var_diff_1[:5]}")
    print(f"  Mean absolute difference (Class 0): {np.mean(np.abs(var_diff_0)):.6f}")
    print(f"  Mean absolute difference (Class 1): {np.mean(np.abs(var_diff_1)):.6f}")


if __name__ == "__main__":
    main()
