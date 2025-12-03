"""
Problem 1: Logistic Regression
CS 4375 - Problem Set 5

This module implements logistic regression with different regularization techniques
and compares them on the Sonar dataset. It also includes SVM implementation for
linearly separable data.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Load data from file.

    Args:
        filename: Path to data file

    Returns:
        X: Feature matrix (M, n) where M is number of data points, n is number of features
        y: Labels (M,) with values in {0, 1}
    """
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    # Convert labels from {1, 2} to {0, 1}
    y = y - 1
    return X, y


def normalize_features(X_train, X_valid, X_test):
    """
    Normalize features to have zero mean and unit variance based on training set statistics.

    Args:
        X_train: Training feature matrix
        X_valid: Validation feature matrix
        X_test: Test feature matrix

    Returns:
        X_train_norm, X_valid_norm, X_test_norm: Normalized feature matrices
        mean: Feature means from training set
        std: Feature standard deviations from training set
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_valid_norm = (X_valid - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_valid_norm, X_test_norm, mean, std


def sigmoid(z):
    """
    Compute sigmoid function.

    Args:
        z: Input array

    Returns:
        Sigmoid of z
    """
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy as a float
    """
    return np.mean(y_true == y_pred)


class LogisticRegression:
    """
    Logistic Regression classifier with optional L1/L2 regularization.
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=100000,
        regularization=None,
        lambda_reg=0.0,
        tolerance=1e-6,
        enable_weight_clipping=True,
        max_weight_value=100.0,
    ):
        """
        Initialize logistic regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Maximum number of iterations
            regularization: Type of regularization ('l1', 'l2', or None)
            lambda_reg: Regularization parameter
            tolerance: Convergence tolerance (set to 0 to disable early stopping)
            enable_weight_clipping: Whether to clip weights to prevent overflow
            max_weight_value: Maximum absolute value for weights (if clipping enabled)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.enable_weight_clipping = enable_weight_clipping
        self.max_weight_value = max_weight_value
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, verbose=False):
        """
        Fit logistic regression model using gradient descent.

        Args:
            X: Feature matrix (M, n) where M is number of data points, n is number of features
            y: Labels (M,)
            verbose: Whether to print progress
        """
        M, n = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        # Gradient descent
        for iteration in range(self.n_iterations):
            # Compute predictions
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            # Compute loss (negative log likelihood)
            loss = -np.mean(
                y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)
            )

            # Add regularization to loss
            if self.regularization == "l2":
                loss += (self.lambda_reg / 2) * np.sum(self.weights**2)
            elif self.regularization == "l1":
                loss += self.lambda_reg * np.sum(np.abs(self.weights))

            self.loss_history.append(loss)

            # Compute gradients
            dz = y_pred - y
            dw = np.dot(X.T, dz) / M
            db = np.mean(dz)

            # Add regularization gradient
            if self.regularization == "l2":
                dw += self.lambda_reg * self.weights

            # Clip gradients to prevent overflow
            max_grad_norm = 10.0
            grad_norm = np.linalg.norm(dw)
            if grad_norm > max_grad_norm:
                dw = dw * (max_grad_norm / grad_norm)
            db = np.clip(db, -max_grad_norm, max_grad_norm)

            # Update parameters
            if self.regularization == "l1":
                # Proximal gradient for L1 (soft thresholding)
                self.weights -= self.learning_rate * dw
                self.weights = np.sign(self.weights) * np.maximum(
                    np.abs(self.weights) - self.learning_rate * self.lambda_reg, 0
                )
                # Set very small values to exactly 0 to enforce sparsity
                self.weights[np.abs(self.weights) < 1e-10] = 0
            else:
                self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            # Clip weights to prevent overflow (if enabled)
            if self.enable_weight_clipping:
                self.weights = np.clip(
                    self.weights, -self.max_weight_value, self.max_weight_value
                )
                self.bias = np.clip(
                    self.bias, -self.max_weight_value, self.max_weight_value
                )

            # Check convergence
            if iteration > 0:
                # Check for NaN or Inf
                if np.isnan(loss) or np.isinf(loss):
                    if verbose:
                        print(f"Training diverged at iteration {iteration}. Stopping.")
                    break
                # Check convergence (if tolerance > 0)
                if (
                    self.tolerance > 0
                    and abs(self.loss_history[-1] - self.loss_history[-2])
                    < self.tolerance
                ):
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break

            if verbose and (iteration + 1) % 1000 == 0:
                print(f"Iteration {iteration + 1}, Loss: {loss:.6f}")

        return self

    def predict_proba(self, X):
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def count_nonzero_weights(self, threshold=1e-5):
        """
        Count number of non-zero weights.

        Args:
            threshold: Threshold below which weights are considered zero

        Returns:
            Number of non-zero weights
        """
        return np.sum(np.abs(self.weights) > threshold)


class SVM:
    """
    Support Vector Machine using gradient descent on hinge loss.
    """

    def __init__(
        self, learning_rate=0.01, n_iterations=100000, lambda_reg=0.01, tolerance=1e-6
    ):
        """
        Initialize SVM model.

        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Maximum number of iterations
            lambda_reg: Regularization parameter
            tolerance: Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, verbose=False):
        """
        Fit SVM model using gradient descent.

        Args:
            X: Feature matrix (M, n) where M is number of data points, n is number of features
            y: Labels (M,) in {0, 1}
            verbose: Whether to print progress
        """
        M, n = X.shape

        # Convert labels from {0, 1} to {-1, 1} for SVM
        y_svm = 2 * y - 1

        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        # Gradient descent
        for iteration in range(self.n_iterations):
            # Compute margins
            margins = y_svm * (np.dot(X, self.weights) + self.bias)

            # Compute hinge loss
            hinge_loss = np.maximum(0, 1 - margins)
            loss = self.lambda_reg * np.sum(self.weights**2) + np.mean(hinge_loss)
            self.loss_history.append(loss)

            # Compute gradients
            # For samples with margin < 1, gradient contributes
            mask = margins < 1
            dw = 2 * self.lambda_reg * self.weights - np.dot(X.T, y_svm * mask) / M
            db = -np.mean(y_svm * mask)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Check convergence
            if iteration > 0:
                # Check for NaN or Inf
                if np.isnan(loss) or np.isinf(loss):
                    if verbose:
                        print(f"Training diverged at iteration {iteration}. Stopping.")
                    break
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break

            if verbose and (iteration + 1) % 1000 == 0:
                print(f"Iteration {iteration + 1}, Loss: {loss:.6f}")

        return self

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels in {0, 1}
        """
        decision = np.dot(X, self.weights) + self.bias
        return (decision >= 0).astype(int)


def generate_linearly_separable_data(M=100, n=2, separation=2.0, random_state=42):
    """
    Generate linearly separable data in R^n.

    Args:
        M: Number of data points
        n: Number of features
        separation: Separation between classes
        random_state: Random seed

    Returns:
        X: Feature matrix (M, n)
        y: Labels (M,)
    """
    np.random.seed(random_state)

    # Generate random points for class 0
    X_class0 = np.random.randn(M // 2, n)

    # Generate random points for class 1, shifted by separation
    X_class1 = np.random.randn(M // 2, n)
    X_class1[:, 0] += separation  # Shift along first dimension

    # Combine data
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(M // 2), np.ones(M // 2)])

    # Shuffle
    indices = np.random.permutation(M)
    X = X[indices]
    y = y[indices]

    return X, y


def plot_decision_boundary_2d(X, y, models, model_names, title="Decision Boundaries"):
    """
    Plot decision boundaries for 2D data.

    Args:
        X: Feature matrix (M, 2) where M is number of data points
        y: Labels (M,)
        models: List of trained models
        model_names: List of model names
        title: Plot title
    """
    plt.figure(figsize=(12, 4))

    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    for idx, (model, name) in enumerate(zip(models, model_names)):
        plt.subplot(1, len(models), idx + 1)

        # Predict on mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        plt.contour(xx, yy, Z, colors="k", linewidths=0.5)

        # Plot data points
        plt.scatter(
            X[y == 0, 0],
            X[y == 0, 1],
            c="red",
            edgecolors="k",
            label="Class 0",
            alpha=0.7,
        )
        plt.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            c="blue",
            edgecolors="k",
            label="Class 1",
            alpha=0.7,
        )

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(name)
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("problem1_decision_boundaries.png", dpi=300, bbox_inches="tight")
    plt.close()


def tune_hyperparameter(
    X_train,
    y_train,
    X_valid,
    y_valid,
    lambda_values,
    regularization_type,
    verbose=False,
):
    """
    Tune regularization hyperparameter using validation set.

    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        lambda_values: List of lambda values to try
        regularization_type: 'l1' or 'l2'
        verbose: Whether to print training progress for each model

    Returns:
        best_lambda: Best lambda value
        best_model: Best trained model
        validation_accuracies: List of validation accuracies
    """
    best_accuracy = 0
    best_lambda = None
    best_model = None
    validation_accuracies = []

    print(f"\nTuning {regularization_type.upper()} regularization parameter:")
    print("-" * 60)

    for lambda_reg in lambda_values:
        # Use smaller learning rate for better stability
        lr = 0.01 if lambda_reg >= 1.0 else 0.1
        model = LogisticRegression(
            learning_rate=lr,
            n_iterations=100000,
            regularization=regularization_type,
            lambda_reg=lambda_reg,
            tolerance=1e-6,
        )
        if verbose:
            print(f"\nTraining with lambda={lambda_reg}:")
        model.fit(X_train, y_train, verbose=verbose)

        # Evaluate on validation set
        y_pred = model.predict(X_valid)
        accuracy = compute_accuracy(y_valid, y_pred)
        validation_accuracies.append(accuracy)

        print(
            f"Lambda: {lambda_reg:.4f}, Validation Accuracy: {accuracy:.4f}, "
            f"Non-zero weights: {model.count_nonzero_weights()}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_reg
            best_model = model

    print(
        f"\nBest lambda: {best_lambda:.4f} with validation accuracy: {best_accuracy:.4f}"
    )

    return best_lambda, best_model, validation_accuracies


def main():
    """
    Main function to run all experiments for Problem 1.
    """
    print("=" * 70)
    print("Problem 1: Logistic Regression")
    print("=" * 70)

    # Load data
    print("\nLoading Sonar dataset...")
    X_train, y_train = load_data("sonar_train.data")
    X_valid, y_valid = load_data("sonar_valid.data")
    X_test, y_test = load_data("sonar_test.data")

    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_valid.shape}")
    print(f"Test set size: {X_test.shape}")

    # Normalize features
    print("\nNormalizing features (zero mean, unit variance)...")
    X_train, X_valid, X_test, mean, std = normalize_features(X_train, X_valid, X_test)

    # -------------------------------------------------------------------------
    # Part 1: Standard Logistic Regression (no regularization)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 1: Standard Logistic Regression (No Regularization)")
    print("=" * 70)

    lr_standard = LogisticRegression(
        learning_rate=0.1, n_iterations=100000, regularization=None, tolerance=1e-6
    )
    lr_standard.fit(X_train, y_train, verbose=True)

    y_pred_test = lr_standard.predict(X_test)
    accuracy_standard = compute_accuracy(y_test, y_pred_test)

    print(f"\nStandard Logistic Regression:")
    print(f"Test Accuracy: {accuracy_standard:.4f}")
    print(
        f"Number of non-zero weights: {lr_standard.count_nonzero_weights()}/{len(lr_standard.weights)}"
    )

    # -------------------------------------------------------------------------
    # Part 2: Logistic Regression with L2 Regularization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 2: Logistic Regression with L2 Regularization")
    print("=" * 70)

    lambda_values_l2 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda_l2, best_model_l2, val_acc_l2 = tune_hyperparameter(
        X_train, y_train, X_valid, y_valid, lambda_values_l2, "l2"
    )

    # Evaluate on test set
    y_pred_test_l2 = best_model_l2.predict(X_test)
    accuracy_l2 = compute_accuracy(y_test, y_pred_test_l2)

    print(f"\nL2 Regularized Logistic Regression Results:")
    print(f"Selected lambda: {best_lambda_l2:.4f}")
    print(f"Test Accuracy: {accuracy_l2:.4f}")
    print(f"Bias: {best_model_l2.bias:.6f}")
    print(
        f"Number of non-zero weights: {best_model_l2.count_nonzero_weights()}/{len(best_model_l2.weights)}"
    )
    print(f"\nWeights: {best_model_l2.weights}")

    # -------------------------------------------------------------------------
    # Part 3: Logistic Regression with L1 Regularization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 3: Logistic Regression with L1 Regularization")
    print("=" * 70)

    lambda_values_l1 = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_lambda_l1, best_model_l1, val_acc_l1 = tune_hyperparameter(
        X_train, y_train, X_valid, y_valid, lambda_values_l1, "l1"
    )

    # Evaluate on test set
    y_pred_test_l1 = best_model_l1.predict(X_test)
    accuracy_l1 = compute_accuracy(y_test, y_pred_test_l1)

    print(f"\nL1 Regularized Logistic Regression Results:")
    print(f"Selected lambda: {best_lambda_l1:.4f}")
    print(f"Test Accuracy: {accuracy_l1:.4f}")
    print(f"Bias: {best_model_l1.bias:.6f}")
    print(
        f"Number of non-zero weights: {best_model_l1.count_nonzero_weights()}/{len(best_model_l1.weights)}"
    )
    print(f"\nWeights: {best_model_l1.weights}")

    # -------------------------------------------------------------------------
    # Part 4: Sparsity Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 4: Sparsity Comparison")
    print("=" * 70)

    print(f"\nNumber of non-zero weights:")
    print(
        f"  Standard LR: {lr_standard.count_nonzero_weights()}/{len(lr_standard.weights)}"
    )
    print(
        f"  L2 Regularized LR: {best_model_l2.count_nonzero_weights()}/{len(best_model_l2.weights)}"
    )
    print(
        f"  L1 Regularized LR: {best_model_l1.count_nonzero_weights()}/{len(best_model_l1.weights)}"
    )

    # -------------------------------------------------------------------------
    # Part 5: Linearly Separable Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 5: Linearly Separable Data")
    print("=" * 70)

    # Generate linearly separable data
    print("\nGenerating linearly separable data in R^2...")
    X_sep, y_sep = generate_linearly_separable_data(
        M=200, n=2, separation=2.0, random_state=42
    )

    # Split into train/val/test
    n_train = 100
    n_val = 50
    X_sep_train = X_sep[:n_train]
    y_sep_train = y_sep[:n_train]
    X_sep_val = X_sep[n_train : n_train + n_val]
    y_sep_val = y_sep[n_train : n_train + n_val]
    X_sep_test = X_sep[n_train + n_val :]
    y_sep_test = y_sep[n_train + n_val :]

    # Part 5a: Standard logistic regression on linearly separable data
    print("\n(a) Standard Logistic Regression on Linearly Separable Data:")
    print("-" * 60)

    lr_sep_standard = LogisticRegression(
        learning_rate=0.1,  # Higher learning rate for more dramatic weight growth
        n_iterations=10000,
        regularization=None,
        tolerance=0,  # Disable early stopping to show non-convergence
        enable_weight_clipping=False,  # Disable weight clipping to show weight growth
    )
    lr_sep_standard.fit(X_sep_train, y_sep_train, verbose=True)

    accuracy_sep_standard = compute_accuracy(
        y_sep_test, lr_sep_standard.predict(X_sep_test)
    )
    print(f"\nTest Accuracy: {accuracy_sep_standard:.4f}")
    print(f"Final weights magnitude: {np.linalg.norm(lr_sep_standard.weights):.4f}")
    print(
        f"Weight vector L-infinity norm (max abs value): {np.max(np.abs(lr_sep_standard.weights)):.4f}"
    )

    # Part 5b: L2 regularized logistic regression on linearly separable data
    print("\n(b) L2 Regularized Logistic Regression on Linearly Separable Data:")
    print("-" * 60)

    lambda_values_sep = [0.01, 0.1, 1.0, 10.0]
    best_lambda_sep, best_model_sep, _ = tune_hyperparameter(
        X_sep_train,
        y_sep_train,
        X_sep_val,
        y_sep_val,
        lambda_values_sep,
        "l2",
        verbose=True,
    )

    accuracy_sep_l2 = compute_accuracy(y_sep_test, best_model_sep.predict(X_sep_test))
    print(f"\nTest Accuracy: {accuracy_sep_l2:.4f}")
    print(f"Weights magnitude: {np.linalg.norm(best_model_sep.weights):.4f}")

    # Part 5c: Fit SVM and compare
    print("\n(c) SVM vs Logistic Regression Comparison:")
    print("-" * 60)

    print("\nTraining SVM...")
    svm = SVM(learning_rate=0.1, n_iterations=10000, lambda_reg=0.01, tolerance=1e-6)
    svm.fit(X_sep_train, y_sep_train, verbose=True)

    accuracy_sep_svm = compute_accuracy(y_sep_test, svm.predict(X_sep_test))
    print(f"\nSVM Test Accuracy: {accuracy_sep_svm:.4f}")
    print(f"SVM Weights magnitude: {np.linalg.norm(svm.weights):.4f}")

    # Plot decision boundaries
    print("\nPlotting decision boundaries...")
    plot_decision_boundary_2d(
        X_sep_test,
        y_sep_test,
        [svm, best_model_sep, lr_sep_standard],
        ["SVM", "L2 Regularized LR", "Standard LR"],
        title="Comparison on Linearly Separable Data",
    )


if __name__ == "__main__":
    main()
