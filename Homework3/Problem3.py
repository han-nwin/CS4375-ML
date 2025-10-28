import csv
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Data loading and utilities
# ----------------------------


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(x.strip()) for x in row])
    data = np.array(rows, dtype=float)
    y_raw = data[:, 0]
    X = data[:, 1:]
    # Map labels to {-1, +1}
    if set(np.unique(y_raw)).issubset({-1.0, 1.0}):
        y = y_raw.astype(int)
    else:
        y = np.where(y_raw > 0, 1, -1).astype(int)
    return X, y


def sign_pm1(z: np.ndarray) -> np.ndarray:
    """
    Deterministic sign in {-1, +1}; break ties at 0 as +1.
    """
    s = np.sign(z)
    s[s == 0] = 1
    return s


@dataclass
class Stump:
    feature: int
    threshold: float
    polarity: int  # either +1 or -1

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        left_is_pos = (X[:, self.feature] <= self.threshold).astype(int) * 2 - 1
        return sign_pm1(self.polarity * left_is_pos)

    def __str__(self) -> str:
        # A readable single-split tree line
        left_pred = 1 if self.polarity == 1 else -1
        right_pred = -left_pred
        return f"if x[{self.feature}] <= {self.threshold:.6g}: predict {left_pred} else: predict {right_pred}"


def ascii_tree(stump: "Stump") -> str:
    """
    Small ASCII rendering of a one-split decision tree for the report.
    """
    lp = 1 if stump.polarity == 1 else -1
    rp = -lp
    return (
        f"feature x[{stump.feature}] <= {stump.threshold:.6g}\n"
        f"|-- True  -> predict {lp}\n"
        f"|__ False -> predict {rp}"
    )


def unique_thresholds_for_feature(x: np.ndarray) -> List[float]:
    """
    Generate candidate thresholds for a feature column x.
    For binary features, this gives [0.5].
    For numeric: midpoints between sorted unique values.
    """
    vals = np.unique(x)
    if len(vals) == 1:
        return [vals[0]]
    mids = []
    for a, b in zip(vals[:-1], vals[1:]):
        mids.append((a + b) / 2.0)
    return mids


def generate_all_stumps(X: np.ndarray) -> List[Stump]:
    """
    Generate the full hypothesis space of decision stumps over all features,
    trying both polarities for each threshold.
    """
    _, n = X.shape
    stumps: List[Stump] = []
    for j in range(n):
        thresholds = unique_thresholds_for_feature(X[:, j])
        for thr in thresholds:
            stumps.append(Stump(j, thr, +1))
            stumps.append(Stump(j, thr, -1))
    return stumps


def weighted_error(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    mism = (y_true != y_pred).astype(float)
    return float(np.dot(w, mism)) / float(np.sum(w))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def safe_alpha_from_epsilon(epsilon: float, eps_floor: float = 1e-12) -> float:
    """
    Compute alpha = 0.5 * log((1 - epsilon) / epsilon) with clipping for stability.
    """
    epsilon = min(max(epsilon, eps_floor), 1 - eps_floor)
    return 0.5 * math.log((1 - epsilon) / epsilon)


# ----------------------------
# AdaBoost (10 rounds)
# ----------------------------


@dataclass
class AdaRound:
    stump: Stump
    epsilon: float
    alpha: float
    Z_t: float
    train_acc: float
    test_acc: float


def adaboost_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    T: int = 10,
    all_stumps: Optional[List[Stump]] = None,
) -> Tuple[List[AdaRound], np.ndarray, List[Stump]]:
    """
    Train AdaBoost classifier with decision stumps for T rounds.
    Returns the selected rounds info, alpha weights, and the selected stump list.
    """
    M = X_train.shape[0]
    if all_stumps is None:
        all_stumps = generate_all_stumps(X_train)

    # Initialize sample weights uniformly
    w = np.ones(M) / M
    rounds: List[AdaRound] = []
    alphas: List[float] = []
    stumps_selected: List[Stump] = []

    # Running margins
    F_train = np.zeros(M)
    F_test = np.zeros(X_test.shape[0])

    for t in range(T):
        # Find best stump under current weights
        best_epsilon = float("inf")
        best_stump = None
        best_pred = None

        for stump in all_stumps:
            pred = stump.predict_raw(X_train)
            eps = weighted_error(y_train, pred, w)
            if eps < best_epsilon:
                best_epsilon = eps
                best_stump = stump
                best_pred = pred

        alpha = safe_alpha_from_epsilon(best_epsilon)

        # Update weights
        w = w * np.exp(-alpha * y_train * best_pred)
        Z_t = float(np.sum(w))  # normalizer
        w = w / Z_t

        # Track model
        stumps_selected.append(best_stump)
        alphas.append(alpha)

        # Update margins and compute accuracies
        F_train += alpha * best_stump.predict_raw(X_train)
        F_test += alpha * best_stump.predict_raw(X_test)

        train_pred = sign_pm1(F_train)
        test_pred = sign_pm1(F_test)
        tr_acc = accuracy(y_train, train_pred)
        te_acc = accuracy(y_test, test_pred)

        rounds.append(AdaRound(best_stump, best_epsilon, alpha, Z_t, tr_acc, te_acc))

    return rounds, np.array(alphas), stumps_selected


def adaboost_predict(
    X: np.ndarray, selected: List[Stump], alphas: np.ndarray
) -> np.ndarray:
    F = np.zeros(X.shape[0])
    for stump, a in zip(selected, alphas):
        F += a * stump.predict_raw(X)
    return sign_pm1(F)


# ----------------------------
# Coordinate Descent (exp loss)
# ----------------------------


def coordinate_descent_exp_loss(
    X: np.ndarray,
    y: np.ndarray,
    stumps: List[Stump],
    max_passes: int = 50,
    tol: float = 1e-6,
    seed: int = 123,
) -> Tuple[np.ndarray, float, List[Tuple[int, float]]]:
    """
    Optimize exponential loss using coordinate descent on alpha weights.
    Loss: sum_i exp(-y_i * F(x_i)), where F(x) = sum_j alpha_j * h_j(x).
    """
    rng = np.random.default_rng(seed)
    m, nst = X.shape[0], len(stumps)
    # Precompute predictions matrix H (m x nst) with h_j(x_i) in {-1,+1}
    H = np.empty((m, nst), dtype=int)
    for j, stump in enumerate(stumps):
        H[:, j] = stump.predict_raw(X)

    # Initialize
    alphas = np.zeros(nst, dtype=float)
    F = np.zeros(m, dtype=float)

    for _ in range(max_passes):
        order = rng.permutation(nst)  # random order
        max_delta = 0.0

        for j in order:
            h_j = H[:, j]
            # Remove current contribution of j
            F_minus = F - alphas[j] * h_j
            w = np.exp(-y * F_minus)
            # Weighted error under w
            mism = (y != h_j).astype(float)
            epsilon_j = float(np.dot(w, mism)) / float(np.sum(w))
            alpha_star = safe_alpha_from_epsilon(epsilon_j)

            delta = alpha_star - alphas[j]
            if abs(delta) > 0:
                F += delta * h_j
                alphas[j] = alpha_star
                max_delta = max(max_delta, abs(delta))

        if max_delta < tol:
            break

    # Final exponential loss
    exp_loss = float(np.sum(np.exp(-y * F)))

    nonzeros = [(j, float(a)) for j, a in enumerate(alphas) if abs(a) > 1e-12]
    return alphas, exp_loss, nonzeros


# ----------------------------
# Bagging (20 bootstraps)
# ----------------------------


def fit_best_stump_unweighted(X: np.ndarray, y: np.ndarray) -> Stump:
    stumps = generate_all_stumps(X)
    best_eps = float("inf")
    best = None
    for s in stumps:
        pred = s.predict_raw(X)
        eps = float(np.mean(y != pred))
        if eps < best_eps:
            best_eps = eps
            best = s
    return best


def bagging_stumps(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    B: int = 20,
    seed: int = 0,
) -> Tuple[List[Stump], float, float]:
    """
    Train B decision stumps using bootstrap aggregating (bagging).
    Each stump is trained on a bootstrap sample (sampling with replacement).
    Final prediction is the majority vote across all B stumps.
    """
    rng = np.random.default_rng(seed)
    M = X_train.shape[0]
    models: List[Stump] = []
    for _ in range(B):
        idx = rng.integers(0, M, size=M)  # with replacement
        Xb = X_train[idx]
        yb = y_train[idx]
        stump = fit_best_stump_unweighted(Xb, yb)
        models.append(stump)

    def predict_majority(X: np.ndarray) -> np.ndarray:
        votes = np.zeros((X.shape[0],), dtype=float)
        for s in models:
            votes += s.predict_raw(X)
        return sign_pm1(votes)

    train_acc = accuracy(y_train, predict_majority(X_train))
    test_acc = accuracy(y_test, predict_majority(X_test))
    return models, train_acc, test_acc


# ----------------------------
# Plotting
# ----------------------------


def plot_adaboost_accuracies(
    rounds: List[AdaRound], out_path: str = "adaboost_accuracy.png"
) -> None:
    iters = np.arange(1, len(rounds) + 1)
    tr = [r.train_acc for r in rounds]
    te = [r.test_acc for r in rounds]
    plt.figure(figsize=(6, 4.5))
    plt.plot(iters, tr, marker="o", label="Train accuracy")
    plt.plot(iters, te, marker="s", label="Test accuracy")
    plt.xlabel("Iteration (t)")
    plt.ylabel("Accuracy")
    plt.title("AdaBoost Accuracy vs Iteration (Decision Stumps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[Plot] Saved AdaBoost accuracy curve to {out_path}")


# ----------------------------
# Main
# ----------------------------


def main():
    """
    Problem 3 implementation:
    (a) AdaBoost with decision stumps (10 rounds)
    (b) Accuracy plots vs iteration
    (c) Coordinate descent on exponential loss (different order/init vs AdaBoost)
    (d) Bagging with 20 bootstraps
    (e) Comparison and recommendation
    """
    # Load data
    Xtr, ytr = load_csv("heart_train.data")
    Xte, yte = load_csv("heart_test.data")
    M, n = Xtr.shape
    print(f"Loaded train: M={M}, n={n} | test: M={Xte.shape[0]}, n={Xte.shape[1]}")

    # Hypothesis space (all stumps)
    all_stumps = generate_all_stumps(Xtr)
    print(f"Hypothesis space size (stumps): {len(all_stumps)}")

    # ---- (a) AdaBoost, 10 rounds
    rounds, alphas, selected_stumps = adaboost_train(
        Xtr, ytr, Xte, yte, T=10, all_stumps=all_stumps
    )

    print(
        "\n=== (a) AdaBoost (10 rounds): selected stumps with epsilon, alpha, Z_t ==="
    )
    for i, r in enumerate(rounds, 1):
        print(f"[{i:02d}] stump (one-split): {r.stump}")
        print(ascii_tree(r.stump))
        print(
            f"     epsilon={r.epsilon:.6f}, alpha={r.alpha:.6f}, Z_t={r.Z_t:.6f}, "
            f"train_acc={r.train_acc:.4f}, test_acc={r.test_acc:.4f}"
        )

    # ---- (b) Plot accuracy vs iteration
    plot_adaboost_accuracies(rounds, out_path="adaboost_accuracy.png")

    # ---- (c) Coordinate descent on exponential loss
    cd_alphas, exp_loss, nonzeros = coordinate_descent_exp_loss(
        Xtr, ytr, all_stumps, max_passes=100, tol=1e-7, seed=2025
    )

    print("\n=== (c) Coordinate Descent: alpha_j and training exponential loss ===")
    print(f"Nonzero alpha count: {len(nonzeros)} (of {len(all_stumps)})")
    nonzeros_sorted = sorted(nonzeros, key=lambda t: abs(t[1]), reverse=True)
    for k, (j, a) in enumerate(nonzeros_sorted[:15], 1):
        print(f"  {k:02d}. alpha[{j}] = {a:.6f}   stump: {all_stumps[j]}")
    if len(nonzeros_sorted) > 15:
        print(f"  ... ({len(nonzeros_sorted) - 15} more)")
    print(f"Training exponential loss: {exp_loss:.6f}")

    def predict_with_cd_alphas(X: np.ndarray) -> np.ndarray:
        F = np.zeros(X.shape[0])
        for j, a in enumerate(cd_alphas):
            if abs(a) > 0:
                F += a * all_stumps[j].predict_raw(X)
        return sign_pm1(F)

    ytr_cd = predict_with_cd_alphas(Xtr)
    yte_cd = predict_with_cd_alphas(Xte)
    print(
        f"CD train_acc={accuracy(ytr, ytr_cd):.4f}, test_acc={accuracy(yte, yte_cd):.4f}"
    )

    # ---- (d) Bagging with 20 bootstraps (average classifier)
    bag_models, bag_tr_acc, bag_te_acc = bagging_stumps(
        Xtr, ytr, Xte, yte, B=20, seed=7
    )
    print("\n=== (d) Bagging (20 stumps): accuracy ===")
    print(f"Bagging train_acc={bag_tr_acc:.4f}, test_acc={bag_te_acc:.4f}")

    # ---- (e) Method preference summary
    ada_te_curve = [r.test_acc for r in rounds]
    best_ada_te = max(ada_te_curve)
    final_ada_te = ada_te_curve[-1]
    print("\n=== (e) Which method to prefer & why ===")
    print(
        f"Final AdaBoost test acc (t=10): {final_ada_te:.4f} | Best AdaBoost test acc: {best_ada_te:.4f}"
    )
    print(f"Coordinate Descent test acc:    {accuracy(yte, yte_cd):.4f}")
    print(f"Bagging test acc (B=20):        {bag_te_acc:.4f}")
    print(
        "Recommendation: Prefer the method with the highest and most stable test accuracy on this split. "
        "AdaBoost often performs best when each stump is slightly better than chance; "
        "bagging reduces variance and can be more robust to noise; "
        "coordinate descent directly minimizes exponential loss but may overfit if many alphas become nonzero."
    )


if __name__ == "__main__":
    main()
