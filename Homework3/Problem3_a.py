import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


# ----------------------------
# I/O (robust to CSV or space)
# ----------------------------


def read_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads rows where the first field is the label and the rest are features.
    Accepts comma-separated or whitespace-separated lines.
    Returns:
        X: (M, n) float64
        y: (M,) in {-1,+1} int32
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [p for p in (ln.split(",") if "," in ln else ln.split()) if p != ""]
            y = float(parts[0])
            x = [float(v) for v in parts[1:]]
            rows.append((y, x))
    if not rows:
        raise ValueError(f"No data in {path}")
    y_raw = np.array([r[0] for r in rows], dtype=float)
    X = np.array([r[1] for r in rows], dtype=float)
    # Map labels to {-1,+1}
    u = set(np.unique(y_raw))
    if u == {0.0, 1.0}:
        y = np.where(y_raw > 0, 1, -1).astype(np.int32)
    elif u == {-1.0, 1.0}:
        y = y_raw.astype(np.int32)
    else:
        y = np.where(y_raw > 0, 1, -1).astype(np.int32)

    return X, y


# ----------------------------
# Decision stump model
# ----------------------------
@dataclass
class Stump:
    feature: int
    threshold: float
    polarity: int  # +1 or -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        # left branch -> +1 (if polarity=+1) else -1; right branch -> opposite
        left = (X[:, self.feature] <= self.threshold).astype(
            np.int32
        ) * 2 - 1  # in {-1,+1}
        return np.sign(self.polarity * left).astype(np.int32)

    def __str__(self) -> str:
        left_pred = 1 if self.polarity == 1 else -1
        right_pred = -left_pred
        return f"if x[{self.feature}] <= {self.threshold:.6g}: predict {left_pred} else: predict {right_pred}"

    def tree_str(self) -> str:
        """Returns a visual tree representation of the stump."""
        left_pred = 1 if self.polarity == 1 else -1
        right_pred = -left_pred
        tree = f"x[{self.feature}] <= {self.threshold:.4g}\n   /     \\\n [{left_pred:+d}]     [{right_pred:+d}]"
        return tree


def unique_thresholds(col: np.ndarray) -> List[float]:
    """Midpoints between sorted unique values (binary 0/1 -> [0.5])."""
    vals = np.unique(col)
    if len(vals) == 1:
        return [float(vals[0])]
    mids = ((vals[:-1] + vals[1:]) / 2.0).astype(float)
    return list(mids)


def generate_all_stumps(X: np.ndarray) -> List[Stump]:
    """All stumps over all features, both polarities, all midpoints."""
    M, n = X.shape
    stumps: List[Stump] = []
    for j in range(n):
        for thr in unique_thresholds(X[:, j]):
            stumps.append(Stump(j, thr, +1))
            stumps.append(Stump(j, thr, -1))
    return stumps


# ----------------------------
# AdaBoost machinery
# ----------------------------


def weighted_error(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    mism = (y != yhat).astype(float)
    return float(np.dot(w, mism) / np.sum(w))


def safe_alpha_from_eps(eps: float, floor: float = 1e-12) -> float:
    eps = min(max(eps, floor), 1.0 - floor)
    return 0.5 * math.log((1.0 - eps) / eps)


@dataclass
class AdaRound:
    stump: Stump
    eps: float
    alpha: float


def adaboost_10_rounds(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: Optional[np.ndarray] = None,
    yte: Optional[np.ndarray] = None,
) -> Tuple[List[AdaRound], np.ndarray, np.ndarray]:
    """
    Return:
        rounds: list of 10 (stump, eps, alpha)
        train_acc_curve: (10,)
        test_acc_curve: (10,) (zeros if no test set given)
    """
    all_stumps = generate_all_stumps(Xtr)
    M = Xtr.shape[0]
    # Start weights at 1/M
    w = np.ones(M) / M
    Ftr = np.zeros(M)
    train_acc_curve = np.zeros(10, dtype=float)
    test_acc_curve = (
        np.zeros(10, dtype=float) if Xte is not None else np.zeros(10, dtype=float)
    )
    rounds: List[AdaRound] = []

    Fte = np.zeros(Xte.shape[0]) if Xte is not None else None

    for t in range(10):
        best_eps = float("inf")
        best_stump = None
        best_pred = None

        # choose the best stump by minimizing the weighted error
        for s in all_stumps:
            pred = s.predict(Xtr)
            eps = weighted_error(ytr, pred, w)
            if eps < best_eps:
                best_eps = eps
                best_stump = s
                best_pred = pred

        alpha = safe_alpha_from_eps(best_eps)
        # weight update
        w *= np.exp(-alpha * ytr * best_pred)
        w /= np.sum(w)

        # track
        rounds.append(AdaRound(best_stump, best_eps, alpha))
        Ftr += alpha * best_stump.predict(Xtr)
        train_acc_curve[t] = np.mean(np.sign(Ftr) == ytr)

        if Xte is not None:
            Fte += alpha * best_stump.predict(Xte)  # type: ignore
            test_acc_curve[t] = np.mean(np.sign(Fte) == yte)  # type: ignore

    return rounds, train_acc_curve, test_acc_curve


# ----------------------------
# Main
# ----------------------------
def main():
    Xtr, ytr = read_xy("heart_train.data")

    try:
        Xte, yte = read_xy("heart_test.data")
    except Exception:
        Xte, yte = None, None

    rounds, tr_curve, te_curve = adaboost_10_rounds(Xtr, ytr, Xte, yte)

    print("\n=== Problem 3(a): AdaBoost (10 rounds) ===")
    for i, r in enumerate(rounds, 1):
        print(f"\n--- Round {i} ---")
        print(r.stump.tree_str())
        print(f"epsilon = {r.eps:.6f},  alpha = {r.alpha:.6f}")
        tr = float(tr_curve[i - 1])
        te = float(te_curve[i - 1]) if te_curve.size > 0 else float("nan")
        if te_curve.size > 0:
            print(f"Train acc: {tr*100:.2f}%,  Test acc: {te*100:.2f}%")
        else:
            print(f"Train acc: {tr*100:.2f}%")

    print()
    if te_curve.size:
        print(
            f"Final train acc (t=10): {tr_curve[-1]*100:.2f}% | Final test acc (t=10): {te_curve[-1]*100:.2f}%"
        )
    else:
        print(
            f"Final train acc (t=10): {tr_curve[-1]*100:.2f}% | No test file provided."
        )


if __name__ == "__main__":
    main()
