import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


# ----------------------------
# I/O
# ----------------------------
def read_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
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
    # Map to {-1, +1}
    u = set(np.unique(y_raw))
    if u == {0.0, 1.0}:
        y = np.where(y_raw > 0, 1, -1).astype(np.int32)
    elif u == {-1.0, 1.0}:
        y = y_raw.astype(np.int32)
    else:
        y = np.where(y_raw > 0, 1, -1).astype(np.int32)
    return X, y


# ----------------------------
# Stumps
# ----------------------------
@dataclass
class Stump:
    feature: int
    threshold: float
    polarity: int  # +1 or -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        left = (X[:, self.feature] <= self.threshold).astype(
            np.int32
        ) * 2 - 1  # {-1,+1}
        return np.sign(self.polarity * left).astype(np.int32)

    def __str__(self) -> str:
        lp = 1 if self.polarity == 1 else -1
        rp = -lp
        return f"x[{self.feature}] <= {self.threshold:.1f} -> {lp}, else {rp}"


def unique_thresholds(col: np.ndarray) -> List[float]:
    vals = np.unique(col)  # sorted unique values
    if len(vals) == 1:
        return [float(vals[0])]
    mids = ((vals[:-1] + vals[1:]) / 2.0).astype(float)
    return list(mids)


def generate_all_stumps(X: np.ndarray) -> List[Stump]:
    M, n = X.shape
    stumps: List[Stump] = []
    for j in range(n):
        for thr in unique_thresholds(X[:, j]):
            stumps.append(Stump(j, thr, +1))
            stumps.append(Stump(j, thr, -1))
    return stumps


# ----------------------------
# Exponential loss & CD
# ----------------------------
def safe_alpha_from_eps(eps: float, floor: float = 1e-12) -> float:
    eps = min(max(eps, floor), 1.0 - floor)
    return 0.5 * math.log((1.0 - eps) / eps)


def exp_loss(y: np.ndarray, F: np.ndarray) -> float:
    return float(np.sum(np.exp(-y * F)))


def coordinate_descent_exp(
    X: np.ndarray,
    y: np.ndarray,
    stumps: List[Stump],
    max_passes: int = 100,
    tol: float = 1e-7,
    seed: int = 2025,
) -> Tuple[np.ndarray, float]:
    """
    Returns:
        alphas: (S,) alpha_j for each stump j
        loss:   final exponential loss on training set
    """
    rng = np.random.default_rng(seed)
    M = X.shape[0]
    S = len(stumps)

    # Precompute H_{ij} = h_j(x_i) in {-1,+1}
    H = np.empty((M, S), dtype=np.int8)
    for j, s in enumerate(stumps):
        H[:, j] = s.predict(X)

    alphas = np.zeros(S, dtype=float)
    F = np.zeros(M, dtype=float)

    for _ in range(max_passes):
        # Shuffle order of stumps
        order = np.arange(S)
        rng.shuffle(order)
        max_delta = 0.0

        for j in order:
            hj = H[:, j].astype(int)

            # Remove current contribution of coordinate j
            F_minus = F - alphas[j] * hj

            # Weights for exp loss along this line
            w = np.exp(-y * F_minus)
            W = float(np.sum(w)) if w.size else 1.0

            # Weighted error of stump j under w
            wrong_mask = hj != y
            eps_j = float(np.sum(w[wrong_mask]) / W)

            # Closed-form optimal alpha_j along this coordinate
            alpha_star = safe_alpha_from_eps(eps_j)
            delta = alpha_star - alphas[j]

            if delta != 0.0:
                F = F_minus + alpha_star * hj
                alphas[j] = alpha_star
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

        if max_delta < tol:
            break

    return alphas, exp_loss(y, F)


# ----------------------------
# Eval helpers
# ----------------------------


def predict_sign_with_alphas(
    X: np.ndarray, stumps: List[Stump], alphas: np.ndarray
) -> np.ndarray:
    M = X.shape[0]
    F = np.zeros(M, dtype=float)
    for j, a in enumerate(alphas):
        if a == 0.0:
            continue
        F += a * stumps[j].predict(X)
    return np.where(F >= 0.0, 1, -1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# ----------------------------
# Main
# ----------------------------


def main():
    Xtr, ytr = read_xy("heart_train.data")
    try:
        Xte, yte = read_xy("heart_test.data")
    except Exception:
        Xte, yte = None, None

    stumps = generate_all_stumps(Xtr)
    print(f"Hypothesis space size (stumps): {len(stumps)}")

    # Coordinate descent with randomized coordinate order
    # Stop when tolerance is reached or max_passes is reached
    alphas, L = coordinate_descent_exp(
        Xtr, ytr, stumps, max_passes=1000, tol=1e-10, seed=2025
    )

    # Report non-zero alphas (sorted by |alpha|)
    nz_idx = np.flatnonzero(np.abs(alphas) > 1e-12)
    sorted_idx = nz_idx[np.argsort(-np.abs(alphas[nz_idx]))]

    print("\n=== Coordinate Descent ===")
    print(f"Non-zero alpha count: {len(nz_idx)} (of {len(stumps)})")

    def pretty_rule_for_binary(s: Stump, X: np.ndarray) -> str:
        # If the feature is binary and threshold is ~0.5, print as 0/1 mapping
        vals = np.unique(X[:, s.feature])
        if set(vals.tolist()).issubset({0.0, 1.0}) and abs(s.threshold - 0.5) < 1e-9:
            if s.polarity == +1:
                # x<=0.5 (x=0) -> +1, x=1 -> -1
                return f"feature x[{s.feature}]: (0 -> +1, 1 -> -1)"
            else:
                # x<=0.5 (x=0) -> -1, x=1 -> +1
                return f"feature x[{s.feature}]: (0 -> -1, 1 -> +1)"
        # fallback to generic threshold form
        return str(s)

    # sort by |alpha| (importance), keep rank
    for rank, j in enumerate(sorted_idx, 1):
        s = stumps[j]
        nice = pretty_rule_for_binary(s, Xtr)  # pass training X to detect binary
        print(
            f"[rank {rank:02d}] stump[{j:02d}] -> alpha[{j}] = {alphas[j]:.6f} | {nice}"
        )

    print(f"\nTraining exponential loss: {L:.6f}")

    # 0-1 accuracy of sign(sum_j alpha_j h_j(x))
    yhat_tr = predict_sign_with_alphas(Xtr, stumps, alphas)
    tr_acc = accuracy(ytr, yhat_tr)
    if Xte is not None and yte is not None:
        yhat_te = predict_sign_with_alphas(Xte, stumps, alphas)
        te_acc = accuracy(yte, yhat_te)
        print(
            f"Train acc (sign(F)): {tr_acc*100:.2f}%   |   Test acc (sign(F)): {te_acc*100:.2f}%"
        )
    else:
        print(f"Train acc (sign(F)): {tr_acc*100:.2f}%")


if __name__ == "__main__":
    main()
