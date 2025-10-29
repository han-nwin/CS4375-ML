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
        return f"x[{self.feature}] <= {self.threshold:.4g} -> {lp}, else {rp}"


def unique_thresholds(col: np.ndarray) -> List[float]:
    vals = np.unique(col)
    if len(vals) == 1:
        return [float(vals[0])]
    mids = ((vals[:-1] + vals[1:]) / 2.0).astype(float)
    return list(mids)


def generate_all_stumps(X: np.ndarray) -> List[Stump]:
    _, n = X.shape
    stumps: List[Stump] = []
    for j in range(n):
        for thr in unique_thresholds(X[:, j]):
            stumps.append(Stump(j, thr, +1))
            stumps.append(Stump(j, thr, -1))
    return stumps


# ----------------------------
# Helpers
# ----------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def miscls_rate(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(y != yhat))


def best_stump_on_subset(
    X: np.ndarray, y: np.ndarray, stumps: List[Stump], idx: np.ndarray
) -> Tuple[Stump, float]:
    # Find stump with lowest (unweighted) error on bootstrap sample idx
    Xb = X[idx]
    yb = y[idx]
    best_err = float("inf")
    best = None
    for s in stumps:
        pred = s.predict(Xb)
        err = miscls_rate(yb, pred)
        if err < best_err:
            best_err = err
            best = s
    return best, best_err


def vote_ensemble(stumps: List[Stump], X: np.ndarray) -> np.ndarray:
    # average votes (equal weights), tie => +1
    if not stumps:
        return np.ones(X.shape[0], dtype=np.int32)
    S = np.zeros(X.shape[0], dtype=float)
    for s in stumps:
        S += s.predict(X)
    return np.where(S >= 0.0, 1, -1)


# ----------------------------
# Main (Bagging with 20 bootstraps)
# ----------------------------
def main():
    Xtr, ytr = read_xy("heart_train.data")
    try:
        Xte, yte = read_xy("heart_test.data")
    except Exception:
        Xte, yte = None, None

    all_stumps = generate_all_stumps(Xtr)
    B = 20  # number of bootstrap samples
    rng = np.random.default_rng(2025)

    chosen: List[Tuple[int, Stump, float]] = []  # (b, stump, train_err_on_bootstrap)

    M = Xtr.shape[0]
    print(
        f"\n=== Train: {M} samples, {Xtr.shape[1]} features, {len(all_stumps)} stumps | Bagging {B} times ==="
    )

    for b in range(B):
        # draw bootstrap indices with replacement
        idx = rng.integers(low=0, high=M, size=M, endpoint=False)
        unique_count = len(np.unique(idx))

        s, err = best_stump_on_subset(Xtr, ytr, all_stumps, idx)
        chosen.append((b, s, err))

        # Show ensemble training performance so far
        ensemble_so_far = [stump for (_, stump, _) in chosen]
        yhat_tr_so_far = vote_ensemble(ensemble_so_far, Xtr)

        print(
            f"B{b+1:2d} | unique:{unique_count:3d}/{M} | Selected Stump: {s} | err:{err*100:5.2f}% "
        )

    # Final evaluation
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    ensemble = [s for (_, s, _) in chosen]
    yhat_tr = vote_ensemble(ensemble, Xtr)
    tr_acc = accuracy(ytr, yhat_tr)
    if Xte is not None:
        yhat_te = vote_ensemble(ensemble, Xte)
        te_acc = accuracy(yte, yhat_te)
        print(
            f"Bagging-{B} | Train acc: {tr_acc*100:.2f}% | Test acc: {te_acc*100:.2f}%"
        )
    else:
        print(f"Bagging-{B} | Train acc: {tr_acc*100:.2f}% | (no test set provided)")


if __name__ == "__main__":
    main()
