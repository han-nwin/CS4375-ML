import sys
import math
import random
from typing import List, Tuple


def read_xy(path: str) -> Tuple[List[int], List[List[float]]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [
                p.strip()
                for p in (ln.split(",") if "," in ln else ln.split())
                if p.strip() != ""
            ]
            y = int(float(parts[0]))
            x = [float(v) for v in parts[1:]]
            rows.append((y, x))
    if not rows:
        raise ValueError(f"No data in {path}")
    y = [r[0] for r in rows]
    X = [r[1] for r in rows]
    return y, X


def compute_mean_std(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    n = len(X)
    d = len(X[0])
    mean = [0.0] * d
    for xi in X:
        for j, v in enumerate(xi):
            mean[j] += v
    mean = [m / n for m in mean]
    var = [0.0] * d
    for xi in X:
        for j, v in enumerate(xi):
            dv = v - mean[j]
            var[j] += dv * dv
    var = [v / n for v in var]
    std = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    return mean, std


def standardize(
    X: List[List[float]], mean: List[float], std: List[float]
) -> List[List[float]]:
    return [[(v - m) / s for v, m, s in zip(xi, mean, std)] for xi in X]


class PegasosSVM:
    def __init__(self, C: float, epochs: int = 5, seed: int = 42):
        self.C = C
        self.epochs = epochs
        self.seed = seed
        self.w: List[float] | None = None  # last coordinate is bias

    def fit(self, X: List[List[float]], y: List[int]):
        random.seed(self.seed)
        n, d = len(X), len(X[0])
        lam = 1.0 / (n * self.C) if self.C > 0 else 1e-6
        Xa = [xi + [1.0] for xi in X]
        self.w = [0.0] * (d + 1)
        t = 0
        for ep in range(self.epochs):
            idxs = list(range(n))
            random.shuffle(idxs)
            for i in idxs:
                t += 1
                eta = 1.0 / (lam * t)
                xi = Xa[i]
                yi = y[i]
                margin = yi * sum(self.w[j] * xi[j] for j in range(d + 1))
                if margin < 1.0:
                    # Regularize non-bias weights; update with gradient on hinge
                    for j in range(d):
                        self.w[j] = (1 - eta * lam) * self.w[j] + eta * yi * xi[j]
                    self.w[d] = self.w[d] + eta * yi  # bias (no reg)
                else:
                    for j in range(d):
                        self.w[j] = (1 - eta * lam) * self.w[j]
                # Project non-bias portion to L2 ball of radius 1/sqrt(lam)
                nb_norm = math.sqrt(sum(wj * wj for wj in self.w[:d]))
                radius = 1.0 / math.sqrt(lam)
                if nb_norm > radius and nb_norm > 0:
                    scale = radius / nb_norm
                    for j in range(d):
                        self.w[j] *= scale

    def predict(self, X: List[List[float]]) -> List[int]:
        d = len(X[0])
        b = self.w[d]
        preds = []
        for xi in X:
            s = sum(self.w[j] * xi[j] for j in range(d)) + b
            preds.append(1 if s >= 0 else -1)
        return preds


def contiguous_kfold_indices(n: int, k: int) -> List[Tuple[int, int]]:
    # Generate start and end indices for k contiguous folds for cross-validation
    base = n // k
    rem = n % k
    blocks = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        blocks.append((start, end))
        start = end
    return blocks


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def main():
    y_tr, X_tr = read_xy("wdbc_train.data")
    y_te, X_te = read_xy("wdbc_test.data")

    M = len(X_tr)
    k = 10
    folds = contiguous_kfold_indices(M, k)

    # C values to tune
    C_grid = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100, 300, 1000]

    # Train each bucket of data with different C values
    epochs_cv = 7
    epochs_final = 25

    results: list[tuple[float, float]] = []
    for C in C_grid:
        accs = []
        for fi, (s, e) in enumerate(folds):
            X_val = X_tr[s:e]
            y_val = y_tr[s:e]
            X_train_fold = X_tr[:s] + X_tr[e:]
            y_train_fold = y_tr[:s] + y_tr[e:]

            mu, sigma = compute_mean_std(X_train_fold)
            Ztr = standardize(X_train_fold, mu, sigma)
            Zva = standardize(X_val, mu, sigma)

            clf = PegasosSVM(C=C, epochs=epochs_cv, seed=123 + fi)
            clf.fit(Ztr, y_train_fold)
            yhat = clf.predict(Zva)
            accs.append(accuracy(y_val, yhat))
        avg = sum(accs) / len(accs)
        results.append((C, avg))

    results.sort(key=lambda t: (-t[1], t[0]))
    best_C, best_cv = results[0]

    mu_full, sigma_full = compute_mean_std(X_tr)
    Ztr_full = standardize(X_tr, mu_full, sigma_full)
    Zte = standardize(X_te, mu_full, sigma_full)

    # Retrain final model with best C
    final = PegasosSVM(C=best_C, epochs=epochs_final, seed=777)
    final.fit(Ztr_full, y_tr)
    test_acc = accuracy(y_te, final.predict(Zte))

    print("=== 10-Fold Cross-Validation ===")
    for C, avg in results:
        print(f"C = {C:<6} CV-avg-acc={avg*100:6.2f}%")
    print("\n=== Selected hyperparameters ===")
    print(f"Best C: {best_C} (CV avg acc = {best_cv*100:.2f}%)")
    print("\n=== Test set performance ===")
    print(f"Accuracy: {test_acc*100:.2f}% on {len(X_te)} examples")


if __name__ == "__main__":
    main()
