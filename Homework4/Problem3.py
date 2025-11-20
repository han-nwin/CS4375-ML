import numpy as np


def load_data():
    """Load and preprocess the leaf dataset"""
    data = np.loadtxt("leaf.data", delimiter=",")

    # First column is the label
    labels = data[:, 0]
    X = data[:, 1:]

    # Preprocess: standardize to mean zero and variance one
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    X_standardized = (X - mean) / std

    return X_standardized, labels


def kmeans_objective(X, centroids, assignments):
    """
    Compute k-means objective (sum of squared distances)
    """
    total = 0
    for i, x in enumerate(X):
        total += np.sum((x - centroids[assignments[i]]) ** 2)
    return total


def kmeans(X, k, max_iters=100):
    """
    K-means algorithm with random initialization

    Parameters:
    X: data matrix (n_samples, n_features)
    k: number of clusters
    max_iters: maximum number of iterations

    Returns:
    centroids: final cluster centers
    assignments: cluster assignment for each point
    objective: final k-means objective value
    """
    n_samples = X.shape[0]

    # Random initialization: select k points uniformly at random with replacement
    indices = np.random.choice(n_samples, size=k, replace=True)
    centroids = X[indices].copy()

    for _ in range(max_iters):
        # Assignment step: assign each point to nearest centroid
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sum((X - centroids[j]) ** 2, axis=1)
        assignments = np.argmin(distances, axis=1)

        # Update step: recompute centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            points_in_cluster = X[assignments == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = np.mean(points_in_cluster, axis=0)
            else:
                # Empty cluster: reinitialize randomly
                new_centroids[j] = X[np.random.randint(n_samples)]

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Compute final objective
    objective = kmeans_objective(X, centroids, assignments)

    return centroids, assignments, objective


def kmeans_plusplus(X, k, max_iters=100):
    """
    K-means++ algorithm with improved initialization

    Parameters:
    X: data matrix (n_samples, n_features)
    k: number of clusters
    max_iters: maximum number of iterations

    Returns:
    centroids: final cluster centers
    assignments: cluster assignment for each point
    objective: final k-means objective value
    """
    n_samples = X.shape[0]

    # K-means++ initialization
    # Step 1: Choose first center uniformly at random
    first_idx = np.random.randint(n_samples)
    centroids = [X[first_idx].copy()]

    # Step 2: Choose remaining centers
    for _ in range(1, k):
        # Compute distance to nearest center for each point
        distances = np.full(n_samples, np.inf)
        for c in centroids:
            dist = np.sum((X - c) ** 2, axis=1)
            distances = np.minimum(distances, dist)

        # Sample proportional to d_x^2
        probs = distances / np.sum(distances)
        new_idx = np.random.choice(n_samples, p=probs)
        centroids.append(X[new_idx].copy())

    centroids = np.array(centroids)

    # Run standard k-means with this initialization
    for _ in range(max_iters):
        # Assignment step
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sum((X - centroids[j]) ** 2, axis=1)
        assignments = np.argmin(distances, axis=1)

        # Update step
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            points_in_cluster = X[assignments == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[j] = X[np.random.randint(n_samples)]

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Compute final objective
    objective = kmeans_objective(X, centroids, assignments)

    return centroids, assignments, objective


def evaluate_clustering(labels, assignments):
    """
    Evaluate clustering quality using purity
    """
    clusters = np.unique(assignments)
    total_correct = 0

    for cluster in clusters:
        cluster_mask = assignments == cluster
        cluster_labels = labels[cluster_mask]
        if len(cluster_labels) > 0:
            # Count most common label in cluster
            values, counts = np.unique(cluster_labels, return_counts=True)
            total_correct += counts.max()

    purity = total_correct / len(labels)
    return purity


def main():
    """Main function"""
    print("=" * 80)
    print("PROBLEM 3: K-Means++")
    print("=" * 80)

    # Load data
    print("\nLoading and preprocessing data...")
    X, labels = load_data()
    print(f"Data shape: {X.shape}")
    print(f"Number of unique labels: {len(np.unique(labels))}")

    # K values to test
    k_values = [10, 20, 30, 36, 40]
    n_runs = 100

    # Store results
    results_kmeans = {}
    results_kmeanspp = {}

    for k in k_values:
        print(f"\n{'-'*80}")
        print(f"Testing k = {k}")
        print(f"{'-'*80}")

        # Part (a): Standard k-means with random initialization
        print(f"\n(a) Running standard k-means {n_runs} times...")
        objectives_kmeans = []
        purities_kmeans = []

        for i in range(n_runs):
            _, assignments, objective = kmeans(X, k)
            objectives_kmeans.append(objective)
            purity = evaluate_clustering(labels, assignments)
            purities_kmeans.append(purity)
            if (i + 1) % 25 == 0:
                print(f"    Completed {i + 1}/{n_runs} runs")

        mean_obj_kmeans = np.mean(objectives_kmeans)
        std_obj_kmeans = np.std(objectives_kmeans)
        mean_purity_kmeans = np.mean(purities_kmeans)

        results_kmeans[k] = {
            "mean_obj": mean_obj_kmeans,
            "std_obj": std_obj_kmeans,
            "mean_purity": mean_purity_kmeans,
        }

        print(f"    K-means objective: {mean_obj_kmeans:.2f} ± {std_obj_kmeans:.2f}")
        print(f"    Average purity: {mean_purity_kmeans*100:.2f}%")

        # Part (b): K-means++ initialization
        print(f"\n(b) Running k-means++ {n_runs} times...")
        objectives_kmeanspp = []
        purities_kmeanspp = []

        for i in range(n_runs):
            _, assignments, objective = kmeans_plusplus(X, k)
            objectives_kmeanspp.append(objective)
            purity = evaluate_clustering(labels, assignments)
            purities_kmeanspp.append(purity)
            if (i + 1) % 25 == 0:
                print(f"    Completed {i + 1}/{n_runs} runs")

        mean_obj_kmeanspp = np.mean(objectives_kmeanspp)
        std_obj_kmeanspp = np.std(objectives_kmeanspp)
        mean_purity_kmeanspp = np.mean(purities_kmeanspp)

        results_kmeanspp[k] = {
            "mean_obj": mean_obj_kmeanspp,
            "std_obj": std_obj_kmeanspp,
            "mean_purity": mean_purity_kmeanspp,
        }

        print(
            f"    K-means++ objective: {mean_obj_kmeanspp:.2f} ± {std_obj_kmeanspp:.2f}"
        )
        print(f"    Average purity: {mean_purity_kmeanspp*100:.2f}%")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("Part 1(a): Standard K-means (random initialization)")
    print("-" * 80)
    print(f"{'k':<8} {'Mean Objective':<20} {'Std Objective':<20} {'Mean Purity (%)':<15}")
    print("-" * 80)
    for k in k_values:
        res = results_kmeans[k]
        print(
            f"{k:<8} {res['mean_obj']:<20.2f} {res['std_obj']:<20.2f} {res['mean_purity']*100:<15.2f}"
        )

    print("\n" + "-" * 80)
    print("Part 1(b): K-means++ initialization")
    print("-" * 80)
    print(f"{'k':<8} {'Mean Objective':<20} {'Std Objective':<20} {'Mean Purity (%)':<15}")
    print("-" * 80)
    for k in k_values:
        res = results_kmeanspp[k]
        print(
            f"{k:<8} {res['mean_obj']:<20.2f} {res['std_obj']:<20.2f} {res['mean_purity']*100:<15.2f}"
        )

    print("\n" + "-" * 80)
    print("Comparison: K-means vs K-means++")
    print("-" * 80)
    print(
        f"{'k':<8} {'KM Obj':<15} {'KM++ Obj':<15} {'Improv (%)':<12} {'KM Pur (%)':<12} {'KM++ Pur (%)':<12}"
    )
    print("-" * 80)
    for k in k_values:
        km = results_kmeans[k]
        kmpp = results_kmeanspp[k]
        improvement = (km["mean_obj"] - kmpp["mean_obj"]) / km["mean_obj"] * 100
        print(
            f"{k:<8} {km['mean_obj']:<15.2f} {kmpp['mean_obj']:<15.2f} {improvement:<12.2f} {km['mean_purity']*100:<12.2f} {kmpp['mean_purity']*100:<12.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
