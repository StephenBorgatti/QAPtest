#!/usr/bin/env python3
"""
Diagnose WHY independent degree heterogeneity introduces correlation.

Hypothesis: The preferential attachment algorithm always starts with nodes 0,1,2
connected, giving them a "head start" in BOTH X and Y. This creates spurious
correlation because the same nodes are systematically high-degree in both networks.

Test this by:
1. Checking correlation of node degrees between X and Y
2. Randomizing the initial nodes to break this correlation
"""

import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def vectorize_symmetric(mat: np.ndarray) -> np.ndarray:
    """Vectorize symmetric matrix (upper triangle only)."""
    return mat[np.triu_indices(mat.shape[0], k=1)]


def generate_degree_het_FIXED_START(n: int, density: float = 0.15) -> np.ndarray:
    """
    Original version: Always starts with nodes 0,1,2 connected.
    """
    n_edges = int(round(n * (n - 1) / 2 * density))
    adj = np.zeros((n, n), dtype=int)

    # Always start with 0,1,2
    adj[0, 1] = adj[1, 0] = 1
    adj[1, 2] = adj[2, 1] = 1
    adj[0, 2] = adj[2, 0] = 1

    degrees = np.sum(adj, axis=1)
    edges_added = 3
    max_attempts = n_edges * 100
    attempts = 0

    while edges_added < n_edges and attempts < max_attempts:
        attempts += 1
        probs = (degrees + 1.0) ** 2
        probs = probs / np.sum(probs)
        i = np.random.choice(n, p=probs)

        probs2 = (degrees + 1.0) * (1 - adj[i, :])
        probs2[i] = 0

        if np.sum(probs2) > 0:
            probs2 = probs2 / np.sum(probs2)
            j = np.random.choice(n, p=probs2)

            if adj[i, j] == 0:
                adj[i, j] = adj[j, i] = 1
                degrees = np.sum(adj, axis=1)
                edges_added += 1

    # Fill remaining edges randomly if needed
    if edges_added < n_edges:
        zeros = np.argwhere((adj == 0) & np.triu(np.ones_like(adj, dtype=bool), k=1))
        if len(zeros) > 0:
            remaining = n_edges - edges_added
            selected = np.random.choice(len(zeros), size=min(remaining, len(zeros)), replace=False)
            for idx in selected:
                i, j = zeros[idx]
                adj[i, j] = adj[j, i] = 1

    return adj


def generate_degree_het_RANDOM_START(n: int, density: float = 0.15) -> np.ndarray:
    """
    Fixed version: Randomizes which 3 nodes start connected.
    """
    n_edges = int(round(n * (n - 1) / 2 * density))
    adj = np.zeros((n, n), dtype=int)

    # RANDOMLY pick 3 starting nodes
    start_nodes = np.random.choice(n, size=3, replace=False)
    a, b, c = start_nodes
    adj[a, b] = adj[b, a] = 1
    adj[b, c] = adj[c, b] = 1
    adj[a, c] = adj[c, a] = 1

    degrees = np.sum(adj, axis=1)
    edges_added = 3
    max_attempts = n_edges * 100
    attempts = 0

    while edges_added < n_edges and attempts < max_attempts:
        attempts += 1
        probs = (degrees + 1.0) ** 2
        probs = probs / np.sum(probs)
        i = np.random.choice(n, p=probs)

        probs2 = (degrees + 1.0) * (1 - adj[i, :])
        probs2[i] = 0

        if np.sum(probs2) > 0:
            probs2 = probs2 / np.sum(probs2)
            j = np.random.choice(n, p=probs2)

            if adj[i, j] == 0:
                adj[i, j] = adj[j, i] = 1
                degrees = np.sum(adj, axis=1)
                edges_added += 1

    # Fill remaining edges randomly if needed
    if edges_added < n_edges:
        zeros = np.argwhere((adj == 0) & np.triu(np.ones_like(adj, dtype=bool), k=1))
        if len(zeros) > 0:
            remaining = n_edges - edges_added
            selected = np.random.choice(len(zeros), size=min(remaining, len(zeros)), replace=False)
            for idx in selected:
                i, j = zeros[idx]
                adj[i, j] = adj[j, i] = 1

    return adj


def run_diagnosis(n_simulations: int = 500, n_nodes: int = 50, density: float = 0.15, seed: int = 42):
    """Compare fixed-start vs random-start preferential attachment."""
    np.random.seed(seed)

    print("=" * 70)
    print("DIAGNOSIS: Why does independent degree heterogeneity introduce correlation?")
    print("=" * 70)
    print()

    # Part 1: Check degree correlation between X and Y (fixed start)
    print("PART 1: Checking node degree correlation (FIXED start: nodes 0,1,2)")
    print("-" * 70)

    degree_corrs_fixed = []
    matrix_corrs_fixed = []

    for i in range(n_simulations):
        X = generate_degree_het_FIXED_START(n_nodes, density)
        Y = generate_degree_het_FIXED_START(n_nodes, density)

        # Correlation of node degrees
        deg_X = np.sum(X, axis=1)
        deg_Y = np.sum(Y, axis=1)
        deg_corr, _ = stats.pearsonr(deg_X, deg_Y)
        degree_corrs_fixed.append(deg_corr)

        # Correlation of edge vectors
        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)
        mat_corr, _ = stats.pearsonr(x_vec, y_vec)
        matrix_corrs_fixed.append(mat_corr)

    print(f"Degree correlation (X vs Y): mean = {np.mean(degree_corrs_fixed):.4f}, std = {np.std(degree_corrs_fixed):.4f}")
    print(f"Matrix correlation (X vs Y): mean = {np.mean(matrix_corrs_fixed):.4f}, std = {np.std(matrix_corrs_fixed):.4f}")
    print()

    # Check which nodes tend to be high degree
    print("Checking average degree rank for each node position:")
    avg_degrees = np.zeros(n_nodes)
    for _ in range(100):
        X = generate_degree_het_FIXED_START(n_nodes, density)
        deg = np.sum(X, axis=1)
        avg_degrees += deg
    avg_degrees /= 100

    top_5 = np.argsort(avg_degrees)[-5:][::-1]
    print(f"Top 5 highest-degree node positions: {top_5}")
    print(f"Their average degrees: {avg_degrees[top_5]}")
    print()

    # Part 2: Random start
    print("PART 2: Checking with RANDOM starting nodes")
    print("-" * 70)

    degree_corrs_random = []
    matrix_corrs_random = []

    for i in range(n_simulations):
        X = generate_degree_het_RANDOM_START(n_nodes, density)
        Y = generate_degree_het_RANDOM_START(n_nodes, density)

        deg_X = np.sum(X, axis=1)
        deg_Y = np.sum(Y, axis=1)
        deg_corr, _ = stats.pearsonr(deg_X, deg_Y)
        degree_corrs_random.append(deg_corr)

        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)
        mat_corr, _ = stats.pearsonr(x_vec, y_vec)
        matrix_corrs_random.append(mat_corr)

    print(f"Degree correlation (X vs Y): mean = {np.mean(degree_corrs_random):.4f}, std = {np.std(degree_corrs_random):.4f}")
    print(f"Matrix correlation (X vs Y): mean = {np.mean(matrix_corrs_random):.4f}, std = {np.std(matrix_corrs_random):.4f}")
    print()

    # Statistical comparison
    print("=" * 70)
    print("COMPARISON: Fixed-start vs Random-start")
    print("=" * 70)
    print()

    t_deg, p_deg = stats.ttest_ind(degree_corrs_fixed, degree_corrs_random)
    t_mat, p_mat = stats.ttest_ind(matrix_corrs_fixed, matrix_corrs_random)

    print(f"Degree correlation difference:")
    print(f"  Fixed:  {np.mean(degree_corrs_fixed):.4f}")
    print(f"  Random: {np.mean(degree_corrs_random):.4f}")
    print(f"  t={t_deg:.4f}, p={p_deg:.6f}")
    print()
    print(f"Matrix correlation difference:")
    print(f"  Fixed:  {np.mean(matrix_corrs_fixed):.4f}")
    print(f"  Random: {np.mean(matrix_corrs_random):.4f}")
    print(f"  t={t_mat:.4f}, p={p_mat:.6f}")
    print()

    # One-sample tests
    print("=" * 70)
    print("ONE-SAMPLE TESTS (H0: mean correlation = 0)")
    print("=" * 70)
    print()

    t_fixed, p_fixed = stats.ttest_1samp(matrix_corrs_fixed, 0)
    t_random, p_random = stats.ttest_1samp(matrix_corrs_random, 0)

    print(f"Fixed-start matrix correlations:  t={t_fixed:.4f}, p={p_fixed:.6f}")
    print(f"Random-start matrix correlations: t={t_random:.4f}, p={p_random:.6f}")
    print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    if np.mean(matrix_corrs_random) < 0.02 and p_random > 0.01:
        print("CONFIRMED: The spurious correlation was caused by the FIXED starting nodes.")
        print("With random starting nodes, the correlation disappears.")
        print()
        print("This means the QAP failure is due to a BUG in the test matrix generation,")
        print("NOT a fundamental problem with degree heterogeneity per se.")
    elif np.mean(matrix_corrs_random) > 0.05:
        print("UNEXPECTED: Even with random starting nodes, there's still correlation.")
        print("This suggests something else is causing the spurious correlation.")
    else:
        print("MIXED RESULT: Random start reduces but doesn't eliminate correlation.")

    return {
        'degree_corrs_fixed': degree_corrs_fixed,
        'matrix_corrs_fixed': matrix_corrs_fixed,
        'degree_corrs_random': degree_corrs_random,
        'matrix_corrs_random': matrix_corrs_random
    }


if __name__ == "__main__":
    results = run_diagnosis(n_simulations=500, n_nodes=50, density=0.15, seed=42)
