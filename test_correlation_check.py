#!/usr/bin/env python3
"""
Test whether independent degree heterogeneity introduces correlation between X and Y.

This test compares:
  Test 5: Degree-heterogeneous X, Random Y
  Test 6: Degree-heterogeneous X, Degree-heterogeneous Y (independently generated)

If degree heterogeneity introduces spurious correlation, we should see higher
correlations between X and Y in Test 6 compared to Test 5.
"""

import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def generate_random_matrix(n: int, density: float = 0.15) -> np.ndarray:
    """Generate random symmetric matrix."""
    adj = np.zeros((n, n), dtype=int)
    upper_idx = np.triu_indices(n, k=1)
    n_possible = len(upper_idx[0])
    edges = np.random.binomial(1, density, n_possible)
    adj[upper_idx] = edges
    adj = adj + adj.T
    return adj


def generate_degree_heterogeneous_matrix(n: int, density: float = 0.15,
                                         degree_cv: float = 0.5,
                                         max_iter: int = 500) -> np.ndarray:
    """Generate matrix with degree heterogeneity using preferential attachment."""
    n_edges = int(round(n * (n - 1) / 2 * density))
    adj = np.zeros((n, n), dtype=int)

    # Start with small connected component
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


def vectorize_symmetric(mat: np.ndarray) -> np.ndarray:
    """Vectorize symmetric matrix (upper triangle only)."""
    return mat[np.triu_indices(mat.shape[0], k=1)]


def run_correlation_tests(n_simulations: int = 500,
                          n_nodes: int = 50,
                          density: float = 0.15,
                          seed: int = 42):
    """
    Run tests 5 and 6, tracking X-Y correlations.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("CORRELATION CHECK: Does degree heterogeneity introduce X-Y correlation?")
    print("=" * 70)
    print()
    print(f"Simulations: {n_simulations}")
    print(f"Network size: {n_nodes} nodes")
    print(f"Density: {density}")
    print()

    # Storage for correlations
    correlations_test5 = []
    correlations_test6 = []

    # Also track network properties
    cv_X_test5 = []
    cv_Y_test5 = []
    cv_X_test6 = []
    cv_Y_test6 = []

    print("Running Test 5: Degree-het X, Random Y...")
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"  Simulation {i + 1}/{n_simulations}")

        X = generate_degree_heterogeneous_matrix(n_nodes, density)
        Y = generate_random_matrix(n_nodes, density)

        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)

        corr, _ = stats.pearsonr(x_vec, y_vec)
        correlations_test5.append(corr)

        # Track degree CV
        deg_X = np.sum(X, axis=1)
        deg_Y = np.sum(Y, axis=1)
        cv_X_test5.append(np.std(deg_X) / max(np.mean(deg_X), 0.001))
        cv_Y_test5.append(np.std(deg_Y) / max(np.mean(deg_Y), 0.001))

    print()
    print("Running Test 6: Degree-het X, Degree-het Y (independent)...")
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"  Simulation {i + 1}/{n_simulations}")

        X = generate_degree_heterogeneous_matrix(n_nodes, density)
        Y = generate_degree_heterogeneous_matrix(n_nodes, density)

        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)

        corr, _ = stats.pearsonr(x_vec, y_vec)
        correlations_test6.append(corr)

        # Track degree CV
        deg_X = np.sum(X, axis=1)
        deg_Y = np.sum(Y, axis=1)
        cv_X_test6.append(np.std(deg_X) / max(np.mean(deg_X), 0.001))
        cv_Y_test6.append(np.std(deg_Y) / max(np.mean(deg_Y), 0.001))

    # Convert to arrays
    correlations_test5 = np.array(correlations_test5)
    correlations_test6 = np.array(correlations_test6)

    print()
    print("=" * 70)
    print("RESULTS: X-Y CORRELATIONS")
    print("=" * 70)
    print()

    print("Test 5 (Degree-het X, Random Y):")
    print(f"  Mean correlation:   {np.mean(correlations_test5):.6f}")
    print(f"  Std correlation:    {np.std(correlations_test5):.6f}")
    print(f"  Min correlation:    {np.min(correlations_test5):.6f}")
    print(f"  Max correlation:    {np.max(correlations_test5):.6f}")
    print(f"  |corr| > 0.1:       {np.mean(np.abs(correlations_test5) > 0.1):.4f}")
    print(f"  Mean degree CV X:   {np.mean(cv_X_test5):.3f}")
    print(f"  Mean degree CV Y:   {np.mean(cv_Y_test5):.3f}")
    print()

    print("Test 6 (Degree-het X, Degree-het Y, independent):")
    print(f"  Mean correlation:   {np.mean(correlations_test6):.6f}")
    print(f"  Std correlation:    {np.std(correlations_test6):.6f}")
    print(f"  Min correlation:    {np.min(correlations_test6):.6f}")
    print(f"  Max correlation:    {np.max(correlations_test6):.6f}")
    print(f"  |corr| > 0.1:       {np.mean(np.abs(correlations_test6) > 0.1):.4f}")
    print(f"  Mean degree CV X:   {np.mean(cv_X_test6):.3f}")
    print(f"  Mean degree CV Y:   {np.mean(cv_Y_test6):.3f}")
    print()

    # Statistical comparison
    print("=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    print()

    # Test if mean correlations differ
    t_stat, t_pval = stats.ttest_ind(correlations_test5, correlations_test6)
    print(f"Two-sample t-test (mean correlations):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {t_pval:.4f}")
    print()

    # Test if absolute correlations differ (this is more relevant)
    abs_corr5 = np.abs(correlations_test5)
    abs_corr6 = np.abs(correlations_test6)

    t_stat_abs, t_pval_abs = stats.ttest_ind(abs_corr5, abs_corr6)
    print(f"Two-sample t-test (absolute correlations):")
    print(f"  Mean |corr| Test 5: {np.mean(abs_corr5):.6f}")
    print(f"  Mean |corr| Test 6: {np.mean(abs_corr6):.6f}")
    print(f"  t-statistic: {t_stat_abs:.4f}")
    print(f"  p-value:     {t_pval_abs:.4f}")
    print()

    # Also check variance of correlations
    print(f"Variance comparison:")
    print(f"  Var(corr) Test 5: {np.var(correlations_test5):.8f}")
    print(f"  Var(corr) Test 6: {np.var(correlations_test6):.8f}")
    f_stat = np.var(correlations_test6) / np.var(correlations_test5)
    print(f"  F-ratio (Test6/Test5): {f_stat:.4f}")
    print()

    # Check if correlations are centered around zero
    print("=" * 70)
    print("ONE-SAMPLE TESTS (H0: mean correlation = 0)")
    print("=" * 70)
    print()

    t5, p5 = stats.ttest_1samp(correlations_test5, 0)
    t6, p6 = stats.ttest_1samp(correlations_test6, 0)

    print(f"Test 5: t={t5:.4f}, p={p5:.4f}")
    print(f"Test 6: t={t6:.4f}, p={p6:.4f}")
    print()

    # Percentile distribution
    print("=" * 70)
    print("CORRELATION PERCENTILES")
    print("=" * 70)
    print()

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"{'Percentile':<12} {'Test 5':<12} {'Test 6':<12}")
    print("-" * 36)
    for p in percentiles:
        v5 = np.percentile(correlations_test5, p)
        v6 = np.percentile(correlations_test6, p)
        print(f"{p:<12} {v5:<12.6f} {v6:<12.6f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if t_pval_abs < 0.05:
        if np.mean(abs_corr6) > np.mean(abs_corr5):
            print("FINDING: Test 6 shows HIGHER absolute correlations than Test 5.")
            print("This suggests degree heterogeneity in BOTH networks may introduce")
            print("spurious correlation even when generated independently.")
        else:
            print("FINDING: Test 5 shows HIGHER absolute correlations than Test 6.")
    else:
        print("FINDING: No significant difference in absolute correlations between tests.")
        print("This suggests independent degree heterogeneity does NOT introduce")
        print("spurious correlation between X and Y.")

    print()

    return {
        'correlations_test5': correlations_test5,
        'correlations_test6': correlations_test6,
        'cv_X_test5': cv_X_test5,
        'cv_Y_test5': cv_Y_test5,
        'cv_X_test6': cv_X_test6,
        'cv_Y_test6': cv_Y_test6
    }


if __name__ == "__main__":
    results = run_correlation_tests(n_simulations=500, n_nodes=50, density=0.15, seed=42)
