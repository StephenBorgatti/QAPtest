#!/usr/bin/env python3
"""
Re-run Tests 5 and 6 with FIXED matrix generation (random starting nodes).

This verifies whether QAP DSP works correctly when degree-heterogeneous
matrices are truly independent.
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


def generate_degree_heterogeneous_matrix_FIXED(n: int, density: float = 0.15) -> np.ndarray:
    """
    Generate matrix with degree heterogeneity using preferential attachment.
    FIXED: Uses RANDOM starting nodes instead of always 0,1,2.
    """
    n_edges = int(round(n * (n - 1) / 2 * density))
    adj = np.zeros((n, n), dtype=int)

    # RANDOMLY pick 3 starting nodes (THE FIX!)
    start_nodes = np.random.choice(n, size=3, replace=False)
    a, b, c = start_nodes
    adj[a, b] = adj[b, a] = 1
    adj[b, c] = adj[c, b] = 1
    adj[a, c] = adj[c, a] = 1

    degrees = np.sum(adj, axis=1)
    edges_added = 3

    while edges_added < n_edges:
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

    return adj


def vectorize_symmetric(mat: np.ndarray) -> np.ndarray:
    """Vectorize symmetric matrix (upper triangle only)."""
    return mat[np.triu_indices(mat.shape[0], k=1)]


def qap_dsp_regression(Y: np.ndarray, X_list: list, nperm: int = 1000) -> dict:
    """DSP QAP Regression."""
    n = Y.shape[0]
    n_x = len(X_list)

    y_vec = vectorize_symmetric(Y)
    n_obs = len(y_vec)

    x_vecs = [vectorize_symmetric(X) for X in X_list]
    x_mat = np.column_stack([np.ones(n_obs)] + x_vecs)
    n_params = x_mat.shape[1]

    obs_coef, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    obs_resid = y_vec - x_mat @ obs_coef

    # Robust SE (HC3)
    xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
    hat_matrix = x_mat @ xtx_inv @ x_mat.T
    h = np.clip(np.diag(hat_matrix), 0, 0.999)
    u = obs_resid / (1 - h)
    meat = x_mat.T @ np.diag(u ** 2) @ x_mat
    robust_se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    obs_tstats = obs_coef / robust_se

    perms = [np.random.permutation(n) for _ in range(nperm)]
    p_values = np.zeros(n_params)

    for k in range(n_x):
        coef_idx = k + 1
        x_k_resid_mat = X_list[k].copy()

        perm_tstats_k = np.zeros(nperm)
        for p_idx, perm in enumerate(perms):
            x_k_perm_mat = x_k_resid_mat[np.ix_(perm, perm)]
            x_k_perm_vec = vectorize_symmetric(x_k_perm_mat)
            x_perm_design = np.column_stack([np.ones(n_obs), x_k_perm_vec])

            try:
                perm_coef, _, _, _ = np.linalg.lstsq(x_perm_design, y_vec, rcond=None)
                perm_resid = y_vec - x_perm_design @ perm_coef
                xtx_inv_perm = np.linalg.inv(x_perm_design.T @ x_perm_design)
                h_perm = np.clip(np.diag(x_perm_design @ xtx_inv_perm @ x_perm_design.T), 0, 0.999)
                u_perm = perm_resid / (1 - h_perm)
                meat_perm = x_perm_design.T @ np.diag(u_perm ** 2) @ x_perm_design
                se_perm = np.sqrt(np.diag(xtx_inv_perm @ meat_perm @ xtx_inv_perm))
                perm_tstats_k[p_idx] = perm_coef[1] / se_perm[1]
            except:
                perm_tstats_k[p_idx] = 0

        p_values[coef_idx] = np.mean(np.abs(perm_tstats_k) >= np.abs(obs_tstats[coef_idx]))

    return {'p_values': p_values, 'coefficients': obs_coef}


def run_tests(n_simulations: int = 200, n_nodes: int = 50, density: float = 0.15,
              n_perms: int = 1000, seed: int = 42):
    """Run Tests 5 and 6 with fixed generation."""
    np.random.seed(seed)

    print("=" * 70)
    print("TESTS 5 & 6 WITH FIXED MATRIX GENERATION")
    print("=" * 70)
    print()
    print(f"Simulations: {n_simulations}")
    print(f"Permutations: {n_perms}")
    print(f"Network size: {n_nodes} nodes")
    print()

    se_alpha = np.sqrt(0.05 * 0.95 / n_simulations)
    ci_lower = 0.05 - 1.96 * se_alpha
    ci_upper = 0.05 + 1.96 * se_alpha
    print(f"Expected Type 1 error: 0.05")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

    # Test 5: Degree-het X, Random Y
    print("-" * 70)
    print("Test 5: Degree-het X (fixed generation), Random Y")
    print("-" * 70)

    p_vals_5 = []
    corrs_5 = []

    for i in range(n_simulations):
        if (i + 1) % 50 == 0:
            print(f"  Simulation {i + 1}/{n_simulations}")

        X = generate_degree_heterogeneous_matrix_FIXED(n_nodes, density)
        Y = generate_random_matrix(n_nodes, density)

        # Track correlation
        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)
        corr, _ = stats.pearsonr(x_vec, y_vec)
        corrs_5.append(corr)

        result = qap_dsp_regression(Y, [X], nperm=n_perms)
        p_vals_5.append(result['p_values'][1])

    type1_5 = np.mean(np.array(p_vals_5) < 0.05)
    ks_stat_5, ks_pval_5 = stats.kstest(p_vals_5, 'uniform')

    print()
    print(f"Type 1 error rate: {type1_5:.4f}", "(OK)" if ci_lower <= type1_5 <= ci_upper else "(WARNING)")
    print(f"KS test for uniformity: stat={ks_stat_5:.4f}, p={ks_pval_5:.4f}")
    print(f"Mean X-Y correlation: {np.mean(corrs_5):.6f}")
    print()

    # Test 6: Degree-het X, Degree-het Y
    print("-" * 70)
    print("Test 6: Degree-het X (fixed), Degree-het Y (fixed, independent)")
    print("-" * 70)

    p_vals_6 = []
    corrs_6 = []

    for i in range(n_simulations):
        if (i + 1) % 50 == 0:
            print(f"  Simulation {i + 1}/{n_simulations}")

        X = generate_degree_heterogeneous_matrix_FIXED(n_nodes, density)
        Y = generate_degree_heterogeneous_matrix_FIXED(n_nodes, density)

        # Track correlation
        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)
        corr, _ = stats.pearsonr(x_vec, y_vec)
        corrs_6.append(corr)

        result = qap_dsp_regression(Y, [X], nperm=n_perms)
        p_vals_6.append(result['p_values'][1])

    type1_6 = np.mean(np.array(p_vals_6) < 0.05)
    ks_stat_6, ks_pval_6 = stats.kstest(p_vals_6, 'uniform')

    print()
    print(f"Type 1 error rate: {type1_6:.4f}", "(OK)" if ci_lower <= type1_6 <= ci_upper else "(WARNING)")
    print(f"KS test for uniformity: stat={ks_stat_6:.4f}, p={ks_pval_6:.4f}")
    print(f"Mean X-Y correlation: {np.mean(corrs_6):.6f}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Test':<10} {'Type1 Error':<15} {'X-Y Correlation':<18} {'Status':<10}")
    print("-" * 53)
    status_5 = "OK" if ci_lower <= type1_5 <= ci_upper else "WARNING"
    status_6 = "OK" if ci_lower <= type1_6 <= ci_upper else "WARNING"
    print(f"{'Test 5':<10} {type1_5:<15.4f} {np.mean(corrs_5):<18.6f} {status_5:<10}")
    print(f"{'Test 6':<10} {type1_6:<15.4f} {np.mean(corrs_6):<18.6f} {status_6:<10}")
    print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    if status_5 == "OK" and status_6 == "OK":
        print("SUCCESS: With fixed matrix generation, QAP controls Type 1 error properly!")
        print()
        print("The original inflated Type 1 error in Test 6 was caused by a BUG in")
        print("the test setup (fixed starting nodes creating spurious correlation),")
        print("NOT a failure of the DSP QAP procedure.")
    else:
        print("Type 1 error is still inflated even with fixed generation.")
        print("This suggests there may be a real issue with QAP and degree heterogeneity.")

    return {
        'p_vals_5': p_vals_5,
        'p_vals_6': p_vals_6,
        'corrs_5': corrs_5,
        'corrs_6': corrs_6
    }


if __name__ == "__main__":
    results = run_tests(n_simulations=200, n_nodes=50, density=0.15, n_perms=1000, seed=42)
