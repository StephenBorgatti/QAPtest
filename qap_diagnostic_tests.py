#!/usr/bin/env python3
"""
QAP Type 1 Error Diagnostic Test Suite
=======================================

Purpose: Systematically test DSP QAP regression under different scenarios
         to diagnose inflated Type 1 error rates.

Test Scenarios:
  1. Random X, Random Y - no dyadic dependencies (baseline)
  2. Two Random IVs (X1, X2), Random Y - multiple predictors
  3. Random X, Transitive Y - Y has clustering
  4. Transitive X, Transitive Y - both have clustering
  5. Degree-heterogeneous X, Random Y - X has hub structure
  6. Degree-heterogeneous X, Degree-heterogeneous Y - both have hubs

Each test uses homophily_strength = 0, so under the null hypothesis
the Type 1 error rate should be approximately 0.05.

Author: Claude for borgworld/QAPtest
Date: 2025-11-29
"""

import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


###############################################################################
# MATRIX GENERATION FUNCTIONS
###############################################################################

def generate_random_matrix(n: int, density: float = 0.15) -> np.ndarray:
    """
    Generate Simple Random Symmetric Matrix (Erdos-Renyi).

    Creates a purely random symmetric binary matrix with no dyadic dependencies.
    Each edge exists independently with probability p.

    Args:
        n: Number of nodes
        density: Target edge density (probability of each edge)

    Returns:
        Symmetric binary adjacency matrix
    """
    # Generate upper triangle with independent Bernoulli trials
    adj = np.zeros((n, n), dtype=int)
    upper_idx = np.triu_indices(n, k=1)
    n_possible = len(upper_idx[0])

    # Each edge exists with probability = density
    edges = np.random.binomial(1, density, n_possible)
    adj[upper_idx] = edges
    adj = adj + adj.T  # Make symmetric

    return adj


def generate_transitive_matrix(n: int, density: float = 0.15,
                               transitivity: float = 0.4,
                               max_iter: int = 500) -> np.ndarray:
    """
    Generate Matrix with Transitivity (Clustering).

    Creates a symmetric matrix with higher-than-chance transitivity
    by preferentially closing triangles.

    Args:
        n: Number of nodes
        density: Target edge density
        transitivity: Target transitivity coefficient (0 to 1)
        max_iter: Maximum iterations for adjustment

    Returns:
        Symmetric binary adjacency matrix
    """
    # Start with random matrix
    adj = generate_random_matrix(n, density)

    for _ in range(max_iter):
        current_trans = calculate_transitivity(adj)
        current_dens = np.sum(adj) / (n * (n - 1))

        # Check if we've reached target
        if (abs(current_trans - transitivity) < 0.03 and
            abs(current_dens - density) < 0.02):
            break

        # Adjust transitivity
        if current_trans < transitivity - 0.03:
            adj = add_transitive_edge(adj)
        elif current_trans > transitivity + 0.03:
            adj = remove_transitive_edge(adj)

        # Maintain density
        current_dens = np.sum(adj) / (n * (n - 1))
        if current_dens < density - 0.02:
            adj = add_random_edge(adj)
        elif current_dens > density + 0.02:
            adj = remove_random_edge(adj)

    return adj


def generate_degree_heterogeneous_matrix(n: int, density: float = 0.15,
                                         degree_cv: float = 0.5,
                                         max_iter: int = 500) -> np.ndarray:
    """
    Generate Matrix with Degree Heterogeneity.

    Creates a symmetric matrix with heterogeneous degree distribution
    using preferential attachment mechanism.

    Args:
        n: Number of nodes
        density: Target edge density
        degree_cv: Target coefficient of variation for degrees
        max_iter: Maximum iterations for adjustment

    Returns:
        Symmetric binary adjacency matrix
    """
    # Target number of edges
    n_edges = int(round(n * (n - 1) / 2 * density))

    # Initialize with preferential attachment
    adj = np.zeros((n, n), dtype=int)

    # Start with small connected component using RANDOM starting nodes
    # (Fixed bug: previously always used nodes 0,1,2 which caused spurious
    # correlation between independently generated networks)
    start_nodes = np.random.choice(n, size=3, replace=False)
    a, b, c = start_nodes
    adj[a, b] = adj[b, a] = 1
    adj[b, c] = adj[c, b] = 1
    adj[a, c] = adj[c, a] = 1

    degrees = np.sum(adj, axis=1)

    # Add edges using preferential attachment
    edges_added = 3
    max_attempts = n_edges * 100  # Safeguard against infinite loop
    attempts = 0

    while edges_added < n_edges and attempts < max_attempts:
        attempts += 1

        # Probability proportional to (degree + 1)^2 for stronger heterogeneity
        probs = (degrees + 1.0) ** 2
        probs = probs / np.sum(probs)

        # Sample first node
        i = np.random.choice(n, p=probs)

        # For second node, also prefer high degree but not connected to i
        probs2 = (degrees + 1.0) * (1 - adj[i, :])
        probs2[i] = 0

        if np.sum(probs2) > 0:
            probs2 = probs2 / np.sum(probs2)
            j = np.random.choice(n, p=probs2)

            if adj[i, j] == 0:
                adj[i, j] = adj[j, i] = 1
                degrees = np.sum(adj, axis=1)
                edges_added += 1

    # If preferential attachment stalled, fill remaining edges randomly
    if edges_added < n_edges:
        zeros = np.argwhere((adj == 0) & np.triu(np.ones_like(adj, dtype=bool), k=1))
        if len(zeros) > 0:
            remaining = n_edges - edges_added
            selected = np.random.choice(len(zeros), size=min(remaining, len(zeros)), replace=False)
            for idx in selected:
                i, j = zeros[idx]
                adj[i, j] = adj[j, i] = 1

    # Fine-tune to reach target CV
    for _ in range(max_iter):
        degrees = np.sum(adj, axis=1)
        mean_deg = np.mean(degrees)
        current_cv = np.std(degrees) / max(mean_deg, 0.001)
        current_dens = np.sum(adj) / (n * (n - 1))

        if (abs(current_cv - degree_cv) < 0.05 and
            abs(current_dens - density) < 0.02):
            break

        # Maintain density
        if current_dens < density - 0.02:
            adj = add_random_edge(adj)
        elif current_dens > density + 0.02:
            adj = remove_random_edge(adj)

    return adj


###############################################################################
# HELPER FUNCTIONS
###############################################################################

def calculate_transitivity(adj: np.ndarray) -> float:
    """Calculate Global Transitivity coefficient."""
    n = adj.shape[0]
    adj3 = adj @ adj @ adj
    triangles = np.trace(adj3) / 6
    degrees = np.sum(adj, axis=1)
    triples = np.sum(degrees * (degrees - 1)) / 2
    if triples == 0:
        return 0.0
    return 3 * triangles / triples


def add_transitive_edge(adj: np.ndarray) -> np.ndarray:
    """Add an edge that closes a triangle (increases transitivity)."""
    n = adj.shape[0]
    adj = adj.copy()

    for _ in range(20):
        degrees = np.sum(adj, axis=1)
        candidates = np.where(degrees >= 2)[0]
        if len(candidates) == 0:
            break

        j = np.random.choice(candidates)
        neighbors = np.where(adj[j, :] == 1)[0]

        if len(neighbors) >= 2:
            pair = np.random.choice(neighbors, size=2, replace=False)
            i, k = pair[0], pair[1]

            if adj[i, k] == 0:
                adj[i, k] = adj[k, i] = 1
                return adj

    return adj


def remove_transitive_edge(adj: np.ndarray) -> np.ndarray:
    """Remove an edge that's part of a triangle (decreases transitivity)."""
    n = adj.shape[0]
    adj = adj.copy()

    upper_edges = np.argwhere((adj == 1) & np.triu(np.ones_like(adj, dtype=bool), k=1))

    if len(upper_edges) == 0:
        return adj

    np.random.shuffle(upper_edges)

    for edge in upper_edges[:20]:
        i, k = edge[0], edge[1]
        # Check if this edge is in a triangle
        common = np.sum((adj[i, :] == 1) & (adj[k, :] == 1))
        if common > 0:
            adj[i, k] = adj[k, i] = 0
            return adj

    return adj


def add_random_edge(adj: np.ndarray) -> np.ndarray:
    """Add a random edge."""
    n = adj.shape[0]
    adj = adj.copy()

    zeros = np.argwhere((adj == 0) & np.triu(np.ones_like(adj, dtype=bool), k=1))
    if len(zeros) > 0:
        idx = np.random.randint(len(zeros))
        i, j = zeros[idx]
        adj[i, j] = adj[j, i] = 1

    return adj


def remove_random_edge(adj: np.ndarray) -> np.ndarray:
    """Remove a random edge."""
    n = adj.shape[0]
    adj = adj.copy()

    ones = np.argwhere((adj == 1) & np.triu(np.ones_like(adj, dtype=bool), k=1))
    if len(ones) > 0:
        idx = np.random.randint(len(ones))
        i, j = ones[idx]
        adj[i, j] = adj[j, i] = 0

    return adj


def vectorize_symmetric(mat: np.ndarray) -> np.ndarray:
    """Vectorize symmetric matrix (upper triangle only)."""
    return mat[np.triu_indices(mat.shape[0], k=1)]


###############################################################################
# DSP QAP REGRESSION
###############################################################################

def qap_dsp_regression(Y: np.ndarray, X_list: list, nperm: int = 1000) -> dict:
    """
    DSP QAP Regression with Robust Standard Errors.

    Implements the Double Semi-Partialling method from Dekker et al. 2007.

    Args:
        Y: Dependent variable matrix
        X_list: List of predictor matrices
        nperm: Number of permutations

    Returns:
        Dictionary with coefficients, SEs, t-stats, and p-values
    """
    n = Y.shape[0]
    n_x = len(X_list)

    # Vectorize matrices
    y_vec = vectorize_symmetric(Y)
    n_obs = len(y_vec)

    x_vecs = [vectorize_symmetric(X) for X in X_list]
    x_mat = np.column_stack([np.ones(n_obs)] + x_vecs)
    n_params = x_mat.shape[1]

    # Observed regression using least squares
    obs_coef, residuals, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    obs_resid = y_vec - x_mat @ obs_coef

    # R-squared
    ss_res = np.sum(obs_resid ** 2)
    ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Robust standard errors (HC3)
    xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
    hat_matrix = x_mat @ xtx_inv @ x_mat.T
    h = np.diag(hat_matrix)
    h = np.clip(h, 0, 0.999)  # Prevent division by zero
    u = obs_resid / (1 - h)  # HC3 adjustment

    meat = x_mat.T @ np.diag(u ** 2) @ x_mat
    robust_vcov = xtx_inv @ meat @ xtx_inv
    robust_se = np.sqrt(np.diag(robust_vcov))

    # Observed t-statistics
    obs_tstats = obs_coef / robust_se

    # Pre-generate permutations
    perms = [np.random.permutation(n) for _ in range(nperm)]

    # DSP Permutation test for each X coefficient
    p_values = np.zeros(n_params)

    for k in range(n_x):
        coef_idx = k + 1

        # Partial out other X's from X_k
        if n_x > 1:
            Z_indices = [j for j in range(n_x) if j != k]
            Z_vecs = [x_vecs[j] for j in Z_indices]
            Z_mat_for_partial = np.column_stack([np.ones(n_obs)] + Z_vecs)

            # Compute residuals: X_k - predicted_X_k(Z)
            delta, _, _, _ = np.linalg.lstsq(Z_mat_for_partial, x_vecs[k], rcond=None)
            x_k_resid_vec = x_vecs[k] - Z_mat_for_partial @ delta

            # Convert to matrix form
            x_k_resid_mat = np.zeros((n, n))
            x_k_resid_mat[np.triu_indices(n, k=1)] = x_k_resid_vec
            x_k_resid_mat = x_k_resid_mat + x_k_resid_mat.T
        else:
            x_k_resid_mat = X_list[k].copy()

        # Permutation test
        perm_tstats_k = np.zeros(nperm)

        for p_idx, perm in enumerate(perms):
            # Permute X_k residuals (rows and columns)
            x_k_perm_mat = x_k_resid_mat[np.ix_(perm, perm)]
            x_k_perm_vec = vectorize_symmetric(x_k_perm_mat)

            # Build design matrix
            if n_x > 1:
                Z_mat = np.column_stack(Z_vecs)
                x_perm_design = np.column_stack([np.ones(n_obs), x_k_perm_vec, Z_mat])
            else:
                x_perm_design = np.column_stack([np.ones(n_obs), x_k_perm_vec])

            # Fit regression
            perm_coef, _, _, _ = np.linalg.lstsq(x_perm_design, y_vec, rcond=None)
            perm_resid = y_vec - x_perm_design @ perm_coef

            # Compute robust t-stat for the permuted X_k coefficient (index 1)
            try:
                xtx_inv_perm = np.linalg.inv(x_perm_design.T @ x_perm_design)
                h_perm = np.diag(x_perm_design @ xtx_inv_perm @ x_perm_design.T)
                h_perm = np.clip(h_perm, 0, 0.999)
                u_perm = perm_resid / (1 - h_perm)
                meat_perm = x_perm_design.T @ np.diag(u_perm ** 2) @ x_perm_design
                vcov_perm = xtx_inv_perm @ meat_perm @ xtx_inv_perm
                se_perm = np.sqrt(np.diag(vcov_perm))
                perm_tstats_k[p_idx] = perm_coef[1] / se_perm[1]
            except np.linalg.LinAlgError:
                perm_tstats_k[p_idx] = 0  # Singular matrix fallback

        # Two-tailed p-value
        p_values[coef_idx] = np.mean(np.abs(perm_tstats_k) >= np.abs(obs_tstats[coef_idx]))

    # Intercept p-value (permute all X residuals with same permutation)
    x_resid_mats = []
    for k in range(n_x):
        if n_x > 1:
            Z_indices = [j for j in range(n_x) if j != k]
            Z_vecs_k = [x_vecs[j] for j in Z_indices]
            Z_mat_for_partial = np.column_stack([np.ones(n_obs)] + Z_vecs_k)
            delta, _, _, _ = np.linalg.lstsq(Z_mat_for_partial, x_vecs[k], rcond=None)
            x_k_resid_vec = x_vecs[k] - Z_mat_for_partial @ delta
            x_k_resid_mat = np.zeros((n, n))
            x_k_resid_mat[np.triu_indices(n, k=1)] = x_k_resid_vec
            x_k_resid_mat = x_k_resid_mat + x_k_resid_mat.T
            x_resid_mats.append(x_k_resid_mat)
        else:
            x_resid_mats.append(X_list[k].copy())

    perm_tstats_intercept = np.zeros(nperm)
    for p_idx, perm in enumerate(perms):
        x_perm_vecs = [vectorize_symmetric(mat[np.ix_(perm, perm)]) for mat in x_resid_mats]
        x_perm_design = np.column_stack([np.ones(n_obs)] + x_perm_vecs)

        try:
            perm_coef, _, _, _ = np.linalg.lstsq(x_perm_design, y_vec, rcond=None)
            perm_resid = y_vec - x_perm_design @ perm_coef

            xtx_inv_perm = np.linalg.inv(x_perm_design.T @ x_perm_design)
            h_perm = np.diag(x_perm_design @ xtx_inv_perm @ x_perm_design.T)
            h_perm = np.clip(h_perm, 0, 0.999)
            u_perm = perm_resid / (1 - h_perm)
            meat_perm = x_perm_design.T @ np.diag(u_perm ** 2) @ x_perm_design
            vcov_perm = xtx_inv_perm @ meat_perm @ xtx_inv_perm
            se_perm = np.sqrt(np.diag(vcov_perm))
            perm_tstats_intercept[p_idx] = perm_coef[0] / se_perm[0]
        except np.linalg.LinAlgError:
            perm_tstats_intercept[p_idx] = 0

    p_values[0] = np.mean(np.abs(perm_tstats_intercept) >= np.abs(obs_tstats[0]))

    return {
        'coefficients': obs_coef,
        'robust_se': robust_se,
        't_statistics': obs_tstats,
        'p_values': p_values,
        'r_squared': r_squared
    }


###############################################################################
# TEST FUNCTIONS
###############################################################################

def run_single_test(args):
    """
    Run a single test iteration.

    Args:
        args: Tuple of (test_type, n_nodes, density, n_perms, seed)

    Returns:
        Dictionary with p-values and network statistics
    """
    test_type, n_nodes, density, n_perms, seed = args

    # Set seed for this simulation
    np.random.seed(seed)

    # Generate matrices based on test type
    if test_type == 1:
        # Test 1: Random X, Random Y
        X = generate_random_matrix(n_nodes, density)
        Y = generate_random_matrix(n_nodes, density)
        X_list = [X]

    elif test_type == 2:
        # Test 2: Two Random IVs, Random Y
        X1 = generate_random_matrix(n_nodes, density)
        X2 = generate_random_matrix(n_nodes, density)
        Y = generate_random_matrix(n_nodes, density)
        X_list = [X1, X2]

    elif test_type == 3:
        # Test 3: Random X, Transitive Y
        X = generate_random_matrix(n_nodes, density)
        Y = generate_transitive_matrix(n_nodes, density, transitivity=0.4)
        X_list = [X]

    elif test_type == 4:
        # Test 4: Transitive X, Transitive Y
        X = generate_transitive_matrix(n_nodes, density, transitivity=0.4)
        Y = generate_transitive_matrix(n_nodes, density, transitivity=0.4)
        X_list = [X]

    elif test_type == 5:
        # Test 5: Degree-heterogeneous X, Random Y
        X = generate_degree_heterogeneous_matrix(n_nodes, density, degree_cv=0.5)
        Y = generate_random_matrix(n_nodes, density)
        X_list = [X]

    elif test_type == 6:
        # Test 6: Degree-heterogeneous X, Degree-heterogeneous Y
        X = generate_degree_heterogeneous_matrix(n_nodes, density, degree_cv=0.5)
        Y = generate_degree_heterogeneous_matrix(n_nodes, density, degree_cv=0.5)
        X_list = [X]

    # Run QAP regression
    result = qap_dsp_regression(Y, X_list, nperm=n_perms)

    # Calculate network statistics
    trans_Y = calculate_transitivity(Y)
    degrees_Y = np.sum(Y, axis=1)
    cv_Y = np.std(degrees_Y) / max(np.mean(degrees_Y), 0.001)
    dens_Y = np.sum(Y) / (n_nodes * (n_nodes - 1))

    X_main = X_list[0]
    trans_X = calculate_transitivity(X_main)
    degrees_X = np.sum(X_main, axis=1)
    cv_X = np.std(degrees_X) / max(np.mean(degrees_X), 0.001)
    dens_X = np.sum(X_main) / (n_nodes * (n_nodes - 1))

    return {
        'p_values': result['p_values'],
        'coefficients': result['coefficients'],
        'r_squared': result['r_squared'],
        'trans_X': trans_X,
        'trans_Y': trans_Y,
        'cv_X': cv_X,
        'cv_Y': cv_Y,
        'dens_X': dens_X,
        'dens_Y': dens_Y
    }


def run_diagnostic_suite(n_simulations: int = 200,
                         n_permutations: int = 1000,
                         n_nodes: int = 50,
                         density: float = 0.15,
                         use_parallel: bool = True,
                         n_cores: int = None,
                         seed: int = 42) -> dict:
    """
    Run complete diagnostic test suite.

    Args:
        n_simulations: Number of simulations per test
        n_permutations: Number of permutations per QAP test
        n_nodes: Number of nodes in networks
        density: Target edge density
        use_parallel: Use parallel processing
        n_cores: Number of cores (None = auto-detect)
        seed: Random seed

    Returns:
        Dictionary with results for all tests
    """
    np.random.seed(seed)

    # Test descriptions
    test_names = [
        "Test 1: Random X, Random Y (baseline)",
        "Test 2: Two Random IVs (X1, X2), Random Y",
        "Test 3: Random X, Transitive Y",
        "Test 4: Transitive X, Transitive Y",
        "Test 5: Degree-het X, Random Y",
        "Test 6: Degree-het X, Degree-het Y"
    ]

    # Set up parallel processing
    if use_parallel:
        if n_cores is None:
            n_cores = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_cores = 1

    print()
    print("=" * 80)
    print("QAP TYPE 1 ERROR DIAGNOSTIC TEST SUITE")
    print("=" * 80)
    print()
    print("PURPOSE: Identify source of inflated Type 1 error rates")
    print()
    print("SIMULATION PARAMETERS:")
    print(f"  Simulations per test: {n_simulations}")
    print(f"  Permutations per QAP: {n_permutations}")
    print(f"  Network size: {n_nodes} nodes")
    print(f"  Target density: {density:.2f}")
    print(f"  Parallel processing: {'YES' if use_parallel else 'NO'} ({n_cores} cores)")
    print()

    # Expected CI for alpha = 0.05
    se_alpha = np.sqrt(0.05 * 0.95 / n_simulations)
    ci_lower = 0.05 - 1.96 * se_alpha
    ci_upper = 0.05 + 1.96 * se_alpha
    print(f"EXPECTED TYPE 1 ERROR RATE: 0.05")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

    # Store all results
    all_results = {}
    summary_data = []

    # Run each test
    for test_type in range(1, 7):
        print("-" * 80)
        print(test_names[test_type - 1])
        print("-" * 80)

        start_time = time.time()

        # Generate seeds for each simulation
        sim_seeds = [seed + test_type * 10000 + i for i in range(n_simulations)]
        args_list = [(test_type, n_nodes, density, n_permutations, s) for s in sim_seeds]

        if use_parallel and n_cores > 1:
            print(f"Running {n_simulations} simulations in parallel...")

            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results_list = list(executor.map(run_single_test, args_list))
        else:
            print(f"Running {n_simulations} simulations sequentially...")
            print("Progress: ", end="", flush=True)

            results_list = []
            for i, args in enumerate(args_list):
                if (i + 1) % 20 == 0:
                    print(f"{i + 1} ", end="", flush=True)
                results_list.append(run_single_test(args))
            print()

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed / 60:.2f} minutes")
        print()

        # Extract p-values
        if test_type == 2:
            # Two predictors
            p_vals_X1 = [r['p_values'][1] for r in results_list]
            p_vals_X2 = [r['p_values'][2] for r in results_list]
            p_vals = {'X1': p_vals_X1, 'X2': p_vals_X2}

            type1_X1 = np.mean(np.array(p_vals_X1) < 0.05)
            type1_X2 = np.mean(np.array(p_vals_X2) < 0.05)

            print("TYPE 1 ERROR RATES:")
            status_X1 = "(OK)" if ci_lower <= type1_X1 <= ci_upper else "(WARNING)"
            status_X2 = "(OK)" if ci_lower <= type1_X2 <= ci_upper else "(WARNING)"
            print(f"  X1 coefficient: {type1_X1:.4f} {status_X1}")
            print(f"  X2 coefficient: {type1_X2:.4f} {status_X2}")

        else:
            # Single predictor
            p_vals_X = [r['p_values'][1] for r in results_list]
            p_vals = {'X': p_vals_X}

            type1_X = np.mean(np.array(p_vals_X) < 0.05)

            print("TYPE 1 ERROR RATE:")
            status_X = "(OK)" if ci_lower <= type1_X <= ci_upper else "(WARNING)"
            print(f"  X coefficient: {type1_X:.4f} {status_X}")

        # P-value uniformity test
        print()
        print("P-VALUE UNIFORMITY (Kolmogorov-Smirnov test):")
        for name, pv in p_vals.items():
            ks_stat, ks_pval = stats.kstest(pv, 'uniform')
            uniform_status = "(uniform)" if ks_pval > 0.05 else "(NOT uniform)"
            print(f"  {name}: KS statistic = {ks_stat:.4f}, p-value = {ks_pval:.4f} {uniform_status}")

            # Add to summary
            type1_rate = np.mean(np.array(pv) < 0.05)
            status = "OK" if ci_lower <= type1_rate <= ci_upper else "INFLATED"
            if type1_rate < ci_lower:
                status = "DEFLATED"
            summary_data.append({
                'Test': f'Test {test_type}',
                'Coefficient': name,
                'Type1_Error': type1_rate,
                'KS_pvalue': ks_pval,
                'Status': status
            })

        # Network statistics
        print()
        print("ACHIEVED NETWORK PROPERTIES:")
        trans_X = np.mean([r['trans_X'] for r in results_list])
        trans_Y = np.mean([r['trans_Y'] for r in results_list])
        cv_X = np.mean([r['cv_X'] for r in results_list])
        cv_Y = np.mean([r['cv_Y'] for r in results_list])
        dens_X = np.mean([r['dens_X'] for r in results_list])
        dens_Y = np.mean([r['dens_Y'] for r in results_list])

        print(f"  Density X: {dens_X:.3f}, Y: {dens_Y:.3f}")
        print(f"  Transitivity X: {trans_X:.3f}, Y: {trans_Y:.3f}")
        print(f"  Degree CV X: {cv_X:.3f}, Y: {cv_Y:.3f}")
        print()

        # Store results
        all_results[test_type] = {
            'test_name': test_names[test_type - 1],
            'p_values': p_vals,
            'results_list': results_list,
            'runtime': elapsed
        }

    # Summary table
    print("=" * 80)
    print("SUMMARY: TYPE 1 ERROR RATES BY TEST SCENARIO")
    print("=" * 80)
    print()

    print(f"{'Test':<10} {'Coefficient':<12} {'Type1_Error':<12} {'KS_pvalue':<12} {'Status':<10}")
    print("-" * 56)
    for row in summary_data:
        print(f"{row['Test']:<10} {row['Coefficient']:<12} {row['Type1_Error']:<12.4f} {row['KS_pvalue']:<12.4f} {row['Status']:<10}")

    print()
    print(f"Expected range for Type 1 error: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Diagnosis
    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    inflated_tests = set(row['Test'] for row in summary_data if row['Status'] == 'INFLATED')

    if not inflated_tests:
        print("All tests show proper Type 1 error control.")
    else:
        print("Tests with inflated Type 1 error rates:")
        for test in sorted(inflated_tests):
            print(f"  - {test}")

        print()
        print("POSSIBLE CAUSES:")

        if "Test 1" in inflated_tests:
            print("  * Test 1 (baseline) is inflated: Problem is in the core DSP algorithm")
        if "Test 2" in inflated_tests and "Test 1" not in inflated_tests:
            print("  * Test 2 inflated but not Test 1: Problem with multiple predictors")
        if "Test 3" in inflated_tests and "Test 1" not in inflated_tests:
            print("  * Test 3 inflated but not Test 1: Transitivity in Y causes inflation")
        if "Test 4" in inflated_tests and "Test 3" not in inflated_tests:
            print("  * Test 4 inflated but not Test 3: Transitivity in X causes inflation")
        if "Test 5" in inflated_tests and "Test 1" not in inflated_tests:
            print("  * Test 5 inflated but not Test 1: Degree heterogeneity in X causes inflation")
        if "Test 6" in inflated_tests and "Test 5" not in inflated_tests:
            print("  * Test 6 inflated but not Test 5: Degree heterogeneity in Y causes inflation")

    print()

    return {
        'summary': summary_data,
        'all_results': all_results,
        'parameters': {
            'n_simulations': n_simulations,
            'n_permutations': n_permutations,
            'n_nodes': n_nodes,
            'density': density
        },
        'ci': (ci_lower, ci_upper)
    }


###############################################################################
# MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='QAP Type 1 Error Diagnostic Tests')
    parser.add_argument('--sims', type=int, default=200,
                        help='Number of simulations per test (default: 200)')
    parser.add_argument('--perms', type=int, default=1000,
                        help='Number of permutations per QAP test (default: 1000)')
    parser.add_argument('--nodes', type=int, default=50,
                        help='Number of nodes in networks (default: 50)')
    parser.add_argument('--density', type=float, default=0.15,
                        help='Target edge density (default: 0.15)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially instead of in parallel')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of cores for parallel processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    results = run_diagnostic_suite(
        n_simulations=args.sims,
        n_permutations=args.perms,
        n_nodes=args.nodes,
        density=args.density,
        use_parallel=not args.sequential,
        n_cores=args.cores,
        seed=args.seed
    )

    print("Diagnostic tests complete.")
