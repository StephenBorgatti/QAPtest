#!/usr/bin/env python3
"""
Run Homophily Tests for QAP Type 1 Error Validation

Tests QAP robustness when networks have homophily-based structure.

Test H1: Independent Homophily
  - X has homophily based on attribute AX
  - Y has homophily based on attribute AY (independent of AX)
  - Since AX ⊥ AY, X and Y should be uncorrelated (null is true)

Test H2a: Shared Attribute with Sender/Receiver Controls
  - Both X and Y have homophily based on same attribute A
  - Control for: AC(i,j)=A(i), AR(i,j)=A(j), AC*AR
  - After controlling, residual X-Y relationship should be null

Test H2b: Shared Attribute with Similarity Control
  - Both X and Y have homophily based on same attribute A
  - Control for: AS(i,j) = |A(i) - A(j)| (absolute difference)
  - After controlling, residual X-Y relationship should be null

Author: Claude for QAPtest
Date: 2025-12-07
"""

import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import warnings
import argparse

warnings.filterwarnings('ignore')

from qap_diagnostic_tests import (
    vectorize_symmetric,
    calculate_transitivity,
    qap_dsp_regression
)


###############################################################################
# HOMOPHILY NETWORK GENERATION
###############################################################################

def generate_homophily_matrix(n: int, attribute: np.ndarray, density: float = 0.15,
                               homophily_strength: float = 3.0,
                               max_iter: int = 100) -> np.ndarray:
    """
    Generate a symmetric binary network with homophily based on a continuous attribute.

    Nodes with similar attribute values are more likely to form ties.

    Args:
        n: Number of nodes
        attribute: 1D array of continuous node attributes (length n)
        density: Target edge density
        homophily_strength: Controls how strongly similarity affects tie probability.
                           Higher values = stronger homophily effect.
                           λ in P(tie) ∝ exp(-λ * |A(i) - A(j)|)
        max_iter: Maximum iterations for density adjustment

    Returns:
        Symmetric binary adjacency matrix
    """
    # Compute pairwise absolute differences
    diff_matrix = np.abs(attribute[:, np.newaxis] - attribute[np.newaxis, :])

    # Compute tie probabilities based on similarity
    # P(tie_ij) ∝ exp(-λ * |A(i) - A(j)|)
    prob_matrix = np.exp(-homophily_strength * diff_matrix)

    # Zero out diagonal
    np.fill_diagonal(prob_matrix, 0)

    # Normalize to achieve target density (approximately)
    # Scale probabilities so expected density matches target
    upper_probs = prob_matrix[np.triu_indices(n, k=1)]
    current_mean = np.mean(upper_probs)

    # Adjust scale to target density
    if current_mean > 0:
        scale = density / current_mean
        prob_matrix = prob_matrix * scale
        prob_matrix = np.clip(prob_matrix, 0, 1)

    # Generate adjacency matrix using these probabilities
    adj = np.zeros((n, n), dtype=int)
    upper_idx = np.triu_indices(n, k=1)

    probs = prob_matrix[upper_idx]
    edges = np.random.binomial(1, probs)
    adj[upper_idx] = edges
    adj = adj + adj.T  # Make symmetric

    # Fine-tune density if needed
    current_density = np.sum(adj) / (n * (n - 1))

    for _ in range(max_iter):
        current_density = np.sum(adj) / (n * (n - 1))

        if abs(current_density - density) < 0.02:
            break

        if current_density < density - 0.02:
            # Add edges preferentially between similar nodes
            adj = add_homophily_edge(adj, diff_matrix)
        elif current_density > density + 0.02:
            # Remove edges preferentially between dissimilar nodes
            adj = remove_dissimilar_edge(adj, diff_matrix)

    return adj


def add_homophily_edge(adj: np.ndarray, diff_matrix: np.ndarray) -> np.ndarray:
    """Add an edge preferentially between similar nodes."""
    n = adj.shape[0]
    adj = adj.copy()

    # Find non-edges
    zeros = np.argwhere((adj == 0) & np.triu(np.ones_like(adj, dtype=bool), k=1))

    if len(zeros) == 0:
        return adj

    # Weight by similarity (inverse of difference)
    weights = []
    for idx in range(len(zeros)):
        i, j = zeros[idx]
        # Higher weight for more similar pairs
        weights.append(np.exp(-3 * diff_matrix[i, j]))

    weights = np.array(weights)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        chosen = np.random.choice(len(zeros), p=weights)
        i, j = zeros[chosen]
        adj[i, j] = adj[j, i] = 1

    return adj


def remove_dissimilar_edge(adj: np.ndarray, diff_matrix: np.ndarray) -> np.ndarray:
    """Remove an edge preferentially between dissimilar nodes."""
    n = adj.shape[0]
    adj = adj.copy()

    # Find edges
    ones = np.argwhere((adj == 1) & np.triu(np.ones_like(adj, dtype=bool), k=1))

    if len(ones) == 0:
        return adj

    # Weight by dissimilarity
    weights = []
    for idx in range(len(ones)):
        i, j = ones[idx]
        # Higher weight for more dissimilar pairs
        weights.append(diff_matrix[i, j])

    weights = np.array(weights)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        chosen = np.random.choice(len(ones), p=weights)
        i, j = ones[chosen]
        adj[i, j] = adj[j, i] = 0

    return adj


def create_control_matrices(attribute: np.ndarray) -> dict:
    """
    Create control matrices for homophily based on a node attribute.

    Args:
        attribute: 1D array of node attributes (length n)

    Returns:
        Dictionary with control matrices:
        - AC: sender effect, ac(i,j) = A(i)
        - AR: receiver effect, ar(i,j) = A(j)
        - AC_AR: interaction, ac(i,j) * ar(i,j) = A(i) * A(j)
        - AS: absolute similarity, |A(i) - A(j)|
        - AS_sq: squared difference, (A(i) - A(j))^2
    """
    n = len(attribute)

    # Sender effect: AC(i,j) = A(i)
    AC = np.tile(attribute[:, np.newaxis], (1, n))

    # Receiver effect: AR(i,j) = A(j)
    AR = np.tile(attribute[np.newaxis, :], (n, 1))

    # Interaction: A(i) * A(j)
    AC_AR = AC * AR

    # Absolute difference (similarity): |A(i) - A(j)|
    AS = np.abs(attribute[:, np.newaxis] - attribute[np.newaxis, :])

    # Squared difference
    AS_sq = (attribute[:, np.newaxis] - attribute[np.newaxis, :]) ** 2

    return {
        'AC': AC,
        'AR': AR,
        'AC_AR': AC_AR,
        'AS': AS,
        'AS_sq': AS_sq
    }


def compute_homophily_correlation(adj: np.ndarray, attribute: np.ndarray) -> float:
    """
    Compute correlation between tie presence and attribute similarity.

    Args:
        adj: Adjacency matrix
        attribute: Node attribute vector

    Returns:
        Correlation coefficient (negative = homophily, since smaller diff = more ties)
    """
    n = adj.shape[0]

    # Get upper triangle values
    upper_idx = np.triu_indices(n, k=1)
    ties = adj[upper_idx]

    # Compute absolute differences
    diff_matrix = np.abs(attribute[:, np.newaxis] - attribute[np.newaxis, :])
    diffs = diff_matrix[upper_idx]

    # Correlation (negative means homophily: small diff → more ties)
    if np.std(ties) > 0 and np.std(diffs) > 0:
        return np.corrcoef(ties, diffs)[0, 1]
    return 0.0


###############################################################################
# TEST FUNCTIONS
###############################################################################

def run_single_homophily_test(args):
    """
    Run a single homophily test iteration.

    Args:
        args: Tuple of (test_type, n_nodes, density, n_perms, homophily_strength, seed)

    Returns:
        Dictionary with p-values and network statistics
    """
    test_type, n_nodes, density, n_perms, homophily_strength, seed = args

    np.random.seed(seed)

    if test_type == "H1":
        # Test H1: Independent Homophily
        # X has homophily based on AX, Y has homophily based on AY
        # AX and AY are independent, so X and Y should be uncorrelated

        AX = np.random.uniform(0, 1, n_nodes)
        AY = np.random.uniform(0, 1, n_nodes)

        X = generate_homophily_matrix(n_nodes, AX, density, homophily_strength)
        Y = generate_homophily_matrix(n_nodes, AY, density, homophily_strength)

        # No controls needed - X and Y should be uncorrelated
        X_list = [X]

        # Run QAP
        result = qap_dsp_regression(Y, X_list, nperm=n_perms)

        # Compute homophily correlations
        hom_X = compute_homophily_correlation(X, AX)
        hom_Y = compute_homophily_correlation(Y, AY)

        return {
            'test_type': test_type,
            'p_values': result['p_values'],
            'coefficients': result['coefficients'],
            'r_squared': result['r_squared'],
            'homophily_X': hom_X,
            'homophily_Y': hom_Y,
            'dens_X': np.sum(X) / (n_nodes * (n_nodes - 1)),
            'dens_Y': np.sum(Y) / (n_nodes * (n_nodes - 1))
        }

    elif test_type == "H2a":
        # Test H2a: Shared Attribute with Sender/Receiver Controls
        # Both X and Y use same attribute A for homophily
        # Control for AC, AR, AC*AR

        A = np.random.uniform(0, 1, n_nodes)

        X = generate_homophily_matrix(n_nodes, A, density, homophily_strength)
        Y = generate_homophily_matrix(n_nodes, A, density, homophily_strength)

        # Create control matrices
        controls = create_control_matrices(A)
        AC = controls['AC']
        AR = controls['AR']
        AC_AR = controls['AC_AR']

        # X_list: [X, AC, AR, AC*AR]
        # After controlling for A effects, X should not predict Y
        X_list = [X, AC, AR, AC_AR]

        result = qap_dsp_regression(Y, X_list, nperm=n_perms)

        hom_X = compute_homophily_correlation(X, A)
        hom_Y = compute_homophily_correlation(Y, A)

        # Raw correlation between X and Y (before controls)
        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)
        raw_corr = np.corrcoef(x_vec, y_vec)[0, 1]

        return {
            'test_type': test_type,
            'p_values': result['p_values'],
            'coefficients': result['coefficients'],
            'r_squared': result['r_squared'],
            'homophily_X': hom_X,
            'homophily_Y': hom_Y,
            'raw_XY_corr': raw_corr,
            'dens_X': np.sum(X) / (n_nodes * (n_nodes - 1)),
            'dens_Y': np.sum(Y) / (n_nodes * (n_nodes - 1))
        }

    elif test_type == "H2b":
        # Test H2b: Shared Attribute with Similarity Control
        # Both X and Y use same attribute A for homophily
        # Control for AS = |A(i) - A(j)|

        A = np.random.uniform(0, 1, n_nodes)

        X = generate_homophily_matrix(n_nodes, A, density, homophily_strength)
        Y = generate_homophily_matrix(n_nodes, A, density, homophily_strength)

        # Create control matrices
        controls = create_control_matrices(A)
        AS = controls['AS']

        # X_list: [X, AS]
        # After controlling for similarity, X should not predict Y
        X_list = [X, AS]

        result = qap_dsp_regression(Y, X_list, nperm=n_perms)

        hom_X = compute_homophily_correlation(X, A)
        hom_Y = compute_homophily_correlation(Y, A)

        # Raw correlation between X and Y (before controls)
        x_vec = vectorize_symmetric(X)
        y_vec = vectorize_symmetric(Y)
        raw_corr = np.corrcoef(x_vec, y_vec)[0, 1]

        return {
            'test_type': test_type,
            'p_values': result['p_values'],
            'coefficients': result['coefficients'],
            'r_squared': result['r_squared'],
            'homophily_X': hom_X,
            'homophily_Y': hom_Y,
            'raw_XY_corr': raw_corr,
            'dens_X': np.sum(X) / (n_nodes * (n_nodes - 1)),
            'dens_Y': np.sum(Y) / (n_nodes * (n_nodes - 1))
        }

    else:
        raise ValueError(f"Unknown test type: {test_type}")


def run_homophily_tests(n_simulations: int = 200,
                        n_permutations: int = 1000,
                        n_nodes: int = 50,
                        density: float = 0.15,
                        homophily_strength: float = 3.0,
                        use_parallel: bool = True,
                        n_cores: int = None,
                        seed: int = 42) -> dict:
    """
    Run complete homophily test suite.

    Args:
        n_simulations: Number of simulations per test
        n_permutations: Number of permutations per QAP test
        n_nodes: Number of nodes in networks
        density: Target edge density
        homophily_strength: Strength of homophily effect
        use_parallel: Use parallel processing
        n_cores: Number of cores (None = auto-detect)
        seed: Random seed

    Returns:
        Dictionary with results for all tests
    """
    np.random.seed(seed)

    if use_parallel:
        if n_cores is None:
            n_cores = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_cores = 1

    print()
    print("=" * 80)
    print("HOMOPHILY TESTS (Tests H1, H2a, H2b)")
    print("=" * 80)
    print()
    print("PURPOSE: Test QAP robustness against homophily-based network structure")
    print()
    print("SIMULATION PARAMETERS:")
    print(f"  Simulations per test: {n_simulations}")
    print(f"  Permutations per QAP: {n_permutations}")
    print(f"  Network size: {n_nodes} nodes")
    print(f"  Target density: {density:.2f}")
    print(f"  Homophily strength: {homophily_strength:.1f}")
    print(f"  Parallel processing: {'YES' if use_parallel else 'NO'} ({n_cores} cores)")
    print()

    # Expected CI for alpha = 0.05
    se_alpha = np.sqrt(0.05 * 0.95 / n_simulations)
    ci_lower = 0.05 - 1.96 * se_alpha
    ci_upper = 0.05 + 1.96 * se_alpha
    print(f"EXPECTED TYPE 1 ERROR RATE: 0.05")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

    # Test definitions
    test_info = {
        "H1": {
            "name": "Test H1: Independent Homophily (AX ⊥ AY)",
            "description": "X based on AX, Y based on AY (independent attributes)",
            "coef_name": "X"
        },
        "H2a": {
            "name": "Test H2a: Shared Attribute + AC/AR/AC*AR Controls",
            "description": "X and Y both based on A, controlling for sender/receiver effects",
            "coef_name": "X (controlling for AC, AR, AC*AR)"
        },
        "H2b": {
            "name": "Test H2b: Shared Attribute + Similarity Control",
            "description": "X and Y both based on A, controlling for |A(i)-A(j)|",
            "coef_name": "X (controlling for |A(i)-A(j)|)"
        }
    }

    all_results = {}
    summary_data = []

    for test_type in ["H1", "H2a", "H2b"]:
        info = test_info[test_type]

        print("-" * 80)
        print(info["name"])
        print("-" * 80)
        print(f"Description: {info['description']}")
        print()

        start_time = time.time()

        # Generate seeds
        sim_seeds = [seed + hash(test_type) % 10000 + i for i in range(n_simulations)]
        args_list = [(test_type, n_nodes, density, n_permutations, homophily_strength, s)
                     for s in sim_seeds]

        if use_parallel and n_cores > 1:
            print(f"Running {n_simulations} simulations in parallel...")

            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results_list = list(executor.map(run_single_homophily_test, args_list))
        else:
            print(f"Running {n_simulations} simulations sequentially...")
            results_list = []
            for i, args in enumerate(args_list):
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{n_simulations}", flush=True)
                results_list.append(run_single_homophily_test(args))

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed / 60:.2f} minutes")
        print()

        # Extract p-values for X coefficient (index 1)
        p_vals_X = [r['p_values'][1] for r in results_list]
        type1_X = np.mean(np.array(p_vals_X) < 0.05)

        print("TYPE 1 ERROR RATE:")
        status_X = "(OK)" if ci_lower <= type1_X <= ci_upper else "(WARNING)"
        print(f"  {info['coef_name']}: {type1_X:.4f} {status_X}")

        # P-value uniformity
        print()
        print("P-VALUE UNIFORMITY (Kolmogorov-Smirnov test):")
        ks_stat, ks_pval = stats.kstest(p_vals_X, 'uniform')
        uniform_status = "(uniform)" if ks_pval > 0.05 else "(NOT uniform)"
        print(f"  X: KS statistic = {ks_stat:.4f}, p-value = {ks_pval:.4f} {uniform_status}")

        # Network properties
        print()
        print("ACHIEVED NETWORK PROPERTIES:")
        mean_hom_X = np.mean([r['homophily_X'] for r in results_list])
        mean_hom_Y = np.mean([r['homophily_Y'] for r in results_list])
        mean_dens_X = np.mean([r['dens_X'] for r in results_list])
        mean_dens_Y = np.mean([r['dens_Y'] for r in results_list])

        print(f"  Density X: {mean_dens_X:.3f}, Y: {mean_dens_Y:.3f}")
        print(f"  Homophily correlation X: {mean_hom_X:.3f} (negative = homophily)")
        print(f"  Homophily correlation Y: {mean_hom_Y:.3f}")

        if test_type in ["H2a", "H2b"]:
            mean_raw_corr = np.mean([r['raw_XY_corr'] for r in results_list])
            print(f"  Raw X-Y correlation (before controls): {mean_raw_corr:.3f}")

        print()

        # Store summary
        status = "OK" if ci_lower <= type1_X <= ci_upper else "INFLATED"
        if type1_X < ci_lower:
            status = "DEFLATED"

        summary_data.append({
            'Test': test_type,
            'Type1_Error': type1_X,
            'KS_pvalue': ks_pval,
            'Status': status
        })

        all_results[test_type] = {
            'test_name': info['name'],
            'p_values': p_vals_X,
            'results_list': results_list,
            'runtime': elapsed
        }

    # Summary table
    print("=" * 80)
    print("SUMMARY: HOMOPHILY TEST RESULTS")
    print("=" * 80)
    print()

    print(f"{'Test':<10} {'Type1_Error':<12} {'KS_pvalue':<12} {'Status':<10}")
    print("-" * 44)
    for row in summary_data:
        print(f"{row['Test']:<10} {row['Type1_Error']:<12.4f} {row['KS_pvalue']:<12.4f} {row['Status']:<10}")

    print()
    print(f"Expected range for Type 1 error: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

    # Conclusions
    all_ok = all(row['Status'] == 'OK' for row in summary_data)

    if all_ok:
        print("CONCLUSION: QAP is robust against homophily.")
        print("All tests show proper Type 1 error control.")
    else:
        print("CONCLUSION: Some tests show issues.")
        for row in summary_data:
            if row['Status'] != 'OK':
                print(f"  - {row['Test']}: {row['Status']} (Type 1 error = {row['Type1_Error']:.4f})")

    print()
    print("Homophily tests complete.")

    return {
        'summary': summary_data,
        'all_results': all_results,
        'parameters': {
            'n_simulations': n_simulations,
            'n_permutations': n_permutations,
            'n_nodes': n_nodes,
            'density': density,
            'homophily_strength': homophily_strength
        },
        'ci': (ci_lower, ci_upper)
    }


###############################################################################
# MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QAP Homophily Tests')
    parser.add_argument('--sims', type=int, default=200,
                        help='Number of simulations per test (default: 200)')
    parser.add_argument('--perms', type=int, default=1000,
                        help='Number of permutations per QAP test (default: 1000)')
    parser.add_argument('--nodes', type=int, default=50,
                        help='Number of nodes in networks (default: 50)')
    parser.add_argument('--density', type=float, default=0.15,
                        help='Target edge density (default: 0.15)')
    parser.add_argument('--homophily', type=float, default=3.0,
                        help='Homophily strength parameter (default: 3.0)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially instead of in parallel')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of cores for parallel processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    results = run_homophily_tests(
        n_simulations=args.sims,
        n_permutations=args.perms,
        n_nodes=args.nodes,
        density=args.density,
        homophily_strength=args.homophily,
        use_parallel=not args.sequential,
        n_cores=args.cores,
        seed=args.seed
    )
