#!/usr/bin/env python3
"""
Run only Tests 5 and 6: Degree Heterogeneity Tests

Test 5: Degree-heterogeneous X, Random Y
Test 6: Degree-heterogeneous X, Degree-heterogeneous Y
"""

import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

# Import from the main diagnostic file
from qap_diagnostic_tests import (
    run_single_test,
    generate_degree_heterogeneous_matrix,
    generate_random_matrix,
    calculate_transitivity
)

def run_degree_het_tests(n_simulations: int = 200,
                         n_permutations: int = 1000,
                         n_nodes: int = 50,
                         density: float = 0.15,
                         use_parallel: bool = True,
                         n_cores: int = None,
                         seed: int = 42):
    """Run only degree heterogeneity tests (5 and 6)."""

    np.random.seed(seed)

    test_names = {
        5: "Test 5: Degree-het X, Random Y",
        6: "Test 6: Degree-het X, Degree-het Y"
    }

    if use_parallel:
        if n_cores is None:
            n_cores = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_cores = 1

    print()
    print("=" * 80)
    print("DEGREE HETEROGENEITY TESTS (Tests 5 & 6)")
    print("=" * 80)
    print()
    print("PURPOSE: Test QAP robustness against degree heterogeneity")
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

    all_results = {}
    summary_data = []

    for test_type in [5, 6]:
        print("-" * 80)
        print(test_names[test_type])
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
        p_vals_X = [r['p_values'][1] for r in results_list]
        type1_X = np.mean(np.array(p_vals_X) < 0.05)

        print("TYPE 1 ERROR RATE:")
        status_X = "(OK)" if ci_lower <= type1_X <= ci_upper else "(WARNING)"
        print(f"  X coefficient: {type1_X:.4f} {status_X}")

        # P-value uniformity test
        print()
        print("P-VALUE UNIFORMITY (Kolmogorov-Smirnov test):")
        ks_stat, ks_pval = stats.kstest(p_vals_X, 'uniform')
        uniform_status = "(uniform)" if ks_pval > 0.05 else "(NOT uniform)"
        print(f"  X: KS statistic = {ks_stat:.4f}, p-value = {ks_pval:.4f} {uniform_status}")

        # Status
        status = "OK" if ci_lower <= type1_X <= ci_upper else "INFLATED"
        if type1_X < ci_lower:
            status = "DEFLATED"

        summary_data.append({
            'Test': f'Test {test_type}',
            'Type1_Error': type1_X,
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

        all_results[test_type] = {
            'test_name': test_names[test_type],
            'p_values': p_vals_X,
            'results_list': results_list,
            'runtime': elapsed
        }

    # Summary
    print("=" * 80)
    print("SUMMARY: DEGREE HETEROGENEITY TEST RESULTS")
    print("=" * 80)
    print()

    print(f"{'Test':<35} {'Type1_Error':<12} {'KS_pvalue':<12} {'Status':<10}")
    print("-" * 70)
    for row in summary_data:
        print(f"{row['Test']:<35} {row['Type1_Error']:<12.4f} {row['KS_pvalue']:<12.4f} {row['Status']:<10}")

    print()
    print(f"Expected range for Type 1 error: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

    # Diagnosis
    inflated = [r for r in summary_data if r['Status'] == 'INFLATED']
    if not inflated:
        print("CONCLUSION: QAP is robust against degree heterogeneity.")
        print("Both tests show proper Type 1 error control.")
    else:
        print("WARNING: Inflated Type 1 error rates detected!")
        for r in inflated:
            print(f"  - {r['Test']}: {r['Type1_Error']:.4f}")

    print()

    return {
        'summary': summary_data,
        'all_results': all_results,
        'ci': (ci_lower, ci_upper)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run QAP Degree Heterogeneity Tests (5 & 6)')
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

    results = run_degree_het_tests(
        n_simulations=args.sims,
        n_permutations=args.perms,
        n_nodes=args.nodes,
        density=args.density,
        use_parallel=not args.sequential,
        n_cores=args.cores,
        seed=args.seed
    )

    print("Degree heterogeneity tests complete.")
