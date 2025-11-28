# QAP Regression Testing with Symmetric Matrices and Homophily

This repository contains scripts for testing QAP (Quadratic Assignment Procedure) regression using the Double Semi-Partialling (DSP) method (Dekker, Krackhardt & Snijders, 2007) on symmetric networks with dyadic dependencies.

## Overview

The simulation tests the DSP QAP regression model:

```
Y = b0 + b1*X + b2*A + b3*B + b4*SameCat
```

Where:
- **Y**: Dependent symmetric network matrix
- **X**: Independent symmetric network matrix
- **A**: Category A membership matrix (0 if neither node is category A, 1 if one is, 2 if both are)
- **B**: Category B membership matrix (same coding as A)
- **SameCat**: Same category indicator (1 if both nodes share the same category, 0 otherwise)

## Key Features

### Network Generation
- **Symmetric matrices** with configurable density
- **GWESP effects**: Higher than chance transitivity (transitive triples)
- **GWDegree effects**: Heterogeneous degree distribution
- **Homophily**: Increased probability of ties within same-category node pairs

### Regression Method
- **Double Semi-Partialling (DSP)**: Proper permutation test for multiple regression
- **Robust Standard Errors**: Heteroskedasticity-consistent (HC3) standard errors
- **Permutation-based p-values**: Two-tailed tests with configurable permutations

### Simulation Features
- **Parallelized execution** for faster runtime
- **200 simulations** (configurable)
- **1000 permutations per test** (configurable)
- Tracks correlation between X and Y before and after adding homophily

## Files

- `qap_symmetric_homophily_test.R`: Main simulation script
- `qap_visualization.R`: Visualization and detailed reporting

## Usage

### Running the Simulation

```r
source("qap_symmetric_homophily_test.R")

# Run with default parameters
results <- run_qap_simulation(
  n_simulations = 200,
  n_permutations = 1000,
  n_nodes = 50,
  density = 0.15,
  transitivity = 0.3,
  degree_heterogeneity = 0.3,
  homophily_strength = 0.3,
  n_categories = 3,
  use_parallel = TRUE,
  seed = 42
)

# Save results
saveRDS(results, "qap_simulation_results.rds")
write.csv(results$results_df, "qap_simulation_results.csv", row.names = FALSE)
```

### Generating Reports

```r
source("qap_visualization.R")

# Load saved results
results <- readRDS("qap_simulation_results.rds")

# Generate full report with plots
generate_qap_report(results, output_dir = "qap_plots")

# Quick summary
quick_summary(results)
```

## Output

### Results Data Frame

Each row represents one simulation with columns for:
- `b0`, `b1_X`, `b2_A`, `b3_B`, `b4_SameCat`: Coefficient estimates
- `p_b0`, `p_b1_X`, `p_b2_A`, `p_b3_B`, `p_b4_SameCat`: P-values
- `se_b0`, `se_b1_X`, `se_b2_A`, `se_b3_B`, `se_b4_SameCat`: Robust standard errors
- `r_squared`, `p_rsquared`: Model R-squared and its p-value
- `cor_XY_before`, `cor_XY_after`: Correlation between X and Y before/after homophily
- Network properties: `transitivity_*`, `degree_cv_*`, `homophily_*`, `density_*`

### Summary Statistics

- Significance rates (proportion of p < 0.05)
- P-value uniformity analysis (Kolmogorov-Smirnov tests)
- Type I error rate analysis

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_simulations | 200 | Number of independent simulations |
| n_permutations | 1000 | Permutations per QAP test |
| n_nodes | 50 | Network size |
| density | 0.15 | Target edge density |
| transitivity | 0.3 | Target global transitivity (GWESP) |
| degree_heterogeneity | 0.3 | Target coefficient of variation for degrees (GWDegree) |
| homophily_strength | 0.3 | Proportion of edges to swap for homophily |
| n_categories | 3 | Number of attribute categories (A, B, C) |

## Dependencies

- R (>= 4.0)
- parallel (built-in)
- sandwich
- lmtest

## References

Dekker, D., Krackhardt, D., & Snijders, T. A. B. (2007). Sensitivity of MRQAP tests to collinearity and autocorrelation conditions. *Psychometrika*, 72(4), 563-581.
