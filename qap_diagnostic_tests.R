################################################################################
# QAP Type 1 Error Diagnostic Test Suite
################################################################################
#
# Purpose: Systematically test DSP QAP regression under different scenarios
#          to diagnose inflated Type 1 error rates.
#
# Test Scenarios:
#   1. Random X, Random Y - no dyadic dependencies (baseline)
#   2. Two Random IVs (X1, X2), Random Y - multiple predictors
#   3. Random X, Transitive Y - Y has clustering
#   4. Transitive X, Transitive Y - both have clustering
#   5. Degree-heterogeneous X, Random Y - X has hub structure
#   6. Degree-heterogeneous X, Degree-heterogeneous Y - both have hubs
#
# Each test uses homophily_strength = 0, so under the null hypothesis
# the Type 1 error rate should be approximately 0.05.
#
# Author: Claude for borgworld/QAPtest
# Date: 2025-11-29
################################################################################

# Load required packages
required_packages <- c("parallel")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
  library(pkg, character.only = TRUE)
}

################################################################################
# MATRIX GENERATION FUNCTIONS
################################################################################

#' Generate Simple Random Symmetric Matrix (Erdos-Renyi)
#'
#' Creates a purely random symmetric binary matrix with no dyadic dependencies.
#' Each edge exists independently with probability p.
#'
#' @param n Number of nodes
#' @param density Target edge density (probability of each edge)
#' @return Symmetric binary adjacency matrix
generate_random_matrix <- function(n, density = 0.15) {
  # Generate upper triangle with independent Bernoulli trials
  adj <- matrix(0, n, n)
  upper_idx <- which(upper.tri(adj), arr.ind = TRUE)
  n_possible <- nrow(upper_idx)

  # Each edge exists with probability = density
  edges <- rbinom(n_possible, 1, density)
  adj[upper_idx] <- edges
  adj <- adj + t(adj)  # Make symmetric

  adj
}


#' Generate Matrix with Transitivity (Clustering)
#'
#' Creates a symmetric matrix with higher-than-chance transitivity
#' by preferentially closing triangles.
#'
#' @param n Number of nodes
#' @param density Target edge density
#' @param transitivity Target transitivity coefficient (0 to 1)
#' @param max_iter Maximum iterations for adjustment
#' @return Symmetric binary adjacency matrix
generate_transitive_matrix <- function(n, density = 0.15, transitivity = 0.4,
                                        max_iter = 500) {
  # Start with random matrix
  adj <- generate_random_matrix(n, density)

  for (iter in 1:max_iter) {
    current_trans <- calculate_transitivity(adj)
    current_dens <- sum(adj) / (n * (n - 1))

    # Check if we've reached target
    if (abs(current_trans - transitivity) < 0.03 &&
        abs(current_dens - density) < 0.02) {
      break
    }

    # Adjust transitivity
    if (current_trans < transitivity - 0.03) {
      adj <- add_transitive_edge(adj)
    } else if (current_trans > transitivity + 0.03) {
      adj <- remove_transitive_edge(adj)
    }

    # Maintain density
    current_dens <- sum(adj) / (n * (n - 1))
    if (current_dens < density - 0.02) {
      adj <- add_random_edge(adj)
    } else if (current_dens > density + 0.02) {
      adj <- remove_random_edge(adj)
    }
  }

  adj
}


#' Generate Matrix with Degree Heterogeneity
#'
#' Creates a symmetric matrix with heterogeneous degree distribution
#' using preferential attachment mechanism.
#'
#' @param n Number of nodes
#' @param density Target edge density
#' @param degree_cv Target coefficient of variation for degrees
#' @param max_iter Maximum iterations for adjustment
#' @return Symmetric binary adjacency matrix
generate_degree_heterogeneous_matrix <- function(n, density = 0.15,
                                                   degree_cv = 0.5,
                                                   max_iter = 500) {
  # Target number of edges
  n_edges <- round(n * (n - 1) / 2 * density)

  # Initialize with preferential attachment
  adj <- matrix(0, n, n)

  # Start with small connected component
  adj[1, 2] <- adj[2, 1] <- 1
  adj[2, 3] <- adj[3, 2] <- 1
  adj[1, 3] <- adj[3, 1] <- 1

  degrees <- rowSums(adj)

  # Add edges using preferential attachment
  edges_added <- 3
  while (edges_added < n_edges) {
    # Probability proportional to (degree + 1)^2 for stronger heterogeneity
    probs <- (degrees + 1)^2
    probs <- probs / sum(probs)

    # Sample first node
    i <- sample(1:n, 1, prob = probs)

    # For second node, also prefer high degree but not connected to i
    probs2 <- (degrees + 1) * (1 - adj[i, ])
    probs2[i] <- 0

    if (sum(probs2) > 0) {
      probs2 <- probs2 / sum(probs2)
      j <- sample(1:n, 1, prob = probs2)

      if (adj[i, j] == 0) {
        adj[i, j] <- adj[j, i] <- 1
        degrees <- rowSums(adj)
        edges_added <- edges_added + 1
      }
    }
  }

  # Fine-tune to reach target CV
  for (iter in 1:max_iter) {
    current_cv <- sd(rowSums(adj)) / max(mean(rowSums(adj)), 0.001)
    current_dens <- sum(adj) / (n * (n - 1))

    if (abs(current_cv - degree_cv) < 0.05 &&
        abs(current_dens - density) < 0.02) {
      break
    }

    # Maintain density
    if (current_dens < density - 0.02) {
      adj <- add_random_edge(adj)
    } else if (current_dens > density + 0.02) {
      adj <- remove_random_edge(adj)
    }
  }

  adj
}


################################################################################
# HELPER FUNCTIONS
################################################################################

#' Calculate Global Transitivity
calculate_transitivity <- function(adj) {
  n <- nrow(adj)
  triangles <- sum(diag(adj %*% adj %*% adj)) / 6
  degrees <- rowSums(adj)
  triples <- sum(degrees * (degrees - 1)) / 2
  if (triples == 0) return(0)
  3 * triangles / triples
}


#' Add Transitive Edge
add_transitive_edge <- function(adj) {
  n <- nrow(adj)
  for (attempt in 1:20) {
    degrees <- rowSums(adj)
    candidates <- which(degrees >= 2)
    if (length(candidates) == 0) break

    j <- sample(candidates, 1)
    neighbors <- which(adj[j, ] == 1)

    if (length(neighbors) >= 2) {
      pair <- sample(neighbors, 2)
      i <- pair[1]
      k <- pair[2]

      if (adj[i, k] == 0) {
        adj[i, k] <- adj[k, i] <- 1
        return(adj)
      }
    }
  }
  adj
}


#' Remove Transitive Edge
remove_transitive_edge <- function(adj) {
  n <- nrow(adj)
  for (attempt in 1:20) {
    edges <- which(adj == 1 & upper.tri(adj), arr.ind = TRUE)
    if (nrow(edges) == 0) break

    idx <- sample(nrow(edges), 1)
    i <- edges[idx, 1]
    k <- edges[idx, 2]

    common <- which(adj[i, ] == 1 & adj[k, ] == 1)
    if (length(common) > 0) {
      adj[i, k] <- adj[k, i] <- 0
      return(adj)
    }
  }
  adj
}


#' Add Random Edge
add_random_edge <- function(adj) {
  zeros <- which(adj == 0 & upper.tri(adj), arr.ind = TRUE)
  if (nrow(zeros) > 0) {
    idx <- sample(nrow(zeros), 1)
    i <- zeros[idx, 1]
    j <- zeros[idx, 2]
    adj[i, j] <- adj[j, i] <- 1
  }
  adj
}


#' Remove Random Edge
remove_random_edge <- function(adj) {
  ones <- which(adj == 1 & upper.tri(adj), arr.ind = TRUE)
  if (nrow(ones) > 0) {
    idx <- sample(nrow(ones), 1)
    i <- ones[idx, 1]
    j <- ones[idx, 2]
    adj[i, j] <- adj[j, i] <- 0
  }
  adj
}


#' Vectorize Symmetric Matrix (upper triangle only)
vectorize_symmetric <- function(mat) {
  mat[upper.tri(mat)]
}


################################################################################
# DSP QAP REGRESSION (Simplified for Diagnostics)
################################################################################

#' DSP QAP Regression with Robust Standard Errors
#'
#' Simplified version focused on Type 1 error testing.
#'
#' @param Y Dependent variable matrix
#' @param X_list List of predictor matrices
#' @param nperm Number of permutations
#' @return List with coefficients and p-values
qap_dsp_regression <- function(Y, X_list, nperm = 1000) {
  n <- nrow(Y)
  n_x <- length(X_list)

  # Vectorize matrices
  y_vec <- vectorize_symmetric(Y)
  n_obs <- length(y_vec)

  x_vecs <- lapply(X_list, vectorize_symmetric)
  x_mat <- cbind(1, do.call(cbind, x_vecs))
  n_params <- ncol(x_mat)

  # Observed regression
  obs_fit <- lm.fit(x_mat, y_vec)
  obs_coef <- obs_fit$coefficients
  obs_resid <- obs_fit$residuals

  # R-squared
  ss_res <- sum(obs_resid^2)
  ss_tot <- sum((y_vec - mean(y_vec))^2)
  r_squared <- 1 - ss_res / ss_tot

  # Robust standard errors (HC3)
  xtx_inv <- solve(t(x_mat) %*% x_mat)
  hat_matrix <- x_mat %*% xtx_inv %*% t(x_mat)
  h <- diag(hat_matrix)
  u <- obs_resid / (1 - h)

  meat <- t(x_mat) %*% diag(u^2) %*% x_mat
  robust_vcov <- xtx_inv %*% meat %*% xtx_inv
  robust_se <- sqrt(diag(robust_vcov))

  # Observed t-statistics
  obs_tstats <- obs_coef / robust_se

  # Pre-generate permutations
  perms <- lapply(1:nperm, function(p) sample(1:n))

  # DSP Permutation test for each X coefficient
  p_values <- numeric(n_params)

  for (k in 1:n_x) {
    coef_idx <- k + 1

    # Partial out other X's from X_k
    if (n_x > 1) {
      Z_indices <- setdiff(1:n_x, k)
      Z_vecs <- lapply(Z_indices, function(j) x_vecs[[j]])
      Z_mat_for_partial <- cbind(1, do.call(cbind, Z_vecs))

      resid_fit <- lm.fit(Z_mat_for_partial, x_vecs[[k]])
      x_k_resid_vec <- resid_fit$residuals

      x_k_resid_mat <- matrix(0, n, n)
      x_k_resid_mat[upper.tri(x_k_resid_mat)] <- x_k_resid_vec
      x_k_resid_mat <- x_k_resid_mat + t(x_k_resid_mat)
    } else {
      x_k_resid_mat <- X_list[[k]]
    }

    # Permutation test
    perm_tstats_k <- numeric(nperm)

    for (p in 1:nperm) {
      perm_idx <- perms[[p]]

      # Permute X_k residuals
      x_k_perm_mat <- x_k_resid_mat[perm_idx, perm_idx]
      x_k_perm_vec <- vectorize_symmetric(x_k_perm_mat)

      # Build design matrix
      if (n_x > 1) {
        Z_mat <- do.call(cbind, Z_vecs)
        x_perm_design <- cbind(1, x_k_perm_vec, Z_mat)
      } else {
        x_perm_design <- cbind(1, x_k_perm_vec)
      }

      # Fit and compute robust t-stat
      perm_fit <- lm.fit(x_perm_design, y_vec)
      perm_coef <- perm_fit$coefficients
      perm_resid <- perm_fit$residuals

      xtx_inv_perm <- solve(t(x_perm_design) %*% x_perm_design)
      h_perm <- diag(x_perm_design %*% xtx_inv_perm %*% t(x_perm_design))
      u_perm <- perm_resid / (1 - h_perm)
      meat_perm <- t(x_perm_design) %*% diag(u_perm^2) %*% x_perm_design
      vcov_perm <- xtx_inv_perm %*% meat_perm %*% xtx_inv_perm
      se_perm <- sqrt(diag(vcov_perm))

      perm_tstats_k[p] <- perm_coef[2] / se_perm[2]
    }

    # Two-tailed p-value
    p_values[coef_idx] <- mean(abs(perm_tstats_k) >= abs(obs_tstats[coef_idx]))
  }

  # Intercept p-value (permute all X residuals)
  x_resid_mats <- list()
  for (k in 1:n_x) {
    if (n_x > 1) {
      Z_indices <- setdiff(1:n_x, k)
      Z_mat_for_partial <- cbind(1, do.call(cbind, lapply(Z_indices, function(j) x_vecs[[j]])))
      resid_fit <- lm.fit(Z_mat_for_partial, x_vecs[[k]])
      x_k_resid_vec <- resid_fit$residuals
      x_k_resid_mat <- matrix(0, n, n)
      x_k_resid_mat[upper.tri(x_k_resid_mat)] <- x_k_resid_vec
      x_k_resid_mat <- x_k_resid_mat + t(x_k_resid_mat)
      x_resid_mats[[k]] <- x_k_resid_mat
    } else {
      x_resid_mats[[k]] <- X_list[[k]]
    }
  }

  perm_tstats_intercept <- numeric(nperm)
  for (p in 1:nperm) {
    perm_idx <- perms[[p]]
    x_perm_vecs <- lapply(x_resid_mats, function(mat) {
      vectorize_symmetric(mat[perm_idx, perm_idx])
    })
    x_perm_design <- cbind(1, do.call(cbind, x_perm_vecs))

    perm_fit <- lm.fit(x_perm_design, y_vec)
    perm_coef <- perm_fit$coefficients
    perm_resid <- perm_fit$residuals

    xtx_inv_perm <- solve(t(x_perm_design) %*% x_perm_design)
    h_perm <- diag(x_perm_design %*% xtx_inv_perm %*% t(x_perm_design))
    u_perm <- perm_resid / (1 - h_perm)
    meat_perm <- t(x_perm_design) %*% diag(u_perm^2) %*% x_perm_design
    vcov_perm <- xtx_inv_perm %*% meat_perm %*% xtx_inv_perm
    se_perm <- sqrt(diag(vcov_perm))

    perm_tstats_intercept[p] <- perm_coef[1] / se_perm[1]
  }

  p_values[1] <- mean(abs(perm_tstats_intercept) >= abs(obs_tstats[1]))

  list(
    coefficients = obs_coef,
    robust_se = robust_se,
    t_statistics = obs_tstats,
    p_values = p_values,
    r_squared = r_squared
  )
}


################################################################################
# TEST FUNCTIONS
################################################################################

#' Run Single Test Iteration
#'
#' @param test_type Which test scenario (1-6)
#' @param n_nodes Number of nodes
#' @param density Target density
#' @param n_perms Number of permutations
#' @return List with p-values and network statistics
run_single_test <- function(test_type, n_nodes = 50, density = 0.15, n_perms = 1000) {

  # Generate matrices based on test type
  if (test_type == 1) {
    # Test 1: Random X, Random Y
    X <- generate_random_matrix(n_nodes, density)
    Y <- generate_random_matrix(n_nodes, density)
    X_list <- list(X)

  } else if (test_type == 2) {
    # Test 2: Two Random IVs, Random Y
    X1 <- generate_random_matrix(n_nodes, density)
    X2 <- generate_random_matrix(n_nodes, density)
    Y <- generate_random_matrix(n_nodes, density)
    X_list <- list(X1, X2)

  } else if (test_type == 3) {
    # Test 3: Random X, Transitive Y
    X <- generate_random_matrix(n_nodes, density)
    Y <- generate_transitive_matrix(n_nodes, density, transitivity = 0.4)
    X_list <- list(X)

  } else if (test_type == 4) {
    # Test 4: Transitive X, Transitive Y
    X <- generate_transitive_matrix(n_nodes, density, transitivity = 0.4)
    Y <- generate_transitive_matrix(n_nodes, density, transitivity = 0.4)
    X_list <- list(X)

  } else if (test_type == 5) {
    # Test 5: Degree-heterogeneous X, Random Y
    X <- generate_degree_heterogeneous_matrix(n_nodes, density, degree_cv = 0.5)
    Y <- generate_random_matrix(n_nodes, density)
    X_list <- list(X)

  } else if (test_type == 6) {
    # Test 6: Degree-heterogeneous X, Degree-heterogeneous Y
    X <- generate_degree_heterogeneous_matrix(n_nodes, density, degree_cv = 0.5)
    Y <- generate_degree_heterogeneous_matrix(n_nodes, density, degree_cv = 0.5)
    X_list <- list(X)
  }

  # Run QAP regression
  result <- qap_dsp_regression(Y, X_list, nperm = n_perms)

  # Calculate network statistics
  trans_Y <- calculate_transitivity(Y)
  cv_Y <- sd(rowSums(Y)) / max(mean(rowSums(Y)), 0.001)
  dens_Y <- sum(Y) / (n_nodes * (n_nodes - 1))

  trans_X <- calculate_transitivity(X_list[[1]])
  cv_X <- sd(rowSums(X_list[[1]])) / max(mean(rowSums(X_list[[1]])), 0.001)
  dens_X <- sum(X_list[[1]]) / (n_nodes * (n_nodes - 1))

  list(
    p_values = result$p_values,
    coefficients = result$coefficients,
    r_squared = result$r_squared,
    trans_X = trans_X,
    trans_Y = trans_Y,
    cv_X = cv_X,
    cv_Y = cv_Y,
    dens_X = dens_X,
    dens_Y = dens_Y
  )
}


#' Run Diagnostic Test Suite
#'
#' Runs all 6 test scenarios and reports Type 1 error rates.
#'
#' @param n_simulations Number of simulations per test
#' @param n_permutations Number of permutations per QAP test
#' @param n_nodes Number of nodes in networks
#' @param density Target edge density
#' @param use_parallel Use parallel processing
#' @param n_cores Number of cores (NULL = auto-detect)
#' @param seed Random seed
#' @return List with results for all tests
run_diagnostic_suite <- function(n_simulations = 200,
                                  n_permutations = 1000,
                                  n_nodes = 50,
                                  density = 0.15,
                                  use_parallel = TRUE,
                                  n_cores = NULL,
                                  seed = 42) {

  set.seed(seed)

  # Test descriptions
  test_names <- c(
    "Test 1: Random X, Random Y (baseline)",
    "Test 2: Two Random IVs (X1, X2), Random Y",
    "Test 3: Random X, Transitive Y",
    "Test 4: Transitive X, Transitive Y",
    "Test 5: Degree-het X, Random Y",
    "Test 6: Degree-het X, Degree-het Y"
  )

  # Set up parallel processing
  if (use_parallel) {
    if (is.null(n_cores)) {
      n_cores <- max(1, detectCores() - 1)
    }
  } else {
    n_cores <- 1
  }

  cat("\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("QAP TYPE 1 ERROR DIAGNOSTIC TEST SUITE\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  cat("PURPOSE: Identify source of inflated Type 1 error rates\n\n")

  cat("SIMULATION PARAMETERS:\n")
  cat(sprintf("  Simulations per test: %d\n", n_simulations))
  cat(sprintf("  Permutations per QAP: %d\n", n_permutations))
  cat(sprintf("  Network size: %d nodes\n", n_nodes))
  cat(sprintf("  Target density: %.2f\n", density))
  cat(sprintf("  Parallel processing: %s (%d cores)\n\n",
              ifelse(use_parallel, "YES", "NO"), n_cores))

  # Expected CI for alpha = 0.05
  se_alpha <- sqrt(0.05 * 0.95 / n_simulations)
  ci_lower <- 0.05 - 1.96 * se_alpha
  ci_upper <- 0.05 + 1.96 * se_alpha
  cat(sprintf("EXPECTED TYPE 1 ERROR RATE: 0.05\n"))
  cat(sprintf("95%% CI: [%.4f, %.4f]\n\n", ci_lower, ci_upper))

  # Store all results
  all_results <- list()

  # Run each test
  for (test_type in 1:6) {
    cat(paste(rep("-", 80), collapse = ""), "\n")
    cat(sprintf("%s\n", test_names[test_type]))
    cat(paste(rep("-", 80), collapse = ""), "\n")

    start_time <- Sys.time()

    if (use_parallel && n_cores > 1) {
      cat(sprintf("Running %d simulations in parallel...\n", n_simulations))

      cl <- makeCluster(n_cores)

      clusterExport(cl, c(
        "generate_random_matrix", "generate_transitive_matrix",
        "generate_degree_heterogeneous_matrix",
        "calculate_transitivity", "add_transitive_edge",
        "remove_transitive_edge", "add_random_edge", "remove_random_edge",
        "vectorize_symmetric", "qap_dsp_regression",
        "run_single_test", "n_permutations", "n_nodes", "density"
      ), envir = environment())

      clusterSetRNGStream(cl, seed + test_type * 1000)

      results_list <- parLapply(cl, 1:n_simulations, function(i) {
        run_single_test(test_type, n_nodes, density, n_permutations)
      }, test_type = test_type, n_nodes = n_nodes,
         density = density, n_permutations = n_permutations)

      stopCluster(cl)

    } else {
      cat(sprintf("Running %d simulations sequentially...\n", n_simulations))
      cat("Progress: ")

      results_list <- list()
      for (i in 1:n_simulations) {
        if (i %% 20 == 0) {
          cat(sprintf("%d ", i))
          flush.console()
        }
        results_list[[i]] <- run_single_test(test_type, n_nodes, density,
                                              n_permutations)
      }
      cat("\n")
    }

    end_time <- Sys.time()
    runtime <- as.numeric(difftime(end_time, start_time, units = "mins"))

    cat(sprintf("Completed in %.2f minutes\n\n", runtime))

    # Extract p-values
    if (test_type == 2) {
      # Two predictors
      p_vals_X1 <- sapply(results_list, function(r) r$p_values[2])
      p_vals_X2 <- sapply(results_list, function(r) r$p_values[3])
      p_vals <- list(X1 = p_vals_X1, X2 = p_vals_X2)

      type1_X1 <- mean(p_vals_X1 < 0.05)
      type1_X2 <- mean(p_vals_X2 < 0.05)

      cat("TYPE 1 ERROR RATES:\n")
      cat(sprintf("  X1 coefficient: %.4f %s\n", type1_X1,
                  ifelse(type1_X1 >= ci_lower & type1_X1 <= ci_upper, "(OK)", "(WARNING)")))
      cat(sprintf("  X2 coefficient: %.4f %s\n", type1_X2,
                  ifelse(type1_X2 >= ci_lower & type1_X2 <= ci_upper, "(OK)", "(WARNING)")))

    } else {
      # Single predictor
      p_vals_X <- sapply(results_list, function(r) r$p_values[2])
      p_vals <- list(X = p_vals_X)

      type1_X <- mean(p_vals_X < 0.05)

      cat("TYPE 1 ERROR RATE:\n")
      cat(sprintf("  X coefficient: %.4f %s\n", type1_X,
                  ifelse(type1_X >= ci_lower & type1_X <= ci_upper, "(OK)", "(WARNING)")))
    }

    # P-value uniformity test
    cat("\nP-VALUE UNIFORMITY (Kolmogorov-Smirnov test):\n")
    for (name in names(p_vals)) {
      ks_result <- ks.test(p_vals[[name]], "punif")
      cat(sprintf("  %s: KS statistic = %.4f, p-value = %.4f %s\n",
                  name, ks_result$statistic, ks_result$p.value,
                  ifelse(ks_result$p.value > 0.05, "(uniform)", "(NOT uniform)")))
    }

    # Network statistics
    cat("\nACHIEVED NETWORK PROPERTIES:\n")
    trans_X <- mean(sapply(results_list, `[[`, "trans_X"))
    trans_Y <- mean(sapply(results_list, `[[`, "trans_Y"))
    cv_X <- mean(sapply(results_list, `[[`, "cv_X"))
    cv_Y <- mean(sapply(results_list, `[[`, "cv_Y"))
    dens_X <- mean(sapply(results_list, `[[`, "dens_X"))
    dens_Y <- mean(sapply(results_list, `[[`, "dens_Y"))

    cat(sprintf("  Density X: %.3f, Y: %.3f\n", dens_X, dens_Y))
    cat(sprintf("  Transitivity X: %.3f, Y: %.3f\n", trans_X, trans_Y))
    cat(sprintf("  Degree CV X: %.3f, Y: %.3f\n", cv_X, cv_Y))

    cat("\n")

    # Store results
    all_results[[test_type]] <- list(
      test_name = test_names[test_type],
      p_values = p_vals,
      results_list = results_list,
      runtime = runtime
    )
  }

  ##############################################################################
  # SUMMARY TABLE
  ##############################################################################

  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("SUMMARY: TYPE 1 ERROR RATES BY TEST SCENARIO\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  summary_df <- data.frame(
    Test = character(),
    Coefficient = character(),
    Type1_Error = numeric(),
    KS_pvalue = numeric(),
    Status = character(),
    stringsAsFactors = FALSE
  )

  for (test_type in 1:6) {
    p_vals <- all_results[[test_type]]$p_values

    for (name in names(p_vals)) {
      type1 <- mean(p_vals[[name]] < 0.05)
      ks_pval <- ks.test(p_vals[[name]], "punif")$p.value

      status <- ifelse(type1 >= ci_lower & type1 <= ci_upper, "OK", "INFLATED")
      if (type1 < ci_lower) status <- "DEFLATED"

      summary_df <- rbind(summary_df, data.frame(
        Test = sprintf("Test %d", test_type),
        Coefficient = name,
        Type1_Error = round(type1, 4),
        KS_pvalue = round(ks_pval, 4),
        Status = status,
        stringsAsFactors = FALSE
      ))
    }
  }

  print(summary_df, row.names = FALSE)

  cat(sprintf("\nExpected range for Type 1 error: [%.4f, %.4f]\n", ci_lower, ci_upper))

  # Diagnosis
  cat("\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("DIAGNOSIS\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  inflated_tests <- unique(summary_df$Test[summary_df$Status == "INFLATED"])

  if (length(inflated_tests) == 0) {
    cat("All tests show proper Type 1 error control.\n")
  } else {
    cat("Tests with inflated Type 1 error rates:\n")
    for (test in inflated_tests) {
      cat(sprintf("  - %s\n", test))
    }

    cat("\nPOSSIBLE CAUSES:\n")

    # Analyze pattern
    if ("Test 1" %in% inflated_tests) {
      cat("  * Test 1 (baseline) is inflated: Problem is in the core DSP algorithm\n")
    }
    if ("Test 2" %in% inflated_tests && !("Test 1" %in% inflated_tests)) {
      cat("  * Test 2 inflated but not Test 1: Problem with multiple predictors\n")
    }
    if ("Test 3" %in% inflated_tests && !("Test 1" %in% inflated_tests)) {
      cat("  * Test 3 inflated but not Test 1: Transitivity in Y causes inflation\n")
    }
    if ("Test 4" %in% inflated_tests && !("Test 3" %in% inflated_tests)) {
      cat("  * Test 4 inflated but not Test 3: Transitivity in X causes inflation\n")
    }
    if ("Test 5" %in% inflated_tests && !("Test 1" %in% inflated_tests)) {
      cat("  * Test 5 inflated but not Test 1: Degree heterogeneity in X causes inflation\n")
    }
    if ("Test 6" %in% inflated_tests && !("Test 5" %in% inflated_tests)) {
      cat("  * Test 6 inflated but not Test 5: Degree heterogeneity in Y causes inflation\n")
    }
  }

  cat("\n")

  # Return results
  invisible(list(
    summary = summary_df,
    all_results = all_results,
    parameters = list(
      n_simulations = n_simulations,
      n_permutations = n_permutations,
      n_nodes = n_nodes,
      density = density
    ),
    ci = c(ci_lower, ci_upper)
  ))
}


################################################################################
# MAIN EXECUTION
################################################################################

if (sys.nframe() == 0) {
  # Run diagnostic suite
  results <- run_diagnostic_suite(
    n_simulations = 200,
    n_permutations = 1000,
    n_nodes = 50,
    density = 0.15,
    use_parallel = TRUE,
    seed = 42
  )

  # Save results
  saveRDS(results, "qap_diagnostic_results.rds")

  cat("\nResults saved to: qap_diagnostic_results.rds\n")
}
