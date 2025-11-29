################################################################################
# QAP Regression Test with Symmetric Matrices, GWESP, GWDegree, and Homophily
################################################################################
#
# Purpose: Test DSP QAP regression on symmetric networks with:
#   - GWESP effects (transitive triples at higher than chance levels)
#   - GWDegree effects (degree distribution heterogeneity)
#   - Homophily via node attribute Z (3 categories)
#
# Model: Y = b0 + b1*X + b2*A + b3*B + b4*SameCat
#   - A(i,j) = 0 if neither i nor j is category A, 2 if both, 1 if one
#   - B(i,j) similarly for category B
#   - SameCat(i,j) = 1 if Z(i) = Z(j), 0 otherwise
#
# Output:
#   - Data frame with coefficients, p-values, R-squared, correlations
#   - Significance rates and p-value uniformity analysis
#
# Author: Claude for borgworld/QAPtest
# Date: 2025-11-28
################################################################################

# Load required packages
required_packages <- c("parallel", "sandwich", "lmtest")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing package: %s", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
  library(pkg, character.only = TRUE)
}

# Set seed for reproducibility
set.seed(42)

################################################################################
# HELPER FUNCTIONS
################################################################################

#' Generate Categorical Node Attributes
#'
#' @param n Number of nodes
#' @param n_categories Number of categories (default 3)
#' @return Vector of category assignments (1, 2, 3)
generate_node_attributes <- function(n, n_categories = 3) {
  base_size <- n %/% n_categories
  remainder <- n %% n_categories
  group_sizes <- rep(base_size, n_categories)
  if (remainder > 0) {
    group_sizes[1:remainder] <- group_sizes[1:remainder] + 1
  }
  categories <- rep(1:n_categories, times = group_sizes)
  sample(categories)
}


#' Create Same Category Matrix (SameCat)
#'
#' @param attributes Vector of node attributes
#' @return Symmetric binary matrix (1 if same category, 0 otherwise)
create_same_cat_matrix <- function(attributes) {
  n <- length(attributes)
  same_cat <- outer(attributes, attributes, "==") * 1L
  diag(same_cat) <- 0
  same_cat
}


#' Create Category Membership Matrix (A or B)
#'
#' For a given category:
#' - Entry = 2 if both i and j are in that category
#' - Entry = 1 if exactly one of i or j is in that category
#' - Entry = 0 if neither is in that category
#'
#' @param attributes Vector of node attributes
#' @param category Which category (1 = A, 2 = B)
#' @return Symmetric matrix with 0/1/2 coding
create_category_matrix <- function(attributes, category) {
  n <- length(attributes)
  is_cat <- (attributes == category) * 1L

  # Matrix where (i,j) = is_cat[i] + is_cat[j]
  # This gives: 2 if both, 1 if one, 0 if neither
  cat_mat <- outer(is_cat, is_cat, "+")
  diag(cat_mat) <- 0
  cat_mat
}


#' Calculate Global Transitivity for Symmetric Matrix
#'
#' @param adj Symmetric adjacency matrix
#' @return Global transitivity coefficient
calculate_transitivity <- function(adj) {
  n <- nrow(adj)

  # Count closed and open triads
  # For symmetric: a triad i-j-k is closed if adj[i,j]=1, adj[j,k]=1, adj[i,k]=1
  adj2 <- adj %*% adj
  triangles <- sum(diag(adj %*% adj %*% adj)) / 6  # Each triangle counted 6 times

  # Number of connected triples (paths of length 2)
  # This is sum of (degree choose 2) = sum(deg*(deg-1)/2)
  degrees <- rowSums(adj)
  triples <- sum(degrees * (degrees - 1)) / 2

  if (triples == 0) return(0)
  return(3 * triangles / triples)
}


#' Calculate Degree Distribution Statistics
#'
#' @param adj Symmetric adjacency matrix
#' @return List with mean, sd, skewness of degree distribution
calculate_degree_stats <- function(adj) {
  degrees <- rowSums(adj)
  n <- length(degrees)
  mean_deg <- mean(degrees)
  sd_deg <- sd(degrees)

  # Skewness (measure of heterogeneity)
  if (sd_deg > 0) {
    skew <- mean((degrees - mean_deg)^3) / sd_deg^3
  } else {
    skew <- 0
  }

  list(mean = mean_deg, sd = sd_deg, skewness = skew, cv = sd_deg / max(mean_deg, 0.001))
}


#' Generate Symmetric Matrix with GWESP and GWDegree Effects
#'
#' Creates a symmetric binary matrix with:
#' - Higher than chance transitivity (GWESP effect)
#' - Heterogeneous degree distribution (GWDegree effect)
#'
#' @param n Number of nodes
#' @param density Target edge density
#' @param transitivity Target transitivity coefficient
#' @param degree_heterogeneity Target coefficient of variation for degrees
#' @param max_iter Maximum iterations
#' @param tol Tolerance for meeting targets
#' @return Symmetric binary adjacency matrix
generate_symmetric_network <- function(n, density = 0.15,
                                       transitivity = 0.3,
                                       degree_heterogeneity = 0.3,
                                       max_iter = 500, tol = 0.05) {

  # Target number of edges (in upper triangle)
  n_edges <- round(n * (n - 1) / 2 * density)

  # Initialize with preferential attachment-like structure for degree heterogeneity
  adj <- matrix(0, n, n)

  # Create initial edges with preferential attachment
  # Start with a small connected component using RANDOM starting nodes
  # (Fixed bug: previously always used nodes 1,2,3 which caused spurious
  # correlation between independently generated networks)
  start_nodes <- sample(1:n, 3, replace = FALSE)
  a <- start_nodes[1]
  b <- start_nodes[2]
  c <- start_nodes[3]
  adj[a, b] <- adj[b, a] <- 1
  adj[b, c] <- adj[c, b] <- 1

  degrees <- rowSums(adj)

  # Add edges preferentially
  for (e in 4:n_edges) {
    # Choose nodes with probability proportional to (degree + 1)
    probs <- (degrees + 1)
    probs <- probs / sum(probs)

    # Sample first node
    i <- sample(1:n, 1, prob = probs)

    # For second node, prefer nodes with high degree but not connected to i
    probs2 <- (degrees + 1) * (1 - adj[i, ])
    probs2[i] <- 0  # No self-loops

    if (sum(probs2) > 0) {
      probs2 <- probs2 / sum(probs2)
      j <- sample(1:n, 1, prob = probs2)

      adj[i, j] <- adj[j, i] <- 1
      degrees <- rowSums(adj)
    }
  }

  # Now adjust for transitivity using local rewiring
  for (iter in 1:max_iter) {
    current_trans <- calculate_transitivity(adj)
    current_deg_stats <- calculate_degree_stats(adj)

    trans_ok <- abs(current_trans - transitivity) < tol
    deg_ok <- abs(current_deg_stats$cv - degree_heterogeneity) < tol

    if (trans_ok && deg_ok) break

    # Adjust transitivity
    if (current_trans < transitivity - tol) {
      # Add transitive edge: find open triad and close it
      adj <- add_transitive_edge_symmetric(adj)
    } else if (current_trans > transitivity + tol) {
      # Remove an edge from a triangle
      adj <- remove_transitive_edge_symmetric(adj)
    }

    # Maintain density
    current_dens <- sum(adj) / (n * (n - 1))
    if (current_dens < density - 0.02) {
      adj <- add_random_edge_symmetric(adj)
    } else if (current_dens > density + 0.02) {
      adj <- remove_random_edge_symmetric(adj)
    }
  }

  adj
}


#' Add Transitive Edge (Symmetric)
#' @param adj Symmetric adjacency matrix
add_transitive_edge_symmetric <- function(adj) {
  n <- nrow(adj)

  for (attempt in 1:20) {
    # Find a node with at least 2 neighbors
    degrees <- rowSums(adj)
    candidates <- which(degrees >= 2)
    if (length(candidates) == 0) break

    j <- sample(candidates, 1)
    neighbors <- which(adj[j, ] == 1)

    if (length(neighbors) >= 2) {
      # Sample two non-adjacent neighbors
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


#' Remove Transitive Edge (Symmetric)
#' @param adj Symmetric adjacency matrix
remove_transitive_edge_symmetric <- function(adj) {
  n <- nrow(adj)

  for (attempt in 1:20) {
    # Find edges that are part of triangles
    edges <- which(adj == 1 & upper.tri(adj), arr.ind = TRUE)
    if (nrow(edges) == 0) break

    idx <- sample(nrow(edges), 1)
    i <- edges[idx, 1]
    k <- edges[idx, 2]

    # Check if this edge is in a triangle
    common <- which(adj[i, ] == 1 & adj[k, ] == 1)
    if (length(common) > 0) {
      adj[i, k] <- adj[k, i] <- 0
      return(adj)
    }
  }
  adj
}


#' Add Random Edge (Symmetric)
add_random_edge_symmetric <- function(adj) {
  zeros <- which(adj == 0 & upper.tri(adj), arr.ind = TRUE)
  if (nrow(zeros) > 0) {
    idx <- sample(nrow(zeros), 1)
    i <- zeros[idx, 1]
    j <- zeros[idx, 2]
    adj[i, j] <- adj[j, i] <- 1
  }
  adj
}


#' Remove Random Edge (Symmetric)
remove_random_edge_symmetric <- function(adj) {
  ones <- which(adj == 1 & upper.tri(adj), arr.ind = TRUE)
  if (nrow(ones) > 0) {
    idx <- sample(nrow(ones), 1)
    i <- ones[idx, 1]
    j <- ones[idx, 2]
    adj[i, j] <- adj[j, i] <- 0
  }
  adj
}


#' Add Homophily to Existing Symmetric Network
#'
#' Modifies a network to have homophily based on node attributes
#'
#' @param adj Symmetric adjacency matrix
#' @param attributes Node attributes
#' @param homophily_strength How much to increase within-category ties (0-1)
#' @return Modified symmetric adjacency matrix
add_homophily_to_network <- function(adj, attributes, homophily_strength = 0.3) {
  n <- nrow(adj)
  same_cat <- create_same_cat_matrix(attributes)

  # Number of edge swaps
  n_swaps <- round(sum(adj) / 2 * homophily_strength)

  for (swap in 1:n_swaps) {
    # Find a between-category edge
    between_edges <- which(adj == 1 & same_cat == 0 & upper.tri(adj), arr.ind = TRUE)
    if (nrow(between_edges) == 0) break

    # Find a within-category non-edge
    within_nonedges <- which(adj == 0 & same_cat == 1 & upper.tri(adj), arr.ind = TRUE)
    if (nrow(within_nonedges) == 0) break

    # Swap: remove between-category edge, add within-category edge
    idx1 <- sample(nrow(between_edges), 1)
    idx2 <- sample(nrow(within_nonedges), 1)

    i1 <- between_edges[idx1, 1]
    j1 <- between_edges[idx1, 2]
    i2 <- within_nonedges[idx2, 1]
    j2 <- within_nonedges[idx2, 2]

    adj[i1, j1] <- adj[j1, i1] <- 0
    adj[i2, j2] <- adj[j2, i2] <- 1
  }

  adj
}


#' Calculate Homophily (proportion of within-category edges)
#'
#' @param adj Adjacency matrix
#' @param attributes Node attributes
#' @return Proportion of edges within same category
calculate_homophily <- function(adj, attributes) {
  same_cat <- create_same_cat_matrix(attributes)
  within_edges <- sum(adj * same_cat) / 2  # Divide by 2 for symmetric
  total_edges <- sum(adj) / 2
  if (total_edges == 0) return(NA)
  within_edges / total_edges
}


################################################################################
# DSP QAP REGRESSION WITH ROBUST STANDARD ERRORS
################################################################################

#' Vectorize Symmetric Matrix (upper triangle only)
#'
#' @param mat Symmetric matrix
#' @return Vector of upper triangle elements
vectorize_symmetric <- function(mat) {
  mat[upper.tri(mat)]
}


#' Compute Robust Standard Errors Efficiently (HC3)
#'
#' Computes HC3 robust standard errors without forming full hat matrix.
#' Uses the identity: diag(X %*% A %*% t(X)) = rowSums(X * (X %*% A))
#'
#' @param X Design matrix (n_obs x p)
#' @param residuals Regression residuals
#' @param xtx_inv Pre-computed (X'X)^-1
#' @return Vector of robust standard errors
compute_robust_se_fast <- function(X, residuals, xtx_inv) {
  # Efficient computation of hat matrix diagonal
  # h_i = X[i,] %*% xtx_inv %*% X[i,]' = sum(X[i,] * (X %*% xtx_inv)[i,])
  X_xtx_inv <- X %*% xtx_inv
  h <- rowSums(X * X_xtx_inv)

  # HC3 adjustment
  u <- residuals / (1 - h)

  # Efficient meat matrix: X' diag(u^2) X = t(X * u) %*% (X * u) for elementwise
  # Actually: sum_i u_i^2 * X[i,]' X[i,] = t(X) %*% diag(u^2) %*% X
  # Efficient: (X * u)' %*% (X * u) where * is columnwise multiplication
  Xu <- X * u  # Each row of X multiplied by corresponding u
  meat <- crossprod(Xu)  # t(Xu) %*% Xu

  robust_vcov <- xtx_inv %*% meat %*% xtx_inv
  sqrt(diag(robust_vcov))
}


#' DSP QAP Regression with Robust Standard Errors
#'
#' Performs MRQAP regression using double semi-partialling (Dekker et al. 2007)
#' with heteroskedasticity-consistent (robust) standard errors.
#'
#' DSP Method (from Dekker, Krackhardt, Snijders 2007):
#' For testing coefficient of X_k while controlling for Z (other predictors):
#' 1. Compute residuals: eps_XZ = X_k - delta_hat * Z (partial out Z from X_k)
#' 2. Permute these residuals: pi(eps_XZ)
#' 3. Fit: Y = beta * pi(eps_XZ) + gamma * Z + E
#' Key: Y is NOT permuted; only X residuals are permuted
#'
#' @param Y Dependent variable matrix (symmetric)
#' @param X_list List of independent variable matrices (symmetric)
#' @param nperm Number of permutations
#' @return List with coefficients, robust SEs, p-values, R-squared
qap_dsp_regression_robust <- function(Y, X_list, nperm = 1000) {

  n <- nrow(Y)
  n_x <- length(X_list)

  # Vectorize matrices (upper triangle for symmetric)
  y_vec <- vectorize_symmetric(Y)
  n_obs <- length(y_vec)

  # Vectorize all X matrices
  x_vecs <- lapply(X_list, vectorize_symmetric)
  x_mat <- cbind(1, do.call(cbind, x_vecs))
  n_params <- ncol(x_mat)

  # Observed regression
  obs_fit <- lm.fit(x_mat, y_vec)
  obs_coef <- obs_fit$coefficients
  obs_resid <- obs_fit$residuals

  # Calculate R-squared
  ss_res <- sum(obs_resid^2)
  ss_tot <- sum((y_vec - mean(y_vec))^2)
  r_squared <- 1 - ss_res / ss_tot

  # Robust standard errors (HC3) - using fast computation
  xtx_inv <- solve(crossprod(x_mat))
  robust_se <- compute_robust_se_fast(x_mat, obs_resid, xtx_inv)

  # T-statistics with robust SEs (pivotal statistic)
  obs_tstats <- obs_coef / robust_se

  # Pre-generate permutations for consistency across all coefficient tests
  perms <- lapply(1:nperm, function(p) sample(1:n))

  # DSP Permutation test for each coefficient
  # For coefficient k, we:
  # 1. Partial out all other X's (Z) from X_k to get X_k residuals
  # 2. Permute these residuals
  # 3. Regress Y (NOT permuted) on permuted X_k residuals + Z (NOT permuted)

  p_values <- numeric(n_params)
  perm_rsq <- numeric(nperm)

  # For intercept (index 1), compute p-value using permutation of all X residuals
  # For X coefficients (indices 2 to n_params), use proper DSP

  for (k in 1:n_x) {
    # Coefficient index in design matrix (k+1 because of intercept)
    coef_idx <- k + 1

    # Z = all X except X_k (the control variables)
    if (n_x > 1) {
      Z_indices <- setdiff(1:n_x, k)
      Z_vecs <- lapply(Z_indices, function(j) x_vecs[[j]])
      Z_mat_for_partial <- cbind(1, do.call(cbind, Z_vecs))

      # Compute X_k residuals after partialing out Z
      resid_fit <- lm.fit(Z_mat_for_partial, x_vecs[[k]])
      x_k_resid_vec <- resid_fit$residuals

      # Convert residual vector back to matrix form for permutation
      x_k_resid_mat <- matrix(0, n, n)
      x_k_resid_mat[upper.tri(x_k_resid_mat)] <- x_k_resid_vec
      x_k_resid_mat <- x_k_resid_mat + t(x_k_resid_mat)  # Make symmetric
    } else {
      # Only one predictor, no partialing needed
      x_k_resid_mat <- X_list[[k]]
    }

    # Permutation test for coefficient k
    perm_tstats_k <- numeric(nperm)

    for (p in 1:nperm) {
      perm_idx <- perms[[p]]

      # DSP: Permute X_k residuals (rows and columns simultaneously)
      x_k_perm_mat <- x_k_resid_mat[perm_idx, perm_idx]
      x_k_perm_vec <- vectorize_symmetric(x_k_perm_mat)

      # Build design matrix: intercept, permuted X_k residuals, and Z (NOT permuted)
      if (n_x > 1) {
        Z_mat <- do.call(cbind, Z_vecs)
        x_perm_design <- cbind(1, x_k_perm_vec, Z_mat)
      } else {
        x_perm_design <- cbind(1, x_k_perm_vec)
      }

      # Fit regression: Y (NOT permuted) on permuted X_k residuals + Z
      perm_fit <- lm.fit(x_perm_design, y_vec)
      perm_coef <- perm_fit$coefficients
      perm_resid <- perm_fit$residuals

      # Compute robust t-stat using fast method
      xtx_inv_perm <- solve(crossprod(x_perm_design))
      se_perm <- compute_robust_se_fast(x_perm_design, perm_resid, xtx_inv_perm)

      # t-stat for the permuted X_k coefficient (always at index 2 in this design)
      perm_tstats_k[p] <- perm_coef[2] / se_perm[2]

      # Store R-squared for first coefficient's permutations
      if (k == 1) {
        perm_rsq[p] <- 1 - sum(perm_resid^2) / ss_tot
      }
    }

    # Two-tailed p-value for coefficient k based on t-statistics (pivotal)
    p_values[coef_idx] <- mean(abs(perm_tstats_k) >= abs(obs_tstats[coef_idx]))
  }

  # For intercept, use permutation test with all X residuals permuted
  perm_tstats_intercept <- numeric(nperm)

  # Compute residuals for all X's (each partialed from others)
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

  for (p in 1:nperm) {
    perm_idx <- perms[[p]]

    # Permute all X residual matrices with same permutation
    x_perm_vecs <- lapply(x_resid_mats, function(mat) {
      vectorize_symmetric(mat[perm_idx, perm_idx])
    })
    x_perm_design <- cbind(1, do.call(cbind, x_perm_vecs))

    # Fit regression: Y (NOT permuted) on permuted X residuals
    perm_fit <- lm.fit(x_perm_design, y_vec)
    perm_coef <- perm_fit$coefficients
    perm_resid <- perm_fit$residuals

    # Compute robust t-stat for intercept using fast method
    xtx_inv_perm <- solve(crossprod(x_perm_design))
    se_perm <- compute_robust_se_fast(x_perm_design, perm_resid, xtx_inv_perm)

    perm_tstats_intercept[p] <- perm_coef[1] / se_perm[1]
  }

  p_values[1] <- mean(abs(perm_tstats_intercept) >= abs(obs_tstats[1]))

  # P-value for R-squared
  p_value_rsq <- mean(perm_rsq >= r_squared)

  # Parameter names
  param_names <- c("Intercept", paste0("X", 1:(n_params - 1)))

  list(
    coefficients = obs_coef,
    robust_se = robust_se,
    t_statistics = obs_tstats,
    p_values = p_values,
    r_squared = r_squared,
    p_value_rsq = p_value_rsq,
    param_names = param_names,
    nperm = nperm
  )
}


################################################################################
# SINGLE SIMULATION FUNCTION
################################################################################

#' Run Single Simulation
#'
#' @param sim_id Simulation ID
#' @param n_nodes Number of nodes
#' @param n_perms Number of permutations for QAP
#' @param density Target density
#' @param transitivity Target transitivity
#' @param degree_het Target degree heterogeneity (CV)
#' @param homophily_strength Strength of homophily to add
#' @param n_categories Number of attribute categories
#' @return List with all results for this simulation
run_single_simulation <- function(sim_id, n_nodes, n_perms,
                                  density, transitivity, degree_het,
                                  homophily_strength, n_categories = 3) {

  # Generate node attributes (same for X and Y)
  attributes <- generate_node_attributes(n_nodes, n_categories)

  # Generate independent symmetric networks with GWESP/GWDegree effects
  X_base <- generate_symmetric_network(n_nodes, density, transitivity, degree_het)
  Y_base <- generate_symmetric_network(n_nodes, density, transitivity, degree_het)

  # Calculate correlation BEFORE adding homophily
  x_vec_base <- vectorize_symmetric(X_base)
  y_vec_base <- vectorize_symmetric(Y_base)
  cor_before <- cor(x_vec_base, y_vec_base)

  # Add homophily to both X and Y
  X <- add_homophily_to_network(X_base, attributes, homophily_strength)
  Y <- add_homophily_to_network(Y_base, attributes, homophily_strength)

  # Calculate correlation AFTER adding homophily
  x_vec <- vectorize_symmetric(X)
  y_vec <- vectorize_symmetric(Y)
  cor_after <- cor(x_vec, y_vec)

  # Create predictor matrices
  SameCat <- create_same_cat_matrix(attributes)
  A <- create_category_matrix(attributes, 1)  # Category A (1)
  B <- create_category_matrix(attributes, 2)  # Category B (2)

  # Run QAP regression: Y = b0 + b1*X + b2*A + b3*B + b4*SameCat
  qap_result <- qap_dsp_regression_robust(Y, list(X, A, B, SameCat), nperm = n_perms)

  # Network statistics
  trans_X <- calculate_transitivity(X)
  trans_Y <- calculate_transitivity(Y)
  deg_stats_X <- calculate_degree_stats(X)
  deg_stats_Y <- calculate_degree_stats(Y)
  homoph_X <- calculate_homophily(X, attributes)
  homoph_Y <- calculate_homophily(Y, attributes)
  dens_X <- sum(X) / (n_nodes * (n_nodes - 1))
  dens_Y <- sum(Y) / (n_nodes * (n_nodes - 1))

  list(
    sim_id = sim_id,
    # Coefficients (b0, b1, b2, b3, b4)
    b0 = qap_result$coefficients[1],
    b1_X = qap_result$coefficients[2],
    b2_A = qap_result$coefficients[3],
    b3_B = qap_result$coefficients[4],
    b4_SameCat = qap_result$coefficients[5],
    # P-values
    p_b0 = qap_result$p_values[1],
    p_b1_X = qap_result$p_values[2],
    p_b2_A = qap_result$p_values[3],
    p_b3_B = qap_result$p_values[4],
    p_b4_SameCat = qap_result$p_values[5],
    # Robust SEs
    se_b0 = qap_result$robust_se[1],
    se_b1_X = qap_result$robust_se[2],
    se_b2_A = qap_result$robust_se[3],
    se_b3_B = qap_result$robust_se[4],
    se_b4_SameCat = qap_result$robust_se[5],
    # R-squared
    r_squared = qap_result$r_squared,
    p_rsquared = qap_result$p_value_rsq,
    # Correlations
    cor_XY_before = cor_before,
    cor_XY_after = cor_after,
    # Network properties
    transitivity_X = trans_X,
    transitivity_Y = trans_Y,
    degree_cv_X = deg_stats_X$cv,
    degree_cv_Y = deg_stats_Y$cv,
    homophily_X = homoph_X,
    homophily_Y = homoph_Y,
    density_X = dens_X,
    density_Y = dens_Y
  )
}


################################################################################
# MAIN SIMULATION
################################################################################

run_qap_simulation <- function(n_simulations = 200,
                               n_permutations = 1000,
                               n_nodes = 50,
                               density = 0.15,
                               transitivity = 0.3,
                               degree_heterogeneity = 0.3,
                               homophily_strength = 0.3,
                               n_categories = 3,
                               use_parallel = TRUE,
                               n_cores = NULL,
                               seed = 42) {

  set.seed(seed)

  # Set up parallel processing
  if (use_parallel) {
    if (is.null(n_cores)) {
      n_cores <- max(1, detectCores() - 1)
    }
  } else {
    n_cores <- 1
  }

  cat("\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("QAP REGRESSION SIMULATION: SYMMETRIC MATRICES WITH HOMOPHILY\n")
  cat("(DSP Method with Robust Standard Errors)\n")
  cat(paste(rep("=", 80), collapse=""), "\n\n")

  cat("SIMULATION PARAMETERS:\n")
  cat(sprintf("  Number of simulations: %d\n", n_simulations))
  cat(sprintf("  Permutations per QAP test: %d\n", n_permutations))
  cat(sprintf("  Network size: %d nodes\n", n_nodes))
  cat(sprintf("  Target density: %.2f\n", density))
  cat(sprintf("  Target transitivity (GWESP): %.2f\n", transitivity))
  cat(sprintf("  Target degree heterogeneity (GWDegree): %.2f\n", degree_heterogeneity))
  cat(sprintf("  Homophily strength: %.2f\n", homophily_strength))
  cat(sprintf("  Number of categories: %d\n", n_categories))
  cat(sprintf("  Parallel processing: %s (%d cores)\n\n",
              ifelse(use_parallel, "YES", "NO"), n_cores))

  cat("MODEL: Y = b0 + b1*X + b2*A + b3*B + b4*SameCat\n\n")

  start_time <- Sys.time()

  if (use_parallel && n_cores > 1) {
    cat(sprintf("Running %d simulations in parallel...\n", n_simulations))

    cl <- makeCluster(n_cores)

    # Export functions and parameters
    clusterExport(cl, c(
      "generate_node_attributes", "create_same_cat_matrix", "create_category_matrix",
      "calculate_transitivity", "calculate_degree_stats", "calculate_homophily",
      "generate_symmetric_network", "add_homophily_to_network",
      "add_transitive_edge_symmetric", "remove_transitive_edge_symmetric",
      "add_random_edge_symmetric", "remove_random_edge_symmetric",
      "vectorize_symmetric", "qap_dsp_regression_robust",
      "run_single_simulation",
      "n_permutations", "n_nodes", "density", "transitivity",
      "degree_heterogeneity", "homophily_strength", "n_categories"
    ), envir = environment())

    # Set different seeds for each worker
    clusterSetRNGStream(cl, seed)

    results_list <- parLapply(cl, 1:n_simulations, function(i) {
      run_single_simulation(i, n_nodes, n_permutations,
                           density, transitivity, degree_heterogeneity,
                           homophily_strength, n_categories)
    })

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
      results_list[[i]] <- run_single_simulation(i, n_nodes, n_permutations,
                                                  density, transitivity,
                                                  degree_heterogeneity,
                                                  homophily_strength, n_categories)
    }
    cat("\n")
  }

  end_time <- Sys.time()
  runtime <- as.numeric(difftime(end_time, start_time, units = "mins"))

  cat(sprintf("\nSimulation completed in %.2f minutes\n\n", runtime))

  ##############################################################################
  # CREATE RESULTS DATA FRAME
  ##############################################################################

  results_df <- data.frame(
    sim_id = sapply(results_list, `[[`, "sim_id"),
    b0 = sapply(results_list, `[[`, "b0"),
    b1_X = sapply(results_list, `[[`, "b1_X"),
    b2_A = sapply(results_list, `[[`, "b2_A"),
    b3_B = sapply(results_list, `[[`, "b3_B"),
    b4_SameCat = sapply(results_list, `[[`, "b4_SameCat"),
    p_b0 = sapply(results_list, `[[`, "p_b0"),
    p_b1_X = sapply(results_list, `[[`, "p_b1_X"),
    p_b2_A = sapply(results_list, `[[`, "p_b2_A"),
    p_b3_B = sapply(results_list, `[[`, "p_b3_B"),
    p_b4_SameCat = sapply(results_list, `[[`, "p_b4_SameCat"),
    se_b0 = sapply(results_list, `[[`, "se_b0"),
    se_b1_X = sapply(results_list, `[[`, "se_b1_X"),
    se_b2_A = sapply(results_list, `[[`, "se_b2_A"),
    se_b3_B = sapply(results_list, `[[`, "se_b3_B"),
    se_b4_SameCat = sapply(results_list, `[[`, "se_b4_SameCat"),
    r_squared = sapply(results_list, `[[`, "r_squared"),
    p_rsquared = sapply(results_list, `[[`, "p_rsquared"),
    cor_XY_before = sapply(results_list, `[[`, "cor_XY_before"),
    cor_XY_after = sapply(results_list, `[[`, "cor_XY_after"),
    transitivity_X = sapply(results_list, `[[`, "transitivity_X"),
    transitivity_Y = sapply(results_list, `[[`, "transitivity_Y"),
    degree_cv_X = sapply(results_list, `[[`, "degree_cv_X"),
    degree_cv_Y = sapply(results_list, `[[`, "degree_cv_Y"),
    homophily_X = sapply(results_list, `[[`, "homophily_X"),
    homophily_Y = sapply(results_list, `[[`, "homophily_Y"),
    density_X = sapply(results_list, `[[`, "density_X"),
    density_Y = sapply(results_list, `[[`, "density_Y")
  )

  ##############################################################################
  # ANALYZE AND REPORT RESULTS
  ##############################################################################

  cat(paste(rep("-", 80), collapse=""), "\n")
  cat("RESULTS SUMMARY\n")
  cat(paste(rep("-", 80), collapse=""), "\n\n")

  # Correlation analysis
  cat("CORRELATION BETWEEN X AND Y:\n")
  cat(sprintf("  Before homophily: mean = %.4f, SD = %.4f\n",
              mean(results_df$cor_XY_before), sd(results_df$cor_XY_before)))
  cat(sprintf("  After homophily:  mean = %.4f, SD = %.4f\n\n",
              mean(results_df$cor_XY_after), sd(results_df$cor_XY_after)))

  # Network properties achieved
  cat("ACHIEVED NETWORK PROPERTIES:\n")
  cat(sprintf("  Density X:       %.3f +/- %.3f (target: %.2f)\n",
              mean(results_df$density_X), sd(results_df$density_X), density))
  cat(sprintf("  Density Y:       %.3f +/- %.3f (target: %.2f)\n",
              mean(results_df$density_Y), sd(results_df$density_Y), density))
  cat(sprintf("  Transitivity X:  %.3f +/- %.3f (target: %.2f)\n",
              mean(results_df$transitivity_X), sd(results_df$transitivity_X), transitivity))
  cat(sprintf("  Transitivity Y:  %.3f +/- %.3f (target: %.2f)\n",
              mean(results_df$transitivity_Y), sd(results_df$transitivity_Y), transitivity))
  cat(sprintf("  Degree CV X:     %.3f +/- %.3f (target: %.2f)\n",
              mean(results_df$degree_cv_X), sd(results_df$degree_cv_X), degree_heterogeneity))
  cat(sprintf("  Degree CV Y:     %.3f +/- %.3f (target: %.2f)\n",
              mean(results_df$degree_cv_Y), sd(results_df$degree_cv_Y), degree_heterogeneity))
  cat(sprintf("  Homophily X:     %.3f +/- %.3f\n",
              mean(results_df$homophily_X, na.rm=TRUE), sd(results_df$homophily_X, na.rm=TRUE)))
  cat(sprintf("  Homophily Y:     %.3f +/- %.3f\n\n",
              mean(results_df$homophily_Y, na.rm=TRUE), sd(results_df$homophily_Y, na.rm=TRUE)))

  # Coefficient summaries
  cat("COEFFICIENT ESTIMATES:\n")
  cat(sprintf("  b0 (Intercept): mean = %.4f, SD = %.4f\n",
              mean(results_df$b0), sd(results_df$b0)))
  cat(sprintf("  b1 (X):         mean = %.4f, SD = %.4f\n",
              mean(results_df$b1_X), sd(results_df$b1_X)))
  cat(sprintf("  b2 (A):         mean = %.4f, SD = %.4f\n",
              mean(results_df$b2_A), sd(results_df$b2_A)))
  cat(sprintf("  b3 (B):         mean = %.4f, SD = %.4f\n",
              mean(results_df$b3_B), sd(results_df$b3_B)))
  cat(sprintf("  b4 (SameCat):   mean = %.4f, SD = %.4f\n\n",
              mean(results_df$b4_SameCat), sd(results_df$b4_SameCat)))

  # R-squared
  cat("R-SQUARED:\n")
  cat(sprintf("  Mean R-squared: %.4f, SD = %.4f\n",
              mean(results_df$r_squared), sd(results_df$r_squared)))
  cat(sprintf("  Proportion significant (p < 0.05): %.4f\n\n",
              mean(results_df$p_rsquared < 0.05)))

  ##############################################################################
  # SIGNIFICANCE ANALYSIS
  ##############################################################################

  cat(paste(rep("-", 80), collapse=""), "\n")
  cat("SIGNIFICANCE ANALYSIS (alpha = 0.05)\n")
  cat(paste(rep("-", 80), collapse=""), "\n\n")

  sig_rates <- data.frame(
    Coefficient = c("b0 (Intercept)", "b1 (X)", "b2 (A)", "b3 (B)", "b4 (SameCat)", "R-squared"),
    Prop_Significant = c(
      mean(results_df$p_b0 < 0.05),
      mean(results_df$p_b1_X < 0.05),
      mean(results_df$p_b2_A < 0.05),
      mean(results_df$p_b3_B < 0.05),
      mean(results_df$p_b4_SameCat < 0.05),
      mean(results_df$p_rsquared < 0.05)
    )
  )

  print(sig_rates, row.names = FALSE)
  cat("\n")

  ##############################################################################
  # P-VALUE UNIFORMITY ANALYSIS
  ##############################################################################

  cat(paste(rep("-", 80), collapse=""), "\n")
  cat("P-VALUE UNIFORMITY ANALYSIS\n")
  cat(paste(rep("-", 80), collapse=""), "\n\n")

  # For each coefficient, test if p-values are uniformly distributed
  p_val_columns <- c("p_b0", "p_b1_X", "p_b2_A", "p_b3_B", "p_b4_SameCat")
  coef_names <- c("b0 (Intercept)", "b1 (X)", "b2 (A)", "b3 (B)", "b4 (SameCat)")

  uniformity_results <- data.frame(
    Coefficient = character(),
    KS_Statistic = numeric(),
    KS_pvalue = numeric(),
    Mean_pval = numeric(),
    Median_pval = numeric(),
    Uniform = character(),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(p_val_columns)) {
    p_vals <- results_df[[p_val_columns[i]]]

    # Kolmogorov-Smirnov test for uniformity
    ks_test <- ks.test(p_vals, "punif")

    uniformity_results <- rbind(uniformity_results, data.frame(
      Coefficient = coef_names[i],
      KS_Statistic = round(ks_test$statistic, 4),
      KS_pvalue = round(ks_test$p.value, 4),
      Mean_pval = round(mean(p_vals), 4),
      Median_pval = round(median(p_vals), 4),
      Uniform = ifelse(ks_test$p.value > 0.05, "YES", "NO")
    ))
  }

  print(uniformity_results, row.names = FALSE)
  cat("\nNote: 'Uniform = YES' indicates p-values are consistent with uniform distribution\n")
  cat("      (KS test p-value > 0.05, suggesting proper Type I error control)\n\n")

  ##############################################################################
  # TYPE I ERROR RATE ANALYSIS
  ##############################################################################

  cat(paste(rep("-", 80), collapse=""), "\n")
  cat("TYPE I ERROR RATE ANALYSIS\n")
  cat(paste(rep("-", 80), collapse=""), "\n\n")

  # 95% CI for nominal alpha = 0.05
  se_alpha <- sqrt(0.05 * 0.95 / n_simulations)
  ci_lower <- 0.05 - 1.96 * se_alpha
  ci_upper <- 0.05 + 1.96 * se_alpha

  cat(sprintf("Expected 95%% CI for Type I error rate: [%.4f, %.4f]\n\n", ci_lower, ci_upper))

  type1_analysis <- data.frame(
    Coefficient = coef_names,
    Type1_Error = c(
      mean(results_df$p_b0 < 0.05),
      mean(results_df$p_b1_X < 0.05),
      mean(results_df$p_b2_A < 0.05),
      mean(results_df$p_b3_B < 0.05),
      mean(results_df$p_b4_SameCat < 0.05)
    ),
    Within_CI = c(
      mean(results_df$p_b0 < 0.05) >= ci_lower & mean(results_df$p_b0 < 0.05) <= ci_upper,
      mean(results_df$p_b1_X < 0.05) >= ci_lower & mean(results_df$p_b1_X < 0.05) <= ci_upper,
      mean(results_df$p_b2_A < 0.05) >= ci_lower & mean(results_df$p_b2_A < 0.05) <= ci_upper,
      mean(results_df$p_b3_B < 0.05) >= ci_lower & mean(results_df$p_b3_B < 0.05) <= ci_upper,
      mean(results_df$p_b4_SameCat < 0.05) >= ci_lower & mean(results_df$p_b4_SameCat < 0.05) <= ci_upper
    )
  )

  type1_analysis$Status <- ifelse(type1_analysis$Within_CI, "OK", "WARNING")

  print(type1_analysis, row.names = FALSE)
  cat("\n")

  ##############################################################################
  # RETURN RESULTS
  ##############################################################################

  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("SIMULATION COMPLETE\n")
  cat(paste(rep("=", 80), collapse=""), "\n\n")

  # Return results
  list(
    results_df = results_df,
    significance_rates = sig_rates,
    uniformity_analysis = uniformity_results,
    type1_analysis = type1_analysis,
    parameters = list(
      n_simulations = n_simulations,
      n_permutations = n_permutations,
      n_nodes = n_nodes,
      density = density,
      transitivity = transitivity,
      degree_heterogeneity = degree_heterogeneity,
      homophily_strength = homophily_strength,
      n_categories = n_categories
    ),
    runtime_minutes = runtime
  )
}


################################################################################
# RUN SIMULATION (if executed as script)
################################################################################

if (sys.nframe() == 0) {
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

  cat("\nResults saved to:\n")
  cat("  - qap_simulation_results.rds (full R object)\n")
  cat("  - qap_simulation_results.csv (data frame)\n")
}
