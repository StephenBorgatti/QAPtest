################################################################################
# QAP Simulation Results Visualization and Reporting
################################################################################
#
# This script generates diagnostic plots and detailed reports from the
# QAP simulation results.
#
# Usage:
#   source("qap_visualization.R")
#   results <- readRDS("qap_simulation_results.rds")
#   generate_qap_report(results)
#
################################################################################

#' Generate Complete QAP Simulation Report with Plots
#'
#' @param results Results object from run_qap_simulation()
#' @param output_dir Directory to save plots (default: current directory)
#' @param save_plots Whether to save plots to files (default: TRUE)
generate_qap_report <- function(results, output_dir = ".", save_plots = TRUE) {

  df <- results$results_df
  params <- results$parameters

  cat("\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("QAP SIMULATION DETAILED REPORT\n")
  cat(paste(rep("=", 80), collapse=""), "\n\n")

  # Create output directory if it doesn't exist
  if (save_plots && !dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  ##############################################################################
  # PLOT 1: P-Value Distributions
  ##############################################################################

  if (save_plots) {
    pdf(file.path(output_dir, "qap_pvalue_distributions.pdf"), width = 12, height = 10)
  }

  par(mfrow = c(3, 2), mar = c(4, 4, 3, 1), oma = c(2, 2, 3, 1))

  # P-values for each coefficient
  p_cols <- c("p_b0", "p_b1_X", "p_b2_A", "p_b3_B", "p_b4_SameCat")
  titles <- c("Intercept (b0)", "X Coefficient (b1)", "A Coefficient (b2)",
              "B Coefficient (b3)", "SameCat Coefficient (b4)")
  colors <- c("gray70", "steelblue", "coral", "seagreen", "purple")

  for (i in seq_along(p_cols)) {
    p_vals <- df[[p_cols[i]]]

    hist(p_vals, breaks = 20, probability = TRUE,
         main = paste("P-values:", titles[i]),
         xlab = "P-value", ylab = "Density",
         col = colors[i], border = "white",
         xlim = c(0, 1))

    # Expected uniform line
    abline(h = 1, col = "red", lty = 2, lwd = 2)

    # Add KS test result
    ks_test <- ks.test(p_vals, "punif")
    text(0.7, max(hist(p_vals, breaks = 20, plot = FALSE)$density) * 0.9,
         sprintf("KS p = %.3f", ks_test$p.value), cex = 0.8)

    # Add significance rate
    sig_rate <- mean(p_vals < 0.05)
    text(0.7, max(hist(p_vals, breaks = 20, plot = FALSE)$density) * 0.7,
         sprintf("Sig rate = %.3f", sig_rate), cex = 0.8)
  }

  # Empty plot for legend
  plot.new()
  legend("center",
         legend = c("Observed distribution", "Expected (uniform)"),
         col = c("gray50", "red"), lty = c(1, 2), lwd = 2,
         bty = "n", cex = 1.2)

  mtext("P-Value Distributions (Should be Uniform Under Null)", outer = TRUE,
        cex = 1.2, line = 1)

  if (save_plots) dev.off()

  ##############################################################################
  # PLOT 2: ECDF Plots (comparing to uniform)
  ##############################################################################

  if (save_plots) {
    pdf(file.path(output_dir, "qap_pvalue_ecdf.pdf"), width = 12, height = 10)
  }

  par(mfrow = c(3, 2), mar = c(4, 4, 3, 1), oma = c(2, 2, 3, 1))

  for (i in seq_along(p_cols)) {
    p_vals <- df[[p_cols[i]]]

    plot(ecdf(p_vals),
         main = paste("ECDF:", titles[i]),
         xlab = "P-value", ylab = "Cumulative Probability",
         col = colors[i], lwd = 2)

    # Diagonal line (expected for uniform)
    abline(0, 1, col = "red", lty = 2, lwd = 2)

    legend("topleft",
           legend = c("Observed", "Expected (uniform)"),
           col = c(colors[i], "red"), lty = c(1, 2), lwd = 2,
           bty = "n", cex = 0.8)
  }

  plot.new()

  mtext("Empirical CDFs of P-Values (Should Follow Diagonal)", outer = TRUE,
        cex = 1.2, line = 1)

  if (save_plots) dev.off()

  ##############################################################################
  # PLOT 3: Coefficient Distributions
  ##############################################################################

  if (save_plots) {
    pdf(file.path(output_dir, "qap_coefficient_distributions.pdf"), width = 12, height = 10)
  }

  par(mfrow = c(3, 2), mar = c(4, 4, 3, 1), oma = c(2, 2, 3, 1))

  coef_cols <- c("b0", "b1_X", "b2_A", "b3_B", "b4_SameCat")

  for (i in seq_along(coef_cols)) {
    coefs <- df[[coef_cols[i]]]

    hist(coefs, breaks = 30,
         main = paste("Coefficient:", titles[i]),
         xlab = "Coefficient value", ylab = "Frequency",
         col = colors[i], border = "white")

    # Add mean line
    abline(v = mean(coefs), col = "blue", lwd = 2)
    abline(v = 0, col = "red", lty = 2, lwd = 2)

    legend("topright",
           legend = c(sprintf("Mean = %.4f", mean(coefs)), "Zero"),
           col = c("blue", "red"), lty = c(1, 2), lwd = 2,
           bty = "n", cex = 0.8)
  }

  plot.new()

  mtext("Coefficient Distributions Across Simulations", outer = TRUE,
        cex = 1.2, line = 1)

  if (save_plots) dev.off()

  ##############################################################################
  # PLOT 4: Correlation Before/After Homophily
  ##############################################################################

  if (save_plots) {
    pdf(file.path(output_dir, "qap_correlations.pdf"), width = 10, height = 5)
  }

  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

  # Before homophily
  hist(df$cor_XY_before, breaks = 30,
       main = "Correlation X-Y (Before Homophily)",
       xlab = "Correlation", ylab = "Frequency",
       col = "lightblue", border = "white")
  abline(v = 0, col = "red", lty = 2, lwd = 2)
  abline(v = mean(df$cor_XY_before), col = "blue", lwd = 2)
  legend("topright",
         legend = c(sprintf("Mean = %.4f", mean(df$cor_XY_before)), "Zero"),
         col = c("blue", "red"), lty = c(1, 2), lwd = 2, bty = "n")

  # After homophily
  hist(df$cor_XY_after, breaks = 30,
       main = "Correlation X-Y (After Homophily)",
       xlab = "Correlation", ylab = "Frequency",
       col = "lightcoral", border = "white")
  abline(v = 0, col = "red", lty = 2, lwd = 2)
  abline(v = mean(df$cor_XY_after), col = "blue", lwd = 2)
  legend("topright",
         legend = c(sprintf("Mean = %.4f", mean(df$cor_XY_after)), "Zero"),
         col = c("blue", "red"), lty = c(1, 2), lwd = 2, bty = "n")

  if (save_plots) dev.off()

  ##############################################################################
  # PLOT 5: Network Properties Achieved
  ##############################################################################

  if (save_plots) {
    pdf(file.path(output_dir, "qap_network_properties.pdf"), width = 12, height = 8)
  }

  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

  # Density
  plot(df$density_X, df$density_Y,
       main = "Network Density",
       xlab = "X Density", ylab = "Y Density",
       pch = 20, col = rgb(0, 0, 1, 0.3))
  abline(h = params$density, v = params$density, col = "red", lty = 2)
  legend("topleft", legend = sprintf("Target = %.2f", params$density),
         col = "red", lty = 2, bty = "n")

  # Transitivity
  plot(df$transitivity_X, df$transitivity_Y,
       main = "Network Transitivity (GWESP)",
       xlab = "X Transitivity", ylab = "Y Transitivity",
       pch = 20, col = rgb(0, 0.5, 0, 0.3))
  abline(h = params$transitivity, v = params$transitivity, col = "red", lty = 2)
  legend("topleft", legend = sprintf("Target = %.2f", params$transitivity),
         col = "red", lty = 2, bty = "n")

  # Degree CV
  plot(df$degree_cv_X, df$degree_cv_Y,
       main = "Degree Heterogeneity (GWDegree)",
       xlab = "X Degree CV", ylab = "Y Degree CV",
       pch = 20, col = rgb(0.5, 0, 0.5, 0.3))
  abline(h = params$degree_heterogeneity, v = params$degree_heterogeneity, col = "red", lty = 2)
  legend("topleft", legend = sprintf("Target = %.2f", params$degree_heterogeneity),
         col = "red", lty = 2, bty = "n")

  # Homophily
  plot(df$homophily_X, df$homophily_Y,
       main = "Network Homophily",
       xlab = "X Homophily", ylab = "Y Homophily",
       pch = 20, col = rgb(1, 0.5, 0, 0.3))

  # R-squared distribution
  hist(df$r_squared, breaks = 30,
       main = "R-squared Distribution",
       xlab = "R-squared", ylab = "Frequency",
       col = "lightgreen", border = "white")
  abline(v = mean(df$r_squared), col = "blue", lwd = 2)

  # R-squared p-values
  hist(df$p_rsquared, breaks = 20,
       main = "R-squared P-values",
       xlab = "P-value", ylab = "Frequency",
       col = "lightyellow", border = "white")
  abline(h = params$n_simulations / 20, col = "red", lty = 2, lwd = 2)

  if (save_plots) dev.off()

  ##############################################################################
  # PLOT 6: Summary Panel
  ##############################################################################

  if (save_plots) {
    pdf(file.path(output_dir, "qap_summary_panel.pdf"), width = 14, height = 10)
  }

  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(2, 2, 3, 1))

  # Significance rates bar plot
  sig_rates <- c(
    mean(df$p_b0 < 0.05),
    mean(df$p_b1_X < 0.05),
    mean(df$p_b2_A < 0.05),
    mean(df$p_b3_B < 0.05),
    mean(df$p_b4_SameCat < 0.05)
  )

  bp <- barplot(sig_rates, names.arg = c("b0", "b1(X)", "b2(A)", "b3(B)", "b4(SC)"),
                main = "Significance Rates (p < 0.05)",
                ylab = "Proportion Significant",
                col = colors, ylim = c(0, max(sig_rates) * 1.3))
  abline(h = 0.05, col = "red", lty = 2, lwd = 2)
  text(bp, sig_rates + 0.02, sprintf("%.3f", sig_rates), cex = 0.8)

  # Mean p-values bar plot
  mean_pvals <- c(
    mean(df$p_b0),
    mean(df$p_b1_X),
    mean(df$p_b2_A),
    mean(df$p_b3_B),
    mean(df$p_b4_SameCat)
  )

  bp <- barplot(mean_pvals, names.arg = c("b0", "b1(X)", "b2(A)", "b3(B)", "b4(SC)"),
                main = "Mean P-values",
                ylab = "Mean P-value",
                col = colors, ylim = c(0, 0.7))
  abline(h = 0.5, col = "red", lty = 2, lwd = 2)
  text(bp, mean_pvals + 0.03, sprintf("%.3f", mean_pvals), cex = 0.8)

  # Mean coefficients bar plot
  mean_coefs <- c(
    mean(df$b0),
    mean(df$b1_X),
    mean(df$b2_A),
    mean(df$b3_B),
    mean(df$b4_SameCat)
  )

  bp <- barplot(mean_coefs, names.arg = c("b0", "b1(X)", "b2(A)", "b3(B)", "b4(SC)"),
                main = "Mean Coefficients",
                ylab = "Mean Coefficient",
                col = colors)
  abline(h = 0, col = "red", lty = 2, lwd = 2)

  # Correlation change
  boxplot(list(Before = df$cor_XY_before, After = df$cor_XY_after),
          main = "Correlation X-Y: Before vs After Homophily",
          ylab = "Correlation",
          col = c("lightblue", "lightcoral"))
  abline(h = 0, col = "red", lty = 2)

  # KS test p-values for uniformity
  ks_pvals <- sapply(p_cols, function(col) {
    ks.test(df[[col]], "punif")$p.value
  })

  bp <- barplot(ks_pvals, names.arg = c("b0", "b1(X)", "b2(A)", "b3(B)", "b4(SC)"),
                main = "KS Test P-values (Uniformity)",
                ylab = "KS Test P-value",
                col = ifelse(ks_pvals > 0.05, "lightgreen", "lightcoral"),
                ylim = c(0, 1))
  abline(h = 0.05, col = "red", lty = 2, lwd = 2)
  text(bp, ks_pvals + 0.05, sprintf("%.3f", ks_pvals), cex = 0.8)

  # Network properties summary
  props <- data.frame(
    Property = c("Density", "Transitivity", "Degree CV", "Homophily"),
    Target = c(params$density, params$transitivity,
               params$degree_heterogeneity, NA),
    Mean_X = c(mean(df$density_X), mean(df$transitivity_X),
               mean(df$degree_cv_X), mean(df$homophily_X, na.rm=TRUE)),
    Mean_Y = c(mean(df$density_Y), mean(df$transitivity_Y),
               mean(df$degree_cv_Y), mean(df$homophily_Y, na.rm=TRUE))
  )

  plot.new()
  text(0.5, 0.9, "Network Properties Summary", cex = 1.2, font = 2)
  text(0.1, 0.7, "Property", cex = 0.9, font = 2, adj = 0)
  text(0.4, 0.7, "Target", cex = 0.9, font = 2)
  text(0.6, 0.7, "Mean X", cex = 0.9, font = 2)
  text(0.8, 0.7, "Mean Y", cex = 0.9, font = 2)

  for (i in 1:nrow(props)) {
    y <- 0.6 - (i - 1) * 0.12
    text(0.1, y, props$Property[i], cex = 0.8, adj = 0)
    text(0.4, y, ifelse(is.na(props$Target[i]), "-", sprintf("%.2f", props$Target[i])), cex = 0.8)
    text(0.6, y, sprintf("%.3f", props$Mean_X[i]), cex = 0.8)
    text(0.8, y, sprintf("%.3f", props$Mean_Y[i]), cex = 0.8)
  }

  mtext(sprintf("QAP Simulation Summary (n=%d simulations, %d permutations)",
                params$n_simulations, params$n_permutations),
        outer = TRUE, cex = 1.2, line = 1)

  if (save_plots) dev.off()

  ##############################################################################
  # TEXT REPORT
  ##############################################################################

  cat("DETAILED STATISTICS:\n")
  cat(paste(rep("-", 60), collapse=""), "\n\n")

  cat("SIMULATION PARAMETERS:\n")
  for (name in names(params)) {
    cat(sprintf("  %s: %s\n", name, params[[name]]))
  }
  cat(sprintf("\nRuntime: %.2f minutes\n\n", results$runtime_minutes))

  cat("COEFFICIENT STATISTICS:\n")
  for (i in seq_along(coef_cols)) {
    cat(sprintf("\n%s:\n", titles[i]))
    cat(sprintf("  Mean: %.6f\n", mean(df[[coef_cols[i]]])))
    cat(sprintf("  SD: %.6f\n", sd(df[[coef_cols[i]]])))
    cat(sprintf("  Median: %.6f\n", median(df[[coef_cols[i]]])))
    cat(sprintf("  95%% CI: [%.6f, %.6f]\n",
                quantile(df[[coef_cols[i]]], 0.025),
                quantile(df[[coef_cols[i]]], 0.975)))
  }

  cat("\n\nP-VALUE STATISTICS:\n")
  for (i in seq_along(p_cols)) {
    p_vals <- df[[p_cols[i]]]
    ks_test <- ks.test(p_vals, "punif")

    cat(sprintf("\n%s:\n", titles[i]))
    cat(sprintf("  Mean p-value: %.4f (should be ~0.5)\n", mean(p_vals)))
    cat(sprintf("  Median p-value: %.4f (should be ~0.5)\n", median(p_vals)))
    cat(sprintf("  Prop significant (p < 0.05): %.4f\n", mean(p_vals < 0.05)))
    cat(sprintf("  Prop significant (p < 0.01): %.4f\n", mean(p_vals < 0.01)))
    cat(sprintf("  KS test for uniformity: D=%.4f, p=%.4f\n",
                ks_test$statistic, ks_test$p.value))
  }

  cat("\n\nCORRELATION BETWEEN X AND Y:\n")
  cat(sprintf("  Before homophily: mean=%.4f, SD=%.4f\n",
              mean(df$cor_XY_before), sd(df$cor_XY_before)))
  cat(sprintf("  After homophily: mean=%.4f, SD=%.4f\n",
              mean(df$cor_XY_after), sd(df$cor_XY_after)))
  cat(sprintf("  Change: %.4f\n",
              mean(df$cor_XY_after) - mean(df$cor_XY_before)))

  if (save_plots) {
    cat(sprintf("\n\nPlots saved to: %s/\n", output_dir))
    cat("  - qap_pvalue_distributions.pdf\n")
    cat("  - qap_pvalue_ecdf.pdf\n")
    cat("  - qap_coefficient_distributions.pdf\n")
    cat("  - qap_correlations.pdf\n")
    cat("  - qap_network_properties.pdf\n")
    cat("  - qap_summary_panel.pdf\n")
  }

  cat("\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
  cat("REPORT COMPLETE\n")
  cat(paste(rep("=", 80), collapse=""), "\n")
}


#' Quick Summary of Results
#'
#' @param results Results object from run_qap_simulation()
quick_summary <- function(results) {
  df <- results$results_df

  cat("\nQUICK SUMMARY\n")
  cat(paste(rep("-", 40), collapse=""), "\n")

  cat("\nCorrelation X-Y:\n")
  cat(sprintf("  Before homophily: %.4f\n", mean(df$cor_XY_before)))
  cat(sprintf("  After homophily:  %.4f\n", mean(df$cor_XY_after)))

  cat("\nSignificance rates (p < 0.05):\n")
  cat(sprintf("  b1 (X):       %.3f\n", mean(df$p_b1_X < 0.05)))
  cat(sprintf("  b2 (A):       %.3f\n", mean(df$p_b2_A < 0.05)))
  cat(sprintf("  b3 (B):       %.3f\n", mean(df$p_b3_B < 0.05)))
  cat(sprintf("  b4 (SameCat): %.3f\n", mean(df$p_b4_SameCat < 0.05)))
  cat(sprintf("  R-squared:    %.3f\n", mean(df$p_rsquared < 0.05)))

  cat("\nMean coefficients:\n")
  cat(sprintf("  b1 (X):       %.4f\n", mean(df$b1_X)))
  cat(sprintf("  b4 (SameCat): %.4f\n", mean(df$b4_SameCat)))

  cat("\nMean R-squared: %.4f\n", mean(df$r_squared))
}


################################################################################
# RUN IF EXECUTED AS SCRIPT
################################################################################

if (sys.nframe() == 0) {
  # Check if results file exists
  if (file.exists("qap_simulation_results.rds")) {
    cat("Loading results from qap_simulation_results.rds...\n")
    results <- readRDS("qap_simulation_results.rds")
    generate_qap_report(results, output_dir = "qap_plots")
  } else {
    cat("Results file not found. Please run qap_symmetric_homophily_test.R first.\n")
  }
}
