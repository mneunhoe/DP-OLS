sample_sizes <- c(100, 250, 500, 1000, 1500, 2000, 10000)

n_list <- list()

for(samp in sample_sizes) {

  dat_samp <- dat[sample(nrow(dat), samp),]
  # Model Formula ff
ff <-
  y ~ x1 + x2 + x1*x2

# Create model frame from formula and data
mf <- model.frame(ff, data = dat_samp)

# Transform model frame to numeric matrix with y in the first column and
# Intercept in the second column.
data_y <- model.response(mf, "numeric")
data_x <- model.matrix(ff, data = mf)
df <- cbind(data_y, data_x)

# Calculate "covariance matrix" t(data) %*% data
covariance <- t(df) %*% df

# Which column contains y?
y_col <- 1

# Which column contains the column of 1s?
int_col <- which(apply(df, 2, function(x)
  all(x == 1)))


# N is 1'1
n <- covariance[int_col, int_col]

# Function to calculate the sensitivity of the covariance matrix
# slightly adapted from PSIlence Library
covariance_sensitivity <- function(n, rng, intercept = 2) {
  diffs <- apply(rng, 1, diff)
  if (!is.null(intercept)) {
    diffs[intercept] <- 1
  }
  sensitivity <- c()
  for (i in 1:length(diffs)) {
    for (j in i:length(diffs)) {
      s <- ((n - 1) / n) * diffs[i] * diffs[j]
      sensitivity <- c(sensitivity, s)
    }
  }
  return(sensitivity)
}

# Set ranges for every data column
rng <- matrix(NA,
              nrow = k + 1,
              ncol = 2,
              byrow = TRUE)

# Calculate ranges from min and max
for (i in 1:ncol(df)) {
  rng[i,] <- c(min(df[, i]), max(df[, i]))
}

# Calculate sensitivities
sens <- covariance_sensitivity(n = n, rng = rng)

# Set interesting values of epsilon
epsilons <- c(0.01, 0.1, 0.3, 0.5, 0.7, 1, 5, 100)

# Set number of repetitions
n_rep <- 100

# Create arrays to collect results
res <- array(NA, dim = c(n_rep, d - 1, length(epsilons)))
res_se <- array(NA, dim = c(n_rep, d - 1, length(epsilons)))



# Loop over values of epsilon and repetitions
for (epsilon in 1:length(epsilons)) {
  for (rep in 1:n_rep) {
    # Laplace Noise as implemented in the PSIlence R-library
    noise_scale <- sens / epsilons[epsilon]
    raw_L <-
      dpNoise(n = length(noise_scale),
              scale = noise_scale,
              dist = 'laplace')
    
    # Put the noise values in the right places in a d x d matrix
    L <- matrix(NA, nrow = d, ncol = d)
    L[lower.tri(L, diag = T)] <- raw_L
    L[upper.tri(L)] <- t(L)[upper.tri(L)]
    
    # Calculate noise covariance matrix
    dp_cov <- (covariance + L)
    
    # Run regression on noisy covariance matrix
    res[rep, , epsilon] <- ols_cov(dp_cov)[, 1]
    res_se[rep, , epsilon] <- ols_cov(dp_cov)[, 2]
  }
}


n_list[[paste0(samp)]] <- list(coef = res, se = res_se)






pdf(
  file = paste0("figures/dp_ols_n", n, ".pdf"),
  width = 16,
  height = 9
)
par(mfrow = c(2, 4))
for (epsilon in 1:length(epsilons)) {
  plot(
    dat_samp$x1,
    dat_samp$y,
    pch = 19,
    col = adjustcolor("black", 0.8),
    xlab = "x1",
    ylab = "y",
    main = paste0("N = ", n, ". Epsilon = ", epsilons[epsilon]),
    las = 1,
    bty = "n"
  )
  
  for (rep in 1:n_rep) {
    abline(
      a = res[rep, , epsilon][1],
      b = res[rep, , epsilon][2],
      col = adjustcolor("maroon3", alpha = 0.2),
      lwd = 2
    )
  }
  abline(
    a = ols_cov(covariance)[1, 1],
    b = ols_cov(covariance)[2, 1],
    col = adjustcolor("grey", alpha = 0.7),
    lwd = 2
  )
  abline(v = 0,
         h = 0,
         col = "grey",
         lty = "dashed")
}
dev.off()
}

mean_bias <- apply(n_list[[paste0(samp)]]$coef - 1, c(2, 3), mean)
lo_bias <- apply(n_list[[paste0(samp)]]$coef - 1, c(2, 3), quantile, 0.025)
hi_bias <- apply(n_list[[paste0(samp)]]$coef - 1, c(2, 3), quantile, 0.975)

plot(mean_bias[,1], 1:4)

