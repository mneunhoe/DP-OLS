library(reticulate)
library(tensorflow)
use_condaenv("r-tensorflow")


# tf privacy needs to be installed in the conda environment r-tensorflow prior to running this
privacy <- import("privacy")

# Create population data frame

n <- 100000
x1 <- rnorm(n, 0, 2)
x2 <- rnorm(n, 0, 2)
y <- 1 + 1 * x1 + 1 * x2 + 1 * x1 * x2 + rnorm(n, 0, 1)

dat <- data.frame(y = y, x1 = x1, x2 = x2)
dat_outlier <- rbind(dat, c(10,-4))

plot(
  x1,
  y,
  pch = 19,
  col = adjustcolor("black", alpha = 0.7),
  bty = "n",
  las = 1
)

norm_vec <- function(x) {
  sqrt(sum(x ^ 2))
}

sample_sizes <- c(100, 500, 1000, 2000, 10000)

n_list <- list()
dat_list <- list()
for (samp in rev(sample_sizes)) {
  dat_samp <- dat[sample(nrow(dat), samp),]
  dat_list[[paste0(samp)]] <- dat_samp
  features <-
    as.matrix(cbind(dat_samp[, 2:3], dat_samp[, 2] * dat_samp[, 3]))
  
  true_y <- as.matrix(dat_samp[, 1, drop = F])
  
  
  N <- nrow(dat_samp)
  Dim <- ncol(features)
  
  mb_size <- as.integer(samp %/% 100L)
  learning_rate <- 0.1
  
  epochs <- 5
  repetitions <- 100
  
  # DP declarations
  
  # DP-SGD
  dp <- T
  num_microbatches <- mb_size
  # How to do this with dp?
  l2_norm_clip <-
    1 / 3 * norm_vec(-2 * t(true_y) %*% cbind(features, 1) / nrow(features))
  noise_multiplier <- 2.0
  
  compute_epsilon <- function(steps) {
    orders <- sapply(1:400, function(x)
      1 + x / 10)
    rdp <- privacy$analysis$rdp_accountant$compute_rdp(
      q = mb_size / N,
      noise_multiplier = noise_multiplier,
      steps = steps,
      orders = orders
    )
    eps <-
      privacy$analysis$rdp_accountant$get_privacy_spent(orders = orders,
                                                        rdp = rdp,
                                                        target_delta = 1 / N)[[1]]
    return(eps)
  }
  
  
  
  xavier_init <- function(size) {
    in_dim = size[[1]]
    xavier_stddev = 1. / tf$sqrt(in_dim / 2.)
    return(tf$random_normal(shape = size, stddev = 0.))
  }
  
  
  for (repetition in 1:repetitions) {
    # Set up the variables in the GAN structure
    X <- tf$placeholder(tf$float32, shape = list(NULL, Dim))
    y <- tf$placeholder(tf$float32, shape = list(NULL, 1L))
    
    W1 <- tf$Variable(xavier_init(list(Dim, 1L)), name = "W1")
    b1 <- tf$Variable(tf$zeros(shape = list(1L)), name = "b1")
    
    weights <- list(W1, b1)
    regression <- function(x) {
      pred <- tf$matmul(x, W1) + b1
      return(pred)
    }
    
    pred <- regression(X)
    
    
    if (dp) {
      regression_loss <-
        tf$losses$mean_squared_error(
          labels = y,
          predictions = pred,
          reduction = tf$losses$Reduction$NONE
        )
      optimizer <-
        privacy$optimizers$dp_optimizer$DPAdamGaussianOptimizer(
          learning_rate = learning_rate,
          num_microbatches = num_microbatches,
          l2_norm_clip = l2_norm_clip,
          noise_multiplier = noise_multiplier,
          ledger = FALSE
        )
      
      solver <- optimizer$minimize(
        loss = regression_loss,
        global_step = tf$train$get_global_step(),
        var_list = weights
      )
    } else{
      regression_loss <-
        tf$reduce_mean(tf$losses$mean_squared_error(labels = y, predictions = pred))
      
      solver <-
        tf$train$AdamOptimizer()$minimize(regression_loss, var_list = weights)
    }
    
    val_loss <-
      tf$reduce_mean(tf$losses$mean_squared_error(labels = y, predictions = pred))
    
    sess <- tf$Session()
    sess$run(tf$global_variables_initializer())
    
    steps_per_epoch <- N %/% mb_size
    
    max_step <- epochs * steps_per_epoch
    max_eps <- compute_epsilon(max_step)
    
    res <- list()
    
    for (epoch in 1:epochs) {
      for (step in 1:steps_per_epoch) {
        ind <- sample(nrow(dat_samp), mb_size)
        features_mb <- features[ind, , drop = F]
        true_y_mb <- true_y[ind, , drop = F]
        loss_curr <- sess$run(list(solver, regression_loss),
                              feed_dict = dict(X = features_mb, y = true_y_mb))[[2]]
        
        coefs <- do.call(rbind, sess$run(list(W1, b1)))
        cat("Epoch: ",
            epoch,
            "Loss: ",
            mean(loss_curr),
            "Coefs:",
            coefs,
            "\n\n")
        
        global_step <-
          epoch * steps_per_epoch + step - steps_per_epoch
        if (dp) {
          eps <- compute_epsilon(global_step)
          cat("Epsilon after ", epoch, " epochs is: ", eps, "\n\n")
        }
        
        res[[global_step]] <-
          list(loss = mean(loss_curr),
               coef = coefs,
               eps = eps)
      }
    }
    
    # Housekeeping
    sess$close()
    gc()
    tf$reset_default_graph()
    
    # Store results in list
    n_list[[paste0(samp)]][[paste0(repetition)]] <- res
  }
}

saveRDS(n_list, "n_list.RDS")
saveRDS(dat_list, "dat_list.RDS")

eps_vec <- unlist(lapply(res, function(x)
  x$eps))
val_loss_vec <- unlist(lapply(res, function(x)
  x$val_loss))
plot(1:global_step, eps_vec, ylim = c(0, 1), type = "l")
plot(1:global_step, val_loss_vec, type = "l")



mean_coef <-
  lapply(c(2, 3, 10), function(ratio)
    lapply(sample_sizes, function(samp)
      t(sapply(lapply(1:repetitions, function(z)
        t(sapply(n_list[[paste0(samp)]][[z]], function(x)
          x$coef))), function(x)
            apply(x[(nrow(x) - floor(nrow(x) / ratio) + 1):nrow(x),], 2, mean)))))



500 - (500/10 + 1)
o <-
  lapply(1:5, function(i)
    apply(lapply(sample_sizes, function(samp)
      (
        sapply(lapply(1:repetitions, function(z)
          (
            sapply(n_list[[paste0(samp)]][[z]], function(x)
              x$loss)
          )), function(x)
            x)
      ))[[i]], 2, order))

mean_coef_l <-
  lapply(c(2, 3, 10), function(ratio)
    lapply(sample_sizes, function(samp)
      t(sapply(lapply(1:repetitions, function(z)
        t(sapply(n_list[[paste0(samp)]][[z]], function(x)
          x$coef))[o[[which(sample_sizes == samp)]][, z], ]), function(x)
            apply(x[1:(floor(nrow(x) / ratio)),], 2, mean)))))

mean_coef_min <-
  lapply(c(2, 3, 10), function(ratio)
    lapply(sample_sizes, function(samp)
      t(sapply(lapply(1:repetitions, function(z)
        t(sapply(n_list[[paste0(samp)]][[z]], function(x)
          x$coef))[o[[which(sample_sizes == samp)]][1, z], ]), function(x)
            x))))

pdf("results.pdf")
ratios <- c(2, 3, 10)
# Add estimate from non private regresion
par(mfrow = c(1, 3))
for (ratio in 1:3) {
  plot(
    1,
    1,
    xlim = c(min(unlist(mean_coef)), max(unlist(mean_coef))),
    ylim = c(1.1, 5),
    type = "n",
    ylab = "",
    yaxt = "n",
    bty = "n",
    xlab = "Estimate",
    main = paste0("Coefficients from 100 repetitions. \n Mean of last 1/", ratios[ratio], " update steps.")
  )
  text(rep(min(unlist(mean_coef)),4), 1:4 + 3/5, labels = c("x1", "x2", "x1*x2", "Intercept"), pos = 4 )
  lapply(1:5, function(z)
    lapply(1:4, function(x)
      points(
        mean_coef[[ratio]][[z]][, x],
        rep(x, repetitions) + z / 5,
        pch = 19,
        col = adjustcolor("black", alpha = 0.5)
      )))
  lapply(1:5, function(z)
    lapply(1:4, function(x)
      points(coef(lm(y ~ x1 * x2, data = rev(dat_list)[[z]]))[c(2:4, 1)][x], x + z/5 , col = "red", pch = "|")))
  abline(h = (2:4) + .1, lty = "dashed")
}


par(mfrow = c(1, 3))
for (ratio in 1:3) {
  plot(
    1,
    1,
    xlim = c(min(unlist(mean_coef_l)), max(unlist(mean_coef_l))),
    ylim = c(1.1, 5),
    type = "n",
    ylab = "",
    yaxt = "n",
    bty = "n",
    xlab = "Estimate",
    main = paste0("Coefficients from 100 repetitions. \n Mean of last 1/", ratios[ratio], " update steps \n ordered by loss.")
  )
  text(rep(min(unlist(mean_coef_l)),4), 1:4 + 3/5, labels = c("x1", "x2", "x1*x2", "Intercept"), pos = 4 )
  lapply(1:5, function(z)
    lapply(1:4, function(x)
      points(
        mean_coef_l[[ratio]][[z]][, x],
        rep(x, repetitions) + z / 5,
        pch = 19,
        col = adjustcolor("black", alpha = 0.5)
      )))
  lapply(1:5, function(z)
    lapply(1:4, function(x)
      points(coef(lm(y ~ x1 * x2, data = rev(dat_list)[[z]]))[c(2:4, 1)][x], x + z/5 , col = "red", pch = "|")))
  abline(h = (2:4) + .1, lty = "dashed")
}

par(mfrow = c(1, 3))
for (ratio in 1:3) {
  plot(
    1,
    1,
    xlim = c(min(unlist(mean_coef_min)), max(unlist(mean_coef_min))),
    ylim = c(1.1, 5),
    type = "n",
    ylab = "",
    yaxt = "n",
    bty = "n",
    xlab = "Estimate"
  )
  lapply(1:5, function(z)
    lapply(1:4, function(x)
      points(
        mean_coef_min[[ratio]][[z]][, x],
        rep(x, repetitions) + z / 5,
        pch = 19,
        col = adjustcolor("black", alpha = 0.5)
      )))
  lapply(1:5, function(z)
    lapply(1:4, function(x)
      points(coef(lm(y ~ x1 * x2, data = rev(dat_list)[[z]]))[c(2:4, 1)][x], x + z/5 , col = "red", pch = "|")))
  abline(h = (2:4) + .1, lty = "dashed")
}

samp_se <-
  t(sapply(rev(dat_list), function(x)
    summary(lm(y ~ x1 * x2, data = x))$coefficients[, 2][c(2, 3, 4, 1)]))

priv_sd <-
  lapply(mean_coef, function(x)
    t(sapply(x, function(z)
      apply(z, 2, sd))))
priv_sd_l <-
  lapply(mean_coef_l, function(x)
    t(sapply(x, function(z)
      apply(z, 2, sd))))

priv_samp_ratio <- lapply(priv_sd, function(x)
  x / samp_se)
priv_samp_ratio_l <- lapply(priv_sd_l, function(x)
  x / samp_se)

par(mfrow = c(4, 1))

greys <- c("grey", "grey50", "grey10")

plot(
  1:5,
  priv_samp_ratio[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of x1"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio[[x]][, 1], col = greys[x]))
abline(h = 1, lty = "dashed")
plot(
  1:5,
  priv_samp_ratio[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of x2"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio[[x]][, 2], col = greys[x]))
abline(h = 1, lty = "dashed")
plot(
  1:5,
  priv_samp_ratio[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of x1*x2"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio[[x]][, 3], col = greys[x]))
abline(h = 1, lty = "dashed")
plot(
  1:5,
  priv_samp_ratio[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of Intercept"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio[[x]][, 4], col = greys[x]))
abline(h = 1, lty = "dashed")


par(mfrow = c(4, 1))

greys <- c("grey", "grey50", "grey10")

plot(
  1:5,
  priv_samp_ratio_l[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio_l))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of x1"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio_l[[x]][, 1], col = greys[x]))
abline(h = 1, lty = "dashed")
plot(
  1:5,
  priv_samp_ratio_l[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio_l))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of x2"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio_l[[x]][, 2], col = greys[x]))
abline(h = 1, lty = "dashed")
plot(
  1:5,
  priv_samp_ratio_l[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio_l))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of x1*x2"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio_l[[x]][, 3], col = greys[x]))
abline(h = 1, lty = "dashed")
plot(
  1:5,
  priv_samp_ratio_l[[1]][, 1],
  ylim = c(0, max(unlist(priv_samp_ratio_l))),
  type = "n",
  bty = "n",
  las = 1,
  ylab = "Privacy SD / Sampling SD",
  xaxt = "n",
  xlab = "Sample Size",
  main = "Coefficient of Intercept"
)
axis(1, at = 1:5, labels = sample_sizes)
lapply(1:3, function(x)
  lines(1:5, priv_samp_ratio_l[[x]][, 4], col = greys[x]))
abline(h = 1, lty = "dashed")
dev.off()
