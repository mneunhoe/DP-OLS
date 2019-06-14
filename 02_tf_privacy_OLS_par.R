
#For parallel computing
library(foreach)


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




sample_sizes <- c(100, 250, 500, 1000, 1500, 2000, 10000)
sample_sizes <- c(100, 250)
repetitions <- 1


cl <- parallel::makeForkCluster(2)
doParallel::registerDoParallel(cl)

foreach(samp = sample_sizes) %dopar% {
#for(samp in sample_sizes) {

run_tf_ols(samp)
}
run_tf_ols <- function(samp = 100){
  
  library(reticulate)
  library(tensorflow)
  use_condaenv("r-tensorflow")
  
  
  # tf privacy needs to be installed in the conda environment r-tensorflow prior to running this
  privacy <- import("privacy")
  
  norm_vec <- function(x) {
    sqrt(sum(x ^ 2))
  }
  
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
  
  regression <- function(x) {
    pred <- tf$matmul(x, W1) + b1
    return(pred)
  }
  
  rep_res <- list()  
dat_samp <- dat[sample(nrow(dat), samp),]

features <-
  as.matrix(cbind(dat_samp[, 2:3], dat_samp[, 2] * dat_samp[, 3]))

true_y <- as.matrix(dat_samp[, 1, drop = F])


N <- nrow(dat_samp)
Dim <- ncol(features)

mb_size <- as.integer(samp %/% 100L)
learning_rate <- 0.1

epochs <- 2


# DP declarations

# DP-SGD
dp <- T
num_microbatches <- mb_size
# How to do this with dp? 
l2_norm_clip <- 1/3*norm_vec(-2 * t(true_y) %*% cbind(features, 1) / nrow(features))
noise_multiplier <- 2.0



for(repetition in 1:repetitions){
# Set up the variables in the GAN structure
X <<- tf$placeholder(tf$float32, shape = list(NULL, Dim))
y <<- tf$placeholder(tf$float32, shape = list(NULL, 1L))

W1 <<- tf$Variable(xavier_init(list(Dim, 1L)), name = "W1")
b1 <<- tf$Variable(tf$zeros(shape = list(1L)), name = "b1")

weights <<- list(W1, b1)


pred <<- regression(X)


if (dp) {
  regression_loss <<-
    tf$losses$mean_squared_error(
      labels = y,
      predictions = pred,
      reduction = tf$losses$Reduction$NONE
    )
  optimizer <<-
    privacy$optimizers$dp_optimizer$DPAdamGaussianOptimizer(
      learning_rate = learning_rate,
      num_microbatches = num_microbatches,
      l2_norm_clip = l2_norm_clip,
      noise_multiplier = noise_multiplier,
      ledger = FALSE
    )
  
  solver <<- optimizer$minimize(
    loss = regression_loss,
    global_step = tf$train$get_global_step(),
    var_list = weights
  )
} else{
  regression_loss <<-
    tf$reduce_mean(tf$losses$mean_squared_error(labels = y, predictions = pred))
  
  solver <<-
    tf$train$AdamOptimizer()$minimize(regression_loss, var_list = weights)
}

val_loss <<-
  tf$reduce_mean(tf$losses$mean_squared_error(labels = y, predictions = pred))




sess <<- tf$Session()
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
    
    global_step <- epoch * steps_per_epoch + step - steps_per_epoch
    if (dp) {
      eps <- compute_epsilon(global_step)
      # cat("Epsilon after ", epoch, " epochs is: ", eps, "\n\n")
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
rep_res[[paste0(repetition)]] <- res
#saveRDS(n_list, "n_list.RDS")
}
return(rep_res)
}


n_list
eps_vec <- unlist(lapply(res, function(x)
  x$eps))
val_loss_vec <- unlist(lapply(res, function(x)
  x$val_loss))
plot(1:global_step, eps_vec, ylim = c(0, 1), type = "l")
plot(1:global_step, val_loss_vec, type = "l")

coefs <- t(sapply(res, function(x) x$coef))

apply(coefs, 2, mean)
apply(coefs[(floor(nrow(coefs)/2)+1):nrow(coefs),], 2, mean)
apply(coefs[(floor(nrow(coefs)/3)+1):nrow(coefs),], 2, mean)
apply(coefs[(floor(nrow(coefs)/10)+1):nrow(coefs),], 2, mean)
