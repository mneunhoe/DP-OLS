library(reticulate)
library(tensorflow)
use_condaenv("r-tensorflow")

privacy <- import("privacy")

samp <- 1112
dat_samp <- dat[sample(nrow(dat), samp),]

train_sel <- sample(samp, floor(0.9*samp))

train_samp <- dat_samp[train_sel,]
val_samp <- dat_samp[-train_sel,]

features <-
  as.matrix(cbind(train_samp[, 2:3], train_samp[, 2] * train_samp[, 3]))

true_y <- as.matrix(train_samp[, 1, drop = F])

val_features <-
  as.matrix(cbind(val_samp[, 2:3], val_samp[, 2] * val_samp[, 3]))

val_true_y <- as.matrix(val_samp[, 1, drop = F])


N <- nrow(train_samp)
Dim <- ncol(features)

mb_size <- 10L
learning_rate <- 0.01

epochs <- 50


# DP declarations

# DP-SGD
dp <- T
num_microbatches <- mb_size
l2_norm_clip <- 32
noise_multiplier <- 3.0

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
compute_epsilon(10000)

xavier_init <- function(size) {
  in_dim = size[[1]]
  xavier_stddev = 1. / tf$sqrt(in_dim / 2.)
  return(tf$random_normal(shape = size, stddev = 0.))
}

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

val_loss <-  tf$reduce_mean(tf$losses$mean_squared_error(labels = y, predictions = pred))

sess <- tf$Session()
sess$run(tf$global_variables_initializer())

steps_per_epoch <- N %/% mb_size
res <- list()

for (epoch in 1:epochs) {
  for (step in 1:steps_per_epoch) {
    ind <- sample(nrow(train_samp), mb_size)
    features_mb <- features[ind, ,drop = F]
    true_y_mb <- true_y[ind,, drop = F]
    loss_curr <- sess$run(list(solver, regression_loss),
                         feed_dict = dict(X = features_mb, y = true_y_mb))[[2]]
    
    val_loss_curr <- sess$run(val_loss, feed_dict = dict(X = val_features, y = val_true_y))
    coefs <- do.call(rbind, sess$run(list(W1, b1)))
    cat(
      "Epoch: ",
      epoch,
      "Loss: ",
      mean(loss_curr),
      "Coefs:",
      coefs,
      "\n\n"
    )
    
    global_step <- epoch * steps_per_epoch + step - steps_per_epoch
    if (dp) {
      eps <- compute_epsilon(global_step)
      cat("Epsilon after ", epoch, " epochs is: ", eps, "\n\n")
    }
    
    res[[global_step]] <- list(loss = mean(loss_curr), val_loss = val_loss_curr, coef = coefs, eps = eps)
  }
}

eps_vec <- unlist(lapply(res, function(x) x$eps))
val_loss_vec <- unlist(lapply(res, function(x) x$val_loss))
plot(1:global_step, eps_vec, ylim = c(0, 1), type = "l")
plot(1:global_step, val_loss_vec, type = "l")
which(val_loss_vec == min(val_loss_vec))
res[[which(val_loss_vec == min(val_loss_vec))]]

grads <- optimizer$compute_gradients(regression_loss, 
                             var_list = weights, gate_gradients = optimizer$GATE_NONE)

sess$run(list(grads),
         feed_dict = dict(X = features_mb, y = true_y_mb))




norm_vec <- function(x) sqrt(sum(x^2))

norm_vec(2*t(cbind(features_mb,1))%*%true_y_mb/nrow(features_mb))

norm_vec(c(1, 2, 3))
