library(PSIlence)

data("PUMS5extract10000")

df <- PUMS5extract10000
n <- 1000

x <- rnorm(n)
x2 <- rnorm(n)
x1x2 <- x * x2
y <- 10 + 1 * x + 2 * x2 - 4 * x1x2 + rnorm(n)

y <- 10 + 1 * x + rnorm(n)

df <- data.frame(y = y, x1 = x, x2 = x2)
df <- rbind(df, c(25, 3, 0.2))
#reg1 <- coef(lm(y~x+x2+x1x2))
reg1 <- lm(y ~ x)
coef1 <- coef(reg1)
se1 <- summary(reg1)$coefficients[, 2]
reg1
#reg2 <- coef(lm(y[-100]~x[-100]+x2[-100]+x1x2[-100]))
reg2 <- coef(lm(y[-100] ~ x[-100]))
reg2

plot(x, y)
points(x[100], y[100], pch = 19, col = "red")
abline(reg1, col = "red")
abline(reg2)
res <- NULL
reg_plus <- c(1, 1)
i <- 1
while (any(abs(reg1 - reg_plus) > 0.00001)) {
  x_plus <- rnorm(1)
  
  y_plus <- reg1[1] + reg1[2] * x_plus + rnorm(1)
  
  reg_plus <- coef(lm(c(y[-100], y_plus) ~ c(x[-100], x_plus)))
  
  res <-
    rbind(res, c(y_plus, x_plus, reg_plus, abs(reg1 - reg_plus)))
  
  i <- i + 1
  cat(i, "\n")
}

points(x_plus, y_plus, pch = 19, col = "darkgreen")

x_test <- runif(200000, min = min(x), max = max(x))
y_test <- runif(200000, min = min(y), max = max(y))
d1 <- data.frame(y = y_test, x = x_test)
res <- NULL

for (i in 1:nrow(d1)) {
  reg_plus <- lm(c(y[-100], d1$y[i]) ~ c(x[-100], d1$x[i]))
  
  coef_plus <- coef(reg_plus)
  se_plus <- summary(reg_plus)$coefficients[, 2]
  tmp <- abs(coef1 - coef_plus) + abs(se1 + se_plus)
  res <- rbind(res, c(tmp, sum(tmp)))
  cat(i, "\n")
}

add.alpha <- function(col, alpha = 1) {
  if (missing(col))
    stop("Please provide a vector of colours.")
  apply(sapply(col, col2rgb) / 255, 2,
        function(x)
          rgb(x[1], x[2], x[3], alpha = alpha))
}

res_wgt <- res
if (any(res > 1))
  res_wgt <- res / max(res)

col_vec1 <- add.alpha("red", alpha = res_wgt[, 1])
col_vec2 <- add.alpha("red", alpha = res_wgt[, 2])
col_vec3 <- add.alpha("red", alpha = res_wgt[, 3])

par(mfrow = c(3, 1))

plot(d1$x, d1$y, pch = 20, col = col_vec1)
points(x[100], y[100], pch = 19, col = "darkgreen")

? points

abline(a = reg1[1]  , b = reg1[2])
plot(d1$x, d1$y, pch = 19, col = col_vec2)
points(x[100], y[100], pch = 19, col = "darkgreen")

plot(d1$x, d1$y, pch = 19, col = col_vec3)
points(x[100], y[100], pch = 19, col = "darkgreen")


ll <- function(theta, x, y) {
  reg_plus <- coef(lm(c(y[-100], theta[1]) ~ c(x[-100], theta[2])))
  tmp <- abs(reg1 - reg_plus)
  
  ll <- sum(tmp)
  
}

res_ll <- list()

for (i in 1:100) {
  sel <- sample(c(1:1000)[-100], 1)
  startvals <- c(y[sel], x[sel])
  res_ll[[i]] <- optim(startvals, ll, y = y, x = x)
}

res_sel <-
  which(sapply(res_ll, function(x)
    x$value) == min(sapply(res_ll, function(x)
      x$value)))

res_ll[[res_sel]]$par
optim(res_ll[[res_sel]]$par,
      ll,
      y = y,
      x = x,
      method = "Nelder-Mead")

c(y[100], x[100])
points(ll_res$par[2], ll_res$par[1])

curve(log(x))

ols_attack <-
  function(ff,
           df_res,
           coef_full,
           se_full,
           coef_digits,
           se_digits,
           n_optim = 100,
           n_chains = 1) {
    mf <- model.frame(ff, df_res)
    
    
    ll <-
      function(theta,
               ff,
               mf,
               coef_full,
               se_full,
               coef_digits,
               se_digits) {
        n_var <- ncol(mf)
        df_new <- rbind(mf, theta[1:n_var])
        
        reg_plus <- lm(ff, data = df_new)
        coef_plus <- round(coef(reg_plus), coef_digits)
        
        se_plus <-
          round(summary(reg_plus)$coefficients[, 2], se_digits)
        
        
        ll <-
          sum((coef_full - coef_plus) ^ 2 + (se_full - se_plus) ^ 2)
        return(ll)
      }
    
    res_chains <- vector(mode = "list", length = n_chains)
    for (chain in 1:n_chains) {
      cat("\n#######\n# Chain: ", chain, "\n#######\n\n")
      
      res_ll <- vector(mode = "list", length = n_optim)
      
      
      
      for (i in 1:n_optim) {
        
        sel <- sample(1:nrow(df_res), floor(nrow(df_res) / 5))
        startvals <- apply(mf[sel, ], 2, mean)
        #startvals <- startvals + rnorm(length(startvals), 0, 0.1 / i)
        
        res_ll[[i]] <-
          optim(
            startvals,
            fn = ll,
            ff = ff,
            mf = mf,
            coef_full = coef_full,
            se_full = se_full,
            coef_digits = coef_digits,
            se_digits = se_digits
          )
        
        #startvals <- res_ll[[i]]$par
        cat("Optim Run: ", i, "\n")
      }
      
      res_sel <-
        which(sapply(res_ll, function(x)
          x$value) == min(sapply(res_ll, function(x)
            x$value)))[1]
      
      startvals <- res_ll[[res_sel]]$par
      # startvals <- apply(t(sapply(res_ll, function(x)
      #   x$par))[(n_optim-50):n_optim, ], 2, mean)
      
      res <-
        optim(
          startvals,
          fn = ll,
          ff = ff,
          mf = mf,
          coef_full = coef_full,
          se_full = se_full,
          method = "Nelder-Mead"
        )
      
      res_chains[[chain]] <- list(res, res_ll)
    }
    
    
    return(res_chains)
  }



ff <- income ~ age + sex + educ
ff <- y ~ x1 * x2

coef_full <- round(coef(lm(ff, data = df)), 10)
se_full <- round(summary(lm(ff, data = df))$coefficients[, 2], 10)
sensitive_data <- df[nrow(df), ]
df_res <- df[-nrow(df), ]

final <-
  ols_attack(
    ff = ff,
    df_res = df_res,
    coef_full = coef_full,
    se_full = se_full,
    coef_digits = 10,
    se_digits = 10,
    n_optim = 250,
    n_chains = 4
  )

#round(final[[1]]$par, 0)
final[[1]][[1]]$par
sensitive_data

par(mfrow = c(4, 1))
for (i in 1:4) {
  see <- t(sapply(final[[i]][[2]], function(x)
    x$par))
  see_val <-
    add.alpha("red", 1 - sapply(final[[i]][[2]], function(x)
      x$value))
  sweep(see, 2, as.numeric(sensitive_data))
  
  plot(
    see[, 2],
    see[, 1],
    pch = 20,
    col = adjustcolor("red", alpha = 0.3),
    xlim = c(min(df$x1), max(df$x1)),
    ylim = c(min(df$y), max(df$y))
  )
  lines(see[, 2], see[, 1], col = adjustcolor("black", alpha = 0.5))
  points(sensitive_data[2], sensitive_data[1], col = "darkgreen")
  points(final[[i]][[1]]$par[2], final[[i]][[1]]$par[1], pch = 20, col = "yellow")
}



?points
which(sapply(final[[2]], function(x)
  x$value) == min(sapply(final[[2]], function(x)
    x$value)))[1]

final[[2]][[3]]

sensitive_data


coef_full
summary(lm(ff, df))$coefficients[, 2]
summary(lm(ff, df_res))
summary(lm(ff, data = rbind(model.frame(ff, df_res), final[[1]]$par)))
plot(1:nrow(df), df$income)

curve(sqrt(x))
