library(PSIlence)

data("PUMS5extract10000")

df <- PUMS5extract10000
n <- 1000

x <- rnorm(n)
x2 <- rnorm(n)
x1x2 <- x*x2
#y <- 10 + 1*x + 2*x2 - 4*x1x2 + rnorm(n)
y <- 10 + 1*x + rnorm(n)

df <- data.frame(y = y, x1 = x, x2 = x2)
#reg1 <- coef(lm(y~x+x2+x1x2))
reg1 <- coef(lm(y~x))
reg1
#reg2 <- coef(lm(y[-100]~x[-100]+x2[-100]+x1x2[-100]))
reg2 <- coef(lm(y[-100]~x[-100]))
reg2

plot(x, y)
points(x[100], y[100], pch = 19, col = "red")
abline(reg1, col = "red")
abline(reg2)
res <- NULL
reg_plus <- c(1, 1)
i <- 1
while(any(abs(reg1 - reg_plus) > 0.00001)){
x_plus <- rnorm(1)

y_plus <- reg1[1] + reg1[2] * x_plus + rnorm(1)

reg_plus <- coef(lm(c(y[-100], y_plus)~c(x[-100], x_plus)))

res <- rbind(res, c(y_plus, x_plus, reg_plus, abs(reg1 - reg_plus) ))

i <- i +1
cat(i, "\n")
}

points(x_plus, y_plus, pch = 19, col = "darkgreen")

x_test <- runif(100000, min = -9, max = 9)
y_test <- runif(100000, min = -7, max = 21)
d1 <- data.frame(y = y_test, x = x_test)
res <- NULL

for(i in 1:nrow(d1)){
  reg_plus <- coef(lm(c(y[-100], d1$y[i])~c(x[-100], d1$x[i])))
  tmp <- abs(reg1 - reg_plus)
  res <- rbind(res, c(tmp, sum(tmp)))
  cat(i, "\n")
}

add.alpha <- function(col, alpha=1){
  if(missing(col))
    stop("Please provide a vector of colours.")
  apply(sapply(col, col2rgb)/255, 2, 
        function(x) 
          rgb(x[1], x[2], x[3], alpha=alpha))  
}


res[res>1] <- 1

col_vec1 <- add.alpha("red", alpha = res[,1])
col_vec2 <- add.alpha("red", alpha = res[,2])
col_vec3 <- add.alpha("red", alpha = res[,3])

par(mfrow = c(3, 1))

plot(d1$x, d1$y, pch = 19, col = col_vec1)
points(x[100], y[100], pch = 19, col = "darkgreen")

plot(d1$x, d1$y, pch = 19, col = col_vec2)
points(x[100], y[100], pch = 19, col = "darkgreen")

plot(d1$x, d1$y, pch = 19, col = col_vec3)
points(x[100], y[100], pch = 19, col = "darkgreen")


ll <- function(theta, x, y){
  
  
  reg_plus <- coef(lm(c(y[-100], theta[1])~c(x[-100], theta[2])))
  tmp <- abs(reg1 - reg_plus)
  
  ll <- sum(tmp)
  
}

res_ll <- list()

for(i in 1:100){
sel <- sample(c(1:1000)[-100], 1)
startvals <- c(y[sel], x[sel])
res_ll[[i]] <- optim(startvals, ll, y = y, x = x)
}

res_sel <- which(sapply(res_ll, function(x) x$value) == min(sapply(res_ll, function(x) x$value)))

res_ll[[res_sel]]$par
optim(res_ll[[res_sel]]$par, ll, y = y, x = x, method = "Nelder-Mead")

c(y[100], x[100])
points(ll_res$par[2], ll_res$par[1])

curve(log(x))

ols_attack <- function(ff, df_res, coef_full, n_optim = 100){
  
  mf <- model.frame(ff, df_res)
  
  
  ll <- function(theta, ff, mf, coef_full){
    
    n_var <- ncol(mf)
    df_new <- rbind(mf, theta[1:n_var])
    
    reg_plus <- lm(ff, data = df_new)
    coef_plus <- coef(reg_plus)
    
    
    
    ll <- sum((coef_full - coef_plus)^2)
    
  }
  
  res_ll <- vector(mode = "list", length = n_optim)
  
  for(i in 1:n_optim){
    sel <- sample(1:nrow(df_res), 1)
    startvals <- mf[sel,]
    res_ll[[i]] <- optim(startvals, fn = ll, ff = ff, mf = mf, coef_full = coef_full)
    cat("Optim Run: ", i, "\n")
  }
  
  res_sel <- which(sapply(res_ll, function(x) x$value) == min(sapply(res_ll, function(x) x$value)))[1]
  
  res <- optim(res_ll[[res_sel]]$par, fn = ll, ff = ff, mf = mf, coef_full = coef_full, method = "Nelder-Mead")
  
  return(list(res, res_ll))
}



ff <- income~age+sex+educ
ff <- y~x

coef_full <- coef(lm(ff, data = df))

sensitive_data <- df[nrow(df),]

df_res <- df[-nrow(df),]
df_res <- t(replicate(nrow(df_res), apply(df_res, 2, mean)))

final <- ols_attack(ff = ff, df_res = df_res, coef_full = coef_full, n_optim = 50)

round(final[[1]]$par,0)

which(sapply(final[[2]], function(x) x$value) == min(sapply(final[[2]], function(x) x$value)))

final[[2]][[11]]

sensitive_data


coef_full
summary(lm(ff, df))$coefficients[,2]
summary(lm(ff, df_res))
summary(lm(ff, data = rbind(model.frame(ff, df_res), final[[1]]$par)))
plot(1:nrow(df), df$income)
