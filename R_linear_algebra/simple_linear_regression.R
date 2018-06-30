


iris <- iris

plot(iris)

dat <- iris[, c("Petal.Length", "Petal.Width")]

names(dat) <- c("x", "y")

head(dat)

# yhat = wx + b
# need to identify m and b


X = as.matrix(data.frame(
    intercept = rep(1, nrow(dat)),
    x = dat$x,
    stringsAsFactors = F))

y = as.matrix(dat$y)


plot(X[,2], y, pch=16, color=rgb(0, 0, 0))


loss_func <- function(yhat, y) {
    return(   0.5  *   (mean((yhat - y)^2))      )
}


w <- as.matrix(rnorm(2))
w <- as.matrix(c(0.2, 0.2))

derive_of_loss <- function(yhat, y) {
    return( mean(yhat - y)  )
}

head(X)

head(X) %*% w

t(w) %*% t(head(X))
head(y)


dim(X)
dim(w)


