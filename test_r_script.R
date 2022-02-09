library(lira)
set.seed(43)

alpha <- 0.0
beta <- 1.0
sigma <- 0.1
n <- 100

x <- rnorm(n, 0, 1)
y <- alpha + beta*x + rnorm(n, 0, sigma)
x_err <- rep(0.02, n)
y_err <- rep(0.02, n)
corr <- rep(0.0, n)

x_obs <- x + rnorm(n, 0.0, x_err)
y_obs <- y + rnorm(n, 0.0, y_err)

d <- data.frame(x=x, y=y, x_obs=x_obs, y_obs=y_obs, x_err=x_err, y_err=y_err, corr=corr)

samples <- lira(
    d$x_obs,
    d$y_obs,
    delta.x=d$x_err,
    delta.y=d$y_err,
    covariance.xy=d$corr * d$x_err * d$y_err,
    n.mixture=2,
    n.iter=10000,
    n.adapt=2000,
    print.summary=T,
)
