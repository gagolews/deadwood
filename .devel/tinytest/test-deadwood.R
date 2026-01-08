library("tinytest")
library("deadwood")

#set.seed(123)
n <- 1000
d <- rpois(1, 10)+2
X <- matrix(rnorm(n*d), nrow=n)

# TODO
