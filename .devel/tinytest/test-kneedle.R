library("tinytest")
library("deadwood")

expect_equal(1, kneedle_increasing(c(1.0, 1.0, 1.0, 1.0, 1.0)))

n <- 1001
x <- seq(0.1, 1.0, length.out=n)
y <- (-1.0/x+5)
expect_equal(kneedle_increasing(y, convex=FALSE, dt=100), 241)

y <- max(y)-rev(y)
expect_equal(kneedle_increasing(y, convex=TRUE, dt=100), 761)
