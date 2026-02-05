library("tinytest")
library("deadwood")


blob_and_aureola <- function(n, m) {
    r <- runif(n)
    u <- runif(n, 0, 2*pi)
    t <- seq(0, 2*pi, length.out=m+1)[-1]
    t(cbind(
        r*rbind(cos(u), sin(u)),
        5*rbind(cos(t), sin(t))
    ))
}


## Blob + Aureola

set.seed(123)
n <- 197
m <- 36
X <- blob_and_aureola(n, m)

d <- deadwood(X, M=13)
expect_equal(NROW(attr(d, "mst")), n+m-1)
expect_equal(NCOL(attr(d, "mst")), 3)
expect_equal(attr(attr(d, "mst"), "Size"), n+m)
expect_equal(attr(d, "cut_edges"), numeric(0))
expect_equal(attr(d, "contamination"), m/(n+m))
expect_equal(mean(d), m/(n+m))
expect_equal(as.logical(d), rep(c(FALSE, TRUE), c(n, m)))


## 2 * (Blob + Aureola)

set.seed(123)
n1 <- 153
m1 <- 21
n2 <- 13
m2 <- 5
X <- rbind(
    blob_and_aureola(n1, m1),
    blob_and_aureola(n2, m2)+12
)
d1 <- deadwood(X, M=11)
# plot(X, col=d1+1)
expect_equal(as.logical(d1), rep(c(FALSE, TRUE, TRUE, TRUE), c(n1, m1, n2, m2)))

d2 <- deadwood(attr(d1, "mst"), cut_edges=NROW(attr(d1, "mst")), max_contamination=0.39, max_debris_size=10)
plot(X, col=d2+1)
expect_equal(as.logical(d2), rep(c(FALSE, TRUE, FALSE, TRUE), c(n1, m1, n2, m2)))
expect_equal(attr(d2, "contamination"), c(m1/(n1+m1), m2/(n2+m2)))
expect_equal(attr(d2, "cut_edges"), NROW(attr(d2, "mst")))
