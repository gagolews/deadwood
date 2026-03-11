



# R Examples

## How to Install

To install the package from [CRAN](https://CRAN.R-project.org/package=deadwood), call:


```r
install.packages("deadwood")
```




## Basic Use

::::{note}
This section is a work in progress.  In the meantime, take a look at
the documentation of the [deadwood](../rapi/deadwood) function.

*To learn more about R, check out my open-access textbook*
[Deep R Programming](https://deepr.gagolewski.com/).
::::


Example noisy dataset[^datasetsource]:

[^datasetsource]: The discussed dataset comes from
G. Karypis, E.H. Han, V. Kumar,
CHAMELEON: A hierarchical clustering algorithm using dynamic modeling,
*IEEE Transactions on Computers* **32**(8), 68-75, 1999
and is available for download from
[the clustering benchmarks repository](https://clustering-benchmarks.gagolewski.com/).


``` r
X1 <- as.matrix(read.table("chameleon_t7_10k.data.gz"))
plot(X1, asp=1, ann=FALSE, col="#00000055")
```

(fig:r_chameleon_t7_10k_dataset)=
```{figure} r-figures/r_chameleon_t7_10k_dataset-1.*
The chameleon_t7_10k dataset
```


Detect outliers with *Deadwood* (default settings):


``` r
library("deadwood")
is_outlier <- deadwood(X1)
plot(X1, asp=1, ann=FALSE, col=c("#00000055","#ff333333")[is_outlier+1])
```

(fig:r_chameleon_t7_10k_deadwood)=
```{figure} r-figures/r_chameleon_t7_10k_deadwood-1.*
Outlier detection on chameleon_t7_10k
```


Fraction of detected outliers:


``` r
mean(is_outlier)
## [1] 0.1014
```


## Clusters of Unequal Densities

The above dataset consists of clusters of relatively equal densities.
[wut/z2](https://clustering-benchmarks.gagolewski.com/) is an
example where there are five clusters of rather non-homogeneous densities.


``` r
X2 <- as.matrix(read.table("z2.data.gz"))
plot(X2, asp=1, ann=FALSE, col="#00000055")
```

(fig:r_z2_dataset)=
```{figure} r-figures/r_z2_dataset-1.*
The z2 dataset
```

Detect outliers with *Deadwood* (default settings):


``` r
is_outlier <- deadwood(X2)
plot(X2, asp=1, ann=FALSE, col=c("#00000055","#ff333333")[is_outlier+1])
```

(fig:r_z2_deadwood)=
```{figure} r-figures/r_z2_deadwood-1.*
Outlier detection on z2
```

Detect outliers with *Deadwood*, separately in each cluster
detected by [*Lumbermark*](https://lumbermark.gagolewski.com/):



``` r
library("lumbermark")
clusters <- lumbermark(X2, 5)
is_outlier <- deadwood(clusters)
plot(X2, asp=1, ann=FALSE, col=c("#00000055","#ff333333")[is_outlier+1])
```

(fig:r_z2_lumbermark)=
```{figure} r-figures/r_z2_lumbermark-1.*
Outlier detection on clusters of z2
```
