



# R Examples

## How to Install

To install the package from [CRAN](https://CRAN.R-project.org/package=deadwood), call:


```r
install.packages("deadwood")
```




## Basic Use

Example noisy dataset[^datasetsource]:

[^datasetsource]: The discussed dataset comes from
G. Karypis, E.H. Han, V. Kumar,
CHAMELEON: A hierarchical clustering algorithm using dynamic modeling,
*IEEE Transactions on Computers* **32**(8), 68-75, 1999
and is available for download from
[the clustering benchmarks repository](https://clustering-benchmarks.gagolewski.com/).


``` r
plot_scatter <- function(X, labels=1)
    plot(X, asp=1, ann=FALSE, col=c("#00000055","#aaaaaa33")[labels], pch=16)

X1 <- as.matrix(read.table("chameleon_t7_10k.data.gz"))
plot_scatter(X1)
```

(fig:r_chameleon_t7_10k_dataset)=
```{figure} r-figures/r_chameleon_t7_10k_dataset-1.*
The chameleon_t7_10k dataset
```


Let us perform outlier detection with *Deadwood*.


``` r
library("deadwood")
is_outlier <- deadwood(X1, max_k=1)
plot_scatter(X1, is_outlier+1)
```

(fig:r_chameleon_t7_10k_deadwood)=
```{figure} r-figures/r_chameleon_t7_10k_deadwood-1.*
Outlier detection in chameleon_t7_10k
```


Fraction of detected outliers:


``` r
mean(is_outlier)
## [1] 0.0998
```



## Clusters of Unequal Densities

The above dataset consists of clusters of relatively equal densities.
Here is an example featuring non-homogeneous subgroups.


``` r
X2 <- as.matrix(read.table("chameleon_t8_8k.data.gz"))

par(mfrow=c(1, 2))
is_outlier <- deadwood(X2, max_k=1)
plot_scatter(X2, is_outlier+1)

is_outlier <- deadwood(X2)
plot_scatter(X2, is_outlier+1)
```

(fig:r_chameleon_t8_8k_dataset)=
```{figure} r-figures/r_chameleon_t8_8k_dataset-1.*
Outlier detection in the chameleon_t8_8k dataset, without and with automatic subcluster detection
```

In the right subfigure, Deadwood was able to identify three subclusters automatically.
In each of them, the outlierness threshold is estimated independently.



## Clusters of Highly Imbalanced Sizes

The above dataset consists of relatively large clusters.
[wut/z2](https://clustering-benchmarks.gagolewski.com/) is an
example where two small clusters on the right are treated as anomalous:


``` r
X3 <- as.matrix(read.table("z2.data.gz"))
is_outlier <- deadwood(X3)
plot_scatter(X3, is_outlier+1)
```

(fig:r_z2_dataset)=
```{figure} r-figures/r_z2_dataset-1.*
Outlier detection in the z2 dataset
```

To remedy this, we can decrease the appropriate size threshold
(`max_debris_size`) parameter:


``` r
is_outlier <- deadwood(X3, max_debris_size=20)
plot_scatter(X3, is_outlier+1)
```

(fig:r_z2_dataset2)=
```{figure} r-figures/r_z2_dataset2-1.*
Outlier detection in z2 with smaller max_debris_size
```

Alternatively, we can first identify the clusters in the dataset
(currently, [*Lumbermark*](https://lumbermark.gagolewski.com/)
and [*Genie*](https://genieclust.gagolewski.com/) are supported).
Then, we can perform outlier detection in each cluster separately:


``` r
library("lumbermark")
clusters <- lumbermark(X3, 5)
is_outlier <- deadwood(clusters)
plot_scatter(X3, is_outlier+1)
```

(fig:r_z2_lumbermark)=
```{figure} r-figures/r_z2_lumbermark-1.*
Detected clusters and outliers in z2
```


::::{note}
*To learn more about R, check out my open-access textbook*
[Deep R Programming](https://deepr.gagolewski.com/).
::::
