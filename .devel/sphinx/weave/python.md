



# Python Examples


## How to Install

To install the package from [PyPI](https://pypi.org/project/deadwood), call:


```bash
pip3 install deadwood  # python3 -m pip install deadwood
```



## Basic Use

::::{note}
This section is a work in progress.
In the meantime, take a look at the examples in the [reference manual](../pythonapi).

*To learn more about Python, check out my open-access textbook*
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/).
::::


``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deadwood

def plot_scatter(X, labels=None):
    deadwood.plot_scatter(X, asp=1, labels=labels, alpha=0.75, markers='o', s=10)
```






Example noisy dataset[^datasetsource]:

[^datasetsource]: The discussed dataset comes from G. Karypis, E.H. Han, V. Kumar,
CHAMELEON: A hierarchical clustering algorithm using dynamic modeling,
*IEEE Transactions on Computers* **32**(8), 68-75, 1999
and is available for download from
[the clustering benchmarks repository](https://clustering-benchmarks.gagolewski.com/).


``` python
X1 = np.loadtxt("chameleon_t7_10k.data.gz", ndmin=2)
plot_scatter(X1)
plt.show()
```

(fig:py_chameleon_t7_10k_dataset)=
```{figure} python-figures/py_chameleon_t7_10k_dataset-1.*
The chameleon_t7_10k dataset
```


Detect outliers with *Deadwood* (default settings):



``` python
is_outlier = deadwood.Deadwood().fit_predict(X1)
plot_scatter(X1, (is_outlier<0))
plt.show()
```

(fig:py_chameleon_t7_10k_deadwood)=
```{figure} python-figures/py_chameleon_t7_10k_deadwood-3.*
Outlier detection on chameleon_t7_10k
```

Note the [**scikit-learn**](https://scikit-learn.org/)-compatible API.

Here is the fraction of detected outliers:


``` python
np.mean(is_outlier<0)
## np.float64(0.1014)
```


## Clusters of Unequal Densities

The above dataset consists of clusters of relatively equal densities.
[wut/z2](https://clustering-benchmarks.gagolewski.com/) is an
example where there are five clusters of rather non-homogeneous densities.


``` python
X2 = np.loadtxt("z2.data.gz", ndmin=2)
plot_scatter(X2)
plt.show()
```

(fig:py_z2_dataset)=
```{figure} python-figures/py_z2_dataset-5.*
The z2 dataset
```

Detect outliers with *Deadwood* (default settings):


``` python
is_outlier = deadwood.Deadwood().fit_predict(X2)
plot_scatter(X2, (is_outlier<0))
plt.show()
```

(fig:py_z2_deadwood)=
```{figure} python-figures/py_z2_deadwood-7.*
Outlier detection on z2
```

Detect outliers with *Deadwood*, separately in each cluster
determined by [*Lumbermark*](https://lumbermark.gagolewski.com/):


``` python
import lumbermark
clusters = lumbermark.Lumbermark(n_clusters=5).fit(X2)
plot_scatter(X2, clusters.labels_)
plt.show()
```

(fig:py_z2_lumbermark)=
```{figure} python-figures/py_z2_lumbermark-9.*
Detected clusters of z2
```


``` python
is_outlier = deadwood.Deadwood().fit_predict(clusters)
plot_scatter(X2, (is_outlier<0))
plt.show()
```

(fig:py_z2_lumbermark_deadwood)=
```{figure} python-figures/py_z2_lumbermark_deadwood-11.*
Outlier detection on clusters of z2
```
