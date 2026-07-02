



# Python Examples


## How to Install

To install the package from [PyPI](https://pypi.org/project/deadwood), call:


```bash
pip3 install deadwood  # python3 -m pip install deadwood
```



## Basic Use


``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deadwood

def plot_scatter(X, labels=None):
    deadwood.plot_scatter(X, asp=1, labels=labels, alpha=0.75, markers='o', s=10)
```






Example noisy dataset[^datasetsource]:

[^datasetsource]: The discussed dataset comes from
G. Karypis, E.H. Han, V. Kumar,
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


Let us perform outlier detection with *Deadwood*.
Note the [**scikit-learn**](https://scikit-learn.org/)-compatible API.


``` python
is_outlier = deadwood.Deadwood(max_n_clusters=1).fit_predict(X1)
plot_scatter(X1, is_outlier)
plt.show()
```

(fig:py_chameleon_t7_10k_deadwood)=
```{figure} python-figures/py_chameleon_t7_10k_deadwood-3.*
Outlier detection in chameleon_t7_10k
```

Here is the fraction of detected outliers:


``` python
np.mean(is_outlier<0)
## np.float64(0.0998)
```


## Clusters of Unequal Densities

The above dataset consists of clusters of relatively equal densities.
Here is an example featuring non-homogeneous subgroups.


``` python
X2 = np.loadtxt("chameleon_t8_8k.data.gz", ndmin=2)

plt.subplot(121)
is_outlier = deadwood.Deadwood(max_n_clusters=1).fit_predict(X2)
plot_scatter(X2, is_outlier)

plt.subplot(122)
is_outlier = deadwood.Deadwood().fit_predict(X2)
plot_scatter(X2, is_outlier)

plt.show()
```

(fig:py_chameleon_t8_8k_dataset)=
```{figure} python-figures/py_chameleon_t8_8k_dataset-5.*
Outlier detection in the chameleon_t8_8k dataset, without and with automatic subcluster detection
```

In the right subfigure, Deadwood was able to identify three subclusters automatically.
In each of them, the outlierness threshold is estimated independently.



## Clusters of Highly Imbalanced Sizes

The above dataset consists of relatively large clusters.
[wut/z2](https://clustering-benchmarks.gagolewski.com/) is an
example where two small clusters on the right are treated as anomalous:


``` python
X3 = np.loadtxt("z2.data.gz", ndmin=2)
is_outlier = deadwood.Deadwood().fit_predict(X3)
plot_scatter(X3, is_outlier)
plt.show()
```

(fig:py_z2_dataset)=
```{figure} python-figures/py_z2_dataset-7.*
Outlier detection in the z2 dataset
```

To remedy this, we can decrease the appropriate size threshold
(`max_debris_size`) parameter:


``` python
is_outlier = deadwood.Deadwood(max_debris_size=20).fit_predict(X3)
plot_scatter(X3, is_outlier)
plt.show()
```

(fig:py_z2_deadwood2)=
```{figure} python-figures/py_z2_deadwood2-9.*
Outlier detection in z2 with smaller max_debris_size
```

Alternatively, we can first identify the clusters in the dataset
(currently, [*Lumbermark*](https://lumbermark.gagolewski.com/)
and [*Genie*](https://genieclust.gagolewski.com/) are supported).
Then, we can perform outlier detection in each cluster separately:


``` python
import lumbermark
clusters = lumbermark.Lumbermark(n_clusters=5).fit(X3)
is_outlier = deadwood.Deadwood().fit_predict(clusters)
plot_scatter(X3, is_outlier)
plt.show()
```

(fig:py_z2_lumbermark)=
```{figure} python-figures/py_z2_lumbermark-11.*
Detected clusters and outliers in z2
```

::::{note}
*To learn more about Python, check out my open-access textbook*
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/).
::::

