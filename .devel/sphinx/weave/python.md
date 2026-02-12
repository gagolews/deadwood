



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
```






Example noisy dataset[^datasetsource]:

[^datasetsource]: The discussed datasets come from
G. Karypis, E.H. Han, V. Kumar,
CHAMELEON: A hierarchical clustering algorithm using dynamic modeling,
*IEEE Transactions on Computers* **32**(8), 68-75, 1999
and are available for download from
[the clustering benchmarks repository](https://clustering-benchmarks.gagolewski.com/).


``` python
X1 = np.loadtxt("chameleon_t7_10k.data.gz", ndmin=2)
deadwood.plot_scatter(X1, asp=1, alpha=0.3)
plt.show()
```

(fig:py_chameleon_t7_10k_dataset)=
```{figure} python-figures/py_chameleon_t7_10k_dataset-1.*
The chameleon_t7_10k dataset
```


Detect outliers with *Deadwood* (default settings):



``` python
is_outlier = deadwood.Deadwood().fit_predict(X1)
deadwood.plot_scatter(X1, asp=1, labels=(is_outlier<0), alpha=0.3)
plt.show()
```

(fig:py_chameleon_t7_10k_deadwood)=
```{figure} python-figures/py_chameleon_t7_10k_deadwood-3.*
Outlier detection on chameleon_t7_10k
```

Fraction of detected outliers:


``` python
np.mean(is_outlier<0)
## np.float64(0.1014)
```


## Clusters of Unequal Densities

The above dataset consists of clusters of relatively equal densities.
Here is another one, where it is clearly not the case.


``` python
X2 = np.loadtxt("chameleon_t8_8k.data.gz", ndmin=2)
deadwood.plot_scatter(X2, asp=1, alpha=0.3)
plt.show()
```

(fig:py_chameleon_t8_8k_dataset)=
```{figure} python-figures/py_chameleon_t8_8k_dataset-5.*
The chameleon_t8_8k dataset
```

Detect outliers with *Deadwood* (default settings):


``` python
is_outlier = deadwood.Deadwood().fit_predict(X2)
deadwood.plot_scatter(X2, asp=1, labels=(is_outlier<0), alpha=0.3)
plt.show()
```

(fig:py_chameleon_t8_8k_deadwood)=
```{figure} python-figures/py_chameleon_t8_8k_deadwood-7.*
Outlier detection on chameleon_t8_8k
```

Detect outliers with *Deadwood*, separately in each cluster
detected by [*Genie*](https://genieclust.gagolewski.com/):



``` python
import genieclust
clusters = genieclust.Genie(n_clusters=10, M=5).fit(X2)
is_outlier = deadwood.Deadwood().fit_predict(clusters)
deadwood.plot_scatter(X2, asp=1, labels=(is_outlier<0), alpha=0.3)
plt.show()
```

(fig:py_chameleon_t8_8k_lumbermark)=
```{figure} python-figures/py_chameleon_t8_8k_lumbermark-9.*
Outlier detection on clusters of chameleon_t8_8k
```
