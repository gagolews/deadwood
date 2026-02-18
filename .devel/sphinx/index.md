# *Deadwood*: Outlier Detection via Trimming of Mutual Reachability Minimum Spanning Trees

::::{image} _static/img/deadwood_toy_example.png
:class: img-right-align-always
:alt: Deadwood
:width: 128px
::::


**Keywords**: Deadwood, outlier detection, anomaly detection, HDBSCAN\*, DBSCAN,
minimum spanning tree, MST, density estimation, mutual reachability distance.

*Deadwood* is an anomaly detection algorithm based on mutual reachability
minimum spanning trees.  It trims protruding tree segments and marks small
debris as outliers.

More precisely:

* the use of a mutual reachability distance pulls peripheral points
    farther away from each other,

* tree edges with weights beyond the detected elbow point are removed,

* all the resulting connected components whose sizes are smaller than
    a given threshold are deemed anomalous.



## Contributing

**deadwood** is distributed under the open source GNU AGPL v3 license.
Its source code can be downloaded from [GitHub](https://github.com/gagolews/deadwood).

The Python version is available from [PyPI](https://pypi.org/project/deadwood).
The R version can be fetched from [CRAN](https://CRAN.R-project.org/package=deadwood).

The core functionality is implemented in the form of a C++ library.
It can thus be easily adapted for use in other environments.
New contributions are welcome, e.g., Julia, Matlab/GNU Octave wrappers.


**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com/)



::::{toctree}
:maxdepth: 2
:caption: deadwood
:hidden:

About <self>
Author <https://www.gagolewski.com/>
Source Code (GitHub) <https://github.com/gagolews/deadwood>
Bug Tracker and Feature Suggestions <https://github.com/gagolews/deadwood/issues>
PyPI Entry <https://pypi.org/project/deadwood>
CRAN Entry <https://CRAN.R-project.org/package=deadwood>
::::



::::{toctree}
:maxdepth: 1
:caption: Python API
:hidden:

weave/python
weave/sklearn_toy_example
pythonapi
::::


::::{toctree}
:maxdepth: 1
:caption: R API
:hidden:

weave/r
rapi
::::


::::{toctree}
:maxdepth: 1
:caption: Other
:hidden:

quitefastmst <https://quitefastmst.gagolewski.com/>
lumbermark <https://lumbermark.gagolewski.com/>
genieclust <https://genieclust.gagolewski.com/>
Clustering Benchmarks <https://clustering-benchmarks.gagolewski.com/>
Minimalist Data Wrangling in Python <https://datawranglingpy.gagolewski.com/>
Deep R Programming <https://deepr.gagolewski.com/>
news
z_bibliography
::::


<!--
Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
-->
