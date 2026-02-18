<a href="https://deadwood.gagolewski.com/"><img src="https://www.gagolewski.com/_static/img/deadwood.png" align="right" height="128" width="128" /></a>
# [**deadwood**](https://deadwood.gagolewski.com/) Package for R and Python

### *Deadwood*: Outlier Detection via Trimming of Mutual Reachability Minimum Spanning Trees

![deadwood for Python](https://github.com/gagolews/deadwood/workflows/deadwood%20for%20Python/badge.svg)
![deadwood for R](https://github.com/gagolews/deadwood/workflows/deadwood%20for%20R/badge.svg)

**Keywords**: Deadwood, outlier detection, anomaly detection, HDBSCAN\*, DBSCAN,
minimum spanning tree, MST, density estimation, mutual reachability distance.


Refer to the package **homepage** at <https://deadwood.gagolewski.com/>
for the reference manual, tutorials, examples, and benchmarks.

**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com/)


## About

*Deadwood* is an anomaly detection algorithm based on mutual reachability
minimum spanning trees.  It trims protruding tree segments and marks small
debris as outliers.

More precisely:

* the use of a mutual reachability distance pulls peripheral points
    farther away from each other,

* tree edges with weights beyond the detected elbow point are removed,

* all the resulting connected components whose sizes are smaller than
    a given threshold are deemed anomalous.


## How to Install

### Python Version

To install from [PyPI](https://pypi.org/project/deadwood), call:

```bash
pip3 install deadwood  # python3 -m pip install deadwood
```

*To learn more about Python, check out my open-access textbook*
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/).


### R Version

To install from [CRAN](https://CRAN.R-project.org/package=deadwood), call:

```r
install.packages("deadwood")
```

*To learn more about R, check out my open-access textbook*
[Deep R Programming](https://deepr.gagolewski.com/).


### Other

The core functionality is implemented in the form of a C++ library.
It can thus be easily adapted for use in other projects.

New contributions are welcome, e.g., Julia, Matlab/GNU Octave wrappers.


## License

Copyright (C) 2025–2026 Marek Gagolewski <https://www.gagolewski.com/>

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License Version 3, 19
November 2007, published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
General Public License Version 3 for more details. You should have
received a copy of the License along with this program. If not, see
(https://www.gnu.org/licenses/).


## References

TODO

See **deadwood**'s [homepage](https://deadwood.gagolewski.com/) for more
references.
