"""
The Deadwood Outlier Detection Algorithm
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2025-2026, Marek Gagolewski <https://www.gagolewski.com/>     #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import os
import sys
import math
import numpy as np
import warnings

import quitefastmst
from . import core

from sklearn.base import BaseEstimator

###############################################################################
###############################################################################
###############################################################################




class MSTBase(BaseEstimator):
    """
    The base class for :any:`genieclust.Genie`, :any:`genieclust.GIc`,
    :any:`lumbermark.Lumbermark`, :any:`deadwood.Deadwood`, and other
    Euclidean and mutual reachability minimum spanning tree-based
    clustering and outlier detection algorithms [2]_.

    A Euclidean minimum spanning tree (MST) provides a computationally
    convenient representation of a dataset: the `n` points are connected
    via `n-1` shortest segments.  Provided that the dataset
    has been appropriately preprocessed so that the distances between the
    points are informative, an MST can be applied in outlier detection,
    clustering, density reduction, and many other topological data
    analysis tasks.


    Parameters
    ----------

    M : int, default=0
        Smoothing factor for the mutual reachability distance [1]_.
        `M = 0` and `M = 1` select the original distance as given by
        the `metric` parameter.

    metric : str, default='l2'
        The metric used to compute the linkage.

        One of:
        ``"l2"`` (synonym: ``"euclidean"``; the default),
        ``"l1"`` (synonym: ``"manhattan"``, ``"cityblock"``),
        ``"cosinesimil"`` (synonym: ``"cosine"``), or
        ``"precomputed"``.

        For ``"precomputed"``, the `X` argument to the ``fit`` method
        must be a distance vector or a square-form distance matrix;
        see :any:`scipy.spatial.distance.pdist`.

        Determining minimum spanning trees with respect to the Euclidean
        distance is much faster than with other metrics thanks to the
        `quitefastmst <https://quitefastmst.gagolewski.com/>`_ package.

    quitefastmst_params : dict or None, default=None
        Additional parameters to be passed to :any:`quitefastmst.mst_euclid`
        if ``metric`` is ``"l2"``.

    verbose : bool, default=False
        Whether to print diagnostic messages and progress information
        onto ``stderr``.


    Notes
    -----

    If ``metric`` is ``"l2"``, then the MST is computed via a call to
    :any:`quitefastmst.mst_euclid`.  It is efficient in low-dimensional spaces.
    Otherwise, a general-purpose implementation of the Jarník
    (Prim/Dijkstra)-like :math:`O(n^2)`-time algorithm is called.

    If `M > 0`, then the minimum spanning tree is computed with respect to a
    mutual reachability distance [1]_:
    :math:`d_M(i,j)=\\max\\{d(i,j), c_M(i), c_M(j)\\}`, where
    :math:`d(i,j)` is an ordinary distance and :math:`c_M(i) is the core
    distance given by :math:`d(i,k)` with :math:`k` being
    :math:`i`'s :math:`M`-th nearest neighbour (not including self,
    unlike in [1]_).  This pulls outliers away from their neighbours.

    If the distances are not unique,  there might be multiple trees
    spanning a given graph that meet the minimality property.


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used for computing the minimum
            spanning tree.


    Attributes
    ----------

    n_samples_ : int
        The number of points in the dataset.

    n_features_ : int or None
        The number of features in the dataset.

    labels_ : ndarray of shape(n_samples_, )
        Detected cluster labels or outlier flags.

        For clustering, ``labels_[i]`` gives the cluster ID of the `i`-th object
        (between 0 and `n_clusters_` - 1).

        For outlier detection, ``1`` denotes an inlier.

        Outliers are labelled ``-1``.


    References
    ----------

    .. [1]
        R.J.G.B. Campello, D. Moulavi, J. Sander,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        https://doi.org/10.1007/978-3-642-37456-2_14

    .. [2]
        M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        https://doi.org/10.1007/s00357-024-09483-1

    .. [3]
        M. Gagolewski, *quitefastmst*, in preparation, 2026, TODO
    """

    def __init__(
            self,
            *,
            M=0,
            metric="l2",
            quitefastmst_params=None,
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__()

        self.M                   = M
        self.metric              = metric
        self.quitefastmst_params = quitefastmst_params
        self.verbose             = verbose

        # sklearn convention: trailing underscore = available after fit()
        self.labels_             = None
        self.n_samples_          = None
        self.n_features_         = None

        # "protected" slots
        self._tree_d_            = None
        self._tree_i_            = None
        self._tree_cumdeg_       = None
        self._tree_inc_          = None
        self._nn_d_              = None
        self._nn_i_              = None
        self._d_core_            = None

        # private slots
        self.__mst_last_params_ = None  # cache for the MST


    def _check_params(self):
        self.verbose = bool(self.verbose)

        self.M = int(self.M)
        if self.M < 0: raise ValueError("M must be >= 0.")

        self.metric = str(self.metric).lower()
        if self.metric in ["euclidean", "lp:p=2"]:
            self.metric = "l2"
        elif self.metric in ["cosine"]:
            self.metric = "cosinesimil"
        elif self.metric in ["manhattan", "cityblock", "lp:p=1"]:
            self.metric = "l1"
        elif self.metric in ["chebyshev", "maximum", "lp:p=inf"]:
            self.metric = "linf"
        elif self.metric in ["euclidean_sparse"]:
            self.metric = "l2_sparse"
        elif self.metric in ["manhattan_sparse", "cityblock_sparse"]:
            self.metric = "l1_sparse"
        elif self.metric in ["chebyshev_sparse", "maximum_sparse"]:
            self.metric = "linf_sparse"
        elif self.metric in ["cosine_sparse"]:
            self.metric = "cosinesimil_sparse"
        elif self.metric in ["cosine_sparse_fast"]:
            self.metric = "cosinesimil_sparse_fast"

        _metrics = ("l2", "l1", "cosinesimil", "precomputed")
        if self.metric not in _metrics:
            raise ValueError("metric should be one of %s" % repr(_metrics))

        if self.quitefastmst_params is None:
            self.quitefastmst_params = dict()
        elif type(self.quitefastmst_params) is not dict:
            raise ValueError("quitefastmst_params must be a dict")


    def _check_mst(self):
        assert self._tree_d_ is not None
        assert self._tree_d_.shape[0] == self.n_samples_-1
        assert self._tree_i_.shape[0] == self.n_samples_-1
        assert self._tree_i_.shape[1] == 2
        assert core.is_increasing(self._tree_d_)
        if self.M >= 1:
            assert self._nn_d_.shape[0]    == self.n_samples_
            assert self._nn_d_.shape[1]    == self.M
            assert self._nn_i_.shape[0]    == self.n_samples_
            assert self._nn_i_.shape[1]    == self.M
            assert self._d_core_.shape[0]  == self.n_samples_
        assert self._tree_cumdeg_.shape[0] == self.n_samples_+1
        assert self._tree_inc_.shape[0]    == 2*(self.n_samples_-1)


    def _get_mst(self, X):
        # call _check_params() first!

        if X is None:
            # reuse last
            if self.__mst_last_params_ is None or self._tree_d_ is None:
                raise ValueError("fit() has not been called yet")
            self.metric = self.__mst_last_params_["metric"]
            self.quitefastmst_params = self.__mst_last_params_["quitefastmst_params"]
            self.M = self.__mst_last_params_["M"]
        elif isinstance(X, MSTBase):
            if X.__mst_last_params_ is None or X._tree_d_ is None:
                raise ValueError("fit() has not been called yet")
            self.__dict__.update(X.__dict__)  # copy all attributes, overwriting params!
        elif self.__mst_last_params_ is not None and \
                self.__mst_last_params_["id(X)"] == id(X) and \
                self.__mst_last_params_["metric"] == self.metric and \
                self.__mst_last_params_["quitefastmst_params"] == self.quitefastmst_params and \
                self.__mst_last_params_["M"] == self.M:
            # the parameters did not change;
            # and so the tree does not have to be recomputed
            pass
        else:
            self.__mst_last_params_ = None

        if self.__mst_last_params_ is not None:
            self._check_mst()
            return

        tree_d     = None
        tree_i     = None
        cumdeg     = None
        inc        = None
        nn_d       = None
        nn_i       = None
        d_core     = None
        n_features = None
        n_samples  = None

        id_X = id(X)  # before the conversion
        X = np.asarray(X, dtype=float, order="C")

        if self.metric == "precomputed":
            X = X.reshape(X.shape[0], -1)  # ensure it's a matrix
            if X.shape[1] == 1:
                # from a very advanced and sophisticated quadratic equation:
                n_samples = int(round((math.sqrt(1.0+8.0*X.shape[0])+1.0)/2.0))
                assert n_samples*(n_samples-1)//2 == X.shape[0]
            elif X.shape[1] == X.shape[0]:
                n_samples  = X.shape[0]
            else:
                raise ValueError(
                    "X must be distance vector or a square-form distance "
                    "matrix; see scipy.spatial.distance.pdist or "
                    "scipy.spatial.distance.squareform.")
        else:
            if X.ndim != 2: raise ValueError("X must be a matrix")
            n_samples  = X.shape[0]
            n_features = X.shape[1]

        if self.M >= n_samples: raise ValueError("M is too large")

        if self.metric == "l2":  # use quitefastmst
            _res = quitefastmst.mst_euclid(
                X,
                M=self.M,
                **self.quitefastmst_params,
                verbose=self.verbose
            )

            if self.M == 0:
                tree_d, tree_i = _res
            else:
                tree_d, tree_i, nn_d, nn_i = _res
                #d_core = internal.get_d_core_(nn_d, nn_i, self.M)
                d_core = nn_d[:, self.M-1].copy()  # make it contiguous
        else:
            from . import oldmst
            if self.M >= 1:  # else d_core   = None
                # determine d_core
                nn_d, nn_i = oldmst.knn_from_distance(
                    X,  # if not c_contiguous, raises an error
                    k=self.M,
                    metric=self.metric,  # supports "precomputed"
                    verbose=self.verbose
                )
                #d_core = internal.get_d_core_(nn_d, nn_i, self.M)
                d_core = nn_d[:, self.M-1].copy()  # make it contiguous

            # Use Prim's algorithm to determine the MST
            # w.r.t. the distances computed on the fly
            tree_d, tree_i = oldmst.mst_from_distance(
                X,  # if not c_contiguous, raises an error
                metric=self.metric,
                d_core=d_core,
                verbose=self.verbose
            )

        cumdeg, inc = core.graph_vertex_incidences(tree_i, n_samples)

        self.n_samples_     = n_samples
        self.n_features_    = n_features
        self._tree_d_       = tree_d
        self._tree_i_       = tree_i
        self._tree_cumdeg_  = cumdeg
        self._tree_inc_     = inc
        self._nn_d_         = nn_d
        self._nn_i_         = nn_i
        self._d_core_       = d_core

        self._check_mst()

        self.__mst_last_params_ = dict()
        self.__mst_last_params_["id(X)"] = id_X
        self.__mst_last_params_["metric"] = self.metric
        self.__mst_last_params_["quitefastmst_params"] = self.quitefastmst_params
        self.__mst_last_params_["M"] = self.M


    def fit_predict(self, X, y=None, **kwargs):
        """
        Performs cluster analysis of a dataset and returns the predicted labels.


        Parameters
        ----------

        X : object
            Typically a matrix or a data frame with ``n_samples`` rows
            and ``n_features`` columns; see below for more details and options.

        y : None
            Ignored.

        **kwargs : dict
            Arguments to be passed to :any:`fit`.


        Returns
        -------

        labels_ : ndarray of shape (n_samples_)
            `self.labels_` attribute.


        Notes
        -----

        Acceptable `X` types are as follows.

        If `X` is None, then the MST is not recomputed; the last spanning tree
        as well as the corresponding `M`, `metric`, and `quitefastmst_params`
        parameters are selected.

        If `X` is an instance of ``MSTBase`` (or its descendant), all
        its attributes are copied into the current object (including its MST).

        For ``metric=="precomputed"``, `X` should either
        be a distance vector of length ``n_samples*(n_samples-1)/2``
        (see :any:`scipy.spatial.distance.pdist`) or a square distance matrix
        of shape ``(n_samples, n_samples)``
        (see :any:`scipy.spatial.distance.squareform`).

        Otherwise, `X` should be real-valued matrix
        (dense ``numpy.ndarray`` or an object coercible to)
        with ``n_samples`` rows and ``n_features`` columns.

        As in the case of all the distance-based methods (including
        k-nearest neighbours and DBSCAN), the standardisation of the input
        features is definitely worth giving a try.  Oftentimes, applying
        feature selection and engineering techniques (e.g., dimensionality
        reduction) might lead to more meaningful results.
        """
        if y is not None:  # it is not a transductive classifier
            raise ValueError("y should be None")

        self.fit(X, **kwargs)
        return self.labels_



class MSTClusterer(MSTBase):
    """
    A base class for :any:`genieclust.Genie`, :any:`genieclust.GIc`,
    :any:`lumbermark.Lumbermark`, and other spanning tree-based clustering
    algorithms [1]_.

    By removing `k-1` edges from a spanning tree, we form `k` connected
    components which can be conceived as clusters.


    Parameters
    ----------

    M, metric, quitefastmst_params, verbose
        see :any:`deadwood.MSTBase`

    n_clusters : int
        The number of clusters to detect.


    Attributes
    ----------

    n_samples_, n_features_
            see :any:`deadwood.MSTBase`

    labels_ : ndarray of shape (n_samples_,)
        Detected cluster labels.

        ``labels_[i]`` gives the cluster ID of the `i`-th input point
        (between 0 and `n_clusters_`-1).

        Eventual outliers are labelled ``-1``.

    n_clusters_ : int
        The actual number of clusters detected by the algorithm.

        If it is different from the requested one, a warning is generated.

    _cut_edges_ : ndarray of shape (n_clusters_-1,)
        Indexes of the MST edges whose removal forms the `n_clusters_` detected
        connected components (clusters).


    References
    ----------

    .. [1]
        M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        https://doi.org/10.1007/s00357-024-09483-1
    """
    def __init__(
            self,
            n_clusters=2,
            *,
            M=0,
            metric="l2",
            quitefastmst_params=None,
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__(
            M=M,
            metric=metric,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.n_clusters  = n_clusters  # requested number of clusters

        self.n_clusters_ = None  # actual number of clusters detected
        self._cut_edges_ = None  # length n_clusters_-1


    def _check_params(self):
        super()._check_params()

        self.n_clusters = int(self.n_clusters)
        if self.n_clusters < 0: raise ValueError("n_clusters must be >= 0")


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "clusterer"
        if tags.transformer_tags is not None:
            tags.transformer_tags.preserves_dtype = []
        return tags



class MSTOutlierDetector(MSTBase):
    """
    A base class for :any:`deadwood.Deadwood` and other
    spanning tree-based outlier detection algorithms.


    Parameters
    ----------

    M, metric, quitefastmst_params, verbose
        see :any:`deadwood.MSTBase`


    Attributes
    ----------

    n_samples_, n_features_
            see :any:`deadwood.MSTBase`

    labels_ : ndarray of shape (n_samples_,)
        ``labels_[i]`` gives the inlier (1) or outlier (-1) status
        of the `i`-th input point.
    """
    def __init__(
            self,
            *,
            M=0,
            metric="l2",
            quitefastmst_params=None,
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__(
            M=M,
            metric=metric,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "outlier_detector"
        if tags.transformer_tags is not None:
            tags.transformer_tags.preserves_dtype = []
        return tags


class Deadwood(MSTOutlierDetector):
    """
    Deadwood [1]_ is an anomaly detection algorithm based on Mutual
    Reachability Minimum Spanning Trees.  It trims protruding tree segments
    and marks small debris as outliers.

    More precisely, the use of a mutual reachability distance [3]_
    pulls peripheral points farther away from each other.
    Tree edges with weights beyond the detected elbow point [2]_
    are removed.  All the resulting connected components whose
    sizes are smaller than a given threshold are considered outliers.


    Parameters
    ----------

    M, metric, quitefastmst_params, verbose
        see :any:`deadwood.MSTBase`

    contamination : 'auto' or float, default='auto'
        The estimated (approximate) proportion of outliers in the data set.
        If ``"auto"``, the contamination amount will be determined
        by identifying the most significant elbow point of the curve
        comprised of increasingly ordered tree edge weights.

    max_debris_size : 'auto' or int, default='auto'
        The maximal size of the leftover connected components that
        will be considered outliers.  If ``"auto"``, ``sqrt(n_samples)``
        is assumed.


    Attributes
    ----------

    n_samples_, n_features_
            see :any:`deadwood.MSTBase`

    labels_ : ndarray of shape (n_samples_,)
        ``labels_[i]`` gives the inlier (1) or outlier (-1) status
        of the `i`-th input point.

    contamination_ : float or ndarray of shape (n_clusters_,)
        Detected contamination threshold(s) (elbow point(s)).
        For the actual number of outliers detected, compute
        ``np.mean(labels_<0)``.

    max_debris_size_ : float
        Computed max debris size.

    _cut_edges_ : None or ndarray of shape (n_clusters_-1,)
        Indexes of MST edges whose removal forms n_clusters_ connected
        components (clusters) in which outliers are to be sought.
        This parameter is usually set by ``fit`` called on a ``MSTClusterer``.


    References
    ----------

    .. [1]
        M. Gagolewski, *deadwood*, in preparation, 2026, TODO

    .. [2]
        V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan, *Finding a "Kneedle"
        in a haystack: Detecting knee points in system behavior*,
        In: *31st Intl. Conf. Distributed Computing Systems Workshops*,
        2011, 166-171, https://doi.org/10.1109/ICDCSW.2011.20

    .. [3]
        R.J.G.B. Campello, D. Moulavi, J. Sander,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        https://doi.org/10.1007/978-3-642-37456-2_14
    """
    def __init__(
            self,
            *,
            contamination="auto",
            max_debris_size="auto",
            M=0,  #TODO: set default
            metric="l2",
            quitefastmst_params=dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__(
            M=M,
            metric=metric,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.contamination      = contamination
        self.max_debris_size    = max_debris_size

        self.contamination_     = None
        self.max_debris_size_   = None

        self._max_contamination = 0.5
        self._ema_dt            = 0.01  # controls the exponential moving average smoothing parameter alpha = 1-exp(-dt) (in elbow detection)

        self._cut_edges_        = None  # actually, _cut_edges


    def _check_params(self):
        super()._check_params()

        self._max_contamination = float(self._max_contamination)
        if not 0.0 <= self._max_contamination <= 1.0:
            raise ValueError("_max_contamination must be in [0, 1]")

        self._ema_dt = float(self._ema_dt)
        if self._ema_dt <= 0.0:
            raise ValueError("_ema_dt must be > 0")

        if self.contamination == "auto":
            pass
        else:
            self.contamination = float(self.contamination)
            if not 0 <= self.contamination <= self._max_contamination:
                raise ValueError("contamination must be 'auto' or in [0, %g]" % self._max_contamination)

        if self.max_debris_size == "auto":
            pass
        else:
            self.max_debris_size = int(self.max_debris_size)
            if self.max_debris_size <= 0:
                raise ValueError("max_debris_size must be 'auto' or > 0")


    # @staticmethod
    # def _get_contamination(
    #         tree_d, max_contamination=0.5, ema_dt=0.01
    #     ):
    #     # finds the elbow point of the right part of the sorted edge length curve (quantiles)
    #     assert core.is_increasing(tree_d)
    #
    #     m = tree_d.shape[0]
    #     shift = int(m*(1.0-max_contamination))
    #     elbow_index = core.kneedle_increasing(tree_d[shift:], convex=True, dt=ema_dt)
    #
    #     if elbow_index == 0:
    #         return 0.0
    #     else:
    #         index = shift+elbow_index+1  # int(m*(1.0-contamination))
    #         contamination = (m-index)/(m+1)
    #         return contamination
    #
    #
    # @staticmethod
    # def _find_outliers(
    #         tree_i,
    #         contamination=0.1,
    #         max_debris_size=50,
    #         tree_cumdeg=None,
    #         tree_inc=None
    #     ):
    #     m = tree_i.shape[0]
    #     if contamination <= 0.0:
    #         return np.zeros(m+1, bool)
    #
    #     weight_threshold_index = int(m*(1.0-contamination))
    #
    #     # tree_i is sorted increasingly wrt weights
    #     skip_edges = np.zeros(m, bool)
    #     skip_edges[weight_threshold_index:] = True
    #
    #     node_labels, cluster_sizes = core.mst_cluster_sizes(
    #         tree_i, skip_edges, tree_cumdeg, tree_inc
    #     )
    #
    #     return (cluster_sizes[node_labels] <= max_debris_size)
    #
    #
    # def _fit_single(self):
    #     if self.contamination == "auto":
    #         self.contamination_ = Deadwood._get_contamination(
    #             self._tree_d_, self._max_contamination, self._ema_dt
    #         )
    #     else:
    #         self.contamination_ = self.contamination
    #
    #     return Deadwood._find_outliers(
    #         self._tree_i_, self.contamination_, self.max_debris_size_,
    #         self._tree_cumdeg_, self._tree_inc_
    #     )
    #
    #
    # def _fit_multi(self):
    #     assert self._cut_edges_ is not None
    #     assert np.all(self._cut_edges_ >= 0)
    #     assert np.all(self._cut_edges_ < self._tree_d_.shape[0])
    #
    #     m = self._tree_i_.shape[0]
    #     skip_edges = np.zeros(m, bool)
    #     skip_edges[self._cut_edges_] = True
    #     node_labels, cluster_sizes = core.mst_cluster_sizes(
    #         self._tree_i_, skip_edges, self._tree_cumdeg_, self._tree_inc_
    #     )
    #
    #     k = cluster_sizes.shape[0]
    #     edge_labels = node_labels[self._tree_i_[:,0]]
    #     edge_labels[edge_labels != node_labels[self._tree_i_[:,1]]] = -1
    #     assert np.sum(edge_labels<0) == k-1
    #
    #     tree_d_grp, ind = core.sort_groups(self._tree_d_, edge_labels, k)
    #
    #     self.contamination_ = np.empty(k, float)
    #     weight_thresholds = np.empty(k, float)
    #     for i in range(k):
    #         mi = cluster_sizes[i]-1  # ind[i+1]-ind[i]
    #         if self.contamination == "auto":
    #             contamination = Deadwood._get_contamination(
    #                 tree_d_grp[ind[i]:ind[i+1]],
    #                 self._max_contamination, self._ema_dt
    #             )
    #         else:
    #             contamination = self.contamination
    #
    #         self.contamination_[i] = contamination
    #         weight_thresholds[i] = tree_d_grp[ind[i]+int(mi*(1.0-contamination))]
    #         skip_edges[(edge_labels == i) & (self._tree_d_>=weight_thresholds[i])] = True
    #
    #
    #     node_labels, cluster_sizes = core.mst_cluster_sizes(
    #         self._tree_i_, skip_edges, self._tree_cumdeg_, self._tree_inc_
    #     )
    #
    #     return (cluster_sizes[node_labels] <= self.max_debris_size_)


    def fit(self, X, y=None):
        """
        Detects outliers in a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix or a data frame with ``n_samples`` rows
            and ``n_features`` columns;
            see :any:`deadwood.MSTBase.fit_predict` for more details.

            If `X` is an instance of :any:`deadwood.MSTClusterer` (with
            ``fit`` method already invoked), e.g., :any:`genieclust.Genie` or
            :any:`lumbermark.Lumbermark`, then the outliers are sought
            separately in each detected cluster.  Note that this overrides
            the ``M``, ``metric``, and ``quitefastmst_params`` parameters
            (amongst others).

            If `X` is None, then the outlier detector is applied on the
            same MST as that determined by the previous call to ``fit``.
            This way we may try out different values of `contamination`
            and `max_debris_size` much more cheaply.


        y : None
            Ignored.


        Returns
        -------

        self : deadwood.Deadwood
            The object that the method was called on.


        Notes
        -----

        Refer to the `labels_` attribute for the result.
        """
        self.labels_ = None

        self._check_params()  # re-check, they might have changed
        self._get_mst(X)  # sets n_samples_, n_features_, _tree_w, _tree_i, _d_core, etc.

        if self.max_debris_size == "auto":
            self.max_debris_size_ = max(1, int(np.sqrt(self.n_samples_)))
        else:
            self.max_debris_size_ = self.max_debris_size

        if self.verbose:
            print("[deadwood] Finding outliers.", file=sys.stderr)

        is_outlier_, contamination_ = core.deadwood_from_mst(
            self._tree_d_,
            self._tree_i_,
            self._cut_edges_ if self._cut_edges_ is not None else np.empty(0, np.intp),
            self._tree_cumdeg_,
            self._tree_inc_,
            max_contamination=self._max_contamination if self.contamination == "auto" else -self.contamination,
            ema_dt=self._ema_dt,
            max_debris_size=self.max_debris_size_
        )

        # if self._cut_edges_ is None:
        #     is_outlier = self._fit_single()
        # else:
        #     is_outlier = self._fit_multi()

        self.labels_ = 1-2*(is_outlier_.astype(int))
        self.contamination_ = contamination_ if contamination_.shape[0]>1 else contamination_[0]

        if self.verbose:
            print("[deadwood] Done.", file=sys.stderr)

        return self

# ###############################################################################
# ###############################################################################
# ###############################################################################
#
#
# class MSTClusterMixinWithProcessors(MSTClusterMixin):
#     """
#     TODO
#
#     TODO For *M > 0*, the underlying clustering algorithm by default
#     leaves out all MST leaves from the clustering process.  Afterwards,
#     some of them (midliers) are merged with the nearest clusters at the
#     postprocessing stage, and other ones are marked as outliers.
#
#
#     Parameters
#     ----------
#
#     n_clusters
#         See :any:`deadwood.MSTClusterMixin`.
#
#     M
#         See :any:`deadwood.MSTClusterMixin`.
#
#     metric
#         See :any:`deadwood.MSTClusterMixin`.
#
#     quitefastmst_params
#         See :any:`deadwood.MSTClusterMixin`.
#
#     verbose
#         See :any:`deadwood.MSTClusterMixin`.
#
#     preprocess : TODO
#         TODO
#
#     postprocess : {``"midliers"``, ``"none"``, ``"all"``}
#         Controls the treatment of outliers after the clusters once identified.
#
#         TODO In effect only if *M > 0*. Each leaf in the spanning tree
#         is omitted from the clustering process.  We call it a *midlier*
#         if it is amongst its adjacent vertex's `M` nearest neighbours.
#         By default, only midliers are merged with their nearest
#         clusters, and the remaining leaves are considered outliers.
#
#         To force a classical `n_clusters`-partition of a data set (one that
#         marks no points as outliers), choose ``"all"``.
#
#         Furthermore, ``"none"`` leaves all leaves marked as outliers.
#
#
#     Attributes
#     ----------
#
#     See :any:`deadwood.MSTClusterMixin`.
#     """
#
#     def __init__(
#             self,
#             *,
#             n_clusters=2,
#             M=0,
#             metric="l2",
#             preprocess="none",
#             postprocess="none",
#             quitefastmst_params=dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
#             verbose=False
#         ):
#         # # # # # # # # # # # #
#         super().__init__(
#             n_clusters=n_clusters,
#             M=M,
#             metric=metric,
#             quitefastmst_params=quitefastmst_params,
#             verbose=verbose
#         )
#
#         self.preprocess          = preprocess
#         self.postprocess         = postprocess
#
#
#     def _check_params(self, cur_state=None):
#         cur_state = super()._check_params(cur_state)
#
#         _preprocess_options = ("auto", "none", "leaves")  # TODO
#         cur_state["preprocess"] = str(self.preprocess).lower()
#         if cur_state["preprocess"] not in _preprocess_options:
#             raise ValueError("`preprocess` should be one of %s" % repr(_preprocess_options))
#
#         _postprocess_options = ("midliers", "none", "all")
#         cur_state["postprocess"] = str(self.postprocess).lower()
#         if cur_state["postprocess"] not in _postprocess_options:
#             raise ValueError("`postprocess` should be one of %s" % repr(_postprocess_options))
#
#         if cur_state["preprocess"] == "auto":
#             if self.M > 0:
#                 cur_state["preprocess"] = "leaves"
#             else:
#                 cur_state["preprocess"] = "none"
#
#         return cur_state
#
#
#     def _postprocess_outputs(self, res, cur_state):
#         """
#         (internal) Updates `self.labels_`
#         """
#         if self.verbose:
#             print("[deadwood] Postprocessing outputs.", file=sys.stderr)
#
#         self.labels_     = res["labels"]
#         self._links_     = res["links"]
#         self._iters_     = res["iters"]
#
#         if res["n_clusters"] != cur_state["n_clusters"]:
#             warnings.warn("The number of clusters detected (%d) is "
#                           "different from the requested one (%d)." % (
#                             res["n_clusters"],
#                             cur_state["n_clusters"]))
#         self.n_clusters_ = res["n_clusters"]
#
#         if self.labels_ is not None:
#             reshaped = False
#             if self.labels_.ndim == 1:
#                 reshaped = True
#                 # promote it to a matrix with 1 row
#                 self.labels_.shape = (1, self.labels_.shape[0])
#                 start_partition = 0
#             else:
#                 # duplicate the 1st row (create the "0"-partition that will
#                 # not be postprocessed):
#                 self.labels_ = np.vstack((self.labels_[0, :], self.labels_))
#                 start_partition = 1  # do not postprocess the "0"-partition
#
#             #self._is_noise    = (self.labels_[0, :] < 0)
#
#             # postprocess labels, if requested to do so
#
#             if cur_state["preprocess"] != "leaves":
#                 # do nothing
#                 pass
#             elif cur_state["postprocess"] == "midliers":
#                 assert self._nn_i_ is not None
#                 assert self._nn_i_.shape[1] >= self.M
#                 for i in range(start_partition, self.labels_.shape[0]):
#                     self.labels_[i, :] = internal.merge_midliers(
#                         self._tree_i_, self.labels_[i, :],
#                         self._nn_i_, self.M
#                     )
#             elif cur_state["postprocess"] == "all":
#                 for i in range(start_partition, self.labels_.shape[0]):
#                     self.labels_[i, :] = internal.merge_all(
#                         self._tree_i_, self.labels_[i, :]
#                     )
#             # elif cur_state["postprocess"] == "none":
#             #     pass
#
#         if reshaped:
#             self.labels_.shape = (self.labels_.shape[1], )
#
#        return cur_state

