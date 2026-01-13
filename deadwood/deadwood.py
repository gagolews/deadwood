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

from sklearn.base import BaseEstimator, ClusterMixin

###############################################################################
###############################################################################
###############################################################################




class MSTClusterMixin(BaseEstimator, ClusterMixin):
    """
    The base class for :any:`genieclust.Genie`,
    :any:`genieclust.GIc`, and other MST-based clustering algorithms [2]_.


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect.

        If *M > 0* and `postprocess` is not ``"all"``, setting
        *n_clusters = 1* turns the algorithm to an outlier detector.

    M : int
        Smoothing factor for the mutual reachability distance [1]_.
        *M = 0* and *M = 1* indicate the original distance as given by
        the `metric` parameter.

    metric : str
        The metric used to compute the linkage.

        One of:
        ``"l2"`` (synonym: ``"euclidean"``; the default),
        ``"l1"`` (synonym: ``"manhattan"``, ``"cityblock"``),
        ``"cosinesimil"`` (synonym: ``"cosine"``), or
        ``"precomputed"``.

        For ``"precomputed"``, the `X` argument to the ``fit`` method
        must be a distance vector or a square-form distance matrix;
        see ``scipy.spatial.distance.pdist``.

        Determining minimum spanning trees with respect to the Euclidean
        distance (the default) is much faster than with other metrics
        thanks to the `quitefastmst <https://quitefastmst.gagolewski.com/>`__
        package.

    quitefastmst_params : dict
        Additional parameters to be passed to ``quitefastmst.mst_euclid``
        if ``metric`` is ``"l2"``

    verbose : bool
        Whether to print diagnostic messages and progress information
        onto ``stderr``.


    Notes
    -----

    If the Euclidean distance is selected, then ``quitefastmst.mst_euclid`` is
    used to compute the MST; it is quite fast in low-dimensional spaces.
    Otherwise, an implementation of the Jarník (Prim/Dijkstra)-like
    :math:`O(n^2)`-time algorithm is called.

    If *M > 0*, then the minimum spanning tree is computed with respect to the
    mutual reachability distance (based, e.g., on the Euclidean metric) [1]_.
    Formally, the distance :math:`d_M(i,j)` is used instead of the
    chosen "raw" distance, :math:`d(i,j)`. It holds
    :math:`d_M(i,j)=\\max\\{d(i,j), c_M(i), c_M(j)\\}`, where the "core"
    distance :math:`c_M(i)` is given by :math:`d(i,k)` with :math:`k` being
    :math:`i`'s :math:`M`-th nearest neighbour
    (not including self, unlike in [1]_).
    This pulls outliers away from their neighbours.

    If ``quitefastmst`` is used, then possible ties between mutually
    reachability distances are resolved in such a way that connecting
    to a neighbour of the smallest core distance is preferred.
    This leads to MSTs with more leaves and hubs.  Moreover, the leaves are
    then reconnected in such a way that they become incident with vertices
    that have them amongst their *M* nearest neighbours (if this is possible
    without violating the minimality condition); see [3]_ for discussion.


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used for computing the minimum
            spanning tree.


    Attributes
    ----------

    labels_ : ndarray
        Detected cluster labels.

        Normally an integer vector such that ``labels_[i]`` gives
        the cluster ID (between 0 and `n_clusters_` - 1) of the `i`-th object.
        Outliers may be labelled ``-1`` (depends on the underlying algorithm).

    n_clusters_ : int
        The actual number of clusters detected by the algorithm.

        It can be different from the requested one if there are too many
        noise points in the dataset.

    n_samples_ : int
        The number of points in the dataset.

    n_features_ : int
        The number of features in the dataset.

        If the information is not available, it will be set to ``-1``.



    References
    ----------

    .. [1]
        Campello R.J.G.B., Moulavi D., Sander J.,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        DOI:10.1007/978-3-642-37456-2_14.

    .. [2]
        Gagolewski M., Cena A., Bartoszuk M., Brzozowski L.,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        DOI:10.1007/s00357-024-09483-1.

    .. [3]
       Gagolewski M., TODO, 2025

    """

    def __init__(
            self,
            *,
            n_clusters=2,
            M=0,
            metric="l2",
            quitefastmst_params=dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__()
        self.n_clusters          = n_clusters
        self.n_features          = None  # can be overwritten by GIc
        self.M                   = M
        self.metric              = metric
        self.quitefastmst_params = quitefastmst_params
        self.verbose             = verbose

        self.n_samples_          = None
        self.n_features_         = None
        self.n_clusters_         = 0  # should not be confused with self.n_clusters
        self.labels_             = None
        self.children_           = None
        self.distances_          = None
        self.counts_             = None

        self._is_noise           = None  # TODO
        self._tree_cutlist       = None  # TODO
        self._tree_w             = None
        self._tree_e             = None
        self._nn_w               = None
        self._nn_e               = None
        self._d_core             = None
        self._links_             = None
        self._iters_             = None

        self._last_state         = None


    def _check_params(self, cur_state=None):
        if cur_state is None:
            cur_state = dict()

        cur_state["M"] = int(self.M)
        if cur_state["M"] < 0:
            raise ValueError("`M` must be >= 0.")

        # cur_state["exact"]             = True  # bool(self.exact)

        cur_state["verbose"]           = bool(self.verbose)

        cur_state["n_clusters"] = int(self.n_clusters)
        if cur_state["n_clusters"] < 0:
            raise ValueError("n_clusters must be >= 0")

        # cur_state["preprocess"] = str(self.preprocess).lower()  # see _postprocess_outputs

        cur_state["metric"] = str(self.metric).lower()
        if cur_state["metric"] in ["euclidean", "lp:p=2"]:
            cur_state["metric"] = "l2"
        elif cur_state["metric"] in ["euclidean_sparse"]:
            cur_state["metric"] = "l2_sparse"
        elif cur_state["metric"] in ["manhattan", "cityblock", "lp:p=1"]:
            cur_state["metric"] = "l1"
        elif cur_state["metric"] in ["manhattan_sparse", "cityblock_sparse"]:
            cur_state["metric"] = "l1_sparse"
        elif cur_state["metric"] in ["chebyshev", "maximum", "lp:p=inf"]:
            cur_state["metric"] = "linf"
        elif cur_state["metric"] in ["chebyshev_sparse", "maximum_sparse"]:
            cur_state["metric"] = "linf_sparse"
        elif cur_state["metric"] in ["cosine"]:
            cur_state["metric"] = "cosinesimil"
        elif cur_state["metric"] in ["cosine_sparse"]:
            cur_state["metric"] = "cosinesimil_sparse"
        elif cur_state["metric"] in ["cosine_sparse_fast"]:
            cur_state["metric"] = "cosinesimil_sparse_fast"

        if type(self.quitefastmst_params) is not dict:
            raise ValueError("`quitefastmst_params` must be a dict")
        cur_state["quitefastmst_params"] = self.quitefastmst_params

        _metric_exact_options = (
            "l2", "l1", "cosinesimil", "precomputed")
        if cur_state["metric"] not in _metric_exact_options:
            raise ValueError("`metric` should be one of %s" % repr(_metric_exact_options))

        # this is more like an inherent dimensionality for GIc
        cur_state["n_features"] = self.n_features   # users can set this manually
        if cur_state["n_features"] is not None:      # only GIc needs this
            cur_state["n_features"] = max(1.0, float(cur_state["n_features"]))
        else:
            cur_state["n_features"] = -1

        return cur_state


    def _get_mst_exact(self, X, cur_state):

        if cur_state["metric"] == "precomputed":
            X = X.reshape(X.shape[0], -1)
            if X.shape[1] not in [1, X.shape[0]]:
                raise ValueError(
                    "`X` must be distance vector or a square-form distance "
                    "matrix; see `scipy.spatial.distance.pdist` or "
                    "`scipy.spatial.distance.squareform`.")
            if X.shape[1] == 1:
                # from a very advanced and sophisticated quadratic equation:
                n_samples = int(round((math.sqrt(1.0+8.0*X.shape[0])+1.0)/2.0))
                assert n_samples*(n_samples-1)//2 == X.shape[0]
            else:
                n_samples  = X.shape[0]
        else:
            # if cur_state["cast_float32"]:
            #     if scipy.sparse.isspmatrix(X):
            #         raise ValueError("Sparse matrices are (currently) only "
            #                          "supported when `exact` is False")
            #     X = np.asarray(X, dtype=np.float32, order="C")
            if X.ndim != 2: raise ValueError("`X` must be a matrix")
            n_samples  = X.shape[0]
            if cur_state["n_features"] < 0:
                cur_state["n_features"] = X.shape[1]

        if cur_state["M"] >= X.shape[0]:
            raise ValueError("`M` is too large")

        tree_w = None
        tree_e = None
        nn_w   = None
        nn_e   = None
        d_core = None

        if self._last_state is not None and \
                cur_state["X"]        == self._last_state["X"] and \
                cur_state["metric"]   == self._last_state["metric"] and \
                cur_state["quitefastmst_params"] == self._last_state["quitefastmst_params"] and \
                cur_state["M"]        == self._last_state["M"]:
            # reuse last MST & M-NNs
            tree_w = self._tree_w
            tree_e = self._tree_e
            nn_w   = self._nn_w
            nn_e   = self._nn_e
            d_core = self._d_core
        else:
            if cur_state["metric"] == "l2":
                _res = quitefastmst.mst_euclid(
                    X,
                    M=cur_state["M"],
                    **cur_state["quitefastmst_params"],
                    verbose=cur_state["verbose"]
                )

                if cur_state["M"] == 0:
                    tree_w, tree_e = _res
                    #d_core = None
                else:
                    tree_w, tree_e, nn_w, nn_e = _res
                    #d_core = internal.get_d_core(nn_w, nn_e, cur_state["M"])
                    d_core = nn_w[:, cur_state["M"]-1].copy()  # make it contiguous
            else:
                from . import oldmst
                if cur_state["M"] >= 1:  # else d_core   = None
                    # Genie+HDBSCAN --- determine d_core
                    nn_w, nn_e = oldmst.knn_from_distance(
                        X,  # if not c_contiguous, raises an error
                        k=cur_state["M"],
                        metric=cur_state["metric"],  # supports "precomputed"
                        verbose=cur_state["verbose"]
                    )
                    #d_core = internal.get_d_core(nn_w, nn_e, cur_state["M"])
                    d_core = nn_w[:, cur_state["M"]-1]

                # Use Prim's algorithm to determine the MST
                # w.r.t. the distances computed on the fly
                tree_w, tree_e = oldmst.mst_from_distance(
                    X,  # if not c_contiguous, raises an error
                    metric=cur_state["metric"],
                    d_core=d_core,
                    verbose=cur_state["verbose"]
                )

        assert tree_w.shape[0] == n_samples-1
        assert tree_e.shape[0] == n_samples-1
        assert tree_e.shape[1] == 2
        if cur_state["M"] >= 2:
            assert nn_w.shape[0] == n_samples
            assert nn_w.shape[1] == cur_state["M"]
            assert nn_e.shape[0] == n_samples
            assert nn_e.shape[1] == cur_state["M"]
            assert d_core.shape[0] == n_samples

        self.n_samples_    = n_samples
        self._tree_w       = tree_w
        self._tree_e       = tree_e
        self._nn_w         = nn_w
        self._nn_e         = nn_e
        self._d_core       = d_core

        self._is_noise     = None  # TODO
        self._tree_cutlist = None  # TODO

        return cur_state


    def _get_mst(self, X, cur_state):
        cur_state["X"] = id(X)

        if cur_state["verbose"]:
            print("[deadwood] Preprocessing data.", file=sys.stderr)

        # if cur_state["exact"]:
        cur_state = self._get_mst_exact(X, cur_state)
        # else: cur_state = self._get_mst_approx(X, cur_state)

        # this might be an "intrinsic" dimensionality:
        self.n_features_  = cur_state["n_features"]
        self._last_state  = cur_state  # will be modified in-place further on

        return cur_state


    def fit_predict(self, X, y=None):
        """
        Perform cluster analysis of a dataset and return the predicted labels.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns; see below for more details and options.

        y : None
            Ignored.


        Returns
        -------

        labels_ : ndarray
            `self.labels_` attribute.


        Notes
        -----

        Acceptable `X` types are as follows.

        For `metric` of ``"precomputed"``, `X` should either
        be a distance vector of length ``n_samples*(n_samples-1)/2``
        (see ``scipy.spatial.distance.pdist``) or a square distance matrix
        of shape ``(n_samples, n_samples)``
        (see ``scipy.spatial.distance.squareform``).

        Otherwise, `X` should be real-valued matrix
        (dense ``numpy.ndarray``, or an object coercible to)
        with ``n_samples`` rows and ``n_features`` columns.

        In the latter case, it might be a good idea to standardise
        or at least somehow preprocess the coordinates of the input data
        points by calling, for instance,
        ``X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)``
        so that the dataset is centred at 0 and has total variance of 1.
        This way the method becomes translation and scale invariant.
        What's more, if data are recorded with small precision (say, up
        to few decimal digits), adding a tiny bit of Gaussian noise will
        ensure the solution is unique (note that this generally applies
        to other distance-based clustering algorithms as well).

        For the clustering result, refer to the `labels_` and `n_clusters_`
        attributes.
        """
        self.fit(X)
        return self.labels_




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
#             if cur_state["M"] > 0:
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
#         if cur_state["verbose"]:
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
#                 assert self._nn_e is not None
#                 assert self._nn_e.shape[1] >= cur_state["M"]
#                 for i in range(start_partition, self.labels_.shape[0]):
#                     self.labels_[i, :] = internal.merge_midliers(
#                         self._tree_e, self.labels_[i, :],
#                         self._nn_e, cur_state["M"]
#                     )
#             elif cur_state["postprocess"] == "all":
#                 for i in range(start_partition, self.labels_.shape[0]):
#                     self.labels_[i, :] = internal.merge_all(
#                         self._tree_e, self.labels_[i, :]
#                     )
#             # elif cur_state["postprocess"] == "none":
#             #     pass
#
#         if reshaped:
#             self.labels_.shape = (self.labels_.shape[1], )
#
#        return cur_state

