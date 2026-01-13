# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3

"""
The "old" (<=2025), slow yet more universal functions to compute
k-nearest neighbours and minimum spanning trees.

See the `quitefastmst` <https://quitefastmst.gagolewski.com/>
package for faster algorithms working in the Euclidean space.
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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


import numpy as np
cimport numpy as np
np.import_array()
import os
import warnings

cimport libc.math
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport INFINITY


ctypedef fused T:
    int
    long
    long long
    Py_ssize_t
    float
    double

ctypedef fused floatT:
    float
    double




cdef extern from "../src/c_oldmst.h":

    cdef cppclass CDistance[T]:
        pass

    cdef cppclass CDistanceMutualReachability[T]: # inherits from CDistance
        CDistanceMutualReachability()
        CDistanceMutualReachability(const T* d_core, Py_ssize_t n, CDistance[T]* d_pairwise)

    cdef cppclass CDistanceEuclidean[T]: # inherits from CDistance
        CDistanceEuclidean()
        CDistanceEuclidean(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceEuclideanSquared[T]: # inherits from CDistance
        CDistanceEuclideanSquared()
        CDistanceEuclideanSquared(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceManhattan[T]: # inherits from CDistance
        CDistanceManhattan()
        CDistanceManhattan(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceCosine[T]: # inherits from CDistance
        CDistanceCosine()
        CDistanceCosine(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistancePrecomputedMatrix[T]: # inherits from CDistance
        CDistancePrecomputedMatrix()
        CDistancePrecomputedMatrix(T* d, Py_ssize_t n)

    cdef cppclass CDistancePrecomputedVector[T]: # inherits from CDistance
        CDistancePrecomputedVector()
        CDistancePrecomputedVector(T* d, Py_ssize_t n)

    void Cknn_from_complete[T](
        CDistance[T]* D, Py_ssize_t n, Py_ssize_t k,
        T* dist, Py_ssize_t* ind, bint verbose) except +

    void Cmst_from_complete[T](
        CDistance[T]* D, Py_ssize_t n,
        T* mst_dist, Py_ssize_t* mst_ind, bint verbose) except +

    #Py_ssize_t Cmst_from_nn[T](  # removed
    #    T* dist, Py_ssize_t* ind, const T* d_core, Py_ssize_t n, Py_ssize_t k,
    #    T* mst_dist, Py_ssize_t* mst_ind, int* maybe_inexact, bint verbose) except +






################################################################################

# cpdef tuple mst_from_nn(  # removed
#     floatT[:,::1] dist,
#     Py_ssize_t[:,::1] ind,
#     floatT[::1] d_core=None,
#     bint stop_disconnected=True,
#     bint stop_inexact=False,
#     bint verbose=False):
#     """
#     deadwood.oldmst.mst_from_nn(dist, ind, d_core=None, stop_disconnected=True, stop_inexact=False, verbose=False)
#
#     Computes a minimum spanning forest for a (<=k)-nearest neighbour graph [DEPRECATED]
#
#
#     Parameters
#     ----------
#
#     dist : ndarray
#         A ``c_contiguous`` `ndarray` of shape (n, k).
#         ``dist[i,:]`` is sorted nondecreasingly for all ``i``,
#         ``dist[i,j]`` gives the weight of the edge ``{i, ind[i,j]}``
#     ind : a ``c_contiguous`` ndarray, shape (n,k)
#         Defines edges of the input graph, interpreted as ``{i, ind[i,j]}``.
#     d_core : a ``c_contiguous`` ndarray, shape (n,), or ``None``
#         Core distances for computing the mutual reachability distance,
#         can be ``None``.
#     stop_disconnected : bool
#         Whether to raise an exception if the input graph is not connected.
#     stop_inexact : bool
#         Whether to raise an exception if the return MST is definitely
#         subobtimal.
#     verbose : bool
#         Whether to print diagnostic messages.
#
#
#     Returns
#     -------
#
#     tuple like ``(mst_dist, mst_ind)``
#         Defines the `n-1` edges of the resulting MST.
#         The `(n-1)`-ary array ``mst_dist`` is such that
#         ``mst_dist[i]`` gives the weight of the ``i``-th edge.
#         Moreover, ``mst_ind`` is a matrix with `n-1` rows and 2 columns,
#         where ``{mst_ind[i,0], mst_ind[i,1]}`` defines the ``i``-th edge of the tree.
#
#         The results are ordered w.r.t. nondecreasing weights.
#         For each ``i``, it holds ``mst_ind[i,0] < mst_ind[i,1]``.
#
#         If `stop_disconnected` is ``False``, then the weights of the
#         last `c-1` edges are set to infinity and the corresponding indexes
#         are set to -1, where `c` is the number of connected components
#         in the resulting minimum spanning forest.
#
#
#
#     See also
#     --------
#
#     Kruskal's algorithm is used.
#
#     Note that in general, the sum of weights in an MST of the (<= k)-nearest
#     neighbour graph might be greater than the sum of weights in a minimum
#     spanning tree of the complete pairwise distances graph.
#
#     If the input graph is not connected, the result is a forest.
#
#     """
#     cdef Py_ssize_t n = dist.shape[0]
#     cdef Py_ssize_t k = dist.shape[1]
#
#     if not (ind.shape[0] == n and ind.shape[1] == k):
#         raise ValueError("shapes of dist and ind must match")
#
#     cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
#     cdef np.ndarray[floatT]         mst_dist = np.empty(n-1,
#         dtype=np.float32 if floatT is float else np.float64)
#
#     cdef int maybe_inexact
#
#     cdef floatT* d_core_ptr = NULL
#     if d_core is not None:
#         if not (d_core.shape[0] == n):
#             raise ValueError("shapes of dist and d_core must match")
#         d_core_ptr = &d_core[0]
#
#     # _openmp_set_num_threads()
#     cdef Py_ssize_t n_edges = Cmst_from_nn(
#         &dist[0,0], &ind[0,0],
#         d_core_ptr,
#         n, k,
#         &mst_dist[0], &mst_ind[0,0], &maybe_inexact, verbose)
#
#     if stop_disconnected and n_edges < n-1:
#         raise ValueError("graph is disconnected")
#
#     if stop_inexact and maybe_inexact:
#         raise ValueError("MST maybe inexact")
#
#     return mst_dist, mst_ind




cpdef tuple mst_from_complete(
        floatT[:,::1] X,
        bint verbose=False
    ): # [:,::1]==c_contiguous
    """
    deadwood.oldmst.mst_from_complete(X, verbose=False)

    A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of a complete undirected graph
    with weights given by a symmetric n*n matrix
    or a distance vector of length n*(n-1)/2.

    The number of threads used is controlled via the
    OMP_NUM_THREADS environment variable or via
    `quitefastmst.omp_set_num_threads` at runtime.

    (*) Note that there might be multiple minimum trees spanning a given graph.


    References
    ----------

    .. [1]
        V. Jarník, O jistém problému minimálním,
        Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    .. [2]
        C.F. Olson, Parallel algorithms for hierarchical clustering,
        Parallel Computing 21(8) (1995) 1313–1325.

    .. [3]
        R. Prim, Shortest connection networks and some generalizations,
        The Bell System Technical Journal 36(6) (1957) 1389–1401.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n*(n-1)/2, 1) or (n,n)
        distance vector or matrix

    verbose: bool
        whether to print diagnostic messages


    Returns
    -------

    pair : tuple
        A pair (mst_dist, mst_ind) defining the n-1 edges of the MST:
          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;
          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
        For each i, it holds mst[i,0]<mst[i,1].
    """
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t n = X.shape[0]
    if d == 1:
        n = <Py_ssize_t>libc.math.round((libc.math.sqrt(1.0+8.0*n)+1.0)/2.0)
        assert n*(n-1)//2 == X.shape[0]

    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT]            mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef CDistance[floatT]* D = NULL
    if d == 1:
        D = <CDistance[floatT]*>new CDistancePrecomputedVector[floatT](&X[0,0], n)
    else:
        assert d == n
        D = <CDistance[floatT]*>new CDistancePrecomputedMatrix[floatT](&X[0,0], n)

    # _openmp_set_num_threads()
    Cmst_from_complete(D, n, &mst_dist[0], &mst_ind[0,0], verbose)

    if D:  del D

    return mst_dist, mst_ind




cpdef tuple mst_from_distance(
        floatT[:,::1] X,
        str metric="euclidean",
        floatT[::1] d_core=None,
        bint verbose=False
    ):
    """
    deadwood.oldmst.mst_from_distance(X, metric="euclidean", d_core=None, verbose=False)

    A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of X with respect to a given metric
    (distance).  Distances are computed on the fly.  Memory use: O(n*d).

    The number of threads used is controlled via the
    OMP_NUM_THREADS environment variable or via
    `quitefastmst.omp_set_num_threads` at runtime.


    References
    ----------

    .. [1]
        Jarník V., O jistém problému minimálním,
        Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    .. [2]
        Olson C.F., Parallel algorithms for hierarchical clustering,
        Parallel Computing 21(8) (1995) 1313–1325.

    .. [3]
        Prim R., Shortest connection networks and some generalizations,
        The Bell System Technical Journal 36(6) (1957) 1389–1401.

    .. [4] Campello R.J.G.B., Moulavi D., Sander J.,
        Density-based clustering based on hierarchical density estimates,
        Lecture Notes in Computer Science 7819 (2013) 160-172.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n,d) or,
            if metric == "precomputed", (n*(n-1)/2,1) or (n,n)
        n data points in a feature space of dimensionality d
        or pairwise distances between n points

    metric : string
        one of ``"euclidean"`` (a.k.a. ``"l2"``),
        ``"manhattan"`` (synonyms: ``"cityblock"``, ``"l1"``),
        ``"cosine"`` (a.k.a. ``"cosinesimil"``), or ``"precomputed"``.
        More metrics/distances might be supported in future versions.

    d_core : c_contiguous ndarray of length n; optional (default=None)
        core distances for computing the mutual reachability distance

    verbose: bool
        whether to print diagnostic messages

    Returns
    -------

    pair : tuple
        A pair (mst_dist, mst_ind) defining the n-1 edges of the MST:

          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;

          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree.

        The results are ordered w.r.t. nondecreasing weights.
        For each i, it holds mst[i,0]<mst[i,1].
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    if d == 1 and metric == "precomputed":
        n = <Py_ssize_t>libc.math.round((libc.math.sqrt(1.0+8.0*n)+1.0)/2.0)
        assert n*(n-1)//2 == X.shape[0]
    #cdef Py_ssize_t i
    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT] mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef CDistance[floatT]* D = NULL
    cdef CDistance[floatT]* D2 = NULL

    if metric == "euclidean" or metric == "l2":
        D = <CDistance[floatT]*>new CDistanceEuclidean[floatT](&X[0,0], n, d)
    elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
        D = <CDistance[floatT]*>new CDistanceManhattan[floatT](&X[0,0], n, d)
    elif metric == "cosine" or metric == "cosinesimil":
        D = <CDistance[floatT]*>new CDistanceCosine[floatT](&X[0,0], n, d)
    elif metric == "precomputed":
        if d == 1:
            D = <CDistance[floatT]*>new CDistancePrecomputedVector[floatT](&X[0,0], n)
        else:
            assert d == n
            D = <CDistance[floatT]*>new CDistancePrecomputedMatrix[floatT](&X[0,0], n)
    else:
        raise NotImplementedError("given `metric` is not supported (yet)")

    if d_core is not None:
        D2 = D  # must be deleted separately
        D  = <CDistance[floatT]*>new CDistanceMutualReachability[floatT](&d_core[0], n, D2)

    # _openmp_set_num_threads()
    Cmst_from_complete(D, n, &mst_dist[0], &mst_ind[0,0], verbose)

    if D:  del D
    if D2: del D2

    return mst_dist, mst_ind



cpdef tuple knn_from_distance(floatT[:,::1] X, Py_ssize_t k,
       str metric="euclidean", floatT[::1] d_core=None, bint verbose=False):
    """
    deadwood.oldmst.knn_from_distance(X, k, metric="euclidean", d_core=None, verbose=False)

    Determines the first k nearest neighbours of each point in X,
    with respect to a given metric (distance).
    Distances are computed on the fly.
    Memory use: O(n*k).

    It is assumed that each query point is not its own neighbour.

    The number of threads used is controlled via the
    OMP_NUM_THREADS environment variable or via
    `quitefastmst.omp_set_num_threads` at runtime.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n,d) or,
            if metric == "precomputed", (n*(n-1)/2,1) or (n,n)
        n data points in a feature space of dimensionality d
        or pairwise distances between n points
    k : int < n
        number of nearest neighbours
    metric : string
        one of ``"euclidean"`` (a.k.a. ``"l2"``),
        ``"manhattan"`` (synonyms: ``"cityblock"``, ``"l1"``),
        ``"cosine"`` (a.k.a. ``"cosinesimil"``), or ``"precomputed"``.
        More metrics/distances might be supported in future versions.
    d_core : c_contiguous ndarray of length n; optional (default=None)
        core distances for computing the mutual reachability distance
    verbose: bool
        whether to print diagnostic messages

    Returns
    -------

    pair : tuple
        A pair (dist, ind) representing the k-NN graph, where:
            dist : a c_contiguous ndarray, shape (n,k)
                dist[i,:] is sorted nondecreasingly for all i,
                dist[i,j] gives the weight of the edge {i, ind[i,j]},
                i.e., the distance between the i-th point and its j-th NN.
            ind : a c_contiguous ndarray, shape (n,k)
                edge definition, interpreted as {i, ind[i,j]};
                ind[i,j] is the index of the j-th nearest neighbour of i.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    if d == 1 and metric == "precomputed":
        n = <Py_ssize_t>libc.math.round((libc.math.sqrt(1.0+8.0*n)+1.0)/2.0)
        assert n*(n-1)//2 == X.shape[0]

    if k >= n:
        raise ValueError("too many nearest neighbours requested")

    cdef Py_ssize_t i
    cdef np.ndarray[Py_ssize_t,ndim=2] ind  = np.empty((n, k), dtype=np.intp)
    cdef np.ndarray[floatT,ndim=2]  dist = np.empty((n, k),
        dtype=np.float32 if floatT is float else np.float64)
    cdef CDistance[floatT]* D = NULL
    cdef CDistance[floatT]* D2 = NULL

    if metric == "euclidean" or metric == "l2":
        D = <CDistance[floatT]*>new CDistanceEuclidean[floatT](&X[0,0], n, d)
    elif metric == "manhattan" or metric == "cityblock" or metric == "l1":
        D = <CDistance[floatT]*>new CDistanceManhattan[floatT](&X[0,0], n, d)
    elif metric == "cosine" or metric == "cosinesimil":
        D = <CDistance[floatT]*>new CDistanceCosine[floatT](&X[0,0], n, d)
    elif metric == "precomputed":
        if d == 1:
            D = <CDistance[floatT]*>new CDistancePrecomputedVector[floatT](&X[0,0], n)
        else:
            assert d == n
            D = <CDistance[floatT]*>new CDistancePrecomputedMatrix[floatT](&X[0,0], n)
    else:
        raise NotImplementedError("given `metric` is not supported (yet)")

    if d_core is not None:
        D2 = D # must be deleted separately
        D  = <CDistance[floatT]*>new CDistanceMutualReachability[floatT](&d_core[0], n, D2)

    # _openmp_set_num_threads()
    Cknn_from_complete(D, n, k, &dist[0,0], &ind[0,0], verbose)

    if D:  del D
    if D2: del D2

    return dist, ind




################################################################################

# cpdef np.ndarray[floatT] _d_core_from_nn(
#         floatT[:,::1] dist,
#         Py_ssize_t[:,::1] ind,
#         Py_ssize_t M
#     ):
#     """
#     (provided for testing only)
#
#     Get "core" distance = distance to the M-th nearest neighbour
#     (if available, otherwise, distance to the furthest away one at hand).
#
#     Note that unlike in Campello et al.'s 2013 paper, the definition
#     of the core distance does not include the distance to self.
#
#
#     Parameters
#     ----------
#
#     dist : a c_contiguous ndarray, shape (n,k)
#         dist[i,:] is sorted nondecreasingly for all i,
#         dist[i,j] gives the weight of the edge {i, ind[i,j]}
#     ind : a c_contiguous ndarray, shape (n,k)
#         edge definition, interpreted as {i, ind[i,j]};
#         -1 denotes a "missing value"
#     M : int
#         "smoothing factor"
#
#
#     Returns
#     -------
#
#     ndarray
#         of length dist.shape[0]
#     """
#
#     cdef Py_ssize_t n = dist.shape[0]
#     cdef Py_ssize_t k = dist.shape[1]
#
#     if not (ind.shape[0] == n and ind.shape[1] == k):
#         raise ValueError("shapes of dist and ind must match")
#
#     if M > k:
#         raise ValueError("too few nearest neighbours provided")
#
#     cdef np.ndarray[floatT] d_core = np.empty(n,
#         dtype=np.float32 if floatT is float else np.float64)
#
#     #Python equivalent if all NNs are available:
#     #assert nn_dist.shape[1] >= cur_state["M"]
#     #d_core = nn_dist[:, cur_state["M"]-1].astype(X.dtype, order="C")
#
#     cdef Py_ssize_t i, j
#     for i in range(n):
#         j = M-1
#         while ind[i, j] < 0:
#             j -= 1
#             if j < 0: raise ValueError("no nearest neighbours provided")
#         d_core[i] = dist[i, j]
#
#     return d_core
#
#
#
# cpdef np.ndarray[floatT] _d_core_from_dist(np.ndarray[floatT,ndim=2] dist, int M):
#     """
#     (provided for testing only)
#
#     Given a pairwise distance matrix, computes the "core distance", i.e.,
#     the distance of each point to its `M`-th nearest neighbour.
#     Note that `M==0` always yields all distances equal to zero.
#     The core distances are needed when computing the mutual reachability
#     distance in the HDBSCAN* algorithm.
#
#     See Campello R.J.G.B., Moulavi D., Sander J.,
#     Density-based clustering based on hierarchical density estimates,
#     *Lecture Notes in Computer Science* 7819, 2013, 160-172,
#     doi:10.1007/978-3-642-37456-2_14 -- but unlike to the definition therein,
#     we do not consider the distance to self as part of the core distance setting.
#
#     The input distance matrix for a given point cloud X may be computed,
#     e.g., via a call to
#     ``scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))``.
#
#
#     Parameters
#     ----------
#
#     dist : ndarray, shape (n_samples,n_samples)
#         A pairwise n*n distance matrix.
#     M : int
#         A smoothing factor >= 1.
#
#
#     Returns
#     -------
#
#     d_core : ndarray, shape (n_samples,)
#         d_core[i] gives the distance between the i-th point and its M-th nearest
#         neighbour.
#     """
#     cdef Py_ssize_t n = dist.shape[0], i, j
#     cdef floatT v
#     cdef np.ndarray[floatT] d_core = np.zeros(n,
#         dtype=np.float32 if floatT is float else np.float64)
#     cdef floatT[::1] row
#
#     if M < 0: raise ValueError("M < 0")
#     if dist.shape[1] != n: raise ValueError("not a square matrix")
#     if M >= n: raise ValueError("M >= matrix size")
#
#     if M == 0: return d_core  # zeros
#
#     cdef vector[Py_ssize_t] buf = vector[Py_ssize_t](M+1)
#     for i in range(n):
#         row = dist[i,:]
#         j = Cargkmin(&row[0], row.shape[0], M, buf.data())
#         d_core[i] = dist[i, j]
#
#     return d_core
#
#
#
# cpdef np.ndarray[floatT,ndim=2] _mutual_reachability_distance(
#         np.ndarray[floatT,ndim=2] dist,
#         np.ndarray[floatT] d_core):
#     """
#     (provided for testing only)
#
#     Given a pairwise distance matrix, computes the mutual reachability
#     distance w.r.t. the given core distance vector,
#     ``new_dist[i,j] = max(dist[i,j], d_core[i], d_core[j])``.
#
#     Note that there may be many ties in the mutual reachability distances.
#
#     See Campello R.J.G.B., Moulavi D., Sander J.,
#     Density-based clustering based on hierarchical density estimates,
#     *Lecture Notes in Computer Science* 7819, 2013, 160-172,
#     doi:10.1007/978-3-642-37456-2_14.
#
#     The input distance matrix for a given point cloud X
#     may be computed, e.g., via a call to
#     ``scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))``.
#
#
#     Parameters
#     ----------
#
#     dist : ndarray, shape (n_samples,n_samples)
#         a pairwise n*n distance matrix
#
#     d_core : ndarray, shape (n_samples,)
#         the core distances
#
#
#     Returns
#     -------
#
#     R : ndarray, shape (n_samples,n_samples)
#         a new distance matrix giving the mutual reachability distance
#     """
#     cdef Py_ssize_t n = dist.shape[0], i, j
#     cdef floatT v
#     if dist.shape[1] != n: raise ValueError("not a square matrix")
#
#     cdef np.ndarray[floatT,ndim=2] R = np.array(dist,
#         dtype=np.float32 if floatT is float else np.float64)
#     for i in range(0, n-1):
#         for j in range(i+1, n):
#             v = dist[i, j]
#             if v < d_core[i]: v = d_core[i]
#             if v < d_core[j]: v = d_core[j]
#             R[i, j] = R[j, i] = v
#
#     return R
#
#
# cpdef tuple nn_list_to_matrix(
#         list nns,
#         Py_ssize_t k_max
#     ):
#     """
#     deadwood.internal.nn_list_to_matrix(nns, k_max)
#
#     Converts a list of (<=`k_max`)-nearest neighbours to a matrix of `k_max` NNs
#
#     [DEPRECATED]
#
#
#     Parameters
#     ----------
#
#     nns : list
#         Each ``nns[i]`` should be a pair of ``c_contiguous`` `ndarray`\ s.
#         An edge ``{i, nns[i][0][j]}`` has weight ``nns[i][1][j]``.
#         Each ``nns[i][0]`` is of type `int32` and ``nns[i][1]``
#         is of type `float32` (for compatibility with `nmslib`).
#     k_max : int
#         If `k_max` is greater than 0, `O(n*k_max)` space will be reserved
#         for auxiliary data.
#
#
#     Returns
#     -------
#
#     tuple like ``(nn_dist, nn_ind)`` :
#         See `deadwood.internal.mst_from_nn`.
#         Unused elements (last items in each row)
#         will be filled with ``INFINITY`` and `-1`, respectively.
#
#
#     See also
#     --------
#
#     deadwood.internal.mst_from_nn :
#         Constructs a minimum spanning tree from a near-neighbour matrix
#
#     """
#     cdef Py_ssize_t n = len(nns)
#     cdef np.ndarray[int]   nn_i
#     cdef np.ndarray[float] nn_d
#
#     cdef np.ndarray[Py_ssize_t,ndim=2] ret_nn_ind  = np.empty((n, k_max), dtype=np.intp)
#     cdef np.ndarray[float,ndim=2]  ret_nn_dist = np.empty((n, k_max), dtype=np.float32)
#
#     cdef Py_ssize_t i, j, k, l
#     cdef Py_ssize_t i1, i2
#     cdef float d
#
#     for i in range(n):
#         nn_i = nns[i][0]
#         nn_d = nns[i][1].astype(np.float32, copy=False)
#         k = nn_i.shape[0]
#         if nn_d.shape[0] != k:
#             raise ValueError("nns has arrays of different lengths as elements")
#
#         l = 0
#         for j in range(k):
#             i2 = nn_i[j]
#             d = nn_d[j]
#             if i2 >= 0 and i != i2:
#                 if l >= k_max: raise ValueError("`k_max` is too small")
#                 ret_nn_ind[i, l]  = i2
#                 ret_nn_dist[i, l] = d
#                 if l > 0 and ret_nn_dist[i, l] < ret_nn_dist[i, l-1]:
#                     raise ValueError("nearest neighbours not sorted")
#                 l += 1
#
#         while l < k_max:
#             ret_nn_ind[i, l]  = -1
#             ret_nn_dist[i, l] = INFINITY
#             l += 1
#
#     return ret_nn_dist, ret_nn_ind
