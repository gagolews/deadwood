# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Core functions and classes
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2025-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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


cimport libc.math
from libcpp cimport bool

import numpy as np
cimport numpy as np
np.import_array()

#import warnings


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


cdef extern from "c_kneedle.h":
    Py_ssize_t Ckneedle_increasing[floatT](
        const floatT* x, Py_ssize_t n, bool convex, floatT dt
    ) except+


cpdef Py_ssize_t kneedle_increasing(
        floatT[::1] x,
        bool convex=True,
        floatT dt=0.01
    ):
    """
    deadwood.kneedle_increasing(x, convex=True, dt=0.01)

    Finds the most significant knee/elbow using the Kneedle algorithm
    with exponential smoothing.


    Parameters
    ----------

    x : ndarray
        data vector (increasing)

    convex : bool
        whether the data in `x` are convex-ish (elbow detection)
        or not (knee lookup)

    dt : float
        controls the smoothing parameter :math:`\\alpha = 1-\\exp(-dt)`
        of the exponential moving average,
        :math:`y_i = \\alpha x_i + (1-\\alpha) y_{i-1}`,
        :math:`y_1 = x_1`.


    Returns
    -------

    index : integer
        the index of the knee/elbow point; 0 if not found


    References
    ----------

    .. [1]
        V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan, *Finding a "Kneedle"
        in a haystack: Detecting knee points in system behavior*,
        In: *31st Intl. Conf. Distributed Computing Systems Workshops*,
        2011, 166-171, https://doi.org/10.1109/ICDCSW.2011.20

    """
    cdef Py_ssize_t n = x.shape[0]
    return Ckneedle_increasing(&x[0], n, convex, dt)






cdef extern from "c_argfuns.h":
    # void Cargsort[T](Py_ssize_t* ret, T* x, Py_ssize_t n, bint stable) except+
    Py_ssize_t Cargkmin[T](T* x, Py_ssize_t n, Py_ssize_t k, Py_ssize_t* buf) except+
    # Py_ssize_t Cargkmax[T](T* x, Py_ssize_t n, Py_ssize_t k, Py_ssize_t* buf)  TODO




cpdef np.ndarray[Py_ssize_t] cummin(T[:] x):
    """
    deadwood.cummin(x)

    Cumulative minimum


    Parameters
    ----------

    x : ndarray
        A vector with *n* elements of type ``int``, ``float``,
        or ``double``.


    Returns
    -------

    ndarray
        A vector of length *n* whose ``i``-th element
        is equal to ``min(x[:i])``.


    See Also
    --------

    deadwood.cummax : Cumulative maximum


    Examples
    --------

    >>> deadwood.cummin(np.r_[3, 4, 2, 1, 5, 6])
    array([3, 3, 2, 1, 1, 1])

    """
    cdef Py_ssize_t n = x.shape[0], i
    cdef np.ndarray[T] ret = np.empty_like(x)
    ret[0] = x[0]
    for i in range(1, n):
        if x[i] <= ret[i-1]:
            ret[i] = x[i]
        else:
            ret[i] = ret[i-1]

    return ret


cpdef np.ndarray[Py_ssize_t] cummax(T[:] x):
    """
    deadwood.cummax(x)

    Cumulative maximum


    Parameters
    ----------

    x : ndarray
        A vector with *n* elements of type ``int``, ``float``,
        or ``double``.


    Returns
    -------

    ndarray
        A vector of length *n* whose ``i``-th element
        is equal to ``max(x[:i])``.


    See Also
    --------

    deadwood.cummin : Cumulative minimum


    Examples
    --------

    >>> deadwood.cummax(np.r_[3, 2, 1, 4, 6, 5])
    array([3, 3, 3, 4, 6, 6])

    """
    cdef Py_ssize_t n = x.shape[0], i
    cdef np.ndarray[T] ret = np.empty_like(x)
    ret[0] = x[0]
    for i in range(1, n):
        if x[i] >= ret[i-1]:
            ret[i] = x[i]
        else:
            ret[i] = ret[i-1]

    return ret



cpdef bool is_increasing(T[:] x):
    """
    deadwood.is_increasing(x)

    Verifies if an array is sorted nondecreasingly


    Parameters
    ----------

    x : ndarray
        A vector with *n* elements of type ``int``, ``float``,
        or ``double``.


    Returns
    -------

    bool


    Examples
    --------

    >>> deadwood.is_increasing(np.r_[1, 2, 2, 3, 10])
    True

    >>> deadwood.is_increasing(np.r_[1, 2, 2, 3, 1])
    False

    """
    cdef size_t i
    cdef size_t n = x.shape[0]
    for i in range(1, n):
        if x[i-1] > x[i]:
            return False

    return True


cpdef Py_ssize_t argkmin(T[::1] x, int k):
    """
    deadwood.argkmin(x, k)

    Finds the position of an order statistic.
    It holds ``argkmin(x, 0) == argmin(x)``, or, more generally,
    ``argkmin(x, k) == np.argsort(x)[k]``.


    Run time is :math:`O(nk)` and working memory is :math:`O(k)`.
    An insertion sort-like scheme is used to locate the order statistic.
    In practice, the function is very fast for small `k` and randomly
    ordered or almost sorted (increasingly) data.


    Parameters
    ----------

    x : ndarray
        A c_contiguous vector with *n* elements of type ``int``, ``float``,
        or ``double``.

    k : int
        An integer between 0 and *n* - 1, preferably small.


    Returns
    -------

    int
        The index where the (`k`-1)-th smallest value in `x` is located.


    Examples
    --------

    >>> x = np.r_[2, 3, 6, 5, 1, 4]
    >>> deadwood.argkmin(x, 0) # index of the smallest value
    4
    >>> deadwood.argkmin(x, 1) # index of the 2nd smallest value
    0
    """
    cdef Py_ssize_t n = x.shape[0]
    # TODO: allow returning the k indexes
    # TODO: argkmax
    return Cargkmin(&x[0], n, k, <Py_ssize_t*>0)


# cpdef np.ndarray[Py_ssize_t] _argsort(np.ndarray[T] x, bint stable=True):
#     """
#     (provided for testing only)
#
#     Finds the/an ordering permutation
#
#
#     Parameters
#     ----------
#
#     x : ndarray
#         A vector with *n* elements of type ``int``, ``float``,
#         or ``double``.
#
#     stable : bool
#         Should a stable (a bit slower) sorting algorithm be used?
#
#
#     Returns
#     -------
#
#     ndarray
#         An ordering permutation of `x`, an integer vector `o` of length *n*
#         with elements between 0 and *n* - 1
#         such that ``x[o[0]] <= x[o[1]] <= ... <= x[o[n-1]]``.
#
#
#     Notes
#     -----
#
#     Finds the/an ordering permutation of a ``c_contiguous`` array `x`
#     The ordering permutation is uniquely defined provided that
#     `stable` is ``True``. Otherwise *an* ordering permutation
#     will be generated.
#
#     """
#     x = np.asarray(x, dtype=x.dtype, order="C")  # ensure c_contiguity
#     cdef Py_ssize_t n = x.shape[0]
#     cdef np.ndarray[Py_ssize_t] ret = np.empty(n, dtype=np.intp)
#     Cargsort(&ret[0], &x[0], n, stable)
#     return ret



cdef extern from "c_deadwood.h":
    void Csort_groups[T](
        const T* x, Py_ssize_t n,
        const Py_ssize_t* c, Py_ssize_t k,
        T* y, Py_ssize_t* ind
    )

    void Cindex_unskip(
        Py_ssize_t* ind,
        Py_ssize_t m,
        const bool* skip,
        Py_ssize_t n
    ) except+

    void Cindex_skip(
        Py_ssize_t* ind,
        Py_ssize_t m,
        const bool* skip,
        Py_ssize_t n
    ) except+

    Py_ssize_t Csum_bool(const bool* x, Py_ssize_t n)

    void Cgraph_vertex_degrees(
        const Py_ssize_t* graph_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* deg
    ) except+

    void Cgraph_vertex_incidences(
        const Py_ssize_t* graph_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* cumdeg,
        Py_ssize_t* inc
    ) except+

    Py_ssize_t Cmst_cluster_sizes(
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* labels,
        Py_ssize_t max_k,
        Py_ssize_t* s,
        const Py_ssize_t* mst_cumdeg,
        const Py_ssize_t* mst_inc,
        const bool* mst_skip
    ) except+

    void Cmst_label_imputer(
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* labels,
        const Py_ssize_t* mst_cumdeg,
        const Py_ssize_t* mst_inc,
        const bool* mst_skip
    ) except+

    void Cdeadwood[floatT](
        const floatT* mst_d,
        const Py_ssize_t* mst_i,
        const Py_ssize_t* mst_cut,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t k,
        floatT max_contamination,
        floatT ema_dt,
        Py_ssize_t max_debris_size,
        floatT* contamination,
        Py_ssize_t* c,
        const Py_ssize_t* mst_cumdeg,
        const Py_ssize_t* mst_inc
    ) except+

    # void Cmst_trim_branches[floatT](  # [DEPRECATED]
    #     const floatT* mst_d, floatT min_d, Py_ssize_t max_size,
    #     const Py_ssize_t* mst_i, Py_ssize_t m, Py_ssize_t n, Py_ssize_t* c,
    #     const Py_ssize_t* cumdeg, const Py_ssize_t* inc, const bool* mst_skip
    # ) except+


cpdef tuple sort_groups(
        T[::1] x,
        Py_ssize_t[::1] c,
        Py_ssize_t k
    ):
    """
    deadwood.sort_groups(x, ind, k)

    Reorders the elements in `x` w.r.t. a factor `c`


    Parameters
    ----------

    x : ndarray of shape (n,)
        data items to sort w.r.t. `c`

    c : ndarray of shape (n,)
        the cluster IDs in {-1,0,...,k-1} corresponding to the items in `x`

    k : int


    Returns
    -------

    y : ndarray of shape (n,)
        a reordered version of `x`

    ind : ndarray of shape (k+1,)
        `y[ind[j]]`, ..., `y[ind[j+1]-1]` give all `x[i]`s,
        in their original relative order, for which `c[i]==j`.

        All the elements corresponding to `c[i] < 0` are put at the start
        of `y`.  `c[i] ≥ k` is disallowed.
    """
    cdef Py_ssize_t n = x.shape[0]
    if c.shape[0] != n:
        raise ValueError("x and c shape mismatch")
    if k <= 0:
        raise ValueError("k must be nonnegative")

    cdef np.ndarray[T] y = np.empty_like(x)
    cdef np.ndarray[Py_ssize_t] ind = np.empty(k+1, dtype=np.intp)

    Csort_groups(&x[0], n, &c[0], k, &y[0], &ind[0])

    return y, ind



cpdef np.ndarray[Py_ssize_t] index_unskip(
        Py_ssize_t[::1] ind,
        bool[::1] skip
    ):
    """
    deadwood.index_unskip(ind, skip)

    If ``skip=[False, True, False, False, True, False, False]``,
    then the indexes in `ind` are mapped in such a way that:
    0 → 0,
    1 → 2,
    2 → 3,
    3 → 5,
    4 → 6.

    This function might be useful if we apply a method on ``X[~skip,:]``
    (a subset of rows in ``X``), obtain a vector of indexes `ind` relative
    to the indexes of rows in ``X[~skip,:]`` as result, and wish to translate
    `ind` back to the original row space of ``X[:,:]``.

    For instance, ``index_unskip([0, 2, 1], [True, False, True, False, False])``
    yields ``[1, 4, 3]``.


    Parameters
    ----------

    ind : c_contiguous array of `m` indexes
        `m` indexes to translate (between `0` and `k-1`)

    skip : Boolean array of length `n` with `k` elements equal to `False`
        `skip[i]` indicates whether an index `i` was skipped or not


    Returns
    -------

    out : ndarray
        `m` translated indexes (between `0` and `n-1`)
    """
    cdef Py_ssize_t n = skip.shape[0]
    cdef Py_ssize_t m = ind.shape[0]

    cdef np.ndarray[Py_ssize_t] ret = np.array(ind, dtype=np.intp)

    Cindex_unskip(&ret[0], m, &skip[0], n)

    return ret


cpdef np.ndarray[Py_ssize_t] index_skip(
        Py_ssize_t[::1] ind,
        bool[::1] skip
    ):
    """
    deadwood.index_skip(ind, skip)

    If ``skip=[False, True, False, False, True, False, False]``,
    then the indexes in `ind` are mapped in such a way that:
    0 ← 0,
    1 ← 2,
    2 ← 3,
    3 ← 5,
    4 ← 6,
    i.e., the indexes for which `skip` is False are mapped
    to consecutive integers.  All other indexes are assigned the value -1.

    For instance, ``index_skip([1, 4, 3], [True, False, True, False, False])``
    yields ``[0, 2, 1]``.


    Parameters
    ----------

    ind : c_contiguous array of `m` indexes
        `m` indexes to translate (between `0` and `n-1`)

    skip : Boolean array of length `n` with `k` elements equal to `False`
        `skip[i]` indicates whether an index `i` was skipped or not


    Returns
    -------

    out : ndarray
        `m` translated indexes (between `-1` and `k-1`)
    """
    cdef Py_ssize_t n = skip.shape[0]
    cdef Py_ssize_t m = ind.shape[0]

    cdef np.ndarray[Py_ssize_t] ret = np.array(ind, dtype=np.intp)

    Cindex_skip(&ret[0], m, &skip[0], n)

    return ret


cpdef np.ndarray[Py_ssize_t] graph_vertex_degrees(
        Py_ssize_t[:,::1] graph_i,
        Py_ssize_t n
    ):
    """
    deadwood.graph_vertex_degrees(graph_i, n)

    Determines the degrees of all nodes in an undirected simple graph over
    a vertex set `{0,...,n-1}` specified via an adjacency list.


    Parameters
    ----------

    graph_i : ndarray, shape (m,2)
        a two-column matrix with elements between `0` and `n-1` such that
        `{graph_i[i,0], graph_i[i,1]}` represents the `i`-th undirected edge

    n : int
        the number of vertices in the graph, typically ``max(graph_i)+1``


    Returns
    -------

    deg : ndarray, shape(n,)
        an integer array of length `n`; ``deg[i]`` denotes the degree of
        the `i`-th vertex;  for instance, ``deg[i]==1`` designates a leaf
    """
    if graph_i.shape[1] != 2: raise ValueError("graph_i must have two columns")
    cdef Py_ssize_t m = graph_i.shape[0]
    cdef np.ndarray[Py_ssize_t] deg = np.empty(n, dtype=np.intp)

    Cgraph_vertex_degrees(&graph_i[0,0], m, n, &deg[0])

    return deg



cpdef tuple graph_vertex_incidences(Py_ssize_t[:,::1] graph_i, Py_ssize_t n):
    """
    deadwood.graph_vertex_incidences(graph_i, n)

    Computes the incidence lists of all vertices in an undirected simple graph
    over a vertex set {0,...,n-1} specified via an adjacency list.


    Parameters
    ----------

    graph_i : ndarray, shape (m,2)
        a two-column matrix with elements between `0` and `n-1` such that
        `{graph_i[i,0], graph_i[i,1]}` represents the `i`-th undirected edge

    n : int
        the number of vertices in the graph, typically ``max(graph_i)+1``


    Returns
    -------

    cumdeg : ndarray, shape (n+1,)
        an integer array of length `n+1`, where ``cumdeg[i+1]`` the sum of
        the degrees of the first `i` vertices

    inc : ndarray, shape (2*m,)
        an integer array of length `2*m`, where ``inc[cumdeg[i]]``, ...,
        ``inc[cumdeg[i+1]-1]`` give the edges incident on the `i`-th vertex
    """
    if graph_i.shape[1] != 2: raise ValueError("graph_i must have two columns")
    cdef Py_ssize_t m = graph_i.shape[0]
    cdef np.ndarray[Py_ssize_t] cumdeg = np.empty(n+1, dtype=np.intp)
    cdef np.ndarray[Py_ssize_t] inc = np.empty(2*m, dtype=np.intp)

    Cgraph_vertex_incidences(&graph_i[0,0], m, n, &cumdeg[0], &inc[0])

    return cumdeg, inc


cpdef np.ndarray[Py_ssize_t] mst_label_imputer(
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t[::1] labels,
        mst_skip=None,
        mst_cumdeg=None,
        mst_inc=None
    ):
    """
    deadwood.mst_label_imputer(mst_i, labels, mst_skip=None)

    Imputes all missing labels down below a given tree's branches.
    All nodes in branches with class ID of `-1` will be assigned
    their parent node's class.


    Parameters
    ----------

    mst_i : c_contiguous array of shape (n-1, 2)
        `n-1` undirected edges of the spanning tree/forest

    labels : c_contiguous array of shape (n,)
        `labels[i]` gives the cluster ID (in `{-1, 0, 1, ..., k-1}` for some `k`)
        of the `i`-th object;  class `-1` represents the missing values
        to be imputed

    mst_skip : c_contiguous array of length n-1 or None
        `mst_skip[i] == True` marks the `i`-th edge as non-existent (ignorable)

    mst_cumdeg, mst_inc : c_contiguous arrays or None
        see ``deadwood.graph_vertex_incidences``


    Returns
    -------

    ret : ndarray, shape (n,)
        an integer vector with `ret[i]` denoting the cluster
        ID (in {0, ..., k-1}) of the `i`-th vertex
    """
    cdef Py_ssize_t n = labels.shape[0]
    cdef Py_ssize_t m = mst_i.shape[0]

    if mst_i.shape[1] != 2: raise ValueError("mst_i must have two columns")
    if m != n-1: raise ValueError("mst_i must have n-1 rows")

    cdef bool[::1] mst_skip_obj
    cdef bool* mst_skip_ptr = NULL
    if mst_skip is None:
        pass
    else:
        mst_skip_obj = mst_skip  # may raise an Exception
        if mst_skip_obj.shape[0] != m:
            raise ValueError("mst_skip should be either of size n-1 or None")
        mst_skip_ptr = &mst_skip_obj[0]


    if (mst_cumdeg is None) != (mst_inc is None):
        raise ValueError("mst_cumdeg and mst_inc should both be None or not")

    cdef Py_ssize_t[::1] mst_cumdeg_obj
    cdef Py_ssize_t* mst_cumdeg_ptr = NULL
    if mst_cumdeg is None:
        pass
    else:
        mst_cumdeg_obj = mst_cumdeg  # may raise an Exception
        if mst_cumdeg_obj.shape[0] != n+1:
            raise ValueError("mst_cumdeg should either of size n+1 or None")
        mst_cumdeg_ptr = &mst_cumdeg_obj[0]

    cdef Py_ssize_t[::1] mst_inc_obj
    cdef Py_ssize_t* mst_inc_ptr = NULL
    if mst_inc is None:
        pass
    else:
        mst_inc_obj = mst_inc  # may raise an Exception
        if mst_inc_obj.shape[0] != 2*m:
            raise ValueError("mst_inc should either of size 2*(n-1) or None")
        mst_inc_ptr = &mst_inc_obj[0]


    cdef np.ndarray[Py_ssize_t] ret = np.array(labels, dtype=np.intp)

    Cmst_label_imputer(
        &mst_i[0,0], m, n, &ret[0],
        mst_cumdeg_ptr, mst_inc_ptr, mst_skip_ptr
    )

    return ret


cpdef tuple mst_cluster_sizes(
        Py_ssize_t[:,::1] mst_i,
        mst_skip=None,
        mst_cumdeg=None,
        mst_inc=None
    ):
    """
    deadwood.mst_cluster_sizes(mst_i, mst_skip=None, mst_cumdeg=None, mst_inc=None)

    Labels connected components in a spanning forest and returns their sizes.


    Parameters
    ----------

    mst_i : c_contiguous array of shape (n-1, 2)
        `n-1` undirected edges of the spanning tree

    mst_skip : c_contiguous array of length n-1 or None
        `mst_skip[i] == True` marks the `i`-th edge as non-existent (ignorable)

    mst_cumdeg, mst_inc : c_contiguous arrays or None
        see ``deadwood.graph_vertex_incidences``


    Returns
    -------

    labels : ndarray, shape (n,)
        an integer vector with `labels[i]` denoting the cluster
        ID (in {0, ..., k-1}) of the `i`-th vertex

    sizes : ndarray, shape (k,)
        an integer vector with `sizes[i]` denoting the size of the `i`-th cluster
    """
    cdef Py_ssize_t m = mst_i.shape[0]
    cdef Py_ssize_t n = m+1

    if mst_i.shape[1] != 2: raise ValueError("mst_i must have two columns")

    cdef Py_ssize_t k = 1  # the number of clusters  (mst_skip affects it)

    cdef bool[::1] mst_skip_obj
    cdef bool* mst_skip_ptr = NULL
    if mst_skip is None:
        pass
    else:
        mst_skip_obj = mst_skip  # may raise an Exception
        if mst_skip_obj.shape[0] != m:
            raise ValueError("mst_skip should be either of size n-1 or None")
        mst_skip_ptr = &mst_skip_obj[0]

        k += Csum_bool(mst_skip_ptr, m)


    if (mst_cumdeg is None) != (mst_inc is None):
        raise ValueError("mst_cumdeg and mst_inc should both be None or not")

    cdef Py_ssize_t[::1] mst_cumdeg_obj
    cdef Py_ssize_t* mst_cumdeg_ptr = NULL
    if mst_cumdeg is None:
        pass
    else:
        mst_cumdeg_obj = mst_cumdeg  # may raise an Exception
        if mst_cumdeg_obj.shape[0] != n+1:
            raise ValueError("mst_cumdeg should either of size n+1 or None")
        mst_cumdeg_ptr = &mst_cumdeg_obj[0]

    cdef Py_ssize_t[::1] mst_inc_obj
    cdef Py_ssize_t* mst_inc_ptr = NULL
    if mst_inc is None:
        pass
    else:
        mst_inc_obj = mst_inc  # may raise an Exception
        if mst_inc_obj.shape[0] != 2*m:
            raise ValueError("mst_inc should either of size 2*(n-1) or None")
        mst_inc_ptr = &mst_inc_obj[0]


    cdef np.ndarray[Py_ssize_t] labels = np.empty(n, dtype=np.intp)
    cdef np.ndarray[Py_ssize_t] sizes = np.zeros(k, dtype=np.intp)

    cdef Py_ssize_t _k = Cmst_cluster_sizes(
        &mst_i[0,0], m, n, &labels[0], k, &sizes[0],
        mst_cumdeg_ptr, mst_inc_ptr, mst_skip_ptr
    )
    assert _k == k

    return labels, sizes




cpdef tuple deadwood_from_mst(
        floatT[::1] mst_d,
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t[::1] mst_cut,
        Py_ssize_t[::1] mst_cumdeg,
        Py_ssize_t[::1] mst_inc,
        floatT max_contamination=0.5,
        floatT ema_dt=0.01,
        Py_ssize_t max_debris_size=50
    ):
    """
    deadwood.deadwood_from_mst(mst_d, mst_i, mst_cut, mst_cumdeg, mst_inc, max_contamination=0.5, ema_dt=0.01, max_debris_size=50)

    The Deadwood outlier detection algorithm based on Euclidean spanning trees.


    Parameters
    ----------

    mst_d, mst_i : ndarray
        a spanning tree with `m=n-1` edges defined by a pair
        `(mst_i, mst_d)`; see ``quitefastmst.mst_euclid``

    mst_cut : ndarray, length k-1 (possibly 0)
        indexes cut edges defining a spanning forest with `k` connected components

    mst_cumdeg : ndarray, length n+1
        see `deadwood.graph_vertex_incidences`

    mst_inc : ndarray, length 2*m
        see `deadwood.graph_vertex_incidences`

    max_contamination : float
        maximal contamination level used in elbow detection in the edge weight
        quantile function; negative values will be used as actual (target)
        contamination levels

    ema_dt : float
        controls the exponential moving average smoothing parameter
        :math:`\\alpha = 1-\\exp(-dt)` (in elbow detection)

    max_debris_size : int
        connected components of size ≤ `max_debris_size` will be treated as outliers


    Returns
    -------

    is_outlier : ndarray, shape (n,)
        `is_outlier[i]=1` if the `i`-th point is an outlier and 0 otherwise

    contamination : ndarray, shape (k,)
        detected contamination levels in each cluster


    References
    ----------

    .. [1]
        M. Gagolewski, *Deadwood*, in preparation, 2026, TODO
    """
    cdef Py_ssize_t m = mst_i.shape[0]
    cdef Py_ssize_t n = m+1
    cdef Py_ssize_t k = mst_cut.shape[0]+1

    if not m == mst_d.shape[0] or m != n-1 or mst_i.shape[1] != 2:
        raise ValueError("ill-defined spanning tree")

    if mst_cumdeg.shape[0] != n+1:
        raise ValueError("mst_cumdeg should be of length n+1")

    if mst_inc.shape[0] != 2*m:
        raise ValueError("mst_inc should be of length 2*m")

    cdef np.ndarray[Py_ssize_t] is_outlier_ = np.empty(n, dtype=np.intp)
    cdef np.ndarray[floatT] contamination_ = np.empty(k,
        dtype=np.float32 if floatT is float else np.float64)

    Cdeadwood(
        &mst_d[0], &mst_i[0,0], &mst_cut[0], m, n, k,
        max_contamination, ema_dt, max_debris_size,
        &contamination_[0],
        &is_outlier_[0],
        &mst_cumdeg[0], &mst_inc[0]
    )

    return is_outlier_, contamination_


# cpdef np.ndarray[Py_ssize_t] trim_branches(
#         floatT[::1] mst_d,
#         Py_ssize_t[:,::1] mst_i,
#         floatT min_d,
#         Py_ssize_t max_size,
#         bool[::1] mst_skip
#         TODO: mst_cumdeg
#         TODO: mst_inc
#     ):
#     """
#     deadwood.trim_branches(mst_d, mst_i, min_d, max_size)
#
#     [DEPRECATED]
#
#     Mark points in certain ("long and small") tree branches.
#
#
#     Parameters
#     ----------
#
#     mst_d, mst_i : ndarray
#         Minimal spanning tree defined by a pair (mst_i, mst_d),
#         with mst_i of shape (n-1,2) giving the edges and mst_d providing the
#         corresponding edge weights
#     min_d
#         Minimal weight of an edge to consider for trimming
#     max_size
#         Maximal size of a tree branch to consider for trimming
#     mst_skip : c_contiguous array, length m or 0
#         mst_skip[i] == True marks the i-th edge as non-existent (ignorable)
#
#     Returns
#     -------
#
#     c : ndarray, shape (n,)
#         A new integer vector c with c[i]==-1 denoting a trimmed-out point
#         and c[i]>=0 indicating a left-out one
#     """
#     cdef Py_ssize_t m = mst_i.shape[0]
#     cdef Py_ssize_t n = m+1
#
#     if mst_i.shape[1] != 2: raise ValueError("mst_i must have two columns")
#
#     cdef bool* mst_skip_ptr = NULL
#     if mst_skip.shape[0] == 0:
#         pass
#     elif mst_skip.shape[0] == m:
#         mst_skip_ptr = &mst_skip[0]
#     else:
#         raise ValueError("mst_skip should be either of size 0 or m")
#
#     cdef np.ndarray[Py_ssize_t] c = np.empty(n, dtype=np.intp)
#
#     Cmst_trim_branches(&mst_d[0], min_d, max_size, &mst_i[0,0], m, n, &c[0], NULL, NULL, mst_skip_ptr)
#
#     return c

