/* Jarník (Prim)'s MST algorithm for complete undirected graphs
 * (the "old"/generic interface; see quitefastmst for a faster
 * implementation).
 *
 *  Copyleft (C) 2018-2026, Marek Gagolewski <https://www.gagolewski.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If this is not the case, refer to <https://www.gnu.org/licenses/>.
 */


#ifndef __c_oldmst_h
#define __c_oldmst_h

#include "c_common.h"
#include <vector>
#include <cmath>
#include <algorithm>


/*! Abstract base class for all distances */
template<class T>
struct CDistance {
    virtual ~CDistance() {}

    /*!
     * @param i point index, 0 <= i < n
     * @param M indexes
     * @param k length of M
     * @return distances from the i-th point to M[0], .., M[k-1],
     *         with ret[M[j]]=d(i, M[j]);
     *         the user does not own ret;
     *         the function is not thread-safe
     */
    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* M, Py_ssize_t k) = 0;
};



/*! A class to "compute" the distances from the i-th point
 *  to all n points based on a pre-computed n*n symmetric,
 *  complete pairwise distance c_contiguous matrix.
 */
template<class T>
struct CDistancePrecomputedMatrix : public CDistance<T> {
    const T* dist;
    Py_ssize_t n;

    /*!
     * @param dist n*n c_contiguous array, dist[i,j] is the distance between
     *    the i-th and the j-th point, the matrix is symmetric
     * @param n number of points
     */
    CDistancePrecomputedMatrix(const T* dist, Py_ssize_t n) {
        this->n = n;
        this->dist = dist;
    }

    CDistancePrecomputedMatrix()
        : CDistancePrecomputedMatrix(NULL, 0) { }

    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* /*M*/, Py_ssize_t /*k*/) {
        return &this->dist[i*n]; // the i-th row of dist
    }
};


/*! A class to "compute" the distances from the i-th point
 *  to all n points based on a pre-computed
 *  c_contiguous distance vector.
 */
template<class T>
struct CDistancePrecomputedVector : public CDistance<T> {
    const T* dist;
    Py_ssize_t n;
    std::vector<T> buf;

    /*!
     * @param dist n*(n-1)/2 c_contiguous vector, dist[ i*n - i*(i+1)/2 + w-i-1 ]
     *    where w is the distance between the i-th and the w-th point
     * @param n number of points
     */
    CDistancePrecomputedVector(const T* dist, Py_ssize_t n)
            : buf(n)
    {
        this->n = n;
        this->dist = dist;
    }

    CDistancePrecomputedVector()
        : CDistancePrecomputedVector(NULL, 0) { }

    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* M, Py_ssize_t k) {
        T* __buf = buf.data();
        for (Py_ssize_t j=0; j<k; ++j) {
            Py_ssize_t w = M[j];
            if (i == w)
                __buf[w] = 0.0;
            else if (i < w)
                __buf[w] = dist[ i*n - i*(i+1)/2 + w-i-1 ];
            else
                __buf[w] = dist[ w*n - w*(w+1)/2 + i-w-1 ];
        }
        return __buf;
    }
};



/*! A class to compute the Euclidean distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceEuclidean : public CDistance<T>  {
    const T* X;
    Py_ssize_t n;
    Py_ssize_t d;
    std::vector<T> buf;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceEuclidean(const T* X, Py_ssize_t n, Py_ssize_t d)
            : buf(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;
    }

    CDistanceEuclidean()
        : CDistanceEuclidean(NULL, 0, 0) { }

    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* M, Py_ssize_t k) {
        T* __buf = buf.data();
        const T* x = X+d*i;

#if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
#endif
        for (Py_ssize_t j=0; j<k; ++j) {
            Py_ssize_t w = M[j];
            const T* y = X+d*w;

            // or we could use the BLAS nrm2 / dot
            __buf[w] = 0.0;
            for (Py_ssize_t u=0; u<d; ++u) {
                __buf[w] += (x[u]-y[u])*(x[u]-y[u]);
            }
            __buf[w] = sqrt(__buf[w]);
        }
        return __buf;
    }
};





/*! A class to compute the squared Euclidean distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceEuclideanSquared : public CDistance<T>  {
    const T* X;
    Py_ssize_t n;
    Py_ssize_t d;
    std::vector<T> buf;
//    std::vector<T> x2;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceEuclideanSquared(const T* X, Py_ssize_t n, Py_ssize_t d)
            : buf(n) /*, x2(n, 0.0)*/
    {
        this->n = n;
        this->d = d;
        this->X = X;

//         T* _x2 = x2.data();
// #if OPENMP_IS_ENABLED
//         #pragma omp parallel for schedule(static)
// #endif
//         for (Py_ssize_t i=0; i<n; ++i) {
//             const T* x = X+d*i;
//             for (Py_ssize_t u=0; u<d; ++u) {
//                 _x2[i] += (*x)*(*x);
//                 ++x;
//             }
//         }
    }

    CDistanceEuclideanSquared()
        : CDistanceEuclideanSquared(NULL, 0, 0) { }

    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* M, Py_ssize_t k) {
        T* __buf = buf.data();
        const T* x = X+d*i;

#if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
#endif
        for (Py_ssize_t j=0; j<k; ++j) {
            Py_ssize_t w = M[j];
            const T* y = X+d*w;

            // or we could use the BLAS nrm2 / dot

            // this is not significantly faster (x-y)*(x-y)=x**2+y**2-2*x*y
            // __buf[w] = x2[i]+x2[w];
            // for (Py_ssize_t u=0; u<d; ++u) {
            //     __buf[w] -= 2.0*x[u]*y[u];
            // }

            __buf[w] = 0.0;
            for (Py_ssize_t u=0; u<d; ++u) {
                __buf[w] += (x[u]-y[u])*(x[u]-y[u]);
            }
        }
        return __buf;
    }
};




/*! A class to compute the CDistanceManhattan distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceManhattan : public CDistance<T>  {
    const T* X;
    Py_ssize_t n;
    Py_ssize_t d;
    std::vector<T> buf;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceManhattan(const T* X, Py_ssize_t n, Py_ssize_t d)
            : buf(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;
    }

    CDistanceManhattan()
        : CDistanceManhattan(NULL, 0, 0) { }

    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* M, Py_ssize_t k) {
        T* __buf = buf.data();
#if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
#endif
        for (Py_ssize_t j=0; j<k; ++j) {
            Py_ssize_t w = M[j];
            // DEADWOOD_ASSERT(w>=0 && w<n)
            __buf[w] = 0.0;

            for (Py_ssize_t u=0; u<d; ++u) {
                __buf[w] += fabs(X[d*i+u]-X[d*w+u]);
            }
        }
        return __buf;
    }
};



/*! A class to compute the cosine distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceCosine : public CDistance<T>  {
    const T* X;
    Py_ssize_t n;
    Py_ssize_t d;
    std::vector<T> buf;
    std::vector<T> norm;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceCosine(const T* X, Py_ssize_t n, Py_ssize_t d)
            : buf(n), norm(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;

        T* __norm = norm.data();
#if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
#endif
        for (Py_ssize_t i=0; i<n; ++i) {
            __norm[i] = 0.0;
            for (Py_ssize_t u=0; u<d; ++u) {
                __norm[i] += X[d*i+u]*X[d*i+u];
            }
            __norm[i] = sqrt(__norm[i]);
        }
    }

    CDistanceCosine()
        : CDistanceCosine(NULL, 0, 0) { }

    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* M, Py_ssize_t k) {
        T*  __buf = buf.data();
        T* __norm = norm.data();
#if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
#endif
        for (Py_ssize_t j=0; j<k; ++j) {
            Py_ssize_t w = M[j];
            // DEADWOOD_ASSERT(w>=0&&w<n)
            __buf[w] = 0.0;

            for (Py_ssize_t u=0; u<d; ++u) {
                __buf[w] += X[d*i+u]*X[d*w+u];
            }
            __buf[w] /= __norm[i];
            __buf[w] /= __norm[w];
            //__buf[w] = sqrt(2.0*(1.0-__buf[w]));
            __buf[w] = 1.0-__buf[w];
        }
        return __buf;
    }
};



/*! A class to compute the "mutual reachability" [1]_
 *  distances from the i-th point to all given k points based on the "core"
 *  distances and a CDistance class instance.
 *
 *  References:
 *  ==========
 *
 *  [1] Campello R.J.G.B., Moulavi D., Sander J.,
 *      Density-based clustering based on hierarchical density estimates,
 *      *Lecture Notes in Computer Science* 7819, 2013, 160-172,
 *      doi:10.1007/978-3-642-37456-2_14.
 *
 */
template<class T>
struct CDistanceMutualReachability : public CDistance<T>
{
    Py_ssize_t n;
    CDistance<T>* d_pairwise;
    std::vector<T> buf;
    std::vector<T> d_core;

    CDistanceMutualReachability(const T* _d_core, Py_ssize_t n, CDistance<T>* d_pairwise)
            : buf(n), d_core(_d_core, _d_core+n)
    {
        this->n = n;
        this->d_pairwise = d_pairwise;
    }

    CDistanceMutualReachability() : CDistanceMutualReachability(NULL, 0, NULL) { }

    virtual const T* operator()(Py_ssize_t i, const Py_ssize_t* M, Py_ssize_t k) {
        // pragma omp parallel for inside::
        const T* d = (*d_pairwise)(i, M, k);
        T*  __buf = buf.data();

        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=0; j<k; ++j)  { //
            Py_ssize_t w = M[j];
            if (w == i) __buf[w] = 0.0;
            else {
                // buf[w] = max{d[w],d_core[i],d_core[w]}
                // __buf[w] = d[w];
                // if (d_core[i] > __buf[w]) __buf[w] = d_core[i];
                // if (d_core[w] > __buf[w]) __buf[w] = d_core[w];

                T d_core_max;
                T d_core_min;

                if (d_core[i] >= d_core[w]) {
                    d_core_max = d_core[i];
                    d_core_min = d_core[w];
                }
                else {
                    d_core_max = d_core[w];
                    d_core_min = d_core[i];
                }

                if (d[w] > d_core_max) {
                    __buf[w] = d[w];
                }
                else {
                    __buf[w] = d_core_max;
                    //+d_core_min*0.00000011920928955078125; // make it unambiguous: // pulled-away from each other, but ordered w.r.t. the core distances (increasingly)
                }
            }
        }
        return __buf;
    }
};


/*! Represents an edge in a weighted graph.
 *  Features a comparer used to sort the edges w.r.t. increasing weights;
 *  more precisely, lexicographically w.r.t. (d, i1, d2).
 */
template <class T>
struct CMstTriple
{
    Py_ssize_t i1;  //!< first  vertex defining an edge
    Py_ssize_t i2;  //!< second vertex defining an edge
    T d;            //!< edge weight

    CMstTriple() {}

    CMstTriple(Py_ssize_t i1, Py_ssize_t i2, T d, bool order=true)
    {
        DEADWOOD_ASSERT(i1 != i2);
        DEADWOOD_ASSERT(i1 >= 0);
        DEADWOOD_ASSERT(i2 >= 0);
        this->d = d;
        if (!order || (i1 < i2)) {
            this->i1 = i1;
            this->i2 = i2;
        }
        else {
            this->i1 = i2;
            this->i2 = i1;
        }
    }

    bool operator<(const CMstTriple<T>& other) const
    {
        if (d == other.d) {
            if (i1 == other.i1)
                return i2 < other.i2;
            else
                return i1 < other.i1;
        }
        else
            return d < other.d;
    }
};


/*! A Jarník (Prim/Dijkstra)-like algorithm for determining
 *  a(*) minimum spanning tree (MST) of a complete undirected graph
 *  with weights given by, e.g., a symmetric n*n matrix.
 *
 *  However, the distances can be computed on the fly, so that O(n) memory is used.
 *
 *  (*) Note that there might be multiple minimum trees spanning a given graph.
 *
 *
 *  References:
 *  ----------
 *
 *  V. Jarník, O jistem problemu minimalnim (z dopisu panu O. Borůvkovi),
 *  Prace Moravske Prirodovedecke Spolecnosti 6, 1930, 57-63
 *
 *  C.F. Olson, Parallel algorithms for hierarchical clustering,
 *  Parallel Computing 21(8), 1995, 1313-1325
 *
 *  R. Prim, Shortest connection networks and some generalisations,
 *  The Bell System Technical Journal 36(6), 1957, 1389-1401
 *
 *
 * @param D a CDistance object such that a call to
 *        <T*>D(j, <Py_ssize_t*>M, Py_ssize_t k) returns a length-n array
 *        with the distances from the j-th point to k points whose indexes
 *        are given in array M
 * @param n number of points
 * @param mst_d [out] vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_i [out] vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 * @param verbose output diagnostic/progress messages?
 */
template <class T>
void Cmst_from_complete(CDistance<T>* D, Py_ssize_t n,
    T* mst_dist, Py_ssize_t* mst_ind, bool verbose=false)
{
    if (verbose) DEADWOOD_PRINT("[deadwood] Computing the MST... %3d%%", 0);

    // NOTE: all changes should also be mirrored in Cmst_euclid_brute()

    // ind_nn[j] is the vertex from the current tree closest to vertex j
    std::vector<Py_ssize_t> ind_nn(n);
    std::vector<T> dist_nn(n, INFINITY);  // dist_nn[j] = d(j, ind_nn[j])

    std::vector<Py_ssize_t> ind_left(n);
    for (Py_ssize_t i=0; i<n; ++i) ind_left[i] = i;

    std::vector< CMstTriple<T> > mst(n-1);

    Py_ssize_t ind_cur = 0;  // start with the first vertex (because we can start with any)
    for (Py_ssize_t i=1; i<n; ++i) {
        // ind_cur is the vertex most recently added to the tree
        // ind_left[i], ind_left[i+1], ..., ind_left[n-1] - vertices not yet in the tree

        // compute the distances (on the fly)
        // between ind_cur and all j=ind_left[i], ind_left[i+1], ..., ind_left[n-1]:
        // dist_from_ind_cur[j] == d(ind_cur, j)
        // pragma omp parallel for inside:
        const T* dist_from_ind_cur = (*D)(ind_cur, ind_left.data()+i, n-i);


        // update ind_nn and dist_nn as maybe now ind_cur (recently added to the tree)
        // is closer to some of the remaining vertices?
        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=i; j<n; ++j) {
            Py_ssize_t cur_ind_left = ind_left[j];
            T cur_dist = dist_from_ind_cur[cur_ind_left]; // d(ind_cur, cur_ind_left)
            if (cur_dist < dist_nn[cur_ind_left]) {
                ind_nn[cur_ind_left] = ind_cur;
                dist_nn[cur_ind_left] = cur_dist;
            }
        }

        // let best_ind_left and best_ind_left_pos = min and argmin of dist_nn,
        // for we want to include the vertex that is closest to the vertices
        // of the tree constructed so far
        Py_ssize_t best_ind_left_pos = i;
        Py_ssize_t best_ind_left = ind_left[i];
        for (Py_ssize_t j=i+1; j<n; ++j) {
            Py_ssize_t cur_ind_left = ind_left[j];
            if (dist_nn[cur_ind_left] < dist_nn[best_ind_left]) {
                best_ind_left = cur_ind_left;
                best_ind_left_pos = j;
            }
        }

        // connect best_ind_left with the tree: add a new edge {best_ind_left, ind_nn[best_ind_left]}
        mst[i-1] = CMstTriple<T>(best_ind_left, ind_nn[best_ind_left], dist_nn[best_ind_left], /*order=*/true);


        // don't visit best_ind_left again
#if 0
        std::swap(ind_left[best_ind_left_pos], ind_left[i]);
#else
        // keep ind_left sorted (a bit better locality of reference) (#62)
        for (Py_ssize_t j=best_ind_left_pos; j>i; --j)
            ind_left[j] = ind_left[j-1];
        ind_left[i] = best_ind_left;  // for readability only
#endif


        ind_cur = best_ind_left;  // start from best_ind_left next time

        if (verbose) DEADWOOD_PRINT("\b\b\b\b%3d%%", (int)((n-1+n-i-1)*(i+1)*100/n/(n-1)));

        #if DEADWOOD_R
        Rcpp::checkUserInterrupt();
        #elif DEADWOOD_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    // sort the resulting MST edges in increasing order w.r.t. d
    std::sort(mst.begin(), mst.end());

    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_dist[i]    = mst[i].d;
        mst_ind[2*i+0] = mst[i].i1; // i1 < i2
        mst_ind[2*i+1] = mst[i].i2;
    }

    if (verbose) DEADWOOD_PRINT("\b\b\b\bdone.\n");
}


/*! Determine the first k nearest neighbours of each point.
 *
 *  Exactly n*(n-1)/2 distance computations are performed.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 *  Worst-case time complexity: O(n*(n-1)/2*d*k)
 *
 *
 *  @param D a callable CDistance object such that a call to
 *         <T*>D(j, <Py_ssize_t*>M, Py_ssize_t l) returns an n-ary array
 *         with the distances from the j-th point to l points whose indexes
 *         are given in array M
 *  @param n number of points
 *  @param k number of nearest neighbours,
 *  @param dist [out]  a c_contiguous array, shape (n,k),
 *         dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 *  @param ind [out]   a c_contiguous array, shape (n,k),
 *         (undirected) edge definition, interpreted as {i, ind[i,j]}
 *  @param verbose output diagnostic/progress messages?
 */
template <class T>
void Cknn_from_complete(CDistance<T>* D, Py_ssize_t n, Py_ssize_t k,
    T* dist, Py_ssize_t* ind, bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (verbose) DEADWOOD_PRINT("[deadwood] Computing the K-nn graph... %3d%%", 0);


    for (Py_ssize_t i=0; i<n*k; ++i) {
        dist[i] = INFINITY;
        ind[i] = -1;
    }

    std::vector<Py_ssize_t> M(n);
    for (Py_ssize_t i=0; i<n; ++i) M[i] = i;

    for (Py_ssize_t i=0; i<n-1; ++i) {
        // pragma omp parallel for inside:
        const T* dij = (*D)(i, M.data()+i+1, n-i-1);
        // let dij[j] == d(x_i, x_j)


        // TODO: the 2nd `if` below can be OpenMP'd
        for (Py_ssize_t j=i+1; j<n; ++j) {
            if (dij[j] < dist[i*k+k-1]) {
                // j might be amongst k-NNs of i
                Py_ssize_t l = k-1;
                while (l > 0 && dij[j] < dist[i*k+l-1]) {
                    dist[i*k+l] = dist[i*k+l-1];
                    ind[i*k+l]  = ind[i*k+l-1];
                    l -= 1;
                }
                dist[i*k+l] = dij[j];
                ind[i*k+l]  = j;
            }
        }

        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t j=i+1; j<n; ++j) {
            if (dij[j] < dist[j*k+k-1]) {
                // i might be amongst k-NNs of j
                Py_ssize_t l = k-1;
                while (l > 0 && dij[j] < dist[j*k+l-1]) {
                    dist[j*k+l] = dist[j*k+l-1];
                    ind[j*k+l]  = ind[j*k+l-1];
                    l -= 1;
                }
                dist[j*k+l] = dij[j];
                ind[j*k+l]  = i;
            }
        }

        if (verbose) DEADWOOD_PRINT("\b\b\b\b%3d%%", (int)((n-1+n-i-1)*(i+1)*100/n/(n-1)));

        #if DEADWOOD_R
        Rcpp::checkUserInterrupt();
        #elif DEADWOOD_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    if (verbose) DEADWOOD_PRINT("\b\b\b\bdone.\n");
}




#if 0  /* Cmst_from_nn DEPRECATED */

#include "c_disjoint_sets.h"

/*! Computes a minimum spanning forest of a (<=k)-nearest neighbour graph
 *  (i.e., one that connects no more than the first k nearest neighbours
 *  (of each point) using Kruskal's algorithm, and orders
 *  its edges w.r.t. increasing weights.  [DEPRECATED]
 *
 *  Note that, in general, an MST of the (<=k)-nearest neighbour graph
 *  might not be equal to the MST of the complete Pairwise Distances Graph.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 * @param dist   a c_contiguous array, shape (n,k),
 *        dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 * @param ind    a c_contiguous array, shape (n,k),
 *        (undirected) edge definition, interpreted as {i, ind[i,j]};
 *        negative indexes as well as those such that ind[i,j]==i are ignored
 * @param d_core "core" distance (or NULL);
 *        if not NULL then the distance between two points will be
 *        d'(i, ind[i,j]) = max(d(i, ind[i,j]), d_core[i], d_core[ind[i,j]])
 * @param n number of nodes
 * @param k minimal degree of all the nodes
 * @param mst_dist [out] c_contiguous vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order;
 *        refer to the function's return value for the actual number
 *        of edges generated (if this is < n-1, the object is padded with INFINITY)
 * @param mst_ind [out] c_contiguous matrix of size (n-1)*2, defining the edges
 *        corresponding to mst_d, with mst_i[j,0] <= mst_i[j,1] for all j;
 *        refer to the function's return value for the actual number
 *        of edges generated (if this is < n-1, the object is padded with -1)
 * @param maybe_inexact [out] true indicates that k should be increased to
 *        guarantee that the resulting tree would be the same if a complete
 *        pairwise distance graph was given.
 * @param verbose output diagnostic/progress messages?
 *
 * @return number of edges in the minimal spanning forest
 */
template <class T>
Py_ssize_t Cmst_from_nn(
    const T* dist,
    const Py_ssize_t* ind,
    const T* d_core,
    Py_ssize_t n,
    Py_ssize_t k,
    T* mst_dist,
    Py_ssize_t* mst_ind,
    int* maybe_inexact,
    bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");
    Py_ssize_t nk = n*k;

    if (verbose) DEADWOOD_PRINT("[deadwood] Computing the MST... %3d%%", 0);

    std::vector< CMstTriple<T> > nns(nk);
    Py_ssize_t c = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        for (Py_ssize_t j = 0; j < k; ++j) {
            Py_ssize_t i2 = ind[k*i+j];
            if (i2 >= 0 && i2 != i) {
                double d = dist[k*i+j];
                if (d_core) {
                    // d(i, i2) = max(d(i,i2), d_core[i], d_core[i2])
                    if (d < d_core[i])  d = d_core[i];
                    if (d < d_core[i2]) d = d_core[i2];
                }
                nns[c++] = CMstTriple<T>(i, i2, d, true);
            }
        }
    }

    std::sort(nns.data(), nns.data()+c);


    Py_ssize_t triple_cur = 0;
    Py_ssize_t mst_edge_cur = 0;

    CDisjointSets ds(n);
    while (mst_edge_cur < n-1) {
        if (triple_cur == c) {
            // The input graph is not connected (we have a forest)
            Py_ssize_t ret = mst_edge_cur;
            while (mst_edge_cur < n-1) {
                mst_ind[2*mst_edge_cur+0] = -1;
                mst_ind[2*mst_edge_cur+1] = -1;
                mst_dist[mst_edge_cur]    = INFINITY;
                mst_edge_cur++;
            }
            if (verbose)
                DEADWOOD_PRINT("\b\b\b\b%3d%%", (int)(mst_edge_cur*100/(n-1)));
            return ret;
        }

        Py_ssize_t u = nns[triple_cur].i1;
        Py_ssize_t v = nns[triple_cur].i2;
        T d = nns[triple_cur].d;
        triple_cur++;

        if (ds.find(u) == ds.find(v))
            continue;

        mst_ind[2*mst_edge_cur+0] = u;
        mst_ind[2*mst_edge_cur+1] = v;
        mst_dist[mst_edge_cur]    = d;

        DEADWOOD_ASSERT(mst_edge_cur == 0 || mst_dist[mst_edge_cur] >= mst_dist[mst_edge_cur-1]);

        ds.merge(u, v);
        mst_edge_cur++;


        if (verbose)
            DEADWOOD_PRINT("\b\b\b\b%3d%%", (int)(mst_edge_cur*100/(n-1)));

        #if DEADWOOD_R
        Rcpp::checkUserInterrupt();
        #elif DEADWOOD_PYTHON
        if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        #endif
    }

    *maybe_inexact = 0;  // TODO !!!!

    if (verbose) DEADWOOD_PRINT("\b\b\b\bdone.\n");

    return mst_edge_cur;
}

#endif /* /Cmst_from_nn DEPRECATED */


#endif
