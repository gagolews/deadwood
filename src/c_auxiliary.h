/*  Deadwood: Outlier Detection via Minimum Spanning Trees
 *
 *  Copyleft (C) 2025-2026, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_auxiliary_h
#define __c_auxiliary_h


#include "c_common.h"
#include "c_kneedle.h"
#include <stdexcept>
#include <memory>



/*! Detect elbow point of a shifted array
 *
 *  @param mst_d size m - edge weights
 *  @param m length of mst_d
 *  @param max_contamination maximal contamination level;
 *         negative values will be used as actual contamination levels
 *  @param ema_dt controls the exponential moving average smoothing parameter
 *         alpha = 1-exp(-dt) (in elbow detection)
 *  @param contamination [out] array of length k;
 *         detected contamination levels in each cluster
 *  @param threshold_index [out] index in mst_d
 */
template<class FLOAT>
void Cget_contamination(
    const FLOAT* mst_d,
    Py_ssize_t m,
    FLOAT max_contamination,
    FLOAT ema_dt,
    FLOAT& contamination,
    Py_ssize_t& threshold_index
) {
    if (max_contamination <= 0.0) {
        contamination = -max_contamination;
        threshold_index =  int(m*(1.0-contamination));
    }
    else {
        Py_ssize_t shift = (int)(m*(1.0-max_contamination));
        Py_ssize_t elbow_index = Ckneedle_increasing(mst_d+shift, m-shift, true, ema_dt);
        if (elbow_index == 0) {
            threshold_index = m;
            contamination = 0.0;
        }
        else {
            threshold_index = shift+elbow_index+1;
            contamination = (m-threshold_index)/(FLOAT)(m+1);
        }
    }
}



/*! Reorders x w.r.t. a factor c
 *
 * y[ind[j]],...,y[ind[j+1]-1] give all x[i]s, in their original relative order,
 * for which c[i]==j.
 *
 * Elements corresponding to c[i] < 0 are put at the start of y.
 * c[i] >= k is disallowed.
 *
 * @param x [in] array of size n
 * @param n
 * @param c [in] array of size n with elements in {...,0,1,..,k-1}
 * @param k
 * @param y [out] array of size n
 * @param ind [out] array of size k+1
 */
template <class FLOAT>
void Csort_groups(
    const FLOAT* x, Py_ssize_t n, const Py_ssize_t* c, Py_ssize_t k,
    FLOAT* y, Py_ssize_t* ind
) {
    for (Py_ssize_t j=0; j<=k; ++j) ind[j] = 0;

    for (Py_ssize_t i=0; i<n; ++i) {
        DEADWOOD_ASSERT(c[i] < k);
        if (c[i] < 0)
            ++ind[0];
        else if (c[i] < k)
            ++ind[c[i]+1];
    }

    Py_ssize_t u = ind[0];
    ind[0] = 0;
    for (Py_ssize_t j=1; j<=k; ++j) {
        Py_ssize_t v = ind[j];
        ind[j] = u;  // sum of the original ind[0]..ind[j-1]
        u += v;
    }

    for (Py_ssize_t i=0; i<n; ++i) {
        if (c[i] < 0)
            y[ind[0]++] = x[i];
        else
            y[ind[c[i]+1]++] = x[i];
    }
}


/*! Identifies which MST edges must be skipped to obtain a forest whose
 *  connected components match a given partition.  If this is not possible,
 *  a more fine-grained split is generated.
 *
 *  This is as easy as finding all MST edges {u,v} for which c[u]≠c[v].
 *
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[i,0], mst_i[i,1]} specifies the i-th (undirected) edge
 *     in the spanning tree
 *  @param m number of rows in mst_i (edges)
 *  @param n length of c and the number of vertices in the spanning tree
 *  @param c [in] array of length n, where
 *      c[i] denotes the cluster ID of the i-th object
 *  @param skip [out] array of length m, indicating which edges
 *      of the tree must be skipped to create a subpartition of c
 *
 *  @return s number of edges in skip;  ideally, s=k-1, where k is the
 *      number of classes in c
 */
Py_ssize_t Cget_skip_edges(
    const Py_ssize_t* mst_i,  // size m [in]
    Py_ssize_t m,
    const Py_ssize_t* c,  // size n [in]
    Py_ssize_t n,
    bool* skip  // size m [out]
) {
    Py_ssize_t s = 0;

    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = mst_i[2*i+0];
        Py_ssize_t v = mst_i[2*i+1];
        DEADWOOD_ASSERT(u >= 0 && u < n);
        DEADWOOD_ASSERT(v >= 0 && v < n);
        if (c[u] != c[v]) {
            s++;
            skip[i] = true;
        }
        else
            skip[i] = false;
    }

    return s;
}


/*! Decode indexes based on a skip array.
 *
 * If `skip=[False, True, False, False, True, False, False]`,
 * then the indexes in `ind` are mapped in such a way that:
 * 0 → 0,
 * 1 → 2,
 * 2 → 3,
 * 3 → 5,
 * 4 → 6.
 *
 * This function might be useful if we apply a method on `X[~skip,:]`
 * (a subset of rows in `X`), obtain a vector of indexes `ind` relative to
 * the indexes of rows in `X[~skip,:]` as a result, and wish to translate `ind`
 * back to the original row space of `X[:,:]`.
 *
 * For instance, `unskip_indexes([0, 2, 1], [True, False, True, False, False])`
 * yields `[1, 4, 3]`.
 *
 * @param ind [in/out] array of m indexes in 0..k-1 to translate
 * @param m size of ind
 * @param skip Boolean array of size n with k elements equal to False
 * @param n size of skip
 */
void Cunskip_indexes(
    Py_ssize_t* ind, Py_ssize_t m,
    const bool* skip, Py_ssize_t n
) {
    if (m <= 0) return;
    DEADWOOD_ASSERT(n > 0);

    std::unique_ptr<Py_ssize_t[]> o(new Py_ssize_t[n]);  // actually, k elems needed
    Py_ssize_t k = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        if (!skip[i]) o[k++] = i;
    }

    for (Py_ssize_t i=0; i<m; ++i) {
        DEADWOOD_ASSERT(ind[i] >= 0 && ind[i] < k);
        ind[i] = o[ind[i]];
    }

    // std::vector<Py_ssize_t> o(m);
    // Cargsort(o.data(), ind, m, false);
    //
    // Py_ssize_t j = 0;
    // Py_ssize_t k = 0;
    // for (Py_ssize_t i=0; i<n; ++i) {
    //     if (skip[i]) continue;
    //
    //     if (ind[o[k]] == j) {
    //         ind[o[k]] = i;
    //         k++;
    //
    //         if (k == m) return;
    //     }
    //
    //     j++;
    // }
    //
    // throw std::domain_error("index to translate out of range");
}


/*! Encode indexes based on a skip array.
 *
 * If `skip=[False, True, False, False, True, False, False]`,
 * then the indexes in `ind` are mapped in such a way that:
 * 0 ← 0,
 * 1 ← 2,
 * 2 ← 3,
 * 3 ← 5,
 * 4 ← 6,
 * i.e., the indexes for which `skip` is False are mapped
 * to consecutive integers.  All other indexes are assigned the value -1.
 *
 * For instance, `skip_indexes([1, 4, 3], [True, False, True, False, False])`
 * yields `[0, 2, 1]`.
 *
 * @param ind [in/out] array of m indexes in 0..n-1 to translate
 * @param m size of ind
 * @param skip Boolean array of size n
 * @param n size of skip
 */
void Cskip_indexes(
    Py_ssize_t* ind, Py_ssize_t m,
    const bool* skip, Py_ssize_t n
) {
    if (m <= 0) return;
    DEADWOOD_ASSERT(n > 0);

    std::unique_ptr<Py_ssize_t[]> o(new Py_ssize_t[n]);

    Py_ssize_t k = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        if (skip[i]) o[i] = -1;
        else o[i] = (k++);
    }

    for (Py_ssize_t i=0; i<m; ++i) {
        DEADWOOD_ASSERT(ind[i] >= 0 && ind[i] < n);
        ind[i] = o[ind[i]];
    }
}


/*! Compute the degree of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *
 * @param ind c_contiguous matrix of size m*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n
 * @param m number of edges (rows in ind)
 * @param n number of vertices
 * @param deg [out] array of size n, where
 *     deg[i] will give the degree of the i-th vertex.
 */
void Cgraph_vertex_degrees(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    Py_ssize_t* deg /*out*/
) {
    for (Py_ssize_t i=0; i<n; ++i)
        deg[i] = 0;

    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];

        if (u < 0 || v < 0)
            throw std::domain_error("All elements must be >= 0");
        else if (u >= n || v >= n)
            throw std::domain_error("All elements must be < n");
        else if (u == v)
            throw std::domain_error("Self-loops are not allowed");

        deg[u]++;
        deg[v]++;
    }
}


/*! Compute the incidence list of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *  @param ind c_contiguous matrix of size m*2,
 *      where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n
 *  @param m number of edges (rows in ind)
 *  @param n number of vertices
 *  @param cumdeg [out] array of size n+1, where cumdeg[i+1] the sum of the first i vertex degrees
 *  @param inc [out] array of size 2*m; inc[cumdeg[i]]..inc[cumdeg[i+1]-1] gives the edges incident on the i-th vertex
 */
void Cgraph_vertex_incidences(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    Py_ssize_t* cumdeg,
    Py_ssize_t* inc
) {
    cumdeg[0] = 0;
    Cgraph_vertex_degrees(ind, m, n, cumdeg+1);

    Py_ssize_t cd = 0;
    for (Py_ssize_t i=1; i<n+1; ++i) {
        Py_ssize_t this_deg = cumdeg[i];
        cumdeg[i] = cd;
        cd += this_deg;
    }
    // that's not it yet; cumdeg is adjusted below


    for (Py_ssize_t e=0; e<m; ++e) {
        Py_ssize_t u = ind[2*e+0];
        Py_ssize_t v = ind[2*e+1];

        *(inc+cumdeg[u+1]) = e;
        ++(cumdeg[u+1]);

        *(inc+cumdeg[v+1]) = e;
        ++(cumdeg[v+1]);
    }

    DEADWOOD_ASSERT(cumdeg[0] == 0);
    DEADWOOD_ASSERT(cumdeg[n] == 2*m);


// #ifdef DEBUG
//     cumdeg = 0;
//     inc[0] = data;
//     for (Py_ssize_t i=0; i<n; ++i) {
//         DEADWOOD_ASSERT(inc[i] == data+cumdeg);
//         cumdeg += deg[i];
//     }
// #endif
}


#endif
