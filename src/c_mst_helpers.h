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


#ifndef __c_mst_helpers_h
#define __c_mst_helpers_h

#include "c_common.h"
#include <stdexcept>
#include <memory>


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

/* ************************************************************************** */


class CMSTProcessorBase
{
protected:
    const Py_ssize_t* mst_i;  // size m*2, elements in [0,n)
    const Py_ssize_t m;  // preferably == n-1; number of edges in mst_i
    const Py_ssize_t n;  // number of vertices

    const Py_ssize_t* cumdeg;  // nullable or length n+1
    const Py_ssize_t* inc;     // nullable or length 2*m

    std::unique_ptr<Py_ssize_t[]> _cumdeg;  // data buffer for cumdeg (optional)
    std::unique_ptr<Py_ssize_t[]> _inc;     // data buffer for inc (optional)


public:

    CMSTProcessorBase(const CMSTProcessorBase&) = delete;
    CMSTProcessorBase& operator=(const CMSTProcessorBase&) = delete;

    CMSTProcessorBase(
        const Py_ssize_t* mst_i,
        const Py_ssize_t m,
        const Py_ssize_t n,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr
    ) :
        mst_i(mst_i), m(m), n(n), cumdeg(cumdeg), inc(inc)
    {
        if (!cumdeg) {
            DEADWOOD_ASSERT(!inc);
            _cumdeg.reset(new Py_ssize_t[n+1]);
            _inc.reset(new Py_ssize_t[2*m]);
            Cgraph_vertex_incidences(mst_i, m, n, _cumdeg.get(), _inc.get());
            this->cumdeg = _cumdeg.get();
            this->inc = _inc.get();
        }
        else {
            DEADWOOD_ASSERT(inc);
        }
    }
};


#endif
