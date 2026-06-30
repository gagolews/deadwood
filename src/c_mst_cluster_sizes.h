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


#ifndef __c_mst_cluster_sizes_h
#define __c_mst_cluster_sizes_h

#include "c_common.h"
#include "c_mst_helpers.h"


/** See Cmst_get_cluster_sizes below.
 */
class CMSTClusterSizeGetter : public CMSTProcessorBase
{
private:

    Py_ssize_t max_k;
    Py_ssize_t* c;  // nullable or length n; cluster IDs of the vertices
    Py_ssize_t* s;  // NULL or of size max_k >= k, where k is the number of clusters; s[i] is the size of the i-th cluster
    Py_ssize_t* mst_cutsizes;  //< size m*2, each pair gives the sizes of the clusters that are formed when we cut out the corresponding edge
    const bool* skip_edges;  // size m or NULL
    Py_ssize_t  k;  // the number of connected components identified

    Py_ssize_t visit(Py_ssize_t v, Py_ssize_t e)
    {
        Py_ssize_t w;

        if (e < 0) {
            w = v;
        }
        else if (skip_edges && skip_edges[e])
            return 0;
        else {
            Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
            w = mst_i[2*e+(1-iv)];
        }

        DEADWOOD_ASSERT(c[w] < 0);
        c[w] = k;

        Py_ssize_t curs = 1;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) curs += visit(w, *pe);
        }

        if (mst_cutsizes && e>=0) {
            Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
            mst_cutsizes[2*e+(1-iv)] = curs;
            //mst_cutsizes[2*e+iv] = this_component_size-curs;  // t.b.d. later
        }

        return curs;
    }


public:
    CMSTClusterSizeGetter(
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* c,
        Py_ssize_t max_k,
        Py_ssize_t* s=nullptr,
        Py_ssize_t* mst_cutsizes=nullptr,
        const bool* skip_edges=nullptr,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr
    ) : CMSTProcessorBase(mst_i, m, n, cumdeg, inc),
        max_k(max_k), c(c), s(s), mst_cutsizes(mst_cutsizes), skip_edges(skip_edges), k(-1)
    {
        DEADWOOD_ASSERT(this->c);
        DEADWOOD_ASSERT(this->cumdeg);
        DEADWOOD_ASSERT(this->inc);
        DEADWOOD_ASSERT(!this->mst_cutsizes || this->s);
    }


    Py_ssize_t process()
    {
        for (Py_ssize_t v=0; v<n; ++v) c[v] = -1;
        if (s) for (Py_ssize_t i=0; i<max_k; ++i) s[i] = 0;
        if (mst_cutsizes) for (Py_ssize_t i=0; i<2*m; ++i) mst_cutsizes[i] = -1;

        k = 0;
        for (Py_ssize_t v=0; v<n; ++v) {
            if (c[v] >= 0) continue;  // already visited -> skip

            if (s) {
                DEADWOOD_ASSERT(k<max_k);
                s[k] = visit(v, -1);
            }
            else
                visit(v, -1);

            k++;
        }

        if (mst_cutsizes) {
            for (Py_ssize_t e=0; e<m; ++e) {
                if (skip_edges && skip_edges[e]) continue;
                if (mst_cutsizes[2*e+0]>=0)
                    mst_cutsizes[2*e+1] = s[c[mst_i[2*e+0]]]-mst_cutsizes[2*e+0];
                else
                    mst_cutsizes[2*e+0] = s[c[mst_i[2*e+1]]]-mst_cutsizes[2*e+1];
            }
        }

        return k;
    }

};


/*! Labels connected components in a spanning forest (where skip_edges
 *  designate the edges omitted from the tree) and fetch their sizes
 *
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[i,0], mst_i[i,1]} specifies the i-th (undirected) edge
 *     in the spanning tree
 *  @param m number of rows in mst_i (edges)
 *  @param n length of c and the number of vertices in the spanning tree
 *  @param c [out] array of length n, where
 *      c[i] denotes the cluster ID (in {0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1
 *  @param max_k the actual size of s (a safeguard)
 *  @param cl_sizes [out] array of length max_k >= k, where k is the number of
 *      connected components in the forest; s[i] gives the size of
 *      the i-th cluster;  pass NULL to get only the cluster labels;
 *      obviously, k<=n; e.g., if m==n-1, then k=sum(skip_edges)+1
 *  @param mst_cutsizes [out] c_contiguous matrix of size m*2,
 *     where {mst_cutsizes[i,0], mst_cutsizes[i,1]} specifies the number
 *     of vertices in the two connected components that arise when we remove
 *     the i-th edge from the forest, or {-1, -1} if this edge is non-existent
 *  @param mst_cumdeg an array of length n+1 or NULL; see Cgraph_vertex_incidences
 *  @param mst_inc an array of length 2*m or NULL; see Cgraph_vertex_incidences
 *  @param mst_skip Boolean array of length m or NULL; indicates the edges to skip
 */
Py_ssize_t Cmst_cluster_sizes(
    const Py_ssize_t* mst_i,
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t* c,
    Py_ssize_t max_k=0,
    Py_ssize_t* cl_sizes=nullptr,
    Py_ssize_t* mst_cutsizes=nullptr,
    const bool* mst_skip=nullptr,
    const Py_ssize_t* mst_cumdeg=nullptr,
    const Py_ssize_t* mst_inc=nullptr
) {
    CMSTClusterSizeGetter get(
        mst_i, m, n, c, max_k, cl_sizes, mst_cutsizes, mst_skip,
        mst_cumdeg, mst_inc
    );
    return get.process();  // modifies c in place
}

#endif
