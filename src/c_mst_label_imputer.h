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


#ifndef __c_mst_label_imputer_h
#define __c_mst_label_imputer_h

#include "c_common.h"
#include "c_mst_helpers.h"


/** See Cmst_label_imputer below.
 */
class CMSTMissingLabelsImputer : public CMSTProcessorBase
{
private:
    Py_ssize_t* c;  // nullable or length n; cluster IDs of the vertices
    const bool* skip_edges;  // size m

    void visit(Py_ssize_t v, Py_ssize_t e)
    {
        if (skip_edges && skip_edges[e]) return;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];

        DEADWOOD_ASSERT(c[v] >= 0);
        DEADWOOD_ASSERT(c[w] < 0);

        c[w] = c[v];

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) visit(w, *pe);
        }
    }


public:
    CMSTMissingLabelsImputer(
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* c,
        const bool* skip_edges=nullptr,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr
    ) : CMSTProcessorBase(mst_i, m, n, cumdeg, inc), c(c), skip_edges(skip_edges)
    {
        DEADWOOD_ASSERT(this->c);
        DEADWOOD_ASSERT(this->cumdeg);
        DEADWOOD_ASSERT(this->inc);
    }


    void process()
    {
        for (Py_ssize_t v=0; v<n; ++v) {
            if (c[v] < 0) continue;

            for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
                if (skip_edges && skip_edges[*pe]) continue;

                Py_ssize_t iv = (Py_ssize_t)(mst_i[2*(*pe)+1]==v);
                Py_ssize_t w = mst_i[2*(*pe)+(1-iv)];

                if (c[w] < 0) {  // descend into this branch to impute missing values
                    visit(v, *pe);
                }
            }
        }
    }

};


/*! Impute missing labels in all tree branches.
 *  All nodes in branches with class ID of -1 will be assigned their parent node's class.
 *
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[i,0], mst_i[i,1]} specifies the i-th (undirected) edge
 *     in the spanning tree
 *  @param m number of rows in mst_i (edges)
 *  @param n length of c and the number of vertices in the spanning tree
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster ID (in {-1, 0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1.  Class -1 represents missing values
 *      to be imputed
 *  @param mst_cumdeg an array of length n+1 or NULL; see Cgraph_vertex_incidences
 *  @param mst_inc an array of length 2*m or NULL; see Cgraph_vertex_incidences
 *  @param mst_skip Boolean array of length m or NULL; indicates the edges to skip
 */
void Cmst_label_imputer(
    const Py_ssize_t* mst_i,
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t* c,
    const bool* mst_skip=nullptr,
    const Py_ssize_t* mst_cumdeg=nullptr,
    const Py_ssize_t* mst_inc=nullptr
) {
    CMSTMissingLabelsImputer imp(mst_i, m, n, c, mst_skip, mst_cumdeg, mst_inc);
    imp.process();  // modifies c in place
}


#endif
