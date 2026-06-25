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


#ifndef __c_deadwood_h
#define __c_deadwood_h


#include "c_common.h"
#include "c_mst_helpers.h"
#include <memory>


#include "c_mst_cluster_sizes.h"


/*! The Deadwood outlier detection algorithm
 *
 *  @param mst_d size m - edge weights
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[i,0], mst_i[i,1]} specifies the i-th (undirected) edge
 *     in the spanning tree
 *  @param mst_cut array of size k-1; indexes of cut edges defining a spanning
 *     forest with k connected components
 *  @param m number of rows in mst_i (edges)
 *  @param n length of c and the number of vertices in the spanning tree
 *  @param k number of initial clusters
 *  @param c [out] array of length n, c[i]==1 marks an outlier
 *         and c[i]==0 denotes an inlier
 *  @param max_debris_size connected components of size <= max_debris_size will
 *         be treated as outliers
 *  @param max_contamination maximal contamination level;
 *         negative values will be used as requested contamination levels
 *  @param ema_dt controls the exponential moving average smoothing parameter
 *         alpha = 1-exp(-dt) (in elbow detection)
 *  @param contamination [out] array of length k;
 *         detected contamination levels in each cluster
 *  @param mst_cumdeg an array of length n+1 or NULL; see Cgraph_vertex_incidences
 *  @param mst_inc an array of length 2*m or NULL; see Cgraph_vertex_incidences
 */
template <class FLOAT>
void Cdeadwood(
    const FLOAT* mst_d,  // size m [in]
    const Py_ssize_t* mst_i,  // size m [in]
    const Py_ssize_t* mst_cut,  // size k-1 [in]
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t k,
    FLOAT max_contamination,
    FLOAT ema_dt,
    Py_ssize_t max_debris_size,
    FLOAT* contamination,  // size k [out]
    Py_ssize_t* c,  // size n [out]
    const Py_ssize_t* mst_cumdeg=nullptr,
    const Py_ssize_t* mst_inc=nullptr
) {
    DEADWOOD_ASSERT(k >= 1 && k <= n);
    DEADWOOD_ASSERT(m == n-1);
    DEADWOOD_ASSERT(n > 1);
    DEADWOOD_ASSERT(max_contamination >= -1.0 && max_contamination <= 1.0);

    std::unique_ptr<Py_ssize_t[]> sizes(new Py_ssize_t[n]);  // upper bound for the number of clusters
    std::unique_ptr<bool[]> mst_skip(new bool[m]);  // std::vector<bool> has no .data()
    for (Py_ssize_t i=0; i<m; ++i) mst_skip[i] = false;

    CMSTClusterSizeGetter size_getter(mst_i, m, n, c, n, sizes.get(), nullptr, mst_cumdeg, mst_inc, mst_skip.get());

    // set mst_skip and contamination
    if (k == 1) {
        Py_ssize_t elbow_index = m-1;
        Cget_contamination(
            mst_d, m, max_contamination, ema_dt,
            /*out*/contamination[0], /*out*/elbow_index
        );

        // DEADWOOD_PRINT("%d-%d\n", threshold_index, m);
        for (Py_ssize_t i=elbow_index+1; i<m; ++i)
            mst_skip[i] = true;
    }
    else {
        for (Py_ssize_t i=0; i<k-1; ++i) {
            DEADWOOD_ASSERT(mst_cut[i] >= 0 && mst_cut[i] < m);
            DEADWOOD_ASSERT(!mst_skip[mst_cut[i]]);
            mst_skip[mst_cut[i]] = true;
        }

        Py_ssize_t _k = size_getter.process();  // sets c and sizes based on the current mst_skip
        DEADWOOD_ASSERT(_k == k);

        std::unique_ptr<Py_ssize_t[]> edge_labels(new Py_ssize_t[m]);
        for (Py_ssize_t i=0; i<m; ++i) {
            if (c[mst_i[2*i+0]]>=0 && c[mst_i[2*i+0]] == c[mst_i[2*i+1]])
                edge_labels[i] = c[mst_i[2*i+0]];
            else
                edge_labels[i] = -1;
        }

        std::unique_ptr<FLOAT[]> mst_d_grp(new FLOAT[n]);
        std::unique_ptr<Py_ssize_t[]> ind_grp(new Py_ssize_t[k+1]);
        Csort_groups(mst_d, m, edge_labels.get(), k, mst_d_grp.get(), ind_grp.get());

        std::unique_ptr<FLOAT[]> weight_thresholds(new FLOAT[k]);
        for (Py_ssize_t i=0; i<k; ++i) {
            Py_ssize_t mi = sizes[i]-1;
            Py_ssize_t elbow_index;
            Cget_contamination(
                mst_d_grp.get()+ind_grp[i], mi, max_contamination, ema_dt,
                /*out*/contamination[i], /*out*/elbow_index
            );
            weight_thresholds[i] = (elbow_index+1 < mi)?mst_d_grp[ind_grp[i]+elbow_index+1]:INFINITY;
        }

        for (Py_ssize_t i=0; i<m; ++i) {
            if (edge_labels[i] >= 0 && mst_d[i] >= weight_thresholds[edge_labels[i]])
                mst_skip[i] = true;
        }
    }


    size_getter.process();  // sets c and sizes based on the current mst_skip
    // DEADWOOD_PRINT("%d\n", _k);

    for (Py_ssize_t i=0; i<n; ++i) {
        DEADWOOD_ASSERT(c[i] >= 0 && c[i] < n);
        DEADWOOD_ASSERT(sizes[c[i]] > 0);
        if (sizes[c[i]] <= max_debris_size)
            c[i] = 1;
        else
            c[i] = 0;
    }
}


/* ************************************************************************** */


#if 0
/** Deadwood, connected=true
 */
template <class FLOAT> class CMSTBranchTrimmer : public CMSTProcessorBase
{
public:
    std::unique_ptr<FLOAT[]> mst_d;

private:
    const Py_ssize_t max_size;

    std::unique_ptr<Py_ssize_t[]> size;  // size (m,2)


    Py_ssize_t visit_get_sizes(Py_ssize_t v, Py_ssize_t e)
    {
        if (skip_edges && skip_edges[e]) return 0;
        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];

        DEADWOOD_ASSERT(e >= 0 && e < m);
        DEADWOOD_ASSERT(v >= 0 && v < n);
        DEADWOOD_ASSERT(w >= 0 && w < n);
        DEADWOOD_ASSERT(c[w] < 0);

        Py_ssize_t this_size = 1;
        c[w] = c[v];

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) this_size += visit_get_sizes(w, *pe);
        }

        size[2*e + (1-iv)] = this_size;
        size[2*e + iv] = -1;

        return this_size;
    }


    void visit_mark(Py_ssize_t v, Py_ssize_t e)
    {
        if (skip_edges && skip_edges[e]) return;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];

        if (c[w] > 0) return;  // already visited

        c[w] = 1;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) visit_mark(w, *pe);
        }
    }


    void visit_update_mst_d(Py_ssize_t v, Py_ssize_t e, FLOAT max_d)
    {
        if (skip_edges && skip_edges[e]) return;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];

        if (size[2*e + (1-iv)] > max_size) {
            // no change to mst_d
            max_d = mst_d[e];
        }
        else {
            if (max_d < mst_d[e]) max_d = mst_d[e];
            else mst_d[e] = max_d;
        }

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) visit_update_mst_d(w, *pe, max_d);
        }
    }


public:
    CMSTBranchTrimmer(
        const FLOAT* orig_mst_d,
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* c,
        Py_ssize_t max_size,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr
    ) :
        CMSTProcessorBase(mst_i, m, n, c, cumdeg, inc),
        max_size(max_size)
    {
        DEADWOOD_ASSERT(this->c);
        DEADWOOD_ASSERT(this->cumdeg);
        DEADWOOD_ASSERT(this->inc);
        DEADWOOD_ASSERT(m == n-1);

        mst_d.reset(new FLOAT[m]);
        for (Py_ssize_t i=0; i<m; ++i) mst_d[i] = orig_mst_d[i];

        size.reset(new Py_ssize_t[2*m]);
        for (Py_ssize_t i=0; i<2*m; ++i) size[i] = -1;

        for (Py_ssize_t v=0; v<n; ++v) c[v] = -1;

        Py_ssize_t v = 0;  // any vertex
        c[v] = 0;
        for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
            visit_get_sizes(v, *pe);
        }

        for (Py_ssize_t e=0; e<m; ++e) {
            DEADWOOD_ASSERT(size[2*e+0] > 0 || size[2*e+1] > 0);
            if (size[2*e+0] > 0)
                size[2*e+1] = n - size[2*e+0];
            else
                size[2*e+0] = n - size[2*e+1];
        }
    }


    void update_mst_d()
    {
        // this makes mst_d not sorted

        Py_ssize_t e;
        for (e=0; e<m; ++e) {
            if (size[2*e+0] > max_size && size[2*e+1] > max_size)
                break;
        }
        DEADWOOD_ASSERT(e<m);

        // e will not be trimmed out
        Py_ssize_t v = mst_i[2*e+0];
        for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
            visit_update_mst_d(v, *pe, mst_d[*pe]);
        }
    }


    void trim(FLOAT trim_d)
    {
        for (Py_ssize_t e=0; e<m; ++e) {
            if (mst_d[e] < trim_d) continue;

            Py_ssize_t iv = (size[2*e+0]>=size[2*e+1])?0:1;
            Py_ssize_t v = mst_i[2*e+iv];
            if (c[v] > 0) continue;
            if (size[2*e+(1-iv)] > max_size) continue;
            visit_mark(v, e);
        }
    }

};


template <class FLOAT>
void Cdeadwood_connected(
    const FLOAT* mst_d,  // size m [in]
    const Py_ssize_t* mst_i,  // size m [in]
    const Py_ssize_t* mst_cut,  // size k-1 [in]
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t k,
    FLOAT max_contamination,
    FLOAT ema_dt,
    Py_ssize_t max_debris_size,
    FLOAT* contamination,  // size k [out]
    Py_ssize_t* c,  // size n [out]
    const Py_ssize_t* mst_cumdeg=nullptr,
    const Py_ssize_t* mst_inc=nullptr
) {
    DEADWOOD_ASSERT(k >= 1 && k <= n);
    DEADWOOD_ASSERT(m == n-1);
    DEADWOOD_ASSERT(n > 1);
    DEADWOOD_ASSERT(max_contamination >= -1.0 && max_contamination <= 1.0);

    DEADWOOD_ASSERT(k == 1);

    CMSTBranchTrimmer tr(
        mst_d, mst_i, m, n, c, max_debris_size, mst_cumdeg, mst_inc
    );

    // tr.update_mst_d();

    std::unique_ptr<FLOAT[]> mst_d2(new FLOAT[m]);
    for (Py_ssize_t i=0; i<m; ++i) mst_d2[i] = tr.mst_d[i];

    std::sort(mst_d2.get(), mst_d2.get()+m);

    Py_ssize_t threshold_index = -1;
    Cget_contamination(
        mst_d2.get(), m, max_contamination, ema_dt,
        /*out*/contamination[0], /*out*/threshold_index
    );

    FLOAT trim_d = (threshold_index+1 < m)?mst_d2[threshold_index+1]:INFINITY;

    tr.trim(trim_d);  // modifies c in place

    return;
}
#endif


#endif
