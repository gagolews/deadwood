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


template <class FLOAT>
class CDeadwood : public CMSTProcessorBase
{
private:

    const FLOAT* mst_d;
    FLOAT max_contamination;
    FLOAT ema_dt;
    Py_ssize_t max_debris_size;
    Py_ssize_t max_k;

    std::unique_ptr<bool[]> is_inlier;

    std::unique_ptr<Py_ssize_t[]> c;  // size n; cluster IDs of vertices
    std::unique_ptr<Py_ssize_t[]> d;  // size n; sub-cluster IDs of vertices
    std::unique_ptr<Py_ssize_t[]> s;  // size max_k; s[i] is the size of the i-th cluster
    std::unique_ptr<bool[]> skip_edges;  // std::vector<bool> has no .data()
    std::unique_ptr<Py_ssize_t[]> mst_cutsizes;  //< size m*2, each pair gives the sizes of the clusters that are formed when we cut out the corresponding edge
    std::unique_ptr<FLOAT[]> contamination;
    std::unique_ptr<FLOAT[]> weight_thresholds;

    Py_ssize_t k;  // the number of connected components identified

    Py_ssize_t cur_s;
    std::unique_ptr<Py_ssize_t[]> cur_e;
    std::unique_ptr<FLOAT[]> cur_d;

    Py_ssize_t cur_p;
    std::unique_ptr<Py_ssize_t[]> cur_v;

    /* helper for mark_cluster */
    Py_ssize_t mark_cluster_visitor(Py_ssize_t v, Py_ssize_t e)
    {
        Py_ssize_t w;

        if (e < 0) {
            w = v;
        }
        else if (skip_edges[e])
            return 0;
        else {
            Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
            w = mst_i[2*e+(1-iv)];
            cur_e[cur_s++] = e;
        }

        DEADWOOD_ASSERT(c[w] < 0);
        c[w] = k;

        Py_ssize_t curs = 1;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) curs += mark_cluster_visitor(w, *pe);
        }

        if (e >= 0) {
            Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
            mst_cutsizes[2*e+(1-iv)] = curs;
            mst_cutsizes[2*e+iv] = -1;
            //mst_cutsizes[2*e+iv] = this_component_size-curs;  // t.b.d. later
        }

        return curs;
    }


    /* Let v and its neighbours be marked as members of the k-th cluster */
    void mark_cluster(Py_ssize_t v)
    {
        cur_s = 0;
        s[k] = mark_cluster_visitor(v, -1);
        DEADWOOD_ASSERT(cur_s+1 == s[k]);

        for (Py_ssize_t i=0; i<cur_s; i++) {
            Py_ssize_t e = cur_e[i];
            cur_d[i] = mst_d[e];
            if (mst_cutsizes[2*e+0]>=0)
                mst_cutsizes[2*e+1] = s[k]-mst_cutsizes[2*e+0];
            else
                mst_cutsizes[2*e+0] = s[k]-mst_cutsizes[2*e+1];
        }
        std::sort(cur_d.get(), cur_d.get()+cur_s);

        Py_ssize_t elbow_index;
        Cget_contamination(
            cur_d.get(), cur_s, max_contamination, ema_dt,
            contamination[k], elbow_index
        );
        weight_thresholds[k] = (elbow_index+1<cur_s)?cur_d[elbow_index+1]:INFINITY;
    }



    void mark_inliers_visitor(Py_ssize_t v, Py_ssize_t e)
    {
        Py_ssize_t w;

        if (e < 0) {
            w = v;
        }
        else if (skip_edges[e] || mst_d[e] >= weight_thresholds[k])
            return;
        else {
            Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
            w = mst_i[2*e+(1-iv)];
        }

        cur_v[cur_p++] = w;
        d[w] = 1;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) mark_inliers_visitor(w, *pe);
        }
    }



    /* Marks inliers in a cluster consisting of edges in cur_e */
    Py_ssize_t mark_inliers()
    {
        Py_ssize_t changed_inliers = 0;

        for (Py_ssize_t i=0; i<cur_s; ++i) {
            d[mst_i[2*cur_e[i]+0]] = -1;
            d[mst_i[2*cur_e[i]+1]] = -1;
        }

        for (Py_ssize_t i=0; i<cur_s; ++i) {
            for (Py_ssize_t j=0; j<=1; ++j) {
                Py_ssize_t v = mst_i[2*cur_e[i]+j];
                if (d[v]>=0) continue;

                cur_p = 0;
                mark_inliers_visitor(v, -1);

                if (cur_p > max_debris_size) {
                    for (Py_ssize_t u=0; u<cur_p; ++u) {
                        if (!is_inlier[cur_v[u]]) {
                            changed_inliers++;
                            is_inlier[cur_v[u]] = true;
                        }
                    }
                }
            }
        }

        return changed_inliers;
    }


public:
    CDeadwood(
        const FLOAT* mst_d,
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        FLOAT max_contamination,
        FLOAT ema_dt,
        Py_ssize_t max_debris_size,
        Py_ssize_t max_k,
        Py_ssize_t k,
        const Py_ssize_t* mst_cut=nullptr,
        const Py_ssize_t* mst_cumdeg=nullptr,
        const Py_ssize_t* mst_inc=nullptr
    ) : CMSTProcessorBase(mst_i, m, n, mst_cumdeg, mst_inc),
        mst_d(mst_d), max_contamination(max_contamination),
        ema_dt(ema_dt), max_debris_size(max_debris_size),
        max_k(max_k)
    {
        DEADWOOD_ASSERT(this->cumdeg);
        DEADWOOD_ASSERT(this->inc);

        is_inlier.reset(new bool[n]);

        c.reset(new Py_ssize_t[n]);
        d.reset(new Py_ssize_t[n]);
        s.reset(new Py_ssize_t[max_k]);
        contamination.reset(new FLOAT[max_k]);
        weight_thresholds.reset(new FLOAT[max_k]);
        mst_cutsizes.reset(new Py_ssize_t[2*m]);
        cur_e.reset(new Py_ssize_t[m]);
        cur_v.reset(new Py_ssize_t[n]);
        cur_d.reset(new FLOAT[m]);

        skip_edges.reset(new bool[m]);
        for (Py_ssize_t e=0; e<m; ++e) skip_edges[e] = false;
        if (mst_cut && k > 1) {
            for (Py_ssize_t i=0; i<k-1; ++i) {
                DEADWOOD_ASSERT(mst_cut[i] >= 0 && mst_cut[i] < m);
                DEADWOOD_ASSERT(!skip_edges[mst_cut[i]]);
                skip_edges[mst_cut[i]] = true;
            }
        }
    }


    Py_ssize_t process()
    {
        for (Py_ssize_t v=0; v<n; ++v) c[v] = -1;
        for (Py_ssize_t v=0; v<n; ++v) is_inlier[v] = false;
        for (Py_ssize_t i=0; i<max_k; ++i) s[i] = 0;
        for (Py_ssize_t i=0; i<2*m; ++i) mst_cutsizes[i] = -1;

        k = 0;
        for (Py_ssize_t v=0; v<n; ++v) {
            if (c[v] >= 0) continue;  // already visited -> skip
            DEADWOOD_ASSERT(k<max_k);
            mark_cluster(v);
            mark_inliers();
            k++;
        }

        return k;
    }


    void get_outliers(Py_ssize_t* c)
    {
        for (Py_ssize_t v=0; v<n; ++v)
            c[v] = 1-(Py_ssize_t)is_inlier[v];
    }


    void get_contamination(FLOAT* c)
    {
        for (Py_ssize_t i=0; i<k; ++i)
            c[i] = contamination[i];
    }


    void get_mst_cut(Py_ssize_t* c)
    {
        Py_ssize_t i=0;
        for (Py_ssize_t e=0; e<m; ++e)
            if (skip_edges[e]) c[i++] = e;
        DEADWOOD_ASSERT(i == k-1);
    }

};


/*! The Deadwood outlier detection algorithm
 *
 *  @param mst_d size m - edge weights
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[i,0], mst_i[i,1]} specifies the i-th (undirected) edge
 *     in the spanning tree
 *  @param mst_cut [in/out] array of size max_k-1; indexes of cut edges
 *     defining a spanning forest with k connected components;
 *     the first k-1 indexes define the initial partition
 *  @param m number of rows in mst_i (edges)
 *  @param n length of c and the number of vertices in the spanning tree
 *  @param k number of initial clusters
 *  @param max_k maximal number of clusters to identify
 *  @param is_outlier [out] array of length n, c[i]==1 marks an outlier
 *         and c[i]==0 denotes an inlier
 *  @param max_debris_size connected components of size <= max_debris_size will
 *         be treated as outliers
 *  @param max_contamination maximal contamination level;
 *         negative values will be used as requested contamination levels
 *  @param ema_dt controls the exponential moving average smoothing parameter
 *         alpha = 1-exp(-dt) (in elbow detection)
 *  @param contamination [out] array of length max_k;
 *         detected contamination levels in each cluster
 *  @param mst_cumdeg an array of length n+1 or NULL; see Cgraph_vertex_incidences
 *  @param mst_inc an array of length 2*m or NULL; see Cgraph_vertex_incidences
 *
 *  @return number of detected clusters
 */
template <class FLOAT>
Py_ssize_t Cdeadwood(
    const FLOAT* mst_d,  // size m [in]
    const Py_ssize_t* mst_i,  // size m [in]
    Py_ssize_t m,
    Py_ssize_t n,
    FLOAT max_contamination,
    FLOAT ema_dt,
    Py_ssize_t max_debris_size,
    Py_ssize_t k,
    Py_ssize_t max_k,
    Py_ssize_t* mst_cut,  // size max_k-1 [in/out]
    FLOAT* contamination,  // size max_k [out]
    Py_ssize_t* is_outlier,  // size n [out]
    const Py_ssize_t* mst_cumdeg=nullptr,
    const Py_ssize_t* mst_inc=nullptr
) {
    DEADWOOD_ASSERT(k >= 1);
    DEADWOOD_ASSERT(k <= max_k);
    DEADWOOD_ASSERT(max_k < n);
    DEADWOOD_ASSERT(m == n-1);
    DEADWOOD_ASSERT(n > 1);
    DEADWOOD_ASSERT(max_contamination >= -1.0 && max_contamination <= 1.0);

    CDeadwood<FLOAT> dw(
        mst_d, mst_i, m, n, max_contamination, ema_dt, max_debris_size, max_k,
        k, mst_cut, mst_cumdeg, mst_inc
    );

    Py_ssize_t _k = dw.process();

    dw.get_outliers(is_outlier);
    dw.get_contamination(contamination);
    if (_k != k) dw.get_mst_cut(mst_cut);

    return _k;
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
