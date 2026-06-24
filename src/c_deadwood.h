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
#include "c_kneedle.h"
#include "c_auxiliary.h"
#include <stdexcept>
#include <memory>



/* ************************************************************************** */


class CMSTProcessorBase
{
protected:
    const Py_ssize_t* mst_i;  // size m*2, elements in [0,n)
    const Py_ssize_t m;  // preferably == n-1; number of edges in mst_i
    const Py_ssize_t n;  // number of vertices

    Py_ssize_t* c;  // nullable or length n; cluster IDs of the vertices

    const Py_ssize_t* cumdeg;  // nullable or length n+1
    const Py_ssize_t* inc;     // nullable or length 2*m
    const bool* skip_edges;    // nullable or length m

    std::unique_ptr<Py_ssize_t[]> _cumdeg;  // data buffer for cumdeg (optional)
    std::unique_ptr<Py_ssize_t[]> _inc;     // data buffer for inc (optional)


public:

    CMSTProcessorBase(
        const Py_ssize_t* mst_i,
        const Py_ssize_t m,
        const Py_ssize_t n,
        Py_ssize_t* c=nullptr,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr,
        const bool* skip_edges=nullptr
    ) :
        mst_i(mst_i), m(m), n(n), c(c),
        cumdeg(cumdeg), inc(inc), skip_edges(skip_edges)
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



/* ************************************************************************** */



/** See Cmst_get_cluster_sizes below.
 */
class CMSTClusterSizeGetter : public CMSTProcessorBase
{
private:

    Py_ssize_t max_k;
    Py_ssize_t* s;  // NULL or of size max_k >= k, where k is the number of clusters
    Py_ssize_t* mst_cutsizes;  //< size m*2, each pair gives the sizes of the clusters that are formed when we cut out the corresponding edge
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
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr,
        const bool* skip_edges=nullptr
    ) : CMSTProcessorBase(mst_i, m, n, c, cumdeg, inc, skip_edges),
        max_k(max_k), s(s), mst_cutsizes(mst_cutsizes), k(-1)
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
    const Py_ssize_t* mst_cumdeg=nullptr,
    const Py_ssize_t* mst_inc=nullptr,
    const bool* mst_skip=nullptr
) {
    CMSTClusterSizeGetter get(
        mst_i, m, n, c, max_k, cl_sizes, mst_cutsizes, mst_cumdeg, mst_inc, mst_skip
    );
    return get.process();  // modifies c in place
}



/* ************************************************************************** */



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

    if (k == 1) {
        Py_ssize_t threshold_index = -1;
        Cget_contamination(
            mst_d, m, max_contamination, ema_dt,
            /*out*/contamination[0], /*out*/threshold_index
        );

        // DEADWOOD_PRINT("%d-%d\n", threshold_index, m);
        DEADWOOD_ASSERT(threshold_index >= 0);
        for (Py_ssize_t i=threshold_index; i<m; ++i)
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
            Py_ssize_t threshold_index;
            Cget_contamination(
                mst_d_grp.get()+ind_grp[i], mi, max_contamination, ema_dt,
                /*out*/contamination[i], /*out*/threshold_index
            );
            DEADWOOD_ASSERT(threshold_index>=0);
            if (threshold_index < mi)
                weight_thresholds[i] = mst_d_grp[ind_grp[i]+threshold_index];
            else
                weight_thresholds[i] = INFINITY;
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

    DEADWOOD_ASSERT(threshold_index >= 0);
    FLOAT trim_d = (threshold_index < m)?mst_d2[threshold_index]:INFINITY;

    tr.trim(trim_d);  // modifies c in place

    return;
}
#endif


/* ************************************************************************** */




/** See Cmst_label_imputer below.
 */
class CMSTMissingLabelsImputer : public CMSTProcessorBase
{
private:

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
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr,
        const bool* skip_edges=nullptr
    ) : CMSTProcessorBase(mst_i, m, n, c, cumdeg, inc, skip_edges)
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
    const Py_ssize_t* mst_cumdeg=nullptr,
    const Py_ssize_t* mst_inc=nullptr,
    const bool* mst_skip=nullptr
) {
    CMSTMissingLabelsImputer imp(mst_i, m, n, c, mst_cumdeg, mst_inc, mst_skip);
    imp.process();  // modifies c in place
}


#endif
