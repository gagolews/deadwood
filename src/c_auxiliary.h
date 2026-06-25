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



/*! Detect outlier contamination levels as an elbow point in a shifted array
 *
 *  @param x array of size n
 *  @param n length of x
 *  @param max_contamination maximal contamination level;
 *         negative values will be used as actual contamination levels
 *  @param ema_dt controls the exponential moving average smoothing parameter
 *         alpha = 1-exp(-dt) (in elbow detection)
 *  @param contamination [out] array of length k;
 *         detected contamination levels in each cluster
 *  @param elbow_index [out] index in x or n if not found
 */
template<class FLOAT>
void Cget_contamination(
    const FLOAT* x,
    Py_ssize_t n,
    FLOAT max_contamination,
    FLOAT ema_dt,
    FLOAT& contamination,
    Py_ssize_t& elbow_index
) {
    DEADWOOD_ASSERT(max_contamination >= -1.0 && max_contamination <= 1.0);
    if (max_contamination <= 0.0) {
        contamination = -max_contamination;
        elbow_index   = int(n*(1.0-contamination))-1;
    }
    else {
        Py_ssize_t shift = (int)(n*(1.0-max_contamination));
        elbow_index = Ckneedle_increasing(x+shift, n-shift, true, ema_dt);
        elbow_index += shift;
        contamination = (n-elbow_index-1)/(FLOAT)(n+1);
    }

    DEADWOOD_ASSERT(contamination >= 0.0 && contamination <= 1.0);
    DEADWOOD_ASSERT(elbow_index >= 0 && elbow_index < n);
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


#endif
