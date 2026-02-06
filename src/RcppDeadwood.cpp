/*  deadwood R interface
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


#include "c_common.h"
#include "c_kneedle.h"
#include "c_deadwood.h"
#include <cmath>

using namespace Rcpp;


// TODO: sort_groups
// TODO: mst_cluster_sizes
// TODO: get_skip_edges
// TODO: unskip_indexes
// TODO: skip_indexes
// TODO: graph_vertex_degrees
// TODO: graph_vertex_incidences
// TODO: mst_label_imputer



//' @title Knee/Elbow Point Detection
//'
//' @description
//' Finds the most significant knee/elbow using the Kneedle algorithm
//' with exponential smoothing.
//'
//' @param x data vector (increasing)
//'
//' @param convex whether the data in \code{x} are convex-ish (elbow detection)
//'         or not (knee lookup)
//'
//' @param dt controls the smoothing parameter \eqn{\alpha = 1-\exp(-dt)}
//'         of the exponential moving average,
//'         \eqn{y_i = \alpha x_i + (1-\alpha) y_{i-1}}, \eqn{y_1 = x_1}
//'
//'
//' @return
//' Returns the index of the knee/elbow point; 1 if not found.
//'
//' @references
//' V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan,
//' Finding a "Kneedle" in a haystack: Detecting knee points in system behavior,
//' In: 31st Intl. Conf. Distributed Computing Systems Workshops,
//' 2011, 166-171, \doi{10.1109/ICDCSW.2011.20}
//'
//' @name kneedle
//' @rdname kneedle
//' @export
// [[Rcpp::export]]
double kneedle_increasing(NumericVector x, bool convex=true, double dt=0.01)
{
    Py_ssize_t n = (Py_ssize_t)x.size();
    return Ckneedle_increasing(REAL(x), n, convex, dt)+1.0;
}


// [[Rcpp::export(".deadwood")]]
LogicalVector dot_deadwood(
    NumericMatrix mst,
    NumericVector cut_edges,
    double max_contamination,
    double ema_dt,
    int max_debris_size,
    bool verbose
) {
    if (verbose) DEADWOOD_PRINT("[deadwood] Determining clusters.\n");

    Py_ssize_t n = mst.nrow()+1;

    std::vector<Py_ssize_t> mst_i((n-1)*2);
    std::vector<double> mst_d(n-1);

    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_i[i*2+0] = (Py_ssize_t)mst(i, 0) - 1;  // 1-based to 0-based indexes
        mst_i[i*2+1] = (Py_ssize_t)mst(i, 1) - 1;  // 1-based to 0-based indexes
        mst_d[i] = mst(i, 2);
    }


    Py_ssize_t k = cut_edges.size()+1;

    std::vector<Py_ssize_t> mst_cut(k-1);
    for (Py_ssize_t i=0; i<k-1; ++i) {
        mst_cut[i] = (Py_ssize_t)cut_edges[i]-1;
        DEADWOOD_ASSERT(mst_cut[i] >= 0 && mst_cut[i] < n-1);
    }

    std::vector<double> contamination(k);
    std::vector<Py_ssize_t> is_outlier(n);
    Cdeadwood(
        mst_d.data(), mst_i.data(), mst_cut.data(), n-1, n, k,
        max_contamination, ema_dt, max_debris_size,
        contamination.data(), is_outlier.data(), NULL, NULL
    );

    LogicalVector res(n);
    for (Py_ssize_t i=0; i<n; ++i) {
        if (is_outlier[i]) res[i] = TRUE;
        else res[i] = FALSE;
    }

    NumericVector contaminationr(k);
    for (Py_ssize_t i=0; i<k; ++i)
        contaminationr[i] = contamination[i];

    res.attr("contamination") = contaminationr;

    if (verbose) DEADWOOD_PRINT("[deadwood] Done.\n");

    return res;

}
