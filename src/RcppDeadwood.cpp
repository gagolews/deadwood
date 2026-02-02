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


//' @title Knee/Elbow Point Detection with Kneedle
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
//' V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan, *
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
RObject dot_deadwood()
{
    return R_NilValue;
}
