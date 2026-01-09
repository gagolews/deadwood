# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Core functions and classes
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2025-2026, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


cimport libc.math
from libcpp cimport bool

#import numpy as np
cimport numpy as np
np.import_array()

#import warnings


ctypedef fused T:
    int
    long
    long long
    Py_ssize_t
    float
    double

ctypedef fused floatT:
    float
    double


cdef extern from "../src/c_kneedle.h":

    Py_ssize_t Ckneedle_increasing[floatT](
        const floatT* x, Py_ssize_t n, bool convex, floatT dt
    )


cpdef Py_ssize_t kneedle_increasing(
        floatT[::1] x, bool convex=True, floatT dt=0.01
    ):
    """
    deadwood.kneedle_increasing(x, convex=True, dt=0.01)

    Finds the most significant knee/elbow using the Kneedle algorithm
    with exponential smoothing.


    Parameters
    ----------

    x : ndarray
        data vector (increasing)
    convex : bool
        whether the data in `x` are convex-ish (elbow detection)
        or not (knee lookup)
    dt : float
        controls the smoothing parameter :math:`\\alpha = 1-\\exp(-dt)`
        of the exponential moving average,
        :math:`y_i = \\alpha x_i + (1-\\alpha) y_{i-1}`,
        :math:`y_1 = x_1`.


    Returns
    -------

    index : integer
        the index of the knee/elbow point; 0 if not found


    References
    ----------

    .. [1]
        V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan, *Finding a "Kneedle"
        in a haystack: Detecting knee points in system behavior*,
        In: *31st Intl. Conf. Distributed Computing Systems Workshops*,
        2011, pp. 166-171, DOI: 10.1109/ICDCSW.2011.20

    """
    cdef Py_ssize_t n = x.shape[0]
    return Ckneedle_increasing(&x[0], n, convex, dt)

