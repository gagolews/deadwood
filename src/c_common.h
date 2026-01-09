/*  Common functions, macros, includes
 *
 *  Copyleft (C) 2018-2026, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_common_h
#define __c_common_h


#ifdef DEADWOOD_PYTHON
#undef DEADWOOD_PYTHON
#define DEADWOOD_PYTHON 1
#endif

#ifdef DEADWOOD_R
#undef DEADWOOD_R
#define DEADWOOD_R 1
#endif


#include <stdexcept>
#include <string>
#include <limits>
#include <cmath>


#ifndef DEADWOOD_ASSERT
#define __DEADWOOD_STR(x) #x
#define DEADWOOD_STR(x) __DEADWOOD_STR(x)

#define DEADWOOD_ASSERT(EXPR) { if (!(EXPR)) \
    throw std::runtime_error( "[deadwood] Assertion " #EXPR " failed in "\
        __FILE__ ":" DEADWOOD_STR(__LINE__) ); }
#endif




#if DEADWOOD_R
#include <Rcpp.h>
#else
#include "Python.h"
#include <cstdio>
#endif


#if DEADWOOD_R
#define DEADWOOD_PRINT(...) REprintf(__VA_ARGS__);
#else
#define DEADWOOD_PRINT(...) fprintf(stderr, __VA_ARGS__);
#endif



#ifdef DEADWOOD_PROFILER
#include <chrono>

#define DEADWOOD_PROFILER_START \
    _deadwood_profiler_t0 = std::chrono::high_resolution_clock::now();

#define DEADWOOD_PROFILER_GETDIFF  \
    _deadwood_profiler_td = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-_deadwood_profiler_t0);

#define DEADWOOD_PROFILER_USE \
    auto DEADWOOD_PROFILER_START \
    auto DEADWOOD_PROFILER_GETDIFF \
    char _deadwood_profiler_strbuf[256];

#define DEADWOOD_PROFILER_STOP(...) \
    DEADWOOD_PROFILER_GETDIFF; \
    snprintf(_deadwood_profiler_strbuf, sizeof(_deadwood_profiler_strbuf), __VA_ARGS__); \
    DEADWOOD_PRINT("%-64s: time=%12.3lf s\n", _deadwood_profiler_strbuf, _deadwood_profiler_td.count()/1000.0);

/* use like:
DEADWOOD_PROFILER_USE
DEADWOOD_PROFILER_START
DEADWOOD_PROFILER_STOP("message %d", 7)
*/
#else
#define DEADWOOD_PROFILER_START ; /* no-op */
#define DEADWOOD_PROFILER_STOP(...) ; /* no-op */
#define DEADWOOD_PROFILER_GETDIFF ; /* no-op */
#define DEADWOOD_PROFILER_USE ; /* no-op */
#endif


#if DEADWOOD_R
typedef ssize_t         Py_ssize_t;
#endif



typedef double FLOAT_T; ///< float type we are working internally with


#define IS_PLUS_INFINITY(x)  ((x) > 0.0 && !std::isfinite(x))
#define IS_MINUS_INFINITY(x) ((x) < 0.0 && !std::isfinite(x))



#ifdef OPENMP_DISABLED
    #define OPENMP_IS_ENABLED 0
    #ifdef _OPENMP
        #undef _OPENMP
    #endif
#else
    #ifdef _OPENMP
        #include <omp.h>
        #define OPENMP_IS_ENABLED 1
    #else
        #define OPENMP_IS_ENABLED 0
    #endif
#endif

#endif
