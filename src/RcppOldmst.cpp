/*  Jarník (Prim)'s MST algorithm for complete undirected graphs
 *  (the "old"/generic interface; see quitefastmst for a faster
 *  implementation).
 *
 *  Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>
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
#include "c_oldmst.h"
#include <cmath>

using namespace Rcpp;


/**
 * Represents a matrix as a C-contiguous array,
 * i.e., in a row-major order.
 */
template <typename T> class CMatrix {
private:
    size_t n, d;
    std::vector<T> elems;

public:
    /** Initialises a new matrix of size _nrow*_ncol, filled with 0s
     *
     * @param _nrow
     * @param _ncol
     */
    CMatrix(size_t _nrow, size_t _ncol)
        : n(_nrow), d(_ncol), elems(_nrow*_ncol)
    {
        ;
    }

    /** Initialises a new matrix of size _nrow*_ncol, filled with _ts
     *
     * @param _nrow
     * @param _ncol
     * @param _t
     */
    CMatrix(size_t _nrow, size_t _ncol, T _t)
        : n(_nrow), d(_ncol), elems(_nrow*_ncol, _t)
    {
        ;
    }


    /** Initialises a new matrix of size _nrow*_ncol based on a contiguous
     * C- or Fortran-style array
     *
     * @param _data
     * @param _nrow
     * @param _ncol
     * @param _c_order whether the first _ncol elements in _data constitute the first row
     * or the first _nrow elements define the first column
     */
    template<class S> CMatrix(const S* _data, size_t _nrow, size_t _ncol, bool _c_order)
        : n(_nrow), d(_ncol), elems(_nrow*_ncol)
    {
        if (_c_order) {
            for (size_t i=0; i<_nrow*_ncol; ++i)
                elems[i] = (T)(_data[i]);
        }
        else {
            size_t k = 0;
            for (size_t i=0; i<_nrow; i++) {
                for (size_t j=0; j<_ncol; j++) {
                    elems[k++] = (T)_data[i+_nrow*j];
                }
            }
        }
    }


    /** Read/write access to an element in the i-th row and the j-th column
     *
     * @param i
     * @param j
     * @return a reference to the indicated matrix element
     */
    inline T& operator()(const size_t i, const size_t j) {
        return elems[d*i + j];
    }

    inline const T& operator()(const size_t i, const size_t j) const {
        return elems[d*i + j];
    }


    /** Returns a direct pointer to the underlying C-contiguous data array:
     * the first ncol elements give the 1st row,
     * the next ncol element give the 2nd row,
     * and so forth.
     *
     * @return pointer
     */
    T* data() {
        return elems.data();
    }

    const T* data() const {
        return elems.data();
    }


    /** Returns a direct pointer to the start of the i-th row
     *
     * @param i
     * @return pointer
     */
    T* row(const size_t i) {
        return elems.data()+i*d;
    }

    const T* row(const size_t i) const {
        return elems.data()+i*d;
    }


    /** Returns the number of rows
     *
     * @return
     */
    size_t nrow() const {
        return n;
    }


    /** Returns the number of columns
     *
     * @return
     */
    size_t ncol() const {
        return d;
    }
};


template<typename T>
NumericMatrix internal_oldmst_compute(
    CDistance<T>* D, Py_ssize_t n, Py_ssize_t M, bool verbose
) {
    NumericMatrix ret(n-1, 3);

    CDistance<T>* D2 = NULL;
    if (M >= 1) {
        // TODO we need it for M==1 as well, but this data can be read from the
        // MST data below!
        if (verbose) DEADWOOD_PRINT("[deadwood] Determining the core distance.\n");

        CMatrix<Py_ssize_t> nn_i(n, M);
        CMatrix<T> nn_d(n, M);
        Cknn_from_complete(D, n, M, nn_d.data(), nn_i.data());

        IntegerMatrix out_nn_ind(n, M);
        NumericMatrix out_nn_dist(n, M);

        std::vector<T> d_core(n);
        for (Py_ssize_t i=0; i<n; ++i) {
            d_core[i] = nn_d(i, M-1); // distance to the M-th nearest neighbour
            DEADWOOD_ASSERT(std::isfinite(d_core[i]));

            for (Py_ssize_t j=0; j<M; ++j) {
                DEADWOOD_ASSERT(nn_i(i,j) != i);
                out_nn_ind(i,j)  = nn_i(i,j)+1; // 1-based indexing
                out_nn_dist(i,j) = nn_d(i,j);
            }
        }

        ret.attr("nn.index") = out_nn_ind;
        ret.attr("nn.dist")  = out_nn_dist;

        D2 = new CDistanceMutualReachability<T>(d_core.data(), n, D);
    }

    CMatrix<Py_ssize_t> mst_i(n-1, 2);
    std::vector<T>  mst_d(n-1);

    if (verbose) DEADWOOD_PRINT("[deadwood] Computing the MST.\n");
    Cmst_from_complete<T>(D2?D2:D, n, mst_d.data(), mst_i.data(), verbose);
    if (verbose) DEADWOOD_PRINT("[deadwood] Done.\n");

    if (D2) delete D2;

    for (Py_ssize_t i=0; i<n-1; ++i) {
        DEADWOOD_ASSERT(mst_i(i,0) < mst_i(i,1));
        DEADWOOD_ASSERT(std::isfinite(mst_d[i]));
        ret(i,0) = mst_i(i,0)+1;  // R-based indexing
        ret(i,1) = mst_i(i,1)+1;  // R-based indexing
        ret(i,2) = mst_d[i];
    }

    return ret;
}


template<typename T>
NumericMatrix internal_oldmst_matrix(
    NumericMatrix X,
    String distance,
    Py_ssize_t M,
    /*bool use_mlpack, */
    bool verbose
) {
    Py_ssize_t n = X.nrow();
    Py_ssize_t d = X.ncol();
    NumericMatrix ret;

    if (M < 0 || M >= n)
        stop("`M` must be an integer in [0, n-1]");

    CMatrix<T> X2(REAL(SEXP(X)), n, d, false); // Fortran- to C-contiguous

    T* _X2 = X2.data();
    for (Py_ssize_t i=0; i<n*d; i++) {
        if (!std::isfinite(_X2[i]))
            Rf_error("All elements in the input matrix must be finite and non-missing.");
    }


    CDistance<T>* D = NULL;
    if (distance == "euclidean" || distance == "l2")
       D = (CDistance<T>*)(new CDistanceEuclideanSquared<T>(X2.data(), n, d));
    else if (distance == "manhattan" || distance == "cityblock" || distance == "l1")
        D = (CDistance<T>*)(new CDistanceManhattan<T>(X2.data(), n, d));
    else if (distance == "cosine")
        D = (CDistance<T>*)(new CDistanceCosine<T>(X2.data(), n, d));
    else
        stop("given `distance` is not supported (yet)");

    ret = internal_oldmst_compute<T>(D, n, M, verbose);
    delete D;

    if (distance == "euclidean" || distance == "l2") {
        for (Py_ssize_t i=0; i<n-1; ++i) {
            ret(i,2) = sqrt(ret(i,2));
        }

        if (M > 0) {
            Rcpp::NumericMatrix out_nn_dist = ret.attr("nn.dist");
            for (Py_ssize_t i=0; i<n; ++i) {
                for (Py_ssize_t j=0; j<M; ++j) {
                    out_nn_dist(i,j) = sqrt(out_nn_dist(i,j));
                }
            }
        }
    }

    return ret;
}


// [[Rcpp::export(".oldmst.matrix")]]
NumericMatrix dot_oldmst_matrix(
    NumericMatrix X,
    String distance="euclidean",
    int M=0,
    bool cast_float32=false,
    bool verbose=false
) {
    if (cast_float32)
        return internal_oldmst_matrix<float >(X, distance, M, verbose);
    else
        return internal_oldmst_matrix<double>(X, distance, M, verbose);
}


// [[Rcpp::export(".oldmst.dist")]]
NumericMatrix dot_oldmst_dist(
    NumericVector d,
    int M=0,
    bool verbose=false
) {
    Py_ssize_t n = (Py_ssize_t)round((sqrt(1.0+8.0*d.size())+1.0)/2.0);
    DEADWOOD_ASSERT(n*(n-1)/2 == d.size());

    CDistancePrecomputedVector<double> D(REAL(SEXP(d)), n);

    return internal_oldmst_compute<double>(&D, n, M, verbose);
}
