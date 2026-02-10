# This file is part of the deadwood package for R.

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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


#' @title Deadwood: Outlier Detection via Trimming of Mutual Reachability Minimum Spanning Trees
#'
#' @description
#' Deadwood is an anomaly detection algorithm based on Mutual Reachability
#' Minimum Spanning Trees.  It trims protruding tree segments and marks small
#' debris as outliers.
#'
#' More precisely, the use of a mutual reachability distance
#' pulls peripheral points farther away from each other.
#' Tree edges with weights beyond the detected elbow point
#' are removed. All the resulting connected components whose
#' sizes are smaller than a given threshold are deemed anomalous.
#'
#'
#' @details
#' As with all distance-based methods (this includes k-means and DBSCAN as well),
#' applying data preprocessing and feature engineering techniques
#' (e.g., feature scaling, feature selection, dimensionality reduction)
#' might lead to more meaningful results.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link{mst}} will be called to compute an MST, which
#' generally takes at most \eqn{O(n^2)} time. However, by default,
#' for low-dimensional Euclidean spaces, a faster algorithm based on K-d trees
#' is selected automatically; see \code{\link[quitefastmst]{mst_euclid}} from
#' the \pkg{quitefastmst} package.
#'
#' Once the spanning tree is determined (\eqn{\Omega(n \log n)}-\eqn{O(n^2)}),
#' the Deadwood algorithm runs in \eqn{O(n)} time.
#' Memory use is also \eqn{O(n)}.
#'
#'
#' @references
#' M. Gagolewski, deadwood, in preparation, 2026, TODO
#'
#' V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan, Finding a "Kneedle"
#' in a haystack: Detecting knee points in system behavior,
#' In: 31st Intl. Conf. Distributed Computing Systems Workshops,
#' 2011, 166-171, \doi{10.1109/ICDCSW.2011.20}
#'
#' R.J.G.B. Campello, D. Moulavi, J. Sander,
#' Density-based clustering based on hierarchical density estimates,
#' Lecture Notes in Computer Science 7819, 2013, 160-172,
#' \doi{10.1007/978-3-642-37456-2_14}
#'
#'
#' @param d a numeric matrix with \eqn{n} rows and \eqn{p} columns
#'     (or an object coercible to one, e.g., a data frame with numeric-like
#'     columns), an object of class \code{dist} (see \code{\link[stats]{dist}}),
#'     an object of class \code{mstclust} (see \pkg{genieclust}
#'     and \pkg{lumbermark}),
#'     or an object of class \code{mst} (see \code{\link{mst}})
#'
#' @param distance metric used in the case where \code{d} is a matrix; one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}
#'
#' @param M smoothing factor; \eqn{M \leq 1} gives the selected \code{distance};
#'     otherwise, the corresponding mutual reachability distance is used
#'
#' @param contamination single numeric value or \code{NA};
#'     the estimated (approximate) proportion of outliers in the data set;
#'     if \code{NA}, the contamination amount will be determined
#'     by identifying the most significant elbow point of the curve
#'     comprised of increasingly ordered tree edge weights
#'
#' @param max_contamination single numeric value or \code{NA};
#'    maximal contamination level assumed when \code{contamination} is \code{NA}
#'
#' @param ema_dt single numeric value or \code{NA};
#'    controls the smoothing parameter \eqn{\alpha = 1-\exp(-dt)}
#'    of the exponential moving average (in elbow detection),
#'    \eqn{y_i = \alpha x_i + (1-\alpha) y_{i-1}}, \eqn{y_1 = x_1}
#'
#' @param max_debris_size single integer value or \code{NA};
#'     the maximal size of the leftover connected components that
#'     will be considered outliers; if \code{NA}, \eqn{\sqrt{n}} is assumed
#'
#' @param cut_edges numeric vector or \code{NULL};
#'     \eqn{k-1} indexes of the tree edges whose omission lead to
#'     \eqn{k} connected components (clusters), where the outliers are to
#'     be sought independently; most frequently this is generated
#'     via \pkg{genieclust} or \pkg{lumbermark}.
#'
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#'
#' @param ... further arguments passed to \code{\link{mst}}
#'
#'
#' @return
#' A logical vector of length \eqn{n}, where TRUE denotes outliers.
#'
#' The \code{mst} attribute gives the computed minimum
#' spanning tree which can be reused in further calls to the functions
#' from \pkg{genieclust}, \pkg{lumbermark}, and \pkg{deadwood}.
#' \code{cut_edges} gives the \code{cut_edges} passed as argument.
#' \code{contamination} gives the detected contamination levels
#' in each cluster (which can be different from the observed proportion
#' of outliers detected).
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- jitter(as.matrix(iris[1:2]))  # some data
#' is_outlier <- deadwood(X, M=5)
#' plot(X, col=c("#ff000066", "#55555555")[is_outlier+1],
#'     pch=c(16, 1)[is_outlier+1], asp=1, las=1)
#'
#' @rdname deadwood
#' @export
deadwood <- function(d, ...)
{
    UseMethod("deadwood")
}


#' @export
#' @rdname deadwood
#' @method deadwood default
deadwood.default <- function(
    d,
    M=5L,  # TODO: set default
    contamination=NA_real_,
    max_debris_size=NA_real_,
    max_contamination=0.5,
    ema_dt=0.01,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    verbose=FALSE,
    ...
) {
    distance <- match.arg(distance)
    tree <- mst(d, M=M, distance=distance, verbose=verbose, ...)
    deadwood.mst(
        tree,
        contamination=contamination,
        max_debris_size=max_debris_size,
        max_contamination=max_contamination,
        ema_dt=ema_dt,
        verbose=verbose
    )
}


#' @export
#' @rdname deadwood
#' @method deadwood dist
deadwood.dist <- function(
    d,
    M=5L,  # TODO: set default
    contamination=NA_real_,
    max_debris_size=NA_real_,
    max_contamination=0.5,
    ema_dt=0.01,
    verbose=FALSE,
    ...
) {
    deadwood.mst(
        mst(d, M=M, verbose=verbose, ...),
        contamination=contamination,
        max_debris_size=max_debris_size,
        max_contamination=max_contamination,
        ema_dt=ema_dt,
        verbose=verbose
    )
}


#' @export
#' @rdname deadwood
#' @method deadwood mstclust
deadwood.mstclust <- function(
    d,
    contamination=NA_real_,
    max_debris_size=NA_real_,
    max_contamination=0.5,
    ema_dt=0.01,
    verbose=FALSE,
    ...
) {
    tree <- attr(d, "mst")
    stopifnot(!is.null(tree))

    cut_edges <- attr(d, "cut_edges")
    stopifnot(!is.null(cut_edges))

    deadwood.mst(
        tree,
        contamination=contamination,
        max_debris_size=max_debris_size,
        max_contamination=max_contamination,
        ema_dt=ema_dt,
        cut_edges=cut_edges,
        verbose=verbose
    )
}


#' @export
#' @rdname deadwood
#' @method deadwood mst
deadwood.mst <- function(
    d,
    contamination=NA_real_,
    max_debris_size=NA_real_,
    max_contamination=0.5,
    ema_dt=0.01,
    cut_edges=NULL,
    verbose=FALSE,
    ...
) {
    verbose <- !identical(verbose, FALSE)
    contamination <- as.double(contamination)[1]
    max_debris_size <- as.integer(max_debris_size)[1]
    max_contamination <- as.double(max_contamination)[1]
    ema_dt <- as.double(ema_dt)[1]
    cut_edges <- as.numeric(cut_edges)   # numeric(0) if NULL

    stopifnot(ema_dt > 0.0)

    n <- NROW(d)+1
    if (is.na(max_debris_size))
        max_debris_size <- as.integer(sqrt(n))
    max_debris_size <- max(1L, max_debris_size)

    stopifnot(max_contamination >= 0, max_contamination <= 1)
    if (!is.na(contamination)) {
        stopifnot(contamination >= 0, contamination <= 1)
        max_contamination <- -contamination
    }

    is_outlier <- .deadwood(
        d, cut_edges, max_contamination, ema_dt, max_debris_size, verbose
    )

    stopifnot(attr(is_outlier, "contamination") >= 0)
    stopifnot(attr(is_outlier, "contamination") <= 1)

    structure(
        is_outlier,
        names=attr(d, "Labels"),
        mst=d,
        cut_edges=cut_edges,
        class="mstoutlier"
    )
}

registerS3method("deadwood", "default",   "deadwood.default")
registerS3method("deadwood", "dist",      "deadwood.dist")
# registerS3method("deadwood", "msthclust", "deadwood.msthclust")  TODO
registerS3method("deadwood", "mstclust",  "deadwood.mstclust")
registerS3method("deadwood", "mst",       "deadwood.mst")
