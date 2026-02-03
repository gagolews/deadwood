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
#' sizes are smaller than a given threshold are considered outliers.
#'
#'
#' @details
#' As in the case of all the distance-based methods (including
#' k-nearest neighbours and DBSCAN), the standardisation of the input features
#' is definitely worth giving a try.  Oftentimes, applying feature selection
#' and engineering techniques (e.g., dimensionality reduction) might lead
#' to more meaningful results.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link[deadwood]{mst}()} will be called to compute an MST, which
#' generally takes at most \eqn{O(n^2)} time. However, by default, a faster
#' algorithm based on K-d trees is selected automatically for low-dimensional
#' Euclidean spaces; see \code{\link[quitefastmst]{mst_euclid}} from
#' the \pkg{quitefastmst} package.
#'
#' Once a minimum spanning tree is determined, the Deadwood algorithm runs in
#' \eqn{O(TODO)} time.
#'
#'
#' @seealso
#' \code{\link[quitefastmst]{mst_euclid}}
#'
#' @references
#' M. Gagolewski, deadwood, in preparation, 2026, TODO
#'
#'
#' @param d a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist} (see \code{\link[stats]{dist}}),
#'     or an object of class \code{mst} (see \code{\link[deadwood]{mst}})
#'
#' @param distance metric used in the case where \code{d} is a matrix; one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}
#'
#' @param M smoothing factor; \eqn{M \leq 1} gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used
#'
#' @param contamination
#'
#' @param max_contamination
#'
#' @param ema_dt
#'
#' @param max_debris_size
#'
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#'
#' @param ... further arguments passed to \code{\link[deadwood]{mst}()}
#'
#'
#' @return
#' TODO
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- jitter(as.matrix(iris[1:2]))  # some data
#' # TODO
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
    M=0L, #TODO: set default
    clusters=NULL,
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
        clusters=clusters,
        max_contamination=max_contamination,
        ema_dt=ema_dt,
        max_debris_size=max_debris_size,
        verbose=verbose
    )
}


#' @export
#' @rdname deadwood
#' @method deadwood dist
deadwood.dist <- function(
    d,
    M=0L, #TODO: set default
    clusters=NULL,
    contamination=NA_real_,
    max_debris_size=NA_real_,
    max_contamination=0.5,
    ema_dt=0.01,
    verbose=FALSE,
    ...
) {
    deadwood.mst(
        mst(d, M=M, verbose=verbose, ...),
        cut_edges=numeric(0),
        max_contamination=max_contamination,
        ema_dt=ema_dt,
        max_debris_size=max_debris_size,
        verbose=verbose
    )
}


#' @export
#' @rdname deadwood
#' @method deadwood mst
deadwood.mst <- function(
    d,
    M=0L, #TODO: set default
    clusters=NULL,
    contamination=NA_real_,
    max_debris_size=NA_real_,
    max_contamination=0.5,
    ema_dt=0.01,
    verbose=FALSE,
    ...
) {
    verbose <- !identical(verbose, FALSE)
    max_debris_size <- as.integer(max_debris_size)[1]
    contamination <- as.double(contamination)[1]
    max_contamination <- as.double(max_contamination)[1]
    ema_dt <- as.double(ema_dt)[1]
    clusters <- as.integer(clusters)

    result <- .deadwood(
        d, clusters, max_contamination, ema_dt, max_debris_size, verbose
    )

#     result[["height"]] <- .correct_height(result[["height"]])
#     result[["labels"]] <- attr(d, "Labels") # yes, >L<abels
#     result[["method"]] <- sprintf("Genie(%g)", gini_threshold)
#     result[["call"]]   <- match.call()
#     result[["dist.method"]] <- attr(d, "method")
#     class(result) <- "hclust"

    result
}

registerS3method("deadwood", "default", "deadwood.default")
registerS3method("deadwood", "dist",    "deadwood.dist")
registerS3method("deadwood", "mst",     "deadwood.mst")
