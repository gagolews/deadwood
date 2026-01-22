import numpy as np
import scipy.spatial.distance
import time
import gc
import deadwood
import sys
import sklearn.datasets
import pandas as pd

# TODO: graph_vertex_degrees
# TODO: graph_vertex_incidences
# TODO: mst_label_imputer
# TODO: mst_cluster_sizes



def test_deadwood_base_classes():
    np.random.seed(123)
    X = np.random.rand(100, 2)
    D = deadwood.MSTClusterer(n_clusters=2, verbose=True)

    D._check_params()
    D._get_mst(X)
    sum_w1 = D._tree_d_.sum()

    print("Not recomputing... ", file=sys.stderr, end="")
    D.M = 0
    D._check_params()
    D._get_mst(X)  # not recomputing! See outputs
    sum_w2 = D._tree_d_.sum()
    assert sum_w1 == sum_w2
    print("[by visual inspection of messages only!] OK", file=sys.stderr)

    D.M = 1
    D._check_params()
    D._get_mst(X)
    sum_w3 = D._tree_d_.sum()
    assert sum_w1 == sum_w3

    D = deadwood.MSTClusterer(n_clusters=2, verbose=True, metric="Manhattan")
    D._check_params()
    D._get_mst(X)


    # D.metric = "COSINE"
    # D.M = 10
    # D._check_params()
    # D._get_mst(X)
    # sum_w1 = D._tree_d_.sum()
    #
    # D.metric = "Euclidean"   !!! __buf[w] = sqrt(2.0*(1.0-__buf[w]));
    # X2 = X/np.sqrt(np.sum(X**2,axis=1)).reshape(-1,1)
    # D._check_params()
    # D._get_mst(X2)
    # sum_w2 = (D._tree_d_).sum()
    #
    # assert np.abs(sum_w1-sum_w2)<1e-9


def test_deadwood_df():
    X, y_true = sklearn.datasets.load_iris(return_X_y=True)
    D = deadwood.MSTClusterer(n_clusters=2, verbose=True)
    D._check_params()
    D._get_mst(X)
    sum_w1 = D._tree_d_.sum()

    X = pd.DataFrame(X)
    D._check_params()
    D._get_mst(X)
    sum_w2 = D._tree_d_.sum()
    assert sum_w1 == sum_w2


def test_deadwood_outliers():
    np.random.seed(123)
    n = 140
    m = 19
    r = np.random.rand(n)
    u = np.random.rand(n)*2*np.pi
    t = np.linspace(0, 2*np.pi, m+1)[1:]
    X = np.vstack((
        r.reshape(-1,1)*np.c_[np.cos(u), np.sin(u)],
        5*np.c_[np.cos(t), np.sin(t)]
    ))
    y_true = np.repeat([1, -1], [n, m])

    D = deadwood.Deadwood(contamination=0)
    y_pred = D.fit_predict(X)
    # print(y_pred)
    assert D.contamination_ == 0.0
    assert np.all(y_pred == np.repeat(1, X.shape[0]))

    D = deadwood.Deadwood(contamination=m/(n+m)).fit(X)
    y_pred = D.labels_
    # print(y_pred)
    assert D.contamination_ == m/(n+m)
    assert np.all(y_pred == y_true)

    D = deadwood.Deadwood()
    D._ema_dt = 0.01
    D._max_contamination = 0.37
    y_pred = D.fit_predict(X)
    # print(y_pred)
    assert np.abs(D.contamination_ - m/(n+m))<1e-6
    assert np.all(y_pred == y_true)


def test_index_unskip():
    assert np.all(deadwood.index_unskip(np.r_[0, 1, 2], np.r_[False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.index_unskip(np.r_[0, 1, 2], np.r_[True, False, False, False]) == np.r_[1, 2, 3])
    assert np.all(deadwood.index_unskip(np.r_[0, 2, 1], np.r_[True, False, True, False, False]) == np.r_[1, 4, 3])


def test_index_skip():
    assert np.all(deadwood.index_skip(np.r_[0, 1, 2], np.r_[False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.index_skip(np.r_[1, 2, 3], np.r_[True, False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.index_skip(np.r_[1, 4, 3], np.r_[True, False, True, False, False]) == np.r_[0, 2, 1])
    assert np.all(deadwood.index_skip(np.r_[0, 1, 4, 3, 2, 0], np.r_[True, False, True, False, False]) == np.r_[-1, 0, 2, 1, -1, -1])


if __name__ == "__main__":
    test_deadwood_base_classes()
    test_deadwood_df()
    test_deadwood_outliers()
    test_index_unskip()
    test_index_skip()
