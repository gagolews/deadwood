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


def _blob_and_aureola(n, m, seed=123):
    if seed is not None:
        np.random.seed(seed)
    r = np.random.rand(n)
    u = np.random.rand(n)*2*np.pi
    t = np.linspace(0, 2*np.pi, m+1)[1:]
    X = np.vstack((
        r.reshape(-1,1)*np.c_[np.cos(u), np.sin(u)],
        5*np.c_[np.cos(t), np.sin(t)]
    ))
    y = np.repeat([1, -1], [n, m])
    return X, y


def test_deadwood_single():
    n, m = 140, 19
    X, y_true = _blob_and_aureola(n, m)

    D = deadwood.Deadwood(contamination=m/(n+m)).fit(X)
    y_pred = D.labels_
    # print(y_pred)
    assert D.contamination_ == m/(n+m)
    assert np.all(y_pred == y_true)

    D = deadwood.Deadwood(contamination=0)
    y_pred = D.fit_predict(X)
    # print(y_pred)
    assert D.contamination_ == 0.0
    assert np.all(y_pred == np.repeat(1, X.shape[0]))

    D = deadwood.Deadwood()
    D._ema_dt = 0.01
    D._max_contamination = 0.37
    y_pred = D.fit_predict(X)
    # print(y_pred)
    assert np.abs(D.contamination_ - m/(n+m))<1e-6
    assert np.all(y_pred == y_true)


def test_deadwood_multi():
    # more tests in genieclust and deadwood

    # two blobs with outlier aureole
    n1, m1 = 140, 19
    X1, y_true1 = _blob_and_aureola(n1, m1)

    n2, m2 = 15, 5
    X2, y_true2 = _blob_and_aureola(n2, m2)

    X = np.vstack((X1, X2+11))
    y_true = np.r_[y_true1, y_true2]

    # deadwood.plot_scatter(X, labels=y_true, asp=1)
    # deadwood.plt.show()

    D = deadwood.Deadwood(M=10)
    D._cut_edges_ = np.r_[X.shape[0]-2]
    D.fit(X)
    y = D.labels_
    print(D.contamination_)
    # import matplotlib.pyplot as plt
    # deadwood.plot_scatter(X, labels=y, asp=1)
    # deadwood.plt.show()
    assert np.abs(D.contamination_[0] - m1/(n1+m1))<1e-6
    assert np.abs(D.contamination_[1] - m2/(n2+m2))<1e-6
    assert np.all(y == y_true)


def test_unskip_indexes():
    assert np.all(deadwood.unskip_indexes(np.r_[0, 1, 2], np.r_[False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.unskip_indexes(np.r_[0, 1, 2], np.r_[True, False, False, False]) == np.r_[1, 2, 3])
    assert np.all(deadwood.unskip_indexes(np.r_[0, 2, 1], np.r_[True, False, True, False, False]) == np.r_[1, 4, 3])


def test_skip_indexes():
    assert np.all(deadwood.skip_indexes(np.r_[0, 1, 2], np.r_[False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.skip_indexes(np.r_[1, 2, 3], np.r_[True, False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.skip_indexes(np.r_[1, 4, 3], np.r_[True, False, True, False, False]) == np.r_[0, 2, 1])
    assert np.all(deadwood.skip_indexes(np.r_[0, 1, 4, 3, 2, 0], np.r_[True, False, True, False, False]) == np.r_[-1, 0, 2, 1, -1, -1])


def test_sort_groups():
    y, ind = deadwood.sort_groups(np.r_[0,1,2], np.r_[0,0,0], 1)
    assert np.all(y == np.r_[0,1,2]) and np.all(ind == np.r_[0, 3])

    y, ind = deadwood.sort_groups(np.r_[0,1,2], np.r_[0,1,0], 3)
    assert np.all(y == np.r_[0,2,1]) and np.all(ind == np.r_[0, 2, 3, 3])

    y, ind = deadwood.sort_groups(np.r_[0,1,2], np.r_[0,1,-1], 3)
    assert np.all(y == np.r_[2,0,1]) and np.all(ind == np.r_[1, 2, 3, 3])


if __name__ == "__main__":
    test_deadwood_base_classes()
    test_deadwood_df()
    test_deadwood_single()
    test_deadwood_multi()
    test_unskip_indexes()
    test_skip_indexes()
    test_sort_groups()
