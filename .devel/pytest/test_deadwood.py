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


def test_deadwood_single():
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


def test_deadwood_multi2():
    try:
        import matplotlib.pyplot as plt
        import genieclust  # TODO
        import lumbermark  # TODO
    except:
        return False

    np.random.seed(123)
    n1 = 140
    m1 = 19
    r1 = np.random.rand(n1)
    u1 = np.random.rand(n1)*2*np.pi
    t1 = np.linspace(0, 2*np.pi, m1+1)[1:]

    n2 = 15
    m2 = 5
    r2 = np.random.rand(n2)
    u2 = np.random.rand(n2)*2*np.pi
    t2 = np.linspace(0, 2*np.pi, m2+1)[1:]
    X = np.vstack((
        r1.reshape(-1,1)*np.c_[np.cos(u1), np.sin(u1)],
        4*np.c_[np.cos(t1), np.sin(t1)],
        r2.reshape(-1,1)*np.c_[np.cos(u2), np.sin(u2)]+np.r_[8, 8],
        4*np.c_[np.cos(t2), np.sin(t2)]+np.r_[8, 8],
    ))
    y_true = np.repeat([1, -1, 1, -1], [n1, m1, n2, m2])
    # genieclust.plots.plot_scatter(X, labels=y_true, asp=1)
    # plt.show()

    L = lumbermark.Lumbermark(2).fit(X)
    y = L.labels_
    # genieclust.plots.plot_scatter(X, labels=y, asp=1)
    # plt.show()

    D = deadwood.Deadwood().fit(L)
    y = D.labels_
    print(D.contamination_)
    assert np.abs(D.contamination_[0] - m1/(n1+m1))<1e-6
    assert np.abs(D.contamination_[1] - m2/(n2+m2))<1e-6
    assert np.all(y == y_true)
    # genieclust.plots.plot_scatter(X, labels=y, asp=1)
    # plt.show()


def test_deadwood_multi1():
    try:
        import matplotlib.pyplot as plt
        import genieclust  # TODO
        import lumbermark  # TODO
    except:
        return False

    np.random.seed(1234)
    n1, n2 = 1000, 250
    X = np.vstack((
        np.random.rand(n1, 2),
        np.random.rand(n2, 2)+[1.2, 0]
    ))
    D = deadwood.Deadwood(M=25).fit(X)
    y = D.labels_
    print(D.contamination_)
    # genieclust.plots.plot_scatter(X, labels=y, asp=1)
    # plt.show()

    L = lumbermark.Lumbermark(2, M=25).fit(X)
    y = L.labels_
    #genieclust.plots.plot_scatter(X, labels=y, asp=1)
    #plt.show()

    D = deadwood.Deadwood()
    o = D.fit_predict(L)
    print(D.contamination_)
    #w = o.copy(); w[w>0] = y[w>0]
    # genieclust.plots.plot_scatter(X, labels=o, asp=1)
    # plt.show()
    assert (o[:n1]<0).mean() > 0.1
    assert (o[n1:]<0).mean() > 0.1


    G = genieclust.Genie(2, gini_threshold=0.5, M=25).fit(X)
    y = G.labels_
    #genieclust.plots.plot_scatter(X, labels=y, asp=1, asp=1)
    #plt.show()

    D = deadwood.Deadwood()
    o = D.fit_predict(G)
    print(D.contamination_)
    #w = o.copy(); w[w>0] = y[w>0]
    # genieclust.plots.plot_scatter(X, labels=o, asp=1)
    # plt.show()
    assert (o[:n1]<0).mean() > 0.1
    assert (o[n1:]<0).mean() > 0.1


def test_index_unskip():
    assert np.all(deadwood.index_unskip(np.r_[0, 1, 2], np.r_[False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.index_unskip(np.r_[0, 1, 2], np.r_[True, False, False, False]) == np.r_[1, 2, 3])
    assert np.all(deadwood.index_unskip(np.r_[0, 2, 1], np.r_[True, False, True, False, False]) == np.r_[1, 4, 3])


def test_index_skip():
    assert np.all(deadwood.index_skip(np.r_[0, 1, 2], np.r_[False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.index_skip(np.r_[1, 2, 3], np.r_[True, False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.index_skip(np.r_[1, 4, 3], np.r_[True, False, True, False, False]) == np.r_[0, 2, 1])
    assert np.all(deadwood.index_skip(np.r_[0, 1, 4, 3, 2, 0], np.r_[True, False, True, False, False]) == np.r_[-1, 0, 2, 1, -1, -1])


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
    test_deadwood_multi1()
    test_deadwood_multi2()
    test_index_unskip()
    test_index_skip()
    test_sort_groups()
