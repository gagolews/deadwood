import numpy as np
import sklearn.neighbors
import scipy.spatial.distance
import time
import gc
import quitefastmst
import scipy.spatial
import deadwood
import deadwood.oldmst


def knn_oldmst(X, k, Y, metric):
    if Y is not None: return None
    return deadwood.oldmst.knn_from_distance(X, k=k, metric=metric)

def knn_ref(X, k, Y, metric):
    if Y is None:  # doesn't work correctly otherwise... omits self
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=metric)
        nn.fit(X)
        return nn.kneighbors(Y)
    else:
        if metric != "euclidean": return None
        t = scipy.spatial.KDTree(X)
        d, i = t.query(Y, k)
        d = np.array(d).reshape(-1, k)
        i = np.array(i).reshape(-1, k)
        return d, i

def knn_fastmst_brute(X, k, Y, metric):
    if metric != "euclidean": return None
    return quitefastmst.knn_euclid(X, k=k, Y=Y, algorithm="brute")


def knn_fastmst_kdtree(X, k, Y, metric):
    if metric != "euclidean": return None
    if X.shape[1] > 20: return None
    return quitefastmst.knn_euclid(X, k=k, Y=Y, algorithm="kd_tree")


knn_algos=dict(
    knn_ref=knn_ref,
    knn_oldmst=knn_oldmst,
    knn_fastmst_kdtree=knn_fastmst_kdtree,
    knn_fastmst_brute=knn_fastmst_brute,
)


def knn_check(X, metric, Y=None):
    n = X.shape[0]
    d = X.shape[1]

    for n_neighbors in [1, 2, 5, 10, n-1]:

        allres = []
        for algo_name, algo_fun in knn_algos.items():
            t0 = time.time()
            thisres = algo_fun(X, n_neighbors, Y, metric=metric)
            t1 = time.time()
            if thisres is None: continue

            print("        %20s(%5d) %10.3fs" % (algo_name, n_neighbors, t1-t0,))

            allres.append(thisres)

            assert thisres[0].shape == allres[0][0].shape
            assert thisres[1].shape == allres[0][1].shape
            assert np.allclose(thisres[0], allres[0][0])
            assert np.all(thisres[1] == allres[0][1])


    return True



def test_knn():
    np.random.seed(234)
    for X in [np.random.randn(1000,5)]:
        print("------- Y=None")
        knn_check(X, metric="euclidean", Y=None)
        print("------- Y=X[:1,:]")
        knn_check(X, metric="euclidean", Y=X[:1,:])
        print("------- Y=X[:10,:]")
        knn_check(X, metric="euclidean", Y=X[:10,:])
        print("------- Y=X")
        knn_check(X, metric="euclidean", Y=X)

        print("------- cityblock")
        knn_check(X, metric='cityblock')
        print("------- cosine")
        knn_check(X, metric='cosine')
        gc.collect()

if __name__ == "__main__":
    test_knn()
