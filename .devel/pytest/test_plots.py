import numpy as np
import quitefastmst
import time
import gc

import scipy.spatial.distance
import numpy as np

import matplotlib.pyplot as plt
import pytest
import deadwood

def test_plot():
    np.random.seed(123)

    n = 100
    X = np.random.rand(n, 2)
    deadwood.plot_scatter(X)
    deadwood.plot_scatter(X[:,0], X[:,1])
    deadwood.plot_scatter(X, labels=np.random.choice(np.arange(10), n))
    mst_d, mst_i = quitefastmst.mst_euclid(X)
    deadwood.plot_segments(mst_i, X)
    deadwood.plot_segments(mst_i, X[:,0], X[:,1])

    with pytest.raises(Exception):
        deadwood.plot_scatter(X.reshape(50,2,2))

    with pytest.raises(Exception):
        deadwood.plot_scatter(X.reshape(50,4))

    with pytest.raises(Exception):
        deadwood.plot_scatter(X, labels=np.r_[1,2])

    with pytest.raises(Exception):
        deadwood.plot_scatter(X[:,1])

    with pytest.raises(Exception):
        deadwood.plot_scatter(X[:,1], X)

    with pytest.raises(Exception):
        deadwood.plot_scatter(X, X[:,1])

    with pytest.raises(Exception):
        deadwood.plot_scatter(X[:,0], X[5:,1])

    with pytest.raises(Exception):
        deadwood.plot_segments(mst_d, X)


if __name__ == "__main__":
    test_plot()
