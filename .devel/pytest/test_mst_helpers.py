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
    test_unskip_indexes()
    test_skip_indexes()
    test_sort_groups()
