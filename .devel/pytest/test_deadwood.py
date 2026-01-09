import numpy as np
import scipy.spatial.distance
import time
import gc
import deadwood


# TODO: graph_vertex_degrees
# TODO: graph_vertex_incidences
# TODO: mst_label_imputer
# TODO: mst_cluster_sizes



def test_deadwood():
    pass  # TODO



def test_index_unskip():
    assert np.all(deadwood.index_unskip(np.r_[0, 1, 2], np.r_[False, False, False]) == np.r_[0, 1, 2])
    assert np.all(deadwood.index_unskip(np.r_[0, 1, 2], np.r_[True, False, False, False]) == np.r_[1, 2, 3])
    assert np.all(deadwood.index_unskip(np.r_[0, 2, 1], np.r_[True, False, True, False, False]) == np.r_[1, 4, 3])


if __name__ == "__main__":
    test_deadwood()
    test_index_unskip()
