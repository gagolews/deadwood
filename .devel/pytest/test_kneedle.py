import numpy as np
import scipy.spatial.distance
import time
import gc
import deadwood


def test_kneedle():
    assert 0 == deadwood.kneedle_increasing(np.r_[1.0,1.0,1.0,1.0,1.0])

    n = 1001
    x = np.linspace(0.1, 1, n)
    y = (-1.0/x+5)
    assert deadwood.kneedle_increasing(y, convex=False, dt=100) == 240
    y = y.max()-y[::-1]
    assert deadwood.kneedle_increasing(y, convex=True, dt=100) == 760



if __name__ == "__main__":
    test_kneedle()
