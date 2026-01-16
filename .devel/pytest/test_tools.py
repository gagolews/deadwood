import numpy as np
import time
import deadwood


def test_argkmin():
    for n in [1, 5, 1000]:
        for k in [k for k in [0, 1, 2, 4, 10, 25, 50, 100] if k < n]:
            x = np.arange(n)
            t0 = time.time()
            y1a = deadwood.argkmin(x, k)
            print("(ascending)  n=%10d, k=%4d: %10.3fs" % (n, k, time.time()-t0), end="\t")
            t0 = time.time()
            y2a = np.argsort(x, kind='mergesort')[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1a == y2a

            x = np.arange(n,0,-1)
            t0 = time.time()
            y1b = deadwood.argkmin(x, k)
            print("(descending)                       %10.3fs" % (time.time()-t0,), end="\t")
            t0 = time.time()
            y2b = np.argsort(x, kind='mergesort')[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1b == y2b

            x = np.round(np.random.rand(n), 5)
            t0 = time.time()
            y1c = deadwood.argkmin(x, k)
            print("(random)                           %10.3fs" % (time.time()-t0,), end="\t")
            t0 = time.time()
            y2c = np.argsort(x, kind='mergesort')[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1c == y2c


def test_cummin():
    for n in [1, 5, 1000]:
        x = np.random.rand(n)
        cx = deadwood.cummin(x)
        for i in range(n):
            assert cx[i] == min(x[:(i+1)])

        x = np.sort(x)
        cx = deadwood.cummin(x)
        for i in range(n):
            assert cx[i] == min(x[:(i+1)])

        x = x[::-1]
        cx = deadwood.cummin(x)
        for i in range(n):
            assert cx[i] == min(x[:(i+1)])


def test_cummax():
    for n in [1, 5, 1000]:
        x = np.random.rand(n)
        cx = deadwood.cummax(x)
        for i in range(n):
            assert cx[i] == max(x[:(i+1)])

        x = np.sort(x)
        cx = deadwood.cummax(x)
        for i in range(n):
            assert cx[i] == max(x[:(i+1)])

        x = x[::-1]
        cx = deadwood.cummax(x)
        for i in range(n):
            assert cx[i] == max(x[:(i+1)])



if __name__ == "__main__":
    print("\n\nargsort\n")
    test_argsort()
    print("\n\nargkmin\n")
    test_argkmin()
    print("\n\ncummin\n")
    test_cummin()
    print("\n\ncummax\n")
    test_cummax()

