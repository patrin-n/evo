import numpy as np
from policy import *


def __main__():
    print("It is a test")
    print("salam")
    x = np.max([1, 2])
    print(x)
    q = np.array([10, 10, 5])
    print(softmax(q, 50))
    print(linear(q))


__main__()
