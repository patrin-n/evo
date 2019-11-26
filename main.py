import numpy as np
from policy import *
from model import EvoMultiPlayerMultiStrategy
from method import RLAverageRewardControl


def __main__():
    n = 30
    m = 2
    resources = np.array([10, 310])
    model = EvoMultiPlayerMultiStrategy(n, m, resources)
    learning = RLAverageRewardControl(model)
    learning.do()


__main__()
