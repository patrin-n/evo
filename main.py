import numpy as np
from policy import *
from model import EvoMultiPlayerMultiStrategy
from method import RLAverageRewardControl
from method import EvolutionaryGame


def __main__():
    n = 100
    m = 3
    resources = np.array([5, 1, 4])
    model = EvoMultiPlayerMultiStrategy(n, m, resources)
    learning = RLAverageRewardControl(model)
    learning.do()
    print(learning.iter)

    game = EvolutionaryGame(model)
    game.do()


__main__()
