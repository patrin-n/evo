import numpy as np
from policy import *
from model import EvoMultiPlayerMultiStrategy
from method import RLAverageRewardControl
from method import EvolutionaryGame


def __main__():
    n = 50
    m = 2
    resources = np.array([5, 1])
    model = EvoMultiPlayerMultiStrategy(n, m, resources)
    learning = RLAverageRewardControl(model)
    learning.do()

    game = EvolutionaryGame(model)
    game.do()

__main__()
