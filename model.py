import numpy as np


class EvoMultiPlayerMultiStrategy:
    def __init__(self, n, m, L):
        self.n = n
        self.m = m
        self.L = L

    def calculate_reward(self, action):
        action_num = np.zeros(self.m)
        for i in range(self.m):
            action_num[i] = sum(action == i)
        return self.L / action_num
