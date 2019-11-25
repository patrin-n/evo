from policy import *


class AverageRewardControl:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.qVal = np.zeros(self.m)
        self.rBar = 0
        self.preference = np.zeros(self.m)
        self.p = np.zeros(self.m)
        self.alphaQ = 0.9
        self.alphaR = 0.9
        self.alphaP = 0.9
        self.temp = 10

    def update_policy(self, r):
        self.qVal = self.qVal + self.alphaQ * (r - self.qVal)
        self.rBar = self.rBar + self.alphaR * (r - self.rBar)
        self.preference = self.preference + self.alphaP * (self.qVal - self.rBar)
        return softmax(self.preference, self.temp)

    def select_action(self):
        p_vector = np.cumsum(self.p)
        r_p = np.random.rand()
        return np.where(r_p < p_vector)[0][0]

    def get_reward(self):
        return calculate_reward(a)

    def do(self):
        while(1):
            a = self.select_action()
            r = self.get_reward(a)
            self.p = self.update_policy(r)
            print("q_values: ", self.qVal)
            print("r_bar: ", self.rBar)
            print("policy: ", self.p)

