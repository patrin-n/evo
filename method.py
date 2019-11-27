from policy import *
import scipy.special as sp

class RLAverageRewardControl:
    def __init__(self, game_model):
        self.game_model = game_model
        self.n = game_model.n
        self.m = game_model.m
        self.qVal = np.zeros(self.m)
        self.rBar = 0
        self.preference = np.zeros(self.m)
        self.probability = np.ones(self.m) / self.m
        self.iter = 0
        self.alphaQ = 0.99
        self.alphaR = 0.99
        self.alphaP = 0.4
        self.temp = 10

    def update_policy(self, r):
        self.qVal = self.qVal + self.alphaQ * (r - self.qVal)
        self.rBar = self.rBar + self.alphaR * (np.mean(r) - self.rBar)
        self.preference = self.preference + self.alphaP * (self.qVal - self.rBar)
        return softmax(self.preference, self.temp)

    def select_action(self):
        action = np.zeros(self.n)
        for i in range(self.n):
            p_vector = np.cumsum(self.probability)
            # print(p_vector)
            r_p = np.random.rand()
            # print(np.where(r_p < p_vector))
            action[i] = np.where(r_p < p_vector)[0][0]
        return action

    def get_reward(self, actions):
        rewards = self.game_model.calculate_reward(actions)
        rewards[rewards == np.inf] = self.qVal[rewards == np.inf]
        return rewards

    def reduce_alpha(self):
        # self.alphaP = 1 / self.iter
        self.alphaQ = 1 / self.iter
        self.alphaR = 1 / self.iter

    def do(self):
        while 1:
            self.iter = self.iter + 1
            past_p = self.probability
            actions = self.select_action()
            rewards = self.get_reward(actions)
            # self.reduce_alpha()
            # print("rewards: ", rewards)
            self.probability = self.update_policy(rewards)
            if all(abs(past_p - self.probability) <= 1e-8):
                break
        print("q_values: ", self.qVal)
        print("r_bar: ", self.rBar)
        print("policy RLAverageRewardControl: ", self.probability)



class EvolutionaryGame:
    def __init__(self, game_method):
        self.n = game_method.n
        self.m = game_method.m
        self.resource = game_method.L
        self.pi = np.zeros(self.m)
        self.piBar = 0
        self.probability = np.ones(self.m) / self.m
        self.alpha = 0.3e-2
        self.iter = 0

    def calculate_pi(self):
        self.pi = np.zeros(self.m)
        for j in range(self.m):
            for k in range(1, self.n + 1):
                # print(self.pi[j], sp.comb(self.n, k, True), (-1) ** (k - 1), self.probability[j] ** (k - 1))
                self.pi[j] += sp.comb(self.n, k, True) * (-1) ** (k - 1) * self.probability[j] ** (k - 1)
                # print(self.n, k, sp.comb(self.n,k,True))
        self.pi *= self.resource / self.n

    def update_p(self):
        self.piBar = np.dot(self.probability, self.pi)
        p_dot = self.probability * (self.pi - self.piBar)
        # print(p_dot)
        self.probability += self.alpha * p_dot

    def reduce_alpha(self):
        self.alpha /= self.iter

    def do(self):
        while 1:
            self.iter = self.iter + 1
            past_p = self.probability.copy()
            # self.reduce_alpha()
            self.calculate_pi()
            self.update_p()
            if all(abs(past_p - self.probability) <= 1e-8):
                break
        print("pi: ", self.pi)
        print("piBar: ", self.piBar)
        print("policy EvolutionaryGame: ", self.probability)



