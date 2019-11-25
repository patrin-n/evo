import numpy as np


def softmax(q_val, temp):
    return np.exp(q_val / temp) / np.sum(np.exp(q_val / temp))


def linear(q_val):
    return q_val / sum(q_val)
