import numpy as np

def softmax(self,q_val):
  return exp(q_val / temp) / np.sum(exp(q_val / temp))
