import numpy as np

class rbf:
  def __init__(self, theta):
    self.theta = theta
    
  def __call__(self, xi=0, xj=0):
    return self.theta[0] * np.exp((-1 * (xi - xj)**2)/self.theta[1])
