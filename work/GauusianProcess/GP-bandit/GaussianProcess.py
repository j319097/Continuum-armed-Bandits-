import numpy as np

class GaussianProcess:
  def __init__(self, kernel, sigma = 1e-2, output_noise = True):
    self.x_train = np.array([])
    self.y_train = np.array([])
    self.kernel = kernel
    self.sigma = sigma
    self.output_noise = output_noise

  def append(self, x_train, y_train):
    self.x_train = np.append(self.x_train, x_train)
    self.y_train = np.append(self.y_train, y_train)
  
  def kernel_matrix(self, xi, xj):
    xis, xjs = np.meshgrid(xj, xi)
    return self.kernel(xis, xjs)

  def predict(self, xs):
    K = self.kernel_matrix(self.x_train, self.x_train) \
      + self.sigma**2 * np.identity(len(self.x_train))
    K_inv = np.linalg.inv(K)

    y_mean = np.array([])
    y_var = np.array([])

    for x in xs:
      k_ = self.kernel_matrix(self.x_train, x)
      k__ = self.kernel_matrix(x, x)

      k_K = k_.T.dot(K_inv)
      y_mean = np.append(y_mean, k_K.dot(self.y_train))
      y_var = np.append(y_var, k__ - k_K.dot(k_) + self.sigma**2 * self.output_noise)
    return y_mean, y_var
