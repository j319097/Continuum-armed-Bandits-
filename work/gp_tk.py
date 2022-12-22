import numpy as np


class RBF:
    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta
        
    def __call__(self, x1, x2):
        return self._alpha * np.exp(-np.square(x1 - x2) / self._beta)


class GP:
    def __init__(self, mu_y, sigma, kernel):
        self._mu_y = mu_y
        self._sigma = sigma
        self._kernel = kernel
        self._x = np.zeros(0)
        self._y = np.zeros(0)
        self._k = np.zeros((0, 0))
        self._invk = np.zeros((0, 0))
    
    def append(self, x, y):
        n = self._y.size
        k0 = self._kernel(x, x)
        k1 = self._kernel(self._x, x)
        k2 = self._k
        self._k = np.zeros((n + 1, n + 1))
        self._k[:n, :n] = k2
        self._k[n, :n] = k1
        self._k[:n, n] = k1
        self._k[n, n] = k0
        self._x = np.append(self._x, x)
        self._y = np.append(self._y, y)
        self._invk = np.linalg.inv(self._k + self._sigma * np.eye(n + 1))

    def mean_var(self, x):
        k0 = self._kernel(x, x)
        k1 = self._kernel(x, self._x[:, None])
        dy = self._y - self._mu_y
        mean = self._mu_y + k1.T @ self._invk @ dy
        var = k0 - np.diag(k1.T @ self._invk @ k1)
        return mean, var
