import numpy as np

def func(x,mu=0.5,sigma=0.1):
  p = 0.5*np.exp(-(x-mu)**2/sigma)
  return np.random.binomial(1,p),p

def func2(x,mu=0.5,sigma=0.1):
  p = 0.05*np.exp(-(x-mu/2)**2/(sigma/5)) + 0.1*np.exp(-(x- (mu/2 + mu)) **2/(sigma/10))
  return np.random.binomial(1,p),p

def func3(x,mu=0.5,sigma=0.00001):
  p = 0.8*np.exp(-(x-mu)**2/(sigma))
  return np.random.binomial(1,p),p
