import matplotlib.pyplot as plt
import numpy as np

def plot():
    plt.figure(figsize=(10,8))
    plt.scatter(range(len(x_train)),x_train)
    plt.axhline(0.25,color='red')
    plt.axhline(0.75,color='red')
    plt.fill_between(np.linspace(0,1000,1000),[0.19 for i in range(1000)],[0.31 for i in range(1000)],alpha=0.5)
    plt.fill_between(np.linspace(0,1000,1000),[0.69 for i in range(1000)],[0.81 for i in range(1000)],alpha=0.5)
    plt.xlim(0,1000)
    plt.ylim(0,1)
    plt.grid()
    plt.title('Transition of Action')
    plt.xlabel('Trial')
    plt.ylabel('Action')
    plt.show()
    
def gp_plot(GP, y_mean, y_std,num_data, train):
  x = np.linspace(0, 1, 1000)
  y = train(x)[1]
  plt.figure(figsize=(12, 6))
  plt.plot(x, y, label='true', linestyle='dashed', color='black')
  plt.plot(x, y_mean, label='mean', color='#0066cc')
  plt.fill_between(x, y_mean-y_std, y_mean+y_std, label='2σ credible region', color='#0066cc', alpha=0.3)
  plt.scatter(GP.x_train, GP.y_train, label='data', color='black', marker='x', s=200)
  plt.xlim(0,1)
  plt.ylim(0,1)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid()
  plt.legend()
  plt.title(f'Number of data = {num_data}')
  plt.show()