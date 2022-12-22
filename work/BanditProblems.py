import numpy as np
import matplotlib.pyplot as plt

# gauusian process
class GP():
    def __init__(self, kernel, sigma = 1e-2, y_mean = 0.5):
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.kernel = kernel
        self.sigma = sigma
        self.y_mean = y_mean

    def append(self, x_train, y_train):
        self.x_train = np.append(self.x_train, x_train)
        self.y_train = np.append(self.y_train, y_train)
  
    def kernel_matrix(self, xi, xj):
        xis, xjs = np.meshgrid(xj, xi)
        return self.kernel(xis, xjs)

    def predict(self, xs):
        K = self.kernel_matrix(self.x_train, self.x_train) + self.sigma**2 * np.identity(len(self.x_train))
        K_inv = np.linalg.inv(K)

        y_mean = np.array([])
        y_var = np.array([])

        for x in xs:
            k_ = self.kernel_matrix(self.x_train, x)
            k__ = self.kernel_matrix(x, x)

            k_K = k_.T.dot(K_inv)
            y_mean = np.append(y_mean, self.y_mean+k_K.dot(self.y_train))
            y_var = np.append(y_var, k__ - k_K.dot(k_))# + self.sigma**2)
        return y_mean, y_var
    
    def mean(self, x):
        K = self.kernel_matrix(self.x_train, self.x_train) + self.sigma**2 * np.identity(len(self.x_train))
        K_inv = np.linalg.inv(K)        
        k_ = self.kernel_matrix(self.x_train, x)
        k__ = self.kernel_matrix(x, x)
        k_K = k_.T.dot(K_inv)
        y_mean = k_K.dot(self.y_train)
          
        return -y_mean
        
    
# thompson sampling
class TS():

    def __init__(self, arms_number, puls_arm_number = 1):
        self.arms_number = arms_number
        self.puls_arm_number = puls_arm_number 
        self.counts = [0 for _ in range(arms_number)]
        self.wins = [0 for _ in range(arms_number)]
  
    def get_arm(self):
        beta = lambda N, a: np.random.beta(a + 1, N - a + 1)
        result = [beta(self.counts[i], self.wins[i]) for i in range(self.arms_number)]
        arm  = result.index(max(result))
        return arm
  
    def sample(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        self.wins[arm] = self.wins[arm] + reward 
        puls_arm_number = self.puls_arm_number 
        
        if arm >= puls_arm_number:
            for i in range(puls_arm_number):
                self.wins[arm -i] = self.wins[arm] + reward
                self.counts[arm -i] = self.counts[arm] + 1
            
        if arm <= (self.arms_number - puls_arm_number):
            for i in range(puls_arm_number):
                self.wins[arm +i] = self.wins[arm] + reward
                self.counts[arm +i] = self.counts[arm] + 1
    
#use kernel in gauusian process
class rbf:
    def __init__(self, theta):
        self.theta = theta
    
    def __call__(self, xi=0, xj=0):
        return self.theta[0] * np.exp((-1 * (xi - xj)**2)/self.theta[1])
    
# train data function
def train_one_peak(x,mu=0.5,sigma=0.1):
    p = 0.5*np.exp(-(x-mu)**2/sigma)
    return np.random.binomial(1,p),p

def train_two_peak(x,mu=0.5,sigma=0.1):
    p = 0.05*np.exp(-(x-mu/2)**2/(sigma/5)) + 0.1*np.exp(-(x- (mu/2 + mu)) **2/(sigma/10))
    return np.random.binomial(1,p),p

def func3(x,mu=0.5,sigma=0.00001):
    p = 0.8*np.exp(-(x-mu)**2/(sigma))
    return np.random.binomial(1,p),p
    
    
## Bundit Algorithm
def gp_ts(y_mean, y_std):
    return [np.argmax(np.random.normal(y_mean,y_std))]

def gp_ucb(y_mean, y_std):
    return [np.argmax(y_mean+y_std)]

# Bandit Play Simulation
def sim_GP(bandit, number_of_action, train_func, x, GauusinProcess, is_gp_plot = False, sigma = 0.001, kernel_sigma = [1,1]):
    y_train = np.array([])
    x_train = np.array([])
    for step in range(number_of_action):
        if step == 0:
            x_t = np.random.random(1)
            y_t = np.array([train_func(x_t)[0]])
        else:
            x_t = x[bandit(y_mean,y_std)]
            y_t = np.array([train_func(x_t)[0]])

        x_train = np.append(x_train,x_t)
        y_train = np.append(y_train,y_t)

        GP = GauusinProcess(kernel = rbf([kernel_sigma[0],kernel_sigma[1]]),sigma = sigma)
        GP.append(x_train, y_train)
        y_mean, y_var = GP.predict(x)
        y_std = np.sqrt(np.maximum(y_var, 0))
        
        if step == number_of_action -1 and is_gp_plot:
            gp_plot(GP, y_mean, y_std,step+1, train_func, x)
    return y_train, x_train

def sim_GP_UCB(bandit, number_of_action, train_func, x, GauusinProcess, is_gp_plot = False):
    y_train = np.array([])
    x_train = np.array([])
    for step in range(number_of_action):
        if step == 0:
            x_t = np.random.random(1)
            y_t = np.array([train_func(x_t)[0]])
        else:
            x_t = x[bandit(y_mean,y_std)]
            y_t = np.array([train_func(x_t)[0]])

        x_train = np.append(x_train,x_t)
        y_train = np.append(y_train,y_t)

        GP = GauusinProcess(kernel = rbf([1,1]),sigma = 0.025)
        GP.append(x_train, y_train)
        y_mean, y_var = GP.predict(x)
        y_std = np.sqrt(np.maximum(y_var, 0))
        
        if step == number_of_action -1 and is_gp_plot:
            gp_plot(GP, y_mean, y_std,step+1, train_func, x)
    return y_train, x_train, GP

def sim_TS(number_of_action, train_func, x, TS, puls_arm_number = 1):
    y_train = np.array([])
    x_train = np.array([])
    ts = TS(len(x), puls_arm_number)
    for step in range(number_of_action):
        if step == 0:
            x_t = np.random.randint(len(x))
            y_t = np.array([train_func(x_t/len(x))[0]])
        else:
            x_t = ts.get_arm()
            y_t = np.array([train_func(x_t/len(x))[0]])

        x_train = np.append(x_train,x_t/len(x))
        y_train = np.append(y_train,y_t)
        ts.sample(int(x_t), int(y_t))
    return int(sum(y_train)), x_train

# Result data plot
def plot_transition_action(x_train, number_of_action, train_area, titel = 'Transition of Action', interval = 1000):
    plt.figure(figsize=(5,3))
    plt.scatter(range(len(x_train)),x_train, s = 5)
    
    for t in train_area:
        plt.axhline(t['mean'],color='red')
        plt.fill_between(np.linspace(0,interval,interval),[t['min'] for i in range(interval)],[t['max'] for i in range(interval)],alpha=0.3)
    plt.xlim(0,number_of_action)
    plt.ylim(0,1)
    plt.grid()
    plt.title(titel)
    plt.xlabel('Trial')
    plt.ylabel('Action')
    plt.show()

def gp_plot(GP, y_mean, y_std,num_data, train, x):
    y = train(x)[1]
    plt.plot(x, y, label='true', linestyle='dashed', color='black')
    plt.plot(x, y_mean, label='mean', color='#0066cc')
    plt.fill_between(x, y_mean-y_std, y_mean+y_std, label='2σ credible region', color='#0066cc', alpha=0.3)
    plt.scatter(GP.x_train, GP.y_train, label='data', color='black', marker='x', s=50)
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.title(f'Number of data = {num_data}')
    plt.show()
    
def plot_sum_reward(y_train):
    sum_train_array = []
    sum_train = 0
    for i in y_train:
        sum_train += i
        sum_train_array.append(sum_train)
    plt.title(f'total reward = {sum_train}')
    plt.xlabel('Trial')
    plt.ylabel('sum reward')
    plt.plot(sum_train_array)
    plt.grid()
    plt.show()
    
def plot_train_func(x, y, x_lim=1, y_lim=1):
    plt.figure(figsize=(3,2))
    plt.plot(x, y, label='true', linestyle='dashed', color='black')
    plt.xlim(0,x_lim)
    plt.ylim(0,y_lim)
    plt.show()
    