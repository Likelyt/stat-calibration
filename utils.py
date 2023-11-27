import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_result(result, path):
    with open(path, 'w') as fp:
        json.dump(result, fp)


def env(configuration, conf, std, n, true_theta, sample, seed):
    # generate noise
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    noise = torch.normal(mean=0, std=std, size=(1,n)).view(-1)
    y = torch.zeros(n)
    delta_value = torch.zeros(n)
    eta_value = torch.zeros(n)
    
    if configuration == "conf_0":
        # generate x in torch tensor
        if sample == 'train':
            x = torch.zeros(n)
            for i in range(n):
                x[i] = 2 * torch.pi * i / (n-1)
            
            for i in range(n):
                delta_value[i] = conf.delta(x[i], true_theta)
            eta_value[i] = conf.eta(x[i], true_theta)
            y[i] = conf.gen_y(x[i], true_theta) + noise[i]
        elif sample == 'test':
            x = torch.zeros(n)
            # uniform sampling from [0,2pi]
            for i in range(n):  
                x[i] = 2 * torch.pi * torch.rand(1)
            
            for i in range(n):
                delta_value[i] = conf.delta(x[i], true_theta)
            eta_value[i] = conf.eta(x[i], true_theta)
            y[i] = conf.gen_y(x[i], true_theta)
            
    elif configuration == "conf_1":
        # uniform sampling from [0,1]  
        if sample == 'train':     
            x = torch.rand(n)
            for i in range(n):
                delta_value[i] = conf.delta(x[i])
                eta_value[i] = conf.eta(x[i], true_theta)
                y[i] = conf.gen_y(x[i], true_theta) + noise[i]
        elif sample == 'test':
            torch.manual_seed(seed + 1)
            x = torch.rand(n)
            for i in range(n):
                delta_value[i] = conf.delta(x[i])
                eta_value[i] = conf.eta(x[i], true_theta)
                y[i] = conf.gen_y(x[i], true_theta)
            
    elif configuration == "conf_2" or configuration == "conf_3" or configuration == "conf_4":
        if sample == 'train': 
            x = torch.rand((n,2))
            for i in range(n):
                delta_value[i] = conf.delta(x[i, ])
                eta_value[i] = conf.eta(x[i, ], true_theta)
                y[i] = conf.gen_y(x[i, ], true_theta) + noise[i]
            
        elif sample == 'test':
            torch.manual_seed(seed + 1)
            x = torch.rand((n,2))
            for i in range(n):
                delta_value[i] = conf.delta(x[i, ])
                eta_value[i] = conf.eta(x[i, ], true_theta)
                y[i] = conf.gen_y(x[i, ], true_theta)
                
    elif configuration == "conf_5":
        if sample == 'train':
            x = torch.rand(n)
            for i in range(n):
                delta_value[i] = conf.delta(x[i], true_theta)
                eta_value[i] = conf.eta(x[i], true_theta)
                y[i] = conf.gen_y(x[i], true_theta) + noise[i]
        elif sample == 'test':
            torch.manual_seed(seed + 1)
            x = torch.rand(n)
            for i in range(n):
                delta_value[i] = conf.delta(x[i], true_theta)
                eta_value[i] = conf.eta(x[i], true_theta)
                y[i] = conf.gen_y(x[i], true_theta)
        #plt.scatter(x, y)
        #plt.show()
    return x, y, delta_value, eta_value




class CONF_0:
    def __init__(self, theta):
        self.theta = theta
        self.d = 1
        
    def eta(self, x, theta):
        self.eta_value = self.gen_y(x, theta) - self.delta(x, theta)
        return self.eta_value
    
    def delta(self, x, theta):
        self.delta_value = torch.sqrt(torch.square(theta) - theta + 1) * \
                (torch.sin(theta * x) + torch.cos(theta * x))
                
        return self.delta_value

    def gen_y(self, x, theta):
        self.y_value = torch.exp(x/10) * torch.sin(x)
        return self.y_value
    
class CONF_1:
    def __init__(self, theta):
        self.theta = theta
        self.d = len(self.theta)
    
    def eta(self, x, theta):
        threshold_first_dim = torch.tensor([0.001, 0.25])
        threshold_second_dim = torch.tensor([0.001, 0.5])
        clamped_theta = torch.tensor([
            torch.clamp(theta[0], threshold_first_dim[0], threshold_first_dim[1]),
            torch.clamp(theta[1], threshold_second_dim[0], threshold_second_dim[1])
        ])
        theta.data = clamped_theta
        part_1 = 7 * torch.square(torch.sin(2 * torch.pi * theta[0] - torch.pi))
        part_2 = 2 * torch.sin(2 * torch.pi * x - torch.pi) * torch.square(2 * torch.pi * theta[1] - torch.pi)
        self.eta_value = part_1 + part_2
        return self.eta_value

    def delta(self, x):
        self.delta_value = torch.cos(2 * torch.pi * x - torch.pi)
        return self.delta_value
    
    def gen_y(self, x, theta):
        self.y_value = self.eta(x, theta) + self.delta(x)
        return self.y_value
    
class CONF_2:
    def __init__(self, theta):
        self.theta = theta
        self.d = len(self.theta)
    
    def eta(self, x, theta):
        threshold_dim = torch.tensor([[0.001, 0.5],[0.001, 1.0], [0.001, 2.0]])
        clamped_theta = torch.tensor([
            torch.clamp(theta[0], threshold_dim[0,0], threshold_dim[0,1]),
            torch.clamp(theta[1], threshold_dim[1,0], threshold_dim[1,1]),
            torch.clamp(theta[2], threshold_dim[2,0], threshold_dim[2,1])
        ])
        theta.data = clamped_theta
        part_1 = 7 * torch.square(torch.sin(2 * torch.pi * theta[0] - torch.pi))
        part_2 = 2 * torch.sin(2 * torch.pi * x[0] - torch.pi) * torch.square(2 * torch.pi * theta[1] - torch.pi)
        part_3 = 6 * theta[2] * (x[1] - 0.5)
        self.eta_value = part_1 + part_2 + part_3
        return self.eta_value
    
    def delta(self, x):
        part_1 = torch.cos(2 * torch.pi * x[0] - torch.pi)
        part_2 = 2 * (torch.square(x[1]) - x[1] + torch.tensor(1/6))
        self.delta_value = part_1 + part_2
        return self.delta_value
    
    def gen_y(self, x, theta):
        self.y_value = self.eta(x, theta) + self.delta(x)
        return self.y_value
    
class CONF_3:
    def __init__(self, theta):
        self.theta = theta
        self.d = len(self.theta)
        
    def eta(self, x, theta):
        threshold_dim = torch.tensor([[0.0, 1.0],[0.0, 1.0]])
        clamped_theta = torch.tensor([
            torch.clamp(theta[0], threshold_dim[0,0], threshold_dim[0,1]),
            torch.clamp(theta[1], threshold_dim[1,0], threshold_dim[1,1])
        ])
        theta.data = clamped_theta
        part_1 = 2/3 * torch.exp(x[0]+theta[0])
        part_2 = - x[1] * torch.sin(theta[1])
        part_3 = theta[1]
        self.eta_value = part_1 + part_2 + part_3
        return self.eta_value
    
    def delta(self, x):
        part_1 = torch.exp(-x[0])
        part_2 = x[0] - 1/2
        part_3 = torch.square(x[1]) - x[1] + 1/6
        self.delta_value = part_1 * part_2 * part_3
        return self.delta_value
    
    def gen_y(self, x, theta):
        self.y_value = self.eta(x, theta) + self.delta(x)
        return self.y_value
    

class CONF_4:
    def __init__(self, theta):
        self.theta = theta
        self.d = len(self.theta)
        
    def eta(self, x, theta):
        threshold_dim = torch.tensor([[0.01, 1.0],[0.01, 1.0]])
        clamped_theta = torch.tensor([
            torch.clamp(theta[0], threshold_dim[0,0], threshold_dim[0,1]),
            torch.clamp(theta[1], threshold_dim[1,0], threshold_dim[1,1])
        ])
        theta.data = clamped_theta
        part_1 = 1/2 * theta[0] * (torch.sqrt(1 + (theta[1] + torch.square(x[0]))*x[1]/torch.square(theta[0])) - 1)
        part_2 = (theta[0] + 3 * x[1]) * torch.exp(1 + torch.sin(x[0]))
        self.eta_value = part_1 + part_2
        if torch.isnan(self.eta_value):
            print(torch.sqrt(1 + (theta[1] + torch.square(x[0]))*x[1]/torch.square(theta[0])) - 1)
            print("eta_value is nan")
            print("theta: ", theta)
            print("x: ", x)
            print("part_1: ", part_1)
            print("part_2: ", part_2)
        return self.eta_value
    
    def delta(self, x):
        self.delta_value = torch.tensor(0)
        return self.delta_value
    
    def gen_y(self, x, theta):
        self.y_value = self.eta(x, theta) + self.delta(x)
        return self.y_value
        
class CONF_5:
    def __init__(self, theta):
        self.theta = theta
        self.d = 1
    
    def eta(self, x, theta):
        self.eta_value = self.gen_y(x, theta) - self.delta(x, theta)
        return self.eta_value
    
    def delta(self, x, theta):
        self.delta_value = torch.sqrt(torch.square(theta) - theta + 1) * \
                (torch.sin(2 * torch.pi * theta * x) + torch.cos(2 * torch.pi * theta * x))
                
        return self.delta_value     
    def gen_y(self, x, theta):
        self.y_value = torch.exp(x * torch.pi /5) * torch.sin(2 * torch.pi * x)
        return self.y_value

def env_real(x, y):
    n = len(x)
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x, y

class CONF_Heart:
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        self.d = 3
    
    def A(self, theta):
        t1 = theta[0]
        t2 = theta[1]
        t3 = theta[2]
        ans = torch.tensor([[-t2 - t3, t1, 0, 0],
                            [t2, -t1 - t2, t1, 0],
                            [0, t2, -t1 - t2, t1],
                            [0, 0, t2, -t1]], requires_grad=True)
        return ans

    def eta(self, x, theta):
        e1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        e4 = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.eta_value = torch.zeros(len(x))
        for i in range(len(x)):
            self.eta_value[i] = torch.dot(torch.matmul(e1, torch.linalg.matrix_exp(self.A(theta) * torch.exp(x[i]))), e4)
        return self.eta_value.double() 

    
    
def data_switch(x, y, y_s, d_x, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(y)
    train1_index = np.random.choice(n, int(n/2), replace=False)
    train1_index.sort()
    train2_index = np.setdiff1d(np.arange(n), train1_index)

    if d_x == 1:
        train1_x = x[train1_index]
        train2_x = x[train2_index]
    else:
        train1_x = x[train1_index, ]
        train2_x = x[train2_index, ]
        
    train1_y = y[train1_index]
    train1_y_s = y_s[train1_index]
    train2_y = y[train2_index]
    train2_y_s = y_s[train2_index]
    
    train_A = {}
    train_A['train1_x'] = train1_x
    train_A['train1_y'] = train1_y
    train_A['train1_y_s'] = train1_y_s
    train_A['train2_x'] = train2_x
    train_A['train2_y'] = train2_y
    train_A['train2_y_s'] = train2_y_s
    
    # Switch the row of train 1 and train 2
    train_B = {}
    train_B['train1_x'] = train2_x
    train_B['train1_y'] = train2_y
    train_B['train1_y_s'] = train2_y_s
    train_B['train2_x'] = train1_x
    train_B['train2_y'] = train1_y
    train_B['train2_y_s'] = train1_y_s
    
    return train_A, train_B

def data_switch_real(x, y, d_x, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(y)
    train1_index = np.random.choice(n, int(n/2), replace=False)
    train1_index.sort()
    train2_index = np.setdiff1d(np.arange(n), train1_index)

    if d_x == 1:
        train1_x = x[train1_index]
        train2_x = x[train2_index]
    else:
        train1_x = x[train1_index, ]
        train2_x = x[train2_index, ]
        
    train1_y = y[train1_index]
    train2_y = y[train2_index]
    
    train_A = {}
    train_A['train1_x'] = train1_x
    train_A['train1_y'] = train1_y
    train_A['train2_x'] = train2_x
    train_A['train2_y'] = train2_y
    
    # Switch the row of train 1 and train 2
    train_B = {}
    train_B['train1_x'] = train2_x
    train_B['train1_y'] = train2_y
    train_B['train2_x'] = train1_x
    train_B['train2_y'] = train1_y
    
    return train_A, train_B


class Mydataset(Dataset):
    def __init__(self, x, y, z, w):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.z = torch.Tensor(z)
        self.w = torch.Tensor(w)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        a = self.x[idx]
        b = self.y[idx]
        c = self.z[idx]
        d = self.w[idx]
        return a,b,c,d
    
class Mydataset_real(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        a = self.x[idx]
        b = self.y[idx]
        return a,b,c,d
    
    
def Best_iter(data, stds, configurations, n_trains=[50]):
    data_new = {}
    for config in configurations:
        for std in stds:
            for n_train in n_trains:
                data_loop = np.array(data['%s_std_%.2f_sample_%d'% (config, std, n_train)])
                min_value = np.min(data_loop, axis=1)
                data_new['%s_std_%.2f_sample_%d'%(config, std, n_train)] = min_value
    return data_new

def Table_Res(data, stds, configs, n_trains=[50]):
    for config in configs:
        for std in stds:
            for n_train in n_trains:
                test_best_result = data['%s_std_%.2f_sample_%d'% (config, std, n_train)]
                mean_test = np.mean(test_best_result)
                std_test = np.std(test_best_result)
                logger.info('{}, sample {}, var {}; Mean PMSE {}, sd {}'.format(config, n_train, np.power(std,2), mean_test, std_test))
                final_result = {}
                final_result['mean'] = mean_test
                final_result['std'] = std_test
                save_result(final_result, 'result/final_result_configs_%s_var_%s.json'%(config, np.power(std,2)))