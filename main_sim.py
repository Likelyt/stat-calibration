from utils import *
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from tqdm.notebook import tqdm
import sklearn.model_selection
import sklearn.kernel_ridge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import Matern,ExpSineSquared
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model_1:
    def __init__(self, conf, init_theta, d_x, d_theta, lr, epoch, loss_space, stop_criterion = 1e-5):
        self.init_theta = init_theta
        self.lr = lr
        self.conf = conf
        self.epoch = epoch
        self.d_theta = d_theta
        self.d_x = d_x
        self.stop_criterion = stop_criterion
        self.theta_list = torch.zeros((self.epoch, self.d_theta))
        self.opt_loss = np.inf
        self.decay = 0.5
        self.gamma_cur = torch.zeros(1)
        self.loss_space = loss_space
        
    def train(self, train_x, train_y, t, theta, model_2=None, gamma=None):
        # define the optimizer with stochastic gradient descent
        # initialize theta
        gap = 5
        if t == 0:
            theta = self.init_theta.clone().detach().requires_grad_(True)
        else:
            #print('Model 2 Start from {}'.format(theta))
            theta = theta.clone().detach().requires_grad_(True)
        
        # define the optimizer
        optimizer = torch.optim.SGD([theta], lr=self.lr, momentum=0.9, weight_decay=0.5, dampening=0.5) # nesterov=True 
        #optimizer = torch.optim.Adam([theta], lr=self.lr, weight_decay=0.5, amsgrad=True, betas=(0.9, 0.999), eps=1e-08)
        # define the loss function
        loss_func = nn.MSELoss()
        
        #store the loss and theta
        loss_list = []
        lrs = []
        
        if t > 0:
            self.n_train = len(train_y)
            _, K = model_2.pred(train_x, 'train')
            K = K + self.n_train * model_2.opt_alpha * np.eye(self.n_train)
            self.K = torch.from_numpy(K)
            #print(self.K)
            if t == 1:
                self.K_dyn = self.K
            else:
                #self.K_dyn = (self.K_dyn + self.K)/(t)
                self.K_dyn = self.K
                
        # train the model        
        for i in range(self.epoch):
            # forward pass
            eta_est = self.y_s_fun(train_x, theta)            
            if self.loss_space == 'RKHS':
                if t == 0:
                    loss = loss_func(eta_est, train_y)
                else:
                    # train_y: y - delta_hat
                    # eta_est: estimated eta(x, theta_hat)
                    train_y = train_y
                    loss = torch.matmul(torch.matmul((train_y - eta_est).view(1, self.n_train), torch.linalg.inv(self.K_dyn).float()), (train_y - eta_est).view(self.n_train, 1)) / self.n_train
            elif self.loss_space == 'L2':
                loss = loss_func(eta_est, train_y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            
            loss_list.append(loss.item())
            self.theta_list[i,] = theta.detach()
            
            # set stop criterion, if the loss moving average (100) change is smaller than 1e-2 percent, stop
            if (i == self.epoch - 1) or i > 2 * gap and abs((np.mean(loss_list[i-gap:i]) - np.mean(loss_list[i - 2* gap:i-gap]))/np.mean(loss_list[i- 2* gap:i-gap])) < self.stop_criterion:
                logger.info('Model 1 stopped after: {} iterations, Model 1 Loss: {}'.format(i, loss.item()))
                #logger.info('Model 1 Loss: {}'.format(loss.item()))
                break 
            
            # print the loss
            # if (i+1) % 5 == 0:
            #     logger.info('epoch: {}, theta: {}, loss: {}'.format(i, theta.detach(), loss.item()))
            #if t > 0 and i % 400 == 0:
            #    logger.info('epoch: %d, theta: {}, loss: {} \n  pred_y_s: {}, pred_delta: {}'.format(i, theta.item(), loss.item(), y_s_hat[0], delta_pred[0], ))

        # select the theta for the minmum loss
        self.opt_loss = np.min(loss_list)
        opt_theta_index = np.argmin(loss_list)
        self.theta_opt = self.theta_list[opt_theta_index]
        logger.info('Model 1 Loss: {}, Theta: {}'.format(self.opt_loss, self.theta_opt))
        
        return self.opt_loss, self.theta_list, self.theta_opt

    def y_s_fun(self, x, theta):
        if self.d_x == 1:
            y_eta_est = self.conf.eta(x, theta)
            return y_eta_est
        else:
            n = len(x)
            y_eta_est = torch.zeros(n)   
            for i in range(n):
                y_eta_est[i] = self.conf.eta(x[i,], theta)
            return y_eta_est
        

    def pred(self, train_x, train_y):
        if self.d_x == 1:
            y_2_eta_hat = self.conf.eta(train_x, self.theta_opt.clone().detach()) # eta 
            y_2_error = train_y - y_2_eta_hat # delta hat
        else:
            n = len(train_x)
            y_2_eta_hat = torch.zeros(n)
            y_2_error = torch.zeros(n)
            for i in range(n):
                y_2_eta_hat[i] = self.conf.eta(train_x[i, ], self.theta_opt.clone().detach())
                y_2_error[i] = train_y[i] - y_2_eta_hat[i]
        return y_2_error
    
    
    
# Define your custom kernel function
def matern_kernel(x, y, gamma=1):
    # check the dimension of the x and y
    # d = x.shape[0]
    # if d == 1:
    #     distance = (1 + np.abs(x - y) / gamma) * np.exp(-np.abs(x - y) / gamma)
    # else:
    #     distance = (1 + np.linalg.norm(x - y,2) / gamma) * np.exp(-np.linalg.norm(x - y,2) / gamma)
    return (1 + np.linalg.norm(x - y,2) / gamma) * np.exp(-np.linalg.norm(x - y,2) / gamma)


# Define a function to compute the Gram matrix
def GramMatrix(X, gamma):
    n_samples, d = X.shape
    K = np.zeros((n_samples, n_samples))
    # if X is a 1 dimensional tensor
    if d == 1:
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = matern_kernel(X[i], X[j], gamma)
    # if X is a d dimensional tensor
    else:
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = matern_kernel(X[i,], X[j,], gamma)
    return K

# Create a custom kernel-based KernelRidge estimator
class CustomKernelRidge(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1, alpha=1):
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X, y):
        self.X_train = X
        gram_matrix = GramMatrix(X, self.gamma)
        n_samples = X.shape[0]
        self.alpha_ = np.linalg.solve(gram_matrix + self.alpha * np.eye(n_samples), y)

    def predict(self, X):
        K_x = np.zeros((X.shape[0], self.X_train.shape[0]))
        # this part can be optimized to accelerate by matrix multiplication to calcaulte the kernel matrix
        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                K_x[i, j] = matern_kernel(X[i], self.X_train[j], self.gamma) 
        return np.dot(K_x, self.alpha_)


class Model_2:
    def __init__(self, d_x, d_theta, kernel_method = 'matern'): # default is matern, rbf
        self.d_x = d_x
        self.d_theta = d_theta
        self.kernel_method = kernel_method
        self.count = 0
        self.opt_gamma = 0
        self.best_model = []
    
    def delta_estimate(self, x, y):
        # x is the point to be estimated
        # x_sample_2 is the sample2 data
        # y_error is the sample2 data's residual
        # return the estimated sample2 data's residual
        
        # use kernel ridge regression to estimate the delta
        if self.kernel_method == 'matern':
            kernel_gpml = Matern(length_scale=1, nu=0.5)
            estimator = sklearn.kernel_ridge.KernelRidge(kernel=kernel_gpml)
            #print(estimator.get_params().keys())
            #estimator = CustomKernelRidge(gamma=1.0, alpha=1)
            hyper_space = dict(kernel__length_scale=np.logspace(-2, 1, 25), 
                               alpha=np.logspace(0, 2, 50))
        elif self.kernel_method == 'rbf' or 'laplacian' or 'polynomial':
            estimator = sklearn.kernel_ridge.KernelRidge(kernel=self.kernel_method)
            hyper_space = dict(alpha=np.logspace(0, 5, 50), 
                               gamma=np.logspace(-2, 2, 25))
        elif self.kernel_method == 'ExpSineSquared':
            #https://stackoverflow.com/questions/58938819/optimise-custom-gaussian-processes-kernel-in-scikit-using-gridsearch
            kernel_gpml = ExpSineSquared(length_scale=0.5, periodicity=1.0)
            estimator = sklearn.kernel_ridge.KernelRidge(kernel=kernel_gpml)
            #print(estimator.get_params().keys())
            hyper_space = dict(alpha=np.logspace(1, 2, 10),
                               #kernel__length_scale=np.logspace(0, 1, 10), 
                               kernel__periodicity=np.logspace(-10, 3, 10, base=2))
            
        gscv = GridSearchCV(
            estimator=estimator,
            param_grid=hyper_space,
            cv = 5,
            scoring='neg_mean_squared_error',
            refit=True,
            verbose=0
        )
        
        self.curr_model = gscv.fit(x.reshape(-1, self.d_x), y.reshape(-1, 1))
        self.best_model.append(self.curr_model)
        if self.kernel_method == 'ExpSineSquared':
            cur_gamma = gscv.best_params_['kernel__periodicity']
            cur_alpha = gscv.best_params_['alpha']
        if self.kernel_method == 'matern':
            cur_gamma = gscv.best_params_['alpha']
            cur_alpha = gscv.best_params_['kernel__length_scale']
        elif self.kernel_method == 'rbf' or 'laplacian':
            cur_alpha = gscv.best_params_['alpha']
            cur_gamma = gscv.best_params_['gamma']
        self.count += 1
        if self.count == 1:
            self.opt_gamma = cur_gamma
            self.opt_alpha = cur_alpha
        else:
            self.opt_gamma = cur_gamma
            self.opt_alpha = cur_alpha
        pred_y, _ = self.pred(x, 'train', self.opt_gamma)
        loss2 = np.mean(np.power(np.array(y) - np.array(pred_y), 2))
        logger.info('Model 2 Loss: {}, cur gamma {}, alpha {}'.format(loss2, cur_gamma, cur_alpha))
        #logger.info('Model 2 Loss: {}\n'.format(loss2))

        return cur_alpha, self.opt_gamma
        
    def pred(self, x, task, gamma_opt=None, best_index=None):
        #print('curr gamma use is: {}'.format(self.opt_gamma))
        x = x.reshape(-1, self.d_x)
        if task=='train':
            pred_y = self.curr_model.predict(x).reshape(1, -1)[0]
        elif task=='test':
            #logger.info("Model 2 Index is {}".format(best_index))
            pred_y = self.best_model[best_index].predict(x).reshape(1, -1)[0]
        # calculate the updated kernel matrix
        # K = pairwise_kernels(x, metric=self.kernel_method, gamma=gamma_opt) 
        K = GramMatrix(x, self.opt_gamma)
        return pred_y, K

def Train(conf, model_choice, loss_space, train_data_init, test_data, batch_size, epoch_out, epoch_in, \
                init_theta, lr, d_x, d_theta, kernel_method, fit_obj, seed, stop_criterion, out_stop_cri='RKHS'):
    # Step 0: Prepare the data
    x_test = test_data['x_test']
    y_test = test_data['y_test']
    theta = init_theta
    training_error_min = np.inf
    opt_T = 1
    
    model_save = {}
    train_data = train_data_init
    train1_x = train_data['train1_x']
    train1_y = train_data['train1_y']
    train2_x = train_data['train2_x']
    train2_y = train_data['train2_y']
    
    train_x_full = torch.cat((train1_x, train2_x), 0)
    train_y_full = torch.cat((train1_y, train2_y), 0)
    for t in range(epoch_out):        
        # Model 1: hyperparameters for the theta estimation
        # When t = 0, the loss is l2 loss (y - eta(x, theta))^2, 
        # When t> 0, the loss function is the y_hat = y - delta_hat
        logger.info("Outer Loop: %d" % (t))
        if t == 0:
            model_1 = Model_1(conf, theta, d_x, d_theta, lr, epoch_in, loss_space, stop_criterion)
            loss_1, theta_list_1, theta_opt = model_1.train(train_x_full, train_y_full, t, theta)
            logger.info('{} kernel is Applied!'.format(kernel_method))
            model_2 = Model_2(d_x, d_theta, kernel_method)
        else:
            delta_pred_1, _ = model_2.pred(train_x_full, 'train')
            if fit_obj =='residual':
                logger.info('Model 1 Fit Residual, {} Loss!'.format(loss_space))
                train1_y_hat = (train_y_full - delta_pred_1).double()
            elif fit_obj == 'y':
                logger.info('Model 1 Fit y, {} Loss!'.format(loss_space))
                train1_y_hat = train_y_full
            loss_1, theta_list_1, theta_opt = model_1.train(train_x_full, train_y_full, t, theta, model_2, gamma_opt)
        
        y_2_tidle = model_1.pred(train_x_full, train_y_full) #delta hat, use x to predict delta
        _, gamma_opt = model_2.delta_estimate(train_x_full, y_2_tidle)
        
        theta = model_1.theta_opt.clone().detach()

        # Full model: predict
        final_model = Full_model(model_1, model_2) 
        if out_stop_cri == 'MSE':
            predicted_y = final_model.predict(train_x_full, theta, gamma_opt, 'test', t)
            training_error = final_model.test_error(predicted_y, train_y_full)
            if training_error < training_error_min:
                training_error_min = training_error
                opt_T = t
                logger.info('Curr opt model %d, Min Train L2 MSE Loss %.5f\n' % (opt_T, training_error_min))
            logger.info('Full Data L2 loss: {}, theta: {}\n'.format(training_error, theta))
        elif out_stop_cri == 'RKHS':
            para_set = {}
            para_set['alpha'] = model_2.opt_alpha
            _, para_set['K']  = model_2.pred(train_x_full, 'train')
            predicted_y = final_model.predict(train_x_full, theta, gamma_opt, 'test-RKHS', t)
            training_error = final_model.test_error(predicted_y, train_y_full, 'test-RKHS', para_set)
            if training_error < training_error_min:
                training_error_min = training_error
                opt_T = t
                logger.info('Curr opt model %d, Min Train RKHS Loss %.5f\n' % (opt_T, training_error_min))
            logger.info('Full Data RKHS loss: {}, theta: {}\n'.format(training_error, theta))

        model_save['T_%d' % (t)] = [final_model, theta, gamma_opt]
    logger.info('Training Completed! Opt model %d, Training loss %.5f' % (opt_T, training_error_min))

    #logger.info('Optimal theta: {}, Optimal sigma:{}'.format(theta_opt, opt_sig))
    best_model_index = opt_T
    return predicted_y, model_save, best_model_index

class Full_model:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        
    def predict(self, x_test, theta_opt, gamma_opt, task, best_index_2=None):
        if task == 'test':
            y_1_eta_hat = self.model_1.y_s_fun(x_test, theta_opt) # estimated eta
            y_2_delta_hat, _ = self.model_2.pred(x_test, task, gamma_opt, best_index_2) # estimated delta
            pred_y = y_1_eta_hat + y_2_delta_hat
        elif task == 'test-RKHS':
            pred_y = self.model_1.y_s_fun(x_test, theta_opt) # estimated eta
        return pred_y
        
    def test_error(self, y_pred, y_test, task='test', para_set=None):
        if task == 'test':
            loss_func = nn.MSELoss()
            loss_val = loss_func(y_pred, y_test)
            return loss_val.item()
        elif task == 'test-RKHS':
            alpha = para_set['alpha']
            K = torch.from_numpy(para_set['K'])
            n = len(y_test)
            K = K + n * alpha * np.eye(n)
            loss_val = torch.matmul(torch.matmul((y_test - y_pred).view(1, n), torch.linalg.inv(K).float()), (y_test - y_pred).view(n, 1)) / n
            return loss_val.item()
        
        
def plot_pred(d, x_test, y_test, y_train_test_pred_A, configuration, std, n_train, i):
    y_pred = y_train_test_pred_A 
    if d == 1:
        plt.figure(figsize=(8, 6))
        plt.plot(x_test, y_test, 'b.', label='True')
        plt.plot(x_test, y_pred, 'r.', label='Predicted')
        plt.legend()
        plt.title('Configuration: %s, var: %.2f, n_train: %d, rep: %d' % (configuration, np.power(std,2), n_train, i+1))
        plt.savefig('figs/ssc_full_%s_var_%.2f_sample_%d_rep_%d.pdf' % (configuration, np.power(std,2), n_train, i+1))
        plt.close()
    elif d == 2:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Scatter plot the data points
        ax.scatter(x_test[:,0], x_test[:,1], y_test, c='r', marker='x', label='True')
        ax.scatter(x_test[:,0], x_test[:,1], y_pred, c='b', marker='o', label='Predicted')
        # Set labels for axes
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        # Set the legend and label 
        ax.legend(['True', 'Predicted'])
        plt.title('Configuration: %s, var: %.2f, n_train: %d, rep: %d' % (configuration, np.power(std,2), n_train, i+1))
        # save the plot
        plt.savefig('figs/ssc_full_%s_var_%.2f_sample_%d_rep_%d.pdf' % (configuration, np.power(std,2), n_train, i+1))
        plt.close()
    else:
        plt.figure(figsize=(8, 6))
        plt.plot(x_test, y_test, 'b.', label='True')
        plt.plot(x_test, y_pred, 'r.', label='Predicted')
        plt.legend()
        plt.title('Configuration: %s, std: %.2f, n_train: %d, rep: %d' % (configuration, np.power(std,2), n_train, i+1))
        plt.savefig('figs/ssc_full_%s_var_%.2f_sample_%d_rep_%d.pdf' % (configuration, np.power(std,2), n_train, i+1))
        plt.close()
    
class Stat_Cali(object):
    def __init__(self, loss_space, n_trains, batch_size, lr, stds, reps, epoch_in, epoch_out, kernel_method, fit_obj, stop_criterion, out_stop_cri, configurations=['conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4']):
        self.seed = 12345
        self.n_test = 500
        self.configurations = configurations

        self.n_trains = n_trains
        self.stds = stds
        self.reps = reps
        self.lr = lr
        self.epoch_out = epoch_out
        self.epoch_in = epoch_in
        self.batch_size = batch_size
        self.stop_criterion = stop_criterion
        self.val_size = 0.2
        self.kernel_method = kernel_method
        self.fit_obj = fit_obj
        self.loss_space = loss_space
        self.out_stop_cri = out_stop_cri
        
    def run(self):
        # Step 1: hyperparameters for the theta estimation
        self.results_test = {}

        for configuration in tqdm(self.configurations, desc=f'Configuration'):
            for std in tqdm(self.stds):
                for batch_s, n_train in enumerate(self.n_trains):
                    test_error = []
                    for i in range(self.reps):
                        self.seed = self.seed + i
                        logger.info("----------------------------------------------------------------------")
                        logger.info('Conf: %s, var: %.2f, rep: %d' % (configuration, np.power(std,2), i+1))
                        sys.stdout.flush()
                        if configuration == 'conf_0':
                            theta = torch.tensor(1)
                            d_theta = 1
                            d_x = 1
                            conf = CONF_0(theta)
                            x_train, y_train, delta_train, y_s_train = env(configuration, conf, std, n_train, theta, 'train', self.seed)
                            #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.seed)
                            x_val, y_val = x_train, y_train
                            train_A, train_B = data_switch(x_train, y_train, y_s_train, d_x, self.seed) 
                            x_test, y_test, delta_test, y_s_test = env(configuration, conf, std, self.n_test, theta, 'test', self.seed)
                            init_theta = torch.zeros(d_theta) 
                            
                        elif configuration == 'conf_1':
                            theta = torch.tensor([0.2, 0.3])
                            d_theta = len(theta)
                            d_x = 1
                            conf = CONF_1(theta)
                            #print('True theta is {}' .format(theta))
                            # train/test data create
                            x_train, y_train, delta_train, y_s_train = env(configuration, conf, std, n_train, theta, 'train', self.seed)
                            # split the train data into train and validation
                            #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.seed)
                            x_val, y_val = x_train, y_train
                            train_A, train_B = data_switch(x_train, y_train, y_s_train, d_x, self.seed)
                            x_test, y_test, delta_test, y_s_test = env(configuration, conf, std, self.n_test, theta, 'test', self.seed)
                            init_theta = torch.tensor([0.1, 0.1])

                        elif configuration == 'conf_2':
                            theta = torch.tensor([0.2,0.3,0.8])
                            d_theta = len(theta)
                            d_x = 2
                            conf = CONF_2(theta)
                            #print('True theta is {}' .format(theta))
                            # train/test data create
                            x_train, y_train, delta_train, y_s_train = env(configuration, conf, std, n_train, theta, 'train', self.seed)
                            #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.seed)
                            x_val, y_val = x_train, y_train
                            train_A, train_B = data_switch(x_train, y_train, y_s_train, d_x, self.seed)
                            x_test, y_test, delta_test, y_s_test = env(configuration, conf, std, self.n_test, theta, 'test', self.seed)
                            init_theta = torch.tensor([0.2, 0.2, 0.2])    
                            
                        elif configuration == 'conf_3':
                            theta = torch.tensor([0.2,0.4])
                            d_theta = len(theta)
                            d_x = 2
                            conf = CONF_3(theta)
                            #print('True theta is {}' .format(theta))
                            x_train, y_train, delta_train, y_s_train = env(configuration, conf, std, n_train, theta, 'train', self.seed)
                            #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.seed)
                            x_val, y_val = x_train, y_train
                            train_A, train_B = data_switch(x_train, y_train, y_s_train, d_x, self.seed)
                            x_test, y_test, delta_test, y_s_test = env(configuration, conf, std, self.n_test, theta, 'test', self.seed)
                            init_theta = torch.tensor([0.1, 0.1])    
                            
                        elif configuration == 'conf_4':
                            theta = torch.tensor([0.6,0.2])
                            d_theta = len(theta)
                            d_x = 2
                            conf = CONF_4(theta)
                            #print('True theta is {}' .format(theta))
                            x_train, y_train, delta_train, y_s_train = env(configuration, conf, std, n_train, theta, 'train', self.seed)
                            #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.seed)
                            x_val, y_val = x_train, y_train
                            train_A, train_B = data_switch(x_train, y_train, y_s_train, d_x, self.seed)
                            x_test, y_test, delta_test, y_s_test = env(configuration, conf, std, self.n_test, theta, 'test', self.seed)
                            init_theta = torch.tensor([0.1, 0.1])    
                            
                        elif configuration == 'conf_5':
                            theta = torch.tensor(0)
                            d_theta = 1
                            d_x = 1
                            conf = CONF_5(theta)
                            #print('True theta is {}' .format(theta))
                            x_train, y_train, delta_train, y_s_train = env(configuration, conf, std, n_train, theta, 'train', self.seed)
                            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.seed)
                            train_A, train_B = data_switch(x_train, y_train, y_s_train, d_x, self.seed)
                            x_test, y_test, delta_test, y_s_test = env(configuration, conf, std, self.n_test, theta, 'test', self.seed)
                            init_theta = torch.zeros(d_theta) 

                        train_data = Mydataset(x_train, y_train, delta_train, y_s_train)
                        val_data = {}
                        val_data['x_val'] = x_val
                        val_data['y_val'] = y_val
                        
                        test_data = {}
                        test_data['x_test'] = x_test
                        test_data['y_test'] = y_test
                        test_data['delta_test'] = delta_test
                        test_data['y_s_test'] = y_s_test
                        
                        logger.info("Start Training Model A!!!")
                        model_choice = 'A'
                        predicted_y_A, model_save_A, best_model_A_index = \
                            Train(conf, model_choice, self.loss_space, train_A, test_data, self.batch_size[batch_s], self.epoch_out, self.epoch_in, init_theta, self.lr, d_x, d_theta, self.kernel_method, self.fit_obj, self.seed, self.stop_criterion, self.out_stop_cri)
                        # logger.info("-------------------------Start Training Model B!!!--------------------")
                        # model_choice = 'B'
                        # predicted_y_B, model_save_B, best_model_B_index =\
                        #         Train(conf, model_choice, self.loss_space, train_B, test_data, self.batch_size[batch_s], self.epoch_out, self.epoch_in, init_theta, self.lr, d_x, d_theta, self.kernel_method, self.fit_obj, self.seed, self.stop_criterion)
                            
                        # run the model on the test data
                        # val_error = np.zeros((len(model_save_A), len(model_save_B)))
                        # # cross square loss
                        # index_A = 0
                        # for key_A in model_save_A.keys():
                        #     index_B =  0
                        #     full_model_A, theta_A, gamma_opt_A = model_save_A[key_A]
                        #     y_val_pred_A = full_model_A.predict(x_val, theta_A, gamma_opt_A)
                        #     for key_B in model_save_B.keys():
                        #         full_model_B, theta_B, gamma_opt_B = model_save_B[key_B]
                        #         y_val_pred_B = full_model_B.predict(x_val, theta_B, gamma_opt_B)
                        #         val_error_AB = full_model_B.test_error((y_val_pred_A + y_val_pred_B)/2, y_val)
                        #         val_error[index_A, index_B] = val_error_AB
                        #         index_B += 1
                        #     index_A += 1
                        
                        # find the optimal index of combination of Mopdel A and Model B based on the validation loss
                        # row is the model A, column is the model B, find the optimal row and column combination
                        #opt_A_index, opt_B_index = np.unravel_index(np.argmin(val_error, axis=None), val_error.shape)
                        #logger.info('Optimal A index is {}, Opt B index is {}'.format(opt_A_index + 1, opt_B_index + 1))
                        #logger.info('Opimial Val Loss: %.6f' % (np.min(val_error, axis=None)))
                        # select the best model index based on the validation loss
                        
                        # Direct use the best
                        # best_model_A_index = 'T_%d' % (opt_A_index+1)
                        # best_model_B_index = 'T_%d' % (opt_B_index+1)

                        full_model_A_train, theta_A_train, gamma_opt_A_train = model_save_A['T_%d' % (best_model_A_index)]
                        #full_model_B_train, theta_B_train, gamma_opt_B_train = model_save_B['T_%d' % (best_model_B_index)]
                        y_train_test_pred_A = full_model_A_train.predict(x_test, theta_A_train, gamma_opt_A_train, 'test', best_model_A_index)
                        #y_train_test_pred_B = full_model_B_train.predict(x_test, theta_B_train, gamma_opt_B_train, 'test', best_model_B_index)
                        test_train_error_AB = full_model_A_train.test_error(y_train_test_pred_A, y_test)
                        # plot the prediction
                        plot_pred(d_x, x_test, y_test, y_train_test_pred_A, configuration, std, n_train, i)
                        logger.info("Best Model Combination is {}, Test Loss {}".format(best_model_A_index, test_train_error_AB))
                        test_error.append(test_train_error_AB)
                        
                    self.results_test['%s_std_%.2f_sample_%d' % (configuration, std, n_train)] = test_error
                    # logger.info('{}, Test Loss: {}, std: {} '.format(configuration, np.mean(test_error), np.std(test_error)))
        # save the best as the json file
        save_result(self.results_test, 'result/ssc_full_std_%s_configs_%s.json' % (np.power(self.stds,2), self.configurations))
        #print('***********************************')
        
        # calculat the test mean and std over 100 replications        
        # best_test = Best_iter(self.results_test, self.stds, self.configurations)
        # Table_Res(best_test, self.stds, self.configurations)
        for key in self.results_test.keys():
            logger.info('{}, Test Loss: {}, std: {} '.format(key, np.mean(self.results_test[key]), np.std(self.results_test[key])))



def main():
    n_trains = [50] # simulation sample size
    stds = [np.sqrt(0.1), np.sqrt(0.25), np.sqrt(0.5), np.sqrt(1)] # np.sqrt(0.1), np.sqrt(0.25), np.sqrt(0.5), np.sqrt(1)
    reps = 100 # repeat for 100 times

    # hyparparameters for the model
    epoch_in = 1000 # model 1 maximum inner iteration
    epoch_out = 3 # maximum outer interation
    kernel_method = 'rbf' # 'rbf', 'matern', 'ExpSineSquared', 'laplacian_kernel', 'polynomial'
    fit_obj = 'y' # 'residual', 'y'
    if fit_obj == 'residual':
        loss_space = 'L2' # 'RKHS', 'L2'
        out_stop_cri= "MSE" # RKHS, MSE
    elif fit_obj == 'y':
        loss_space = 'RKHS' # 'RKHS', 'L2'
        out_stop_cri= "RKHS" # RKHS, MSE


    stop_criterion = 1e-4 # the minimum relative change of the loss function for the inner iteration to stop
    lr = 1e-4 # learning rate for sgd
    batch_size = [50] 
    configurations = ['conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4'] # ['conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4']

    stat = Stat_Cali(loss_space, n_trains, batch_size, lr, stds, reps, epoch_in, epoch_out, kernel_method, fit_obj, stop_criterion, out_stop_cri, configurations)
    stat.run()
    
if __name__ == '__main__':
    main()