import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
from utils import *
from sklearn.model_selection import train_test_split, KFold
from tqdm.notebook import tqdm
import sklearn.model_selection
import sklearn.kernel_ridge
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import inv
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
            #theta = self.init_theta.clone().detach().requires_grad_(True)
            theta = self.init_theta
            logger.info('Model 1 Start from {}'.format(theta))
        else:
            logger.info('Model 2 Start from {}'.format(theta))
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
        epsilon = 1e-5
        for i in range(self.epoch):
            # forward pass
            eta_est = self.y_s_fun(train_x, theta) 
            if self.loss_space == 'RKHS':
                if t == 0:
                    loss = loss_func(eta_est, train_y)
                    # perturbed
                    gradient_approximation = torch.zeros(self.d_theta)
                    for d in range(self.d_theta):
                        theta_perturbed = theta.clone().detach().requires_grad_(False)
                        #logger.info("Theta Perturbed: {}".format(theta_perturbed))
                        theta_perturbed[d] += epsilon
                        theta_perturbed = theta_perturbed.requires_grad_(True)
                        eta_est_pert = self.y_s_fun(train_x, theta_perturbed) 
                        loss_pert = loss_func(eta_est_pert, train_y)
                        gradient_approximation[d] = (loss_pert - loss) / epsilon 
                    # Perturb the input tensor
                else:
                    # train_y: y - delta_hat
                    # eta_est: estimated eta(x, theta_hat)
                    loss = torch.matmul(torch.matmul((train_y - eta_est).view(1, self.n_train), torch.linalg.inv(self.K_dyn)), (train_y - eta_est).view(self.n_train, 1)) / self.n_train
                    # perturbed
                    gradient_approximation = torch.zeros(self.d_theta)
                    for d in range(self.d_theta):
                        theta_perturbed = theta.clone().detach().requires_grad_(False)
                        #logger.info("Theta Perturbed: {}".format(theta_perturbed))
                        theta_perturbed[d] += epsilon
                        theta_perturbed = theta_perturbed.requires_grad_(True)
                        eta_est_pert = self.y_s_fun(train_x, theta_perturbed) 
                        loss_pert = torch.matmul(torch.matmul((train_y - eta_est_pert).view(1, self.n_train), torch.linalg.inv(self.K_dyn)), (train_y - eta_est_pert).view(self.n_train, 1)) / self.n_train
                        gradient_approximation[d] = (loss_pert - loss) / epsilon 
                theta.grad = gradient_approximation.to(theta.dtype)
                # if ((i+1)%100) == 0:
                #     logger.info("Grad App: {}".format(theta.grad))
            
            elif self.loss_space == 'L2':
                loss = loss_func(eta_est, train_y)
                gradient_approximation = torch.zeros(self.d_theta)
                for d in range(self.d_theta):
                    theta_perturbed = theta.clone().detach().requires_grad_(False)
                    #logger.info("Theta Perturbed: {}".format(theta_perturbed))
                    theta_perturbed[d] += epsilon
                    theta_perturbed = theta_perturbed.requires_grad_(True)
                    eta_est_pert = self.y_s_fun(train_x, theta_perturbed) 
                    loss_pert = loss_func(eta_est_pert, train_y)
                    gradient_approximation[d] = (loss_pert - loss) / epsilon 
                theta.grad = gradient_approximation.to(theta.dtype)
            # backward pass
            # print(theta.grad)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #scheduler.step()
            theta = theta - self.lr * (1.0/np.sqrt(i+1)) * theta.grad
            loss_list.append(loss.item())
            self.theta_list[i,] = theta.detach()
            
            # set stop criterion, if the loss moving average (100) change is smaller than 1e-2 percent, stop
            if (i == self.epoch - 1) or i > 2 * gap and abs((np.mean(loss_list[i-gap:i]) - np.mean(loss_list[i - 2* gap:i-gap]))/np.mean(loss_list[i- 2* gap:i-gap])) < self.stop_criterion:
                logger.info('Model 1 stopped after: {} iterations, Model 1 Loss: {}'.format(i, loss.item()))
                #logger.info('Model 1 Loss: {}'.format(loss.item()))
                break 
            
            # print the loss
            # if (i+1) % 100 == 0:
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
            hyper_space = dict(alpha=np.logspace(-1, 2, 100), 
                               gamma=np.logspace(-2, 2, 50))
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

def Train(conf, model_choice, loss_space, train_data_init, test_data, epoch_out, epoch_in, \
                init_theta, lr, d_x, d_theta, kernel_method, fit_obj, seed, stop_criterion, out_stop_cri='MSE'):
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
        #logger.info("y_2_tidle: {}".format(y_2_tidle.clone().detach()))
        _, gamma_opt = model_2.delta_estimate(train_x_full, y_2_tidle.clone().detach())
        
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
            #logger.info("y_1_eta_hat: {}".format(y_1_eta_hat.clone().detach()))
            pred_y = y_1_eta_hat.clone().detach() + y_2_delta_hat
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
            loss_val = torch.matmul(torch.matmul((y_test - y_pred).view(1, n), torch.linalg.inv(K)), (y_test - y_pred).view(n, 1)) / n
            return loss_val.item()
        
        
def plot_pred(d, x_test, y_test, y_train_test_pred_A, configuration, i):
    y_pred = y_train_test_pred_A 
    if d == 1:
        plt.figure(figsize=(8, 6))
        plt.plot(x_test, y_test, 'b.', label='True')
        plt.plot(x_test, y_pred, 'r.', label='Predicted')
        plt.legend()
        plt.title('Configuration: %s, rep: %d' % (configuration, i))
        plt.savefig('figs/ssc_real_full_%s_rep_%d.pdf' % (configuration, i))
        plt.close()
        if i == 6:
            fig = plt.figure(figsize=(8, 6))
            # plot x_test and true point y_test and predicted point y_pred with different marker
            plt.scatter(x_test, y_test, label='True', marker='x', color='r')
            plt.scatter(x_test, y_pred, label='Predicted', marker='o', color='b')
            plt.legend(fontsize=14)
            plt.title('Ion Channels Example', fontsize=14)
            plt.xlabel('log(time)', fontsize=14)
            plt.ylabel('Normalized Current', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.savefig('figs/ssc_real_final_%s_rep_%d.pdf' % (configuration, i))
            plt.close()
    
        
class Stat_Cali(object):
    def __init__(self, rep, lr, epoch_in, epoch_out, stop_criterion, configurations, kernel_method, fit_obj, loss_space, out_stop_cri):
        self.seed = 12345
        self.n_test = 200
        self.configurations = configurations

        self.lr = lr
        self.epoch_in = epoch_in
        self.epoch_out = epoch_out
        self.stop_criterion = stop_criterion
        self.val_size = 0.2
        self.kernel_method = kernel_method
        self.fit_obj = fit_obj
        self.loss_space = loss_space
        self.out_stop_cri = out_stop_cri
        self.reps = rep  
        
    def run(self):
        # Step 1: hyperparameters for the theta estimation
        self.results_test = {}
        self.results_test['test'] = []
        d_x = 1
        d_theta = 3
        init_theta = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        
        x = torch.tensor([-1.71479842809193, -1.02165124753198, -0.579818495252942, -0.198450938723838, 0.0953101798043249,
                    0.350656871613169, 0.598836501088704, 0.824175442966349, 1.03673688495002, 1.23547147138531,
                    1.43031124653667, 1.62136648329937, 1.80500469597808, 1.98513086220859, 2.16332302566054,
                    2.33795223683134, 2.50959926237837, 2.67965072658051, 2.84954976337591]).double()

        y = torch.tensor([0.0538181818181818, 0.0878181818181818, 0.121090909090909, 0.135636363636364, 0.118363636363636,
                        0.0900000000000000, 0.0649090909090909, 0.0470909090909091, 0.0338181818181818,
                        0.0243636363636364, 0.0174545454545455, 0.0118181818181818, 0.00890909090909091, 0.00600000000000000,
                        0.00454545454545455, 0.00272727272727273, 0.00236363636363636, 0.00181818181818182, 0.00127272727272727]).double()

        # create the obs data into dataframe
        obs_data = pd.DataFrame({'x': x, 'y': y})
        #obs_data.to_csv('obs_data.csv')
        
        # create a cross validation set with 5 folds
        x_test_all = []
        y_test_all = []
        y_test_pred_all = []
        for rep in range(self.reps):
            logger.info('--------------------    REP {}     -----------------------'.format(rep))
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed+rep)
            # create the train and test data
            z = 1
            test_error_i = []
            for train_index, test_index in kf.split(obs_data):                
                logger.info('{}, fold {}'.format(self.configurations, z))
                obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
                
                conf = CONF_Heart(torch.tensor(np.array(obs_train['x'])), torch.tensor(np.array(obs_train['y'])))
                
                x_train_obs, y_train_obs = env_real(np.array(obs_train['x']), np.array(obs_train['y']))
                train_A, train_B = data_switch_real(x_train_obs, y_train_obs, d_x, self.seed)

                test_data = {}
                test_data['x_test'] = obs_test['x'].tolist()
                test_data['y_test'] = obs_test['y'].tolist()
                x_test = torch.tensor(np.array(test_data['x_test']))
                y_test = torch.tensor(np.array(test_data['y_test']))

                logger.info("Start Training Model A!!!")
                model_choice = 'A'
                predicted_y, model_save, best_model_index = \
                    Train(conf, model_choice, self.loss_space, train_A, test_data, self.epoch_out, self.epoch_in, init_theta, self.lr, d_x, d_theta, self.kernel_method, self.fit_obj, self.seed, self.stop_criterion, self.out_stop_cri)               
                
                full_model_A_train, theta_A_train, gamma_opt_A_train = model_save['T_%d' % (best_model_index)]
                y_train_test_pred_A = full_model_A_train.predict(x_test, theta_A_train, gamma_opt_A_train, 'test', best_model_index)
                test_train_error_AB = full_model_A_train.test_error(y_train_test_pred_A, y_test)
                test_error_i.append(test_train_error_AB)
                self.results_test['rep_%s_fold_%d_x_test_y_test_y_pred' % (rep+1, z)] = [np.array(x_test).tolist(), np.array(y_test).tolist(), np.array(y_train_test_pred_A).tolist()]
                
                # x_test_all.append(np.array(x_test))
                # y_test_all.append(np.array(y_test))
                # y_test_pred_all.append(np.array(y_train_test_pred_A))
                
                # plot_pred(d_x, x_test, y_test, y_train_test_pred_A, self.configurations,z)
                logger.info('{}, Opimial Test error is {}'.format(self.configurations, test_train_error_AB))
                z += 1
                
            logger.info('The rep {}: mean {}'.format(rep+1, np.mean(test_error_i)))
            self.results_test['test'].append(np.mean(test_error_i))
        # save the best to the json file
        save_result(self.results_test, 'result/real_ssc_full_data_result.json')
        logger.info("The final mean is {}, sd is {}".format(np.mean(self.results_test['test']), np.std(self.results_test['test'])))
        # plot the prediction
        # save_result_for_plot = {}
        # save_result_for_plot['x_test'] = np.concatenate(x_test_all).tolist()
        # save_result_for_plot['y_test'] = np.concatenate(y_test_all).tolist()
        # save_result_for_plot['y_pred'] = np.concatenate(y_test_pred_all).tolist()
        # save_result(save_result_for_plot, 'result/real_ssc_full_data_result.json')
        # plot_pred(d_x, np.concatenate(x_test_all), np.concatenate(y_test_all), np.concatenate(y_test_pred_all), self.configurations, z)

def main():
    rep = 100  
    epoch_in = 2000 # 2000
    epoch_out = 3 # 3
    kernel_method = 'rbf' # 'rbf', 'matern', 'ExpSineSquared', 'laplacian_kernel', 'polynomial'
    fit_obj = 'y' # 'residual', 'y'
    if fit_obj == 'residual':
        loss_space = 'L2' # 'RKHS', 'L2'
        out_stop_cri= "MSE" # RKHS
    elif fit_obj == 'y':
        loss_space = 'RKHS' # 'RKHS', 'L2'
        out_stop_cri= "RKHS" # RKHS


    stop_criterion = 1e-4
    lr = 20
    configurations = 'conf_heart' 
    stat = Stat_Cali(rep, lr, epoch_in, epoch_out, stop_criterion, configurations, kernel_method, fit_obj, loss_space, out_stop_cri)
    stat.run()

if __name__ == '__main__':
    main()