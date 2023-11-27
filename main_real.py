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
from sklearn.experimental import enable_halving_search_cv
import sklearn.kernel_ridge
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import inv
from sklearn.gaussian_process.kernels import Matern
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model_1:
    def __init__(self, conf, init_theta, d_x, d_theta, lr, epoch, stop_criterion = 1e-5):
        self.init_theta = init_theta
        self.lr = lr
        self.epoch = epoch
        self.conf = conf
        self.d_theta = d_theta
        self.d_x = d_x
        self.stop_criterion = stop_criterion
        self.theta_list = torch.zeros((self.epoch, self.d_theta))
        self.opt_loss = np.inf
        self.decay = 0.5
        self.gamma_cur = torch.zeros(1)
        
    def train(self, train_x, train_y, t, theta, model_2=None, gamma=None):
        # define the optimizer with stochastic gradient descent
        # initialize theta
        gap = 5
        if t == 0:
            theta = self.init_theta.clone().detach().requires_grad_(True)
        else:
            theta = theta.clone().detach().requires_grad_(True)
        
        # define the optimizer
        optimizer = torch.optim.SGD([theta], lr=self.lr, momentum=0.95, weight_decay=0.5) # nesterov=True 
        loss_func = nn.MSELoss()
        
        #store the loss and theta
        loss_list = []
        lrs = []
        
        if t > 0:
            self.n_train = len(train_y)
            _, K = model_2.pred(train_x)
            K = K + self.n_train * model_2.opt_alpha * np.eye(self.n_train)
            # we will use the y_s_hat as the prediction
            # however, it will be weighted by the kernel matrix
            # so the loss will be (train_y - y_s_hat)^T K (train_y - y_s_hat)/n
            self.K = torch.from_numpy(K)
            if t == 1:
                self.K_dyn = self.K
            else:
                self.K_dyn = (self.K_dyn + self.K)/(t)
        # train the model
        for i in range(self.epoch):
            # forward pass
            y_eta_est = self.y_s_fun(train_x, theta)
            if t == 0:
                loss = loss_func(y_eta_est, train_y)
            else:
                # train_y: y - delta
                # y_s_hat: eta
                # train_y - y_eta_est: residual
                loss = torch.matmul(torch.matmul((train_y - y_eta_est).view(1, self.n_train), torch.linalg.inv(self.K_dyn)), (train_y - y_eta_est).view(self.n_train, 1)) / self.n_train
            loss_list.append(loss.item())
            self.theta_list[i,] = theta.detach()
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            
            # set stop criterion, if the loss moving average (100) change is smaller than 1e-2 percent, stop
            if (i == self.epoch - 1) or i > 2 * gap and abs((np.mean(loss_list[i-gap:i]) - np.mean(loss_list[i- 2* gap:i-gap]))/np.mean(loss_list[i- 2* gap:i-gap])) < self.stop_criterion:
                #logger.info('Model 1 stopped after: {} iterations, Model 1 Loss: {}'.format(i+1, loss.item()))
                logger.info('Model 1 Loss: {}'.format(loss.item()))
                break 
            
            # print the loss
            #if (i+1) % 1000 == 0:
            #    print('epoch: {}, theta: {}, loss: {}'.format(i+1, theta.detach(), loss.item()))
            #if t > 0 and i % 400 == 0:
            #    print('epoch: %d, theta: {}, loss: {} \n  pred_y_s: {}, pred_delta: {}'.format(i, theta.item(), loss.item(), y_s_hat[0], delta_pred[0], ))

        # select the theta for the minmum loss
        self.opt_loss = np.min(loss_list)
        self.theta_opt = self.theta_list[np.argmin(loss_list)]
        #logger.info('Theta Opt is %s' % (self.theta_opt.tolist()))
        
        return loss_list, self.theta_list, self.theta_opt

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
            y_2_hat = torch.zeros(n)
            y_2_error = torch.zeros(n)
            for i in range(n):
                y_2_hat[i] = self.conf.eta(train_x[i, ], self.theta_opt.clone().detach())
                y_2_error[i] = train_y[i] - y_2_hat[i]
        return y_2_error
    
class Model_2:
    def __init__(self, d_x, d_theta, kernel_method = 'rbf'): #rbf
        self.d_x = d_x
        self.d_theta = d_theta
        self.kernel_method = kernel_method
        self.count = 0
        self.opt_gamma = 0
        
    def delta_estimate(self, x, y):
        # x is the point to be estimated
        # x_sample_2 is the sample2 data
        # y_error is the sample2 data's residual
        # return the estimated sample2 data's residual
        
        # use kernel ridge regression to estimate the delta
        estimator = sklearn.kernel_ridge.KernelRidge(kernel=self.kernel_method)
        gscv = sklearn.model_selection.GridSearchCV(
            estimator=estimator,
            param_grid=dict(alpha=np.logspace(-2, 5, 100), gamma=np.logspace(-2, 1, 10)),
            cv = 5,
            scoring='neg_mean_squared_error',
            refit=True,
            verbose=0
        )
        
        self.best_model = gscv.fit(x.reshape(-1, self.d_x), y.reshape(-1, 1))
        cur_alpha = gscv.best_params_['alpha']
        cur_gamma = gscv.best_params_['gamma']
        self.count += 1
        if self.count == 1:
            self.opt_gamma = cur_gamma
            self.opt_alpha = cur_alpha
        else:
            self.opt_gamma = self.opt_gamma * (self.count-1)/self.count + cur_gamma/self.count
            self.opt_alpha = cur_alpha
        pred_y, _ = self.pred(x, self.opt_gamma)
        loss2 = np.mean(np.power(np.array(y) - np.array(pred_y), 2))
        #logger.info('Model 2 Loss: {}, cur gamma {}, ave gamma {}, lambda {}\n'.format(loss2, cur_gamma, self.opt_gamma, 1/cur_alpha))
        logger.info('Model 2 Loss: {}\n'.format(loss2))
        
        return cur_alpha, self.opt_gamma
        
    def pred(self, x, gamma_opt=None):
        #print('curr gamma use is: {}'.format(self.opt_gamma))
        x = x.reshape(-1, self.d_x)
        pred_y = self.best_model.predict(x).reshape(1, -1)[0]
        # calculate the updated kernel matrix
        K = pairwise_kernels(x, metric=self.kernel_method, gamma=gamma_opt) 
        return pred_y, K

def Train(conf, model_choice, train_data_init, test_data, batch_size, epoch_out, epoch_in, \
                init_theta, lr, d_x, d_theta, seed, stop_criterion):
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
        logger.info("Outer Loop: %d" % (t+1))
        if t == 0:
            model_1 = Model_1(conf, theta, d_x, d_theta, lr, epoch_in, stop_criterion)
            loss_list_1, theta_list_1, theta_opt = model_1.train(train1_x, train1_y, t, theta)
            model_2 = Model_2(d_x, d_theta)
        else:
            delta_pred_1, _ = model_2.pred(train1_x) # predict delta hat
            train1_y_eta_est = train1_y - delta_pred_1 # y - y_delta
            loss_list_1, theta_list_1, theta_opt = model_1.train(train1_x, train1_y_eta_est, t, theta, model_2, gamma_opt)
        
        y_2_error = model_1.pred(train2_x, train2_y) #delta hat, use x to predict delta
        _, gamma_opt = model_2.delta_estimate(train2_x.detach().numpy(), y_2_error.detach().numpy())
        
        theta = model_1.theta_opt.clone().detach()
        
        # Full model: predict
        final_model = Full_model(model_1, model_2) 
        predicted_y = final_model.predict(train_x_full, theta, gamma_opt)
        training_error = final_model.test_error(predicted_y, train_y_full)
        if training_error < training_error_min:
            training_error_min = training_error
            opt_T = t+1
            logger.info('Curr opt model %d, Min training loss %.5f \n' % (opt_T, training_error_min))

        #logger.info('Train loss: %.5f \n' % (training_error))
        model_save['T_%d' % (t+1)] = [final_model, theta, gamma_opt]
    logger.info('Training Completed! Opt model %d, Training loss %.5f \n' % (opt_T, training_error_min))

    #logger.info('Optimal theta: {}, Optimal sigma:{}'.format(theta_opt, opt_sig))
    return predicted_y, model_save

class Full_model:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        
    def predict(self, x_test, theta_opt, gamma_opt):
        self.y_1_eta_hat = self.model_1.y_s_fun(x_test, theta_opt) # estimated eta
        self.y_2_delta_hat, _ = self.model_2.pred(x_test, gamma_opt) # estimated delta
        self.pred_y = self.y_1_eta_hat.detach() + torch.tensor(self.y_2_delta_hat)

        return self.pred_y
        
    def test_error(self, y_pred, y_test):
        loss_func = nn.MSELoss()
        loss_val = loss_func(y_pred, y_test)
    
        # print the loss and optimal theta
        #print('Test error: %.3f' % (loss_val)) 
        return loss_val.item()
        #print("***********************************")
        
class Stat_Cali(object):
    def __init__(self, lr, epoch, epoch_out, stop_criterion, configurations):
        self.seed = 12345
        self.n_test = 200
        self.configurations = configurations

        self.epoch = epoch
        self.lr = lr
        self.epoch_out = epoch_out
        self.epoch_in = epoch
        self.stop_criterion = stop_criterion
        
    def run(self):
        # Step 1: hyperparameters for the theta estimation
        self.results_test = []
        d_x = 1
        d_theta = 3
        init_theta = torch.tensor([0.5, 0.5, 0.5])
        
        x = np.array([-1.71479842809193, -1.02165124753198, -0.579818495252942, -0.198450938723838, 0.0953101798043249,
                    0.350656871613169, 0.598836501088704, 0.824175442966349, 1.03673688495002, 1.23547147138531,
                    1.43031124653667, 1.62136648329937, 1.80500469597808, 1.98513086220859, 2.16332302566054,
                    2.33795223683134, 2.50959926237837, 2.67965072658051, 2.84954976337591])

        y = np.array([0.0538181818181818, 0.0878181818181818, 0.121090909090909, 0.135636363636364, 0.118363636363636,
                        0.0900000000000000, 0.0649090909090909, 0.0470909090909091, 0.0338181818181818,
                        0.0243636363636364, 0.0174545454545455, 0.0118181818181818, 0.00890909090909091, 0.00600000000000000,
                        0.00454545454545455, 0.00272727272727273, 0.00236363636363636, 0.00181818181818182, 0.00127272727272727])

        # create the obs data into dataframe
        obs_data = pd.DataFrame({'x': x, 'y': y})
        #obs_data.to_csv('obs_data.csv')
        
        # create a cross validation set with 5 folds
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        # create the train and test data
        z = 1
        for train_index, test_index in kf.split(obs_data):
            logger.info('{}, fold {}'.format(self.configurations, z))
            obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
            
            test_error = []
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
            _, model_save_A = \
                Train(conf, model_choice, train_A, test_data, len(y_train_obs), self.epoch_out, self.epoch_in, init_theta, self.lr, d_x, d_theta, self.seed, self.stop_criterion)
            logger.info("Start Training Model B!!!")
            model_choice = 'B'
            _, model_save_B =\
                    Train(conf, model_choice, train_B, test_data, len(y_train_obs), self.epoch_out, self.epoch_in, init_theta, self.lr, d_x, d_theta, self.seed, self.stop_criterion)
                
            # run the model on the test data
            test_error_i = []
            # cross square loss
            for key_A in model_save_A.keys():
                full_model_A, theta_A, gamma_opt_A = model_save_A[key_A]
                #print(key_A, theta_A, gamma_opt_A)
                y_test_pred_A = full_model_A.predict(x_test, theta_A, gamma_opt_A)
                test_error_A_list = []
                for key_B in model_save_B.keys():
                    full_model_B, theta_B, gamma_opt_B = model_save_B[key_B]
                    y_test_pred_B = full_model_B.predict(x_test, theta_B, gamma_opt_B)
                    test_error_AB = full_model_B.test_error((y_test_pred_A + y_test_pred_B)/2, y_test)
                    test_error_A_list.append(test_error_AB)
                test_error_i.append(np.min(test_error_A_list))
            #logger.info('{}, Test error is {}'.format(self.configurations, np.min(test_error_i)))
            test_error.append(test_error_i)
            logger.info('Opimial Test Loss is {}'.format(np.min(test_error_i)))
            self.results_test.append(np.min(test_error))
            z += 1
        logger.info('The final mean {}, sd {}'.format(np.mean(self.results_test), np.std(self.results_test)))
        # save the best to the json file
        save_result(self.results_test, 'result/real_data_result.json')

epoch_in = 1000
epoch_out = 5
stop_criterion = 1e-4
lr = 1e-4
configurations = 'conf_heart' 
stat = Stat_Cali(lr, epoch_in, epoch_out, stop_criterion, configurations)
stat.run()