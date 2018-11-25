#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:02:23 2018

@author: arent
"""

import sys
sys.path.append('../src')

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from Model import Model
from Dataset import Dataset

np.random.seed(123)

N_train = 5000
lr = 1e-6
N_epochs = 50000
N_hidden = [64, 32, 16]
act_funcs = [tf.nn.relu, tf.nn.relu, tf.nn.relu, lambda x: x]
#act_funcs = [lambda x: x, lambda x: x]
sigma_1 = 1e-1
sigma_m = 1e-1


data = Dataset(N_train)
model = Model(lr, N_epochs, data, N_hidden, act_funcs, sigma_1, sigma_m)



#model.fig = plt.figure()
#ax = model.fig.add_subplot(1,1,1)
#model.fig.canvas.draw()
#model.h1, = ax.plot(np.arange(model.N_params), np.zeros(model.N_params), 'r')
#ax.set_ylim([-10, 10])
#ax.set_xlim([0,model.N_params])

loss_lst, val_loss_lst, log_post_list, log_prior_list, log_likelihood_list = model.train()


x_pred = np.linspace(-0.2, 1.2, 100)
x_pred_tf = x_pred[:,np.newaxis]

y_pred = model.predict(x_pred_tf, N_samples=100)

mean = y_pred.mean(axis=0)
std = y_pred.std(axis=0)

plt.figure()
for i in range(100):
    plt.plot(x_pred, y_pred[i,:], 'k', alpha=0.05)
plt.show()


plt.figure()
plt.plot(data.x_true, data.y_true, label='True')
plt.plot(data.x_train, data.y_train, 'xk', label='Train', alpha=0.03)
plt.plot(x_pred, mean, 'r', label='Pred')
plt.fill_between(x_pred, mean+3*std, mean, alpha=0.3, color='r')
plt.fill_between(x_pred, mean-3*std, mean, alpha=0.3, color='r')
plt.legend()
plt.show()

plt.figure()
plt.plot(loss_lst, label='Loss')
plt.plot(log_post_list, label='Post')
plt.plot(log_prior_list, label='Prior')
plt.plot(log_likelihood_list, label='Likelihood')
#plt.plot(val_loss_lst, label='Validation loss')
plt.legend()
plt.show()



