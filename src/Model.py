#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:44:14 2018

@author: arent
"""

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns
import pdb


class Model:
    
    def __init__(self, lr, N_epochs, data, N_nodes, act_funcs, sigma_1, 
                 sigma_m):
        self.lr = lr
        self.N_epochs = N_epochs
        self.data = data
        self.N_nodes = N_nodes
        self.act_funcs = act_funcs
        self.sigma_1 = sigma_1
        self.sigma_m = sigma_m
        
        self.build_model()
        
        
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.data.N_feat])
        self.y = tf.placeholder(tf.float32, shape=[None, self.data.N_out])
        
        N_nodes = [self.data.N_feat] + self.N_nodes + [self.data.N_out]
        self.N_layers = len(N_nodes)
        
        mu_w = []
        mu_b = []
        rho_w = []
        rho_b = []
        
        N_params = 0
        
        self.w_shapes = []
        self.b_shapes = []
        
        for i in range(1, self.N_layers):
            w_shape = [N_nodes[i-1], N_nodes[i]]
            b_shape = [N_nodes[i]]
            
            self.w_shapes.append(w_shape)
            self.b_shapes.append(b_shape)
            
            N_params += N_nodes[i-1]*N_nodes[i] + N_nodes[i]
            
            mu_w.append(tf.Variable(tf.truncated_normal(w_shape, stddev=0.1),
                                    name='mu_w%i'%i))
            mu_b.append(tf.Variable(tf.zeros(b_shape), name='mu_b%i'%i))
            
            rho_w.append(tf.Variable(-5*tf.ones(w_shape), name='rho_w%i'%i))
            rho_b.append(tf.Variable(-5*tf.ones(b_shape), name='rho_b%i'%i))
        
        self.N_params = N_params
        
        
        self.flatten_theta(mu_w, rho_w, mu_b, rho_b)
        self.sigma_flat = tf.math.log(1 + tf.math.exp(self.rho_flat))
                
        log_posts = []
        log_priors = []
        log_likelihoods = []
        
        N_samples = 1
        
        for i in range(N_samples):
            params_flat = self.sample_params()
            w, b = self.unpack_parameters(params_flat)
            self.y_pred = self.forward(self.x, w, b)
            
            log_post =   tf.reduce_sum(tf.math.log(self.sigma_flat)) \
                       - 0.5*tf.reduce_sum(((params_flat - self.mu_flat)/self.sigma_flat)**2)
            
            log_prior = - 1.0/(2.0*self.sigma_1**2)*tf.reduce_sum(params_flat**2)
            
            log_likelihood = - 0.5*tf.reduce_sum(((self.y_pred - self.y)/self.sigma_m)**2)
            
            log_posts.append(log_post)
            log_priors.append(log_prior)
            log_likelihoods.append(log_likelihood)
        
        
        self.lijstje = tf.trainable_variables(scope=None)
        self.loss_post = tf.reduce_mean(log_posts)
        self.loss_prior = tf.reduce_mean(log_priors)
        self.loss_likelihood = tf.reduce_mean(log_likelihoods)
        
        self.loss = self.loss_post - self.loss_prior - self.loss_likelihood
        
        self.dfdw = tf.gradients(self.loss, params_flat)
        self.dfdmu = tf.gradients(self.loss, self.mu_flat)
        self.dfdrho = tf.gradients(self.loss, self.rho_flat)
        
        self.grad_post_mu = tf.gradients(self.loss_post, self.mu_flat)
        self.grad_post_rho = tf.gradients(self.loss_post, self.rho_flat)
        self.grad_prior_mu = tf.gradients(self.loss_prior, self.mu_flat)
        self.grad_prior_rho = tf.gradients(self.loss_prior, self.rho_flat)
        self.grad_likelihood_mu = tf.gradients(self.loss_likelihood, self.mu_flat)
        self.grad_likelihood_rho = tf.gradients(self.loss_likelihood, self.rho_flat)
        
#        clip_gradient_norm = 0.2
        
        self.var_list = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.var_list)
        
        
        
#        
#        if clip_gradient_norm > 0.0:
#            clip_global = tf.Variable(clip_gradient_norm,trainable=False)
#            grads, self.gradient_norm = tf.clip_by_global_norm(grads, clip_global)
#        else:
#            self.gradient_norm = tf.global_norm(grads)
        
        self.params_flat = params_flat
        
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        
        gvs = list(zip(self.grads, self.var_list))
        self.optimizer = optimizer.apply_gradients(gvs)
#        self.optimizer = self.update_parameters()
        
        
    def sample_params(self):
        eps = tf.random_normal(shape=[self.N_params])
        params_flat = self.mu_flat + tf.math.log(1 + tf.math.exp(self.rho_flat))*eps
        return params_flat
            
        
    def flatten_theta(self, mu_w, rho_w, mu_b, rho_b):
        mu_w_flat = [tf.reshape(mu_w[i], 
                                [np.shape(mu_w[i])[0]*np.shape(mu_w[i])[1]]) for i in range(self.N_layers-1)]
        rho_w_flat = [tf.reshape(rho_w[i], 
                                 [np.shape(rho_w[i])[0]*np.shape(rho_w[i])[1]]) for i in range(self.N_layers-1)]
    
        mu_w_flat = tf.concat(mu_w_flat, axis=0)
        mu_b_flat = tf.concat(mu_b, axis=0)
        rho_w_flat = tf.concat(rho_w_flat, axis=0)
        rho_b_flat = tf.concat(rho_b, axis=0)
        
        self.mu_flat = tf.concat([mu_w_flat, mu_b_flat], axis=0)
        self.rho_flat = tf.concat([rho_w_flat, rho_b_flat], axis=0)
    
    
    def forward(self, inp, w, b):
        for i in range(self.N_layers-1):
            inp = self.act_funcs[i](tf.matmul(inp, w[i]) + b[i])
        
        return inp
    
        
    def update_parameters(self):
        self.mu_flat -= tf.multiply(self.lr, (self.dfdw + self.dfdmu))
        self.rho_flat -= tf.multiply(self.lr, (self.dfdw*self.eps/(1 + tf.math.exp(-self.rho_flat)) + self.dfdrho))        


    def unpack_parameters(self, params_flat):
        w = []
        b = []
        
        i = 0
        
        for w_shape in self.w_shapes:
            shape0, shape1 = w_shape
            w0 = tf.reshape(params_flat[i:i + shape0*shape1], [shape0, shape1])
            w.append(w0)
            
            i += shape0*shape1
            
        for b_shape in self.b_shapes:
            b.append(params_flat[i:i + b_shape[0]])
            
            i += b_shape[0]
        
        return w, b


    def train(self):
        loss_lst = []
        val_loss_lst = []
        
        log_post_list = []
        log_prior_list = []
        log_likelihood_list = []
        
        
        
        
        
        init = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        self.sess.run(init)
        
        for epoch in range(self.N_epochs):
            (_, 
             loss, 
             loss_post, 
             loss_prior, 
             loss_likelihood,
             mu_flat,
             rho_flat,
             ) = self.sess.run([
                                         self.optimizer, 
                                         self.loss, 
                                         self.loss_post,
                                         self.loss_prior, 
                                         self.loss_likelihood,
                                         self.mu_flat,
                                         self.rho_flat,
                                         ], 
                                         feed_dict={self.x:self.data.x_train,
                                                    self.y:self.data.y_train})

            loss_lst.append(loss)
           
#            pdb.set_trace()
            
#            self.h1.set_ydata(rho_flat)
#            self.fig.canvas.draw()
#            self.fig.canvas.flush_events()
#            plt.pause(1e-6)

#            print(np.allclose(sigma_flat, ))
            
            log_post_list.append(loss_post)
            log_prior_list.append(-loss_prior)
            log_likelihood_list.append(-loss_likelihood)
            
            
            print('Epoch %i/%i, loss = %.3e'%(epoch+1, self.N_epochs, loss))
        
        return loss_lst, val_loss_lst, log_post_list, log_prior_list, log_likelihood_list
    
    
    def predict(self, x, N_samples=1):
        samples = []
        
        for i in range(N_samples):
            y_pred = self.sess.run([self.y_pred], feed_dict={self.x:x})
            samples.append(y_pred)
        
        samples = np.vstack(samples)
        
        return samples.squeeze()


    

















