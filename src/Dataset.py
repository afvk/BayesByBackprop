#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:03:21 2018

@author: arent
"""



import numpy as np
import matplotlib.pyplot as plt





class Dataset:
    
    def __init__(self, N_train):
        x = np.random.uniform(low=0.0, high=0.5, size=round(1.2*N_train))
        y = self.gen_data(x, noisy=True)
        
        self.x_train = x[:N_train]
        self.y_train = y[:N_train]
        
        self.x_val = x[N_train:]
        self.y_val = y[N_train:]
        
        
        self.x_test = np.random.uniform(low=0.0, high=0.5, size=100)
        self.y_test = self.gen_data(self.x_test)
        
        self.x_true = np.linspace(-0.2, 1.3, 1000)
        self.y_true = self.gen_data(self.x_true)
        
        self.N_feat = 1
        self.N_out = 1
        
        self.x_train = self.x_train[:,np.newaxis]
        self.y_train = self.y_train[:,np.newaxis]
        self.x_val = self.x_val[:,np.newaxis]
        self.y_val = self.y_val[:,np.newaxis]
        self.x_test = self.x_test[:,np.newaxis]
        
    
    def gen_data(self, x, noisy=False):
        if noisy:
            eps = np.random.normal(loc=0, scale=0.02, size=len(x))
            
        else:
            eps = 0
        
        y = x + 0.3*np.sin(2*np.pi*(x+eps)) + 0.3*np.sin(4*np.pi*(x+eps)) + eps
        
        return y
    

                
    
    
    


#    def visualize(self):
#        plt.figure()
#        plt.plot(self.x, self.y)
#        plt.plot(self.x_train, self.y_train, '.')
#        plt.show()

