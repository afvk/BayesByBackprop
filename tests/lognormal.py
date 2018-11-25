#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:25:27 2018

@author: arent
"""

import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt



D = 20

mu = np.random.random(D)
sigma = np.random.random(D)

x = np.random.random(D)


ans_scipy = multivariate_normal.logpdf(x, mu, sigma**2)

def log_normal(x, mu, sigma):
    D = len(x)
    return - 0.5*D*np.log(2*np.pi) - np.sum(np.log(sigma)) - 0.5*np.sum(((x-mu)/sigma)**2)

ans_own = log_normal(x, mu, sigma)

