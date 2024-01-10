#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:21:19 2024

@author: ruchak
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(0)

mu = 170
sd = 7

# generate samples from our distribution
x = norm.rvs(loc = mu, scale = sd, size = 100)

#maximum likelihood mean
print('Maximum likelihood of mean', x.mean())

#maximum likelihood variance
print('Maximum likelihood of variance', x.var())

print('Self calculation of maximum likelihood estimate of mean', ((x-x.mean())**2).mean())

#maximum likelihood std
print('Maximum likelihood of std dev.', x.std())

#unbiased variance
print('Unbiased estimate of variance', x.var(ddof = 1)) #delta degrees of freedom

print('Self calculation of the unbiased estimate of variance', ((x-x.mean())**2).sum()/(len(x)-1))

#unbiased std
print('Unbiased estimate of std dev.', x.std(ddof = 1))

#at what height are you in the 95th percentile?
print(norm.ppf(0.95, loc = mu, scale = sd))

#you are 160 cm tall, what percentile are you in?
print(norm.cdf(160, loc = mu, scale = sd))

#you are 180 cm tall, what is the probability that someone is taller than you?
print(1-norm.cdf(180, loc = mu, scale = sd))

#1-cdf is also called as survival function. So we could do the following too:
print(norm.sf(180, loc = mu, scale = sd))