# Task01

import numpy as np 
from numpy import random


x = random.uniform(low=0.0, high=1.0, size=100) # random numbers as input values 
x = sorted (x) # make an order 
t = np.empty(100) #  target array 

 
for i in range (0,100):
    t[i]=x[i]**3-x[i]**2 # target values 

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot (t,x)
plt.show()
