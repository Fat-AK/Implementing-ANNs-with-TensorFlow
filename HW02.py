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

# Task02

def threshold (x):
    if x> 0:
        return 1
    else:
        return 0

class Perceptron():
    def __init__(self, n_units, input_units, biasvector, weightmatrix,threshold_function = threshold):  
        
        self. n_units = n_units
        self. input_units = input_units
        self. biasvector = biasvector 
        self. weightmatrix = weight
        self. threshold_function = threshold_function
        
    def __call__ (self, input_units:random.uniform(low=0.0, high=1.0, size=10)): 
        T = self. weightmatrix @ input_units + self. biasvector
        return self.threshold_function(T)
    
weight = random.uniform(low=1, high=2, size=10)
bias = np.zeros (10)

input_units = random.uniform(low=0.0, high=1.0, size=10)

perceptron01 = Perceptron (10, input_units, weight, bias)
