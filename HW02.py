# Task01

import numpy as np 
from numpy import random
import tensorflow as tf


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

## n_unit = number of units in the layer
## input_unit= number of unit in preceding Layer
## 
class layer():
    def __init__(self, n_units, input_units, biasvector, weightmatrix, layer_input, layer_preactivation, layer_activation):  
        self.n_units = n_units
        self.input_units = input_units
        self.biasvector = biasvector
        self.weightmatrix = weightmatrix
        layer_input = None
        layer_preactivation = None
        layer_activation = None

## weightmatrix with dimensions of hidden layer x input layer and random values
        weightmatrix = np.random.rand(n_units, input_units)
## added assert to change the shape of the matrix for every layer
        assert(weightmatrix.shape == (n_units, input_units))

## biasvector with dimensions of hidden layer x 1, since it is a vector, with values of 0
        bias = np.zeros (n_units, 1)
        assert(bias.shape == (n_units, 1))
    
        def forward_step():
## z= value of layer without relu threshold
## z= weight * previous layer + bias
## might have to include a case when we are in layer 1, since there is no preceding layer
         layer_preactivation = np.dot(weightmatrix, input_units)
         layer_activation = np.dot(weightmatrix, input_units) + biasvector
         return tf.nn.relu(layer_activation)


## general question: backpropagation via sigmoid or relu
##def backward_step():   
##with tf.GradientTape() as tape:
 ##   tape.watch(weight)
    ## gradient in respect to weight 
  ##  dW =
    ## gradient in respect to bias
   ## db = 
   
           def backward_step(): 
                # layer 1
                x1 = tf.matmul(x, weightmatrix)
                # activation function of layer 1
                x0 = tf.nn.sigmoid(x1)
                with tf.GradientTape() as tape:
                        tape.watch([weightmatrix])
                        x1 = tf.matmul(x, weightmatrix)  # x = input values in task01
                        x0 = tf.nn.sigmoid(x1)
                        
                        x1 = tf.reduce_mean(0.5 * (x0 - t)**2) # mean squared error, t = target in task01
                        real_grad_weightmatrix = tape.gradient(x1, [weightmatrix])
                        real_grad_biasvector = tape.gradient(x1, [biasvector])
                  
                  
            def update_weights(weightmatrix, weight_gradients, learning_rate=1e-2):
                        for w, w_grad in zip(weightmatrix, weight_gradients):
                                w = w.assign(w - learning_rate*w_grad)
                        return weightmatrix
                
             
            def update_bias(biasvector, bias_gradients, learning_rate=1e-2):
                        for w, w_grad in zip(biasvector, bias_gradients):
                                w = w.assign(w - learning_rate*w_grad)
                        return biasvector
