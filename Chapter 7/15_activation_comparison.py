# comparing all activations

import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.array([-2, -1, 0, 1, 2])

print("x:        ", x)
print("relu:     ", relu(x))
print("leaky:    ", np.round(leaky_relu(x), 2))
print("sigmoid:  ", np.round(sigmoid(x), 2))
print("tanh:     ", np.round(tanh(x), 2))
