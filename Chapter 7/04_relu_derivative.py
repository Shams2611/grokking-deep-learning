# ReLU derivative - needed for backprop
# super simple: 1 if positive, 0 if negative

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

x = np.array([-2, -1, 0, 1, 2])

print("x:", x)
print("relu(x):", relu(x))
print("relu'(x):", relu_derivative(x))
print()
print("gradient flows through positive values")
print("gradient is blocked for negative values")
