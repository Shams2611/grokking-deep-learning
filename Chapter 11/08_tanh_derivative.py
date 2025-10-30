# tanh derivative for BPTT

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    """derivative: 1 - tanh(x)^2"""
    t = tanh(x)
    return 1 - t**2

# test
x = np.array([-2, -1, 0, 1, 2])

print("tanh and its derivative:")
print()
print("x:", x)
print("tanh(x):", np.round(tanh(x), 3))
print("tanh'(x):", np.round(tanh_derivative(x), 3))
print()
print("derivative is max (1.0) at x=0")
print("gets smaller as |x| increases")
print("this causes vanishing gradients!")
