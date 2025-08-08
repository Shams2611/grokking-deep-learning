# using ReLU between layers

import numpy as np
np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

# simple 2-layer network with ReLU
x = np.array([0.5, 0.8])
w1 = np.random.randn(2, 3) * 0.5
w2 = np.random.randn(3, 1) * 0.5

# forward pass
hidden_raw = x @ w1
print("before relu:", hidden_raw)

hidden = relu(hidden_raw)
print("after relu:", hidden)

output = hidden @ w2
print("output:", output)
