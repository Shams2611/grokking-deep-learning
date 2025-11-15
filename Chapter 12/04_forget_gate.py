# forget gate - what to remove from memory

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("FORGET GATE")
print()
print("decides what to throw away from cell state")
print()
print("f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)")
print()

# example
np.random.seed(42)
hidden_size = 4

# previous hidden + current input
h_prev = np.array([0.5, -0.3, 0.1, 0.8])
x = np.array([1.0, 0.0, 0.5])
combined = np.concatenate([h_prev, x])

# forget gate weights
W_f = np.random.randn(len(combined), hidden_size) * 0.5
b_f = np.zeros(hidden_size)

# forget gate output
f = sigmoid(combined @ W_f + b_f)

print("forget gate output:", np.round(f, 3))
print()
print("values close to 0: forget this info")
print("values close to 1: keep this info")
