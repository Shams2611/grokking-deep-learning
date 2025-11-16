# input gate - what new info to store

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

print("INPUT GATE")
print()
print("two parts:")
print("  1. what to update: i_t = sigmoid(...)")
print("  2. candidate values: c_tilde = tanh(...)")
print()

np.random.seed(42)
hidden_size = 4

h_prev = np.array([0.5, -0.3, 0.1, 0.8])
x = np.array([1.0, 0.0, 0.5])
combined = np.concatenate([h_prev, x])

# input gate weights
W_i = np.random.randn(len(combined), hidden_size) * 0.5
W_c = np.random.randn(len(combined), hidden_size) * 0.5

# input gate
i = sigmoid(combined @ W_i)
print("input gate (what to update):", np.round(i, 3))

# candidate values
c_tilde = tanh(combined @ W_c)
print("candidate values:", np.round(c_tilde, 3))

# what we'll add to cell state
to_add = i * c_tilde
print("to add to cell:", np.round(to_add, 3))
