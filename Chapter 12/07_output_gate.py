# output gate - what to output

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

print("OUTPUT GATE")
print()
print("decides what parts of cell state to output")
print()
print("o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)")
print("h_t = o_t * tanh(c_t)")
print()

np.random.seed(42)
hidden_size = 4

# cell state (after update)
c = np.array([1.5, -0.5, 0.8, 2.0])

# output gate (pretend we computed this)
o = np.array([0.8, 0.2, 0.9, 0.5])

print("cell state:", c)
print("output gate:", o)
print()

# hidden state output
h = o * tanh(c)
print("tanh(cell):", np.round(tanh(c), 3))
print("hidden output:", np.round(h, 3))
print()
print("cell state is filtered before output")
