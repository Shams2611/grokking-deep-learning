# basic RNN cell computation

import numpy as np

def tanh(x):
    return np.tanh(x)

# RNN cell: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)

# dimensions
input_size = 4
hidden_size = 3

# weights
W_xh = np.random.randn(input_size, hidden_size) * 0.1  # input to hidden
W_hh = np.random.randn(hidden_size, hidden_size) * 0.1  # hidden to hidden
b_h = np.zeros(hidden_size)

# RNN cell function
def rnn_cell(x, h_prev):
    """single RNN step"""
    h_new = tanh(x @ W_xh + h_prev @ W_hh + b_h)
    return h_new

# test
x = np.array([1, 0, 1, 0])  # input
h = np.zeros(hidden_size)   # initial hidden

print("RNN Cell:")
print(f"  input: {x}")
print(f"  hidden before: {h}")

h = rnn_cell(x, h)
print(f"  hidden after: {np.round(h, 3)}")
