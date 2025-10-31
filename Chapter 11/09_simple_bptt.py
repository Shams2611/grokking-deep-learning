# simple BPTT implementation

import numpy as np

def tanh(x): return np.tanh(x)
def tanh_deriv(x): return 1 - np.tanh(x)**2

np.random.seed(42)
input_size, hidden_size = 2, 3

W_xh = np.random.randn(input_size, hidden_size) * 0.5
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5

# forward pass (save everything)
def forward(sequence, h0):
    h = h0
    h_list = [h]
    h_raw_list = []

    for x_t in sequence:
        h_raw = x_t @ W_xh + h @ W_hh
        h = tanh(h_raw)
        h_raw_list.append(h_raw)
        h_list.append(h)

    return h_list, h_raw_list

# backward pass
def backward(sequence, h_list, h_raw_list, d_h_final):
    d_W_xh = np.zeros_like(W_xh)
    d_W_hh = np.zeros_like(W_hh)

    d_h = d_h_final
    T = len(sequence)

    for t in reversed(range(T)):
        # gradient through tanh
        d_h_raw = d_h * tanh_deriv(h_raw_list[t])

        # accumulate weight gradients
        d_W_xh += np.outer(sequence[t], d_h_raw)
        d_W_hh += np.outer(h_list[t], d_h_raw)

        # gradient to previous hidden state
        d_h = d_h_raw @ W_hh.T

    return d_W_xh, d_W_hh

# test
sequence = np.array([[1, 0], [0, 1], [1, 1]])
h0 = np.zeros(hidden_size)

h_list, h_raw_list = forward(sequence, h0)
d_h_final = np.ones(hidden_size)  # fake gradient

d_W_xh, d_W_hh = backward(sequence, h_list, h_raw_list, d_h_final)
print("BPTT gradients computed!")
print(f"d_W_xh shape: {d_W_xh.shape}")
print(f"d_W_hh shape: {d_W_hh.shape}")
