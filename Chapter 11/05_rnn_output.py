# RNN outputs - can output at each step or just final

import numpy as np

def tanh(x): return np.tanh(x)

np.random.seed(42)
input_size, hidden_size, output_size = 2, 4, 3

W_xh = np.random.randn(input_size, hidden_size) * 0.5
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5
W_hy = np.random.randn(hidden_size, output_size) * 0.5  # hidden to output

def rnn_step(x, h):
    h_new = tanh(x @ W_xh + h @ W_hh)
    y = h_new @ W_hy  # output
    return h_new, y

# sequence
sequence = np.array([[1, 0], [0, 1], [1, 1]])

print("RNN with outputs at each step:")
print()

h = np.zeros(hidden_size)
for t, x_t in enumerate(sequence):
    h, y = rnn_step(x_t, h)
    print(f"t={t}: output = {np.round(y, 3)}")

print()
print("many-to-many: output at each step")
print("many-to-one: only use final output")
