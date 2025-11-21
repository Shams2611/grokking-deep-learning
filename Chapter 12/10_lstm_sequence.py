# processing a sequence with LSTM

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

np.random.seed(42)

# LSTM parameters
input_size, hidden_size = 2, 4
combined_size = input_size + hidden_size

W_f = np.random.randn(combined_size, hidden_size) * 0.1
W_i = np.random.randn(combined_size, hidden_size) * 0.1
W_c = np.random.randn(combined_size, hidden_size) * 0.1
W_o = np.random.randn(combined_size, hidden_size) * 0.1

def lstm_step(x, h, c):
    combined = np.concatenate([h, x])
    f = sigmoid(combined @ W_f)
    i = sigmoid(combined @ W_i)
    c_tilde = tanh(combined @ W_c)
    o = sigmoid(combined @ W_o)
    c_new = f * c + i * c_tilde
    h_new = o * tanh(c_new)
    return h_new, c_new

# sequence
sequence = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
])

print("processing sequence through LSTM:")
print()

h = np.zeros(hidden_size)
c = np.zeros(hidden_size)

for t, x in enumerate(sequence):
    h, c = lstm_step(x, h, c)
    print(f"t={t}: h={np.round(h, 3)}")
    print(f"    c={np.round(c, 3)}")
