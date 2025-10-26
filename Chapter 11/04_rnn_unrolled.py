# unrolling RNN through time

import numpy as np

def tanh(x): return np.tanh(x)

# weights
np.random.seed(42)
input_size, hidden_size = 2, 3
W_xh = np.random.randn(input_size, hidden_size) * 0.5
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5
b_h = np.zeros(hidden_size)

def rnn_cell(x, h):
    return tanh(x @ W_xh + h @ W_hh + b_h)

# sequence of 4 inputs
sequence = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
])

print("unrolling RNN through sequence:")
print()

h = np.zeros(hidden_size)
hidden_states = [h]

for t, x_t in enumerate(sequence):
    h = rnn_cell(x_t, h)
    hidden_states.append(h)
    print(f"t={t}: input={x_t}, hidden={np.round(h, 3)}")

print()
print("same weights used at each timestep!")
