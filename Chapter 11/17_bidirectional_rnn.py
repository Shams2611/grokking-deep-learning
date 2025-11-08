# bidirectional RNN - process both directions

import numpy as np

def tanh(x): return np.tanh(x)

print("BIDIRECTIONAL RNN")
print()
print("idea: some tasks need future context too!")
print()
print('example: "I saw the ___ in the sky"')
print("  forward only: could be many things")
print('  with future: "...it was beautiful" -> sun/moon')
print()

np.random.seed(42)
input_size, hidden_size = 3, 4

# forward and backward weights
W_xh_f = np.random.randn(input_size, hidden_size) * 0.5
W_hh_f = np.random.randn(hidden_size, hidden_size) * 0.5
W_xh_b = np.random.randn(input_size, hidden_size) * 0.5
W_hh_b = np.random.randn(hidden_size, hidden_size) * 0.5

# sequence
X = np.random.randn(5, input_size)

# forward pass
h_f = np.zeros(hidden_size)
forward_states = []
for x in X:
    h_f = tanh(x @ W_xh_f + h_f @ W_hh_f)
    forward_states.append(h_f)

# backward pass (reverse sequence)
h_b = np.zeros(hidden_size)
backward_states = []
for x in reversed(X):
    h_b = tanh(x @ W_xh_b + h_b @ W_hh_b)
    backward_states.append(h_b)
backward_states = backward_states[::-1]

# concatenate
combined = [np.concatenate([f, b]) for f, b in zip(forward_states, backward_states)]

print(f"forward hidden size: {hidden_size}")
print(f"backward hidden size: {hidden_size}")
print(f"combined hidden size: {len(combined[0])}")
