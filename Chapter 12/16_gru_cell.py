# GRU cell implementation

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        scale = 0.1

        self.W_z = np.random.randn(combined, hidden_size) * scale
        self.W_r = np.random.randn(combined, hidden_size) * scale
        self.W_h = np.random.randn(combined, hidden_size) * scale

    def forward(self, x, h_prev):
        combined = np.concatenate([h_prev, x])

        # update gate
        z = sigmoid(combined @ self.W_z)

        # reset gate
        r = sigmoid(combined @ self.W_r)

        # candidate hidden
        combined_reset = np.concatenate([r * h_prev, x])
        h_tilde = tanh(combined_reset @ self.W_h)

        # new hidden state
        h = (1 - z) * h_prev + z * h_tilde

        return h

# test
np.random.seed(42)
gru = GRUCell(input_size=3, hidden_size=4)

x = np.array([1.0, 0.5, -0.3])
h = np.zeros(4)

h_new = gru.forward(x, h)
print(f"input: {x}")
print(f"new hidden: {np.round(h_new, 3)}")
