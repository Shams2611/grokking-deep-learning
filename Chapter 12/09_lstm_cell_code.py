# LSTM cell implementation

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # initialize all weights
        scale = 0.1
        self.W_f = np.random.randn(combined_size, hidden_size) * scale
        self.W_i = np.random.randn(combined_size, hidden_size) * scale
        self.W_c = np.random.randn(combined_size, hidden_size) * scale
        self.W_o = np.random.randn(combined_size, hidden_size) * scale

        self.b_f = np.zeros(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)

    def forward(self, x, h_prev, c_prev):
        # concatenate input and previous hidden
        combined = np.concatenate([h_prev, x])

        # gates
        f = sigmoid(combined @ self.W_f + self.b_f)
        i = sigmoid(combined @ self.W_i + self.b_i)
        c_tilde = tanh(combined @ self.W_c + self.b_c)
        o = sigmoid(combined @ self.W_o + self.b_o)

        # cell state update
        c = f * c_prev + i * c_tilde

        # hidden state
        h = o * tanh(c)

        return h, c

# test
np.random.seed(42)
lstm = LSTMCell(input_size=3, hidden_size=4)

x = np.array([1.0, 0.5, -0.3])
h = np.zeros(4)
c = np.zeros(4)

h_new, c_new = lstm.forward(x, h, c)
print(f"input: {x}")
print(f"new hidden: {np.round(h_new, 3)}")
print(f"new cell: {np.round(c_new, 3)}")
