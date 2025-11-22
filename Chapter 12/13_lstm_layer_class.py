# LSTM as a layer class

import numpy as np

class LSTMLayer:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        scale = 0.1

        # all weights in one matrix for efficiency
        self.W = np.random.randn(combined, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

    def forward(self, X):
        """X: (seq_len, input_size)"""
        seq_len = len(X)
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        self.h_list = [h]
        self.c_list = [c]

        for t in range(seq_len):
            combined = np.concatenate([h, X[t]])
            gates = combined @ self.W + self.b

            # split into 4 gates
            hs = self.hidden_size
            f = 1 / (1 + np.exp(-gates[:hs]))
            i = 1 / (1 + np.exp(-gates[hs:2*hs]))
            c_tilde = np.tanh(gates[2*hs:3*hs])
            o = 1 / (1 + np.exp(-gates[3*hs:]))

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

            self.h_list.append(h)
            self.c_list.append(c)

        return np.array(self.h_list[1:])

# test
np.random.seed(42)
lstm = LSTMLayer(input_size=3, hidden_size=8)

X = np.random.randn(5, 3)  # 5 timesteps
outputs = lstm.forward(X)

print(f"input shape: {X.shape}")
print(f"output shape: {outputs.shape}")
print(f"hidden size: {lstm.hidden_size}")
