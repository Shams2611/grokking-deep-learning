# RNN as a layer class

import numpy as np

class RNNLayer:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        # initialize weights
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_h = np.zeros(hidden_size)

    def forward(self, X, h0=None):
        """
        X: (seq_len, input_size)
        returns: all hidden states
        """
        seq_len = len(X)
        if h0 is None:
            h0 = np.zeros(self.hidden_size)

        self.X = X
        self.h_list = [h0]
        self.h_raw_list = []

        h = h0
        for t in range(seq_len):
            h_raw = X[t] @ self.W_xh + h @ self.W_hh + self.b_h
            h = np.tanh(h_raw)
            self.h_raw_list.append(h_raw)
            self.h_list.append(h)

        return np.array(self.h_list[1:])  # exclude h0

# test
np.random.seed(42)
rnn = RNNLayer(input_size=4, hidden_size=8)

X = np.random.randn(5, 4)  # 5 timesteps, 4 features
outputs = rnn.forward(X)

print(f"input shape: {X.shape}")
print(f"output shape: {outputs.shape}")
print(f"each timestep produces hidden state of size {rnn.hidden_size}")
