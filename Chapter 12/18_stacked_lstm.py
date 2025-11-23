# stacked LSTM - multiple layers

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

def lstm_step(x, h, c, W, hidden_size):
    combined = np.concatenate([h, x])
    gates = combined @ W
    hs = hidden_size
    f = sigmoid(gates[:hs])
    i = sigmoid(gates[hs:2*hs])
    c_tilde = tanh(gates[2*hs:3*hs])
    o = sigmoid(gates[3*hs:])
    c_new = f * c + i * c_tilde
    h_new = o * tanh(c_new)
    return h_new, c_new

np.random.seed(42)

# 2-layer LSTM
input_size = 3
hidden_sizes = [8, 8]
num_layers = 2

# weights for each layer
weights = []
for i, hs in enumerate(hidden_sizes):
    in_size = input_size if i == 0 else hidden_sizes[i-1]
    W = np.random.randn(in_size + hs, 4 * hs) * 0.1
    weights.append(W)

# sequence
X = np.random.randn(4, input_size)

print("2-layer LSTM:")
print()

# process sequence
h_layers = [np.zeros(hs) for hs in hidden_sizes]
c_layers = [np.zeros(hs) for hs in hidden_sizes]

for t, x in enumerate(X):
    layer_input = x
    for layer in range(num_layers):
        h_layers[layer], c_layers[layer] = lstm_step(
            layer_input, h_layers[layer], c_layers[layer],
            weights[layer], hidden_sizes[layer]
        )
        layer_input = h_layers[layer]  # output to next layer

    print(f"t={t}: layer 1 h norm={np.linalg.norm(h_layers[0]):.3f}, "
          f"layer 2 h norm={np.linalg.norm(h_layers[1]):.3f}")
