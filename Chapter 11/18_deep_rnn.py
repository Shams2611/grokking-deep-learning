# stacked/deep RNN - multiple layers

import numpy as np

def tanh(x): return np.tanh(x)

print("DEEP RNN (stacked layers)")
print()

np.random.seed(42)

input_size = 4
hidden_sizes = [8, 8, 8]  # 3 layers
num_layers = len(hidden_sizes)

# weights for each layer
weights = []
for i, h_size in enumerate(hidden_sizes):
    in_size = input_size if i == 0 else hidden_sizes[i-1]
    W_xh = np.random.randn(in_size, h_size) * 0.5
    W_hh = np.random.randn(h_size, h_size) * 0.5
    weights.append((W_xh, W_hh))

# sequence
X = np.random.randn(5, input_size)

# forward pass
print(f"input shape: {X.shape}")
print()

# initialize hidden states for each layer
h_layers = [np.zeros(h_size) for h_size in hidden_sizes]

for t, x in enumerate(X):
    layer_input = x
    for layer in range(num_layers):
        W_xh, W_hh = weights[layer]
        h_layers[layer] = tanh(layer_input @ W_xh + h_layers[layer] @ W_hh)
        layer_input = h_layers[layer]  # input to next layer

    print(f"t={t}: layer outputs = {[h.shape for h in h_layers]}")

print()
print("each layer adds more abstraction")
