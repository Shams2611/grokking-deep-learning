# dropout in forward pass

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)

def forward_with_dropout(x, w1, w2, keep_prob, training=True):
    # layer 1
    h_raw = x @ w1
    h = relu(h_raw)

    # dropout on hidden layer
    if training:
        mask = (np.random.rand(*h.shape) < keep_prob) / keep_prob
        h = h * mask

    # layer 2
    out = h @ w2
    return out, h_raw

# test
x = np.array([0.5, 0.8])
w1 = np.random.randn(2, 4) * 0.5
w2 = np.random.randn(4, 1) * 0.5

print("same input, different outputs (due to dropout):")
for i in range(3):
    out, _ = forward_with_dropout(x, w1, w2, 0.5, training=True)
    print(f"  forward {i+1}: {out[0]:.4f}")

print()
print("at test time (no dropout):")
out, _ = forward_with_dropout(x, w1, w2, 0.5, training=False)
print(f"  output: {out[0]:.4f}")
