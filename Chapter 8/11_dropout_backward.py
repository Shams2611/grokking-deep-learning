# dropout in backward pass
# gradients only flow through non-dropped neurons

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

# forward pass with dropout (saving mask)
x = np.array([0.5, 0.8])
w1 = np.random.randn(2, 4) * 0.5
w2 = np.random.randn(4, 1) * 0.5
keep_prob = 0.5

# forward
h_raw = x @ w1
h = relu(h_raw)
mask = (np.random.rand(*h.shape) < keep_prob) / keep_prob
h_dropped = h * mask
out = h_dropped @ w2

print("forward pass:")
print(f"  hidden (before dropout): {np.round(h, 3)}")
print(f"  mask: {np.round(mask, 1)}")
print(f"  hidden (after dropout): {np.round(h_dropped, 3)}")

# backward - use same mask!
target = 1.0
d_out = out - target
d_h_dropped = d_out @ w2.T
d_h = d_h_dropped * mask  # apply same mask!

print()
print("backward pass:")
print(f"  gradient before mask: {np.round(d_h_dropped, 3)}")
print(f"  gradient after mask: {np.round(d_h, 3)}")
