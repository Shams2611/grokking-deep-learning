# backward pass for classification

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# setup
x = np.array([0.5, 0.8])
w1 = np.random.randn(2, 4) * 0.5
w2 = np.random.randn(4, 3) * 0.5
target = np.array([1, 0, 0])  # one-hot

# forward (save intermediates)
h_raw = x @ w1
hidden = relu(h_raw)
logits = hidden @ w2
probs = softmax(logits)

print("BACKWARD PASS:")
print()

# gradient of loss w.r.t logits (the simple formula!)
d_logits = probs - target
print(f"d_logits: {np.round(d_logits, 3)}")

# gradient w.r.t w2
d_w2 = np.outer(hidden, d_logits)
print(f"d_w2 shape: {d_w2.shape}")

# gradient w.r.t hidden
d_hidden = d_logits @ w2.T
print(f"d_hidden: {np.round(d_hidden, 3)}")

# gradient w.r.t h_raw (through ReLU)
d_h_raw = d_hidden * relu_deriv(h_raw)
print(f"d_h_raw: {np.round(d_h_raw, 3)}")

# gradient w.r.t w1
d_w1 = np.outer(x, d_h_raw)
print(f"d_w1 shape: {d_w1.shape}")
