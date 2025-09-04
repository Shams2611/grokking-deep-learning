# full training with dropout

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

# XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

w1 = np.random.randn(2, 8) * 0.5
w2 = np.random.randn(8, 1) * 0.5
lr = 0.5
keep_prob = 0.8

print("training with dropout (keep_prob=0.8):")
for epoch in range(1000):
    total_error = 0

    for i in range(len(X)):
        # forward with dropout
        h_raw = X[i] @ w1
        h = relu(h_raw)
        mask = (np.random.rand(*h.shape) < keep_prob) / keep_prob
        h_drop = h * mask
        out = h_drop @ w2

        error = (out - y[i])**2
        total_error += error[0]

        # backward with same mask
        d_out = out - y[i]
        d_h_drop = d_out @ w2.T
        d_h = d_h_drop * mask * relu_deriv(h_raw)

        w2 -= lr * np.outer(h_drop, d_out)
        w1 -= lr * np.outer(X[i], d_h)

    if epoch % 200 == 0:
        print(f"  epoch {epoch}: error = {total_error:.4f}")

# test without dropout
print("\ntesting (no dropout):")
for i in range(len(X)):
    h = relu(X[i] @ w1)
    out = h @ w2
    print(f"  {X[i]} -> {out[0]:.3f} (target: {y[i][0]})")
