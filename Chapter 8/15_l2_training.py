# training with L2 regularization

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

# data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

w1 = np.random.randn(2, 8) * 0.5
w2 = np.random.randn(8, 1) * 0.5
lr = 0.5
lambda_reg = 0.01

print("training with L2 regularization:")
for epoch in range(1000):
    total_error = 0

    for i in range(len(X)):
        # forward
        h = relu(X[i] @ w1)
        out = h @ w2

        error = (out - y[i])**2
        total_error += error[0]

        # backward with L2
        d_out = out - y[i]
        d_h = d_out @ w2.T * relu_deriv(X[i] @ w1)

        # update with weight decay
        w2 = w2 * (1 - lr * lambda_reg) - lr * np.outer(h, d_out)
        w1 = w1 * (1 - lr * lambda_reg) - lr * np.outer(X[i], d_h)

    if epoch % 200 == 0:
        w_norm = np.sum(w1**2) + np.sum(w2**2)
        print(f"  epoch {epoch}: error={total_error:.4f}, weight_norm={w_norm:.2f}")

print("\nweights stay smaller with L2!")
