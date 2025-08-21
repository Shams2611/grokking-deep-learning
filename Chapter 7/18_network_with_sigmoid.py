# network with sigmoid activation

import numpy as np
np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# XOR problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# weights
w1 = np.random.randn(2, 4) * 0.5
w2 = np.random.randn(4, 1) * 0.5
lr = 1.0  # need higher lr for sigmoid

print("training with sigmoid:")
for epoch in range(1000):
    total_error = 0

    for i in range(len(X)):
        # forward
        h_raw = X[i] @ w1
        h = sigmoid(h_raw)
        out = h @ w2

        # error
        error = (out - y[i]) ** 2
        total_error += error[0]

        # backward
        d_out = out - y[i]
        d_h = d_out @ w2.T * sigmoid_derivative(h_raw)

        # update
        w2 -= lr * np.outer(h, d_out)
        w1 -= lr * np.outer(X[i], d_h)

    if epoch % 200 == 0:
        print(f"  epoch {epoch}: error = {total_error:.4f}")
