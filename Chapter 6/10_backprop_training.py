# backprop training loop

import numpy as np
np.random.seed(42)

# data
X = np.array([[1,0,1], [0,1,1], [1,1,1], [0,0,1]])
y = np.array([[0], [1], [1], [0]])

# weights
w01 = np.random.randn(3, 4) * 0.5
w12 = np.random.randn(4, 1) * 0.5
lr = 0.5

print("training 2-layer network:")
for epoch in range(100):
    total_error = 0

    for i in range(len(X)):
        # forward
        hidden = X[i] @ w01
        output = hidden @ w12

        # error
        error = (output - y[i]) ** 2
        total_error += error[0]

        # backward
        d_out = output - y[i]
        d_hid = d_out @ w12.T

        # update
        w12 -= lr * np.outer(hidden, d_out)
        w01 -= lr * np.outer(X[i], d_hid)

    if epoch % 20 == 0:
        print(f"  epoch {epoch}: error = {total_error:.4f}")
