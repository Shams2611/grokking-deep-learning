# stochastic gradient descent (SGD)
# update weights after EACH example

import numpy as np

X = np.array([
    [8.5, 0.65, 1.2],
    [9.5, 0.80, 1.3],
    [9.9, 0.80, 0.5],
    [9.0, 0.90, 1.0],
])
goals = np.array([1, 1, 0, 1])

weights = np.array([0.1, 0.2, -0.1])
lr = 0.01

print("SGD: update after each example")
for epoch in range(3):
    for i in range(len(X)):
        pred = X[i] @ weights
        delta = pred - goals[i]
        weights = weights - lr * delta * X[i]

    total_error = sum((X @ weights - goals) ** 2)
    print(f"  epoch {epoch}: total_error={total_error:.4f}")
