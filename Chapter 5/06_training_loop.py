# training loop with multiple weights

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])
goal = 0.8
lr = 0.01

print("training:")
for i in range(20):
    pred = inputs @ weights
    error = (pred - goal) ** 2
    delta = pred - goal
    gradients = delta * inputs
    weights = weights - lr * gradients

    if i % 5 == 0:
        print(f"  iter {i:2d}: error={error:.6f}")

print(f"\nfinal weights: {weights}")
print(f"final prediction: {inputs @ weights:.4f}")
