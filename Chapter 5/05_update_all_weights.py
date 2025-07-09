# update all weights at once

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])
goal = 0.8
lr = 0.01

print(f"before: {weights}")

pred = inputs @ weights
delta = pred - goal
gradients = delta * inputs

# update all weights
weights = weights - lr * gradients

print(f"after:  {weights}")
