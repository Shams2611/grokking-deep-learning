# multiple weights problem

# before: 1 input, 1 weight
# now: multiple inputs, multiple weights

# how does each weight contribute to error?
# need gradient for EACH weight

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])
goal = 0.8

pred = inputs @ weights
error = (pred - goal) ** 2

print(f"inputs: {inputs}")
print(f"weights: {weights}")
print(f"prediction: {pred:.4f}")
print(f"error: {error:.4f}")
