# multi input multi output gradient

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1],
])
goals = np.array([0.1, 1.0, 0.1])
lr = 0.01

# forward
preds = weights @ inputs
deltas = preds - goals

# gradients: outer product of deltas and inputs
gradients = np.outer(deltas, inputs)

# update
weights = weights - lr * gradients

print("weight gradients shape:", gradients.shape)
print("same shape as weights!")
