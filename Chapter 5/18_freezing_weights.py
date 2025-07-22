# freezing weights

import numpy as np

# sometimes we want some weights to NOT learn
# set their gradient to 0

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])
goal = 0.8
lr = 0.1

# mask: 1 = learn, 0 = freeze
freeze_mask = np.array([1, 1, 0])  # freeze last weight

pred = inputs @ weights
delta = pred - goal
gradients = delta * inputs

# apply mask
gradients = gradients * freeze_mask

print(f"raw gradients: {delta * inputs}")
print(f"masked gradients: {gradients}")
print()
print("last weight wont change!")
