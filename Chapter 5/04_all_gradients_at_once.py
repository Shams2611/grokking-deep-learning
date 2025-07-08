# compute all gradients at once

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])
goal = 0.8

pred = inputs @ weights
delta = pred - goal

# all gradients in one line!
gradients = delta * inputs

print(f"gradients: {gradients}")

# this is element-wise multiply
# delta gets broadcast to match inputs shape
