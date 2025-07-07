# gradient for each weight

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])
goal = 0.8

pred = inputs @ weights
delta = pred - goal

# gradient for weight[i] = delta * input[i]
gradients = delta * inputs

print(f"delta: {delta:.4f}")
print()
print("gradients for each weight:")
for i, (inp, g) in enumerate(zip(inputs, gradients)):
    print(f"  weight[{i}]: delta * {inp} = {g:.4f}")
