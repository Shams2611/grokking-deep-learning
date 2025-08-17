# leaky ReLU - fixes dead neuron problem
# small slope for negative values

import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

x = np.array([-3, -1, 0, 1, 3])

print("leaky ReLU (alpha=0.01):")
for val in x:
    print(f"  leaky_relu({val:2}) = {leaky_relu(val):5.2f}")
print()
print("negative values get small gradient")
print("neurons can recover!")
