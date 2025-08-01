# backprop hidden layer

import numpy as np
np.random.seed(42)

input = np.array([1, 0, 1])
weights_0_1 = np.random.randn(3, 4) * 0.1
weights_1_2 = np.random.randn(4, 1) * 0.1
goal = np.array([1.0])

# forward
hidden = input @ weights_0_1
output = hidden @ weights_1_2

# backward
delta_output = output - goal

# propagate error to hidden layer
# each hidden unit's "responsibility" for the error
delta_hidden = delta_output @ weights_1_2.T

print(f"delta_output shape: {delta_output.shape}")
print(f"delta_hidden shape: {delta_hidden.shape}")
print(f"delta_hidden: {delta_hidden}")
print()
print("delta_hidden tells us how to adjust hidden layer")
