# full backprop

import numpy as np
np.random.seed(42)

input = np.array([1, 0, 1])
weights_0_1 = np.random.randn(3, 4) * 0.1
weights_1_2 = np.random.randn(4, 1) * 0.1
goal = np.array([1.0])
lr = 0.1

# FORWARD
hidden = input @ weights_0_1
output = hidden @ weights_1_2

# BACKWARD
# output layer
delta_output = output - goal
grad_1_2 = np.outer(hidden, delta_output)

# hidden layer
delta_hidden = delta_output @ weights_1_2.T
grad_0_1 = np.outer(input, delta_hidden)

# UPDATE
weights_1_2 = weights_1_2 - lr * grad_1_2
weights_0_1 = weights_0_1 - lr * grad_0_1

print("gradients computed and weights updated!")
print(f"grad_0_1 shape: {grad_0_1.shape}")
print(f"grad_1_2 shape: {grad_1_2.shape}")
