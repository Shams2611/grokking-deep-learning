# backprop output layer

import numpy as np
np.random.seed(42)

# setup
hidden = np.array([0.1, 0.2, 0.3, 0.4])
weights_1_2 = np.random.randn(4, 1) * 0.1
goal = np.array([1.0])

# forward
output = hidden @ weights_1_2

# backward - output layer
delta_output = output - goal
gradient_1_2 = np.outer(hidden, delta_output)

print(f"output: {output}")
print(f"goal: {goal}")
print(f"delta_output: {delta_output}")
print()
print("gradient for weights_1_2:")
print(gradient_1_2)
