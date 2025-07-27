# forward pass through 2 layers

import numpy as np
np.random.seed(42)

input = np.array([1, 0, 1])

# two sets of weights
weights_0_1 = np.random.randn(3, 4) * 0.1  # input to hidden
weights_1_2 = np.random.randn(4, 1) * 0.1  # hidden to output

# forward pass
hidden = input @ weights_0_1
output = hidden @ weights_1_2

print(f"input:  {input}")
print(f"hidden: {hidden.round(3)}")
print(f"output: {output.round(3)}")
