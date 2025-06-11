# matrix @ vector = vector of outputs

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])

weights = np.array([
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1],
])

# matrix vector multiply
predictions = weights @ inputs
# or: predictions = weights.dot(inputs)

print(f"inputs: {inputs}")
print(f"predictions: {predictions}")
print()
print("each output is dot product of its weight row with inputs")
