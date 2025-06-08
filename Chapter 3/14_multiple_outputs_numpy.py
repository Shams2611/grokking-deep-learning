# multiple outputs with numpy

import numpy as np

input = 8.5
weights = np.array([0.1, 0.2, 0.3])

predictions = input * weights

print(f"predictions: {predictions}")

# element-wise multiplication
# input gets multiplied by each weight
