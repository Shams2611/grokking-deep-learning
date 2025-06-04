# numpy makes this easy

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])

# one line!
prediction = inputs.dot(weights)
# or: prediction = np.dot(inputs, weights)
# or: prediction = inputs @ weights

print(f"prediction: {prediction:.3f}")
