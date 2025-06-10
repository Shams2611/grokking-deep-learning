# weight matrix
# rows = outputs, cols = inputs

import numpy as np

# weights[output][input]
weights = np.array([
    [0.1, 0.1, -0.3],  # weights for 'hurt' output
    [0.1, 0.2, 0.0],   # weights for 'win' output
    [0.0, 1.3, 0.1],   # weights for 'sad' output
])

print("weight matrix:")
print(weights)
print()
print("shape:", weights.shape)
print("3 outputs x 3 inputs")
