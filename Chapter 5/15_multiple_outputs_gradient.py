# gradient with multiple outputs

import numpy as np

# single input, multiple outputs
input = np.array([8.5, 0.65, 1.2])
# weight matrix: each row is weights for one output
weights = np.array([
    [0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2],
])
goals = np.array([1, 0])

preds = weights @ input  # shape: (2,)
deltas = preds - goals   # shape: (2,)

# gradient for each weight
# gradient[i,j] = delta[i] * input[j]
gradients = np.outer(deltas, input)

print(f"preds: {preds}")
print(f"deltas: {deltas}")
print()
print("gradients:")
print(gradients)
