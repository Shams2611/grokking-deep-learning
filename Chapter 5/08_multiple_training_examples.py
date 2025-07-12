# multiple training examples

import numpy as np

# 4 training examples
X = np.array([
    [8.5, 0.65, 1.2],
    [9.5, 0.80, 1.3],
    [9.9, 0.80, 0.5],
    [9.0, 0.90, 1.0],
])

goals = np.array([1, 1, 0, 1])

weights = np.array([0.1, 0.2, -0.1])

print("4 examples, same weights:")
for i in range(4):
    pred = X[i] @ weights
    print(f"  example {i}: pred={pred:.3f}, goal={goals[i]}")
