# vectorized batch computation

import numpy as np

X = np.array([
    [8.5, 0.65, 1.2],
    [9.5, 0.80, 1.3],
    [9.9, 0.80, 0.5],
    [9.0, 0.90, 1.0],
])
y = np.array([1, 1, 0, 1])
weights = np.array([0.1, 0.2, -0.1])

# all predictions at once
preds = X @ weights  # shape: (4,)

# all errors at once
errors = preds - y  # shape: (4,)

# all gradients at once
# X.T @ errors gives sum of (error * input) for each weight
gradients = X.T @ errors / len(X)  # shape: (3,)

print(f"preds shape: {preds.shape}")
print(f"gradients shape: {gradients.shape}")
print(f"gradients: {gradients}")
