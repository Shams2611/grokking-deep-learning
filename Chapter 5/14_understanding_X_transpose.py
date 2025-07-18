# why X.T @ errors works

import numpy as np

# X shape: (n_samples, n_features)
# errors shape: (n_samples,)
# we want gradients shape: (n_features,)

# X.T @ errors:
# - X.T is (n_features, n_samples)
# - errors is (n_samples,)
# - result is (n_features,)

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])
errors = np.array([0.1, 0.2, 0.3])

gradients = X.T @ errors

print("X.T:")
print(X.T)
print()
print(f"errors: {errors}")
print(f"gradients: {gradients}")
print()
print("gradient[0] = 1*0.1 + 3*0.2 + 5*0.3 = 2.2")
