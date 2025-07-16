# full batch training example

import numpy as np
np.random.seed(42)

# data
X = np.random.randn(100, 3)
true_weights = np.array([0.5, -0.3, 0.8])
y = X @ true_weights + np.random.randn(100) * 0.1

# train
weights = np.zeros(3)
lr = 0.01

for epoch in range(100):
    preds = X @ weights
    errors = preds - y
    gradients = X.T @ errors / len(X)
    weights = weights - lr * gradients

print(f"true weights: {true_weights}")
print(f"learned weights: {weights.round(3)}")
