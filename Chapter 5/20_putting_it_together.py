# putting it all together

import numpy as np
np.random.seed(42)

# generate data
n_samples = 100
n_features = 5

X = np.random.randn(n_samples, n_features)
true_w = np.array([0.5, -0.3, 0.8, 0.1, -0.6])
y = X @ true_w

# normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# train
weights = np.zeros(n_features)
lr = 0.1

for epoch in range(100):
    preds = X @ weights
    errors = preds - y
    gradients = X.T @ errors / n_samples
    weights = weights - lr * gradients

    if epoch % 20 == 0:
        mse = (errors ** 2).mean()
        print(f"epoch {epoch:3d}: MSE = {mse:.6f}")

print()
print(f"true weights: {true_w}")
print(f"learned:      {weights.round(3)}")
print()
print("CHAPTER 5 DONE - gradient descent with multiple inputs!")
