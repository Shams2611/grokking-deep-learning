# L2 regularization = penalize large weights
# also called "weight decay"

import numpy as np

print("L2 REGULARIZATION")
print()
print("idea: add penalty for large weights to loss")
print()
print("loss = original_loss + lambda * sum(weights^2)")
print()

# example
weights = np.array([0.1, 0.5, 2.0, 3.0])
original_loss = 0.5
lambda_reg = 0.01

penalty = lambda_reg * np.sum(weights**2)
total_loss = original_loss + penalty

print(f"weights: {weights}")
print(f"original loss: {original_loss}")
print(f"L2 penalty: {penalty:.4f}")
print(f"total loss: {total_loss:.4f}")
print()
print("large weights (2.0, 3.0) contribute more to penalty")
print("encourages smaller, distributed weights")
