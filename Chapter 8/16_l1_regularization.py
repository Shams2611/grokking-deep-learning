# L1 regularization = sum of absolute weights
# encourages sparsity (many weights become 0)

import numpy as np

print("L1 vs L2 REGULARIZATION")
print()

weights = np.array([0.1, 0.5, -2.0, 3.0])
lambda_reg = 0.1

l1_penalty = lambda_reg * np.sum(np.abs(weights))
l2_penalty = lambda_reg * np.sum(weights**2)

print(f"weights: {weights}")
print(f"L1 penalty: {l1_penalty:.2f}")
print(f"L2 penalty: {l2_penalty:.2f}")
print()

# L1 gradient
l1_grad = lambda_reg * np.sign(weights)
l2_grad = 2 * lambda_reg * weights

print("gradients from regularization:")
print(f"L1: {l1_grad}")
print(f"L2: {np.round(l2_grad, 2)}")
print()
print("L1: constant push toward 0 (can reach exactly 0)")
print("L2: proportional push (never quite reaches 0)")
