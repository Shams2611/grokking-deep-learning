# L2 regularization gradient
# adds extra term to weight updates

import numpy as np

print("L2 GRADIENT")
print()
print("loss = original_loss + lambda * sum(w^2)")
print("d_loss/d_w = d_original/d_w + 2 * lambda * w")
print()
print("weight update becomes:")
print("w = w - lr * (gradient + 2 * lambda * w)")
print("w = w - lr * gradient - lr * 2 * lambda * w")
print("w = w * (1 - lr * 2 * lambda) - lr * gradient")
print()
print("this is why its called 'weight decay'!")
print("weights shrink by factor (1 - lr * 2 * lambda) each step")

# example
w = 1.0
lr = 0.1
lambda_reg = 0.01
gradient = 0.2

w_new = w * (1 - lr * 2 * lambda_reg) - lr * gradient
print()
print(f"w before: {w}")
print(f"w after: {w_new:.4f}")
print(f"decay factor: {1 - lr * 2 * lambda_reg}")
