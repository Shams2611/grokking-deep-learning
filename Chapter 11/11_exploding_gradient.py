# exploding gradients and gradient clipping

import numpy as np

print("EXPLODING GRADIENTS")
print()
print("if |W_hh eigenvalues| > 1, gradients explode!")
print()

gradient = 1.0
for t in range(10):
    gradient *= 1.5  # eigenvalue > 1
    print(f"  t={t+1}: gradient ~ {gradient:.1f}")

print()
print("gradient explodes exponentially!")
print()

# solution: gradient clipping
def clip_gradient(grad, max_norm=5.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * max_norm / norm
    return grad

print("GRADIENT CLIPPING")
big_gradient = np.array([100, 200, 300])
clipped = clip_gradient(big_gradient, max_norm=10)

print(f"  before: {big_gradient}, norm={np.linalg.norm(big_gradient):.1f}")
print(f"  after:  {np.round(clipped, 2)}, norm={np.linalg.norm(clipped):.1f}")
