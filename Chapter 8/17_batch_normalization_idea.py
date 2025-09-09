# batch normalization idea
# normalize layer inputs to have mean=0, std=1

import numpy as np

# batch of activations
batch = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
print("original batch:", batch)
print(f"  mean: {batch.mean():.2f}")
print(f"  std: {batch.std():.2f}")

# normalize
mean = batch.mean()
std = batch.std()
normalized = (batch - mean) / (std + 1e-8)

print()
print("normalized batch:", np.round(normalized, 3))
print(f"  mean: {normalized.mean():.2f}")
print(f"  std: {normalized.std():.2f}")
print()
print("benefits:")
print("  - stabilizes training")
print("  - allows higher learning rates")
print("  - reduces sensitivity to initialization")
