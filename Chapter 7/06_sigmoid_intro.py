# sigmoid squashes values to 0-1 range
# good for probabilities

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-5, -2, 0, 2, 5])

print("sigmoid: 1 / (1 + e^-x)")
print()
for val in x:
    print(f"sigmoid({val:2}) = {sigmoid(val):.4f}")
print()
print("big negative -> ~0")
print("zero -> 0.5")
print("big positive -> ~1")
