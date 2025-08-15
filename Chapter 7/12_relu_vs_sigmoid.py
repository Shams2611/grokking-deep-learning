# why ReLU often beats sigmoid

import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-2, 0, 2, 5, 10])

print("x:", x)
print("relu:", relu(x))
print("sigmoid:", np.round(sigmoid(x), 4))
print()
print("ReLU advantages:")
print("  1. no vanishing gradient (for positive values)")
print("  2. computationally cheaper (no exp)")
print("  3. sparse activation (some neurons off)")
