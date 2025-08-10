# sigmoid derivative
# nice property: s'(x) = s(x) * (1 - s(x))

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

x = np.array([-2, -1, 0, 1, 2])

print("x:", x)
print("sigmoid(x):", np.round(sigmoid(x), 3))
print("sigmoid'(x):", np.round(sigmoid_derivative(x), 3))
print()
print("max derivative at x=0 (0.25)")
print("derivative shrinks at extremes -> vanishing gradient!")
