# tanh derivative
# tanh'(x) = 1 - tanh(x)^2

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

x = np.array([-2, -1, 0, 1, 2])

print("x:", x)
print("tanh(x):", np.round(tanh(x), 3))
print("tanh'(x):", np.round(tanh_derivative(x), 3))
print()
print("max derivative at x=0 (1.0)")
print("still has vanishing gradient but less severe")
