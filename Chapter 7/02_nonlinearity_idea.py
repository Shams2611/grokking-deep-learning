# nonlinearity breaks the linear collapse
# lets the network learn complex patterns

import numpy as np

# simple nonlinear function: max with 0
def simple_nonlinear(x):
    return np.maximum(0, x)

x = np.array([-2, -1, 0, 1, 2])

print("input:", x)
print("after nonlinearity:", simple_nonlinear(x))
print()
print("negative values -> 0")
print("positive values -> unchanged")
