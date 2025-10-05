# kernel/filter - small matrix that detects features

import numpy as np

# vertical edge detector kernel
vertical_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])

# horizontal edge detector kernel
horizontal_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
])

print("KERNELS (filters):")
print()
print("vertical edge detector:")
print(vertical_kernel)
print()
print("horizontal edge detector:")
print(horizontal_kernel)
print()
print("kernel slides across image")
print("detects specific patterns")
