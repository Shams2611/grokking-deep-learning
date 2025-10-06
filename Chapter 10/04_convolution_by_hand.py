# convolution by hand - one position

import numpy as np

# 3x3 image patch
patch = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

# 3x3 kernel
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])

# convolution = element-wise multiply then sum
result = np.sum(patch * kernel)

print("image patch:")
print(patch)
print()
print("kernel:")
print(kernel)
print()
print("element-wise multiply:")
print(patch * kernel)
print()
print("sum:", result)
