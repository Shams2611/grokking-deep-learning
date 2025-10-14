# max pooling implementation

import numpy as np

def max_pool2d(x, pool_size=2):
    h, w = x.shape
    ph, pw = pool_size, pool_size

    oh = h // ph
    ow = w // pw

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            patch = x[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            output[i, j] = np.max(patch)

    return output

# test
x = np.array([
    [1, 3, 2, 4],
    [5, 6, 8, 7],
    [9, 2, 1, 5],
    [3, 4, 6, 8],
])

print("input (4x4):")
print(x)
print()

pooled = max_pool2d(x, pool_size=2)
print("after 2x2 max pooling:")
print(pooled.astype(int))
print()
print("keeps the strongest activations")
