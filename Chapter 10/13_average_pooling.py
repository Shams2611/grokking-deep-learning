# average pooling - take mean instead of max

import numpy as np

def avg_pool2d(x, pool_size=2):
    h, w = x.shape
    ph, pw = pool_size, pool_size

    oh = h // ph
    ow = w // pw

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            patch = x[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            output[i, j] = np.mean(patch)

    return output

x = np.array([
    [1, 3, 2, 4],
    [5, 6, 8, 7],
    [9, 2, 1, 5],
    [3, 4, 6, 8],
])

print("input (4x4):")
print(x)
print()

# compare max vs average
max_out = np.array([[6, 8], [9, 8]])  # from previous
avg_out = avg_pool2d(x, pool_size=2)

print("max pooling:")
print(max_out)
print()
print("average pooling:")
print(avg_out)
print()
print("max pool: keeps strongest signal")
print("avg pool: smoother, preserves more info")
