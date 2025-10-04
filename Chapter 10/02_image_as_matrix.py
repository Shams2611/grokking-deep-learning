# images are just matrices of numbers

import numpy as np

# simple 5x5 grayscale image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

print("5x5 grayscale image (a square):")
print(image)
print()
print("0 = black, 1 = white")
print()
print("shape:", image.shape)
print("total pixels:", image.size)
