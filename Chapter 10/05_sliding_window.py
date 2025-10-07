# convolution = sliding the kernel across the image

import numpy as np

# 5x5 image
image = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
])

# 3x3 kernel (just sums the patch)
kernel = np.ones((3, 3))

print("image (5x5):")
print(image)
print()
print("kernel (3x3, all ones):")
print(kernel.astype(int))
print()

# manual convolution
output_size = 5 - 3 + 1  # 3
output = np.zeros((output_size, output_size))

for i in range(output_size):
    for j in range(output_size):
        patch = image[i:i+3, j:j+3]
        output[i, j] = np.sum(patch * kernel)

print(f"output ({output_size}x{output_size}):")
print(output.astype(int))
print()
print("output is smaller! (no padding)")
