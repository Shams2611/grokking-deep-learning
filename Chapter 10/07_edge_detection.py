# edge detection with convolution

import numpy as np

def convolve2d(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

# image with vertical edge
image = np.array([
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
])

# vertical edge detector
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])

output = convolve2d(image, kernel)

print("image (has vertical edge in middle):")
print(image)
print()
print("vertical edge detector:")
print(kernel)
print()
print("output (edge detected!):")
print(output.astype(int))
print()
print("high values = edge found!")
