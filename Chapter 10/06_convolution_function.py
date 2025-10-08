# convolution function from scratch

import numpy as np

def convolve2d(image, kernel):
    """2D convolution without padding"""
    ih, iw = image.shape
    kh, kw = kernel.shape

    # output size
    oh = ih - kh + 1
    ow = iw - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            patch = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(patch * kernel)

    return output

# test
image = np.random.randint(0, 10, (6, 6))
kernel = np.array([[1, 0], [0, -1]])  # diagonal detector

print("image (6x6):")
print(image)
print()
print("kernel (2x2):")
print(kernel)
print()
print("output (5x5):")
print(convolve2d(image, kernel).astype(int))
