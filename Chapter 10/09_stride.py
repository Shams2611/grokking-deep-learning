# stride - how far to move the kernel each step

import numpy as np

def convolve2d_stride(image, kernel, stride=1):
    ih, iw = image.shape
    kh, kw = kernel.shape

    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            ii, jj = i * stride, j * stride
            output[i, j] = np.sum(image[ii:ii+kh, jj:jj+kw] * kernel)

    return output

image = np.arange(36).reshape(6, 6)
kernel = np.ones((2, 2))

print("image (6x6):")
print(image)
print()

for stride in [1, 2, 3]:
    out = convolve2d_stride(image, kernel, stride=stride)
    print(f"stride={stride} -> output shape: {out.shape}")

print()
print("larger stride = smaller output = faster")
