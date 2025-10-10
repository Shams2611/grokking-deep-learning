# padding - keep output same size as input

import numpy as np

def convolve2d_padded(image, kernel, padding=0):
    # add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1

    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

image = np.ones((4, 4))
kernel = np.ones((3, 3)) / 9  # average filter

print("image (4x4):")
print(image.astype(int))
print()

# without padding
out_no_pad = convolve2d_padded(image, kernel, padding=0)
print(f"no padding -> output shape: {out_no_pad.shape}")

# with padding=1
out_pad = convolve2d_padded(image, kernel, padding=1)
print(f"padding=1 -> output shape: {out_pad.shape}")
print()
print("same padding: pad = (kernel_size - 1) / 2")
