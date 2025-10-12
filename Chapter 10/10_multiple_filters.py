# multiple filters = multiple feature maps

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

# image
image = np.random.randn(8, 8)

# multiple filters
filters = [
    np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # vertical
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),  # horizontal
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,  # blur
]
filter_names = ["vertical", "horizontal", "blur"]

print("applying multiple filters:")
print()

outputs = []
for kernel, name in zip(filters, filter_names):
    out = convolve2d(image, kernel)
    outputs.append(out)
    print(f"{name} filter -> output shape: {out.shape}")

# stack into feature maps
feature_maps = np.stack(outputs)
print()
print(f"stacked feature maps shape: {feature_maps.shape}")
print("(num_filters, height, width)")
