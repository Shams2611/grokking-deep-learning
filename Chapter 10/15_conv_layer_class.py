# conv layer as a class

import numpy as np

class Conv2D:
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # random filters
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) * 0.1

    def forward(self, x):
        self.input = x
        h, w = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        oh, ow = h - kh + 1, w - kw + 1

        output = np.zeros((self.num_filters, oh, ow))

        for f in range(self.num_filters):
            for i in range(oh):
                for j in range(ow):
                    patch = x[i:i+kh, j:j+kw]
                    output[f, i, j] = np.sum(patch * self.filters[f])

        return output

# test
np.random.seed(42)
conv = Conv2D(num_filters=3, kernel_size=3)

image = np.random.randn(8, 8)
output = conv.forward(image)

print(f"input shape: {image.shape}")
print(f"num filters: {conv.num_filters}")
print(f"kernel size: {conv.kernel_size}")
print(f"output shape: {output.shape}")
