# convolution backward pass

import numpy as np

def conv_forward(x, kernel):
    """forward pass"""
    ih, iw = x.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(x[i:i+kh, j:j+kw] * kernel)
    return output

def conv_backward(x, kernel, d_out):
    """backward pass - gradient w.r.t kernel"""
    ih, iw = x.shape
    kh, kw = kernel.shape
    oh, ow = d_out.shape

    d_kernel = np.zeros_like(kernel)

    for i in range(oh):
        for j in range(ow):
            patch = x[i:i+kh, j:j+kw]
            d_kernel += patch * d_out[i, j]

    return d_kernel

# test
np.random.seed(42)
x = np.random.randn(5, 5)
kernel = np.random.randn(3, 3)

# forward
output = conv_forward(x, kernel)
print(f"input: {x.shape}, kernel: {kernel.shape}, output: {output.shape}")

# backward (pretend gradient from next layer)
d_out = np.ones_like(output)
d_kernel = conv_backward(x, kernel, d_out)

print(f"d_kernel shape: {d_kernel.shape}")
print("gradient computed!")
