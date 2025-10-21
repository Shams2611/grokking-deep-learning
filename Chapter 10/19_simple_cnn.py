# simple CNN forward pass

import numpy as np

def relu(x): return np.maximum(0, x)

def conv2d(x, kernel):
    ih, iw = x.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i:i+kh, j:j+kw] * kernel)
    return out

def maxpool(x, size=2):
    h, w = x.shape
    oh, ow = h // size, w // size
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.max(x[i*size:(i+1)*size, j*size:(j+1)*size])
    return out

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# simple CNN
np.random.seed(42)

# 8x8 image
image = np.random.randn(8, 8)

# conv layer (2 filters)
filters = [np.random.randn(3, 3) * 0.5 for _ in range(2)]

# forward pass
print("Simple CNN forward pass:")
print(f"input: {image.shape}")

# conv + relu
conv_out = [relu(conv2d(image, f)) for f in filters]
conv_stack = np.stack(conv_out)
print(f"after conv+relu: {conv_stack.shape}")

# pooling
pool_out = np.stack([maxpool(c) for c in conv_out])
print(f"after maxpool: {pool_out.shape}")

# flatten
flat = pool_out.flatten()
print(f"after flatten: {flat.shape}")

# dense layer (3 classes)
w_dense = np.random.randn(flat.shape[0], 3) * 0.5
logits = flat @ w_dense
probs = softmax(logits)
print(f"output probs: {np.round(probs, 3)}")
