# max pooling backward pass
# gradient flows only to the max element

import numpy as np

def maxpool_forward(x, pool_size=2):
    h, w = x.shape
    oh, ow = h // pool_size, w // pool_size

    output = np.zeros((oh, ow))
    mask = np.zeros_like(x)  # remember where max was

    for i in range(oh):
        for j in range(ow):
            patch = x[i*pool_size:(i+1)*pool_size,
                     j*pool_size:(j+1)*pool_size]
            max_val = np.max(patch)
            output[i, j] = max_val

            # mark location of max
            for pi in range(pool_size):
                for pj in range(pool_size):
                    if patch[pi, pj] == max_val:
                        mask[i*pool_size+pi, j*pool_size+pj] = 1
                        break

    return output, mask

def maxpool_backward(d_out, mask, pool_size=2):
    """gradient flows only through max positions"""
    h, w = mask.shape
    oh, ow = d_out.shape
    d_input = np.zeros_like(mask)

    for i in range(oh):
        for j in range(ow):
            d_input[i*pool_size:(i+1)*pool_size,
                   j*pool_size:(j+1)*pool_size] = \
                mask[i*pool_size:(i+1)*pool_size,
                    j*pool_size:(j+1)*pool_size] * d_out[i, j]

    return d_input

# test
x = np.array([[1, 3, 2, 4],
              [5, 6, 7, 8],
              [9, 2, 3, 1],
              [4, 5, 6, 7]], dtype=float)

out, mask = maxpool_forward(x)
print("input:")
print(x)
print("\nmax pool output:")
print(out)
print("\nmask (where max was):")
print(mask)
