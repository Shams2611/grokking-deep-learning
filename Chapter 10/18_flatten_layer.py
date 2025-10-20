# flatten - convert 2D/3D to 1D for dense layer

import numpy as np

# simulated output from conv layers
# shape: (num_filters, height, width)
conv_output = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]],
])

print("conv output shape:", conv_output.shape)
print("(3 filters, 2x2 each)")
print()
print(conv_output)
print()

# flatten
flattened = conv_output.flatten()
print("flattened shape:", flattened.shape)
print(flattened)
print()

# or use reshape
flattened2 = conv_output.reshape(-1)
print("same with reshape(-1):", flattened2)
print()
print("now can feed to dense layer!")
