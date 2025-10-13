# pooling - downsample to reduce size

import numpy as np

print("POOLING")
print()
print("why pool?")
print("  1. reduce computation")
print("  2. reduce overfitting")
print("  3. translation invariance")
print()

# example 4x4 feature map
feature_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
])

print("feature map (4x4):")
print(feature_map)
print()

# 2x2 max pooling
# take max of each 2x2 block
result = np.array([
    [max(1,2,5,6), max(3,4,7,8)],
    [max(9,10,13,14), max(11,12,15,16)],
])

print("after 2x2 max pooling:")
print(result)
print()
print("size reduced from 4x4 to 2x2!")
