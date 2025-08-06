# ReLU = Rectified Linear Unit
# most popular activation function today

import numpy as np

def relu(x):
    return np.maximum(0, x)

# test values
x = np.array([-3, -1, 0, 1, 3])

print("ReLU: max(0, x)")
print()
for val in x:
    print(f"relu({val:2}) = {relu(val)}")
