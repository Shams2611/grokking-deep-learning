# tanh = hyperbolic tangent
# like sigmoid but outputs -1 to 1

import numpy as np

def tanh(x):
    return np.tanh(x)

x = np.array([-3, -1, 0, 1, 3])

print("tanh outputs: -1 to 1")
print()
for val in x:
    print(f"tanh({val:2}) = {tanh(val):6.3f}")
print()
print("centered around 0 (unlike sigmoid)")
print("often works better than sigmoid")
