# input normalization

import numpy as np

# problem: inputs on different scales
inputs = np.array([1000, 0.5, 50])

# gradient = delta * input
# big input = big gradient = unstable!

delta = 0.1
gradients = delta * inputs

print("unnormalized inputs:", inputs)
print("gradients:", gradients)
print("weight for input[0] changes 2000x more!")
print()

# solution: normalize inputs
inputs_norm = (inputs - inputs.mean()) / inputs.std()
gradients_norm = delta * inputs_norm

print("normalized inputs:", inputs_norm.round(3))
print("gradients:", gradients_norm.round(3))
print("much more balanced!")
