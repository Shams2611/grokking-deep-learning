# dead ReLU problem
# if a neuron always outputs negative, it never learns

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# neuron with bad weights
weight = -2.0
inputs = np.array([0.1, 0.5, 0.3, 0.8])

print("inputs:", inputs)
print("weight:", weight)
print()

for x in inputs:
    raw = x * weight
    activated = relu(raw)
    gradient = relu_derivative(raw)
    print(f"input {x}: raw={raw:.2f}, relu={activated}, grad={gradient}")

print()
print("all gradients are 0!")
print("neuron is 'dead' - can never recover")
