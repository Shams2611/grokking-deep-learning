# vanishing gradient problem with sigmoid
# gradients get tiny in deep networks

import numpy as np

def sigmoid_derivative_at(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

# max sigmoid derivative is 0.25
# after many layers, gradient shrinks!

print("gradient through layers (sigmoid):")
gradient = 1.0
for layer in range(10):
    gradient *= 0.25  # max sigmoid derivative
    print(f"  layer {layer+1}: gradient = {gradient:.10f}")

print()
print("gradient basically disappears!")
print("this is why ReLU became popular")
