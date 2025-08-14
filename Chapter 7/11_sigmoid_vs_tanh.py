# comparing sigmoid and tanh

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.array([-2, -1, 0, 1, 2])

print("x:", x)
print("sigmoid:", np.round(sigmoid(x), 3))
print("tanh:", np.round(tanh(x), 3))
print()
print("sigmoid: 0 to 1")
print("tanh: -1 to 1")
print()
print("fun fact: tanh(x) = 2*sigmoid(2x) - 1")
