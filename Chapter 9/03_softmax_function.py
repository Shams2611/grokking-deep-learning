# softmax function implementation

import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

# test cases
print("softmax function:")
print()

test1 = np.array([1.0, 2.0, 3.0])
print(f"input: {test1}")
print(f"output: {np.round(softmax(test1), 3)}")
print()

test2 = np.array([1.0, 1.0, 1.0])
print(f"input: {test2}")
print(f"output: {np.round(softmax(test2), 3)}")
print("(equal inputs -> equal probabilities)")
print()

test3 = np.array([10.0, 0.0, 0.0])
print(f"input: {test3}")
print(f"output: {np.round(softmax(test3), 3)}")
print("(big difference -> almost certain)")
