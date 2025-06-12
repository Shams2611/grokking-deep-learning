# full network function

import numpy as np

def neural_network(inputs, weights):
    """multi-input multi-output neural net"""
    return weights @ inputs

# setup
inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1],
])

# predict
predictions = neural_network(inputs, weights)

print("hurt:", predictions[0])
print("win:", predictions[1])
print("sad:", predictions[2])
