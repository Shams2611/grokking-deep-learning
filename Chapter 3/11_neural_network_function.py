# wrap it in a neural_network function

import numpy as np

def neural_network(inputs, weights):
    return inputs.dot(weights)

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])

pred = neural_network(inputs, weights)
print(f"prediction: {pred:.3f}")

# thats it! simplest neural net
