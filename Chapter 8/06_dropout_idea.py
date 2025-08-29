# dropout = randomly turn off neurons during training
# forces network to not rely on any single neuron

import numpy as np

# example: layer with 5 neurons
neurons = np.array([0.5, 0.8, 0.3, 0.9, 0.2])
print("original neurons:", neurons)

# dropout with p=0.5 (50% chance to drop)
np.random.seed(42)
mask = np.random.rand(5) > 0.5
dropped = neurons * mask

print("dropout mask:", mask.astype(int))
print("after dropout:", dropped)
print()
print("some neurons are 'turned off' (set to 0)")
print("different neurons dropped each forward pass")
print("network cant rely on specific neurons")
