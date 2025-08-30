# creating dropout masks

import numpy as np

def create_dropout_mask(size, keep_prob):
    """keep_prob = probability of KEEPING a neuron"""
    mask = np.random.rand(size) < keep_prob
    return mask.astype(float)

np.random.seed(0)

print("dropout masks (keep_prob=0.8):")
for i in range(5):
    mask = create_dropout_mask(8, 0.8)
    print(f"  trial {i+1}: {mask.astype(int)}")

print()
print("each time different neurons are dropped")
print("keeps ~80% of neurons active")
