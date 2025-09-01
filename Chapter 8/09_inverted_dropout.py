# inverted dropout - scale during training instead
# simpler: test code stays the same

import numpy as np

def dropout_train(x, keep_prob):
    """apply dropout and scale up"""
    mask = (np.random.rand(*x.shape) < keep_prob)
    return x * mask / keep_prob  # scale up!

def dropout_test(x):
    """no dropout at test time"""
    return x  # unchanged!

np.random.seed(42)
neurons = np.array([1.0, 1.0, 1.0, 1.0])
keep_prob = 0.5

print("INVERTED DROPOUT:")
print()
print("training:")
result = dropout_train(neurons, keep_prob)
print(f"  input: {neurons}")
print(f"  output: {result}")
print(f"  (dropped neurons=0, kept neurons scaled up)")

print()
print("testing:")
result = dropout_test(neurons)
print(f"  input: {neurons}")
print(f"  output: {result}")
print(f"  (no changes needed!)")
