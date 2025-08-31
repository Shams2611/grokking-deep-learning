# dropout scaling - important detail!
# at test time we use all neurons, so scale down

import numpy as np

neurons = np.array([1.0, 1.0, 1.0, 1.0])
keep_prob = 0.5

# training: drop 50%
np.random.seed(42)
mask = np.random.rand(4) < keep_prob
train_output = neurons * mask
print("TRAINING (with dropout):")
print("  neurons:", neurons)
print("  mask:", mask.astype(int))
print("  output:", train_output)
print("  sum:", train_output.sum())

# test: use all, but scale by keep_prob
test_output = neurons * keep_prob
print()
print("TESTING (no dropout, but scaled):")
print("  neurons:", neurons)
print("  output:", test_output)
print("  sum:", test_output.sum())

print()
print("both have same expected sum!")
