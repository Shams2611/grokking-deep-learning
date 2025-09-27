# batch softmax - multiple samples at once

import numpy as np

def softmax_batch(x):
    """softmax for batch of samples"""
    # x shape: (batch_size, num_classes)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# batch of 3 samples, 4 classes each
logits = np.array([
    [2.0, 1.0, 0.1, 0.5],
    [0.5, 2.5, 0.3, 0.1],
    [0.1, 0.2, 3.0, 0.5],
])

probs = softmax_batch(logits)

print("batch softmax:")
print()
print("logits:")
print(logits)
print()
print("probabilities:")
print(np.round(probs, 3))
print()
print("row sums:", np.round(probs.sum(axis=1), 3))
print("each row sums to 1!")
