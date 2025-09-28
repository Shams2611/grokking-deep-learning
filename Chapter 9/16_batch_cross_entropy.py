# batch cross-entropy loss

import numpy as np

def softmax_batch(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_batch(probs, targets):
    """
    probs: (batch_size, num_classes)
    targets: (batch_size,) class indices
    """
    batch_size = len(targets)
    # get probability of correct class for each sample
    correct_probs = probs[np.arange(batch_size), targets]
    # average negative log probability
    return -np.mean(np.log(correct_probs + 1e-15))

# batch
logits = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
    [0.1, 0.2, 3.0],
])
targets = np.array([0, 1, 2])  # correct classes

probs = softmax_batch(logits)
loss = cross_entropy_batch(probs, targets)

print("predictions (rows):")
print(np.round(probs, 3))
print()
print(f"targets: {targets}")
print(f"average loss: {loss:.4f}")
