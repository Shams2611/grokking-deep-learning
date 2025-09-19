# cross-entropy loss function

import numpy as np

def cross_entropy(predictions, targets):
    """
    predictions: softmax outputs (probabilities)
    targets: one-hot encoded labels
    """
    # small epsilon to avoid log(0)
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)

    # -sum(target * log(prediction))
    return -np.sum(targets * np.log(predictions))

# examples
pred1 = np.array([0.9, 0.05, 0.05])  # confident right
pred2 = np.array([0.1, 0.8, 0.1])    # confident wrong
target = np.array([1, 0, 0])          # true class is 0

print("target:", target)
print()
print("prediction [0.9, 0.05, 0.05]:")
print(f"  loss: {cross_entropy(pred1, target):.4f}")
print()
print("prediction [0.1, 0.8, 0.1]:")
print(f"  loss: {cross_entropy(pred2, target):.4f}")
