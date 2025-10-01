# computing accuracy for multiclass

import numpy as np

def accuracy(predictions, targets):
    """
    predictions: (n_samples, n_classes) probabilities
    targets: (n_samples,) true class indices
    """
    predicted_classes = np.argmax(predictions, axis=1)
    correct = np.sum(predicted_classes == targets)
    return correct / len(targets)

# example predictions
predictions = np.array([
    [0.8, 0.1, 0.1],  # predicts class 0
    [0.2, 0.7, 0.1],  # predicts class 1
    [0.1, 0.2, 0.7],  # predicts class 2
    [0.6, 0.3, 0.1],  # predicts class 0
    [0.3, 0.4, 0.3],  # predicts class 1
])

targets = np.array([0, 1, 2, 1, 1])  # true classes

print("predictions (argmax):", np.argmax(predictions, axis=1))
print("targets:", targets)
print()

acc = accuracy(predictions, targets)
print(f"accuracy: {acc*100:.1f}%")
print(f"({int(acc * len(targets))}/{len(targets)} correct)")
