# simplified cross-entropy for single correct class
# only need -log(prob of correct class)

import numpy as np

def cross_entropy_simple(predictions, correct_class):
    """
    predictions: softmax outputs
    correct_class: index of correct class
    """
    eps = 1e-15
    prob = np.clip(predictions[correct_class], eps, 1 - eps)
    return -np.log(prob)

# examples
predictions = np.array([0.7, 0.2, 0.1])

print("predictions:", predictions)
print()

for correct in range(3):
    loss = cross_entropy_simple(predictions, correct)
    print(f"if correct class is {correct}: loss = {loss:.4f}")

print()
print("lowest loss when correct class has highest prob")
