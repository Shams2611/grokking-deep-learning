# one-hot encoding for labels
# represents class as vector

import numpy as np

print("ONE-HOT ENCODING")
print()
print("classes: cat=0, dog=1, bird=2")
print()

# label -> one-hot
def to_one_hot(label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

print("cat (0):", to_one_hot(0, 3))
print("dog (1):", to_one_hot(1, 3))
print("bird (2):", to_one_hot(2, 3))
print()

# useful for computing loss
prediction = np.array([0.7, 0.2, 0.1])
target = to_one_hot(0, 3)  # cat
print("prediction:", prediction)
print("target:", target)
print("correct class prob:", prediction[0])
