# the magic: softmax + cross-entropy gradient
# beautifully simple!

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

print("SOFTMAX + CROSS-ENTROPY GRADIENT")
print()
print("the gradient of CE loss w.r.t. logits is simply:")
print()
print("  gradient = softmax(logits) - one_hot_target")
print()
print("thats it! super clean.")
print()

# example
logits = np.array([2.0, 1.0, 0.1])
target = np.array([1, 0, 0])  # class 0

pred = softmax(logits)
gradient = pred - target

print(f"logits: {logits}")
print(f"softmax: {np.round(pred, 3)}")
print(f"target: {target}")
print(f"gradient: {np.round(gradient, 3)}")
print()
print("gradient pushes prediction toward target")
