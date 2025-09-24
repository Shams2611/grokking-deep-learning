# forward pass for classification

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def cross_entropy(pred, target_idx):
    return -np.log(pred[target_idx] + 1e-15)

# network: 2 inputs -> 4 hidden -> 3 outputs (3 classes)
x = np.array([0.5, 0.8])
w1 = np.random.randn(2, 4) * 0.5
w2 = np.random.randn(4, 3) * 0.5
target = 0  # correct class

# forward pass
hidden = relu(x @ w1)
logits = hidden @ w2
probs = softmax(logits)
loss = cross_entropy(probs, target)

print("FORWARD PASS:")
print(f"  input: {x}")
print(f"  hidden: {np.round(hidden, 3)}")
print(f"  logits: {np.round(logits, 3)}")
print(f"  softmax: {np.round(probs, 3)}")
print(f"  target class: {target}")
print(f"  loss: {loss:.4f}")
