# training a classifier

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# simple 3-class dataset
X = np.array([
    [0.1, 0.9], [0.2, 0.8], [0.15, 0.85],  # class 0
    [0.9, 0.1], [0.8, 0.2], [0.85, 0.15],  # class 1
    [0.5, 0.5], [0.4, 0.6], [0.6, 0.4],    # class 2
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# one-hot encode
y_onehot = np.zeros((len(y), 3))
for i, label in enumerate(y):
    y_onehot[i, label] = 1

# network
w1 = np.random.randn(2, 8) * 0.5
w2 = np.random.randn(8, 3) * 0.5
lr = 0.5

print("training classifier:")
for epoch in range(200):
    total_loss = 0
    correct = 0

    for i in range(len(X)):
        # forward
        h_raw = X[i] @ w1
        h = relu(h_raw)
        logits = h @ w2
        probs = softmax(logits)

        # loss
        total_loss += -np.log(probs[y[i]] + 1e-15)
        if np.argmax(probs) == y[i]:
            correct += 1

        # backward
        d_logits = probs - y_onehot[i]
        d_h = d_logits @ w2.T * relu_deriv(h_raw)

        # update
        w2 -= lr * np.outer(h, d_logits)
        w1 -= lr * np.outer(X[i], d_h)

    if epoch % 40 == 0:
        acc = correct / len(X) * 100
        print(f"  epoch {epoch}: loss={total_loss:.3f}, acc={acc:.0f}%")
