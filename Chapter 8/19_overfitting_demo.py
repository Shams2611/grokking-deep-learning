# demonstrating overfitting vs regularization

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

# noisy data
X = np.random.randn(20, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
y += np.random.randn(20, 1) * 0.1  # add noise

X_train, X_test = X[:15], X[15:]
y_train, y_test = y[:15], y[15:]

def train(use_l2=False, lambda_reg=0.01):
    w1 = np.random.randn(2, 16) * 0.5
    w2 = np.random.randn(16, 1) * 0.5
    lr = 0.1

    for epoch in range(500):
        for i in range(len(X_train)):
            h = relu(X_train[i] @ w1)
            out = h @ w2
            d_out = out - y_train[i]
            d_h = d_out @ w2.T * relu_deriv(X_train[i] @ w1)

            if use_l2:
                w2 = w2 * (1 - lr * lambda_reg) - lr * np.outer(h, d_out)
                w1 = w1 * (1 - lr * lambda_reg) - lr * np.outer(X_train[i], d_h)
            else:
                w2 -= lr * np.outer(h, d_out)
                w1 -= lr * np.outer(X_train[i], d_h)

    # compute errors
    train_err = sum((relu(x @ w1) @ w2 - t)**2 for x, t in zip(X_train, y_train))[0] / len(X_train)
    test_err = sum((relu(x @ w1) @ w2 - t)**2 for x, t in zip(X_test, y_test))[0] / len(X_test)
    return train_err, test_err

print("without regularization:")
train_e, test_e = train(use_l2=False)
print(f"  train error: {train_e:.4f}")
print(f"  test error: {test_e:.4f}")
print(f"  gap: {test_e - train_e:.4f}")

print()
print("with L2 regularization:")
train_e, test_e = train(use_l2=True)
print(f"  train error: {train_e:.4f}")
print(f"  test error: {test_e:.4f}")
print(f"  gap: {test_e - train_e:.4f}")
