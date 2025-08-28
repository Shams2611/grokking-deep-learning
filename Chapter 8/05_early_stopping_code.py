# early stopping implementation

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

# data - split into train and validation
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]

# model
w1 = np.random.randn(2, 8) * 0.5
w2 = np.random.randn(8, 1) * 0.5
lr = 0.1

best_val_error = float('inf')
best_weights = (w1.copy(), w2.copy())
patience = 5
wait = 0

print("training with early stopping:")
for epoch in range(100):
    # train
    for i in range(len(X_train)):
        h = relu(X_train[i] @ w1)
        out = h @ w2
        d_out = out - y_train[i]
        d_h = d_out @ w2.T * relu_deriv(X_train[i] @ w1)
        w2 -= lr * np.outer(h, d_out)
        w1 -= lr * np.outer(X_train[i], d_h)

    # validation error
    val_error = 0
    for i in range(len(X_val)):
        h = relu(X_val[i] @ w1)
        out = h @ w2
        val_error += (out - y_val[i])**2

    val_error = val_error[0] / len(X_val)

    if val_error < best_val_error:
        best_val_error = val_error
        best_weights = (w1.copy(), w2.copy())
        wait = 0
    else:
        wait += 1

    if epoch % 10 == 0:
        print(f"  epoch {epoch}: val_error = {val_error:.4f}")

    if wait >= patience:
        print(f"\nstopped at epoch {epoch}!")
        break

print(f"best validation error: {best_val_error:.4f}")
