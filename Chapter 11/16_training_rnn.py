# training an RNN

import numpy as np

def tanh(x): return np.tanh(x)
def tanh_deriv(h): return 1 - h**2  # derivative given output

np.random.seed(42)

# simple sequence prediction: predict next number
# [0,1,2] -> [1,2,3]

input_size = 1
hidden_size = 10
output_size = 1

W_xh = np.random.randn(input_size, hidden_size) * 0.1
W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
W_hy = np.random.randn(hidden_size, output_size) * 0.1

lr = 0.01

# training data
X = np.array([[0], [1], [2], [3], [4]])  # input sequence
Y = np.array([[1], [2], [3], [4], [5]])  # targets (next number)

print("training RNN to predict next number:")

for epoch in range(100):
    # forward
    h = np.zeros(hidden_size)
    h_list, y_list = [h], []

    for t in range(len(X)):
        h = tanh(X[t] @ W_xh + h @ W_hh)
        y = h @ W_hy
        h_list.append(h)
        y_list.append(y)

    # loss
    loss = sum((y - t)**2 for y, t in zip(y_list, Y))[0] / len(Y)

    # backward (simplified)
    d_W_hy = np.zeros_like(W_hy)
    d_W_xh = np.zeros_like(W_xh)
    d_W_hh = np.zeros_like(W_hh)
    d_h_next = np.zeros(hidden_size)

    for t in reversed(range(len(X))):
        d_y = 2 * (y_list[t] - Y[t]) / len(Y)
        d_W_hy += np.outer(h_list[t+1], d_y)
        d_h = d_y @ W_hy.T + d_h_next
        d_h_raw = d_h * tanh_deriv(h_list[t+1])
        d_W_xh += np.outer(X[t], d_h_raw)
        d_W_hh += np.outer(h_list[t], d_h_raw)
        d_h_next = d_h_raw @ W_hh.T

    # update
    W_xh -= lr * d_W_xh
    W_hh -= lr * d_W_hh
    W_hy -= lr * d_W_hy

    if epoch % 20 == 0:
        print(f"  epoch {epoch}: loss = {loss:.4f}")
