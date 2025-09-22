# softmax derivative
# needed for backprop

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# softmax derivative is complex...
# but with cross-entropy, it simplifies!

print("SOFTMAX DERIVATIVE")
print()
print("full derivative is a Jacobian matrix:")
print("  d_softmax_i/d_x_j = s_i * (1 - s_j) if i == j")
print("                    = -s_i * s_j     if i != j")
print()

s = softmax(np.array([1.0, 2.0, 0.5]))
print(f"softmax output: {np.round(s, 3)}")
print()

# Jacobian
n = len(s)
jacobian = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            jacobian[i, j] = s[i] * (1 - s[j])
        else:
            jacobian[i, j] = -s[i] * s[j]

print("Jacobian matrix:")
print(np.round(jacobian, 3))
