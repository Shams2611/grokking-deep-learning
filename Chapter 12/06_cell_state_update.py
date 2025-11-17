# updating cell state

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)

print("CELL STATE UPDATE")
print()
print("c_t = f_t * c_{t-1} + i_t * c_tilde")
print()
print("forget old + add new")
print()

# example
c_prev = np.array([1.0, 2.0, -1.0, 0.5])  # previous cell state

# gates (pretend we computed these)
f = np.array([0.9, 0.1, 0.5, 0.8])  # forget gate
i = np.array([0.2, 0.9, 0.5, 0.3])  # input gate
c_tilde = np.array([0.5, 1.0, -0.5, 0.2])  # candidates

print("previous cell state:", c_prev)
print("forget gate:", f)
print("input gate:", i)
print("candidates:", c_tilde)
print()

# update
c_new = f * c_prev + i * c_tilde

print("new cell state:", np.round(c_new, 3))
print()
print("some memories kept (f~1)")
print("some forgotten (f~0)")
print("new info added (i * c_tilde)")
