# gates - controlling information flow

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("LSTM GATES")
print()
print("gates are sigmoid outputs (0 to 1)")
print("act like valves controlling flow")
print()

# example
values = np.array([1.0, 2.0, 3.0, 4.0])
gate = np.array([0.0, 0.5, 1.0, 0.2])

result = values * gate

print("values:", values)
print("gate:", gate)
print("values * gate:", result)
print()
print("gate=0: block completely")
print("gate=1: let through completely")
print("gate=0.5: let half through")
