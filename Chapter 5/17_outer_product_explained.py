# outer product explained

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5])

outer = np.outer(a, b)

print(f"a: {a}")
print(f"b: {b}")
print()
print("outer product:")
print(outer)
print()
print("outer[i,j] = a[i] * b[j]")
print()
print("for gradients:")
print("  delta is (n_outputs,)")
print("  input is (n_inputs,)")
print("  gradient is (n_outputs, n_inputs)")
