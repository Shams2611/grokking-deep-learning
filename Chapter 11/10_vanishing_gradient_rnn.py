# vanishing gradient in RNNs

import numpy as np

print("VANISHING GRADIENT IN RNNs")
print()

# gradient flows through W_hh and tanh' at each step
# max tanh' = 1.0, but typically < 1

# if |W_hh eigenvalues| < 1 and tanh' < 1
# gradient shrinks exponentially!

W_hh_small = np.array([[0.5, 0.1], [0.1, 0.5]])
tanh_deriv_max = 1.0

# gradient after T steps roughly proportional to:
print("gradient magnitude through time:")
print()

gradient = 1.0
for t in range(10):
    # simplified: gradient *= largest_eigenvalue * tanh_deriv
    gradient *= 0.6 * 0.8  # typical values
    print(f"  t={t+1}: gradient ~ {gradient:.6f}")

print()
print("gradient nearly disappears!")
print("hard to learn long-range dependencies")
print()
print("solutions:")
print("  - gradient clipping")
print("  - better architectures (LSTM, GRU)")
