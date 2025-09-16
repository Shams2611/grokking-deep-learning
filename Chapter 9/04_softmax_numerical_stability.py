# numerical stability trick for softmax
# large values cause overflow in exp()

import numpy as np

def softmax_unstable(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

def softmax_stable(x):
    # subtract max for stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()

# large values
large = np.array([1000, 1001, 1002])

print("large values:", large)
print()

# unstable version
try:
    result = softmax_unstable(large)
    print("unstable:", result)
except:
    print("unstable: OVERFLOW ERROR!")

# stable version
result = softmax_stable(large)
print("stable:", np.round(result, 3))
print()
print("trick: subtract max(x) before exp")
print("doesnt change the result, just prevents overflow")
