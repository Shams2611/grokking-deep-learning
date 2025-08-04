# why do we need activation functions?
# without them, stacking layers is pointless

import numpy as np

# two layers without activation = still linear!
w1 = np.array([[0.5, 0.2], [0.3, 0.4]])
w2 = np.array([[0.1], [0.6]])

x = np.array([1.0, 2.0])

# layer 1 then layer 2
hidden = x @ w1
output = hidden @ w2

# same as single layer!
w_combined = w1 @ w2
output_direct = x @ w_combined

print("two layers:", output)
print("combined:", output_direct)
print("identical?", np.allclose(output, output_direct))
