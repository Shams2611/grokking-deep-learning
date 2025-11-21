# LSTM backward pass (simplified)

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def sigmoid_deriv(s): return s * (1 - s)
def tanh_deriv(t): return 1 - t**2

print("LSTM BACKWARD PASS")
print()
print("more complex than RNN, but same idea")
print()
print("key gradients at each timestep:")
print()
print("d_h: gradient from output/next timestep")
print("d_c: gradient from next timestep cell")
print()
print("then compute gradients for each gate:")
print("  d_o = d_h * tanh(c) * sigmoid'(o)")
print("  d_c += d_h * o * tanh'(c)")
print("  d_f = d_c * c_prev * sigmoid'(f)")
print("  d_i = d_c * c_tilde * sigmoid'(i)")
print("  d_c_tilde = d_c * i * tanh'(c_tilde)")
print()
print("propagate d_c to previous timestep:")
print("  d_c_prev = d_c * f")
print()
print("this is why f~1 helps gradients flow!")
