# GRU - simplified LSTM

import numpy as np

print("GRU: Gated Recurrent Unit")
print()
print("simpler than LSTM, similar performance")
print()
print("differences:")
print("  - no separate cell state (only hidden)")
print("  - 2 gates instead of 3")
print("  - fewer parameters")
print()
print("GRU equations:")
print("  z_t = sigmoid(W_z * [h_{t-1}, x_t])  # update gate")
print("  r_t = sigmoid(W_r * [h_{t-1}, x_t])  # reset gate")
print("  h_tilde = tanh(W * [r_t * h_{t-1}, x_t])")
print("  h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde")
print()
print("z controls how much to update")
print("r controls how much history to use")
