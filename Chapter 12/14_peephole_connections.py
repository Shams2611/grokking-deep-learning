# peephole connections - gates look at cell state

import numpy as np

print("PEEPHOLE CONNECTIONS")
print()
print("standard LSTM: gates see [h_{t-1}, x_t]")
print("peephole LSTM: gates also see c_{t-1}")
print()
print("modified equations:")
print("  f_t = sigmoid(W_f * [h, x] + V_f * c_{t-1})")
print("  i_t = sigmoid(W_i * [h, x] + V_i * c_{t-1})")
print("  o_t = sigmoid(W_o * [h, x] + V_o * c_t)")
print()
print("gates can directly use cell state info")
print("sometimes helps, sometimes not")
print()
print("not always used in practice")
print("standard LSTM works well enough")
