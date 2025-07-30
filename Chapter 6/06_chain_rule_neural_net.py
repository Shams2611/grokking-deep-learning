# chain rule in neural net

# output = hidden @ weights_1_2
# hidden = input @ weights_0_1

# to get gradient for weights_0_1:
# d_error/d_weights_0_1 = d_error/d_hidden * d_hidden/d_weights_0_1

# d_error/d_hidden is the "backpropagated error"
# tells us how much each hidden neuron contributed

print("neural net chain rule:")
print()
print("forward:")
print("  hidden = input @ W_01")
print("  output = hidden @ W_12")
print("  error = (output - goal)^2")
print()
print("backward:")
print("  d_error/d_W_12 = delta_out * hidden")
print("  d_error/d_hidden = delta_out @ W_12.T")
print("  d_error/d_W_01 = delta_hidden * input")
