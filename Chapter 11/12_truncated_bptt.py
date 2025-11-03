# truncated BPTT - limit backprop steps

import numpy as np

print("TRUNCATED BPTT")
print()
print("problem: long sequences = expensive + vanishing gradients")
print()
print("solution: only backprop through last K steps")
print()

# example: sequence of 100 timesteps
# but only backprop through last 20

sequence_length = 100
truncation_length = 20

print(f"sequence length: {sequence_length}")
print(f"truncation length: {truncation_length}")
print()
print("forward: process all 100 steps")
print("backward: only last 20 steps")
print()
print("benefits:")
print("  - faster training")
print("  - more stable gradients")
print("  - still captures recent context")
print()
print("tradeoff: cant learn very long-range patterns")
print("thats why we need LSTM/GRU!")
