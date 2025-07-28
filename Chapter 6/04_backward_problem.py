# the backward problem

# we have error at output
# how much did weights_0_1 contribute?
# it's not directly connected to output!

# need to figure out:
# 1. how much did hidden layer contribute to error?
# 2. how much did each input->hidden weight contribute?

print("backprop problem:")
print()
print("  error is at OUTPUT")
print("  but we need to update weights at INPUT layer")
print()
print("solution: CHAIN RULE")
print("  propagate error backwards through network")
