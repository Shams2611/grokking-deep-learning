# when to use which activation

print("HIDDEN LAYERS:")
print("  ReLU - default choice, fast, works well")
print("  Leaky ReLU - if you have dead neurons")
print("  tanh - sometimes for RNNs")
print()
print("OUTPUT LAYER:")
print("  None (linear) - regression")
print("  sigmoid - binary classification (0 to 1)")
print("  softmax - multi-class classification")
print("  tanh - outputs between -1 and 1")
print()
print("general rule: start with ReLU for hidden layers")
