# why CNNs for images?
# regular networks dont understand spatial structure

print("WHY CNNs?")
print()
print("problem with regular neural networks for images:")
print()
print("28x28 image = 784 pixels")
print("each pixel connects to every neuron")
print("= tons of parameters, no spatial awareness")
print()
print("CNN key ideas:")
print("  1. local connectivity - neurons see small patches")
print("  2. parameter sharing - same filter across image")
print("  3. translation invariance - cat is cat anywhere")
print()
print("result: fewer parameters, better performance!")
