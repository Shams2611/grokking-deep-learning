# temperature in softmax
# controls how "sharp" the distribution is

import numpy as np

def softmax_with_temp(x, temperature=1.0):
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.5])

print("TEMPERATURE SCALING")
print(f"logits: {logits}")
print()

for temp in [0.5, 1.0, 2.0, 5.0]:
    probs = softmax_with_temp(logits, temp)
    print(f"T={temp}: {np.round(probs, 3)}")

print()
print("low T -> sharper (more confident)")
print("high T -> softer (more uncertain)")
print()
print("T=0 would be argmax (one-hot)")
print("T=inf would be uniform")
