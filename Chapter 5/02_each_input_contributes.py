# each input contributes to prediction

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([0.1, 0.2, 0.0])

# prediction = sum of (input * weight)
contributions = inputs * weights

print("contribution from each input:")
for i, (inp, w, c) in enumerate(zip(inputs, weights, contributions)):
    print(f"  input[{i}]: {inp} * {w} = {c:.3f}")

print(f"\ntotal prediction: {sum(contributions):.3f}")
