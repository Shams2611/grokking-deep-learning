# interpreting predictions

import numpy as np

inputs = np.array([8.5, 0.65, 1.2])
weights = np.array([
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1],
])

preds = weights @ inputs

print("PREDICTIONS:")
print(f"  hurt: {preds[0]:.3f}")
print(f"  win:  {preds[1]:.3f}")
print(f"  sad:  {preds[2]:.3f}")
print()
print("these are just numbers for now")
print("need labels and training to mean something!")
print()
print("CHAPTER 3 DONE - we can make predictions")
print("next: how to LEARN the right weights?")
