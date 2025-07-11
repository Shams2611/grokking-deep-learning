# why input matters for gradient

# gradient = delta * input

# big input -> big gradient -> big weight change
# small input -> small gradient -> small weight change
# zero input -> zero gradient -> no change!

import numpy as np

delta = 0.5
inputs = np.array([10.0, 1.0, 0.0])

gradients = delta * inputs

print("input matters for learning speed:")
for i, (inp, g) in enumerate(zip(inputs, gradients)):
    print(f"  input={inp:4.1f} -> gradient={g:.1f}")

print()
print("input=0 means that weight CANT learn!")
