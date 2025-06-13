# negative weights

# positive weight: input increases output
# negative weight: input decreases output
# zero weight: input ignored

import numpy as np

inputs = np.array([1.0, 1.0, 1.0])

weights = np.array([0.5, -0.5, 0.0])

result = inputs @ weights

print("inputs:", inputs)
print("weights:", weights)
print()
print("0.5: positive, adds to output")
print("-0.5: negative, subtracts from output")
print("0.0: ignored")
print()
print(f"result: {result}")
