# training vs test error
# the gap tells you about overfitting

import numpy as np

# simulated learning curves
epochs = [0, 10, 20, 30, 40, 50]
train_error = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
test_error =  [1.0, 0.6, 0.4, 0.5, 0.6, 0.8]

print("epoch | train | test  | gap")
print("-" * 30)
for i, e in enumerate(epochs):
    gap = test_error[i] - train_error[i]
    print(f"  {e:2}  | {train_error[i]:.2f}  | {test_error[i]:.2f}  | {gap:.2f}")

print()
print("gap increases = overfitting!")
print("best model: epoch 20 (lowest test error)")
