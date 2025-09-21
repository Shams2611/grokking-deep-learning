# why cross-entropy instead of MSE?

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# compare gradients at different points
logits = np.array([0.0, 0.0, 0.0])
target = np.array([1, 0, 0])

print("WHY CROSS-ENTROPY FOR CLASSIFICATION?")
print()

# MSE gradient: 2(pred - target)
# CE gradient: (pred - target) for softmax + CE combo

pred = softmax(logits)
print(f"prediction: {np.round(pred, 3)}")
print(f"target: {target}")
print()

mse_grad = 2 * (pred - target)
ce_grad = pred - target

print("gradients:")
print(f"  MSE: {np.round(mse_grad, 3)}")
print(f"  CE:  {np.round(ce_grad, 3)}")
print()
print("CE gradient is simpler and faster to compute!")
print("also: CE penalizes confident wrong answers more")
