# softmax turns scores into probabilities
# all outputs sum to 1

import numpy as np

# raw scores from network (logits)
scores = np.array([2.0, 1.0, 0.1])

# softmax: e^x / sum(e^x)
exp_scores = np.exp(scores)
probabilities = exp_scores / exp_scores.sum()

print("raw scores:", scores)
print("exp(scores):", np.round(exp_scores, 3))
print("sum of exp:", round(exp_scores.sum(), 3))
print("probabilities:", np.round(probabilities, 3))
print("sum of probs:", round(probabilities.sum(), 3))
print()
print("highest score -> highest probability")
print("all probs sum to 1!")
