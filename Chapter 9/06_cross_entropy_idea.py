# cross-entropy loss
# measures how wrong our probability distribution is

import numpy as np

print("CROSS-ENTROPY LOSS")
print()
print("idea: penalize wrong predictions heavily")
print()

# predictions for class 0 (cat)
pred_confident_right = 0.9   # confident and correct
pred_uncertain = 0.5         # uncertain
pred_confident_wrong = 0.1   # confident but wrong

# cross-entropy: -log(predicted_prob_of_correct_class)
print("true class: cat")
print()
print("prediction | -log(p) | interpretation")
print("-" * 45)
print(f"   0.9     | {-np.log(0.9):.3f}   | confident & right (low loss)")
print(f"   0.5     | {-np.log(0.5):.3f}   | uncertain (medium loss)")
print(f"   0.1     | {-np.log(0.1):.3f}   | confident & wrong (high loss)")
print()
print("being confidently wrong is heavily penalized!")
