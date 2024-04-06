"""
Chapter 9: Softmax - Turning Scores into Probabilities

Softmax is THE activation function for multi-class classification!

What it does:
Takes a vector of raw scores (logits) and turns them into probabilities
that sum to 1.

softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

Example: classify an image as [cat, dog, bird]
- Network outputs: [2.0, 1.0, 0.1]  (raw scores)
- After softmax: [0.66, 0.24, 0.10]  (probabilities!)

Now we can say "66% confident it's a cat"

Why exp()?
- Makes all values positive
- Bigger scores get WAY bigger (exponentially)
- Creates a clear "winner" while still being smooth

Softmax is the multi-class version of sigmoid!
- Sigmoid: 2 classes (binary)
- Softmax: N classes
"""

import numpy as np


def softmax(x):
    """
    Softmax activation.

    Convert raw scores to probabilities.
    Output sums to 1!
    """
    # Subtract max for numerical stability
    # (prevents overflow with large values)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def softmax_batch(X):
    """
    Softmax for a batch of samples.

    X shape: (batch_size, num_classes)
    """
    # Subtract max per sample (along axis=1)
    exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(softmax_output):
    """
    Derivative of softmax.

    This is actually a Jacobian matrix, not a simple derivative!
    For softmax output s:
    ds_i/dz_j = s_i * (1 - s_j) if i == j
              = -s_i * s_j      if i != j

    In practice, we usually don't compute this directly.
    When combined with cross-entropy loss, the math simplifies!
    """
    s = softmax_output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Softmax Activation")
    print("=" * 60)

    # Example: classify an image
    print("\n--- Image Classification Example ---")

    classes = ["cat", "dog", "bird"]
    raw_scores = np.array([2.0, 1.0, 0.1])

    print(f"\nClasses: {classes}")
    print(f"Raw network output (logits): {raw_scores}")

    probs = softmax(raw_scores)
    print(f"After softmax (probabilities): {probs.round(4)}")
    print(f"Sum of probabilities: {probs.sum():.4f}")

    print(f"\nInterpretation:")
    for cls, prob in zip(classes, probs):
        print(f"  P({cls}) = {prob:.2%}")

    predicted = classes[np.argmax(probs)]
    print(f"\nPrediction: {predicted} (highest probability)")

    # Show how softmax exaggerates differences
    print("\n" + "=" * 60)
    print("How Softmax Amplifies Differences")
    print("=" * 60)

    test_cases = [
        [1.0, 1.0, 1.0],      # equal scores
        [2.0, 1.0, 1.0],      # slight difference
        [5.0, 1.0, 1.0],      # bigger difference
        [10.0, 1.0, 1.0],     # huge difference
    ]

    print(f"\n{'Scores':<20} | {'Softmax Output':<35} | Winner Prob")
    print("-" * 70)

    for scores in test_cases:
        scores = np.array(scores)
        probs = softmax(scores)
        print(f"{str(scores):<20} | {str(probs.round(4)):<35} | {probs.max():.2%}")

    print("\nNotice: bigger differences -> more confident predictions")

    # Numerical stability
    print("\n" + "=" * 60)
    print("Why We Subtract Max (Numerical Stability)")
    print("=" * 60)

    big_scores = np.array([1000.0, 1001.0, 1002.0])

    print(f"\nBig scores: {big_scores}")

    # Without stability trick (would overflow!)
    print("\nWithout max subtraction:")
    print(f"  exp({big_scores[0]}) = overflow! (too big for float)")

    # With stability trick
    print("\nWith max subtraction:")
    shifted = big_scores - np.max(big_scores)
    print(f"  Shifted scores: {shifted}")
    print(f"  exp values: {np.exp(shifted).round(4)}")
    print(f"  Softmax: {softmax(big_scores).round(4)}")
    print("\nSame result, no overflow!")

    # Temperature scaling
    print("\n" + "=" * 60)
    print("Bonus: Temperature Scaling")
    print("=" * 60)

    def softmax_with_temp(x, temperature=1.0):
        """
        Softmax with temperature.

        temperature > 1: softer probabilities (more uniform)
        temperature < 1: sharper probabilities (more confident)
        temperature -> 0: argmax (one-hot)
        """
        return softmax(x / temperature)

    scores = np.array([2.0, 1.0, 0.5])

    print(f"\nScores: {scores}")
    print(f"\n{'Temperature':<15} | {'Softmax Output':<35}")
    print("-" * 55)

    for temp in [0.5, 1.0, 2.0, 5.0]:
        probs = softmax_with_temp(scores, temp)
        print(f"{temp:<15} | {str(probs.round(4)):<35}")

    print("\nLower temp = more confident, Higher temp = more uncertain")
    print("(Used in techniques like knowledge distillation!)")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Softmax converts scores to probabilities (sum to 1)")
    print("2. Used for multi-class classification output layer")
    print("3. Exponential makes big scores dominate")
    print("4. Subtract max for numerical stability")
    print("5. Combined with cross-entropy loss for training")
