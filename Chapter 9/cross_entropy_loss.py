"""
Chapter 9: Cross-Entropy Loss

Cross-entropy is THE loss function for classification!

What does it measure?
The "distance" between two probability distributions:
- The network's predicted probabilities
- The true labels (as probabilities)

Formula (for one sample):
    loss = -sum(y_true * log(y_pred))

For one-hot encoded labels (only one class is 1, rest are 0):
    loss = -log(y_pred[correct_class])

Intuition:
- If we predict 0.9 for the correct class: loss = -log(0.9) = 0.1 (low!)
- If we predict 0.1 for the correct class: loss = -log(0.1) = 2.3 (high!)
- If we predict 0.0001: loss = -log(0.0001) = 9.2 (huge!)

The loss penalizes confident wrong predictions HEAVILY.
"""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss for one sample.

    y_pred: predicted probabilities (from softmax)
    y_true: true labels (one-hot encoded)

    Returns: loss value (scalar)
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Cross entropy: -sum(y_true * log(y_pred))
    return -np.sum(y_true * np.log(y_pred))


def cross_entropy_batch(y_pred, y_true):
    """
    Cross-entropy for a batch of samples.

    Returns: average loss across batch
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Sum for each sample, then average
    losses = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(losses)


def cross_entropy_from_logits(logits, y_true):
    """
    Cross-entropy directly from logits (before softmax).

    This is numerically more stable!
    Combines softmax + cross-entropy in one step.
    """
    # Subtract max for stability
    logits = logits - np.max(logits, axis=-1, keepdims=True)

    # Log-sum-exp trick
    log_sum_exp = np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))

    # Cross entropy = -logits[true_class] + log_sum_exp
    return np.mean(np.sum(-y_true * logits + y_true * log_sum_exp, axis=-1))


def softmax(x):
    """Softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Cross-Entropy Loss")
    print("=" * 60)

    # Simple example
    print("\n--- Single Sample Example ---")

    # True label: class 0 (cat)
    y_true = np.array([1, 0, 0])  # one-hot: [cat, dog, bird]

    # Different predictions
    predictions = [
        ([0.9, 0.05, 0.05], "Confident and correct"),
        ([0.6, 0.2, 0.2], "Somewhat confident, correct"),
        ([0.34, 0.33, 0.33], "Unsure"),
        ([0.1, 0.45, 0.45], "Wrong prediction"),
        ([0.01, 0.49, 0.50], "Confident but wrong"),
    ]

    print(f"\nTrue label: cat (one-hot: {y_true})")
    print(f"\n{'Prediction':<25} | {'Loss':<10} | Description")
    print("-" * 60)

    for pred, desc in predictions:
        pred = np.array(pred)
        loss = cross_entropy_loss(pred, y_true)
        print(f"{str(pred):<25} | {loss:<10.4f} | {desc}")

    # Gradient computation
    print("\n" + "=" * 60)
    print("Why Cross-Entropy + Softmax Work So Well Together")
    print("=" * 60)

    print("""
The beautiful thing about cross-entropy with softmax:

The gradient simplifies to: y_pred - y_true

That's it! No complicated derivatives.

Example:
- y_true = [1, 0, 0]  (cat)
- y_pred = [0.7, 0.2, 0.1]

Gradient = [0.7-1, 0.2-0, 0.1-0] = [-0.3, 0.2, 0.1]

This tells us:
- Push class 0 (cat) probability UP (negative gradient)
- Push other classes DOWN (positive gradient)
    """)

    # Demo the gradient
    print("Gradient demonstration:")
    y_true = np.array([1, 0, 0])
    y_pred = np.array([0.7, 0.2, 0.1])

    gradient = y_pred - y_true
    print(f"  y_true: {y_true}")
    print(f"  y_pred: {y_pred}")
    print(f"  gradient (y_pred - y_true): {gradient}")

    # Multi-class classification example
    print("\n" + "=" * 60)
    print("Training Example: Digit Classification")
    print("=" * 60)

    np.random.seed(42)

    # Simulated: 5 samples, 3 classes
    # Raw network outputs (logits)
    logits = np.array([
        [2.0, 1.0, 0.1],   # should be class 0
        [0.5, 2.5, 0.3],   # should be class 1
        [0.1, 0.2, 2.0],   # should be class 2
        [1.5, 1.0, 0.5],   # should be class 0
        [0.3, 1.8, 1.5],   # should be class 1
    ])

    # True labels (one-hot)
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])

    print("\nBefore training:")
    print(f"Logits:\n{logits}")

    probs = softmax(logits)
    print(f"\nPredicted probabilities:\n{probs.round(3)}")

    loss = cross_entropy_batch(probs, y_true)
    print(f"\nCross-entropy loss: {loss:.4f}")

    # Calculate gradient
    gradients = probs - y_true
    print(f"\nGradients (what to update):\n{gradients.round(3)}")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Cross-entropy measures 'distance' between distributions")
    print("2. Formula: -sum(y_true * log(y_pred))")
    print("3. Heavily penalizes confident wrong predictions")
    print("4. With softmax, gradient = y_pred - y_true (elegant!)")
    print("5. Use 'from logits' version for numerical stability")
    print("6. THE loss function for classification tasks")
