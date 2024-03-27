"""
Chapter 5: Full Batch Gradient Descent

So far we've been learning from one example at a time.
But real datasets have THOUSANDS of examples!

Batch gradient descent: average the gradients across all examples

Why average?
- Each example might have noise or weird values
- Averaging smooths things out
- More stable learning

It's like asking 100 people for directions vs just 1 person.
The average is usually more reliable!

Trade-off:
- Full batch: stable but slow (process ALL data before one update)
- Single example: fast but noisy (update after each example)
- Mini-batch: compromise (update after N examples) - most common!
"""

import numpy as np


def forward(inputs, weights):
    """Predict for one example."""
    return np.dot(inputs, weights)


def full_batch_gradient_descent(X, y, weights, lr=0.01, epochs=100):
    """
    Train on multiple examples at once.

    X: matrix of inputs (num_examples x num_features)
    y: array of goals (num_examples,)
    weights: initial weights (num_features,)

    For each epoch:
    1. Calculate gradient for EACH example
    2. Average all gradients
    3. Update weights once

    This is 'full batch' because we use ALL examples before updating.
    """
    weights = np.array(weights, dtype=float)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    num_examples = len(X)

    for epoch in range(epochs):
        # Accumulate gradients from all examples
        total_gradient = np.zeros_like(weights)
        total_error = 0

        for i in range(num_examples):
            # Forward pass for this example
            pred = forward(X[i], weights)

            # Error and delta
            error = (pred - y[i]) ** 2
            delta = pred - y[i]

            # Gradient for this example
            gradient = delta * X[i]

            # Accumulate
            total_gradient += gradient
            total_error += error

        # Average gradient and error
        avg_gradient = total_gradient / num_examples
        avg_error = total_error / num_examples

        # Update weights ONCE per epoch (using averaged gradient)
        weights = weights - (lr * avg_gradient)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: avg_error = {avg_error:.6f}")

    return weights


def vectorized_batch_gd(X, y, weights, lr=0.01, epochs=100):
    """
    Same thing but using matrix operations - much faster!

    This is how you'd actually implement it in practice.
    """
    weights = np.array(weights, dtype=float)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    num_examples = len(X)

    for epoch in range(epochs):
        # All predictions at once (matrix multiply)
        predictions = X @ weights  # (num_examples,)

        # All errors at once
        errors = (predictions - y) ** 2
        deltas = predictions - y

        # Average gradient (magic of linear algebra!)
        # gradient = average of (delta * x) for all examples
        # = (X.T @ deltas) / num_examples
        avg_gradient = (X.T @ deltas) / num_examples

        # Update weights
        weights = weights - (lr * avg_gradient)

        if epoch % 10 == 0:
            avg_error = np.mean(errors)
            print(f"Epoch {epoch}: avg_error = {avg_error:.6f}")

    return weights


# ============================================
# Demo time!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Full Batch Gradient Descent")
    print("=" * 60)

    # Create a small dataset
    # 4 examples, 3 features each
    X = np.array([
        [8.5, 0.65, 1.2],   # game 1
        [9.5, 0.80, 1.3],   # game 2
        [9.0, 0.75, 0.9],   # game 3
        [8.0, 0.55, 1.0],   # game 4
    ])

    # Goals (did the team win? 1=yes, 0=no)
    y = np.array([1, 1, 1, 0])

    initial_weights = np.array([0.1, 0.1, 0.1])

    print(f"\nDataset: {len(X)} examples, {X.shape[1]} features")
    print(f"Initial weights: {initial_weights}")

    print("\n--- Training (loop version) ---\n")
    weights1 = full_batch_gradient_descent(X, y, initial_weights.copy(), lr=0.01, epochs=50)

    print("\n--- Training (vectorized version) ---\n")
    weights2 = vectorized_batch_gd(X, y, initial_weights.copy(), lr=0.01, epochs=50)

    print(f"\n--- Results ---")
    print(f"Loop version weights:      {weights1}")
    print(f"Vectorized version weights: {weights2}")
    print(f"Weights match: {np.allclose(weights1, weights2)}")

    # Test predictions
    print("\n--- Predictions on training data ---")
    for i in range(len(X)):
        pred = forward(X[i], weights1)
        print(f"Example {i}: pred={pred:.3f}, actual={y[i]}")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Full batch = use ALL examples before updating weights")
    print("2. Average gradient = smoother, more stable learning")
    print("3. Vectorized = same result but MUCH faster")
    print("4. In practice, people use 'mini-batches' of 32-256 examples")
    print("   to get the benefits of both approaches!")
