"""
Chapter 8: Overfitting - The Enemy

Overfitting is when your model memorizes the training data
instead of learning the underlying patterns.

Signs of overfitting:
- Training error keeps going down
- But validation/test error goes UP
- Model doesn't generalize to new data

It's like a student who memorizes all the practice problems
but can't solve new ones on the exam.

This file demonstrates overfitting with a simple example
so we can SEE what's happening.
"""

import numpy as np

np.random.seed(42)


def generate_data(n_samples, noise=0.3):
    """
    Generate data from y = sin(x) + noise

    This is a smooth function, but if our model is too complex,
    it'll try to fit the noise too!
    """
    X = np.random.uniform(-3, 3, n_samples)
    y = np.sin(X) + np.random.randn(n_samples) * noise
    return X, y


def polynomial_features(X, degree):
    """
    Create polynomial features.

    Higher degree = more complex model = more likely to overfit!
    """
    X = np.array(X).reshape(-1, 1)
    features = np.ones((len(X), 1))

    for d in range(1, degree + 1):
        features = np.hstack([features, X ** d])

    return features


def fit_polynomial(X, y, degree, lr=0.0001, iterations=1000):
    """
    Fit a polynomial using gradient descent.
    """
    X_poly = polynomial_features(X, degree)
    n_features = X_poly.shape[1]

    # Initialize weights
    weights = np.random.randn(n_features) * 0.01

    for i in range(iterations):
        # Predictions
        pred = X_poly @ weights

        # Error
        error = pred - y
        mse = np.mean(error ** 2)

        # Gradient
        gradient = (X_poly.T @ error) / len(X)

        # Update
        weights -= lr * gradient

    return weights


def evaluate(X, y, weights, degree):
    """Calculate MSE for given data."""
    X_poly = polynomial_features(X, degree)
    pred = X_poly @ weights
    mse = np.mean((pred - y) ** 2)
    return mse


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Overfitting Demonstration")
    print("=" * 60)

    # Generate training and test data
    X_train, y_train = generate_data(20, noise=0.3)
    X_test, y_test = generate_data(50, noise=0.3)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("\nTrue function: y = sin(x) + noise")

    # Try different polynomial degrees
    degrees = [1, 3, 5, 10, 15]

    print("\n" + "=" * 60)
    print("Training models with different complexity:")
    print("=" * 60)
    print(f"\n{'Degree':>8} | {'Train MSE':>12} | {'Test MSE':>12} | Status")
    print("-" * 55)

    for degree in degrees:
        weights = fit_polynomial(X_train, y_train, degree, lr=0.001, iterations=5000)

        train_mse = evaluate(X_train, y_train, weights, degree)
        test_mse = evaluate(X_test, y_test, weights, degree)

        # Determine status
        if test_mse > train_mse * 2:
            status = "OVERFITTING!"
        elif test_mse < train_mse * 1.2:
            status = "Good"
        else:
            status = "Slight overfit"

        print(f"{degree:>8} | {train_mse:>12.4f} | {test_mse:>12.4f} | {status}")

    # Explain what happened
    print("\n" + "=" * 60)
    print("What's happening here?")
    print("=" * 60)

    print("""
Degree 1:  Linear model - too simple (underfitting)
           Can't capture the sin curve at all.

Degree 3:  Just right!
           Captures the general shape without memorizing noise.

Degree 5:  Starting to overfit
           Fits training data better but generalizes worse.

Degree 10: Definite overfitting
           Training error is great, but test error is worse!

Degree 15: Severe overfitting
           Model is memorizing individual training points.
           """)

    # Visual demonstration of what the model "sees"
    print("=" * 60)
    print("Training Data Visualization (ASCII)")
    print("=" * 60)

    # Sort for plotting
    sort_idx = np.argsort(X_train)
    X_sorted = X_train[sort_idx]
    y_sorted = y_train[sort_idx]

    print("\nTraining points (x, y):")
    for i in range(min(10, len(X_sorted))):
        bar = "#" * int((y_sorted[i] + 1.5) * 10)
        print(f"  x={X_sorted[i]:5.2f}: {bar} ({y_sorted[i]:.2f})")
    if len(X_sorted) > 10:
        print("  ...")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. More complex models CAN fit training data better")
    print("2. But they might just be memorizing noise")
    print("3. Test error is what really matters!")
    print("4. We need techniques to PREVENT overfitting:")
    print("   - More training data")
    print("   - Simpler models")
    print("   - Regularization (L1, L2)")
    print("   - Dropout")
    print("   - Early stopping")
