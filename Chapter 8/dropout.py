"""
Chapter 8: Dropout - Randomly Ignoring Neurons

Dropout is a super clever regularization technique!

The idea:
During training, randomly "turn off" neurons (set them to 0).
Each forward pass uses a different random subset of neurons.

Why does this help?
1. Prevents neurons from co-adapting too much
2. It's like training many different networks and averaging
3. Forces the network to be more robust

During testing:
- Don't use dropout (use all neurons)
- BUT scale the outputs to account for the missing neurons during training

The scaling thing is called "inverted dropout" - we handle it
during training instead of testing. More convenient!
"""

import numpy as np

np.random.seed(42)


def dropout(x, keep_prob=0.5, training=True):
    """
    Apply dropout to a layer's activations.

    Args:
        x: activations from the layer
        keep_prob: probability of KEEPING a neuron (not dropping)
        training: whether we're training or testing

    Returns:
        x with some neurons zeroed out (if training)
    """
    if not training:
        # During testing, use all neurons (no dropout)
        return x

    # Create random mask: 1 to keep, 0 to drop
    mask = (np.random.rand(*x.shape) < keep_prob).astype(float)

    # Apply mask and scale by 1/keep_prob
    # Scaling keeps expected value the same!
    return x * mask / keep_prob


def forward_with_dropout(X, W1, W2, keep_prob=0.5, training=True):
    """
    Forward pass with dropout after hidden layer.
    """
    # Hidden layer
    hidden = np.maximum(0, X @ W1)  # ReLU activation

    # Apply dropout to hidden layer
    hidden_dropped = dropout(hidden, keep_prob, training)

    # Output layer
    output = hidden_dropped @ W2

    return output, hidden, hidden_dropped


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Dropout Regularization")
    print("=" * 60)

    # Simple demo showing what dropout does
    print("\n--- What dropout looks like ---\n")

    # A layer's activations (pretend these came from ReLU)
    activations = np.array([1.0, 0.5, 0.8, 0.3, 1.2, 0.7, 0.4, 0.9])

    print(f"Original activations: {activations}")
    print(f"Sum: {activations.sum():.2f}")

    # Apply dropout multiple times to see the randomness
    print("\nWith 50% dropout (different each time):")
    for i in range(5):
        np.random.seed(i)  # different seed each time
        dropped = dropout(activations.copy(), keep_prob=0.5, training=True)
        print(f"  Trial {i+1}: {np.round(dropped, 2)}, sum={dropped.sum():.2f}")

    print("\nNotice:")
    print("- Different neurons are dropped each time")
    print("- Kept neurons are scaled by 2x (since keep_prob=0.5)")
    print("- Expected sum stays roughly the same!")

    # During testing
    print("\n" + "-" * 40)
    print("During testing (no dropout):")
    print("-" * 40)

    test_result = dropout(activations.copy(), keep_prob=0.5, training=False)
    print(f"Result: {test_result}")
    print("All neurons active, no scaling needed!")

    # Full network example
    print("\n" + "=" * 60)
    print("Dropout in a Neural Network")
    print("=" * 60)

    np.random.seed(42)

    # Network architecture: 4 -> 8 -> 2
    input_size, hidden_size, output_size = 4, 8, 2

    W1 = np.random.randn(input_size, hidden_size) * 0.5
    W2 = np.random.randn(hidden_size, output_size) * 0.5

    # Sample input
    X = np.random.randn(1, input_size)

    print(f"\nInput: {X.flatten().round(2)}")

    # Forward pass WITHOUT dropout
    out_no_dropout, hidden, _ = forward_with_dropout(X, W1, W2, training=False)
    print(f"\nWithout dropout:")
    print(f"  Hidden activations: {hidden.flatten().round(2)}")
    print(f"  Output: {out_no_dropout.flatten().round(2)}")

    # Forward pass WITH dropout (multiple times)
    print(f"\nWith 50% dropout (different each forward pass):")
    for i in range(3):
        np.random.seed(100 + i)
        out_dropout, _, hidden_dropped = forward_with_dropout(X, W1, W2, keep_prob=0.5, training=True)
        active = (hidden_dropped.flatten() != 0).sum()
        print(f"  Pass {i+1}: {active}/8 neurons active, output={out_dropout.flatten().round(2)}")

    # Training simulation
    print("\n" + "=" * 60)
    print("Why Dropout Helps (Intuition)")
    print("=" * 60)

    print("""
Imagine each neuron is a "student" in a group project:

Without dropout:
- Some students do all the work (strong neurons)
- Others slack off and just copy (co-adaptation)
- If the strong students are gone, the group fails

With dropout:
- Each student has to be able to work alone sometimes
- Everyone learns to be useful
- The group is more robust!

In neural network terms:
- Dropout prevents neurons from relying too much on each other
- Each neuron learns more general features
- The network is less likely to overfit
    """)

    print("=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Dropout randomly zeroes out neurons during training")
    print("2. Typical keep_prob: 0.5 for hidden layers, 0.8 for input")
    print("3. Scale outputs by 1/keep_prob to maintain expected value")
    print("4. Don't use dropout during testing!")
    print("5. It's like training an ensemble of smaller networks")
