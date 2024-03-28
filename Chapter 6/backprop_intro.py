"""
Chapter 6: Introduction to Backpropagation

This is THE fundamental algorithm of deep learning!

So far we've only had one layer of weights. But deep networks
have MANY layers. How do we update weights in the middle layers?

That's what backpropagation solves!

The key insight: use the CHAIN RULE from calculus.

If you have: input -> layer1 -> layer2 -> output
And you know the error at output...
You can "propagate" that error BACKWARDS through the network!

Each layer passes its error back to the previous layer.
It's like a game of telephone, but for gradients.

This file shows the simplest case: 2 layers, no activation functions.
"""

import numpy as np


def forward_pass(input_val, weights_0_1, weights_1_2):
    """
    Forward pass through 2 layers.

    input -> hidden -> output

    hidden = input * weights_0_1
    output = hidden * weights_1_2
    """
    hidden = input_val * weights_0_1
    output = hidden * weights_1_2
    return hidden, output


def backward_pass(input_val, hidden, output, goal, weights_1_2):
    """
    Backward pass - propagate error back through layers.

    We need to figure out:
    1. How to update weights_1_2 (the output layer)
    2. How to update weights_0_1 (the hidden layer)

    For output layer: same as before
    For hidden layer: we need to know how much IT contributed to the error
    """
    # Output layer gradient (same as single layer)
    output_delta = output - goal
    grad_1_2 = output_delta * hidden  # gradient for weights_1_2

    # Hidden layer gradient - this is the backprop magic!
    # How much did hidden contribute to output_delta?
    # The error flows back THROUGH weights_1_2
    hidden_delta = output_delta * weights_1_2

    grad_0_1 = hidden_delta * input_val  # gradient for weights_0_1

    return grad_0_1, grad_1_2


def train(input_val, goal, lr=0.1, iterations=20):
    """
    Train a 2-layer network using backpropagation.
    """
    # Initialize weights randomly
    weights_0_1 = np.random.randn()
    weights_1_2 = np.random.randn()

    print(f"Initial weights: w1={weights_0_1:.4f}, w2={weights_1_2:.4f}")

    for i in range(iterations):
        # Forward pass
        hidden, output = forward_pass(input_val, weights_0_1, weights_1_2)

        # Calculate error
        error = (output - goal) ** 2

        # Backward pass - get gradients
        grad_0_1, grad_1_2 = backward_pass(
            input_val, hidden, output, goal, weights_1_2
        )

        # Update weights
        weights_0_1 -= lr * grad_0_1
        weights_1_2 -= lr * grad_1_2

        if i % 5 == 0:
            print(f"Iter {i}: output={output:.4f}, error={error:.6f}")

    return weights_0_1, weights_1_2


# ============================================
# Demo!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Backpropagation Introduction - 2 Layer Network")
    print("=" * 60)

    np.random.seed(42)  # for reproducibility

    input_val = 2.0
    goal = 0.5

    print(f"\nInput: {input_val}")
    print(f"Goal: {goal}")

    print("\n--- Training ---\n")
    w1, w2 = train(input_val, goal, lr=0.1, iterations=25)

    print(f"\n--- Results ---")
    print(f"Final weights: w1={w1:.4f}, w2={w2:.4f}")

    hidden, output = forward_pass(input_val, w1, w2)
    print(f"Final output: {output:.4f}")
    print(f"Goal was: {goal}")

    # Trace through backprop manually
    print("\n" + "=" * 60)
    print("Manual backprop trace:")
    print("=" * 60)

    # Use simple values for clarity
    inp = 2.0
    w_01 = 0.5
    w_12 = 0.5
    target = 0.5

    print(f"\nForward pass:")
    h = inp * w_01
    print(f"  hidden = input * w1 = {inp} * {w_01} = {h}")
    o = h * w_12
    print(f"  output = hidden * w2 = {h} * {w_12} = {o}")

    print(f"\nBackward pass:")
    o_delta = o - target
    print(f"  output_delta = output - goal = {o} - {target} = {o_delta}")

    g_12 = o_delta * h
    print(f"  grad_w2 = output_delta * hidden = {o_delta} * {h} = {g_12}")

    h_delta = o_delta * w_12
    print(f"  hidden_delta = output_delta * w2 = {o_delta} * {w_12} = {h_delta}")

    g_01 = h_delta * inp
    print(f"  grad_w1 = hidden_delta * input = {h_delta} * {inp} = {g_01}")

    print("\n" + "=" * 60)
    print("Key Insight:")
    print("=" * 60)
    print("Error flows BACKWARDS through the network!")
    print("hidden_delta = output_delta * weights_1_2")
    print("This is the chain rule in action.")
    print("\nThe hidden layer's error depends on:")
    print("1. The error at the output")
    print("2. How much the hidden layer contributed (via its weight)")
