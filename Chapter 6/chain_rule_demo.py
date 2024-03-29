"""
Chapter 6: The Chain Rule - Heart of Backpropagation

Why does backprop work? THE CHAIN RULE!

If you have a composition of functions:
    y = f(g(h(x)))

The derivative dy/dx is:
    dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)

In neural networks:
    output = activation(weighted_sum(previous_layer))

Each layer is a function, and we chain them together!

This file demonstrates the chain rule with simple examples
so the math feels intuitive before we apply it to networks.
"""

import numpy as np


# ============================================
# Simple chain rule examples
# ============================================

def example_1_simple_chain():
    """
    Simple example: y = (x + 1)^2

    Let's break it into two functions:
    u = x + 1      (inner function)
    y = u^2        (outer function)

    By chain rule:
    dy/dx = dy/du * du/dx
          = 2u * 1
          = 2(x+1)
    """
    print("Example 1: y = (x + 1)^2")
    print("-" * 40)

    x = 3.0

    # Forward pass - compute step by step
    u = x + 1        # inner: u = x + 1
    y = u ** 2       # outer: y = u^2

    print(f"x = {x}")
    print(f"u = x + 1 = {u}")
    print(f"y = u^2 = {y}")

    # Backward pass - chain rule
    dy_du = 2 * u    # derivative of y with respect to u
    du_dx = 1        # derivative of u with respect to x
    dy_dx = dy_du * du_dx  # chain rule!

    print(f"\nDerivatives:")
    print(f"dy/du = 2u = {dy_du}")
    print(f"du/dx = 1")
    print(f"dy/dx = dy/du * du/dx = {dy_du} * {du_dx} = {dy_dx}")

    # Verify numerically
    epsilon = 0.0001
    y1 = ((x + epsilon) + 1) ** 2
    y0 = ((x - epsilon) + 1) ** 2
    numerical_derivative = (y1 - y0) / (2 * epsilon)

    print(f"\nNumerical check: {numerical_derivative:.4f}")
    print(f"Our answer: {dy_dx:.4f}")
    print(f"Match: {np.isclose(numerical_derivative, dy_dx)}")


def example_2_neural_network_chain():
    """
    Neural network example with 2 layers.

    x -> [w1] -> h -> [w2] -> y

    h = x * w1
    y = h * w2 = x * w1 * w2

    We want: dy/dw1 (how does y change when we change w1?)

    Chain rule:
    dy/dw1 = dy/dh * dh/dw1
           = w2 * x
    """
    print("\n" + "=" * 50)
    print("Example 2: Two-layer network")
    print("-" * 40)

    x = 2.0
    w1 = 0.5
    w2 = 0.3

    # Forward pass
    h = x * w1       # hidden layer
    y = h * w2       # output layer

    print(f"x = {x}, w1 = {w1}, w2 = {w2}")
    print(f"h = x * w1 = {h}")
    print(f"y = h * w2 = {y}")

    # Backward pass for w1 (the interesting one!)
    dy_dh = w2       # how does y change with h?
    dh_dw1 = x       # how does h change with w1?
    dy_dw1 = dy_dh * dh_dw1  # chain rule!

    print(f"\nBackward pass for gradient of w1:")
    print(f"dy/dh = w2 = {dy_dh}")
    print(f"dh/dw1 = x = {dh_dw1}")
    print(f"dy/dw1 = dy/dh * dh/dw1 = {dy_dh} * {dh_dw1} = {dy_dw1}")

    # Also do backward pass for w2 (simpler)
    dy_dw2 = h  # direct connection

    print(f"\nBackward pass for gradient of w2:")
    print(f"dy/dw2 = h = {dy_dw2}")

    # Numerical verification
    eps = 0.0001
    y_up = x * (w1 + eps) * w2
    y_down = x * (w1 - eps) * w2
    numerical = (y_up - y_down) / (2 * eps)

    print(f"\nNumerical check for dy/dw1: {numerical:.4f}")
    print(f"Our answer: {dy_dw1:.4f}")


def example_3_with_error():
    """
    Full example with error and gradient for learning.

    x -> [w1] -> h -> [w2] -> pred -> error = (pred - goal)^2

    We want gradients to minimize error!
    """
    print("\n" + "=" * 50)
    print("Example 3: Full training example")
    print("-" * 40)

    x = 2.0
    w1 = 0.5
    w2 = 0.3
    goal = 1.0

    # Forward pass
    h = x * w1
    pred = h * w2
    error = (pred - goal) ** 2

    print(f"Forward pass:")
    print(f"  h = x * w1 = {h}")
    print(f"  pred = h * w2 = {pred}")
    print(f"  error = (pred - goal)^2 = {error}")

    # Backward pass
    # Start from error and work backwards

    # d(error)/d(pred) = 2(pred - goal)
    d_error_d_pred = 2 * (pred - goal)

    # d(pred)/d(w2) = h
    d_pred_d_w2 = h

    # d(error)/d(w2) = d(error)/d(pred) * d(pred)/d(w2)
    grad_w2 = d_error_d_pred * d_pred_d_w2

    # d(pred)/d(h) = w2
    d_pred_d_h = w2

    # d(h)/d(w1) = x
    d_h_d_w1 = x

    # d(error)/d(w1) = d(error)/d(pred) * d(pred)/d(h) * d(h)/d(w1)
    grad_w1 = d_error_d_pred * d_pred_d_h * d_h_d_w1

    print(f"\nBackward pass:")
    print(f"  d(error)/d(pred) = 2(pred - goal) = {d_error_d_pred}")
    print(f"  d(pred)/d(w2) = h = {d_pred_d_w2}")
    print(f"  grad_w2 = {d_error_d_pred} * {d_pred_d_w2} = {grad_w2}")
    print(f"")
    print(f"  d(pred)/d(h) = w2 = {d_pred_d_h}")
    print(f"  d(h)/d(w1) = x = {d_h_d_w1}")
    print(f"  grad_w1 = {d_error_d_pred} * {d_pred_d_h} * {d_h_d_w1} = {grad_w1}")

    # Update weights
    lr = 0.1
    w1_new = w1 - lr * grad_w1
    w2_new = w2 - lr * grad_w2

    print(f"\nWeight updates (lr={lr}):")
    print(f"  w1: {w1} -> {w1_new}")
    print(f"  w2: {w2} -> {w2_new}")


# ============================================
# Main
# ============================================
if __name__ == "__main__":

    print("=" * 50)
    print("The Chain Rule - Foundation of Backpropagation")
    print("=" * 50)

    example_1_simple_chain()
    example_2_neural_network_chain()
    example_3_with_error()

    print("\n" + "=" * 50)
    print("Key Takeaways:")
    print("=" * 50)
    print("1. Chain rule lets us compute derivatives through layers")
    print("2. We work BACKWARDS from the error")
    print("3. Each layer passes its delta back, multiplied by its weights")
    print("4. This is why it's called BACKpropagation!")
