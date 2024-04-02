"""
Chapter 7: Tanh (Hyperbolic Tangent) Activation

Tanh is sigmoid's cooler cousin!

tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Or equivalently: tanh(x) = 2 * sigmoid(2x) - 1

What it does:
- Squashes input to (-1, 1) instead of (0, 1)
- Zero-centered! (unlike sigmoid)
- Large negative -> -1
- Large positive -> +1
- Zero -> 0

Why tanh > sigmoid for hidden layers:
- Zero-centered outputs help with gradient flow
- Stronger gradients (derivative ranges 0 to 1, vs 0 to 0.25 for sigmoid)
- Often converges faster

Still has vanishing gradient problem, but less severe than sigmoid.
For most modern networks, ReLU is preferred for hidden layers.

Tanh is still used in:
- RNNs and LSTMs (especially for cell state)
- When you need outputs centered around 0
"""

import numpy as np


def tanh(x):
    """
    Hyperbolic tangent.

    Squashes any value to (-1, 1)
    """
    return np.tanh(x)  # numpy has it built in!


def tanh_manual(x):
    """
    Tanh computed from the definition.
    Same result, just to see the math.
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_from_sigmoid(x):
    """
    Tanh in terms of sigmoid.
    Shows the relationship between them!
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    return 2 * sigmoid(2 * x) - 1


def tanh_derivative(x):
    """
    Derivative of tanh.

    tanh'(x) = 1 - tanh(x)^2

    Notice: max derivative is 1 (at x=0)
    Sigmoid's max derivative is only 0.25!
    """
    t = tanh(x)
    return 1 - t ** 2


def tanh_derivative_from_output(tanh_output):
    """
    If we already have tanh output.
    Useful during backprop.
    """
    return 1 - tanh_output ** 2


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Tanh Activation Function")
    print("=" * 60)

    # Test values
    x = np.array([-5, -2, -1, 0, 1, 2, 5])

    print("\nInput values:", x)
    print("Tanh output:", np.round(tanh(x), 4))
    print("Tanh derivative:", np.round(tanh_derivative(x), 4))

    # Verify different implementations match
    print("\n" + "-" * 40)
    print("Verifying implementations:")
    print("-" * 40)
    print("Built-in:", np.round(tanh(x), 4))
    print("Manual:  ", np.round(tanh_manual(x), 4))
    print("Sigmoid: ", np.round(tanh_from_sigmoid(x), 4))
    print("All match:", np.allclose(tanh(x), tanh_manual(x)) and np.allclose(tanh(x), tanh_from_sigmoid(x)))

    # Compare tanh vs sigmoid
    print("\n" + "=" * 60)
    print("Tanh vs Sigmoid Comparison")
    print("=" * 60)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    test_vals = [-3, -1, 0, 1, 3]
    print("\nOutputs:")
    print(f"{'x':>5} | {'sigmoid':>10} | {'tanh':>10}")
    print("-" * 30)
    for v in test_vals:
        print(f"{v:5.1f} | {sigmoid(v):10.4f} | {tanh(v):10.4f}")

    print("\nDerivatives:")
    print(f"{'x':>5} | {'sigmoid':>10} | {'tanh':>10}")
    print("-" * 30)
    for v in test_vals:
        print(f"{v:5.1f} | {sigmoid_derivative(v):10.4f} | {tanh_derivative(v):10.4f}")

    print("\nKey differences:")
    print("- Sigmoid range: (0, 1), Tanh range: (-1, 1)")
    print("- Sigmoid max derivative: 0.25, Tanh max derivative: 1.0")
    print("- Tanh is zero-centered, sigmoid is not")

    # Show zero-centering benefit
    print("\n" + "=" * 60)
    print("Why Zero-Centering Matters")
    print("=" * 60)

    print("\nImagine a layer's outputs are always positive (like sigmoid):")
    print("- All gradients to the weights have the same sign")
    print("- Updates can only move in certain 'quadrants'")
    print("- Learning zigzags instead of going straight")

    print("\nWith zero-centered outputs (like tanh):")
    print("- Gradients can be positive or negative")
    print("- Updates can point in any direction")
    print("- Learning is more direct")

    # ASCII visualization
    print("\n" + "=" * 60)
    print("Tanh Shape (ASCII):")
    print("=" * 60)

    width = 50
    height = 10

    x_vals = np.linspace(-4, 4, width)
    y_vals = tanh(x_vals)

    for row in range(height, -1, -1):
        y_level = -1 + 2 * row / height  # -1 to 1
        line = ""
        for col in range(width):
            if abs(y_vals[col] - y_level) < 0.15:
                line += "*"
            elif abs(y_level) < 0.1:
                line += "-"
            elif col == width // 2:
                line += "|"
            else:
                line += " "
        label = " 1" if row == height else (" 0" if row == height//2 else ("-1" if row == 0 else "  "))
        print(f"{label} | {line}")
    print(f"   +{'-' * width}")
    print(f"   -4{'': ^{width-4}}+4")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Tanh squashes to (-1, 1) - zero-centered")
    print("2. Derivative = 1 - tanh(x)^2")
    print("3. Stronger gradients than sigmoid")
    print("4. Good for RNNs and when you need centered outputs")
    print("5. Still has vanishing gradient for large inputs")
    print("6. ReLU is usually better for deep feedforward networks")
