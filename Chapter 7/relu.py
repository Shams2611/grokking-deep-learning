"""
Chapter 7: ReLU - Rectified Linear Unit

The most popular activation function in deep learning!

ReLU(x) = max(0, x)

That's it. Dead simple. If x is negative, output 0. Otherwise, output x.

Why is it so good?
1. Super fast to compute
2. Doesn't have the "vanishing gradient" problem
3. Introduces non-linearity (which we NEED)
4. Works great in practice

The derivative is also simple:
- If x > 0: derivative = 1
- If x < 0: derivative = 0
- At x = 0: technically undefined, but we usually say 0

One issue: "dead neurons" - if a neuron always outputs 0, it stops learning.
That's why we have variants like Leaky ReLU.
"""

import numpy as np


def relu(x):
    """
    ReLU activation function.

    max(0, x) for each element
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU.

    1 if x > 0, else 0
    """
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU - allows small gradient when x < 0

    Instead of outputting 0 for negative inputs,
    output alpha * x (a small slope).

    This prevents "dead neurons"!
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU."""
    return np.where(x > 0, 1, alpha)


# ============================================
# Visualization and demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("ReLU - Rectified Linear Unit")
    print("=" * 60)

    # Test values
    x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])

    print("\nInput values:", x)
    print("\nReLU output:", relu(x))
    print("ReLU derivative:", relu_derivative(x))

    print("\n" + "-" * 40)
    print("Leaky ReLU (alpha=0.01):")
    print("Leaky ReLU output:", leaky_relu(x))
    print("Leaky derivative:", leaky_relu_derivative(x))

    # ASCII visualization
    print("\n" + "=" * 60)
    print("ASCII Plot of ReLU:")
    print("=" * 60)

    def ascii_plot(func, x_range, title):
        """Simple ASCII plot."""
        print(f"\n{title}")
        print("-" * 40)

        height = 10
        width = 40

        x_vals = np.linspace(x_range[0], x_range[1], width)
        y_vals = func(x_vals)

        y_min, y_max = min(y_vals.min(), 0), max(y_vals.max(), 1)

        for row in range(height, -1, -1):
            y_level = y_min + (y_max - y_min) * row / height
            line = ""
            for col in range(width):
                if abs(y_vals[col] - y_level) < (y_max - y_min) / height:
                    line += "*"
                elif row == height // 2 and col == width // 2:
                    line += "+"
                elif abs(y_level) < (y_max - y_min) / height:
                    line += "-"
                elif col == width // 2:
                    line += "|"
                else:
                    line += " "
            print(f"{y_level:5.1f} | {line}")
        print(f"      +{'-' * width}")
        print(f"       {x_range[0]}{'': ^{width-8}}{x_range[1]}")

    ascii_plot(relu, (-3, 3), "ReLU")

    # Demo in a simple network
    print("\n" + "=" * 60)
    print("ReLU in a Neural Network Layer")
    print("=" * 60)

    # Simulating a layer
    inputs = np.array([0.5, -0.3, 0.8])
    weights = np.array([
        [0.2, -0.5, 0.3],
        [-0.1, 0.4, -0.2],
        [0.6, 0.1, -0.3]
    ])

    print(f"\nInputs: {inputs}")
    print(f"\nWeights matrix:\n{weights}")

    # Linear transformation
    z = np.dot(inputs, weights)
    print(f"\nBefore ReLU (z = inputs @ weights): {z}")

    # Apply ReLU
    a = relu(z)
    print(f"After ReLU: {a}")

    print("\nNotice: negative values become 0!")
    print("This non-linearity is what allows networks to learn complex patterns.")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. ReLU = max(0, x) - simple!")
    print("2. Gradient = 1 (for x>0) or 0 (for x<0)")
    print("3. Fast to compute, works great in practice")
    print("4. Leaky ReLU prevents 'dead neurons' by allowing small negative gradients")
    print("5. ReLU is the default choice for hidden layers in most networks")
