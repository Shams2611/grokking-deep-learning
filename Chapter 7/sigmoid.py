"""
Chapter 7: Sigmoid Activation Function

The OG activation function! Used since the 1980s.

sigmoid(x) = 1 / (1 + e^(-x))

What it does:
- Squashes any input to a value between 0 and 1
- Large negative inputs -> close to 0
- Large positive inputs -> close to 1
- Zero input -> 0.5 (right in the middle)

Why it's useful:
- Output looks like a probability (0 to 1)
- Smooth and differentiable everywhere
- Historically important

Why it's NOT used much anymore for hidden layers:
- "Vanishing gradient" problem: gradients become tiny for large inputs
- Outputs aren't centered at 0 (causes zigzagging during training)
- Slower than ReLU

Still used for:
- Output layer for binary classification (probability of class 1)
- Gates in LSTMs and GRUs
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.

    Squashes any value to (0, 1)
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid.

    The beautiful thing: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

    If we already computed sigmoid(x), the derivative is cheap!
    """
    s = sigmoid(x)
    return s * (1 - s)


def sigmoid_derivative_from_output(sigmoid_output):
    """
    If we already have sigmoid output, derivative is even simpler.

    This is what we usually use during backprop.
    """
    return sigmoid_output * (1 - sigmoid_output)


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Sigmoid Activation Function")
    print("=" * 60)

    # Test values
    x = np.array([-5, -2, -1, 0, 1, 2, 5])

    print("\nInput values:", x)
    print("Sigmoid output:", np.round(sigmoid(x), 4))
    print("Sigmoid derivative:", np.round(sigmoid_derivative(x), 4))

    # Show the squashing behavior
    print("\n" + "-" * 40)
    print("Squashing behavior:")
    print("-" * 40)

    test_vals = [-100, -10, -5, -1, 0, 1, 5, 10, 100]
    for v in test_vals:
        s = sigmoid(v)
        print(f"  sigmoid({v:4d}) = {s:.10f}")

    print("\nNotice: very negative -> ~0, very positive -> ~1")

    # Vanishing gradient problem
    print("\n" + "=" * 60)
    print("The Vanishing Gradient Problem")
    print("=" * 60)

    print("\nDerivatives at different inputs:")
    for v in test_vals:
        deriv = sigmoid_derivative(v)
        print(f"  sigmoid'({v:4d}) = {deriv:.10f}")

    print("\nNotice: derivatives get TINY for large inputs!")
    print("This means gradients basically disappear in deep networks.")
    print("Neurons with large activations stop learning - bad!")

    # Binary classification example
    print("\n" + "=" * 60)
    print("Sigmoid for Binary Classification")
    print("=" * 60)

    # Pretend these are raw outputs from a network (logits)
    logits = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    probabilities = sigmoid(logits)

    print("\nRaw network outputs (logits):", logits)
    print("After sigmoid (probabilities):", np.round(probabilities, 4))

    print("\nDecisions (threshold = 0.5):")
    for l, p in zip(logits, probabilities):
        decision = "Class 1" if p >= 0.5 else "Class 0"
        print(f"  logit={l:5.1f} -> prob={p:.4f} -> {decision}")

    # ASCII visualization
    print("\n" + "=" * 60)
    print("Shape of Sigmoid (ASCII):")
    print("=" * 60)

    def ascii_sigmoid():
        width = 50
        height = 10

        x_vals = np.linspace(-6, 6, width)
        y_vals = sigmoid(x_vals)

        for row in range(height, -1, -1):
            y_level = row / height
            line = ""
            for col in range(width):
                if abs(y_vals[col] - y_level) < 0.1:
                    line += "*"
                elif abs(y_level - 0.5) < 0.05:
                    line += "-"
                elif col == width // 2:
                    line += "|"
                else:
                    line += " "
            label = "1.0" if row == height else ("0.5" if row == height//2 else ("0.0" if row == 0 else "   "))
            print(f"{label} | {line}")
        print(f"    +{'-' * width}")
        print(f"    -6{'': ^{width-4}}+6")

    ascii_sigmoid()

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Sigmoid squashes to (0, 1) - great for probabilities")
    print("2. Derivative = sigmoid(x) * (1 - sigmoid(x)) - elegant!")
    print("3. Vanishing gradient problem limits its use in hidden layers")
    print("4. Still used for output layer in binary classification")
    print("5. Also used in LSTM gates (which we'll see later)")
