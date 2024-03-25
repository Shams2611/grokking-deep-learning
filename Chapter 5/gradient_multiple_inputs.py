"""
Chapter 5: Generalizing Gradient Descent - Multiple Inputs

Now we level up! Instead of one input, we have MULTIPLE inputs.
Each input has its own weight, and we need to update ALL of them.

The beautiful thing: the math is almost the same!

For each weight:
    gradient_i = delta * input_i

That's it. Each weight's gradient depends on:
1. How wrong our prediction was (delta)
2. How much that particular input contributed (input_i)

This makes intuitive sense:
- If an input was 0, changing its weight won't help (gradient = 0)
- If an input was large, changing its weight has a big effect

It's like figuring out who to blame when a group project fails.
The person who contributed most to the bad parts gets "blamed" more.
"""

import numpy as np


def neural_network(inputs, weights):
    """Forward pass with multiple inputs."""
    return np.dot(inputs, weights)


def gradient_descent_multiple_inputs(inputs, goal, weights, lr=0.01, iterations=20):
    """
    Learn weights for multiple inputs.

    The gradient for each weight is: delta * that_input

    We update each weight proportional to how much it contributed
    to the error.
    """
    weights = np.array(weights, dtype=float)
    inputs = np.array(inputs, dtype=float)

    history = []

    for i in range(iterations):
        # Forward pass
        pred = neural_network(inputs, weights)

        # Error (for logging)
        error = (pred - goal) ** 2
        delta = pred - goal

        # Calculate gradient for each weight
        # gradient = delta * inputs (element-wise)
        gradients = delta * inputs

        # Update all weights
        weights = weights - (lr * gradients)

        history.append({
            'iter': i,
            'pred': pred,
            'error': error,
            'weights': weights.copy()
        })

        print(f"Iter {i}: pred={pred:.4f}, error={error:.6f}")

    return weights, history


# ============================================
# Demo!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Gradient Descent with Multiple Inputs")
    print("=" * 60)

    # Same example from Chapter 3
    toes = 8.5
    wlrec = 0.65
    nfans = 1.2

    inputs = [toes, wlrec, nfans]
    initial_weights = [0.1, 0.1, 0.1]  # start with random weights

    goal = 1.0  # target prediction

    print(f"\nInputs: toes={toes}, wlrec={wlrec}, nfans={nfans}")
    print(f"Initial weights: {initial_weights}")
    print(f"Goal: {goal}")

    # Initial prediction
    init_pred = neural_network(inputs, initial_weights)
    print(f"Initial prediction: {init_pred}")
    print(f"Initial error: {(init_pred - goal)**2:.6f}")

    print("\n--- Training ---\n")
    final_weights, history = gradient_descent_multiple_inputs(
        inputs, goal, initial_weights, lr=0.01, iterations=15
    )

    print(f"\n--- Results ---")
    print(f"Final weights: {final_weights}")
    print(f"Final prediction: {neural_network(inputs, final_weights):.4f}")

    # Let's trace through one update manually
    print("\n" + "=" * 60)
    print("Manual trace of weight updates (first iteration):")
    print("=" * 60)

    weights = np.array([0.1, 0.1, 0.1])
    inp = np.array([8.5, 0.65, 1.2])
    g = 1.0
    learning_rate = 0.01

    pred = np.dot(inp, weights)
    print(f"prediction = {inp} · {weights} = {pred}")

    delta = pred - g
    print(f"delta = pred - goal = {pred} - {g} = {delta}")

    grads = delta * inp
    print(f"\nGradients for each weight:")
    print(f"  grad_toes  = delta * toes  = {delta} * {inp[0]} = {grads[0]:.4f}")
    print(f"  grad_wlrec = delta * wlrec = {delta} * {inp[1]} = {grads[1]:.4f}")
    print(f"  grad_nfans = delta * nfans = {delta} * {inp[2]} = {grads[2]:.4f}")

    print(f"\nWeight updates (lr={learning_rate}):")
    for j in range(3):
        new_w = weights[j] - learning_rate * grads[j]
        print(f"  w[{j}]: {weights[j]:.4f} - {learning_rate} * {grads[j]:.4f} = {new_w:.4f}")

    print("\n" + "=" * 60)
    print("Key Insight:")
    print("=" * 60)
    print("Notice 'toes' has the BIGGEST gradient because it's the biggest input.")
    print("Bigger inputs have more 'responsibility' for the prediction,")
    print("so their weights get updated more aggressively.")
