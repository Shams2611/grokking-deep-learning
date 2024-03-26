"""
Chapter 5: Gradient Descent with Multiple Outputs

What if we're predicting multiple things at once?

Each output has its own goal, its own error, and its own set of weights.
We basically run gradient descent INDEPENDENTLY for each output!

The outputs don't affect each other during training (for now).
They share the same input but have separate weights.

Example:
- Input: number of hours studied
- Output 1: test score prediction (weights1)
- Output 2: confidence level (weights2)

Each output learns its own relationship with the input.
"""

import numpy as np


def neural_network(input_val, weights):
    """
    Single input, multiple outputs.
    Each weight gives us one output.
    """
    return input_val * np.array(weights)


def train_multiple_outputs(input_val, goals, weights, lr=0.1, iterations=20):
    """
    Train weights for multiple outputs.

    Each output is independent - we just run gradient descent
    separately for each one. But we can do it in parallel with vectors!
    """
    weights = np.array(weights, dtype=float)
    goals = np.array(goals, dtype=float)

    for i in range(iterations):
        # Forward pass - all outputs at once
        preds = neural_network(input_val, weights)

        # Error for each output
        errors = (preds - goals) ** 2
        deltas = preds - goals

        # Gradient for each weight (same formula, just vectorized)
        gradients = deltas * input_val

        # Update all weights at once
        weights = weights - (lr * gradients)

        total_error = np.sum(errors)
        print(f"Iter {i}: preds={preds}, total_error={total_error:.6f}")

    return weights


# ============================================
# Demo!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Gradient Descent with Multiple Outputs")
    print("=" * 60)

    # Single input
    input_val = 0.65  # win/loss record

    # Multiple goals (what we want each output to be)
    goals = [0.1, 1.0, 0.5]  # [hurt?, win?, sad?]

    # Initial weights - one for each output
    initial_weights = [0.3, 0.2, 0.9]

    print(f"\nInput (win rate): {input_val}")
    print(f"Goals: hurt={goals[0]}, win={goals[1]}, sad={goals[2]}")
    print(f"Initial weights: {initial_weights}")

    # Initial predictions
    init_preds = neural_network(input_val, initial_weights)
    print(f"Initial predictions: {init_preds}")
    print(f"Initial errors: {(init_preds - goals)**2}")

    print("\n--- Training ---\n")
    final_weights = train_multiple_outputs(
        input_val, goals, initial_weights, lr=0.5, iterations=10
    )

    print(f"\n--- Results ---")
    print(f"Final weights: {final_weights}")
    final_preds = neural_network(input_val, final_weights)
    print(f"Final predictions: {final_preds}")
    print(f"Goals were: {goals}")

    # What should the weights be?
    print("\n" + "=" * 60)
    print("Sanity check - what SHOULD the weights be?")
    print("=" * 60)
    print(f"For output = input * weight:")
    print(f"  weight = output / input = goal / input")
    for i, g in enumerate(goals):
        correct_w = g / input_val
        print(f"  weight[{i}] should be: {g} / {input_val} = {correct_w:.4f}")

    print("\nCompare to learned weights:", final_weights)
    print("Pretty close! (More iterations would get even closer)")

    # Key insight
    print("\n" + "=" * 60)
    print("Key Insight:")
    print("=" * 60)
    print("Each output learns independently!")
    print("The weight for 'hurt' doesn't affect the weight for 'win'.")
    print("This is because each output has its own error and own weight.")
    print("\nLater, with hidden layers, things get more interesting -")
    print("errors from different outputs WILL affect shared weights!")
