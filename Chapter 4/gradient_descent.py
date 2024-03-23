"""
Chapter 4: Gradient Descent - The Real Deal

Okay so hot/cold learning works but it's super slow.
We have to try BOTH directions for each weight.

What if we could just KNOW which direction to go?

That's gradient descent!

The gradient tells us:
1. Which direction makes error go DOWN
2. How MUCH to change the weight

The math:
- error = (prediction - goal)^2
- derivative = 2 * (prediction - goal) * input
- This tells us the SLOPE - which way is downhill!

If the derivative is positive -> weight is too big -> decrease it
If the derivative is negative -> weight is too small -> increase it

We just go in the OPPOSITE direction of the gradient!

weight_new = weight_old - (learning_rate * gradient)

The minus sign is because we want to go DOWNHILL (minimize error).
"""

import numpy as np


def simple_network(input_val, weight):
    """Forward pass - just prediction."""
    return input_val * weight


def gradient_descent(input_val, goal, initial_weight=0.0, lr=0.1, iterations=20):
    """
    Learn using gradient descent.

    Instead of trying both directions, we CALCULATE which way to go!

    The gradient of squared error is:
    d(error)/d(weight) = 2 * (pred - goal) * input

    We can simplify to just: (pred - goal) * input
    (the 2 just gets absorbed into learning rate anyway)
    """
    weight = initial_weight
    history = []

    for i in range(iterations):
        # Forward pass
        pred = simple_network(input_val, weight)

        # How wrong are we?
        error = (pred - goal) ** 2
        delta = pred - goal  # this is also called "pure error"

        # The gradient! This is the magic sauce
        gradient = delta * input_val

        # Update weight - go opposite direction of gradient
        weight = weight - (lr * gradient)

        history.append({
            'iter': i,
            'weight': weight,
            'pred': pred,
            'error': error,
            'delta': delta,
            'gradient': gradient
        })

        # Print progress
        print(f"Iter {i}: weight={weight:.4f}, pred={pred:.4f}, error={error:.6f}")

    return weight, history


# ============================================
# Demo!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Gradient Descent Learning")
    print("=" * 60)

    input_val = 2.0
    goal = 0.8

    print(f"\nInput: {input_val}")
    print(f"Goal: {goal}")
    print(f"Target weight: {goal / input_val}")

    print("\n--- Training with Gradient Descent ---\n")
    final_weight, history = gradient_descent(input_val, goal, initial_weight=0.0, lr=0.1, iterations=10)

    print(f"\n--- Results ---")
    print(f"Learned weight: {final_weight:.6f}")
    print(f"Expected weight: {goal / input_val:.6f}")

    # Let's trace through one iteration manually
    print("\n" + "=" * 60)
    print("Manual trace of first iteration:")
    print("=" * 60)

    weight = 0.0
    input_v = 2.0
    goal_v = 0.8
    lr = 0.1

    print(f"Starting weight: {weight}")
    print(f"prediction = input * weight = {input_v} * {weight} = {input_v * weight}")
    pred = input_v * weight

    print(f"delta = pred - goal = {pred} - {goal_v} = {pred - goal_v}")
    delta = pred - goal_v

    print(f"gradient = delta * input = {delta} * {input_v} = {delta * input_v}")
    gradient = delta * input_v

    print(f"weight_change = lr * gradient = {lr} * {gradient} = {lr * gradient}")
    weight_change = lr * gradient

    print(f"new_weight = weight - change = {weight} - ({weight_change}) = {weight - weight_change}")

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("1. We never tried 'both directions' - gradient told us where to go")
    print("2. Bigger errors = bigger gradients = bigger weight updates")
    print("3. As we get closer, updates get smaller (natural 'fine-tuning')")
    print("4. Learning rate controls how big each step is")
