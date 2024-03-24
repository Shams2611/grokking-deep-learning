"""
Chapter 4: Learning Rate Experiments

The learning rate (alpha, lr, η) is SUPER important!

Too small -> learning takes forever, might get stuck
Too big -> overshoots the target, might diverge (explode!)
Just right -> smooth, fast convergence

Let's experiment and see what happens!

This is one of the first "hyperparameters" you'll tune.
It's not learned - YOU have to choose it.
"""

import numpy as np


def train_with_lr(input_val, goal, lr, iterations=20, verbose=False):
    """
    Train a simple network with a given learning rate.
    Returns the error history so we can plot/compare.
    """
    weight = 0.0
    errors = []

    for i in range(iterations):
        pred = input_val * weight
        error = (pred - goal) ** 2
        errors.append(error)

        # Gradient descent update
        delta = pred - goal
        gradient = delta * input_val
        weight = weight - (lr * gradient)

        if verbose:
            print(f"Iter {i}: weight={weight:.4f}, error={error:.6f}")

        # Check for explosion (NaN or huge values)
        if np.isnan(weight) or abs(weight) > 1e10:
            print(f"DIVERGED at iteration {i}!")
            errors.extend([float('inf')] * (iterations - i - 1))
            break

    return errors, weight


def visualize_errors(all_errors, learning_rates):
    """
    Simple ASCII visualization of error curves.
    (Real projects would use matplotlib but we're keeping it simple!)
    """
    print("\nError over iterations (ASCII plot):")
    print("-" * 60)

    max_error = max(e for errors in all_errors for e in errors if e != float('inf') and e < 100)
    height = 10

    for lr, errors in zip(learning_rates, all_errors):
        print(f"\nLR = {lr}:")
        # Show first 15 iterations
        for i, err in enumerate(errors[:15]):
            if err == float('inf'):
                bar = "X DIVERGED"
            else:
                bar_len = int((min(err, max_error) / max_error) * 40)
                bar = "#" * bar_len
            print(f"  {i:2d} | {bar} ({err:.4f})" if err != float('inf') else f"  {i:2d} | {bar}")


# ============================================
# Experiment time!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Learning Rate Experiments")
    print("=" * 60)

    input_val = 2.0
    goal = 0.8

    print(f"\nProblem: input={input_val}, goal={goal}")
    print(f"Correct weight should be: {goal/input_val}")

    # Try different learning rates
    learning_rates = [0.01, 0.1, 0.5, 1.0, 1.5]

    print("\n" + "=" * 60)
    print("Testing different learning rates:")
    print("=" * 60)

    all_errors = []
    final_weights = []

    for lr in learning_rates:
        print(f"\n--- Learning Rate: {lr} ---")
        errors, final_weight = train_with_lr(input_val, goal, lr, iterations=20, verbose=True)
        all_errors.append(errors)
        final_weights.append(final_weight)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"{'LR':<8} {'Final Weight':<15} {'Final Error':<15} {'Status'}")
    print("-" * 60)

    for lr, errors, weight in zip(learning_rates, all_errors, final_weights):
        final_err = errors[-1] if errors[-1] != float('inf') else float('inf')
        if final_err == float('inf') or final_err > 1:
            status = "DIVERGED!"
        elif final_err < 0.0001:
            status = "Converged nicely"
        elif final_err < 0.01:
            status = "Good"
        else:
            status = "Still learning..."

        print(f"{lr:<8} {weight:<15.6f} {final_err:<15.6f} {status}")

    # Visualize
    visualize_errors(all_errors, learning_rates)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. lr=0.01: Too slow! Would need many more iterations")
    print("2. lr=0.1:  Good balance - converges smoothly")
    print("3. lr=0.5:  Faster but might oscillate a bit")
    print("4. lr=1.0:  Right on the edge - might oscillate")
    print("5. lr=1.5:  TOO BIG - overshoots and diverges!")
    print("\nIn practice, people often start with lr=0.01 or 0.001")
    print("and use techniques like 'learning rate decay' to adjust over time.")
