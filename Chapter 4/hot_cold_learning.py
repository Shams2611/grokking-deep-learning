"""
Chapter 4: Introduction to Neural Learning - Hot and Cold Method

This is where things get REALLY interesting!

So far we've just been making up weights. But how do we actually
FIND good weights? This is the core of machine learning!

The simplest approach: hot and cold learning (or "guess and check")

1. Make a prediction
2. See how wrong we are (calculate error)
3. Wiggle each weight up and down a tiny bit
4. Keep the change that made error go DOWN

It's like playing "hotter/colder" as a kid - you just keep
moving in the direction that gets you closer to the goal!

This is NOT how real neural networks learn (too slow), but it
helps build intuition before we get to gradient descent.
"""

import numpy as np


def simple_network(input_val, weight):
    """Just multiply input by weight. Dead simple."""
    return input_val * weight


def hot_cold_learning(input_val, goal, initial_weight=0.5, step=0.001, max_iters=1000):
    """
    Learn the right weight by trial and error.

    The idea:
    - Try making weight slightly bigger
    - Try making weight slightly smaller
    - Keep whichever gives less error

    It's dumb but it works!
    """
    weight = initial_weight
    history = []  # track our progress

    for i in range(max_iters):
        # Current prediction and error
        pred = simple_network(input_val, weight)
        error = (pred - goal) ** 2  # squared error

        history.append({
            'iter': i,
            'weight': weight,
            'pred': pred,
            'error': error
        })

        # Are we close enough?
        if error < 0.0001:
            print(f"Converged at iteration {i}!")
            break

        # Try going up
        weight_up = weight + step
        pred_up = simple_network(input_val, weight_up)
        error_up = (pred_up - goal) ** 2

        # Try going down
        weight_down = weight - step
        pred_down = simple_network(input_val, weight_down)
        error_down = (pred_down - goal) ** 2

        # Which direction is better?
        if error_up < error_down:
            weight = weight_up  # going up helped!
        else:
            weight = weight_down  # going down helped!

    return weight, history


# ============================================
# Let's see it in action!
# ============================================
if __name__ == "__main__":

    print("=" * 50)
    print("Hot and Cold Learning")
    print("=" * 50)

    # Our training example
    input_val = 2.0
    goal = 0.8  # we want: input * weight = 0.8

    # So the "correct" weight should be 0.8 / 2.0 = 0.4
    print(f"\nInput: {input_val}")
    print(f"Goal output: {goal}")
    print(f"Correct weight should be: {goal / input_val}")

    print("\n--- Learning ---")
    final_weight, history = hot_cold_learning(input_val, goal, initial_weight=0.1)

    print(f"\nLearned weight: {final_weight}")
    print(f"Final prediction: {simple_network(input_val, final_weight)}")

    # Show how error decreased
    print("\n--- Learning Progress ---")
    print("Iter\tWeight\t\tPred\t\tError")
    print("-" * 50)

    # Show first 5 and last 5 iterations
    show_iters = history[:5] + [{'iter': '...', 'weight': '...', 'pred': '...', 'error': '...'}] + history[-5:]

    for h in show_iters:
        if h['iter'] == '...':
            print("...")
        else:
            print(f"{h['iter']}\t{h['weight']:.6f}\t{h['pred']:.6f}\t{h['error']:.8f}")

    print("\n" + "=" * 50)
    print("Why This Matters:")
    print("=" * 50)
    print("We just made a computer LEARN a weight automatically!")
    print("It started with a random guess and found the answer.")
    print("\nBut this method is SLOW - we have to try both directions")
    print("for every weight. Imagine doing this with millions of weights!")
    print("\nNext up: Gradient Descent - a smarter way to learn!")
