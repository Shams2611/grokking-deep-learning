"""
Chapter 3: Forward Propagation - Multiple Outputs

What if we want to predict MORE than one thing at a time?

Like, given a player's stats, we might want to predict:
- Will they score?
- Will they get injured?
- Are they happy/sad?

The trick: each OUTPUT gets its own set of weights!

So if we have 1 input and 3 outputs, we need 3 weights.
If we have 3 inputs and 3 outputs... we need 9 weights (a whole matrix!)

But let's start simple: 1 input, multiple outputs.
"""

import numpy as np


def neural_network(input_value, weights):
    """
    One input, multiple outputs.

    Each weight produces one output.
    It's like asking 3 different questions about the same input.
    """
    # weights is a list/array - each one gives us a different prediction
    predictions = []

    for weight in weights:
        pred = input_value * weight
        predictions.append(pred)

    return predictions


def neural_network_numpy(input_value, weights):
    """
    Same thing but with numpy - way cleaner!

    Scalar times a vector = vector (element-wise multiplication)
    """
    return input_value * np.array(weights)


# ============================================
# Let's test this out!
# ============================================
if __name__ == "__main__":

    print("=" * 50)
    print("Multiple Outputs from Single Input")
    print("=" * 50)

    # Our input: just the win/loss record
    wlrec = 0.65  # 65% win rate

    # Three different weights for three different predictions
    # weight for: [hurt?, win?, sad?]
    weights = [0.3, 0.2, 0.9]

    print(f"\nInput (win rate): {wlrec}")
    print(f"Weights: {weights}")
    print("  weights[0] -> hurt prediction")
    print("  weights[1] -> win prediction")
    print("  weights[2] -> sad prediction")

    # Get predictions
    preds = neural_network(wlrec, weights)

    print(f"\nPredictions:")
    print(f"  Hurt?  {preds[0]:.4f}")
    print(f"  Win?   {preds[1]:.4f}")
    print(f"  Sad?   {preds[2]:.4f}")

    # Verify with numpy version
    preds_np = neural_network_numpy(wlrec, weights)
    print(f"\nNumpy version gives same results: {np.allclose(preds, preds_np)}")

    # Let's break it down
    print("\n--- The Math ---")
    print(f"  hurt = {wlrec} * {weights[0]} = {wlrec * weights[0]}")
    print(f"  win  = {wlrec} * {weights[1]} = {wlrec * weights[1]}")
    print(f"  sad  = {wlrec} * {weights[2]} = {wlrec * weights[2]}")

    print("\n" + "=" * 50)
    print("Interesting observation:")
    print("=" * 50)
    print("High weight on 'sad' (0.9) means the network thinks")
    print("win rate strongly correlates with sadness... which is")
    print("kinda backwards lol. But remember - we're not learning")
    print("weights yet, just making them up!")
