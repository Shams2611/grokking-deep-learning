"""
Chapter 3: Forward Propagation - Multiple Inputs

My notes on how neural networks handle multiple inputs!

So far we've seen a simple network: one input, one weight, one output.
But real problems have LOTS of inputs. Like predicting if it'll rain:
- temperature
- humidity
- wind speed
- etc.

The key insight: each input gets its OWN weight!

prediction = input1 * weight1 + input2 * weight2 + input3 * weight3 + ...

This is literally just a weighted sum (or "dot product" if you wanna be fancy).

Think of it like this: you're rating a restaurant
- Food quality (input1) matters A LOT to you (weight1 = 0.5)
- Ambiance (input2) matters a bit (weight2 = 0.2)
- Price (input3) matters some (weight3 = 0.3)

Your overall rating = food*0.5 + ambiance*0.2 + price*0.3
"""

import numpy as np


def weighted_sum(inputs, weights):
    """
    The explicit way - so we can see exactly what's happening.
    Just multiply each input by its weight and add them all up.
    """
    # Make sure we have the same number of inputs and weights!
    assert len(inputs) == len(weights), "Oops! Need same number of inputs and weights"

    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]

    return output


def neural_network(inputs, weights):
    """
    The numpy way - does the same thing but faster.
    np.dot() computes the dot product for us.
    """
    prediction = np.dot(inputs, weights)
    return prediction


# ============================================
# Let's test this out!
# ============================================
if __name__ == "__main__":

    print("=" * 50)
    print("Testing Multiple Inputs Neural Network")
    print("=" * 50)

    # Example: Predicting if a sports team will win
    # Inputs are their stats from the season

    # toes = avg number of toes (weird stat but from the book lol)
    # wlrec = win/loss record percentage
    # nfans = number of fans (in millions)

    toes = 8.5
    wlrec = 0.65  # 65% win rate
    nfans = 1.2   # 1.2 million fans

    inputs = [toes, wlrec, nfans]

    # These weights were "learned" (for now we just make them up)
    # Later we'll learn how to find good weights automatically!
    weights = [0.1, 0.2, 0.0]  # nfans has 0 weight - doesn't affect prediction

    print(f"\nInputs: toes={toes}, win_rate={wlrec}, fans={nfans}")
    print(f"Weights: {weights}")

    # Try both methods
    pred1 = weighted_sum(inputs, weights)
    pred2 = neural_network(inputs, weights)

    print(f"\nWeighted sum method: {pred1}")
    print(f"Numpy dot product:   {pred2}")

    # Let's break it down step by step
    print("\n--- Breaking it down ---")
    print(f"  {toes} * {weights[0]} = {toes * weights[0]}")
    print(f"  {wlrec} * {weights[1]} = {wlrec * weights[1]}")
    print(f"  {nfans} * {weights[2]} = {nfans * weights[2]}")
    print(f"  Total: {pred1}")

    # Another example with different inputs
    print("\n" + "=" * 50)
    print("Another game, different stats:")
    print("=" * 50)

    inputs2 = [9.0, 0.85, 0.9]  # better record this time!
    pred = neural_network(inputs2, weights)
    print(f"Inputs: {inputs2}")
    print(f"Prediction: {pred}")
    print(f"Higher prediction because better win rate!")
