"""
Chapter 3: Forward Propagation - Multiple Inputs AND Multiple Outputs

Now we're getting to the good stuff!

Multiple inputs -> Multiple outputs

This is basically what a full neural network layer looks like.
We need a MATRIX of weights now!

If we have:
- 3 inputs
- 3 outputs

We need a 3x3 weight matrix = 9 weights total.

Each output is computed by taking a dot product of ALL inputs
with that output's row of weights.

Visual:
          inputs
            |
            v
    [i1, i2, i3]
          |
          v
    +-------------+
    | w11 w12 w13 |  -> output1 = i1*w11 + i2*w12 + i3*w13
    | w21 w22 w23 |  -> output2 = i1*w21 + i2*w22 + i3*w23
    | w31 w32 w33 |  -> output3 = i1*w31 + i2*w32 + i3*w33
    +-------------+
          |
          v
    [o1, o2, o3]
"""

import numpy as np


def neural_network_explicit(inputs, weights):
    """
    The manual way - so we really understand what's happening.

    For each output, we compute a weighted sum of ALL inputs.
    """
    outputs = []

    # weights is a list of lists (matrix)
    # each row contains the weights for one output
    for weight_row in weights:
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * weight_row[i]
        outputs.append(output)

    return outputs


def neural_network_numpy(inputs, weights):
    """
    The clean numpy way.

    Matrix multiplication does all the dot products at once!
    weights @ inputs = outputs

    Note: we need inputs as column vector, or just use .dot()
    """
    inputs = np.array(inputs)
    weights = np.array(weights)

    # Matrix-vector multiplication
    # Each row of weights dotted with inputs gives one output
    return weights.dot(inputs)


def vector_matrix_multiply(inputs, weights):
    """
    Another way to think about it - as multiple weighted sums.

    This is exactly what np.dot does under the hood.
    """
    outputs = np.zeros(len(weights))

    for i, weight_row in enumerate(weights):
        outputs[i] = np.dot(inputs, weight_row)

    return outputs


# ============================================
# Demo time!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Multiple Inputs -> Multiple Outputs")
    print("=" * 60)

    # Same inputs as before
    toes = 8.5
    wlrec = 0.65
    nfans = 1.2

    inputs = [toes, wlrec, nfans]

    # Now we need a weight MATRIX
    # Each row = weights for one output
    # Format: weights[output_idx][input_idx]

    weights = [
        [0.1, 0.1, -0.3],   # weights for "hurt" prediction
        [0.1, 0.2, 0.0],    # weights for "win" prediction
        [0.0, 1.3, 0.1]     # weights for "sad" prediction
    ]

    print(f"\nInputs: toes={toes}, wlrec={wlrec}, nfans={nfans}")
    print(f"\nWeight matrix:")
    for i, row in enumerate(weights):
        print(f"  Output {i}: {row}")

    # Calculate outputs using all three methods
    out1 = neural_network_explicit(inputs, weights)
    out2 = neural_network_numpy(inputs, weights)
    out3 = vector_matrix_multiply(inputs, weights)

    print(f"\n--- Predictions ---")
    print(f"Hurt?: {out1[0]:.4f}")
    print(f"Win?:  {out1[1]:.4f}")
    print(f"Sad?:  {out1[2]:.4f}")

    print(f"\nAll methods match: {np.allclose(out1, out2) and np.allclose(out2, out3)}")

    # Let's trace through one calculation manually
    print("\n" + "=" * 60)
    print("Manual trace for 'hurt' prediction (first output):")
    print("=" * 60)
    print(f"  hurt_weights = {weights[0]}")
    print(f"  hurt = toes*w1 + wlrec*w2 + nfans*w3")
    print(f"  hurt = {toes}*{weights[0][0]} + {wlrec}*{weights[0][1]} + {nfans}*{weights[0][2]}")
    print(f"  hurt = {toes*weights[0][0]} + {wlrec*weights[0][1]} + {nfans*weights[0][2]}")
    print(f"  hurt = {out1[0]:.4f}")

    print("\n" + "=" * 60)
    print("Key Insight:")
    print("=" * 60)
    print("Notice nfans has a NEGATIVE weight (-0.3) for hurt prediction.")
    print("This means more fans -> less likely to be hurt (in this model).")
    print("The network 'learns' these relationships from data!")
    print("\nRight now we're just making up weights, but soon we'll")
    print("learn how to FIND good weights automatically. That's the")
    print("magic of machine learning!")
