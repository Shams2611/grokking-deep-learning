"""
Chapter 11: Simple RNN - Recurrent Neural Networks

RNNs are neural networks with MEMORY!

Regular neural networks:
    input -> network -> output
    (each input is independent)

Recurrent neural networks:
    input1 -> network -> output1
                |
                v (hidden state carries forward!)
    input2 -> network -> output2
                |
                v
    input3 -> network -> output3

The "hidden state" is the memory - it gets updated at each step
and passed to the next step. This lets RNNs handle SEQUENCES!

Applications:
- Text (sequence of words/characters)
- Time series (sequence of measurements)
- Speech (sequence of audio frames)
- Music generation

The key equation:
    h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)

h_t = new hidden state (memory)
h_{t-1} = previous hidden state
x_t = current input
W_hh = weights connecting hidden to hidden (memory)
W_xh = weights connecting input to hidden
"""

import numpy as np

np.random.seed(42)


class SimpleRNN:
    """A simple vanilla RNN cell."""

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN.

        Args:
            input_size: dimension of input at each timestep
            hidden_size: dimension of hidden state (memory)
            output_size: dimension of output
        """
        self.hidden_size = hidden_size

        # Initialize weights (Xavier initialization)
        scale = 1.0 / np.sqrt(hidden_size)

        # Input to hidden weights
        self.W_xh = np.random.randn(input_size, hidden_size) * scale

        # Hidden to hidden weights (the recurrent connection!)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale

        # Hidden to output weights
        self.W_hy = np.random.randn(hidden_size, output_size) * scale

        # Biases
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)

    def forward_step(self, x, h_prev):
        """
        One step of the RNN.

        Args:
            x: current input
            h_prev: previous hidden state

        Returns:
            h_new: new hidden state
            y: output
        """
        # Combine previous hidden state and current input
        h_new = np.tanh(
            np.dot(x, self.W_xh) +
            np.dot(h_prev, self.W_hh) +
            self.b_h
        )

        # Compute output
        y = np.dot(h_new, self.W_hy) + self.b_y

        return h_new, y

    def forward(self, inputs):
        """
        Forward pass through entire sequence.

        Args:
            inputs: list of inputs for each timestep

        Returns:
            outputs: list of outputs
            hidden_states: list of hidden states
        """
        # Initialize hidden state to zeros
        h = np.zeros(self.hidden_size)

        outputs = []
        hidden_states = [h]

        for x in inputs:
            h, y = self.forward_step(x, h)
            hidden_states.append(h)
            outputs.append(y)

        return outputs, hidden_states


def one_hot(idx, size):
    """Create one-hot vector."""
    vec = np.zeros(size)
    vec[idx] = 1
    return vec


# ============================================
# Demo: Character-level RNN
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Simple RNN - Recurrent Neural Network")
    print("=" * 60)

    # Simple character prediction example
    print("\n--- Character-level RNN Demo ---")

    # Our vocabulary
    chars = ['h', 'e', 'l', 'o', ' ']
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    vocab_size = len(chars)
    hidden_size = 10

    print(f"\nVocabulary: {chars}")
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden size: {hidden_size}")

    # Create RNN
    rnn = SimpleRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=vocab_size
    )

    # Input sequence: "hello"
    text = "hello"
    print(f"\nInput sequence: '{text}'")

    # Convert to one-hot encoded inputs
    inputs = [one_hot(char_to_idx[c], vocab_size) for c in text]

    print("\nOne-hot encoded inputs:")
    for i, c in enumerate(text):
        print(f"  '{c}' -> {inputs[i]}")

    # Forward pass
    outputs, hidden_states = rnn.forward(inputs)

    print("\n--- Forward Pass Results ---")
    print(f"\nNumber of timesteps: {len(inputs)}")

    for t, (inp_char, out, h) in enumerate(zip(text, outputs, hidden_states[1:])):
        pred_idx = np.argmax(out)
        pred_char = idx_to_char[pred_idx]
        print(f"\nTimestep {t}:")
        print(f"  Input: '{inp_char}'")
        print(f"  Hidden state (first 5): {h[:5].round(3)}")
        print(f"  Output scores: {out.round(3)}")
        print(f"  Predicted next char: '{pred_char}'")

    # Show how hidden state evolves
    print("\n" + "=" * 60)
    print("Hidden State Evolution (Memory)")
    print("=" * 60)

    print("\nThe hidden state changes at each timestep, encoding context:")
    for t in range(len(hidden_states)):
        state_summary = f"norm={np.linalg.norm(hidden_states[t]):.3f}"
        print(f"  Step {t}: {state_summary} {hidden_states[t][:3].round(3)}...")

    # Explain the recurrence
    print("\n" + "=" * 60)
    print("The Recurrence Visualized")
    print("=" * 60)

    print("""
    h = 'h' ─────┐
                 ├──> [RNN] ──> h1 ──> predict 'e'
    h0 = 0 ─────┘         │
                          │ (h1 carries forward!)
    e = 'e' ─────┐        │
                 ├──> [RNN] ──> h2 ──> predict 'l'
    h1 ──────────┘         │
                          │
    l = 'l' ─────┐        │
                 ├──> [RNN] ──> h3 ──> predict 'l'
    h2 ──────────┘         │
                          ... and so on

    The hidden state h_t encodes everything seen so far!
    """)

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. RNNs have a 'hidden state' that acts as memory")
    print("2. At each step: new_hidden = tanh(W_hh * old_hidden + W_xh * input)")
    print("3. The hidden state accumulates information over time")
    print("4. Same weights are used at every timestep (parameter sharing)")
    print("5. Problem: gradients can vanish/explode over long sequences")
    print("6. Solution: LSTM and GRU (coming next!)")
