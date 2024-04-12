"""
Chapter 11: Backpropagation Through Time (BPTT)

How do we train RNNs? We need to backpropagate through TIME!

The challenge:
- RNN is "unrolled" over timesteps
- Each timestep uses the SAME weights
- Gradients need to flow backwards through all timesteps

BPTT = Backprop Through Time:
1. Unroll the RNN for the sequence length
2. Compute loss at each timestep
3. Backpropagate gradients through the entire unrolled network
4. Sum up gradients for shared weights
5. Update weights once

The problem: Vanishing/Exploding Gradients
- Gradients get multiplied at each step (by W_hh)
- If |W_hh| < 1: gradient shrinks exponentially (vanishes)
- If |W_hh| > 1: gradient grows exponentially (explodes)

This is why vanilla RNNs struggle with long sequences.
LSTMs and GRUs solve this problem!
"""

import numpy as np

np.random.seed(42)


class RNNWithBPTT:
    """RNN with full BPTT implementation."""

    def __init__(self, input_size, hidden_size, output_size):
        scale = 0.1

        self.hidden_size = hidden_size

        # Weights
        self.W_xh = np.random.randn(input_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.W_hy = np.random.randn(hidden_size, output_size) * scale

        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)

        # Storage for BPTT
        self.inputs = []
        self.hidden_states = []
        self.outputs = []

    def forward(self, inputs):
        """
        Forward pass - store everything for BPTT.
        """
        self.inputs = inputs
        self.hidden_states = [np.zeros(self.hidden_size)]

        outputs = []

        for x in inputs:
            h_prev = self.hidden_states[-1]

            # Hidden state update
            h = np.tanh(
                np.dot(x, self.W_xh) +
                np.dot(h_prev, self.W_hh) +
                self.b_h
            )

            # Output
            y = np.dot(h, self.W_hy) + self.b_y

            self.hidden_states.append(h)
            outputs.append(y)

        self.outputs = outputs
        return outputs

    def backward(self, targets, learning_rate=0.01):
        """
        Backpropagation Through Time!

        This is where the magic (and the pain) happens.
        """
        T = len(self.inputs)

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # Gradient that flows back through hidden states
        dh_next = np.zeros(self.hidden_size)

        total_loss = 0

        # Go BACKWARDS through time!
        for t in reversed(range(T)):
            x = self.inputs[t]
            h = self.hidden_states[t + 1]
            h_prev = self.hidden_states[t]
            y = self.outputs[t]
            target = targets[t]

            # Output layer gradient (MSE loss)
            dy = y - target
            total_loss += np.sum(dy ** 2)

            # Gradient to output weights
            dW_hy += np.outer(h, dy)
            db_y += dy

            # Gradient flowing into h from two sources:
            # 1. From output layer
            # 2. From next timestep (dh_next)
            dh = np.dot(dy, self.W_hy.T) + dh_next

            # Backprop through tanh
            dh_raw = dh * (1 - h ** 2)  # tanh derivative

            # Gradient to weights
            dW_xh += np.outer(x, dh_raw)
            dW_hh += np.outer(h_prev, dh_raw)
            db_h += dh_raw

            # Gradient to pass to previous timestep
            dh_next = np.dot(dh_raw, self.W_hh.T)

        # Clip gradients to prevent explosion
        for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(grad, -5, 5, out=grad)

        # Update weights
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

        return total_loss / T


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Backpropagation Through Time (BPTT)")
    print("=" * 60)

    # Simple sequence learning task
    # Learn to output a delayed version of input

    print("\n--- Learning a Simple Sequence Pattern ---")
    print("Task: Given input sequence, output same sequence delayed by 1")

    # Create training data
    # Input:  [1, 0, 1, 0, 1, ...]
    # Output: [0, 1, 0, 1, 0, ...]  (shifted by 1)

    sequence_length = 10

    inputs = []
    targets = []

    for i in range(sequence_length):
        inp = np.array([1.0 if i % 2 == 0 else 0.0])
        target = np.array([0.0 if i % 2 == 0 else 1.0])
        inputs.append(inp)
        targets.append(target)

    print(f"\nInput sequence:  {[int(x[0]) for x in inputs]}")
    print(f"Target sequence: {[int(t[0]) for t in targets]}")

    # Create and train RNN
    rnn = RNNWithBPTT(input_size=1, hidden_size=5, output_size=1)

    print("\n--- Training ---")
    epochs = 200

    for epoch in range(epochs):
        outputs = rnn.forward(inputs)
        loss = rnn.backward(targets, learning_rate=0.1)

        if epoch % 40 == 0:
            preds = [round(o[0], 2) for o in outputs]
            print(f"Epoch {epoch:3d}: loss={loss:.4f}, preds={preds[:5]}...")

    # Test
    print("\n--- After Training ---")
    outputs = rnn.forward(inputs)

    print(f"Input:      {[int(x[0]) for x in inputs]}")
    print(f"Target:     {[int(t[0]) for t in targets]}")
    print(f"Predicted:  {[round(o[0], 1) for o in outputs]}")

    # Visualize gradient flow
    print("\n" + "=" * 60)
    print("Gradient Flow Visualization")
    print("=" * 60)

    print("""
    Time: t=0 ─────> t=1 ─────> t=2 ─────> t=3
           │         │         │         │
    Forward│         │         │         │
           ▼         ▼         ▼         ▼
          h0 ──────> h1 ──────> h2 ──────> h3
           │         │         │         │
           ▼         ▼         ▼         ▼
          y0        y1        y2        y3
           │         │         │         │
    Loss:  L0        L1        L2        L3
           │         │         │         │
    Back-  │         │         │         │
    ward   ◄─────────◄─────────◄─────────◄─
           │         │         │         │
    dh0 ◄──┴── dh1 ◄─┴── dh2 ◄─┴── dh3 ◄─┘

    Gradients flow BACKWARDS through the hidden states!
    Each dh_t depends on dh_{t+1} - this is the "Through Time" part.
    """)

    # Demonstrate vanishing gradient
    print("\n" + "=" * 60)
    print("The Vanishing Gradient Problem")
    print("=" * 60)

    print("\nAt each timestep, gradient is multiplied by W_hh.")
    print("If |W_hh| < 1, gradients shrink exponentially:")

    W = 0.5  # typical weight magnitude
    steps = [1, 5, 10, 20, 50]

    print(f"\nGradient magnitude after n steps (|W|={W}):")
    for n in steps:
        grad = W ** n
        print(f"  After {n:2d} steps: {grad:.2e}")

    print("\nAfter 50 steps, gradient is basically ZERO!")
    print("This is why vanilla RNNs can't learn long-range dependencies.")
    print("\nSolution: LSTM and GRU use 'gates' to control gradient flow!")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. BPTT unrolls the RNN through time")
    print("2. Gradients flow backwards through all timesteps")
    print("3. Same weights = sum up gradients from all steps")
    print("4. Gradients multiply at each step (W_hh)")
    print("5. Vanishing gradients: can't learn long sequences")
    print("6. Gradient clipping prevents explosion")
    print("7. LSTMs solve vanishing gradients with gating")
