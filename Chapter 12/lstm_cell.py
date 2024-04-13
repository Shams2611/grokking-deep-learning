"""
Chapter 12: LSTM Cell - Long Short-Term Memory

LSTMs are the solution to the vanishing gradient problem!

The key innovation: a "cell state" that flows through time
with minimal modification, like a conveyor belt.

LSTM has THREE gates that control information flow:
1. Forget gate (f): what to THROW AWAY from memory
2. Input gate (i): what NEW info to STORE in memory
3. Output gate (o): what to OUTPUT from memory

Plus a "candidate" value (g) for potential new memories.

The equations:
    f = sigmoid(W_f * [h_prev, x] + b_f)     # forget gate
    i = sigmoid(W_i * [h_prev, x] + b_i)     # input gate
    g = tanh(W_g * [h_prev, x] + b_g)        # candidate
    o = sigmoid(W_o * [h_prev, x] + b_o)     # output gate

    c = f * c_prev + i * g   # new cell state
    h = o * tanh(c)          # new hidden state

Why does this solve vanishing gradients?
- The cell state c can flow unchanged (when f=1, i=0)
- Gradients can backprop through the cell state easily
- Gates learn WHEN to remember and WHEN to forget
"""

import numpy as np

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    return np.tanh(x)


class LSTMCell:
    """A single LSTM cell."""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights for all gates
        # Combined input: [x, h_prev] has size (input_size + hidden_size)
        combined_size = input_size + hidden_size
        scale = 1.0 / np.sqrt(combined_size)

        # Forget gate
        self.W_f = np.random.randn(combined_size, hidden_size) * scale
        self.b_f = np.ones(hidden_size)  # start with bias=1 (remember by default!)

        # Input gate
        self.W_i = np.random.randn(combined_size, hidden_size) * scale
        self.b_i = np.zeros(hidden_size)

        # Candidate (new memory)
        self.W_g = np.random.randn(combined_size, hidden_size) * scale
        self.b_g = np.zeros(hidden_size)

        # Output gate
        self.W_o = np.random.randn(combined_size, hidden_size) * scale
        self.b_o = np.zeros(hidden_size)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through LSTM cell.

        Args:
            x: current input (input_size,)
            h_prev: previous hidden state (hidden_size,)
            c_prev: previous cell state (hidden_size,)

        Returns:
            h: new hidden state
            c: new cell state
            cache: stuff we need for backprop
        """
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev])

        # Forget gate: what to forget from cell state
        f = sigmoid(np.dot(combined, self.W_f) + self.b_f)

        # Input gate: what new info to add
        i = sigmoid(np.dot(combined, self.W_i) + self.b_i)

        # Candidate: potential new memories
        g = tanh(np.dot(combined, self.W_g) + self.b_g)

        # Output gate: what to output
        o = sigmoid(np.dot(combined, self.W_o) + self.b_o)

        # New cell state: forget old + add new
        c = f * c_prev + i * g

        # New hidden state
        h = o * tanh(c)

        # Store for backprop
        cache = (x, h_prev, c_prev, combined, f, i, g, o, c)

        return h, c, cache


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("LSTM Cell - Long Short-Term Memory")
    print("=" * 60)

    # Create LSTM cell
    input_size = 4
    hidden_size = 8

    lstm = LSTMCell(input_size, hidden_size)

    print(f"\nLSTM Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")

    # Initial states
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)

    print(f"\nInitial hidden state: {h[:4]}...")
    print(f"Initial cell state: {c[:4]}...")

    # Process a sequence
    sequence = [
        np.array([1, 0, 0, 0]),  # "event A"
        np.array([0, 1, 0, 0]),  # "event B"
        np.array([0, 0, 1, 0]),  # "event C"
        np.array([0, 0, 0, 1]),  # "event D"
    ]

    print("\n--- Processing Sequence ---")

    for t, x in enumerate(sequence):
        h, c, cache = lstm.forward(x, h, c)
        _, _, _, _, f, i, g, o, _ = cache

        print(f"\nTimestep {t}: input = {x}")
        print(f"  Forget gate (avg): {f.mean():.3f}")
        print(f"  Input gate (avg):  {i.mean():.3f}")
        print(f"  Output gate (avg): {o.mean():.3f}")
        print(f"  Cell state norm:   {np.linalg.norm(c):.3f}")
        print(f"  Hidden state:      {h[:3].round(3)}...")

    # Explain the gates
    print("\n" + "=" * 60)
    print("Understanding the Gates")
    print("=" * 60)

    print("""
    FORGET GATE (f):
    ┌─────────────────────────────────────────────┐
    │  f = sigmoid(W_f * [x, h] + b_f)            │
    │                                             │
    │  f ≈ 1: Keep old memories                   │
    │  f ≈ 0: Forget old memories                 │
    │                                             │
    │  Example: At end of sentence, forget the    │
    │  subject to make room for the next one.     │
    └─────────────────────────────────────────────┘

    INPUT GATE (i) + CANDIDATE (g):
    ┌─────────────────────────────────────────────┐
    │  i = sigmoid(W_i * [x, h] + b_i)            │
    │  g = tanh(W_g * [x, h] + b_g)               │
    │                                             │
    │  i ≈ 1: Add new memory (i * g)              │
    │  i ≈ 0: Don't update memory                 │
    │                                             │
    │  Example: When seeing important info,       │
    │  open the gate to store it.                 │
    └─────────────────────────────────────────────┘

    OUTPUT GATE (o):
    ┌─────────────────────────────────────────────┐
    │  o = sigmoid(W_o * [x, h] + b_o)            │
    │  h = o * tanh(c)                            │
    │                                             │
    │  o ≈ 1: Use memory for output               │
    │  o ≈ 0: Keep memory hidden                  │
    │                                             │
    │  Example: Only output when relevant info    │
    │  is needed for the current task.            │
    └─────────────────────────────────────────────┘

    CELL STATE UPDATE:
    ┌─────────────────────────────────────────────┐
    │  c_new = f * c_old + i * g                  │
    │           ↑           ↑                     │
    │     what to keep  what to add               │
    │                                             │
    │  This is the "conveyor belt" that lets      │
    │  gradients flow easily through time!        │
    └─────────────────────────────────────────────┘
    """)

    print("\n" + "=" * 60)
    print("Why LSTM Works (Gradient Flow)")
    print("=" * 60)

    print("""
    In vanilla RNN:
        h_t = tanh(W * h_{t-1} + ...)
        Gradient = W * W * W * ... (multiplies at each step)
        → Vanishes or explodes!

    In LSTM:
        c_t = f * c_{t-1} + i * g
        Gradient = f * f * f * ... (can stay close to 1!)
        → Gradients flow easily through time

    The forget gate f is usually close to 1 for important memories,
    so gradients don't vanish as badly.
    """)

    print("=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. LSTM has 4 components: forget, input, candidate, output")
    print("2. Cell state is the 'long-term memory'")
    print("3. Hidden state is the 'working memory' / output")
    print("4. Gates control what to remember/forget/output")
    print("5. Cell state allows gradients to flow over long sequences")
    print("6. Forget gate bias often initialized to 1 (remember by default)")
