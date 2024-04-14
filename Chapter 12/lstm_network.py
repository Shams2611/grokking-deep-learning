"""
Chapter 12: LSTM Network

Putting LSTM cells together to form a full network!

Architecture:
    input -> LSTM cells (over time) -> Dense layer -> output

Each LSTM cell passes its hidden state and cell state
to the next timestep.

This file shows:
1. Processing sequences with LSTM
2. Many-to-one: sequence input, single output (e.g., sentiment)
3. Many-to-many: sequence input, sequence output (e.g., translation)
"""

import numpy as np

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class LSTM:
    """Full LSTM layer."""

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        combined = input_size + hidden_size
        scale = 1.0 / np.sqrt(combined)

        # All gate weights
        self.W_f = np.random.randn(combined, hidden_size) * scale
        self.W_i = np.random.randn(combined, hidden_size) * scale
        self.W_g = np.random.randn(combined, hidden_size) * scale
        self.W_o = np.random.randn(combined, hidden_size) * scale

        self.b_f = np.ones(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_g = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)

    def forward_step(self, x, h_prev, c_prev):
        """One LSTM step."""
        combined = np.concatenate([x, h_prev])

        f = sigmoid(np.dot(combined, self.W_f) + self.b_f)
        i = sigmoid(np.dot(combined, self.W_i) + self.b_i)
        g = np.tanh(np.dot(combined, self.W_g) + self.b_g)
        o = sigmoid(np.dot(combined, self.W_o) + self.b_o)

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        return h, c

    def forward(self, sequence):
        """
        Forward pass through entire sequence.

        Returns all hidden states.
        """
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        hidden_states = []

        for x in sequence:
            h, c = self.forward_step(x, h, c)
            hidden_states.append(h)

        return hidden_states, h, c


class LSTMClassifier:
    """LSTM for sequence classification (many-to-one)."""

    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = LSTM(input_size, hidden_size)

        # Dense layer for classification
        self.W_out = np.random.randn(hidden_size, output_size) * 0.1
        self.b_out = np.zeros(output_size)

    def forward(self, sequence):
        """
        Process sequence, output single prediction.

        Uses final hidden state for classification.
        """
        _, final_h, _ = self.lstm.forward(sequence)

        # Dense layer
        output = np.dot(final_h, self.W_out) + self.b_out

        # Softmax for classification
        exp_out = np.exp(output - np.max(output))
        probs = exp_out / np.sum(exp_out)

        return probs, final_h


class LSTMSequenceModel:
    """LSTM for sequence-to-sequence (many-to-many)."""

    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = LSTM(input_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size) * 0.1
        self.b_out = np.zeros(output_size)

    def forward(self, sequence):
        """
        Process sequence, output at each timestep.
        """
        hidden_states, _, _ = self.lstm.forward(sequence)

        outputs = []
        for h in hidden_states:
            out = np.dot(h, self.W_out) + self.b_out
            outputs.append(out)

        return outputs, hidden_states


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("LSTM Network")
    print("=" * 60)

    # Demo 1: Sentiment Classification (Many-to-One)
    print("\n" + "=" * 60)
    print("Demo 1: Sentiment Classification (Many-to-One)")
    print("=" * 60)

    # Fake word embeddings (normally you'd use real ones)
    vocab = {"great": 0, "terrible": 1, "movie": 2, "bad": 3, "good": 4}
    embedding_dim = 4
    embeddings = np.random.randn(len(vocab), embedding_dim)

    def words_to_embeddings(words):
        return [embeddings[vocab[w]] for w in words]

    # Create classifier
    classifier = LSTMClassifier(
        input_size=embedding_dim,
        hidden_size=8,
        output_size=2  # positive / negative
    )

    # Test sentences
    sentences = [
        ["great", "movie"],
        ["terrible", "movie"],
        ["bad", "bad", "movie"],
        ["good", "good", "great"],
    ]

    print("\nClassifying sentences (random weights, just for demo):")
    for sentence in sentences:
        embeddings_seq = words_to_embeddings(sentence)
        probs, _ = classifier.forward(embeddings_seq)
        sentiment = "positive" if probs[1] > probs[0] else "negative"
        print(f"  '{' '.join(sentence)}' -> {sentiment} ({probs[1]:.2%} positive)")

    # Demo 2: Sequence Prediction (Many-to-Many)
    print("\n" + "=" * 60)
    print("Demo 2: Sequence Prediction (Many-to-Many)")
    print("=" * 60)

    # Predict next value in a sine wave
    seq_model = LSTMSequenceModel(
        input_size=1,
        hidden_size=10,
        output_size=1
    )

    # Generate sine wave data
    t = np.linspace(0, 4 * np.pi, 20)
    sine_wave = np.sin(t)

    # Format as sequence
    sequence = [np.array([x]) for x in sine_wave[:-1]]

    outputs, hidden_states = seq_model.forward(sequence)

    print("\nPredicting sine wave (random weights, just for demo):")
    print("\nTimestep | Input    | Hidden norm | Prediction")
    print("-" * 55)

    for i in range(min(10, len(sequence))):
        inp = sequence[i][0]
        h_norm = np.linalg.norm(hidden_states[i])
        pred = outputs[i][0]
        print(f"    {i:2d}   | {inp:7.3f}  |   {h_norm:.3f}     |  {pred:.3f}")

    if len(sequence) > 10:
        print("    ...")

    # Show architecture
    print("\n" + "=" * 60)
    print("LSTM Architectures")
    print("=" * 60)

    print("""
    MANY-TO-ONE (Classification):
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  x1 ──> LSTM ──> x2 ──> LSTM ──> x3 ──> LSTM ──> h3    │
    │                                                   │     │
    │                                                   ▼     │
    │                                              [Dense]    │
    │                                                   │     │
    │                                                   ▼     │
    │                                             [Output]    │
    │                                                         │
    │  Use final hidden state for classification              │
    │  Example: Sentiment analysis                            │
    └─────────────────────────────────────────────────────────┘

    MANY-TO-MANY (Sequence-to-Sequence):
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  x1 ──> LSTM ──> x2 ──> LSTM ──> x3 ──> LSTM           │
    │           │              │              │               │
    │           ▼              ▼              ▼               │
    │        [Dense]        [Dense]        [Dense]            │
    │           │              │              │               │
    │           ▼              ▼              ▼               │
    │          y1             y2             y3               │
    │                                                         │
    │  Output at every timestep                               │
    │  Example: Named entity recognition, translation         │
    └─────────────────────────────────────────────────────────┘

    ONE-TO-MANY (Generation):
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  x ──> LSTM ──> LSTM ──> LSTM ──> ...                  │
    │          │        │        │                            │
    │          ▼        ▼        ▼                            │
    │         y1       y2       y3                            │
    │                                                         │
    │  One input, generate sequence output                    │
    │  Example: Image captioning, text generation             │
    └─────────────────────────────────────────────────────────┘
    """)

    print("=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. LSTMs can handle variable-length sequences")
    print("2. Many-to-one: use final hidden state")
    print("3. Many-to-many: output at each timestep")
    print("4. Hidden state carries information through time")
    print("5. Can stack multiple LSTM layers for more capacity")
    print("6. Bidirectional LSTMs process forward AND backward")
