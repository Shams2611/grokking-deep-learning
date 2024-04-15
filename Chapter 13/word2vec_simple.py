"""
Chapter 13: Word2Vec - Learning Word Embeddings

Word embeddings turn words into meaningful vectors!

The problem with one-hot encoding:
- "cat" = [1, 0, 0, 0, ...]
- "dog" = [0, 1, 0, 0, ...]
- These vectors are ORTHOGONAL - no similarity info!

Word2Vec learns dense embeddings where:
- Similar words have similar vectors
- king - man + woman ≈ queen (vector arithmetic!)

Two architectures:
1. Skip-gram: predict context words from center word
2. CBOW: predict center word from context words

Skip-gram is more popular for learning good embeddings.

The key insight:
"You shall know a word by the company it keeps"
- Words that appear in similar contexts have similar meanings
- Train a network to predict context words
- The hidden layer becomes the word embedding!
"""

import numpy as np
from collections import Counter

np.random.seed(42)


class Word2VecSkipGram:
    """
    Simple Skip-gram Word2Vec implementation.

    Architecture:
    - Input: one-hot word vector
    - Hidden: embedding layer (no activation)
    - Output: softmax over vocabulary

    The hidden layer weights become the word embeddings!
    """

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # W1: embedding matrix (what we want to learn!)
        # Shape: (vocab_size, embedding_dim)
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01

        # W2: output weights
        # Shape: (embedding_dim, vocab_size)
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    def forward(self, center_idx):
        """
        Forward pass.

        center_idx: index of the center word
        """
        # Get embedding for center word (just lookup, no matrix multiply needed!)
        self.hidden = self.W1[center_idx]

        # Output layer
        self.output = np.dot(self.hidden, self.W2)

        # Softmax
        exp_out = np.exp(self.output - np.max(self.output))
        self.probs = exp_out / np.sum(exp_out)

        return self.probs

    def backward(self, center_idx, context_idx, learning_rate=0.01):
        """
        Backward pass and weight update.

        center_idx: index of center word
        context_idx: index of context word (target)
        """
        # Gradient of softmax cross-entropy
        d_output = self.probs.copy()
        d_output[context_idx] -= 1  # subtract 1 at correct class

        # Gradient for W2
        d_W2 = np.outer(self.hidden, d_output)

        # Gradient for hidden layer
        d_hidden = np.dot(d_output, self.W2.T)

        # Gradient for W1 (only update the row for center_idx)
        d_W1 = d_hidden

        # Update weights
        self.W2 -= learning_rate * d_W2
        self.W1[center_idx] -= learning_rate * d_W1

        # Return loss
        loss = -np.log(self.probs[context_idx] + 1e-10)
        return loss

    def get_embedding(self, word_idx):
        """Get the embedding vector for a word."""
        return self.W1[word_idx]

    def most_similar(self, word_idx, idx_to_word, top_n=5):
        """Find most similar words using cosine similarity."""
        word_vec = self.W1[word_idx]

        # Compute cosine similarity with all words
        similarities = []
        for i in range(self.vocab_size):
            if i == word_idx:
                continue
            other_vec = self.W1[i]

            # Cosine similarity
            cos_sim = np.dot(word_vec, other_vec) / (
                np.linalg.norm(word_vec) * np.linalg.norm(other_vec) + 1e-10
            )
            similarities.append((i, cos_sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [(idx_to_word[i], sim) for i, sim in similarities[:top_n]]


def create_training_data(corpus, word_to_idx, window_size=2):
    """
    Create (center, context) pairs from corpus.

    For each word, create pairs with words within window_size.
    """
    pairs = []

    for sentence in corpus:
        indices = [word_to_idx[w] for w in sentence if w in word_to_idx]

        for i, center in enumerate(indices):
            # Get context words within window
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)

            for j in range(start, end):
                if i != j:  # don't pair word with itself
                    pairs.append((center, indices[j]))

    return pairs


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Word2Vec Skip-gram - Word Embeddings from Scratch")
    print("=" * 60)

    # Simple corpus
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "in", "the", "park"],
        ["a", "cat", "and", "dog", "are", "friends"],
        ["the", "cat", "ran", "fast"],
        ["the", "dog", "sat", "down"],
        ["cat", "and", "dog", "play", "in", "park"],
        ["mat", "is", "on", "the", "floor"],
        ["the", "park", "is", "nice"],
    ]

    # Build vocabulary
    all_words = [word for sentence in corpus for word in sentence]
    word_counts = Counter(all_words)

    vocab = list(word_counts.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    vocab_size = len(vocab)
    embedding_dim = 8

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Vocabulary: {vocab}")

    # Create training pairs
    training_pairs = create_training_data(corpus, word_to_idx, window_size=2)
    print(f"\nNumber of training pairs: {len(training_pairs)}")

    # Show some pairs
    print("\nSample (center, context) pairs:")
    for center, context in training_pairs[:5]:
        print(f"  {idx_to_word[center]} -> {idx_to_word[context]}")

    # Create and train model
    print("\n--- Training Word2Vec ---")

    model = Word2VecSkipGram(vocab_size, embedding_dim)

    epochs = 100
    lr = 0.1

    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(training_pairs)

        for center, context in training_pairs:
            model.forward(center)
            loss = model.backward(center, context, learning_rate=lr)
            total_loss += loss

        if epoch % 20 == 0:
            avg_loss = total_loss / len(training_pairs)
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")

    # Look at learned embeddings
    print("\n--- Learned Word Embeddings ---")

    print("\nWord embeddings (first 4 dimensions):")
    for word in ["cat", "dog", "the", "park"]:
        idx = word_to_idx[word]
        emb = model.get_embedding(idx)
        print(f"  {word}: {emb[:4].round(3)}...")

    # Find similar words
    print("\n--- Most Similar Words ---")

    for word in ["cat", "dog", "the"]:
        idx = word_to_idx[word]
        similar = model.most_similar(idx, idx_to_word, top_n=3)
        print(f"\nMost similar to '{word}':")
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.3f}")

    # Explain the magic
    print("\n" + "=" * 60)
    print("How Word2Vec Works")
    print("=" * 60)

    print("""
    SKIP-GRAM ARCHITECTURE:
    ┌────────────────────────────────────────────────────────┐
    │                                                        │
    │   Input        Hidden         Output                   │
    │  (one-hot)   (embedding)     (softmax)                │
    │                                                        │
    │    cat    ──>  [0.2, -0.1,   ──>  P(the)              │
    │  [0,1,0,...]     0.5, ...]        P(sat)              │
    │                    ↑              P(on)               │
    │                    │              ...                  │
    │                    │                                   │
    │            This IS the                                 │
    │            word embedding!                             │
    │                                                        │
    └────────────────────────────────────────────────────────┘

    THE TRAINING OBJECTIVE:
    Given "the cat sat on the mat"

    For center word "sat":
        Predict: "the", "cat", "on", "the"

    This forces similar words (that appear in similar contexts)
    to have similar embeddings!

    WHY IT WORKS:
    - "cat" and "dog" both appear near "the", "sat", "ran", etc.
    - To predict these context words, they need similar embeddings
    - The network learns semantic relationships automatically!
    """)

    # Vector arithmetic demo
    print("\n" + "=" * 60)
    print("Word Vector Arithmetic")
    print("=" * 60)

    print("""
    Famous example (needs more data to work well):

    king - man + woman ≈ queen

    This works because the embeddings capture semantic relationships:
    - (king - man) captures "royalty" direction
    - Adding "woman" gives "female royalty" = queen

    Our tiny corpus can't learn this, but here's a simple example:
    """)

    if "cat" in word_to_idx and "dog" in word_to_idx and "park" in word_to_idx:
        cat_vec = model.get_embedding(word_to_idx["cat"])
        dog_vec = model.get_embedding(word_to_idx["dog"])

        print(f"\ncat vector: {cat_vec[:4].round(3)}...")
        print(f"dog vector: {dog_vec[:4].round(3)}...")

        diff = dog_vec - cat_vec
        print(f"dog - cat: {diff[:4].round(3)}...")
        print("\nThis difference vector captures 'something about dogs vs cats'")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Word2Vec learns dense, meaningful word vectors")
    print("2. Skip-gram: predict context from center word")
    print("3. Similar words get similar embeddings")
    print("4. The hidden layer weights ARE the embeddings")
    print("5. Real Word2Vec uses negative sampling for efficiency")
    print("6. Pre-trained embeddings (GloVe, FastText) are commonly used")
