# embedding layer in neural network

import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # random initialization
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, indices):
        """
        indices: word indices (can be single int or array)
        returns: embeddings
        """
        self.indices = indices
        return self.W[indices]

    def backward(self, d_out):
        """
        d_out: gradient from next layer
        updates embedding weights
        """
        d_W = np.zeros_like(self.W)
        np.add.at(d_W, self.indices, d_out)
        return d_W

# test
np.random.seed(42)
embed = EmbeddingLayer(vocab_size=100, embedding_dim=16)

# single word
word_idx = 42
embedding = embed.forward(word_idx)
print(f"single word embedding shape: {embedding.shape}")

# batch of words
batch_indices = np.array([1, 5, 42, 7, 3])
batch_embeddings = embed.forward(batch_indices)
print(f"batch embedding shape: {batch_embeddings.shape}")

# sequence
sequence = np.array([1, 2, 3, 4, 5])
seq_embeddings = embed.forward(sequence)
print(f"sequence embedding shape: {seq_embeddings.shape}")
