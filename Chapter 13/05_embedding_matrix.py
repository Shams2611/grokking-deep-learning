# embedding matrix - lookup table

import numpy as np

# vocabulary
vocab = ["the", "cat", "sat", "on", "mat"]
vocab_size = len(vocab)
embedding_dim = 4

# embedding matrix: (vocab_size, embedding_dim)
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.1

print("embedding matrix shape:", embedding_matrix.shape)
print()
print("each row is one word's embedding:")
for i, word in enumerate(vocab):
    print(f"  {word}: {np.round(embedding_matrix[i], 3)}")

print()
print("to get embedding: index into matrix")

word_idx = 1  # "cat"
cat_embedding = embedding_matrix[word_idx]
print(f"\nembedding['cat'] = matrix[1] = {np.round(cat_embedding, 3)}")
