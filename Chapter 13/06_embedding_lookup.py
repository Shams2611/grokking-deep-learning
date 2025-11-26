# embedding lookup - one-hot times matrix

import numpy as np

vocab = ["the", "cat", "sat", "on", "mat"]
vocab_size = len(vocab)
embedding_dim = 4

np.random.seed(42)
E = np.random.randn(vocab_size, embedding_dim) * 0.1

def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

print("two ways to get embedding:")
print()

word_idx = 1  # "cat"

# method 1: direct indexing
method1 = E[word_idx]
print(f"direct index E[{word_idx}]:")
print(f"  {np.round(method1, 3)}")

# method 2: one-hot times matrix
one_hot_vec = one_hot(word_idx, vocab_size)
method2 = one_hot_vec @ E
print(f"\none-hot @ E:")
print(f"  one-hot: {one_hot_vec.astype(int)}")
print(f"  result: {np.round(method2, 3)}")

print(f"\nsame result: {np.allclose(method1, method2)}")
print("\nembedding = learned lookup table!")
