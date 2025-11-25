# one-hot encoding implementation

import numpy as np

# vocabulary
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)

def one_hot(word):
    vec = np.zeros(vocab_size)
    vec[word_to_idx[word]] = 1
    return vec

print("vocabulary:", vocab)
print(f"vocab size: {vocab_size}")
print()

# encode some words
for word in ["cat", "dog", "sat"]:
    print(f"'{word}': {one_hot(word).astype(int)}")

print()
print("each word is a sparse vector")
print("no relationship between vectors!")
