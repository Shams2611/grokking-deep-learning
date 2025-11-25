# word embeddings - dense, meaningful vectors

import numpy as np

print("WORD EMBEDDINGS")
print()
print("instead of sparse one-hot:")
print("  'cat' -> [1, 0, 0, 0, 0, ...]  (vocab_size dims)")
print()
print("use dense learned vectors:")
print("  'cat' -> [0.2, -0.5, 0.8, 0.1]  (embedding_size dims)")
print()
print("benefits:")
print("  - much smaller (e.g., 300 vs 50000)")
print("  - captures meaning")
print("  - similar words = similar vectors")
print()

# example embeddings (pretend these are learned)
embeddings = {
    "cat":    np.array([0.8, 0.2, -0.1, 0.5]),
    "dog":    np.array([0.7, 0.3, -0.2, 0.4]),  # similar to cat
    "kitten": np.array([0.75, 0.25, -0.05, 0.6]),  # very similar to cat
    "car":    np.array([-0.5, 0.8, 0.3, -0.2]),  # different
}

print("example embeddings:")
for word, vec in embeddings.items():
    print(f"  {word}: {vec}")
