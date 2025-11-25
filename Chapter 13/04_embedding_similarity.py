# measuring similarity between embeddings

import numpy as np

def cosine_similarity(a, b):
    """similarity between -1 and 1"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# example embeddings
embeddings = {
    "cat":    np.array([0.8, 0.2, -0.1, 0.5]),
    "dog":    np.array([0.7, 0.3, -0.2, 0.4]),
    "kitten": np.array([0.75, 0.25, -0.05, 0.55]),
    "car":    np.array([-0.5, 0.8, 0.3, -0.2]),
}

print("COSINE SIMILARITY")
print()

pairs = [
    ("cat", "dog"),
    ("cat", "kitten"),
    ("cat", "car"),
    ("dog", "car"),
]

for w1, w2 in pairs:
    sim = cosine_similarity(embeddings[w1], embeddings[w2])
    print(f"  {w1} <-> {w2}: {sim:.3f}")

print()
print("similar concepts = high similarity")
print("different concepts = low similarity")
