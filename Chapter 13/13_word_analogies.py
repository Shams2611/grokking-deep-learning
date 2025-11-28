# word analogies - the magic of embeddings

import numpy as np

print("WORD ANALOGIES")
print()
print("famous example:")
print("  king - man + woman = queen")
print()
print("embeddings capture relationships!")
print()

# simulated embeddings (in reality, trained on huge corpus)
embeddings = {
    "king":   np.array([0.8, 0.6, 0.3, -0.2]),
    "queen":  np.array([0.75, 0.65, -0.3, 0.2]),
    "man":    np.array([0.5, 0.4, 0.5, -0.4]),
    "woman":  np.array([0.45, 0.45, -0.1, 0.0]),
}

def analogy(a, b, c, embeddings):
    """a is to b as c is to ?"""
    # result = b - a + c
    result = embeddings[b] - embeddings[a] + embeddings[c]
    return result

def most_similar(vec, embeddings, exclude=[]):
    best_word = None
    best_sim = -float('inf')
    for word, emb in embeddings.items():
        if word in exclude:
            continue
        sim = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb))
        if sim > best_sim:
            best_sim = sim
            best_word = word
    return best_word, best_sim

# king - man + woman = ?
result_vec = analogy("man", "king", "woman", embeddings)
answer, sim = most_similar(result_vec, embeddings, exclude=["man", "king", "woman"])

print("king - man + woman = ?")
print(f"answer: {answer} (similarity: {sim:.3f})")
