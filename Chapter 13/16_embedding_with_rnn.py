# using embeddings with RNN

import numpy as np

def tanh(x): return np.tanh(x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

np.random.seed(42)

# vocabulary
vocab = ["<pad>", "i", "love", "deep", "learning", "hate", "boring", "stuff"]
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# dimensions
embedding_dim = 8
hidden_size = 16
num_classes = 2  # positive/negative

# weights
E = np.random.randn(vocab_size, embedding_dim) * 0.1  # embeddings
W_xh = np.random.randn(embedding_dim, hidden_size) * 0.1
W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
W_hy = np.random.randn(hidden_size, num_classes) * 0.1

def classify_sentence(sentence):
    words = sentence.lower().split()
    indices = [word_to_idx.get(w, 0) for w in words]

    # embed words
    embeddings = E[indices]

    # RNN
    h = np.zeros(hidden_size)
    for emb in embeddings:
        h = tanh(emb @ W_xh + h @ W_hh)

    # classify
    logits = h @ W_hy
    probs = softmax(logits)
    return probs

# test
sentences = ["i love deep learning", "i hate boring stuff"]
for sent in sentences:
    probs = classify_sentence(sent)
    sentiment = "positive" if probs[1] > 0.5 else "negative"
    print(f"'{sent}'")
    print(f"  probs: {np.round(probs, 3)}")
