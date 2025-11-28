# simple Word2Vec implementation

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# tiny corpus
corpus = "the cat sat on the mat the dog sat on the rug"
words = corpus.split()
vocab = list(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"vocabulary: {vocab}")
print(f"vocab size: {vocab_size}")

# embeddings
embedding_dim = 4
np.random.seed(42)
W_in = np.random.randn(vocab_size, embedding_dim) * 0.1   # input embeddings
W_out = np.random.randn(vocab_size, embedding_dim) * 0.1  # output embeddings

# create training pairs
def get_pairs(words, window=1):
    pairs = []
    for i, center in enumerate(words):
        for j in range(max(0, i-window), min(len(words), i+window+1)):
            if i != j:
                pairs.append((word_to_idx[center], word_to_idx[words[j]]))
    return pairs

pairs = get_pairs(words, window=1)
print(f"\ntraining pairs: {len(pairs)}")

# train with negative sampling
lr = 0.1
n_neg = 2

for epoch in range(100):
    loss = 0
    for center_idx, context_idx in pairs:
        # positive example
        center_vec = W_in[center_idx]
        context_vec = W_out[context_idx]

        score = sigmoid(np.dot(center_vec, context_vec))
        loss += -np.log(score + 1e-10)

        # gradient
        grad = (score - 1) * context_vec
        W_in[center_idx] -= lr * grad
        W_out[context_idx] -= lr * (score - 1) * center_vec

        # negative examples
        for _ in range(n_neg):
            neg_idx = np.random.randint(vocab_size)
            neg_vec = W_out[neg_idx]
            score_neg = sigmoid(np.dot(center_vec, neg_vec))
            loss += -np.log(1 - score_neg + 1e-10)

            W_in[center_idx] -= lr * score_neg * neg_vec
            W_out[neg_idx] -= lr * score_neg * center_vec

    if epoch % 20 == 0:
        print(f"epoch {epoch}: loss = {loss:.2f}")
