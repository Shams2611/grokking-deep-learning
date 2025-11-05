# many-to-one RNN (e.g., sentiment analysis)

import numpy as np

def tanh(x): return np.tanh(x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

np.random.seed(42)

# dimensions
vocab_size = 10
embed_size = 4
hidden_size = 8
num_classes = 2  # positive/negative

# weights
W_embed = np.random.randn(vocab_size, embed_size) * 0.5
W_xh = np.random.randn(embed_size, hidden_size) * 0.5
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5
W_hy = np.random.randn(hidden_size, num_classes) * 0.5

# input: sequence of word indices
sentence = [3, 7, 2, 5]  # "I love this movie"

print("many-to-one RNN:")
print(f"input sequence: {sentence}")
print()

# forward pass
h = np.zeros(hidden_size)
for word_idx in sentence:
    x = W_embed[word_idx]  # get embedding
    h = tanh(x @ W_xh + h @ W_hh)

# only use final hidden state for prediction
logits = h @ W_hy
probs = softmax(logits)

print(f"final hidden state shape: {h.shape}")
print(f"output probabilities: {np.round(probs, 3)}")
print(f"prediction: {'positive' if probs[1] > 0.5 else 'negative'}")
