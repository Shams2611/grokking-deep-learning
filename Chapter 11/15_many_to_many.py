# many-to-many RNN (e.g., language model)

import numpy as np

def tanh(x): return np.tanh(x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

np.random.seed(42)

vocab_size = 10
hidden_size = 8

W_xh = np.random.randn(vocab_size, hidden_size) * 0.5
W_hh = np.random.randn(hidden_size, hidden_size) * 0.5
W_hy = np.random.randn(hidden_size, vocab_size) * 0.5

def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

# input sequence
sequence = [1, 4, 2, 7]

print("many-to-many RNN (language model):")
print(f"input: {sequence}")
print()

h = np.zeros(hidden_size)
predictions = []

for word_idx in sequence:
    x = one_hot(word_idx, vocab_size)
    h = tanh(x @ W_xh + h @ W_hh)
    logits = h @ W_hy
    probs = softmax(logits)
    pred = np.argmax(probs)
    predictions.append(pred)
    print(f"input {word_idx} -> predict next: {pred}")

print()
print(f"predictions: {predictions}")
