# hidden state = memory of the network

import numpy as np

print("RNN HIDDEN STATE")
print()
print("hidden state carries information through time")
print()

# simple example
words = ["I", "love", "deep", "learning"]
hidden_size = 3

# hidden state starts at zero
hidden = np.zeros(hidden_size)
print(f"initial hidden state: {hidden}")
print()

# process each word (simplified)
for word in words:
    # in reality: hidden = f(word_embedding, hidden)
    hidden = hidden + 0.1  # fake update
    print(f"after '{word}': hidden = {np.round(hidden, 2)}")

print()
print("hidden state accumulates information!")
