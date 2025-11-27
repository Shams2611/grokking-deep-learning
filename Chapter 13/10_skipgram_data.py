# creating skip-gram training data

import numpy as np

def create_skipgram_data(sentence, window_size=2):
    """create (center, context) pairs"""
    words = sentence.lower().split()
    pairs = []

    for i, center in enumerate(words):
        # context words within window
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)

        for j in range(start, end):
            if j != i:  # skip the center word
                pairs.append((center, words[j]))

    return pairs

# example
sentence = "the quick brown fox jumps over the lazy dog"
pairs = create_skipgram_data(sentence, window_size=2)

print(f"sentence: '{sentence}'")
print(f"window size: 2")
print()
print("skip-gram pairs (center, context):")
for center, context in pairs[:10]:
    print(f"  {center} -> {context}")
print(f"  ... ({len(pairs)} total pairs)")
