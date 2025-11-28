# negative sampling - making training efficient

import numpy as np

print("NEGATIVE SAMPLING")
print()
print("problem: softmax over entire vocabulary is slow!")
print("  vocab of 100k words = 100k class softmax")
print()
print("solution: turn into binary classification")
print()
print("positive example: (cat, sat) from real text")
print("negative examples: (cat, elephant), (cat, quantum), ...")
print("  randomly sampled words unlikely to appear together")
print()
print("now just predict: real pair or fake pair?")
print("  sigmoid instead of softmax!")
print()

# example
def sample_negatives(vocab, positive_word, k=5):
    """sample k negative words"""
    negatives = []
    while len(negatives) < k:
        word = np.random.choice(vocab)
        if word != positive_word:
            negatives.append(word)
    return negatives

vocab = ["cat", "dog", "sat", "mat", "tree", "car", "book", "run"]
positives = [("cat", "sat")]
k = 3

for center, context in positives:
    negs = sample_negatives(vocab, context, k)
    print(f"positive: ({center}, {context})")
    print(f"negatives: {[(center, n) for n in negs]}")
