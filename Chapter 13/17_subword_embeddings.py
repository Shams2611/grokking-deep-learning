# subword embeddings - handling unknown words

print("SUBWORD EMBEDDINGS")
print()
print("problem with word embeddings:")
print("  - unknown words (OOV) have no embedding")
print('  - "unfriendliness" might not be in vocab')
print()
print("solution: break into subwords")
print()
print("FastText approach:")
print('  "unfriendliness" -> ["un", "friend", "li", "ness"]')
print("  word embedding = sum of subword embeddings")
print()
print("benefits:")
print("  - handles any word (even misspellings)")
print("  - captures morphology (un-, -ness, -ing)")
print("  - better for rare words")
print()

# simple example
def simple_subwords(word, n=3):
    """character n-grams"""
    word = f"<{word}>"  # add boundary markers
    subwords = []
    for i in range(len(word) - n + 1):
        subwords.append(word[i:i+n])
    return subwords

word = "learning"
print(f"subwords of '{word}':")
print(f"  {simple_subwords(word, n=3)}")
