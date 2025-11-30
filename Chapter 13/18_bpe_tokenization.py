# BPE - Byte Pair Encoding

print("BYTE PAIR ENCODING (BPE)")
print()
print("modern tokenization method")
print("used by GPT, BERT, etc.")
print()
print("algorithm:")
print("  1. start with character-level tokens")
print("  2. count all adjacent pairs")
print("  3. merge most frequent pair")
print("  4. repeat until vocab size reached")
print()

# simple demonstration
def simple_bpe_step(text):
    # count pairs
    pairs = {}
    words = text.split()
    for word in words:
        chars = list(word) + ['</w>']
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

text = "low low low lower lowest"
print(f"text: {text}")
print()
pairs = simple_bpe_step(text)
print("pair frequencies:")
for pair, count in sorted(pairs.items(), key=lambda x: -x[1])[:5]:
    print(f"  {pair}: {count}")
print()
print("most common pair would be merged first")
print("'lo' + 'w' -> 'low' as single token")
