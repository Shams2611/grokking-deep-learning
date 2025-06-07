# multiple outputs with lists

input = 8.5
weights = [0.1, 0.2, 0.3]

predictions = [input * w for w in weights]

print(f"input: {input}")
print(f"weights: {weights}")
print(f"predictions: {predictions}")
