# cleaner with lists

inputs = [8.5, 0.65, 1.2]
weights = [0.1, 0.2, 0.0]

# weighted sum with zip
prediction = sum(i * w for i, w in zip(inputs, weights))

print(f"prediction: {prediction:.3f}")

# this is called a DOT PRODUCT btw
