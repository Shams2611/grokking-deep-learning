# multiclass classification
# more than 2 classes to choose from

print("MULTICLASS CLASSIFICATION")
print()
print("binary: cat or not cat")
print("multiclass: cat, dog, or bird")
print()
print("example outputs we want:")
print("  image of cat -> [0.9, 0.05, 0.05]")
print("  image of dog -> [0.1, 0.8, 0.1]")
print("  image of bird -> [0.0, 0.1, 0.9]")
print()
print("outputs should:")
print("  1. be between 0 and 1 (probabilities)")
print("  2. sum to 1.0")
print()
print("sigmoid doesnt work - outputs are independent!")
print("we need: softmax")
