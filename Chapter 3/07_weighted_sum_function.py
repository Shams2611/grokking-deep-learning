# make it a function

def weighted_sum(inputs, weights):
    result = 0
    for i, w in zip(inputs, weights):
        result += i * w
    return result

inputs = [8.5, 0.65, 1.2]
weights = [0.1, 0.2, 0.0]

pred = weighted_sum(inputs, weights)
print(f"inputs: {inputs}")
print(f"weights: {weights}")
print(f"prediction: {pred:.3f}")
