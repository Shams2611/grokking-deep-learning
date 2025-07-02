# trace through weight update

input = 2.0
goal = 0.8
weight = 0.5
lr = 0.1

print("BEFORE:")
print(f"  input: {input}")
print(f"  weight: {weight}")
print(f"  goal: {goal}")
print()

pred = input * weight
print(f"prediction: {pred}")
print(f"delta: {pred - goal}")
print()

gradient = (pred - goal) * input
print(f"gradient: {gradient}")
print()

weight_change = lr * gradient
print(f"weight change: {weight_change}")
print()

new_weight = weight - weight_change
print(f"new weight: {new_weight}")
print(f"new prediction: {input * new_weight}")
