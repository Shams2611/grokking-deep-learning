# derivative of squared error

# error = (pred - goal)^2
# pred = input * weight

# chain rule:
# d_error/d_weight = 2 * (pred - goal) * input

# simplified (ignore the 2):
# gradient = (pred - goal) * input

input = 0.5
goal = 0.8
weight = 0.5

pred = input * weight
gradient = (pred - goal) * input

print(f"prediction: {pred}")
print(f"goal: {goal}")
print(f"gradient: {gradient}")
print()
print("negative gradient = increase weight")
