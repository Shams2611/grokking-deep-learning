# delta = pred - goal

# this is also called the "error signal"
# its how wrong we are (with direction)

pred = 0.85
goal = 1.0

delta = pred - goal

print(f"prediction: {pred}")
print(f"goal: {goal}")
print(f"delta: {delta}")
print()
print("delta is negative = we predicted too low")
print("delta is positive = we predicted too high")
print()
print("gradient = delta * input")
