# gradient descent as a function

def gradient_descent(input, goal, lr=0.1, iterations=100):
    weight = 0.0

    for i in range(iterations):
        pred = input * weight
        error = (pred - goal) ** 2
        gradient = (pred - goal) * input
        weight = weight - lr * gradient

    return weight

# test it
input = 0.5
goal = 0.8

weight = gradient_descent(input, goal, lr=1.0, iterations=20)

print(f"learned weight: {weight:.4f}")
print(f"prediction: {input * weight:.4f}")
print(f"goal: {goal}")
print()
print("CHAPTER 4 DONE")
print("learned: gradient descent finds weights!")
