# when to stop training?

input = 0.5
goal = 0.8
weight = 0.0
lr = 1.0
threshold = 0.0001

iteration = 0
while True:
    pred = input * weight
    error = (pred - goal) ** 2

    if error < threshold:
        print(f"converged at iteration {iteration}!")
        break

    gradient = (pred - goal) * input
    weight = weight - lr * gradient
    iteration += 1

    if iteration > 1000:
        print("max iterations reached")
        break

print(f"final weight: {weight:.6f}")
print(f"final error: {error:.8f}")
