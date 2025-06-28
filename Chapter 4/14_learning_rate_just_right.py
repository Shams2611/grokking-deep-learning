# learning rate just right

input = 0.5
goal = 0.8
weight = 0.0
lr = 1.0  # good for this problem

print(f"learning rate: {lr}")
print()

for i in range(10):
    pred = input * weight
    error = (pred - goal) ** 2
    gradient = (pred - goal) * input
    weight = weight - lr * gradient
    print(f"  iter {i}: weight={weight:.4f}, error={error:.6f}")

print()
print("converges quickly and smoothly!")
