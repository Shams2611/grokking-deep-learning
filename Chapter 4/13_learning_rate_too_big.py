# learning rate too big

input = 0.5
goal = 0.8
weight = 0.0
lr = 10  # way too big!

print(f"learning rate: {lr} (too big)")
print()

for i in range(10):
    pred = input * weight
    error = (pred - goal) ** 2
    gradient = (pred - goal) * input
    weight = weight - lr * gradient
    print(f"  iter {i}: weight={weight:.2f}, error={error:.2f}")

print()
print("oscillating wildly! never converges")
