# gradient descent loop

input = 0.5
goal = 0.8
weight = 0.0
lr = 0.1

print("gradient descent training:")
for i in range(20):
    pred = input * weight
    error = (pred - goal) ** 2
    gradient = (pred - goal) * input
    weight = weight - lr * gradient

    if i % 4 == 0:
        print(f"  iter {i:2d}: weight={weight:.4f}, error={error:.6f}")

print(f"\nfinal weight: {weight:.4f}")
print(f"final pred: {input * weight:.4f}")
