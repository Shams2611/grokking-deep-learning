# learning rate too small

input = 0.5
goal = 0.8
weight = 0.0
lr = 0.001  # very small!

print(f"learning rate: {lr} (too small)")
print()

for i in range(20):
    pred = input * weight
    error = (pred - goal) ** 2
    gradient = (pred - goal) * input
    weight = weight - lr * gradient

print(f"after 20 iterations:")
print(f"  weight: {weight:.4f}")
print(f"  error: {error:.6f}")
print()
print("barely moved! would need thousands of iterations")
