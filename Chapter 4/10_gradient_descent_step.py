# one step of gradient descent

input = 0.5
goal = 0.8
weight = 0.5
learning_rate = 0.1

# forward pass
pred = input * weight
error = (pred - goal) ** 2

# compute gradient
gradient = (pred - goal) * input

# update weight (go opposite direction of gradient)
weight = weight - learning_rate * gradient

print(f"old weight: 0.5")
print(f"gradient: {gradient}")
print(f"new weight: {weight:.4f}")
