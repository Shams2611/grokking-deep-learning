# comparing learning rates

input = 0.5
goal = 0.8

learning_rates = [0.01, 0.1, 1.0, 2.0]

for lr in learning_rates:
    weight = 0.0
    for _ in range(20):
        pred = input * weight
        gradient = (pred - goal) * input
        weight = weight - lr * gradient

    final_error = (input * weight - goal) ** 2
    print(f"lr={lr:4.2f}: final_weight={weight:.4f}, error={final_error:.6f}")
