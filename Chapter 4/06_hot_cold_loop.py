# hot cold learning loop

input = 0.5
goal = 0.8
weight = 0.0
step = 0.01

for i in range(20):
    pred = input * weight
    error = (pred - goal) ** 2

    # try going up
    weight_up = weight + step
    pred_up = input * weight_up
    error_up = (pred_up - goal) ** 2

    # try going down
    weight_down = weight - step
    pred_down = input * weight_down
    error_down = (pred_down - goal) ** 2

    # which is better?
    if error_up < error_down:
        weight = weight_up
    else:
        weight = weight_down

    if i % 5 == 0:
        print(f"step {i}: weight={weight:.3f}, error={error:.4f}")

print(f"final weight: {weight:.3f}")
