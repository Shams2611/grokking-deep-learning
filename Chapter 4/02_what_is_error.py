# error = how wrong we are

prediction = 0.85
goal = 1.0

error = prediction - goal

print(f"error: {error}")
# negative means we predicted too low
# positive means we predicted too high
