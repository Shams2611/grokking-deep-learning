# why gradient works

# gradient = (pred - goal) * input

# if pred > goal (too high):
#   (pred - goal) is positive
#   gradient is positive
#   weight -= positive = weight goes down
#   prediction goes down (good!)

# if pred < goal (too low):
#   (pred - goal) is negative
#   gradient is negative
#   weight -= negative = weight goes up
#   prediction goes up (good!)

print("gradient automatically adjusts direction!")
print()
print("pred too high -> decrease weight")
print("pred too low  -> increase weight")
print()
print("magnitude tells us HOW MUCH to change")
