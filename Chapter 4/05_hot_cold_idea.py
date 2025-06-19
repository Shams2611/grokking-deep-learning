# hot and cold learning
# like the kids game!

# try a weight, check error
# adjust weight up or down
# see if error gets better

weight = 0.5
input = 0.5
goal = 0.8

pred = input * weight
error = (pred - goal) ** 2

print(f"weight: {weight}")
print(f"prediction: {pred}")
print(f"error: {error}")
print()
print("is 0.5 a good weight? lets try others...")
