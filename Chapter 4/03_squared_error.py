# squared error
# square the error to make it positive

prediction = 0.85
goal = 1.0

error = prediction - goal
squared_error = error ** 2

print(f"error: {error}")
print(f"squared error: {squared_error}")

# always positive!
# big errors get punished more
