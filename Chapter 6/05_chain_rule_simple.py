# chain rule simple example

# if y = f(g(x))
# dy/dx = dy/dg * dg/dx

# example:
# g(x) = 2x
# f(g) = g^2
# y = f(g(x)) = (2x)^2 = 4x^2

x = 3

g = 2 * x
y = g ** 2

# derivatives
dg_dx = 2
dy_dg = 2 * g

# chain rule
dy_dx = dy_dg * dg_dx

print(f"x = {x}")
print(f"g = 2x = {g}")
print(f"y = g^2 = {y}")
print()
print(f"dy/dg = 2g = {dy_dg}")
print(f"dg/dx = 2")
print(f"dy/dx = dy/dg * dg/dx = {dy_dx}")
