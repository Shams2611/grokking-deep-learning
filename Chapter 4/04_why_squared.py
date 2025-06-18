# why squared error?

errors = [0.1, -0.1, 0.5, -0.5]

print("raw errors vs squared:")
for e in errors:
    print(f"  {e:+.1f} -> {e**2:.2f}")

print()
print("benefits of squaring:")
print("  1. always positive")
print("  2. big errors hurt more")
print("  3. smooth curve (good for math)")
