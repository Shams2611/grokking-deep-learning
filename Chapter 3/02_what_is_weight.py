# what is a weight?
# its just a number that scales the input
# bigger weight = input matters more

input = 8.5

# try different weights
weights = [0.1, 0.5, 1.0, 2.0]

print("same input, different weights:")
for w in weights:
    pred = input * w
    print(f"  weight={w} -> prediction={pred}")
