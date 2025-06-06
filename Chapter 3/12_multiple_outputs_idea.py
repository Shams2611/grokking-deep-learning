# what if we want MULTIPLE predictions?
# like predicting: win?, hurt?, sad?

# one input, three outputs
# need three weights!

input = 8.5

weight_win = 0.1
weight_hurt = 0.2
weight_sad = 0.3

pred_win = input * weight_win
pred_hurt = input * weight_hurt
pred_sad = input * weight_sad

print(f"input: {input}")
print(f"predictions:")
print(f"  win: {pred_win}")
print(f"  hurt: {pred_hurt}")
print(f"  sad: {pred_sad}")
