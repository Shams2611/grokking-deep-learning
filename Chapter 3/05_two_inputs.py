# now with TWO inputs!
# each input has its own weight

toes = 8.5
wins = 0.65  # win percentage

weight_toes = 0.1
weight_wins = 0.2

# prediction is sum of weighted inputs
prediction = (toes * weight_toes) + (wins * weight_wins)

print(f"toes: {toes}, wins: {wins}")
print(f"prediction: {prediction:.3f}")
