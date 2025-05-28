# predict if team wins based on average toes
# weird example from the book but whatever

toes = 8.5  # average toes per player (lol)
weight = 0.1

win_prediction = toes * weight

print(f"average toes: {toes}")
print(f"win prediction: {win_prediction}")
print(f"closer to 1 = more likely to win")
