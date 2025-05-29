# predict multiple games

weight = 0.1

# toes for each game
toes = [8.5, 9.5, 10, 9]

print("predictions for each game:")
for i, t in enumerate(toes):
    pred = t * weight
    print(f"  game {i+1}: toes={t}, prediction={pred:.2f}")
