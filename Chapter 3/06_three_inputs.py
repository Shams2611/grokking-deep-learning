# three inputs now
# toes, win%, and num fans

toes = 8.5
wins = 0.65
fans = 1.2  # in millions

w_toes = 0.1
w_wins = 0.2
w_fans = 0.0  # fans dont matter apparently

prediction = toes*w_toes + wins*w_wins + fans*w_fans

print(f"inputs: toes={toes}, wins={wins}, fans={fans}")
print(f"weights: {w_toes}, {w_wins}, {w_fans}")
print(f"prediction: {prediction:.3f}")
