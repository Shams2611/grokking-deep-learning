# early stopping = stop before overfitting
# monitor validation error, stop when it increases

import numpy as np

# simulated training
print("watching validation error...")
print()

val_errors = [0.8, 0.6, 0.5, 0.45, 0.43, 0.42, 0.43, 0.45, 0.48]
best_error = float('inf')
patience = 2
wait = 0

for epoch, error in enumerate(val_errors):
    status = ""
    if error < best_error:
        best_error = error
        wait = 0
        status = "<- best so far"
    else:
        wait += 1
        status = f"(no improvement: {wait}/{patience})"
        if wait >= patience:
            status = "<- STOP HERE"
            print(f"epoch {epoch}: val_error = {error:.2f} {status}")
            break

    print(f"epoch {epoch}: val_error = {error:.2f} {status}")

print()
print(f"stopped at epoch {epoch}, best error was {best_error:.2f}")
