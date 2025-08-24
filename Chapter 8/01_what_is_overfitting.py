# overfitting = memorizing instead of learning
# model works great on training data, fails on new data

import numpy as np

# imagine a model that just memorizes
training_data = {
    "cat1": "cat",
    "cat2": "cat",
    "dog1": "dog"
}

# perfect on training!
print("training accuracy: 100%")
print()

# but on new data...
print("new image 'cat3':")
print("  memorizing model: ???")
print("  generalizing model: cat")
print()
print("overfitting = high training accuracy, low test accuracy")
