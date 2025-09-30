# making predictions with a trained classifier

import numpy as np
np.random.seed(42)

def relu(x): return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# pretend these are trained weights
w1 = np.random.randn(2, 8) * 0.5
w2 = np.random.randn(8, 3) * 0.5

class_names = ["cat", "dog", "bird"]

def predict(x):
    hidden = relu(x @ w1)
    logits = hidden @ w2
    probs = softmax(logits)

    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class]

    return predicted_class, confidence, probs

# test predictions
test_inputs = [
    np.array([0.1, 0.9]),
    np.array([0.9, 0.1]),
    np.array([0.5, 0.5]),
]

print("PREDICTIONS:")
print()
for x in test_inputs:
    cls, conf, probs = predict(x)
    print(f"input: {x}")
    print(f"  probs: {np.round(probs, 3)}")
    print(f"  prediction: {class_names[cls]} ({conf*100:.1f}% confident)")
    print()
