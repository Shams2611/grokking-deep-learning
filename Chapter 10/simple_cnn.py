"""
Chapter 10: Simple CNN Implementation

Putting it all together - a complete CNN from scratch!

Architecture:
    Input (28x28)
    -> Conv (3x3, 8 filters)
    -> ReLU
    -> MaxPool (2x2)
    -> Flatten
    -> Dense (output)

This is a simplified version of what you'd use for MNIST.
The key is seeing how conv layers connect to dense layers!
"""

import numpy as np

np.random.seed(42)


class Conv2D:
    """2D Convolutional layer."""

    def __init__(self, num_filters, kernel_size, input_shape):
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # Initialize filters randomly
        # Shape: (num_filters, kernel_h, kernel_w)
        self.filters = np.random.randn(
            num_filters, kernel_size, kernel_size
        ) / (kernel_size * kernel_size)

        # Store for backprop
        self.last_input = None

    def forward(self, input_data):
        """Forward pass through conv layer."""
        self.last_input = input_data

        h, w = input_data.shape
        out_h = h - self.kernel_size + 1
        out_w = w - self.kernel_size + 1

        output = np.zeros((self.num_filters, out_h, out_w))

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = input_data[
                        i : i + self.kernel_size,
                        j : j + self.kernel_size
                    ]
                    output[f, i, j] = np.sum(region * self.filters[f])

        return output

    def backward(self, d_out, learning_rate):
        """Backward pass - update filters."""
        d_filters = np.zeros_like(self.filters)

        for f in range(self.num_filters):
            for i in range(d_out.shape[1]):
                for j in range(d_out.shape[2]):
                    region = self.last_input[
                        i : i + self.kernel_size,
                        j : j + self.kernel_size
                    ]
                    d_filters[f] += d_out[f, i, j] * region

        self.filters -= learning_rate * d_filters


class MaxPool:
    """Max pooling layer."""

    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.last_input = None

    def forward(self, input_data):
        """Forward pass."""
        self.last_input = input_data

        num_filters, h, w = input_data.shape
        out_h = h // self.pool_size
        out_w = w // self.pool_size

        output = np.zeros((num_filters, out_h, out_w))

        for f in range(num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = input_data[
                        f,
                        i * self.pool_size : (i + 1) * self.pool_size,
                        j * self.pool_size : (j + 1) * self.pool_size
                    ]
                    output[f, i, j] = np.max(region)

        return output

    def backward(self, d_out):
        """Backward pass - route gradient to max positions."""
        d_input = np.zeros_like(self.last_input)

        num_filters, out_h, out_w = d_out.shape

        for f in range(num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    i_start = i * self.pool_size
                    j_start = j * self.pool_size

                    region = self.last_input[
                        f,
                        i_start : i_start + self.pool_size,
                        j_start : j_start + self.pool_size
                    ]

                    # Find max position
                    max_pos = np.unravel_index(np.argmax(region), region.shape)

                    # Route gradient to max position
                    d_input[f, i_start + max_pos[0], j_start + max_pos[1]] = d_out[f, i, j]

        return d_input


class Dense:
    """Fully connected layer."""

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)
        self.last_input = None

    def forward(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_out, learning_rate):
        d_weights = np.outer(self.last_input, d_out)
        d_biases = d_out
        d_input = np.dot(d_out, self.weights.T)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class SimpleCNN:
    """Simple CNN for demonstration."""

    def __init__(self):
        # Architecture: 28x28 -> Conv(8 filters) -> Pool -> Dense -> Output

        # Conv: 28x28 -> 8 x 26x26
        self.conv = Conv2D(num_filters=8, kernel_size=3, input_shape=(28, 28))

        # Pool: 8 x 26x26 -> 8 x 13x13
        self.pool = MaxPool(pool_size=2)

        # Dense: 8*13*13 = 1352 -> 10 (for 10 digits)
        self.dense = Dense(8 * 13 * 13, 10)

    def forward(self, image):
        """Forward pass through entire network."""
        # Conv + ReLU
        out = self.conv.forward(image)
        out = relu(out)

        # Pool
        out = self.pool.forward(out)

        # Flatten for dense layer
        self.flat_shape = out.shape
        out_flat = out.flatten()

        # Dense + Softmax
        out = self.dense.forward(out_flat)
        return softmax(out)

    def train_step(self, image, label, learning_rate=0.005):
        """One training step."""
        # Forward pass
        probs = self.forward(image)

        # Cross-entropy loss gradient
        d_out = probs.copy()
        d_out[label] -= 1

        # Backprop through dense
        d_flat = self.dense.backward(d_out, learning_rate)

        # Reshape back to pool output shape
        d_pool = d_flat.reshape(self.flat_shape)

        # Backprop through pool
        d_pool_input = self.pool.backward(d_pool)

        # ReLU derivative
        conv_out = self.conv.forward(image)  # recalculate for derivative
        d_relu = d_pool_input * relu_derivative(conv_out)

        # Backprop through conv
        self.conv.backward(d_relu, learning_rate)

        # Return loss for monitoring
        return -np.log(probs[label] + 1e-10)


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Simple CNN Implementation")
    print("=" * 60)

    # Create fake MNIST-like data
    print("\nCreating simple test data...")

    # Fake images: 10 samples of 28x28
    # Each has a pattern indicating its class
    X = np.random.randn(10, 28, 28) * 0.5

    # Add simple patterns to make classes distinguishable
    for i in range(10):
        # Add a diagonal line whose position depends on class
        for j in range(10):
            X[i, j + i, j + 5] = 2.0
            X[i, j + 5, j + i] = 2.0

    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print(f"Data shape: {X.shape}")
    print(f"Labels: {y}")

    # Create and train CNN
    print("\n--- Training CNN ---")
    cnn = SimpleCNN()

    print("\nArchitecture:")
    print("  Input:  28 x 28")
    print("  Conv:   8 filters x 26 x 26 (3x3 kernel)")
    print("  ReLU:   8 x 26 x 26")
    print("  Pool:   8 x 13 x 13 (2x2 max pool)")
    print("  Flat:   1352")
    print("  Dense:  10 (softmax output)")

    # Train for a few epochs
    epochs = 50
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for i in range(len(X)):
            loss = cnn.train_step(X[i], y[i], learning_rate=0.005)
            total_loss += loss

            # Check accuracy
            pred = np.argmax(cnn.forward(X[i]))
            if pred == y[i]:
                correct += 1

        if epoch % 10 == 0:
            acc = correct / len(X)
            print(f"Epoch {epoch}: loss={total_loss/len(X):.4f}, accuracy={acc:.2%}")

    # Final predictions
    print("\n--- Final Predictions ---")
    for i in range(len(X)):
        probs = cnn.forward(X[i])
        pred = np.argmax(probs)
        conf = probs[pred]
        status = "correct" if pred == y[i] else "WRONG"
        print(f"Sample {i}: true={y[i]}, pred={pred} ({conf:.2%}) - {status}")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Conv layers detect local features")
    print("2. Pooling reduces spatial size, adds invariance")
    print("3. Dense layers combine features for classification")
    print("4. Backprop flows through all layers")
    print("5. Real CNNs are deeper and use batches")
    print("6. Libraries like PyTorch/TensorFlow make this easier!")
