"""
Chapter 6: Multi-Layer Backpropagation

Now let's do backprop with actual layers (multiple neurons per layer)!

This is the real deal - what happens in actual neural networks.

Structure:
    input (3 neurons) -> hidden (4 neurons) -> output (1 neuron)

Each layer has a weight MATRIX connecting it to the next layer.
The backprop algorithm stays the same - just with matrices!

Key insight: when we backprop through a layer, we multiply by
the TRANSPOSE of the weight matrix. This "reverses" the flow.
"""

import numpy as np

np.random.seed(42)


def sigmoid(x):
    """Sigmoid activation - squashes values to (0, 1)."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    """
    Derivative of sigmoid.
    Note: takes the OUTPUT of sigmoid, not the input!
    d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    return output * (1 - output)


class SimpleNeuralNetwork:
    """
    A simple 2-layer neural network.

    input -> hidden (with sigmoid) -> output (with sigmoid)
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        # Using small random values to start
        self.weights_ih = np.random.randn(input_size, hidden_size) * 0.5
        self.weights_ho = np.random.randn(hidden_size, output_size) * 0.5

        print(f"Network shape: {input_size} -> {hidden_size} -> {output_size}")
        print(f"weights_ih shape: {self.weights_ih.shape}")
        print(f"weights_ho shape: {self.weights_ho.shape}")

    def forward(self, inputs):
        """Forward pass - compute output from inputs."""
        # Input to hidden
        self.hidden_input = np.dot(inputs, self.weights_ih)
        self.hidden_output = sigmoid(self.hidden_input)

        # Hidden to output
        self.final_input = np.dot(self.hidden_output, self.weights_ho)
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, inputs, targets, learning_rate):
        """
        Backward pass - update weights based on error.

        This is where the magic happens!
        """
        inputs = np.array(inputs).reshape(1, -1)  # make it a row vector
        targets = np.array(targets).reshape(1, -1)

        # 1. Calculate output error
        output_error = targets - self.final_output

        # 2. Calculate output delta (error * derivative)
        output_delta = output_error * sigmoid_derivative(self.final_output)

        # 3. Calculate hidden error (backpropagate!)
        # Error flows back THROUGH the weights
        hidden_error = output_delta.dot(self.weights_ho.T)

        # 4. Calculate hidden delta
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # 5. Update weights
        # weights_ho: hidden contributed to output error
        self.weights_ho += self.hidden_output.T.dot(output_delta) * learning_rate

        # weights_ih: input contributed to hidden error
        self.weights_ih += inputs.T.dot(hidden_delta) * learning_rate

        return np.mean(output_error ** 2)

    def train(self, X, y, epochs, learning_rate):
        """Train the network on data."""
        for epoch in range(epochs):
            total_error = 0

            for i in range(len(X)):
                # Forward pass
                output = self.forward(X[i])

                # Backward pass
                error = self.backward(X[i], y[i], learning_rate)
                total_error += error

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: avg_error = {total_error/len(X):.6f}")


# ============================================
# Demo: Learn XOR function!
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Multi-Layer Backpropagation - Learning XOR")
    print("=" * 60)

    # XOR is famous because it CAN'T be learned by single layer!
    # It's a classic test of multi-layer networks

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ])

    print("\nXOR Truth Table:")
    print("Input -> Output")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i]}")

    print("\n--- Creating Network ---")
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    print("\n--- Training ---\n")
    nn.train(X, y, epochs=10001, learning_rate=1.0)

    print("\n--- Testing ---")
    print("\nPredictions after training:")
    for i in range(len(X)):
        pred = nn.forward(X[i])
        rounded = round(pred[0][0])
        actual = y[i][0]
        status = "✓" if rounded == actual else "✗"
        print(f"  {X[i]} -> {pred[0][0]:.4f} (rounded: {rounded}) {status}")

    print("\n" + "=" * 60)
    print("Why XOR is special:")
    print("=" * 60)
    print("XOR is NOT linearly separable - you can't draw a straight")
    print("line to separate 0s from 1s. A single layer can only learn")
    print("linear patterns!")
    print("\nThe hidden layer learns intermediate representations that")
    print("MAKE the problem linearly separable. That's the power of depth!")

    print("\n" + "=" * 60)
    print("Backprop Summary:")
    print("=" * 60)
    print("1. Forward pass: input -> hidden -> output")
    print("2. Calculate output error")
    print("3. Backpropagate: error flows back through weights")
    print("4. Update weights in both layers")
    print("5. Repeat until error is low!")
