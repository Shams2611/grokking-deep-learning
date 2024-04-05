"""
Chapter 8: Early Stopping

The simplest regularization technique: just stop training!

The idea:
- Monitor validation error during training
- When validation error starts going UP, stop training
- Even if training error is still going down

It's like knowing when to stop studying for an exam.
At some point, more studying doesn't help and might even
make you overthink simple questions!

This prevents the network from overfitting by not letting
it train long enough to memorize the training data.
"""

import numpy as np

np.random.seed(42)


class EarlyStoppingTrainer:
    """
    A trainer that implements early stopping.
    """

    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience: how many epochs to wait for improvement
            min_delta: minimum change to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta

    def train(self, X_train, y_train, X_val, y_val, lr=0.01, max_epochs=1000):
        """
        Train with early stopping.
        """
        # Simple 2-layer network
        n_features = X_train.shape[1]
        n_hidden = 10
        n_output = 1

        W1 = np.random.randn(n_features, n_hidden) * 0.5
        W2 = np.random.randn(n_hidden, n_output) * 0.5

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = (W1.copy(), W2.copy())

        history = {
            'train_loss': [],
            'val_loss': [],
            'stopped_at': None
        }

        for epoch in range(max_epochs):
            # Forward pass (training)
            hidden = np.maximum(0, X_train @ W1)  # ReLU
            output = hidden @ W2

            train_loss = np.mean((output - y_train.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y_train.reshape(-1, 1)
            hidden_error = output_error @ W2.T
            hidden_error[hidden <= 0] = 0  # ReLU derivative

            W2 -= lr * (hidden.T @ output_error) / len(X_train)
            W1 -= lr * (X_train.T @ hidden_error) / len(X_train)

            # Validation
            hidden_val = np.maximum(0, X_val @ W1)
            output_val = hidden_val @ W2
            val_loss = np.mean((output_val - y_val.reshape(-1, 1)) ** 2)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                # Improvement!
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_weights = (W1.copy(), W2.copy())
            else:
                # No improvement
                epochs_without_improvement += 1

            if epoch % 50 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}!")
                print(f"Best validation loss was {best_val_loss:.4f}")
                history['stopped_at'] = epoch
                break

        # Restore best weights
        W1, W2 = best_weights
        return W1, W2, history


def generate_data(n_samples, noise=0.5):
    """Generate some noisy data."""
    X = np.random.randn(n_samples, 5)
    # True relationship: weighted sum of first 2 features
    y = 2 * X[:, 0] - X[:, 1] + noise * np.random.randn(n_samples)
    return X, y


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Early Stopping Demonstration")
    print("=" * 60)

    # Generate data
    X_train, y_train = generate_data(100, noise=0.5)
    X_val, y_val = generate_data(30, noise=0.5)
    X_test, y_test = generate_data(50, noise=0.5)

    print(f"\nDataset sizes:")
    print(f"  Training: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Train with early stopping
    print("\n" + "=" * 60)
    print("Training with Early Stopping (patience=10)")
    print("=" * 60 + "\n")

    trainer = EarlyStoppingTrainer(patience=10, min_delta=0.001)
    W1, W2, history = trainer.train(X_train, y_train, X_val, y_val, lr=0.01, max_epochs=500)

    # Show the training history
    print("\n" + "=" * 60)
    print("Training History:")
    print("=" * 60)

    # ASCII plot of loss curves
    n_epochs = len(history['train_loss'])

    print(f"\nTrain vs Validation Loss (ASCII):")
    print("-" * 50)

    max_loss = max(max(history['train_loss']), max(history['val_loss']))

    for i in range(0, n_epochs, max(1, n_epochs // 15)):
        train_bar = int(history['train_loss'][i] / max_loss * 30)
        val_bar = int(history['val_loss'][i] / max_loss * 30)

        print(f"Epoch {i:3d}: Train |{'#' * train_bar}")
        print(f"           Val   |{'*' * val_bar}")

    if history['stopped_at']:
        print(f"\n[STOPPED at epoch {history['stopped_at']}]")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final Evaluation:")
    print("=" * 60)

    hidden_test = np.maximum(0, X_test @ W1)
    pred_test = hidden_test @ W2
    test_loss = np.mean((pred_test - y_test.reshape(-1, 1)) ** 2)

    print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Explain early stopping
    print("\n" + "=" * 60)
    print("How Early Stopping Works:")
    print("=" * 60)

    print("""
1. Split data into training and validation sets

2. During training:
   - Compute loss on training data (for learning)
   - Also compute loss on validation data (for monitoring)

3. Keep track of the best validation loss

4. If validation loss doesn't improve for 'patience' epochs:
   - STOP training!
   - Restore weights from when validation was best

5. Benefits:
   - Prevents overfitting automatically
   - Don't need to guess the right number of epochs
   - No extra hyperparameters to tune (just patience)
   - Computationally free (we're already computing val loss)
    """)

    print("=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Monitor validation loss, not training loss")
    print("2. Stop when validation loss stops improving")
    print("3. 'Patience' = epochs to wait before stopping")
    print("4. Save the best weights, not the final weights!")
    print("5. Simple but very effective regularization")
