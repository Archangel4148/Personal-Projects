from typing import Callable

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from math_definitions import relu, rmse_per_sample, sigmoid, loss_fn_general, loss_per_sample
from tools import plot_2d_new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Layer:
    def __init__(self, in_features: int, out_features: int, activation: Callable | None, initialization: str = None,
                 device=None):
        device = device or torch.device("cpu")
        if initialization == "kaiming":
            self.W = torch.randn(in_features, out_features, device=device) * np.sqrt(2 / in_features)
        else:
            self.W = torch.randn(in_features, out_features, device=device) * 0.01
        self.b = torch.zeros(out_features, device=device)

        self.W.requires_grad_(True)
        self.b.requires_grad_(True)

        self.activation = activation

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation:
            z = self.activation(z)
        return z

    def parameters(self):
        return [self.W, self.b]


class Network:
    def __init__(self, node_counts: list[int], activations: list[Callable | None], initializations: list[str | None]):
        # Validate inputs
        assert (len(node_counts) - 1) == len(activations) == len(initializations)  # Lists must be same size
        assert len(node_counts) >= 2  # Need at least input + output nodes

        # Save variables for later storage
        self.node_counts = node_counts
        self.activations = activations
        self.initializations = initializations

        # Construct the layers
        self.layers = []
        for i in range(len(node_counts) - 1):
            layer = Layer(
                in_features=node_counts[i],
                out_features=node_counts[i + 1],
                activation=activations[i],
                initialization=initializations[i],
                device=device
            )
            self.layers.append(layer)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass x through all layers
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        # Get the parameters of all layers
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def save(self, path: str):
        save_dict = {
            "node_counts": self.node_counts,
            "activations": [act.__name__ if act else None for act in self.activations],
            "initializations": self.initializations,
            "parameters": [(layer.W.detach().cpu(), layer.b.detach().cpu()) for layer in self.layers]
        }
        torch.save(save_dict, path)

    @classmethod
    def from_save(cls, path: str, device=None):
        """Load a network from a saved file."""
        save_dict = torch.load(path, map_location=device or torch.device("cpu"))

        # Map string activations back to functions
        activation_map = {
            "relu": relu,
            "sigmoid": sigmoid,
            None: None,
            "none": None
        }
        activations = [activation_map[a.lower()] if isinstance(a, str) else a for a in save_dict["activations"]]

        net = cls(
            node_counts=save_dict["node_counts"],
            activations=activations,
            initializations=save_dict["initializations"]
        )
        return net


def y_hat_network(network, x):
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        y_hat = network.forward(x_t)
        return y_hat.cpu().numpy().squeeze()


def main():
    # Model hyperparameters
    is_classifier = True
    max_steps = 1000
    learning_rate = 0.05
    feature_count = 16

    # Early stopping arguments
    early_stop_enabled = True
    patience = 10
    eval_every = max_steps / 20

    features = pd.read_csv("datasets/letter_recognition/letter_recognition_features.csv").to_numpy()

    # Extract labels and convert to integers
    labels = pd.read_csv("datasets/letter_recognition/letter_recognition_targets.csv")
    labels = labels.iloc[:, 0].values  # extract column
    labels = np.array([ord(c) - ord('A') for c in labels], dtype=int)

    repetitions = 1
    for _ in range(repetitions):
        # data = generate_dummy_data(n_features=feature_count, n_points=100, seed=1, pattern="linear", task="classification" if is_classifier else "regression")
        # data = generate_polynomial_data(n_features=feature_count, n_points=200, seed=None, degree=5, noise_std=0.5, task="classification" if is_classifier else "regression")

        # Build the network
        # network = Network(node_counts=[feature_count, 1], activations=[sigmoid], initializations=[None])  # Logistic regression

        # Network that can learn "any" shape
        # network = Network(node_counts=[feature_count, 8, 8, 1], activations=[relu, relu, None], initializations=["kaiming", "kaiming", None])
        network = Network(
            node_counts=[16, 64, 64, 26],
            activations=[relu, relu, None],
            initializations=["kaiming", "kaiming", None]
        )

        # Function to make final classification decision
        def decision_func(y):
            # return y > 0.5
            return torch.argmax(y, dim=1)

        # Prepare the data
        # features = np.array([p[:-1] for p in data], dtype=float)
        # labels = np.array([p[-1] for p in data], dtype=float)
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        if early_stop_enabled:
            # Split into validation and test set
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        else:
            X_test, y_test = X_temp, y_temp

        # Normalize the training points (only based on the training data to avoid data leakage)
        mean_vals = X_train.mean(axis=0)
        std_vals = X_train.std(axis=0)
        X_train_norm = (X_train - mean_vals) / (std_vals + 1e-8)
        X_test_norm = (X_test - mean_vals) / (std_vals + 1e-8)
        if early_stop_enabled:
            X_val_norm = (X_val - mean_vals) / (std_vals + 1e-8)

        # For regression models, normalize target points (to avoid exploding outputs)
        if is_classifier:
            y_train_norm = y_train
            y_test_norm = y_test
            if early_stop_enabled:
                y_val_norm = y_val
        else:
            y_mean = y_train.mean()
            y_std = y_train.std()
            y_train_norm = (y_train - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std
            if early_stop_enabled:
                y_val_norm = (y_val - y_mean) / y_std

        # Convert to tensors for GPU acceleration
        X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        y_train_tensor_norm = torch.tensor(y_train_norm, dtype=torch.long, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
        y_test_tensor_norm = torch.tensor(y_test_norm, dtype=torch.long, device=device)
        if early_stop_enabled:
            X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
            y_val_tensor_norm = torch.tensor(y_val_norm, dtype=torch.long, device=device)

        # Tracking variables for loss tracking and early stopping
        train_loss_history = []
        test_loss_history = []
        step_history = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        # Training loop
        for step in range(max_steps):

            # Track training/test loss as the test progresses
            if step % eval_every == 0:
                with torch.no_grad():
                    # Training loss
                    y_hat_train = network.forward(X_train_tensor)
                    train_loss, _ = loss_fn_general(
                        y_hat_train, y_train_tensor_norm, network.parameters(),
                        lambda_l1=0.0, lambda_l2=0.0
                    )

                    # Test loss
                    y_hat_test = network.forward(X_test_tensor)
                    test_loss, _ = loss_fn_general(
                        y_hat_test, y_test_tensor_norm, network.parameters(),
                        lambda_l1=0.0, lambda_l2=0.0
                    )
                    if early_stop_enabled:
                        # Validation loss
                        y_hat_val = network.forward(X_val_tensor)
                        val_loss, _ = loss_fn_general(
                            y_hat_val, y_val_tensor_norm, network.parameters(),
                            lambda_l1=0.0, lambda_l2=0.0
                        )

                        if val_loss.item() < best_val_loss - 1e-6:
                            best_val_loss = val_loss.item()
                            patience_counter = 0

                            best_state = [
                                (layer.W.detach().clone(), layer.b.detach().clone())
                                for layer in network.layers
                            ]
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            print(f"Early stopping at step {step}")
                            break

                # Track losses for plotting
                train_loss_history.append(train_loss.item())
                test_loss_history.append(test_loss.item())
                step_history.append(step)

            y_hat = network.forward(X_train_tensor)
            loss, _ = loss_fn_general(y_hat, y_train_tensor_norm, network.parameters(), lambda_l1=0.0, lambda_l2=0.0)
            loss.backward()

            # Apply gradient descent to each parameter
            with torch.no_grad():
                for p in network.parameters():
                    p -= learning_rate * p.grad
                    p.grad.zero_()

        # Apply the best state (if any)
        if best_state is not None:
            with torch.no_grad():
                for layer, (W_best, b_best) in zip(network.layers, best_state):
                    layer.W.copy_(W_best)
                    layer.b.copy_(b_best)

        print("\nLearned parameters:")
        for i, layer in enumerate(network.layers):
            W = layer.W.detach().cpu().numpy()
            b = layer.b.detach().cpu().numpy()
            print(f"Layer {i}:")
            print(f"  W = {W.flatten()}")
            print(f"  b = {b.flatten()}")
        print()

        with torch.no_grad():
            y_hat_train = network.forward(X_train_tensor)
            y_hat_test = network.forward(X_test_tensor)

            if is_classifier:
                y_hat_train_rescaled = y_hat_train
                y_hat_test_rescaled = y_hat_test
            else:
                y_hat_train_rescaled = y_hat_train * y_std + y_mean
                y_hat_test_rescaled = y_hat_test * y_std + y_mean

            # Training loss
            train_losses, loss_label = loss_per_sample(y_hat_train_rescaled, y_train_tensor)
            mean_train_loss = torch.mean(train_losses).item()
            print(f"Training {loss_label} (mean): {mean_train_loss}")

            # Test loss
            test_losses, loss_label = loss_per_sample(y_hat_test_rescaled, y_test_tensor)
            mean_test_loss = torch.mean(test_losses).item()
            print(f"Test {loss_label} (mean): {mean_test_loss}")

            if is_classifier:
                # Test predictions (only for classification model)
                y_pred = decision_func(y_hat_test).int()
                accuracy = (y_pred == y_test_tensor_norm.int()).float().mean().item()
                print(f"Test Accuracy: {round(accuracy * 100, 3)}%")
            else:
                # RMSE
                train_rmse = rmse_per_sample(y_hat_train_rescaled, y_train_tensor).item()
                test_rmse = rmse_per_sample(y_hat_test_rescaled, y_test_tensor).item()

                print(f"Training RMSE: {train_rmse}")
                print(f"Test RMSE: {test_rmse}")

        # Save learned weights
        network.save("model_weights.pt")

        # Prepare plotting data
        # training_data = np.hstack([X_train_tensor.cpu().numpy(), y_train_tensor_norm.cpu().numpy()])
        # testing_data = np.hstack([X_test_tensor.cpu().numpy(), y_test_tensor_norm.cpu().numpy()])
        training_data = np.hstack([X_train_tensor.cpu().numpy(), y_train_tensor_norm.cpu().numpy()[:, None]])
        testing_data = np.hstack([X_test_tensor.cpu().numpy(), y_test_tensor_norm.cpu().numpy()[:, None]])

        # Plot training & test loss vs steps
        plt.figure(figsize=(8, 5))
        plt.plot(step_history, train_loss_history, label="Training loss")
        plt.plot(step_history, test_loss_history, label="Test loss")

        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Training vs Test Loss Over Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # If possible, plot the resulting model
        if feature_count == 1:
            plot_2d_new(training_points=training_data, test_points=testing_data,
                        y_hat_func=lambda x: y_hat_network(network, x), is_classifier=is_classifier,
                        decision_function=decision_func)
        # elif feature_count == 2:
        #     plot_3d_new(data_points=training_data, y_hat_func=lambda x: y_hat_network(network, x))


if __name__ == '__main__':
    main()
