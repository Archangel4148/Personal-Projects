import dataclasses
from typing import Callable, Self

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor

from math_definitions import rmse, loss_fn_general, loss_per_sample, sigmoid, relu
from tools import plot_2d_new, plot_3d_new, generate_dummy_data, generate_polynomial_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclasses.dataclass
class LearningData:
    x_train_tensor_norm: Tensor
    x_test_tensor_norm: Tensor
    y_train_tensor_norm: Tensor
    y_train_tensor: Tensor
    y_test_tensor_norm: Tensor
    y_test_tensor: Tensor
    x_val_tensor_norm: Tensor | None
    y_val_tensor_norm: Tensor | None
    y_val_tensor: Tensor | None

    @classmethod
    def build(cls, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
              is_classifier: bool, x_val: np.ndarray = None, y_val: np.ndarray = None) -> Self:
        # Normalize the training points (only based on the training data to avoid data leakage)
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_train_norm = (x_train - x_mean) / (x_std + 1e-8)
        x_test_norm = (x_test - x_mean) / (x_std + 1e-8)

        # Validation normalization
        if x_val is not None:
            x_val_norm = (x_val - x_mean) / (x_std + 1e-8)
        else:
            x_val_norm = None

        # For regression models, normalize target points (to avoid exploding outputs)
        if is_classifier:
            y_train_norm = y_train
            y_test_norm = y_test
            y_val_norm = y_val
        else:
            y_mean = y_train.mean()
            y_std = y_train.std()
            y_train_norm = (y_train - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std
            y_val_norm = ((y_val - y_mean) / y_std) if y_val is not None else None

        # Convert to tensors for GPU acceleration
        x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32, device=device)
        x_test_tensor = torch.tensor(x_test_norm, dtype=torch.float32, device=device)
        y_train_tensor_norm = torch.tensor(y_train_norm, dtype=torch.float32, device=device).view(-1, 1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
        y_test_tensor_norm = torch.tensor(y_test_norm, dtype=torch.float32, device=device).view(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)

        if x_val_norm is not None:
            x_val_tensor = torch.tensor(x_val_norm, dtype=torch.float32, device=device)
            y_val_tensor_norm = torch.tensor(y_val_norm, dtype=torch.float32, device=device).view(-1, 1)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
        else:
            x_val_tensor = y_val_tensor_norm = y_val_tensor = None

        return LearningData(
            x_train_tensor_norm=x_train_tensor,
            x_test_tensor_norm=x_test_tensor,
            y_train_tensor_norm=y_train_tensor_norm,
            y_train_tensor=y_train_tensor,
            y_test_tensor_norm=y_test_tensor_norm,
            y_test_tensor=y_test_tensor,
            x_val_tensor_norm=x_val_tensor,
            y_val_tensor_norm=y_val_tensor_norm,
            y_val_tensor=y_val_tensor
        )


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

    def train(self, data: LearningData, learning_rate: float, num_steps: int, patience: int):
        """Train the network on the provided data (handles early stopping and restoring stop state)"""
        best_val_loss: float = float("inf")
        best_params = None
        wait = 0

        # Training loop
        for step in range(num_steps):
            y_hat = self.forward(data.x_train_tensor_norm)
            loss, _ = loss_fn_general(y_hat, data.y_train_tensor_norm, self.parameters(), lambda_l1=0.0, lambda_l2=1e-3)
            loss.backward()

            # Apply gradient descent to each parameter
            with torch.no_grad():
                for p in self.parameters():
                    p -= learning_rate * p.grad
                    p.grad.zero_()

            # Early stopping check
            if data.x_val_tensor_norm is not None:
                with torch.no_grad():
                    y_val_hat = self.forward(data.x_val_tensor_norm)
                    val_loss, _ = loss_fn_general(y_val_hat, data.y_val_tensor_norm, [], lambda_l1=0.0,
                                                  lambda_l2=1e-3)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        # Save best parameters
                        best_params = [p.clone().detach() for p in self.parameters()]
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"Early stopping at step {step}")
                            break

            # Restore the best parameters (if stopping early)
            if best_params is not None:
                with torch.no_grad():
                    for p, best_p in zip(self.parameters(), best_params):
                        p.copy_(best_p)


def parse_data(features, labels, is_classifier: bool, early_stopping_enabled: bool, test_set_ratio: float = 0.2,
               val_set_ratio: float = 0.5, split_random_seed=42):
    """Parse features and labels into training data, including training/testing split"""

    x_val, y_val = None, None
    # Split data into training/testing
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=test_set_ratio, random_state=split_random_seed
    )
    if early_stopping_enabled:
        x_test, x_val, y_test, y_val = train_test_split(
            x_temp, y_temp, test_size=val_set_ratio, random_state=split_random_seed
        )
    else:
        x_test, y_test = x_temp, y_temp

    # Build learning data and de-normalizing functions
    learning_data = LearningData.build(
        x_train=x_train,
        x_test=x_test,
        x_val=x_val,
        y_train=y_train,
        y_test=y_test,
        y_val=y_val,
        is_classifier=is_classifier
    )
    return learning_data


def y_hat_network(network, x, dim: int):
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device).view(-1, dim)
        y_hat = network.forward(x_t)
        return y_hat.cpu().numpy().squeeze()


def evaluate_network(network: Network, learning_data: LearningData, is_classifier: bool, decision_func: Callable):
    """Evaluate the results of a trained network (accuracy, loss, etc.)"""
    with torch.no_grad():
        # Re-scale for evaluation/display
        y_hat_train = network.forward(learning_data.x_train_tensor_norm)
        y_hat_test = network.forward(learning_data.x_test_tensor_norm)

        # Training loss
        train_losses, loss_label = loss_per_sample(y_hat_train, learning_data.y_train_tensor_norm)
        mean_train_loss = torch.mean(train_losses).item()
        print(f"Training {loss_label} (mean): {mean_train_loss}")

        # Test loss
        test_losses, loss_label = loss_per_sample(y_hat_test, learning_data.y_test_tensor_norm)
        mean_test_loss = torch.mean(test_losses).item()
        print(f"Test {loss_label} (mean): {mean_test_loss}")

        if is_classifier:
            # Test predictions (only for classification model)
            y_pred = decision_func(y_hat_test).int()
            accuracy = (y_pred == learning_data.y_test_tensor_norm.int()).float().mean().item()
            print(f"Test Accuracy: {round(accuracy * 100, 3)}%")
        else:
            # RMSE
            train_rmse = rmse(y_hat_train, learning_data.y_train_tensor_norm).item()
            test_rmse = rmse(y_hat_test, learning_data.y_test_tensor_norm).item()

            print(f"Training RMSE: {train_rmse}")
            print(f"Test RMSE: {test_rmse}")


def plot_results(network: Network, learning_data: LearningData, feature_count: int, is_classifier: bool,
                 decision_func: Callable):
    """Handles plotting for 1D or 2D input features."""
    training_data = np.hstack([learning_data.x_train_tensor_norm.cpu().numpy(),
                               learning_data.y_train_tensor_norm.cpu().numpy()])
    testing_data = np.hstack([learning_data.x_test_tensor_norm.cpu().numpy(),
                              learning_data.y_test_tensor_norm.cpu().numpy()])

    y_hat_func = lambda x: y_hat_network(network, x, feature_count)

    if feature_count == 1:
        plot_2d_new(training_points=training_data, test_points=testing_data,
                    y_hat_func=y_hat_func, is_classifier=is_classifier, decision_function=decision_func)
    elif feature_count == 2:
        plot_3d_new(training_points=training_data, test_points=testing_data,
                    y_hat_func=y_hat_func, is_classifier=is_classifier, decision_function=decision_func)


def main():
    # Training hyperparameters
    is_classifier = False
    num_steps = 20000
    learning_rate = 0.05
    feature_count = 1

    # Early stopping parameters
    early_stopping_enabled = True
    patience = 10

    # data = generate_dummy_data(n_features=feature_count, n_points=100, seed=1, pattern="linear",
    #                            task="classification" if is_classifier else "regression")
    data = generate_polynomial_data(n_features=feature_count, n_points=250, seed=None, degree=3,
                                    task="classification" if is_classifier else "regression")

    # Prepare the data
    features = np.array([p[:-1] for p in data], dtype=float)
    labels = np.array([p[-1] for p in data], dtype=float)
    learning_data = parse_data(features=features, labels=labels, is_classifier=is_classifier,
                               early_stopping_enabled=early_stopping_enabled)

    # Build the network
    # network = Network(node_counts=[feature_count, 1], activations=[sigmoid], initializations=[None])
    network = Network(node_counts=[feature_count, 8, 8, 1], activations=[relu, relu, None],
                      initializations=["kaiming", "kaiming", None])

    # Train the network
    network.train(learning_data, learning_rate, num_steps, patience)

    print("\nLearned parameters:")
    for i, layer in enumerate(network.layers):
        W = layer.W.detach().cpu().numpy()
        b = layer.b.detach().cpu().numpy()
        print(f"Layer {i}:")
        print(f"  W = {W.flatten()}")
        print(f"  b = {b.flatten()}")
    print()

    # Evaluate the network and display results
    decision_func = lambda x: x > 0.5
    evaluate_network(network=network, learning_data=learning_data, is_classifier=is_classifier,
                     decision_func=decision_func)
    plot_results(network=network, learning_data=learning_data, feature_count=feature_count, is_classifier=is_classifier,
                 decision_func=decision_func)


if __name__ == '__main__':
    main()
