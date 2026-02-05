import dataclasses
from typing import Callable, Self

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor

from math_definitions import rmse_per_sample, sigmoid, loss_fn_general, loss_per_sample, relu
from tools import generate_dummy_data, plot_2d_new, plot_3d_new, generate_polynomial_data

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


@dataclasses.dataclass
class LearningData:
    x_train_tensor_norm: Tensor
    x_test_tensor_norm: Tensor
    y_train_tensor_norm: Tensor
    y_train_tensor: Tensor
    y_test_tensor_norm: Tensor
    y_test_tensor: Tensor

    @classmethod
    def build(cls, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
              is_classifier: bool) -> tuple[Self, Callable]:
        # Normalize the training points (only based on the training data to avoid data leakage)
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_train_norm = (x_train - x_mean) / (x_std + 1e-8)
        x_test_norm = (x_test - x_mean) / (x_std + 1e-8)

        # For regression models, normalize target points (to avoid exploding outputs)
        if is_classifier:
            y_train_norm = y_train
            y_test_norm = y_test

            denorm_y = lambda y: y
        else:
            y_mean = y_train.mean()
            y_std = y_train.std()
            y_train_norm = (y_train - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std

            denorm_y = lambda y: y * (y_std + 1e-8) + y_mean

        # Convert to tensors for GPU acceleration
        x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32, device=device)
        x_test_tensor = torch.tensor(x_test_norm, dtype=torch.float32, device=device)
        y_train_tensor_norm = torch.tensor(y_train_norm, dtype=torch.float32, device=device).view(-1, 1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
        y_test_tensor_norm = torch.tensor(y_test_norm, dtype=torch.float32, device=device).view(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)

        return (
            LearningData(
                x_train_tensor_norm=x_train_tensor,
                x_test_tensor_norm=x_test_tensor,
                y_train_tensor_norm=y_train_tensor_norm,
                y_train_tensor=y_train_tensor,
                y_test_tensor_norm=y_test_tensor_norm,
                y_test_tensor=y_test_tensor,
            ),
            denorm_y,
        )


def parse_data(features, labels, is_classifier: bool):
    """Parse features and labels into training data, including training/testing split"""
    # Split data into training/testing
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Build learning data and de-normalizing functions
    learning_data, denorm_y = LearningData.build(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        is_classifier=is_classifier
    )
    return learning_data, denorm_y


def y_hat_network(network, x, dim: int):
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device).view(-1, dim)
        y_hat = network.forward(x_t)
        return y_hat.cpu().numpy().squeeze()


def main():
    is_classifier = False
    num_steps = 2000
    learning_rate = 0.01
    feature_count = 1
    # data = generate_dummy_data(n_features=feature_count, n_points=100, seed=1, pattern="linear",
    #                            task="classification" if is_classifier else "regression")
    data = generate_polynomial_data(n_features=feature_count, n_points=250, seed=None, degree=4, task="classification" if is_classifier else "regression")

    # Build the network
    # network = Network(node_counts=[feature_count, 1], activations=[sigmoid], initializations=[None])

    network = Network(node_counts=[feature_count, 8, 8, 1], activations=[relu, relu, None], initializations=["kaiming", "kaiming", None])

    # Function to make final classification decision
    def decision_func(x: float) -> bool:
        return x > 0.5

    # Prepare the data
    features = np.array([p[:-1] for p in data], dtype=float)
    labels = np.array([p[-1] for p in data], dtype=float)

    learning_data, denorm_y = parse_data(features=features, labels=labels, is_classifier=is_classifier)

    # Training loop
    for _ in range(num_steps):
        y_hat = network.forward(learning_data.x_train_tensor_norm)
        loss, _ = loss_fn_general(y_hat, learning_data.y_train_tensor_norm, network.parameters(), lambda_l1=0.0, lambda_l2=0.0)
        loss.backward()

        # Apply gradient descent to each parameter
        with torch.no_grad():
            for p in network.parameters():
                p -= learning_rate * p.grad
                p.grad.zero_()

    print("\nLearned parameters:")
    for i, layer in enumerate(network.layers):
        W = layer.W.detach().cpu().numpy()
        b = layer.b.detach().cpu().numpy()
        print(f"Layer {i}:")
        print(f"  W = {W.flatten()}")
        print(f"  b = {b.flatten()}")
    print()

    with torch.no_grad():
        # Re-scale for evaluation/display
        y_hat_train = denorm_y(network.forward(learning_data.x_train_tensor_norm))
        y_hat_test = denorm_y(network.forward(learning_data.x_test_tensor_norm))

        # Training loss
        train_losses, loss_label = loss_per_sample(y_hat_train, learning_data.y_train_tensor)
        mean_train_loss = torch.mean(train_losses).item()
        print(f"Training {loss_label} (mean): {mean_train_loss}")

        # Test loss
        test_losses, loss_label = loss_per_sample(y_hat_test, learning_data.y_test_tensor)
        mean_test_loss = torch.mean(test_losses).item()
        print(f"Test {loss_label} (mean): {mean_test_loss}")

        if is_classifier:
            # Test predictions (only for classification model)
            y_pred = decision_func(y_hat_test).int()
            accuracy = (y_pred == learning_data.y_test_tensor_norm.int()).float().mean().item()
            print(f"Test Accuracy: {round(accuracy * 100, 3)}%")
        else:
            # RMSE
            train_rmse = rmse_per_sample(y_hat_train, learning_data.y_train_tensor).item()
            test_rmse = rmse_per_sample(y_hat_test, learning_data.y_test_tensor).item()

            print(f"Training RMSE: {train_rmse}")
            print(f"Test RMSE: {test_rmse}")

    # Prepare plotting data
    training_data = np.hstack([learning_data.x_train_tensor_norm.cpu().numpy(), learning_data.y_train_tensor_norm.cpu().numpy()])
    testing_data = np.hstack([learning_data.x_test_tensor_norm.cpu().numpy(), learning_data.y_test_tensor_norm.cpu().numpy()])

    # If possible, plot the resulting model
    if feature_count == 1:
        plot_2d_new(training_points=training_data, test_points=testing_data,
                    y_hat_func=lambda x: y_hat_network(network, x, 1), is_classifier=is_classifier,
                    decision_function=decision_func)
    elif feature_count == 2:
        plot_3d_new(training_points=training_data, test_points=testing_data,
                    y_hat_func=lambda x: y_hat_network(network, x, 2), is_classifier=is_classifier,
                    decision_function=decision_func)


if __name__ == '__main__':
    main()
