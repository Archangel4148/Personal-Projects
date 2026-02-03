from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from math_definitions import sigmoid, loss_fn_general, loss_per_sample
from tools import generate_dummy_data, plot_2d_new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class Layer:
    def __init__(self, in_features: int, out_features: int, activation: Callable | None, device=None):
        device = device or torch.device("cpu")
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
    def __init__(self, layers: list[Layer]):
        self.layers = layers

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


def y_hat_network(network, x):
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        y_hat = network.forward(x_t)
        return y_hat.cpu().numpy().squeeze()


def main():
    num_steps = 100
    learning_rate = 0.05
    feature_count = 1
    data = generate_dummy_data(n_features=feature_count, n_points=100, seed=None, pattern="linear", task="classification")

    # Build the network
    is_classifier = True
    layers = [Layer(in_features=feature_count, out_features=1, activation=sigmoid)]  # Logistic regression
    # layers = [Layer(in_features=feature_count, out_features=1, activation=None)]  # Simple linear model
    network = Network(layers)

    # Function to make final classification decision
    def decision_func(x: float) -> bool:
        return x > 0.5

    # Prepare the data
    features = np.array([p[:-1] for p in data], dtype=float)
    labels = np.array([p[-1] for p in data], dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    # Normalize the data points (only based on the training data to avoid data leakage)
    mean_vals = X_train.mean(axis=0)
    std_vals = X_train.std(axis=0)
    X_train_norm = (X_train - mean_vals) / (std_vals + 1e-8)
    X_test_norm = (X_test - mean_vals) / (std_vals + 1e-8)

    # Convert to tensors for GPU acceleration
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)

    # Training loop
    for _ in range(num_steps):
        y_hat = network.forward(X_train_tensor)
        loss = loss_fn_general(y_hat, y_train_tensor, network.parameters(), lambda_l1=0.0, lambda_l2=0.0)
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

    with torch.no_grad():
        y_hat_test = network.forward(X_test_tensor)
        if is_classifier:
            # Test predictions (only for classification model)
            y_pred = decision_func(y_hat_test).int()
            accuracy = (y_pred == y_test_tensor.int()).float().mean().item()
            print(f"Test Accuracy: {round(accuracy * 100, 3)}%")

        # Training loss
        y_hat_train = network.forward(X_train_tensor)
        train_losses = loss_per_sample(y_hat_train, y_train_tensor)
        mean_train_loss = torch.mean(train_losses).item()
        print("Training Loss (mean):", mean_train_loss)

        # Test loss
        test_losses = loss_per_sample(y_hat_test, y_test_tensor)
        mean_test_loss = torch.mean(test_losses).item()
        print("Test Loss (mean):", mean_test_loss)

    # Prepare plotting data
    training_data = np.hstack([X_train_tensor.cpu().numpy(), y_train_tensor.cpu().numpy()])
    testing_data = np.hstack([X_test_tensor.cpu().numpy(), y_test_tensor.cpu().numpy()])

    # If possible, plot the resulting model
    if feature_count == 1:
        plot_2d_new(training_points=training_data, test_points=testing_data,
                    y_hat_func=lambda x: y_hat_network(network, x), is_classifier=is_classifier, decision_function=decision_func)
    # elif feature_count == 2:
    #     plot_3d_new(data_points=training_data, y_hat_func=lambda x: y_hat_network(network, x))


if __name__ == '__main__':
    main()
