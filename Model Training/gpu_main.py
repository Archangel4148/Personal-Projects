import numpy as np
import torch
from sklearn.model_selection import train_test_split

from math_definitions import model, loss_fn, loss_per_sample
from tools import generate_dummy_data, plot_2d, plot_3d

VERBOSE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# Hyperparameters
LEARNING_RATE = 0.05
NUM_STEPS = 100
DATA_POINTS = generate_dummy_data(n_features=2, n_points=100, seed=1, pattern="linear", task="classification")

def y_hat_func(*args) -> np.ndarray:
    """Wrapper for y-hat calculation to match format of plotting functions"""
    num_weights = (len(args) - 1) // 2
    b_val = args[-1]  # Scalar float
    w_vals = torch.tensor([[v] for v in args[-(num_weights + 1):-1]], dtype=torch.float32, device=device)  # Scalar float32
    x_vals = args[:-(num_weights + 1)]  # numpy array per feature

    x_stack = torch.tensor(np.stack(x_vals, axis=-1), dtype=torch.float32, device=device)
    params = (w_vals, torch.tensor(b_val, dtype=torch.float32, device=device))
    # Output as a numpy array
    y_hat = model(params, x_stack).cpu().numpy().squeeze(-1)
    return y_hat


def main():
    feature_count = len(DATA_POINTS[0]) - 1

    features = np.array([p[:-1] for p in DATA_POINTS], dtype=float)
    labels = np.array([p[-1] for p in DATA_POINTS], dtype=float)

    # Split the provided data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Normalize the data points (only based on the training data to avoid data leakage)
    mean_vals = X_train.mean(axis=0)
    std_vals = X_train.std(axis=0)
    X_train_norm = (X_train - mean_vals) / (std_vals + 1e-8)
    X_test_norm = (X_test - mean_vals) / (std_vals + 1e-8)

    # Convert data to GPU-ready tensors
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    # Initialize parameters
    w = torch.randn(feature_count, device=device, dtype=torch.float32) * 0.01  # Small random offset to break up linearity
    b = torch.tensor(0.0, device=device, dtype=torch.float32)
    # Enable gradient calculations for parameter tensors
    w.requires_grad_(True)
    b.requires_grad_(True)
    params = (w, b)

    for step in range(NUM_STEPS):
        # Backward pass
        loss = loss_fn(params, X_train_tensor, y_train_tensor, lambda_l1=0.0, lambda_l2=0.0)
        loss.backward()

        # Gradient descent step
        with torch.no_grad():
            w -= LEARNING_RATE * w.grad
            b -= LEARNING_RATE * b.grad
            w.grad.zero_()
            b.grad.zero_()

    # Get the data ready for evaluation/plotting
    training_data = [list(x_row) + [y] for x_row, y in zip(X_train_norm, y_train)]
    testing_data = [list(x_row) + [y] for x_row, y in zip(X_test_norm, y_test)]
    w_learned = w.cpu().detach().numpy()
    b_learned = b.item()

    print(f"Final Learned Parameters:")
    for i, w in enumerate(w_learned):
        print(f"w{i}: {w}")
    print("b:", b_learned, end="\n\n")

    # Evaluate accuracy of predictions using the test data set
    y_hat_test = model(params, X_test_tensor).cpu().detach().numpy()
    y_pred = (np.array(y_hat_test) > 0.5).astype(int)
    accuracy = (y_pred == y_test).mean()
    print(f"Test Accuracy: {round(accuracy * 100, 3)}%")

    # Calculate the training loss
    y_hat_train = model(params, X_train_tensor)
    train_losses = loss_per_sample(y_hat_train, y_train_tensor)
    mean_train_loss = torch.mean(train_losses).item()
    print("Training Loss (mean):", mean_train_loss)
    vprint("Training Loss Per Sample:\n", train_losses.cpu().detach().numpy())

    # Calculate the test loss
    test_losses = loss_per_sample(model(params, X_test_tensor), y_test_tensor)
    mean_test_loss = torch.mean(test_losses).item()
    print("Test Loss (mean):", mean_test_loss)
    vprint("Test Loss Per Sample:\n", test_losses.cpu().detach().numpy())

    # If possible, plot the resulting model
    if feature_count == 1:
        plot_2d(training_points=training_data, w_val=w_learned, b_val=b_learned, y_hat_func=y_hat_func, test_points=testing_data)
    elif feature_count == 2:
        plot_3d(data_points=training_data, w_vals=w_learned, b_val=b_learned, y_hat_func=y_hat_func)


if __name__ == '__main__':
    main()
