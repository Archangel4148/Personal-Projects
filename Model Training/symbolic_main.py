import numpy as np
import sympy as sp
from sklearn.model_selection import train_test_split

from tools import generate_dummy_data, compute_dataset_loss, plot_2d, plot_3d

VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# Hyperparameters
LEARNING_RATE = 0.05
NUM_STEPS = 100
DATA_POINTS = generate_dummy_data(n_features=1, n_points=100, seed=1, pattern="linear", task="classification")


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
    training_data = [list(f) + [y] for f, y in zip(X_train_norm, y_train)]

    # Define symbols for equations
    x = sp.symbols(f"x0:{feature_count}")
    y = sp.symbols("y")
    w = sp.symbols(f"w0:{feature_count}")
    b = sp.symbols("b")

    # Main model (linear combination of features * weights, plus bias)
    z = sum(w[i] * x[i] for i in range(feature_count)) + b

    # Activation function (sigmoid)
    y_hat = 1 / (1 + sp.exp(-z))

    # Loss function
    # L = 0.5 * (y - y_hat) ** 2  # Mean squared error
    # L = (y - y_hat)**2  # Weird other error I did in my notebook
    L = - (y * sp.log(y_hat) + (1 - y) * sp.log(1 - y_hat))  # Binary cross-entropy (Log Loss)

    loss_func = sp.lambdify(tuple(x) + (y,) + tuple(w) + (b,), L, "numpy")

    def get_gradient(loss, parameter):
        # Calculate the gradient from provided loss and parameter
        return sp.simplify(sp.factor(sp.diff(loss, parameter)))

    # Calculate the gradients for all weights and bias
    parameters = list(w) + [b]
    print("Gradient formulas:")
    for p in parameters:
        print(p, ":", get_gradient(L, p))
    print()

    # Convert the gradient formulas to callable functions
    grad_funcs = {p: sp.lambdify(tuple(x) + (y,) + tuple(w) + (b,), get_gradient(L, p), "numpy") for p in parameters}

    # Accumulate the values with each step (start with a small random value to break up linearity)
    total_params = {p: np.random.uniform(-0.01, 0.01) for p in parameters}

    for step in range(NUM_STEPS):
        # Reset gradient count every step
        total_grads = {p: 0.0 for p in parameters}

        vprint(f"=== Step {step + 1} ===")
        # Loop through and accumulate gradients for each point
        for point in training_data:
            x_vals = point[:-1]
            y_val = point[-1]
            for p in parameters:
                # Pass all existing parameters into the gradient function for each parameter
                grad_args = tuple(x_vals) + (y_val,) + tuple(total_params[w_i] for w_i in w) + (total_params[b],)
                grad = grad_funcs[p](*grad_args)
                # Update the running total gradient
                total_grads[p] += grad

        # Take a learning step (down the gradient)
        for p in parameters:
            total_params[p] -= LEARNING_RATE * (total_grads[p] / len(training_data))

        vprint("Gradients:")
        for k, v in total_grads.items():
            vprint(k, ":", round(v, 10))

        vprint("Parameters:")
        for k, v in total_params.items():
            vprint(k, ":", round(v, 10))
        vprint()

    # Convert final trained model to a callable function
    y_hat_func = sp.lambdify(tuple(x) + tuple(w) + (b,), y_hat, "numpy")

    w_learned = [total_params[vw] for vw in w]
    b_learned = total_params[b]

    print(f"Final Learned Parameters:")
    for i, w in enumerate(w_learned):
        print(f"w{i}: {w}")
    print("b:", b_learned, end="\n\n")

    # Evaluate accuracy of predictions using the test data set
    y_hat_vals = np.array([y_hat_func(*x_row, *w_learned, b_learned) for x_row in X_test_norm])
    y_pred = (y_hat_vals > 0.5).astype(int)
    accuracy = (y_pred == y_test).mean()
    print(f"Test Accuracy: {round(accuracy * 100, 3)}%")

    # Calculate the training loss
    training_loss = compute_dataset_loss(training_data, w_learned, b_learned, loss_func)
    print("Training Loss:", training_loss)

    # Calculate the test loss
    test_set = list(zip(X_test_norm, y_test))
    test_loss = compute_dataset_loss(test_set, w_learned, b_learned, loss_func)
    print("Test Loss:", test_loss)

    # If possible, plot the resulting model
    if feature_count == 1:
        plot_2d(training_points=training_data, w_val=w_learned, b_val=b_learned, y_hat_func=y_hat_func)
    elif feature_count == 2:
        plot_3d(data_points=training_data, w_vals=w_learned, b_val=b_learned, y_hat_func=y_hat_func)


if __name__ == '__main__':
    main()
