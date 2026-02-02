import numpy as np


def generate_dummy_data(n_features=1, n_points=50, pattern="linear", seed=None):
    if seed is not None:
        np.random.seed(seed)

    data_points = []

    if pattern == "linear":
        # Linear separation: random hyperplane
        w = np.random.uniform(-1, 1, size=n_features)
        b = np.random.uniform(-0.5, 0.5)
        for _ in range(n_points):
            x = np.random.uniform(-5, 5, size=n_features)
            y = int(np.dot(w, x) + b > 0)
            data_points.append(list(x) + [y])

    elif pattern == "clusters":
        # Gaussian clusters for each class
        n_clusters_per_class = 2
        for class_label in [0, 1]:
            for _ in range(n_clusters_per_class):
                # Random cluster center
                center = np.random.uniform(-3, 3, size=n_features)
                # Generate points around center
                for _ in range(n_points // (2 * n_clusters_per_class)):
                    x = center + np.random.normal(0, 0.5, size=n_features)
                    data_points.append(list(x) + [class_label])
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    np.random.shuffle(data_points)
    return data_points


def plot_2d(training_points, w_val, b_val, y_hat_func, test_points=None):
    import matplotlib.pyplot as plt

    # Generate smooth x range
    x_plot = np.linspace(
        min(p[0] for p in training_points) - 0.5,
        max(p[0] for p in training_points) + 0.5,
        200
    )

    y_plot = y_hat_func(x_plot, w_val[0], b_val)

    # Plot training data points
    for px, py in training_points:
        color = "orange" if py > 0.5 else "b"
        plt.scatter(px, py, c=color, zorder=3)

    # Plot test points (if any)
    if test_points:
        for tx, ty in test_points:
            y_pred = y_hat_func(np.array([tx]), w_val[0], b_val)[0]
            if (y_pred > 0.5) == bool(ty):
                marker = "^"  # green triangle for correct
                color = "g"
            else:
                marker = "x"  # red x for wrong
                color = "r"
            plt.scatter(tx, ty, c=color, marker=marker, s=80, zorder=4, label="_nolegend_")

    # Plot learned curve
    plt.plot(x_plot, y_plot, label="Learned model", zorder=2)

    # Plot decision boundary (where y_hat = 0.5)
    if w_val[0] != 0:
        x_boundary = -b_val / w_val[0]
        plt.axvline(x=x_boundary, color="k", linestyle="--", label="Decision boundary")

    # Placeholder scatter points to add more to the legend
    plt.scatter([], [], c='b', label='Training Class 0')
    plt.scatter([], [], c='orange', label='Training Class 1')
    if test_points:
        plt.scatter([], [], c='g', marker='^', s=80, label='Test Correct')
        plt.scatter([], [], c='r', marker='x', s=80, label='Test Incorrect')

    # Set up plot
    plt.xlabel("x")
    plt.ylabel("ŷ")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3d(data_points, w_vals, b_val, y_hat_func):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter training points
    for p in data_points:
        x1, x2, y_val = p
        color = "r" if y_val > 0.5 else "b"
        ax.scatter(x1, x2, y_val, c=color)

    # Create input grid
    x1_vals = np.linspace(
        min(p[0] for p in data_points) - 0.5,
        max(p[0] for p in data_points) + 0.5,
        40
    )
    x2_vals = np.linspace(
        min(p[1] for p in data_points) - 0.5,
        max(p[1] for p in data_points) + 0.5,
        40
    )

    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Evaluate model on grid
    Y_hat = y_hat_func(X1, X2, *w_vals, b_val)

    # Plot learned surface
    ax.plot_surface(X1, X2, Y_hat, alpha=0.5)

    ax.contour(X1, X2, Y_hat, levels=[0.5], colors='k')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("ŷ")

    plt.show()

def compute_dataset_loss(dataset, w_vals, b_val, loss_func):
    # Apply the loss function to the whole dataset using the provided parameters
    total_loss = 0.0
    for point in dataset:
        x_vals = np.array(point[:-1]).flatten()
        y_val = point[-1]
        args = tuple(x_vals) + (y_val,) + tuple(w_vals) + (b_val,)
        total_loss += loss_func(*args)
    return total_loss / len(dataset)