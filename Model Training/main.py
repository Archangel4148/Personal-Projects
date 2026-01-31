import sympy as sp

# Hyperparameters
FEATURE_COUNT = 2
LEARNING_RATE = 1
DATA_POINTS = [
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [2, 1, 1],
]

# Define symbols for equations
x = sp.symbols(f"x0:{FEATURE_COUNT}")
y = sp.symbols("y")
w = sp.symbols(f"w0:{FEATURE_COUNT}")
b = sp.symbols("b")


# Main model (linear combination of features * weights, plus bias)
z = sum(w[i] * x[i] for i in range(FEATURE_COUNT)) + b

# Activation function (sigmoid)
y_hat = 1 / (1 + sp.exp(-z))

# Loss function
L = 0.5 * (y - y_hat)**2  # Mean squared error
# L = (y - y_hat)**2  # Weird other error I did in my notebook
# L = - (y * sp.log(y_hat) + (1 - y) * sp.log(1 - y_hat))  # Binary cross-entropy

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
grad_funcs = {p: sp.lambdify(tuple(x) + (y,) + tuple(w) + (b,), get_gradient(L, p),"numpy") for p in parameters}

# Accumulate the values with each step
total_params = {p: 0.0 for p in parameters}

NUM_STEPS = 2
for step in range(NUM_STEPS):
    # Reset gradient count every step
    total_grads = {p: 0.0 for p in parameters}

    print(f"=== Step {step+1} ===")
    # Loop through and accumulate gradients for each point
    for point in DATA_POINTS:
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
        total_params[p] -= LEARNING_RATE * total_grads[p]

    print("Gradients:")
    for k, v in total_grads.items():
        print(k, ":", round(v, 10))

    print("Parameters:")
    for k, v in total_params.items():
        print(k, ":", round(v, 10))
    print()
