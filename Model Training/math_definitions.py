import torch

# # Main model (linear combination of features * weights, plus bias)
# z = sum(w[i] * x[i] for i in range(feature_count)) + b
#
# # Activation function (sigmoid)
# y_hat = 1 / (1 + sp.exp(-z))
#
# # Loss function
# # L = 0.5 * (y - y_hat) ** 2  # Mean squared error
# # L = (y - y_hat)**2  # Weird other error I did in my notebook
# L = - (y * sp.log(y_hat) + (1 - y) * sp.log(1 - y_hat))  # Binary cross-entropy (Log Loss)

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def model(params, x):
    w, b = params
    z = torch.matmul(x, w) + b
    return sigmoid(z)


def binary_cross_entropy_per_sample(y_hat, y):
    eps = 1e-8
    return -(
            y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps)
    )


def loss_per_sample(y_hat, y):
    return binary_cross_entropy_per_sample(y_hat, y)


def loss_fn(params, x, y, lambda_l1=0.00, lambda_l2=0.00):
    w, _ = params
    y_hat = model(params, x)
    loss = torch.mean(loss_per_sample(y_hat, y))
    # Add L2 (Ridge) regularization term
    l2_loss = lambda_l2 * torch.sum(w ** 2)
    # Add L1 (Lasso) regularization term
    l1_loss = lambda_l1 * torch.sum(torch.abs(w))
    return loss + l1_loss + l2_loss
