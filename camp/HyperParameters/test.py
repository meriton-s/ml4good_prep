# =========================
# Imports
# =========================
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from typing import Literal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("Using", device)


# =========================
# Functions
# =========================


def spiral(phi):
    x = (phi + 1) * torch.cos(phi)
    y = phi * torch.sin(phi)
    return torch.cat((x, y), dim=1)


def generate_data(num_data):
    angles = torch.empty((num_data, 1)).uniform_(1, 15)
    data = spiral(angles)
    # Add some noise to the data.
    data += torch.empty((num_data, 2)).normal_(0.0, 0.4)
    labels = torch.zeros((num_data,), dtype=torch.int)
    # Flip half of the points to create two classes.
    data[num_data // 2 :, :] *= -1
    labels[num_data // 2 :] = 1
    return data, labels


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().squeeze()
    return t


def plot_data(arg_x, arg_y):
    """Plot data points x with labels y. Label 1 is a red +, label 0 is a blue +."""
    x = to_numpy(arg_x)
    y = to_numpy(arg_y)

    plt.figure(figsize=(5, 5))
    plt.plot(x[y == 1, 0], x[y == 1, 1], "r+")
    plt.plot(x[y == 0, 0], x[y == 0, 1], "b+")
    plt.show()


# =========================
# Dataset generation
# =========================
x_train, y_train = generate_data(4000)

training_set = TensorDataset(x_train, y_train)

# =========================
# Model parameters (Initialization)
# =========================

class Model(nn.Module):
    """
    A fully connected neural network with any number of layers.
    """

    NAME_TO_NONLINEARITY = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        layer_sizes: list[int] = [2, 8, 8, 8, 1],
        non_linearity: Literal["relu", "sigmoid", "tanh"] = "relu"
    ):
        super(Model, self).__init__()

        modules = []
        for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            modules.append(nn.Linear(input_dim, output_dim))
            # After each linear layer, apply a non-linearity.
            modules.append(self.NAME_TO_NONLINEARITY[non_linearity]())

        # Remove the last non-linearity, since the last layer is the output layer.
        self.layers = nn.Sequential(*modules[:-1])

    def forward(self, inputs):
        outputs = self.layers(inputs)
        # We want the model to predict 0 for one class and 1 for the other class.
        # A sigmoid function maps the output from [-inf, inf] to [0, 1].
        prediction = torch.sigmoid(outputs)
        return prediction


# =========================
# Hyperparameters (Constants)
# =========================

# =========================
# Learning cycle
# =========================

# --- Forward ---
# --- Backward ---
# --- Update ---

# =========================
# Evaluation
# =========================

# =========================
# Visualization
# =========================

# Visualize the data.
# plot_data(x_train, y_train)
print('done')
