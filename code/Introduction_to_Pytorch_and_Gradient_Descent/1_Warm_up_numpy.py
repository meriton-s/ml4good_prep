# =========================
# Imports
# =========================
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Functions
# =========================
def draw_plot(arg_x, arg_y1, arg_y2, label_y1="y1", label_y2="y2"):
    plt.plot(arg_x, arg_y1, label=label_y1)
    plt.plot(arg_x, arg_y2, label=label_y2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# =========================
# Dataset generation
# =========================
x = np.linspace(-np.pi, np.pi, 1000)
true_y = np.sin(x)

# =========================
# Model parameters (Initialization)
# =========================
a = np.random.rand()
b = np.random.rand()
c = np.random.rand()
d = np.random.rand()
predicted_y: np.ndarray = np.zeros_like(x)

# =========================
# Hyperparameters (Constants)
# =========================
epochs = 3000
learning_rate = 1e-3

# =========================
# Learning cycle
# =========================
for i in range(epochs):
    # --- Forward ---
    predicted_y = a + b * x + c * x ** 2 + d * x ** 3

    # --- Backward ---
    e_i = true_y - predicted_y
    a_grad = -2 * e_i.mean()
    b_grad = -2 * (e_i * x).mean()
    c_grad = -2 * (e_i * x**2).mean()
    d_grad = -2 * (e_i * x**3).mean()

    # --- Update ---
    a -= a_grad * learning_rate
    b -= b_grad * learning_rate
    c -= c_grad * learning_rate
    d -= d_grad * learning_rate

# =========================
# Visualization
# =========================
draw_plot(x, true_y, predicted_y, "true_y", "predicted_y")


