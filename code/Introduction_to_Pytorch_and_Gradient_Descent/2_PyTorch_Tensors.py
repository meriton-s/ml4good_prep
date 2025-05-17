# =========================
# Imports
# =========================
import matplotlib.pyplot as plt
import torch


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
x = torch.linspace(-torch.pi, torch.pi, 1000)
true_y = torch.sin(x)

# =========================
# Model parameters (Initialization)
# =========================
a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

predicted_y = None

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
    predicted_y = a + b*x + c*x**2 + d*x**3

    # --- Backward ---
    e_i = true_y - predicted_y
    a_grad = -2 * e_i.mean()
    b_grad = -2*(x*e_i).mean()
    c_grad = -2*(x**2*e_i).mean()
    d_grad = -2*(x**3*e_i).mean()

    # --- Update ---
    a -= learning_rate * a_grad
    b -= learning_rate * b_grad
    c -= learning_rate * c_grad
    d -= learning_rate * d_grad

# =========================
# Visualization
# =========================
draw_plot(x, true_y, predicted_y, "true_y", "predicted_y")
