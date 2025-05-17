# =========================
# Imports
# =========================
import matplotlib.pyplot as plt
import torch
import funcs

# =========================
# Functions
# =========================


# =========================
# Dataset generation
# =========================
x = torch.linspace(-torch.pi, torch.pi, 1000)
true_y = torch.sin(x)

# =========================
# Model parameters (Initialization)
# =========================
a = torch.rand((), requires_grad=True)
b = torch.rand((), requires_grad=True)
c = torch.rand((), requires_grad=True)
d = torch.rand((), requires_grad=True)

predicted_y = None

# =========================
# Hyperparameters (Constants)
# =========================
epochs = 2000
learning_rate = 1e-3

# =========================
# Learning cycle
# =========================
for i in range(epochs):
    # --- Forward ---
    predicted_y = a + b*x + c*x**2 + d*x**3

    e_i = true_y - predicted_y
    loss = e_i.pow(2).mean()

    # --- Backward ---
    loss.backward()

    # --- Update ---
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
# =========================
# Evaluation
# =========================

# =========================
# Visualization
# =========================
funcs.draw_plot(x, predicted_y.detach(), true_y, "predicted", "sin(x)")
