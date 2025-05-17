# =========================
# Imports
# =========================
import funcs
import torch

# =========================
# Functions
# =========================


# =========================
# Dataset generation
# =========================
x = torch.linspace(-torch.pi, torch.pi, 1000)
true_y = torch.sin(x)

powers = torch.tensor([1, 2, 3])
powers_of_x = x.unsqueeze(-1).pow(powers)
# =========================
# Model parameters (Initialization)
# =========================
model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(start_dim=0, end_dim=1),)

loss_fn = torch.nn.MSELoss(reduction="mean")

# =========================
# Hyperparameters (Constants)
# =========================

learning_rate = 1e-3
epochs = 2000

# =========================
# Learning cycle
# =========================
for t in range(epochs):
    # --- Forward ---
    predicted_y = model(powers_of_x)
    loss = loss_fn(predicted_y, true_y)

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # --- Backward ---
    loss.backward()

    # --- Update ---
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# =========================
# Evaluation
# =========================

# =========================
# Visualization
# =========================
funcs.draw_plot(x, predicted_y.detach(), true_y, "predicted", "true")
