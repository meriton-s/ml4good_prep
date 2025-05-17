# =========================
# Imports
# =========================
import torch
import funcs


# =========================
# Functions
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("Using", device)

# =========================
# Dataset generation
# =========================
points = torch.linspace(-torch.pi, torch.pi, 1000)
powers = torch.tensor([1, 2, 3])

x = points.unsqueeze(-1).pow(powers)

true_y = torch.sin(points)

# =========================
# Model parameters (Initialization)
# =========================

model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

loss_fn = torch.nn.MSELoss(reduction="mean")

# =========================
# Hyperparameters (Constants)
# =========================

learning_rate = 1e-3
epochs = 3000

# =========================
# Learning cycle
# =========================
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for i in range(epochs):
    # --- Forward ---
    predicted_y = model(x)
    loss = loss_fn(predicted_y, true_y)
    optimizer.zero_grad()
    # --- Backward ---
    loss.backward()
    # --- Update ---
    optimizer.step()
# =========================
# Evaluation
# =========================
linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
# =========================
# Visualization
# =========================

funcs.draw_plot(points, predicted_y.detach(), true_y, "predicted_y", "true_y")
