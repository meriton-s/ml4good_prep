import matplotlib.pyplot as plt
import torch


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().squeeze()
    return t


def draw_plot(arg_x, arg_y1, arg_y2, label_y1="y1", label_y2="y2"):

    x = to_numpy(arg_x)
    y1 = to_numpy(arg_y1)
    y2 = to_numpy(arg_y2)

    plt.plot(x, y1, label=label_y1)
    plt.plot(x, y2, label=label_y2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


