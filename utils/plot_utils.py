import matplotlib.pyplot as plt
import torch
import numpy as np
from .math_utils import rotation_matrix_to_euler_angles

def plot_samples(manifold, target, sample):
    if manifold.name == 'Torus':
        if sample.shape[-1] == 2:
            return plot_t2(target, sample)
        else:
            return plot_tn(target, sample)
    if manifold.name == 'Special Orthogonal':
        # if sample.shape[-1] == 3:
        #     sampled_arr = sample.detach().numpy()
        #     sample = np.array([rotation_matrix_to_euler_angles(R) for R in sampled_arr])
        #     target_arr = target.detach().numpy()
        #     target = np.array([rotation_matrix_to_euler_angles(R) for R in target_arr])
        #     return plot_so3(target, sample)
        return plot_son(target, sample)
    else:
        return plot_Un(target, sample)
    return

def plot_so3(x0, xt, size=5):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 8),
        sharex=False,
        sharey=False,
        tight_layout=True,
    )

    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        x = x.detach().cpu() if isinstance(x, torch.Tensor) else x
        axes[i].scatter(x[..., 0], x[..., 2], s=0.1)

    for ax in axes:
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_aspect("equal")
    plt.close(fig)
    return fig

def plot_son(x0, xt, size=5):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 8),
        sharex=False,
        sharey=False,
        tight_layout=True,
    )

    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        x = x.detach().cpu() if isinstance(x, torch.Tensor) else x
        axes[i].scatter(x[..., 0], x[..., 2], s=0.1)

    for ax in axes:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect("equal")
    plt.close(fig)
    return fig

def plot_Un(x0, x1, size=5):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(5, 5),
        sharex=False,
        sharey=False,
        tight_layout=True,
    )

    for i, x in enumerate([x0, x1]):
        if x is None:
            continue
        x = x.detach().cpu() if isinstance(x, torch.Tensor) else x
        axes[i].scatter(x[..., 1, 0].real, x[..., 1, 1].real, s=0.1)

    for ax in axes:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect("equal")
    plt.close(fig)
    return fig

def plot_t2(x0, xt, size=5):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=False,
        tight_layout=False,
    )

    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        x = x.detach().cpu() if isinstance(x, torch.Tensor) else x
        axes[i].scatter(x[..., 0], x[..., 1], s=0.1)

    for ax in axes:
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])
        ax.set_aspect("equal")
        
    plt.close(fig)
    return fig

def plot_tn(x0, xt, size=5):
    n = xt.shape[-1] if xt is not None else x0.shape[-1]
    d = n
    n = min(n, 3)

    fig, axes = plt.subplots(
        2,
        n-1,
        figsize=(0.6 * size, 0.6 * size * n / 2),
        sharex=False,
        sharey=False,
        tight_layout=True,
        squeeze=False,
    )
    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        for j in range(n - 1):
            x_ = x[..., j : j + 2]
            x_ = x_.detach().cpu() if isinstance(x_, torch.Tensor) else x_
            axes[i][j].scatter(x_[..., 0], x_[..., 1], s=0.1)

    axes = [item for sublist in axes for item in sublist]
    for ax in axes:
        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])
        ax.set_aspect("equal")
    plt.close(fig)
    return fig