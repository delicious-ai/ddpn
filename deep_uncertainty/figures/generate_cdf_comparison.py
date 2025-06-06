import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_uncertainty.datamodules import TabularDataModule
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import GaussianNN


def plot_discrete_cdf(
    support: np.ndarray, values: np.ndarray, color: str, label: str, ax: plt.Axes
):
    jump_indices = np.where(np.abs(values[:-1] - values[1:]) > 1e-5)[0] + 1

    for i in range(len(jump_indices) - 1):
        start = jump_indices[i]
        end = jump_indices[i + 1]
        ax.hlines(y=values[start], xmin=support[start], xmax=support[end], color=color)

    ax.hlines(y=values[0], xmin=support[0], xmax=support[jump_indices[0]], color=color)
    ax.hlines(
        y=values[-1], xmin=support[jump_indices[-1]], xmax=support[-1], color=color, label=label
    )
    ax.scatter(support[jump_indices[:-1]], values[jump_indices[:-1]], color=color, s=20, zorder=2)
    ax.scatter(
        support[jump_indices[:-1]],
        values[jump_indices[:-1] - 1],
        color="white",
        s=10,
        zorder=3,
        edgecolor=color,
    )
    ax.scatter(0, 0, color=color, s=20, zorder=2)


if __name__ == "__main__":

    dm = TabularDataModule(
        "data/length-of-stay/length_of_stay.npz",
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    dm.setup("test")
    loader = dm.test_dataloader()

    inputs = []
    targets = []
    for (x, y) in loader:
        inputs.append(x)
        targets.append(y)

    gaussian = GaussianNN.load_from_checkpoint(
        "chkp/length-of-stay/seitzer_0.5/version_0/best_loss.ckpt"
    )
    ddpn = DoublePoissonNN.load_from_checkpoint(
        "chkp/length-of-stay/beta_ddpn_0.5/version_0/best_loss.ckpt"
    )

    fig1, ax1 = plt.subplots(figsize=(3, 3))
    x = inputs[6]
    y = targets[6]
    gauss_dist = gaussian.predictive_dist(gaussian.predict(x))
    ddpn_dist = ddpn.predictive_dist(ddpn.predict(x))
    max_val = (
        max(ddpn_dist.ppf(0.999), gauss_dist.icdf(torch.tensor(0.999)).round().long().item()) + 2
    )
    support = torch.linspace(0, max_val, steps=250)
    disc_support = torch.arange(0, max_val + 1)
    gauss_cdf_vals = gauss_dist.cdf(support)
    ddpn_cdf_vals = ddpn_dist.cdf(disc_support)

    ax1.plot(support, gauss_cdf_vals.detach(), color="tab:blue", label=r"Gaussian", zorder=-1)
    plot_discrete_cdf(
        disc_support, ddpn_cdf_vals.detach(), color="tab:green", ax=ax1, label=r"DDPN"
    )
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.vlines(y, -0.1, 1.1, color="black", linewidth=1, zorder=-3, linestyle="dashed", alpha=0.8)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    fig1.tight_layout()
    fig1.savefig("deep_uncertainty/figures/artifacts/pathologies_1.pdf", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(3, 3))
    x = inputs[47]
    y = targets[47]
    gauss_dist = gaussian.predictive_dist(gaussian.predict(x))
    ddpn_dist = ddpn.predictive_dist(ddpn.predict(x))
    max_val = (
        max(ddpn_dist.ppf(0.999), gauss_dist.icdf(torch.tensor(0.999)).round().long().item()) + 2
    )
    support = torch.linspace(0, max_val, steps=250)
    disc_support = torch.arange(0, max_val + 1)
    gauss_cdf_vals = gauss_dist.cdf(support)
    ddpn_cdf_vals = ddpn_dist.cdf(disc_support)

    ax2.plot(support, gauss_cdf_vals.detach(), color="tab:blue", label=r"Gaussian", zorder=-1)
    plot_discrete_cdf(
        disc_support, ddpn_cdf_vals.detach(), color="tab:green", ax=ax2, label=r"DDPN"
    )
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.vlines(y, -0.1, 1.1, color="black", linewidth=1, zorder=-3, linestyle="dashed", alpha=0.8)
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    fig2.savefig("deep_uncertainty/figures/artifacts/pathologies_2.pdf", dpi=150)
