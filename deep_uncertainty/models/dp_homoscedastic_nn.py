from functools import partial
from typing import Type

import torch
from torch import nn
from torchmetrics import Metric

from deep_uncertainty.enums import BetaSchedulerType
from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.custom_torchmetrics import ContinuousRankedProbabilityScore
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.random_variables import DoublePoisson
from deep_uncertainty.training.beta_schedulers import CosineAnnealingBetaScheduler
from deep_uncertainty.training.beta_schedulers import LinearBetaScheduler
from deep_uncertainty.training.losses import double_poisson_nll


class DoublePoissonHomoscedasticNN(DiscreteRegressionNN):
    """A neural network that learns the parameters of a Double Poisson distribution over each regression target (conditioned on the input).

    A single value of phi is learned over the dataset, as opposed to the heteroscedastic approach.

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing". Defaults to None.
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}. Defaults to None.
        beta_scheduler_type (BetaSchedulerType | None, optional): If specified, the type of beta scheduler to use for training loss (if applicable). Defaults to None.
        beta_scheduler_kwargs (dict | None, optional): If specified, key-value argument specifications for the chosen beta scheduler, e.g. {"beta_0": 1.0, "beta_1": 0.5}. Defaults to None.
    """

    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
        beta_scheduler_type: BetaSchedulerType | None = None,
        beta_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a DoublePoissonHomoscedasticNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
            beta_scheduler_type (BetaSchedulerType | None, optional): If specified, the type of beta scheduler to use for training loss (if applicable). Defaults to None.
            beta_scheduler_kwargs (dict | None, optional): If specified, key-value argument specifications for the chosen beta scheduler, e.g. {"beta_0": 1.0, "beta_1": 0.5}. Defaults to None.
        """
        if beta_scheduler_type == BetaSchedulerType.COSINE_ANNEALING:
            self.beta_scheduler = CosineAnnealingBetaScheduler(**beta_scheduler_kwargs)
        elif beta_scheduler_type == BetaSchedulerType.LINEAR:
            self.beta_scheduler = LinearBetaScheduler(**beta_scheduler_kwargs)
        else:
            self.beta_scheduler = None

        super().__init__(
            loss_fn=partial(
                double_poisson_nll,
                beta=(
                    self.beta_scheduler.current_value if self.beta_scheduler is not None else None
                ),
            ),
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Linear(self.backbone.output_dim, 1)
        self.logphi = nn.Parameter(torch.randn(1))

        self.mp = MedianPrecision()
        self.crps = ContinuousRankedProbabilityScore(mode="discrete")

        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (logmu, logphi), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        logmu = self.head(h)
        logphi = self.logphi.expand_as(logmu)
        y_hat = torch.cat((logmu, logphi), dim=-1)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, phi), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)  # Interpreted as (logmu, logphi)
        self.backbone.train()

        return torch.exp(y_hat)

    def _point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        dist: DoublePoisson = self.predictive_dist(y_hat, training=training)
        mode = torch.argmax(dist.pmf_vals, axis=0)
        return mode

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "mp": self.mp,
            "crps": self.crps,
        }

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        dist: DoublePoisson = self.predictive_dist(y_hat, training=False)
        mu, phi = dist.mu, dist.phi
        precision = phi / mu
        targets = y.flatten()
        support = torch.arange(2000, device=y_hat.device).view(-1, 1)
        probs_over_support = dist.pmf(support).T

        self.mp.update(precision)
        self.crps.update(probs_over_support, targets)

    def _predictive_dist_impl(self, y_hat: torch.Tensor, training: bool = False) -> DoublePoisson:
        output = y_hat.exp() if training else y_hat
        mu, phi = torch.split(output, [1, 1], dim=-1)
        mu = mu.flatten()
        phi = phi.flatten()
        dist = DoublePoisson(mu, phi)
        return dist

    def on_train_epoch_end(self):
        if self.beta_scheduler is not None:
            self.beta_scheduler.step()
            self.loss_fn = partial(double_poisson_nll, beta=self.beta_scheduler.current_value)
        super().on_train_epoch_end()
