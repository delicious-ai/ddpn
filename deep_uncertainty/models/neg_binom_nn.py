from typing import Type

import torch
from torch import nn
from torchmetrics import Metric

from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.evaluation.custom_torchmetrics import ContinuousRankedProbabilityScore
from deep_uncertainty.evaluation.custom_torchmetrics import MedianPrecision
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.training.losses import neg_binom_nll


class NegBinomNN(DiscreteRegressionNN):
    """A neural network that learns the parameters of a Negative Binomial distribution over each regression target (conditioned on the input).

    The mean-scale (mu, alpha) parametrization of the Negative Binomial is used for this network.

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing". Defaults to None.
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}. Defaults to None.
    """

    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a NegBinomNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(NegBinomNN, self).__init__(
            loss_fn=neg_binom_nll,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 2),
            nn.Softplus(),  # To ensure positivity of output params.
        )

        self.mp = MedianPrecision()
        self.crps = ContinuousRankedProbabilityScore(mode="discrete")

        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, alpha), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, alpha), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()
        return y_hat

    def _point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        dist = self.predictive_dist(y_hat)
        return dist.mode

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "mp": self.mp,
            "crps": self.crps,
        }

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        dist = self.predictive_dist(y_hat)
        var = dist.variance
        precision = 1 / var
        targets = y.flatten()
        support = torch.arange(2000, device=y_hat.device).view(-1, 1)
        probs_over_support = torch.exp(dist.log_prob(support)).T

        self.mp.update(precision)
        self.crps.update(probs_over_support, targets)

    def _predictive_dist_impl(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.NegativeBinomial:
        eps = torch.tensor(1e-3, device=y_hat.device)

        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        alpha = torch.clamp(alpha, min=eps).flatten()

        # Convert to standard parametrization.
        var = mu + alpha * mu**2
        p = mu / torch.maximum(var, eps)

        # Torch docs lie and say this should be P(success).
        failure_prob = torch.clamp(1 - p, min=eps, max=1 - eps)

        n = mu**2 / torch.maximum(var - mu, eps)
        dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
        return dist
