import random
from pathlib import Path
from typing import Type

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from deep_uncertainty.datamodules import COCOPeopleDataModule
from deep_uncertainty.datamodules import MNISTDataModule
from deep_uncertainty.datamodules import ReviewsDataModule
from deep_uncertainty.datamodules import TabularDataModule
from deep_uncertainty.enums import DatasetType
from deep_uncertainty.enums import HeadType
from deep_uncertainty.enums import ImageDatasetName
from deep_uncertainty.enums import TextDatasetName
from deep_uncertainty.models import DoublePoissonHomoscedasticNN
from deep_uncertainty.models import DoublePoissonNN
from deep_uncertainty.models import FaithfulGaussianNN
from deep_uncertainty.models import GaussianNN
from deep_uncertainty.models import LogFaithfulGaussianNN
from deep_uncertainty.models import LogGaussianNN
from deep_uncertainty.models import NaturalGaussianNN
from deep_uncertainty.models import NegBinomNN
from deep_uncertainty.models import PoissonNN
from deep_uncertainty.models.backbones import DistilBert
from deep_uncertainty.models.backbones import Identity
from deep_uncertainty.models.backbones import MLP
from deep_uncertainty.models.backbones import MNISTCNN
from deep_uncertainty.models.backbones import MobileNetV3
from deep_uncertainty.models.backbones import SmallerMLP
from deep_uncertainty.models.backbones import ViT
from deep_uncertainty.models.discrete_regression_nn import DiscreteRegressionNN
from deep_uncertainty.utils.configs import TrainingConfig
from deep_uncertainty.utils.generic_utils import partialclass


def get_model(config: TrainingConfig, return_initializer: bool = False) -> DiscreteRegressionNN:

    initializer: Type[DiscreteRegressionNN]

    if config.head_type == HeadType.GAUSSIAN:
        if config.beta_scheduler_type is not None:
            initializer = partialclass(
                GaussianNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = GaussianNN
    elif config.head_type == HeadType.LOG_GAUSSIAN:
        if config.beta_scheduler_type is not None:
            initializer = partialclass(
                LogGaussianNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = LogGaussianNN
    elif config.head_type == HeadType.FAITHFUL_GAUSSIAN:
        initializer = FaithfulGaussianNN
    elif config.head_type == HeadType.LOG_FAITHFUL_GAUSSIAN:
        initializer = LogFaithfulGaussianNN
    elif config.head_type == HeadType.NATURAL_GAUSSIAN:
        initializer = NaturalGaussianNN
    elif config.head_type in (HeadType.POISSON, HeadType.POISSON_GLM):
        initializer = PoissonNN
    elif config.head_type == HeadType.DOUBLE_POISSON:
        if config.beta_scheduler_type is not None:
            initializer = partialclass(
                DoublePoissonNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = DoublePoissonNN
    elif config.head_type == HeadType.DOUBLE_POISSON_GLM:
        if config.beta_scheduler_type is not None:
            initializer = partialclass(
                DoublePoissonHomoscedasticNN,
                beta_scheduler_type=config.beta_scheduler_type,
                beta_scheduler_kwargs=config.beta_scheduler_kwargs,
            )
        else:
            initializer = DoublePoissonHomoscedasticNN
    elif config.head_type in (HeadType.NEGATIVE_BINOMIAL, HeadType.NEGATIVE_BINOMIAL_GLM):
        initializer = NegBinomNN

    if config.dataset_type == DatasetType.TABULAR:
        if config.head_type in (
            HeadType.POISSON_GLM,
            HeadType.NEGATIVE_BINOMIAL_GLM,
            HeadType.DOUBLE_POISSON_GLM,
        ):
            backbone_type = Identity
        elif "isolated" in str(config.dataset_spec):
            backbone_type = SmallerMLP
        else:
            backbone_type = MLP
        backbone_kwargs = {"input_dim": config.input_dim}
    elif config.dataset_type == DatasetType.TEXT:
        backbone_type = DistilBert
        backbone_kwargs = {}
    elif config.dataset_type == DatasetType.IMAGE:
        if config.dataset_spec == ImageDatasetName.MNIST:
            backbone_type = MNISTCNN
        elif config.dataset_spec == ImageDatasetName.COCO_PEOPLE:
            backbone_type = ViT
        else:
            backbone_type = MobileNetV3
        backbone_kwargs = {}
    backbone_kwargs["output_dim"] = config.hidden_dim
    backbone_kwargs["freeze_backbone"] = config.freeze_backbone

    model = initializer(
        backbone_type=backbone_type,
        backbone_kwargs=backbone_kwargs,
        optim_type=config.optim_type,
        optim_kwargs=config.optim_kwargs,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
    )
    if return_initializer:
        return model, initializer
    else:
        return model


def get_datamodule(
    dataset_type: DatasetType, dataset_spec: Path | ImageDatasetName, batch_size: int
) -> L.LightningDataModule:
    if dataset_type == DatasetType.TABULAR:
        return TabularDataModule(
            dataset_path=dataset_spec,
            batch_size=batch_size,
            num_workers=9,
            persistent_workers=True,
        )
    elif dataset_type == DatasetType.IMAGE:
        if dataset_spec == ImageDatasetName.MNIST:
            return MNISTDataModule(
                root_dir="./data/mnist",
                batch_size=batch_size,
                num_workers=8,
                persistent_workers=True,
            )
        elif dataset_spec == ImageDatasetName.COCO_PEOPLE:
            return COCOPeopleDataModule(
                root_dir="./data/coco-people",
                batch_size=batch_size,
                num_workers=16,
                persistent_workers=True,
            )
    elif dataset_type == DatasetType.TEXT:
        if dataset_spec == TextDatasetName.REVIEWS:
            return ReviewsDataModule(
                root_dir="./data/amazon-reviews",
                batch_size=batch_size,
                num_workers=16,
                persistent_workers=True,
            )


def save_metrics_plots(log_dir: Path):
    metrics = pd.read_csv(log_dir / "metrics.csv").iloc[:-1]
    train_loss = metrics["train_loss_epoch"].dropna()
    val_loss = metrics["val_loss"].dropna()
    train_mae = metrics["train_mae_epoch"].dropna()
    val_mae = metrics["val_mae"].dropna()
    train_rmse = metrics["train_rmse_epoch"].dropna()
    val_rmse = metrics["val_rmse"].dropna()

    # Losses plot.
    fig, ax = plt.subplots(1, 1)
    train_loss.plot(ax=ax)
    val_loss.plot(ax=ax)
    ax.legend()
    fig.savefig(log_dir / "losses.png")
    plt.close(fig)

    # MAE plot.
    fig, ax = plt.subplots(1, 1)
    train_mae.plot(ax=ax)
    val_mae.plot(ax=ax)
    ax.legend()
    fig.savefig(log_dir / "mae.png")
    plt.close(fig)

    # RMSE plot.
    fig, ax = plt.subplots(1, 1)
    train_rmse.plot(ax=ax)
    val_rmse.plot(ax=ax)
    ax.legend()
    fig.savefig(log_dir / "rmse.png")
    plt.close(fig)


def fix_random_seed(random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


def get_chkp_callbacks(chkp_dir: Path, chkp_freq: int) -> list[ModelCheckpoint]:
    temporal_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        every_n_epochs=chkp_freq,
        filename="{epoch}",
        save_top_k=-1,
        save_last=True,
    )
    best_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        monitor="val_loss",
        every_n_epochs=1,
        filename="best_loss",
        save_top_k=1,
        enable_version_counter=False,
    )
    best_mae_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        monitor="val_mae",
        every_n_epochs=1,
        filename="best_mae",
        save_top_k=1,
        enable_version_counter=False,
    )
    return [
        temporal_checkpoint_callback,
        best_loss_checkpoint_callback,
        best_mae_checkpoint_callback,
    ]
