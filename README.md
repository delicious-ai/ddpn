# Fully Heteroscedastic Count Regression with Deep Double Poisson Networks

This repository contains the official implementation of the ICML 2025 paper "Fully Heteroscedastic Count Regression with Deep Double Poisson Networks".

Authors: [Spencer Young*](https://github.com/spencermyoung513), [Porter Jenkins*](https://science.byu.edu/directory/porter-jenkins), [Longchao Da](https://longchaoda.github.io/LongchaoHere/), [Jeffrey Dotson](https://fisher.osu.edu/people/dotson.83), [Hua Wei](https://search.asu.edu/profile/3095662).

\*Equal contribution

## Overview

We introduce DDPN, a novel neural count regression model that outputs the parameters of the Double Poisson distribution, enabling *arbitrarily high or low predictive aleatoric uncertainty* for count data. DDPN also exhibits *learnable loss attenuation*, a robust property that is present in heteroscedastic Gaussian models (used for real-valued probabilistic regression). When ensembled, DDPN can effectively estimate epistemic uncertainty in addition to aleatoric uncertainty.

![An overview of the Deep Double Poisson Network](deep_uncertainty/figures/artifacts/ddpn-fig.png)

## Important Links

Important figures used in the paper, along with the code that generated them, can be found in [this directory](deep_uncertainty/figures).

Our implementations of various "aleatoric only" techniques referenced in the paper can be found at the following locations:

- [Gaussian DNN](deep_uncertainty/models/gaussian_nn.py)
- [Poisson DNN](deep_uncertainty/models/poisson_nn.py)
- [NB DNN](deep_uncertainty/models/neg_binom_nn.py)
- [Stirn et al.](deep_uncertainty/models/faithful_gaussian_nn.py)
- [Seitzer et al.](deep_uncertainty/models/gaussian_nn.py) (with a `beta_scheduler`)
- [Immer et al.](deep_uncertainty/models/natural_gaussian_nn.py)
- [DDPN (ours)](deep_uncertainty/models/double_poisson_nn.py)
- [β-DDPN (ours)](deep_uncertainty/models/double_poisson_nn.py) (with a `beta_scheduler`)

Implementations of "deep ensembles" referenced in the paper are found at:

- [Laksh. et al.](deep_uncertainty/models/ensembles/gaussian_mixture_nn.py)
- [Immer et al.](deep_uncertainty/models/ensembles/natural_gaussian_mixture_nn.py)
- [Stirn et al.](deep_uncertainty/models/ensembles/faithful_gaussian_mixture_nn.py)
- [Seitzer et al.](deep_uncertainty/models/ensembles/gaussian_mixture_nn.py) (where individual models were trained with a `beta_scheduler`)
- [Poisson DNN Mixture](deep_uncertainty/models/ensembles/poisson_mixture_nn.py)
- [NB DNN Mixture](deep_uncertainty/models/ensembles/neg_binom_mixture_nn.py)
- [DDPN Mixture](deep_uncertainty/models/ensembles/double_poisson_mixture_nn.py)
- [β-DDPN Mixture](deep_uncertainty/models/ensembles/double_poisson_mixture_nn.py) (where individual models were trained with a `beta_scheduler`)

## Setup

### Install Dependencies

```bash
conda create --name ddpn python=3.10
conda activate ddpn
pip install -r requirements.txt
```

### Download Data

Contact the authors for access to the datasets used in the experiments.

### Train a Model

To train a model, first fill out a config (using [this config](deep_uncertainty/training/sample_train_config.yaml) as a template). Then, from the terminal, run

```bash
python deep_uncertainty/training/train_model.py --config path/to/your/config.yaml
```

Logs / saved model weights will be found at the locations specified in your config.

#### Tabular Datasets

If fitting a model on tabular data, the training script assumes the dataset will be stored locally in `.npz` files with `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test` splits (for reference, these files are automatically produced by our [synthetic data generating code](deep_uncertainty/data_generator.py)). Pass a path to this `.npz` file in the `dataset` `spec` key in the config (also ensure that the `dataset` `type` is set to `tabular` and the `dataset` `input_dim` key is properly specified).

#### Image Datasets

The currently-supported image datasets for training models are:

- `MNIST` (Regressing the digit labels instead of classifying)
- `COCO-People` (All images in COCO containing people, labeled with the count of "person" annotations)

To train a model on any of these datasets, simply specify `"image"` for the `dataset` `type` key in the config, then set `dataset` `spec` to the requisite dataset name (see the options in the `ImageDatasetName` class [here](deep_uncertainty/enums.py))

#### Text Datasets

The currently-supported text datasets for training models are:

- `Amazon Reviews` (We predict the review rating (1-5 stars) from its associated text)

To train a model on this dataset, simply specify `"text"` for the `dataset` `type` key in the config, then set `dataset` `spec` to the requisite dataset name (see the options in the `TextDatasetName` class [here](deep_uncertainty/enums.py))

### Evaluate a Model

#### Individual Models

To obtain evaluation metrics for a given model (and have them save to the specified log directory), use the following command:

```bash
python deep_uncertainty/evaluation/eval_model.py \
--log-dir path/to/training/log/dir \
--chkp-path path/to/model.ckpt
```

#### Ensembles

Sometimes, we may wish to evaluate an ensemble of models. To do this, first fill out a config using [this file](deep_uncertainty/evaluation/sample_ensemble_config.yaml) as a template. Then run:

```bash
python deep_uncertainty/evaluation/eval_ensemble.py --config path/to/config.yaml
```

## Reproducing Results

Training configs for each model benchmarked in "Fully Heteroscedastic Count Regression with Deep Double Poisson Networks" can be found in the top-level [configs directory](configs).

To re-run the experiments from the paper, first contact the authors for the requisite dataset files and ensure they are saved in a top-level `data` directory. Then run the following command for any valid dataset:

```bash
bash deep_uncertainty/scripts/train_models.sh <dataset-name>
```

The resultant model weights will be saved to `chkp/{dataset-name}`. Note that some aspects of training configs may need to be adjusted depending on available hardware (# GPUs / GPU capacity) to ensure that the effective batch size matches what is reported in the paper.

Models can then be evaluated via

```bash
bash deep_uncertainty/scripts/eval_models.sh <dataset-name> <results-dir>
```

Ensembles are evaluated via

```bash
bash deep_uncertainty/scripts/eval_ensembles.sh <dataset_name>
```

Results will be saved to `{results-dir}/{dataset-name}`.

## Development

### Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting. There will also be a flake8 check that provides warnings about various Python styling violations. These must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

### Adding New Models

All regression models should inherit from the `DiscreteRegressionNN` class (found [here](deep_uncertainty/models/discrete_regression_nn.py)). This base class is a `lightning` module, which allows for a lot of typical NN boilerplate code to be abstracted away. Beyond setting a few class attributes like `loss_fn` while calling the super-initializer, the only methods you need to actually write to make a new module are:

- `_forward_impl` (defines a forward pass through the network)
- `_predict_impl` (defines how to make predictions with the network, including any transformations on the output of the forward pass)
- `_point_prediction` (defines how to interpret network output as a single point prediction for a regression target)
- `_addl_test_metrics_dict` (defines any metrics beyond rmse/mae that are computed during model evaluation)
- `_update_addl_test_metrics_batch` (defines how to update additional metrics beyond rmse/mae for each test batch).

See existing model classes like `GaussianNN` (found [here](deep_uncertainty/models/gaussian_nn.py)) for an example of these steps.


## Citations

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{young2025ddpn,
  title={Fully Heteroscedastic Count Regression with Deep Double Poisson Networks},
  author={Young, Spencer and Jenkins, Porter and Da, Longchao and Dotson, Jeffrey and Wei, Hua},
  booktitle={International Conference on Machine Learning},
  year={2025},
  month={July},
  abbr={ICML 2025}
}
```
