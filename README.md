# PthFunctions

Library of functions for PyTorch development.

## Installation

To install the package use the following command.

```
pip install git+https://github.com/yer-marti/pthfunctions.git
```

To install the package in Google Colab, add `!` at the beginning.

```
!pip install git+https://github.com/yer-marti/pthfunctions.git
```

## Requirements

This library requires the following dependencies:

* torch
* tqdm
* pandas
* matplotlib

## Documentation

1. [Plotting module](#plotting-module)
2. [Training module](#training-module)

### Plotting module

#### pthfunctions.plotting.plot_results_comparison

```
pthfunctions.plotting.plot_results_comparison(results, metric_to_plot=None, size=(15, 10))
```

Plots a comparison of all models' results, including train/test loss and accuracy.

| Parameters | Description |
| --- | --- |
| results: dict[str, pd.DataFrame] | A dictionary with model names as keys and DataFrames with the results as values. The DataFrames should have columns for "train_loss", "test_loss", "train_acc", and "test_acc". |
| metric_to_plot: str, ddefault=None | A string representing the metric to plot ("train_loss", "test_loss", "train_acc", and "test_acc"). If None, all metrics are plotted. |
| size: tuple[float, float], default=(15, 10) | A tuple of floats representing the size of the plot. |

### Training module

#### pthfunctions.training.train_step

```
pthfunctions.training.train_step(model, dataloader, loss_fn, optimizer, device="cpu")
```

Trains the model for one epoch using the provided dataloader, loss function and optimizer.

| Parameters | Description |
| --- | --- |
| model: torch.nn.Module | The model to be trained. |
| dataloader: torch.utils.data.DataLoader | Dataloader containing the training data. |
| loss_fn: torch.nn.Module | Loss function to be used in the training step. |
| optimizer:torch.optim.Optimizer | Optimizer to be used in the training step. |
| device: str, default="cpu" | String specifying the device to be used in the training step (e.g., "cpu" or "cuda"). |

#### pthfunctions.training.test_step

```
pthfunctions.training.test_step(model, dataloader, loss_fn, device="cpu")
```

Tests the model for one epoch using the provided dataloader and loss function.

| Parameters | Description |
| --- | --- |
| model: torch.nn.Module | The model to be tested. |
| dataloader: torch.utils.data.DataLoader | Dataloader containing the testing data. |
| loss_fn: torch.nn.Module | Loss function to be used in the testing step. |
| device: str, default="cpu" | String specifying the device to be used in the testing step (e.g., "cpu" or "cuda"). |

#### pthfunctions.training.train

```
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs=5, device="cpu")
```

Trains a model using the provided training and testing dataloaders, optimizer, loss function, and number of epochs.

| Parameters | Description |
| --- | --- |
| model: torch.nn.Module | The model in which perform the training loop. |
| train_dataloader: torch.utils.data.DataLoader | DataLoader with the training data. |
| test_dataloader: torch.utils.data.DataLoader | DataLoader with the testing data. |
| loss_fn: torch.nn.Module | Loss function to be used in the training loop. |
| optimizer: torch.optim.Optimizer | Optimizer to be used in the training loop. |
| epochs: int, default=5 | Number of epochs to train the model. |
| device: str, default="cpu" | String specifying the device to use in the training loop (e.g., "cpu" or "cuda"). |

| Returns | Description |
| --- | --- |
| results: pandas.DataFrame | A pandas DataFrame containing the training and testing results for each epoch. | 