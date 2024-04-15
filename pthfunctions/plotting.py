import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple


def _convert_metric_name(metric_name: str) -> str:
  """
  Converts a metric name to a format with spaces and capitalized first letters.

  Args:
      metric_name: A string representing the metric name (e.g., "test_loss").

  Returns:
      A string representing the metric name in a better format (e.g., "Test Loss").
  """
  words = metric_name.split('_')
  words = [word.capitalize() for word in words]
  return ' '.join(words)


def plot_results_comparison(metric_to_plot: str = None, size: Tuple[float, float] = (15, 10), **kwargs: pd.DataFrame) -> None:
  """
  Plots a comparison of all models' results, including train/test loss and accuracy,
  given that they trained for the same number of epochs.

  Args:
      metric_to_plot: A string representing the metric to plot ("train_loss", "test_loss",
        "train_acc", and "test_acc"). If None, all metrics are plotted.
      size: A tuple of floats representing the size of the plot.
      **kwargs: Named arguments where keys are model names and values are pandas DataFrames
        containing training and testing data. Each DataFrame is expected to have columns
        named "train_loss", "test_loss", "train_acc", and "test_acc".
  
  Returns:
      None if the plot is successfully displayed.
      ValueError if the metric_to_plot is not a valid metric name.
  """
  valid_metrics = ["train_loss", "test_loss", "train_acc", "test_acc"]

  if metric_to_plot:
    if metric_to_plot not in valid_metrics:
      raise ValueError(f"Invalid metric: {metric_to_plot}. Valid options are: {', '.join(valid_metrics)}")
    metrics = [metric_to_plot]
  else:
    metrics = valid_metrics

  plt.figure(figsize=size)

  for metric in metrics:
    for model_name, df in kwargs.items():
      epochs = range(len(df))
      plt.plot(epochs, df[metric], label=model_name)

    plt.title(_convert_metric_name(metric))
    plt.xlabel("Epochs")
    plt.legend()

  plt.show()