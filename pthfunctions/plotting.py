import matplotlib.pyplot as plt
import pandas as pd


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


def plot_results_comparison(results: dict[str, pd.DataFrame], metric_to_plot: str = None, size: tuple[float, float] = (15, 10)) -> None:
  """
  Plots a comparison of all models' results, including train/test loss and accuracy.

  Args:
      metric_to_plot: A string representing the metric to plot ("train_loss", "test_loss",
        "train_acc", and "test_acc"). If None, all metrics are plotted.
      size: A tuple of floats representing the size of the plot.
      **results: Named arguments where keys are model names and values are pandas DataFrames
        containing training and testing data. Each DataFrame is expected to have columns
        named "train_loss", "test_loss", "train_acc", and "test_acc".
  
  Returns:
      None if the plot is successfully displayed.
  """
  valid_metrics = ["train_loss", "test_loss", "train_acc", "test_acc"]

  if metric_to_plot:
    if metric_to_plot not in valid_metrics:
      raise ValueError(f"Invalid metric: {metric_to_plot}. Valid options are: {', '.join(valid_metrics)}")
    metrics = [metric_to_plot]
  else:
    metrics = valid_metrics

  plt.figure(figsize=size)

  for graph, metric in enumerate(metrics):
    if len(metrics) > 1:
      plt.subplot(2, 2, graph + 1)

    for model_name, df in results.items():
      epochs = range(len(df))
      plt.plot(epochs, df[metric], label=model_name)

    plt.title(_convert_metric_name(metric))
    plt.xlabel("Epochs")
    plt.legend()

  plt.show()