import pandas as pd
import torch
from tqdm.auto import tqdm


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str = "cpu"):
  """
  Trains the model for one epoch using the provided dataloader, loss function and optimizer.

  Args:
      :model: The model to be trained.
      :dataloader: Dataloader containing the training data.
      :loss_fn: Loss function to be used in the training step.
      :optimizer: Optimizer to be used in the training step.
      :device: String specifying the device to be used in the training step (e.g., "cpu" or "cuda").
  """
  # Put the model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to the target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    y_pred = model(X) # output model logits

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Calculate accuracy metric
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class==y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device="cpu"):
  """
  Tests the model for one epoch using the provided dataloader and loss function.
  
  Args:
      :model: The model to be tested.
      :dataloader: DataLoader containing the testing data.
      :loss_fn: Loss function to be used in the testing step.
      :device: String specifying the device to be used in the testing step (e.g., "cpu" or "cuda").
  """
  # Put model in eval mode
  model.eval()

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0,  0

  # Turn on inference mode
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
      # Send data to the target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      test_pred_logits = model(X)

      # 2. Calculate the loss
      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      # Calculate the accuracy
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch
  test_loss /= len(dataloader)
  test_acc /= len(dataloader)
  return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int = 5,
          device: str = "cpu",
          console_ratio: int = 0) -> pd.DataFrame:
  """
  Trains a model using the provided training and testing dataloaders, optimizer, loss function, and number of epochs.
  
  Args:
      :model: The model in which perform the training loop.
      :train_dataloader: DataLoader with the training data.
      :test_dataloader: DataLoader with the testing data.
      :loss_fn: Loss function to be used in the training loop.
      :optimizer: Optimizer to be used in the training loop.
      :epochs: Number of epochs to train the model.
      :device: String specifying the device to use in the training loop (e.g., "cpu" or "cuda").
      :console_ratio: Ratio of epochs to print out the training and testing results.
  
  Returns:
      :result: DataFrame containing the training and testing results for each epoch.
  """

  # 1. Create empty results dictionary
  results = pd.DataFrame({"epoch": [],
             "train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []})
  
  console_ratio = max(0, min(console_ratio, 1)) # Ensure the console_ratio is between 0 and 1

  # 2. Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):

    console_out = console_ratio > 0 and epoch % (1 / console_ratio) == 0

    if console_out:
      print(f"-----------\nEpoch {epoch}")

    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)

    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    # 3. Print out results
    if console_out:
      print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    # 4. Update results dictionary
    results.loc[len(results.index)] = [epoch, train_loss, train_acc, test_loss, test_acc] # Adds a new row of data to the dataframe

  # 5. Return the filled results at the end of the epochs
  return results