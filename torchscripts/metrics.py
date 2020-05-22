import torch
import torch.nn.functional as F


def classification_metric(y_hat, y):
  probability = F.softmax(y_hat, dim=1)
  probability, prediction = torch.max(probability, axis=1)
  number_of_correct = (prediction == y).sum().item()
  return (prediction.detach().cpu(), probability.detach().cpu()), \
      number_of_correct


def regression_metric(y_hat, y):
  # In this case we will use MSE
  distance = torch.nn.MSELoss()(y_hat, y).item()
  return y_hat, distance
