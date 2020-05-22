import torch
import torch.nn.functional as F

def classification_metric(y_hat, y):
  probability, prediction = torch.max(y_hat, axis=1)
  probability = F.softmax(probability, dim=1)
  inverse_distance = (prediction == y).sum().item()
  return (prediction, probability), inverse_distance

def regression_metric(y_hat, y):
  # In this case we will use MSE
  distance = torch.nn.MSELoss()(y_hat, y).item()
  return y_hat, distance
