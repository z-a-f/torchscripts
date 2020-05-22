"""Utilities for training."""
import os

import torch
from torch import nn
from torch.utils import data

from .metrics import classification_metric
from .metrics import regression_metric

from .utils import dataset_has_target

class Trainer(object):
  def __init__(self,
               model,
               optimizer,
               loss_function,
               metric_function):
    assert callable(loss_function), 'Loss function must be callable'
    assert callable(metric_function), 'Metric must be callable'

    self.model = model
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.metric_function = metric_function

  ########################### Training Routines ################################
  def fit(self, train_data, validation_data=None, epochs=10, batch_size=128,
          shuffle=True, train_transform=None, validation_transform=None,
          verbose=True):
    X, y = None, None
    Xv, yv = None, None
    if isinstance(train_data, (list, tuple)):
      X, y = train_data
      if validation_data is not None:
        Xv, yv = validation_data
    else:
      X = train_data
      if validation_data is not None:
        Xv = validation_data
    history = {
      'epoch': [],
      'train': {
        'loss': [],
        'metric': []
      },
      'validation': {
        'loss': [],
        'metric': []
      },
    }
    for epoch in range(epochs):
      history['epoch'].append(epoch)
      loss, metric = self.fit_epoch(X, y, batch_size=batch_size, shuffle=shuffle,
                               transform=train_transform)
      history['train']['loss'].append(loss)
      history['train']['metric'].append(metric)
      if verbose:
        print(f'{epoch+1}/{epochs}')
        print(f'\tTrain Loss: {loss}, Metric: {metric}')

      if Xv is not None:
        _, loss, metric = self.predict(Xv, yv, batch_size=batch_size,
                                       transform=validation_transform)
        history['validation']['loss'].append(loss)
        history['validation']['metric'].append(metric)
        print(f'\tValid Loss: {loss}, Metric: {metric}')
    return history


  def fit_epoch(self, X_or_data, y=None, batch_size=None, shuffle=None,
                transform=None):
    if y is not None:
      assert isinstance(X_or_data, type(y)), "X and y must be the same type"
      return self._fit_xy(X_or_data, y, batch_size,
                         shuffle, transform=transform)
    elif isinstance(X_or_data, data.Dataset):
      # Dataset case
      return self._fit_dataset(X_or_data, batch_size, shuffle)
    elif isinstance(X_or_data, data.DataLoader):
      # Dataloader case
      return self._fit_dataloader(dataloader=X_or_data)
    raise NotImplementedError("Cannot run fit with the first argument being",
                              type(X_or_train_data))

  def _fit_xy(self, *, X, y, batch_size=None, shuffle=None, transform=None):
    dataset = data.TensorDataset(X, y, transform=transform)
    return self._fit_dataset(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle)

  def _fit_dataset(self, *, dataset, batch_size=None, shuffle=None):
    dataloader = data.DataLoader(dataset, shuffle=shuffle,
                                 batch_size=batch_size, pin_memory=True,
                                 num_workers=os.cpu_count())
    return self._fit_dataloader(dataloader=dataloader)

  def _fit_dataloader(self, *, dataloader):
    was_training = self.model.training
    device = next(self.model.parameters()).device
    self.model.train(True)
    epoch_loss = 0.0
    epoch_metric = 0.0
    samples = 0
    for X, y in dataloader:
      X = X.to(device)
      y = y.to(device)
      self.optimizer.zero_grad()

      y_hat = self.model(X)
      loss = self.loss_function(y_hat, y)
      loss.backward()
      self.optimizer.step()

      prediction, metric = self.metric_function(y_hat, y)  # torch.max(y_hat, axis=1)

      batch_size = X.size(0)
      samples += batch_size
      epoch_loss += loss.item() * batch_size
      epoch_metric += metric
    self.model.train(was_training)
    return epoch_loss / samples, epoch_metric / samples

  ########################### Inference Routines ###############################
  def predict(self, X_or_data, y=None, transform=None, batch_size=128):
    if isinstance(X_or_data, torch.Tensor):
      if y is not None:
        assert isinstance(X_or_data, type(y)), "X and y must be the same type"
      return self._predict_xy(X_or_data, y, batch_size, transform=transform)
    elif isinstance(X_or_data, data.Dataset):
      # Dataset case
      return self._predict_dataset(X_or_data, batch_size)
    elif isinstance(X_or_data, data.DataLoader):
      # Dataloader case
      return self._predict_dataloader(dataloader=X_or_data)
    raise NotImplementedError("Cannot run fit with the first argument being",
                              type(X_or_train_data))

  def _predict_xy(self, *, X, y=None, batch_size=None, transform=None):
    if y is None:
      dataset = data.TensorDataset(X, transform=transform)
    else:
      dataset = data.TensorDataset(X, y, transform=transform)
    return self._predict_dataset(dataset=dataset, batch_size=batch_size)

  def _predict_dataset(self, *, dataset, batch_size=None):
    dataloader = data.DataLoader(dataset, batch_size=batch_size,
                                 pin_memory=True, num_workers=os.cpu_count())
    return self._predict_dataloader(dataloader=dataloader)

  def _predict_dataloader(self, *, dataloader):
    was_training = self.model.training
    device = next(self.model.parameters()).device
    self.model.train(False)

    has_target = dataset_has_target(dataloader.dataset)

    epoch_loss = 0.0
    epoch_metric = 0.0
    samples = 0
    with torch.no_grad():
      for sample in dataloader:
        X = sample[0].to(device)
        if has_target:
          y = sample[1].to(device)

        y_hat = self.model(X)
        if not has_target:
          y = y_hat
        loss = self.loss_function(y_hat, y)
        prediction, metric = self.metric_function(y_hat, y)

        batch_size = X.size(0)
        samples += batch_size
        epoch_loss += loss.item() * batch_size
        epoch_metric += metric
    self.model.train(was_training)
    return prediction, epoch_loss / samples, epoch_metric / samples


class ClassificationTrainer(Trainer):
  def __init__(self, model, optimizer):
    super(ClassificationTrainer, self).__init__(
      model,
      optimizer,
      loss_function=nn.CrossEntropyLoss(),
      metric_function=classification_metric
    )

class RegressionTrainer(Trainer):
  def __init__(self, model, optimizer):
    super(RegressionTrainer, self).__init__(
      model,
      optimizer,
      loss_function=nn.MSELoss(),
      metric_function=regression_metric
    )
