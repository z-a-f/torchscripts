# torchscripts -- Scripts for simplifying PyTorch training and Inference

## Requirements

- [PyTorch](https://pytorch.org/)

## Example Usage

```python
from torchscripts import ClassificationTrainer
from torchscripts import classification_metric

# Assume there is a `*_data_loader` of type `torch.utils.data.DataLoader`
# Assume there is a PyTorch `model` and `optimizer`
trainer = ClassificationTrainer(model, optimizer)
history = trainer.fit(training_data_loader, validation_data_loader,
                      epochs=25, batch_size=128)
```

In this example, `history` will have the training history over the epochs.
History is stored as a dict with the following keys:

```
history
├── epochs        # List of epochs
├── train         # Training dict
│   ├── loss      # - List of training losses for every epoch
│   └── metric    # - List of training metrics (i.e. accuracy) for every epoch
└── validation    # Validation dict
    ├── loss      # - List of validation losses for every epoch
    └── metric    # - List of validation metrics (i.e. accuracy) for every epoch
```

## API (kinda)

### `class Trainer(object)`

Generic trainer class.
Implements the `fit` and `predict` methods

**Example**

```python
trainer = Trainer(model, optim.SGD(model.parameters(), lr=1.0),
                  loss_function=nn.MSELoss(), metric_function=nn.MSELoss())
training_history = trainer.fit(train_data_loader, validation_data_loader, batch_size=16)
prediction = trainer.predict(test_data_loader)
```

**Constructor**

```
model             #  PyTorch model
optimizer         #  Optimizer function.
loss_function     #  Loss function. Must be callable with (prediction, target) arguments
metric_function   # Metric to report. Must be callable with (prediction, target) arguments
```

**`fit`**

Fits and validates the model.
Validation is optional.

```
train_data,                   # Training data
validation_data=None,         # Validation data
epochs=10,                    # Number of epochs to train for
batch_size=128,               # Batch size
shuffle=True,                 # Shuffle during training?
train_transform=None,         # Data transformation
validation_transform=None,    # Data transformation
verbose=True                  # Report progress
```

The `train_data` and `validation_data` can be either `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, or `(X, y)` tensor pairs.
The `train_transform` and `validation_transform` only apply if the data provided is a tuple of tensors `(X, y)`.
Otherwise the tranformation within the `torch.utils.data.Dataset` is used.

**`fit_epoch`**

Fits the model for a single epoch.
No validation is performed.

```
X_or_data,        # Training data
y=None,           # Training targets
batch_size=None,  # Batch size
shuffle=None,     # Shuffle during training?
transform=None    # Data transformation (for training only)
```

The `X_or_data` can be either a `torch.Tensor`, `torch.utils.data.Dataset`, or `torch.utils.data.DataLoader`.
If it is a `Tensor`, argument `y` MUST be provided.
Otherwise, `y` MUST be `None`.

**`predict`**

Runs the model inference on some input.
If the targets are provided, otherwise, the loss and metric will be 0.

```
X_or_data       # Inference data
y=None          # Inference targets (for loss computation)
transform=None  # Data transformation
batch_size=128  # Batch size
```

Returns `prediction`, `loss`, `metric`

### `class ClassificationTrainer(Trainer)`

Trainer for classification.
Implements `Trainer` with `CrossEntropyLoss` and `classification_metric`.

**Example**

```python
trainer = ClassificationTrainer(model, optim.SGD(model.parameters(), lr=1.0))
training_history = trainer.fit(train_data_loader, validation_data_loader, batch_size=16)
prediction = trainer.predict(test_data_loader)
```

**Constructor**

```
model             #  PyTorch model
optimizer         #  Optimizer function.
```

### `class RegressionTrainer(Trainer)`

Trainer for classification.
Implements `Trainer` with `MSELoss` and `regression_metric`.

**Example**

```python
trainer = RegressionTrainer(model, optim.SGD(model.parameters(), lr=1.0))
training_history = trainer.fit(train_data_loader, validation_data_loader, batch_size=16)
prediction = trainer.predict(test_data_loader)
```

**Constructor**

```
model             #  PyTorch model
optimizer         #  Optimizer function.
```


### Metrics

**`classification_metric`**

Converts one-hot encoded classification using `argmax`.
Returns `(prediction, probability), number_of_correct`.

**`regression_metric`**

Runs the MSE loss.
Returns `(y_hat, MSE_loss)`
