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

### `class Trainer`

Generic trainer class.

#### Methods
