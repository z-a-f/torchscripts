import torch


class TensorDataset(torch.utils.data.Dataset):
  r"""Dataset wrapping tensors.

  Each sample will be retrieved by indexing tensors along the first dimension.

  Arguments:
    *tensors (Tensor): tensors that have the same size of the first dimension.
  """

  def __init__(self, *tensors, transform=None):
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    self.tensors = tensors
    self.transform = transform
    if isinstance(self.transform, (list, tuple)):
      assert len(self.transform) == len(self.tensors), \
          "Transformations must be the same length as number of tensors"

  def __getitem__(self, index):
    if self.transform is None:
      return tuple(tensor[index] for tensor in self.tensors)
    elif isinstance(self.transform, (list, tuple)):
      return tuple(self.transform[idx](self.tensors[idx][index])
                   for idx in range(self.tensors))
    else:
      return tuple(self.transform(self.tensors[0][index])) + \
          tuple(tensor[index] for tensor in self.tensors[1:])

  def __len__(self):
    return self.tensors[0].size(0)
