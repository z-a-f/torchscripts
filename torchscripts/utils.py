
def dataset_has_target(dataset):
  sample = dataset[0]
  return (isinstance(sample, (list, tuple)) and
          len(sample) > 1 and
          sample[1] is not None)
