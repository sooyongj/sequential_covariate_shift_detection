import numpy as np

from torch.utils.data import Subset


def split_train_val(org_train_set, shuffle=False, valid_ratio=0.1):
  num_train = len(org_train_set)
  split = int(np.floor(valid_ratio * num_train))

  indices = list(range(num_train))

  if shuffle:
    np.random.shuffle(indices)

  train_idx, val_idx = indices[split:], indices[:split]

  new_train_set = Subset(org_train_set, train_idx)
  val_set = Subset(org_train_set, val_idx)

  assert num_train - split == len(new_train_set)
  assert split == len(val_set)

  return new_train_set, val_set
