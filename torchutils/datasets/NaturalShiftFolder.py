import logging
import torchvision

from torchvision.datasets.folder import default_loader


class NaturalShiftFolder(torchvision.datasets.ImageFolder):
  def __init__(self, root, transform=None, target_transform=None, src_labels=None, tgt_labels=None, original_label_together=False, loader=default_loader, is_valid_file=None):
    super(NaturalShiftFolder, self).__init__(root,
                                             transform=transform,
                                             target_transform=target_transform,
                                             loader=loader,
                                             is_valid_file=is_valid_file)
    self.logger = logging.getLogger(__name__)
    self.original_label_together = original_label_together

    self.src_labels = set(self.class_to_idx[lbl] for lbl in src_labels)
    self.tgt_labels = set(self.class_to_idx[lbl] for lbl in tgt_labels)

    self._convert_sample()

    import numpy as np
    if self.original_label_together:
      np_labels = np.array([t[0] for t in self.targets])
    else:
      np_labels = np.array(self.targets)
    n_tgt = np_labels.sum()
    n_src = np_labels.shape[0] - np_labels.sum()
    print('Src: {} Tgt: {}'.format(n_src, n_tgt))

    if n_src != n_tgt:
      self.logger.info("Different number of images.".format())
      more_label = 1 if n_tgt > n_src else 0
      diff = abs(n_tgt - n_src)

      logical_indices = np_labels == more_label
      indices = np.arange(np_labels.shape[0])[logical_indices]
      selected_indices = sorted(np.random.choice(indices, diff), reverse=True)
      for idx in selected_indices:
        self.logger.info("removed. {}".format(self.imgs[idx]))
        self.samples.pop(idx)
        self.targets.pop(idx)
        self.imgs.pop(idx)
      if self.original_label_together:
        np_labels = np.array([t[0] for t in self.targets])
      else:
        np_labels = np.array(self.targets)
      n_tgt = np_labels.sum()
      n_src = np_labels.shape[0] - np_labels.sum()
      print('Src: {} Tgt: {}'.format(n_src, n_tgt))

  def _convert_sample(self):
    if self.original_label_together:
      self.samples = [(x, (0, y)) if y in self.src_labels else (x, (1, y)) for (i, (x, y)) in enumerate(self.samples) if y in self.src_labels or y in self.tgt_labels]
    else:
      self.samples = [(x, 0) if y in self.src_labels else (x, 1) for (i, (x, y)) in enumerate(self.samples) if y in self.src_labels or y in self.tgt_labels]

    self.targets = [s[1] for s in self.samples]

  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)

    if self.target_transform is not None:
      assert not self.original_label_together

      target = self.target_transform(target)

    return sample, target

