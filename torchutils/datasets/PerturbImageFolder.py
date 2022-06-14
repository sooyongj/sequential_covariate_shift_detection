import logging
import torchvision

from torchvision.datasets.folder import default_loader
import numpy as np


class PerturbImageFolder(torchvision.datasets.ImageFolder):
  def __init__(self, root,
               transform=None,
               target_transform=None,
               perturb_transform=None,
               src_idx=None,
               original_label_together=False,
               loader=default_loader,
               is_valid_file=None):
    super(PerturbImageFolder, self).__init__(root,
                                             transform=transform,
                                             target_transform=target_transform,
                                             loader=loader,
                                             is_valid_file=is_valid_file)

    self.original_label_together = original_label_together
    self.perturb_transform = perturb_transform
    self.logger = logging.getLogger(__name__)

    if len(self.samples) % 2 == 1:
      self.logger.info("Odd number of images. Dropped image: {}".format(self.samples[-1]))
      self.samples = self.samples[:-1]
      self.targets = self.targets[:-1]
      self.imgs = self.imgs[:-1]

    if src_idx is None:
      n_data = len(self)
      indices = list(range(n_data))
      np.random.shuffle(indices)
      self.src_idx = set(indices[n_data // 2:])
    else:
      self.src_idx = src_idx

    self._convert_sample()
    print('Src: {} Tgt: {}'.format(len(self.src_idx), len(self) - len(self.src_idx)))

  def _convert_sample(self):
    if self.original_label_together:
      self.samples = [(x, (0, y)) if i in self.src_idx else (x, (1, y)) for (i, (x, y)) in enumerate(self.samples)]
    else:
      self.samples = [(x, 0) if i in self.src_idx else (x, 1) for (i, (x, y)) in enumerate(self.samples)]

    self.targets = [s[1] for s in self.samples]

  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      if index in self.src_idx:
        sample = self.transform(sample)
      else:
        sample = self.perturb_transform(sample)

    if self.target_transform is not None:
      assert not self.original_label_together

      target = self.target_transform(target)

    return sample, target
