import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from advertorch.attacks import LinfPGDAttack

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def load_data(dataset_dir, batch_size=100, src_val_idx=None, src_test_idx=None):
  val_dir = os.path.join(dataset_dir, 'val')
  test_dir = os.path.join(dataset_dir, 'test')

  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                               torchvision.transforms.CenterCrop(224),
                                               torchvision.transforms.ToTensor()])

  val_dataset = torchvision.datasets.ImageFolder(val_dir, transforms)

  def _get_src_tgt_index(dataset, src_idx):
    n_data = len(dataset)
    indices = list(range(n_data))

    if src_idx is None:
      np.random.shuffle(indices)
      return indices[n_data // 2:], indices[:n_data // 2]
    else:
      return src_idx, list(set(indices).difference(set(src_idx)))

  src_val_idx, tgt_val_idx = _get_src_tgt_index(val_dataset, src_val_idx)

  assert len(set(src_val_idx).intersection(set(tgt_val_idx))) == 0
  assert len(set(src_val_idx).union(set(tgt_val_idx))) == len(val_dataset)

  src_val_dataset = torch.utils.data.Subset(val_dataset, src_val_idx)
  tgt_val_dataset = torch.utils.data.Subset(val_dataset, tgt_val_idx)

  src_val_loader = torch.utils.data.DataLoader(src_val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
  tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

  test_dataset = torchvision.datasets.ImageFolder(test_dir, transforms)
  src_test_idx, tgt_test_idx = _get_src_tgt_index(test_dataset, src_test_idx)

  assert len(set(src_test_idx).intersection(set(tgt_test_idx))) == 0
  assert len(set(src_test_idx).union(set(tgt_test_idx))) == len(test_dataset)

  src_test_dataset = torch.utils.data.Subset(test_dataset, src_test_idx)
  tgt_test_dataset = torch.utils.data.Subset(test_dataset, tgt_test_idx)

  src_test_loader = torch.utils.data.DataLoader(src_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
  tgt_test_loader = torch.utils.data.DataLoader(tgt_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

  return src_val_loader, tgt_val_loader, src_test_loader, tgt_test_loader, src_val_idx, src_test_idx


def run_model(model, loader, device, is_adv=False):
  model.eval()
  correct = 0

  if is_adv:
    for idx, (xs, ys) in enumerate(loader):
      print('\r{}/{} (adv) exporting'.format(idx + 1, len(loader)), end='')
      xs, ys = xs.to(device), ys.to(device)

      adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

      adv_untargeted = adversary.perturb(xs, ys)
      adv_untargeted = TF.normalize(adv_untargeted, mean, std)

      with torch.no_grad():
        data = model(adv_untargeted)
        correct += (data.argmax(axis=1) == ys).float().sum()
  else:
    with torch.no_grad():
      for idx, (xs, ys) in enumerate(loader):
        print('\r{}/{} exporting'.format(idx + 1, len(loader)), end='')
        xs, ys = xs.to(device), ys.to(device)

        xs = TF.normalize(xs, mean, std)

        data = model(xs)
        correct += (data.argmax(axis=1) == ys).float().sum()

  total = len(loader.dataset)

  return correct, total


def main(args):
  print(args)

  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  args.dataset_dir = os.path.expanduser(args.dataset_dir)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  #
  model = torchvision.models.resnet152(pretrained=True, progress=True)
  model = model.to(device)

  src_val_loader, tgt_val_loader, src_test_loader, tgt_test_loader, src_val_idx, src_test_idx = load_data(
    args.dataset_dir)

  correct, total = run_model(model, src_val_loader, device)
  print("SRC VAL: {}/{} = {:.2f}".format(correct, total, correct/total))
  correct, total = run_model(model, tgt_val_loader, device)
  print("TGT VAL: {}/{} = {:.2f}".format(correct, total, correct / total))
  correct, total = run_model(model, src_test_loader, device)
  print("SRC TEST: {}/{} = {:.2f}".format(correct, total, correct / total))
  correct, total = run_model(model, tgt_test_loader, device)
  print("TGT TEST: {}/{} = {:.2f}".format(correct, total, correct / total))

  correct, total = run_model(model, tgt_val_loader, device, is_adv=True)
  print("TGT VAL: {}/{} = {:.2f}".format(correct, total, correct / total))
  correct, total = run_model(model, tgt_test_loader, device, is_adv=True)
  print("TGT TEST: {}/{} = {:.2f}".format(correct, total, correct / total))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset_name', default='imagenet', type=str)
  parser.add_argument('--dataset_dir', default='~/datasets/imagenet', type=str)

  parser.add_argument('--seed', default=100)
  parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

  args = parser.parse_args()

  main(args)
