import argparse
from datetime import datetime
import numpy as np
import json
import logging.config
import os
import random
import time
import torch
import torchvision

from torchutils.datasets.NaturalShiftFolder import NaturalShiftFolder

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def read_labels(fn):
  with open(fn, 'r') as f:
    return [x.rstrip() for x in f.readlines()]


def load_data(dataset_dir, src_labels, tgt_labels, batch_size=100):
  val_dir = os.path.join(dataset_dir, 'val')
  test_dir = os.path.join(dataset_dir, 'test')

  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                               torchvision.transforms.CenterCrop(224),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=mean, std=std)])

  val_dataset = NaturalShiftFolder(val_dir, transform=transforms, src_labels=src_labels, tgt_labels=tgt_labels, original_label_together=True)
  test_dataset = NaturalShiftFolder(test_dir, transform=transforms, src_labels=src_labels, tgt_labels=tgt_labels, original_label_together=True)

  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

  return val_loader, test_loader


def run_model(model, loader, device, comb_y):
  model.eval()
  src_outputs = []
  tgt_outputs = []

  src_ys = []
  tgt_ys = []
  with torch.no_grad():
    for idx, (xs, ys) in enumerate(loader):
      print('\r{}/{} exporting'.format(idx + 1, len(loader)), end='')
      xs = xs.to(device)
      if comb_y:
        ys[0] = ys[0].to(device)
        ys[1] = ys[1].to(device)
      else:
        ys = ys.to(device)

      data = model(xs)

      if comb_y:
        src_outputs.append(data[ys[0] == 0])
        tgt_outputs.append(data[ys[0] == 1])

        src_ys.append(ys[1][ys[0] == 0])
        tgt_ys.append(ys[1][ys[0] == 1])
      else:
        src_outputs.append(data[ys == 0])
        tgt_outputs.append(data[ys == 1])

    src_outputs = torch.cat(src_outputs).cpu().numpy()
    tgt_outputs = torch.cat(tgt_outputs).cpu().numpy()

    if comb_y:
      src_ys = torch.cat(src_ys).cpu().numpy()
      tgt_ys = torch.cat(tgt_ys).cpu().numpy()
    else:
      src_ys = None
      tgt_ys = None
  return src_outputs, tgt_outputs, src_ys, tgt_ys


def store_output(dataset_name, output_dir, model, loader, device, title):
  src_output, tgt_output, src_ys, tgt_ys = run_model(model, loader, device, comb_y=True)
  src_output_fn = os.path.join(output_dir, dataset_name, 'src_{}.npy'.format(title))
  np.save(src_output_fn, src_output)
  tgt_output_fn = os.path.join(output_dir, dataset_name, 'tgt_{}.npy'.format(title))
  np.save(tgt_output_fn, tgt_output)

  if src_ys is not None:
    src_ys_fn = os.path.join(output_dir, dataset_name, 'src_{}_ys.npy'.format(title))
    np.save(src_ys_fn, src_ys)
  if tgt_ys is not None:
    tgt_ys_fn = os.path.join(output_dir, dataset_name, 'tgt_{}_ys.npy'.format(title))
    np.save(tgt_ys_fn, tgt_ys)


def main(args):
  print(args)
  dataset_name = "imagenet_dogs"

  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  #
  model = torchvision.models.resnet152(pretrained=True, progress=True)
  model.fc = torch.nn.Identity()  # To get the input of the last FC layer. Make it as the output
  model = model.to(device)

  logger.info('starting to load data.')
  start_time = time.time()
  dogs_labels = read_labels(args.label_fn)
  np.random.shuffle(dogs_labels)
  middle_idx = int(len(dogs_labels) / 2)
  src_labels = dogs_labels[:middle_idx]
  tgt_labels = dogs_labels[middle_idx:]

  val_loader, test_loader = load_data(args.dataset_dir, src_labels=src_labels, tgt_labels=tgt_labels)
  logger.info('finished to load data. took {:.2f} secs'.format(time.time() - start_time))

  if not os.path.exists(os.path.join(args.output_dir, dataset_name)):
    os.makedirs(os.path.join(args.output_dir, dataset_name))

  store_output(dataset_name, args.output_dir, model, val_loader, device, 'val_features')
  store_output(dataset_name, args.output_dir, model, test_loader, device, 'test_features')

  src_labels_fn = os.path.join(args.output_dir, dataset_name, 'src_labels.txt')
  with open(src_labels_fn, 'w') as lbl_f:
    lbl_f.writelines('\n'.join(list(src_labels)))
  logger.info("stored. {}".format(src_labels_fn))
  tgt_labels_fn = os.path.join(args.output_dir, dataset_name, 'tgt_labels.txt')
  with open(tgt_labels_fn, 'w') as lbl_f:
    lbl_f.writelines('\n'.join(list(tgt_labels)))
  logger.info("stored. {}".format(tgt_labels_fn))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset_dir', default='~/datasets/imagenet', type=str)

  parser.add_argument('--label_fn', default='./etc_codes/dogs_labels.txt', type=str)

  parser.add_argument('--log_dir', default='./logs', type=str)
  parser.add_argument('--output_dir', default='./outputs', type=str)

  parser.add_argument('--seed', default=100, type=int)
  parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

  args = parser.parse_args()

  args.dataset_dir = os.path.expanduser(args.dataset_dir)
  args.log_dir = os.path.expanduser(args.log_dir)
  args.output_dir = os.path.expanduser(args.output_dir)
  args.label_fn = os.path.expanduser(args.label_fn)

  #
  with open('logging.json', 'rt') as f:
    config = json.load(f)

  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  date_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
  config['handlers']['file_handler']['filename'] = os.path.join(args.log_dir, 'detect_log_{}.log'.format(date_str))
  logging.config.dictConfig(config)
  logger = logging.getLogger()

  #

  main(args)
