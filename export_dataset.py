import argparse
import cv2
import json
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.signal
import time
import torch
import torch.nn.functional as F
import torchvision

from datetime import datetime
from functools import lru_cache

from torchutils.datasets.PerturbImageFolder import PerturbImageFolder

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def draw_plot_example(data, targets, predictions=None):
  plt.figure()
  for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    img = np.transpose(data[i], (1, 2, 0))
    img = np.clip(img * std + mean, 0, 1)
    plt.imshow(img)
    if predictions is not None:
      plt.title('Ground Truth: {}\n Prediction: {}'.format(targets[i], predictions[i]))
    else:
      plt.title('Ground Truth: {}'.format(targets[i]))
    plt.xticks([])
    plt.yticks([])


### The perturbation code is from https://github.com/hendrycks/robustness
@lru_cache(maxsize=6)
def disk(radius, alias_blur, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    kernel = cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
    conv_kernel = np.zeros((3, 3, *kernel.shape))
    for i in range(3):
        conv_kernel[i][i] = kernel
    conv_kernel = torch.from_numpy(conv_kernel).float()
    conv_kernel = conv_kernel.flip(2).flip(3)
    return conv_kernel


def contrast(image, severity):
  severity = [0.4, .3, .2, .1, .05][severity]
  means = image.mean([1, 2], keepdim=True)
  image = (image - means) * severity + means
  image = image.clamp(0, 1)
  return image


def defocus_blur(image, severity, gpu=False):
  severity = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity]
  kernel = disk(radius=severity[0], alias_blur=severity[1])
  if gpu:
    kernel = kernel.cuda()
  image = F.pad(image.unsqueeze(0), [kernel.size(-1) // 2] * 4, mode='reflect')
  image = F.conv2d(image, kernel)[0]
  image = image.clamp(0, 1)
  return image


def gaussian_noise(img, arg):
  arg = [0.08, 0.12, 0.18, 0.26, 0.38][arg]
  normal = torch.randn_like(img)
  img += normal * arg
  img = img.clamp(0, 1)
  return img


@lru_cache(maxsize=20)
def gaussian_kernel(size, sigma, num_channels=3):
    x = np.linspace(- (size // 2), size // 2, size)
    x = x**2 / (2 * sigma**2)
    kernel = np.exp(- x[:, None] - x[None, :])
    kernel = kernel / kernel.sum()
    conv_kernel = np.zeros((num_channels, num_channels, *kernel.shape))
    for i in range(num_channels):
        conv_kernel[i][i] = kernel
    return torch.from_numpy(conv_kernel).float()


def gaussian_blur_helper(image, size, sigma):
    kernel = gaussian_kernel(size, sigma, num_channels=image.shape[0]).to(image.device).type(image.dtype)
    image = F.pad(image.unsqueeze(0), [kernel.size(-1) // 2] * 4,
                  mode='reflect')
    return F.conv2d(image, kernel)[0]


def gaussian_blur_separated(image, size, sigma):
  """
  >>> image = torch.rand(3, 5, 5)
  >>> expected = gaussian_blur_helper(image, 3, 1)
  >>> real = gaussian_blur_separated(image, 3, 1)
  >>> assert torch.allclose(expected, real), (
  ...     f"Expected:\\n{expected}\\nSaw:\\n{real}")
  """
  kernel_1d = scipy.signal.gaussian(size, sigma)
  kernel_1d /= kernel_1d.sum()
  c = image.shape[0]
  conv1d_x = image.new_zeros(c, c, size, 1)
  for c_i in range(c):
    conv1d_x[c_i, c_i, :, 0] = torch.from_numpy(kernel_1d)
  image = F.pad(image.unsqueeze(0), [size // 2] * 4, mode='reflect')
  image = F.conv2d(image, conv1d_x)
  image = F.conv2d(image, conv1d_x.permute((0, 1, 3, 2)))
  return image[0]


def gaussian_blur(img, arg):
  arg = [1, 2, 3, 4, 6][arg]
  img = gaussian_blur_helper(img, arg * 4 - 1, arg)
  img = img.clamp(0, 1)
  return img


def elastic_transform(image, severity, gpu=False):
  image = image.permute((1, 2, 0))

  image = image.cpu().numpy()
  shape = image.shape
  h, w = shape[:2]

  c = [
    # 244 should have been 224, but ultimately nothing is incorrect
    (244 * 2, 244 * 0.7, 244 * 0.1),
    (244 * 2, 244 * 0.08, 244 * 0.2),
    (244 * 0.05, 244 * 0.01, 244 * 0.02),
    (244 * 0.07, 244 * 0.01, 244 * 0.02),
    (244 * 0.12, 244 * 0.01, 244 * 0.02)
  ][severity]

  # random affine
  center_square = np.float32((h, w)) // 2
  square_size = min((h, w)) // 3
  pts1 = np.float32([
    center_square + square_size,
    [center_square[0] + square_size, center_square[1] - square_size],
    center_square - square_size
  ])
  pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(
    np.float32)
  M = cv2.getAffineTransform(pts1, pts2)
  image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
  image_th = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0)
  if gpu:
    image_th = image_th.cuda()

  # Generate a kernel matching scipy's gaussian filter
  # https://github.com/scipy/scipy/blob/e1e44d12637997606b1bcc0c6de232349e11eee0/scipy/ndimage/filters.py#L214
  sigma = c[1]
  truncate = 3
  radius = min(int(truncate * sigma + 0.5), h)

  deltas = torch.FloatTensor(2, h, w).uniform_(-1, 1)
  if gpu:
    deltas = deltas.cuda()
  deltas = gaussian_blur_separated(deltas, 2 * radius - 1, sigma) * c[0]
  dx, dy = deltas[0], deltas[1]

  dx = dx.squeeze(0).unsqueeze(-1).float()
  dy = dy.squeeze(0).unsqueeze(-1).float()

  # y : [[0, 0, 0, 0], [1, 1, 1, 1], ...]
  # x : [[0, 1, 2, 3], [0, 1, 2, 3], ...]
  y, x = torch.meshgrid(torch.arange(w), torch.arange(h))
  x = x.unsqueeze(-1).to(dx.device).float()
  y = y.unsqueeze(-1).to(dy.device).float()
  indices = torch.stack((x + dx, y + dy), dim=-1)
  indices = indices.permute((2, 0, 1, 3))
  indices[..., 0] = ((indices[..., 0] / h) - 0.5) * 2
  indices[..., 1] = ((indices[..., 1] / w) - 0.5) * 2
  output = F.grid_sample(image_th,
                         indices,
                         mode='bilinear',
                         padding_mode='reflection').clamp(0, 1).squeeze(0)

  return output


def load_data(dataset_dir, batch_size=100, perturb_method=None, arg=0, src_train_idx=None, src_val_idx=None, src_test_idx=None):
  train_dir = os.path.join(dataset_dir, 'train')
  val_dir = os.path.join(dataset_dir, 'val')
  test_dir = os.path.join(dataset_dir, 'test')

  # SRC
  before_transforms = [torchvision.transforms.Resize(256),
                       torchvision.transforms.CenterCrop(224),
                       torchvision.transforms.ToTensor()]
  normalize_transforms = [torchvision.transforms.Normalize(mean=mean, std=std)]

  src_transforms = torchvision.transforms.Compose(before_transforms
                                                  + normalize_transforms)
  # TGT
  tgt_trans_array = before_transforms
  if perturb_method == 'None':
    pass
  elif perturb_method == 'contrast':
    tgt_trans_array += [torchvision.transforms.Lambda(lambda img: contrast(img, arg))]
  elif perturb_method == 'defocus_blur':
    tgt_trans_array += [torchvision.transforms.Lambda(lambda img: defocus_blur(img, arg))]
  elif perturb_method == 'elastic_transform':
    tgt_trans_array += [torchvision.transforms.Lambda(lambda img: elastic_transform(img, arg))]
  elif perturb_method == 'gaussian_noise':
    tgt_trans_array += [torchvision.transforms.Lambda(lambda img: gaussian_noise(img, arg))]
  elif perturb_method == 'gaussian_blur':
    tgt_trans_array += [torchvision.transforms.Lambda(lambda img: gaussian_blur(img, arg))]
  else:
    raise ValueError("Not supported perturbation: {}".format(perturb_method))

  tgt_trans_array += normalize_transforms

  tgt_transforms = torchvision.transforms.Compose(tgt_trans_array)

  train_dataset = PerturbImageFolder(train_dir, transform=src_transforms, perturb_transform=tgt_transforms, src_idx=src_train_idx)
  val_dataset = PerturbImageFolder(val_dir, transform=src_transforms, perturb_transform=tgt_transforms, src_idx=src_val_idx)
  test_dataset = PerturbImageFolder(test_dir, transform=src_transforms, perturb_transform=tgt_transforms, src_idx=src_test_idx)

  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

  src_train_idx = train_dataset.src_idx
  src_val_idx = val_dataset.src_idx
  src_test_idx = test_dataset.src_idx

  return train_loader, val_loader, test_loader, src_train_idx, src_val_idx, src_test_idx


def run_model(model, loader, device):
  model.eval()
  src_outputs = []
  tgt_outputs = []

  with torch.no_grad():
    for idx, (xs, ys) in enumerate(loader):
      print('\r{}/{} exporting'.format(idx + 1, len(loader)), end='')
      xs, ys = xs.to(device), ys.to(device)

      data = model(xs)
      src_outputs.append(data[ys == 0])
      tgt_outputs.append(data[ys == 1])

    src_outputs = torch.cat(src_outputs).cpu().numpy()
    tgt_outputs = torch.cat(tgt_outputs).cpu().numpy()
  return src_outputs, tgt_outputs


def store_output(dataset_name, output_dir, model, loader, device, title):
  src_output, tgt_output = run_model(model, loader, device)
  src_output_fn = os.path.join(output_dir, dataset_name, 'src_{}.npy'.format(title))
  np.save(src_output_fn, src_output)
  tgt_output_fn = os.path.join(output_dir, dataset_name, 'tgt_{}.npy'.format(title))
  np.save(tgt_output_fn, tgt_output)


def get_raw(loader):
  src_raw = []
  tgt_raw = []
  with torch.no_grad():
    for idx, (xs, ys) in enumerate(loader):
      src_raw.append(xs[ys == 0])
      tgt_raw.append(xs[ys == 1])

  src_raw = torch.cat(src_raw).cpu().numpy()
  tgt_raw = torch.cat(tgt_raw).cpu().numpy()
  return src_raw, tgt_raw


def store_raw(dataset_name, output_dir, loader, title):
  src_raw, tgt_raw = get_raw(loader)
  src_raw_fn = os.path.join(output_dir, dataset_name, 'src_{}.npy'.format(title))
  np.save(src_raw_fn, src_raw)
  tgt_raw_fn = os.path.join(output_dir, dataset_name, 'tgt_{}.npy'.format(title))
  np.save(tgt_raw_fn, tgt_raw)


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
  if args.perturb_method is not None and args.perturb_method != 'None':
    assert args.perturb_level > 0, "Perturb Level should be bigger than 0."

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
  logger.info(args)

  #
  model = torchvision.models.resnet152(pretrained=True, progress=True)
  model.fc = torch.nn.Identity()  # To get the input of the last FC layer. Make it as the output
  model = model.to(device)

  src_train_idx = None
  src_val_idx = None
  src_test_idx = None

  logger.info('starting to load data.')
  start_time = time.time()
  train_loader, val_loader, test_loader, src_train_idx, src_val_idx, src_test_idx = load_data(args.dataset_dir,
                                                                                              perturb_method=args.perturb_method,
                                                                                              arg=args.perturb_level - 1,  # zero-based index
                                                                                              src_train_idx=src_train_idx,
                                                                                              src_val_idx=src_val_idx,
                                                                                              src_test_idx=src_test_idx)
  logger.info('finished to load data. took {:.2f} secs'.format(time.time() - start_time))

  if args.show_sample_imgs:
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    draw_plot_example(images.numpy(), labels)
    plt.show()

  #
  if not os.path.exists(os.path.join(args.output_dir, args.dataset_name)):
    os.makedirs(os.path.join(args.output_dir, args.dataset_name))

  if args.feature_type == 'raw':
    store_raw(args.dataset_name, args.output_dir, val_loader, 'val_raw_{}_{}'.format(args.perturb_method, args.perturb_level))
    store_raw(args.dataset_name, args.output_dir, test_loader, 'test_raw_{}_{}'.format(args.perturb_method, args.perturb_level))
  elif args.feature_type == 'features':
    store_output(args.dataset_name, args.output_dir, model, val_loader, device, 'val_features_{}_{}'.format(args.perturb_method, args.perturb_level))
    store_output(args.dataset_name, args.output_dir, model, test_loader, device, 'test_features_{}_{}'.format(args.perturb_method, args.perturb_level))

  src_val_idx_fn = os.path.join(args.output_dir, args.dataset_name, 'src_val_idx_{}_{}.npy'.format(args.perturb_method, args.perturb_level))
  np.save(src_val_idx_fn, np.array(list(src_val_idx)))
  logger.info("stored. {}".format(src_val_idx_fn))
  src_test_idx_fn = os.path.join(args.output_dir, args.dataset_name, 'src_test_idx_{}_{}.npy'.format(args.perturb_method, args.perturb_level))
  np.save(src_test_idx_fn, np.array(list(src_test_idx)))
  logger.info("stored. {}".format(src_test_idx_fn))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset_name', default='imagenet', type=str)
  parser.add_argument('--dataset_dir', default='~/datasets/imagenet', type=str)

  parser.add_argument('--show_sample_imgs', action='store_true')
  parser.set_defaults(show_sample_imgs=False)

  parser.add_argument('--feature_type', default='features', choices=['raw', 'features'])

  parser.add_argument('--perturb_method', default='None')
  parser.add_argument('--perturb_level', default=0, type=int)

  parser.add_argument('--log_dir', default='./logs', type=str)
  parser.add_argument('--output_dir', default='./outputs', type=str)

  parser.add_argument('--seed', default=100)
  parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

  args = parser.parse_args()

  main(args)
