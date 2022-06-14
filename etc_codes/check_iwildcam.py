import argparse
import numpy as np
import os
import random
import torch
import torchvision


def load_model(model_file, n_classes, device):
  model_dict = torch.load(model_file, map_location=device)
  model_weight_dict = {x.replace("model.", ""): model_dict['algorithm'][x] for x in model_dict['algorithm']}

  ori_model = torchvision.models.resnet50(pretrained=True, progress=True)
  ori_model.fc = torch.nn.Linear(ori_model.fc.in_features, n_classes)
  ori_model.load_state_dict(model_weight_dict)

  return torch.nn.Sequential(ori_model.fc)


def prepare_fn(root_dir, val_test):
  file_fn = os.path.join(root_dir, '{}_{}_{}.npy')

  src_val_feat_fn = file_fn.format('src', val_test, 'features')
  src_val_y_fn = file_fn.format('src', val_test, 'ys')
  tgt_feat_fn = file_fn.format('tgt', val_test, 'features')
  tgt_y_fn = file_fn.format('tgt', val_test, 'ys')

  return src_val_feat_fn, src_val_y_fn, tgt_feat_fn, tgt_y_fn


def load_files(root_dir, val_test):
  src_feat_fn, src_y_fn, tgt_feat_fn, tgt_y_fn = prepare_fn(root_dir, val_test)
  print(src_feat_fn, src_y_fn, tgt_feat_fn, tgt_y_fn)

  src_feat = np.load(src_feat_fn)
  src_ys = np.load(src_y_fn)
  tgt_feat = np.load(tgt_feat_fn)
  tgt_ys = np.load(tgt_y_fn)

  print(src_feat_fn, tgt_feat_fn, src_feat.shape, tgt_feat.shape, src_ys.shape, tgt_ys.shape)

  return src_feat, src_ys, tgt_feat, tgt_ys


def load_data(root_dir):
  src_val_feat, src_val_ys, tgt_val_feat, tgt_val_ys = load_files(root_dir, 'val')
  src_val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(src_val_feat), torch.from_numpy(src_val_ys))
  tgt_val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(tgt_val_feat), torch.from_numpy(tgt_val_ys))

  src_val_loader = torch.utils.data.DataLoader(src_val_dataset, batch_size=100)
  tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=100)

  src_test_feat, src_test_ys, tgt_test_feat, tgt_test_ys = load_files(root_dir, 'test')
  src_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(src_test_feat), torch.from_numpy(src_test_ys))
  tgt_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(tgt_test_feat), torch.from_numpy(tgt_test_ys))

  src_test_loader = torch.utils.data.DataLoader(src_test_dataset, batch_size=100)
  tgt_test_loader = torch.utils.data.DataLoader(tgt_test_dataset, batch_size=100)

  return src_val_loader, tgt_val_loader, src_test_loader, tgt_test_loader


def run_model(model, loader, device):
  correct = 0
  n_total = 0
  for i, (xs, ys) in enumerate(loader):
    print("\r{}/{} - processing".format(i + 1, len(loader)), end='')
    if i > 0:
      break
    xs, ys = xs.to(device), ys.to(device)

    output = model(xs)

    correct += (output.argmax(dim=1) == ys).sum().item()
    n_total += ys.shape[0]

  print()
  return correct, n_total


def main(args):
  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  print(args)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  src_val_loader, tgt_val_loader, src_test_loader, tgt_test_loader = load_data(args.root_dir)

  model = load_model(args.model_file, n_classes=182, device=device)
  model = model.to(device)

  src_val_cor, src_val_total = run_model(model, src_val_loader, device)
  tgt_val_cor, tgt_val_total = run_model(model, tgt_val_loader, device)
  print("Src Val: {}/{} = {:.2f} %".format(src_val_cor, src_val_total, 100.0 * src_val_cor / src_val_total))
  print("Tgt Val: {}/{} = {:.2f} %".format(tgt_val_cor, tgt_val_total, 100.0 * tgt_val_cor / tgt_val_total))

  src_test_cor, src_test_total = run_model(model, src_test_loader, device)
  tgt_test_cor, tgt_test_total = run_model(model, tgt_test_loader, device)
  print("Src Test: {}/{} = {:.2f} %".format(src_test_cor, src_test_total, 100.0 * src_test_cor / src_test_total))
  print("Tgt Test: {}/{} = {:.2f} %".format(tgt_test_cor, tgt_test_total, 100.0 * tgt_test_cor / tgt_test_total))




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_dir', default='./outputs/iwildcam', type=str)
  parser.add_argument('--model_file', default='./iwildcam_seed:0_epoch:best_model.pth', type=str)

  parser.add_argument('--seed', default=100, type=int)
  parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])


  args = parser.parse_args()
  main(args)
