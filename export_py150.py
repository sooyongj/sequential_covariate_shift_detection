import argparse
import numpy as np
import os
import random
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader


def load_dataset(root_dir, val_test, batch_size):
  dataset = get_dataset(dataset="py150", root_dir=root_dir, download=True)

  ood_data = dataset.get_subset(
    val_test,
  )

  id_data = dataset.get_subset(
    "id_" + val_test,
  )

  id_loader = get_train_loader("standard", id_data, batch_size=batch_size)
  ood_loader = get_train_loader("standard", ood_data, batch_size=batch_size)

  return id_loader, ood_loader


class Featurizer(torch.nn.Module):
  def __init__(self, native_transformer):
    super().__init__()
    self.native_transformer = native_transformer

  def __call__(self, x):
    outputs = self.native_transformer(x)
    hidden_states = outputs[0]
    return hidden_states


def load_model(model_file, n_classes, device):
  model_dict = torch.load(model_file, map_location=device)
  model_weight_dict = {x.replace("model.", ""): model_dict['algorithm'][x] for x in model_dict['algorithm']}

  name = 'microsoft/CodeGPT-small-py'
  tokenizer = GPT2Tokenizer.from_pretrained(name)
  ori_model = GPT2LMHeadModel.from_pretrained(name)
  ori_model.resize_token_embeddings(len(tokenizer))
  ori_model.load_state_dict(model_weight_dict)
  ori_model = ori_model.to(device)
  model = Featurizer(ori_model.transformer)
  model = model.to(device)

  return model, ori_model


def store_output(dataset_name, output_dir, model, ori_model, src_loader, tgt_loader, device, title, title_ys):
  src_output, src_ys = run_model(model, ori_model, src_loader, device)

  final_output_dir = os.path.join(output_dir, dataset_name)
  if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

  src_output_fn = os.path.join(output_dir, dataset_name, 'src_{}.npy'.format(title))
  np.save(src_output_fn, src_output)

  src_ys_fn = os.path.join(output_dir, dataset_name, 'src_{}.npy'.format(title_ys))
  np.save(src_ys_fn, src_ys)

  tgt_output, tgt_ys = run_model(model, ori_model, tgt_loader, device)
  tgt_output_fn = os.path.join(output_dir, dataset_name, 'tgt_{}.npy'.format(title))
  np.save(tgt_output_fn, tgt_output)

  tgt_ys_fn = os.path.join(output_dir, dataset_name, 'tgt_{}.npy'.format(title_ys))
  np.save(tgt_ys_fn, tgt_ys)


def run_model(model, ori_model, loader, device):
  model.eval()
  ori_model.eval()

  cor, tot = 0, 0
  feats = []
  all_ys = []

  with torch.no_grad():
    for i, (xs, ys, _) in enumerate(loader):
      print("\r{}/{}".format(i + 1, len(loader)), end='')

      xs, ys = xs.to(device), ys.to(device)
      all_ys.append(ys)

      feat = model(xs)

      # embedding - average
      feat = feat.mean(dim=1)
      feats.append(feat)

      output = ori_model(xs).logits

      eval_pos = ~torch.isnan(ys)

      cor += (output[eval_pos].argmax(dim=-1) == ys[eval_pos]).sum().item()
      tot += ys[eval_pos].shape[0]
  print()
  print("{}/{} = {:.2f} %".format(cor, tot, cor / tot * 100))

  feats = torch.cat(feats).cpu().numpy()
  all_ys = torch.cat(all_ys).cpu().numpy()
  return feats, all_ys


def main(args):
  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  print(args)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  id_val_loader, ood_val_loader = load_dataset(root_dir=args.root_dir, val_test='val', batch_size=args.batch_size)
  id_test_loader, ood_test_loader = load_dataset(root_dir=args.root_dir, val_test='test', batch_size=args.batch_size)

  n_classes = id_test_loader.dataset.n_classes

  model, ori_model = load_model(args.model_file, n_classes, device)

  store_output('py150', args.output_dir, model, ori_model, id_val_loader, ood_val_loader, device,
               "val_features", "val_ys")
  store_output('py150', args.output_dir, model, ori_model, id_test_loader, ood_test_loader, device,
               "test_features", "test_ys")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_dir', default='~/datasets/wilds', type=str)
  parser.add_argument('--model_file', default='./py150_seed:0_epoch:best_model.pth', type=str)

  parser.add_argument('--show_sample_imgs', action='store_true')
  parser.set_defaults(show_sample_imgs=False)

  parser.add_argument('--output_dir', default='./outputs', type=str)

  parser.add_argument('--batch_size', default=6, type=int)

  parser.add_argument('--seed', default=100, type=int)
  parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

  args = parser.parse_args()

  args.root_dir = os.path.expanduser(args.root_dir)
  args.model_file = os.path.expanduser(args.model_file)

  main(args)
