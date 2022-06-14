import argparse
import csv
import json
import logging.config
import numpy as np
import os
import pickle
import random
import time
import torch
import torchvision

from datetime import datetime

from detect_utils import prepare_fn_single, prepare_ys_fn_single, load_fn, mix_perturb_and_none_prob,\
  prepare_fn_adv_single, prepare_ys_fn_adv_single
from ours_utils import run_test, run_test_holdout


def _is_adv_schedule_file(fn):
  return os.path.basename(fn).startswith('adv')


def load_schedule_file(fn):
  schedules = []
  with open(fn, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
      if len(line) > 3:
        schedules.append((int(line[0]), line[1], int(line[2]), float(line[3])))  # Sample #, type, severity, mix_prob
      else:
        schedules.append((int(line[0]), line[1], int(line[2]), 1.0))  # Sample #, type, severity

  return schedules


def compute_req_samples(schedules):
  analysis_result = []
  assert schedules[-1][1] == 'END', 'The schedule should end with "END" type.'

  for i, (sample_start, p_type, p_severity, mix_prob) in enumerate(schedules[:-1]):
    next_sample_start, _, _, _ = schedules[i+1]
    n_samples = next_sample_start - sample_start

    analysis_result.append((p_type, p_severity, n_samples, mix_prob))

  return analysis_result


def load_req_samples(required_samples, args):
  samples_map = {}
  for p_type, p_severity, _, _ in required_samples:
    samples_map_key = (p_type, p_severity,)
    if samples_map_key in samples_map:
      continue
    if p_type == 'None':
      continue

    if p_type == 'adv':
      eps_arr = [0.003, 0.03, 0.3]
      eps = eps_arr[p_severity - 1]  # 1-> 0.003, 2->0.03, 3->0.3
      src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn = prepare_fn_adv_single(args.root_dir, 'features', eps)
    else:
      src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn = prepare_fn_single(args.root_dir, 'features', p_type, p_severity)

    print(src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn)

    src_val = load_fn(src_val_fn)
    tgt_val = load_fn(tgt_val_fn)
    src_test = load_fn(src_test_fn)
    tgt_test = load_fn(tgt_test_fn)

    samples_map[samples_map_key] = (src_val, tgt_val, src_test, tgt_test)

  # adding none samples
  if _is_adv_schedule_file(args.schedule_file):
    src_val_none_fn, tgt_val_none_fn, src_test_none_fn, tgt_test_none_fn = prepare_fn_adv_single(args.root_dir, 'features', -1)
  else:
    src_val_none_fn, tgt_val_none_fn, src_test_none_fn, tgt_test_none_fn = prepare_fn_single(args.root_dir, 'features', 'None', 0)

  src_val_none = load_fn(src_val_none_fn)
  tgt_val_none = load_fn(tgt_val_none_fn)
  src_test_none = load_fn(src_test_none_fn)
  tgt_test_none = load_fn(tgt_test_none_fn)

  samples_map[('None', 0,)] = (src_val_none, tgt_val_none, src_test_none, tgt_test_none)

  return samples_map


def load_ys(root_dir, schedule_file):
  if _is_adv_schedule_file(schedule_file):
    src_val_ys_fn, tgt_val_ys_fn, src_test_ys_fn, tgt_test_ys_fn = prepare_ys_fn_adv_single(root_dir)
  else:
    src_val_ys_fn, tgt_val_ys_fn, src_test_ys_fn, tgt_test_ys_fn = prepare_ys_fn_single(root_dir)
  src_val_ys = load_fn(src_val_ys_fn).reshape(-1)
  tgt_val_ys = load_fn(tgt_val_ys_fn).reshape(-1)
  src_test_ys = load_fn(src_test_ys_fn).reshape(-1)
  tgt_test_ys = load_fn(tgt_test_ys_fn).reshape(-1)

  ys_map = {'src_val': src_val_ys, 'tgt_val': tgt_val_ys, 'src_test': src_test_ys, 'tgt_test': tgt_test_ys}

  return ys_map


def run_resnet152(model, device, src_sample, tgt_sample, src_ys, tgt_ys, B, W):
  tgt_xs = torch.from_numpy(tgt_sample).to(device)

  tgt_pred = model(tgt_xs).argmax(axis=1)

  tgt_correct = (tgt_pred.cpu().numpy() == tgt_ys).astype(np.int32)

  assert src_sample.shape[0] % B == 0, "Consider only when N is the multiple of B."
  assert src_sample.shape[0] == tgt_sample.shape[0]

  accuracy_history = []
  for t in range(src_sample.shape[0] // B):
    if W > 0:
      start_pos = max(0, (t+1) * B - W)
    else:
      start_pos = 0
    # only interested in target sample accuracy
    accuracy_history.append(tgt_correct[start_pos:(t+1) * B].mean())
  return accuracy_history


def main(args):
  logger.info(args)

  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  if args.CP_window_size > 0 and args.holdout_interval > 0:
    assert args.CP_window_size % args.batch_size == 0, "Window size should be multiple of batch size."

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  schedules = load_schedule_file(args.schedule_file)
  schedules.append((args.n_samples, 'END', -1, 1.0))
  print(schedules)

  required_samples = compute_req_samples(schedules)
  print(required_samples)

  samples_map = load_req_samples(required_samples, args)
  ys_map = load_ys(args.root_dir, args.schedule_file)

  #
  model = torchvision.models.resnet152(pretrained=True, progress=True)
  model = model.to(device)
  model = torch.nn.Sequential(model.fc)
  #

  total_rejects = []
  total_accuracy_histories = []
  start_time = time.time()
  logger.debug('started to sample data')

  indices_arrays = []

  for repeat_idx in range(args.n_repeats):
    indices_array_ = []

    logger.info("[{}/{}] repeat".format(repeat_idx+1, args.n_repeats))
    src_sample = []
    tgt_sample = []
    src_ys = []
    tgt_ys = []

    none_key = ('None', 0,)
    src_val_none, tgt_val_none, src_test_none, tgt_test_none = samples_map[none_key]

    src_val_ys, tgt_val_ys = ys_map['src_val'], ys_map['tgt_val']
    src_test_ys, tgt_test_ys = ys_map['src_test'], ys_map['tgt_test']

    for p_type, p_severity, n_samples, mix_prob in required_samples:
      print(p_type, p_severity, mix_prob)
      src_val, tgt_val, src_test, tgt_test = samples_map[(p_type, p_severity,)]

      if mix_prob == 1.00:
        # val
        src_val_sample_idx = np.random.choice(src_val.shape[0], size=n_samples // 2, replace=False)
        tgt_val_sample_idx = np.random.choice(tgt_val.shape[0], size=n_samples // 2, replace=False)

        sub_src_val_sample = src_val[src_val_sample_idx, :]
        sub_tgt_val_sample = tgt_val[tgt_val_sample_idx, :]

        src_sample.append(sub_src_val_sample)
        tgt_sample.append(sub_tgt_val_sample)

        sub_src_val_y = src_val_ys[src_val_sample_idx]
        sub_tgt_val_y = tgt_val_ys[tgt_val_sample_idx]

        src_ys.append(sub_src_val_y)
        tgt_ys.append(sub_tgt_val_y)

        # test
        src_test_sample_idx = np.random.choice(src_test.shape[0], size=n_samples // 2, replace=False)
        tgt_test_sample_idx = np.random.choice(tgt_test.shape[0], size=n_samples // 2, replace=False)

        sub_src_test_sample = src_test[src_test_sample_idx, :]
        sub_tgt_test_sample = tgt_test[tgt_test_sample_idx, :]

        src_sample.append(sub_src_test_sample)
        tgt_sample.append(sub_tgt_test_sample)

        sub_src_test_y = src_test_ys[src_test_sample_idx]
        sub_tgt_test_y = tgt_test_ys[tgt_test_sample_idx]

        src_ys.append(sub_src_test_y)
        tgt_ys.append(sub_tgt_test_y)

        indices_array_.append((p_type, p_severity, n_samples, mix_prob,
                               src_val_sample_idx, tgt_val_sample_idx,
                               src_test_sample_idx, tgt_test_sample_idx))
      else:
        sub_src_sample, sub_tgt_sample, sub_src_y, sub_tgt_y, indices = mix_perturb_and_none_prob(np.concatenate((src_val, src_test)),
                                                                                                  np.concatenate((tgt_val, tgt_test)),
                                                                                                  np.concatenate((src_val_none, src_test_none)),
                                                                                                  np.concatenate((tgt_val_none, tgt_test_none)),
                                                                                                  n_samples,
                                                                                                  mix_prob,
                                                                                                  src_ys=np.concatenate((src_val_ys, src_test_ys)),
                                                                                                  tgt_ys=np.concatenate((tgt_val_ys, tgt_test_ys)))
        src_sample.append(sub_src_sample)
        tgt_sample.append(sub_tgt_sample)

        src_ys.append(sub_src_y)
        tgt_ys.append(sub_tgt_y)

        indices_array_.append((p_type, p_severity, n_samples, mix_prob,
                              indices))

    indices_arrays.append(indices_array_)

    src_sample = np.concatenate(src_sample).astype(np.float32)
    tgt_sample = np.concatenate(tgt_sample).astype(np.float32)

    src_ys = np.concatenate(src_ys).astype(np.float32)
    tgt_ys = np.concatenate(tgt_ys).astype(np.float32)

    if args.holdout_interval < 2:
      reject, _ = run_test(src_sample, tgt_sample, args.alpha, device,
                           batch_size=args.batch_size,
                           lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           holdout_interval=args.holdout_interval,
                           CP_window_size=args.CP_window_size,
                           scheduled=True)
    else:
      reject, _ = run_test_holdout(src_sample, tgt_sample, args.alpha, device,
                                   batch_size=args.batch_size,
                                   lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay,
                                   holdout_interval=args.holdout_interval,
                                   CP_window_size=args.CP_window_size,
                                   scheduled=True)

    total_rejects.append(reject)

    accuracy_history = run_resnet152(model, device, src_sample, tgt_sample, src_ys, tgt_ys,
                                     args.batch_size,
                                     args.CP_window_size)
    total_accuracy_histories.append(accuracy_history)

  total_rejects_array = np.array(total_rejects, dtype=np.int32)
  total_accuracy_histories_array = np.array(total_accuracy_histories, dtype=np.float32)

  logger.info('*** Test Summaries ***')
  logger.info('Rejects: {}'.format(total_rejects_array))
  logger.info('took {:.2f} secs'.format(time.time() - start_time))
  res_dict = {'PerturbRatio': args.perturb_ratio,
              'batch_size': args.batch_size,
              'alpha': args.alpha,
              'n_repeats': args.n_repeats,
              'n_samples': args.n_samples,
              'dataset_size': src_sample.shape[0] + tgt_sample.shape[0],
              'Schedule': args.schedule_file,
              'learning_rate': args.lr
              }

  ###
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

  result_fn = os.path.join(args.result_dir, 'result_ours_sch.csv')
  header = ['PerturbRatio', 'batch_size', 'alpha', 'n_samples', 'n_repeats',
            'dataset_size', 'Schedule', 'learning_rate']

  if not os.path.exists(result_fn):
    with open(result_fn, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(header)

  with open(result_fn, 'a') as f:
    writer = csv.DictWriter(f, header)
    writer.writerow(res_dict)

  sch_name = os.path.basename(args.schedule_file)
  sch_name = os.path.splitext(sch_name)[0]

  ###
  indices_fn = 'indices_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{:}.pkl'.format(sch_name,
                                                                                        args.holdout_interval,
                                                                                        args.alpha,
                                                                                        args.n_samples,
                                                                                        args.n_repeats,
                                                                                        args.CP_window_size,
                                                                                        args.lr,
                                                                                        args.seed)
  with open(indices_fn, 'wb') as f:
    pickle.dump(indices_arrays, f)
  logger.info("stored {}".format(indices_fn))

  total_reject_fn = os.path.join(args.result_dir, 'result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{:}.csv'.format(sch_name,
                                                                                                                          args.holdout_interval,
                                                                                                                          args.alpha,
                                                                                                                          args.n_samples,
                                                                                                                          args.n_repeats,
                                                                                                                          args.CP_window_size,
                                                                                                                          args.lr,
                                                                                                                          args.seed))
  np.savetxt(total_reject_fn, total_rejects_array, fmt='%i', delimiter=',')
  logger.info("stored {}".format(total_reject_fn))

  total_acc_histories_fn = os.path.join(args.result_dir,
                                        'result_acc_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{:}.csv'.format(sch_name,
                                                                                                                    args.holdout_interval,
                                                                                                                    args.alpha,
                                                                                                                    args.n_samples,
                                                                                                                    args.n_repeats,
                                                                                                                    args.CP_window_size,
                                                                                                                    args.lr,
                                                                                                                    args.seed))
  np.savetxt(total_acc_histories_fn, total_accuracy_histories_array, delimiter=',')
  logger.info("stored {}".format(total_acc_histories_fn))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--root_dir', default='./test_sample/')
  parser.add_argument('--schedule_file', default='schedules/schedule.txt')

  parser.add_argument('--holdout_interval', default=-1, type=int,
                      help='H')

  parser.add_argument('--CP_window_size', default=-1, type=int,
                      help='Number of samples in the window (W = w * bach_size (B))')

  parser.add_argument('--n_samples', default=10000, type=int,
                      help='Number of samples (m)')
  parser.add_argument('--perturb_ratio', default=1.0, type=float)

  parser.add_argument('--batch_size', default=1, type=int,
                      help='Batch size (B)')
  parser.add_argument('--lr', default=0.1, type=float)
  parser.add_argument('--momentum', default=0.9, type=float)
  parser.add_argument('--weight_decay', default=1e-4, type=float)

  parser.add_argument('--n_repeats', default=100, type=int,
                      help='Number of repetitions (R)')

  parser.add_argument('--log_dir', default='./logs', type=str)
  parser.add_argument('--result_dir', default='./results', type=str)

  parser.add_argument('--alpha', default=0.01, type=float)

  parser.add_argument('--seed', default=100, type=int)
  parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

  args = parser.parse_args()

  #
  with open('detect_logging.json', 'rt') as f:
    config = json.load(f)

  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  date_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
  config['handlers']['file_handler']['filename'] = os.path.join(args.log_dir, 'detect_schedule_log_{}.log'.format(date_str))
  logging.config.dictConfig(config)
  logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
  #

  main(args)
