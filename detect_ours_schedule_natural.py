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

from datetime import datetime

from detect_utils import prepare_fn_natural, load_fn, mix_perturb_and_none_prob
from ours_utils import run_test, run_test_holdout


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


def load_req_samples(args):
  src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn = prepare_fn_natural(args.root_dir, 'features')

  src_val = load_fn(src_val_fn)
  tgt_val = load_fn(tgt_val_fn)
  src_test = load_fn(src_test_fn)
  tgt_test = load_fn(tgt_test_fn)

  return src_val, tgt_val, src_test, tgt_test


def extract_notnone_none(src_total):
  indices = np.random.choice(src_total.shape[0], size=src_total.shape[0], replace=False)
  notnone_indices = indices[:src_total.shape[0] // 2]
  none_indices = indices[src_total.shape[0] // 2:]

  return src_total[notnone_indices, :], src_total[none_indices, :]


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

  src_val_all, tgt_val_all, src_test_all, tgt_test_all = load_req_samples(args)

  total_rejects = []

  indices_arrays = []

  start_time = time.time()
  logger.debug('started to sample data')
  for repeat_idx in range(args.n_repeats):
    indices_array_ = []

    logger.info("[{}/{}] repeat".format(repeat_idx+1, args.n_repeats))

    src_sample = []
    tgt_sample = []

    # for dogs, tgt_none is from src, src_none is src
    src_val, tgt_val_none = extract_notnone_none(src_val_all)
    src_val_none = src_val
    src_test, tgt_test_none = extract_notnone_none(src_test_all)
    src_test_none = src_test

    # to make the same number of samples with src
    tgt_val, _ = extract_notnone_none(tgt_val_all)
    tgt_test, _ = extract_notnone_none(tgt_test_all)

    for p_type, p_severity, n_samples, mix_prob in required_samples:
      print(p_type, p_severity, mix_prob)
      # src_val, tgt_val, src_test, tgt_test = samples_map[(p_type, p_severity,)]

      if p_type == 'None':
        src_val_iter = src_val_none
        tgt_val_iter = tgt_val_none

        src_test_iter = src_test_none
        tgt_test_iter = tgt_test_none
      else:
        src_val_iter = src_val
        tgt_val_iter = tgt_val

        src_test_iter = src_test
        tgt_test_iter = tgt_test

      if mix_prob == 1.00:
        # mixed val and test
        mixed_src = np.zeros((n_samples, src_val_iter.shape[1]), dtype=np.float32)
        mixed_tgt = np.zeros((n_samples, tgt_val_iter.shape[1]), dtype=np.float32)
        val_indices = np.zeros(n_samples, dtype=np.bool)
        val_indices[:n_samples // 2] = True
        np.random.shuffle(val_indices)

        # val
        src_val_sample_idx = np.random.choice(src_val_iter.shape[0], size=n_samples // 2, replace=False)
        tgt_val_sample_idx = np.random.choice(tgt_val_iter.shape[0], size=n_samples // 2, replace=False)

        sub_src_val_sample = src_val_iter[src_val_sample_idx, :]
        sub_tgt_val_sample = tgt_val_iter[tgt_val_sample_idx, :]

        mixed_src[val_indices, :] = sub_src_val_sample
        mixed_tgt[val_indices, :] = sub_tgt_val_sample

        # test
        src_test_sample_idx = np.random.choice(src_test_iter.shape[0], size=n_samples // 2, replace=False)
        tgt_test_sample_idx = np.random.choice(tgt_test_iter.shape[0], size=n_samples // 2, replace=False)

        sub_src_test_sample = src_test_iter[src_test_sample_idx, :]
        sub_tgt_test_sample = tgt_test_iter[tgt_test_sample_idx, :]

        mixed_src[~val_indices, :] = sub_src_test_sample
        mixed_tgt[~val_indices, :] = sub_tgt_test_sample

        src_sample.append(mixed_src)
        tgt_sample.append(mixed_tgt)

        indices_array_.append((p_type, p_severity, n_samples, mix_prob,
                               src_val_sample_idx, tgt_val_sample_idx,
                               src_test_sample_idx, tgt_test_sample_idx,
                               val_indices))

      else:
        sub_src_sample, sub_tgt_sample, _, _, indices = mix_perturb_and_none_prob(np.concatenate((src_val, src_test)),
                                                                                  np.concatenate((tgt_val, tgt_test)),
                                                                                  np.concatenate((src_val_none, src_test_none)),
                                                                                  np.concatenate((tgt_val_none, tgt_test_none)),
                                                                                  n_samples,
                                                                                  mix_prob)

        src_sample.append(sub_src_sample)
        tgt_sample.append(sub_tgt_sample)

        indices_array_.append((p_type, p_severity, n_samples, mix_prob,
                              indices))

    indices_arrays.append(indices_array_)

    src_sample = np.concatenate(src_sample).astype(np.float32)
    tgt_sample = np.concatenate(tgt_sample).astype(np.float32)

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

  total_rejects_array = np.array(total_rejects, dtype=np.int32)

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--root_dir', default='./outputs/iwildcam/test_samples')
  parser.add_argument('--schedule_file', default='schedules/iwildcam_c3.txt')

  parser.add_argument('--holdout_interval', default=-1, type=int,
                      help='H')

  parser.add_argument('--CP_window_size', default=-1, type=int,
                      help='Number of samples in the window (W = w * bach_size (B))')

  parser.add_argument('--n_samples', default=6000, type=int,
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
  config['handlers']['file_handler']['filename'] = os.path.join(args.log_dir, 'detect_schedule_natural_{}.log'.format(date_str))
  logging.config.dictConfig(config)
  logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
  #

  main(args)
