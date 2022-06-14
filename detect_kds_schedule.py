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

from detect_utils import prepare_fn_single, load_fn, prepare_fn_adv_single
from KDSwitchTest.KDSwitchTest import KDSwitchTest


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


def load_indices_file(sch_name, alpha, n_samples, R, W, seed=100):
  if seed == 100:
    # seed =100 was default.
    fn = 'indices_pkls/indices_ours_sch{}_H-1_alpha{}_M{}_R{}_W{}_lr0.0010.pkl'.format(sch_name, alpha, n_samples, R, W)
  else:
    fn = 'indices_pkls/indices_ours_sch{}_H-1_alpha{}_M{}_R{}_W{}_lr0.0010_seed{}.pkl'.format(sch_name,
                                                                                              alpha,
                                                                                              n_samples,
                                                                                              R,
                                                                                              W,
                                                                                              seed)

  with open(fn, 'rb') as f:
    d = pickle.load(f)

  assert len(d) == R, 'Number of repeats: {}'.format(R)

  return d


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


def main(args):
  print(args)

  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  schedules = load_schedule_file(args.schedule_file)
  schedules.append((args.n_samples, 'END', -1, 1.0))
  print(schedules)

  required_samples = compute_req_samples(schedules)
  print(required_samples)

  samples_map = load_req_samples(required_samples, args)

  sch_name = os.path.basename(args.schedule_file)
  sch_name = os.path.splitext(sch_name)[0]

  indices = load_indices_file(sch_name, args.alpha, args.n_samples, args.n_repeats, args.CP_window_size, args.seed)

  total_rejects = []

  start_time = time.time()
  for repeat_idx in range(args.n_repeats):
    logger.info("[{}/{}] repeat".format(repeat_idx+1, args.n_repeats))

    ind = indices[repeat_idx]

    src_sample = []
    tgt_sample = []

    rejects = []

    none_key = ('None', 0,)
    src_val_none, tgt_val_none, src_test_none, tgt_test_none = samples_map[none_key]
    src_none = np.concatenate((src_val_none, src_test_none))
    tgt_none = np.concatenate((tgt_val_none, tgt_test_none))

    for sch_i, (p_type, p_severity, n_samples, mix_prob) in enumerate(required_samples):
      print(p_type, p_severity, mix_prob)
      src_val, tgt_val, src_test, tgt_test = samples_map[(p_type, p_severity,)]

      assert ind[sch_i][0] == p_type
      assert ind[sch_i][1] == p_severity
      assert ind[sch_i][2] == n_samples
      assert ind[sch_i][3] == mix_prob

      if mix_prob == 1.0:
        # val
        sub_src_val_sample = src_val[ind[sch_i][4], :]
        sub_tgt_val_sample = tgt_val[ind[sch_i][5], :]

        src_sample.append(sub_src_val_sample)
        tgt_sample.append(sub_tgt_val_sample)

        # test
        sub_src_test_sample = src_test[ind[sch_i][6], :]
        sub_tgt_test_sample = tgt_test[ind[sch_i][7], :]

        src_sample.append(sub_src_test_sample)
        tgt_sample.append(sub_tgt_test_sample)
      else:
        src = np.concatenate((src_val, src_test))
        tgt = np.concatenate((tgt_val, tgt_test))

        sub_src_sample = np.zeros((n_samples, src_val.shape[1]), dtype=np.float32)
        sub_src_sample[ind[sch_i][4][0], :] = src[ind[sch_i][4][1], :]
        sub_src_sample[~ind[sch_i][4][0], :] = src_none[ind[sch_i][4][2], :]

        sub_tgt_sample = np.zeros((n_samples, tgt_val.shape[1]), dtype=np.float32)
        sub_tgt_sample[ind[sch_i][4][0], :] = tgt[ind[sch_i][4][3], :]
        sub_tgt_sample[~ind[sch_i][4][0], :] = tgt_none[ind[sch_i][4][4], :]

        src_sample.append(sub_src_sample)
        tgt_sample.append(sub_tgt_sample)

    src_sample = np.concatenate(src_sample).astype(np.float32)
    tgt_sample = np.concatenate(tgt_sample).astype(np.float32)

    assert src_sample.shape[0] == args.n_samples
    assert tgt_sample.shape[0] == args.n_samples

    # KDS
    kdswitch_test = KDSwitchTest(model=args.model, dim=src_sample.shape[1], alpha=args.alpha)

    src_batched_sample = src_sample.reshape((-1, args.batch_size, src_sample.shape[-1]))
    tgt_batched_sample = tgt_sample.reshape((-1, args.batch_size, tgt_sample.shape[-1]))

    for i in range(src_batched_sample.shape[0]):
      if i == 0:
        rejects.append(0)
        continue

      test_src_sample = src_batched_sample[i]
      test_tgt_sample = tgt_batched_sample[i]

      reject, p_value = kdswitch_test.offer(test_src_sample, test_tgt_sample)
      print('[{}/{}] {}: P-value: {}'.format(i,
                                             tgt_sample.shape[0] // args.batch_size,
                                             'REJECT' if reject == 1 else 'ACCEPT',
                                             '{:.4f}'.format(p_value)))

      rejects.append(reject)

      if reject == 1:
        kdswitch_test.reset()

    total_rejects.append(rejects)

  total_rejects_array = np.array(total_rejects, dtype=np.int32)
  logger.info('*** Test Summaries ***')
  logger.info('Rejects: {}'.format(total_rejects_array))
  logger.info('took {:.2f} secs'.format(time.time() - start_time))

  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

  total_reject_fn = os.path.join(args.result_dir,
                                 'result_kds_{}_sch{}_alpha{}_M{}_R{}_W{}_seed{}.csv'.format(args.model,
                                                                                             sch_name,
                                                                                             args.alpha,
                                                                                             args.n_samples,
                                                                                             args.n_repeats,
                                                                                             args.CP_window_size,
                                                                                             args.seed))
  np.savetxt(total_reject_fn, total_rejects_array, fmt='%i', delimiter=',')
  logger.info("stored {}".format(total_reject_fn))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--root_dir', default='./test_sample/')
  parser.add_argument('--schedule_file', default='schedules/contrast2_c3.txt')

  parser.add_argument('--CP_window_size', default=100, type=int,
                      help='Number of samples in the window (W = w * bach_size (B))')

  parser.add_argument('--n_samples', default=10000, type=int,
                      help='Number of samples (m)')
  parser.add_argument('--perturb_ratio', default=1.0, type=float)

  # KDSwitchTest
  parser.add_argument('--model', default='kds', choices=['kds', 'knn'])

  parser.add_argument('--batch_size', default=10, type=int,
                      help='Batch size (B)')
  parser.add_argument('--lr', default=0.00001, type=float)

  parser.add_argument('--n_repeats', default=100, type=int,
                      help='Number of repetitions (R)')

  parser.add_argument('--holdout_interval', default=2, type=int)

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
  config['handlers']['file_handler']['filename'] = os.path.join(args.log_dir, 'dk_log_{}.log'.format(date_str))
  logging.config.dictConfig(config)
  logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
  #

  main(args)
