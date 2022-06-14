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
from ICM.ICMDetection import ICMDetection


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
  logger.info('Loading {}'.format(fn))

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
      src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn = prepare_fn_single(args.root_dir, 'features', p_type,
                                                                           p_severity)

    print(src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn)

    src_val = load_fn(src_val_fn)
    tgt_val = load_fn(tgt_val_fn)
    src_test = load_fn(src_test_fn)
    tgt_test = load_fn(tgt_test_fn)

    samples_map[samples_map_key] = (src_val, tgt_val, src_test, tgt_test)

  # adding none samples
  if _is_adv_schedule_file(args.schedule_file):
    src_val_none_fn, tgt_val_none_fn, src_test_none_fn, tgt_test_none_fn = prepare_fn_adv_single(args.root_dir,
                                                                                                 'features', -1)
  else:
    src_val_none_fn, tgt_val_none_fn, src_test_none_fn, tgt_test_none_fn = prepare_fn_single(args.root_dir, 'features',
                                                                                             'None', 0)

  src_val_none = load_fn(src_val_none_fn)
  tgt_val_none = load_fn(tgt_val_none_fn)
  src_test_none = load_fn(src_test_none_fn)
  tgt_test_none = load_fn(tgt_test_none_fn)

  samples_map[('None', 0,)] = (src_val_none, tgt_val_none, src_test_none, tgt_test_none)

  return samples_map


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

  sch_name = os.path.basename(args.schedule_file)
  sch_name = os.path.splitext(sch_name)[0]

  indices = load_indices_file(sch_name, args.alpha, args.n_samples, args.n_repeats, args.CP_window_size, args.seed)

  total_rejects = []

  start_time = time.time()
  logger.debug('started to sample data')
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

    icm_detector = ICMDetection(alpha=args.alpha,
                                batch_size=args.batch_size,
                                delta=args.delta,
                                epsilon=args.epsilon,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                device=device)

    assert args.CP_window_size % args.batch_size == 0
    assert args.holdout_interval > 0

    holdoutset_src, holdoutset_tgt = [], []
    testset_src, testset_tgt = [], []
    h = float('inf')

    for i in range(0, tgt_sample.shape[0], args.batch_size):

      cur_src_sample = src_sample[i:i+args.batch_size, :]
      cur_tgt_sample = tgt_sample[i:i+args.batch_size, :]

      t = i // args.batch_size

      if t % args.holdout_interval == 0:
        testset_src += [cur_src_sample]
        testset_tgt += [cur_tgt_sample]
      else:
        holdoutset_src += [cur_src_sample]
        holdoutset_tgt += [cur_tgt_sample]

      if len(holdoutset_src) == 0:
        reject = 0
        logger.info("{} [{}] INITIAL".format(i, "ACCEPT"))
      else:
        holdoutset_src_arr = np.concatenate(holdoutset_src)
        holdoutset_tgt_arr = np.concatenate(holdoutset_tgt)
        start_train_idx = -args.CP_window_size if holdoutset_src_arr.shape[0] >= args.batch_size else 0
        recent_train_src = holdoutset_src_arr[start_train_idx:, :]
        recent_train_tgt = holdoutset_tgt_arr[start_train_idx:, :]

        testset_src_arr = np.concatenate(testset_src)
        testset_tgt_arr = np.concatenate(testset_tgt)
        start_test_idx = -args.CP_window_size if testset_src_arr.shape[0] >= args.batch_size else 0
        recent_test_src = testset_src_arr[start_test_idx:, :]
        recent_test_tgt = testset_tgt_arr[start_test_idx:, :]

        if t % args.holdout_interval == 0:
          reject, h, cn_values = icm_detector.test(h, recent_test_src, recent_test_tgt)
          logger.info(
            "{} [{}] Values: {} vs. threshold {},".format(i, "REJECT" if reject == 1 else "ACCEPT", cn_values, h, ))
        else:
          if recent_train_src.shape[0] <= args.batch_size:
            reject = 0
          else:
            h = icm_detector.update(recent_train_src, recent_train_tgt)

      rejects.append(reject)

    total_rejects.append(rejects)

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

  result_fn = os.path.join(args.result_dir, 'result_icm_sch.csv')
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

  total_reject_fn = os.path.join(args.result_dir, 'result_icm_d{}_e{:.2f}_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{:}.csv'.format(args.delta,
                                                                                                                                     args.epsilon,
                                                                                                                                     sch_name,
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

  parser.add_argument('--root_dir', default='./test_sample')
  parser.add_argument('--schedule_file', default='schedules/contrast2_c3.txt')

  parser.add_argument('--holdout_interval', default=2, type=int,
                      help='H')

  parser.add_argument('--CP_window_size', default=100, type=int,
                      help='Number of samples in the window (W = w * bach_size (B))')

  # ICM
  parser.add_argument('--delta', default=100, type=int,
                      help="Significance level")
  parser.add_argument('--epsilon', default=10, type=int,
                      help="Significance level")

  parser.add_argument('--n_samples', default=10000, type=int,
                      help='Number of samples (m)')
  parser.add_argument('--perturb_ratio', default=1.0, type=float)

  parser.add_argument('--batch_size', default=10, type=int,
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
  config['handlers']['file_handler']['filename'] = os.path.join(args.log_dir, 'detect_icm_natural_log_{}.log'.format(date_str))
  logging.config.dictConfig(config)
  logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
  #

  main(args)
