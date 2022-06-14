import argparse
import csv
from datetime import datetime
from sklearn.gaussian_process.kernels import RBF
import json
import logging.config
import multiprocessing
import numpy as np
import os
import random
import time

from detect_utils import load_fn, prepare_fn, mix_perturb_and_none
from mmd_utils import decide_sigma, compute_test_stat, bootstrap


def load_and_mix(x_fn, y_fn, x_none_fn, y_none_fn, x_sample_idx, y_sample_idx, ratio):
  X = load_fn(x_fn)
  Y = load_fn(y_fn)

  X_none = load_fn(x_none_fn)
  Y_none = load_fn(y_none_fn)

  X_sample = X[x_sample_idx, :]
  Y_sample = Y[y_sample_idx, :]

  X_none_sample = X_none[x_sample_idx, :]
  Y_none_sample = Y_none[y_sample_idx, :]

  X_sample, Y_sample = mix_perturb_and_none(X_sample, Y_sample, X_none_sample, Y_none_sample, ratio)

  return X_sample, Y_sample


def run_test(idx, x_val_fn, y_val_fn, x_val_none_fn, y_val_none_fn,
             x_test_fn, y_test_fn, x_test_none_fn, y_test_none_fn,
             x_val_sample_idx, y_val_sample_idx,
             x_test_sample_idx, y_test_sample_idx, ratio, args):

  X_val_sample, Y_val_sample = load_and_mix(x_val_fn, y_val_fn,
                                            x_val_none_fn, y_val_none_fn,
                                            x_val_sample_idx, y_val_sample_idx, ratio)

  ###
  if args.sigma == -1:
    logger.debug('started to compute sigma value for val')
    sigma_val = decide_sigma(X_val_sample, Y_val_sample)
  else:
    sigma_val = args.sigma
  logger.debug('sigma_val: {:.2f}'.format(sigma_val))

  X_test_sample, Y_test_sample = load_and_mix(x_test_fn, y_test_fn,
                                              x_test_none_fn, y_test_none_fn,
                                              x_test_sample_idx, y_test_sample_idx, ratio)

  rep_start_time = time.time()
  logger.debug('started to compute values on val set')
  if args.mode == 'unbiased':
    kernel_val = RBF(sigma_val)
    k_val = kernel_val(X_val_sample, X_val_sample)
    l_val = kernel_val(Y_val_sample, Y_val_sample)
    kl_val = kernel_val(X_val_sample, Y_val_sample)
    K = max(k_val[~np.eye(*k_val.shape, dtype=bool)].max(),
            l_val[~np.eye(*l_val.shape, dtype=bool)].max(),
            kl_val.max())
    logger.debug('Max K: {:.4f}'.format(K))
    threshold = 4 * K / np.sqrt(args.n_samples) * np.sqrt(np.log(1 / args.alpha))
  elif args.mode == 'bootstrap':
    K = 1
    threshold = -1  # To be computed with test set data

  sigma_test = sigma_val

  logger.debug('[Exp {}] sigma_test: {:.2f}'.format(idx, sigma_test))
  testStat, k_test, l_test, kl_test, m_test = compute_test_stat(X_test_sample, Y_test_sample, sigma_test, K)
  logger.debug('[Exp {}]m_test: {}'.format(idx, m_test))
  logger.debug('[Exp {}]k_test: {}'.format(idx, k_test))
  logger.debug('[Exp {}]l_test: {}'.format(idx, l_test))
  logger.debug('[Exp {}]kl_test: {}'.format(idx, kl_test))
  logger.debug('[Exp {}]sum: {}'.format(idx, np.sum(k_test + l_test - kl_test - kl_test.transpose())))

  if args.mode == 'bootstrap':
    threshold = bootstrap(k_test, l_test, kl_test, X_test_sample.shape[0], args.n_shuffles, args.alpha)

  reject = 1 if testStat >= threshold else 0

  print("One test time: {:.2f} secs".format(time.time() - rep_start_time))

  logger.info('[EXP {}] {}: testStat: {:.6f}, threshold: {:.6f}'.format(idx, 'REJECT' if reject == 1 else 'ACCEPT',
                                                                        testStat, threshold))

  return reject


def main(args):
  if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)

  if args.perturb_type is not None and args.perturb_type != 'None':
    assert int(args.perturb_level) > 0

  #
  logger.info(args)

  x_val_fn, y_val_fn, x_test_fn, y_test_fn,\
  x_val_none_fn, y_val_none_fn,\
  x_test_none_fn, y_test_none_fn = prepare_fn(args.root_dir,
                                              args.ftype,
                                              args.perturb_type,
                                              args.perturb_level)

  print(x_val_fn, y_val_fn, x_test_fn, y_test_fn, x_test_none_fn, y_test_none_fn)

  X_val = load_fn(x_val_fn)
  Y_val = load_fn(y_val_fn)
  X_test = load_fn(x_test_fn)
  Y_test = load_fn(y_test_fn)

  X_val_none = load_fn(x_val_none_fn)
  Y_val_none = load_fn(y_val_none_fn)

  assert X_val.shape[0] == Y_val.shape[0]
  assert X_val.shape[0] == X_val_none.shape[0]
  assert Y_val.shape[0] == Y_val_none.shape[0]

  start_time = time.time()

  ####
  x_val_sample_indices = []
  y_val_sample_indices = []
  
  x_test_sample_indices = []
  y_test_sample_indices = []

  logger.debug('started to sample data')
  for repeat_idx in range(args.n_repeats):
    X_val_sample_idx = np.random.choice(X_val.shape[0], size=args.n_samples, replace=False)
    Y_val_sample_idx = np.random.choice(Y_val.shape[0], size=args.n_samples, replace=False)

    x_val_sample_indices.append(X_val_sample_idx)
    y_val_sample_indices.append(Y_val_sample_idx)

    X_test_sample_idx = np.random.choice(X_test.shape[0], size=args.n_samples, replace=False)
    Y_test_sample_idx = np.random.choice(Y_test.shape[0], size=args.n_samples, replace=False)

    x_test_sample_indices.append(X_test_sample_idx)
    y_test_sample_indices.append(Y_test_sample_idx)
  logger.debug('finished sampling data')
  #####

  n_process = args.n_workers if args.n_workers > 0 else multiprocessing.cpu_count() * 2
  logger.info("Number of workers: {}".format(n_process))
  pool = multiprocessing.Pool(processes=n_process)

  rejects = pool.starmap(run_test,
                         [(i,
                           x_val_fn, y_val_fn, x_val_none_fn, y_val_none_fn,
                           x_test_fn, y_test_fn, x_test_none_fn, y_test_none_fn,
                           x_val_sample_indices[i], y_val_sample_indices[i],
                           x_test_sample_indices[i], y_test_sample_indices[i],
                           args.perturb_ratio, args) for i in range(args.n_repeats)])
  pool.close()
  pool.join()

  reject_cnt = np.sum(rejects)

  logger.info('*** Test Summaries ***')
  logger.info('Rejects: {}/{} ({:.2f} %)'.format(reject_cnt, args.n_repeats, 100.0 * reject_cnt / args.n_repeats))
  logger.info('took {:.2f} secs'.format(time.time() - start_time))

  res_dict = {'mode': args.mode,
              'PerturbType': args.perturb_type,
              'PerturbLevel': args.perturb_level,
              'PerturbRatio': args.perturb_ratio,
              'alpha': args.alpha,
              'n_shuffle': args.n_shuffles,
              'n_repeats': args.n_repeats,
              'n_samples': args.n_samples,
              'RejectCount': reject_cnt,
              'RejectRatio': 100.0 * reject_cnt / args.n_repeats}

  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

  result_fn = os.path.join(args.result_dir, 'result_mmd.csv'.format(args.perturb_type, args.perturb_level))
  header = ['mode', 'PerturbType', 'PerturbLevel', 'PerturbRatio', 'alpha', 'n_shuffle', 'n_samples', 'n_repeats', 'RejectCount', 'RejectRatio']

  if not os.path.exists(result_fn):
    with open(result_fn, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(header)

  with open(result_fn, 'a') as f:
    writer = csv.DictWriter(f, header)
    writer.writerow(res_dict)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--root_dir', default='./test_sample/')
  parser.add_argument('--ftype', default='features', choices=['features', 'raw'])
  parser.add_argument('--perturb_type', default='gaussian_noise', choices=['contrast', 'defocus_blur', 'elastic_transform', 'gaussian_noise', 'gaussian_blur', 'None'])
  parser.add_argument('--perturb_level', default=1)

  parser.add_argument('--n_samples', default=1000, type=int)
  parser.add_argument('--perturb_ratio', default=1.0, type=float)
  parser.add_argument('--sigma', default=-1, type=float)
  parser.add_argument('--alpha', default=0.05, type=float)

  parser.add_argument('--mode', default='unbiased', choices=['unbiased', 'bootstrap'])
  parser.add_argument('--n_shuffles', default=3000, type=int)   # Bootstrap related parameter

  parser.add_argument('--n_repeats', default=100, type=int)

  parser.add_argument('--n_workers', default=-1, type=int)
  parser.add_argument('--log_dir', default='./logs', type=str)
  parser.add_argument('--result_dir', default='./results', type=str)

  parser.add_argument('--seed', default=100, type=int)

  args = parser.parse_args()

  #
  with open('mmd_logging.json', 'rt') as f:
    config = json.load(f)

  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  date_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
  config['handlers']['file_handler']['filename'] = os.path.join(args.log_dir, 'mmd_log_{}.log'.format(date_str))
  logging.config.dictConfig(config)
  logger = logging.getLogger()

  main(args)
