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
import torch.optim as optim

from datetime import datetime

from detect_utils import prepare_fn_natural, load_fn
from FCModel import FCModel
from detect_wald_schedule import WaldTestModel


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
  fn = 'indices_pkls/indices_ours_sch{}_H-1_alpha{}_M{}_R{}_W{}_lr0.0100_seed{}.pkl'.format(sch_name,
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

    # for dogs, tgt_none is from src, src_none is src
    src_val, tgt_val_none = extract_notnone_none(src_val_all)
    src_val_none = src_val
    src_test, tgt_test_none = extract_notnone_none(src_test_all)
    src_test_none = src_test

    # to make the same number of samples with src
    tgt_val, _ = extract_notnone_none(tgt_val_all)
    tgt_test, _ = extract_notnone_none(tgt_test_all)

    src_none = np.concatenate((src_val_none, src_test_none))
    tgt_none = np.concatenate((tgt_val_none, tgt_test_none))

    for sch_i, (p_type, p_severity, n_samples, mix_prob) in enumerate(required_samples):
      print(p_type, p_severity, mix_prob)
      # src_val, tgt_val, src_test, tgt_test = samples_map[(p_type, p_severity,)]

      assert ind[sch_i][0] == p_type
      assert ind[sch_i][1] == p_severity
      assert ind[sch_i][2] == n_samples
      assert ind[sch_i][3] == mix_prob

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
        # val
        sub_src_val_sample = src_val_iter[ind[sch_i][4], :]
        sub_tgt_val_sample = tgt_val_iter[ind[sch_i][5], :]

        if len(ind[sch_i]) == 8: # old method
          src_sample.append(sub_src_val_sample)
          tgt_sample.append(sub_tgt_val_sample)
        else:  # new method - mixed
          mixed_src = np.zeros((n_samples, src_val_iter.shape[1]), dtype=np.float32)
          mixed_tgt = np.zeros((n_samples, tgt_val_iter.shape[1]), dtype=np.float32)

          mixed_src[ind[sch_i][8], :] = sub_src_val_sample
          mixed_tgt[ind[sch_i][8], :] = sub_tgt_val_sample

        # test
        sub_src_test_sample = src_test_iter[ind[sch_i][6], :]
        sub_tgt_test_sample = tgt_test_iter[ind[sch_i][7], :]

        if len(ind[sch_i]) == 8:
          src_sample.append(sub_src_test_sample)
          tgt_sample.append(sub_tgt_test_sample)
        else:  # new method - mixed
          mixed_src[~ind[sch_i][8], :] = sub_src_test_sample
          mixed_tgt[~ind[sch_i][8], :] = sub_tgt_test_sample

          src_sample.append(mixed_src)
          tgt_sample.append(mixed_tgt)

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

    # wald test
    network = FCModel(input_size=src_sample.shape[1])
    network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    wald_test = WaldTestModel(epsilon=args.epsilon, alpha=args.alpha, beta=args.alpha)
    reject = 0

    zts_tgt = []
    zpts_src = []

    for i in range(0, tgt_sample.shape[0], args.batch_size):
      test_src_sample = src_sample[i:i + args.batch_size, :]
      test_tgt_sample = tgt_sample[i:i + args.batch_size, :]

      network.eval()

      with torch.no_grad():
        xs = torch.vstack([
          torch.from_numpy(test_src_sample),
          torch.from_numpy(test_tgt_sample)])
        ys = torch.cat([torch.zeros(test_src_sample.shape[0]), torch.ones(test_tgt_sample.shape[0])])

        xs, ys = xs.to(device), ys.to(device)

        output = network(xs)
        sigmoid_output = torch.sigmoid(output)

        zpt_src = (sigmoid_output[:test_src_sample.shape[0]] > 0.5).int()
        zt_tgt = (sigmoid_output[test_src_sample.shape[0]:] > 0.5).int()

      if (i // args.batch_size) % args.holdout_interval == 0:
        if len(zpt_src.squeeze().shape) == 0:
          zpts_src += [zpt_src.squeeze().tolist()]
          zts_tgt += [zt_tgt.squeeze().tolist()]
        else:
          zpts_src += zpt_src.squeeze().tolist()
          zts_tgt += zt_tgt.squeeze().tolist()

        zpts_src_array = np.array(zpts_src)
        zts_tgt_array = np.array(zts_tgt)

        assert args.CP_window_size % args.batch_size == 0

        cnt_upto_now = (i // args.batch_size) // args.holdout_interval + 1  # (including the first one)
        if (i // args.batch_size) - (args.CP_window_size // args.batch_size) < 0:
          cnt_upto_before = 0
        else:
          cnt_upto_before = ((i // args.batch_size) - (args.CP_window_size // args.batch_size)) // args.holdout_interval + 1

        cnt = (cnt_upto_now - cnt_upto_before) * args.batch_size

        actual_window_size = min(args.CP_window_size, cnt)

        pred_correct = np.concatenate([1 - zpts_src_array[-actual_window_size:], zts_tgt_array[-actual_window_size:]])

        hp, s_i = wald_test.test(pred_correct=pred_correct)
        if hp < 2:
          # if decision made, use it
          # otherwise use recent testing result
          reject = hp
          wald_test.reset()
        else:
          reject = reject

        logger.info("Stat: {} vs. threshold {}, {}: {} [{}/{} = {:.2f}] %".format(s_i, wald_test.a, wald_test.b, hp,
                                                                                  pred_correct.sum(),
                                                                                  pred_correct.shape[0],
                                                                                  pred_correct.mean() * 100))

      else:
        val_cur_src_sample = src_sample[i:i+args.batch_size, :]
        val_cur_tgt_sample = tgt_sample[i:i+args.batch_size, :]

        xs = torch.vstack([
          torch.from_numpy(val_cur_src_sample),
          torch.from_numpy(val_cur_tgt_sample)])
        ys = torch.cat([torch.zeros(val_cur_src_sample.shape[0]), torch.ones(val_cur_tgt_sample.shape[0])])

        xs, ys = xs.to(device), ys.to(device)

        network.train()
        optimizer.zero_grad()
        output = network(xs)
        loss = criterion(output, ys.unsqueeze(1))
        loss.backward()
        optimizer.step()
        logger.debug("[{}/{}] Training Loss: {:.4f}".format(i, tgt_sample.shape[0], loss.item()))

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

  total_reject_fn = os.path.join(args.result_dir, 'result_wald_eps{}_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{:}.csv'.format(args.epsilon,
                                                                                                                            sch_name,
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

  parser.add_argument('--root_dir', default='./test_sample/imagenet_dogs/')
  parser.add_argument('--schedule_file', default='schedules/dogs_gradinc.txt')

  parser.add_argument('--holdout_interval', default=2, type=int,
                      help='H')

  parser.add_argument('--CP_window_size', default=100, type=int,
                      help='Number of samples in the window (W = w * bach_size (B))')

  parser.add_argument('--n_samples', default=1000, type=int,
                      help='Number of samples (m)')
  parser.add_argument('--perturb_ratio', default=1.0, type=float)

  parser.add_argument('--batch_size', default=1, type=int,
                      help='Batch size (B)')
  parser.add_argument('--lr', default=0.1, type=float)
  parser.add_argument('--momentum', default=0.9, type=float)
  parser.add_argument('--weight_decay', default=1e-4, type=float)

  # Wald
  parser.add_argument('--epsilon', default=0.2, type=float,
                      help='Difference between H0 and H1')

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
  config['handlers']['file_handler']['filename'] = os.path.join(args.log_dir, 'detect_wald_dogs_log_{}.log'.format(date_str))
  logging.config.dictConfig(config)
  logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
  #

  main(args)
