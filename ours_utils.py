import logging
import numpy as np
import scipy.stats
import time
import torch
import torch.optim as optim

from math import isnan

from FCModel import FCModel


def run_test(src_x, tgt_x, alpha, device, batch_size, lr, momentum, weight_decay, holdout_interval, CP_window_size, scheduled=False):
  logger = logging.getLogger(__name__)
  logger.info('started to run test')

  dataset = torch.utils.data.TensorDataset(torch.from_numpy(src_x).float(),
                                           torch.from_numpy(tgt_x).float())
  loader = torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=not scheduled)

  network = FCModel(input_size=src_x.shape[1])
  network.to(device)
  optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  criterion = torch.nn.BCEWithLogitsLoss()

  start_time = time.time()
  zts_tgt = []
  zpts_src = []
  cp_intervals = []
  rejects = []
  for t, (src, tgt) in enumerate(loader):
    xs = torch.vstack([src, tgt])
    ys = torch.cat([torch.zeros(src.shape[0]), torch.ones(tgt.shape[0])])
    xs, ys = xs.to(device), ys.to(device)

    network.eval()

    with torch.no_grad():
      output = network(xs)
      sigmoid_output = torch.sigmoid(output)

      step_correct_pred = ((sigmoid_output > 0.5).squeeze() == ys).float()
      print("{}/{} Step accuracy: {:.4f} ({}/{})".format(t+1, len(loader), step_correct_pred.mean(),
                                                           step_correct_pred.sum().int(),
                                                           step_correct_pred.shape[0]))

      zpt_src = (sigmoid_output[:src.shape[0]] > 0.5).int()
      zt_tgt = (sigmoid_output[src.shape[0]:] > 0.5).int()

      assert step_correct_pred.sum().item() == (zpt_src == 0).float().sum() + (zt_tgt == 1).float().sum().item()

      if len(zpt_src.squeeze().shape) == 0:
        zpts_src += [zpt_src.squeeze().tolist()]
        zts_tgt += [zt_tgt.squeeze().tolist()]
      else:
        zpts_src += zpt_src.squeeze().tolist()
        zts_tgt += zt_tgt.squeeze().tolist()

      zpts_src_array = np.array(zpts_src)
      zts_tgt_array = np.array(zts_tgt)

      if CP_window_size > -1:
        actual_window_size = min(len(zpts_src), CP_window_size)

        n = actual_window_size * 2  # SRC and TGT
        k = (1 - zpts_src_array[-actual_window_size:]).sum() + zts_tgt_array[-actual_window_size:].sum()

        if holdout_interval < -1:  # using only stepped samples when holdout_interval < -1
          step = abs(holdout_interval)
          assert actual_window_size % step == 0
          n = actual_window_size / step * 2
          k = (1 - zpts_src_array[-actual_window_size::step]).sum() + zts_tgt_array[-actual_window_size::step].sum()

          assert (zpts_src_array[-actual_window_size::step].shape[0] + zts_tgt_array[-actual_window_size::step].shape[0] == n)

      else:
        n = len(zpts_src) + len(zts_tgt)
        k = (1 - zpts_src_array).sum() + zts_tgt_array.sum()

        if holdout_interval < -1:  # using only stepped samples when holdout_interval < -1
          step = abs(holdout_interval)
          n = (zpts_src_array[::step].shape[0] + zts_tgt_array[::step].shape[0])
          k = (1 - zpts_src_array[::step]).sum() + zts_tgt_array[::step].sum()

      l, u = clopper_pearson_interval(n, k, alpha=alpha)
      if isnan(l) or (l <= 0.5 <= u):
        reject = 0
      else:
        reject = 1
      rejects.append(reject)
      print("CP Interval: {}, {} -> ({:.6f}, {}):{}".format(n, k, l, u, "REJECT" if reject else "ACCEPT"))

      cp_intervals.append([l, u])

    # TRAIN
    network.train()
    optimizer.zero_grad()
    output = network(xs)
    loss = criterion(output, ys.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
      logger.debug("[{}/{}] Training Loss: {:.4f}".format(t, len(loader), loss.item()))

  end_time = time.time()

  logger.info("took {:.2f} secs".format(end_time - start_time))
  logger.info(cp_intervals)
  return rejects, cp_intervals


def run_test_holdout(src_x, tgt_x, alpha, device, batch_size, lr, momentum, weight_decay, holdout_interval, CP_window_size, scheduled=False):
  logger = logging.getLogger(__name__)
  logger.info('started to run test')

  dataset = torch.utils.data.TensorDataset(torch.from_numpy(src_x).float(),
                                           torch.from_numpy(tgt_x).float())
  loader = torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=not scheduled)

  network = FCModel(input_size=src_x.shape[1])
  network.to(device)
  optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  criterion = torch.nn.BCEWithLogitsLoss()

  start_time = time.time()
  zts_tgt = []
  zpts_src = []
  cp_intervals = []
  rejects = []
  for t, (src, tgt) in enumerate(loader):
    xs = torch.vstack([src, tgt])
    ys = torch.cat([torch.zeros(src.shape[0]), torch.ones(tgt.shape[0])])
    xs, ys = xs.to(device), ys.to(device)

    network.eval()

    with torch.no_grad():
      output = network(xs)
      sigmoid_output = torch.sigmoid(output)

      step_correct_pred = ((sigmoid_output > 0.5).squeeze() == ys).float()
      print("{}/{} Step accuracy: {:.4f} ({}/{})".format(t+1, len(loader), step_correct_pred.mean(),
                                                           step_correct_pred.sum().int(),
                                                           step_correct_pred.shape[0]))

      zpt_src = (sigmoid_output[:src.shape[0]] > 0.5).int()
      zt_tgt = (sigmoid_output[src.shape[0]:] > 0.5).int()

      assert step_correct_pred.sum().item() == (zpt_src == 0).float().sum() + (zt_tgt == 1).float().sum().item()

      # Add examples to hold out set every ${holdout_interval} time steps
      if t % holdout_interval == 0:
        if len(zpt_src.squeeze().shape) == 0:
          zpts_src += [zpt_src.squeeze().tolist()]
          zts_tgt += [zt_tgt.squeeze().tolist()]
        else:
          zpts_src += zpt_src.squeeze().tolist()
          zts_tgt += zt_tgt.squeeze().tolist()

      zpts_src_array = np.array(zpts_src)
      zts_tgt_array = np.array(zts_tgt)

      if CP_window_size > -1:
        assert CP_window_size % batch_size == 0

        cnt_upto_now = t // holdout_interval + 1 # (including the first one)
        if t - (CP_window_size // batch_size) < 0:
          cnt_upto_before = 0
        else:
          cnt_upto_before = (t - (CP_window_size // batch_size)) // holdout_interval + 1

        cnt = (cnt_upto_now - cnt_upto_before) * batch_size

        actual_window_size = min(CP_window_size, cnt)

        n = actual_window_size * 2  # SRC and TGT
        k = (1 - zpts_src_array[-actual_window_size:]).sum() + zts_tgt_array[-actual_window_size:].sum()
      else:
        n = len(zpts_src) + len(zts_tgt)
        k = (1 - zpts_src_array).sum() + zts_tgt_array.sum()

      l, u = clopper_pearson_interval(n, k, alpha=alpha)
      if isnan(l) or (l <= 0.5 <= u):
        reject = 0
      else:
        reject = 1
      rejects.append(reject)
      print("CP Interval: {}, {} -> ({},{}):{}".format(n, k, l, u, "REJECT" if reject else "ACCEPT"))

      cp_intervals.append([l, u])

    # TRAIN
    if t % holdout_interval != 0:
      network.train()
      optimizer.zero_grad()
      output = network(xs)
      loss = criterion(output, ys.unsqueeze(1))
      loss.backward()
      optimizer.step()

      if t % 100 == 1:
        logger.debug("[{}/{}] Training Loss: {:.4f}".format(t, len(loader), loss.item()))

  end_time = time.time()

  logger.info("took {:.2f} secs".format(end_time - start_time))
  logger.info(cp_intervals)
  return rejects, cp_intervals


def clopper_pearson_interval(n, k, alpha=0.05):
  # one-sided
  lo = scipy.stats.beta.ppf(alpha, k, n - k + 1)
  hi = 1
  return lo, hi
