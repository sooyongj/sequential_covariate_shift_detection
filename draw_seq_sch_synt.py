import argparse

import matplotlib.pyplot as plt
import numpy as np


def get_exp_type(sch):
  return args.sch.split("_")[-1]  # 'c3' or 'gradinc' or 'gradincdec'


def load_data(fn, dtype=np.int):
  data = np.loadtxt(fn, delimiter=',').astype(dtype)
  return data


def draw_sum_plot(data, acc_data, labels, batch_size, sch, M, R, W, lr, colors=None, markers=None, markevery=10):
  plt.rcParams.update({'font.size': 15})
  plt.rcParams.update({'xtick.labelsize': 12})
  plt.rcParams.update({'ytick.labelsize': 12})

  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()

  for i, d, in enumerate(data):
    data_sum = d.mean(axis=0) * 100
    linewidth = 2 if labels[i] == 'Ours' else 1
    if colors is not None and len(colors) == len(data):
      ax1.plot(np.arange(d.shape[1]) * batch_size, data_sum, color=colors[i], marker=markers[i], markersize=4, markevery=markevery, linewidth=linewidth)
    else:
      ax1.plot(np.arange(d.shape[1]) * batch_size, data_sum, marker=markers[i], markersize=4, markevery=markevery, linewidth=linewidth)

  a = acc_data[0]
  ax2.plot(np.arange(d.shape[1]) * batch_size, a.mean(axis=0) * 100, color='r', linestyle=':')

  ax1.set_ylim([-2, 102])
  # plt.grid()

  ax1.set_xlabel("Number of samples")
  ax1.set_ylabel("Detection Rate over repetition (%) ")
  ax1.legend(labels, loc='upper left')

  ratios = np.zeros(data[0].shape[1] * batch_size)
  exp_type = get_exp_type(sch)
  if exp_type == 'c3':
    ratios[2500:5000] = 100
    ratios[7500:] = 100
  elif exp_type == 'gradinc':
    ratios[2000:4000] = 20
    ratios[4000:6000] = 40
    ratios[6000:8000] = 60
    ratios[8000:] = 80
  elif exp_type == 'gradincdec':
    ratios[2000:4000] = 40
    ratios[4000:6000] = 80
    ratios[6000:8000] = 40
  else:
    raise ValueError("Wrong exp type: {}".format(exp_type))

  ax2.plot(np.arange(data[0].shape[1] * batch_size), ratios, color='k', linestyle='--')
  ax2.set_ylabel('Shifted sample ratio / Accuracy (%)')
  ax2.set_ylim([-2, 102])

  output_fn = 'sch_sum_{}_B{}_M{}_R{}_W{}_lr{:.4f}.png'.format(sch.lower(),
                                                               batch_size,
                                                               M,
                                                               R,
                                                               W,
                                                               lr)
  plt.savefig(output_fn)


def main(args):
  window_size = [args.W]
  hs = [-1, 2, 5]
  data = []
  acc_data = []

  exp_type = get_exp_type(args.sch)
  folder_dir = 'sch_{}{}_batch_{}'.format(exp_type,
                                          '_adv' if args.sch.startswith('adv') else '',
                                          args.batch_size)

  for h in hs:
    for w in window_size:
      fn = './results/{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}.csv'.format(folder_dir,
                                                                                        args.sch,
                                                                                        h,
                                                                                        args.alpha,
                                                                                        args.M,
                                                                                        args.R,
                                                                                        w,
                                                                                        args.lr)

      d = load_data(fn)
      data.append(d)

      fn = './results/{}/result_acc_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}.csv'.format(folder_dir,
                                                                                            args.sch,
                                                                                            h,
                                                                                            args.alpha,
                                                                                            args.M,
                                                                                            args.R,
                                                                                            w,
                                                                                            args.lr)

      a = load_data(fn, dtype=np.float32)
      acc_data.append(a)

  # qtree_type = 'tv_dist_free'
  # qtree_d = 64
  # qtree_k = 128
  # qtree_fn = './results/{}/result_qtree_{}_D{}_K{}_sch{}_alpha{}_M{}_R{}_W{}.csv'.format(folder_dir,
  #                                                                                        qtree_type,
  #                                                                                        qtree_d,
  #                                                                                        qtree_k,
  #                                                                                        args.sch,
  #                                                                                        args.alpha,
  #                                                                                        args.M,
  #                                                                                        args.R,
  #                                                                                        w)
  # d = load_data(qtree_fn)
  # data.append(d)

  # wald
  wald_eps = 0.2
  wald_fn = './results/{}/result_wald_eps{}_sch{}_alpha{}_M{}_R{}_W{}_seed{}.csv'.format(folder_dir,
                                                                                  wald_eps,
                                                                                  args.sch,
                                                                                  args.alpha,
                                                                                  args.M,
                                                                                  args.R,
                                                                                  w,
                                                                                  args.seed)
  d = load_data(wald_fn)
  data.append(d)

  # DK
  dk_fn = './results/{}/result_dk_sch{}_alpha{}_M{}_R{}_W{}_seed{}.csv'.format(folder_dir,
                                                                               args.sch,
                                                                               args.alpha,
                                                                               args.M,
                                                                               args.R,
                                                                               w,
                                                                               args.seed)
  d = load_data(dk_fn)
  data.append(d)

  # KDS
  kds_kds_fn = './results/{}/result_kds_kds_sch{}_alpha{}_M{}_R{}_W{}_seed{}.csv'.format(
    folder_dir,
    args.sch,
    args.alpha,
    args.M,
    args.R,
    w,
    100)
  d = load_data(kds_kds_fn)
  data.append(d)

  # ICM
  icm_fn = './results/{}/result_icm_d{}_e{:.2f}_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
    folder_dir,
    100,
    0.1,
    args.sch,
    2,
    args.alpha,
    args.M,
    args.R,
    w,
    args.lr,
    100)
  d = load_data(icm_fn)
  data.append(d)

  labels = []
  for h in hs:
    for w in window_size:
      if w == -1:
        lbl = 'H{}'.format(h) if h != -1 else 'Ours'
      else:
        lbl = 'H{}'.format(h) if h != -1 else 'Ours'
      labels.append(lbl)
  # labels.append("QTree")
  labels.append("Wald")
  labels.append("DK")
  labels.append("KDS")
  labels.append('ICM')

  assert args.R == data[0].shape[0], "R should match to the number of experiments."

  draw_sum_plot(data, acc_data, labels, args.batch_size, args.sch, args.M, args.R, args.W, args.lr,
                colors=['royalblue', 'olive', 'maroon', 'darkturquoise', 'lightcoral', 'darkorchid', 'darkgreen'],
                # colors=['black'] *7,
                # colors=['royalblue', 'olive', 'maroon', 'darkturquoise', 'lightcoral', 'darkgreen'])
                # colors=['royalblue', 'olive', 'maroon', 'darkturquoise', 'lightcoral'])
                markers=['', 'x', 'o', 's', 'D', '<', 'v'],
                markevery=20)

  # plt.show()
  plt.close('all')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', default=10, type=int)
  parser.add_argument('--sch', default='gaussian_noise2_c3', type=str)
  parser.add_argument('--alpha', default=0.01, type=float)
  parser.add_argument('--M', default=10000, type=int)
  parser.add_argument('--R', default=100, type=int)
  parser.add_argument('--W', default=100, type=int)
  parser.add_argument('--lr', default=0.001, type=float)
  parser.add_argument('--seed', default=100, type=int)

  args = parser.parse_args()

  main(args)
