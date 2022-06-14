import argparse

import matplotlib.pyplot as plt
import numpy as np


def load_data(fn, dtype=np.int):
  data = np.loadtxt(fn, delimiter=',').astype(dtype)
  return data


def draw_sum_plot(data, labels, batch_size, sch, M, R, W, lr, colors=None, markers=None):
  plt.rcParams.update({'font.size': 15})
  plt.rcParams.update({'xtick.labelsize': 12})
  plt.rcParams.update({'ytick.labelsize': 12})
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()

  for i, d in enumerate(data):
    data_sum = d.mean(axis=0)*100
    linewidth = 2 if labels[i] == 'Ours' else 1
    if colors is not None and len(colors) == len(data):
      ax1.plot(np.arange(d.shape[1]) * batch_size, data_sum, color=colors[i], marker=markers[i], markersize=4, markevery=5, linewidth=linewidth)
    else:
      ax1.plot(np.arange(d.shape[1]) * batch_size, data_sum, marker=markers[i], markersize=4, markevery=5, linewidth=linewidth)

  ax1.set_ylim([-2, 102])
  # plt.grid()

  ax1.set_xlabel("Number of samples")
  ax1.set_ylabel("Detection Rate over repetition (%) ")
  leg = ax1.legend(labels, loc="upper left")
  for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

  ratios = np.zeros(data[0].shape[1] * batch_size)
  if sch == 'dogs_c3':
    ratios[250:500] = 100
    ratios[750:] = 100
  elif sch == 'dogs_gradinc':
    ratios[200:400] = 20
    ratios[400:600] = 40
    ratios[600:800] = 60
    ratios[800:] = 80
  elif sch == 'dogs_gradincdec':
    ratios[200:400] = 40
    ratios[400:600] = 80
    ratios[600:800] = 40
  elif sch == 'iwildcam_c3':
    ratios[1500:3000] = 100
    ratios[4500:] = 100
  elif sch == 'iwildcam_gradinc':
    ratios[1200:2400] = 20
    ratios[2400:3600] = 40
    ratios[3600:4800] = 60
    ratios[4800:] = 80
  elif sch == 'iwildcam_gradincdec':
    ratios[1200:2400] = 40
    ratios[2400:3600] = 80
    ratios[3600:4800] = 40
  elif sch == 'py150_c3':
    ratios[2500:5000] = 100
    ratios[7500:] = 100
  elif sch == 'py150_gradinc':
    ratios[2000:4000] = 20
    ratios[4000:6000] = 40
    ratios[6000:8000] = 60
    ratios[8000:] = 80
  elif sch == 'py150_gradincdec':
    ratios[2000:4000] = 40
    ratios[4000:6000] = 80
    ratios[6000:8000] = 40

  ax2.plot(np.arange(data[0].shape[1] * batch_size), ratios, color='k', linestyle='--')
  ax2.set_ylabel('Shifted sample ratio (%)')
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

  type = args.sch.split('_')[0]
  folder_dir = 'sch_{}_batch_{}'.format(type, args.batch_size)
  for h in hs:
   for w in window_size:
      fn = './results/{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed100.csv'.format(folder_dir,
                                                                                        args.sch,
                                                                                        h,
                                                                                        args.alpha,
                                                                                        args.M,
                                                                                        args.R,
                                                                                        w,
                                                                                        args.lr)
      d = load_data(fn)
      data.append(d)

  wald_eps = 0.2
  wald_fn = './results/{}/result_wald_eps{}_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(folder_dir,
                                                                                                  wald_eps,
                                                                                                  args.sch,
                                                                                                  args.alpha,
                                                                                                  args.M,
                                                                                                  args.R,
                                                                                                  w,
                                                                                                  args.lr,
                                                                                                  100)
  d = load_data(wald_fn)
  data.append(d)

  # DK
  dk_fn = './results/{}/result_dk_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
    folder_dir,
    args.sch,
    args.alpha,
    args.M,
    args.R,
    w,
    0.1,
    100)
  d = load_data(dk_fn)
  data.append(d)

  # KDS
  kds_kds_fn = './results/{}/result_kds_kds_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
    folder_dir,
    args.sch,
    args.alpha,
    args.M,
    args.R,
    w,
    0.1,
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
    100,
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
  labels.append("Wald")
  labels.append("DK")
  labels.append("KDS")
  labels.append("ICM")

  assert args.R == data[0].shape[0], "R should match to the number of experiments."

  draw_sum_plot(data, labels, args.batch_size, args.sch, args.M, args.R, args.W, args.lr,
                # colors=['mediumseagreen', 'darkorange', 'blueviolet', 'aqua', 'saddlebrown'])
                colors=['royalblue', 'olive', 'maroon', 'darkturquoise', 'lightcoral', 'darkorchid', 'darkgreen'],
                # colors = ['royalblue', 'olive', 'maroon', 'darkturquoise', 'lightcoral'])
                # colors=['mediumseagreen', 'darkorange', 'blueviolet', 'aqua'])
                markers=['', 'x', 'o', 's', 'D', '<', 'v'])


  # plt.show()
  plt.close('all')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', default=10, type=int)
  parser.add_argument('--sch', default='iwildcam_c3', type=str)
  parser.add_argument('--alpha', default=0.01, type=float)
  parser.add_argument('--M', default=6000, type=int)
  parser.add_argument('--R', default=100, type=int)
  parser.add_argument('--W', default=100, type=int)
  parser.add_argument('--lr', default=0.01, type=float)

  args = parser.parse_args()

  main(args)
