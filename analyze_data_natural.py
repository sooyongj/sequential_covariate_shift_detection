import argparse

import numpy as np
import os


def load_data(fn, dtype=np.int):
  data = np.loadtxt(fn, delimiter=',').astype(dtype)
  return data


def _load_additional_result(sch, r_seeds, folder_dir, type, h, w, args, wald_eps=0.2):
  data = []
  for (r, seed) in r_seeds:
    if type == 'ours':
      fn = './results/{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{:}.csv'.format(folder_dir,
                                                                                                sch,
                                                                                                h,
                                                                                                args.alpha,
                                                                                                args.M,
                                                                                                r,
                                                                                                w,
                                                                                                args.lr,
                                                                                                seed)
    elif type == 'wald':
      fn = './results/{}/result_wald_eps{}_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(folder_dir,
                                                                                                 wald_eps,
                                                                                                 sch,
                                                                                                 args.alpha,
                                                                                                 args.M,
                                                                                                 r,
                                                                                                 args.W,
                                                                                                 args.lr,
                                                                                                 seed)
    elif type == 'dk':
      fn = './results/{}/result_dk_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(folder_dir,
                                                                                         sch,
                                                                                         args.alpha,
                                                                                         args.M,
                                                                                         r,
                                                                                         args.W,
                                                                                         0.1,
                                                                                         seed)
    elif type == 'kds':
      fn = './results/{}/result_kds_kds_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(folder_dir,
                                                                                              sch,
                                                                                              args.alpha,
                                                                                              args.M,
                                                                                              r,
                                                                                              args.W,
                                                                                              0.1,
                                                                                              seed)
    elif type == 'icm':
      fn = './results/{}/result_icm_d{}_e{:.2f}_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(folder_dir,
                                                                                                          100,
                                                                                                          0.1,
                                                                                                          sch,
                                                                                                          2,
                                                                                                          args.alpha,
                                                                                                          args.M,
                                                                                                          r,
                                                                                                          args.W,
                                                                                                          args.lr,
                                                                                                          seed)
    d = load_data(fn)
    data.append(d)
  data = np.concatenate(data)
  return data


def count_detection_req_sample(datas, batch_size, n_perturb_sample, thr=0.8):
  res = []
  for d in datas:
    found = np.where(d.mean(axis=0) >= thr)[0]
    if len(found) == 0:
      res.append(-1)
    else:
      res.append((found[0]+1) * batch_size - n_perturb_sample)

  return res


def cnt_fp(datas, batch_size, timepoints):
  res = []

  for d in datas:
    fpr_cnts = {}
    for sample_point in timepoints:
      idx = sample_point // batch_size
      fpr_cnts[sample_point] = d[:, idx].mean()
    res.append(fpr_cnts)
  return res


def change_perturb_type(old_name, sch_type):
  new_name = old_name.replace('_' + sch_type, '')
  new_name = new_name.replace('_', ' ')
  new_name = new_name.title()
  new_name = new_name[:-1]

  return new_name


def convert_to_text(cnt_map):
  res = {}
  for key in cnt_map:
    cnts = cnt_map[key]
    min_cnt = np.min(cnts)
    txt_cnts = []
    for v in cnts:
      if v == min_cnt:
        txt_cnts.append('\\textbf{{{}}}'.format(v))
      else:
        txt_cnts.append('{}'.format(v))
    res[key] = txt_cnts
  return res


def convert_to_text_fp(fp_map, alpha):
  res = {}
  for key in fp_map:
    fprs = fp_map[key]
    txt_fprs = []
    for res_by_alg in fprs:
      item = {}
      for s in res_by_alg:
        if res_by_alg[s] > alpha:
          item[s] = ('\\textbf{{{:.2f}}}'.format(res_by_alg[s]*100))
        else:
          item[s] = '{:.2f}'.format(res_by_alg[s]*100)
      txt_fprs.append(item)
    res[key] = txt_fprs
  return res


def write_latex(result_data, table_title):
  str = '\\begin{table}\n'
  str += '\\caption{{{}}}\n'.format(table_title)
  str += '\\label{tab:}\n'
  str += '\\begin{center}\n'
  str += '\\begin{tabular}{' + 'l' + ('c' * (len(result_data))) + '}\n'
  str += '\\bf Algorithms & ' + ' & '.join(['\\bf {}'.format(change_perturb_type(c, args.sch)) for c in result_data]) + '\n'
  str += '\\\\ \\hline \\\\\n'
  str += 'Ours & ' + ' & '.join(['{}'.format(result_data[key][0]) for key in result_data]) + '\\\\\n'
  str += 'H2 & ' + ' & '.join(['{}'.format(result_data[key][1]) for key in result_data]) + '\\\\\n'
  str += 'H5 & ' + ' & '.join(['{}'.format(result_data[key][2]) for key in result_data]) + '\\\\\n'
  str += 'Wald & ' + ' & '.join(['{}'.format(result_data[key][3]) for key in result_data]) + '\\\\\n'
  str += 'DK & ' + ' & '.join(['{}'.format(result_data[key][4]) for key in result_data]) + '\\\\\n'
  str += 'KDS & ' + ' & '.join(['{}'.format(result_data[key][5]) for key in result_data]) + '\\\\\n'
  str += 'ICM & ' + ' & '.join(['{}'.format(result_data[key][6]) for key in result_data]) + '\\\\\n'
  str += '\\end{tabular}\n'
  str += '\\end{center}\n'
  str += '\\end{table}'
  return str


def write_fpr_latex(result_data, timepoints, table_title):
  str = '\\begin{table}\n'
  str += '\\caption{{{}}}\n'.format(table_title)
  str += '\\label{tab:}\n'
  str += '\\begin{center}\n'
  str += '\\begin{tabular}{' + 'l' + ('c' * (len(timepoints))) + '}\n'
  str += '\\bf Algorithms & ' + ' & '.join(['\\bf {}'.format(s) for s in timepoints]) + '\n'
  str += '\\\\ \\hline \\\\\n'
  str += 'Ours & ' + ' & '.join([v for v in list(result_data[list(result_data.keys())[0]][0].values())]) + '\\\\\n'
  str += 'H2 & ' + ' & '.join([v for v in list(result_data[list(result_data.keys())[0]][1].values())]) + '\\\\\n'
  str += 'H5 & ' + ' & '.join([v for v in list(result_data[list(result_data.keys())[0]][2].values())]) + '\\\\\n'
  str += 'Wald & ' + ' & '.join([v for v in list(result_data[list(result_data.keys())[0]][3].values())]) + '\\\\\n'
  str += 'DK & ' + ' & '.join([v for v in list(result_data[list(result_data.keys())[0]][4].values())]) + '\\\\\n'
  str += 'KDS & ' + ' & '.join([v for v in list(result_data[list(result_data.keys())[0]][5].values())]) + '\\\\\n'
  str += 'ICM & ' + ' & '.join([v for v in list(result_data[list(result_data.keys())[0]][6].values())]) + '\\\\\n'
  str += '\\end{tabular}\n'
  str += '\\end{center}\n'
  str += '\\end{table}'
  return str


def main(args):
  hs = [-1, 2, 5]
  exp_name_map = {'c3': 'Multiple shift change',
                  'gradinc': 'Gradually increasing',
                  'gradincdec': 'Gradually increasing-then-decreasing'}

  r_seeds = [(1000, 200), (3000, 300), (5000, 350), (5000, 500), (5000, 600)]

  schs = ['{}'.format(args.sch)]

  if args.sch == 'dogs_c3':
    n_perturb_start = 250
    timepoints = [50, 100, 150, 200]  # in # samples
  elif args.sch == 'dogs_gradinc' or args.sch == 'dogs_gradincdec':
    n_perturb_start = 200
    timepoints = [50, 100, 150, 200]  # in # samples
  elif args.sch == 'iwildcam_c3':
    n_perturb_start = 1500
    timepoints = [300, 600, 900, 1200]
  elif args.sch == 'iwildcam_gradinc' or args.sch =='iwildcam_gradincdec':
    n_perturb_start = 1200
    timepoints = [300, 600, 900, 1200]
  elif args.sch == 'py150_c3':
    n_perturb_start = 2500
    timepoints = [500, 1000, 1500, 2000]
  elif args.sch == 'py150_gradinc' or args.sch == 'py150_gradincdec':
    n_perturb_start = 2000
    timepoints = [500, 1000, 1500, 2000]

  dataset_type = args.sch.split('_')[0]
  sch_type = args.sch.split('_')[1]
  folder_dir = 'sch_{}_batch_{}'.format(dataset_type, args.batch_size)

  cnt_map = {}
  fp_map = {}
  for sch in schs:
    print(sch)
    datas_req = []
    datas_fpr = []
    for h in hs:
      fn_req = './results/{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}.csv'.format(
        folder_dir,
        sch,
        h,
        args.alpha,
        args.M,
        args.R_for_req,
        args.W,
        args.lr)

      if not os.path.exists(fn_req):
        fn_req = './results/{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed100.csv'.format(
          folder_dir,
          sch,
          h,
          args.alpha,
          args.M,
          args.R_for_req,
          args.W,
          args.lr)

      data = np.loadtxt(fn_req, delimiter=',').astype(np.int)
      datas_req.append(data)

      fn_fpr = './results/{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}.csv'.format(
        folder_dir,
        sch,
        h,
        args.alpha,
        args.M,
        args.R_for_fpr,
        args.W,
        args.lr)

      if not os.path.exists(fn_fpr):
        fn_fpr = './results/{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed100.csv'.format(
          folder_dir,
          sch,
          h,
          args.alpha,
          args.M,
          args.R_for_fpr,
          args.W,
          args.lr)

      data = load_data(fn_fpr, np.int)
      add_data = _load_additional_result(sch, r_seeds, folder_dir, 'ours', h, args.W, args)
      datas_fpr.append(np.concatenate([data, add_data]))

    wald_eps = 0.2
    wald_fn = './results/{}/result_wald_eps{}_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      wald_eps,
      sch,
      args.alpha,
      args.M,
      args.R_for_req,
      args.W,
      args.lr,
      100)

    data = load_data(wald_fn, np.int)
    datas_req.append(data)

    wald_fn_fpr = './results/{}/result_wald_eps{}_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      wald_eps,
      sch,
      args.alpha,
      args.M,
      args.R_for_fpr,
      args.W,
      args.lr,
      100)

    data = load_data(wald_fn_fpr, np.int)
    add_data = _load_additional_result(sch, r_seeds, folder_dir, 'wald', h, args.W, args, wald_eps=wald_eps)
    datas_fpr.append(np.concatenate([data, add_data]))

    # dk
    dk_fn = './results/{}/result_dk_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      sch,
      args.alpha,
      args.M,
      args.R_for_req,
      args.W,
      0.1,
      100)

    data = load_data(dk_fn, np.int)
    datas_req.append(data)

    dk_fn_fpr = './results/{}/result_dk_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      sch,
      args.alpha,
      args.M,
      args.R_for_fpr,
      args.W,
      0.1,
      100)

    data = load_data(dk_fn_fpr, np.int)
    add_data = _load_additional_result(sch, r_seeds, folder_dir, 'dk', -1, args.W, args, wald_eps=None)
    datas_fpr.append(np.concatenate([data, add_data]))

    # KDS
    kds_fn = './results/{}/result_kds_kds_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      sch,
      args.alpha,
      args.M,
      args.R_for_req,
      args.W,
      0.1,
      100)

    data = load_data(kds_fn, np.int)
    datas_req.append(data)

    kds_fn_fpr = './results/{}/result_kds_kds_sch{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      sch,
      args.alpha,
      args.M,
      100,
      # args.R_for_fpr,
      args.W,
      0.1,
      100)

    data = load_data(kds_fn_fpr, np.int)
    datas_fpr.append(data)
    # add_data = _load_additional_result(sch, r_seeds, folder_dir, 'kds', -1, args.W, args, wald_eps=None)
    # datas_fpr.append(np.concatenate([data, add_data]))

    # icm
    d, e = 100, 0.1
    icm_fn = './results/{}/result_icm_d{}_e{:.2f}_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      d,
      e,
      sch,
      2,
      args.alpha,
      args.M,
      args.R_for_req,
      args.W,
      args.lr,
      100)

    data = load_data(icm_fn, np.int)
    datas_req.append(data)

    icm_fn_fpr = './results/{}/result_icm_d{}_e{:.2f}_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}_seed{}.csv'.format(
      folder_dir,
      d,
      e,
      sch,
      2,
      args.alpha,
      args.M,
      args.R_for_fpr,
      args.W,
      args.lr,
      100)

    data = load_data(icm_fn_fpr, np.int)
    add_data = _load_additional_result(sch, r_seeds, folder_dir, 'icm', -1, args.W, args, wald_eps=None)
    datas_fpr.append(np.concatenate([data, add_data]))
    datas_fpr.append(data)

    # analysis
    n_samples_det = count_detection_req_sample(datas_req, args.batch_size, n_perturb_start, thr=0.80)
    cnt_map[sch] = n_samples_det
    fprs = cnt_fp(datas_fpr, args.batch_size, timepoints)
    fp_map[sch] = fprs

  table_title = '{}, $R={}$ $w={}$'.format(exp_name_map[sch_type],
                                           datas_fpr[0].shape[0],
                                           args.W // args.batch_size)
  latex_str = write_latex(convert_to_text(cnt_map), table_title)

  latex_str_fp = write_fpr_latex(convert_to_text_fp(fp_map, args.alpha), timepoints, table_title)
  print(latex_str)
  print(latex_str_fp)
  with open('tbl_latex_{}_R{}_R{}_W{}_lr{}.txt'.format(args.sch,
                                                       args.R_for_req,
                                                       args.R_for_fpr,
                                                       args.W,
                                                       args.lr), 'w') as f:
    f.write(latex_str)
    f.write('\n\n')
    f.write(latex_str_fp)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', default=10, type=int)
  parser.add_argument('--sch', default='dogs_gradincdec', type=str)
  parser.add_argument('--alpha', default=0.01, type=float)
  parser.add_argument('--M', default=1000, type=int)
  parser.add_argument('--R_for_req', default=100, type=int)
  parser.add_argument('--R_for_fpr', default=1000, type=int)
  parser.add_argument('--W', default=100, type=int)
  parser.add_argument('--lr', default=0.01, type=float)

  args = parser.parse_args()

  main(args)
