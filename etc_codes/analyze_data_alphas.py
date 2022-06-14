import argparse

import numpy as np

alphas = [0.01, 0.02, 0.03, 0.04, 0.05]


def count_detection_req_sample(datas, batch_size, n_perturb_sample, thr=0.8):
  res = []
  for d in datas:
    res.append((np.where(d.mean(axis=0) >= thr)[0][0]+1) * batch_size - n_perturb_sample)

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
  new_name = new_name.replace('_', '\\\\')
  new_name = new_name.title()
  new_name = new_name[:-1]
  new_name = "\makecell{" + new_name + "}"

  return new_name


def change_alpha_name(alpha):
  new_name = "{:.0f} \\%".format(alpha * 100)
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


def convert_to_text_fp(fp_map):
  res = {}
  for key in fp_map:
    fprs = fp_map[key]
    txt_fprs = []
    for res_by_alg in fprs:
      item = {}
      for s in res_by_alg:
        if res_by_alg[s] > float(key):
          item[s] = ('\\textbf{{{:.2f}}}'.format(res_by_alg[s]*100))
        else:
          item[s] = '{:.2f}'.format(res_by_alg[s]*100)
      txt_fprs.append(item)
    res[key] = txt_fprs
  return res


def write_latex(result_data, table_title):
  str = '\\begin{table}[!hbt]\n'
  str += '\\caption{{{}}}\n'.format(table_title)
  str += '\\label{tab:}\n'
  str += '\\begin{center}\n'
  str += '\\begin{tabular}{' + 'c' + ('c' * (len(result_data))) + '}\n'
  str += '\\toprule\n'
  str += '\\bf Algorithms & ' + ' & '.join(['\\bf {}'.format(change_alpha_name(c)) for c in result_data]) + '\\\\\n'
  str += '\\midrule\n'
  str += 'Ours & ' + ' & '.join(['{}'.format(result_data[key][0]) for key in result_data]) + '\\\\\n'
  str += 'H2 & ' + ' & '.join(['{}'.format(result_data[key][1]) for key in result_data]) + '\\\\\n'
  str += 'H5 & ' + ' & '.join(['{}'.format(result_data[key][2]) for key in result_data]) + '\\\\\n'
  str += '\\bottomrule\n'
  str += '\\end{tabular}\n'
  str += '\\end{center}\n'
  str += '\\end{table}'
  return str


def write_all_req_latex(result_data, table_title):
  str = '\\begin{table}[!hbt]\n'
  str += '\\caption{{{}}}\n'.format(table_title)
  str += '\\label{tab:}\n'
  str += '\\begin{center}\n'
  str += '\\begin{tabular}{' + 'cc' + ('c' * (len(result_data[0]))) + '}\n'
  str += '\\toprule \n'
  str += '\\bf Severity & \\bf Algorithms & ' + ' & '.join(['\\bf {}'.format(change_perturb_type(c, args.sch_type)) for c in result_data[0]]) + '\\\\\n'
  str += '\\midrule\n'
  for s, data in enumerate(result_data):
    str += '\\multirow{{3}}{{*}}{{{}}}'.format(s+1) + ' & Ours & ' + ' & '.join(['{}'.format(data[key][0]) for key in data]) + '\\\\\n'
    str += '& H2 & ' + ' & '.join(['{}'.format(data[key][1]) for key in data]) + '\\\\\n'
    str += '& H5 & ' + ' & '.join(['{}'.format(data[key][2]) for key in data]) + '\\\\\n'
    if s < len(result_data) - 1:
      str += "\\midrule\n"
    else:
      str += "\\bottomrule\n"
  str += '\\end{tabular}\n'
  str += '\\end{center}\n'
  str += '\\end{table}'
  return str


def write_fpr_latex(result_data, timepoints, table_title):
  str = ""
  for key in result_data:
    str += '\\begin{table}\n'
    str += '\\caption{{{}}}\n'.format(table_title + ", \\alpha={:.0f}\\%".format(key*100))
    str += '\\label{tab:}\n'
    str += '\\begin{center}\n'
    str += '\\begin{tabular}{' + 'l' + ('c' * (len(timepoints))) + '}\n'
    str += '\\bf Algorithms & ' + ' & '.join(['\\bf {}'.format(s) for s in timepoints]) + '\n'
    str += '\\\\ \\hline \\\\\n'
    str += 'Ours & ' + ' & '.join([v for v in list(result_data[key][0].values())]) + '\\\\\n'
    str += 'H2 & ' + ' & '.join([v for v in list(result_data[key][1].values())]) + '\\\\\n'
    str += 'H5 & ' + ' & '.join([v for v in list(result_data[key][2].values())]) + '\\\\\n'
    str += '\\end{tabular}\n'
    str += '\\end{center}\n'
    str += '\\end{table}'
  return str


def main(args):
  hs = [-1, 2, 5]

  n_perturb_start = 2000
  timepoints = [500, 1000, 1500, 2000]  #  in # samples
  table_title_req = 'Gradually increasing-then-decreasing with $R={}$, $w={}$'.format(args.R,
                                                                                      args.W//args.batch_size)
  table_title_fpr = 'Gradually increasing-then-decreasing with $R={}$, $w={}$'.format(args.R,
                                                                                      args.W // args.batch_size)

  cnt_map = {}
  fp_map = {}

  for alpha in alphas:
    print(alpha)
    datas_req = []
    datas_fpr = []
    for h in hs:
      fn = './results/sch_{}_batch_{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}.csv'.format(
        args.sch_type,
        args.batch_size,
        args.sch,
        h,
        alpha,
        args.M,
        args.R,
        args.W,
        args.lr)

      data = np.loadtxt(fn, delimiter=',').astype(np.int)
      datas_req.append(data)

      fn = './results/sch_{}_batch_{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}.csv'.format(
        args.sch_type,
        args.batch_size,
        args.sch,
        h,
        alpha,
        args.M,
        args.R,
        args.W,
        args.lr)
      data = np.loadtxt(fn, delimiter=',').astype(np.int)
      datas_fpr.append(data)

    n_samples_det = count_detection_req_sample(datas_req, args.batch_size, n_perturb_start, thr=0.8)
    cnt_map[alpha] = n_samples_det
    fprs = cnt_fp(datas_fpr, args.batch_size, timepoints)
    fp_map[alpha] = fprs
  latex_str = write_latex(convert_to_text(cnt_map), table_title_req)
  latex_str_fp = write_fpr_latex(convert_to_text_fp(fp_map), timepoints, table_title_fpr)

  print(latex_str)
  print(latex_str_fp)

  with open('tbl_latex_alphas_{}_R{}_W{}_lr{}.txt'.format(args.sch,
                                                          args.R,
                                                          args.W,
                                                          args.lr), 'w') as f:
    f.write(latex_str)
    f.write('\n\n')
    f.write(latex_str_fp)


def all_req_samples(args):
  assert args.sch == 'c3'

  hs = [-1, 2, 5]
  cnt_maps = []

  for s in range(5):
    schs = ['{}{}_c3'.format(t, s + 1) for t in perturb_types]
    n_perturb_start = 2500

    cnt_map = {}
    for sch in schs:
      datas = []
      for h in hs:
        fn = './results/sch_{}_batch_{}/result_ours_sch{}_H{}_alpha{}_M{}_R{}_W{}_lr{:.4f}.csv'.format(
          args.sch,
          args.batch_size,
          sch,
          h,
          args.alpha,
          args.M,
          args.R_for_req,
          args.W,
          args.lr)

        data = np.loadtxt(fn, delimiter=',').astype(np.int)
        datas.append(data)

      n_samples_det = count_detection_req_sample(datas, args.batch_size, n_perturb_start, thr=0.8)
      cnt_map[sch] = n_samples_det

    cnt_maps.append(cnt_map)

  cnt_maps = [convert_to_text(cnt_map) for cnt_map in cnt_maps]

  str = write_all_req_latex(cnt_maps, 'Number of samples for detection with $w={}$'.format(args.W // args.batch_size))
  print(str)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--batch_size', default=10, type=int)
  parser.add_argument('--sch_type', default='gradincdec', type=str)
  parser.add_argument('--sch', default='gaussian_noise2_gradincdec', type=str)
  parser.add_argument('--M', default=10000, type=int)
  parser.add_argument('--R', default=1000, type=int)
  parser.add_argument('--W', default=100, type=int)
  parser.add_argument('--lr', default=0.001, type=float)

  args = parser.parse_args()

  main(args)
