import numpy as np
import os


def prepare_fn(root_dir, ftype, perturb_type, perturb_level):
  fn_base = '{}_{}_{}_{}_{}.npy'

  src_val_fn = os.path.join(root_dir, fn_base.format('src', 'val', ftype, perturb_type, perturb_level))
  tgt_val_fn = os.path.join(root_dir, fn_base.format('tgt', 'val', ftype, perturb_type, perturb_level))
  src_test_fn = os.path.join(root_dir, fn_base.format('src', 'test', ftype, perturb_type, perturb_level))
  tgt_test_fn = os.path.join(root_dir, fn_base.format('tgt', 'test', ftype, perturb_type, perturb_level))

  srv_val_none_fn = os.path.join(root_dir, fn_base.format('src', 'val', ftype, 'None', 0))
  tgt_val_none_fn = os.path.join(root_dir, fn_base.format('tgt', 'val', ftype, 'None', 0))

  src_test_none_fn = os.path.join(root_dir, fn_base.format('src', 'test', ftype, 'None', 0))
  tgt_test_none_fn = os.path.join(root_dir, fn_base.format('tgt', 'test', ftype, 'None', 0))

  return src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn, srv_val_none_fn, tgt_val_none_fn, src_test_none_fn, tgt_test_none_fn


def prepare_fn_single(root_dir, ftype, perturb_type, perturb_severity):
  fn_base = '{}_{}_{}_{}_{}.npy'

  src_val_fn = os.path.join(root_dir, fn_base.format('src', 'val', ftype, perturb_type, perturb_severity))
  tgt_val_fn = os.path.join(root_dir, fn_base.format('tgt', 'val', ftype, perturb_type, perturb_severity))
  src_test_fn = os.path.join(root_dir, fn_base.format('src', 'test', ftype, perturb_type, perturb_severity))
  tgt_test_fn = os.path.join(root_dir, fn_base.format('tgt', 'test', ftype, perturb_type, perturb_severity))

  return src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn


def prepare_fn_natural(root_dir, ftype):
  fn_base = '{}_{}_{}.npy'

  src_val_fn = os.path.join(root_dir, fn_base.format('src', 'val', ftype))
  tgt_val_fn = os.path.join(root_dir, fn_base.format('tgt', 'val', ftype))
  src_test_fn = os.path.join(root_dir, fn_base.format('src', 'test', ftype))
  tgt_test_fn = os.path.join(root_dir, fn_base.format('tgt', 'test', ftype))

  return src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn


def prepare_ys_fn_single(root_dir):
  ys_fn_format = '{}_{}_ys_None_0.npy'
  src_val_ys_fn = os.path.join(root_dir, ys_fn_format.format('src', 'val'))
  tgt_val_ys_fn = os.path.join(root_dir, ys_fn_format.format('tgt', 'val'))
  src_test_ys_fn = os.path.join(root_dir, ys_fn_format.format('src', 'test'))
  tgt_test_ys_fn = os.path.join(root_dir, ys_fn_format.format('tgt', 'test'))

  return src_val_ys_fn, tgt_val_ys_fn, src_test_ys_fn, tgt_test_ys_fn


def prepare_fn_natural_ys(root_dir, ftype):
  fn_base = '{}_{}_{}_ys.npy'

  src_val_fn = os.path.join(root_dir, fn_base.format('src', 'val', ftype))
  tgt_val_fn = os.path.join(root_dir, fn_base.format('tgt', 'val', ftype))
  src_test_fn = os.path.join(root_dir, fn_base.format('src', 'test', ftype))
  tgt_test_fn = os.path.join(root_dir, fn_base.format('tgt', 'test', ftype))

  return src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn


def prepare_fn_adv_single(root_dir, ftype, eps):
  if eps < 0:
    fn_base = '{}_{}_{}_adv_None.npy'
    src_val_fn = os.path.join(root_dir, fn_base.format('src', 'val', ftype))
    tgt_val_fn = os.path.join(root_dir, fn_base.format('tgt', 'val', ftype))
    src_test_fn = os.path.join(root_dir, fn_base.format('src', 'test', ftype))
    tgt_test_fn = os.path.join(root_dir, fn_base.format('tgt', 'test', ftype))
  else:
    fn_base = '{}_{}_{}_adv_e{:.4f}.npy'
    src_val_fn = os.path.join(root_dir, 'src_val_features_adv_None.npy')  # source does not contain adv. examples
    tgt_val_fn = os.path.join(root_dir, fn_base.format('tgt', 'val', ftype, eps))
    src_test_fn = os.path.join(root_dir, 'src_test_features_adv_None.npy')  # source does not contain adv. examples
    tgt_test_fn = os.path.join(root_dir, fn_base.format('tgt', 'test', ftype, eps))

  return src_val_fn, tgt_val_fn, src_test_fn, tgt_test_fn


def prepare_ys_fn_adv_single(root_dir):
  ys_fn_format = '{}_{}_ys_adv.npy'
  src_val_ys_fn = os.path.join(root_dir, ys_fn_format.format('src', 'val'))
  tgt_val_ys_fn = os.path.join(root_dir, ys_fn_format.format('tgt', 'val'))
  src_test_ys_fn = os.path.join(root_dir, ys_fn_format.format('src', 'test'))
  tgt_test_ys_fn = os.path.join(root_dir, ys_fn_format.format('tgt', 'test'))

  return src_val_ys_fn, tgt_val_ys_fn, src_test_ys_fn, tgt_test_ys_fn


def load_fn(fn):
  data = np.load(fn)
  if len(data) > 2:
    data = data.reshape((data.shape[0], -1))
  return data


def mix_perturb_and_none(src_sample, tgt_sample, src_none_sample, tgt_none_sample, ratio):
  assert src_sample.shape[0] == src_none_sample.shape[0]
  assert src_none_sample.shape[0] == tgt_sample.shape[0]
  # assert tgt_sample.shape[0] == tgt_none_sample.shape[0]

  if ratio == 0:
    return src_none_sample, tgt_none_sample
  elif ratio == 1.0:
    return src_sample, tgt_sample

  n_no_perturb = int(np.around(src_sample.shape[0] * (1 - ratio)))

  mix_indices = np.random.permutation(src_sample.shape[0])
  mix_indices = mix_indices[:n_no_perturb]

  src_sample[mix_indices, :] = src_none_sample[mix_indices, :]
  tgt_sample[mix_indices, :] = tgt_none_sample[mix_indices, :]

  return src_sample, tgt_sample


def mix_perturb_and_none_prob(src_sample, tgt_sample, src_none_sample, tgt_none_sample, total_cnt, prob, src_ys=None, tgt_ys=None):
  assert src_sample.shape[0] == src_none_sample.shape[0]
  assert src_none_sample.shape[0] == tgt_sample.shape[0]
  # assert tgt_sample.shape[0] == tgt_none_sample.shape[0]

  # these two cases are not reachable.
  if prob == 0:
    return src_none_sample, tgt_none_sample, src_ys, tgt_ys, None
  elif prob == 1.0:
    return src_sample, tgt_sample, src_ys, tgt_ys, None

  random_val = np.random.rand(total_cnt)
  perturb_indices = random_val < prob

  mixed_src_sample = np.zeros((total_cnt, src_sample.shape[1]), dtype=np.float32)  # assume 2D
  mixed_tgt_sample = np.zeros((total_cnt, tgt_sample.shape[1]), dtype=np.float32)

  mixed_src_ys = np.zeros((total_cnt,), dtype=np.long) if src_ys is not None else None
  mixed_tgt_ys = np.zeros((total_cnt,), dtype=np.long) if tgt_ys is not None else None

  perturb_cnt = perturb_indices.sum()
  no_pertub_cnt = total_cnt - perturb_cnt

  src_perturb_samples_indices = np.random.choice(src_sample.shape[0], size=perturb_cnt, replace=False)
  src_none_samples_indices = np.random.choice(src_none_sample.shape[0], size=no_pertub_cnt, replace=False)

  mixed_src_sample[perturb_indices, :] = src_sample[src_perturb_samples_indices, :]
  mixed_src_sample[~perturb_indices, :] = src_none_sample[src_none_samples_indices, :]

  tgt_perturb_samples_indices = np.random.choice(tgt_sample.shape[0], size=perturb_cnt, replace=False)
  tgt_none_samples_indices = np.random.choice(tgt_none_sample.shape[0], size=no_pertub_cnt, replace=False)

  mixed_tgt_sample[perturb_indices, :] = tgt_sample[tgt_perturb_samples_indices, :]
  mixed_tgt_sample[~perturb_indices, :] = tgt_none_sample[tgt_none_samples_indices, :]

  if src_ys is not None and tgt_ys is not None:
    mixed_src_ys[perturb_indices] = src_ys[src_perturb_samples_indices]  # ys is same
    mixed_src_ys[~perturb_indices] = src_ys[src_none_samples_indices]

    mixed_tgt_ys[perturb_indices] = tgt_ys[tgt_perturb_samples_indices]
    mixed_tgt_ys[~perturb_indices] = tgt_ys[tgt_none_samples_indices]

  return mixed_src_sample, mixed_tgt_sample, mixed_src_ys, mixed_tgt_ys, (perturb_indices,
                                                                          src_perturb_samples_indices,
                                                                          src_none_samples_indices,
                                                                          tgt_perturb_samples_indices,
                                                                          tgt_none_samples_indices)
