import numpy as np
from sklearn.gaussian_process.kernels import RBF


def decide_sigma(x, y):
  z = np.vstack([x, y])
  size1 = z.shape[0]
  if size1 > 100:
    zmed = z[:100, :]
    size1 = 100
  else:
    zmed = z

  g = np.sum(np.square(zmed), axis=1)
  g = np.expand_dims(g, axis=1)
  q = np.tile(g, size1)
  r = np.tile(g.transpose(), (size1, 1))
  dists = q + r - 2 * np.matmul(zmed, zmed.transpose())
  dists = dists - np.tril(dists)
  dists = np.reshape(dists, (size1 * size1, -1))
  sigma = np.sqrt(0.5 * np.median(dists[dists > 0]))
  return sigma


def bootstrap(K, L, KL, m, n_shuffle, alpha):
  Kz = np.vstack([np.hstack([K, KL]), np.hstack([KL.transpose(), L])])

  MMDarr = np.zeros([n_shuffle, 1])

  for i in range(n_shuffle):
    index = np.argsort(np.random.rand(2 * m))
    KzShuff = Kz[index][:, index]
    K = KzShuff[:m, :m]
    L = KzShuff[m:2*m, m:2*m]
    KL = KzShuff[:m, m:2*m]

    MMDarr[i, :] = 1 / m / (m - 1) * np.sum(K + L - KL - KL.transpose())

  MMDarr = np.sort(MMDarr)
  thresh = MMDarr[round((1 - alpha) * n_shuffle), 0]

  return thresh


def compute_kernel_val(x, y, sigma):
  kernel = RBF(sigma)
  k = kernel(x, x)
  l = kernel(y, y)
  kl = kernel(x, y)

  return k, l, kl


def compute_test_stat(x, y, sigma, K):
  k, l, kl = compute_kernel_val(x, y, sigma)
  k[(k > K) & (~np.eye(*k.shape, dtype=bool))] = K
  l[(l > K) & (~np.eye(*l.shape, dtype=bool))] = K
  kl[(kl > K) & (~np.eye(*kl.shape, dtype=bool))] = K
  m = x.shape[0]
  h = k + l - kl - kl.transpose()
  hsum = np.sum(h) - np.trace(h)
  testStat = 1 / m / (m - 1) * hsum

  return testStat, k, l, kl, m
