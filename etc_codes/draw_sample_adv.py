import matplotlib.pyplot as plt
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def draw_plot_example(data):
  plt.figure()
  for i in range(10):
    plt.subplot(2, 5, i + 1)
    # plt.tight_layout()
    img = np.transpose(data[i], (1, 2, 0))
    img = np.clip(img * std + mean, 0, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def draw():
  fn = './sample_xs_e0.0300.npy'
  sample = np.load(fn)

  draw_plot_example(sample)
  plt.show()


if __name__ == '__main__':
  draw()
