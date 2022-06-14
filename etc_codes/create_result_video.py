import cv2
import glob
import numpy as np
import os
import re


def load_result(name, W, lr):
  Hs = [-1, 2, 5]
  data = []
  for h in Hs:
    result_fn = './results/video/result_ours_video{}_H{}_alpha0.01_W{}_lr{:.4f}.csv'.format(name, h, W, lr)
    d = np.loadtxt(result_fn, delimiter=',').astype(np.int)
    data.append(d)

  return data


def main():
  root_dir = '../images/'
  dataset_name = 'nighttoday'
  dataset_dir = os.path.join(root_dir, dataset_name)
  video_output_fn = os.path.join('../videos', dataset_name + "_result.mp4")

  # weather change
  # n_src = 3000
  # batch_size = 30
  # W = 600
  # lr = 0.0001

  # nighttoday, driving
  n_src = 300
  batch_size = 10
  W = 100
  lr = 0.0005

  result_data = load_result(dataset_name, W, lr)

  files = glob.glob(os.path.join(dataset_dir, "*.jpg"))
  files = sorted(files, key=lambda x: float(re.findall("(\d+)", os.path.basename(x))[0]))

  assert abs(len(files) - n_src - result_data[0].shape[0] * batch_size) < batch_size

  frame = cv2.imread(files[0])
  height, width, layers = frame.shape
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  # fourcc = cv2.VideoWriter_fourcc(*'avc1')
  video = cv2.VideoWriter(video_output_fn, fourcc, 30, (width, height))

  for i, fn in enumerate(files):
    im = cv2.imread(fn)

    if i < n_src:
      im = cv2.putText(im,
                       text="SOURCE".format(i + 1),
                       org=(75, 75),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=1,
                       color=(128, 128, 128),
                       thickness=2)
    else:
      idx_diff = i - n_src
      idx = idx_diff // batch_size

      detect_ours = result_data[0][idx] > 0
      detect_H2 = result_data[1][idx] > 0
      detect_H5 = result_data[2][idx] > 0

      text = ''
      if not (detect_ours or detect_H2 or detect_H5):
        text = 'No shift'
      else:
        text = 'Shift Detected by ('
        if detect_ours:
          text += 'Ours,'
        if detect_H2:
          text += 'H2,'
        if detect_H5:
          text += 'H5,'
        if text.endswith(','):
          text = text[:-1]
        text += ')'

      im = cv2.putText(im,
                       text=text,
                       org=(75, 75),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=1,
                       color=(255, 255, 255),
                       thickness=2)
    video.write(im)
    print("\rFrame {}/{} - finished".format(i + 1, len(files)), end='')
  print("\nFinished. {}".format(video_output_fn))
  cv2.destroyAllWindows()
  video.release()


if __name__ == '__main__':
  main()
