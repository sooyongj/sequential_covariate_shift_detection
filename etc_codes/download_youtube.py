import argparse
import cv2
import os

from pytube import YouTube


def main(args):
  yt = YouTube(args.url)

  st = yt.streams.filter(file_extension='mp4', resolution='720p').first()
  fn = st.default_filename
  print(fn)
  st.download(output_path=args.video_output_path)
  full_path = os.path.join(args.video_output_path, fn)
  print("video download completed")

  img_output_path = os.path.join(args.image_output_path, args.name)
  if not os.path.exists(img_output_path):
    os.makedirs(img_output_path)
  print("start to convert to images")
  vidcap = cv2.VideoCapture(full_path)
  success, image = vidcap.read()
  count = 0
  while success:
    cv2.imwrite(os.path.join(img_output_path, "frame{:04d}.jpg".format(count)), image)  # save frame as JPEG file
    success, image = vidcap.read()
    # print('Read a new frame: ', success)
    print("\rNew frame {} - {}".format(count, success), end='')
    count += 1
  print("\nFinished.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument('--url', default='https://www.youtube.com/watch?v=kQM6Q9Axyx0', type=str)
  # parser.add_argument('--url', default='https://www.youtube.com/watch?v=DrInAUzKH5g', type=str)
  parser.add_argument('--url', default='https://www.youtube.com/watch?v=1gIx20DbtvE', type=str)

  # parser.add_argument('--name', default='nighttoday', type=str)
  # parser.add_argument('--name', default='weatherchange', type=str)
  parser.add_argument('--name', default='driving', type=str)
  #
  parser.add_argument('--video_output_path', default='../videos', type=str)
  parser.add_argument('--image_output_path', default='../images', type=str)
  args = parser.parse_args()

  main(args)