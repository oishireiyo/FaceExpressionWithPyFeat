# Standard python modules
import os
import sys
import math

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from feat import Detector
from feat.data import Fex
from feat.utils.io import read_feat

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import MovingAverage

class FacialExpressionAnalysis(object):
  def __init__(self) -> None:
    self.detector = None
    self.face_model = 'retinaface'
    self.landmark_model = 'mobilenet'
    self.au_model = 'svm'
    self.emotion_model = 'resmasknet'
    self.facepose_model = 'img2pose'
    self.interval_in_frame = 30

  def set_face_model(self, face_model: str='retinaface') -> None:
    self.face_model = face_model

  def set_landmark_model(self, landmark_model: str='mobilenet') -> None:
    self.landmark_model = landmark_model

  def set_au_model(self, au_model: str='svm') -> None:
    self.au_model = au_model

  def set_emotion_model(self, emotion_model: str='resmasknet') -> None:
    self.emotion_model = emotion_model

  def set_facepose_model(self, facepose_model: str='img2pose') -> None:
    self.facepose_model = facepose_model

  def set_detector(self) -> None:
    self.detector = Detector(
      face_model=self.face_model,
      landmark_model=self.landmark_model,
      au_model=self.au_model,
      emotion_model=self.emotion_model,
      facepose_model=self.facepose_model,
    )

  def change_detector_model(self, models: dict[str, str]) -> None:
    self.detector.change_model(**models)

  def detect_image(self, image_path: str) -> pd.DataFrame:
    results = self.detector.detect_image(image_path)
    return results

  def detect_images(self, image_path_list: list[str]) -> pd.DataFrame:
    results = self.detector.detect_image(image_path_list, batch_size=1)
    return results

  def detect_video(self, video_path: str, skip_frames: int=30) -> pd.DataFrame:
    results = self.detector.detect_video(video_path, skip_frames=skip_frames)
    return results

  def remove_appendix(self, file_path: str) -> str:
    file_path = file_path.split('/')[-1]
    return ''.join(file_path.split('.')[:-1])

  def read_frame(self, video_capture, iframe: int, image_path: str) -> bool:
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, iframe)
    ret, frame = video_capture.read()
    if ret: cv2.imwrite(image_path, frame)
    return ret

  def detect_video_with_images(self, video_path: str, interval_in_frame: int=30) -> pd.DataFrame:
    if not os.path.isfile(video_path):
      logger.error('No such video file was found.')
      sys.exit(1)

    self.interval_in_frame = interval_in_frame
    video_capture = cv2.VideoCapture(video_path)
    nframes = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    results = Fex()
    for iframe in range(0, nframes, interval_in_frame):
      image_path = '../deliverables/' + self.remove_appendix(video_path) + '_frame_%d.png' % (iframe)
      if self.read_frame(video_capture=video_capture, iframe=iframe, image_path=image_path):
        _results = self.detect_image(image_path=image_path)
        if not _results.isna().any().any():
          _results['frame'] = iframe
          results = results.append(data=_results, axis=0)
        else:
          logger.warning('No face was detected for frame: %4d' % (iframe))
    return results

  def get_results_faceboxes(self, results: pd.DataFrame) -> pd.DataFrame:
    return results.faceboxes

  def get_results_aus(self, results: pd.DataFrame) -> pd.DataFrame:
    return results.aus

  def get_results_emotions(self, results: pd.DataFrame) -> pd.DataFrame:
    return results.emotions

  def get_results_poses(self, results: pd.DataFrame) -> pd.DataFrame:
    return results.poses

  def get_moving_average(self, values: pd.Series, window: int):
    return values.rolling(window=window, min_periods=1).mean()

  def make_plots(self, results: pd.DataFrame, columns: list[str], plot_name: str, moving_average_in_frame: int=300) -> None:
    window_size = math.ceil(moving_average_in_sec / self.interval_in_frame)

    plt.figure()

    Xs = results['frame']
    for column in columns:
      Ys = results[column]
      Ys = MovingAverage(array=Ys, window_size=window_size)
      plt.plot(Xs, Ys, label=column)

    plt.title('Expectations')
    plt.xlabel('Frame number')
    plt.ylabel('Probabilities')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)

  def save_as_csv(self, results: pd.DataFrame, csvfile: str='../deliverables/output.csv'):
    results.to_csv(csvfile, index=False)

  def read_results(self, csvfile: str):
    results = read_feat(csvfile)
    return results

if __name__ == '__main__':
  obj = FacialExpressionAnalysis()
  obj.set_detector()
  results = obj.detect_video_with_images(video_path='../assets/tako.mp4', interval_in_frame=30)
  # results = obj.read_results(csvfile='../deliverables/output.csv')
  obj.make_plots(
    results=results,
    columns=['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'],
    plot_name='../deliverables/emotions.png',
    moving_average_in_sec=30.0,
  )
  obj.make_plots(
    results=results,
    columns=['Pitch', 'Roll', 'Yaw'],
    plot_name='../deliverables/poses.png',
    moving_average_in_sec=30.0,
  )
  obj.save_as_csv(results=results, csvfile='../deliverables/output.csv')