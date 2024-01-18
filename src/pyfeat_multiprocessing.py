import os
from multiprocessing import Pool
from facial_expression import FacialExpressionAnalysis

def single_process(video_path: str, csvfile: str):
  obj = FacialExpressionAnalysis()
  obj.set_detector()
  results = obj.detect_video_with_images(video_path='../assets/aho.mp4', interval_in_sec=5.0)
  obj.save_as_csv(results=results, csvfile='aho.csv')

if __name__ == '__main__':
  video_paths = ['../assets/aho.mp4']
  csvfiles = ['aho.csv']

  p = Pool(4)
  for video_path, csvfile in zip(video_paths, csvfiles):
    p.apply_async(single_process, args=(video_path, csvfile))