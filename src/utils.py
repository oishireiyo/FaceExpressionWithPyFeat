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
import numpy as np

def MovingAverage(array: np.ndarray, window_size: int, mode: str='valid', same_length: bool=True) -> np.ndarray:
  '''
  Wrapper function of numpy.convolve.
  mode should be either `full`, `same` or `valid`. `valid` is default.
  In case `valid` is given and same length of numpy array is expected as return value, same_length need to be True.
  '''
  if mode == 'valid' and same_length:
    array = np.pad(array, [(0, window_size-1)], 'edge')

  window = np.ones(window_size) / window_size
  array_average = np.convolve(array, window, mode=mode)
  return array_average

if __name__ == '__main__':
  a = np.array([1, 2, 3, 4, 5])
  b = MovingAverage(array=a, window_size=3)
  print(b)