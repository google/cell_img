"""IO helpers for writing either locally or to a cloud bucket."""

import codecs
import json
from typing import IO, Union

import apache_beam as beam
import numpy as np
import pandas as pd
from PIL import Image as PilImage

_IO = Union[IO, codecs.StreamWriter, codecs.StreamReader]


def read_csv(filename: str, **kwargs) -> pd.DataFrame:
  with _open(filename, 'rt') as f:
    return pd.read_csv(f, **kwargs)


def write_csv(df: pd.DataFrame, output_path: str, **kwargs) -> None:
  with _open(output_path, 'wt') as f:
    df.to_csv(f, **kwargs)


def read_image(filename: str) -> np.ndarray:
  with _open(filename, 'rb') as f:
    return np.asarray(PilImage.open(f))


def write_image(array: np.ndarray, output_path: str) -> None:
  pil_image = PilImage.fromarray(array)
  with _open(output_path, 'wb') as f:
    pil_image.save(f, 'png')


def read_json_file(filename: str):
  with _open(filename, 'rt') as f:
    return json.load(f)


def write_json_file(to_encode, output_path: str) -> None:
  json.dumps(to_encode)  # Check it can be encoded before touching the file.
  with _open(output_path, 'wt') as f:
    json.dump(to_encode, f)


def _open(path, mode) -> _IO:
  """Opens a file."""
  if mode == 'rb':
    return _open_binary(path, 'r')
  elif mode == 'wb':
    return _open_binary(path, 'w')
  elif mode == 'rt':
    read_wrapper = codecs.getreader('utf-8')
    return read_wrapper(_open_binary(path, 'r'))
  elif mode == 'wt':
    write_wrapper = codecs.getwriter('utf-8')
    return write_wrapper(_open_binary(path, 'w'))
  else:
    raise NotImplementedError()


def _open_binary(path, read_or_write_mode):
  """Opens a file in binary mode since gcsio.GcsIO can't handle text."""
  assert read_or_write_mode in 'rw'
  if path.startswith('gs://'):
    gcs = beam.io.gcp.gcsio.GcsIO()
    return gcs.open(path, read_or_write_mode)
  else:
    return open(path, read_or_write_mode + 'b')
