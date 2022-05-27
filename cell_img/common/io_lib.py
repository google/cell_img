"""IO helpers for writing either locally or to a cloud bucket."""

import codecs
import json
from typing import IO, Union

import fsspec
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
  return fsspec.open(path, mode)
