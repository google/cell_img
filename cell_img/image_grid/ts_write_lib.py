"""Methods for creating and writing to an image grid tensorstore."""

import copy
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cell_img.common import io_lib
from cell_img.image_grid import ts_metadata_lib
import pandas as pd
import tensorstore as ts

GCS_PREFIX = 'gs://'


def create_spec_from_path(tensorstore_path: str):
  """Creates a tensorstore spec to open the tensorstore at the given path."""
  if tensorstore_path.startswith(GCS_PREFIX):
    path_without_prefix = tensorstore_path[len(GCS_PREFIX):]
    bucket, path_under_bucket = path_without_prefix.split('/', 1)
    return {
        'driver': 'n5',
        'kvstore': {
            'driver': 'gcs',
            'bucket': bucket,
        },
        'path': path_under_bucket,
    }
  else:
    return {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': tensorstore_path,
        }
    }


def create_spec_with_metadata(spec_without_metadata: Dict[str, Any],
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
  """Creates a tensorstore spec dict with the given metadata."""
  metadata = copy.deepcopy(metadata)
  spec_without_metadata = copy.deepcopy(spec_without_metadata)
  if 'compression' not in metadata:
    metadata['compression'] = {'type': 'gzip'}
  spec = {
      **spec_without_metadata,
      'metadata': metadata,
  }
  return spec


def read_image_properties(image_path: str) -> Tuple[str, Sequence[int]]:
  image = io_lib.read_image(image_path)
  image_dtype = str(image.dtype)
  image_shape = image.shape
  return image_dtype, image_shape


def create_tensorstore(
    spec_without_metadata: Dict[str, Any],
    image_metadata_df: pd.DataFrame,
    axes: List[str],
    axes_wrap: Dict[str, List[str]],
    image_dtype: str,
    image_shape: Sequence[int],
    coordinate_arrays_override: Optional[Dict[str, List[str]]] = None):
  """Creates a new tensorstore."""
  if len(image_shape) != 2:
    raise ValueError('Image shape should be 2 dimensional but got: %s.' %
                     str(image_shape))
  unit_sizes = {'X': image_shape[1], 'Y': image_shape[0]}
  coordinate_arrays = ts_metadata_lib.coordinate_arrays_from_dataframe(
      image_metadata_df, axes, axes_wrap)
  if coordinate_arrays_override:
    coordinate_arrays = ts_metadata_lib.apply_coordinate_arrays_override(
        coordinate_arrays, coordinate_arrays_override)
  tensorstore_metadata = ts_metadata_lib.create_tensorstore_metadata(
      axes, axes_wrap, coordinate_arrays, unit_sizes, image_dtype)
  spec = create_spec_with_metadata(spec_without_metadata, tensorstore_metadata)
  return ts.open(
      spec,
      create=True,
  ).result()


def make_axes_wrap(axes: List[str], x_axis_wrap: List[str],
                   y_axis_wrap: List[str]) -> Dict[str, List[str]]:
  """Returns an axes_wrap dict for tensorsotre metadata."""
  axes_wrap = {}
  if x_axis_wrap is not None:
    if 'X' not in axes:
      raise ValueError('X must be in axes if x_axis_wrap. is specified.')
    axes_wrap['X'] = list(x_axis_wrap)
  if y_axis_wrap is not None:
    if 'Y' not in axes:
      raise ValueError('Y must be in axes if y_axis_wrap. is specified.')
    axes_wrap['Y'] = list(y_axis_wrap)
  return axes_wrap


def open_existing_tensorstore(spec_without_metadata: Dict[str, Any]):
  return ts.open(spec_without_metadata).result()


def overwrite_tensorstore_metadata(original_spec: Dict[str, Any],
                                   new_tensorstore_metadata: Dict[str, Any]):
  """Overwrites the attributes.json metadata file for a tensorstore."""
  tensorstore_dir = get_path_from_spec(original_spec)
  output_path = get_attributes_path(tensorstore_dir)
  io_lib.write_json_file(new_tensorstore_metadata, output_path)
  new_spec = copy.deepcopy(original_spec)
  new_spec['metadata'] = new_tensorstore_metadata
  return open_existing_tensorstore(new_spec)


def get_view_slice_and_read_image(image_metadata, ts_index,
                                  image_path_col: str):
  image_path = image_metadata[image_path_col]
  image_array = io_lib.read_image(image_path)
  view_slice = ts_index.get_whole_image_slice(image_metadata)
  return view_slice, image_array


def get_attributes_path(tensorstore_dir: str) -> str:
  return os.path.join(tensorstore_dir, 'attributes.json')


def get_path_from_spec(spec: Dict[str, Any]) -> str:
  """Returns the path from a tensorstore spec."""
  if spec['kvstore']['driver'] == 'gcs':
    prefix = os.path.join(GCS_PREFIX, spec['kvstore']['bucket'])
  elif spec['kvstore']['driver'] == 'file':
    prefix = ''
  else:
    raise NotImplementedError()
  kvstore_path = spec['kvstore'].get('path')
  if kvstore_path is not None:
    prefix = os.path.join(prefix, kvstore_path)
  path = spec.get('path')
  if path is not None:
    prefix = os.path.join(prefix, path)
  return prefix


def set_path_in_spec(spec: Dict[str, Any], path: str) -> None:
  """Overwrites the path in a tensorstore spec."""
  spec.pop('path', None)
  spec['kvstore'].pop('path', None)
  if spec['kvstore']['driver'] == 'gcs':
    bucket = spec['kvstore']['bucket']
    expected_prefix = os.path.join(GCS_PREFIX, bucket)
    if not path.startswith(expected_prefix):
      raise ValueError('Path should start with %s but got %s' %
                       (expected_prefix, path))
    subdir = path[len(expected_prefix) + 1:]
    spec['path'] = subdir
  elif spec['kvstore']['driver'] == 'file':
    spec['kvstore']['path'] = path
  else:
    raise NotImplementedError()


def assert_yx_axes(axes: List[str]) -> None:
  """Asserts that the axes names start with 'Y' followed by 'X'."""
  if len(axes) < 2:
    raise ValueError('Axes length must be >= 2 but axes is %s' % str(axes))
  if not (axes[0] == 'Y' and axes[1] == 'X'):
    raise ValueError('Axes must start with Y, X but axes is %s' % str(axes))
