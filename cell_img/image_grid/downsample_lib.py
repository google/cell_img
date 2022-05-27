"""Methods for downsampling tensorstore so you can zoom out."""

import copy
import itertools
import math
import os
import re
from typing import Any, Dict, Generator, List, Tuple, Union

import apache_beam as beam
from cell_img.common import io_lib
from cell_img.image_grid import ts_write_lib
import numpy as np
import tensorstore as ts

DOWNSAMPLING_METHOD = 'mean'
# Factor to downsample by between consecutive levels.
YX_DOWNSAMPLE_DELTA_FACTOR = 2
# Use a fixed block size for every downsampled level.
YX_DOWNSAMPLE_BLOCKSIZE = 512
# Stop downsampling when the smallest Y-X dimension reaches this size.
YX_DOWNSAMPLE_MIN_SIZE = YX_DOWNSAMPLE_BLOCKSIZE

# Regex for a downsampled path. Ends with /s followed by an int. E.g. mypath/s5
RE_DOWNSAMPLED_PATH = r'\/s[0-9]+$'

_tensorstore_context = ts.Context()


def read_downsampling_factors(path_to_tensorstore_root: str) -> List[List[int]]:
  """Reads the downsampling factors for all the levels relative to s0."""
  assert_path_is_not_downsampled(path_to_tensorstore_root)
  attributes_path = ts_write_lib.get_attributes_path(path_to_tensorstore_root)
  attributes_json = io_lib.read_json_file(attributes_path)
  return attributes_json['downsamplingFactors']


def create_downsample_levels(s0_spec: ts.Spec):
  """Create tensorstores for all the downsampled levels using the s0 level."""
  input_path = ts_write_lib.get_path_from_spec(s0_spec.to_json())
  root_path, in_downsample_level = split_downsample_level_from_path(input_path)
  if in_downsample_level != 0:
    raise ValueError(
        'Input should be a spec at downsample level 0 but path is %s.' %
        input_path)

  s0_metadata = s0_spec.to_json()['metadata']
  axes = s0_metadata['axes']
  ts_write_lib.assert_yx_axes(axes)

  # The delta downsample factor is relative between any two consecutive levels.
  delta_downsample_factors = [YX_DOWNSAMPLE_DELTA_FACTOR] * 2 + [1] * (
      len(axes) - 2)

  # Store the cumulative downsample factor relative to the 0th level.
  # The 0th level is not dowsampled at all. So all the factors are 1.
  cumulative_downsample_factors = np.ones(len(axes))

  # Store the list of cumulative downsample factors for the root metadata.
  downsampling_factors = [_array_to_int_list(cumulative_downsample_factors)]

  # Use a fixed block size for YX on downsampled levels but copy for other dims.
  orig_block_size = s0_metadata['blockSize']
  block_size = [YX_DOWNSAMPLE_BLOCKSIZE] * 2 + orig_block_size[2:]

  # Loop to iteratively downsample from the 0th level until min size is reached.
  prev_spec = s0_spec
  while True:
    prev_downsampled = ts.downsample(
        prev_spec, delta_downsample_factors, method=DOWNSAMPLING_METHOD)
    dimensions = prev_downsampled.transform.input_exclusive_max
    min_yx = min(dimensions[:2])
    if min_yx <= YX_DOWNSAMPLE_MIN_SIZE:
      break  # No need to downsample further.

    cumulative_downsample_factors *= delta_downsample_factors
    downsampling_factors.append(
        _array_to_int_list(cumulative_downsample_factors))

    new_spec_json = copy.deepcopy(prev_spec.to_json())
    new_spec_json = inc_downsample_level_in_spec(new_spec_json)
    new_spec_json['metadata']['blockSize'] = block_size
    new_spec_json['metadata']['dimensions'] = dimensions
    del new_spec_json['transform']
    # Prevent someone making an index from downsampled levels. Should use s0.
    for name in ['coordinateArrays', 'axesWrap', 'unitSizes', 'axes']:
      if name in new_spec_json['metadata']:
        del new_spec_json['metadata'][name]

    output_ts = ts.open(new_spec_json, create=True).result()
    prev_spec = output_ts.spec()

  top_level_coord_arrays = {
      k: v for k, v in s0_metadata['coordinateArrays'].items() if k in axes
  }

  top_level_attributes = {
      'axes': axes,
      'downsamplingFactors': downsampling_factors,
      'coordinateArrays': top_level_coord_arrays,
  }
  top_level_attributes_path = ts_write_lib.get_attributes_path(root_path)
  io_lib.write_json_file(top_level_attributes, top_level_attributes_path)
  return downsampling_factors


def _array_to_int_list(array: np.ndarray) -> List[int]:
  array_int = array.astype(int)
  if (array_int != array).any():
    raise ValueError('Only integers are allowed but got %s' % array)
  return [int(x) for x in array_int]


def get_delta_factors(downsampling_factors: List[List[int]]) -> List[List[int]]:
  """Computes the delta factors from the s0 relative downsampling levels.

  Args:
    downsampling_factors: The downsampling factors for all the levels relative
      to the s0 level. This is stored in the attributes.json file for the root
      of the tensorstore.

  Returns:
    A list of delta factors. The length of this list is one less than the input.
    Each delta factor is the relative downsampling factor between consecutive
    levels. A single delta factor is a list of ints corresponding to each axes
    of the tensorstore.
  """
  deltas = []
  prev_downsampling_factor = None
  for downsampling_factor in downsampling_factors:
    if prev_downsampling_factor:
      delta = np.array(downsampling_factor) / prev_downsampling_factor
      deltas.append(_array_to_int_list(delta))
    prev_downsampling_factor = downsampling_factor
  return deltas


def assert_path_is_not_downsampled(path: str) -> None:
  """Asserts that the path is not a downsampled level (ending with e.g. s0)."""
  if path.endswith(os.path.sep):
    path = path[:-1]
  m = re.search(RE_DOWNSAMPLED_PATH, path)
  if m is not None:
    raise ValueError(
        'Path %s appears to already be a downsample path e.g. ending in /s0' %
        path)


def split_downsample_level_from_path(path: str) -> Tuple[str, int]:
  """Returns the root path and the downsample level for a downsampled path."""
  if path.endswith(os.path.sep):
    path = path[:-1]
  m = re.search(RE_DOWNSAMPLED_PATH, path)
  if m is None:
    raise ValueError(
        'Path %s does not appear to be a downsample path e.g. ending in /s0' %
        path)
  slash_s_level = m.group(0)
  downsample_level = int(slash_s_level[2:])
  root_path = path[:-len(slash_s_level)]
  return root_path, downsample_level


def join_downsample_level_to_path(path: str, downsample_level: int) -> str:
  """Joins a root path and the downsample level to make a downsampled path."""
  assert_path_is_not_downsampled(path)
  return os.path.join(path, 's%d' % downsample_level)


def inc_downsample_level_in_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
  """Returns a new spec with the downsample level in the path incremented."""
  spec = copy.deepcopy(spec)
  old_path = ts_write_lib.get_path_from_spec(spec)
  root_path, old_level = split_downsample_level_from_path(old_path)
  new_level = old_level + 1
  new_path = join_downsample_level_to_path(root_path, new_level)
  ts_write_lib.set_path_in_spec(spec, new_path)
  return spec


def update_downsamped_dimensions(expanded_s0_dataset) -> None:
  """Updates the dimensions for all downsampled levels to match the s0 level."""
  s0_path = ts_write_lib.get_path_from_spec(
      expanded_s0_dataset.spec().to_json())
  root_path, s0_level = split_downsample_level_from_path(s0_path)
  if s0_level != 0:
    raise ValueError('Input should be at downsample level 0 but path is %s.' %
                     s0_path)
  downsampling_factors_list = read_downsampling_factors(root_path)
  for i, downsampling_factors in enumerate(downsampling_factors_list):
    if i == 0:
      # s0 level has already been enlarged. No need to update its dimensions.
      continue
    downsampled = ts.downsample(
        expanded_s0_dataset, downsampling_factors, method=DOWNSAMPLING_METHOD)
    downsampled_dimensions = downsampled.spec().transform.input_exclusive_max

    downsampled_path = join_downsample_level_to_path(root_path, i)
    attributes_path = ts_write_lib.get_attributes_path(downsampled_path)
    metadata = io_lib.read_json_file(attributes_path)
    metadata['dimensions'] = downsampled_dimensions
    io_lib.write_json_file(metadata, attributes_path)


def get_block_coords_one_dim(slice_or_int: Union[int, slice],
                             downsampling_factor: int,
                             block_size: int) -> Generator[int, None, None]:
  """Yields the block indexes for a slice along one downsampled dimension.

  This takes a slice at one level and returns the indexes for the blocks touched
  by it at the downsampled level.

  Args:
    slice_or_int: A slice or int coordinate at the larger level.
    downsampling_factor: The factor to get to the downsampled level.
    block_size: The block size for this dimension at the downsampled level.

  Yields:
    Block coords (int) at the downsampled level covered by the slice at the
    larger level.
  """
  if isinstance(slice_or_int, int):
    yield math.floor(slice_or_int / downsampling_factor / block_size)
  elif isinstance(slice_or_int, slice):
    assert slice_or_int.step is None  # Implement if needed.
    start = math.floor(slice_or_int.start / downsampling_factor / block_size)
    stop = math.ceil(slice_or_int.stop / downsampling_factor / block_size)
    for i in range(start, stop):
      yield i
  else:
    raise TypeError('Must be int or slice but got %s' % type(slice_or_int))


def get_block_coords(
    slice_or_int_tuple: Tuple[Union[int, slice],
                              ...], downsampling_factors: List[int],
    block_sizes: List[int]) -> Generator[Tuple[int, ...], None, None]:
  """Returns the block indexes for slices along downsampled dimensions."""
  all_dims_blocks = []
  for slice_or_int, downsampling_factor, block_size in zip(
      slice_or_int_tuple, downsampling_factors, block_sizes):
    all_dims_blocks.append(
        get_block_coords_one_dim(slice_or_int, downsampling_factor, block_size))
  for block_coords in itertools.product(*all_dims_blocks):
    yield block_coords


def block_coords_to_slice_or_int_tuple(
    block_coords: Tuple[int, ...], block_sizes: List[int],
    dimensions: List[int]) -> Tuple[Union[int, slice], ...]:
  """Returns the tuple of slices corresponding to the coordinates for a block."""
  slice_or_int_list = []
  for block_coord, block_size, dimension in zip(block_coords, block_sizes,
                                                dimensions):
    start = block_coord * block_size
    stop = (block_coord + 1) * block_size
    stop = min(dimension, stop)  # Prevent indexing beyond dimension limit.
    slice_or_int_list.append(slice(start, stop))
  return tuple(slice_or_int_list)


class _ReadThenWriteToDownsampledDoFn(beam.DoFn):
  """DoFn that reads from one level and writes to the downsampled level."""

  def __init__(self, tensorstore_spec_in: ts.Spec,
               tensorstore_spec_out: ts.Spec, downsample_factors: List[int]):
    super(_ReadThenWriteToDownsampledDoFn, self).__init__()
    self._tensorstore_spec_in = tensorstore_spec_in
    self._tensorstore_spec_out = tensorstore_spec_out
    self._downsample_factors = downsample_factors

  def setup(self):
    input_store = ts.open(
        self._tensorstore_spec_in, open=True,
        context=_tensorstore_context).result()
    self._input_store_downsampled = ts.downsample(
        input_store, self._downsample_factors, method=DOWNSAMPLING_METHOD)
    self._output_store = ts.open(
        self._tensorstore_spec_out,
        write=True,
        open=True,
        context=_tensorstore_context).result()

  def process(self, slice_or_int_tuple: Tuple[Union[int, slice], ...]) -> None:
    self._output_store[slice_or_int_tuple] = self._input_store_downsampled[
        slice_or_int_tuple].read().result()


class _DownsampleOne(beam.PTransform):
  """Writes only the affected blocks from a level to the downsampled level."""

  def __init__(self, spec_in: ts.Spec, spec_out: ts.Spec,
               delta_factor: List[int]):
    """Constructor.

    Args:
      spec_in: The tensorstore spec for the input (bigger) level.
      spec_out: The tensorstore spec for the output (smaller) level.
      delta_factor: The relative downsampling factor between these two levels.
        This is a list of ints corresponding to the tensorstore axes.
    """
    self._spec_in = spec_in
    self._spec_out = spec_out
    self._delta_factor = delta_factor
    dataset_out = ts_write_lib.open_existing_tensorstore(spec_out)
    self._block_sizes = [
        int(x) for x in dataset_out.spec().to_json()['metadata']['blockSize']
    ]
    self._dimensions = [
        int(x) for x in dataset_out.spec().to_json()['metadata']['dimensions']
    ]

  def expand(self, p_slice_in: beam.PCollection) -> beam.PCollection:
    p_block_coords = (
        p_slice_in
        # Find the coordinates (at the output level) of the affected blocks.
        # Use a affected block coords (int) rather than slice so we can dedupe.
        | beam.FlatMap(get_block_coords, self._delta_factor, self._block_sizes)
        # Remove duplicates in affected blocks so we will write each one once.
        # This also forces a shuffle so the downsampled level is written after
        # the larger one is completed.
        | beam.Distinct())
    p_slice_out = (
        # Convert block coordinates to tuples of slices to write.
        p_block_coords | beam.Map(block_coords_to_slice_or_int_tuple,
                                  self._block_sizes, self._dimensions))
    _ = p_slice_out | 'read_then_write_downsample' >> beam.ParDo(
        _ReadThenWriteToDownsampledDoFn(self._spec_in, self._spec_out,
                                        self._delta_factor))
    return p_slice_out


class Downsample(beam.PTransform):
  """Propagates the affected blocks from s0 level to all downsampled levels."""

  def __init__(self, s0_tensorstore_spec: ts.Spec,
               downsampling_factors: List[List[int]]):
    """Constructor.

    Args:
      s0_tensorstore_spec: The tensorstore spec for the s0 (biggest) level.
      downsampling_factors: The downsampling factors for all the levels relative
        to s0.
    """
    super(Downsample, self).__init__()
    self._s0_tensorstore_spec = s0_tensorstore_spec
    self._delta_factors = get_delta_factors(downsampling_factors)

  def expand(self,
             p_s0_slice: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    spec_in = self._s0_tensorstore_spec
    p_slice_in = p_s0_slice
    for i, delta_factor in enumerate(self._delta_factors):
      spec_out = inc_downsample_level_in_spec(spec_in)
      p_slice_out = (
          p_slice_in | ('downsample_%d' % i) >> _DownsampleOne(
              spec_in, spec_out, delta_factor))
      p_slice_in = p_slice_out
      spec_in = spec_out

    return p_s0_slice
