"""Methods for working with the metadata for an image grid tensorstore.

The metadata is stored as the attributes.json file with the tensorstore. It is
also in the 'metadata' field of the tensorstore spec.

When using the tensorstore as an image grid, we use the metadata to store the
additional information needed to index into the grid. These additional fields
are ignored by the tensorstore library.

This library defines methods for creating the metadata with these additional
fields and updating it when the tensorstore is expanded to add new images to the
grid.
"""

import copy
from typing import Any, Dict, List, Optional

from cell_img.image_grid import ts_index_lib
import pandas as pd


def create_tensorstore_metadata(axes: List[str], axes_wrap: Dict[str,
                                                                 List[str]],
                                coordinate_arrays: Dict[str, List[str]],
                                unit_sizes: Dict[str, int],
                                dtype: str) -> Dict[str, Any]:
  """Creates metadata for an image grid tensorstore."""
  ts_index = ts_index_lib.TensorstoreIndex(axes, axes_wrap, coordinate_arrays,
                                           unit_sizes)
  block_size = [unit_sizes.get(axis_name, 1) for axis_name in axes]

  tensorstore_metadata = {
      'axes': axes,
      'axesWrap': axes_wrap,
      'coordinateArrays': coordinate_arrays,
      'unitSizes': unit_sizes,
      'dimensions': ts_index.get_dimensions(),
      'dataType': dtype,
      'blockSize': block_size
  }
  return tensorstore_metadata


def coordinate_arrays_from_dataframe(
    image_metadata_df: pd.DataFrame, axes: List[str],
    axes_wrap: Dict[str, List[str]]) -> Dict[str, List[str]]:
  """Creates coordinate arrays from and image metadata dataframe.

  Args:
    image_metadata_df: A pandas dataframe with one row per image and columns
      corresponding to the different image metadata.
    axes: The names of the tensorstore axes.
    axes_wrap: A dict mapping tensorstore axes to the wrapped axes within each.

  Returns:
    A dict mapping each coordinate name to a sorted list of unique values
  for that coordinate.
  """
  coordinate_arrays = {}
  for axis_name in axes:
    if axis_name in axes_wrap:
      cols = axes_wrap[axis_name]
    else:
      cols = [axis_name]
    for col_name in cols:
      all_strings = image_metadata_df[col_name].map(type).eq(str).all()
      if not all_strings:
        raise ValueError('Values of a coordinate array should all be strings. '
                         'Failed on column %s' % col_name)
      unique_vals = image_metadata_df[col_name].unique()
      coordinate_arrays[col_name] = list(sorted(unique_vals))
  return coordinate_arrays


def get_new_coordinate_values(
    original_tensorstore_metadata: Dict[str, Any],
    new_image_metadata_df: pd.DataFrame) -> Dict[str, List[str]]:
  """Returns new coordinate values required to add images to the tensorstore.

  For instance if 'batch3' of images is present in new_image_metadata_df but had
  not been in the original tensorstore then this method will return this dict
  {'batch': ['batch3']}

  Args:
    original_tensorstore_metadata: The original tensorstore metadata.
    new_image_metadata_df: A pandas DataFrame with one row per image and columns
      corresponding to the image metadata types.

  Returns:
    A dict of axis name to a list of values to append to that axis.
  """
  axes = original_tensorstore_metadata['axes']
  axes_wrap = original_tensorstore_metadata['axesWrap']
  orig_coordinate_arrays = original_tensorstore_metadata['coordinateArrays']

  needed_cols = orig_coordinate_arrays.keys()
  missing_cols = set(needed_cols).difference(set(new_image_metadata_df.columns))
  if missing_cols:
    raise ValueError(
        'The following columns were missing from the dataframe: %s' %
        str(list(sorted(missing_cols))))

  to_add_coordinate_arrays = coordinate_arrays_from_dataframe(
      new_image_metadata_df, axes, axes_wrap)

  axis_to_values_to_add = {}
  for axis_name in orig_coordinate_arrays:
    original_values = orig_coordinate_arrays[axis_name]
    new_values = to_add_coordinate_arrays[axis_name]
    values_to_add = list(
        sorted(set(new_values).difference(set(original_values))))
    if values_to_add:
      axis_to_values_to_add[axis_name] = values_to_add
  return axis_to_values_to_add


def _get_frozen_unfrozen(tensorstore_metadata):
  """Returns two lists with the names of axes that are frozen and unfrozen."""
  frozen = []
  unfrozen = []
  for axis_name in tensorstore_metadata['axes']:
    if axis_name in tensorstore_metadata['axesWrap']:
      wrapped_axes = tensorstore_metadata['axesWrap'][axis_name]
      # The first name in the axesWrap list is the coarsest. It can be expanded.
      unfrozen.extend(wrapped_axes[:1])
      # All later axes names in the axesWrap list is the finer and are frozen.
      frozen.extend(wrapped_axes[1:])
    else:
      unfrozen.append(axis_name)
  return frozen, unfrozen


def expand_tensorstore_metadata(
    orig_metadata: Dict[str, Any],
    axis_to_values_to_add: Dict[str, List[str]],
    coordinate_arrays_override: Optional[Dict[str, List[str]]] = None):
  """Returns a metadata for an expanded tensorstore.

  We do not want the coordinates of any existing images to change due to the
  expansion so the tensorstore is enlarged while leaving existing data
  unchanged. New values for any dimension are appended to the existing
  coordinate array.

  If an axis has wrapped subdimensions then only the coarsest dimension can be
  expanded. The finer dimensions are considered frozen since they have already
  been laid out and repeated for each value of the next coarser dimension.

  Args:
    orig_metadata: The original tensorstore metadata.
    axis_to_values_to_add: A dict of axis name to a list of values to append to
      that axis.
    coordinate_arrays_override: Optional override values for coordinate arrays.
      Dict where keys are the axis names to override and value is a list of
      coordinate values. None for no override.

  Returns:
    The metadata for the expanded tensorstore.

  Raises:
    ValueError: If axis_to_values_to_add has any keys for wrapped inner
    dimensions. The coordinate values for these cannot be changed after the
    tensorstore is created.
    ValueError: If the axis_to_values_to_add has any keys for non existent axes.
    ValueError: If the coordinate_arrays_override tries to change the order of
    existing (pre-expanded) values in the coordinate arrays.
  """
  frozen, unfrozen = _get_frozen_unfrozen(orig_metadata)

  trying_to_enlarge_frozen = set(axis_to_values_to_add).intersection(frozen)
  if trying_to_enlarge_frozen:
    raise ValueError(
        'Cannot add new values to inner wrapped coordinate arrays. '
        'Trying to add to %s' % trying_to_enlarge_frozen)

  trying_to_enlarge_non_existent = set(axis_to_values_to_add).difference(
      unfrozen)
  if trying_to_enlarge_non_existent:
    raise ValueError('Cannot add new values to non-existent coordinate arrays. '
                     'Trying to add %s' % trying_to_enlarge_non_existent)

  orig_coordinate_arrays = orig_metadata['coordinateArrays']
  new_coordinate_arrays = copy.deepcopy(orig_coordinate_arrays)
  for axis_name, values_to_add in axis_to_values_to_add.items():
    if values_to_add:
      new_coordinate_arrays[axis_name].extend(values_to_add)

  if coordinate_arrays_override:
    new_coordinate_arrays = apply_coordinate_arrays_override(
        new_coordinate_arrays, coordinate_arrays_override)
    # Check that the override does not try to change the order of the values
    # present before expanding.
    for axis_name, orig_values in orig_coordinate_arrays.items():
      num_orig_values = len(orig_values)
      new_values = new_coordinate_arrays[axis_name]
      for i in range(num_orig_values):
        if orig_values[i] != new_values[i]:
          raise ValueError(
              'When expanding and overriding coordinate arrays, you '
              'cannot change the order of existing values in any '
              'coordinate arrays. For %s original is %s but override '
              'is %s.' % (axis_name, str(orig_values), str(new_values)))

  new_dimensions = ts_index_lib.TensorstoreIndex(
      orig_metadata['axes'], orig_metadata['axesWrap'], new_coordinate_arrays,
      orig_metadata['unitSizes']).get_dimensions()
  new_tensorstore_metadata = copy.deepcopy(orig_metadata)
  new_tensorstore_metadata['dimensions'] = new_dimensions
  new_tensorstore_metadata['coordinateArrays'] = new_coordinate_arrays
  return new_tensorstore_metadata


def apply_coordinate_arrays_override(
    orig_coord_arrays: Dict[str, List[str]],
    override_coord_arrays: Dict[str, List[str]]) -> Dict[str, List[str]]:
  """Override coordinate array values.

  This allows a user to specify the order of values in a coordinate array.

  It also allows the user to specify additional values in a coordinate array
  that are not already present.

  Args:
    orig_coord_arrays: The existing coordinate arrays to which the overrides
      should be applied.
    override_coord_arrays: The overrides to apply. This is a dict where the key
      is the axis name (string) and the value is the complete desired coordinate
      array values for that axis (list of strings).

  Returns:
    New coordinate arrays with overrides applied.

  Raises:
    ValueError: If the override_coord_arrays contains any duplicate values.
    ValueError: If the override_coord_arrays contains keys that are not present
    in orig_coord_arrays. Override should not be adding a new axis.
    ValueError: If the override_coord_arrays is missing a coordinate value that
    is present in orig_coord_arrays. The override must specify the order of all
    the existing values if it tries to override an axis.
  """
  extra_overrides = [
      k for k in override_coord_arrays if k not in orig_coord_arrays
  ]
  if extra_overrides:
    raise ValueError(
        'Override coordinate array should only contain keys present in '
        'the original. Extras: %s' % str(extra_overrides))

  joined = {}
  for k in orig_coord_arrays:
    if k not in override_coord_arrays:
      # Not overridden so just use the original.
      joined[k] = orig_coord_arrays[k]
    else:
      override_values = override_coord_arrays[k]
      override_values_set = set(override_values)
      if len(override_values_set) != len(override_values):
        raise ValueError('Coordinate array values cannot contain duplicates '
                         'but for axis %s got: %s' % (k, str(override_values)))
      orig_values_set = set(orig_coord_arrays[k])
      missing_values = orig_values_set.difference(override_values_set)
      if missing_values:
        raise ValueError(
            'Override coordinate array values should contain all values in '
            'the original array. %s is missing: %s' % (k, str(missing_values)))
      joined[k] = override_values

  return joined
