"""Library for making annotations to view in neuroglancer.

This assumes the annotations are being added for a layer created by the
image_grid pipeline. The index for this layer is used to convert from metadata
labels like batch plate etc. to YX coordinates.
"""

import collections
import os
import re
from typing import List, Tuple

from cell_img.common import io_lib
from cell_img.image_grid import ts_index_lib
import numpy as np
import pandas as pd

# Only make a annotation property for strings if there are not too many unique.
_MAX_UNIQUE_FOR_ENUM = 200000

# Neuroglancer property names must conform to this regex. This is because they
# are converted to variables for use in the shader code.
_ANNOT_PROP_ID_REGEX = '^[a-z][a-zA-Z0-9_]*$'


def _downcast_ints(series):
  """Downcast a pd.Series of ints to the smallest annotation property type."""
  if not pd.api.types.is_integer_dtype(series):
    raise ValueError('Must be type integer but got %s' % series.dtype)
  # TODO: Try casting to smaller ints first.
  # Debug why the values are corrupt in neuroglancer when using non-int32.
  # Desired order of casting: uint8, int8, uint16, int16, uint32, int32
  # Neuroglancer annotation values should be encoded little endian.
  for dtype in ['<i4']:
    series_downcast = series.astype(dtype)
    if (series == series_downcast).all():
      return series_downcast
  raise ValueError('Could not downcast %s' %
                   str(series.loc[series != series_downcast].head()))


def _downcast_floats(series):
  """Downcast a pd.Series of floats to the smallest annotation property type."""
  if not pd.api.types.is_float_dtype(series):
    raise ValueError('Must be type float but got %s' % series.dtype)
  # Neuroglancer annotation floats are encoded as little endian float 32.
  return series.astype('<f4')


def _get_enum_values(unique_strings):
  """Returns annotation property enum values for unique strings."""
  if not pd.api.types.is_string_dtype(unique_strings):
    raise ValueError('Must be type string but got %s' % unique_strings.dtype)
  for dtype in ['int', 'float']:
    # If the string can be cast as a int or float, use those as the enum values.
    # The neuroglancer UI will show the string label for enums.
    # Shader code uses the numerical enum value. It is nice to have the number
    # to match the string in the cases where the string can be cast to a number.
    try:
      enum_values = unique_strings.astype(dtype)
      return enum_values
    except ValueError:
      continue
  # If the strings are not numeric then just use a unique int for each.
  return np.arange(len(unique_strings))


def _string_to_enum(orig_series):
  """Convert a pd.Series of strings for an enum annotation property."""
  if not pd.api.types.is_string_dtype(orig_series):
    raise ValueError('Must be type string but got %s' % orig_series.dtype)
  unique_strings = np.sort(orig_series.unique())
  enum_values = _get_enum_values(unique_strings)
  enum_labels_to_values = pd.Series(enum_values, index=unique_strings)
  if pd.api.types.is_integer_dtype(enum_labels_to_values):
    enum_labels_to_values = _downcast_ints(enum_labels_to_values)
  if pd.api.types.is_float_dtype(enum_labels_to_values):
    enum_labels_to_values = _downcast_floats(enum_labels_to_values)
  series_as_enum = orig_series.map(enum_labels_to_values)
  series_as_enum.name = orig_series.name
  return series_as_enum, enum_labels_to_values


def _to_annotation_property_id(orig_name):
  """Convert a string to an acceptable neuroglancer annotation property id."""
  # Property ids for neuroglancer must conform to spec:
  # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md#info-json-file-format
  # TODO: Implement a proper fix. Quick lowercase hack for now.
  property_id = orig_name.lower()
  match = re.match(_ANNOT_PROP_ID_REGEX, property_id)
  if match is None:
    raise ValueError(
        'Annotation property id "%s" must match regex %s (after lowercase)' %
        (orig_name, _ANNOT_PROP_ID_REGEX))
  return property_id


def make_properties_df(orig_df):
  """Convert a dataframe so it can be used for annotation properties."""
  converted_series = []
  enum_properties = {}

  for orig_col_name in orig_df.columns:
    new_col_name = _to_annotation_property_id(orig_col_name)
    series = orig_df[orig_col_name].copy()
    series.name = new_col_name
    if pd.api.types.is_integer_dtype(series):
      converted_series.append(_downcast_ints(series))
      continue
    if pd.api.types.is_float_dtype(series):
      converted_series.append(_downcast_floats(series))
      continue
    if pd.api.types.is_bool_dtype(series):
      # Handle booleans as a string.
      # Enum label strings are sorted so 'False'=0 and 'True'=1
      series = series.astype(str)
      # Fall through to handle as a string now.
    if pd.api.types.is_string_dtype(series):
      series_as_enum, enum_labels_to_values = _string_to_enum(series)
      # Avoid making a property for columns with too many unique string values.
      if len(enum_labels_to_values) > _MAX_UNIQUE_FOR_ENUM:
        raise ValueError(
            'Column %s has too many unique entries. Received %d '
            'but max allowed is %d.' % orig_col_name,
            len(enum_labels_to_values), _MAX_UNIQUE_FOR_ENUM)
      else:
        converted_series.append(series_as_enum)
        enum_properties[new_col_name] = enum_labels_to_values
      continue
    raise ValueError('Unhandled type %s' % series.dtype)
  properties_df = pd.concat(converted_series, axis='columns')

  # Check if we have duplicate column names
  # Find all duplicate column names and raise an error
  duplicate_cols = properties_df.columns[properties_df.columns.duplicated()]
  if not duplicate_cols.empty:
    raise ValueError('Found columns %s with the same name. '
                     'Note that column names are formatted by '
                     '_to_annotation_property_id().' % list(duplicate_cols))
  return properties_df, enum_properties


def read_s0_metadata(tensorstore_root_path: str):
  """Reads the tensorstore metadata for the s0 downsample layer."""
  s0_attributes_path = os.path.join(tensorstore_root_path, 's0',
                                    'attributes.json')
  s0_attributes = io_lib.read_json_file(s0_attributes_path)
  return s0_attributes


def _get_available_coords(
    df: pd.DataFrame, ts_index: ts_index_lib.TensorstoreIndex) -> pd.DataFrame:
  """Gets available tensorstore coordinates for the rows of the dataframe.

  A coordinate is unavailable if any of the needed tensorstore index labels are
  not in the input dataframe. For example this lets us get coordinates for Y and
  X even if the dataframe does not have a column for stain.

  Args:
    df: A DataFrame with one row per annotation. There must be columns
      corresponding to the labels in the image_grid tensorstore index.
    ts_index: The image_grid tensorstore index to map labels to coordinates.

  Returns:
    A dataframe with rows corresponding to the input dataframe and columns for
    each available coordinate.

  Raises:
    ValueError: If there are no available coordinates at all.
  """
  df = df.reset_index()  # Move batch plate etc. to columns.
  coords_dict = collections.OrderedDict()
  missing_keys = []
  for axis_name in ts_index.axes:
    try:
      coords_dict[axis_name] = ts_index.get_coords_for_axis(axis_name, df)
    except KeyError as e:
      missing_keys.append(e.args[0])
      # Ignore missing axes unless they are all missing.
  if not coords_dict:
    # All coordinate axes are missing.
    raise ValueError(
        'Could not get any tensorstore coords. Missing labels: %s' %
        str(missing_keys))
  coords_df = pd.DataFrame.from_dict(coords_dict)
  return coords_df


def _get_available_block_sizes_dict(
    df: pd.DataFrame, ts_index: ts_index_lib.TensorstoreIndex
) -> collections.OrderedDict:
  """Gets available block sizes for the rows of the dataframe.

  A block size is unavailable if any of the needed tensorstore index labels are
  not in the input dataframe. For example this lets us get coordinates for Y and
  X even if the dataframe does not have a column for channel.

  Args:
    df: A DataFrame with one row per annotation. There must be columns
      corresponding to the labels in the image_grid tensorstore index.
    ts_index: The image_grid tensorstore index to map labels to coordinates.

  Returns:
    A dictionary with keys corresponding to the ts_index.axes and values
    corresponding to the block size for each dimension.

  Raises:
    ValueError: If there are no available coordinates at all.
  """
  df = df.reset_index()  # Move batch plate etc. to columns.
  block_size_dict = collections.OrderedDict()
  missing_keys = []
  for axis_name in ts_index.axes:
    try:
      block_size_dict[axis_name] = ts_index.get_block_size_for_axis(
          axis_name, df)
    except KeyError as e:
      missing_keys.append(e.args[0])
      # Ignore missing axes unless they are all missing.
  if not block_size_dict:
    # All coordinate axes are missing.
    raise ValueError(
        'Could not get any block sizes. Missing labels: %s' %
        str(missing_keys))
  return block_size_dict


def _shift_image_coords(df: pd.DataFrame, image_origin_coords_df: pd.DataFrame,
                        block_sizes) -> pd.DataFrame:
  """Shifts the coordinates from the image origin.

  For whole images this moves the coordinates to the center of the image.

  For patches, this moves the coordinates to the center of the patch.

  The logic depends on the structure of the dataframes used in diagnose_a_well
  analysis produced by the im2embed pipeline.

  Args:
    df: A DataFrame with columns matching a dataframe produced by im2embed.
    image_origin_coords_df: A dataframe of tensorstore coordinates for the
      origin of the images.
    block_sizes: A dict of coordinate name to size of a unit in the image_grid.
      Typically this is image size * len(smallest_given_coordinate).

  Returns:
    A dataframe similar to the input df with coordinates shifted to the center
    of the patch or whole image.
  """
  missing_dimensions = set(block_sizes) - set(image_origin_coords_df.columns)
  if missing_dimensions:
    raise ValueError('Missing dimensions: %s' % str(missing_dimensions))

  new_coords_df = image_origin_coords_df.copy()
  if _is_patch_df(df):
    df = df.reset_index()  # Move batch plate etc. to columns.
    new_coords_df['X'] += df['center_col']
    new_coords_df['Y'] += df['center_row']
  else:
    for axis_name in block_sizes:
      new_coords_df[axis_name] = (
          new_coords_df[axis_name] + (block_sizes[axis_name] / 2.0))
  return new_coords_df


def _is_patch_df(df: pd.DataFrame) -> bool:
  """Returns True if the dataframe has patches. False for whole image."""
  df = df.reset_index()  # Move batch plate etc. to columns.
  if ('center_row' in df.columns) and ('center_col' in df.columns):
    centers_df = df[['center_row', 'center_col']]
    if not (centers_df == 0).all().all():
      return True
  return False


def get_point_coords(df: pd.DataFrame,
                     ts_index: ts_index_lib.TensorstoreIndex) -> pd.DataFrame:
  """Uses the labels in the dataframe to get the point coordinates."""
  image_origin_coords_df = _get_available_coords(df, ts_index)
  block_sizes = _get_available_block_sizes_dict(df, ts_index)
  coords_df = _shift_image_coords(df, image_origin_coords_df,
                                  block_sizes)
  return coords_df


def to_rect_coords(df: pd.DataFrame, coords_df,
                   ts_index: ts_index_lib.TensorstoreIndex) -> pd.DataFrame:
  """Convert point coordinates to rectandgle coordinates.

  Args:
    df: A DataFrame with one row per annotation. If it has columns named 'width'
      and 'height' then the values are used for the rectangles. If these columns
      are missing then the image unit size in the ts_index is used.
    coords_df: The dataframe of coordinates returned by _get_point_coords.
    ts_index: The TensorstoreIndex. The unit size from this is used to get the
      width and height for rectangle annotations if they are not provided in df.

  Returns:
    The rectangle coordinates as a Numpy array.
  """
  if 'width' in df.columns and 'height' in df.columns:
    half_width = df['width'] / 2
    half_height = df['height'] / 2
  else:
    if _is_patch_df(df):
      raise ValueError('Patch df should have width and height for rectangles')
    block_sizes = _get_available_block_sizes_dict(df, ts_index)
    half_width = block_sizes['Y'] / 2
    half_height = block_sizes['X'] / 2

  orig_yx = coords_df.values
  rect_coords = np.stack([
      orig_yx[:, 0] - half_width,
      orig_yx[:, 1] - half_height,
      orig_yx[:, 0] + half_width,
      orig_yx[:, 1] + half_height
  ], axis=1)

  return rect_coords


def get_dimension_info(
    s0_metadata, available_coords) -> Tuple[List[str], List[str], List[float]]:
  """Returns the dimension names, units and scales for the annotations."""
  dimension_names = list(available_coords)

  # Check that the available_coords are all in s0 metadata['axes']
  all_axes = s0_metadata['axes']
  unknown_dims = set(dimension_names).difference(all_axes)
  if unknown_dims:
    raise ValueError('dimension_names %s should all be in s0 metadata axes %s' %
                     (unknown_dims, all_axes))

  coord_arrays = s0_metadata['coordinateArrays']
  dimension_units = []
  dimension_scales = []
  for dim_name in dimension_names:
    # For backward compatibility reasons, Neuroglancer uses 1nm as the default
    # unit/scale for n5.  However, for dimensions with a coordinate array
    # specified, the dimension is set as unitless (i.e. a unit of "") with a
    # scale of 1.
    if dim_name in coord_arrays:
      dimension_units.append('')
      dimension_scales.append(1)
    else:
      dimension_units.append('m')
      dimension_scales.append(1e-9)
  return dimension_names, dimension_units, dimension_scales
