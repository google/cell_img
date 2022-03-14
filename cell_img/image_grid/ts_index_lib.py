"""Methods for indexing into a Tensorstore holding a grid of images.

This is designed to allow efficient lookups of the coordinates of an image from
the image metadata labels. It can be used to lookup the coordinates of one image
or many.

You can get the coordinates of all the rows of an image metadata dataframe where
the columns correspond to the axes names.
"""

import collections
import copy
from typing import Any, Dict, List, Tuple, Union
import numpy as np


class _Single1D():
  """Allows lookup of coordinates of ids within a 1D array."""

  def __init__(self, values):
    """Constructor.

    Args:
      values: An iterable of values that provides the order of the labels in
        this coordinate array.
    """
    self._values = np.array(values)
    if not self._values.size:
      raise ValueError('Input cannot be empty.')
    if len(self._values.shape) != 1:
      raise ValueError('Expected 1D input but got shape %s' %
                       str(self._values.shape))
    if len(np.unique(self._values)) != len(self._values):
      raise ValueError('values cannot contain duplicates but got %s' %
                       collections.Counter(self._values))
    # Make a sorted version so we can use np.searchsorted for fast lookup.
    # Keep the argsort so I can transform back to the original order.
    self._values_argsort = np.argsort(self._values)
    self._values_sorted = self._values[self._values_argsort]
    self.size = len(self._values)

  def get_index(self, query):
    """Returns the coordinates for the query.

    Args:
      query: A single value or list of values for which to get the index.

    Returns:
      The index for the query values. The length is the same as the query.
    """
    query = np.array(query)
    present = np.in1d(query, self._values_sorted)
    if not present.all():
      if not query.ndim:
        not_present = query
      else:
        not_present = np.unique(query[~present])
      raise ValueError(
          'Not all values of the query are in the index. missing=%s' %
          not_present)
    search_res = np.searchsorted(self._values_sorted, query)
    idxs = self._values_argsort[search_res]
    return idxs

  def get_value(self, idx):
    """Returns the value at the given coordinates.

    Args:
      idx: A single coord or list of coords for which to get the value.

    Returns:
      The values for the query coordinates. The length is the same as the query.
    """
    return self._values[idx]


class _Multi1D():
  """Allows lookup of coordinates of ids within a wrapped 1D array."""

  def __init__(self, coordinate_names, coordinate_arrays):
    """Constructor.

    Args:
      coordinate_names: An iterable of strings naming the wrapped dimensions.
        The order indicates the hierarchy. The first name is the outermost
        dimension. Later dimensions "wrap" by repeating their list of values
        within the higher dimension.
      coordinate_arrays: A dict where the keys should include the names in
        coordinate_names and the values are the array of ids for that dimension.
    """
    self._index_lookups = {}
    for name in coordinate_names:
      self._index_lookups[name] = _Single1D(coordinate_arrays[name])
    self._coordinate_names = copy.deepcopy(coordinate_names)
    self.size = np.product([x.size for x in self._index_lookups.values()],
                           dtype=int)

  def _get_missing_coords(self, labels):
    """Return coordinates that are present in the index but not in the labels.

    Args:
      labels: A mapping from name to values. At least one of the names in
        self._coordinate_names should be present. All of the value lists should
        be the same length. This can be a pandas DataFrame where you look up the
        coords for each row of labeled values.
    Returns:
      missing_coords: A list of strings representing the names of the
        coordinates missing from labels.
    """
    missing_coords = []
    for name in self._coordinate_names:
      try:
        labels[name]
      except KeyError:
        missing_coords.append(name)
    self._check_valid_missing_coords(missing_coords)
    return missing_coords

  def _check_valid_missing_coords(self, missing_coords):
    """Check that the missing coordinates are valid.

    We support dropping the smallest coordinates e.g. ('site', 'well'). If we
    try to drop 'well' without 'site' then we raise a KeyError. Similarly, if
    all coordinates are dropped we raise a KeyError.

    Args:
      missing_coords: A list of strings representing the coordinate_names that
        are dropped.
    Raises:
      KeyError: If the coordinates dropped are not the smallest, or if they are
        non-consecutive.
    """
    if not self._coordinate_names:
      if not missing_coords:
        return
      else:
        raise KeyError('No coordinates found, but requested coords '
                       f'{missing_coords}')

    if np.all([i in missing_coords for i in self._coordinate_names]):
      raise KeyError('No indices found in labels! Checked coordinates: %s' %
                     str(missing_coords))

    # Check that only the smallest indices are missing (if any)
    if missing_coords:
      missing_smallest_value = (self._coordinate_names[-1] in missing_coords)
      if not missing_smallest_value:
        raise KeyError('The smallest coordinate: %s is not dropped.' %
                       str(self._coordinate_names[-1]))

      missing_indices = [
          self._coordinate_names.index(i) for i in missing_coords
      ]
      missing_indices.sort()
      consecutive_indices = np.all(np.diff(missing_indices) == 1)
      if not consecutive_indices:
        raise KeyError(
            'Missing non-consecutive coordinates: %s' %
            str(missing_coords))

  def get_index(self, labels):
    """Return the index for each of the labels.

    Args:
      labels: A mapping from name to values. At least one of the names in
        self._coordinate_names should be present. All of the value lists should
        be the same length. This can be a pandas DataFrame where you look up the
        coords for each row of labeled values.

    Returns:
      A numpy array with the index corresponding to each of the same
      length as the input values.

    Raises:
      KeyError: If no indices are found for any coordinate.
    """
    result = 0

    block_size = self.get_smallest_block_size(labels)

    missing_coords = self._get_missing_coords(labels)
    present_coords = [
        name for name in reversed(self._coordinate_names)
        if name not in missing_coords
    ]
    for name in present_coords:
      query = labels[name]
      index_lookup = self._index_lookups[name]
      index_res = index_lookup.get_index(query)
      to_sum = index_res * block_size
      result += to_sum
      block_size *= index_lookup.size

    return result

  def get_smallest_block_size(self, labels):
    """Return the block size for the smallest, present label.

    Args:
      labels: A mapping from name to values. At least one of the names in
        self._coordinate_names should be present. All of the value lists should
        be the same length. This can be a pandas DataFrame where you look up the
        coords for each row of labeled values.

    Returns:
      A float representing the block size for the smallest, present label. Ex:
      if the ts_index has (site, well, plate) but the labels only have (well,
      plate) this will return 1*num_sites_per_well.

    Raises:
      KeyError: If no indices are found for any coordinate.
    """
    block_size = 1

    missing_names = self._get_missing_coords(labels)

    for name in missing_names:
      index_lookup = self._index_lookups[name]
      block_size *= index_lookup.size

    return block_size


class TensorstoreIndex():
  """Allows lookup of coordinates within an image grid tensorstore."""

  def __init__(self, axes: List[str], axes_wrap: Dict[str, List[str]],
               coordinate_arrays: Dict[str, List[str]], unit_sizes: Dict[str,
                                                                         int]):
    """Constructor.

    Args:
      axes: An ordered list of string names for the tensorstore axes.
      axes_wrap: A dict specifying if an axes is made up of wrapping
        subdimensions. The key is the primary tensorstore axis name (must be in
        axes). The value is the ordered list of subdimension names. These are in
        order from coarsest to finest. The labels of the finer subdimension are
        fully repeated for each value of the next coarser dimension.
      coordinate_arrays: A dict where the key is the axis name and the value is
        the ordered list of values for the corrdinates along that dimension. A
        key is expected for each name in axes and axis_wrap.
      unit_sizes: A dictionary where the key is an axis name and the value is
        the size of a unit along that dimension. If a dimensions name is not
        present then the unit size is assumed to be 1. This allows image size to
        be specified as the unit size in the xy dimensions.
    """
    if not axes:
      raise ValueError('axes cannot be empty.')
    self.axes = copy.deepcopy(axes)
    self._index_lookups = {}
    self.unit_sizes = copy.deepcopy(unit_sizes)
    for name in self.axes:
      if name in axes_wrap:
        self._index_lookups[name] = _Multi1D(axes_wrap[name], coordinate_arrays)
      else:
        self._index_lookups[name] = _Multi1D([name], coordinate_arrays)

  def get_dimensions(self) -> List[int]:
    """Returns the dimensions of the tensorstore as a list of ints.

    The order of the returned list corresponds to self.axes.
    """
    dimensions = []
    for axis_name in self.axes:
      dimensions.append(self._index_lookups[axis_name].size *
                        self.unit_sizes.get(axis_name, 1))
    return [int(x) for x in dimensions]

  def get_coords_for_axis(self, axis_name: str, labels):
    """Returns the coordinates for the given labels on the given axis."""
    index_lookup = self._index_lookups[axis_name]
    unit_coords = index_lookup.get_index(labels)
    unit_size = self.unit_sizes.get(axis_name, 1)
    return unit_coords * unit_size

  def get_block_size_for_axis(self, axis_name: str, labels):
    """Returns the block size for the smallest label on the given axis."""
    index_lookup = self._index_lookups[axis_name]
    unit_block_size = index_lookup.get_smallest_block_size(labels)
    unit_size = self.unit_sizes.get(axis_name, 1)
    return unit_block_size * unit_size

  def get_coords(self, labels):
    """Returns the coordinates for the given labels.

    Args:
      labels: A dict like object that allows lookup of values for each
        dimension. To lookup a single coordinate a dictionary of dimension name
        to value can be used. For lookup of many coordinates at once, you can
        use a pandas dataframe where a column name exists for each dimension
        name and the rows have the values to be queried.

    Returns:
      A list of coordinates where the order corresponds to self.axes. If
      multiple coordinates are looked up at once then each value of the returned
      list is a numpy array of the coordinates for that dimension.
    """
    coords = []
    for axis_name in self.axes:
      axis_coords = self.get_coords_for_axis(axis_name, labels)
      coords.append(axis_coords)
    return coords

  def get_block_sizes(self, labels):
    """Returns the block size for the smallest, given labels on each axis.

    Args:
      labels: A dict like object that allows lookup of values for each
        dimension. To lookup a single coordinate a dictionary of dimension name
        to value can be used. For lookup of many coordinates at once, you can
        use a pandas dataframe where a column name exists for each dimension
        name and the rows have the values to be queried.

    Returns:
      A list of sizes where the order corresponds to self.axes.
    """
    block_sizes = []
    for axis_name in self.axes:
      axis_size = self.get_block_size_for_axis(axis_name, labels)
      block_sizes.append(axis_size)
    return block_sizes

  def get_coords_dict(self, labels):
    """Returns the coordinates of the given labels as an OrderedDict.

    The keys are the axes names in self.axes.

    Args:
      labels: labels to be looked up. See get_coords.

    Returns:
      A dict version of the coordinates. See get_coords.
    """
    coords = self.get_coords(labels)
    coords_dict = collections.OrderedDict()
    for axis_name, axis_coords in zip(self.axes, coords):
      coords_dict[axis_name] = axis_coords
    return coords_dict

  def get_block_sizes_dict(self, labels):
    """Returns the block sizes of the given labels as an OrderedDict.

    The keys are the axes names in self.axes.

    Args:
      labels: labels to be looked up. See get_coords.

    Returns:
      A dict version of the block sizes. See get_block_sizes.
    """
    block_sizes = self.get_block_sizes(labels)
    block_size_dict = collections.OrderedDict()
    for axis_name, axis_block_size in zip(self.axes, block_sizes):
      block_size_dict[axis_name] = axis_block_size
    return block_size_dict

  def get_whole_image_slice(self, labels) -> Tuple[Union[int, slice]]:
    """Returns a slice specifying the location of an image in the tensorstore.

    Sample usage:
    # Get the slice to specify the image coords in tensorstore.
    image_slice = get_whole_image_slice(single_image_metadata_dict)
    # Write the image array to tensorstore.
    tensorstore_dataset[image_slice] = image_array_to_store
    # Retrieve the image array from tensorstore.
    image_array_retrieved = tensorstore_dataset[image_slice]

    Args:
      labels: A dict of image metadata label names to values for the image.

    Returns:
      A tuple of slices and ints suitable for reading/writing the image from
      tensorstore.
    """
    start_coords = self.get_coords_dict(labels)
    to_return = []
    for axis_name in self.axes:
      start_coord = start_coords[axis_name]
      unit_size = self.unit_sizes.get(axis_name, 1)
      if unit_size == 1:
        to_return.append(int(start_coord))
      else:
        to_return.append(slice(start_coord, (start_coord + unit_size)))
    return tuple(to_return)


def index_from_tensorstore_metadata(
    tensorstore_metadata: Dict[str, Any]) -> TensorstoreIndex:
  """Create a TensorstoreIndex from the tensorstore metadata.

  Args:
    tensorstore_metadata: The dictionary of metadata from the tensorstore spec.
      Read the spec by opening the tensorstore or reading the attributes.json
      file. Use to_dict and then get the "metadata" value.

  Returns:
    A tensorstore index.
  """
  axes = tensorstore_metadata['axes']
  axes_wrap = tensorstore_metadata['axesWrap']
  coordinate_arrays = tensorstore_metadata['coordinateArrays']
  unit_sizes = tensorstore_metadata['unitSizes']
  return TensorstoreIndex(axes, axes_wrap, coordinate_arrays, unit_sizes)


def index_from_spec(tensorstore_spec: Dict[str, Any]) -> TensorstoreIndex:
  """Create a TensorstoreIndex from the tensorstore spec.

  Same as index_from_tensorstore_metadata

  Args:
    tensorstore_spec: A dict version of the tensorstore spec.

  Returns:
    A tensorstore index.
  """
  metadata = tensorstore_spec['metadata']
  return index_from_tensorstore_metadata(metadata)
