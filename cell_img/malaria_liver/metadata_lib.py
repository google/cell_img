"""Library for malaria_liver metadata.

The MetadataIndex class wraps the image_grid.TensorstoreIndex class using
specific knowledge of the metadata used in the malaria_liver project to provide
easy access to the image_grid for the project.

For example:
  -- in our experimental data, we use batch, plate, well, site to identify the
     images laid out in the image_grid. This class has functions that take in
     our metadata and return the position of that image within the image_grid.
  -- This class has convenience functions that wrap the image_grid instance
     and return images formatted for this project.
"""

import os
from typing import Any, Dict, List, Tuple, Union

from absl import logging
from cell_img.common import image_lib
from cell_img.common import io_lib
from cell_img.image_grid import ts_index_lib
from cell_img.image_grid import ts_write_lib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorstore as ts

METADATA_COLS = ['batch', 'plate', 'well', 'site']

WELL_LAYOUT_DICT = {
    '0': 'experiment/well_0.csv'
}

SITE_LAYOUT_DICT = {
    '0': 'experiment/site_0.csv',
    '1': 'experiment/site_1.csv',
}


class MetadataIndex(object):
  """Convenience wrapper on the TensorstoreIndex for access."""

  def __init__(self, tensorstore_root_path: str, channel_list: List[str],
               metadata_root_path: str):
    """Constructor.

    Args:
      tensorstore_root_path: String path of the directory where the tensorstore
        index dictionary is stored. It is expected that the
        tensorstore index will be at: tensorstore_root_path/s0/attributes.json
      channel_list: String list of channel names for the image stack, e.g.
        ['w3', 'w2', 'w1']. Images pulled from tensorstore will have these
        channels in this order (so, you often want them in RGB order).
      metadata_root_path: String path of the directory where the metadata is
        stored.
    """
    self.ts_root_path = tensorstore_root_path
    self.ts_index = self._get_ts_index(tensorstore_root_path)
    self.channel_list = channel_list
    self.metadata_cache = MetadataCache(metadata_root_path)

  def _get_ts_index(
      self, tensorstore_root_path: str) -> ts_index_lib.TensorstoreIndex:
    """Reads the tensorstore metadata for the s0 downsample layer."""
    s0_attributes_path = os.path.join(tensorstore_root_path, 's0',
                                      'attributes.json')
    s0_attributes = io_lib.read_json_file(s0_attributes_path)
    return ts_index_lib.index_from_tensorstore_metadata(s0_attributes)

  def _query_from_metadata(self, batch: str, plate: str, well: str, site: str,
                           channel: str) -> Dict[str, Any]:
    """Creates a image_grid label dictionary from metadata values."""
    plate_uid = self.metadata_cache.get_plate_uid(plate)
    plate_row, plate_col = self.metadata_cache.get_plate_row_col(plate_uid)
    well_row, well_col = self.metadata_cache.get_well_row_col(well, plate_uid)
    site_row, site_col = self.metadata_cache.get_site_row_col(site, plate_uid)
    query = {
        'batch': batch,
        'plate_row': plate_row,
        'plate_col': plate_col,
        'well_row': well_row,
        'well_col': well_col,
        'site_row': site_row,
        'site_col': site_col,
        'channel': channel,
    }
    return query

  def ts_position_from_metadata(self, batch: str, plate: str, well: str,
                                site: str, x_offset: float,
                                y_offset: float) -> Tuple[float, float]:
    """Given metadata, returns the tensorstore coordinates."""

    # (the channel is unused for position queries)
    query = self._query_from_metadata(batch, plate, well, site,
                                      self.channel_list[0])
    try:
      x = self.ts_index.get_coords_for_axis('X', query) + x_offset
      y = self.ts_index.get_coords_for_axis('Y', query) + y_offset
    except ValueError:
      # print the query to help debug which piece is problematic
      logging.info('Error finding x/y for query: %s', query)
      raise

    return x, y

  def _patch_slices_from_query(
      self, base_query: Dict[str,
                             Any], channel_list: List[str], x_offset: float,
      y_offset: float, patch_size: int) -> List[Tuple[Union[int, slice]]]:
    """Creates the image_grid slice stack."""
    query = base_query.copy()
    slices = []
    for channel in channel_list:
      query['channel'] = channel
      patch_s = self.ts_index.get_square_patch_slice(
          query, {
              'X': int(x_offset),
              'Y': int(y_offset)
          },
          patch_size,
          require_within_single_image=False)
      slices.append(patch_s)

    return slices

  def ts_get_patch_from_metadata(
      self,
      batch: str,
      plate: str,
      well: str,
      site: str,
      x_offset: float,
      y_offset: float,
      patch_size: int) -> np.ndarray:
    """Returns the np array image for the given patch location.

    Args:
      batch: String batch name, e.g. 'Experiment2022_05_02'
      plate: String plate name, e.g. 'plate32'
      well: String well name, e.g. 'D20'
      site: String site name, e.g. '0002'
      x_offset: The float x value for the center of the patch relative to the
        corner of the site. This value will be converted to an integar because
        the raw image cannot have partial pixels.
      y_offset: The float y value for the center of the patch relative to the
        corner of the site. This value will be converted to an integar because
        the raw image cannot have partial pixels.
      patch_size: The integer length of one side of the square patch.

    Returns:
    A numpy array with the raw channel images from tensorstore.
    """

    # set up slices for all the stains in this project
    base_query = self._query_from_metadata(batch, plate, well, site,
                                           self.channel_list[0])
    slices = self._patch_slices_from_query(
        base_query, self.channel_list, x_offset, y_offset, patch_size)

    spec = ts_write_lib.create_spec_from_path(
        os.path.join(self.ts_root_path, 's0'))
    dataset = ts.open(spec).result()

    patch_list = []
    for s in slices:
      patch_list.append(dataset[s].read().result())

    return np.dstack(patch_list)

  def _validate_df(self, df, required_cols):
    missing_cols = []
    for col in required_cols:
      if col not in df.columns:
        missing_cols.append(col)

    if missing_cols:
      raise ValueError('Invalid Dataframe missing columns: %s.\n'
                       'Required columns: %s\n'
                       'Columns in the input dataframe: %s\n' % (
                           missing_cols, required_cols, df.columns))

  def get_raw_images_for_df(self, example_df: pd.DataFrame, patch_size: int,
                            name_for_x_col: str,
                            name_for_y_col: str) -> List[np.ndarray]:
    """Given a dataframe, gets raw images for each row in the frame.

    Args:
      example_df: A dataframe, one row per patch. Must have columns: batch,
        plate, well, site, and x and y coordinates.
      patch_size: Integer length of one side of the square patch.
      name_for_x_col: String name of the column in the dataframe with the x
        coordinate of the patch center, within the site image. The values in
        this column can be floats or ints, but will be converted to ints in
        order to grab whole pixels for the patch image.
      name_for_y_col: String name of the column in the dataframe with the y
        coordinate of the patch center, within the site image. The values in
        this column can be floats or ints, but will be converted to ints in
        order to grab whole pixels for the patch image.

    Returns:
      A list of numpy ndarrays with the image information.
    """
    self._validate_df(example_df,
                      METADATA_COLS + [name_for_x_col, name_for_y_col])
    image_list = []

    for _, row in example_df.iterrows():
      img_for_row = self.ts_get_patch_from_metadata(
          row['batch'], row['plate'], row['well'], row['site'],
          int(row[name_for_x_col]), int(row[name_for_y_col]), patch_size)
      image_list.append(img_for_row)

    return image_list

  def contact_sheet_for_df(self,
                           example_df: pd.DataFrame,
                           patch_size: int,
                           ncols: int,
                           nrows: int,
                           name_for_x_col: str,
                           name_for_y_col: str,
                           norm_then_stack: bool = True) -> plt.Figure:
    """Given a dataframe, creates a thumbnail contact sheet.

    Args:
      example_df: A dataframe, one row per patch. It must have columns: batch,
        plate, well, site, and x and y coordinates.
      patch_size: Integer length of one side of the square patch.
      ncols: Integer number of columns in the contact sheet.
      nrows: Integer number of rows in the contact sheet.
      name_for_x_col: String name of the column in the dataframe with the x
        coordinate of the patch center, within the site image. The values in
        this column can be floats or ints, but will be converted to ints in
        order to grab whole pixels for the patch image.
      name_for_y_col: String name of the column in the dataframe with the y
        coordinate of the patch center, within the site image. The values in
        this column can be floats or ints, but will be converted to ints in
        order to grab whole pixels for the patch image.
      norm_then_stack: Boolean indicating how normalization should be done. The
        default True indicates that each stain will be normalized to the full
        brightness range, then the stains will be stacked. This makes sure dim
        stains are visible in the composite. A False value will stack the stains
        and then normalize, ensuring that relative brightness values are
        conserved.

    Returns:
      A matplotlib figure with the contact sheet.
    """
    self._validate_df(example_df,
                      METADATA_COLS + [name_for_x_col, name_for_y_col])
    # Getting images can take a while. Fail quickly if we know we have too
    # many examples.
    if ncols * nrows < len(example_df):
      raise ValueError(
          'The example dataframe has more examples than the rows/cols.\n'
          'There are %d examples but %d rows and %d cols (room for %d)' %
          (len(example_df), nrows, ncols, nrows * ncols))

    raw_image_list = self.get_raw_images_for_df(
        example_df,
        name_for_x_col=name_for_x_col,
        name_for_y_col=name_for_y_col,
        patch_size=patch_size)
    if norm_then_stack:
      norm_imgs = [
          image_lib.normalize_per_color_image(x) for x in raw_image_list
      ]
    else:
      norm_imgs = [image_lib.normalize_image(x) for x in raw_image_list]
    figure = image_lib.create_contact_sheet(norm_imgs, ncols=ncols, nrows=nrows)

    return figure

  def canonical_per_parasite_for_df(
      self, example_df: pd.DataFrame, patch_size: int, name_for_x_col: str,
      name_for_y_col: str, figure_config: List[Any],
      stain_to_index_map: Dict[Any, int]) -> List[plt.Figure]:
    """Returns one figure for each row of the dataframe.

    Args:
      example_df: A dataframe, one row per patch. Must have columns batch,
        plate, well, site, and x and y coordinates.
      patch_size: Integer length of one side of the square patch.
      name_for_x_col: String name of the column in the dataframe with the x
        coordinate of the patch center, within the site image. The values in
        this column can be floats or ints, but will be converted to ints in
        order to grab whole pixels for the patch image.
      name_for_y_col: String name of the column in the dataframe with the y
        coordinate of the patch center, within the site image. The values in
        this column can be floats or ints, but will be converted to ints in
        order to grab whole pixels for the patch image.
      figure_config: List representing the images to show in the figure.
      stain_to_index_map: Dictionary mapping the stain value used in the config
        to an index in the stain_stack_img.
    """
    self._validate_df(example_df,
                      METADATA_COLS + [name_for_x_col, name_for_y_col])
    fig_list = []
    for _, row in example_df.iterrows():
      img_for_row = self.ts_get_patch_from_metadata(row['batch'], row['plate'],
                                                    row['well'], row['site'],
                                                    int(row[name_for_x_col]),
                                                    int(row[name_for_y_col]),
                                                    patch_size)
      fig = image_lib.create_multi_figure(img_for_row, figure_config,
                                          stain_to_index_map)
      fig_list.append(fig)

    return fig_list


class MetadataCache(object):
  """Wraps metadata retrieval and caches results."""

  def __init__(self, metadata_root_path):
    self.metadata_root_path = metadata_root_path
    self.cached_plate_uids = {}  # keys are plate numbers
    self.cached_batches = {}  # keys are plate_uids
    self.cached_plate_layout_ints = {}  # keys are plate_uids
    self.cached_site_layouts = {}  # keys are plate_uids
    self.cached_well_layouts = {}  # keys are plate_uids
    self.cached_site_pos = {}  # Nested dict; keys (site_layout_int, site_str)
    self.cached_well_pos = {}  # Nested dict; keys (well_layout_int, well_str)

    self.read_and_cache_plate_level()

  def get_plate_uid(self, plate):
    """Return the plate uid for a given (plate,).

    Args:
      plate: String representing the plate name

    Returns:
      plate_uid: String representing the plate uid
    Raises:
      ValueError: If the requested plate uid is not found.
    """

    return self.cached_plate_uids[plate]

  def get_batch(self, plate_uid):
    """Retrieve the batch for the given plate uid.

    Args:
      plate_uid: String representing the plate uid

    Returns:
      batch: String name of the experimental batch for this plate.
    Raises:
      ValueError: If the requested plate uid is not found.
    """

    return self.cached_batches[plate_uid]

  def get_plate_layout_int(self, plate_uid):
    """Retrieve the plate_layout_int for display in Neuroglancer.

    Args:
      plate_uid: String representing the plate uid

    Returns:
      plate_layout_int: Int representing the number of plates per row in
        Neuroglancer.
    """

    return self.cached_plate_layout_ints[plate_uid]

  def get_site_layout(self, plate_uid):
    """Retrieve the site_layout_int for display in Neuroglancer.

    Args:
      plate_uid: String representing the plate_uid

    Returns:
      site_layout: String name of the site layout scheme.
    """

    return self.cached_site_layouts[plate_uid]

  def get_well_layout(self, plate_uid):
    """Retrieve the well_layout_int for display in Neuroglancer.

    Args:
      plate_uid: String representing the plate uid

    Returns:
      well_layout: String name of the well layout scheme
    """

    return self.cached_well_layouts[plate_uid]

  def get_plate_row_col(self, plate_uid):
    """Calculate the plate_row and plate_col for display in neuroglancer.

    This is used when calculating neuroglancer coordinates. We need to calculate
    neuroglancer coordinates when writing tensorstore objects and when creating
    annotation layers.

    Args:
      plate_uid: String representing the plate uid

    Returns:
      plate_row: String representing the row to display this plate in
      plate_col: String representing the column to display this plate in
    """
    plate_layout_int = self.get_plate_layout_int(plate_uid)
    plate_int = int(plate_uid)
    plate_row_int = plate_int // plate_layout_int
    plate_col_int = plate_int % plate_layout_int

    return str(plate_row_int), str(plate_col_int)

  def get_site_row_col(self, site, plate_uid):
    """Calculate the site_row and site_col for display in neuroglancer.

    This is used when calculating neuroglancer coordinates. We need to calculate
    neuroglancer coordinates when writing tensorstore objects and when creating
    annotation layers.

    Args:
      site: String representing the site (e.g. "00001")
      plate_uid: String representating the plate, to look up the site layout.

    Returns:
      site_row: String representing the row to display this site in
      site_col: String representing the column to display this site in
    """
    site_layout = self.get_site_layout(plate_uid)
    return self.cached_site_pos[site_layout][site]

  def get_well_row_col(self, well, plate_uid):
    """Calculate the well_row and well_col for display in neuroglancer.

    This is used when calculating neuroglancer coordinates. We need to calculate
    neuroglancer coordinates when writing tensorstore objects and when creating
    annotation layers.

    Args:
      well: String representing the well (e.g. "A10")
      plate_uid: String representating the plate, to look up the site layout.

    Returns:
      well_row: String representing the row to display this well in
      well_col: String representing the column to display this well in
    """
    well_layout = self.get_well_layout(plate_uid)
    return self.cached_well_pos[well_layout][well]

  def read_and_cache_plate_level(self):
    """Code to read in the plate-level csv.

    Returns:
      df: A pd.DataFrame indexed by ['plate'] containing plate level data:
        plate, plate_uid, site_layout, well_layout, batch, and plate_layout_int.
    Raises:
      AttributeError: If file doesn't exist
    """
    path = os.path.join(self.metadata_root_path, 'experiment', 'plate_uids.csv')

    plate_df = io_lib.read_csv(path, dtype=str)
    plate_df['plate_layout_int'] = plate_df['plate_layout_int'].astype(int)
    for _, prow in plate_df.iterrows():
      # save all the plate level data
      self.cached_plate_uids[prow['plate']] = prow['plate_uid']
      self.cached_batches[prow['plate_uid']] = prow['batch']
      self.cached_plate_layout_ints[prow['plate_uid']] = (
          prow['plate_layout_int'])
      self.cached_well_layouts[prow['plate_uid']] = prow['well_layout']
      self.cached_site_layouts[prow['plate_uid']] = prow['site_layout']

    # save the position of each well for each well layout that is used.
    all_well_layouts = plate_df.well_layout.unique()
    for well_layout in all_well_layouts:
      self.cached_well_pos[well_layout] = {}
      well_layout_path = os.path.join(
          self.metadata_root_path, WELL_LAYOUT_DICT[well_layout])
      well_df = io_lib.read_csv(well_layout_path, index_col=0, dtype=str)
      for _, row in well_df.iterrows():
        self.cached_well_pos[well_layout][row['well']] = (
            row['well_row'], row['well_col'])

    # save the position of the site for each site layout that is used.
    all_site_layouts = plate_df.site_layout.unique()
    for site_layout in all_site_layouts:
      self.cached_site_pos[site_layout] = {}
      site_layout_path = os.path.join(self.metadata_root_path,
                                      SITE_LAYOUT_DICT[site_layout])
      site_df = io_lib.read_csv(site_layout_path, index_col=0, dtype=str)
      for _, row in site_df.iterrows():
        self.cached_site_pos[site_layout][row['site']] = (
            row['site_row'], row['site_col'])

    plate_df.set_index(['plate',], inplace=True)
    return plate_df


