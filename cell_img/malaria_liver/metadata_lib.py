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

import math
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


class MetadataIndex(object):
  """Convenience wrapper on the TensorstoreIndex for access."""

  def __init__(self, tensorstore_root_path, channel_list, metadata_root_path):
    """Constructor.

    Args:
      tensorstore_root_path: String path of the directory where the
        tensorstore metadata dictionary is stored. It is expected that the
        tensorstore metadata will be at:
        tensorstore_root_path/s0/attributes.json
      channel_list: String list of channel names for the image stack,
        e.g. ['w3', 'w2', 'w1']. Images pulled from tensorstore will have
        these channels in this order (so, you often want them in RGB order).
    """
    self.ts_root_path = tensorstore_root_path
    self.ts_index = self._get_ts_index(tensorstore_root_path)
    self.channel_list = channel_list
    self.metadata_root_path = metadata_root_path

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
    query = {
        'batch': batch,
        'plate_row': str(int(plate[:3])),
        'plate_col': str(int(plate[3:])),
        'well_row': well[0],
        'well_col': well[1:],
        'site_row': str(math.floor((int(site) - 1) / 5)),
        'site_col': str((int(site) - 1) % 5),
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
      example_df: A dataframe, one row per patch. Must have columns: batch,
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
