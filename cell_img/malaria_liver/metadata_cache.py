
"""Library for caching malaria_liver metadata.

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
from cell_img.common import io_lib

WELL_LAYOUT_DICT = {
    '0': 'experiment/well_0.csv'
}

SITE_LAYOUT_DICT = {
    '0': 'experiment/site_0.csv',
    '1': 'experiment/site_1.csv',
}


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
