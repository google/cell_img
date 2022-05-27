"""Sample data for unit tests."""

import itertools
import os

from cell_img.common import io_lib
import numpy as np
import pandas as pd

AXES = ['Y', 'X', 'stain']  # Put Y, X to match numpy image array (row col).
AXES_WRAP = {
    'X': ['plate', 'well_col', 'site_col'],
    'Y': ['batch', 'well_row', 'site_row']
}
IMAGE_DTYPE = 'uint8'
IMAGE_SHAPE = (3, 5)  # Small so tests go fast. Non-square to catch bugs.
UNIT_SIZES = {'X': IMAGE_SHAPE[1], 'Y': IMAGE_SHAPE[0]}
IMAGE_PATH_COL = 'image_path'

BATCH_LIST = ['batch1', 'batch2']
PLATE_LIST = ['plate1', 'plate2', 'plate3']
WELL_ROW_LIST = ['A', 'B']
WELL_COL_LIST = ['1', '2']
SITE_ROW_LIST = ['site_row1', 'site_row2']
SITE_COL_LIST = ['site_col1', 'site_col2']
STAIN_LIST = ['stain1', 'stain2']


def generate_image_metadata_df():
  records = list(
      itertools.product(BATCH_LIST, PLATE_LIST, WELL_ROW_LIST, WELL_COL_LIST,
                        SITE_ROW_LIST, SITE_COL_LIST, STAIN_LIST))
  return pd.DataFrame.from_records(
      records,
      columns=[
          'batch', 'plate', 'well_row', 'well_col', 'site_row', 'site_col',
          'stain'
      ])


def write_image_metadata_csv_and_images(output_dir, image_metadata_df):
  """Writes the image metadata csv and images to the output directory."""

  def get_img_path(elem):
    # Make a unique filename for the image by joining all the metadata.
    # Filename is intended to be unique but not parsed.
    filename = '-'.join(elem) + '.png'
    return os.path.join(output_dir, filename)

  image_metadata_df[IMAGE_PATH_COL] = image_metadata_df.apply(
      get_img_path, axis='columns')

  for _, row in image_metadata_df.iterrows():
    array = np.random.randint(0, 255, size=IMAGE_SHAPE, dtype=IMAGE_DTYPE)
    io_lib.write_image(array, row['image_path'])

  image_metadata_path = os.path.join(output_dir, 'image_metadata.csv')
  io_lib.write_csv(
      image_metadata_df, image_metadata_path, header=True, index=False)
  return image_metadata_path
