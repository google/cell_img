"""Library for fetching and preprocessing images prior to analysis."""
import copy
from typing import Any, Dict, List

from cell_img.common import io_lib
from cell_img.malaria_liver.parasite_emb import config

import numpy as np
import pandas as pd

# This is the input CSV with the filenames for the whole image files.
IMAGE_REQUIRED_COLUMNS = [config.CHANNEL, config.IMAGE_PATH, config.PLATE,
                          config.WELL, config.SITE]
# This is the input CSV with metadata per well, used when running the
# pipeline that finds and stages parasites.
WELL_METADATA_REQUIRED_COLUMNS = [config.PLATE, config.WELL]
# This is the input CSV with the metadata per object, used when finding and
# staging is done outside this pipeline.
OBJECT_METADATA_REQUIRED_COLUMNS = [
    config.PLATE, config.WELL, config.SITE, config.CENTER_ROW,
    config.CENTER_COL, config.STAGE_RESULT]


def _validate_columns(df: pd.DataFrame, required_columns: List[str],
                      df_name_for_error: str):
  cols_not_found = []
  for c in required_columns:
    if c not in df.columns:
      cols_not_found.append(c)
  if cols_not_found:
    raise ValueError('The dataframe %s is missing required columns: %s' % (
        df_name_for_error, cols_not_found))


def load_and_validate_one_csv(
    csv_path: str, required_cols: List[str], name_for_error_str: str
    ) ->pd.DataFrame:
  """Loads dataframes and checks required columns."""

  df = io_lib.read_csv(csv_path)
  # if plate_uid exists but no plate, use plate for now.
  if config.PLATE not in df.columns:
    if config.PLATE_UID in df.columns:
      df[config.PLATE] = df[config.PLATE_UID]
  _validate_columns(df, required_cols, name_for_error_str)

  # If the CSV was saved with no index, drop the unnamed column
  unnamed_c = [c for c in df.columns if c.startswith('Unnamed:')]
  if unnamed_c:
    df = df.drop(columns=unnamed_c)

  return df


def prep_image_df(image_csv: str):
  """Loads the image df from csv, validates and preps it."""
  image_metadata_df = load_and_validate_one_csv(
      image_csv, IMAGE_REQUIRED_COLUMNS, 'image')

  image_columns = ['channel', 'image_path']
  # Stain columns may not always be present, but if they are present, then
  # they prevent proper aggregation of the image filepaths.
  image_columns += [
      col for col in image_metadata_df.columns if col.startswith('stain')
  ]
  image_groupby_columns = [
      col for col in image_metadata_df.columns if col not in image_columns
  ]
  image_metadata_df = image_metadata_df.groupby(image_groupby_columns)[[
      config.CHANNEL, config.IMAGE_PATH
  ]].agg(' '.join).reset_index()

  return image_metadata_df


def read_metadata(image_csv: str, well_metadata_csv: str
                  ) -> List[Dict[str, Any]]:
  """Extracts image and well metadata dataFrames and merge them."""

  image_metadata_df = prep_image_df(image_csv)
  well_metadata_df = load_and_validate_one_csv(
      well_metadata_csv, WELL_METADATA_REQUIRED_COLUMNS, 'well_metadata')

  df = pd.merge(
      image_metadata_df,
      well_metadata_df,
      how='inner',
      on=[
          config.PLATE,
          config.WELL,
      ]).reset_index()

  # validate that all the images will be loaded
  image_paths_input = set(image_metadata_df.image_path.unique())
  image_paths_merged = set(df.image_path.unique())
  image_paths_no_metadata = image_paths_input - image_paths_merged
  if image_paths_no_metadata:
    raise ValueError('Some of the image paths did not have associated well '
                     'metadata:\n   ' + '\n   '.join(image_paths_no_metadata))

  # Modify columns to match expected format.
  df[config.PLATE] = df[config.PLATE].apply(
      lambda x: str(x).zfill(5))
  df[config.SITE] = df[config.SITE].apply(
      lambda x: str(x).zfill(5))

  if config.BATCH not in df.columns:
    df[config.BATCH] = df[config.PLATE]
  if config.PLATE_UID not in df.columns:
    df[config.PLATE_UID] = df[config.PLATE]

  return df.to_dict('records')


def read_object_metadata(image_csv: str, object_metadata_csv: str
                         ) -> List[Dict[str, Any]]:
  """Extracts image and object metadata dataFrames and merge them."""

  image_metadata_df = prep_image_df(image_csv)
  object_metadata_df = load_and_validate_one_csv(
      object_metadata_csv, OBJECT_METADATA_REQUIRED_COLUMNS, 'object_metadata')

  df = pd.merge(
      image_metadata_df,
      object_metadata_df,
      how='inner',
      on=[
          config.PLATE,
          config.WELL,
          config.SITE,
      ]).reset_index()

  # validate that all the found centers have images after the inner merge
  # (it is okay if some images are unused, we do not expect that every site
  # must have objects)
  if df.shape[0] != object_metadata_df.shape[0]:
    raise ValueError('Some of the objects did not have associated images. '
                     'There were %d objects, now %d object rows after merge.' %
                     (object_metadata_df.shape[0], df.shape[0]))

  # Modify columns to match expected format.
  df[config.PLATE] = df[config.PLATE].apply(
      lambda x: str(x).zfill(5))
  df[config.SITE] = df[config.SITE].apply(
      lambda x: str(x).zfill(5))

  if config.BATCH not in df.columns:
    df[config.BATCH] = df[config.PLATE]
  if config.PLATE not in df.columns:
    df[config.PLATE] = df[config.PLATE]

  # group by site and create one record per site with a list of patches
  records = []
  groupby_cols = [config.BATCH, config.PLATE, config.PLATE, config.WELL,
                  config.SITE, config.CHANNEL, config.IMAGE_PATH]
  for grouped_items, site_df in df.groupby(groupby_cols):
    one_record = {}
    for i, groupby_col_name in enumerate(groupby_cols):
      one_record[groupby_col_name] = grouped_items[i]
    one_record[config.CENTER_RECORDS] = (
        site_df[OBJECT_METADATA_REQUIRED_COLUMNS].to_dict('records'))
    records.append(one_record)
  return records


def load_img(elem: Dict[str, Any], raw_channel_order: List[str],
             channel_order: List[str],
             whole_image_size: List[int]) -> Dict[str, Any]:
  """Loads images given their sites and paths."""
  all_images = []
  for channel, channel_path in zip(elem['channel'].split(),
                                   elem['image_path'].split()):
    img = io_lib.read_image(channel_path)
    img = img / 65535.
    if tuple(img.shape) != tuple(whole_image_size):
      raise ValueError(f'Image: {channel_path} has shape {img.shape} when '
                       f'expected {whole_image_size}')
    channel_num = raw_channel_order.index(channel)
    all_images.append((channel_num, img))
  # Ensure that we have all the expected number of channels.
  if len(all_images) != len(channel_order):
    channel_paths = elem['image_path'].split()
    raise ValueError(
        f'Found {len(all_images)} images when expected {len(channel_order)}. '
        f'Image paths: {channel_paths}')
  # Sort the images by the channel so there is a consistent order.
  all_images = sorted(all_images, key=lambda x: x[0])
  # Create a single multi-channel image and add to dictionary.
  all_images = [img for _, img in all_images]
  elem[config.IMAGE] = np.stack(all_images, axis=-1)
  elem['channel_order'] = channel_order
  return elem


def convert_img_for_output(element: Dict[str, Any]) -> Dict[str, Any]:
  """Flatten the image to save out in parquet."""
  element_dict = copy.deepcopy(element)
  img = element_dict[config.IMAGE]
  img_dict = {'shape': img.shape, 'values': img.flatten().astype(np.float32)}
  element_dict[config.IMAGE] = img_dict
  return element_dict


def log_and_rescale_img(elem: Dict[str, Any], log_brightness_min: List[float],
                        log_brightness_max: List[float]) -> Dict[str, Any]:
  """Log and rescale images given a min and max. Rescaled images are [0, 1]."""
  img = elem[config.IMAGE].astype(np.float32)
  num_channels = img.shape[-1]

  if len(log_brightness_max) != num_channels:
    raise ValueError('Number of channels %d does not match number of max '
                     'brightness values %d' %
                     (num_channels, len(log_brightness_max)))

  if len(log_brightness_min) != num_channels:
    raise ValueError('Number of channels %d does not match number of min '
                     'brightness values %d' %
                     (num_channels, len(log_brightness_min)))

  def _log_one_channel_image(image, log_min, log_max):
    if not log_min < log_max:
      raise ValueError('Log brightness min %d not less than log brightness max '
                       '%d' % (log_min, log_max))
    min_val = np.exp(log_min)
    max_val = np.exp(log_max)

    if min_val <= 0:
      raise ValueError('Brightness min %d is <= 0' % min_val)

    # Clip to (min, max) to avoid taking the log of a 0-value pixel
    clip_image = np.clip(image, min_val, max_val)
    return np.log(clip_image)

  def _rescale_one_channel_image(log_image, log_min, log_max):
    if not log_min < log_max:
      raise ValueError('Log brightness min %d not less than log brightness max '
                       '%d' % (log_min, log_max))

    scaled_image = (log_image - log_min) / (log_max - log_min)
    return np.clip(scaled_image, 0., 1.)

  for c in range(num_channels):
    log_img = _log_one_channel_image(img[:, :, c], log_brightness_min[c],
                                     log_brightness_max[c])
    img[:, :, c] = _rescale_one_channel_image(log_img, log_brightness_min[c],
                                              log_brightness_max[c])
  elem[config.IMAGE] = img
  return elem
