"""Global constants and utils for parquet conversion."""

import re

import numpy as np
import pyarrow

BATCH = 'batch'
PLATE = 'plate'
PLATE_UID = 'plate_uid'
WELL = 'well'
SITE = 'site'
CENTER_ROW = 'center_row'
CENTER_COL = 'center_col'
CROP_ROW = 'crop_row'
CROP_COL = 'crop_col'
HYPNOZOITE = 'hypnozoite'
SCHIZONT = 'schizont'
ARTIFACT = 'artifact'
ARTIFACT_PRED = 'artifact_pred'
STAGE_INFER = 'parasite_stage_infer'
STAGE_NAMES = 'parasite_stage_names'
STAGING_CLASSES = [ARTIFACT, HYPNOZOITE, SCHIZONT]
STAGE_RESULT = 'stage_result'
FINDING_CONFIDENCE = 'finding_confidence'
FINDING_OVERLAP = 'finding_overlap'
MEAN_EMBEDDING = 'mean_embedding'

CHANNEL = 'channel'
IMAGE_PATH = 'image_path'
CHANNEL_ORDER = 'channel_order'
IMAGE = 'image'
EMBEDDING = 'embedding'


def format_plate_strings(plate_names):
  """Format the plate strings as strings of five digit ints.

  Args:
    plate_names: A pd.Series of strings representing the plate names that we
      want to format.
  Raises:
    ValueError: If plate_names contains a name that is more than five digits
      long.
  Returns:
    formatted_plates: A pd.Series representing the formatted plate names.
  """
  # Format the plate strings as 5 character strings with no decimal
  formatted_plates = plate_names.astype(str).apply(
      lambda x: x.split('.')[0].zfill(5))
  # If any of the plates are more than 5 digits, scream loudly.
  len_plate_names = np.array([len(p) for p in formatted_plates.values])
  if np.any(len_plate_names > 5):
    raise ValueError('Plate name > 5 characters found')
  # If any of the plates have non-digit characters, scream loudly.
  if not np.all([re.fullmatch(r'^\d+', p) for p in formatted_plates.values]):
    raise ValueError('Plate with non-digit characters found')
  return formatted_plates


def get_whole_image_schema(whole_image_dim1, whole_image_dim2,
                           num_channels):
  flat_image_size = whole_image_dim1 * whole_image_dim2 * num_channels
  image_schema = pyarrow.struct([
      ('shape', pyarrow.list_(pyarrow.uint16(), list_size=3)),
      ('values', pyarrow.list_(pyarrow.float32(), list_size=flat_image_size))
  ])
  return pyarrow.schema([
      # Schema is minimal metadata to enable a late-join on metadata.
      (BATCH, pyarrow.string()),
      (PLATE, pyarrow.string()),
      (WELL, pyarrow.string()),
      (SITE, pyarrow.string()),
      (IMAGE, image_schema),
      (CHANNEL_ORDER, pyarrow.list_(pyarrow.string(), list_size=num_channels)),
  ])


def get_processed_patch_schema(patch_image_dim1, patch_image_dim2, num_channels,
                               emb_dim_size):
  flat_image_size = patch_image_dim1 * patch_image_dim2 * num_channels
  image_schema = pyarrow.struct([
      ('shape', pyarrow.list_(pyarrow.uint16(), list_size=3)),
      ('values', pyarrow.list_(pyarrow.float32(), list_size=flat_image_size))
  ])
  return pyarrow.schema([
      (BATCH, pyarrow.string()),
      (PLATE, pyarrow.string()),
      (WELL, pyarrow.string()),
      (SITE, pyarrow.string()),
      (CENTER_ROW, pyarrow.int32()),
      (CENTER_COL, pyarrow.int32()),
      (CROP_ROW, pyarrow.int32()),
      (CROP_COL, pyarrow.int32()),
      (EMBEDDING, pyarrow.list_(pyarrow.float32(), list_size=emb_dim_size)),
      (STAGE_INFER,
       pyarrow.list_(pyarrow.float32(), list_size=len(STAGING_CLASSES))),
      (STAGE_NAMES,
       pyarrow.list_(pyarrow.string(), list_size=len(STAGING_CLASSES))),
      (IMAGE, image_schema),
      (CHANNEL_ORDER, pyarrow.list_(pyarrow.string(), list_size=num_channels)),
      (ARTIFACT_PRED, pyarrow.float32()),
      (STAGE_RESULT, pyarrow.string()),
      (FINDING_CONFIDENCE, pyarrow.float32()),
      (FINDING_OVERLAP, pyarrow.bool_())
  ])


def get_mean_embedding_schema(emb_dim_size):
  return pyarrow.schema([
      (BATCH, pyarrow.string()),
      (PLATE, pyarrow.string()),
      (WELL, pyarrow.string()),
      (MEAN_EMBEDDING,
       pyarrow.list_(pyarrow.float32(), list_size=emb_dim_size)),
  ])
