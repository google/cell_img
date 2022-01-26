"""Utilities for reading and writing data from GCS buckets from colab.

Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import os
import re

import base64
import zlib
import gzip

import numpy as np
import pandas as pd
import fsspec

INTERNAL_HEADER=''

def do_external_setup(project_id):
  # DO NOT use user authentication from VM kernels!
  with open('/var/colab/hostname') as f:
    vm_name = f.read()
  if vm_name.startswith('colab-'):
    del os.environ['NO_GCE_CHECK']
    del os.environ['GCE_METADATA_TIMEOUT']
  else:
    from google.colab import auth
    auth.authenticate_user()
  project_id = project_id



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
  if not np.all([re.match(r'^\d+$', p) for p in formatted_plates.values]):
    raise ValueError('Plate with non-digit characters found')
  return formatted_plates


def read_csv_from_cloud(filepath):
  """Loads a csv into a dataframe from cloud storage.

  Designed to work either internal or external depending
  on a global "IS_INTERNAL" constant being set.

  Any spaces or dashes in column names are converted to underscores so
  they play nice with pandas.

  Any 'plate' columns will be adjusted to be the 5 character string used
  by Google. Other columns will NOT be adjusted, you may need to do
  additional cleaning up because CSV does not save type information.

  Args:
    filepath: The cloud path to the file, starting from the bucket.
      Example: 'bucket_name/folder_name/test.csv'

  Returns:
    pandas dataframe with the contents of the CSV.
  """

  if IS_INTERNAL:
    internal_filepath = os.path.join(INTERNAL_HEADER, filepath)
    with gfile.Open(internal_filepath) as infile:
      df = pd.read_csv(infile)
  else:
    external_filepath = os.path.join('gs://', filepath)
    df = pd.read_csv(external_filepath)

  if 'plate' in df.columns:
    df['plate'] = format_plate_strings(df['plate'])

  column_names = list(df.columns)
  column_names = [x.replace(' ', '_') for x in column_names]
  column_names = [x.replace('-', '_') for x in column_names]
  df.columns = column_names

  return df


def to_csv_on_cloud(df, filepath):
  """Writes the dataframe to cloud as a CSV.

  Designed to work either internal or external depending
  on a global "IS_INTERNAL" constant being set.

  Args:
    df: A pandas dataframe.
    filepath: A cloud bucket path, eg:
      'bucket_name/folder_name/test.csv'
  """

  if IS_INTERNAL:
    internal_filepath = os.path.join(INTERNAL_HEADER, filepath)
    with gfile.Open(internal_filepath, 'w') as outfile:
      df.to_csv(outfile)
  else:
    external_filepath = os.path.join('gs://', filepath)
    df.to_csv(external_filepath)


def to_file_on_cloud(txt, filepath):
  """Writes the given text to cloud.

  Designed to work either internal or external depending
  on a global "IS_INTERNAL" constant being set.

  Args:
    txt: The text to write to the cloud.
    filepath: A cloud bucket path, eg:
      'bucket_name/folder_name/test.html'
  """

  if IS_INTERNAL:
    internal_filepath = os.path.join(INTERNAL_HEADER, filepath)
    with gfile.Open(internal_filepath, 'wt') as f:
      f.write(txt)
  else:
    import fsspec
    external_filepath = os.path.join('gs://', filepath)
    with fsspec.open(external_filepath, 'w') as f:
      f.write(txt)

def read_h5_df_from_cloud(filepath):
  if IS_INTERNAL:
    internal_filepath = os.path.join(INTERNAL_HEADER, filepath)
    internal_filepaths = gfile.Glob(internal_filepath)
    df_shards = [pd.DataFrame()]
    for path in internal_filepaths:
      with gfile.GFile(path, 'r') as f:
        with pd.HDFStore(
            'in_memory',
            mode='r',
            driver='H5FD_CORE',
            driver_core_backing_store=0,
            driver_core_image=f.read()) as store:
          df_shards.append(store['data'])
  else:
    import fsspec
    external_filepath = os.path.join('gs://', filepath)
    # We can't read h5 files directly from cloud;
    # I'll download it to a temp folder and load it here
    # fsspec.open_files handles glob if '*' is in filepath
    openfiles = fsspec.open_files(external_filepath)
    # Make a random string to attach to the filename to prevent people
    # from writing to the file at the same time
    randstr = str(np.random.random())[2:]
    df_shards = [pd.DataFrame()]
    for idx, openfile in enumerate(openfiles):
      with openfile as f:
        data = f.read()
      temp_path = 'temp_h5_{}.h5-{:05d}'.format(randstr,idx)
      with open(temp_path,'wb') as fw:
        fw.write(data)
      df_shards.append(pd.read_hdf(temp_path))
      os.remove(temp_path)
  return pd.concat(df_shards).sort_index()
