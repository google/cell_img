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
import base64
import gzip
import os
import re

import fsspec
import numpy as np
import pandas as pd

CLOUD_HEADER = 'gs://'


def do_external_setup(project_id):
  """Sets up GCP credentials for external colab.

  Args:
    project_id: GCP project id associated with the colab.
  """
  with open('/var/colab/hostname') as f:
    vm_name = f.read()
  if vm_name.startswith('colab-'):
    del os.environ['NO_GCE_CHECK']
    del os.environ['GCE_METADATA_TIMEOUT']
  else:
    from google.colab import auth
    auth.authenticate_user()
  os.environ['GOOGLE_CLOUD_PROJECT'] = project_id


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

  Any spaces or dashes in column names are converted to underscores so
  they play nice with pandas.

  Any 'plate' columns will be adjusted to be the 5 character string used
  by Google. Other columns will NOT be adjusted, you may need to do
  additional cleaning up because CSV does not save type information.

  Args:
    filepath: The cloud path to the file, starting from the bucket. Example -
      'bucket_name/folder_name/test.csv'

  Returns:
    pandas dataframe with the contents of the CSV.
  """
  cloud_filepath = os.path.join(CLOUD_HEADER, filepath)
  df = pd.read_csv(cloud_filepath)

  if 'plate' in df.columns:
    df['plate'] = format_plate_strings(df['plate'])

  column_names = list(df.columns)
  column_names = [x.replace(' ', '_') for x in column_names]
  column_names = [x.replace('-', '_') for x in column_names]
  df.columns = column_names

  return df


def to_csv_on_cloud(df, filepath):
  """Writes the dataframe to cloud as a CSV.

  Args:
    df: A pandas dataframe.
    filepath: A cloud bucket path, eg: 'bucket_name/folder_name/test.csv'
  """
  cloud_filepath = os.path.join(CLOUD_HEADER, filepath)
  with fsspec.open(cloud_filepath, 'w') as outfile:
    df.to_csv(outfile)


def to_file_on_cloud(txt, filepath):
  """Writes the given text to cloud.

  Args:
    txt: The text to write to the cloud.
    filepath: A cloud bucket path, eg: 'bucket_name/folder_name/test.html'
  """
  cloud_filepath = os.path.join(CLOUD_HEADER, filepath)
  with fsspec.open(cloud_filepath, 'w') as f:
    f.write(txt)


def read_h5_df_from_cloud(filepath):
  """Reads dataframes saved as HDF5 format from cloud.

  Args:
    filepath: A cloud bucket path, eg: 'bucket_name/folder_name/test.html'

  Returns:
    pd.DataFrame read from cloud
  """
  cloud_filepath = os.path.join(CLOUD_HEADER, filepath)

  # We can't read h5 files directly from cloud;
  # I'll download it to a temp folder and load it here
  # fsspec.open_files handles glob if '*' is in filepath
  openfiles = fsspec.open_files(cloud_filepath)
  # Make a random string to attach to the filename to prevent people
  # from writing to the file at the same time
  randstr = str(np.random.random())[2:]
  df_shards = [pd.DataFrame()]
  for idx, openfile in enumerate(openfiles):
    with openfile as f:
      data = f.read()
    temp_path = 'temp_h5_{}.h5-{:05d}'.format(randstr, idx)
    with open(temp_path, 'wb') as fw:
      fw.write(data)
    df_shards.append(pd.read_hdf(temp_path))
    os.remove(temp_path)
  return pd.concat(df_shards).sort_index()


def merge_dfs(df_list, keys_for_merge):
  """Joins all the dataframes on the keys.

  Does an outer join, does not check that all rows exist in all dataframes.

  Args:
    df_list: A list of pandas dataframes. Each needs to have the keys_for_merge
      as columns.
    keys_for_merge: A string list of column names to merge on. Typically
      ['plate', 'well'] aka WELL_MERGE_KEYS.

  Returns:
    The joined pandas dataframe.
  """
  if len(df_list) < 1:
    raise ValueError('I need at least 1 df to merge')

  for k in keys_for_merge:
    if k not in df_list[0]:
      raise ValueError('df missing column %s. Has: %s' %
                       (k, df_list[0].columns))
  merged_df = df_list[0]

  for df in df_list[1:]:
    cols_to_add = keys_for_merge + [c for c in df.columns if c not in merged_df]
    for k in keys_for_merge:
      if k not in df:
        raise ValueError('df missing column %s. Has: %s' % (k, df[0].columns))
    merged_df = merged_df.merge(df[cols_to_add], on=keys_for_merge, how='outer')

  return merged_df


def get_html_list(df_for_html, base_neuroglancer_url):
  """Creates the html for one pandas dataframe as a neuroglancer table.

  Args:
    df_for_html: The pandas dataframe to create the html from.
    base_neuroglancer_url: The string url for neuroglancer to load in the top
      iframe.

  Returns:
    A list of HTML strings that can be written to make the HTML page.
  """

  df_html = df_for_html.to_html(escape=False, table_id='link_df')
  encoded_html = base64.encodebytes(gzip.compress(
      df_html.encode('utf-8'))).decode().rstrip()

  include_html = """
  <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
  <script src="https://rawgit.com/nodeca/pako/master/dist/pako.js"></script>
  <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.js"></script>
  """
  # note backticks! these capture the \n's introduced by base64.encodebytes
  encoding_invocation = f'encoded_html=`{encoded_html}`;\n'
  invocation_js = encoding_invocation + """
  function stringify(x) {
    accum = [];
    for (var i=0; i<x.length; i+= 10000) {
        accum.push(String.fromCharCode.apply(null, x.slice(i, Math.min(i + 10000, x.length))));
    }
    return(accum.join(''));
  }
  function decode_base64_encoded_gzip(encoded_html) {
      gzip_encoded_bytes = new Uint8Array(atob(encoded_html).split('').map(function(x) {return x.charCodeAt(0)}));
      decoded_bytes = pako.inflate(gzip_encoded_bytes);
      decoded_bytes16 = new Uint16Array(decoded_bytes);
      decoded_html_str = stringify(decoded_bytes16);
      return decoded_html_str;
  }
  decoded_html_str = decode_base64_encoded_gzip(encoded_html);

  $('body').append($(decoded_html_str));
  $(document).ready( function () {
      $('#link_df').DataTable();
      $('#wait').hide();
  } );
  """

  accum = []
  accum.append(include_html)
  accum.append(
      f'<iframe src="{base_neuroglancer_url}" height="900" width="1600" title="neuroglancer" name="neuroglancer"> </iframe>'
  )
  accum.append('<p>')
  base_neuroglancer_link = ('<a href="%s" target="neuroglancer_context">Open '
                            'Neuroglancer</a>') % (
                                base_neuroglancer_url)
  accum.append(base_neuroglancer_link)
  accum.append(
      '<div id="wait"><h4>Please wait while the table fully loads.</h4></div>')
  accum.append(f'<script>{invocation_js}</script>')
  return accum


def set_up_for_html(df, url_col_name):
  """Moves the url_col_name so it is the first column in the html output."""
  if url_col_name not in df.columns:
    raise ValueError('expected url col "%s" but did not find it.' %
                     (url_col_name))
  cols_without_url = list(df.columns)
  cols_without_url.remove(url_col_name)
  cols_in_order = [url_col_name] + cols_without_url
  df_for_html = df[cols_in_order]
  return df_for_html


def write_html_for_df(df,
                      out_path,
                      base_neuroglancer_url,
                      per_plate_pattern=None,
                      cloud_root_dir=''):
  """Writes a dataframe out as a neuroglancer table.

  Args:
    df: Pandas dataframe.
    out_path: String filepath for the output. Should be a cloud bucket path,
      eg -
        'path/to/folder/my_table.html'
    base_neuroglancer_url: The string url for neuroglancer to load in the top
      iframe.
    per_plate_pattern: Optional string filepath with one '%s'. If provided, the
      df must have a column "plate". One HTML file will be created for each
      plate.
    cloud_root_dir: Optional string filepath to folder containing the plate
      files. Use CLOUD_ROOT_DIR in colab.
  """
  accum = get_html_list(df, base_neuroglancer_url)
  to_file_on_cloud(''.join(accum), out_path)

  if per_plate_pattern:
    plates = list(df.plate.unique())
    for plate in plates:
      plate_df = df.query('plate == "%s"' % plate)
      plate_accum = get_html_list(plate_df, base_neuroglancer_url)
      plate_path = os.path.join(cloud_root_dir, per_plate_pattern % plate)
      to_file_on_cloud(''.join(plate_accum), plate_path)
