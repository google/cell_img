"""Methods for binary creating and writing to an image grid tensorstore."""

from absl import logging
import apache_beam as beam
from cell_img.common import io_lib
from cell_img.image_grid import downsample_lib
from cell_img.image_grid import tensorstore_beam
from cell_img.image_grid import ts_index_lib
from cell_img.image_grid import ts_metadata_lib
from cell_img.image_grid import ts_write_lib
import pandas as pd


_NAMESPACE = 'ts'


def run_write_to_ts_pipeline(tensorstore_path,
                             create_new_tensorstore,
                             allow_expansion_of_tensorstore,
                             image_metadata_path,
                             image_path_col,
                             axes,
                             x_axis_wrap,
                             y_axis_wrap,
                             override_coord_arrays_path=None,
                             pipeline_options=None):
  """Runs the pipeline to write to an image grid tensorstore."""
  _validate_run_write_to_ts_pipeline_args(tensorstore_path,
                                          create_new_tensorstore,
                                          allow_expansion_of_tensorstore,
                                          image_metadata_path, image_path_col,
                                          axes, x_axis_wrap, y_axis_wrap,
                                          override_coord_arrays_path)

  tensorstore_path_s0 = downsample_lib.join_downsample_level_to_path(
      tensorstore_path, 0)
  spec_without_metadata = ts_write_lib.create_spec_from_path(
      tensorstore_path_s0)
  logging.info('Tensorstore spec from flags: %s', str(spec_without_metadata))

  if isinstance(image_metadata_path, list):
    image_metadata_list = []
    for p in image_metadata_path:
      im_df = io_lib.read_csv(p, dtype=str)
      image_metadata_list.append(im_df)
    image_metadata_df = pd.concat(image_metadata_list)
  else:
    image_metadata_df = io_lib.read_csv(image_metadata_path, dtype=str)
  logging.info('Number of rows in image_metadata_df = %d',
               len(image_metadata_df))
  if override_coord_arrays_path:
    coordinate_arrays_override = io_lib.read_json_file(
        override_coord_arrays_path)
  else:
    coordinate_arrays_override = None

  if create_new_tensorstore:
    logging.info('Creating a new tensorstore.')
    axes_wrap = ts_write_lib.make_axes_wrap(axes, x_axis_wrap, y_axis_wrap)
    image_dtype, image_shape = ts_write_lib.read_image_properties(
        image_metadata_df[image_path_col].iloc[0])
    dataset = ts_write_lib.create_tensorstore(spec_without_metadata,
                                              image_metadata_df, axes,
                                              axes_wrap, image_dtype,
                                              image_shape,
                                              coordinate_arrays_override)
    downsampling_factors = downsample_lib.create_downsample_levels(
        dataset.spec())
  else:
    logging.info('Using an existing tensorstore.')
    dataset = ts_write_lib.open_existing_tensorstore(spec_without_metadata)
    dataset = _maybe_expand_tensorstore(image_metadata_df, dataset,
                                        allow_expansion_of_tensorstore,
                                        coordinate_arrays_override)
    downsampling_factors = downsample_lib.read_downsampling_factors(
        tensorstore_path)

  tensorstore_spec = dataset.spec().to_json()
  logging.info('Tensorstore spec = %s', str(tensorstore_spec))

  # Check that we can get coords for all of the images to be written.
  ts_index = ts_index_lib.index_from_spec(tensorstore_spec)
  ts_index.get_coords_dict(image_metadata_df)

  # Write the images to the tensorstore with beam.
  p = beam.Pipeline(options=pipeline_options)
  p_image_metadata = p | beam.Create(
      image_metadata_df.to_dict(orient='records'))
  p_image_metadata |= beam.Reshuffle()  #  Distribute input across workers.
  p_view_slice_and_array = (
      p_image_metadata
      | 'yield_view_slice_and_read_image' >> beam.ParDo(
          _yield_view_slice_and_read_image, ts_index, image_path_col))
  p_view_slice = (
      p_view_slice_and_array
      | tensorstore_beam.WriteTensorStore(tensorstore_spec))
  _ = p_view_slice | downsample_lib.Downsample(spec_without_metadata,
                                               downsampling_factors)
  return p.run()


def _yield_view_slice_and_read_image(image_metadata, ts_index, image_path_col):
  """Yield the image and view slice if possible. Otherwise log helpful error."""
  try:
    (view_slice, image_array) = ts_write_lib.get_view_slice_and_read_image(
        image_metadata, ts_index, image_path_col)
    yield view_slice, image_array
  except Exception as e:  # pylint: disable=broad-except
    logging.info('Exception in get_view_slice_and_read_image %s', str(e))
    beam.metrics.Metrics.counter(
        _NAMESPACE, 'ERROR-get_view_slice_and_read_image-SEE_LOGS').inc()
    image_path = image_metadata[image_path_col]
    logging.info(
        'Error reading image or getting view slice with path %s.'
        'Got error: %s', image_path, str(e))


def _validate_run_write_to_ts_pipeline_args(tensorstore_path,
                                            create_new_tensorstore,
                                            allow_expansion_of_tensorstore,
                                            image_metadata_path, image_path_col,
                                            axes, x_axis_wrap, y_axis_wrap,
                                            override_coord_arrays_path):
  """Validates flag values and their combinations."""
  if not tensorstore_path:
    raise ValueError('Flag tensorstore_path must be set.')
  if not image_metadata_path:
    raise ValueError('Flag image_metadata_path must be set.')
  if not image_path_col:
    raise ValueError('Flag image_path_col must be set.')

  if create_new_tensorstore:
    if not axes:
      raise ValueError(
          'Flag axes must be set when create_new_tensorstore is True')
    ts_write_lib.assert_yx_axes(axes)
    if allow_expansion_of_tensorstore:
      raise ValueError(
          'Flag allow_expansion_of_tensorstore must not be set when '
          'create_new_tensorstore is True')
  else:
    if axes:
      raise ValueError(
          'Flag axes must not be set when create_new_tensorstore is False')
    if x_axis_wrap:
      raise ValueError(
          'Flag x_axis_wrap must not be set when create_new_tensorstore is False'
      )
    if y_axis_wrap:
      raise ValueError(
          'Flag y_axis_wrap must not be set when create_new_tensorstore is False'
      )
    if override_coord_arrays_path and not allow_expansion_of_tensorstore:
      raise ValueError(
          'Flag override_coord_arrays_path must only be set if '
          'create_new_tensorstore is True or both create_new_tensorstore is '
          'False and allow_expansion_of_tensorstore is True.')


def _maybe_expand_tensorstore(image_metadata_df, existing_dataset,
                              allow_expansion_of_tensorstore,
                              coordinate_arrays_override):
  """If needed, expands the existing tensorstore to fit the new images."""
  existing_spec_json = existing_dataset.spec().to_json()
  tensorstore_metadata = existing_spec_json['metadata']
  axis_to_values_to_add = ts_metadata_lib.get_new_coordinate_values(
      tensorstore_metadata, image_metadata_df)
  if not axis_to_values_to_add:
    logging.info('No need to expand Tensorstore.')
    return existing_dataset

  logging.info('Tensorstore requires expansion. axis_to_values_to_add=%s',
               str(axis_to_values_to_add))
  if not allow_expansion_of_tensorstore:
    raise ValueError(
        'Cannot write the new images without expanding the '
        'tensorstore first. Set flag allow_expansion_of_tensorstore '
        'to True. axis_to_values_to_add=%s' % str(axis_to_values_to_add))
  new_tensorstore_metadata = ts_metadata_lib.expand_tensorstore_metadata(
      tensorstore_metadata, axis_to_values_to_add, coordinate_arrays_override)
  new_dataset = ts_write_lib.overwrite_tensorstore_metadata(
      existing_spec_json, new_tensorstore_metadata)
  downsample_lib.update_downsamped_dimensions(new_dataset)
  return new_dataset
